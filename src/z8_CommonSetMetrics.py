"""Score every method for every target on one common evaluation set.

The metrics the study reports are not currently comparable between methods,
because each run was scored on whatever samples its own predictor subset left
valid. A subset containing one partial-coverage predictor costs the families that
require complete rows 17 of 22 test samples, so a Gaussian process scored on 5
segments was being compared against an XGBoost model scored on 22. R^2 is
computed against the mean of the period it was scored on, so those two numbers
answer different questions, and the same mismatch is why one target reports a
higher R^2 than its reference while also reporting negative skill.

This rebuilds the comparison on the intersection of the test segments the methods
actually share, from the `predictions.csv` files the sweep already writes.

**Leakage cannot occur, by construction.** A segment enters the common set only if
it is labelled ``kind == "test"`` in every run being compared, and a run's
`predictions.csv` contains only its own test rows. The intersection can therefore
only ever exclude segments, never admit one a model trained on. Because the
common set is a subset of every run's test set, its first segment is necessarily
at or after the latest split point among them; the script asserts this rather
than assuming it.

What it does not fix: the models were still *trained* on configuration-specific
splits, so this is a fair comparison of the selected models rather than of
identically-trained ones. That caveat belongs in the methods text.

Usage:
    python src/z8_CommonSetMetrics.py
    python src/z8_CommonSetMetrics.py --root data/output/CV19 --verbose
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils import evidence as ev
from utils import run_paths as rp

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = rp.DEFAULT_ROOT
OUTPUT_NAME = "common_set_metrics.csv"

# Reference forecasts are columns inside every run's predictions.csv rather than
# runs of their own.
REFERENCE_COLUMNS = ("Naive", "Seasonal", "Linear")
META_COLUMNS = {"kind", "sample_file", "gp_uncertainty_mode", "metric_semantics",
                "metric_contract_version", "target", "mc_n_replicates"}

ML_FAMILIES = ("GP", "XGB", "Transformer")


@dataclass
class RunRecord:
    """One scored configuration, restricted to the common evaluation set."""

    run: str
    family: str
    feature_tag: str
    subset_label: str
    retained: bool
    has_profiler: bool
    segments: set
    preds: dict          # sample_file -> prediction
    targets: dict        # sample_file -> target
    refs: dict           # reference name -> {sample_file: prediction}


def _family_of(name: str) -> str | None:
    n = name.lower()
    if n.startswith("gp"):
        return "GP"
    if n.startswith("xgb"):
        return "XGB"
    if n.startswith("mlr_avgall"):
        return "MLR-All"
    if n.startswith("mlr_avg12"):
        return "MLR-12"
    if n.startswith("mlr"):
        return "MLR"
    if "transformer" in n and not n.startswith("model_recurrent"):
        return "Transformer"
    return None


def _prediction_column(columns: list[str]) -> str | None:
    """The model's own prediction column sits immediately after ``target``."""
    if "target" in columns:
        after = columns[columns.index("target") + 1:]
        for c in after:
            if c not in REFERENCE_COLUMNS and c not in META_COLUMNS and not c.endswith(("_std", "_var")):
                return c
    for c in columns:
        if c not in REFERENCE_COLUMNS and c not in META_COLUMNS and not c.endswith(("_std", "_var")):
            return c
    return None


def _profiler_tags(sweeps: Path) -> dict[str, bool]:
    traces = sorted(sweeps.glob("feature_search_trace_*.csv"))
    if not traces:
        return {}
    try:
        t = pd.read_csv(traces[0], encoding="utf-8", encoding_errors="replace")
    except Exception:
        return {}
    t = t[t["features"].notna()] if "features" in t.columns else t.iloc[0:0]
    return {tag: bool(g["features"].iloc[0].count("Pfl -"))
            for tag, g in t.groupby("feature_tag")}


def _model_family(model_name: str) -> str | None:
    """Family of a `model` value in feature_sweep_final_metrics.csv."""
    n = str(model_name).strip().lower()
    if n.startswith("mlr_avgall"):
        return "MLR-All"
    if n.startswith("mlr_avg12"):
        return "MLR-12"
    if n.startswith("mlr"):
        return "MLR"
    if "gp" in n:
        return "GP"
    if "xgb" in n:
        return "XGB"
    if "transformer" in n and "recurrent" not in n:
        return "Transformer"
    return None


def _retained_keys(sweeps: Path) -> set[tuple[str, str]]:
    """(family, key) pairs the sweep retained, from the final metrics.

    A sweep leaves a run directory behind for every configuration it evaluated
    during the search, not only the ones it kept: for one target that is 277
    XGBoost directories against 12 retained configurations. Scoring the best of
    all 277 would report a selection the pipeline never made and would never
    have reported, so the comparison is restricted to what the search actually
    retained. ``--include-search-runs`` opts out for diagnostics.

    Both the feature tag and the subset label are accepted as keys, because MLR
    run directories are named by subset label alone and carry no tag.
    """
    metrics = sweeps / "feature_sweep_final_metrics.csv"
    if not metrics.exists():
        return set()
    try:
        d = pd.read_csv(metrics, encoding="utf-8", encoding_errors="replace")
    except Exception:
        return set()
    keys: set[tuple[str, str]] = set()
    for _, r in d.iterrows():
        fam = _model_family(r.get("model", ""))
        if fam is None:
            continue
        for col in ("feature_tag", "subset_label"):
            v = str(r.get(col, "") or "").strip()
            if v and v.lower() != "nan":
                keys.add((fam, v))
    return keys


def _search_holdout_scorings(sweeps: Path) -> int:
    """How many times the feature search scored a candidate on the test split.

    The beam search's objective is test-split R^2: ``_objective_from_metrics`` in
    ``h_RunMCFeatureSelectionSweep.py`` is ``(1 - r2) + lambda_drop * drop_rate``
    with ``r2`` taken from the ``kind == 'test'`` row that ``evaluate_single_config``
    returns. So every row of the search trace is one consultation of the holdout,
    and the candidate pool the reported result was drawn from is the product of all
    of them.

    This is the number the paper has to disclose. Counting only the configurations
    the search *retained* -- about twelve -- describes the last step of the
    selection and not the selection, and understates the exposure by more than an
    order of magnitude.

    Returns 0 when no trace is present, which is the case for smoke-test trees.
    """
    total = 0
    for tr in sorted(sweeps.glob("feature_search_trace_*.csv")):
        try:
            total += len(pd.read_csv(tr, encoding="utf-8", encoding_errors="replace"))
        except Exception:
            continue
    return total


def load_runs(sweeps: Path) -> list[RunRecord]:
    tag_map = _profiler_tags(sweeps)
    retained = _retained_keys(sweeps)
    records: list[RunRecord] = []
    for d in sorted(p for p in sweeps.iterdir() if p.is_dir()):
        family = _family_of(d.name)
        pf = d / "predictions.csv"
        if family is None or not pf.exists():
            continue
        try:
            t = pd.read_csv(pf, encoding="utf-8", encoding_errors="replace")
        except Exception:
            continue
        if "kind" not in t.columns or "sample_file" not in t.columns or "target" not in t.columns:
            continue
        t = t[t["kind"].astype(str) == "test"]
        if t.empty:
            continue
        col = _prediction_column(list(t.columns))
        if col is None:
            continue
        m = re.search(r"(f\d+_[0-9a-f]+)", d.name)
        tag = m.group(1) if m else ""
        sm = re.search(r"_((?:shap_)?[klms]\d+)$", d.name)
        subset = sm.group(1) if sm else ""
        is_retained = ((family, tag) in retained) or (bool(subset) and (family, subset) in retained)
        # MC replicates share a segment; collapse to one value per segment so a
        # segment counts once, matching the independent-sample contract.
        grp = t.groupby("sample_file")
        preds = {k: float(v) for k, v in grp[col].mean().items() if np.isfinite(v)}
        targets = {k: float(v) for k, v in grp["target"].mean().items() if np.isfinite(v)}
        refs = {}
        for r in REFERENCE_COLUMNS:
            if r in t.columns:
                refs[r] = {k: float(v) for k, v in grp[r].mean().items() if np.isfinite(v)}
        segs = set(preds) & set(targets)
        if not segs:
            continue
        records.append(RunRecord(d.name, family, tag, subset, is_retained,
                                 tag_map.get(tag, False), segs, preds, targets, refs))
    return records


def _record_sigma(dataset_dir: Path) -> float:
    """Target standard deviation over the complete record, for NRMSE.

    Normalizing by the standard deviation of the common evaluation set would make
    NRMSE exactly ``sqrt(1 - R^2)`` -- a restatement of a number already
    reported. The record-wide scale instead expresses the error relative to how
    much the target varies overall, which is what makes it comparable between
    targets that differ by orders of magnitude, and it is what Section 2.6 of the
    manuscript describes.
    """
    samples = dataset_dir / "samples"
    files = sorted(samples.glob("segment_*.csv"))
    if not files:
        return float("nan")
    try:
        header = pd.read_csv(files[0], nrows=0, encoding="utf-8",
                             encoding_errors="replace").columns
    except Exception:
        return float("nan")
    target_cols = [c for c in header if c.endswith(("_diff", "_res"))]
    if not target_cols:
        return float("nan")
    col = target_cols[0]
    values: list[float] = []
    for f in files:
        try:
            v = pd.read_csv(f, usecols=[col], encoding="utf-8",
                            encoding_errors="replace")[col].dropna()
        except Exception:
            continue
        values.extend(float(x) for x in v)
    if len(values) < 2:
        return float("nan")
    s = float(np.std(np.asarray(values, dtype=float), ddof=1))
    return s if np.isfinite(s) and s > 0 else float("nan")


def _metrics(y: np.ndarray, p: np.ndarray, sigma: float) -> dict:
    err = p - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": rmse,
        "r2": float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan"),
        "nrmse": rmse / sigma if sigma > 0 else float("nan"),
    }


def segment_index(sample_file: str) -> int:
    """Temporal position of a segment, from its ``segment_NNNN`` filename."""
    m = re.search(r"(\d+)", str(sample_file))
    return int(m.group(1)) if m else -1


def target_label(dataset_dir: Path) -> str:
    return re.sub(r"_diff$|_res$", "", dataset_dir.name.replace("MC_", ""))


def common_evaluation_set(runs: list[RunRecord], label: str):
    """The segments every method can be scored on, and the runs that cover them.

    Returns ``(common, union, eligible, ordered)`` or ``None`` when no shared
    segment exists. Factored out so that any analysis built on the common set --
    the headline metrics here, the temporal re-selection in
    ``z11_TemporalReselect.py`` -- derives it identically. Two copies of this
    would be free to drift apart, and a silent disagreement about which segments
    are being compared is exactly the class of defect this module exists to
    remove.

    **Leakage-safety is asserted, not assumed.** The common set is an
    intersection of test sets, so it can only ever exclude segments; the check
    below confirms it begins at or after every compared run's own test start.
    """
    families = sorted({r.family for r in runs})
    # The intersection across families of what each family can score anywhere.
    # Taking the union within a family first stops one unlucky configuration from
    # shrinking the comparison for everyone.
    per_family_union = {f: set().union(*(r.segments for r in runs if r.family == f))
                        for f in families}
    common = set.intersection(*per_family_union.values())
    union = set().union(*per_family_union.values())
    if not common:
        print(f"[WARN] {label}: families share no test segment; skipping.")
        return None

    # Only runs covering the whole common set can be scored on it, so every
    # reported number describes exactly the same segments.
    eligible = [r for r in runs if common <= r.segments]
    if not eligible:
        print(f"[WARN] {label}: no run covers the common set of {len(common)} segment(s); skipping.")
        return None

    common_start = min(segment_index(s) for s in common)
    latest_split = max(min(segment_index(s) for s in r.segments) for r in eligible)
    if common_start < latest_split:
        raise AssertionError(
            f"{label}: common set starts at segment {common_start} but a compared run's "
            f"test set starts at {latest_split}; the intersection is not leakage-safe.")

    return common, union, eligible, sorted(common, key=segment_index)


def analyse_target(dataset_dir: Path, verbose: bool = False,
                   include_search_runs: bool = False) -> dict | None:
    sweeps = dataset_dir / "forecasts" / "feature_sweeps"
    if not sweeps.is_dir():
        return None
    runs = load_runs(sweeps)
    if not runs:
        print(f"[WARN] {dataset_dir.name}: no scorable runs.")
        return None
    if not include_search_runs:
        kept = [r for r in runs if r.retained]
        if kept:
            runs = kept
        else:
            print(f"[WARN] {dataset_dir.name}: no run matched a retained configuration; "
                  "falling back to every run directory, which includes search trials.")

    label = target_label(dataset_dir)
    families = sorted({r.family for r in runs})

    cs = common_evaluation_set(runs, label)
    if cs is None:
        return None
    common, union, eligible, ordered = cs
    y = np.array([eligible[0].targets[s] for s in ordered], dtype=float)
    # ddof=0: R^2 divides by the population sum of squares of the evaluation set,
    # so this is the scale that makes NRMSE = sqrt(1 - R^2) * sigma_common /
    # sigma_record hold exactly. With ddof=1 the relation is off by
    # sqrt((n-1)/n), which is 11% at n = 5 -- large enough to matter for the
    # targets where it matters most.
    sigma_common = float(np.std(y, ddof=0)) if y.size > 1 else float("nan")
    # NRMSE uses the record-wide scale; R^2, RMSE and skill use the common set.
    sigma = _record_sigma(dataset_dir)
    if not np.isfinite(sigma):
        print(f"[WARN] {label}: could not read a record-wide sigma; NRMSE falls back to "
              "the common-set standard deviation, where it restates R^2.")
        sigma = sigma_common

    row: dict = {
        "dataset": dataset_dir.name,
        "target": label,
        "n_common": len(common),
        "n_union": len(union),
        "sigma_common": sigma_common,
        "sigma_record": sigma,
        "n_runs_total": len(runs),
        "n_runs_eligible": len(eligible),
        # Selection exposure: the holdout consultations behind the candidate pool.
        # Shared across families, because one XGBoost surrogate search decides which
        # subsets exist for all of them.
        "n_search_holdout_scorings": _search_holdout_scorings(sweeps),
    }

    # Per family: the best configuration on the common set, and how many
    # configurations that maximum was taken over.
    best: dict[str, dict] = {}
    for fam in families:
        cand = [r for r in eligible if r.family == fam]
        if not cand:
            continue
        scored = []
        for r in cand:
            p = np.array([r.preds[s] for s in ordered], dtype=float)
            scored.append((r, _metrics(y, p, sigma), p))
        scored.sort(key=lambda t: (-(t[1]["r2"] if np.isfinite(t[1]["r2"]) else -np.inf)))
        r, m, p = scored[0]
        r2s = np.array([s[1]["r2"] for s in scored], dtype=float)
        r2s = r2s[np.isfinite(r2s)]
        best[fam] = {"record": r, "metrics": m, "preds": p}
        key = fam.lower().replace("-", "")
        row[f"{key}_r2"] = m["r2"]
        row[f"{key}_rmse"] = m["rmse"]
        row[f"{key}_nrmse"] = m["nrmse"]
        row[f"{key}_run"] = r.run
        row[f"{key}_has_profiler"] = r.has_profiler
        row[f"{key}_n_candidates"] = len(scored)
        # A median within one target across its own configurations summarizes the
        # selection headroom; it is not a cross-target aggregate.
        row[f"{key}_r2_min"] = float(r2s.min()) if r2s.size else float("nan")
        row[f"{key}_r2_median"] = float(np.median(r2s)) if r2s.size else float("nan")

    # References are columns rather than runs. Take them from the eligible run with
    # the widest coverage, and check the others agree.
    donor = max(eligible, key=lambda r: (len(r.segments), r.run))
    for ref in REFERENCE_COLUMNS:
        series = [r for r in eligible if ref in r.refs and common <= set(r.refs[ref])]
        if not series:
            continue
        src = donor if (ref in donor.refs and common <= set(donor.refs[ref])) else series[0]
        p = np.array([src.refs[ref][s] for s in ordered], dtype=float)
        disagree = sum(
            1 for r in series
            if not np.allclose(np.array([r.refs[ref][s] for s in ordered], dtype=float), p,
                               rtol=1e-6, atol=1e-9))
        if disagree:
            print(f"[WARN] {label}: '{ref}' differs across {disagree} of {len(series)} runs; "
                  f"using {src.run}.")
        m = _metrics(y, p, sigma)
        best[ref] = {"record": src, "metrics": m, "preds": p}
        key = ref.lower()
        row[f"{key}_r2"] = m["r2"]
        row[f"{key}_rmse"] = m["rmse"]
        row[f"{key}_nrmse"] = m["nrmse"]

    # Best of each side, on identical segments.
    ml = {f: v for f, v in best.items() if f in ML_FAMILIES}
    ref = {f: v for f, v in best.items() if f not in ML_FAMILIES}
    if ml:
        f = max(ml, key=lambda k: ml[k]["metrics"]["r2"] if np.isfinite(ml[k]["metrics"]["r2"]) else -np.inf)
        row["best_ml_family"], row["best_ml_r2"] = f, ml[f]["metrics"]["r2"]
    if ref:
        f = max(ref, key=lambda k: ref[k]["metrics"]["r2"] if np.isfinite(ref[k]["metrics"]["r2"]) else -np.inf)
        row["best_ref_family"], row["best_ref_r2"] = f, ref[f]["metrics"]["r2"]
    if ml and ref:
        allm = {**ml, **ref}
        f = max(allm, key=lambda k: allm[k]["metrics"]["r2"] if np.isfinite(allm[k]["metrics"]["r2"]) else -np.inf)
        row["best_family"], row["best_r2"] = f, allm[f]["metrics"]["r2"]
        row["best_is_ml"] = f in ML_FAMILIES

        mlf, reff = row["best_ml_family"], row["best_ref_family"]
        rm, rr = ml[mlf]["metrics"]["rmse"], ref[reff]["metrics"]["rmse"]
        row["skill_vs_best_ref"] = float(1.0 - rm / rr) if rr > 0 else float("nan")

        a = ev.assess(np.asarray((ml[mlf]["preds"] - y) ** 2, dtype=float),
                      np.asarray((ref[reff]["preds"] - y) ** 2, dtype=float),
                      ordered)
        for k in ("skill", "skill_ci05", "skill_ci95", "n_groups", "bootstrap_degenerate",
                  "sign_wins", "sign_losses", "sign_n_pairs", "sign_win_rate", "sign_p",
                  "sign_direction", "min_attainable_p", "power_attainable", "verdict"):
            row[f"aligned_{k}"] = a.get(k)

        # On one evaluation set these cannot disagree; if they do, something in
        # the construction is wrong and the target should not be reported.
        if np.isfinite(row["skill_vs_best_ref"]):
            if (row["best_ml_r2"] > row["best_ref_r2"]) != (row["skill_vs_best_ref"] > 0):
                print(f"[ERROR] {label}: R^2 ordering and skill sign still disagree on the "
                      "common set; do not report this target until resolved.")
                row["consistency_error"] = True

    if verbose:
        print(f"[INFO] {label}: n_common={len(common)} of {len(union)}; "
              f"{len(eligible)} of {len(runs)} runs eligible; families={', '.join(families)}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None,
                    help="Defaults to <root>/summaries/%s, so the analysis is written "
                         "beside the tree it was read from." % OUTPUT_NAME)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--include-search-runs", action="store_true",
                    help="Also score configurations the search evaluated but did not retain. "
                         "Diagnostic only: it reports a best-of-N the pipeline never made.")
    args = ap.parse_args()

    root = rp.resolve_root(args.root)
    output = rp.resolve_output(args.output, root, OUTPUT_NAME)
    if not root.is_dir():
        raise SystemExit(f"Output root not found: {root}")

    rows = [r for r in (analyse_target(d, args.verbose, args.include_search_runs)
                        for d in sorted(root.glob("MC_*")) if d.is_dir()) if r]
    if not rows:
        raise SystemExit(f"No targets could be scored under {root}")

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"\n[INFO] Wrote {output}")

    # Counts only. Targets have unrelated dynamics, so no statistic is averaged
    # across them.
    print(f"[INFO] {len(df)} target(s) scored on a common evaluation set.")
    if "best_is_ml" in df.columns:
        n_ml = int(df["best_is_ml"].fillna(False).sum())
        print(f"[INFO] best method is a learned model for {n_ml} of {len(df)} targets, "
              f"a reference for {len(df) - n_ml}.")
    if "consistency_error" in df.columns:
        bad = df[df["consistency_error"].fillna(False)]
        if len(bad):
            print(f"[ERROR] {len(bad)} target(s) still inconsistent: "
                  f"{', '.join(bad['target'])}")
    shrink = df[df["n_common"] < df["n_union"]]
    if len(shrink):
        print(f"[INFO] common set is smaller than the union for {len(shrink)} target(s):")
        for _, r in shrink.iterrows():
            print(f"         {r['target'][:34]:<34} {int(r['n_common'])} of {int(r['n_union'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

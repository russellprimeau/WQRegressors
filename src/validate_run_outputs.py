"""Validate that a sweep output tree contains everything the analysis needs.

The sweep is expensive enough that it can realistically be run once, so the cost
of discovering a missing or unattributable output afterwards is another full run.
This script walks an output root and fails loudly on anything the downstream
figures, tables and statistics depend on, so the gap is found before the run
rather than after it.

The rule it enforces throughout: **every dropped sample must be attributable.**
A run that scored 5 of 22 test samples is not a problem in itself; a run that
scored 5 of 22 without recording which predictor cost it the other 17 is, because
that is how a partial-coverage predictor removed three quarters of a target's
evaluation set unnoticed.

Checks, per target:

  samples        the sample directory exists and is non-empty
  trace          exactly one search trace, its row_count matching final metrics
  subsets        the selected-subsets file exists
  metrics        feature_sweep_final_metrics.csv exists and is non-empty
  sigma          exactly one std_target value for the target, positive and finite
  nrmse          populated for every row that has an rmse
  families       every expected model family produced at least one scored row

and per run directory:

  predictions    predictions.csv exists, with the kind and sample_file columns
                 the common-evaluation-set analysis joins on
  splits         train_files.txt and test_files.txt exist and are non-empty
  summary        evaluation_summary.csv exists
  train_size     n_train_samples is populated, and agrees with train_files.txt
  attribution    a run reporting dropped test samples also names the predictors
  support        no prediction lies outside the target's normalized support

Usage:
    python src/validate_run_outputs.py
    python src/validate_run_outputs.py --root data/output/CV19 --verbose
    python src/validate_run_outputs.py --root data/output/SMOKE_GP --families gp
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = Path("data/output/CV19")

# Families the analysis expects to be able to compare. Naive/seasonal/linear are
# columns inside other families' predictions.csv rather than run directories of
# their own, so they are checked separately.
EXPECTED_FAMILIES = ("gp", "xgb", "transformer", "mlr")
REFERENCE_COLUMNS = ("Naive", "Seasonal", "Linear")

# Predictions are of min-max normalized targets, so this is the known support.
TARGET_SUPPORT = (0.0, 1.0)
SUPPORT_TOLERANCE = 1e-6


@dataclass
class Findings:
    """Collected problems, separated by whether they invalidate the analysis."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, target: str, message: str) -> None:
        self.errors.append(f"{target}: {message}")

    def warn(self, target: str, message: str) -> None:
        self.warnings.append(f"{target}: {message}")

    @property
    def ok(self) -> bool:
        return not self.errors


def _family_of(run_dir_name: str) -> str | None:
    name = run_dir_name.lower()
    if name.startswith("gp"):
        return "gp"
    if name.startswith("xgb"):
        return "xgb"
    if "transformer" in name:
        return "transformer"
    if name.startswith("mlr"):
        return "mlr"
    if name.startswith("lstm") or name.startswith("model_lstm"):
        return "lstm"
    return None


def _read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs)
    except Exception:
        return None


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return 0


def check_target(dataset_dir: Path, families: tuple[str, ...], found: Findings,
                 verbose: bool = False) -> None:
    label = re.sub(r"_diff$|_res$", "", dataset_dir.name.replace("MC_", ""))
    sweeps = dataset_dir / "forecasts" / "feature_sweeps"

    samples = dataset_dir / "samples"
    n_samples = len(list(samples.glob("segment_*.csv"))) if samples.is_dir() else 0
    if n_samples == 0:
        found.error(label, "no sample files; nothing downstream can be reproduced")

    if not (dataset_dir / "normalization.json").exists():
        found.warn(label, "no normalization.json; target values cannot be returned "
                          "to physical units for checking")

    metrics_path = sweeps / "feature_sweep_final_metrics.csv"
    if not metrics_path.exists():
        found.error(label, "no feature_sweep_final_metrics.csv; the target is absent "
                           "from every figure and table")
        return
    metrics = _read_csv(metrics_path)
    if metrics is None or metrics.empty:
        found.error(label, "feature_sweep_final_metrics.csv is empty or unreadable")
        return

    traces = sorted(sweeps.glob("feature_search_trace_*.csv"))
    if not traces:
        found.error(label, "no feature_search_trace_*.csv; predictor membership cannot "
                           "be recovered")
    elif len(traces) > 1:
        found.error(label, f"{len(traces)} search traces present ({', '.join(t.name for t in traces)}); "
                           "results from different window lengths may be mixed")
    else:
        m = re.search(r"_r(\d+)\.csv$", traces[0].name)
        if m and "row_count" in metrics.columns:
            row_counts = sorted(pd.to_numeric(metrics["row_count"], errors="coerce")
                                .dropna().unique().astype(int).tolist())
            if row_counts and int(m.group(1)) not in row_counts:
                found.error(label, f"search trace is r{m.group(1)} but final metrics record "
                                   f"row_count {row_counts}; the trace does not describe these runs")

    if not sorted(sweeps.glob("feature_selected_subsets_*.csv")):
        found.warn(label, "no feature_selected_subsets_*.csv")

    # One sigma per target: NRMSE is only comparable between methods if its
    # denominator describes the target rather than the configuration scored.
    if "std_target" in metrics.columns:
        sigmas = pd.to_numeric(metrics["std_target"], errors="coerce").dropna()
        n_distinct = int(sigmas.round(12).nunique())
        if sigmas.empty:
            found.error(label, "std_target is empty; NRMSE cannot be computed")
        elif n_distinct > 1:
            found.error(label, f"std_target holds {n_distinct} distinct values "
                               f"({sigmas.min():.4g}..{sigmas.max():.4g}); NRMSE is not "
                               "comparable between methods")
        elif not (sigmas.iloc[0] > 0):
            found.error(label, f"std_target is {sigmas.iloc[0]!r}; NRMSE is undefined")
    else:
        found.error(label, "no std_target column; NRMSE cannot be computed")

    if {"rmse", "nrmse"}.issubset(metrics.columns):
        scored = metrics[pd.to_numeric(metrics["rmse"], errors="coerce").notna()]
        missing = int(pd.to_numeric(scored["nrmse"], errors="coerce").isna().sum())
        if missing:
            found.error(label, f"{missing} of {len(scored)} scored rows have no nrmse; "
                               "those rows are silently absent from the nRMSE figure")

    # Model families present among scored rows.
    if "model" in metrics.columns:
        scored_models = (metrics.loc[pd.to_numeric(metrics.get("r2"), errors="coerce").notna(),
                                     "model"].astype(str).str.lower().unique().tolist())
        for fam in families:
            if not any(_family_of(m) == fam for m in scored_models):
                found.warn(label, f"no scored rows for the {fam} family; it will be a gap "
                                  "in every per-family comparison")

    # Per-run checks.
    run_dirs = [d for d in sorted(sweeps.iterdir()) if d.is_dir() and _family_of(d.name)]
    if not run_dirs:
        found.error(label, "no model run directories under feature_sweeps")
        return

    ref_seen = False
    for run in run_dirs:
        rl = f"{label}/{run.name}"
        preds_path = run / "predictions.csv"
        if not preds_path.exists():
            found.error(rl, "no predictions.csv; excluded from the common evaluation set")
            continue
        preds = _read_csv(preds_path)
        if preds is None or preds.empty:
            found.error(rl, "predictions.csv is empty or unreadable")
            continue
        for col in ("kind", "sample_file"):
            if col not in preds.columns:
                found.error(rl, f"predictions.csv has no '{col}' column; the common "
                                "evaluation set cannot be constructed")
        if any(c in preds.columns for c in REFERENCE_COLUMNS):
            ref_seen = True

        # Predictions must lie in the target's normalized support. One
        # extrapolation is enough to dominate a squared-error metric.
        numeric = preds.select_dtypes(include=[np.number])
        pred_cols = [c for c in numeric.columns
                     if c not in {"metric_contract_version", "mc_n_replicates"}
                     and not c.startswith("n_")
                     and not c.endswith(("_std", "_var"))
                     and c != "target"]
        for col in pred_cols:
            v = pd.to_numeric(preds[col], errors="coerce").dropna().to_numpy()
            if v.size == 0:
                continue
            oob = int(np.sum((v < TARGET_SUPPORT[0] - SUPPORT_TOLERANCE)
                             | (v > TARGET_SUPPORT[1] + SUPPORT_TOLERANCE)))
            if oob:
                found.error(rl, f"{oob} '{col}' prediction(s) outside the target support "
                                f"[{TARGET_SUPPORT[0]:g}, {TARGET_SUPPORT[1]:g}], worst "
                                f"|{np.max(np.abs(v)):.3g}|; one such point can decide "
                                "which model is reported as best")

        for split in ("train_files.txt", "test_files.txt"):
            sp = run / split
            if not sp.exists():
                found.error(rl, f"no {split}; the split cannot be audited for leakage")
            elif _count_lines(sp) == 0:
                found.error(rl, f"{split} is empty")

        summary_path = run / "evaluation_summary.csv"
        if not summary_path.exists():
            found.error(rl, "no evaluation_summary.csv")
            continue
        summary = _read_csv(summary_path)
        if summary is None or summary.empty:
            found.error(rl, "evaluation_summary.csv is empty or unreadable")
            continue

        if "n_train_samples" not in summary.columns:
            found.error(rl, "evaluation_summary.csv has no n_train_samples column")
        else:
            n_train = pd.to_numeric(summary["n_train_samples"], errors="coerce").dropna()
            if n_train.empty:
                found.error(rl, "n_train_samples is null; the training-set cost of this "
                                "predictor choice is unrecorded")
            else:
                expected = _count_lines(run / "train_files.txt")
                got = int(n_train.max())
                if expected and got and abs(got - expected) > max(1, 0.02 * expected):
                    found.warn(rl, f"n_train_samples={got} disagrees with "
                                   f"train_files.txt={expected}")

        # Any dropped sample must name the predictor responsible.
        for prefix in ("train", "test"):
            n_dropped_col = f"{prefix}_n_dropped"
            culprit_col = f"{prefix}_drop_predictors"
            if n_dropped_col not in summary.columns:
                found.warn(rl, f"no {n_dropped_col} column; sample loss is unrecorded")
                continue
            dropped = pd.to_numeric(summary[n_dropped_col], errors="coerce").fillna(0)
            if float(dropped.max()) <= 0:
                continue
            culprits = (summary.get(culprit_col, pd.Series(dtype=object))
                        .astype(str).str.strip().replace({"nan": ""}))
            if not culprits.any():
                found.error(rl, f"{int(dropped.max())} {prefix} sample(s) dropped with no "
                                f"predictor named in {culprit_col}; the loss is "
                                "unattributable")

    if not ref_seen:
        found.error(label, "no run recorded the reference forecasts (Naive/Seasonal/Linear); "
                           "skill cannot be computed")

    if verbose:
        print(f"[INFO] {label}: {n_samples} samples, {len(run_dirs)} run dirs, "
              f"{len(metrics)} metric rows")


def _finding_kind(line: str) -> str:
    """Collapse a finding to its kind, so identical problems group together."""
    _, _, message = line.partition(": ")
    message = message or line
    message = re.sub(r"\d+", "N", message)
    message = re.sub(r"'[^']*'", "'X'", message)
    return message.strip()


def _report_group(title: str, lines: list[str], max_kinds: int = 25) -> None:
    """Print findings grouped by kind.

    A pre-run gate is only useful if it can be read, and one systemic gap can
    produce thousands of identical findings -- the missing drop-attribution
    columns alone account for over eight thousand. Grouping keeps a real but
    rare problem from being buried under a common one.
    """
    if not lines:
        return
    groups: dict[str, list[str]] = {}
    for line in lines:
        groups.setdefault(_finding_kind(line), []).append(line)
    print(f"\n{title}: {len(lines)} finding(s) in {len(groups)} kind(s)")
    for kind, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:max_kinds]:
        targets = sorted({m.split(":")[0].split("/")[0] for m in members})
        shown = ", ".join(targets[:6]) + (f" (+{len(targets) - 6} more)" if len(targets) > 6 else "")
        print(f"  [{len(members):>5}x] {kind}")
        print(f"          targets: {shown}")
        print(f"          example: {members[0]}")
    if len(groups) > max_kinds:
        print(f"  ... and {len(groups) - max_kinds} further kinds")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None,
                    help=f"Output root to validate (default: {DEFAULT_ROOT} under the repo).")
    ap.add_argument("--families", nargs="*", default=list(EXPECTED_FAMILIES),
                    help="Model families required to be present.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve() if args.root else (REPO_ROOT / DEFAULT_ROOT)
    if not root.is_dir():
        raise SystemExit(f"Output root not found: {root}")

    datasets = [d for d in sorted(root.glob("MC_*")) if d.is_dir()]
    if not datasets:
        raise SystemExit(f"No MC_* target directories under {root}")

    found = Findings()
    for ds in datasets:
        check_target(ds, tuple(args.families), found, verbose=args.verbose)

    print(f"\nValidated {len(datasets)} target(s) under {root}")
    print(f"  errors:   {len(found.errors)}")
    print(f"  warnings: {len(found.warnings)}")
    _report_group("ERRORS (the analysis cannot be trusted until these are resolved)",
                  found.errors)
    _report_group("WARNINGS", found.warnings)

    print("\nRESULT:", "PASS" if found.ok else "FAIL")
    return 0 if found.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

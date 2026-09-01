"""Check a finished run tree against the invariants the pipeline is supposed to hold.

Why this exists
---------------
Every defect this repository has hit in the last stretch was found by hand, once, and
then had to be re-found by hand on the next tree. Several were reported wrongly the first
time because the report was an inference rather than a measurement. This script encodes
each of those checks so that the answer to "is this tree sound?" is a command and its
output, not a judgement.

It reuses the pipeline's own functions -- ``h_RunMCFeatureSelectionSweep._pooled_r2``,
``z8_CommonSetMetrics.load_runs`` -- rather than reimplementing them. A checker with its
own copy of the metric is a checker that can agree with itself while disagreeing with the
pipeline, which is the failure it is meant to catch.

What each check is for, and the incident behind it
--------------------------------------------------
1. run integrity        A run directory with no ``predictions.csv`` is a fit that failed
                        silently. 55 GP configs did this in CV22 before the
                        ``DenseAdditiveKernel`` fix and nothing flagged it.
2. metrics vs on-disk   ``feature_sweep_final_metrics.csv`` must agree with the
   predictions           predictions it claims to describe. In CV22 it does not for the 80
                        runs ``z17`` overwrote after the fact: the table holds the
                        single-seed score, the file holds the ensemble.
3. candidate pool       ``z8`` scores every directory with a family prefix. Seed
                        replicates written into ``feature_sweeps/`` therefore competed as
                        independent candidates, so ``--seeds N`` let the maximum of N
                        draws win -- the opposite of what the flag is for.
4. horizon anchor       Horizon 0 is a refit of the reported configuration, so the curve's
                        leftmost point must equal the results table. It did not, because
                        one side averaged R^2 and the other scored the mean prediction.
5. search discrimination A surrogate that predicts a constant ranks every feature subset
                        identically. CV22's Chromium search evaluated 240 candidates that
                        all returned r2 = -0.0013608598215928; the "selected" subset was a
                        tie-break. The code already knew -- it set the degenerate flag 240
                        times -- and said nothing.
6. ensemble provenance  A row claiming an N-seed ensemble must have the single-seed
                        predictions preserved beside it, and vice versa.
7. output containment   ``--root`` and ``--output`` defaulted independently, so a run
                        against one tree wrote its summaries into another's. CV19's
                        summaries were overwritten with CV20's analysis this way.

Exit status is non-zero if any check FAILs, so this can gate a re-run.

Usage:
    python src/v4_CheckPipelineInvariants.py --root data/output/CV22_profilerless
    python src/v4_CheckPipelineInvariants.py --root data/output/CV23_profiler --verbose
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h_RunMCFeatureSelectionSweep as h  # noqa: E402
import z8_CommonSetMetrics as z8  # noqa: E402
from utils import run_paths as rp  # noqa: E402

SEED_RE = re.compile(r"_seed\d+$")
# Selection-stability probes hold only `train_files.txt` and `test_files.txt`: they
# record where a split fell, and were never meant to fit anything. Counting them as
# failed runs would bury the eight genuine GP failures they outnumber.
STAB_RE = re.compile(r"_stab\d+$")
NON_RUN_DIRS = {"configs", "seed_refit", "seed_reps", "tmp"}
R2_TOL = 1e-6
ANCHOR_TOL = 1e-4


class Result:
    """One check's outcome: a status, a headline, and the specifics behind it."""

    def __init__(self, name: str):
        self.name = name
        self.fails: list[str] = []
        self.warns: list[str] = []
        self.notes: list[str] = []

    @property
    def status(self) -> str:
        return "FAIL" if self.fails else ("WARN" if self.warns else "PASS")

    def report(self, verbose: bool) -> None:
        print("[%-4s] %s" % (self.status, self.name))
        for line in self.notes:
            print("         %s" % line)
        shown = self.fails if not verbose else self.fails
        for line in shown[: (None if verbose else 12)]:
            print("   FAIL  %s" % line)
        if not verbose and len(self.fails) > 12:
            print("   FAIL  ... and %d more (use --verbose)" % (len(self.fails) - 12))
        for line in self.warns[: (None if verbose else 12)]:
            print("   WARN  %s" % line)
        if not verbose and len(self.warns) > 12:
            print("   WARN  ... and %d more (use --verbose)" % (len(self.warns) - 12))


def _sweeps(root: Path):
    """(dataset dir, feature_sweeps dir) for every dataset that has one."""
    for ds in sorted(root.glob("MC_*")):
        sw = ds / "forecasts" / "feature_sweeps"
        if sw.is_dir():
            yield ds, sw


def _run_dirs(sweeps: Path):
    """Directories under feature_sweeps that are meant to be scoreable runs."""
    for p in sorted(sweeps.iterdir()):
        if not p.is_dir() or p.name in NON_RUN_DIRS:
            continue
        if z8._family_of(p.name) is None or STAB_RE.search(p.name):
            continue
        yield p


def _match_row_to_dir(sweeps: Path, row: pd.Series) -> "Path | None":
    """The single run directory a metrics row describes, or None if ambiguous.

    Matching on the variant prefix as well as the feature tag and subset label: the tag
    and label alone are shared by every family that scored the same subset.
    """
    tag = str(row.get("feature_tag", ""))
    sub = str(row.get("subset_label", ""))
    var = str(row.get("variant", ""))
    if not tag or not sub or sub == "nan":
        return None
    pref = "" if var in ("", "nan") else var
    hits = [p for p in sweeps.iterdir()
            if p.is_dir() and tag in p.name and p.name.endswith("_" + sub)
            and (not pref or p.name.startswith(pref))]
    return hits[0] if len(hits) == 1 else None


def check_run_integrity(root: Path) -> Result:
    r = Result("run integrity: every run directory has predictions")
    total = missing = 0
    for ds, sw in _sweeps(root):
        for p in _run_dirs(sw):
            total += 1
            if not (p / "predictions.csv").exists():
                missing += 1
                r.fails.append("%s/%s has no predictions.csv" % (ds.name, p.name))
    r.notes.append("%d run directories, %d without predictions" % (total, missing))
    return r


def check_metrics_match_predictions(root: Path) -> Result:
    r = Result("metrics table agrees with the predictions it describes")
    checked = ok = 0
    for ds, sw in _sweeps(root):
        fm = sw / "feature_sweep_final_metrics.csv"
        if not fm.is_file():
            continue
        d = pd.read_csv(fm, encoding="utf-8", encoding_errors="replace")
        if "r2" not in d.columns:
            continue
        for _, row in d[d["r2"].notna()].iterrows():
            p = _match_row_to_dir(sw, row)
            if p is None:
                continue
            y, pv = h._pooled_predictions(p)
            if y.size < 2:
                continue
            checked += 1
            got = h._pooled_r2(y, pv)
            if abs(got - float(row["r2"])) <= R2_TOL:
                ok += 1
                continue
            # z17 rewrote predictions after the row was written. That is a real
            # inconsistency in the tree, but a known and repairable one, so it is
            # separated from an unexplained disagreement.
            posthoc = (p / "predictions_seed0.csv").exists()
            n_seeds = pd.to_numeric(row.get("n_seeds_ensembled", np.nan), errors="coerce")
            line = ("%s/%s: table r2=%+.6f, predictions give %+.6f"
                    % (ds.name, p.name, float(row["r2"]), got))
            if posthoc and not (np.isfinite(n_seeds) and n_seeds > 1):
                r.warns.append(line + "  [post-hoc ensemble; table not recomputed]")
            else:
                r.fails.append(line)
    r.notes.append("%d of %d matched rows agree to %g" % (ok, checked, R2_TOL))
    if checked == 0:
        r.warns.append("no rows could be matched to a unique run directory")
    return r


def check_candidate_pool(root: Path) -> Result:
    r = Result("candidate pool excludes seed replicates")
    total_reps = leaked = 0
    for ds, sw in _sweeps(root):
        reps = [p.name for p in _run_dirs(sw)
                if SEED_RE.search(p.name) and (p / "predictions.csv").exists()]
        total_reps += len(reps)
        if not reps:
            continue
        loaded = {rec.run for rec in z8.load_runs(sw)}
        bad = sorted(set(reps) & loaded)
        leaked += len(bad)
        for n in bad:
            r.fails.append("%s/%s is a seed replicate but z8 scores it as a candidate"
                           % (ds.name, n))
    r.notes.append("%d seed replicate directories, %d reaching the candidate pool"
                   % (total_reps, leaked))
    return r


def check_horizon_anchor(root: Path) -> Result:
    r = Result("horizon 0 equals the reported result")
    met_path = root / "summaries" / "common_set_metrics.csv"
    if not met_path.is_file():
        r.notes.append("no common_set_metrics.csv; skipped")
        return r
    met = pd.read_csv(met_path, encoding="utf-8", encoding_errors="replace")
    met = met.set_index("dataset")
    n = 0
    for ds in sorted(root.glob("MC_*")):
        sweep = ds / "horizons" / "lookahead_sweeps"
        if not sweep.is_dir():
            continue
        ens = sweep / "lookahead_ensemble.csv"
        if not ens.is_file():
            # Without it z16 falls back to the mean of the replicates' R^2, which is a
            # different quantity from the table and cannot match it for a stochastic
            # winner. See z18_HorizonEnsembles.
            r.fails.append("%s has horizon results but no lookahead_ensemble.csv "
                           "(run z18_HorizonEnsembles.py)" % ds.name)
            continue
        t = pd.read_csv(ens, encoding="utf-8", encoding_errors="replace")
        row = t[t["horizon"] == t["horizon"].min()]
        if row.empty or ds.name not in met.index:
            continue
        n += 1
        delta = float(row.iloc[0]["r2"]) - float(met.loc[ds.name, "best_r2"])
        if abs(delta) > ANCHOR_TOL:
            r.warns.append("%s: horizon 0 differs from the table by %+.4f"
                           % (ds.name, delta))
    r.notes.append("%d target(s) anchored, %d differing by more than %g"
                   % (n, len(r.warns), ANCHOR_TOL))
    return r


def check_search_discrimination(root: Path) -> Result:
    r = Result("the feature search could separate its candidates")
    for ds, sw in _sweeps(root):
        traces = sorted(sw.glob("feature_search_trace_*.csv"))
        if not traces:
            continue
        d = pd.read_csv(traces[0], encoding="utf-8", encoding_errors="replace")
        obj = pd.to_numeric(d.get("objective"), errors="coerce")
        if obj.notna().sum() < 3:
            continue
        deg = (d.get("degenerate", pd.Series(dtype=object))
               .astype(str).str.lower().isin(["true", "1"]))
        n_distinct = int(obj.round(10).nunique())
        if bool(deg.all()) or n_distinct == 1:
            r.fails.append(
                "%s: all %d candidates score identically (objective=%.10f%s); the "
                "selected subset is a tie-break, not a measurement"
                % (ds.name, len(obj), float(obj.iloc[0]),
                   ", all flagged degenerate" if bool(deg.all()) else ""))
            continue
        se = pd.to_numeric(d.get("objective_se"), errors="coerce")
        i = obj.idxmin()
        s = float(se[i]) if se is not None and np.isfinite(se.get(i, np.nan)) else np.nan
        spread = float(obj.max() - obj.min())
        if np.isfinite(s) and s > 0 and spread / s < 1.0:
            r.warns.append("%s: objective spread %.3f is smaller than its standard error "
                           "%.3f; the ranking is not separable" % (ds.name, spread, s))
        if int(deg.sum()):
            r.notes.append("%s: %d of %d candidates degenerate"
                           % (ds.name, int(deg.sum()), len(deg)))
    return r


def check_ensemble_provenance(root: Path) -> Result:
    r = Result("ensembled rows keep their single-seed predictions")
    for ds, sw in _sweeps(root):
        fm = sw / "feature_sweep_final_metrics.csv"
        if not fm.is_file():
            continue
        d = pd.read_csv(fm, encoding="utf-8", encoding_errors="replace")
        if "n_seeds_ensembled" not in d.columns:
            continue
        for _, row in d.iterrows():
            n = pd.to_numeric(row.get("n_seeds_ensembled"), errors="coerce")
            if not (np.isfinite(n) and n > 1):
                continue
            p = _match_row_to_dir(sw, row)
            if p is None:
                continue
            if not (p / "predictions_seed0.csv").exists():
                r.fails.append("%s/%s claims a %d-seed ensemble but has no "
                               "predictions_seed0.csv" % (ds.name, p.name, int(n)))
    return r


def check_output_containment(root: Path) -> Result:
    r = Result("summaries describe this root and no other")
    met = root / "summaries" / "common_set_metrics.csv"
    if not met.is_file():
        r.notes.append("no common_set_metrics.csv; skipped")
        return r
    d = pd.read_csv(met, encoding="utf-8", encoding_errors="replace")
    missing = [n for n in d.get("dataset", pd.Series(dtype=str)).astype(str)
               if not (root / n).is_dir()]
    for n in missing:
        r.fails.append("summaries name dataset %r, which does not exist under %s "
                       "(summaries written from a different root)" % (n, root.name))
    r.notes.append("%d dataset row(s), %d not present here" % (len(d), len(missing)))
    return r


CHECKS = (
    check_run_integrity,
    check_metrics_match_predictions,
    check_candidate_pool,
    check_horizon_anchor,
    check_search_discrimination,
    check_ensemble_provenance,
    check_output_containment,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true",
                    help="List every failure and warning rather than the first 12.")
    args = ap.parse_args()

    root = rp.resolve_root(args.root)
    if not root.is_dir():
        raise SystemExit("Not a directory: %s" % root)
    print("Checking %s\n" % root)

    results = []
    for fn in CHECKS:
        try:
            res = fn(root)
        except Exception as exc:                     # a check that cannot run is a
            res = Result(fn.__name__)                # failure of the check, not a pass
            res.fails.append("check raised %s: %s" % (type(exc).__name__, exc))
        results.append(res)
        res.report(args.verbose)
        print()

    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_warn = sum(1 for r in results if r.status == "WARN")
    print("-" * 72)
    print("%d check(s): %d passed, %d with warnings, %d failed"
          % (len(results), len(results) - n_fail - n_warn, n_warn, n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

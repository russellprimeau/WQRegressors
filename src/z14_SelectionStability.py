"""Report which predictors a feature search actually established, per target.

A sweep reports one winning subset per target. That subset is the argmin of an
objective estimated on 12-47 independent samples, and the subsets ranked just behind
it routinely differ from it by half their membership while scoring within 0.01. Under
those conditions "the selected feature set" is a stronger claim than the search can
support, and a retention count built from it inherits the same weakness.

This reads the search traces already on disk and reports, per target, how often each
predictor appears among the subsets that the objective could not separate from the
best one. Nothing is re-run and no model is fitted, so it can be applied to a finished
sweep.

Usage:
    python src/z14_SelectionStability.py --root data/output/CV20_profilerless
    python src/z14_SelectionStability.py --root data/output/CV21 --tolerance 0.01
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from utils.selection_stability import selection_stability_from_trace

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "data" / "output" / "CV20_profilerless"
OUTPUT_NAME = "feature_selection_stability.csv"


def _target_label(dataset_name: str) -> str:
    return re.sub(r"^MC_", "", dataset_name)


def _sweeps_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "forecasts" / "feature_sweeps"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="Dataset root holding the MC_* target directories.")
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="Objective band defining the near-optimal set.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Combined CSV path. Defaults to <root>/summaries/"
                         + OUTPUT_NAME + ", derived from the root actually read.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")

    # Derive the output from the root that was read, never from a constant: a default
    # anchored elsewhere writes one tree's analysis into another tree's summaries.
    out_path = Path(args.output) if args.output else root / "summaries" / OUTPUT_NAME

    combined: list[pd.DataFrame] = []
    print(f"{'target':<32} {'band':>6} {'near-opt':>9} {'always':>7} {'best n':>7}")
    for dataset_dir in sorted(root.glob("MC_*")):
        sweeps = _sweeps_dir(dataset_dir)
        for trace_csv in sorted(sweeps.glob("feature_search_trace_r*.csv")):
            m = re.search(r"_r(\d+)\.csv$", trace_csv.name)
            row_count = int(m.group(1)) if m else 0
            try:
                trace_df = pd.read_csv(trace_csv)
            except Exception as exc:
                print(f"[WARN] Could not read {trace_csv}: {exc}")
                continue

            table, summary = selection_stability_from_trace(
                trace_df, tolerance=float(args.tolerance)
            )
            if table.empty:
                print(f"[WARN] No usable candidates in {trace_csv}")
                continue

            per_target = sweeps / f"feature_retention_frequency_r{row_count:03d}.csv"
            table.to_csv(per_target, index=False)

            tagged = table.copy()
            tagged.insert(0, "dataset", dataset_dir.name)
            tagged.insert(1, "target", _target_label(dataset_dir.name))
            tagged.insert(2, "row_count", row_count)
            for key in ("n_evaluated", "best_objective", "objective_band"):
                tagged[key] = summary[key]
            combined.append(tagged)

            print("%-32s %6.3f %9d %7d %7d" % (
                _target_label(dataset_dir.name)[:32],
                summary["objective_band"],
                summary["n_near_optimal"],
                summary["n_features_always_retained"],
                summary["n_features_best_subset"],
            ))

    if not combined:
        print("Nothing to report: no search traces found.")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = pd.concat(combined, ignore_index=True)
    all_rows.to_csv(out_path, index=False)
    print()
    print(f"Wrote {out_path} ({len(all_rows)} rows)")

    # A feature retained by every near-optimal subset for a target is a result; one
    # retained by some of them is not, and the difference is what this table exists to
    # keep visible.
    always = all_rows[all_rows["retention_frequency"] >= 1.0]
    print(f"Predictors always retained, summed over targets: {len(always)}")
    unresolved = all_rows[
        (all_rows["retention_frequency"] > 0.0) & (all_rows["retention_frequency"] < 1.0)
    ]
    print(f"Predictor/target pairs the search left unresolved: {len(unresolved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

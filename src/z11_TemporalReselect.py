"""Measure, and partly remove, the selection optimism in the reported R^2.

The sweep's feature search minimises ``(1 - r2) + lambda_drop * drop_rate`` where
``r2`` is the **test-split** R^2 (``_objective_from_metrics`` in
``h_RunMCFeatureSelectionSweep.py``, fed by the ``kind == 'test'`` row that
``evaluate_single_config`` returns). Verified against the archived trees: the
trace's ``r2`` reproduces R^2 recomputed from ``predictions.csv`` on the test rows
bit-for-bit, in the 2026-03-29 tree as well as the 2026-04-21 one. Every result
root on disk was produced this way.

So the reported R^2 is a maximum over a search steered by the same segments it is
reported on. It is an optimistically biased quantity, not an out-of-sample
estimate, and the bias is asymmetric: the naive, seasonal and linear forecasts are
single prediction columns with nothing to select among, while the learned families
and each MLR variant take a maximum over configurations.

This script splits the common evaluation set in time, chooses each family's
configuration on the earlier block, and scores that choice on the later one. Two
numbers come out per family:

  honest      R^2 on the later block of the configuration chosen on the earlier
              block -- the pick never saw the block it is scored on.
  optimistic  the best R^2 any of that family's configurations achieves on the
              later block -- what selecting on the reporting data would report.

**The measurement is the gap between them, not the level of either.** ``honest``
is computed on a shorter, later evaluation set than the headline R^2 and against
that block's own mean, so it is not comparable to the headline number and must
never be substituted for it. The gap is comparable, because both sides describe
the same segments.

**What this does not fix, and why the gap is a lower bound.** The candidate pool
was still produced by a search that saw the whole test split -- roughly 270
holdout scorings per target -- so this cleans the final pick and not the pool. The
gap therefore measures the optimism of choosing among the handful of retained
configurations, which is the smaller of the two stages; the optimism of the search
that produced them is not measured here and is not removed. Report the gap as a
lower bound on the total, never as the total. Removing the rest needs a third
block reserved before the search runs, which is a change to the sweep and
therefore to a run.

Targets whose common set is too short to divide are skipped and named, since a
block of three or four segments would produce a number too noisy to mean anything.

Usage:
    python src/z11_TemporalReselect.py
    python src/z11_TemporalReselect.py --root data/output/CV19 --min-block 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from utils import run_paths as rp

from z8_CommonSetMetrics import (
    ML_FAMILIES,
    REFERENCE_COLUMNS,
    _metrics,
    _record_sigma,
    common_evaluation_set,
    load_runs,
    target_label,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = rp.DEFAULT_ROOT


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((p - y) ** 2) / denom)


def analyse_target(dataset_dir: Path, min_block: int, verbose: bool = False) -> list[dict]:
    sweeps = dataset_dir / "forecasts" / "feature_sweeps"
    if not sweeps.is_dir():
        return []
    runs = [r for r in load_runs(sweeps) if r.retained]
    if not runs:
        return []
    label = target_label(dataset_dir)
    cs = common_evaluation_set(runs, label)
    if cs is None:
        return []
    _common, _union, eligible, ordered = cs

    n = len(ordered)
    n_sel = n // 2
    n_rep = n - n_sel
    if n_sel < min_block or n_rep < min_block:
        print(f"[SKIP] {label}: common set of {n} segment(s) splits to {n_sel}/{n_rep}, "
              f"below the {min_block}-segment minimum; no honest estimate is possible.")
        return []

    sel, rep = ordered[:n_sel], ordered[n_sel:]
    y_sel = np.array([eligible[0].targets[s] for s in sel], dtype=float)
    y_rep = np.array([eligible[0].targets[s] for s in rep], dtype=float)
    sigma = _record_sigma(dataset_dir)

    rows: list[dict] = []
    for fam in sorted({r.family for r in eligible}):
        cand = [r for r in eligible if r.family == fam]
        scored = []
        for r in cand:
            p_sel = np.array([r.preds[s] for s in sel], dtype=float)
            p_rep = np.array([r.preds[s] for s in rep], dtype=float)
            scored.append((r, _r2(y_sel, p_sel), _r2(y_rep, p_rep), p_rep))
        finite = [t for t in scored if np.isfinite(t[1])]
        rep_finite = [t for t in scored if np.isfinite(t[2])]
        if not finite or not rep_finite:
            continue
        pick = max(finite, key=lambda t: t[1])
        best_rep = max(rep_finite, key=lambda t: t[2])
        m = _metrics(y_rep, pick[3], sigma)
        rows.append(dict(
            dataset=dataset_dir.name,
            target=label,
            family=fam,
            n_common=n,
            n_select=n_sel,
            n_report=n_rep,
            n_candidates=len(cand),
            r2_select=pick[1],
            r2_report_honest=pick[2],
            r2_report_optimistic=best_rep[2],
            optimism_gap=best_rep[2] - pick[2],
            rmse_report_honest=m["rmse"],
            nrmse_report_honest=m["nrmse"],
            run_honest=pick[0].run,
            run_optimistic=best_rep[0].run,
            same_pick=bool(pick[0].run == best_rep[0].run),
        ))

    # References need no selection, so they are scored on the reporting block
    # directly. They are the fixed point the learned families have to beat, and
    # the one side of the comparison that carries no selection bias at all.
    donor = max(eligible, key=lambda r: (len(r.segments), r.run))
    for ref in REFERENCE_COLUMNS:
        src = donor if (ref in donor.refs and set(rep) <= set(donor.refs[ref])) else None
        if src is None:
            src = next((r for r in eligible
                        if ref in r.refs and set(rep) <= set(r.refs[ref])), None)
        if src is None:
            continue
        p = np.array([src.refs[ref][s] for s in rep], dtype=float)
        m = _metrics(y_rep, p, sigma)
        rows.append(dict(
            dataset=dataset_dir.name, target=label, family=ref,
            n_common=n, n_select=n_sel, n_report=n_rep, n_candidates=1,
            r2_select=float("nan"), r2_report_honest=m["r2"],
            r2_report_optimistic=m["r2"], optimism_gap=0.0,
            rmse_report_honest=m["rmse"], nrmse_report_honest=m["nrmse"],
            run_honest="(column)", run_optimistic="(column)", same_pick=True,
        ))

    if verbose:
        for r in rows:
            print(f"    {r['family']:12s} honest={r['r2_report_honest']:+.3f} "
                  f"optimistic={r['r2_report_optimistic']:+.3f} "
                  f"gap={r['optimism_gap']:+.3f} ({r['n_candidates']} cfg)")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--min-block", type=int, default=8,
                    help="Smallest acceptable selection and reporting block. Below this an "
                         "R^2 is too noisy to carry a claim, so the target is skipped and "
                         "named rather than reported.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = rp.resolve_root(args.root)
    output = (args.output.resolve() if args.output
              else root / "summaries" / "temporal_reselect.csv")
    if not root.is_dir():
        raise SystemExit(f"run root not found: {root}")

    all_rows: list[dict] = []
    for ds in sorted(root.glob("MC_*")):
        if ds.is_dir():
            if args.verbose:
                print(f"[INFO] {ds.name}")
            all_rows.extend(analyse_target(ds, args.min_block, args.verbose))

    if not all_rows:
        raise SystemExit("No target had a common evaluation set long enough to split.")

    df = pd.DataFrame(all_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8")
    print(f"[INFO] Wrote {output}")

    targets = sorted(df["target"].unique())
    print(f"[INFO] {len(targets)} target(s) had a common set long enough to split.")
    print()
    header = (f"{'target':30s} {'method':12s} {'cfg':>4s} {'honest':>8s} "
              f"{'optimistic':>11s} {'gap':>7s}")
    print(header)
    for t in targets:
        sub = df[df["target"] == t].sort_values("r2_report_honest", ascending=False)
        for i, (_, r) in enumerate(sub.iterrows()):
            name = t[:30] if i == 0 else ""
            print(f"{name:30s} {r['family']:12s} {int(r['n_candidates']):4d} "
                  f"{r['r2_report_honest']:+8.3f} {r['r2_report_optimistic']:+11.3f} "
                  f"{r['optimism_gap']:+7.3f}")
        print()

    learned = df[df["family"].isin(list(ML_FAMILIES))]
    changed = learned[~learned["same_pick"]]
    print(f"[INFO] For {len(changed)} of {len(learned)} learned-family rows the configuration "
          "chosen on the earlier block is not the one that scores best on the later block.")
    print("[INFO] The gap column is the measurement. The honest column is scored on a "
          "shorter, later block than the headline R^2 and is not a substitute for it.")
    print("[INFO] The candidate pool was still produced by a search that saw the whole test "
          "split, so this cleans the final pick, not the pool: the gap is a LOWER BOUND on "
          "the total selection optimism, not an estimate of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

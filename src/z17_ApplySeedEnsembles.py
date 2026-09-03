"""Replace refitted candidates' predictions with their seed ensemble, for z8 to select on.

Why
---
A candidate fitted once is a single draw from whatever its family's randomness is, and the
winner was chosen from that draw. XGBoost's ``random_state`` never reached the model, so
six seeds of the reported winners give an R^2 standard deviation of 0.03 (median) and up to
0.44; Turbidity's XGBoost configuration was chosen precisely because it looked best at
seed 0.

This is not confined to XGBoost. A Gaussian process ignores ``random_state`` entirely, but
its uncertain-input kernel draws Monte Carlo samples seeded by ``uncertain_kernel_mc_seed``,
and three seeds of pH's winner give a standard deviation of 0.0123 -- against a 0.0180 gap
between pH and the next target in the accuracy ordering. Since a Gaussian process wins 11
of 14 targets, that is most of the reported table. ``v3`` refits whichever families are
asked for; this applies whatever it produced, and the run directory glob below is
family-agnostic.

``v3_SeedVarianceRefit`` measures the spread. This applies it, by writing each refitted
candidate's **mean prediction vector across seeds** into the run directory ``z8`` reads.
Selection, the reported R^2 and the significance verdicts are then all computed from one
vector, which a mean-of-R^2 cannot give: there is no prediction series whose R^2 is the
average of six others, so reporting a mean R^2 beside a verdict derived from a single
seed would be quoting two different models.

Scored both ways, the seed-mean R^2 and the ensemble R^2 selected the identical family for
all 13 targets refitted in the first XGBoost-only pass; the ensemble is simply the one that
is internally consistent, and it is slightly higher because averaging predictions cancels
part of the seed noise (Cadmium +0.279 as a mean of six, +0.314 as an ensemble of six).
That agreement was measured on that pass and is not a guarantee for a wider one.

The original single-seed predictions are preserved alongside as
``predictions_seed0.csv``, so this is reversible.

Usage:
    python src/z17_ApplySeedEnsembles.py
    python src/z17_ApplySeedEnsembles.py --revert
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import z8_CommonSetMetrics as z8  # noqa: E402
from utils import run_paths as rp  # noqa: E402

BACKUP_NAME = 'predictions_seed0.csv'


def revert(root: Path, only: "tuple[str, ...] | None" = None) -> int:
    n = 0
    for bak in root.glob('MC_*/forecasts/feature_sweeps/*/' + BACKUP_NAME):
        if only and not any(t.lower() in str(bak).lower() for t in only):
            continue
        shutil.copy2(bak, bak.with_name('predictions.csv'))
        bak.unlink()
        n += 1
    print('[INFO] restored %d single-seed prediction file(s)' % n)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=None)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--datasets', default=None, help='Comma-separated dataset names or substrings; only these targets are processed. Without it every target in the root is, which re-fits work already on disk and can perturb values that are currently trusted. Use it when a single target has been re-run and only that target needs its seed ensemble rebuilt, e.g. --datasets Chromium,Lead.')
    ap.add_argument('--revert', action='store_true')
    args = ap.parse_args()

    root = rp.resolve_root(args.root)
    only = tuple(x.strip() for x in (args.datasets or '').split(',') if x.strip())
    if only:
        print('[INFO] restricted to target(s) matching: %s' % ', '.join(only))
    if args.revert:
        return revert(root, only)

    refit = root / 'summaries' / 'seed_refit.csv'
    if not refit.is_file():
        raise SystemExit('Not found: %s\nRun src/v3_SeedVarianceRefit.py first.' % refit)
    rf = pd.read_csv(refit)
    # A candidate whose seed 0 does not reproduce its reported score was fitted under
    # different conditions from the run being replaced, so its ensemble is not a
    # like-for-like substitute.
    rf = rf[rf['reproduces'].astype(bool)]
    if only:
        rf = rf[rf['dataset'].astype(str).apply(
            lambda d: any(t.lower() in d.lower() for t in only))]
        if rf.empty:
            raise SystemExit('--datasets %s matched no refitted candidate in %s'
                             % (args.datasets, refit))

    applied = skipped = 0
    for _, c in rf.iterrows():
        ds, run = str(c['dataset']), str(c['run'])
        vecs, frame = [], None
        for k in range(int(args.seeds)):
            d = root / ds / 'forecasts' / 'seed_refit' / ('%s_s%02d' % (run, k))
            p = next(d.rglob('predictions.csv'), None) if d.is_dir() else None
            if p is None:
                continue
            t = pd.read_csv(p, encoding='utf-8', encoding_errors='replace')
            col = z8._prediction_column(list(t.columns))
            if col is None:
                continue
            t = t.sort_values(['kind', 'sample_file']).reset_index(drop=True)
            vecs.append(t[col].to_numpy(dtype=float))
            frame, pred_col = t, col
        # One candidate can have several run directories: the beam-search stage writes
        # `<variant>_r###_<tag>_k01` and the final re-fit writes `<variant>_r###_<tag>`
        # with no suffix. They are the same model, and z8 scores whichever it finds, so
        # every copy has to receive the ensemble -- ensembling only the suffixed one left
        # z8 still selecting the single-seed sibling and nothing appeared to change.
        sweeps = root / ds / 'forecasts' / 'feature_sweeps'
        variant, tag = str(c['variant']), str(c['feature_tag'])
        targets = [d for d in sweeps.glob('%s_*%s*' % (variant, tag))
                   if d.is_dir() and (d / 'predictions.csv').exists()]
        if len(vecs) < 2 or frame is None or not targets:
            skipped += 1
            continue
        if any(v.shape != vecs[0].shape for v in vecs):
            skipped += 1
            continue

        out = frame.copy()
        out[pred_col] = np.mean(vecs, axis=0)
        for target_dir in targets:
            dest = target_dir / 'predictions.csv'
            if dest.exists() and not (target_dir / BACKUP_NAME).exists():
                shutil.copy2(dest, target_dir / BACKUP_NAME)
            out.to_csv(dest, index=False)
            applied += 1

    print('[INFO] wrote %d seed-ensemble prediction file(s); skipped %d' % (applied, skipped))
    print('[INFO] originals kept as %s in each run directory' % BACKUP_NAME)
    print('[INFO] now re-run: python src/z8_CommonSetMetrics.py --root %s' % root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

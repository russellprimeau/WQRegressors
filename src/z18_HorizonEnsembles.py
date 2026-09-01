"""Score each horizon's replicates as one ensemble, so the curve matches Table 3.

The inconsistency this removes
------------------------------
The horizon sweep fits a target's winning configuration once per replicate, differing in
nothing but the model seed, and ``z2``/``z16`` plot the **mean of the replicates' R^2**.
Table 3 reports the **R^2 of the mean prediction** -- the six-seed ensemble that
``z17_ApplySeedEnsembles`` installed. Those are not the same number, and cannot be:
squared error is convex in the prediction, so by Jensen's inequality

    R^2(mean prediction) - mean(R^2)  =  n * (mean across-seed variance) / SS_tot  >=  0

Measured on Cadmium at horizon 0 the two sides agree to six decimals (0.035342), and the
mean of the R^2s is 0.2787 against Table 3's 0.3141. The consequence is that Figure 8's
leftmost point disagreed with the results table for exactly the targets whose winner is
stochastic -- Cadmium and Lead. The twelve deterministic targets matched, because their
replicates are identical and the two aggregations coincide.

Ensembling is the side to converge on. Only the ensemble has a prediction series behind
it, so the R^2, the skill score and the permutation test all describe one model; and a
six-seed ensemble is something that could be deployed, where "the average score of six
models you would still have to choose between" is not.

Nothing is retrained: every replicate's ``predictions.csv`` is already on disk. The
per-horizon ensemble is written to ``lookahead_ensemble.csv`` beside
``lookahead_metrics.csv``, which ``z16`` prefers when present.

Usage:
    python src/z18_HorizonEnsembles.py
    python src/z18_HorizonEnsembles.py --root data/output/CV22_profilerless
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import z8_CommonSetMetrics as z8  # noqa: E402
from utils import run_paths as rp  # noqa: E402

OUTPUT_NAME = 'lookahead_ensemble.csv'


def _vector(preds_csv: Path, segments: list):
    """(prediction vector, target vector) on the common set, or (None, None)."""
    t = pd.read_csv(preds_csv, encoding='utf-8', encoding_errors='replace')
    t = t[t['kind'].astype(str) == 'test']
    col = z8._prediction_column(list(t.columns))
    if col is None:
        return None, None
    g = t.groupby('sample_file')[[col, 'target']].mean()
    if any(s not in g.index for s in segments):
        return None, None
    return (np.array([g.loc[s, col] for s in segments], dtype=float),
            np.array([g.loc[s, 'target'] for s in segments], dtype=float))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=None)
    args = ap.parse_args()
    root = rp.resolve_root(args.root)

    met = pd.read_csv(root / 'summaries' / 'common_set_metrics.csv').set_index('dataset')
    seg = pd.read_csv(root / 'summaries' / z8.SEGMENTS_NAME,
                      encoding='utf-8', encoding_errors='replace')

    n_targets = 0
    print('%-26s %8s %10s %10s %9s' % ('target', 'horizon', 'mean R2', 'ens R2', 'gap'))
    print('-' * 70)
    for ds_dir in sorted(root.glob('MC_*')):
        sweep = ds_dir / 'horizons' / 'lookahead_sweeps'
        metrics = sweep / 'lookahead_metrics.csv'
        if not metrics.is_file() or ds_dir.name not in met.index:
            continue
        segments = list(seg[seg.dataset == ds_dir.name]
                        .sort_values('order')['sample_file'].astype(str))
        sigma = float(met.loc[ds_dir.name, 'sigma_record'])
        base = pd.read_csv(metrics, encoding='utf-8', encoding_errors='replace')

        rows = []
        for horizon, g in base.groupby('horizon'):
            hdir = ds_dir / 'horizons' / ('%03dhr' % int(horizon))
            vecs, y = [], None
            for rep in sorted((hdir / 'ml' / 'forecasts').glob('rep_*')) \
                    if (hdir / 'ml' / 'forecasts').is_dir() else []:
                p = rep / 'predictions.csv'
                if not p.exists():
                    continue
                v, yy = _vector(p, segments)
                if v is not None:
                    vecs.append(v)
                    y = yy
            if not vecs or y is None:
                continue
            m = z8._metrics(y, np.mean(vecs, axis=0), sigma)
            row = {c: g.iloc[0][c] for c in
                   ('dataset', 'target', 'family', 'run', 'model_class', 'model_name',
                    'input_rows_included', 'input_rows_excluded') if c in g.columns}
            row.update(horizon=int(horizon), replicate='ensemble', n_fits=len(vecs),
                       r2=float(m['r2']), rmse=float(m['rmse']), nrmse=float(m['nrmse']),
                       mae=float(m['mae']) if 'mae' in m else float('nan'),
                       r2_mean_of_replicates=float(g['r2'].mean()))
            rows.append(row)
            if abs(row['r2'] - row['r2_mean_of_replicates']) > 1e-4:
                print('%-26s %8d %10.4f %10.4f %9.4f'
                      % (ds_dir.name.replace('MC_', '').replace('_diff', '')[:25],
                         int(horizon), row['r2_mean_of_replicates'], row['r2'],
                         row['r2'] - row['r2_mean_of_replicates']))
        if rows:
            pd.DataFrame(rows).to_csv(sweep / OUTPUT_NAME, index=False)
            n_targets += 1

    print('-' * 70)
    print('[INFO] wrote %s for %d target(s)' % (OUTPUT_NAME, n_targets))
    print('[INFO] rows not listed above are identical either way (deterministic winner)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

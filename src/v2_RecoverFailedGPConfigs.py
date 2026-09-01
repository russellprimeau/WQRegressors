"""Re-run the GP configurations that failed during the feature sweep, and score them.

Why this exists
---------------
On the CV22 sweep, 47 candidate GP configurations produced a run directory containing
only ``train_files.txt`` and ``test_files.txt`` and no result. They died inside the
training loop at a *logging-only* forward pass (the ``val_rmse`` line in
``train_gp_regressor_model``, which the code itself notes "does not drive model selection
or early stopping"), because summing the low-rank ``LinearKernel`` onto the dense Matern
term dispatched to ``LinearOperator.add_low_rank`` and its SVD did not converge.
``utils.gp_utils.DenseAdditiveKernel`` removes that path.

46 of the 47 are the two ``+linear`` variants, and ``gp_04`` (``matern52`` alone) failed
zero times, so the defect is the kernel rather than the data. Every one of the 47 had a
sibling variant that trained successfully on the identical predictor subset.

What this answers, and what it does not
--------------------------------------
It answers whether any recovered configuration would have beaten the GP result the
results table reports for its target -- which is the question that decides whether the
full 24 h sweep is worth repeating. R^2 varies by a median of 0.64 between GP variants on
one subset, so that cannot be inferred from the surviving siblings; it has to be measured.

It does **not** simulate a full re-run. The beam search's trajectory would also have
differed: a configuration that had not crashed might have seeded further subset
exploration. This bounds the impact rather than reproducing it.

Outputs are written to ``<dataset>/forecasts/gp_recovery/<run>`` so that nothing under
``feature_sweeps/`` changes and the reported results stay exactly as they are.

Usage:
    python src/v2_RecoverFailedGPConfigs.py
    python src/v2_RecoverFailedGPConfigs.py --limit 5      # smoke test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import z8_CommonSetMetrics as z8  # noqa: E402
from utils import run_paths as rp  # noqa: E402

RECOVERY_SUBDIR = 'gp_recovery'
# A flattened 11-predictor GP on Total coliforms ran 25 minutes on 0.016 s of CPU -- hung,
# not slow -- and stalled the whole recovery because subprocess.run waits forever by
# default. A fit that has not finished in this long is not going to.
FIT_TIMEOUT_S = 1800


def failed_candidates(root: Path) -> list[dict]:
    """Candidate GP runs with no predictions, paired with the config that made them."""
    out = []
    for ds in sorted(root.glob('MC_*')):
        sweeps = ds / 'forecasts' / 'feature_sweeps'
        if not sweeps.is_dir():
            continue
        for run in sorted(sweeps.iterdir()):
            if not (run.is_dir() and run.name.startswith('gp')):
                continue
            if (run / 'predictions.csv').exists():
                continue
            # _stab* runs are selection-stability replicates, not candidates for the
            # reported result, and have no config of their own.
            if '_stab' in run.name:
                continue
            cfg = sweeps / 'configs' / ('config_%s.yml' % run.name)
            if cfg.exists():
                out.append({'dataset': ds, 'run': run.name, 'config': cfg})
    return out


def stage_config(item: dict) -> Path:
    """Copy the config with its output redirected out of feature_sweeps/."""
    cfg = yaml.safe_load(open(item['config'], 'r', encoding='utf-8'))
    cfg['data']['forecast_name'] = '%s/%s' % (RECOVERY_SUBDIR, item['run'])
    staged = item['dataset'] / 'forecasts' / RECOVERY_SUBDIR / ('config_%s.yml' % item['run'])
    staged.parent.mkdir(parents=True, exist_ok=True)
    with open(staged, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    return staged


def score(preds_csv: Path, segments: list, sigma: float) -> dict:
    t = pd.read_csv(preds_csv, encoding='utf-8', encoding_errors='replace')
    t = t[t['kind'].astype(str) == 'test']
    col = z8._prediction_column(list(t.columns))
    if col is None:
        return {}
    grp = t.groupby('sample_file')
    p, y = grp[col].mean(), grp['target'].mean()
    keep = [s for s in segments if s in p.index and s in y.index
            and np.isfinite(p[s]) and np.isfinite(y[s])]
    if len(keep) < len(segments):
        return {'n_scored': len(keep), 'partial': True}
    m = z8._metrics(np.array([y[s] for s in keep], dtype=float),
                   np.array([p[s] for s in keep], dtype=float), sigma)
    return {'n_scored': len(keep), 'r2': float(m['r2']), 'rmse': float(m['rmse'])}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=None)
    ap.add_argument('--limit', type=int, default=None,
                    help='Run only the first N, for a smoke test.')
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()

    root = rp.resolve_root(args.root)
    summary = root / 'summaries' / 'common_set_metrics.csv'
    met = pd.read_csv(summary).set_index('dataset')
    seg = pd.read_csv(root / 'summaries' / z8.SEGMENTS_NAME,
                      encoding='utf-8', encoding_errors='replace')

    items = failed_candidates(root)
    if args.limit:
        items = items[:args.limit]
    print('[INFO] %d failed GP candidate configuration(s) to re-run' % len(items))

    rows = []
    for i, item in enumerate(items, 1):
        ds = item['dataset'].name
        staged = stage_config(item)
        print('[%2d/%2d] %-42s %s' % (i, len(items), ds.replace('MC_', '')[:41], item['run'][:34]))
        rec = {'dataset': ds, 'run': item['run'], 'variant': item['run'][:5]}
        try:
            subprocess.run([sys.executable, 'src/e_Train.py', '--config', str(staged)],
                           check=True, timeout=FIT_TIMEOUT_S,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired:
            rec['status'] = 'timeout'
            print('         timed out after %d s; skipping' % FIT_TIMEOUT_S)
            rows.append(rec)
            continue
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raw = getattr(exc, 'stderr', None) or b''
            tail = raw.decode(errors='replace').strip().splitlines() or ['timed out']
            rec['status'] = 'train_failed'
            rec['reason'] = tail[-1][:160] if tail else ''
            print('         still fails: %s' % rec['reason'][:110])
            rows.append(rec)
            continue

        out_dir = item['dataset'] / 'forecasts' / RECOVERY_SUBDIR / item['run']
        eval_cfg = out_dir / ('config_evaluate_%s.yml' % item['run'])
        try:
            subprocess.run([sys.executable, 'src/f_Evaluate.py', '--config',
                            str(eval_cfg if eval_cfg.exists() else staged)],
                           check=True, timeout=FIT_TIMEOUT_S,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raw = getattr(exc, 'stderr', None) or b''
            tail = raw.decode(errors='replace').strip().splitlines() or ['timed out']
            rec['status'] = 'eval_failed'
            rec['reason'] = tail[-1][:160] if tail else ''
            rows.append(rec)
            continue

        preds = next(out_dir.rglob('predictions.csv'), None)
        if preds is None:
            rec['status'] = 'no_predictions'
            rows.append(rec)
            continue

        segments = list(seg[seg['dataset'] == ds].sort_values('order')['sample_file'].astype(str))
        rec['status'] = 'ok'
        rec.update(score(preds, segments, float(met.loc[ds, 'sigma_record'])))
        rec['reported_gp_r2'] = float(met.loc[ds, 'gp_r2'])
        rec['reported_best_r2'] = float(met.loc[ds, 'best_r2'])
        rec['reported_best_family'] = str(met.loc[ds, 'best_family'])
        if 'r2' in rec:
            print('         R2 = %+.4f   (reported GP %+.4f, reported best %+.4f %s)'
                  % (rec['r2'], rec['reported_gp_r2'], rec['reported_best_r2'],
                     rec['reported_best_family']))
        rows.append(rec)

    df = pd.DataFrame(rows)
    out = args.output or (root / 'summaries' / 'gp_recovery.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print()
    print('[INFO] Wrote %s' % out)

    ok = df[df.get('status', pd.Series(dtype=str)) == 'ok'].copy()
    print('[INFO] %d of %d now train and score; %d still fail'
          % (len(ok), len(df), len(df) - len(ok)))
    if len(ok) and 'r2' in ok.columns:
        ok = ok[ok['r2'].notna()]
        beat_gp = ok[ok['r2'] > ok['reported_gp_r2'] + 1e-9]
        beat_best = ok[ok['r2'] > ok['reported_best_r2'] + 1e-9]
        print('[INFO] beat the reported GP result for their target : %d of %d'
              % (len(beat_gp), len(ok)))
        print('[INFO] beat the reported BEST result for their target: %d of %d'
              % (len(beat_best), len(ok)))
        if len(beat_best):
            print()
            print('       targets whose reported result would change:')
            for ds, g in beat_best.groupby('dataset'):
                r = g.loc[g['r2'].idxmax()]
                print('         %-40s %+.4f -> %+.4f  (%s)'
                      % (ds.replace('MC_', '')[:39], r['reported_best_r2'], r['r2'], r['run'][:30]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

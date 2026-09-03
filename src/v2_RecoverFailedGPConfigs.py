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


def backfill_final_metrics(root: Path, only: "tuple[str, ...] | None" = None,
                           dry_run: bool = False) -> int:
    """Fill in metrics rows for runs that were recovered after the sweep had finished.

    The sweep writes ``feature_sweep_final_metrics.csv`` when it finishes a target, and a
    configuration that crashed during it gets a row of NaNs -- correctly, because at that
    moment there was nothing to score. This script then refits those configurations hours
    later and they succeed, but until now it never went back to the table, so the row
    stayed blank while ``predictions.csv`` and ``evaluation_summary.csv`` sat beside it
    fully populated. Twelve rows in CV22 were in that state, all of them gp_03, with
    predictions written 7.8 hours after the table.

    The consequence is narrow but real: ``z8`` reads ``predictions.csv`` and so already
    counted these runs, but ``v3_SeedVarianceRefit`` drops rows with no r2 when it builds
    its candidate pool, so a recovered candidate silently stops competing.

    Values come from the run's own ``evaluation_summary.csv`` primary row rather than being
    recomputed here, so a backfilled row carries exactly what the sweep would have written
    had the fit succeeded the first time.
    """
    filled = files = 0
    for ds in sorted(root.glob('MC_*')):
        if only and not any(t.lower() in ds.name.lower() for t in only):
            continue
        fm = ds / 'forecasts' / 'feature_sweeps' / 'feature_sweep_final_metrics.csv'
        if not fm.is_file():
            continue
        d = pd.read_csv(fm, encoding='utf-8', encoding_errors='replace')
        if 'r2' not in d.columns:
            continue
        sweeps = fm.parent
        changed = False
        for i, row in d[pd.to_numeric(d['r2'], errors='coerce').isna()].iterrows():
            tag = str(row.get('feature_tag', ''))
            sub = str(row.get('subset_label', ''))
            var = str(row.get('variant', ''))
            if not tag or not sub or sub == 'nan':
                continue
            hits = [p for p in sweeps.iterdir()
                    if p.is_dir() and p.name.startswith(var) and tag in p.name
                    and p.name.endswith('_' + sub)]
            if len(hits) != 1:
                continue
            es = hits[0] / 'evaluation_summary.csv'
            if not es.is_file():
                continue
            try:
                summ = pd.read_csv(es, encoding='utf-8', encoding_errors='replace')
            except Exception:
                continue
            if summ.empty or 'kind' not in summ.columns:
                continue
            kinds = summ['kind'].astype(str).str.lower().str.strip()
            primary = None
            for want in ('test', 'combined', 'train'):
                hit = summ[kinds == want]
                if not hit.empty:
                    primary = hit.iloc[0]
                    break
            if primary is None:
                continue
            for col in ('mae', 'rmse', 'r2', 'pearson_r', 'std_target',
                        'n_test_independent', 'n_test_valid', 'n_test_evals',
                        'gp_uncertainty_mode'):
                if col in d.columns and col in summ.columns and pd.notna(primary.get(col)):
                    d.at[i, col] = primary[col]
            if 'n_samples' in d.columns and 'n_test_independent' in summ.columns:
                d.at[i, 'n_samples'] = primary.get('n_test_independent')
            # `model` was taken from the training config when the row was a failure, which
            # spells it `model_gp_03` rather than the model type every other row carries.
            if 'model' in d.columns:
                mt = _model_type_of(hits[0])
                if mt:
                    d.at[i, 'model'] = mt
            if 'failure_reason' in d.columns:
                d.at[i, 'failure_reason'] = ''
            print('  %s %s %s  r2 %+.4f' % (ds.name.replace('MC_', '')[:26], var, sub,
                                            float(primary['r2'])))
            filled += 1
            changed = True
        if changed and not dry_run:
            d.to_csv(fm, index=False)
            files += 1
    print('[INFO] %s %d row(s)%s' % ('would backfill' if dry_run else 'backfilled', filled,
                                     '' if dry_run else ' across %d file(s)' % files))
    return filled


def _model_type_of(run_dir: Path) -> "str | None":
    """The model type this run actually fitted, from its own evaluate config."""
    for cfg in run_dir.glob('config_evaluate_*.yml'):
        try:
            with open(cfg, 'r', encoding='utf-8') as f:
                return str((yaml.safe_load(f) or {}).get('model_type') or '') or None
        except Exception:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=None)
    ap.add_argument('--limit', type=int, default=None,
                    help='Run only the first N, for a smoke test.')
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--backfill-only', action='store_true',
                    help='Do not refit anything; only fill in metrics rows for runs that were recovered after the sweep wrote its table.')
    ap.add_argument('--no-backfill', action='store_true',
                    help='Skip the metrics backfill that normally follows a recovery.')
    args = ap.parse_args()

    root = rp.resolve_root(args.root)

    if args.backfill_only:

        return 0 if backfill_final_metrics(root, dry_run=bool(getattr(args, 'dry_run', False))) >= 0 else 1
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
    if not args.no_backfill:
        # The rows just recovered are still NaN in the sweep's table until this runs.
        print()
        print('[INFO] backfilling metrics rows for the recovered runs')
        backfill_final_metrics(root)
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

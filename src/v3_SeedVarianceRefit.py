"""Refit near-winning candidates at several seeds, and re-select on the seed average.

The problem
-----------
XGBoost's ``random_state`` never reached the model (it was absent from the constructor
argument list in ``_train_xgb_model``), so every candidate in the sweep was fitted at
XGBoost's default seed of 0 and the winner was chosen from a single draw. Once the seed
is honoured, six seeds of each target's *winning* configuration give a standard deviation
of 0.078 for Cadmium, 0.079 for Total coliforms, 0.116 for Turbidity and 0.441 for Lead --
and three of the five reported XGBoost wins fall behind their target's Gaussian process
when averaged over seeds.

An error bar on the reported winner does not fix that, because the bias is in the
*selection*: Turbidity's XGBoost configuration was chosen precisely because it looked best
at seed 0, and regresses to +0.077 over six. Fixing it requires seed-averaging the
candidates that could plausibly win, then re-selecting.

Why this is cheap
-----------------
The candidate pool is already on disk with every config, so nothing needs re-searching. A
candidate can only change a target's outcome if its single-seed score is within roughly
the seed noise of the current best, which is a small fraction of the pool. Gaussian
processes and MLR are deterministic -- measured seed standard deviation is exactly 0.0000
on eight of nine GP targets -- so only the stochastic families need refitting.

The margin is checked, not assumed
----------------------------------
Seed noise is not a single number: across the five XGBoost winners it spans 0.0021
(Copper, whose winner uses one predictor, leaving ``colsample_bytree`` nothing to sample)
to 0.4414 (Lead). So candidates are selected with a deliberately generous margin, the
standard deviation is then *measured* per candidate from its own refits, and afterwards
every excluded candidate is checked against twice the measured value. Targets that fail
that check are reported, and can be re-run at a wider margin rather than silently trusted.

What this does not do
---------------------
It cannot recover the beam search's trajectory. Which predictor subsets were explored is
fixed at what the sweep found, and both the GP crashes and the surrogate's own seed would
have changed that. This makes the *selection* among discovered candidates seed-robust; it
does not simulate a full re-run.

Usage:
    python src/v3_SeedVarianceRefit.py --margin 0.25 --seeds 6
    python src/v3_SeedVarianceRefit.py --dry-run          # cost only, fits nothing
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import z8_CommonSetMetrics as z8  # noqa: E402
from utils import run_paths as rp  # noqa: E402
from utils.training import aggregation_slug  # noqa: E402

REFIT_SUBDIR = 'seed_refit'
# A flattened 11-predictor GP on Total coliforms ran 25 minutes on 0.016 s of CPU -- hung,
# not slow -- and stalled the whole recovery because subprocess.run waits forever by
# default. A fit that has not finished in this long is not going to.
FIT_TIMEOUT_S = 1800
# Model key in feature_sweep_final_metrics.csv -> whether the fit is stochastic.
FAMILY_KEY = {'xgb': 'xgb_regressor', 'transformer': 'transformer'}
# Per-fit cost, measured from consecutive prediction-file timestamps in the sweep.
FIT_SECONDS = {'xgb': 5.1, 'transformer': 27.2}


def candidates(root: Path, margin: float, families: tuple[str, ...]) -> pd.DataFrame:
    """Every stochastic candidate close enough to its target's best to matter."""
    z = pd.read_csv(root / 'summaries' / 'common_set_metrics.csv').set_index('dataset')
    rows = []
    for ds in sorted(root.glob('MC_*')):
        f = ds / 'forecasts' / 'feature_sweeps' / 'feature_sweep_final_metrics.csv'
        if not f.exists() or ds.name not in z.index:
            continue
        d = pd.read_csv(f, encoding='utf-8', encoding_errors='replace')
        d['r2'] = pd.to_numeric(d['r2'], errors='coerce')
        best = float(z.loc[ds.name, 'best_r2'])
        for fam in families:
            g = d[d['model'].astype(str) == FAMILY_KEY[fam]].dropna(subset=['r2'])
            for _, r in g.iterrows():
                rows.append(dict(
                    dataset=ds.name, family=fam, variant=str(r.get('variant', '')),
                    feature_tag=str(r.get('feature_tag', '')),
                    subset_label=str(r.get('subset_label', '')),
                    r2_single=float(r['r2']), target_best=best,
                    selected=bool(float(r['r2']) >= best - margin)))
    return pd.DataFrame(rows)


def find_config(root: Path, ds: str, variant: str, feature_tag: str,
                subset_label: str) -> Path | None:
    """The sweep's own config for this candidate, matched on variant/tag/subset."""
    cfg_dir = root / ds / 'forecasts' / 'feature_sweeps' / 'configs'
    if not cfg_dir.is_dir():
        return None
    pats = ['config_%s_*%s_%s.yml' % (variant, feature_tag, subset_label),
            'config_%s_*%s*.yml' % (variant, feature_tag)]
    for pat in pats:
        hits = sorted(cfg_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _reported_rounds(run_dir: Path) -> int | None:
    """Boosting rounds the reported fit actually trained, from its saved booster."""
    model = run_dir / 'xgboost_model.json'
    if not model.exists():
        return None
    try:
        with open(model, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        n = int(payload['learner']['gradient_booster']['model']
                ['gbtree_model_param']['num_trees'])
    except Exception:
        return None
    return n if n > 0 else None


def _stage_seed_config(base: dict, cfg_path: Path, run_dir: Path, seed: int,
                       out_name: str) -> dict:
    """Config for one seed, with the tuning cache neutralised.

    Leaving ``cv_tuning`` enabled makes this measurement meaningless, and the first
    version of this script did exactly that. The cache is applied *after* the config is
    parsed and replaces the hyperparameter dict wholesale, which both discards the
    ``random_state`` set here -- every candidate came back with a seed spread of exactly
    zero -- and restores the cached ``n_estimators`` in place of the CV-derived round
    budget the reported fit used. Cadmium's winner refitted to -0.805 instead of +0.428,
    which is the same over-boosting the horizon sweep suffered from the same cause.

    So the tuned values are inlined, the cache is switched off, the reported round count
    is kept as a ceiling, and only then is the seed set. Seed 0 must then reproduce the
    reported score, and the caller checks that it does.
    """
    cfg = yaml.safe_load(yaml.dump(base))
    hyper = cfg.setdefault('hyperparameters', {})
    agg = (cfg.get('data') or {}).get('input_aggregation', 'none')
    slug = aggregation_slug(agg)
    cache = cfg_path.parent.parent / ('xgb_cv_tuning_cache%s.json' % slug)
    if cache.exists():
        try:
            with open(cache, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            tuned = payload.get('tuned_hyperparameters') or payload.get('best_params') or {}
            hyper.update({k: v for k, v in tuned.items() if k != 'cv_tuning'})
        except Exception:
            pass
    rounds = _reported_rounds(run_dir)
    if rounds is not None:
        hyper['n_estimators'] = rounds
        # Also silence the CV round-budget estimator. With the cache off it runs again
        # and re-derives a budget that overrides the pin -- for xgb_01 configs it landed
        # somewhere else and 13 candidates then failed to reproduce their reported score.
        # The reported fit trained exactly `rounds` rounds, so training exactly that many
        # is what reproduces it; the estimator only has a job when the budget is unknown.
        hyper['early_stopping_rounds'] = None
    cv = dict(hyper.get('cv_tuning') or {})
    cv['enabled'] = False
    hyper['cv_tuning'] = cv
    hyper['random_state'] = int(seed)
    cfg['data']['forecast_name'] = out_name
    return cfg


def score(preds_csv: Path, segments: list, sigma: float) -> float:
    t = pd.read_csv(preds_csv, encoding='utf-8', encoding_errors='replace')
    t = t[t['kind'].astype(str) == 'test']
    col = z8._prediction_column(list(t.columns))
    if col is None:
        return float('nan')
    grp = t.groupby('sample_file')
    p, y = grp[col].mean(), grp['target'].mean()
    if any(s not in p.index or s not in y.index for s in segments):
        return float('nan')
    return float(z8._metrics(np.array([y[s] for s in segments], dtype=float),
                             np.array([p[s] for s in segments], dtype=float),
                             sigma)['r2'])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=None)
    ap.add_argument('--margin', type=float, default=0.25,
                    help='Refit candidates within this much of the target best (default 0.25).')
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--base-seed', type=int, default=0,
                    help='Seed 0 reproduces the reported fit, since XGBoost defaults to 0.')
    ap.add_argument('--families', default='xgb',
                    help='Comma-separated: xgb,transformer. Default xgb -- the Transformer '
                         'wins no target and its seeding was only just made to work.')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    root = rp.resolve_root(args.root)
    fams = tuple(x.strip() for x in args.families.split(',') if x.strip())
    cand = candidates(root, args.margin, fams)
    sel = cand[cand.selected].reset_index(drop=True)
    if args.limit:
        sel = sel.iloc[:args.limit]

    extra = int(args.seeds) - 1
    cost = sum(FIT_SECONDS[f] for f in sel.family) * extra
    print('[INFO] %d of %d candidates are within %.3f of their target best'
          % (len(sel), len(cand), args.margin))
    print('[INFO] %d extra fits at %d seeds  =>  about %.1f h'
          % (len(sel) * extra, args.seeds, cost / 3600))
    for fam, g in sel.groupby('family'):
        print('         %-12s %3d candidates' % (fam, len(g)))
    if args.dry_run:
        return 0

    z = pd.read_csv(root / 'summaries' / 'common_set_metrics.csv').set_index('dataset')
    segs_all = pd.read_csv(root / 'summaries' / z8.SEGMENTS_NAME,
                           encoding='utf-8', encoding_errors='replace')

    out_rows = []
    for i, c in sel.iterrows():
        ds = c['dataset']
        cfg_path = find_config(root, ds, c['variant'], c['feature_tag'], c['subset_label'])
        if cfg_path is None:
            print('[WARN] no config for %s %s %s; skipping'
                  % (ds, c['variant'], c['feature_tag']))
            continue
        segments = list(segs_all[segs_all['dataset'] == ds]
                        .sort_values('order')['sample_file'].astype(str))
        sigma = float(z.loc[ds, 'sigma_record'])
        base = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
        run_id = cfg_path.stem.replace('config_', '')
        print('[%3d/%3d] %-38s %s' % (i + 1, len(sel), ds.replace('MC_', '')[:37], run_id[:34]))

        vals = []
        for k in range(int(args.seeds)):
            seed = int(args.base_seed) + k
            name = '%s/%s_s%02d' % (REFIT_SUBDIR, run_id, seed)
            cfg = _stage_seed_config(
                base, cfg_path,
                root / ds / 'forecasts' / 'feature_sweeps' / run_id, seed, name)
            staged = root / ds / 'forecasts' / REFIT_SUBDIR / ('config_%s_s%02d.yml' % (run_id, seed))
            staged.parent.mkdir(parents=True, exist_ok=True)
            with open(staged, 'w', encoding='utf-8') as fh:
                yaml.dump(cfg, fh, sort_keys=False, allow_unicode=True)
            try:
                subprocess.run([sys.executable, 'src/e_Train.py', '--config', str(staged)],
                               check=True, timeout=FIT_TIMEOUT_S,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                out_dir = root / ds / 'forecasts' / name
                ev = out_dir / ('config_evaluate_%s_s%02d.yml' % (run_id, seed))
                subprocess.run([sys.executable, 'src/f_Evaluate.py', '--config',
                                str(ev if ev.exists() else staged)],
                               check=True, timeout=FIT_TIMEOUT_S,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                preds = next(out_dir.rglob('predictions.csv'), None)
                vals.append(score(preds, segments, sigma) if preds else float('nan'))
            except subprocess.TimeoutExpired:
                print('          seed %d timed out after %d s' % (seed, FIT_TIMEOUT_S))
                vals.append(float('nan'))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raw = getattr(exc, 'stderr', None) or b''
                tail = raw.decode(errors='replace').strip().splitlines() or ['timed out']
                print('          seed %d failed: %s' % (seed, tail[-1][:100] if tail else ''))
                vals.append(float('nan'))

        v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
        rec = dict(c)
        rec.update(run=run_id, n_ok=int(v.size),
                   r2_mean=float(v.mean()) if v.size else float('nan'),
                   r2_sd=float(v.std(ddof=1)) if v.size > 1 else float('nan'),
                   r2_min=float(v.min()) if v.size else float('nan'),
                   r2_max=float(v.max()) if v.size else float('nan'))
        # Seed 0 is XGBoost's own default, so it must reproduce the reported score.
        # Without this check a systematically wrong refit looks like seed variance.
        s0 = vals[0] if vals else float('nan')
        rec['r2_seed0'] = float(s0)
        rec['reproduces'] = bool(np.isfinite(s0) and abs(s0 - c['r2_single']) < 0.002)
        print('          seed0 %+.4f vs reported %+.4f %s | mean %+.4f  sd %.4f'
              % (s0, c['r2_single'], 'OK' if rec['reproduces'] else '<-- MISMATCH',
                 rec['r2_mean'], rec['r2_sd']))
        out_rows.append(rec)

    df = pd.DataFrame(out_rows)
    out = root / 'summaries' / 'seed_refit.csv'
    df.to_csv(out, index=False)
    print()
    print('[INFO] Wrote %s' % out)
    if 'reproduces' in df.columns and len(df):
        ok = int(df['reproduces'].sum())
        print('[INFO] seed 0 reproduces the reported score for %d of %d candidates'
              % (ok, len(df)))
        if ok < len(df):
            print('[WARN] %d candidate(s) do NOT reproduce at seed 0. Their seed spread is'
                  % (len(df) - ok))
            print('       not comparable with the reported value and the re-selection below')
            print('       should not be trusted for the targets involved.')

    # --- re-selection and the margin check -----------------------------------------
    print()
    print('%-26s %10s %10s %-10s %s' % ('target', 'reported', 'seed-avg', 'new best', 'margin check'))
    print('-' * 82)
    for ds, g in df.groupby('dataset'):
        rep_best = float(z.loc[ds, 'best_r2'])
        rep_fam = str(z.loc[ds, 'best_family'])
        det = max(float(z.loc[ds, k]) for k in ('gp_r2', 'mlr_r2')
                  if k in z.columns and pd.notna(z.loc[ds, k]))
        g = g[g['reproduces']] if 'reproduces' in g.columns else g
        stoch = g['r2_mean'].max() if len(g) and g['r2_mean'].notna().any() else float('-inf')
        new_best = max(det, stoch)
        new_fam = ('deterministic (GP/MLR)' if det >= stoch
                   else str(g.loc[g['r2_mean'].idxmax(), 'family']))
        # Median, not max: one candidate with sd 1.04 would otherwise widen the band to
        # +/-2.09 and flag every excluded candidate, which is not a useful check.
        sd_max = float(g['r2_sd'].median()) if g['r2_sd'].notna().any() else 0.0
        excluded = cand[(cand.dataset == ds) & (~cand.selected)]
        risky = excluded[excluded.r2_single >= new_best - 2 * sd_max]
        flag = 'ok' if risky.empty else 'EXTEND: %d excluded within 2sd' % len(risky)
        print('%-26s %10.4f %10.4f %-10s %s'
              % (ds.replace('MC_', '').replace('_diff', '')[:25], rep_best, new_best,
                 new_fam[:10], flag))
    print()
    print('[INFO] "EXTEND" means an excluded candidate sits within twice the measured seed')
    print('       spread of the new best, so the margin was too tight for that target.')
    print('       Re-run with a larger --margin for those before trusting the selection.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

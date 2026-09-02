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

Which candidates are refitted
-----------------------------
Seed noise is not one number: across these configurations the measured spread runs from
0.0024 (Zinc) to 0.2395 (pH), two orders of magnitude. A fixed R2 margin therefore
over- and under-selects at the same time -- at 0.25 it refitted all 21 of E. coli's
candidates, whose spread is 0.0036 and which cannot reorder at all, and only 4 of pH's,
whose spread is 0.24 and which certainly can.

The band is measured instead: ``k * sqrt(2) * sd`` around each family's best, where sd is
the seed spread of that family's refits and sqrt(2) accounts for both compared values
being single draws. sd is not known before refitting, so it is bootstrapped -- the
family's leader is refitted first, the band computed from its spread, everything inside
refitted, sd recomputed, and the band widened until it stops growing.

Checked against what actually happened: the largest gap between the single-seed leader
and the eventual seed-mean winner was 0.0988 (Total coliforms), which k=3 clears by
0.0017. The default k=4 leaves headroom.

The anchor is the family's own best, not the overall best, so a family whose best sits far
below the winner still gets a seed-robust value of its own -- under a cross-family anchor
Arsenic and Chromium refitted no XGBoost candidate at all. A candidate that overtakes the
overall best on the seed mean still takes the win: z17 installs the ensembled predictions
and z8 re-scores every family against them.

Candidates are deduplicated on (variant, feature_tag). The l01/m01/s01 subset routes
resolve to the same feature set on 11 of 14 targets, which is 21% of the pool; z17 installs
one ensemble into every run directory matching the pair, so fitting each separately would
repeat identical work.

What this does not do
---------------------
It cannot recover the beam search's trajectory. Which predictor subsets were explored is
fixed at what the sweep found, and both the GP crashes and the surrogate's own seed would
have changed that. This makes the *selection* among discovered candidates seed-robust; it
does not simulate a full re-run.

Usage:
    # every target in the default root
    python src/v3_SeedVarianceRefit.py --root data/output/CV22_profilerless --seeds 6

    # only the targets that were re-run, leaving the rest untouched
    python src/v3_SeedVarianceRefit.py --root data/output/CV22_profilerless \
        --datasets Chromium,Lead --seeds 6

    # cost only, fits nothing
    python src/v3_SeedVarianceRefit.py --root data/output/CV22_profilerless --dry-run

    # reproduce the superseded fixed-margin behaviour
    python src/v3_SeedVarianceRefit.py --root data/output/CV22_profilerless --margin 0.25

Then apply the ensembles and re-score:
    python src/z17_ApplySeedEnsembles.py --root <root> [--datasets ...] --seeds 6
    python src/z8_CommonSetMetrics.py    --root <root>
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
# The band can only widen, and the pool is finite, so this is a guard against a
# pathological sd rather than an expected limit.
MAX_EXPANSIONS = 6
# A flattened 11-predictor GP on Total coliforms ran 25 minutes on 0.016 s of CPU -- hung,
# not slow -- and stalled the whole recovery because subprocess.run waits forever by
# default. A fit that has not finished in this long is not going to.
FIT_TIMEOUT_S = 1800
# Model key in feature_sweep_final_metrics.csv -> whether the fit is stochastic.
FAMILY_KEY = {'xgb': 'xgb_regressor', 'transformer': 'transformer'}
# Per-fit cost, measured from consecutive prediction-file timestamps in the sweep.
FIT_SECONDS = {'xgb': 5.1, 'transformer': 27.2}


def candidates(root: Path, families: tuple[str, ...],
               only: "tuple[str, ...] | None" = None) -> pd.DataFrame:
    """Every candidate of each family, with that family's best single-seed score.

    Selection is no longer a fixed R2 margin. A fixed threshold cannot be right for
    every target: the measured seed spread of these configurations ranges from 0.0024
    (Zinc) to 0.2395 (pH), two orders of magnitude, so one number over- and
    under-selects at the same time. At 0.25 it refit all 21 of E. coli's candidates,
    whose spread is 0.0036 and which therefore cannot reorder at all, while refitting
    only 4 of pH's, whose spread is 0.24 and which certainly can. `main` sets the band
    from measured seed variance instead.

    The anchor is the family's own best, not the overall best. The two questions differ:
    whether a candidate can take the *overall* win is settled by the cross-family best,
    but each family's reported value is also quoted per target, and under the old anchor
    a family whose best sat far below the winner had none of its candidates refit at all
    -- Arsenic and Chromium refit zero XGBoost candidates. A candidate that does overtake
    the overall best on the seed mean still wins it: z17 installs the ensembled
    predictions and z8 re-scores every family against them.

    *only* restricts the scan to the named targets, so a single re-run target can be
    rebuilt without disturbing work already on disk.
    """
    z = pd.read_csv(root / 'summaries' / 'common_set_metrics.csv').set_index('dataset')
    rows = []
    for ds in sorted(root.glob('MC_*')):
        if only and not any(t.lower() in ds.name.lower() for t in only):
            continue
        f = ds / 'forecasts' / 'feature_sweeps' / 'feature_sweep_final_metrics.csv'
        if not f.exists() or ds.name not in z.index:
            continue
        d = pd.read_csv(f, encoding='utf-8', encoding_errors='replace')
        d['r2'] = pd.to_numeric(d['r2'], errors='coerce')
        best = float(z.loc[ds.name, 'best_r2'])
        for fam in families:
            g = d[d['model'].astype(str) == FAMILY_KEY[fam]].dropna(subset=['r2'])
            if g.empty:
                continue
            fam_best = float(g['r2'].max())
            for _, r in g.iterrows():
                rows.append(dict(
                    dataset=ds.name, family=fam, variant=str(r.get('variant', '')),
                    feature_tag=str(r.get('feature_tag', '')),
                    subset_label=str(r.get('subset_label', '')),
                    r2_single=float(r['r2']), target_best=best, family_best=fam_best))
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


def refit_one(root: Path, c, args, z, segs_all, idx: int, total: int):
    """Refit one candidate at `args.seeds` seeds; return its record or None."""
    ds = c['dataset']
    cfg_path = find_config(root, ds, c['variant'], c['feature_tag'], c['subset_label'])
    if cfg_path is None:
        print('[WARN] no config for %s %s %s; skipping'
              % (ds, c['variant'], c['feature_tag']))
        return None
    segments = list(segs_all[segs_all['dataset'] == ds]
                    .sort_values('order')['sample_file'].astype(str))
    sigma = float(z.loc[ds, 'sigma_record'])
    base = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    run_id = cfg_path.stem.replace('config_', '')
    print('[%3d/%3d] %-38s %s' % (idx, total, ds.replace('MC_', '')[:37], run_id[:34]))

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
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=None)
    ap.add_argument('--k', type=float, default=4.0,
                    help='Width of the refit band in seed standard deviations (default 4). '
                         'The band is k*sqrt(2)*sd around the family best, sd being the '
                         'measured spread across seeds and sqrt(2) accounting for both '
                         'values being single draws. Checked against what actually '
                         'happened: the largest gap between the single-seed leader and '
                         'the eventual seed-mean winner was 0.0988 (Total coliforms), '
                         'which k=3 captures by 0.0017; k=4 leaves headroom.')
    ap.add_argument('--margin', type=float, default=None,
                    help='Fixed R2 margin instead of the seed-variance band. Only for '
                         'reproducing an earlier run; not defensible as a default.')
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--base-seed', type=int, default=0,
                    help='Seed 0 reproduces the reported fit, since XGBoost defaults to 0.')
    ap.add_argument('--families', default='xgb',
                    help='Comma-separated: xgb,transformer. Default xgb -- the Transformer '
                         'wins no target and its seeding was only just made to work.')
    ap.add_argument('--datasets', default=None, help='Comma-separated dataset names or substrings; only these targets are processed. Without it every target in the root is, which re-fits work already on disk and can perturb values that are currently trusted. Use it when a single target has been re-run and only that target needs its seed ensemble rebuilt, e.g. --datasets Chromium,Lead.')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    root = rp.resolve_root(args.root)
    fams = tuple(x.strip() for x in args.families.split(',') if x.strip())
    only = tuple(x.strip() for x in (args.datasets or '').split(',') if x.strip())
    if only:
        print('[INFO] restricted to target(s) matching: %s' % ', '.join(only))
    cand = candidates(root, fams, only)
    if only and cand.empty:
        raise SystemExit('--datasets %s matched no target under %s'
                         % (args.datasets, root))
    if args.margin is not None:
        # The fixed rule, kept only so an earlier run can be reproduced exactly.
        cand['selected'] = cand['r2_single'] >= cand['target_best'] - float(args.margin)
        print('[INFO] fixed margin %.3f: %d of %d candidates selected (the seed-variance '
              'band is the default; this is for reproducing an earlier run)'
              % (args.margin, int(cand['selected'].sum()), len(cand)))
    sel = cand[cand['selected']].reset_index(drop=True) if args.margin is not None else cand
    if args.limit:
        sel = sel.iloc[:args.limit]
        cand = cand.loc[sel.index] if args.margin is None else cand

    extra = int(args.seeds) - 1
    if args.margin is None:
        print('[INFO] refit band is k=%g seed standard deviations around each family best; '
              'the pool is %d candidate(s) across %d target(s) and the band is measured '
              'per target, so the number actually refit is decided as it runs.'
              % (args.k, len(cand), cand['dataset'].nunique()))
        print('[INFO] worst case %d extra fits at %d seeds  =>  up to about %.1f h'
              % (len(cand) * extra, args.seeds,
                 sum(FIT_SECONDS[f] for f in cand.family) * extra / 3600))
    else:
        print('[INFO] %d extra fits at %d seeds  =>  about %.1f h'
              % (len(sel) * extra, args.seeds,
                 sum(FIT_SECONDS[f] for f in sel.family) * extra / 3600))
    if args.dry_run:
        return 0

    z = pd.read_csv(root / 'summaries' / 'common_set_metrics.csv').set_index('dataset')
    segs_all = pd.read_csv(root / 'summaries' / z8.SEGMENTS_NAME,
                           encoding='utf-8', encoding_errors='replace')

    out_rows = []
    done: set = set()
    total_planned = len(sel)
    for (ds, fam), pool in cand.groupby(['dataset', 'family'], sort=True):
        pool = pool.sort_values('r2_single', ascending=False)
        fam_best = float(pool['r2_single'].max())
        # Stage 1 is the family's own leader: its seed spread is the only estimate of
        # the band available before anything has been refitted.
        todo = [pool.iloc[0]]
        margin = 0.0
        for _round in range(MAX_EXPANSIONS):
            for c in todo:
                # Keyed without subset_label: l01/m01/s01 are different routes to the
                # same feature set and resolve to the same tag on 11 of 14 targets, so
                # fitting each separately repeats identical work -- 21% of this pool.
                # One refit is enough because z17 installs the ensemble into every run
                # directory matching `<variant>_*<tag>*`, which is all of them.
                key = (ds, fam, c['variant'], c['feature_tag'])
                if key in done:
                    continue
                done.add(key)
                rec = refit_one(root, c, args, z, segs_all, len(done), total_planned)
                if rec is not None:
                    rec['family_best'] = fam_best
                    rec['margin'] = margin
                    out_rows.append(rec)
            got = [r for r in out_rows
                   if r['dataset'] == ds and r['family'] == fam
                   and r.get('reproduces') and np.isfinite(r.get('r2_sd', np.nan))]
            if not got:
                break
            # Median, not max: one candidate with sd 1.04 would otherwise widen the band
            # for every other candidate in the family.
            sd = float(np.median([r['r2_sd'] for r in got]))
            # sqrt(2) because both values being compared are single draws.
            new_margin = float(args.k) * np.sqrt(2.0) * sd
            if new_margin <= margin + 1e-12:
                break
            margin = new_margin
            todo = [r for _, r in pool.iterrows()
                    if float(r['r2_single']) >= fam_best - margin
                    and (ds, fam, r['variant'], r['feature_tag'], r['subset_label']) not in done]
            if not todo:
                break
        if margin:
            print('       %-30s %-6s band +/-%.4f (k=%g, sd=%.4f) -> %d fit'
                  % (ds.replace('MC_', '')[:29], fam, margin, args.k, margin /
                     (float(args.k) * np.sqrt(2.0)),
                     sum(1 for r in out_rows if r['dataset'] == ds and r['family'] == fam)))

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
        # Median, not max: one candidate with sd 1.04 would otherwise widen the band
        # for the whole family and tell us nothing.
        sd = float(g['r2_sd'].median()) if g['r2_sd'].notna().any() else 0.0
        # How far below the band the nearest *unrefit* candidate sits, in seed standard
        # deviations. Under the adaptive band nothing within k sd can be excluded -- the
        # expansion is driven by the same sd -- so this reports the headroom rather than
        # hunting for a failure. A small number here means the band only just reached far
        # enough and k deserves raising.
        refit_keys = {(r['variant'], r['feature_tag'])
                      for _, r in df[df.dataset == ds].iterrows()}
        pool = cand[cand.dataset == ds]
        excluded = pool[[(r['variant'], r['feature_tag']) not in refit_keys
                         for _, r in pool.iterrows()]]
        if excluded.empty or sd <= 0:
            flag = 'all refit' if excluded.empty else 'sd 0 (nothing can move)'
        else:
            near = float(excluded['r2_single'].max())
            gap_sd = (float(pool['family_best'].max()) - near) / (sd * (2 ** 0.5))
            flag = 'nearest excluded %.1f sd below' % gap_sd
            if gap_sd < float(args.k):
                flag = 'EXTEND: excluded candidate only %.1f sd below (k=%g)' % (gap_sd, args.k)
        print('%-26s %10.4f %10.4f %-10s %s'
              % (ds.replace('MC_', '').replace('_diff', '')[:25], rep_best, new_best,
                 new_fam[:10], flag))
    print()
    print('[INFO] The refit band is k*sqrt(2)*sd around each family best, with sd measured')
    print('       across seeds and the band widened until it stops growing. "EXTEND" would')
    print('       mean a candidate outside the band sits closer than k standard deviations')
    print('       to it, which the expansion is meant to prevent; raise --k if it appears.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

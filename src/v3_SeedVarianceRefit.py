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
the seed noise of the current best, which is a small fraction of the pool.

An earlier version of this note claimed Gaussian processes were deterministic, on a
measured seed standard deviation of exactly 0.0000. That measurement was real and the
conclusion was wrong: it varied ``random_state``, which a Gaussian process does not read.
Its randomness is the Monte Carlo draw of the uncertain-input kernel, seeded by
``uncertain_kernel_mc_seed``, and three seeds of pH's winner span 0.012 -- against a 0.018
gap to the next target in the accuracy ordering. Only MLR is deterministic.

Measured seed standard deviations, for scale:

    XGBoost      0.03 median, up to 0.44
    GP           0.012 (pH), 0.002 (Turbidity)
    Transformer  0.42 at 12 held-out measurements, 0.019 at 47

The transformer's spread is dominated by how few measurements the target has rather than by
the model, and it is the reason no family is exempt on the grounds that its effect looks
small: that is not knowable before measuring it.

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
# How far the previously reported score may sit from the base-seed refit before the
# refit is treated as misconfigured rather than merely re-seeded. Absolute floor for a
# deterministic family; otherwise this many measured seed standard deviations.
REPRO_TOL_ABS = 0.002
REPRO_TOL_SD = 4.0
# A flattened 11-predictor GP on Total coliforms ran 25 minutes on 0.016 s of CPU -- hung,
# not slow -- and stalled the whole recovery because subprocess.run waits forever by
# default. A fit that has not finished in this long is not going to.
FIT_TIMEOUT_S = 1800
# Family name -> the `model` value it carries in feature_sweep_final_metrics.csv.
FAMILY_KEY = {'xgb': 'xgb_regressor', 'transformer': 'transformer',
              'gp': 'gp_regressor'}
# Per-fit cost, measured from consecutive prediction-file timestamps in the sweep.
FIT_SECONDS = {'xgb': 5.1, 'transformer': 27.2, 'gp': 6.6}
# The hyperparameter that actually carries the seed, per family. A Gaussian process
# ignores `random_state` completely -- two fits at 7 and 99 give bit-identical
# predictions -- because its randomness is the Monte Carlo draw of the uncertain-input
# kernel, seeded by `uncertain_kernel_mc_seed`. Writing `random_state` for every family,
# as this script used to, measured a GP's seed spread as exactly zero by varying the one
# number it does not read.
SEED_FIELD = {'xgb_regressor': 'random_state', 'xgb_classifier': 'random_state',
              'transformer': 'random_state',
              'gp_regressor': 'uncertain_kernel_mc_seed'}


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

    Candidates that re-seeding cannot move are dropped here rather than refitted and
    found to have zero spread: see `seed_changes_this_fit`. On the profiler-free
    predictor set that removes 227 of 375 Gaussian process candidates, whose subsets
    contain no predictor carrying an uncertainty distribution.

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
            # The family best is taken over every candidate, including any that re-seeding
            # cannot move: it is the score to beat, not a candidate for refitting.
            fam_best = float(g['r2'].max())
            for _, r in g.iterrows():
                var = str(r.get('variant', ''))
                tag = str(r.get('feature_tag', ''))
                sub = str(r.get('subset_label', ''))
                rd = run_dir_for(root, ds.name, var, tag, sub)
                if not seed_changes_this_fit(rd, FAMILY_KEY[fam]):
                    continue
                rows.append(dict(
                    dataset=ds.name, family=fam, variant=var,
                    feature_tag=tag, subset_label=sub,
                    r2_single=float(r['r2']), target_best=best, family_best=fam_best))
    return pd.DataFrame(rows)


def ds_path(root: Path, ds: str) -> Path:
    """The dataset directory for a dataset name."""
    return root / ds


def run_dir_for(root: Path, ds: str, variant: str, feature_tag: str,
                subset_label: str) -> "Path | None":
    """The run directory a metrics row describes, or None if it is not unique."""
    sweeps = root / ds / 'forecasts' / 'feature_sweeps'
    if not sweeps.is_dir():
        return None
    hits = [p for p in sweeps.iterdir()
            if p.is_dir() and p.name.startswith(variant) and feature_tag in p.name
            and p.name.endswith('_' + subset_label)]
    return hits[0] if len(hits) == 1 else None


def seed_changes_this_fit(run_dir: "Path | None", model_type: str) -> bool:
    """Whether re-seeding this particular fit can change its predictions.

    Family membership is not enough for a Gaussian process. Its randomness enters only
    through the perturbation applied to predictors that carry an uncertainty
    distribution, so on a subset containing none of them every Monte Carlo draw is the
    zero vector and the kernel reduces exactly to the plain Matern -- verified to
    8.3e-17. Of 375 GP candidates on the profiler-free predictor set, 227 are in that
    position and refitting them would spend six fits each to reproduce one number.

    The test is per candidate rather than per family, and it reads the fitted artifact
    rather than intersecting the subset with `UNCERTAINTY_DISTRIBUTION_FEATURES`: that
    constant lists only the six profiler channels, yet `SCADA - pH` carries a variance
    of 0.0433 and is not in it, so the constant is not a reliable statement of which
    predictors are uncertain.
    """
    mt = str(model_type or '').lower()
    if mt in ('xgb_regressor', 'xgb_classifier', 'transformer'):
        return True
    if mt != 'gp_regressor' or run_dir is None:
        return False
    art = run_dir / 'gp_model.pt'
    if not art.exists():
        return False
    try:
        import torch
        payload = torch.load(art, map_location='cpu', weights_only=False)
        var = np.asarray(payload.get('input_uncertainty_var'))
    except Exception:
        return False
    return bool(var is not None and var.size and (var != 0).any())


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
    """Config for one seed, with whatever the reported fit re-derives pinned in place.

    Each family needs a different thing held still, and a different field set as the
    seed. XGBoost re-derives its round budget and reads `random_state`; a Gaussian
    process re-derives its epoch budget and reads `uncertain_kernel_mc_seed`, ignoring
    `random_state` entirely. The seed field is looked up in `SEED_FIELD` rather than
    assumed.

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
    model_type = str(cfg.get('model_type', '')).strip().lower()

    if model_type in ('xgb_regressor', 'xgb_classifier'):
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
            # Also silence the CV round-budget estimator. With the cache off it runs
            # again and re-derives a budget that overrides the pin -- for xgb_01 configs
            # it landed somewhere else and 13 candidates then failed to reproduce their
            # reported score. The reported fit trained exactly `rounds` rounds, so
            # training exactly that many is what reproduces it; the estimator only has a
            # job when the budget is unknown.
            hyper['early_stopping_rounds'] = None
        cv = dict(hyper.get('cv_tuning') or {})
        cv['enabled'] = False
        hyper['cv_tuning'] = cv

    # A Gaussian process needs nothing pinned, and pinning its epoch budget actively
    # breaks reproduction. That was not obvious and is worth recording, because the
    # analogy with XGBoost is so close: the budget is CV-derived at fit time, varies from
    # 1 to 250 across fits, and `_gp_cv_estimate_epochs` even takes `mc_seed` as an
    # argument. Pinning it looked necessary.
    #
    # Measured on a pH candidate whose reported fit trained 64 epochs:
    #
    #     reported                      +0.491655   cv_epoch_budget_exhausted
    #     num_epochs pinned to 64       +0.521807   max_epochs_exhausted
    #     budget left to be re-derived  +0.491655   cv_epoch_budget_exhausted
    #
    # Both trained 64 epochs, so the count was never the problem. `_gp_cv_estimate_epochs`
    # trains a GP on each fold and so advances the global torch generator before the real
    # fit begins; skipping it leaves the generator in a different state and the fit lands
    # somewhere else. Re-deriving the budget is part of what reproduces the fit, not a
    # confound to remove -- and unlike XGBoost's cached rounds, it is derived from the same
    # training data by the same procedure every time, so at seed 0 it comes back identical.
    # Verified on two candidates: +0.534806 against a reported +0.534800, and the exact
    # match above.

    field = SEED_FIELD.get(model_type)
    if field is None:
        raise ValueError('no seed field known for model_type %r' % model_type)
    hyper[field] = int(seed)
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
    # Is the previously reported score consistent with this refit distribution?
    #
    # This used to demand that the base seed reproduce the reported value to within
    # 0.002, which is the right test only when the reported fit was itself seeded. It is
    # not, for anything fitted before `_seed_model_rng` existed: a transformer's weights
    # then came from process-start entropy, so no seed reproduces it and every candidate
    # failed a check it could not pass. Reproducing a stale run is not the goal anyway --
    # a defensible forward methodology is.
    #
    # What the check is actually for is catching a refit that is *misconfigured* rather
    # than merely re-seeded: v3's first version left the XGBoost tuning cache enabled,
    # which replaced the hyperparameters wholesale and refitted Cadmium's winner to
    # -0.805 against a reported +0.428. That is many standard deviations away, whereas a
    # differently-seeded fit of the same configuration is a draw from the same
    # distribution and should land within a few.
    #
    # So the tolerance is the measured seed spread, floored at the old absolute value so
    # a deterministic family is still held to it exactly.
    s0 = vals[0] if vals else float('nan')
    sd = rec['r2_sd']
    tol = REPRO_TOL_ABS
    if np.isfinite(sd) and sd > 0:
        tol = max(REPRO_TOL_ABS, REPRO_TOL_SD * float(sd))
    gap = abs(s0 - c['r2_single']) if np.isfinite(s0) else float('inf')
    rec['r2_seed0'] = float(s0)
    rec['repro_gap'] = float(gap)
    rec['repro_tol'] = float(tol)
    rec['repro_gap_sd'] = float(gap / sd) if (np.isfinite(sd) and sd > 0) else float('nan')
    rec['repro_exact'] = bool(np.isfinite(s0) and gap < REPRO_TOL_ABS)
    rec['reproduces'] = bool(np.isfinite(s0) and gap <= tol)
    note = 'OK' if rec['reproduces'] else '<-- INCONSISTENT'
    if rec['reproduces'] and not rec['repro_exact']:
        note = 'ok (%.1f sd)' % rec['repro_gap_sd']
    print('          seed0 %+.4f vs reported %+.4f %-16s | mean %+.4f  sd %.4f'
          % (s0, c['r2_single'], note, rec['r2_mean'], rec['r2_sd']))
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
    ap.add_argument('--families', default='xgb,gp,transformer',
                    help='Comma-separated: xgb, gp, transformer. Every family whose fit a '
                         'seed can move is included by default. A Gaussian process is not '
                         'exempt -- it ignores random_state but its uncertain-input kernel '
                         'draws Monte Carlo samples, and three seeds of pH\'s winner span '
                         '0.012 against a 0.018 gap to the next target. Candidates a seed '
                         'cannot move are dropped per candidate, not per family, so this '
                         'costs nothing where it would achieve nothing. MLR is deterministic '
                         'and has no entry.')
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
            print('[WARN] %d candidate(s) sit further from their previously reported score '
                  'than' % (len(df) - ok))
            print('       %.0f measured seed standard deviations, which is more than '
                  're-seeding the same' % REPRO_TOL_SD)
            print('       configuration explains. Those are excluded from the re-selection: '
                  'check the')
            print('       staged config against the run it replaces before trusting the '
                  'targets involved.')
        n_inexact = int(len(df) - df.get('repro_exact', pd.Series([True] * len(df))).sum())
        if n_inexact:
            print('[INFO] %d candidate(s) are consistent with their reported score but do not '
                  'match it' % n_inexact)
            print('       exactly. That is expected wherever the reported fit predates '
                  'seeding: an')
            print('       unseeded transformer cannot be reproduced by any seed, and the '
                  'ensemble is')
            print('       the defensible value rather than the draw it replaces.')

    # --- re-selection and the margin check -----------------------------------------
    print()
    print('%-26s %10s %10s %-18s %s' % ('target', 'reported', 'seed-avg', 'new best', 'margin check'))
    print('-' * 82)
    for ds, g in df.groupby('dataset'):
        rep_best = float(z.loc[ds, 'best_r2'])
        rep_fam = str(z.loc[ds, 'best_family'])
        g = g[g['reproduces']] if 'reproduces' in g.columns else g
        # Post-refit best, taken candidate by candidate rather than family by family.
        #
        # An earlier version of this block excluded a whole family from the single-seed
        # pool as soon as any one of its candidates had been refitted. Under the adaptive
        # band that is almost always wrong: the band admits only the few candidates a
        # seed could reorder, so a family's actual best is usually *not* among them. On pH
        # it discarded the Gaussian process winner at +0.5348 because one unrelated GP
        # candidate at -7.95 had been refitted, and handed the target to MLR at +0.5011.
        #
        # A refitted candidate contributes its seed mean; every other candidate keeps its
        # single-seed score. Both are read per candidate from the sweep's own metrics
        # table, which lists all of them -- `cand` holds only the ones a seed can move.
        refit_keys = {(str(r['family']), str(r['variant']), str(r['feature_tag']))
                      for _, r in g.iterrows()}
        per_family = {}
        fm = ds_path(root, ds) / 'forecasts' / 'feature_sweeps' / \
            'feature_sweep_final_metrics.csv'
        if fm.is_file():
            allrows = pd.read_csv(fm, encoding='utf-8', encoding_errors='replace')
            allrows['r2'] = pd.to_numeric(allrows['r2'], errors='coerce')
            for fam, key in FAMILY_KEY.items():
                sub = allrows[allrows['model'].astype(str) == key].dropna(subset=['r2'])
                for _, r in sub.iterrows():
                    k = (fam, str(r.get('variant', '')), str(r.get('feature_tag', '')))
                    if k in refit_keys:
                        continue
                    v = float(r['r2'])
                    if fam not in per_family or v > per_family[fam][0]:
                        per_family[fam] = (v, 'single seed')
        # MLR is deterministic and has no refit path, so its reported value stands.
        if 'mlr_r2' in z.columns and pd.notna(z.loc[ds, 'mlr_r2']):
            v = float(z.loc[ds, 'mlr_r2'])
            if 'mlr' not in per_family or v > per_family['mlr'][0]:
                per_family['mlr'] = (v, 'single seed')
        for fam, gg in (g.groupby('family') if len(g) else []):
            if not gg['r2_mean'].notna().any():
                continue
            v = float(gg['r2_mean'].max())
            if fam not in per_family or v > per_family[fam][0]:
                per_family[fam] = (v, '%d-seed' % int(args.seeds))
        if per_family:
            fam_win = max(per_family, key=lambda k: per_family[k][0])
            new_best, how = per_family[fam_win]
            new_fam = '%s (%s)' % (fam_win, how)
        else:
            new_best, new_fam = float('nan'), '-'
        # Headroom, computed per family. The band is a family-level quantity and the
        # spreads differ by an order of magnitude between families -- on this target the
        # median seed sd is 0.009 for the Gaussian process, 0.19 for the transformer and
        # 0.20 for XGBoost -- so a per-target median mixed across families compares an
        # excluded candidate of one family against another family's noise scale and
        # reports nonsense. An earlier version did exactly that and raised EXTEND on pH
        # by measuring a GP candidate against a transformer's spread.
        worst = None
        for fam, gg in df[df.dataset == ds].groupby('family'):
            fsd = float(gg['r2_sd'].median()) if gg['r2_sd'].notna().any() else 0.0
            if fsd <= 0:
                continue
            # From every candidate actually refitted, not just those that passed the
            # consistency gate. Taking the keys from the gate-filtered frame makes a
            # candidate that was refitted and then failed the gate look as though the
            # band never reached it, which raised a spurious EXTEND on pH: all 18
            # transformer candidates were inside the +/-2.71 band, but the 5 that failed
            # the gate were counted as excluded. "Band too narrow" and "refit but
            # inconsistent" are different problems and are reported separately.
            keys = {(r['variant'], r['feature_tag'])
                    for _, r in df[(df.dataset == ds) & (df.family == fam)].iterrows()}
            pool = cand[(cand.dataset == ds) & (cand.family == fam)]
            if pool.empty:
                continue
            excl = pool[[(r['variant'], r['feature_tag']) not in keys
                         for _, r in pool.iterrows()]]
            if excl.empty:
                continue
            gap_sd = ((float(pool['family_best'].max()) - float(excl['r2_single'].max()))
                      / (fsd * (2 ** 0.5)))
            if worst is None or gap_sd < worst[0]:
                worst = (gap_sd, fam)
        if worst is None:
            flag = 'all refit'
        elif worst[0] < float(args.k):
            flag = 'EXTEND: %s candidate only %.1f sd below (k=%g)' % (worst[1], worst[0], args.k)
        else:
            flag = 'nearest excluded %.1f sd below (%s)' % (worst[0], worst[1])
        print('%-26s %10.4f %10.4f %-18s %s'
              % (ds.replace('MC_', '').replace('_diff', '')[:25], rep_best, new_best,
                 new_fam[:18], flag))
    print()
    print('[INFO] The refit band is k*sqrt(2)*sd around each family best, with sd measured')
    print('       across seeds and the band widened until it stops growing. "EXTEND" would')
    print('       mean a candidate outside the band sits closer than k standard deviations')
    print('       to it, which the expansion is meant to prevent; raise --k if it appears.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""
Horizon sweep script: for each dataset, take the model the results table reports as
best for that target, then for each horizon N call d_RunResample.split(gap_rows=N) to
generate a fresh set of samples where the predictor window ends N hours before the
target, retrain and evaluate, and save metrics/plots to horizons/lookahead_sweeps/.

Selection, the evaluation set and the metric all come from
``<root>/summaries/common_set_metrics.csv`` and ``common_set_segments.csv``:

  * The model is the one named by ``best_family`` and its ``<family>_run`` column --
    a specific run, so the GP variant, the window length and the feature subset are
    all the ones that were selected, not a per-family template that merely resembles
    them. Only that one model is swept, whether it is a machine-learning family or a
    statistical one.
  * The train/test split is reused from that run rather than recomputed, so a window
    that becomes invalid at a longer horizon cannot move the split boundary.
  * Metrics are computed on the target's common evaluation set with z8's own metric
    function, so horizon 0 reproduces the results table exactly.

Reference forecasts are not computed: the horizon figures report accuracy, not skill.

Layout written per horizon:

    <dataset_dir>/horizons/NNNhr/<ml|mlr>/
        samples/                    – raw sample CSVs (gap_rows=N), shared across replicates
        mc_replicates/              – uncertainty-perturbed replicates, shared across replicates
        config_rep_NNN.yml          – the winning run's config, retargeted at this replicate
        forecasts/
            rep_000/
                evaluation_summary.csv   – model test row only
                predictions.csv          – what the common-set metrics are computed from
                train_files.txt / test_files.txt  – the reused split, copied in
                model.json / model.pt    – model artifact
            rep_001/
                ...

Only one of ml/ and mlr/ is populated: whichever class the target's reported model
belongs to.

With --replicates M each horizon trains M times; variation comes from per-replicate
training seeds, offset from the winning run's own seed so that replicate 0 is that
run's configuration.  Deterministic model types (MLR) are collapsed to 1 replicate.

Aggregate metrics (all replicates, all horizons) are written to:

    <dataset_dir>/horizons/lookahead_sweeps/
        lookahead_metrics.csv           – one row per (horizon, replicate)
        rmse_vs_lookahead.png
        r2_vs_lookahead.png

``r2``/``rmse``/``nrmse`` in that file are computed on the common evaluation set.
``r2_own_split`` is the run's own test split, carried for audit only: it is a
different evaluation set and is not what the paper reports.

Use z2_HorizonPostProcess.py to produce cross-dataset comparison figures.

CLI arguments:
    --data-root PATH        Root directory containing MC_* dataset subdirectories.
                            Default: data/output/regression
    --dataset-prefix STR    Only process datasets whose name starts with this prefix.
                            Default: MC
    --resample-config PATH  Path to the d_RunResample YAML config that was used to
                            generate the original samples.  Provides input_csv, column
                            lists, and Monte Carlo settings.  Required.
    --horizons INT [INT …]  Space-separated list of horizon values (hours) to sweep.
                            Default: 0 1 2 6 12 24 48 96 120 167
    --replicates M          Number of times to train and evaluate each horizon.
                            Default: 1

Examples:
python src/k_RunHorizonSweep.py --resample-config data/output/sampling/resample_config.yml
python src/k_RunHorizonSweep.py --data-root data/output/CV14 --dataset-prefix MC --resample-config data/output/sampling/resample_config.yml --horizons 0 6 12 24 48 96 168 336 --replicates 7

"""

import sys
import copy
import json
import shutil
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import subprocess

from d_RunResample import (
    split as resample_split,
    _normalize_once,
    _load_and_prepare_sensor_uncertainties,
)
from utils.config_utils import load_config
# The horizon sweep scores the segments z8 selected, with z8's metric function, so
# that horizon 0 reproduces the results table rather than resembling it.
import z8_CommonSetMetrics as z8

PREFERRED_LOOKAHEADS = [0, 1, 2, 6, 12, 24, 48, 96, 120, 167]

_MLR_MODEL_NAMES = {'mlr', 'mlr_avg12', 'mlr_avgall'}

# Deterministic models produce identical predictions for identical inputs — no benefit
# to multiple replicates.  The sweep automatically collapses to 1 replicate for these.
_DETERMINISTIC_MODEL_KEYS = _MLR_MODEL_NAMES

_MODEL_TYPE_TO_KEY = {
    'xgb_regressor': 'xgb',
    'xgb_classifier': 'xgb',
    'transformer': 'transformer',
    'gp_regressor': 'gp',
    # display names from feature_sweep_final_metrics.csv
    'XGBRegressor': 'xgb',
    'Transformer': 'transformer',
    'GPRegressor': 'gp',
    # MLR variants
    'mlr':        'mlr',
    'mlr_avg12':  'mlr_avg12',
    'mlr_avgall': 'mlr_avgall',
}

# XGB cross-validation tuning is cached once per dataset, and that cache -- not the
# config -- is what records the hyperparameters the reported model was fitted with.
# e_Train resolves it as <data_dir>/forecasts/<this name>.
_CV_TUNING_CACHE = 'xgb_cv_tuning_cache.json'

# Per-replicate seed field injected into training config for each model type.
# These map model_key → hyperparameter key that controls randomness.
_MODEL_KEY_TO_SEED_FIELD = {
    'xgb': 'random_state',
    'gp':  'uncertain_kernel_mc_seed',
    'transformer': 'random_state',
}


def _base_window_rows_from_config(config_path: Path) -> int:
    """Return base predictor window length from a training/eval config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get('data', {})
    base_start = int(data_cfg['input_row_1'])
    base_stop = int(data_cfg['input_row_2'])
    return int(base_stop - base_start + 1)  # indices are inclusive: rows input_row_1..input_row_2


def _build_lookahead_schedule(preferred: list | None = None) -> list[int]:
    """Return the lookahead schedule as non-negative integers, sorted and deduplicated."""
    candidates = preferred if preferred is not None else PREFERRED_LOOKAHEADS
    return sorted(set(int(v) for v in candidates if int(v) >= 0))


def _read_common_set(root: Path):
    """Table 3's per-target winner and the exact segments it was scored on.

    Both come from the summaries z8 writes. Recomputing either here would let the
    horizon figures and the results table drift apart, which is the failure this
    whole selection path exists to prevent.
    """
    summaries = root / 'summaries'
    metrics = summaries / 'common_set_metrics.csv'
    segments = summaries / z8.SEGMENTS_NAME
    for f in (metrics, segments):
        if not f.exists():
            raise SystemExit(
                "[FATAL] %s not found. Run:\n"
                "    python src/z8_CommonSetMetrics.py --root %s\n"
                "The horizon sweep takes its model choice and its evaluation set from it."
                % (f, root))
    df = pd.read_csv(metrics, encoding='utf-8', encoding_errors='replace')
    seg = pd.read_csv(segments, encoding='utf-8', encoding_errors='replace')
    by_ds = {k: list(g.sort_values('order')['sample_file'])
             for k, g in seg.groupby('dataset')}
    return df, by_ds


# Display family in common_set_metrics.csv -> the column prefix carrying that
# family's winning run, and (for MLR) the model_type the horizon run must fit.
_FAMILY_COLUMN = {
    'GP': 'gp', 'XGB': 'xgb', 'Transformer': 'transformer',
    'MLR': 'mlr', 'MLR-12': 'mlr12', 'MLR-All': 'mlrall',
}
_FAMILY_MODEL_KEY = {
    'MLR': 'mlr', 'MLR-12': 'mlr_avg12', 'MLR-All': 'mlr_avgall',
}


def _config_for_run(dataset_dir: Path, run: str):
    """Return (config_path, run_dir) for the run named *run*.

    Non-MLR runs have a training config under ``configs/`` named after the run; MLR
    runs have only an evaluation config inside the run directory. The run name is the
    whole identity -- ``gp_04_r671_f5_21eb03415a_k03`` names the GP variant, the
    window length and the feature subset -- so resolving by name, rather than by
    family and a template, is what keeps the horizon model identical to the reported
    one. Resolving by family is what silently substituted gp_01 for gp_04.
    """
    for base in ('feature_sweeps', 'Shapley_sweeps'):
        sweep = dataset_dir / 'forecasts' / base
        if not sweep.is_dir():
            continue
        run_dir = sweep / run
        train_cfg = sweep / 'configs' / ('config_%s.yml' % run)
        if train_cfg.exists():
            return train_cfg, run_dir
        eval_cfg = run_dir / ('config_evaluate_%s.yml' % run)
        if eval_cfg.exists():
            return eval_cfg, run_dir
    raise SystemExit("[FATAL] %s: no config on disk for winning run %r."
                     % (dataset_dir.name, run))


def _select_from_common_set(data_root, dataset_prefix):
    """One selection per target: the model Table 3 reports, and nothing else.

    The winner may be a machine-learning family or a statistical one; whichever it
    is, it is the only model swept. The horizon question is how that model's accuracy
    decays, not which family wins again at each horizon.
    """
    root = Path(data_root)
    df, seg_by_ds = _read_common_set(root)
    selected = []
    for _, r in df.iterrows():
        ds_name = str(r['dataset'])
        if not ds_name.startswith(dataset_prefix):
            continue
        dataset_dir = root / ds_name
        if not dataset_dir.is_dir():
            raise SystemExit("[FATAL] %s is named in common_set_metrics.csv but is not on disk."
                             % dataset_dir)
        family = str(r['best_family'])
        col = _FAMILY_COLUMN.get(family)
        if col is None:
            raise SystemExit("[FATAL] %s: unrecognised best_family %r." % (ds_name, family))
        run = str(r.get(col + '_run', '') or '').strip()
        if not run or run.lower() == 'nan':
            raise SystemExit("[FATAL] %s: best_family is %s but column %s_run is empty."
                             % (ds_name, family, col))
        cfg_path, run_dir = _config_for_run(dataset_dir, run)
        model_key = _FAMILY_MODEL_KEY.get(family)
        if model_key is None:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                model_key = _model_name_to_key(str((yaml.safe_load(f) or {}).get('model_type', '')))
        segments = seg_by_ds.get(ds_name, [])
        if not segments:
            raise SystemExit("[FATAL] %s: no common-set segments recorded." % ds_name)
        sigma = float(r['sigma_record'])
        selected.append({
            'dataset_dir': dataset_dir,
            'config': cfg_path,
            'run_dir': run_dir,
            'run': run,
            'family': family,
            'model_key': model_key,
            'model_class': 'mlr' if model_key in _MLR_MODEL_NAMES else 'ml',
            'segments': segments,
            'sigma': sigma,
        })
    if not selected:
        raise SystemExit("[FATAL] No targets selected under %s." % root)
    return selected


def _migrate_flat_horizon_layout(horizon_dir: Path) -> 'str | None':
    """Move old flat layout (forecasts/ directly under NNNhr/) into ml/ or mlr/ subdirectory.

    Detection: horizon_dir/forecasts/ exists and neither horizon_dir/ml/ nor
    horizon_dir/mlr/ exists.

    Class detection: reads horizon_dir/forecasts/rep_000/model_config.json for
    'model_type' key.  MLR models write this file; if absent or model_type is not
    in _MLR_MODEL_NAMES, class is 'ml'.

    Returns the detected class name ('ml' or 'mlr'), or None if already migrated
    or no flat layout found.  Idempotent.
    """
    import shutil

    forecasts_dir = horizon_dir / 'forecasts'
    ml_dir = horizon_dir / 'ml'
    mlr_dir = horizon_dir / 'mlr'

    if not forecasts_dir.exists():
        return None
    if ml_dir.exists() or mlr_dir.exists():
        # Already migrated — idempotent
        return None

    # Determine class from model_config.json
    model_cfg_path = forecasts_dir / 'rep_000' / 'model_config.json'
    model_class = 'ml'
    if model_cfg_path.exists():
        try:
            with open(model_cfg_path, 'r', encoding='utf-8') as f:
                mcfg = json.load(f)
            mt = str(mcfg.get('model_type', '')).strip().lower()
            if mt in _MLR_MODEL_NAMES:
                model_class = 'mlr'
        except Exception:
            pass

    class_dir = horizon_dir / model_class
    class_dir.mkdir(parents=True, exist_ok=True)

    # Move flat-layout items into class_dir
    for item_name in ('forecasts', 'samples', 'mc_replicates', 'config.yml', 'baseline_summary.csv'):
        src = horizon_dir / item_name
        if src.exists():
            dst = class_dir / item_name
            shutil.move(str(src), str(dst))
            print(f"  [MIGRATE] {horizon_dir.name}/{item_name} -> {model_class}/{item_name}")

    return model_class


def _class_sweep_complete(class_dir: Path, n_reps: int) -> bool:
    """Return True if all replicates have been evaluated for this model class.

    Checks for existence of class_dir/forecasts/rep_{n_reps-1:03d}/evaluation_summary.csv.
    n_reps=1 is the common case for deterministic models (MLR).
    """
    last_rep = f'rep_{n_reps - 1:03d}'
    return (class_dir / 'forecasts' / last_rep / 'evaluation_summary.csv').exists()


def _load_normalization_params(dataset_dir: Path) -> dict | None:
    """Load normalization params from dataset dir or global sensors path."""
    candidates = [
        dataset_dir / 'normalization.json',
        Path(__file__).resolve().parent.parent / 'data' / 'output' / 'sensors' / 'normalization.json',
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, 'r') as f:
                return json.load(f)
    return None


def _apply_normalization(df: pd.DataFrame, normalization_params: dict) -> pd.DataFrame:
    """Apply stored min/max normalization params to a DataFrame."""
    df = df.copy()
    for col, params in normalization_params.items():
        if col not in df.columns:
            continue
        col_min = params['min']
        col_max = params['max']
        if col_max == col_min:
            df[col] = 0.5
        else:
            df[col] = (df[col] - col_min) / (col_max - col_min)
    return df


def _model_name_to_key(model_name: str) -> str:
    return _MODEL_TYPE_TO_KEY.get(model_name, 'xgb')


def _build_horizon_config(winning_config: Path, class_dir: Path, rep_dir: Path,
                          rep_name: str, rep_idx: int, split_source: Path) -> dict:
    """The winning configuration, retargeted at one horizon replicate.

    Everything that defines the model -- model_type, input_columns, input_aggregation,
    the window bounds and every hyperparameter -- is carried over untouched. Only the
    paths, the seed and the split source change. Rebuilding the config from a
    per-family template instead is what substituted a different Gaussian process
    variant for the one that was selected.
    """
    with open(winning_config, 'r', encoding='utf-8') as f:
        cfg = copy.deepcopy(yaml.safe_load(f))

    data = cfg.setdefault('data', {})
    data['data_dir'] = str(class_dir.resolve())
    data['forecast_dir'] = str(rep_dir.resolve())
    data['forecast_name'] = rep_name
    # sample_subdir is part of the model's definition, not a path to be normalised:
    # XGB and the Transformer train on the Monte Carlo replicates, the Gaussian
    # process on the collapsed segments, and the reused split lists names from
    # whichever of the two the winning run used.
    data.setdefault('sample_subdir', 'samples')

    # Pin the evaluation set. Recomputing the split at each horizon moves its
    # boundary whenever a shifted window turns a sample invalid, which would mix a
    # drifting evaluation set into what is meant to be a pure horizon effect.
    split = cfg.setdefault('data_split', {})
    split['reuse_split'] = True
    split['split_source'] = str(split_source.resolve())

    # Replicate seeds are offsets from the winning run's own seed, so replicate 0 is
    # that run's configuration rather than a neighbour of it. Where the winning run
    # left the seed unset, 0 is the base -- the replicates are then reproducible even
    # though replicate 0 cannot reproduce an unseeded fit.
    seed_field = _MODEL_KEY_TO_SEED_FIELD.get(
        _model_name_to_key(str(cfg.get('model_type', ''))))
    if seed_field:
        hyper = cfg.setdefault('hyperparameters', {})
        base_seed = hyper.get(seed_field)
        hyper[seed_field] = (0 if base_seed is None else int(base_seed)) + rep_idx

    # Reference forecasts are not reported per horizon, so do not compute them.
    cfg.setdefault('evaluation', {})['run_baselines'] = False
    return cfg


def _score_on_common_set(predictions_csv: Path, segments: list, sigma: float) -> dict:
    """Metrics for one replicate, restricted to the common evaluation set.

    Uses z8's own metric function and its own prediction-column rule, so the
    horizon-0 point is the same statistic as the results table rather than a
    re-implementation that could diverge from it.
    """
    t = pd.read_csv(predictions_csv, encoding='utf-8', encoding_errors='replace')
    t = t[t['kind'].astype(str) == 'test']
    col = z8._prediction_column(list(t.columns))
    if col is None:
        raise SystemExit("[FATAL] %s: no prediction column." % predictions_csv)
    grp = t.groupby('sample_file')
    preds = grp[col].mean()
    targets = grp['target'].mean()

    missing = [s for s in segments
               if s not in preds.index or not np.isfinite(preds[s])
               or s not in targets.index or not np.isfinite(targets[s])]
    if missing:
        raise SystemExit(
            "[FATAL] %s: %d of %d common-set segments are absent from the test "
            "predictions (%s%s).\n"
            "The evaluation set has to be identical at every horizon; scoring only the "
            "segments that survived would compare different sets and report the "
            "difference as forecast decay."
            % (predictions_csv, len(missing), len(segments), ', '.join(missing[:5]),
               ' ...' if len(missing) > 5 else ''))

    y = np.array([targets[s] for s in segments], dtype=float)
    pr = np.array([preds[s] for s in segments], dtype=float)
    m = z8._metrics(y, pr, sigma)
    m['n_common_scored'] = len(segments)
    return m


_MLR_AGG_MODE = {'mlr': 'last', 'mlr_avg12': 'avg12', 'mlr_avgall': 'avgall'}



def _run_mlr_horizon_rep(
    *,
    class_dir: Path,
    rep_name: str,
    base_config_path: Path,
    model_key: str,
    split_source: Path,
    subset_label: str,
) -> 'Path | None':
    """Fit and evaluate one MLR replicate.  Returns the rep output dir Path on success.

    Samples are read from class_dir/samples/ (per-class sample set).
    Output is written to class_dir/forecasts/rep_NNN/.

    *split_source* is the winning run's directory: its train/test file lists are
    reused verbatim so the horizon is evaluated on the same segments as the results
    table. Reference forecasts are not computed -- the horizon figures report
    accuracy only.
    """
    import h_RunMCFeatureSelectionSweep as _h

    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)
    data_cfg = base_cfg.get('data', {})

    input_columns = list(data_cfg['input_columns'])
    output_columns = list(data_cfg['output_columns'])
    input_row_1 = int(data_cfg['input_row_1'])
    input_row_2 = int(data_cfg['input_row_2'])
    output_rows = list(data_cfg['output_rows'])
    input_aggregation = str(data_cfg.get('input_aggregation', 'last'))

    sample_dir = class_dir / 'samples'
    if not sample_dir.exists():
        print(f"  [WARN] MLR: samples dir not found: {sample_dir}")
        return None

    from utils.training import splitter
    input_rows = slice(input_row_1, input_row_2)
    try:
        train_samples, test_samples = splitter(
            str(class_dir),
            rep_name,
            input_columns,
            input_rows,
            output_columns,
            output_rows,
            fault_tolerant=True,
            reuse_split=True,
            split_source=split_source,
            sample_subdir='samples',
            input_aggregation='none',   # MLR applies its own aggregation internally
            min_test_independent=5,
        )
    except Exception as exc:
        print(f"  [ERROR] MLR split failed for {rep_name}: {exc}")
        return None

    if len(train_samples) < 3 or len(test_samples) < 1:
        print(f"  [SKIP] MLR {rep_name}: insufficient samples "
              f"(train={len(train_samples)}, test={len(test_samples)})")
        return None

    aggregation_mode = _MLR_AGG_MODE.get(model_key, 'last')

    try:
        preds, targets, train_samples, test_samples, meta, _ = \
            _h._evaluate_mlr_with_rebalance(
                train_samples=train_samples,
                test_samples=test_samples,
                feature_names=input_columns,
                selection_config=None,
                aggregation_mode=aggregation_mode,
                min_test_independent=5,
                model_name=f'MLR {rep_name}',
                use_spearman_prefilter=True,
            )
    except Exception as exc:
        print(f"  [ERROR] MLR evaluation failed for {rep_name}: {exc}")
        return None

    forecasts_dir = class_dir / 'forecasts'
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    # Write model-only evaluation_summary.csv (no baseline rows — those go to
    # baseline_summary.csv at the class level via _write_horizon_baselines).
    import f_Evaluate as eval_module
    rep_dir = forecasts_dir / rep_name
    rep_dir.mkdir(parents=True, exist_ok=True)

    test_split_files = [str(s[2]) for s in test_samples]
    train_split_files = [str(s[2]) for s in train_samples]
    _model_label = model_key.upper().replace('_', '-')
    summary_row = eval_module._compute_regression_summary(
        f'{_model_label} (test)',
        preds,
        targets,
        len(test_samples),
        metadata={'kind': 'test', 'gp_uncertainty_mode': 'not_gp'},
        split_files=test_split_files,
    )
    summary_row['n_train_samples'] = len(train_samples)
    summary_row['n_test_samples'] = len(test_samples)
    summary_row['input_dim'] = len(input_columns)
    summary_row['target_dim'] = len(output_columns)
    summary_row['data_dir'] = str(class_dir)
    eval_module._write_summary_csv([summary_row], rep_dir / 'evaluation_summary.csv')

    # Generate diagnostic outputs matching those produced by f_Evaluate.py for ML models.
    # Uses the same utilities; no model loading required since we already have preds/targets.
    try:
        from utils.evaluation import visualizer as _visualizer
        from utils.evaluation import boxplot_from_error_rows as _boxplot_from_error_rows

        _pred_entry = {
            'label': f'{_model_label} (test)',
            'kind': 'test',
            'preds': preds,
            'targets': targets,
            'split_files': test_split_files,
            'include_mc_stats': False,
            'gp_var': None,
        }
        _pred_rows, _pred_cols = eval_module._build_predictions_table(
            [_pred_entry],
            gp_uncertainty_mode='not_gp',
            include_mc_output_columns=False,
        )
        eval_module._write_predictions_csv(
            _pred_rows, rep_dir / 'predictions.csv', _pred_cols,
        )

        _visualizer(
            (preds, targets),
            labels=[f'{_model_label} (test)'],
            forecast_name=rep_name,
            directory=str(class_dir),
            split_files_by_pair=[test_split_files],
        )

        _boxplot_rows = eval_module._build_boxplot_error_rows_from_predictions(
            _pred_rows,
            model_label=f'{_model_label} (test)',
            baseline_labels=[],
        )
        _boxplot_from_error_rows(
            _boxplot_rows,
            directory=str(class_dir),
            forecast_name=rep_name,
        )
    except Exception as _vis_exc:
        print(f"  [WARN] MLR diagnostic plot generation failed for {rep_name}: {_vis_exc}")

    # Write split files and model config so the rep dir is self-describing.
    (rep_dir / 'train_files.txt').write_text(
        '\n'.join(train_split_files) + ('\n' if train_split_files else ''), encoding='utf-8')
    (rep_dir / 'test_files.txt').write_text(
        '\n'.join(test_split_files) + ('\n' if test_split_files else ''), encoding='utf-8')

    model_config = {
        'model_type': model_key,
        'subset_label': subset_label,
        'input_columns': input_columns,
        'output_columns': output_columns,
        'input_row_1': input_row_1,
        'input_row_2': input_row_2,
        'output_rows': output_rows,
        'per_target_meta': [
            {
                'target_index': i,
                'target_name': output_columns[i] if i < len(output_columns) else f'target_{i}',
                'selected_features': m.get('selected_features', []),
                'n_selected': m.get('n_selected', 0),
                'coefficients': m.get('coefficients', []),
                'intercept': m.get('intercept', None),
                'n_train_valid': m.get('n_train', 0),
            }
            for i, m in enumerate(meta)
        ],
    }
    with open(rep_dir / 'model_config.json', 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, default=str)

    return rep_dir




def run_horizon_sweep(
    data_root,
    dataset_prefix,
    resample_config_path,
    preferred_lookaheads=None,
    n_replicates=1,
):
    resample_cfg = load_config(resample_config_path)
    config_dir = Path(resample_cfg['__config_dir'])

    input_csv = Path(resample_cfg['input_csv'])
    if not input_csv.is_absolute():
        input_csv = (config_dir / input_csv).resolve()

    use_uncertainty_perturbation = bool(resample_cfg.get('use_uncertainty_perturbation', True))
    n_mc_replicates = int(resample_cfg.get('n_mc_replicates', 10))
    random_seed = int(resample_cfg.get('random_seed', 1))
    verbose = bool(resample_cfg.get('verbose', False))

    print("[INFO] Loading data from %s" % input_csv)
    df_raw = pd.read_csv(input_csv, parse_dates=['TIMESTAMP'])
    df_raw = df_raw.sort_values('TIMESTAMP').reset_index(drop=True)

    selected = _select_from_common_set(data_root, dataset_prefix)
    print("[INFO] %d target(s) taken from the common evaluation set:" % len(selected))
    for s in selected:
        print("   %-38s %-12s %-40s %3d segments"
              % (s['dataset_dir'].name[:38], s['family'], s['run'][:40], len(s['segments'])))

    for sel in selected:
        dataset_dir = sel['dataset_dir']
        base_config = sel['config']
        model_key = sel['model_key']
        model_class = sel['model_class']
        segments = sel['segments']
        sigma = sel['sigma']
        is_mlr = model_key in _MLR_MODEL_NAMES

        print("\n[DATASET] %s  (%s, run %s)" % (dataset_dir.name, sel['family'], sel['run']))

        with open(base_config, 'r', encoding='utf-8') as f:
            base_cfg_yaml = yaml.safe_load(f)
        data_cfg = base_cfg_yaml.get('data', {})
        output_columns = list(data_cfg.get('output_columns', []))
        if not output_columns:
            raise SystemExit("[FATAL] %s: winning config has no output_columns." % dataset_dir.name)
        target = output_columns[0]
        predictor_cols = list(data_cfg.get('input_columns', []))
        class_sample_length = _base_window_rows_from_config(base_config)

        # --- Normalization: reuse existing params for consistency across horizons ---
        norm_params = _load_normalization_params(dataset_dir)
        if norm_params is not None:
            print("  [INFO] Reusing normalization params from %s" % dataset_dir.name)
            df_norm = _apply_normalization(df_raw, norm_params)
            normalization_params = norm_params
        else:
            to_normalize = list(dict.fromkeys(predictor_cols + output_columns))
            df_norm, normalization_params = _normalize_once(df_raw, to_normalize)

        shared_sensor_uncertainties = None
        if use_uncertainty_perturbation:
            try:
                shared_sensor_uncertainties = _load_and_prepare_sensor_uncertainties(
                    str(dataset_dir),
                    normalization_params=normalization_params,
                    verbose=verbose,
                )
            except Exception as exc:
                print("  [WARN] Could not load sensor uncertainties: %s. Disabling perturbation." % exc)
                use_uncertainty_perturbation = False

        lookaheads = _build_lookahead_schedule(preferred=preferred_lookaheads)
        if not lookaheads:
            raise SystemExit("[FATAL] No horizons to sweep.")
        effective_replicates = 1 if model_key in _DETERMINISTIC_MODEL_KEYS else n_replicates
        print("  [INFO] window=%d rows; horizons=%s; replicates=%d; scoring on %d common segments"
              % (class_sample_length, lookaheads, effective_replicates, len(segments)))

        metrics_rows = []
        for horizon in lookaheads:
            horizon_dir = dataset_dir / 'horizons' / ('%03dhr' % horizon)
            horizon_dir.mkdir(parents=True, exist_ok=True)
            _migrate_flat_horizon_layout(horizon_dir)
            class_dir = horizon_dir / model_class
            class_dir.mkdir(parents=True, exist_ok=True)

            # Hand the horizon the tuning the reported model was fitted with. Without
            # it e_Train re-tunes from scratch at every horizon, so each horizon gets
            # a different model -- for Cadmium that was n_estimators 165 against 249
            # and a learning rate 3.5x lower, and R^2 at horizon 0 of -0.31 where the
            # results table reports +0.33. Copied rather than pointed at, so a rerun
            # cannot write back into the sweep's cache.
            src_cache = sel['run_dir'].parent / _CV_TUNING_CACHE
            if src_cache.exists():
                dst_cache = class_dir / 'forecasts' / _CV_TUNING_CACHE
                dst_cache.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_cache, dst_cache)

            if not _class_sweep_complete(class_dir, effective_replicates):
                print("  [RESAMPLE] horizon %dhr (seed=%d)" % (horizon, random_seed))
                try:
                    result = resample_split(
                        df_norm,
                        str(class_dir),
                        [target],
                        class_sample_length,
                        nan_tol=0.8,
                        to_normalize=[],
                        fault_tolerant=True,
                        predictor_cols=predictor_cols,
                        use_uncertainty_perturbation=use_uncertainty_perturbation,
                        n_mc_replicates=n_mc_replicates,
                        random_seed=random_seed,
                        pre_normalized=True,
                        normalization_params=normalization_params,
                        sensor_uncertainties=shared_sensor_uncertainties,
                        verbose=False,
                        gap_rows=horizon,
                        training_config_defaults={},
                    )
                except Exception as exc:
                    raise SystemExit("[FATAL] Resampling failed for horizon %dhr: %s" % (horizon, exc))

                if result['n_samples'] == 0:
                    raise SystemExit("[FATAL] horizon %dhr produced no samples." % horizon)

                # The horizon config is built from the winning run, so every template
                # split() writes is unused and would only invite confusion about
                # which config the horizon was actually trained from. split() does
                # not report all of them, so clear by pattern rather than by list.
                for cfg_path in class_dir.glob('config_*.yml'):
                    if not cfg_path.name.startswith('config_rep_'):
                        cfg_path.unlink(missing_ok=True)

            for rep_idx in range(effective_replicates):
                rep_name = 'rep_%03d' % rep_idx
                rep_dir = class_dir / 'forecasts' / rep_name
                eval_csv = rep_dir / 'evaluation_summary.csv'
                preds_csv = rep_dir / 'predictions.csv'

                if eval_csv.exists() and preds_csv.exists():
                    print("  [SKIP] horizon %dhr %s - already evaluated" % (horizon, rep_name))
                else:
                    rep_dir.mkdir(parents=True, exist_ok=True)
                    if is_mlr:
                        print("  [MLR] horizon %dhr %s" % (horizon, rep_name))
                        got = _run_mlr_horizon_rep(
                            class_dir=class_dir,
                            rep_name=rep_name,
                            base_config_path=base_config,
                            model_key=model_key,
                            split_source=sel['run_dir'],
                            subset_label=sel['run'],
                        )
                        if got is None:
                            raise SystemExit(
                                "[FATAL] MLR evaluation failed for %s horizon %dhr %s."
                                % (dataset_dir.name, horizon, rep_name))
                    else:
                        cfg = _build_horizon_config(
                            base_config, class_dir, rep_dir, rep_name,
                            rep_idx, sel['run_dir'],
                        )
                        cfg_path = class_dir / ('config_%s.yml' % rep_name)
                        with open(cfg_path, 'w', encoding='utf-8') as f:
                            yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)

                        print("  [TRAIN] horizon %dhr %s" % (horizon, rep_name))
                        try:
                            subprocess.run(
                                [sys.executable, 'src/e_Train.py', '--config', str(cfg_path)],
                                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        except subprocess.CalledProcessError as exc:
                            raise SystemExit("[FATAL] Training failed for %s horizon %dhr %s:\n%s"
                                             % (dataset_dir.name, horizon, rep_name,
                                                exc.stderr.decode(errors='replace')))

                        eval_cfg_path = rep_dir / ('config_evaluate_%s.yml' % rep_name)
                        eval_arg = str(eval_cfg_path) if eval_cfg_path.exists() else str(cfg_path)
                        print("  [EVAL]  horizon %dhr %s" % (horizon, rep_name))
                        try:
                            subprocess.run(
                                [sys.executable, 'src/f_Evaluate.py', '--config', eval_arg],
                                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        except subprocess.CalledProcessError as exc:
                            raise SystemExit("[FATAL] Evaluation failed for %s horizon %dhr %s:\n%s"
                                             % (dataset_dir.name, horizon, rep_name,
                                                exc.stderr.decode(errors='replace')))

                if not preds_csv.exists():
                    raise SystemExit("[FATAL] %s not written." % preds_csv)

                row = {
                    'dataset': dataset_dir.name,
                    'target': target,
                    'family': sel['family'],
                    'run': sel['run'],
                    'model_class': model_class,
                    'model_name': model_key,
                    'horizon': horizon,
                    'replicate': rep_idx,
                    'input_rows_included': class_sample_length,
                    'input_rows_excluded': horizon,
                }
                row.update(_score_on_common_set(preds_csv, segments, sigma))

                # The run's own test split, kept for audit only: it is a different
                # evaluation set from the one the paper reports.
                if eval_csv.exists():
                    ev_df = pd.read_csv(eval_csv, encoding='utf-8', encoding_errors='replace')
                    if 'kind' in ev_df.columns:
                        ev_df = ev_df[ev_df['kind'].astype(str) == 'test']
                    if len(ev_df) and 'r2' in ev_df.columns:
                        row['r2_own_split'] = float(ev_df['r2'].iloc[0])
                        n_own = (ev_df['n_test_samples'].iloc[0]
                                 if 'n_test_samples' in ev_df.columns else None)
                        row['n_own_split'] = int(n_own) if pd.notna(n_own) else None
                metrics_rows.append(row)
                print("       R2(common,%d) = %+.3f" % (row['n_common_scored'], row['r2']))

        sweep_dir = dataset_dir / 'horizons' / 'lookahead_sweeps'
        sweep_dir.mkdir(parents=True, exist_ok=True)
        all_metrics = pd.DataFrame(metrics_rows)
        all_metrics.to_csv(sweep_dir / 'lookahead_metrics.csv', index=False)
        print("  [INFO] Wrote %s" % (sweep_dir / 'lookahead_metrics.csv'))

        import matplotlib.pyplot as plt

        for metric, ylabel, filename in [
            ('rmse', 'RMSE', 'rmse_vs_lookahead.png'),
            ('r2',   'R\u00b2', 'r2_vs_lookahead.png'),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            has_reps = all_metrics['replicate'].nunique() > 1
            if has_reps:
                for _, rep_df in all_metrics.groupby('replicate'):
                    ax.scatter(rep_df['horizon'], rep_df[metric],
                               s=18, alpha=0.35, color='steelblue', zorder=2)
            mean_df = all_metrics.groupby('horizon')[[metric]].mean().reset_index()
            ax.plot(mean_df['horizon'], mean_df[metric], marker='o', markersize=5,
                    linewidth=1.8, color='steelblue', zorder=3,
                    label='%s (%s)' % (sel['family'], model_key))
            ax.set_xlabel('Horizon (hours)')
            ax.set_ylabel(ylabel)
            ax.set_title('%s - %s vs horizon, common evaluation set (n=%d)'
                         % (dataset_dir.name, ylabel, len(segments)))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(sweep_dir / filename, dpi=150)
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Horizon sweep: resample with increasing gap_rows per horizon')
    parser.add_argument('--data-root', type=str, default='data/output/regression')
    parser.add_argument('--dataset-prefix', type=str, default='MC')
    parser.add_argument(
        '--resample-config',
        type=str,
        required=True,
        help='Path to the original d_RunResample YAML config (provides input_csv, columns, MC settings).',
    )
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        default=None,
        metavar='INT',
        help='Horizon values (hours) to sweep. Default: 0 1 2 6 12 24 48 96 120 167',
    )
    parser.add_argument(
        '--replicates',
        type=int,
        default=1,
        metavar='M',
        help='Number of times to train and evaluate each horizon (default: 1). '
             'Deterministic model types (MLR) are automatically collapsed to 1.',
    )
    args = parser.parse_args()
    run_horizon_sweep(
        args.data_root,
        args.dataset_prefix,
        args.resample_config,
        args.horizons,
        n_replicates=args.replicates,
    )

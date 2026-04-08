"""
Horizon sweep script: For each dataset, take the best-performing model/config (from
feature_sweeps), then for each horizon N call d_RunResample.split(gap_rows=N) to generate
a fresh set of samples where the predictor window ends N hours before the target, retrain
and evaluate the model, and save metrics/plots to horizons/lookahead_sweeps/.

Layout written per horizon:

    <dataset_dir>/horizons/NNNhr/
        samples/                    – raw sample CSVs (gap_rows=N), shared across replicates
        mc_replicates/              – uncertainty-perturbed replicates, shared across replicates
        config.yml                  – training config for the best model type only
        baseline_summary.csv        – Naive/Seasonal/Linear metrics, written once per horizon
        forecasts/
            rep_000/
                evaluation_summary.csv   – model test row only
                model.json / model.pt    – model artifact
                config_evaluate_rep_000.yml
            rep_001/
                ...

With --replicates M each horizon trains M times; variation comes from per-replicate
training seeds (random_seed + rep_idx injected into the training config), not from
re-resampling.  Deterministic model types (MLR) are automatically collapsed to 1 replicate.

Aggregate metrics (all replicates, all horizons) are written to:

    <dataset_dir>/horizons/lookahead_sweeps/
        lookahead_metrics.csv           – one row per (horizon, replicate)
        rmse_vs_lookahead.png
        r2_vs_lookahead.png

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
import json
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
from utils.config_utils import load_config, select_best_model_row

PREFERRED_LOOKAHEADS = [0, 1, 2, 6, 12, 24, 48, 96, 120, 167]

_BASELINE_MODEL_NAMES = {'naive', 'seasonal', 'linear'}

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

_MODEL_KEY_TO_CONFIG_NAME = {
    'xgb': 'config_xgb_01.yml',
    'transformer': 'config_transformer_01.yml',
    'gp': 'config_gp_01.yml',
}

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


def _resolve_config_for_row(dataset_dir: Path, best: pd.Series) -> 'tuple[Path, Path] | None':
    """Return (config_path, artifact_dir) for *best* row, or None if not resolvable.

    Works for both MLR and non-MLR model types.  The returned config_path is the
    training/eval config to use as the base config for the horizon sweep.
    """
    sweep_dir = dataset_dir / 'forecasts' / 'feature_sweeps'
    model_name = str(best['model'])
    _SHAP_PFX = 'shap_'

    if model_name in _MLR_MODEL_NAMES:
        subset_label = str(best.get('subset_label', 's01')).strip().lower()
        if subset_label.startswith(_SHAP_PFX):
            mlr_search_dir = dataset_dir / 'forecasts' / 'Shapley_sweeps'
            subset_label = subset_label[len(_SHAP_PFX):]
        else:
            mlr_search_dir = sweep_dir
        mlr_artifact_dir = mlr_search_dir / f'{model_name}_{subset_label}'
        config_path = mlr_artifact_dir / f'config_evaluate_{mlr_artifact_dir.name}.yml'
        if not config_path.exists():
            print(f"  [WARN] MLR eval config not found: {config_path}")
            return None
        return config_path, mlr_artifact_dir

    model_map = {
        'Transformer': 'transformer_01',
        'XGBRegressor': 'xgb_01',
        'GPRegressor': 'gp_01',
        'gp_regressor': 'gp_01',
        'xgb_regressor': 'xgb_01',
        'xgb_classifier': 'xgb_01',
        'transformer': 'transformer_01',
    }
    mapped_model_name = model_map.get(model_name, model_name.lower())

    subset_label_raw = str(best.get('subset_label', '')).strip().lower()
    if subset_label_raw.startswith(_SHAP_PFX):
        configs_dir = dataset_dir / 'forecasts' / 'Shapley_sweeps' / 'configs'
        sweep_namespace = dataset_dir / 'forecasts' / 'Shapley_sweeps'
    else:
        configs_dir = sweep_dir / 'configs'
        sweep_namespace = sweep_dir

    config_path = None
    row_count_raw = best['row_count']
    row_count_known = pd.notna(row_count_raw)
    if row_count_known:
        row_count_str = f"r{int(row_count_raw):03d}_"
        for cfg in configs_dir.glob(f"*{mapped_model_name}*{row_count_str}{best['feature_tag']}*.yml"):
            config_path = cfg
            break
        if config_path is None:
            for cfg in configs_dir.glob(f"*{row_count_str}{best['feature_tag']}*.yml"):
                config_path = cfg
                break
    if config_path is None:
        for cfg in configs_dir.glob(f"*{mapped_model_name}*{best['feature_tag']}*.yml"):
            config_path = cfg
            break
    if config_path is None:
        return None

    model_dir_name = model_map.get(model_name, model_name.lower())
    row_count_val = int(row_count_raw) if row_count_known else 0
    feature_tag = str(best['feature_tag'])
    subset_rank_val = int(best['subset_rank'])
    subset_rank_str = f"k{subset_rank_val:02d}"
    forecast_dir = sweep_namespace / f"{model_dir_name}_r{row_count_val}_{feature_tag}_{subset_rank_str}"
    return config_path, forecast_dir


def find_best_configs(data_root, dataset_prefix):
    """Return list of (dataset_dir, config_path, best_row, forecast_dir) for the single
    best model per dataset (any class).  Kept for backward compatibility."""
    best_configs = []
    for dataset_dir in sorted(Path(data_root).iterdir()):
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith(dataset_prefix):
            continue
        sweep_dir = dataset_dir / 'forecasts' / 'feature_sweeps'
        if not sweep_dir.exists():
            continue
        metrics_file = sweep_dir / 'feature_sweep_final_metrics.csv'
        if not metrics_file.exists():
            continue
        metrics_df = pd.read_csv(metrics_file)
        if metrics_df.empty or 'rmse' not in metrics_df.columns or 'r2' not in metrics_df.columns:
            continue
        is_baseline = metrics_df['model'].astype(str).str.strip().str.lower().isin(_BASELINE_MODEL_NAMES)
        non_baseline = metrics_df[~is_baseline].copy()
        if non_baseline.empty:
            continue
        best = select_best_model_row(non_baseline)
        result = _resolve_config_for_row(dataset_dir, best)
        if result is None:
            print(f"  [WARN] Could not resolve config for {dataset_dir.name}. Skipping.")
            continue
        config_path, forecast_dir = result
        best_configs.append((dataset_dir, config_path, best, forecast_dir))
    return best_configs


def find_best_configs_dual(data_root, dataset_prefix):
    """Return list of (dataset_dir, ml_entry, mlr_entry) for each dataset.

    Each entry is (config_path, best_row) or None if no valid model of that class exists.
    ml_entry  — best non-MLR model (GP, XGB, Transformer)
    mlr_entry — best MLR variant (mlr, mlr_avg12, mlr_avgall)
    """
    results = []
    for dataset_dir in sorted(Path(data_root).iterdir()):
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith(dataset_prefix):
            continue
        sweep_dir = dataset_dir / 'forecasts' / 'feature_sweeps'
        if not sweep_dir.exists():
            continue
        metrics_file = sweep_dir / 'feature_sweep_final_metrics.csv'
        if not metrics_file.exists():
            continue
        metrics_df = pd.read_csv(metrics_file)
        if metrics_df.empty or 'rmse' not in metrics_df.columns or 'r2' not in metrics_df.columns:
            continue

        is_baseline = metrics_df['model'].astype(str).str.strip().str.lower().isin(_BASELINE_MODEL_NAMES)
        is_mlr = metrics_df['model'].astype(str).str.strip().str.lower().isin(_MLR_MODEL_NAMES)

        ml_rows  = metrics_df[~is_baseline & ~is_mlr].copy()
        mlr_rows = metrics_df[is_mlr].copy()

        ml_entry = None
        if not ml_rows.empty:
            best_ml = select_best_model_row(ml_rows)
            result = _resolve_config_for_row(dataset_dir, best_ml)
            if result is not None:
                ml_entry = (result[0], best_ml)
            else:
                print(f"  [WARN] {dataset_dir.name}: could not resolve ML config for {best_ml['model']}")

        mlr_entry = None
        if not mlr_rows.empty:
            best_mlr = select_best_model_row(mlr_rows)
            result = _resolve_config_for_row(dataset_dir, best_mlr)
            if result is not None:
                mlr_entry = (result[0], best_mlr)
            else:
                print(f"  [WARN] {dataset_dir.name}: could not resolve MLR config for {best_mlr['model']}")

        if ml_entry is None and mlr_entry is None:
            print(f"  [WARN] {dataset_dir.name}: no valid ML or MLR config found. Skipping.")
            continue

        results.append((dataset_dir, ml_entry, mlr_entry))
    return results


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
            print(f"  [MIGRATE] {horizon_dir.name}/{item_name} → {model_class}/{item_name}")

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


def _extract_model_overrides(base_config_path: Path) -> dict:
    """Extract hyperparameters and data_split from the best model config as training_config_defaults."""
    cfg = load_config(str(base_config_path))
    model_type = cfg.get('model_type', '')
    key = _model_name_to_key(model_type)
    return {
        key: {
            'hyperparameters': dict(cfg.get('hyperparameters', {})),
            'data_split': dict(cfg.get('data_split', {})),
        }
    }


def _select_horizon_config(config_paths: list, model_key: str) -> Path | None:
    """Select the config file matching the best model type from split() output."""
    target_name = _MODEL_KEY_TO_CONFIG_NAME.get(model_key, 'config_xgb_01.yml')
    for p in config_paths:
        if Path(p).name == target_name:
            return Path(p)
    return Path(config_paths[0]) if config_paths else None


_MLR_AGG_MODE = {'mlr': 'last', 'mlr_avg12': 'avg12', 'mlr_avgall': 'avgall'}
_MLR_SUBSET_LABEL = {'mlr': 's01', 'mlr_avg12': 'm01', 'mlr_avgall': 'l01'}


def _find_eval_config_with_historic(dataset_dir: Path) -> 'Path | None':
    """Return the first eval config under *dataset_dir* that contains historic_path.

    Search order:
      1. Any config_evaluate_*.yml directly under forecasts/feature_sweeps/<subdir>/
      2. Any config_evaluate_*.yml anywhere under the dataset directory (fallback)

    This is used for MLR models, which never produce a per-horizon eval config but
    need historic_path to compute Naive/Seasonal/Linear baselines.
    """
    feature_sweeps = dataset_dir / 'forecasts' / 'feature_sweeps'
    candidates: list[Path] = []
    if feature_sweeps.is_dir():
        candidates.extend(sorted(feature_sweeps.rglob('config_evaluate_*.yml')))
    # Generic fallback
    candidates.extend(p for p in sorted(dataset_dir.rglob('config_evaluate_*.yml'))
                      if p not in candidates)
    for cfg_path in candidates:
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                doc = yaml.safe_load(f)
            if doc.get('evaluation', {}).get('historic_path'):
                return cfg_path
        except Exception:
            continue
    return None


def _run_mlr_horizon_rep(
    *,
    class_dir: Path,
    rep_name: str,
    base_config_path: Path,
    model_key: str,
) -> 'Path | None':
    """Fit and evaluate one MLR replicate.  Returns the rep output dir Path on success.

    Samples are read from class_dir/samples/ (per-class sample set).
    Output is written to class_dir/forecasts/rep_NNN/.
    Baseline rows are NOT written here — they are written once per horizon by
    _write_horizon_baselines().
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
            reuse_split=False,
            split_type='temporal',
            test_size=0.3,
            random_state=42,
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
    subset_label = _MLR_SUBSET_LABEL.get(model_key, 's01')

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


def _write_horizon_baselines(
    *,
    horizon_dir: Path,
    base_config_path: Path,
    eval_config_path: 'Path | None' = None,
    model_key: str,
    test_samples,
    test_split_files: list[str],
    output_dir: 'Path | None' = None,
) -> bool:
    """Compute Naive/Seasonal/Linear baselines and write baseline_summary.csv.

    Writes to output_dir/baseline_summary.csv when output_dir is given, otherwise
    falls back to horizon_dir/baseline_summary.csv (legacy behaviour).

    Uses data fields from base_config_path (output_columns, output_rows, sample_subdir)
    and evaluation fields (historic_path, window_hours, etc.) from eval_config_path when
    provided, otherwise falls back to base_config_path.  For non-MLR models, pass the
    rep_000 eval config as eval_config_path since it carries historic_path while the
    training config does not.

    Returns True on success, False if baselines could not be computed.
    """
    import h_RunMCFeatureSelectionSweep as _h
    import f_Evaluate as eval_module

    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)
    data_cfg = base_cfg.get('data', {})
    output_columns = list(data_cfg['output_columns'])
    output_rows = list(data_cfg['output_rows'])
    sample_subdir = str(data_cfg.get('sample_subdir', 'samples'))
    # MLR always evaluates on the non-perturbed samples/ directory.
    # The base config (written by d_RunResample for XGB/Transformer) carries
    # sample_subdir: mc_replicates — override that for MLR.
    if model_key in _MLR_MODEL_NAMES:
        sample_subdir = 'samples'

    # For evaluation params (historic_path etc.), prefer eval_config_path if given.
    if eval_config_path is not None:
        with open(eval_config_path, 'r', encoding='utf-8') as f:
            eval_cfg_doc = yaml.safe_load(f)
        ref_cfg = eval_cfg_doc
        ref_cfg_path = eval_config_path
        ref_data_cfg = eval_cfg_doc.get('data', data_cfg)
    else:
        ref_cfg = base_cfg
        ref_cfg_path = base_config_path
        ref_data_cfg = data_cfg

    summary_rows: list[dict] = []
    _h._append_mlr_baseline_outputs(
        summary_rows,
        [],
        ref_cfg=ref_cfg,
        ref_cfg_path=ref_cfg_path,
        ref_data_cfg=ref_data_cfg,
        data_dir=str(horizon_dir.resolve()),
        sample_subdir=sample_subdir,
        output_columns=output_columns,
        output_rows=output_rows,
        forecast_name='',
        test_samples=test_samples,
        test_split_files=test_split_files,
    )

    if not summary_rows:
        print(f"  [WARN] No baseline rows produced for {horizon_dir.name}")
        return False

    out_dir = output_dir if output_dir is not None else horizon_dir
    eval_module._write_summary_csv(summary_rows, out_dir / 'baseline_summary.csv')
    return True


def run_horizon_sweep(
    data_root,
    dataset_prefix,
    resample_config_path,
    preferred_lookaheads=None,
    n_replicates=1,
):
    # --- Load resample config ---
    resample_cfg = load_config(resample_config_path)
    config_dir = Path(resample_cfg['__config_dir'])

    input_csv = Path(resample_cfg['input_csv'])
    if not input_csv.is_absolute():
        input_csv = (config_dir / input_csv).resolve()

    use_uncertainty_perturbation = bool(resample_cfg.get('use_uncertainty_perturbation', True))
    n_mc_replicates = int(resample_cfg.get('n_mc_replicates', 10))
    random_seed = int(resample_cfg.get('random_seed', 1))
    verbose = bool(resample_cfg.get('verbose', False))

    # --- Load consolidated CSV ---
    print(f"[INFO] Loading data from {input_csv}")
    df_raw = pd.read_csv(input_csv, parse_dates=['TIMESTAMP'])
    df_raw = df_raw.sort_values('TIMESTAMP').reset_index(drop=True)

    # --- Find best ML and MLR configs per dataset ---
    best_configs_dual = find_best_configs_dual(data_root, dataset_prefix)

    for dataset_dir, ml_entry, mlr_entry in best_configs_dual:
        print(f"\n[DATASET] {dataset_dir.name}")

        # Determine the reference config to load common data dimensions from.
        # Prefer the ML entry; fall back to MLR.
        ref_entry = ml_entry if ml_entry is not None else mlr_entry
        ref_config, ref_best_row = ref_entry

        with open(ref_config, 'r', encoding='utf-8') as f:
            ref_cfg_yaml = yaml.safe_load(f)
        output_columns = ref_cfg_yaml.get('data', {}).get('output_columns', [])
        if not output_columns:
            print(f"  [WARN] No output_columns in ref config. Skipping.")
            continue
        target = output_columns[0]
        # predictor_cols is used for normalization column set; may differ per class but
        # we union them here so the normalization covers all columns.
        predictor_cols = ref_cfg_yaml.get('data', {}).get('input_columns', [])
        if ml_entry is not None:
            with open(ml_entry[0], 'r', encoding='utf-8') as f:
                _ml_cfg = yaml.safe_load(f)
            predictor_cols = list(dict.fromkeys(
                predictor_cols + _ml_cfg.get('data', {}).get('input_columns', [])
            ))

        # --- Normalization: reuse existing params for consistency across horizons ---
        norm_params = _load_normalization_params(dataset_dir)
        if norm_params is not None:
            print(f"  [INFO] Reusing normalization params from {dataset_dir.name}")
            df_norm = _apply_normalization(df_raw, norm_params)
            normalization_params = norm_params
        else:
            print(f"  [INFO] Computing normalization from raw data")
            to_normalize = predictor_cols + [target]
            df_norm, normalization_params = _normalize_once(df_raw, to_normalize)

        # --- Load sensor uncertainties once per dataset ---
        shared_sensor_uncertainties = None
        if use_uncertainty_perturbation:
            try:
                shared_sensor_uncertainties = _load_and_prepare_sensor_uncertainties(
                    str(dataset_dir),
                    normalization_params=normalization_params,
                    verbose=verbose,
                )
            except Exception as exc:
                print(f"  [WARN] Could not load sensor uncertainties: {exc}. Disabling perturbation.")
                use_uncertainty_perturbation = False

        # --- Build lookahead schedule from ref config ---
        sample_length = _base_window_rows_from_config(ref_config)
        lookaheads = _build_lookahead_schedule(preferred=preferred_lookaheads)
        if not lookaheads:
            print(f"  [WARN] No horizons to sweep. Skipping.")
            continue
        print(f"  [INFO] sample_length={sample_length}; horizons={lookaheads}; "
              f"ml={'yes' if ml_entry else 'no'}; mlr={'yes' if mlr_entry else 'no'}")

        # --- Sweep horizons ---
        # metrics_by_class collects DataFrames keyed by model class.
        metrics_by_class: dict[str, list] = {'ml': [], 'mlr': []}
        for horizon in lookaheads:
            horizon_label = f'{horizon:03d}hr'
            horizon_dir = dataset_dir / 'horizons' / horizon_label
            horizon_dir.mkdir(parents=True, exist_ok=True)

            # Migrate any pre-existing flat layout into ml/ or mlr/ subdirectory.
            _migrate_flat_horizon_layout(horizon_dir)

            # --- Per-class sweep ---
            for model_class, entry in [('ml', ml_entry), ('mlr', mlr_entry)]:
                if entry is None:
                    continue

                base_config, best_row = entry
                model_key = _model_name_to_key(str(best_row['model']))
                is_mlr = model_key in _MLR_MODEL_NAMES

                # Deterministic models: collapse replicates to 1.
                if model_key in _DETERMINISTIC_MODEL_KEYS and n_replicates > 1:
                    effective_replicates = 1
                else:
                    effective_replicates = n_replicates

                class_dir = horizon_dir / model_class
                class_dir.mkdir(parents=True, exist_ok=True)

                # Skip if all replicates already evaluated.
                if _class_sweep_complete(class_dir, effective_replicates):
                    print(f"  [SKIP] {model_class.upper()} horizon {horizon}hr — already complete")
                    # Still collect existing metrics for the CSV.
                    for rep_idx in range(effective_replicates):
                        rep_name = f'rep_{rep_idx:03d}'
                        eval_csv = class_dir / 'forecasts' / rep_name / 'evaluation_summary.csv'
                        if eval_csv.exists():
                            df_eval = pd.read_csv(eval_csv)
                            df_eval['horizon'] = horizon
                            df_eval['replicate'] = rep_idx
                            df_eval['model_class'] = model_class
                            df_eval['model_name'] = model_key
                            metrics_by_class[model_class].append(df_eval)
                    continue

                with open(base_config, 'r', encoding='utf-8') as f:
                    base_cfg_yaml = yaml.safe_load(f)
                class_predictor_cols = base_cfg_yaml.get('data', {}).get('input_columns', predictor_cols)
                class_sample_length = _base_window_rows_from_config(base_config)

                # Hyperparameter overrides from best model (ML only).
                if not is_mlr:
                    base_overrides = _extract_model_overrides(base_config)
                    training_cfg_defaults_horizon = {k: dict(v) for k, v in base_overrides.items()}
                else:
                    training_cfg_defaults_horizon = {}

                # Resample into class_dir (per class, not shared).
                print(f"  [RESAMPLE] {model_class.upper()} horizon {horizon}hr (seed={random_seed})")
                try:
                    result = resample_split(
                        df_norm,
                        str(class_dir),
                        [target],
                        class_sample_length,
                        nan_tol=0.8,
                        to_normalize=[],
                        fault_tolerant=True,
                        predictor_cols=class_predictor_cols,
                        use_uncertainty_perturbation=use_uncertainty_perturbation,
                        n_mc_replicates=n_mc_replicates,
                        random_seed=random_seed,
                        pre_normalized=True,
                        normalization_params=normalization_params,
                        sensor_uncertainties=shared_sensor_uncertainties,
                        verbose=False,
                        gap_rows=horizon,
                        training_config_defaults=training_cfg_defaults_horizon,
                    )
                except Exception as exc:
                    print(f"  [ERROR] Resampling failed for {model_class} horizon {horizon}hr: {exc}")
                    continue

                if result['n_samples'] == 0:
                    print(f"  [SKIP] {model_class.upper()} horizon {horizon}hr: no samples generated.")
                    continue

                if not is_mlr:
                    # Keep only the config matching the best model; delete the others.
                    for cfg_path in result['config_paths']:
                        p = Path(cfg_path)
                        if p.name != _MODEL_KEY_TO_CONFIG_NAME.get(model_key, ''):
                            p.unlink(missing_ok=True)
                        else:
                            target_path = class_dir / 'config.yml'
                            p.rename(target_path)

                # --- Per-replicate training / evaluation ---
                for rep_idx in range(effective_replicates):
                    rep_name = f'rep_{rep_idx:03d}'
                    rep_seed = random_seed + rep_idx
                    rep_dir_abs = str((class_dir / 'forecasts' / rep_name).resolve())

                    if is_mlr:
                        print(f"  [MLR] Horizon {horizon}hr rep {rep_idx}")
                        rep_dir = _run_mlr_horizon_rep(
                            class_dir=class_dir,
                            rep_name=rep_name,
                            base_config_path=base_config,
                            model_key=model_key,
                        )
                        if rep_dir is None:
                            continue
                        eval_csv = rep_dir / 'evaluation_summary.csv'
                    else:
                        # XGB / Transformer / GP branch
                        horizon_cfg = class_dir / 'config.yml'
                        if not horizon_cfg.exists():
                            print(f"  [SKIP] {model_class} horizon {horizon}hr rep {rep_idx}: config.yml missing.")
                            continue

                        with open(horizon_cfg, 'r', encoding='utf-8') as _f:
                            _cfg = yaml.safe_load(_f)
                        _cfg.setdefault('data', {})['forecast_name'] = rep_name
                        _cfg['data']['forecast_dir'] = rep_dir_abs
                        _cfg['data']['data_dir'] = str(class_dir.resolve())
                        seed_field = _MODEL_KEY_TO_SEED_FIELD.get(model_key)
                        if seed_field:
                            _cfg.setdefault('hyperparameters', {})[seed_field] = rep_seed
                        _cfg.setdefault('evaluation', {})['run_baselines'] = False
                        with open(horizon_cfg, 'w', encoding='utf-8') as _f:
                            yaml.dump(_cfg, _f, sort_keys=False)

                        train_cmd = [sys.executable, 'src/e_Train.py', '--config', str(horizon_cfg)]
                        print(f"  [TRAIN] {model_class.upper()} horizon {horizon}hr rep {rep_idx} (seed={rep_seed})")
                        try:
                            subprocess.run(train_cmd, check=True, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE)
                        except subprocess.CalledProcessError as exc:
                            print(f"  [ERROR] Training failed:")
                            print(exc.stderr.decode(errors='replace'))
                            continue

                        eval_cfg_path = class_dir / 'forecasts' / rep_name / f'config_evaluate_{rep_name}.yml'
                        eval_config_arg = str(eval_cfg_path) if eval_cfg_path.exists() else str(horizon_cfg)
                        eval_cmd = [sys.executable, 'src/f_Evaluate.py', '--config', eval_config_arg]
                        print(f"  [EVAL] {model_class.upper()} horizon {horizon}hr rep {rep_idx}")
                        try:
                            subprocess.run(eval_cmd, check=True, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE)
                        except subprocess.CalledProcessError as exc:
                            print(f"  [ERROR] Evaluation failed:")
                            print(exc.stderr.decode(errors='replace'))
                            continue

                        eval_csv = class_dir / 'forecasts' / rep_name / 'evaluation_summary.csv'

                    if eval_csv.exists():
                        df_eval = pd.read_csv(eval_csv)
                        df_eval['horizon'] = horizon
                        df_eval['replicate'] = rep_idx
                        df_eval['model_class'] = model_class
                        df_eval['model_name'] = model_key
                        df_eval['input_rows_included'] = class_sample_length
                        df_eval['input_rows_excluded'] = horizon
                        metrics_by_class[model_class].append(df_eval)
                    else:
                        print(f"  [WARN] evaluation_summary.csv not found: {eval_csv}")

                # --- Write baselines once per class per horizon ---
                baseline_csv = class_dir / 'baseline_summary.csv'
                _baseline_valid = False
                if baseline_csv.exists():
                    try:
                        _bl_check = pd.read_csv(baseline_csv)
                        _baseline_valid = (
                            'rmse' in _bl_check.columns
                            and _bl_check['rmse'].notna().any()
                        )
                    except Exception:
                        pass
                if not _baseline_valid:
                    _bl_written = False
                    if is_mlr:
                        from utils.training import splitter as _splitter
                        _data_cfg = base_cfg_yaml.get('data', {})
                        _in_r1 = int(_data_cfg.get('input_row_1', 0))
                        _in_r2 = int(_data_cfg.get('input_row_2', class_sample_length - 1))
                        _out_cols = list(_data_cfg.get('output_columns', [target]))
                        _out_rows = list(_data_cfg.get('output_rows', [_in_r2]))
                        _rep0_model_cfg = class_dir / 'forecasts' / 'rep_000' / 'model_config.json'
                        if _rep0_model_cfg.exists():
                            with open(_rep0_model_cfg, 'r', encoding='utf-8') as _f:
                                _mlr_cols = json.load(_f).get('input_columns', class_predictor_cols)
                        else:
                            _mlr_cols = class_predictor_cols
                        try:
                            _, _test_samples = _splitter(
                                str(class_dir),
                                'baseline_split',
                                _mlr_cols,
                                slice(_in_r1, _in_r2),
                                _out_cols,
                                _out_rows,
                                fault_tolerant=True,
                                reuse_split=False,
                                split_type='temporal',
                                test_size=0.3,
                                random_state=42,
                                sample_subdir='samples',
                                input_aggregation='none',
                                min_test_independent=5,
                            )
                            _test_split_files = [str(s[2]) for s in _test_samples]
                            _mlr_eval_cfg = _find_eval_config_with_historic(dataset_dir)
                            _bl_written = _write_horizon_baselines(
                                horizon_dir=class_dir,
                                base_config_path=base_config,
                                eval_config_path=_mlr_eval_cfg,
                                model_key=model_key,
                                test_samples=_test_samples,
                                test_split_files=_test_split_files,
                                output_dir=class_dir,
                            )
                        except Exception as exc:
                            print(f"  [WARN] Baseline split failed for {model_class} horizon {horizon}hr: {exc}")
                    else:
                        from utils.training import splitter as _splitter
                        rep0_dir = class_dir / 'forecasts' / 'rep_000'
                        rep0_eval_cfg = rep0_dir / 'config_evaluate_rep_000.yml'
                        if rep0_eval_cfg.exists():
                            try:
                                with open(rep0_eval_cfg, 'r', encoding='utf-8') as _f:
                                    _rep0_cfg = yaml.safe_load(_f)
                                _rep0_data = _rep0_cfg.get('data', {})
                                _in_r1 = int(_rep0_data.get('input_row_1', 0))
                                _in_r2 = int(_rep0_data.get('input_row_2', class_sample_length - 1))
                                _out_cols = list(_rep0_data.get('output_columns', [target]))
                                _out_rows = list(_rep0_data.get('output_rows', [_in_r2]))
                                _inp_cols = list(_rep0_data.get('input_columns', class_predictor_cols))
                                _sample_subdir = str(_rep0_data.get('sample_subdir', 'samples'))
                                _, _test_samples = _splitter(
                                    str(class_dir.resolve()),
                                    'rep_000',
                                    _inp_cols,
                                    slice(_in_r1, _in_r2),
                                    _out_cols,
                                    _out_rows,
                                    fault_tolerant=True,
                                    reuse_split=True,
                                    split_source=rep0_dir,
                                    sample_subdir=_sample_subdir,
                                    input_aggregation='none',
                                )
                                _test_split_files = [str(s[2]) for s in _test_samples]
                                _bl_written = _write_horizon_baselines(
                                    horizon_dir=class_dir,
                                    base_config_path=base_config,
                                    eval_config_path=rep0_eval_cfg,
                                    model_key=model_key,
                                    test_samples=_test_samples,
                                    test_split_files=_test_split_files,
                                    output_dir=class_dir,
                                )
                            except Exception as exc:
                                print(f"  [WARN] Baseline load failed for {model_class} horizon {horizon}hr: {exc}")
                    if not _bl_written:
                        print(f"  [WARN] Could not write baseline_summary.csv for {model_class} horizon {horizon}hr")

        # --- Save aggregate metrics and plots ---
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        all_class_dfs = []
        for model_class, df_list in metrics_by_class.items():
            if df_list:
                all_class_dfs.extend(df_list)

        if all_class_dfs:
            filtered = []
            for df in all_class_dfs:
                if 'kind' in df.columns:
                    filtered.append(df[df['kind'] == 'test'])
                elif 'label' in df.columns:
                    filtered.append(df[df['label'].str.contains('test', case=False, na=False)])
                else:
                    filtered.append(df)

            all_metrics = pd.concat(filtered, ignore_index=True)
            drop_cols = [col for col in ['label', 'kind'] if col in all_metrics.columns]
            all_metrics = all_metrics.drop(columns=drop_cols, errors='ignore')

            # Front columns: model_class, model_name, horizon, replicate
            cols = list(all_metrics.columns)
            front = [c for c in ['model_class', 'model_name', 'horizon', 'replicate'] if c in cols]
            rest = [c for c in cols if c not in front]
            all_metrics = all_metrics[front + rest]

            sweep_dir = dataset_dir / 'horizons' / 'lookahead_sweeps'
            sweep_dir.mkdir(parents=True, exist_ok=True)
            all_metrics.to_csv(sweep_dir / 'lookahead_metrics.csv', index=False)

            # Per-dataset summary plots — one series per model class.
            _CLASS_COLORS = {'ml': 'steelblue', 'mlr': 'darkorange'}
            _CLASS_STYLES = {'ml': '-', 'mlr': '--'}

            for metric, ylabel, filename in [
                ('rmse', 'RMSE', 'rmse_vs_lookahead.png'),
                ('r2',   'R²',   'r2_vs_lookahead.png'),
            ]:
                fig, ax = plt.subplots(figsize=(8, 5))
                legend_handles = []
                for model_class in ['ml', 'mlr']:
                    class_df = all_metrics[all_metrics['model_class'] == model_class] if 'model_class' in all_metrics.columns else pd.DataFrame()
                    if class_df.empty or metric not in class_df.columns:
                        continue
                    color = _CLASS_COLORS.get(model_class, 'steelblue')
                    ls = _CLASS_STYLES.get(model_class, '-')
                    has_reps = 'replicate' in class_df.columns and class_df['replicate'].nunique() > 1
                    mean_df = class_df.groupby('horizon')[[metric]].mean().reset_index()
                    if has_reps:
                        for _, rep_df in class_df.groupby('replicate'):
                            ax.scatter(rep_df['horizon'], rep_df[metric],
                                       s=18, alpha=0.35, color=color, zorder=2)
                    model_name = class_df['model_name'].iloc[0] if 'model_name' in class_df.columns else model_class
                    line, = ax.plot(mean_df['horizon'], mean_df[metric],
                                    marker='o', markersize=5, linewidth=1.8,
                                    color=color, linestyle=ls, label=f'{model_class.upper()} ({model_name})', zorder=3)
                    legend_handles.append(line)
                ax.set_xlabel('Horizon (hours)')
                ax.set_ylabel(ylabel)
                ax.set_title(f'{dataset_dir.name} – {ylabel} vs Horizon')
                ax.grid(True, alpha=0.3)
                if legend_handles:
                    ax.legend(handles=legend_handles, fontsize=9)
                fig.tight_layout()
                fig.savefig(sweep_dir / filename, dpi=150)
                plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Horizon sweep: resample with increasing gap_rows per horizon')
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

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


def find_best_configs(data_root, dataset_prefix):
    best_configs = []
    for dataset_dir in Path(data_root).iterdir():
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith(dataset_prefix):
            continue
        sweep_dir = dataset_dir / 'forecasts' / 'feature_sweeps'
        if not sweep_dir.exists():
            continue
        metrics_file = sweep_dir / 'feature_sweep_final_metrics.csv'
        if not metrics_file.exists():
            continue
        metrics_df = pd.read_csv(metrics_file)
        if metrics_df.empty or 'rmse' not in metrics_df.columns:
            continue
        is_baseline = metrics_df['model'].astype(str).str.strip().str.lower().isin(_BASELINE_MODEL_NAMES)
        ml_rows = metrics_df[~is_baseline].copy()
        if ml_rows.empty:
            continue
        if 'r2' not in ml_rows.columns:
            continue
        best = select_best_model_row(ml_rows)
        # Find config file for this subset and model
        model_name = str(best['model'])

        # MLR variants: configs live in feature_sweeps/{model_name}_{subset_label}/ or,
        # for rows merged from the Shapley sweep, in Shapley_sweeps/{model_name}_{subset_label}/
        if model_name in _MLR_MODEL_NAMES:
            subset_label = str(best.get('subset_label', 's01')).strip().lower()
            _SHAP_PFX = 'shap_'
            if subset_label.startswith(_SHAP_PFX):
                mlr_search_dir = dataset_dir / 'forecasts' / 'Shapley_sweeps'
                subset_label = subset_label[len(_SHAP_PFX):]
            else:
                mlr_search_dir = sweep_dir
            mlr_artifact_dir = mlr_search_dir / f'{model_name}_{subset_label}'
            config_path = mlr_artifact_dir / f'config_evaluate_{mlr_artifact_dir.name}.yml'
            if not config_path.exists():
                print(f"  [WARN] MLR eval config not found: {config_path}. Skipping {dataset_dir.name}.")
                continue
            best_configs.append((dataset_dir, config_path, best, mlr_artifact_dir))
            continue

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

        # Rows merged from the Shapley sweep have a 'shap_' prefix on subset_label;
        # their configs live in Shapley_sweeps/configs/ rather than feature_sweeps/configs/.
        subset_label_raw = str(best.get('subset_label', '')).strip().lower()
        _SHAP_PFX = 'shap_'
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
            # fallback: match by feature_tag only (covers NaN row_count or unmatched patterns)
            for cfg in configs_dir.glob(f"*{mapped_model_name}*{best['feature_tag']}*.yml"):
                config_path = cfg
                break
        if config_path is None:
            continue
        model_dir_name = model_map.get(model_name, model_name.lower())
        row_count_val = int(row_count_raw) if row_count_known else 0
        feature_tag = str(best['feature_tag'])
        subset_rank_val = int(best['subset_rank'])
        subset_rank_str = f"k{subset_rank_val:02d}"
        forecast_dir = sweep_namespace / f"{model_dir_name}_r{row_count_val}_{feature_tag}_{subset_rank_str}"
        best_configs.append((dataset_dir, config_path, best, forecast_dir))
    return best_configs


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


def _run_mlr_horizon_rep(
    *,
    horizon_dir: Path,
    rep_name: str,
    base_config_path: Path,
    model_key: str,
) -> 'Path | None':
    """Fit and evaluate one MLR replicate.  Returns the rep output dir Path on success.

    Samples are read from horizon_dir/samples/ (shared across all reps).
    Output is written to horizon_dir/forecasts/rep_NNN/.
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

    sample_dir = horizon_dir / 'samples'
    if not sample_dir.exists():
        print(f"  [WARN] MLR: samples dir not found: {sample_dir}")
        return None

    from utils.training import splitter
    input_rows = slice(input_row_1, input_row_2)
    try:
        train_samples, test_samples = splitter(
            str(horizon_dir),
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

    forecasts_dir = horizon_dir / 'forecasts'
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    # Write model-only evaluation_summary.csv (no baseline rows — those go to
    # baseline_summary.csv at the horizon level via _write_horizon_baselines).
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
    summary_row['data_dir'] = str(horizon_dir)
    eval_module._write_summary_csv([summary_row], rep_dir / 'evaluation_summary.csv')

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
) -> bool:
    """Compute Naive/Seasonal/Linear baselines and write horizon_dir/baseline_summary.csv.

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

    baseline_rows, _, _ = _h._append_mlr_baseline_outputs(
        [],
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

    if not baseline_rows:
        print(f"  [WARN] No baseline rows produced for {horizon_dir.name}")
        return False

    eval_module._write_summary_csv(baseline_rows, horizon_dir / 'baseline_summary.csv')
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

    # --- Find best configs per dataset ---
    best_configs = find_best_configs(data_root, dataset_prefix)

    for dataset_dir, base_config, best_row, _ in best_configs:
        print(f"\n[DATASET] {dataset_dir.name}")

        # --- Determine target column and model type ---
        with open(base_config, 'r', encoding='utf-8') as f:
            base_cfg_yaml = yaml.safe_load(f)
        output_columns = base_cfg_yaml.get('data', {}).get('output_columns', [])
        if not output_columns:
            print(f"  [WARN] No output_columns in base config. Skipping.")
            continue
        target = output_columns[0]
        predictor_cols = base_cfg_yaml.get('data', {}).get('input_columns', [])

        model_key = _model_name_to_key(str(best_row['model']))

        # Deterministic models produce identical results for every replicate — collapse to 1.
        if model_key in _DETERMINISTIC_MODEL_KEYS and n_replicates > 1:
            print(f"  [INFO] {model_key} is deterministic — collapsing {n_replicates} replicates to 1.")
            effective_replicates = 1
        else:
            effective_replicates = n_replicates

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

        # --- Build lookahead schedule ---
        sample_length = _base_window_rows_from_config(base_config)
        lookaheads = _build_lookahead_schedule(preferred=preferred_lookaheads)
        if not lookaheads:
            print(f"  [WARN] No horizons to sweep. Skipping.")
            continue
        print(f"  [INFO] sample_length={sample_length}; horizons={lookaheads}; "
              f"replicates={effective_replicates}; model={model_key}")

        # --- Base hyperparameter overrides from best model ---
        if model_key not in _MLR_MODEL_NAMES:
            base_overrides = _extract_model_overrides(base_config)
        else:
            base_overrides = {}

        # --- Sweep horizons ---
        metrics = []
        for horizon in lookaheads:
            horizon_label = f'{horizon:03d}hr'
            horizon_dir = dataset_dir / 'horizons' / horizon_label

            # --- Resample once per horizon (shared across all replicates) ---
            # forecast_name/dir are not meaningful at the horizon level; they will be
            # overridden per-replicate in the config before training.
            training_cfg_defaults_horizon = {}
            for k, v in base_overrides.items():
                training_cfg_defaults_horizon[k] = dict(v)

            print(f"  [RESAMPLE] Horizon {horizon}hr (seed={random_seed})")
            try:
                result = resample_split(
                    df_norm,
                    str(horizon_dir),
                    [target],
                    sample_length,
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
                    training_config_defaults=training_cfg_defaults_horizon,
                )
            except Exception as exc:
                print(f"  [ERROR] Resampling failed for horizon {horizon}hr: {exc}")
                continue

            if result['n_samples'] == 0:
                print(f"  [SKIP] Horizon {horizon}hr: no samples generated.")
                continue

            # Keep only the config matching the best model; delete the other two.
            for cfg_path in result['config_paths']:
                p = Path(cfg_path)
                if p.name != _MODEL_KEY_TO_CONFIG_NAME.get(model_key, ''):
                    p.unlink(missing_ok=True)
                else:
                    # Rename to the canonical single-config name.
                    target_path = horizon_dir / 'config.yml'
                    p.rename(target_path)

            # --- Per-replicate training ---
            for rep_idx in range(effective_replicates):
                rep_name = f'rep_{rep_idx:03d}'
                rep_seed = random_seed + rep_idx
                rep_dir_abs = str((horizon_dir / 'forecasts' / rep_name).resolve())

                if model_key in _MLR_MODEL_NAMES:
                    print(f"  [MLR] Horizon {horizon}hr rep {rep_idx}")
                    rep_dir = _run_mlr_horizon_rep(
                        horizon_dir=horizon_dir,
                        rep_name=rep_name,
                        base_config_path=base_config,
                        model_key=model_key,
                    )
                    if rep_dir is None:
                        continue
                    eval_csv = rep_dir / 'evaluation_summary.csv'
                else:
                    # --- XGB / Transformer / GP branch ---
                    horizon_cfg = horizon_dir / 'config.yml'
                    if not horizon_cfg.exists():
                        print(f"  [SKIP] Horizon {horizon}hr rep {rep_idx}: config.yml missing.")
                        continue

                    # Patch per-replicate values: forecast_name, forecast_dir, training seed.
                    with open(horizon_cfg, 'r', encoding='utf-8') as _f:
                        _cfg = yaml.safe_load(_f)
                    _cfg.setdefault('data', {})['forecast_name'] = rep_name
                    _cfg['data']['forecast_dir'] = rep_dir_abs
                    _cfg['data']['data_dir'] = str(horizon_dir.resolve())
                    # Inject per-replicate seed for training variance.
                    seed_field = _MODEL_KEY_TO_SEED_FIELD.get(model_key)
                    if seed_field:
                        _cfg.setdefault('hyperparameters', {})[seed_field] = rep_seed
                    # Disable baselines in f_Evaluate — they are written once per horizon below.
                    _cfg.setdefault('evaluation', {})['run_baselines'] = False
                    with open(horizon_cfg, 'w', encoding='utf-8') as _f:
                        yaml.dump(_cfg, _f, sort_keys=False)

                    # Train
                    train_cmd = [sys.executable, 'src/e_Train.py', '--config', str(horizon_cfg)]
                    print(f"  [TRAIN] Horizon {horizon}hr rep {rep_idx} (seed={rep_seed})")
                    try:
                        subprocess.run(train_cmd, check=True, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE)
                    except subprocess.CalledProcessError as exc:
                        print(f"  [ERROR] Training failed for horizon {horizon}hr rep {rep_idx}:")
                        print(exc.stderr.decode(errors='replace'))
                        continue

                    # Evaluate using the eval config written by e_Train.py.
                    eval_cfg_path = horizon_dir / 'forecasts' / rep_name / f'config_evaluate_{rep_name}.yml'
                    eval_config_arg = str(eval_cfg_path) if eval_cfg_path.exists() else str(horizon_cfg)
                    eval_cmd = [sys.executable, 'src/f_Evaluate.py', '--config', eval_config_arg]
                    print(f"  [EVAL] Horizon {horizon}hr rep {rep_idx}")
                    try:
                        subprocess.run(eval_cmd, check=True, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE)
                    except subprocess.CalledProcessError as exc:
                        print(f"  [ERROR] Evaluation failed for horizon {horizon}hr rep {rep_idx}:")
                        print(exc.stderr.decode(errors='replace'))
                        continue

                    eval_csv = horizon_dir / 'forecasts' / rep_name / 'evaluation_summary.csv'

                if eval_csv.exists():
                    df_eval = pd.read_csv(eval_csv)
                    df_eval['horizon'] = horizon
                    df_eval['replicate'] = rep_idx
                    df_eval['input_rows_included'] = sample_length
                    df_eval['input_rows_excluded'] = horizon
                    metrics.append(df_eval)
                else:
                    print(f"  [WARN] evaluation_summary.csv not found: {eval_csv}")

            # --- Write baselines once per horizon (after all reps, using last test split) ---
            # For non-MLR models, load the test split from rep_000 to run baselines.
            baseline_csv = horizon_dir / 'baseline_summary.csv'
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
                if model_key in _MLR_MODEL_NAMES:
                    # Re-derive test_samples from the shared samples directory for baseline use.
                    from utils.training import splitter as _splitter
                    _data_cfg = base_cfg_yaml.get('data', {})
                    _in_r1 = int(_data_cfg.get('input_row_1', 0))
                    _in_r2 = int(_data_cfg.get('input_row_2', sample_length - 1))
                    _out_cols = list(_data_cfg.get('output_columns', [target]))
                    _out_rows = list(_data_cfg.get('output_rows', [_in_r2]))
                    try:
                        _, _test_samples = _splitter(
                            str(horizon_dir),
                            'baseline_split',
                            predictor_cols,
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
                        _bl_written = _write_horizon_baselines(
                            horizon_dir=horizon_dir,
                            base_config_path=base_config,
                            model_key=model_key,
                            test_samples=_test_samples,
                            test_split_files=_test_split_files,
                        )
                    except Exception as exc:
                        print(f"  [WARN] Baseline split failed for horizon {horizon}hr: {exc}")
                else:
                    # For XGB/GP/Transformer: load test samples from rep_000's split and
                    # call _write_horizon_baselines directly.  The rep_000 eval config
                    # carries historic_path; the training config does not.
                    from utils.training import splitter as _splitter
                    rep0_dir = horizon_dir / 'forecasts' / 'rep_000'
                    rep0_eval_cfg = rep0_dir / 'config_evaluate_rep_000.yml'
                    if rep0_eval_cfg.exists():
                        try:
                            with open(rep0_eval_cfg, 'r', encoding='utf-8') as _f:
                                _rep0_cfg = yaml.safe_load(_f)
                            _rep0_data = _rep0_cfg.get('data', {})
                            _in_r1 = int(_rep0_data.get('input_row_1', 0))
                            _in_r2 = int(_rep0_data.get('input_row_2', sample_length - 1))
                            _out_cols = list(_rep0_data.get('output_columns', [target]))
                            _out_rows = list(_rep0_data.get('output_rows', [_in_r2]))
                            _inp_cols = list(_rep0_data.get('input_columns', predictor_cols))
                            _sample_subdir = str(_rep0_data.get('sample_subdir', 'samples'))
                            _, _test_samples = _splitter(
                                str(horizon_dir.resolve()),
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
                                horizon_dir=horizon_dir,
                                base_config_path=base_config,
                                eval_config_path=rep0_eval_cfg,
                                model_key=model_key,
                                test_samples=_test_samples,
                                test_split_files=_test_split_files,
                            )
                        except Exception as exc:
                            print(f"  [WARN] Baseline load failed for horizon {horizon}hr: {exc}")
                if not _bl_written:
                    print(f"  [WARN] Could not write baseline_summary.csv for horizon {horizon}hr")

        # --- Save aggregate metrics and plots ---
        if metrics:
            import matplotlib.pyplot as plt

            filtered = []
            for df in metrics:
                if 'kind' in df.columns:
                    filtered.append(df[df['kind'] == 'test'])
                elif 'label' in df.columns:
                    filtered.append(df[df['label'].str.contains('test', case=False, na=False)])
                else:
                    filtered.append(df)

            all_metrics = pd.concat(filtered, ignore_index=True)
            drop_cols = [col for col in ['label', 'kind'] if col in all_metrics.columns]
            all_metrics = all_metrics.drop(columns=drop_cols, errors='ignore')

            # Put horizon and replicate first
            cols = list(all_metrics.columns)
            front = [c for c in ['horizon', 'replicate'] if c in cols]
            rest = [c for c in cols if c not in front]
            all_metrics = all_metrics[front + rest]

            sweep_dir = dataset_dir / 'horizons' / 'lookahead_sweeps'
            sweep_dir.mkdir(parents=True, exist_ok=True)
            all_metrics.to_csv(sweep_dir / 'lookahead_metrics.csv', index=False)

            # Per-dataset summary plots
            has_reps = 'replicate' in all_metrics.columns and all_metrics['replicate'].nunique() > 1
            mean_metrics = all_metrics.groupby('horizon')[['rmse', 'r2']].mean().reset_index()

            for metric, ylabel, filename in [
                ('rmse', 'RMSE', 'rmse_vs_lookahead.png'),
                ('r2',   'R²',   'r2_vs_lookahead.png'),
            ]:
                fig, ax = plt.subplots(figsize=(8, 5))
                if has_reps:
                    for rep_id, rep_df in all_metrics.groupby('replicate'):
                        ax.scatter(rep_df['horizon'], rep_df[metric],
                                   s=18, alpha=0.45, color='steelblue', zorder=2)
                ax.plot(mean_metrics['horizon'], mean_metrics[metric],
                        marker='o', markersize=5, linewidth=1.8,
                        color='steelblue', label='Mean', zorder=3)
                ax.set_xlabel('Horizon (hours)')
                ax.set_ylabel(ylabel)
                ax.set_title(f'{dataset_dir.name} – {ylabel} vs Horizon')
                ax.grid(True, alpha=0.3)
                if has_reps:
                    from matplotlib.lines import Line2D
                    handles = [
                        Line2D([0], [0], color='steelblue', linewidth=1.8,
                               marker='o', markersize=5, label='Mean'),
                        Line2D([0], [0], color='steelblue', linewidth=0,
                               marker='o', markersize=5, alpha=0.45, label='Replicates'),
                    ]
                    ax.legend(handles=handles, fontsize=9)
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

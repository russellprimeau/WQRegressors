"""
Horizon sweep script: For each dataset, take the best-performing model/config (from
feature_sweeps), then for each horizon N call d_RunResample.split(gap_rows=N) to generate
a fresh set of samples where the predictor window ends N hours before the target, retrain
and evaluate the model, and save metrics/plots to forecasts/lookahead_sweeps/.

Each horizon's samples and model outputs are written to an isolated subdirectory:

    <dataset_dir>/horizons/horizon_NNNhr/
        samples/                        – raw sample CSVs (gap_rows=N)
        mc_replicates/                  – uncertainty-perturbed replicates
        config_<model>_01.yml           – training config for this horizon
        forecasts/horizon_NNNhr/
            evaluation_summary.csv      – per-set metrics for this horizon

Aggregate metrics and per-dataset plots are written to:

    <dataset_dir>/forecasts/lookahead_sweeps/
        lookahead_metrics.csv           – one row per horizon (test-set metrics)
        rmse_vs_lookahead.png
        r2_vs_lookahead.png

Use z2_horizon_post.py to produce cross-dataset comparison figures.

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

Examples:
    python src/k_RunHorizonSweep.py --resample-config data/output/sampling/resample_config.yml
    python src/k_RunHorizonSweep.py --data-root data/output/CV4 --dataset-prefix MC --resample-config data/output/sampling/resample_config.yml --horizons 0 6 12 24 48

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
from utils.config_utils import load_config

PREFERRED_LOOKAHEADS = [0, 1, 2, 6, 12, 24, 48, 96, 120, 167]

_MODEL_TYPE_TO_KEY = {
    'xgb_regressor': 'xgb',
    'xgb_classifier': 'xgb',
    'transformer': 'transformer',
    'gp_regressor': 'gp',
    # display names from feature_sweep_final_metrics.csv
    'XGBRegressor': 'xgb',
    'Transformer': 'transformer',
    'GPRegressor': 'gp',
}

_MODEL_KEY_TO_CONFIG_NAME = {
    'xgb': 'config_xgb_01.yml',
    'transformer': 'config_transformer_01.yml',
    'gp': 'config_gp_01.yml',
}


def _base_window_rows_from_config(config_path: Path) -> int:
    """Return base predictor window length from a training/eval config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get('data', {})
    base_start = int(data_cfg['input_row_1'])
    base_stop = int(data_cfg['input_row_2'])
    return int(base_stop - base_start + 1)  # indices are inclusive: rows input_row_1..input_row_2


def _build_lookahead_schedule(base_rows: int, preferred: list | None = None) -> list[int]:
    """Build a valid lookahead schedule bounded by the available base window.

    For a base window of N rows, max valid lookahead is N-1 (must leave >=1 row).
    The schedule keeps preferred milestones that fit and also includes the max
    endpoint to ensure full-range coverage for long-window targets.
    """
    if base_rows <= 0:
        return []

    max_lookahead = int(base_rows - 1)
    candidates = preferred if preferred is not None else PREFERRED_LOOKAHEADS
    schedule = [int(v) for v in candidates if 0 <= int(v) <= max_lookahead]
    if max_lookahead not in schedule:
        schedule.append(max_lookahead)
    return sorted(set(schedule))


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
        if metrics_df.empty or 'r2' not in metrics_df.columns:
            continue
        best_idx = metrics_df['r2'].idxmax()
        best = metrics_df.iloc[best_idx]
        # Find config file for this subset and model
        configs_dir = sweep_dir / 'configs'
        model_map = {
            'Transformer': 'transformer_01',
            'XGBRegressor': 'xgb_01',
            'GPRegressor': 'gp_01',
        }
        model_name = str(best['model'])
        mapped_model_name = model_map.get(model_name, model_name.lower())
        config_path = None
        for cfg in configs_dir.glob(f"*{mapped_model_name}*r{int(best['row_count']):03d}_{best['feature_tag']}*.yml"):
            config_path = cfg
            break
        if config_path is None:
            # fallback: match by feature_tag and row_count only
            for cfg in configs_dir.glob(f"*r{int(best['row_count']):03d}_{best['feature_tag']}*.yml"):
                config_path = cfg
                break
        if config_path is None:
            continue
        model_dir_name = model_map.get(model_name, model_name.lower())
        row_count_val = int(best['row_count'])
        feature_tag = str(best['feature_tag'])
        subset_rank_val = int(best['subset_rank'])
        subset_rank_str = f"k{subset_rank_val:02d}"
        forecast_dir = dataset_dir / 'forecasts' / 'feature_sweeps' / f"{model_dir_name}_r{row_count_val}_{feature_tag}_{subset_rank_str}"
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


def run_horizon_sweep(data_root, dataset_prefix, resample_config_path, preferred_lookaheads=None):
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
        lookaheads = _build_lookahead_schedule(base_rows=sample_length, preferred=preferred_lookaheads)
        if not lookaheads:
            print(f"  [WARN] No valid lookahead schedule for sample_length={sample_length}. Skipping.")
            continue
        print(f"  [INFO] sample_length={sample_length}; horizon schedule={lookaheads}")

        # --- Base hyperparameter overrides from best model ---
        base_overrides = _extract_model_overrides(base_config)

        # --- Sweep horizons ---
        metrics = []
        for horizon in lookaheads:
            horizon_label = f'horizon_{horizon:03d}hr'
            horizon_dir = dataset_dir / 'horizons' / horizon_label

            # Per-horizon training_config_defaults: inject forecast_name and forecast_dir.
            # forecast_dir (absolute) is included so f_Evaluate.py resolves test_files.txt
            # correctly even when falling back to the training config instead of the eval config.
            forecast_dir_abs = str((horizon_dir / 'forecasts' / horizon_label).resolve())
            training_cfg_defaults = {}
            for k, v in base_overrides.items():
                training_cfg_defaults[k] = dict(v)
                training_cfg_defaults[k]['data'] = {
                    'forecast_name': horizon_label,
                    'forecast_dir': forecast_dir_abs,
                }

            print(f"  [RESAMPLE] Horizon {horizon}hr -> {horizon_dir.name}")
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
                    training_config_defaults=training_cfg_defaults,
                )
            except Exception as exc:
                print(f"  [ERROR] Resampling failed for horizon {horizon}hr: {exc}")
                continue

            if result['n_samples'] == 0:
                print(f"  [SKIP] Horizon {horizon}hr: no samples generated.")
                continue

            horizon_cfg = _select_horizon_config(result['config_paths'], model_key)
            if horizon_cfg is None:
                print(f"  [SKIP] Horizon {horizon}hr: could not find config for model_key={model_key}.")
                continue

            # Patch forecast_name and forecast_dir directly in the config file.
            # d_RunResample.split() hardcodes 'xgb_01' as the forecast_name arg; the
            # training_config_defaults override is not reliable when the horizon dir already
            # contains a config from a prior run. Writing here guarantees the correct values.
            with open(horizon_cfg, 'r', encoding='utf-8') as _f:
                _cfg = yaml.safe_load(_f)
            _cfg.setdefault('data', {})['forecast_name'] = horizon_label
            _cfg['data']['forecast_dir'] = forecast_dir_abs
            with open(horizon_cfg, 'w', encoding='utf-8') as _f:
                yaml.dump(_cfg, _f, sort_keys=False)

            # Train
            train_cmd = [sys.executable, 'src/e_Train.py', '--config', str(horizon_cfg)]
            print(f"  [TRAIN] Horizon {horizon}hr")
            try:
                subprocess.run(train_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as exc:
                print(f"  [ERROR] Training failed for horizon {horizon}hr:")
                print(exc.stderr.decode(errors='replace'))
                continue

            # Evaluate — prefer the eval config written by e_Train.py (paths resolved correctly).
            # e_Train.py names it config_evaluate_<forecast_name>.yml (line 651 of e_Train.py).
            eval_cfg_path = horizon_dir / 'forecasts' / horizon_label / f'config_evaluate_{horizon_label}.yml'
            eval_config_arg = str(eval_cfg_path) if eval_cfg_path.exists() else str(horizon_cfg)
            eval_cmd = [sys.executable, 'src/f_Evaluate.py', '--config', eval_config_arg]
            print(f"  [EVAL] Horizon {horizon}hr")
            try:
                subprocess.run(eval_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as exc:
                print(f"  [ERROR] Evaluation failed for horizon {horizon}hr:")
                print(exc.stderr.decode(errors='replace'))
                continue

            eval_csv = horizon_dir / 'forecasts' / horizon_label / 'evaluation_summary.csv'
            if eval_csv.exists():
                df_eval = pd.read_csv(eval_csv)
                df_eval['horizon'] = horizon
                df_eval['input_rows_included'] = sample_length
                df_eval['input_rows_excluded'] = horizon
                metrics.append(df_eval)
            else:
                print(f"  [WARN] evaluation_summary.csv not found: {eval_csv}")

        # --- Save metrics and plots ---
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

            cols = list(all_metrics.columns)
            if 'horizon' in cols:
                cols.insert(0, cols.pop(cols.index('horizon')))
                all_metrics = all_metrics[cols]

            sweep_dir = dataset_dir / 'forecasts' / 'lookahead_sweeps'
            sweep_dir.mkdir(parents=True, exist_ok=True)
            all_metrics.to_csv(sweep_dir / 'lookahead_metrics.csv', index=False)

            plt.figure(figsize=(8, 5))
            plt.plot(all_metrics['horizon'], all_metrics['rmse'], marker='o', label='RMSE')
            plt.xlabel('Horizon (hours)')
            plt.ylabel('RMSE')
            plt.title(f'{dataset_dir.name} - RMSE vs Horizon')
            plt.grid(True)
            plt.savefig(sweep_dir / 'rmse_vs_lookahead.png')
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(all_metrics['horizon'], all_metrics['r2'], marker='o', label='R2')
            plt.xlabel('Horizon (hours)')
            plt.ylabel('R2')
            plt.title(f'{dataset_dir.name} - R2 vs Horizon')
            plt.grid(True)
            plt.savefig(sweep_dir / 'r2_vs_lookahead.png')
            plt.close()


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
    args = parser.parse_args()
    run_horizon_sweep(args.data_root, args.dataset_prefix, args.resample_config, args.horizons)

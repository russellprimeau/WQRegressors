'''
Analyze the combined dataset to identify commonly-sized chunks of data which can be used for forecasting over
different horizons.
Then, split the dataset into files for each equivalent sample.

Example:
python src/d_RunResample.py --config data/input/splitting/resample_config.yml
'''

import os
import sys
import json
import re
import argparse
import pandas as pd
import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from pathlib import Path
from utils.preprocessing import normalize_columns
from utils.config_utils import load_config


DEFAULT_SAMPLE_LENGTH_ROWS = 168
EUROFINS_SUMMARY_DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "output"
    / "sensors"
    / "tables"
    / "Eurofins_summary.csv"
)

NORMALIZATION_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "output"
    / "sensors"
    / "normalization.json"
)


def _normalize_target_for_eurofins_lookup(name):
    """Normalize target/parameter names for robust Eurofins matching."""
    text = str(name).strip()
    for suffix in ("_diff", "_res", "_state"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return text.casefold()


def _load_eurofins_summary_df(summary_csv_path=EUROFINS_SUMMARY_DEFAULT_PATH):
    """Load Eurofins summary table and add normalized lookup keys."""
    summary_path = Path(summary_csv_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Eurofins summary CSV not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    required_columns = {"parameter", "median_hours_between_measurements"}
    missing = required_columns.difference(summary_df.columns)
    if missing:
        raise ValueError(
            f"Eurofins summary CSV missing required columns {sorted(missing)}: {summary_path}"
        )

    summary_df = summary_df.copy()
    summary_df["_parameter_normalized"] = summary_df["parameter"].map(
        _normalize_target_for_eurofins_lookup
    )
    return summary_df


def _get_target_sample_length_rows(
    target_name,
    eurofins_summary_df,
    default_rows=DEFAULT_SAMPLE_LENGTH_ROWS,
    verbose=True,
):
    """
    Resolve sample length (rows) from Eurofins median measurement interval.

    Falls back to default_rows when target lookup fails or median is invalid.
    """
    target_key = _normalize_target_for_eurofins_lookup(target_name)
    matches = eurofins_summary_df.loc[
        eurofins_summary_df["_parameter_normalized"] == target_key
    ]

    if matches.empty:
        if verbose:
            print(
                f"[WARN] No Eurofins summary match for target '{target_name}'. "
                f"Using fallback sample length={default_rows}."
            )
        return int(default_rows)

    if len(matches) > 1 and verbose:
        print(
            f"[WARN] Multiple Eurofins matches for target '{target_name}'. "
            "Using the first match."
        )

    median_value = pd.to_numeric(
        matches.iloc[0]["median_hours_between_measurements"], errors="coerce"
    )
    if pd.isna(median_value) or median_value <= 0:
        if verbose:
            print(
                f"[WARN] Invalid median_hours_between_measurements for target '{target_name}': "
                f"{median_value}. Using fallback sample length={default_rows}."
            )
        return int(default_rows)

    return max(1, int(round(float(median_value))))


def _default_aggregate_offset_csv():
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "data" / "output" / "calibration" / "aggregate" / "offset_gain_model_results.csv",
        root / "data" / "output" / "calibration" / "summaries" / "aggregate" / "offset_gain_model_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _fit_offset_t_params_from_aggregate(sensor_names):
    """Fit per-sensor Student's t parameters from aggregate offset data."""
    agg_csv = _default_aggregate_offset_csv()
    if agg_csv is None:
        raise FileNotFoundError(
            "Could not find offset_gain_model_results.csv in expected calibration aggregate paths."
        )

    agg_df = pd.read_csv(agg_csv)
    if "Sensor" not in agg_df.columns or "Offset" not in agg_df.columns:
        raise ValueError(f"Aggregate CSV missing required columns Sensor/Offset: {agg_csv}")

    def _normalize_sensor_name(name):
        text = str(name).strip().replace("Âµ", "µ")
        if "Sp Cond" in text:
            return "Sp Cond (microS_cm)"
        return text

    agg_df = agg_df.copy()
    agg_df["Sensor_Normalized"] = agg_df["Sensor"].map(_normalize_sensor_name)
    agg_df["Offset"] = pd.to_numeric(agg_df["Offset"], errors='coerce')

    out = {}
    for sensor_name in sensor_names:
        offsets = (
            agg_df.loc[agg_df["Sensor_Normalized"] == sensor_name, "Offset"]
            .dropna()
            .to_numpy(dtype=float)
        )
        if offsets.size < 3:
            raise ValueError(
                f"Insufficient offset data for Student's t fit: sensor={sensor_name}, n={offsets.size}"
            )

        try:
            t_df, t_loc, t_scale = stats.t.fit(offsets)
        except Exception as exc:
            raise ValueError(f"Could not fit Student's t for sensor={sensor_name}: {exc}") from exc

        if (
            (not np.isfinite(t_df))
            or (not np.isfinite(t_loc))
            or (not np.isfinite(t_scale))
            or t_scale <= 0
        ):
            raise ValueError(
                f"Invalid Student's t parameters for sensor={sensor_name}: "
                f"df={t_df}, loc={t_loc}, scale={t_scale}"
            )

        out[sensor_name] = {
            "Offset_Distribution": "t",
            "Offset_t_df": float(t_df),
            "Offset_t_loc": float(t_loc),
            "Offset_t_scale": float(t_scale),
            "Offset_Mean": float(np.mean(offsets)),
            "Offset_Std": float(np.std(offsets, ddof=0)),
        }

    return out


def _normalize_once(df, columns, min_val=0.0, max_val=1.0):
    """Normalize selected columns once and return (normalized_df, normalization_params)."""
    df_norm = df.copy()
    normalization_params = {}

    for col in columns:
        if col not in df_norm.columns:
            continue

        series = pd.to_numeric(df_norm[col], errors='coerce')
        col_min = series.min()
        col_max = series.max()
        normalization_params[col] = {"min": col_min, "max": col_max}

        if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
            df_norm[col] = (min_val + max_val) / 2
        else:
            df_norm[col] = ((series - col_min) / (col_max - col_min)) * (max_val - min_val) + min_val

    return df_norm, normalization_params


def _write_normalization_params(normalization_params, output_path=NORMALIZATION_OUTPUT_PATH):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(normalization_params, f)


def _load_and_prepare_sensor_uncertainties(output_dir, normalization_params=None, verbose=False):
    """Load uncertainty summaries once and optionally convert to normalized scale."""
    sensor_uncertainties = {}
    if verbose:
        print("Loading sensor uncertainty summaries for Monte Carlo sampling...")

    corrections_dir = Path(__file__).parent.parent / "data" / "output" / "calibration" / "summaries"
    sensor_names = ['Sp Cond (microS_cm)', 'pH', 'DO (% Sat)', 'Turbidity (FNU)', 'fDOM (RFU)', 'fDOM (QSU)']
    for sensor_name in sensor_names:
        summary = load_sensor_uncertainty_summary(sensor_name, corrections_dir)
        if summary is not None:
            sensor_uncertainties[sensor_name] = summary
            if verbose:
                print(f"  OK: Loaded {sensor_name}")
        else:
            if verbose:
                print(f"  X: No uncertainty summary found for {sensor_name}")

    # Force Student's t offset model from aggregate fits. Fail fast if parameters cannot be fitted.
    t_fit_map = _fit_offset_t_params_from_aggregate(sensor_names)
    for sensor_name in sensor_names:
        if sensor_name not in sensor_uncertainties:
            sensor_uncertainties[sensor_name] = {}
        sensor_uncertainties[sensor_name].update(t_fit_map[sensor_name])

    if normalization_params is None:
        normalization_file = NORMALIZATION_OUTPUT_PATH
        try:
            with open(normalization_file, 'r') as f:
                normalization_params = json.load(f)
        except FileNotFoundError:
            normalization_params = None

    if normalization_params:
        if verbose:
            print("\nTransforming uncertainty parameters to normalized scale...")
        sensor_column_map = {
            'Sp Cond (microS_cm)': 'Pfl - Sp Cond (microS_cm)',
            'pH': 'Pfl - pH',
            'DO (% Sat)': 'Pfl - DO (% Sat)',
            'Turbidity (FNU)': 'Pfl - Turbidity (FNU)',
            'fDOM (RFU)': 'Pfl - fDOM (RFU)',
            'fDOM (QSU)': 'Pfl - fDOM (QSU)',
        }
        for sensor_name in list(sensor_uncertainties.keys()):
            col_name = sensor_column_map.get(sensor_name)
            if col_name:
                sensor_uncertainties[sensor_name] = normalize_uncertainty_params(
                    sensor_uncertainties[sensor_name], col_name, normalization_params
                )
                if verbose:
                    print(f"  OK: Transformed {sensor_name} to normalized scale")
    else:
        if verbose:
            print("\n! Warning: Could not load normalization parameters.")
            print("  Using raw-scale uncertainty parameters (may cause over-perturbation)")

    return sensor_uncertainties

def clean_directory(directory_path):
    """
    Deletes all files within the specified directory.
    Subdirectories and their contents are not affected.
    """
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f"Deleted file: {file_path}")
    except OSError as e:
        print(f"Error deleting files in {directory_path}: {e}")


def generate_training_config_template(output_dir, forecast_name, input_columns, output_columns, sample_length, model_type='xgb', overrides=None):
    """
    Generate a template configuration file for e_Train.py.
    
    :param output_dir: Output directory where config will be saved
    :param forecast_name: Name of the forecast/dataset (used as model_name base)
    :param input_columns: List of input column names
    :param output_columns: List of output column names
    :param sample_length: Length of each sample (number of rows)
    :param model_type: Type of model for naming ('xgb' or 'transformer')
    :return: Path to the generated config file

    Notes:
    - XGBoost configs now include optional train-set-only CV tuning settings
       under hyperparameters.cv_tuning (including param_space). These are
       enabled by default for predictive-performance-focused runs.
    """
    # Create the template configuration
    config = {
        'model_type': 'xgb_regressor',  # Options: 'transformer', 'gp_regressor', 'xgb_regressor', 'xgb_classifier'
        'model_name': f'model_{model_type}_01',
        'device': 'cuda',  # or 'cpu'
        'matplotlib_backend': 'Agg',
        'data': {
            'data_dir': '.',
            'sample_subdir': 'mc_replicates',
            'forecast_name': forecast_name,
            'input_columns': input_columns,
            'input_row_1': 0,
            'input_row_2': sample_length - 1,  # All rows except the last
            'output_columns': output_columns,
            'output_rows': [sample_length - 1],  # Last row is the target
        },
        'data_split': {
            'random_state': 42,
            'test_size': 0.3,
            'reuse_split': False,
            'split_source': None,
            'split_type': 'temporal',
            'fault_tolerant': True,
            'nan_tolerance': 0.8,
        },
        'hyperparameters': {
            # XGBoost Regressor hyperparameters (adjust for your use case)
            'metric': 'rmse',
            'tree_method': 'hist',
            'objective': 'reg:squarederror',
            'n_estimators': 800,
            'max_depth': 7,
            'subsample': 0.2,
            'colsample_bytree': 0.8,
            'min_child_weight': 8,
            'gamma': 0,
            'reg_lambda': 1,
            'reg_alpha': 0,
            'learning_rate': 0.01,
            'n_jobs': -1,
            'early_stopping_rounds': 20,
            'train_loss_plateau_rounds': 10,
            'train_loss_min_relative_improvement': 0.01,
            # Optional CV tuning for regularization (train-set only).
            'cv_tuning': {
                'enabled': True,
                'n_folds': 5,
                'n_trials': 250,
                'seed': 21,
                'parallel_jobs': 1,
                'metric': 'rmse_r2',
                'selection_rule': 'best',
                'search_method': 'optuna',
                'optuna_sampler': 'tpe',
                'random_start_trials': 30,
                'use_early_stopping': False,
                'early_stopping_rounds': None,
                'verbose': False,
                'min_r2': 0.0,        # or 0.05, 0.1 depending on how strict you want to be
                'r2_penalty': 10.0,   # larger => harsher penalty
                'param_space': {
                    'n_estimators': {'low': 100, 'high': 1300, 'type': 'int'},
                    'max_depth': {'low': 2, 'high': 9, 'type': 'int'},
                    'min_child_weight': {'low': 1, 'high': 20, 'type': 'int'},
                    'gamma': {'low': 0.0, 'high': 2.0, 'type': 'float'},
                    'subsample': {'low': 0.5, 'high': 0.9, 'type': 'float'},
                    'colsample_bytree': {'low': 0.3, 'high': 1.0, 'type': 'float'},
                    'reg_lambda': {'low': 1e-3, 'high': 10.0, 'type': 'log'},
                    'reg_alpha': {'low': 1e-4, 'high': 5.0, 'type': 'log'},
                    'learning_rate': {'low': 0.003, 'high': 0.2, 'type': 'log'},
                },
            },
        },
        '_comments': {
            'model_type': "Options: 'transformer', 'gp_regressor', 'xgb_regressor', 'xgb_classifier'",
            'hyperparameters': "Adjust based on your model_type selection",
            'data_dir': "Base dataset directory, resolved relative to this config file location.",
            'sample_subdir': "Defaults to 'mc_replicates' so XGBoost trains on uncertainty-perturbed Monte Carlo samples.",
            'input_row_2': f"Currently {sample_length - 1} (all rows except last). Can be adjusted.",
            'output_rows': f"Currently [sample_length - 1] to predict last row only. Can be adjusted.",
            'split_type': "Temporal split prevents data leakage. If MC replicates detected, temporal split is auto-enforced.",
            'fault_tolerant': "True for XGBoost (handles NaN in inputs), False for Transformer/GP (require complete data)",
            'nan_tolerance': "Applied in e_Train.py before split when fault_tolerant=True. Maximum allowed NaN fraction in predictor windows.",
            'train_loss_plateau_rounds': "Optional rolling-window size N for train-loss plateau stopping. Rule is disabled when not provided.",
            'train_loss_min_relative_improvement': "Optional minimum net relative improvement required from round t-N to t (e.g., 0.001 = 0.1%). Stop triggers when rolling-window net improvement is not greater than this threshold.",
            'stop_rule_combination': "If enabled, train-loss plateau stop is non-exclusive with early_stopping_rounds (training stops when either rule triggers first).",
            'cv_tuning': "Optional train-set-only CV tuning for regularization. When enabled, tuned hyperparameters are saved and reused in future runs.",
        }
    }
    
    if overrides:
        config['hyperparameters'].update(overrides.get('hyperparameters', {}))
        for key in ('model_type', 'model_name', 'device', 'matplotlib_backend'):
            if key in overrides:
                config[key] = overrides[key]
        if 'data' in overrides:
            config['data'].update(overrides['data'])
        if 'data_split' in overrides:
            config['data_split'].update(overrides['data_split'])

    # Save as YAML
    # Named after the forecast, not the model type. A fixed `_01` meant every window
    # representation of a family wrote to one file and only the last survived: a run
    # generated with three XGBoost variants ended up with a single config_xgb_01.yml
    # holding the third one's settings under the first one's name.
    _stem = str(forecast_name).strip() or f'{model_type}_01'
    config_path = Path(output_dir) / f'config_{_stem}.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return str(config_path)


def generate_transformer_config_template(output_dir, forecast_name, input_columns, output_columns, sample_length, overrides=None):
    """
    Generate a template configuration file for Transformer models.
    
    :param output_dir: Output directory where config will be saved
    :param forecast_name: Name of the forecast/dataset
    :param input_columns: List of input column names
    :param output_columns: List of output column names
    :param sample_length: Length of each sample (number of rows)
    :return: Path to the generated config file
    """
    # Create the template configuration for Transformer
    config = {
        'model_type': 'transformer',
        'model_name': f'model_transformer_01',
        'device': 'cuda',  # or 'cpu'
        'matplotlib_backend': 'Agg',
        'data': {
            'data_dir': '.',
            'sample_subdir': 'mc_replicates',
            'forecast_name': forecast_name,
            'input_columns': input_columns,
            'input_row_1': 0,
            'input_row_2': sample_length - 1,  # All rows except the last
            'output_columns': output_columns,
            'output_rows': [sample_length - 1],  # Last row is the target
        },
        'data_split': {
            'random_state': 42,
            'test_size': 0.3,
            'reuse_split': False,
            'split_source': None,
            'split_type': 'temporal',
            'fault_tolerant': False,  # Transformer requires complete data
            'nan_tolerance': 0.0,
        },
        'hyperparameters': {
            # Transformer hyperparameters
            'model_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 100,
            'loss_threshold': 0.000001,
            'patience': 10,
            # Pure MSE is the default; correlation regularization is opt-in.
            'corr_lambda': 0.0,
            'corr_eps': 1e-8,
            'corr_clip': True,
        },
        '_comments': {
            'model_type': "This is a Transformer model configuration",
            'hyperparameters': "Transformer-specific hyperparameters",
            'data_dir': "Base dataset directory, resolved relative to this config file location.",
            'sample_subdir': "Defaults to 'mc_replicates' so Transformer trains on uncertainty-perturbed Monte Carlo samples.",
            'input_row_2': f"Currently {sample_length - 1} (all rows except last). Can be adjusted.",
            'output_rows': f"Currently [sample_length - 1] to predict last row only. Can be adjusted.",
            'split_type': "Temporal split prevents data leakage. If MC replicates detected, temporal split is auto-enforced.",
            'fault_tolerant': "False - Transformer requires complete data without NaN in inputs",
            'nan_tolerance': "0.0 for strict no-NaN predictor policy.",
        }
    }
    
    if overrides:
        config['hyperparameters'].update(overrides.get('hyperparameters', {}))
        for key in ('model_type', 'model_name', 'device', 'matplotlib_backend'):
            if key in overrides:
                config[key] = overrides[key]
        if 'data' in overrides:
            config['data'].update(overrides['data'])
        if 'data_split' in overrides:
            config['data_split'].update(overrides['data_split'])

    # Save as YAML
    _stem = str(forecast_name).strip() or 'transformer_01'
    config_path = Path(output_dir) / f'config_{_stem}.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return str(config_path)


# The GP was previously emitted as a single configuration: the flattened window
# (167 rows x n predictors = up to 835 input dimensions) with ARD and a
# matern52+linear kernel. That combination cannot be fitted from the 11-52
# training samples a target supplies -- the saved models show all 835 ARD
# lengthscales identical to within half a percent and every hyperparameter still
# at its initial value, because the cross-validated epoch budget halts after a
# median of 24 of 250 epochs. The unbounded linear term then dominates away from
# the training data, producing predictions up to 106x outside the target range.
#
# These four variants separate the two factors responsible, keeping the previous
# configuration as gp_01 so the effect of each change is measurable rather than
# asserted:
#
#   gp_01  flattened window, matern52+linear   (previous behaviour, control)
#   gp_02  flattened window, matern52          (isolates the unbounded linear term)
#   gp_03  aggregated window, matern52+linear  (isolates the input dimension)
#   gp_04  aggregated window, matern52         (both changes)
#
# Aggregating the window takes the input dimension to one value per predictor, at
# which point ARD and the epoch budget have something estimable to work with.
# How the input window is presented to the tree and sequence models. The GP variants
# already contrast a flattened window against a per-predictor mean; these give the
# other two families the same axis, and two intermediate representations that neither
# throws the window away nor hands over every hour of it.
#
#   none       the window as-is. XGBoost flattens it -- 7381 columns for a 28-day
#              window against roughly 30 distinct training samples -- and the
#              transformer carries a positional embedding of one block per hour.
#   stats:24h  daily blocks, each summarised by mean/min/max/std. Level, spread and
#              extremes within each day survive; hour-to-hour detail does not.
#   stats:6h   the same summary at six-hourly resolution: four blocks per day, so
#              within-day structure survives too, at four times the width.
#
# The two summaries differ only in block width, which is the axis the evidence
# actually turns on. Measured per target, the daily summary beat the raw window on
# both 671-row targets (pH 0.217 against 0.142, Turbidity 0.204 against 0.027) and
# lost badly on the 167-row one (Cadmium 0.005 against 0.075) -- where a daily block
# leaves only seven timesteps. Six-hourly blocks give a short window the resolution it
# appears to want without imposing it on a long one.
#
# Daily rolling means (lag:24) were tried in this slot and dropped: they keep only the
# level of each day and discard the spread and extremes that the summary retains from
# the same data, and they finished last on every target that could be resolved.
#
# Block width is specified as an interval rather than a count because the window is
# not the same length for every target -- 671 rows for pH, Colour, Turbidity and
# intestinal enterococci, 167 for the other ten. A fixed count would make one
# configuration name mean daily blocks on one target and six-hourly blocks on another.
_WINDOW_REPRESENTATIONS = (
    {"suffix": "01", "input_aggregation": "none",
     "note": "Control: the full window, flattened by the model as before."},
    {"suffix": "02", "input_aggregation": "stats:24h",
     "note": "Daily blocks summarised by mean, min, max and standard deviation."},
    {"suffix": "03", "input_aggregation": "stats:6h",
     "note": "The same summary at six-hourly resolution."},
)


def generate_xgb_config_templates(output_dir, input_columns, output_columns,
                                  sample_length, overrides=None):
    """Emit an XGBoost config per window representation and return their paths."""
    paths = []
    for variant in _WINDOW_REPRESENTATIONS:
        ov = dict(overrides or {})
        # Overrides are merged by section, so the representation has to be placed in
        # the data section rather than at the top level or it is silently ignored.
        ov["data"] = dict(ov.get("data") or {})
        ov["data"]["input_aggregation"] = variant["input_aggregation"]
        paths.append(generate_training_config_template(
            output_dir,
            f"xgb_{variant['suffix']}",
            input_columns,
            output_columns,
            sample_length,
            model_type='xgb',
            overrides=ov,
        ))
    return paths


def generate_transformer_config_templates(output_dir, input_columns, output_columns,
                                          sample_length, overrides=None):
    """Emit a transformer config per window representation and return their paths."""
    paths = []
    for variant in _WINDOW_REPRESENTATIONS:
        ov = dict(overrides or {})
        ov["data"] = dict(ov.get("data") or {})
        ov["data"]["input_aggregation"] = variant["input_aggregation"]
        paths.append(generate_transformer_config_template(
            output_dir,
            f"transformer_{variant['suffix']}",
            input_columns,
            output_columns,
            sample_length,
            overrides=ov,
        ))
    return paths


GP_CONFIG_VARIANTS = (
    {"suffix": "01", "kernel": "matern52+linear", "input_aggregation": "none",
     "note": "Control: previous configuration, flattened window with a linear kernel term."},
    {"suffix": "02", "kernel": "matern52", "input_aggregation": "none",
     "note": "Bounded kernel on the flattened window; isolates the linear term."},
    {"suffix": "03", "kernel": "matern52+linear", "input_aggregation": "mean",
     "note": "Aggregated window; isolates the input dimension."},
    {"suffix": "04", "kernel": "matern52", "input_aggregation": "mean",
     "note": "Aggregated window and bounded kernel."},
)


def generate_gp_config_templates(output_dir, input_columns, output_columns, sample_length,
                                 overrides=None):
    """Emit every GP variant and return their config paths."""
    paths = []
    for variant in GP_CONFIG_VARIANTS:
        paths.append(generate_gp_config_template(
            output_dir,
            f"gp_{variant['suffix']}",
            input_columns,
            output_columns,
            sample_length,
            overrides=overrides,
            variant=variant,
        ))
    return paths


def generate_gp_config_template(output_dir, forecast_name, input_columns, output_columns, sample_length, overrides=None, variant=None):
    """
    Generate a template configuration file for Gaussian Process Regressor models.

    :param output_dir: Output directory where config will be saved
    :param forecast_name: Name of the forecast/dataset
    :param input_columns: List of input column names
    :param output_columns: List of output column names
    :param sample_length: Length of each sample (number of rows)
    :return: Path to the generated config file
    """
    variant = dict(variant or {"suffix": "01", "kernel": "matern52+linear",
                              "input_aggregation": "none", "note": ""})
    _suffix = str(variant.get("suffix", "01"))
    config = {
        'model_type': 'gp_regressor',
        'model_name': f'model_gp_{_suffix}',
        'device': 'cuda',  # or 'cpu'
        'matplotlib_backend': 'Agg',
        'data': {
            'data_dir': '.',
            'sample_subdir': 'samples',
            'forecast_name': forecast_name,
            'input_columns': input_columns,
            'input_row_1': 0,
            'input_row_2': sample_length - 1,  # All rows except the last
            'output_columns': output_columns,
            'output_rows': [sample_length - 1],  # Last row is the target
            'input_aggregation': variant.get('input_aggregation', 'none'),
        },
        'data_split': {
            'random_state': 42,
            'test_size': 0.3,
            'reuse_split': False,
            'split_source': None,
            'split_type': 'temporal',
            'fault_tolerant': False,  # GP requires complete data
            'nan_tolerance': 0.0,
        },
        'hyperparameters': {
            # Gaussian Process Regressor hyperparameters
            'kernel': variant.get('kernel', 'matern52+linear'),
            'ard': True,
            # Above this many dimensions per training sample, ARD is dropped for a
            # single isotropic lengthscale rather than left unfitted.
            'ard_min_samples_per_dim': 1.0,
            'input_standardize': True,
            'target_standardize': True,
            'use_uncertain_input_kernel': True,
            'uncertain_kernel_mc_samples': 64,
            'uncertain_kernel_mc_seed': 0,
            'uncertainty_source_mode': 'aggregate_t',
            'uncertainty_summary_dir': None,
            'uncertainty_aggregate_csv': None,
            'early_stop_metric': 'val_rmse',
            'early_stop_alpha': 0.5,
            'lengthscale_min': 0.05,
            'lengthscale_max': 10.0,
            'noise_min': 1e-5,
            'noise_max': 1.0,
            'outputscale_prior': {'type': 'gamma', 'shape': 2.0, 'rate': 0.15},
            'noise_prior': {'type': 'gamma', 'shape': 1.5, 'rate': 0.5},
            'learning_rate': 0.01,
            'num_epochs': 250,
            'patience': 20,
            'max_train_size': 5000,
        },
        '_comments': {
            'model_type': "This is a Gaussian Process Regressor model configuration",
            'hyperparameters': "GP-specific hyperparameters",
            'data_dir': "Base dataset directory, resolved relative to this config file location.",
            'sample_subdir': "Defaults to 'samples' so GP trains on unperturbed baseline samples.",
            'input_row_2': f"Currently {sample_length - 1} (all rows except last). Can be adjusted.",
            'output_rows': f"Currently [sample_length - 1] to predict last row only. Can be adjusted.",
            'split_type': "Temporal split prevents data leakage. If MC replicates detected, temporal split is auto-enforced.",
            'fault_tolerant': "False - GP requires complete data without NaN in inputs",
            'nan_tolerance': "0.0 for strict no-NaN predictor policy.",
            'kernel': "Use 'matern52+linear' (or 'rbf+linear') with use_uncertain_input_kernel=True to add a linear trend term while keeping uncertainty-aware kernels.",
            'uncertainty_source_mode': "'aggregate_t' matches MC replicate assumptions using aggregate offset fits; falls back to summary std if unavailable.",
            'uncertain_kernel_mc_samples': "MC samples used to approximate expected kernel under input uncertainty (higher = smoother but slower).",
            'uncertain_kernel_mc_seed': "Seed for deterministic uncertain-kernel sampling so train/eval remain aligned.",
            'early_stop_metric': "Early stopping metric: 'train_nll', 'val_rmse', or 'mixed' (alpha-weighted).",
            'early_stop_alpha': "Weight for train_nll in mixed early stopping (0-1).",
            'lengthscale_min': "Lower bound on GP lengthscales to prevent collapse.",
            'lengthscale_max': "Upper bound on GP lengthscales to prevent flat kernels.",
            'noise_min': "Lower bound on likelihood noise.",
            'noise_max': "Upper bound on likelihood noise to prevent degenerate high-noise fits.",
            'outputscale_prior': "Gamma prior on kernel outputscale (shape, rate). Set to null to disable.",
            'noise_prior': "Gamma prior on likelihood noise (shape, rate). Set to null to disable.",
        }
    }

    if overrides:
        config['hyperparameters'].update(overrides.get('hyperparameters', {}))
        for key in ('model_type', 'model_name', 'device', 'matplotlib_backend'):
            if key in overrides:
                config[key] = overrides[key]
        if 'data' in overrides:
            config['data'].update(overrides['data'])
        if 'data_split' in overrides:
            config['data_split'].update(overrides['data_split'])

    config['_comments']['variant'] = variant.get('note', '')
    config_path = Path(output_dir) / f'config_gp_{_suffix}.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return str(config_path)


def generate_recurrent_transformer_config_template(
    output_dir, forecast_name, input_columns, state_column, sample_length, overrides=None
):
    """
    Generate a template configuration file for e_TrainRecurrent.py (RecurrentTransformer).

    :param output_dir: Output directory where config will be saved
    :param forecast_name: Name of the forecast/dataset
    :param input_columns: List of input column names (must include state_column)
    :param state_column: Column used as both an input feature and the prediction target
    :param sample_length: Length of each sample (number of rows)
    :return: Path to the generated config file

    Notes:
    - input_row_2 is sample_length (all rows), not sample_length - 1.
      The leakage guard (zeroing state_column in the last row) is applied
      inside RecurrentTimeSeriesDataset, not at the data-loading level.
    - Uses sample_subdir='samples' (unperturbed baseline); MC replicates are
      not needed for recurrent models.
    - Run with e_TrainRecurrent.py, not e_Train.py.
    """
    config = {
        'model_type': 'recurrent_transformer',
        'model_name': 'model_recurrent_transformer_01',
        'device': 'cuda',
        'matplotlib_backend': 'Agg',
        'data': {
            'data_dir': '.',
            'sample_subdir': 'samples',
            'forecast_name': forecast_name,
            'input_columns': input_columns,
            'state_column': state_column,
            'input_row_1': 0,
            'input_row_2': sample_length,
            'output_columns': [state_column],
            'output_rows': [sample_length - 1],
        },
        'data_split': {
            'random_state': 42,
            'test_size': 0.3,
            'reuse_split': False,
            'split_source': None,
            'split_type': 'temporal',
            'fault_tolerant': False,
            'nan_tolerance': 0.0,
        },
        'hyperparameters': {
            'num_epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'patience': 20,
            'model_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'cv_tuning': {
                'enabled': True,
                'n_folds': 5,
                'n_trials': 20,
                'cv_epochs': 50,
                'cv_patience': 10,
                'param_space': {
                    'model_dim': [64, 128, 256],
                    'num_heads': [2, 4, 8],
                    'num_layers': {'low': 2, 'high': 6, 'type': 'int'},
                    'dropout': {'low': 0.05, 'high': 0.4},
                    'learning_rate': {'low': 0.0005, 'high': 0.05, 'log': True},
                    'batch_size': [8, 16, 32],
                },
            },
        },
        '_comments': {
            'model_type': "Run with e_TrainRecurrent.py, not e_Train.py",
            'state_column': "Column used as both input feature and prediction target (last row value)",
            'input_row_2': f"Includes all {sample_length} rows; leakage guard applied inside RecurrentTimeSeriesDataset",
            'output_rows': f"[{sample_length - 1}] — last row index (0-based) is the target",
            'sample_subdir': "Uses unperturbed baseline samples; MC replicates not needed for recurrent models",
            'fault_tolerant': "False — recurrent models require complete data without NaN in inputs",
        },
    }

    if overrides:
        config['hyperparameters'].update(overrides.get('hyperparameters', {}))
        for key in ('model_type', 'model_name', 'device', 'matplotlib_backend'):
            if key in overrides:
                config[key] = overrides[key]
        if 'data' in overrides:
            config['data'].update(overrides['data'])
        if 'data_split' in overrides:
            config['data_split'].update(overrides['data_split'])

    config_path = Path(output_dir) / 'config_recurrent_transformer_01.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return str(config_path)


def generate_lstm_config_template(
    output_dir, forecast_name, input_columns, state_column, sample_length, overrides=None
):
    """
    Generate a template configuration file for e_TrainRecurrent.py (LSTM).

    :param output_dir: Output directory where config will be saved
    :param forecast_name: Name of the forecast/dataset
    :param input_columns: List of input column names (must include state_column)
    :param state_column: Column used as both an input feature and the prediction target
    :param sample_length: Length of each sample (number of rows)
    :return: Path to the generated config file

    Notes:
    - Same structural differences from existing models as recurrent_transformer.
    - LSTM-specific hyperparameters: hidden_dim instead of model_dim/num_heads.
    - Run with e_TrainRecurrent.py, not e_Train.py.
    """
    config = {
        'model_type': 'lstm',
        'model_name': 'model_lstm_01',
        'device': 'cuda',
        'matplotlib_backend': 'Agg',
        'data': {
            'data_dir': '.',
            'sample_subdir': 'samples',
            'forecast_name': forecast_name,
            'input_columns': input_columns,
            'state_column': state_column,
            'input_row_1': 0,
            'input_row_2': sample_length,
            'output_columns': [state_column],
            'output_rows': [sample_length - 1],
        },
        'data_split': {
            'random_state': 42,
            'test_size': 0.3,
            'reuse_split': False,
            'split_source': None,
            'split_type': 'temporal',
            'fault_tolerant': False,
            'nan_tolerance': 0.0,
        },
        'hyperparameters': {
            'num_epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'patience': 20,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'cv_tuning': {
                'enabled': True,
                'n_folds': 5,
                'n_trials': 20,
                'cv_epochs': 50,
                'cv_patience': 10,
                'param_space': {
                    'hidden_dim': [32, 64, 128, 256],
                    'num_layers': {'low': 1, 'high': 4, 'type': 'int'},
                    'dropout': {'low': 0.0, 'high': 0.5},
                    'learning_rate': {'low': 0.0005, 'high': 0.05, 'log': True},
                    'batch_size': [8, 16, 32],
                },
            },
        },
        '_comments': {
            'model_type': "Run with e_TrainRecurrent.py, not e_Train.py",
            'state_column': "Column used as both input feature and prediction target (last row value)",
            'input_row_2': f"Includes all {sample_length} rows; leakage guard applied inside RecurrentTimeSeriesDataset",
            'output_rows': f"[{sample_length - 1}] — last row index (0-based) is the target",
            'sample_subdir': "Uses unperturbed baseline samples; MC replicates not needed for recurrent models",
            'fault_tolerant': "False — recurrent models require complete data without NaN in inputs",
            'hidden_dim': "LSTM hidden state size (replaces model_dim/num_heads from Transformer config)",
        },
    }

    if overrides:
        config['hyperparameters'].update(overrides.get('hyperparameters', {}))
        for key in ('model_type', 'model_name', 'device', 'matplotlib_backend'):
            if key in overrides:
                config[key] = overrides[key]
        if 'data' in overrides:
            config['data'].update(overrides['data'])
        if 'data_split' in overrides:
            config['data_split'].update(overrides['data_split'])

    config_path = Path(output_dir) / 'config_lstm_01.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return str(config_path)


def load_sensor_uncertainty_summary(sensor_name, corrections_dir):
    """
    Load sensor uncertainty summary for Monte Carlo sampling.
    Returns dict with distribution parameters, or None if not found.
    """
    summary_path = corrections_dir / sensor_name / f'{sensor_name}_uncertainty_summary.csv'
    if not summary_path.exists():
        return None
    
    try:
        summary_df = pd.read_csv(summary_path)
        return summary_df.iloc[0].to_dict()
    except Exception as e:
        print(f"  Warning: Could not load uncertainty summary for {sensor_name}: {e}")
        return None


def sample_from_distribution(
    mean,
    std,
    distribution_type='normal',
    size=1,
    t_df=None,
    t_loc=None,
    t_scale=None,
):
    """
    Sample from a distribution given mean/std or explicit Student's t parameters.
    """
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return np.zeros(size) if size > 1 else 0
    
    if distribution_type == 'normal' or distribution_type == 'equivalent':
        return np.random.normal(mean, std, size)
    elif distribution_type == 't':
        if (
            t_df is None
            or t_loc is None
            or t_scale is None
            or (not np.isfinite(t_df))
            or (not np.isfinite(t_loc))
            or (not np.isfinite(t_scale))
            or t_scale <= 0
        ):
            raise ValueError(
                f"Missing/invalid Student's t params: df={t_df}, loc={t_loc}, scale={t_scale}"
            )
        return stats.t.rvs(df=t_df, loc=t_loc, scale=t_scale, size=size)
    else:
        raise ValueError(f"Unsupported distribution_type={distribution_type!r}")


def normalize_uncertainty_params(params, col_name, norm_params):
    """
    Transform uncertainty parameters from raw scale to normalized scale.
    
    If offset ~ N(mu, sigma) in raw scale,
    then normalized_offset ~ N(mu/range, sigma/range) where range = max - min
    """
    if col_name not in norm_params:
        return params  # No normalization available
    
    norm_spec = norm_params[col_name]
    v_min = norm_spec.get('min', 0)
    v_max = norm_spec.get('max', 1)
    v_range = v_max - v_min
    
    if v_range == 0:
        return params
    
    # Create a copy so we don't modify original
    params_normalized = params.copy()
    
    # Scale additive components (offset, noise) by range
    for key in ['Offset_Mean', 'Offset_Std', 'Noise_Std_Mean', 'Noise_Std_Std', 'Offset_t_loc', 'Offset_t_scale']:
        if key in params_normalized:
            params_normalized[key] = params_normalized.get(key, 0) / v_range
    
    # Gain is multiplicative, so it's unchanged
    # Gain affects (value * (1 + gain)), so the gain percentage stays the same
    
    return params_normalized


def apply_uncertainty_perturbation(segment_df, sensor_uncertainties, random_seed=None):
    """
    Apply Monte Carlo uncertainty perturbations to a segment.
    
    Simple model: perturbed = measured + offset
    
    Where offset is drawn once per replicate from the Correction1 distribution
    for each sensor (based on calibration event analysis).
    
    sensor_uncertainties: dict mapping sensor_name -> uncertainty_summary_dict
    
    Returns: perturbed DataFrame
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    df_perturbed = segment_df.copy()
    
    # Sensor column mappings
    sensor_column_map = {
        'Sp Cond (microS_cm)': 'Pfl - Sp Cond (microS_cm)',
        'pH': 'Pfl - pH',
        'DO (% Sat)': 'Pfl - DO (% Sat)',
        'Turbidity (FNU)': 'Pfl - Turbidity (FNU)',
        'fDOM (RFU)': 'Pfl - fDOM (RFU)',
        'fDOM (QSU)': 'Pfl - fDOM (QSU)',
    }
    
    for sensor_key, column_name in sensor_column_map.items():
        if column_name not in df_perturbed.columns:
            continue
        
        if sensor_key not in sensor_uncertainties or sensor_uncertainties[sensor_key] is None:
            continue
        
        uncertainty_info = sensor_uncertainties[sensor_key]
        
        # Draw offset once per replicate from Offset distribution (based on Correction1)
        offset = sample_from_distribution(
            uncertainty_info.get('Offset_Mean', 0),
            uncertainty_info.get('Offset_Std', 0),
            uncertainty_info.get('Offset_Distribution', 'normal'),
            size=1,
            t_df=uncertainty_info.get('Offset_t_df'),
            t_loc=uncertainty_info.get('Offset_t_loc'),
            t_scale=uncertainty_info.get('Offset_t_scale'),
        )[0]
        
        mask = df_perturbed[column_name].notna()
        if mask.any():
            df_perturbed.loc[mask, column_name] = df_perturbed.loc[mask, column_name] + offset
    
    return df_perturbed


def analyze_valid(df, targets, predictors, span, valid, name="FaultTolerantSampleSize"):
    """
    Summarize availability for a single span using only target validity and contiguous segment checks.
    targets: list of target columns to evaluate independently.
    span: integer window size in rows.
    valid: retained for backward compatibility (not used).
    """
    plt.figure(figsize=(12, 8))

    results = []
    for col in targets:
        count = 0
        for i in range(span - 1, len(df)):
            segment = df.iloc[i - span + 1:i + 1]
            if pd.notnull(segment.iloc[-1][col]) and segment["Segment"].nunique() == 1:
                count += 1
        results.append((col, count))

    labels, counts = zip(*results) if results else ([], [])
    plt.bar(labels, counts)
    plt.xlabel("Target column")
    plt.ylabel(f"Number of {span}-hour Samples")
    plt.title("Sample size with valid target and contiguous segment")
    plt.xticks(rotation=45, ha="right")

    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join("../data/output/regression/availability", name + ".png"))
    # plt.show()

def gapless(df, target_columns, name="length_v_count_analysis"):
    """
    Evaluate # of samples with time series leading directly into the Eurofin sample time (for "nowcast").
    Plot results for each parameter.
    :return:
    """

    # Initialize a dictionary to store results
    results = {col: [] for col in target_columns}

    # Loop through each variable and segment length
    for col in target_columns:
        for seg_len in range(10, 169, 25):
            count = 0
            for i in range(seg_len - 1, len(df)):
                segment = df.iloc[i - seg_len + 1:i + 1]
                if pd.notnull(segment.iloc[-1][col]) and segment["Segment"].nunique() == 1:
                    count += 1
            results[col].append((seg_len, count))

    # Plotting
    plt.figure(figsize=(12, 8))
    for col in target_columns:
        lengths, counts = zip(*results[col])
        plt.plot(lengths, counts, '--', label=col)
        print(col, max(counts))

    plt.xlabel("Segment Length")
    plt.ylabel("Number of Valid Segments")
    plt.title("Segment Length vs. Number of Valid Segments for Each Variable")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("../data/output/regression/availability", name + ".png"))
    # plt.show()

def gapped(df, target_columns, seg_length, name="length_v_count_analysis"):
    """
    Evaluate a range of gaps from time series to Eurofins values.
    :return:
    """
    os.makedirs(os.path.join("..data/output/regression/availability"), exist_ok=True)

    results_df = pd.DataFrame({"Gap Hours", "Valid Segments", "Variable Name"})
    for column in target_columns:
        # Initialize a dictionary to store results
        gapults = {gap: 0 for gap in range(0, 167, 5)}
        # print(column)
        # Iterate over each row with a non-null value in the last column
        for idx, row in df[df[column].notna()].iterrows():
            current_time = row["TIMESTAMP"]
            current_segment = row["Segment"]
            for gap in range(0, 167, 5):
                target_time = current_time - pd.Timedelta(hours=gap)
                # Find the first row before current_time that matches target_time
                preceding_rows = df[(df["TIMESTAMP"] <= target_time)].sort_values(by="TIMESTAMP", ascending=False)
                if not preceding_rows.empty:
                    first_row = preceding_rows.iloc[0]
                    segment_value = first_row["Segment"]
                    segment_rows = df[(df["TIMESTAMP"] <= first_row["TIMESTAMP"]) & (df["Segment"] == segment_value)]
                    if len(segment_rows) >= seg_length:
                        gapults[gap] += 1

        result = pd.DataFrame({
            "Gap Hours": list(gapults.keys()),
            "Valid Segments": list(gapults.values()),
            "Variable Name": column,
        })

        # Append results to DataFrame for plotting
        results_df = pd.concat([results_df, result], ignore_index=True)

    fig = px.line(results_df, x="Gap Hours", y="Valid Segments", color="Variable Name",
                  title=f"Effect of Gap on Valid Segments", markers=True)
    fig.write_image(os.path.join("../data/output/regression/availability", name + ".png"))

def split(df, output_dir, target_columns=['01-Farge', '04-Turbiditet', '06-E.coli',
                        '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen',
                        '24-Bly', '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)'],
          length=DEFAULT_SAMPLE_LENGTH_ROWS, nan_tol=0.0, to_normalize=[],fault_tolerant=False, offset=0,
          predictor_cols=['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
                        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                        'Pfl - fDOM (QSU)', "Wind speed x (m/s)", "Wind speed y (m/s)",
                        'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                        'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                        '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)'],
          use_uncertainty_perturbation=False, n_mc_replicates=10, random_seed=21,
          pre_normalized=False, normalization_params=None, sensor_uncertainties=None,
          verbose=False, gap_rows=0, training_config_defaults=None):
    """
    Break up a dataset which contains gaps into many files of standard size, which do not contain gaps.

    :param df: dataframe of consolidated dataset to be broken up
    :param output_dir: where to save the output files
    :param length: number of predictor rows per sample
    :param gap_rows: rows between the last predictor row and the target row (default 0).
        Each segment CSV always contains exactly `length` rows. When gap_rows > 0, the
        predictor window is shifted `gap_rows` rows earlier; the last row of each segment
        has its TIMESTAMP and target-column values replaced with those from the target row
        (gap_rows later), while predictor column values come from the earlier row.
        The contiguity of the predictor window is checked via the Segment column when present.
    :param use_uncertainty_perturbation: if True, generate K Monte Carlo replicates with uncertainty
    :param n_mc_replicates: number of Monte Carlo replicates to generate (K)
    :param random_seed: seed for reproducibility
    :param training_config_defaults: optional dict with keys 'xgb', 'transformer', 'gp'; values
        are merged into the corresponding generated template configs.
    :param nan_tol: deprecated here; NaN filtering is handled in e_Train.py
    :param fault_tolerant: deprecated here; NaN filtering is handled in e_Train.py
    :return:
    """
    if training_config_defaults is None:
        training_config_defaults = {}
    if pre_normalized:
        df = df.copy()
        if normalization_params is not None:
            _write_normalization_params(
                normalization_params,
                output_path=Path(output_dir) / "normalization.json",
            )
        else:
            # Fallback: preserve old behavior if caller did not provide params.
            df = normalize_columns(
                df,
                to_normalize,
                param_file=None,
                min_val=0,
                max_val=1,
                save=True,
                directory=NORMALIZATION_OUTPUT_PATH.parent,
            )
    else:
        df = normalize_columns(
            df,
            to_normalize,
            param_file=None,
            min_val=0,
            max_val=1,
            save=True,
            directory=NORMALIZATION_OUTPUT_PATH.parent,
        )

    samples_dir = os.path.join(output_dir, 'samples')
    perturbed_samples_dir = os.path.join(output_dir, 'mc_replicates')

    os.makedirs(samples_dir, exist_ok=True)  # Create directory for unperturbed output files
    clean_directory(samples_dir)

    if use_uncertainty_perturbation:
        os.makedirs(perturbed_samples_dir, exist_ok=True)  # Create directory for perturbed MC output files
        clean_directory(perturbed_samples_dir)

    # Load sensor uncertainty summaries if perturbation is enabled
    if use_uncertainty_perturbation:
        if sensor_uncertainties is None:
            sensor_uncertainties = _load_and_prepare_sensor_uncertainties(
                output_dir,
                normalization_params=normalization_params,
                verbose=verbose,
            )

        # Set random seed once for reproducibility
        np.random.seed(random_seed)
        if verbose:
            print(f"Monte Carlo sampling enabled: K={n_mc_replicates} replicates, seed={random_seed}\n")

    # Initialize a counter for naming output files
    segment_counter = 1
    metadata_cols = [col for col in ["TIMESTAMP", "Segment", "Interpolated"] if col in df.columns]
    predictor_write_cols = [col for col in predictor_cols if col in df.columns]
    target_write_cols = [col for col in target_columns if col in df.columns]
    sample_columns = metadata_cols + predictor_write_cols + [
        col for col in target_write_cols if col not in metadata_cols and col not in predictor_write_cols
    ]

    if not predictor_write_cols:
        if verbose:
            print("[WARN] None of predictor_cols were found in dataframe; writing targets/metadata only.")
    if not target_write_cols:
        raise ValueError(
            f"None of target_columns found in dataframe for output_dir={output_dir}: {target_columns}"
        )

    n_rows = len(df)
    has_segment_col = "Segment" in df.columns
    min_rows_needed = length + gap_rows
    if n_rows < min_rows_needed:
        sample_count = 0
        xgb_config_paths = generate_xgb_config_templates(
            output_dir,
            predictor_cols,
            target_columns,
            length,
            overrides=training_config_defaults.get('xgb', {}),
        )
        xgb_config_path = xgb_config_paths[0]
        transformer_config_paths = generate_transformer_config_templates(
            output_dir,
            predictor_cols,
            target_columns,
            length,
            overrides=training_config_defaults.get('transformer', {}),
        )
        transformer_config_path = transformer_config_paths[0]
        gp_config_paths = generate_gp_config_templates(
            output_dir,
            predictor_cols,
            target_columns,
            length,
            overrides=training_config_defaults.get('gp', {}),
        )
        gp_config_path = gp_config_paths[0]
        state_col = next((c for c in predictor_cols if c.endswith("_state")), None)
        if state_col is None:
            if verbose:
                print("[INFO] No _state column found in predictor_cols; skipping recurrent model configs.")
            recurrent_transformer_config_path = None
            lstm_config_path = None
        else:
            recurrent_transformer_config_path = generate_recurrent_transformer_config_template(
                output_dir,
                'recurrent_transformer_01',
                predictor_cols,
                state_col,
                length,
                overrides=training_config_defaults.get('recurrent_transformer', {}),
            )
            lstm_config_path = generate_lstm_config_template(
                output_dir,
                'lstm_01',
                predictor_cols,
                state_col,
                length,
                overrides=training_config_defaults.get('lstm', {}),
            )
        all_config_paths = [
            p for p in [
                xgb_config_path, transformer_config_path, gp_config_path,
                recurrent_transformer_config_path, lstm_config_path,
            ]
            if p is not None
        ]
        return {
            "sample_set_name": Path(output_dir).name,
            "target_columns": list(target_columns),
            "predictor_columns": list(predictor_cols),
            "n_samples": sample_count,
            "config_paths": all_config_paths,
        }

    # Target rows: every row from index (min_rows_needed - 1) onward is a candidate.
    # With gap_rows=0 this is identical to the previous behaviour.
    target_valid = df[target_columns].notnull().all(axis=1).to_numpy()
    candidate_target_indices = np.arange(min_rows_needed - 1, n_rows)
    valid_target_indices = candidate_target_indices[target_valid[candidate_target_indices]]

    for target_idx in valid_target_indices:
        # Predictor window: length rows ending gap_rows before the target
        pred_end_idx = int(target_idx - gap_rows)
        pred_start_idx = int(pred_end_idx - length + 1)

        predictor_window = df.iloc[pred_start_idx:pred_end_idx + 1]

        # Check predictor-window contiguity when Segment column is present
        if has_segment_col and predictor_window["Segment"].nunique() > 1:
            continue

        # Build the segment: predictor_window rows, then compose the last row
        if gap_rows > 0:
            # Compose the last row: predictor values from pred_end row, but
            # TIMESTAMP and target-column values from the actual target row.
            segment_rows = predictor_window.copy()
            target_row = df.iloc[target_idx]
            last_row = segment_rows.iloc[-1].copy()
            if "TIMESTAMP" in df.columns:
                last_row["TIMESTAMP"] = target_row["TIMESTAMP"]
            for col in target_write_cols:
                if col in last_row.index:
                    last_row[col] = target_row[col]
            segment_rows.iloc[-1] = last_row
            segment = segment_rows
        else:
            segment = predictor_window

        segment_out = segment.loc[:, sample_columns]

        output_file = os.path.join(samples_dir, f"segment_{segment_counter:04d}.csv")
        segment_out.to_csv(output_file, index=False)

        if use_uncertainty_perturbation:
            for k in range(1, n_mc_replicates + 1):
                replicate_seed = random_seed + k
                segment_perturbed = apply_uncertainty_perturbation(
                    segment_out, sensor_uncertainties, random_seed=replicate_seed
                )
                output_file = os.path.join(
                    perturbed_samples_dir,
                    f"segment_{segment_counter:04d}_mc_{k:03d}.csv"
                )
                segment_perturbed.to_csv(output_file, index=False)

        segment_counter += 1

    n_samples = segment_counter - 1

    # Generate template configuration files for e_Train.py
    xgb_config_paths = generate_xgb_config_templates(
        output_dir,
        predictor_cols,
        target_columns,
        length,
        overrides=training_config_defaults.get('xgb', {}),
    )
    xgb_config_path = xgb_config_paths[0]
    transformer_config_paths = generate_transformer_config_templates(
        output_dir,
        predictor_cols,
        target_columns,
        length,
        overrides=training_config_defaults.get('transformer', {}),
    )
    transformer_config_path = transformer_config_paths[0]
    gp_config_paths = generate_gp_config_templates(
        output_dir,
        predictor_cols,
        target_columns,
        length,
        overrides=training_config_defaults.get('gp', {}),
    )
    gp_config_path = gp_config_paths[0]
    state_col = next((c for c in predictor_cols if c.endswith("_state")), None)
    if state_col is None:
        if verbose:
            print("[INFO] No _state column found in predictor_cols; skipping recurrent model configs.")
        recurrent_transformer_config_path = None
        lstm_config_path = None
    else:
        recurrent_transformer_config_path = generate_recurrent_transformer_config_template(
            output_dir,
            'recurrent_transformer_01',
            predictor_cols,
            state_col,
            length,
            overrides=training_config_defaults.get('recurrent_transformer', {}),
        )
        lstm_config_path = generate_lstm_config_template(
            output_dir,
            'lstm_01',
            predictor_cols,
            state_col,
            length,
            overrides=training_config_defaults.get('lstm', {}),
        )
    all_config_paths = [
        p for p in [
            xgb_config_path, transformer_config_path, gp_config_path,
            recurrent_transformer_config_path, lstm_config_path,
        ]
        if p is not None
    ]
    return {
        "sample_set_name": Path(output_dir).name,
        "target_columns": list(target_columns),
        "predictor_columns": list(predictor_cols),
        "n_samples": n_samples,
        "config_paths": all_config_paths,
    }


def _build_auto_normalize_list(predictor_cols, target_base_names, df_columns):
    """
    Build the to_normalize list automatically from predictors + target base/state/res/diff variants.
    """
    candidates = list(predictor_cols)
    for name in target_base_names:
        for suffix in ["", "_state", "_res", "_diff"]:
            col = name + suffix
            if col in df_columns:
                candidates.append(col)
    return list(dict.fromkeys(candidates))  # dedup, preserve order


def _resolve_columns_from_config(config, df_columns):
    """
    Resolve final target_columns, predictor_columns, and to_normalize list from config dict.

    Handles auto_target_suffixes.diff / res / state:
      - diff=True -> each name in target_columns becomes <name>_diff (if present in df)
      - res=True  -> each name in target_columns becomes <name>_res (if present in df)
      - state=True -> <name>_state is appended to predictors (if present in df)

    Returns (target_columns, predictor_columns, to_normalize)
    """
    df_col_set = set(df_columns)
    raw_target_names = list(config.get("target_columns", []))
    suffix_opts = config.get("auto_target_suffixes", {})
    use_diff = suffix_opts.get("diff", False)
    use_res = suffix_opts.get("res", True)

    target_columns = []
    for name in raw_target_names:
        candidate_columns = []
        if use_diff:
            candidate_columns.append(f"{name}_diff")
        if use_res:
            candidate_columns.append(f"{name}_res")
        candidate_columns.append(name)

        resolved = next((candidate for candidate in candidate_columns if candidate in df_col_set), None)
        if resolved is not None:
            target_columns.append(resolved)
        else:
            display_candidates = "', '".join(candidate_columns)
            print(f"[WARN] Target '{name}' not found as '{display_candidates}' in data. Skipping.")

    predictor_columns = list(config.get("predictor_columns", []))

    normalize_cfg = config.get("to_normalize", "auto")
    if normalize_cfg == "auto":
        to_normalize = _build_auto_normalize_list(predictor_columns, raw_target_names, df_col_set)
    else:
        to_normalize = list(normalize_cfg)

    return target_columns, predictor_columns, to_normalize


def _resolve_state_predictor_column(target_name, available_columns):
    """
    Resolve the target-linked state predictor column.

    Rules:
    - "<target>_diff" -> "<target>_state" (strip "_diff" first)
    - "<target>_res"  -> "<target>_state" (strip "_res" first)
    - "<target>"      -> "<target>_state"
    - If target is already a "_state" column, keep it as-is
    - Return None if no candidate exists in available_columns
    """
    base = str(target_name)
    for suffix in ("_diff", "_res"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    candidates = [base] if base.endswith("_state") else [f"{base}_state"]
    for candidate in candidates:
        if candidate in available_columns:
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate resampled training datasets from a consolidated sensor CSV."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to resampling configuration YAML file.",
    )
    args = parser.parse_args()

    matplotlib.use('Agg')

    config = load_config(args.config)
    config_dir = Path(config["__config_dir"])

    # --- Resolve paths ---
    input_csv = Path(config["input_csv"])
    if not input_csv.is_absolute():
        input_csv = (config_dir / input_csv).resolve()

    base_output_dir = Path(config["output_dir"])
    if not base_output_dir.is_absolute():
        base_output_dir = (config_dir / base_output_dir).resolve()

    set_name_template = config.get("set_name_template", "MC_{target_slug}")
    sample_length_cfg = config.get("sample_length", "auto")
    fallback_length = int(config.get("sample_length_fallback", DEFAULT_SAMPLE_LENGTH_ROWS))
    gap_rows = int(config.get("gap_rows", 0))
    use_uncertainty_perturbation = bool(config.get("use_uncertainty_perturbation", True))
    n_mc_replicates = int(config.get("n_mc_replicates", 10))
    random_seed = int(config.get("random_seed", 1))
    verbose = bool(config.get("verbose", False))
    training_config_defaults = config.get("training_config_defaults", {})

    eurofins_path_cfg = config.get("eurofins_summary_path", None)
    eurofins_summary_path = (
        Path(eurofins_path_cfg) if eurofins_path_cfg else EUROFINS_SUMMARY_DEFAULT_PATH
    )
    if eurofins_path_cfg and not Path(eurofins_path_cfg).is_absolute():
        eurofins_summary_path = (config_dir / eurofins_path_cfg).resolve()

    # --- Load data ---
    print(f"[INFO] Loading data from {input_csv}")
    df = pd.read_csv(input_csv, parse_dates=["TIMESTAMP"])
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)

    # --- Resolve columns ---
    target_columns, predictor_cols, to_normalize = _resolve_columns_from_config(config, df.columns)
    use_state = config.get("auto_target_suffixes", {}).get("state", True)
    if not target_columns:
        print("[ERROR] No valid target columns found. Exiting.")
        sys.exit(1)

    # --- Normalize once ---
    df_norm, normalization_params = _normalize_once(df, to_normalize, min_val=0, max_val=1)

    # --- Load shared sensor uncertainties ---
    shared_sensor_uncertainties = None
    if use_uncertainty_perturbation:
        shared_sensor_uncertainties = _load_and_prepare_sensor_uncertainties(
            output_dir=str(base_output_dir),
            normalization_params=normalization_params,
            verbose=verbose,
        )

    # --- Eurofins lookup for per-target sample lengths ---
    eurofins_summary_df = None
    if sample_length_cfg == "auto":
        try:
            eurofins_summary_df = _load_eurofins_summary_df(eurofins_summary_path)
        except Exception as exc:
            print(
                f"[WARN] Could not load Eurofins summary table: {exc}. "
                f"Using fallback sample length={fallback_length} for all targets."
            )

    # --- Per-target loop ---
    for target in target_columns:
        target_slug = re.sub(r"[^\w]", "_", target)
        output_dir = str(base_output_dir / set_name_template.format(target_slug=target_slug))

        if sample_length_cfg == "auto":
            if eurofins_summary_df is None:
                target_length = fallback_length
            else:
                target_length = _get_target_sample_length_rows(
                    target,
                    eurofins_summary_df,
                    default_rows=fallback_length,
                    verbose=True,
                )
        else:
            target_length = int(sample_length_cfg)

        available_cols = set(df_norm.columns)
        per_target_predictors = [c for c in predictor_cols if c in available_cols]
        if use_state:
            state_col = _resolve_state_predictor_column(target, available_cols)
            if state_col and state_col not in per_target_predictors:
                per_target_predictors.append(state_col)

        missing_predictors = [c for c in predictor_cols if c not in available_cols]
        if missing_predictors:
            print(
                f"[WARN] Dropping missing predictors for target '{target}': "
                + ", ".join(missing_predictors)
            )

        print(f"[INFO] Target '{target}': sample_length={target_length}, gap_rows={gap_rows}")
        result = split(
            df_norm,
            output_dir,
            [target],
            target_length,
            0.8,
            to_normalize,
            True,
            predictor_cols=per_target_predictors,
            use_uncertainty_perturbation=use_uncertainty_perturbation,
            n_mc_replicates=n_mc_replicates,
            random_seed=random_seed,
            pre_normalized=True,
            normalization_params=normalization_params,
            sensor_uncertainties=shared_sensor_uncertainties,
            verbose=verbose,
            gap_rows=gap_rows,
            training_config_defaults=training_config_defaults,
        )

        print(f"  Sample set : {result['sample_set_name']}")
        print(f"  Targets    : {result['target_columns']}")
        print(f"  Predictors : {len(result['predictor_columns'])} columns")
        print(f"  N samples  : {result['n_samples']}")
        for cfg_path in result['config_paths']:
            print(f"  Config     : {cfg_path}")


if __name__ == '__main__':
    main()

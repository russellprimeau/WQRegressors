'''
Analyze the combined dataset to identify commonly-sized chunks of data which can be used for forecasting over
different horizons.
Then, split the dataset into files for each equivalent sample.
'''

import os
import json
import re
import pandas as pd
import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from pathlib import Path
from utils.preprocessing import normalize_columns


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
    if text.endswith("_res"):
        text = text[:-4]
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


def generate_training_config_template(output_dir, forecast_name, input_columns, output_columns, sample_length, model_type='xgb'):
    """
    Generate a template configuration file for e_Train.py.
    
    :param output_dir: Output directory where config will be saved
    :param forecast_name: Name of the forecast/dataset (used as model_name base)
    :param input_columns: List of input column names
    :param output_columns: List of output column names
    :param sample_length: Length of each sample (number of rows)
    :param model_type: Type of model for naming ('xgb' or 'transformer')
    :return: Path to the generated config file
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
            'n_estimators': 1100,
            'max_depth': 10,
            'subsample': 0.2,
            'colsample_bytree': 0.8,
            'learning_rate': 0.01,
            'n_jobs': -1,
            'early_stopping_rounds': 200,
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
        }
    }
    
    # Save as YAML
    config_path = Path(output_dir) / f'config_{model_type}_01.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return str(config_path)


def generate_transformer_config_template(output_dir, forecast_name, input_columns, output_columns, sample_length):
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
            # Composite objective: combined_loss = mse_loss - corr_lambda * pearson_corr
            'corr_lambda': 0.1,
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
    
    # Save as YAML
    config_path = Path(output_dir) / f'config_transformer_01.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return str(config_path)


def generate_gp_config_template(output_dir, forecast_name, input_columns, output_columns, sample_length):
    """
    Generate a template configuration file for Gaussian Process Regressor models.

    :param output_dir: Output directory where config will be saved
    :param forecast_name: Name of the forecast/dataset
    :param input_columns: List of input column names
    :param output_columns: List of output column names
    :param sample_length: Length of each sample (number of rows)
    :return: Path to the generated config file
    """
    config = {
        'model_type': 'gp_regressor',
        'model_name': 'model_gp_01',
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
            'kernel': 'matern52',
            'ard': True,
            'input_standardize': True,
            'target_standardize': True,
            'use_uncertain_input_kernel': True,
            'uncertain_kernel_mc_samples': 64,
            'uncertain_kernel_mc_seed': 0,
            'uncertainty_source_mode': 'aggregate_t',
            'uncertainty_summary_dir': None,
            'uncertainty_aggregate_csv': None,
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
            'kernel': "Use 'matern52' with use_uncertain_input_kernel=True to apply uncertainty-aware Matérn-5/2 via MC kernel expectation.",
            'uncertainty_source_mode': "'aggregate_t' matches MC replicate assumptions using aggregate offset fits; falls back to summary std if unavailable.",
            'uncertain_kernel_mc_samples': "MC samples used to approximate expected kernel under input uncertainty (higher = smoother but slower).",
            'uncertain_kernel_mc_seed': "Seed for deterministic uncertain-kernel sampling so train/eval remain aligned.",
        }
    }

    config_path = Path(output_dir) / 'config_gp_01.yml'
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
          verbose=False):
    """
    Break up a dataset which contains gaps into many files of standard size, which do not contain gaps
    :param df: dataframe of consolidated dataset to be broken up
    :param output_dir: where to save the output files
    :param length: chunk size (threshold for consecutive rows to include in each sample)
    :param use_uncertainty_perturbation: if True, generate K Monte Carlo replicates with uncertainty
    :param n_mc_replicates: number of Monte Carlo replicates to generate (K)
    :param random_seed: seed for reproducibility
    :param nan_tol: deprecated here; NaN filtering is handled in e_Train.py
    :param fault_tolerant: deprecated here; NaN filtering is handled in e_Train.py
    :return:
    """
    if pre_normalized:
        df = df.copy()
        if normalization_params is not None:
            _write_normalization_params(normalization_params)
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
    if n_rows < length:
        sample_count = 0
        xgb_config_path = generate_training_config_template(
            output_dir,
            'xgb_01',
            predictor_cols,
            target_columns,
            length,
            model_type='xgb'
        )
        transformer_config_path = generate_transformer_config_template(
            output_dir,
            'transformer_01',
            predictor_cols,
            target_columns,
            length
        )
        gp_config_path = generate_gp_config_template(
            output_dir,
            'gp_01',
            predictor_cols,
            target_columns,
            length
        )
        return {
            "sample_set_name": Path(output_dir).name,
            "target_columns": list(target_columns),
            "predictor_columns": list(predictor_cols),
            "n_samples": sample_count,
            "config_paths": [xgb_config_path, transformer_config_path, gp_config_path],
        }

    end_indices = np.arange(length - 1, n_rows)
    target_valid = df[target_columns].notnull().all(axis=1).to_numpy()
    valid_end_mask = target_valid[end_indices]

    valid_end_indices = end_indices[valid_end_mask]

    for end_idx in valid_end_indices:
        start_idx = int(end_idx - length + 1)
        segment = df.iloc[start_idx:end_idx + 1]
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

    # Generate template configuration files for e_Train.py
    # Use model type with index for forecast naming (not sample set directory name)

    # Generate XGBoost template
    xgb_config_path = generate_training_config_template(
        output_dir, 
        'xgb_01',
        predictor_cols,
        target_columns,
        length,
        model_type='xgb'
    )

    # Generate Transformer template
    transformer_config_path = generate_transformer_config_template(
        output_dir, 
        'transformer_01',
        predictor_cols,
        target_columns,
        length
    )

    # Generate GP template
    gp_config_path = generate_gp_config_template(
        output_dir,
        'gp_01',
        predictor_cols,
        target_columns,
        length
    )
    return {
        "sample_set_name": Path(output_dir).name,
        "target_columns": list(target_columns),
        "predictor_columns": list(predictor_cols),
        "n_samples": int(len(valid_end_indices)),
        "config_paths": [xgb_config_path, transformer_config_path, gp_config_path],
    }


def _resolve_state_predictor_column(target_name, available_columns):
    """
    Resolve the target-linked state predictor column.

    Rules:
    - "<target>_res"  -> "<target>_state" (strip "_res" first)
    - "<target>"      -> "<target>_state"
    - If target is already a "_state" column, keep it as-is
    - Return None if no candidate exists in available_columns
    """
    base = str(target_name)
    if base.endswith("_res"):
        base = base[:-4]

    candidates = [base] if base.endswith("_state") else [f"{base}_state"]
    for candidate in candidates:
        if candidate in available_columns:
            return candidate
    return None


if __name__ == '__main__':
    matplotlib.use('Agg')  # Non-interactive backend for file output to handle remote machine installation errors
    ## Load sensor data
    ## For binary classification (of Eurofins parameters):
    # df = pd.read_csv("../data/output/classification/Consolidated_binarized.csv",
    #                  parse_dates=["TIMESTAMP"])
    ## For regression (of any parameters:
    df = pd.read_csv("../data/output/regression/Consolidated_sparse.csv",parse_dates=["TIMESTAMP"])
    df = df.sort_values("TIMESTAMP")

    ## Identify prediction target columns. Output will only include samples with valid value in last row.
    predictor_cols  = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
                        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                        'Pfl - fDOM (QSU)', "Wind speed x (m/s)", "Wind speed y (m/s)",
                        "Atmospheric pressure (mBar)", 'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                        '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
        'SCADA - pH', 'SCADA - Temperature (°C)']
    
    target_columns = ['pH_res']

    # target_columns = ['Color_res',
    #     'Turbidity (FNU)_res', 'pH_res', 'E.coli (CFU/100mL)_res', 'Intestinal enterococci (CFU/100mL)_res', 
    #     'Colony Count 22°C (CFU/mL)_res', 'Total coliforms 37°C (CFU/100mL)_res', 'Arsenic (µg/L)_res',
    #     'Lead (µg/L)_res', 'Cadmium (µg/L)_res', 'Copper filtered (mg/L)_res', 'Chromium (µg/L)_res', 'Nickel (µg/L)_res', 
    #     'Zinc (µg/L)_res']  # alternative 1: name-based selection
    # target_columns = df.columns[-9:]  # alternative: index-based selection

    ## Alternative with better coverage
    # predictor_cols_max = ["Wind speed x (m/s)", "Wind speed y (m/s)",
    #                   'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
    #                   'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
    #                   '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
    #                   'SCADA - Temperature (°C)']
    # target_columns_max = ['06-E.coli', '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C']


    ## To analyze the impact of sample dimensions on the # of available samples:
    # gapless(df, target_columns, name="Sparse_Eurofins_availability")  # Analysis function #1
    # seg_length = 24  # fixed segment length for evaluating range of lengths of gap betweeen input and output
    # gapped(df, target_columns, seg_length)  # Analysis function #2
    # analyze_valid(df, target_columns, predictor_cols, 96, 0.1, name="Max_Input_96hr_Set")
    # analyze_valid(df, target_columns_max, predictor_cols_max, 96, 0.1, name="Max_Coverage_96hr_Set")
    # analyze_valid(df, ['09-Koliforme bakterier 37°C'], predictor_cols, 96, 0.1, name="Koli_96hr_Set")

    ## Name the dataset and select the size of each sample (# of timesteps/rows)
    set_name  = "MC_all"  # Name of subdirectory where samples will be organized
    fallback_length = DEFAULT_SAMPLE_LENGTH_ROWS  # Fallback rows if target lookup fails

    ## Select columns where values in samples will be normalized, which helps with calculating loss accurately
    # to_normalize = df.columns[3:]
    to_normalize = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
                        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                        'Pfl - fDOM (QSU)', "Wind speed x (m/s)", "Wind speed y (m/s)",
                        'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                        'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                        '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
        'SCADA - pH', 'SCADA - Temperature (°C)', 'Color', 'Turbidity (FNU)', 'pH', 'E.coli (CFU/100mL)',
        'Intestinal enterococci (CFU/100mL)', 'Colony Count 22°C (CFU/mL)', 'Total coliforms 37°C (CFU/100mL)', 'Arsenic (µg/L)',
        'Lead (µg/L)', 'Cadmium (µg/L)', 'Copper filtered (mg/L)', 'Chromium (µg/L)', 'Nickel (µg/L)', 'Zinc (µg/L)', 'Color_state',
        'Turbidity (FNU)_state', 'pH_state', 'E.coli (CFU/100mL)_state', 'Intestinal enterococci (CFU/100mL)_state', 
        'Colony Count 22°C (CFU/mL)_state', 'Total coliforms 37°C (CFU/100mL)_state', 'Arsenic (µg/L)_state',
        'Lead (µg/L)_state', 'Cadmium (µg/L)_state', 'Copper filtered (mg/L)_state', 'Chromium (µg/L)_state', 'Nickel (µg/L)_state', 
        'Zinc (µg/L)_state', 'Color_res',
        'Turbidity (FNU)_res', 'pH_res', 'E.coli (CFU/100mL)_res', 'Intestinal enterococci (CFU/100mL)_res', 
        'Colony Count 22°C (CFU/mL)_res', 'Total coliforms 37°C (CFU/100mL)_res', 'Arsenic (µg/L)_res',
        'Lead (µg/L)_res', 'Cadmium (µg/L)_res', 'Copper filtered (mg/L)_res', 'Chromium (µg/L)_res', 'Nickel (µg/L)_res', 
        'Zinc (µg/L)_res']

    # to_normalize = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
    #                     'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
    #                     'Pfl - fDOM (QSU)', "Wind speed x (m/s)", "Wind speed y (m/s)",
    #                     'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
    #                     'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
    #                     '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
    #     'SCADA - pH', 'SCADA - Temperature (°C)', '01-Farge', '04-Turbiditet', '44-pH', '06-E.coli',
    #     '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen',
    #     '24-Bly', '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)', '01-Farge_state',
    #                 '04-Turbiditet_state', '06-E.coli_state',
    # '07-Intestinale enterokokker_state', '08-Kimtall 22°C_state', '09-Koliforme bakterier 37°C_state', '21-Arsen_state',
    #          '24-Bly_state', '32-Kadmium_state', '36-Kopper filtrert_state', '37-Krom_state', '41-Nikkel_state',
    #                 'Sink (Zn)_state', '01-Farge', '04-Turbiditet', '06-E.coli',
    # '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen',
    #          '24-Bly', '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']

    # to_normalize = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
    #                     'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
    #                     'Pfl - fDOM (QSU)', "Wind speed x (m/s)", "Wind speed y (m/s)",
    #                     'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
    #                     'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
    #                     '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
    #     'SCADA - pH', 'SCADA - Temperature (°C)']
    # split(df, output_dir, target_columns, length, to_normalize, 0)

    # split(df, output_dir, target_columns, length, 0.8, to_normalize,
    #       True, predictor_cols=predictor_cols,
    #       use_uncertainty_perturbation=True, n_mc_replicates=10, random_seed=1)
    
    # Normalize once and reuse across all per-target dataset writes.
    df_norm, normalization_params = _normalize_once(df, to_normalize, min_val=0, max_val=1)
    shared_sensor_uncertainties = _load_and_prepare_sensor_uncertainties(
        output_dir="../data/output/missing",
        normalization_params=normalization_params,
        verbose=False,
    )
    try:
        eurofins_summary_df = _load_eurofins_summary_df()
    except Exception as exc:
        eurofins_summary_df = None
        print(
            "[WARN] Could not load Eurofins summary table for per-target lengths: "
            f"{exc}. Using fallback sample length={fallback_length} for all targets."
        )

    for target in target_columns:
        target_slug = target.replace(" ", "_").replace("/", "_")
        output_dir = os.path.join("../data/output/missing", f"MC_ex{target_slug}")
        if eurofins_summary_df is None:
            target_length = fallback_length
        else:
            target_length = _get_target_sample_length_rows(
                target,
                eurofins_summary_df,
                default_rows=fallback_length,
                verbose=True,
            )
        available_cols = set(df_norm.columns)
        target_state_col = _resolve_state_predictor_column(target, available_cols)
        per_target_predictors = list(predictor_cols)
        if target_state_col is not None and target_state_col not in per_target_predictors:
            per_target_predictors.append(target_state_col)

        missing_predictors = [col for col in per_target_predictors if col not in available_cols]
        if missing_predictors:
            print(
                f"[WARN] Dropping missing predictors for target '{target}': "
                + ", ".join(missing_predictors)
            )
            per_target_predictors = [col for col in per_target_predictors if col in available_cols]

        if target_state_col is None:
            print(f"[WARN] No matching state predictor column found for target '{target}'.")
        print(f"[INFO] Target '{target}' sample length (rows): {target_length}")
        result = split(
            df_norm,
            output_dir,
            [target],
            target_length,
            0.8,
            to_normalize,
            True,
            predictor_cols=per_target_predictors,
            use_uncertainty_perturbation=True,
            n_mc_replicates=10,
            random_seed=1,
            pre_normalized=True,
            normalization_params=normalization_params,
            sensor_uncertainties=shared_sensor_uncertainties,
            verbose=False,
        )

        print(f"Sample set: {result['sample_set_name']}")
        print(f"Target columns: {result['target_columns']}")
        print(f"Predictor columns: {result['predictor_columns']}")
        print(f"Number of samples included: {result['n_samples']}")
        for cfg in result['config_paths']:
            print(f"Config file generated: {cfg}")

    # for target in target_columns:
    #     target_slug = target.replace(" ", "_").replace("/", "_")
    #     output_dir = os.path.join("../data/output/regression", f"MC_ex{target_slug}")
    #     target_state_col = target.replace('', '_state') if target.endswith('') else f"{target}_state"
    #     per_target_predictors = predictor_cols + [target_state_col] if target_state_col not in predictor_cols else predictor_cols
    #     result = split(
    #         df_norm,
    #         output_dir,
    #         [target],
    #         length,
    #         0.8,
    #         to_normalize,
    #         True,
    #         predictor_cols=per_target_predictors,
    #         use_uncertainty_perturbation=True,
    #         n_mc_replicates=10,
    #         random_seed=1,
    #         pre_normalized=True,
    #         normalization_params=normalization_params,
    #         sensor_uncertainties=shared_sensor_uncertainties,
    #         verbose=False,
    #     )

    #     print(f"Sample set: {result['sample_set_name']}")
    #     print(f"Target columns: {result['target_columns']}")
    #     print(f"Predictor columns: {result['predictor_columns']}")
    #     print(f"Number of samples included: {result['n_samples']}")
    #     for cfg in result['config_paths']:
    #         print(f"Config file generated: {cfg}")

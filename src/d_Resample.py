'''
Analyze the combined dataset to identify commonly-sized chunks of data which can be used for forecasting over
different horizons.
Then, split the dataset into files for each equivalent sample.
'''

import os
import json
import pandas as pd
import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from utils.preprocessing import normalize_columns

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
            'test_size': 0.2,
            'reuse_split': False,
            'split_source': None,
            'split_type': 'temporal',
            'fault_tolerant': True,
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
            'test_size': 0.2,
            'reuse_split': False,
            'split_source': None,
            'split_type': 'temporal',
            'fault_tolerant': False,  # Transformer requires complete data
        },
        'hyperparameters': {
            # Transformer hyperparameters
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
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
            'test_size': 0.2,
            'reuse_split': False,
            'split_source': None,
            'split_type': 'temporal',
            'fault_tolerant': False,  # GP requires complete data
        },
        'hyperparameters': {
            # Gaussian Process Regressor hyperparameters
            'kernel': 'matern52',
            'ard': True,
            'input_standardize': True,
            'target_standardize': True,
            'use_uncertain_input_kernel': True,
            'uncertainty_summary_dir': None,
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


def sample_from_distribution(mean, std, distribution_type='normal', size=1):
    """
    Sample from a distribution given mean and std.
    distribution_type: 'normal' or 't' (Student's t approximated as normal for simplicity)
    """
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return np.zeros(size) if size > 1 else 0
    
    if distribution_type == 'normal' or distribution_type == 'equivalent':
        return np.random.normal(mean, std, size)
    elif distribution_type == 't':
        # Approximate t-distribution as normal (could use scipy.stats.t if needed)
        return np.random.normal(mean, std, size)
    else:
        return np.random.normal(mean, std, size)


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
    for key in ['Offset_Mean', 'Offset_Std', 'Noise_Std_Mean', 'Noise_Std_Std']:
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
            size=1
        )[0]
        
        # Apply offset to all rows in this replicate
        for idx in df_perturbed.index:
            measured = df_perturbed.loc[idx, column_name]
            
            # Skip NaN values
            if pd.isna(measured):
                continue
            
            # Apply simple offset perturbation
            perturbed_value = measured + offset
            df_perturbed.loc[idx, column_name] = perturbed_value
    
    return df_perturbed


def find_valid(df, targets, predictors, span, nan_tol):
    valid_indices = []

    for i in range(len(df)):
        # Check targets in current row
        if df.loc[i, targets].isna().any():
            continue

        # Define window for previous rows
        start = max(0, i - span)
        window = df.iloc[start:i]

        if window.empty:
            continue

        # Count non-NaN predictor values in the window
        total_values = len(window) * len(predictors)
        non_nan_values = window[predictors].notna().sum().sum()
        print(f'valid in {i}:', 100*(1 - (non_nan_values / total_values)), f'% NaN values with limit of {100*nan_tol}%')

        # Check if proportion meets threshold
        if total_values > 0 and 1- (non_nan_values / total_values) <= nan_tol:
            valid_indices.append(i)
            print(f'{i} added')


    return valid_indices

def analyze_valid(df, targets, predictors, span, valid, name="FaultTolerantSampleSize"):
    """
    Summarize availability for a single span and fault tolerance.
    targets: list of target columns to evaluate independently.
    span: integer window size in rows.
    valid: fault tolerance as a fraction (e.g., 0.1 for 10%).
    """
    plt.figure(figsize=(12, 8))

    results = []
    for col in targets:
        count = len(find_valid(df, [col], predictors, span, valid))
        results.append((col, count))

    labels, counts = zip(*results) if results else ([], [])
    plt.bar(labels, counts)
    plt.xlabel("Target column")
    plt.ylabel(f"Number of {span}-hour Samples")
    plt.title(f"Sample size at {100 * valid:.0f}% fault tolerance")
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
        gap_results = {gap: 0 for gap in range(0, 167, 5)}
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
                        gap_results[gap] += 1

        result = pd.DataFrame({
            "Gap Hours": list(gap_results.keys()),
            "Valid Segments": list(gap_results.values()),
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
          length=1, nan_tol=0.0, to_normalize=[],fault_tolerant=False, offset=0,
          predictor_cols=['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
                        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                        'Pfl - fDOM (QSU)', "Wind speed x (m/s)", "Wind speed y (m/s)",
                        'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                        'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                        '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)'],
          use_uncertainty_perturbation=False, n_mc_replicates=10, random_seed=21):
    """
    Break up a dataset which contains gaps into many files of standard size, which do not contain gaps
    :param df: dataframe of consolidated dataset to be broken up
    :param output_dir: where to save the output files
    :param length: chunk size (threshold for consecutive rows to include in each sample)
    :param use_uncertainty_perturbation: if True, generate K Monte Carlo replicates with uncertainty
    :param n_mc_replicates: number of Monte Carlo replicates to generate (K)
    :param random_seed: seed for reproducibility
    :return:
    """

    df = normalize_columns(df, to_normalize, param_file=None, min_val=0, max_val=1, save=True, directory=output_dir)

    samples_dir = os.path.join(output_dir, 'samples')
    perturbed_samples_dir = os.path.join(output_dir, 'mc_replicates')

    os.makedirs(samples_dir, exist_ok=True)  # Create directory for unperturbed output files
    clean_directory(samples_dir)

    if use_uncertainty_perturbation:
        os.makedirs(perturbed_samples_dir, exist_ok=True)  # Create directory for perturbed MC output files
        clean_directory(perturbed_samples_dir)

    # Load sensor uncertainty summaries if perturbation is enabled
    sensor_uncertainties = {}
    if use_uncertainty_perturbation:
        print("Loading sensor uncertainty summaries for Monte Carlo sampling...")
        corrections_dir = Path(__file__).parent.parent / "data" / "output" / "calibration" / "summaries"
        sensor_names = ['Sp Cond (microS_cm)', 'pH', 'DO (% Sat)', 'Turbidity (FNU)', 'fDOM (RFU)', 'fDOM (QSU)']
        for sensor_name in sensor_names:
            summary = load_sensor_uncertainty_summary(sensor_name, corrections_dir)
            if summary is not None:
                sensor_uncertainties[sensor_name] = summary
                print(f"  OK: Loaded {sensor_name}")
            else:
                print(f"  X: No uncertainty summary found for {sensor_name}")
        
        # Load normalization parameters and transform uncertainty parameters to normalized scale
        print("\nTransforming uncertainty parameters to normalized scale...")
        normalization_file = Path(output_dir) / "normalization.json"
        try:
            with open(normalization_file, 'r') as f:
                norm_params = json.load(f)
            
            # Map sensor names to column names for normalization lookup
            sensor_column_map = {
                'Sp Cond (microS_cm)': 'Pfl - Sp Cond (microS_cm)',
                'pH': 'Pfl - pH',
                'DO (% Sat)': 'Pfl - DO (% Sat)',
                'Turbidity (FNU)': 'Pfl - Turbidity (FNU)',
                'fDOM (RFU)': 'Pfl - fDOM (RFU)',
                'fDOM (QSU)': 'Pfl - fDOM (QSU)',
            }
            
            # Transform each sensor's uncertainties
            for sensor_name in list(sensor_uncertainties.keys()):
                col_name = sensor_column_map.get(sensor_name)
                if col_name:
                    sensor_uncertainties[sensor_name] = normalize_uncertainty_params(
                        sensor_uncertainties[sensor_name], col_name, norm_params
                    )
                    print(f"  OK: Transformed {sensor_name} to normalized scale")
        except FileNotFoundError:
            print(f"  ! Warning: Could not load normalization parameters from {normalization_file}")
            print(f"    Using raw-scale uncertainty parameters (may cause over-perturbation)")
        
        # Set random seed once for reproducibility
        np.random.seed(random_seed)
        print(f"Monte Carlo sampling enabled: K={n_mc_replicates} replicates, seed={random_seed}\n")

    # Initialize a counter for naming output files
    segment_counter = 1

    ## Iterate through the dataframe to find valid segments
    if fault_tolerant:
        indices = find_valid(df, target_columns, predictor_cols, length, nan_tol)
        segment_counter = 1
        for i, idx in enumerate(indices):
            # Compute start and end of the segment
            start = max(0, idx - length + 1)
            end = idx + 1  # include the current row

            # Slice the DataFrame
            segment = df.iloc[start:end]

            # Save unperturbed version
            output_file = os.path.join(samples_dir, f"segment_{segment_counter:04d}.csv")
            segment.to_csv(output_file, index=False)

            if use_uncertainty_perturbation:
                # Generate K Monte Carlo replicates
                for k in range(1, n_mc_replicates + 1):
                    # Apply perturbation with a derived seed for each replicate
                    replicate_seed = random_seed + k
                    segment_perturbed = apply_uncertainty_perturbation(
                        segment, sensor_uncertainties, random_seed=replicate_seed
                    )
                    
                    # Save with replicate label
                    output_file = os.path.join(
                        perturbed_samples_dir,
                        f"segment_{segment_counter:04d}_mc_{k:03d}.csv"
                    )
                    segment_perturbed.to_csv(output_file, index=False)
            
            segment_counter += 1
    else:
        for i in range(len(df) - (length-1)):
            segment = df.iloc[i:i+length]
            last_row = segment.iloc[-1]
            preceding_rows = segment.iloc[:-1]

            # Check if the last row has any non-null value in the target columns
            if last_row[target_columns].notnull().all():

                # Check if the 'Segment' column has a constant value in all rows
                if preceding_rows['Segment'].nunique() == 1:
                    # Save unperturbed version
                    output_file = os.path.join(samples_dir, f"segment_{segment_counter:04d}.csv")
                    segment.to_csv(output_file, index=False)
                    
                    if use_uncertainty_perturbation:
                        # Generate K Monte Carlo replicates
                        for k in range(1, n_mc_replicates + 1):
                            # Apply perturbation with a derived seed for each replicate
                            replicate_seed = random_seed + k
                            segment_perturbed = apply_uncertainty_perturbation(
                                segment, sensor_uncertainties, random_seed=replicate_seed
                            )
                            
                            # Save with replicate label
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
    print(f"\nXGBoost config file generated: {xgb_config_path}")
    
    # Generate Transformer template
    transformer_config_path = generate_transformer_config_template(
        output_dir, 
        'transformer_01',
        predictor_cols,
        target_columns,
        length
    )
    print(f"Transformer config file generated: {transformer_config_path}")

    # Generate GP template
    gp_config_path = generate_gp_config_template(
        output_dir,
        'gp_01',
        predictor_cols,
        target_columns,
        length
    )
    print(f"GP config file generated: {gp_config_path}")

    print("\nEdit the appropriate config file with your desired settings and pass it to e_Train.py")


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
                        'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                        'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                        '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
        'SCADA - pH', 'SCADA - Temperature (°C)']
    target_columns = ['Color', 'Turbidity (FNU)', 'pH', 'E.coli (CFU/100mL)',
    'Intestinal enterococci (CFU/100mL)', 'Colony Count 22°C (CFU/mL)', 'Total coliforms 37°C (CFU/100mL)', 'Arsenic (µg/L)',
             'Lead (µg/L)', 'Cadmium (µg/L)', 'Copper filtered (mg/L)', 'Chromium (µg/L)', 'Nickel (µg/L)', 'Zinc (µg/L)']  # alternative 1: name-based selection
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
    set_name  = "MC_pH"  # Name of subdirectory where samples will be organized
    length = 169  # Hours of contiguous data per sample
    output_dir = os.path.join("../data/output/regression", set_name)

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
        'Lead (µg/L)', 'Cadmium (µg/L)', 'Copper filtered (mg/L)', 'Chromium (µg/L)', 'Nickel (µg/L)', 'Zinc (µg/L)']

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
    #                 'Sink (Zn)_state', '01-Farge_res', '04-Turbiditet_res', '06-E.coli_res',
    # '07-Intestinale enterokokker_res', '08-Kimtall 22°C_res', '09-Koliforme bakterier 37°C_res', '21-Arsen_res',
    #          '24-Bly_res', '32-Kadmium_res', '36-Kopper filtrert_res', '37-Krom_res', '41-Nikkel_res', 'Sink (Zn)_res']

    # to_normalize = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
    #                     'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
    #                     'Pfl - fDOM (QSU)', "Wind speed x (m/s)", "Wind speed y (m/s)",
    #                     'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
    #                     'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
    #                     '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
    #     'SCADA - pH', 'SCADA - Temperature (°C)']
    # split(df, output_dir, target_columns, length, to_normalize, 0)

    split(df, output_dir, ['pH'], length, 0.8, to_normalize,
          True, predictor_cols=predictor_cols,
          use_uncertainty_perturbation=True, n_mc_replicates=10, random_seed=1)
    
    print("\nOK: d_Resample.py completed successfully!")
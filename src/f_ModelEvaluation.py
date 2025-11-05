"""
Compares accuracy of Time Series Forecasting using various models.
"""

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from d_SplitAnalysis import normalize_columns
from e_TrainTransformer import load_samples
from e_TrainTransformer import TimeSeriesTransformer
from e_TrainTransformer import TimeSeriesTargetDataset


def evaluate_model(model, dataset):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y, filename = dataset[i]
            x = x.unsqueeze(0).to(device)  # Add batch dimension
            pred = model(x).squeeze().cpu().numpy()
            # Ensure pred is a 1-D numpy array (even when scalar)
            pred = np.array(pred).reshape(-1)

            true = y.cpu().numpy()
            # Ensure true is also a 1-D numpy array
            true = np.array(true).reshape(-1)

            predictions.append(pred)
            targets.append(true)

    # Turn lists of 1-D arrays into 2-D arrays of shape (n_samples, n_outputs)
    if len(predictions) == 0:
        return np.empty((0, 0)), np.empty((0, 0))

    predictions = np.array(predictions)
    targets = np.array(targets)

    # If they ended up 1-D (happens when numpy collapses to shape (n_samples,)),
    # convert to shape (n_samples, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    mask = np.isfinite(predictions).all(axis=1) & np.isfinite(targets).all(axis=1)
    return predictions[mask], targets[mask]

def evaluate_naive(dataset, historic, output_columns, data_dir, output_rows=-1, gap_hours=5):

    # Load lookup table for baseline model
    df = pd.read_csv(historic,parse_dates=["TIMESTAMP"])
    sort_df = df.sort_values("TIMESTAMP")
    historical_df = normalize_columns(sort_df, output_columns)

    predictions, targets = [], []
    for i in range(len(dataset)):
        _, y, filename = dataset[i]

        # Load the sample file to get the output times
        sample_df = pd.read_csv(os.path.join(data_dir, 'samples', filename), parse_dates=["TIMESTAMP"])
        output_times = sample_df["TIMESTAMP"].iloc[output_rows:]

        # Apply gap constraint (before the first output timestamp)
        cutoff_time = output_times.iloc[0] - pd.Timedelta(hours=gap_hours)

        # Filter historical data before cutoff_time
        earlier_values = historical_df[historical_df["TIMESTAMP"] < cutoff_time][output_columns]

        # Drop NaN values and select the last valid row
        valid_values = earlier_values.dropna()
        if valid_values.empty:
            baseline_pred = np.full((len(output_times), len(output_columns)), np.nan)
        else:
            last_value = valid_values.iloc[-1].values
            # Repeat last known value for each output timestamp
            baseline_pred = np.tile(last_value, (len(output_times), 1))

        predictions.append(baseline_pred.reshape(-1))
        targets.append(y.numpy().reshape(-1))  # Ensure target is flattened

    return np.array(predictions), np.array(targets)

def evaluate_linear(directory, dataset, historic, output_columns, data_dir,
                    output_rows=-1, window_hours=6, gap_hours=5,
                    debug_plot=False, examples=10):
    """
    Linear baseline with causal gap constraint and optional debug visualization.

    Now also plots the true target values (ground truth) in green for comparison.
    """
    # Load full time series as input for simple "baseline" models
    df = pd.read_csv(historic, parse_dates=["TIMESTAMP"])
    sort_df = df.sort_values("TIMESTAMP")
    historical_df = normalize_columns(sort_df, output_columns, directory=data_dir)

    if debug_plot:
        os.makedirs(os.path.join(directory, "model", "examples_linear"), exist_ok=True)

    predictions, targets = [], []

    for i in range(len(dataset)):
        _, y, filename = dataset[i]

        # Load sample to get output timestamps
        sample_df = pd.read_csv(os.path.join(data_dir, 'samples', filename), parse_dates=["TIMESTAMP"])
        output_times = sample_df["TIMESTAMP"].iloc[output_rows:]

        # Forecast window definition
        forecast_start = output_times.iloc[0]
        window_end = forecast_start - pd.Timedelta(hours=gap_hours)
        window_start = window_end - pd.Timedelta(hours=window_hours)

        # Select regression window
        window_df = historical_df[
            (historical_df["TIMESTAMP"] >= window_start) &
            (historical_df["TIMESTAMP"] < window_end)
        ][["TIMESTAMP"] + output_columns].dropna(subset=output_columns)

        if window_df.empty:
            pred_matrix = np.full((len(output_times), len(output_columns)), np.nan)
        else:
            times = (window_df["TIMESTAMP"] - window_start).dt.total_seconds().values.reshape(-1, 1)
            pred_matrix = np.zeros((len(output_times), len(output_columns)))

            for j, col in enumerate(output_columns):
                values = window_df[col].values
                if len(values) == 0 or np.isnan(values).all():
                    pred_matrix[:, j] = np.nan
                else:
                    model = LinearRegression()
                    model.fit(times, values)

                    # Predict for each future timestamp
                    forecast_secs = (output_times - window_start).dt.total_seconds().values.reshape(-1, 1)
                    pred_matrix[:, j] = model.predict(forecast_secs)

                    # === Debug plot for this variable ===
                    if debug_plot and i < examples:
                        plt.figure(figsize=(8, 5))
                        # Historical training data
                        plt.scatter(window_df["TIMESTAMP"], values, label="Training data", color="blue", alpha=0.6)
                        # Regression line (continuous fit within training window)
                        fit_times = np.linspace(times.min(), times.max(), 100).reshape(-1, 1)
                        fit_dates = [window_start + pd.Timedelta(seconds=s) for s in fit_times.flatten()]
                        plt.plot(fit_dates, model.predict(fit_times), "k--", label="Fitted line")
                        # Forecast predictions
                        plt.scatter(output_times, pred_matrix[:, j], color="orange", label="Predictions", zorder=5)
                        # === NEW: plot ground truth (targets) ===
                        plt.plot(output_times, y.numpy().reshape(-1)[j::len(output_columns)],
                                 color="green", marker="o", linestyle="", label="Ground truth", zorder=6)

                        # Reference verticals
                        plt.axvline(window_end, color="red", linestyle="--", label=f"Gap start (-{gap_hours}h)")
                        plt.axvline(forecast_start, color="red", linestyle=":", label="Forecast start")

                        plt.title(f"Linear Regression Forecast — {col}\nSample: {filename}")
                        plt.xlabel("Timestamp")
                        plt.ylabel(col)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(os.path.join(directory, "model", "examples_linear"),
                                                 f"{filename}_{col}_debug.png"))
                        plt.close()

        predictions.append(pred_matrix.reshape(-1))
        targets.append(y.numpy().reshape(-1))

    return np.array(predictions), np.array(targets)

def evaluate_seasonal(dataset, historic, output_columns, data_dir, output_rows=-1, diurnal_window=2):
    """
    Seasonal baseline with temporal proximity (±time_window_hours) and hierarchical fallbacks.
    For each forecast timestamp, the prediction is computed as:
        1. Mean of all past-year values from the same ISO week and within ±time_window_hours of the same time of day.
        2. If none found, mean of all past-year values from the same ISO week (any time).
        3. If none found, mean of all past-year values from the same calendar month (any time).
        4. If none found, mean of all past-year values (any time).

    Parameters
    ----------
    dataset : Dataset
        Sequence of (input, target, filename) tuples.
    historic : str
        Path to full historical CSV with 'TIMESTAMP' and relevant columns.
    output_columns : list
        Columns to forecast.
    data_dir : str
        Directory containing the per-sample CSVs.
    output_rows : slice or int
        Slice defining the forecast horizon rows in each file.
    diurnal_window : float
        Maximum allowed deviation in hours for time-of-day matching.

    Returns
    -------
    predictions : np.ndarray
        Array of shape (n_samples, n_outputs) matching evaluate_model().
    targets : np.ndarray
        Array of shape (n_samples, n_outputs) matching evaluate_model().
    """
    # Load and normalize historical data
    df = pd.read_csv(historic, parse_dates=["TIMESTAMP"])
    sort_df = df.sort_values("TIMESTAMP")
    historical_df = normalize_columns(sort_df, output_columns, directory=data_dir)

    predictions, targets = [], []

    # Precompute datetime components
    historical_df["TIMESTAMP"] = pd.to_datetime(historical_df["TIMESTAMP"])
    historical_df["YEAR"] = historical_df["TIMESTAMP"].dt.year
    historical_df["WEEK"] = historical_df["TIMESTAMP"].apply(lambda ts: ts.isocalendar().week)
    historical_df["MONTH"] = historical_df["TIMESTAMP"].dt.month
    historical_df["HOUR"] = historical_df["TIMESTAMP"].dt.hour + historical_df["TIMESTAMP"].dt.minute / 60.0

    for i in range(len(dataset)):
        _, y, filename = dataset[i]

        # Load the sample file to get forecast timestamps
        sample_df = pd.read_csv(os.path.join(data_dir, 'samples', filename), parse_dates=["TIMESTAMP"])
        output_times = sample_df["TIMESTAMP"].iloc[output_rows:]
        if len(output_times) == 0:
            continue

        pred_matrix = np.zeros((len(output_times), len(output_columns)))

        for t_idx, ts in enumerate(output_times):
            target_year = ts.year
            target_week = ts.isocalendar().week
            target_month = ts.month
            target_hour = ts.hour + ts.minute / 60.0

            # Exclude same calendar year
            candidates = historical_df[historical_df["YEAR"] != target_year]

            # --- Step 1: same week, within ±time_window_hours of same time of day ---
            within_hours = np.abs(candidates["HOUR"] - target_hour) <= diurnal_window
            week_match = candidates["WEEK"] == target_week
            subset = candidates[week_match & within_hours]

            # --- Step 2: fallback to same week (any time) ---
            if subset.empty:
                subset = candidates[week_match]

            # --- Step 3: fallback to same month (any time) ---
            if subset.empty:
                subset = candidates[candidates["MONTH"] == target_month]

            # --- Step 4: fallback to all available past years ---
            if subset.empty:
                subset = candidates

            seasonal_values = subset[output_columns].dropna()
            if seasonal_values.empty:
                pred_matrix[t_idx, :] = np.nan
            else:
                pred_matrix[t_idx, :] = seasonal_values.mean().values

        predictions.append(pred_matrix.reshape(-1))
        targets.append(y.numpy().reshape(-1))
    return np.array(predictions), np.array(targets)

def visualizer(*pred_target_pairs, labels=None, directory="../data/output/regression/model", num_samples=100):
    sns.set_style("whitegrid")

    # === Scatter plot of predictions vs actuals ===
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette("husl", len(pred_target_pairs))
    min_val, max_val = float("inf"), float("-inf")
    metrics = []  # store (label, MAE, RMSE, R2)

    for i, (preds, targets) in enumerate(pred_target_pairs):
        preds = np.array(preds)
        targets = np.array(targets)
        preds = preds[:num_samples].reshape(-1)
        targets = targets[:num_samples].reshape(-1)
        label = labels[i] if labels else f"Model {i+1}"
        ax.scatter(targets, preds, label=label, alpha=0.7, color=colors[i])
        min_val = min(min_val, np.nanmin(targets), np.nanmin(preds))
        max_val = max(max_val, np.nanmax(targets), np.nanmax(preds))
        mask = np.isfinite(preds) & np.isfinite(targets)
        if mask.any():
            mae = mean_absolute_error(targets[mask], preds[mask])
            rmse = np.sqrt(mean_squared_error(targets[mask], preds[mask]))
            r2 = r2_score(targets[mask], preds[mask])
            metrics.append((label, mae, rmse, r2))
            print(f"{label}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        else:
            metrics.append((label, np.nan, np.nan, np.nan))
            print(f"{label}: no valid data for metrics")

    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Predicted vs Actual Values")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(directory, "model", "predictions.png"))
    plt.close(fig)

    # === Metrics summary figure ===
    if metrics:
        labels_m = [m[0] for m in metrics]
        mae_vals = [m[1] for m in metrics]
        rmse_vals = [m[2] for m in metrics]
        r2_vals = [m[3] for m in metrics]
        fig, ax = plt.subplots(1, 3, figsize=(14, 5))
        bar_kwargs = dict(alpha=0.7)
        ax[0].bar(labels_m, mae_vals, color=colors, **bar_kwargs)
        ax[0].set_title("Mean Absolute Error")
        ax[0].set_ylabel("MAE")
        ax[1].bar(labels_m, rmse_vals, color=colors, **bar_kwargs)
        ax[1].set_title("Root Mean Squared Error")
        ax[1].set_ylabel("RMSE")
        ax[2].bar(labels_m, r2_vals, color=colors, **bar_kwargs)
        ax[2].set_title("R² Score")
        ax[2].set_ylabel("R²")
        for a in ax:
            a.set_xticks(range(len(labels_m)))
            a.set_xticklabels(labels_m, rotation=30, ha="right")
            a.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.suptitle("Model Performance Metrics", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(Path(directory, "model", "metrics_summary.png"))
        plt.close(fig)

    # === NEW: RMSE vs Forecast Horizon ===
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (preds, targets) in enumerate(pred_target_pairs):
        preds = np.array(preds)
        targets = np.array(targets)
        label = labels[i] if labels else f"Model {i+1}"
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if preds.shape != targets.shape:
            continue
        horizon = preds.shape[1]
        rmse_per_step = []
        for t in range(horizon):
            mask = np.isfinite(preds[:, t]) & np.isfinite(targets[:, t])
            if mask.any():
                rmse = np.sqrt(mean_squared_error(targets[mask, t], preds[mask, t]))
            else:
                rmse = np.nan
            rmse_per_step.append(rmse)
        ax.plot(range(1, horizon + 1), rmse_per_step, marker='o', label=label, color=colors[i])

    ax.set_title("RMSE vs Forecast Horizon")
    ax.set_xlabel("Forecast Step (T+)")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(Path(directory, "model", "horizon_rmse.png"))
    plt.close(fig)

def reverse_normalize_columns(df, columns, min=0, max=1, directory="../data/output/regression"):
    """
    Reverse normalization of specified columns using saved parameters.

    Parameters:
    - df: pandas.DataFrame
    - columns: list of column names to reverse normalize
    - min: minimum value of target range used during normalization
    - max: maximum value of target range used during normalization
    - param_path: path to load normalization parameters

    Returns:
    - A copy of the DataFrame with columns restored to original scale.
    """
    df_restored = df.copy()
    min_val, max_val = min, max

    # Load normalization parameters from file
    with open(Path(directory,"model","normalization_params.json"), "r") as f:
        normalization_params = json.load(f)

    for col in columns:
        if col in normalization_params:
            col_min = normalization_params[col]["min"]
            col_max = normalization_params[col]["max"]
            if col_max != col_min:
                df_restored[col] = ((df[col] - min_val) / (max_val - min_val)) * (col_max - col_min) + col_min
            else:
                df_restored[col] = col_min  # All values were the same originally

    return df_restored


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")

    matplotlib.use('Agg')  # Non-interactive backend for file output to handle remote machine installation errors

    ##################################################################################################################
    # Load input, output and model hyperparameters from data_dir


    data_dir = "../data/output/regression/Kimtall24hr"  # Parent directory of test/train sample folder

    with open(Path(data_dir, 'model', 'model_config.json'), 'r') as f:
        config = json.load(f)

    input_columns = config["input_columns"]
    output_columns = config["output_columns"]
    input_rows = slice(config["input_row_1"], config["input_row_2"])
    output_rows = config["output_rows"]

    # Configure simple non-ML ("baseline") model calculation methods
    historic = "../data/output/regression/Combined_Cleaned.csv"  # Path to file with baseline model input
    gap_hours = 0  # Period before first forecast value from which input data is not used in baseline models
    window_hours = 550  # Length of period for linear regression training (min. ~530 hrs for Eurofins params)
    diurnal_window = 1  # Number of hours before/after target time to include in average for seasonal model

    ################################################################################################################
    # Prepare data and models for evaluation
    model = TimeSeriesTransformer(config).to(device)
    model.load_state_dict(torch.load(os.path.join(data_dir, "model","transformer_model.pt"), map_location=device))
    model.eval()  # Set to evaluation mode

    reloadset = Path(data_dir, "model","test_files.txt")
    with open(reloadset) as f:
        test_files = [line.strip() for line in f]
    test_samples = load_samples(os.path.join(data_dir,"samples"),input_columns=input_columns,output_columns=output_columns,
        input_rows=input_rows, output_rows=output_rows, file_list=test_files)
    test_dataset = TimeSeriesTargetDataset(test_samples)

    ##################################################################################################################
    # Evaluate models

    model_preds, targets = evaluate_model(model, test_dataset)
    naive_preds, naive_targets = evaluate_naive(test_dataset, historic, output_columns, data_dir,
                                                output_rows=output_rows, gap_hours=gap_hours)
    linear_preds, linear_targets = evaluate_linear(data_dir, test_dataset, historic, output_columns, data_dir,
                                                   output_rows=output_rows, window_hours=window_hours, gap_hours=gap_hours,
                                                   debug_plot=True, examples=10)
    seasonal_preds, seasonal_targets = evaluate_seasonal(test_dataset, historic, output_columns, data_dir,
                                                         output_rows=output_rows, diurnal_window=diurnal_window)

    visualizer((model_preds, targets), (naive_preds, naive_targets), (linear_preds, linear_targets),
               (seasonal_preds, seasonal_targets),
               labels=["Transformer", "Naive", "Linear", "Seasonal"], directory=data_dir, num_samples=200)

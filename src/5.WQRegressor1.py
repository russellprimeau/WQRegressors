"""
Time Series Forecasting using Transformer Model in PyTorch.
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_samples(directory, input_columns, output_columns, input_rows, output_rows, file_list=None):
    samples = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".csv"):
            continue
        if file_list is not None and filename not in file_list:
            continue  # Skip files not in the provided list
        df = pd.read_csv(os.path.join(directory, filename))
        if not set(input_columns + output_columns).issubset(df.columns):
            continue  # skip files with missing columns
        if len(df) < input_rows.stop:
            print(f"Sample {filename} skipped — not enough rows ({len(df)} < {input_rows.stop})")
            continue  # skip files without enough rows

        input_seq = df.iloc[input_rows, :][input_columns].values
        output_seq = df.iloc[output_rows:, :][output_columns].values

        if np.isnan(input_seq).any() or np.isnan(output_seq).any():
            print(f"Sample {filename} skipped - contains NaN values")
            continue  # skip invalid samples
        samples.append((input_seq, output_seq, filename))
    print("Samples loaded")
    return samples

def train_model(directory, model, dataloader, num_epochs=100, learning_rate=1e-3, loss_threshold=1e-3):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}")

        # # Early stopping condition
        # if loss_threshold is not None and avg_loss <= loss_threshold:
        #     print(f"Stopping early at epoch {epoch + 1} because loss reached {avg_loss:.6f}")
        #     break

        # Plotting loss vs. epochs on log-log scale
        plt.figure(figsize=(8, 6))
        x_vals = list(range(1, len(epoch_losses) + 1))
        y_vals = epoch_losses
        plt.loglog(x_vals, y_vals, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss vs. Epochs (Log-Log Scale)")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(directory, "model", "loss_plot.png"))
        plt.close()

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
    historical_df = normalize_columns(sort_df, data_columns)

    predictions, targets = [], []
    for i in range(len(dataset)):
        _, y, filename = dataset[i]

        # Load the sample file to get the output times
        sample_df = pd.read_csv(os.path.join(data_dir, filename), parse_dates=["TIMESTAMP"])
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
    historical_df = normalize_columns(sort_df, data_columns, directory=data_dir)

    if debug_plot:
        os.makedirs(os.path.join(directory, "model", "examples_linear"), exist_ok=True)

    predictions, targets = [], []

    for i in range(len(dataset)):
        _, y, filename = dataset[i]

        # Load sample to get output timestamps
        sample_df = pd.read_csv(os.path.join(data_dir, filename), parse_dates=["TIMESTAMP"])
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
    historical_df = normalize_columns(sort_df, data_columns, directory=data_dir)

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
        sample_df = pd.read_csv(os.path.join(data_dir, filename), parse_dates=["TIMESTAMP"])
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

        # Flatten multi-dimensional outputs
        preds = preds[:num_samples].reshape(-1)
        targets = targets[:num_samples].reshape(-1)

        label = labels[i] if labels else f"Model {i+1}"
        ax.scatter(targets, preds, label=label, alpha=0.7, color=colors[i])

        min_val = min(min_val, np.nanmin(targets), np.nanmin(preds))
        max_val = max(max_val, np.nanmax(targets), np.nanmax(preds))

        # Compute metrics
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

    # Diagonal reference line
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
            a.set_xticks(range(len(labels_m)))  # or use actual tick positions
            a.set_xticklabels(labels_m, rotation=30, ha="right")
            a.grid(True, axis="y", linestyle="--", alpha=0.6)

        plt.suptitle("Model Performance Metrics", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(Path(directory, "model", "metrics_summary.png"))
        plt.close(fig)

def normalize_columns(df, columns, min=0, max=1, directory="../data/output/regression"):
    """
    Normalize specified columns in a DataFrame to a given range and save original min/max values.

    Parameters:
    - df: pandas.DataFrame
    - columns: list of column names to normalize
    - min: minimum value of target range
    - max: maximum value of target range
    - save_path: path to save normalization parameters

    Returns:
    - A copy of the DataFrame with normalized columns.
    """
    df_normalized = df.copy()
    min_val, max_val = min, max
    normalization_params = {}

    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        normalization_params[col] = {"min": col_min, "max": col_max}
        if col_max != col_min:
            df_normalized[col] = ((df[col] - col_min) / (col_max - col_min)) * (max_val - min_val) + min_val
        else:
            df_normalized[col] = (min_val + max_val) / 2

    # Save normalization parameters to file
    file = Path(directory, "model", "normalized.csv")
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as f:
        json.dump(normalization_params, f)

    return df_normalized

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

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=4, dropout=0.1, output_dim=1, seq_len=72):
        super(TimeSeriesTransformer, self).__init__()
        self.model_dim = model_dim

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, model_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # <-- Add this
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to scalar
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_len=24, input_dim=28]
        returns: [batch_size, output_dim=1]
        """
        batch_size, seq_len, _ = x.size()

        # Project input features
        x = self.input_proj(x)  # [batch_size, seq_len, model_dim]

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        # Encode
        encoded = self.transformer_encoder(x)

        # Use last timestep's encoding
        last_encoding = encoded[:, -1, :]  # [batch_size, model_dim]

        # Project to output
        output = self.output_proj(last_encoding)  # [batch_size, 1]

        return output

class TimeSeriesTargetDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq, filename = self.samples[idx]
        x = torch.tensor(input_seq, dtype=torch.float32)
        y = torch.tensor(target_seq, dtype=torch.float32).flatten()
        return x, y, filename

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")

    matplotlib.use('Agg')  # Non-interactive backend for file output to handle remote machine installation errors

    ## Configure input, output and model hyperparameters
    all_columns = ['TIMESTAMP', 'Segment', 'Interpolated', 'Pfl - Temp (C)', 'Pfl - Sp Cond (microS_cm)',
        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)', 'Pfl - fDOM (QSU)',
        'Instantaneous atmospheric pressure (mBar)', 'Wind direction 10minRollingAvg (°)_x',
        'Wind direction 10minRollingAvg (°)_y', 'Hourly average wind direction (°)_x',
        'Hourly average wind direction (°)_y', 'Average wind speed (m/s)',
        'Maximum sustained wind speed, 3-second span (m/s)', 'Time of maximum 3s Gust',
        'Maximum sustained wind speed, 10-minute span (m/s)', 'Time of maximum 10 minute gust',
        'Hourly average atmospheric pressure at station (mBar)', 'Maximum pressure differential, 3-hour span (mBar)',
        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
        'Longwave (IR) radiation (W/m2)', 'Instantaneous sea-level atmospheric pressure (mBar)',
        'Shortwave (solar) radiation (W/m2)', 'Precipitation (mm/hr)', 'Instantaneous temperature (°C)',
        'Maximum temperature (°C)', 'Minimum temperature (°C)', 'Average humidity (% relative humidity)',
        'SCADA - pH', 'SCADA - Temperature (°C)', '06-E.coli', '08-Kimtall 22°C', '21-Arsen', '24-Bly',
        '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']

    data_columns = ['Pfl - Temp (C)', 'Pfl - Sp Cond (microS_cm)',
        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)', 'Pfl - fDOM (QSU)',
        'Instantaneous atmospheric pressure (mBar)', 'Wind direction 10minRollingAvg (°)_x',
        'Wind direction 10minRollingAvg (°)_y', 'Hourly average wind direction (°)_x',
        'Hourly average wind direction (°)_y', 'Average wind speed (m/s)',
        'Maximum sustained wind speed, 3-second span (m/s)', 'Time of maximum 3s Gust',
        'Maximum sustained wind speed, 10-minute span (m/s)', 'Time of maximum 10 minute gust',
        'Hourly average atmospheric pressure at station (mBar)', 'Maximum pressure differential, 3-hour span (mBar)',
        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
        'Longwave (IR) radiation (W/m2)', 'Instantaneous sea-level atmospheric pressure (mBar)',
        'Shortwave (solar) radiation (W/m2)', 'Precipitation (mm/hr)', 'Instantaneous temperature (°C)',
        'Maximum temperature (°C)', 'Minimum temperature (°C)', 'Average humidity (% relative humidity)',
        'SCADA - pH', 'SCADA - Temperature (°C)', '06-E.coli', '08-Kimtall 22°C', '21-Arsen', '24-Bly',
        '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']

    data_dir = "../data/output/regression/SCADATemp96hr"  # Directory with samples for test/train dataset
    historic = "../data/output/regression/Combined_Cleaned.csv"  # Path to file with baseline model input
    input_columns = ['Pfl - Temp (C)',
        'Pfl - Sp Cond (microS_cm)',
        'Pfl - pH',
        'Pfl - DO (% Sat)',
        'Pfl - Turbidity (FNU)',
        'Pfl - fDOM (QSU)',
        'Instantaneous atmospheric pressure (mBar)',
        'Hourly average wind direction (°)_x',
        'Hourly average wind direction (°)_y',
        'Average wind speed (m/s)',
        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
        'Longwave (IR) radiation (W/m2)',
        'Shortwave (solar) radiation (W/m2)',
        'Precipitation (mm/hr)',
        'Instantaneous temperature (°C)',
        'Average humidity (% relative humidity)'
                     ]
    output_columns = ['SCADA - Temperature (°C)']
    input_rows = slice(0, 83)
    output_rows = -12

    # ML Training Hyperparameters
    random_state = 35  # Random seed which deterministically sets the test/train split
    test_size = 0.15  # Fraction of samples saved for evaluation after training
    batch_size = 10  # Minibatch size. Smaller batches -> noisier, but escapes local minima quicker
    num_epochs = 1000  # Training duration (excessive epochs can cause overfitting to training data)
    loss_threshold = 0.000001  # Threshold of acceptably small loss to terminate training early
    learning_rate = 1e-4  # Limit on parameter adjustment size per epoch
    model_dim = 256  # Model size
    num_heads = 4  # Parallel attention heads
    num_layers = 8  # Depth of NN
    dropout = 0.1  # Regularization technique to prevent overtraining by randomly removing some neurons each epoch

    # Baseline model calculation parameters
    gap_hours = 0  # Period before first forecast value from which input data is not used in baseline models
    window_hours = 6  # Length of period for linear regression training (must be ~500 hrs for Eurofins params)
    diurnal_window = 1  # Number of hours before/after target time to include in average for seasonal model

    # Generate additional model dimensions parametrically based on selection
    input_dim = len(input_columns)
    files = [ f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,f)) ]
    sample_df = pd.read_csv(os.path.join(data_dir, sorted(files)[0]))
    output_dim = len(output_columns) * len(sample_df.iloc[output_rows:])
    seq_len = input_rows.stop - input_rows.start

    # Pre-process dataset
    samples = load_samples(data_dir, input_columns=input_columns, output_columns=output_columns,
                                          input_rows=input_rows, output_rows=output_rows)
    all_filenames = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=random_state)
    os.makedirs(os.path.join(data_dir, "model"), exist_ok=True)
    file1 = Path(data_dir, "model", "train_files.txt")
    with open(file1, "w") as f:
        f.writelines(f"{s[2]}\n" for s in train_samples)
    file2 = Path(data_dir, "model", "test_files.txt")
    with open(file2, "w") as f:
        f.writelines(f"{s[2]}\n" for s in test_samples)

    ## Train
    train_dataset = TimeSeriesTargetDataset(train_samples)
    dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    model = TimeSeriesTransformer(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers,
                                  dropout=dropout, output_dim=output_dim, seq_len=seq_len).to(device)
    train_model(data_dir, model, dataloader, num_epochs, learning_rate, loss_threshold)
    torch.save(model.state_dict(), Path(data_dir, "model","transformer_model.pt"))

    ## Post-processing
    model = TimeSeriesTransformer(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers,
                                  dropout=dropout, output_dim=output_dim, seq_len=seq_len).to(device)
    model.load_state_dict(torch.load(os.path.join(data_dir, "model","transformer_model.pt"), map_location=device))
    model.eval()  # Set to evaluation mode

    reloadset = Path(data_dir, "model","test_files.txt")
    with open(reloadset) as f:
        test_files = [line.strip() for line in f]
    test_samples = load_samples(data_dir,input_columns=input_columns,output_columns=output_columns,
        input_rows=input_rows, output_rows=output_rows, file_list=test_files)
    test_dataset = TimeSeriesTargetDataset(test_samples)

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

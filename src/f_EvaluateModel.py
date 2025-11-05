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
    historical_df = normalize_columns(sort_df, output_columns, save=False, directory=Path(data_dir, 'examples_naive'))

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

def evaluate_linear(data_dir, model_name, dataset, historic, output_columns,
                    output_rows=-1, window_hours=6, gap_hours=5,
                    debug_plot=False, examples=10):
    """
    Linear baseline with causal gap constraint and optional debug visualization.

    Now also plots the true target values (ground truth) in green for comparison.
    """
    # Load full time series as input for simple "baseline" models
    df = pd.read_csv(historic, parse_dates=["TIMESTAMP"])
    sort_df = df.sort_values("TIMESTAMP")
    historical_df = normalize_columns(sort_df, output_columns, save=False, directory=Path(data_dir, 'examples_linear'))

    if debug_plot:
        os.makedirs(os.path.join(data_dir, "models", model_name, "examples_linear"), exist_ok=True)

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
                        plt.savefig(os.path.join(os.path.join(data_dir, "models", model_name, "examples_linear"),
                                                 f"{filename}_{col}_debug.png"))
                        plt.close()

        predictions.append(pred_matrix.reshape(-1))
        targets.append(y.numpy().reshape(-1))

    return np.array(predictions), np.array(targets)

def evaluate_seasonal(dataset, historic, output_columns, data_dir, model_name,
                      output_rows=-1, diurnal_window=2):
    """
    Seasonal baseline with sliding ±day windows (continuous rather than discrete week/month bins).

    For each forecast timestamp:
        1. Exclude data from the same calendar year.
        2. Use hierarchical selection:
            a. |Δday| ≤ 4 days  AND  |Δhour| ≤ diurnal_window
            b. |Δday| ≤ 4 days
            c. |Δday| ≤ 15 days
            d. all remaining rows
        3. Return column-wise mean of the first non-empty subset.

    Also produces diagnostic plots:
        - one connected line per actual year (ground truth),
        - one continuous hourly "Predicted" series across a generic year,
          with month labels on the x-axis.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- Load and normalize historical data ---
    df = pd.read_csv(historic, parse_dates=["TIMESTAMP"])
    sort_df = df.sort_values("TIMESTAMP")
    historical_df = normalize_columns(sort_df, output_columns, directory=data_dir)

    predictions, targets = [], []

    # Precompute datetime components
    historical_df["TIMESTAMP"] = pd.to_datetime(historical_df["TIMESTAMP"])
    historical_df["YEAR"] = historical_df["TIMESTAMP"].dt.year
    historical_df["DAYOFYEAR"] = historical_df["TIMESTAMP"].dt.dayofyear
    historical_df["HOUR"] = historical_df["TIMESTAMP"].dt.hour + historical_df["TIMESTAMP"].dt.minute / 60.0

    # --- Core seasonal calculation ---
    for i in range(len(dataset)):
        _, y, filename = dataset[i]

        # Load forecast timestamps
        sample_df = pd.read_csv(os.path.join(data_dir, 'samples', filename), parse_dates=["TIMESTAMP"])
        output_times = sample_df["TIMESTAMP"].iloc[output_rows:]
        if len(output_times) == 0:
            continue

        pred_matrix = np.zeros((len(output_times), len(output_columns)))

        for t_idx, ts in enumerate(output_times):
            target_year = ts.year
            target_day = ts.timetuple().tm_yday
            target_hour = ts.hour + ts.minute / 60.0

            # Exclude same calendar year, but include both earlier and later years
            candidates = historical_df[historical_df["YEAR"] != target_year].copy()
            if candidates.empty:
                pred_matrix[t_idx, :] = np.nan
                continue

            # Compute absolute day difference with wrap-around (cyclic year)
            day_diff = np.abs(candidates["DAYOFYEAR"] - target_day)
            day_diff = np.minimum(day_diff, 365 - day_diff)  # wrap near year-end
            candidates["DAY_DIFF"] = day_diff
            candidates["HOUR_DIFF"] = np.abs(candidates["HOUR"] - target_hour)

            # --- Hierarchical filtering ---
            subset = candidates[(candidates["DAY_DIFF"] <= 4) &
                                (candidates["HOUR_DIFF"] <= diurnal_window)]
            if subset.empty:
                subset = candidates[candidates["DAY_DIFF"] <= 4]
            if subset.empty:
                subset = candidates[candidates["DAY_DIFF"] <= 15]
            if subset.empty:
                subset = candidates

            seasonal_values = subset[output_columns].dropna()
            if seasonal_values.empty:
                pred_matrix[t_idx, :] = np.nan
            else:
                pred_matrix[t_idx, :] = seasonal_values.mean().values

        predictions.append(pred_matrix.reshape(-1))
        targets.append(y.numpy().reshape(-1))

    predictions = np.array(predictions)
    targets = np.array(targets)

    # === Diagnostic plot: continuous prediction & month-labeled x-axis ===
    try:
        plot_dir = os.path.join(data_dir, "models", model_name, "examples_seasonal")
        os.makedirs(plot_dir, exist_ok=True)

        # Ground truth
        gt_records = []
        for i in range(len(dataset)):
            _, y, filename = dataset[i]
            sample_df = pd.read_csv(os.path.join(data_dir, 'samples', filename), parse_dates=["TIMESTAMP"])
            output_times = sample_df["TIMESTAMP"].iloc[output_rows:]
            if len(output_times) == 0:
                continue
            flat = y.numpy().reshape(-1)
            for j, col in enumerate(output_columns):
                vals = flat[j::len(output_columns)]
                gt_records.append(pd.DataFrame({
                    "YEAR": output_times.dt.year,
                    "DAYOFYEAR": output_times.dt.dayofyear +
                                 (output_times.dt.hour + output_times.dt.minute / 60.0) / 24.0,
                    "VALUE": vals,
                    "COLUMN": col
                }))
        if len(gt_records) == 0:
            return predictions, targets
        gt_df = pd.concat(gt_records, ignore_index=True)

        # Synthetic hourly grid for one generic year (smooth continuous line)
        hist_years = historical_df["YEAR"].unique()
        synthetic_year = int(historical_df["YEAR"].max()) + 1
        hours_per_year = 24 * 366
        start = pd.Timestamp(year=synthetic_year, month=1, day=1, hour=0)
        synthetic_times = pd.date_range(start=start, periods=hours_per_year, freq='h', tz=None)
        synth_dayofyear = synthetic_times.dayofyear + \
            (synthetic_times.hour + synthetic_times.minute / 60.0) / 24.0
        synth_hour = synthetic_times.hour + synthetic_times.minute / 60.0
        synth_day = synthetic_times.dayofyear

        continuous_preds = {col: [] for col in output_columns}
        hist = historical_df

        for idx in range(len(synthetic_times)):
            target_day = synth_day[idx]
            target_hour = synth_hour[idx]

            candidates = hist.copy()
            day_diff = np.abs(candidates["DAYOFYEAR"] - target_day)
            day_diff = np.minimum(day_diff, 365 - day_diff)
            candidates["DAY_DIFF"] = day_diff
            candidates["HOUR_DIFF"] = np.abs(candidates["HOUR"] - target_hour)

            subset = candidates[(candidates["DAY_DIFF"] <= 4) &
                                (candidates["HOUR_DIFF"] <= diurnal_window)]
            if subset.empty:
                subset = candidates[candidates["DAY_DIFF"] <= 4]
            if subset.empty:
                subset = candidates[candidates["DAY_DIFF"] <= 15]
            if subset.empty:
                subset = candidates

            seasonal_values = subset[output_columns].dropna()
            if seasonal_values.empty:
                for col in output_columns:
                    continuous_preds[col].append(np.nan)
            else:
                means = seasonal_values.mean().values
                for j, col in enumerate(output_columns):
                    continuous_preds[col].append(means[j])

        # --- Plot with month labels ---
        sns.set_style("whitegrid")
        full_days = np.array(synth_dayofyear.tolist(), dtype=float)
        ref_year = 2021
        month_starts = pd.date_range(f"{ref_year}-01-01", f"{ref_year}-12-31", freq="MS")
        month_dayofyear = [d.timetuple().tm_yday for d in month_starts]
        month_labels = [d.strftime("%b") for d in month_starts]

        for col in output_columns:
            plt.figure(figsize=(12, 6))
            sub_gt = gt_df[gt_df["COLUMN"] == col]

            for yr, group in sub_gt.groupby("YEAR"):
                g = group.sort_values("DAYOFYEAR")
                plt.plot(
                    g["DAYOFYEAR"], g["VALUE"],
                    marker="o", linestyle="-",
                    label=f"Actual {yr}", alpha=0.75
                )

            pred_vals = np.array(continuous_preds[col], dtype=float)
            if np.isnan(pred_vals).any():
                s = pd.Series(pred_vals)
                s = s.interpolate().bfill().ffill()
                pred_vals = s.values

            order = np.argsort(full_days)
            x_sorted = full_days[order]
            y_sorted = pred_vals[order]
            unique_mask = np.concatenate(([True], np.diff(x_sorted) > 0))
            x_sorted = x_sorted[unique_mask]
            y_sorted = y_sorted[unique_mask]

            plt.plot(
                x_sorted, y_sorted,
                color="black", linewidth=2.5,
                label="Predicted"
            )

            plt.xticks(month_dayofyear, month_labels)
            plt.xlim(0, 366)
            plt.xlabel("Month")
            plt.ylabel(col)
            plt.title(f"Seasonality-based model for {col}")
            plt.legend(loc="best", fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"seasonal_{col}.png"))
            plt.close()

        print(f"[Info] Saved plot of seasonality model to: {plot_dir}")

    except Exception as e:
        print(f"[Warning] Could not generate plot of seasonality model: {e}")

    return predictions, targets

def visualizer(*pred_target_pairs, labels=None, directory, model_name, num_samples=100):
    """
    Visualize predictions and targets for a range of gaps from time series.
    :param pred_target_pairs:
    :param labels:
    :param directory:
    :param model_name:
    :param num_samples:
    :return:
    """
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
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Seasonal baseline model")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(directory, "models", model_name, "predictions.png"))
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
        plt.savefig(Path(directory, "models", model_name, "metrics_summary.png"))
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
    plt.savefig(Path(directory, "models", model_name, "horizon_rmse.png"))
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
    with open(Path(directory,"models", model_name, "normalization_params.json"), "r") as f:
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
    ## Configure execution space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")

    matplotlib.use('Agg')  # Non-interactive backend for file output to handle remote machine installation errors

    ##################################################################################################################
    ## Load input, output and model hyperparameters from data_dir
    data_dir = "../data/output/regression/SCADATemp168hr"  # Parent directory of test/train sample folder
    model_name = "24hr_fore"

    with open(Path(data_dir, 'models', model_name, 'model_config.json'), 'r') as f:
        config = json.load(f)

    input_columns = config["input_columns"]
    output_columns = config["output_columns"]
    input_rows = slice(config["input_row_1"], config["input_row_2"])
    output_rows = config["output_rows"]

    ## Configure simple non-ML ("baseline") model calculation methods
    historic = "../data/output/regression/Combined_Cleaned.csv"  # Path to file with baseline model input
    gap_hours = 1  # Period before first forecast value from which input data is not used in baseline models
    window_hours = 5  # Length of period for linear regression training (min. ~530 hrs for Eurofins params)
    diurnal_window = 1  # Number of hours before/after target time to include in average for seasonal model

    ################################################################################################################
    ## Prepare data and models for evaluation
    model = TimeSeriesTransformer(config).to(device)
    model.load_state_dict(torch.load(os.path.join(data_dir, "models", model_name, "transformer_model.pt"), map_location=device))
    model.eval()  # Set to evaluation mode

    # Run evaluation using samples excluded from training
    reloadset = Path(data_dir, "models", model_name, "test_files.txt")
    with open(reloadset) as f:
        test_files = [line.strip() for line in f]
    test_samples = load_samples(os.path.join(data_dir,"samples"),input_columns=input_columns,output_columns=output_columns,
        input_rows=input_rows, output_rows=output_rows, file_list=test_files)
    test_dataset = TimeSeriesTargetDataset(test_samples)

    # ## Alternative: for full-coverage plotting of sparse data, evaluate models on complete sample set (train + test)
    # samples = load_samples(os.path.join(data_dir, 'samples'), input_columns=input_columns,
    #                        output_columns=output_columns,
    #                        input_rows=input_rows, output_rows=output_rows)
    # test_dataset = TimeSeriesTargetDataset(samples)
    ##################################################################################################################
    ## Evaluate models

    model_preds, targets = evaluate_model(model, test_dataset)
    naive_preds, naive_targets = evaluate_naive(test_dataset, historic, output_columns, data_dir,
                                                output_rows=output_rows, gap_hours=gap_hours)
    linear_preds, linear_targets = evaluate_linear(data_dir, model_name, test_dataset, historic, output_columns,
                                                   output_rows=output_rows, window_hours=window_hours,
                                                   gap_hours=gap_hours,
                                                   debug_plot=True, examples=10)

    seasonal_preds, seasonal_targets = evaluate_seasonal(test_dataset, historic, output_columns, data_dir, model_name,
                                                         output_rows=output_rows, diurnal_window=diurnal_window)

    visualizer((model_preds, targets), (naive_preds, naive_targets), (linear_preds, linear_targets),
               (seasonal_preds, seasonal_targets),
               labels=["Transformer", "Naive", "Linear", "Seasonal"], model_name=model_name, directory=data_dir,
               num_samples=200)

    # results = []
    # labels = []
    #
    # for value in range(1,143,1):
    #     preds, targets = evaluate_linear(data_dir, model_name, test_dataset, historic, output_columns, data_dir,
    #                                                output_rows=output_rows, window_hours=window_hours, gap_hours=gap_hours,
    #                                                debug_plot=True, examples=10)
    #     results.append((preds, targets))
    #     labels.append(f"Window {value}h")
    #
    # visualizer(*results, labels=labels, directory=..., num_samples=...)

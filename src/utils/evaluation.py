import os
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc)
from .preprocessing import normalize_columns


def evaluate_transformer(model, dataset, device):
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


@lru_cache(maxsize=4096)
def _cached_sample_timestamps(data_dir, filename, sample_subdir="samples"):
    sample_path = Path(data_dir, sample_subdir, filename)
    sample_df = pd.read_csv(sample_path, usecols=["TIMESTAMP"], parse_dates=["TIMESTAMP"])
    return tuple(sample_df["TIMESTAMP"].tolist())


def _get_output_times(data_dir, filename, output_rows, sample_subdir="samples"):
    timestamps = pd.DatetimeIndex(_cached_sample_timestamps(str(Path(data_dir).resolve()), filename, sample_subdir=sample_subdir))

    if isinstance(output_rows, slice):
        return timestamps[output_rows]

    if isinstance(output_rows, (list, tuple, np.ndarray)):
        if len(output_rows) == 0:
            return timestamps[:0]
        idx = np.array(output_rows, dtype=int)
        idx = np.where(idx < 0, len(timestamps) + idx, idx)
        idx = idx[(idx >= 0) & (idx < len(timestamps))]
        return timestamps[idx]

    # Backward-compatible behavior: scalar means start row to end
    return timestamps[int(output_rows):]


def _seasonal_window_mean(days, hours, values, target_day, target_hour, diurnal_window, day_windows=(4, 15)):
    day_diff = np.abs(days - target_day)
    day_diff = np.minimum(day_diff, 365 - day_diff)
    hour_diff = np.abs(hours - target_hour)

    mask = (day_diff <= day_windows[0]) & (hour_diff <= diurnal_window)
    if not np.any(mask):
        mask = day_diff <= day_windows[0]
    if not np.any(mask):
        mask = day_diff <= day_windows[1]
    if not np.any(mask):
        mask = np.ones_like(day_diff, dtype=bool)

    candidate_vals = values[mask]
    if candidate_vals.size == 0:
        return np.nan
    finite_vals = candidate_vals[np.isfinite(candidate_vals)]
    if finite_vals.size == 0:
        return np.nan
    return float(np.mean(finite_vals))

def evaluate_naive(dataset, historic, output_columns, data_dir, output_rows=-1, gap_hours=5, sample_subdir="samples"):

    # Load lookup table for baseline model
    df = pd.read_csv(historic,parse_dates=["TIMESTAMP"])
    sort_df = df.sort_values("TIMESTAMP")
    historical_df = normalize_columns(sort_df, output_columns, save=False, directory=Path(data_dir, 'examples_naive'))

    valid_mask = historical_df[output_columns].notna().all(axis=1)
    valid_times = historical_df.loc[valid_mask, "TIMESTAMP"].to_numpy(dtype="datetime64[ns]")
    valid_values = historical_df.loc[valid_mask, output_columns].to_numpy(dtype=float)

    predictions, targets = [], []
    for i in range(len(dataset)):
        _, y, filename = dataset[i]

        output_times = _get_output_times(data_dir, filename, output_rows, sample_subdir=sample_subdir)
        if len(output_times) == 0:
            continue

        cutoff_time = output_times[0] - pd.Timedelta(hours=gap_hours)
        cutoff_np = np.datetime64(cutoff_time.to_datetime64())
        idx = np.searchsorted(valid_times, cutoff_np, side="left") - 1

        if idx < 0:
            baseline_pred = np.full((len(output_times), len(output_columns)), np.nan)
        else:
            last_value = valid_values[idx]
            baseline_pred = np.tile(last_value, (len(output_times), 1))

        predictions.append(baseline_pred.reshape(-1))
        targets.append(y.reshape(-1))  # Ensure target is flattened

    return np.array(predictions), np.array(targets)

def evaluate_linear(data_dir, forecast_name, dataset, historic, output_columns, output_rows=-1, window_hours=340,
                    gap_hours=0, debug_plot=False, examples=10, sample_subdir="samples"):
    """
    Linear baseline with causal gap constraint and optional debug visualization.

    Now also plots the true target values (ground truth) in green for comparison.
    """
    # Load full time series as input for simple "baseline" forecasts
    df = pd.read_csv(historic, parse_dates=["TIMESTAMP"])
    sort_df = df.sort_values("TIMESTAMP")
    historical_df = normalize_columns(sort_df, output_columns, save=False, directory=Path(data_dir, 'linear'))
    historical_df = historical_df.sort_values("TIMESTAMP").set_index("TIMESTAMP")

    if debug_plot:
        os.makedirs(os.path.join(data_dir, "forecasts", forecast_name, "linear"), exist_ok=True)

    predictions, targets = [], []

    for i in range(len(dataset)):
        _, y, filename = dataset[i]

        output_times = _get_output_times(data_dir, filename, output_rows, sample_subdir=sample_subdir)
        if len(output_times) == 0:
            continue

        forecast_start = output_times[0]
        window_end = forecast_start - pd.Timedelta(hours=gap_hours)
        window_start = window_end - pd.Timedelta(hours=window_hours)

        window_df = historical_df.loc[window_start:window_end - pd.Timedelta(nanoseconds=1), output_columns]
        window_df = window_df.dropna(subset=output_columns)

        if window_df.empty:
            pred_matrix = np.full((len(output_times), len(output_columns)), np.nan)
        else:
            times = (window_df.index - window_start).total_seconds().to_numpy().reshape(-1, 1)
            forecast_secs = (output_times - window_start).total_seconds().to_numpy().reshape(-1, 1)
            pred_matrix = np.zeros((len(output_times), len(output_columns)))

            for j, col in enumerate(output_columns):
                values = window_df[col].values
                if len(values) == 0 or np.isnan(values).all():
                    pred_matrix[:, j] = np.nan
                else:
                    model = LinearRegression()
                    model.fit(times, values)

                    pred_matrix[:, j] = model.predict(forecast_secs)

                    # === Debug plot for this variable ===
                    if debug_plot and i < examples:
                        plt.figure(figsize=(8, 5))
                        plt.scatter(window_df.index, values, label="Training data", color="blue", alpha=0.6)
                        fit_times = np.linspace(times.min(), times.max(), 100).reshape(-1, 1)
                        fit_dates = [window_start + pd.Timedelta(seconds=s) for s in fit_times.flatten()]
                        plt.plot(fit_dates, model.predict(fit_times), "k--", label="Fitted line")
                        plt.scatter(output_times, pred_matrix[:, j], color="orange", label="Predictions", zorder=5)
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
                        plt.savefig(os.path.join(os.path.join(data_dir, "forecasts", forecast_name, "linear"),
                                                 f"{filename}_{col}_debug.png"))
                        plt.close()

        predictions.append(pred_matrix.reshape(-1))
        targets.append(y.reshape(-1))

    return np.array(predictions), np.array(targets)

def _prepare_time_columns(df):
    df["YEAR"] = df["TIMESTAMP"].dt.year
    df["DAYOFYEAR"] = df["TIMESTAMP"].dt.dayofyear
    df["HOUR"] = df["TIMESTAMP"].dt.hour + df["TIMESTAMP"].dt.minute / 60.0
    return df


def evaluate_seasonal(dataset, historic, output_columns, data_dir, output_rows=-1, diurnal_window=2,
                      secondary=None, sample_subdir="samples"):
    """
    Seasonal baseline using either full `historical_df` or, if available,
    a secondary CSV file for specific output columns.

    Now also uses *all available values* from the chosen data source
    (secondary or primary) as ground truth in the diagnostic plots,
    separated into yearly series.
    """
    # --- Load and prepare data sources ---
    df = pd.read_csv(historic, parse_dates=["TIMESTAMP"]).sort_values("TIMESTAMP")
    historical_df = normalize_columns(df, output_columns, directory=data_dir)

    # --- Load secondary data (if provided) ---
    secondary_df = None
    if secondary and os.path.exists(secondary):
        try:
            secondary_df = pd.read_csv(secondary, sep=";", decimal=".", parse_dates=["Time"]).sort_values("Time")
        except:
            secondary_df = pd.read_csv(secondary, sep=";", decimal=",", parse_dates=["Time"]).sort_values("Time")
        secondary_df.rename(columns={"Time": "TIMESTAMP"}, inplace=True)
        secondary_df = normalize_columns(secondary_df, output_columns, directory=data_dir)

    historical_df = _prepare_time_columns(historical_df)
    if secondary_df is not None:
        secondary_df = _prepare_time_columns(secondary_df)

    source_arrays = {}
    for col in output_columns:
        if secondary_df is not None and col in secondary_df.columns:
            src_df = secondary_df
        elif col in historical_df.columns:
            src_df = historical_df
        else:
            source_arrays[col] = None
            continue

        col_vals = pd.to_numeric(src_df[col], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(col_vals)
        source_arrays[col] = {
            "year": src_df["YEAR"].to_numpy(dtype=int)[valid_mask],
            "day": src_df["DAYOFYEAR"].to_numpy(dtype=int)[valid_mask],
            "hour": src_df["HOUR"].to_numpy(dtype=float)[valid_mask],
            "value": col_vals[valid_mask],
        }

    predictions, targets = [], []

    # === Predict values for each sample ===
    for i in range(len(dataset)):
        _, y, filename = dataset[i]
        output_times = _get_output_times(data_dir, filename, output_rows, sample_subdir=sample_subdir)
        if len(output_times) == 0:
            continue

        pred_matrix = np.zeros((len(output_times), len(output_columns)))

        for t_idx, ts in enumerate(output_times):
            target_year = ts.year
            target_day = ts.timetuple().tm_yday
            target_hour = ts.hour + ts.minute / 60.0

            for j, col in enumerate(output_columns):
                src = source_arrays.get(col)
                if src is None:
                    pred_matrix[t_idx, j] = np.nan
                    continue

                year_mask = src["year"] != target_year
                if not np.any(year_mask):
                    pred_matrix[t_idx, j] = np.nan
                    continue

                pred_matrix[t_idx, j] = _seasonal_window_mean(
                    src["day"][year_mask],
                    src["hour"][year_mask],
                    src["value"][year_mask],
                    target_day,
                    target_hour,
                    diurnal_window,
                )

        predictions.append(pred_matrix.reshape(-1))
        targets.append(y.reshape(-1))

    predictions = np.array(predictions)
    targets = np.array(targets)

    # === Diagnostic plot ===
    try:
        plot_dir = os.path.join(data_dir, "forecasts", "seasonal")
        os.makedirs(plot_dir, exist_ok=True)
        sns.set_style("whitegrid")

        # Synthetic hourly timeline for predicted curve
        synthetic_year = int(historical_df["YEAR"].max()) + 1
        start = pd.Timestamp(year=synthetic_year, month=1, day=1, hour=0)
        synthetic_times = pd.date_range(start=start, periods=24 * 366, freq="h")
        synth_day = synthetic_times.dayofyear
        synth_hour = synthetic_times.hour + synthetic_times.minute / 60.0
        synth_dayofyear = synth_day + synth_hour / 24.0

        ref_year = 2021
        month_starts = pd.date_range(f"{ref_year}-01-01", f"{ref_year}-12-31", freq="MS")
        month_dayofyear = [d.timetuple().tm_yday for d in month_starts]
        month_labels = [d.strftime("%b") for d in month_starts]

        for col in output_columns:
            src = source_arrays.get(col)
            if src is None:
                continue

            continuous_vals = []
            for idx in range(len(synthetic_times)):
                target_day = synth_day[idx]
                target_hour = synth_hour[idx]

                continuous_vals.append(
                    _seasonal_window_mean(
                        src["day"],
                        src["hour"],
                        src["value"],
                        target_day,
                        target_hour,
                        diurnal_window,
                    )
                )

            gt_source_df = secondary_df if (secondary_df is not None and col in secondary_df.columns) else historical_df
            gt_df = gt_source_df[["YEAR", "DAYOFYEAR", "HOUR", col]].dropna()
            gt_df["DAYOFYEAR"] = gt_df["DAYOFYEAR"] + gt_df["HOUR"] / 24.0

            # --- Plot ---
            plt.figure(figsize=(12, 6))

            # Ground truth lines (all years)
            for yr, group in gt_df.groupby("YEAR"):
                g = group.sort_values("DAYOFYEAR")
                plt.plot(
                    g["DAYOFYEAR"],
                    g[col],
                    marker="o",
                    linestyle="-",
                    label=f"Actual {yr}",
                    alpha=0.6,
                )

            # Predicted continuous curve
            y_vals = pd.Series(continuous_vals).interpolate().bfill().ffill()
            x_vals = np.array(synth_dayofyear, dtype=float)

            # Prevent wraparound line (no Dec→Jan jump)
            jump_mask = np.diff(x_vals) < 0
            if np.any(jump_mask):
                jump_idx = np.where(jump_mask)[0][0] + 1
                plt.plot(x_vals[:jump_idx], y_vals[:jump_idx],
                         color="black", linewidth=2.5, label="Predicted")
                plt.plot(x_vals[jump_idx:], y_vals[jump_idx:], color="black", linewidth=2.5)
            else:
                plt.plot(x_vals, y_vals, color="black", linewidth=2.5, label="Predicted")

            plt.xticks(month_dayofyear, month_labels)
            plt.xlim(0, 366)
            plt.xlabel("Month")
            plt.ylabel(col)
            plt.title(f"Seasonality-based Model for {col}")
            plt.legend(loc="best", fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"seasonal_{col}.png"))
            plt.close()
    except Exception as e:
        print(f"[Warning] Could not generate plot of seasonality model: {e}")

    return predictions, targets

def binarize_predictions(preds, output_columns, thresholds_df):
    """
    Convert regression outputs to binary classification based on column-wise thresholds.
    """
    preds = np.array(preds)
    # Ensure shape is (n_samples, n_outputs)
    if preds.ndim == 1:
        preds = preds.reshape(-1, len(output_columns))

    binarized = np.zeros_like(preds, dtype=int)

    for i, col in enumerate(output_columns):
        if col not in thresholds_df.columns:
            raise ValueError(f"Threshold for column '{col}' not found in thresholds_df.")
        upper = thresholds_df[col].iloc[0]
        lower = thresholds_df[f"{col}__lower"].iloc[0] if f"{col}__lower" in thresholds_df.columns else np.nan
        exceed = np.zeros(preds.shape[0], dtype=bool)
        if pd.notna(upper):
            upper_val = float(upper)
            if np.isclose(upper_val, 0.0):
                exceed |= preds[:, i] > upper_val
            else:
                exceed |= preds[:, i] >= upper_val
        if pd.notna(lower):
            lower_val = float(lower)
            if np.isclose(lower_val, 0.0):
                exceed |= preds[:, i] < lower_val
            else:
                exceed |= preds[:, i] <= lower_val
        binarized[:, i] = exceed.astype(int)
    return binarized

def visualizer(*pred_target_pairs, labels=None, directory=None, forecast_name=None, num_samples=200, sample_labels=None):
    """
    Visualize predictions and targets for a range of gaps from time series.
    :param pred_target_pairs:
    :param labels:
    :param directory:
    :param forecast_name:
    :param num_samples:
    :return:
    """
    sns.set_style("whitegrid")

    def _aligned_flat(preds, targets, limit=None):
        preds_flat = np.asarray(preds).reshape(-1)
        targets_flat = np.asarray(targets).reshape(-1)
        if limit is not None:
            preds_flat = preds_flat[:limit]
            targets_flat = targets_flat[:limit]
        n = min(len(preds_flat), len(targets_flat))
        return preds_flat[:n], targets_flat[:n]

    # === Scatter plot of predictions vs actuals ===
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette("husl", len(pred_target_pairs))
    min_val, max_val = float("inf"), float("-inf")
    metrics = []  # store (label, MAE, RMSE, R2)


    for i, (preds, targets) in enumerate(pred_target_pairs):
        preds, targets = _aligned_flat(preds, targets, limit=num_samples)
        label = labels[i] if labels else f"Model {i+1}"
        if len(preds) == 0:
            metrics.append((label, np.nan, np.nan, np.nan))
            print(f"{label}: no valid data for metrics")
            continue

        mask = np.isfinite(preds) & np.isfinite(targets)
        # If sample_labels is provided, plot train/test with different symbology
        if sample_labels is not None and len(sample_labels) == len(preds):
            for group, marker, color in [("train", "o", "#1f77b4"), ("test", "s", "#ff7f0e")]:
                group_mask = (np.array(sample_labels) == group) & mask
                if np.any(group_mask):
                    ax.scatter(targets[group_mask], preds[group_mask], label=f"{label} ({group})", alpha=0.7, marker=marker, color=color)
                    min_val = min(min_val, np.nanmin(targets[group_mask]), np.nanmin(preds[group_mask]))
                    max_val = max(max_val, np.nanmax(targets[group_mask]), np.nanmax(preds[group_mask]))
        else:
            if mask.any():
                ax.scatter(targets[mask], preds[mask], label=label, alpha=0.7, color=colors[i])
                min_val = min(min_val, np.nanmin(targets[mask]), np.nanmin(preds[mask]))
                max_val = max(max_val, np.nanmax(targets[mask]), np.nanmax(preds[mask]))
        if mask.any():
            mae = mean_absolute_error(targets[mask], preds[mask])
            rmse = np.sqrt(mean_squared_error(targets[mask], preds[mask]))
            r2 = r2_score(targets[mask], preds[mask])
            metrics.append((label, mae, rmse, r2))
            print(f"{label}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            print()
        else:
            metrics.append((label, np.nan, np.nan, np.nan))
            print(f"{label}: no valid data for metrics")

    if not np.isfinite(min_val) or not np.isfinite(max_val):
        min_val, max_val = 0.0, 1.0

    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted Value")
    ax.set_title("ML & Baseline Models")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(directory, "forecasts", forecast_name, "predictions.png"))
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
        ax[2].set_ylim(bottom=-0.1)
        for a in ax:
            a.set_xticks(range(len(labels_m)))
            a.set_xticklabels(labels_m, rotation=30, ha="right")
            a.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.suptitle("Model Performance Metrics", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(Path(directory, "forecasts", forecast_name, "metrics_summary.png"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (preds, targets) in enumerate(pred_target_pairs):
        preds = np.array(preds)
        targets = np.array(targets)
        label = labels[i] if labels else f"Model {i+1}"
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if preds.ndim != 2 or targets.ndim != 2:
            continue
        if preds.shape[0] == 0 or targets.shape[0] == 0:
            continue
        n_rows = min(preds.shape[0], targets.shape[0])
        horizon = min(preds.shape[1], targets.shape[1])
        if horizon == 0:
            continue
        preds = preds[:n_rows, :horizon]
        targets = targets[:n_rows, :horizon]
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
    plt.savefig(Path(directory, "forecasts", forecast_name, "horizon_rmse.png"))
    plt.close(fig)

    n_sets = len(pred_target_pairs)
    if labels is None:
        labels = [f"Set {i+1}" for i in range(n_sets)]
    elif len(labels) != n_sets:
        raise ValueError("Length of labels must match number of result sets.")

    # Prepare combined error data (supports different lengths per model)
    combined_frames = []
    for (pred, target), label in zip(pred_target_pairs, labels):
        pred_flat, target_flat = _aligned_flat(pred, target)
        if len(pred_flat) == 0:
            continue
        errors = pred_flat - target_flat
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            continue
        combined_frames.append(pd.DataFrame({"Dataset": label, "Error": errors}))

    if not combined_frames:
        print("[WARN] Skipping error distribution plot: no finite errors available.")
        df_long_combined = None
    else:
        df_long_combined = pd.concat(combined_frames, ignore_index=True)

    if df_long_combined is not None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        sns.boxplot(x="Dataset", y="Error", data=df_long_combined,
                    showcaps=True, boxprops={'facecolor': 'lightgray', 'alpha': 0.3, 'linewidth': 0.5},
                    whiskerprops={'linewidth': 0.5}, medianprops={'color': 'blue', 'linewidth': 1}, showfliers=False, ax=ax)

        sns.stripplot(x="Dataset", y="Error", data=df_long_combined,
                      jitter=True, size=6, color='black', ax=ax)

        for artist in ax.collections:
            artist.set_facecolor('red')
            artist.set_edgecolor('red')

        ax.set_title("Prediction Error Distribution")
        ax.set_ylabel("Error (Absolute)")
        ax.set_xlabel("Model")
        plt.tight_layout()
        plt.savefig(Path(directory, "forecasts", forecast_name, "boxplot.png"))
        plt.close()

    # Individual figures per pair comparing columns
    for (pred, target), label in zip(pred_target_pairs, labels):
        pred = np.array(pred)
        target = np.array(target)

        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        if pred.ndim != 2 or target.ndim != 2:
            continue

        n_rows = min(pred.shape[0], target.shape[0])
        n_cols = min(pred.shape[1], target.shape[1])
        if n_rows == 0 or n_cols <= 1:
            continue

        errors_matrix = pred[:n_rows, :n_cols] - target[:n_rows, :n_cols]

        # Skip if 1D or single column
        if errors_matrix.ndim == 1 or errors_matrix.shape[1] == 1:
            continue

        n_cols = errors_matrix.shape[1]
        col_labels = [f"Col {i + 1}" for i in range(n_cols)]

        # Prepare DataFrame for this pair
        df_pair = pd.DataFrame({col_label: errors_matrix[:, i] for i, col_label in enumerate(col_labels)})
        df_long_pair = df_pair.melt(var_name="Column", value_name="Error")

        # Overlay boxplot and jitterplot on one axis
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        sns.boxplot(x="Column", y="Error", data=df_long_pair,
                    showcaps=True, boxprops={'facecolor': 'lightgray', 'alpha': 0.3, 'linewidth': 0.5},
                    whiskerprops={'linewidth': 0.5}, ax=ax)

        sns.stripplot(x="Column", y="Error", data=df_long_pair,
                      jitter=True, size=6, color='black', alpha=0.8, ax=ax)

        ax.set_title(f"Error by Column for {label}")
        ax.set_ylabel("Error")
        ax.set_xlabel("Column")
        plt.tight_layout()
        plt.savefig(Path(directory, "forecasts", forecast_name, f"{label.replace(' ', '_')}_overlay_emphasized.png"))
        plt.close()

def classification_visualizer(*pred_target_pairs, labels=None, directory='.', forecast_name='Classifier',
                              num_samples=200):
    os.makedirs(os.path.join(directory, "forecasts", forecast_name, "classification"), exist_ok=True)
    sns.set_style("whitegrid")
    metrics = []
    all_conf_matrices = []
    roc_data = []
    pr_data = []
    auc_scores = []

    for i, (preds, targets) in enumerate(pred_target_pairs):
        preds = np.array(preds).reshape(-1)[:num_samples]
        targets = np.array(targets).reshape(-1)[:num_samples]
        n = min(len(preds), len(targets))
        preds = preds[:n]
        targets = targets[:n]
        label = labels[i] if labels else f"Model {i+1}"
        if n == 0:
            continue
        mask = np.isfinite(preds) & np.isfinite(targets)
        preds, targets = preds[mask], targets[mask]
        if len(preds) == 0:
            continue

        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        metrics.append((label, acc, prec, rec, f1))

        cm = confusion_matrix(targets, preds)
        all_conf_matrices.append((label, cm))

        try:
            fpr, tpr, _ = roc_curve(targets, preds)
            roc_auc = auc(fpr, tpr)
            roc_data.append((label, fpr, tpr, roc_auc))
            auc_scores.append((label, roc_auc))

            precision, recall, _ = precision_recall_curve(targets, preds)
            pr_auc = auc(recall, precision)
            pr_data.append((label, recall, precision, pr_auc))
        except ValueError:
            print(f"[WARN] Skipping ROC/PR for {label}: requires both classes in targets.")

    if not all_conf_matrices:
        print("[WARN] Skipping classification plots: no valid prediction/target pairs.")
        return

    # Combined Confusion Matrix Plot
    fig, axes = plt.subplots(1, len(all_conf_matrices), figsize=(5 * len(all_conf_matrices), 4))
    if len(all_conf_matrices) == 1:
        axes = [axes]
    for ax, (label, cm) in zip(axes, all_conf_matrices):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{label}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(Path(directory, "forecasts", forecast_name, "classification", f"confusion.png"))
    plt.close()

    # Combined ROC Curve Plot
    if roc_data:
        plt.figure(figsize=(6, 5))
        for label, fpr, tpr, roc_auc in roc_data:
            plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curves")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(directory, "forecasts", forecast_name, "classification", f"roc.png"))
        plt.close()

    # Combined Precision-Recall Curve Plot
    if pr_data:
        plt.figure(figsize=(6, 5))
        for label, recall, precision, pr_auc in pr_data:
            plt.plot(recall, precision, label=f"{label} (AUC = {pr_auc:.2f})")
        plt.title("Precision-Recall Curves")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(directory, "forecasts", forecast_name, "classification", "pr.png"))
        plt.close()

    # AUC Bar Plot
    labels_auc = [x[0] for x in auc_scores]
    auc_vals = [x[1] for x in auc_scores]
    if auc_scores:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=labels_auc, y=auc_vals, hue=auc_scores, legend=False)
        plt.title("AUC Scores")
        plt.ylabel("AUC")
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(Path(directory, "forecasts", forecast_name, "classification", "auc_scores.png"))
        plt.close()

def apply_saved_normalize(df, param_file, min_val=0, max_val=1):
    """
    Normalize columns in df using saved min/max values from param_file.

    Parameters:
    - df: pandas DataFrame with columns to normalize
    - param_file: path to JSON file with saved normalization parameters
    - min_val, max_val: range used during original normalization

    Returns:
    - df_normalized: DataFrame with normalized values
    """
    df_normalized = df.copy()
    with open(param_file, 'r') as f:
        normalization_params = json.load(f)

    for col in df.columns:
        if col in normalization_params:
            col_min = normalization_params[col]["min"]
            col_max = normalization_params[col]["max"]
            if col_max != col_min:
                df_normalized[col] = ((df[col] - col_min) / (col_max - col_min)) * (max_val - min_val) + min_val
            else:
                df_normalized[col] = (min_val + max_val) / 2
    return df_normalized

def reverse_normalize(array, output_columns, param_file, min_val=0, max_val=1):
    """
    Reverse normalization on a NumPy array using saved parameters.

    Parameters:
    - array: NumPy array of shape (n_samples, n_outputs)
    - output_columns: list of column names corresponding to array columns
    - param_file: path to JSON file with saved normalization parameters
    - min_val, max_val: range used during original normalization

    Returns:
    - array_restored: NumPy array with values in original scale
    """
    with open(param_file, 'r') as f:
        normalization_params = json.load(f)

    array_restored = np.copy(array)
    for i, col in enumerate(output_columns):
        if col in normalization_params:
            col_min = normalization_params[col]["min"]
            col_max = normalization_params[col]["max"]
            if col_max != col_min:
                array_restored[:, i] = ((array[:, i] - min_val) / (max_val - min_val)) * (col_max - col_min) + col_min
            else:
                array_restored[:, i] = col_min  # All values were the same originally
    return array_restored

def load_secondary(output_columns, window_hours=3):
    """
    Check which secondary source to use for the seasonal model, based on the output columns.
    If Eurofins (very low sample rate) is output, set linear model window long enough to include multiple samples
    """

    Eurofin_columns = ['01-Farge', '04-Turbiditet', '06-E.coli',
                        '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen',
                        '24-Bly', '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']
    SCADA_columns = ['SCADA - pH', 'SCADA - Temperature (°C)',]
    FullHourly_columns = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
                        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                        'Pfl - fDOM (QSU)']
    Weather_columns = ["Wind speed, x (m/s)", "Wind speed, y (m/s)",
                        'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                        'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                        '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)']

    new_window_hours = window_hours
    if any(param in Eurofin_columns for param in output_columns):
        secondary = "../data/input/sensors/Eurofins.csv"
        new_window_hours = 550
    elif any(param in SCADA_columns for param in output_columns):
        secondary = "../data/input/sensors/SCADA.csv"
    elif any(param in FullHourly_columns for param in output_columns):
        secondary = "../data/input/sensors/FullHourly.csv"
    elif any(param in Weather_columns for param in output_columns):
        secondary = "../data/input/sensors/Weather.csv"
    else:
        secondary = False
    return secondary, new_window_hours

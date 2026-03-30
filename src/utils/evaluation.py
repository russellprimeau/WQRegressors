import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc)
from .preprocessing import normalize_columns


EUROFINS_SUMMARY_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "output"
    / "sensors"
    / "tables"
    / "Eurofins_summary.csv"
)


def _safe_plot_filename(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = text.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9._()-]+", "_", text).strip("._")
    return text or "output"


def _normalize_target_for_eurofins_lookup(name):
    text = str(name).strip()
    if text.endswith("_res"):
        text = text[:-4]
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return text.casefold()


@lru_cache(maxsize=1)
def _load_eurofins_sample_length_map(summary_csv_path=str(EUROFINS_SUMMARY_DEFAULT_PATH)):
    path = Path(summary_csv_path)
    if not path.exists():
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    required = {"parameter", "median_hours_between_measurements"}
    if not required.issubset(set(df.columns)):
        return {}

    out = {}
    for _, row in df.iterrows():
        key = _normalize_target_for_eurofins_lookup(row.get("parameter", ""))
        if not key:
            continue
        value = pd.to_numeric(row.get("median_hours_between_measurements"), errors="coerce")
        if pd.isna(value) or float(value) <= 0:
            continue
        out[key] = max(1.0, float(value))
    return out


def _target_sample_length_hours(target_name, default_hours):
    default_value = max(1.0, float(default_hours))
    lookup_key = _normalize_target_for_eurofins_lookup(target_name)
    sample_len_map = _load_eurofins_sample_length_map()
    return float(sample_len_map.get(lookup_key, default_value))


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

    masks = [
        (day_diff <= day_windows[0]) & (hour_diff <= diurnal_window),
        day_diff <= day_windows[0],
        day_diff <= day_windows[1],
    ]
    for mask in masks:
        finite_vals = values[mask]
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        if finite_vals.size >= 3:
            return float(np.mean(finite_vals))

    # Last fallback: all rows not of the current year — no minimum sample size requirement
    finite_vals = values[np.isfinite(values)]
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

    fit_scale = 3.1

    for i in range(len(dataset)):
        _, y, filename = dataset[i]

        output_times = _get_output_times(data_dir, filename, output_rows, sample_subdir=sample_subdir)
        if len(output_times) == 0:
            continue

        forecast_start = output_times[0]
        window_end = forecast_start - pd.Timedelta(hours=gap_hours)
        pred_matrix = np.full((len(output_times), len(output_columns)), np.nan, dtype=float)

        for j, col in enumerate(output_columns):
            # Per-target handling prevents one sparse output from blanking all outputs.
            series = historical_df[col].dropna()
            if series.empty:
                continue

            history = series.loc[: window_end - pd.Timedelta(nanoseconds=1)]
            if history.empty:
                continue

            sample_length_hours = _target_sample_length_hours(col, default_hours=window_hours)
            fit_span_hours = max(1.0, float(fit_scale * sample_length_hours))
            fit_start = window_end - pd.Timedelta(hours=fit_span_hours)
            fit_series = history.loc[fit_start: window_end - pd.Timedelta(nanoseconds=1)]

            method = "persistence"
            if len(fit_series) >= 2:
                times = (fit_series.index - fit_start).total_seconds().to_numpy().reshape(-1, 1)
                forecast_secs = (output_times - fit_start).total_seconds().to_numpy().reshape(-1, 1)
                values = fit_series.to_numpy(dtype=float)

                model = LinearRegression()
                model.fit(times, values)
                pred_matrix[:, j] = model.predict(forecast_secs)
                method = "linear"
            else:
                # Fallback when span is too sparse: hold the last valid value.
                pred_matrix[:, j] = float(history.iloc[-1])

            if debug_plot and i < examples:
                plt.figure(figsize=(8, 5))
                plt.scatter(fit_series.index, fit_series.to_numpy(dtype=float), label="Training data", color="blue", alpha=0.6)

                if method == "linear":
                    fit_times = np.linspace(times.min(), times.max(), 100).reshape(-1, 1)
                    fit_dates = [fit_start + pd.Timedelta(seconds=s) for s in fit_times.flatten()]
                    plt.plot(fit_dates, model.predict(fit_times), "k--", label="Fitted line")

                plt.scatter(output_times, pred_matrix[:, j], color="orange", label="Predictions", zorder=5)
                y_arr = np.asarray(y).reshape(-1)
                plt.plot(
                    output_times,
                    y_arr[j::len(output_columns)],
                    color="green",
                    marker="o",
                    linestyle="",
                    label="Ground truth",
                    zorder=6,
                )

                plt.axvline(window_end, color="red", linestyle="--", label=f"Gap start (-{gap_hours}h)")
                plt.axvline(forecast_start, color="red", linestyle=":", label="Forecast start")
                plt.title(f"{col} ({method})")
                plt.xlabel("Timestamp")
                plt.ylabel(col)
                plt.legend()
                plt.tight_layout()
                safe_col = _safe_plot_filename(col)
                plt.savefig(
                    os.path.join(
                        os.path.join(data_dir, "forecasts", forecast_name, "linear"),
                        f"{filename}_{safe_col}_debug.png",
                    )
                )
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
            plt.legend(loc="best", fontsize=9)
            plt.tight_layout()
            safe_col = _safe_plot_filename(col)
            plot_path = os.path.join(plot_dir, f"seasonal_{safe_col}.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
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

def visualizer(
    *pred_target_pairs,
    labels=None,
    directory=None,
    forecast_name=None,
    num_samples=None,
    sample_labels=None,
    split_files_by_pair=None,
    collapse_error_points_by_pair=None,
):
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

    n_sets = len(pred_target_pairs)
    if labels is None:
        labels = [f"Set {i+1}" for i in range(n_sets)]
    elif len(labels) != n_sets:
        raise ValueError("Length of labels must match number of result sets.")

    if split_files_by_pair is None:
        split_files_by_pair = [None] * n_sets
    elif len(split_files_by_pair) != n_sets:
        raise ValueError("Length of split_files_by_pair must match number of result sets.")

    if collapse_error_points_by_pair is None:
        collapse_error_points_by_pair = [False] * n_sets
    elif len(collapse_error_points_by_pair) != n_sets:
        raise ValueError("Length of collapse_error_points_by_pair must match number of result sets.")

    def _aligned_flat(preds, targets, limit=None):
        preds_flat = np.asarray(preds).reshape(-1)
        targets_flat = np.asarray(targets).reshape(-1)
        if limit is not None:
            preds_flat = preds_flat[:limit]
            targets_flat = targets_flat[:limit]
        n = min(len(preds_flat), len(targets_flat))
        return preds_flat[:n], targets_flat[:n]

    def _aligned_matrix(preds, targets, row_limit=None):
        pred_arr = np.asarray(preds)
        target_arr = np.asarray(targets)

        if pred_arr.ndim > 2:
            pred_arr = pred_arr.reshape(pred_arr.shape[0], -1)
        if target_arr.ndim > 2:
            target_arr = target_arr.reshape(target_arr.shape[0], -1)
        if pred_arr.ndim == 1:
            pred_arr = pred_arr.reshape(-1, 1)
        if target_arr.ndim == 1:
            target_arr = target_arr.reshape(-1, 1)

        if row_limit is not None:
            pred_arr = pred_arr[:row_limit]
            target_arr = target_arr[:row_limit]

        n_rows = min(pred_arr.shape[0], target_arr.shape[0])
        n_cols = min(pred_arr.shape[1], target_arr.shape[1])
        if n_rows <= 0 or n_cols <= 0:
            return pred_arr[:0], target_arr[:0], 0, 0

        pred_arr = pred_arr[:n_rows, :n_cols]
        target_arr = target_arr[:n_rows, :n_cols]
        return pred_arr, target_arr, n_rows, n_cols

    def _collapse_errors_by_base_sample(preds, targets, split_files, row_limit=None):
        pred_arr, target_arr, n_rows, n_cols = _aligned_matrix(preds, targets, row_limit=row_limit)
        if n_rows <= 0 or n_cols <= 0:
            return np.array([], dtype=float)

        if not split_files:
            return np.array([], dtype=float)

        aligned_rows = min(n_rows, len(split_files))
        if aligned_rows <= 0:
            return np.array([], dtype=float)

        pred_arr = pred_arr[:aligned_rows, :]
        target_arr = target_arr[:aligned_rows, :]

        grouped_errors = {}
        group_order = []
        for row_idx in range(aligned_rows):
            group_id = re.sub(r"_mc_\d+(?=\.csv$)", "", Path(str(split_files[row_idx])).name)
            if group_id not in grouped_errors:
                grouped_errors[group_id] = []
                group_order.append(group_id)

            row_errors = pred_arr[row_idx, :] - target_arr[row_idx, :]
            finite_row_errors = row_errors[np.isfinite(row_errors)]
            if finite_row_errors.size > 0:
                grouped_errors[group_id].append(float(np.mean(finite_row_errors)))

        collapsed_errors = []
        for group_id in group_order:
            values = grouped_errors[group_id]
            if values:
                collapsed_errors.append(float(np.mean(values)))

        return np.asarray(collapsed_errors, dtype=float)

    def _marker_for_series(index):
        # Keep marker assignment deterministic so model shapes stay stable across runs.
        marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
        return marker_cycle[index % len(marker_cycle)]

    def _scalarize_row_mean_finite(row_values):
        arr = np.asarray(row_values, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.nan
        return float(np.mean(finite))

    def _strip_base_sample_id(sample_id):
        name = Path(str(sample_id)).name if sample_id is not None else ""
        if not name:
            return ""
        no_mc = re.sub(r"_mc_\d+(?=\.csv$)", "", name)
        no_mc = re.sub(r"_mc_\d+$", "", Path(no_mc).stem)
        return no_mc

    def _resolve_sample_csv_path(split_entry):
        if split_entry is None:
            return None

        entry = Path(str(split_entry))
        candidates = []
        if entry.is_absolute():
            candidates.append(entry)

        if directory is not None:
            data_root = Path(directory)
            candidates.append(data_root / entry)
            candidates.append(data_root / "samples" / entry.name)
            candidates.append(data_root / "mc_replicates" / entry.name)

        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate
            except Exception:
                continue
        return None

    _series_label_cache = {}

    def _extract_date_label_from_final_timestamp(base_id, split_entry):
        key = str(split_entry) if split_entry is not None else ""
        if key in _series_label_cache:
            return _series_label_cache[key]

        fallback = base_id if base_id else "unknown"
        sample_csv = _resolve_sample_csv_path(split_entry)
        if sample_csv is None:
            _series_label_cache[key] = fallback
            return fallback

        try:
            sample_df = pd.read_csv(sample_csv, usecols=["TIMESTAMP"])
            if sample_df.empty:
                _series_label_cache[key] = fallback
                return fallback
            ts = pd.to_datetime(sample_df["TIMESTAMP"], errors="coerce")
            ts = ts.dropna()
            if ts.empty:
                _series_label_cache[key] = fallback
                return fallback
            label = str(ts.iloc[-1].date())
            _series_label_cache[key] = label
            return label
        except Exception:
            _series_label_cache[key] = fallback
            return fallback

    def _collect_prediction_values_by_series(preds, targets, split_files, row_limit=None, collapse_replicates=False):
        pred_arr, target_arr, n_rows, n_cols = _aligned_matrix(preds, targets, row_limit=row_limit)
        if n_rows <= 0 or n_cols <= 0:
            return [], {}, {}, {}

        sample_entries = list(split_files) if split_files else []
        row_count = min(n_rows, len(sample_entries)) if sample_entries else n_rows
        if row_count <= 0:
            return [], {}, {}, {}

        ordered_ids = []
        seen_ids = set()
        point_values = {}
        target_values = {}
        target_counts = {}
        series_labels = {}

        for row_idx in range(row_count):
            split_entry = sample_entries[row_idx] if row_idx < len(sample_entries) else None
            if split_entry is None:
                base_id = f"row_{row_idx + 1}"
            else:
                base_id = _strip_base_sample_id(split_entry)
                if not base_id:
                    base_id = f"row_{row_idx + 1}"

            pred_scalar = _scalarize_row_mean_finite(pred_arr[row_idx, :])
            target_scalar = _scalarize_row_mean_finite(target_arr[row_idx, :])
            if not np.isfinite(pred_scalar) or not np.isfinite(target_scalar):
                continue

            if base_id not in seen_ids:
                seen_ids.add(base_id)
                ordered_ids.append(base_id)
                point_values[base_id] = []
                target_values[base_id] = 0.0
                target_counts[base_id] = 0
                series_labels[base_id] = _extract_date_label_from_final_timestamp(base_id, split_entry)

            point_values[base_id].append(float(pred_scalar))
            target_values[base_id] += float(target_scalar)
            target_counts[base_id] += 1

        filtered_ids = []
        for base_id in ordered_ids:
            count = target_counts.get(base_id, 0)
            if count <= 0:
                continue
            values = point_values.get(base_id, [])
            if not values:
                continue
            if collapse_replicates:
                point_values[base_id] = [float(np.mean(values))]
            target_values[base_id] = float(target_values[base_id] / count)
            filtered_ids.append(base_id)

        return filtered_ids, point_values, target_values, series_labels

    # === Scatter plot of predictions vs actuals ===
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette("husl", len(pred_target_pairs))
    min_val, max_val = float("inf"), float("-inf")
    metrics = []  # store (label, MAE, RMSE, r, R2)


    for i, (preds, targets) in enumerate(pred_target_pairs):
        preds, targets = _aligned_flat(preds, targets, limit=num_samples)
        label = labels[i] if labels else f"Model {i+1}"
        if len(preds) == 0:
            metrics.append((label, np.nan, np.nan, np.nan, np.nan))
            print(f"{label}: no valid data for metrics")
            continue

        mask = np.isfinite(preds) & np.isfinite(targets)
        model_marker = _marker_for_series(i)
        # If sample_labels is provided, plot train/test with different symbology
        if sample_labels is not None and len(sample_labels) == len(preds):
            for group, color in [("train", "#1f77b4"), ("test", "#ff7f0e")]:
                group_mask = (np.array(sample_labels) == group) & mask
                if np.any(group_mask):
                    ax.scatter(
                        targets[group_mask],
                        preds[group_mask],
                        label=f"{label} ({group})",
                        alpha=0.7,
                        marker=model_marker,
                        color=color,
                    )
                    min_val = min(min_val, np.nanmin(targets[group_mask]), np.nanmin(preds[group_mask]))
                    max_val = max(max_val, np.nanmax(targets[group_mask]), np.nanmax(preds[group_mask]))
        else:
            if mask.any():
                ax.scatter(targets[mask], preds[mask], label=label, alpha=0.7, color=colors[i], marker=model_marker)
                min_val = min(min_val, np.nanmin(targets[mask]), np.nanmin(preds[mask]))
                max_val = max(max_val, np.nanmax(targets[mask]), np.nanmax(preds[mask]))
        if mask.any():
            target_vals = targets[mask]
            pred_vals = preds[mask]
            mae = mean_absolute_error(target_vals, pred_vals)
            rmse = np.sqrt(mean_squared_error(target_vals, pred_vals))
            if len(target_vals) > 1:
                r2 = r2_score(target_vals, pred_vals)
            else:
                r2 = np.nan
            if len(target_vals) > 1 and np.std(target_vals) > 0 and np.std(pred_vals) > 0:
                r_value = float(np.corrcoef(target_vals, pred_vals)[0, 1])
            else:
                r_value = np.nan
            metrics.append((label, mae, rmse, r_value, r2))
            print(f"{label}: MAE={mae:.4f}, RMSE={rmse:.4f}, r={r_value:.4f}, R²={r2:.4f}")
            print()
        else:
            metrics.append((label, np.nan, np.nan, np.nan, np.nan))
            print(f"{label}: no valid data for metrics")

    if not np.isfinite(min_val) or not np.isfinite(max_val):
        min_val, max_val = 0.0, 1.0

    if max_val > min_val:
        pad = 0.03 * (max_val - min_val)
    else:
        pad = 0.03 * max(1.0, abs(max_val))
    axis_min = min_val - pad
    axis_max = max_val + pad

    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="red", linestyle="--")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted Value")
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(directory, "forecasts", forecast_name, "predictions.png"))
    plt.close(fig)

    # Prediction series by sample, preserving sample order and optional baseline replicate collapse.
    series_order = []
    series_index = {}
    target_by_sample = {}
    sample_label_map = {}
    model_points = {}
    plottable = False

    for idx, ((preds, targets), label) in enumerate(zip(pred_target_pairs, labels)):
        split_files = split_files_by_pair[idx]
        collapse_replicates = bool(collapse_error_points_by_pair[idx])
        ids, pred_points, target_points, sample_labels_map = _collect_prediction_values_by_series(
            preds,
            targets,
            split_files,
            row_limit=num_samples,
            collapse_replicates=collapse_replicates,
        )
        if not ids:
            continue

        model_points[label] = pred_points
        plottable = True

        for base_id in ids:
            if base_id not in series_index:
                series_index[base_id] = len(series_order)
                series_order.append(base_id)
            if base_id not in target_by_sample:
                target_by_sample[base_id] = float(target_points[base_id])
            if base_id not in sample_label_map:
                sample_label_map[base_id] = sample_labels_map.get(base_id, base_id)

    if plottable and series_order:
        fig_w = max(10.0, 0.5 * len(series_order) + 6.0)
        fig, ax = plt.subplots(figsize=(fig_w, 6.5))

        x_positions = np.arange(len(series_order), dtype=float)
        for x, base_id in zip(x_positions, series_order):
            target_val = target_by_sample.get(base_id)
            if target_val is None or not np.isfinite(target_val):
                continue
            ax.hlines(
                y=target_val,
                xmin=x - 0.36,
                xmax=x + 0.36,
                color="red",
                linewidth=4,
                alpha=0.85,
                zorder=1,
            )

        legend_handles = [Line2D([0], [0], color="red", linewidth=4, label="Target")]
        for idx, label in enumerate(labels):
            if label not in model_points:
                continue
            marker = _marker_for_series(idx)
            color = colors[idx]
            legend_handles.append(
                Line2D([0], [0], marker=marker, linestyle="", markersize=7, color=color, label=label)
            )

            pred_points = model_points[label]
            for base_id in series_order:
                values = pred_points.get(base_id, [])
                if not values:
                    continue
                x_center = float(series_index[base_id])
                if len(values) > 1:
                    jitter = np.linspace(-0.18, 0.18, num=len(values))
                else:
                    jitter = np.array([0.0])
                x_vals = x_center + jitter
                y_vals = np.asarray(values, dtype=float)
                finite_mask = np.isfinite(y_vals)
                if not finite_mask.any():
                    continue
                ax.scatter(
                    x_vals[finite_mask],
                    y_vals[finite_mask],
                    marker=marker,
                    color=color,
                    alpha=0.85,
                    s=40,
                    zorder=2,
                )

        xtick_labels = [sample_label_map.get(base_id, base_id) for base_id in series_order]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Predicted Value")
        ax.legend(handles=legend_handles)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(Path(directory, "forecasts", forecast_name, "prediction_series.png"))
        plt.close(fig)
    else:
        print("[WARN] Skipping prediction series plot: no finite per-sample values available.")

    # === Metrics summary figure ===
    if metrics:
        labels_m = [m[0] for m in metrics]
        mae_vals = [m[1] for m in metrics]
        rmse_vals = [m[2] for m in metrics]
        r_vals = [m[3] for m in metrics]
        r2_vals = [m[4] for m in metrics]
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        bar_kwargs = dict(alpha=0.7)
        plot_defs = [
            (ax[0, 0], mae_vals, "MAE"),
            (ax[0, 1], rmse_vals, "RMSE"),
            (ax[1, 0], r_vals, "r"),
            (ax[1, 1], r2_vals, "R²"),
        ]
        for a, values, ylabel in plot_defs:
            a.bar(labels_m, values, color=colors, **bar_kwargs)
            a.set_ylabel(ylabel)
            if ylabel == "R²":
                a.set_ylim(bottom=-0.1)
            a.set_xticks(range(len(labels_m)))
            a.set_xticklabels(labels_m, rotation=30, ha="right")
            a.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(Path(directory, "forecasts", forecast_name, "metrics_summary.png"))
        plt.close(fig)

    # Prepare combined error data (supports different lengths per model)
    combined_frames = []
    for idx, ((pred, target), label) in enumerate(zip(pred_target_pairs, labels)):
        collapse_errors = bool(collapse_error_points_by_pair[idx])
        split_files = split_files_by_pair[idx]

        if collapse_errors:
            # Baseline series are deterministic across MC replicates, so show one error point per base sample.
            errors = _collapse_errors_by_base_sample(pred, target, split_files, row_limit=num_samples)
            if errors.size == 0:
                # Fallback preserves legacy behavior if split-file mapping is unavailable.
                pred_flat, target_flat = _aligned_flat(pred, target)
                if len(pred_flat) == 0:
                    continue
                errors = pred_flat - target_flat
                errors = errors[np.isfinite(errors)]
        else:
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
        n_datasets = max(1, int(df_long_combined["Dataset"].nunique()))
        fig_w = max(6.0, min(12.0, 3.0 + 1.1 * n_datasets))
        plt.figure(figsize=(fig_w, 6))
        ax = plt.gca()
        sns.boxplot(x="Dataset", y="Error", data=df_long_combined,
                    showcaps=True, boxprops={'facecolor': 'lightgray', 'alpha': 0.3, 'linewidth': 0.5},
                    whiskerprops={'linewidth': 0.5}, medianprops={'color': 'blue', 'linewidth': 1}, showfliers=False, ax=ax)

        sns.stripplot(x="Dataset", y="Error", data=df_long_combined,
                      jitter=True, size=6, color='black', ax=ax)

        for artist in ax.collections:
            artist.set_facecolor('red')
            artist.set_edgecolor('red')

        ax.set_ylabel("Error (Absolute)")
        ax.set_xlabel("Model")
        ax.margins(x=0.02)
        plt.tight_layout()
        plt.savefig(
            Path(directory, "forecasts", forecast_name, "boxplot.png"),
            bbox_inches="tight",
            pad_inches=0.1,
        )
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

        ax.set_ylabel("Error")
        ax.set_xlabel("Column")
        ax.margins(x=0.02)
        plt.tight_layout()
        plt.savefig(
            Path(directory, "forecasts", forecast_name, f"{label.replace(' ', '_')}_overlay_emphasized.png"),
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()


def boxplot_from_error_rows(error_rows_df, directory, forecast_name):
    """Render boxplot.png from precomputed long-form error rows.

    Expected columns: Dataset, Error, Kind.
    """
    if error_rows_df is None or len(error_rows_df) == 0:
        print("[WARN] Skipping error distribution plot: no rows provided.")
        return

    df = error_rows_df.copy()
    required = {"Dataset", "Error", "Kind"}
    if not required.issubset(set(df.columns)):
        print("[WARN] Skipping error distribution plot: missing required columns.")
        return

    df["Dataset"] = df["Dataset"].astype(str)
    df["Kind"] = df["Kind"].fillna("unknown").astype(str)
    df["Error"] = pd.to_numeric(df["Error"], errors="coerce")
    df = df[np.isfinite(df["Error"])]
    if df.empty:
        print("[WARN] Skipping error distribution plot: no finite errors available.")
        return

    n_datasets = max(1, int(df["Dataset"].nunique()))
    fig_w = max(6.0, min(12.0, 3.0 + 1.1 * n_datasets))
    plt.figure(figsize=(fig_w, 6))
    ax = plt.gca()

    sns.boxplot(
        x="Dataset",
        y="Error",
        data=df,
        showcaps=True,
        boxprops={"facecolor": "lightgray", "alpha": 0.3, "linewidth": 0.5},
        whiskerprops={"linewidth": 0.5},
        medianprops={"color": "blue", "linewidth": 1},
        showfliers=False,
        ax=ax,
    )

    sns.stripplot(
        x="Dataset",
        y="Error",
        hue="Kind",
        data=df,
        jitter=True,
        size=5,
        alpha=0.8,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        unique = {}
        for handle, label in zip(handles, labels):
            if label not in unique:
                unique[label] = handle
        ax.legend(unique.values(), unique.keys(), title="Kind")

    ax.set_ylabel("Error (Prediction - Target)")
    ax.set_xlabel("Model")
    ax.margins(x=0.02)
    plt.tight_layout()
    plt.savefig(
        Path(directory, "forecasts", forecast_name, "boxplot.png"),
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()

def classification_visualizer(*pred_target_pairs, labels=None, directory='.', forecast_name='Classifier',
                              num_samples=None):
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
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(Path(directory, "forecasts", forecast_name, "classification", f"confusion.png"))
    plt.close()

    # Combined ROC Curve Plot
    if roc_data:
        plt.figure(figsize=(6, 5))
        for label, fpr, tpr, roc_auc in roc_data:
            plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
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

"""
Compares accuracy of Time Series Forecasting using various forecasts.
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
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc)
from utils.preprocessing import normalize_columns
from utils.training import load_samples

from utils.transformer import TimeSeriesTargetDataset, TimeSeriesTransformer
from utils.evaluation import (load_secondary, evaluate_naive, evaluate_seasonal, evaluate_linear, evaluate_transformer,
                              visualizer, classification_visualizer, reverse_normalize, binarize_predictions)

# def evaluate_model(model, dataset):
#     model.eval()
#     predictions = []
#     targets = []
#
#     with torch.no_grad():
#         for i in range(len(dataset)):
#             x, y, filename = dataset[i]
#             x = x.unsqueeze(0).to(device)  # Add batch dimension
#             pred = model(x).squeeze().cpu().numpy()
#             # Ensure pred is a 1-D numpy array (even when scalar)
#             pred = np.array(pred).reshape(-1)
#
#             true = y.cpu().numpy()
#             # Ensure true is also a 1-D numpy array
#             true = np.array(true).reshape(-1)
#
#             predictions.append(pred)
#             targets.append(true)
#
#     # Turn lists of 1-D arrays into 2-D arrays of shape (n_samples, n_outputs)
#     if len(predictions) == 0:
#         return np.empty((0, 0)), np.empty((0, 0))
#
#     predictions = np.array(predictions)
#     targets = np.array(targets)
#
#     # If they ended up 1-D (happens when numpy collapses to shape (n_samples,)),
#     # convert to shape (n_samples, 1)
#     if predictions.ndim == 1:
#         predictions = predictions.reshape(-1, 1)
#     if targets.ndim == 1:
#         targets = targets.reshape(-1, 1)
#
#     mask = np.isfinite(predictions).all(axis=1) & np.isfinite(targets).all(axis=1)
#     return predictions[mask], targets[mask]
#
# def evaluate_naive(dataset, historic, output_columns, data_dir, output_rows=-1, gap_hours=5):
#
#     # Load lookup table for baseline model
#     df = pd.read_csv(historic,parse_dates=["TIMESTAMP"])
#     sort_df = df.sort_values("TIMESTAMP")
#     historical_df = normalize_columns(sort_df, output_columns, save=False, directory=Path(data_dir, 'examples_naive'))
#
#     predictions, targets = [], []
#     for i in range(len(dataset)):
#         _, y, filename = dataset[i]
#
#         # Load the sample file to get the output times
#         sample_df = pd.read_csv(os.path.join(data_dir, 'samples', filename), parse_dates=["TIMESTAMP"])
#         output_times = sample_df["TIMESTAMP"].iloc[output_rows:]
#
#         # Apply gap constraint (before the first output timestamp)
#         cutoff_time = output_times.iloc[0] - pd.Timedelta(hours=gap_hours)
#
#         # Filter historical data before cutoff_time
#         earlier_values = historical_df[historical_df["TIMESTAMP"] < cutoff_time][output_columns]
#
#         # Drop NaN values and select the last valid row
#         valid_values = earlier_values.dropna()
#         if valid_values.empty:
#             baseline_pred = np.full((len(output_times), len(output_columns)), np.nan)
#         else:
#             last_value = valid_values.iloc[-1].values
#             # Repeat last known value for each output timestamp
#             baseline_pred = np.tile(last_value, (len(output_times), 1))
#
#         predictions.append(baseline_pred.reshape(-1))
#         targets.append(y.numpy().reshape(-1))  # Ensure target is flattened
#
#     return np.array(predictions), np.array(targets)
#
# def evaluate_linear(data_dir, forecast_name, dataset, historic, output_columns, output_rows=-1, window_hours=6,
#                     gap_hours=5, debug_plot=False, examples=10):
#     """
#     Linear baseline with causal gap constraint and optional debug visualization.
#
#     Now also plots the true target values (ground truth) in green for comparison.
#     """
#     # Load full time series as input for simple "baseline" forecasts
#     df = pd.read_csv(historic, parse_dates=["TIMESTAMP"])
#     sort_df = df.sort_values("TIMESTAMP")
#     historical_df = normalize_columns(sort_df, output_columns, save=False, directory=Path(data_dir, 'linear'))
#
#     if debug_plot:
#         os.makedirs(os.path.join(data_dir, "forecasts", forecast_name, "linear"), exist_ok=True)
#
#     predictions, targets = [], []
#
#     for i in range(len(dataset)):
#         _, y, filename = dataset[i]
#
#         # Load sample to get output timestamps
#         sample_df = pd.read_csv(os.path.join(data_dir, 'samples', filename), parse_dates=["TIMESTAMP"])
#         output_times = sample_df["TIMESTAMP"].iloc[output_rows:]
#
#         # Forecast window definition
#         forecast_start = output_times.iloc[0]
#         window_end = forecast_start - pd.Timedelta(hours=gap_hours)
#         window_start = window_end - pd.Timedelta(hours=window_hours)
#
#         # Select regression window
#         window_df = historical_df[
#             (historical_df["TIMESTAMP"] >= window_start) &
#             (historical_df["TIMESTAMP"] < window_end)
#         ][["TIMESTAMP"] + output_columns].dropna(subset=output_columns)
#
#         if window_df.empty:
#             pred_matrix = np.full((len(output_times), len(output_columns)), np.nan)
#         else:
#             times = (window_df["TIMESTAMP"] - window_start).dt.total_seconds().values.reshape(-1, 1)
#             pred_matrix = np.zeros((len(output_times), len(output_columns)))
#
#             for j, col in enumerate(output_columns):
#                 values = window_df[col].values
#                 if len(values) == 0 or np.isnan(values).all():
#                     pred_matrix[:, j] = np.nan
#                 else:
#                     model = LinearRegression()
#                     model.fit(times, values)
#
#                     # Predict for each future timestamp
#                     forecast_secs = (output_times - window_start).dt.total_seconds().values.reshape(-1, 1)
#                     pred_matrix[:, j] = model.predict(forecast_secs)
#
#                     # === Debug plot for this variable ===
#                     if debug_plot and i < examples:
#                         plt.figure(figsize=(8, 5))
#                         # Historical training data
#                         plt.scatter(window_df["TIMESTAMP"], values, label="Training data", color="blue", alpha=0.6)
#                         # Regression line (continuous fit within training window)
#                         fit_times = np.linspace(times.min(), times.max(), 100).reshape(-1, 1)
#                         fit_dates = [window_start + pd.Timedelta(seconds=s) for s in fit_times.flatten()]
#                         plt.plot(fit_dates, model.predict(fit_times), "k--", label="Fitted line")
#                         # Forecast predictions
#                         plt.scatter(output_times, pred_matrix[:, j], color="orange", label="Predictions", zorder=5)
#                         # === NEW: plot ground truth (targets) ===
#                         plt.plot(output_times, y.numpy().reshape(-1)[j::len(output_columns)],
#                                  color="green", marker="o", linestyle="", label="Ground truth", zorder=6)
#
#                         # Reference verticals
#                         plt.axvline(window_end, color="red", linestyle="--", label=f"Gap start (-{gap_hours}h)")
#                         plt.axvline(forecast_start, color="red", linestyle=":", label="Forecast start")
#
#                         plt.title(f"Linear Regression Forecast — {col}\nSample: {filename}")
#                         plt.xlabel("Timestamp")
#                         plt.ylabel(col)
#                         plt.legend()
#                         plt.tight_layout()
#                         plt.savefig(os.path.join(os.path.join(data_dir, "forecasts", forecast_name, "linear"),
#                                                  f"{filename}_{col}_debug.png"))
#                         plt.close()
#
#         predictions.append(pred_matrix.reshape(-1))
#         targets.append(y.numpy().reshape(-1))
#
#     return np.array(predictions), np.array(targets)
#
# def evaluate_seasonal(dataset, historic, output_columns, data_dir, forecast_name, output_rows=-1, diurnal_window=2,
#                       secondary=None):
#     """
#     Seasonal baseline using either full `historical_df` or, if available,
#     a secondary CSV file for specific output columns.
#
#     Now also uses *all available values* from the chosen data source
#     (secondary or primary) as ground truth in the diagnostic plots,
#     separated into yearly series.
#     """
#     import os
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#
#     # --- Load and prepare data sources ---
#     df = pd.read_csv(historic, parse_dates=["TIMESTAMP"]).sort_values("TIMESTAMP")
#     historical_df = normalize_columns(df, output_columns, directory=data_dir)
#
#     # --- Load secondary data (if provided) ---
#     secondary_df = None
#     if secondary and os.path.exists(secondary):
#         try:
#             secondary_df = pd.read_csv(secondary, sep=";", decimal=".", parse_dates=["Time"]).sort_values("Time")
#         except:
#             secondary_df = pd.read_csv(secondary, sep=";", decimal=",", parse_dates=["Time"]).sort_values("Time")
#         secondary_df.rename(columns={"Time": "TIMESTAMP"}, inplace=True)
#         secondary_df = normalize_columns(secondary_df, output_columns, directory=data_dir)
#
#     def prepare_time_columns(df):
#         df["YEAR"] = df["TIMESTAMP"].dt.year
#         df["DAYOFYEAR"] = df["TIMESTAMP"].dt.dayofyear
#         df["HOUR"] = df["TIMESTAMP"].dt.hour + df["TIMESTAMP"].dt.minute / 60.0
#         return df
#
#     historical_df = prepare_time_columns(historical_df)
#     if secondary_df is not None:
#         secondary_df = prepare_time_columns(secondary_df)
#
#     predictions, targets = [], []
#
#     # === Predict values for each sample ===
#     for i in range(len(dataset)):
#         _, y, filename = dataset[i]
#         sample_df = pd.read_csv(os.path.join(data_dir, 'samples', filename), parse_dates=["TIMESTAMP"])
#         output_times = sample_df["TIMESTAMP"].iloc[output_rows:]
#         if len(output_times) == 0:
#             continue
#
#         pred_matrix = np.zeros((len(output_times), len(output_columns)))
#
#         for t_idx, ts in enumerate(output_times):
#             target_year = ts.year
#             target_day = ts.timetuple().tm_yday
#             target_hour = ts.hour + ts.minute / 60.0
#
#             for j, col in enumerate(output_columns):
#                 # Select data source
#                 if secondary_df is not None and col in secondary_df.columns:
#                     src_df = secondary_df
#                 elif col in historical_df.columns:
#                     src_df = historical_df
#                 else:
#                     pred_matrix[t_idx, j] = np.nan
#                     continue
#
#                 candidates = src_df[src_df["YEAR"] != target_year].copy()
#                 if candidates.empty:
#                     pred_matrix[t_idx, j] = np.nan
#                     continue
#
#                 day_diff = np.abs(candidates["DAYOFYEAR"] - target_day)
#                 day_diff = np.minimum(day_diff, 365 - day_diff)
#                 candidates["DAY_DIFF"] = day_diff
#                 candidates["HOUR_DIFF"] = np.abs(candidates["HOUR"] - target_hour)
#
#                 subset = candidates[(candidates["DAY_DIFF"] <= 4) & (candidates["HOUR_DIFF"] <= diurnal_window)]
#                 if subset.empty:
#                     subset = candidates[candidates["DAY_DIFF"] <= 4]
#                 if subset.empty:
#                     subset = candidates[candidates["DAY_DIFF"] <= 15]
#                 if subset.empty:
#                     subset = candidates
#
#                 vals = subset[[col]].dropna()
#                 pred_matrix[t_idx, j] = vals[col].mean() if not vals.empty else np.nan
#
#         predictions.append(pred_matrix.reshape(-1))
#         targets.append(y.numpy().reshape(-1))
#
#     predictions = np.array(predictions)
#     targets = np.array(targets)
#
#     # === Diagnostic plot ===
#     try:
#         plot_dir = os.path.join(data_dir, "forecasts", forecast_name, "seasonal")
#         os.makedirs(plot_dir, exist_ok=True)
#         sns.set_style("whitegrid")
#
#         # Synthetic hourly timeline for predicted curve
#         synthetic_year = int(historical_df["YEAR"].max()) + 1
#         start = pd.Timestamp(year=synthetic_year, month=1, day=1, hour=0)
#         synthetic_times = pd.date_range(start=start, periods=24 * 366, freq="h")
#         synth_day = synthetic_times.dayofyear
#         synth_hour = synthetic_times.hour + synthetic_times.minute / 60.0
#         synth_dayofyear = synth_day + synth_hour / 24.0
#
#         ref_year = 2021
#         month_starts = pd.date_range(f"{ref_year}-01-01", f"{ref_year}-12-31", freq="MS")
#         month_dayofyear = [d.timetuple().tm_yday for d in month_starts]
#         month_labels = [d.strftime("%b") for d in month_starts]
#
#         for col in output_columns:
#             # Choose data source
#             if secondary_df is not None and col in secondary_df.columns:
#                 src_df = secondary_df
#             else:
#                 src_df = historical_df
#
#             # --- Compute continuous predicted curve ---
#             continuous_vals = []
#             for idx in range(len(synthetic_times)):
#                 target_day = synth_day[idx]
#                 target_hour = synth_hour[idx]
#
#                 candidates = src_df.copy()
#                 day_diff = np.abs(candidates["DAYOFYEAR"] - target_day)
#                 day_diff = np.minimum(day_diff, 365 - day_diff)
#                 candidates["DAY_DIFF"] = day_diff
#                 candidates["HOUR_DIFF"] = np.abs(candidates["HOUR"] - target_hour)
#
#                 subset = candidates[(candidates["DAY_DIFF"] <= 4) & (candidates["HOUR_DIFF"] <= diurnal_window)]
#                 if subset.empty:
#                     subset = candidates[candidates["DAY_DIFF"] <= 4]
#                 if subset.empty:
#                     subset = candidates[candidates["DAY_DIFF"] <= 15]
#                 if subset.empty:
#                     subset = candidates
#
#                 vals = subset[[col]].dropna()
#                 continuous_vals.append(vals[col].mean() if not vals.empty else np.nan)
#
#             # --- Build ground truth series for all years from source ---
#             gt_df = src_df[["YEAR", "DAYOFYEAR", "HOUR", col]].dropna()
#             gt_df["DAYOFYEAR"] = gt_df["DAYOFYEAR"] + gt_df["HOUR"] / 24.0
#
#             # --- Plot ---
#             plt.figure(figsize=(12, 6))
#
#             # Ground truth lines (all years)
#             for yr, group in gt_df.groupby("YEAR"):
#                 g = group.sort_values("DAYOFYEAR")
#                 plt.plot(
#                     g["DAYOFYEAR"],
#                     g[col],
#                     marker="o",
#                     linestyle="-",
#                     label=f"Actual {yr}",
#                     alpha=0.6,
#                 )
#
#             # Predicted continuous curve
#             y_vals = pd.Series(continuous_vals).interpolate().bfill().ffill()
#             x_vals = np.array(synth_dayofyear, dtype=float)
#
#             # Prevent wraparound line (no Dec→Jan jump)
#             jump_mask = np.diff(x_vals) < 0
#             if np.any(jump_mask):
#                 jump_idx = np.where(jump_mask)[0][0] + 1
#                 plt.plot(x_vals[:jump_idx], y_vals[:jump_idx],
#                          color="black", linewidth=2.5, label="Predicted")
#                 plt.plot(x_vals[jump_idx:], y_vals[jump_idx:], color="black", linewidth=2.5)
#             else:
#                 plt.plot(x_vals, y_vals, color="black", linewidth=2.5, label="Predicted")
#
#             plt.xticks(month_dayofyear, month_labels)
#             plt.xlim(0, 366)
#             plt.xlabel("Month")
#             plt.ylabel(col)
#             plt.title(f"Seasonality-based Model for {col}")
#             plt.legend(loc="best", fontsize=9)
#             plt.tight_layout()
#             plt.savefig(os.path.join(plot_dir, f"seasonal_{col}.png"))
#             plt.close()
#     except Exception as e:
#         print(f"[Warning] Could not generate plot of seasonality model: {e}")
#
#     return predictions, targets
#
# def binarize_predictions(preds, output_columns, thresholds_df):
#     """
#     Convert regression outputs to binary classification based on column-wise thresholds.
#     """
#     preds = np.array(preds)
#     # Ensure shape is (n_samples, n_outputs)
#     if preds.ndim == 1:
#         preds = preds.reshape(-1, len(output_columns))
#
#     binarized = np.zeros_like(preds, dtype=int)
#
#     for i, col in enumerate(output_columns):
#         if col not in thresholds_df.columns:
#             raise ValueError(f"Threshold for column '{col}' not found in thresholds_df.")
#         threshold = thresholds_df[col].iloc[0]
#         binarized[:, i] = (preds[:, i] > threshold).astype(int)
#     return binarized
#
# def visualizer(*pred_target_pairs, labels=None, directory, forecast_name, num_samples=200):
#     """
#     Visualize predictions and targets for a range of gaps from time series.
#     :param pred_target_pairs:
#     :param labels:
#     :param directory:
#     :param forecast_name:
#     :param num_samples:
#     :return:
#     """
#     sns.set_style("whitegrid")
#
#     # === Scatter plot of predictions vs actuals ===
#     fig, ax = plt.subplots(figsize=(8, 8))
#     colors = sns.color_palette("husl", len(pred_target_pairs))
#     min_val, max_val = float("inf"), float("-inf")
#     metrics = []  # store (label, MAE, RMSE, R2)
#
#     for i, (preds, targets) in enumerate(pred_target_pairs):
#         preds = np.array(preds)
#         targets = np.array(targets)
#         preds = preds[:num_samples].reshape(-1)
#         targets = targets[:num_samples].reshape(-1)
#         label = labels[i] if labels else f"Model {i+1}"
#         ax.scatter(targets, preds, label=label, alpha=0.7, color=colors[i])
#         min_val = min(min_val, np.nanmin(targets), np.nanmin(preds))
#         max_val = max(max_val, np.nanmax(targets), np.nanmax(preds))
#         mask = np.isfinite(preds) & np.isfinite(targets)
#         if mask.any():
#             mae = mean_absolute_error(targets[mask], preds[mask])
#             rmse = np.sqrt(mean_squared_error(targets[mask], preds[mask]))
#             r2 = r2_score(targets[mask], preds[mask])
#             metrics.append((label, mae, rmse, r2))
#             print(f"{label}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
#             print()
#         else:
#             metrics.append((label, np.nan, np.nan, np.nan))
#             print(f"{label}: no valid data for metrics")
#
#     ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
#     ax.set_xlabel("Ground truth")
#     ax.set_ylabel("Predicted Value")
#     ax.set_title("ML & Baseline Models")
#     ax.set_xlim(min_val, max_val)
#     ax.set_ylim(min_val, max_val)
#     ax.set_aspect("equal", adjustable="box")
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(Path(directory, "forecasts", forecast_name, "predictions.png"))
#     plt.close(fig)
#
#     # === Metrics summary figure ===
#     if metrics:
#         labels_m = [m[0] for m in metrics]
#         mae_vals = [m[1] for m in metrics]
#         rmse_vals = [m[2] for m in metrics]
#         r2_vals = [m[3] for m in metrics]
#         fig, ax = plt.subplots(1, 3, figsize=(14, 5))
#         bar_kwargs = dict(alpha=0.7)
#         ax[0].bar(labels_m, mae_vals, color=colors, **bar_kwargs)
#         ax[0].set_title("Mean Absolute Error")
#         ax[0].set_ylabel("MAE")
#         ax[1].bar(labels_m, rmse_vals, color=colors, **bar_kwargs)
#         ax[1].set_title("Root Mean Squared Error")
#         ax[1].set_ylabel("RMSE")
#         ax[2].bar(labels_m, r2_vals, color=colors, **bar_kwargs)
#         ax[2].set_title("R² Score")
#         ax[2].set_ylabel("R²")
#         for a in ax:
#             a.set_xticks(range(len(labels_m)))
#             a.set_xticklabels(labels_m, rotation=30, ha="right")
#             a.grid(True, axis="y", linestyle="--", alpha=0.6)
#         plt.suptitle("Model Performance Metrics", fontsize=14)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.savefig(Path(directory, "forecasts", forecast_name, "metrics_summary.png"))
#         plt.close(fig)
#
#     fig, ax = plt.subplots(figsize=(10, 6))
#     for i, (preds, targets) in enumerate(pred_target_pairs):
#         preds = np.array(preds)
#         targets = np.array(targets)
#         label = labels[i] if labels else f"Model {i+1}"
#         if preds.ndim == 1:
#             preds = preds.reshape(-1, 1)
#         if targets.ndim == 1:
#             targets = targets.reshape(-1, 1)
#         if preds.shape != targets.shape:
#             continue
#         horizon = preds.shape[1]
#         rmse_per_step = []
#         for t in range(horizon):
#             mask = np.isfinite(preds[:, t]) & np.isfinite(targets[:, t])
#             if mask.any():
#                 rmse = np.sqrt(mean_squared_error(targets[mask, t], preds[mask, t]))
#             else:
#                 rmse = np.nan
#             rmse_per_step.append(rmse)
#         ax.plot(range(1, horizon + 1), rmse_per_step, marker='o', label=label, color=colors[i])
#
#     ax.set_title("RMSE vs Forecast Horizon")
#     ax.set_xlabel("Forecast Step (T+)")
#     ax.set_ylabel("RMSE")
#     ax.legend()
#     ax.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(Path(directory, "forecasts", forecast_name, "horizon_rmse.png"))
#     plt.close(fig)
#
#     n_sets = len(pred_target_pairs)
#     if labels is None:
#         labels = [f"Set {i+1}" for i in range(n_sets)]
#     elif len(labels) != n_sets:
#         raise ValueError("Length of labels must match number of result sets.")
#
#     # Prepare combined error data
#     combined_errors = []
#     for (pred, target) in pred_target_pairs:
#         errors = (pred - target).flatten()
#         combined_errors.append(errors)
#
#     # Combined DataFrame for all errors
#     df_combined = pd.DataFrame({label: data for label, data in zip(labels, combined_errors)})
#     df_long_combined = df_combined.melt(var_name="Dataset", value_name="Error")
#
#     # Combined figure: emphasize points, de-emphasize boxplot
#     plt.figure(figsize=(8, 6))
#     ax = plt.gca()
#     sns.boxplot(x="Dataset", y="Error", data=df_long_combined,
#                 showcaps=True, boxprops={'facecolor': 'lightgray', 'alpha': 0.3, 'linewidth': 0.5},
#                 whiskerprops={'linewidth': 0.5}, medianprops={'color': 'blue', 'linewidth': 1}, ax=ax)
#
#     sns.stripplot(x="Dataset", y="Error", data=df_long_combined,
#                   jitter=True, size=6, color='black', alpha=0.8, ax=ax)
#
#     ax.set_title("Prediction Error Distribution")
#     ax.set_ylabel("Error (Absolute)")
#     ax.set_xlabel("Model")
#     plt.tight_layout()
#     plt.savefig(Path(directory, "forecasts", forecast_name, "boxplot.png"))
#     plt.close()
#
#     # Individual figures per pair comparing columns
#     for (pred, target), label in zip(pred_target_pairs, labels):
#         errors_matrix = pred - target
#
#         # Skip if 1D or single column
#         if errors_matrix.ndim == 1 or errors_matrix.shape[1] == 1:
#             continue
#
#         n_cols = errors_matrix.shape[1]
#         col_labels = [f"Col {i + 1}" for i in range(n_cols)]
#
#         # Prepare DataFrame for this pair
#         df_pair = pd.DataFrame({col_label: errors_matrix[:, i] for i, col_label in enumerate(col_labels)})
#         df_long_pair = df_pair.melt(var_name="Column", value_name="Error")
#
#         # Overlay boxplot and jitterplot on one axis
#         plt.figure(figsize=(8, 6))
#         ax = plt.gca()
#         sns.boxplot(x="Column", y="Error", data=df_long_pair,
#                     showcaps=True, boxprops={'facecolor': 'lightgray', 'alpha': 0.3, 'linewidth': 0.5},
#                     whiskerprops={'linewidth': 0.5}, ax=ax)
#
#         sns.stripplot(x="Column", y="Error", data=df_long_pair,
#                       jitter=True, size=6, color='black', alpha=0.8, ax=ax)
#
#         ax.set_title(f"Error by Column for {label}")
#         ax.set_ylabel("Error")
#         ax.set_xlabel("Column")
#         plt.tight_layout()
#         plt.savefig(Path(directory, "forecasts", forecast_name, f"{label.replace(' ', '_')}_overlay_emphasized.png"))
#         plt.close()
#
# def classification_visualizer(*pred_target_pairs, labels=None, directory='.', forecast_name='Classifier',
#                               num_samples=200):
#     os.makedirs(os.path.join(directory, "forecasts", forecast_name, "classification"), exist_ok=True)
#     sns.set_style("whitegrid")
#     metrics = []
#     all_conf_matrices = []
#     roc_data = []
#     pr_data = []
#     auc_scores = []
#
#     for i, (preds, targets) in enumerate(pred_target_pairs):
#         preds = np.array(preds).reshape(-1)[:num_samples]
#         targets = np.array(targets).reshape(-1)[:num_samples]
#         label = labels[i] if labels else f"Model {i+1}"
#         mask = np.isfinite(preds) & np.isfinite(targets)
#         preds, targets = preds[mask], targets[mask]
#
#         acc = accuracy_score(targets, preds)
#         prec = precision_score(targets, preds, zero_division=0)
#         rec = recall_score(targets, preds, zero_division=0)
#         f1 = f1_score(targets, preds, zero_division=0)
#         metrics.append((label, acc, prec, rec, f1))
#
#         cm = confusion_matrix(targets, preds)
#         all_conf_matrices.append((label, cm))
#
#         fpr, tpr, _ = roc_curve(targets, preds)
#         roc_auc = auc(fpr, tpr)
#         roc_data.append((label, fpr, tpr, roc_auc))
#         auc_scores.append((label, roc_auc))
#
#         precision, recall, _ = precision_recall_curve(targets, preds)
#         pr_auc = auc(recall, precision)
#         pr_data.append((label, recall, precision, pr_auc))
#
#     # Combined Confusion Matrix Plot
#     fig, axes = plt.subplots(1, len(all_conf_matrices), figsize=(5 * len(all_conf_matrices), 4))
#     if len(all_conf_matrices) == 1:
#         axes = [axes]
#     for ax, (label, cm) in zip(axes, all_conf_matrices):
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#         ax.set_title(f"{label}")
#         ax.set_xlabel("Predicted")
#         ax.set_ylabel("Actual")
#     plt.suptitle("Confusion Matrices")
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.savefig(Path(directory, "forecasts", forecast_name, "classification", f"confusion.png"))
#     plt.close()
#
#     # Combined ROC Curve Plot
#     plt.figure(figsize=(6, 5))
#     for label, fpr, tpr, roc_auc in roc_data:
#         plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.title("ROC Curves")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(Path(directory, "forecasts", forecast_name, "classification", f"roc.png"))
#     plt.close()
#
#     # Combined Precision-Recall Curve Plot
#     plt.figure(figsize=(6, 5))
#     for label, recall, precision, pr_auc in pr_data:
#         plt.plot(recall, precision, label=f"{label} (AUC = {pr_auc:.2f})")
#     plt.title("Precision-Recall Curves")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(Path(directory, "forecasts", forecast_name, "classification", "pr.png"))
#     plt.close()
#
#     # AUC Bar Plot
#     labels_auc = [x[0] for x in auc_scores]
#     auc_vals = [x[1] for x in auc_scores]
#     plt.figure(figsize=(8, 5))
#     sns.barplot(x=labels_auc, y=auc_vals, hue=auc_scores, legend=False)
#     plt.title("AUC Scores")
#     plt.ylabel("AUC")
#     plt.xticks(rotation=30, ha="right")
#     plt.grid(True, axis="y", linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(Path(directory, "forecasts", forecast_name, "classification", "auc_scores.png"))
#     plt.close()
#
# def apply_saved_normalize(df, param_file, min_val=0, max_val=1):
#     """
#     Normalize columns in df using saved min/max values from param_file.
#
#     Parameters:
#     - df: pandas DataFrame with columns to normalize
#     - param_file: path to JSON file with saved normalization parameters
#     - min_val, max_val: range used during original normalization
#
#     Returns:
#     - df_normalized: DataFrame with normalized values
#     """
#     df_normalized = df.copy()
#     with open(param_file, 'r') as f:
#         normalization_params = json.load(f)
#
#     for col in df.columns:
#         if col in normalization_params:
#             col_min = normalization_params[col]["min"]
#             col_max = normalization_params[col]["max"]
#             if col_max != col_min:
#                 df_normalized[col] = ((df[col] - col_min) / (col_max - col_min)) * (max_val - min_val) + min_val
#             else:
#                 df_normalized[col] = (min_val + max_val) / 2
#     return df_normalized
#
# def reverse_normalize(array, output_columns, param_file, min_val=0, max_val=1):
#     """
#     Reverse normalization on a NumPy array using saved parameters.
#
#     Parameters:
#     - array: NumPy array of shape (n_samples, n_outputs)
#     - output_columns: list of column names corresponding to array columns
#     - param_file: path to JSON file with saved normalization parameters
#     - min_val, max_val: range used during original normalization
#
#     Returns:
#     - array_restored: NumPy array with values in original scale
#     """
#     with open(param_file, 'r') as f:
#         normalization_params = json.load(f)
#
#     array_restored = np.copy(array)
#     for i, col in enumerate(output_columns):
#         if col in normalization_params:
#             col_min = normalization_params[col]["min"]
#             col_max = normalization_params[col]["max"]
#             if col_max != col_min:
#                 array_restored[:, i] = ((array[:, i] - min_val) / (max_val - min_val)) * (col_max - col_min) + col_min
#             else:
#                 array_restored[:, i] = col_min  # All values were the same originally
#     return array_restored


if __name__ == '__main__':
    ## Configure execution space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    matplotlib.use('Agg')  # Non-interactive backend for file output to handle remote machine installation errors

    ##################################################################################################################
    ## Load input, output and model hyperparameters from data_dir
    # data_dir = "../data/output/regression/Kimtall12hr"  # Parent directory of test/train sample folder
    data_dir = "../data/output/classification/Ecoli24hr"
    forecast_name = "nowcast"
    model_name = "xgbclassifier"

    with open(Path(data_dir, 'forecasts', forecast_name, model_name, 'model_config.json'), 'r') as f:
        config = json.load(f)
    # with open(Path(data_dir, 'forecasts', forecast_name, 'model_config.json'), 'r') as f:
    #     config = json.load(f)

    input_columns = config["input_columns"]
    output_columns = config["output_columns"]
    input_rows = slice(config["input_row_1"], config["input_row_2"])
    output_rows = config["output_rows"]

    ## Configure simple non-ML ("baseline") model calculation methods
    historic = "../data/output/regression/Consolidated.csv"  # Path to file with baseline model input
    gap_hours = 1  # Period before first forecast value from which input data is not used in baseline forecasts
    window_hours = 550  # Length of period for linear regression training (min. ~530 hrs for Eurofins params)
    diurnal_window = 1  # Number of hours before/after target time to include in diurnal average for seasonality model
    secondary, window_hours = load_secondary(output_columns, window_hours)  # Check output_columns and automatically adjust some baseline forecasts

    ## To retain normalized values:
    # raw_thresholds_df = pd.read_csv(Path('../data/input', "Limits.csv"), sep=';', decimal='.')
    # thresholds_df = apply_saved_normalize(raw_thresholds_df, param_file=Path('../data/input', "normalization.json"))
    ## If working in real (not normalized) values:
    thresholds_df = pd.read_csv(Path('../data/input', "Limits.csv"), sep=';', decimal='.')

    ################################################################################################################
    ## Prepare data for evaluation in various forecasts
    ## Run evaluation using samples excluded from training
    reloadset = Path(data_dir, "forecasts", forecast_name, "test_files.txt")
    with open(reloadset) as f:
        test_files = [line.strip() for line in f]
    test_samples = load_samples(os.path.join(data_dir,"samples"),input_columns=input_columns,output_columns=output_columns,
        input_rows=input_rows, output_rows=output_rows, file_list=test_files, fault_tolerant=True)
    test_dataset = TimeSeriesTargetDataset(test_samples)

    # Alternative: for full-coverage plotting of sparse data, evaluate forecasts on complete sample set (train + test)
    # samples = load_samples(os.path.join(data_dir, 'samples'), input_columns=input_columns,
    #                        output_columns=output_columns,
    #                        input_rows=input_rows, output_rows=output_rows, fault_tolerant=True)
    # test_dataset = TimeSeriesTargetDataset(samples)
    # test_samples = samples

    X_test = np.array([s[0].flatten() for s in test_samples])
    y_test = np.array([s[1].flatten()[0] for s in test_samples])

    ################################################################################################################
    ## Prepare transformer model for evaluation
    # transformer_model = TimeSeriesTransformer(config).to(device)
    # transformer_model.load_state_dict(torch.load(os.path.join(data_dir, "forecasts", forecast_name, "transformer",
    #                                               "transformer_model.pt"), map_location=device))
    # transformer_model.eval()  # Set to evaluation mode

    ## Prepare XGBRegresssor model for evaluation
    # xgbr_model = xgb.XGBRegressor()
    # xgbr_path = Path(data_dir, "forecasts", forecast_name, "XGBRegressor", "xgboost_model.json")
    # xgbr_model.load_model(xgbr_path)

    ##################################################################################################################
    # Evaluate regression forecasts
    # transformer_preds, transformer_targets = evaluate_transformer(transformer_model, test_dataset, device)
    naive_preds, naive_targets = evaluate_naive(test_dataset, historic, output_columns, data_dir,
                                                output_rows=output_rows, gap_hours=gap_hours)
    linear_preds, linear_targets = evaluate_linear(data_dir, forecast_name, test_dataset, historic, output_columns,
                                                   output_rows=output_rows, window_hours=window_hours,
                                                   gap_hours=gap_hours,
                                                   debug_plot=True, examples=10)

    seasonal_preds, seasonal_targets = evaluate_seasonal(test_dataset, historic, output_columns, data_dir, forecast_name,
                                                         output_rows=output_rows, diurnal_window=diurnal_window,
                                                         secondary=secondary)

    alternatives = [(naive_preds, naive_targets), (linear_preds, linear_targets), (seasonal_preds, seasonal_targets)]
    labels = ["Naive", "Linear", "Seasonal"]

    # alternatives = [(naive_preds, naive_targets), (linear_preds, linear_targets),
    #                 (seasonal_preds, seasonal_targets)]
    # labels = ["Naive", "Linear", "Seasonal"]

    reconstituted = []
    for preds, targets in alternatives:
        preds_original = reverse_normalize(preds, output_columns, Path('../data/input', "normalization.json"))
        targets_original = reverse_normalize(targets, output_columns, Path('../data/input', "normalization.json"))
        reconstituted.append((preds_original, targets_original))
    visualizer(*alternatives, labels=labels, forecast_name=forecast_name, directory=data_dir, num_samples=200)
    # visualizer((xgbr_pred, xgbr_target), labels=labels, forecast_name=forecast_name, directory=data_dir, num_samples=200)

    #################################################################################################################
    ## Convert regression model outputs to classes based on thresholds for each output column, and
    # evaluate success of regressors on classification problem

    class_results = []
    for preds, targets in reconstituted:
        bin_preds = binarize_predictions(preds, output_columns=output_columns, thresholds_df=thresholds_df)
        bin_targets = binarize_predictions(targets, output_columns=output_columns, thresholds_df=thresholds_df)
        class_results.append((bin_preds, bin_targets))

    # classification_visualizer(*class_results, labels=labels, directory=data_dir, forecast_name=forecast_name,
    #                           num_samples=200)

    ##################################################################################################################
    # Pure classification
    # Prepare XGBClassifier model for evaluation
    xgbc_model = xgb.XGBClassifier()
    xgbc_path = Path(data_dir, "forecasts", forecast_name, "xgbclassifier", "xgboost_model.json")
    xgbc_model.load_model(xgbc_path)

    xgbc_pred_flat = xgbc_model.predict(X_test)

    # Compute output_dim dynamically
    sample_df = pd.read_csv(Path(data_dir, 'samples', sorted(os.listdir(Path(data_dir, 'samples')))[0]))
    output_dim = len(output_columns) * len(sample_df.iloc[output_rows:])

    ## Reshape y_pred to [num_samples, output_dim]
    xgbc_pred = xgbc_pred_flat.reshape(-1, output_dim)
    xgbc_target = y_test.reshape(-1, output_dim)
    labels = labels + ['XGBClassifier']

    class_results = class_results + [(xgbc_pred, xgbc_target)]

    classification_visualizer(*class_results, labels=labels, directory=data_dir, forecast_name=forecast_name,
                              num_samples=200)
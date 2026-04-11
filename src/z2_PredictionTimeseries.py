"""Generate continuous prediction timeseries figures from saved sweep models.

For each target found in the feature sweep under the given dataset root, loads
the best-performing saved model (selected by the same criterion as
z1_FeaturePostProcess.py: highest min_skill_rmse among rows with r2 > 0), then
applies it at every timestep in Consolidated_sparse.csv by sliding a
``seq_len``-row input window across the full time series.

Two figures are produced, each with one subplot per target:

Figure 1 — Normalized residual predictions (``Target_timeseries_res_predictions.png``):
  - Continuous line: model predictions in normalized ``_res`` units
  - Scatter points: actual measured ``_res`` values

Figure 2 — Reconstructed original-unit timeseries (``Target_timeseries_reconstructed.png``):
  - Continuous line: cumulative chain reconstruction back to physical units
  - Scatter points: actual measured target values
  - Legal limit lines from Limits.csv

A CSV (``Consolidated_predictions.csv``) is also written containing all columns
from Consolidated_sparse.csv plus ``<target>_res_pred`` and
``<target>_reconstructed`` columns for each target.

A summary CSV (``best_models_used.csv``) lists which model was selected per target.

Usage examples::

    python src/z2_PredictionTimeseries.py --path data/output/CV14 --sweep-namespace feature_sweeps
    python src/z2_PredictionTimeseries.py --path data/output/CV14 --sweep-namespace Shapley_sweeps
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

import e_Train as train_module
import f_Evaluate as eval_module
from utils.config_utils import select_best_model_row
from utils.limits import load_limits_records, map_limits_to_columns
from fill_gaps_mlr import _compute_state_offset, DEFAULT_STATE_OFFSET
from h_RunMCFeatureSelectionSweep import (
    build_parser,
    discover_mc_dataset_plans,
    DatasetPlan,
    _resolve_dataset_inclusion,
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import torch
    import gpytorch
except ImportError:
    torch = None
    gpytorch = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_MODEL_IDS: set = {"naive", "seasonal", "linear"}
MIN_REQUIRED_VALID_INDEPENDENT: int = 5

_DEFAULT_CONSOLIDATED_RELPATH = Path("data") / "output" / "regression" / "Consolidated_sparse.csv"
_DEFAULT_LIMITS_RELPATH = Path("data") / "input" / "Limits.csv"

START_DATE = pd.Timestamp("2021-12-15")
END_DATE   = pd.Timestamp("2025-03-01")

# ---------------------------------------------------------------------------
# Figure helpers (adapted from plot_mlr_fill_timeseries.py)
# ---------------------------------------------------------------------------

def _strip_common_affixes(labels: list) -> list:
    if not labels:
        return labels
    if len(labels) == 1:
        return [labels[0].strip()]

    def _common_prefix(strings):
        pref = strings[0]
        for s in strings[1:]:
            i, max_i = 0, min(len(pref), len(s))
            while i < max_i and pref[i] == s[i]:
                i += 1
            pref = pref[:i]
            if not pref:
                break
        return pref

    def _common_suffix(strings):
        return _common_prefix([s[::-1] for s in strings])[::-1]

    prefix = _common_prefix(labels)
    suffix = _common_suffix(labels)
    if suffix.strip() == ")":
        suffix = ""
    cleaned = []
    for label in labels:
        start = len(prefix) if prefix else 0
        end = len(label) - len(suffix) if suffix else len(label)
        new_label = label[start:end].strip()
        cleaned.append(new_label if new_label else label.strip())
    return cleaned


def _safe_series_colors(n: int) -> list:
    base = [
        "#1f77b4", "#2ca02c", "#17becf", "#9467bd", "#7f7f7f",
        "#bcbd22", "#aec7e8", "#98df8a", "#c5b0d5", "#9edae5",
        "#8c564b", "#c7c7c7",
    ]
    if n <= len(base):
        return base[:n]
    repeats = (n // len(base)) + 1
    return (base * repeats)[:n]


def _limit_bounds(limit_spec) -> tuple:
    if not isinstance(limit_spec, dict):
        return None, None
    upper = limit_spec.get("upper")
    lower = limit_spec.get("lower")
    upper = float(upper) if upper is not None and pd.notna(upper) else None
    lower = float(lower) if lower is not None and pd.notna(lower) else None
    return upper, lower


# ---------------------------------------------------------------------------
# Row-filtering helpers (mirror z1_FeaturePostProcess.py)
# ---------------------------------------------------------------------------

def _filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "std_target" in out.columns:
        out = out[(out["std_target"].notnull()) & (out["std_target"] > 0)]
    if "r2" in out.columns:
        out = out[out["r2"].notnull()]
    if "r2" in out.columns:
        out = out[out["r2"].notnull() & np.isfinite(out["r2"].astype(float))]
    return out


def _filter_min_valid_independent(df: pd.DataFrame, min_required: int = MIN_REQUIRED_VALID_INDEPENDENT) -> pd.DataFrame:
    out = df.copy()
    count_col = None
    if "n_test_valid" in out.columns:
        count_col = "n_test_valid"
    elif "n_test_independent" in out.columns:
        count_col = "n_test_independent"
    if count_col is None:
        return out
    vals = pd.to_numeric(out[count_col], errors="coerce")
    return out[np.isfinite(vals) & (vals >= int(min_required))].copy()


def _exclude_baseline_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model" not in out.columns:
        return out
    return out[~out["model"].apply(lambda v: str(v).strip().lower() in BASELINE_MODEL_IDS)].copy()


def _normalize_model_key(value: str) -> str:
    return str(value).strip().lower().replace("_", "").replace(" ", "")


# ---------------------------------------------------------------------------
# Model directory discovery
# ---------------------------------------------------------------------------

def _find_model_dir(dataset_dir: Path, sweep_namespace: str, best_row: pd.Series) -> Path | None:
    """Find the forecast variant directory matching best_row.

    Scans for ``config_evaluate_*.yml`` files in the sweep directory, matching
    on subset_label suffix and model_type, identical to z1_FeaturePostProcess's
    ``_find_best_variant_eval_config``.
    """
    subset_label = str(best_row.get("subset_label", "")).strip().lower()
    model_key = _normalize_model_key(str(best_row.get("model", "")))

    _SHAPLEY_PREFIX = "shap_"
    if subset_label.startswith(_SHAPLEY_PREFIX):
        sweep_dir = dataset_dir / "forecasts" / "Shapley_sweeps"
        subset_label = subset_label[len(_SHAPLEY_PREFIX):]
    else:
        sweep_dir = dataset_dir / "forecasts" / sweep_namespace

    if not sweep_dir.exists():
        return None

    exact_match = None
    substring_match = None
    label_fallback = None

    for eval_cfg in sorted(sweep_dir.glob("*/config_evaluate_*.yml")):
        variant_dir = eval_cfg.parent
        if subset_label and not variant_dir.name.endswith(f"_{subset_label}"):
            continue
        try:
            cfg = train_module.load_config(str(eval_cfg))
        except Exception:
            continue
        cfg_keys = [
            _normalize_model_key(str(cfg.get("model_type", ""))),
            _normalize_model_key(str(cfg.get("model_name", ""))),
        ]
        if model_key and any(model_key == k for k in cfg_keys if k):
            exact_match = variant_dir
            break
        if model_key and not substring_match and any(model_key in k or k in model_key for k in cfg_keys if k):
            substring_match = variant_dir
        if label_fallback is None:
            label_fallback = variant_dir

    return exact_match or substring_match or label_fallback


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@dataclass
class _ModelBundle:
    """Container for a loaded model and its metadata."""
    model_type: str          # "mlr", "mlr_avg12", "mlr_avgall", "xgb_regressor", "gp_regressor", "transformer"
    input_columns: list      # feature column names (raw, before normalization)
    output_column: str       # _res column name being predicted
    seq_len: int             # input window length
    input_aggregation: str   # "none", "avg12", "avgall", or "last"
    norm_bounds: dict        # {col: {"min": float, "max": float}} from normalization.json
    # MLR-specific
    mlr_selected_features: list | None = None
    mlr_coefficients: np.ndarray | None = None
    mlr_intercept: float | None = None
    # Non-MLR model object
    model: object = None
    # GP-specific
    gp_bundle: dict | None = None
    # Device for torch models
    device: str = "cpu"


def _load_model_bundle(
    dataset_dir: Path,
    sweep_namespace: str,
    best_row: pd.Series,
) -> _ModelBundle | None:
    """Load best model and return an _ModelBundle ready for inference."""
    model_dir = _find_model_dir(dataset_dir, sweep_namespace, best_row)
    if model_dir is None:
        print(f"  [WARN] Could not find model directory for {dataset_dir.name}")
        return None

    model_config_path = model_dir / "model_config.json"
    if not model_config_path.exists():
        print(f"  [WARN] model_config.json not found: {model_config_path}")
        return None

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    norm_path = dataset_dir / "normalization.json"
    if not norm_path.exists():
        print(f"  [WARN] normalization.json not found: {norm_path}")
        return None
    with open(norm_path, "r", encoding="utf-8") as f:
        norm_bounds = json.load(f)

    input_columns = list(model_config.get("input_columns", []))
    output_columns = list(model_config.get("output_columns", []))
    if not output_columns:
        print(f"  [WARN] No output_columns in model_config: {model_config_path}")
        return None
    output_column = output_columns[0]

    input_row_1 = int(model_config.get("input_row_1", 0))
    input_row_2 = int(model_config.get("input_row_2", 167))
    seq_len = input_row_2 - input_row_1

    raw_agg = str(model_config.get("input_aggregation", "none")).lower()
    model_type_raw = str(best_row.get("model", "")).strip().lower()

    # MLR variants don't have an "input_aggregation" field; infer from model_type
    if model_type_raw in ("mlr", "mlr_avg12", "mlr_avgall"):
        if model_type_raw == "mlr_avg12":
            input_aggregation = "avg12"
        elif model_type_raw == "mlr_avgall":
            input_aggregation = "avgall"
        else:
            input_aggregation = "last"

        per_target = model_config.get("per_target_meta", [])
        if not per_target:
            print(f"  [WARN] No per_target_meta in MLR model_config: {model_config_path}")
            return None
        meta = per_target[0]
        selected_features = list(meta.get("selected_features", []))
        coefficients = np.array(meta.get("coefficients", []), dtype=float)
        intercept = float(meta.get("intercept", 0.0))

        return _ModelBundle(
            model_type=model_type_raw,
            input_columns=input_columns,
            output_column=output_column,
            seq_len=seq_len,
            input_aggregation=input_aggregation,
            norm_bounds=norm_bounds,
            mlr_selected_features=selected_features,
            mlr_coefficients=coefficients,
            mlr_intercept=intercept,
        )

    # Non-MLR models — determine aggregation
    if raw_agg in ("avg12", "avgall", "last"):
        input_aggregation = raw_agg
    elif raw_agg == "none":
        input_aggregation = "none"  # keep full window for transformer/GP
    else:
        input_aggregation = "none"

    # Load model
    device = "cpu"
    if torch is not None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = str(device_obj)

    if model_type_raw == "xgb_regressor":
        if xgb is None:
            print("  [WARN] xgboost not installed; skipping XGB model")
            return None
        xgb_path = model_dir / "xgboost_model.json"
        if not xgb_path.exists():
            print(f"  [WARN] xgboost_model.json not found: {xgb_path}")
            return None
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(str(xgb_path))
        return _ModelBundle(
            model_type=model_type_raw,
            input_columns=input_columns,
            output_column=output_column,
            seq_len=seq_len,
            input_aggregation=input_aggregation,
            norm_bounds=norm_bounds,
            model=xgb_model,
            device=device,
        )

    if model_type_raw == "gp_regressor":
        if torch is None or gpytorch is None:
            print("  [WARN] torch/gpytorch not installed; skipping GP model")
            return None
        # GP requires train_samples to reconstruct the model; find eval config and load them
        eval_cfg_path = next(model_dir.glob("config_evaluate_*.yml"), None)
        if eval_cfg_path is None:
            print(f"  [WARN] No eval config found in {model_dir}")
            return None
        try:
            cfg = eval_module.load_config(str(eval_cfg_path))
            config_dir = cfg["__config_dir"]
            data_cfg = cfg["data"]
            split_cfg = cfg.get("data_split", {"random_state": 42})
            data_cfg["data_dir"], data_cfg["sample_subdir"] = eval_module._resolve_data_paths(data_cfg, config_dir)
            split_source_dir = str(Path(data_cfg["data_dir"]) / "forecasts" / data_cfg["forecast_name"])
            train_samples = eval_module.load_split_samples(
                data_cfg["data_dir"],
                data_cfg["sample_subdir"],
                data_cfg["forecast_name"],
                input_columns,
                output_columns,
                slice(input_row_1, input_row_2),
                model_config.get("output_rows", [input_row_2 - 1]),
                split_file="train_files.txt",
                split_source_dir=split_source_dir,
                fault_tolerant=True,
                input_aggregation=raw_agg,
            )
            gp_bundle = eval_module._load_gp_bundle(data_cfg, split_cfg, "", train_samples, device, config_dir)
        except Exception as exc:
            print(f"  [WARN] GP bundle load failed for {dataset_dir.name}: {exc}")
            return None
        return _ModelBundle(
            model_type=model_type_raw,
            input_columns=input_columns,
            output_column=output_column,
            seq_len=seq_len,
            input_aggregation=input_aggregation,
            norm_bounds=norm_bounds,
            gp_bundle=gp_bundle,
            device=device,
        )

    if model_type_raw == "transformer":
        if torch is None:
            print("  [WARN] torch not installed; skipping Transformer model")
            return None
        from utils.transformer import TimeSeriesTransformer
        transformer_path = model_dir / "transformer_model.pt"
        if not transformer_path.exists():
            print(f"  [WARN] transformer_model.pt not found: {transformer_path}")
            return None
        try:
            device_obj = torch.device(device)
            model = TimeSeriesTransformer(model_config).to(device_obj)
            model.load_state_dict(torch.load(str(transformer_path), map_location=device_obj))
            model.eval()
        except Exception as exc:
            print(f"  [WARN] Transformer load failed for {dataset_dir.name}: {exc}")
            return None
        return _ModelBundle(
            model_type=model_type_raw,
            input_columns=input_columns,
            output_column=output_column,
            seq_len=seq_len,
            input_aggregation=input_aggregation,
            norm_bounds=norm_bounds,
            model=model,
            device=device,
        )

    print(f"  [WARN] Unknown model_type '{model_type_raw}' for {dataset_dir.name}")
    return None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_value(val: float, col: str, norm_bounds: dict) -> float:
    bounds = norm_bounds.get(col)
    if bounds is None:
        return val
    lo, hi = float(bounds.get("min", 0.0)), float(bounds.get("max", 1.0))
    if hi <= lo:
        return 0.0
    return (val - lo) / (hi - lo)


def _denormalize_value(val: float, col: str, norm_bounds: dict) -> float:
    bounds = norm_bounds.get(col)
    if bounds is None:
        return val
    lo, hi = float(bounds.get("min", 0.0)), float(bounds.get("max", 1.0))
    return val * (hi - lo) + lo


def _normalize_matrix(arr: np.ndarray, columns: list, norm_bounds: dict) -> np.ndarray:
    """Normalize each column of arr in-place using min/max bounds."""
    out = arr.copy().astype(float)
    for j, col in enumerate(columns):
        bounds = norm_bounds.get(col)
        if bounds is None:
            continue
        lo, hi = float(bounds.get("min", 0.0)), float(bounds.get("max", 1.0))
        if hi <= lo:
            out[:, j] = 0.0
        else:
            out[:, j] = (out[:, j] - lo) / (hi - lo)
    return out


# ---------------------------------------------------------------------------
# Sliding-window aggregation
# ---------------------------------------------------------------------------

def _aggregate_window(window: np.ndarray, mode: str) -> np.ndarray:
    """Aggregate a (seq_len, n_features) window to a 1-D row vector."""
    if mode == "avg12":
        chunk = window[-12:] if len(window) >= 12 else window
        with np.errstate(all="ignore"):
            return np.nanmean(chunk, axis=0)
    if mode == "avgall":
        with np.errstate(all="ignore"):
            return np.nanmean(window, axis=0)
    # "last" or default
    return window[-1].copy()


# ---------------------------------------------------------------------------
# Inference: build predictions for all rows
# ---------------------------------------------------------------------------

def _predict_all_rows(bundle: _ModelBundle, df_raw: pd.DataFrame) -> np.ndarray:
    """Run the model at every row of df_raw; return 1-D float array of predictions.

    df_raw must contain bundle.input_columns (raw, un-normalized values).
    Predictions are in normalized ``_res`` units.

    - MLR: apply coefficients directly; NaN if any selected feature is NaN.
    - XGB: build full aggregated matrix; XGBoost handles NaN natively.
    - GP: batch through _predict_gp_bundle; NaN inputs → NaN predictions.
    - Transformer: batch through model; NaN inputs → NaN predictions.
    """
    n = len(df_raw)
    predictions = np.full(n, np.nan, dtype=float)

    # Ensure all input columns exist (fill missing with NaN)
    missing_cols = [c for c in bundle.input_columns if c not in df_raw.columns]
    if missing_cols:
        warnings.warn(f"Missing input columns {missing_cols}; will be treated as NaN.", RuntimeWarning)
    df_feat = df_raw.reindex(columns=bundle.input_columns).astype(float)

    seq_len = max(1, bundle.seq_len)
    agg_mode = bundle.input_aggregation

    # -----------------------------------------------------------------------
    # MLR: vectorized apply — build aggregated matrix first, then predict
    # -----------------------------------------------------------------------
    if bundle.model_type in ("mlr", "mlr_avg12", "mlr_avgall"):
        selected = bundle.mlr_selected_features
        coeff = bundle.mlr_coefficients
        intercept = bundle.mlr_intercept
        if not selected or coeff is None:
            return predictions

        # Build agg_rows: (n, len(selected))
        feat_arr = df_feat.values  # (n, n_input_cols)
        col_idx = {c: j for j, c in enumerate(bundle.input_columns)}
        sel_idx = [col_idx[c] for c in selected if c in col_idx]
        if not sel_idx:
            return predictions

        # Normalize the input feature array using saved bounds
        feat_arr_norm = _normalize_matrix(feat_arr, bundle.input_columns, bundle.norm_bounds)

        agg_matrix = np.full((n, len(sel_idx)), np.nan, dtype=float)
        for i in range(n):
            start = max(0, i - seq_len + 1)
            window = feat_arr_norm[start: i + 1]  # (w, n_input_cols)
            row_agg = _aggregate_window(window, agg_mode)
            agg_matrix[i] = row_agg[sel_idx]

        # Predict where all selected features are finite
        finite_mask = np.all(np.isfinite(agg_matrix), axis=1)
        if finite_mask.any():
            predictions[finite_mask] = intercept + agg_matrix[finite_mask] @ coeff
        return predictions

    # -----------------------------------------------------------------------
    # XGBoost: vectorized across all rows
    # -----------------------------------------------------------------------
    if bundle.model_type == "xgb_regressor":
        feat_arr = df_feat.values.astype(float)
        feat_arr_norm = _normalize_matrix(feat_arr, bundle.input_columns, bundle.norm_bounds)
        n_feat = feat_arr_norm.shape[1]

        if agg_mode == "none":
            # XGB was trained on flattened full windows: shape (n_samples, seq_len * n_features)
            flat_len = seq_len * n_feat
            X_flat = np.full((n, flat_len), np.nan, dtype=float)
            for i in range(n):
                start = max(0, i - seq_len + 1)
                window = feat_arr_norm[start: i + 1]
                if len(window) < seq_len:
                    pad = np.full((seq_len - len(window), n_feat), np.nan, dtype=float)
                    window = np.vstack([pad, window])
                X_flat[i] = window.flatten()
            agg_matrix = X_flat
        else:
            agg_matrix = np.full((n, n_feat), np.nan, dtype=float)
            for i in range(n):
                start = max(0, i - seq_len + 1)
                window = feat_arr_norm[start: i + 1]
                agg_matrix[i] = _aggregate_window(window, agg_mode)

        try:
            predictions = bundle.model.predict(agg_matrix).astype(float)
        except Exception as exc:
            warnings.warn(f"XGB predict failed: {exc}", RuntimeWarning)
        return predictions

    # -----------------------------------------------------------------------
    # GP: batch predict
    # -----------------------------------------------------------------------
    if bundle.model_type == "gp_regressor" and bundle.gp_bundle is not None:
        feat_arr = df_feat.values.astype(float)
        feat_arr_norm = _normalize_matrix(feat_arr, bundle.input_columns, bundle.norm_bounds)
        n_feat = feat_arr_norm.shape[1]

        if agg_mode == "none":
            # GP was trained on flattened full windows: shape (n_samples, seq_len * n_features)
            flat_len = seq_len * n_feat
            X_flat = np.full((n, flat_len), np.nan, dtype=float)
            for i in range(n):
                start = max(0, i - seq_len + 1)
                window = feat_arr_norm[start: i + 1]  # (w, n_feat)
                # Pad to seq_len on the left with the first available row
                if len(window) < seq_len:
                    pad = np.full((seq_len - len(window), n_feat), np.nan, dtype=float)
                    window = np.vstack([pad, window])
                X_flat[i] = window.flatten()
            agg_matrix = X_flat
        else:
            agg_matrix = np.full((n, n_feat), np.nan, dtype=float)
            for i in range(n):
                start = max(0, i - seq_len + 1)
                window = feat_arr_norm[start: i + 1]
                agg_matrix[i] = _aggregate_window(window, agg_mode)

        device_obj = torch.device(bundle.device) if torch is not None else None
        # Batch GP inference to avoid GPU OOM on large timeseries.
        # Start with batch_size=8 and fall back to CPU at batch_size=1 on OOM.
        gp_batch_size = 8
        all_means = np.full(n, np.nan, dtype=float)
        cpu_device = torch.device("cpu") if torch is not None else None
        use_device = device_obj
        i_batch = 0
        while i_batch < n:
            batch = agg_matrix[i_batch: i_batch + gp_batch_size]
            try:
                result = eval_module._predict_gp_bundle(bundle.gp_bundle, batch, use_device)
                all_means[i_batch: i_batch + len(batch)] = result["mean"][:, 0].astype(float)
                i_batch += len(batch)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and use_device != cpu_device:
                    warnings.warn(
                        f"GP GPU OOM (batch={gp_batch_size}); switching to CPU for remaining rows.",
                        RuntimeWarning,
                    )
                    if torch is not None:
                        torch.cuda.empty_cache()
                    use_device = cpu_device
                    # Move GP bundle models to CPU
                    for entry in bundle.gp_bundle["models"]:
                        entry["model"] = entry["model"].cpu()
                        entry["likelihood"] = entry["likelihood"].cpu()
                    bundle.gp_bundle["models"] = bundle.gp_bundle["models"]
                    gp_batch_size = 4
                else:
                    warnings.warn(f"GP predict failed at row {i_batch}: {exc}", RuntimeWarning)
                    i_batch += gp_batch_size  # skip this batch
        predictions = all_means
        return predictions

    # -----------------------------------------------------------------------
    # Transformer: batch predict
    # -----------------------------------------------------------------------
    if bundle.model_type == "transformer" and bundle.model is not None:
        if torch is None:
            return predictions

        feat_arr = df_feat.values.astype(float)
        feat_arr_norm = _normalize_matrix(feat_arr, bundle.input_columns, bundle.norm_bounds)

        n_feat = feat_arr_norm.shape[1]
        # Rows where all input features are NaN — cannot predict
        all_nan_rows = np.all(~np.isfinite(feat_arr), axis=1)

        # Batch transformer inference to avoid GPU OOM
        t_batch_size = 512
        device_obj = torch.device(bundle.device)
        cpu_device = torch.device("cpu")
        use_device = device_obj
        i_batch = 0
        while i_batch < n:
            batch_end = min(i_batch + t_batch_size, n)
            batch_size = batch_end - i_batch

            X_3d = np.full((batch_size, seq_len, n_feat), np.nan, dtype=float)
            for bi, i in enumerate(range(i_batch, batch_end)):
                start = max(0, i - seq_len + 1)
                window = feat_arr_norm[start: i + 1]
                pad_len = seq_len - len(window)
                if pad_len > 0:
                    X_3d[bi, pad_len:] = window
                else:
                    X_3d[bi] = window

            # Replace NaN with 0 for transformer
            X_tensor = torch.tensor(
                np.nan_to_num(X_3d, nan=0.0), dtype=torch.float32, device=use_device
            )
            try:
                with torch.no_grad():
                    out = bundle.model(X_tensor)
                preds_np = out.detach().cpu().numpy()
                if preds_np.ndim == 2:
                    predictions[i_batch:batch_end] = preds_np[:, 0].astype(float)
                elif preds_np.ndim == 1:
                    predictions[i_batch:batch_end] = preds_np.astype(float)
                i_batch = batch_end
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and use_device != cpu_device:
                    warnings.warn(
                        f"Transformer GPU OOM (batch={t_batch_size}); switching to CPU.",
                        RuntimeWarning,
                    )
                    if torch is not None:
                        torch.cuda.empty_cache()
                    bundle.model = bundle.model.cpu()
                    use_device = cpu_device
                    t_batch_size = max(64, t_batch_size // 4)
                else:
                    warnings.warn(f"Transformer predict failed at row {i_batch}: {exc}", RuntimeWarning)
                    i_batch = batch_end  # skip this batch

        # Mark all-NaN rows as NaN
        predictions[all_nan_rows] = np.nan
        return predictions

    return predictions


# ---------------------------------------------------------------------------
# Cumulative chain reconstruction
# ---------------------------------------------------------------------------

def _estimate_base_persistence(
    lookback: int,
    current_row: int,
    observed: np.ndarray,
    obs_ilocs: np.ndarray,
) -> float:
    """Estimate the target value at index ``lookback`` using persistence.

    Strategy (in priority order):
    1. Direct: if observed[lookback] is a real measurement, return it.
    2. Persistence: use the last measurement before ``lookback``.
    3. Return NaN if no prior measurement exists.

    Parameters
    ----------
    lookback : target index to estimate (= i - state_offset).
    current_row : the row being reconstructed (= i); unused here but kept for
        interface consistency with alternative estimators.
    observed : full 1-D array of actual measurements (NaN where absent).
    obs_ilocs : sorted array of integer indices where observed is non-NaN.
    """
    # 1. Direct
    if np.isfinite(observed[lookback]):
        return float(observed[lookback])

    if len(obs_ilocs) == 0:
        return np.nan

    # 2. Persistence: last measurement strictly before lookback
    pos = np.searchsorted(obs_ilocs, lookback)  # first obs >= lookback
    if pos > 0:
        return float(observed[int(obs_ilocs[pos - 1])])

    return np.nan


def _reconstruct_original_units(
    pred_res_norm: np.ndarray,
    observed: np.ndarray,
    output_col: str,
    norm_bounds: dict,
    state_offset: int,
    base_estimator=_estimate_base_persistence,
) -> np.ndarray:
    """Reconstruct absolute target values from normalized residual predictions.

    At each row i with a finite prediction:
      1. Denormalize pred_res_norm[i] → raw residual Δ.
      2. Estimate the target value at row ``i - state_offset`` using
         ``base_estimator`` (default: linear interpolation between measurements,
         with constant fallback).
      3. reconstructed[i] = base + Δ.

    Each row is computed independently (no chaining of reconstructed values).
    Discontinuities at actual measurement rows are expected.

    Parameters
    ----------
    pred_res_norm : 1-D array of model predictions in normalized _res units.
    observed : 1-D array of actual measured target values (NaN where absent).
    output_col : ``<target>_res`` column name (used for denormalization).
    norm_bounds : normalization.json dict.
    state_offset : lookback in rows (median inter-observation gap).
    base_estimator : callable(lookback, current_row, observed, obs_ilocs) → float.
        Swap this to change the base-estimation strategy.
    """
    n = len(pred_res_norm)
    reconstructed = np.full(n, np.nan, dtype=float)

    # Pre-compute sorted observation indices once
    obs_ilocs = np.where(np.isfinite(observed))[0]

    for i in range(n):
        pred_norm = pred_res_norm[i]
        if not np.isfinite(pred_norm):
            continue

        pred_raw = _denormalize_value(float(pred_norm), output_col, norm_bounds)
        if not np.isfinite(pred_raw):
            continue

        lookback = i - state_offset
        if lookback < 0:
            continue

        base = base_estimator(lookback, i, observed, obs_ilocs)
        if not np.isfinite(base):
            continue

        reconstructed[i] = base + pred_raw

    return reconstructed


def _reconstruct_chained(
    pred_res_norm: np.ndarray,
    observed: np.ndarray,
    output_col: str,
    norm_bounds: dict,
    state_offset: int,
) -> np.ndarray:
    """Chained reconstruction: accumulated predicted residuals, reset at measurements.

    Like ``_reconstruct_original_units`` but the base at ``i - state_offset`` is
    taken from the previously reconstructed array when no real measurement exists
    there, allowing the series to drift freely between measurements.

    Base priority at lookback:
      1. Real measurement at lookback → resets chain to ground truth.
      2. Previously reconstructed value at lookback → chain accumulates.
      3. Persistence (last measurement before lookback) → bootstrap before chain starts.
      4. NaN → skip.
    """
    n = len(pred_res_norm)
    reconstructed = np.full(n, np.nan, dtype=float)
    obs_ilocs = np.where(np.isfinite(observed))[0]

    for i in range(n):
        pred_norm = pred_res_norm[i]
        if not np.isfinite(pred_norm):
            continue

        pred_raw = _denormalize_value(float(pred_norm), output_col, norm_bounds)
        if not np.isfinite(pred_raw):
            continue

        lookback = i - state_offset
        if lookback < 0:
            continue

        # 1. Real measurement at lookback
        if np.isfinite(observed[lookback]):
            base = float(observed[lookback])
        # 2. Previously chained value at lookback
        elif np.isfinite(reconstructed[lookback]):
            base = float(reconstructed[lookback])
        # 3. Persistence bootstrap
        else:
            pos = np.searchsorted(obs_ilocs, lookback)
            if pos > 0:
                base = float(observed[int(obs_ilocs[pos - 1])])
            else:
                continue

        reconstructed[i] = base + pred_raw

    return reconstructed


# ---------------------------------------------------------------------------
# Main figure builders
# ---------------------------------------------------------------------------

def _build_timeseries_figure(
    target_cols: list,
    wrapped_labels: list,
    x_index: pd.DatetimeIndex,
    series_line: dict,        # {col: pd.Series}  — continuous line values
    series_scatter: dict,     # {col: pd.Series}  — sparse scatter values
    line_label: str,
    scatter_label: str,
    limit_specs: dict,
    title: str,
    fig_width: float = 13.0,
    row_height: float = 0.88,
    font_size: int = 10,
) -> plt.Figure:
    n_rows = len(target_cols)
    fig_h = max(2.8, row_height * n_rows)
    palette = _safe_series_colors(n_rows)

    y_label_font_size = int(round(font_size * 1.5))
    x_label_font_size = int(round(font_size * 1.5))
    y_value_font_size = font_size
    y_label_pad = 8
    shared_hspace = 0.08
    top_margin_in = 0.10
    bottom_margin_in = 0.90

    global_max_label_line_len = max(
        (max(len(line) for line in lbl.splitlines()) if lbl else 0)
        for lbl in wrapped_labels
    ) if wrapped_labels else 10
    est_char_width_in = max(0.045, 0.0055 * y_label_font_size)
    left_margin_in = global_max_label_line_len * est_char_width_in + 0.55
    shared_left_margin = max(0.14, min(0.26, left_margin_in / fig_width))
    shared_right_margin = 0.995

    fig, axes_raw = plt.subplots(
        n_rows, 1, sharex=True,
        figsize=(fig_width, fig_h),
        gridspec_kw={"hspace": shared_hspace},
    )
    axes = [axes_raw] if n_rows == 1 else list(axes_raw)

    quarter_ticks = pd.date_range(start=START_DATE, end=END_DATE, freq="QS-JAN")

    for i, (ax, col, label) in enumerate(zip(axes, target_cols, wrapped_labels)):
        color = palette[i]

        line_ser = series_line.get(col)
        scatter_ser = series_scatter.get(col)

        obs_mask = scatter_ser.notna() if scatter_ser is not None else None

        # Limit lines
        if col in limit_specs:
            upper_limit, lower_limit = _limit_bounds(limit_specs[col])
            if upper_limit is not None:
                ax.axhline(y=upper_limit, color="#ff0000", linewidth=0.7, linestyle="-", zorder=1.2)
            if lower_limit is not None:
                ax.axhline(y=lower_limit, color="#2ca02c", linewidth=0.7, linestyle="-", zorder=1.2)

        # Break the model line at each measured point after the first so
        # prediction gaps remain visually explicit.
        if line_ser is not None and line_ser.notna().any():
            line_vals = line_ser.to_numpy(dtype=float)
            segment_mask = np.isfinite(line_vals)
            if obs_mask is not None:
                obs_ilocs = np.flatnonzero(obs_mask.to_numpy(dtype=bool))
                if obs_ilocs.size > 1:
                    segment_mask[obs_ilocs[1:]] = False

            seg_start = None
            for row_idx in range(len(segment_mask)):
                if bool(segment_mask[row_idx]):
                    if seg_start is None:
                        seg_start = row_idx
                    continue
                if seg_start is None:
                    continue
                seg_slice = slice(seg_start, row_idx)
                ax.plot(
                    line_ser.index[seg_slice], line_vals[seg_slice],
                    linestyle="-", linewidth=0.7,
                    color=color, alpha=0.65,
                    zorder=1.5, label=line_label if seg_start == 0 else None,
                )
                seg_start = None
            if seg_start is not None:
                seg_slice = slice(seg_start, len(segment_mask))
                ax.plot(
                    line_ser.index[seg_slice], line_vals[seg_slice],
                    linestyle="-", linewidth=0.7,
                    color=color, alpha=0.65,
                    zorder=1.5, label=line_label if seg_start == 0 else None,
                )

        # Scatter
        if scatter_ser is not None:
            if obs_mask.any():
                ax.plot(
                    scatter_ser.index[obs_mask], scatter_ser.values[obs_mask],
                    linestyle="", marker="o", markersize=4.5,
                    color=color,
                    markeredgecolor="black", markeredgewidth=0.5,
                    alpha=0.95, zorder=2.5, label=scatter_label,
                )

        # Scale to the measured values only; fall back to the line if there are
        # no measurements for this subplot.
        y_low = y_high = None
        if scatter_ser is not None:
            scatter_vals = scatter_ser.to_numpy(dtype=float)
            finite_scatter = scatter_vals[np.isfinite(scatter_vals)]
            if finite_scatter.size > 0:
                y_low = float(np.min(finite_scatter))
                y_high = float(np.max(finite_scatter))
        if y_low is None and line_ser is not None:
            line_vals = line_ser.to_numpy(dtype=float)
            finite_line = line_vals[np.isfinite(line_vals)]
            if finite_line.size > 0:
                y_low = float(np.min(finite_line))
                y_high = float(np.max(finite_line))

        if y_low is not None and y_high is not None:
            y_span = y_high - y_low
            if y_span <= 0:
                y_pad = 0.12 * max(abs(y_low), 1.0)
            else:
                y_pad = max(0.08 * y_span, 0.03 * max(abs(y_low), abs(y_high), 1.0))
            ax.set_ylim(y_low - y_pad, y_high + y_pad)

        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontsize=y_label_font_size, labelpad=y_label_pad)
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.4)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
        ax.tick_params(axis="y", labelsize=y_value_font_size)
        ax.tick_params(axis="x", labelsize=x_label_font_size)
        if i < n_rows - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    axes[-1].set_xticks(quarter_ticks)
    axes[-1].set_xticklabels(
        [dt.strftime("%Y-%m-%d") for dt in quarter_ticks],
        rotation=25, ha="right", fontsize=x_label_font_size,
    )
    axes[-1].set_xlim(START_DATE, END_DATE)

    shared_top_margin = max(0.86, min(0.995, 1.0 - (top_margin_in / fig_h)))
    shared_bottom_margin = max(0.08, min(0.30, bottom_margin_in / fig_h))
    fig.subplots_adjust(
        left=shared_left_margin, right=shared_right_margin,
        top=shared_top_margin,  bottom=shared_bottom_margin,
        hspace=shared_hspace,
    )

    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=0.7, linestyle="-",
                   alpha=0.65, label=line_label),
        plt.Line2D([0], [0], color="black", linestyle="", marker="o",
                   markersize=4.5, markeredgecolor="black", markeredgewidth=0.5,
                   label=scatter_label),
        plt.Line2D([0], [0], color="#ff0000", linewidth=0.7, linestyle="-",
                   label="Upper limit"),
        plt.Line2D([0], [0], color="#2ca02c", linewidth=0.7, linestyle="-",
                   label="Lower limit"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        framealpha=0.85,
        fontsize=font_size,
        borderaxespad=0.12,
    )

    return fig


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run(
    plans: list[DatasetPlan],
    consolidated_csv: Path,
    limits_csv: Path,
    out_dir: Path,
    sweep_namespace: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load Consolidated_sparse.csv
    # ------------------------------------------------------------------
    if not consolidated_csv.exists():
        raise FileNotFoundError(f"Consolidated_sparse.csv not found: {consolidated_csv}")

    print(f"Loading {consolidated_csv} ...")
    df_full = pd.read_csv(consolidated_csv, low_memory=False)
    df_full["TIMESTAMP"] = pd.to_datetime(df_full["TIMESTAMP"], errors="coerce")
    df_full = df_full[df_full["TIMESTAMP"].notna()].sort_values("TIMESTAMP").copy()
    df_full = df_full[
        (df_full["TIMESTAMP"] >= START_DATE) & (df_full["TIMESTAMP"] <= END_DATE)
    ].copy()
    df_full = df_full.set_index("TIMESTAMP")
    print(f"  {len(df_full)} rows after date filter.")

    # Limits
    limits_records = load_limits_records(limits_csv) if limits_csv.exists() else []

    # We'll accumulate per-target predictions into the output DataFrame
    df_out = df_full.copy()

    # Track which targets were processed for the figures and summary
    processed_targets = []   # list of (raw_target_col, res_col, bundle, res_pred_series, recon_series)
    summary_rows = []

    # ------------------------------------------------------------------
    # Process each dataset (= each target's MC_<Target>_res directory)
    # ------------------------------------------------------------------
    for plan in plans:
        dataset_name = plan.dataset_dir.name
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")

        # Locate feature sweep metrics CSV
        metrics_csv = plan.dataset_dir / "forecasts" / sweep_namespace / "feature_sweep_final_metrics.csv"
        if not metrics_csv.exists():
            print(f"  [SKIP] feature_sweep_final_metrics.csv not found: {metrics_csv}")
            continue

        df_metrics = pd.read_csv(metrics_csv)
        if df_metrics.empty:
            print(f"  [SKIP] Empty metrics CSV: {metrics_csv}")
            continue

        # Select best model row
        valid = _exclude_baseline_metric_rows(
            _filter_min_valid_independent(
                _filter_valid_rows(df_metrics),
                min_required=MIN_REQUIRED_VALID_INDEPENDENT,
            )
        )
        if valid.empty:
            print(f"  [SKIP] No valid model rows meeting quality thresholds for {dataset_name}")
            continue

        best_row = select_best_model_row(valid)
        best_model_type = str(best_row.get("model", "unknown"))
        best_r2 = float(best_row.get("r2", float("nan")))
        best_skill = float(best_row.get("min_skill_rmse", float("nan")))
        print(f"  Best model: {best_model_type}  r²={best_r2:.3f}  skill={best_skill:.3f}")

        # Load model bundle
        bundle = _load_model_bundle(plan.dataset_dir, sweep_namespace, best_row)
        if bundle is None:
            continue

        output_col = bundle.output_column  # e.g. "Arsenic (µg/L)_res"
        target_col = output_col[:-4] if output_col.endswith("_res") else output_col  # e.g. "Arsenic (µg/L)"

        # Infer state_offset for reconstruction
        if target_col in df_full.columns:
            state_offset = _compute_state_offset(df_full, target_col)
        else:
            state_offset = DEFAULT_STATE_OFFSET
        print(f"  Output column: {output_col}  |  state_offset (lookback rows): {state_offset}")

        # Generate predictions at all rows
        print(f"  Running sliding-window inference across {len(df_full)} rows ...")
        try:
            pred_norm = _predict_all_rows(bundle, df_full)
        except Exception as exc:
            print(f"  [WARN] Inference failed for {dataset_name}: {exc}")
            traceback.print_exc()
            continue

        pred_series = pd.Series(pred_norm, index=df_full.index, dtype=float)
        n_valid_preds = int(np.isfinite(pred_norm).sum())
        print(f"  Valid predictions: {n_valid_preds} / {len(pred_norm)}")

        # Observed target values (original units)
        observed_raw = (
            pd.to_numeric(df_full[target_col], errors="coerce")
            if target_col in df_full.columns
            else pd.Series(np.nan, index=df_full.index)
        )

        # Observed _res values (normalized)
        observed_res = (
            pd.to_numeric(df_full[output_col], errors="coerce")
            if output_col in df_full.columns
            else pd.Series(np.nan, index=df_full.index)
        )

        # Cumulative chain reconstruction
        recon = _reconstruct_original_units(
            pred_norm,
            observed_raw.values,
            output_col,
            bundle.norm_bounds,
            state_offset,
        )
        recon_series = pd.Series(recon, index=df_full.index, dtype=float)

        # Chained reconstruction variant
        recon_chained = _reconstruct_chained(
            pred_norm,
            observed_raw.values,
            output_col,
            bundle.norm_bounds,
            state_offset,
        )
        recon_chained_series = pd.Series(recon_chained, index=df_full.index, dtype=float)

        # Store in output DataFrame
        res_pred_col = f"{output_col}_pred"
        recon_col = f"{target_col}_reconstructed"
        recon_chained_col = f"{target_col}_reconstructed_chained"
        df_out[res_pred_col] = pred_series
        df_out[recon_col] = recon_series
        df_out[recon_chained_col] = recon_chained_series

        # Record for figures
        processed_targets.append((target_col, output_col, bundle, observed_res, pred_series, observed_raw, recon_series, recon_chained_series))

        summary_rows.append({
            "dataset": dataset_name,
            "target_col": target_col,
            "res_col": output_col,
            "model_type": best_model_type,
            "subset_label": str(best_row.get("subset_label", "")),
            "feature_tag": str(best_row.get("feature_tag", "")),
            "r2": best_r2,
            "min_skill_rmse": best_skill,
            "state_offset_rows": state_offset,
            "n_valid_predictions": n_valid_preds,
        })

    # ------------------------------------------------------------------
    # Write output CSV
    # ------------------------------------------------------------------
    predictions_csv = out_dir / "Consolidated_predictions.csv"
    df_out.reset_index().to_csv(predictions_csv, index=False)
    print(f"\nWrote predictions CSV: {predictions_csv}")

    # Write best-models summary
    if summary_rows:
        summary_csv = out_dir / "best_models_used.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"Wrote model summary: {summary_csv}")

    if not processed_targets:
        print("\nNo targets successfully processed — no figures generated.")
        return

    # ------------------------------------------------------------------
    # Build figures
    # ------------------------------------------------------------------
    # Collect label info
    target_cols_fig = [t[0] for t in processed_targets]
    res_cols_fig    = [t[1] for t in processed_targets]

    raw_labels = list(target_cols_fig)
    raw_labels_stripped = _strip_common_affixes(raw_labels)
    wrapped_labels = [textwrap.fill(lbl, width=17) for lbl in raw_labels_stripped]

    # Limit specs (keyed by raw target column)
    limit_specs = map_limits_to_columns(target_cols_fig, limits_records) if limits_records else {}

    # ---- Figure 1: normalized _res units (both predictions and observed _res normalized) ----
    series_line_res_remapped = {}
    series_scatter_res_remapped = {}
    for t in processed_targets:
        target_col, output_col, bundle, obs_res, pred_series, obs_raw, recon_series, recon_chained_series = t
        # Normalize the observed _res scatter to match the prediction scale
        bounds = bundle.norm_bounds.get(output_col)
        if bounds is not None:
            lo, hi = float(bounds["min"]), float(bounds["max"])
            span = hi - lo if hi > lo else 1.0
            obs_res_norm = obs_res.apply(
                lambda v: (float(v) - lo) / span if np.isfinite(v) else np.nan
            )
        else:
            obs_res_norm = obs_res
        series_line_res_remapped[target_col] = pred_series      # already normalized
        series_scatter_res_remapped[target_col] = obs_res_norm

    fig1 = _build_timeseries_figure(
        target_cols=target_cols_fig,
        wrapped_labels=wrapped_labels,
        x_index=df_full.index,
        series_line=series_line_res_remapped,
        series_scatter=series_scatter_res_remapped,
        line_label="Model prediction (normalized residual)",
        scatter_label="Measured residual (normalized)",
        limit_specs={},  # no limits on normalized _res figure
        title="Predicted vs. measured normalized residuals",
    )
    fig1_path = out_dir / "Target_timeseries_res_predictions.png"
    fig1.savefig(str(fig1_path), dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig1)
    print(f"Wrote figure 1: {fig1_path}")

    # ---- Figure 2: reconstructed original units (persistence base) ----
    series_line_recon    = {t[0]: t[6] for t in processed_targets}  # recon_series
    series_scatter_recon = {t[0]: t[5] for t in processed_targets}  # observed_raw

    fig2 = _build_timeseries_figure(
        target_cols=target_cols_fig,
        wrapped_labels=wrapped_labels,
        x_index=df_full.index,
        series_line=series_line_recon,
        series_scatter=series_scatter_recon,
        line_label="Reconstructed estimate (original units)",
        scatter_label="Measured value",
        limit_specs=limit_specs,
        title="Reconstructed parameter timeseries",
    )
    fig2_path = out_dir / "Target_timeseries_reconstructed.png"
    fig2.savefig(str(fig2_path), dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig2)
    print(f"Wrote figure 2: {fig2_path}")

    # ---- Figure 3: chained reconstruction (accumulated residuals, resets at measurements) ----
    series_line_chained    = {t[0]: t[7] for t in processed_targets}  # recon_chained_series
    series_scatter_chained = {t[0]: t[5] for t in processed_targets}  # observed_raw

    fig3 = _build_timeseries_figure(
        target_cols=target_cols_fig,
        wrapped_labels=wrapped_labels,
        x_index=df_full.index,
        series_line=series_line_chained,
        series_scatter=series_scatter_chained,
        line_label="Chained reconstruction (original units)",
        scatter_label="Measured value",
        limit_specs=limit_specs,
        title="Chained reconstruction timeseries",
    )
    fig3_path = out_dir / "Target_timeseries_reconstructed_chained.png"
    fig3.savefig(str(fig3_path), dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig3)
    print(f"Wrote figure 3: {fig3_path}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    parser.set_defaults(limit_datasets=0)

    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Optional alias for --data-root; dataset root directory to scan.",
    )
    parser.add_argument(
        "--sweep-namespace",
        type=str,
        default="feature_sweeps",
        help=(
            "Forecast sweep subdirectory to read model results from "
            "(e.g. 'feature_sweeps' or 'Shapley_sweeps')."
        ),
    )
    parser.add_argument(
        "--consolidated-csv",
        type=str,
        default=None,
        help=(
            "Path to Consolidated_sparse.csv. "
            "Defaults to <workspace_root>/data/output/regression/Consolidated_sparse.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for output files. "
            "Defaults to <data_root>/summaries/predictions/."
        ),
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        default=False,
        help="Process all dataset folders, ignoring --dataset-prefix and --limit-datasets.",
    )

    args = parser.parse_args()
    workspace_root = Path(__file__).resolve().parent.parent

    data_root_arg = args.path if args.path else args.data_root
    data_root = Path(data_root_arg)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()

    sweep_namespace = str(args.sweep_namespace).strip() or "feature_sweeps"

    consolidated_csv = (
        Path(args.consolidated_csv).resolve()
        if args.consolidated_csv
        else (workspace_root / _DEFAULT_CONSOLIDATED_RELPATH).resolve()
    )

    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (data_root / "summaries" / "predictions").resolve()
    )

    limits_csv = (workspace_root / _DEFAULT_LIMITS_RELPATH).resolve()

    include_regular, include_res = _resolve_dataset_inclusion(args)
    # Override: only process _res datasets (each represents one target)
    include_regular = False
    include_res = True

    dataset_prefix = "" if args.all_datasets else str(args.dataset_prefix)
    limit_datasets = 0 if args.all_datasets else int(args.limit_datasets)

    plans = discover_mc_dataset_plans(
        data_root=data_root,
        dataset_prefix=dataset_prefix,
        config_pattern=args.config_pattern,
        limit_datasets=limit_datasets,
        include_regular=include_regular,
        include_res=include_res,
    )

    if not plans:
        print(f"[WARN] No dataset plans found under {data_root}")
        return 1

    print(f"Found {len(plans)} dataset(s) under {data_root}")
    run(
        plans=plans,
        consolidated_csv=consolidated_csv,
        limits_csv=limits_csv,
        out_dir=out_dir,
        sweep_namespace=sweep_namespace,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

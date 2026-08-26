"""
Unified evaluation script for regression, transformer, and classification models.
Uses a YAML/JSON config file to control what gets evaluated.

Example terminal usage:
python src/f_Evaluate.py --config data/output/regression/MC_pH/forecasts/gp_01/config_evaluate_gp_01.yml
python src/f_Evaluate.py --config data/output/regression/MC_pH/forecasts/transformer_01/config_evaluate_transformer_01.yml
python src/f_Evaluate.py --config data/output/regression/MC_pH/forecasts/xgb_01/config_evaluate_xgb_01.yml

Multiple configs in one command (single line only):
python src/f_Evaluate.py --config data/output/regression/MC_pH/forecasts/gp_01/config_evaluate_gp_01.yml data/output/regression/MC_pH/forecasts/transformer_01/config_evaluate_model_transformer_01.yml data/output/regression/MC_pH/forecasts/xgb_01/config_evaluate_xgb_01.yml

Comma-separated and glob patterns are also supported:
python src/f_Evaluate.py --config "data/output/regression/MC_pH/forecasts/*/config_evaluate*.yml"
python src/f_Evaluate.py --config "cfg_a.yml,cfg_b.yml"
"""

import os
import re
import json
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import torch
import xgboost as xgb
try:
    import gpytorch
except ImportError:
    gpytorch = None

from utils.training import load_samples
from utils.transformer import TimeSeriesTargetDataset, TimeSeriesTransformer
from utils.evaluation import (
    load_secondary,
    evaluate_naive,
    evaluate_seasonal,
    evaluate_linear,
    evaluate_transformer,
    visualizer,
    classification_visualizer,
    reverse_normalize,
    binarize_predictions,
    apply_saved_normalize,
    boxplot_from_error_rows,
)
from utils.config_utils import (
    load_config,
    _resolve_path_from_config,
    _resolve_data_paths,
    _canonical_feature_name,
    _resolve_summary_dir,
    _load_uncertainty_std_map,
    _build_feature_uncertainty_variance,
    _build_feature_uncertainty_bundle,
)
from utils.gp_utils import apply_gp_constraints_and_priors, build_base_kernel, ExactGPRegressor
from utils.limits import load_limits_records


DEFAULT_EVAL_CONFIG = {
    "run_regression": True,
    "run_threshold_classification": False,
    "run_pure_classification": False,
    "run_baselines": False,
    "num_samples": None,
    "debug_plot": False,
    "debug_examples": 10,
    "gap_hours": 1,
    "window_hours": 550,
    "diurnal_window": 1,
    "historic_path": "../data/output/regression/Consolidated_sparse.csv",
    "thresholds_path": "../data/input/Limits.csv",
    "normalization_path": "../data/output/sensors/normalization.json",
    "use_normalized_thresholds": False,
    "baseline_sample_subdir": "samples",
    "baseline_split_file": "test_files.txt",
    "baseline_split_source": None,
    "baseline_match_mc_to_raw": True,
    "save_plots": True,
    "evaluate_all": False,  # If true, combine train and test samples for evaluation
    "collapse_mc_replicates_for_eval": False,
    "include_mc_stats_in_predictions": True,
}

EVAL_METRIC_SEMANTICS = "independent_sample_primary"
EVAL_METRIC_CONTRACT_VERSION = 1


def _parse_xgb_version() -> tuple[int, int]:
    """Parse xgboost major/minor version safely for runtime feature gating."""
    raw = str(getattr(xgb, "__version__", "0.0"))
    parts = []
    for token in raw.split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
        else:
            parts.append(0)
    major = parts[0] if len(parts) > 0 else 0
    minor = parts[1] if len(parts) > 1 else 0
    return int(major), int(minor)


def _xgb_eval_model_kwargs(device: torch.device) -> dict:
    """Build XGBoost model kwargs for GPU where available, with version fallback."""
    use_cuda = bool(torch.cuda.is_available()) and (str(device).lower().startswith("cuda"))
    major, _minor = _parse_xgb_version()
    if not use_cuda:
        return {"tree_method": "hist"}
    if major >= 2:
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}



def merge_eval_config(cfg):
    eval_cfg = cfg.get("evaluation", {})
    merged = DEFAULT_EVAL_CONFIG.copy()
    merged.update(eval_cfg)
    return merged


def load_model_config(data_dir, forecast_name, model_name, fallback_data=None):
    # First, look for model_config.json directly in the forecast_name directory
    direct_path = Path(data_dir, "forecasts", forecast_name, "model_config.json")
    if direct_path.exists():
        with open(direct_path, "r") as f:
            config_json = json.load(f)
        return config_json

    # Fallback: look for model_config.json in a model_name subdirectory (legacy)
    subdir_path = Path(data_dir, "forecasts", forecast_name, model_name, "model_config.json")
    if subdir_path.exists():
        with open(subdir_path, "r") as f:
            config_json = json.load(f)
        return config_json

    if fallback_data is None:
        raise FileNotFoundError(f"model_config.json not found at {direct_path} or {subdir_path}")

    return {
        "input_columns": fallback_data["input_columns"],
        "output_columns": fallback_data["output_columns"],
        "input_row_1": fallback_data["input_row_1"],
        "input_row_2": fallback_data["input_row_2"],
        "output_rows": fallback_data["output_rows"],
    }


def load_test_samples(
    data_dir,
    sample_subdir,
    forecast_name,
    input_columns,
    output_columns,
    input_rows,
    output_rows,
    input_aggregation="none",
):
    # Always resolve split file relative to data_dir/forecasts/forecast_name
    split_source_dir = os.path.join(data_dir, "forecasts", forecast_name)
    return load_split_samples(
        data_dir,
        sample_subdir,
        forecast_name,
        input_columns,
        output_columns,
        input_rows,
        output_rows,
        "test_files.txt",
        split_source_dir=split_source_dir,
        input_aggregation=input_aggregation,
    )


def _read_split_files(split_source_dir, split_file):
    split_path = Path(split_source_dir) / split_file
    abs_split_path = split_path.resolve()
    if not abs_split_path.exists():
        raise FileNotFoundError(f"Split file not found: {abs_split_path}")
    with open(abs_split_path) as f:
        return [line.strip() for line in f if line.strip()]


def _map_split_files_mc_to_raw(split_files):
    mapped = []
    seen = set()
    for file_name in split_files:
        mapped_name = re.sub(r"_mc_\d+(?=\.csv$)", "", file_name)
        if mapped_name not in seen:
            seen.add(mapped_name)
            mapped.append(mapped_name)
    return mapped


def _dedupe_split_files_by_base_sample(split_files):
    """Keep one existing split entry per base sample id while preserving order."""
    deduped = []
    seen = set()
    for file_name in split_files:
        base_name = re.sub(r"_mc_\d+(?=\.csv$)", "", str(file_name))
        if base_name in seen:
            continue
        seen.add(base_name)
        deduped.append(str(file_name))
    return deduped


def load_split_samples(
    data_dir,
    sample_subdir,
    forecast_name,
    input_columns,
    output_columns,
    input_rows,
    output_rows,
    split_file,
    split_source_dir=None,
    split_files_override=None,
    fault_tolerant=False,
    input_aggregation="none",
    drop_report=None,
):
    source_dir = Path(split_source_dir) if split_source_dir is not None else Path(data_dir, "forecasts", forecast_name)
    split_files = split_files_override if split_files_override is not None else _read_split_files(source_dir, split_file)

    samples = load_samples(
        os.path.join(data_dir, sample_subdir),
        input_columns=input_columns,
        output_columns=output_columns,
        input_rows=input_rows,
        output_rows=output_rows,
        file_list=split_files,
        fault_tolerant=fault_tolerant,
        input_aggregation=input_aggregation,
        drop_report=drop_report,
    )

    return samples


def get_output_dim(data_dir, sample_subdir, output_columns, output_rows):
    sample_files = sorted(os.listdir(Path(data_dir, sample_subdir)))
    sample_df = pd.read_csv(Path(data_dir, sample_subdir, sample_files[0]))
    return len(output_columns) * len(sample_df.iloc[output_rows:])


def _prepare_gp_train_arrays(train_samples, split_cfg, hyper_cfg):
    X_train_np = np.array([s[0].flatten() for s in train_samples], dtype=np.float32)
    y_train_np = np.array([s[1].flatten() for s in train_samples], dtype=np.float32)
    if y_train_np.ndim == 1:
        y_train_np = y_train_np.reshape(-1, 1)

    max_train_size = hyper_cfg.get("max_train_size")
    if max_train_size is not None and len(X_train_np) > max_train_size:
        rng = np.random.default_rng(split_cfg["random_state"])
        keep_idx = rng.choice(len(X_train_np), size=max_train_size, replace=False)
        X_train_np = X_train_np[keep_idx]
        y_train_np = y_train_np[keep_idx]

    return X_train_np, y_train_np


def _load_gp_bundle(data_cfg, split_cfg, model_name, train_samples, device, config_dir):
    if gpytorch is None:
        raise ImportError("gpytorch is not installed. Install it with: pip install gpytorch")

    model_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"], model_name)
    artifact = torch.load(model_path / "gp_model.pt", map_location=device, weights_only=False)
    hyper_cfg = artifact.get("hyperparameters", {})
    kernel_meta = artifact.get("kernel_metadata", {})

    X_train_np, y_train_np = _prepare_gp_train_arrays(train_samples, split_cfg, hyper_cfg)

    x_mean = np.array(artifact["input_mean"], dtype=np.float32)
    x_std = np.array(artifact["input_std"], dtype=np.float32)
    x_std[x_std < 1e-8] = 1.0

    if hyper_cfg.get("input_standardize", True):
        X_train_used = (X_train_np - x_mean) / x_std
    else:
        X_train_used = X_train_np

    X_train = torch.tensor(X_train_used, dtype=torch.float32, device=device)
    kernel_name = str(kernel_meta.get("requested_kernel", hyper_cfg.get("kernel", "matern52"))).lower()
    use_uncertain_kernel = bool(kernel_meta.get("use_uncertain_kernel", hyper_cfg.get("use_uncertain_input_kernel", True)))
    effective_ard = bool(kernel_meta.get("effective_ard", (hyper_cfg.get("ard", True) or use_uncertain_kernel)))
    ard_dims = X_train.shape[1] if effective_ard else None
    mc_samples = int(kernel_meta.get("uncertain_kernel_mc_samples", hyper_cfg.get("uncertain_kernel_mc_samples", 64)))
    mc_seed = int(kernel_meta.get("uncertain_kernel_mc_seed", hyper_cfg.get("uncertain_kernel_mc_seed", 0)))

    input_uncertainty_var = None
    uncertainty_noise_deltas = None
    if use_uncertain_kernel:
        saved_var = artifact.get("input_uncertainty_var")
        saved_deltas = artifact.get("uncertainty_noise_deltas")
        if saved_var is not None:
            input_uncertainty_var = torch.tensor(saved_var, dtype=torch.float32, device=device)
        if saved_deltas is not None:
            uncertainty_noise_deltas = torch.tensor(saved_deltas, dtype=torch.float32, device=device)

        if input_uncertainty_var is None:
            uncertainty_bundle = _build_feature_uncertainty_bundle(data_cfg, hyper_cfg, config_dir, verbose=False)
            input_uncertainty_var = torch.tensor(
                uncertainty_bundle["feature_variances"], dtype=torch.float32, device=device
            )
            uncertainty_noise_deltas = torch.tensor(
                uncertainty_bundle["noise_delta_samples"], dtype=torch.float32, device=device
            )

    models = []
    for state in artifact["models"]:
        output_idx = state["output_index"]
        y_train_col = y_train_np[:, output_idx]
        if hyper_cfg.get("target_standardize", True):
            y_train_used = (y_train_col - state["target_mean"]) / max(state["target_std"], 1e-8)
        else:
            y_train_used = y_train_col

        y_train = torch.tensor(y_train_used, dtype=torch.float32, device=device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPRegressor(
            X_train, y_train, likelihood,
            build_base_kernel(
                kernel_name,
                use_uncertain_kernel,
                input_uncertainty_var,
                ard_dims,
                uncertainty_noise_deltas=uncertainty_noise_deltas,
                uncertain_kernel_mc_samples=mc_samples,
                uncertain_kernel_mc_seed=mc_seed,
            )
        ).to(device)
        apply_gp_constraints_and_priors(model, likelihood, hyper_cfg)
        model.load_state_dict(state["model_state_dict"])
        likelihood.load_state_dict(state["likelihood_state_dict"])
        model.eval()
        likelihood.eval()

        models.append(
            {
                "model": model,
                "likelihood": likelihood,
                "target_mean": float(state["target_mean"]),
                "target_std": float(state["target_std"]),
            }
        )

    return {
        "models": models,
        "hyperparameters": hyper_cfg,
        "input_mean": x_mean,
        "input_std": x_std,
    }


def _predict_gp_bundle(gp_bundle, X_np, device):
    """Return dict with 'mean' [n, n_outputs] and 'variance' [n, n_outputs] arrays."""
    hyper_cfg = gp_bundle["hyperparameters"]
    input_mean = gp_bundle["input_mean"]
    input_std = gp_bundle["input_std"]

    if hyper_cfg.get("input_standardize", True):
        X_used = (X_np - input_mean) / input_std
    else:
        X_used = X_np

    X_tensor = torch.tensor(X_used, dtype=torch.float32, device=device)
    preds_mean = []
    preds_var = []
    for entry in gp_bundle["models"]:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = entry["likelihood"](entry["model"](X_tensor))
            pred_mean = pred_dist.mean.detach().cpu().numpy()
            pred_var  = pred_dist.variance.detach().cpu().numpy()
        pred_mean = pred_mean * entry["target_std"] + entry["target_mean"]
        pred_var  = pred_var  * (entry["target_std"] ** 2)
        preds_mean.append(pred_mean)
        preds_var.append(pred_var)

    return {
        "mean": np.stack(preds_mean, axis=1),
        "variance": np.stack(preds_var, axis=1),
    }


def load_model(model_type, data_cfg, split_cfg, model_name, model_config, device, train_samples=None, config_dir=None):
    # First, look for model files directly in the forecast_name directory
    direct_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
    subdir_path = direct_path / model_name

    def try_file(filename):
        # Try direct path first
        file = direct_path / filename
        if file.exists():
            return file
        # Fallback: try model_name subdirectory
        file = subdir_path / filename
        if file.exists():
            return file
        return None

    if model_type == "transformer":
        model = TimeSeriesTransformer(model_config).to(device)
        model_file = try_file("transformer_model.pt")
        if not model_file:
            raise FileNotFoundError(f"transformer_model.pt not found in {direct_path} or {subdir_path}")
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
        return model

    if model_type == "gp_regressor":
        if train_samples is None:
            raise ValueError("train_samples are required to evaluate gp_regressor")
        # Try direct path first (no model_name subdir)
        try:
            return _load_gp_bundle(data_cfg, split_cfg, "", train_samples, device, config_dir)
        except FileNotFoundError:
            try:
                return _load_gp_bundle(data_cfg, split_cfg, model_name, train_samples, device, config_dir)
            except Exception as e:
                raise FileNotFoundError(f"GP model not found in {direct_path} or {subdir_path}: {e}")

    if model_type == "xgb_regressor":
        model = xgb.XGBRegressor(**_xgb_eval_model_kwargs(device))
        model_file = try_file("xgboost_model.json")
        if not model_file:
            raise FileNotFoundError(f"xgboost_model.json not found in {direct_path} or {subdir_path}")
        model.load_model(model_file)
        print(
            "XGBoost eval runtime mode: "
            f"tree_method={model.get_params().get('tree_method')}, "
            f"device={model.get_params().get('device', 'n/a')}, "
            f"predictor={model.get_params().get('predictor', 'n/a')}, "
            f"xgboost_version={getattr(xgb, '__version__', 'unknown')}"
        )
        return model

    if model_type == "xgb_classifier":
        model = xgb.XGBClassifier(**_xgb_eval_model_kwargs(device))
        model_file = try_file("xgboost_model.json")
        if not model_file:
            raise FileNotFoundError(f"xgboost_model.json not found in {direct_path} or {subdir_path}")
        model.load_model(model_file)
        print(
            "XGBoost eval runtime mode: "
            f"tree_method={model.get_params().get('tree_method')}, "
            f"device={model.get_params().get('device', 'n/a')}, "
            f"predictor={model.get_params().get('predictor', 'n/a')}, "
            f"xgboost_version={getattr(xgb, '__version__', 'unknown')}"
        )
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


def load_thresholds(eval_cfg):
    thresholds_path = Path(eval_cfg["thresholds_path"])
    limits_records = load_limits_records(thresholds_path)
    if limits_records:
        row = {}
        for rec in limits_records:
            upper = rec.get("upper")
            lower = rec.get("lower")
            names = [n for n in [rec.get("translated_name"), rec.get("original_name")] if n]
            if not names:
                names = rec.get("names", [])
            for name in names:
                if upper is not None and name not in row:
                    row[name] = upper
                if lower is not None and f"{name}__lower" not in row:
                    row[f"{name}__lower"] = lower
        thresholds_df = pd.DataFrame([row]) if row else pd.DataFrame()
    else:
        thresholds_df = pd.read_csv(thresholds_path, sep=";", decimal=".")
    if eval_cfg["use_normalized_thresholds"]:
        thresholds_df = apply_saved_normalize(
            thresholds_df, param_file=Path(eval_cfg["normalization_path"])
        )
    return thresholds_df


def _baseline_output_rows_start(output_rows):
    if isinstance(output_rows, (list, tuple, np.ndarray)):
        if len(output_rows) == 0:
            return -1
        return int(output_rows[0])
    return output_rows


def _aligned_arrays(preds, targets, row_limit=None):
    pred_arr = np.array(preds)
    target_arr = np.array(targets)

    # Keep model I/O untouched; normalize metric inputs to 2D (n_samples, n_outputs_flat).
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


def _compute_flat_metrics(pred_arr: np.ndarray, target_arr: np.ndarray) -> dict:
    pred_flat = pred_arr.reshape(-1)
    target_flat = target_arr.reshape(-1)
    finite_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
    finite_count = int(np.sum(finite_mask))

    if finite_count <= 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "n_eval_points_finite": 0,
        }

    pred_vals = pred_flat[finite_mask]
    target_vals = target_flat[finite_mask]
    errors = pred_vals - target_vals

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    if finite_count > 1:
        ss_res = float(np.sum(np.square(errors)))
        ss_tot = float(np.sum(np.square(target_vals - np.mean(target_vals))))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    else:
        r2 = np.nan

    if finite_count > 1 and np.std(pred_vals) > 0 and np.std(target_vals) > 0:
        pearson_r = float(np.corrcoef(pred_vals, target_vals)[0, 1])
    else:
        pearson_r = np.nan

    std_target = float(np.std(target_vals)) if finite_count > 0 else np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "std_target": std_target,
        "n_eval_points_finite": finite_count,
    }


def _aggregate_by_independent_sample(
    pred_arr: np.ndarray,
    target_arr: np.ndarray,
    split_files: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Aggregate replicate rows into one mean row per independent sample id."""
    n_rows = pred_arr.shape[0]
    n_cols = pred_arr.shape[1] if pred_arr.ndim == 2 else 0
    if n_rows <= 0 or n_cols <= 0:
        return pred_arr[:0], target_arr[:0], 0, 0

    if not split_files:
        # No filename mapping available: treat each row as an independent sample.
        return pred_arr.copy(), target_arr.copy(), int(n_rows), int(n_rows)

    aligned_rows = min(n_rows, len(split_files))
    if aligned_rows <= 0:
        return pred_arr[:0], target_arr[:0], 0, 0

    pred_arr = pred_arr[:aligned_rows, :]
    target_arr = target_arr[:aligned_rows, :]

    grouped_pred: dict[str, list[np.ndarray]] = {}
    grouped_target: dict[str, list[np.ndarray]] = {}
    group_order: list[str] = []

    for idx in range(aligned_rows):
        gid = _base_sample_id(split_files[idx])
        if gid not in grouped_pred:
            group_order.append(gid)
            grouped_pred[gid] = []
            grouped_target[gid] = []
        grouped_pred[gid].append(pred_arr[idx, :])
        grouped_target[gid].append(target_arr[idx, :])

    agg_pred_rows: list[np.ndarray] = []
    agg_target_rows: list[np.ndarray] = []
    n_valid_independent = 0

    for gid in group_order:
        pred_group = np.asarray(grouped_pred[gid], dtype=float)
        target_group = np.asarray(grouped_target[gid], dtype=float)

        pred_count = np.sum(np.isfinite(pred_group), axis=0)
        pred_sum = np.nansum(pred_group, axis=0)
        pred_mean = np.full(pred_group.shape[1], np.nan, dtype=float)
        np.divide(pred_sum, pred_count, out=pred_mean, where=pred_count > 0)

        target_count = np.sum(np.isfinite(target_group), axis=0)
        target_sum = np.nansum(target_group, axis=0)
        target_mean = np.full(target_group.shape[1], np.nan, dtype=float)
        np.divide(target_sum, target_count, out=target_mean, where=target_count > 0)

        finite_pair_mask = np.isfinite(pred_mean) & np.isfinite(target_mean)
        if np.any(finite_pair_mask):
            n_valid_independent += 1

        agg_pred_rows.append(pred_mean)
        agg_target_rows.append(target_mean)

    return (
        np.asarray(agg_pred_rows, dtype=float),
        np.asarray(agg_target_rows, dtype=float),
        len(group_order),
        n_valid_independent,
    )


def _sanitize_label_for_filename(label):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_").lower()
    return safe or "model"


def _base_sample_id(file_name):
    return re.sub(r"_mc_\d+(?=\.csv$)", "", Path(str(file_name)).name)


def _build_replicate_groups(preds, targets, split_files, row_limit=None):
    pred_arr, target_arr, n_rows, n_cols = _aligned_arrays(preds, targets, row_limit=row_limit)
    if n_rows == 0 or n_cols == 0:
        return [], {}, {}, 0

    if split_files is None:
        split_files = []
    n_rows = min(n_rows, len(split_files))
    if n_rows == 0:
        return [], {}, {}, 0

    pred_arr = pred_arr[:n_rows, :]
    target_arr = target_arr[:n_rows, :]

    group_order = []
    grouped_preds = {}
    grouped_targets = {}

    for i in range(n_rows):
        group_id = _base_sample_id(split_files[i])
        if group_id not in grouped_preds:
            group_order.append(group_id)
            grouped_preds[group_id] = []
            grouped_targets[group_id] = []
        grouped_preds[group_id].append(pred_arr[i, :])
        grouped_targets[group_id].append(target_arr[i, :])

    return group_order, grouped_preds, grouped_targets, n_cols


def _plot_uncertainty_boxplots(regression_pairs, regression_labels, split_files_by_pair, directory, forecast_name, num_samples, sample_labels=None):
    if not regression_pairs:
        return

    base_dir = Path(directory, "forecasts", forecast_name) if forecast_name else Path(directory, "forecasts")
    base_dir.mkdir(parents=True, exist_ok=True)

    for idx, ((preds, targets), label, split_files) in enumerate(zip(regression_pairs, regression_labels, split_files_by_pair)):
        group_order, grouped_preds, grouped_targets, n_cols = _build_replicate_groups(
            preds,
            targets,
            split_files,
            row_limit=num_samples,
        )
        if not group_order or n_cols == 0:
            continue

        box_width = 0.05
        max_outputs = min(4, n_cols)
        fig_h = max(3.4, 2.6 * max_outputs)
        fig_w = max(9.5, min(18.0, 0.24 * len(group_order) + 8.5))
        fig, axes = plt.subplots(max_outputs, 1, figsize=(fig_w, fig_h), constrained_layout=True)
        if max_outputs == 1:
            axes = [axes]

        valid_axes = 0
        global_min = np.inf
        global_max = -np.inf

        for out_idx in range(max_outputs):
            ax = axes[out_idx]
            box_data = []
            x_ground_truth = []
            positions = []

            for i, group_id in enumerate(group_order):
                pred_group = np.array(grouped_preds[group_id], dtype=float)
                target_group = np.array(grouped_targets[group_id], dtype=float)
                if pred_group.ndim != 2 or target_group.ndim != 2:
                    continue

                pred_vals = pred_group[:, out_idx]
                target_vals_group = target_group[:, out_idx]

                pred_vals = pred_vals[np.isfinite(pred_vals)]
                target_vals_group = target_vals_group[np.isfinite(target_vals_group)]
                if len(pred_vals) == 0 or len(target_vals_group) == 0:
                    continue

                gt_x = float(np.median(target_vals_group))
                if not np.isfinite(gt_x):
                    continue

                # If sample_labels is provided, color train/test differently
                if sample_labels is not None and len(sample_labels) == len(split_files):
                    group_indices = [j for j, f in enumerate(split_files) if _base_sample_id(f) == group_id]
                    group_sample_labels = [sample_labels[j] for j in group_indices]
                    # Use color for train/test
                    if all(lbl == "train" for lbl in group_sample_labels):
                        box_color = "#1f77b4"
                    elif all(lbl == "test" for lbl in group_sample_labels):
                        box_color = "#ff7f0e"
                    else:
                        box_color = "gray"
                    # box_width is set to default above; can be updated later if needed
                    bp = ax.boxplot([pred_vals], positions=[gt_x], widths=box_width, showfliers=False, patch_artist=True, boxprops=dict(facecolor=box_color, alpha=0.5))
                else:
                    box_data.append(pred_vals)
                    x_ground_truth.append(gt_x)
                    positions.append(gt_x)

            if not box_data:
                ax.set_visible(False)
                continue

            valid_axes += 1
            x_span = float(np.nanmax(x_ground_truth) - np.nanmin(x_ground_truth)) if len(x_ground_truth) > 1 else 0.0
            box_width = max(0.015, min(0.2, 0.015 * x_span)) if x_span > 0 else 0.05

            ax.boxplot(box_data, positions=positions, widths=box_width, showfliers=False)

            all_pred = np.concatenate([np.asarray(vals, dtype=float) for vals in box_data]) if box_data else np.array([])
            finite_pred = all_pred[np.isfinite(all_pred)]
            finite_gt = np.array(x_ground_truth, dtype=float)
            finite_gt = finite_gt[np.isfinite(finite_gt)]
            if len(finite_pred) > 0 and len(finite_gt) > 0:
                diag_min = float(min(np.min(finite_gt), np.min(finite_pred)))
                diag_max = float(max(np.max(finite_gt), np.max(finite_pred)))
                global_min = min(global_min, diag_min)
                global_max = max(global_max, diag_max)

            ax.set_ylabel("Prediction")
            ax.set_xlabel("Ground truth")
            ax.grid(alpha=0.25)

        if valid_axes == 0:
            plt.close(fig)
            print(f"[INFO] Skipping uncertainty boxplot for {label}: no finite grouped values.")
            continue

        if np.isfinite(global_min) and np.isfinite(global_max):
            span = global_max - global_min
            if span > 0:
                pad = 0.03 * span
            else:
                pad = 0.03 * max(1.0, abs(global_max))
            axis_min = global_min - pad
            axis_max = global_max + pad
            tick_fmt = FuncFormatter(lambda val, _: f"{val:.3g}")

            for ax in axes:
                if not ax.get_visible():
                    continue
                ax.set_xlim(axis_min, axis_max)
                ax.set_ylim(axis_min, axis_max)
                ax.set_aspect("equal", adjustable="box")
                ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.xaxis.set_major_formatter(tick_fmt)
                ax.yaxis.set_major_formatter(tick_fmt)
                ax.tick_params(axis="both", labelsize=9)

        out_path = base_dir / f"predictions_uncertainty_boxplot_{_sanitize_label_for_filename(label)}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"[INFO] Wrote uncertainty boxplot: {out_path}")


def _has_mc_replicate_distribution_for_uncertainty_plot(
    regression_pairs,
    split_files_by_pair,
    row_limit=None,
):
    """Return True when any model plot entry contains >1 evaluated rows for a base sample id."""
    for pair, split_files in zip(regression_pairs, split_files_by_pair):
        if split_files is None:
            continue
        preds, targets = pair
        _, _, n_rows, n_cols = _aligned_arrays(preds, targets, row_limit=row_limit)
        if n_rows <= 0 or n_cols <= 0:
            continue

        aligned_rows = min(n_rows, len(split_files))
        if aligned_rows <= 1:
            continue

        counts_by_base = {}
        for idx in range(aligned_rows):
            base_id = _base_sample_id(split_files[idx])
            counts_by_base[base_id] = counts_by_base.get(base_id, 0) + 1

        if any(count > 1 for count in counts_by_base.values()):
            return True

    return False


def _compute_regression_summary(label, preds, targets, num_samples, metadata=None, split_files=None):
    metadata = metadata or {}
    pred_arr, target_arr, n_rows, n_cols = _aligned_arrays(preds, targets, row_limit=num_samples)

    # Replicate-population metrics: preserves prior behavior for uncertainty diagnostics.
    rep_metrics = _compute_flat_metrics(pred_arr, target_arr)

    # Independent-sample metrics: aggregate replicates by base sample id, then score.
    ind_pred, ind_target, n_test_independent, n_test_valid = _aggregate_by_independent_sample(
        pred_arr,
        target_arr,
        split_files,
    )
    ind_metrics = _compute_flat_metrics(ind_pred, ind_target)

    row = {
        "label": label,
        "n_eval_rows": int(n_rows),
        "n_eval_outputs": int(n_cols),
        # Primary metrics for ranking/comparison: independent-sample aggregation.
        "n_eval_points_finite": int(ind_metrics["n_eval_points_finite"]),
        "mae": float(ind_metrics["mae"]),
        "rmse": float(ind_metrics["rmse"]),
        "r2": float(ind_metrics["r2"]),
        "pearson_r": float(ind_metrics["pearson_r"]),
        "std_target": float(ind_metrics["std_target"]),
        # Replicate-population diagnostic metrics.
        "mae_replicate": float(rep_metrics["mae"]),
        "rmse_replicate": float(rep_metrics["rmse"]),
        "r2_replicate": float(rep_metrics["r2"]),
        "pearson_r_replicate": float(rep_metrics["pearson_r"]),
        "n_eval_points_finite_replicate": int(rep_metrics["n_eval_points_finite"]),
        # Explicit count semantics.
        "n_test_independent": int(n_test_independent),
        "n_test_valid": int(n_test_valid),
        "n_test_evals": int(n_rows),
        # Explicit contract marker for downstream strict consumers.
        "metric_semantics": EVAL_METRIC_SEMANTICS,
        "metric_contract_version": int(EVAL_METRIC_CONTRACT_VERSION),
    }
    row.update(metadata)
    return row


def _compute_classification_summary(label, preds, targets, num_samples, metadata=None):
    metadata = metadata or {}
    pred_arr = np.array(preds).reshape(-1)[:num_samples]
    target_arr = np.array(targets).reshape(-1)[:num_samples]
    n = min(len(pred_arr), len(target_arr))
    pred_arr = pred_arr[:n]
    target_arr = target_arr[:n]

    finite_mask = np.isfinite(pred_arr) & np.isfinite(target_arr)
    pred_arr = pred_arr[finite_mask]
    target_arr = target_arr[finite_mask]

    if len(pred_arr) > 0:
        pred_bin = np.rint(pred_arr).astype(int)
        target_bin = np.rint(target_arr).astype(int)

        tp = int(np.sum((pred_bin == 1) & (target_bin == 1)))
        tn = int(np.sum((pred_bin == 0) & (target_bin == 0)))
        fp = int(np.sum((pred_bin == 1) & (target_bin == 0)))
        fn = int(np.sum((pred_bin == 0) & (target_bin == 1)))

        denom = tp + tn + fp + fn
        accuracy = float((tp + tn) / denom) if denom > 0 else np.nan
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    else:
        accuracy = np.nan
        precision = np.nan
        recall = np.nan
        f1 = np.nan

    row = {
        "label": label,
        "n_eval_rows": int(n),
        "n_eval_outputs": 1,
        "n_eval_points_finite": int(np.sum(np.isfinite(np.array(preds).reshape(-1)[:n]) & np.isfinite(np.array(targets).reshape(-1)[:n]))),
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "pearson_r": np.nan,
        "metric_semantics": EVAL_METRIC_SEMANTICS,
        "metric_contract_version": int(EVAL_METRIC_CONTRACT_VERSION),
    }
    row.update(metadata)
    return row


# All variables are min-max normalized to [0, 1] before training, so this is the
# target's known support rather than a tuning choice.
TARGET_SUPPORT = (0.0, 1.0)


def _clip_to_target_support(preds, label, bounds=TARGET_SUPPORT):
    """Clip predictions to the target's normalized support, reporting how many moved.

    A prediction outside [0, 1] is not a forecast of a min-max normalized target;
    it is unbounded extrapolation. The Matern-5/2 + linear kernel has an
    unbounded linear component, so a Gaussian process given a test input far from
    its training data can return an arbitrarily large mean: across this study 15
    of 58 GP runs contained at least one such point, the worst 106x outside the
    target range, while XGBoost (3716 runs) and the Transformer (75) contained
    none. A single point is enough to dominate a squared-error metric -- one
    prediction of -15.3 on a target spanning 0.42 to 0.53 drove an R^2 of
    -22173 -- so leaving them unconstrained lets one extrapolation decide which
    model is reported as best.

    Clipping is applied to every model family so the constraint stays a property
    of the target rather than of the estimator, and the count is returned so the
    correction is recorded next to the metrics it changes rather than applied
    silently.
    """
    if preds is None:
        return None, 0
    arr = np.asarray(preds, dtype=float)
    lo, hi = float(bounds[0]), float(bounds[1])
    finite = np.isfinite(arr)
    clipped = arr.copy()
    clipped[finite] = np.clip(arr[finite], lo, hi)
    n_clipped = int(np.sum(finite & (clipped != arr)))
    if n_clipped:
        worst = float(np.nanmax(np.abs(arr[finite]))) if np.any(finite) else float("nan")
        print(f"[WARN] {label}: {n_clipped} prediction(s) outside the target support "
              f"[{lo:g}, {hi:g}] clipped; largest magnitude was {worst:.3g}. "
              "This indicates extrapolation beyond the training data, not a forecast.")
    return clipped.reshape(arr.shape), n_clipped


def _split_size_metadata(train_samples, test_samples, train_drops=None, test_drops=None):
    """Summary columns recording how large each split was and what it lost.

    ``n_train_samples`` was declared in the summary schema but never populated by
    this module, so it was null in the overwhelming majority of run directories
    and the training-set cost of a predictor choice could not be read off the
    outputs at all. The drop counts and the attributed predictor make that cost
    a recorded quantity: one partial-coverage predictor can remove three
    quarters of a target's samples, and that has to be visible next to the
    metrics it produced.
    """
    meta = {
        "n_train_samples": int(len(train_samples)) if train_samples is not None else np.nan,
        "n_test_samples_loaded": int(len(test_samples)) if test_samples is not None else np.nan,
    }
    for prefix, report in (("train", train_drops), ("test", test_drops)):
        if not isinstance(report, dict) or not report:
            continue
        considered = report.get("n_considered")
        loaded = report.get("n_loaded")
        meta[f"{prefix}_n_considered"] = considered
        meta[f"{prefix}_n_dropped"] = (
            considered - loaded if considered is not None and loaded is not None else np.nan)
        meta[f"{prefix}_drop_rate"] = (
            float(1.0 - loaded / considered)
            if considered not in (None, 0) and loaded is not None else np.nan)
        meta[f"{prefix}_dropped_nan_input"] = report.get("dropped_nan_input")
        meta[f"{prefix}_dropped_nan_output"] = report.get("dropped_nan_output")
        meta[f"{prefix}_dropped_missing_columns"] = report.get("dropped_missing_columns")
        culprits = sorted((report.get("nan_input_columns") or {}).items(),
                          key=lambda kv: -kv[1])
        meta[f"{prefix}_drop_predictors"] = "; ".join(
            f"{col}:{n}" for col, n in culprits) or ""
    return meta


def _write_summary_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = [
        "label",
        "mae",
        "rmse",
        "r2",
        "pearson_r",
        "std_target",
        "mae_replicate",
        "rmse_replicate",
        "r2_replicate",
        "pearson_r_replicate",
        "skill_v_naive",
        "kind",
        "gp_uncertainty_mode",
        "n_test_independent",
        "n_test_valid",
        "n_test_evals",
        "n_train_samples",
        "n_test_samples",
        "n_test_samples_loaded",
        "n_predictions_clipped",
        "train_n_considered",
        "train_n_dropped",
        "train_drop_rate",
        "train_dropped_nan_input",
        "train_dropped_nan_output",
        "train_dropped_missing_columns",
        "train_drop_predictors",
        "test_n_considered",
        "test_n_dropped",
        "test_drop_rate",
        "test_dropped_nan_input",
        "test_dropped_nan_output",
        "test_dropped_missing_columns",
        "test_drop_predictors",
        "n_eval_rows",
        "n_eval_outputs",
        "n_eval_points_finite",
        "n_eval_points_finite_replicate",
        "metric_semantics",
        "metric_contract_version",
        "input_dim",
        "target_dim",
        "data_dir",
    ]

    if rows:
        df = pd.DataFrame(rows)
        for col in ordered_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[ordered_columns]
        df.to_csv(output_path, index=False)
    else:
        pd.DataFrame(columns=ordered_columns).to_csv(output_path, index=False)
    print(f"[INFO] Wrote evaluation summary CSV: {output_path}")


def _prediction_target_columns(n_outputs):
    if n_outputs <= 1:
        return ["target"]
    return [f"target_{i}" for i in range(n_outputs)]


def _prediction_value_columns(label, n_outputs):
    base = str(label)
    if n_outputs <= 1:
        return [base]
    return [f"{base}_{i}" for i in range(n_outputs)]


def _extract_mc_index(file_name):
    match = re.search(r"_mc_(\d+)(?=\.csv$)", Path(str(file_name)).name)
    if match:
        return int(match.group(1))
    return None


def _prediction_mc_columns(replicate_ids, n_outputs):
    ordered_ids = sorted(int(r) for r in replicate_ids)
    cols = []
    if n_outputs <= 1:
        for rep_id in ordered_ids:
            cols.append(f"mc_{rep_id:03d}")
        return cols

    for rep_id in ordered_ids:
        for out_idx in range(n_outputs):
            cols.append(f"mc_{rep_id:03d}_{out_idx}")
    return cols


def _group_independent_prediction_stats(preds, targets, split_files):
    pred_arr, target_arr, n_rows, n_cols = _aligned_arrays(preds, targets)
    if n_rows <= 0 or n_cols <= 0:
        return [], 0

    if split_files:
        n_rows = min(n_rows, len(split_files))
        split_names = [Path(str(s)).name for s in split_files[:n_rows]]
    else:
        split_names = [f"sample_{i:06d}.csv" for i in range(n_rows)]

    if n_rows <= 0:
        return [], 0

    pred_arr = pred_arr[:n_rows, :]
    target_arr = target_arr[:n_rows, :]

    grouped_pred = {}
    grouped_target = {}
    grouped_replicates = {}
    grouped_used_rep_ids = {}
    grouped_next_rep_id = {}
    group_order = []

    for idx in range(n_rows):
        sample_file = _base_sample_id(split_names[idx])
        if sample_file not in grouped_pred:
            group_order.append(sample_file)
            grouped_pred[sample_file] = []
            grouped_target[sample_file] = []
            grouped_replicates[sample_file] = []
            grouped_used_rep_ids[sample_file] = set()
            grouped_next_rep_id[sample_file] = 1
        grouped_pred[sample_file].append(pred_arr[idx, :])
        grouped_target[sample_file].append(target_arr[idx, :])

        rep_id = _extract_mc_index(split_names[idx])
        if rep_id is None:
            rep_id = grouped_next_rep_id[sample_file]
        while rep_id in grouped_used_rep_ids[sample_file]:
            rep_id += 1
        grouped_used_rep_ids[sample_file].add(rep_id)
        grouped_next_rep_id[sample_file] = max(grouped_next_rep_id[sample_file], rep_id + 1)
        grouped_replicates[sample_file].append((int(rep_id), np.asarray(pred_arr[idx, :], dtype=float).copy()))

    grouped_rows = []
    for sample_file in group_order:
        pred_group = np.asarray(grouped_pred[sample_file], dtype=float)
        target_group = np.asarray(grouped_target[sample_file], dtype=float)

        pred_count = np.sum(np.isfinite(pred_group), axis=0)
        pred_sum = np.nansum(pred_group, axis=0)
        pred_mean = np.full(pred_group.shape[1], np.nan, dtype=float)
        np.divide(pred_sum, pred_count, out=pred_mean, where=pred_count > 0)

        pred_std = np.full(pred_group.shape[1], np.nan, dtype=float)
        for col_idx in range(pred_group.shape[1]):
            col_vals = pred_group[:, col_idx]
            col_vals = col_vals[np.isfinite(col_vals)]
            if len(col_vals) > 0:
                pred_std[col_idx] = float(np.std(col_vals, ddof=0))

        target_count = np.sum(np.isfinite(target_group), axis=0)
        target_sum = np.nansum(target_group, axis=0)
        target_mean = np.full(target_group.shape[1], np.nan, dtype=float)
        np.divide(target_sum, target_count, out=target_mean, where=target_count > 0)

        grouped_rows.append(
            {
                "sample_file": sample_file,
                "target_mean": target_mean,
                "pred_mean": pred_mean,
                "pred_std": pred_std,
                "replicate_preds": sorted(grouped_replicates[sample_file], key=lambda item: item[0]),
                "n_replicates": int(pred_group.shape[0]),
            }
        )

    return grouped_rows, int(n_cols)


def _group_gp_var_by_sample(gp_var, split_files, n_outputs):
    """Return dict mapping sample_file base ID -> mean predictive std (shape [n_outputs]).

    gp_var is shape [n_rows, n_outputs]. Rows belonging to the same base sample
    (MC replicates) are averaged in variance space before taking the sqrt.
    Returns None if gp_var is None or empty.
    """
    if gp_var is None:
        return None
    gp_var = np.asarray(gp_var, dtype=float)
    n_rows = gp_var.shape[0]
    if n_rows == 0:
        return None
    if split_files:
        n_rows = min(n_rows, len(split_files))
        split_names = [Path(str(s)).name for s in split_files[:n_rows]]
    else:
        split_names = [f"sample_{i:06d}.csv" for i in range(n_rows)]

    grouped = {}
    order = []
    for idx in range(n_rows):
        key = _base_sample_id(split_names[idx])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(gp_var[idx, :n_outputs])

    result = {}
    for key in order:
        arr = np.asarray(grouped[key], dtype=float)  # [k, n_outputs]
        mean_var = np.nanmean(arr, axis=0)            # average variance across replicates
        result[key] = np.sqrt(np.maximum(mean_var, 0.0))
    return result


def _build_predictions_table(entries, gp_uncertainty_mode, include_mc_output_columns=True):
    rows_by_key = {}
    key_order = []
    predictor_columns = []
    target_columns = []
    mc_mean_columns = []
    mc_std_columns = []
    gp_std_columns = []
    mc_replicate_ids = set()
    n_outputs_ref = None
    has_gp_std = False

    for entry in entries:
        grouped_rows, n_outputs = _group_independent_prediction_stats(
            entry.get("preds"),
            entry.get("targets"),
            entry.get("split_files"),
        )
        if not grouped_rows or n_outputs <= 0:
            continue

        if n_outputs_ref is None:
            n_outputs_ref = n_outputs
            target_columns = _prediction_target_columns(n_outputs_ref)
            mc_mean_columns = _prediction_value_columns("mc_pred_mean", n_outputs_ref)
            mc_std_columns = _prediction_value_columns("mc_pred_std", n_outputs_ref)
            gp_std_columns = _prediction_value_columns("gp_pred_std", n_outputs_ref)
        elif n_outputs != n_outputs_ref:
            # Keep a single stable schema across model/baseline entries.
            for row in grouped_rows:
                row["target_mean"] = row["target_mean"][:n_outputs_ref]
                row["pred_mean"] = row["pred_mean"][:n_outputs_ref]
                row["pred_std"] = row["pred_std"][:n_outputs_ref]
                row["replicate_preds"] = [
                    (rep_id, np.asarray(rep_vals, dtype=float)[:n_outputs_ref]) for rep_id, rep_vals in row["replicate_preds"]
                ]

        value_columns = _prediction_value_columns(entry["label"], n_outputs_ref)
        for col in value_columns:
            if col not in predictor_columns:
                predictor_columns.append(col)

        kind = str(entry.get("kind", "test"))
        include_mc_stats = bool(entry.get("include_mc_stats", False))
        gp_std_by_sample = _group_gp_var_by_sample(
            entry.get("gp_var"),
            entry.get("split_files"),
            n_outputs_ref,
        )
        if gp_std_by_sample:
            has_gp_std = True
        if include_mc_stats:
            for grouped in grouped_rows:
                for rep_id, _ in grouped.get("replicate_preds", []):
                    mc_replicate_ids.add(int(rep_id))
        for grouped in grouped_rows:
            key = (kind, grouped["sample_file"])
            if key not in rows_by_key:
                row_template = {
                    "kind": kind,
                    "sample_file": grouped["sample_file"],
                    "gp_uncertainty_mode": gp_uncertainty_mode,
                    "metric_semantics": EVAL_METRIC_SEMANTICS,
                    "metric_contract_version": int(EVAL_METRIC_CONTRACT_VERSION),
                }
                if include_mc_output_columns:
                    row_template["mc_n_replicates"] = np.nan
                rows_by_key[key] = row_template
                key_order.append(key)

            row = rows_by_key[key]
            for col_idx, target_col in enumerate(target_columns):
                target_val = float(grouped["target_mean"][col_idx])
                if target_col not in row or not np.isfinite(row[target_col]):
                    row[target_col] = target_val

            for col_idx, pred_col in enumerate(value_columns):
                row[pred_col] = float(grouped["pred_mean"][col_idx])

            if include_mc_output_columns and include_mc_stats:
                row["mc_n_replicates"] = int(grouped["n_replicates"])
                for rep_id, rep_vals in grouped.get("replicate_preds", []):
                    if n_outputs_ref <= 1:
                        row[f"mc_{int(rep_id):03d}"] = float(rep_vals[0])
                    else:
                        for out_idx in range(n_outputs_ref):
                            row[f"mc_{int(rep_id):03d}_{out_idx}"] = float(rep_vals[out_idx])
                for col_idx, mc_col in enumerate(mc_mean_columns):
                    row[mc_col] = float(grouped["pred_mean"][col_idx])
                for col_idx, mc_col in enumerate(mc_std_columns):
                    row[mc_col] = float(grouped["pred_std"][col_idx])

            if gp_std_by_sample is not None:
                sample_std = gp_std_by_sample.get(grouped["sample_file"])
                if sample_std is not None:
                    for col_idx, gp_col in enumerate(gp_std_columns):
                        if col_idx < len(sample_std):
                            row[gp_col] = float(sample_std[col_idx])

    rows = [rows_by_key[key] for key in key_order]
    ordered_columns = [
        "kind",
        "sample_file",
        "gp_uncertainty_mode",
        "metric_semantics",
        "metric_contract_version",
    ]
    if include_mc_output_columns:
        ordered_columns.append("mc_n_replicates")
    ordered_columns.extend(target_columns)
    ordered_columns.extend(predictor_columns)
    if include_mc_output_columns:
        ordered_columns.extend(_prediction_mc_columns(mc_replicate_ids, n_outputs_ref if n_outputs_ref is not None else 0))
        ordered_columns.extend(mc_mean_columns)
        ordered_columns.extend(mc_std_columns)
    if has_gp_std:
        ordered_columns.extend(gp_std_columns)

    return rows, ordered_columns


def _write_predictions_csv(rows, output_path, ordered_columns):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        df = pd.DataFrame(rows)
        for col in ordered_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[ordered_columns]
    else:
        df = pd.DataFrame(columns=ordered_columns)

    df.to_csv(output_path, index=False)
    print(f"[INFO] Wrote predictions CSV: {output_path}")


def _build_boxplot_error_rows_from_predictions(
    predictions_rows,
    model_label=None,
    baseline_labels=None,
    gp_boxplot_seed=0,
):
    """Build long-form boxplot rows from predictions-table semantics.

    Output columns: Dataset, Error, Kind.
    - ML model uses mc replicate columns (if present), otherwise model column.
    - For GP models with gp_pred_std_* columns and no MC replicates, synthetic
      samples are drawn from Normal(mean, std) using the mc_n_replicates count
      (or 10 if unavailable) to produce a comparable distribution.
    - Baselines always use their own prediction columns.
    - All row kinds are retained (train/test/combined/etc.).
    """
    baseline_labels = baseline_labels or []
    if not predictions_rows:
        return pd.DataFrame(columns=["Dataset", "Error", "Kind"])

    df = pd.DataFrame(predictions_rows)
    if df.empty:
        return pd.DataFrame(columns=["Dataset", "Error", "Kind"])

    kind_series = df.get("kind", pd.Series(["unknown"] * len(df)))
    kind_series = kind_series.fillna("unknown").astype(str)

    target_specs = []
    if "target" in df.columns:
        target_specs.append((None, "target"))

    indexed_targets = []
    for col in df.columns:
        match = re.fullmatch(r"target_(\d+)", str(col))
        if match:
            indexed_targets.append((int(match.group(1)), str(col)))
    indexed_targets.sort(key=lambda x: x[0])
    target_specs.extend(indexed_targets)

    if not target_specs:
        return pd.DataFrame(columns=["Dataset", "Error", "Kind"])

    out_frames = []

    def _append_errors(dataset_label, pred_series, target_series, kind_override=None):
        pred_vals = pd.to_numeric(pred_series, errors="coerce")
        target_vals = pd.to_numeric(target_series, errors="coerce")
        err_vals = pred_vals - target_vals
        finite_mask = np.isfinite(err_vals.to_numpy(dtype=float))
        if not np.any(finite_mask):
            return
        k_series = kind_override if kind_override is not None else kind_series.to_numpy()[finite_mask]
        out_frames.append(
            pd.DataFrame(
                {
                    "Dataset": [str(dataset_label)] * int(np.sum(finite_mask)),
                    "Error": err_vals.to_numpy(dtype=float)[finite_mask],
                    "Kind": k_series,
                }
            )
        )

    def _append_gp_synth_errors(dataset_label, mean_series, std_series, target_series, n_samples_series):
        """Synthesize Normal(mean, std) samples and compute errors against target."""
        rng = np.random.default_rng(gp_boxplot_seed)
        mean_arr = pd.to_numeric(mean_series, errors="coerce").to_numpy(dtype=float)
        std_arr  = pd.to_numeric(std_series,  errors="coerce").to_numpy(dtype=float)
        tgt_arr  = pd.to_numeric(target_series, errors="coerce").to_numpy(dtype=float)
        n_rows = len(mean_arr)
        synth_errors = []
        synth_kinds = []
        for i in range(n_rows):
            if not (np.isfinite(mean_arr[i]) and np.isfinite(std_arr[i]) and np.isfinite(tgt_arr[i])):
                continue
            n_s = int(n_samples_series.iloc[i]) if np.isfinite(n_samples_series.iloc[i]) else 10
            n_s = max(1, n_s)
            samples = rng.normal(mean_arr[i], std_arr[i], size=n_s)
            errors = samples - tgt_arr[i]
            synth_errors.extend(errors.tolist())
            synth_kinds.extend([kind_series.iloc[i]] * n_s)
        if not synth_errors:
            return
        out_frames.append(
            pd.DataFrame(
                {
                    "Dataset": [str(dataset_label)] * len(synth_errors),
                    "Error": synth_errors,
                    "Kind": synth_kinds,
                }
            )
        )

    # Determine n_samples_series from mc_n_replicates column, defaulting to 10
    if "mc_n_replicates" in df.columns:
        n_samples_series = pd.to_numeric(df["mc_n_replicates"], errors="coerce").fillna(10)
    else:
        n_samples_series = pd.Series([10] * len(df))

    for output_idx, target_col in target_specs:
        target_vals = df[target_col]

        if output_idx is None:
            mc_cols = sorted([c for c in df.columns if re.fullmatch(r"mc_\d{3}", str(c))])
            model_col = str(model_label) if model_label else None
            gp_std_col = "gp_pred_std" if "gp_pred_std" in df.columns else None
        else:
            mc_cols = sorted([c for c in df.columns if re.fullmatch(rf"mc_\d{{3}}_{output_idx}", str(c))])
            model_col = f"{model_label}_{output_idx}" if model_label else None
            gp_std_col_candidate = f"gp_pred_std_{output_idx}"
            gp_std_col = gp_std_col_candidate if gp_std_col_candidate in df.columns else None

        if model_col and model_col in df.columns:
            if mc_cols:
                for mc_col in mc_cols:
                    _append_errors(model_col, df[mc_col], target_vals)
            elif gp_std_col is not None:
                _append_gp_synth_errors(model_col, df[model_col], df[gp_std_col], target_vals, n_samples_series)
            else:
                _append_errors(model_col, df[model_col], target_vals)

        for baseline_label in baseline_labels:
            baseline_col = str(baseline_label) if output_idx is None else f"{baseline_label}_{output_idx}"
            if baseline_col in df.columns:
                _append_errors(baseline_col, df[baseline_col], target_vals)

    if not out_frames:
        return pd.DataFrame(columns=["Dataset", "Error", "Kind"])

    return pd.concat(out_frames, ignore_index=True)


def _model_label(model_type):
    mapping = {
        "transformer": "Transformer",
        "xgb_regressor": "XGBRegressor",
        "gp_regressor": "GPRegressor",
        "xgb_classifier": "XGBClassifier",
    }
    return mapping.get(model_type, model_type)


def _combined_model_label(data_cfg, model_type):
    base = data_cfg.get("forecast_name", "model")
    return str(base)


def _resolve_gp_uncertainty_mode(model_type, model_config):
    if str(model_type).lower() != "gp_regressor":
        return "not_gp"

    use_uncertain = bool(model_config.get("use_uncertain_input_kernel", True))
    source_mode = str(model_config.get("uncertainty_source_mode_effective", "unknown")).strip() or "unknown"

    if use_uncertain:
        return f"uncertain_input_kernel:{source_mode}"
    return "point_input_kernel"


def _expand_config_inputs(config_args):
    expanded = []
    seen = set()

    continuation_tokens = {"\\", "/", "`"}

    for arg in config_args:
        parts = [p.strip() for p in str(arg).split(",") if p.strip()]
        for part in parts:
            if part in continuation_tokens:
                continue

            matches = sorted(glob.glob(part)) if any(ch in part for ch in "*?[]") else [part]
            for match in matches:
                if str(match).strip() in continuation_tokens:
                    continue

                match_path = Path(match)
                if match_path.is_dir():
                    continue

                resolved = str(Path(match).resolve())
                suffix = Path(resolved).suffix.lower()
                if suffix not in {".yml", ".yaml", ".json"}:
                    continue

                if resolved not in seen:
                    seen.add(resolved)
                    expanded.append(resolved)

    return expanded


def evaluate_single_config(config_path, save_plots_override=None):
    print(f"\n=== Evaluating config: {config_path} ===")

    config = load_config(config_path)
    if gpytorch is None and str(config.get("model_type", "")) == "gp_regressor":
        raise ImportError("gpytorch is not installed. Install it with: pip install gpytorch")
    config_dir = config["__config_dir"]

    model_type = config["model_type"]
    model_name = config["model_name"]
    data_cfg = config["data"]
    eval_cfg = merge_eval_config(config)
    if save_plots_override is None:
        save_plots = bool(eval_cfg.get("save_plots", True))
    else:
        save_plots = bool(save_plots_override)

    data_cfg["data_dir"], data_cfg["sample_subdir"] = _resolve_data_paths(data_cfg, config_dir)
    for key in ["historic_path", "thresholds_path", "normalization_path"]:
        if eval_cfg.get(key):
            eval_cfg[key] = str(_resolve_path_from_config(eval_cfg[key], config_dir))
    if eval_cfg.get("baseline_split_source"):
        eval_cfg["baseline_split_source"] = str(_resolve_path_from_config(eval_cfg["baseline_split_source"], config_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    matplotlib.use("Agg")

    model_config = load_model_config(
        data_cfg["data_dir"],
        data_cfg["forecast_name"],
        model_name,
        fallback_data=data_cfg,
    )
    gp_uncertainty_mode = _resolve_gp_uncertainty_mode(model_type, model_config)

    input_columns = model_config["input_columns"]
    output_columns = model_config["output_columns"]
    input_rows = slice(model_config["input_row_1"], model_config["input_row_2"])
    output_rows = model_config["output_rows"]
    input_aggregation = str(model_config.get("input_aggregation", data_cfg.get("input_aggregation", "none"))).lower()
    split_cfg = config.get("data_split", {"random_state": 42})
    split_fault_tolerant = bool(split_cfg.get("fault_tolerant", False))
    collapse_mc_for_eval = bool(eval_cfg.get("collapse_mc_replicates_for_eval", False))
    include_mc_stats_in_predictions = bool(eval_cfg.get("include_mc_stats_in_predictions", True))


    # Use the original forecast directory if present, else fallback to config_path.parent
    config_path = Path(config_path)
    split_base_dir = Path(data_cfg.get("forecast_dir", config_path.parent))
    model_split_files = _read_split_files(
        split_base_dir,
        "test_files.txt",
    )
    if collapse_mc_for_eval:
        model_split_files = _dedupe_split_files_by_base_sample(model_split_files)
    # Drop reports are threaded through so the run records how many samples the
    # chosen predictor set cost it, and which predictors were responsible.
    test_drop_report: dict = {}
    train_drop_report: dict = {}
    test_samples = load_split_samples(
        data_cfg["data_dir"],
        data_cfg["sample_subdir"],
        data_cfg["forecast_name"],
        input_columns,
        output_columns,
        input_rows,
        output_rows,
        "test_files.txt",
        split_source_dir=split_base_dir,
        split_files_override=model_split_files,
        fault_tolerant=split_fault_tolerant,
        input_aggregation=input_aggregation,
        drop_report=test_drop_report,
    )
    if collapse_mc_for_eval:
        print(
            "[MC-POLICY] collapse_mc_replicates_for_eval=True "
            f"(test split unique samples: {len(model_split_files)})"
        )
    test_dataset = TimeSeriesTargetDataset(test_samples)
    train_samples = None
    train_split_files = None
    if model_type == "gp_regressor":
        train_samples = load_split_samples(
            data_cfg["data_dir"],
            data_cfg["sample_subdir"],
            data_cfg["forecast_name"],
            input_columns,
            output_columns,
            input_rows,
            output_rows,
            "train_files.txt",
            split_source_dir=split_base_dir,
            split_files_override=(
                _dedupe_split_files_by_base_sample(_read_split_files(split_base_dir, "train_files.txt"))
                if collapse_mc_for_eval
                else None
            ),
            fault_tolerant=split_fault_tolerant,
            input_aggregation=input_aggregation,
            drop_report=train_drop_report,
        )
        train_split_files = _read_split_files(split_base_dir, "train_files.txt")
        if collapse_mc_for_eval:
            train_split_files = _dedupe_split_files_by_base_sample(train_split_files)
    elif model_type in ("xgb_regressor", "transformer", "xgb_classifier"):
        # For these, optionally load train samples if evaluate_all is set
        if eval_cfg.get("evaluate_all", False):
            train_samples = load_split_samples(
                data_cfg["data_dir"],
                data_cfg["sample_subdir"],
                data_cfg["forecast_name"],
                input_columns,
                output_columns,
                input_rows,
                output_rows,
                "train_files.txt",
                split_source_dir=split_base_dir,
                split_files_override=(
                    _dedupe_split_files_by_base_sample(_read_split_files(split_base_dir, "train_files.txt"))
                    if collapse_mc_for_eval
                    else None
                ),
                fault_tolerant=split_fault_tolerant,
                input_aggregation=input_aggregation,
                drop_report=train_drop_report,
            )
            train_split_files = _read_split_files(split_base_dir, "train_files.txt")
            if collapse_mc_for_eval:
                train_split_files = _dedupe_split_files_by_base_sample(train_split_files)

    # If evaluate_all is true, combine train and test samples for evaluation, but keep track of which is which
    if eval_cfg.get("evaluate_all", False) and train_samples is not None:
        all_samples = list(train_samples) + list(test_samples)
        all_split_files = (train_split_files or []) + model_split_files
        all_labels = ["train"] * len(train_samples) + ["test"] * len(test_samples)
        eval_samples = all_samples
        eval_split_files = all_split_files
        eval_labels = all_labels
    else:
        eval_samples = test_samples
        eval_split_files = model_split_files
        eval_labels = ["test"] * len(test_samples)

    # Use eval_samples for evaluation below
    # For plotting, pass eval_labels to visualizer and uncertainty boxplot

    # Prepare input arrays differently for transformer vs other models
    if model_type == "transformer":
        X_test = np.array([s[0] for s in test_samples])  # shape: (n_samples, seq_len, n_features)
        y_test = np.array([s[1] for s in test_samples])
        input_dim = X_test.shape[2] if X_test.ndim == 3 else 0
        output_dim = y_test.shape[1] if y_test.ndim > 1 else 1
        target_dim = int(y_test.shape[1]) if y_test.ndim > 1 else (1 if len(y_test) > 0 else 0)
        # Also handle X_train and X_all for transformer
        if eval_cfg.get("evaluate_all", False) and train_samples is not None:
            X_train = np.array([s[0] for s in train_samples])
            y_train = np.array([s[1] for s in train_samples])
            X_all = np.concatenate([X_train, X_test], axis=0)
            y_all = np.concatenate([y_train, y_test], axis=0)
        else:
            X_train = None
            y_train = None
            X_all = X_test
            y_all = y_test
    else:
        X_test = np.array([s[0].flatten() for s in test_samples])
        y_test = np.array([s[1].flatten() for s in test_samples])
        input_dim = int(X_test.shape[1]) if X_test.ndim > 1 else (int(len(X_test[0])) if len(X_test) > 0 else 0)
        output_dim = y_test.shape[1] if y_test.ndim > 1 else 1
        target_dim = int(y_test.shape[1]) if y_test.ndim > 1 else (1 if len(y_test) > 0 else 0)
        if eval_cfg.get("evaluate_all", False) and train_samples is not None:
            X_train = np.array([s[0].flatten() for s in train_samples])
            y_train = np.array([s[1].flatten() for s in train_samples])
            X_all = np.concatenate([X_train, X_test], axis=0)
            y_all = np.concatenate([y_train, y_test], axis=0)
        else:
            X_train = None
            y_train = None
            X_all = X_test
            y_all = y_test

    model = load_model(model_type, data_cfg, split_cfg, model_name, model_config, device, train_samples, config_dir)

    regression_pairs = []
    regression_labels = []
    regression_split_files = []
    model_regression_pair = None
    summary_rows = []
    baseline_split_files = []
    per_set_metrics = []
    predictions_entries = []
    model_column_label = None

    # --- Regression metrics for train/test and optional combined set ---
    # Combined metrics are emitted only when evaluate_all truly evaluates train+test.
    if eval_cfg.get("run_regression", True):
        include_combined_metrics = bool(eval_cfg.get("evaluate_all", False) and train_samples is not None)
        model_column_label = _model_label(model_type)
        preds_train = None
        preds_test = None
        preds_all = None
        gp_var_train = gp_var_test = gp_var_all = None
        if model_type == "gp_regressor":
            if X_train is not None:
                _gp_train = _predict_gp_bundle(model, X_train, device)
                preds_train = _gp_train["mean"]
                gp_var_train = _gp_train["variance"]
            _gp_test = _predict_gp_bundle(model, X_test, device)
            preds_test = _gp_test["mean"]
            gp_var_test = _gp_test["variance"]
            if include_combined_metrics:
                _gp_all = _predict_gp_bundle(model, X_all, device)
                preds_all = _gp_all["mean"]
                gp_var_all = _gp_all["variance"]
        elif model_type == "transformer":
            if X_train is not None:
                preds_train = model(torch.tensor(X_train, dtype=torch.float32, device=device)).detach().cpu().numpy()
            preds_test = model(torch.tensor(X_test, dtype=torch.float32, device=device)).detach().cpu().numpy()
            if include_combined_metrics:
                preds_all = model(torch.tensor(X_all, dtype=torch.float32, device=device)).detach().cpu().numpy()
        elif model_type == "xgb_regressor":
            if X_train is not None:
                preds_train = model.predict(X_train).reshape(-1, y_train.shape[1] if y_train.ndim > 1 else 1)
            preds_test = model.predict(X_test).reshape(-1, y_test.shape[1] if y_test.ndim > 1 else 1)
            if include_combined_metrics:
                preds_all = model.predict(X_all).reshape(-1, y_all.shape[1] if y_all.ndim > 1 else 1)
        # Constrain to the target's support before anything consumes the
        # predictions, so the metrics, the written predictions.csv and the
        # selection that reads them all describe the same numbers.
        preds_train, n_clipped_train = _clip_to_target_support(
            preds_train, f"{_model_label(model_type)} (train)")
        preds_test, n_clipped_test = _clip_to_target_support(
            preds_test, f"{_model_label(model_type)} (test)")
        preds_all, n_clipped_all = _clip_to_target_support(
            preds_all, f"{_model_label(model_type)} (combined)")

        # Compute metrics for each set
        if preds_train is not None:
            row_train = _compute_regression_summary(
                f"{_model_label(model_type)} (train)",
                preds_train,
                y_train,
                len(y_train),
                metadata={"kind": "train", "gp_uncertainty_mode": gp_uncertainty_mode,
                          "n_predictions_clipped": n_clipped_train,
                          **_split_size_metadata(train_samples, test_samples,
                                                 train_drop_report, test_drop_report)},
                split_files=(train_split_files or []),
            )
            per_set_metrics.append(row_train)
            predictions_entries.append(
                {
                    "kind": "train",
                    "label": model_column_label,
                    "preds": preds_train,
                    "targets": y_train,
                    "split_files": (train_split_files or []),
                    "include_mc_stats": include_mc_stats_in_predictions,
                    "gp_var": gp_var_train,
                }
            )
        if preds_test is not None:
            row_test = _compute_regression_summary(
                f"{_model_label(model_type)} (test)",
                preds_test,
                y_test,
                len(y_test),
                metadata={"kind": "test", "gp_uncertainty_mode": gp_uncertainty_mode,
                          "n_predictions_clipped": n_clipped_test,
                          **_split_size_metadata(train_samples, test_samples,
                                                 train_drop_report, test_drop_report)},
                split_files=model_split_files,
            )
            per_set_metrics.append(row_test)
            predictions_entries.append(
                {
                    "kind": "test",
                    "label": model_column_label,
                    "preds": preds_test,
                    "targets": y_test,
                    "split_files": model_split_files,
                    "include_mc_stats": include_mc_stats_in_predictions,
                    "gp_var": gp_var_test,
                }
            )
        if preds_all is not None:
            row_all = _compute_regression_summary(
                f"{_model_label(model_type)} (combined)",
                preds_all,
                y_all,
                len(y_all),
                metadata={"kind": "combined", "gp_uncertainty_mode": gp_uncertainty_mode,
                          "n_predictions_clipped": n_clipped_all,
                          **_split_size_metadata(train_samples, test_samples,
                                                 train_drop_report, test_drop_report)},
                split_files=eval_split_files,
            )
            per_set_metrics.append(row_all)
            predictions_entries.append(
                {
                    "kind": "combined",
                    "label": model_column_label,
                    "preds": preds_all,
                    "targets": y_all,
                    "split_files": eval_split_files,
                    "include_mc_stats": include_mc_stats_in_predictions,
                    "gp_var": gp_var_all,
                }
            )
    # Add per-set metrics to summary_rows for CSV output.
    summary_rows.extend(per_set_metrics)

    # --- Baseline metrics and plotting ---
    baseline_pairs = []
    baseline_labels = []
    baseline_split_files = []
    # Evaluate baselines on the same set as the model (test or combined)
    if eval_cfg.get("run_baselines", False):
        # Load historic data for baselines
        historic = eval_cfg["historic_path"]
        sample_subdir = data_cfg.get("sample_subdir", "samples")
        baseline_output_rows = _baseline_output_rows_start(data_cfg.get("output_rows", -1))
        secondary, baseline_window_hours = load_secondary(
            output_columns,
            int(eval_cfg.get("window_hours", 340)),
        )
        # Naive baseline
        preds_naive, targets_naive = evaluate_naive(
            eval_samples,
            historic,
            output_columns,
            data_cfg["data_dir"],
            output_rows=baseline_output_rows,
            gap_hours=int(eval_cfg.get("gap_hours", 5)),
            sample_subdir=sample_subdir
        )
        # Seasonal baseline
        preds_seasonal, targets_seasonal = evaluate_seasonal(
            eval_samples,
            historic,
            output_columns,
            data_cfg["data_dir"],
            output_rows=baseline_output_rows,
            diurnal_window=int(eval_cfg.get("diurnal_window", 2)),
            secondary=secondary,
            sample_subdir=sample_subdir
        )
        # Linear baseline
        preds_linear, targets_linear = evaluate_linear(
            data_cfg["data_dir"],
            data_cfg["forecast_name"],
            eval_samples,
            historic,
            output_columns,
            output_rows=baseline_output_rows,
            window_hours=int(baseline_window_hours),
            gap_hours=int(eval_cfg.get("gap_hours", 0)),
            debug_plot=bool(eval_cfg.get("debug_plot", False)),
            examples=int(eval_cfg.get("debug_examples", 10)),
            sample_subdir=sample_subdir
        )
        baseline_pairs = [
            (preds_naive, targets_naive),
            (preds_seasonal, targets_seasonal),
            (preds_linear, targets_linear),
        ]
        baseline_labels = ["Naive", "Seasonal", "Linear"]
        # Baselines are deterministic per independent sample; collapse MC replicates for plotting.
        baseline_plot_split_files = _dedupe_split_files_by_base_sample(eval_split_files)
        baseline_split_files = [baseline_plot_split_files] * 3
        baseline_kind = "combined" if (eval_cfg.get("evaluate_all", False) and train_samples is not None) else "test"
        # Add baseline metrics to summary_rows
        for (preds, targets), label in zip(baseline_pairs, baseline_labels):
            row = _compute_regression_summary(
                label,
                preds,
                targets,
                len(eval_samples),
                metadata={"kind": "baseline", "gp_uncertainty_mode": gp_uncertainty_mode,
                          **_split_size_metadata(train_samples, test_samples,
                                                 train_drop_report, test_drop_report)},
                split_files=eval_split_files,
            )
            summary_rows.append(row)
            predictions_entries.append(
                {
                    "kind": baseline_kind,
                    "label": label,
                    "preds": preds,
                    "targets": targets,
                    "split_files": eval_split_files,
                    "include_mc_stats": False,
                }
            )

    # --- Plotting ---
    # Always plot model and baselines together
    plot_pairs = []
    plot_labels = []
    plot_split_files = []
    # Model (split train/test if evaluate_all, else test)
    if eval_cfg.get("run_regression", True):
        if eval_cfg.get("evaluate_all", False) and X_train is not None:
            # Plot train and test as separate series
            plot_pairs.append((preds_train, y_train))
            plot_labels.append(f"{_model_label(model_type)} (train)")
            plot_split_files.append(train_split_files or [])
            plot_pairs.append((preds_test, y_test))
            plot_labels.append(f"{_model_label(model_type)} (test)")
            plot_split_files.append(model_split_files)
        else:
            plot_pairs.append((preds_test, y_test))
            plot_labels.append(_model_label(model_type))
            plot_split_files.append(model_split_files)
    # Baselines (plotted on eval_samples: combined only when evaluate_all is enabled)
    plot_pairs.extend(baseline_pairs)
    plot_labels.extend(baseline_labels)
    plot_split_files.extend(baseline_split_files)

    model_plot_count = len(plot_pairs) - len(baseline_pairs)
    collapse_error_points_by_pair = ([False] * model_plot_count) + ([True] * len(baseline_pairs))

    # Call visualizer and uncertainty plots only when plot output is enabled.
    if plot_pairs and save_plots:
        visualizer(
            *plot_pairs,
            labels=plot_labels,
            forecast_name=data_cfg["forecast_name"],
            directory=data_cfg["data_dir"],
            num_samples=eval_cfg.get("num_samples"),
            sample_labels=None,
            split_files_by_pair=plot_split_files,
            collapse_error_points_by_pair=collapse_error_points_by_pair,
        )
        model_plot_pairs = plot_pairs[:model_plot_count]
        model_plot_labels = plot_labels[:model_plot_count]
        model_plot_split_files = plot_split_files[:model_plot_count]

        has_model_mc_distribution = _has_mc_replicate_distribution_for_uncertainty_plot(
            model_plot_pairs,
            model_plot_split_files,
            row_limit=eval_cfg.get("num_samples"),
        )
        # Keep uncertainty boxplots model-only and only when replicate distributions exist.
        # This matches predictions.csv semantics where mc_xxx columns appear only with MC replicate evaluations.
        if has_model_mc_distribution:
            _plot_uncertainty_boxplots(
                model_plot_pairs,
                model_plot_labels,
                model_plot_split_files,
                directory=data_cfg["data_dir"],
                forecast_name=data_cfg["forecast_name"],
                num_samples=eval_cfg.get("num_samples"),
                sample_labels=None,
            )
        else:
            print(
                "[PLOT-POLICY] Skipping model uncertainty boxplots: "
                "no MC replicate distributions were evaluated."
            )

    # Write summary CSV for regression/classification results.
    # Model rows include test always, train optionally, and combined only when evaluate_all=true.
    # Output path logic: forecasts/<forecast_name>/evaluation_summary.csv
    summary_dir = Path(data_cfg["data_dir"]) / "forecasts" / data_cfg["forecast_name"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "evaluation_summary.csv"
    _write_summary_csv(summary_rows, summary_path)

    # Write per-sample prediction table (one row per independent sample file).
    predictions_rows, predictions_columns = _build_predictions_table(
        predictions_entries,
        gp_uncertainty_mode,
        include_mc_output_columns=include_mc_stats_in_predictions,
    )
    predictions_path = summary_dir / "predictions.csv"
    _write_predictions_csv(predictions_rows, predictions_path, predictions_columns)

    if save_plots:
        boxplot_rows = _build_boxplot_error_rows_from_predictions(
            predictions_rows,
            model_label=model_column_label,
            baseline_labels=baseline_labels,
        )
        boxplot_from_error_rows(
            boxplot_rows,
            directory=data_cfg["data_dir"],
            forecast_name=data_cfg["forecast_name"],
        )

    # Return the main model summary row (prefer test, then combined, then train)
    # Look for kind == 'test', else 'combined', else 'train', else first row
    main_row = None
    for row in summary_rows:
        if str(row.get("kind", "")).lower() == "test":
            main_row = row
            break
    if main_row is None:
        for row in summary_rows:
            if str(row.get("kind", "")).lower() == "combined":
                main_row = row
                break
    if main_row is None:
        for row in summary_rows:
            if str(row.get("kind", "")).lower() == "train":
                main_row = row
                break
    if main_row is None and summary_rows:
        main_row = summary_rows[0]
    return main_row


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="One or more YAML/JSON config paths on a single command line (supports comma-separated values and glob patterns)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation for this invocation only (does not modify config files).",
    )
    args = parser.parse_args()

    config_paths = _expand_config_inputs(args.config)
    if not config_paths:
        raise ValueError(
            "No valid config files found after expanding --config arguments. "
            "Use .yml/.yaml/.json files on a single command line."
        )

    save_plots_override = False if args.no_plots else True
    for config_path in config_paths:
        evaluate_single_config(config_path, save_plots_override=save_plots_override)


if __name__ == "__main__":
    main()

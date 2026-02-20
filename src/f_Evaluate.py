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

import yaml
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import torch
import xgboost as xgb
import gpytorch

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
)


DEFAULT_EVAL_CONFIG = {
    "run_regression": True,
    "run_threshold_classification": False,
    "run_pure_classification": False,
    "run_baselines": False,
    "num_samples": 200,
    "debug_plot": False,
    "debug_examples": 10,
    "gap_hours": 1,
    "window_hours": 550,
    "diurnal_window": 1,
    "historic_path": "../data/output/regression/Consolidated_sparse.csv",
    "thresholds_path": "../data/input/Limits.csv",
    "normalization_path": "../data/input/normalization.json",
    "use_normalized_thresholds": False,
    "baseline_sample_subdir": "samples",
    "baseline_split_file": "test_files.txt",
    "baseline_split_source": None,
    "baseline_match_mc_to_raw": True,
    "save_plots": True,
    "evaluate_all": False,  # If true, combine train and test samples for evaluation
}


def load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    def _read_text_with_fallback(path_obj):
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path_obj, "r", encoding="cp1252") as f:
                return f.read()

    raw_text = _read_text_with_fallback(path)

    if path.suffix in [".yaml", ".yml"]:
        config = yaml.safe_load(raw_text)
        config["__config_dir"] = str(path.resolve().parent)
        return config
    if path.suffix == ".json":
        config = json.loads(raw_text)
        config["__config_dir"] = str(path.resolve().parent)
        return config

    raise ValueError(f"Unsupported config file format: {path.suffix}")


def _resolve_path_from_config(path_value, config_dir):
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (Path(config_dir) / path_obj).resolve()


def _resolve_data_paths(data_cfg, config_dir):
    configured_subdir = data_cfg.get("sample_subdir")
    data_dir_path = _resolve_path_from_config(data_cfg["data_dir"], config_dir)

    if configured_subdir:
        return str(data_dir_path), configured_subdir

    if data_dir_path.name in {"samples", "mc_replicates"}:
        return str(data_dir_path.parent), data_dir_path.name

    return str(data_dir_path), "samples"


def merge_eval_config(cfg):
    eval_cfg = cfg.get("evaluation", {})
    merged = DEFAULT_EVAL_CONFIG.copy()
    merged.update(eval_cfg)
    return merged


def load_model_config(data_dir, forecast_name, model_name, fallback_data=None):
    config_path = Path(data_dir, "forecasts", forecast_name, model_name, "model_config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)

    if fallback_data is None:
        raise FileNotFoundError(f"model_config.json not found at {config_path}")

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
    return load_split_samples(
        data_dir,
        sample_subdir,
        forecast_name,
        input_columns,
        output_columns,
        input_rows,
        output_rows,
        "test_files.txt",
        input_aggregation=input_aggregation,
    )


def _read_split_files(split_source_dir, split_file):
    split_path = Path(split_source_dir, split_file)
    with open(split_path) as f:
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
    input_aggregation="none",
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
        fault_tolerant=True,
        input_aggregation=input_aggregation,
    )

    return samples


def get_output_dim(data_dir, sample_subdir, output_columns, output_rows):
    sample_files = sorted(os.listdir(Path(data_dir, sample_subdir)))
    sample_df = pd.read_csv(Path(data_dir, sample_subdir, sample_files[0]))
    return len(output_columns) * len(sample_df.iloc[output_rows:])


def _canonical_feature_name(name):
    text = str(name).strip().lower().replace("µ", "u")
    text = text.replace("micro", "u")
    text = text.replace("_", " ")
    if " - " in text:
        text = text.split(" - ", 1)[1].strip()
    for token in ["(", ")", "/", "%", "°", "-", ".", ","]:
        text = text.replace(token, " ")
    return " ".join(text.split())


def _resolve_summary_dir(hyper_cfg, config_dir):
    if hyper_cfg.get("uncertainty_summary_dir"):
        return _resolve_path_from_config(hyper_cfg["uncertainty_summary_dir"], config_dir)
    return Path(__file__).parent.parent / "data" / "output" / "calibration" / "summaries"


def _load_uncertainty_std_map(summary_dir):
    if not summary_dir.exists():
        return {}

    summary_map = {}
    for file_path in summary_dir.rglob("*_uncertainty_summary.csv"):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            row = df.iloc[0]
            sensor_name = row.get("Sensor")
            if pd.isna(sensor_name):
                continue
            offset_std = row.get("Offset_Std", 0.0)
            if pd.isna(offset_std):
                offset_std = 0.0
            summary_map[_canonical_feature_name(sensor_name)] = float(offset_std)
        except Exception:
            continue
    return summary_map


def _build_feature_uncertainty_variance(data_cfg, hyper_cfg, config_dir):
    input_columns = data_cfg["input_columns"]
    seq_len = data_cfg["input_row_2"] - data_cfg["input_row_1"]

    summary_std_map = _load_uncertainty_std_map(_resolve_summary_dir(hyper_cfg, config_dir))

    norm_path = Path(data_cfg["data_dir"]) / "normalization.json"
    norm_params = {}
    if norm_path.exists():
        try:
            with open(norm_path, "r") as f:
                norm_params = json.load(f)
        except Exception:
            norm_params = {}

    feature_variances = []
    for feature in input_columns:
        candidates = [_canonical_feature_name(feature)]
        if " - " in feature:
            candidates.append(_canonical_feature_name(feature.split(" - ", 1)[1]))

        matched_std = None
        for candidate in candidates:
            if candidate in summary_std_map:
                matched_std = summary_std_map[candidate]
                break

        if matched_std is None:
            matched_std = 0.0

        if matched_std > 0 and feature in norm_params:
            v_min = norm_params[feature].get("min", 0)
            v_max = norm_params[feature].get("max", 1)
            v_range = v_max - v_min
            if v_range not in [0, 0.0]:
                matched_std = matched_std / v_range

        feature_variances.append(float(matched_std ** 2))

    return np.tile(np.array(feature_variances, dtype=np.float32), seq_len)


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
    model_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"], model_name)
    artifact = torch.load(model_path / "gp_model.pt", map_location=device, weights_only=False)
    hyper_cfg = artifact["hyperparameters"]

    X_train_np, y_train_np = _prepare_gp_train_arrays(train_samples, split_cfg, hyper_cfg)

    x_mean = np.array(artifact["input_mean"], dtype=np.float32)
    x_std = np.array(artifact["input_std"], dtype=np.float32)
    x_std[x_std < 1e-8] = 1.0

    if hyper_cfg.get("input_standardize", True):
        X_train_used = (X_train_np - x_mean) / x_std
    else:
        X_train_used = X_train_np

    X_train = torch.tensor(X_train_used, dtype=torch.float32, device=device)
    kernel_name = str(hyper_cfg.get("kernel", "matern52")).lower()
    use_uncertain_kernel = bool(hyper_cfg.get("use_uncertain_input_kernel", True))
    ard_dims = X_train.shape[1] if (hyper_cfg.get("ard", True) or use_uncertain_kernel) else None

    input_uncertainty_var = None
    if use_uncertain_kernel:
        input_uncertainty_var = torch.tensor(
            _build_feature_uncertainty_variance(data_cfg, hyper_cfg, config_dir), dtype=torch.float32, device=device
        )

    class UncertainInputRBFKernel(gpytorch.kernels.Kernel):
        has_lengthscale = True

        def __init__(self, input_variance, **kwargs):
            super().__init__(**kwargs)
            self.register_buffer("input_variance", input_variance)

        def forward(self, x1, x2, diag=False, **params):
            if diag:
                return torch.ones(x1.shape[-2], device=x1.device, dtype=x1.dtype)

            lengthscale = self.lengthscale.squeeze()
            if lengthscale.dim() == 0:
                lengthscale = lengthscale.repeat(x1.shape[-1])

            ls2 = lengthscale.pow(2)
            denom = torch.clamp(ls2 + 2.0 * self.input_variance, min=1e-10)
            sq_dist = ((x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(2) / denom).sum(dim=-1)
            det_term = torch.sqrt(torch.prod(ls2 / denom))
            return det_term * torch.exp(-0.5 * sq_dist)

    def build_base_kernel():
        if use_uncertain_kernel:
            return UncertainInputRBFKernel(input_variance=input_uncertainty_var, ard_num_dims=ard_dims)
        if kernel_name == "rbf":
            return gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        if kernel_name == "matern32":
            return gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_dims)
        return gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)

    class ExactGPRegressor(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(build_base_kernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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
        model = ExactGPRegressor(X_train, y_train, likelihood).to(device)
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
    hyper_cfg = gp_bundle["hyperparameters"]
    input_mean = gp_bundle["input_mean"]
    input_std = gp_bundle["input_std"]

    if hyper_cfg.get("input_standardize", True):
        X_used = (X_np - input_mean) / input_std
    else:
        X_used = X_np

    X_tensor = torch.tensor(X_used, dtype=torch.float32, device=device)
    preds = []
    for entry in gp_bundle["models"]:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = entry["likelihood"](entry["model"](X_tensor))
            pred_mean = pred_dist.mean.detach().cpu().numpy()
        pred_mean = pred_mean * entry["target_std"] + entry["target_mean"]
        preds.append(pred_mean)

    return np.stack(preds, axis=1)


def load_model(model_type, data_cfg, split_cfg, model_name, model_config, device, train_samples=None, config_dir=None):
    model_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"], model_name)

    if model_type == "transformer":
        model = TimeSeriesTransformer(model_config).to(device)
        model.load_state_dict(torch.load(model_path / "transformer_model.pt", map_location=device))
        model.eval()
        return model

    if model_type == "gp_regressor":
        if train_samples is None:
            raise ValueError("train_samples are required to evaluate gp_regressor")
        return _load_gp_bundle(data_cfg, split_cfg, model_name, train_samples, device, config_dir)

    if model_type == "xgb_regressor":
        model = xgb.XGBRegressor()
        model.load_model(model_path / "xgboost_model.json")
        return model

    if model_type == "xgb_classifier":
        model = xgb.XGBClassifier()
        model.load_model(model_path / "xgboost_model.json")
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


def load_thresholds(eval_cfg):
    thresholds_df = pd.read_csv(Path(eval_cfg["thresholds_path"]), sep=";", decimal=".")
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

        # Set a default box_width before any use
        box_width = 0.05
        if not group_order or n_cols == 0:
            print(f"[INFO] Skipping uncertainty boxplot for {label}: no aligned grouped samples.")
            continue

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

            # If not using sample_labels, plot all at once
            if sample_labels is None or len(box_data) > 0:
                if len(box_data) > 0:
                    ax.boxplot(box_data, positions=positions, widths=box_width, showfliers=False)

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
                if diag_max > diag_min:
                    global_min = min(global_min, diag_min)
                    global_max = max(global_max, diag_max)

            ax.set_ylabel("Prediction")
            ax.set_xlabel("Ground truth")
            ax.grid(alpha=0.25)
            if out_idx == 0:
                ax.set_title(
                    f"Prediction uncertainty (x=ground truth, y=prediction replicate distribution) — {label}\n"
                    f"n_groups={len(group_order)}"
                )
            ax.text(0.01, 0.98, f"Output {out_idx + 1}", transform=ax.transAxes, ha="left", va="top", fontsize=9)

        if valid_axes == 0:
            plt.close(fig)
            print(f"[INFO] Skipping uncertainty boxplot for {label}: no finite grouped values.")
            continue

        if np.isfinite(global_min) and np.isfinite(global_max) and global_max > global_min:
            pad = 0.03 * (global_max - global_min)
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

        if n_cols > max_outputs:
            fig.text(0.99, 0.01, f"Showing first {max_outputs}/{n_cols} outputs", ha="right", va="bottom", fontsize=8)

        out_path = base_dir / f"predictions_uncertainty_boxplot_{_sanitize_label_for_filename(label)}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"[INFO] Wrote uncertainty boxplot: {out_path}")


def _compute_regression_summary(label, preds, targets, num_samples, metadata=None):
    metadata = metadata or {}
    pred_arr, target_arr, n_rows, n_cols = _aligned_arrays(preds, targets, row_limit=num_samples)

    pred_flat = pred_arr.reshape(-1)
    target_flat = target_arr.reshape(-1)
    finite_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
    finite_count = int(np.sum(finite_mask))

    if finite_count > 0:
        errors = pred_flat[finite_mask] - target_flat[finite_mask]
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        if finite_count > 1:
            target_vals = target_flat[finite_mask]
            ss_res = float(np.sum(np.square(errors)))
            ss_tot = float(np.sum(np.square(target_vals - np.mean(target_vals))))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        else:
            r2 = np.nan
    else:
        mae = np.nan
        rmse = np.nan
        r2 = np.nan

    row = {
        "label": label,
        "n_eval_rows": int(n_rows),
        "n_eval_outputs": int(n_cols),
        "n_eval_points_finite": finite_count,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
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
    }
    row.update(metadata)
    return row


def _write_summary_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = [
        "label",
        "mae",
        "rmse",
        "r2",
        "kind",
        "n_train_samples",
        "n_test_samples",
        "n_eval_rows",
        "n_eval_outputs",
        "n_eval_points_finite",
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

    input_columns = model_config["input_columns"]
    output_columns = model_config["output_columns"]
    input_rows = slice(model_config["input_row_1"], model_config["input_row_2"])
    output_rows = model_config["output_rows"]
    input_aggregation = str(model_config.get("input_aggregation", data_cfg.get("input_aggregation", "none"))).lower()


    # Load test and train samples
    test_samples = load_test_samples(
        data_cfg["data_dir"],
        data_cfg["sample_subdir"],
        data_cfg["forecast_name"],
        input_columns,
        output_columns,
        input_rows,
        output_rows,
        input_aggregation=input_aggregation,
    )
    model_split_files = _read_split_files(
        Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"]),
        "test_files.txt",
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
            input_aggregation=input_aggregation,
        )
        train_split_files = _read_split_files(
            Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"]),
            "train_files.txt",
        )
    elif model_type in ("xgb_regressor", "transformer", "xgb_classifier"):
        # For these, optionally load train samples if evaluate_all is set
        if eval_cfg.get("evaluate_all", True):
            train_samples = load_split_samples(
                data_cfg["data_dir"],
                data_cfg["sample_subdir"],
                data_cfg["forecast_name"],
                input_columns,
                output_columns,
                input_rows,
                output_rows,
                "train_files.txt",
                input_aggregation=input_aggregation,
            )
            train_split_files = _read_split_files(
                Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"]),
                "train_files.txt",
            )

    # If evaluate_all is true, combine train and test samples for evaluation, but keep track of which is which
    if eval_cfg.get("evaluate_all", True) and train_samples is not None:
        all_samples = list(train_samples) + list(test_samples)
        all_split_files = (train_split_files or []) + model_split_files
        all_labels = ["train"] * len(train_samples) + ["test"] * len(test_samples)
        eval_samples = all_samples
        eval_split_files = all_split_files
        eval_labels = all_labels
        # Prepare arrays for per-set metrics
        X_train = np.array([s[0].flatten() for s in train_samples])
        y_train = np.array([s[1].flatten() for s in train_samples])
        X_test = np.array([s[0].flatten() for s in test_samples])
        y_test = np.array([s[1].flatten() for s in test_samples])
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
    else:
        eval_samples = test_samples
        eval_split_files = model_split_files
        eval_labels = ["test"] * len(test_samples)
        X_train = None
        y_train = None
        X_test = np.array([s[0].flatten() for s in test_samples])
        y_test = np.array([s[1].flatten() for s in test_samples])
        X_all = X_test
        y_all = y_test

    # Use eval_samples for evaluation below
    # For plotting, pass eval_labels to visualizer and uncertainty boxplot

    X_test = np.array([s[0].flatten() for s in test_samples])
    y_test = np.array([s[1].flatten() for s in test_samples])
    output_dim = y_test.shape[1] if y_test.ndim > 1 else 1
    input_dim = int(X_test.shape[1]) if X_test.ndim > 1 else (int(len(X_test[0])) if len(X_test) > 0 else 0)
    target_dim = int(y_test.shape[1]) if y_test.ndim > 1 else (1 if len(y_test) > 0 else 0)

    split_cfg = config.get("data_split", {"random_state": 42})
    model = load_model(model_type, data_cfg, split_cfg, model_name, model_config, device, train_samples, config_dir)

    regression_pairs = []
    regression_labels = []
    regression_split_files = []
    model_regression_pair = None
    summary_rows = []
    baseline_split_files = []
    per_set_metrics = []

    # --- LOOCV logic controlled by config flag ---
    def run_loocv_mc_group():
        """
        Perform leave-one-out cross-validation at the MC group (raw segment) level.
        For each unique group (raw segment), leave out all samples for that group, train on the rest, predict on the left-out group.
        Write results to a summary CSV in the forecast directory.
        """
        from utils.training import group_samples_by_segment
        import copy

        # Group test samples by raw segment
        group_map = group_samples_by_segment(test_samples, model_split_files)
        group_ids = list(group_map.keys())
        loocv_preds = []
        loocv_targets = []
        loocv_group_ids = []

        for left_out_group in group_ids:
            # Prepare train/test split for this fold
            test_idx = group_map[left_out_group]
            train_idx = [i for g, idxs in group_map.items() if g != left_out_group for i in idxs]
            if not train_idx or not test_idx:
                continue
            train_samples_fold = [test_samples[i] for i in train_idx]
            test_samples_fold = [test_samples[i] for i in test_idx]

            # Retrain model on train_samples_fold
            # Only supporting xgb_regressor and gp_regressor for now
            if model_type == "xgb_regressor":
                X_train = np.array([s[0].flatten() for s in train_samples_fold])
                y_train = np.array([s[1].flatten() for s in train_samples_fold])
                xgb_model = xgb.XGBRegressor()
                xgb_model.fit(X_train, y_train)
                X_test_fold = np.array([s[0].flatten() for s in test_samples_fold])
                preds = xgb_model.predict(X_test_fold)
                if np.ndim(preds) == 1:
                    preds = preds.reshape(-1, 1)
                targets = np.array([s[1].flatten() for s in test_samples_fold])
            elif model_type == "gp_regressor":
                # Use the same logic as in _prepare_gp_train_arrays and _load_gp_bundle
                X_train = np.array([s[0].flatten() for s in train_samples_fold], dtype=np.float32)
                y_train = np.array([s[1].flatten() for s in train_samples_fold], dtype=np.float32)
                # Use the same model config as before
                # For simplicity, use the first output only if multi-output
                y_train_col = y_train[:, 0] if y_train.ndim > 1 else y_train
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
                class DummyGP(gpytorch.models.ExactGP):
                    def __init__(self, train_x, train_y, likelihood):
                        super().__init__(train_x, train_y, likelihood)
                        self.mean_module = gpytorch.means.ConstantMean()
                        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                    def forward(self, x):
                        mean_x = self.mean_module(x)
                        covar_x = self.covar_module(x)
                        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
                y_train_tensor = torch.tensor(y_train_col, dtype=torch.float32, device=device)
                gp_model = DummyGP(X_train_tensor, y_train_tensor, likelihood).to(device)
                gp_model.train()
                likelihood.train()
                optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
                for _ in range(30):
                    optimizer.zero_grad()
                    output = gp_model(X_train_tensor)
                    loss = -mll(output, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                gp_model.eval()
                likelihood.eval()
                X_test_fold = np.array([s[0].flatten() for s in test_samples_fold], dtype=np.float32)
                X_test_tensor = torch.tensor(X_test_fold, dtype=torch.float32, device=device)
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred_dist = likelihood(gp_model(X_test_tensor))
                    preds = pred_dist.mean.cpu().numpy().reshape(-1, 1)
                targets = np.array([s[1].flatten() for s in test_samples_fold])
            else:
                print(f"[WARN] LOOCV not implemented for model_type={model_type}")
                continue

            loocv_preds.append(preds)
            loocv_targets.append(targets)
            loocv_group_ids.extend([left_out_group] * len(preds))

        # Concatenate all predictions and targets
        if loocv_preds:
            all_preds = np.vstack(loocv_preds)
            all_targets = np.vstack(loocv_targets)
            # Write summary CSV
            summary = _compute_regression_summary(
                f"LOOCV_{model_type}",
                all_preds,
                all_targets,
                num_samples=len(all_preds),
                metadata={
                    "kind": "loocv",
                    "data_dir": str(data_cfg["data_dir"]),
                    "n_test_samples": len(all_preds),
                    "input_dim": input_dim,
                    "target_dim": target_dim,
                },
            )
            summary_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"], "loocv_summary.csv")
            _write_summary_csv([summary], summary_path)
            print(f"[INFO] LOOCV summary written to {summary_path}")
        else:
            print("[WARN] No LOOCV predictions generated.")

    # --- End LOOCV logic ---

    # Run LOOCV if requested in config
    if eval_cfg.get("run_loocv", False):
        run_loocv_mc_group()

    # --- Regression metrics for train, test, and combined sets ---
    # Only for regression tasks
    if eval_cfg.get("run_regression", True):
        preds_train = None
        preds_test = None
        preds_all = None
        if model_type == "gp_regressor":
            if X_train is not None:
                preds_train = _predict_gp_bundle(model, X_train, device)
            preds_test = _predict_gp_bundle(model, X_test, device)
            preds_all = _predict_gp_bundle(model, X_all, device)
        elif model_type == "transformer":
            if X_train is not None:
                preds_train = model(torch.tensor(X_train, dtype=torch.float32, device=device)).detach().cpu().numpy()
            preds_test = model(torch.tensor(X_test, dtype=torch.float32, device=device)).detach().cpu().numpy()
            preds_all = model(torch.tensor(X_all, dtype=torch.float32, device=device)).detach().cpu().numpy()
        elif model_type == "xgb_regressor":
            if X_train is not None:
                preds_train = model.predict(X_train).reshape(-1, y_train.shape[1] if y_train.ndim > 1 else 1)
            preds_test = model.predict(X_test).reshape(-1, y_test.shape[1] if y_test.ndim > 1 else 1)
            preds_all = model.predict(X_all).reshape(-1, y_all.shape[1] if y_all.ndim > 1 else 1)
        # Compute metrics for each set
        if preds_train is not None:
            row_train = _compute_regression_summary(f"{_model_label(model_type)} (train)", preds_train, y_train, len(y_train), metadata={"kind": "train"})
            per_set_metrics.append(row_train)
        if preds_test is not None:
            row_test = _compute_regression_summary(f"{_model_label(model_type)} (test)", preds_test, y_test, len(y_test), metadata={"kind": "test"})
            per_set_metrics.append(row_test)
        if preds_all is not None:
            row_all = _compute_regression_summary(f"{_model_label(model_type)} (combined)", preds_all, y_all, len(y_all), metadata={"kind": "combined"})
            per_set_metrics.append(row_all)
    # Add per-set metrics to summary_rows for CSV output
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
        # Naive baseline
        preds_naive, targets_naive = evaluate_naive(
            eval_samples,
            historic,
            output_columns,
            data_cfg["data_dir"],
            sample_subdir=sample_subdir
        )
        # Seasonal baseline
        preds_seasonal, targets_seasonal = evaluate_seasonal(
            eval_samples,
            historic,
            output_columns,
            data_cfg["data_dir"],
            sample_subdir=sample_subdir
        )
        # Linear baseline
        preds_linear, targets_linear = evaluate_linear(
            data_cfg["data_dir"],
            data_cfg["forecast_name"],
            eval_samples,
            historic,
            output_columns,
            sample_subdir=sample_subdir
        )
        baseline_pairs = [
            (preds_naive, targets_naive),
            (preds_seasonal, targets_seasonal),
            (preds_linear, targets_linear),
        ]
        baseline_labels = ["Naive", "Seasonal", "Linear"]
        baseline_split_files = [eval_split_files] * 3
        # Add baseline metrics to summary_rows
        for (preds, targets), label in zip(baseline_pairs, baseline_labels):
            row = _compute_regression_summary(label, preds, targets, len(eval_samples), metadata={"kind": "baseline"})
            summary_rows.append(row)

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
    # Baselines (still plotted on combined set)
    plot_pairs.extend(baseline_pairs)
    plot_labels.extend(baseline_labels)
    plot_split_files.extend(baseline_split_files)

    # Call visualizer and _plot_uncertainty_boxplots
    if plot_pairs:
        visualizer(
            *plot_pairs,
            labels=plot_labels,
            forecast_name=data_cfg["forecast_name"],
            directory=data_cfg["data_dir"],
            num_samples=eval_cfg.get("num_samples", 200),
            sample_labels=None,
        )
        _plot_uncertainty_boxplots(
            plot_pairs,
            plot_labels,
            plot_split_files,
            directory=data_cfg["data_dir"],
            forecast_name=data_cfg["forecast_name"],
            num_samples=eval_cfg.get("num_samples", 200),
            sample_labels=None,
        )

    # Write summary CSV for regression/classification results (train/test/combined)
    # Output path logic: forecasts/<forecast_name>/evaluation_summary.csv
    summary_dir = Path(data_cfg["data_dir"]) / "forecasts" / data_cfg["forecast_name"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "evaluation_summary.csv"
    _write_summary_csv(summary_rows, summary_path)


def write_combined_outputs(results, save_plots=True):
    grouped = {}
    for result in results:
        grouped.setdefault(result["data_dir"], []).append(result)

    for data_dir, group in grouped.items():
        combined_pairs = []
        combined_labels = []
        combined_split_files = []
        combined_meta = []
        combined_rows = []

        for result in group:
            if result["model_pair"] is not None:
                combined_pairs.append(result["model_pair"])
                combined_labels.append(result["model_label"])
                combined_split_files.append(result.get("model_split_files", []))
                combined_meta.append(
                    {
                        "data_dir": result["data_dir"],
                        "kind": "model",
                        "n_train_samples": result["n_train_samples"],
                        "n_test_samples": result["n_test_samples"],
                        "input_dim": result["input_dim"],
                        "target_dim": result["target_dim"],
                    }
                )

        baseline_source = next((r for r in group if r["baseline_pairs"]), None)
        if baseline_source is not None:
            combined_pairs.extend(baseline_source["baseline_pairs"])
            combined_labels.extend(baseline_source["baseline_labels"])
            combined_split_files.extend([baseline_source.get("baseline_split_files", [])] * len(baseline_source["baseline_labels"]))
            for label in baseline_source["baseline_labels"]:
                combined_meta.append(
                    {
                        "data_dir": baseline_source["data_dir"],
                        "kind": "baseline",
                        "n_train_samples": np.nan,
                        "n_test_samples": np.nan,
                        "input_dim": baseline_source["input_dim"],
                        "target_dim": baseline_source["target_dim"],
                    }
                )

        if len(combined_pairs) < 2:
            print(
                "[INFO] Skipping combined top-level plots for "
                f"{data_dir}; need at least two result sets, got {len(combined_pairs)}."
            )
            continue

        num_samples = max(r["num_samples"] for r in group)
        if save_plots:
            visualizer(
                *combined_pairs,
                labels=combined_labels,
                forecast_name="",
                directory=data_dir,
                num_samples=num_samples,
            )
            _plot_uncertainty_boxplots(
                combined_pairs,
                combined_labels,
                combined_split_files,
                directory=data_dir,
                forecast_name="",
                num_samples=num_samples,
            )

        for (preds, targets), label, meta in zip(combined_pairs, combined_labels, combined_meta):
            combined_rows.append(
                _compute_regression_summary(
                    label,
                    preds,
                    targets,
                    num_samples,
                    metadata=meta,
                )
            )

        _write_summary_csv(combined_rows, Path(data_dir, "forecasts", "evaluation_summary_combined.csv"))

        print(
            "[INFO] Wrote combined comparison outputs to top forecasts directory: "
            f"{Path(data_dir, 'forecasts')}"
        )


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
    results = [evaluate_single_config(config_path, save_plots_override=save_plots_override) for config_path in config_paths]

    if len(results) > 1:
        write_combined_outputs(results)


if __name__ == "__main__":
    main()

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


def load_test_samples(data_dir, sample_subdir, forecast_name, input_columns, output_columns, input_rows, output_rows):
    return load_split_samples(
        data_dir,
        sample_subdir,
        forecast_name,
        input_columns,
        output_columns,
        input_rows,
        output_rows,
        "test_files.txt",
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


def _plot_uncertainty_boxplots(regression_pairs, regression_labels, split_files_by_pair, directory, forecast_name, num_samples):
    if not regression_pairs:
        return

    base_dir = Path(directory, "forecasts", forecast_name) if forecast_name else Path(directory, "forecasts")
    base_dir.mkdir(parents=True, exist_ok=True)

    for (preds, targets), label, split_files in zip(regression_pairs, regression_labels, split_files_by_pair):
        group_order, grouped_preds, grouped_targets, n_cols = _build_replicate_groups(
            preds,
            targets,
            split_files,
            row_limit=num_samples,
        )
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

            for group_id in group_order:
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
        "n_pred_rows": int(np.array(preds).shape[0]) if np.array(preds).ndim > 0 else 0,
        "n_target_rows": int(np.array(targets).shape[0]) if np.array(targets).ndim > 0 else 0,
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
        "n_pred_rows": int(len(np.array(preds).reshape(-1))),
        "n_target_rows": int(len(np.array(targets).reshape(-1))),
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
        "n_pred_rows",
        "n_target_rows",
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


def evaluate_single_config(config_path):
    print(f"\n=== Evaluating config: {config_path} ===")

    config = load_config(config_path)
    config_dir = config["__config_dir"]

    model_type = config["model_type"]
    model_name = config["model_name"]
    data_cfg = config["data"]
    eval_cfg = merge_eval_config(config)

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

    test_samples = load_test_samples(
        data_cfg["data_dir"],
        data_cfg["sample_subdir"],
        data_cfg["forecast_name"],
        input_columns,
        output_columns,
        input_rows,
        output_rows,
    )
    model_split_files = _read_split_files(
        Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"]),
        "test_files.txt",
    )
    test_dataset = TimeSeriesTargetDataset(test_samples)
    train_samples = None
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
        )

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

    if eval_cfg["run_regression"]:
        if model_type == "transformer":
            preds, targets = evaluate_transformer(model, test_dataset, device)
            regression_pairs.append((preds, targets))
            regression_labels.append("Transformer")
            regression_split_files.append(model_split_files)
            model_regression_pair = (preds, targets)
        elif model_type == "xgb_regressor":
            preds_flat = model.predict(X_test)
            if np.ndim(preds_flat) == 1:
                preds = preds_flat.reshape(-1, 1)
            else:
                preds = np.array(preds_flat)
            targets = y_test.reshape(y_test.shape[0], -1)
            if preds.shape[1] != targets.shape[1]:
                common_dim = min(preds.shape[1], targets.shape[1])
                print(
                    f"[WARN] Prediction dim ({preds.shape[1]}) != target dim ({targets.shape[1]}). "
                    f"Evaluating first {common_dim} output(s)."
                )
                preds = preds[:, :common_dim]
                targets = targets[:, :common_dim]
            regression_pairs.append((preds, targets))
            regression_labels.append("XGBRegressor")
            regression_split_files.append(model_split_files)
            model_regression_pair = (preds, targets)
        elif model_type == "gp_regressor":
            preds = _predict_gp_bundle(model, X_test, device)
            targets = y_test.reshape(y_test.shape[0], -1)
            if preds.shape[1] != targets.shape[1]:
                common_dim = min(preds.shape[1], targets.shape[1])
                print(
                    f"[WARN] Prediction dim ({preds.shape[1]}) != target dim ({targets.shape[1]}). "
                    f"Evaluating first {common_dim} output(s)."
                )
                preds = preds[:, :common_dim]
                targets = targets[:, :common_dim]
            regression_pairs.append((preds, targets))
            regression_labels.append("GPRegressor")
            regression_split_files.append(model_split_files)
            model_regression_pair = (preds, targets)
        elif model_type == "xgb_classifier":
            print("Skipping regression evaluation for xgb_classifier model_type")

    baseline_pairs = []
    baseline_labels = []
    baseline_test_samples = []

    if eval_cfg["run_baselines"]:
        is_mc_trained_model = data_cfg["sample_subdir"] == "mc_replicates"
        baseline_sample_subdir = eval_cfg.get("baseline_sample_subdir") or data_cfg["sample_subdir"]
        if is_mc_trained_model and baseline_sample_subdir != "samples":
            print(
                "[INFO] MC-trained model detected; forcing baseline_sample_subdir='samples' "
                "to evaluate on unique raw segments only."
            )
            baseline_sample_subdir = "samples"
        baseline_split_file = eval_cfg.get("baseline_split_file", "test_files.txt")
        baseline_split_source = eval_cfg.get("baseline_split_source") or str(
            Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
        )
        baseline_split_files = None

        baseline_split_path = Path(baseline_split_source, baseline_split_file)
        if baseline_split_path.exists():
            baseline_split_files = _read_split_files(baseline_split_source, baseline_split_file)
            if (
                eval_cfg.get("baseline_match_mc_to_raw", True)
                and is_mc_trained_model
                and baseline_sample_subdir == "samples"
                and any("_mc_" in name for name in baseline_split_files)
            ):
                original_count = len(baseline_split_files)
                baseline_split_files = _map_split_files_mc_to_raw(baseline_split_files)
                print(
                    "[INFO] Baseline split entries mapped from MC replicates to raw samples "
                    f"({original_count} -> {len(baseline_split_files)} files)."
                )
        else:
            model_split_files = _read_split_files(
                Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"]), "test_files.txt"
            )
            if (
                eval_cfg.get("baseline_match_mc_to_raw", True)
                and is_mc_trained_model
                and baseline_sample_subdir == "samples"
            ):
                baseline_split_files = _map_split_files_mc_to_raw(model_split_files)
                print(
                    "[INFO] Baseline split file not found; mapped model MC split to raw sample filenames "
                    f"({len(model_split_files)} -> {len(baseline_split_files)} files)."
                )
            else:
                baseline_split_files = model_split_files
                print(
                    "[WARN] Baseline split file not found; reusing model split filenames as-is: "
                    f"{baseline_split_path}"
                )

        baseline_test_samples = load_split_samples(
            data_cfg["data_dir"],
            baseline_sample_subdir,
            data_cfg["forecast_name"],
            input_columns,
            output_columns,
            input_rows,
            output_rows,
            baseline_split_file,
            split_source_dir=baseline_split_source,
            split_files_override=baseline_split_files,
        )
        if not baseline_test_samples:
            print(
                "[WARN] No baseline test samples loaded after split/path resolution; "
                "skipping baseline evaluation."
            )
        baseline_test_dataset = TimeSeriesTargetDataset(baseline_test_samples) if baseline_test_samples else None
    else:
        baseline_test_dataset = None

    if eval_cfg["run_baselines"] and baseline_test_dataset is not None:
        baseline_output_rows = output_rows
        secondary, eval_cfg["window_hours"] = load_secondary(output_columns, eval_cfg["window_hours"])
        naive_preds, naive_targets = evaluate_naive(
            baseline_test_dataset,
            eval_cfg["historic_path"],
            output_columns,
            data_cfg["data_dir"],
            output_rows=baseline_output_rows,
            gap_hours=eval_cfg["gap_hours"],
        )
        linear_preds, linear_targets = evaluate_linear(
            data_cfg["data_dir"],
            data_cfg["forecast_name"],
            baseline_test_dataset,
            eval_cfg["historic_path"],
            output_columns,
            output_rows=baseline_output_rows,
            window_hours=eval_cfg["window_hours"],
            gap_hours=eval_cfg["gap_hours"],
            debug_plot=eval_cfg["debug_plot"],
            examples=eval_cfg["debug_examples"],
        )
        seasonal_preds, seasonal_targets = evaluate_seasonal(
            baseline_test_dataset,
            eval_cfg["historic_path"],
            output_columns,
            data_cfg["data_dir"],
            data_cfg["forecast_name"],
            output_rows=baseline_output_rows,
            diurnal_window=eval_cfg["diurnal_window"],
            secondary=secondary,
        )

        baseline_pairs = [
            (naive_preds, naive_targets),
            (linear_preds, linear_targets),
            (seasonal_preds, seasonal_targets),
        ]
        baseline_labels = ["Naive", "Linear", "Seasonal"]
        regression_pairs.extend(baseline_pairs)
        regression_labels.extend(baseline_labels)
        regression_split_files.extend([baseline_split_files] * len(baseline_pairs))

    if eval_cfg["run_regression"] and regression_pairs:
        common_meta = {
            "data_dir": str(data_cfg["data_dir"]),
            "n_train_samples": int(len(train_samples)) if train_samples is not None else np.nan,
            "n_test_samples": int(len(test_samples)),
            "input_dim": input_dim,
            "target_dim": target_dim,
        }

        if model_regression_pair is not None and regression_labels:
            summary_rows.append(
                _compute_regression_summary(
                    regression_labels[0],
                    model_regression_pair[0],
                    model_regression_pair[1],
                    eval_cfg["num_samples"],
                    metadata={**common_meta, "kind": "model"},
                )
            )

        if baseline_pairs:
            baseline_meta = {
                **common_meta,
                "kind": "baseline",
                "n_train_samples": np.nan,
                "n_test_samples": int(len(baseline_test_samples)) if baseline_test_samples else 0,
            }
            for (preds, targets), label in zip(baseline_pairs, baseline_labels):
                summary_rows.append(
                    _compute_regression_summary(
                        label,
                        preds,
                        targets,
                        eval_cfg["num_samples"],
                        metadata=baseline_meta,
                    )
                )

        visualizer(
            *regression_pairs,
            labels=regression_labels,
            forecast_name=data_cfg["forecast_name"],
            directory=data_cfg["data_dir"],
            num_samples=eval_cfg["num_samples"],
        )
        _plot_uncertainty_boxplots(
            regression_pairs,
            regression_labels,
            regression_split_files,
            directory=data_cfg["data_dir"],
            forecast_name=data_cfg["forecast_name"],
            num_samples=eval_cfg["num_samples"],
        )

    if eval_cfg["run_threshold_classification"] and regression_pairs:
        thresholds_df = load_thresholds(eval_cfg)
        class_results = []
        for preds, targets in regression_pairs:
            if eval_cfg["use_normalized_thresholds"]:
                preds_eval = preds
                targets_eval = targets
            else:
                preds_eval = reverse_normalize(preds, output_columns, Path(eval_cfg["normalization_path"]))
                targets_eval = reverse_normalize(targets, output_columns, Path(eval_cfg["normalization_path"]))

            bin_preds = binarize_predictions(preds_eval, output_columns=output_columns, thresholds_df=thresholds_df)
            bin_targets = binarize_predictions(targets_eval, output_columns=output_columns, thresholds_df=thresholds_df)
            class_results.append((bin_preds, bin_targets))

        classification_visualizer(
            *class_results,
            labels=regression_labels,
            directory=data_cfg["data_dir"],
            forecast_name=data_cfg["forecast_name"],
            num_samples=eval_cfg["num_samples"],
        )

        class_meta = {
            "data_dir": str(data_cfg["data_dir"]),
            "kind": "threshold_classification",
            "n_train_samples": int(len(train_samples)) if train_samples is not None else np.nan,
            "n_test_samples": int(len(test_samples)),
            "input_dim": input_dim,
            "target_dim": target_dim,
        }
        for (preds, targets), label in zip(class_results, regression_labels):
            summary_rows.append(
                _compute_classification_summary(
                    label,
                    preds,
                    targets,
                    eval_cfg["num_samples"],
                    metadata=class_meta,
                )
            )

    if eval_cfg["run_pure_classification"]:
        if model_type != "xgb_classifier":
            print("Skipping pure classification: model_type is not xgb_classifier")
        else:
            preds_flat = model.predict(X_test)
            preds = preds_flat.reshape(-1, output_dim)
            targets = np.array([np.rint(s[1].flatten()) for s in test_samples]).reshape(-1, output_dim)

            classification_visualizer(
                (preds, targets),
                labels=["XGBClassifier"],
                directory=data_cfg["data_dir"],
                forecast_name=data_cfg["forecast_name"],
                num_samples=eval_cfg["num_samples"],
            )

            summary_rows.append(
                _compute_classification_summary(
                    "XGBClassifier",
                    preds,
                    targets,
                    eval_cfg["num_samples"],
                    metadata={
                        "data_dir": str(data_cfg["data_dir"]),
                        "kind": "model",
                        "n_train_samples": int(len(train_samples)) if train_samples is not None else np.nan,
                        "n_test_samples": int(len(test_samples)),
                        "input_dim": input_dim,
                        "target_dim": target_dim,
                    },
                )
            )

    single_summary_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"], "evaluation_summary.csv")
    _write_summary_csv(summary_rows, single_summary_path)

    return {
        "config_path": str(config_path),
        "data_dir": str(data_cfg["data_dir"]),
        "forecast_name": data_cfg["forecast_name"],
        "model_type": model_type,
        "model_pair": model_regression_pair,
        "model_split_files": model_split_files,
        "model_label": _combined_model_label(data_cfg, model_type),
        "baseline_pairs": baseline_pairs,
        "baseline_labels": baseline_labels,
        "baseline_split_files": baseline_split_files,
        "num_samples": int(eval_cfg["num_samples"]),
        "n_train_samples": int(len(train_samples)) if train_samples is not None else np.nan,
        "n_test_samples": int(len(test_samples)),
        "input_dim": input_dim,
        "target_dim": target_dim,
        "summary_rows": summary_rows,
    }


def write_combined_outputs(results):
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
    args = parser.parse_args()

    config_paths = _expand_config_inputs(args.config)
    if not config_paths:
        raise ValueError(
            "No valid config files found after expanding --config arguments. "
            "Use .yml/.yaml/.json files on a single command line."
        )

    results = [evaluate_single_config(config_path) for config_path in config_paths]

    if len(results) > 1:
        write_combined_outputs(results)


if __name__ == "__main__":
    main()

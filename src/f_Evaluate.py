"""
Unified evaluation script for regression, transformer, and classification models.
Uses a YAML/JSON config file to control what gets evaluated.

Example terminal usage:
python src/f_Evaluate.py --config data/output/regression/MC_pH/forecasts/gp_01/config_evaluate_model_gp_01.yaml
python src/f_Evaluate.py --config data/output/regression/MC_pH/forecasts/transformer_01/config_evaluate_model_transformer_01.yaml
python src/f_Evaluate.py --config data/output/regression/MC_pH/forecasts/xgb_01/config_evaluate_model_xgb_01.yaml
"""

import os
import json
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
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
}


def load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if path.suffix in [".yaml", ".yml"]:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            config["__config_dir"] = str(path.resolve().parent)
            return config
    if path.suffix == ".json":
        with open(path, "r") as f:
            config = json.load(f)
            config["__config_dir"] = str(path.resolve().parent)
            return config

    raise ValueError(f"Unsupported config file format: {path.suffix}")


def _resolve_path_from_config(path_value, config_dir):
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (Path(config_dir) / path_obj).resolve()


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


def load_split_samples(data_dir, sample_subdir, forecast_name, input_columns, output_columns, input_rows, output_rows, split_file):
    reloadset = Path(data_dir, "forecasts", forecast_name, "test_files.txt")
    if split_file != "test_files.txt":
        reloadset = Path(data_dir, "forecasts", forecast_name, split_file)
    with open(reloadset) as f:
        split_files = [line.strip() for line in f]

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


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    config_dir = config["__config_dir"]

    model_type = config["model_type"]
    model_name = config["model_name"]
    data_cfg = config["data"]
    eval_cfg = merge_eval_config(config)

    data_cfg["data_dir"] = str(_resolve_path_from_config(data_cfg["data_dir"], config_dir))
    data_cfg["sample_subdir"] = data_cfg.get("sample_subdir", "samples")
    for key in ["historic_path", "thresholds_path", "normalization_path"]:
        if eval_cfg.get(key):
            eval_cfg[key] = str(_resolve_path_from_config(eval_cfg[key], config_dir))

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

    split_cfg = config.get("data_split", {"random_state": 42})
    model = load_model(model_type, data_cfg, split_cfg, model_name, model_config, device, train_samples, config_dir)

    regression_pairs = []
    regression_labels = []

    if eval_cfg["run_regression"]:
        if model_type == "transformer":
            preds, targets = evaluate_transformer(model, test_dataset, device)
            regression_pairs.append((preds, targets))
            regression_labels.append("Transformer")
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
        elif model_type == "xgb_classifier":
            print("Skipping regression evaluation for xgb_classifier model_type")

    if eval_cfg["run_baselines"]:
        secondary, eval_cfg["window_hours"] = load_secondary(output_columns, eval_cfg["window_hours"])
        naive_preds, naive_targets = evaluate_naive(
            test_dataset,
            eval_cfg["historic_path"],
            output_columns,
            data_cfg["data_dir"],
            output_rows=output_rows,
            gap_hours=eval_cfg["gap_hours"],
        )
        linear_preds, linear_targets = evaluate_linear(
            data_cfg["data_dir"],
            data_cfg["forecast_name"],
            test_dataset,
            eval_cfg["historic_path"],
            output_columns,
            output_rows=output_rows,
            window_hours=eval_cfg["window_hours"],
            gap_hours=eval_cfg["gap_hours"],
            debug_plot=eval_cfg["debug_plot"],
            examples=eval_cfg["debug_examples"],
        )
        seasonal_preds, seasonal_targets = evaluate_seasonal(
            test_dataset,
            eval_cfg["historic_path"],
            output_columns,
            data_cfg["data_dir"],
            data_cfg["forecast_name"],
            output_rows=output_rows,
            diurnal_window=eval_cfg["diurnal_window"],
            secondary=secondary,
        )

        regression_pairs.extend(
            [(naive_preds, naive_targets), (linear_preds, linear_targets), (seasonal_preds, seasonal_targets)]
        )
        regression_labels.extend(["Naive", "Linear", "Seasonal"])

    if eval_cfg["run_regression"] and regression_pairs:
        visualizer(
            *regression_pairs,
            labels=regression_labels,
            forecast_name=data_cfg["forecast_name"],
            directory=data_cfg["data_dir"],
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


if __name__ == "__main__":
    main()

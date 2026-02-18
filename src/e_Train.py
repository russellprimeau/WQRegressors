"""
Consolidated training script for Transformer, XGBoost Regressor, and XGBoost Classifier models.
Supports configuration via YAML/JSON config files.

Example terminal usage:
python src/e_Train.py --config data/output/regression/MC_pH/config_gp_01.yml
python src/e_Train.py --config data/output/regression/MC_pH/config_transformer_01.yml
python src/e_Train.py --config data/output/regression/MC_pH/config_xgb_01.yml
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
from torch.utils.data import DataLoader
import gpytorch
from utils.training import write_config, splitter
from utils.transformer import (
    train_model as train_transformer,
    TimeSeriesTransformer,
    TimeSeriesTargetDataset,
)


# ===========================================================================================
# DEFAULT CONFIGURATIONS
# ===========================================================================================

DEFAULT_COMMON_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "matplotlib_backend": "Agg",
}

DEFAULT_TRANSFORMER_CONFIG = {
    "model_dim": 128,
    "num_heads": 4,
    "num_layers": 8,
    "dropout": 0.1,
    "batch_size": 10,
    "num_epochs": 100,
    "loss_threshold": 0.000001,
    "learning_rate": 1e-4,
    "patience": 10,
}

DEFAULT_XGB_REGRESSOR_CONFIG = {
    "metric": "rmse",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_estimators": 1100,
    "max_depth": 10,
    "subsample": 0.2,
    "colsample_bytree": 0.8,
    "learning_rate": 0.01,
    "n_jobs": -1,
    "early_stopping_rounds": 200,
}

DEFAULT_XGB_CLASSIFIER_CONFIG = {
    "eval_metric": "logloss",
    "tree_method": "hist",
    "objective": "binary:logistic",
    "n_estimators": 1500,
    "max_depth": 10,
    "subsample": 0.2,
    "colsample_bytree": 0.8,
    "learning_rate": 0.01,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

DEFAULT_GP_REGRESSOR_CONFIG = {
    "kernel": "matern52",
    "ard": True,
    "input_standardize": True,
    "target_standardize": True,
    "use_uncertain_input_kernel": True,
    "uncertainty_summary_dir": None,
    "learning_rate": 0.01,
    "num_epochs": 250,
    "patience": 20,
    "max_train_size": 5000,
}

DEFAULT_DATA_SPLIT_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "reuse_split": False,
    "split_source": None,
    "split_type": "random",
    "nan_tolerance": 0.8,
}

DEFAULT_EVALUATION_CONFIG = {
    "run_regression": True,
    "run_threshold_classification": False,
    "run_pure_classification": False,
    "run_baselines": True,
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


# ===========================================================================================
# CONFIG LOADING AND MERGING
# ===========================================================================================

def load_config(config_path):
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    def _read_text_with_fallback(path_obj):
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path_obj, 'r', encoding='cp1252') as f:
                return f.read()

    raw_text = _read_text_with_fallback(path)

    if path.suffix in ['.yaml', '.yml']:
        config = yaml.safe_load(raw_text)
    elif path.suffix == '.json':
        config = json.loads(raw_text)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")

    # Store absolute config directory so all relative paths resolve from config location
    config["__config_dir"] = str(path.resolve().parent)
    
    return config


def merge_with_defaults(config, model_type):
    """Merge provided config with defaults, printing when defaults are applied."""
    merged_config = config.copy()
    
    # Get model-specific defaults
    if model_type == "transformer":
        defaults = DEFAULT_TRANSFORMER_CONFIG
    elif model_type == "gp_regressor":
        defaults = DEFAULT_GP_REGRESSOR_CONFIG
    elif model_type == "xgb_regressor":
        defaults = DEFAULT_XGB_REGRESSOR_CONFIG
    elif model_type == "xgb_classifier":
        defaults = DEFAULT_XGB_CLASSIFIER_CONFIG
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Merge in data split defaults
    if "data_split" not in merged_config:
        merged_config["data_split"] = {}
    
    for key, default_value in DEFAULT_DATA_SPLIT_CONFIG.items():
        if key not in merged_config["data_split"]:
            merged_config["data_split"][key] = default_value
            print(f"  [DEFAULT] data_split.{key} = {default_value}")
    
    # Merge in model-specific defaults
    if "hyperparameters" not in merged_config:
        merged_config["hyperparameters"] = {}
    
    for key, default_value in defaults.items():
        if key not in merged_config["hyperparameters"]:
            merged_config["hyperparameters"][key] = default_value
            print(f"  [DEFAULT] hyperparameters.{key} = {default_value}")
    
    # Add device if not specified
    if "device" not in merged_config:
        merged_config["device"] = DEFAULT_COMMON_CONFIG["device"]
        print(f"  [DEFAULT] device = {merged_config['device']}")
    
    # Add matplotlib backend if not specified
    if "matplotlib_backend" not in merged_config:
        merged_config["matplotlib_backend"] = DEFAULT_COMMON_CONFIG["matplotlib_backend"]
    
    return merged_config


# ===========================================================================================
# DATA LOADING
# ===========================================================================================

def _resolve_path_from_config(path_value, config_dir):
    """Resolve path value relative to config file directory."""
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (Path(config_dir) / path_obj).resolve()


def _resolve_data_paths(data_cfg, config_dir):
    """Resolve base data directory and sample subdirectory with backward compatibility."""
    configured_subdir = data_cfg.get("sample_subdir")
    data_dir_path = _resolve_path_from_config(data_cfg["data_dir"], config_dir)

    if configured_subdir:
        return str(data_dir_path), configured_subdir

    # Backward compatibility: if data_dir points directly to samples folder, infer parent + subdir
    if data_dir_path.name in {"samples", "mc_replicates"}:
        return str(data_dir_path.parent), data_dir_path.name

    return str(data_dir_path), "samples"


def _resolve_eval_path_for_write(path_value, config_dir):
    """Resolve evaluation reference paths for write_evaluation_config with sensible fallback."""
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj.resolve()

    config_based = (Path(config_dir) / path_obj).resolve()
    if config_based.exists():
        return config_based

    src_based = (Path(__file__).parent / path_obj).resolve()
    if src_based.exists():
        return src_based

    return config_based

def load_and_split_data(config):
    """Load and split data according to configuration."""
    data_cfg = config["data"]
    split_cfg = config["data_split"]
    config_dir = config["__config_dir"]

    base_data_dir, sample_subdir = _resolve_data_paths(data_cfg, config_dir)
    data_cfg["data_dir"] = base_data_dir
    data_cfg["sample_subdir"] = sample_subdir

    if split_cfg.get("split_source") is not None:
        split_cfg["split_source"] = str(_resolve_path_from_config(split_cfg["split_source"], config_dir))

    nan_tolerance = split_cfg.get("nan_tolerance", DEFAULT_DATA_SPLIT_CONFIG["nan_tolerance"])
    if nan_tolerance is not None:
        nan_tolerance = float(nan_tolerance)
        if nan_tolerance < 0 or nan_tolerance > 1:
            raise ValueError(f"data_split.nan_tolerance must be in [0, 1], got {nan_tolerance}")

    # print(f"Sample directory: {Path(base_data_dir) / sample_subdir}")
    
    input_rows = slice(data_cfg["input_row_1"], data_cfg["input_row_2"])
    
    train_samples, test_samples = splitter(
        data_cfg["data_dir"],
        data_cfg["forecast_name"],
        data_cfg["input_columns"],
        input_rows,
        data_cfg["output_columns"],
        data_cfg["output_rows"],
        split_cfg["fault_tolerant"],
        split_cfg["reuse_split"],
        split_cfg["split_source"],
        split_cfg["split_type"],
        split_cfg["test_size"],
        split_cfg["random_state"],
        data_cfg["sample_subdir"],
        nan_tolerance,
    )
    
    return train_samples, test_samples, input_rows


def _canonical_feature_name(name):
    text = str(name).strip().lower().replace("µ", "u")
    text = text.replace("micro", "u")
    text = text.replace("_", " ")
    if " - " in text:
        text = text.split(" - ", 1)[1].strip()
    for token in ["(", ")", "/", "%", "°", "-", ".", ","]:
        text = text.replace(token, " ")
    text = " ".join(text.split())
    return text


def _resolve_summary_dir(hyper_cfg, config_dir):
    if hyper_cfg.get("uncertainty_summary_dir"):
        return _resolve_path_from_config(hyper_cfg["uncertainty_summary_dir"], config_dir)
    return Path(__file__).parent.parent / "data" / "output" / "calibration" / "summaries"


def _load_uncertainty_std_map(summary_dir):
    if not summary_dir.exists():
        print(f"[WARN] Uncertainty summary directory not found: {summary_dir}")
        return {}

    summary_map = {}
    summary_files = list(summary_dir.rglob("*_uncertainty_summary.csv"))
    for file_path in summary_files:
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
            canonical_name = _canonical_feature_name(sensor_name)
            if canonical_name in summary_map:
                print(
                    f"[WARN] Duplicate uncertainty entry for '{sensor_name}' "
                    f"(canonical='{canonical_name}'). Keeping first source: "
                    f"{summary_map[canonical_name]['source_file']}"
                )
                continue
            summary_map[canonical_name] = {
                "offset_std": float(offset_std),
                "sensor_name": str(sensor_name),
                "source_file": str(file_path)
            }
        except Exception as exc:
            print(f"[WARN] Could not parse uncertainty summary file {file_path}: {exc}")
    return summary_map


def _build_feature_uncertainty_variance(config):
    data_cfg = config["data"]
    hyper_cfg = config["hyperparameters"]
    config_dir = config["__config_dir"]
    input_columns = data_cfg["input_columns"]
    seq_len = data_cfg["input_row_2"] - data_cfg["input_row_1"]

    summary_dir = _resolve_summary_dir(hyper_cfg, config_dir)
    summary_std_map = _load_uncertainty_std_map(summary_dir)
    print(f"[INFO] Uncertainty source directory: {summary_dir}")

    norm_path = Path(data_cfg["data_dir"]) / "normalization.json"
    norm_params = {}
    if norm_path.exists():
        try:
            with open(norm_path, "r") as f:
                norm_params = json.load(f)
        except Exception as exc:
            print(f"[WARN] Could not read normalization.json at {norm_path}: {exc}")

    feature_variances = []
    for feature in input_columns:
        candidates = [_canonical_feature_name(feature)]
        if " - " in feature:
            candidates.append(_canonical_feature_name(feature.split(" - ", 1)[1]))

        matched_entry = None
        matched_key = None
        for candidate in candidates:
            if candidate in summary_std_map:
                matched_entry = summary_std_map[candidate]
                matched_key = candidate
                break

        raw_std = 0.0 if matched_entry is None else float(matched_entry["offset_std"])
        applied_std = raw_std
        scaling_applied = False

        if applied_std > 0 and feature in norm_params:
            v_min = norm_params[feature].get("min", 0)
            v_max = norm_params[feature].get("max", 1)
            v_range = v_max - v_min
            if v_range not in [0, 0.0]:
                applied_std = applied_std / v_range
                scaling_applied = True

        variance = float(applied_std ** 2)
        feature_variances.append(variance)

        if matched_entry is None:
            print(
                f"  [UNCERTAINTY] feature='{feature}' | source='none' | "
                f"matched_key='none' | raw_offset_std=0 | applied_std=0 | var=0"
            )
        else:
            scaling_note = "scaled_by_normalization" if scaling_applied else "no_scaling"
            print(
                f"  [UNCERTAINTY] feature='{feature}' | source='{matched_entry['source_file']}' | "
                f"sensor='{matched_entry['sensor_name']}' | matched_key='{matched_key}' | "
                f"raw_offset_std={raw_std:.6g} | applied_std={applied_std:.6g} | "
                f"var={variance:.6g} | {scaling_note}"
            )

    flattened_variances = np.tile(np.array(feature_variances, dtype=np.float32), seq_len)
    return flattened_variances


def write_evaluation_config(config):
    """Write an evaluation config file for f_Evaluate.py next to trained model artifacts."""
    model_type = config["model_type"]
    model_name = config["model_name"]
    data_cfg = config["data"]
    config_dir = config.get("__config_dir", str(Path.cwd()))

    evaluation_cfg = DEFAULT_EVALUATION_CONFIG.copy()
    if model_type == "xgb_classifier":
        evaluation_cfg["run_regression"] = False
        evaluation_cfg["run_pure_classification"] = True

    normalization_candidate = Path(data_cfg["data_dir"]) / "normalization.json"
    if normalization_candidate.exists():
        evaluation_cfg["normalization_path"] = str(normalization_candidate)

    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
    os.makedirs(save_path, exist_ok=True)

    # Store config paths relative to evaluation config file location
    relative_data_dir = os.path.relpath(data_cfg["data_dir"], start=save_path)
    for key in ["historic_path", "thresholds_path", "normalization_path"]:
        if evaluation_cfg.get(key):
            abs_path = _resolve_eval_path_for_write(evaluation_cfg[key], config_dir)
            evaluation_cfg[key] = os.path.relpath(abs_path, start=save_path)

    eval_config = {
        "model_type": model_type,
        "model_name": "",
        "data": {
            "data_dir": relative_data_dir,
            "sample_subdir": data_cfg.get("sample_subdir", "samples"),
            "forecast_name": data_cfg["forecast_name"],
            "input_columns": data_cfg["input_columns"],
            "input_row_1": data_cfg["input_row_1"],
            "input_row_2": data_cfg["input_row_2"],
            "output_columns": data_cfg["output_columns"],
            "output_rows": data_cfg["output_rows"],
        },
        "data_split": config.get("data_split", DEFAULT_DATA_SPLIT_CONFIG),
        "evaluation": evaluation_cfg,
    }

    model_name_for_file = Path(str(data_cfg["forecast_name"]).strip()).name
    eval_config_path = save_path / f"config_evaluate_{model_name_for_file}.yml"
    with open(eval_config_path, "w") as f:
        yaml.dump(eval_config, f, sort_keys=False)

    print(f"Evaluation config saved to: {eval_config_path}")


# ===========================================================================================
# TRANSFORMER TRAINING
# ===========================================================================================

def train_transformer_model(config, train_samples, test_samples):
    """Train Transformer model."""
    print("\n" + "="*80)
    print("TRAINING TRANSFORMER MODEL")
    print("="*80)
    
    device = torch.device(config["device"])
    data_cfg = config["data"]
    hyper_cfg = config["hyperparameters"]
    
    # Prepare data
    train_dataset = TimeSeriesTargetDataset(train_samples)
    test_dataset = TimeSeriesTargetDataset(test_samples)
    trainloader = DataLoader(train_dataset, batch_size=hyper_cfg["batch_size"], shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=hyper_cfg["batch_size"], shuffle=True)
    
    model_config = {
        'input_dim': len(data_cfg["input_columns"]),
        'model_dim': hyper_cfg["model_dim"],
        'num_heads': hyper_cfg["num_heads"],
        'num_layers': hyper_cfg["num_layers"],
        'dropout': hyper_cfg["dropout"],
        'output_dim': len(data_cfg["output_columns"]) * len(data_cfg["output_rows"]),
        'seq_len': data_cfg["input_row_2"] - data_cfg["input_row_1"],
        'input_columns': data_cfg["input_columns"],
        'input_row_1': data_cfg["input_row_1"],
        'input_row_2': data_cfg["input_row_2"],
        'output_columns': data_cfg["output_columns"],
        'output_rows': data_cfg["output_rows"],
    }
    
    # Write config
    write_config(model_config, data_cfg["data_dir"], data_cfg["forecast_name"], "")
    
    # Create and train model
    model = TimeSeriesTransformer(model_config).to(device)
    
    train_transformer(
        data_cfg["data_dir"],
        model,
        data_cfg["forecast_name"],
        trainloader,
        testloader,
        device,
        hyper_cfg["num_epochs"],
        hyper_cfg["learning_rate"],
        hyper_cfg["loss_threshold"],
        hyper_cfg["patience"],
        ""
    )
    
    # Save model
    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), save_path / "transformer_model.pt")
    print(f"\nModel saved to: {save_path / 'transformer_model.pt'}")
    write_evaluation_config(config)


# ===========================================================================================
# GAUSSIAN PROCESS REGRESSOR TRAINING
# ===========================================================================================

def train_gp_regressor_model(config, train_samples, test_samples):
    """Train GPyTorch Gaussian Process Regressor model(s)."""
    if gpytorch is None:
        raise ImportError(
            "gpytorch is not installed. Install it with: pip install gpytorch"
        )

    print("\n" + "="*80)
    print("TRAINING GAUSSIAN PROCESS REGRESSOR MODEL")
    print("="*80)

    device = torch.device(config["device"])
    data_cfg = config["data"]
    hyper_cfg = config["hyperparameters"]
    split_cfg = config["data_split"]

    X_train_np = np.array([s[0].flatten() for s in train_samples], dtype=np.float32)
    y_train_np = np.array([s[1].flatten() for s in train_samples], dtype=np.float32)
    X_test_np = np.array([s[0].flatten() for s in test_samples], dtype=np.float32)
    y_test_np = np.array([s[1].flatten() for s in test_samples], dtype=np.float32)

    if y_train_np.ndim == 1:
        y_train_np = y_train_np.reshape(-1, 1)
    if y_test_np.ndim == 1:
        y_test_np = y_test_np.reshape(-1, 1)

    max_train_size = hyper_cfg["max_train_size"]
    if max_train_size is not None and len(X_train_np) > max_train_size:
        rng = np.random.default_rng(split_cfg["random_state"])
        keep_idx = rng.choice(len(X_train_np), size=max_train_size, replace=False)
        X_train_np = X_train_np[keep_idx]
        y_train_np = y_train_np[keep_idx]
        print(f"Subsampled train set to {max_train_size} for exact GP tractability")

    x_mean = X_train_np.mean(axis=0)
    x_std = X_train_np.std(axis=0)
    x_std[x_std < 1e-8] = 1.0

    if hyper_cfg["input_standardize"]:
        X_train_used = (X_train_np - x_mean) / x_std
        X_test_used = (X_test_np - x_mean) / x_std
    else:
        X_train_used = X_train_np
        X_test_used = X_test_np
        x_mean = np.zeros_like(x_mean)
        x_std = np.ones_like(x_std)

    X_train = torch.tensor(X_train_used, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test_used, dtype=torch.float32, device=device)

    kernel_name = str(hyper_cfg["kernel"]).lower()
    use_uncertain_kernel = bool(hyper_cfg.get("use_uncertain_input_kernel", True))

    if use_uncertain_kernel and not hyper_cfg["ard"]:
        print("[WARN] Uncertain-input kernel works best with ARD. Forcing ARD=True for GP kernel.")

    ard_dims = X_train.shape[1] if (hyper_cfg["ard"] or use_uncertain_kernel) else None
    input_uncertainty_var = None
    if use_uncertain_kernel:
        input_uncertainty_var = torch.tensor(
            _build_feature_uncertainty_variance(config),
            dtype=torch.float32,
            device=device
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
            denom = ls2 + 2.0 * self.input_variance
            denom = torch.clamp(denom, min=1e-10)

            x1_exp = x1.unsqueeze(-2)
            x2_exp = x2.unsqueeze(-3)
            sq_dist = ((x1_exp - x2_exp).pow(2) / denom).sum(dim=-1)

            det_term = torch.sqrt(torch.prod(ls2 / denom))
            return det_term * torch.exp(-0.5 * sq_dist)

    def build_base_kernel():
        if use_uncertain_kernel:
            if kernel_name != "rbf":
                print(f"[WARN] Uncertain-input kernel currently supports RBF closed form. Overriding kernel '{kernel_name}' -> 'rbf'.")
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

    output_dim = y_train_np.shape[1]
    output_train_losses = []
    output_rmse = []
    models_state = []

    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
    os.makedirs(save_path, exist_ok=True)

    for output_idx in range(output_dim):
        y_train_col = y_train_np[:, output_idx]
        y_test_col = y_test_np[:, output_idx]

        y_mean = float(y_train_col.mean())
        y_std = float(y_train_col.std())
        if y_std < 1e-8:
            y_std = 1.0

        if hyper_cfg["target_standardize"]:
            y_train_used = (y_train_col - y_mean) / y_std
        else:
            y_train_used = y_train_col
            y_mean = 0.0
            y_std = 1.0

        y_train = torch.tensor(y_train_used, dtype=torch.float32, device=device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPRegressor(X_train, y_train, likelihood).to(device)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_cfg["learning_rate"])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        losses = []
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_likelihood_state = None

        for epoch in range(hyper_cfg["num_epochs"]):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            losses.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
                best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_likelihood_state = {k: v.detach().cpu() for k, v in likelihood.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= hyper_cfg["patience"]:
                    break

        model.load_state_dict(best_model_state)
        likelihood.load_state_dict(best_likelihood_state)

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(model(X_test))
            pred_mean = pred_dist.mean.detach().cpu().numpy()

        pred_mean = pred_mean * y_std + y_mean
        rmse = float(np.sqrt(np.mean((pred_mean - y_test_col) ** 2)))

        output_train_losses.append(losses)
        output_rmse.append(rmse)
        models_state.append({
            "output_index": output_idx,
            "model_state_dict": best_model_state,
            "likelihood_state_dict": best_likelihood_state,
            "target_mean": y_mean,
            "target_std": y_std,
            "train_nll": best_loss,
            "test_rmse": rmse,
        })

        print(f"Output {output_idx + 1}/{output_dim} - best train NLL: {best_loss:.6f}, test RMSE: {rmse:.6f}")

    model_config = {
        'input_dim': len(data_cfg["input_columns"]),
        'output_dim': output_dim,
        'seq_len': data_cfg["input_row_2"] - data_cfg["input_row_1"],
        'input_columns': data_cfg["input_columns"],
        'input_row_1': data_cfg["input_row_1"],
        'input_row_2': data_cfg["input_row_2"],
        'output_columns': data_cfg["output_columns"],
        'output_rows': data_cfg["output_rows"],
        'kernel': hyper_cfg["kernel"],
        'ard': hyper_cfg["ard"],
        'input_standardize': hyper_cfg["input_standardize"],
        'target_standardize': hyper_cfg["target_standardize"],
        'use_uncertain_input_kernel': use_uncertain_kernel,
    }
    write_config(model_config, data_cfg["data_dir"], data_cfg["forecast_name"], "")

    artifact = {
        "model_type": "gp_regressor",
        "hyperparameters": hyper_cfg,
        "input_mean": x_mean,
        "input_std": x_std,
        "input_dim": int(X_train.shape[1]),
        "output_dim": output_dim,
        "models": models_state,
    }
    torch.save(artifact, save_path / "gp_model.pt")
    print(f"\nModel saved to: {save_path / 'gp_model.pt'}")

    plt.figure(figsize=(8, 5))
    for idx, losses in enumerate(output_train_losses):
        plt.plot(range(1, len(losses) + 1), losses, label=f'Output {idx + 1} train NLL')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Marginal Likelihood')
    plt.grid(True, ls="--")
    plt.title('GP Training Loss by Output')
    plt.legend()
    plt.savefig(save_path / "loss_plot.png")
    plt.close()
    print(f"Loss plot saved to: {save_path / 'loss_plot.png'}")
    write_evaluation_config(config)


# ===========================================================================================
# XGBOOST REGRESSOR TRAINING
# ===========================================================================================

def train_xgb_regressor_model(config, train_samples, test_samples):
    """Train XGBoost Regressor model."""
    print("\n" + "="*80)
    print("TRAINING XGBOOST REGRESSOR MODEL")
    print("="*80)
    
    data_cfg = config["data"]
    hyper_cfg = config["hyperparameters"]
    
    # Prepare data
    X_train = np.array([s[0].flatten() for s in train_samples])
    y_train = np.array([s[1].flatten()[0] for s in train_samples])
    X_test = np.array([s[0].flatten() for s in test_samples])
    y_test = np.array([s[1].flatten()[0] for s in test_samples])
    
    model_config = {
        'input_dim': len(data_cfg["input_columns"]),
        'output_dim': len(data_cfg["output_columns"]) * len(data_cfg["output_rows"]),
        'seq_len': data_cfg["input_row_2"] - data_cfg["input_row_1"],
        'input_columns': data_cfg["input_columns"],
        'input_row_1': data_cfg["input_row_1"],
        'input_row_2': data_cfg["input_row_2"],
        'output_columns': data_cfg["output_columns"],
        'output_rows': data_cfg["output_rows"],
    }
    
    write_config(model_config, data_cfg["data_dir"], data_cfg["forecast_name"], "")
    
    # Create and train model
    model = xgb.XGBRegressor(
        tree_method=hyper_cfg["tree_method"],
        objective=hyper_cfg["objective"],
        n_estimators=hyper_cfg["n_estimators"],
        max_depth=hyper_cfg["max_depth"],
        subsample=hyper_cfg["subsample"],
        colsample_bytree=hyper_cfg["colsample_bytree"],
        learning_rate=hyper_cfg["learning_rate"],
        n_jobs=hyper_cfg["n_jobs"],
        early_stopping_rounds=hyper_cfg["early_stopping_rounds"]
    )
    
    print("Training XGBRegressor...")
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
    
    # Save model
    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
    os.makedirs(save_path, exist_ok=True)
    model.save_model(save_path / "xgboost_model.json")
    print(f"\nModel saved to: {save_path / 'xgboost_model.json'}")
    
    # Plot results
    results = model.evals_result()
    metric = hyper_cfg["metric"]
    epochs = len(results['validation_0'][metric])
    plt.figure(figsize=(8, 5))
    plt.loglog(range(epochs), results['validation_0'][metric], label='Training Loss')
    plt.loglog(range(epochs), results['validation_1'][metric], label='Validation Loss')
    plt.xlabel('Boosting Rounds')
    plt.ylabel(metric)
    plt.grid(True, which="both", ls="--")
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(save_path / "loss_plot.png")
    plt.close()
    print(f"Loss plot saved to: {save_path / 'loss_plot.png'}")
    write_evaluation_config(config)


# ===========================================================================================
# XGBOOST CLASSIFIER TRAINING
# ===========================================================================================

def train_xgb_classifier_model(config, train_samples, test_samples):
    """Train XGBoost Classifier model."""
    print("\n" + "="*80)
    print("TRAINING XGBOOST CLASSIFIER MODEL")
    print("="*80)
    
    data_cfg = config["data"]
    hyper_cfg = config["hyperparameters"]
    
    # Prepare data and ensure binary outputs
    X_train = np.array([s[0].flatten() for s in train_samples])
    y_train = np.array([int(round(s[1].flatten()[0])) for s in train_samples])
    X_test = np.array([s[0].flatten() for s in test_samples])
    y_test = np.array([int(round(s[1].flatten()[0])) for s in test_samples])
    
    model_config = {
        'input_dim': len(data_cfg["input_columns"]),
        'output_dim': len(data_cfg["output_columns"]) * len(data_cfg["output_rows"]),
        'seq_len': data_cfg["input_row_2"] - data_cfg["input_row_1"],
        'input_columns': data_cfg["input_columns"],
        'input_row_1': data_cfg["input_row_1"],
        'input_row_2': data_cfg["input_row_2"],
        'output_columns': data_cfg["output_columns"],
        'output_rows': data_cfg["output_rows"],
    }
    
    write_config(model_config, data_cfg["data_dir"], data_cfg["forecast_name"], "")
    
    # Create and train model
    model = xgb.XGBClassifier(
        eval_metric=hyper_cfg["eval_metric"],
        tree_method=hyper_cfg["tree_method"],
        objective=hyper_cfg["objective"],
        n_estimators=hyper_cfg["n_estimators"],
        max_depth=hyper_cfg["max_depth"],
        subsample=hyper_cfg["subsample"],
        colsample_bytree=hyper_cfg["colsample_bytree"],
        learning_rate=hyper_cfg["learning_rate"],
        n_jobs=hyper_cfg["n_jobs"],
        early_stopping_rounds=hyper_cfg["early_stopping_rounds"],
    )
    
    print("Training XGBClassifier...")
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
    
    # Save model
    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
    os.makedirs(save_path, exist_ok=True)
    model.save_model(save_path / "xgboost_model.json")
    print(f"\nModel saved to: {save_path / 'xgboost_model.json'}")
    
    # Plot results
    results = model.evals_result()
    metric = hyper_cfg["eval_metric"]
    epochs = len(results['validation_0'][metric])
    plt.figure(figsize=(8, 5))
    plt.loglog(range(epochs), results['validation_0'][metric], label='Train logloss')
    plt.loglog(range(epochs), results['validation_1'][metric], label='Validation logloss')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Logloss')
    plt.grid(True, which="both", ls="--")
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(save_path / "loss_plot.png")
    plt.close()
    print(f"Loss plot saved to: {save_path / 'loss_plot.png'}")
    write_evaluation_config(config)


# ===========================================================================================
# MAIN ORCHESTRATION
# ===========================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Consolidated training script for Transformer, GP Regressor, XGBoost Regressor, and XGBoost Classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML or JSON)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("LOADING CONFIGURATION")
    print("="*80)
    print(f"Config file: {args.config}")
    
    # Load config
    config = load_config(args.config)
    
    # Validate required fields
    required_fields = ["model_type", "model_name", "data"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    model_type = config["model_type"]
    print(f"Model type: {model_type}")
    
    # Merge with defaults and print applied defaults
    print("\nApplying defaults:")
    config = merge_with_defaults(config, model_type)
    
    # Configure device and backend
    device = torch.device(config["device"])
    print(f"\nUsing device: {device}")
    matplotlib.use(config["matplotlib_backend"])
    
    # Load and split data
    print("\n" + "="*80)
    print("LOADING AND SPLITTING DATA")
    print("="*80)
    train_samples, test_samples, input_rows = load_and_split_data(config)
    # print(f"Training samples: {len(train_samples)}")
    # print(f"Test samples: {len(test_samples)}")
    
    # Train appropriate model
    if model_type == "transformer":
        train_transformer_model(config, train_samples, test_samples)
    elif model_type == "gp_regressor":
        train_gp_regressor_model(config, train_samples, test_samples)
    elif model_type == "xgb_regressor":
        train_xgb_regressor_model(config, train_samples, test_samples)
    elif model_type == "xgb_classifier":
        train_xgb_classifier_model(config, train_samples, test_samples)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

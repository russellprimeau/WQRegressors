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
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
from torch.utils.data import DataLoader
try:
    import gpytorch
except ImportError:
    gpytorch = None
from utils.training import write_config, splitter
from utils.transformer import (
    train_model as train_transformer,
    TimeSeriesTransformer,
    TimeSeriesTargetDataset,
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
from utils.gp_utils import build_base_kernel, ExactGPRegressor


NORMALIZATION_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "output"
    / "sensors"
    / "normalization.json"
)


# ===========================================================================================
# DEFAULT CONFIGURATIONS
# ===========================================================================================

DEFAULT_COMMON_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "matplotlib_backend": "Agg",
    "save_training_plots": True,
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
    "corr_lambda": 0.1,
    "corr_eps": 1e-8,
    "corr_clip": True,
    "num_workers": 0,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
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


def _resolve_xgb_runtime_hyperparameters(hyper_cfg: dict, preferred_device: str) -> dict:
    """Return effective XGB hyperparameters with version-aware GPU defaults."""
    effective = dict(hyper_cfg)
    prefer_cuda = str(preferred_device).lower().startswith("cuda") and torch.cuda.is_available()
    major, _minor = _parse_xgb_version()

    if prefer_cuda:
        if major >= 2:
            effective.setdefault("device", "cuda")
            effective.setdefault("tree_method", "hist")
            # predictor is not required in xgboost >= 2 and can be ignored.
            effective.pop("predictor", None)
        else:
            effective.setdefault("tree_method", "gpu_hist")
            effective.setdefault("predictor", "gpu_predictor")
            effective.pop("device", None)
        # In GPU mode, avoid high host CPU contention from n_jobs=-1.
        effective["n_jobs"] = int(effective.get("n_jobs", 1)) if int(effective.get("n_jobs", 1)) > 0 else 1
    else:
        effective.setdefault("tree_method", "hist")
        effective.setdefault("n_jobs", -1)
        effective.pop("device", None)
        if str(effective.get("predictor", "")).strip().lower() == "gpu_predictor":
            effective.pop("predictor", None)

    return effective

DEFAULT_GP_REGRESSOR_CONFIG = {
    "kernel": "matern52",
    "ard": True,
    "input_standardize": True,
    "target_standardize": True,
    "use_uncertain_input_kernel": True,
    "uncertain_kernel_mc_samples": 64,
    "uncertain_kernel_mc_seed": 0,
    "uncertainty_source_mode": "aggregate_t",
    "uncertainty_summary_dir": None,
    "uncertainty_aggregate_csv": None,
    "learning_rate": 0.01,
    "num_epochs": 250,
    "patience": 20,
    "max_train_size": 5000,
}

DEFAULT_DATA_SPLIT_CONFIG = {
    "random_state": 42,
    "test_size": 0.3,
    "reuse_split": False,
    "split_source": None,
    "split_type": "random",
    "fault_tolerant": False,
    "nan_tolerance": 0.8,
    "min_test_independent": None,
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
    "normalization_path": "../data/output/sensors/normalization.json",
    "use_normalized_thresholds": False,
    "save_plots": True,
}


# ===========================================================================================
# CONFIG LOADING AND MERGING
# ===========================================================================================


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

    # Add training plot toggle if not specified
    if "save_training_plots" not in merged_config:
        merged_config["save_training_plots"] = DEFAULT_COMMON_CONFIG["save_training_plots"]

    if model_type in {"xgb_regressor", "xgb_classifier"}:
        effective_hyper = _resolve_xgb_runtime_hyperparameters(
            merged_config["hyperparameters"],
            merged_config["device"],
        )
        merged_config["hyperparameters"] = effective_hyper
    
    return merged_config


# ===========================================================================================
# DATA LOADING
# ===========================================================================================

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


def _preferred_eval_path_for_key(key: str, data_root: Path) -> Path | None:
    """Return a preferred absolute path for eval defaults under the active data root."""
    root = Path(data_root).resolve()
    if key == "historic_path":
        candidates = [
            root / "Consolidated_sparse.csv",
            root.parent / "regression" / "Consolidated_sparse.csv",
        ]
    elif key == "normalization_path":
        candidates = [
            root.parent / "sensors" / "normalization.json",
            Path(NORMALIZATION_OUTPUT_PATH),
        ]
    elif key == "thresholds_path":
        candidates = [
            root.parent.parent / "input" / "Limits.csv",
        ]
    else:
        return None

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None

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

    input_rows = slice(data_cfg["input_row_1"], data_cfg["input_row_2"])
    input_aggregation = str(data_cfg.get("input_aggregation", "none")).lower()
    
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
        input_aggregation,
        split_cfg.get("min_test_independent"),
    )
    
    return train_samples, test_samples


def write_evaluation_config(config):
    """Write an evaluation config file for f_Evaluate.py next to trained model artifacts."""
    model_type = config["model_type"]
    model_name = config["model_name"]
    data_cfg = config["data"]
    config_dir = config.get("__config_dir", str(Path.cwd()))

    evaluation_cfg = DEFAULT_EVALUATION_CONFIG.copy()
    evaluation_cfg.update(config.get("evaluation", {}))
    evaluation_cfg["evaluate_all"] = False
    if model_type == "xgb_classifier":
        evaluation_cfg["run_regression"] = False
        evaluation_cfg["run_pure_classification"] = True

    data_root = Path(data_cfg["data_dir"]).resolve().parent
    # Preserve explicit normalization path from user config; otherwise prefer active data root.
    if "normalization_path" not in config.get("evaluation", {}):
        preferred_norm = _preferred_eval_path_for_key("normalization_path", data_root)
        evaluation_cfg["normalization_path"] = str(preferred_norm or NORMALIZATION_OUTPUT_PATH)

    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
    os.makedirs(save_path, exist_ok=True)

    # Store config paths relative to evaluation config file location
    relative_data_dir = os.path.relpath(data_cfg["data_dir"], start=save_path)
    for key in ["historic_path", "thresholds_path", "normalization_path"]:
        if evaluation_cfg.get(key):
            preferred = _preferred_eval_path_for_key(key, data_root)
            if preferred is not None and not Path(str(evaluation_cfg[key])).is_absolute():
                abs_path = preferred
            else:
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
    num_workers = max(0, int(hyper_cfg.get("num_workers", 0)))
    pin_memory = bool(hyper_cfg.get("pin_memory", device.type == "cuda")) and (device.type == "cuda")
    persistent_workers = bool(hyper_cfg.get("persistent_workers", True)) and (num_workers > 0)
    loader_kwargs = {
        "batch_size": hyper_cfg["batch_size"],
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(hyper_cfg.get("prefetch_factor", 2)))
    trainloader = DataLoader(train_dataset, **loader_kwargs)
    testloader = DataLoader(test_dataset, **loader_kwargs)
    
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
        "",
        hyper_cfg["corr_lambda"],
        hyper_cfg["corr_eps"],
        hyper_cfg["corr_clip"],
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

    max_train_size = hyper_cfg.get("max_train_size")
    if max_train_size is not None and len(X_train_np) > max_train_size:
        rng = np.random.default_rng(split_cfg["random_state"])
        keep_idx = rng.choice(len(X_train_np), size=max_train_size, replace=False)
        X_train_np = X_train_np[keep_idx]
        y_train_np = y_train_np[keep_idx]
        print(f"Subsampled train set to {max_train_size} for exact GP tractability")

    x_mean = X_train_np.mean(axis=0)
    x_std = X_train_np.std(axis=0)
    x_std[x_std < 1e-8] = 1.0

    input_standardize = bool(hyper_cfg.get("input_standardize", True))
    target_standardize = bool(hyper_cfg.get("target_standardize", True))
    requested_ard = bool(hyper_cfg.get("ard", True))

    if input_standardize:
        X_train_used = (X_train_np - x_mean) / x_std
        X_test_used = (X_test_np - x_mean) / x_std
    else:
        X_train_used = X_train_np
        X_test_used = X_test_np
        x_mean = np.zeros_like(x_mean)
        x_std = np.ones_like(x_std)

    X_train = torch.tensor(X_train_used, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test_used, dtype=torch.float32, device=device)

    kernel_name = str(hyper_cfg.get("kernel", "matern52")).lower()
    use_uncertain_kernel = bool(hyper_cfg.get("use_uncertain_input_kernel", True))
    mc_samples = int(hyper_cfg.get("uncertain_kernel_mc_samples", 64))
    mc_seed = int(hyper_cfg.get("uncertain_kernel_mc_seed", 0))

    if use_uncertain_kernel and not requested_ard:
        print("[WARN] Uncertain-input kernel works best with ARD. Forcing ARD=True for GP kernel.")

    effective_ard = bool(requested_ard or use_uncertain_kernel)
    ard_dims = X_train.shape[1] if effective_ard else None
    input_uncertainty_var = None
    uncertainty_noise_deltas = None
    uncertainty_bundle = {
        "source_mode_effective": "none",
        "source_mode_requested": str(hyper_cfg.get("uncertainty_source_mode", "aggregate_t")).lower(),
        "aggregate_csv_path": None,
        "summary_dir": None,
        "source_details": [],
    }
    if use_uncertain_kernel:
        uncertainty_bundle = _build_feature_uncertainty_bundle(
            data_cfg,
            hyper_cfg,
            config["__config_dir"],
            verbose=True,
        )
        input_uncertainty_var = torch.tensor(
            uncertainty_bundle["feature_variances"],
            dtype=torch.float32,
            device=device
        )
        uncertainty_noise_deltas = torch.tensor(
            uncertainty_bundle["noise_delta_samples"],
            dtype=torch.float32,
            device=device,
        )

    effective_kernel = "uncertain_matern52" if (use_uncertain_kernel and kernel_name == "matern52") else (
        "uncertain_rbf" if (use_uncertain_kernel and kernel_name == "rbf") else kernel_name
    )

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

        if target_standardize:
            y_train_used = (y_train_col - y_mean) / y_std
        else:
            y_train_used = y_train_col
            y_mean = 0.0
            y_std = 1.0

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
        'kernel': kernel_name,
        'ard': requested_ard,
        'effective_ard': effective_ard,
        'ard_num_dims': ard_dims,
        'input_standardize': input_standardize,
        'target_standardize': target_standardize,
        'use_uncertain_input_kernel': use_uncertain_kernel,
        'effective_kernel': effective_kernel,
        'uncertain_kernel_mc_samples': mc_samples,
        'uncertain_kernel_mc_seed': mc_seed,
        'uncertainty_source_mode_requested': uncertainty_bundle.get('source_mode_requested'),
        'uncertainty_source_mode_effective': uncertainty_bundle.get('source_mode_effective'),
        'uncertainty_summary_dir': uncertainty_bundle.get('summary_dir'),
        'uncertainty_aggregate_csv': uncertainty_bundle.get('aggregate_csv_path'),
    }
    write_config(model_config, data_cfg["data_dir"], data_cfg["forecast_name"], "")

    kernel_metadata = {
        "requested_kernel": kernel_name,
        "effective_kernel": effective_kernel,
        "use_uncertain_kernel": use_uncertain_kernel,
        "requested_ard": requested_ard,
        "effective_ard": effective_ard,
        "ard_num_dims": ard_dims,
        "uncertain_kernel_mc_samples": mc_samples,
        "uncertain_kernel_mc_seed": mc_seed,
        "uncertainty_source_mode_requested": uncertainty_bundle.get("source_mode_requested"),
        "uncertainty_source_mode_effective": uncertainty_bundle.get("source_mode_effective"),
        "uncertainty_summary_dir": uncertainty_bundle.get("summary_dir"),
        "uncertainty_aggregate_csv": uncertainty_bundle.get("aggregate_csv_path"),
        "uncertainty_source_details": uncertainty_bundle.get("source_details", []),
    }

    artifact = {
        "artifact_version": 2,
        "model_type": "gp_regressor",
        "hyperparameters": dict(hyper_cfg),
        "input_mean": x_mean,
        "input_std": x_std,
        "input_dim": int(X_train.shape[1]),
        "output_dim": output_dim,
        "models": models_state,
        "kernel_metadata": kernel_metadata,
        "input_uncertainty_var": None if input_uncertainty_var is None else input_uncertainty_var.detach().cpu().numpy(),
        "uncertainty_noise_deltas": None if uncertainty_noise_deltas is None else uncertainty_noise_deltas.detach().cpu().numpy(),
    }
    torch.save(artifact, save_path / "gp_model.pt")
    print(f"\nModel saved to: {save_path / 'gp_model.pt'}")

    if config.get("save_training_plots", True):
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

def _train_xgb_model(config, train_samples, test_samples, model_cls, metric_key, cast_y=None):
    """Shared XGBoost training implementation for regressor and classifier."""
    print("\n" + "="*80)
    print(f"TRAINING {model_cls.__name__.upper()}")
    print("="*80)

    data_cfg = config["data"]
    hyper_cfg = _resolve_xgb_runtime_hyperparameters(config["hyperparameters"], config.get("device", "cpu"))

    X_train = np.ascontiguousarray(np.array([s[0].flatten() for s in train_samples], dtype=np.float32))
    X_test = np.ascontiguousarray(np.array([s[0].flatten() for s in test_samples], dtype=np.float32))
    if cast_y is not None:
        y_train = np.ascontiguousarray(np.array([cast_y(s[1].flatten()[0]) for s in train_samples]))
        y_test = np.ascontiguousarray(np.array([cast_y(s[1].flatten()[0]) for s in test_samples]))
    else:
        y_train = np.ascontiguousarray(np.array([s[1].flatten()[0] for s in train_samples], dtype=np.float32))
        y_test = np.ascontiguousarray(np.array([s[1].flatten()[0] for s in test_samples], dtype=np.float32))

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

    metric = hyper_cfg[metric_key]
    model_kwargs = {
        "tree_method": hyper_cfg["tree_method"],
        "objective": hyper_cfg["objective"],
        "n_estimators": hyper_cfg["n_estimators"],
        "max_depth": hyper_cfg["max_depth"],
        "subsample": hyper_cfg["subsample"],
        "colsample_bytree": hyper_cfg["colsample_bytree"],
        "learning_rate": hyper_cfg["learning_rate"],
        "early_stopping_rounds": hyper_cfg["early_stopping_rounds"],
        "n_jobs": int(hyper_cfg.get("n_jobs", -1)),
    }
    if metric_key == "eval_metric":
        model_kwargs["eval_metric"] = metric
    if "device" in hyper_cfg:
        model_kwargs["device"] = hyper_cfg["device"]
    if "predictor" in hyper_cfg:
        model_kwargs["predictor"] = hyper_cfg["predictor"]

    model = model_cls(**model_kwargs)

    is_gpu_mode = (
        str(hyper_cfg.get("device", "")).lower().startswith("cuda")
        or str(hyper_cfg.get("tree_method", "")).lower() == "gpu_hist"
    )
    print(
        "XGBoost runtime mode: "
        f"{'GPU' if is_gpu_mode else 'CPU'} "
        f"(tree_method={hyper_cfg.get('tree_method')}, "
        f"device={hyper_cfg.get('device', 'n/a')}, "
        f"predictor={hyper_cfg.get('predictor', 'n/a')}, "
        f"xgboost_version={getattr(xgb, '__version__', 'unknown')})"
    )

    print(f"Training {model_cls.__name__}...")
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)

    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
    os.makedirs(save_path, exist_ok=True)
    model.save_model(save_path / "xgboost_model.json")
    print(f"\nModel saved to: {save_path / 'xgboost_model.json'}")

    if config.get("save_training_plots", True):
        results = model.evals_result()
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
# XGBOOST REGRESSOR / CLASSIFIER TRAINING
# ===========================================================================================

def train_xgb_regressor_model(config, train_samples, test_samples):
    """Train XGBoost Regressor model."""
    _train_xgb_model(config, train_samples, test_samples, xgb.XGBRegressor, "metric")


def train_xgb_classifier_model(config, train_samples, test_samples):
    """Train XGBoost Classifier model."""
    _train_xgb_model(
        config, train_samples, test_samples,
        xgb.XGBClassifier, "eval_metric", cast_y=lambda v: int(round(v))
    )


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
    train_samples, test_samples = load_and_split_data(config)
    
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

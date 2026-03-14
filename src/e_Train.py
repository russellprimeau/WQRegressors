"""
Consolidated training script for Transformer, XGBoost Regressor, and XGBoost Classifier models.
Supports configuration via YAML/JSON config files.

Example terminal usage:
python src/e_Train.py --config data/output/regression/MC_pH/config_gp_01.yml
python src/e_Train.py --config data/output/regression/MC_pH/config_transformer_01.yml
python src/e_Train.py --config data/output/regression/MC_pH/config_xgb_01.yml

Notes:
- Optional XGBoost CV tuning (train-set only) is controlled by
  hyperparameters.cv_tuning.enabled. When enabled, tuned hyperparameters are
    reused via dataset-level cache `forecasts/xgb_cv_tuning_cache.json`.
- Trial-level diagnostics are written to `xgb_cv_trials.csv` under the CV
    artifact directory.
- In GPU mode, CV tuning uses a reduced histogram bin count (max_bin=64 by
    default) to limit per-fold QuantileDMatrix GPU memory. Override via
    hyperparameters.cv_tuning.cv_max_bin. This setting is CV-tuning-only and
    does not affect final model training.
- In GPU mode, each CV fold explicitly deletes the XGBoost booster (freeing its
    CUDA QuantileDMatrix memory) and CuPy fold arrays (returning them to the pool
    for reuse). No full pool flush is performed — pool reuse is intentional and
    necessary for GPU throughput.
- CV tuning pre-computes fold index arrays once before the trial loop and
    pre-transfers the full feature matrix X to the GPU once (as a CuPy array).
    This eliminates n_trials × n_folds redundant host-to-device copies and
    list-comprehension rebuilds. y stays as numpy since it is used in scalar
    metric computations alongside numpy prediction outputs.
- In GPU mode, n_jobs is not overridden by the runtime resolver. Users control
    parallelism via hyperparameters.n_jobs (default -1, all cores). CV tuning
    overrides n_jobs per-trial only when parallel_jobs > 1 to prevent contention
    across concurrent fold processes.
- Final model training pre-transfers X_train, X_test, y_train, y_test to GPU
    (CuPy) before model.fit() so that XGBoost builds its internal QuantileDMatrix
    directly from device memory. eval_set_override is not converted since its
    format is controlled by the caller.
"""

import os
import sys
import math
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import yaml
import json
try:
    import optuna
except Exception:
    optuna = None
try:
    import plotly.express as px
except Exception:
    px = None
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
import inspect
try:
    import cupy as cp
except ImportError:
    cp = None
from torch.utils.data import DataLoader
try:
    import gpytorch
except ImportError:
    gpytorch = None
from utils.training import write_config, splitter, _base_sample_id, load_samples
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
from utils.gp_utils import (
    apply_gp_constraints_and_priors,
    build_base_kernel,
    describe_effective_kernel,
    ExactGPRegressor,
)


NORMALIZATION_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "output"
    / "sensors"
    / "normalization.json"
)

# Avoid repeated spam when GPU CV runs without CuPy GPU arrays.
_XGB_GPU_CPU_INPUT_WARNING_EMITTED = False


def _compact_stop_reason_for_plot(stop_reason_text: str) -> str:
    """Shorten stop annotations for plotting without dropping numeric details."""
    compact_text = " ".join(str(stop_reason_text).split())
    replacements = (
        ("Early stopping:", "Early stop:"),
        ("Scheduled stop:", "Scheduled:"),
        ("all configured epochs exhausted", "epochs exhausted"),
        ("all configured boosting rounds exhausted", "rounds exhausted"),
        ("round(s)", "rounds"),
        ("epoch(s)", "epochs"),
    )
    for old, new in replacements:
        compact_text = compact_text.replace(old, new)
    return compact_text


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
    "corr_lambda": 0.1,
    "corr_eps": 1e-8,
    "corr_clip": True,
    "num_workers": 0,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
}

DEFAULT_XGB_CV_TUNING_REGRESSOR = {
    "enabled": False,
    "n_folds": 5,
    "n_trials": 24,
    "seed": 42,
    "parallel_jobs": 1,
    "metric": "rmse",
    "search_method": "random",
    "optuna_sampler": "tpe",
    "optuna_startup_trials": 10,
    "random_start_trials": 10,
    "use_early_stopping": False,
    "early_stopping_rounds": None,
    "verbose": False,
    # Optional scalarized objective for regressors.
    # Preferred: set metric to "rmse_r2" (or "rmse-r2") to enable score = rmse * (1 - r2).
    # Legacy: scalarize_rmse_r2=True with metric=rmse.
    "scalarize_rmse_r2": False,
    # Optional guardrail to avoid degenerate underfit trials.
    # When set, mean CV score is penalized if mean_r2 < min_r2.
    "min_r2": None,
    "r2_penalty": 10.0,
    # GPU-only: reduced histogram bin count for CV tuning to limit QuantileDMatrix GPU
    # memory per fold. Default 64 (vs XGBoost default 256). Does not affect final training.
    "cv_max_bin": 64,
    "param_space": {
        "n_estimators": {"low": 200, "high": 1600, "type": "int"},
        "max_depth": {"low": 2, "high": 9, "type": "int"},
        "min_child_weight": {"low": 1, "high": 10, "type": "int"},
        "gamma": {"low": 0.0, "high": 5.0, "type": "float"},
        "subsample": {"low": 0.5, "high": 1.0, "type": "float"},
        "colsample_bytree": {"low": 0.5, "high": 1.0, "type": "float"},
        "reg_lambda": {"low": 1e-3, "high": 10.0, "type": "log"},
        "reg_alpha": {"low": 1e-4, "high": 5.0, "type": "log"},
        "learning_rate": {"low": 0.003, "high": 0.2, "type": "log"},
    },
}

DEFAULT_XGB_CV_TUNING_CLASSIFIER = {
    **DEFAULT_XGB_CV_TUNING_REGRESSOR,
    "metric": "logloss",
}

DEFAULT_XGB_REGRESSOR_CONFIG = {
    "metric": "rmse",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_estimators": 1100,
    "max_depth": 7,
    "subsample": 0.2,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_lambda": 1,
    "reg_alpha": 0,
    "learning_rate": 0.01,
    "n_jobs": -1,
    "early_stopping_rounds": 200,
    # Optional second stop rule (disabled unless both keys are set in config).
    "train_loss_plateau_rounds": None,
    "train_loss_min_relative_improvement": None,
    "cv_tuning": DEFAULT_XGB_CV_TUNING_REGRESSOR,
}

DEFAULT_XGB_CLASSIFIER_CONFIG = {
    "eval_metric": "logloss",
    "tree_method": "hist",
    "objective": "binary:logistic",
    "n_estimators": 1100,
    "max_depth": 7,
    "subsample": 0.2,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_lambda": 1,
    "reg_alpha": 0,
    "learning_rate": 0.01,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    # Optional second stop rule (disabled unless both keys are set in config).
    "train_loss_plateau_rounds": None,
    "train_loss_min_relative_improvement": None,
    "cv_tuning": DEFAULT_XGB_CV_TUNING_CLASSIFIER,
}


class _TrainLossPlateauCallback(xgb.callback.TrainingCallback):
    """Stop training when rolling-window net train metric improvement plateaus."""

    def __init__(self, metric_name: str, patience_rounds: int, min_relative_improvement: float, state: dict):
        self.metric_name = str(metric_name)
        self.patience_rounds = int(patience_rounds)
        self.min_relative_improvement = float(min_relative_improvement)
        self.state = state
        self.best_metric = None
        self.best_iteration = None
        self.metric_window = []

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        eval0 = evals_log.get("validation_0", {})
        metric_history = eval0.get(self.metric_name)
        if not metric_history:
            return False

        raw_value = metric_history[-1]
        if isinstance(raw_value, (list, tuple)):
            raw_value = raw_value[0]
        try:
            current = float(raw_value)
        except Exception:
            return False
        if not math.isfinite(current):
            return False

        self.metric_window.append(current)
        max_window_len = self.patience_rounds + 1
        if len(self.metric_window) > max_window_len:
            self.metric_window.pop(0)

        if self.best_metric is None:
            self.best_metric = current
            self.best_iteration = int(epoch)
            self.state["best_metric"] = self.best_metric
            self.state["best_iteration"] = self.best_iteration
            return False

        # Keep global best stats for diagnostics, independent of stop-rule semantics.
        if current < self.best_metric:
            self.best_metric = current
            self.best_iteration = int(epoch)
            self.state["best_metric"] = self.best_metric
            self.state["best_iteration"] = self.best_iteration

        # Need one full N-round window to evaluate net improvement from t-N to t.
        if len(self.metric_window) < max_window_len:
            return False

        reference_metric = float(self.metric_window[0])
        denom = max(abs(reference_metric), 1e-12)
        rel_net_improvement = (reference_metric - current) / denom

        self.state["rolling_reference_metric"] = reference_metric
        self.state["rolling_current_metric"] = float(current)
        self.state["rolling_relative_net_improvement"] = float(rel_net_improvement)

        # Trigger when net improvement across the last N rounds is not greater than the threshold.
        if rel_net_improvement <= self.min_relative_improvement:
            self.state["triggered"] = True
            self.state["trigger_iteration"] = int(epoch)
            self.state["stop_reason"] = "train_loss_plateau"
            return True

        return False


def _build_train_loss_plateau_callback(hyper_cfg: dict, metric_name: str):
    """Build optional XGB callback for train-loss plateau stop criterion."""
    patience = hyper_cfg.get("train_loss_plateau_rounds")
    min_rel = hyper_cfg.get("train_loss_min_relative_improvement")

    # Rule is opt-in and only active when both arguments are explicitly provided.
    if patience is None or min_rel is None:
        return None, None

    patience = int(patience)
    min_rel = float(min_rel)
    if patience <= 0:
        raise ValueError(f"hyperparameters.train_loss_plateau_rounds must be > 0, got {patience}")
    if min_rel < 0:
        raise ValueError(
            "hyperparameters.train_loss_min_relative_improvement must be >= 0, "
            f"got {min_rel}"
        )

    state = {
        "enabled": True,
        "triggered": False,
        "trigger_iteration": None,
        "best_metric": None,
        "best_iteration": None,
        "rolling_window_size": patience,
        "rolling_reference_metric": None,
        "rolling_current_metric": None,
        "rolling_relative_net_improvement": None,
        "metric_name": str(metric_name),
        "patience_rounds": patience,
        "min_relative_improvement": min_rel,
    }
    cb = _TrainLossPlateauCallback(str(metric_name), patience, min_rel, state)
    return cb, state


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
        # n_jobs is left as configured. CV tuning overrides it explicitly per-trial
        # when parallel_jobs > 1; final model training benefits from full parallelism.
    else:
        effective.setdefault("tree_method", "hist")
        effective.setdefault("n_jobs", -1)
        effective.pop("device", None)
        if str(effective.get("predictor", "")).strip().lower() == "gpu_predictor":
            effective.pop("predictor", None)

    return effective

DEFAULT_GP_REGRESSOR_CONFIG = {
    "kernel": "matern52+linear",
    "ard": True,
    "input_standardize": True,
    "target_standardize": True,
    "use_uncertain_input_kernel": True,
    "uncertain_kernel_mc_samples": 64,
    "uncertain_kernel_mc_seed": 0,
    "uncertainty_source_mode": "aggregate_t",
    "uncertainty_summary_dir": None,
    "uncertainty_aggregate_csv": None,
    "early_stop_metric": "val_rmse",
    "early_stop_alpha": 0.5,
    "lengthscale_min": 0.05,
    "lengthscale_max": 10.0,
    "noise_min": 1e-5,
    "noise_max": 1.0,
    "outputscale_prior": {"type": "gamma", "shape": 2.0, "rate": 0.15},
    "noise_prior": {"type": "gamma", "shape": 1.5, "rate": 0.5},
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


def _merge_cv_tuning_defaults(hyper_cfg: dict, defaults: dict) -> None:
    """Deep-merge cv_tuning defaults into hyperparameters without mutating defaults."""
    default_cv = defaults.get("cv_tuning")
    if not isinstance(default_cv, dict):
        return

    if "cv_tuning" not in hyper_cfg or hyper_cfg["cv_tuning"] is None:
        hyper_cfg["cv_tuning"] = copy.deepcopy(default_cv)
        return

    if not isinstance(hyper_cfg["cv_tuning"], dict):
        return

    merged = copy.deepcopy(default_cv)
    merged.update(hyper_cfg["cv_tuning"])
    if isinstance(default_cv.get("param_space"), dict) and isinstance(hyper_cfg["cv_tuning"].get("param_space"), dict):
        merged_space = copy.deepcopy(default_cv["param_space"])
        merged_space.update(hyper_cfg["cv_tuning"]["param_space"])
        merged["param_space"] = merged_space

    hyper_cfg["cv_tuning"] = merged


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
            if key == "cv_tuning" and isinstance(default_value, dict):
                print(f"  [DEFAULT] hyperparameters.{key}.enabled = {default_value.get('enabled', False)}")
            else:
                print(f"  [DEFAULT] hyperparameters.{key} = {default_value}")

    if model_type in {"xgb_regressor", "xgb_classifier"}:
        _merge_cv_tuning_defaults(merged_config["hyperparameters"], defaults)
    
    # Add device if not specified
    if "device" not in merged_config:
        merged_config["device"] = DEFAULT_COMMON_CONFIG["device"]
        print(f"  [DEFAULT] device = {merged_config['device']}")
    
    # Add matplotlib backend if not specified
    if "matplotlib_backend" not in merged_config:
        merged_config["matplotlib_backend"] = DEFAULT_COMMON_CONFIG["matplotlib_backend"]

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


def _to_json_safe(value):
    """Convert nested values to JSON-serializable primitives."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        fv = float(value)
        return fv if math.isfinite(fv) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_training_stop_summary(config: dict, summary: dict) -> Path:
    """Persist a standardized training stop-reason artifact next to model files."""
    data_cfg = config["data"]
    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
    save_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": 1,
        "model_type": str(config.get("model_type", "unknown")),
        "model_name": str(config.get("model_name", "")),
        "forecast_name": str(data_cfg.get("forecast_name", "")),
    }
    payload.update(_to_json_safe(summary or {}))
    out_path = save_path / "training_stop_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Training stop summary saved to: {out_path}")
    return out_path


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
    
    stop_summary = train_transformer(
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
    if isinstance(stop_summary, dict):
        _write_training_stop_summary(config, stop_summary)
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

    effective_kernel = describe_effective_kernel(kernel_name, use_uncertain_kernel)

    output_dim = y_train_np.shape[1]
    output_train_losses = []
    output_val_rmse_history = []
    output_rmse = []
    models_state = []
    output_stop_summaries = []

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
        constraint_meta = apply_gp_constraints_and_priors(model, likelihood, hyper_cfg)
        if any(value is not None for value in constraint_meta.values()):
            print(f"[INFO] Applied GP constraints/priors: {constraint_meta}")

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_cfg["learning_rate"])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        losses = []
        val_rmse_list = []
        best_loss = float("inf")
        best_val_rmse = float("inf")
        best_score = float("inf")
        patience_counter = 0
        best_model_state = None
        best_likelihood_state = None
        baseline_train_nll = None
        baseline_val_rmse = None
        best_epoch = None
        early_stop_metric = str(hyper_cfg.get("early_stop_metric", "train_nll")).lower()
        early_stop_alpha = float(hyper_cfg.get("early_stop_alpha", 0.5))
        if early_stop_metric not in {"train_nll", "val_rmse", "mixed"}:
            raise ValueError(
                "hyperparameters.early_stop_metric must be one of "
                "['train_nll', 'val_rmse', 'mixed']"
            )
        if early_stop_metric == "mixed" and not (0.0 <= early_stop_alpha <= 1.0):
            raise ValueError("hyperparameters.early_stop_alpha must be in [0, 1]")

        stop_reason_code = "max_epochs_exhausted"
        stop_epoch = int(hyper_cfg["num_epochs"])
        stop_reason_text = (
            "Scheduled stop: all configured epochs exhausted "
            f"({int(hyper_cfg['num_epochs'])}/{int(hyper_cfg['num_epochs'])} epochs)."
        )

        for epoch in range(hyper_cfg["num_epochs"]):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                loss_value = float("inf")
            losses.append(loss_value)

            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                _pred_val = likelihood(model(X_test)).mean.detach().cpu().numpy()
            val_rmse = float(np.sqrt(np.mean((_pred_val * y_std + y_mean - y_test_col) ** 2)))
            if not math.isfinite(val_rmse):
                val_rmse = float("inf")
            val_rmse_list.append(val_rmse)
            model.train()
            likelihood.train()

            if baseline_train_nll is None:
                baseline_train_nll = loss_value if math.isfinite(loss_value) else 1.0
            if baseline_val_rmse is None:
                baseline_val_rmse = val_rmse if math.isfinite(val_rmse) else 1.0

            if early_stop_metric == "train_nll":
                score = loss_value
            elif early_stop_metric == "val_rmse":
                score = val_rmse
            else:
                train_den = max(baseline_train_nll, 1e-12)
                val_den = max(baseline_val_rmse, 1e-12)
                score = early_stop_alpha * (loss_value / train_den) + (1.0 - early_stop_alpha) * (val_rmse / val_den)

            if score < best_score:
                best_score = score
                best_loss = loss_value
                best_val_rmse = val_rmse
                patience_counter = 0
                best_epoch = int(epoch + 1)
                best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_likelihood_state = {k: v.detach().cpu() for k, v in likelihood.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= hyper_cfg["patience"]:
                    stop_reason_code = "patience_exhausted"
                    stop_epoch = int(epoch + 1)
                    stop_reason_text = (
                        f"Early stopping: no improvement in early_stop_metric='{early_stop_metric}' "
                        f"for {int(hyper_cfg['patience'])} epoch(s); "
                        f"best_score={float(best_score):.6g} at epoch {int(best_epoch or 1)}, "
                        f"stopped at epoch {stop_epoch}/{int(hyper_cfg['num_epochs'])}."
                    )
                    break

        if stop_reason_code == "max_epochs_exhausted":
            stop_epoch = int(len(losses))
            stop_reason_text = (
                "Scheduled stop: all configured epochs exhausted "
                f"({stop_epoch}/{int(hyper_cfg['num_epochs'])} epochs)."
            )

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
        output_val_rmse_history.append(val_rmse_list)
        output_rmse.append(rmse)
        models_state.append({
            "output_index": output_idx,
            "model_state_dict": best_model_state,
            "likelihood_state_dict": best_likelihood_state,
            "target_mean": y_mean,
            "target_std": y_std,
            "train_nll": best_loss,
            "val_rmse_at_best": best_val_rmse,
            "test_rmse": rmse,
            "stop_reason_code": stop_reason_code,
            "stop_reason_text": stop_reason_text,
            "stop_epoch": int(stop_epoch),
            "best_epoch": None if best_epoch is None else int(best_epoch),
        })

        output_stop_summaries.append(
            {
                "output_index": int(output_idx),
                "stop_reason_code": stop_reason_code,
                "stop_reason_text": stop_reason_text,
                "stop_epoch": int(stop_epoch),
                "best_epoch": None if best_epoch is None else int(best_epoch),
                "early_stop_metric": early_stop_metric,
                "patience": int(hyper_cfg["patience"]),
                "num_epochs": int(hyper_cfg["num_epochs"]),
            }
        )

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

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (train_nll, val_rmse) in enumerate(zip(output_train_losses, output_val_rmse_history)):
        color = color_cycle[idx % len(color_cycle)]
        label_suffix = f" (out {idx + 1})" if output_dim > 1 else ""
        ax1.plot(range(1, len(train_nll) + 1), train_nll,
                 color=color, linestyle="-", label=f"Train NLL{label_suffix}")
        if val_rmse:
            ax2.plot(range(1, len(val_rmse) + 1), val_rmse,
                     color=color, linestyle="--", label=f"Val RMSE{label_suffix}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Negative Log Marginal Likelihood")
    ax2.set_ylabel("Validation RMSE")
    ax1.grid(True, ls="--", alpha=0.4)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    if output_stop_summaries:
        annotation_lines = []
        for stop_item in output_stop_summaries[:4]:
            compact_reason = _compact_stop_reason_for_plot(stop_item["stop_reason_text"])
            annotation_lines.append(
                f"out{int(stop_item['output_index']) + 1} [{stop_item['stop_reason_code']}]: {compact_reason}"
            )
        if len(output_stop_summaries) > 4:
            annotation_lines.append(f"... and {len(output_stop_summaries) - 4} more output(s)")
        fig.text(
            0.01,
            0.99,
            "\n".join(annotation_lines),
            fontsize=6.5,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#666666", "boxstyle": "round,pad=0.15"},
        )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    plt.savefig(save_path / "loss_plot.png")
    plt.close()
    print(f"Loss plot saved to: {save_path / 'loss_plot.png'}")

    unique_codes = {item["stop_reason_code"] for item in output_stop_summaries}
    if len(unique_codes) == 1 and output_stop_summaries:
        overall_code = next(iter(unique_codes))
        if len(output_stop_summaries) == 1:
            overall_text = output_stop_summaries[0]["stop_reason_text"]
        else:
            overall_text = (
                f"Per-output stop condition identical ({overall_code}) "
                f"for {len(output_stop_summaries)} output(s)."
            )
    else:
        overall_code = "per_output_mixed"
        overall_text = (
            f"Per-output mixed stop conditions across {len(output_stop_summaries)} output(s)."
            if output_stop_summaries
            else "No output stop summaries captured."
        )

    _write_training_stop_summary(
        config,
        {
            "stop_reason_code": overall_code,
            "stop_reason_text": overall_text,
            "stopped_early": bool(any(item["stop_reason_code"] != "max_epochs_exhausted" for item in output_stop_summaries)),
            "per_output": output_stop_summaries,
            "configured": {
                "num_epochs": int(hyper_cfg["num_epochs"]),
                "patience": int(hyper_cfg["patience"]),
                "early_stop_metric": str(hyper_cfg.get("early_stop_metric", "train_nll")),
                "early_stop_alpha": float(hyper_cfg.get("early_stop_alpha", 0.5)),
            },
        },
    )
    write_evaluation_config(config)


# ===========================================================================================
# XGBOOST REGRESSOR TRAINING
# ===========================================================================================

def _xgb_cv_tuning_enabled(config: dict) -> bool:
    hyper_cfg = config.get("hyperparameters", {}) or {}
    cv_cfg = hyper_cfg.get("cv_tuning", {}) or {}
    return bool(cv_cfg.get("enabled", False))


def _xgb_samples_to_arrays(samples, cast_y=None):
    X = np.ascontiguousarray(np.array([s[0].flatten() for s in samples], dtype=np.float32))
    if cast_y is not None:
        y = np.ascontiguousarray(np.array([cast_y(s[1].flatten()[0]) for s in samples]))
    else:
        y = np.ascontiguousarray(np.array([s[1].flatten()[0] for s in samples], dtype=np.float32))
    names = [str(s[2]) for s in samples]
    return X, y, names


def _resolve_cv_raw_sample_source(config: dict, train_samples) -> dict:
    """Resolve a de-duplicated list of raw sample filenames for CV tuning.

    Reads train_files.txt (preferred) or falls back to filenames carried in
    train_samples.  Strips any _mc_NNN replicate suffix via _base_sample_id
    (which is a no-op for datasets that have no MC replicates) then deduplicates
    with dict.fromkeys to produce one entry per independent raw sample.

    Returns a dict with keys:
        raw_base_ids        : list[str] – ordered unique base filenames
        resolved_files      : list[str] – base IDs confirmed present in samples/
        unresolved_base_ids : list[str] – base IDs not found in samples/
        samples_dir         : Path      – <data_dir>/samples
        name_source         : str       – "train_files_txt" | "train_samples_fallback"
    """
    data_cfg = config["data"]
    data_dir = Path(data_cfg["data_dir"])
    forecast_name = data_cfg["forecast_name"]

    # Prefer train_files.txt written by the outer splitter; it lists the
    # exact set of files (possibly mc_replicates) that belong to the train split.
    train_txt = data_dir / "forecasts" / str(forecast_name) / "train_files.txt"
    if train_txt.is_file():
        source_names = [
            line.strip()
            for line in train_txt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        name_source = "train_files_txt"
    else:
        source_names = [str(s[2]) for s in train_samples]
        name_source = "train_samples_fallback"

    # Collapse to base IDs (strips _mc_NNN suffix; no-op for non-MC names)
    # and deduplicate preserving first-occurrence order.
    raw_base_ids = list(dict.fromkeys(
        _base_sample_id(n) for n in source_names if n.strip()
    ))

    samples_dir = data_dir / "samples"
    if not samples_dir.is_dir():
        print(f"[WARN] CV raw-sample resolution: samples/ directory not found: {samples_dir}")
        return {
            "raw_base_ids": raw_base_ids,
            "resolved_files": [],
            "unresolved_base_ids": raw_base_ids,
            "samples_dir": samples_dir,
            "name_source": name_source,
        }

    existing = {f.name for f in samples_dir.iterdir() if f.suffix == ".csv"}
    resolved = [bid for bid in raw_base_ids if bid in existing]
    unresolved = [bid for bid in raw_base_ids if bid not in existing]
    return {
        "raw_base_ids": raw_base_ids,
        "resolved_files": resolved,
        "unresolved_base_ids": unresolved,
        "samples_dir": samples_dir,
        "name_source": name_source,
    }


def _load_cv_raw_samples(
    config: dict,
    train_samples,
    cast_y=None,
) -> tuple[np.ndarray, np.ndarray, list, dict]:
    """Load raw (non-replicate) samples from <data_dir>/samples/ for CV tuning.

    Delegates resolution to _resolve_cv_raw_sample_source then calls load_samples
    with fault_tolerant=True so that partially-bad raw files are skipped rather
    than aborting the CV run.  Returns (X, y, names, diagnostics).  On failure
    X and y are empty, names is [], and cv_dataset_source is set to
    "train_samples_fallback" so the caller can switch paths cleanly.
    """
    data_cfg = config["data"]
    resolution = _resolve_cv_raw_sample_source(config, train_samples)

    diagnostics = {
        "cv_dataset_source": "raw_samples",
        "name_source": resolution["name_source"],
        "raw_unique_base_count": len(resolution["raw_base_ids"]),
        "raw_resolved_count": len(resolution["resolved_files"]),
        "unresolved_base_ids": resolution["unresolved_base_ids"],
    }

    if resolution["unresolved_base_ids"]:
        print(
            f"[WARN] CV raw-sample resolution: "
            f"{len(resolution['unresolved_base_ids'])} base ID(s) not found in "
            f"{resolution['samples_dir']}; missing: {resolution['unresolved_base_ids']}"
        )

    if not resolution["resolved_files"]:
        print(
            "[WARN] CV raw-sample resolution: no raw samples resolved; "
            "CV will use train_samples instead."
        )
        diagnostics["cv_dataset_source"] = "train_samples_fallback"
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
            diagnostics,
        )

    raw_samples = load_samples(
        directory=resolution["samples_dir"],
        input_columns=data_cfg["input_columns"],
        output_columns=data_cfg["output_columns"],
        input_rows=slice(data_cfg["input_row_1"], data_cfg["input_row_2"]),
        output_rows=data_cfg["output_rows"],
        file_list=resolution["resolved_files"],
        fault_tolerant=True,
        input_aggregation=str(data_cfg.get("input_aggregation", "none")).lower(),
    )

    if not raw_samples:
        print(
            "[WARN] CV raw-sample resolution: load_samples returned no samples; "
            "CV will use train_samples instead."
        )
        diagnostics["cv_dataset_source"] = "train_samples_fallback"
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
            diagnostics,
        )

    X, y, names = _xgb_samples_to_arrays(raw_samples, cast_y=cast_y)
    return X, y, names, diagnostics


def _build_group_folds(file_names: list[str], n_folds: int, seed: int) -> list[list[int]]:
    group_to_indices: dict[str, list[int]] = {}
    for idx, name in enumerate(file_names):
        gid = _base_sample_id(name)
        group_to_indices.setdefault(gid, []).append(idx)

    groups = list(group_to_indices.items())
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)

    n_folds = max(2, min(int(n_folds), len(groups)))
    fold_bins: list[list[int]] = [[] for _ in range(n_folds)]
    fold_sizes = [0 for _ in range(n_folds)]

    for _gid, indices in sorted(groups, key=lambda item: len(item[1]), reverse=True):
        target_fold = int(np.argmin(fold_sizes))
        fold_bins[target_fold].extend(indices)
        fold_sizes[target_fold] += len(indices)

    return [sorted(indices) for indices in fold_bins if indices]


def _sample_cv_param(rng: np.random.Generator, spec, param_name: str):
    if isinstance(spec, dict):
        low = spec.get("low")
        high = spec.get("high")
        mode = str(spec.get("type", "float")).lower()
        if spec.get("log", False):
            mode = "log"
        if low is None or high is None:
            return spec.get("value")
        if mode == "int":
            return int(rng.integers(int(low), int(high) + 1))
        if mode == "log":
            low = max(float(low), 1e-12)
            high = max(float(high), low * 1.000001)
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))
        return float(rng.uniform(float(low), float(high)))

    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in spec):
            low, high = spec
            if param_name in {"max_depth", "min_child_weight"}:
                return int(rng.integers(int(low), int(high) + 1))
            return float(rng.uniform(float(low), float(high)))
        return spec[int(rng.integers(0, len(spec)))]

    return spec


def _optuna_suggest_param(trial, spec, param_name: str):
    if isinstance(spec, dict):
        low = spec.get("low")
        high = spec.get("high")
        mode = str(spec.get("type", "float")).lower()
        if spec.get("log", False):
            mode = "log"
        if low is None or high is None:
            return spec.get("value")
        if mode == "int":
            return trial.suggest_int(param_name, int(low), int(high))
        if mode == "log":
            low = max(float(low), 1e-12)
            high = max(float(high), low * 1.000001)
            return trial.suggest_float(param_name, float(low), float(high), log=True)
        return trial.suggest_float(param_name, float(low), float(high))
    if isinstance(spec, (list, tuple)):
        return trial.suggest_categorical(param_name, list(spec))
    return spec


def _sanitize_xgb_params(params: dict) -> dict:
    out = dict(params)
    if "n_estimators" in out and out["n_estimators"] is not None:
        out["n_estimators"] = int(max(1, int(out["n_estimators"])))
    for key in ("subsample", "colsample_bytree"):
        if key in out and out[key] is not None:
            out[key] = float(min(1.0, max(0.05, float(out[key]))))
    for key in ("reg_lambda", "reg_alpha", "gamma"):
        if key in out and out[key] is not None:
            out[key] = float(max(0.0, float(out[key])))
    if "max_depth" in out and out["max_depth"] is not None:
        out["max_depth"] = int(max(1, int(out["max_depth"])))
    if "min_child_weight" in out and out["min_child_weight"] is not None:
        out["min_child_weight"] = float(max(0.0, float(out["min_child_weight"])))
    return out


def _binary_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-15
    probs = np.clip(y_prob.astype(float), eps, 1 - eps)
    y = y_true.astype(float)
    return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if y.size == 0:
        return float("nan")
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_res = float(np.sum((y - pred) ** 2))
    if not np.isfinite(ss_tot) or ss_tot <= 0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def _xgb_cv_fold_score(
    model_kind: str,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    metric: str,
    use_early_stopping: bool,
    early_stopping_rounds: int | None,
) -> tuple[float, float | None]:
    global _XGB_GPU_CPU_INPUT_WARNING_EMITTED

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    is_gpu_mode = (
        str(model_kwargs.get("device", "")).lower().startswith("cuda")
        or str(model_kwargs.get("tree_method", "")).lower() == "gpu_hist"
    )
    use_gpu_arrays = bool(is_gpu_mode and cp is not None)

    if use_gpu_arrays:
        X_train_fit = cp.asarray(X_train)
        y_train_fit = cp.asarray(y_train)
        X_val_fit = cp.asarray(X_val)
        y_val_fit = cp.asarray(y_val)
    else:
        X_train_fit = X_train
        y_train_fit = y_train
        X_val_fit = X_val
        y_val_fit = y_val

    if model_kind == "classifier":
        model = xgb.XGBClassifier(**model_kwargs)
    else:
        model = xgb.XGBRegressor(**model_kwargs)

    fit_kwargs = {"verbose": False}
    if use_early_stopping:
        fit_kwargs["eval_set"] = [(X_train_fit, y_train_fit), (X_val_fit, y_val_fit)]
        if early_stopping_rounds is not None:
            model.set_params(early_stopping_rounds=int(early_stopping_rounds))

    try:
        model.fit(X_train_fit, y_train_fit, **fit_kwargs)
    except xgb.core.XGBoostError as _gpu_err:
        if not use_gpu_arrays:
            raise
        # GPU OOM fallback: free GPU memory and retry this fold on CPU.
        print(f"[WARN] GPU OOM in CV fold — retrying on CPU: {_gpu_err}")
        del model
        del X_train_fit, y_train_fit, X_val_fit, y_val_fit
        cpu_kwargs = {k: v for k, v in model_kwargs.items()}
        cpu_kwargs["device"] = "cpu"
        cpu_kwargs.pop("tree_method", None)
        if model_kind == "classifier":
            model = xgb.XGBClassifier(**cpu_kwargs)
        else:
            model = xgb.XGBRegressor(**cpu_kwargs)
        X_train_fit, y_train_fit = X_train, y_train
        X_val_fit, y_val_fit = X_val, y_val
        if use_early_stopping:
            fit_kwargs["eval_set"] = [(X_train_fit, y_train_fit), (X_val_fit, y_val_fit)]
        model.fit(X_train_fit, y_train_fit, **fit_kwargs)
        use_gpu_arrays = False

    if model_kind == "classifier":
        if use_gpu_arrays:
            raw_probs = model.get_booster().inplace_predict(X_val_fit, validate_features=False)
            probs = cp.asnumpy(raw_probs).reshape(-1)
        else:
            if is_gpu_mode and (not _XGB_GPU_CPU_INPUT_WARNING_EMITTED):
                print(
                    "[WARN] GPU CV fold scoring is using CPU numpy arrays because CuPy is not available; "
                    "XGBoost may fall back to DMatrix for prediction with extra overhead."
                )
                _XGB_GPU_CPU_INPUT_WARNING_EMITTED = True
            probs = model.predict_proba(X_val)[:, 1]
        score = _binary_logloss(y_val, probs)
        r2 = None
    else:
        if use_gpu_arrays:
            raw_preds = model.get_booster().inplace_predict(X_val_fit, validate_features=False)
            preds = cp.asnumpy(raw_preds).reshape(-1)
        else:
            if is_gpu_mode and (not _XGB_GPU_CPU_INPUT_WARNING_EMITTED):
                print(
                    "[WARN] GPU CV fold scoring is using CPU numpy arrays because CuPy is not available; "
                    "XGBoost may fall back to DMatrix for prediction with extra overhead."
                )
                _XGB_GPU_CPU_INPUT_WARNING_EMITTED = True
            preds = model.predict(X_val)
        score = float(np.sqrt(np.mean(np.square(preds - y_val))))
        r2 = _r2_score(y_val, preds)

    # Free GPU resources immediately: del model releases XGBoost's booster and its
    # CUDA-side QuantileDMatrix memory. del on CuPy arrays returns blocks to the pool
    # for reuse by the next fold without going back to the CUDA allocator.
    del model
    if use_gpu_arrays:
        del X_train_fit, y_train_fit, X_val_fit, y_val_fit

    return score, r2


def _xgb_tune_hyperparameters_cv(
    config: dict,
    train_samples,
    model_kind: str,
    metric_key: str,
    cast_y=None,
) -> tuple[dict, dict]:
    hyper_cfg = _resolve_xgb_runtime_hyperparameters(config["hyperparameters"], config.get("device", "cpu"))
    cv_cfg = hyper_cfg.get("cv_tuning", {}) or {}
    if not cv_cfg.get("enabled", False):
        return {}, {"enabled": False}

    # --- Attempt to build the CV dataset from the original raw samples/ files ---
    # _base_sample_id is idempotent (no-op) when names carry no _mc_NNN suffix,
    # so both MC-replicate and plain-sample datasets are handled correctly.
    X_raw, y_raw, names_raw, raw_diagnostics = _load_cv_raw_samples(
        config, train_samples, cast_y=cast_y
    )
    raw_folds = (
        _build_group_folds(names_raw, int(cv_cfg.get("n_folds", 5)), int(cv_cfg.get("seed", 42)))
        if len(names_raw) >= 2
        else []
    )

    if len(names_raw) >= 2 and len(raw_folds) >= 2:
        X, y, names = X_raw, y_raw, names_raw
        folds = raw_folds
        print(
            f"[INFO] CV tuning dataset: {len(names)} raw samples "
            f"({raw_diagnostics['raw_resolved_count']} resolved from "
            f"{raw_diagnostics['raw_unique_base_count']} unique base IDs, "
            f"source={raw_diagnostics['name_source']})"
        )
    else:
        # Fall back to the full train_samples set (the behaviour prior to this change).
        if raw_diagnostics.get("cv_dataset_source") != "train_samples_fallback":
            print(
                f"[WARN] CV raw-sample dataset insufficient for folding "
                f"(resolved={len(names_raw)}, folds={len(raw_folds)}); "
                f"falling back to train_samples."
            )
        raw_diagnostics["cv_dataset_source"] = "train_samples_fallback"
        X, y, names = _xgb_samples_to_arrays(train_samples, cast_y=cast_y)
        if len(names) < 2:
            print("[WARN] CV tuning skipped: not enough training samples.")
            return {}, {"enabled": False, "reason": "insufficient_samples"}
        folds = _build_group_folds(names, int(cv_cfg.get("n_folds", 5)), int(cv_cfg.get("seed", 42)))

    if len(folds) < 2:
        print("[WARN] CV tuning skipped: not enough independent groups for folds.")
        return {}, {"enabled": False, "reason": "insufficient_groups"}

    param_space = cv_cfg.get("param_space") or {}
    rng = np.random.default_rng(int(cv_cfg.get("seed", 42)))
    n_trials = int(max(1, cv_cfg.get("n_trials", 10)))

    base_kwargs = {
        "tree_method": hyper_cfg["tree_method"],
        "objective": hyper_cfg["objective"],
        "n_estimators": hyper_cfg["n_estimators"],
        "max_depth": hyper_cfg["max_depth"],
        "subsample": hyper_cfg["subsample"],
        "colsample_bytree": hyper_cfg["colsample_bytree"],
        "min_child_weight": hyper_cfg["min_child_weight"],
        "gamma": hyper_cfg["gamma"],
        "reg_lambda": hyper_cfg["reg_lambda"],
        "reg_alpha": hyper_cfg["reg_alpha"],
        "learning_rate": hyper_cfg["learning_rate"],
        "n_jobs": int(hyper_cfg.get("n_jobs", -1)),
    }
    if metric_key == "eval_metric":
        base_kwargs["eval_metric"] = hyper_cfg[metric_key]
    if "device" in hyper_cfg:
        base_kwargs["device"] = hyper_cfg["device"]
    if "predictor" in hyper_cfg:
        base_kwargs["predictor"] = hyper_cfg["predictor"]

    is_gpu_mode = (
        str(hyper_cfg.get("device", "")).lower().startswith("cuda")
        or str(hyper_cfg.get("tree_method", "")).lower() == "gpu_hist"
    )
    # Reduce histogram bin count for CV tuning in GPU mode to lower QuantileDMatrix
    # memory footprint. cv_max_bin defaults to 64 (vs XGBoost default of 256).
    # Only applies to CV tuning; final training uses the config's max_bin unchanged.
    if is_gpu_mode:
        base_kwargs["max_bin"] = int(cv_cfg.get("cv_max_bin", 64))
    parallel_jobs = int(cv_cfg.get("parallel_jobs", 1))
    if is_gpu_mode and parallel_jobs > 1:
        print("[WARN] CV tuning parallel_jobs > 1 is not supported in GPU mode; forcing to 1.")
        parallel_jobs = 1

    use_early_stopping = bool(cv_cfg.get("use_early_stopping", False))
    early_stop_rounds = cv_cfg.get("early_stopping_rounds")

    if not use_early_stopping:
        base_kwargs["early_stopping_rounds"] = None

    metric_raw = str(cv_cfg.get("metric", "rmse")).lower()
    scalarized_metric_aliases = {"rmse_r2", "rmse-r2", "rmse*r2", "rmse_r2_scalar"}
    use_scalarized_rmse_r2 = bool(
        model_kind == "regressor"
        and (
            metric_raw in scalarized_metric_aliases
            or (metric_raw == "rmse" and cv_cfg.get("scalarize_rmse_r2", False))
        )
    )
    metric = "rmse" if metric_raw in scalarized_metric_aliases else metric_raw
    verbose = bool(cv_cfg.get("verbose", False))
    search_method = str(cv_cfg.get("search_method", "random")).lower()
    r2_min = cv_cfg.get("min_r2", cv_cfg.get("r2_guardrail_min"))
    r2_penalty = cv_cfg.get("r2_penalty")
    if r2_penalty is None:
        r2_penalty = 10.0 if r2_min is not None else 0.0
    trial_results: list[dict] = []
    best_score = float("inf")
    best_params: dict = {}

    # Pre-compute fold index arrays once; they are constant across all trials.
    # This avoids rebuilding them via list comprehension for every trial × fold.
    precomputed_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for _fi in range(len(folds)):
        _val_idx = np.array(folds[_fi], dtype=int)
        _train_idx = np.array(
            [i for _fj, _f in enumerate(folds) if _fj != _fi for i in _f],
            dtype=int,
        )
        precomputed_folds.append((_train_idx, _val_idx))

    # Pre-transfer X to GPU once if in GPU mode. cp.asarray() on a CuPy slice
    # inside _xgb_cv_fold_score is then a no-op, eliminating n_trials × n_folds
    # host-to-device copies of the full feature matrix. y stays as numpy because it
    # is used in scalar metric computations alongside numpy prediction outputs.
    if is_gpu_mode and cp is not None:
        try:
            X_cv = cp.asarray(X)
        except Exception as _cp_err:
            print(
                f"[WARN] GPU pre-transfer of feature matrix failed "
                f"({type(_cp_err).__name__}); falling back to CPU for all CV folds."
            )
            X_cv = X
            is_gpu_mode = False
            base_kwargs["device"] = "cpu"
            base_kwargs["tree_method"] = "hist"
            base_kwargs.pop("predictor", None)
    else:
        X_cv = X

    executor = None
    try:
        if parallel_jobs > 1:
            executor = ProcessPoolExecutor(max_workers=parallel_jobs)

        if search_method == "optuna":
            if optuna is None:
                raise RuntimeError("Optuna is not installed but search_method='optuna' was requested.")
            sampler_name = str(cv_cfg.get("optuna_sampler", "tpe")).lower()
            startup_trials = int(cv_cfg.get("random_start_trials", cv_cfg.get("optuna_startup_trials", 10)))
            if sampler_name == "cmaes":
                sampler = optuna.samplers.CmaEsSampler(seed=int(cv_cfg.get("seed", 42)))
            else:
                sampler = optuna.samplers.TPESampler(
                    seed=int(cv_cfg.get("seed", 42)),
                    n_startup_trials=max(0, startup_trials),
                )

            def _objective(trial):
                sampled = {}
                for param_name, spec in param_space.items():
                    sampled[param_name] = _optuna_suggest_param(trial, spec, param_name)
                sampled = _sanitize_xgb_params(sampled)

                trial_kwargs = dict(base_kwargs)
                trial_kwargs.update(sampled)
                if parallel_jobs > 1:
                    trial_kwargs["n_jobs"] = 1

                fold_scores: list[float] = []
                fold_r2: list[float] = []
                if executor is not None:
                    futures = []
                    for train_idx, val_idx in precomputed_folds:
                        futures.append(
                            executor.submit(
                                _xgb_cv_fold_score,
                                model_kind,
                                trial_kwargs,
                                X_cv,
                                y,
                                train_idx,
                                val_idx,
                                metric,
                                use_early_stopping,
                                early_stop_rounds,
                            )
                        )
                    for future in as_completed(futures):
                        try:
                            score, r2 = future.result()
                        except Exception as _fold_err:
                            print(f"[WARN] Parallel CV fold worker failed: {_fold_err}")
                            continue
                        fold_scores.append(float(score))
                        if r2 is not None and np.isfinite(r2):
                            fold_r2.append(float(r2))
                else:
                    for train_idx, val_idx in precomputed_folds:
                        score, r2 = _xgb_cv_fold_score(
                            model_kind,
                            trial_kwargs,
                            X_cv,
                            y,
                            train_idx,
                            val_idx,
                            metric,
                            use_early_stopping,
                            early_stop_rounds,
                        )
                        fold_scores.append(float(score))
                        if r2 is not None and np.isfinite(r2):
                            fold_r2.append(float(r2))

                mean_rmse = float(np.mean(fold_scores)) if fold_scores else float("inf")
                std_score = float(np.std(fold_scores)) if fold_scores else float("nan")
                mean_r2 = float(np.mean(fold_r2)) if fold_r2 else float("nan")
                std_r2 = float(np.std(fold_r2)) if fold_r2 else float("nan")
                if use_scalarized_rmse_r2 and np.isfinite(mean_r2):
                    mean_score = mean_rmse * (1.0 - mean_r2)
                else:
                    mean_score = mean_rmse
                if r2_min is not None and np.isfinite(mean_r2):
                    deficit = float(r2_min) - mean_r2
                    if deficit > 0:
                        mean_score += float(r2_penalty) * deficit
                trial.set_user_attr("std_score", std_score)
                trial.set_user_attr("mean_r2", mean_r2)
                trial.set_user_attr("std_r2", std_r2)
                trial.set_user_attr("mean_rmse", mean_rmse)
                trial.set_user_attr("params", sampled)
                if verbose:
                    print(
                        f"[CV] Trial {trial.number + 1}/{n_trials} "
                        f"mean={mean_score:.6f} std={std_score:.6f} "
                        f"mean_rmse={mean_rmse:.6f} "
                        f"mean_r2={mean_r2:.6f} params={sampled}"
                    )
                return mean_score

            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(_objective, n_trials=n_trials, catch=(xgb.core.XGBoostError,))
            try:
                best_params = dict(study.best_trial.user_attrs.get("params", {}))
                best_score = float(study.best_value)
            except ValueError:
                print(
                    "[WARN] All Optuna trials failed (e.g. all GPU OOM); "
                    "no best trial available — returning empty hyperparameters."
                )
                best_params = {}
                best_score = float("inf")
            for t in study.trials:
                trial_results.append(
                    {
                        "trial": int(t.number + 1),
                        "mean_score": float(t.value) if t.value is not None else float("nan"),
                        "std_score": float(t.user_attrs.get("std_score", float("nan"))),
                        "mean_r2": float(t.user_attrs.get("mean_r2", float("nan"))),
                        "std_r2": float(t.user_attrs.get("std_r2", float("nan"))),
                        "mean_rmse": float(t.user_attrs.get("mean_rmse", float("nan"))),
                        "params": dict(t.user_attrs.get("params", {})),
                    }
                )
        else:
            for trial_idx in range(n_trials):
                sampled = {}
                for param_name, spec in param_space.items():
                    sampled[param_name] = _sample_cv_param(rng, spec, param_name)
                sampled = _sanitize_xgb_params(sampled)

                trial_kwargs = dict(base_kwargs)
                trial_kwargs.update(sampled)
                if parallel_jobs > 1:
                    trial_kwargs["n_jobs"] = 1

                fold_scores: list[float] = []
                fold_r2: list[float] = []
                if executor is not None:
                    futures = []
                    for train_idx, val_idx in precomputed_folds:
                        futures.append(
                            executor.submit(
                                _xgb_cv_fold_score,
                                model_kind,
                                trial_kwargs,
                                X_cv,
                                y,
                                train_idx,
                                val_idx,
                                metric,
                                use_early_stopping,
                                early_stop_rounds,
                            )
                        )
                    for future in as_completed(futures):
                        try:
                            score, r2 = future.result()
                        except Exception as _fold_err:
                            print(f"[WARN] Parallel CV fold worker failed: {_fold_err}")
                            continue
                        fold_scores.append(float(score))
                        if r2 is not None and np.isfinite(r2):
                            fold_r2.append(float(r2))
                else:
                    for train_idx, val_idx in precomputed_folds:
                        score, r2 = _xgb_cv_fold_score(
                            model_kind,
                            trial_kwargs,
                            X_cv,
                            y,
                            train_idx,
                            val_idx,
                            metric,
                            use_early_stopping,
                            early_stop_rounds,
                        )
                        fold_scores.append(float(score))
                        if r2 is not None and np.isfinite(r2):
                            fold_r2.append(float(r2))

                mean_rmse = float(np.mean(fold_scores)) if fold_scores else float("inf")
                std_score = float(np.std(fold_scores)) if fold_scores else float("nan")
                mean_r2 = float(np.mean(fold_r2)) if fold_r2 else float("nan")
                std_r2 = float(np.std(fold_r2)) if fold_r2 else float("nan")
                if use_scalarized_rmse_r2 and np.isfinite(mean_r2):
                    mean_score = mean_rmse * (1.0 - mean_r2)
                else:
                    mean_score = mean_rmse
                if r2_min is not None and np.isfinite(mean_r2):
                    deficit = float(r2_min) - mean_r2
                    if deficit > 0:
                        mean_score += float(r2_penalty) * deficit
                trial_results.append(
                    {
                        "trial": int(trial_idx + 1),
                        "mean_score": mean_score,
                        "std_score": std_score,
                        "mean_r2": mean_r2,
                        "std_r2": std_r2,
                        "mean_rmse": mean_rmse,
                        "params": dict(sampled),
                    }
                )
                if mean_score < best_score:
                    best_score = mean_score
                    best_params = dict(sampled)
                if verbose:
                    print(
                        f"[CV] Trial {trial_idx + 1}/{n_trials} "
                        f"mean={mean_score:.6f} std={std_score:.6f} "
                        f"mean_r2={mean_r2:.6f} params={sampled}"
                    )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    summary = {
        "enabled": True,
        "metric": metric,
        "n_folds": int(len(folds)),
        "n_trials": int(n_trials),
        "seed": int(cv_cfg.get("seed", 42)),
        "search_method": search_method,
        "use_early_stopping": bool(use_early_stopping),
        "early_stopping_rounds": None if early_stop_rounds is None else int(early_stop_rounds),
        "best_score": float(best_score),
        "trials": trial_results,
        "param_space": param_space,
        "score_definition": "rmse*(1-r2)" if use_scalarized_rmse_r2 else metric_raw,
        "r2_guardrail_min": None if r2_min is None else float(r2_min),
        "r2_penalty": float(r2_penalty),
        "cv_dataset_source": raw_diagnostics.get("cv_dataset_source", "unknown"),
        "name_source": raw_diagnostics.get("name_source", "unknown"),
        "raw_unique_base_count": raw_diagnostics.get("raw_unique_base_count", 0),
        "raw_resolved_count": raw_diagnostics.get("raw_resolved_count", 0),
        "unresolved_base_ids": raw_diagnostics.get("unresolved_base_ids", []),
        "gpu_mode": bool(is_gpu_mode),
        "gpu_array_backend": (
            "cupy" if bool(is_gpu_mode and cp is not None) else
            ("numpy_cpu_inputs" if bool(is_gpu_mode) else "numpy_cpu")
        ),
    }
    return best_params, summary


def _resolve_cv_artifact_dir(config: dict) -> Path:
    hyper_cfg = config.get("hyperparameters", {}) or {}
    cv_cfg = hyper_cfg.get("cv_tuning", {}) or {}
    artifact_dir = cv_cfg.get("artifact_dir")
    data_cfg = config.get("data", {}) or {}
    if artifact_dir:
        cfg_dir = config.get("__config_dir", str(Path.cwd()))
        return Path(_resolve_path_from_config(str(artifact_dir), cfg_dir))
    return Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])


def _resolve_cv_cache_path(config: dict) -> Path:
    hyper_cfg = config.get("hyperparameters", {}) or {}
    cv_cfg = hyper_cfg.get("cv_tuning", {}) or {}
    cache_path = cv_cfg.get("cache_path")
    data_cfg = config.get("data", {}) or {}
    if cache_path:
        cfg_dir = config.get("__config_dir", str(Path.cwd()))
        return Path(_resolve_path_from_config(str(cache_path), cfg_dir))
    return Path(data_cfg["data_dir"], "forecasts", "xgb_cv_tuning_cache.json")


def _write_xgb_cv_tuning_cache(cache_path: Path, hyper_cfg: dict, best_params: dict, summary: dict) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tuned_hyper = dict(hyper_cfg)
    tuned_hyper.update(best_params)
    payload = {
        "artifact_version": 1,
        "tuned_hyperparameters": tuned_hyper,
        "best_params": best_params,
        "cv_summary": summary,
        "scope": "dataset_cache",
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return cache_path


def run_xgb_cv_tuning_only(
    config: dict,
    train_samples,
    model_kind: str,
    metric_key: str,
    cast_y=None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> tuple[dict, dict]:
    """Run XGB CV tuning only (no final model training). Returns (best_params, summary)."""
    hyper_cfg = _resolve_xgb_runtime_hyperparameters(config["hyperparameters"], config.get("device", "cpu"))
    cv_cfg = hyper_cfg.get("cv_tuning", {}) or {}
    cv_requested = bool(cv_cfg.get("enabled", False))
    data_cfg = config["data"]

    best_params: dict = {}
    summary: dict = {"enabled": False}
    tuning_ran = False
    tuning_ran = False

    cache_path = _resolve_cv_cache_path(config)
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_params = cached.get("tuned_hyperparameters") or cached.get("best_params")
            if isinstance(cached_params, dict) and cached_params:
                best_params = dict(cached_params)
                summary = cached.get("cv_summary", {}) or {"enabled": False}
                summary["source"] = "cache"
                print(f"[INFO] Using cached CV hyperparameters from {cache_path}")
        except Exception as exc:
            print(f"[WARN] Failed to read CV cache {cache_path}: {exc}")

    if not best_params:
        best_params, summary = _xgb_tune_hyperparameters_cv(
            config=config,
            train_samples=train_samples,
            model_kind=model_kind,
            metric_key=metric_key,
            cast_y=cast_y,
        )
        tuning_ran = True
        if cv_requested and best_params and write_cache:
            cache_written = _write_xgb_cv_tuning_cache(cache_path, hyper_cfg, best_params, summary)
            print(f"[INFO] CV tuning cache saved to: {cache_written}")

    if tuning_ran:
        trials_csv = _write_xgb_cv_trials_csv(config, summary)
        if trials_csv is not None:
            print(f"[INFO] CV tuning trials CSV saved to: {trials_csv}")
        trials_plot = _write_xgb_cv_trials_plot(config, summary)
        if trials_plot is not None:
            print(f"[INFO] CV tuning trials plot saved to: {trials_plot}")
        parallel_plot = _write_xgb_cv_parallel_coords_plot(config, summary, top_k=20)
        if parallel_plot is not None:
            print(f"[INFO] CV tuning parallel-coordinates plot saved to: {parallel_plot}")
        parallel_plot = _write_xgb_cv_parallel_coords_plot(config, summary, top_k=20)
        if parallel_plot is not None:
            print(f"[INFO] CV tuning parallel-coordinates plot saved to: {parallel_plot}")

    return best_params, summary


def _write_xgb_cv_trials_csv(config: dict, summary: dict) -> Path | None:
    trials = summary.get("trials") if isinstance(summary, dict) else None
    if not isinstance(trials, list) or not trials:
        return None
    forecast_dir = _resolve_cv_artifact_dir(config)
    forecast_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(trials)
    out_path = forecast_dir / "xgb_cv_trials.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _write_xgb_cv_trials_plot(config: dict, summary: dict) -> Path | None:
    trials = summary.get("trials") if isinstance(summary, dict) else None
    if not isinstance(trials, list) or not trials:
        return None
    df = pd.DataFrame(trials)
    if df.empty or "trial" not in df.columns or "mean_score" not in df.columns:
        return None
    forecast_dir = _resolve_cv_artifact_dir(config)
    forecast_dir.mkdir(parents=True, exist_ok=True)
    x = pd.to_numeric(df["trial"], errors="coerce")
    y = pd.to_numeric(df["mean_score"], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None
    x = x[valid]
    y = y[valid]
    best_idx = int(y.idxmin()) if not y.empty else None
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, marker="o", linestyle="-", label="Mean CV score")
    if best_idx is not None and best_idx in df.index:
        plt.scatter([df.loc[best_idx, "trial"]], [df.loc[best_idx, "mean_score"]], color="#d62728", zorder=3, label="Best")
    plt.xlabel("Trial")
    plt.ylabel("Mean CV score")
    plt.grid(True, ls="--", alpha=0.4)
    plt.title("XGB CV tuning: mean score by trial")
    plt.legend()
    out_path = forecast_dir / "xgb_cv_trials.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def _write_xgb_cv_parallel_coords_plot(config: dict, summary: dict, top_k: int = 20) -> Path | None:
    trials = summary.get("trials") if isinstance(summary, dict) else None
    if not isinstance(trials, list) or not trials:
        return None
    df = pd.DataFrame(trials)
    if df.empty or "mean_score" not in df.columns or "params" not in df.columns:
        return None
    df = df.sort_values("mean_score", ascending=True, kind="stable").head(int(top_k))
    if df.empty:
        return None

    params_expanded = pd.json_normalize(df["params"])
    params_expanded = params_expanded.apply(pd.to_numeric, errors="coerce")
    plot_df = pd.concat([df[["mean_score"]].reset_index(drop=True), params_expanded.reset_index(drop=True)], axis=1)
    plot_df = plot_df.dropna(axis=1, how="all")
    if plot_df.shape[1] <= 1:
        return None

    # Normalize to [0,1] for parallel coordinates.
    norm_df = plot_df.copy()
    for col in norm_df.columns:
        col_vals = pd.to_numeric(norm_df[col], errors="coerce")
        col_min = float(col_vals.min()) if col_vals.notna().any() else 0.0
        col_max = float(col_vals.max()) if col_vals.notna().any() else 1.0
        if col_max - col_min > 0:
            norm_df[col] = (col_vals - col_min) / (col_max - col_min)
        else:
            norm_df[col] = 0.5

    forecast_dir = _resolve_cv_artifact_dir(config)
    forecast_dir.mkdir(parents=True, exist_ok=True)
    out_path = forecast_dir / "xgb_cv_top_trials_parallel_coords.png"

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * norm_df.shape[1]), 4.8))
    x = np.arange(len(norm_df.columns))
    cmap = plt.cm.viridis
    scores = pd.to_numeric(df["mean_score"], errors="coerce").to_numpy()
    if np.all(~np.isfinite(scores)):
        scores = np.linspace(0, 1, len(norm_df))
    score_min = np.nanmin(scores)
    score_max = np.nanmax(scores)
    denom = (score_max - score_min) if score_max > score_min else 1.0
    colors = cmap((scores - score_min) / denom)

    for idx, row in norm_df.iterrows():
        ax.plot(x, row.to_numpy(dtype=float), color=colors[idx], alpha=0.8, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(norm_df.columns, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized value")
    ax.set_title(f"XGB CV tuning: top-{min(len(norm_df), int(top_k))} trials")
    ax.grid(axis="y", alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=score_min, vmax=score_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Mean CV score")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path

def _train_xgb_model(
    config,
    train_samples,
    test_samples,
    model_cls,
    metric_key,
    cast_y=None,
    eval_set_override=None,
    disable_early_stopping: bool = False,
):
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
        'min_child_weight': hyper_cfg["min_child_weight"],
        'gamma': hyper_cfg["gamma"],
        'reg_lambda': hyper_cfg["reg_lambda"],
        'reg_alpha': hyper_cfg["reg_alpha"],
    }
    write_config(model_config, data_cfg["data_dir"], data_cfg["forecast_name"], "")

    metric = hyper_cfg[metric_key]
    plateau_callback, plateau_state = _build_train_loss_plateau_callback(hyper_cfg, metric)
    model_kwargs = {
        "tree_method": hyper_cfg["tree_method"],
        "objective": hyper_cfg["objective"],
        "n_estimators": hyper_cfg["n_estimators"],
        "max_depth": hyper_cfg["max_depth"],
        "subsample": hyper_cfg["subsample"],
        "colsample_bytree": hyper_cfg["colsample_bytree"],
        "min_child_weight": hyper_cfg["min_child_weight"],
        "gamma": hyper_cfg["gamma"],
        "reg_lambda": hyper_cfg["reg_lambda"],
        "reg_alpha": hyper_cfg["reg_alpha"],
        "learning_rate": hyper_cfg["learning_rate"],
        "n_jobs": int(hyper_cfg.get("n_jobs", -1)),
    }
    if hyper_cfg.get("early_stopping_rounds") is not None:
        model_kwargs["early_stopping_rounds"] = hyper_cfg["early_stopping_rounds"]
    if disable_early_stopping:
        model_kwargs.pop("early_stopping_rounds", None)
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

    # Pre-transfer training data to GPU once so that XGBoost builds its internal
    # QuantileDMatrix directly from device memory instead of copying from host.
    # eval_set_override is not converted since its format is caller-controlled.
    if is_gpu_mode and cp is not None:
        X_train_fit = cp.asarray(X_train)
        y_train_fit = cp.asarray(y_train)
        X_test_fit = cp.asarray(X_test)
        y_test_fit = cp.asarray(y_test)
    else:
        X_train_fit, y_train_fit = X_train, y_train
        X_test_fit, y_test_fit = X_test, y_test

    print(f"Training {model_cls.__name__}...")
    fit_kwargs = {
        "verbose": True,
    }
    if eval_set_override is not None:
        fit_kwargs["eval_set"] = eval_set_override
    else:
        fit_kwargs["eval_set"] = [(X_train_fit, y_train_fit), (X_test_fit, y_test_fit)]

    supports_callbacks = "callbacks" in inspect.signature(model.fit).parameters

    if plateau_callback is not None:
        if supports_callbacks:
            fit_kwargs["callbacks"] = [plateau_callback]
        else:
            print(
                "[WARN] train-loss plateau stop requested but this xgboost sklearn "
                "version does not support fit(callbacks=...). Skipping plateau rule."
            )
    model.fit(X_train_fit, y_train_fit, **fit_kwargs)

    if plateau_state and plateau_state.get("triggered"):
        print(
            "Additional stop rule triggered: "
            f"training loss plateau at boosting round {plateau_state.get('trigger_iteration')}"
        )
    elif plateau_state and plateau_state.get("enabled"):
        print("Additional stop rule enabled but did not trigger before training end.")

    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
    os.makedirs(save_path, exist_ok=True)
    model.save_model(save_path / "xgboost_model.json")
    print(f"\nModel saved to: {save_path / 'xgboost_model.json'}")

    results = model.evals_result()
    train_series = results.get("validation_0", {}).get(metric)
    val_series = results.get("validation_1", {}).get(metric)
    executed_rounds = int(len(train_series)) if train_series else 0
    best_iteration = getattr(model, "best_iteration", None)
    configured_early_stopping_rounds = model_kwargs.get("early_stopping_rounds")
    if (
        configured_early_stopping_rounds is not None
        and best_iteration is not None
        and executed_rounds <= 0
    ):
        executed_rounds = int(best_iteration) + 1

    if plateau_state and plateau_state.get("triggered"):
        trigger_round = int(plateau_state.get("trigger_iteration", -1)) + 1
        observed_rel_net = plateau_state.get("rolling_relative_net_improvement")
        if observed_rel_net is None:
            observed_rel_net = float("nan")
        stop_reason_code = "train_loss_plateau"
        stop_reason_text = (
            "Early stopping: rolling-window net relative improvement in training "
            f"{metric} over the last {int(plateau_state.get('patience_rounds', 0))} round(s) "
            f"was {float(observed_rel_net):.6g}, which is not greater than "
            f"{float(plateau_state.get('min_relative_improvement', 0.0)):.6g}; "
            f"triggered at round {trigger_round}."
        )
    elif (
        configured_early_stopping_rounds is not None
        and best_iteration is not None
        and executed_rounds > 0
        and executed_rounds < int(hyper_cfg["n_estimators"])
    ):
        stop_reason_code = "validation_early_stopping"
        stop_reason_text = (
            f"Early stopping: no decrease in best validation {metric} for "
            f"{int(configured_early_stopping_rounds)} round(s); "
            f"best_iteration={int(best_iteration) + 1}, rounds_executed={executed_rounds}/{int(hyper_cfg['n_estimators'])}."
        )
    else:
        stop_reason_code = "n_estimators_exhausted"
        rounds_value = executed_rounds if executed_rounds > 0 else int(hyper_cfg["n_estimators"])
        stop_reason_text = (
            "Scheduled stop: all configured boosting rounds exhausted "
            f"({rounds_value}/{int(hyper_cfg['n_estimators'])} rounds)."
        )

    if train_series:
        epochs = len(train_series)
        plt.figure(figsize=(8, 5))
        plt.loglog(range(epochs), train_series, label='Training Loss')
        if val_series:
            plt.loglog(range(len(val_series)), val_series, label='Validation Loss')
        if hasattr(model, "best_iteration") and getattr(model, "best_iteration") is not None:
            plt.axvline(
                float(model.best_iteration),
                color="#d62728",
                linestyle="--",
                linewidth=1.0,
                label=f"Validation stop/best iter ({model.best_iteration})",
            )
        if plateau_state and plateau_state.get("triggered") and plateau_state.get("trigger_iteration") is not None:
            plt.axvline(
                float(plateau_state["trigger_iteration"]),
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.0,
                label=f"Train plateau stop ({plateau_state['trigger_iteration']})",
            )
        plt.xlabel('Boosting Rounds')
        plt.ylabel(metric)
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.gcf().text(
            0.01,
            0.99,
            _compact_stop_reason_for_plot(stop_reason_text),
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#666666", "boxstyle": "round,pad=0.15"},
        )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
        plt.savefig(save_path / "loss_plot.png")
        plt.close()
        print(f"Loss plot saved to: {save_path / 'loss_plot.png'}")
    else:
        print("[WARN] Training plot skipped: no evaluation history found.")

    _write_training_stop_summary(
        config,
        {
            "stop_reason_code": stop_reason_code,
            "stop_reason_text": stop_reason_text,
            "stopped_early": bool(stop_reason_code != "n_estimators_exhausted"),
            "configured": {
                "n_estimators": int(hyper_cfg["n_estimators"]),
                "early_stopping_rounds": (
                    None if configured_early_stopping_rounds is None else int(configured_early_stopping_rounds)
                ),
                "metric": str(metric),
            },
            "observed": {
                "best_iteration": None if best_iteration is None else int(best_iteration) + 1,
                "executed_rounds": int(executed_rounds),
                "plateau_state": plateau_state,
            },
        },
    )
    write_evaluation_config(config)
    return {
        "stop_reason_code": stop_reason_code,
        "stop_reason_text": stop_reason_text,
    }


def _train_xgb_model_cv_tuned(
    config,
    train_samples,
    test_samples,
    model_cls,
    metric_key,
    model_kind: str,
    cast_y=None,
):
    hyper_cfg = _resolve_xgb_runtime_hyperparameters(config["hyperparameters"], config.get("device", "cpu"))
    cv_cfg = hyper_cfg.get("cv_tuning", {}) or {}
    cv_requested = bool(cv_cfg.get("enabled", False))

    data_cfg = config["data"]
    cache_path = _resolve_cv_cache_path(config)
    best_params = {}
    summary = {"enabled": False}
    tuning_ran = False
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_params = cached.get("tuned_hyperparameters") or cached.get("best_params")
            if isinstance(cached_params, dict) and cached_params:
                best_params = dict(cached_params)
                summary = cached.get("cv_summary", {}) or {"enabled": False}
                summary["source"] = "cache"
                print(f"[INFO] Using cached CV hyperparameters from {cache_path}")
        except Exception as exc:
            print(f"[WARN] Failed to read CV cache {cache_path}: {exc}")

    if not best_params:
        best_params, summary = _xgb_tune_hyperparameters_cv(
            config=config,
            train_samples=train_samples,
            model_kind=model_kind,
            metric_key=metric_key,
            cast_y=cast_y,
        )
        tuning_ran = True
        if cv_requested and best_params:
            cache_written = _write_xgb_cv_tuning_cache(cache_path, hyper_cfg, best_params, summary)
            print(f"[INFO] CV tuning cache saved to: {cache_written}")

    tuned_config = copy.deepcopy(config)
    tuned_hyper = _resolve_xgb_runtime_hyperparameters(
        tuned_config["hyperparameters"], tuned_config.get("device", "cpu")
    )
    if best_params:
        tuned_hyper.update(best_params)
    elif cv_requested:
        print("[WARN] CV tuning requested but skipped; using base hyperparameters.")

    tuned_hyper["early_stopping_rounds"] = None
    tuned_config["hyperparameters"] = tuned_hyper

    if tuning_ran:
        trials_csv = _write_xgb_cv_trials_csv(config, summary)
        if trials_csv is not None:
            print(f"[INFO] CV tuning trials CSV saved to: {trials_csv}")
        trials_plot = _write_xgb_cv_trials_plot(config, summary)
        if trials_plot is not None:
            print(f"[INFO] CV tuning trials plot saved to: {trials_plot}")

    X_train, y_train, _ = _xgb_samples_to_arrays(train_samples, cast_y=cast_y)
    X_test, y_test, _ = _xgb_samples_to_arrays(test_samples, cast_y=cast_y)
    eval_set_override = [(X_train, y_train), (X_test, y_test)]

    _train_xgb_model(
        tuned_config,
        train_samples,
        test_samples,
        model_cls,
        metric_key,
        cast_y=cast_y,
        eval_set_override=eval_set_override,
        disable_early_stopping=True,
    )


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


def train_xgb_regressor_model_cv_tuned(config, train_samples, test_samples):
    """Train XGBoost Regressor with CV-tuned regularization (no test exposure)."""
    _train_xgb_model_cv_tuned(
        config,
        train_samples,
        test_samples,
        xgb.XGBRegressor,
        "metric",
        model_kind="regressor",
    )


def train_xgb_classifier_model_cv_tuned(config, train_samples, test_samples):
    """Train XGBoost Classifier with CV-tuned regularization (no test exposure)."""
    _train_xgb_model_cv_tuned(
        config,
        train_samples,
        test_samples,
        xgb.XGBClassifier,
        "eval_metric",
        model_kind="classifier",
        cast_y=lambda v: int(round(v)),
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
        if _xgb_cv_tuning_enabled(config):
            train_xgb_regressor_model_cv_tuned(config, train_samples, test_samples)
        else:
            train_xgb_regressor_model(config, train_samples, test_samples)
    elif model_type == "xgb_classifier":
        if _xgb_cv_tuning_enabled(config):
            train_xgb_classifier_model_cv_tuned(config, train_samples, test_samples)
        else:
            train_xgb_classifier_model(config, train_samples, test_samples)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

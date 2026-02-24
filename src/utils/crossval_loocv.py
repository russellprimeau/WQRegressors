"""
Leave-One-Out Cross-Validation (LOOCV) with MC-aware grouping, model training, and summary CSV output.

Usage:
    from utils.crossval_loocv import run_loocv
    run_loocv(config_path, config_dict, sample_txt, output_dir=None, mc_grouping=True)

Arguments:
    config_path: Path to the config file (used for output location)
    config_dict: Model/data config dict (must include model_type, data, etc.)
    sample_txt: Path to .txt file listing sample filenames (one per line)
    output_dir: Optional override for output directory (default: config file's directory)
    mc_grouping: If True, group samples by segment number (MC-aware LOOCV)

This module loads samples, performs MC-aware or standard LOOCV, trains and evaluates the model for each fold,
and writes a summary CSV (loocv_summary.csv) in the output directory.
"""
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from utils.training import load_samples, group_samples_by_segment

def run_loocv(config_path, config_dict, sample_txt, output_dir=None, mc_grouping=True):
    """
    Run LOOCV (MC-aware or standard), train/evaluate model for each fold, write summary CSV.
    """
    # Determine output directory
    config_path = Path(config_path)
    if output_dir is None:
        output_dir = config_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "loocv_summary.csv"

    # Load sample filenames from txt
    with open(sample_txt, "r") as f:
        file_list = [line.strip() for line in f if line.strip()]
    # Load all samples using config
    data_cfg = config_dict["data"]
    sample_dir = data_cfg.get("data_dir", ".")
    input_columns = data_cfg["input_columns"]
    output_columns = data_cfg["output_columns"]
    input_rows = slice(int(data_cfg["input_row_1"]), int(data_cfg["input_row_2"]))
    output_rows = data_cfg["output_rows"]
    sample_subdir = data_cfg.get("sample_subdir", "samples")
    fault_tolerant = data_cfg.get("fault_tolerant", False)
    input_aggregation = data_cfg.get("input_aggregation", "none")
    # Use sample_dir/sample_subdir as directory
    sample_dir_full = Path(sample_dir) / sample_subdir
    samples = load_samples(
        str(sample_dir_full),
        input_columns=input_columns,
        output_columns=output_columns,
        input_rows=input_rows,
        output_rows=output_rows,
        file_list=file_list,
        fault_tolerant=fault_tolerant,
        input_aggregation=input_aggregation,
    )
    if not samples:
        raise RuntimeError("No samples loaded for LOOCV.")

    # Group samples if MC-aware
    if mc_grouping:
        # Use group_samples_by_segment to group by segment number
        segment_groups = group_samples_by_segment(samples)
        groups = [group for seg, group in segment_groups]
        group_labels = [seg for seg, group in segment_groups]
    else:
        # Each sample is its own group
        groups = [[s] for s in samples]
        group_labels = [i for i in range(len(samples))]

    n_groups = len(groups)
    if n_groups < 2:
        raise RuntimeError("Not enough groups for LOOCV.")

    # Prepare results
    fold_metrics = []
    # For each group, leave it out as test, use others as train
    for fold_idx in range(n_groups):
        test_group = groups[fold_idx]
        train_groups = [g for i, g in enumerate(groups) if i != fold_idx]
        # Flatten train/test groups
        train_samples = [s for g in train_groups for s in g]
        test_samples = list(test_group)
        # Get filenames for reporting
        train_files = [s[2] for s in train_samples]
        test_files = [s[2] for s in test_samples]

        # Prepare config for this fold (deepcopy to avoid mutation)
        import copy
        fold_config = copy.deepcopy(config_dict)
        # Optionally, update config with fold-specific info if needed

        # Train and evaluate model for this fold
        metrics = _train_and_eval_fold(fold_config, train_samples, test_samples, fold_idx, group_labels[fold_idx])
        metrics["fold"] = fold_idx
        metrics["test_group_label"] = group_labels[fold_idx]
        metrics["n_train_samples"] = len(train_samples)
        metrics["n_test_samples"] = len(test_samples)
        metrics["train_files"] = "|".join(train_files)
        metrics["test_files"] = "|".join(test_files)
        fold_metrics.append(metrics)

    # Compute summary statistics
    summary = _summarize_folds(fold_metrics)
    # Write summary CSV
    df = pd.DataFrame([summary] + fold_metrics)
    df.to_csv(summary_csv, index=False)
    print(f"[LOOCV] Wrote summary: {summary_csv}")
    return summary_csv

def _train_and_eval_fold(config, train_samples, test_samples, fold_idx, test_group_label):
    """
    Train and evaluate model for a single fold. Returns dict of metrics.
    """
    # Dispatch based on model_type
    model_type = config.get("model_type")
    # Import train_module here to avoid circular import
    from utils import training as train_module
    import numpy as np
    # Prepare data arrays
    X_train = np.array([s[0].flatten() for s in train_samples])
    y_train = np.array([s[1].flatten() for s in train_samples])
    X_test = np.array([s[0].flatten() for s in test_samples])
    y_test = np.array([s[1].flatten() for s in test_samples])

    # Model training and prediction
    if model_type == "xgb_regressor":
        import xgboost as xgb
        model = xgb.XGBRegressor(**config.get("model_params", {}))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_type == "gp_regressor":
        # Placeholder: implement GP regressor logic as in your workflow
        # For now, raise NotImplementedError
        raise NotImplementedError("GP regressor LOOCV not implemented.")
    elif model_type == "transformer":
        # Placeholder: implement transformer logic as in your workflow
        raise NotImplementedError("Transformer LOOCV not implemented.")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Compute metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    try:
        r2 = r2_score(y_test, y_pred)
    except Exception:
        r2 = float('nan')
    # Add more metrics as needed
    return {
        "label": f"LOOCV_fold_{test_group_label}_{model_type}",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "kind": "loocv_fold",
    }

def _summarize_folds(fold_metrics):
    """
    Aggregate fold metrics into a summary row (mean, etc.).
    """
    # Compute mean of each metric
    keys = [k for k in fold_metrics[0].keys() if k not in ("label", "kind", "fold", "test_group_label", "train_files", "test_files")]
    summary = {k: np.mean([m[k] for m in fold_metrics if k in m and m[k] is not None]) for k in keys}
    summary["label"] = f"LOOCV_{fold_metrics[0]['label'].split('_')[-1]}"
    summary["kind"] = "loocv"
    return summary
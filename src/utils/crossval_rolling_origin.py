# Rolling Origin Cross-Validation for MC datasets (full workflow, MC-aware, config-driven).
# - Expanding window, single-step forecast, min train size applies to groups.
# - Handles MC replicates via segment grouping (default), or ungrouped mode.
# - Loads samples/configs as in main workflow.
# - Trains/evaluates models, writes summary CSV in config directory.
# - Supports multiple model types (xgb_regressor implemented, others as placeholders).

import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from utils.training import load_samples, group_samples_by_segment

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def rolling_origin_splits_grouped(samples: List[Any], group_ids: List[Any], min_train_size: int = 3):
    """
    Generator for rolling origin splits with expanding window, grouped by group_ids.
    Each group is treated as a unit (e.g., MC segment).
    Yields (train_indices, test_indices) for each split.
    """
    # Group samples by group_id
    group_to_indices = {}
    for idx, gid in enumerate(group_ids):
        group_to_indices.setdefault(gid, []).append(idx)
    group_keys = list(group_to_indices.keys())
    n_groups = len(group_keys)
    for i in range(min_train_size, n_groups):
        train_groups = group_keys[:i]
        test_group = group_keys[i]
        train_indices = [ix for g in train_groups for ix in group_to_indices[g]]
        test_indices = group_to_indices[test_group]
        yield train_indices, test_indices

def rolling_origin_splits_ungrouped(samples: List[Any], min_train_size: int = 3):
    """
    Generator for rolling origin splits with expanding window, ungrouped (by sample index).
    Yields (train_indices, test_indices) for each split.
    """
    n = len(samples)
    for i in range(min_train_size, n):
        train_indices = list(range(i))
        test_indices = [i]
        yield train_indices, test_indices

def run_rolling_origin_cv(
    config: Dict[str, Any],
    samples: List[Any],
    group_by_segment: bool = True,
    min_train_size: int = 3,
    model_type: str = "xgb_regressor",
    output_dir: Optional[Path] = None,
    summary_csv_name: str = "rolling_origin_summary.csv",
    verbose: bool = True,
) -> Path:
    """
    Run full rolling origin cross-validation on samples/config, MC-aware, write summary CSV.
    Args:
        config: config dict (as loaded from config file)
        samples: list of loaded samples (tuples: (X, y, filename, ...))
        group_by_segment: if True, group by segment (MC-aware)
        min_train_size: minimum number of groups/samples for first train split
        model_type: model type string ("xgb_regressor" supported)
        output_dir: directory to write summary CSV (defaults to config dir)
        summary_csv_name: name of summary CSV file
        verbose: print progress
    Returns:
        Path to summary CSV file
    """
    if output_dir is None:
        # Try to resolve config dir
        output_dir = Path(config.get("__config_dir", "."))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prepare group ids if needed
    if group_by_segment:
        segment_groups = group_samples_by_segment(samples)
        # Build a flat group_ids list: for each sample, the segment number it belongs to.
        # group_samples_by_segment returns [(seg_num, [samples_in_seg]), ...] but uses
        # actual sample objects (not indices), so we need to map back by identity.
        sample_id_to_seg = {}
        for seg_num, seg_samples in segment_groups:
            for s in seg_samples:
                sample_id_to_seg[id(s)] = seg_num
        group_ids = [sample_id_to_seg.get(id(s), i) for i, s in enumerate(samples)]
        split_gen = rolling_origin_splits_grouped(samples, group_ids, min_train_size)
    else:
        split_gen = rolling_origin_splits_ungrouped(samples, min_train_size)

    # Prepare X/y extraction
    def get_Xy(indices):
        X = np.stack([samples[i][0] for i in indices])
        y = np.stack([samples[i][1] for i in indices])
        return X, y

    results = []
    split_idx = 0
    for train_indices, test_indices in split_gen:
        split_idx += 1
        X_train, y_train = get_Xy(train_indices)
        X_test, y_test = get_Xy(test_indices)
        # Model training and prediction
        if model_type == "xgb_regressor":
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        elif model_type == "gp_regressor":
            # Placeholder: implement GP regressor if needed
            y_pred = np.full_like(y_test, np.nan)
        elif model_type == "transformer":
            # Placeholder: implement transformer if needed
            y_pred = np.full_like(y_test, np.nan)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        # Metrics
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        # Record filenames for test samples
        test_files = [samples[i][2] for i in test_indices]
        # Count independent samples (distinct segment groups).
        if group_by_segment:
            train_groups_set = set(group_ids[i] for i in train_indices)
            test_groups_set = set(group_ids[i] for i in test_indices)
            n_train_independent = len(train_groups_set)
            n_test_independent = len(test_groups_set)
        else:
            n_train_independent = len(train_indices)
            n_test_independent = len(test_indices)
        results.append({
            "split": split_idx,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "n_train_independent": n_train_independent,
            "n_test_independent": n_test_independent,
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "test_files": "|".join(map(str, test_files)),
        })
        if verbose:
            print(f"[RollingOrigin] Split {split_idx}: train={len(train_indices)} test={len(test_indices)} rmse={rmse:.4f} r2={r2:.4f}")

    # Write summary CSV
    df = pd.DataFrame(results)
    summary_csv = output_dir / summary_csv_name
    df.to_csv(summary_csv, index=False)
    if verbose:
        print(f"[RollingOrigin] Wrote summary: {summary_csv}")
    return summary_csv

# Example usage (not run):
# from utils.training import load_samples
# config = ... # load config dict
# samples = load_samples(...)
# run_rolling_origin_cv(config, samples, group_by_segment=True, min_train_size=3)
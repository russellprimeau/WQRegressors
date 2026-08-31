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

def rolling_origin_block_splits(
    n_groups: int,
    n_folds: int = 5,
    min_train_fraction: float = 0.5,
) -> List[Tuple[int, Tuple[int, int]]]:
    """Expanding-window folds over an ordered group list, in contiguous blocks.

    ``rolling_origin_splits_grouped`` advances one group at a time, which costs one
    model fit per group -- 39 fits for a 42-segment target. That is affordable once,
    on a chosen model, but not as the objective of a 240-candidate search. Blocking
    the held-out groups keeps the expanding-window property, and therefore the
    temporal ordering, at a fixed and much smaller number of fits.

    The first ``min_train_fraction`` of the groups is training-only and is never
    scored, so every fold trains on a run of history that precedes everything it
    predicts. The remainder is divided into ``n_folds`` contiguous blocks; fold *i*
    trains on all groups before its block and tests on the block itself.

    Args:
        n_groups: Number of ordered groups (segments) available.
        n_folds: Requested number of folds. Reduced when too few groups remain.
        min_train_fraction: Fraction of groups reserved as the initial training run.

    Returns:
        ``[(n_train_groups, (test_lo, test_hi)), ...]`` with half-open test ranges.

    Raises:
        ValueError: When no group is left to score after the initial training run.

    Example:
        ``rolling_origin_block_splits(42, n_folds=5, min_train_fraction=0.5)`` gives
        five folds training on 21, 25, 29, 33 and 37 groups.
    """
    n_groups = int(n_groups)
    n_folds = max(1, int(n_folds))
    if n_groups < 2:
        raise ValueError(f"Need at least 2 groups for rolling-origin folds, got {n_groups}.")

    start = int(np.ceil(n_groups * float(min_train_fraction)))
    start = max(1, min(start, n_groups - 1))
    n_scorable = n_groups - start
    if n_scorable < 1:
        raise ValueError(
            f"No groups left to score: {n_groups} groups with "
            f"min_train_fraction={min_train_fraction} reserves all of them."
        )

    # A fold that would hold no group is not a fold; drop it rather than emitting an
    # empty test block that scores nothing and still costs a fit.
    n_folds = min(n_folds, n_scorable)
    edges = [start + int(round(i * n_scorable / n_folds)) for i in range(n_folds + 1)]
    edges[-1] = n_groups

    splits: List[Tuple[int, Tuple[int, int]]] = []
    for i in range(n_folds):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        splits.append((lo, (lo, hi)))
    return splits

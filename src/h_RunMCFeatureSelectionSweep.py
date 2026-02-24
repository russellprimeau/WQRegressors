"""
Feature-selection sweeper for MC datasets using efficient subset search.

Search strategy:
- Surrogate-guided beam backward elimination with optional swap refinement.
- Objective: objective = rmse + lambda_drop * drop_rate.
- drop_rate is computed from raw sample coverage (MC replicate names collapsed to unique raw segments).

Then:
- Retrain/evaluate all discovered model configs on top-K subsets.
- Write trace, selected subsets, and final metrics to forecasts/feature_sweeps.

Examples:
python src/i_RunMCFeatureSelectionSweep.py --dry-run
python src/i_RunMCFeatureSelectionSweep.py --limit-datasets 1 --max-rounds 8 --beam-width 6
python src/i_RunMCFeatureSelectionSweep.py --row-counts 1,2,3,5 --limit-datasets 1
python src/i_RunMCFeatureSelectionSweep.py --exclude=MC_Trial1,MC_Trial2
python src/i_RunMCFeatureSelectionSweep.py --postprocess-only --keep-search-plots
"""

import contextlib
import argparse
import copy
import glob
import hashlib
import io
import re
import sys
import time
import pandas as pd
import importlib.util
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import e_Train as train_module
import f_Evaluate as eval_module
import subprocess
import os
import unicodedata
import seaborn as sns
import fnmatch
import traceback
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from utils.training import load_samples, group_samples_by_segment
from utils.model_runner import ModelRunner


SUPPORTED_CONFIG_SUFFIXES = {".yml", ".yaml", ".json"}


@dataclass
class DatasetPlan:
    dataset_dir: Path
    train_configs: list[Path]


@dataclass
class CandidateResult:
    dataset: str
    target: str
    row_count: int
    n_features: int
    feature_tag: str
    features: tuple[str, ...]
    objective: float
    rmse: float
    r2: float
    mae: float
    drop_rate: float
    n_valid_raw: float
    n_total_raw: float
    n_valid_loaded: float
    n_test_samples: float
    input_dim: float
    target_dim: float


def _derive_target_name(dataset_name: str, dataset_prefix: str) -> str:
    if dataset_name.startswith(dataset_prefix):
        return dataset_name[len(dataset_prefix):].lstrip("_")
    return dataset_name


def _feature_tag(features: tuple[str, ...]) -> str:
    joined = "||".join(features)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"f{len(features)}_{digest}"


def _model_sort_key(config_path: Path) -> tuple[int, str]:
    name = config_path.name.lower()
    if "gp" in name:
        return 0, name
    if "transformer" in name:
        return 1, name
    if "xgb" in name:
        return 2, name
    return 3, name


def _parse_row_counts(value: str | None, default_span: int) -> list[int]:
    if value is None or str(value).strip() == "":
        return [int(default_span)]

    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    rows: list[int] = []
    for part in parts:
        num = int(part)
        if num > 0:
            rows.append(num)
    rows = sorted(set(rows))
    return rows


def discover_mc_dataset_plans(
    data_root: Path,
    dataset_prefix: str,
    config_pattern: str,
    limit_datasets: int,
    include_regular: bool,
    include_res: bool,
) -> list[DatasetPlan]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    # Debug print: resolved data_root
    print(f"[DEBUG] discover_mc_dataset_plans: data_root = {data_root}")
    # Debug print: all raw subdirectory names
    all_subdirs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    print("[DEBUG] All subdirectories in data_root:", [p.name for p in all_subdirs])

    def _dataset_allowed(name: str) -> bool:
        if not name.startswith(dataset_prefix):
            return False
        is_res = name.endswith("_res")
        return (is_res and include_res) or ((not is_res) and include_regular)

    dataset_dirs = [path for path in all_subdirs if _dataset_allowed(path.name)]

    plans: list[DatasetPlan] = []
    for dataset_dir in dataset_dirs:
        raw_matches = sorted(dataset_dir.glob(config_pattern))
        train_configs = [path for path in raw_matches if path.suffix.lower() in SUPPORTED_CONFIG_SUFFIXES]
        if not train_configs:
            continue
        train_configs.sort(key=_model_sort_key)
        plans.append(DatasetPlan(dataset_dir=dataset_dir, train_configs=train_configs))

    if limit_datasets > 0:
        plans = plans[:limit_datasets]

    return plans


def _resolve_dataset_inclusion(args: argparse.Namespace) -> tuple[bool, bool]:
    include_regular = True
    include_res = True

    if args.regular_only and args.res_only:
        raise ValueError("Cannot use both --regular-only and --res-only.")

    if args.regular_only:
        include_regular, include_res = True, False
    elif args.res_only:
        include_regular, include_res = False, True
    else:
        if args.include_regular or args.include_res:
            include_regular = bool(args.include_regular)
            include_res = bool(args.include_res)

    if not include_regular and not include_res:
        raise ValueError("At least one dataset group must be included.")

    return include_regular, include_res


def _variant_forecast_name(base_forecast_name: str, row_count: int, feature_tag: str) -> str:
    base_name = str(base_forecast_name).replace("\\", "/")
    if base_name.startswith("feature_sweeps/"):
        base_name = base_name[len("feature_sweeps/") :]
    return f"feature_sweeps/{base_name}_r{row_count:03d}_{feature_tag}"


def _prepare_variant_config(
    base_config_path: Path,
    row_count: int,
    features: tuple[str, ...],
    feature_tag: str,
    tmp_dir: Path,
) -> Path:
    cfg = train_module.load_config(str(base_config_path))
    cfg_copy = copy.deepcopy(cfg)

    if "data" not in cfg_copy:
        raise ValueError(f"Missing data section in {base_config_path}")

    data_cfg = cfg_copy["data"]
    source_config_dir = Path(cfg.get("__config_dir", base_config_path.parent))
    required_data = ["input_row_1", "input_row_2", "forecast_name", "data_dir"]
    for field in required_data:
        if field not in data_cfg:
            raise ValueError(f"Missing data.{field} in {base_config_path}")

    base_stop = int(data_cfg["input_row_2"])
    base_start = int(data_cfg["input_row_1"])
    base_span = base_stop - base_start
    if base_span <= 0:
        raise ValueError(f"Invalid input row span in {base_config_path}: {base_start}..{base_stop}")
    if row_count > base_span:
        raise ValueError(f"row_count={row_count} exceeds base span={base_span} for {base_config_path.name}")

    data_cfg["input_columns"] = list(features)
    data_cfg["input_row_1"] = int(base_stop - row_count)
    data_cfg["input_row_2"] = int(base_stop)


    # Always resolve forecast_name as a relative path under the correct data_dir
    # and ensure data_dir is absolute and correct
    resolved_data_dir = train_module._resolve_path_from_config(data_cfg["data_dir"], source_config_dir)
    data_cfg["data_dir"] = str(resolved_data_dir)
    # Ensure forecast_name is always relative to the correct data_dir
    data_cfg["forecast_name"] = _variant_forecast_name(str(data_cfg["forecast_name"]), row_count, feature_tag)
    # Store the original forecast directory for downstream evaluation
    forecast_dir = Path(resolved_data_dir) / "forecasts" / "feature_sweeps" / data_cfg["forecast_name"]
    data_cfg["forecast_dir"] = str(forecast_dir.resolve())


    # Ensure evaluation config requests baselines for LOOCV
    if "evaluation" not in cfg_copy:
        cfg_copy["evaluation"] = {}
    cfg_copy["evaluation"]["run_baselines"] = True

    cfg_copy.pop("__config_dir", None)

    variant_name = f"{base_config_path.stem}_r{row_count:03d}_{feature_tag}{base_config_path.suffix}"
    variant_path = tmp_dir / variant_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with open(variant_path, "w", encoding="utf-8") as f:
        if variant_path.suffix.lower() in {".yml", ".yaml"}:
            yaml.safe_dump(cfg_copy, f, sort_keys=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported config suffix for variant writing: {variant_path.suffix}")

    return variant_path


def _train_single_config(
    config_path: Path,
    dataset_dir: Path,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
) -> Path:
    config = train_module.load_config(str(config_path))

    required_fields = ["model_type", "model_name", "data"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field '{field}' in {config_path}")

    model_type = config["model_type"]
    config = train_module.merge_with_defaults(config, model_type)
    if disable_training_plots:
        config["save_training_plots"] = False

    device = torch.device(config["device"])
    matplotlib.use(config["matplotlib_backend"])

    if suppress_training_logs:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train_samples, test_samples, _ = train_module.load_and_split_data(config)
    else:
        train_samples, test_samples, _ = train_module.load_and_split_data(config)
        print(f"    [TRAIN] {config_path.name}: train={len(train_samples)} test={len(test_samples)}")

    def _run_train():
        if model_type == "transformer":
            train_module.train_transformer_model(config, train_samples, test_samples)
        elif model_type == "gp_regressor":
            train_module.train_gp_regressor_model(config, train_samples, test_samples)
        elif model_type == "xgb_regressor":
            train_module.train_xgb_regressor_model(config, train_samples, test_samples)
        elif model_type == "xgb_classifier":
            train_module.train_xgb_classifier_model(config, train_samples, test_samples)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    if suppress_training_logs:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _run_train()
    else:
        _run_train()

    data_cfg = config["data"]
    forecast_name = data_cfg["forecast_name"]
    forecast_file_name = Path(str(forecast_name)).name
    forecast_dir = dataset_dir / "forecasts" / "feature_sweeps" / Path(forecast_name)
    return (forecast_dir / f"config_evaluate_{forecast_file_name}.yml").resolve()


def _set_eval_overrides(eval_config_path: Path, run_baselines: bool) -> None:
    # Always force run_baselines True for evaluation summary output
    with open(eval_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "evaluation" not in cfg:
        cfg["evaluation"] = {}
    cfg["evaluation"]["run_baselines"] = True

    with open(eval_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def _map_to_raw_filenames(file_names: list[str]) -> list[str]:
    mapped = []
    seen = set()
    for file_name in file_names:
        mapped_name = re.sub(r"_mc_\d+(?=\.csv$)", "", str(file_name))
        if mapped_name not in seen:
            seen.add(mapped_name)
            mapped.append(mapped_name)
    return mapped


def _count_raw_totals(sample_dir: Path) -> int:
    files = [p.name for p in sorted(sample_dir.glob("*.csv"))]
    if any("_mc_" in name for name in files):
        return len(_map_to_raw_filenames(files))
    return len(files)


def _count_valid_samples_raw(
    data_dir: Path,
    sample_subdir: str,
    input_columns: list[str],
    input_rows: slice,
    output_columns: list[str],
    output_rows: list[int],
) -> tuple[int, int, int]:
    sample_dir = Path(data_dir, sample_subdir)
    all_total_raw = _count_raw_totals(sample_dir)

    loaded = load_samples(
        str(sample_dir),
        input_columns=input_columns,
        output_columns=output_columns,
        input_rows=input_rows,
        output_rows=output_rows,
        fault_tolerant=False,
    )
    loaded_count = len(loaded)
    loaded_names = [str(sample[2]) for sample in loaded]
    valid_raw = len(_map_to_raw_filenames(loaded_names)) if any("_mc_" in name for name in loaded_names) else loaded_count

    return int(valid_raw), int(all_total_raw), int(loaded_count)


def _extract_model_summary(eval_result: dict) -> dict:
    # Confirm baseline results are present in summary_rows
    found_baseline = False
    found_model = False
    for row in eval_result.get("summary_rows", []):
        kind = str(row.get("kind", "")).lower()
        if kind == "model":
            found_model = True
        if kind in ("naive", "seasonal", "linear"):
            found_baseline = True
    if found_baseline:
        print("[CHECK] Baseline results found in evaluation summary.")
    else:
        print("[WARN] Baseline results NOT found in evaluation summary!")
    # Return the main model row as before
    for row in eval_result.get("summary_rows", []):
        if str(row.get("kind", "")).lower() == "model":
            return row
    return {}


def _objective_from_metrics(rmse: float, drop_rate: float, lambda_drop: float) -> float:
    if not np.isfinite(rmse):
        return float("inf")
    if not np.isfinite(drop_rate):
        drop_rate = 1.0
    return float(rmse + lambda_drop * drop_rate)


def _evaluate_candidate(
    dataset_dir: Path,
    target_name: str,
    surrogate_config_path: Path,
    row_count: int,
    features: tuple[str, ...],
    feature_tag: str,
    lambda_drop: float,
    tmp_cfg_dir: Path,
    disable_baselines_for_search: bool,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
) -> CandidateResult:
    base_cfg = train_module.load_config(str(surrogate_config_path))
    base_data = base_cfg["data"]
    base_stop = int(base_data["input_row_2"])
    input_rows = slice(base_stop - row_count, base_stop)

    data_dir_resolved = Path(train_module._resolve_path_from_config(base_data["data_dir"], Path(base_cfg["__config_dir"])))
    sample_subdir = str(base_data.get("sample_subdir", "samples"))
    output_columns = list(base_data["output_columns"])
    output_rows = list(base_data["output_rows"])

    valid_raw, total_raw, valid_loaded = _count_valid_samples_raw(
        data_dir_resolved,
        sample_subdir,
        list(features),
        input_rows,
        output_columns,
        output_rows,
    )
    drop_rate = float(1.0 - (valid_raw / total_raw)) if total_raw > 0 else 1.0

    variant_cfg = _prepare_variant_config(
        base_config_path=surrogate_config_path,
        row_count=row_count,
        features=features,
        feature_tag=feature_tag,
        tmp_dir=tmp_cfg_dir,
        dataset_dir=dataset_dir,
    )
    eval_cfg = _train_single_config(
        variant_cfg,
        dataset_dir,
        disable_training_plots=disable_training_plots,
        disable_eval_plots=disable_eval_plots,
        suppress_training_logs=suppress_training_logs,
    )
    _set_eval_overrides(
        eval_cfg,
        run_baselines=not disable_baselines_for_search,
    )

    eval_result = eval_module.evaluate_single_config(
        str(eval_cfg),
        save_plots_override=not disable_eval_plots,
    )
    model_row = _extract_model_summary(eval_result)

    rmse = float(model_row.get("rmse", np.nan))
    r2 = float(model_row.get("r2", np.nan))
    mae = float(model_row.get("mae", np.nan))
    n_test_samples = float(model_row.get("n_test_samples", np.nan))
    input_dim = float(model_row.get("input_dim", np.nan))
    target_dim = float(model_row.get("target_dim", np.nan))
    objective = _objective_from_metrics(rmse=rmse, drop_rate=drop_rate, lambda_drop=lambda_drop)

    return CandidateResult(
        dataset=dataset_dir.name,
        target=target_name,
        row_count=int(row_count),
        n_features=int(len(features)),
        feature_tag=feature_tag,
        features=tuple(features),
        objective=objective,
        rmse=rmse,
        r2=r2,
        mae=mae,
        drop_rate=drop_rate,
        n_valid_raw=float(valid_raw),
        n_total_raw=float(total_raw),
        n_valid_loaded=float(valid_loaded),
        n_test_samples=n_test_samples,
        input_dim=input_dim,
        target_dim=target_dim,
    )


def _candidate_key(row_count: int, features: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    return int(row_count), tuple(features)


def _plot_feature_importance_bar(
    feature_sensitivities: dict[str, tuple[float, int]],
    dataset_name: str,
    target_name: str,
    row_count: int,
    output_dir: Path,
) -> Path:
    """Plot feature importance (removal sensitivity) as horizontal bar chart."""
    ranked = sorted(feature_sensitivities.items(), key=lambda x: -x[1][0])
    features = [f for f, _ in ranked]
    scores = [s[0] for _, s in ranked]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)), constrained_layout=True)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(features)))
    ax.barh(features, scores, color=colors)
    ax.set_xlabel("Removal Sensitivity (avg delta)")
    ax.set_title(f"Feature Removal Sensitivity: {target_name} (rows={row_count})\n(Positive = valuable)")
    ax.grid(axis='x', alpha=0.3)
    
    plot_path = output_dir / f"feature_importance_bar_r{row_count:03d}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def _plot_removal_sensitivity(
    feature_removal_deltas: dict[str, list[float]],
    dataset_name: str,
    target_name: str,
    row_count: int,
    output_dir: Path,
) -> Path:
    """Plot removal sensitivity as box plot showing distribution of objective deltas."""
    features = sorted(feature_removal_deltas.keys())
    tested_pairs = [(f, feature_removal_deltas[f]) for f in features if len(feature_removal_deltas[f]) > 0]

    # Sort by median delta (most valuable-to-keep at top in the horizontal plot)
    tested_pairs.sort(key=lambda item: float(np.median(item[1])), reverse=True)

    tested_features = [item[0] for item in tested_pairs]
    tested_deltas = [item[1] for item in tested_pairs]

    if not tested_features:
        return Path()  # No data to plot

    fig_h = max(7, len(tested_features) * 0.45)
    fig, ax = plt.subplots(figsize=(14, fig_h), constrained_layout=True)
    bp = ax.boxplot(tested_deltas, vert=False, patch_artist=True)

    # Color boxes by median delta (positive = valuable feature; bad to remove)
    for patch, deltas in zip(bp['boxes'], tested_deltas):
        median_delta = np.median(deltas)
        if median_delta > 0.01:
            patch.set_facecolor('lightgreen')
        elif median_delta > 0.001:
            patch.set_facecolor('lightyellow')
        else:
            patch.set_facecolor('lightcoral')

    ax.set_yticks(np.arange(1, len(tested_features) + 1))
    ax.set_yticklabels(tested_features, fontsize=8)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Objective Delta (removing feature)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Removal Sensitivity Distribution: {target_name} (rows={row_count})\n(green=valuable, yellow=neutral, red=detrimental)")
    ax.grid(axis='x', alpha=0.3)

    plot_path = output_dir / f"removal_sensitivity_box_r{row_count:03d}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path

def _plot_feature_frequency(
    feature_improvement_counts: dict[str, int],
    feature_sensitivities: dict[str, tuple[float, int]],
    dataset_name: str,
    target_name: str,
    row_count: int,
    output_dir: Path,
) -> Path:
    """Plot feature frequency in improving solutions."""
    ranked = sorted(feature_sensitivities.items(), key=lambda x: -x[1][0])
    features = [f for f, _ in ranked]
    frequencies = [feature_improvement_counts[f] for f in features]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)), constrained_layout=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
    bars = ax.barh(features, frequencies, color=colors)
    
    # Add value labels on bars
    for bar, freq in zip(bars, frequencies):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(freq)}', 
                ha='left', va='center', fontsize=9)
    
    ax.set_xlabel("Frequency in Improving Solutions")
    ax.set_title(f"Feature Inclusion Frequency: {target_name} (rows={row_count})")
    ax.grid(axis='x', alpha=0.3)
    
    plot_path = output_dir / f"feature_frequency_r{row_count:03d}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def _write_feature_stats_artifacts(
    dataset_dir: Path,
    row_count: int,
    feature_sensitivities: dict[str, tuple[float, int]],
    feature_removal_deltas: dict[str, list[float]],
    feature_improvement_counts: dict[str, int],
) -> tuple[Path, Path]:
    out_dir = dataset_dir / "forecasts" / "feature_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    for feature in sorted(feature_sensitivities.keys()):
        avg_delta, _ = feature_sensitivities.get(feature, (0.0, 0))
        deltas = feature_removal_deltas.get(feature, [])
        median_delta = float(np.median(deltas)) if deltas else np.nan
        stats_rows.append(
            {
                "feature": feature,
                "avg_removal_delta": float(avg_delta),
                "median_removal_delta": median_delta,
                "n_removal_tests": int(len(deltas)),
                "improvement_count": int(feature_improvement_counts.get(feature, 0)),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df = stats_df.sort_values(["avg_removal_delta", "feature"], ascending=[False, True], kind="stable")

    stats_csv = out_dir / f"feature_importance_stats_r{row_count:03d}.csv"
    stats_df.to_csv(stats_csv, index=False)

    delta_rows = []
    for feature in sorted(feature_removal_deltas.keys()):
        for delta in feature_removal_deltas.get(feature, []):
            delta_rows.append(
                {
                    "feature": feature,
                    "delta": float(delta),
                }
            )

    deltas_df = pd.DataFrame(delta_rows, columns=["feature", "delta"])
    deltas_csv = out_dir / f"feature_removal_deltas_r{row_count:03d}.csv"
    deltas_df.to_csv(deltas_csv, index=False)
    return stats_csv, deltas_csv


def _load_feature_stats_artifacts(
    dataset_dir: Path,
    row_count: int,
) -> tuple[dict[str, tuple[float, int]], dict[str, int], dict[str, list[float]]]:
    out_dir = dataset_dir / "forecasts" / "feature_sweeps"
    stats_csv = out_dir / f"feature_importance_stats_r{row_count:03d}.csv"
    deltas_csv = out_dir / f"feature_removal_deltas_r{row_count:03d}.csv"

    if not stats_csv.exists() or not deltas_csv.exists():
        return {}, {}, {}

    stats_df = pd.read_csv(stats_csv)
    feature_sensitivities: dict[str, tuple[float, int]] = {}
    feature_improvement_counts: dict[str, int] = {}

    for _, row in stats_df.iterrows():
        feature = str(row.get("feature", "")).strip()
        if not feature:
            continue
        avg_delta = float(pd.to_numeric(row.get("avg_removal_delta", 0.0), errors="coerce"))
        if not np.isfinite(avg_delta):
            avg_delta = 0.0
        improvement_count = int(pd.to_numeric(row.get("improvement_count", 0), errors="coerce"))
        feature_sensitivities[feature] = (avg_delta, improvement_count)
        feature_improvement_counts[feature] = improvement_count

    deltas_df = pd.read_csv(deltas_csv)
    feature_removal_deltas: dict[str, list[float]] = {feature: [] for feature in feature_sensitivities.keys()}
    if not deltas_df.empty and "feature" in deltas_df.columns and "delta" in deltas_df.columns:
        for feature, group in deltas_df.groupby("feature", sort=True):
            key = str(feature)
            values = pd.to_numeric(group["delta"], errors="coerce")
            finite_vals = [float(v) for v in values.to_numpy(dtype=float) if np.isfinite(v)]
            feature_removal_deltas[key] = finite_vals
            if key not in feature_sensitivities:
                avg_delta = float(np.mean(finite_vals)) if finite_vals else 0.0
                feature_sensitivities[key] = (avg_delta, 0)
                feature_improvement_counts[key] = 0

    return feature_sensitivities, feature_improvement_counts, feature_removal_deltas


def _available_row_counts_for_postprocess(dataset_dir: Path) -> list[int]:
    out_dir = dataset_dir / "forecasts" / "feature_sweeps"
    patterns = [
        "feature_importance_stats_r*.csv",
        "feature_search_trace_r*.csv",
        "feature_selected_subsets_r*.csv",
    ]
    row_counts: set[int] = set()
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            match = re.search(r"_r(\d{3})\.csv$", path.name)
            if match:
                row_counts.add(int(match.group(1)))
    return sorted(row_counts)


def _regenerate_saved_outputs_for_row(
    dataset_dir: Path,
    target_name: str,
    row_count: int,
    keep_search_plots: bool,
) -> dict[str, Path]:
    out_dir = dataset_dir / "forecasts" / "feature_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    trace_csv = out_dir / f"feature_search_trace_r{row_count:03d}.csv"
    if keep_search_plots and trace_csv.exists():
        trace_df = pd.read_csv(trace_csv)
        if not trace_df.empty and {"drop_rate", "rmse"}.issubset(set(trace_df.columns)):
            selected_csv = out_dir / f"feature_selected_subsets_r{row_count:03d}.csv"
            selected_df = pd.read_csv(selected_csv) if selected_csv.exists() else pd.DataFrame()
            plot_path = out_dir / f"feature_search_pareto_r{row_count:03d}.png"

            fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5), constrained_layout=True)
            ax.scatter(trace_df["drop_rate"], trace_df["rmse"], s=20, alpha=0.6)
            if not selected_df.empty and {"drop_rate", "rmse"}.issubset(set(selected_df.columns)):
                ax.scatter(selected_df["drop_rate"], selected_df["rmse"], s=60, marker="*", color="red")
            ax.set_xlabel("Drop rate (raw sample loss)")
            ax.set_ylabel("RMSE (surrogate)")
            ax.set_title(f"Feature search Pareto-like view (rows={row_count})")
            ax.grid(alpha=0.25)
            fig.savefig(plot_path, dpi=180)
            plt.close(fig)
            written["pareto_plot"] = plot_path

    feature_sensitivities, feature_improvement_counts, feature_removal_deltas = _load_feature_stats_artifacts(
        dataset_dir=dataset_dir,
        row_count=row_count,
    )
    if not feature_sensitivities:
        return written

    bar_plot = _plot_feature_importance_bar(feature_sensitivities, dataset_dir.name, target_name, row_count, out_dir)
    written["bar_plot"] = bar_plot

    sensitivity_plot = _plot_removal_sensitivity(feature_removal_deltas, dataset_dir.name, target_name, row_count, out_dir)
    if sensitivity_plot.exists():
        written["sensitivity_plot"] = sensitivity_plot

    frequency_plot = _plot_feature_frequency(
        feature_improvement_counts,
        feature_sensitivities,
        dataset_dir.name,
        target_name,
        row_count,
        out_dir,
    )
    written["frequency_plot"] = frequency_plot
    return written


def _beam_search_subsets(
    dataset_dir: Path,
    dataset_prefix: str,
    surrogate_config_path: Path,
    row_count: int,
    lambda_drop: float,
    beam_width: int,
    max_rounds: int,
    no_improve_patience: int,
    min_features: int,
    eval_budget: int,
    max_swap_attempts: int,
    disable_baselines_for_search: bool,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
    seed: int,
) -> tuple[list[CandidateResult], list[CandidateResult], dict[str, tuple[float, int]]]:
    target_name = _derive_target_name(dataset_dir.name, dataset_prefix)
    tmp_cfg_dir = dataset_dir / "forecasts" / "feature_sweeps" / "configs"

    base_cfg = train_module.load_config(str(surrogate_config_path))
    full_features = tuple(base_cfg["data"]["input_columns"])
    if len(full_features) <= min_features:
        raise ValueError(f"min_features={min_features} must be < number of features ({len(full_features)})")

    rng = np.random.default_rng(seed)
    cache: dict[tuple[int, tuple[str, ...]], CandidateResult] = {}
    trace: list[CandidateResult] = []
    eval_count = 0
    search_start_time = time.time()
    
    # Feature importance tracking
    feature_removal_deltas: dict[str, list[float]] = {feat: [] for feat in full_features}
    feature_improvement_counts: dict[str, int] = {feat: 0 for feat in full_features}

    def _eval(features: tuple[str, ...]) -> CandidateResult | None:
        nonlocal eval_count
        key = _candidate_key(row_count, features)
        if key in cache:
            return cache[key]
        if eval_count >= eval_budget:
            return None

        tag = _feature_tag(features)
        result = _evaluate_candidate(
            dataset_dir=dataset_dir,
            target_name=target_name,
            surrogate_config_path=surrogate_config_path,
            row_count=row_count,
            features=features,
            feature_tag=tag,
            lambda_drop=lambda_drop,
            tmp_cfg_dir=tmp_cfg_dir,
            disable_baselines_for_search=disable_baselines_for_search,
            disable_training_plots=disable_training_plots,
            disable_eval_plots=disable_eval_plots,
            suppress_training_logs=suppress_training_logs,
        )
        cache[key] = result
        trace.append(result)
        eval_count += 1
        
        elapsed = time.time() - search_start_time
        avg_time_per_eval = elapsed / eval_count
        remaining_evals = eval_budget - eval_count
        eta_seconds = avg_time_per_eval * remaining_evals
        
        return result

    first = _eval(full_features)
    if first is None:
        raise RuntimeError("Search budget exhausted before evaluating initial subset.")

    beam: list[CandidateResult] = [first]
    best = first
    no_improve = 0
    
    elapsed = time.time() - search_start_time
    avg_time_per_eval = elapsed / eval_count if eval_count > 0 else 0
    remaining_evals = eval_budget - eval_count
    eta_seconds = avg_time_per_eval * remaining_evals
    eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 0 else "unknown"
    print(f"[SEARCH] Initial (all {len(full_features)} features): objective={best.objective:.4f} rmse={best.rmse:.6f} (evals: {eval_count}/{eval_budget}, ETA: {eta_str})")

    for _round in range(max_rounds):
        candidates: list[tuple[str, ...]] = []
        seen = set()

        for item in beam:
            feat_list = list(item.features)
            if len(feat_list) <= min_features:
                continue
            for idx in range(len(feat_list)):
                child = tuple(feat_list[:idx] + feat_list[idx + 1 :])
                if len(child) < min_features:
                    continue
                key = _candidate_key(row_count, child)
                if key in cache:
                    continue
                if child in seen:
                    continue
                seen.add(child)
                candidates.append(child)

        if not candidates:
            print(f"[SEARCH] Round {_round + 1}: no new candidates, stopping.")
            break

        rng.shuffle(candidates)
        scored: list[CandidateResult] = []
        for child in candidates:
            out = _eval(child)
            if out is None:
                print(f"[SEARCH] Round {_round + 1}: eval budget exhausted after {eval_count} evals.")
                break
            scored.append(out)
            
            # Track removal sensitivity: which feature was removed from beam members to create this child?
            # Find parent by checking which single feature difference exists
            for parent_item in beam:
                parent_set = set(parent_item.features)
                child_set = set(child)
                if len(parent_set - child_set) == 1:  # exactly one feature removed
                    removed_feat = list(parent_set - child_set)[0]
                    delta = out.objective - parent_item.objective
                    feature_removal_deltas[removed_feat].append(delta)
                    break

        if not scored:
            print(f"[SEARCH] Round {_round + 1}: no scored candidates, stopping.")
            break

        scored.sort(key=lambda x: (x.objective, x.rmse, -x.n_features))
        beam = scored[:beam_width]
        prev_best = best.objective
        
        elapsed = time.time() - search_start_time
        avg_time_per_eval = elapsed / eval_count if eval_count > 0 else 0
        remaining_evals = eval_budget - eval_count
        eta_seconds = avg_time_per_eval * remaining_evals
        eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 0 else "unknown"
        
        if beam and beam[0].objective + 1e-12 < best.objective:
            best = beam[0]
            no_improve = 0
            # Track features in improving solution (Option A)
            for feat in best.features:
                feature_improvement_counts[feat] += 1
            print(f"[SEARCH] Round {_round + 1}: improved! objective={best.objective:.4f} rmse={best.rmse:.6f} n_features={best.n_features} (evals: {eval_count}/{eval_budget}, ETA: {eta_str})")
        else:
            no_improve += 1
            print(f"[SEARCH] Round {_round + 1}: no improvement ({no_improve}/{no_improve_patience}). Best: objective={best.objective:.4f} rmse={best.rmse:.6f} (evals: {eval_count}/{eval_budget}, ETA: {eta_str})")
            if no_improve >= no_improve_patience:
                print(f"[SEARCH] Patience exhausted, stopping.")

    current = best
    all_features_set = set(full_features)
    attempts = 0
    improved = True
    swap_iter = 0
    
    elapsed = time.time() - search_start_time
    avg_time_per_eval = elapsed / eval_count if eval_count > 0 else 0
    remaining_evals = eval_budget - eval_count
    eta_seconds = avg_time_per_eval * remaining_evals
    eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 0 else "unknown"
    print(f"[SEARCH] Starting swap refinement from: objective={current.objective:.4f} rmse={current.rmse:.6f} n_features={current.n_features} (ETA: {eta_str})")
    
    while improved and attempts < max_swap_attempts and eval_count < eval_budget:
        improved = False
        included = list(current.features)
        excluded = list(all_features_set - set(included))
        if not excluded or len(included) <= min_features:
            break

        swap_pairs = [(drop_f, add_f) for drop_f in included for add_f in excluded]
        rng.shuffle(swap_pairs)

        for drop_f, add_f in swap_pairs:
            attempts += 1
            if attempts > max_swap_attempts:
                break

            new_features = [f for f in included if f != drop_f] + [add_f]
            new_features = tuple(sorted(new_features, key=lambda s: full_features.index(s)))
            out = _eval(new_features)
            if out is None:
                print(f"[SEARCH] Swap refinement: eval budget exhausted after {eval_count} evals.")
                break
            if out.objective + 1e-12 < current.objective:
                swap_iter += 1
                current = out
                best = out
                improved = True
                
                elapsed = time.time() - search_start_time
                avg_time_per_eval = elapsed / eval_count if eval_count > 0 else 0
                remaining_evals = eval_budget - eval_count
                eta_seconds = avg_time_per_eval * remaining_evals
                eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 0 else "unknown"
                print(f"[SEARCH] Swap refinement #{swap_iter}: improved! objective={best.objective:.4f} rmse={best.rmse:.6f} n_features={best.n_features} (evals: {eval_count}/{eval_budget}, ETA: {eta_str})")
                break
    
    if not improved and eval_count < eval_budget:
        print(f"[SEARCH] Swap refinement: no improvements found (attempts: {attempts}/{max_swap_attempts}, evals: {eval_count}/{eval_budget})")

    top_sorted = sorted(trace, key=lambda x: (x.objective, x.rmse, -x.n_features))
    total_elapsed = time.time() - search_start_time
    elapsed_min = int(total_elapsed // 60)
    elapsed_sec = int(total_elapsed % 60)
    avg_time_per_eval = total_elapsed / eval_count if eval_count > 0 else 0
    print(f"[SEARCH] Complete: {eval_count}/{eval_budget} evaluations in {elapsed_min}m {elapsed_sec}s ({avg_time_per_eval:.1f}s/eval). Best: objective={best.objective:.4f} rmse={best.rmse:.6f} r2={best.r2:.6f} n_features={best.n_features}")
    
    # Print feature importance summary
    print(f"\n[SEARCH] Feature importance (removal sensitivity — positive delta = bad to remove):")
    
    # Compute average removal sensitivity for each feature
    feature_sensitivities: dict[str, tuple[float, int]] = {}  # (avg_removal_delta, frequency_count)
    for feat in full_features:
        deltas = feature_removal_deltas[feat]
        counts = feature_improvement_counts[feat]
        
        # Average removal delta (positive = valuable)
        avg_delta = float(np.mean(deltas)) if deltas else 0.0
        feature_sensitivities[feat] = (avg_delta, counts)
    
    # Sort by removal sensitivity descending
    ranked_features = sorted(feature_sensitivities.items(), key=lambda x: -x[1][0])
    
    # Print ranked by removal sensitivity
    print(f"\n[SEARCH] Ranked feature importance (by removal sensitivity):")
    for rank, (feat, (avg_delta, frequency)) in enumerate(ranked_features, 1):
        deltas = feature_removal_deltas[feat]
        n_removals = len(deltas)
        if n_removals > 0:
            print(f"  {rank}. {feat:30s} | removal_sensitivity={avg_delta:+.6f} (n_tests={n_removals}, frequency={int(frequency)})")
        else:
            print(f"  {rank}. {feat:30s} | never removed (frequency={int(frequency)})")
    
    # Recommendation: Show best subset found
    print(f"\n[SEARCH] Recommended feature subset (from search best):")
    print(f"  Features ({best.n_features}): {', '.join(best.features)}")
    print(f"  Objective: {best.objective:.6f} (rmse={best.rmse:.6f}, r2={best.r2:.6f}, drop_rate={best.drop_rate:.4f})")
    
    # Optional: High-value feature recommendation (top performers by removal sensitivity)
    top_k_features = min(len(full_features) // 2, 8)  # Show top ~half or top 8
    essential = [feat for feat, (_, _) in ranked_features[:top_k_features]]
    print(f"\n[SEARCH] Top {len(essential)} essential features (by removal sensitivity):")
    print(f"  {', '.join(essential)}")
    
    # Generate feature importance visualizations
    print(f"\n[SEARCH] Generating feature importance visualizations...")
    out_dir = dataset_dir / "forecasts" / "feature_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        stats_csv, deltas_csv = _write_feature_stats_artifacts(
            dataset_dir=dataset_dir,
            row_count=row_count,
            feature_sensitivities=feature_sensitivities,
            feature_removal_deltas=feature_removal_deltas,
            feature_improvement_counts=feature_improvement_counts,
        )
        print(f"[INFO] Wrote feature stats table: {stats_csv}")
        print(f"[INFO] Wrote feature deltas table: {deltas_csv}")

        bar_plot = _plot_feature_importance_bar(feature_sensitivities, dataset_dir.name, target_name, row_count, out_dir)
        print(f"[INFO] Wrote feature importance bar chart: {bar_plot}")
        
        sensitivity_plot = _plot_removal_sensitivity(feature_removal_deltas, dataset_dir.name, target_name, row_count, out_dir)
        if sensitivity_plot.exists():
            print(f"[INFO] Wrote removal sensitivity plot: {sensitivity_plot}")
        
        frequency_plot = _plot_feature_frequency(feature_improvement_counts, feature_sensitivities, dataset_dir.name, target_name, row_count, out_dir)
        print(f"[INFO] Wrote feature frequency plot: {frequency_plot}")
    except Exception as e:
        print(f"[WARN] Failed to generate feature importance plots: {e}")
    
    return top_sorted, trace, feature_sensitivities


def _select_surrogate_config(train_configs: list[Path]) -> Path:
    for cfg in train_configs:
        name = cfg.name.lower()
        if "xgb" in name and "classifier" not in name:
            return cfg
    return train_configs[0]


def _compile_multi_target_comparison(
    sweep_results: dict[str, dict],  # target -> {row_count -> feature_sensitivities}
    data_root: Path,
) -> Path:
    """Compile and visualize feature importance across multiple targets using removal sensitivity."""
    if not sweep_results:
        return Path()
    
    # Collect all unique features across all targets
    all_features_set = set()
    for target_data in sweep_results.values():
        for feature_sensitivities in target_data.values():
            all_features_set.update(feature_sensitivities.keys())

    # Read feature order from Consolidated_sparse.csv
    csv_path = data_root.parent / "regression" / "Consolidated_sparse.csv"
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        # Remove timestamp and non-feature columns if needed
        csv_features = [col for col in header if col in all_features_set]
        # Add any features not in CSV at the end (shouldn't happen, but for safety)
        all_features = csv_features + [f for f in sorted(all_features_set) if f not in csv_features]
    except Exception:
        all_features = sorted(all_features_set)
    
    if not all_features:
        return Path()
    
    # Build matrix: targets x features with removal sensitivity scores
    # Sort targets by their order in the CSV, if present, otherwise alphabetically
    # Use the same csv_path and header as above
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        res_targets = [col for col in header if col.endswith("_res")]
        sweep_keys = list(sweep_results.keys())
        matched_keys = []
        yticklabels = []
        used_keys = set()
        def norm(s):
            # Lowercase, replace µ->u, °->deg, remove all non-alphanumeric chars
            s = s.lower().replace('µ', 'u').replace('°', 'deg')
            s = unicodedata.normalize('NFKD', s)
            s = ''.join(c for c in s if not unicodedata.combining(c))
            s = re.sub(r'[^a-z0-9]', '', s)
            return s

        # Debug prints removed

        for csv_col in res_targets:
            # Prefer exact match
            if csv_col in sweep_results:
                matched_keys.append(csv_col)
                yticklabels.append(csv_col)
                used_keys.add(csv_col)
                continue
            # Otherwise, try suffix/unicode-normalized match
            matches = [k for k in sweep_keys if norm(k).endswith(norm(csv_col)) and k not in used_keys]
            if matches:
                matched_keys.append(matches[0])
                yticklabels.append(csv_col)
                used_keys.add(matches[0])
        # Add any sweep_results keys not already used, at the end
        extra_keys = [t for t in sweep_keys if t not in used_keys]
        matched_keys += extra_keys
        yticklabels += extra_keys
        targets = matched_keys
    except Exception:
        targets = sorted(sweep_results.keys())
        yticklabels = targets

    matrix = np.zeros((len(targets), len(all_features)))

    for i, target in enumerate(targets):
        for j, feat in enumerate(all_features):
            # Use the first (finest) row_count's data for comparison
            for feature_sensitivities in sweep_results[target].values():
                if feat in feature_sensitivities:
                    matrix[i, j] = feature_sensitivities[feat][0]  # removal sensitivity
                    break
    
    fig, ax = plt.subplots(figsize=(max(12, len(all_features) * 0.4), max(8, len(targets) * 0.5)), constrained_layout=True)
    vmin = np.percentile(matrix, 5)
    vmax = np.percentile(matrix, 95)
    annot_fmt = ".2e"
    # Dynamically set font size based on grid size
    min_dim = min(len(all_features), len(targets))
    annot_fontsize = max(5, min(8, int(120 / max(len(all_features), len(targets), 1))))
    # Draw heatmap without annotation first
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        annot=False,
        cbar_kws={"label": "Removal Sensitivity (avg delta)"},
        xticklabels=all_features,
        yticklabels=yticklabels,
        linewidths=0.5,
        linecolor="#eeeeee",
        square=False,
    )
    # Add rotated annotation labels manually, ensuring they fit in the cell
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j + 0.5, i + 0.5, f"{value:.2e}",
                ha="center", va="center", color="black",
                fontsize=annot_fontsize, rotation=90, clip_on=True
            )
    ax.set_xticklabels(all_features, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(targets, fontsize=9)
    ax.set_title("Feature Removal Sensitivity Heatmap Across Targets\n(Positive = valuable)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")

    # Save to root output directory
    summaries_dir = (data_root.parent / "regression" / "summaries").resolve()
    summaries_dir.mkdir(parents=True, exist_ok=True)
    plot_path = summaries_dir / "multi_target_importance_heatmap.png"
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # Create a single bar chart: sum removal sensitivities for each predictor over all targets
    summed_sensitivity = matrix.sum(axis=0)
    sorted_indices = np.argsort(-summed_sensitivity)
    top_features = [all_features[i] for i in sorted_indices]
    summed_scores = [summed_sensitivity[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(max(14, len(top_features) * 0.5), 6), constrained_layout=True)
    x = np.arange(len(top_features))
    bars = ax.bar(x, summed_scores, color=plt.cm.RdYlGn((np.array(summed_scores) - np.min(summed_scores)) / (np.ptp(summed_scores) if np.ptp(summed_scores) > 0 else 1)))
    ax.set_xlabel("Feature (sorted by total removal sensitivity)")
    ax.set_ylabel("Total Removal Sensitivity (sum across targets)")
    ax.set_title(f"Total Feature Removal Sensitivity Across All Targets\n(Positive = valuable feature)")
    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels (smaller font)
    for bar, score in zip(bars, summed_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{score:.2e}', ha='center', va='bottom', fontsize=7, rotation=90)

    bar_path = summaries_dir / "multi_target_importance_bars.png"
    fig.savefig(bar_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def _write_search_outputs(
    dataset_dir: Path,
    row_count: int,
    trace: list[CandidateResult],
    selected: list[CandidateResult],
    save_plots: bool,
) -> tuple[Path, Path, Path]:
    out_dir = dataset_dir / "forecasts" / "feature_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = []
    for idx, item in enumerate(trace, start=1):
        trace_rows.append(
            {
                "eval_index": idx,
                "target": item.target,
                "row_count": item.row_count,
                "feature_tag": item.feature_tag,
                "n_features": item.n_features,
                "objective": item.objective,
                "rmse": item.rmse,
                "r2": item.r2,
                "mae": item.mae,
                "drop_rate": item.drop_rate,
                "n_valid_raw": item.n_valid_raw,
                "n_total_raw": item.n_total_raw,
                "n_valid_loaded": item.n_valid_loaded,
                "n_test_samples": item.n_test_samples,
                "input_dim": item.input_dim,
                "target_dim": item.target_dim,
                "features": "|".join(item.features),
            }
        )
    trace_df = pd.DataFrame(trace_rows)
    trace_csv = out_dir / f"feature_search_trace_r{row_count:03d}.csv"
    trace_df.to_csv(trace_csv, index=False)

    selected_rows = []
    for rank, item in enumerate(selected, start=1):
        selected_rows.append(
            {
                "rank": rank,
                "target": item.target,
                "row_count": item.row_count,
                "feature_tag": item.feature_tag,
                "n_features": item.n_features,
                "objective": item.objective,
                "rmse": item.rmse,
                "r2": item.r2,
                "mae": item.mae,
                "drop_rate": item.drop_rate,
                "n_valid_raw": item.n_valid_raw,
                "n_total_raw": item.n_total_raw,
                "features": "|".join(item.features),
            }
        )
    selected_df = pd.DataFrame(selected_rows)
    selected_csv = out_dir / f"feature_selected_subsets_r{row_count:03d}.csv"
    selected_df.to_csv(selected_csv, index=False)

    plot_path = out_dir / f"feature_search_pareto_r{row_count:03d}.png"
    if save_plots and not trace_df.empty:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5), constrained_layout=True)
        ax.scatter(trace_df["drop_rate"], trace_df["rmse"], s=20, alpha=0.6)
        if not selected_df.empty:
            ax.scatter(selected_df["drop_rate"], selected_df["rmse"], s=60, marker="*", color="red")
        ax.set_xlabel("Drop rate (raw sample loss)")
        ax.set_ylabel("RMSE (surrogate)")
        ax.set_title(f"Feature search Pareto-like view (rows={row_count})")
        ax.grid(alpha=0.25)
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)

    return trace_csv, selected_csv, plot_path


def _evaluate_selected_subsets_all_models(
    dataset_plan: DatasetPlan,
    dataset_prefix: str,
    selected: list[CandidateResult],
    run_baselines_in_final: bool,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
) -> Path:
    rows = []
    output_dir = dataset_plan.dataset_dir / "forecasts" / "feature_sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = output_dir / "configs"

    target_name = _derive_target_name(dataset_plan.dataset_dir.name, dataset_prefix)

    for rank, cand in enumerate(selected, start=1):
        for base_cfg in dataset_plan.train_configs:
            variant_cfg = _prepare_variant_config(
                base_config_path=base_cfg,
                row_count=cand.row_count,
                features=cand.features,
                feature_tag=f"{cand.feature_tag}_k{rank:02d}",
                tmp_dir=cfg_dir,
            )

            eval_cfg = _train_single_config(
                variant_cfg,
                disable_training_plots=disable_training_plots,
                disable_eval_plots=disable_eval_plots,
                suppress_training_logs=suppress_training_logs,
            )
            _set_eval_overrides(
                eval_cfg,
                run_baselines=run_baselines_in_final,
            )
            eval_result = eval_module.evaluate_single_config(
                str(eval_cfg),
                save_plots_override=not disable_eval_plots,
            )

            for srow in eval_result.get("summary_rows", []):
                kind = str(srow.get("kind", "")).lower()
                if kind != "model":
                    continue

                split_files = eval_result.get("model_split_files", [])
                if split_files and any("_mc_" in str(name) for name in split_files):
                    n_samples = len(_map_to_raw_filenames([str(name) for name in split_files]))
                else:
                    n_samples = len(split_files) if split_files else srow.get("n_test_samples", np.nan)

                # Compute std(target) for this model/config
                std_target = float('nan')
                try:
                    with open(eval_cfg, 'r', encoding='utf-8') as f:
                        cfg = yaml.safe_load(f)
                    target_cols = cfg['data'].get('output_columns', None)
                    data_dir = cfg['data']['data_dir']
                    sample_subdir = cfg['data'].get('sample_subdir', 'samples')
                    sample_dir = os.path.join(data_dir, sample_subdir)
                    csv_files = glob.glob(os.path.join(sample_dir, '*.csv'))
                    target_vals = []
                    for csvf in csv_files:
                        try:
                            df_csv = pd.read_csv(csvf)
                            if target_cols and all(tc in df_csv.columns for tc in target_cols):
                                for tc in target_cols:
                                    target_vals.extend(df_csv[tc].dropna().values.tolist())
                        except Exception:
                            continue
                    if target_vals:
                        std_target = float(np.std(target_vals, ddof=1))
                except Exception as e:
                    print(f"[WARN] Could not compute std(target) for {dataset_plan.dataset_dir.name}: {e}")

                rows.append(
                    {
                        "dataset": dataset_plan.dataset_dir.name,
                        "target": target_name,
                        "subset_rank": rank,
                        "feature_tag": cand.feature_tag,
                        "row_count": cand.row_count,
                        "n_features": cand.n_features,
                        "objective_search": cand.objective,
                        "drop_rate_search": cand.drop_rate,
                        "model": srow.get("label", "unknown"),
                        "n_samples": float(n_samples),
                        "input_dim": float(srow.get("input_dim", np.nan)),
                        "target_dim": float(srow.get("target_dim", np.nan)),
                        "mae": float(srow.get("mae", np.nan)),
                        "rmse": float(srow.get("rmse", np.nan)),
                        "r2": float(srow.get("r2", np.nan)),
                        "std_target": std_target,
                    }
                )

    final_df = pd.DataFrame(rows)
    out_csv = output_dir / "feature_sweep_final_metrics.csv"
    final_df.to_csv(out_csv, index=False)
    return out_csv


def run_feature_selection_sweep(args: argparse.Namespace) -> int:
    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()


    include_regular, include_res = _resolve_dataset_inclusion(args)

    # In postprocess-only mode, ignore limit_datasets and dataset_prefix to process all datasets
    if args.postprocess_only:
        plans = discover_mc_dataset_plans(
            data_root=data_root,
            dataset_prefix="",  # match all
            config_pattern=args.config_pattern,
            limit_datasets=0,   # no limit
            include_regular=include_regular,
            include_res=include_res,
        )
    else:
        plans = discover_mc_dataset_plans(
            data_root=data_root,
            dataset_prefix=args.dataset_prefix,
            config_pattern=args.config_pattern,
            limit_datasets=args.limit_datasets,
            include_regular=include_regular,
            include_res=include_res,
        )

    # --exclude logic removed
    if not plans:
        print("No matching datasets/configs found.")
        return 1

    print("\nExecution plan")
    print("-" * 100)
    print(f"Data root                 : {data_root}")
    print(f"Dataset prefix            : {args.dataset_prefix}")
    print(f"Config pattern            : {args.config_pattern}")
    print(f"Datasets found            : {len(plans)}")
    print(f"Beam width                : {args.beam_width}")
    print(f"Max rounds                : {args.max_rounds}")
    print(f"Patience                  : {args.no_improve_patience}")
    print(f"Eval budget               : {args.eval_budget}")
    print(f"Swap attempts             : {args.max_swap_attempts}")
    print(f"Lambda drop               : {args.lambda_drop}")
    print(f"Top-K for final models    : {args.final_top_k}")
    print(f"Dry run                   : {args.dry_run}")
    print(f"Keep train plots          : {args.keep_training_plots}")
    print(f"Keep eval plots           : {args.keep_eval_plots}")
    print(f"Keep search plots         : {args.keep_search_plots}")
    print(f"Show train logs           : {args.show_training_logs}")

    if args.dry_run:
        for plan in plans:
            surrogate = _select_surrogate_config(plan.train_configs)
            cfg = train_module.load_config(str(surrogate))
            base_span = int(cfg["data"]["input_row_2"]) - int(cfg["data"]["input_row_1"])
            row_counts = _parse_row_counts(args.row_counts, default_span=base_span)
            print(f"  - {plan.dataset_dir.name}: surrogate={surrogate.name}, row_counts={row_counts}")
        return 0


    if args.postprocess_only:
        for plan in plans:
            output_dir = plan.dataset_dir / "forecasts" / "feature_sweeps"
            metrics_csv = output_dir / "feature_sweep_final_metrics.csv"
            if not metrics_csv.exists():
                print(f"[WARN] No feature_sweep_final_metrics.csv found for {plan.dataset_dir.name}; skipping.")
                continue
            df = pd.read_csv(metrics_csv)
            if df.empty:
                print(f"[WARN] feature_sweep_final_metrics.csv is empty for {plan.dataset_dir.name}; skipping.")
                continue
            # Select the best model row using highest r2 value
            valid_r2 = df[df['r2'].notnull() & np.isfinite(df['r2'])]
            if not valid_r2.empty:
                idx_best = valid_r2['r2'].idxmax()
                best_row = valid_r2.loc[idx_best]
            else:
                print(f"[WARN] No valid r2 values found in feature_sweep_final_metrics.csv for {plan.dataset_dir.name}; skipping.")
                continue
            feature_tag = best_row["feature_tag"]
            row_count_val = int(best_row["row_count"])
            model_name = str(best_row["model"])
            # Construct model directory and config path
            model_dir = output_dir / f"{model_name}_r{row_count_val}_{feature_tag}"
            forecast_file_name = f"{model_name}_r{row_count_val}_{feature_tag}"
            eval_config_path = model_dir / f"config_evaluate_{forecast_file_name}.yml"
            if not eval_config_path.exists():
                print(f"[WARN] Evaluation config not found: {eval_config_path}")
                continue
            # Run evaluation with run_baselines enabled at runtime (via env var)
            env = os.environ.copy()
            env["RUN_BASELINES"] = "true"
            try:
                subprocess.run([
                    sys.executable, 'src/f_Evaluate.py',
                    '--config', str(eval_config_path)
                ], check=True, env=env)
            except Exception as e:
                print(f"[WARN] Could not re-run evaluation for {eval_config_path}: {e}")

        sweep_results: dict[str, dict[int, dict[str, tuple[float, int]]]] = {}
        datasets_with_outputs = 0
        best_model_performance = []

        for plan in plans:
            target_name = _derive_target_name(plan.dataset_dir.name, args.dataset_prefix)
            surrogate_cfg = _select_surrogate_config(plan.train_configs)
            surrogate_data = train_module.load_config(str(surrogate_cfg))['data']
            base_span = int(surrogate_data['input_row_2']) - int(surrogate_data['input_row_1'])
            requested_rows = _parse_row_counts(args.row_counts, default_span=base_span)
            row_counts = requested_rows if args.row_counts else _available_row_counts_for_postprocess(plan.dataset_dir)

            if not row_counts:
                print(f"[WARN] No saved feature-sweep artifacts found for {plan.dataset_dir.name}; skipping.")
                continue

            output_dir = plan.dataset_dir / "forecasts" / "feature_sweeps"
            metrics_csv = output_dir / "feature_sweep_final_metrics.csv"
            if metrics_csv.exists():
                df = pd.read_csv(metrics_csv)
                if "std_target" not in df.columns or df["std_target"].isnull().all():
                    std_targets = [None] * len(df)
                    for idx, row in df.iterrows():
                        feature_tag = row.get("feature_tag", "")
                        row_count_val = int(row.get("row_count", 0))
                        model = row.get("model", None)
                        cfg_dir = output_dir / "configs"
                        # Only compute std_target for the config/model that matches a config file AND model name
                        cfg_candidates = [p for p in cfg_dir.glob(f"*_r{row_count_val:03d}_{feature_tag}*.yml") if model and model.lower() in p.name.lower()]
                        if not cfg_candidates:
                            std_targets[idx] = None
                            continue
                        cfg_path = cfg_candidates[0]
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            cfg = yaml.safe_load(f)
                        data_cfg = cfg["data"]
                        data_dir = Path(train_module._resolve_path_from_config(data_cfg["data_dir"], Path(cfg.get("__config_dir", cfg_path.parent))))
                        sample_subdir = str(data_cfg.get("sample_subdir", "samples"))
                        output_columns = list(data_cfg["output_columns"])
                        output_rows = list(data_cfg["output_rows"])
                        samples = load_samples(
                            str(data_dir / sample_subdir),
                            input_columns=list(data_cfg["input_columns"]),
                            output_columns=output_columns,
                            input_rows=slice(data_cfg["input_row_1"], data_cfg["input_row_2"]),
                            output_rows=output_rows,
                            fault_tolerant=True,
                        )
                        if samples and len(samples) > 0:
                            outputs = np.array([s[1] for s in samples], dtype=float)
                            if outputs.ndim == 2 and outputs.shape[1] == 1:
                                std_target = float(np.std(outputs[:, 0], ddof=1))
                            elif outputs.ndim == 2:
                                std_target = float(np.mean(np.std(outputs, axis=0, ddof=1)))
                            else:
                                std_target = float(np.std(outputs, ddof=1))
                            std_targets[idx] = std_target
                        else:
                            std_targets[idx] = None
                    # Only update std_target for rows where it was computed; leave others empty
                    df["std_target"] = std_targets
                    df.to_csv(metrics_csv, index=False)

            print(f"\n[POST] Rebuilding saved outputs for {plan.dataset_dir.name}: rows={row_counts}")
            wrote_any = False
            for row_count in row_counts:
                written = _regenerate_saved_outputs_for_row(
                    dataset_dir=plan.dataset_dir,
                    target_name=target_name,
                    row_count=row_count,
                    keep_search_plots=bool(args.keep_search_plots),
                )

                feature_sensitivities, _, _ = _load_feature_stats_artifacts(
                    dataset_dir=plan.dataset_dir,
                    row_count=row_count,
                )
                if feature_sensitivities:
                    if target_name not in sweep_results:
                        sweep_results[target_name] = {}
                    sweep_results[target_name][row_count] = feature_sensitivities

                if written:
                    wrote_any = True
                    for label, path in written.items():
                        print(f"[POST] Wrote {label}: {path}")
                else:
                    print(
                        f"[WARN] Could not rebuild plots for {plan.dataset_dir.name} rows={row_count}; "
                        "missing feature stats/delta artifacts."
                    )

            # Collect best model performance for summary plot, re-evaluating best model with LOOCV at runtime
            try:
                final_metrics_csv = plan.dataset_dir / "forecasts" / "feature_sweeps" / "feature_sweep_final_metrics.csv"
                if final_metrics_csv.exists():
                    df = pd.read_csv(final_metrics_csv)
                    if not df.empty:
                        # Only consider rows with valid (non-NaN, >0) std_target and r2
                        df_valid = df.copy()
                        if 'std_target' in df_valid.columns:
                            df_valid = df_valid[(df_valid['std_target'].notnull()) & (df_valid['std_target'] > 0)]
                        if 'r2' in df_valid.columns:
                            df_valid = df_valid[df_valid['r2'].notnull()]
                        # Compute nrmse for all valid rows
                        if 'std_target' in df_valid.columns:
                            df_valid['nrmse'] = df_valid['rmse'] / df_valid['std_target']
                        else:
                            df_valid['nrmse'] = np.nan
                        # Always select the best model by highest R², and report both nRMSE and R² from that row
                        best_row = None
                        best_row_idx = None
                        valid_r2 = df_valid[df_valid['r2'].notnull() & np.isfinite(df_valid['r2'])]
                        if not valid_r2.empty:
                            # Find the index in the original DataFrame
                            idx_in_valid = valid_r2['r2'].idxmax()
                            best_row = valid_r2.loc[idx_in_valid]
                            # Map back to the original DataFrame index
                            if idx_in_valid in df.index:
                                best_row_idx = idx_in_valid
                            else:
                                # fallback: try to match on unique columns
                                best_row_idx = None
                        if best_row is not None:
                            nrmse = best_row['nrmse'] if 'nrmse' in best_row and pd.notnull(best_row['nrmse']) else float('nan')
                            r2 = best_row['r2'] if 'r2' in best_row and pd.notnull(best_row['r2']) else float('nan')
                            rmse = best_row['rmse'] if 'rmse' in best_row and pd.notnull(best_row['rmse']) else float('nan')
                            n_test_samples = best_row['n_samples'] if 'n_samples' in best_row and pd.notnull(best_row['n_samples']) else float('nan')
                            # Strictly require these fields to be present
                            required_fields = ['feature_tag', 'row_count', 'model', 'subset_rank']
                            for field in required_fields:
                                if field not in best_row or pd.isnull(best_row[field]):
                                    raise ValueError(f"Required field '{field}' is missing in best_row: {best_row}")
                            feature_tag = best_row['feature_tag']
                            row_count_val = int(best_row['row_count'])
                            model = best_row['model']
                            subset_rank = int(best_row['subset_rank'])
                            forecast_name = best_row.get('forecast_name', None)
                            if forecast_name:
                                model_dir = plan.dataset_dir / "forecasts" / forecast_name
                            else:
                                # Construct model directory name with model type and index
                                # Use consistent abbreviation for model_dir_name
                                model_abbrev_map = {
                                    'transformer': 'transformer',
                                    'xgbregressor': 'xgb',
                                    'gpregressor': 'gp',
                                }
                                model_name = str(model).strip().lower()
                                abbrev = model_abbrev_map.get(model_name, model_name)
                                model_dir_name = f"{abbrev}_01_r{row_count_val:03d}_{feature_tag}_k{subset_rank:02d}"
                                model_dir = plan.dataset_dir / "forecasts" / "feature_sweeps" / model_dir_name
                            subset_rank_str = f"_k{subset_rank:02d}.yml"
                            search_pattern = f"config_evaluate*{subset_rank_str}"
                            print(f"[DEBUG] Attempting to find model-specific config:")
                            print(f"[DEBUG] Constructed model_dir: {model_dir}")
                            resolved_model_dir = model_dir.resolve() if hasattr(model_dir, 'resolve') else model_dir
                            print(f"[DEBUG] Resolved model_dir (actual path checked): {resolved_model_dir}")
                            print(f"[DEBUG] search_pattern: {search_pattern}")
                            print(f"[DEBUG] Checking if directory exists: {resolved_model_dir}")
                            if model_dir.exists():
                                print(f"[DEBUG] Files in model_dir: {[p.name for p in model_dir.iterdir()]}")
                            else:
                                print(f"[DEBUG] model_dir does not exist!")
                            cfg_candidates = list(model_dir.glob(search_pattern))
                            print(f"[DEBUG] cfg_candidates: {[str(p) for p in cfg_candidates]}")
                            if not cfg_candidates:
                                print(f"[WARN] Could not find model-specific config for best model in {plan.dataset_dir.name}")
                                print(f"[DEBUG] Search pattern: {search_pattern}")
                                print(f"[DEBUG] Model dir: {model_dir}")
                                if model_dir.exists():
                                    all_files = [p.name for p in model_dir.iterdir()]
                                    print(f"[DEBUG] All files in model_dir: {all_files}")
                                    # Try case-insensitive match for pattern
                                    ci_matches = [f for f in all_files if fnmatch.fnmatch(f.lower(), search_pattern.lower())]
                                    print(f"[DEBUG] Case-insensitive matches for pattern '{search_pattern}': {ci_matches}")
                                    if ci_matches:
                                        print(f"[WARN] Files exist that match the pattern case-insensitively but not case-sensitively. Filesystem may be case-sensitive.")
                                else:
                                    print(f"[DEBUG] model_dir does not exist!")
                                continue
                            cfg_path = cfg_candidates[0]
                            # --- PATCH: Always use model_dir for LOOCV summary lookup ---
                            loocv_summary_path = model_dir / "loocv_summary.csv"
                            print(f"[DEBUG] Checking for loocv_summary.csv at: {loocv_summary_path}")
                            # For all models, including transformers, patch config YAML from configs subdirectory for LOOCV
                            print(f"[DEBUG] Loading config for LOOCV from: {cfg_path}")
                            with open(cfg_path, "r", encoding="utf-8") as f:
                                orig_cfg = yaml.safe_load(f)
                            print(f"[DEBUG] Config keys loaded: {list(orig_cfg.keys())}")
                            cfg = dict(orig_cfg)  # shallow copy
                            if "evaluation" not in cfg:
                                cfg["evaluation"] = {}
                            cfg["evaluation"]["run_loocv"] = True
                            loocv_cfg_path = model_dir / "config_loocv.yml"
                            with open(loocv_cfg_path, "w", encoding="utf-8") as f:
                                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
                            print(f"[DEBUG] About to run LOOCV evaluation:")
                            print(f"[DEBUG]   loocv_cfg_path: {loocv_cfg_path}")
                            print(f"[DEBUG]   model_dir: {model_dir}")
                            print(f"[DEBUG]   Should expect split files at: {model_dir / 'test_files.txt'} and {model_dir / 'train_files.txt'}")
                            try:

                                feval_path = str((Path(__file__).parent.parent / "src" / "f_Evaluate.py").resolve())
                                spec = importlib.util.spec_from_file_location("f_Evaluate", feval_path)
                                feval = importlib.util.module_from_spec(spec)
                                sys.modules["f_Evaluate"] = feval
                                spec.loader.exec_module(feval)
                                # Directly call LOOCV evaluation so debug output is visible
                                feval.evaluate_single_config(str(loocv_cfg_path), save_plots_override=False)
                            except Exception as e:
                                print(f"[WARN] Error during LOOCV evaluation for {plan.dataset_dir.name}: {e}")
                                print(f"[WARN]   loocv_cfg_path: {loocv_cfg_path}")
                                print(f"[WARN]   model_dir: {model_dir}")
                                print(f"[WARN]   Expected split files at: {model_dir / 'test_files.txt'} and {model_dir / 'train_files.txt'}")
                                traceback.print_exc()
                            # Always use the already-resolved model_dir for LOOCV and split file lookups
                            loocv_summary_path = model_dir / "loocv_summary.csv"
                            test_files_path = model_dir / "test_files.txt"
                            train_files_path = model_dir / "train_files.txt"
                            print(f"[DEBUG] Checking for loocv_summary.csv at: {loocv_summary_path}")
                            print(f"[DEBUG] File exists: {loocv_summary_path.exists()}")
                            loocv_r2 = loocv_rmse = loocv_mae = float('nan')
                            loocv_skill_v_naive = float('nan')
                            if loocv_summary_path.exists():
                                try:
                                    print(f"[DEBUG] Attempting to read LOOCV summary from: {loocv_summary_path}")
                                    df_loocv = pd.read_csv(loocv_summary_path)
                                    print(f"[DEBUG] loocv_summary.csv shape: {df_loocv.shape}")
                                    print(f"[DEBUG] loocv_summary.csv columns: {df_loocv.columns.tolist()}")
                                    print(f"[DEBUG] loocv_summary.csv head:\n{df_loocv.head()}\n")
                                    if not df_loocv.empty:
                                        agg_row = df_loocv.iloc[0]
                                        def safe_float(val):
                                            raw = val
                                            print(f"[DEBUG] Raw LOOCV value: {raw!r} (type: {type(raw)})")
                                            if pd.isnull(raw):
                                                return float('nan')
                                            if isinstance(raw, str):
                                                raw = raw.strip()
                                                if raw == '' or raw.lower() == 'nan':
                                                    return float('nan')
                                            try:
                                                return float(raw)
                                            except Exception:
                                                return float('nan')
                                        # Print raw values and types before conversion
                                        print(f"[DEBUG] Raw r2 from CSV: {agg_row.get('r2', None)!r} (type: {type(agg_row.get('r2', None))})")
                                        print(f"[DEBUG] Raw rmse from CSV: {agg_row.get('rmse', None)!r} (type: {type(agg_row.get('rmse', None))})")
                                        print(f"[DEBUG] Raw mae from CSV: {agg_row.get('mae', None)!r} (type: {type(agg_row.get('mae', None))})")
                                        loocv_r2 = safe_float(agg_row['r2']) if 'r2' in agg_row else float('nan')
                                        loocv_rmse = safe_float(agg_row['rmse']) if 'rmse' in agg_row else float('nan')
                                        loocv_mae = safe_float(agg_row['mae']) if 'mae' in agg_row else float('nan')
                                        if 'skill_v_naive' in agg_row:
                                            loocv_skill_v_naive = safe_float(agg_row['skill_v_naive'])
                                        print(f"[DEBUG] Extracted LOOCV values: r2={loocv_r2}, rmse={loocv_rmse}, mae={loocv_mae}, skill_v_naive={loocv_skill_v_naive}")
                                except Exception as e:
                                    print(f"[WARN] Could not parse LOOCV summary for {plan.dataset_dir.name}: {e}")
                            # Write LOOCV results into feature_sweep_final_metrics.csv for the best model row only (column-based match)

                            # Robust: Always append a row, even if LOOCV or update fails
                            try:
                                df_metrics = pd.read_csv(final_metrics_csv)
                                # Add columns if missing
                                for col in ['loocv_r2', 'loocv_rmse', 'loocv_mae', 'loocv_skill_v_naive']:
                                    if col not in df_metrics.columns:
                                        df_metrics[col] = float('nan')
                                # Use column-based matching to find the correct row using best_row values
                                feature_tag = best_row.get('feature_tag', '')
                                row_count_val = int(best_row.get('row_count', 0))
                                model = best_row.get('model', None)
                                row_mask = (
                                    (df_metrics['feature_tag'] == feature_tag)
                                    & (df_metrics['row_count'] == row_count_val)
                                    & (df_metrics['model'] == model)
                                )
                                print(f"[DEBUG] Attempting to write LOOCV results for dataset: {plan.dataset_dir.name}")
                                print(f"[DEBUG] feature_tag: {feature_tag}, row_count: {row_count_val}, model: {model}")
                                print(f"[DEBUG] LOOCV values: r2={loocv_r2}, rmse={loocv_rmse}, mae={loocv_mae}")
                                if row_mask.any():
                                    print(f"[DEBUG] Row(s) before update: {df_metrics.loc[row_mask].to_dict('records')}")
                                    df_metrics.loc[row_mask, 'loocv_r2'] = loocv_r2
                                    df_metrics.loc[row_mask, 'loocv_rmse'] = loocv_rmse
                                    df_metrics.loc[row_mask, 'loocv_mae'] = loocv_mae
                                    df_metrics.loc[row_mask, 'loocv_skill_v_naive'] = loocv_skill_v_naive
                                    print(f"[DEBUG] Row(s) after update: {df_metrics.loc[row_mask].to_dict('records')}")
                                    print(f"[POST] Wrote LOOCV results to feature_sweep_final_metrics.csv for best model in {plan.dataset_dir.name}")
                                else:
                                    print(f"[WARN] Could not find matching row for LOOCV update in feature_sweep_final_metrics.csv for {plan.dataset_dir.name}")
                                df_metrics.to_csv(final_metrics_csv, index=False)
                            except Exception as e:
                                print(f"[WARN] Could not write LOOCV results to feature_sweep_final_metrics.csv for {plan.dataset_dir.name}: {e}")
                            finally:
                                # Always append a row, even if LOOCV failed
                                try:
                                    # Re-read the best row from the updated metrics file to ensure LOOCV values are included
                                    try:
                                        df_metrics_updated = pd.read_csv(final_metrics_csv)
                                        # Use the same selection logic as above
                                        df_valid = df_metrics_updated.copy()
                                        if 'std_target' in df_valid.columns:
                                            df_valid = df_valid[(df_valid['std_target'].notnull()) & (df_valid['std_target'] > 0)]
                                        if 'r2' in df_valid.columns:
                                            df_valid = df_valid[df_valid['r2'].notnull()]
                                        if 'std_target' in df_valid.columns:
                                            df_valid['nrmse'] = df_valid['rmse'] / df_valid['std_target']
                                        else:
                                            df_valid['nrmse'] = np.nan
                                        valid_r2 = df_valid[df_valid['r2'].notnull() & np.isfinite(df_valid['r2'])]
                                        if not valid_r2.empty:
                                            idx_in_valid = valid_r2['r2'].idxmax()
                                            best_row_updated = valid_r2.loc[idx_in_valid]
                                            best_model_performance.append({
                                                'dataset': plan.dataset_dir.name,
                                                'nrmse': best_row_updated['nrmse'] if 'nrmse' in best_row_updated and pd.notnull(best_row_updated['nrmse']) else float('nan'),
                                                'rmse': best_row_updated['rmse'] if 'rmse' in best_row_updated and pd.notnull(best_row_updated['rmse']) else float('nan'),
                                                'r2': best_row_updated['r2'] if 'r2' in best_row_updated and pd.notnull(best_row_updated['r2']) else float('nan'),
                                                'n_test_samples': best_row_updated['n_samples'] if 'n_samples' in best_row_updated and pd.notnull(best_row_updated['n_samples']) else float('nan'),
                                                'loocv_r2': best_row_updated['loocv_r2'] if 'loocv_r2' in best_row_updated and pd.notnull(best_row_updated['loocv_r2']) else float('nan'),
                                                'loocv_rmse': best_row_updated['loocv_rmse'] if 'loocv_rmse' in best_row_updated and pd.notnull(best_row_updated['loocv_rmse']) else float('nan'),
                                                'loocv_mae': best_row_updated['loocv_mae'] if 'loocv_mae' in best_row_updated and pd.notnull(best_row_updated['loocv_mae']) else float('nan'),
                                                'loocv_skill_v_naive': best_row_updated['loocv_skill_v_naive'] if 'loocv_skill_v_naive' in best_row_updated and pd.notnull(best_row_updated['loocv_skill_v_naive']) else float('nan'),
                                            })
                                        else:
                                            # fallback: use previous values
                                            best_model_performance.append({
                                                'dataset': plan.dataset_dir.name,
                                                'nrmse': nrmse,
                                                'rmse': rmse,
                                                'r2': r2,
                                                'n_test_samples': n_test_samples,
                                                'loocv_r2': loocv_r2,
                                                'loocv_rmse': loocv_rmse,
                                                'loocv_mae': loocv_mae,
                                                'loocv_skill_v_naive': loocv_skill_v_naive,
                                            })
                                    except Exception as reread_exc:
                                        print(f"[WARN] Could not re-read updated metrics for LOOCV: {reread_exc}")
                                        best_model_performance.append({
                                            'dataset': plan.dataset_dir.name,
                                            'nrmse': nrmse,
                                            'rmse': rmse,
                                            'r2': r2,
                                            'n_test_samples': n_test_samples,
                                            'loocv_r2': loocv_r2,
                                            'loocv_rmse': loocv_rmse,
                                            'loocv_mae': loocv_mae,
                                            'loocv_skill_v_naive': loocv_skill_v_naive,
                                        })
                                except Exception as append_exc:
                                    print(f"[ERROR] Could not append best model performance for {plan.dataset_dir.name}: {append_exc}")
            except Exception as e:
                print(f"[WARN] Could not process best model performance for {plan.dataset_dir.name}: {e}")

            if wrote_any:
                datasets_with_outputs += 1

        # Generate summary_best_model_performance.png (nRMSE and R²)
        try:
            if best_model_performance:
                # --- Augment with baseline stats ---
                # Find the corresponding evaluation summary for each dataset
                summaries_dir = (data_root.parent / "regression" / "summaries").resolve()
                summaries_dir.mkdir(parents=True, exist_ok=True)
                # Try to find the evaluation summary for each dataset
                for entry in best_model_performance:
                    dataset = entry['dataset']
                    # Try to find the evaluation summary CSV for this dataset
                    # Assume standard location: data/output/regression/<dataset>/evaluation_summary.csv
                    eval_csv = os.path.join(data_root, dataset, 'evaluation_summary.csv')
                    baseline_stats = {'naive': {}, 'seasonal': {}, 'linear': {}}
                    if os.path.exists(eval_csv):
                        try:
                            df_eval = pd.read_csv(eval_csv)
                            for kind in baseline_stats.keys():
                                row = df_eval[df_eval['label'].str.lower().str.contains(kind)].iloc[0] if not df_eval[df_eval['label'].str.lower().str.contains(kind)].empty else None
                                if row is not None:
                                    for stat in ['mae','rmse','r2']:
                                        baseline_stats[kind][stat] = row.get(stat, np.nan)
                                else:
                                    for stat in ['mae','rmse','r2']:
                                        baseline_stats[kind][stat] = np.nan
                        except Exception as e:
                            print(f"[WARN] Could not read baseline stats for {dataset}: {e}")
                            for kind in baseline_stats.keys():
                                for stat in ['mae','rmse','r2']:
                                    baseline_stats[kind][stat] = np.nan
                    else:
                        for kind in baseline_stats.keys():
                            for stat in ['mae','rmse','r2']:
                                baseline_stats[kind][stat] = np.nan
                    # Add to entry
                    for kind in baseline_stats.keys():
                        for stat in ['mae','rmse','r2']:
                            entry[f'{kind}_{stat}'] = baseline_stats[kind][stat]

                perf_df = pd.DataFrame(best_model_performance)
                perf_df = perf_df.sort_values('r2', ascending=False)
                summary_csv = summaries_dir / "summary_best_model_performance.csv"
                perf_df.to_csv(summary_csv, index=False)
                print(f"[INFO] Wrote summary CSV: {summary_csv}")
                # Clustered bar chart for model and baselines
                x = np.arange(len(perf_df))
                width = 0.2
                labels = perf_df['dataset']
                methods = ['model', 'naive', 'seasonal', 'linear']
                # Prepare data for each metric
                nrmse_data = [
                    perf_df['nrmse'],
                    perf_df['naive_rmse'] / perf_df['std_target'],
                    perf_df['seasonal_rmse'] / perf_df['std_target'],
                    perf_df['linear_rmse'] / perf_df['std_target'],
                ]
                r2_data = [
                    perf_df['r2'],
                    perf_df['naive_r2'],
                    perf_df['seasonal_r2'],
                    perf_df['linear_r2'],
                ]
                # For skill_v_naive, only model is relevant, but for completeness, show NaN for baselines
                skill_data = [
                    perf_df['loocv_skill_v_naive'],
                    [float('nan')]*len(perf_df),
                    [float('nan')]*len(perf_df),
                    [float('nan')]*len(perf_df),
                ]
                colors = ['tab:blue', 'tab:gray', 'tab:green', 'tab:orange']
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(12, len(perf_df)*0.8), 13), constrained_layout=True, sharex=True)
                # nRMSE subplot
                for i, (vals, color, method) in enumerate(zip(nrmse_data, colors, methods)):
                    bars = ax1.bar(x + (i-1.5)*width, vals, width, label=method.capitalize(), color=color)
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax1.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2e}', ha='center', va='bottom', fontsize=8, rotation=90)
                ax1.set_ylabel('nRMSE')
                ax1.grid(axis='y', alpha=0.3)
                ax1.legend()
                # R2 subplot
                for i, (vals, color, method) in enumerate(zip(r2_data, colors, methods)):
                    bars = ax2.bar(x + (i-1.5)*width, vals, width, label=method.capitalize(), color=color)
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax2.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
                ax2.set_ylabel('R²')
                ax2.grid(axis='y', alpha=0.3)
                ax2.legend()
                # Skill vs Naive subplot
                for i, (vals, color, method) in enumerate(zip(skill_data, colors, methods)):
                    bars = ax3.bar(x + (i-1.5)*width, vals, width, label=method.capitalize(), color=color)
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax3.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
                ax3.set_ylabel('LOOCV Skill vs Naive')
                ax3.grid(axis='y', alpha=0.3)
                ax3.legend()
                # X axis
                ax3.set_xticks(x)
                ax3.set_xticklabels(labels, rotation=45, ha='right')
                fig.suptitle('Best Model and Baseline Performance per Dataset (nRMSE, R², Skill vs Naive)')
                plot_path = summaries_dir / "summary_best_model_performance.png"
                fig.savefig(plot_path, dpi=180, bbox_inches='tight')
                plt.close(fig)
                print(f"[INFO] Wrote summary_best_model_performance.png to {plot_path}")

                # --- Standalone publication-quality subplots ---
                # nRMSE only
                fig_nrmse, ax_nrmse = plt.subplots(figsize=(max(10, len(perf_df)*0.7), 5), constrained_layout=True)
                for i, (vals, color, method) in enumerate(zip(nrmse_data, colors, methods)):
                    bars = ax_nrmse.bar(x + (i-1.5)*width, vals, width, label=method.capitalize(), color=color)
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax_nrmse.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2e}', ha='center', va='bottom', fontsize=8, rotation=90)
                ax_nrmse.set_ylabel('nRMSE')
                ax_nrmse.set_xticks(x)
                ax_nrmse.set_xticklabels(labels, rotation=45, ha='right')
                ax_nrmse.grid(axis='y', alpha=0.3)
                ax_nrmse.legend()
                fig_nrmse.suptitle('Best Model and Baseline nRMSE per Dataset')
                nrmse_path = summaries_dir / "summary_best_model_nrmse.png"
                fig_nrmse.savefig(nrmse_path, dpi=300, bbox_inches='tight')
                plt.close(fig_nrmse)
                print(f"[INFO] Wrote nRMSE subplot: {nrmse_path}")

                # R² only
                fig_r2, ax_r2 = plt.subplots(figsize=(max(10, len(perf_df)*0.7), 5), constrained_layout=True)
                for i, (vals, color, method) in enumerate(zip(r2_data, colors, methods)):
                    bars = ax_r2.bar(x + (i-1.5)*width, vals, width, label=method.capitalize(), color=color)
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax_r2.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
                ax_r2.set_ylabel('R²')
                ax_r2.set_xticks(x)
                ax_r2.set_xticklabels(labels, rotation=45, ha='right')
                ax_r2.grid(axis='y', alpha=0.3)
                ax_r2.legend()
                fig_r2.suptitle('Best Model and Baseline R² per Dataset')
                r2_path = summaries_dir / "summary_best_model_r2.png"
                fig_r2.savefig(r2_path, dpi=300, bbox_inches='tight')
                plt.close(fig_r2)
                print(f"[INFO] Wrote R² subplot: {r2_path}")

                # Skill vs Naive only
                fig_skill, ax_skill = plt.subplots(figsize=(max(10, len(perf_df)*0.7), 5), constrained_layout=True)
                for i, (vals, color, method) in enumerate(zip(skill_data, colors, methods)):
                    bars = ax_skill.bar(x + (i-1.5)*width, vals, width, label=method.capitalize(), color=color)
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax_skill.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
                ax_skill.set_ylabel('LOOCV Skill vs Naive')
                ax_skill.set_xticks(x)
                ax_skill.set_xticklabels(labels, rotation=45, ha='right')
                ax_skill.grid(axis='y', alpha=0.3)
                ax_skill.legend()
                fig_skill.suptitle('Best Model and Baseline LOOCV Skill vs Naive per Dataset')
                skill_path = summaries_dir / "summary_best_model_skill_vs_naive.png"
                fig_skill.savefig(skill_path, dpi=300, bbox_inches='tight')
                plt.close(fig_skill)
                print(f"[INFO] Wrote Skill vs Naive subplot: {skill_path}")
            else:
                print("[WARN] No best model performance data found; summary plot not generated.")
        except Exception as e:
            print(f"[ERROR] Failed to generate summary_best_model_performance.png: {e}")

        if len(sweep_results) > 1:
            print("\n" + "=" * 100)
            print("MULTI-TARGET FEATURE IMPORTANCE COMPARISON (POSTPROCESS)")
            print("=" * 100)
            try:
                comparison_plot = _compile_multi_target_comparison(sweep_results, data_root)
                if comparison_plot.exists():
                    print(f"[INFO] Wrote multi-target comparison plots to {comparison_plot.parent}")
            except Exception as e:
                print(f"[WARN] Failed to regenerate multi-target comparison: {e}")

        if datasets_with_outputs == 0:
            print("[WARN] No dataset outputs were regenerated.")
            return 1

        print(f"[INFO] Regenerated feature-sweep outputs for {datasets_with_outputs} dataset(s).")
        return 0

    # Track feature importance across targets for multi-target comparison
    sweep_results: dict[str, dict[int, dict[str, tuple[float, int]]]] = {}  # target -> row_count -> feature_sensitivities

    failed = 0
    for plan in plans:
        print("\n" + "=" * 100)
        print(f"DATASET: {plan.dataset_dir.name}")
        print("=" * 100)

        surrogate_cfg = _select_surrogate_config(plan.train_configs)
        surrogate_data = train_module.load_config(str(surrogate_cfg))["data"]
        base_span = int(surrogate_data["input_row_2"]) - int(surrogate_data["input_row_1"])
        row_counts = _parse_row_counts(args.row_counts, default_span=base_span)

        for row_count in row_counts:
            try:
                print(f"\n[SEARCH] rows={row_count} surrogate={surrogate_cfg.name}")
                top_sorted, trace, feature_sensitivities = _beam_search_subsets(
                    dataset_dir=plan.dataset_dir,
                    dataset_prefix=args.dataset_prefix,
                    surrogate_config_path=surrogate_cfg,
                    row_count=row_count,
                    lambda_drop=args.lambda_drop,
                    beam_width=args.beam_width,
                    max_rounds=args.max_rounds,
                    no_improve_patience=args.no_improve_patience,
                    min_features=args.min_features,
                    eval_budget=args.eval_budget,
                    max_swap_attempts=args.max_swap_attempts,
                    disable_baselines_for_search=args.disable_baselines_for_search,
                    disable_training_plots=not args.keep_training_plots,
                    disable_eval_plots=not args.keep_eval_plots,
                    suppress_training_logs=not args.show_training_logs,
                    seed=args.seed,
                )
                # Track feature importance for multi-target comparison
                target_name = _derive_target_name(plan.dataset_dir.name, args.dataset_prefix)
                if target_name not in sweep_results:
                    sweep_results[target_name] = {}
                sweep_results[target_name][row_count] = feature_sensitivities

                selected = top_sorted[: args.final_top_k]
                trace_csv, selected_csv, plot_path = _write_search_outputs(
                    dataset_dir=plan.dataset_dir,
                    row_count=row_count,
                    trace=trace,
                    selected=selected,
                    save_plots=bool(args.keep_search_plots),
                )
                print(f"[INFO] Wrote search trace: {trace_csv}")
                print(f"[INFO] Wrote selected subsets: {selected_csv}")
                print(f"[INFO] Wrote search plot: {plot_path}")

                final_metrics_csv = _evaluate_selected_subsets_all_models(
                    dataset_plan=plan,
                    dataset_prefix=args.dataset_prefix,
                    selected=selected,
                    run_baselines_in_final=args.run_baselines_in_final,
                    disable_training_plots=not args.keep_training_plots,
                    disable_eval_plots=not args.keep_eval_plots,
                    suppress_training_logs=not args.show_training_logs,
                )
                print(f"[INFO] Wrote final model metrics: {final_metrics_csv}")

                # --- LOOCV for best model (highest R2) ---
                if selected:
                    try:
                        # Pick the best by R2
                        best_model = max(selected, key=lambda c: c.r2 if np.isfinite(c.r2) else float('-inf'))
                        # Find config path for best model
                        best_cfg_path = None
                        for cfg in plan.train_configs:
                            if best_model.feature_tag in str(cfg):
                                best_cfg_path = cfg
                                break
                        if best_cfg_path is None:
                            best_cfg_path = plan.train_configs[0]
                        # --- PATCH START: Write LOOCV config to correct model dir and run f_Evaluate.py ---
                        # Consistent model abbreviation mapping
                        model_abbrev_map = {
                            'Transformer': 'transformer',
                            'XGBRegressor': 'xgb',
                            'GPRegressor': 'gp',
                        }
                        model_name = str(best_model.model) if hasattr(best_model, 'model') else None
                        abbrev = model_abbrev_map.get(model_name, model_name.lower() if model_name else "unknown")
                        mapped_model_name = f"{abbrev}_01"
                        row_count_val = int(best_model.row_count)
                        feature_tag = str(best_model.feature_tag)
                        subset_rank_val = int(getattr(best_model, 'subset_rank', 1))
                        subset_rank_str = f"k{subset_rank_val:02d}"
                        model_dir = plan.dataset_dir / 'forecasts' / 'feature_sweeps' / f"{mapped_model_name}_r{row_count_val}_{feature_tag}_{subset_rank_str}"
                        model_dir.mkdir(parents=True, exist_ok=True)
                        configs_dir = plan.dataset_dir / 'forecasts' / 'feature_sweeps' / 'configs'
                        config_path = None
                        for cfg_file in configs_dir.glob(f"*{abbrev}*r{row_count_val:03d}_{feature_tag}*.yml"):
                            config_path = cfg_file
                            break
                        if config_path is None:
                            for cfg_file in configs_dir.glob(f"*r{row_count_val:03d}_{feature_tag}*.yml"):
                                config_path = cfg_file
                                break
                        if config_path is None:
                            print(f"[WARN] Could not find config for LOOCV: {mapped_model_name}, r{row_count_val}, {feature_tag}")
                            continue
                        with open(config_path, 'r', encoding='utf-8') as f:
                            cfg = yaml.safe_load(f)
                        if 'evaluation' not in cfg:
                            cfg['evaluation'] = {}
                        cfg['evaluation']['run_loocv'] = True
                        cfg['data']['forecast_name'] = f"feature_sweeps/{mapped_model_name}_r{row_count_val}_{feature_tag}_{subset_rank_str}"
                        loocv_cfg_path = model_dir / "config_loocv.yml"
                        with open(loocv_cfg_path, 'w', encoding='utf-8') as f:
                            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
                        eval_cmd = [sys.executable, 'src/f_Evaluate.py', '--config', str(loocv_cfg_path)]
                        print(f"[INFO] Running LOOCV via f_Evaluate.py: {' '.join(eval_cmd)}")
                        subprocess.run(eval_cmd, check=True)
                        print(f"[INFO] LOOCV complete for best model: {loocv_cfg_path}")
                        # --- PATCH END ---
                        # Always use model_dir for LOOCV metrics and split files
                        loocv_metrics_path = model_dir / f"loocv_metrics_{best_model.feature_tag}.csv"
                        test_files_path = model_dir / "test_files.txt"
                        train_files_path = model_dir / "train_files.txt"
                        loocv_metrics = {}
                        if loocv_metrics_path.exists():
                            try:
                                df_loocv = pd.read_csv(loocv_metrics_path)
                                if not df_loocv.empty:
                                    for col in df_loocv.columns:
                                        loocv_metrics[f"loocv_{col}"] = df_loocv.iloc[0][col]
                            except Exception as e:
                                print(f"[WARN] Could not parse LOOCV metrics: {e}")
                        else:
                            print(f"[WARN] LOOCV metrics file not found: {loocv_metrics_path}")
                        # Check for split files in model_dir
                        if not test_files_path.exists():
                            print(f"[WARN] test_files.txt not found in model_dir: {test_files_path}")
                        if not train_files_path.exists():
                            print(f"[WARN] train_files.txt not found in model_dir: {train_files_path}")
                        if best_model_performance:
                            for entry in best_model_performance:
                                if entry['dataset'] == plan.dataset_dir.name:
                                    entry.update(loocv_metrics)
                                    break
                    except Exception as exc:
                        print(f"[WARN] LOOCV or LOOCV summary failed for dataset {plan.dataset_dir.name}, row_count {row_count}: {exc}")
            except Exception as exc:
                failed += 1
                print(f"[ERROR] Dataset failed: {plan.dataset_dir.name}, row_count {row_count}")
                print(f"[ERROR] {exc}")
                if args.stop_on_error:
                    raise

    print("\nRun summary")
    print("-" * 100)
    print(f"Datasets completed: {len(plans) - failed}")
    print(f"Datasets failed   : {failed}")
    
    # Compile multi-target comparison if multiple targets found
    if len(sweep_results) > 1:
        print("\n" + "=" * 100)
        print("MULTI-TARGET FEATURE IMPORTANCE COMPARISON")
        print("=" * 100)
        try:
            comparison_plot = _compile_multi_target_comparison(sweep_results, data_root)
            if comparison_plot.exists():
                print(f"[INFO] Wrote multi-target comparison plots to {comparison_plot.parent}")
        except Exception as e:
            print(f"[WARN] Failed to generate multi-target comparison: {e}")
    
    return 0 if failed == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature-selection sweeper using beam+swap surrogate search and final full-model evaluation."
    )
    parser.add_argument("--data-root", type=str, default="data/output/regression")
    parser.add_argument("--dataset-prefix", type=str, default="MC")
    parser.add_argument("--config-pattern", type=str, default="config_*.yml")
    parser.add_argument("--limit-datasets", type=int, default=1)

    parser.add_argument("--row-counts", type=str, default=None)
    parser.add_argument("--min-features", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--no-improve-patience", type=int, default=3)
    parser.add_argument("--eval-budget", type=int, default=240)
    parser.add_argument("--max-swap-attempts", type=int, default=60)
    parser.add_argument("--lambda-drop", type=float, default=0.25)
    parser.add_argument("--final-top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--include-regular", action="store_true")
    parser.add_argument("--include-res", action="store_true")
    parser.add_argument("--regular-only", action="store_true")
    parser.add_argument("--res-only", action="store_true")

    parser.add_argument("--disable-baselines-for-search", action="store_true")
    parser.add_argument("--run-baselines-in-final", action="store_true")
    parser.add_argument(
        "--keep-training-plots",
        action="store_true",
        help="Keep per-model training plots during feature sweeps (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-eval-plots",
        action="store_true",
        help="Keep per-config evaluation plots during feature sweeps (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-search-plots",
        action="store_true",
        help="Keep feature-search Pareto plots (disabled by default for speed).",
    )
    parser.add_argument(
        "--show-training-logs",
        action="store_true",
        help="Show verbose model training logs (epoch metrics, sample-loading details).",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Regenerate feature-sweep plots from saved artifacts without running search/training.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    # --exclude argument removed
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_feature_selection_sweep(args)


if __name__ == "__main__":
    sys.exit(main())

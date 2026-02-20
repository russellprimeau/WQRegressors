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
python src/h_RunMCFeatureSelectionSweep.py --dry-run
python src/h_RunMCFeatureSelectionSweep.py --limit-datasets 1 --max-rounds 8 --beam-width 6
python src/h_RunMCFeatureSelectionSweep.py --row-counts 1,2,3,5 --limit-datasets 1
python src/h_RunMCFeatureSelectionSweep.py --exclude=MC_Trial1,MC_Trial2
python src/h_RunMCFeatureSelectionSweep.py --postprocess-only --keep-search-plots
"""

from __future__ import annotations
import argparse
import contextlib
import copy
import glob
import hashlib
import io
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from utils.training import load_samples, group_samples_by_segment
import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

import e_Train as train_module
import f_Evaluate as eval_module
from utils.training import load_samples


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


def _expand_config_inputs(config_args: list[str]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[str] = set()
    continuation_tokens = {"\\", "/", "`"}

    for arg in config_args:
        parts = [part.strip() for part in str(arg).split(",") if part.strip()]
        for part in parts:
            if part in continuation_tokens:
                continue

            matches = sorted(glob.glob(part)) if any(ch in part for ch in "*?[]") else [part]
            for match in matches:
                clean_match = str(match).strip()
                if clean_match in continuation_tokens:
                    continue

                match_path = Path(clean_match)
                if match_path.is_dir():
                    continue

                resolved = match_path.resolve()
                if resolved.suffix.lower() not in SUPPORTED_CONFIG_SUFFIXES:
                    continue

                key = str(resolved)
                if key not in seen:
                    seen.add(key)
                    expanded.append(resolved)

    return expanded


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
    data_cfg["forecast_name"] = _variant_forecast_name(str(data_cfg["forecast_name"]), row_count, feature_tag)

    resolved_data_dir = train_module._resolve_path_from_config(data_cfg["data_dir"], source_config_dir)
    data_cfg["data_dir"] = str(resolved_data_dir)

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
    return Path(
        data_cfg["data_dir"],
        "forecasts",
        forecast_name,
        f"config_evaluate_{forecast_file_name}.yml",
    ).resolve()


def _set_eval_overrides(eval_config_path: Path, run_baselines: bool) -> None:
    with open(eval_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "evaluation" not in cfg:
        cfg["evaluation"] = {}
    cfg["evaluation"]["run_baselines"] = bool(run_baselines)

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
    )
    eval_cfg = _train_single_config(
        variant_cfg,
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
    all_features = set()
    for target_data in sweep_results.values():
        for feature_sensitivities in target_data.values():
            all_features.update(feature_sensitivities.keys())
    all_features = sorted(all_features)
    
    if not all_features:
        return Path()
    
    # Build matrix: targets x features with removal sensitivity scores
    targets = sorted(sweep_results.keys())
    matrix = np.zeros((len(targets), len(all_features)))
    
    for i, target in enumerate(targets):
        for j, feat in enumerate(all_features):
            # Use the first (finest) row_count's data for comparison
            for feature_sensitivities in sweep_results[target].values():
                if feat in feature_sensitivities:
                    matrix[i, j] = feature_sensitivities[feat][0]  # removal sensitivity
                    break
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(all_features) * 0.4), max(8, len(targets) * 0.5)), 
                           constrained_layout=True)
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=np.percentile(matrix, 5), vmax=np.percentile(matrix, 95))
    
    ax.set_xticks(np.arange(len(all_features)))
    ax.set_yticks(np.arange(len(targets)))
    ax.set_xticklabels(all_features, rotation=45, ha='right')
    ax.set_yticklabels(targets)
    
    # Add value annotations
    for i in range(len(targets)):
        for j in range(len(all_features)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title("Feature Removal Sensitivity Heatmap Across Targets\n(Positive = valuable)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    cbar = plt.colorbar(im, ax=ax, label="Removal Sensitivity (avg delta)")
    
    # Save to root output directory
    output_dir = data_root.parent / "feature_importance_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "multi_target_importance_heatmap.png"
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # Also create a grouped bar chart for easier comparison of all features
    # Sort features by maximum removal sensitivity across all targets (descending)
    max_sensitivity = matrix.max(axis=0)
    sorted_indices = np.argsort(-max_sensitivity)
    top_features = [all_features[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(max(14, len(top_features) * 0.5), 6), constrained_layout=True)
    x = np.arange(len(top_features))
    width = 0.8 / len(targets)
    
    for i, target in enumerate(targets):
        scores = [matrix[targets.index(target), all_features.index(f)] for f in top_features]
        ax.bar(x + i * width, scores, width, label=target)
    
    ax.set_xlabel("Feature (sorted by max removal sensitivity)")
    ax.set_ylabel("Removal Sensitivity (avg delta)")
    ax.set_title(f"Feature Removal Sensitivity Across All {len(top_features)} Features and {len(targets)} Targets\n(Positive = valuable feature)")
    ax.set_xticks(x + width * (len(targets) - 1) / 2)
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    bar_path = output_dir / "multi_target_importance_bars.png"
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
        sweep_results: dict[str, dict[int, dict[str, tuple[float, int]]]] = {}
        datasets_with_outputs = 0

        for plan in plans:
            target_name = _derive_target_name(plan.dataset_dir.name, args.dataset_prefix)
            surrogate_cfg = _select_surrogate_config(plan.train_configs)
            surrogate_data = train_module.load_config(str(surrogate_cfg))["data"]
            base_span = int(surrogate_data["input_row_2"]) - int(surrogate_data["input_row_1"])
            requested_rows = _parse_row_counts(args.row_counts, default_span=base_span)
            row_counts = requested_rows if args.row_counts else _available_row_counts_for_postprocess(plan.dataset_dir)

            if not row_counts:
                print(f"[WARN] No saved feature-sweep artifacts found for {plan.dataset_dir.name}; skipping.")
                continue

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

            if wrote_any:
                datasets_with_outputs += 1

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

        try:
            surrogate_cfg = _select_surrogate_config(plan.train_configs)
            surrogate_data = train_module.load_config(str(surrogate_cfg))["data"]
            base_span = int(surrogate_data["input_row_2"]) - int(surrogate_data["input_row_1"])
            row_counts = _parse_row_counts(args.row_counts, default_span=base_span)

            for row_count in row_counts:
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
                    # Load config, set run_loocv: true, save to temp file
                    import yaml, tempfile, shutil
                    with open(best_cfg_path, 'r', encoding='utf-8') as f:
                        cfg = yaml.safe_load(f)
                    if 'evaluation' not in cfg:
                        cfg['evaluation'] = {}
                    cfg['evaluation']['run_loocv'] = True
                    # Save to a temp config file in the same directory
                    temp_cfg_path = plan.dataset_dir / 'forecasts' / 'feature_sweeps' / f"loocv_config_{best_model.feature_tag}.yml"
                    with open(temp_cfg_path, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(cfg, f)
                    # Call f_Evaluate.py on this config
                    import subprocess
                    eval_cmd = [sys.executable, 'src/f_Evaluate.py', '--config', str(temp_cfg_path)]
                    print(f"[INFO] Running LOOCV via f_Evaluate.py: {' '.join(eval_cmd)}")
                    subprocess.run(eval_cmd, check=True)
                    print(f"[INFO] LOOCV complete for best model: {temp_cfg_path}")
        except Exception as exc:
            failed += 1
            print(f"[ERROR] Dataset failed: {plan.dataset_dir.name}")
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

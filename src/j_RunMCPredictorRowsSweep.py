"""
Batch train + evaluate runner for MC datasets with predictor-row window sweeps.

Alternative 1 implemented here:
- Keep the end of the original input window fixed.
- Sweep the number of included predictor rows:
    1, 10, then 20..max in steps of 10 (customizable).
- Also run a default AVG mode that collapses each predictor to one value
    by averaging all available values in the original configured row window.
- Train/evaluate all discovered model configs for each row count.
- Plot RMSE and R² vs predictor rows for each target.

Examples:
python src/j_RunMCPredictorRowsSweep.py --dry-run
python src/j_RunMCPredictorRowsSweep.py --limit-datasets 1 --limit-rows 5
python src/j_RunMCPredictorRowsSweep.py --dataset-prefix MC --max-rows 160
"""

from __future__ import annotations
import argparse
import contextlib
import copy
import glob
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
import e_Train as train_module
import f_Evaluate as eval_module


SUPPORTED_CONFIG_SUFFIXES = {".yml", ".yaml", ".json"}


@dataclass
class DatasetPlan:
    dataset_dir: Path
    train_configs: list[Path]


@dataclass
class SweepMetricRow:
    dataset: str
    target: str
    row_label: str
    row_order: int
    row_count: int
    model: str
    n_samples: float
    n_predictors: float
    input_dim: float
    target_dim: float
    mae: float
    rmse: float
    r2: float


@dataclass
class SweepOption:
    row_label: str
    row_order: int
    row_count: int
    input_aggregation: str


def _annotate_heatmap_cells(ax, values: np.ndarray) -> None:
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return

    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    denom = (vmax - vmin) if vmax > vmin else 1.0

    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            if not np.isfinite(val):
                continue
            norm = (float(val) - vmin) / denom
            text_color = "black" if norm > 0.55 else "white"
            ax.text(j, i, f"{val:.2e}", ha="center", va="center", color=text_color, fontsize=7)


def _derive_target_name(dataset_name: str, dataset_prefix: str) -> str:
    if dataset_name.startswith(dataset_prefix):
        return dataset_name[len(dataset_prefix):].lstrip("_")
    return dataset_name


def _rowsweep_super_summary_paths(data_root: Path) -> tuple[Path, Path, Path]:
    summary_dir = data_root / "summaries"
    return (
        summary_dir / "mc_rowsweep_super_summary_models.csv",
        summary_dir / "mc_rowsweep_super_summary_best_by_target_row.csv",
        summary_dir / "mc_rowsweep_super_summary_heatmap.png",
    )


def build_super_summary_from_metrics(data_root: Path, metrics_df: pd.DataFrame) -> tuple[Path, Path, Path] | None:
    if metrics_df is None or metrics_df.empty:
        print("[INFO] No metrics available to build row-sweep super-summary.")
        return None

    model_df = metrics_df.copy()
    baseline_labels = {"Naive", "Linear", "Seasonal"}
    model_df = model_df[~model_df["model"].astype(str).isin(baseline_labels)].copy()
    if model_df.empty:
        print("[INFO] No model rows available to build row-sweep super-summary.")
        return None

    model_df["mae"] = pd.to_numeric(model_df.get("mae", np.nan), errors="coerce")
    model_df["rmse"] = pd.to_numeric(model_df.get("rmse", np.nan), errors="coerce")
    model_df["r2"] = pd.to_numeric(model_df.get("r2", np.nan), errors="coerce")
    model_df["row_count"] = pd.to_numeric(model_df.get("row_count", np.nan), errors="coerce")
    if "row_label" not in model_df.columns:
        model_df["row_label"] = model_df["row_count"].apply(lambda x: f"N{int(x)}" if pd.notna(x) else "NA")
    if "row_order" not in model_df.columns:
        model_df["row_order"] = np.arange(len(model_df), dtype=int)
    model_df["row_order"] = pd.to_numeric(model_df.get("row_order", np.nan), errors="coerce")

    # Rank row modes for each model (more informative than ranking models within one row mode).
    group_cols = ["dataset", "target", "model"]
    model_df["rank_rmse"] = model_df.groupby(group_cols)["rmse"].rank(method="min", ascending=True)
    model_df["rank_r2"] = model_df.groupby(group_cols)["r2"].rank(method="min", ascending=False)
    model_df["is_best_rmse"] = model_df["rank_rmse"] == 1
    model_df["is_best_r2"] = model_df["rank_r2"] == 1

    ordered_cols = [
        "dataset",
        "target",
        "row_label",
        "row_order",
        "row_count",
        "model",
        "n_samples",
        "n_predictors",
        "input_dim",
        "target_dim",
        "mae",
        "rmse",
        "r2",
        "rank_rmse",
        "rank_r2",
        "is_best_rmse",
        "is_best_r2",
    ]
    for col in ordered_cols:
        if col not in model_df.columns:
            model_df[col] = np.nan
    model_df = model_df[ordered_cols].sort_values(
        ["dataset", "target", "model", "rank_rmse", "row_order", "row_label"],
        kind="stable",
    )

    best_rows = []
    for (dataset, target, model), group in model_df.groupby(["dataset", "target", "model"], sort=True):
        best_rmse = group.loc[group["rank_rmse"] == 1]
        best_r2 = group.loc[group["rank_r2"] == 1]

        for _, row in best_rmse.iterrows():
            best_rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "model": model,
                    "row_label": row["row_label"],
                    "row_count": row["row_count"],
                    "selection": "best_rmse_row_mode",
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "r2": row["r2"],
                    "n_samples": row["n_samples"],
                    "input_dim": row["input_dim"],
                }
            )
        for _, row in best_r2.iterrows():
            best_rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "model": model,
                    "row_label": row["row_label"],
                    "row_count": row["row_count"],
                    "selection": "best_r2_row_mode",
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "r2": row["r2"],
                    "n_samples": row["n_samples"],
                    "input_dim": row["input_dim"],
                }
            )

    best_df = pd.DataFrame(
        best_rows,
        columns=[
            "dataset",
            "target",
            "model",
            "row_label",
            "row_count",
            "selection",
            "mae",
            "rmse",
            "r2",
            "n_samples",
            "input_dim",
        ],
    )

    super_csv_path, best_csv_path, plot_path = _rowsweep_super_summary_paths(data_root)
    super_csv_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(super_csv_path, index=False)
    best_df.to_csv(best_csv_path, index=False)

    row_order_map = (
        model_df[["row_label", "row_order"]]
        .drop_duplicates(subset=["row_label"], keep="first")
        .sort_values(["row_order", "row_label"], kind="stable")
    )
    ordered_row_labels = row_order_map["row_label"].astype(str).tolist()

    rmse_heat = model_df.pivot_table(index="row_label", columns="model", values="rmse", aggfunc="mean")
    r2_heat = model_df.pivot_table(index="row_label", columns="model", values="r2", aggfunc="mean")

    all_rows = [label for label in ordered_row_labels if label in rmse_heat.index or label in r2_heat.index]
    all_models = sorted(set(rmse_heat.columns).union(set(r2_heat.columns)))
    rmse_heat = rmse_heat.reindex(index=all_rows, columns=all_models)
    r2_heat = r2_heat.reindex(index=all_rows, columns=all_models)

    fig_h = max(4.5, 0.45 * len(all_rows) + 1.6)
    fig_w = max(8.0, 1.1 * len(all_models) + 3.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w * 1.7, fig_h), constrained_layout=True)

    rmse_arr = rmse_heat.to_numpy(dtype=float)
    r2_arr = r2_heat.to_numpy(dtype=float)

    rmse_ma = np.ma.masked_invalid(rmse_arr)
    r2_ma = np.ma.masked_invalid(r2_arr)

    im0 = axes[0].imshow(rmse_ma, aspect="auto", interpolation="nearest", cmap="viridis_r")
    axes[0].set_title("RMSE (lower is better)")
    axes[0].set_xticks(np.arange(len(all_models)))
    axes[0].set_xticklabels(all_models, rotation=45, ha="right")
    axes[0].set_yticks(np.arange(len(all_rows)))
    axes[0].set_yticklabels([str(r) for r in all_rows])
    axes[0].set_ylabel("Row mode")
    _annotate_heatmap_cells(axes[0], rmse_arr)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(r2_ma, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[1].set_title("R² (higher is better)")
    axes[1].set_xticks(np.arange(len(all_models)))
    axes[1].set_xticklabels(all_models, rotation=45, ha="right")
    axes[1].set_yticks(np.arange(len(all_rows)))
    axes[1].set_yticklabels([str(r) for r in all_rows])
    axes[1].set_ylabel("Row mode")
    _annotate_heatmap_cells(axes[1], r2_arr)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("MC Row-Sweep Super-Summary", fontsize=12)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"[INFO] Wrote row-sweep super-summary table: {super_csv_path}")
    print(f"[INFO] Wrote row-sweep best-model table: {best_csv_path}")
    print(f"[INFO] Wrote row-sweep super-summary heatmap: {plot_path}")

    return super_csv_path, best_csv_path, plot_path


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


def _model_sort_key(config_path: Path) -> tuple[int, str]:
    name = config_path.name.lower()
    if "gp" in name:
        return 0, name
    if "transformer" in name:
        return 1, name
    if "xgb" in name:
        return 2, name
    return 3, name


def _row_count_schedule(max_rows: int) -> list[int]:
    if max_rows < 1:
        return []

    default_milestones = [1, 12, 24, 48, 96, 144, 168]
    rows = [row for row in default_milestones if row <= int(max_rows)]
    return rows


def _build_sweep_options(row_counts: list[int], avg_label: str = "AVG") -> list[SweepOption]:
    options: list[SweepOption] = [
        SweepOption(
            row_label=avg_label,
            row_order=0,
            row_count=1,
            input_aggregation="mean",
        )
    ]
    for idx, row_count in enumerate(row_counts, start=1):
        options.append(
            SweepOption(
                row_label=f"N{int(row_count)}",
                row_order=idx,
                row_count=int(row_count),
                input_aggregation="none",
            )
        )
    return options


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

    def _dataset_allowed(name: str) -> bool:
        if not name.startswith(dataset_prefix):
            return False
        is_res = name.endswith("_res")
        return (is_res and include_res) or ((not is_res) and include_regular)

    dataset_dirs = [path for path in sorted(data_root.iterdir()) if path.is_dir() and _dataset_allowed(path.name)]

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


def _variant_forecast_name(original_forecast_name: str, row_label: str) -> str:
    base_name = str(original_forecast_name).replace("\\", "/")
    if base_name.startswith("rowsweeps/"):
        base_name = base_name[len("rowsweeps/") :]
    return f"rowsweeps/{base_name}_{row_label}"


def _prepare_variant_config(
    base_config_path: Path,
    option: SweepOption,
    tmp_dir: Path,
) -> Path:
    cfg = train_module.load_config(str(base_config_path))
    cfg_copy = copy.deepcopy(cfg)

    if "data" not in cfg_copy:
        raise ValueError(f"Missing data section in {base_config_path}")

    data_cfg = cfg_copy["data"]
    source_config_dir = Path(cfg.get("__config_dir", base_config_path.parent))
    required_data = ["input_row_1", "input_row_2", "forecast_name"]
    for field in required_data:
        if field not in data_cfg:
            raise ValueError(f"Missing data.{field} in {base_config_path}")

    base_start = int(data_cfg["input_row_1"])
    base_stop = int(data_cfg["input_row_2"])
    base_span = base_stop - base_start
    if base_span <= 0:
        raise ValueError(f"Invalid input row span in {base_config_path}: {base_start}..{base_stop}")
    if option.input_aggregation == "none" and option.row_count > base_span:
        raise ValueError(
            f"Requested row_count={option.row_count} exceeds base span={base_span} in {base_config_path.name}"
        )

    if option.input_aggregation == "mean":
        # Average across full original window for each predictor.
        data_cfg["input_row_1"] = int(base_start)
        data_cfg["input_row_2"] = int(base_stop)
        data_cfg["input_aggregation"] = "mean"
    else:
        data_cfg["input_row_1"] = int(base_stop - option.row_count)
        data_cfg["input_row_2"] = int(base_stop)
        data_cfg["input_aggregation"] = "none"

    data_cfg["forecast_name"] = _variant_forecast_name(str(data_cfg["forecast_name"]), option.row_label)

    # Keep data directory anchored to the original config location.
    # Without this, writing variant configs under forecasts/_rowsweep_configs would make
    # relative data_dir values (often '.') incorrectly resolve to that temp folder.
    resolved_data_dir = train_module._resolve_path_from_config(data_cfg["data_dir"], source_config_dir)
    data_cfg["data_dir"] = str(resolved_data_dir)

    cfg_copy.pop("__config_dir", None)

    variant_name = f"{base_config_path.stem}_{option.row_label}{base_config_path.suffix}"
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
    print(f"    [TRAIN] Using device: {device}")

    if suppress_training_logs:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train_samples, test_samples = train_module.load_and_split_data(config)
    else:
        train_samples, test_samples = train_module.load_and_split_data(config)
        print(f"    [TRAIN] Samples loaded: train={len(train_samples)} test={len(test_samples)}")

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


def _collect_rows_from_eval_results(
    eval_results: list[dict],
    dataset_name: str,
    target_name: str,
    row_label: str,
    row_order: int,
    row_count: int,
    include_baselines_in_plot: bool,
) -> list[SweepMetricRow]:
    rows: list[SweepMetricRow] = []

    def _raw_sample_count_from_split_files(split_files: list[str] | None) -> float:
        if not split_files:
            return np.nan
        split_files = [str(name) for name in split_files if str(name).strip()]
        if not split_files:
            return np.nan
        if any("_mc_" in name for name in split_files):
            mapped = {re.sub(r"_mc_\d+(?=\.csv$)", "", name) for name in split_files}
            return float(len(mapped))
        return float(len(split_files))

    for result in eval_results:
        model_split_files = result.get("model_split_files", [])
        model_n_samples = _raw_sample_count_from_split_files(model_split_files)

        for summary_row in result.get("summary_rows", []):
            kind = str(summary_row.get("kind", "")).lower()
            if kind == "baseline" and not include_baselines_in_plot:
                continue
            if kind not in {"model", "baseline"}:
                continue

            input_dim = float(summary_row.get("input_dim", np.nan))
            n_predictors = np.nan
            if row_count > 0 and np.isfinite(input_dim):
                n_predictors = input_dim / float(row_count)

            default_n_samples = float(summary_row.get("n_test_samples", np.nan))
            n_samples = model_n_samples if kind == "model" and np.isfinite(model_n_samples) else default_n_samples

            rows.append(
                SweepMetricRow(
                    dataset=dataset_name,
                    target=target_name,
                    row_label=row_label,
                    row_order=int(row_order),
                    row_count=int(row_count),
                    model=str(summary_row.get("label", "unknown")),
                    n_samples=n_samples,
                    n_predictors=float(n_predictors) if np.isfinite(n_predictors) else np.nan,
                    input_dim=input_dim,
                    target_dim=float(summary_row.get("target_dim", np.nan)),
                    mae=float(summary_row.get("mae", np.nan)),
                    rmse=float(summary_row.get("rmse", np.nan)),
                    r2=float(summary_row.get("r2", np.nan)),
                )
            )
    return rows


def _plot_target_curves(metrics_df: pd.DataFrame, output_dir: Path, include_baselines: bool) -> list[Path]:
    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for target, target_df in metrics_df.groupby("target", sort=True):
        plot_df = target_df.copy()
        if plot_df.empty:
            continue

        model_order = sorted(plot_df["model"].dropna().unique().tolist())
        if not model_order:
            continue

        order_df = (
            plot_df[["row_label", "row_order"]]
            .drop_duplicates(subset=["row_label"], keep="first")
            .sort_values(["row_order", "row_label"], kind="stable")
        )
        x_labels = order_df["row_label"].astype(str).tolist()
        x_positions = np.arange(len(x_labels), dtype=float)
        label_to_x = {label: pos for label, pos in zip(x_labels, x_positions)}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

        for label in model_order:
            series = (
                plot_df[plot_df["model"] == label]
                .sort_values(["row_order", "row_label"], kind="stable")
                .drop_duplicates(subset=["row_label", "model"], keep="last")
            )
            x = np.array([label_to_x[str(lbl)] for lbl in series["row_label"].astype(str).tolist()], dtype=float)
            y_rmse = series["rmse"].to_numpy(dtype=float)
            y_r2 = series["r2"].to_numpy(dtype=float)

            axes[0].plot(x, y_rmse, marker="o", linewidth=1.8, label=label)
            axes[1].plot(x, y_r2, marker="o", linewidth=1.8, label=label)

        axes[0].set_title("RMSE vs predictor rows")
        axes[0].set_xlabel("Row mode")
        axes[0].set_ylabel("RMSE")
        axes[0].grid(alpha=0.3)
        axes[0].set_xticks(x_positions)
        axes[0].set_xticklabels(x_labels, rotation=45, ha="right")

        axes[1].set_title("R² vs predictor rows")
        axes[1].set_xlabel("Row mode")
        axes[1].set_ylabel("R²")
        axes[1].grid(alpha=0.3)
        axes[1].set_xticks(x_positions)
        axes[1].set_xticklabels(x_labels, rotation=45, ha="right")

        axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.4)

        for ax in axes:
            ax.legend(loc="best", fontsize=8)

        fig.suptitle(f"MC predictor-row sweep: {target}", fontsize=12)
        safe_target = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(target)).strip("_")
        out_path = output_dir / f"rowsweep_{safe_target}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(out_path)

    return written


def _write_metrics_table(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "mc_predictor_rows_sweep_metrics.csv"

    clean_columns = [
        "target",
        "model",
        "row_label",
        "row_order",
        "row_count",
        "n_samples",
        "n_predictors",
        "input_dim",
        "target_dim",
        "mae",
        "rmse",
        "r2",
    ]
    df = metrics_df.copy()
    for col in clean_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[clean_columns].sort_values(["target", "model", "row_order", "row_label"], kind="stable")
    df.to_csv(out_path, index=False)
    return out_path


def _write_dataset_outputs(metrics_df: pd.DataFrame, plans: list[DatasetPlan], include_baselines_in_plot: bool) -> list[Path]:
    written_metric_files: list[Path] = []

    if metrics_df.empty:
        return written_metric_files

    for plan in plans:
        dataset_name = plan.dataset_dir.name
        dataset_df = metrics_df[metrics_df["dataset"] == dataset_name].copy()
        if dataset_df.empty:
            continue

        output_dir = plan.dataset_dir / "forecasts" / "rowsweeps"
        metrics_csv = _write_metrics_table(dataset_df, output_dir)
        _plot_target_curves(dataset_df, output_dir, include_baselines=include_baselines_in_plot)
        written_metric_files.append(metrics_csv)

    return written_metric_files


def run_rowsweep(
    plans: list[DatasetPlan],
    sweep_options: list[SweepOption],
    dataset_prefix: str,
    dry_run: bool,
    stop_on_error: bool,
    include_baselines_in_plot: bool,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
) -> tuple[pd.DataFrame, int, int, int]:
    metric_rows: list[SweepMetricRow] = []
    completed_runs = 0
    failed_runs = 0
    skipped_runs = 0

    for dataset_index, plan in enumerate(plans, start=1):
        dataset_metric_rows: list[SweepMetricRow] = []
        print("\n" + "=" * 100)
        print(f"DATASET {dataset_index}/{len(plans)}: {plan.dataset_dir.name}")
        print("=" * 100)
        for cfg in plan.train_configs:
            print(f"  - train config: {cfg.name}")

        target_name = _derive_target_name(plan.dataset_dir.name, dataset_prefix)
        tmp_cfg_dir = plan.dataset_dir / "forecasts" / "rowsweeps" / "configs"

        for option in sweep_options:
            print("\n" + "-" * 100)
            print(f"[ROWS] mode={option.row_label} (aggregation={option.input_aggregation}) for {plan.dataset_dir.name}")
            print("-" * 100)

            eval_configs: list[Path] = []
            try:
                for base_cfg in plan.train_configs:
                    variant_cfg_path = _prepare_variant_config(base_cfg, option=option, tmp_dir=tmp_cfg_dir)
                    if dry_run:
                        print(f"  [DRY-RUN] variant config: {variant_cfg_path}")
                        continue

                    print(f"  [TRAIN] {variant_cfg_path.name}")
                    eval_cfg = _train_single_config(
                        variant_cfg_path,
                        disable_training_plots=disable_training_plots,
                        disable_eval_plots=disable_eval_plots,
                        suppress_training_logs=suppress_training_logs,
                    )
                    eval_configs.append(eval_cfg)
                    print(f"  [TRAIN] Evaluation config generated: {eval_cfg}")

                if dry_run:
                    completed_runs += 1
                    continue

                expanded_eval_configs = _expand_config_inputs([str(path) for path in eval_configs])
                if not expanded_eval_configs:
                    skipped_runs += 1
                    print("  [WARN] No evaluation configs found after expansion; skipping this row-count run.")
                    continue

                print(f"  [EVAL] Running {len(expanded_eval_configs)} evaluation config(s).")
                eval_results = [
                    eval_module.evaluate_single_config(
                        str(path),
                        save_plots_override=not disable_eval_plots,
                    )
                    for path in expanded_eval_configs
                ]

                rows_for_option = _collect_rows_from_eval_results(
                    eval_results,
                    dataset_name=plan.dataset_dir.name,
                    target_name=target_name,
                    row_label=option.row_label,
                    row_order=option.row_order,
                    row_count=option.row_count,
                    include_baselines_in_plot=include_baselines_in_plot,
                )
                if not rows_for_option:
                    print(
                        f"  [WARN] No model/baseline summary rows collected for mode={option.row_label}. "
                        "This usually means evaluations produced no usable metrics for this mode."
                    )
                metric_rows.extend(rows_for_option)
                dataset_metric_rows.extend(rows_for_option)

                completed_runs += 1
            except Exception as exc:
                failed_runs += 1
                print(f"  [ERROR] Row-count run failed for {plan.dataset_dir.name}, mode={option.row_label}")
                print(f"  [ERROR] {exc}")
                if stop_on_error:
                    raise

        if not dry_run and dataset_metric_rows:
            dataset_df = pd.DataFrame([row.__dict__ for row in dataset_metric_rows])
            output_dir = plan.dataset_dir / "forecasts" / "rowsweeps"
            metrics_csv = _write_metrics_table(dataset_df, output_dir)
            written_plots = _plot_target_curves(dataset_df, output_dir, include_baselines=include_baselines_in_plot)
            print(f"[INFO] Wrote dataset row-sweep metrics: {metrics_csv}")
            if written_plots:
                print(f"[INFO] Wrote {len(written_plots)} dataset row-sweep plot(s) to {output_dir}")

    if metric_rows:
        metrics_df = pd.DataFrame([row.__dict__ for row in metric_rows])
    else:
        metrics_df = pd.DataFrame(
            columns=[
                "dataset",
                "target",
                "row_label",
                "row_order",
                "row_count",
                "model",
                "n_samples",
                "n_predictors",
                "input_dim",
                "target_dim",
                "mae",
                "rmse",
                "r2",
            ]
        )

    return metrics_df, completed_runs, failed_runs, skipped_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep predictor row-count windows for MC datasets and compare RMSE/R² across models."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/output/regression",
        help="Root directory containing dataset folders (default: data/output/regression)",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="MC",
        help="Only dataset folders whose names start with this prefix are included (default: MC)",
    )
    parser.add_argument(
        "--config-pattern",
        type=str,
        default="config_*.yml",
        help="Train config filename pattern within each dataset folder (default: config_*.yml)",
    )
    parser.add_argument(
        "--limit-datasets",
        type=int,
        default=0,
        help="Max datasets to process. Use 0 for all (default: 0).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=160,
        help="Maximum rows to include in the schedule (default: 160).",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Limit number of row-count values from schedule start for quick pilots (0 = all).",
    )
    parser.add_argument(
        "--include-regular",
        action="store_true",
        help="Include non-_res datasets (default behavior unless explicitly disabled with --res-only).",
    )
    parser.add_argument(
        "--include-res",
        action="store_true",
        help="Include _res datasets (default behavior unless explicitly disabled with --regular-only).",
    )
    parser.add_argument(
        "--regular-only",
        action="store_true",
        help="Only include datasets not ending in _res.",
    )
    parser.add_argument(
        "--res-only",
        action="store_true",
        help="Only include datasets ending in _res.",
    )
    parser.add_argument(
        "--include-baselines-in-plot",
        action="store_true",
        help="Include Naive/Linear/Seasonal curves in row-sweep plots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan without running training/evaluation.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on first failure.",
    )
    parser.add_argument(
        "--keep-training-plots",
        action="store_true",
        help="Keep per-model training plots during row sweeps (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-eval-plots",
        action="store_true",
        help="Keep per-config evaluation plots during row sweeps (disabled by default for speed).",
    )
    parser.add_argument(
        "--show-training-logs",
        action="store_true",
        help="Show verbose model training logs (epoch metrics, sample-loading details).",
    )
    return parser


def _resolve_dataset_inclusion(args: argparse.Namespace) -> tuple[bool, bool]:
    # Defaults to including both regular and _res.
    include_regular = True
    include_res = True

    if args.regular_only and args.res_only:
        raise ValueError("Cannot use both --regular-only and --res-only.")

    if args.regular_only:
        include_regular, include_res = True, False
    elif args.res_only:
        include_regular, include_res = False, True
    else:
        # Explicit include flags can be used to override defaults if needed.
        if args.include_regular or args.include_res:
            include_regular = bool(args.include_regular)
            include_res = bool(args.include_res)

    if not include_regular and not include_res:
        raise ValueError("At least one dataset group must be included.")

    return include_regular, include_res


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

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

    if not plans:
        print("No matching datasets/configs found.")
        return 1

    row_counts = _row_count_schedule(max_rows=args.max_rows)
    if args.limit_rows > 0:
        row_counts = row_counts[: args.limit_rows]
    sweep_options = _build_sweep_options(row_counts=row_counts, avg_label="AVG")

    if not row_counts:
        print("Row-count schedule is empty. Check --max-rows / --limit-rows.")
        return 1

    print("\nExecution plan")
    print("-" * 100)
    print(f"Data root                 : {data_root}")
    print(f"Dataset prefix            : {args.dataset_prefix}")
    print(f"Config pattern            : {args.config_pattern}")
    print(f"Include regular datasets  : {include_regular}")
    print(f"Include _res datasets     : {include_res}")
    print(f"Dataset limit             : {args.limit_datasets} (0 = all)")
    print(f"Row schedule              : {row_counts}")
    print("Default extra mode        : AVG (mean over full original input window)")
    print(f"Dry run                   : {args.dry_run}")
    print(f"Keep train plots          : {args.keep_training_plots}")
    print(f"Keep eval plots           : {args.keep_eval_plots}")
    print(f"Show train logs           : {args.show_training_logs}")
    print(f"Datasets found            : {len(plans)}")

    total_configs = 0
    for plan in plans:
        total_configs += len(plan.train_configs)
        print(f"  - {plan.dataset_dir.name}: {len(plan.train_configs)} config(s)")
    print(f"Total model configs       : {total_configs}")
    print(f"Total planned row-runs    : {len(plans) * len(sweep_options)}")

    metrics_df, completed, failed, skipped = run_rowsweep(
        plans=plans,
        sweep_options=sweep_options,
        dataset_prefix=args.dataset_prefix,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        include_baselines_in_plot=args.include_baselines_in_plot,
        disable_training_plots=not args.keep_training_plots,
        disable_eval_plots=not args.keep_eval_plots,
        suppress_training_logs=not args.show_training_logs,
    )

    if not args.dry_run and not metrics_df.empty:
        build_super_summary_from_metrics(data_root=data_root, metrics_df=metrics_df)
        print("\n[INFO] Wrote per-dataset row-sweep outputs incrementally during execution.")

    print("\nRun summary")
    print("-" * 100)
    print(f"Row-runs completed: {completed}")
    print(f"Row-runs skipped  : {skipped}")
    print(f"Row-runs failed   : {failed}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

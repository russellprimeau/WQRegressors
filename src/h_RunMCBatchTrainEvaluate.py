"""
Batch train + evaluate runner for MC datasets.

Default behavior runs a limited pilot on the first 3 datasets that start with "MC"
under data/output/regression so behavior can be verified quickly.

Examples:
python src/h_RunMCBatchTrainEvaluate.py
python src/h_RunMCBatchTrainEvaluate.py --limit 0
python src/h_RunMCBatchTrainEvaluate.py --dataset-prefix MC_ --data-root data/output/regression
python src/h_RunMCBatchTrainEvaluate.py --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import e_Train as train_module
import f_Evaluate as eval_module


SUPPORTED_CONFIG_SUFFIXES = {".yml", ".yaml", ".json"}


@dataclass
class DatasetPlan:
    dataset_dir: Path
    train_configs: list[Path]


def _annotate_heatmap_cells(ax, values: np.ndarray, invert_contrast: bool = False) -> None:
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
            if invert_contrast:
                text_color = "black" if norm > 0.55 else "white"
            else:
                text_color = "white" if norm > 0.55 else "black"
            ax.text(j, i, f"{val:.2e}", ha="center", va="center", color=text_color, fontsize=7)


def _derive_target_name(dataset_name: str, dataset_prefix: str) -> str:
    if dataset_name.startswith(dataset_prefix):
        return dataset_name[len(dataset_prefix):].lstrip("_")
    return dataset_name


def _super_summary_paths(data_root: Path) -> tuple[Path, Path, Path]:
    summary_dir = data_root / "summaries"
    return (
        summary_dir / "mc_super_summary_models.csv",
        summary_dir / "mc_super_summary_best_by_target.csv",
        summary_dir / "mc_super_summary_heatmap.png",
    )


def build_super_summary_from_combined(
    data_root: Path,
    dataset_prefix: str,
    dataset_dirs: list[Path] | None = None,
) -> tuple[Path, Path, Path] | None:
    if dataset_dirs is None:
        dataset_dirs = [
            path for path in sorted(data_root.iterdir()) if path.is_dir() and path.name.startswith(dataset_prefix)
        ]

    summary_rows: list[pd.DataFrame] = []
    for dataset_dir in dataset_dirs:
        combined_path = dataset_dir / "forecasts" / "evaluation_summary_combined.csv"
        if not combined_path.exists():
            continue

        try:
            df = pd.read_csv(combined_path)
        except Exception as exc:
            print(f"[WARN] Could not read {combined_path}: {exc}")
            continue

        if df.empty:
            continue

        model_df = df[df.get("kind", "").astype(str).str.lower() == "model"].copy()
        if model_df.empty:
            continue

        model_df["dataset"] = dataset_dir.name
        model_df["target"] = _derive_target_name(dataset_dir.name, dataset_prefix)
        model_df["combined_summary_path"] = str(combined_path)
        summary_rows.append(model_df)

    if not summary_rows:
        print("[INFO] No combined summary files found to build super-summary.")
        return None

    super_df = pd.concat(summary_rows, ignore_index=True)

    for metric in ["mae", "rmse", "r2"]:
        if metric in super_df.columns:
            super_df[metric] = pd.to_numeric(super_df[metric], errors="coerce")
        else:
            super_df[metric] = np.nan

    super_df["rank_rmse"] = super_df.groupby("target")["rmse"].rank(method="min", ascending=True)
    super_df["rank_r2"] = super_df.groupby("target")["r2"].rank(method="min", ascending=False)
    super_df["is_best_rmse"] = super_df["rank_rmse"] == 1
    super_df["is_best_r2"] = super_df["rank_r2"] == 1

    ordered_cols = [
        "dataset",
        "target",
        "label",
        "mae",
        "rmse",
        "r2",
        "rank_rmse",
        "rank_r2",
        "is_best_rmse",
        "is_best_r2",
        "n_test_samples",
        "input_dim",
        "target_dim",
        "combined_summary_path",
    ]
    for col in ordered_cols:
        if col not in super_df.columns:
            super_df[col] = np.nan
    super_df = super_df[ordered_cols].sort_values(["target", "rank_rmse", "label"], kind="stable")

    best_rows = []
    for target, group in super_df.groupby("target", sort=True):
        best_rmse = group.loc[group["rank_rmse"] == 1]
        best_r2 = group.loc[group["rank_r2"] == 1]

        for _, row in best_rmse.iterrows():
            best_rows.append(
                {
                    "target": target,
                    "selection": "best_rmse",
                    "model": row["label"],
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "r2": row["r2"],
                }
            )
        for _, row in best_r2.iterrows():
            best_rows.append(
                {
                    "target": target,
                    "selection": "best_r2",
                    "model": row["label"],
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "r2": row["r2"],
                }
            )

    best_df = pd.DataFrame(best_rows, columns=["target", "selection", "model", "mae", "rmse", "r2"])

    super_csv_path, best_csv_path, plot_path = _super_summary_paths(data_root)
    super_csv_path.parent.mkdir(parents=True, exist_ok=True)
    super_df.to_csv(super_csv_path, index=False)
    best_df.to_csv(best_csv_path, index=False)

    rmse_heat = super_df.pivot_table(index="target", columns="label", values="rmse", aggfunc="mean")
    r2_heat = super_df.pivot_table(index="target", columns="label", values="r2", aggfunc="mean")

    all_targets = sorted(set(rmse_heat.index).union(set(r2_heat.index)))
    all_models = sorted(set(rmse_heat.columns).union(set(r2_heat.columns)))
    rmse_heat = rmse_heat.reindex(index=all_targets, columns=all_models)
    r2_heat = r2_heat.reindex(index=all_targets, columns=all_models)

    fig_h = max(4.5, 0.45 * len(all_targets) + 1.6)
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
    axes[0].set_yticks(np.arange(len(all_targets)))
    axes[0].set_yticklabels(all_targets)
    _annotate_heatmap_cells(axes[0], rmse_arr)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(r2_ma, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[1].set_title("R² (higher is better)")
    axes[1].set_xticks(np.arange(len(all_models)))
    axes[1].set_xticklabels(all_models, rotation=45, ha="right")
    axes[1].set_yticks(np.arange(len(all_targets)))
    axes[1].set_yticklabels(all_targets)
    _annotate_heatmap_cells(axes[1], r2_arr, invert_contrast=True)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("MC Model Effectiveness Super-Summary", fontsize=12)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"[INFO] Wrote super-summary table: {super_csv_path}")
    print(f"[INFO] Wrote super-summary best-model table: {best_csv_path}")
    print(f"[INFO] Wrote super-summary heatmap: {plot_path}")

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


def discover_mc_dataset_plans(
    data_root: Path,
    dataset_prefix: str,
    config_pattern: str,
    limit: int,
) -> list[DatasetPlan]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    dataset_dirs = [
        path for path in sorted(data_root.iterdir()) if path.is_dir() and path.name.startswith(dataset_prefix)
    ]

    plans: list[DatasetPlan] = []
    for dataset_dir in dataset_dirs:
        raw_matches = sorted(dataset_dir.glob(config_pattern))
        train_configs = [path for path in raw_matches if path.suffix.lower() in SUPPORTED_CONFIG_SUFFIXES]
        if not train_configs:
            continue

        train_configs.sort(key=_model_sort_key)
        plans.append(DatasetPlan(dataset_dir=dataset_dir, train_configs=train_configs))

    if limit > 0:
        plans = plans[:limit]

    return plans


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
    print(f"  [TRAIN] Using device: {device}")

    if suppress_training_logs:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train_samples, test_samples, _ = train_module.load_and_split_data(config)
    else:
        train_samples, test_samples, _ = train_module.load_and_split_data(config)
        print(f"  [TRAIN] Samples loaded: train={len(train_samples)} test={len(test_samples)}")

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
    return Path(data_cfg["data_dir"], "forecasts", forecast_name, f"config_evaluate_{forecast_file_name}.yml").resolve()


def run_plan(
    plans: list[DatasetPlan],
    dry_run: bool,
    stop_on_error: bool,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
) -> tuple[int, int]:
    completed_datasets = 0
    failed_datasets = 0

    for dataset_index, plan in enumerate(plans, start=1):
        print("\n" + "=" * 100)
        print(f"DATASET {dataset_index}/{len(plans)}: {plan.dataset_dir.name}")
        print("=" * 100)

        for cfg in plan.train_configs:
            print(f"  - train config: {cfg}")

        if dry_run:
            completed_datasets += 1
            continue

        eval_configs: list[Path] = []
        try:
            for cfg in plan.train_configs:
                print(f"\n[TRAIN] {cfg.name}")
                eval_config_path = _train_single_config(
                    cfg,
                    disable_training_plots=disable_training_plots,
                    disable_eval_plots=disable_eval_plots,
                    suppress_training_logs=suppress_training_logs,
                )
                eval_configs.append(eval_config_path)
                print(f"[TRAIN] Evaluation config generated: {eval_config_path}")

            if not eval_configs:
                print("[WARN] No evaluation configs generated; skipping evaluation.")
                completed_datasets += 1
                continue

            eval_args = [str(path) for path in eval_configs]
            expanded_eval_configs = _expand_config_inputs(eval_args)
            print(f"\n[EVAL] Running {len(expanded_eval_configs)} evaluation config(s).")
            eval_results = [
                eval_module.evaluate_single_config(
                    str(config_path),
                    save_plots_override=not disable_eval_plots,
                )
                for config_path in expanded_eval_configs
            ]
            if len(eval_results) > 1:
                eval_module.write_combined_outputs(eval_results, save_plots=not disable_eval_plots)

            completed_datasets += 1
        except Exception as exc:
            failed_datasets += 1
            print(f"[ERROR] Dataset failed: {plan.dataset_dir.name}")
            print(f"[ERROR] {exc}")
            if stop_on_error:
                raise

    return completed_datasets, failed_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train + evaluate all models for MC datasets using config-driven train/evaluate logic."
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
        "--limit",
        type=int,
        default=3,
        help="Max datasets to process. Use 0 for no limit (default: 3 for pilot).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running training/evaluation.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on first dataset failure.",
    )
    parser.add_argument(
        "--super-summary-only",
        action="store_true",
        help="Skip training/evaluation and only build the super-summary from existing combined outputs.",
    )
    parser.add_argument(
        "--keep-training-plots",
        action="store_true",
        help="Keep per-model training plots during batch runs (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-eval-plots",
        action="store_true",
        help="Keep per-config evaluation plots during batch runs (disabled by default for speed).",
    )
    parser.add_argument(
        "--show-training-logs",
        action="store_true",
        help="Show verbose model training logs (epoch metrics, sample-loading details).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()

    plans = discover_mc_dataset_plans(
        data_root=data_root,
        dataset_prefix=args.dataset_prefix,
        config_pattern=args.config_pattern,
        limit=args.limit,
    )

    if not plans:
        print("No matching datasets/configs found.")
        return 1

    print("\nExecution plan")
    print("-" * 100)
    print(f"Data root       : {data_root}")
    print(f"Dataset prefix  : {args.dataset_prefix}")
    print(f"Config pattern  : {args.config_pattern}")
    print(f"Dataset limit   : {args.limit} (0 = all)")
    print(f"Dry run         : {args.dry_run}")
    print(f"Keep train plots: {args.keep_training_plots}")
    print(f"Keep eval plots : {args.keep_eval_plots}")
    print(f"Show train logs : {args.show_training_logs}")
    print(f"Datasets found  : {len(plans)}")

    total_configs = 0
    for plan in plans:
        total_configs += len(plan.train_configs)
        print(f"  - {plan.dataset_dir.name}: {len(plan.train_configs)} config(s)")
    print(f"Total configs   : {total_configs}")

    if args.super_summary_only:
        summary_outputs = build_super_summary_from_combined(
            data_root=data_root,
            dataset_prefix=args.dataset_prefix,
            dataset_dirs=[plan.dataset_dir for plan in plans],
        )
        return 0 if summary_outputs is not None else 1

    completed, failed = run_plan(
        plans,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        disable_training_plots=not args.keep_training_plots,
        disable_eval_plots=not args.keep_eval_plots,
        suppress_training_logs=not args.show_training_logs,
    )

    if not args.dry_run:
        build_super_summary_from_combined(
            data_root=data_root,
            dataset_prefix=args.dataset_prefix,
            dataset_dirs=[plan.dataset_dir for plan in plans],
        )

    print("\nRun summary")
    print("-" * 100)
    print(f"Datasets completed: {completed}")
    print(f"Datasets failed   : {failed}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

"""
Train + evaluate default regression models for each dataset under data/output/regression,
then generate clustered bar charts for R2 and nRMSE.

Default models:
- GP Regressor        (config_gp_01.yml)
- Transformer         (config_transformer_01.yml)
- XGB Regressor       (config_xgb_01.yml)

Default baselines (from evaluation_summary.csv in forecasts/*):
- Naive
- Seasonal
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import e_Train as train_module
import f_Evaluate as eval_module


SUPPORTED_CONFIGS = [
    "config_gp_01.yml",
    "config_transformer_01.yml",
    "config_xgb_01.yml",
]

MODEL_TYPE_TO_CANONICAL = {
    "gp_regressor": "GPRegressor",
    "transformer": "Transformer",
    "xgb_regressor": "XGBRegressor",
}

DEFAULT_BASELINES = ["Naive", "Seasonal"]


def _base_sample_id(file_name: str) -> str:
    return re.sub(r"_mc_\d+(?=\.csv$)", "", Path(str(file_name)).name)


def _independent_sample_count(samples) -> int:
    ids = set()
    for sample in samples:
        if not isinstance(sample, (tuple, list)) or len(sample) < 3:
            continue
        ids.add(_base_sample_id(str(sample[2])))
    return len(ids)


def _model_sort_key(model_type: str) -> int:
    order = {"gp_regressor": 0, "transformer": 1, "xgb_regressor": 2}
    return order.get(model_type, 99)


def _discover_dataset_dirs(data_root: Path) -> list[Path]:
    out = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name in {"summaries", "storage", "classification", "calibration"}:
            continue
        if (p / "samples").exists():
            out.append(p)
    return out


def _extract_test_target_std(test_samples) -> float:
    flat = []
    for sample in test_samples:
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            continue
        y = sample[1]
        arr = np.asarray(y, dtype=float).reshape(-1)
        if arr.size:
            flat.append(arr)
    if not flat:
        return float("nan")
    y_all = np.concatenate(flat)
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        return float("nan")
    return float(np.std(y_all))


def _extract_test_target_range(test_samples) -> float:
    flat = []
    for sample in test_samples:
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            continue
        y = sample[1]
        arr = np.asarray(y, dtype=float).reshape(-1)
        if arr.size:
            flat.append(arr)
    if not flat:
        return float("nan")
    y_all = np.concatenate(flat)
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        return float("nan")
    return float(np.max(y_all) - np.min(y_all))


def _compute_nrmse(rmse: float, std_target: float, range_target: float) -> float:
    if not np.isfinite(rmse):
        return float("nan")

    if np.isfinite(std_target) and std_target > 0:
        return float(rmse / std_target)
    if np.isfinite(range_target) and range_target > 0:
        return float(rmse / range_target)

    # Constant-target fallback to keep outputs finite and comparable.
    return float(rmse)


def _coerce_r2(r2: float, rmse: float) -> float:
    if np.isfinite(r2):
        return float(r2)
    if not np.isfinite(rmse):
        return float("nan")
    # Mirror sklearn force_finite behavior for undefined constant-target R2.
    return 1.0 if abs(rmse) <= 1e-12 else 0.0


def _train_with_default_config(config_path: Path, keep_training_plots: bool):
    config = train_module.load_config(str(config_path))
    model_type = config["model_type"]
    config = train_module.merge_with_defaults(config, model_type)
    if not keep_training_plots:
        config["save_training_plots"] = False

    device = torch.device(config["device"])
    matplotlib.use(config["matplotlib_backend"])
    print(f"    [TRAIN] {config_path.name} on {device}")

    train_samples, test_samples = train_module.load_and_split_data(config)

    if model_type == "transformer":
        train_module.train_transformer_model(config, train_samples, test_samples)
    elif model_type == "gp_regressor":
        train_module.train_gp_regressor_model(config, train_samples, test_samples)
    elif model_type == "xgb_regressor":
        train_module.train_xgb_regressor_model(config, train_samples, test_samples)
    else:
        raise ValueError(f"Unsupported default model_type: {model_type}")

    data_cfg = config["data"]
    forecast_name = data_cfg["forecast_name"]
    forecast_file_name = Path(str(forecast_name)).name
    eval_config = Path(
        data_cfg["data_dir"],
        "forecasts",
        forecast_name,
        f"config_evaluate_{forecast_file_name}.yml",
    ).resolve()

    return config, eval_config, test_samples


def _canonical_model_label(eval_row: dict, model_type: str) -> str:
    # Prefer model type canonical name so labels are consistent across charts.
    return MODEL_TYPE_TO_CANONICAL.get(model_type, str(eval_row.get("label", model_type)))


def _load_baselines_from_summary(
    summary_csv: Path,
    dataset_name: str,
    baseline_labels: list[str],
    std_target: float,
    range_target: float,
    n_test_samples_independent: int | float,
) -> list[dict]:
    if not summary_csv.exists():
        return []

    df = pd.read_csv(summary_csv)
    if "kind" in df.columns:
        df = df[df["kind"].astype(str).str.lower() == "baseline"]

    out = []
    for label in baseline_labels:
        row_df = df[df["label"].astype(str) == label]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        rmse = float(row["rmse"]) if pd.notnull(row.get("rmse")) else float("nan")
        r2 = float(row["r2"]) if pd.notnull(row.get("r2")) else float("nan")
        r2 = _coerce_r2(r2, rmse)
        nrmse = _compute_nrmse(rmse, std_target, range_target)
        n_independent = n_test_samples_independent
        if pd.notnull(row.get("n_test_samples_independent")):
            n_independent = int(row["n_test_samples_independent"])
        out.append(
            {
                "dataset": dataset_name,
                "model": label,
                "r2": r2,
                "rmse": rmse,
                "nrmse": nrmse,
                "n_test_samples_independent": n_independent,
                "kind": "baseline",
            }
        )
    return out


def _plot_clustered_bars(
    df: pd.DataFrame,
    dataset_order: list[str],
    model_order: list[str],
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
    baseline_labels: list[str] | None = None,
):
    pivot = df.pivot_table(index="dataset", columns="model", values=metric, aggfunc="first")
    pivot = pivot.reindex(index=dataset_order, columns=model_order)

    x = np.arange(len(dataset_order), dtype=float)
    n_models = len(model_order)
    width = 0.82 / max(1, n_models)

    fig_w = max(12.0, 0.8 * len(dataset_order) + 6.0)
    fig_h = 6.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    cmap = plt.get_cmap("tab10")
    baseline_set = {str(x) for x in (baseline_labels or [])}
    for i, model_name in enumerate(model_order):
        vals = pivot[model_name].to_numpy(dtype=float)
        xpos = x + (i - (n_models - 1) / 2.0) * width
        drawn = np.where(np.isfinite(vals), vals, 0.0)
        is_baseline = model_name in baseline_set
        bars = ax.bar(
            xpos,
            drawn,
            width=width,
            color=cmap(i % 10),
            label=model_name,
            alpha=0.9,
            hatch="xx" if is_baseline else None,
            edgecolor="black" if is_baseline else None,
            linewidth=0.8 if is_baseline else None,
        )
        for b, v in zip(bars, vals):
            if not np.isfinite(v):
                b.set_alpha(0.2)
                b.set_hatch("//")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_order, rotation=40, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=1, fontsize=9)
    if y_min is not None:
        ax.set_ylim(bottom=float(y_min))
    if y_max is not None:
        ax.set_ylim(top=float(y_max))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[INFO] Wrote chart: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run default model train/eval sweep and produce clustered metric charts.")
    parser.add_argument("--data-root", type=str, default="data/output/regression")
    parser.add_argument("--baseline-labels", type=str, nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--keep-training-plots", action="store_true")
    parser.add_argument("--keep-eval-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (repo_root / data_root).resolve()

    dataset_dirs = _discover_dataset_dirs(data_root)
    if not dataset_dirs:
        print(f"[ERROR] No dataset folders found under: {data_root}")
        return 1

    all_rows: list[dict] = []

    for ds in dataset_dirs:
        print(f"\n=== DATASET: {ds.name} ===")
        std_target = float("nan")
        range_target = float("nan")
        model_eval_summaries: dict[str, Path] = {}
        model_independent_counts: dict[str, int] = {}

        configs = [ds / name for name in SUPPORTED_CONFIGS if (ds / name).exists()]
        # Sort by expected default model order.
        configs.sort(
            key=lambda p: _model_sort_key(train_module.load_config(str(p))["model_type"])
        )

        if not configs:
            print("  [WARN] No default configs found (expected gp/transformer/xgb).")
            continue

        for cfg in configs:
            if args.dry_run:
                print(f"  [DRY] Would train/eval: {cfg}")
                continue

            try:
                config, eval_cfg, test_samples = _train_with_default_config(cfg, args.keep_training_plots)
                model_type = str(config["model_type"])
                if not np.isfinite(std_target):
                    std_target = _extract_test_target_std(test_samples)
                if not np.isfinite(range_target):
                    range_target = _extract_test_target_range(test_samples)
                independent_count = _independent_sample_count(test_samples)
                model_independent_counts[model_type] = independent_count

                row = eval_module.evaluate_single_config(
                    str(eval_cfg),
                    save_plots_override=args.keep_eval_plots,
                )
                model_label = _canonical_model_label(row, config["model_type"])
                rmse = float(row.get("rmse", float("nan")))
                r2 = float(row.get("r2", float("nan")))
                r2 = _coerce_r2(r2, rmse)
                nrmse = _compute_nrmse(rmse, std_target, range_target)

                all_rows.append(
                    {
                        "dataset": ds.name,
                        "model": model_label,
                        "r2": r2,
                        "rmse": rmse,
                        "nrmse": nrmse,
                        "n_test_samples_independent": independent_count,
                        "kind": "model",
                    }
                )

                forecast_name = config["data"]["forecast_name"]
                summary_csv = Path(config["data"]["data_dir"]) / "forecasts" / forecast_name / "evaluation_summary.csv"
                model_eval_summaries[model_type] = summary_csv

            except Exception as exc:
                print(f"  [ERROR] Failed {cfg.name}: {exc}")

        # Pull baseline rows from the first available model evaluation summary.
        if not args.dry_run:
            baseline_added = False
            baseline_summary_candidates: list[tuple[Path, int | float]] = []
            if "xgb_regressor" in model_eval_summaries:
                baseline_summary_candidates.append(
                    (
                        model_eval_summaries["xgb_regressor"],
                        float(model_independent_counts.get("xgb_regressor", float("nan"))),
                    )
                )
            for mt, summary_csv in model_eval_summaries.items():
                if mt == "xgb_regressor":
                    continue
                baseline_summary_candidates.append(
                    (summary_csv, float(model_independent_counts.get(mt, float("nan"))))
                )

            for summary_csv, n_independent in baseline_summary_candidates:
                baseline_rows = _load_baselines_from_summary(
                    summary_csv=summary_csv,
                    dataset_name=ds.name,
                    baseline_labels=args.baseline_labels,
                    std_target=std_target,
                    range_target=range_target,
                    n_test_samples_independent=n_independent,
                )
                if baseline_rows:
                    all_rows.extend(baseline_rows)
                    baseline_added = True
                    break
            if not baseline_added:
                print("  [WARN] No baseline rows found in evaluation summaries.")

    if args.dry_run:
        print("\n[DRY] Completed planning only.")
        return 0

    if not all_rows:
        print("[ERROR] No metrics were collected.")
        return 2

    metrics_df = pd.DataFrame(all_rows)

    # Dataset order rule: decreasing maximum R2 score.
    max_r2_by_dataset = (
        metrics_df.groupby("dataset", as_index=False)["r2"]
        .max()
        .rename(columns={"r2": "max_r2"})
        .sort_values("max_r2", ascending=False, kind="stable")
    )
    dataset_order = max_r2_by_dataset["dataset"].tolist()

    preferred_model_order = ["GPRegressor", "Transformer", "XGBRegressor"] + args.baseline_labels
    present_models = [m for m in preferred_model_order if m in set(metrics_df["model"])]
    extras = [m for m in sorted(set(metrics_df["model"])) if m not in present_models]
    model_order = present_models + extras

    out_summary = data_root / "model_baseline_metrics_summary.csv"
    metrics_df.to_csv(out_summary, index=False)
    print(f"[INFO] Wrote summary table: {out_summary}")

    out_r2 = data_root / "clustered_r2_by_dataset_model.png"
    _plot_clustered_bars(
        df=metrics_df,
        dataset_order=dataset_order,
        model_order=model_order,
        metric="r2",
        title="R2 by Model Type, Clustered by Dataset",
        ylabel="R2",
        out_path=out_r2,
        y_min=-1.0,
        y_max=1.0,
        baseline_labels=args.baseline_labels,
    )

    out_nrmse = data_root / "clustered_nrmse_by_dataset_model.png"
    _plot_clustered_bars(
        df=metrics_df,
        dataset_order=dataset_order,
        model_order=model_order,
        metric="nrmse",
        title="nRMSE by Model Type, Clustered by Dataset",
        ylabel="nRMSE (RMSE / std(target))",
        out_path=out_nrmse,
        baseline_labels=args.baseline_labels,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
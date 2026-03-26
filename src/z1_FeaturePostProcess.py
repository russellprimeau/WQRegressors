"""
Post-processor script for generating outputs of feature-selection sweep for MC datasets without re-running the entire sweep.

Key CLI groups (detailed):
- Inherited sweep-discovery arguments (from `h_RunMCFeatureSelectionSweep.build_parser`):
    `--data-root`, `--config-pattern`, dataset include/exclude selectors, and
    legacy sweep toggles used to discover dataset/config plans.
- Path/namespace selection:
    `--path PATH`: Optional alias for `--data-root`; path scanned for dataset folders.
    `--sweep-namespace NAME`: Forecast subdirectory namespace to post-process
      (for example `feature_sweeps` or `Shapley_sweeps`).
- Rolling CV control:
    `--run-rolling-cv`: Enable optional rolling-origin CV execution.
- Statistical evidence controls:
    `--dm-max-lag`, `--bootstrap-iterations`, `--bootstrap-seed`,
    `--bootstrap-mode`, `--bootstrap-block-len`, `--evidence-alpha`,
    `--evidence-min-raw-samples`, `--evidence-min-prob`,
    `--evidence-ref-raw-samples`, `--interval-alpha`, `--coverage-tolerance`.

Postprocess behavior highlights:
- Rebuilds saved search artifacts from CSVs where available.
- Regenerates `feature_sweep_final_metrics_summary.png` from
  `feature_sweep_final_metrics.csv` so full sweep re-run is not required.
- Does not retroactively rewrite historical split files.
- By default, scans all datasets matching `--dataset-prefix` under the selected
    data root (`--limit-datasets 0` behavior in this script).
- Use `--limit-datasets N` to cap discovery, or `--all-datasets` to clear
    prefix filtering and process every dataset folder.

Examples:
python src/z1_FeaturePostProcess.py --keep-search-plots
python src/z1_FeaturePostProcess.py --sweep-namespace feature_sweeps
python src/z1_FeaturePostProcess.py --sweep-namespace Shapley_sweeps
python src/z1_FeaturePostProcess.py --path data/output/regression_alt --sweep-namespace Shapley_sweeps
python src/z1_FeaturePostProcess.py --sweep-namespace feature_sweeps --run-rolling-cv
python src/z1_FeaturePostProcess.py --path data/output/regression --sweep-namespace feature_sweeps --bootstrap-mode moving_block --bootstrap-block-len 5
python src/z1_FeaturePostProcess.py --all-datasets
python src/z1_FeaturePostProcess.py --limit-datasets 1
"""
from __future__ import annotations
import contextlib
import argparse
import copy
import json
import shutil
import subprocess
import glob
import hashlib
import io
import math
import re
import sys
import time
import textwrap
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import e_Train as train_module
import f_Evaluate as eval_module
import os
import unicodedata
import seaborn as sns
import traceback
from dataclasses import dataclass
from pathlib import Path
from utils.training import load_samples, group_samples_by_segment, _filter_samples_by_nan_tolerance
from utils.names import clean_target_label
from utils.mlr import evaluate_mlr as _evaluate_mlr
from h_RunMCFeatureSelectionSweep import build_parser, discover_mc_dataset_plans, _derive_target_name, _select_surrogate_config, _parse_row_counts, _available_row_counts_for_postprocess, _regenerate_saved_outputs_for_row, _load_feature_stats_artifacts_with_source, _compile_multi_target_comparison, _resolve_dataset_inclusion, _run_rolling_origin_cv, _ensure_k01_baselines, _write_dataset_evaluation_summary, _forecast_sweeps_dir, _plot_final_metrics_comparison

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


SUPPORTED_CONFIG_SUFFIXES = {".yml", ".yaml", ".json"}
BASELINE_ORDER = ("naive", "seasonal", "linear")
BASELINE_PLOT_LABELS = {
    "naive": "Naive",
    "seasonal": "Seasonal",
    "linear": "Linear",
}
BASELINE_PLOT_COLORS = {
    "naive": "tab:gray",
    "seasonal": "tab:green",
    "linear": "tab:orange",
}
BASELINE_MODEL_IDS = {"naive", "seasonal", "linear"}
MIN_REQUIRED_VALID_INDEPENDENT = 5

ML_COMPARISON_MODEL_TYPES = ['XGB', 'Trans.', 'GP', 'MLR']
ML_COMPARISON_COLORS = {'XGB': 'tab:blue', 'Trans.': 'tab:purple', 'GP': 'tab:olive', 'MLR': 'tab:red'}


def _resolve_summaries_dir(data_root: Path, sweep_namespace: str) -> Path:
    # Keep summary outputs anchored to the selected data root.
    base_dir = (data_root / "summaries").resolve()
    namespace = str(sweep_namespace).strip() or "feature_sweeps"
    if namespace == "feature_sweeps":
        return base_dir
    return (base_dir / namespace).resolve()


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
    input_dim: float
    target_dim: float

def _safe_float(val) -> float:
    """Return float(val) if val is non-null, otherwise float('nan')."""
    try:
        return float(val) if pd.notnull(val) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _is_baseline_model_value(value: object) -> bool:
    return str(value).strip().lower() in BASELINE_MODEL_IDS


def _exclude_baseline_metric_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    out = df.copy()
    if 'model' not in out.columns:
        return out
    return out[~out['model'].apply(_is_baseline_model_value)].copy()


def _build_perf_entry(
    dataset_name: str,
    row: "pd.Series",
    rolling_cv_r2: float = float('nan'),
    rolling_cv_r2_median: float = float('nan'),
    rolling_cv_r2_last50: float = float('nan'),
    rolling_cv_r2_pooled: float = float('nan'),
    rolling_cv_rmse: float = float('nan'),
    rolling_cv_mae: float = float('nan'),
    rolling_cv_n_folds: float = float('nan'),
    extra_metrics: "dict | None" = None,
) -> dict:
    """Build a best-model-performance dict from a metrics row and rolling CV stats."""
    out = {
        'dataset': dataset_name,
        'model': str(row.get('model', '')),
        'subset_rank': _safe_float(row.get('subset_rank', float('nan'))),
        'row_count': _safe_float(row.get('row_count', float('nan'))),
        'feature_tag': str(row.get('feature_tag', '')),
        'nrmse': _safe_float(row.get('nrmse', float('nan'))),
        'rmse': _safe_float(row.get('rmse', float('nan'))),
        'r2': _safe_float(row.get('r2', float('nan'))),
        'pearson_r': _safe_float(row.get('pearson_r', float('nan'))),
        'std_target': _safe_float(row.get('std_target', float('nan'))),
        'n_test_independent_source': _safe_float(row.get('n_test_independent', float('nan'))),
        'n_test_valid_source': _safe_float(row.get('n_test_valid', float('nan'))),
        'rolling_cv_r2': rolling_cv_r2,
        'rolling_cv_r2_median': rolling_cv_r2_median,
        'rolling_cv_r2_last50': rolling_cv_r2_last50,
        'rolling_cv_r2_pooled': rolling_cv_r2_pooled,
        'rolling_cv_rmse': rolling_cv_rmse,
        'rolling_cv_mae': rolling_cv_mae,
        'rolling_cv_n_folds': rolling_cv_n_folds,
    }
    if extra_metrics:
        out.update(extra_metrics)
    return out


def _filter_valid_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    """Filter metrics DataFrame to rows with valid std_target and finite r2, adding nrmse."""
    out = df.copy()
    if 'std_target' in out.columns:
        out = out[(out['std_target'].notnull()) & (out['std_target'] > 0)]
    if 'r2' in out.columns:
        out = out[out['r2'].notnull()]
    if 'std_target' in out.columns:
        out['nrmse'] = out['rmse'] / out['std_target']
    else:
        out['nrmse'] = np.nan
    if 'r2' in out.columns:
        out = out[out['r2'].notnull() & np.isfinite(out['r2'])]
    return out


def _filter_min_valid_independent(df: "pd.DataFrame", min_required: int = MIN_REQUIRED_VALID_INDEPENDENT) -> "pd.DataFrame":
    """Keep rows with explicit valid independent test count meeting threshold."""
    out = df.copy()
    count_col = None
    if "n_test_valid" in out.columns:
        count_col = "n_test_valid"
    elif "n_test_independent" in out.columns:
        count_col = "n_test_independent"
    if count_col is None:
        return out
    vals = pd.to_numeric(out[count_col], errors="coerce")
    return out[np.isfinite(vals) & (vals >= int(min_required))].copy()


def _annotate_bars_within_ylim(ax, bars, fmt: str, fontsize: int = 8) -> None:
    """Annotate bars only when the bar-top y value falls within current y-axis limits."""
    ymin, ymax = ax.get_ylim()
    yspan = float(ymax - ymin) if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
    pad = 0.01 * yspan
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        if h < ymin or h > ymax:
            continue
        y_txt = h + pad
        va = 'bottom'
        if y_txt > (ymax - pad):
            y_txt = h - pad
            va = 'top'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_txt,
            f'{h:{fmt}}',
            ha='center',
            va=va,
            fontsize=fontsize,
            rotation=90,
        )


def _annotate_ml_bars(ax, bars, vals, ns, fmt: str, fontsize: int = 7) -> None:
    """Annotate bars with combined value and sample-count label, e.g. '1.23e-02, n=8'."""
    ymin, ymax = ax.get_ylim()
    yspan = float(ymax - ymin) if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
    pad = 0.01 * yspan
    for bar, val, n in zip(bars, vals, ns):
        if not np.isfinite(val):
            continue
        h = bar.get_height()
        if h < ymin or h > ymax:
            continue
        n_str = f", n={int(n)}" if np.isfinite(n) else ""
        label = f"{val:{fmt}}{n_str}"
        y_txt = h + pad
        va = 'bottom'
        if y_txt > (ymax - pad):
            y_txt = h - pad
            va = 'top'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_txt,
            label,
            ha='center',
            va=va,
            fontsize=fontsize,
            rotation=90,
        )


def _compute_rolling_cv_r2_stats(df_cv: "pd.DataFrame") -> tuple[float, float, float, float]:
    """Return CV R2 stats as (mean, median, last50_mean, pooled)."""
    try:
        fold_rows = df_cv[df_cv['fold'].astype(str) != 'mean'].copy()
    except Exception:
        return float('nan'), float('nan'), float('nan'), float('nan')
    if fold_rows.empty:
        return float('nan'), float('nan'), float('nan'), float('nan')

    fold_rows['_fold_num'] = pd.to_numeric(fold_rows['fold'], errors='coerce')
    fold_rows = fold_rows.sort_values('_fold_num')

    r2_vals = pd.to_numeric(fold_rows['r2'], errors='coerce').to_numpy(dtype=float)
    finite_r2 = r2_vals[np.isfinite(r2_vals)]
    r2_mean = float(np.mean(finite_r2)) if finite_r2.size else float('nan')
    r2_median = float(np.median(finite_r2)) if finite_r2.size else float('nan')

    n_last = max(1, int(np.ceil(len(fold_rows) * 0.5)))
    last_rows = fold_rows.tail(n_last)
    last_r2 = pd.to_numeric(last_rows['r2'], errors='coerce').to_numpy(dtype=float)
    finite_last_r2 = last_r2[np.isfinite(last_r2)]
    r2_last50 = float(np.mean(finite_last_r2)) if finite_last_r2.size else float('nan')

    r2_pooled = float('nan')
    if {'ss_res', 'ss_tot'}.issubset(set(fold_rows.columns)):
        ss_res = pd.to_numeric(fold_rows['ss_res'], errors='coerce').to_numpy(dtype=float)
        ss_tot = pd.to_numeric(fold_rows['ss_tot'], errors='coerce').to_numpy(dtype=float)
        ss_res_sum = float(np.nansum(ss_res))
        ss_tot_sum = float(np.nansum(ss_tot))
        if np.isfinite(ss_tot_sum) and ss_tot_sum > 0:
            r2_pooled = float(1.0 - ss_res_sum / ss_tot_sum)

    return r2_mean, r2_median, r2_last50, r2_pooled


def _draw_bar_group(ax, x, width: float, data, colors, methods, fmt: str,
                    center_offset: float = 1.0, annotate: bool = True,
                    fontsize: int = 8) -> list:
    """Draw a grouped bar chart and optionally annotate each bar with its value."""
    bar_groups = []
    for i, (vals, color, method) in enumerate(zip(data, colors, methods)):
        bars = ax.bar(x + (i - center_offset) * width, vals, width, label=method, color=color)
        bar_groups.append(bars)
    if annotate:
        for bars in bar_groups:
            _annotate_bars_within_ylim(ax, bars, fmt, fontsize=fontsize)
    return bar_groups


def _wrap_label(text: str, width: int = 34) -> str:
    parts = str(text).split("\n")
    wrapped = [textwrap.fill(p, width=width, break_long_words=False) if p else "" for p in parts]
    return "\n".join(wrapped)


def _style_stacked_axes(axes, y_fontsize: int = 8, tick_fontsize: int = 8, legend_fontsize: int = 7) -> None:
    arr = np.atleast_1d(axes)
    for ax in arr:
        ax.set_ylabel(_wrap_label(ax.get_ylabel(), width=34), fontsize=y_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(legend_fontsize)


def _finalize_stacked_figure(fig, axes, left: float = 0.30, right: float = 0.98, top: float = 0.97,
                             bottom: float = 0.12, hspace: float = 0.50) -> None:
    _style_stacked_axes(axes)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace)


def _resolve_summary_plot_dirs(summaries_dir: Path) -> tuple[Path, Path, Path]:
    combined_dir = (summaries_dir / "combined").resolve()
    individual_dir = (summaries_dir / "individual").resolve()
    evaluation_dir = (summaries_dir / "evaluation").resolve()
    for out_dir in (combined_dir, individual_dir, evaluation_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
    return combined_dir, individual_dir, evaluation_dir


def _append_mlr_to_final_metrics(
    plan: "DatasetPlan",
    final_metrics_csv: Path,
    best_row: "pd.Series",
    args: "argparse.Namespace",
) -> None:
    """Fit MLR on train data, evaluate on test data, append row to final_metrics CSV.

    Uses the same train/test split as the best ML model for the dataset.
    Feature selection (MI + L1 + VIF) is performed independently on training data.
    Also writes per-variant output artifacts (evaluation_summary.csv, predictions.csv,
    model_config.json, boxplot.png, predictions.png, metrics_summary.png) matching
    the standard output format of other data-driven models.
    """
    df = pd.read_csv(final_metrics_csv)
    # Remove any existing MLR row so it is always recomputed
    if "model" in df.columns:
        df = df[df["model"].astype(str).str.lower() != "mlr"]

    variant_dir, eval_cfg_path, match_status = _find_best_variant_eval_config(plan, best_row)
    if eval_cfg_path is None or not eval_cfg_path.exists():
        print(f"[WARN] Cannot compute MLR for {plan.dataset_dir.name}: no eval config ({match_status})")
        return

    cfg = eval_module.load_config(str(eval_cfg_path))
    config_dir = cfg["__config_dir"]
    data_cfg = cfg["data"]
    split_cfg = cfg.get("data_split", {"random_state": 42})

    data_cfg["data_dir"], data_cfg["sample_subdir"] = eval_module._resolve_data_paths(data_cfg, config_dir)

    model_config = eval_module.load_model_config(
        data_cfg["data_dir"],
        data_cfg["forecast_name"],
        cfg.get("model_name", ""),
        fallback_data=data_cfg,
    )
    output_columns = model_config["output_columns"]
    input_rows = slice(model_config["input_row_1"], model_config["input_row_2"])
    output_rows = model_config["output_rows"]
    input_aggregation = str(model_config.get("input_aggregation", data_cfg.get("input_aggregation", "none"))).lower()

    # MLR feature selection is fully independent of the ML feature sweep.
    # Load the FULL set of input columns from the surrogate (base) config,
    # not the reduced subset selected for the best ML variant.
    surrogate_cfg_path = _select_surrogate_config(plan.train_configs)
    surrogate_cfg = train_module.load_config(str(surrogate_cfg_path))
    input_columns = list(surrogate_cfg["data"]["input_columns"])

    model_type = cfg.get("model_type", "")
    load_fault_tolerant, enforced_nan_tolerance = _model_sample_policy(model_type, split_cfg)
    split_base_dir = Path(data_cfg.get("forecast_dir", eval_cfg_path.parent))

    load_kw = dict(
        data_dir=data_cfg["data_dir"],
        sample_subdir=data_cfg["sample_subdir"],
        forecast_name=data_cfg["forecast_name"],
        input_columns=input_columns,
        output_columns=output_columns,
        input_rows=input_rows,
        output_rows=output_rows,
        split_source_dir=split_base_dir,
        fault_tolerant=load_fault_tolerant,
        input_aggregation=input_aggregation,
    )

    train_samples = eval_module.load_split_samples(**load_kw, split_file="train_files.txt")
    test_samples = eval_module.load_split_samples(**load_kw, split_file="test_files.txt")

    if load_fault_tolerant and enforced_nan_tolerance is not None:
        train_samples = _filter_samples_by_nan_tolerance(train_samples, float(enforced_nan_tolerance))
        test_samples = _filter_samples_by_nan_tolerance(test_samples, float(enforced_nan_tolerance))

    if len(train_samples) < 3 or len(test_samples) < 1:
        print(f"[WARN] Insufficient samples for MLR ({plan.dataset_dir.name}): "
              f"train={len(train_samples)}, test={len(test_samples)}")
        return

    # Read split file lists for predictions table
    test_split_files = eval_module._read_split_files(split_base_dir, "test_files.txt")

    predictions, targets, meta = _evaluate_mlr(
        train_samples, test_samples, feature_names=input_columns, verbose=False,
    )

    # Compute metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    finite_mask = np.all(np.isfinite(predictions), axis=1) & np.all(np.isfinite(targets), axis=1)
    if finite_mask.sum() < 1:
        print(f"[WARN] MLR produced no finite predictions for {plan.dataset_dir.name}")
        return

    p_flat = predictions[finite_mask].flatten()
    t_flat = targets[finite_mask].flatten()
    mae_val = float(mean_absolute_error(t_flat, p_flat))
    rmse_val = float(np.sqrt(mean_squared_error(t_flat, p_flat)))
    r2_val = float(r2_score(t_flat, p_flat))
    pearson_val = float(np.corrcoef(t_flat, p_flat)[0, 1]) if len(t_flat) >= 2 else float("nan")

    std_target = _safe_float(best_row.get("std_target", float("nan")))
    nrmse_val = rmse_val / std_target if np.isfinite(std_target) and std_target > 0 else float("nan")

    n_selected = sum(m.get("n_selected", 0) for m in meta) / max(len(meta), 1)

    mlr_row = {
        "dataset": best_row.get("dataset", plan.dataset_dir.name),
        "target": best_row.get("target", ""),
        "subset_rank": best_row.get("subset_rank", 1),
        "feature_tag": "mlr",
        "row_count": best_row.get("row_count", ""),
        "n_features": n_selected,
        "model": "mlr",
        "mae": mae_val,
        "rmse": rmse_val,
        "r2": r2_val,
        "pearson_r": pearson_val,
        "std_target": std_target,
        "nrmse": nrmse_val,
        "n_test_independent": best_row.get("n_test_independent", ""),
        "n_test_valid": int(finite_mask.sum()),
        "n_test_evals": int(finite_mask.sum()),
        "n_samples": best_row.get("n_samples", ""),
        "input_dim": float(len(input_columns)),
        "target_dim": float(len(output_columns)),
    }

    df = pd.concat([df, pd.DataFrame([mlr_row])], ignore_index=True)
    df.to_csv(final_metrics_csv, index=False)
    print(f"[INFO] Appended MLR row for {plan.dataset_dir.name}: "
          f"R²={r2_val:.3f}, RMSE={rmse_val:.4f}, n_features={n_selected:.0f}")

    # --- Write per-variant output artifacts ---
    sweep_dir = _forecast_sweeps_dir(plan.dataset_dir)
    mlr_dir = sweep_dir / "mlr"
    mlr_dir.mkdir(parents=True, exist_ok=True)

    # Derive forecast_name relative to dataset_dir so visualizer outputs land in mlr_dir
    try:
        mlr_forecast_name = str(mlr_dir.relative_to(plan.dataset_dir / "forecasts"))
    except ValueError:
        mlr_forecast_name = mlr_dir.name

    # 1. model_config.json — selected features and MLR metadata per target
    mlr_model_config = {
        "model_type": "mlr",
        "input_columns": input_columns,
        "output_columns": output_columns,
        "input_row_1": model_config.get("input_row_1"),
        "input_row_2": model_config.get("input_row_2"),
        "output_rows": output_rows,
        "input_aggregation": input_aggregation,
        "n_train_samples": len(train_samples),
        "n_test_samples": len(test_samples),
        "feature_selection": {
            "method": "Spearman + MI + L1/Lasso + VIF",
            "spearman_p_threshold": 0.05,
            "spearman_rho_threshold": 0.25,
            "mi_quantile": 0.25,
            "vif_threshold": 10.0,
        },
        "per_target_meta": [],
    }
    for j, m in enumerate(meta):
        target_meta = {
            "target_index": j,
            "target_name": output_columns[j] if j < len(output_columns) else f"target_{j}",
            "selected_features": m.get("selected_features", []),
            "n_selected": m.get("n_selected", 0),
            "coefficients": m.get("coefficients", []),
            "intercept": m.get("intercept", None),
            "n_train_valid": m.get("n_train", 0),
            "spearman_kept_columns": m.get("spearman_kept_columns", []),
        }
        mlr_model_config["per_target_meta"].append(target_meta)

    model_config_path = mlr_dir / "model_config.json"
    with open(model_config_path, "w") as f:
        json.dump(mlr_model_config, f, indent=2, default=str)
    print(f"[INFO] Wrote MLR model config: {model_config_path}")

    # 2. mlr_equation.txt — human-readable regression equations per target
    eq_lines = []
    for tm in mlr_model_config["per_target_meta"]:
        tgt = tm["target_name"]
        feats = tm["selected_features"]
        coefs = tm["coefficients"]
        intercept = tm["intercept"]
        eq_lines.append(f"Target: {tgt}")
        eq_lines.append(f"  n_selected = {tm['n_selected']}")
        eq_lines.append(f"  n_train    = {tm['n_train_valid']}")
        if intercept is not None and feats and coefs:
            terms = [f"{intercept:.6g}"]
            for feat, c in zip(feats, coefs):
                sign = "+" if c >= 0 else "-"
                terms.append(f"{sign} {abs(c):.6g} * {feat}")
            eq_lines.append(f"  y = {terms[0]}")
            for term in terms[1:]:
                eq_lines.append(f"      {term}")
        else:
            eq_lines.append("  (no features selected)")
        eq_lines.append("")
    eq_path = mlr_dir / "mlr_equation.txt"
    with open(eq_path, "w", encoding="utf-8") as f:
        f.write("\n".join(eq_lines))
    print(f"[INFO] Wrote MLR equation: {eq_path}")

    # 4. Copy train_files.txt / test_files.txt from reference variant
    for split_name in ("train_files.txt", "test_files.txt"):
        src = split_base_dir / split_name
        dst = mlr_dir / split_name
        if src.exists():
            shutil.copy2(src, dst)

    # 5. Evaluate baselines on the same test samples
    model_label = "MLR"
    eval_cfg = eval_module.merge_eval_config(cfg)
    for key in ["historic_path"]:
        if eval_cfg.get(key):
            eval_cfg[key] = str(eval_module._resolve_path_from_config(eval_cfg[key], config_dir))

    summary_rows = []
    predictions_entries = []
    baseline_labels = []
    baseline_pairs = []
    baseline_split_files = []

    # MLR model entry
    summary_row = eval_module._compute_regression_summary(
        f"{model_label} (test)",
        predictions,
        targets,
        len(targets),
        metadata={"kind": "test", "gp_uncertainty_mode": "not_gp"},
        split_files=test_split_files,
    )
    summary_row["n_train_samples"] = len(train_samples)
    summary_row["n_test_samples"] = len(test_samples)
    summary_row["input_dim"] = len(input_columns)
    summary_row["target_dim"] = len(output_columns)
    summary_row["data_dir"] = str(data_cfg["data_dir"])
    summary_rows.append(summary_row)
    predictions_entries.append({
        "kind": "test",
        "label": model_label,
        "preds": predictions,
        "targets": targets,
        "split_files": test_split_files,
        "include_mc_stats": False,
    })

    # Baselines (naive, seasonal, linear) — same as evaluate_single_config
    try:
        historic = eval_cfg.get("historic_path")
        if historic:
            sample_subdir = data_cfg.get("sample_subdir", "samples")
            baseline_output_rows = eval_module._baseline_output_rows_start(
                data_cfg.get("output_rows", output_rows))
            secondary, baseline_window_hours = eval_module.load_secondary(
                output_columns,
                int(eval_cfg.get("window_hours", 340)),
            )
            baseline_deduped_split_files = eval_module._dedupe_split_files_by_base_sample(test_split_files)

            preds_naive, tgts_naive = eval_module.evaluate_naive(
                test_samples, historic, output_columns, data_cfg["data_dir"],
                output_rows=baseline_output_rows,
                gap_hours=int(eval_cfg.get("gap_hours", 5)),
                sample_subdir=sample_subdir,
            )
            preds_seasonal, tgts_seasonal = eval_module.evaluate_seasonal(
                test_samples, historic, output_columns, data_cfg["data_dir"],
                output_rows=baseline_output_rows,
                diurnal_window=int(eval_cfg.get("diurnal_window", 2)),
                secondary=secondary,
                sample_subdir=sample_subdir,
            )
            preds_linear, tgts_linear = eval_module.evaluate_linear(
                data_cfg["data_dir"], data_cfg["forecast_name"],
                test_samples, historic, output_columns,
                output_rows=baseline_output_rows,
                window_hours=int(baseline_window_hours),
                gap_hours=int(eval_cfg.get("gap_hours", 0)),
                debug_plot=False, examples=0,
                sample_subdir=sample_subdir,
            )
            baseline_pairs = [
                (preds_naive, tgts_naive),
                (preds_seasonal, tgts_seasonal),
                (preds_linear, tgts_linear),
            ]
            baseline_labels = ["Naive", "Seasonal", "Linear"]
            baseline_split_files = [baseline_deduped_split_files] * 3

            for (bl_preds, bl_tgts), bl_label in zip(baseline_pairs, baseline_labels):
                bl_row = eval_module._compute_regression_summary(
                    bl_label, bl_preds, bl_tgts, len(test_samples),
                    metadata={"kind": "baseline", "gp_uncertainty_mode": "not_gp"},
                    split_files=test_split_files,
                )
                summary_rows.append(bl_row)
                predictions_entries.append({
                    "kind": "test",
                    "label": bl_label,
                    "preds": bl_preds,
                    "targets": bl_tgts,
                    "split_files": test_split_files,
                    "include_mc_stats": False,
                })
    except Exception as exc:
        print(f"[WARN] MLR baseline evaluation failed for {plan.dataset_dir.name}: {exc}")

    # 6. evaluation_summary.csv
    eval_module._write_summary_csv(summary_rows, mlr_dir / "evaluation_summary.csv")

    # 7. predictions.csv
    pred_rows, pred_columns = eval_module._build_predictions_table(
        predictions_entries,
        gp_uncertainty_mode="not_gp",
        include_mc_output_columns=False,
    )
    eval_module._write_predictions_csv(pred_rows, mlr_dir / "predictions.csv", pred_columns)

    # 8. Plots (predictions.png, metrics_summary.png, boxplot.png)
    matplotlib.use("Agg")
    plot_pairs = [(predictions, targets)] + baseline_pairs
    plot_labels = [model_label] + baseline_labels
    plot_split_files = [test_split_files] + baseline_split_files
    collapse_flags = [False] + [True] * len(baseline_pairs)

    try:
        eval_module.visualizer(
            *plot_pairs,
            labels=plot_labels,
            directory=str(plan.dataset_dir),
            forecast_name=mlr_forecast_name,
            num_samples=200,
            split_files_by_pair=plot_split_files,
            collapse_error_points_by_pair=collapse_flags,
        )
    except Exception as exc:
        print(f"[WARN] MLR predictions/metrics plots failed for {plan.dataset_dir.name}: {exc}")

    try:
        boxplot_rows = eval_module._build_boxplot_error_rows_from_predictions(
            pred_rows,
            model_label=model_label,
            baseline_labels=baseline_labels,
        )
        eval_module.boxplot_from_error_rows(
            boxplot_rows,
            directory=str(plan.dataset_dir),
            forecast_name=mlr_forecast_name,
        )
    except Exception as exc:
        print(f"[WARN] MLR boxplot failed for {plan.dataset_dir.name}: {exc}")


def _normalize_ml_model_display(val: str) -> str:
    """Map raw model string to ML_COMPARISON_MODEL_TYPES display key."""
    key = str(val).strip().lower()
    if 'xgb' in key:
        return 'XGB'
    if 'transformer' in key:
        return 'Trans.'
    if 'gp' in key:
        return 'GP'
    if 'mlr' in key:
        return 'MLR'
    return None


def _plot_ml_model_comparison(
    plans: "list",
    data_root: "Path",
    summaries_dir: "Path",
    args: "argparse.Namespace",
) -> None:
    """Generate 4 clustered bar charts comparing best-per-model-type scores across targets.

    Output files in summaries_dir/ML_comparison/:
      ml_comparison_rmse.png
      ml_comparison_nrmse.png
      ml_comparison_r2.png
      ml_comparison_skill_vs_best_baseline.png
    """
    ml_comp_dir = (summaries_dir / "ML_comparison").resolve()
    ml_comp_dir.mkdir(parents=True, exist_ok=True)

    dataset_prefix = str(getattr(args, "dataset_prefix", "MC"))
    records = []  # list of dicts, one per (dataset, model_display)

    for plan in plans:
        dataset_name = plan.dataset_dir.name
        final_metrics_csv = _forecast_sweeps_dir(plan.dataset_dir) / "feature_sweep_final_metrics.csv"
        if not final_metrics_csv.exists():
            continue
        try:
            df = pd.read_csv(final_metrics_csv)
        except Exception:
            continue
        if df.empty:
            continue

        # Apply the same validity filters used for best-model selection
        valid_df = _exclude_baseline_metric_rows(
            _filter_min_valid_independent(
                _filter_valid_rows(df),
                min_required=MIN_REQUIRED_VALID_INDEPENDENT,
            )
        )
        if valid_df.empty:
            continue

        # Determine n_samples column
        if 'n_test_valid' in valid_df.columns:
            n_col = 'n_test_valid'
        elif 'n_test_independent' in valid_df.columns:
            n_col = 'n_test_independent'
        else:
            n_col = None

        # Read baseline RMSE from evaluation_summary.csv
        eval_csv = data_root / dataset_name / 'evaluation_summary.csv'
        baseline_rmse = {}
        if eval_csv.exists():
            try:
                df_eval = pd.read_csv(eval_csv)
                for kind in BASELINE_ORDER:
                    match = df_eval[df_eval['label'].str.lower().str.contains(kind)]
                    if not match.empty:
                        baseline_rmse[kind] = _safe_float(match.iloc[0].get('rmse', float('nan')))
                    else:
                        baseline_rmse[kind] = float('nan')
            except Exception:
                for kind in BASELINE_ORDER:
                    baseline_rmse[kind] = float('nan')
        else:
            for kind in BASELINE_ORDER:
                baseline_rmse[kind] = float('nan')

        best_baseline_rmse = min(
            (v for v in baseline_rmse.values() if np.isfinite(v) and v > 0),
            default=float('nan'),
        )

        # Add normalized display type column
        valid_df = valid_df.copy()
        valid_df['_model_display'] = valid_df['model'].apply(_normalize_ml_model_display)

        for model_display in ML_COMPARISON_MODEL_TYPES:
            subset = valid_df[valid_df['_model_display'] == model_display]
            if subset.empty:
                continue
            best_idx = subset['r2'].idxmax()
            row = subset.loc[best_idx]
            rmse_val = _safe_float(row.get('rmse', float('nan')))
            nrmse_val = _safe_float(row.get('nrmse', float('nan')))
            r2_val = _safe_float(row.get('r2', float('nan')))
            n_val = _safe_float(row.get(n_col, float('nan'))) if n_col else float('nan')

            # Skill vs best baseline: 1 - model_rmse / best_baseline_rmse
            if np.isfinite(rmse_val) and np.isfinite(best_baseline_rmse) and best_baseline_rmse > 0:
                skill_val = 1.0 - rmse_val / best_baseline_rmse
            else:
                skill_val = float('nan')

            records.append({
                'dataset': dataset_name,
                'target_label': clean_target_label(dataset_name, dataset_prefix),
                'model_display': model_display,
                'rmse': rmse_val,
                'nrmse': nrmse_val,
                'r2': r2_val,
                'skill_vs_best': skill_val,
                'n_samples': n_val,
            })

    if not records:
        print("[INFO] ML comparison: no valid per-model-type data found; skipping.")
        return

    comp_df = pd.DataFrame(records)

    # Determine target order: sort by best R2 per dataset (descending)
    best_r2_per_dataset = (
        comp_df.groupby('dataset')['r2'].max().sort_values(ascending=False)
    )
    ordered_datasets = list(best_r2_per_dataset.index)
    # Deduplicated target labels in the same order
    seen: set = set()
    ordered_targets: list[tuple[str, str]] = []  # (dataset, target_label)
    for ds in ordered_datasets:
        lbl = comp_df.loc[comp_df['dataset'] == ds, 'target_label'].iloc[0]
        if ds not in seen:
            seen.add(ds)
            ordered_targets.append((ds, lbl))

    n_targets = len(ordered_targets)
    x = np.arange(n_targets)
    width = 0.22
    n_models = len(ML_COMPARISON_MODEL_TYPES)
    offsets = np.array([(i - (n_models - 1) / 2) for i in range(n_models)])
    _FS = 12  # unified font size for all text elements

    metric_specs = [
        ('rmse',          'RMSE',                    '.2e', False),
        ('nrmse',         'nRMSE',                   '.2e', False),
        ('r2',            'R²',                      '.2f', False),
        ('skill_vs_best', 'Skill vs. Best Baseline', '.2f', True),
    ]
    file_names = {
        'rmse':          'ml_comparison_rmse.png',
        'nrmse':         'ml_comparison_nrmse.png',
        'r2':            'ml_comparison_r2.png',
        'skill_vs_best': 'ml_comparison_skill_vs_best_baseline.png',
    }

    for metric_key, ylabel, fmt, add_hline in metric_specs:
        fig, ax = plt.subplots(figsize=(max(10, n_targets * 1.4), 6))
        # Build proxy Patch handles upfront so labels are always correct.
        legend_handles = [
            matplotlib.patches.Patch(facecolor=ML_COMPARISON_COLORS[m], label=m)
            for m in ML_COMPARISON_MODEL_TYPES
        ]

        for mi, model_display in enumerate(ML_COMPARISON_MODEL_TYPES):
            color = ML_COMPARISON_COLORS[model_display]
            vals = []
            ns = []
            bar_x = []
            for ti, (ds, _lbl) in enumerate(ordered_targets):
                row_match = comp_df[
                    (comp_df['dataset'] == ds) & (comp_df['model_display'] == model_display)
                ]
                if row_match.empty or not np.isfinite(_safe_float(row_match.iloc[0][metric_key])):
                    continue
                bar_x.append(x[ti] + offsets[mi] * width)
                vals.append(_safe_float(row_match.iloc[0][metric_key]))
                ns.append(_safe_float(row_match.iloc[0]['n_samples']))

            if not bar_x:
                continue

            bars = ax.bar(bar_x, vals, width, color=color)
            _annotate_ml_bars(ax, bars, vals, ns, fmt)

        if add_hline:
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

        # Constrain y-axis lower limit: never below -1; if all bars positive, floor at 0.
        ymin_cur, ymax_cur = ax.get_ylim()
        all_metric_vals = comp_df[metric_key].dropna().to_numpy(dtype=float)
        all_metric_vals = all_metric_vals[np.isfinite(all_metric_vals)]
        min_val = float(np.min(all_metric_vals)) if all_metric_vals.size else 0.0
        if min_val >= 0.0:
            ax.set_ylim(bottom=0.0)
        else:
            ax.set_ylim(bottom=max(ymin_cur, -1.0))

        # Tight horizontal bounds: ~0.07 units of padding beyond the outermost bar edges.
        cluster_half = 1.5 * width
        if n_targets > 0:
            ax.set_xlim(x[0] - cluster_half - 0.07, x[-1] + cluster_half + 0.07)

        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.tick_params(axis='y', labelsize=_FS)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [lbl for _ds, lbl in ordered_targets],
            rotation=45,
            ha='right',
            fontsize=_FS,
        )
        ax.grid(axis='y', alpha=0.3)

        # Legend in a single row above the plot area
        ax.legend(
            handles=legend_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(ML_COMPARISON_MODEL_TYPES),
            frameon=False,
            fontsize=_FS,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out_path = ml_comp_dir / file_names[metric_key]
        fig.savefig(out_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Wrote ML comparison figure: {out_path}")


def _save_subplot_panels(
    fig,
    axes,
    out_dir: Path,
    base_name: str,
    dpi: int = 300,
) -> list[Path]:
    """Save each axis from a combined figure as its own image without axis titles."""
    saved: list[Path] = []
    ax_list = list(np.atleast_1d(axes).ravel())
    if not ax_list:
        return saved

    # Shared-x figures usually render category tick labels on the bottom subplot only.
    # Copy that tick context to all axes so each exported panel is self-contained.
    ref_ticks = None
    ref_labels = None
    ref_style = None
    ref_xlabel = ""
    for ref_ax in reversed(ax_list):
        labels = [t.get_text() for t in ref_ax.get_xticklabels()]
        if any(str(lbl).strip() for lbl in labels):
            ref_ticks = ref_ax.get_xticks()
            ref_labels = labels
            for t in ref_ax.get_xticklabels():
                if str(t.get_text()).strip():
                    ref_style = {
                        "rotation": t.get_rotation(),
                        "ha": t.get_ha(),
                        "va": t.get_va(),
                        "fontsize": t.get_fontsize(),
                    }
                    break
            ref_xlabel = str(ref_ax.get_xlabel() or "")
            break

    for ax in ax_list:
        ax.set_title("")
        ax.tick_params(axis='x', labelbottom=True)
        if ref_ticks is not None:
            ax.set_xticks(ref_ticks)
        if ref_labels is not None and ref_ticks is not None and len(ref_labels) == len(ref_ticks):
            ax.set_xticklabels(ref_labels)
        if ref_style:
            for tick in ax.get_xticklabels():
                tick.set_rotation(ref_style["rotation"])
                tick.set_ha(ref_style["ha"])
                tick.set_va(ref_style["va"])
                tick.set_fontsize(ref_style["fontsize"])
        if ref_xlabel and not str(ax.get_xlabel() or "").strip():
            ax.set_xlabel(ref_xlabel)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for idx, ax in enumerate(ax_list, start=1):
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            bbox = ax.get_window_extent(renderer)
        bbox_inches = bbox.expanded(1.03, 1.08).transformed(fig.dpi_scale_trans.inverted())
        panel_path = out_dir / f"{base_name}__panel_{idx:02d}.png"
        fig.savefig(panel_path, dpi=dpi, bbox_inches=bbox_inches)
        saved.append(panel_path)
    return saved


def _save_individual_panel(
    out_dir: Path,
    base_name: str,
    panel_index: int,
    labels,
    draw_fn,
    figsize: tuple[float, float],
    dpi: int = 300,
    left: float = 0.34,
    right: float = 0.98,
    top: float = 0.96,
    bottom: float = 0.30,
) -> Path:
    """Render and save a standalone panel image using a dedicated figure canvas."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)

    label_list = [str(v) for v in list(labels)]
    x_ticks = np.arange(len(label_list), dtype=float)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(label_list, rotation=45, ha='right')
    _style_stacked_axes(np.array([ax]), y_fontsize=9, tick_fontsize=8, legend_fontsize=8)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    panel_path = out_dir / f"{base_name}__panel_{int(panel_index):02d}.png"
    fig.savefig(panel_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return panel_path


def _save_individual_panels_from_builders(
    out_dir: Path,
    base_name: str,
    labels,
    builders,
    figsize: tuple[float, float],
    dpi: int = 300,
    left: float = 0.34,
    right: float = 0.98,
    top: float = 0.96,
    bottom: float = 0.30,
) -> list[Path]:
    """Render a sequence of standalone panels, one figure per builder callback."""
    saved: list[Path] = []
    for idx, builder in enumerate(list(builders), start=1):
        panel_path = _save_individual_panel(
            out_dir=out_dir,
            base_name=base_name,
            panel_index=idx,
            labels=labels,
            draw_fn=builder,
            figsize=figsize,
            dpi=dpi,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        saved.append(panel_path)
    return saved


def _normalize_model_key(value: str) -> str:
    return str(value).strip().lower().replace("_", "").replace(" ", "")


def _base_sample_id(name: str) -> str:
    return re.sub(r"_mc_\d+(?=\.csv$)", "", Path(str(name)).name)


def _model_sample_policy(model_type: str, split_cfg: dict) -> tuple[bool, float | None]:
    """Return enforced (fault_tolerant, nan_tolerance) for postprocess sample loading."""
    model_key = str(model_type).strip().lower()
    if model_key == "xgb_regressor":
        tol = split_cfg.get("nan_tolerance", train_module.DEFAULT_DATA_SPLIT_CONFIG.get("nan_tolerance", 0.8))
        try:
            tol_val = float(tol)
        except (TypeError, ValueError):
            tol_val = float(train_module.DEFAULT_DATA_SPLIT_CONFIG.get("nan_tolerance", 0.8))
        tol_val = float(min(1.0, max(0.0, tol_val)))
        return True, tol_val
    if model_key in {"gp_regressor", "transformer"}:
        return False, None
    # Conservative fallback for any unexpected model types.
    return False, None


def _sample_names_from_loaded_samples(samples) -> list[str]:
    names: list[str] = []
    for sample in samples:
        if isinstance(sample, (tuple, list)) and len(sample) >= 3:
            names.append(Path(str(sample[2])).name)
    return names


def _find_best_variant_eval_config(plan: DatasetPlan, row: "pd.Series") -> "tuple[Path | None, Path | None, str]":
    try:
        row_count = int(row.get("row_count"))
        feature_tag = str(row.get("feature_tag", ""))
    except Exception:
        return None, None, "invalid_best_row"

    model_key = _normalize_model_key(str(row.get("model", "")))
    output_dir = _forecast_sweeps_dir(plan.dataset_dir)
    variant_dirs = [
        p for p in sorted(output_dir.glob(f"*_r{row_count:03d}_{feature_tag}_k*"))
        if p.is_dir()
    ]
    if not variant_dirs:
        return None, None, "missing_variant_dir"

    best_fallback = None
    for variant_dir in variant_dirs:
        eval_cfg = variant_dir / f"config_evaluate_{variant_dir.name}.yml"
        if not eval_cfg.exists():
            continue
        if best_fallback is None:
            best_fallback = (variant_dir, eval_cfg)
        try:
            cfg = train_module.load_config(str(eval_cfg))
        except Exception:
            continue
        cfg_keys = [
            _normalize_model_key(str(cfg.get("model_type", ""))),
            _normalize_model_key(str(cfg.get("model_name", ""))),
            _normalize_model_key(str(cfg.get("data", {}).get("forecast_name", ""))),
        ]
        if model_key and any(model_key == k or model_key in k or k in model_key for k in cfg_keys if k):
            return variant_dir, eval_cfg, "exact_match"

    if best_fallback is not None:
        variant_dir, eval_cfg = best_fallback
        return variant_dir, eval_cfg, "fallback_variant_mismatch"
    return None, None, "missing_eval_config"


def _safe_as_2d(arr) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    return out


def _aligned_pred_target(preds, targets, row_limit=None) -> "tuple[np.ndarray, np.ndarray]":
    pred_arr, tgt_arr, _, _ = eval_module._aligned_arrays(preds, targets, row_limit=row_limit)
    return _safe_as_2d(pred_arr), _safe_as_2d(tgt_arr)


def _compute_point_metrics(preds, targets) -> dict:
    pred_arr, tgt_arr = _aligned_pred_target(preds, targets)
    if pred_arr.size == 0 or tgt_arr.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n_finite": 0}

    pf = pred_arr.reshape(-1)
    tf = tgt_arr.reshape(-1)
    mask = np.isfinite(pf) & np.isfinite(tf)
    n = int(np.sum(mask))
    if n == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n_finite": 0}

    err = pf[mask] - tf[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    if n > 1:
        ss_res = float(np.sum(np.square(err)))
        tvals = tf[mask]
        ss_tot = float(np.sum(np.square(tvals - np.mean(tvals))))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        r2 = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n_finite": n}


def _compute_per_sample_losses(preds, targets) -> "tuple[np.ndarray, np.ndarray]":
    pred_arr, tgt_arr = _aligned_pred_target(preds, targets)
    n_rows = min(len(pred_arr), len(tgt_arr))
    if n_rows <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    pred_arr = np.asarray(pred_arr[:n_rows], dtype=float)
    tgt_arr = np.asarray(tgt_arr[:n_rows], dtype=float)

    mae = np.full(n_rows, np.nan, dtype=float)
    mse = np.full(n_rows, np.nan, dtype=float)
    for i in range(n_rows):
        # Flatten per-row values so mixed 2D/3D model outputs remain comparable.
        row_pred = np.asarray(pred_arr[i], dtype=float).reshape(-1)
        row_tgt = np.asarray(tgt_arr[i], dtype=float).reshape(-1)
        n = min(row_pred.size, row_tgt.size)
        if n <= 0:
            continue
        row_pred = row_pred[:n]
        row_tgt = row_tgt[:n]
        mask = np.isfinite(row_pred) & np.isfinite(row_tgt)
        if not np.any(mask):
            continue
        diff = row_pred[mask] - row_tgt[mask]
        mae[i] = float(np.mean(np.abs(diff)))
        mse[i] = float(np.mean(np.square(diff)))
    return mae, mse


def _aggregate_by_group(values: np.ndarray, group_ids: list[str]) -> np.ndarray:
    if values.size == 0 or not group_ids:
        return np.array([], dtype=float)
    n = min(len(values), len(group_ids))
    values = np.asarray(values[:n], dtype=float)
    group_ids = list(group_ids[:n])
    grouped: dict[str, list[float]] = {}
    order: list[str] = []
    for gid, val in zip(group_ids, values):
        if not np.isfinite(val):
            continue
        if gid not in grouped:
            grouped[gid] = []
            order.append(gid)
        grouped[gid].append(float(val))
    return np.array([np.mean(grouped[gid]) for gid in order if grouped[gid]], dtype=float)


def _aggregate_pred_target_by_group(preds, targets, group_ids: list[str]) -> "tuple[np.ndarray, np.ndarray]":
    """Aggregate predictions/targets to one mean row per independent sample id."""
    pred_arr, tgt_arr = _aligned_pred_target(preds, targets)
    n_rows = min(len(pred_arr), len(tgt_arr), len(group_ids))
    if n_rows <= 0:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)

    pred_arr = np.asarray(pred_arr[:n_rows, :], dtype=float)
    tgt_arr = np.asarray(tgt_arr[:n_rows, :], dtype=float)
    gids = list(group_ids[:n_rows])

    grouped_pred: dict[str, list[np.ndarray]] = {}
    grouped_tgt: dict[str, list[np.ndarray]] = {}
    order: list[str] = []
    for idx, gid in enumerate(gids):
        if gid not in grouped_pred:
            grouped_pred[gid] = []
            grouped_tgt[gid] = []
            order.append(gid)
        grouped_pred[gid].append(pred_arr[idx, :])
        grouped_tgt[gid].append(tgt_arr[idx, :])

    agg_pred: list[np.ndarray] = []
    agg_tgt: list[np.ndarray] = []
    for gid in order:
        pgrp = np.asarray(grouped_pred[gid], dtype=float)
        tgrp = np.asarray(grouped_tgt[gid], dtype=float)

        pcount = np.sum(np.isfinite(pgrp), axis=0)
        psum = np.nansum(pgrp, axis=0)
        pmean = np.full(pgrp.shape[1], np.nan, dtype=float)
        np.divide(psum, pcount, out=pmean, where=pcount > 0)

        tcount = np.sum(np.isfinite(tgrp), axis=0)
        tsum = np.nansum(tgrp, axis=0)
        tmean = np.full(tgrp.shape[1], np.nan, dtype=float)
        np.divide(tsum, tcount, out=tmean, where=tcount > 0)

        agg_pred.append(pmean)
        agg_tgt.append(tmean)

    return np.asarray(agg_pred, dtype=float), np.asarray(agg_tgt, dtype=float)


def _compute_point_metrics_grouped(preds, targets, group_ids: list[str]) -> dict:
    """Compute point metrics after collapsing MC replicates to independent groups."""
    agg_pred, agg_tgt = _aggregate_pred_target_by_group(preds, targets, group_ids)
    metrics = _compute_point_metrics(agg_pred, agg_tgt)
    metrics["n_groups"] = int(len(agg_pred)) if agg_pred.ndim == 2 else 0
    return metrics


def _rowwise_mean(arr) -> np.ndarray:
    vals = np.asarray(arr, dtype=float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        row = np.asarray(vals[i], dtype=float).reshape(-1)
        row = row[np.isfinite(row)]
        if row.size:
            out[i] = float(np.mean(row))
    return out


def _group_summary(values: np.ndarray, group_ids: list[str]) -> dict[str, tuple[float, float, int]]:
    n = min(len(values), len(group_ids))
    vals = np.asarray(values[:n], dtype=float)
    gids = list(group_ids[:n])
    out: dict[str, tuple[float, float, int]] = {}
    for gid in dict.fromkeys(gids):
        idx = [i for i, g in enumerate(gids) if g == gid]
        grp = vals[idx]
        grp = grp[np.isfinite(grp)]
        if grp.size == 0:
            continue
        mu = float(np.mean(grp))
        sd = float(np.std(grp, ddof=1)) if grp.size > 1 else 0.0
        out[gid] = (mu, sd, int(grp.size))
    return out


def _anova_variance_components(values: np.ndarray, group_ids: list[str]) -> dict:
    n = min(len(values), len(group_ids))
    vals = np.asarray(values[:n], dtype=float)
    gids = list(group_ids[:n])
    mask = np.isfinite(vals)
    vals = vals[mask]
    gids = [g for g, m in zip(gids, mask) if m]
    n_tot = int(vals.size)
    if n_tot < 3:
        return {}

    grouped: dict[str, list[float]] = {}
    for g, v in zip(gids, vals):
        grouped.setdefault(g, []).append(float(v))
    groups = list(grouped.keys())
    g = len(groups)
    if g < 2:
        return {}

    grand = float(np.mean(vals))
    ss_within = 0.0
    ss_between = 0.0
    for gid in groups:
        arr = np.asarray(grouped[gid], dtype=float)
        mu = float(np.mean(arr))
        ss_within += float(np.sum((arr - mu) ** 2))
        ss_between += float(arr.size * (mu - grand) ** 2)

    df_within = n_tot - g
    df_between = g - 1
    ms_within = float(ss_within / df_within) if df_within > 0 else float("nan")
    ms_between = float(ss_between / df_between) if df_between > 0 else float("nan")
    ratio = float(ms_within / ms_between) if np.isfinite(ms_within) and np.isfinite(ms_between) and ms_between > 0 else float("nan")
    total = ms_within + ms_between if np.isfinite(ms_within) and np.isfinite(ms_between) else float("nan")
    noise_fraction = float(ms_within / total) if np.isfinite(total) and total > 0 else float("nan")
    icc = float(ms_between / total) if np.isfinite(total) and total > 0 else float("nan")
    return {
        "n_total": n_tot,
        "n_groups": g,
        "ss_within": float(ss_within),
        "ss_between": float(ss_between),
        "ms_within": ms_within,
        "ms_between": ms_between,
        "within_between_ratio": ratio,
        "noise_fraction": noise_fraction,
        "icc": icc,
    }


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _dm_test_from_diff(diff: np.ndarray, max_lag: int = 1) -> "tuple[float, float]":
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 5:
        return float("nan"), float("nan")
    mean_d = float(np.mean(d))
    centered = d - mean_d
    gamma0 = float(np.dot(centered, centered) / n)
    max_lag = int(max(0, min(max_lag, n - 1)))
    var_hac = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        var_hac += 2.0 * weight * cov
    if not np.isfinite(var_hac) or var_hac <= 0:
        return float("nan"), float("nan")
    stat = float(mean_d / math.sqrt(var_hac / n))
    p = float(2.0 * (1.0 - _normal_cdf(abs(stat))))
    return stat, p


def _wilcoxon_from_diff(diff: np.ndarray) -> "tuple[float, float]":
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size < 5 or scipy_stats is None:
        return float("nan"), float("nan")
    try:
        res = scipy_stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def _sign_test_from_diff(diff: np.ndarray) -> "tuple[float, float, float]":
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 5:
        return float("nan"), float("nan"), float("nan")
    wins = int(np.sum(d < 0))
    losses = int(np.sum(d > 0))
    n = wins + losses
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    win_rate = float(wins / n)
    p = float("nan")
    if scipy_stats is not None and hasattr(scipy_stats, "binomtest"):
        try:
            p = float(scipy_stats.binomtest(wins, n=n, p=0.5, alternative="two-sided").pvalue)
        except Exception:
            p = float("nan")
    return float(wins), win_rate, p


def _bh_adjust(pvals: list[float]) -> list[float]:
    n = len(pvals)
    out = [float("nan")] * n
    finite_idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    if not finite_idx:
        return out
    m = len(finite_idx)
    ordered = sorted(finite_idx, key=lambda i: pvals[i])
    prev = float("inf")
    for rank, idx in enumerate(reversed(ordered), start=1):
        i_rank = m - rank + 1
        raw = float(pvals[idx]) * m / i_rank
        val = min(prev, raw, 1.0)
        prev = val
        out[idx] = float(val)
    return out


def _cohen_d_from_diff(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return float("nan")
    mu = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(mu / sd)


def _interval_proxy_metrics(preds, targets, alpha: float = 0.1) -> dict:
    pred_arr, tgt_arr = _aligned_pred_target(preds, targets)
    pf = pred_arr.reshape(-1)
    tf = tgt_arr.reshape(-1)
    mask = np.isfinite(pf) & np.isfinite(tf)
    if not np.any(mask):
        return {
            "picp": float("nan"),
            "nominal_coverage": float("nan"),
            "coverage_gap": float("nan"),
            "coverage_deficit": float("nan"),
            "mpiw": float("nan"),
            "nmpiw": float("nan"),
            "interval_score": float("nan"),
            "q_abs_resid": float("nan"),
            "n_points": 0,
        }
    err = tf[mask] - pf[mask]
    abs_err = np.abs(err)
    alpha = float(min(max(alpha, 1e-6), 0.999999))
    q = float(np.quantile(abs_err, 1.0 - alpha))
    lower = pf[mask] - q
    upper = pf[mask] + q
    covered = (tf[mask] >= lower) & (tf[mask] <= upper)
    picp = float(np.mean(covered)) if covered.size else float("nan")
    nominal = float(1.0 - alpha)
    gap = float(picp - nominal) if np.isfinite(picp) else float("nan")
    deficit = float(max(0.0, nominal - picp)) if np.isfinite(picp) else float("nan")
    mpiw = float(2.0 * q) if np.isfinite(q) else float("nan")
    std_t = float(np.std(tf[mask], ddof=1)) if np.sum(mask) > 1 else float("nan")
    nmpiw = float(mpiw / std_t) if np.isfinite(mpiw) and np.isfinite(std_t) and std_t > 0 else float("nan")
    penalties = (2.0 / alpha) * ((lower - tf[mask]) * (tf[mask] < lower) + (tf[mask] - upper) * (tf[mask] > upper))
    interval_score = float(np.mean((upper - lower) + penalties)) if penalties.size else float("nan")
    return {
        "picp": picp,
        "nominal_coverage": nominal,
        "coverage_gap": gap,
        "coverage_deficit": deficit,
        "mpiw": mpiw,
        "nmpiw": nmpiw,
        "interval_score": interval_score,
        "q_abs_resid": q,
        "n_points": int(np.sum(mask)),
    }


def _bootstrap_grouped_skill(
    y_true,
    y_model,
    y_base,
    group_ids: list[str],
    n_boot: int,
    seed: int,
    mode: str = "iid",
    block_len: int = 3,
) -> dict:
    y_t = _safe_as_2d(y_true)
    y_m = _safe_as_2d(y_model)
    y_b = _safe_as_2d(y_base)
    n_rows = min(len(y_t), len(y_m), len(y_b), len(group_ids))
    if n_rows < 3:
        return {}
    y_t = y_t[:n_rows, :]
    y_m = y_m[:n_rows, :]
    y_b = y_b[:n_rows, :]
    gids = list(group_ids[:n_rows])

    group_to_idx: dict[str, list[int]] = {}
    order: list[str] = []
    for idx, gid in enumerate(gids):
        if gid not in group_to_idx:
            group_to_idx[gid] = []
            order.append(gid)
        group_to_idx[gid].append(idx)
    n_groups = len(order)
    if n_groups < 3:
        return {}

    rng = np.random.default_rng(seed)
    rmse_diff = []
    mae_diff = []
    r2_diff = []
    skill_vals = []
    beats = []
    block_len = int(max(1, block_len))
    for _ in range(int(max(1, n_boot))):
        if str(mode).lower() == "moving_block" and n_groups > 1:
            chosen = []
            while len(chosen) < n_groups:
                start = int(rng.integers(0, n_groups))
                for j in range(block_len):
                    chosen.append(order[(start + j) % n_groups])
                    if len(chosen) >= n_groups:
                        break
        else:
            chosen = rng.choice(order, size=n_groups, replace=True)
        sel_idx = []
        for gid in chosen:
            sel_idx.extend(group_to_idx[gid])
        if not sel_idx:
            continue
        yt = y_t[sel_idx, :]
        ym = y_m[sel_idx, :]
        yb = y_b[sel_idx, :]
        m_model = _compute_point_metrics(ym, yt)
        m_base = _compute_point_metrics(yb, yt)
        if not (np.isfinite(m_model["rmse"]) and np.isfinite(m_base["rmse"])):
            continue
        rmse_diff.append(m_model["rmse"] - m_base["rmse"])
        if np.isfinite(m_model["mae"]) and np.isfinite(m_base["mae"]):
            mae_diff.append(m_model["mae"] - m_base["mae"])
        if np.isfinite(m_model["r2"]) and np.isfinite(m_base["r2"]):
            r2_diff.append(m_model["r2"] - m_base["r2"])
        if m_base["rmse"] == 0:
            beats.append(0.0)
            continue
        skill = float(1.0 - m_model["rmse"] / m_base["rmse"])
        skill_vals.append(skill)
        beats.append(1.0 if skill > 0 else 0.0)

    def _q(arr, q):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.quantile(arr, q))

    skill_arr = np.asarray(skill_vals, dtype=float)
    rmse_arr = np.asarray(rmse_diff, dtype=float)
    mae_arr = np.asarray(mae_diff, dtype=float)
    r2_arr = np.asarray(r2_diff, dtype=float)
    beats_arr = np.asarray(beats, dtype=float)

    return {
        "n_boot_ok": int(np.sum(np.isfinite(skill_arr))),
        "skill_mean": float(np.nanmean(skill_arr)) if skill_arr.size else float("nan"),
        "skill_ci05": _q(skill_arr, 0.05),
        "skill_ci95": _q(skill_arr, 0.95),
        "rmse_diff_mean": float(np.nanmean(rmse_arr)) if rmse_arr.size else float("nan"),
        "rmse_diff_ci05": _q(rmse_arr, 0.05),
        "rmse_diff_ci95": _q(rmse_arr, 0.95),
        "mae_diff_mean": float(np.nanmean(mae_arr)) if mae_arr.size else float("nan"),
        "r2_diff_mean": float(np.nanmean(r2_arr)) if r2_arr.size else float("nan"),
        "prob_skill_gt0": float(np.nanmean(beats_arr)) if beats_arr.size else float("nan"),
    }


def _collect_prediction_payload(eval_cfg_path: Path) -> dict:
    cfg = eval_module.load_config(str(eval_cfg_path))
    config_dir = cfg["__config_dir"]
    model_type = cfg["model_type"]
    model_name = cfg.get("model_name", "")
    data_cfg = cfg["data"]
    eval_cfg = eval_module.merge_eval_config(cfg)

    data_cfg["data_dir"], data_cfg["sample_subdir"] = eval_module._resolve_data_paths(data_cfg, config_dir)
    for key in ["historic_path", "thresholds_path", "normalization_path"]:
        if eval_cfg.get(key):
            eval_cfg[key] = str(eval_module._resolve_path_from_config(eval_cfg[key], config_dir))

    model_config = eval_module.load_model_config(
        data_cfg["data_dir"],
        data_cfg["forecast_name"],
        model_name,
        fallback_data=data_cfg,
    )
    input_columns = model_config["input_columns"]
    output_columns = model_config["output_columns"]
    input_rows = slice(model_config["input_row_1"], model_config["input_row_2"])
    output_rows = model_config["output_rows"]
    input_aggregation = str(model_config.get("input_aggregation", data_cfg.get("input_aggregation", "none"))).lower()
    split_cfg = cfg.get("data_split", {"random_state": 42})
    load_fault_tolerant, enforced_nan_tolerance = _model_sample_policy(model_type, split_cfg)

    split_base_dir = Path(data_cfg.get("forecast_dir", eval_cfg_path.parent))
    test_samples = eval_module.load_split_samples(
        data_cfg["data_dir"],
        data_cfg["sample_subdir"],
        data_cfg["forecast_name"],
        input_columns,
        output_columns,
        input_rows,
        output_rows,
        "test_files.txt",
        split_source_dir=split_base_dir,
        fault_tolerant=load_fault_tolerant,
        input_aggregation=input_aggregation,
    )
    if load_fault_tolerant and enforced_nan_tolerance is not None:
        before_n = len(test_samples)
        test_samples = _filter_samples_by_nan_tolerance(test_samples, float(enforced_nan_tolerance))
        dropped_n = max(0, before_n - len(test_samples))
        if dropped_n > 0:
            print(
                f"[INFO] {eval_cfg_path.parent.name}: model={model_type} "
                f"enforced fault_tolerant=True with nan_tolerance={enforced_nan_tolerance:.3f}; "
                f"dropped {dropped_n} test sample(s)."
            )

    # Use filenames from loaded samples so evidence support counts reflect evaluated rows.
    split_files = _sample_names_from_loaded_samples(test_samples)
    if not split_files:
        split_files = eval_module._read_split_files(split_base_dir, "test_files.txt")

    train_samples = None
    if model_type == "gp_regressor":
        train_samples = eval_module.load_split_samples(
            data_cfg["data_dir"],
            data_cfg["sample_subdir"],
            data_cfg["forecast_name"],
            input_columns,
            output_columns,
            input_rows,
            output_rows,
            "train_files.txt",
            split_source_dir=split_base_dir,
            fault_tolerant=load_fault_tolerant,
            input_aggregation=input_aggregation,
        )

    if model_type == "transformer":
        X_test = np.array([s[0] for s in test_samples], dtype=float)
        y_test = np.array([s[1] for s in test_samples], dtype=float)
    else:
        X_test = np.array([s[0].flatten() for s in test_samples], dtype=float)
        y_test = np.array([s[1].flatten() for s in test_samples], dtype=float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval_module.load_model(model_type, data_cfg, split_cfg, model_name, model_config, device, train_samples, config_dir)

    if model_type == "gp_regressor":
        pred_model = eval_module._predict_gp_bundle(model, X_test, device)
    elif model_type == "transformer":
        pred_model = model(torch.tensor(X_test, dtype=torch.float32, device=device)).detach().cpu().numpy()
    elif model_type == "xgb_regressor":
        out_dim = y_test.shape[1] if y_test.ndim > 1 else 1
        pred_model = model.predict(X_test).reshape(-1, out_dim)
    else:
        raise ValueError(f"Unsupported model_type for inference: {model_type}")

    historic = eval_cfg["historic_path"]
    sample_subdir = data_cfg.get("sample_subdir", "samples")
    baseline_preds = {}
    for label, fn in {
        "naive": eval_module.evaluate_naive,
        "seasonal": eval_module.evaluate_seasonal,
        "linear": eval_module.evaluate_linear,
    }.items():
        try:
            if label == "linear":
                pred_b, _ = fn(
                    data_cfg["data_dir"],
                    data_cfg["forecast_name"],
                    test_samples,
                    historic,
                    output_columns,
                    sample_subdir=sample_subdir,
                )
            else:
                pred_b, _ = fn(
                    test_samples,
                    historic,
                    output_columns,
                    data_cfg["data_dir"],
                    sample_subdir=sample_subdir,
                )
            baseline_preds[label] = _safe_as_2d(pred_b)
        except Exception as exc:
            print(f"[WARN] Could not compute {label} baseline for {eval_cfg_path.parent.name}: {exc}")
            baseline_preds[label] = np.full_like(_safe_as_2d(y_test), np.nan, dtype=float)

    return {
        "y_test": _safe_as_2d(y_test),
        "X_test": X_test,
        "pred_model": _safe_as_2d(pred_model),
        "baseline_preds": baseline_preds,
        "split_files": split_files,
        "model_type": model_type,
        "model_name": model_name,
        "model_config": model_config,
        "data_cfg": data_cfg,
        "split_cfg": split_cfg,
        "train_samples": train_samples,
    }


def _compute_retraining_stability(
    eval_cfg_path: Path,
    variant_dir: Path,
    args: argparse.Namespace,
    payload: dict,
) -> dict:
    """Run N independent retraining replicates and return stability statistics.

    Replicate-0 is the already-trained model from *payload* (free — no retraining).
    Replicates 1..N-1 are trained by modifying the ``cv_tuning.seed`` in the training
    config and calling ``e_Train.py`` as a subprocess.  All replicates are evaluated on
    the **same** held-out test data as replicate-0 (same split, same samples) so that
    predictions are directly comparable and can be ensemble-averaged.

    Only ``hyperparameters.cv_tuning.seed`` is varied — ``data_split.random_state`` is
    kept fixed — so that the test set is identical across replicates and prediction
    averaging is meaningful.

    Returns an empty dict when ``args.stability_replicates <= 1``.
    """
    n_reps = int(getattr(args, "stability_replicates", 5))
    if n_reps <= 1:
        return {}

    base_seed    = int(getattr(args, "bootstrap_seed", 42))
    cv_thr       = float(getattr(args, "stability_cv_threshold", 0.15))
    y_test       = payload["y_test"]
    X_test       = payload["X_test"]
    pred_rep0    = payload["pred_model"]
    model_type   = payload["model_type"]
    model_name   = payload["model_name"]
    model_config = payload["model_config"]
    data_cfg_abs = payload["data_cfg"]    # data_dir already resolved to absolute
    split_cfg    = payload["split_cfg"]
    train_samples = payload["train_samples"]  # non-None only for gp_regressor

    data_dir_abs  = Path(str(data_cfg_abs["data_dir"]))
    orig_forecast = str(data_cfg_abs.get("forecast_name", ""))

    # Rep-0 metrics (original trained model)
    m0 = _compute_point_metrics(pred_rep0, y_test)
    r2_vals   = [_safe_float(m0["r2"])]
    rmse_vals = [_safe_float(m0["rmse"])]
    mae_vals  = [_safe_float(m0["mae"])]
    pred_list = [_safe_as_2d(pred_rep0)]

    train_script = Path(__file__).resolve().parent / "e_Train.py"
    # Prefer the variant's training config as the base; fall back to eval config
    train_cfg_candidate = variant_dir / f"config_train_{variant_dir.name}.yml"
    base_cfg_for_retrain = train_cfg_candidate if train_cfg_candidate.exists() else eval_cfg_path

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_paths: list[Path] = []

    for r in range(1, n_reps):
        rep_seed     = base_seed + r
        rep_forecast = f"{orig_forecast}_stab{r:03d}"
        rep_cfg_path = variant_dir / f"config_stability_rep{r:03d}.yml"
        tmp_paths.append(rep_cfg_path)

        # Build a modified copy of the training config
        raw_cfg = train_module.load_config(str(base_cfg_for_retrain))
        raw_cfg.pop("__config_dir", None)
        rep_cfg = copy.deepcopy(raw_cfg)
        # Vary only the training/tuning seed; keep data split identical
        rep_cfg.setdefault("hyperparameters", {}).setdefault("cv_tuning", {})["seed"] = rep_seed
        # Redirect model output so we don't overwrite the original
        rep_cfg.setdefault("data", {})["forecast_name"] = rep_forecast

        with open(rep_cfg_path, "w", encoding="utf-8") as fh:
            yaml.dump(rep_cfg, fh, default_flow_style=False, allow_unicode=True)

        print(f"[INFO] Stability rep {r}/{n_reps - 1}: seed={rep_seed}")
        proc = subprocess.run(
            [sys.executable, str(train_script), "--config", str(rep_cfg_path)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            print(f"[WARN] Stability rep {r} training failed (rc={proc.returncode})")
            if proc.stderr:
                print(proc.stderr[-800:])
            continue

        # Load the newly trained model.
        # model_config is reloaded per-replicate because Optuna may have found a
        # different architecture (e.g. different d_model or num_layers) under the new
        # seed.  Reusing rep-0's model_config would cause state_dict shape mismatches.
        try:
            rep_data_cfg = {**data_cfg_abs, "forecast_name": rep_forecast}
            rep_model_config = eval_module.load_model_config(
                str(data_dir_abs),
                rep_forecast,
                model_name,
                fallback_data=rep_data_cfg,
            )
            rep_model = eval_module.load_model(
                model_type, rep_data_cfg, split_cfg,
                model_name, rep_model_config, device, train_samples, variant_dir,
            )
        except Exception as exc:
            print(f"[WARN] Stability rep {r} model load failed: {exc}")
            continue

        # Inference on the same X_test as rep-0
        try:
            if model_type == "gp_regressor":
                pred_r = _safe_as_2d(eval_module._predict_gp_bundle(rep_model, X_test, device))
            elif model_type == "transformer":
                pred_r = _safe_as_2d(
                    rep_model(torch.tensor(X_test, dtype=torch.float32, device=device))
                    .detach().cpu().numpy()
                )
            else:  # xgb_regressor / xgb_classifier
                out_dim = y_test.shape[1] if y_test.ndim > 1 else 1
                pred_r = _safe_as_2d(rep_model.predict(X_test).reshape(-1, out_dim))
        except Exception as exc:
            print(f"[WARN] Stability rep {r} inference failed: {exc}")
            continue

        mr = _compute_point_metrics(pred_r, y_test)
        r2_vals.append(_safe_float(mr["r2"]))
        rmse_vals.append(_safe_float(mr["rmse"]))
        mae_vals.append(_safe_float(mr["mae"]))
        pred_list.append(pred_r)

        # Delete the replicate model files to reclaim disk space
        rep_model_dir = data_dir_abs / "forecasts" / rep_forecast
        if rep_model_dir.exists():
            shutil.rmtree(rep_model_dir, ignore_errors=True)

    # Remove temp config files
    for p in tmp_paths:
        p.unlink(missing_ok=True)

    n_ok = int(len(r2_vals))
    if n_ok < 2:
        return {
            "stability_n_replicates": n_reps,
            "stability_n_successful": n_ok,
            "stability_cv_threshold": cv_thr,
            "gate_stability": False,
        }

    # --- Per-model metric statistics (used for gate and variance diagnostics) ---
    r2_arr   = np.array([v for v in r2_vals   if np.isfinite(v)], dtype=float)
    rmse_arr = np.array([v for v in rmse_vals  if np.isfinite(v)], dtype=float)
    mae_arr  = np.array([v for v in mae_vals   if np.isfinite(v)], dtype=float)

    r2_mean   = float(np.mean(r2_arr))                           if r2_arr.size   else float("nan")
    r2_std    = float(np.std(r2_arr,   ddof=1))                  if r2_arr.size   >= 2 else float("nan")
    rmse_mean = float(np.mean(rmse_arr))                         if rmse_arr.size else float("nan")
    rmse_std  = float(np.std(rmse_arr,  ddof=1))                 if rmse_arr.size >= 2 else float("nan")

    r2_cv   = (float(r2_std   / abs(r2_mean))
               if np.isfinite(r2_std)   and np.isfinite(r2_mean)   and abs(r2_mean)   > 1e-12
               else float("nan"))
    rmse_cv = (float(rmse_std / abs(rmse_mean))
               if np.isfinite(rmse_std) and np.isfinite(rmse_mean) and abs(rmse_mean) > 1e-12
               else float("nan"))
    r2_lcb  = (float(r2_mean - 2.0 * r2_std)
               if np.isfinite(r2_std) and np.isfinite(r2_mean)
               else float("nan"))

    # --- Ensemble (prediction-averaged) statistics ---
    # All reps use the same test set, so predictions are directly comparable.
    # Per Jensen's inequality, r2_ensemble >= r2_mean always; the gap measures
    # how much ensembling helps.
    n_test = pred_list[0].shape[0]
    stack = [p for p in pred_list if p.shape[0] == n_test]
    r2_ens = r2_mean
    rmse_ens = rmse_mean
    mae_ens  = float(np.mean(mae_arr)) if mae_arr.size else float("nan")
    if len(stack) >= 2:
        pred_ensemble = np.mean(np.stack(stack, axis=0), axis=0)
        ens_m = _compute_point_metrics(pred_ensemble, y_test)
        r2_ens   = _safe_float(ens_m["r2"])
        rmse_ens = _safe_float(ens_m["rmse"])
        mae_ens  = _safe_float(ens_m["mae"])

    ens_benefit = (float(r2_ens - r2_mean)
                   if np.isfinite(r2_ens) and np.isfinite(r2_mean)
                   else float("nan"))

    # --- Gate: CV low enough AND worst-case lower tail still positive ---
    gate_stability = bool(
        n_ok >= 2
        and np.isfinite(r2_cv)
        and r2_cv <= cv_thr
        and np.isfinite(r2_lcb)
        and r2_lcb > 0.0
    )

    return {
        "stability_r2_mean":          r2_mean,
        "stability_r2_std":           r2_std,
        "stability_r2_cv":            r2_cv,
        "stability_r2_lcb":           r2_lcb,
        "stability_rmse_cv":          rmse_cv,
        "stability_r2_ensemble":      r2_ens,
        "stability_rmse_ensemble":    rmse_ens,
        "stability_mae_ensemble":     mae_ens,
        "stability_ensemble_benefit": ens_benefit,
        "stability_n_replicates":     n_reps,
        "stability_n_successful":     n_ok,
        "stability_cv_threshold":     cv_thr,
        "gate_stability":             gate_stability,
    }


def _compute_statistical_evidence(plan: DatasetPlan, best_row: "pd.Series", args: argparse.Namespace) -> dict:
    evidence: dict[str, float | str | int | bool] = {}
    variant_dir, eval_cfg_path, resolve_status = _find_best_variant_eval_config(plan, best_row)
    evidence["evidence_variant_resolution"] = str(resolve_status)
    if resolve_status == "fallback_variant_mismatch":
        evidence["evidence_status"] = "variant_mismatch"
        evidence["evidence_variant_dir"] = str(variant_dir) if variant_dir is not None else ""
        return evidence
    if eval_cfg_path is None or not eval_cfg_path.exists():
        evidence["evidence_status"] = "missing_eval_config"
        return evidence

    try:
        payload = _collect_prediction_payload(eval_cfg_path)
    except Exception as exc:
        evidence["evidence_status"] = f"prediction_failed: {type(exc).__name__}"
        print(f"[WARN] Could not collect prediction payload for {plan.dataset_dir.name}: {exc}")
        traceback.print_exc()
        return evidence

    y_test = payload["y_test"]
    pred_model = payload["pred_model"]
    split_files = payload["split_files"]
    n_rows = min(len(y_test), len(pred_model), len(split_files))
    if n_rows <= 0:
        evidence["evidence_status"] = "empty_test_payload"
        return evidence

    y_test = y_test[:n_rows, :]
    pred_model = pred_model[:n_rows, :]
    split_files = list(split_files[:n_rows])
    group_ids = [_base_sample_id(s) for s in split_files]
    n_raw = len(list(dict.fromkeys(group_ids)))

    model_metrics = _compute_point_metrics_grouped(pred_model, y_test, group_ids)
    mae_model_rows, mse_model_rows = _compute_per_sample_losses(pred_model, y_test)
    y_row_mean = _rowwise_mean(y_test)
    target_vc = _anova_variance_components(y_row_mean, group_ids)
    err_vc = _anova_variance_components(mae_model_rows, group_ids)
    grp_y = _group_summary(y_row_mean, group_ids)
    grp_mae = _group_summary(mae_model_rows, group_ids)
    evidence["n_eval_rows_test"] = int(n_rows)
    evidence["n_eval_raw_segments"] = int(n_raw)
    # Semantics markers keep downstream usage explicit.
    evidence["metric_semantics_point"] = "group_aggregated_independent_samples"
    evidence["metric_semantics_tests"] = "group_aggregated_differences"
    evidence["metric_semantics_interval"] = "replicate_pooled_diagnostic"
    evidence["n_eval_points_finite_model"] = int(model_metrics["n_finite"])
    evidence["n_eval_points_finite_model_grouped"] = int(model_metrics["n_finite"])
    evidence["n_eval_groups_model"] = int(model_metrics.get("n_groups", 0))
    evidence["sample_reliability_weight"] = float(min(1.0, math.sqrt(max(0.0, n_raw) / max(1.0, float(args.evidence_ref_raw_samples)))))
    evidence["mc_target_within_ms"] = _safe_float(target_vc.get("ms_within", float("nan")))
    evidence["mc_target_between_ms"] = _safe_float(target_vc.get("ms_between", float("nan")))
    evidence["mc_target_wb_ratio"] = _safe_float(target_vc.get("within_between_ratio", float("nan")))
    evidence["mc_target_noise_fraction"] = _safe_float(target_vc.get("noise_fraction", float("nan")))
    evidence["mc_target_icc"] = _safe_float(target_vc.get("icc", float("nan")))
    evidence["mc_model_mae_within_ms"] = _safe_float(err_vc.get("ms_within", float("nan")))
    evidence["mc_model_mae_between_ms"] = _safe_float(err_vc.get("ms_between", float("nan")))
    evidence["mc_model_mae_wb_ratio"] = _safe_float(err_vc.get("within_between_ratio", float("nan")))
    evidence["mc_model_mae_noise_fraction"] = _safe_float(err_vc.get("noise_fraction", float("nan")))
    evidence["mc_model_mae_icc"] = _safe_float(err_vc.get("icc", float("nan")))
    within_sd_t = math.sqrt(evidence["mc_target_within_ms"]) if np.isfinite(evidence["mc_target_within_ms"]) and evidence["mc_target_within_ms"] >= 0 else float("nan")
    between_sd_t = math.sqrt(evidence["mc_target_between_ms"]) if np.isfinite(evidence["mc_target_between_ms"]) and evidence["mc_target_between_ms"] >= 0 else float("nan")
    evidence["mc_target_within_sd"] = _safe_float(within_sd_t)
    evidence["mc_target_between_sd"] = _safe_float(between_sd_t)
    evidence["rmse_to_mc_within_sd"] = float(model_metrics["rmse"] / within_sd_t) if np.isfinite(model_metrics["rmse"]) and np.isfinite(within_sd_t) and within_sd_t > 0 else float("nan")
    evidence["rmse_to_mc_between_sd"] = float(model_metrics["rmse"] / between_sd_t) if np.isfinite(model_metrics["rmse"]) and np.isfinite(between_sd_t) and between_sd_t > 0 else float("nan")

    # Correlation: do higher MC replicate spread segments also have higher model error?
    corr_pairs = []
    for gid in sorted(set(grp_y.keys()) & set(grp_mae.keys())):
        _, y_sd, _ = grp_y[gid]
        mae_mu, _, _ = grp_mae[gid]
        if np.isfinite(y_sd) and np.isfinite(mae_mu):
            corr_pairs.append((y_sd, mae_mu))
    if len(corr_pairs) >= 3:
        a = np.asarray([p[0] for p in corr_pairs], dtype=float)
        b = np.asarray([p[1] for p in corr_pairs], dtype=float)
        if np.isfinite(a).all() and np.isfinite(b).all() and np.std(a) > 0 and np.std(b) > 0:
            evidence["mc_uncertainty_vs_error_corr"] = float(np.corrcoef(a, b)[0, 1])
        else:
            evidence["mc_uncertainty_vs_error_corr"] = float("nan")
    else:
        evidence["mc_uncertainty_vs_error_corr"] = float("nan")

    baseline_preds = payload["baseline_preds"]
    pval_records: list[tuple[str, str, float]] = []
    baseline_scores: list[int] = []
    interval_alpha = float(getattr(args, "interval_alpha", 0.1))
    coverage_tol = float(getattr(args, "coverage_tolerance", 0.03))
    model_int = _interval_proxy_metrics(pred_model, y_test, alpha=interval_alpha)
    evidence["model_picp"] = _safe_float(model_int["picp"])
    evidence["model_nominal_coverage"] = _safe_float(model_int["nominal_coverage"])
    evidence["model_coverage_gap"] = _safe_float(model_int["coverage_gap"])
    evidence["model_coverage_deficit"] = _safe_float(model_int["coverage_deficit"])
    evidence["model_nmpiw"] = _safe_float(model_int["nmpiw"])
    evidence["model_interval_score"] = _safe_float(model_int["interval_score"])
    evidence["model_interval_is_diagnostic"] = True

    for bname in BASELINE_ORDER:
        pred_b = _safe_as_2d(baseline_preds.get(bname, np.full_like(y_test, np.nan, dtype=float)))[:n_rows, :]
        mae_m, mse_m = mae_model_rows, mse_model_rows
        mae_b, mse_b = _compute_per_sample_losses(pred_b, y_test)
        mse_diff = mse_m - mse_b
        ae_diff = mae_m - mae_b
        mse_diff_group = _aggregate_by_group(mse_diff, group_ids)
        ae_diff_group = _aggregate_by_group(ae_diff, group_ids)
        n_groups_eff = int(np.sum(np.isfinite(mse_diff_group)))
        evidence[f"n_groups_{bname}"] = n_groups_eff

        dm_stat, dm_p = _dm_test_from_diff(mse_diff_group, max_lag=int(args.dm_max_lag))
        w_stat, w_p = _wilcoxon_from_diff(ae_diff_group)
        sign_wins, sign_win_rate, sign_p = _sign_test_from_diff(ae_diff_group)
        boot = _bootstrap_grouped_skill(
            y_test,
            pred_model,
            pred_b,
            group_ids=group_ids,
            n_boot=int(args.bootstrap_iterations),
            seed=int(args.bootstrap_seed),
            mode=str(getattr(args, "bootstrap_mode", "iid")),
            block_len=int(getattr(args, "bootstrap_block_len", 3)),
        )

        model_rmse = model_metrics["rmse"]
        base_metrics = _compute_point_metrics_grouped(pred_b, y_test, group_ids)
        baseline_rmse = base_metrics["rmse"]
        evidence[f"n_eval_groups_{bname}"] = int(base_metrics.get("n_groups", 0))
        skill = float(1.0 - model_rmse / baseline_rmse) if np.isfinite(model_rmse) and np.isfinite(baseline_rmse) and baseline_rmse > 0 else float("nan")
        int_base = _interval_proxy_metrics(pred_b, y_test, alpha=interval_alpha)

        prefix = f"vs_{bname}"
        evidence[f"dm_stat_{prefix}"] = dm_stat
        evidence[f"dm_p_{prefix}"] = dm_p
        evidence[f"wilcoxon_stat_{prefix}"] = w_stat
        evidence[f"wilcoxon_p_{prefix}"] = w_p
        evidence[f"sign_wins_{prefix}"] = sign_wins
        evidence[f"sign_win_rate_{prefix}"] = sign_win_rate
        evidence[f"sign_p_{prefix}"] = sign_p
        evidence[f"skill_{prefix}"] = skill
        evidence[f"effect_median_ae_diff_{prefix}"] = float(np.nanmedian(ae_diff_group)) if ae_diff_group.size else float("nan")
        evidence[f"effect_mean_ae_diff_{prefix}"] = float(np.nanmean(ae_diff_group)) if ae_diff_group.size else float("nan")
        evidence[f"effect_cohen_d_ae_diff_{prefix}"] = _cohen_d_from_diff(ae_diff_group)
        evidence[f"bootstrap_n_{prefix}"] = int(boot.get("n_boot_ok", 0))
        evidence[f"bootstrap_skill_mean_{prefix}"] = _safe_float(boot.get("skill_mean"))
        evidence[f"bootstrap_skill_ci05_{prefix}"] = _safe_float(boot.get("skill_ci05"))
        evidence[f"bootstrap_skill_ci95_{prefix}"] = _safe_float(boot.get("skill_ci95"))
        evidence[f"bootstrap_prob_skill_gt0_{prefix}"] = _safe_float(boot.get("prob_skill_gt0"))
        evidence[f"bootstrap_rmse_diff_mean_{prefix}"] = _safe_float(boot.get("rmse_diff_mean"))
        evidence[f"bootstrap_rmse_diff_ci05_{prefix}"] = _safe_float(boot.get("rmse_diff_ci05"))
        evidence[f"bootstrap_rmse_diff_ci95_{prefix}"] = _safe_float(boot.get("rmse_diff_ci95"))
        evidence[f"bootstrap_r2_diff_mean_{prefix}"] = _safe_float(boot.get("r2_diff_mean"))
        evidence[f"lcb95_skill_{prefix}"] = _safe_float(boot.get("skill_ci05"))
        evidence[f"{bname}_picp"] = _safe_float(int_base["picp"])
        evidence[f"{bname}_nominal_coverage"] = _safe_float(int_base["nominal_coverage"])
        evidence[f"{bname}_coverage_gap"] = _safe_float(int_base["coverage_gap"])
        evidence[f"{bname}_coverage_deficit"] = _safe_float(int_base["coverage_deficit"])
        evidence[f"{bname}_nmpiw"] = _safe_float(int_base["nmpiw"])
        evidence[f"{bname}_interval_score"] = _safe_float(int_base["interval_score"])
        evidence[f"{bname}_interval_is_diagnostic"] = True
        evidence[f"picp_delta_{prefix}"] = _safe_float(model_int["picp"]) - _safe_float(int_base["picp"])
        evidence[f"nmpiw_delta_{prefix}"] = _safe_float(model_int["nmpiw"]) - _safe_float(int_base["nmpiw"])
        evidence[f"interval_score_delta_{prefix}"] = _safe_float(model_int["interval_score"]) - _safe_float(int_base["interval_score"])

        pval_records.extend([
            (prefix, "dm", dm_p),
            (prefix, "wilcoxon", w_p),
            (prefix, "sign", sign_p),
        ])

        gate_min_n = bool(n_raw >= int(args.evidence_min_raw_samples))
        gate_prob = bool(np.isfinite(evidence[f"bootstrap_prob_skill_gt0_{prefix}"]) and evidence[f"bootstrap_prob_skill_gt0_{prefix}"] >= float(args.evidence_min_prob))
        gate_lcb = bool(np.isfinite(evidence[f"lcb95_skill_{prefix}"]) and evidence[f"lcb95_skill_{prefix}"] > 0)
        gate_dm = bool(np.isfinite(dm_p) and dm_p < float(args.evidence_alpha) and np.isfinite(dm_stat) and dm_stat < 0)
        gate_wilc = bool(np.isfinite(w_p) and w_p < float(args.evidence_alpha))
        gate_sign = bool(np.isfinite(sign_p) and sign_p < float(args.evidence_alpha) and np.isfinite(sign_win_rate) and sign_win_rate > 0.5)
        gate_cov = bool(
            np.isfinite(_safe_float(model_int["coverage_deficit"]))
            and np.isfinite(_safe_float(int_base["coverage_deficit"]))
            and _safe_float(model_int["coverage_deficit"]) <= coverage_tol
            and _safe_float(model_int["coverage_deficit"]) <= _safe_float(int_base["coverage_deficit"]) + coverage_tol
        )
        evidence[f"gate_min_raw_{prefix}"] = gate_min_n
        evidence[f"gate_prob_{prefix}"] = gate_prob
        evidence[f"gate_lcb_{prefix}"] = gate_lcb
        evidence[f"gate_dm_{prefix}"] = gate_dm
        evidence[f"gate_wilcoxon_{prefix}"] = gate_wilc
        evidence[f"gate_sign_{prefix}"] = gate_sign
        evidence[f"gate_coverage_{prefix}"] = gate_cov
        score = int(gate_lcb) + int(gate_dm) + int(gate_wilc) + int(gate_sign) + int(gate_cov)
        evidence[f"evidence_score_{prefix}"] = score
        baseline_scores.append(score)

    if pval_records:
        adj = _bh_adjust([p for _, _, p in pval_records])
        for (prefix, test_name, _), q in zip(pval_records, adj):
            evidence[f"{test_name}_q_{prefix}"] = float(q)

        for bname in BASELINE_ORDER:
            prefix = f"vs_{bname}"
            dm_q = _safe_float(evidence.get(f"dm_q_{prefix}", float("nan")))
            wilc_q = _safe_float(evidence.get(f"wilcoxon_q_{prefix}", float("nan")))
            sign_q = _safe_float(evidence.get(f"sign_q_{prefix}", float("nan")))
            dm_stat = _safe_float(evidence.get(f"dm_stat_{prefix}", float("nan")))
            sign_wr = _safe_float(evidence.get(f"sign_win_rate_{prefix}", float("nan")))
            evidence[f"gate_dm_q_{prefix}"] = bool(np.isfinite(dm_q) and dm_q < float(args.evidence_alpha) and np.isfinite(dm_stat) and dm_stat < 0)
            evidence[f"gate_wilcoxon_q_{prefix}"] = bool(np.isfinite(wilc_q) and wilc_q < float(args.evidence_alpha))
            evidence[f"gate_sign_q_{prefix}"] = bool(np.isfinite(sign_q) and sign_q < float(args.evidence_alpha) and np.isfinite(sign_wr) and sign_wr > 0.5)

    if baseline_scores:
        evidence["evidence_score_overall_min"] = int(min(baseline_scores))
        evidence["evidence_score_overall_mean"] = float(np.mean(baseline_scores))
    else:
        evidence["evidence_score_overall_min"] = 0
        evidence["evidence_score_overall_mean"] = 0.0
    evidence["interval_alpha"] = interval_alpha
    evidence["evidence_alpha"] = float(args.evidence_alpha)
    evidence["bootstrap_mode"] = str(getattr(args, "bootstrap_mode", "iid"))
    evidence["bootstrap_block_len"] = int(getattr(args, "bootstrap_block_len", 3))

    if getattr(args, "stability_replicates", 1) > 1:
        try:
            stab = _compute_retraining_stability(eval_cfg_path, variant_dir, args, payload)
            evidence.update(stab)
            if stab:
                print(
                    f"[INFO] Stability check: {stab.get('stability_n_successful', 0)}/"
                    f"{stab.get('stability_n_replicates', 0)} reps OK, "
                    f"R² CV={stab.get('stability_r2_cv', float('nan')):.3f}, "
                    f"gate_stability={stab.get('gate_stability', False)}"
                )
        except Exception as _stab_exc:
            print(f"[WARN] Retraining stability check failed: {_stab_exc}")
            traceback.print_exc()

    evidence["evidence_status"] = "ok"
    evidence["evidence_variant_dir"] = str(variant_dir) if variant_dir is not None else ""
    return evidence


def _compile_feature_inclusion_heatmap(
    perf_df: "pd.DataFrame",
    plans: list,
    data_root: Path,
    target_order: list[str],
    dataset_prefix: str = "MC",
    sweep_namespace: str = "feature_sweeps",
) -> Path:
    """Binary heatmap: 1 if predictor is in the best model's feature set for a target, 0 otherwise."""
    dataset_to_plan = {plan.dataset_dir.name: plan for plan in plans}
    best_per_dataset = perf_df.drop_duplicates(subset="dataset", keep="first")

    # Resolve feature lists per target
    target_features: dict[str, list[str]] = {}  # target_name -> [feature, ...]
    target_labels: dict[str, str] = {}  # target_name -> display label

    for _, row in best_per_dataset.iterrows():
        dataset_name = str(row["dataset"])
        plan = dataset_to_plan.get(dataset_name)
        if plan is None:
            continue
        row_count = int(row["row_count"])
        feature_tag = str(row["feature_tag"])
        out_dir = _forecast_sweeps_dir(plan.dataset_dir)

        # Try feature_selected_subsets first, then feature_search_trace as fallback
        features_list: list[str] | None = None
        for csv_name in [
            f"feature_selected_subsets_r{row_count:03d}.csv",
            f"feature_search_trace_r{row_count:03d}.csv",
        ]:
            csv_path = out_dir / csv_name
            if not csv_path.exists():
                continue
            try:
                df_sub = pd.read_csv(csv_path)
                matched = df_sub[df_sub["feature_tag"].astype(str) == feature_tag]
                if not matched.empty and "features" in matched.columns:
                    raw = str(matched.iloc[0]["features"]).strip()
                    if raw and raw.lower() != "nan":
                        features_list = [f.strip() for f in raw.split("|") if f.strip()]
                        break
            except Exception as exc:
                print(f"[WARN] Could not read {csv_path.name} for {dataset_name}: {exc}")

        if not features_list:
            print(f"[WARN] Could not resolve features for {dataset_name} tag={feature_tag}; skipping.")
            continue

        target_name = _derive_target_name(dataset_name, dataset_prefix)
        target_features[target_name] = features_list
        target_labels[target_name] = clean_target_label(target_name, dataset_prefix)

    if not target_features:
        return Path()

    # Collect all unique features
    all_features_set: set[str] = set()
    for feats in target_features.values():
        all_features_set.update(feats)

    # Canonical feature order from Consolidated_sparse.csv
    csv_candidates = [
        data_root / "Consolidated_sparse.csv",
        data_root.parent / "regression" / "Consolidated_sparse.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), csv_candidates[0])
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        csv_features = [col for col in header if col in all_features_set]
        all_features = csv_features + [f for f in sorted(all_features_set) if f not in csv_features]
    except Exception:
        all_features = sorted(all_features_set)

    if not all_features:
        return Path()

    # Order targets by R2 rank (target_order), append unmatched
    def _norm(s: str) -> str:
        text = str(s).lower().replace("\u00b5", "u").replace("\u00b0", "deg")
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        return re.sub(r"[^a-z0-9]", "", text)

    available_keys = set(target_features.keys())
    used: set[str] = set()
    targets: list[str] = []
    yticklabels: list[str] = []

    for requested in (target_order or []):
        req_norm = _norm(requested)
        match = None
        for key in available_keys - used:
            key_norm = _norm(key)
            if key_norm == req_norm or key_norm.endswith(req_norm) or req_norm.endswith(key_norm):
                match = key
                break
        if match is not None:
            targets.append(match)
            yticklabels.append(target_labels.get(match, clean_target_label(match, dataset_prefix)))
            used.add(match)

    for key in sorted(available_keys - used):
        targets.append(key)
        yticklabels.append(target_labels.get(key, clean_target_label(key, dataset_prefix)))

    # Build binary matrix
    feature_to_idx = {feat: idx for idx, feat in enumerate(all_features)}
    matrix = np.zeros((len(targets), len(all_features)), dtype=float)
    for i, target in enumerate(targets):
        for feat in target_features[target]:
            if feat in feature_to_idx:
                matrix[i, feature_to_idx[feat]] = 1.0

    # Group: multi-target features (>1 target) first, then single-target
    target_feature_sets = {t: set(target_features[t]) for t in targets}
    presence_count = {
        feat: sum(1 for t in targets if feat in target_feature_sets.get(t, set()))
        for feat in all_features
    }
    target_rank = {t: idx for idx, t in enumerate(targets)}

    multi_target_features = [f for f in all_features if presence_count.get(f, 0) > 1]
    single_target_features = [f for f in all_features if presence_count.get(f, 0) == 1]

    multi_target_features.sort(key=lambda f: (-presence_count.get(f, 0), f))

    # Sort single-target features by the rank of the target they belong to
    def _single_sort_key(feat: str) -> tuple:
        for t in targets:
            if feat in target_feature_sets.get(t, set()):
                return (target_rank[t], feat)
        return (len(targets), feat)

    single_target_features.sort(key=_single_sort_key)

    ordered_features = multi_target_features + single_target_features
    if ordered_features:
        ordered_indices = [feature_to_idx[f] for f in ordered_features]
        matrix = matrix[:, ordered_indices]
        all_features = ordered_features

    # Recompute indices after reordering
    multi_idx = list(range(len(multi_target_features)))
    single_idx = list(range(len(multi_target_features), len(all_features)))

    # Total row
    total_row = matrix.sum(axis=0)

    # Font sizing (consistent with existing heatmap)
    heat_font = 8

    # Annotation helper: show integer for >=1, blank for 0
    def _annotate_inclusion_cells(ax_obj, values: np.ndarray, fontsize: int) -> None:
        for row_i in range(values.shape[0]):
            for col_j in range(values.shape[1]):
                val = values[row_i, col_j]
                if not np.isfinite(val) or val < 0.5:
                    continue
                ax_obj.text(
                    col_j + 0.5, row_i + 0.5,
                    str(int(val)),
                    ha="center", va="center",
                    color="black", fontsize=fontsize,
                    clip_on=True,
                )

    n_targets = len(targets)
    heat_h = max(4, (n_targets + 1) * 0.38)

    if multi_idx and single_idx:
        yticklabels_with_total = yticklabels + ["Total"]
        left_block = np.vstack([matrix[:, multi_idx],
                                total_row[multi_idx][None, :]])
        right_block = np.vstack([matrix[:, single_idx],
                                 total_row[single_idx][None, :]])
        sep_col = np.full((left_block.shape[0], 1), np.nan)
        combined_matrix = np.hstack([left_block, sep_col, right_block])
        sep_pos = left_block.shape[1]
        xticklabels_with_sep = multi_target_features + [""] + single_target_features

        n_total_cols = combined_matrix.shape[1]
        heat_w = max(14, n_total_cols * 0.85)
        fig, ax = plt.subplots(figsize=(heat_w, heat_h), constrained_layout=True)

        sns.heatmap(
            combined_matrix, ax=ax,
            cmap="YlGn", vmin=0, vmax=max(n_targets, 1),
            annot=False,
            cbar_kws={"label": "Number of targets including predictor", "pad": 0.01},
            xticklabels=xticklabels_with_sep,
            yticklabels=yticklabels_with_total,
            linewidths=0.5, linecolor="#eeeeee", square=False,
        )
        _annotate_inclusion_cells(ax, combined_matrix, heat_font)

        ax.add_patch(plt.Rectangle(
            (sep_pos, 0), 1, combined_matrix.shape[0],
            facecolor=ax.get_facecolor(), edgecolor="none", zorder=3,
        ))
        ax.set_xticklabels(xticklabels_with_sep, rotation=45, ha="right", fontsize=heat_font)
        ax.set_yticklabels(
            [textwrap.fill(lbl, 20) for lbl in yticklabels_with_total],
            rotation=0, fontsize=heat_font,
        )
    else:
        matrix_with_total = np.vstack([matrix, total_row[None, :]])
        yticklabels_with_total = yticklabels + ["Total"]
        n_total_features = max(len(all_features), 1)
        fig, ax = plt.subplots(
            figsize=(max(13, n_total_features * 0.85), heat_h),
            constrained_layout=True,
        )
        sns.heatmap(
            matrix_with_total, ax=ax,
            cmap="YlGn", vmin=0, vmax=max(n_targets, 1),
            annot=False,
            cbar_kws={"label": "Number of targets including predictor", "pad": 0.01},
            xticklabels=all_features,
            yticklabels=yticklabels_with_total,
            linewidths=0.5, linecolor="#eeeeee", square=False,
        )
        _annotate_inclusion_cells(ax, matrix_with_total, heat_font)
        ax.set_xticklabels(all_features, rotation=45, ha="right", fontsize=heat_font)
        ax.set_yticklabels(
            [textwrap.fill(lbl, 20) for lbl in yticklabels_with_total],
            rotation=0, fontsize=heat_font,
        )

    ax.set_xlabel("Predictor", fontsize=heat_font)
    ax.set_ylabel("Target", fontsize=heat_font)
    ax.set_title("Best Model Feature Inclusion", fontsize=heat_font + 2)

    summaries_dir = (data_root / "summaries").resolve()
    namespace = str(sweep_namespace).strip() or "feature_sweeps"
    if namespace != "feature_sweeps":
        summaries_dir = (summaries_dir / namespace).resolve()
    summaries_dir.mkdir(parents=True, exist_ok=True)
    plot_path = summaries_dir / "multi_target_feature_inclusion_heatmap.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def post(plans: list[DatasetPlan], args: argparse.Namespace) -> int:
    workspace_root = Path(__file__).resolve().parent.parent
    data_root_arg = args.path if getattr(args, "path", None) else args.data_root
    data_root = Path(data_root_arg)
    run_rolling_cv = bool(getattr(args, "run_rolling_cv", False))
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()

    print(
        "[INFO] Postprocess uses existing stored split files and does not "
        "retroactively rebalance final-top-k train/test splits."
    )

    sweep_results: dict[str, dict[int, dict[str, tuple[float, int]]]] = {}
    importance_sources_used: set[str] = set()
    datasets_with_outputs = 0
    best_model_performance = []
    target_order_by_skill: list[str] = []

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

        output_dir = _forecast_sweeps_dir(plan.dataset_dir)
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

            try:
                final_df = pd.read_csv(metrics_csv)
                if not final_df.empty:
                    summary_plot = _plot_final_metrics_comparison(final_df, output_dir)
                    print(f"[INFO] Rebuilt final metrics comparison plot: {summary_plot}")
            except Exception as exc:
                print(f"[WARN] Could not rebuild final metrics comparison plot for {plan.dataset_dir.name}: {exc}")

        print(f"\n[INFO] Rebuilding saved outputs for {plan.dataset_dir.name}: rows={row_counts}")
        wrote_any = False
        include_row_count_in_plot_names = len(row_counts) > 1
        for row_count in row_counts:
            written = _regenerate_saved_outputs_for_row(
                dataset_dir=plan.dataset_dir,
                target_name=target_name,
                row_count=row_count,
                keep_search_plots=bool(args.keep_search_plots),
                include_row_count_in_plot_names=include_row_count_in_plot_names,
            )

            feature_sensitivities, _, _, importance_source = _load_feature_stats_artifacts_with_source(
                dataset_dir=plan.dataset_dir,
                row_count=row_count,
            )
            if feature_sensitivities:
                if target_name not in sweep_results:
                    sweep_results[target_name] = {}
                sweep_results[target_name][row_count] = feature_sensitivities
                importance_sources_used.add(str(importance_source))
                print(
                    f"[INFO] Loaded feature importance source for {plan.dataset_dir.name} "
                    f"r{int(row_count):03d}: {importance_source}"
                )

            if written:
                wrote_any = True
                for label, path in written.items():
                    print(f"[INFO] Wrote {label}: {path}")
            else:
                print(
                    f"[WARN] Could not rebuild plots for {plan.dataset_dir.name} rows={row_count}; "
                    "missing native feature stats and Shapley fallback artifacts."
                )

        # Run rolling origin CV and collect best model performance for summary plot
        try:
            final_metrics_csv = _forecast_sweeps_dir(plan.dataset_dir) / "feature_sweep_final_metrics.csv"
            if final_metrics_csv.exists():
                df = pd.read_csv(final_metrics_csv)
                if not df.empty:
                    # Select best row across all models/subsets by R2
                    valid_r2 = _exclude_baseline_metric_rows(
                        _filter_min_valid_independent(_filter_valid_rows(df), min_required=MIN_REQUIRED_VALID_INDEPENDENT)
                    )
                    if valid_r2.empty:
                        print(
                            f"[WARN] No valid r2 rows meeting min valid independent test samples "
                            f"({MIN_REQUIRED_VALID_INDEPENDENT}) for {plan.dataset_dir.name}; skipping rolling CV."
                        )
                    else:
                        best_row = valid_r2.loc[valid_r2['r2'].idxmax()]

                        rolling_cv_r2 = rolling_cv_r2_median = rolling_cv_r2_last50 = rolling_cv_r2_pooled = float('nan')
                        rolling_cv_rmse = rolling_cv_mae = rolling_cv_n_folds = float('nan')
                        stat_evidence: dict = {}

                        cv_summary_path = None
                        if run_rolling_cv:
                            # Optional execution path, disabled by default due runtime cost.
                            print(f"[INFO] Running rolling origin CV for {plan.dataset_dir.name}")
                            try:
                                cv_summary_path = _run_rolling_origin_cv(plan=plan, final_metrics_csv=final_metrics_csv)
                            except Exception as exc:
                                print(f"[WARN] Rolling origin CV failed for {plan.dataset_dir.name}: {exc}")
                                traceback.print_exc()
                        else:
                            print(f"[INFO] Rolling origin CV execution skipped for {plan.dataset_dir.name} (disabled).")

                        # Ensure the variant's evaluation_summary.csv has baseline rows.
                        try:
                            _ensure_k01_baselines(plan=plan, final_metrics_csv=final_metrics_csv)
                        except Exception as exc:
                            print(f"[WARN] Failed to ensure k01 baselines for {plan.dataset_dir.name}: {exc}")
                            traceback.print_exc()

                        # Write dataset-level evaluation_summary.csv (best k01 model + baselines)
                        try:
                            _write_dataset_evaluation_summary(plan=plan, final_metrics_csv=final_metrics_csv)
                        except Exception as exc:
                            print(f"[WARN] Failed to write dataset evaluation summary for {plan.dataset_dir.name}: {exc}")
                            traceback.print_exc()

                        # Compute MLR baseline and append to final_metrics CSV
                        try:
                            _append_mlr_to_final_metrics(plan, final_metrics_csv, best_row, args)
                            # Re-generate final metrics comparison plot now that MLR row is included
                            try:
                                _updated_df = pd.read_csv(final_metrics_csv)
                                if not _updated_df.empty:
                                    _plot_final_metrics_comparison(_updated_df, output_dir)
                            except Exception:
                                pass
                        except Exception as exc:
                            print(f"[WARN] MLR computation failed for {plan.dataset_dir.name}: {exc}")
                            traceback.print_exc()

                        # Re-run best model evaluation context and compute statistical evidence
                        try:
                            stat_evidence = _compute_statistical_evidence(plan, best_row, args)
                            status = str(stat_evidence.get("evidence_status", ""))
                            if status == "ok":
                                print(f"[INFO] Statistical evidence computed for {plan.dataset_dir.name}")
                            else:
                                print(f"[WARN] Statistical evidence incomplete for {plan.dataset_dir.name}: {status}")
                        except Exception as exc:
                            print(f"[WARN] Statistical evidence failed for {plan.dataset_dir.name}: {exc}")
                            traceback.print_exc()

                        # Read/write rolling CV metrics only when explicitly requested.
                        if run_rolling_cv:
                            if cv_summary_path is not None and cv_summary_path.exists():
                                try:
                                    df_cv = pd.read_csv(cv_summary_path)
                                    fold_rows = df_cv[df_cv['fold'].astype(str) != 'mean']
                                    rolling_cv_n_folds = float(len(fold_rows))
                                    (
                                        rolling_cv_r2,
                                        rolling_cv_r2_median,
                                        rolling_cv_r2_last50,
                                        rolling_cv_r2_pooled,
                                    ) = _compute_rolling_cv_r2_stats(df_cv)
                                    mean_rows = df_cv[df_cv['fold'].astype(str) == 'mean']
                                    if not mean_rows.empty:
                                        agg = mean_rows.iloc[0]
                                        rolling_cv_rmse = _safe_float(agg.get('rmse'))
                                        rolling_cv_mae = _safe_float(agg.get('mae'))
                                        print(
                                            f"[INFO] Rolling CV: r2_mean={rolling_cv_r2:.4f}, "
                                            f"r2_median={rolling_cv_r2_median:.4f}, "
                                            f"r2_last50={rolling_cv_r2_last50:.4f}, "
                                            f"r2_pooled={rolling_cv_r2_pooled:.4f}, "
                                            f"rmse={rolling_cv_rmse:.4f}, mae={rolling_cv_mae:.4f}, "
                                            f"n_folds={int(rolling_cv_n_folds)}"
                                        )
                                    else:
                                        print(f"[WARN] rolling_origin_summary.csv has no 'mean' row for {plan.dataset_dir.name}")
                                except Exception as exc:
                                    print(f"[WARN] Could not read rolling CV results for {plan.dataset_dir.name}: {exc}")
                                    traceback.print_exc()
                            else:
                                print(f"[WARN] Rolling origin CV summary not available for {plan.dataset_dir.name}")

                            try:
                                df_metrics = pd.read_csv(final_metrics_csv)
                                for col in [
                                    'rolling_cv_r2',
                                    'rolling_cv_r2_median',
                                    'rolling_cv_r2_last50',
                                    'rolling_cv_r2_pooled',
                                    'rolling_cv_rmse',
                                    'rolling_cv_mae',
                                ]:
                                    if col not in df_metrics.columns:
                                        df_metrics[col] = float('nan')
                                row_mask = (
                                    (df_metrics['feature_tag'] == best_row['feature_tag'])
                                    & (df_metrics['row_count'] == int(best_row['row_count']))
                                    & (df_metrics['model'] == best_row['model'])
                                )
                                if row_mask.any():
                                    df_metrics.loc[row_mask, 'rolling_cv_r2'] = rolling_cv_r2
                                    df_metrics.loc[row_mask, 'rolling_cv_r2_median'] = rolling_cv_r2_median
                                    df_metrics.loc[row_mask, 'rolling_cv_r2_last50'] = rolling_cv_r2_last50
                                    df_metrics.loc[row_mask, 'rolling_cv_r2_pooled'] = rolling_cv_r2_pooled
                                    df_metrics.loc[row_mask, 'rolling_cv_rmse'] = rolling_cv_rmse
                                    df_metrics.loc[row_mask, 'rolling_cv_mae'] = rolling_cv_mae
                                    print(f"[INFO] Wrote rolling CV results to feature_sweep_final_metrics.csv for {plan.dataset_dir.name}")
                                else:
                                    print(f"[WARN] Could not find matching row for rolling CV update in feature_sweep_final_metrics.csv for {plan.dataset_dir.name}")
                                df_metrics.to_csv(final_metrics_csv, index=False)
                            except Exception as exc:
                                print(f"[WARN] Could not write rolling CV results to feature_sweep_final_metrics.csv for {plan.dataset_dir.name}: {exc}")

                        # Re-read updated metrics and append best model performance entry
                        try:
                            valid_r2_2 = _exclude_baseline_metric_rows(
                                _filter_min_valid_independent(
                                    _filter_valid_rows(pd.read_csv(final_metrics_csv)),
                                    min_required=MIN_REQUIRED_VALID_INDEPENDENT,
                                )
                            )
                            if not valid_r2_2.empty:
                                best_updated = valid_r2_2.loc[valid_r2_2['r2'].idxmax()]
                                cv_r2_to_write = _safe_float(best_updated.get('rolling_cv_r2')) if run_rolling_cv else rolling_cv_r2
                                cv_r2_median_to_write = _safe_float(best_updated.get('rolling_cv_r2_median')) if run_rolling_cv else rolling_cv_r2_median
                                cv_r2_last50_to_write = _safe_float(best_updated.get('rolling_cv_r2_last50')) if run_rolling_cv else rolling_cv_r2_last50
                                cv_r2_pooled_to_write = _safe_float(best_updated.get('rolling_cv_r2_pooled')) if run_rolling_cv else rolling_cv_r2_pooled
                                cv_rmse_to_write = _safe_float(best_updated.get('rolling_cv_rmse')) if run_rolling_cv else rolling_cv_rmse
                                cv_mae_to_write = _safe_float(best_updated.get('rolling_cv_mae')) if run_rolling_cv else rolling_cv_mae
                                best_model_performance.append(_build_perf_entry(
                                    plan.dataset_dir.name, best_updated,
                                    rolling_cv_r2=cv_r2_to_write,
                                    rolling_cv_r2_median=cv_r2_median_to_write,
                                    rolling_cv_r2_last50=cv_r2_last50_to_write,
                                    rolling_cv_r2_pooled=cv_r2_pooled_to_write,
                                    rolling_cv_rmse=cv_rmse_to_write,
                                    rolling_cv_mae=cv_mae_to_write,
                                    rolling_cv_n_folds=rolling_cv_n_folds,
                                    extra_metrics=stat_evidence,
                                ))
                            else:
                                # Fallback: use pre-update values
                                best_model_performance.append(_build_perf_entry(
                                    plan.dataset_dir.name, best_row,
                                    rolling_cv_r2=rolling_cv_r2,
                                    rolling_cv_r2_median=rolling_cv_r2_median,
                                    rolling_cv_r2_last50=rolling_cv_r2_last50,
                                    rolling_cv_r2_pooled=rolling_cv_r2_pooled,
                                    rolling_cv_rmse=rolling_cv_rmse,
                                    rolling_cv_mae=rolling_cv_mae,
                                    rolling_cv_n_folds=rolling_cv_n_folds,
                                    extra_metrics=stat_evidence,
                                ))
                        except Exception as exc:
                            print(f"[WARN] Could not re-read updated metrics for {plan.dataset_dir.name}: {exc}")
                            best_model_performance.append(_build_perf_entry(
                                plan.dataset_dir.name, best_row,
                                rolling_cv_r2=rolling_cv_r2,
                                rolling_cv_r2_median=rolling_cv_r2_median,
                                rolling_cv_r2_last50=rolling_cv_r2_last50,
                                rolling_cv_r2_pooled=rolling_cv_r2_pooled,
                                rolling_cv_rmse=rolling_cv_rmse,
                                rolling_cv_mae=rolling_cv_mae,
                                rolling_cv_n_folds=rolling_cv_n_folds,
                                extra_metrics=stat_evidence,
                            ))
        except Exception as e:
            print(f"[WARN] Could not process best model performance for {plan.dataset_dir.name}: {e}")
            traceback.print_exc()

        if wrote_any:
            datasets_with_outputs += 1

    # Generate summary_best_model_performance.png (nRMSE, R2, Rolling CV R2)
    try:
        if best_model_performance:
            # --- Augment with baseline stats ---
            summaries_dir = _resolve_summaries_dir(
                data_root=data_root,
                sweep_namespace=str(getattr(args, "sweep_namespace", "feature_sweeps")),
            )
            summaries_dir.mkdir(parents=True, exist_ok=True)
            combined_dir, individual_dir, evaluation_dir = _resolve_summary_plot_dirs(summaries_dir)
            for entry in best_model_performance:
                dataset = entry['dataset']
                # Standard location for a full-dataset evaluation summary with baselines
                eval_csv = os.path.join(data_root, dataset, 'evaluation_summary.csv')
                baseline_stats = {name: {} for name in BASELINE_ORDER}
                if os.path.exists(eval_csv):
                    try:
                        df_eval = pd.read_csv(eval_csv)
                        for kind in baseline_stats.keys():
                            row = df_eval[df_eval['label'].str.lower().str.contains(kind)].iloc[0] if not df_eval[df_eval['label'].str.lower().str.contains(kind)].empty else None
                            if row is not None:
                                for stat in ['mae', 'rmse', 'r2', 'pearson_r']:
                                    baseline_stats[kind][stat] = row.get(stat, np.nan)
                            else:
                                for stat in ['mae', 'rmse', 'r2', 'pearson_r']:
                                    baseline_stats[kind][stat] = np.nan
                    except Exception as e:
                        print(f"[WARN] Could not read baseline stats for {dataset}: {e}")
                        for kind in baseline_stats.keys():
                            for stat in ['mae', 'rmse', 'r2', 'pearson_r']:
                                baseline_stats[kind][stat] = np.nan
                else:
                    for kind in baseline_stats.keys():
                        for stat in ['mae', 'rmse', 'r2', 'pearson_r']:
                            baseline_stats[kind][stat] = np.nan
                for kind in baseline_stats.keys():
                    for stat in ['mae', 'rmse', 'r2', 'pearson_r']:
                        entry[f'{kind}_{stat}'] = baseline_stats[kind][stat]

            perf_df = pd.DataFrame(best_model_performance)
            perf_df = perf_df.sort_values('r2', ascending=False)
            valid_src = pd.to_numeric(perf_df.get('n_test_valid_source', np.nan), errors='coerce')
            evidence_ok = perf_df.get('evidence_status', '').astype(str).str.lower().eq('ok') if 'evidence_status' in perf_df.columns else pd.Series(False, index=perf_df.index)
            perf_df['compliance_status'] = np.where(
                np.isfinite(valid_src) & (valid_src >= float(MIN_REQUIRED_VALID_INDEPENDENT)) & evidence_ok,
                'ok',
                'failed',
            )
            perf_df['compliance_reason'] = np.where(
                perf_df['compliance_status'].eq('ok'),
                '',
                np.where(
                    ~np.isfinite(valid_src),
                    'missing_n_test_valid_source',
                    np.where(
                        valid_src < float(MIN_REQUIRED_VALID_INDEPENDENT),
                        f'n_test_valid_source_below_{MIN_REQUIRED_VALID_INDEPENDENT}',
                        perf_df.get('evidence_status', 'evidence_not_ok').astype(str) if 'evidence_status' in perf_df.columns else 'evidence_not_ok',
                    ),
                ),
            )
            # Compute skill vs best baseline for target ordering in heatmaps.
            _baseline_rmse_cols = [
                perf_df.get(f'{name}_rmse', pd.Series(dtype=float)).replace(0, np.nan)
                for name in BASELINE_ORDER
            ]
            _best_baseline_rmse = pd.concat(_baseline_rmse_cols, axis=1).min(axis=1)
            perf_df['skill_vs_best_baseline'] = 1.0 - perf_df['rmse'] / _best_baseline_rmse
            # Order targets by increasing skill (worst first → best last).
            _skill_sorted = perf_df.sort_values('skill_vs_best_baseline', ascending=True, na_position='first')
            target_order_by_skill = []
            seen_targets = set()
            for dataset_name in _skill_sorted['dataset'].astype(str).tolist():
                tgt = _derive_target_name(dataset_name, args.dataset_prefix)
                if tgt not in seen_targets:
                    seen_targets.add(tgt)
                    target_order_by_skill.append(tgt)
            summary_csv = summaries_dir / "summary_best_model_performance.csv"
            perf_df.to_csv(summary_csv, index=False)
            print(f"[INFO] Wrote summary CSV: {summary_csv}")

            x = np.arange(len(perf_df))
            width = 0.20
            labels = perf_df['dataset']
            # Use the actual ML model type(s) as the label; fall back to 'Model' if not recorded.
            if 'model' in perf_df.columns:
                _model_types = perf_df['model'].dropna().tolist()
                _unique = list(dict.fromkeys(_model_types))  # preserve order, deduplicate
                model_series_label = _unique[0] if len(_unique) == 1 else (
                    max(set(_model_types), key=_model_types.count) if _model_types else 'Model'
                )
            else:
                model_series_label = 'Model'
            methods = [model_series_label] + [BASELINE_PLOT_LABELS[name] for name in BASELINE_ORDER]
            colors = ['tab:blue'] + [BASELINE_PLOT_COLORS[name] for name in BASELINE_ORDER]

            std_target_col = perf_df['std_target'].replace(0, np.nan)
            nrmse_data = [
                perf_df['nrmse'],
                perf_df['naive_rmse'] / std_target_col,
                perf_df['seasonal_rmse'] / std_target_col,
                perf_df['linear_rmse'] / std_target_col,
            ]
            r2_data = [
                perf_df['r2'],
                perf_df['naive_r2'],
                perf_df['seasonal_r2'],
                perf_df['linear_r2'],
            ]
            # Skill score: 1 - (model_rmse / baseline_rmse); positive = better than baseline
            skill_naive = 1.0 - perf_df['rmse'] / perf_df['naive_rmse'].replace(0, np.nan)
            skill_seasonal = 1.0 - perf_df['rmse'] / perf_df['seasonal_rmse'].replace(0, np.nan)
            skill_linear = 1.0 - perf_df['rmse'] / perf_df['linear_rmse'].replace(0, np.nan)
            skill_data = [skill_naive, skill_seasonal, skill_linear]
            skill_methods = [
                'Compared with Naive Baseline',
                'Compared with Seasonal Baseline',
                'Compared with Linear Baseline',
            ]
            skill_colors = [BASELINE_PLOT_COLORS['naive'], BASELINE_PLOT_COLORS['seasonal'], BASELINE_PLOT_COLORS['linear']]

            # --- Combined 3-panel figure (no title): Skill, nRMSE, R2 ---
            fig, (ax_skill_combo, ax_nrmse_combo, ax_r2_combo) = plt.subplots(
                3, 1, figsize=(max(12, len(perf_df)*0.8), 13), sharex=True
            )
            _draw_bar_group(
                ax_skill_combo, x, width, skill_data, skill_colors, skill_methods, '.2f',
                center_offset=0.5
            )
            ax_skill_combo.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax_skill_combo.set_ylabel('Skill Score')
            ax_skill_combo.grid(axis='y', alpha=0.3)
            ax_skill_combo.legend()
            _draw_bar_group(ax_nrmse_combo, x, width, nrmse_data, colors, methods, '.2e')
            ax_nrmse_combo.set_ylabel('nRMSE')
            ax_nrmse_combo.grid(axis='y', alpha=0.3)
            ax_nrmse_combo.legend()
            r2_bars_combo = _draw_bar_group(
                ax_r2_combo, x, width, r2_data, colors, methods, '.2f', annotate=False
            )
            ax_r2_combo.set_ylabel('Coefficient of Determination')
            ax_r2_combo.set_ylim(-0.1, 1.0)
            for bars in r2_bars_combo:
                _annotate_bars_within_ylim(ax_r2_combo, bars, '.2f')
            ax_r2_combo.grid(axis='y', alpha=0.3)
            ax_r2_combo.legend()
            ax_r2_combo.set_xticks(x)
            ax_r2_combo.set_xticklabels(labels, rotation=45, ha='right')
            plt.tight_layout()
            plot_path = combined_dir / "summary_best_model_performance.png"
            fig.savefig(plot_path, dpi=180, bbox_inches='tight')
            plt.close(fig)
            print(f"[INFO] Wrote summary_best_model_performance.png to {plot_path}")

            # --- Standalone nRMSE subplot ---
            fig_nrmse, ax_nrmse = plt.subplots(figsize=(max(10, len(perf_df)*0.7), 5))
            _draw_bar_group(ax_nrmse, x, width, nrmse_data, colors, methods, '.2e')
            ax_nrmse.set_ylabel('nRMSE')
            ax_nrmse.set_xticks(x)
            ax_nrmse.set_xticklabels(labels, rotation=45, ha='right')
            ax_nrmse.grid(axis='y', alpha=0.3)
            ax_nrmse.legend()
            plt.tight_layout()
            nrmse_path = individual_dir / "summary_best_model_nrmse.png"
            fig_nrmse.savefig(nrmse_path, dpi=300, bbox_inches='tight')
            plt.close(fig_nrmse)
            print(f"[INFO] Wrote nRMSE subplot: {nrmse_path}")

            # --- Standalone R2 subplot ---
            fig_r2, ax_r2 = plt.subplots(figsize=(max(10, len(perf_df)*0.7), 5))
            r2_bars = _draw_bar_group(ax_r2, x, width, r2_data, colors, methods, '.2f', annotate=False)
            ax_r2.set_ylabel('Coefficient of Determination')
            ax_r2.set_ylim(-0.1, 1.0)
            for bars in r2_bars:
                _annotate_bars_within_ylim(ax_r2, bars, '.2f')
            ax_r2.set_xticks(x)
            ax_r2.set_xticklabels(labels, rotation=45, ha='right')
            ax_r2.grid(axis='y', alpha=0.3)
            ax_r2.legend()
            plt.tight_layout()
            r2_path = individual_dir / "summary_best_model_r2.png"
            fig_r2.savefig(r2_path, dpi=300, bbox_inches='tight')
            plt.close(fig_r2)
            print(f"[INFO] Wrote R2 subplot: {r2_path}")

            # --- Standalone skill score subplot ---
            fig_skill, ax_skill = plt.subplots(figsize=(max(10, len(perf_df)*0.7), 5))
            _draw_bar_group(ax_skill, x, width, skill_data, skill_colors, skill_methods, '.2f', center_offset=0.5)
            ax_skill.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax_skill.set_ylabel('Skill Score')
            ax_skill.set_xticks(x)
            ax_skill.set_xticklabels(labels, rotation=45, ha='right')
            ax_skill.grid(axis='y', alpha=0.3)
            ax_skill.legend()
            plt.tight_layout()
            skill_path = individual_dir / "summary_best_model_skill.png"
            fig_skill.savefig(skill_path, dpi=300, bbox_inches='tight')
            plt.close(fig_skill)
            print(f"[INFO] Wrote skill score subplot: {skill_path}")

            # --- Confidence / uncertainty subplot ---
            n_perf = len(perf_df)

            def _perf_col(name: str) -> pd.Series:
                if name in perf_df.columns:
                    return pd.to_numeric(perf_df[name], errors="coerce")
                return pd.Series([float("nan")] * n_perf)

            baseline_prob_cols = [_perf_col(f"bootstrap_prob_skill_gt0_vs_{name}") for name in BASELINE_ORDER]
            baseline_lcb_cols = [_perf_col(f"lcb95_skill_vs_{name}") for name in BASELINE_ORDER]
            overall_score = _perf_col("evidence_score_overall_min")
            model_picp = _perf_col("model_picp")
            naive_picp = _perf_col("naive_picp")
            seasonal_picp = _perf_col("seasonal_picp")
            linear_picp = _perf_col("linear_picp")
            nominal_cov = _perf_col("model_nominal_coverage")
            fig_conf, conf_axes = plt.subplots(
                4, 1, figsize=(max(12, len(perf_df) * 0.8), 15), sharex=True
            )
            # Order: component diagnostics first, overall summaries last.
            _draw_bar_group(
                conf_axes[0], x, width,
                baseline_prob_cols,
                [BASELINE_PLOT_COLORS[name] for name in BASELINE_ORDER],
                [f"Prob(skill>0) vs {BASELINE_PLOT_LABELS[name]}" for name in BASELINE_ORDER],
                '.2f',
            )
            conf_axes[0].axhline(float(args.evidence_min_prob), color='black', linewidth=0.8, linestyle='--')
            conf_axes[0].set_ylabel('Bootstrap Probability\n(Model Skill > 0)')
            conf_axes[0].set_ylim(0.0, 1.05)
            conf_axes[0].grid(axis='y', alpha=0.3)
            conf_axes[0].legend()

            _draw_bar_group(
                conf_axes[1], x, width,
                baseline_lcb_cols,
                [BASELINE_PLOT_COLORS[name] for name in BASELINE_ORDER],
                [f"95% Lower Confidence Bound of Skill\nCompared with {BASELINE_PLOT_LABELS[name]} Baseline" for name in BASELINE_ORDER],
                '.2f',
            )
            conf_axes[1].axhline(0.0, color='black', linewidth=0.8, linestyle='--')
            conf_axes[1].set_ylabel('Skill Lower Confidence Bound')
            conf_axes[1].grid(axis='y', alpha=0.3)
            conf_axes[1].legend()

            _draw_bar_group(
                conf_axes[2], x, width,
                [model_picp, naive_picp, seasonal_picp, linear_picp],
                ['tab:blue', BASELINE_PLOT_COLORS['naive'], BASELINE_PLOT_COLORS['seasonal'], BASELINE_PLOT_COLORS['linear']],
                [model_series_label, 'Naive', 'Seasonal', 'Linear'],
                '.2f',
            )
            nom_arr = nominal_cov.to_numpy(dtype=float) if hasattr(nominal_cov, "to_numpy") else np.array(nominal_cov, dtype=float)
            if np.isfinite(nom_arr).any():
                conf_axes[2].axhline(float(np.nanmean(nom_arr)), color='black', linewidth=0.8, linestyle='--', label='Nominal Coverage')
            conf_axes[2].set_ylabel('Prediction Interval Coverage\nProbability (Proxy)')
            conf_axes[2].set_ylim(0.0, 1.05)
            conf_axes[2].grid(axis='y', alpha=0.3)
            conf_axes[2].legend()

            bars_score = conf_axes[3].bar(x, overall_score, width=0.5, color='tab:blue')
            _annotate_bars_within_ylim(conf_axes[3], bars_score, '.0f')
            conf_axes[3].set_ylabel('Overall Evidence Score\n(Minimum Across Baselines)')
            conf_axes[3].grid(axis='y', alpha=0.3)
            conf_axes[3].set_xticks(x)
            conf_axes[3].set_xticklabels(labels, rotation=45, ha='right')
            _finalize_stacked_figure(fig_conf, conf_axes, left=0.30, hspace=0.48)
            conf_path = combined_dir / "summary_best_model_confidence.png"
            fig_conf.savefig(conf_path, dpi=300, bbox_inches='tight')

            def _conf_panel_prob(ax):
                _draw_bar_group(
                    ax, x, width,
                    baseline_prob_cols,
                    [BASELINE_PLOT_COLORS[name] for name in BASELINE_ORDER],
                    [f"Prob(skill>0) vs {BASELINE_PLOT_LABELS[name]}" for name in BASELINE_ORDER],
                    '.2f',
                )
                ax.axhline(float(args.evidence_min_prob), color='black', linewidth=0.8, linestyle='--')
                ax.set_ylabel('Bootstrap Probability\n(Model Skill > 0)')
                ax.set_ylim(0.0, 1.05)
                ax.grid(axis='y', alpha=0.3)
                ax.legend()

            def _conf_panel_lcb(ax):
                _draw_bar_group(
                    ax, x, width,
                    baseline_lcb_cols,
                    [BASELINE_PLOT_COLORS[name] for name in BASELINE_ORDER],
                    [f"95% Lower Confidence Bound of Skill\nCompared with {BASELINE_PLOT_LABELS[name]} Baseline" for name in BASELINE_ORDER],
                    '.2f',
                )
                ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
                ax.set_ylabel('Skill Lower Confidence Bound')
                ax.grid(axis='y', alpha=0.3)
                ax.legend()

            def _conf_panel_picp(ax):
                _draw_bar_group(
                    ax, x, width,
                    [model_picp, naive_picp, seasonal_picp, linear_picp],
                    ['tab:blue', BASELINE_PLOT_COLORS['naive'], BASELINE_PLOT_COLORS['seasonal'], BASELINE_PLOT_COLORS['linear']],
                    [model_series_label, 'Naive', 'Seasonal', 'Linear'],
                    '.2f',
                )
                nom_arr = nominal_cov.to_numpy(dtype=float) if hasattr(nominal_cov, "to_numpy") else np.array(nominal_cov, dtype=float)
                if np.isfinite(nom_arr).any():
                    ax.axhline(float(np.nanmean(nom_arr)), color='black', linewidth=0.8, linestyle='--', label='Nominal Coverage')
                ax.set_ylabel('Prediction Interval Coverage\nProbability (Proxy)')
                ax.set_ylim(0.0, 1.05)
                ax.grid(axis='y', alpha=0.3)
                ax.legend()

            def _conf_panel_score(ax):
                bars = ax.bar(x, overall_score, width=0.5, color='tab:blue')
                _annotate_bars_within_ylim(ax, bars, '.0f')
                ax.set_ylabel('Overall Evidence Score\n(Minimum Across Baselines)')
                ax.grid(axis='y', alpha=0.3)

            conf_panels = _save_individual_panels_from_builders(
                out_dir=individual_dir,
                base_name="summary_best_model_confidence",
                labels=labels,
                builders=[_conf_panel_prob, _conf_panel_lcb, _conf_panel_picp, _conf_panel_score],
                figsize=(max(11, len(perf_df) * 0.85), 6.2),
                dpi=300,
                left=0.36,
                bottom=0.30,
            )
            plt.close(fig_conf)
            print(f"[INFO] Wrote confidence subplot: {conf_path}")
            print(f"[INFO] Wrote {len(conf_panels)} confidence panel(s) to {individual_dir}")

            # --- Evidence diagnostics figures (tests, effects, intervals, gates) ---
            def _col(name: str) -> pd.Series:
                if name in perf_df.columns:
                    return pd.to_numeric(perf_df[name], errors='coerce')
                return pd.Series([float('nan')] * len(perf_df))

            baseline_colors = [BASELINE_PLOT_COLORS[name] for name in BASELINE_ORDER]
            baseline_methods = [f"{BASELINE_PLOT_LABELS[name]} Baseline" for name in BASELINE_ORDER]
            trio_colors = ['tab:blue'] + baseline_colors
            trio_methods = [model_series_label] + [BASELINE_PLOT_LABELS[name] for name in BASELINE_ORDER]

            def _baseline_panel(ax, cols: list[str], ylabel: str, fmt: str = '.2f',
                            hline: float | None = None, ylim: tuple[float, float] | None = None) -> None:
                _draw_bar_group(
                    ax, x, width,
                    [_col(c) for c in cols],
                    baseline_colors,
                    baseline_methods,
                    fmt,
                )
                if hline is not None and np.isfinite(hline):
                    ax.axhline(float(hline), color='black', linewidth=0.8, linestyle='--')
                if ylim is not None:
                    ax.set_ylim(float(ylim[0]), float(ylim[1]))
                ax.set_ylabel(ylabel)
                ax.grid(axis='y', alpha=0.3)
                ax.legend()

            # Statistical tests and adjusted significance (decision-first ordering).
            fig_tests, axes_tests = plt.subplots(9, 1, figsize=(max(12, len(perf_df) * 0.8), 28), sharex=True)
            _baseline_panel(axes_tests[0], ['dm_p_vs_naive', 'dm_p_vs_seasonal', 'dm_p_vs_linear'], 'Diebold-Mariano Test p-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[1], ['dm_q_vs_naive', 'dm_q_vs_seasonal', 'dm_q_vs_linear'], 'Diebold-Mariano Test\nFalse Discovery Rate Adjusted q-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[2], ['wilcoxon_p_vs_naive', 'wilcoxon_p_vs_seasonal', 'wilcoxon_p_vs_linear'], 'Wilcoxon Signed-Rank Test p-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[3], ['wilcoxon_q_vs_naive', 'wilcoxon_q_vs_seasonal', 'wilcoxon_q_vs_linear'], 'Wilcoxon Signed-Rank Test\nFalse Discovery Rate Adjusted q-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[4], ['sign_p_vs_naive', 'sign_p_vs_seasonal', 'sign_p_vs_linear'], 'Sign Test p-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[5], ['sign_q_vs_naive', 'sign_q_vs_seasonal', 'sign_q_vs_linear'], 'Sign Test\nFalse Discovery Rate Adjusted q-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[6], ['dm_stat_vs_naive', 'dm_stat_vs_seasonal', 'dm_stat_vs_linear'], 'Diebold-Mariano Test Statistic', '.2f', hline=0.0)
            _baseline_panel(axes_tests[7], ['wilcoxon_stat_vs_naive', 'wilcoxon_stat_vs_seasonal', 'wilcoxon_stat_vs_linear'], 'Wilcoxon Statistic', '.2f')
            _baseline_panel(axes_tests[8], ['sign_win_rate_vs_naive', 'sign_win_rate_vs_seasonal', 'sign_win_rate_vs_linear'], 'Sign Test Win Rate', '.2f', hline=0.5, ylim=(0.0, 1.05))
            axes_tests[0].set_title("Evidence Tests (p/q thresholds first, diagnostics after)")
            axes_tests[-1].set_xticks(x)
            axes_tests[-1].set_xticklabels(labels, rotation=45, ha='right')
            _finalize_stacked_figure(fig_tests, axes_tests, left=0.34, hspace=0.55)
            tests_path = combined_dir / "summary_evidence_tests.png"
            fig_tests.savefig(tests_path, dpi=300, bbox_inches='tight')

            test_specs = [
                (['dm_p_vs_naive', 'dm_p_vs_seasonal', 'dm_p_vs_linear'], 'Diebold-Mariano Test p-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['dm_q_vs_naive', 'dm_q_vs_seasonal', 'dm_q_vs_linear'], 'Diebold-Mariano Test\nFalse Discovery Rate Adjusted q-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['wilcoxon_p_vs_naive', 'wilcoxon_p_vs_seasonal', 'wilcoxon_p_vs_linear'], 'Wilcoxon Signed-Rank Test p-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['wilcoxon_q_vs_naive', 'wilcoxon_q_vs_seasonal', 'wilcoxon_q_vs_linear'], 'Wilcoxon Signed-Rank Test\nFalse Discovery Rate Adjusted q-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['sign_p_vs_naive', 'sign_p_vs_seasonal', 'sign_p_vs_linear'], 'Sign Test p-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['sign_q_vs_naive', 'sign_q_vs_seasonal', 'sign_q_vs_linear'], 'Sign Test\nFalse Discovery Rate Adjusted q-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['dm_stat_vs_naive', 'dm_stat_vs_seasonal', 'dm_stat_vs_linear'], 'Diebold-Mariano Test Statistic', '.2f', 0.0, None),
                (['wilcoxon_stat_vs_naive', 'wilcoxon_stat_vs_seasonal', 'wilcoxon_stat_vs_linear'], 'Wilcoxon Statistic', '.2f', None, None),
                (['sign_win_rate_vs_naive', 'sign_win_rate_vs_seasonal', 'sign_win_rate_vs_linear'], 'Sign Test Win Rate', '.2f', 0.5, (0.0, 1.05)),
            ]
            tests_builders = [
                (lambda cols=cols, ylabel=ylabel, fmt=fmt, hline=hline, ylim=ylim: (lambda ax: _baseline_panel(ax, cols, ylabel, fmt, hline=hline, ylim=ylim)))()
                for cols, ylabel, fmt, hline, ylim in test_specs
            ]
            tests_panels = _save_individual_panels_from_builders(
                out_dir=individual_dir,
                base_name="summary_evidence_tests",
                labels=labels,
                builders=tests_builders,
                figsize=(max(11, len(perf_df) * 0.85), 6.4),
                dpi=300,
                left=0.40,
                bottom=0.30,
            )
            plt.close(fig_tests)
            print(f"[INFO] Wrote evidence tests figure: {tests_path}")
            print(f"[INFO] Wrote {len(tests_panels)} evidence test panel(s) to {individual_dir}")

            # Effect size and bootstrap summaries (interpretation flow).
            fig_eff, axes_eff = plt.subplots(10, 1, figsize=(max(12, len(perf_df) * 0.8), 31), sharex=True)
            _baseline_panel(axes_eff[0], ['skill_vs_naive', 'skill_vs_seasonal', 'skill_vs_linear'], 'Skill (RMSE-Based)', '.2f', hline=0.0)
            _baseline_panel(axes_eff[1], ['bootstrap_skill_mean_vs_naive', 'bootstrap_skill_mean_vs_seasonal', 'bootstrap_skill_mean_vs_linear'], 'Bootstrap Skill Mean', '.2f', hline=0.0)
            _baseline_panel(axes_eff[2], ['lcb95_skill_vs_naive', 'lcb95_skill_vs_seasonal', 'lcb95_skill_vs_linear'], '95% Lower Confidence Bound of Skill', '.2f', hline=0.0)
            _baseline_panel(axes_eff[3], ['effect_median_ae_diff_vs_naive', 'effect_median_ae_diff_vs_seasonal', 'effect_median_ae_diff_vs_linear'], 'Median MAE Difference', '.2e', hline=0.0)
            _baseline_panel(axes_eff[4], ['effect_mean_ae_diff_vs_naive', 'effect_mean_ae_diff_vs_seasonal', 'effect_mean_ae_diff_vs_linear'], 'Mean MAE Difference', '.2e', hline=0.0)
            _baseline_panel(axes_eff[5], ['effect_cohen_d_ae_diff_vs_naive', 'effect_cohen_d_ae_diff_vs_seasonal', 'effect_cohen_d_ae_diff_vs_linear'], "Cohen's d for MAE Difference", '.2f', hline=0.0)
            _baseline_panel(axes_eff[6], ['bootstrap_rmse_diff_mean_vs_naive', 'bootstrap_rmse_diff_mean_vs_seasonal', 'bootstrap_rmse_diff_mean_vs_linear'], 'Bootstrap RMSE Difference Mean', '.2e', hline=0.0)
            _baseline_panel(axes_eff[7], ['bootstrap_rmse_diff_ci05_vs_naive', 'bootstrap_rmse_diff_ci05_vs_seasonal', 'bootstrap_rmse_diff_ci05_vs_linear'], 'Bootstrap RMSE Difference\n5th Percentile', '.2e', hline=0.0)
            _baseline_panel(axes_eff[8], ['bootstrap_rmse_diff_ci95_vs_naive', 'bootstrap_rmse_diff_ci95_vs_seasonal', 'bootstrap_rmse_diff_ci95_vs_linear'], 'Bootstrap RMSE Difference\n95th Percentile', '.2e', hline=0.0)
            _baseline_panel(axes_eff[9], ['bootstrap_r2_diff_mean_vs_naive', 'bootstrap_r2_diff_mean_vs_seasonal', 'bootstrap_r2_diff_mean_vs_linear'], 'Bootstrap Coefficient of Determination Difference Mean', '.2f', hline=0.0)
            axes_eff[0].set_title("Evidence Effects (skill, effect sizes, then bootstrap deltas)")
            axes_eff[-1].set_xticks(x)
            axes_eff[-1].set_xticklabels(labels, rotation=45, ha='right')
            _finalize_stacked_figure(fig_eff, axes_eff, left=0.36, hspace=0.56)
            eff_path = combined_dir / "summary_evidence_effects.png"
            fig_eff.savefig(eff_path, dpi=300, bbox_inches='tight')

            eff_specs = [
                (['skill_vs_naive', 'skill_vs_seasonal', 'skill_vs_linear'], 'Skill (RMSE-Based)', '.2f', 0.0, None),
                (['bootstrap_skill_mean_vs_naive', 'bootstrap_skill_mean_vs_seasonal', 'bootstrap_skill_mean_vs_linear'], 'Bootstrap Skill Mean', '.2f', 0.0, None),
                (['lcb95_skill_vs_naive', 'lcb95_skill_vs_seasonal', 'lcb95_skill_vs_linear'], '95% Lower Confidence Bound of Skill', '.2f', 0.0, None),
                (['effect_median_ae_diff_vs_naive', 'effect_median_ae_diff_vs_seasonal', 'effect_median_ae_diff_vs_linear'], 'Median MAE Difference', '.2e', 0.0, None),
                (['effect_mean_ae_diff_vs_naive', 'effect_mean_ae_diff_vs_seasonal', 'effect_mean_ae_diff_vs_linear'], 'Mean MAE Difference', '.2e', 0.0, None),
                (['effect_cohen_d_ae_diff_vs_naive', 'effect_cohen_d_ae_diff_vs_seasonal', 'effect_cohen_d_ae_diff_vs_linear'], "Cohen's d for MAE Difference", '.2f', 0.0, None),
                (['bootstrap_rmse_diff_mean_vs_naive', 'bootstrap_rmse_diff_mean_vs_seasonal', 'bootstrap_rmse_diff_mean_vs_linear'], 'Bootstrap RMSE Difference Mean', '.2e', 0.0, None),
                (['bootstrap_rmse_diff_ci05_vs_naive', 'bootstrap_rmse_diff_ci05_vs_seasonal', 'bootstrap_rmse_diff_ci05_vs_linear'], 'Bootstrap RMSE Difference\n5th Percentile', '.2e', 0.0, None),
                (['bootstrap_rmse_diff_ci95_vs_naive', 'bootstrap_rmse_diff_ci95_vs_seasonal', 'bootstrap_rmse_diff_ci95_vs_linear'], 'Bootstrap RMSE Difference\n95th Percentile', '.2e', 0.0, None),
                (['bootstrap_r2_diff_mean_vs_naive', 'bootstrap_r2_diff_mean_vs_seasonal', 'bootstrap_r2_diff_mean_vs_linear'], 'Bootstrap Coefficient of Determination Difference Mean', '.2f', 0.0, None),
            ]
            eff_builders = [
                (lambda cols=cols, ylabel=ylabel, fmt=fmt, hline=hline, ylim=ylim: (lambda ax: _baseline_panel(ax, cols, ylabel, fmt, hline=hline, ylim=ylim)))()
                for cols, ylabel, fmt, hline, ylim in eff_specs
            ]
            eff_panels = _save_individual_panels_from_builders(
                out_dir=individual_dir,
                base_name="summary_evidence_effects",
                labels=labels,
                builders=eff_builders,
                figsize=(max(11, len(perf_df) * 0.85), 6.4),
                dpi=300,
                left=0.42,
                bottom=0.30,
            )
            plt.close(fig_eff)
            print(f"[INFO] Wrote evidence effects figure: {eff_path}")
            print(f"[INFO] Wrote {len(eff_panels)} evidence effect panel(s) to {individual_dir}")

            # Interval diagnostics and sample support
            fig_int, axes_int = plt.subplots(9, 1, figsize=(max(12, len(perf_df) * 0.8), 28), sharex=True)
            _draw_bar_group(
                axes_int[0], x, width,
                [_col('model_picp'), _col('naive_picp'), _col('seasonal_picp'), _col('linear_picp')],
                trio_colors,
                trio_methods,
                '.2f',
            )
            axes_int[0].axhline(1.0 - float(args.interval_alpha), color='black', linewidth=0.8, linestyle='--')
            axes_int[0].set_ylabel('Prediction Interval Coverage\nProbability')
            axes_int[0].set_ylim(0.0, 1.05)
            axes_int[0].grid(axis='y', alpha=0.3)
            axes_int[0].legend()
            _draw_bar_group(
                axes_int[1], x, width,
                [_col('model_coverage_deficit'), _col('naive_coverage_deficit'), _col('seasonal_coverage_deficit'), _col('linear_coverage_deficit')],
                trio_colors,
                trio_methods,
                '.3f',
            )
            axes_int[1].axhline(float(args.coverage_tolerance), color='black', linewidth=0.8, linestyle='--')
            axes_int[1].set_ylabel('Coverage Deficit')
            axes_int[1].grid(axis='y', alpha=0.3)
            axes_int[1].legend()
            _draw_bar_group(
                axes_int[2], x, width,
                [_col('model_nmpiw'), _col('naive_nmpiw'), _col('seasonal_nmpiw'), _col('linear_nmpiw')],
                trio_colors,
                trio_methods,
                '.2f',
            )
            axes_int[2].set_ylabel('Normalized Mean Prediction\nInterval Width')
            axes_int[2].grid(axis='y', alpha=0.3)
            axes_int[2].legend()
            _draw_bar_group(
                axes_int[3], x, width,
                [_col('model_interval_score'), _col('naive_interval_score'), _col('seasonal_interval_score'), _col('linear_interval_score')],
                trio_colors,
                trio_methods,
                '.2e',
            )
            axes_int[3].set_ylabel('Interval Score')
            axes_int[3].grid(axis='y', alpha=0.3)
            axes_int[3].legend()
            _baseline_panel(axes_int[4], ['picp_delta_vs_naive', 'picp_delta_vs_seasonal', 'picp_delta_vs_linear'], 'Prediction Interval Coverage Probability Difference\n(Model minus Baseline)', '.2f', hline=0.0)
            _baseline_panel(axes_int[5], ['nmpiw_delta_vs_naive', 'nmpiw_delta_vs_seasonal', 'nmpiw_delta_vs_linear'], 'Normalized Mean Prediction Interval Width Difference\n(Model minus Baseline)', '.2f', hline=0.0)
            _baseline_panel(axes_int[6], ['interval_score_delta_vs_naive', 'interval_score_delta_vs_seasonal', 'interval_score_delta_vs_linear'], 'Interval Score Delta', '.2e', hline=0.0)
            b_raw = axes_int[7].bar(x, _col('n_eval_raw_segments'), width=0.5, color='tab:orange')
            _annotate_bars_within_ylim(axes_int[7], b_raw, '.0f')
            axes_int[7].set_ylabel('Evaluated Independent Raw Segments (Count)')
            axes_int[7].grid(axis='y', alpha=0.3)
            b_w = axes_int[8].bar(x, _col('sample_reliability_weight'), width=0.5, color='tab:purple')
            _annotate_bars_within_ylim(axes_int[8], b_w, '.2f')
            axes_int[8].set_ylim(0.0, 1.05)
            axes_int[8].set_ylabel('Sample Reliability Weight')
            axes_int[8].grid(axis='y', alpha=0.3)
            axes_int[0].set_title("Interval and Support Diagnostics")
            axes_int[-1].set_xticks(x)
            axes_int[-1].set_xticklabels(labels, rotation=45, ha='right')
            _finalize_stacked_figure(fig_int, axes_int, left=0.36, hspace=0.56)
            int_path = combined_dir / "summary_evidence_intervals_support.png"
            fig_int.savefig(int_path, dpi=300, bbox_inches='tight')

            def _int_panel_picp(ax):
                _draw_bar_group(
                    ax, x, width,
                    [_col('model_picp'), _col('naive_picp'), _col('seasonal_picp'), _col('linear_picp')],
                    trio_colors,
                    trio_methods,
                    '.2f',
                )
                ax.axhline(1.0 - float(args.interval_alpha), color='black', linewidth=0.8, linestyle='--')
                ax.set_ylabel('Prediction Interval Coverage\nProbability')
                ax.set_ylim(0.0, 1.05)
                ax.grid(axis='y', alpha=0.3)
                ax.legend()

            def _int_panel_deficit(ax):
                _draw_bar_group(
                    ax, x, width,
                    [_col('model_coverage_deficit'), _col('naive_coverage_deficit'), _col('seasonal_coverage_deficit'), _col('linear_coverage_deficit')],
                    trio_colors,
                    trio_methods,
                    '.3f',
                )
                ax.axhline(float(args.coverage_tolerance), color='black', linewidth=0.8, linestyle='--')
                ax.set_ylabel('Coverage Deficit')
                ax.grid(axis='y', alpha=0.3)
                ax.legend()

            def _int_panel_nmpiw(ax):
                _draw_bar_group(
                    ax, x, width,
                    [_col('model_nmpiw'), _col('naive_nmpiw'), _col('seasonal_nmpiw'), _col('linear_nmpiw')],
                    trio_colors,
                    trio_methods,
                    '.2f',
                )
                ax.set_ylabel('Normalized Mean Prediction\nInterval Width')
                ax.grid(axis='y', alpha=0.3)
                ax.legend()

            def _int_panel_score(ax):
                _draw_bar_group(
                    ax, x, width,
                    [_col('model_interval_score'), _col('naive_interval_score'), _col('seasonal_interval_score'), _col('linear_interval_score')],
                    trio_colors,
                    trio_methods,
                    '.2e',
                )
                ax.set_ylabel('Interval Score')
                ax.grid(axis='y', alpha=0.3)
                ax.legend()

            def _int_panel_picp_delta(ax):
                _baseline_panel(ax, ['picp_delta_vs_naive', 'picp_delta_vs_seasonal', 'picp_delta_vs_linear'], 'Prediction Interval Coverage Probability Difference\n(Model minus Baseline)', '.2f', hline=0.0)

            def _int_panel_nmpiw_delta(ax):
                _baseline_panel(ax, ['nmpiw_delta_vs_naive', 'nmpiw_delta_vs_seasonal', 'nmpiw_delta_vs_linear'], 'Normalized Mean Prediction Interval Width Difference\n(Model minus Baseline)', '.2f', hline=0.0)

            def _int_panel_interval_delta(ax):
                _baseline_panel(ax, ['interval_score_delta_vs_naive', 'interval_score_delta_vs_seasonal', 'interval_score_delta_vs_linear'], 'Interval Score Delta', '.2e', hline=0.0)

            def _int_panel_raw_segments(ax):
                bars = ax.bar(x, _col('n_eval_raw_segments'), width=0.5, color='tab:orange')
                _annotate_bars_within_ylim(ax, bars, '.0f')
                ax.set_ylabel('Evaluated Independent Raw Segments (Count)')
                ax.grid(axis='y', alpha=0.3)

            def _int_panel_weight(ax):
                bars = ax.bar(x, _col('sample_reliability_weight'), width=0.5, color='tab:purple')
                _annotate_bars_within_ylim(ax, bars, '.2f')
                ax.set_ylim(0.0, 1.05)
                ax.set_ylabel('Sample Reliability Weight')
                ax.grid(axis='y', alpha=0.3)

            int_panels = _save_individual_panels_from_builders(
                out_dir=individual_dir,
                base_name="summary_evidence_intervals_support",
                labels=labels,
                builders=[
                    _int_panel_picp,
                    _int_panel_deficit,
                    _int_panel_nmpiw,
                    _int_panel_score,
                    _int_panel_picp_delta,
                    _int_panel_nmpiw_delta,
                    _int_panel_interval_delta,
                    _int_panel_raw_segments,
                    _int_panel_weight,
                ],
                figsize=(max(11, len(perf_df) * 0.85), 6.4),
                dpi=300,
                left=0.42,
                bottom=0.30,
            )
            plt.close(fig_int)
            print(f"[INFO] Wrote evidence interval/support figure: {int_path}")
            print(f"[INFO] Wrote {len(int_panels)} interval/support panel(s) to {individual_dir}")

            # Gate-by-gate outcomes used in evidence scoring
            fig_gate, axes_gate = plt.subplots(10, 1, figsize=(max(12, len(perf_df) * 0.8), 30), sharex=True)
            gate_specs = [
                (['gate_min_raw_vs_naive', 'gate_min_raw_vs_seasonal', 'gate_min_raw_vs_linear'], 'Gate: Minimum Independent Raw Sample Count'),
                (['gate_prob_vs_naive', 'gate_prob_vs_seasonal', 'gate_prob_vs_linear'], 'Gate: Bootstrap Probability of Positive Skill'),
                (['gate_lcb_vs_naive', 'gate_lcb_vs_seasonal', 'gate_lcb_vs_linear'], 'Gate: 95% Lower Confidence Bound of Skill > 0'),
                (['gate_dm_vs_naive', 'gate_dm_vs_seasonal', 'gate_dm_vs_linear'], 'Gate: Diebold-Mariano p-value < alpha and statistic < 0'),
                (['gate_wilcoxon_vs_naive', 'gate_wilcoxon_vs_seasonal', 'gate_wilcoxon_vs_linear'], 'Gate: Wilcoxon p-value < alpha'),
                (['gate_sign_vs_naive', 'gate_sign_vs_seasonal', 'gate_sign_vs_linear'], 'Gate: Sign Test p-value < alpha and win rate > 0.5'),
                (['gate_coverage_vs_naive', 'gate_coverage_vs_seasonal', 'gate_coverage_vs_linear'], 'Gate: Coverage Quality'),
                (['gate_dm_q_vs_naive', 'gate_dm_q_vs_seasonal', 'gate_dm_q_vs_linear'], 'Gate: Diebold-Mariano q-value < alpha and statistic < 0'),
                (['gate_wilcoxon_q_vs_naive', 'gate_wilcoxon_q_vs_seasonal', 'gate_wilcoxon_q_vs_linear'], 'Gate: Wilcoxon q-value < alpha'),
                (['gate_sign_q_vs_naive', 'gate_sign_q_vs_seasonal', 'gate_sign_q_vs_linear'], 'Gate: Sign Test q-value < alpha and win rate > 0.5'),
            ]
            for ax_g, (g_cols, ylab) in zip(axes_gate, gate_specs):
                _baseline_panel(ax_g, g_cols, ylab, '.0f', ylim=(0.0, 1.05))
            axes_gate[0].set_title("Evidence Gates (top-to-bottom follows score construction)")
            axes_gate[-1].set_xticks(x)
            axes_gate[-1].set_xticklabels(labels, rotation=45, ha='right')
            _finalize_stacked_figure(fig_gate, axes_gate, left=0.40, hspace=0.60)
            gate_path = combined_dir / "summary_evidence_gates.png"
            fig_gate.savefig(gate_path, dpi=300, bbox_inches='tight')

            gate_builders = [
                (lambda g_cols=g_cols, ylab=ylab: (lambda ax: _baseline_panel(ax, g_cols, ylab, '.0f', ylim=(0.0, 1.05))))()
                for g_cols, ylab in gate_specs
            ]
            gate_panels = _save_individual_panels_from_builders(
                out_dir=individual_dir,
                base_name="summary_evidence_gates",
                labels=labels,
                builders=gate_builders,
                figsize=(max(11, len(perf_df) * 0.85), 6.4),
                dpi=300,
                left=0.46,
                bottom=0.30,
            )
            plt.close(fig_gate)
            print(f"[INFO] Wrote evidence gates figure: {gate_path}")
            print(f"[INFO] Wrote {len(gate_panels)} evidence gate panel(s) to {individual_dir}")

            # MC replicate uncertainty impact on final accuracy/evidence
            def _is_informative_series(s: pd.Series, atol: float = 1e-12) -> bool:
                vals = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size <= 1:
                    return False
                return bool((np.nanmax(vals) - np.nanmin(vals)) > float(atol))

            mc_specs = [
                ("mc_target_wb_ratio", 'Target Within-Segment to Between-Segment\nMean Square Ratio', 'tab:orange', '.2f', 1.0, None),
                ("mc_target_icc", 'Target Intraclass Correlation Coefficient\n(Between-Segment Variance Share)', 'tab:olive', '.2f', None, (0.0, 1.05)),
                ("mc_target_noise_fraction", 'Target Noise Fraction\n(Within-Segment Variance Share)', 'tab:red', '.2f', None, (0.0, 1.05)),
                ("rmse_to_mc_within_sd", 'RMSE /\nMonte Carlo Within-Segment SD', 'tab:blue', '.2f', 1.0, None),
                ("rmse_to_mc_between_sd", 'RMSE /\nBetween-Segment SD', 'tab:cyan', '.2f', 1.0, None),
                ("mc_uncertainty_vs_error_corr", 'Correlation: Replicate SD vs\nSegment MAE', 'tab:brown', '.2f', 0.0, (-1.05, 1.05)),
            ]
            mc_specs_informative = [spec for spec in mc_specs if _is_informative_series(_col(spec[0]))]
            if mc_specs_informative:
                fig_mc, axes_mc = plt.subplots(
                    len(mc_specs_informative), 1,
                    figsize=(max(12, len(perf_df) * 0.8), max(8, 3.0 * len(mc_specs_informative))),
                    sharex=True,
                )
                if not isinstance(axes_mc, np.ndarray):
                    axes_mc = np.array([axes_mc])
                for ax_mc, (col_name, ylab, color, fmt, hline, ylim) in zip(axes_mc, mc_specs_informative):
                    bars = ax_mc.bar(x, _col(col_name), width=0.5, color=color)
                    _annotate_bars_within_ylim(ax_mc, bars, fmt)
                    if hline is not None and np.isfinite(hline):
                        ax_mc.axhline(float(hline), color='black', linewidth=0.8, linestyle='--')
                    if ylim is not None:
                        ax_mc.set_ylim(float(ylim[0]), float(ylim[1]))
                    ax_mc.set_ylabel(ylab)
                    ax_mc.grid(axis='y', alpha=0.3)
                axes_mc[0].set_title("Monte Carlo Replicate Uncertainty Impact")
                axes_mc[-1].set_xticks(x)
                axes_mc[-1].set_xticklabels(labels, rotation=45, ha='right')
                _finalize_stacked_figure(fig_mc, axes_mc, left=0.38, hspace=0.58)
                mc_path = combined_dir / "summary_mc_uncertainty_impact.png"
                fig_mc.savefig(mc_path, dpi=300, bbox_inches='tight')

                mc_builders = []
                for col_name, ylab, color, fmt, hline, ylim in mc_specs_informative:
                    def _builder(ax, col_name=col_name, ylab=ylab, color=color, fmt=fmt, hline=hline, ylim=ylim):
                        bars = ax.bar(x, _col(col_name), width=0.5, color=color)
                        _annotate_bars_within_ylim(ax, bars, fmt)
                        if hline is not None and np.isfinite(hline):
                            ax.axhline(float(hline), color='black', linewidth=0.8, linestyle='--')
                        if ylim is not None:
                            ax.set_ylim(float(ylim[0]), float(ylim[1]))
                        ax.set_ylabel(ylab)
                        ax.grid(axis='y', alpha=0.3)

                    mc_builders.append(_builder)

                mc_panels = _save_individual_panels_from_builders(
                    out_dir=individual_dir,
                    base_name="summary_mc_uncertainty_impact",
                    labels=labels,
                    builders=mc_builders,
                    figsize=(max(11, len(perf_df) * 0.85), 6.2),
                    dpi=300,
                    left=0.44,
                    bottom=0.30,
                )
                plt.close(fig_mc)
                print(f"[INFO] Wrote MC uncertainty impact figure: {mc_path}")
                print(f"[INFO] Wrote {len(mc_panels)} MC uncertainty panel(s) to {individual_dir}")
            else:
                print("[INFO] Skipped MC uncertainty impact figure: all panels were constant or non-finite.")

            # --- Single model quality matrix (accuracy, precision, reliability, support) ---
            # Matrix-specific ordering: evidence tier, then p-value, then q-value.
            matrix_perf_df = perf_df.copy()
            _bare_prefix = args.dataset_prefix.rstrip("_")
            _final_labels = [
                clean_target_label(_name, _bare_prefix)
                for _name in matrix_perf_df["dataset"].astype(str).tolist()
            ]
            matrix_perf_df["_target_label"] = _final_labels

            n_rows_mat = len(matrix_perf_df)

            q_cols = [
                "dm_q_vs_naive", "dm_q_vs_seasonal", "dm_q_vs_linear",
                "wilcoxon_q_vs_naive", "wilcoxon_q_vs_seasonal", "wilcoxon_q_vs_linear",
                "sign_q_vs_naive", "sign_q_vs_seasonal", "sign_q_vs_linear",
            ]
            p_cols = [
                "dm_p_vs_naive", "dm_p_vs_seasonal", "dm_p_vs_linear",
                "wilcoxon_p_vs_naive", "wilcoxon_p_vs_seasonal", "wilcoxon_p_vs_linear",
                "sign_p_vs_naive", "sign_p_vs_seasonal", "sign_p_vs_linear",
            ]
            present_q_cols = [c for c in q_cols if c in matrix_perf_df.columns]
            present_p_cols = [c for c in p_cols if c in matrix_perf_df.columns]

            # Compute sort-key arrays from the pre-sort DataFrame.
            _sort_q_min = pd.to_numeric(matrix_perf_df[present_q_cols].min(axis=1, skipna=True), errors="coerce").to_numpy(dtype=float) if present_q_cols else np.full(n_rows_mat, np.nan, dtype=float)
            _sort_p_min = pd.to_numeric(matrix_perf_df[present_p_cols].min(axis=1, skipna=True), errors="coerce").to_numpy(dtype=float) if present_p_cols else np.full(n_rows_mat, np.nan, dtype=float)
            _sort_q_tie = np.where(np.isfinite(_sort_q_min), _sort_q_min, _sort_p_min)
            _sort_score = pd.to_numeric(
                matrix_perf_df.get("evidence_score_overall_min", pd.Series([0] * n_rows_mat)),
                errors="coerce",
            ).to_numpy(dtype=float)
            matrix_perf_df["_tier_sort"] = np.where(np.isfinite(_sort_score), _sort_score, np.inf)
            matrix_perf_df["_p_sort"] = np.where(np.isfinite(_sort_p_min), _sort_p_min, np.inf)
            matrix_perf_df["_q_sort"] = np.where(np.isfinite(_sort_q_tie), _sort_q_tie, np.inf)
            matrix_perf_df = matrix_perf_df.sort_values(
                by=["_tier_sort", "_p_sort", "_q_sort", "_target_label"],
                ascending=[True, False, False, True],
                kind="mergesort",
            )
            matrix_perf_df = matrix_perf_df.drop(columns=["_tier_sort", "_p_sort", "_q_sort"])

            matrix_index = matrix_perf_df["_target_label"].astype(str).tolist()
            n_rows_mat = len(matrix_perf_df)

            # Recompute metric arrays from the sorted DataFrame so quality_df rows align.
            present_q_cols = [c for c in q_cols if c in matrix_perf_df.columns]
            present_p_cols = [c for c in p_cols if c in matrix_perf_df.columns]
            q_min = pd.to_numeric(matrix_perf_df[present_q_cols].min(axis=1, skipna=True), errors="coerce").to_numpy(dtype=float) if present_q_cols else np.full(n_rows_mat, np.nan, dtype=float)
            p_min = pd.to_numeric(matrix_perf_df[present_p_cols].min(axis=1, skipna=True), errors="coerce").to_numpy(dtype=float) if present_p_cols else np.full(n_rows_mat, np.nan, dtype=float)
            tier_vals = pd.to_numeric(
                matrix_perf_df.get("evidence_score_overall_min", pd.Series([0] * n_rows_mat)),
                errors="coerce",
            ).to_numpy(dtype=float)

            def _col_values(name: str) -> np.ndarray:
                if name in matrix_perf_df.columns:
                    return pd.to_numeric(matrix_perf_df[name], errors="coerce").to_numpy(dtype=float)
                return np.full(n_rows_mat, np.nan, dtype=float)

            quality_df = pd.DataFrame({
                "Test Sample Count": _col_values("n_eval_raw_segments"),
                "R²": _col_values("r2"),
                "nRMSE": _col_values("nrmse"),
                "Skill vs. Best Baseline": pd.concat([
                    pd.Series(_col_values("skill_vs_naive")),
                    pd.Series(_col_values("skill_vs_seasonal")),
                    pd.Series(_col_values("skill_vs_linear")),
                ], axis=1).min(axis=1, skipna=True).to_numpy(dtype=float),
                "Coverage Gap (PICP − Nominal)": _col_values("model_coverage_gap"),
                "Normalized Mean Prediction Interval Width": _col_values("model_nmpiw"),
                "Minimum 95% Lower Confidence Bound of Skill": pd.concat([
                    pd.Series(_col_values("lcb95_skill_vs_naive")),
                    pd.Series(_col_values("lcb95_skill_vs_seasonal")),
                    pd.Series(_col_values("lcb95_skill_vs_linear")),
                ], axis=1).min(axis=1, skipna=True).to_numpy(dtype=float),
                "Best False Discovery Rate Adjusted q-value": q_min,
                "Best p-value": p_min,
                "Evidence Score": tier_vals,
            }, index=matrix_index)

            # Fallback to p-values if q-values are unavailable.
            if not np.isfinite(quality_df["Best False Discovery Rate Adjusted q-value"].to_numpy(dtype=float)).any():
                quality_df["Best False Discovery Rate Adjusted q-value"] = quality_df["Best p-value"]

            # Column-wise directional scaling for heatmap coloring only.
            higher_better = {
                "Test Sample Count": True,
                "R²": True,
                "nRMSE": False,
                "Skill vs. Best Baseline": True,
                "Normalized Mean Prediction Interval Width": False,
                "Minimum 95% Lower Confidence Bound of Skill": True,
                "Best False Discovery Rate Adjusted q-value": False,
                "Evidence Score": True,
            }
            if "Best p-value" in quality_df.columns:
                higher_better["Best p-value"] = False

            non_gate_cols = [
                "Test Sample Count",
                "Best Model",
                "R²",
                "nRMSE",
                "Skill vs. Best Baseline",
                "Normalized Mean Prediction Interval Width",
            ]
            gate_cols = [
                "Coverage Gap (PICP − Nominal)",
                "Minimum 95% Lower Confidence Bound of Skill",
                "Best False Discovery Rate Adjusted q-value",
                "Evidence Score",
            ]
            if np.isfinite(quality_df["Best p-value"].to_numpy(dtype=float)).any():
                gate_cols.insert(2, "Best p-value")

            # Visual separator between descriptive metrics and gate-evaluated metrics.
            # Insert 'Best Model' column after 'Test Sample Count'
            if 'model' in matrix_perf_df.columns:
                # Normalize model type for display (XGB, Transformer, GP)
                def _display_model_type(val):
                    key = str(val).strip().lower()
                    if 'xgb' in key:
                        return 'XGB'
                    elif 'transformer' in key:
                        return 'Trans.'
                    elif 'gp' in key:
                        return 'GP'
                    else:
                        return key.title()
                quality_df['Best Model'] = matrix_perf_df['model'].map(_display_model_type).values
            else:
                quality_df['Best Model'] = 'Unknown'
            # Place as second column (after Test Sample Count)
            col_order = list(quality_df.columns)
            if 'Best Model' in col_order:
                col_order.insert(1, col_order.pop(col_order.index('Best Model')))
            quality_df = quality_df[col_order]
            quality_df[""] = np.nan
            heat_cols = col_order[:len(non_gate_cols)+2] + [""] + col_order[len(non_gate_cols)+2:]
            display_df = pd.concat(
                [
                    quality_df[non_gate_cols],
                    quality_df[[""]],
                    quality_df[gate_cols],
                ],
                axis=1,
            ).copy()

            zero_centered_cols = {"Coverage Gap (PICP − Nominal)"}
            norm = display_df.copy()
            for c in norm.columns:
                if c == 'Best Model':
                    # Categorical column: fill with 0.5 (no heatmap)
                    norm[c] = 0.5
                elif c in zero_centered_cols:
                    vals = pd.to_numeric(norm[c], errors="coerce")
                    finite = vals[np.isfinite(vals)]
                    if finite.empty:
                        norm[c] = np.nan
                        continue
                    max_abs = float(finite.abs().max())
                    if max_abs == 0:
                        norm[c] = pd.Series([1.0] * len(vals), index=vals.index, dtype=float)
                    else:
                        norm[c] = 1.0 - vals.abs() / max_abs
                else:
                    vals = pd.to_numeric(norm[c], errors="coerce")
                    finite = vals[np.isfinite(vals)]
                    if finite.empty:
                        norm[c] = np.nan
                        continue
                    vmin = float(finite.min())
                    vmax = float(finite.max())
                    if np.isclose(vmin, vmax):
                        scaled = pd.Series([0.5] * len(vals), index=vals.index, dtype=float)
                    else:
                        scaled = (vals - vmin) / (vmax - vmin)
                    if not higher_better.get(c, True):
                        scaled = 1.0 - scaled
                    norm[c] = scaled

            annot = display_df.copy()
            for c in annot.columns:
                if c == 'Best Model':
                    annot[c] = ""  # drawn manually after rectangles
                elif c in {"Evidence Score", "Test Sample Count"}:
                    annot[c] = annot[c].map(lambda v: "" if not np.isfinite(v) else f"{int(round(v))}")
                elif c in {"Best False Discovery Rate Adjusted q-value", "Best p-value"}:
                    annot[c] = annot[c].map(lambda v: "" if not np.isfinite(v) else f"{v:.3f}")
                else:
                    annot[c] = annot[c].map(lambda v: "" if not np.isfinite(v) else f"{v:.2f}")

            fig_mat, ax_mat = plt.subplots(figsize=(max(8, 0.75 * len(heat_cols)), max(4, 0.32 * len(display_df))))
            # Custom coloring for 'Best Model' column
            from matplotlib.colors import ListedColormap
            model_color_map = {'XGB': 'tab:blue', 'Trans.': 'tab:purple', 'GP': 'tab:olive'}
            # Build color matrix
            color_matrix = np.full(norm.shape, np.nan, dtype=object)
            for i, col in enumerate(norm.columns):
                if col == 'Best Model':
                    for j, val in enumerate(display_df['Best Model']):
                        color_matrix[j, i] = model_color_map.get(val, 'tab:gray')
                else:
                    color_matrix[:, i] = None
            def _cell_color_func(val, row, col):
                if norm.columns[col] == 'Best Model':
                    return model_color_map.get(display_df['Best Model'].iloc[row], 'tab:gray')
                return None
            # Draw heatmap
            sns.heatmap(
                norm,
                ax=ax_mat,
                cmap="RdYlGn",
                vmin=0.0,
                vmax=1.0,
                cbar=False,
                linewidths=0.5,
                linecolor="white",
                annot=annot.values,
                fmt="",
                annot_kws={"fontsize": 8, "rotation": 0},
            )
            # Overlay colored rectangles for 'Best Model' column, then draw labels on top
            for i, col in enumerate(norm.columns):
                if col == 'Best Model':
                    for j, val in enumerate(display_df['Best Model']):
                        rect = plt.Rectangle((i, j), 1, 1, facecolor=model_color_map.get(val, 'tab:gray'), edgecolor='white', linewidth=0.5, zorder=4)
                        ax_mat.add_patch(rect)
                        ax_mat.text(i + 0.5, j + 0.5, str(val), ha='center', va='center', fontsize=8, zorder=5)
            if "" in display_df.columns:
                sep_col = int(display_df.columns.get_loc(""))
                ax_mat.add_patch(
                    plt.Rectangle(
                        (sep_col, 0),
                        1,
                        int(display_df.shape[0]),
                        facecolor=ax_mat.get_facecolor(),
                        edgecolor="none",
                        zorder=3,
                    )
                )
            ax_mat.set_title("")
            # ax_mat.set_xlabel("Metrics")
            # ax_mat.set_ylabel("Target")
            ax_mat.set_yticklabels(ax_mat.get_yticklabels(), rotation=0, fontsize=8)
            dm_suffix_cols = {"Best False Discovery Rate Adjusted q-value", "Best p-value"}
            wrapped_xlabels = []
            for xt in ax_mat.get_xticklabels():
                txt = xt.get_text()
                if txt in dm_suffix_cols:
                    txt = txt + " (DM/Wilcoxon/Sign)"
                wrapped_xlabels.append(textwrap.fill(txt, width=18))
            ax_mat.set_xticklabels(wrapped_xlabels, rotation=60, ha="right", fontsize=8)
            plt.tight_layout()
            matrix_path = evaluation_dir / "summary_model_quality_matrix.png"
            fig_mat.savefig(matrix_path, dpi=300, bbox_inches='tight')
            plt.close(fig_mat)
            print(f"[INFO] Wrote model quality matrix: {matrix_path}")

            # --- Cross-validation figure (sorted by descending R2, same order as perf_df) ---
            cv_cols = [
                c for c in ['rolling_cv_r2', 'rolling_cv_r2_median', 'rolling_cv_r2_last50', 'rolling_cv_r2_pooled']
                if c in perf_df.columns
            ]
            cv_data_available = run_rolling_cv and bool(cv_cols) and perf_df[cv_cols].notnull().any().any()
            if cv_data_available:
                # Generalization gap: test R2 - CV R2 (positive = test was optimistic / overfit)
                cv_r2_mean_col = perf_df['rolling_cv_r2'] if 'rolling_cv_r2' in perf_df.columns else pd.Series([float('nan')] * len(perf_df))
                cv_r2_median_col = perf_df['rolling_cv_r2_median'] if 'rolling_cv_r2_median' in perf_df.columns else pd.Series([float('nan')] * len(perf_df))
                cv_r2_last50_col = perf_df['rolling_cv_r2_last50'] if 'rolling_cv_r2_last50' in perf_df.columns else pd.Series([float('nan')] * len(perf_df))
                cv_r2_pooled_col = perf_df['rolling_cv_r2_pooled'] if 'rolling_cv_r2_pooled' in perf_df.columns else pd.Series([float('nan')] * len(perf_df))
                gen_gap_mean = perf_df['r2'] - cv_r2_mean_col
                gen_gap_last50 = perf_df['r2'] - cv_r2_last50_col
                n_folds_col = perf_df['rolling_cv_n_folds'] if 'rolling_cv_n_folds' in perf_df.columns else pd.Series([float('nan')] * len(perf_df))
                n_samples_col = perf_df['n_eval_raw_segments'] if 'n_eval_raw_segments' in perf_df.columns else (
                    perf_df['n_eval_rows_test'] if 'n_eval_rows_test' in perf_df.columns else pd.Series([float('nan')] * len(perf_df))
                )

                cv_panel_specs = [
                    (cv_r2_mean_col,           'Cross-Validation Coefficient of Determination\n(All Folds Mean)', 'tab:blue',   '.2f'),
                    (cv_r2_median_col,         'Cross-Validation Coefficient of Determination\n(All Folds Median)', 'tab:cyan', '.2f'),
                    (cv_r2_last50_col,         'Cross-Validation Coefficient of Determination\n(Last 50% of Folds)', 'tab:green', '.2f'),
                    (cv_r2_pooled_col,         'Cross-Validation Coefficient of Determination\n(Pooled Sum of Squares)', 'tab:olive', '.2f'),
                    (gen_gap_mean,             'Generalization Gap\n(Test Coefficient of Determination - Cross-Validation Mean)', 'tab:red', '.2e'),
                    (gen_gap_last50,           'Generalization Gap\n(Test Coefficient of Determination - Cross-Validation Last 50%)', 'tab:pink', '.2e'),
                    (n_folds_col,              'Cross-Validation Fold Count',      'tab:purple', '.0f'),
                    (n_samples_col,            'Evaluation Support (Count)',  'tab:orange', '.0f'),
                ]
                fig_cv, cv_axes = plt.subplots(
                    len(cv_panel_specs), 1,
                    figsize=(max(10, len(perf_df) * 0.7), 3.5 * len(cv_panel_specs)),
                    sharex=True,
                )
                for ax_cv, (vals, ylabel, color, fmt) in zip(cv_axes, cv_panel_specs):
                    bars = ax_cv.bar(x, vals, width=0.5, color=color)
                    if ylabel.startswith('Cross-Validation Coefficient of Determination'):
                        ax_cv.set_ylim(-0.1, 1.0)
                    _annotate_bars_within_ylim(ax_cv, bars, fmt)
                    if ylabel.startswith('Generalization'):
                        ax_cv.axhline(0, color='black', linewidth=0.8, linestyle='--')
                    ax_cv.set_ylabel(ylabel)
                    ax_cv.grid(axis='y', alpha=0.3)
                cv_axes[-1].set_xticks(x)
                cv_axes[-1].set_xticklabels(labels, rotation=45, ha='right')
                plt.tight_layout()
                cv_path = combined_dir / "cross-validation.png"
                fig_cv.savefig(cv_path, dpi=300, bbox_inches='tight')

                cv_builders = []
                for vals, ylabel, color, fmt in cv_panel_specs:
                    def _builder(ax, vals=vals, ylabel=ylabel, color=color, fmt=fmt):
                        bars = ax.bar(x, vals, width=0.5, color=color)
                        if ylabel.startswith('Cross-Validation Coefficient of Determination'):
                            ax.set_ylim(-0.1, 1.0)
                        _annotate_bars_within_ylim(ax, bars, fmt)
                        if ylabel.startswith('Generalization'):
                            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
                        ax.set_ylabel(ylabel)
                        ax.grid(axis='y', alpha=0.3)

                    cv_builders.append(_builder)

                cv_panels = _save_individual_panels_from_builders(
                    out_dir=individual_dir,
                    base_name="cross-validation",
                    labels=labels,
                    builders=cv_builders,
                    figsize=(max(11, len(perf_df) * 0.85), 6.2),
                    dpi=300,
                    left=0.40,
                    bottom=0.30,
                )
                plt.close(fig_cv)
                print(f"[INFO] Wrote cross-validation figure: {cv_path}")
                print(f"[INFO] Wrote {len(cv_panels)} cross-validation panel(s) to {individual_dir}")
            else:
                if run_rolling_cv:
                    print("[WARN] No cross-validation data available; cross-validation.png not generated.")
                else:
                    print("[INFO] Cross-validation figure skipped (enable with --run-rolling-cv).")
        else:
            print("[WARN] No best model performance data found; summary plot not generated.")
    except Exception as e:
        print(f"[ERROR] Failed to generate summary_best_model_performance.png: {e}")
        traceback.print_exc()

    # Generate ML model comparison charts (one figure per metric, bars clustered by target)
    try:
        if plans:
            summaries_dir_ml = _resolve_summaries_dir(
                data_root=data_root,
                sweep_namespace=str(getattr(args, "sweep_namespace", "feature_sweeps")),
            )
            _plot_ml_model_comparison(plans, data_root, summaries_dir_ml, args)
    except Exception as e:
        print(f"[ERROR] Failed to generate ML model comparison plots: {e}")
        traceback.print_exc()

    if len(sweep_results) > 1:
        print("\n" + "=" * 100)
        print("MULTI-TARGET FEATURE IMPORTANCE COMPARISON (POSTPROCESS)")
        print("=" * 100)
        try:
            if importance_sources_used and all(src.startswith("shapley_") for src in importance_sources_used):
                comparison_plot = _compile_multi_target_comparison(
                    sweep_results,
                    data_root,
                    importance_label="Shapley Expected Contribution (mean marginal objective delta)",
                    summary_axis_label="Total Shapley Contribution",
                    target_order=target_order_by_skill,
                    dataset_prefix=args.dataset_prefix,
                )
            elif importance_sources_used and any(src.startswith("shapley_") for src in importance_sources_used):
                comparison_plot = _compile_multi_target_comparison(
                    sweep_results,
                    data_root,
                    importance_label="Feature Importance (mixed: removal delta + Shapley contribution)",
                    summary_axis_label="Total Feature Importance",
                    target_order=target_order_by_skill,
                    dataset_prefix=args.dataset_prefix,
                )
            else:
                comparison_plot = _compile_multi_target_comparison(
                    sweep_results,
                    data_root,
                    target_order=target_order_by_skill,
                    dataset_prefix=args.dataset_prefix,
                )
            if comparison_plot.exists():
                print(f"[INFO] Wrote multi-target comparison plots to {comparison_plot.parent}")
        except Exception as e:
            print(f"[WARN] Failed to regenerate multi-target comparison: {e}")

    # Feature inclusion heatmap (binary: is predictor in best model's feature set?)
    if best_model_performance:
        try:
            inclusion_plot = _compile_feature_inclusion_heatmap(
                perf_df=perf_df,
                plans=plans,
                data_root=data_root,
                target_order=target_order_by_skill,
                dataset_prefix=args.dataset_prefix,
                sweep_namespace=str(getattr(args, "sweep_namespace", "feature_sweeps")),
            )
            if inclusion_plot.exists():
                print(f"[INFO] Wrote feature inclusion heatmap: {inclusion_plot}")
        except Exception as e:
            print(f"[WARN] Failed to generate feature inclusion heatmap: {e}")
            traceback.print_exc()

    if datasets_with_outputs == 0:
        print("[WARN] No dataset outputs were regenerated.")
        return 1

    print(f"[INFO] Regenerated feature-sweep outputs for {datasets_with_outputs} dataset(s).")
    return 0


def main() -> int:
    parser = build_parser()
    # Override inherited sweep default so z1 processes all matching datasets unless capped.
    parser.set_defaults(limit_datasets=0)
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Optional alias for --data-root; dataset root directory to scan.",
    )
    parser.add_argument("--dm-max-lag", type=int, default=1, help="Max HAC lag for Diebold-Mariano test.")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000, help="Bootstrap iterations for grouped skill confidence.")
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="Random seed for bootstrap evidence.")
    parser.add_argument(
        "--bootstrap-mode",
        type=str,
        default="iid",
        choices=["iid", "moving_block"],
        help="Grouped bootstrap mode: iid resampling of groups or moving-block resampling.",
    )
    parser.add_argument("--bootstrap-block-len", type=int, default=3, help="Block length for moving-block bootstrap mode.")
    parser.add_argument("--evidence-alpha", type=float, default=0.05, help="Alpha threshold for statistical tests and FDR-adjusted tests.")
    parser.add_argument("--evidence-min-raw-samples", type=int, default=12, help="Minimum independent raw samples required for high confidence.")
    parser.add_argument("--evidence-min-prob", type=float, default=0.8, help="Minimum bootstrap probability of skill > 0.")
    parser.add_argument("--evidence-ref-raw-samples", type=int, default=40, help="Reference raw-sample count for reliability weighting.")
    parser.add_argument("--interval-alpha", type=float, default=0.1, help="Alpha for post-hoc residual interval proxy metrics (PICP/NMPIW).")
    parser.add_argument("--coverage-tolerance", type=float, default=0.03, help="Allowable shortfall tolerance for coverage gate.")
    parser.add_argument(
        "--stability-replicates",
        type=int,
        default=5,
        help=(
            "Number of independent retraining replicates for the stability gate "
            "(includes the original fit as replicate-0). "
            "Set to 1 to skip the stability check entirely. Default: 5."
        ),
    )
    parser.add_argument(
        "--stability-cv-threshold",
        type=float,
        default=0.15,
        help=(
            "Maximum allowed coefficient of variation of R² across retraining replicates "
            "for gate_stability to pass (CV = std / |mean|). Default: 0.15."
        ),
    )
    parser.add_argument(
        "--run-rolling-cv",
        action="store_true",
        help="Run and include rolling-origin cross-validation outputs (disabled unless this keyword is provided).",
    )
    parser.add_argument(
        "--sweep-namespace",
        type=str,
        default="feature_sweeps",
        help=(
            "Forecast sweep subdirectory to post-process "
            "(e.g., 'feature_sweeps' or 'Shapley_sweeps')."
        ),
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help=(
            "Process all dataset folders in --data-root; clears dataset-prefix "
            "filtering and disables dataset-count capping."
        ),
    )
    args = parser.parse_args()
    os.environ["WQ_FEATURE_SWEEP_NAMESPACE"] = str(args.sweep_namespace).strip() or "feature_sweeps"
    workspace_root = Path(__file__).resolve().parent.parent
    data_root_arg = args.path if args.path else args.data_root
    data_root = Path(data_root_arg)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()
    include_regular, include_res = _resolve_dataset_inclusion(args)
    dataset_prefix = "" if args.all_datasets else str(args.dataset_prefix)
    limit_datasets = 0 if args.all_datasets else int(args.limit_datasets)
    plans = discover_mc_dataset_plans(
            data_root=data_root,
            dataset_prefix=dataset_prefix,
            config_pattern=args.config_pattern,
            limit_datasets=limit_datasets,
            include_regular=include_regular,
            include_res=include_res,
        )
    output = post(plans, args)
    return output

if __name__ == "__main__":
    sys.exit(main())

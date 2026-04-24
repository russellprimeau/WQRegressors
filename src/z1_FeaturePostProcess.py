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
    `--ml-selection {best,xgb}`: Choose whether ML-family best-model summaries use
      the best of XGB/GP/Transformer or restrict ML-family selection to XGB only.
    `--treat-mlr-as-baseline`: Include the best MLR result as an additional
      baseline candidate when computing “best baseline” skill summaries.
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
python src/z1_FeaturePostProcess.py --path data/output/CV19 --sweep-namespace feature_sweeps --bootstrap-mode moving_block --bootstrap-block-len 7 --treat-mlr-as-baseline
python src/z1_FeaturePostProcess.py --ml-selection xgb
python src/z1_FeaturePostProcess.py --treat-mlr-as-baseline
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
from utils.config_utils import select_best_model_row
from utils.names import clean_target_label
from utils.mlr import evaluate_mlr as _evaluate_mlr
from h_RunMCFeatureSelectionSweep import build_parser, discover_mc_dataset_plans, _derive_target_name, _select_surrogate_config, _parse_row_counts, _available_row_counts_for_postprocess, _regenerate_saved_outputs_for_row, _load_feature_stats_artifacts_with_source, _compile_multi_target_comparison, _resolve_dataset_inclusion, _run_rolling_origin_cv, _ensure_k01_baselines, _write_dataset_evaluation_summary, _forecast_sweeps_dir, _plot_final_metrics_comparison, _feature_tag, _mlr_artifact_dir, _write_mlr_artifacts, _run_mlr_variants_on_existing_split

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
    "mlr": "MLR",
    "mlr_avg12": "MLR-12",
    "mlr_avgall": "MLR-All",
}
BASELINE_PLOT_COLORS = {
    "naive": "tab:gray",
    "seasonal": "tab:green",
    "linear": "tab:orange",
    "mlr": "tab:red",
    "mlr_avg12": "tab:purple",
    "mlr_avgall": "tab:brown",
}
BASELINE_MODEL_IDS = {"naive", "seasonal", "linear"}
MLR_MODEL_IDS = {"mlr", "mlr_avg12", "mlr_avgall"}
XGB_MODEL_IDS = {"xgb", "xgbregressor", "xgb_regressor", "xgbclassifier", "xgb_classifier"}
MIN_REQUIRED_VALID_INDEPENDENT = 5

ML_COMPARISON_MODEL_TYPES = ['XGB', 'Trans.', 'GP', 'MLR', 'MLR12', 'MLRall']
ML_COMPARISON_COLORS = {
    'GP': 'tab:blue', 'Trans.': 'tab:orange', 'XGB': 'tab:green',
    'MLR': 'tab:red', 'MLR12': 'tab:purple', 'MLRall': 'tab:brown',
}


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


@dataclass
class SelectionRecord:
    dataset: str
    target: str
    best_model_row: pd.Series
    best_baseline_row: pd.Series
    best_model_id: str
    best_model_label: str
    best_baseline_id: str
    best_baseline_label: str
    baseline_model_set: tuple[str, ...]
    best_model_is_configured_baseline: bool
    best_model_equals_best_baseline: bool
    skill_vs_best_baseline: float

def _safe_float(val) -> float:
    """Return float(val) if val is non-null, otherwise float('nan')."""
    try:
        return float(val) if pd.notnull(val) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _is_baseline_model_value(value: object) -> bool:
    return str(value).strip().lower() in BASELINE_MODEL_IDS


def configured_baseline_ids(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the model ids considered baselines for this postprocess run."""
    ids = list(BASELINE_ORDER)
    if bool(getattr(args, "treat_mlr_as_baseline", False)):
        ids.extend(("mlr", "mlr_avg12", "mlr_avgall"))
    return tuple(ids)


def is_configured_baseline(value: object, args: argparse.Namespace) -> bool:
    return str(value).strip().lower() in set(configured_baseline_ids(args))


def _exclude_baseline_metric_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    out = df.copy()
    if 'model' not in out.columns:
        return out
    return out[~out['model'].apply(_is_baseline_model_value)].copy()


def _normalize_model_id(value: object) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


_NORMALIZED_XGB_MODEL_IDS = {_normalize_model_id(v) for v in XGB_MODEL_IDS}
_NORMALIZED_MLR_MODEL_IDS = {_normalize_model_id(v) for v in MLR_MODEL_IDS}

BEST_BASELINE_PLOT_LABELS = {
    **BASELINE_PLOT_LABELS,
    "mlr": "MLR",
}


def _is_xgb_model_value(value: object) -> bool:
    return _normalize_model_id(value) in _NORMALIZED_XGB_MODEL_IDS


def _is_mlr_model_value(value: object) -> bool:
    return _normalize_model_id(value) in _NORMALIZED_MLR_MODEL_IDS


def _filter_best_model_candidates(df: "pd.DataFrame", ml_selection: str) -> "pd.DataFrame":
    """Restrict ML-family candidates for best-model selection when requested."""
    out = df.copy()
    if ml_selection != "xgb" or "model" not in out.columns:
        return out

    model_vals = out["model"]
    keep_mask = model_vals.apply(_is_xgb_model_value) | model_vals.apply(_is_mlr_model_value)
    return out[keep_mask].copy()


def _best_baseline_order(args: argparse.Namespace) -> tuple[str, ...]:
    return configured_baseline_ids(args)


def _load_best_mlr_baseline_stats(data_root: Path, dataset_name: str) -> dict[str, float]:
    """Return stats for the best MLR row from feature_sweep_final_metrics.csv."""
    csv_path = Path(data_root) / dataset_name / "forecasts" / "feature_sweeps" / "feature_sweep_final_metrics.csv"
    if not csv_path.exists():
        return {k: float("nan") for k in ("mae", "rmse", "r2", "pearson_r")}
    try:
        df = pd.read_csv(csv_path)
        if df.empty or "model" not in df.columns:
            raise ValueError("no MLR metrics rows available")
        mlr_df = df[df["model"].apply(_is_mlr_model_value)].copy()
        mlr_df = _filter_min_valid_independent(_filter_valid_rows(mlr_df), min_required=MIN_REQUIRED_VALID_INDEPENDENT)
        if mlr_df.empty:
            raise ValueError("no valid MLR rows available")
        best_mlr = select_best_model_row(mlr_df)
        return {
            "mae": _safe_float(best_mlr.get("mae", float("nan"))),
            "rmse": _safe_float(best_mlr.get("rmse", float("nan"))),
            "r2": _safe_float(best_mlr.get("r2", float("nan"))),
            "pearson_r": _safe_float(best_mlr.get("pearson_r", float("nan"))),
        }
    except Exception:
        return {k: float("nan") for k in ("mae", "rmse", "r2", "pearson_r")}


_SHAPLEY_MERGE_LABEL_PREFIX = "shap_"


def _recompute_min_skill_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute min_skill_rmse for all non-baseline rows in df.

    Uses best baseline RMSE per subset_rank as primary lookup, with a
    subset_label fallback (stripping the shap_ prefix) for rows whose rank
    has no matching baseline (e.g. MLR rows at novel ranks, offset Shapley ranks).
    """
    if not {"subset_rank", "rmse", "model"}.issubset(df.columns):
        return df
    _is_bl = df["model"].apply(_is_baseline_model_value)
    _bl_by_rank = (
        df[_is_bl].groupby("subset_rank")["rmse"].min().rename("_best_bl_rmse")
    )
    df = df.join(_bl_by_rank, on="subset_rank")
    if "subset_label" in df.columns:
        _label_norm = df["subset_label"].astype(str).str.removeprefix(_SHAPLEY_MERGE_LABEL_PREFIX)
        _bl_by_label = (
            df[_is_bl]
            .groupby(df.loc[_is_bl, "subset_label"].astype(str))["rmse"]
            .min()
        )
        _missing = df["_best_bl_rmse"].isna()
        df.loc[_missing, "_best_bl_rmse"] = _label_norm[_missing].map(_bl_by_label)
    _ml_rmse = pd.to_numeric(df["rmse"], errors="coerce")
    _needs_skill = df["_best_bl_rmse"].notna() & (~_is_bl)
    df.loc[_needs_skill, "min_skill_rmse"] = (
        (df.loc[_needs_skill, "_best_bl_rmse"] - _ml_rmse[_needs_skill])
        / df.loc[_needs_skill, "_best_bl_rmse"]
    )
    df.drop(columns=["_best_bl_rmse"], inplace=True)
    return df


def _merge_shapley_into_final_metrics(plan: "DatasetPlan", df: pd.DataFrame) -> pd.DataFrame:
    """Append best Shapley sweep results into the Feature sweep final metrics DataFrame.

    Returns the updated DataFrame.  Idempotent: any previously-merged Shapley rows
    (identified by ``subset_label`` starting with *shap_*) are removed before re-appending.
    Does not write to disk or redraw the plot — the caller handles that.
    """
    shapley_csv = plan.dataset_dir / "forecasts" / "Shapley_sweeps" / "feature_sweep_final_metrics.csv"
    if not shapley_csv.exists():
        return df
    try:
        df_shapley = pd.read_csv(shapley_csv)
    except Exception:
        return df
    if df_shapley.empty:
        return df

    # Remove any previously-merged Shapley rows (idempotency).
    df_feat = df
    if "subset_label" in df_feat.columns:
        df_feat = df_feat[
            ~df_feat["subset_label"].astype(str).str.startswith(_SHAPLEY_MERGE_LABEL_PREFIX)
        ].copy()

    # Keep only non-baseline ML model rows from Shapley.
    if "model" in df_shapley.columns:
        df_shapley = df_shapley[~df_shapley["model"].apply(_is_baseline_model_value)].copy()
    if df_shapley.empty:
        return df

    # Deduplicate: skip Shapley rows whose (feature_tag, model) already exists
    # in the Feature sweep data (same feature set already evaluated in Stage 2).
    if "feature_tag" in df_feat.columns and "model" in df_feat.columns:
        existing_keys = set(
            zip(df_feat["feature_tag"].astype(str), df_feat["model"].astype(str))
        )
        keep_mask = [
            (str(row.get("feature_tag", "")), str(row.get("model", ""))) not in existing_keys
            for _, row in df_shapley.iterrows()
        ]
        df_shapley = df_shapley[keep_mask].copy()
    if df_shapley.empty:
        return df

    # Normalise target name in Shapley rows to match the feature sweep canonical target.
    # Shapley MLR rows store the raw output_column name (e.g. "Chromium (µg/L)_res")
    # while ML rows store the _derive_target_name() sanitised form.  Using the first
    # non-null target value from df_feat as the canonical name is reliable because all
    # ML model rows in df_feat share the same sanitised target.
    if "target" in df_feat.columns and "target" in df_shapley.columns:
        _canonical_targets = df_feat["target"].dropna().unique()
        if len(_canonical_targets) == 1:
            df_shapley["target"] = _canonical_targets[0]
        elif len(_canonical_targets) > 1:
            # Multiple target names in the feature sweep — pick the most common one.
            _canonical_target = df_feat["target"].dropna().mode().iloc[0]
            df_shapley["target"] = _canonical_target

    # Relabel to avoid colliding with Feature sweep subset_rank / subset_label.
    max_rank = 0
    if "subset_rank" in df_feat.columns:
        numeric_ranks = pd.to_numeric(df_feat["subset_rank"], errors="coerce").dropna()
        if not numeric_ranks.empty:
            max_rank = int(numeric_ranks.max())

    if "subset_label" in df_shapley.columns:
        df_shapley["subset_label"] = df_shapley["subset_label"].astype(str).apply(
            lambda lbl: lbl if lbl.startswith(_SHAPLEY_MERGE_LABEL_PREFIX)
            else _SHAPLEY_MERGE_LABEL_PREFIX + lbl
        )
    if "subset_rank" in df_shapley.columns:
        df_shapley["subset_rank"] = (
            pd.to_numeric(df_shapley["subset_rank"], errors="coerce") + max_rank
        )

    merged = pd.concat([df_feat, df_shapley], ignore_index=True)
    n_merged = len(df_shapley)
    print(
        f"[INFO] Merged {n_merged} Shapley sweep result row(s) for {plan.dataset_dir.name}"
    )
    return merged


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
        'subset_label': str(row.get('subset_label', '')),
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


def valid_selection_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    """Rows eligible for headline best-model/best-baseline selection."""
    return _filter_min_valid_independent(
        _filter_valid_rows(df),
        min_required=MIN_REQUIRED_VALID_INDEPENDENT,
    )


def select_best_row(df: "pd.DataFrame") -> "pd.Series | None":
    """Select the best row from already-valid candidates, or None."""
    if df is None or df.empty or "r2" not in df.columns:
        return None
    vals = pd.to_numeric(df["r2"], errors="coerce")
    out = df[np.isfinite(vals)].copy()
    if out.empty:
        return None
    return select_best_model_row(out)


def _row_identity_key(row: "pd.Series | None") -> tuple[str, str, str, str]:
    if row is None:
        return ("", "", "", "")
    return (
        str(row.get("model", "")).strip().lower(),
        str(row.get("feature_tag", "")),
        str(row.get("row_count", "")),
        str(row.get("subset_label", "")),
    )


def _selection_metric(row: "pd.Series", name: str) -> float:
    if name == "nrmse":
        val = _safe_float(row.get("nrmse", float("nan")))
        if np.isfinite(val):
            return val
        rmse = _safe_float(row.get("rmse", float("nan")))
        std = _safe_float(row.get("std_target", float("nan")))
        return float(rmse / std) if np.isfinite(rmse) and np.isfinite(std) and std > 0 else float("nan")
    return _safe_float(row.get(name, float("nan")))


def _display_model_id(value: object) -> str:
    return str(value).strip().lower()


def build_selection_record(plan: DatasetPlan, df: "pd.DataFrame", args: argparse.Namespace) -> "SelectionRecord | None":
    """Build the authoritative per-target selection record used by summary outputs."""
    if df is None or df.empty or "model" not in df.columns:
        return None
    target_name = _derive_target_name(plan.dataset_dir.name, args.dataset_prefix)
    target_df = df[df["target"].astype(str) == target_name].copy() if "target" in df.columns else df.copy()
    if target_df.empty:
        target_df = df.copy()
        print(f"[WARN] No rows match target_name={target_name!r} for {plan.dataset_dir.name}; using all rows for selection.")

    candidates = valid_selection_rows(target_df)
    best_model_row = select_best_row(candidates)
    if best_model_row is None:
        return None

    best_model_id = _display_model_id(best_model_row.get("model", ""))
    baseline_ids = configured_baseline_ids(args)
    is_best_baseline = best_model_id in set(baseline_ids)
    if is_best_baseline:
        best_baseline_row = best_model_row
    else:
        baseline_rows = candidates[candidates["model"].astype(str).str.strip().str.lower().isin(baseline_ids)].copy()
        best_baseline_row = select_best_row(baseline_rows)
        if best_baseline_row is None:
            print(f"[WARN] No configured baseline candidate for {plan.dataset_dir.name}; skipping headline selection.")
            return None

    best_baseline_id = _display_model_id(best_baseline_row.get("model", ""))
    same_row = _row_identity_key(best_model_row) == _row_identity_key(best_baseline_row)
    model_rmse = _selection_metric(best_model_row, "rmse")
    baseline_rmse = _selection_metric(best_baseline_row, "rmse")
    if same_row or is_best_baseline:
        skill = 0.0
    elif np.isfinite(model_rmse) and np.isfinite(baseline_rmse) and baseline_rmse > 0:
        skill = float(1.0 - model_rmse / baseline_rmse)
    else:
        skill = float("nan")

    return SelectionRecord(
        dataset=plan.dataset_dir.name,
        target=target_name,
        best_model_row=best_model_row,
        best_baseline_row=best_baseline_row,
        best_model_id=best_model_id,
        best_model_label=_display_model_type(best_model_id),
        best_baseline_id=best_baseline_id,
        best_baseline_label=_display_model_type(best_baseline_id),
        baseline_model_set=baseline_ids,
        best_model_is_configured_baseline=bool(is_best_baseline),
        best_model_equals_best_baseline=bool(same_row or is_best_baseline),
        skill_vs_best_baseline=skill,
    )


def _axis_uses_scientific_bar_annotations(ax) -> bool:
    """Return True when any finite nonzero bar height on *ax* is below 0.01 in magnitude."""
    heights = [
        rect.get_height()
        for rect in ax.patches
        if np.isfinite(rect.get_height()) and rect.get_height() != 0
    ]
    return any(abs(h) < 0.01 for h in heights)


def _format_bar_annotation_value(value: float, fmt: str, scientific: bool) -> str:
    """Format a bar annotation value using either fixed chart-wide scientific notation or *fmt*."""
    if scientific:
        s = f"{value:.2e}"
        mantissa, exp = s.split("e")
        return f"{mantissa}e{int(exp)}"
    return f"{value:{fmt}}"


def _annotate_bars_within_ylim(ax, bars, fmt: str, fontsize: int = 8) -> None:
    """Annotate bars only when the bar-top y value falls within current y-axis limits."""
    ymin, ymax = ax.get_ylim()
    yspan = float(ymax - ymin) if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
    pad = 0.01 * yspan
    use_scientific = _axis_uses_scientific_bar_annotations(ax)
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
            _format_bar_annotation_value(h, fmt, use_scientific),
            ha='center',
            va=va,
            fontsize=fontsize,
            rotation=90,
            clip_on=True,
        )


def _annotate_ml_bars(ax, bars, vals, ns, fmt: str, fontsize: int = 10) -> None:
    """Annotate bars with combined value and sample-count label, e.g. '1.23e-02, n=8'."""
    ymin, ymax = ax.get_ylim()
    yspan = float(ymax - ymin) if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
    pad = 0.02 * yspan
    use_scientific = _axis_uses_scientific_bar_annotations(ax)
    for bar, val, n in zip(bars, vals, ns):
        if not np.isfinite(val):
            continue
        h = bar.get_height()
        if h < ymin or h > ymax:
            continue
        n_str = f", n={int(n)}" if np.isfinite(n) else ""
        label = f"{_format_bar_annotation_value(val, fmt, use_scientific)}{n_str}"
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
            clip_on=False,
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
                    center_offset: float | None = None, annotate: bool = True,
                    fontsize: int = 8) -> list:
    """Draw a grouped bar chart and optionally annotate each bar with its value."""
    bar_groups = []
    n_series = len(data)
    if center_offset is None:
        center_offset = (n_series - 1) / 2.0
    for i, (vals, color, method) in enumerate(zip(data, colors, methods)):
        bars = ax.bar(x + (i - center_offset) * width, vals, width, label=method, color=color)
        bar_groups.append(bars)
    if len(x) > 0 and n_series > 0:
        cluster_half = n_series * width / 2.0
        edge_margin = 0.2 * width
        ax.set_xlim(x[0] - cluster_half - edge_margin, x[-1] + cluster_half + edge_margin)
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


def _expand_ylim_to_fit_annotations(ax, pad_pixels: float = 2.0, max_passes: int = 3) -> None:
    """Expand y-limits just enough so existing annotation texts fit inside the axes box."""
    texts = [txt for txt in ax.texts if txt.get_visible()]
    if not texts:
        return
    fig = ax.figure
    for _ in range(max_passes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax.get_window_extent(renderer=renderer)
        top_over = 0.0
        bottom_over = 0.0
        for txt in texts:
            bbox = txt.get_window_extent(renderer=renderer)
            top_over = max(top_over, bbox.y1 - ax_bbox.y1)
            bottom_over = max(bottom_over, ax_bbox.y0 - bbox.y0)
        if top_over <= 0 and bottom_over <= 0:
            break
        x_ref = 0.5 * (ax_bbox.x0 + ax_bbox.x1)
        y_lo, y_hi = ax.get_ylim()
        new_y_lo = y_lo
        new_y_hi = y_hi
        if bottom_over > 0:
            new_y_lo = ax.transData.inverted().transform((x_ref, ax_bbox.y0 - bottom_over - pad_pixels))[1]
        if top_over > 0:
            new_y_hi = ax.transData.inverted().transform((x_ref, ax_bbox.y1 + top_over + pad_pixels))[1]
        if new_y_lo == y_lo and new_y_hi == y_hi:
            break
        ax.set_ylim(new_y_lo, new_y_hi)


def _expand_ylims_to_fit_annotations(axes, pad_pixels: float = 2.0) -> None:
    """Apply annotation-aware y-limit expansion to one axis or a sequence of axes."""
    for ax in np.atleast_1d(axes):
        _expand_ylim_to_fit_annotations(ax, pad_pixels=pad_pixels)


def _finalize_stacked_figure(fig, axes, left: float = 0.30, right: float = 0.98, top: float = 0.97,
                             bottom: float = 0.12, hspace: float = 0.50) -> None:
    _style_stacked_axes(axes)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace)
    _expand_ylims_to_fit_annotations(axes)


def _resolve_summary_plot_dirs(summaries_dir: Path) -> tuple[Path, Path, Path]:
    combined_dir = (summaries_dir / "combined").resolve()
    individual_dir = (summaries_dir / "individual").resolve()
    evaluation_dir = (summaries_dir / "evaluation").resolve()
    for out_dir in (combined_dir, individual_dir, evaluation_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
    return combined_dir, individual_dir, evaluation_dir


def _read_mlr_k_cluster_rows(sweep_dir: Path, df: pd.DataFrame) -> list[dict]:
    """Recover per-k-cluster MLR rows from artifact directories not yet in df.

    Globs for mlr*_k## directories (e.g. mlr_k01, mlr_avg12_k03) that have an
    evaluation_summary.csv and config_evaluate_*.yml. Reads metrics from the
    test-kind summary row and looks up subset_rank from existing df rows that
    share the same feature_tag. Skips any (feature_tag, model) already in df.
    """
    import re as _re
    _mlr_models = {"mlr", "mlr_avg12", "mlr_avgall"}
    existing_keys: set[tuple[str, str]] = set()
    if "feature_tag" in df.columns and "model" in df.columns:
        existing_keys = set(
            zip(df["feature_tag"].astype(str), df["model"].astype(str).str.lower())
        )

    # Build feature_tag → subset_rank lookup from non-MLR rows.
    ftag_to_rank: dict[str, int] = {}
    ftag_to_std: dict[str, float] = {}
    ftag_to_dataset: dict[str, str] = {}
    ftag_to_target: dict[str, str] = {}
    ftag_to_rowcount: dict[str, object] = {}
    if "feature_tag" in df.columns and "subset_rank" in df.columns:
        non_mlr = df[~df["model"].astype(str).str.lower().isin(_mlr_models)]
        for ft, grp in non_mlr.groupby("feature_tag"):
            ranks = pd.to_numeric(grp["subset_rank"], errors="coerce").dropna()
            if not ranks.empty:
                ftag_to_rank[str(ft)] = int(ranks.min())
            stds = pd.to_numeric(grp.get("std_target", pd.Series(dtype=float)), errors="coerce").dropna()
            if not stds.empty:
                ftag_to_std[str(ft)] = float(stds.iloc[0])
            if "dataset" in grp.columns:
                ftag_to_dataset[str(ft)] = str(grp["dataset"].iloc[0])
            if "target" in grp.columns:
                ftag_to_target[str(ft)] = str(grp["target"].iloc[0])
            if "row_count" in grp.columns:
                ftag_to_rowcount[str(ft)] = grp["row_count"].iloc[0]

    rows: list[dict] = []
    _dir_pattern = _re.compile(r"^(mlr(?:_avg12|_avgall)?)_k(\d{2})$")
    for candidate in sorted(sweep_dir.iterdir()):
        if not candidate.is_dir():
            continue
        m = _dir_pattern.match(candidate.name)
        if m is None:
            continue
        model_name = m.group(1)
        subset_label = f"k{m.group(2)}"

        cfg_candidates = sorted(candidate.glob("config_evaluate_*.yml"))
        if not cfg_candidates:
            continue
        try:
            with open(cfg_candidates[0], "r", encoding="utf-8") as _f:
                _cfg = yaml.safe_load(_f)
        except Exception:
            continue
        _dcfg = _cfg.get("data", {})
        input_columns = list(_dcfg.get("input_columns") or [])
        output_columns = list(_dcfg.get("output_columns") or [])
        if not input_columns:
            continue

        # Use final selected features from model_config.json (post-MI/Lasso/VIF) when
        # available, falling back to spearman_kept_columns then input_columns.
        _model_cfg_path = candidate / "model_config.json"
        _sel_cols: list[str] = []
        if _model_cfg_path.exists():
            try:
                with open(_model_cfg_path, encoding="utf-8") as _mf:
                    _mc = json.load(_mf)
                for _tm in _mc.get("per_target_meta", []):
                    _sel_cols = _tm.get("selected_features") or []
                    if _sel_cols:
                        break
                if not _sel_cols:
                    for _tm in _mc.get("per_target_meta", []):
                        _sel_cols = _tm.get("spearman_kept_columns") or []
                        if _sel_cols:
                            break
            except Exception:
                pass
        effective_input_columns = list(_sel_cols) if _sel_cols else input_columns
        feature_tag = _feature_tag(tuple(sorted(effective_input_columns)))
        if (feature_tag, model_name) in existing_keys:
            continue

        subset_rank = ftag_to_rank.get(feature_tag)
        if subset_rank is None:
            subset_rank = int(m.group(2))

        summary_csv = candidate / "evaluation_summary.csv"
        if not summary_csv.exists():
            continue
        try:
            _sumdf = pd.read_csv(summary_csv)
        except Exception:
            continue

        # Prefer kind=="test", fall back to first non-baseline row.
        _test_rows = _sumdf[_sumdf.get("kind", pd.Series(dtype=str)).astype(str).str.lower() == "test"] if "kind" in _sumdf.columns else pd.DataFrame()
        if _test_rows.empty:
            _non_bl = _sumdf[~_sumdf["label"].astype(str).apply(_is_baseline_model_value)] if "label" in _sumdf.columns else pd.DataFrame()
            _test_rows = _non_bl if not _non_bl.empty else _sumdf

        if _test_rows.empty:
            continue
        srow = _test_rows.iloc[0]

        def _sfloat(key: str) -> float:
            return float(pd.to_numeric(srow.get(key, float("nan")), errors="coerce"))

        mae = _sfloat("mae")
        rmse = _sfloat("rmse")
        r2 = _sfloat("r2")
        pearson_r = _sfloat("pearson_r")
        n_test_independent = _sfloat("n_test_independent")
        n_test_valid = _sfloat("n_test_valid")
        n_samples = _sfloat("n_eval_points_finite") if "n_eval_points_finite" in srow.index else n_test_valid
        n_features = _sfloat("input_dim") if "input_dim" in srow.index else float(len(input_columns))
        target_dim = _sfloat("n_eval_outputs") if "n_eval_outputs" in srow.index else float(len(output_columns))

        if not (np.isfinite(mae) and np.isfinite(rmse)):
            continue

        std_target = ftag_to_std.get(feature_tag, float("nan"))
        nrmse = rmse / std_target if np.isfinite(std_target) and std_target > 0 else float("nan")

        row: dict = {
            "dataset": ftag_to_dataset.get(feature_tag, sweep_dir.parent.parent.name),
            "target": ftag_to_target.get(feature_tag, output_columns[0] if output_columns else ""),
            "subset_rank": subset_rank,
            "subset_label": subset_label,
            "feature_tag": feature_tag,
            "row_count": ftag_to_rowcount.get(feature_tag, float("nan")),
            "n_features": n_features,
            "model": model_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson_r": pearson_r,
            "std_target": std_target,
            "nrmse": nrmse,
            "n_test_independent": n_test_independent,
            "n_test_valid": n_test_valid,
            "n_test_evals": n_test_valid,
            "n_samples": n_samples,
            "input_dim": float(len(input_columns)),
            "target_dim": target_dim,
            "n_eval_raw_segments": n_test_independent,
        }
        for col in df.columns:
            if col not in row:
                row[col] = float("nan")
        rows.append(row)
        print(f"[INFO] Recovered {model_name} {subset_label} row from {candidate.name}: "
              f"R²={r2:.3f}, RMSE={rmse:.4f}")
    return rows


def _append_mlr_to_final_metrics(
    plan: "DatasetPlan",
    df: pd.DataFrame,
    best_row: "pd.Series",
    args: "argparse.Namespace",
) -> pd.DataFrame:
    """Fit MLR on train data, evaluate on test data, return updated metrics DataFrame.

    Uses the same train/test split as the best ML model for the dataset.
    Feature selection (MI + L1 + VIF) is performed independently on training data.
    Also writes per-variant output artifacts (evaluation_summary.csv, predictions.csv,
    model_config.json, boxplot.png, predictions.png, metrics_summary.png) matching
    the standard output format of other data-driven models.
    Does not write to disk or redraw the plot — the caller handles that.
    """
    variant_dir, eval_cfg_path, match_status = _find_best_variant_eval_config(plan, best_row)
    if eval_cfg_path is None or not eval_cfg_path.exists():
        print(f"[WARN] Cannot compute MLR for {plan.dataset_dir.name}: no eval config ({match_status})")
        return df

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
    subset_label = str(best_row.get("subset_label", "")).strip().lower()
    use_preselected_mlr_inputs = (
        subset_label.startswith(_SHAPLEY_MERGE_LABEL_PREFIX)
        or "shapley_sweeps" in str(eval_cfg_path).lower()
        or str(getattr(args, "sweep_namespace", "")).strip().lower() == "shapley_sweeps"
    )

    # MLR feature selection is fully independent of the ML feature sweep.
    # Load the FULL set of input columns from the surrogate (base) config,
    # not the reduced subset selected for the best ML variant.
    if use_preselected_mlr_inputs:
        input_columns = list(model_config.get("input_columns", []) or [])
    else:
        surrogate_cfg_path = _select_surrogate_config(plan.train_configs)
        surrogate_cfg = train_module.load_config(str(surrogate_cfg_path))
        input_columns = list(surrogate_cfg["data"]["input_columns"])

    # Remove only the s01/m01/l01 MLR rows (the variants we are about to re-evaluate).
    # MLR rows for other feature clusters (k## rows written by the main sweep) must be
    # preserved so they continue to appear in the CSV and summary plot.
    if "model" in df.columns:
        _mlr_model_mask = df["model"].astype(str).str.lower().isin({"mlr", "mlr_avg12", "mlr_avgall"})
        _variant_label_mask = df["subset_label"].astype(str).isin({"s01", "m01", "l01"}) if "subset_label" in df.columns else _mlr_model_mask
        df = df[~(_mlr_model_mask & _variant_label_mask)].copy()

    mlr_selection_config = None
    mlr_use_spearman_prefilter = True
    if use_preselected_mlr_inputs:
        mlr_selection_config = {
            "use_mutual_info": False,
            "use_lasso": False,
            "deduplicate_threshold": None,
            "vif_threshold": float("inf"),
        }
        mlr_use_spearman_prefilter = False

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
        fault_tolerant=True,
        input_aggregation=input_aggregation,
    )

    train_samples = eval_module.load_split_samples(**load_kw, split_file="train_files.txt")
    test_samples = eval_module.load_split_samples(**load_kw, split_file="test_files.txt")

    if len(train_samples) < 3 or len(test_samples) < 1:
        print(f"[WARN] Insufficient samples for MLR ({plan.dataset_dir.name}): "
              f"train={len(train_samples)}, test={len(test_samples)}")
        return df

    from utils.mlr import MLR_VARIANTS as _MLR_VARIANTS

    mlr_results = _run_mlr_variants_on_existing_split(
        train_samples=train_samples,
        test_samples=test_samples,
        feature_names=input_columns,
        min_test_independent=MIN_REQUIRED_VALID_INDEPENDENT,
        model_context=plan.dataset_dir.name,
        use_preselected_feature_set=use_preselected_mlr_inputs,
    )
    results_by_name = {
        str(res.get("variant", {}).get("model_name", "")): res
        for res in mlr_results
    }
    rows_to_append = []

    for _mlr_v in _MLR_VARIANTS:
        _v_name = _mlr_v["model_name"]
        _v_prefix = _mlr_v["dir_prefix"]
        _v_subset = _mlr_v["subset_label"]
        res = results_by_name.get(_v_name, {})
        if res.get("error") is not None:
            print(f"[WARN] {_v_name} evaluation failed for {plan.dataset_dir.name}: {res['error']}")
            continue

        predictions = res.get("preds")
        targets = res.get("targets")
        meta = res.get("meta", [])
        train_used = res.get("train_samples", train_samples)
        test_used = res.get("test_samples", test_samples)

        if int(res.get("n_samples", 0)) < 1:
            print(
                f"[WARN] {_v_name} produced no finite predictions for {plan.dataset_dir.name}; "
                f"failure_reasons={res.get('failure_reasons') or ['<none>']} "
                f"fallback_modes={res.get('fallback_modes') or ['<none>']}"
            )
            continue

        mlr_ftag = str(res.get("feature_tag", _feature_tag(tuple(sorted(input_columns)))))
        existing_mlr_ranks = []
        if "feature_tag" in df.columns:
            existing_mlr_ranks = df.loc[
                df["feature_tag"].astype(str) == mlr_ftag, "subset_rank"
            ].dropna().unique().tolist()
        if existing_mlr_ranks:
            mlr_rank = int(existing_mlr_ranks[0])
        elif "subset_rank" in df.columns and not df["subset_rank"].dropna().empty:
            mlr_rank = int(df["subset_rank"].dropna().max()) + 1
        else:
            mlr_rank = 1

        std_target = float(res.get("std_target_empirical", float("nan")))
        nrmse_val = float(res["rmse"]) / std_target if np.isfinite(std_target) and std_target > 0 else float("nan")
        n_selected = sum(m.get("n_selected", 0) for m in meta) / max(len(meta), 1)
        n_eval_raw_segments = int(res["n_test_independent"])
        mlr_row = {
            "dataset": best_row.get("dataset", plan.dataset_dir.name),
            "target": best_row.get("target", ""),
            "subset_rank": mlr_rank,
            "subset_label": _v_subset,
            "feature_tag": mlr_ftag,
            "row_count": best_row.get("row_count", ""),
            "n_features": n_selected,
            "model": _v_name,
            "mae": float(res["mae"]),
            "rmse": float(res["rmse"]),
            "r2": float(res["r2"]),
            "pearson_r": float(res["pearson_r"]),
            "std_target": std_target,
            "nrmse": nrmse_val,
            "n_test_independent": int(res["n_test_independent"]),
            "n_test_valid": int(res["n_test_valid"]),
            "n_test_evals": int(res["n_test_valid"]),
            "n_samples": int(res["n_samples"]),
            "input_dim": float(len(input_columns)),
            "target_dim": float(len(output_columns)),
            "n_eval_raw_segments": n_eval_raw_segments,
        }
        for col in df.columns:
            if col not in mlr_row:
                mlr_row[col] = float("nan")
        for field in ["n_test_valid", "nrmse", "r2", "mae", "rmse", "pearson_r", "std_target", "n_eval_raw_segments"]:
            val = mlr_row.get(field, None)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                print(f"[WARN] {_v_name} row missing or invalid value for '{field}' in {plan.dataset_dir.name}")
        rows_to_append.append(mlr_row)
        print(f"[INFO] Appended {_v_name} row for {plan.dataset_dir.name}: "
              f"R²={float(res['r2']):.3f}, RMSE={float(res['rmse']):.4f}, n_features={n_selected:.0f}")

        sweep_dir = _forecast_sweeps_dir(plan.dataset_dir)
        mlr_dir = _write_mlr_artifacts(
            output_dir=sweep_dir,
            dataset_dir=plan.dataset_dir,
            subset_label=_v_subset,
            data_dir=str(data_cfg["data_dir"]),
            sample_subdir=data_cfg.get("sample_subdir", "samples"),
            input_columns=input_columns,
            output_columns=output_columns,
            input_row_1=model_config.get("input_row_1"),
            input_row_2=model_config.get("input_row_2"),
            output_rows=output_rows,
            input_aggregation=input_aggregation,
            train_samples=train_used,
            test_samples=test_used,
            preds=predictions,
            targets=targets,
            per_target_meta=meta,
            split_source_dir=split_base_dir,
            ref_cfg=cfg,
            ref_cfg_path=eval_cfg_path,
            ref_data_cfg=data_cfg,
            model_config_extra={
                "spearman_kept_columns": sorted(res.get("effective_feature_names") or input_columns),
                **(res.get("feature_selection_extra") or {}),
            },
            model_prefix=_v_prefix,
        )
        print(f"[INFO] Wrote {_v_name} artifacts to {mlr_dir}")

    # Recover any per-k-cluster MLR rows written by the sweep but not yet in df.
    sweep_dir = _forecast_sweeps_dir(plan.dataset_dir)
    rows_to_append.extend(_read_mlr_k_cluster_rows(sweep_dir, df))

    if rows_to_append:
        df = pd.concat([df, pd.DataFrame(rows_to_append)], ignore_index=True)
    return df


def _normalize_ml_model_display(val: str) -> str:
    """Map raw model string to ML_COMPARISON_MODEL_TYPES display key."""
    key = str(val).strip().lower()
    if 'xgb' in key:
        return 'XGB'
    if 'transformer' in key:
        return 'Trans.'
    if 'gp' in key:
        return 'GP'
    if key == 'mlr':
        return 'MLR'
    if key == 'mlr_avg12':
        return 'MLR12'
    if key == 'mlr_avgall':
        return 'MLRall'
    if 'mlr' in key:
        return 'MLR'
    return None


def _display_model_type(val: object) -> str:
    key = str(val).strip().lower()
    if 'xgb' in key:
        return 'XGB'
    if 'transformer' in key:
        return 'Trans.'
    if key == 'gp_regressor' or key == 'gpregressor' or key == 'gp':
        return 'GP'
    if key == 'mlr':
        return 'MLR'
    if key == 'mlr_avg12':
        return 'MLR-12'
    if key == 'mlr_avgall':
        return 'MLR-All'
    if key == 'naive':
        return 'Naive'
    if key == 'seasonal':
        return 'Seasonal'
    if key == 'linear':
        return 'Linear'
    return key.title()


def _select_best_ml_row_for_eval_figure(df: pd.DataFrame, args: argparse.Namespace) -> "pd.Series | None":
    """Return the best ML-category row for the evaluation figure under the current flags."""
    if df.empty or "model" not in df.columns:
        return None
    model_series = df["model"].astype(str)
    ml_selection = str(getattr(args, "ml_selection", "best")).strip().lower()
    treat_mlr_as_baseline = bool(getattr(args, "treat_mlr_as_baseline", False))

    if ml_selection == "xgb":
        ml_mask = model_series.apply(lambda v: _normalize_ml_model_display(v) == "XGB")
    else:
        ml_mask = ~model_series.apply(_is_baseline_model_value)
        if treat_mlr_as_baseline:
            ml_mask = ml_mask & ~model_series.apply(_is_mlr_model_value)

    ml_rows_raw = df[ml_mask].copy()
    if ml_rows_raw.empty:
        return None

    ml_rows = _filter_min_valid_independent(
        _filter_valid_rows(ml_rows_raw),
        min_required=MIN_REQUIRED_VALID_INDEPENDENT,
    )
    if ml_rows.empty:
        # Fallback: keep the requested model family constraint, but relax the
        # stricter post-process validity gates so an existing finite-R² XGB row
        # is still surfaced in the figure instead of silently disappearing.
        ml_rows = ml_rows_raw.copy()
        ml_rows = ml_rows[pd.to_numeric(ml_rows.get("r2"), errors="coerce").notna()].copy()
    if ml_rows.empty:
        return None
    best_row = select_best_model_row(ml_rows)
    if ml_selection == "xgb" and _normalize_ml_model_display(best_row.get("model", "")) != "XGB":
        print(f"[WARN] Expected XGB row for evaluation figure, got {best_row.get('model', '')!r}; discarding.")
        return None
    return best_row


def _select_best_baseline_row_for_eval_figure(df: pd.DataFrame, args: argparse.Namespace) -> "pd.Series | None":
    """Return the best baseline-category row for the evaluation figure under the current flags."""
    if df.empty or "model" not in df.columns:
        return None
    model_series = df["model"].astype(str)
    baseline_mask = model_series.apply(_is_baseline_model_value)
    if bool(getattr(args, "treat_mlr_as_baseline", False)):
        baseline_mask = baseline_mask | model_series.apply(_is_mlr_model_value)
    baseline_rows_raw = df[baseline_mask].copy()
    if baseline_rows_raw.empty:
        return None
    baseline_rows = _filter_min_valid_independent(
        _filter_valid_rows(baseline_rows_raw),
        min_required=MIN_REQUIRED_VALID_INDEPENDENT,
    )
    if baseline_rows.empty:
        baseline_rows = baseline_rows_raw.copy()
        baseline_rows = baseline_rows[pd.to_numeric(baseline_rows.get("r2"), errors="coerce").notna()].copy()
    if baseline_rows.empty:
        return None
    return select_best_model_row(baseline_rows)


def _annotate_bars_with_model_labels(
    ax,
    bars,
    vals,
    model_labels,
    fmt: str = ".2f",
    fontsize: int = 10,
) -> None:
    """Annotate bars with value plus model-type label."""
    ymin, ymax = ax.get_ylim()
    yspan = float(ymax - ymin) if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
    pad = 0.02 * yspan
    for bar, val, model_label in zip(bars, vals, model_labels):
        if not np.isfinite(val):
            continue
        h = bar.get_height()
        if h < ymin or h > ymax:
            continue
        label = f"{float(val):{fmt}}, {model_label}"
        anchor_y = max(float(h), 0.0)
        y_txt = anchor_y + pad
        va = "bottom"
        if y_txt > (ymax - pad):
            y_txt = anchor_y - pad
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_txt,
            label,
            ha="center",
            va=va,
            fontsize=fontsize,
            rotation=90,
            clip_on=False,
        )


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

        # Apply the same validity filters, then remove the configured baseline
        # set so this remains a model-family comparison rather than a headline
        # best-model plot.
        valid_df = valid_selection_rows(df)
        baseline_ids = set(configured_baseline_ids(args))
        valid_df = valid_df[~valid_df["model"].astype(str).str.strip().str.lower().isin(baseline_ids)].copy()
        valid_df = _filter_best_model_candidates(valid_df, getattr(args, "ml_selection", "best"))
        if valid_df.empty:
            continue

        # Determine n_samples column
        if 'n_test_valid' in valid_df.columns:
            n_col = 'n_test_valid'
        elif 'n_test_independent' in valid_df.columns:
            n_col = 'n_test_independent'
        else:
            n_col = None

        selection = build_selection_record(plan, df, args)
        best_baseline_rmse = (
            _selection_metric(selection.best_baseline_row, "rmse")
            if selection is not None else float("nan")
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

    # Determine target order: sort by best skill_vs_best per dataset (descending)
    best_skill_per_dataset = (
        comp_df.groupby('dataset')['skill_vs_best'].max().sort_values(ascending=False)
    )
    ordered_datasets = list(best_skill_per_dataset.index)
    # Deduplicated target labels in the same order
    seen: set = set()
    ordered_targets: list[tuple[str, str]] = []  # (dataset, target_label)
    for ds in ordered_datasets:
        lbl = comp_df.loc[comp_df['dataset'] == ds, 'target_label'].iloc[0]
        if ds not in seen:
            seen.add(ds)
            ordered_targets.append((ds, lbl))

    # Determine within-cluster model order: sort by mean skill_vs_best (descending)
    mean_skill_per_model = (
        comp_df.groupby('model_display')['skill_vs_best']
        .mean()
        .reindex(ML_COMPARISON_MODEL_TYPES)
        .fillna(-np.inf)
        .sort_values(ascending=False)
    )
    ordered_model_types = list(mean_skill_per_model.index)

    n_targets = len(ordered_targets)
    x = np.arange(n_targets)
    n_models = len(ordered_model_types)
    # Use contiguous bars within a cluster with an inter-cluster gap equal to
    # one bar width, so spacing never exceeds the width of a column.
    width = 1.0 / max(n_models + 1, 1)
    offsets = np.array([(i - (n_models - 1) / 2) for i in range(n_models)])
    _FS = 14  # unified font size for all text elements

    metric_specs = [
        ('rmse',          'RMSE',                    '.2e', False, 'ascending'),
        ('nrmse',         'nRMSE',                   '.2e', False, 'ascending'),
        ('r2',            'R²',                      '.2f', False, 'descending'),
        ('skill_vs_best', 'Skill vs. Best Baseline', '.2f', True, 'descending'),
    ]
    file_names = {
        'rmse':          'ml_comparison_rmse.png',
        'nrmse':         'ml_comparison_nrmse.png',
        'r2':            'ml_comparison_r2.png',
        'skill_vs_best': 'ml_comparison_skill_vs_best_baseline.png',
    }

    for metric_key, ylabel, fmt, add_hline, sort_order in metric_specs:
        # Sort clusters based on metric and sort_order
        # For ascending: prefer lowest positive value
        # For descending: prefer highest value
        cluster_sort_vals = {}
        for cluster_idx, ds in enumerate(ordered_datasets):
            cluster_data = comp_df[comp_df['dataset'] == ds][metric_key]
            valid_vals = cluster_data.dropna().to_numpy(dtype=float)
            valid_vals = valid_vals[np.isfinite(valid_vals)]

            if sort_order == 'ascending':
                # For RMSE/nRMSE: prefer lowest positive value
                positive_vals = valid_vals[valid_vals > 0]
                if len(positive_vals) > 0:
                    cluster_sort_vals[ds] = np.min(positive_vals)
                else:
                    cluster_sort_vals[ds] = np.inf
            else:
                # For R2/skill: prefer highest value
                if len(valid_vals) > 0:
                    cluster_sort_vals[ds] = np.max(valid_vals)
                else:
                    cluster_sort_vals[ds] = -np.inf

        # Re-sort ordered_targets based on cluster sort values
        sorted_targets = sorted(ordered_targets, key=lambda x: cluster_sort_vals.get(x[0], np.inf if sort_order == 'ascending' else -np.inf), reverse=(sort_order == 'descending'))

        fig, ax = plt.subplots(figsize=(max(8, n_targets * 0.72 + 1.6), 6))
        # Build proxy Patch handles upfront so labels are always correct.
        legend_handles = [
            matplotlib.patches.Patch(facecolor=ML_COMPARISON_COLORS[m], label=m)
            for m in ordered_model_types
        ]
        pending_annotations = []

        # Use sorted_targets instead of ordered_targets for this metric
        sorted_x = np.arange(len(sorted_targets))

        for mi, model_display in enumerate(ordered_model_types):
            color = ML_COMPARISON_COLORS[model_display]
            vals = []
            ns = []
            bar_x = []
            for ti, (ds, _lbl) in enumerate(sorted_targets):
                row_match = comp_df[
                    (comp_df['dataset'] == ds) & (comp_df['model_display'] == model_display)
                ]
                if row_match.empty or not np.isfinite(_safe_float(row_match.iloc[0][metric_key])):
                    continue
                bar_x.append(sorted_x[ti] + offsets[mi] * width)
                vals.append(_safe_float(row_match.iloc[0][metric_key]))
                ns.append(_safe_float(row_match.iloc[0]['n_samples']))

            if not bar_x:
                continue

            bars = ax.bar(bar_x, vals, width, color=color)
            pending_annotations.append((bars, vals, ns))

        if add_hline:
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

        if metric_key in {'r2', 'skill_vs_best'}:
            ax.set_ylim(-1.1, 1.1)
        else:
            # Constrain y-axis lower limit: never below -1; if all bars positive, floor at 0.
            ymin_cur, ymax_cur = ax.get_ylim()
            all_metric_vals = comp_df[metric_key].dropna().to_numpy(dtype=float)
            all_metric_vals = all_metric_vals[np.isfinite(all_metric_vals)]
            min_val = float(np.min(all_metric_vals)) if all_metric_vals.size else 0.0
            if min_val >= 0.0:
                ax.set_ylim(bottom=-0.1, top=ymax_cur * 1.1)
            else:
                ax.set_ylim(bottom=max(ymin_cur, -1.0), top=ymax_cur * 1.1)

        for bars, vals, ns in pending_annotations:
            _annotate_ml_bars(ax, bars, vals, ns, fmt)

        # Tight horizontal bounds with only a small edge margin beyond the outer bars.
        cluster_half = (n_models - 1) / 2 * width + width / 2
        if len(sorted_targets) > 0:
            ax.set_xlim(sorted_x[0] - cluster_half - 0.2 * width, sorted_x[-1] + cluster_half + 0.2 * width)

        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.tick_params(axis='y', labelsize=_FS)
        ax.set_xticks(sorted_x)
        ax.set_xticklabels(
            [lbl for _ds, lbl in sorted_targets],
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
            ncol=len(ordered_model_types),
            frameon=False,
            fontsize=_FS,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        _expand_ylim_to_fit_annotations(ax)
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
    if model_key in {"mlr", "mlr_avg12", "mlr_avgall"}:
        return True, None
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
        _rc = row.get("row_count")
        row_count = int(_rc) if _rc is not None and pd.notna(_rc) else None
    except Exception:
        row_count = None
    feature_tag = str(row.get("feature_tag", ""))

    model_key = _normalize_model_key(str(row.get("model", "")))
    output_dir = _forecast_sweeps_dir(plan.dataset_dir)
    subset_label = str(row.get("subset_label", "")).strip().lower()

    # If this row was merged from the Shapley sweep, its artifacts live in the
    # Shapley namespace and the original subset_label has no "shap_" prefix.
    if subset_label.startswith(_SHAPLEY_MERGE_LABEL_PREFIX):
        output_dir = plan.dataset_dir / "forecasts" / "Shapley_sweeps"
        subset_label = subset_label[len(_SHAPLEY_MERGE_LABEL_PREFIX):]

    # Scan all eval config files in the output directory and match on the
    # model type recorded inside each config.  This avoids fragile directory-
    # name pattern matching that breaks when naming conventions differ (e.g.
    # MLR dirs are named ``mlr_s01`` while ML dirs are named
    # ``gp_01_r167_f6_xxx_k01``).
    exact_match: tuple[Path, Path] | None = None
    substring_match: tuple[Path, Path] | None = None
    label_fallback: tuple[Path, Path] | None = None
    for eval_cfg in sorted(output_dir.glob("*/config_evaluate_*.yml")):
        variant_dir = eval_cfg.parent
        # Quick filter: the subset_label must appear at the end of the dir name.
        if subset_label and not variant_dir.name.endswith(f"_{subset_label}"):
            continue
        try:
            cfg = train_module.load_config(str(eval_cfg))
        except Exception:
            continue
        cfg_keys = [
            _normalize_model_key(str(cfg.get("model_type", ""))),
            _normalize_model_key(str(cfg.get("model_name", ""))),
        ]
        # Prefer exact model_type match over substring containment.
        if model_key and any(model_key == k for k in cfg_keys if k):
            exact_match = (variant_dir, eval_cfg)
            break
        if model_key and not substring_match and any(model_key in k or k in model_key for k in cfg_keys if k):
            substring_match = (variant_dir, eval_cfg)
        if label_fallback is None:
            label_fallback = (variant_dir, eval_cfg)

    if exact_match is not None:
        return exact_match[0], exact_match[1], "exact_match"
    if substring_match is not None:
        return substring_match[0], substring_match[1], "exact_match"
    if label_fallback is not None:
        return label_fallback[0], label_fallback[1], "fallback_variant_mismatch"
    return None, None, "missing_variant_dir"


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


def _gp_interval_metrics(pred_means, pred_stds, targets, alpha: float = 0.1) -> dict:
    """Compute prediction interval metrics using GP predictive std directly.

    Constructs symmetric intervals as mean ± z * std where z is the normal
    quantile for the specified coverage (1 - alpha). Returns the same keys as
    _interval_proxy_metrics except q_abs_resid (replaced by gp_z_score).
    Returns NaN for all metrics when GP std is unavailable.
    """
    nan_result = {
        "gp_picp": float("nan"),
        "gp_nominal_coverage": float("nan"),
        "gp_coverage_gap": float("nan"),
        "gp_coverage_deficit": float("nan"),
        "gp_mpiw": float("nan"),
        "gp_nmpiw": float("nan"),
        "gp_interval_score": float("nan"),
        "gp_z_score": float("nan"),
        "gp_n_points": 0,
    }
    if pred_stds is None:
        return nan_result

    pf = np.asarray(pred_means, dtype=float).reshape(-1)
    sf = np.asarray(pred_stds,  dtype=float).reshape(-1)
    tf = np.asarray(targets,    dtype=float).reshape(-1)
    mask = np.isfinite(pf) & np.isfinite(sf) & (sf >= 0) & np.isfinite(tf)
    if not np.any(mask):
        return nan_result

    alpha = float(min(max(alpha, 1e-6), 0.999999))
    if scipy_stats is not None:
        z = float(scipy_stats.norm.ppf(1.0 - alpha / 2))
    else:
        # Fallback: approximate z for common alpha values
        z = float(math.sqrt(2) * math.erfc(alpha) if hasattr(math, "erfc") else 1.6449)

    lower = pf[mask] - z * sf[mask]
    upper = pf[mask] + z * sf[mask]
    covered = (tf[mask] >= lower) & (tf[mask] <= upper)
    picp = float(np.mean(covered)) if covered.size else float("nan")
    nominal = float(1.0 - alpha)
    gap = float(picp - nominal) if np.isfinite(picp) else float("nan")
    deficit = float(max(0.0, nominal - picp)) if np.isfinite(picp) else float("nan")
    mpiw = float(np.mean(upper - lower))
    std_t = float(np.std(tf[mask], ddof=1)) if np.sum(mask) > 1 else float("nan")
    nmpiw = float(mpiw / std_t) if np.isfinite(mpiw) and np.isfinite(std_t) and std_t > 0 else float("nan")
    penalties = (2.0 / alpha) * ((lower - tf[mask]) * (tf[mask] < lower) + (tf[mask] - upper) * (tf[mask] > upper))
    interval_score = float(np.mean((upper - lower) + penalties)) if penalties.size else float("nan")
    return {
        "gp_picp": picp,
        "gp_nominal_coverage": nominal,
        "gp_coverage_gap": gap,
        "gp_coverage_deficit": deficit,
        "gp_mpiw": mpiw,
        "gp_nmpiw": nmpiw,
        "gp_interval_score": interval_score,
        "gp_z_score": z,
        "gp_n_points": int(np.sum(mask)),
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

    _is_mlr = model_type in {"mlr", "mlr_avg12", "mlr_avgall"}

    train_samples = None
    if model_type == "gp_regressor" or _is_mlr:
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

    gp_pred_var = None
    if _is_mlr:
        from utils.mlr import evaluate_mlr as _eval_mlr, MLR_VARIANTS as _MLR_VARIANTS
        _agg_mode = "last"
        for _v in _MLR_VARIANTS:
            if _v["model_name"] == model_type:
                _agg_mode = _v["aggregation_mode"]
                break
        _mlr_preds, _mlr_tgts, _mlr_meta = _eval_mlr(
            train_samples,
            test_samples,
            feature_names=input_columns,
            aggregation_mode=_agg_mode,
        )
        y_test = np.array([s[1].flatten() for s in test_samples], dtype=float)
        X_test = np.array([s[0].flatten() for s in test_samples], dtype=float)
        pred_model = _safe_as_2d(_mlr_preds)
    else:
        if model_type == "transformer":
            X_test = np.array([s[0] for s in test_samples], dtype=float)
            y_test = np.array([s[1] for s in test_samples], dtype=float)
        else:
            X_test = np.array([s[0].flatten() for s in test_samples], dtype=float)
            y_test = np.array([s[1].flatten() for s in test_samples], dtype=float)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = eval_module.load_model(model_type, data_cfg, split_cfg, model_name, model_config, device, train_samples, config_dir)

        if model_type == "gp_regressor":
            _gp_result = eval_module._predict_gp_bundle(model, X_test, device)
            pred_model = _gp_result["mean"]
            gp_pred_var = _gp_result["variance"]
        elif model_type == "transformer":
            pred_model = model(torch.tensor(X_test, dtype=torch.float32, device=device)).detach().cpu().numpy()
        elif model_type == "xgb_regressor":
            out_dim = y_test.shape[1] if y_test.ndim > 1 else 1
            pred_model = model.predict(X_test).reshape(-1, out_dim)
        else:
            raise ValueError(f"Unsupported model_type for inference: {model_type}")

    historic = eval_cfg.get("historic_path")
    sample_subdir = data_cfg.get("sample_subdir", "samples")
    baseline_preds = {}
    if not historic:
        # MLR eval configs may lack historic_path; fill with NaN baselines.
        for label in ("naive", "seasonal", "linear"):
            baseline_preds[label] = np.full_like(_safe_as_2d(y_test), np.nan, dtype=float)
    else:
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
        "gp_pred_var": gp_pred_var,
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

    if str(model_type).strip().lower() in {"mlr", "mlr_avg12", "mlr_avgall"}:
        return {
            "stability_status":           "deterministic_skipped",
            "stability_skip_reason":      "deterministic_mlr",
            "stability_r2_mean":          _safe_float(m0["r2"]),
            "stability_r2_std":           0.0,
            "stability_r2_cv":            0.0,
            "stability_r2_lcb":           _safe_float(m0["r2"]),
            "stability_rmse_cv":          0.0,
            "stability_r2_ensemble":      _safe_float(m0["r2"]),
            "stability_rmse_ensemble":    _safe_float(m0["rmse"]),
            "stability_mae_ensemble":     _safe_float(m0["mae"]),
            "stability_ensemble_benefit": 0.0,
            "stability_n_replicates":     1,
            "stability_n_successful":     1,
            "stability_cv_threshold":     cv_thr,
            "gate_stability":             True,
        }

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
                pred_r = _safe_as_2d(eval_module._predict_gp_bundle(rep_model, X_test, device)["mean"])
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


def _neutral_self_comparison_evidence(selection: "SelectionRecord | None", args: argparse.Namespace) -> dict:
    """Evidence payload for a best model that is also the configured best baseline."""
    label = selection.best_baseline_label if selection is not None else ""
    return {
        "evidence_status": "ok",
        "evidence_variant_resolution": "self_baseline",
        "evidence_variant_dir": "",
        "evidence_comparison": "best_model_equals_best_baseline",
        "best_baseline_evidence_model": label,
        "skill_vs_best_baseline": 0.0,
        "bootstrap_skill_mean_vs_best_baseline": 0.0,
        "lcb95_skill_vs_best_baseline": 0.0,
        "bootstrap_prob_skill_gt0_vs_best_baseline": 0.0,
        "bootstrap_rmse_diff_mean_vs_best_baseline": 0.0,
        "bootstrap_rmse_diff_ci05_vs_best_baseline": 0.0,
        "bootstrap_rmse_diff_ci95_vs_best_baseline": 0.0,
        "bootstrap_r2_diff_mean_vs_best_baseline": 0.0,
        "effect_median_ae_diff_vs_best_baseline": 0.0,
        "effect_mean_ae_diff_vs_best_baseline": 0.0,
        "effect_cohen_d_ae_diff_vs_best_baseline": 0.0,
        "dm_stat_vs_best_baseline": 0.0,
        "dm_p_vs_best_baseline": 1.0,
        "dm_q_vs_best_baseline": 1.0,
        "wilcoxon_stat_vs_best_baseline": 0.0,
        "wilcoxon_p_vs_best_baseline": 1.0,
        "wilcoxon_q_vs_best_baseline": 1.0,
        "sign_wins_vs_best_baseline": 0.0,
        "sign_win_rate_vs_best_baseline": 0.5,
        "sign_p_vs_best_baseline": 1.0,
        "sign_q_vs_best_baseline": 1.0,
        "picp_delta_vs_best_baseline": 0.0,
        "nmpiw_delta_vs_best_baseline": 0.0,
        "interval_score_delta_vs_best_baseline": 0.0,
        "gate_min_raw_vs_best_baseline": True,
        "gate_prob_vs_best_baseline": False,
        "gate_lcb_vs_best_baseline": False,
        "gate_dm_vs_best_baseline": False,
        "gate_wilcoxon_vs_best_baseline": False,
        "gate_sign_vs_best_baseline": False,
        "gate_coverage_vs_best_baseline": True,
        "gate_dm_q_vs_best_baseline": False,
        "gate_wilcoxon_q_vs_best_baseline": False,
        "gate_sign_q_vs_best_baseline": False,
        "evidence_score_vs_best_baseline": 1,
        "evidence_score_overall_min": 1,
        "evidence_score_overall_mean": 1.0,
        "interval_alpha": float(getattr(args, "interval_alpha", 0.1)),
        "evidence_alpha": float(getattr(args, "evidence_alpha", 0.05)),
        "bootstrap_mode": str(getattr(args, "bootstrap_mode", "iid")),
        "bootstrap_block_len": int(getattr(args, "bootstrap_block_len", 3)),
    }


def _compute_statistical_evidence(
    plan: DatasetPlan,
    best_row: "pd.Series",
    args: argparse.Namespace,
    best_baseline_row: "pd.Series | None" = None,
    selection: "SelectionRecord | None" = None,
) -> dict:
    evidence: dict[str, float | str | int | bool] = {}
    is_self_baseline = bool(selection is not None and selection.best_model_equals_best_baseline)
    variant_dir, eval_cfg_path, resolve_status = _find_best_variant_eval_config(plan, best_row)
    evidence["evidence_variant_resolution"] = str(resolve_status)
    evidence["evidence_variant_dir"] = str(variant_dir) if variant_dir is not None else ""
    if resolve_status == "fallback_variant_mismatch":
        evidence["evidence_status"] = "variant_mismatch"
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
    gp_pred_var = payload.get("gp_pred_var")
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

    # GP-distributional prediction intervals (only populated for GP models)
    gp_pred_std = np.sqrt(np.maximum(gp_pred_var[:n_rows, :], 0.0)) if gp_pred_var is not None else None
    gp_int = _gp_interval_metrics(pred_model, gp_pred_std, y_test, alpha=interval_alpha)
    evidence["model_gp_picp"] = _safe_float(gp_int["gp_picp"])
    evidence["model_gp_nominal_coverage"] = _safe_float(gp_int["gp_nominal_coverage"])
    evidence["model_gp_coverage_gap"] = _safe_float(gp_int["gp_coverage_gap"])
    evidence["model_gp_coverage_deficit"] = _safe_float(gp_int["gp_coverage_deficit"])
    evidence["model_gp_nmpiw"] = _safe_float(gp_int["gp_nmpiw"])
    evidence["model_gp_interval_score"] = _safe_float(gp_int["gp_interval_score"])

    if is_self_baseline:
        evidence.update(_neutral_self_comparison_evidence(selection, args))
        return evidence

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

    best_baseline_id = ""
    if selection is not None:
        best_baseline_id = selection.best_baseline_id
        evidence["best_baseline_evidence_model"] = selection.best_baseline_label
    elif best_baseline_row is not None:
        best_baseline_id = _display_model_id(best_baseline_row.get("model", ""))
        evidence["best_baseline_evidence_model"] = _display_model_type(best_baseline_id)

    if best_baseline_id in BASELINE_ORDER:
        src_prefix = f"vs_{best_baseline_id}"
        alias_names = [
            "dm_stat", "dm_p", "dm_q",
            "wilcoxon_stat", "wilcoxon_p", "wilcoxon_q",
            "sign_wins", "sign_win_rate", "sign_p", "sign_q",
            "skill",
            "effect_median_ae_diff", "effect_mean_ae_diff", "effect_cohen_d_ae_diff",
            "bootstrap_n", "bootstrap_skill_mean", "bootstrap_skill_ci05", "bootstrap_skill_ci95",
            "bootstrap_prob_skill_gt0", "bootstrap_rmse_diff_mean",
            "bootstrap_rmse_diff_ci05", "bootstrap_rmse_diff_ci95", "bootstrap_r2_diff_mean",
            "lcb95_skill", "picp_delta", "nmpiw_delta", "interval_score_delta",
            "gate_min_raw", "gate_prob", "gate_lcb", "gate_dm", "gate_wilcoxon",
            "gate_sign", "gate_coverage", "gate_dm_q", "gate_wilcoxon_q", "gate_sign_q",
            "evidence_score",
        ]
        for name in alias_names:
            src = f"{name}_{src_prefix}"
            if src in evidence:
                evidence[f"{name}_vs_best_baseline"] = evidence[src]
        for name in ("picp", "nominal_coverage", "coverage_gap", "coverage_deficit", "nmpiw", "interval_score", "interval_is_diagnostic"):
            src = f"{best_baseline_id}_{name}"
            if src in evidence:
                evidence[f"best_baseline_{name}"] = evidence[src]
    elif best_baseline_row is not None:
        try:
            baseline_payload = _collect_prediction_payload(_find_best_variant_eval_config(plan, best_baseline_row)[1])
            pred_b = _safe_as_2d(baseline_payload["pred_model"])[:n_rows, :]
            n_direct = min(len(pred_b), len(y_test), len(pred_model), len(group_ids))
            pred_b = pred_b[:n_direct, :]
            y_direct = y_test[:n_direct, :]
            pred_direct = pred_model[:n_direct, :]
            group_direct = group_ids[:n_direct]
            mae_b, mse_b = _compute_per_sample_losses(pred_b, y_direct)
            mae_m, mse_m = _compute_per_sample_losses(pred_direct, y_direct)
            mse_diff_group = _aggregate_by_group(mse_m - mse_b, group_direct)
            ae_diff_group = _aggregate_by_group(mae_m - mae_b, group_direct)
            dm_stat, dm_p = _dm_test_from_diff(mse_diff_group, max_lag=int(args.dm_max_lag))
            w_stat, w_p = _wilcoxon_from_diff(ae_diff_group)
            sign_wins, sign_win_rate, sign_p = _sign_test_from_diff(ae_diff_group)
            boot = _bootstrap_grouped_skill(
                y_direct,
                pred_direct,
                pred_b,
                group_ids=group_direct,
                n_boot=int(args.bootstrap_iterations),
                seed=int(args.bootstrap_seed),
                mode=str(getattr(args, "bootstrap_mode", "iid")),
                block_len=int(getattr(args, "bootstrap_block_len", 3)),
            )
            base_metrics = _compute_point_metrics_grouped(pred_b, y_direct, group_direct)
            baseline_rmse = base_metrics["rmse"]
            skill = float(1.0 - model_metrics["rmse"] / baseline_rmse) if np.isfinite(model_metrics["rmse"]) and np.isfinite(baseline_rmse) and baseline_rmse > 0 else float("nan")
            int_base = _interval_proxy_metrics(pred_b, y_direct, alpha=interval_alpha)
            evidence["dm_stat_vs_best_baseline"] = dm_stat
            evidence["dm_p_vs_best_baseline"] = dm_p
            evidence["wilcoxon_stat_vs_best_baseline"] = w_stat
            evidence["wilcoxon_p_vs_best_baseline"] = w_p
            evidence["sign_wins_vs_best_baseline"] = sign_wins
            evidence["sign_win_rate_vs_best_baseline"] = sign_win_rate
            evidence["sign_p_vs_best_baseline"] = sign_p
            evidence["skill_vs_best_baseline"] = skill
            evidence["effect_median_ae_diff_vs_best_baseline"] = float(np.nanmedian(ae_diff_group)) if ae_diff_group.size else float("nan")
            evidence["effect_mean_ae_diff_vs_best_baseline"] = float(np.nanmean(ae_diff_group)) if ae_diff_group.size else float("nan")
            evidence["effect_cohen_d_ae_diff_vs_best_baseline"] = _cohen_d_from_diff(ae_diff_group)
            evidence["bootstrap_n_vs_best_baseline"] = int(boot.get("n_boot_ok", 0))
            evidence["bootstrap_skill_mean_vs_best_baseline"] = _safe_float(boot.get("skill_mean"))
            evidence["bootstrap_skill_ci05_vs_best_baseline"] = _safe_float(boot.get("skill_ci05"))
            evidence["bootstrap_skill_ci95_vs_best_baseline"] = _safe_float(boot.get("skill_ci95"))
            evidence["bootstrap_prob_skill_gt0_vs_best_baseline"] = _safe_float(boot.get("prob_skill_gt0"))
            evidence["bootstrap_rmse_diff_mean_vs_best_baseline"] = _safe_float(boot.get("rmse_diff_mean"))
            evidence["bootstrap_rmse_diff_ci05_vs_best_baseline"] = _safe_float(boot.get("rmse_diff_ci05"))
            evidence["bootstrap_rmse_diff_ci95_vs_best_baseline"] = _safe_float(boot.get("rmse_diff_ci95"))
            evidence["bootstrap_r2_diff_mean_vs_best_baseline"] = _safe_float(boot.get("r2_diff_mean"))
            evidence["lcb95_skill_vs_best_baseline"] = _safe_float(boot.get("skill_ci05"))
            evidence["best_baseline_picp"] = _safe_float(int_base["picp"])
            evidence["best_baseline_nominal_coverage"] = _safe_float(int_base["nominal_coverage"])
            evidence["best_baseline_coverage_gap"] = _safe_float(int_base["coverage_gap"])
            evidence["best_baseline_coverage_deficit"] = _safe_float(int_base["coverage_deficit"])
            evidence["best_baseline_nmpiw"] = _safe_float(int_base["nmpiw"])
            evidence["best_baseline_interval_score"] = _safe_float(int_base["interval_score"])
            evidence["picp_delta_vs_best_baseline"] = _safe_float(model_int["picp"]) - _safe_float(int_base["picp"])
            evidence["nmpiw_delta_vs_best_baseline"] = _safe_float(model_int["nmpiw"]) - _safe_float(int_base["nmpiw"])
            evidence["interval_score_delta_vs_best_baseline"] = _safe_float(model_int["interval_score"]) - _safe_float(int_base["interval_score"])
            evidence["dm_q_vs_best_baseline"] = dm_p
            evidence["wilcoxon_q_vs_best_baseline"] = w_p
            evidence["sign_q_vs_best_baseline"] = sign_p
            evidence["gate_min_raw_vs_best_baseline"] = bool(n_raw >= int(args.evidence_min_raw_samples))
            evidence["gate_prob_vs_best_baseline"] = bool(np.isfinite(evidence["bootstrap_prob_skill_gt0_vs_best_baseline"]) and evidence["bootstrap_prob_skill_gt0_vs_best_baseline"] >= float(args.evidence_min_prob))
            evidence["gate_lcb_vs_best_baseline"] = bool(np.isfinite(evidence["lcb95_skill_vs_best_baseline"]) and evidence["lcb95_skill_vs_best_baseline"] > 0)
            evidence["gate_dm_vs_best_baseline"] = bool(np.isfinite(dm_p) and dm_p < float(args.evidence_alpha) and np.isfinite(dm_stat) and dm_stat < 0)
            evidence["gate_wilcoxon_vs_best_baseline"] = bool(np.isfinite(w_p) and w_p < float(args.evidence_alpha))
            evidence["gate_sign_vs_best_baseline"] = bool(np.isfinite(sign_p) and sign_p < float(args.evidence_alpha) and np.isfinite(sign_win_rate) and sign_win_rate > 0.5)
            evidence["gate_coverage_vs_best_baseline"] = bool(
                np.isfinite(_safe_float(model_int["coverage_deficit"]))
                and np.isfinite(_safe_float(int_base["coverage_deficit"]))
                and _safe_float(model_int["coverage_deficit"]) <= coverage_tol
                and _safe_float(model_int["coverage_deficit"]) <= _safe_float(int_base["coverage_deficit"]) + coverage_tol
            )
            evidence["gate_dm_q_vs_best_baseline"] = evidence["gate_dm_vs_best_baseline"]
            evidence["gate_wilcoxon_q_vs_best_baseline"] = evidence["gate_wilcoxon_vs_best_baseline"]
            evidence["gate_sign_q_vs_best_baseline"] = evidence["gate_sign_vs_best_baseline"]
            evidence["evidence_score_vs_best_baseline"] = int(evidence["gate_lcb_vs_best_baseline"]) + int(evidence["gate_dm_vs_best_baseline"]) + int(evidence["gate_wilcoxon_vs_best_baseline"]) + int(evidence["gate_sign_vs_best_baseline"]) + int(evidence["gate_coverage_vs_best_baseline"])
        except Exception as exc:
            print(f"[WARN] Could not compute direct best-baseline evidence for {plan.dataset_dir.name}: {exc}")

    if "skill_vs_best_baseline" not in evidence and selection is not None:
        evidence["skill_vs_best_baseline"] = selection.skill_vs_best_baseline
    if "evidence_score_vs_best_baseline" in evidence:
        baseline_scores = [int(evidence["evidence_score_vs_best_baseline"])]

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
                if str(stab.get("stability_status", "")).strip().lower() == "deterministic_skipped":
                    print(
                        f"[INFO] Stability check skipped for deterministic model "
                        f"{payload.get('model_type', '')}: gate_stability={stab.get('gate_stability', False)}"
                    )
                else:
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
        feature_tag = str(row.get("feature_tag", ""))
        features_list: list[str] | None = None

        # Prefer the variant dir recorded directly in the perf entry (set at selection time).
        # Fall back to the filesystem search only if that field is absent or empty.
        stored_variant_dir = str(row.get("evidence_variant_dir", "")).strip()
        eval_cfg_path: Path | None = None
        if stored_variant_dir:
            vdir = Path(stored_variant_dir)
            cfgs = sorted(vdir.glob("config_evaluate_*.yml"))
            if cfgs:
                eval_cfg_path = cfgs[0]
        if eval_cfg_path is None or not eval_cfg_path.exists():
            _, eval_cfg_path, _ = _find_best_variant_eval_config(plan, row)

        if eval_cfg_path is not None and eval_cfg_path.exists():
            try:
                cfg = train_module.load_config(str(eval_cfg_path))
                cols = cfg.get("data", {}).get("input_columns", [])
                if cols:
                    features_list = list(cols)
            except Exception as exc:
                print(f"[WARN] Could not read eval config for {dataset_name}: {exc}")

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

    multi_target_features = [f for f in all_features if not f.endswith("_state")]
    single_target_features = [f for f in all_features if f.endswith("_state")]

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
    # Shift vmin below zero so that 0-cells map to the near-white tail of the colormap,
    # clearly distinct from any non-zero value, while the Total row retains a full gradient.
    _heat_vmin = -0.5
    _heat_vmax = max(n_targets, 1)
    _cell_w = 0.32  # inches per cell (square)

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
        heat_w = max(6, n_total_cols * _cell_w + 3.5)  # +3.5 for y-labels & colorbar
        fig, ax = plt.subplots(figsize=(heat_w, heat_h), constrained_layout=True)

        sns.heatmap(
            combined_matrix, ax=ax,
            cmap="YlGn", vmin=_heat_vmin, vmax=_heat_vmax,
            annot=False,
            cbar_kws={"label": "Number of targets including predictor", "pad": 0.01},
            xticklabels=xticklabels_with_sep,
            yticklabels=yticklabels_with_total,
            linewidths=0.5, linecolor="#eeeeee", square=True,
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
        heat_w = max(6, n_total_features * _cell_w + 3.5)
        fig, ax = plt.subplots(
            figsize=(heat_w, heat_h),
            constrained_layout=True,
        )
        sns.heatmap(
            matrix_with_total, ax=ax,
            cmap="YlGn", vmin=_heat_vmin, vmax=_heat_vmax,
            annot=False,
            cbar_kws={"label": "Number of targets including predictor", "pad": 0.01},
            xticklabels=all_features,
            yticklabels=yticklabels_with_total,
            linewidths=0.5, linecolor="#eeeeee", square=True,
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
    final_metrics_by_dataset: dict[str, pd.DataFrame] = {}
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
                    # Filter to the canonical target name for this dataset before any
                    # selection — the CSV may contain rows with inconsistent target strings
                    # (e.g. raw output_column names from MLR vs sanitised names from ML).
                    _df_for_selection = df[df["target"].astype(str) == target_name].copy() if "target" in df.columns else df
                    if _df_for_selection.empty:
                        _df_for_selection = df  # fall back to unfiltered if nothing matches
                        print(f"[WARN] No rows match target_name={target_name!r} in {final_metrics_csv.name}; using all rows for selection.")
                    # Select an initial seed row across all valid model types.  This
                    # is only used for artifact maintenance before the final
                    # authoritative SelectionRecord is built after MLR/Shapley merge.
                    valid_r2 = valid_selection_rows(_df_for_selection)
                    if valid_r2.empty:
                        print(
                            f"[WARN] No valid r2 rows meeting min valid independent test samples "
                            f"({MIN_REQUIRED_VALID_INDEPENDENT}) for {plan.dataset_dir.name}; skipping rolling CV."
                        )
                    else:
                        best_row = select_best_model_row(valid_r2)

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

                        # Append MLR rows in-memory.
                        try:
                            df = _append_mlr_to_final_metrics(plan, df, best_row, args)
                        except Exception as exc:
                            print(f"[WARN] MLR computation failed for {plan.dataset_dir.name}: {exc}")
                            traceback.print_exc()

                        # Re-select best row after MLR append so evidence targets the true best model.
                        try:
                            _df_post = df[df["target"].astype(str) == target_name].copy() if "target" in df.columns else df
                            if _df_post.empty:
                                _df_post = df
                            _post_mlr = valid_selection_rows(_df_post)
                            if not _post_mlr.empty:
                                best_row = select_best_model_row(_post_mlr)
                        except Exception:
                            pass  # keep pre-MLR best_row as fallback

                        # Patch rolling CV metrics into the in-memory DataFrame.
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
                                for col in [
                                    'rolling_cv_r2',
                                    'rolling_cv_r2_median',
                                    'rolling_cv_r2_last50',
                                    'rolling_cv_r2_pooled',
                                    'rolling_cv_rmse',
                                    'rolling_cv_mae',
                                ]:
                                    if col not in df.columns:
                                        df[col] = float('nan')
                                row_mask = (
                                    (df['feature_tag'] == best_row['feature_tag'])
                                    & (df['row_count'] == int(best_row['row_count']))
                                    & (df['model'] == best_row['model'])
                                )
                                if row_mask.any():
                                    df.loc[row_mask, 'rolling_cv_r2'] = rolling_cv_r2
                                    df.loc[row_mask, 'rolling_cv_r2_median'] = rolling_cv_r2_median
                                    df.loc[row_mask, 'rolling_cv_r2_last50'] = rolling_cv_r2_last50
                                    df.loc[row_mask, 'rolling_cv_r2_pooled'] = rolling_cv_r2_pooled
                                    df.loc[row_mask, 'rolling_cv_rmse'] = rolling_cv_rmse
                                    df.loc[row_mask, 'rolling_cv_mae'] = rolling_cv_mae
                                    print(f"[INFO] Updated rolling CV results in-memory for {plan.dataset_dir.name}")
                                else:
                                    print(f"[WARN] Could not find matching row for rolling CV update for {plan.dataset_dir.name}")
                            except Exception as exc:
                                print(f"[WARN] Could not update rolling CV results for {plan.dataset_dir.name}: {exc}")

                        # Merge Shapley rows in-memory.
                        try:
                            df = _merge_shapley_into_final_metrics(plan, df)
                        except Exception as exc:
                            print(f"[WARN] Shapley merge failed for {plan.dataset_dir.name}: {exc}")

                        # Backfill NaN std_target and row_count using the first non-NaN value
                        # from the same subset_rank. Fixes MLR rows whose values were not
                        # populated correctly in the Shapley sweep CSV before being merged here.
                        if "subset_rank" in df.columns:
                            for _col in ("std_target", "row_count"):
                                if _col not in df.columns:
                                    continue
                                _vals = pd.to_numeric(df[_col], errors="coerce")
                                _by_rank = df[_vals.notna()].groupby("subset_rank")[_col].first()
                                _nan_mask = _vals.isna()
                                if _nan_mask.any():
                                    df.loc[_nan_mask, _col] = df.loc[_nan_mask, "subset_rank"].map(_by_rank)

                        # Recompute min_skill_rmse once across all sources, write once, plot once.
                        df = _recompute_min_skill_rmse(df)
                        df.to_csv(final_metrics_csv, index=False)
                        final_metrics_by_dataset[plan.dataset_dir.name] = df.copy()
                        try:
                            _plot_final_metrics_comparison(df, output_dir)
                        except Exception as exc:
                            print(f"[WARN] Could not generate final metrics comparison plot for {plan.dataset_dir.name}: {exc}")

                        # Select final best model/baseline rows for the summary entry.
                        selection = build_selection_record(plan, df, args)
                        if selection is None:
                            print(f"[WARN] Could not build selection record for {plan.dataset_dir.name}; skipping summary row.")
                            continue
                        best_updated = selection.best_model_row
                        try:
                            stat_evidence = _compute_statistical_evidence(
                                plan,
                                selection.best_model_row,
                                args,
                                best_baseline_row=selection.best_baseline_row,
                                selection=selection,
                            )
                            status = str(stat_evidence.get("evidence_status", ""))
                            if status == "ok":
                                print(f"[INFO] Statistical evidence computed for {plan.dataset_dir.name}")
                            else:
                                print(f"[WARN] Statistical evidence incomplete for {plan.dataset_dir.name}: {status}")
                        except Exception as exc:
                            print(f"[WARN] Statistical evidence failed for {plan.dataset_dir.name}: {exc}")
                            traceback.print_exc()

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
                            extra_metrics={
                                **stat_evidence,
                                "selection_semantics": "best_model_all_valid_models_vs_configured_best_baseline",
                                "baseline_model_set": "|".join(selection.baseline_model_set),
                                "best_model_id": selection.best_model_id,
                                "best_model_label": selection.best_model_label,
                                "best_baseline_id": selection.best_baseline_id,
                                "best_baseline_label": selection.best_baseline_label,
                                "best_model_is_configured_baseline": selection.best_model_is_configured_baseline,
                                "best_model_equals_best_baseline": selection.best_model_equals_best_baseline,
                                "best_model_rmse": _selection_metric(selection.best_model_row, "rmse"),
                                "best_model_r2": _selection_metric(selection.best_model_row, "r2"),
                                "best_model_nrmse": _selection_metric(selection.best_model_row, "nrmse"),
                                "best_model_mae": _selection_metric(selection.best_model_row, "mae"),
                                "best_baseline_rmse": _selection_metric(selection.best_baseline_row, "rmse"),
                                "best_baseline_r2": _selection_metric(selection.best_baseline_row, "r2"),
                                "best_baseline_nrmse": _selection_metric(selection.best_baseline_row, "nrmse"),
                                "best_baseline_mae": _selection_metric(selection.best_baseline_row, "mae"),
                                "skill_vs_best_baseline": selection.skill_vs_best_baseline,
                            },
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
                baseline_order = _best_baseline_order(args)
                baseline_stats = {name: {} for name in baseline_order}
                if os.path.exists(eval_csv):
                    try:
                        df_eval = pd.read_csv(eval_csv)
                        for kind in BASELINE_ORDER:
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
                if 'mlr' in baseline_stats:
                    baseline_stats['mlr'] = _load_best_mlr_baseline_stats(data_root, dataset)
                for kind in baseline_stats.keys():
                    for stat in ['mae', 'rmse', 'r2', 'pearson_r']:
                        if stat not in baseline_stats[kind]:
                            baseline_stats[kind][stat] = np.nan
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
            # Use the authoritative selection-record columns for headline outputs.
            if "best_model_nrmse" not in perf_df.columns:
                perf_df["best_model_nrmse"] = perf_df.get("nrmse", pd.Series(dtype=float))
            if "best_model_r2" not in perf_df.columns:
                perf_df["best_model_r2"] = perf_df.get("r2", pd.Series(dtype=float))
            if "best_baseline_nrmse" not in perf_df.columns:
                perf_df["best_baseline_nrmse"] = float("nan")
            if "best_baseline_r2" not in perf_df.columns:
                perf_df["best_baseline_r2"] = float("nan")
            if "skill_vs_best_baseline" not in perf_df.columns:
                perf_df["skill_vs_best_baseline"] = 1.0 - (
                    pd.to_numeric(perf_df.get("best_model_rmse"), errors="coerce")
                    / pd.to_numeric(perf_df.get("best_baseline_rmse"), errors="coerce").replace(0, np.nan)
                )
            same_baseline = perf_df.get("best_model_equals_best_baseline", pd.Series([False] * len(perf_df))).astype(bool)
            perf_df.loc[same_baseline, "skill_vs_best_baseline"] = 0.0
            required_headline_cols = [
                "best_model_label", "best_baseline_label",
                "best_model_nrmse", "best_baseline_nrmse",
                "best_model_r2", "best_baseline_r2",
                "skill_vs_best_baseline",
            ]
            missing_headline_cols = [c for c in required_headline_cols if c not in perf_df.columns]
            if missing_headline_cols:
                print(f"[WARN] Headline summary missing expected selection columns: {missing_headline_cols}")
            if same_baseline.any():
                nonzero_self = pd.to_numeric(
                    perf_df.loc[same_baseline, "skill_vs_best_baseline"],
                    errors="coerce",
                ).abs() > 1e-12
                if bool(nonzero_self.any()):
                    print("[WARN] Correcting nonzero self-baseline skill values to 0.")
                    perf_df.loc[same_baseline, "skill_vs_best_baseline"] = 0.0
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
            labels = perf_df['dataset']
            model_series_label = 'Best Model'
            methods = ['Best Model', 'Best Baseline']
            colors = ['tab:blue', 'tab:orange']
            width = 1.0 / (len(methods) + 0.5)

            nrmse_data = [
                pd.to_numeric(perf_df['best_model_nrmse'], errors='coerce'),
                pd.to_numeric(perf_df['best_baseline_nrmse'], errors='coerce'),
            ]
            r2_data = [
                pd.to_numeric(perf_df['best_model_r2'], errors='coerce'),
                pd.to_numeric(perf_df['best_baseline_r2'], errors='coerce'),
            ]
            best_model_labels = perf_df.get("best_model_label", perf_df.get("model", pd.Series(["Model"] * len(perf_df)))).astype(str).tolist()
            best_baseline_labels = perf_df.get("best_baseline_label", pd.Series(["Baseline"] * len(perf_df))).astype(str).tolist()
            skill_data = [pd.to_numeric(perf_df['skill_vs_best_baseline'], errors='coerce')]
            skill_methods = ['Best Model vs Best Baseline']
            skill_colors = ['tab:blue']
            skill_width = 1.0 / (len(skill_methods) + 0.5)

            # --- Combined 3-panel figure (no title): Skill, nRMSE, R2 ---
            fig, (ax_skill_combo, ax_nrmse_combo, ax_r2_combo) = plt.subplots(
                3, 1, figsize=(max(8, len(perf_df) * 0.72 + 1.6), 13), sharex=True
            )
            skill_bars_combo = _draw_bar_group(
                ax_skill_combo, x, skill_width, skill_data, skill_colors, skill_methods, '.2f',
                center_offset=0.5
            )
            ax_skill_combo.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax_skill_combo.set_ylabel('Skill Score')
            ax_skill_combo.grid(axis='y', alpha=0.3)
            ax_skill_combo.legend()
            nrmse_bars_combo = _draw_bar_group(ax_nrmse_combo, x, width, nrmse_data, colors, methods, '.2e', annotate=False)
            _annotate_bars_with_model_labels(ax_nrmse_combo, nrmse_bars_combo[0], nrmse_data[0], best_model_labels, fmt=".2e", fontsize=8)
            _annotate_bars_with_model_labels(ax_nrmse_combo, nrmse_bars_combo[1], nrmse_data[1], best_baseline_labels, fmt=".2e", fontsize=8)
            ax_nrmse_combo.set_ylabel('nRMSE')
            ax_nrmse_combo.grid(axis='y', alpha=0.3)
            ax_nrmse_combo.legend()
            r2_bars_combo = _draw_bar_group(
                ax_r2_combo, x, width, r2_data, colors, methods, '.2f', annotate=False
            )
            ax_r2_combo.set_ylabel('Coefficient of Determination')
            ax_r2_combo.set_ylim(-0.1, 1.0)
            _annotate_bars_with_model_labels(ax_r2_combo, r2_bars_combo[0], r2_data[0], best_model_labels, fmt=".2f", fontsize=8)
            _annotate_bars_with_model_labels(ax_r2_combo, r2_bars_combo[1], r2_data[1], best_baseline_labels, fmt=".2f", fontsize=8)
            ax_r2_combo.grid(axis='y', alpha=0.3)
            ax_r2_combo.legend()
            ax_r2_combo.set_xticks(x)
            ax_r2_combo.set_xticklabels(labels, rotation=45, ha='right')
            plt.tight_layout()
            _expand_ylims_to_fit_annotations((ax_skill_combo, ax_nrmse_combo, ax_r2_combo))
            plot_path = combined_dir / "summary_best_model_performance.png"
            fig.savefig(plot_path, dpi=180, bbox_inches='tight')
            plt.close(fig)
            print(f"[INFO] Wrote summary_best_model_performance.png to {plot_path}")

            # --- Standalone nRMSE subplot ---
            fig_nrmse, ax_nrmse = plt.subplots(figsize=(max(7, len(perf_df) * 0.72 + 1.2), 5))
            nrmse_bars = _draw_bar_group(ax_nrmse, x, width, nrmse_data, colors, methods, '.2e', annotate=False)
            _annotate_bars_with_model_labels(ax_nrmse, nrmse_bars[0], nrmse_data[0], best_model_labels, fmt=".2e", fontsize=8)
            _annotate_bars_with_model_labels(ax_nrmse, nrmse_bars[1], nrmse_data[1], best_baseline_labels, fmt=".2e", fontsize=8)
            ax_nrmse.set_ylabel('nRMSE')
            ax_nrmse.set_xticks(x)
            ax_nrmse.set_xticklabels(labels, rotation=45, ha='right')
            ax_nrmse.grid(axis='y', alpha=0.3)
            ax_nrmse.legend()
            plt.tight_layout()
            _expand_ylim_to_fit_annotations(ax_nrmse)
            nrmse_path = individual_dir / "summary_best_model_nrmse.png"
            fig_nrmse.savefig(nrmse_path, dpi=300, bbox_inches='tight')
            plt.close(fig_nrmse)
            print(f"[INFO] Wrote nRMSE subplot: {nrmse_path}")

            # --- Standalone R2 subplot ---
            fig_r2, ax_r2 = plt.subplots(figsize=(max(7, len(perf_df) * 0.72 + 1.2), 5))
            r2_bars = _draw_bar_group(ax_r2, x, width, r2_data, colors, methods, '.2f', annotate=False)
            ax_r2.set_ylabel('Coefficient of Determination')
            ax_r2.set_ylim(-0.1, 1.0)
            _annotate_bars_with_model_labels(ax_r2, r2_bars[0], r2_data[0], best_model_labels, fmt=".2f", fontsize=8)
            _annotate_bars_with_model_labels(ax_r2, r2_bars[1], r2_data[1], best_baseline_labels, fmt=".2f", fontsize=8)
            ax_r2.set_xticks(x)
            ax_r2.set_xticklabels(labels, rotation=45, ha='right')
            ax_r2.grid(axis='y', alpha=0.3)
            ax_r2.legend()
            plt.tight_layout()
            _expand_ylim_to_fit_annotations(ax_r2)
            r2_path = individual_dir / "summary_best_model_r2.png"
            fig_r2.savefig(r2_path, dpi=300, bbox_inches='tight')
            plt.close(fig_r2)
            print(f"[INFO] Wrote R2 subplot: {r2_path}")

            # --- Standalone skill score subplot ---
            fig_skill, ax_skill = plt.subplots(figsize=(max(7, len(perf_df) * 0.72 + 1.2), 5))
            _draw_bar_group(ax_skill, x, skill_width, skill_data, skill_colors, skill_methods, '.2f', center_offset=0.5)
            ax_skill.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax_skill.set_ylabel('Skill Score')
            ax_skill.set_xticks(x)
            ax_skill.set_xticklabels(labels, rotation=45, ha='right')
            ax_skill.grid(axis='y', alpha=0.3)
            ax_skill.legend()
            plt.tight_layout()
            _expand_ylim_to_fit_annotations(ax_skill)
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

            baseline_prob_cols = [_perf_col("bootstrap_prob_skill_gt0_vs_best_baseline")]
            baseline_lcb_cols = [_perf_col("lcb95_skill_vs_best_baseline")]
            overall_score = _perf_col("evidence_score_overall_min")
            model_picp = _perf_col("model_picp")
            best_baseline_picp = _perf_col("best_baseline_picp")
            nominal_cov = _perf_col("model_nominal_coverage")
            fig_conf, conf_axes = plt.subplots(
                4, 1, figsize=(max(12, len(perf_df) * 0.8), 15), sharex=True
            )
            # Order: component diagnostics first, overall summaries last.
            _draw_bar_group(
                conf_axes[0], x, width,
                baseline_prob_cols,
                ['tab:orange'],
                ["Prob(skill>0) vs Best Baseline"],
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
                ['tab:orange'],
                ["95% Lower Confidence Bound of Skill\nCompared with Best Baseline"],
                '.2f',
            )
            conf_axes[1].axhline(0.0, color='black', linewidth=0.8, linestyle='--')
            conf_axes[1].set_ylabel('Skill Lower Confidence Bound')
            conf_axes[1].grid(axis='y', alpha=0.3)
            conf_axes[1].legend()

            _draw_bar_group(
                conf_axes[2], x, width,
                [model_picp, best_baseline_picp],
                ['tab:blue', 'tab:orange'],
                ['Best Model', 'Best Baseline'],
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
                    ['tab:orange'],
                    ["Prob(skill>0) vs Best Baseline"],
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
                    ['tab:orange'],
                    ["95% Lower Confidence Bound of Skill\nCompared with Best Baseline"],
                    '.2f',
                )
                ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
                ax.set_ylabel('Skill Lower Confidence Bound')
                ax.grid(axis='y', alpha=0.3)
                ax.legend()

            def _conf_panel_picp(ax):
                _draw_bar_group(
                    ax, x, width,
                    [model_picp, best_baseline_picp],
                    ['tab:blue', 'tab:orange'],
                    ['Best Model', 'Best Baseline'],
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

            baseline_colors = ['tab:orange']
            baseline_methods = ["Best Baseline"]
            trio_colors = ['tab:blue'] + baseline_colors
            trio_methods = ['Best Model'] + baseline_methods

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
            _baseline_panel(axes_tests[0], ['dm_p_vs_best_baseline'], 'Diebold-Mariano Test p-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[1], ['dm_q_vs_best_baseline'], 'Diebold-Mariano Test\nFalse Discovery Rate Adjusted q-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[2], ['wilcoxon_p_vs_best_baseline'], 'Wilcoxon Signed-Rank Test p-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[3], ['wilcoxon_q_vs_best_baseline'], 'Wilcoxon Signed-Rank Test\nFalse Discovery Rate Adjusted q-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[4], ['sign_p_vs_best_baseline'], 'Sign Test p-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[5], ['sign_q_vs_best_baseline'], 'Sign Test\nFalse Discovery Rate Adjusted q-value', '.3f', hline=float(args.evidence_alpha), ylim=(0.0, 1.05))
            _baseline_panel(axes_tests[6], ['dm_stat_vs_best_baseline'], 'Diebold-Mariano Test Statistic', '.2f', hline=0.0)
            _baseline_panel(axes_tests[7], ['wilcoxon_stat_vs_best_baseline'], 'Wilcoxon Statistic', '.2f')
            _baseline_panel(axes_tests[8], ['sign_win_rate_vs_best_baseline'], 'Sign Test Win Rate', '.2f', hline=0.5, ylim=(0.0, 1.05))
            axes_tests[0].set_title("Evidence Tests (p/q thresholds first, diagnostics after)")
            axes_tests[-1].set_xticks(x)
            axes_tests[-1].set_xticklabels(labels, rotation=45, ha='right')
            _finalize_stacked_figure(fig_tests, axes_tests, left=0.34, hspace=0.55)
            tests_path = combined_dir / "summary_evidence_tests.png"
            fig_tests.savefig(tests_path, dpi=300, bbox_inches='tight')

            test_specs = [
                (['dm_p_vs_best_baseline'], 'Diebold-Mariano Test p-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['dm_q_vs_best_baseline'], 'Diebold-Mariano Test\nFalse Discovery Rate Adjusted q-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['wilcoxon_p_vs_best_baseline'], 'Wilcoxon Signed-Rank Test p-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['wilcoxon_q_vs_best_baseline'], 'Wilcoxon Signed-Rank Test\nFalse Discovery Rate Adjusted q-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['sign_p_vs_best_baseline'], 'Sign Test p-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['sign_q_vs_best_baseline'], 'Sign Test\nFalse Discovery Rate Adjusted q-value', '.3f', float(args.evidence_alpha), (0.0, 1.05)),
                (['dm_stat_vs_best_baseline'], 'Diebold-Mariano Test Statistic', '.2f', 0.0, None),
                (['wilcoxon_stat_vs_best_baseline'], 'Wilcoxon Statistic', '.2f', None, None),
                (['sign_win_rate_vs_best_baseline'], 'Sign Test Win Rate', '.2f', 0.5, (0.0, 1.05)),
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
            _baseline_panel(axes_eff[0], ['skill_vs_best_baseline'], 'Skill (RMSE-Based)', '.2f', hline=0.0)
            _baseline_panel(axes_eff[1], ['bootstrap_skill_mean_vs_best_baseline'], 'Bootstrap Skill Mean', '.2f', hline=0.0)
            _baseline_panel(axes_eff[2], ['lcb95_skill_vs_best_baseline'], '95% Lower Confidence Bound of Skill', '.2f', hline=0.0)
            _baseline_panel(axes_eff[3], ['effect_median_ae_diff_vs_best_baseline'], 'Median MAE Difference', '.2e', hline=0.0)
            _baseline_panel(axes_eff[4], ['effect_mean_ae_diff_vs_best_baseline'], 'Mean MAE Difference', '.2e', hline=0.0)
            _baseline_panel(axes_eff[5], ['effect_cohen_d_ae_diff_vs_best_baseline'], "Cohen's d for MAE Difference", '.2f', hline=0.0)
            _baseline_panel(axes_eff[6], ['bootstrap_rmse_diff_mean_vs_best_baseline'], 'Bootstrap RMSE Difference Mean', '.2e', hline=0.0)
            _baseline_panel(axes_eff[7], ['bootstrap_rmse_diff_ci05_vs_best_baseline'], 'Bootstrap RMSE Difference\n5th Percentile', '.2e', hline=0.0)
            _baseline_panel(axes_eff[8], ['bootstrap_rmse_diff_ci95_vs_best_baseline'], 'Bootstrap RMSE Difference\n95th Percentile', '.2e', hline=0.0)
            _baseline_panel(axes_eff[9], ['bootstrap_r2_diff_mean_vs_best_baseline'], 'Bootstrap Coefficient of Determination Difference Mean', '.2f', hline=0.0)
            axes_eff[0].set_title("Evidence Effects (skill, effect sizes, then bootstrap deltas)")
            axes_eff[-1].set_xticks(x)
            axes_eff[-1].set_xticklabels(labels, rotation=45, ha='right')
            _finalize_stacked_figure(fig_eff, axes_eff, left=0.36, hspace=0.56)
            eff_path = combined_dir / "summary_evidence_effects.png"
            fig_eff.savefig(eff_path, dpi=300, bbox_inches='tight')

            eff_specs = [
                (['skill_vs_best_baseline'], 'Skill (RMSE-Based)', '.2f', 0.0, None),
                (['bootstrap_skill_mean_vs_best_baseline'], 'Bootstrap Skill Mean', '.2f', 0.0, None),
                (['lcb95_skill_vs_best_baseline'], '95% Lower Confidence Bound of Skill', '.2f', 0.0, None),
                (['effect_median_ae_diff_vs_best_baseline'], 'Median MAE Difference', '.2e', 0.0, None),
                (['effect_mean_ae_diff_vs_best_baseline'], 'Mean MAE Difference', '.2e', 0.0, None),
                (['effect_cohen_d_ae_diff_vs_best_baseline'], "Cohen's d for MAE Difference", '.2f', 0.0, None),
                (['bootstrap_rmse_diff_mean_vs_best_baseline'], 'Bootstrap RMSE Difference Mean', '.2e', 0.0, None),
                (['bootstrap_rmse_diff_ci05_vs_best_baseline'], 'Bootstrap RMSE Difference\n5th Percentile', '.2e', 0.0, None),
                (['bootstrap_rmse_diff_ci95_vs_best_baseline'], 'Bootstrap RMSE Difference\n95th Percentile', '.2e', 0.0, None),
                (['bootstrap_r2_diff_mean_vs_best_baseline'], 'Bootstrap Coefficient of Determination Difference Mean', '.2f', 0.0, None),
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
                [_col('model_picp'), _col('best_baseline_picp')],
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
                [_col('model_coverage_deficit'), _col('best_baseline_coverage_deficit')],
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
                [_col('model_nmpiw'), _col('best_baseline_nmpiw')],
                trio_colors,
                trio_methods,
                '.2f',
            )
            axes_int[2].set_ylabel('Normalized Mean Prediction\nInterval Width')
            axes_int[2].grid(axis='y', alpha=0.3)
            axes_int[2].legend()
            _draw_bar_group(
                axes_int[3], x, width,
                [_col('model_interval_score'), _col('best_baseline_interval_score')],
                trio_colors,
                trio_methods,
                '.2e',
            )
            axes_int[3].set_ylabel('Interval Score')
            axes_int[3].grid(axis='y', alpha=0.3)
            axes_int[3].legend()
            _baseline_panel(axes_int[4], ['picp_delta_vs_best_baseline'], 'Prediction Interval Coverage Probability Difference\n(Model minus Best Baseline)', '.2f', hline=0.0)
            _baseline_panel(axes_int[5], ['nmpiw_delta_vs_best_baseline'], 'Normalized Mean Prediction Interval Width Difference\n(Model minus Best Baseline)', '.2f', hline=0.0)
            _baseline_panel(axes_int[6], ['interval_score_delta_vs_best_baseline'], 'Interval Score Delta', '.2e', hline=0.0)
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
                    [_col('model_picp'), _col('best_baseline_picp')],
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
                    [_col('model_coverage_deficit'), _col('best_baseline_coverage_deficit')],
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
                    [_col('model_nmpiw'), _col('best_baseline_nmpiw')],
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
                    [_col('model_interval_score'), _col('best_baseline_interval_score')],
                    trio_colors,
                    trio_methods,
                    '.2e',
                )
                ax.set_ylabel('Interval Score')
                ax.grid(axis='y', alpha=0.3)
                ax.legend()

            def _int_panel_picp_delta(ax):
                _baseline_panel(ax, ['picp_delta_vs_best_baseline'], 'Prediction Interval Coverage Probability Difference\n(Model minus Best Baseline)', '.2f', hline=0.0)

            def _int_panel_nmpiw_delta(ax):
                _baseline_panel(ax, ['nmpiw_delta_vs_best_baseline'], 'Normalized Mean Prediction Interval Width Difference\n(Model minus Best Baseline)', '.2f', hline=0.0)

            def _int_panel_interval_delta(ax):
                _baseline_panel(ax, ['interval_score_delta_vs_best_baseline'], 'Interval Score Delta', '.2e', hline=0.0)

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
                (['gate_min_raw_vs_best_baseline'], 'Gate: Minimum Independent Raw Sample Count'),
                (['gate_prob_vs_best_baseline'], 'Gate: Bootstrap Probability of Positive Skill'),
                (['gate_lcb_vs_best_baseline'], 'Gate: 95% Lower Confidence Bound of Skill > 0'),
                (['gate_dm_vs_best_baseline'], 'Gate: Diebold-Mariano p-value < alpha and statistic < 0'),
                (['gate_wilcoxon_vs_best_baseline'], 'Gate: Wilcoxon p-value < alpha'),
                (['gate_sign_vs_best_baseline'], 'Gate: Sign Test p-value < alpha and win rate > 0.5'),
                (['gate_coverage_vs_best_baseline'], 'Gate: Coverage Quality'),
                (['gate_dm_q_vs_best_baseline'], 'Gate: Diebold-Mariano q-value < alpha and statistic < 0'),
                (['gate_wilcoxon_q_vs_best_baseline'], 'Gate: Wilcoxon q-value < alpha'),
                (['gate_sign_q_vs_best_baseline'], 'Gate: Sign Test q-value < alpha and win rate > 0.5'),
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
                "dm_q_vs_best_baseline",
                "wilcoxon_q_vs_best_baseline",
                "sign_q_vs_best_baseline",
            ]
            p_cols = [
                "dm_p_vs_best_baseline",
                "wilcoxon_p_vs_best_baseline",
                "sign_p_vs_best_baseline",
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
                "R²": _col_values("best_model_r2"),
                "nRMSE": _col_values("best_model_nrmse"),
                "Skill vs. Best Baseline": _col_values("skill_vs_best_baseline"),
                "Coverage Gap (PICP − Nominal)": _col_values("model_coverage_gap"),
                "Normalized Mean Prediction Interval Width": _col_values("model_nmpiw"),
                "Minimum 95% Lower Confidence Bound of Skill": _col_values("lcb95_skill_vs_best_baseline"),
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
                "Best Baseline",
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

            quality_df['Best Model'] = matrix_perf_df.get('best_model_label', matrix_perf_df.get('model', pd.Series(["Unknown"] * len(matrix_perf_df)))).map(_display_model_type).values
            quality_df['Best Baseline'] = matrix_perf_df.get('best_baseline_label', pd.Series(["Unknown"] * len(matrix_perf_df))).astype(str).values
            col_order = list(quality_df.columns)
            if 'Best Model' in col_order:
                col_order.insert(1, col_order.pop(col_order.index('Best Model')))
            if 'Best Baseline' in col_order:
                col_order.insert(2, col_order.pop(col_order.index('Best Baseline')))
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
                if c in {'Best Model', 'Best Baseline'}:
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
                if c in {'Best Model', 'Best Baseline'}:
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
            model_color_map = {
                'XGB': 'tab:blue', 'Trans.': 'tab:purple', 'GP': 'tab:olive',
                'MLR': 'tab:red', 'MLR-12': 'tab:pink', 'MLR-All': 'brown',
                'Naive': 'tab:gray', 'Seasonal': 'tab:green', 'Linear': 'tab:orange',
            }
            # Build color matrix
            color_matrix = np.full(norm.shape, np.nan, dtype=object)
            for i, col in enumerate(norm.columns):
                if col in {'Best Model', 'Best Baseline'}:
                    for j, val in enumerate(display_df[col]):
                        color_matrix[j, i] = model_color_map.get(val, 'tab:gray')
                else:
                    color_matrix[:, i] = None
            def _cell_color_func(val, row, col):
                if norm.columns[col] in {'Best Model', 'Best Baseline'}:
                    return model_color_map.get(display_df[norm.columns[col]].iloc[row], 'tab:gray')
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
            # Overlay colored rectangles for categorical model columns, then draw labels on top
            for i, col in enumerate(norm.columns):
                if col in {'Best Model', 'Best Baseline'}:
                    for j, val in enumerate(display_df[col]):
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
            _expand_ylims_to_fit_annotations(ax_mat)
            matrix_path = evaluation_dir / "summary_model_quality_matrix.png"
            fig_mat.savefig(matrix_path, dpi=300, bbox_inches='tight')
            plt.close(fig_mat)
            print(f"[INFO] Wrote model quality matrix: {matrix_path}")

            # --- Best model vs best baseline R² clustered chart (evaluation/) ---
            eval_r2_rows: list[dict[str, object]] = []
            for _, row in perf_df.iterrows():
                eval_r2_rows.append({
                    "dataset": row.get("dataset", ""),
                    "target_label": clean_target_label(str(row.get("dataset", "")), args.dataset_prefix),
                    "ml_r2": _safe_float(row.get("best_model_r2", row.get("r2", float("nan")))),
                    "ml_model_label": str(row.get("best_model_label", _display_model_type(row.get("model", "")))),
                    "baseline_r2": _safe_float(row.get("best_baseline_r2", float("nan"))),
                    "baseline_model_label": str(row.get("best_baseline_label", "Baseline")),
                })

            if eval_r2_rows:
                eval_r2_df = pd.DataFrame(eval_r2_rows)
                eval_r2_df = eval_r2_df.sort_values(["ml_r2", "baseline_r2"], ascending=[False, False], na_position="last")

                x_eval = np.arange(len(eval_r2_df), dtype=float)
                labels_eval = eval_r2_df["target_label"].astype(str).tolist()
                ml_vals = pd.to_numeric(eval_r2_df["ml_r2"], errors="coerce").to_numpy(dtype=float)
                baseline_vals = pd.to_numeric(eval_r2_df["baseline_r2"], errors="coerce").to_numpy(dtype=float)
                ml_model_labels = eval_r2_df["ml_model_label"].astype(str).tolist()
                baseline_model_labels = eval_r2_df["baseline_model_label"].astype(str).tolist()

                cluster_w = 0.44
                fig_eval, ax_eval = plt.subplots(figsize=(max(10, len(eval_r2_df) * 0.82 + 1.2), 6.4))
                bars_ml = ax_eval.bar(
                    x_eval - cluster_w / 2,
                    ml_vals,
                    width=cluster_w,
                    color="tab:blue",
                    label="Best Model",
                )
                bars_bl = ax_eval.bar(
                    x_eval + cluster_w / 2,
                    baseline_vals,
                    width=cluster_w,
                    color="tab:orange",
                    label="Best Baseline",
                )
                ax_eval.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
                ax_eval.set_ylabel("Coefficient of Determination ($R^2$)")
                finite_eval_vals = np.concatenate([ml_vals, baseline_vals])
                finite_eval_vals = finite_eval_vals[np.isfinite(finite_eval_vals)]
                if finite_eval_vals.size and float(np.nanmin(finite_eval_vals)) >= 0.0:
                    ax_eval.set_ylim(0.0, 1.0)
                else:
                    min_eval_val = float(np.nanmin(finite_eval_vals)) if finite_eval_vals.size else -1.0
                    if min_eval_val <= -0.9:
                        y_min = -1.0
                    else:
                        y_min = round((1.1 * min_eval_val) / 0.05) * 0.05
                    ax_eval.set_ylim(y_min, 1.0)
                # Safety check: never let the chosen display limits clip selected bars.
                if finite_eval_vals.size:
                    cur_ymin, cur_ymax = ax_eval.get_ylim()
                    true_min = float(np.nanmin(finite_eval_vals))
                    if true_min < cur_ymin:
                        yspan = float(cur_ymax - cur_ymin) if np.isfinite(cur_ymax - cur_ymin) and (cur_ymax - cur_ymin) > 0 else 1.0
                        pad = max(0.02, 0.04 * yspan)
                        new_ymin = true_min - pad
                        if true_min <= -0.9:
                            new_ymin = min(-1.0, new_ymin)
                        else:
                            new_ymin = round(new_ymin / 0.05) * 0.05
                        ax_eval.set_ylim(new_ymin, cur_ymax)
                ax_eval.set_xticks(x_eval)
                ax_eval.set_xticklabels(labels_eval, rotation=45, ha="right")
                if len(x_eval) > 0:
                    ax_eval.set_xlim(x_eval[0] - cluster_w - 0.10, x_eval[-1] + cluster_w + 0.10)
                ax_eval.grid(axis="y", alpha=0.3)
                ax_eval.legend(
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=2,
                    frameon=False,
                    fontsize=10,
                )
                _annotate_bars_with_model_labels(ax_eval, bars_ml, ml_vals, ml_model_labels, fmt=".2f", fontsize=9)
                _annotate_bars_with_model_labels(ax_eval, bars_bl, baseline_vals, baseline_model_labels, fmt=".2f", fontsize=9)
                fig_eval.tight_layout(rect=[0, 0, 1, 0.93])
                _expand_ylims_to_fit_annotations(ax_eval)
                eval_r2_path = evaluation_dir / "summary_best_model_vs_best_baseline_r2.png"
                fig_eval.savefig(eval_r2_path, dpi=300, bbox_inches="tight")
                plt.close(fig_eval)
                legacy_eval_r2_path = evaluation_dir / "summary_best_ml_vs_best_baseline_r2.png"
                if legacy_eval_r2_path.exists():
                    try:
                        legacy_eval_r2_path.unlink()
                    except Exception as exc:
                        print(f"[WARN] Could not remove legacy ML-vs-baseline figure {legacy_eval_r2_path}: {exc}")
                print(f"[INFO] Wrote best-model-vs-best-baseline R² figure: {eval_r2_path}")
            else:
                print("[INFO] Skipped best-model-vs-best-baseline R² figure: no datasets had both model and baseline candidates.")

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
                    summary_axis_label="Summed Target-wise Shapley z-score",
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
        "--ml-selection",
        choices=["best", "xgb"],
        default="best",
        help=(
            "How to choose ML-family results in best-model summaries and comparisons: "
            '"best" uses the best of XGB/GP/Transformer, '
            '"xgb" restricts ML-family selection to XGB only.'
        ),
    )
    parser.add_argument(
        "--treat-mlr-as-baseline",
        action="store_true",
        help=(
            "Include the best MLR result as an additional baseline candidate in "
            "best-baseline skill summaries."
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

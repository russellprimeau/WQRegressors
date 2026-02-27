"""
Post-processor script for generating outputs of feature-selection sweep for MC datasets without re-running the entire sweep.

Examples:
python src/i_PostProcess.py --keep-search-plots
python src/i_PostProcess.py --sweep-namespace feature_sweeps
python src/i_PostProcess.py --sweep-namespace Shapley_sweeps
"""
from __future__ import annotations
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
from utils.training import load_samples, group_samples_by_segment
from h_RunMCFeatureSelectionSweep import build_parser, discover_mc_dataset_plans, _derive_target_name, _select_surrogate_config, _parse_row_counts, _available_row_counts_for_postprocess, _regenerate_saved_outputs_for_row, _load_feature_stats_artifacts, _compile_multi_target_comparison, _resolve_dataset_inclusion, _run_rolling_origin_cv, _ensure_k01_baselines, _write_dataset_evaluation_summary, _forecast_sweeps_dir


SUPPORTED_CONFIG_SUFFIXES = {".yml", ".yaml", ".json"}


def _resolve_summaries_dir(data_root: Path, sweep_namespace: str) -> Path:
    base_dir = (data_root.parent / "regression" / "summaries").resolve()
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
    n_test_samples: float
    input_dim: float
    target_dim: float

def _safe_float(val) -> float:
    """Return float(val) if val is non-null, otherwise float('nan')."""
    try:
        return float(val) if pd.notnull(val) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _build_perf_entry(
    dataset_name: str,
    row: "pd.Series",
    rolling_cv_r2: float = float('nan'),
    rolling_cv_rmse: float = float('nan'),
    rolling_cv_mae: float = float('nan'),
    rolling_cv_n_folds: float = float('nan'),
    n_test_samples_raw: float = float('nan'),
) -> dict:
    """Build a best-model-performance dict from a metrics row and rolling CV stats."""
    n_test_samples_mc = _safe_float(row.get('n_samples', float('nan')))
    n_test_samples = n_test_samples_raw if np.isfinite(n_test_samples_raw) else n_test_samples_mc
    return {
        'dataset': dataset_name,
        'model': str(row.get('model', '')),
        'nrmse': _safe_float(row.get('nrmse', float('nan'))),
        'rmse': _safe_float(row.get('rmse', float('nan'))),
        'r2': _safe_float(row.get('r2', float('nan'))),
        'std_target': _safe_float(row.get('std_target', float('nan'))),
        'n_test_samples': n_test_samples,
        'n_test_samples_mc': n_test_samples_mc,
        'rolling_cv_r2': rolling_cv_r2,
        'rolling_cv_rmse': rolling_cv_rmse,
        'rolling_cv_mae': rolling_cv_mae,
        'rolling_cv_n_folds': rolling_cv_n_folds,
    }


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


def _map_to_raw_filenames(file_names: list[str]) -> list[str]:
    mapped = []
    seen = set()
    for file_name in file_names:
        mapped_name = re.sub(r"_mc_\d+(?=\.csv$)", "", str(file_name))
        if mapped_name not in seen:
            seen.add(mapped_name)
            mapped.append(mapped_name)
    return mapped


def _count_independent_test_samples(plan: DatasetPlan, row: "pd.Series") -> float:
    """Count unique raw test samples for the selected best-model variant."""
    try:
        row_count = int(row.get("row_count"))
        feature_tag = str(row.get("feature_tag", ""))
    except Exception:
        return float("nan")

    model_key = str(row.get("model", "")).strip().lower()
    output_dir = _forecast_sweeps_dir(plan.dataset_dir)
    variant_dirs = [
        p for p in sorted(output_dir.glob(f"*_r{row_count:03d}_{feature_tag}_k*"))
        if p.is_dir()
    ]

    def _read_test_files(split_path: Path) -> "list[str] | None":
        try:
            with open(split_path, "r", encoding="utf-8") as sf:
                return [line.strip() for line in sf if line.strip()]
        except Exception:
            return None

    for variant_dir in variant_dirs:
        eval_cfg = variant_dir / f"config_evaluate_{variant_dir.name}.yml"
        local_train_cfg = variant_dir / f"config_train_{variant_dir.name}.yml"
        cfg_candidates = [p for p in (eval_cfg, local_train_cfg) if p.exists()]
        for cfg_path in cfg_candidates:
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception:
                continue
            cfg_model = str(cfg.get("model_name") or cfg.get("model_type") or "").strip().lower()
            if model_key and cfg_model and model_key not in cfg_model and cfg_model not in model_key:
                continue
            split_path = variant_dir / "test_files.txt"
            if not split_path.exists():
                continue
            files = _read_test_files(split_path)
            if files:
                return float(len(_map_to_raw_filenames(files)))

    for variant_dir in variant_dirs:
        split_path = variant_dir / "test_files.txt"
        if split_path.exists():
            files = _read_test_files(split_path)
            if files:
                return float(len(_map_to_raw_filenames(files)))

    return float("nan")


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


def post(plans: list[DatasetPlan], args: argparse.Namespace) -> int:
    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()

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

        print(f"\n[INFO] Rebuilding saved outputs for {plan.dataset_dir.name}: rows={row_counts}")
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
                    print(f"[INFO] Wrote {label}: {path}")
            else:
                print(
                    f"[WARN] Could not rebuild plots for {plan.dataset_dir.name} rows={row_count}; "
                    "missing feature stats/delta artifacts."
                )

        # Run rolling origin CV and collect best model performance for summary plot
        try:
            final_metrics_csv = _forecast_sweeps_dir(plan.dataset_dir) / "feature_sweep_final_metrics.csv"
            if final_metrics_csv.exists():
                df = pd.read_csv(final_metrics_csv)
                if not df.empty:
                    # Select best row across all models/subsets by R²
                    valid_r2 = _filter_valid_rows(df)
                    if valid_r2.empty:
                        print(f"[WARN] No valid r2 values in metrics for {plan.dataset_dir.name}; skipping rolling CV.")
                    else:
                        best_row = valid_r2.loc[valid_r2['r2'].idxmax()]

                        rolling_cv_r2 = rolling_cv_rmse = rolling_cv_mae = rolling_cv_n_folds = float('nan')
                        n_test_samples_raw = float('nan')

                        # Always run rolling origin CV (recomputes on every post-process run)
                        print(f"[INFO] Running rolling origin CV for {plan.dataset_dir.name}")
                        cv_summary_path = None
                        try:
                            cv_summary_path = _run_rolling_origin_cv(plan=plan, final_metrics_csv=final_metrics_csv)
                        except Exception as exc:
                            print(f"[WARN] Rolling origin CV failed for {plan.dataset_dir.name}: {exc}")
                            traceback.print_exc()

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

                        try:
                            n_test_samples_raw = _count_independent_test_samples(plan, best_row)
                            if np.isfinite(n_test_samples_raw):
                                print(f"[INFO] Independent raw test samples: n={int(n_test_samples_raw)}")
                            else:
                                print(f"[WARN] Could not determine independent raw test sample count for {plan.dataset_dir.name}")
                        except Exception as exc:
                            print(f"[WARN] Failed to count independent raw test samples for {plan.dataset_dir.name}: {exc}")

                        # Read rolling CV results
                        if cv_summary_path is not None and cv_summary_path.exists():
                            try:
                                df_cv = pd.read_csv(cv_summary_path)
                                fold_rows = df_cv[df_cv['fold'].astype(str) != 'mean']
                                rolling_cv_n_folds = float(len(fold_rows))
                                mean_rows = df_cv[df_cv['fold'].astype(str) == 'mean']
                                if not mean_rows.empty:
                                    agg = mean_rows.iloc[0]
                                    rolling_cv_r2 = _safe_float(agg.get('r2'))
                                    rolling_cv_rmse = _safe_float(agg.get('rmse'))
                                    rolling_cv_mae = _safe_float(agg.get('mae'))
                                    print(f"[INFO] Rolling CV: r2={rolling_cv_r2:.4f}, rmse={rolling_cv_rmse:.4f}, mae={rolling_cv_mae:.4f}, n_folds={int(rolling_cv_n_folds)}")
                                else:
                                    print(f"[WARN] rolling_origin_summary.csv has no 'mean' row for {plan.dataset_dir.name}")
                            except Exception as exc:
                                print(f"[WARN] Could not read rolling CV results for {plan.dataset_dir.name}: {exc}")
                                traceback.print_exc()
                        else:
                            print(f"[WARN] Rolling origin CV summary not available for {plan.dataset_dir.name}")

                        # Write rolling CV metrics into feature_sweep_final_metrics.csv for the best model row
                        try:
                            df_metrics = pd.read_csv(final_metrics_csv)
                            for col in ['rolling_cv_r2', 'rolling_cv_rmse', 'rolling_cv_mae']:
                                if col not in df_metrics.columns:
                                    df_metrics[col] = float('nan')
                            row_mask = (
                                (df_metrics['feature_tag'] == best_row['feature_tag'])
                                & (df_metrics['row_count'] == int(best_row['row_count']))
                                & (df_metrics['model'] == best_row['model'])
                            )
                            if row_mask.any():
                                df_metrics.loc[row_mask, 'rolling_cv_r2'] = rolling_cv_r2
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
                            valid_r2_2 = _filter_valid_rows(pd.read_csv(final_metrics_csv))
                            if not valid_r2_2.empty:
                                best_updated = valid_r2_2.loc[valid_r2_2['r2'].idxmax()]
                                best_model_performance.append(_build_perf_entry(
                                    plan.dataset_dir.name, best_updated,
                                    rolling_cv_r2=_safe_float(best_updated.get('rolling_cv_r2')),
                                    rolling_cv_rmse=_safe_float(best_updated.get('rolling_cv_rmse')),
                                    rolling_cv_mae=_safe_float(best_updated.get('rolling_cv_mae')),
                                    rolling_cv_n_folds=rolling_cv_n_folds,
                                    n_test_samples_raw=n_test_samples_raw,
                                ))
                            else:
                                # Fallback: use pre-update values
                                best_model_performance.append(_build_perf_entry(
                                    plan.dataset_dir.name, best_row,
                                    rolling_cv_r2=rolling_cv_r2,
                                    rolling_cv_rmse=rolling_cv_rmse,
                                    rolling_cv_mae=rolling_cv_mae,
                                    rolling_cv_n_folds=rolling_cv_n_folds,
                                    n_test_samples_raw=n_test_samples_raw,
                                ))
                        except Exception as exc:
                            print(f"[WARN] Could not re-read updated metrics for {plan.dataset_dir.name}: {exc}")
                            best_model_performance.append(_build_perf_entry(
                                plan.dataset_dir.name, best_row,
                                rolling_cv_r2=rolling_cv_r2,
                                rolling_cv_rmse=rolling_cv_rmse,
                                rolling_cv_mae=rolling_cv_mae,
                                rolling_cv_n_folds=rolling_cv_n_folds,
                                n_test_samples_raw=n_test_samples_raw,
                            ))
        except Exception as e:
            print(f"[WARN] Could not process best model performance for {plan.dataset_dir.name}: {e}")
            traceback.print_exc()

        if wrote_any:
            datasets_with_outputs += 1

    # Generate summary_best_model_performance.png (nRMSE, R², Rolling CV R²)
    try:
        if best_model_performance:
            # --- Augment with baseline stats ---
            summaries_dir = _resolve_summaries_dir(
                data_root=data_root,
                sweep_namespace=str(getattr(args, "sweep_namespace", "feature_sweeps")),
            )
            summaries_dir.mkdir(parents=True, exist_ok=True)
            for entry in best_model_performance:
                dataset = entry['dataset']
                # Standard location for a full-dataset evaluation summary with baselines
                eval_csv = os.path.join(data_root, dataset, 'evaluation_summary.csv')
                baseline_stats = {'naive': {}, 'seasonal': {}}
                if os.path.exists(eval_csv):
                    try:
                        df_eval = pd.read_csv(eval_csv)
                        for kind in baseline_stats.keys():
                            row = df_eval[df_eval['label'].str.lower().str.contains(kind)].iloc[0] if not df_eval[df_eval['label'].str.lower().str.contains(kind)].empty else None
                            if row is not None:
                                for stat in ['mae', 'rmse', 'r2']:
                                    baseline_stats[kind][stat] = row.get(stat, np.nan)
                            else:
                                for stat in ['mae', 'rmse', 'r2']:
                                    baseline_stats[kind][stat] = np.nan
                    except Exception as e:
                        print(f"[WARN] Could not read baseline stats for {dataset}: {e}")
                        for kind in baseline_stats.keys():
                            for stat in ['mae', 'rmse', 'r2']:
                                baseline_stats[kind][stat] = np.nan
                else:
                    for kind in baseline_stats.keys():
                        for stat in ['mae', 'rmse', 'r2']:
                            baseline_stats[kind][stat] = np.nan
                for kind in baseline_stats.keys():
                    for stat in ['mae', 'rmse', 'r2']:
                        entry[f'{kind}_{stat}'] = baseline_stats[kind][stat]

            perf_df = pd.DataFrame(best_model_performance)
            perf_df = perf_df.sort_values('r2', ascending=False)
            summary_csv = summaries_dir / "summary_best_model_performance.csv"
            perf_df.to_csv(summary_csv, index=False)
            print(f"[INFO] Wrote summary CSV: {summary_csv}")

            x = np.arange(len(perf_df))
            width = 0.25
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
            methods = [model_series_label, 'Naive', 'Seasonal']
            colors = ['tab:blue', 'tab:gray', 'tab:green']

            std_target_col = perf_df['std_target'].replace(0, np.nan)
            nrmse_data = [
                perf_df['nrmse'],
                perf_df['naive_rmse'] / std_target_col,
                perf_df['seasonal_rmse'] / std_target_col,
            ]
            r2_data = [
                perf_df['r2'],
                perf_df['naive_r2'],
                perf_df['seasonal_r2'],
            ]
            # Skill score: 1 - (model_rmse / baseline_rmse); positive = better than baseline
            skill_naive = 1.0 - perf_df['rmse'] / perf_df['naive_rmse'].replace(0, np.nan)
            skill_seasonal = 1.0 - perf_df['rmse'] / perf_df['seasonal_rmse'].replace(0, np.nan)
            skill_data = [skill_naive, skill_seasonal]
            skill_methods = ['vs Naive', 'vs Seasonal']
            skill_colors = ['tab:gray', 'tab:green']

            # --- Combined 3-panel figure (no title): Skill, nRMSE, R² ---
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
            ax_r2_combo.set_ylabel('R²')
            ax_r2_combo.set_ylim(-0.1, 1.0)
            for bars in r2_bars_combo:
                _annotate_bars_within_ylim(ax_r2_combo, bars, '.2f')
            ax_r2_combo.grid(axis='y', alpha=0.3)
            ax_r2_combo.legend()
            ax_r2_combo.set_xticks(x)
            ax_r2_combo.set_xticklabels(labels, rotation=45, ha='right')
            plt.tight_layout()
            plot_path = summaries_dir / "summary_best_model_performance.png"
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
            nrmse_path = summaries_dir / "summary_best_model_nrmse.png"
            fig_nrmse.savefig(nrmse_path, dpi=300, bbox_inches='tight')
            plt.close(fig_nrmse)
            print(f"[INFO] Wrote nRMSE subplot: {nrmse_path}")

            # --- Standalone R² subplot ---
            fig_r2, ax_r2 = plt.subplots(figsize=(max(10, len(perf_df)*0.7), 5))
            r2_bars = _draw_bar_group(ax_r2, x, width, r2_data, colors, methods, '.2f', annotate=False)
            ax_r2.set_ylabel('R²')
            ax_r2.set_ylim(-0.1, 1.0)
            for bars in r2_bars:
                _annotate_bars_within_ylim(ax_r2, bars, '.2f')
            ax_r2.set_xticks(x)
            ax_r2.set_xticklabels(labels, rotation=45, ha='right')
            ax_r2.grid(axis='y', alpha=0.3)
            ax_r2.legend()
            plt.tight_layout()
            r2_path = summaries_dir / "summary_best_model_r2.png"
            fig_r2.savefig(r2_path, dpi=300, bbox_inches='tight')
            plt.close(fig_r2)
            print(f"[INFO] Wrote R² subplot: {r2_path}")

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
            skill_path = summaries_dir / "summary_best_model_skill.png"
            fig_skill.savefig(skill_path, dpi=300, bbox_inches='tight')
            plt.close(fig_skill)
            print(f"[INFO] Wrote skill score subplot: {skill_path}")

            # --- Cross-validation figure (sorted by descending R², same order as perf_df) ---
            cv_r2_col = 'rolling_cv_r2' if 'rolling_cv_r2' in perf_df.columns else None
            cv_data_available = cv_r2_col is not None and perf_df[cv_r2_col].notnull().any()
            if cv_data_available:
                # Generalization gap: test R² - CV R² (positive = test was optimistic / overfit)
                gen_gap = perf_df['r2'] - perf_df['rolling_cv_r2']
                n_folds_col = perf_df['rolling_cv_n_folds'] if 'rolling_cv_n_folds' in perf_df.columns else pd.Series([float('nan')] * len(perf_df))
                n_samples_col = perf_df['n_test_samples'] if 'n_test_samples' in perf_df.columns else pd.Series([float('nan')] * len(perf_df))

                cv_panel_specs = [
                    (perf_df['rolling_cv_r2'], 'CV R²',            'tab:blue',   '.2f'),
                    (gen_gap,                  'Generalization Gap\n(test R² − CV R²)', 'tab:red', '.2e'),
                    (n_folds_col,              'CV Folds (n)',      'tab:purple', '.0f'),
                    (n_samples_col,            'Test Samples (n)',  'tab:orange', '.0f'),
                ]
                fig_cv, cv_axes = plt.subplots(
                    len(cv_panel_specs), 1,
                    figsize=(max(10, len(perf_df) * 0.7), 3.5 * len(cv_panel_specs)),
                    sharex=True,
                )
                for ax_cv, (vals, ylabel, color, fmt) in zip(cv_axes, cv_panel_specs):
                    bars = ax_cv.bar(x, vals, width=0.5, color=color)
                    if ylabel.startswith('CV R'):
                        ax_cv.set_ylim(-0.1, 1.0)
                    _annotate_bars_within_ylim(ax_cv, bars, fmt)
                    if ylabel.startswith('Generalization'):
                        ax_cv.axhline(0, color='black', linewidth=0.8, linestyle='--')
                    ax_cv.set_ylabel(ylabel)
                    ax_cv.grid(axis='y', alpha=0.3)
                cv_axes[-1].set_xticks(x)
                cv_axes[-1].set_xticklabels(labels, rotation=45, ha='right')
                plt.tight_layout()
                cv_path = summaries_dir / "cross-validation.png"
                fig_cv.savefig(cv_path, dpi=300, bbox_inches='tight')
                plt.close(fig_cv)
                print(f"[INFO] Wrote cross-validation figure: {cv_path}")
            else:
                print("[WARN] No cross-validation data available; cross-validation.png not generated.")
        else:
            print("[WARN] No best model performance data found; summary plot not generated.")
    except Exception as e:
        print(f"[ERROR] Failed to generate summary_best_model_performance.png: {e}")
        traceback.print_exc()

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


def main() -> int:
    parser = build_parser()
    parser.add_argument(
        "--sweep-namespace",
        type=str,
        default="feature_sweeps",
        help=(
            "Forecast sweep subdirectory to post-process "
            "(e.g., 'feature_sweeps' or 'Shapley_sweeps')."
        ),
    )
    args = parser.parse_args()
    os.environ["WQ_FEATURE_SWEEP_NAMESPACE"] = str(args.sweep_namespace).strip() or "feature_sweeps"
    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()
    include_regular, include_res = _resolve_dataset_inclusion(args)
    plans = discover_mc_dataset_plans(
            data_root=data_root,
            dataset_prefix="",  # match all
            config_pattern=args.config_pattern,
            limit_datasets=0,   # no limit
            include_regular=include_regular,
            include_res=include_res,
        )
    output = post(plans, args)
    return output

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Standalone script to test ml_comparison_nrmse.png figure generation.

This allows rapid iteration on plot styling without re-running the full postprocess script.

Usage:
    python test_ml_comparison_plot.py --data-root data/output --output-dir /tmp
    python test_ml_comparison_plot.py --data-root data/output --output-dir ./test_output
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches

# Add src to path so we can import utilities
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.names import clean_target_label
from h_RunMCFeatureSelectionSweep import _forecast_sweeps_dir

# Constants (copied from z1_FeaturePostProcess.py)
BASELINE_ORDER = ("naive", "seasonal", "linear")
MIN_REQUIRED_VALID_INDEPENDENT = 5

ML_COMPARISON_MODEL_TYPES = ['XGB', 'Trans.', 'GP', 'MLR', 'MLR12', 'MLRall']
ML_COMPARISON_COLORS = {
    'GP': 'tab:blue', 'Trans.': 'tab:orange', 'XGB': 'tab:green',
    'MLR': 'tab:red', 'MLR12': 'tab:purple', 'MLRall': 'tab:brown',
}


def _safe_float(val) -> float:
    """Return float(val) if val is non-null, otherwise float('nan')."""
    try:
        return float(val) if pd.notnull(val) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _is_baseline_model_value(value: object) -> bool:
    BASELINE_MODEL_IDS = {"naive", "seasonal", "linear"}
    return str(value).strip().lower() in BASELINE_MODEL_IDS


def _exclude_baseline_metric_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    out = df.copy()
    if 'model' not in out.columns:
        return out
    return out[~out['model'].apply(_is_baseline_model_value)].copy()


def _filter_valid_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    """Keep rows where n_test_valid >= 1."""
    out = df.copy()
    for col in ['n_test_valid', 'n_test_independent']:
        if col in out.columns:
            out = out[out[col].fillna(0) >= 1]
            break
    return out


def _filter_min_valid_independent(df: "pd.DataFrame", min_required: int = MIN_REQUIRED_VALID_INDEPENDENT) -> "pd.DataFrame":
    """Keep rows where n_test_valid >= min_required."""
    out = df.copy()
    for col in ['n_test_valid', 'n_test_independent']:
        if col in out.columns:
            out = out[out[col].fillna(0) >= min_required]
            break
    return out


def _normalize_ml_model_display(val: str) -> str:
    """Map raw model string to ML_COMPARISON_MODEL_TYPES display key."""
    key = str(val).strip().lower()
    if 'xgb' in key:
        return 'XGB'
    if 'transformer' in key:
        return 'Trans.'
    if 'gp' in key:
        return 'GP'
    if 'mlrall' in key or 'mlr_avgall' in key:
        return 'MLRall'
    if 'mlr12' in key or 'mlr_avg12' in key:
        return 'MLR12'
    if 'mlr' in key:
        return 'MLR'
    return str(val).strip()


def _annotate_ml_bars(ax, bars, vals, ns, fmt: str, fontsize: int = 10) -> None:
    """Annotate bars with combined value and sample-count label, e.g. '1.23e-02, n=8'."""
    ymin, ymax = ax.get_ylim()
    yspan = float(ymax - ymin) if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
    pad = 0.02 * yspan
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
            clip_on=False,
        )


def generate_ml_comparison_plot(data_root: Path, output_dir: Path, metric_key: str = 'nrmse') -> None:
    """Generate a single ML comparison plot (nrmse by default).

    Args:
        data_root: Path to data root containing dataset directories
        output_dir: Directory where PNG will be saved
        metric_key: Metric to plot ('rmse', 'nrmse', 'r2', 'skill_vs_best')
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_prefix = "MC"
    records = []

    # Scan all dataset directories
    for dataset_dir in data_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        # Look for final metrics CSV
        final_metrics_csv = _forecast_sweeps_dir(dataset_dir) / "feature_sweep_final_metrics.csv"
        if not final_metrics_csv.exists():
            print(f"[SKIP] No metrics CSV for {dataset_name}")
            continue

        try:
            df = pd.read_csv(final_metrics_csv)
        except Exception as e:
            print(f"[SKIP] Failed to read metrics CSV for {dataset_name}: {e}")
            continue

        if df.empty:
            continue

        # Compute nrmse: nrmse = rmse / std_target (overwrite NaN values)
        df['nrmse'] = df['rmse'] / df['std_target']

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

        # Read baseline RMSE from evaluation_summary.csv (for skill metric)
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
        print("[INFO] No valid per-model-type data found; skipping.")
        return

    comp_df = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(comp_df)} records from {len(comp_df['dataset'].unique())} datasets")

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
    n_models = len(ordered_model_types)

    # =========== TUNABLE PARAMETERS ===========
    width = 0.0225  # Individual bar width reduced by 70%
    offsets = np.array([(i - (n_models - 1) / 2) for i in range(n_models)])
    cluster_step = (n_models + 1) * width  # One cluster width plus one-column gap
    _FS = 16  # Shared font size (annotations use their own size)
    # ==========================================

    # Metric specs: (key, ylabel, fmt, add_hline, sort_order)
    metric_specs = {
        'nrmse': ('nRMSE', '.2e', False, 'ascending'),
    }

    if metric_key not in metric_specs:
        print(f"[ERROR] Unknown metric: {metric_key}")
        return

    ylabel, fmt, add_hline, sort_order = metric_specs[metric_key]

    # Sort clusters based on metric and sort_order
    cluster_sort_vals = {}
    for ds in ordered_datasets:
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

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, n_targets * (n_models * 0.35 + 0.5)), 6))
    legend_handles = [
        matplotlib.patches.Patch(facecolor=ML_COMPARISON_COLORS[m], label=m)
        for m in ordered_model_types
    ]
    pending_annotations = []

    sorted_x = np.arange(len(sorted_targets)) * cluster_step

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

    # Set y-axis limits
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

    # Tight horizontal bounds aligned to the outermost bar edges.
    cluster_half = n_models * width / 2
    if len(sorted_targets) > 0:
        ax.set_xlim(sorted_x[0] - cluster_half, sorted_x[-1] + cluster_half)

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
        loc='lower left',
        bbox_to_anchor=(0.0, 1.02),
        ncol=len(ordered_model_types),
        frameon=False,
        fontsize=_FS,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = output_dir / f"ml_comparison_{metric_key}.png"
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Wrote figure: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root', type=Path, default=Path('data/output'),
        help='Path to data root containing dataset directories (default: data/output)'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('test_output'),
        help='Directory to save PNG output (default: test_output)'
    )
    parser.add_argument(
        '--metric', type=str, default='nrmse',
        choices=['rmse', 'nrmse', 'r2', 'skill_vs_best'],
        help='Metric to plot (default: nrmse)'
    )

    args = parser.parse_args()
    generate_ml_comparison_plot(args.data_root, args.output_dir, args.metric)

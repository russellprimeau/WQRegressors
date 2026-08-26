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

# Add src to path so we can import utilities
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.names import clean_target_label
from h_RunMCFeatureSelectionSweep import _forecast_sweeps_dir

# Styling and drawing come from the real post-processor so this harness cannot drift from
# what the pipeline actually produces.  Only the data-gathering below is local, so that
# iterating on a figure does not require constructing DatasetPlan objects.
from z1_FeaturePostProcess import (
    BASELINE_ORDER,
    MIN_REQUIRED_VALID_INDEPENDENT,
    ML_COMPARISON_MODEL_TYPES,
    ML_COMPARISON_COLORS,
    render_ml_comparison_figures,
)


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

    # Drawing is delegated to the post-processor so this harness always reflects the
    # figure the pipeline actually writes.  All four metric figures are produced; the
    # --metric argument selects which one to report as the primary output.
    render_ml_comparison_figures(comp_df, output_dir)
    out_path = output_dir / f"ml_comparison_{metric_key}.png"
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

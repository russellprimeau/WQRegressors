import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from pathlib import Path

def _plot_feature_importance_bar(feature_sensitivities, dataset_name, target_name, row_count, output_dir):
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

def _plot_removal_sensitivity(feature_removal_deltas, dataset_name, target_name, row_count, output_dir):
    features = sorted(feature_removal_deltas.keys())
    tested_pairs = [(f, feature_removal_deltas[f]) for f in features if len(feature_removal_deltas[f]) > 0]
    tested_pairs.sort(key=lambda item: float(np.median(item[1])), reverse=True)
    tested_features = [item[0] for item in tested_pairs]
    tested_deltas = [item[1] for item in tested_pairs]
    if not tested_features:
        return Path()
    fig_h = max(7, len(tested_features) * 0.45)
    fig, ax = plt.subplots(figsize=(14, fig_h), constrained_layout=True)
    bp = ax.boxplot(tested_deltas, vert=False, patch_artist=True)
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

def _plot_feature_frequency(feature_improvement_counts, feature_sensitivities, dataset_name, target_name, row_count, output_dir):
    ranked = sorted(feature_sensitivities.items(), key=lambda x: -x[1][0])
    features = [f for f, _ in ranked]
    frequencies = [feature_improvement_counts[f] for f in features]
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)), constrained_layout=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
    bars = ax.barh(features, frequencies, color=colors)
    for bar, freq in zip(bars, frequencies):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(freq)}', ha='left', va='center', fontsize=9)
    ax.set_xlabel("Frequency in Improving Solutions")
    ax.set_title(f"Feature Inclusion Frequency: {target_name} (rows={row_count})")
    ax.grid(axis='x', alpha=0.3)
    plot_path = output_dir / f"feature_frequency_r{row_count:03d}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path

def _write_feature_stats_artifacts(dataset_dir, row_count, feature_sensitivities, feature_removal_deltas, feature_improvement_counts):
    out_dir = dataset_dir / "forecasts" / "feature_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_rows = []
    for feature in sorted(feature_sensitivities.keys()):
        avg_delta, _ = feature_sensitivities.get(feature, (0.0, 0))
        deltas = feature_removal_deltas.get(feature, [])
        median_delta = float(np.median(deltas)) if deltas else np.nan
        stats_rows.append({
            "feature": feature,
            "avg_removal_delta": float(avg_delta),
            "median_removal_delta": median_delta,
            "n_removal_tests": int(len(deltas)),
            "improvement_count": int(feature_improvement_counts.get(feature, 0)),
        })
    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df = stats_df.sort_values(["avg_removal_delta", "feature"], ascending=[False, True], kind="stable")
    stats_csv = out_dir / f"feature_importance_stats_r{row_count:03d}.csv"
    stats_df.to_csv(stats_csv, index=False)
    delta_rows = []
    for feature in sorted(feature_removal_deltas.keys()):
        for delta in feature_removal_deltas.get(feature, []):
            delta_rows.append({
                "feature": feature,
                "delta": float(delta),
            })
    deltas_df = pd.DataFrame(delta_rows, columns=["feature", "delta"])
    deltas_csv = out_dir / f"feature_removal_deltas_r{row_count:03d}.csv"
    deltas_df.to_csv(deltas_csv, index=False)
    return stats_csv, deltas_csv

def _load_feature_stats_artifacts(dataset_dir, row_count):
    out_dir = dataset_dir / "forecasts" / "feature_sweeps"
    stats_csv = out_dir / f"feature_importance_stats_r{row_count:03d}.csv"
    deltas_csv = out_dir / f"feature_removal_deltas_r{row_count:03d}.csv"
    if not stats_csv.exists() or not deltas_csv.exists():
        return {}, {}, {}
    stats_df = pd.read_csv(stats_csv)
    feature_sensitivities = {}
    feature_improvement_counts = {}
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
    feature_removal_deltas = {feature: [] for feature in feature_sensitivities.keys()}
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

def _available_row_counts_for_postprocess(dataset_dir):
    out_dir = dataset_dir / "forecasts" / "feature_sweeps"
    patterns = [
        "feature_importance_stats_r*.csv",
        "feature_search_trace_r*.csv",
        "feature_selected_subsets_r*.csv",
    ]
    row_counts = set()
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            match = re.search(r"_r(\d{3})\.csv$", path.name)
            if match:
                row_counts.add(int(match.group(1)))
    return sorted(row_counts)

def _regenerate_saved_outputs_for_row(dataset_dir, target_name, row_count, keep_search_plots):
    out_dir = dataset_dir / "forecasts" / "feature_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
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

def main():
    parser = argparse.ArgumentParser(description="Post-process feature sweep outputs.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to dataset directory (e.g., data/experiment/SomeTrial)")
    parser.add_argument("--target-name", type=str, required=True, help="Target name for plots and outputs")
    parser.add_argument("--row-count", type=int, required=True, help="Row count (int) for which to post-process outputs")
    parser.add_argument("--keep-search-plots", action="store_true", help="If set, also regenerate search/pareto plots")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    outputs = _regenerate_saved_outputs_for_row(
        dataset_dir=dataset_dir,
        target_name=args.target_name,
        row_count=args.row_count,
        keep_search_plots=args.keep_search_plots,
    )
    if outputs:
        print("Generated the following post-process artifacts:")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
    else:
        print("No artifacts generated (check input paths and data).")

if __name__ == "__main__":
    main()
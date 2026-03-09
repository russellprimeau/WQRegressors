"""
TMC-Shapley feature-selection sweeper for MC datasets.

Overview:
- Mirrors the core orchestration in `h_RunMCFeatureSelectionSweep.py`:
    discovery -> surrogate search -> top-K final retrain/evaluation.
- Uses TMC-Shapley search instead of beam+swap:
    permutation marginals, antithetic ordering, truncation near full-subset utility.
- Writes to `forecasts/Shapley_sweeps` namespace.

Outputs:
- `feature_shapley_scores_r###.csv` (Shapley estimates + CI diagnostics).
- `feature_shapley_rankings_r###.json` (ranked feature payload).
- `feature_seed_subsets_r###_d########.csv` (optimizer seed subsets from ranked prefixes;
    legacy readers may still load `feature_seed_subsets_r###.csv`).
- `feature_shapley_samples_r###.json` (full marginal sample lists for re-visualization).
- Plot outputs (when --keep-shapley-plots enabled):
    - `feature_shapley_ranking_bar_r###.png` (ranked Shapley values with 95% CI error bars).
    - `feature_shapley_distribution_r###.png` (violin plot of marginal distributions).
    - `feature_shapley_coverage_r###.png` (sample accumulation per feature).
    - `shapley_search_budget_r###.png` (eval budget utilization stacked bar).
    - `shapley_search_progress_r###.png` (permutation progression curve).
    - `feature_shapley_rank_stability_r###.png` (rank vs sample stability scatter).
- Shared search/final artifacts via base helpers:
    `feature_search_trace_r###.csv`, `feature_selected_subsets_r###.csv`,
    optional `feature_search_pareto_r###.png`, `feature_sweep_final_metrics.csv`.
- Post-final compatibility hooks:
    baseline enforcement and dataset-level `evaluation_summary.csv` propagation.

Important behavior:
- Search baseline controls now match `h` semantics:
    `--run-baselines-in-search` is mutually exclusive with
    `--disable-baselines-for-search`.
- Search phase is performance-oriented and keeps standard temporal-by-coverage
    split behavior (target 70/30 by default).
- Final top-K evaluation enforces a minimum of 5 test samples by moving latest
    train samples into test when needed; if fewer than 5 total split samples are
    available, that model/subset evaluation is skipped.
- Optional `--run-rolling-origin-cv` runs rolling-origin CV for best k01 model.
- Visualization generation disabled by default for speed; enable with `--keep-shapley-plots`.

Key CLI groups (detailed):
- Data selection:
    `--data-root PATH`: Root directory containing regression dataset folders.
    `--dataset-prefix PREFIX`: Dataset name prefix filter (for example, `MC`).
    `--config-pattern GLOB`: Glob used to discover per-dataset train configs.
    `--limit-datasets N`: Max matching datasets to process (`0` means all).
    `--row-counts CSV_INTS`: Comma-separated input window sizes to evaluate.
    `--include-regular`: Include regular (non-`_res`) datasets.
    `--include-res`: Include `_res` datasets.
    `--regular-only`: Include only regular datasets.
    `--res-only`: Include only `_res` datasets.
- Shapley controls:
    `--eval-budget N`: Max candidate evaluations for the TMC-Shapley stage.
    `--shapley-samples-per-feature N`: Target marginal samples per feature before stopping.
    `--shapley-min-coalition-size N`: Minimum coalition size to start each permutation path.
    `--tmc-max-permutations N`: Hard cap on permutation runs (`0` means no explicit cap).
    `--tmc-truncation-epsilon FLOAT`: Truncation tolerance near full-subset objective.
    `--tmc-bootstrap-resamples N`: Bootstrap resamples for Shapley confidence intervals.
    `--seed N`: Random seed for permutation ordering and bootstrap sampling.
- Final/model controls:
    `--min-features N`: Minimum feature count allowed for subset generation/search.
    `--lambda-drop FLOAT`: Penalty weight for sample drop rate in objective.
    `--final-top-k N`: Number of best discovered subsets retrained in final stage.
- Baselines/logging:
    `--disable-baselines-for-search`: Disable baseline evals during search for speed.
    `--run-baselines-in-search`: Enable baseline evals during search (mutually exclusive with disable flag).
    `--show-training-logs`: Show verbose model training/sample logs.
- Artifacts/runtime:
    `--keep-training-plots`: Keep per-model training plots during sweeps.
    `--keep-eval-plots`: Keep per-config evaluation plots during sweeps.
    `--keep-search-plots`: Keep feature-search Pareto plot outputs.
    `--keep-shapley-plots`: Keep Shapley attribution/search diagnostic plots.
    `--run-rolling-origin-cv`: Run rolling-origin CV for the best k01 model.
    `--dry-run`: Print discovered execution plan and exit.
    `--stop-on-error`: Raise immediately on first dataset failure.

Examples:
        python src/i_RunMCFeatureSelectionShapleySweep.py --dry-run

        python src/i_RunMCFeatureSelectionShapleySweep.py \
            --dataset-prefix MC --limit-datasets 0 --eval-budget 240 --final-top-k 4 \
            --shapley-samples-per-feature 3 --tmc-truncation-epsilon 0.0025 \
            --tmc-bootstrap-resamples 300 --keep-shapley-plots

        python src/i_RunMCFeatureSelectionShapleySweep.py \
            --dataset-prefix MC --eval-budget 180 --run-baselines-in-search \
            --run-rolling-origin-cv --keep-shapley-plots
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import e_Train as train_module
import h_RunMCFeatureSelectionSweep as base

_SHAPLEY_SWEEP_NAMESPACE = "Shapley_sweeps"


def _activate_shapley_sweep_namespace() -> None:
    os.environ["WQ_FEATURE_SWEEP_NAMESPACE"] = _SHAPLEY_SWEEP_NAMESPACE


def _ordered_tuple(features: list[str] | tuple[str, ...], reference: tuple[str, ...]) -> tuple[str, ...]:
    ref_idx = {f: i for i, f in enumerate(reference)}
    return tuple(sorted(tuple(features), key=lambda f: ref_idx[f]))


def _write_shapley_round_artifact(
    dataset_dir: Path,
    row_count: int,
    rows: list[dict],
) -> Path:
    out_dir = base._forecast_sweeps_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"feature_shapley_scores_r{row_count:03d}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _write_shapley_seed_artifacts(
    dataset_dir: Path,
    row_count: int,
    shapley_rows: list[dict],
    full_features: tuple[str, ...],
    min_features: int,
) -> tuple[Path, Path]:
    out_dir = base._forecast_sweeps_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_features = [str(r.get("feature", "")) for r in shapley_rows if str(r.get("feature", "")).strip()]
    ranked_features = [f for f in ranked_features if f in set(full_features)]
    if not ranked_features:
        ranked_features = list(full_features)

    n_features = len(ranked_features)
    min_k = max(1, int(min_features))
    prefix_max = min(n_features, max(min_k, min_k + 5))
    candidate_sizes = list(range(min_k, prefix_max + 1))
    if n_features not in candidate_sizes:
        candidate_sizes.append(n_features)

    seed_rows: list[dict] = []
    for subset_id, k in enumerate(candidate_sizes, start=1):
        prefix = ranked_features[:k]
        ordered = _ordered_tuple(prefix, full_features)
        seed_rows.append(
            {
                "row_count": int(row_count),
                "subset_id": int(subset_id),
                "source": "shapley_top_prefix",
                "n_features": int(len(ordered)),
                "max_shapley_rank": int(k),
                "features": "|".join(ordered),
            }
        )

    seed_csv = out_dir / base._seed_subsets_filename(row_count=row_count, dataset_dir=dataset_dir)
    pd.DataFrame(seed_rows).to_csv(seed_csv, index=False)

    rankings_json = out_dir / f"feature_shapley_rankings_r{row_count:03d}.json"
    payload = {
        "row_count": int(row_count),
        "n_features": int(len(full_features)),
        "features": shapley_rows,
    }
    with open(rankings_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return rankings_json, seed_csv


def _write_shapley_samples_json(
    dataset_dir: Path,
    row_count: int,
    shapley_samples: dict[str, list[float]],
    shapley_rows: list[dict],
) -> Path:
    """Save full marginal samples per feature for distribution analysis."""
    out_dir = base._forecast_sweeps_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build metadata dict for each feature
    features_data = {}
    for row in shapley_rows:
        feature = row.get("feature", "")
        if feature:
            samples = shapley_samples.get(feature, [])
            features_data[feature] = {
                "values": [float(v) for v in samples],
                "metadata": {
                    "mean": float(row.get("shapley_value_est", np.nan)),
                    "std": float(row.get("shapley_std", np.nan)),
                    "median": float(row.get("shapley_median", np.nan)),
                    "ci95_low": float(row.get("ci95_low", np.nan)),
                    "ci95_high": float(row.get("ci95_high", np.nan)),
                    "n_samples": int(row.get("n_marginal_samples", 0)),
                    "rank": int(row.get("shapley_rank", 0)),
                },
            }

    payload = {
        "row_count": int(row_count),
        "features": features_data,
    }
    
    samples_json = out_dir / f"feature_shapley_samples_r{row_count:03d}.json"
    with open(samples_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    
    return samples_json


def _plot_shapley_ranking_bar(
    shapley_rows: list[dict],
    dataset_name: str,
    row_count: int,
    output_dir: Path,
    include_row_count_in_name: bool = False,
) -> Path:
    """Plot Shapley value ranking as horizontal bar chart with confidence interval whiskers."""
    features = [str(r.get("feature", "")) for r in shapley_rows]
    estimates = [float(r.get("shapley_value_est", np.nan)) for r in shapley_rows]
    ci_lows = [float(r.get("ci95_low", np.nan)) for r in shapley_rows]
    ci_highs = [float(r.get("ci95_high", np.nan)) for r in shapley_rows]

    # Errors for plotting (differences from mean)
    errors = [
        [est - low for est, low in zip(estimates, ci_lows)],
        [high - est for high, est in zip(ci_highs, estimates)],
    ]

    fig, ax = plt.subplots(figsize=(12, max(7, len(features) * 0.35)), constrained_layout=True)
    
    if estimates:
        est_min = float(np.nanmin(estimates))
        est_max = float(np.nanmax(estimates))
        if est_max > est_min:
            norm = matplotlib.colors.Normalize(vmin=est_min, vmax=est_max)
            colors = [plt.cm.RdYlGn(float(norm(v))) for v in estimates]
        else:
            colors = [plt.cm.RdYlGn(0.5) for _ in estimates]
    else:
        colors = []

    ax.barh(features, estimates, xerr=errors, color=colors, capsize=4, error_kw={"elinewidth": 1.5})
    ax.set_xlabel("Shapley Value Estimate (with 95% CI)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    # Title intentionally omitted to reduce redundancy with companion diagnostics.

    plot_filename = f"feature_shapley_ranking_bar_r{row_count:03d}.png" if include_row_count_in_name else "feature_shapley_ranking_bar.png"
    plot_path = output_dir / plot_filename
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def _plot_shapley_distribution(
    shapley_samples: dict[str, list[float]],
    shapley_rows: list[dict],
    dataset_name: str,
    row_count: int,
    output_dir: Path,
    include_row_count_in_name: bool = False,
) -> Path:
    """Plot Shapley sample distribution as violin/KDE plot per feature."""
    features_ordered = [str(r.get("feature", "")) for r in shapley_rows]
    features_ordered = [f for f in features_ordered if f in shapley_samples and len(shapley_samples[f]) > 0]
    
    if not features_ordered:
        return Path()
    
    # Prepare data for seaborn violin plot
    plot_data = []
    for feat in features_ordered:
        for sample in shapley_samples.get(feat, []):
            plot_data.append({"Feature": feat, "Shapley Marginal": float(sample)})
    
    plot_df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(12, max(7, len(features_ordered) * 0.35)), constrained_layout=True)
    
    # Color by sample count (darker = more samples)
    sample_counts = [len(shapley_samples.get(f, [])) for f in features_ordered]
    count_min = float(np.min(sample_counts)) if sample_counts else 0
    count_max = float(np.max(sample_counts)) if sample_counts else 1
    
    if count_max > count_min:
        norm = matplotlib.colors.Normalize(vmin=count_min, vmax=count_max)
        palette = {f: plt.cm.Blues(0.3 + 0.65 * float(norm(len(shapley_samples.get(f, []))))) 
                   for f in features_ordered}
    else:
        palette = {f: plt.cm.Blues(0.6) for f in features_ordered}
    
    sns.violinplot(data=plot_df, y="Feature", x="Shapley Marginal", ax=ax, palette=palette, inner="box")
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_xlabel("Shapley Marginal Value", fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.set_title(f"Shapley Sample Distribution ({dataset_name})", fontsize=12, fontweight='bold')

    plot_filename = f"feature_shapley_distribution_r{row_count:03d}.png" if include_row_count_in_name else "feature_shapley_distribution.png"
    plot_path = output_dir / plot_filename
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def _plot_shapley_coverage(
    shapley_rows: list[dict],
    samples_per_feature: int,
    dataset_name: str,
    row_count: int,
    output_dir: Path,
    include_row_count_in_name: bool = False,
) -> Path:
    """Plot Shapley sample accumulation per feature (coverage diagnostic)."""
    features = [str(r.get("feature", "")) for r in shapley_rows]
    sample_counts = [int(r.get("n_marginal_samples", 0)) for r in shapley_rows]
    
    fig, ax = plt.subplots(figsize=(12, max(7, len(features) * 0.35)), constrained_layout=True)
    
    # Color gradient by sample count
    if sample_counts:
        count_min = float(np.min(sample_counts))
        count_max = float(np.max(sample_counts))
        if count_max > count_min:
            norm = matplotlib.colors.Normalize(vmin=count_min, vmax=count_max)
            colors = [plt.cm.Greens(0.25 + 0.7 * float(norm(v))) for v in sample_counts]
        else:
            colors = [plt.cm.Greens(0.6) for _ in sample_counts]
    else:
        colors = []
    
    bars = ax.barh(features, sample_counts, color=colors)
    
    # Add horizontal line at target
    ax.axvline(x=samples_per_feature, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Target ({samples_per_feature})')
    
    # Add value labels on bars
    for bar, count in zip(bars, sample_counts):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f'{int(count)}',
                ha='left', va='center', fontsize=9)
    
    ax.set_xlabel("Number of Marginal Samples", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_title(f"Shapley Sample Coverage ({dataset_name})", fontsize=12, fontweight='bold')

    plot_filename = f"feature_shapley_coverage_r{row_count:03d}.png" if include_row_count_in_name else "feature_shapley_coverage.png"
    plot_path = output_dir / plot_filename
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def _plot_search_budget(
    eval_count: int,
    eval_budget: int,
    permutation_runs: int,
    truncation_events: int,
    dataset_name: str,
    row_count: int,
    output_dir: Path,
    include_row_count_in_name: bool = False,
) -> Path:
    """Plot search budget utilization (eval count vs budget)."""
    remaining = max(0, eval_budget - eval_count)
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # Stacked horizontal bar
    categories = ["Evals Used", "Budget Remaining"]
    counts = [eval_count, remaining]
    colors = ["#2ecc71", "#95a5a6"]
    
    ax.barh(["Budget Utilization"], counts, color=colors, label=categories)
    
    # Add value labels
    x_offset = 0
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax.text(x_offset + count / 2, 0, f"{cat}\n{int(count)}", 
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        x_offset += count
    
    ax.set_xlim(0, eval_budget * 1.05)
    ax.set_xlabel(f"Evaluation Count (total budget: {eval_budget})", fontsize=11)
    ax.set_title(f"Search Budget Utilization ({dataset_name})\nPermutations: {int(permutation_runs)}, Truncations: {int(truncation_events)}", 
                 fontsize=12, fontweight='bold')
    ax.set_yticks([])

    plot_filename = f"shapley_search_budget_r{row_count:03d}.png" if include_row_count_in_name else "shapley_search_budget.png"
    plot_path = output_dir / plot_filename
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def _plot_permutation_progression(
    trace: list,
    permutation_runs: int,
    truncation_events: int,
    dataset_name: str,
    row_count: int,
    output_dir: Path,
    include_row_count_in_name: bool = False,
) -> Path:
    """Plot permutation progression (cumulative evals vs achievements)."""
    if not trace:
        return Path()
    
    # Extract objectives from trace
    eval_indices = np.arange(len(trace)) + 1
    objectives = [float(r.objective) for r in trace]
    cumulative_best = np.minimum.accumulate(objectives)
    
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    
    # Plot cumulative best found
    ax.plot(eval_indices, cumulative_best, 'o-', color='#3498db', linewidth=2, markersize=5, label='Best Found')
    
    # Add light background for objective history
    ax.plot(eval_indices, objectives, 'x', color='#95a5a6', alpha=0.5, markersize=4, label='All Evaluations')
    
    ax.set_xlabel("Evaluation Number", fontsize=11)
    ax.set_ylabel("Objective Value", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    ax.set_title(f"Permutation Progression ({dataset_name})\nTotal Perms: {int(permutation_runs)}, Truncations: {int(truncation_events)}", 
                 fontsize=12, fontweight='bold')

    plot_filename = f"shapley_search_progress_r{row_count:03d}.png" if include_row_count_in_name else "shapley_search_progress.png"
    plot_path = output_dir / plot_filename
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def _plot_rank_stability(
    shapley_rows: list[dict],
    dataset_name: str,
    row_count: int,
    output_dir: Path,
    include_row_count_in_name: bool = False,
) -> Path:
    """Plot feature rank stability (rank vs sample count, colored by Shapley value)."""
    features = [str(r.get("feature", "")) for r in shapley_rows]
    ranks = [int(r.get("shapley_rank", 0)) for r in shapley_rows]
    sample_counts = [int(r.get("n_marginal_samples", 0)) for r in shapley_rows]
    estimates = [float(r.get("shapley_value_est", np.nan)) for r in shapley_rows]

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    
    # Color by Shapley value estimate
    if estimates:
        est_min = float(np.nanmin(estimates))
        est_max = float(np.nanmax(estimates))
        if est_max > est_min:
            norm = matplotlib.colors.Normalize(vmin=est_min, vmax=est_max)
            colors = [plt.cm.RdYlGn(float(norm(v))) for v in estimates]
        else:
            colors = [plt.cm.RdYlGn(0.5) for _ in estimates]
    else:
        colors = ['gray'] * len(features)

    scatter = ax.scatter(ranks, sample_counts, c=estimates, cmap='RdYlGn', s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add feature labels
    for rank, count, feat in zip(ranks, sample_counts, features):
        ax.annotate(feat, (rank, count), fontsize=8, ha='center', va='center')
    
    ax.set_xlabel("Shapley Rank (1 = Highest)", fontsize=11)
    ax.set_ylabel("Number of Marginal Samples", fontsize=11)
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Shapley Estimate", fontsize=10)
    ax.set_title(f"Rank vs Sample Stability ({dataset_name})", fontsize=12, fontweight='bold')

    plot_filename = f"feature_shapley_rank_stability_r{row_count:03d}.png" if include_row_count_in_name else "feature_shapley_rank_stability.png"
    plot_path = output_dir / plot_filename
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path



def _tmc_shapley_subsets(
    dataset_dir: Path,
    dataset_prefix: str,
    surrogate_config_path: Path,
    row_count: int,
    lambda_drop: float,
    min_features: int,
    eval_budget: int,
    samples_per_feature: int,
    min_coalition_size: int,
    tmc_max_permutations: int,
    tmc_truncation_epsilon: float,
    tmc_bootstrap_resamples: int,
    disable_baselines_for_search: bool,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
    seed: int,
    keep_shapley_plots: bool = True,
    include_row_count_in_plot_names: bool = False,
) -> tuple[list[base.CandidateResult], list[base.CandidateResult], Path, Path, Path]:
    target_name = base._derive_target_name(dataset_dir.name, dataset_prefix)
    tmp_cfg_dir = base._forecast_sweeps_dir(dataset_dir) / "configs"
    _ = include_row_count_in_plot_names  # Reserved for parity with h_* plot-disambiguation flow.

    cfg = train_module.load_config(str(surrogate_config_path))
    full_features = tuple(cfg["data"]["input_columns"])
    feature_count = len(full_features)
    if feature_count <= min_features:
        raise ValueError(f"min_features={min_features} must be < number of features ({len(full_features)})")
    if eval_budget <= 0:
        raise ValueError("eval_budget must be > 0")

    rng = np.random.default_rng(seed)
    cache: dict[tuple[int, tuple[str, ...]], base.CandidateResult | None] = {}
    trace: list[base.CandidateResult] = []
    eval_count = 0
    start_time = time.time()
    shapley_samples: dict[str, list[float]] = {feat: [] for feat in full_features}
    permutation_runs = 0
    truncation_events = 0

    def _eval(features: tuple[str, ...]) -> base.CandidateResult | None:
        nonlocal eval_count
        key = base._candidate_key(row_count, features)
        if key in cache:
            return cache[key]
        if eval_count >= eval_budget:
            return None

        tag = base._feature_tag(features)
        result = base._evaluate_candidate(
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
        eval_count += 1
        if result is not None:
            trace.append(result)
        return result

    full_tuple = _ordered_tuple(list(full_features), full_features)
    full_result = _eval(full_tuple)
    if full_result is None:
        raise RuntimeError("Search budget exhausted before evaluating the full subset.")

    print(
        f"[SHAPLEY-TMC] Initial ({feature_count} features): objective={full_result.objective:.4f} "
        f"rmse={full_result.rmse:.6f} (evals: {eval_count}/{eval_budget}, ETA: {base._format_eta(start_time, eval_count, eval_budget)})"
    )

    min_with_feature = max(min_features, min_coalition_size if min_coalition_size > 0 else min_features)
    if min_with_feature >= feature_count:
        raise ValueError(
            f"Minimum coalition size {min_with_feature} must be < number of features ({feature_count})."
        )
    sample_target_per_feature = max(1, int(samples_per_feature))
    max_perms = int(tmc_max_permutations)
    should_continue = True

    min_permutation_runs = max(feature_count, 4)
    while should_continue and eval_count < eval_budget:
        # Stop when each feature has enough marginal samples.
        if permutation_runs >= min_permutation_runs and all(
            len(shapley_samples[f]) >= sample_target_per_feature for f in full_features
        ):
            break
        if max_perms > 0 and permutation_runs >= max_perms:
            break

        order = rng.permutation(full_features).tolist()
        for perm_order in (order, list(reversed(order))):
            if eval_count >= eval_budget:
                should_continue = False
                break
            if max_perms > 0 and permutation_runs >= max_perms:
                should_continue = False
                break

            permutation_runs += 1
            base_set = _ordered_tuple(perm_order[:min_with_feature], full_features)
            current_res = _eval(base_set)
            if current_res is None:
                should_continue = False
                break

            active_set = set(base_set)
            if abs(float(current_res.objective) - float(full_result.objective)) <= float(tmc_truncation_epsilon):
                for rem in perm_order[min_with_feature:]:
                    shapley_samples[rem].append(0.0)
                truncation_events += 1
                continue

            for feat in perm_order[min_with_feature:]:
                if eval_count >= eval_budget:
                    should_continue = False
                    break
                next_tuple = _ordered_tuple(list(active_set) + [feat], full_features)
                next_res = _eval(next_tuple)
                if next_res is None:
                    should_continue = False
                    break

                marginal = float(current_res.objective - next_res.objective)
                shapley_samples[feat].append(marginal)
                active_set.add(feat)
                current_res = next_res

                if abs(float(current_res.objective) - float(full_result.objective)) <= float(tmc_truncation_epsilon):
                    for rem in perm_order:
                        if rem not in active_set:
                            shapley_samples[rem].append(0.0)
                    truncation_events += 1
                    break

            if permutation_runs % 10 == 0 or eval_count >= eval_budget:
                print(
                    f"[SHAPLEY-TMC] perms={permutation_runs} evals={eval_count}/{eval_budget} "
                    f"ETA={base._format_eta(start_time, eval_count, eval_budget)}"
                )

    top_sorted = sorted(trace, key=lambda x: (x.objective, x.rmse, -x.n_features))
    if not top_sorted:
        raise RuntimeError("No successful candidate evaluations were produced.")

    shapley_rows: list[dict] = []
    for feature in full_features:
        vals = np.array(shapley_samples.get(feature, []), dtype=float)
        n = int(vals.size)
        mean_val = float(np.mean(vals)) if n > 0 else float("nan")
        std_val = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        median_val = float(np.median(vals)) if n > 0 else float("nan")

        if n > 0 and tmc_bootstrap_resamples > 0:
            idx = rng.integers(0, n, size=(int(tmc_bootstrap_resamples), n))
            means = vals[idx].mean(axis=1)
            ci_low = float(np.percentile(means, 2.5))
            ci_high = float(np.percentile(means, 97.5))
        else:
            ci_low = mean_val
            ci_high = mean_val

        shapley_rows.append(
            {
                "row_count": int(row_count),
                "feature": feature,
                "shapley_value_est": mean_val,
                "shapley_std": std_val,
                "shapley_median": median_val,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_marginal_samples": n,
                "permutation_runs": int(permutation_runs),
                "truncation_events": int(truncation_events),
                "eval_count": int(eval_count),
                "eval_budget": int(eval_budget),
            }
        )

    shapley_rows.sort(
        key=lambda r: (
            float("-inf") if not np.isfinite(r["shapley_value_est"]) else float(r["shapley_value_est"]),
            r["feature"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(shapley_rows, start=1):
        row["shapley_rank"] = int(rank)

    elapsed = time.time() - start_time
    print(
        f"[SHAPLEY-TMC] Complete: {eval_count}/{eval_budget} evaluations in "
        f"{int(elapsed // 60)}m {int(elapsed % 60)}s. "
        f"Best objective={top_sorted[0].objective:.4f} rmse={top_sorted[0].rmse:.6f} "
        f"n_features={top_sorted[0].n_features} perms={permutation_runs} truncations={truncation_events}"
    )

    shapley_csv = _write_shapley_round_artifact(dataset_dir, row_count, shapley_rows)
    shapley_rankings_json, seed_csv = _write_shapley_seed_artifacts(
        dataset_dir=dataset_dir,
        row_count=row_count,
        shapley_rows=shapley_rows,
        full_features=full_features,
        min_features=min_features,
    )

    if keep_shapley_plots:
        out_dir = base._forecast_sweeps_dir(dataset_dir)
        dataset_name = dataset_dir.name

        _plot_shapley_ranking_bar(
            shapley_rows=shapley_rows,
            dataset_name=dataset_name,
            row_count=row_count,
            output_dir=out_dir,
            include_row_count_in_name=include_row_count_in_plot_names,
        )
        print(f"[INFO] Wrote Shapley ranking bar plot")

        _plot_shapley_distribution(
            shapley_samples=shapley_samples,
            shapley_rows=shapley_rows,
            dataset_name=dataset_name,
            row_count=row_count,
            output_dir=out_dir,
            include_row_count_in_name=include_row_count_in_plot_names,
        )
        print(f"[INFO] Wrote Shapley distribution plot")

        _plot_shapley_coverage(
            shapley_rows=shapley_rows,
            samples_per_feature=samples_per_feature,
            dataset_name=dataset_name,
            row_count=row_count,
            output_dir=out_dir,
            include_row_count_in_name=include_row_count_in_plot_names,
        )
        print(f"[INFO] Wrote Shapley coverage plot")

        _plot_search_budget(
            eval_count=eval_count,
            eval_budget=eval_budget,
            permutation_runs=permutation_runs,
            truncation_events=truncation_events,
            dataset_name=dataset_name,
            row_count=row_count,
            output_dir=out_dir,
            include_row_count_in_name=include_row_count_in_plot_names,
        )
        print(f"[INFO] Wrote search budget plot")

        _plot_permutation_progression(
            trace=trace,
            permutation_runs=permutation_runs,
            truncation_events=truncation_events,
            dataset_name=dataset_name,
            row_count=row_count,
            output_dir=out_dir,
            include_row_count_in_name=include_row_count_in_plot_names,
        )
        print(f"[INFO] Wrote permutation progression plot")

        _plot_rank_stability(
            shapley_rows=shapley_rows,
            dataset_name=dataset_name,
            row_count=row_count,
            output_dir=out_dir,
            include_row_count_in_name=include_row_count_in_plot_names,
        )
        print(f"[INFO] Wrote rank stability plot")
    else:
        print("[INFO] Shapley diagnostic plots disabled by default (use --keep-shapley-plots to enable).")

    _write_shapley_samples_json(
        dataset_dir=dataset_dir,
        row_count=row_count,
        shapley_samples=shapley_samples,
        shapley_rows=shapley_rows,
    )
    print(f"[INFO] Wrote Shapley samples JSON")

    return top_sorted, trace, shapley_csv, shapley_rankings_json, seed_csv


def run_feature_selection_sweep(args: argparse.Namespace) -> int:
    _activate_shapley_sweep_namespace()

    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()

    single_plan = getattr(args, "_internal_single_plan", None)
    if single_plan is not None:
        plans = [single_plan]
    else:
        include_regular, include_res = base._resolve_dataset_inclusion(args)
        plans = base.discover_mc_dataset_plans(
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

    if args.run_baselines_in_search and args.disable_baselines_for_search:
        raise ValueError("Cannot use both --run-baselines-in-search and --disable-baselines-for-search.")

    # Match h_* phase policy: search is performance-oriented; final phase restores outputs.
    search_run_baselines = bool(args.run_baselines_in_search) and (not args.disable_baselines_for_search)
    search_disable_training_plots = not bool(args.keep_training_plots)
    search_disable_eval_plots = not bool(args.keep_eval_plots)
    search_save_plots = bool(args.keep_search_plots)
    final_run_baselines = True
    final_disable_training_plots = False
    final_disable_eval_plots = False

    print("\nExecution plan")
    print("-" * 100)
    print(f"Data root                 : {data_root}")
    print(f"Dataset prefix            : {args.dataset_prefix}")
    print(f"Config pattern            : {args.config_pattern}")
    print(f"Datasets found            : {len(plans)}")
    print(f"Eval budget               : {args.eval_budget}")
    print(f"Lambda drop               : {args.lambda_drop}")
    print(f"Top-K for final models    : {args.final_top_k}")
    print(f"TMC samples/feature       : {args.shapley_samples_per_feature}")
    print(f"TMC max permutations      : {args.tmc_max_permutations}")
    print(f"TMC truncation epsilon    : {args.tmc_truncation_epsilon}")
    print(f"TMC bootstrap resamples   : {args.tmc_bootstrap_resamples}")
    print(f"Dry run                   : {args.dry_run}")
    print(f"Keep train plots          : {args.keep_training_plots}")
    print(f"Keep eval plots           : {args.keep_eval_plots}")
    print(f"Keep search plots         : {args.keep_search_plots}")
    print(f"Show train logs           : {args.show_training_logs}")
    print(f"Search run baselines      : {search_run_baselines}")
    print(f"Search train plots enabled: {not search_disable_training_plots}")
    print(f"Search eval plots enabled : {not search_disable_eval_plots}")
    print(f"Search summary plots      : {search_save_plots}")
    print(f"Final run baselines       : {final_run_baselines}")
    print(f"Final eval plots enabled  : {not final_disable_eval_plots}")

    if args.dry_run:
        for plan in plans:
            surrogate = base._select_surrogate_config(plan.train_configs)
            cfg = train_module.load_config(str(surrogate))
            base_span = int(cfg["data"]["input_row_2"]) - int(cfg["data"]["input_row_1"])
            row_counts = base._parse_row_counts(args.row_counts, default_span=base_span)
            print(f"  - {plan.dataset_dir.name}: surrogate={surrogate.name}, row_counts={row_counts}")
        return 0

    failed = 0
    for plan in plans:
        print("\n" + "=" * 100)
        print(f"DATASET: {plan.dataset_dir.name}")
        print("=" * 100)

        surrogate_cfg = base._select_surrogate_config(plan.train_configs)
        surrogate_data = train_module.load_config(str(surrogate_cfg))["data"]
        base_span = int(surrogate_data["input_row_2"]) - int(surrogate_data["input_row_1"])
        row_counts = base._parse_row_counts(args.row_counts, default_span=base_span)
        include_row_count_in_plot_names = len(row_counts) > 1

        for row_count in row_counts:
            try:
                print(f"\n[SHAPLEY-TMC] rows={row_count} surrogate={surrogate_cfg.name}")
                top_sorted, trace, shapley_csv, shapley_rankings_json, seed_csv = _tmc_shapley_subsets(
                    dataset_dir=plan.dataset_dir,
                    dataset_prefix=args.dataset_prefix,
                    surrogate_config_path=surrogate_cfg,
                    row_count=row_count,
                    lambda_drop=args.lambda_drop,
                    min_features=args.min_features,
                    eval_budget=args.eval_budget,
                    samples_per_feature=args.shapley_samples_per_feature,
                    min_coalition_size=args.shapley_min_coalition_size,
                    tmc_max_permutations=args.tmc_max_permutations,
                    tmc_truncation_epsilon=args.tmc_truncation_epsilon,
                    tmc_bootstrap_resamples=args.tmc_bootstrap_resamples,
                    disable_baselines_for_search=not search_run_baselines,
                    disable_training_plots=search_disable_training_plots,
                    disable_eval_plots=search_disable_eval_plots,
                    suppress_training_logs=not args.show_training_logs,
                    seed=args.seed,
                    keep_shapley_plots=bool(args.keep_shapley_plots),
                    include_row_count_in_plot_names=include_row_count_in_plot_names,
                )
                selected = top_sorted[: args.final_top_k]
                trace_csv, selected_csv, plot_path = base._write_search_outputs(
                    dataset_dir=plan.dataset_dir,
                    row_count=row_count,
                    trace=trace,
                    selected=selected,
                    save_plots=search_save_plots,
                )
                print(f"[INFO] Wrote Shapley scores: {shapley_csv}")
                print(f"[INFO] Wrote Shapley rankings: {shapley_rankings_json}")
                print(f"[INFO] Wrote seed subsets: {seed_csv}")
                print(f"[INFO] Wrote search trace: {trace_csv}")
                print(f"[INFO] Wrote selected subsets: {selected_csv}")
                if search_save_plots:
                    print(f"[INFO] Wrote search plot: {plot_path}")
                else:
                    print("[INFO] Search Pareto plot disabled by default (use --keep-search-plots to enable).")

                final_metrics_csv = base._evaluate_selected_subsets_all_models(
                    dataset_plan=plan,
                    dataset_prefix=args.dataset_prefix,
                    selected=selected,
                    run_baselines_in_final=final_run_baselines,
                    disable_training_plots=final_disable_training_plots,
                    disable_eval_plots=final_disable_eval_plots,
                    suppress_training_logs=not args.show_training_logs,
                )
                print(f"[INFO] Wrote final model metrics: {final_metrics_csv}")

                # Keep post-process baseline artifacts aligned with h_* behavior.
                base._ensure_k01_baselines(plan, final_metrics_csv)
                dataset_eval_summary = base._write_dataset_evaluation_summary(plan, final_metrics_csv)
                if dataset_eval_summary is not None:
                    print(f"[INFO] Wrote dataset evaluation summary: {dataset_eval_summary}")

                if args.run_rolling_origin_cv:
                    cv_summary = base._run_rolling_origin_cv(plan, final_metrics_csv)
                    if cv_summary is not None:
                        print(f"[INFO] Wrote rolling-origin summary: {cv_summary}")

            except Exception as exc:
                failed += 1
                print(f"[ERROR] Dataset failed: {plan.dataset_dir.name}, row_count {row_count}")
                print(f"[ERROR] {exc}")
                if args.stop_on_error:
                    raise

    print("\nRun summary")
    print("-" * 100)
    print(f"Datasets completed: {len(plans) - failed}")
    print(f"Datasets failed   : {failed}")
    return 0 if failed == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature-selection sweeper using TMC-Shapley surrogate search and final full-model evaluation."
    )
    parser.add_argument("--data-root", type=str, default="data/output/regression")
    parser.add_argument("--dataset-prefix", type=str, default="MC")
    parser.add_argument("--config-pattern", type=str, default="config_*.yml")
    parser.add_argument("--limit-datasets", type=int, default=1)

    parser.add_argument("--row-counts", type=str, default=None)
    parser.add_argument("--min-features", type=int, default=4)
    parser.add_argument("--eval-budget", type=int, default=240)
    parser.add_argument("--lambda-drop", type=float, default=0.25)
    parser.add_argument("--final-top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--shapley-samples-per-feature", type=int, default=3)
    parser.add_argument("--shapley-min-coalition-size", type=int, default=0)
    parser.add_argument(
        "--tmc-max-permutations",
        type=int,
        default=0,
        help="Hard cap on TMC permutation runs (0 means no explicit permutation cap).",
    )
    parser.add_argument(
        "--tmc-truncation-epsilon",
        type=float,
        default=0.0025,
        help="Truncate a permutation path when objective is within this tolerance of the full-feature objective.",
    )
    parser.add_argument(
        "--tmc-bootstrap-resamples",
        type=int,
        default=300,
        help="Bootstrap resamples for 95%% CI around each feature's TMC-Shapley estimate.",
    )

    parser.add_argument("--include-regular", action="store_true")
    parser.add_argument("--include-res", action="store_true")
    parser.add_argument("--regular-only", action="store_true")
    parser.add_argument("--res-only", action="store_true")

    parser.add_argument("--disable-baselines-for-search", action="store_true")
    parser.add_argument(
        "--run-baselines-in-search",
        action="store_true",
        help="Enable baselines during search evaluations (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-training-plots",
        action="store_true",
        help="Keep per-candidate training plots during search (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-eval-plots",
        action="store_true",
        help="Keep per-candidate evaluation plots during search (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-search-plots",
        action="store_true",
        help="Keep search summary Pareto plots (disabled by default).",
    )
    parser.add_argument(
        "--keep-shapley-plots",
        action="store_true",
        help="Keep Shapley attribution/search diagnostic plots (disabled by default).",
    )
    parser.add_argument(
        "--show-training-logs",
        action="store_true",
        help="Show verbose model training logs (epoch metrics, sample-loading details).",
    )
    parser.add_argument(
        "--run-rolling-origin-cv",
        action="store_true",
        help="Run rolling-origin CV for the best k01 model after final metrics are written.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_feature_selection_sweep(args)


if __name__ == "__main__":
    sys.exit(main())

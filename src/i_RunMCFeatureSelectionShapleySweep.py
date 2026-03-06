"""
TMC-Shapley feature-selection sweeper for MC datasets.

This mirrors the orchestration in h_RunMCFeatureSelectionSweep.py:
- Discover MC datasets/configs
- Search feature subsets with a surrogate model
- Write search trace + selected subsets under forecasts/Shapley_sweeps
- Retrain/evaluate all discovered model configs on top-K subsets
- Write feature_sweep_final_metrics.csv

Search strategy here differs:
- One-pass TMC-Shapley (permutation-based) under a strict eval budget cap
- Antithetic permutations (forward + reverse) to reduce variance
- Early truncation when subset utility is close to the full-feature utility

To get docs:
python src/i_RunMCFeatureSelectionShapleySweep.py --help

Quick test:
python src/i_RunMCFeatureSelectionShapleySweep.py --limit-datasets 1 --eval-budget 60 --final-top-k 2 --dry-run

Full run:
python src/i_RunMCFeatureSelectionShapleySweep.py `
  --dataset-prefix MC `
  --limit-datasets 0 `
  --eval-budget 240 `
  --final-top-k 4 `
  --shapley-samples-per-feature 3 `
  --tmc-truncation-epsilon 0.0025 `
  --tmc-bootstrap-resamples 300 `
  --run-baselines-in-final

Additional args:
--row-counts 24,48,72: evaluate multiple input window sizes.
--eval-budget: main runtime/quality knob.
--shapley-samples-per-feature: more samples = stabler Shapley estimates, slower.
--tmc-max-permutations: hard cap on permutation runs (0 = no explicit cap).
--tmc-truncation-epsilon: larger = more truncation/faster, potentially noisier.
--regular-only / --res-only / --include-regular / --include-res: dataset filtering.
--keep-search-plots, --keep-training-plots, --keep-eval-plots: retain more figures/artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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

    seed_csv = out_dir / f"feature_seed_subsets_r{row_count:03d}.csv"
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
    return top_sorted, trace, shapley_csv, shapley_rankings_json, seed_csv


def run_feature_selection_sweep(args: argparse.Namespace) -> int:
    _activate_shapley_sweep_namespace()

    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()

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
    print(f"Search eval plots enabled : {not search_disable_eval_plots}")
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
                    include_row_count_in_plot_names=include_row_count_in_plot_names,
                )
                selected = top_sorted[: args.final_top_k]
                trace_csv, selected_csv, plot_path = base._write_search_outputs(
                    dataset_dir=plan.dataset_dir,
                    row_count=row_count,
                    trace=trace,
                    selected=selected,
                    save_plots=bool(args.keep_search_plots),
                )
                print(f"[INFO] Wrote Shapley scores: {shapley_csv}")
                print(f"[INFO] Wrote Shapley rankings: {shapley_rankings_json}")
                print(f"[INFO] Wrote seed subsets: {seed_csv}")
                print(f"[INFO] Wrote search trace: {trace_csv}")
                print(f"[INFO] Wrote selected subsets: {selected_csv}")
                print(f"[INFO] Wrote search plot: {plot_path}")

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
    parser.add_argument("--run-baselines-in-final", action="store_true")
    parser.add_argument(
        "--run-baselines-in-search",
        action="store_true",
        help="Enable baselines during search evaluations (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-training-plots",
        action="store_true",
        help="Keep per-model training plots during feature sweeps (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-eval-plots",
        action="store_true",
        help="Keep per-config evaluation plots during feature sweeps (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-search-plots",
        action="store_true",
        help="Keep feature-search Pareto plots (disabled by default for speed).",
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

"""
Combined Shapley -> optimizer feature-selection pipeline.

Intended use:
- Run attribution first (`i_RunMCFeatureSelectionShapleySweep.py`) to generate
    ranked Shapley features and seed subset artifacts.
- Run optimizer second (`h_RunMCFeatureSelectionSweep.py`) using those seed
    subsets to warm-start beam+swap search.
- Optionally run multiple optimizer seeds in one command via
    `--optimizer-seeds`.

Stage behavior:
- Stage 1 (Shapley): writes `feature_seed_subsets_r###.csv` under
    `forecasts/Shapley_sweeps`.
- Stage 2 (Optimizer): writes optimizer artifacts under
    `forecasts/feature_sweeps` and by default reads Shapley seed subsets
    unless `--seed-subsets-csv` is provided.
- Search phases keep standard temporal-by-coverage split behavior
    (target 70/30 by default).
- Final top-K phases enforce minimum test coverage (>=5 test samples):
    latest train samples are moved into test when needed; subsets with
    fewer than 5 total split samples are skipped.
- Supports stage skipping for iterative workflows:
    `--skip-shapley-stage` or `--skip-optimizer-stage`.

Common usage:
1) Dry-run full pipeline:
        python src/l_RunMCShapleyOptimizerPipeline.py --dry-run

2) Full pipeline with explicit budgets:
        python src/l_RunMCShapleyOptimizerPipeline.py --dataset-prefix MC --limit-datasets 14 --shapley-eval-budget 270 --optimizer-eval-budget 270 --final-top-k 4
        python src/l_RunMCShapleyOptimizerPipeline.py --dataset-prefix MC --limit-datasets 14 --shapley-eval-budget 270 --optimizer-eval-budget 270 --final-top-k 4 --shapley-samples-per-feature 10 --tmc-truncation-epsilon 0.0005 --beam-width 16 --max-rounds 30 --max-swap-attempts 200
3) Multi-seed optimizer stage in one run:
        python src/l_RunMCShapleyOptimizerPipeline.py \
            --dataset-prefix MC --optimizer-seeds 7,11,19

4) Reuse existing Shapley artifacts and skip Stage 1:
        python src/l_RunMCShapleyOptimizerPipeline.py \
            --skip-shapley-stage --optimizer-seeds 7,11,19

5) Override seed subset source with explicit CSV:
        python src/l_RunMCShapleyOptimizerPipeline.py \
            --seed-subsets-csv data/output/regression/MC_exColor_res/forecasts/Shapley_sweeps/feature_seed_subsets_r671.csv \
            --skip-shapley-stage
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h_RunMCFeatureSelectionSweep as optimizer_module
import i_RunMCFeatureSelectionShapleySweep as shapley_module


_NAMESPACE_ENV = "WQ_FEATURE_SWEEP_NAMESPACE"
_SHAPLEY_NAMESPACE = "Shapley_sweeps"
_OPTIMIZER_DEFAULT_NAMESPACE = "feature_sweeps"


def _set_pipeline_stage_namespace(namespace: str | None) -> None:
    """Set or clear sweep namespace for stage isolation in one-process runs."""
    if namespace is None or str(namespace).strip() == "":
        os.environ.pop(_NAMESPACE_ENV, None)
        return
    os.environ[_NAMESPACE_ENV] = str(namespace).strip()


def _resolve_data_root(data_root_value: str) -> Path:
    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(data_root_value)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()
    return data_root


def _discover_pipeline_plans(args: argparse.Namespace) -> list[optimizer_module.DatasetPlan]:
    include_regular, include_res = optimizer_module._resolve_dataset_inclusion(args)
    return optimizer_module.discover_mc_dataset_plans(
        data_root=_resolve_data_root(args.data_root),
        dataset_prefix=args.dataset_prefix,
        config_pattern=args.config_pattern,
        limit_datasets=args.limit_datasets,
        include_regular=include_regular,
        include_res=include_res,
    )


def _parse_seed_list(raw: str | None) -> list[int]:
    if raw is None or str(raw).strip() == "":
        return []
    out: list[int] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _build_shapley_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=args.data_root,
        dataset_prefix=args.dataset_prefix,
        config_pattern=args.config_pattern,
        limit_datasets=args.limit_datasets,
        row_counts=args.row_counts,
        min_features=args.min_features,
        eval_budget=args.shapley_eval_budget,
        lambda_drop=args.lambda_drop,
        final_top_k=args.final_top_k,
        seed=args.shapley_seed,
        shapley_samples_per_feature=args.shapley_samples_per_feature,
        shapley_min_coalition_size=args.shapley_min_coalition_size,
        tmc_max_permutations=args.tmc_max_permutations,
        tmc_truncation_epsilon=args.tmc_truncation_epsilon,
        tmc_bootstrap_resamples=args.tmc_bootstrap_resamples,
        include_regular=args.include_regular,
        include_res=args.include_res,
        regular_only=args.regular_only,
        res_only=args.res_only,
        disable_baselines_for_search=args.disable_baselines_for_search,
        run_baselines_in_search=args.run_baselines_in_search,
        keep_training_plots=args.keep_training_plots,
        keep_eval_plots=args.keep_eval_plots,
        keep_search_plots=args.keep_search_plots,
        keep_shapley_plots=args.keep_shapley_plots,
        show_training_logs=args.show_training_logs,
        run_rolling_origin_cv=args.run_rolling_origin_cv,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
    )


def _build_optimizer_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=args.data_root,
        dataset_prefix=args.dataset_prefix,
        config_pattern=args.config_pattern,
        limit_datasets=args.limit_datasets,
        row_counts=args.row_counts,
        min_features=args.min_features,
        beam_width=args.beam_width,
        max_rounds=args.max_rounds,
        no_improve_patience=args.no_improve_patience,
        eval_budget=args.optimizer_eval_budget,
        max_swap_attempts=args.max_swap_attempts,
        lambda_drop=args.lambda_drop,
        final_top_k=args.final_top_k,
        seed=args.optimizer_seed,
        seed_subsets_csv=args.seed_subsets_csv,
        # Default behavior: if no explicit seed CSV is provided, try Shapley seed subsets.
        seed_subsets_from_shapley=(not bool(args.seed_subsets_csv)),
        max_seed_subsets=args.max_seed_subsets,
        include_regular=args.include_regular,
        include_res=args.include_res,
        regular_only=args.regular_only,
        res_only=args.res_only,
        disable_baselines_for_search=args.disable_baselines_for_search,
        run_baselines_in_search=args.run_baselines_in_search,
        keep_training_plots=args.keep_training_plots,
        keep_eval_plots=args.keep_eval_plots,
        keep_search_plots=args.keep_search_plots,
        show_training_logs=args.show_training_logs,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a combined Shapley-attribution + seeded-optimizer feature-selection pipeline."
    )

    parser.add_argument("--data-root", type=str, default="data/output/regression")
    parser.add_argument("--dataset-prefix", type=str, default="MC")
    parser.add_argument("--config-pattern", type=str, default="config_*.yml")
    parser.add_argument("--limit-datasets", type=int, default=14)

    parser.add_argument("--row-counts", type=str, default=None)
    parser.add_argument("--min-features", type=int, default=4)
    parser.add_argument("--lambda-drop", type=float, default=0.25)
    parser.add_argument("--final-top-k", type=int, default=4)

    parser.add_argument("--shapley-eval-budget", type=int, default=270)
    parser.add_argument("--shapley-seed", type=int, default=42)
    parser.add_argument("--shapley-samples-per-feature", type=int, default=10)
    parser.add_argument("--shapley-min-coalition-size", type=int, default=0)
    parser.add_argument("--tmc-max-permutations", type=int, default=0)
    parser.add_argument("--tmc-truncation-epsilon", type=float, default=0.0005)
    parser.add_argument("--tmc-bootstrap-resamples", type=int, default=300)

    parser.add_argument("--optimizer-eval-budget", type=int, default=270)
    parser.add_argument("--optimizer-seed", type=int, default=42)
    parser.add_argument(
        "--optimizer-seeds",
        type=str,
        default=None,
        help="Comma-separated optimizer seeds for multi-seed runs (overrides --optimizer-seed).",
    )
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--no-improve-patience", type=int, default=3)
    parser.add_argument("--max-swap-attempts", type=int, default=200)

    parser.add_argument("--seed-subsets-csv", type=str, default=None)
    parser.add_argument("--max-seed-subsets", type=int, default=0)

    parser.add_argument("--include-regular", action="store_true")
    parser.add_argument("--include-res", action="store_true")
    parser.add_argument("--regular-only", action="store_true")
    parser.add_argument("--res-only", action="store_true")

    parser.add_argument("--disable-baselines-for-search", action="store_true")
    parser.add_argument("--run-baselines-in-search", action="store_true")

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
    parser.add_argument("--show-training-logs", action="store_true")
    parser.add_argument("--run-rolling-origin-cv", action="store_true")

    parser.add_argument("--skip-shapley-stage", action="store_true")
    parser.add_argument("--skip-optimizer-stage", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.skip_shapley_stage and args.skip_optimizer_stage:
        raise ValueError("Cannot skip both stages.")

    if args.run_baselines_in_search and args.disable_baselines_for_search:
        raise ValueError("Cannot use both --run-baselines-in-search and --disable-baselines-for-search.")

    plans = _discover_pipeline_plans(args)
    if not plans:
        print("No matching datasets/configs found.")
        return 1

    data_root_resolved = _resolve_data_root(args.data_root)

    print("\nPipeline plan")
    print("-" * 100)
    print(f"Shapley stage enabled     : {not args.skip_shapley_stage}")
    print(f"Optimizer stage enabled   : {not args.skip_optimizer_stage}")
    print(f"Data root                 : {data_root_resolved}")
    print(f"Dataset prefix            : {args.dataset_prefix}")
    print(f"Datasets found            : {len(plans)}")
    print(f"Shapley eval budget       : {args.shapley_eval_budget}")
    print(f"Optimizer eval budget     : {args.optimizer_eval_budget}")
    optimizer_seeds = _parse_seed_list(args.optimizer_seeds)
    if not optimizer_seeds:
        optimizer_seeds = [int(args.optimizer_seed)]
    print(f"Optimizer seeds           : {optimizer_seeds}")
    print(f"Seed subsets CSV override : {args.seed_subsets_csv}")
    print(f"Use Shapley seed subsets  : {not bool(args.seed_subsets_csv)}")
    print("Search split policy       : temporal coverage split (target 70/30)")
    print("Final top-K split policy  : enforce >=5 test samples (skip if total<5)")
    print(f"Dry run                   : {args.dry_run}")

    prior_namespace = os.environ.get(_NAMESPACE_ENV)
    shapley_failures = 0
    optimizer_failures = 0
    try:
        for plan_idx, plan in enumerate(plans, start=1):
            print("\n" + "=" * 100)
            print(f"[PIPELINE] DATASET {plan_idx}/{len(plans)}: {plan.dataset_dir.name}")
            print("=" * 100)

            if not args.skip_shapley_stage:
                _set_pipeline_stage_namespace(_SHAPLEY_NAMESPACE)
                shapley_args = _build_shapley_args(args)
                shapley_args._internal_single_plan = plan

                print("\n[PIPELINE] Stage 1/2: Running Shapley attribution sweep...")
                print(f"[PIPELINE] Stage 1 namespace      : {_SHAPLEY_NAMESPACE}")
                shapley_rc = shapley_module.run_feature_selection_sweep(shapley_args)
                if shapley_rc != 0:
                    shapley_failures += 1
                    print(f"[PIPELINE] Shapley stage failed for {plan.dataset_dir.name} with exit code {shapley_rc}.")
                    if args.stop_on_error:
                        return shapley_rc
                    continue

            if not args.skip_optimizer_stage:
                # Critical isolation: optimizer outputs must not inherit Shapley namespace.
                _set_pipeline_stage_namespace(None)
                print("\n[PIPELINE] Stage 2/2: Running seeded optimizer sweep(s)...")
                print(f"[PIPELINE] Stage 2 namespace      : {_OPTIMIZER_DEFAULT_NAMESPACE}")
                for seed_idx, seed in enumerate(optimizer_seeds, start=1):
                    run_args = _build_optimizer_args(args)
                    run_args.seed = int(seed)
                    run_args._internal_single_plan = plan

                    print(
                        f"[PIPELINE] Optimizer run {seed_idx}/{len(optimizer_seeds)} "
                        f"for {plan.dataset_dir.name} (seed={seed})"
                    )
                    optimizer_rc = optimizer_module.run_feature_selection_sweep(run_args)
                    if optimizer_rc != 0:
                        optimizer_failures += 1
                        print(
                            f"[PIPELINE] Optimizer seed {seed} failed for "
                            f"{plan.dataset_dir.name} with exit code {optimizer_rc}."
                        )
                        if args.stop_on_error:
                            return optimizer_rc
    finally:
        # Restore caller environment after pipeline completes.
        if prior_namespace is None:
            os.environ.pop(_NAMESPACE_ENV, None)
        else:
            os.environ[_NAMESPACE_ENV] = prior_namespace

    if shapley_failures > 0 or optimizer_failures > 0:
        print("\n[PIPELINE] Completed with failures.")
        print(f"[PIPELINE] Shapley failures        : {shapley_failures}")
        print(f"[PIPELINE] Optimizer failures      : {optimizer_failures}")
        return 2

    print("\n[PIPELINE] Complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

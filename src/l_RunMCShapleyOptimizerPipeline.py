"""
Combined Shapley -> optimizer feature-selection pipeline.

Stage 1 (attribution):
- Run TMC-Shapley sweep to produce ranked feature artifacts and seed subsets.

Stage 2 (optimization):
- Run beam+swap optimizer sweep using Shapley seed subsets to initialize search.

This script is intentionally orchestration-only and delegates heavy work to:
- i_RunMCFeatureSelectionShapleySweep.py
- h_RunMCFeatureSelectionSweep.py
"""

from __future__ import annotations

import argparse
import sys

import h_RunMCFeatureSelectionSweep as optimizer_module
import i_RunMCFeatureSelectionShapleySweep as shapley_module


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
        run_baselines_in_final=args.run_baselines_in_final,
        keep_training_plots=args.keep_training_plots,
        keep_eval_plots=args.keep_eval_plots,
        keep_search_plots=args.keep_search_plots,
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
        seed_subsets_from_shapley=(not bool(args.seed_subsets_csv)) and (not args.skip_shapley_stage),
        max_seed_subsets=args.max_seed_subsets,
        include_regular=args.include_regular,
        include_res=args.include_res,
        regular_only=args.regular_only,
        res_only=args.res_only,
        disable_baselines_for_search=args.disable_baselines_for_search,
        run_baselines_in_search=args.run_baselines_in_search,
        run_baselines_in_final=args.run_baselines_in_final,
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
    parser.add_argument("--limit-datasets", type=int, default=1)

    parser.add_argument("--row-counts", type=str, default=None)
    parser.add_argument("--min-features", type=int, default=4)
    parser.add_argument("--lambda-drop", type=float, default=0.25)
    parser.add_argument("--final-top-k", type=int, default=4)

    parser.add_argument("--shapley-eval-budget", type=int, default=180)
    parser.add_argument("--shapley-seed", type=int, default=42)
    parser.add_argument("--shapley-samples-per-feature", type=int, default=3)
    parser.add_argument("--shapley-min-coalition-size", type=int, default=0)
    parser.add_argument("--tmc-max-permutations", type=int, default=0)
    parser.add_argument("--tmc-truncation-epsilon", type=float, default=0.0025)
    parser.add_argument("--tmc-bootstrap-resamples", type=int, default=300)

    parser.add_argument("--optimizer-eval-budget", type=int, default=180)
    parser.add_argument("--optimizer-seed", type=int, default=42)
    parser.add_argument(
        "--optimizer-seeds",
        type=str,
        default=None,
        help="Comma-separated optimizer seeds for multi-seed runs (overrides --optimizer-seed).",
    )
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--no-improve-patience", type=int, default=3)
    parser.add_argument("--max-swap-attempts", type=int, default=60)

    parser.add_argument("--seed-subsets-csv", type=str, default=None)
    parser.add_argument("--max-seed-subsets", type=int, default=0)

    parser.add_argument("--include-regular", action="store_true")
    parser.add_argument("--include-res", action="store_true")
    parser.add_argument("--regular-only", action="store_true")
    parser.add_argument("--res-only", action="store_true")

    parser.add_argument("--disable-baselines-for-search", action="store_true")
    parser.add_argument("--run-baselines-in-search", action="store_true")
    parser.add_argument("--run-baselines-in-final", action="store_true")

    parser.add_argument("--keep-training-plots", action="store_true")
    parser.add_argument("--keep-eval-plots", action="store_true")
    parser.add_argument("--keep-search-plots", action="store_true")
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

    print("\nPipeline plan")
    print("-" * 100)
    print(f"Shapley stage enabled     : {not args.skip_shapley_stage}")
    print(f"Optimizer stage enabled   : {not args.skip_optimizer_stage}")
    print(f"Data root                 : {args.data_root}")
    print(f"Dataset prefix            : {args.dataset_prefix}")
    print(f"Shapley eval budget       : {args.shapley_eval_budget}")
    print(f"Optimizer eval budget     : {args.optimizer_eval_budget}")
    optimizer_seeds = _parse_seed_list(args.optimizer_seeds)
    if not optimizer_seeds:
        optimizer_seeds = [int(args.optimizer_seed)]
    print(f"Optimizer seeds           : {optimizer_seeds}")
    print(f"Seed subsets CSV override : {args.seed_subsets_csv}")
    print(f"Use Shapley seed subsets  : {(not bool(args.seed_subsets_csv)) and (not args.skip_shapley_stage)}")
    print(f"Dry run                   : {args.dry_run}")

    if not args.skip_shapley_stage:
        print("\n[PIPELINE] Stage 1/2: Running Shapley attribution sweep...")
        shapley_rc = shapley_module.run_feature_selection_sweep(_build_shapley_args(args))
        if shapley_rc != 0:
            print(f"[PIPELINE] Shapley stage failed with exit code {shapley_rc}.")
            return shapley_rc

    if not args.skip_optimizer_stage:
        print("\n[PIPELINE] Stage 2/2: Running seeded optimizer sweep(s)...")
        optimizer_failures = 0
        for idx, seed in enumerate(optimizer_seeds, start=1):
            run_args = _build_optimizer_args(args)
            run_args.seed = int(seed)
            print(f"[PIPELINE] Optimizer run {idx}/{len(optimizer_seeds)} (seed={seed})")
            optimizer_rc = optimizer_module.run_feature_selection_sweep(run_args)
            if optimizer_rc != 0:
                optimizer_failures += 1
                print(f"[PIPELINE] Optimizer seed {seed} failed with exit code {optimizer_rc}.")
                if args.stop_on_error:
                    return optimizer_rc

        if optimizer_failures > 0:
            print(f"[PIPELINE] Optimizer stage completed with failures: {optimizer_failures}/{len(optimizer_seeds)}")
            return 2

    print("\n[PIPELINE] Complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

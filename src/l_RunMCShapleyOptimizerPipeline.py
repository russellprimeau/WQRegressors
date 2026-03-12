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
- CV tuning policy for XGBoost:
    * The pipeline enables CV tuning the first time it encounters an XGB config
      (train-set-only, no test exposure).
    * The first tuned run writes a dataset-level cache
      `forecasts/xgb_cv_tuning_cache.json`.
    * Subsequent runs reuse cached hyperparameters for consistent regularization
      across feature subsets and to reduce compute.
    * Stage A (pre-Shapley): run CV tuning on the full feature set.
    * Stage B (post-Shapley): run CV tuning on top-k Shapley subsets and write
      comparison artifacts (`xgb_cv_topk_params_r###.csv/.png`) under
      `forecasts/Shapley_sweeps`.
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
python src/l_RunMCShapleyOptimizerPipeline.py --dataset-prefix MC --limit-datasets 3 --shapley-eval-budget 3 --optimizer-eval-budget 3 --final-top-k 1 --shapley-samples-per-feature 10 --tmc-truncation-epsilon 0.0005 --beam-width 16 --max-rounds 30 --max-swap-attempts 200 --keep-shapley-plots --keep-search-plots
python src/l_RunMCShapleyOptimizerPipeline.py --dataset-prefix MC --limit-datasets 14 --shapley-eval-budget 270 --optimizer-eval-budget 270 --final-top-k 4 --shapley-samples-per-feature 10 --tmc-truncation-epsilon 0.0005 --beam-width 16 --max-rounds 30 --max-swap-attempts 200 --keep-shapley-plots --keep-search-plots --parallel-evaluators 1


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
import contextlib
import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h_RunMCFeatureSelectionSweep as optimizer_module
import i_RunMCFeatureSelectionShapleySweep as shapley_module
import e_Train as train_module


_NAMESPACE_ENV = "WQ_FEATURE_SWEEP_NAMESPACE"
_SHAPLEY_NAMESPACE = "Shapley_sweeps"
_OPTIMIZER_DEFAULT_NAMESPACE = "feature_sweeps"
_CONSENSUS_VERSION = 2
_CONSENSUS_METHOD = "elite_trial_observed_candidate"
_ELITE_REL_MARGIN = 0.03
_ELITE_ABS_MARGIN = 0.0
_ELITE_INCLUDE_ONE_SE = True
_STAGE2_CVTUNE_DEFAULT_N_TRIALS = 50
_STAGE2_CVTUNE_DEFAULT_RANDOM_START_TRIALS = 10


class _TeeTextIO:
    """Write-through stream that mirrors output to multiple text streams."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _build_pipeline_log_path(data_root: Path) -> Path:
    log_dir = (data_root.parent / "pipeline_logs").resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return log_dir / f"shapley_optimizer_pipeline_{timestamp}.log"


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


def _load_raw_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text)
    if path.suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def _write_raw_config(path: Path, cfg: dict) -> None:
    cfg = dict(cfg)
    cfg.pop("__config_dir", None)
    if path.suffix in {".yml", ".yaml"}:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        return
    if path.suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return
    raise ValueError(f"Unsupported config format: {path.suffix}")


def _resolve_data_dir(cfg: dict, cfg_path: Path) -> Path:
    data_cfg = cfg.get("data", {}) or {}
    data_dir_raw = data_cfg.get("data_dir")
    if not data_dir_raw:
        raise ValueError(f"Missing data.data_dir in {cfg_path}")
    config_dir = cfg.get("__config_dir", str(cfg_path.parent))
    return Path(train_module._resolve_path_from_config(data_dir_raw, config_dir))


def _shapley_cache_path(dataset_dir: Path) -> Path:
    return dataset_dir / "forecasts" / _SHAPLEY_NAMESPACE / "xgb_cv_tuning_cache.json"


def _feature_sweep_cache_path(dataset_dir: Path) -> Path:
    return dataset_dir / "forecasts" / _OPTIMIZER_DEFAULT_NAMESPACE / "xgb_cv_tuning_cache.json"


def _apply_tuned_hyperparameters(cfg: dict, tuned: dict) -> dict:
    hyper_cfg = cfg.setdefault("hyperparameters", {})
    tuned_hyper = tuned.get("tuned_hyperparameters") or tuned.get("best_params")
    if not isinstance(tuned_hyper, dict) or not tuned_hyper:
        return cfg
    hyper_cfg.update(tuned_hyper)
    cv_cfg = hyper_cfg.get("cv_tuning")
    if isinstance(cv_cfg, dict):
        cv_cfg.setdefault("enabled", False)
        cv_cfg["enabled"] = False
        cv_cfg["source"] = "xgb_cv_tuning_cache.json"
    return cfg


def _sync_feature_sweep_cache(plans: list[optimizer_module.DatasetPlan]) -> None:
    for plan in plans:
        cache_path = _feature_sweep_cache_path(plan.dataset_dir)
        if not cache_path.exists():
            continue
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_payload = json.load(f)
        except Exception as exc:
            print(f"[PIPELINE][WARN] Failed to read feature_sweeps cache {cache_path}: {exc}")
            continue

        for cfg_path in plan.train_configs:
            try:
                cfg = train_module.load_config(str(cfg_path))
                model_type = str(cfg.get("model_type", ""))
                if model_type not in {"xgb_regressor", "xgb_classifier"}:
                    continue
                raw_cfg = _load_raw_config(cfg_path)
                raw_cfg = _apply_tuned_hyperparameters(raw_cfg, cache_payload)
                _write_raw_config(cfg_path, raw_cfg)
                print(f"[PIPELINE] Applied feature_sweeps cache {cache_path} -> {cfg_path}")
            except Exception as exc:
                print(f"[PIPELINE][WARN] Could not apply feature_sweeps cache for {cfg_path}: {exc}")


def _find_first_xgb_config(plan: optimizer_module.DatasetPlan) -> Path | None:
    for cfg_path in plan.train_configs:
        try:
            cfg = train_module.load_config(str(cfg_path))
            if str(cfg.get("model_type", "")) in {"xgb_regressor", "xgb_classifier"}:
                return cfg_path
        except Exception:
            continue
    return None


def _run_full_feature_cv_tuning(plan: optimizer_module.DatasetPlan) -> None:
    base_cfg_path = _find_first_xgb_config(plan)
    if base_cfg_path is None:
        return
    cache_path = _shapley_cache_path(plan.dataset_dir)
    if cache_path.exists():
        return

    cfg = train_module.load_config(str(base_cfg_path))
    cfg = train_module.merge_with_defaults(cfg, cfg.get("model_type", "xgb_regressor"))
    hyper_cfg = cfg.setdefault("hyperparameters", {})
    cv_cfg = hyper_cfg.get("cv_tuning")
    if isinstance(cv_cfg, dict):
        cv_cfg["enabled"] = True
        cv_cfg["cache_path"] = str(cache_path)
        cv_cfg["artifact_dir"] = str(_shapley_cache_path(plan.dataset_dir).parent)
    else:
        hyper_cfg["cv_tuning"] = {
            "enabled": True,
            "cache_path": str(cache_path),
            "artifact_dir": str(_shapley_cache_path(plan.dataset_dir).parent),
        }

    train_samples, _ = train_module.load_and_split_data(cfg)
    model_kind = "classifier" if str(cfg.get("model_type")) == "xgb_classifier" else "regressor"
    metric_key = "eval_metric" if model_kind == "classifier" else "metric"
    cast_y = (lambda v: int(round(v))) if model_kind == "classifier" else None

    print(f"[PIPELINE] Stage A CV tuning (full features): {base_cfg_path}")
    best_params, summary = train_module.run_xgb_cv_tuning_only(
        config=cfg,
        train_samples=train_samples,
        model_kind=model_kind,
        metric_key=metric_key,
        cast_y=cast_y,
        use_cache=False,
        write_cache=False,
    )

    stage_a_candidates = _extract_elite_trials(
        summary or {},
        rel_margin=_ELITE_REL_MARGIN,
        abs_margin=_ELITE_ABS_MARGIN,
        include_one_se=_ELITE_INCLUDE_ONE_SE,
    )
    stage_a_selected, stage_a_meta = _select_consensus_from_candidates(stage_a_candidates)

    if stage_a_selected:
        selected_params = stage_a_selected
    else:
        selected_params = best_params or {}
        if selected_params:
            stage_a_meta = {
                "method": "fallback_run_xgb_cv_tuning_only_best",
                "version": _CONSENSUS_VERSION,
                "reason": "no_elite_candidates",
            }

    if selected_params:
        summary = dict(summary or {})
        summary["selection_method"] = stage_a_meta.get("method")
        summary["selection_version"] = stage_a_meta.get("version")
        summary["elite_rel_margin"] = float(_ELITE_REL_MARGIN)
        summary["elite_abs_margin"] = float(_ELITE_ABS_MARGIN)
        summary["elite_include_one_se"] = bool(_ELITE_INCLUDE_ONE_SE)
        summary["elite_candidate_count"] = int(len(stage_a_candidates))
        summary["parsed_trial_count"] = int(len((summary.get("trials") or [])))
        summary["selected_params_source"] = "stability_consensus" if stage_a_selected else "best_params_fallback"

        train_module._write_xgb_cv_tuning_cache(cache_path, hyper_cfg, selected_params, summary)
        print(
            "[PIPELINE] Stage A selected params via "
            f"{summary['selected_params_source']} "
            f"(elite={summary['elite_candidate_count']}, rel_margin={_ELITE_REL_MARGIN:.2%})."
        )


def _load_topk_shapley_subsets(plan: optimizer_module.DatasetPlan, row_count: int, top_k: int) -> list[tuple[str, ...]]:
    seed_csv = optimizer_module._resolve_seed_subsets_csv_path(
        dataset_dir=plan.dataset_dir,
        row_count=row_count,
        explicit_path=None,
        from_shapley=True,
    )
    if seed_csv is None or not seed_csv.exists():
        return []
    try:
        df = pd.read_csv(seed_csv)
    except Exception:
        return []
    if "features" not in df.columns:
        return []
    subsets: list[tuple[str, ...]] = []
    for raw in df["features"].tolist():
        feats = tuple([p.strip() for p in str(raw).split("|") if p.strip()])
        if feats:
            subsets.append(feats)
        if len(subsets) >= top_k:
            break
    return subsets


def _write_topk_param_reports(
    out_dir: Path,
    row_count: int,
    records: list[dict],
) -> None:
    if not records:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    csv_path = out_dir / f"xgb_cv_topk_params_r{row_count:03d}.csv"
    df.to_csv(csv_path, index=False)

    param_cols = [c for c in df.columns if c.startswith("param__")]
    if not param_cols:
        return
    data = []
    labels = []
    for col in param_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        data.append(series.to_numpy(dtype=float))
        labels.append(col.replace("param__", ""))
    if not data:
        return
    plt.figure(figsize=(max(8, 1.2 * len(labels)), 4.8))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title("Stage B: CV-tuned parameter distribution (top-k subsets)")
    plt.ylabel("Value")
    plt.grid(axis="y", alpha=0.35)
    plot_path = out_dir / f"xgb_cv_topk_params_r{row_count:03d}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()


def _compute_consensus_params(records: list[dict]) -> dict:
    if not records:
        return {}
    df = pd.DataFrame(records)
    param_cols = [c for c in df.columns if c.startswith("param__")]
    consensus: dict = {}
    for col in param_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        # Determine if values are effectively integers
        is_int_like = np.all(np.isclose(vals, np.round(vals)))
        if is_int_like:
            mode_val = vals.round().mode()
            if not mode_val.empty:
                consensus[col.replace("param__", "")] = int(mode_val.iloc[0])
        else:
            consensus[col.replace("param__", "")] = float(vals.mean())
    return consensus


def _extract_elite_trials(
    summary: dict,
    *,
    rel_margin: float = _ELITE_REL_MARGIN,
    abs_margin: float = 0.0,
    include_one_se: bool = _ELITE_INCLUDE_ONE_SE,
) -> list[dict]:
    """Return near-optimal trials from one CV run using run-local tolerance."""
    trials = summary.get("trials") if isinstance(summary, dict) else None
    if not isinstance(trials, list) or not trials:
        return []

    parsed = []
    for item in trials:
        if not isinstance(item, dict):
            continue
        params = item.get("params")
        if not isinstance(params, dict) or not params:
            continue
        mean = pd.to_numeric(pd.Series([item.get("mean_score")]), errors="coerce").iloc[0]
        if pd.isna(mean):
            continue
        std = pd.to_numeric(pd.Series([item.get("std_score")]), errors="coerce").iloc[0]
        parsed.append(
            {
                "trial": item.get("trial"),
                "mean_score": float(mean),
                "std_score": float(std) if not pd.isna(std) else float("nan"),
                "params": dict(params),
            }
        )

    if not parsed:
        return []

    best_mean = min(x["mean_score"] for x in parsed)
    best_std_vals = [x["std_score"] for x in parsed if np.isfinite(x["std_score"]) and x["mean_score"] == best_mean]
    best_std = float(best_std_vals[0]) if best_std_vals else 0.0

    rel_tol = abs(float(rel_margin)) * max(abs(best_mean), 1e-12)
    tol = max(float(abs_margin), rel_tol)
    if include_one_se:
        tol = max(tol, max(0.0, best_std))

    elite = [x for x in parsed if x["mean_score"] <= (best_mean + tol)]
    if not elite:
        elite = [min(parsed, key=lambda x: x["mean_score"])]

    out = []
    for row in elite:
        delta = max(0.0, float(row["mean_score"] - best_mean))
        std = float(row["std_score"]) if np.isfinite(row["std_score"]) else 0.0
        # Prefer low delta and low fold spread; denominator keeps scale bounded.
        score_weight = 1.0 / (1.0 + (delta / max(tol, 1e-12)))
        stability_weight = 1.0 / (1.0 + std)
        out.append(
            {
                **row,
                "best_mean": float(best_mean),
                "tolerance": float(tol),
                "delta_from_best": float(delta),
                "candidate_weight_raw": float(score_weight * stability_weight),
            }
        )
    return out


def _params_signature(params: dict) -> str:
    safe = {str(k): params[k] for k in sorted(params.keys())}
    return json.dumps(safe, sort_keys=True, separators=(",", ":"))


def _select_consensus_from_candidates(candidates: list[dict]) -> tuple[dict, dict]:
    """Choose one observed hyperparameter set from elite-trial candidates."""
    if not candidates:
        return {}, {
            "method": _CONSENSUS_METHOD,
            "version": _CONSENSUS_VERSION,
            "reason": "no_candidates",
        }

    by_sig: dict[str, dict] = {}
    for cand in candidates:
        params = cand.get("params")
        if not isinstance(params, dict) or not params:
            continue
        sig = _params_signature(params)
        bucket = by_sig.setdefault(
            sig,
            {
                "params": dict(params),
                "support": 0,
                "weight": 0.0,
                "avg_delta_num": 0.0,
                "avg_std_num": 0.0,
                "sources": [],
            },
        )
        w = float(max(0.0, cand.get("candidate_weight", cand.get("candidate_weight_raw", 0.0))))
        delta = float(max(0.0, cand.get("delta_from_best", 0.0)))
        std = cand.get("std_score", float("nan"))
        std = float(std) if np.isfinite(std) else 0.0
        bucket["support"] += 1
        bucket["weight"] += w
        bucket["avg_delta_num"] += w * delta
        bucket["avg_std_num"] += w * std
        bucket["sources"].append(
            {
                "row_count": cand.get("row_count"),
                "feature_tag": cand.get("feature_tag"),
                "trial": cand.get("trial"),
            }
        )

    if not by_sig:
        return {}, {
            "method": _CONSENSUS_METHOD,
            "version": _CONSENSUS_VERSION,
            "reason": "invalid_candidates",
        }

    flattened = []
    for sig, bucket in by_sig.items():
        denom = max(bucket["weight"], 1e-12)
        flattened.append(
            {
                "signature": sig,
                "params": bucket["params"],
                "support": int(bucket["support"]),
                "weight": float(bucket["weight"]),
                "weighted_avg_delta": float(bucket["avg_delta_num"] / denom),
                "weighted_avg_std": float(bucket["avg_std_num"] / denom),
                "sources": bucket["sources"],
            }
        )

    # Deterministic preference: max weight, then support, then lower delta/std.
    flattened.sort(
        key=lambda x: (
            -x["weight"],
            -x["support"],
            x["weighted_avg_delta"],
            x["weighted_avg_std"],
            x["signature"],
        )
    )
    chosen = flattened[0]
    meta = {
        "method": _CONSENSUS_METHOD,
        "version": _CONSENSUS_VERSION,
        "candidate_count": len(candidates),
        "unique_param_sets": len(flattened),
        "selected_signature": chosen["signature"],
        "selected_weight": chosen["weight"],
        "selected_support": chosen["support"],
        "selected_weighted_avg_delta": chosen["weighted_avg_delta"],
        "selected_weighted_avg_std": chosen["weighted_avg_std"],
        "top_candidates": [
            {
                "signature": c["signature"],
                "support": c["support"],
                "weight": c["weight"],
                "weighted_avg_delta": c["weighted_avg_delta"],
                "weighted_avg_std": c["weighted_avg_std"],
            }
            for c in flattened[:10]
        ],
    }
    return dict(chosen["params"]), meta


def _write_stage_b_consensus_artifacts(
    cvtune_dir: Path,
    all_records: list[dict],
    candidates: list[dict],
    selected_params: dict,
    meta: dict,
) -> None:
    cvtune_dir.mkdir(parents=True, exist_ok=True)
    derivation_path = cvtune_dir / "xgb_cv_consensus_derivation.json"
    with open(derivation_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "artifact_version": _CONSENSUS_VERSION,
                "consensus_method": _CONSENSUS_METHOD,
                "selected_params": selected_params,
                "meta": meta,
            },
            f,
            indent=2,
        )

    metadata_path = cvtune_dir / "xgb_cv_stage_b_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "artifact_version": _CONSENSUS_VERSION,
                "consensus_method": _CONSENSUS_METHOD,
                "records": all_records,
                "elite_candidates": candidates,
                "selected_params": selected_params,
            },
            f,
            indent=2,
        )


def _run_topk_subset_cv_tuning(
    plan: optimizer_module.DatasetPlan,
    row_counts: list[int],
    top_k: int,
) -> None:
    base_cfg_path = _find_first_xgb_config(plan)
    if base_cfg_path is None:
        return
    base_cfg = train_module.load_config(str(base_cfg_path))
    base_cfg = train_module.merge_with_defaults(base_cfg, base_cfg.get("model_type", "xgb_regressor"))

    stage_a_cache = _shapley_cache_path(plan.dataset_dir)
    stage_a_payload = None
    if stage_a_cache.exists():
        try:
            with open(stage_a_cache, "r", encoding="utf-8") as f:
                stage_a_payload = json.load(f)
        except Exception:
            stage_a_payload = None

    all_records: list[dict] = []
    elite_candidates: list[dict] = []

    if stage_a_payload is not None:
        stage_a_summary = stage_a_payload.get("cv_summary") if isinstance(stage_a_payload, dict) else None
        stage_a_candidates = _extract_elite_trials(
            stage_a_summary or {},
            rel_margin=_ELITE_REL_MARGIN,
            abs_margin=_ELITE_ABS_MARGIN,
            include_one_se=_ELITE_INCLUDE_ONE_SE,
        )
        if stage_a_candidates:
            norm = sum(max(0.0, c.get("candidate_weight_raw", 0.0)) for c in stage_a_candidates)
            norm = norm if norm > 0 else float(len(stage_a_candidates))
            for c in stage_a_candidates:
                weight = max(0.0, c.get("candidate_weight_raw", 0.0))
                c["candidate_weight"] = float((weight if norm > 0 else 1.0) / norm)
                c["row_count"] = "stage_a"
                c["feature_tag"] = "stage_a_full"
                elite_candidates.append(c)

    for row_count in row_counts:
        subsets = _load_topk_shapley_subsets(plan, row_count, top_k)
        if not subsets:
            continue
        out_dir = plan.dataset_dir / "forecasts" / _SHAPLEY_NAMESPACE / "CVtune"
        records: list[dict] = []
        tmp_dir = out_dir / "_cv_tuning_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        if stage_a_payload is not None:
            stage_a_params = stage_a_payload.get("tuned_hyperparameters") or stage_a_payload.get("best_params") or {}
            stage_a_record = {
                "row_count": int(row_count),
                "feature_tag": "stage_a_full",
                "n_features": int(len(base_cfg.get("data", {}).get("input_columns", []))),
                "input_dim": int(len(base_cfg.get("data", {}).get("input_columns", []))),
                "n_train_samples": None,
                "seed": (stage_a_payload.get("cv_summary") or {}).get("seed"),
                "metric": (stage_a_payload.get("cv_summary") or {}).get("metric"),
                "best_score": (stage_a_payload.get("cv_summary") or {}).get("best_score"),
            }
            for k, v in stage_a_params.items():
                stage_a_record[f"param__{k}"] = v
            records.append(stage_a_record)

        for feats in subsets:
            feature_tag = optimizer_module._feature_tag(feats)
            cfg_path = optimizer_module._prepare_variant_config(
                base_config_path=base_cfg_path,
                row_count=row_count,
                features=feats,
                feature_tag=feature_tag,
                tmp_dir=tmp_dir,
                forced_data_dir=plan.dataset_dir,
            )
            # Stage 2-only CVtune defaults: keep subset tuning lightweight.
            raw_cfg = _load_raw_config(cfg_path)
            raw_hyper = raw_cfg.setdefault("hyperparameters", {})
            raw_cv_cfg = raw_hyper.get("cv_tuning")
            if isinstance(raw_cv_cfg, dict):
                raw_cv_cfg.setdefault("n_trials", int(_STAGE2_CVTUNE_DEFAULT_N_TRIALS))
                raw_cv_cfg.setdefault(
                    "random_start_trials",
                    int(_STAGE2_CVTUNE_DEFAULT_RANDOM_START_TRIALS),
                )
            else:
                raw_hyper["cv_tuning"] = {
                    "n_trials": int(_STAGE2_CVTUNE_DEFAULT_N_TRIALS),
                    "random_start_trials": int(_STAGE2_CVTUNE_DEFAULT_RANDOM_START_TRIALS),
                }
            _write_raw_config(cfg_path, raw_cfg)

            cfg = train_module.load_config(str(cfg_path))
            cfg = train_module.merge_with_defaults(cfg, cfg.get("model_type", "xgb_regressor"))
            hyper_cfg = cfg.setdefault("hyperparameters", {})
            cv_cfg = hyper_cfg.get("cv_tuning")
            if isinstance(cv_cfg, dict):
                cv_cfg["enabled"] = True
                cv_cfg["artifact_dir"] = str(out_dir / f"subset_{feature_tag}")
                cv_cfg.setdefault("n_trials", int(_STAGE2_CVTUNE_DEFAULT_N_TRIALS))
                cv_cfg.setdefault(
                    "random_start_trials",
                    int(_STAGE2_CVTUNE_DEFAULT_RANDOM_START_TRIALS),
                )
                # Vary seed per subset to avoid identical trial sequences.
                cv_cfg["seed"] = int(abs(hash(feature_tag)) % 10_000_000)
            else:
                hyper_cfg["cv_tuning"] = {
                    "enabled": True,
                    "artifact_dir": str(out_dir / f"subset_{feature_tag}"),
                    "n_trials": int(_STAGE2_CVTUNE_DEFAULT_N_TRIALS),
                    "random_start_trials": int(_STAGE2_CVTUNE_DEFAULT_RANDOM_START_TRIALS),
                    "seed": int(abs(hash(feature_tag)) % 10_000_000),
                }

            train_samples, _ = train_module.load_and_split_data(cfg)
            model_kind = "classifier" if str(cfg.get("model_type")) == "xgb_classifier" else "regressor"
            metric_key = "eval_metric" if model_kind == "classifier" else "metric"
            cast_y = (lambda v: int(round(v))) if model_kind == "classifier" else None

            best_params, summary = train_module.run_xgb_cv_tuning_only(
                config=cfg,
                train_samples=train_samples,
                model_kind=model_kind,
                metric_key=metric_key,
                cast_y=cast_y,
                use_cache=False,
                write_cache=False,
            )

            run_candidates = _extract_elite_trials(
                summary or {},
                rel_margin=_ELITE_REL_MARGIN,
                abs_margin=_ELITE_ABS_MARGIN,
                include_one_se=_ELITE_INCLUDE_ONE_SE,
            )
            if run_candidates:
                norm = sum(max(0.0, c.get("candidate_weight_raw", 0.0)) for c in run_candidates)
                norm = norm if norm > 0 else float(len(run_candidates))
                for c in run_candidates:
                    weight = max(0.0, c.get("candidate_weight_raw", 0.0))
                    c["candidate_weight"] = float((weight if norm > 0 else 1.0) / norm)
                    c["row_count"] = int(row_count)
                    c["feature_tag"] = feature_tag
                    elite_candidates.append(c)

            input_dim = int(len(cfg.get("data", {}).get("input_columns", [])))
            record = {
                "row_count": int(row_count),
                "feature_tag": feature_tag,
                "n_features": int(len(feats)),
                "input_dim": input_dim,
                "n_train_samples": int(len(train_samples)),
                "seed": (summary.get("seed") if isinstance(summary, dict) else None),
                "metric": summary.get("metric"),
                "best_score": summary.get("best_score"),
            }
            for k, v in (best_params or {}).items():
                record[f"param__{k}"] = v
            records.append(record)

        all_records.extend(records)

        _write_topk_param_reports(out_dir, row_count, records)

    # Build consensus across all records and write feature_sweeps cache + summary
    cvtune_dir = plan.dataset_dir / "forecasts" / _SHAPLEY_NAMESPACE / "CVtune"
    consensus, consensus_meta = _select_consensus_from_candidates(elite_candidates)
    if not consensus and all_records:
        consensus = _compute_consensus_params(all_records)
        consensus_meta = {
            "method": "fallback_paramwise_mean_mode",
            "version": 1,
            "reason": "no_elite_candidates",
        }

    if consensus:
        _write_stage_b_consensus_artifacts(cvtune_dir, all_records, elite_candidates, consensus, consensus_meta)
    if consensus:
        feature_cache = _feature_sweep_cache_path(plan.dataset_dir)
        base_cfg_for_cache = train_module.load_config(str(base_cfg_path))
        base_cfg_for_cache = train_module.merge_with_defaults(base_cfg_for_cache, base_cfg_for_cache.get("model_type", "xgb_regressor"))
        hyper_cfg = base_cfg_for_cache.get("hyperparameters", {})
        train_module._write_xgb_cv_tuning_cache(feature_cache, hyper_cfg, consensus, {
            "enabled": True,
            "metric": None,
            "source": "stage_b_consensus",
            "consensus_method": consensus_meta.get("method"),
            "consensus_version": consensus_meta.get("version"),
            "elite_rel_margin": float(_ELITE_REL_MARGIN),
            "elite_abs_margin": float(_ELITE_ABS_MARGIN),
            "elite_include_one_se": bool(_ELITE_INCLUDE_ONE_SE),
            "consensus_derivation_path": str(cvtune_dir / "xgb_cv_consensus_derivation.json"),
            "consensus_metadata_path": str(cvtune_dir / "xgb_cv_stage_b_metadata.json"),
        })
        summary_path = cvtune_dir / "xgb_cv_consensus_params.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "consensus_params": consensus,
                "consensus_meta": consensus_meta,
            }, f, indent=2)


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
        parallel_evaluators=args.parallel_evaluators,
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
    parser.add_argument(
        "--parallel-evaluators",
        type=int,
        default=1,
        help="Number of parallel candidate evaluators for optimizer beam rounds (1 keeps sequential behavior).",
    )
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

    data_root_resolved = _resolve_data_root(args.data_root)
    log_path = _build_pipeline_log_path(data_root_resolved)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, mode="w", encoding="utf-8") as log_file:
        tee_stdout = _TeeTextIO(sys.stdout, log_file)
        tee_stderr = _TeeTextIO(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            print(f"[PIPELINE] Logging terminal output to: {log_path}")
            return _run_pipeline(args, data_root_resolved)


def _run_pipeline(args: argparse.Namespace, data_root_resolved: Path) -> int:

    if args.skip_shapley_stage and args.skip_optimizer_stage:
        raise ValueError("Cannot skip both stages.")

    if args.run_baselines_in_search and args.disable_baselines_for_search:
        raise ValueError("Cannot use both --run-baselines-in-search and --disable-baselines-for-search.")

    plans = _discover_pipeline_plans(args)
    if not plans:
        print("No matching datasets/configs found.")
        return 1


    print("\nPipeline plan")
    print("-" * 100)
    print(f"Shapley stage enabled     : {not args.skip_shapley_stage}")
    print(f"Optimizer stage enabled   : {not args.skip_optimizer_stage}")
    print(f"Data root                 : {data_root_resolved}")
    print(f"Dataset prefix            : {args.dataset_prefix}")
    print(f"Datasets found            : {len(plans)}")
    print(f"Shapley eval budget       : {args.shapley_eval_budget}")
    print(f"Optimizer eval budget     : {args.optimizer_eval_budget}")
    print(f"Parallel evaluators       : {max(1, int(args.parallel_evaluators))}")
    optimizer_seeds = _parse_seed_list(args.optimizer_seeds)
    if not optimizer_seeds:
        optimizer_seeds = [int(args.optimizer_seed)]
    print(f"Optimizer seeds           : {optimizer_seeds}")
    print(f"Seed subsets CSV override : {args.seed_subsets_csv}")
    print(f"Use Shapley seed subsets  : {not bool(args.seed_subsets_csv)}")
    print("Search split policy       : temporal coverage split (target 70/30), require >=5 independent test groups")
    print("Final top-K split policy  : hard fail per variant when independent/valid minimum is not met; loop continues unless --stop-on-error")
    print(f"Dry run                   : {args.dry_run}")

    prior_namespace = os.environ.get(_NAMESPACE_ENV)
    shapley_failures = 0
    optimizer_failures = 0
    try:
        for plan_idx, plan in enumerate(plans, start=1):
            print("\n" + "=" * 100)
            print(f"[PIPELINE] DATASET {plan_idx}/{len(plans)}: {plan.dataset_dir.name}")
            print("=" * 100)

            surrogate_cfg = optimizer_module._select_surrogate_config(plan.train_configs)
            surrogate_data = train_module.load_config(str(surrogate_cfg))["data"]
            base_span = int(surrogate_data["input_row_2"]) - int(surrogate_data["input_row_1"])
            row_counts = optimizer_module._parse_row_counts(args.row_counts, default_span=base_span)

            _run_full_feature_cv_tuning(plan)

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

                _run_topk_subset_cv_tuning(plan, row_counts=row_counts, top_k=int(args.final_top_k))

            if not args.skip_optimizer_stage:
                # Critical isolation: optimizer outputs must not inherit Shapley namespace.
                _set_pipeline_stage_namespace(None)
                _sync_feature_sweep_cache([plan])
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

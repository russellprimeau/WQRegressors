"""
Beam+swap feature-selection sweeper for MC datasets.

Search strategy:
- Surrogate-guided beam backward elimination with swap refinement.
- Objective: `objective = (1 - r2) + lambda_drop * drop_rate`.
- `r2` is pooled across rolling-origin cross-validation folds by default
    (`--cv-folds`, default 5). Scoring candidates on a single 70/30 holdout of 12-47
    independent samples cannot separate the near-best subsets from one another: in
    practice subsets sharing fewer than half their features score within 0.01, so the
    argmin is close to arbitrary and 240 evaluations against one holdout also bias the
    reported accuracy upward. Folds are fixed for the whole search, so no candidate is
    ever compared against a different evaluation set. `--cv-folds 0` restores the old
    single-holdout objective.
- Selection applies a one-standard-error rule (`--selection-tolerance-se`): among the
    subsets within one standard error of the best objective, the smallest is chosen.
    The standard error comes from the spread across folds, so the rule is inert when
    cross-validation is disabled rather than resting on a fabricated interval.
- `drop_rate` is computed from raw sample coverage after MC replicate collapse.
- Search and final sweep-generated train/evaluate runs enforce a minimum of 5
    independent non-replicate test samples. Cross-validation folds are exempt: their
    membership is pinned, and a rebalance would silently move segments between train
    and test for one candidate but not the others.
- The surrogate family (`--surrogate-model`, default `xgb`) scores every candidate and
    therefore fixes the feature set that *all* families then use. When the reported
    winner for a target is a different family, that is a real limitation of the design
    and should be stated rather than left implicit.

Then:
- Retrain/evaluate all discovered model configs on top-K subsets.
- When a 70/30 split is short, the latest train groups are moved into test; if
    fewer than 5 independent groups exist overall, that subset/model evaluation
    fails compliance for that variant (outer loops continue unless stop-on-error).
- Write trace, selected subsets, and final metrics to the active sweep namespace.

Key CLI groups (detailed):
- Data selection:
    `--data-root PATH`: Root directory containing regression dataset folders.
    `--dataset-prefix PREFIX`: Dataset name prefix filter (for example, `MC`).
    `--config-pattern GLOB`: Glob used to discover per-dataset train configs.
    `--limit-datasets N`: Max matching datasets to process (`0` means all).
    `--include-regular`: Include regular (non-`_res`) datasets.
    `--include-res`: Include `_res` datasets.
    `--regular-only`: Include only regular datasets.
    `--res-only`: Include only `_res` datasets.
- Search controls:
    `--row-counts CSV_INTS`: Comma-separated input window sizes to evaluate.
    `--min-features N`: Minimum feature count allowed during elimination.
    `--beam-width N`: Number of best candidates retained each search round.
    `--max-rounds N`: Max beam-elimination rounds before swap refinement.
    `--no-improve-patience N`: Stop beam rounds after N non-improving rounds.
    `--eval-budget N`: Max candidate evaluations per dataset/row-count search.
    `--max-swap-attempts N`: Cap on swap-refinement attempts.
    `--lambda-drop FLOAT`: Penalty weight for sample drop rate in objective.
    `--seed N`: Random seed for candidate ordering and swap sampling.
    `--cv-folds N`: Rolling-origin folds per candidate (0 disables; default 5).
    `--cv-min-train-fraction F`: Segment fraction reserved as the initial training run.
    `--selection-tolerance-se F`: One-SE band, in standard errors (0 selects the argmin).
    `--retention-tolerance F`: Objective band defining the near-optimal set.
    `--surrogate-model NAME`: Model family that scores candidates during the search.
- Seeded optimizer controls:
    `--seed-subsets-csv PATH`: Seed subset CSV (or directory containing per-row seed CSVs).
    `--seed-subsets-from-shapley`: Load seeds from
        `forecasts/Shapley_sweeps/feature_seed_subsets_r###_d########.csv`
        (legacy fallback: `feature_seed_subsets_r###.csv`).
    `--max-seed-subsets N`: Cap loaded seed subsets (`0` means no explicit cap).
- Final/model controls:
    `--final-top-k N`: Number of best discovered subsets retrained in final stage.
- Baselines/logging:
    `--disable-baselines-for-search`: Disable baseline evals during search for speed.
    `--run-baselines-in-search`: Enable baseline evals during search (mutually exclusive with disable flag).
    `--show-training-logs`: Show verbose model training/sample logs.
- Artifacts/runtime:
        `--keep-training-plots`: Enable per-candidate training plots during search phase.
            Default is disabled for speed; final top-K evaluation still keeps plots.
        `--keep-eval-plots`: Enable per-candidate evaluation plots (for example, boxplots)
            during search phase. Default is disabled for speed; final top-K keeps plots.
        `--keep-search-plots`: Enable search summary plots
            (`feature_search_pareto_r###.png`, feature-importance charts).
            Default is disabled for speed.
    `--dry-run`: Print discovered execution plan and exit.
    `--stop-on-error`: Raise immediately on first dataset failure.

Search/output behavior:
- Search artifacts are written under `forecasts/feature_sweeps` unless
    `WQ_FEATURE_SWEEP_NAMESPACE` overrides the namespace.
- Writes `feature_search_trace_r###.csv`, `feature_selected_subsets_r###.csv`,
    `feature_retention_frequency_r###.csv`, optional `feature_search_pareto_r###.png`,
    and `feature_sweep_final_metrics.csv`.
- `feature_retention_frequency_r###.csv` records how often each predictor appears among
    the subsets within `--retention-tolerance` of the best objective. Report this rather
    than the single winning subset: a predictor retained by every near-optimal subset is
    a result, one retained by 40% of them is a coin flip the search did not resolve.
    `z14_SelectionStability.py` regenerates it from traces already on disk.
- Search-phase defaults are intentionally artifact-light:
    candidate runs write required training/evaluation artifacts
    (model outputs, split files, `evaluation_summary.csv`) while plot-heavy
    outputs are opt-in via `--keep-training-plots`, `--keep-eval-plots`, and
    `--keep-search-plots`.
- Maintains baseline/report compatibility via post-final hooks
    (`_ensure_k01_baselines`, dataset-level `evaluation_summary.csv`).

Examples:
python src/h_RunMCFeatureSelectionSweep.py --dry-run

python src/h_RunMCFeatureSelectionSweep.py --dataset-prefix MC --limit-datasets 0 --eval-budget 240 --final-top-k 4

python src/h_RunMCFeatureSelectionSweep.py --dataset-prefix MC --eval-budget 180 --seed-subsets-from-shapley

python src/h_RunMCFeatureSelectionSweep.py --dataset-prefix MC --eval-budget 180 --seed-subsets-csv data/output/regression/MC_exColor_res/forecasts/Shapley_sweeps/feature_seed_subsets_r671_d1234abcd.csv --max-seed-subsets 6

# Opt in to search-phase plots when diagnosing candidate behavior:
python src/h_RunMCFeatureSelectionSweep.py --dataset-prefix MC --eval-budget 60 --keep-training-plots --keep-eval-plots --keep-search-plots
"""
from __future__ import annotations
import contextlib
import argparse
import copy
import glob
import hashlib
import io
import json
import re
import shutil
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import importlib.util
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import yaml
import e_Train as train_module
import f_Evaluate as eval_module
from utils.notifications import notify
from utils import provenance
import os
import unicodedata
import seaborn as sns
import traceback
from dataclasses import dataclass
from pathlib import Path
from utils.training import load_samples, group_samples_by_segment, SampleComplianceError
from utils.crossval_rolling_origin import rolling_origin_block_splits
from utils.selection_stability import selection_stability_from_trace
from utils.training import aggregation_slug
from utils.names import clean_target_label, label as names_label
from utils.plotstyle import PAGE_WIDTH_IN, apply_paper_style, legend_above, save_figure

apply_paper_style()


def _force_utf8_console() -> None:
    """Stop a single unencodable character from aborting a stage.

    On Windows the console encoding defaults to cp1252, which cannot represent an
    arrow. One such character in a progress message raised UnicodeEncodeError inside a
    broad ``except Exception``, and the whole MLR k-cluster integration was skipped
    with nothing but a warning to show for it. Reconfiguring the streams removes the
    failure mode rather than the characters, so a stage can no longer be lost to a
    glyph. ``errors="replace"`` keeps output flowing on any stream that still cannot
    represent something.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_console()


SUPPORTED_CONFIG_SUFFIXES = {".yml", ".yaml", ".json"}
FINAL_TOPK_MIN_TEST_SAMPLES = 5
BASELINE_MODEL_IDS = {"naive", "seasonal", "linear"}
FINAL_METRICS_MODEL_ORDER = [
    "gp_regressor",
    "transformer",
    "xgb_regressor",
    "mlr",
    "mlr_avg12",
    "mlr_avgall",
    "naive",
    "seasonal",
    "linear",
]
FINAL_METRICS_MODEL_STYLE = {
    "gp_regressor": {"label": "GP", "color": "#1f77b4", "hatch": ""},
    "transformer": {"label": "Transformer", "color": "#ff7f0e", "hatch": ""},
    "xgb_regressor": {"label": "XGB", "color": "#2ca02c", "hatch": ""},
    "mlr": {"label": "MLR", "color": "#d62728", "hatch": ""},
    "mlr_avg12": {"label": "MLR-12", "color": "#e377c2", "hatch": ""},
    "mlr_avgall": {"label": "MLR-All", "color": "#9467bd", "hatch": ""},
    "naive": {"label": "Naive", "color": "#7f7f7f", "hatch": "//"},
    "seasonal": {"label": "Seasonal", "color": "#17becf", "hatch": "//"},
    "linear": {"label": "Linear", "color": "#bcbd22", "hatch": "//"},
}

_EXPECTED_EVAL_METRIC_SEMANTICS = "independent_sample_primary"

# Families the surrogate measurement will not consider. Neither appears in
# FINAL_METRICS_MODEL_ORDER, so neither is reported: measuring them would spend fits
# choosing a scorer that could never produce a reported result.
_SURROGATE_EXCLUDED_TOKENS = ("lstm", "recurrent_transformer", "classifier")

# Model types `_train_single_config` can actually train. A config naming anything else
# fails at the end of the final stage with "Unknown model_type", after the subset has
# already been selected -- so it costs time on every target and contributes nothing.
_TRAINABLE_MODEL_TYPES = {
    "transformer",
    "gp_regressor",
    "xgb_regressor",
    "xgb_classifier",
}

# Only these predictors currently carry uncertainty distributions used to
# generate meaningful Monte Carlo perturbation replicates.
UNCERTAINTY_DISTRIBUTION_FEATURES = {
    "Pfl - Sp Cond (microS_cm)",
    "Pfl - pH",
    "Pfl - DO (% Sat)",
    "Pfl - Turbidity (FNU)",
    "Pfl - fDOM (RFU)",
    "Pfl - fDOM (QSU)",
}


def _candidate_uses_uncertainty_distributions(features: tuple[str, ...]) -> bool:
    return any(str(feat) in UNCERTAINTY_DISTRIBUTION_FEATURES for feat in features)


def _extract_required_independent_metric(model_row: dict, key: str, context: str) -> float:
    """Read a required primary independent-sample metric as a finite float."""
    val = pd.to_numeric(model_row.get(key, np.nan), errors="coerce")
    out = float(val) if np.isfinite(val) else float("nan")
    if not np.isfinite(out):
        raise ValueError(
            f"Missing or non-finite independent metric '{key}' in {context}. "
            "Primary scoring metrics must come from independent-sample aggregation."
        )
    return out


def _validate_eval_metric_contract(model_row: dict, context: str) -> None:
    """Enforce metric contract for strict downstream scoring/selection."""
    semantics = str(model_row.get("metric_semantics", "")).strip()
    if semantics != _EXPECTED_EVAL_METRIC_SEMANTICS:
        raise ValueError(
            f"Unexpected metric semantics '{semantics or '<missing>'}' in {context}; "
            f"expected '{_EXPECTED_EVAL_METRIC_SEMANTICS}'."
        )

    contract_version = pd.to_numeric(model_row.get("metric_contract_version", np.nan), errors="coerce")
    if not np.isfinite(contract_version) or int(contract_version) < 1:
        raise ValueError(
            f"Missing/invalid metric_contract_version in {context}; expected integer >= 1."
        )

    required_keys = ("rmse", "mae", "n_test_independent")
    for key in required_keys:
        _extract_required_independent_metric(model_row, key, context=context)


def _normalize_baseline_label(value: object) -> "str | None":
    text = str(value).strip().lower()
    if "naive" in text:
        return "naive"
    if "seasonal" in text:
        return "seasonal"
    if "linear" in text:
        return "linear"
    return None


def _is_baseline_model_value(value: object) -> bool:
    text = str(value).strip().lower()
    return text in BASELINE_MODEL_IDS


def _select_best_model_by_min_skill_rmse(
    df_rank: "pd.DataFrame",
) -> "pd.Series | None":
    """Select the best non-baseline model row by RMSE-based minimum skill.

    min_skill = (rmse_best_baseline - rmse_model) / rmse_best_baseline

    Returns the best-skill model row, or None if baseline rows are absent,
    no ML model rows exist, or the best baseline RMSE is non-positive.
    """
    if df_rank.empty or "rmse" not in df_rank.columns or "model" not in df_rank.columns:
        return None
    is_baseline = df_rank["model"].apply(_is_baseline_model_value)
    baseline_rows = df_rank[is_baseline]
    ml_rows = df_rank[~is_baseline].copy()
    if baseline_rows.empty or ml_rows.empty:
        return None
    baseline_rmse = pd.to_numeric(baseline_rows["rmse"], errors="coerce").dropna()
    if baseline_rmse.empty:
        return None
    best_baseline_rmse = float(baseline_rmse.min())
    if not np.isfinite(best_baseline_rmse) or best_baseline_rmse <= 0:
        return None
    ml_rmse = pd.to_numeric(ml_rows["rmse"], errors="coerce")
    ml_rows["_min_skill"] = (best_baseline_rmse - ml_rmse) / best_baseline_rmse
    valid = ml_rows[ml_rows["_min_skill"].notna() & np.isfinite(ml_rows["_min_skill"])]
    if valid.empty:
        return None
    return valid.loc[valid["_min_skill"].idxmax()]


def _sweep_namespace() -> str:
    """Return the forecast subdirectory name used by feature sweep artifacts."""
    raw = str(os.environ.get("WQ_FEATURE_SWEEP_NAMESPACE", "")).strip()
    return raw or "feature_sweeps"


def _forecast_sweeps_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "forecasts" / _sweep_namespace()


def _feature_sweep_cache_path(dataset_dir: Path, input_aggregation=None) -> Path:
    """Where the tuned XGBoost hyperparameters for one window representation live.

    One cache per dataset was enough while there was a single XGBoost configuration.
    With three, they share a dataset but not an input: ``none`` presents 7381 flattened
    columns, ``stats:28`` presents 44. Hyperparameters tuned on one are not meaningful
    on the other -- a ``colsample_bytree`` of 0.5 samples 3690 columns in one case and
    22 in the other -- yet a single cache handed whichever representation tuned first to
    all of them. On pH that difference was worth 0.84 in R2. The representation is
    therefore part of the key; ``none`` keeps the original filename so existing caches
    are still found.
    """
    return (_forecast_sweeps_dir(dataset_dir)
            / f"xgb_cv_tuning_cache{aggregation_slug(input_aggregation)}.json")


def _load_feature_sweep_cache(dataset_dir: Path, input_aggregation=None) -> dict | None:
    cache_path = _feature_sweep_cache_path(dataset_dir, input_aggregation)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _apply_pre_rerun_model_policy(config: dict) -> None:
    """Apply the performance-focused model settings used by this sweep."""
    model_type = str(config.get("model_type", "")).lower()
    hyper_cfg = config.setdefault("hyperparameters", {})
    if model_type == "transformer":
        # Batch correlation regularization is noisy on these small MC datasets.
        hyper_cfg["corr_lambda"] = 0.0
    elif model_type in {"xgb_regressor", "xgb_classifier"}:
        cv_cfg = hyper_cfg.setdefault("cv_tuning", {})
        cv_cfg["enabled"] = True
        cv_cfg["selection_rule"] = "best"


def _feature_sweep_cache_is_compatible(dataset_dir: Path, input_aggregation=None) -> bool:
    """Require a cache produced by the current best-CV selection policy."""
    cache_payload = _load_feature_sweep_cache(dataset_dir, input_aggregation)
    if not isinstance(cache_payload, dict):
        return False
    return train_module._xgb_cv_cache_matches_config(
        cache_payload,
        {"selection_rule": "best"},
    )


def _ensure_feature_sweep_cache(plan: DatasetPlan, surrogate_model: str = "xgb") -> None:
    """Tune XGBoost once per window representation present in this dataset.

    Every XGBoost configuration in the final stage reads the cache for its own
    representation, so each one needs a cache; tuning only the surrogate's would leave
    the others to fall back on untuned defaults or, worse, on a cache fitted to a
    different input shape.
    """
    xgb_cfgs = []
    for cfg_path in plan.train_configs:
        try:
            cfg = train_module.load_config(str(cfg_path))
        except Exception:
            continue
        if str(cfg.get("model_type")) in {"xgb_regressor", "xgb_classifier"}:
            xgb_cfgs.append((cfg_path, str(cfg.get("data", {}).get("input_aggregation", "none"))))

    # The surrogate first, so an interrupted run still leaves the search able to start.
    try:
        preferred = _select_surrogate_config(plan.train_configs, surrogate_model)
    except Exception:
        preferred = None
    xgb_cfgs.sort(key=lambda item: item[0] != preferred)

    for cfg_path, aggregation in xgb_cfgs:
        _ensure_feature_sweep_cache_for(plan, cfg_path, aggregation)


def _ensure_feature_sweep_cache_for(plan: DatasetPlan, base_cfg: Path,
                                    aggregation: str) -> None:
    cache_path = _feature_sweep_cache_path(plan.dataset_dir, aggregation)
    if cache_path.exists() and _feature_sweep_cache_is_compatible(plan.dataset_dir, aggregation):
        return
    if base_cfg is None:
        return
    cfg = train_module.load_config(str(base_cfg))
    if str(cfg.get("model_type")) not in {"xgb_regressor", "xgb_classifier"}:
        return
    cfg = train_module.merge_with_defaults(cfg, cfg.get("model_type", "xgb_regressor"))
    _apply_pre_rerun_model_policy(cfg)
    hyper_cfg = cfg.setdefault("hyperparameters", {})
    cv_cfg = hyper_cfg.get("cv_tuning")
    if isinstance(cv_cfg, dict):
        cv_cfg["enabled"] = True
        cv_cfg["cache_path"] = str(cache_path)
    else:
        hyper_cfg["cv_tuning"] = {"enabled": True, "cache_path": str(cache_path)}

    if _pin_split_enabled():
        # These hyperparameters are used by every candidate and therefore by the
        # reported models, so they must be tuned on the same training data everything
        # else is. This config is built straight from the base config rather than
        # through _prepare_variant_config, so it would otherwise compute its own
        # split -- and any divergence would tune on segments the results are scored on.
        data_cfg = cfg["data"]
        pinned = _materialize_pinned_split(
            dataset_dir=plan.dataset_dir,
            sample_subdir=str(data_cfg.get("sample_subdir", "samples")),
        )
        split_cfg = cfg.setdefault("data_split", {})
        split_cfg["reuse_split"] = True
        split_cfg["split_source"] = str(pinned.resolve())
        split_cfg["allow_rebalance"] = False

    train_samples, _ = train_module.load_and_split_data(cfg)
    model_kind = "classifier" if str(cfg.get("model_type")) == "xgb_classifier" else "regressor"
    metric_key = "eval_metric" if model_kind == "classifier" else "metric"
    cast_y = (lambda v: int(round(v))) if model_kind == "classifier" else None
    print(f"[INFO] Feature sweep CV tuning for {base_cfg.stem.replace('config_', '')} "
          f"(input_aggregation={aggregation!r}, full features, selection_rule=best) "
          f"-> {cache_path.name}")
    train_module.run_xgb_cv_tuning_only(
        config=cfg,
        train_samples=train_samples,
        model_kind=model_kind,
        metric_key=metric_key,
        cast_y=cast_y,
        use_cache=False,
        write_cache=True,
    )


def _pin_all_sample_subdirs(plan: "DatasetPlan", surrogate_model: str) -> None:
    """Write every pinned split this dataset will need, before any parallel work starts.

    `_prepare_variant_config` creates a pinned split on first use, which is fine while
    runs are sequential. With `--parallel-evaluators` above one, several workers can
    find the same split missing at the same moment and write it concurrently, and an
    interleaved write would leave a split file that is neither run's. Doing it here,
    once, in the single main process removes the race rather than locking around it.

    Both subdirectories are covered because the families divide across them: GP and MLR
    train on `samples`, XGBoost and the Transformer on `mc_replicates`.
    """
    subdirs = []
    for cfg_path in plan.train_configs:
        try:
            data_cfg = train_module.load_config(str(cfg_path))["data"]
        except Exception:
            continue
        sub = str(data_cfg.get("sample_subdir", "samples"))
        if sub not in subdirs:
            subdirs.append(sub)

    for sub in subdirs:
        if not (Path(plan.dataset_dir) / sub).is_dir():
            continue
        _materialize_pinned_split(dataset_dir=plan.dataset_dir, sample_subdir=sub)


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
    source: str = "search"
    seeded_input_rank: int | None = None
    training_stop_reason: str | None = None
    # Rolling-origin cross-validation, when enabled. `cv_folds` is 0 for a
    # single-holdout evaluation, in which case the spread fields stay NaN and the
    # one-standard-error rule has nothing to work with.
    cv_folds: int = 0
    cv_r2_mean: float = float("nan")
    cv_r2_se: float = float("nan")
    cv_objective_se: float = float("nan")
    # Spread of the model's own predictions on the evaluation set, and whether that
    # spread has collapsed. A model predicting a constant carries no information about
    # any predictor, so every feature subset scores identically and the search has
    # nothing to choose between -- which is what happens on E. coli and Chromium,
    # where the fitted trees are all single leaves.
    pred_std: float = float("nan")
    degenerate: bool = False


def _load_training_stop_reason(forecast_dir: Path) -> str | None:
    """Load concise training stop reason from forecast-level summary artifact."""
    summary_path = Path(forecast_dir) / "training_stop_summary.json"
    if not summary_path.is_file():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        reason = payload.get("stop_reason_text")
        if reason is not None:
            reason = str(reason).strip()
            return reason or None
    except Exception as exc:
        print(f"[WARN] Could not read training stop summary at {summary_path}: {exc}")
    return None


_MODEL_ID_ALIASES: dict[str, str] = {
    # XGBoost class name and variants → canonical model_type
    "xgbregressor": "xgb_regressor",
    "xgb": "xgb_regressor",
    # GP variants → canonical model_type
    "gaussianprocessregressor": "gp_regressor",
    "exactgp": "gp_regressor",
    "gpregressor": "gp_regressor",
    # Linear
    "linearregression": "linear_regressor",
    "ridge": "linear_regressor",
    "elasticnet": "linear_regressor",
}


def _normalize_model_id(s: str) -> str:
    """Normalize a model identifier to a canonical lowercase form.

    Handles Python class names (e.g. 'XGBRegressor'), canonical model_type strings
    (e.g. 'xgb_regressor'), and short model_name prefixes (e.g. 'xgb_01').
    """
    key = s.lower().replace("_", "").replace(" ", "")
    return _MODEL_ID_ALIASES.get(key, key)


def _match_model_str(raw_value: str, cfg_model_type: str, cfg_model_name: str) -> bool:
    """Return True if *raw_value* (from the 'model' column of feature_sweep_final_metrics.csv)
    matches the config entry described by *cfg_model_type* and *cfg_model_name*.

    Handles three naming conventions that may appear in the CSV:
      - canonical model_type  ('xgb_regressor')
      - Python class name     ('XGBRegressor')
      - short model_name      ('xgb_01')
    """
    # Direct string equality (fast path)
    if raw_value == cfg_model_type or (cfg_model_name and raw_value == cfg_model_name):
        return True
    # Normalized comparison — maps class names to canonical model_type
    if _normalize_model_id(raw_value) == _normalize_model_id(cfg_model_type):
        return True
    return False


def _strip_fs_prefix(s: str) -> str:
    """Strip a leading sweep-subdir prefix from a forecast name string."""
    normalized = str(s).replace("\\", "/")
    prefixes = []
    for prefix in (_sweep_namespace(), "feature_sweeps", "Shapley_sweeps"):
        p = str(prefix).strip().strip("/")
        if p and p not in prefixes:
            prefixes.append(p)
    for p in prefixes:
        token = f"{p}/"
        if normalized.startswith(token):
            return normalized[len(token):]
    return normalized


def _find_matching_config(
    model_str: str,
    train_configs: list[Path],
) -> "tuple[str, str, Path] | None":
    """Return ``(resolved_model_type, base_forecast_name, config_path)`` for the first
    entry in *train_configs* whose model_type or model_name matches *model_str*.
    Returns ``None`` when no config matches.
    """
    for base_cfg in train_configs:
        try:
            _cfg = train_module.load_config(str(base_cfg))
            cfg_model_type = _cfg.get("model_type", "")
            cfg_model_name = _cfg.get("model_name", "")
            if _match_model_str(model_str, cfg_model_type, cfg_model_name):
                fn = _cfg.get("data", {}).get("forecast_name") or cfg_model_name or "unknown_01"
                return cfg_model_type, _strip_fs_prefix(str(fn)), base_cfg
        except Exception:
            continue
    return None


def _format_eta(start_time: float, eval_count: int, eval_budget: int) -> str:
    """Return a human-readable ETA string for the running search."""
    elapsed = time.time() - start_time
    avg = elapsed / eval_count if eval_count > 0 else 0
    eta_s = avg * (eval_budget - eval_count)
    return f"{int(eta_s // 60)}m {int(eta_s % 60)}s" if eta_s > 0 else "unknown"


def _derive_target_name(dataset_name: str, dataset_prefix: str) -> str:
    if dataset_name.startswith(dataset_prefix):
        return dataset_name[len(dataset_prefix):].lstrip("_")
    return dataset_name


def _feature_tag(features: tuple[str, ...]) -> str:
    joined = "||".join(features)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"f{len(features)}_{digest}"


def _model_sort_key(config_path: Path) -> tuple[int, str]:
    name = config_path.name.lower()
    if "gp" in name:
        return 0, name
    if "transformer" in name:
        return 1, name
    if "xgb" in name:
        return 2, name
    return 3, name


def _parse_row_counts(value: str | None, default_span: int) -> list[int]:
    if value is None or str(value).strip() == "":
        return [int(default_span)]

    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    rows: list[int] = []
    for part in parts:
        num = int(part)
        if num > 0:
            rows.append(num)
    rows = sorted(set(rows))
    return rows


def _parse_seed_features(raw: str) -> tuple[str, ...]:
    parts = [p.strip() for p in str(raw).split("|") if p.strip()]
    return tuple(parts)


def _data_root_key(dataset_dir: Path) -> str:
    """Return a short deterministic key for the parent dataset root."""
    root = Path(dataset_dir).resolve().parent
    return hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:8]


def _seed_subsets_filename(row_count: int, dataset_dir: Path | None = None) -> str:
    """Build seed-subset filename; includes root key when dataset_dir is provided."""
    if dataset_dir is None:
        return f"feature_seed_subsets_r{row_count:03d}.csv"
    return f"feature_seed_subsets_r{row_count:03d}_d{_data_root_key(dataset_dir)}.csv"


def _resolve_seed_subsets_csv_candidates(
    dataset_dir: Path,
    row_count: int,
    explicit_path: Path | None,
    from_shapley: bool,
) -> list[Path]:
    """Return candidate seed-subset CSV paths in priority order."""
    if from_shapley:
        shapley_dir = dataset_dir / "forecasts" / "Shapley_sweeps"
        return [
            shapley_dir / _seed_subsets_filename(row_count, dataset_dir=dataset_dir),
            shapley_dir / _seed_subsets_filename(row_count, dataset_dir=None),
        ]

    if explicit_path is None:
        return []

    if explicit_path.is_dir():
        return [
            explicit_path / _seed_subsets_filename(row_count, dataset_dir=dataset_dir),
            explicit_path / _seed_subsets_filename(row_count, dataset_dir=None),
        ]

    return [explicit_path]


def _resolve_seed_subsets_csv_path(
    dataset_dir: Path,
    row_count: int,
    explicit_path: Path | None,
    from_shapley: bool,
) -> Path | None:
    candidates = _resolve_seed_subsets_csv_candidates(
        dataset_dir=dataset_dir,
        row_count=row_count,
        explicit_path=explicit_path,
        from_shapley=from_shapley,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _load_seed_subsets(
    dataset_dir: Path,
    row_count: int,
    explicit_path: Path | None,
    from_shapley: bool,
    max_seed_subsets: int,
) -> list[tuple[str, ...]]:
    seed_csv = _resolve_seed_subsets_csv_path(
        dataset_dir=dataset_dir,
        row_count=row_count,
        explicit_path=explicit_path,
        from_shapley=from_shapley,
    )
    if seed_csv is None:
        return []

    if not seed_csv.exists():
        if from_shapley or explicit_path is not None:
            print(f"[WARN] Seed subsets not found at {seed_csv}; proceeding with unseeded search.")
        return []

    try:
        df = pd.read_csv(seed_csv)
    except Exception as exc:
        print(f"[WARN] Failed to load seed subsets from {seed_csv}: {exc}")
        return []

    if "features" not in df.columns:
        print(f"[WARN] Seed subsets CSV missing 'features' column: {seed_csv}")
        print("[WARN] Proceeding with unseeded search.")
        return []

    seen: set[tuple[str, ...]] = set()
    out: list[tuple[str, ...]] = []
    for raw in df["features"].tolist():
        feats = _parse_seed_features(str(raw))
        if not feats or feats in seen:
            continue
        seen.add(feats)
        out.append(feats)
        if max_seed_subsets > 0 and len(out) >= max_seed_subsets:
            break

    if out:
        print(f"[INFO] Loaded {len(out)} seed subset(s) from: {seed_csv}")
        _report_shapley_rank_separation(seed_csv, row_count)
    else:
        print(f"[WARN] No valid seed subsets parsed from {seed_csv}; proceeding with unseeded search.")
    return out


def _report_shapley_rank_separation(seed_csv: Path, row_count: int) -> float:
    """Report how much of the Shapley ranking the attribution actually resolved.

    The seed subsets are nested top-k prefixes of a Shapley ranking, so the ranking's
    *order* is what the beam search inherits. If adjacent ranks have overlapping 95%
    confidence intervals then that order is noise at those positions, and seeding from
    it starts the search somewhere arbitrary while looking principled. This does not
    change what is loaded -- it states what the seeds are worth, so the run can be
    judged rather than assumed.

    Args:
        seed_csv: The loaded seed-subset CSV; the scores file sits beside it.
        row_count: Row count used to locate `feature_shapley_scores_r###.csv`.

    Returns:
        Fraction of adjacent rank pairs whose intervals are disjoint. NaN when the
        scores file is absent or carries no intervals.
    """
    scores_csv = Path(seed_csv).parent / f"feature_shapley_scores_r{int(row_count):03d}.csv"
    if not scores_csv.is_file():
        print(f"[WARN] No Shapley scores beside the seeds ({scores_csv.name}); "
              "rank separation unknown, so seed order is unverified.")
        return float("nan")
    try:
        df = pd.read_csv(scores_csv)
    except Exception as exc:
        print(f"[WARN] Could not read {scores_csv}: {exc}")
        return float("nan")

    needed = {"ci95_low", "ci95_high"}
    if not needed.issubset(df.columns):
        print(f"[WARN] {scores_csv.name} carries no confidence intervals; "
              "rank separation cannot be checked.")
        return float("nan")

    sort_col = "shapley_rank" if "shapley_rank" in df.columns else "shapley_value_est"
    ascending = sort_col == "shapley_rank"
    df = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    lo = pd.to_numeric(df["ci95_low"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(df["ci95_high"], errors="coerce").to_numpy(dtype=float)
    if len(lo) < 2:
        return float("nan")

    # Adjacent ranks are separated when the higher-ranked feature's lower bound clears
    # the next feature's upper bound.
    pairs = len(lo) - 1
    disjoint = int(np.sum(np.isfinite(lo[:-1]) & np.isfinite(hi[1:]) & (lo[:-1] > hi[1:])))
    frac = disjoint / pairs if pairs else float("nan")
    verdict = "usable" if frac >= 0.5 else "weak -- treat the seed order as arbitrary"
    print(
        f"[SEED] Shapley rank separation: {disjoint}/{pairs} adjacent rank pairs have "
        f"disjoint 95% intervals ({frac:.0%}); {verdict}."
    )
    return frac


def discover_mc_dataset_plans(
    data_root: Path,
    dataset_prefix: str,
    config_pattern: str,
    limit_datasets: int,
    include_regular: bool,
    include_res: bool,
) -> list[DatasetPlan]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    # Debug print: resolved data_root
    print(f"[DEBUG] discover_mc_dataset_plans: data_root = {data_root}")
    # Debug print: all raw subdirectory names
    all_subdirs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    print("[DEBUG] All subdirectories in data_root:", [p.name for p in all_subdirs])

    def _dataset_allowed(name: str) -> bool:
        if not name.startswith(dataset_prefix):
            return False
        is_res = name.endswith("_res")
        return (is_res and include_res) or ((not is_res) and include_regular)

    dataset_dirs = [path for path in all_subdirs if _dataset_allowed(path.name)]

    plans: list[DatasetPlan] = []
    for dataset_dir in dataset_dirs:
        raw_matches = sorted(dataset_dir.glob(config_pattern))
        train_configs = []
        for path in raw_matches:
            if path.suffix.lower() not in SUPPORTED_CONFIG_SUFFIXES:
                continue
            try:
                cfg = train_module.load_config(str(path))
                if "model_name" not in cfg and "model_type" not in cfg:
                    print(f"[WARN] Skipping config without model_name/model_type: {path}")
                    continue
                model_type = str(cfg.get("model_type", "")).strip()
                if model_type and model_type not in _TRAINABLE_MODEL_TYPES:
                    print(
                        f"[WARN] Skipping {path.name}: model_type '{model_type}' is not "
                        "implemented by this sweep and would fail after selection."
                    )
                    continue
                train_configs.append(path)
            except Exception as e:
                print(f"[WARN] Could not load config {path}: {e}")
        if not train_configs:
            print(f"[WARN] Skipping dataset (no matching configs found): {dataset_dir.name}")
            continue
        train_configs.sort(key=_model_sort_key)
        plans.append(DatasetPlan(dataset_dir=dataset_dir, train_configs=train_configs))

    if limit_datasets > 0:
        plans = plans[:limit_datasets]
    return plans


def _resolve_dataset_inclusion(args: argparse.Namespace) -> tuple[bool, bool]:
    include_regular = True
    include_res = True

    if args.regular_only and args.res_only:
        raise ValueError("Cannot use both --regular-only and --res-only.")

    if args.regular_only:
        include_regular, include_res = True, False
    elif args.res_only:
        include_regular, include_res = False, True
    else:
        if args.include_regular or args.include_res:
            include_regular = bool(args.include_regular)
            include_res = bool(args.include_res)

    if not include_regular and not include_res:
        raise ValueError("At least one dataset group must be included.")

    return include_regular, include_res


def _variant_forecast_name(base_forecast_name: str, row_count: int, feature_tag: str) -> str:
    base_name = _strip_fs_prefix(str(base_forecast_name).replace("\\", "/"))
    return f"{_sweep_namespace()}/{base_name}_r{row_count:03d}_{feature_tag}"


def _prepare_variant_config(
    base_config_path: Path,
    row_count: int,
    features: tuple[str, ...],
    feature_tag: str,
    tmp_dir: Path,
    forced_data_dir: Path | None = None,
    cv_fold_dir: Path | None = None,
    cv_fold_index: int | None = None,
    pin_split: bool | None = None,
) -> Path:
    cfg = train_module.load_config(str(base_config_path))
    cfg_copy = copy.deepcopy(cfg)
    _apply_pre_rerun_model_policy(cfg_copy)

    if "data" not in cfg_copy:
        raise ValueError(f"Missing data section in {base_config_path}")

    data_cfg = cfg_copy["data"]
    source_config_dir = Path(cfg.get("__config_dir", base_config_path.parent))
    required_data = ["input_row_1", "input_row_2", "forecast_name", "data_dir"]
    for field in required_data:
        if field not in data_cfg:
            raise ValueError(f"Missing data.{field} in {base_config_path}")

    base_stop = int(data_cfg["input_row_2"])
    base_start = int(data_cfg["input_row_1"])
    base_span = base_stop - base_start
    if base_span <= 0:
        raise ValueError(f"Invalid input row span in {base_config_path}: {base_start}..{base_stop}")
    if row_count > base_span:
        raise ValueError(f"row_count={row_count} exceeds base span={base_span} for {base_config_path.name}")

    data_cfg["input_columns"] = list(features)
    data_cfg["input_row_1"] = int(base_stop - row_count)
    data_cfg["input_row_2"] = int(base_stop)

    # Resolve data_dir from the discovered dataset path when provided.
    # This keeps read/write locations consistent with --data-root.
    if forced_data_dir is not None:
        resolved_data_dir = Path(forced_data_dir).resolve()
    else:
        resolved_data_dir = train_module._resolve_path_from_config(data_cfg["data_dir"], source_config_dir)
    data_cfg["data_dir"] = str(resolved_data_dir)
    # Ensure forecast_name is always relative to the correct data_dir
    data_cfg["forecast_name"] = _variant_forecast_name(str(data_cfg["forecast_name"]), row_count, feature_tag)
    forecast_name_rel = _strip_fs_prefix(str(data_cfg["forecast_name"]))
    # Store the original forecast directory for downstream evaluation
    forecast_dir = Path(resolved_data_dir) / "forecasts" / _sweep_namespace() / forecast_name_rel
    data_cfg["forecast_dir"] = str(forecast_dir.resolve())

    # Ensure model_name is set
    if "model_name" not in cfg_copy or not cfg_copy["model_name"]:
        # Try to inherit from base config, fallback to model_type
        cfg_copy["model_name"] = cfg.get("model_name", cfg.get("model_type", "unknown"))

    if "evaluation" not in cfg_copy:
        cfg_copy["evaluation"] = {}
    uses_uncertainty = _candidate_uses_uncertainty_distributions(tuple(features))
    cfg_copy["evaluation"]["run_baselines"] = True
    # If no uncertainty-enabled predictors are present, evaluating MC replicates
    # is redundant because replicate predictions are identical.
    cfg_copy["evaluation"]["collapse_mc_replicates_for_eval"] = not uses_uncertainty
    cfg_copy["evaluation"]["include_mc_stats_in_predictions"] = uses_uncertainty

    # Sweep/pipeline variants must reserve enough independent non-replicate test samples.
    split_cfg = cfg_copy.setdefault("data_split", {})
    split_cfg.setdefault("split_type", "temporal")
    split_cfg.setdefault("test_size", 0.3)
    split_cfg["min_test_independent"] = int(FINAL_TOPK_MIN_TEST_SAMPLES)

    variant_suffix = ""
    pin = _pin_split_enabled() if pin_split is None else bool(pin_split)
    if cv_fold_dir is None and pin:
        # One split per target, reused by every run. Without this each run recomputes
        # its own boundary from its own valid-sample count, so families are not
        # compared on the same test set and the common-set intersection silently
        # discards the segments they disagree about.
        try:
            pinned = _materialize_pinned_split(
                dataset_dir=Path(resolved_data_dir),
                sample_subdir=str(data_cfg.get("sample_subdir", "samples")),
            )
        except Exception as exc:
            # Reverting to a per-run split here would reintroduce the drift this
            # exists to remove, and would do it quietly, one run at a time.
            raise RuntimeError(
                f"Could not pin the split for {Path(resolved_data_dir).name}: {exc}. "
                "Pass --no-pin-split to accept per-run boundaries instead."
            ) from exc
        split_cfg["reuse_split"] = True
        split_cfg["split_source"] = str(pinned.resolve())
        split_cfg["allow_rebalance"] = False

    if cv_fold_dir is not None:
        # Pin this evaluation to one rolling-origin fold. The fold list is fixed for
        # the whole search, so `min_test_independent` must not be allowed to rebalance
        # it -- a rebalance would move segments between train and test and quietly give
        # this candidate a different evaluation set from every other candidate.
        split_cfg["reuse_split"] = True
        split_cfg["split_source"] = str(Path(cv_fold_dir).resolve())
        split_cfg["min_test_independent"] = 1
        # The run directory keeps the candidate's own name, with no per-fold suffix.
        # Folds of one candidate run sequentially and each fold's predictions are read
        # before the next overwrites them, so one directory suffices -- and Windows
        # applies a 260-character path limit that these already-long names leave little
        # room under. Only the throwaway config file carries the fold number.
        variant_suffix = f"_cv{int(cv_fold_index or 0):02d}"

    cfg_copy.pop("__config_dir", None)

    variant_name = f"{base_config_path.stem}_r{row_count:03d}_{feature_tag}{variant_suffix}{base_config_path.suffix}"
    variant_path = tmp_dir / variant_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with open(variant_path, "w", encoding="utf-8") as f:
        if variant_path.suffix.lower() in {".yml", ".yaml"}:
            yaml.safe_dump(cfg_copy, f, sort_keys=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported config suffix for variant writing: {variant_path.suffix}")

    return variant_path


def _train_single_config(
    config_path: Path,
    dataset_dir: Path,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
) -> Path:
    config = train_module.load_config(str(config_path))

    required_fields = ["model_type", "model_name", "data"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field '{field}' in {config_path}")

    model_type = config["model_type"]
    config = train_module.merge_with_defaults(config, model_type)
    if disable_training_plots:
        config["save_training_plots"] = False

    if model_type in {"xgb_regressor", "xgb_classifier"}:
        _agg = config.get("data", {}).get("input_aggregation", "none")
        cache_payload = _load_feature_sweep_cache(dataset_dir, _agg)
        if cache_payload is not None:
            tuned = cache_payload.get("tuned_hyperparameters") or cache_payload.get("best_params") or {}
            if isinstance(tuned, dict) and tuned:
                hyper_cfg = config.setdefault("hyperparameters", {})
                hyper_cfg.update(tuned)
                cv_cfg = hyper_cfg.get("cv_tuning")
                if isinstance(cv_cfg, dict):
                    cv_cfg["enabled"] = False
                    cv_cfg["cache_path"] = str(_feature_sweep_cache_path(dataset_dir, _agg))
                else:
                    hyper_cfg["cv_tuning"] = {
                        "enabled": False,
                        "cache_path": str(_feature_sweep_cache_path(dataset_dir, _agg)),
                    }

    device = torch.device(config["device"])
    matplotlib.use(config["matplotlib_backend"])

    if suppress_training_logs:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train_samples, test_samples = train_module.load_and_split_data(config)
    else:
        train_samples, test_samples = train_module.load_and_split_data(config)
        print(f"    [TRAIN] {config_path.name}: train={len(train_samples)} test={len(test_samples)}")

    def _run_train():
        if model_type == "transformer":
            train_module.train_transformer_model(config, train_samples, test_samples)
        elif model_type == "gp_regressor":
            train_module.train_gp_regressor_model(config, train_samples, test_samples)
        elif model_type == "xgb_regressor":
            if train_module._xgb_cv_tuning_enabled(config):
                train_module.train_xgb_regressor_model_cv_tuned(config, train_samples, test_samples)
            else:
                train_module.train_xgb_regressor_model(config, train_samples, test_samples)
        elif model_type == "xgb_classifier":
            if train_module._xgb_cv_tuning_enabled(config):
                train_module.train_xgb_classifier_model_cv_tuned(config, train_samples, test_samples)
            else:
                train_module.train_xgb_classifier_model(config, train_samples, test_samples)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    if suppress_training_logs:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _run_train()
    else:
        _run_train()

    data_cfg = config["data"]
    forecast_name = data_cfg["forecast_name"]
    forecast_file_name = Path(str(forecast_name)).name
    forecast_name_rel = _strip_fs_prefix(str(forecast_name))
    forecast_dir = _forecast_sweeps_dir(dataset_dir) / Path(forecast_name_rel)
    return (forecast_dir / f"config_evaluate_{forecast_file_name}.yml").resolve()


def _set_eval_overrides(eval_config_path: Path, run_baselines: bool) -> None:
    # Apply explicit per-phase baseline policy (search vs final).
    with open(eval_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "evaluation" not in cfg:
        cfg["evaluation"] = {}
    cfg["evaluation"]["run_baselines"] = bool(run_baselines)

    with open(eval_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def _copy_eval_directory(source_dir: Path, dest_dir: Path) -> None:
    """Copy an evaluation results directory, renaming internal config files to match."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)
    # Rename config_evaluate_*.yml to match the new directory name.
    old_cfg = dest_dir / f"config_evaluate_{source_dir.name}.yml"
    new_cfg = dest_dir / f"config_evaluate_{dest_dir.name}.yml"
    if old_cfg.exists() and old_cfg != new_cfg:
        old_cfg.rename(new_cfg)


def _variant_key(base_config_path) -> str:
    """Registry key identifying one model configuration, not its family.

    The evaluation-reuse cache was keyed on ``model_type``, which is ``gp_regressor``
    for all four Gaussian process configurations. The first of them to be evaluated
    registered under that key and the other three matched it, so their directories were
    filled by copying its results and they were never trained: four rows reporting one
    model's score under four names. Keying on the configuration stem keeps the reuse --
    which is sound when the model really is the same -- while making a variant distinct.
    """
    return Path(base_config_path).stem.replace("config_", "")


_EMPTY_VARIANT_FIELDS = {
    "variant": "",
    "input_aggregation": "",
    "kernel": "",
    "effective_kernel": "",
    "effective_ard": "",
}


def _degeneracy_fields(run_dir) -> dict:
    """``pred_std`` and ``degenerate`` for a finished run, read from its predictions.

    Recorded on every row so the results table can state that a model predicts a
    constant, rather than leaving it to be inferred from a suspiciously round R2. For
    a target where every model is degenerate, that is the finding: no predictor set is
    distinguishable because none of them changes the prediction.
    """
    if run_dir is None:
        return {"pred_std": float("nan"), "degenerate": ""}
    y, pr = _pooled_predictions(Path(run_dir))
    if pr.size < 2:
        return {"pred_std": float("nan"), "degenerate": ""}
    std, deg = _prediction_spread(y, pr)
    return {"pred_std": std, "degenerate": bool(deg)}


def _model_variant_fields(base_config_path, run_dir=None) -> dict:
    """The structural identity of a fitted run, for the metrics table.

    ``model`` alone is not an identity. Four Gaussian process configurations share the
    model type ``gp_regressor`` while differing in how the input window is reduced and
    which kernel is used, and on one target they span R2 from -4.32 to +0.56. A table
    recording only ``gp_regressor`` cannot say which model produced a result, and
    anything downstream that resolves a config from it has to guess -- which is how a
    run selected as gp_04 was later retrained as gp_01.

    ``variant`` comes from the configuration that was used, so it is always present.
    The remaining fields are read from the run's own ``model_config.json`` where one
    exists, so they describe what was actually fitted rather than what was requested --
    ``effective_ard`` in particular differs from the requested ``ard`` whenever the
    dimensionality guard fires.

    Args:
        base_config_path: The training config this run was built from.
        run_dir: The run's output directory, when it got far enough to have one.

    Returns:
        ``variant``, ``input_aggregation``, ``kernel``, ``effective_kernel`` and
        ``effective_ard``; empty strings where a field does not apply to the family.
    """
    out = {
        "variant": Path(base_config_path).stem.replace("config_", "") if base_config_path else "",
        "input_aggregation": "",
        "kernel": "",
        "effective_kernel": "",
        "effective_ard": "",
    }
    if run_dir is None:
        return out
    cfg_path = Path(run_dir) / "model_config.json"
    if not cfg_path.is_file():
        return out
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return out
    for key in ("input_aggregation", "kernel", "effective_kernel", "effective_ard"):
        if key in payload:
            out[key] = payload[key]
    return out


def _mlr_artifact_dir(output_dir: Path, subset_label: str,
                      model_prefix: str = "mlr") -> Path:
    token = str(subset_label).strip().lower()
    if not token:
        token = "main"
    return output_dir / f"{model_prefix}_{token}"


def _append_mlr_baseline_outputs(
    summary_rows: list[dict],
    pred_entries: list[dict],
    *,
    ref_cfg: dict | None,
    ref_cfg_path: Path | None,
    ref_data_cfg: dict | None,
    data_dir: str,
    sample_subdir: str,
    output_columns: list[str],
    output_rows,
    forecast_name: str,
    test_samples,
    test_split_files: list[str],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str], list[list[str]]]:
    if ref_cfg is None or ref_cfg_path is None or ref_data_cfg is None:
        return [], [], []

    try:
        eval_cfg_merged = eval_module.merge_eval_config(ref_cfg)
        if eval_cfg_merged.get("historic_path"):
            eval_cfg_merged["historic_path"] = str(
                eval_module._resolve_path_from_config(eval_cfg_merged["historic_path"], ref_cfg_path.parent)
            )
        historic = eval_cfg_merged.get("historic_path")
        if not historic:
            return [], [], []

        baseline_output_rows = eval_module._baseline_output_rows_start(ref_data_cfg.get("output_rows", output_rows))
        secondary, baseline_window_hours = eval_module.load_secondary(
            output_columns,
            int(eval_cfg_merged.get("window_hours", 340)),
        )
        deduped_split_files = eval_module._dedupe_split_files_by_base_sample(test_split_files)

        preds_naive, targets_naive = eval_module.evaluate_naive(
            test_samples,
            historic,
            output_columns,
            data_dir,
            output_rows=baseline_output_rows,
            gap_hours=int(eval_cfg_merged.get("gap_hours", 5)),
            sample_subdir=sample_subdir,
        )
        preds_seasonal, targets_seasonal = eval_module.evaluate_seasonal(
            test_samples,
            historic,
            output_columns,
            data_dir,
            output_rows=baseline_output_rows,
            diurnal_window=int(eval_cfg_merged.get("diurnal_window", 2)),
            secondary=secondary,
            sample_subdir=sample_subdir,
        )
        preds_linear, targets_linear = eval_module.evaluate_linear(
            data_dir,
            forecast_name,
            test_samples,
            historic,
            output_columns,
            output_rows=baseline_output_rows,
            window_hours=int(baseline_window_hours),
            gap_hours=int(eval_cfg_merged.get("gap_hours", 0)),
            debug_plot=False,
            examples=0,
            sample_subdir=sample_subdir,
        )
        preds_naive, _ = eval_module._clip_to_target_support(preds_naive, "Naive baseline")
        preds_seasonal, _ = eval_module._clip_to_target_support(preds_seasonal, "Seasonal baseline")
        preds_linear, _ = eval_module._clip_to_target_support(preds_linear, "Linear baseline")
    except Exception as exc:
        print(f"[WARN] MLR baseline evaluation failed: {exc}")
        return [], [], []

    baseline_pairs = [
        (preds_naive, targets_naive),
        (preds_seasonal, targets_seasonal),
        (preds_linear, targets_linear),
    ]
    baseline_labels = ["Naive", "Seasonal", "Linear"]
    baseline_split_files = [deduped_split_files] * len(baseline_pairs)

    for (preds, targets), label in zip(baseline_pairs, baseline_labels):
        summary_rows.append(
            eval_module._compute_regression_summary(
                label,
                preds,
                targets,
                len(test_samples),
                metadata={"kind": "baseline", "gp_uncertainty_mode": "not_gp"},
                split_files=test_split_files,
            )
        )
        pred_entries.append(
            {
                "kind": "test",
                "label": label,
                "preds": preds,
                "targets": targets,
                "split_files": test_split_files,
                "include_mc_stats": False,
            }
        )

    return baseline_pairs, baseline_labels, baseline_split_files


def _evaluate_mlr_with_rebalance(
    train_samples,
    test_samples,
    feature_names,
    selection_config,
    aggregation_mode,
    min_test_independent,
    model_name="MLR",
    use_spearman_prefilter=True,
):
    """Evaluate MLR, exclude unpredictable test samples, rebalance, re-fit.

    1. ``evaluate_mlr`` on initial train/test.
    2. ``filter_predictable`` → remove NaN-prediction test samples.
    3. If samples were excluded, re-enforce ``min_test_independent`` by
       calling ``_rebalance_to_min_test_independent`` (may move samples
       from train → test).  Then re-fit on adjusted sets.
    4. Repeat up to ``_MAX_REBALANCE_ITERS`` times to handle newly-moved
       samples that are themselves unpredictable.

    Step 3 is skipped where the split is pinned, because moving a training segment
    into the test set would both break the test set shared with every other model and
    evaluate this one on data it was fitted on. The exclusions from step 2 still
    apply and are reported: MLR is then scored on a subset of the shared test set,
    which is visible in its sample count rather than papered over.

    Returns
    -------
    preds, targets, train_samples, test_samples, meta, total_excluded
    """
    from utils.mlr import evaluate_mlr as _eval_mlr
    from utils.mlr import filter_predictable as _filter_pred
    from utils.training import _rebalance_to_min_test_independent, _base_sample_id

    _MAX_REBALANCE_ITERS = 3
    total_excluded = 0

    for iteration in range(_MAX_REBALANCE_ITERS + 1):
        preds, targets, meta = _eval_mlr(
            train_samples, test_samples,
            feature_names=feature_names,
            selection_config=selection_config,
            aggregation_mode=aggregation_mode,
            use_spearman_prefilter=use_spearman_prefilter,
        )
        preds, targets, test_samples, n_excl = _filter_pred(preds, targets, test_samples)
        total_excluded += n_excl

        if n_excl == 0:
            break

        print(
            f"[MLR] {n_excl} test sample(s) excluded (unpredictable after "
            f"{aggregation_mode} aggregation) for {model_name}, iteration {iteration + 1}."
        )

        if _pin_split_enabled():
            print(
                f"[MLR] Split is pinned: not rebalancing {model_name}. It is scored on "
                f"{len(test_samples)} of the shared test segments; the {total_excluded} "
                "excluded one(s) are reported as unscored rather than replaced with "
                "segments the model trained on."
            )
            break

        # Re-enforce minimum test guarantee.
        train_names = [str(s[2]) for s in train_samples]
        test_names = [str(s[2]) for s in test_samples]
        skip_eval, status, new_train_names, new_test_names = _rebalance_to_min_test_independent(
            train_names, test_names, min_test_independent,
        )

        if skip_eval:
            from utils.training import SampleComplianceError
            raise SampleComplianceError(
                reason="insufficient_total_independent_after_mlr_filter",
                message=(
                    f"{model_name}: after excluding {total_excluded} unpredictable sample(s), "
                    f"cannot satisfy min_test_independent={min_test_independent}."
                ),
                context={
                    "model_name": model_name,
                    "aggregation_mode": aggregation_mode,
                    "total_excluded": total_excluded,
                    "target_min_independent": min_test_independent,
                },
            )

        if status == "already_sufficient":
            break

        # Rebalance moved samples from train → test.  Rebuild sample lists.
        all_samples = {str(s[2]): s for s in list(train_samples) + list(test_samples)}
        train_samples = [all_samples[n] for n in new_train_names if n in all_samples]
        test_samples = [all_samples[n] for n in new_test_names if n in all_samples]

        print(
            f"[MLR] Rebalanced split for {model_name}: "
            f"train={len(train_samples)}, test={len(test_samples)}."
        )
        # Loop: re-fit on adjusted sets (new test samples may also be unpredictable).
    else:
        # Exhausted iterations — check if we still meet the minimum.
        test_indep = len(set(_base_sample_id(str(s[2])) for s in test_samples))
        if test_indep < min_test_independent:
            from utils.training import SampleComplianceError
            raise SampleComplianceError(
                reason="rebalance_iterations_exhausted",
                message=(
                    f"{model_name}: after {_MAX_REBALANCE_ITERS} rebalance iterations, "
                    f"test_independent={test_indep} < min={min_test_independent}."
                ),
                context={
                    "model_name": model_name,
                    "aggregation_mode": aggregation_mode,
                    "total_excluded": total_excluded,
                    "target_min_independent": min_test_independent,
                    "test_independent": test_indep,
                },
            )

    return preds, targets, train_samples, test_samples, meta, total_excluded


def _write_mlr_artifacts(
    *,
    output_dir: Path,
    dataset_dir: Path,
    subset_label: str,
    data_dir: str,
    sample_subdir: str,
    input_columns: list[str],
    output_columns: list[str],
    input_row_1: int,
    input_row_2: int,
    output_rows,
    input_aggregation: str,
    train_samples,
    test_samples,
    preds,
    targets,
    per_target_meta: list[dict],
    split_source_dir: Path,
    ref_cfg: dict | None = None,
    ref_cfg_path: Path | None = None,
    ref_data_cfg: dict | None = None,
    model_config_extra: dict | None = None,
    model_prefix: str = "mlr",
) -> Path:
    mlr_dir = _mlr_artifact_dir(output_dir, subset_label, model_prefix=model_prefix)
    mlr_dir.mkdir(parents=True, exist_ok=True)
    preds, _ = eval_module._clip_to_target_support(preds, f"{model_prefix.upper().replace('_', '-')} model")

    model_config = {
        "model_type": model_prefix,
        "subset_label": str(subset_label),
        "input_columns": input_columns,
        "output_columns": output_columns,
        "input_row_1": input_row_1,
        "input_row_2": input_row_2,
        "output_rows": output_rows,
        "input_aggregation": input_aggregation,
        "n_train_samples": len(train_samples),
        "n_test_samples": len(test_samples),
        "feature_selection": {
            "method": "Spearman + MI + L1/Lasso + VIF",
            "spearman_p_threshold": 0.05,
            "spearman_rho_threshold": 0.20,
            "mi_quantile": 0.25,
            "deduplicate_threshold": 0.9999,
            "vif_threshold": 10.0,
            "use_mutual_info": True,
            "use_lasso": True,
        },
        "per_target_meta": [],
    }
    if model_config_extra:
        feature_selection_extra = model_config_extra.get("feature_selection")
        if isinstance(feature_selection_extra, dict):
            model_config["feature_selection"].update(feature_selection_extra)
        model_config.update(model_config_extra)

    for idx, meta in enumerate(per_target_meta):
        model_config["per_target_meta"].append(
            {
                "target_index": idx,
                "target_name": output_columns[idx] if idx < len(output_columns) else f"target_{idx}",
                "selected_features": meta.get("selected_features", []),
                "n_selected": meta.get("n_selected", 0),
                "coefficients": meta.get("coefficients", []),
                "intercept": meta.get("intercept", None),
                "n_train_valid": meta.get("n_train", 0),
                "spearman_kept_columns": meta.get("spearman_kept_columns", []),
            }
        )

    with open(mlr_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, default=str)

    # Write config_evaluate_*.yml so _find_best_variant_eval_config can discover
    # this MLR artifact dir and _collect_prediction_payload can load its data.
    _eval_cfg_name = f"config_evaluate_{mlr_dir.name}.yml"
    _eval_cfg = {
        "model_type": model_prefix,
        "model_name": model_prefix,
        "data": {
            "data_dir": str(os.path.relpath(data_dir, mlr_dir)),
            "sample_subdir": sample_subdir,
            "forecast_name": str(mlr_dir.relative_to(dataset_dir / "forecasts"))
            if dataset_dir is not None
            else mlr_dir.name,
            "input_columns": input_columns,
            "output_columns": output_columns,
            "input_row_1": input_row_1,
            "input_row_2": input_row_2,
            "output_rows": output_rows,
            "input_aggregation": input_aggregation,
        },
        "data_split": {
            "random_state": 42,
            "fault_tolerant": True,
        },
    }
    # Inherit evaluation section (historic_path, window_hours, etc.) from ref config
    if ref_cfg is not None:
        if "evaluation" in ref_cfg:
            _eval_cfg["evaluation"] = dict(ref_cfg["evaluation"])
        if "data_split" in ref_cfg:
            for _ds_key in ("random_state", "test_size", "split_type"):
                if _ds_key in ref_cfg["data_split"]:
                    _eval_cfg["data_split"][_ds_key] = ref_cfg["data_split"][_ds_key]
    with open(mlr_dir / _eval_cfg_name, "w", encoding="utf-8") as f:
        yaml.dump(_eval_cfg, f, default_flow_style=False, sort_keys=False)

    equation_lines = []
    for target_meta in model_config["per_target_meta"]:
        equation_lines.append(f"Target: {target_meta['target_name']}")
        equation_lines.append(f"  n_selected = {target_meta['n_selected']}")
        equation_lines.append(f"  n_train    = {target_meta['n_train_valid']}")
        selected_features = target_meta["selected_features"]
        coefficients = target_meta["coefficients"]
        intercept = target_meta["intercept"]
        if intercept is not None and selected_features and coefficients:
            terms = [f"{intercept:.6g}"]
            for feature_name, coef in zip(selected_features, coefficients):
                sign = "+" if coef >= 0 else "-"
                terms.append(f"{sign} {abs(coef):.6g} * {feature_name}")
            equation_lines.append(f"  y = {terms[0]}")
            for term in terms[1:]:
                equation_lines.append(f"      {term}")
        else:
            equation_lines.append("  (no features selected)")
        equation_lines.append("")
    with open(mlr_dir / "mlr_equation.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(equation_lines))

    # Write split files from actual sample lists (which may have been rebalanced
    # after excluding unpredictable samples).
    train_split_files = [str(s[2]) for s in train_samples]
    test_split_files = [str(s[2]) for s in test_samples]
    (mlr_dir / "train_files.txt").write_text(
        "\n".join(train_split_files) + ("\n" if train_split_files else ""), encoding="utf-8",
    )
    (mlr_dir / "test_files.txt").write_text(
        "\n".join(test_split_files) + ("\n" if test_split_files else ""), encoding="utf-8",
    )

    summary_rows = []
    pred_entries = []
    baseline_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    baseline_labels: list[str] = []
    baseline_split_files: list[list[str]] = []

    _model_label = model_prefix.upper().replace("_", "-")
    summary_row = eval_module._compute_regression_summary(
        f"{_model_label} (test)",
        preds,
        targets,
        len(test_samples),
        metadata={"kind": "test", "gp_uncertainty_mode": "not_gp"},
        split_files=test_split_files,
    )
    summary_row["n_train_samples"] = len(train_samples)
    summary_row["n_test_samples"] = len(test_samples)
    summary_row["input_dim"] = len(input_columns)
    summary_row["target_dim"] = len(output_columns)
    summary_row["data_dir"] = data_dir
    summary_rows.append(summary_row)
    pred_entries.append(
        {
            "kind": "test",
            "label": _model_label,
            "preds": preds,
            "targets": targets,
            "split_files": test_split_files,
            "include_mc_stats": False,
        }
    )

    baseline_pairs, baseline_labels, baseline_split_files = _append_mlr_baseline_outputs(
        summary_rows,
        pred_entries,
        ref_cfg=ref_cfg,
        ref_cfg_path=ref_cfg_path,
        ref_data_cfg=ref_data_cfg,
        data_dir=data_dir,
        sample_subdir=sample_subdir,
        output_columns=output_columns,
        output_rows=output_rows,
        forecast_name=str(ref_data_cfg.get("forecast_name", "")) if ref_data_cfg else "",
        test_samples=test_samples,
        test_split_files=test_split_files,
    )

    eval_module._write_summary_csv(summary_rows, mlr_dir / "evaluation_summary.csv")
    prediction_rows, prediction_cols = eval_module._build_predictions_table(
        pred_entries,
        gp_uncertainty_mode="not_gp",
        include_mc_output_columns=False,
    )
    eval_module._write_predictions_csv(prediction_rows, mlr_dir / "predictions.csv", prediction_cols)

    try:
        forecast_name_rel = str(mlr_dir.relative_to(dataset_dir / "forecasts"))
    except ValueError:
        forecast_name_rel = mlr_dir.name

    matplotlib.use("Agg")
    plot_pairs = [(preds, targets)] + baseline_pairs
    plot_labels = ["MLR"] + baseline_labels
    plot_split_files = [test_split_files] + baseline_split_files
    collapse_error_points = [False] + [True] * len(baseline_pairs)

    try:
        eval_module.visualizer(
            *plot_pairs,
            labels=plot_labels,
            directory=str(dataset_dir),
            forecast_name=forecast_name_rel,
            num_samples=None,
            split_files_by_pair=plot_split_files,
            collapse_error_points_by_pair=collapse_error_points,
        )
    except Exception as exc:
        print(f"[WARN] MLR {subset_label} plots failed: {exc}")

    try:
        boxplot_rows = eval_module._build_boxplot_error_rows_from_predictions(
            prediction_rows,
            model_label="MLR",
            baseline_labels=baseline_labels,
        )
        eval_module.boxplot_from_error_rows(
            boxplot_rows,
            directory=str(dataset_dir),
            forecast_name=forecast_name_rel,
        )
    except Exception as exc:
        print(f"[WARN] MLR {subset_label} boxplot failed: {exc}")

    return mlr_dir


def _mlr_selection_policy(use_preselected_feature_set: bool) -> tuple[dict | None, bool, dict]:
    if not use_preselected_feature_set:
        return None, True, {}
    return (
        {
            "use_mutual_info": False,
            "use_lasso": False,
            "deduplicate_threshold": None,
            "vif_threshold": float("inf"),
        },
        False,
        {
            "feature_selection": {
                "method": "Preselected feature set + constant-column filtering only",
                "use_mutual_info": False,
                "use_lasso": False,
                "deduplicate_threshold": None,
                "vif_threshold": float("inf"),
                "use_spearman_prefilter": False,
            }
        },
    )


def _run_mlr_variants_on_existing_split(
    *,
    train_samples,
    test_samples,
    feature_names: list[str],
    min_test_independent: int,
    model_context: str,
    use_preselected_feature_set: bool,
) -> list[dict]:
    from utils.mlr import MLR_VARIANTS as _MLR_VARIANTS

    selection_config, use_spearman_prefilter, feature_selection_extra = _mlr_selection_policy(
        use_preselected_feature_set=use_preselected_feature_set
    )
    results: list[dict] = []

    for variant in _MLR_VARIANTS:
        variant_name = str(variant["model_name"])
        try:
            preds, targets, train_used, test_used, meta, n_excl = _evaluate_mlr_with_rebalance(
                train_samples,
                test_samples,
                feature_names=feature_names,
                selection_config=selection_config,
                aggregation_mode=variant["aggregation_mode"],
                min_test_independent=int(min_test_independent),
                model_name=f"{variant_name} {model_context}",
                use_spearman_prefilter=use_spearman_prefilter,
            )
        except Exception as exc:
            results.append(
                {
                    "variant": dict(variant),
                    "error": exc,
                }
            )
            continue

        pf = np.asarray(preds, dtype=float).reshape(-1)
        tf = np.asarray(targets, dtype=float).reshape(-1)
        finite_mask = np.isfinite(pf) & np.isfinite(tf)
        pf = pf[finite_mask]
        tf = tf[finite_mask]

        failure_reasons = [
            str(m.get("failure_reason", "")).strip()
            for m in meta
            if str(m.get("failure_reason", "")).strip()
        ]
        fallback_modes = [
            str(m.get("fallback_mode", "")).strip()
            for m in meta
            if str(m.get("fallback_mode", "")).strip()
        ]

        if len(pf) < 1:
            results.append(
                {
                    "variant": dict(variant),
                    "preds": preds,
                    "targets": targets,
                    "train_samples": train_used,
                    "test_samples": test_used,
                    "meta": meta,
                    "n_excluded": int(n_excl),
                    "failure_reasons": failure_reasons,
                    "fallback_modes": fallback_modes,
                    "feature_selection_extra": feature_selection_extra,
                }
            )
            continue

        # Use the final post-selection feature set (selected_features from per-target meta),
        # falling back to Spearman-kept columns then all input features if unavailable.
        # This ensures different MLR variants only share a feature_tag if they genuinely
        # selected the same predictors (post-MI/Lasso/VIF), not just the same Spearman pool.
        selected_names: list[str] = []
        for meta_row in meta:
            selected_names = meta_row.get("selected_features") or []
            if selected_names:
                break
        if not selected_names:
            for meta_row in meta:
                selected_names = meta_row.get("spearman_kept_columns") or []
                if selected_names:
                    break
        effective_feature_names = list(selected_names) if selected_names else list(feature_names)
        feature_tag = _feature_tag(tuple(sorted(effective_feature_names)))

        results.append(
            {
                "variant": dict(variant),
                "preds": preds,
                "targets": targets,
                "train_samples": train_used,
                "test_samples": test_used,
                "meta": meta,
                "n_excluded": int(n_excl),
                "n_test_valid": int(len(pf)),
                "n_test_independent": _independent_name_count([str(s[2]) for s in test_used]),
                "mae": float(mean_absolute_error(tf, pf)),
                "rmse": float(np.sqrt(mean_squared_error(tf, pf))),
                "r2": float(r2_score(tf, pf)),
                "pearson_r": float(np.corrcoef(tf, pf)[0, 1]) if len(tf) >= 2 else float("nan"),
                "std_target_empirical": float(np.std(tf, ddof=1)) if len(tf) > 1 else float("nan"),
                "n_samples": int(len(pf)),
                "feature_tag": feature_tag,
                "effective_feature_names": effective_feature_names,
                "failure_reasons": failure_reasons,
                "fallback_modes": fallback_modes,
                "feature_selection_extra": feature_selection_extra,
            }
        )

    return results


def _map_to_raw_filenames(file_names: list[str]) -> list[str]:
    mapped = []
    seen = set()
    for file_name in file_names:
        mapped_name = re.sub(r"_mc_\d+(?=\.csv$)", "", str(file_name))
        if mapped_name not in seen:
            seen.add(mapped_name)
            mapped.append(mapped_name)
    return mapped


def _count_raw_totals(sample_dir: Path) -> int:
    files = [p.name for p in sorted(sample_dir.glob("*.csv"))]
    if any("_mc_" in name for name in files):
        return len(_map_to_raw_filenames(files))
    return len(files)


def _count_valid_samples_raw(
    data_dir: Path,
    sample_subdir: str,
    input_columns: list[str],
    input_rows: slice,
    output_columns: list[str],
    output_rows: list[int],
) -> tuple[int, int, int]:
    sample_dir = Path(data_dir, sample_subdir)
    all_total_raw = _count_raw_totals(sample_dir)

    loaded = load_samples(
        str(sample_dir),
        input_columns=input_columns,
        output_columns=output_columns,
        input_rows=input_rows,
        output_rows=output_rows,
        fault_tolerant=False,
    )
    loaded_count = len(loaded)
    loaded_names = [str(sample[2]) for sample in loaded]
    valid_raw = len(_map_to_raw_filenames(loaded_names)) if any("_mc_" in name for name in loaded_names) else loaded_count

    return int(valid_raw), int(all_total_raw), int(loaded_count)


def _objective_from_metrics(r2: float, drop_rate: float, lambda_drop: float) -> float:
    """Convert maximize-R2 search intent into a minimized scalar objective."""
    if not np.isfinite(r2):
        return float("inf")
    if not np.isfinite(drop_rate):
        drop_rate = 1.0
    return float((1.0 - r2) + lambda_drop * drop_rate)


def _candidate_rank_key(item: CandidateResult) -> tuple[float, float, float, str]:
    """Deterministic subset ranking key for objective minimization and R2 tie-breaks.

    Exact ties are broken toward the *smaller* subset. Two subsets that score
    identically are not equally good evidence: the smaller one claims less, and with
    11 candidate predictors against 12-47 independent test samples the larger one is
    the likelier to be fitting the holdout. This only settles exact ties; near-ties
    are the business of `_apply_one_se_rule`.
    """
    r2_tie = -float(item.r2) if np.isfinite(item.r2) else float("inf")
    # Degeneracy breaks ties and nothing more. It sits below the objective, so a model
    # whose predictions do not vary can still win outright if it genuinely scores best;
    # it only loses to a candidate it could not be separated from, which would otherwise
    # have been settled arbitrarily.
    return (float(item.objective), bool(item.degenerate), r2_tie,
            float(item.n_features), str(item.feature_tag))


def _bootstrap_objective_se(
    y: np.ndarray,
    p: np.ndarray,
    n_resamples: int = 400,
    seed: int = 0,
) -> float:
    """Standard error of ``1 - r2`` from resampling the scored points.

    With rolling-origin folds the spread across folds supplies this. Without them the
    objective is a single number and the one-standard-error rule would have nothing to
    act on -- yet the uncertainty is real and is precisely why the top few subsets
    cannot be told apart. Resampling the held-out points the candidate was actually
    scored on measures that uncertainty without fitting anything again. It is the same
    device the evidence statistics already use for the skill interval.

    Args:
        y: Held-out targets.
        p: Predictions for those targets.
        n_resamples: Bootstrap replicates.
        seed: Fixed so the same candidate always yields the same interval.

    Returns:
        The standard deviation of the resampled objective, or NaN when there are too
        few points, or when the resampled target variance collapses too often for the
        spread to mean anything.
    """
    if y.size < 4:
        return float("nan")
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    n = y.size
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    # Resample the squared errors only. Resampling the denominator too lets a draw
    # with little target variance produce an enormous ratio, which inflated the
    # estimate by an order of magnitude (Lead: 55.8 against 7.1). The denominator is
    # a property of the evaluation set, not of the model, so it stays fixed.
    sq_err = (p - y) ** 2
    vals = sq_err[idx].mean(axis=1) * n / sst
    return float(np.std(vals, ddof=1))


def _apply_one_se_rule(
    candidates: list[CandidateResult],
    best: CandidateResult,
    tolerance_se: float,
) -> CandidateResult:
    """Return the smallest subset statistically indistinguishable from *best*.

    The search objective is an estimate, not a measurement, and the spread between
    the best few subsets is routinely smaller than the uncertainty of the estimate
    itself. Selecting the argmin regardless reports a subset the search never
    actually distinguished from its neighbours. The one-standard-error rule -- the
    standard remedy -- keeps every candidate within `tolerance_se` standard errors of
    the best objective and returns the most parsimonious of them.

    The standard error comes from the spread across rolling-origin folds when those
    are in use, and otherwise from resampling the held-out points the candidate was
    scored on. Either way it is measured, not assumed; if neither is available *best*
    is returned unchanged rather than an interval being invented for it.

    Args:
        candidates: All scored candidates to consider.
        best: The argmin of the objective.
        tolerance_se: Width of the acceptance band, in standard errors.

    Returns:
        The selected candidate, which is *best* itself when nothing smaller qualifies.
    """
    if tolerance_se <= 0 or not candidates:
        return best
    se = float(best.cv_objective_se)
    if not np.isfinite(se) or se <= 0:
        return best

    threshold = float(best.objective) + tolerance_se * se
    within = [c for c in candidates if np.isfinite(c.objective) and c.objective <= threshold]
    if not within:
        return best
    # Smallest subset first; among equally small ones, the best objective.
    within.sort(key=lambda c: (int(c.n_features), float(c.objective), str(c.feature_tag)))
    chosen = within[0]
    if chosen.feature_tag != best.feature_tag:
        print(
            f"[SELECT] One-SE rule: {len(within)} subset(s) within {tolerance_se:.1f} SE "
            f"({se:.4f}) of the best objective {best.objective:.4f}. "
            f"Selecting {chosen.n_features} features (objective={chosen.objective:.4f}, "
            f"r2={chosen.r2:.4f}) over {best.n_features} features "
            f"(objective={best.objective:.4f}, r2={best.r2:.4f})."
        )
    return chosen


_SEGMENT_MC_SUFFIX = re.compile(r"_mc_[0-9]+$")


def _segment_base(name: str) -> str:
    """Segment id for a sample file, with any Monte Carlo replicate suffix removed."""
    return _SEGMENT_MC_SUFFIX.sub("", Path(str(name)).stem)


def _segment_order(name: str) -> int:
    m = re.search(r"([0-9]+)", _segment_base(name))
    return int(m.group(1)) if m else -1


def _pin_split_enabled() -> bool:
    """Whether every run reuses one pinned split per target.

    Carried in the environment rather than in a module global so that the parallel
    candidate evaluators, which re-import this module in a fresh process, inherit the
    setting instead of silently reverting to the default.
    """
    raw = str(os.environ.get("WQ_PIN_SPLIT", "1")).strip().lower()
    return raw not in {"0", "false", "no", ""}


def _canonical_probe_config(dataset_dir: Path) -> Path:
    """The single config whose split defines this target's boundary.

    The boundary must not depend on which model is being prepared. It did: each
    config was used to compute its own, and because GP trains on `samples` while
    XGBoost trains on `mc_replicates`, the two disagreed -- 30 training segments
    against 31, so one family was scored on 12 test segments and the other on 11.
    Resolving one config here, deterministically, from the dataset directory means
    every family lands on the same cut.
    """
    cfgs = _surrogate_candidates(sorted(Path(dataset_dir).glob("config_*.yml")))
    if not cfgs:
        raise FileNotFoundError(f"No training configs in {dataset_dir}")

    # The probe decides *where* the 70/30 boundary falls, not whether the split is
    # shared: every run reuses the pinned lists, so no run can train on a segment
    # another one is scored on whichever configuration defined them. What the probe
    # does control is which samples count as usable, through two settings the families
    # disagree on -- the window representation (`none` rejects a sample for any missing
    # value; an aggregation only where a predictor is missing throughout) and
    # `nan_tolerance` (0.0 for the Gaussian processes, 0.8 for XGBoost).
    #
    # The most inclusive pair is chosen, so the boundary reflects the fullest view of
    # the usable record rather than one family's tolerance. That this matters is not
    # hypothetical: on Turbidity the boundary moved from 30/12 to 31/11 purely because
    # the probe fell through from an XGBoost config to a Gaussian process one.
    def _inclusiveness(path):
        try:
            cfg = train_module.load_config(str(path))
        except Exception:
            return (1, 1.0)
        data = cfg.get("data", {}) or {}
        split = cfg.get("data_split", {}) or {}
        aggregated = str(data.get("input_aggregation", "none")).strip().lower() not in ("", "none")
        tol = split.get("nan_tolerance")
        try:
            tol = float(tol) if tol is not None else 1.0
        except (TypeError, ValueError):
            tol = 1.0
        # Sorted ascending, so: unaggregated first, then the highest tolerance.
        return (1 if aggregated else 0, -tol)

    ranked = sorted(cfgs, key=lambda c: _inclusiveness(c) + (c.name,))
    best = _inclusiveness(ranked[0])
    tied = [c for c in ranked if _inclusiveness(c) == best]
    # Among equally inclusive configurations, a stable preference so the boundary is
    # reproducible rather than dependent on which names happen to exist.
    for preferred in ("xgb_01", "transformer_01", "gp_01"):
        for c in tied:
            if c.stem.replace("config_", "") == preferred:
                return c
    return tied[0]


def _pinned_split_dir(dataset_dir: Path, sample_subdir: str) -> Path:
    """Where the one split per target lives, keyed by the subdirectory it names files in.

    Not keyed by row count: the boundary is a point in time, shared by every model of
    this target whatever window length it reads. A model whose window cannot be filled
    for some segment simply drops it, which shows up in its own sample count.
    """
    return (Path(dataset_dir) / "forecasts" / "pinned_split" / str(sample_subdir))


def _materialize_pinned_split(
    dataset_dir: Path,
    sample_subdir: str,
) -> Path:
    """Fix one train/test boundary per target and write it once, for every run to reuse.

    Each run currently recomputes its own boundary as `train_fraction` of *that run's*
    valid sample count, and validity depends on the feature subset. For 8 of the 14
    targets the boundary therefore lands in different places for different runs -- the
    Total coliforms test set starts at segment 114, 115, 116, 118, 119 or 120 depending
    on the run. There is consequently no single test set, comparisons across families
    are not like-for-like, and the common-set intersection discards whatever the runs
    disagree about.

    Pinning removes the cause. The boundary is computed once, from the full feature
    set, and every run reuses it. Both lists name *all* segments on their side rather
    than only the ones the probing feature set could use, so a subset that can use a
    segment the full set cannot still gets it; `load_samples` drops the rest per run.

    Returns:
        The directory holding `train_files.txt` and `test_files.txt`.

    Raises:
        ValueError: When either side of the boundary would be empty.
    """
    out_dir = _pinned_split_dir(dataset_dir, sample_subdir)
    train_file = out_dir / "train_files.txt"
    test_file = out_dir / "test_files.txt"
    if train_file.exists() and test_file.exists():
        return out_dir

    probe = _canonical_probe_config(dataset_dir)
    cfg = train_module.load_config(str(probe))
    probe_data = cfg["data"]
    full_features = tuple(probe_data["input_columns"])
    row_count = int(probe_data["input_row_2"]) - int(probe_data["input_row_1"])
    tmp_cfg_dir = _forecast_sweeps_dir(dataset_dir) / "configs"
    train_segments = _training_portion_segments(
        dataset_dir=dataset_dir,
        surrogate_config_path=probe,
        row_count=row_count,
        full_features=full_features,
        tmp_cfg_dir=tmp_cfg_dir,
    )
    if not train_segments:
        raise ValueError(f"Pinned split for {dataset_dir.name} has an empty training side.")
    boundary = max(_segment_order(seg) for seg in train_segments)

    sample_dir = Path(dataset_dir) / sample_subdir
    files = sorted(p.name for p in sample_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No sample files under {sample_dir}")

    train_names, test_names = [], []
    for name in files:
        (train_names if _segment_order(name) <= boundary else test_names).append(name)
    if not train_names or not test_names:
        raise ValueError(
            f"Pinned split for {dataset_dir.name} [{sample_subdir}] puts every segment "
            f"on one side of segment {boundary}."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_split_file_names(train_file, train_names)
    _write_split_file_names(test_file, test_names)
    n_train_seg = len({_segment_base(n) for n in train_names})
    n_test_seg = len({_segment_base(n) for n in test_names})
    print(
        f"[PIN] {dataset_dir.name} [{sample_subdir}]: boundary after segment {boundary} "
        f"(from {probe.stem}); {n_train_seg} training and {n_test_seg} test segment(s). "
        "Every run of this target reuses this split."
    )
    return out_dir


def _training_portion_segments(
    dataset_dir: Path,
    surrogate_config_path: Path,
    row_count: int,
    full_features: tuple[str, ...],
    tmp_cfg_dir: Path,
) -> set[str]:
    """Segments on the training side of the reported split, and only those.

    Feature selection must not see the segments the results table is scored on. It
    currently does: a search-phase candidate run and the final reported run resolve to
    the same 70/30 temporal split, so all 240 candidate evaluations land on the same
    test segments that Table 3 reports. That is the source of the optimistic bias the
    manuscript discloses, and confining the search to the training side removes it
    rather than documenting it.

    The boundary is taken from the *full* feature set. Validity is monotone in the
    number of columns -- dropping a predictor can only make more segments usable -- so
    the full set has the fewest valid samples and therefore the earliest boundary. Any
    subset the search tries has a boundary at or after this one, so folds built here
    can never reach into a candidate's own test segments.

    Returns:
        Base segment names (no replicate suffix) making up the training portion.
    """
    cfg_path = _prepare_variant_config(
        base_config_path=surrogate_config_path,
        row_count=row_count,
        features=full_features,
        feature_tag="splitprobe",
        tmp_dir=tmp_cfg_dir,
        forced_data_dir=dataset_dir,
        # This probe computes the boundary the pinned split is made from, so it must
        # compute its own; reusing a pinned split here would be circular.
        pin_split=False,
    )
    cfg = train_module.load_config(str(cfg_path))
    cfg = train_module.merge_with_defaults(cfg, cfg["model_type"])
    with contextlib.redirect_stdout(io.StringIO()):
        train_samples, test_samples = train_module.load_and_split_data(cfg)
    train_segs = {_segment_base(str(sample[2])) for sample in train_samples}
    test_segs = {_segment_base(str(sample[2])) for sample in test_samples}

    # The probe trains nothing; it exists only to read the split. Leaving its directory
    # behind puts a run in the results tree that has split files but no model, which
    # every downstream scan then has to recognise and skip.
    try:
        probe_dir = Path(cfg["data"]["forecast_dir"])
        if probe_dir.is_dir() and "splitprobe" in probe_dir.name:
            shutil.rmtree(probe_dir, ignore_errors=True)
    except Exception:
        pass
    overlap = train_segs & test_segs
    if overlap:
        raise RuntimeError(
            f"Reported split places {len(overlap)} segment(s) in both train and test; "
            "folds cannot be built safely from it."
        )
    print(
        f"[CV] Reported split for {dataset_dir.name}: {len(train_segs)} training "
        f"segment(s), {len(test_segs)} held back for the results table and excluded "
        "from feature selection entirely."
    )
    return train_segs


def _cv_folds_root(dataset_dir: Path, row_count: int) -> Path:
    """Where the pinned fold split files live.

    Deliberately a sibling of the sweep namespace, not a child of it: the
    post-processors enumerate every directory under `feature_sweeps` as a candidate
    run, and a fold directory holding split files but no model would be walked as
    though it were one.
    """
    return Path(dataset_dir) / "forecasts" / "cv_folds" / f"r{int(row_count):03d}"


def _cv_folds_dir(dataset_dir: Path, row_count: int, sample_subdir: str) -> Path:
    """Fold directory for one sample subdirectory.

    GP and MLR train on `samples` while XGBoost and the Transformer train on
    `mc_replicates`, so the file lists differ even though the segment boundaries do
    not. Keeping them apart stops one family's list being handed to another.
    """
    return _cv_folds_root(dataset_dir, row_count) / str(sample_subdir)


def _materialize_cv_folds(
    dataset_dir: Path,
    sample_subdir: str,
    row_count: int,
    n_folds: int,
    min_train_fraction: float,
    eligible_segments: set[str] | None = None,
) -> list[Path]:
    """Write one pinned train/test split per rolling-origin fold, and return their dirs.

    Every candidate subset is scored on the *same* folds. That is the point: if the
    folds moved with the subset, differences between candidates would confound the
    subset with the evaluation set, which is the defect this whole change exists to
    remove. Samples a given subset cannot use are dropped by `load_samples` as usual,
    and that loss is already priced into the objective through `drop_rate`.

    Folds are built over segment groups, never over individual Monte Carlo
    replicates, so replicates of one segment can never straddle a fold boundary.

    Args:
        dataset_dir: Dataset directory holding the sample subdirectories.
        sample_subdir: Subdirectory the surrogate trains on (`samples` or `mc_replicates`).
        row_count: Lookback row count, used only to namespace the fold directory.
        n_folds: Requested fold count; reduced when too few segments exist.
        min_train_fraction: Fraction of segments reserved as the initial training run.

    Returns:
        Fold directories, in temporal order, each containing `train_files.txt` and
        `test_files.txt`.

    Raises:
        FileNotFoundError: When the sample subdirectory holds no CSV files.
    """
    sample_dir = Path(dataset_dir) / sample_subdir
    files = sorted(p.name for p in sample_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No sample files under {sample_dir}")

    groups: dict[str, list[str]] = {}
    for name in files:
        groups.setdefault(_segment_base(name), []).append(name)
    ordered_keys = sorted(groups, key=_segment_order)

    if eligible_segments is not None:
        ordered_keys = [k for k in ordered_keys if k in eligible_segments]
        if len(ordered_keys) < 2:
            raise ValueError(
                f"Only {len(ordered_keys)} training segment(s) available under "
                f"{sample_dir}; cannot build rolling-origin folds."
            )

    splits = rolling_origin_block_splits(
        n_groups=len(ordered_keys),
        n_folds=int(n_folds),
        min_train_fraction=float(min_train_fraction),
    )

    root = _cv_folds_dir(dataset_dir, row_count, sample_subdir)
    root.mkdir(parents=True, exist_ok=True)
    fold_dirs: list[Path] = []
    for idx, (n_train_groups, (lo, hi)) in enumerate(splits, start=1):
        train_names = [n for k in ordered_keys[:n_train_groups] for n in sorted(groups[k])]
        test_names = [n for k in ordered_keys[lo:hi] for n in sorted(groups[k])]
        fold_dir = root / f"fold_{idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_split_file_names(fold_dir / "train_files.txt", train_names)
        _write_split_file_names(fold_dir / "test_files.txt", test_names)
        fold_dirs.append(fold_dir)

    scope = "training-portion" if eligible_segments is not None else "all"
    print(
        f"[CV] {dataset_dir.name} r{int(row_count):03d} [{sample_subdir}]: "
        f"{len(fold_dirs)} rolling-origin fold(s) over {len(ordered_keys)} "
        f"{scope} segments; first fold trains on {splits[0][0]}, "
        f"scoring {len(ordered_keys) - splits[0][0]} segments in total."
    )
    return fold_dirs


def _pooled_predictions(forecast_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Held-out targets and predictions from one evaluated run."""
    pred_csv = Path(forecast_dir) / "predictions.csv"
    if not pred_csv.is_file():
        return np.array([]), np.array([])
    df = pd.read_csv(pred_csv)
    if "kind" in df.columns:
        df = df[df["kind"].astype(str) == "test"]
    if df.empty or "target" not in df.columns:
        return np.array([]), np.array([])
    # The model's own prediction column sits immediately after `target`; reference
    # forecasts and uncertainty columns follow it.
    after = list(df.columns[df.columns.get_loc("target") + 1:])
    skip = {"Naive", "Seasonal", "Linear"}
    pred_col = next(
        (c for c in after if c not in skip and not str(c).endswith(("_std", "_var"))),
        None,
    )
    if pred_col is None:
        return np.array([]), np.array([])
    y = pd.to_numeric(df["target"], errors="coerce").to_numpy(dtype=float)
    pv = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(pv)
    return y[ok], pv[ok]


def _prediction_spread(y: np.ndarray, p: np.ndarray) -> tuple:
    """``(pred_std, degenerate)`` for one set of held-out predictions.

    Degeneracy is judged against the target's own spread rather than an absolute
    threshold, so it means the same thing whatever the units. It is deliberately a test
    of variance, not of accuracy: a model can be inaccurate and still informative, and
    only a model whose predictions do not vary at all is uninformative by construction.
    That distinction matters because the existing `min_r2` guardrail tests accuracy and
    therefore fires on every trial for targets where nothing fits, discriminating
    nothing.
    """
    if p.size < 2:
        return float("nan"), False
    ps = float(np.std(p))
    ys = float(np.std(y)) if y.size >= 2 else 0.0
    if not np.isfinite(ps):
        return float("nan"), False
    threshold = max(1e-12, 1e-4 * ys)
    return ps, bool(ps < threshold)


def _pooled_r2(y: np.ndarray, p: np.ndarray) -> float:
    if y.size < 2:
        return float("nan")
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((p - y) ** 2) / denom)


# Families whose fit depends on a random draw. GP and MLR are deterministic given their
# data -- six seeds of a GP winner reproduce bit-identical predictions on eight of nine
# targets -- so seeding them costs time and changes nothing.

# Set once from --seeds/--seed-base. Default 1 reproduces the previous behaviour exactly:
# one fit per candidate, at whatever seed the config carries.
_CANDIDATE_SEEDS = 1
_CANDIDATE_SEED_BASE = 0


# Suffix marking a seed replicate. Deliberately not `_s%02d`: `_s01` is already a subset
# label in the run-directory convention, beside `_k01`-`_k04`, `_l01` and `_m01` -- CV22
# has 182 legitimate `_s01` directories -- so a seed replicate and a subset run were
# indistinguishable by name and could collide outright. `_seed01` matches no subset
# pattern, which is what lets `z8_CommonSetMetrics` exclude replicates by name.
SEED_REPLICATE_RE = r"_seed\d+$"


def _is_stochastic_model(model_type: str, hyper: dict | None = None) -> bool:
    """Whether refitting this model at another seed can change its predictions.

    XGBoost and the transformer draw from ``random_state``. The Gaussian process was
    long treated as deterministic, and its optimizer is, but its uncertain-input kernel
    draws 64 Monte Carlo samples per fit from ``uncertain_kernel_mc_seed``. That is not
    a technicality: it is why pH and Turbidity disagreed between Table 3 and the horizon
    curve, whose replicates varied that draw where the reported fit held it at 0.
    """
    mt = str(model_type or "").strip().lower()
    if mt in ("xgb_regressor", "xgb_classifier", "transformer"):
        return True
    if mt == "gp_regressor":
        return bool((hyper or {}).get("use_uncertain_input_kernel"))
    return False


def _seeded_variant_config(variant_cfg: Path, seed: int) -> Path:
    """A copy of *variant_cfg* fitted at *seed*, in a directory z8 does not score.

    ``hyperparameters.random_state`` is the model seed, reaching XGBoost through the
    constructor and the transformer through ``_seed_model_rng``. Where the Gaussian
    process uses its uncertain-input kernel, ``uncertain_kernel_mc_seed`` is the seed
    that matters and is set alongside it.

    Two things about *where* these go were wrong and are fixed here.

    The suffix must not be ``_s%02d``. ``_s01`` is already a subset label in the
    run-directory convention, beside ``_k01``-``_k04``, ``_l01`` and ``_m01`` -- CV22 has
    182 legitimate ``_s01`` directories -- so a seed replicate and a subset run were
    indistinguishable by name and could collide outright.

    Second, the replicates must not be *scored* as candidates.
    ``z8_CommonSetMetrics.load_runs`` scans ``feature_sweeps/`` indiscriminately: it takes
    every directory with a family prefix and a ``predictions.csv`` as a candidate. Every
    replicate therefore entered the selection pool in its own right, so ``--seeds N``
    added N single-seed draws per candidate and let the maximum be chosen -- making
    selection *more* seed-fragile, the opposite of the flag's purpose. In the CV23 smoke
    run 41 of 119 scoreable directories were replicates.

    They stay inside ``feature_sweeps/`` because the two resolvers disagree about what a
    forecast name is relative to -- ``e_Train`` joins it under ``forecasts/`` verbatim
    while this module prepends the sweep namespace -- so a directory prefix here lands a
    replicate in ``forecasts/feature_sweeps/seed_reps/`` for one and
    ``forecasts/seed_reps/`` for the other. ``z8`` excludes them by the ``_seedNN``
    suffix instead, which is unambiguous by construction.
    """
    with open(variant_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    hyper = cfg.setdefault("hyperparameters", {})
    hyper["random_state"] = int(seed)
    if hyper.get("use_uncertain_input_kernel"):
        hyper["uncertain_kernel_mc_seed"] = int(seed)
    data_cfg = cfg.setdefault("data", {})
    # Suffix the final component only, leaving whatever namespace prefix the caller set.
    name = str(data_cfg.get("forecast_name", "candidate")).replace("\\", "/")
    head, _, leaf = name.rpartition("/")
    data_cfg["forecast_name"] = "%s%s_seed%02d" % (
        head + "/" if head else "", leaf, int(seed))
    out = variant_cfg.with_name(
        "%s_seed%02d%s" % (variant_cfg.stem, int(seed), variant_cfg.suffix))
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out


def _mean_metric_rows(rows: list[dict]) -> dict:
    """Average the numeric metrics of repeated fits, keeping the last row's structure.

    Only the scores are averaged. Counts such as ``n_test_independent`` are identical
    across seeds by construction -- the split does not move -- so taking the last row's
    value for everything else keeps the contract validation intact.
    """
    if len(rows) == 1:
        return rows[0]
    out = dict(rows[-1])
    for key in ("r2", "rmse", "mae", "pearson_r", "nrmse"):
        vals = [pd.to_numeric(r.get(key, np.nan), errors="coerce") for r in rows]
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _evaluate_candidate(
    dataset_dir: Path,
    target_name: str,
    surrogate_config_path: Path,
    row_count: int,
    features: tuple[str, ...],
    feature_tag: str,
    lambda_drop: float,
    tmp_cfg_dir: Path,
    disable_baselines_for_search: bool,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
    cv_fold_dirs: list[Path] | None = None,
    n_seeds: int | None = None,
    seed_base: int | None = None,
) -> CandidateResult:
    """Score one candidate subset.

    With `cv_fold_dirs`, the subset is fitted once per rolling-origin fold and scored
    on the pooled held-out predictions, and the spread across folds gives the standard
    error the one-standard-error rule needs. Without it, the subset is scored on the
    run's single 70/30 holdout -- 12 to 47 independent samples depending on target,
    which is too few to separate candidates reliably once 240 of them have been tried
    against it.
    """
    try:
        uses_uncertainty = _candidate_uses_uncertainty_distributions(tuple(features))
        base_cfg = train_module.load_config(str(surrogate_config_path))
        base_data = base_cfg["data"]
        base_stop = int(base_data["input_row_2"])
        input_rows = slice(base_stop - row_count, base_stop)

        # Use discovered dataset_dir as the authoritative data root for this run.
        data_dir_resolved = Path(dataset_dir).resolve()
        sample_subdir = str(base_data.get("sample_subdir", "samples"))
        output_columns = list(base_data["output_columns"])
        output_rows = list(base_data["output_rows"])

        valid_raw, total_raw, valid_loaded = _count_valid_samples_raw(
            data_dir_resolved,
            sample_subdir,
            list(features),
            input_rows,
            output_columns,
            output_rows,
        )
        drop_rate = float(1.0 - (valid_raw / total_raw)) if total_raw > 0 else 1.0

        if not uses_uncertainty:
            print(
                f"[MC-POLICY] {dataset_dir.name} r{int(row_count):03d} {feature_tag}: "
                "no uncertainty-enabled predictors in subset; evaluating collapsed originals only."
            )

        def _fit_and_score(fold_dir: Path | None, fold_index: int | None,
                           seed: int | None = None) -> tuple[dict, Path]:
            """Train and evaluate this subset once; return the metric row and its dir."""
            variant_cfg = _prepare_variant_config(
                base_config_path=surrogate_config_path,
                row_count=row_count,
                features=features,
                feature_tag=feature_tag,
                tmp_dir=tmp_cfg_dir,
                forced_data_dir=dataset_dir,
                cv_fold_dir=fold_dir,
                cv_fold_index=fold_index,
            )
            if seed is not None:
                variant_cfg = _seeded_variant_config(variant_cfg, seed)
            eval_cfg_path = _train_single_config(
                variant_cfg,
                dataset_dir,
                disable_training_plots=disable_training_plots,
                disable_eval_plots=disable_eval_plots,
                suppress_training_logs=suppress_training_logs,
            )
            _set_eval_overrides(
                eval_cfg_path,
                run_baselines=not disable_baselines_for_search,
            )
            row = eval_module.evaluate_single_config(
                str(eval_cfg_path),
                save_plots_override=not disable_eval_plots,
            )
            if row is None:
                raise RuntimeError(f"Evaluation returned None for config: {eval_cfg_path}")
            ctx = (f"{eval_cfg_path.parent.name} "
                   f"[{dataset_dir.name} r{int(row_count):03d} {feature_tag}]")
            _validate_eval_metric_contract(row, context=ctx)
            return row, eval_cfg_path.parent

        cv_folds_used = 0
        cv_r2_mean = float("nan")
        cv_r2_se = float("nan")
        cv_objective_se = float("nan")
        pred_std = float("nan")
        is_degenerate = False

        if cv_fold_dirs:
            fold_rows: list[dict] = []
            fold_r2s: list[float] = []
            ys, ps = [], []
            last_dir: Path | None = None
            for fold_index, fold_dir in enumerate(cv_fold_dirs, start=1):
                try:
                    row, forecast_dir = _fit_and_score(fold_dir, fold_index)
                except SampleComplianceError as exc:
                    # A fold this subset cannot satisfy is a property of the subset,
                    # not an error to swallow: record the shortfall and let the fold
                    # count fall, rather than scoring the candidate on fewer folds
                    # than its competitors without saying so.
                    print(
                        f"[CV] {feature_tag} fold {fold_index}: not evaluable "
                        f"({exc.reason}); excluded from this candidate's score."
                    )
                    continue
                # Harvest before the next fold reuses this directory.
                y_fold, p_fold = _pooled_predictions(forecast_dir)
                if y_fold.size:
                    ys.append(y_fold)
                    ps.append(p_fold)
                fold_rows.append(row)
                last_dir = forecast_dir
                fold_r2s.append(float(pd.to_numeric(row.get("r2", np.nan), errors="coerce")))

            if not fold_rows:
                print(f"[ERROR] No rolling-origin fold was evaluable for {feature_tag}.")
                return None

            if not ys:
                print(f"[ERROR] No held-out predictions recovered for {feature_tag}.")
                return None
            y_all = np.concatenate(ys)
            p_all = np.concatenate(ps)

            # Pooled, not averaged. A fold holding two segments produces an R2 whose
            # denominator is those two segments' own variance, which is unstable
            # enough to dominate a mean over folds. Pooling scores every held-out
            # point against one target variance.
            r2 = _pooled_r2(y_all, p_all)
            pred_std, is_degenerate = _prediction_spread(y_all, p_all)
            err = p_all - y_all
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            n_test_samples = float(sum(
                _extract_required_independent_metric(r, "n_test_independent", context=feature_tag)
                for r in fold_rows
            ))
            model_row = fold_rows[-1]
            cv_folds_used = len(fold_rows)

            finite = [v for v in fold_r2s if np.isfinite(v)]
            if len(finite) > 1:
                cv_r2_mean = float(np.mean(finite))
                cv_r2_se = float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                # The objective is (1 - r2) + const, so its standard error is the
                # standard error of r2.
                cv_objective_se = cv_r2_se
            elif finite:
                cv_r2_mean = float(finite[0])

            eval_dir_for_stop = last_dir
        else:
            model_type = str(base_cfg.get("model_type", "")).strip().lower()
            want = _CANDIDATE_SEEDS if n_seeds is None else int(n_seeds)
            base_seed = _CANDIDATE_SEED_BASE if seed_base is None else int(seed_base)
            # Same test as the final stage. The search previously used a literal tuple
            # that omitted the Gaussian process, so a GP surrogate was selected on one
            # draw while the final re-fit of the same model was ensembled over N -- the
            # two stages disagreeing about what is stochastic.
            reps = (int(want) if int(want) > 1
                    and _is_stochastic_model(model_type, base_cfg.get("hyperparameters"))
                    else 1)
            try:
                if reps > 1:
                    seed_rows: list[dict] = []
                    for k in range(reps):
                        r_k, eval_dir_for_stop = _fit_and_score(
                            None, None, seed=int(base_seed) + k)
                        seed_rows.append(r_k)
                    finite = [float(pd.to_numeric(r.get("r2", np.nan), errors="coerce"))
                              for r in seed_rows]
                    finite = [v for v in finite if np.isfinite(v)]
                    if len(finite) > 1:
                        print("         seeds=%d  r2 mean %+.4f  sd %.4f"
                              % (reps, float(np.mean(finite)), float(np.std(finite, ddof=1))))
                    model_row = _mean_metric_rows(seed_rows)
                else:
                    model_row, eval_dir_for_stop = _fit_and_score(None, None)
            except RuntimeError as exc:
                print(f"[ERROR] {exc}")
                print(f"         Features: {features}")
                print(f"         Row count: {row_count}")
                print(f"         Data dir: {data_dir_resolved}")
                print(f"         Surrogate config: {surrogate_config_path}")
                return None
            context = f"{eval_dir_for_stop.name} [{dataset_dir.name} r{int(row_count):03d} {feature_tag}]"
            rmse = _extract_required_independent_metric(model_row, "rmse", context=context)
            mae = _extract_required_independent_metric(model_row, "mae", context=context)
            n_test_samples = _extract_required_independent_metric(
                model_row, "n_test_independent", context=context)
            r2 = float(pd.to_numeric(model_row.get("r2", np.nan), errors="coerce"))
            y_hold, p_hold = _pooled_predictions(eval_dir_for_stop)
            cv_objective_se = _bootstrap_objective_se(y_hold, p_hold)
            pred_std, is_degenerate = _prediction_spread(y_hold, p_hold)

        input_dim = float(model_row.get("input_dim", np.nan))
        target_dim = float(model_row.get("target_dim", np.nan))
        objective = _objective_from_metrics(r2=r2, drop_rate=drop_rate, lambda_drop=lambda_drop)
        training_stop_reason = _load_training_stop_reason(eval_dir_for_stop)

        return CandidateResult(
            dataset=dataset_dir.name,
            target=target_name,
            row_count=int(row_count),
            n_features=int(len(features)),
            feature_tag=feature_tag,
            features=tuple(features),
            objective=objective,
            rmse=rmse,
            r2=r2,
            mae=mae,
            drop_rate=drop_rate,
            n_valid_raw=float(valid_raw),
            n_total_raw=float(total_raw),
            n_valid_loaded=float(valid_loaded),
            n_test_samples=n_test_samples,
            input_dim=input_dim,
            target_dim=target_dim,
            training_stop_reason=training_stop_reason,
            cv_folds=int(cv_folds_used),
            cv_r2_mean=cv_r2_mean,
            cv_r2_se=cv_r2_se,
            cv_objective_se=cv_objective_se,
            pred_std=pred_std,
            degenerate=bool(is_degenerate),
        )
    except Exception as exc:
        print(f"[ERROR] Exception in _evaluate_candidate for config: {surrogate_config_path}")
        print(f"        Features: {features}")
        print(f"        Row count: {row_count}")
        print(f"        Data dir: {dataset_dir}")
        import traceback
        traceback.print_exc()
        return None


def _evaluate_candidate_worker(payload: dict) -> CandidateResult | None:
    """Process worker for candidate evaluation payloads."""
    return _evaluate_candidate(
        dataset_dir=Path(payload["dataset_dir"]),
        target_name=str(payload["target_name"]),
        surrogate_config_path=Path(payload["surrogate_config_path"]),
        row_count=int(payload["row_count"]),
        features=tuple(payload["features"]),
        feature_tag=str(payload["feature_tag"]),
        lambda_drop=float(payload["lambda_drop"]),
        tmp_cfg_dir=Path(payload["tmp_cfg_dir"]),
        disable_baselines_for_search=bool(payload["disable_baselines_for_search"]),
        disable_training_plots=bool(payload["disable_training_plots"]),
        disable_eval_plots=bool(payload["disable_eval_plots"]),
        suppress_training_logs=bool(payload["suppress_training_logs"]),
        n_seeds=int(payload.get("n_seeds", 1)),
        seed_base=int(payload.get("seed_base", 0)),
        cv_fold_dirs=[Path(d) for d in payload.get("cv_fold_dirs") or []] or None,
    )


def _candidate_key(row_count: int, features: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    return int(row_count), tuple(features)


def _plot_final_metrics_comparison(final_df: pd.DataFrame, output_dir: Path) -> Path:
    """Write a 4-panel clustered-bar comparison figure from final metrics rows.

    Panels: MAE, RMSE, Pearson's r, and R^2. Clusters are candidate subsets
    ordered by descending R^2 for the model type with the best (highest) subset R^2.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "feature_sweep_final_metrics_summary.png"

    def _format_bar_label(value: float) -> str:
        if not np.isfinite(value):
            return ""
        abs_v = abs(float(value))
        if abs_v >= 100.0:
            # 2 significant figures in scientific notation.
            return f"{value:.1e}"
        if abs_v == 0.0:
            return "0"
        decimals = max(0, 2 - 1 - int(np.floor(np.log10(abs_v))))
        txt = f"{value:.{decimals}f}"
        if "." in txt:
            txt = txt.rstrip("0").rstrip(".")
        return txt

    if final_df is None or final_df.empty:
        raise ValueError("Cannot plot final metrics summary: final_df is empty.")

    required_cols = ["subset_rank", "model", "mae", "rmse", "pearson_r", "r2"]
    missing = [c for c in required_cols if c not in final_df.columns]
    if missing:
        raise ValueError(f"Cannot plot final metrics summary: missing column(s) {missing}")

    df = final_df.copy()
    df["subset_rank"] = pd.to_numeric(df["subset_rank"], errors="coerce")
    df = df[df["subset_rank"].notnull()].copy()
    df["subset_rank"] = df["subset_rank"].astype(int)

    # Merge clusters that share the same feature_tag (identical feature sets should
    # be plotted once, with a combined label like "k01/s01/l01").
    _combined_rank_labels: dict[int, str] = {}
    if "feature_tag" in df.columns:
        # Each feature_tag claims its minimum rank.  If two different feature_tags
        # would claim the same minimum rank, bump the later one to an unoccupied rank
        # so that they remain distinct clusters rather than silently overwriting each other.
        _ft_min_raw = df.groupby("feature_tag")["subset_rank"].min().to_dict()
        _assigned_ranks: dict[int, str] = {}   # rank -> feature_tag that owns it
        _next_free = int(df["subset_rank"].max()) + 1
        _ft_primary: dict[str, int] = {}
        for ft, raw_min in sorted(_ft_min_raw.items(), key=lambda kv: kv[1]):
            rank = int(raw_min)
            if rank not in _assigned_ranks:
                _assigned_ranks[rank] = ft
                _ft_primary[ft] = rank
            else:
                # Collision: give this feature_tag the next free rank.
                _ft_primary[ft] = _next_free
                _assigned_ranks[_next_free] = ft
                _next_free += 1

        _ft_labels: dict[str, str] = {}
        for ft in df["feature_tag"].dropna().unique():
            labels = sorted(
                df.loc[df["feature_tag"] == ft, "subset_label"].dropna().unique()
            )
            _ft_labels[ft] = "/".join(labels) if labels else ft
        df["subset_rank"] = df["feature_tag"].map(_ft_primary).fillna(df["subset_rank"]).astype(int)
        _combined_rank_labels = {
            _ft_primary[ft]: _ft_labels[ft] for ft in _ft_primary
        }

    def _normalize_plot_model(raw: object) -> str:
        baseline_id = _normalize_baseline_label(raw)
        if baseline_id is not None:
            return baseline_id
        lowered = str(raw).strip().lower()
        if lowered in FINAL_METRICS_MODEL_STYLE:
            return lowered
        if "transformer" in lowered:
            return "transformer"
        if "xgb" in lowered:
            return "xgb_regressor"
        if "gp" in lowered:
            return "gp_regressor"
        return lowered

    df["model_norm"] = df["model"].apply(_normalize_plot_model)
    df = df[df["model_norm"].isin(FINAL_METRICS_MODEL_ORDER)].copy()

    # One series per model configuration, not per family. A family now spans several
    # window representations -- xgb_01 reads the flattened window, xgb_02 daily
    # summaries, xgb_03 six-hourly -- and they are different models, not repeats of one.
    # Collapsing them into a family bar averaged +0.428 against -1.985 and drew -0.505,
    # a number no model produced and which hid the best result on the target.
    if "variant" in df.columns:
        _variant = df["variant"].fillna("").astype(str).str.strip()
        df["plot_key"] = _variant.where(_variant != "", df["model_norm"])
    else:
        df["plot_key"] = df["model_norm"]

    # Order: families in their established sequence, variants in name order within each.
    _family_of_key = dict(zip(df["plot_key"], df["model_norm"]))
    _family_rank = {m: i for i, m in enumerate(FINAL_METRICS_MODEL_ORDER)}
    plot_order = sorted(
        dict.fromkeys(df["plot_key"]),
        key=lambda k: (_family_rank.get(_family_of_key.get(k, k), 99), str(k)),
    )

    # Shade the variants of a family around its base colour so the family still reads as
    # a block, while each configuration stays separately identifiable.
    def _shaded(base_hex: str, position: int, count: int) -> str:
        if count <= 1:
            return base_hex
        r, g, b = (int(base_hex[i:i + 2], 16) for i in (1, 3, 5))
        # Spread from 70% to 130% of the base luminance.
        f = 0.70 + 0.60 * (position / max(1, count - 1))
        return "#%02x%02x%02x" % tuple(min(255, max(0, int(round(c * f)))) for c in (r, g, b))

    plot_style = {}
    for _fam in dict.fromkeys(_family_of_key.get(k, k) for k in plot_order):
        _keys = [k for k in plot_order if _family_of_key.get(k, k) == _fam]
        _base = FINAL_METRICS_MODEL_STYLE.get(_fam, {"label": _fam, "color": "#777777", "hatch": ""})
        for _i, _k in enumerate(_keys):
            plot_style[_k] = {
                "label": _base["label"] if len(_keys) == 1 else str(_k),
                "color": _shaded(_base["color"], _i, len(_keys)),
                "hatch": _base.get("hatch", ""),
            }
    if df.empty:
        raise ValueError("Cannot plot final metrics summary: no recognized model rows found.")

    metric_cols = ["mae", "rmse", "pearson_r", "r2"]
    for metric in metric_cols:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    # Collapse genuinely duplicate rows -- a baseline is written once per ML config, with
    # identical values -- by keeping one whole row rather than averaging across rows. Every
    # bar is then a single real model, and its metrics are mutually consistent.
    _agg_cols = metric_cols + (["min_skill_rmse"] if "min_skill_rmse" in df.columns else [])
    _ranked = df.sort_values("r2", ascending=False, na_position="last")
    grouped = (
        _ranked.groupby(["subset_rank", "plot_key"], as_index=False)
        .first()[["subset_rank", "plot_key"] + _agg_cols]
    )
    grouped = grouped.rename(columns={"plot_key": "model_norm"})

    # Build rank → display label mapping.
    # Use combined labels from feature_tag merging when available; fall back to subset_label.
    if _combined_rank_labels:
        rank_to_label = dict(_combined_rank_labels)
        # Ensure all ranks in grouped are present (in case some lack feature_tag).
        for rank in grouped["subset_rank"].dropna().unique():
            rank_to_label.setdefault(int(rank), f"k{int(rank):02d}")
    elif "subset_label" in df.columns:
        rank_to_label = {}
        _lbl_pairs = df[["subset_rank", "subset_label"]].dropna(subset=["subset_rank"]).drop_duplicates()
        for _, _row in _lbl_pairs.iterrows():
            r = int(_row["subset_rank"])
            lbl = str(_row["subset_label"]) if pd.notna(_row["subset_label"]) else f"k{r:02d}"
            rank_to_label[r] = lbl
    else:
        rank_to_label = {
            int(rank): f"k{int(rank):02d}"
            for rank in sorted(grouped["subset_rank"].dropna().unique().tolist())
        }

    r2_pivot = grouped.pivot(index="subset_rank", columns="model_norm", values="r2")
    rmse_pivot = grouped.pivot(index="subset_rank", columns="model_norm", values="rmse")
    finite_max_r2 = {}
    for model in plot_order:
        if model not in r2_pivot.columns:
            continue
        vals = pd.to_numeric(r2_pivot[model], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            finite_max_r2[model] = float(np.max(vals))

    if not finite_max_r2:
        raise ValueError("Cannot plot final metrics summary: R2 values are all non-finite.")

    # Build RMSE-based minimum skill pivot for cross-model ordering decisions.
    # min_skill = (rmse_best_baseline - rmse_model) / rmse_best_baseline
    # Use the pre-computed min_skill_rmse column when present (avoids recomputation
    # and handles rows at novel ranks that have no matching baseline in this DataFrame).
    _baseline_mask = grouped["model_norm"].apply(_is_baseline_model_value)
    if "min_skill_rmse" in grouped.columns and grouped["min_skill_rmse"].notna().any():
        grouped["min_skill_rmse"] = pd.to_numeric(grouped["min_skill_rmse"], errors="coerce")
    else:
        # Fallback: recompute from RMSE with rank-then-label baseline lookup.
        _bl_mask_df = df["model_norm"].apply(_is_baseline_model_value)
        _bl_rmse_by_rank = (
            df[_bl_mask_df].groupby("subset_rank")["rmse"].min().rename("_best_baseline_rmse")
        )
        _grouped_skill = grouped.join(_bl_rmse_by_rank, on="subset_rank")
        if "subset_label" in df.columns:
            _shap_pfx = "shap_"
            _bl_rmse_by_label = (
                df[_bl_mask_df]
                .groupby(df.loc[_bl_mask_df, "subset_label"].astype(str))["rmse"]
                .min()
            )
            _rank_to_label = (
                df[["subset_rank", "subset_label"]]
                .drop_duplicates("subset_rank")
                .set_index("subset_rank")["subset_label"]
                .astype(str)
                .str.removeprefix(_shap_pfx)
            )
            _missing = _grouped_skill["_best_baseline_rmse"].isna()
            _grouped_skill.loc[_missing, "_best_baseline_rmse"] = (
                _grouped_skill.loc[_missing, "subset_rank"].map(_rank_to_label).map(_bl_rmse_by_label)
            )
        grouped["min_skill_rmse"] = np.where(
            (~_baseline_mask) & (_grouped_skill["_best_baseline_rmse"] > 0),
            (_grouped_skill["_best_baseline_rmse"] - _grouped_skill["rmse"]) / _grouped_skill["_best_baseline_rmse"],
            np.nan,
        )
    skill_pivot = grouped.pivot(index="subset_rank", columns="model_norm", values="min_skill_rmse")

    def _skill_for_rank_model(rank: int, model: str) -> float:
        if rank in skill_pivot.index and model in skill_pivot.columns:
            val = pd.to_numeric(skill_pivot.loc[rank, model], errors="coerce")
            return float(val) if np.isfinite(val) else float("-inf")
        return float("-inf")

    def _r2_for_rank_model(rank: int, model: str) -> float:
        if rank in r2_pivot.index and model in r2_pivot.columns:
            val = pd.to_numeric(r2_pivot.loc[rank, model], errors="coerce")
            return float(val) if np.isfinite(val) else float("inf")
        return float("inf")

    def _rmse_for_rank_model(rank: int, model: str) -> float:
        if rank in rmse_pivot.index and model in rmse_pivot.columns:
            val = pd.to_numeric(rmse_pivot.loc[rank, model], errors="coerce")
            return float(val) if np.isfinite(val) else float("inf")
        return float("inf")

    # Select best model by highest max skill across subsets; fall back to r2 if skill unavailable.
    finite_max_skill = {}
    for model in plot_order:
        if model not in skill_pivot.columns or _is_baseline_model_value(model):
            continue
        vals = pd.to_numeric(skill_pivot[model], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            finite_max_skill[model] = float(np.max(vals))

    if finite_max_skill:
        best_model = max(finite_max_skill.items(), key=lambda item: item[1])[0]
    else:
        best_model = max(finite_max_r2.items(), key=lambda item: item[1])[0]

    # Separate baseline and non-baseline model lists.
    non_baseline_models = [m for m in plot_order if not _is_baseline_model_value(m)]
    baseline_models = [m for m in plot_order if _is_baseline_model_value(m)]

    # Sort feature clusters by highest skill across ALL non-baseline models,
    # then by highest R² (both descending).
    def _best_skill_in_cluster(rank: int) -> float:
        best = float("-inf")
        for model in non_baseline_models:
            s = _skill_for_rank_model(rank, model)
            if np.isfinite(s) and s > best:
                best = s
        return best

    def _best_r2_in_cluster(rank: int) -> float:
        best = float("-inf")
        for model in non_baseline_models:
            v = _r2_for_rank_model(rank, model)
            if np.isfinite(v) and v > best:
                best = v
        return best

    subset_order = sorted(
        rank_to_label.keys(),
        key=lambda rank: (
            -_best_skill_in_cluster(rank),
            -_best_r2_in_cluster(rank),
        ),
    )

    # For each subset cluster, order non-baseline model bars by descending local min-skill.
    model_order_index = {model: idx for idx, model in enumerate(plot_order)}

    models_by_rank: dict[int, list[str]] = {}
    for rank in subset_order:
        def _sort_key(model: str, _rank: int = rank) -> tuple:
            skill = _skill_for_rank_model(int(_rank), model)
            has_skill = 0 if np.isfinite(skill) else 1
            r2_val = _r2_for_rank_model(int(_rank), model)
            neg_r2 = -r2_val if np.isfinite(r2_val) else float("inf")
            return (
                has_skill,
                -skill if np.isfinite(skill) else float("inf"),
                neg_r2,
                model_order_index[model],
            )
        models_by_rank[int(rank)] = sorted(non_baseline_models, key=_sort_key)

    # Append a single "Baselines" cluster (baseline metrics are feature-independent).
    _BASELINES_LABEL = "Baselines"
    cluster_labels = [rank_to_label[r] for r in subset_order] + [_BASELINES_LABEL]

    # Sort baseline bars within their cluster using the same skill/R² key (rank arbitrary, use first).
    _any_rank = subset_order[0] if subset_order else 0
    baseline_bar_order = sorted(baseline_models, key=lambda m: (
        0 if np.isfinite(_skill_for_rank_model(_any_rank, m)) else 1,
        -_skill_for_rank_model(_any_rank, m) if np.isfinite(_skill_for_rank_model(_any_rank, m)) else float("inf"),
        -_r2_for_rank_model(_any_rank, m) if np.isfinite(_r2_for_rank_model(_any_rank, m)) else float("inf"),
        model_order_index[m],
    ))

    n_clusters = len(cluster_labels)
    x = np.arange(n_clusters, dtype=float)
    n_bars_max = max(len(non_baseline_models), len(baseline_models))
    cluster_width = 0.86
    bar_w = cluster_width / max(1, n_bars_max)

    fig, axes = plt.subplots(5, 1, figsize=(max(10, 0.85 * n_clusters + 6), 17), sharex=True, constrained_layout=False)
    metric_specs = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("pearson_r", "Pearson's r"),
        ("r2", "R\N{SUPERSCRIPT TWO}"),
        ("min_skill_rmse", "Min Skill (RMSE)"),
    ]

    # Pre-compute average baseline metric values (feature-independent).
    _baseline_rows = grouped[grouped["model_norm"].isin(baseline_models)]
    _baseline_avg = _baseline_rows.groupby("model_norm")[metric_cols + ["min_skill_rmse"]].mean(numeric_only=True)

    legend_handles_by_model: dict[str, object] = {}
    legend_labels_by_model: dict[str, str] = {}
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        metric_pivot = grouped.pivot(index="subset_rank", columns="model_norm", values=metric)
        axis_vals: list[float] = []
        axis_bars_and_vals: list[tuple[object, float]] = []

        for j, rank in enumerate(subset_order):
            # Feature clusters: non-baseline models only.
            ordered_models = models_by_rank.get(int(rank), non_baseline_models)
            for i, model in enumerate(ordered_models):
                style = plot_style[model]
                if rank in metric_pivot.index and model in metric_pivot.columns:
                    raw_val = metric_pivot.loc[rank, model]
                    val = float(raw_val) if np.isfinite(raw_val) else float("nan")
                else:
                    val = float("nan")

                xpos = float(x[j]) - (cluster_width / 2.0) + (i + 0.5) * bar_w
                bars = ax.bar(
                    [xpos],
                    [val],
                    width=bar_w,
                    color=style["color"],
                    hatch=style["hatch"],
                    edgecolor="black" if style["hatch"] else "none",
                    linewidth=0.8 if style["hatch"] else 0.0,
                )
                bar = bars[0]
                axis_bars_and_vals.append((bar, val))
                if np.isfinite(val):
                    axis_vals.append(float(val))
                if model not in legend_handles_by_model:
                    legend_handles_by_model[model] = bar
                    legend_labels_by_model[model] = style["label"]

        # Baselines cluster (last position).
        j_bl = len(subset_order)
        for i, model in enumerate(baseline_bar_order):
            style = plot_style[model]
            if model in _baseline_avg.index and metric in _baseline_avg.columns:
                raw_val = _baseline_avg.loc[model, metric]
                val = float(raw_val) if np.isfinite(raw_val) else float("nan")
            else:
                val = float("nan")

            xpos = float(x[j_bl]) - (cluster_width / 2.0) + (i + 0.5) * bar_w
            bars = ax.bar(
                [xpos],
                [val],
                width=bar_w,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="black" if style["hatch"] else "none",
                linewidth=0.8 if style["hatch"] else 0.0,
            )
            bar = bars[0]
            axis_bars_and_vals.append((bar, val))
            if np.isfinite(val):
                axis_vals.append(float(val))
            if model not in legend_handles_by_model:
                legend_handles_by_model[model] = bar
                legend_labels_by_model[model] = style["label"]

        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

        metric_vals = pd.to_numeric(grouped[metric], errors="coerce").to_numpy(dtype=float)
        finite_vals = metric_vals[np.isfinite(metric_vals)]
        if metric in ("mae", "rmse"):
            # Error metrics remain zero-anchored for comparability across subsets.
            ymax = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
            if ymax <= 0.0:
                ymax = 1.0
            ax.set_ylim(0.0, ymax * 1.08)
        elif metric in ("pearson_r", "r2"):
            # Keep correlation-style panels on a fixed bounded scale.
            ax.set_ylim(-1.0, 1.0)
        elif metric == "min_skill_rmse":
            # Keep the skill panel on a fixed bounded scale and pin labels inside the axes.
            ax.set_ylim(-1.0, 1.0)
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        else:
            ymin_base = float(np.min(finite_vals)) if finite_vals.size > 0 else -1.0
            ymax_base = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
            ax.set_ylim(ymin_base, ymax_base)

        if not axis_vals:
            continue

        y_low, y_high = ax.get_ylim()
        y_span = float(y_high - y_low) if np.isfinite(y_high - y_low) and (y_high - y_low) > 0 else 1.0
        text_pad = 0.02 * y_span
        edge_band = 0.15 * y_span

        for bar, val in axis_bars_and_vals:
            if not np.isfinite(val):
                continue
            f_val = float(val)
            y_pref = f_val + text_pad if f_val >= 0 else f_val - text_pad
            y = min(max(y_pref, y_low + text_pad), y_high - text_pad)
            va = "bottom" if f_val >= 0 else "top"
            # Keep clamped edge labels inside the panel while preserving readability.
            if y >= (y_high - edge_band):
                va = "top"
            elif y <= (y_low + edge_band):
                va = "bottom"
            ax.text(
                bar.get_x() + (bar.get_width() / 2.0),
                y,
                _format_bar_label(f_val),
                ha="center",
                va=va,
                fontsize=7,
                rotation=90,
                clip_on=True,
            )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [lbl.replace("/", "/\n") for lbl in cluster_labels],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axes[-1].set_xlabel("Candidate Feature Subset")
    # Tighten horizontal margins to less than one cluster width.
    h_margin = cluster_width * 0.6
    for ax in axes:
        ax.set_xlim(x[0] - h_margin, x[-1] + h_margin)

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    fig.subplots_adjust(top=0.89, hspace=0.16)
    legend_handles = [legend_handles_by_model[m] for m in plot_order if m in legend_handles_by_model]
    legend_labels = [legend_labels_by_model[m] for m in plot_order if m in legend_labels_by_model]
    legend_above(fig, legend_handles, legend_labels, fontsize=9)
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _feature_plot_filename(stem: str, row_count: int, include_row_count_in_name: bool) -> str:
    """Return standard feature-plot filename with optional row-count disambiguation."""
    if include_row_count_in_name:
        return f"{stem}_r{row_count:03d}.png"
    return f"{stem}.png"


def _plot_feature_importance_bar(
    feature_sensitivities: dict[str, tuple[float, int]],
    dataset_name: str,
    target_name: str,
    row_count: int,
    output_dir: Path,
    include_row_count_in_name: bool = False,
) -> Path:
    """Plot feature importance (removal sensitivity) as horizontal bar chart."""
    ranked = sorted(feature_sensitivities.items(), key=lambda x: -x[1][0])
    features = [f for f, _ in ranked]
    scores = [s[0] for _, s in ranked]

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)), constrained_layout=True)
    if scores:
        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        if score_max > score_min:
            norm = matplotlib.colors.Normalize(vmin=score_min, vmax=score_max)
            colors = [plt.cm.RdYlGn(float(norm(s))) for s in scores]
        else:
            colors = [plt.cm.RdYlGn(0.5) for _ in scores]
    else:
        colors = []
    ax.barh(features, scores, color=colors)
    ax.set_xlabel("Removal Sensitivity (avg delta)")
    ax.grid(axis='x', alpha=0.3)

    plot_path = output_dir / _feature_plot_filename(
        "feature_importance_bar",
        row_count,
        include_row_count_in_name,
    )
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def _plot_removal_sensitivity(
    feature_removal_deltas: dict[str, list[float]],
    dataset_name: str,
    target_name: str,
    row_count: int,
    output_dir: Path,
    include_row_count_in_name: bool = False,
) -> Path:
    """Plot removal sensitivity as box plot showing distribution of objective deltas."""
    features = sorted(feature_removal_deltas.keys())
    tested_pairs = [(f, feature_removal_deltas[f]) for f in features if len(feature_removal_deltas[f]) > 0]

    # Sort by median delta (most valuable-to-keep at top in the horizontal plot)
    tested_pairs.sort(key=lambda item: float(np.median(item[1])), reverse=True)

    tested_features = [item[0] for item in tested_pairs]
    tested_deltas = [item[1] for item in tested_pairs]

    if not tested_features:
        return Path()  # No data to plot

    median_deltas = [float(np.median(deltas)) for deltas in tested_deltas]
    med_min = float(np.min(median_deltas))
    med_max = float(np.max(median_deltas))
    if med_max > med_min:
        if med_min < 0.0 < med_max:
            color_norm = matplotlib.colors.TwoSlopeNorm(vmin=med_min, vcenter=0.0, vmax=med_max)
        else:
            color_norm = matplotlib.colors.Normalize(vmin=med_min, vmax=med_max)
    else:
        color_norm = None

    fig_h = max(7, len(tested_features) * 0.45)
    fig, ax = plt.subplots(figsize=(14, fig_h), constrained_layout=True)
    bp = ax.boxplot(tested_deltas, vert=False, patch_artist=True)

    # Continuous shading by median removal delta (red=low, green=high).
    for patch, median_delta in zip(bp['boxes'], median_deltas):
        if color_norm is None:
            patch.set_facecolor(plt.cm.RdYlGn(0.5))
        else:
            patch.set_facecolor(plt.cm.RdYlGn(float(color_norm(median_delta))))

    ax.set_yticks(np.arange(1, len(tested_features) + 1))
    ax.set_yticklabels(tested_features, fontsize=8)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Objective Delta (removing feature)")
    ax.set_ylabel("Feature")
    ax.grid(axis='x', alpha=0.3)

    plot_path = output_dir / _feature_plot_filename(
        "removal_sensitivity_box",
        row_count,
        include_row_count_in_name,
    )
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path

def _plot_feature_frequency(
    feature_improvement_counts: dict[str, int],
    feature_sensitivities: dict[str, tuple[float, int]],
    dataset_name: str,
    target_name: str,
    row_count: int,
    output_dir: Path,
    include_row_count_in_name: bool = False,
) -> Path:
    """Plot feature frequency in improving solutions."""
    ranked = sorted(feature_sensitivities.items(), key=lambda x: -x[1][0])
    features = [f for f, _ in ranked]
    frequencies = [feature_improvement_counts[f] for f in features]

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)), constrained_layout=True)
    if frequencies:
        freq_min = float(np.min(frequencies))
        freq_max = float(np.max(frequencies))
        if freq_max > freq_min:
            norm = matplotlib.colors.Normalize(vmin=freq_min, vmax=freq_max)
            colors = [plt.cm.Greens(0.25 + (0.7 * float(norm(v)))) for v in frequencies]
        else:
            colors = [plt.cm.Greens(0.6) for _ in frequencies]
    else:
        colors = []
    bars = ax.barh(features, frequencies, color=colors)

    # Add value labels on bars
    for bar, freq in zip(bars, frequencies):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(freq)}', 
                ha='left', va='center', fontsize=9)

    ax.set_xlabel("Frequency in Improving Solutions")
    ax.grid(axis='x', alpha=0.3)

    plot_path = output_dir / _feature_plot_filename(
        "feature_frequency",
        row_count,
        include_row_count_in_name,
    )
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def _write_feature_stats_artifacts(
    dataset_dir: Path,
    row_count: int,
    feature_sensitivities: dict[str, tuple[float, int]],
    feature_removal_deltas: dict[str, list[float]],
    feature_improvement_counts: dict[str, int],
) -> tuple[Path, Path]:
    out_dir = _forecast_sweeps_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    for feature in sorted(feature_sensitivities.keys()):
        avg_delta, _ = feature_sensitivities.get(feature, (0.0, 0))
        deltas = feature_removal_deltas.get(feature, [])
        median_delta = float(np.median(deltas)) if deltas else np.nan
        stats_rows.append(
            {
                "feature": feature,
                "avg_removal_delta": float(avg_delta),
                "median_removal_delta": median_delta,
                "n_removal_tests": int(len(deltas)),
                "improvement_count": int(feature_improvement_counts.get(feature, 0)),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df = stats_df.sort_values(["avg_removal_delta", "feature"], ascending=[False, True], kind="stable")

    stats_csv = out_dir / f"feature_importance_stats_r{row_count:03d}.csv"
    stats_df.to_csv(stats_csv, index=False)

    delta_rows = []
    for feature in sorted(feature_removal_deltas.keys()):
        for delta in feature_removal_deltas.get(feature, []):
            delta_rows.append(
                {
                    "feature": feature,
                    "delta": float(delta),
                }
            )

    deltas_df = pd.DataFrame(delta_rows, columns=["feature", "delta"])
    deltas_csv = out_dir / f"feature_removal_deltas_r{row_count:03d}.csv"
    deltas_df.to_csv(deltas_csv, index=False)
    return stats_csv, deltas_csv


def _load_feature_stats_artifacts(
    dataset_dir: Path,
    row_count: int,
) -> tuple[dict[str, tuple[float, int]], dict[str, int], dict[str, list[float]]]:
    feature_sensitivities, feature_improvement_counts, feature_removal_deltas, _ = _load_feature_stats_artifacts_with_source(
        dataset_dir=dataset_dir,
        row_count=row_count,
    )
    return feature_sensitivities, feature_improvement_counts, feature_removal_deltas


def _load_feature_stats_artifacts_with_source(
    dataset_dir: Path,
    row_count: int,
) -> tuple[dict[str, tuple[float, int]], dict[str, int], dict[str, list[float]], str]:
    """Load feature-importance artifacts and report their provenance.

    Returns source in {"native_removal_delta", "shapley_marginal_samples", "shapley_score_estimate", "missing"}.
    """
    native_sens, native_counts, native_deltas = _load_native_feature_stats_artifacts(
        dataset_dir=dataset_dir,
        row_count=row_count,
    )
    if native_sens:
        return native_sens, native_counts, native_deltas, "native_removal_delta"

    shapley_sens, shapley_counts, shapley_deltas, shapley_source = _load_shapley_feature_stats_artifacts(
        dataset_dir=dataset_dir,
        row_count=row_count,
    )
    if shapley_sens:
        return shapley_sens, shapley_counts, shapley_deltas, shapley_source
    return {}, {}, {}, "missing"


def _load_native_feature_stats_artifacts(
    dataset_dir: Path,
    row_count: int,
) -> tuple[dict[str, tuple[float, int]], dict[str, int], dict[str, list[float]]]:
    out_dir = _forecast_sweeps_dir(dataset_dir)
    stats_csv = out_dir / f"feature_importance_stats_r{row_count:03d}.csv"
    deltas_csv = out_dir / f"feature_removal_deltas_r{row_count:03d}.csv"

    if not stats_csv.exists() or not deltas_csv.exists():
        return {}, {}, {}

    stats_df = pd.read_csv(stats_csv)
    feature_sensitivities: dict[str, tuple[float, int]] = {}
    feature_improvement_counts: dict[str, int] = {}

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
    feature_removal_deltas: dict[str, list[float]] = {feature: [] for feature in feature_sensitivities.keys()}
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


def _load_shapley_feature_stats_artifacts(
    dataset_dir: Path,
    row_count: int,
) -> tuple[dict[str, tuple[float, int]], dict[str, int], dict[str, list[float]], str]:
    """Adapt Shapley outputs to the feature-sensitivity contract used by postprocess plots."""
    out_dir = _forecast_sweeps_dir(dataset_dir)
    samples_json = out_dir / f"feature_shapley_samples_r{row_count:03d}.json"
    shapley_csv = out_dir / f"feature_shapley_scores_r{row_count:03d}.csv"

    feature_sensitivities: dict[str, tuple[float, int]] = {}
    feature_improvement_counts: dict[str, int] = {}
    feature_removal_deltas: dict[str, list[float]] = {}
    source = "missing"

    if samples_json.exists():
        try:
            with open(samples_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            features_payload = payload.get("features", {}) if isinstance(payload, dict) else {}
            if isinstance(features_payload, dict):
                for feature, raw in features_payload.items():
                    if not str(feature).strip():
                        continue
                    values = []
                    if isinstance(raw, dict):
                        values = raw.get("values", [])
                    finite_vals: list[float] = []
                    for val in list(values) if isinstance(values, list) else []:
                        num = pd.to_numeric(val, errors="coerce")
                        if np.isfinite(num):
                            finite_vals.append(float(num))
                    if not finite_vals:
                        continue
                    avg_delta = float(np.mean(finite_vals))
                    improvement_count = int(np.sum(np.asarray(finite_vals, dtype=float) > 0.0))
                    key = str(feature)
                    feature_sensitivities[key] = (avg_delta, improvement_count)
                    feature_improvement_counts[key] = improvement_count
                    feature_removal_deltas[key] = finite_vals
                if feature_sensitivities:
                    source = "shapley_marginal_samples"
        except Exception:
            # Keep going and try CSV fallback.
            pass

    if shapley_csv.exists():
        try:
            df = pd.read_csv(shapley_csv)
            for _, row in df.iterrows():
                feature = str(row.get("feature", "")).strip()
                if not feature:
                    continue
                est = float(pd.to_numeric(row.get("shapley_value_est", np.nan), errors="coerce"))
                if not np.isfinite(est):
                    continue
                n_samples = int(pd.to_numeric(row.get("n_marginal_samples", 0), errors="coerce"))
                n_samples = max(1, n_samples)
                if feature not in feature_sensitivities:
                    synthetic = [float(est)] * n_samples
                    improvement_count = n_samples if est > 0 else 0
                    feature_sensitivities[feature] = (float(est), int(improvement_count))
                    feature_improvement_counts[feature] = int(improvement_count)
                    feature_removal_deltas[feature] = synthetic
                elif not feature_removal_deltas.get(feature):
                    # Backfill sparse cases where JSON exists but this feature has no finite samples.
                    synthetic = [float(est)] * n_samples
                    feature_removal_deltas[feature] = synthetic
                    if feature not in feature_improvement_counts:
                        feature_improvement_counts[feature] = n_samples if est > 0 else 0
                if source == "missing":
                    source = "shapley_score_estimate"
        except Exception:
            pass

    return feature_sensitivities, feature_improvement_counts, feature_removal_deltas, source


def _available_row_counts_for_postprocess(dataset_dir: Path) -> list[int]:
    out_dir = _forecast_sweeps_dir(dataset_dir)
    patterns = [
        "feature_importance_stats_r*.csv",
        "feature_search_trace_r*.csv",
        "feature_selected_subsets_r*.csv",
    ]
    row_counts: set[int] = set()
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            match = re.search(r"_r(\d{3})\.csv$", path.name)
            if match:
                row_counts.add(int(match.group(1)))
    return sorted(row_counts)


def _regenerate_saved_outputs_for_row(
    dataset_dir: Path,
    target_name: str,
    row_count: int,
    keep_search_plots: bool,
    include_row_count_in_plot_names: bool = False,
) -> dict[str, Path]:
    out_dir = _forecast_sweeps_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    _ = keep_search_plots  # Legacy compatibility: search Pareto plots are always regenerated when possible.
    trace_csv = out_dir / f"feature_search_trace_r{row_count:03d}.csv"
    if trace_csv.exists():
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

    bar_plot = _plot_feature_importance_bar(
        feature_sensitivities,
        dataset_dir.name,
        target_name,
        row_count,
        out_dir,
        include_row_count_in_name=include_row_count_in_plot_names,
    )
    written["bar_plot"] = bar_plot

    sensitivity_plot = _plot_removal_sensitivity(
        feature_removal_deltas,
        dataset_dir.name,
        target_name,
        row_count,
        out_dir,
        include_row_count_in_name=include_row_count_in_plot_names,
    )
    if sensitivity_plot.exists():
        written["sensitivity_plot"] = sensitivity_plot

    frequency_plot = _plot_feature_frequency(
        feature_improvement_counts,
        feature_sensitivities,
        dataset_dir.name,
        target_name,
        row_count,
        out_dir,
        include_row_count_in_name=include_row_count_in_plot_names,
    )
    written["frequency_plot"] = frequency_plot
    return written


def _beam_search_subsets(
    dataset_dir: Path,
    dataset_prefix: str,
    surrogate_config_path: Path,
    row_count: int,
    lambda_drop: float,
    beam_width: int,
    max_rounds: int,
    no_improve_patience: int,
    min_features: int,
    eval_budget: int,
    max_swap_attempts: int,
    disable_baselines_for_search: bool,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
    seed: int,
    save_search_plots: bool,
    parallel_evaluators: int = 1,
    include_row_count_in_plot_names: bool = False,
    seeded_subsets: list[tuple[str, ...]] | None = None,
    cv_folds: int = 0,
    cv_min_train_fraction: float = 0.5,
    selection_tolerance_se: float = 1.0,
    retention_tolerance: float = 0.02,
    cv_fold_dirs: list[Path] | None = None,
) -> tuple[list[CandidateResult], list[CandidateResult], dict[str, tuple[float, int]]]:
    """Run beam+swap feature-subset search for one dataset and one row-count.

    Args:
        dataset_dir: Dataset directory containing samples/config-derived forecast outputs.
        dataset_prefix: Prefix used to derive human-readable target name.
        surrogate_config_path: Base train config used as the surrogate evaluator.
        row_count: Input lookback rows to evaluate.
        lambda_drop: Drop-rate penalty weight in objective.
        beam_width: Number of best candidates kept each elimination round.
        max_rounds: Maximum elimination rounds before swap refinement.
        no_improve_patience: Consecutive non-improving rounds allowed.
        min_features: Minimum subset size allowed during elimination.
        eval_budget: Maximum candidate evaluations for this search run.
        max_swap_attempts: Maximum swap refinements attempted.
        disable_baselines_for_search: If True, disable baseline evaluation during search.
        disable_training_plots: If True, suppress training plots during candidate training.
        disable_eval_plots: If True, suppress evaluation plots during candidate evaluation.
        suppress_training_logs: If True, hide verbose model training logs.
        seed: Random seed for candidate ordering and swap sampling.
        save_search_plots: If True, write search summary plots (Pareto/feature charts).
            When False, search still writes feature stats CSV artifacts needed downstream.
        include_row_count_in_plot_names: If True, append `_r###` to saved search-plot names.
        seeded_subsets: Optional explicit seed feature subsets to evaluate before beam rounds.

    Returns:
        `(top_sorted, trace, feature_sensitivities)` where:
        - `top_sorted`: All evaluated candidates sorted by objective.
        - `trace`: Chronological candidate evaluation trace.
        - `feature_sensitivities`: Per-feature `(avg_removal_delta, improvement_count)`.

    Example:
        `top_sorted, trace, sens = _beam_search_subsets(..., save_search_plots=False)`
        to keep search fast while retaining CSV artifacts for postprocess.
    """
    target_name = _derive_target_name(dataset_dir.name, dataset_prefix)
    tmp_cfg_dir = _forecast_sweeps_dir(dataset_dir) / "configs"

    base_cfg = train_module.load_config(str(surrogate_config_path))
    full_features = tuple(base_cfg["data"]["input_columns"])
    if len(full_features) <= min_features:
        raise ValueError(f"min_features={min_features} must be < number of features ({len(full_features)})")

    if int(cv_folds) > 0 and cv_fold_dirs is None:
        training_segments = _training_portion_segments(
            dataset_dir=dataset_dir,
            surrogate_config_path=surrogate_config_path,
            row_count=row_count,
            full_features=full_features,
            tmp_cfg_dir=tmp_cfg_dir,
        )
        cv_fold_dirs = _materialize_cv_folds(
            dataset_dir=dataset_dir,
            sample_subdir=str(base_cfg["data"].get("sample_subdir", "samples")),
            row_count=row_count,
            n_folds=int(cv_folds),
            min_train_fraction=float(cv_min_train_fraction),
            eligible_segments=training_segments,
        )
    elif int(cv_folds) <= 0:
        print(
            "[SEARCH] Rolling-origin CV disabled (--cv-folds 0): candidates are scored on "
            "the same 70/30 holdout the results table reports, so the search sees the test "
            "segments and the reported accuracy is an optimistically biased upper estimate."
        )

    rng = np.random.default_rng(seed)
    parallel_workers = max(1, int(parallel_evaluators))
    if parallel_workers > 1:
        print(f"[SEARCH] Parallel evaluators enabled: {parallel_workers}")
    cache: dict[tuple[int, tuple[str, ...]], CandidateResult] = {}
    trace: list[CandidateResult] = []
    eval_count = 0
    effective_eval_budget = int(eval_budget)
    search_start_time = time.time()
    
    # Feature importance tracking
    feature_removal_deltas: dict[str, list[float]] = {feat: [] for feat in full_features}
    feature_improvement_counts: dict[str, int] = {feat: 0 for feat in full_features}

    # Prepare seeded subsets up front so their evaluations can be guaranteed.
    seeded_prepared: list[tuple[tuple[str, ...], int]] = []
    seeded_seen: set[tuple[str, ...]] = set()
    seeded_loaded_count = len(seeded_subsets or [])
    seeded_skipped_too_small = 0
    seeded_skipped_empty = 0
    seeded_skipped_duplicate = 0
    seeded_evaluated_count = 0
    if seeded_subsets:
        for in_rank, raw_feats in enumerate(seeded_subsets, start=1):
            filtered = [f for f in raw_feats if f in full_features]
            deduped = list(dict.fromkeys(filtered))
            if not deduped:
                seeded_skipped_empty += 1
                continue
            if len(deduped) < min_features:
                seeded_skipped_too_small += 1
                continue
            ordered = tuple(sorted(deduped, key=lambda s: full_features.index(s)))
            if ordered in seeded_seen:
                seeded_skipped_duplicate += 1
                continue
            seeded_seen.add(ordered)
            seeded_prepared.append((ordered, in_rank))

    # Guarantee room for full-feature anchor + all valid seeded subsets.
    guaranteed_seed_budget = 1 + len(seeded_prepared)
    if seeded_prepared and effective_eval_budget < guaranteed_seed_budget:
        print(
            f"[SEARCH] Expanding eval budget from {effective_eval_budget} to {guaranteed_seed_budget} "
            "to guarantee seeded subset inclusion."
        )
        effective_eval_budget = guaranteed_seed_budget

    def _eval(features: tuple[str, ...]) -> CandidateResult | None:
        nonlocal eval_count
        key = _candidate_key(row_count, features)
        if key in cache:
            return cache[key]
        if eval_count >= effective_eval_budget:
            return None

        tag = _feature_tag(features)
        result = _evaluate_candidate(
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
            cv_fold_dirs=cv_fold_dirs,
        )
        cache[key] = result
        if result is not None:
            trace.append(result)
        eval_count += 1
        return result

    first = _eval(full_features)
    if first is None:
        raise RuntimeError("Search budget exhausted before evaluating initial subset.")
    first.source = "search"

    beam: list[CandidateResult] = [first]
    best = first
    if seeded_prepared:
        seeded_scored: list[CandidateResult] = []
        for ordered, in_rank in seeded_prepared:
            out = _eval(ordered)
            if out is not None:
                out.source = "shapley_seed"
                out.seeded_input_rank = int(in_rank)
                seeded_scored.append(out)
                seeded_evaluated_count += 1
        if seeded_scored:
            seeded_scored.sort(key=_candidate_rank_key)
            beam = sorted([first] + seeded_scored, key=_candidate_rank_key)[:beam_width]
            best = beam[0]
            print(f"[SEARCH] Seeded initialization: {len(seeded_scored)} subset(s) evaluated; best objective={best.objective:.4f} r2={best.r2:.6f} n_features={best.n_features}")
    if seeded_loaded_count > 0:
        print(
            "[SEARCH] Seed summary: "
            f"loaded={seeded_loaded_count}, valid_unique={len(seeded_prepared)}, "
            f"evaluated={seeded_evaluated_count}, "
            f"skipped_empty={seeded_skipped_empty}, skipped_too_small={seeded_skipped_too_small}, "
            f"skipped_duplicate={seeded_skipped_duplicate}"
        )

    no_improve = 0
    print(f"[SEARCH] Initial (all {len(full_features)} features): objective={best.objective:.4f} r2={best.r2:.6f} (evals: {eval_count}/{effective_eval_budget}, ETA: {_format_eta(search_start_time, eval_count, effective_eval_budget)})")

    for _round in range(max_rounds):
        candidates: list[tuple[str, ...]] = []
        seen = set()

        for item in beam:
            feat_list = list(item.features)
            if len(feat_list) <= min_features:
                continue
            for idx in range(len(feat_list)):
                child = tuple(feat_list[:idx] + feat_list[idx + 1 :])
                if len(child) < min_features:
                    continue
                key = _candidate_key(row_count, child)
                if key in cache:
                    continue
                if child in seen:
                    continue
                seen.add(child)
                candidates.append(child)

        if not candidates:
            print(f"[SEARCH] Round {_round + 1}: no new candidates, stopping.")
            break

        rng.shuffle(candidates)
        scored: list[CandidateResult] = []
        remaining_budget = max(0, int(effective_eval_budget - eval_count))
        batch = list(candidates[:remaining_budget])

        if parallel_workers <= 1 or len(batch) <= 1:
            for child in batch:
                out = _eval(child)
                if out is None:
                    print(f"[SEARCH] Round {_round + 1}: eval budget exhausted after {eval_count} evals.")
                    break
                scored.append(out)

                # Track removal sensitivity: which feature was removed from beam members to create this child?
                # Find parent by checking which single feature difference exists
                for parent_item in beam:
                    parent_set = set(parent_item.features)
                    child_set = set(child)
                    if len(parent_set - child_set) == 1:  # exactly one feature removed
                        removed_feat = list(parent_set - child_set)[0]
                        delta = out.objective - parent_item.objective
                        feature_removal_deltas[removed_feat].append(delta)
                        break
        else:
            payloads = []
            payload_features = []
            for child in batch:
                payload_features.append(child)
                payloads.append(
                    {
                        "dataset_dir": str(dataset_dir),
                        "target_name": target_name,
                        "surrogate_config_path": str(surrogate_config_path),
                        "row_count": int(row_count),
                        "features": list(child),
                        "feature_tag": _feature_tag(child),
                        "lambda_drop": float(lambda_drop),
                        "tmp_cfg_dir": str(tmp_cfg_dir),
                        "disable_baselines_for_search": bool(disable_baselines_for_search),
                        "disable_training_plots": bool(disable_training_plots),
                        "disable_eval_plots": bool(disable_eval_plots),
                        "suppress_training_logs": bool(suppress_training_logs),
                        "n_seeds": int(_CANDIDATE_SEEDS),
                        "seed_base": int(_CANDIDATE_SEED_BASE),
                        "cv_fold_dirs": [str(d) for d in (cv_fold_dirs or [])],
                    }
                )

            with ProcessPoolExecutor(max_workers=parallel_workers) as pool:
                for child, out in zip(payload_features, pool.map(_evaluate_candidate_worker, payloads)):
                    key = _candidate_key(row_count, child)
                    cache[key] = out
                    if out is not None:
                        out.source = "search"
                        trace.append(out)
                    eval_count += 1
                    if out is None:
                        continue
                    scored.append(out)
                    for parent_item in beam:
                        parent_set = set(parent_item.features)
                        child_set = set(child)
                        if len(parent_set - child_set) == 1:
                            removed_feat = list(parent_set - child_set)[0]
                            delta = out.objective - parent_item.objective
                            feature_removal_deltas[removed_feat].append(delta)
                            break

        if not scored:
            print(f"[SEARCH] Round {_round + 1}: no scored candidates, stopping.")
            break

        scored.sort(key=_candidate_rank_key)
        beam = scored[:beam_width]
        if beam and _candidate_rank_key(beam[0]) < _candidate_rank_key(best):
            best = beam[0]
            no_improve = 0
            # Track features in improving solution (Option A)
            for feat in best.features:
                feature_improvement_counts[feat] += 1
            print(f"[SEARCH] Round {_round + 1}: improved! objective={best.objective:.4f} r2={best.r2:.6f} n_features={best.n_features} (evals: {eval_count}/{effective_eval_budget}, ETA: {_format_eta(search_start_time, eval_count, effective_eval_budget)})")
        else:
            no_improve += 1
            print(f"[SEARCH] Round {_round + 1}: no improvement ({no_improve}/{no_improve_patience}). Best: objective={best.objective:.4f} r2={best.r2:.6f} (evals: {eval_count}/{effective_eval_budget}, ETA: {_format_eta(search_start_time, eval_count, effective_eval_budget)})")
            if no_improve >= no_improve_patience:
                # This said "stopping" and then carried on: there was no break, so the
                # loop always ran to max_rounds or budget exhaustion and the knob did
                # nothing. Stopping here returns the unspent budget to swap refinement,
                # which at the default budget never got to run at all.
                print(f"[SEARCH] Patience exhausted after {_round + 1} rounds; stopping "
                      f"elimination with {effective_eval_budget - eval_count} eval(s) left "
                      f"for swap refinement.")
                break

    current = best
    all_features_set = set(full_features)
    attempts = 0
    improved = True
    swap_iter = 0
    print(f"[SEARCH] Starting swap refinement from: objective={current.objective:.4f} r2={current.r2:.6f} n_features={current.n_features} (ETA: {_format_eta(search_start_time, eval_count, effective_eval_budget)})")
    
    while improved and attempts < max_swap_attempts and eval_count < effective_eval_budget:
        improved = False
        included = list(current.features)
        excluded = list(all_features_set - set(included))
        if not excluded or len(included) <= min_features:
            break

        swap_pairs = [(drop_f, add_f) for drop_f in included for add_f in excluded]
        rng.shuffle(swap_pairs)

        for drop_f, add_f in swap_pairs:
            attempts += 1
            if attempts > max_swap_attempts:
                break

            new_features = [f for f in included if f != drop_f] + [add_f]
            new_features = tuple(sorted(new_features, key=lambda s: full_features.index(s)))
            out = _eval(new_features)
            if out is None:
                print(f"[SEARCH] Swap refinement: eval budget exhausted after {eval_count} evals.")
                break
            if _candidate_rank_key(out) < _candidate_rank_key(current):
                swap_iter += 1
                current = out
                best = out
                improved = True
                print(f"[SEARCH] Swap refinement #{swap_iter}: improved! objective={best.objective:.4f} r2={best.r2:.6f} n_features={best.n_features} (evals: {eval_count}/{effective_eval_budget}, ETA: {_format_eta(search_start_time, eval_count, effective_eval_budget)})")
                break
    
    if not improved and eval_count < effective_eval_budget:
        print(f"[SEARCH] Swap refinement: no improvements found (attempts: {attempts}/{max_swap_attempts}, evals: {eval_count}/{effective_eval_budget})")

    top_sorted = sorted(trace, key=_candidate_rank_key)

    # The argmin is where the search stopped, not necessarily what it established.
    # Prefer the smallest subset the objective cannot separate from it.
    chosen = _apply_one_se_rule(top_sorted, top_sorted[0] if top_sorted else best,
                                float(selection_tolerance_se))
    if top_sorted and chosen.feature_tag != top_sorted[0].feature_tag:
        best = chosen
        top_sorted = [chosen] + [c for c in top_sorted if c.feature_tag != chosen.feature_tag]

    total_elapsed = time.time() - search_start_time
    elapsed_min = int(total_elapsed // 60)
    elapsed_sec = int(total_elapsed % 60)
    avg_time_per_eval = total_elapsed / eval_count if eval_count > 0 else 0
    print(f"[SEARCH] Complete: {eval_count}/{effective_eval_budget} evaluations in {elapsed_min}m {elapsed_sec}s ({avg_time_per_eval:.1f}s/eval). Best: objective={best.objective:.4f} rmse={best.rmse:.6f} r2={best.r2:.6f} n_features={best.n_features}")
    
    # Compute average removal sensitivity for each feature
    feature_sensitivities: dict[str, tuple[float, int]] = {}  # (avg_removal_delta, frequency_count)
    for feat in full_features:
        deltas = feature_removal_deltas[feat]
        counts = feature_improvement_counts[feat]
        avg_delta = float(np.mean(deltas)) if deltas else 0.0
        feature_sensitivities[feat] = (avg_delta, counts)

    try:
        _write_selection_stability_artifacts(
            dataset_dir=dataset_dir,
            row_count=row_count,
            trace=trace,
            tolerance=float(retention_tolerance),
        )
    except Exception as exc:
        print(f"[WARN] Could not write selection-stability artifacts: {exc}")

    print(f"\n[SEARCH] Recommended feature subset (from search best):")
    print(f"  Features ({best.n_features}): {', '.join(best.features)}")
    print(f"  Objective: {best.objective:.6f} (rmse={best.rmse:.6f}, r2={best.r2:.6f}, drop_rate={best.drop_rate:.4f})")
    print(f"  (Full ranked importance written to feature_stats CSV)")
    
    # Generate feature stats always; plots are optional for search-speed defaults.
    print(f"\n[SEARCH] Writing feature-importance artifacts...")
    out_dir = _forecast_sweeps_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        stats_csv, deltas_csv = _write_feature_stats_artifacts(
            dataset_dir=dataset_dir,
            row_count=row_count,
            feature_sensitivities=feature_sensitivities,
            feature_removal_deltas=feature_removal_deltas,
            feature_improvement_counts=feature_improvement_counts,
        )
        print(f"[INFO] Wrote feature stats table: {stats_csv}")
        print(f"[INFO] Wrote feature deltas table: {deltas_csv}")

        if save_search_plots:
            bar_plot = _plot_feature_importance_bar(
                feature_sensitivities,
                dataset_dir.name,
                target_name,
                row_count,
                out_dir,
                include_row_count_in_name=include_row_count_in_plot_names,
            )
            print(f"[INFO] Wrote feature importance bar chart: {bar_plot}")

            sensitivity_plot = _plot_removal_sensitivity(
                feature_removal_deltas,
                dataset_dir.name,
                target_name,
                row_count,
                out_dir,
                include_row_count_in_name=include_row_count_in_plot_names,
            )
            if sensitivity_plot.exists():
                print(f"[INFO] Wrote removal sensitivity plot: {sensitivity_plot}")

            frequency_plot = _plot_feature_frequency(
                feature_improvement_counts,
                feature_sensitivities,
                dataset_dir.name,
                target_name,
                row_count,
                out_dir,
                include_row_count_in_name=include_row_count_in_plot_names,
            )
            print(f"[INFO] Wrote feature frequency plot: {frequency_plot}")
        else:
            print("[INFO] Search plots disabled by default (use --keep-search-plots to enable).")
    except Exception as e:
        print(f"[WARN] Failed to generate feature importance plots: {e}")
    
    return top_sorted, trace, feature_sensitivities


def _surrogate_candidates(train_configs: list[Path]) -> list[Path]:
    """Configs eligible to score the search, in the order they were discovered."""
    return [
        c for c in train_configs
        if not any(tok in c.name.lower() for tok in _SURROGATE_EXCLUDED_TOKENS)
    ]


def _select_surrogate_config(train_configs: list[Path], surrogate_model: str = "xgb") -> Path:
    """Pick the config whose name matches *surrogate_model*.

    `_choose_surrogate_config` picks by measurement instead of by name; this remains
    for the explicit case and for the XGBoost tuning cache, which is XGBoost-specific
    by construction.
    """
    token = str(surrogate_model).strip().lower()
    candidates = _surrogate_candidates(train_configs)
    if token.startswith("auto"):
        # `auto` and `auto:<prefix>` are resolved by measurement in
        # `_choose_surrogate_config`; this only supplies a stand-in for the callers
        # that need any configuration of the right shape (the row-count span, the
        # tuning order). Keep it inside the requested family so the stand-in is never
        # mistaken for the choice.
        prefix = token.split(":", 1)[1].strip() if ":" in token else ""
        pool = [c for c in candidates
                if c.stem.replace("config_", "").startswith(prefix)] if prefix else candidates
        return (pool or candidates or train_configs)[0]
    if token and token != "":
        matches = [cfg for cfg in candidates if token in cfg.name.lower()]
        if not matches:
            raise ValueError(
                f"No training config matches --surrogate-model '{surrogate_model}'. "
                f"Eligible: {', '.join(c.name for c in candidates)}. "
                "A dataset generated before the window representations were added "
                "carries only xgb_01: regenerate it with d_RunResample, or name one "
                "of the configurations listed above."
            )
        if len(matches) > 1:
            raise ValueError(
                f"--surrogate-model '{surrogate_model}' matches "
                f"{len(matches)} configs: {', '.join(c.name for c in matches)}. "
                "Name one exactly, or use 'auto' to choose by measurement. Taking the "
                "first match is what silently substituted one variant for another."
            )
        return matches[0]
    for cfg in candidates:
        if "xgb" in cfg.name.lower():
            return cfg
    return candidates[0] if candidates else train_configs[0]


def _choose_surrogate_config(
    dataset_dir: Path,
    dataset_prefix: str,
    train_configs: list[Path],
    row_count: int,
    surrogate_model: str,
    lambda_drop: float,
    cv_folds: int,
    cv_min_train_fraction: float,
    disable_baselines_for_search: bool,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
) -> tuple[Path, list[Path] | None]:
    """Choose the surrogate by measuring the families, not by matching a name.

    Selecting "the first config whose name contains xgb" fixes the scoring family
    before any evidence exists. That matters because the feature set the surrogate
    picks is then used by every family: on the profiler-free arm XGBoost scored the
    candidates while a Gaussian process won 7 of the 14 targets, so most reported
    subsets were chosen on a family that did not produce the reported result.

    Here each available family is fitted once on the full feature set, under exactly
    the objective the search will use, and the best one goes on to score the search.

    Args:
        surrogate_model: `auto` to measure; any other token forces that family.

    Returns:
        `(config_path, cv_fold_dirs)`. The fold list belongs to the chosen family's
        sample subdirectory and is reused by the search, so folds are built once.

    Raises:
        RuntimeError: When no family could be evaluated, rather than falling back to a
            name match that the measurement was meant to replace.
    """
    token = str(surrogate_model).strip().lower()
    candidates = _surrogate_candidates(train_configs)
    if not candidates:
        raise ValueError(f"No usable training configs in {dataset_dir}")

    # `auto:<prefix>` measures only one family. Comparing every configuration would
    # spend fits ranking families against each other, which is not the question: the
    # surrogate exists to separate feature subsets, and the evidence that the best
    # window representation depends on window length is what this is here to settle.
    auto_prefix = ""
    if token.startswith("auto:"):
        auto_prefix = token.split(":", 1)[1].strip()
        candidates = [c for c in candidates
                      if c.stem.replace("config_", "").startswith(auto_prefix)]
        if not candidates:
            raise ValueError(
                f"--surrogate-model '{surrogate_model}' matches no configuration. "
                f"Eligible: {', '.join(c.stem.replace('config_', '') for c in _surrogate_candidates(train_configs))}"
            )

    tmp_cfg_dir = _forecast_sweeps_dir(dataset_dir) / "configs"
    target_name = _derive_target_name(dataset_dir.name, dataset_prefix)

    def _folds_for(cfg_path: Path, training_segments: "set[str] | None") -> "list[Path] | None":
        if int(cv_folds) <= 0:
            return None
        cfg = train_module.load_config(str(cfg_path))
        return _materialize_cv_folds(
            dataset_dir=dataset_dir,
            sample_subdir=str(cfg["data"].get("sample_subdir", "samples")),
            row_count=row_count,
            n_folds=int(cv_folds),
            min_train_fraction=float(cv_min_train_fraction),
            eligible_segments=training_segments,
        )

    training_segments = None
    if int(cv_folds) > 0:
        # The reported split groups by segment and weights by valid-sample count, and
        # replicates multiply those counts uniformly, so the segment boundary is the
        # same whichever family probes for it.
        probe_cfg = candidates[0]
        probe_features = tuple(
            train_module.load_config(str(probe_cfg))["data"]["input_columns"]
        )
        training_segments = _training_portion_segments(
            dataset_dir=dataset_dir,
            surrogate_config_path=probe_cfg,
            row_count=row_count,
            full_features=probe_features,
            tmp_cfg_dir=tmp_cfg_dir,
        )

    if not token.startswith("auto"):
        chosen = _select_surrogate_config(train_configs, token)
        print(f"[SURROGATE] Fixed by --surrogate-model: {chosen.name}")
        return chosen, _folds_for(chosen, training_segments)

    n_fits = len(candidates) * max(1, int(cv_folds))
    scope = f" matching '{auto_prefix}'" if auto_prefix else ""
    print(
        f"[SURROGATE] Measuring {len(candidates)} configuration(s){scope} on the full "
        f"feature set to choose which one scores the search ({n_fits} fit(s))."
    )
    def _measure(cfgs: list) -> list:
        """Fit each config once on the full feature set and score it."""
        out = []
        for cfg_path in cfgs:
            cfg = train_module.load_config(str(cfg_path))
            features = tuple(cfg["data"]["input_columns"])
            try:
                result = _evaluate_candidate(
                    dataset_dir=dataset_dir,
                    target_name=target_name,
                    surrogate_config_path=cfg_path,
                    row_count=row_count,
                    features=features,
                    feature_tag=_feature_tag(features),
                    lambda_drop=lambda_drop,
                    tmp_cfg_dir=tmp_cfg_dir,
                    disable_baselines_for_search=disable_baselines_for_search,
                    disable_training_plots=disable_training_plots,
                    disable_eval_plots=disable_eval_plots,
                    suppress_training_logs=suppress_training_logs,
                    cv_fold_dirs=_folds_for(cfg_path, training_segments),
                )
            except Exception as exc:
                print(f"[SURROGATE] {cfg_path.name}: failed to evaluate ({exc}); not eligible.")
                continue
            if result is None:
                print(f"[SURROGATE] {cfg_path.name}: no result; not eligible.")
                continue
            out.append((float(result.objective), cfg_path, result))
            print(f"[SURROGATE]   {cfg_path.name:<38} objective={result.objective:.4f} "
                  f"r2={result.r2:.4f} folds={result.cv_folds} "
                  f"{'DEGENERATE' if bool(result.degenerate) else ''}".rstrip())
        return out

    scored = _measure(candidates)

    # If nothing in the requested family can vary its prediction, the family cannot rank
    # feature subsets at all and narrowing to it was the wrong call for this target. The
    # pool is widened and re-measured rather than proceeding with a constant.
    #
    # This is a rule, not a special case: it is evaluated for every target, from a
    # quantity measured before any search happens, and it fires only where the condition
    # holds. On CV22 that is exactly one target of fourteen -- Chromium, where all three
    # XGBoost configurations return pred_std 0.00000 on the full feature set. Lead comes
    # closest and does not qualify: xgb_03 varies (pred_std 0.02315), so the degeneracy
    # guard below picks it and the pool stays as requested.
    if scored and not any(not bool(item[2].degenerate) for item in scored) and auto_prefix:
        extra = [c for c in _surrogate_candidates(train_configs) if c not in candidates]
        if extra:
            print(f"[SURROGATE][WARN] every configuration matching '{auto_prefix}' predicts "
                  f"a constant on the full feature set, so none of them can separate "
                  f"feature subsets. Widening the pool to all {len(extra)} remaining "
                  f"configuration(s) rather than searching with a surrogate that would "
                  f"rank every subset identically.")
            scored = scored + _measure(extra)

    if not scored:
        raise RuntimeError(
            f"No family could be evaluated on the full feature set for "
            f"{dataset_dir.name}; the surrogate cannot be chosen by measurement."
        )

    # A degenerate surrogate predicts a constant, so every feature subset scores
    # identically and the beam search ranks nothing. This is not hypothetical: CV22's
    # Chromium search was scored by XGBoost, which returned r2 = -0.0013608598215928 for
    # all 240 candidates from 4 to 11 features, and the "selected" subset was settled by
    # the tie-break rule rather than by measurement. The Gaussian process was not
    # degenerate on that target at all -- gp_03 reached r2 = +0.50 -- so a usable
    # surrogate existed and was passed over because it scored worse on the full feature
    # set, which is the one subset where a degenerate constant is hardest to beat.
    #
    # Objective alone therefore cannot choose the surrogate: a model that cannot separate
    # subsets is worthless for the search however well it scores. Usable candidates rank
    # first; a degenerate one is used only when no family can separate anything.
    usable = [item for item in scored if not bool(item[2].degenerate)]
    n_degenerate = len(scored) - len(usable)
    pool = sorted(usable if usable else scored, key=lambda item: item[0])
    best_obj, chosen, best_result = pool[0]

    if usable and n_degenerate:
        argmin_obj, argmin_cfg, argmin_res = min(scored, key=lambda item: item[0])
        if bool(argmin_res.degenerate):
            print(
                f"[SURROGATE][WARN] {argmin_cfg.name} has the best objective "
                f"({argmin_obj:.4f}) but predicts a constant, so it would rank every "
                f"feature subset identically and the search would select by tie-break. "
                f"Using {chosen.name} (objective={best_obj:.4f}) to score the search "
                f"instead."
            )
        else:
            print(f"[SURROGATE] {n_degenerate} candidate(s) predict a constant and were "
                  f"excluded from the choice.")
    elif not usable:
        # Reported, not silently accepted: the subset this target ends up with was not
        # established by measurement, and Section 3.2 has to be able to say so.
        print(
            f"[SURROGATE][WARN] Every candidate surrogate predicts a constant on the full "
            f"feature set for {dataset_dir.name}. No family can separate feature subsets "
            f"here, so the search's selected subset is a tie-break rather than a "
            f"measurement. Proceeding with {chosen.name}; treat this target's retained "
            f"predictors as unestablished."
        )

    runner_up = (f", next best {pool[1][1].name} at {pool[1][0]:.4f}"
                 if len(pool) > 1 else "")
    print(
        f"[SURROGATE] Chosen: {chosen.name} (objective={best_obj:.4f}, "
        f"r2={best_result.r2:.4f}){runner_up}."
    )
    return chosen, _folds_for(chosen, training_segments)


def _compile_multi_target_comparison(
    sweep_results: dict[str, dict],  # target -> {row_count -> feature_sensitivities}
    data_root: Path,
    importance_label: str = "Removal Sensitivity (avg delta)",
    summary_axis_label: str = "Summed Target-wise Removal Sensitivity z-score",
    target_order: list[str] | None = None,
    dataset_prefix: str = "MC",
) -> Path:
    """Compile and visualize feature importance across multiple targets.

    Heatmap cells show raw per-target feature-importance values. Cross-target
    summary bars, feature ordering, and the heatmap total row use within-target
    z-score standardization before aggregation so targets with different units
    or scales are comparable.
    """
    if not sweep_results:
        return Path()
    
    # Collect all unique features across all targets
    all_features_set = set()
    target_feature_sets: dict[str, set[str]] = {}
    for target, target_data in sweep_results.items():
        feature_set = set()
        for feature_sensitivities in target_data.values():
            all_features_set.update(feature_sensitivities.keys())
            feature_set.update(feature_sensitivities.keys())
        # Track per-target feature presence for grouped ordering in summary figures
        target_feature_sets[target] = feature_set

    # Read feature order from Consolidated_sparse.csv.
    # Prefer the selected data root; keep legacy fallback for existing layouts.
    csv_candidates = [
        data_root / "Consolidated_sparse.csv",
        data_root.parent / "regression" / "Consolidated_sparse.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), csv_candidates[0])

    def _norm_name(value: str) -> str:
        text = str(value).lower().replace('µ', 'u').replace('°', 'deg')
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        return re.sub(r'[^a-z0-9]', '', text)

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        # Remove timestamp and non-feature columns if needed
        csv_features = [col for col in header if col in all_features_set]
        # Add any features not in CSV at the end (shouldn't happen, but for safety)
        all_features = csv_features + [f for f in sorted(all_features_set) if f not in csv_features]
    except Exception:
        all_features = sorted(all_features_set)
    
    if not all_features:
        return Path()
    
    # Build matrix: targets x features with removal sensitivity scores
    sweep_keys = list(sweep_results.keys())
    targets: list[str] = []
    yticklabels: list[str] = []
    used_keys: set[str] = set()

    def _match_target_key(raw_name: str) -> str | None:
        raw = str(raw_name)
        if raw in sweep_results and raw not in used_keys:
            return raw
        norm_raw = _norm_name(raw)
        if not norm_raw:
            return None
        for key in sweep_keys:
            if key in used_keys:
                continue
            norm_key = _norm_name(key)
            if not norm_key:
                continue
            if norm_key == norm_raw or norm_key.endswith(norm_raw) or norm_raw.endswith(norm_key):
                return key
        return None

    # Prefer explicit target order (for example, R2-ranked order from postprocess summary).
    if target_order:
        for requested in list(target_order):
            matched = _match_target_key(str(requested))
            if matched is None:
                continue
            targets.append(matched)
            raw_lbl = str(requested) if str(requested).strip() else matched
            yticklabels.append(clean_target_label(raw_lbl, dataset_prefix))
            used_keys.add(matched)

    # Fallback to CSV-based target ordering when no explicit order is provided.
    if not targets:
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            res_targets = [col for col in header if col.endswith("_res")]
            for csv_col in res_targets:
                matched = _match_target_key(csv_col)
                if matched is None:
                    continue
                targets.append(matched)
                yticklabels.append(clean_target_label(csv_col, dataset_prefix))
                used_keys.add(matched)
        except Exception:
            pass

    # Always append unmatched targets deterministically so nothing is dropped.
    for key in sweep_keys:
        if key in used_keys:
            continue
        targets.append(key)
        yticklabels.append(clean_target_label(key, dataset_prefix))
        used_keys.add(key)

    matrix = np.full((len(targets), len(all_features)), np.nan, dtype=float)

    for i, target in enumerate(targets):
        for j, feat in enumerate(all_features):
            # Use the first (finest) row_count's data for comparison
            for feature_sensitivities in sweep_results[target].values():
                if feat in feature_sensitivities:
                    matrix[i, j] = feature_sensitivities[feat][0]  # raw per-target importance
                    break

    zscore_matrix = np.zeros_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        finite_mask = np.isfinite(row)
        if not np.any(finite_mask):
            continue
        finite_vals = row[finite_mask]
        if finite_vals.size < 2:
            print(
                f"[WARN] Skipping z-score standardization for target '{targets[i]}' "
                "(fewer than 2 finite feature scores)."
            )
            continue
        row_mean = float(np.mean(finite_vals))
        row_std = float(np.std(finite_vals, ddof=0))
        if not np.isfinite(row_std) or row_std <= 0.0:
            print(
                f"[WARN] Skipping z-score standardization for target '{targets[i]}' "
                "(zero or non-finite feature-score variance)."
            )
            continue
        zscore_matrix[i, finite_mask] = (finite_vals - row_mean) / row_std

    # Order features globally by decreasing aggregated standardized score so the
    # heatmap columns and bar charts share the same ranking.
    feature_to_idx = {feat: idx for idx, feat in enumerate(all_features)}
    summed_sensitivity_raw = np.nansum(zscore_matrix, axis=0)
    feature_total_score = {
        feat: float(summed_sensitivity_raw[idx]) for feat, idx in feature_to_idx.items()
    }
    n_targets = len(targets)
    presence_count = {
        feat: sum(1 for target in targets if feat in target_feature_sets.get(target, set()))
        for feat in all_features
    }

    target_rank = {target: idx for idx, target in enumerate(targets)}
    target_rank_norm: dict[str, int] = {}
    for target, idx in target_rank.items():
        norm_target = _norm_name(target)
        if norm_target:
            target_rank_norm[norm_target] = min(idx, target_rank_norm.get(norm_target, idx))

    # Direct feature→target-rank lookup built from membership data.
    # This correctly handles cases where feature names and target names use
    # different languages or formats (e.g. "Turbidity (FNU)" vs "04-Turbiditet").
    feature_to_target_rank: dict[str, int] = {}
    for _tgt, _feats in target_feature_sets.items():
        _rank = target_rank.get(_tgt)
        if _rank is None:
            continue
        for _feat in _feats:
            if _feat not in feature_to_target_rank or _rank < feature_to_target_rank[_feat]:
                feature_to_target_rank[_feat] = _rank

    def _feature_target_candidates(feature_name: str) -> list[str]:
        feature_name = str(feature_name)
        candidates = [feature_name]
        if feature_name.endswith("_state"):
            stem = feature_name[: -len("_state")]
            candidates.extend([f"{stem}_res", stem])
        elif feature_name.endswith("_res"):
            candidates.append(feature_name[: -len("_res")])
        else:
            candidates.append(feature_name.replace("_state", "_res"))

        out: list[str] = []
        seen = set()
        for cand in candidates:
            cand = str(cand)
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out

    def _matching_target_rank(feature_name: str) -> int | None:
        # Primary: membership-based lookup — works regardless of name language/format.
        if feature_name in feature_to_target_rank:
            return feature_to_target_rank[feature_name]

        # Fallback: name-based matching for features absent from target_feature_sets.
        candidates = _feature_target_candidates(feature_name)
        for cand in candidates:
            if cand in target_rank:
                return int(target_rank[cand])

        best_rank: int | None = None
        for cand in candidates:
            norm_cand = _norm_name(cand)
            if not norm_cand:
                continue
            if norm_cand in target_rank_norm:
                rank = int(target_rank_norm[norm_cand])
                best_rank = rank if best_rank is None else min(best_rank, rank)
                continue
            for norm_target, rank in target_rank_norm.items():
                if norm_target == norm_cand or norm_target.endswith(norm_cand) or norm_cand.endswith(norm_target):
                    best_rank = int(rank) if best_rank is None else min(best_rank, int(rank))
        return best_rank

    multi_target_features = [feat for feat in all_features if presence_count.get(feat, 0) > 1]
    single_target_features = [feat for feat in all_features if presence_count.get(feat, 0) == 1]

    single_target_rank = {feat: _matching_target_rank(feat) for feat in single_target_features}
    single_matched = [feat for feat in single_target_features if single_target_rank.get(feat) is not None]
    single_unmatched = [feat for feat in single_target_features if single_target_rank.get(feat) is None]

    single_matched.sort(
        key=lambda feat: (
            int(single_target_rank.get(feat, 10**9)),
            -float(feature_total_score.get(feat, float("-inf"))),
            str(feat),
        )
    )
    single_unmatched.sort(
        key=lambda feat: (
            -float(feature_total_score.get(feat, float("-inf"))),
            str(feat),
        )
    )
    single_target_features = single_matched + single_unmatched

    ordered_features = sorted(
        all_features,
        key=lambda feat: (
            -float(feature_total_score.get(feat, float("-inf"))),
            str(feat),
        ),
    )
    if ordered_features:
        ordered_indices = [feature_to_idx[feat] for feat in ordered_features]
        matrix = matrix[:, ordered_indices]
        zscore_matrix = zscore_matrix[:, ordered_indices]
        all_features = ordered_features

    ordered_feature_to_idx = {feat: idx for idx, feat in enumerate(all_features)}
    multi_target_features = [feat for feat in all_features if presence_count.get(feat, 0) > 1]
    single_target_features = [feat for feat in all_features if presence_count.get(feat, 0) == 1]
    multi_idx = [ordered_feature_to_idx[feat] for feat in multi_target_features]
    single_idx = [ordered_feature_to_idx[feat] for feat in single_target_features]
    max_group_len = max(len(multi_target_features), len(single_target_features), 1)
    n_total_features = max(len(all_features), 1)

    # Single uniform font size used for all heatmap text (annotations, tick labels, axis labels).
    heat_font = 8
    heat_xtick_font = heat_font
    heat_ytick_font = heat_font
    heat_axis_label_font = heat_font
    # Bar typography.  The figure is now drawn at the width it is printed at, so these are
    # literal point sizes on the page rather than pre-shrunk values; scaling them by the
    # feature count would only reintroduce the illegibility it was meant to avoid.
    bar_tick_font = 7
    bar_title_font = 8
    bar_value_font = 7
    bar_axis_label_font = 8

    # These figures plot importance z-scores and inclusion counts, never measurements, so
    # predictor labels carry no units.  The source qualifier is kept because the predictor
    # pool contains both Surface and SCADA pH and water temperature.
    def _feature_display(feat: str) -> str:
        return names_label(feat, with_unit=False, qualified=True)

    if matrix.size:
        vmin = float(np.percentile(matrix, 5))
        vmax = float(np.percentile(matrix, 95))
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        finite_vals = matrix[np.isfinite(matrix)] if matrix.size else np.array([], dtype=float)
        if finite_vals.size:
            vmin = float(np.min(finite_vals))
            vmax = float(np.max(finite_vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmax = float(vmin + 1.0)

    annot_fontsize = heat_font

    def _annotate_heat_cells(ax_obj, values: np.ndarray, fontsize: int) -> None:
        for row_i in range(values.shape[0]):
            for col_j in range(values.shape[1]):
                value = values[row_i, col_j]
                if not np.isfinite(value):
                    continue
                ax_obj.text(
                    col_j + 0.5,
                    row_i + 0.5,
                    f"{value:.2e}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=fontsize,
                    rotation=0,
                    clip_on=True,
                )

    raw_summed_sensitivity = np.nansum(matrix, axis=0)
    summed_sensitivity = np.nansum(zscore_matrix, axis=0)
    top_features = list(all_features)
    raw_summed_scores = [float(v) for v in raw_summed_sensitivity]
    summed_scores = [float(v) for v in summed_sensitivity]

    heat_h = max(4, (len(targets) + 1) * 0.38)
    # Wrap the colorbar label so it doesn't extend beyond the colorbar height.
    # At heat_font pt, each character occupies ~0.65*heat_font points; colorbar height ~ heat_h*72 pts.
    cbar_label_wrap = max(20, int(heat_h * 72 / (heat_font * 0.65)))
    wrapped_importance_label = textwrap.fill(str(importance_label), cbar_label_wrap)

    if multi_idx and single_idx:
        yticklabels_with_total = yticklabels + ["Summed z-score"]
        left_block = np.vstack([matrix[:, multi_idx],
                                 np.array([summed_scores[i] for i in multi_idx])[None, :]])
        right_block = np.vstack([matrix[:, single_idx],
                                   np.array([summed_scores[i] for i in single_idx])[None, :]])
        sep_col = np.full((left_block.shape[0], 1), np.nan)
        combined_matrix = np.hstack([left_block, sep_col, right_block])
        sep_pos = left_block.shape[1]
        xticklabels_with_sep = (
            [_feature_display(f) for f in multi_target_features]
            + [""]
            + [_feature_display(f) for f in single_target_features]
        )

        n_total_cols = combined_matrix.shape[1]
        heat_w = max(14, n_total_cols * 0.85)
        fig, ax = plt.subplots(figsize=(heat_w, heat_h), constrained_layout=True)

        sns.heatmap(
            combined_matrix,
            ax=ax,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            annot=False,
            cbar_kws={"label": wrapped_importance_label, "pad": 0.01},
            xticklabels=xticklabels_with_sep,
            yticklabels=yticklabels_with_total,
            linewidths=0.5,
            linecolor="#eeeeee",
            square=False,
        )
        _annotate_heat_cells(ax, combined_matrix, annot_fontsize)

        # Cover separator column with background color to conceal grid lines.
        ax.add_patch(plt.Rectangle(
            (sep_pos, 0),
            1,
            combined_matrix.shape[0],
            facecolor=ax.get_facecolor(),
            edgecolor="none",
            zorder=3,
        ))

        ax.set_xticklabels(xticklabels_with_sep, rotation=45, ha='right', fontsize=heat_xtick_font)
        ax.set_yticklabels([textwrap.fill(lbl, 20) for lbl in yticklabels_with_total], rotation=0, fontsize=heat_ytick_font)
        ax.set_xlabel("Predictor", fontsize=heat_axis_label_font)
        ax.set_ylabel("Target", fontsize=heat_axis_label_font)
    else:
        # Add standardized aggregate row to matrix and yticklabels
        matrix_with_total = np.vstack([matrix, np.array(summed_scores)[None, :]])
        yticklabels_with_total = yticklabels + ["Summed z-score"]
        fig, ax = plt.subplots(
            figsize=(max(13, n_total_features * 0.85), heat_h),
            constrained_layout=True,
        )
        sns.heatmap(
            matrix_with_total,
            ax=ax,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            annot=False,
            cbar_kws={"label": wrapped_importance_label, "pad": 0.01},
            xticklabels=[_feature_display(f) for f in all_features],
            yticklabels=yticklabels_with_total,
            linewidths=0.5,
            linecolor="#eeeeee",
            square=False,
        )
        _annotate_heat_cells(ax, matrix_with_total, annot_fontsize)
        ax.set_xticklabels([_feature_display(f) for f in all_features], rotation=45, ha='right', fontsize=heat_xtick_font)
        ax.set_yticklabels([textwrap.fill(lbl, 20) for lbl in yticklabels_with_total], rotation=0, fontsize=heat_ytick_font)
        ax.set_xlabel("Predictor", fontsize=heat_axis_label_font)
        ax.set_ylabel("Target", fontsize=heat_axis_label_font)

    # Save to root output directory (namespace-specific for non-default sweeps)
    summaries_dir = (data_root / "summaries").resolve()
    namespace = _sweep_namespace()
    if namespace != "feature_sweeps":
        summaries_dir = (summaries_dir / namespace).resolve()
    summaries_dir.mkdir(parents=True, exist_ok=True)
    plot_path = summaries_dir / "multi_target_importance_heatmap.png"
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    # --- Transposed standalone heatmaps (predictors on y-axis, targets on x-axis) ---
    # Each feature block gets its own narrow figure, document-friendly width.
    _xtlabels_t = [textwrap.fill(lbl, 20) for lbl in yticklabels_with_total]
    _n_cols_t = len(yticklabels_with_total)
    if multi_idx and single_idx:
        for _t_block, _t_feats, _t_suffix in [
            (left_block, multi_target_features, "multi"),
            (right_block, single_target_features, "single"),
        ]:
            _t_mat = _t_block.T  # shape: (n_features, n_targets+1)
            _fig_t, _ax_t = plt.subplots(
                figsize=(max(5, _n_cols_t * 0.85), max(4, len(_t_feats) * 0.38)),
                constrained_layout=True,
            )
            sns.heatmap(
                _t_mat,
                ax=_ax_t,
                cmap="RdYlGn",
                vmin=vmin,
                vmax=vmax,
                annot=False,
                cbar_kws={"label": wrapped_importance_label, "pad": 0.01},
                xticklabels=_xtlabels_t,
                yticklabels=[textwrap.fill(_feature_display(f), 20) for f in _t_feats],
                linewidths=0.5,
                linecolor="#eeeeee",
                square=False,
            )
            _annotate_heat_cells(_ax_t, _t_mat, annot_fontsize)
            _ax_t.set_xticklabels(_xtlabels_t, rotation=45, ha='right', fontsize=heat_xtick_font)
            _ax_t.set_yticklabels([textwrap.fill(_feature_display(f), 20) for f in _t_feats], rotation=0, fontsize=heat_ytick_font)
            _ax_t.set_xlabel("Target", fontsize=heat_axis_label_font)
            _ax_t.set_ylabel("Predictor", fontsize=heat_axis_label_font)
            _t_path = summaries_dir / f"multi_target_importance_heatmap_{_t_suffix}.png"
            _fig_t.savefig(_t_path, dpi=180, bbox_inches='tight')
            plt.close(_fig_t)
            print(f"[INFO] Wrote transposed heatmap ({_t_suffix}): {_t_path}")
    else:
        _t_mat = matrix_with_total.T  # shape: (n_features, n_targets+1)
        _fig_t, _ax_t = plt.subplots(
            figsize=(max(5, _n_cols_t * 0.85), max(4, len(all_features) * 0.38)),
            constrained_layout=True,
        )
        sns.heatmap(
            _t_mat,
            ax=_ax_t,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            annot=False,
            cbar_kws={"label": wrapped_importance_label, "pad": 0.01},
            xticklabels=_xtlabels_t,
            yticklabels=[textwrap.fill(f, 20) for f in all_features],
            linewidths=0.5,
            linecolor="#eeeeee",
            square=False,
        )
        _annotate_heat_cells(_ax_t, _t_mat, annot_fontsize)
        _ax_t.set_xticklabels(_xtlabels_t, rotation=45, ha='right', fontsize=heat_xtick_font)
        _ax_t.set_yticklabels([textwrap.fill(f, 20) for f in all_features], rotation=0, fontsize=heat_ytick_font)
        _ax_t.set_xlabel("Target", fontsize=heat_axis_label_font)
        _ax_t.set_ylabel("Predictor", fontsize=heat_axis_label_font)
        _t_path = summaries_dir / "multi_target_importance_heatmap_all.png"
        _fig_t.savefig(_t_path, dpi=180, bbox_inches='tight')
        plt.close(_fig_t)
        print(f"[INFO] Wrote transposed heatmap (all): {_t_path}")

    # Create grouped bar charts using the same aggregated-score ordering as the heatmap.
    score_map = {feat: float(score) for feat, score in zip(top_features, summed_scores)}
    multi_scores = [score_map[feat] for feat in multi_target_features if feat in score_map]
    single_target_features_bar = [feat for feat in single_target_features if feat in score_map]
    single_scores = [score_map[feat] for feat in single_target_features_bar]
    all_scores = multi_scores + single_scores

    score_min = float(np.min(all_scores)) if all_scores else 0.0
    score_max = float(np.max(all_scores)) if all_scores else 1.0
    span = max(score_max - score_min, 1e-12)
    # Extra headroom prevents large-value label text from being clipped at plot bounds.
    pad = 0.24 * span
    y_lower = min(score_min, 0.0) - pad
    y_upper = max(score_max, 0.0) + pad
    color_norm = matplotlib.colors.Normalize(vmin=score_min, vmax=score_max) if score_max > score_min else None

    def _bar_colors(vals: list[float]) -> list:
        if color_norm is None:
            return [plt.cm.RdYlGn(0.5) for _ in vals]
        return [plt.cm.RdYlGn(float(color_norm(v))) for v in vals]

    def _draw_group_bars(ax_obj, features: list[str], values: list[float], title: str,
                         strip_suffix: bool = False) -> None:
        x_vals = np.arange(len(features), dtype=float)
        bars = ax_obj.bar(x_vals, values, color=_bar_colors(values))
        # The group name goes on the y-axis rather than in a title, because captions live
        # in the LaTeX document.  It still has to be stated somewhere: the two panels are
        # multi-target and single-target features, which is not inferable from the bars.
        ax_obj.set_xticks(x_vals)
        # The y-axis is a summed z-score, so predictor names must not carry concentration
        # units here.  The source qualifier is kept: the pool holds both Surface and
        # SCADA pH and water temperature.  ``strip_suffix`` drops the ", previous value"
        # wording when every bar in the panel is a state feature, since repeating it 14
        # times says nothing the axis label does not already say.
        ax_obj.set_xticklabels(
            [names_label(f, with_unit=False, qualified=True, with_suffix=not strip_suffix)
             for f in features],
            rotation=45, ha='right', fontsize=bar_tick_font,
        )
        ax_obj.grid(axis='y', alpha=0.3)
        ax_obj.axhline(0.0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
        ax_obj.margins(x=0.01)
        y_span = max(abs(y_upper - y_lower), 1.0)
        # Reserve a larger interior band for rotated labels at document-scale fonts.
        label_margin = 0.06 * y_span
        for bar, val in zip(bars, values):
            y = bar.get_height()
            offset = 0.012 * y_span
            y_text = y + offset if y >= 0 else y - offset
            y_text = float(np.clip(y_text, y_lower + label_margin, y_upper - label_margin))
            ax_obj.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                # Summed z-scores are O(1)-O(10); scientific notation made every label
                # three times longer than the value warranted.
                f"{val:.2f}",
                ha='center',
                va='center',
                fontsize=bar_value_font,
                rotation=90,
                clip_on=True,
            )
        ax_obj.set_ylim(y_lower, y_upper)

    # Draw at the width the figure is actually printed at.  Previously this was at least
    # 15 in wide and then scaled to the 6.5 in text block, which shrank the 13 pt
    # predictor labels to ~5.6 pt on the page while leaving the colour-coded bars far
    # wider than they needed to be.
    if multi_target_features and single_target_features:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(PAGE_WIDTH_IN, 6.4),
            sharey=True,
            constrained_layout=True,
        )
        _draw_group_bars(ax_top, multi_target_features, multi_scores, "Multi-target features")
        # Every single-target feature is a target's own previous value, so that is stated
        # once here instead of on all fourteen tick labels.
        _single_are_all_state = all(
            str(f).endswith("_state") for f in single_target_features_bar
        )
        _draw_group_bars(ax_bottom, single_target_features_bar, single_scores,
                         "Single-target features", strip_suffix=_single_are_all_state)
        _wrapped_ylabel = textwrap.fill(str(summary_axis_label), width=24)
        # The panel distinction moves from the (removed) titles into the y-axis labels,
        # so it survives without duplicating the caption.
        ax_top.set_ylabel(f"{_wrapped_ylabel}\n(multi-target features)",
                          fontsize=bar_axis_label_font)
        _bottom_note = ("previous target value" if _single_are_all_state
                        else "single-target features")
        ax_bottom.set_ylabel(f"{_wrapped_ylabel}\n({_bottom_note})",
                             fontsize=bar_axis_label_font)
        ax_bottom.set_xlabel("")
    else:
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.6), constrained_layout=True)
        _draw_group_bars(ax, top_features, summed_scores, "Feature importance")
        ax.set_ylabel(textwrap.fill(str(summary_axis_label), width=24),
                      fontsize=bar_axis_label_font)
        ax.set_xlabel("")

    bar_path = summaries_dir / "multi_target_importance_bars.png"
    save_figure(fig, bar_path)
    plt.close(fig)

    return plot_path


def _write_selection_stability_artifacts(
    dataset_dir: Path,
    row_count: int,
    trace: list[CandidateResult],
    tolerance: float = 0.02,
) -> Path | None:
    """Write the near-optimal retention-frequency table for one dataset and row count."""
    if not trace:
        return None
    trace_df = pd.DataFrame([
        {"features": "|".join(item.features), "objective": item.objective}
        for item in trace
    ])
    table, summary = selection_stability_from_trace(trace_df, tolerance=tolerance)
    if table.empty:
        return None

    out_dir = _forecast_sweeps_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"feature_retention_frequency_r{int(row_count):03d}.csv"
    table.to_csv(out_csv, index=False)

    always = table[table["retention_frequency"] >= 1.0]["feature"].tolist()
    print(
        f"[STABILITY] {summary['n_near_optimal']} of {summary['n_evaluated']} subsets lie "
        f"within {tolerance:g} of the best objective. "
        f"{len(always)} feature(s) appear in all of them: "
        f"{', '.join(always) if always else 'none'}."
    )
    print(f"[STABILITY] Wrote {out_csv}")
    return out_csv


def _write_search_outputs(
    dataset_dir: Path,
    row_count: int,
    trace: list[CandidateResult],
    selected: list[CandidateResult],
    save_plots: bool,
) -> tuple[Path, Path, Path]:
    """Write search trace/selection CSVs and optional Pareto plot.

    Args:
        dataset_dir: Dataset directory whose sweep namespace receives outputs.
        row_count: Evaluated lookback row-count for filename suffixing.
        trace: Chronological candidate evaluations to persist as trace CSV.
        selected: Final selected subsets to persist as ranked CSV.
        save_plots: If True, also write `feature_search_pareto_r###.png`.

    Returns:
        `(trace_csv, selected_csv, plot_path)` where `plot_path` is returned even
        when `save_plots=False` (file may not exist in that case).

    Example:
        `_write_search_outputs(..., save_plots=False)` writes only CSVs for
        lightweight search runs.
    """
    out_dir = _forecast_sweeps_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = []
    for idx, item in enumerate(trace, start=1):
        trace_rows.append(
            {
                "eval_index": idx,
                "target": item.target,
                "row_count": item.row_count,
                "feature_tag": item.feature_tag,
                "n_features": item.n_features,
                "objective": item.objective,
                "rmse": item.rmse,
                "r2": item.r2,
                "mae": item.mae,
                "drop_rate": item.drop_rate,
                "n_valid_raw": item.n_valid_raw,
                "n_total_raw": item.n_total_raw,
                "n_valid_loaded": item.n_valid_loaded,
                "n_test_samples": item.n_test_samples,
                "input_dim": item.input_dim,
                "target_dim": item.target_dim,
                "source": item.source,
                "seeded_input_rank": item.seeded_input_rank,
                "training_stop_reason": item.training_stop_reason,
                "cv_folds": item.cv_folds,
                "cv_r2_mean": item.cv_r2_mean,
                "cv_r2_se": item.cv_r2_se,
                "objective_se": item.cv_objective_se,
                "pred_std": item.pred_std,
                "degenerate": item.degenerate,
                "features": "|".join(item.features),
            }
        )
    trace_df = pd.DataFrame(trace_rows)
    trace_csv = out_dir / f"feature_search_trace_r{row_count:03d}.csv"
    trace_df.to_csv(trace_csv, index=False)

    selected_rows = []
    for rank, item in enumerate(selected, start=1):
        selected_rows.append(
            {
                "rank": rank,
                "target": item.target,
                "row_count": item.row_count,
                "feature_tag": item.feature_tag,
                "n_features": item.n_features,
                "objective": item.objective,
                "rmse": item.rmse,
                "r2": item.r2,
                "mae": item.mae,
                "drop_rate": item.drop_rate,
                "n_valid_raw": item.n_valid_raw,
                "n_total_raw": item.n_total_raw,
                "source": item.source,
                "seeded_input_rank": item.seeded_input_rank,
                "features": "|".join(item.features),
            }
        )
    selected_df = pd.DataFrame(selected_rows)
    selected_csv = out_dir / f"feature_selected_subsets_r{row_count:03d}.csv"
    selected_df.to_csv(selected_csv, index=False)

    plot_path = out_dir / f"feature_search_pareto_r{row_count:03d}.png"
    if save_plots and not trace_df.empty:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5), constrained_layout=True)
        ax.scatter(trace_df["drop_rate"], trace_df["rmse"], s=20, alpha=0.6)
        if not selected_df.empty:
            ax.scatter(selected_df["drop_rate"], selected_df["rmse"], s=60, marker="*", color="red")
        ax.set_xlabel("Drop rate (raw sample loss)")
        ax.set_ylabel("RMSE (surrogate)")
        ax.grid(alpha=0.25)
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)

    return trace_csv, selected_csv, plot_path


def _read_split_file_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if str(line).strip()]


def _write_split_file_names(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{name}\n")


def _raw_file_name(name: str) -> str:
    return re.sub(r"_mc_\d+(?=\.csv$)", "", str(name))


def _independent_name_count(names: list[str]) -> int:
    return len(dict.fromkeys(_raw_file_name(n) for n in names if str(n).strip()))


def _ensure_min_test_samples_for_final(
    split_dir: Path,
    min_test_samples: int = FINAL_TOPK_MIN_TEST_SAMPLES,
) -> tuple[bool, str, int, int]:
    """Rebalance split files for final-top-k evaluation using independent sample groups.

    Returns:
            - skip_eval: True when total available independent groups are < min_test_samples.
      - status: one of {'already_sufficient', 'rebalanced', 'insufficient_total'}.
      - train_count: resulting train file count.
      - test_count: resulting test file count.
    """
    train_file = split_dir / "train_files.txt"
    test_file = split_dir / "test_files.txt"

    train_names = _read_split_file_names(train_file)
    test_names = _read_split_file_names(test_file)

    if _independent_name_count(test_names) >= min_test_samples:
        return False, "already_sufficient", len(train_names), len(test_names)

    total_independent = _independent_name_count(train_names + test_names)
    if total_independent < min_test_samples:
        return True, "insufficient_total", len(train_names), len(test_names)

    test_raw_ids = set(_raw_file_name(n) for n in test_names)
    train_raw_order = []
    seen_raw = set()
    for name in train_names:
        raw = _raw_file_name(name)
        if raw not in seen_raw:
            seen_raw.add(raw)
            train_raw_order.append(raw)

    moved_raw_ids: list[str] = []
    for raw in reversed(train_raw_order):
        if raw in test_raw_ids:
            continue
        moved_raw_ids.append(raw)
        test_raw_ids.add(raw)
        if len(test_raw_ids) >= min_test_samples:
            break

    if len(test_raw_ids) < min_test_samples:
        return True, "insufficient_total", len(train_names), len(test_names)

    moved_raw_set = set(moved_raw_ids)
    moved = [name for name in train_names if _raw_file_name(name) in moved_raw_set]
    new_train = [name for name in train_names if _raw_file_name(name) not in moved_raw_set]
    # Keep temporal order: moved train tail should precede existing test tail.
    new_test = moved + test_names

    _write_split_file_names(train_file, new_train)
    _write_split_file_names(test_file, new_test)
    return False, "rebalanced", len(new_train), len(new_test)


def _install_seed_ensemble(
    variant_cfg: Path,
    primary_dir: Path,
    dataset_dir: Path,
    model_type: str,
    hyper: dict,
    n_seeds: int,
    seed_base: int,
    *,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
) -> dict | None:
    """Refit the evaluated run at further seeds and install their mean prediction.

    This is what makes ``--seeds`` mean something for the reported result. The final
    per-family re-fit is the run ``z8`` scores and Table 3 quotes, and until now it was
    a single draw: the seed averaging stopped at the beam search, so the number the paper
    reports was still chosen from one draw of a model whose seed moves R^2 by a standard
    deviation of 0.03 at the median and up to 0.44.

    Averaging *predictions* rather than scores is the only internally consistent choice.
    There is no prediction series whose R^2 is the mean of six others, so reporting a
    mean R^2 beside a significance verdict computed from one seed would quote two
    different models. The ensemble has a single prediction vector, so R^2, the skill
    score and the permutation test all describe the same thing -- and it is deployable,
    where "the average score of six models you would still have to choose between" is
    not. Squared error being convex, the ensemble also scores at or above the mean of the
    parts, by the across-seed variance term (0.035 on Cadmium at horizon 0).

    Seed *base* is the primary run already trained and evaluated by the caller, so only
    ``n_seeds - 1`` further fits are needed. The replicates go to ``seed_reps/`` and are
    never scored as candidates in their own right. The original single-seed predictions
    are kept beside the ensemble as ``predictions_seed0.csv``, so this is reversible.

    Returns:
        The metrics recomputed from the ensembled predictions, for the caller to write
        onto the row, or ``None`` if no ensemble was installed.
    """
    if int(n_seeds) <= 1 or not _is_stochastic_model(model_type, hyper):
        return None

    primary_csv = Path(primary_dir) / "predictions.csv"
    if not primary_csv.is_file():
        print("[WARN] seed ensemble: %s has no predictions.csv; left single-seed."
              % Path(primary_dir).name)
        return None

    def _keyed(csv_path: Path):
        """(sorted frame, prediction column, alignment key) for one run."""
        t = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
        if "target" not in t.columns:
            return None, None, None
        after = list(t.columns[t.columns.get_loc("target") + 1:])
        col = next((c for c in after
                    if c not in {"Naive", "Seasonal", "Linear"}
                    and not str(c).endswith(("_std", "_var"))), None)
        if col is None:
            return None, None, None
        sort_cols = [c for c in ("kind", "sample_file") if c in t.columns]
        if sort_cols:
            t = t.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        key = tuple(map(tuple, t[sort_cols].astype(str).to_numpy())) if sort_cols else None
        return t, col, key

    base_frame, base_col, base_key = _keyed(primary_csv)
    if base_frame is None:
        print("[WARN] seed ensemble: no prediction column in %s; left single-seed."
              % Path(primary_dir).name)
        return None

    vecs = [base_frame[base_col].to_numpy(dtype=float)]
    for k in range(1, int(n_seeds)):
        seed = int(seed_base) + k
        try:
            seeded_cfg = _seeded_variant_config(variant_cfg, seed)
            rep_eval_cfg = _train_single_config(
                seeded_cfg,
                dataset_dir,
                disable_training_plots=disable_training_plots,
                disable_eval_plots=True,
                suppress_training_logs=suppress_training_logs,
            )
            _set_eval_overrides(rep_eval_cfg, run_baselines=False)
            eval_module.evaluate_single_config(str(rep_eval_cfg), save_plots_override=False)
            frame, col, key = _keyed(Path(rep_eval_cfg).parent / "predictions.csv")
        except Exception as exc:
            # A seed that will not fit is recorded and dropped, not silently absorbed
            # into a smaller ensemble that still claims n_seeds.
            print("[WARN] seed ensemble: seed %d of %s failed (%s: %s); excluded."
                  % (seed, Path(primary_dir).name, type(exc).__name__,
                     str(exc).splitlines()[0][:120]))
            continue
        if frame is None:
            print("[WARN] seed ensemble: seed %d of %s produced no predictions; excluded."
                  % (seed, Path(primary_dir).name))
            continue
        # Aligning on the row key rather than position: a replicate whose split moved is
        # not a like-for-like fit and must not be averaged into the ensemble.
        if key != base_key or len(frame) != len(base_frame):
            print("[WARN] seed ensemble: seed %d of %s is not row-aligned with the "
                  "primary fit; excluded." % (seed, Path(primary_dir).name))
            continue
        vecs.append(frame[col].to_numpy(dtype=float))

    if len(vecs) < 2:
        print("[WARN] seed ensemble: %s kept only the primary fit; left single-seed."
              % Path(primary_dir).name)
        return None

    backup = Path(primary_dir) / "predictions_seed0.csv"
    if not backup.exists():
        shutil.copy2(primary_csv, backup)
    out = base_frame.copy()
    out[base_col] = np.mean(vecs, axis=0)
    out.to_csv(primary_csv, index=False)

    y, p = _pooled_predictions(Path(primary_dir))
    if y.size < 2:
        return None
    resid = y - p
    std, deg = _prediction_spread(y, p)
    metrics = {
        "r2": _pooled_r2(y, p),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "pred_std": std,
        "degenerate": bool(deg),
        "n_seeds_ensembled": len(vecs),
    }
    if np.std(y) > 0 and np.std(p) > 0:
        metrics["pearson_r"] = float(np.corrcoef(y, p)[0, 1])
    print("[SEED] %s: ensembled %d fits -> R2 %+.4f (was %+.4f)"
          % (Path(primary_dir).name, len(vecs), metrics["r2"],
             _pooled_r2(*_pooled_predictions_from(backup))))
    return metrics


def _pooled_predictions_from(pred_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """``_pooled_predictions`` against a named file, for reporting the pre-ensemble score."""
    df = pd.read_csv(pred_csv, encoding="utf-8", encoding_errors="replace")
    if "kind" in df.columns:
        df = df[df["kind"].astype(str) == "test"]
    if df.empty or "target" not in df.columns:
        return np.array([]), np.array([])
    after = list(df.columns[df.columns.get_loc("target") + 1:])
    col = next((c for c in after
                if c not in {"Naive", "Seasonal", "Linear"}
                and not str(c).endswith(("_std", "_var"))), None)
    if col is None:
        return np.array([]), np.array([])
    y = pd.to_numeric(df["target"], errors="coerce").to_numpy(dtype=float)
    pv = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(pv)
    return y[ok], pv[ok]


def _evaluate_selected_subsets_all_models(
    dataset_plan: DatasetPlan,
    dataset_prefix: str,
    selected: list[CandidateResult],
    run_baselines_in_final: bool,
    disable_training_plots: bool,
    disable_eval_plots: bool,
    suppress_training_logs: bool,
) -> Path:
    rows = []
    output_dir = _forecast_sweeps_dir(dataset_plan.dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = output_dir / "configs"

    # Registry of evaluated (base_feature_tag, model_name) → (source_dir, row_payload).
    # Used to skip retraining when a candidate feature set has already been evaluated
    # under a different subset label (e.g. k01 and s01 share the same features).
    _eval_registry: dict[tuple[str, str], tuple[Path, dict]] = {}

    target_name = _derive_target_name(dataset_plan.dataset_dir.name, dataset_prefix)

    for rank, cand in enumerate(selected, start=1):
        if not _candidate_uses_uncertainty_distributions(tuple(cand.features)):
            print(
                f"[MC-POLICY] Final top-k subset rank={rank} ({cand.feature_tag}) has no uncertainty-enabled predictors; "
                "evaluation will use collapsed original samples and omit MC prediction stats."
            )
        # Keep one best row per baseline id (naive/seasonal/linear) across all model evaluations.
        best_baseline_rows: dict[str, tuple[float, float, dict]] = {}
        for base_cfg in dataset_plan.train_configs:
            variant_cfg = _prepare_variant_config(
                base_config_path=base_cfg,
                row_count=cand.row_count,
                features=cand.features,
                feature_tag=f"{cand.feature_tag}_k{rank:02d}",
                tmp_dir=cfg_dir,
                forced_data_dir=dataset_plan.dataset_dir,
            )

            _failure_reason = ""
            try:
                eval_cfg = _train_single_config(
                    variant_cfg,
                    dataset_plan.dataset_dir,
                    disable_training_plots=disable_training_plots,
                    disable_eval_plots=disable_eval_plots,
                    suppress_training_logs=suppress_training_logs,
                )
                skip_eval, split_status, train_n, test_n = _ensure_min_test_samples_for_final(
                    split_dir=eval_cfg.parent,
                    min_test_samples=FINAL_TOPK_MIN_TEST_SAMPLES,
                )
                if split_status == "rebalanced":
                    print(
                        f"[INFO] Final split rebalance for {eval_cfg.parent.name}: "
                        f"train={train_n}, test={test_n} "
                        f"(min_test_independent={FINAL_TOPK_MIN_TEST_SAMPLES})"
                    )
                if skip_eval:
                    raise SampleComplianceError(
                        reason="insufficient_total_independent",
                        message=(
                            f"Final split for {eval_cfg.parent.name} cannot satisfy minimum "
                            f"independent test groups ({FINAL_TOPK_MIN_TEST_SAMPLES})."
                        ),
                        context={
                            "dataset": dataset_plan.dataset_dir.name,
                            "variant_dir": str(eval_cfg.parent),
                            "train_rows": int(train_n),
                            "test_rows": int(test_n),
                            "target_min_independent": int(FINAL_TOPK_MIN_TEST_SAMPLES),
                        },
                    )
                else:
                    _set_eval_overrides(
                        eval_cfg,
                        run_baselines=run_baselines_in_final,
                    )
                    eval_result = eval_module.evaluate_single_config(
                        str(eval_cfg),
                        save_plots_override=not disable_eval_plots,
                    )

                    # --seeds applies here, to the run z8 scores and Table 3 quotes,
                    # and not only to the beam search that shortlisted it.
                    _ens_metrics = None
                    if _CANDIDATE_SEEDS > 1:
                        try:
                            with open(variant_cfg, "r", encoding="utf-8") as _vf:
                                _vcfg = yaml.safe_load(_vf) or {}
                        except Exception:
                            _vcfg = {}
                        _ens_metrics = _install_seed_ensemble(
                            variant_cfg,
                            Path(eval_cfg).parent,
                            dataset_plan.dataset_dir,
                            str(_vcfg.get("model_type", "")),
                            _vcfg.get("hyperparameters") or {},
                            _CANDIDATE_SEEDS,
                            _CANDIDATE_SEED_BASE,
                            disable_training_plots=disable_training_plots,
                            disable_eval_plots=disable_eval_plots,
                            suppress_training_logs=suppress_training_logs,
                        )

                    def _apply_ensemble(row):
                        """Overwrite a model row's scores with the ensemble's.

                        Counts are untouched: the split does not move between seeds, so
                        n_test_independent and the contract fields stay valid. Baseline
                        rows are never passed here -- the reference forecasts are not
                        refitted and their scores are unchanged.
                        """
                        if not _ens_metrics or row is None:
                            return row
                        for _k in ("r2", "rmse", "mae", "pearson_r"):
                            if _k in _ens_metrics:
                                row[_k] = _ens_metrics[_k]
                        row["n_seeds_ensembled"] = _ens_metrics["n_seeds_ensembled"]
                        return row

                    summary_rows = []
                    eval_summary_csv = Path(eval_cfg).parent / "evaluation_summary.csv"
                    if eval_summary_csv.exists():
                        try:
                            summary_df = pd.read_csv(eval_summary_csv)

                            # Pick one primary model row using the same preference order as evaluate_single_config.
                            primary_model_row = None
                            if not summary_df.empty:
                                if "kind" in summary_df.columns:
                                    kinds = summary_df["kind"].astype(str).str.lower().str.strip()
                                    for preferred_kind in ("test", "combined", "train"):
                                        hit = summary_df[kinds == preferred_kind]
                                        if not hit.empty:
                                            primary_model_row = hit.iloc[0].to_dict()
                                            break

                                if primary_model_row is None:
                                    for _, _row in summary_df.iterrows():
                                        if _normalize_baseline_label(_row.get("label", "")) is None:
                                            primary_model_row = _row.to_dict()
                                            break

                            if primary_model_row is not None:
                                summary_rows.append(_apply_ensemble(primary_model_row))

                            # Add one row per baseline model (Naive/Seasonal/Linear).
                            seen_baselines: set[str] = set()
                            for _, _row in summary_df.iterrows():
                                baseline_id = _normalize_baseline_label(_row.get("label", ""))
                                if baseline_id is None or baseline_id in seen_baselines:
                                    continue
                                summary_rows.append(_row.to_dict())
                                seen_baselines.add(baseline_id)

                        except Exception as read_exc:
                            print(f"[WARN] Could not read evaluation_summary.csv at {eval_summary_csv}: {read_exc}")

                    if not summary_rows and eval_result is not None:
                        summary_rows = [_apply_ensemble(eval_result)]
            except SampleComplianceError as e:
                ctx = getattr(e, "context", {}) or {}
                print(
                    f"[COMPLIANCE] {e.reason}: {e}. "
                    f"dataset={ctx.get('dataset', dataset_plan.dataset_dir.name)} "
                    f"variant={ctx.get('variant_dir', str(variant_cfg))}"
                )
                summary_rows = []
                _failure_reason = f"compliance:{e.reason}"
            except Exception as e:
                print(f"[ERROR] Evaluation failed for config {variant_cfg}: {e}")
                summary_rows = []
                # Recorded on the row itself. A metrics table showing NaN with no reason
                # cannot distinguish a model that failed to fit from one never attempted,
                # and the terminal output holding the explanation is long gone by the
                # time anyone reads the table.
                _failure_reason = f"{type(e).__name__}: {str(e).splitlines()[0][:180]}"

            if not summary_rows:
                (print(f"[WARN] No summary_rows > no evaluation results for config {variant_cfg}, writing NaNs for metrics."))
                # Write a row with error info/NaNs
                try:
                    with open(variant_cfg, 'r', encoding='utf-8') as f:
                        cfg = yaml.safe_load(f)
                    model_name = cfg.get("model_name", "unknown")
                except Exception:
                    model_name = "unknown"
                rows.append({
                    "dataset": dataset_plan.dataset_dir.name,
                    "target": target_name,
                    "subset_rank": rank,
                    "subset_label": f"k{rank:02d}",
                    "feature_tag": cand.feature_tag,
                    "row_count": cand.row_count,
                    "n_features": cand.n_features,
                    "objective_search": cand.objective,
                    "drop_rate_search": cand.drop_rate,
                    "model": model_name,
                    **(dict(_EMPTY_VARIANT_FIELDS)
                       if _is_baseline_model_value(model_name)
                       else _model_variant_fields(base_cfg)),
                    "failure_reason": _failure_reason,
                    "gp_uncertainty_mode": "",
                    "n_samples": float('nan'),
                    "n_test_independent": float('nan'),
                    "n_test_valid": float('nan'),
                    "n_test_evals": float('nan'),
                    "input_dim": float('nan'),
                    "target_dim": float('nan'),
                    "mae": float('nan'),
                    "rmse": float('nan'),
                    "r2": float('nan'),
                    "pearson_r": float('nan'),
                    "std_target": float('nan'),
                    "n_seeds_ensembled": float('nan'),
                })
                continue

            for srow in summary_rows:
                # Load eval config once for all derived values
                _eval_cfg_data = {}
                try:
                    with open(eval_cfg, 'r', encoding='utf-8') as f:
                        _eval_cfg_data = yaml.safe_load(f) or {}
                except Exception:
                    pass

                # model: write_evaluation_config always sets model_name=""; fall back to model_type
                model_name_default = (
                    _eval_cfg_data.get("model_name")
                    or _eval_cfg_data.get("model_type")
                    or srow.get("label", "unknown")
                )
                baseline_id = _normalize_baseline_label(srow.get("label", ""))
                model_name = baseline_id if baseline_id is not None else model_name_default
                gp_uncertainty_mode = str(srow.get("gp_uncertainty_mode", ""))

                row_context = (
                    f"final_summary:{eval_cfg.parent.name} "
                    f"dataset={dataset_plan.dataset_dir.name} subset_rank={rank} model={model_name}"
                )
                _validate_eval_metric_contract(srow, context=row_context)
                mae_val = _extract_required_independent_metric(srow, "mae", context=row_context)
                rmse_val = _extract_required_independent_metric(srow, "rmse", context=row_context)
                n_samples = _extract_required_independent_metric(srow, "n_test_independent", context=row_context)
                r2_val = float(pd.to_numeric(srow.get("r2", np.nan), errors="coerce"))
                pearson_val = float(pd.to_numeric(srow.get("pearson_r", np.nan), errors="coerce"))

                # Prefer explicit independent-sample semantics for the primary count.
                n_test_independent = float(srow.get("n_test_independent", n_samples))
                n_test_valid = float(srow.get("n_test_valid", np.nan))
                n_test_evals = float(srow.get("n_test_evals", srow.get("n_eval_rows", np.nan)))

                if baseline_id is None and (not np.isfinite(n_test_valid) or n_test_valid < float(FINAL_TOPK_MIN_TEST_SAMPLES)):
                    raise SampleComplianceError(
                        reason="insufficient_valid_independent",
                        message=(
                            f"Non-baseline evaluation has insufficient valid independent test samples "
                            f"({n_test_valid}) for minimum {FINAL_TOPK_MIN_TEST_SAMPLES}."
                        ),
                        context={
                            "dataset": dataset_plan.dataset_dir.name,
                            "variant_dir": str(eval_cfg.parent),
                            "subset_rank": int(rank),
                            "row_count": int(cand.row_count),
                            "feature_tag": str(cand.feature_tag),
                            "model": str(model_name),
                            "n_test_valid": float(n_test_valid) if np.isfinite(n_test_valid) else float("nan"),
                            "target_min_independent": int(FINAL_TOPK_MIN_TEST_SAMPLES),
                        },
                    )

                # input_dim: count input_columns from the eval config
                _input_cols = _eval_cfg_data.get("data", {}).get("input_columns") or []
                input_dim = float(len(_input_cols)) if _input_cols else np.nan

                # target_dim: n_eval_outputs is the number of output columns
                target_dim = float(srow.get("n_eval_outputs", np.nan))

                # Compute std(target) for this model/config
                std_target = float('nan')
                try:
                    target_cols = _eval_cfg_data.get('data', {}).get('output_columns', None)
                    # data_dir in eval config is relative to eval config file location
                    _raw_data_dir = _eval_cfg_data.get('data', {}).get('data_dir', '')
                    _cfg_dir = Path(eval_cfg).parent
                    data_dir = str((_cfg_dir / _raw_data_dir).resolve())
                    sample_subdir = _eval_cfg_data.get('data', {}).get('sample_subdir', 'samples')
                    sample_dir = os.path.join(data_dir, sample_subdir)
                    csv_files = glob.glob(os.path.join(sample_dir, '*.csv'))
                    target_vals = []
                    for csvf in csv_files:
                        try:
                            df_csv = pd.read_csv(csvf)
                            if target_cols and all(tc in df_csv.columns for tc in target_cols):
                                for tc in target_cols:
                                    target_vals.extend(df_csv[tc].dropna().values.tolist())
                        except Exception:
                            continue
                    if target_vals:
                        std_target = float(np.std(target_vals, ddof=1))
                except Exception as e:
                    print(f"[WARN] Could not compute std(target) for {dataset_plan.dataset_dir.name}: {e}")

                row_payload = {
                    "dataset": dataset_plan.dataset_dir.name,
                    "target": target_name,
                    "subset_rank": rank,
                    "subset_label": f"k{rank:02d}",
                    "feature_tag": cand.feature_tag,
                    "row_count": cand.row_count,
                    "n_features": cand.n_features,
                    "objective_search": cand.objective,
                    "drop_rate_search": cand.drop_rate,
                    "model": model_name,
                    **(dict(_EMPTY_VARIANT_FIELDS)
                       if _is_baseline_model_value(model_name)
                       else _model_variant_fields(base_cfg, eval_cfg.parent)),
                    **_degeneracy_fields(eval_cfg.parent),
                    "failure_reason": "",
                    "gp_uncertainty_mode": gp_uncertainty_mode,
                    "n_samples": n_samples,
                    "n_test_independent": n_test_independent,
                    "n_test_valid": n_test_valid,
                    "n_test_evals": n_test_evals,
                    "input_dim": input_dim,
                    "target_dim": target_dim,
                    "mae": mae_val,
                    "rmse": rmse_val,
                    "r2": r2_val,
                    "pearson_r": pearson_val,
                    "std_target": std_target,
                    # 1 for a single fit; N when the scores above were recomputed from an
                    # N-seed mean prediction, so a reader can tell which rows are ensembles.
                    "n_seeds_ensembled": int(srow.get("n_seeds_ensembled", 1) or 1),
                }

                if baseline_id is None:
                    rows.append(row_payload)
                    _eval_registry[(cand.feature_tag, _variant_key(base_cfg))] = (eval_cfg.parent, row_payload)
                else:
                    valid_score = n_test_valid if np.isfinite(n_test_valid) else float("-inf")
                    eval_score = n_test_evals if np.isfinite(n_test_evals) else float("-inf")
                    current = best_baseline_rows.get(baseline_id)
                    if current is None or (valid_score, eval_score) > (current[0], current[1]):
                        best_baseline_rows[baseline_id] = (valid_score, eval_score, row_payload)

        for baseline_id in ("naive", "seasonal", "linear"):
            best = best_baseline_rows.get(baseline_id)
            if best is not None:
                rows.append(best[2])
        for baseline_id, (_, _, payload) in best_baseline_rows.items():
            if baseline_id not in {"naive", "seasonal", "linear"}:
                rows.append(payload)

        # --- MLR model variants on this k## feature set ---
        try:
            from utils.mlr import MLR_VARIANTS as _MLR_VARIANTS
            _k_tag = f"{cand.feature_tag}_k{rank:02d}"
            _k_variant_dirs = sorted(output_dir.glob(f"*_r{cand.row_count:03d}_{_k_tag}*"))
            if not _k_variant_dirs:
                # Fall back to the MLR artifact directory itself, which already contains
                # train_files.txt / test_files.txt written earlier in this sweep run.
                _mlr_fallback = output_dir / f"mlr_k{rank:02d}"
                if _mlr_fallback.exists():
                    _k_variant_dirs = [_mlr_fallback]
            _mlr_on_k_done = False
            for _vd in _k_variant_dirs:
                _ecfg_path = _vd / f"config_evaluate_{_vd.name}.yml"
                if not _ecfg_path.exists():
                    continue
                try:
                    _kcfg = yaml.safe_load(open(_ecfg_path, encoding="utf-8"))
                    _kdcfg = _kcfg.get("data", {})
                    _kcfg_dir = _ecfg_path.parent
                    _kdata_dir = str((_kcfg_dir / _kdcfg.get("data_dir", "")).resolve())
                    _ksample_sub = _kdcfg.get("sample_subdir", "samples")
                    _koutput_cols = _kdcfg.get("output_columns", [])
                    _kin_r1 = _kdcfg.get("input_row_1", 0)
                    _kin_r2 = _kdcfg.get("input_row_2", 96)
                    _kout_rows = _kdcfg.get("output_rows", -1)
                    _kin_agg = str(_kdcfg.get("input_aggregation", "none")).lower()

                    _candidate_input_cols = list(cand.features)
                    _k_selection_cfg = {
                        "use_mutual_info": True,
                        "use_lasso": False,
                        "deduplicate_threshold": 0.9999,
                        "vif_threshold": 10.0,
                    }

                    _kload_kw = dict(
                        data_dir=_kdata_dir, sample_subdir=_ksample_sub,
                        forecast_name=_kdcfg.get("forecast_name", ""),
                        input_columns=_candidate_input_cols, output_columns=_koutput_cols,
                        input_rows=slice(_kin_r1, _kin_r2), output_rows=_kout_rows,
                        split_source_dir=_vd, input_aggregation=_kin_agg,
                    )
                    _ktr = eval_module.load_split_samples(**_kload_kw, split_file="train_files.txt", fault_tolerant=True)
                    _kte = eval_module.load_split_samples(**_kload_kw, split_file="test_files.txt", fault_tolerant=True)
                    if len(_ktr) >= 3 and len(_kte) >= 1:
                        for _mlr_v in _MLR_VARIANTS:
                            try:
                                _kpreds, _ktgts, _ktr_rb, _kte_rb, _kmeta, _k_excl = _evaluate_mlr_with_rebalance(
                                    _ktr, _kte,
                                    feature_names=_candidate_input_cols,
                                    selection_config=_k_selection_cfg,
                                    aggregation_mode=_mlr_v["aggregation_mode"],
                                    min_test_independent=FINAL_TOPK_MIN_TEST_SAMPLES,
                                    model_name=f"{_mlr_v['model_name']} k{rank:02d}",
                                )
                                if len(_kpreds) >= 1:
                                    _kpf = _kpreds.flatten()
                                    _ktf = _ktgts.flatten()
                                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                                    _kn_sel = sum(m.get("n_selected", 0) for m in _kmeta) / max(len(_kmeta), 1)
                                    _kn = len(_kpf)
                                    rows.append({
                                        "dataset": dataset_plan.dataset_dir.name,
                                        "target": target_name,
                                        "subset_rank": rank,
                                        "subset_label": f"k{rank:02d}",
                                        "feature_tag": cand.feature_tag,
                                        "row_count": cand.row_count,
                                        "n_features": _kn_sel,
                                        "objective_search": cand.objective,
                                        "drop_rate_search": cand.drop_rate,
                                        "model": _mlr_v["model_name"],
                                        **dict(zip(("pred_std", "degenerate"),
                                                   _prediction_spread(np.asarray(_ktf, dtype=float).ravel(),
                                                                      np.asarray(_kpf, dtype=float).ravel()))),
                                        "gp_uncertainty_mode": "",
                                        "n_samples": _kn,
                                        "n_test_independent": _independent_name_count([str(s[2]) for s in _kte_rb]),
                                        "mae": float(mean_absolute_error(_ktf, _kpf)),
                                        "rmse": float(np.sqrt(mean_squared_error(_ktf, _kpf))),
                                        "r2": float(r2_score(_ktf, _kpf)),
                                        "pearson_r": float(np.corrcoef(_ktf, _kpf)[0, 1]),
                                        "std_target": float(np.std(_ktf, ddof=1)) if _kn > 1 else float("nan"),
                                        "n_test_valid": _kn,
                                        "n_test_evals": _kn,
                                        "input_dim": float(len(_candidate_input_cols)),
                                        "target_dim": float(len(_koutput_cols)),
                                    })
                                    print(f"[INFO] {_mlr_v['model_name']} on k{rank:02d}: R²={float(r2_score(_ktf, _kpf)):.3f}")

                                    _mlr_k_dir = _write_mlr_artifacts(
                                        output_dir=output_dir,
                                        dataset_dir=dataset_plan.dataset_dir,
                                        subset_label=f"k{rank:02d}",
                                        data_dir=_kdata_dir,
                                        sample_subdir=_ksample_sub,
                                        input_columns=_candidate_input_cols,
                                        output_columns=_koutput_cols,
                                        input_row_1=_kin_r1,
                                        input_row_2=_kin_r2,
                                        output_rows=_kout_rows,
                                        input_aggregation=_kin_agg,
                                        train_samples=_ktr_rb,
                                        test_samples=_kte_rb,
                                        preds=_kpreds,
                                        targets=_ktgts,
                                        per_target_meta=_kmeta,
                                        split_source_dir=_vd,
                                        ref_cfg=_kcfg,
                                        ref_cfg_path=_ecfg_path,
                                        ref_data_cfg=_kdcfg,
                                        model_config_extra={
                                            "candidate_features": list(cand.features),
                                            "feature_selection": dict(_k_selection_cfg),
                                        },
                                        model_prefix=_mlr_v["dir_prefix"],
                                    )
                                    print(f"[INFO] {_mlr_v['model_name']} k{rank:02d} artifacts written to {_mlr_k_dir}")
                                    _eval_registry[(cand.feature_tag, _mlr_v["model_name"])] = (_mlr_k_dir, rows[-1])
                            except Exception as _mlr_v_exc:
                                print(f"[WARN] {_mlr_v['model_name']} eval on k{rank:02d}: {_mlr_v_exc}")
                        _mlr_on_k_done = True
                except Exception as _mlr_k_exc:
                    print(f"[WARN] MLR eval on k{rank:02d} variant {_vd.name}: {_mlr_k_exc}")
                if _mlr_on_k_done:
                    break
            if not _mlr_on_k_done:
                print(f"[INFO] MLR not computed for k{rank:02d} (no suitable variant).")
        except Exception as _mlr_k_outer:
            print(f"[WARN] MLR on k{rank:02d} skipped: {_mlr_k_outer}")

    # --- MLR k-clusters: Spearman-filtered features evaluated by ALL models ---
    try:
        from utils.mlr import get_spearman_features as _get_spearman_features
        from utils.mlr import MLR_VARIANTS as _MLR_VARIANTS
        _mlr_done = False
        if selected:
            k01 = selected[0]
            _ref_tag = f"{k01.feature_tag}_k01"
            _ref_variant_dirs = sorted(output_dir.glob(f"*_r{k01.row_count:03d}_{_ref_tag}*"))

            # --- Phase A: load samples and determine Spearman feature sets for each MLR variant ---
            _spearman_by_variant: dict[str, tuple[list[str], dict]] = {}
            _ref_vd = None
            _ref_ecfg = None
            _ref_cfg = None
            _ref_dcfg = None
            _tr = None
            _te = None
            for _vd in _ref_variant_dirs:
                _ecfg = _vd / f"config_evaluate_{_vd.name}.yml"
                if not _ecfg.exists():
                    continue
                try:
                    _cfg = yaml.safe_load(open(_ecfg, encoding="utf-8"))
                    _dcfg = _cfg.get("data", {})
                    _cfg_dir = _ecfg.parent
                    _data_dir = str((_cfg_dir / _dcfg.get("data_dir", "")).resolve())
                    _sample_sub = _dcfg.get("sample_subdir", "samples")
                    _output_cols = _dcfg.get("output_columns", [])
                    _in_r1 = _dcfg.get("input_row_1", 0)
                    _in_r2 = _dcfg.get("input_row_2", 96)
                    _out_rows = _dcfg.get("output_rows", -1)
                    _in_agg = str(_dcfg.get("input_aggregation", "none")).lower()
                    _split_dir = _vd

                    # Only the predictor list is wanted here, which every configuration
                    # of a dataset shares. Resolving by family name would raise as soon
                    # as a family has more than one window representation, and the
                    # enclosing `except Exception` would turn that into a silently
                    # skipped stage: the MLR-derived subsets l01/m01/s01 disappeared
                    # from a whole run that way.
                    _surrogate_path = _canonical_probe_config(dataset_plan.dataset_dir)
                    _surrogate_cfg = train_module.load_config(str(_surrogate_path))
                    _input_cols = list(_surrogate_cfg["data"]["input_columns"])

                    _load_kw = dict(
                        data_dir=_data_dir, sample_subdir=_sample_sub,
                        forecast_name=_dcfg.get("forecast_name", ""),
                        input_columns=_input_cols, output_columns=_output_cols,
                        input_rows=slice(_in_r1, _in_r2), output_rows=_out_rows,
                        split_source_dir=_split_dir, input_aggregation=_in_agg,
                    )
                    _tr = eval_module.load_split_samples(**_load_kw, split_file="train_files.txt", fault_tolerant=True)
                    _te = eval_module.load_split_samples(**_load_kw, split_file="test_files.txt", fault_tolerant=True)
                    if len(_tr) >= 3 and len(_te) >= 1:
                        for _fv in _MLR_VARIANTS:
                            try:
                                _sp_cols, _sp_results = _get_spearman_features(
                                    _tr, _input_cols, target_idx=0,
                                    aggregation_mode=_fv["aggregation_mode"])
                                _spearman_by_variant[_fv["model_name"]] = (_sp_cols, _sp_results)
                            except Exception as _sp_v_exc:
                                print(f"[WARN] Spearman pre-filter failed for {_fv['model_name']}: {_sp_v_exc}")
                        _ref_vd = _vd
                        _ref_ecfg = _ecfg
                        _ref_cfg = _cfg
                        _ref_dcfg = _dcfg
                        break
                except Exception as _sp_exc:
                    print(f"[WARN] Spearman pre-filter failed for {_vd.name}: {_sp_exc}")
                    print("[WARN] The MLR-derived subsets (l01, m01, s01) will be absent "
                          "for this target if no variant succeeds.")

            # --- Phase B & C: for each variant's feature set, evaluate all models + all MLR variants ---
            for _fv_idx, _fv in enumerate(_MLR_VARIANTS):
                _spearman_entry = _spearman_by_variant.get(_fv["model_name"])
                if not _spearman_entry:
                    continue
                _spearman_cols, _spearman_results = _spearman_entry
                if not _spearman_cols:
                    continue

                mlr_rank = len(selected) + 1 + _fv_idx
                mlr_feature_tag = _feature_tag(tuple(sorted(_spearman_cols)))
                mlr_subset_label = _fv["subset_label"]
                mlr_full_tag = f"{mlr_feature_tag}_{mlr_subset_label}"
                print(f"[INFO] {_fv['model_name']} Spearman pre-filter: {len(_spearman_cols)}/{len(_input_cols)} "
                      f"base columns pass -> {mlr_subset_label} ({mlr_feature_tag})")

                # Phase B: evaluate ALL data-driven (non-MLR) models on this feature set
                best_baseline_rows_mlr: dict[str, tuple[float, float, dict]] = {}
                for base_cfg in dataset_plan.train_configs:
                    variant_cfg = _prepare_variant_config(
                        base_config_path=base_cfg,
                        row_count=k01.row_count,
                        features=tuple(sorted(_spearman_cols)),
                        feature_tag=mlr_full_tag,
                        tmp_dir=cfg_dir,
                        forced_data_dir=dataset_plan.dataset_dir,
                    )
                    try:
                        # --- Dedup: skip training if this feature set was already evaluated for this model ---
                        try:
                            _base_model_type = yaml.safe_load(open(base_cfg, encoding="utf-8")).get("model_type", "")
                        except Exception:
                            _base_model_type = ""
                        _dedup_source = _eval_registry.get((mlr_feature_tag, _variant_key(base_cfg)))

                        if _dedup_source is not None:
                            _src_dir, _src_row = _dedup_source
                            _vc_data = yaml.safe_load(open(variant_cfg, encoding="utf-8"))
                            _vc_fn = _strip_fs_prefix(str(_vc_data["data"]["forecast_name"]))
                            _dest_dir = output_dir / Path(_vc_fn)
                            _copy_eval_directory(_src_dir, _dest_dir)
                            eval_cfg = (_dest_dir / f"config_evaluate_{_dest_dir.name}.yml").resolve()
                            print(f"[DEDUP] {_variant_key(base_cfg)} ({_base_model_type}) on "
                                  f"{mlr_subset_label}: copied from {_src_dir.name} "
                                  f"(same model, same features {mlr_feature_tag})")
                        else:
                            eval_cfg = _train_single_config(
                                variant_cfg,
                                dataset_plan.dataset_dir,
                                disable_training_plots=disable_training_plots,
                                disable_eval_plots=disable_eval_plots,
                                suppress_training_logs=suppress_training_logs,
                            )
                            skip_eval, split_status, train_n, test_n = _ensure_min_test_samples_for_final(
                                split_dir=eval_cfg.parent,
                                min_test_samples=FINAL_TOPK_MIN_TEST_SAMPLES,
                            )
                            if split_status == "rebalanced":
                                print(f"[INFO] Final split rebalance for {mlr_subset_label} cluster {eval_cfg.parent.name}: "
                                      f"train={train_n}, test={test_n}")
                            if skip_eval:
                                print(f"[WARN] {mlr_subset_label} cluster variant {eval_cfg.parent.name} has insufficient test samples; skipping.")
                                continue

                            _set_eval_overrides(eval_cfg, run_baselines=run_baselines_in_final)
                            eval_module.evaluate_single_config(
                                str(eval_cfg), save_plots_override=not disable_eval_plots)

                        # Extract rows from evaluation_summary.csv (same logic as main loop)
                        summary_rows_mlr = []
                        eval_summary_csv = Path(eval_cfg).parent / "evaluation_summary.csv"
                        if eval_summary_csv.exists():
                            try:
                                summary_df = pd.read_csv(eval_summary_csv)
                                primary_model_row = None
                                if not summary_df.empty:
                                    if "kind" in summary_df.columns:
                                        kinds = summary_df["kind"].astype(str).str.lower().str.strip()
                                        for preferred_kind in ("test", "combined", "train"):
                                            hit = summary_df[kinds == preferred_kind]
                                            if not hit.empty:
                                                primary_model_row = hit.iloc[0].to_dict()
                                                break
                                    if primary_model_row is None:
                                        for _, _row in summary_df.iterrows():
                                            if _normalize_baseline_label(_row.get("label", "")) is None:
                                                primary_model_row = _row.to_dict()
                                                break
                                if primary_model_row is not None:
                                    summary_rows_mlr.append(primary_model_row)
                                seen_baselines: set[str] = set()
                                for _, _row in summary_df.iterrows():
                                    baseline_id = _normalize_baseline_label(_row.get("label", ""))
                                    if baseline_id is None or baseline_id in seen_baselines:
                                        continue
                                    summary_rows_mlr.append(_row.to_dict())
                                    seen_baselines.add(baseline_id)
                            except Exception as read_exc:
                                print(f"[WARN] Could not read {mlr_subset_label} cluster eval summary: {read_exc}")

                        for srow in summary_rows_mlr:
                            _eval_cfg_data = {}
                            try:
                                with open(eval_cfg, 'r', encoding='utf-8') as f:
                                    _eval_cfg_data = yaml.safe_load(f) or {}
                            except Exception:
                                pass

                            model_name_default = (
                                _eval_cfg_data.get("model_name")
                                or _eval_cfg_data.get("model_type")
                                or srow.get("label", "unknown")
                            )
                            baseline_id = _normalize_baseline_label(srow.get("label", ""))
                            model_name = baseline_id if baseline_id is not None else model_name_default
                            gp_uncertainty_mode = str(srow.get("gp_uncertainty_mode", ""))

                            row_context = (
                                f"{mlr_subset_label}_cluster:{eval_cfg.parent.name} "
                                f"dataset={dataset_plan.dataset_dir.name} subset_rank={mlr_rank} model={model_name}"
                            )
                            try:
                                _validate_eval_metric_contract(srow, context=row_context)
                            except Exception:
                                continue
                            mae_val = _extract_required_independent_metric(srow, "mae", context=row_context)
                            rmse_val = _extract_required_independent_metric(srow, "rmse", context=row_context)
                            n_samples = _extract_required_independent_metric(srow, "n_test_independent", context=row_context)
                            r2_val = float(pd.to_numeric(srow.get("r2", np.nan), errors="coerce"))
                            pearson_val = float(pd.to_numeric(srow.get("pearson_r", np.nan), errors="coerce"))
                            n_test_independent = float(srow.get("n_test_independent", n_samples))
                            n_test_valid = float(srow.get("n_test_valid", np.nan))
                            n_test_evals = float(srow.get("n_test_evals", srow.get("n_eval_rows", np.nan)))

                            ec_input_cols = _eval_cfg_data.get("data", {}).get("input_columns") or []
                            input_dim = float(len(ec_input_cols)) if ec_input_cols else np.nan
                            target_dim = float(srow.get("n_eval_outputs", np.nan))

                            std_target = float('nan')
                            try:
                                target_cols = _eval_cfg_data.get('data', {}).get('output_columns', None)
                                _raw_data_dir = _eval_cfg_data.get('data', {}).get('data_dir', '')
                                _ec_cfg_dir = Path(eval_cfg).parent
                                ec_data_dir = str((_ec_cfg_dir / _raw_data_dir).resolve())
                                sample_subdir = _eval_cfg_data.get('data', {}).get('sample_subdir', 'samples')
                                sample_dir = os.path.join(ec_data_dir, sample_subdir)
                                csv_files = glob.glob(os.path.join(sample_dir, '*.csv'))
                                target_vals = []
                                for csvf in csv_files:
                                    try:
                                        df_csv = pd.read_csv(csvf)
                                        if target_cols and all(tc in df_csv.columns for tc in target_cols):
                                            for tc in target_cols:
                                                target_vals.extend(df_csv[tc].dropna().values.tolist())
                                    except Exception:
                                        continue
                                if target_vals:
                                    std_target = float(np.std(target_vals, ddof=1))
                            except Exception:
                                pass

                            row_payload = {
                                "dataset": dataset_plan.dataset_dir.name,
                                "target": target_name,
                                "subset_rank": mlr_rank,
                                "subset_label": mlr_subset_label,
                                "feature_tag": mlr_feature_tag,
                                "row_count": k01.row_count,
                                "n_features": len(_spearman_cols),
                                "objective_search": float("nan"),
                                "drop_rate_search": float("nan"),
                                "model": model_name,
                                **(dict(_EMPTY_VARIANT_FIELDS)
                                   if _is_baseline_model_value(model_name)
                                   else _model_variant_fields(base_cfg, eval_cfg.parent)),
                                **_degeneracy_fields(eval_cfg.parent),
                                "gp_uncertainty_mode": gp_uncertainty_mode,
                                "n_samples": n_samples,
                                "n_test_independent": n_test_independent,
                                "n_test_valid": n_test_valid,
                                "n_test_evals": n_test_evals,
                                "input_dim": input_dim,
                                "target_dim": target_dim,
                                "mae": mae_val,
                                "rmse": rmse_val,
                                "r2": r2_val,
                                "pearson_r": pearson_val,
                                "std_target": std_target,
                                "n_seeds_ensembled": int(srow.get("n_seeds_ensembled", 1) or 1),
                            }

                            if baseline_id is None:
                                rows.append(row_payload)
                                _eval_registry.setdefault(
                                    (mlr_feature_tag, _variant_key(base_cfg)),
                                    (eval_cfg.parent, row_payload))
                            else:
                                valid_score = n_test_valid if np.isfinite(n_test_valid) else float("-inf")
                                eval_score = n_test_evals if np.isfinite(n_test_evals) else float("-inf")
                                current = best_baseline_rows_mlr.get(baseline_id)
                                if current is None or (valid_score, eval_score) > (current[0], current[1]):
                                    best_baseline_rows_mlr[baseline_id] = (valid_score, eval_score, row_payload)

                    except SampleComplianceError as e:
                        ctx = getattr(e, "context", {}) or {}
                        print(f"[COMPLIANCE] {mlr_subset_label} cluster: {e.reason}: {e}. "
                              f"variant={ctx.get('variant_dir', str(variant_cfg))}")
                    except Exception as e:
                        print(f"[ERROR] {mlr_subset_label} cluster evaluation failed for {variant_cfg}: {e}")

                # Append deduplicated baseline rows for this cluster
                for baseline_id in ("naive", "seasonal", "linear"):
                    best = best_baseline_rows_mlr.get(baseline_id)
                    if best is not None:
                        rows.append(best[2])
                for baseline_id, (_, _, payload) in best_baseline_rows_mlr.items():
                    if baseline_id not in {"naive", "seasonal", "linear"}:
                        rows.append(payload)

                # Phase C: run ALL MLR model variants on this feature set
                try:
                    # Check which MLR variants need fresh evaluation vs dedup copy
                    _mlr_to_evaluate: list[dict] = []
                    _mlr_to_copy: list[tuple[dict, Path, dict]] = []  # (variant, source_dir, source_row)
                    for _mv in _MLR_VARIANTS:
                        _dedup_src = _eval_registry.get((mlr_feature_tag, _mv["model_name"]))
                        if _dedup_src is not None:
                            _mlr_to_copy.append((_mv, _dedup_src[0], _dedup_src[1]))
                        else:
                            _mlr_to_evaluate.append(_mv)

                    # Copy dedup'd MLR variants (directory + adjusted row)
                    for _mv, _src_dir, _src_row in _mlr_to_copy:
                        _dest_mlr_dir = _mlr_artifact_dir(output_dir, mlr_subset_label, model_prefix=_mv["dir_prefix"])
                        _copy_eval_directory(_src_dir, _dest_mlr_dir)
                        _copied_row = dict(_src_row)
                        _copied_row["subset_rank"] = mlr_rank
                        _copied_row["subset_label"] = mlr_subset_label
                        _copied_row["feature_tag"] = mlr_feature_tag
                        rows.append(_copied_row)
                        print(f"[DEDUP] {_mv['model_name']} on {mlr_subset_label}: "
                              f"copied from {_src_dir.name} (same features {mlr_feature_tag})")

                    # Evaluate remaining MLR variants that haven't been seen before
                    if _mlr_to_evaluate:
                        _mlr_variant_dirs = sorted(output_dir.glob(f"*_r{k01.row_count:03d}_{mlr_full_tag}*"))
                        _mlr_split_dir = _mlr_variant_dirs[0] if _mlr_variant_dirs else _ref_vd
                        _load_kw_mlr = dict(
                            data_dir=_data_dir, sample_subdir=_sample_sub,
                            forecast_name=_ref_dcfg.get("forecast_name", ""),
                            input_columns=list(_spearman_cols), output_columns=_output_cols,
                            input_rows=slice(_in_r1, _in_r2), output_rows=_out_rows,
                            split_source_dir=_mlr_split_dir, input_aggregation=_in_agg,
                        )
                        _tr_mlr = eval_module.load_split_samples(**_load_kw_mlr, split_file="train_files.txt", fault_tolerant=True)
                        _te_mlr = eval_module.load_split_samples(**_load_kw_mlr, split_file="test_files.txt", fault_tolerant=True)
                        if len(_tr_mlr) >= 3 and len(_te_mlr) >= 1:
                            _mlr_results = _run_mlr_variants_on_existing_split(
                                train_samples=_tr_mlr,
                                test_samples=_te_mlr,
                                feature_names=list(_spearman_cols),
                                min_test_independent=FINAL_TOPK_MIN_TEST_SAMPLES,
                                model_context=mlr_subset_label,
                                use_preselected_feature_set=True,
                            )
                            _result_by_name = {
                                str(_res.get("variant", {}).get("model_name", "")): _res
                                for _res in _mlr_results
                            }
                            for _mv in _mlr_to_evaluate:
                                _res = _result_by_name.get(_mv["model_name"], {})
                                if _res.get("error") is not None:
                                    print(f"[WARN] {_mv['model_name']} evaluation on {mlr_subset_label} failed: {_res['error']}")
                                    continue
                                _preds = _res.get("preds")
                                _tgts = _res.get("targets")
                                _tr_mlr_rb = _res.get("train_samples", _tr_mlr)
                                _te_mlr_rb = _res.get("test_samples", _te_mlr)
                                _meta = _res.get("meta", [])
                                if _res.get("n_samples", 0) >= 1:
                                    _n_sel = sum(m.get("n_selected", 0) for m in _meta) / max(len(_meta), 1)
                                    _mlr_row = {
                                        "dataset": dataset_plan.dataset_dir.name,
                                        "target": target_name,
                                        "subset_rank": mlr_rank,
                                        "subset_label": mlr_subset_label,
                                        "feature_tag": mlr_feature_tag,
                                        "row_count": k01.row_count,
                                        "n_features": _n_sel,
                                        "objective_search": float("nan"),
                                        "drop_rate_search": float("nan"),
                                        "model": _mv["model_name"],
                                        **dict(zip(("pred_std", "degenerate"),
                                                   _prediction_spread(
                                                       np.asarray(_res.get("targets", []), dtype=float).ravel(),
                                                       np.asarray(_res.get("preds", []), dtype=float).ravel()))),
                                        "gp_uncertainty_mode": "",
                                        "n_samples": _res["n_samples"],
                                        "n_test_independent": _res["n_test_independent"],
                                        "mae": _res["mae"],
                                        "rmse": _res["rmse"],
                                        "r2": _res["r2"],
                                        "pearson_r": _res["pearson_r"],
                                        "std_target": _res["std_target_empirical"],
                                        "n_test_valid": _res["n_test_valid"],
                                        "n_test_evals": _res["n_test_valid"],
                                        "input_dim": float(len(_spearman_cols)),
                                        "target_dim": float(len(_output_cols)),
                                    }
                                    rows.append(_mlr_row)
                                    print(f"[INFO] {_mv['model_name']} row appended for {dataset_plan.dataset_dir.name}: "
                                          f"R²={_res['r2']:.3f}, subset={mlr_subset_label}")
                                else:
                                    print(
                                        f"[WARN] {_mv['model_name']} produced no finite predictions for "
                                        f"{dataset_plan.dataset_dir.name} on {mlr_subset_label}; "
                                        f"failure_reasons={_res.get('failure_reasons') or ['<none>']} "
                                        f"fallback_modes={_res.get('fallback_modes') or ['<none>']}"
                                    )
                                    _mlr_row = None
                                _mlr_dir = _write_mlr_artifacts(
                                    output_dir=output_dir,
                                    dataset_dir=dataset_plan.dataset_dir,
                                    subset_label=mlr_subset_label,
                                    data_dir=_data_dir,
                                    sample_subdir=_sample_sub,
                                    input_columns=list(_spearman_cols),
                                    output_columns=_output_cols,
                                    input_row_1=_in_r1,
                                    input_row_2=_in_r2,
                                    output_rows=_out_rows,
                                    input_aggregation=_in_agg,
                                    train_samples=_tr_mlr_rb,
                                    test_samples=_te_mlr_rb,
                                    preds=_preds,
                                    targets=_tgts,
                                    per_target_meta=_meta,
                                    split_source_dir=_mlr_split_dir,
                                    ref_cfg=_ref_cfg,
                                    ref_cfg_path=_ref_ecfg,
                                    ref_data_cfg=_ref_dcfg,
                                    model_config_extra={
                                        "spearman_kept_columns": sorted(_spearman_cols),
                                        **(_res.get("feature_selection_extra") or {}),
                                    },
                                    model_prefix=_mv["dir_prefix"],
                                )
                                print(f"[INFO] {_mv['model_name']} artifacts written to {_mlr_dir}")
                                if _mlr_row is not None:
                                    _eval_registry.setdefault((mlr_feature_tag, _mv["model_name"]), (_mlr_dir, _mlr_row))
                except Exception as _mlr_model_exc:
                    print(f"[WARN] MLR model evaluation on {mlr_subset_label} failed: {_mlr_model_exc}")

                _mlr_done = True

            if not _spearman_by_variant:
                print(f"[INFO] MLR k-clusters not computed for {dataset_plan.dataset_dir.name} "
                      "(no suitable variant found for Spearman pre-filter).")
        if not _mlr_done and selected:
            print(f"[INFO] MLR k-clusters not added for {dataset_plan.dataset_dir.name}.")
    except Exception as _mlr_outer_exc:
        print(f"[WARN] MLR k-cluster integration skipped for {dataset_plan.dataset_dir.name}: {_mlr_outer_exc}")

    final_df = pd.DataFrame(rows)

    # Backfill NaN std_target and row_count using the first non-NaN value from the same
    # subset_rank. This corrects dedup-copied MLR rows whose values were not populated correctly.
    if "subset_rank" in final_df.columns:
        for _col in ("std_target", "row_count"):
            if _col not in final_df.columns:
                continue
            _vals = pd.to_numeric(final_df[_col], errors="coerce")
            _by_rank = final_df[_vals.notna()].groupby("subset_rank")[_col].first()
            _nan_mask = _vals.isna()
            if _nan_mask.any():
                final_df.loc[_nan_mask, _col] = final_df.loc[_nan_mask, "subset_rank"].map(_by_rank)

    # Compute RMSE-based minimum skill for each ML model row.
    if not final_df.empty and {"subset_rank", "rmse", "model"}.issubset(final_df.columns):
        _is_bl = final_df["model"].apply(_is_baseline_model_value)
        _bl_rmse = (
            final_df[_is_bl]
            .groupby("subset_rank")["rmse"]
            .min()
            .rename("_best_bl_rmse")
        )
        final_df = final_df.join(_bl_rmse, on="subset_rank")
        _ml_rmse = pd.to_numeric(final_df["rmse"], errors="coerce")
        final_df["min_skill_rmse"] = np.where(
            ~_is_bl & (final_df["_best_bl_rmse"] > 0),
            (final_df["_best_bl_rmse"] - _ml_rmse) / final_df["_best_bl_rmse"],
            np.nan,
        )
        final_df.drop(columns=["_best_bl_rmse"], inplace=True)

    out_csv = output_dir / "feature_sweep_final_metrics.csv"
    final_df.to_csv(out_csv, index=False)
    try:
        summary_plot = _plot_final_metrics_comparison(final_df, output_dir)
        print(f"[INFO] Wrote final metrics comparison plot: {summary_plot}")
    except Exception as exc:
        print(f"[WARN] Could not generate final metrics comparison plot for {dataset_plan.dataset_dir.name}: {exc}")
    return out_csv


def _run_rolling_origin_cv(
    plan: DatasetPlan,
    final_metrics_csv: Path,
    min_train_groups: int = 3,
) -> "Path | None":
    """
    Run rolling-origin (expanding-window) cross-validation for the best model type
    identified from the final evaluation metrics (highest R2 at subset_rank == 1).

    Groups samples by MC segment (temporal order) and trains a fresh model on the
    first N segments, evaluates on segment N+1, expanding the window each step.
    Writes rolling_origin_summary.csv to the model's forecast directory.
    """
    # --- Determine best model type from final metrics ---
    try:
        df_final = pd.read_csv(final_metrics_csv)
    except Exception as e:
        print(f"[WARN] Rolling origin CV: could not read final metrics: {e}")
        return

    k01_all = df_final[df_final["subset_rank"] == 1].copy()
    best_row = _select_best_model_by_min_skill_rmse(k01_all)
    if best_row is None:
        print("[WARN] Rolling origin CV: no baseline rows at subset_rank 1; cannot compute skill score. Skipping.")
        return

    best_model_str = str(best_row.get("model", ""))
    best_row_count = int(best_row["row_count"])
    best_feature_tag = str(best_row["feature_tag"])

    if not best_model_str:
        print("[WARN] Rolling origin CV: could not determine model type. Skipping.")
        return

    # --- Find base config matching best model type OR model name ---
    # The 'model' column may hold model_type (new sweeps) or model_name (old sweeps).
    # Variant dirs are named after data.forecast_name, not model_name.
    resolved_model_type = None
    best_model_name = None
    _matched_base_cfg = None

    # Prefer the recorded variant. Matching on the model type alone returns the first
    # configuration of that family, which for a Gaussian process means gp_01 whatever
    # was actually selected -- the substitution that turned a selected +0.577 into a
    # retrained -12.952. The variant column exists so this no longer has to guess.
    best_variant = str(best_row.get("variant", "") or "").strip()
    if best_variant:
        exact = [c for c in plan.train_configs
                 if c.stem.replace("config_", "") == best_variant]
        if exact:
            _cfg = train_module.load_config(str(exact[0]))
            _fn = (_cfg.get("data", {}).get("forecast_name")
                   or _cfg.get("model_name") or best_variant)
            resolved_model_type = str(_cfg.get("model_type", ""))
            best_model_name = _strip_fs_prefix(str(_fn))
            _matched_base_cfg = exact[0]
            print(f"[RollingOrigin] Resolved by recorded variant: {best_variant}")
        else:
            print(
                f"[WARN] Rolling origin CV: recorded variant '{best_variant}' has no "
                "matching config; falling back to model-type matching, which cannot "
                "distinguish variants of the same family."
            )

    if resolved_model_type is None:
        _match = _find_matching_config(best_model_str, plan.train_configs)
        if _match is not None:
            resolved_model_type, best_model_name, _matched_base_cfg = _match

    # Fall back to treating the raw value as model_type if no config was matched
    if resolved_model_type is None:
        resolved_model_type = best_model_str

    if resolved_model_type not in ("xgb_regressor", "gp_regressor"):
        print(f"[WARN] Rolling origin CV not implemented for model_type={resolved_model_type!r} (column value={best_model_str!r}). Skipping.")
        return

    if best_model_name is None:
        print(f"[WARN] Rolling origin CV: no base config found for '{best_model_str}'. Skipping.")
        return

    # --- Locate eval config (written by write_evaluation_config after training) ---
    # :03d is required to match directory names created by _variant_forecast_name
    variant_forecast_name = f"{best_model_name}_r{best_row_count:03d}_{best_feature_tag}_k01"
    model_dir = _forecast_sweeps_dir(plan.dataset_dir) / variant_forecast_name
    eval_cfg_path = model_dir / f"config_evaluate_{variant_forecast_name}.yml"
    local_train_cfg_path = model_dir / f"config_train_{variant_forecast_name}.yml"

    # Try eval config → local variant train config → base train config from plan
    _cfg_file_to_use = None
    _cfg_dir_for_paths = None
    if eval_cfg_path.exists():
        _cfg_file_to_use = eval_cfg_path
        _cfg_dir_for_paths = eval_cfg_path.parent
        print(f"[INFO] Rolling origin CV: using eval config: {eval_cfg_path.name}")
    elif local_train_cfg_path.exists():
        _cfg_file_to_use = local_train_cfg_path
        _cfg_dir_for_paths = local_train_cfg_path.parent
        print(f"[INFO] Rolling origin CV: eval config missing; using local train config: {local_train_cfg_path.name}")
    elif _matched_base_cfg is not None:
        _cfg_file_to_use = _matched_base_cfg
        _cfg_dir_for_paths = _cfg_file_to_use.parent
        print(f"[INFO] Rolling origin CV: no local config; falling back to base train config: {_cfg_file_to_use.name}")
    else:
        print(f"[WARN] Rolling origin CV: no config found for variant '{variant_forecast_name}'. Skipping.")
        return

    with open(_cfg_file_to_use, "r", encoding="utf-8") as _f:
        eval_cfg_dict = yaml.safe_load(_f) or {}

    # --- Resolve data paths (data_dir is relative to the config file's location) ---
    _data_cfg = eval_cfg_dict.get("data", {})
    _raw_data_dir = _data_cfg.get("data_dir", ".")
    data_dir_abs = str((_cfg_dir_for_paths / _raw_data_dir).resolve())
    sample_subdir = _data_cfg.get("sample_subdir", "samples")
    sample_dir = Path(data_dir_abs) / sample_subdir
    input_columns = _data_cfg.get("input_columns", [])
    output_columns = _data_cfg.get("output_columns", [])
    input_row_1 = int(_data_cfg.get("input_row_1", 0))
    input_row_2 = int(_data_cfg.get("input_row_2", 168))
    output_rows = _data_cfg.get("output_rows", [input_row_2])

    # --- Load all samples (fault_tolerant=True to maximise coverage) ---
    print(f"[INFO] Rolling origin CV: loading all samples from {sample_dir}")
    all_samples = load_samples(
        str(sample_dir),
        input_columns=input_columns,
        output_columns=output_columns,
        input_rows=slice(input_row_1, input_row_2),
        output_rows=output_rows,
        fault_tolerant=True,
    )
    if not all_samples:
        print(f"[WARN] Rolling origin CV: no samples loaded from {sample_dir}. Skipping.")
        return

    # --- Group by MC segment (temporal order) ---
    segment_groups = group_samples_by_segment(all_samples)
    groups = [sg for _, sg in segment_groups]
    group_labels = [seg for seg, _ in segment_groups]
    n_groups = len(groups)

    if n_groups < min_train_groups + 1:
        print(
            f"[WARN] Rolling origin CV: only {n_groups} segment group(s); "
            f"need at least {min_train_groups + 1}. Skipping."
        )
        return

    print(
        f"[INFO] Rolling origin CV: {n_groups} segment groups, "
        f"model={resolved_model_type}, min_train_groups={min_train_groups}"
    )

    # --- Rolling-origin folds ---
    fold_metrics = []
    for test_idx in range(min_train_groups, n_groups):
        train_samples_fold = [s for g in groups[:test_idx] for s in g]
        test_samples_fold = list(groups[test_idx])

        X_train = np.array([s[0].flatten() for s in train_samples_fold], dtype=np.float32)
        y_train = np.array([s[1].flatten() for s in train_samples_fold], dtype=np.float32)
        X_test = np.array([s[0].flatten() for s in test_samples_fold], dtype=np.float32)
        y_test = np.array([s[1].flatten() for s in test_samples_fold], dtype=np.float32)

        try:
            if resolved_model_type == "xgb_regressor":
                import xgboost as _xgb
                _model = _xgb.XGBRegressor()
                _model.fit(X_train, y_train)
                y_pred = _model.predict(X_test).reshape(y_test.shape)

            elif resolved_model_type == "gp_regressor":
                import gpytorch
                _X_tr = torch.tensor(X_train)
                _y_tr = torch.tensor(y_train[:, 0] if y_train.ndim > 1 else y_train)
                _lk = gpytorch.likelihoods.GaussianLikelihood()

                class _DummyGP(gpytorch.models.ExactGP):
                    def __init__(self, tx, ty, lk):
                        super().__init__(tx, ty, lk)
                        self.mean_module = gpytorch.means.ConstantMean()
                        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                    def forward(self, x):
                        return gpytorch.distributions.MultivariateNormal(
                            self.mean_module(x), self.covar_module(x)
                        )

                _gp = _DummyGP(_X_tr, _y_tr, _lk)
                _gp.train(); _lk.train()
                _opt = torch.optim.Adam(_gp.parameters(), lr=0.1)
                _mll = gpytorch.mlls.ExactMarginalLogLikelihood(_lk, _gp)
                for _ in range(30):
                    _opt.zero_grad()
                    _loss = -_mll(_gp(_X_tr), _y_tr)
                    _loss.backward()
                    _opt.step()
                _gp.eval(); _lk.eval()
                _X_te = torch.tensor(X_test)
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    y_pred = _lk(_gp(_X_te)).mean.numpy().reshape(-1, 1)
            else:
                # Should not reach here (checked above)
                return

        except Exception as fold_exc:
            print(f"[WARN] Rolling origin CV fold {test_idx} failed: {fold_exc}")
            continue

        errors = y_pred - y_test
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        fold_metrics.append({
            "fold": test_idx - min_train_groups + 1,
            "n_train_groups": test_idx,
            "n_test_groups": 1,
            "n_train_samples": len(train_samples_fold),
            "n_test_samples": len(test_samples_fold),
            "test_segment": group_labels[test_idx],
            "ss_res": ss_res,
            "ss_tot": ss_tot,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        })
        print(
            f"[INFO]   fold {test_idx - min_train_groups + 1}: "
            f"train_groups={test_idx} test_segment={group_labels[test_idx]} "
            f"rmse={rmse:.4f} r2={r2:.4f}"
        )

    if not fold_metrics:
        print("[WARN] Rolling origin CV: no folds completed.")
        return

    # --- Compute mean summary row ---
    finite_r2 = [m["r2"] for m in fold_metrics if np.isfinite(m["r2"])]
    fold_r2 = np.array([m["r2"] for m in fold_metrics], dtype=float)
    finite_fold_r2 = fold_r2[np.isfinite(fold_r2)]
    n_last = max(1, int(np.ceil(len(fold_metrics) * 0.5)))
    last_fold_metrics = fold_metrics[-n_last:]
    last_r2 = np.array([m["r2"] for m in last_fold_metrics], dtype=float)
    finite_last_r2 = last_r2[np.isfinite(last_r2)]
    ss_res_sum = float(np.sum([m["ss_res"] for m in fold_metrics if np.isfinite(m["ss_res"])]))
    ss_tot_sum = float(np.sum([m["ss_tot"] for m in fold_metrics if np.isfinite(m["ss_tot"])]))
    r2_pooled = float(1.0 - ss_res_sum / ss_tot_sum) if ss_tot_sum > 0 else float("nan")

    summary_row = {
        "fold": "mean",
        "n_train_groups": float("nan"),
        "n_test_groups": float("nan"),
        "n_train_samples": float("nan"),
        "n_test_samples": float("nan"),
        "test_segment": "all",
        "ss_res": ss_res_sum,
        "ss_tot": ss_tot_sum,
        "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "r2": float(np.mean(finite_r2)) if finite_r2 else float("nan"),
        "r2_median": float(np.median(finite_fold_r2)) if finite_fold_r2.size else float("nan"),
        "r2_last50_mean": float(np.mean(finite_last_r2)) if finite_last_r2.size else float("nan"),
        "r2_pooled": r2_pooled,
    }

    df_cv = pd.DataFrame([summary_row] + fold_metrics)
    model_dir.mkdir(parents=True, exist_ok=True)
    cv_summary_path = model_dir / "rolling_origin_summary.csv"
    df_cv.to_csv(cv_summary_path, index=False)

    try:
        df_plot = pd.DataFrame(fold_metrics).sort_values("n_train_samples")
        fig_cv_size, axes_cv_size = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        axes_cv_size[0].plot(df_plot["n_train_samples"], df_plot["r2"], marker="o", color="tab:blue")
        axes_cv_size[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes_cv_size[0].set_ylabel("R²")
        axes_cv_size[0].grid(alpha=0.3)
        axes_cv_size[1].plot(df_plot["n_train_samples"], df_plot["rmse"], marker="o", color="tab:red")
        axes_cv_size[1].set_ylabel("RMSE")
        axes_cv_size[1].grid(alpha=0.3)
        axes_cv_size[2].plot(df_plot["n_train_samples"], df_plot["mae"], marker="o", color="tab:green")
        axes_cv_size[2].set_ylabel("MAE")
        axes_cv_size[2].set_xlabel("Train Samples (n)")
        axes_cv_size[2].grid(alpha=0.3)
        plt.tight_layout()
        cv_size_plot_path = model_dir / "rolling_origin_vs_train_size.png"
        fig_cv_size.savefig(cv_size_plot_path, dpi=220, bbox_inches="tight")
        plt.close(fig_cv_size)
        print(f"[INFO] Rolling origin CV train-size plot written: {cv_size_plot_path}")
    except Exception as exc:
        print(f"[WARN] Could not write rolling_origin_vs_train_size.png: {exc}")

    print(
        f"[INFO] Rolling origin CV summary: rmse={summary_row['rmse']:.4f} "
        f"mae={summary_row['mae']:.4f} r2_mean={summary_row['r2']:.4f} "
        f"r2_median={summary_row['r2_median']:.4f} r2_last50={summary_row['r2_last50_mean']:.4f} "
        f"r2_pooled={summary_row['r2_pooled']:.4f}"
    )
    print(f"[INFO] Rolling origin CV written: {cv_summary_path}")
    return cv_summary_path


def _ensure_k01_baselines(plan: DatasetPlan, final_metrics_csv: Path) -> None:
    """Re-evaluate the best k01 model with run_baselines=True if evaluation_summary.csv
    is missing or contains no baseline rows.

    Safe to call repeatedly — exits immediately if baselines are already present.
    Requires the variant's config_evaluate_*.yml to already exist (i.e. the model was
    previously trained); does NOT re-train.
    """
    try:
        df_final = pd.read_csv(final_metrics_csv)
    except Exception as e:
        print(f"[WARN] _ensure_k01_baselines: could not read {final_metrics_csv}: {e}")
        return

    k01_all = df_final[df_final["subset_rank"] == 1].copy()
    best_row = _select_best_model_by_min_skill_rmse(k01_all)
    if best_row is None:
        print(f"[WARN] _ensure_k01_baselines: no baseline rows at subset_rank 1 for {plan.dataset_dir.name}; skipping.")
        return

    best_model_str = str(best_row.get("model", ""))
    _rc = pd.to_numeric(best_row.get("row_count", float("nan")), errors="coerce")
    if not np.isfinite(_rc):
        print(f"[WARN] _ensure_k01_baselines: row_count is NaN for best row in {plan.dataset_dir.name}; skipping.")
        return
    best_row_count = int(_rc)
    best_feature_tag = str(best_row["feature_tag"])

    _mlr_model_names = {"mlr", "mlr_avg12", "mlr_avgall"}
    if str(best_model_str).strip().lower() in _mlr_model_names:
        _best_mlr_prefix = str(best_model_str).strip().lower()
        if _best_mlr_prefix not in _mlr_model_names:
            _best_mlr_prefix = "mlr"
        variant_dir = _mlr_artifact_dir(_forecast_sweeps_dir(plan.dataset_dir), "k01", model_prefix=_best_mlr_prefix)
        eval_csv = variant_dir / "evaluation_summary.csv"
        if eval_csv.exists():
            try:
                df_eval = pd.read_csv(eval_csv)
                if not df_eval.empty and "label" in df_eval.columns:
                    labels = df_eval["label"].astype(str).str.lower()
                    has_all = all(labels.str.contains(name).any() for name in ("naive", "seasonal", "linear"))
                    if has_all:
                        print(f"[DEBUG] _ensure_k01_baselines: {variant_dir.name} already has Naive/Seasonal/Linear rows; skipping re-evaluation.")
                        return
            except Exception:
                pass
        print(f"[WARN] _ensure_k01_baselines: MLR k01 artifacts missing baseline rows at {eval_csv}; cannot auto re-evaluate without an eval config.")
        return

    # Match train config on model_type OR model_name
    best_model_name = None
    _match = _find_matching_config(best_model_str, plan.train_configs)
    if _match is not None:
        _, best_model_name, _ = _match

    if best_model_name is None:
        print(f"[WARN] _ensure_k01_baselines: no matching config for model '{best_model_str}' in {plan.dataset_dir.name}; skipping.")
        return

    variant_name = f"{best_model_name}_r{best_row_count:03d}_{best_feature_tag}_k01"
    variant_dir = _forecast_sweeps_dir(plan.dataset_dir) / variant_name
    eval_csv = variant_dir / "evaluation_summary.csv"
    eval_cfg_path = variant_dir / f"config_evaluate_{variant_name}.yml"

    # Check if all baseline rows are already present
    if eval_csv.exists():
        try:
            df_eval = pd.read_csv(eval_csv)
            if "label" in df_eval.columns:
                labels = df_eval["label"].astype(str).str.lower()
                has_all = all(labels.str.contains(name).any() for name in ("naive", "seasonal", "linear"))
                if has_all:
                    print(f"[DEBUG] _ensure_k01_baselines: {variant_name} already has Naive/Seasonal/Linear rows; skipping re-evaluation.")
                    return
        except Exception:
            pass

    if not eval_cfg_path.exists():
        print(f"[WARN] _ensure_k01_baselines: eval config not found at {eval_cfg_path}; cannot re-evaluate without re-training.")
        return

    print(f"[INFO] _ensure_k01_baselines: re-evaluating {variant_name} with baselines for {plan.dataset_dir.name}")
    try:
        _set_eval_overrides(eval_cfg_path, run_baselines=True)
        eval_module.evaluate_single_config(str(eval_cfg_path), save_plots_override=False)
        print(f"[INFO] _ensure_k01_baselines: re-evaluation complete for {plan.dataset_dir.name}")
    except Exception as exc:
        print(f"[WARN] _ensure_k01_baselines: re-evaluation failed for {plan.dataset_dir.name}: {exc}")
        import traceback as _tb
        _tb.print_exc()


def _write_dataset_evaluation_summary(plan: DatasetPlan, final_metrics_csv: Path) -> "Path | None":
    """Copy the best k01 model's evaluation_summary.csv (which includes baseline rows)
    to the dataset root as evaluation_summary.csv.

    This enables i2_PostProcess.py to find baseline performance stats without having
    to navigate into the feature_sweeps subdirectories.
    """
    print(f"[INFO] _write_dataset_evaluation_summary: processing {plan.dataset_dir.name}")

    try:
        df_final = pd.read_csv(final_metrics_csv)
    except Exception as e:
        print(f"[WARN] _write_dataset_evaluation_summary: could not read {final_metrics_csv}: {e}")
        return None

    k01_all = df_final[df_final["subset_rank"] == 1].copy()
    best_row = _select_best_model_by_min_skill_rmse(k01_all)
    if best_row is None:
        print(f"[WARN] _write_dataset_evaluation_summary: no baseline rows at subset_rank 1 for {plan.dataset_dir.name}; skipping.")
        return None

    best_model_type = str(best_row.get("model", ""))
    _rc = pd.to_numeric(best_row.get("row_count", float("nan")), errors="coerce")
    if not np.isfinite(_rc):
        print(f"[WARN] _write_dataset_evaluation_summary: row_count is NaN for best row in {plan.dataset_dir.name}; skipping.")
        return None
    best_row_count = int(_rc)
    best_feature_tag = str(best_row["feature_tag"])

    _mlr_model_names_wr = {"mlr", "mlr_avg12", "mlr_avgall"}
    if str(best_model_type).strip().lower() in _mlr_model_names_wr:
        _wr_prefix = str(best_model_type).strip().lower()
        if _wr_prefix not in _mlr_model_names_wr:
            _wr_prefix = "mlr"
        src_csv = _mlr_artifact_dir(_forecast_sweeps_dir(plan.dataset_dir), "k01", model_prefix=_wr_prefix) / "evaluation_summary.csv"
        if not src_csv.exists():
            print(f"[WARN] _write_dataset_evaluation_summary: evaluation_summary.csv not found at {src_csv}")
            return None

        dst_csv = plan.dataset_dir / "evaluation_summary.csv"
        try:
            pd.read_csv(src_csv).to_csv(dst_csv, index=False)
            print(f"[INFO] _write_dataset_evaluation_summary: wrote {dst_csv}")
            return dst_csv
        except Exception as exc:
            print(f"[WARN] _write_dataset_evaluation_summary: failed to write {dst_csv}: {exc}")
            return None

    # Find base forecast_name from train configs by matching model_type OR model_name.
    # The 'model' column may hold model_type (new sweeps) or model_name (old sweeps).
    # Variant dirs are named after data.forecast_name, NOT model_name.
    base_model_name = None
    _match = _find_matching_config(best_model_type, plan.train_configs)
    if _match is not None:
        _, base_model_name, _ = _match

    if base_model_name is None:
        print(
            f"[WARN] _write_dataset_evaluation_summary: no base config found for "
            f"model_type={best_model_type!r} in {plan.dataset_dir.name}"
        )
        return None

    # Construct variant directory name (must use :03d to match _variant_forecast_name)
    variant_name = f"{base_model_name}_r{best_row_count:03d}_{best_feature_tag}_k01"
    variant_dir = _forecast_sweeps_dir(plan.dataset_dir) / variant_name
    src_csv = variant_dir / "evaluation_summary.csv"

    if not src_csv.exists():
        print(f"[WARN] _write_dataset_evaluation_summary: evaluation_summary.csv not found at {src_csv}")
        return None

    dst_csv = plan.dataset_dir / "evaluation_summary.csv"
    try:
        pd.read_csv(src_csv).to_csv(dst_csv, index=False)
        print(f"[INFO] _write_dataset_evaluation_summary: wrote {dst_csv}")
        return dst_csv
    except Exception as exc:
        print(f"[WARN] _write_dataset_evaluation_summary: failed to write {dst_csv}: {exc}")
        return None


def run_feature_selection_sweep(args: argparse.Namespace) -> int:
    # Set before any dataset is touched, and in the environment so that the parallel
    # candidate evaluators inherit it instead of re-importing the default.
    os.environ["WQ_PIN_SPLIT"] = "0" if getattr(args, "no_pin_split", False) else "1"
    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()

    explicit_seed_csv: Path | None = None
    if args.seed_subsets_csv:
        explicit_seed_csv = Path(args.seed_subsets_csv)
        if not explicit_seed_csv.is_absolute():
            explicit_seed_csv = (workspace_root / explicit_seed_csv).resolve()


    single_plan = getattr(args, "_internal_single_plan", None)
    if single_plan is not None:
        plans = [single_plan]
    else:
        include_regular, include_res = _resolve_dataset_inclusion(args)
        plans = discover_mc_dataset_plans(
            data_root=data_root,
            dataset_prefix=args.dataset_prefix,
            config_pattern=args.config_pattern,
            limit_datasets=args.limit_datasets,
            include_regular=include_regular,
            include_res=include_res,
        )

    if getattr(args, "notify", False):
        notify(
            title="h_RunMCFeatureSelectionSweep \u2014 Starting",
            message=(
                f"Datasets: {len(plans)}\n"
                f"Data root: {data_root.name}\n"
                f"Row counts: {getattr(args, 'row_counts', 'default')}"
            ),
        )

    if args.run_baselines_in_search and args.disable_baselines_for_search:
        raise ValueError("Cannot use both --run-baselines-in-search and --disable-baselines-for-search.")
    if args.seed_subsets_from_shapley and explicit_seed_csv is not None:
        raise ValueError("Cannot use both --seed-subsets-from-shapley and --seed-subsets-csv.")

    # Search is performance-oriented by default; final phase restores outputs.
    search_run_baselines = bool(args.run_baselines_in_search) and (not args.disable_baselines_for_search)
    search_disable_training_plots = not bool(args.keep_training_plots)
    search_disable_eval_plots = not bool(args.keep_eval_plots)
    search_save_plots = bool(args.keep_search_plots)
    final_run_baselines = True
    final_disable_training_plots = False
    final_disable_eval_plots = False

    # --exclude logic removed
    if not plans:
        print("No matching datasets/configs found.")
        return 1

    for plan in plans:
        if not args.no_pin_split:
            _pin_all_sample_subdirs(plan, args.surrogate_model)
        _ensure_feature_sweep_cache(plan, args.surrogate_model)

    print("\nExecution plan")
    print("-" * 100)
    print(f"Data root                 : {data_root}")
    print(f"Dataset prefix            : {args.dataset_prefix}")
    print(f"Config pattern            : {args.config_pattern}")
    print(f"Datasets found            : {len(plans)}")
    print(f"Beam width                : {args.beam_width}")
    print(f"Max rounds                : {args.max_rounds}")
    print(f"Patience                  : {args.no_improve_patience}")
    print(f"Eval budget               : {args.eval_budget}")
    print(f"Swap attempts             : {args.max_swap_attempts}")
    print(f"Lambda drop               : {args.lambda_drop}")
    print(f"Rolling-origin CV folds   : {args.cv_folds}"
          + ("" if args.cv_folds > 0
             else "  (candidates scored on the reported holdout)"))
    print(f"CV initial train fraction : {args.cv_min_train_fraction}")
    print(f"One-SE selection band     : {args.selection_tolerance_se} SE")
    print(f"Retention band            : {args.retention_tolerance}")
    print(f"Surrogate family          : {args.surrogate_model}")
    print(f"Pinned split per target   : {not args.no_pin_split}"
          + ("" if not args.no_pin_split else "  (per-run boundaries; they drift)"))
    if int(args.cv_folds) > 1:
        fits = int(args.eval_budget) * int(args.cv_folds)
        print(
            f"Model fits per target     : up to {fits} "
            f"({args.eval_budget} candidates x {args.cv_folds} folds)"
        )
        if int(args.eval_budget) >= 240:
            print(
                "[NOTE] --eval-budget is at its single-holdout default while "
                f"--cv-folds={args.cv_folds}. Cross-validation buys a separable "
                "objective, not more candidates: a smaller budget against it beats a "
                "large one against a single holdout. Consider --eval-budget 80."
            )
    print(f"Top-K for final models    : {args.final_top_k}")
    print(f"Dry run                   : {args.dry_run}")
    print(f"Keep train plots (search) : {args.keep_training_plots}")
    print(f"Keep eval plots (search)  : {args.keep_eval_plots}")
    print(f"Keep search plots         : {args.keep_search_plots}")
    print(f"Show train logs           : {args.show_training_logs}")
    print(f"Search run baselines      : {search_run_baselines}")
    print(f"Search train plots enabled: {not search_disable_training_plots}")
    print(f"Search eval plots enabled : {not search_disable_eval_plots}")
    print(f"Search summary plots      : {search_save_plots}")
    parallel_evaluators = max(1, int(getattr(args, "parallel_evaluators", 1)))
    print(f"Parallel evaluators       : {parallel_evaluators}")
    print(f"Final run baselines       : {final_run_baselines}")
    print(f"Final eval plots enabled  : {not final_disable_eval_plots}")
    print(f"Seed subsets from Shapley : {args.seed_subsets_from_shapley}")
    print(f"Seed subsets CSV          : {explicit_seed_csv}")
    print(f"Max seed subsets          : {args.max_seed_subsets}")

    if args.dry_run:
        for plan in plans:
            surrogate = _select_surrogate_config(plan.train_configs, args.surrogate_model)
            cfg = train_module.load_config(str(surrogate))
            base_span = int(cfg["data"]["input_row_2"]) - int(cfg["data"]["input_row_1"])
            row_counts = _parse_row_counts(args.row_counts, default_span=base_span)
            print(f"  - {plan.dataset_dir.name}: surrogate={surrogate.name}, row_counts={row_counts}")
        return 0

    failed = 0
    for plan in plans:
        print("\n" + "=" * 100)
        print(f"DATASET: {plan.dataset_dir.name}")
        print("=" * 100)

        surrogate_cfg = _select_surrogate_config(plan.train_configs, args.surrogate_model)
        chosen_folds = None
        surrogate_data = train_module.load_config(str(surrogate_cfg))["data"]
        base_span = int(surrogate_data["input_row_2"]) - int(surrogate_data["input_row_1"])
        row_counts = _parse_row_counts(args.row_counts, default_span=base_span)
        include_row_count_in_plot_names = len(row_counts) > 1

        for row_count in row_counts:
            try:
                # Not naming a surrogate here: `surrogate_cfg` is still the stand-in
                # from `_select_surrogate_config`, and the one that actually scores the
                # search is measured a few lines down. With `auto:xgb` the stand-in was
                # usually an XGBoost config and the line looked right by coincidence;
                # under plain `auto` it printed config_gp_01 while config_gp_03 did the
                # scoring. A log naming the wrong surrogate is exactly what later has to
                # be reverse-engineered.
                print(f"\n[SEARCH] rows={row_count}")
                seeded_subsets = _load_seed_subsets(
                    dataset_dir=plan.dataset_dir,
                    row_count=row_count,
                    explicit_path=explicit_seed_csv,
                    from_shapley=bool(args.seed_subsets_from_shapley),
                    max_seed_subsets=int(args.max_seed_subsets),
                )
                surrogate_cfg, chosen_folds = _choose_surrogate_config(
                    dataset_dir=plan.dataset_dir,
                    dataset_prefix=args.dataset_prefix,
                    train_configs=plan.train_configs,
                    row_count=row_count,
                    surrogate_model=args.surrogate_model,
                    lambda_drop=args.lambda_drop,
                    cv_folds=args.cv_folds,
                    cv_min_train_fraction=args.cv_min_train_fraction,
                    disable_baselines_for_search=not search_run_baselines,
                    disable_training_plots=search_disable_training_plots,
                    disable_eval_plots=search_disable_eval_plots,
                    suppress_training_logs=not args.show_training_logs,
                )
                print(f"[SEARCH] rows={row_count} surrogate={surrogate_cfg.name} "
                      f"(scores every candidate in this search)")
                top_sorted, trace, _ = _beam_search_subsets(
                    dataset_dir=plan.dataset_dir,
                    dataset_prefix=args.dataset_prefix,
                    surrogate_config_path=surrogate_cfg,
                    row_count=row_count,
                    lambda_drop=args.lambda_drop,
                    beam_width=args.beam_width,
                    max_rounds=args.max_rounds,
                    no_improve_patience=args.no_improve_patience,
                    min_features=args.min_features,
                    eval_budget=args.eval_budget,
                    max_swap_attempts=args.max_swap_attempts,
                    disable_baselines_for_search=not search_run_baselines,
                    disable_training_plots=search_disable_training_plots,
                    disable_eval_plots=search_disable_eval_plots,
                    suppress_training_logs=not args.show_training_logs,
                    seed=args.seed,
                    save_search_plots=search_save_plots,
                    parallel_evaluators=parallel_evaluators,
                    include_row_count_in_plot_names=include_row_count_in_plot_names,
                    seeded_subsets=seeded_subsets,
                    cv_folds=args.cv_folds,
                    cv_min_train_fraction=args.cv_min_train_fraction,
                    selection_tolerance_se=args.selection_tolerance_se,
                    retention_tolerance=args.retention_tolerance,
                    cv_fold_dirs=chosen_folds,
                )
                selected = top_sorted[: args.final_top_k]
                trace_csv, selected_csv, plot_path = _write_search_outputs(
                    dataset_dir=plan.dataset_dir,
                    row_count=row_count,
                    trace=trace,
                    selected=selected,
                    save_plots=search_save_plots,
                )
                print(f"[INFO] Wrote search trace: {trace_csv}")
                print(f"[INFO] Wrote selected subsets: {selected_csv}")
                if search_save_plots:
                    print(f"[INFO] Wrote search plot: {plot_path}")
                else:
                    print("[INFO] Search Pareto plot disabled by default (use --keep-search-plots to enable).")

                final_metrics_csv = _evaluate_selected_subsets_all_models(
                    dataset_plan=plan,
                    dataset_prefix=args.dataset_prefix,
                    selected=selected,
                    run_baselines_in_final=final_run_baselines,
                    disable_training_plots=final_disable_training_plots,
                    disable_eval_plots=final_disable_eval_plots,
                    suppress_training_logs=not args.show_training_logs,
                )
                print(f"[INFO] Wrote final model metrics: {final_metrics_csv}")

            except SampleComplianceError as exc:
                failed += 1
                ctx = getattr(exc, "context", {}) or {}
                print(f"[COMPLIANCE] Dataset failed: {plan.dataset_dir.name}, row_count {row_count}")
                print(
                    f"[COMPLIANCE] reason={exc.reason} message={exc} "
                    f"variant={ctx.get('variant_dir', 'n/a')}"
                )
                if args.stop_on_error:
                    raise
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

    rc = 0 if failed == 0 else 2
    if getattr(args, "notify", False):
        status = "Complete" if rc == 0 else f"Failed ({failed} dataset(s))"
        notify(
            title=f"h_RunMCFeatureSelectionSweep \u2014 {status}",
            message=(
                f"Datasets completed: {len(plans) - failed}\n"
                f"Datasets failed: {failed}\n"
                f"Exit code: {rc}"
            ),
        )
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature-selection sweeper using beam+swap surrogate search and final full-model evaluation."
    )
    parser.add_argument("--data-root", type=str, default="data/output/regression")
    parser.add_argument("--dataset-prefix", type=str, default="MC")
    parser.add_argument("--config-pattern", type=str, default="config_*.yml")
    parser.add_argument("--limit-datasets", type=int, default=1)

    parser.add_argument("--row-counts", type=str, default=None)
    parser.add_argument("--min-features", type=int, default=4)
    parser.add_argument(
        "--beam-width",
        type=int,
        default=6,
        help=(
            "Candidates kept each elimination round. Against 11 predictors a width of 8 "
            "was already near-exhaustive at the first level; extra width buys more "
            "chances to fit the holdout, not a better subset."
        ),
    )
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--no-improve-patience", type=int, default=3)
    parser.add_argument("--eval-budget", type=int, default=240)
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Fit each stochastic candidate this many times with different model seeds. The "
             "beam search and surrogate choice select on the mean score; the final per-family "
             "re-fit -- the run z8 scores and the results table quotes -- installs the mean "
             "PREDICTION across seeds, so its R2, skill score and significance verdict all "
             "describe one model. Replicates are suffixed _seedNN and are excluded from "
             "scoring as candidates. Stochastic means XGBoost, the transformer, and the GP "
             "when its uncertain-input kernel is enabled; MLR and the plain GP are fitted "
             "once. Six seeds of the CV22 winners give an R2 standard deviation of 0.03 "
             "(median) and up to 0.44, and three of five XGBoost wins do not survive it, "
             "so a single draw is not a safe basis for choosing between close candidates. "
             "Default 1 reproduces the previous behaviour.")
    parser.add_argument(
        "--seed-base", type=int, default=0,
        help="First seed used by --seeds (default 0, which is XGBoost's own default and "
             "therefore reproduces a single-seed run as its first replicate).")
    parser.add_argument("--max-swap-attempts", type=int, default=60)
    parser.add_argument("--lambda-drop", type=float, default=0.25)
    parser.add_argument("--final-top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help=(
            "Rolling-origin folds used to score each candidate, built inside the "
            "training portion so the search never reads a reported-test segment. "
            "Default 0 keeps the established behaviour: candidates are scored on the "
            "same holdout the results table reports, which is what lets every "
            "candidate train on the full 70% and converge, at the cost of an "
            "optimistically biased reported accuracy. Raising it is only worthwhile "
            "where the training portion is large enough that the earliest fold still "
            "fits on a representative share of the data -- with 30 training segments "
            "a 5-fold split fits the first model on 15."
        ),
    )
    parser.add_argument(
        "--cv-min-train-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of segments reserved as the initial training run, never scored. "
            "The rest is divided into --cv-folds contiguous blocks, each predicted from "
            "the history preceding it."
        ),
    )
    parser.add_argument(
        "--selection-tolerance-se",
        type=float,
        default=0.0,
        help=(
            "Width, in standard errors of the objective, of the band within which the "
            "smallest subset is preferred over the best-scoring one. Default 0 selects "
            "the argmin. Turning this on is not recommended at these sample sizes: the "
            "standard error of the objective is 0.26 to 7.8 across the 14 targets while "
            "the best few subsets differ by 0.002 to 0.16, so any honest band contains "
            "nearly every candidate and the rule collapses to picking the smallest "
            "subset allowed. The standard error is written to the trace as "
            "objective_se, where it is useful as a statement of how little the search "
            "resolved -- which is what feature_retention_frequency_r###.csv reports."
        ),
    )
    parser.add_argument(
        "--retention-tolerance",
        type=float,
        default=0.02,
        help=(
            "Objective band defining the near-optimal set whose per-feature retention "
            "frequency is written to feature_retention_frequency_r###.csv."
        ),
    )
    parser.add_argument(
        "--surrogate-model",
        type=str,
        default="auto:xgb",
        help=(
            "Which family scores candidates during the search. A substring matched "
            "against the config filenames; it must match exactly one configuration or "
            "the run stops, so 'gp' is rejected while 'gp_04' is accepted -- the four "
            "GP configs differ by window aggregation and kernel, and picking among "
            "them by name is what made one score -12.952 where another scored +0.577. "
            "'auto' instead fits every eligible configuration once on the full feature "
            "set and picks the best under the search objective. The chosen family "
            "fixes the feature set that all families then use, so where the reported "
            "winner is a different family that should be stated. "
            "The default is xgb_02, the daily-summary representation, measured against "
            "the alternatives on four targets: mean best R2 0.172 for xgb_02 against "
            "0.070 for xgb_01 (flattened) and 0.040 for xgb_03 (daily lag sampling), "
            "with the widest margin between competing subsets. The gain comes from the "
            "derived statistics rather than the smaller column count -- xgb_03 has "
            "fewer columns still and scores worst."
        ),
    )
    parser.add_argument(
        "--parallel-evaluators",
        type=int,
        default=1,
        help="Number of parallel candidate evaluators for beam rounds (1 keeps sequential behavior).",
    )
    parser.add_argument(
        "--seed-subsets-csv",
        type=str,
        default=None,
        help=(
            "Path to seed subset CSV (or directory containing "
            "feature_seed_subsets_r###_d########.csv with legacy fallback to "
            "feature_seed_subsets_r###.csv)."
        ),
    )
    parser.add_argument(
        "--seed-subsets-from-shapley",
        action="store_true",
        help=(
            "Load seed subsets from forecasts/Shapley_sweeps using root-isolated "
            "feature_seed_subsets_r###_d########.csv (legacy fallback supported)."
        ),
    )
    parser.add_argument(
        "--max-seed-subsets",
        type=int,
        default=0,
        help="Optional cap on loaded seed subsets (0 means no explicit cap).",
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
        help="Enable training plots during search-phase candidate evaluation (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-eval-plots",
        action="store_true",
        help="Enable evaluation plots during search-phase candidate evaluation (disabled by default for speed).",
    )
    parser.add_argument(
        "--keep-search-plots",
        action="store_true",
        help="Enable search-phase summary plots (Pareto and feature-importance plots), disabled by default for speed.",
    )
    parser.add_argument(
        "--show-training-logs",
        action="store_true",
        help="Show verbose model training logs (epoch metrics, sample-loading details).",
    )
    parser.add_argument(
        "--no-pin-split",
        action="store_true",
        help=(
            "Let every run recompute its own train/test boundary, as before. The "
            "boundary then depends on how many samples that run's feature subset can "
            "use, so it moves between runs -- for 8 of the 14 targets it does, by up "
            "to 6 segments -- and families are no longer scored on the same test set."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send a push notification on completion via ntfy.sh (requires NTFY_TOPIC env var).",
    )
    # --exclude argument removed
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    # Set once here rather than threaded through the search: the two call sites that
    # evaluate a candidate sit several frames down, and adding an argument to each of
    # those signatures is more surface than this needs. Parallel workers are handed the
    # value explicitly in their payload, since a spawned process inherits neither.
    global _CANDIDATE_SEEDS, _CANDIDATE_SEED_BASE
    _CANDIDATE_SEEDS = max(1, int(getattr(args, "seeds", 1) or 1))
    _CANDIDATE_SEED_BASE = int(getattr(args, "seed_base", 0) or 0)
    if _CANDIDATE_SEEDS > 1:
        print("[INFO] Seed averaging is on: %d seeds from %d. The search selects on the mean "
              "score; the final re-fit installs the mean prediction, so the reported result "
              "is a %d-seed ensemble." % (_CANDIDATE_SEEDS, _CANDIDATE_SEED_BASE,
                                          _CANDIDATE_SEEDS))
        print("[INFO] Replicates are suffixed _seedNN and excluded from candidate scoring. "
              "Each ensembled run keeps its single-seed predictions as predictions_seed0.csv.")

    # Written before the work starts, so a crashed run still records what it attempted,
    # and stamped with the outcome on the way out. The resolved values below are the ones
    # that were not obvious from the command line and had to be reverse-engineered later.
    manifest = provenance.write_manifest(
        Path(args.data_root), Path(__file__).name, args,
        extra={"candidate_seeds": _CANDIDATE_SEEDS,
               "candidate_seed_base": _CANDIDATE_SEED_BASE,
               "surrogate_scope": args.surrogate_model,
               "resolved_defaults": {
                   "beam_width": args.beam_width,
                   "eval_budget": args.eval_budget,
                   "max_rounds": args.max_rounds,
                   "no_improve_patience": args.no_improve_patience,
                   "min_features": args.min_features,
                   "final_top_k": args.final_top_k,
                   "cv_folds": args.cv_folds,
                   "selection_tolerance_se": args.selection_tolerance_se,
                   "shuffle_seed": args.seed,
               }})
    try:
        rc = run_feature_selection_sweep(args)
    except BaseException as exc:
        provenance.finalize_manifest(
            manifest, "failed", "%s: %s" % (type(exc).__name__,
                                            str(exc).splitlines()[0][:200] if str(exc) else ""))
        raise
    provenance.finalize_manifest(manifest, "completed" if rc == 0 else "nonzero_exit",
                                 "return code %s" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())

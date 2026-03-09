"""
Beam+swap feature-selection sweeper for MC datasets.

Search strategy:
- Surrogate-guided beam backward elimination with swap refinement.
- Objective: `objective = rmse + lambda_drop * drop_rate`.
- `drop_rate` is computed from raw sample coverage after MC replicate collapse.
- Search split behavior remains temporal-by-coverage (target 70/30 by default).

Then:
- Retrain/evaluate all discovered model configs on top-K subsets.
- Final top-K evaluations enforce a minimum of 5 test samples per subset by
    moving the latest samples from train -> test when needed; if fewer than 5
    total valid split samples exist, that subset/model evaluation is skipped.
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
    optional `feature_search_pareto_r###.png`, and `feature_sweep_final_metrics.csv`.
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
import sys
import time
import pandas as pd
import importlib.util
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


SUPPORTED_CONFIG_SUFFIXES = {".yml", ".yaml", ".json"}
FINAL_TOPK_MIN_TEST_SAMPLES = 5
BASELINE_MODEL_IDS = {"naive", "seasonal", "linear"}
FINAL_METRICS_MODEL_ORDER = [
    "gp_regressor",
    "transformer",
    "xgb_regressor",
    "naive",
    "seasonal",
    "linear",
]
FINAL_METRICS_MODEL_STYLE = {
    "gp_regressor": {"label": "GP", "color": "#1f77b4", "hatch": ""},
    "transformer": {"label": "Transformer", "color": "#ff7f0e", "hatch": ""},
    "xgb_regressor": {"label": "XGB", "color": "#2ca02c", "hatch": ""},
    "naive": {"label": "Naive", "color": "#7f7f7f", "hatch": "//"},
    "seasonal": {"label": "Seasonal", "color": "#17becf", "hatch": "//"},
    "linear": {"label": "Linear", "color": "#bcbd22", "hatch": "//"},
}

_EXPECTED_EVAL_METRIC_SEMANTICS = "independent_sample_primary"


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


def _sweep_namespace() -> str:
    """Return the forecast subdirectory name used by feature sweep artifacts."""
    raw = str(os.environ.get("WQ_FEATURE_SWEEP_NAMESPACE", "")).strip()
    return raw or "feature_sweeps"


def _forecast_sweeps_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "forecasts" / _sweep_namespace()


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
    else:
        print(f"[WARN] No valid seed subsets parsed from {seed_csv}; proceeding with unseeded search.")
    return out


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
                if "model_name" in cfg or "model_type" in cfg:
                    train_configs.append(path)
                else:
                    print(f"[WARN] Skipping config without model_name/model_type: {path}")
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
) -> Path:
    cfg = train_module.load_config(str(base_config_path))
    cfg_copy = copy.deepcopy(cfg)

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
    cfg_copy["evaluation"]["run_baselines"] = True

    cfg_copy.pop("__config_dir", None)

    variant_name = f"{base_config_path.stem}_r{row_count:03d}_{feature_tag}{base_config_path.suffix}"
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
            train_module.train_xgb_regressor_model(config, train_samples, test_samples)
        elif model_type == "xgb_classifier":
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


def _objective_from_metrics(rmse: float, drop_rate: float, lambda_drop: float) -> float:
    if not np.isfinite(rmse):
        return float("inf")
    if not np.isfinite(drop_rate):
        drop_rate = 1.0
    return float(rmse + lambda_drop * drop_rate)


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
) -> CandidateResult:
    try:
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

        variant_cfg = _prepare_variant_config(
            base_config_path=surrogate_config_path,
            row_count=row_count,
            features=features,
            feature_tag=feature_tag,
            tmp_dir=tmp_cfg_dir,
            forced_data_dir=dataset_dir,
        )
        eval_cfg = _train_single_config(
            variant_cfg,
            dataset_dir,
            disable_training_plots=disable_training_plots,
            disable_eval_plots=disable_eval_plots,
            suppress_training_logs=suppress_training_logs,
        )
        _set_eval_overrides(
            eval_cfg,
            run_baselines=not disable_baselines_for_search,
        )

        eval_result = eval_module.evaluate_single_config(
            str(eval_cfg),
            save_plots_override=not disable_eval_plots,
        )
        if eval_result is None:
            print(f"[ERROR] Evaluation returned None for config: {eval_cfg}")
            print(f"         Features: {features}")
            print(f"         Row count: {row_count}")
            print(f"         Data dir: {data_dir_resolved}")
            print(f"         Surrogate config: {surrogate_config_path}")
            return None
        model_row = eval_result

        context = f"{eval_cfg.parent.name} [{dataset_dir.name} r{int(row_count):03d} {feature_tag}]"
        _validate_eval_metric_contract(model_row, context=context)

        rmse = _extract_required_independent_metric(model_row, "rmse", context=context)
        mae = _extract_required_independent_metric(model_row, "mae", context=context)
        n_test_samples = _extract_required_independent_metric(model_row, "n_test_independent", context=context)
        r2 = float(pd.to_numeric(model_row.get("r2", np.nan), errors="coerce"))
        input_dim = float(model_row.get("input_dim", np.nan))
        target_dim = float(model_row.get("target_dim", np.nan))
        objective = _objective_from_metrics(rmse=rmse, drop_rate=drop_rate, lambda_drop=lambda_drop)

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
        )
    except Exception as exc:
        print(f"[ERROR] Exception in _evaluate_candidate for config: {surrogate_config_path}")
        print(f"        Features: {features}")
        print(f"        Row count: {row_count}")
        print(f"        Data dir: {dataset_dir}")
        import traceback
        traceback.print_exc()
        return None


def _candidate_key(row_count: int, features: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    return int(row_count), tuple(features)


def _plot_final_metrics_comparison(final_df: pd.DataFrame, output_dir: Path) -> Path:
    """Write a 4-panel clustered-bar comparison figure from final metrics rows.

    Panels: MAE, RMSE, Pearson's r, and R^2. Clusters are candidate subsets
    ordered by ascending RMSE for the model type with the best (lowest) subset RMSE.
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
    if df.empty:
        raise ValueError("Cannot plot final metrics summary: no recognized model rows found.")

    metric_cols = ["mae", "rmse", "pearson_r", "r2"]
    for metric in metric_cols:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    # Collapse potential duplicate rows (for example, baseline rows repeated per ML config).
    grouped = (
        df.groupby(["subset_rank", "model_norm"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )

    rank_to_label = {
        int(rank): f"k{int(rank):02d}"
        for rank in sorted(grouped["subset_rank"].dropna().unique().tolist())
    }

    rmse_pivot = grouped.pivot(index="subset_rank", columns="model_norm", values="rmse")
    finite_min_rmse = {}
    for model in FINAL_METRICS_MODEL_ORDER:
        if model not in rmse_pivot.columns:
            continue
        vals = pd.to_numeric(rmse_pivot[model], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            finite_min_rmse[model] = float(np.min(vals))

    if not finite_min_rmse:
        raise ValueError("Cannot plot final metrics summary: RMSE values are all non-finite.")

    best_model = min(finite_min_rmse.items(), key=lambda item: item[1])[0]

    subset_order = sorted(
        rank_to_label.keys(),
        key=lambda rank: (
            float(rmse_pivot.loc[rank, best_model])
            if (rank in rmse_pivot.index and best_model in rmse_pivot.columns and np.isfinite(rmse_pivot.loc[rank, best_model]))
            else float("inf"),
            int(rank),
        ),
    )

    x = np.arange(len(subset_order), dtype=float)
    n_models = len(FINAL_METRICS_MODEL_ORDER)
    cluster_width = 0.86
    bar_w = cluster_width / max(1, n_models)

    fig, axes = plt.subplots(4, 1, figsize=(max(10, 0.85 * len(subset_order) + 6), 14), sharex=True, constrained_layout=False)
    metric_specs = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("pearson_r", "Pearson's r"),
        ("r2", "R\N{SUPERSCRIPT TWO}"),
    ]

    legend_handles = []
    legend_labels = []
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        metric_pivot = grouped.pivot(index="subset_rank", columns="model_norm", values=metric)
        axis_vals: list[float] = []
        axis_bar_groups = []

        for i, model in enumerate(FINAL_METRICS_MODEL_ORDER):
            style = FINAL_METRICS_MODEL_STYLE[model]
            vals = []
            for rank in subset_order:
                if rank in metric_pivot.index and model in metric_pivot.columns:
                    val = metric_pivot.loc[rank, model]
                    vals.append(float(val) if np.isfinite(val) else np.nan)
                else:
                    vals.append(np.nan)

            xpos = x - (cluster_width / 2.0) + (i + 0.5) * bar_w
            bars = ax.bar(
                xpos,
                vals,
                width=bar_w,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="black" if style["hatch"] else "none",
                linewidth=0.8 if style["hatch"] else 0.0,
                label=style["label"],
            )
            axis_bar_groups.append((bars, vals))
            axis_vals.extend([float(v) for v in vals if np.isfinite(v)])
            if len(legend_handles) < n_models:
                legend_handles.append(bars[0])
                legend_labels.append(style["label"])

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
        else:
            ymin_base = float(np.min(finite_vals)) if finite_vals.size > 0 else -1.0
            ymax_base = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
            ax.set_ylim(ymin_base, ymax_base)

        if not axis_vals:
            continue

        y_low, y_high = ax.get_ylim()
        y_span = float(y_high - y_low) if np.isfinite(y_high - y_low) and (y_high - y_low) > 0 else 1.0
        text_pad = 0.02 * y_span
        edge_band = 0.03 * y_span

        for bars, vals in axis_bar_groups:
            for bar, val in zip(bars, vals):
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
                    clip_on=False,
                )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([rank_to_label[r] for r in subset_order], rotation=0)
    axes[-1].set_xlabel("Candidate Feature Subset")

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    fig.subplots_adjust(top=0.89, hspace=0.16)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=6,
        frameon=True,
        fontsize=9,
    )
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
    include_row_count_in_plot_names: bool = False,
    seeded_subsets: list[tuple[str, ...]] | None = None,
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

    rng = np.random.default_rng(seed)
    cache: dict[tuple[int, tuple[str, ...]], CandidateResult] = {}
    trace: list[CandidateResult] = []
    eval_count = 0
    search_start_time = time.time()
    
    # Feature importance tracking
    feature_removal_deltas: dict[str, list[float]] = {feat: [] for feat in full_features}
    feature_improvement_counts: dict[str, int] = {feat: 0 for feat in full_features}

    def _eval(features: tuple[str, ...]) -> CandidateResult | None:
        nonlocal eval_count
        key = _candidate_key(row_count, features)
        if key in cache:
            return cache[key]
        if eval_count >= eval_budget:
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
        )
        cache[key] = result
        trace.append(result)
        eval_count += 1
        return result

    first = _eval(full_features)
    if first is None:
        raise RuntimeError("Search budget exhausted before evaluating initial subset.")

    beam: list[CandidateResult] = [first]
    best = first
    if seeded_subsets:
        seeded_scored: list[CandidateResult] = []
        for raw_feats in seeded_subsets:
            filtered = [f for f in raw_feats if f in full_features]
            deduped = list(dict.fromkeys(filtered))
            if len(deduped) < min_features:
                continue
            ordered = tuple(sorted(deduped, key=lambda s: full_features.index(s)))
            out = _eval(ordered)
            if out is not None:
                seeded_scored.append(out)
        if seeded_scored:
            seeded_scored.sort(key=lambda x: (x.objective, x.rmse, -x.n_features))
            beam = sorted([first] + seeded_scored, key=lambda x: (x.objective, x.rmse, -x.n_features))[:beam_width]
            best = beam[0]
            print(f"[SEARCH] Seeded initialization: {len(seeded_scored)} subset(s) evaluated; best objective={best.objective:.4f} rmse={best.rmse:.6f} n_features={best.n_features}")

    no_improve = 0
    print(f"[SEARCH] Initial (all {len(full_features)} features): objective={best.objective:.4f} rmse={best.rmse:.6f} (evals: {eval_count}/{eval_budget}, ETA: {_format_eta(search_start_time, eval_count, eval_budget)})")

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
        for child in candidates:
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

        if not scored:
            print(f"[SEARCH] Round {_round + 1}: no scored candidates, stopping.")
            break

        scored.sort(key=lambda x: (x.objective, x.rmse, -x.n_features))
        beam = scored[:beam_width]
        prev_best = best.objective
        if beam and beam[0].objective + 1e-12 < best.objective:
            best = beam[0]
            no_improve = 0
            # Track features in improving solution (Option A)
            for feat in best.features:
                feature_improvement_counts[feat] += 1
            print(f"[SEARCH] Round {_round + 1}: improved! objective={best.objective:.4f} rmse={best.rmse:.6f} n_features={best.n_features} (evals: {eval_count}/{eval_budget}, ETA: {_format_eta(search_start_time, eval_count, eval_budget)})")
        else:
            no_improve += 1
            print(f"[SEARCH] Round {_round + 1}: no improvement ({no_improve}/{no_improve_patience}). Best: objective={best.objective:.4f} rmse={best.rmse:.6f} (evals: {eval_count}/{eval_budget}, ETA: {_format_eta(search_start_time, eval_count, eval_budget)})")
            if no_improve >= no_improve_patience:
                print(f"[SEARCH] Patience exhausted, stopping.")

    current = best
    all_features_set = set(full_features)
    attempts = 0
    improved = True
    swap_iter = 0
    print(f"[SEARCH] Starting swap refinement from: objective={current.objective:.4f} rmse={current.rmse:.6f} n_features={current.n_features} (ETA: {_format_eta(search_start_time, eval_count, eval_budget)})")
    
    while improved and attempts < max_swap_attempts and eval_count < eval_budget:
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
            if out.objective + 1e-12 < current.objective:
                swap_iter += 1
                current = out
                best = out
                improved = True
                print(f"[SEARCH] Swap refinement #{swap_iter}: improved! objective={best.objective:.4f} rmse={best.rmse:.6f} n_features={best.n_features} (evals: {eval_count}/{eval_budget}, ETA: {_format_eta(search_start_time, eval_count, eval_budget)})")
                break
    
    if not improved and eval_count < eval_budget:
        print(f"[SEARCH] Swap refinement: no improvements found (attempts: {attempts}/{max_swap_attempts}, evals: {eval_count}/{eval_budget})")

    top_sorted = sorted(trace, key=lambda x: (x.objective, x.rmse, -x.n_features))
    total_elapsed = time.time() - search_start_time
    elapsed_min = int(total_elapsed // 60)
    elapsed_sec = int(total_elapsed % 60)
    avg_time_per_eval = total_elapsed / eval_count if eval_count > 0 else 0
    print(f"[SEARCH] Complete: {eval_count}/{eval_budget} evaluations in {elapsed_min}m {elapsed_sec}s ({avg_time_per_eval:.1f}s/eval). Best: objective={best.objective:.4f} rmse={best.rmse:.6f} r2={best.r2:.6f} n_features={best.n_features}")
    
    # Compute average removal sensitivity for each feature
    feature_sensitivities: dict[str, tuple[float, int]] = {}  # (avg_removal_delta, frequency_count)
    for feat in full_features:
        deltas = feature_removal_deltas[feat]
        counts = feature_improvement_counts[feat]
        avg_delta = float(np.mean(deltas)) if deltas else 0.0
        feature_sensitivities[feat] = (avg_delta, counts)

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


def _select_surrogate_config(train_configs: list[Path]) -> Path:
    for cfg in train_configs:
        name = cfg.name.lower()
        if "xgb" in name and "classifier" not in name:
            return cfg
    return train_configs[0]


def _compile_multi_target_comparison(
    sweep_results: dict[str, dict],  # target -> {row_count -> feature_sensitivities}
    data_root: Path,
    importance_label: str = "Removal Sensitivity (avg delta)",
    summary_axis_label: str = "Total Removal Sensitivity (sum across targets)",
) -> Path:
    """Compile and visualize feature importance across multiple targets using removal sensitivity."""
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
    # Sort targets by their order in the CSV, if present, otherwise alphabetically
    # Use the same csv_path and header as above
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        res_targets = [col for col in header if col.endswith("_res")]
        sweep_keys = list(sweep_results.keys())
        matched_keys = []
        yticklabels = []
        used_keys = set()
        def norm(s):
            # Lowercase, replace µ->u, °->deg, remove all non-alphanumeric chars
            s = s.lower().replace('µ', 'u').replace('°', 'deg')
            s = unicodedata.normalize('NFKD', s)
            s = ''.join(c for c in s if not unicodedata.combining(c))
            s = re.sub(r'[^a-z0-9]', '', s)
            return s

        # Debug prints removed

        for csv_col in res_targets:
            # Prefer exact match
            if csv_col in sweep_results:
                matched_keys.append(csv_col)
                yticklabels.append(csv_col)
                used_keys.add(csv_col)
                continue
            # Otherwise, try suffix/unicode-normalized match
            matches = [k for k in sweep_keys if norm(k).endswith(norm(csv_col)) and k not in used_keys]
            if matches:
                matched_keys.append(matches[0])
                yticklabels.append(csv_col)
                used_keys.add(matches[0])
        # Add any sweep_results keys not already used, at the end
        extra_keys = [t for t in sweep_keys if t not in used_keys]
        matched_keys += extra_keys
        yticklabels += extra_keys
        targets = matched_keys
    except Exception:
        targets = sorted(sweep_results.keys())
        yticklabels = targets

    matrix = np.zeros((len(targets), len(all_features)))

    for i, target in enumerate(targets):
        for j, feat in enumerate(all_features):
            # Use the first (finest) row_count's data for comparison
            for feature_sensitivities in sweep_results[target].values():
                if feat in feature_sensitivities:
                    matrix[i, j] = feature_sensitivities[feat][0]  # removal sensitivity
                    break

    # Group feature order so multi-target features (common + partial) come first,
    # followed by strictly single-target features.
    feature_to_idx = {feat: idx for idx, feat in enumerate(all_features)}
    summed_sensitivity_raw = matrix.sum(axis=0)
    feature_total_score = {
        feat: float(summed_sensitivity_raw[idx]) for feat, idx in feature_to_idx.items()
    }
    n_targets = len(targets)
    presence_count = {
        feat: sum(1 for target in targets if feat in target_feature_sets.get(target, set()))
        for feat in all_features
    }

    multi_target_features = [feat for feat in all_features if presence_count.get(feat, 0) > 1]
    single_target_features = [feat for feat in all_features if presence_count.get(feat, 0) == 1]

    multi_target_features.sort(key=lambda feat: feature_total_score.get(feat, float("-inf")), reverse=True)
    single_target_features.sort(key=lambda feat: feature_total_score.get(feat, float("-inf")), reverse=True)

    ordered_features = multi_target_features + single_target_features
    if ordered_features:
        ordered_indices = [feature_to_idx[feat] for feat in ordered_features]
        matrix = matrix[:, ordered_indices]
        all_features = ordered_features

    ordered_feature_to_idx = {feat: idx for idx, feat in enumerate(all_features)}
    multi_target_features = [feat for feat in all_features if presence_count.get(feat, 0) > 1]
    single_target_features = [feat for feat in all_features if presence_count.get(feat, 0) == 1]
    multi_idx = [ordered_feature_to_idx[feat] for feat in multi_target_features]
    single_idx = [ordered_feature_to_idx[feat] for feat in single_target_features]
    max_group_len = max(len(multi_target_features), len(single_target_features), 1)
    n_total_features = max(len(all_features), 1)

    # Density-aware typography keeps dense plots readable without changing plot semantics.
    heat_xtick_font = max(5, min(8, int(200 / max_group_len)))
    heat_ytick_font = max(6, min(9, int(140 / max(len(targets), 1))))
    heat_axis_label_font = max(8, min(10, int(170 / max_group_len)))
    # Bar typography is tuned for document embedding where figures are often resized down.
    bar_tick_font = max(8, min(13, int(340 / max_group_len)))
    bar_title_font = max(12, min(16, int(420 / max_group_len)))
    bar_value_font = max(7, min(11, int(320 / max_group_len)))
    bar_axis_label_font = max(11, min(15, int(360 / max_group_len)))

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

    # Dynamically set font size based on grid density
    annot_fontsize = max(3, min(8, int(120 / max(len(all_features), len(targets), 1))))

    def _annotate_heat_cells(ax_obj, values: np.ndarray, fontsize: int) -> None:
        for row_i in range(values.shape[0]):
            for col_j in range(values.shape[1]):
                value = values[row_i, col_j]
                ax_obj.text(
                    col_j + 0.5,
                    row_i + 0.5,
                    f"{value:.2e}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=fontsize,
                    rotation=90,
                    clip_on=True,
                )

    if multi_idx and single_idx:
        heat_w = max(14, n_total_features * 0.55)
        heat_h = max(8, len(targets) * 0.58)
        fig, (ax_left, ax_right) = plt.subplots(
            1,
            2,
            figsize=(heat_w, heat_h),
            gridspec_kw={
                "width_ratios": [max(1, len(multi_target_features)), max(1, len(single_target_features))],
                "wspace": 0.08,
            },
            constrained_layout=True,
        )

        left_matrix = matrix[:, multi_idx]
        right_matrix = matrix[:, single_idx]

        sns.heatmap(
            left_matrix,
            ax=ax_left,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            annot=False,
            cbar=False,
            xticklabels=multi_target_features,
            yticklabels=yticklabels,
            linewidths=0.5,
            linecolor="#eeeeee",
            square=False,
        )
        sns.heatmap(
            right_matrix,
            ax=ax_right,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            annot=False,
            cbar=False,
            xticklabels=single_target_features,
            yticklabels=False,
            linewidths=0.5,
            linecolor="#eeeeee",
            square=False,
        )

        _annotate_heat_cells(ax_left, left_matrix, annot_fontsize)
        _annotate_heat_cells(ax_right, right_matrix, annot_fontsize)

        ax_left.set_xticklabels(multi_target_features, rotation=45, ha='right', fontsize=heat_xtick_font)
        ax_right.set_xticklabels(single_target_features, rotation=45, ha='right', fontsize=heat_xtick_font)
        ax_left.set_yticklabels(yticklabels, fontsize=heat_ytick_font)
        ax_left.set_xlabel(
            f"Multi-target Features (n={len(multi_target_features)})",
            fontsize=heat_axis_label_font,
        )
        ax_right.set_xlabel(
            f"Single-target Features (n={len(single_target_features)})",
            fontsize=heat_axis_label_font,
        )
        ax_left.set_ylabel("Target", fontsize=heat_axis_label_font)
        ax_right.set_ylabel("")

        sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap="RdYlGn")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_left, ax_right], fraction=0.025, pad=0.02)
        cbar.set_label(str(importance_label), fontsize=heat_axis_label_font)
        cbar.ax.tick_params(labelsize=max(6, heat_xtick_font))
    else:
        fig, ax = plt.subplots(
            figsize=(max(13, n_total_features * 0.5), max(8, len(targets) * 0.58)),
            constrained_layout=True,
        )
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            annot=False,
            cbar_kws={"label": str(importance_label)},
            xticklabels=all_features,
            yticklabels=yticklabels,
            linewidths=0.5,
            linecolor="#eeeeee",
            square=False,
        )
        _annotate_heat_cells(ax, matrix, annot_fontsize)
        ax.set_xticklabels(all_features, rotation=45, ha='right', fontsize=heat_xtick_font)
        ax.set_yticklabels(yticklabels, fontsize=heat_ytick_font)
        ax.set_xlabel("Feature", fontsize=heat_axis_label_font)
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
    
    # Create grouped bar charts: multi-target features and single-target features.
    summed_sensitivity = matrix.sum(axis=0)
    top_features = list(all_features)
    summed_scores = [float(v) for v in summed_sensitivity]

    score_map = {feat: float(score) for feat, score in zip(top_features, summed_scores)}
    multi_scores = [score_map[feat] for feat in multi_target_features if feat in score_map]
    single_scores = [score_map[feat] for feat in single_target_features if feat in score_map]
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

    def _draw_group_bars(ax_obj, features: list[str], values: list[float], title: str) -> None:
        x_vals = np.arange(len(features), dtype=float)
        bars = ax_obj.bar(x_vals, values, color=_bar_colors(values))
        ax_obj.set_title(title, fontsize=bar_title_font)
        ax_obj.set_xticks(x_vals)
        ax_obj.set_xticklabels(features, rotation=45, ha='right', fontsize=bar_tick_font)
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
                f"{val:.2e}",
                ha='center',
                va='center',
                fontsize=bar_value_font,
                rotation=90,
                clip_on=True,
            )
        ax_obj.set_ylim(y_lower, y_upper)

    if multi_target_features and single_target_features:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(max(15, len(top_features) * 0.58), max(10, 7 + 0.05 * len(top_features))),
            sharey=True,
            constrained_layout=True,
        )
        _draw_group_bars(
            ax_top,
            multi_target_features,
            multi_scores,
            f"Multi-target Features (n={len(multi_target_features)})",
        )
        _draw_group_bars(
            ax_bottom,
            single_target_features,
            single_scores,
            f"Single-target Features (n={len(single_target_features)})",
        )
        ax_top.set_ylabel(str(summary_axis_label), fontsize=bar_axis_label_font)
        ax_bottom.set_ylabel(str(summary_axis_label), fontsize=bar_axis_label_font)
        ax_bottom.set_xlabel("")
    else:
        fig, ax = plt.subplots(figsize=(max(15, len(top_features) * 0.58), 6.5), constrained_layout=True)
        _draw_group_bars(ax, top_features, summed_scores, "Feature Importance")
        ax.set_ylabel(str(summary_axis_label), fontsize=bar_axis_label_font)
        ax.set_xlabel("")

    bar_path = summaries_dir / "multi_target_importance_bars.png"
    fig.savefig(bar_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    return plot_path


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


def _ensure_min_test_samples_for_final(
    split_dir: Path,
    min_test_samples: int = FINAL_TOPK_MIN_TEST_SAMPLES,
) -> tuple[bool, str, int, int]:
    """Rebalance split files for final-top-k evaluation.

    Returns:
      - skip_eval: True when total available split files are < min_test_samples.
      - status: one of {'already_sufficient', 'rebalanced', 'insufficient_total'}.
      - train_count: resulting train file count.
      - test_count: resulting test file count.
    """
    train_file = split_dir / "train_files.txt"
    test_file = split_dir / "test_files.txt"

    train_names = _read_split_file_names(train_file)
    test_names = _read_split_file_names(test_file)

    if len(test_names) >= min_test_samples:
        return False, "already_sufficient", len(train_names), len(test_names)

    total = len(train_names) + len(test_names)
    if total < min_test_samples:
        return True, "insufficient_total", len(train_names), len(test_names)

    deficit = min_test_samples - len(test_names)
    moved = train_names[-deficit:] if deficit > 0 else []
    new_train = train_names[:-deficit] if deficit > 0 else list(train_names)
    # Keep temporal order: moved train tail should precede existing test tail.
    new_test = moved + test_names

    _write_split_file_names(train_file, new_train)
    _write_split_file_names(test_file, new_test)
    return False, "rebalanced", len(new_train), len(new_test)


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

    target_name = _derive_target_name(dataset_plan.dataset_dir.name, dataset_prefix)

    for rank, cand in enumerate(selected, start=1):
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
                        f"(min_test={FINAL_TOPK_MIN_TEST_SAMPLES})"
                    )
                if skip_eval:
                    print(
                        f"[WARN] Skipping final eval for {eval_cfg.parent.name}: "
                        f"only {train_n + test_n} total split samples available "
                        f"(< {FINAL_TOPK_MIN_TEST_SAMPLES})."
                    )
                    summary_rows = []
                else:
                    _set_eval_overrides(
                        eval_cfg,
                        run_baselines=run_baselines_in_final,
                    )
                    eval_result = eval_module.evaluate_single_config(
                        str(eval_cfg),
                        save_plots_override=not disable_eval_plots,
                    )
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
                                summary_rows.append(primary_model_row)

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
                        summary_rows = [eval_result]
            except Exception as e:
                print(f"[ERROR] Evaluation failed for config {variant_cfg}: {e}")
                summary_rows = []

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
                    "feature_tag": cand.feature_tag,
                    "row_count": cand.row_count,
                    "n_features": cand.n_features,
                    "objective_search": cand.objective,
                    "drop_rate_search": cand.drop_rate,
                    "model": model_name,
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
                    "feature_tag": cand.feature_tag,
                    "row_count": cand.row_count,
                    "n_features": cand.n_features,
                    "objective_search": cand.objective,
                    "drop_rate_search": cand.drop_rate,
                    "model": model_name,
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
                }

                if baseline_id is None:
                    rows.append(row_payload)
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

    final_df = pd.DataFrame(rows)
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

    k01_rows = df_final[df_final["subset_rank"] == 1].copy()
    if "model" in k01_rows.columns:
        k01_rows = k01_rows[~k01_rows["model"].apply(_is_baseline_model_value)]
    if k01_rows.empty or k01_rows["r2"].isna().all():
        print("[WARN] Rolling origin CV: no valid k01 rows in final metrics. Skipping.")
        return

    best_r2_idx = k01_rows["r2"].idxmax()
    best_row = k01_rows.loc[best_r2_idx]
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

    k01_rows = df_final[df_final["subset_rank"] == 1].copy()
    if "model" in k01_rows.columns:
        k01_rows = k01_rows[~k01_rows["model"].apply(_is_baseline_model_value)]
    valid_k01 = k01_rows[k01_rows["r2"].notnull() & np.isfinite(k01_rows["r2"].astype(float))] if not k01_rows.empty else k01_rows
    if valid_k01.empty:
        print(f"[WARN] _ensure_k01_baselines: no valid k01 rows for {plan.dataset_dir.name}; skipping.")
        return

    best_row = valid_k01.loc[valid_k01["r2"].idxmax()]
    best_model_str = str(best_row.get("model", ""))
    best_row_count = int(best_row["row_count"])
    best_feature_tag = str(best_row["feature_tag"])

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

    k01_rows = df_final[df_final["subset_rank"] == 1].copy()
    if "model" in k01_rows.columns:
        k01_rows = k01_rows[~k01_rows["model"].apply(_is_baseline_model_value)]
    if k01_rows.empty or k01_rows["r2"].isna().all():
        print(f"[WARN] _write_dataset_evaluation_summary: no valid k01 rows with r2 in {final_metrics_csv}")
        return None

    valid_k01 = k01_rows[k01_rows["r2"].notnull() & np.isfinite(k01_rows["r2"].astype(float))]
    if valid_k01.empty:
        print(f"[WARN] _write_dataset_evaluation_summary: no finite r2 values in k01 rows for {plan.dataset_dir.name}")
        return None

    best_idx = valid_k01["r2"].idxmax()
    best_row = valid_k01.loc[best_idx]
    best_model_type = str(best_row.get("model", ""))
    best_row_count = int(best_row["row_count"])
    best_feature_tag = str(best_row["feature_tag"])

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
    print(f"Final run baselines       : {final_run_baselines}")
    print(f"Final eval plots enabled  : {not final_disable_eval_plots}")
    print(f"Seed subsets from Shapley : {args.seed_subsets_from_shapley}")
    print(f"Seed subsets CSV          : {explicit_seed_csv}")
    print(f"Max seed subsets          : {args.max_seed_subsets}")

    if args.dry_run:
        for plan in plans:
            surrogate = _select_surrogate_config(plan.train_configs)
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

        surrogate_cfg = _select_surrogate_config(plan.train_configs)
        surrogate_data = train_module.load_config(str(surrogate_cfg))["data"]
        base_span = int(surrogate_data["input_row_2"]) - int(surrogate_data["input_row_1"])
        row_counts = _parse_row_counts(args.row_counts, default_span=base_span)
        include_row_count_in_plot_names = len(row_counts) > 1

        for row_count in row_counts:
            try:
                print(f"\n[SEARCH] rows={row_count} surrogate={surrogate_cfg.name}")
                seeded_subsets = _load_seed_subsets(
                    dataset_dir=plan.dataset_dir,
                    row_count=row_count,
                    explicit_path=explicit_seed_csv,
                    from_shapley=bool(args.seed_subsets_from_shapley),
                    max_seed_subsets=int(args.max_seed_subsets),
                )
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
                    include_row_count_in_plot_names=include_row_count_in_plot_names,
                    seeded_subsets=seeded_subsets,
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
        description="Feature-selection sweeper using beam+swap surrogate search and final full-model evaluation."
    )
    parser.add_argument("--data-root", type=str, default="data/output/regression")
    parser.add_argument("--dataset-prefix", type=str, default="MC")
    parser.add_argument("--config-pattern", type=str, default="config_*.yml")
    parser.add_argument("--limit-datasets", type=int, default=1)

    parser.add_argument("--row-counts", type=str, default=None)
    parser.add_argument("--min-features", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--no-improve-patience", type=int, default=3)
    parser.add_argument("--eval-budget", type=int, default=240)
    parser.add_argument("--max-swap-attempts", type=int, default=60)
    parser.add_argument("--lambda-drop", type=float, default=0.25)
    parser.add_argument("--final-top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    # --exclude argument removed
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_feature_selection_sweep(args)


if __name__ == "__main__":
    sys.exit(main())

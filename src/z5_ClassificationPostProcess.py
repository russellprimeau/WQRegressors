"""
Post-process feature-sweep predictions as threshold-based classifications.

Example:
    python src/z5_ClassificationPostProcess.py --run-dir data/output/CV14

Outputs are written under:
    <run-dir>/summaries/classification/

Optional arguments:
    --summaries-subdir <name>
        Output subdirectory under <run-dir>/summaries/. Defaults to "classification".

    --targets <target_dir_1> <target_dir_2> ...
        Restrict processing to one or more target dataset directories under the run directory.

    --sweep-namespace <name>
        Forecast sweep subdirectory under each target's forecasts/ directory.
        Defaults to "feature_sweeps".

    --evaluation-scope {test,combined}
        "test" uses the saved test-set predictions only.
        "combined" keeps best-model selection based on test r2, then recomputes
        predictions over the combined train + test evaluation samples for the selected model.

    --exclude-no-out-of-limit
        Exclude targets whose active evaluation scope contains zero ground-truth
        out-of-limit values. Excluded targets are still recorded in the threshold
        coverage output with an explicit exclusion reason.

    --no-figures
        Skip PNG figure generation. CSV outputs are still written.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import f_Evaluate as eval_module
from utils.config_utils import _resolve_data_paths, _resolve_path_from_config, load_config, select_best_model_row
from utils.limits import limit_exceedance_mask, load_limits_records, map_limits_to_columns
from utils.names import clean_target_label


_NORM_SUFFIX_RE = re.compile(r"(?:(?:_+)?(?:res|state))+$", re.IGNORECASE)
_BASELINE_MODELS = {"naive", "seasonal", "linear"}
_MODEL_ORDER = ["XGB", "transformer", "GP", "MLR", "MLR12", "MLRAll", "seasonal", "linear", "naive"]
_MODEL_COLORS = {
    "XGB": "#2a9d8f",
    "transformer": "#457b9d",
    "GP": "#1d3557",
    "MLR": "#e76f51",
    "MLR12": "#b56576",
    "MLRAll": "#8d6a9f",
    "seasonal": "#6c757d",
    "linear": "#495057",
    "naive": "#adb5bd",
}
_PREDICTION_COLUMN_BY_GROUP = {
    "XGB": "XGBRegressor",
    "transformer": "Transformer",
    "GP": "GPRegressor",
    "MLR": "MLR",
    "MLR12": "MLR-AVG12",
    "MLRAll": "MLR-AVGALL",
    "seasonal": "Seasonal",
    "linear": "Linear",
    "naive": "Naive",
}
_MLR_AGGREGATION_BY_MODEL = {
    "mlr": "last",
    "mlr_avg12": "avg12",
    "mlr_avgall": "avgall",
}


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent.parent / path).resolve()


def _normalize_dataset_name(name: str) -> str:
    return _NORM_SUFFIX_RE.sub("", str(name)).rstrip("_")


def _safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _safe_int(value) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _base_sample_file(name: str) -> str:
    return re.sub(r"_mc_\d+(?=\.csv$)", "", str(name or ""))


def _normalize_model_name(raw_model: str) -> str:
    return str(raw_model or "").strip().lower()


def _model_group(raw_model: str) -> str | None:
    key = _normalize_model_name(raw_model)
    if key == "xgb_regressor":
        return "XGB"
    if key == "transformer":
        return "transformer"
    if key == "gp_regressor":
        return "GP"
    if key == "mlr":
        return "MLR"
    if key == "mlr_avg12":
        return "MLR12"
    if key == "mlr_avgall":
        return "MLRAll"
    if key in _BASELINE_MODELS:
        return key
    return None


def _build_target_limit_spec(target_name: str, dataset_dir_name: str, limits_records: list[dict]) -> dict | None:
    candidates = []
    raw_values = [str(target_name or "").strip(), str(dataset_dir_name or "").strip()]
    for raw in raw_values:
        if not raw:
            continue
        candidates.append(raw)
        norm = _normalize_dataset_name(raw)
        if norm:
            candidates.append(norm)
        for prefix in ("MC_", "mc_"):
            if raw.startswith(prefix):
                candidates.append(raw[len(prefix):])
        candidates.append(raw.replace("__", " "))
        candidates.append(raw.replace("__", "_"))
        candidates.append(raw.replace("_", " "))
        if norm:
            candidates.append(norm.replace("__", " "))
            candidates.append(norm.replace("__", "_"))
            candidates.append(norm.replace("_", " "))
        candidates.append(clean_target_label(raw, "MC"))
        candidates.append(clean_target_label(norm, "MC"))

    ordered = []
    seen = set()
    for cand in candidates:
        cand = str(cand).strip()
        if cand and cand not in seen:
            seen.add(cand)
            ordered.append(cand)

    mapped = map_limits_to_columns(ordered, limits_records)
    for cand in ordered:
        if cand in mapped:
            spec = mapped[cand].copy()
            spec["matched_candidate"] = cand
            return spec
    return None


def _list_target_dirs(run_dir: Path, requested_targets: list[str] | None) -> list[Path]:
    out = []
    requested = set(requested_targets or [])
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "summaries":
            continue
        if requested and child.name not in requested:
            continue
        out.append(child)
    return out


def _read_metrics_csv(target_dir: Path, sweep_namespace: str) -> pd.DataFrame | None:
    path = target_dir / "forecasts" / sweep_namespace / "feature_sweep_final_metrics.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def _filter_candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["model_group"] = out.get("model", pd.Series(dtype=str)).apply(_model_group)
    out["r2_num"] = pd.to_numeric(out.get("r2"), errors="coerce")
    out["n_test_valid_num"] = pd.to_numeric(out.get("n_test_valid"), errors="coerce")
    out = out[out["model_group"].notna()].copy()
    out = out[np.isfinite(out["r2_num"])].copy()
    if "n_test_valid_num" in out.columns:
        out = out[out["n_test_valid_num"].fillna(0) > 0].copy()
    return out


def _select_best_rows(metrics_df: pd.DataFrame) -> list[pd.Series]:
    selected: list[pd.Series] = []
    if metrics_df.empty:
        return selected

    for group in _MODEL_ORDER:
        subset = metrics_df[metrics_df["model_group"] == group].copy()
        if subset.empty:
            continue
        if group in {"XGB", "transformer", "GP", "MLR", "MLR12", "MLRAll"}:
            selected.append(select_best_model_row(subset))
            continue
        sort_cols = []
        if "subset_rank" in subset.columns:
            subset["subset_rank_num"] = pd.to_numeric(subset["subset_rank"], errors="coerce")
            sort_cols.append("subset_rank_num")
        if "row_count" in subset.columns:
            subset["row_count_num"] = pd.to_numeric(subset["row_count"], errors="coerce")
            sort_cols.append("row_count_num")
        sort_cols.append("r2_num")
        selected.append(subset.sort_values(sort_cols, ascending=[True] * (len(sort_cols) - 1) + [False]).iloc[0])
    return selected


def _artifact_dir_for_row(target_dir: Path, sweep_namespace: str, row: pd.Series, metrics_df: pd.DataFrame) -> Path | None:
    sweep_dir = target_dir / "forecasts" / sweep_namespace
    if not sweep_dir.is_dir():
        return None

    raw_model = _normalize_model_name(row.get("model", ""))
    subset_label = str(row.get("subset_label", "")).strip()
    row_count = _safe_int(row.get("row_count"))
    feature_tag = str(row.get("feature_tag", "")).strip()

    candidate_names: list[str] = []
    if raw_model == "xgb_regressor":
        prefix = "xgb_01"
        if row_count is not None and feature_tag and subset_label:
            candidate_names.append(f"{prefix}_r{row_count}_{feature_tag}_{subset_label}")
        if row_count is not None and feature_tag:
            candidate_names.append(f"{prefix}_r{row_count}_{feature_tag}")
    elif raw_model == "transformer":
        prefix = "transformer_01"
        if row_count is not None and feature_tag and subset_label:
            candidate_names.append(f"{prefix}_r{row_count}_{feature_tag}_{subset_label}")
        if row_count is not None and feature_tag:
            candidate_names.append(f"{prefix}_r{row_count}_{feature_tag}")
    elif raw_model == "gp_regressor":
        prefix = "gp_01"
        if row_count is not None and feature_tag and subset_label:
            candidate_names.append(f"{prefix}_r{row_count}_{feature_tag}_{subset_label}")
        if row_count is not None and feature_tag:
            candidate_names.append(f"{prefix}_r{row_count}_{feature_tag}")
    elif raw_model in {"mlr", "mlr_avg12", "mlr_avgall"} and subset_label:
        candidate_names.append(f"{raw_model}_{subset_label.lower()}")

    for name in candidate_names:
        candidate_dir = sweep_dir / name
        if candidate_dir.is_dir():
            return candidate_dir

    if raw_model in _BASELINE_MODELS:
        matching_rows = metrics_df.copy()
        for preferred_model in ["xgb_regressor", "transformer", "gp_regressor", "mlr", "mlr_avg12", "mlr_avgall"]:
            candidate_pool = matching_rows[matching_rows["model"].astype(str).str.lower() == preferred_model]
            if subset_label:
                candidate_pool = candidate_pool[candidate_pool.get("subset_label", pd.Series(dtype=str)).astype(str) == subset_label]
            if feature_tag and "feature_tag" in candidate_pool.columns:
                candidate_pool = candidate_pool[candidate_pool["feature_tag"].astype(str) == feature_tag]
            for _, pool_row in candidate_pool.iterrows():
                artifact_dir = _artifact_dir_for_row(target_dir, sweep_namespace, pool_row, metrics_df)
                if artifact_dir is not None:
                    return artifact_dir

    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.lower()
        if raw_model == "xgb_regressor" and name.startswith("xgb_01"):
            return child
        if raw_model == "transformer" and name.startswith("transformer_01"):
            return child
        if raw_model == "gp_regressor" and name.startswith("gp_01"):
            return child
        if raw_model == "mlr" and name.startswith("mlr_") and not name.startswith("mlr_avg"):
            return child
        if raw_model == "mlr_avg12" and name.startswith("mlr_avg12_"):
            return child
        if raw_model == "mlr_avgall" and name.startswith("mlr_avgall_"):
            return child
    return None


def _undefined_note(denominator: float, empty_message: str) -> str:
    return empty_message if denominator == 0 else ""


def _compute_classification_metrics(
    truth_values: np.ndarray,
    pred_values: np.ndarray,
    threshold_upper,
    threshold_lower,
) -> dict:
    truth_numeric = pd.to_numeric(pd.Series(truth_values), errors="coerce").to_numpy(dtype=float)
    pred_numeric = pd.to_numeric(pd.Series(pred_values), errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(truth_numeric) & np.isfinite(pred_numeric)
    n_total = int(len(truth_numeric))
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return {
            "n_total": n_total,
            "n_valid": 0,
            "n_ground_truth_positive": 0,
            "n_ground_truth_negative": 0,
            "n_pred_positive": 0,
            "n_pred_negative": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": np.nan,
            "precision_out": np.nan,
            "recall_out": np.nan,
            "f1_out": np.nan,
            "specificity_within": np.nan,
            "prevalence_out": np.nan,
            "metric_status": "no_valid_pairs",
            "accuracy_note": "No valid truth/prediction pairs.",
            "precision_note": "No valid truth/prediction pairs.",
            "recall_note": "No valid truth/prediction pairs.",
            "f1_note": "No valid truth/prediction pairs.",
            "specificity_note": "No valid truth/prediction pairs.",
            "prevalence_note": "No valid truth/prediction pairs.",
        }

    truth_valid = truth_numeric[valid_mask]
    pred_valid = pred_numeric[valid_mask]
    actual_positive = limit_exceedance_mask(truth_valid, upper=threshold_upper, lower=threshold_lower)
    pred_positive = limit_exceedance_mask(pred_valid, upper=threshold_upper, lower=threshold_lower)

    tp = int(np.sum(actual_positive & pred_positive))
    tn = int(np.sum((~actual_positive) & (~pred_positive)))
    fp = int(np.sum((~actual_positive) & pred_positive))
    fn = int(np.sum(actual_positive & (~pred_positive)))
    n_truth_pos = int(actual_positive.sum())
    n_truth_neg = int((~actual_positive).sum())
    n_pred_pos = int(pred_positive.sum())
    n_pred_neg = int((~pred_positive).sum())

    precision_den = tp + fp
    recall_den = tp + fn
    specificity_den = tn + fp
    valid_den = tp + tn + fp + fn

    precision = float(tp / precision_den) if precision_den > 0 else np.nan
    recall = float(tp / recall_den) if recall_den > 0 else np.nan
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else np.nan
    )
    accuracy = float((tp + tn) / valid_den) if valid_den > 0 else np.nan
    specificity = float(tn / specificity_den) if specificity_den > 0 else np.nan
    prevalence = float(n_truth_pos / valid_den) if valid_den > 0 else np.nan

    status_tokens = []
    if n_truth_pos == 0:
        status_tokens.append("no_ground_truth_positive")
    if n_truth_neg == 0:
        status_tokens.append("no_ground_truth_negative")
    if n_pred_pos == 0:
        status_tokens.append("no_predicted_positive")
    if n_pred_neg == 0:
        status_tokens.append("no_predicted_negative")
    if not status_tokens:
        status_tokens.append("ok")

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_ground_truth_positive": n_truth_pos,
        "n_ground_truth_negative": n_truth_neg,
        "n_pred_positive": n_pred_pos,
        "n_pred_negative": n_pred_neg,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision_out": precision,
        "recall_out": recall,
        "f1_out": f1,
        "specificity_within": specificity,
        "prevalence_out": prevalence,
        "metric_status": "|".join(status_tokens),
        "accuracy_note": _undefined_note(valid_den, "No valid truth/prediction pairs."),
        "precision_note": _undefined_note(precision_den, "No predicted out-of-limit cases."),
        "recall_note": _undefined_note(recall_den, "No ground-truth out-of-limit cases."),
        "f1_note": "F1 undefined because precision or recall is undefined." if not np.isfinite(f1) else "",
        "specificity_note": _undefined_note(specificity_den, "No ground-truth within-limit cases."),
        "prevalence_note": _undefined_note(valid_den, "No valid truth/prediction pairs."),
    }


def _threshold_mode(upper, lower) -> str:
    has_upper = upper is not None and not pd.isna(upper)
    has_lower = lower is not None and not pd.isna(lower)
    if has_upper and has_lower:
        return "both"
    if has_upper:
        return "upper"
    if has_lower:
        return "lower"
    return "missing"


def _empty_metric_payload(status: str, message: str) -> dict:
    out = {
        "metric_status": status,
        "precision_note": message,
        "recall_note": message,
        "accuracy_note": message,
        "f1_note": message,
        "specificity_note": message,
        "prevalence_note": message,
    }
    for key in [
        "n_total",
        "n_valid",
        "n_ground_truth_positive",
        "n_ground_truth_negative",
        "n_pred_positive",
        "n_pred_negative",
        "tp",
        "tn",
        "fp",
        "fn",
    ]:
        out[key] = 0
    for key in ["accuracy", "precision_out", "recall_out", "f1_out", "specificity_within", "prevalence_out"]:
        out[key] = np.nan
    return out


def _candidate_sample_paths(data_dir: Path, sample_subdir: str, sample_file: str) -> list[Path]:
    sample_file = str(sample_file or "")
    base_name = _base_sample_file(sample_file)
    paths = [data_dir / sample_subdir / sample_file]
    if base_name != sample_file:
        paths.append(data_dir / sample_subdir / base_name)
    paths.append(data_dir / "samples" / base_name)
    unique_paths = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


def _resolve_output_indices(df: pd.DataFrame, output_rows) -> list[int]:
    n_rows = len(df)
    if n_rows <= 0:
        return []
    if isinstance(output_rows, slice):
        return list(range(*output_rows.indices(n_rows)))
    if isinstance(output_rows, (list, tuple, np.ndarray)):
        resolved = []
        for value in output_rows:
            idx = int(value)
            if idx < 0:
                idx = n_rows + idx
            if 0 <= idx < n_rows:
                resolved.append(idx)
        return resolved
    idx = int(output_rows)
    if idx < 0:
        idx = n_rows + idx
    if 0 <= idx < n_rows:
        return list(range(idx, n_rows))
    return []


def _nanmean_or_nan(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def _extract_sample_metadata(
    data_dir: Path,
    sample_subdir: str,
    sample_file: str,
    output_rows,
    output_column: str,
    target_name: str,
    sample_cache: dict,
) -> dict:
    cache_key = (str(data_dir), str(sample_subdir), str(sample_file), str(output_column), str(target_name), str(output_rows))
    if cache_key in sample_cache:
        return sample_cache[cache_key]

    meta = {
        "sample_exists": False,
        "sample_path": "",
        "timestamp": pd.NaT,
        "state_norm": np.nan,
        "prev_state_norm": np.nan,
        "target_norm": np.nan,
        "target_full": np.nan,
    }
    state_column = f"{target_name}_state"
    full_column = str(target_name)
    for candidate_path in _candidate_sample_paths(data_dir, sample_subdir, sample_file):
        if not candidate_path.exists():
            continue
        try:
            sample_df = pd.read_csv(candidate_path)
        except Exception:
            continue
        output_idx = _resolve_output_indices(sample_df, output_rows)
        if not output_idx:
            continue
        meta["sample_exists"] = True
        meta["sample_path"] = str(candidate_path)
        if "TIMESTAMP" in sample_df.columns:
            timestamps = pd.to_datetime(sample_df.loc[output_idx, "TIMESTAMP"], errors="coerce")
            if len(timestamps) > 0:
                meta["timestamp"] = timestamps.iloc[-1]
        if state_column in sample_df.columns:
            meta["state_norm"] = _nanmean_or_nan(sample_df.loc[output_idx, state_column])
            prev_idx = [idx - 1 for idx in output_idx if int(idx) - 1 >= 0]
            if prev_idx:
                meta["prev_state_norm"] = _nanmean_or_nan(sample_df.loc[prev_idx, state_column])
        if output_column in sample_df.columns:
            meta["target_norm"] = _nanmean_or_nan(sample_df.loc[output_idx, output_column])
        if full_column in sample_df.columns:
            meta["target_full"] = _nanmean_or_nan(sample_df.loc[output_idx, full_column])
        break

    sample_cache[cache_key] = meta
    return meta


def _load_normalization_bounds(target_dir: Path) -> dict:
    path = target_dir / "normalization.json"
    if not path.exists():
        return {}
    try:
        return pd.read_json(path, typ="series").to_dict()
    except Exception:
        import json

        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


def _denormalize_value(value, column: str, norm_bounds: dict) -> float:
    out = _safe_float(value)
    if not np.isfinite(out):
        return np.nan
    bounds = norm_bounds.get(column)
    if not isinstance(bounds, dict):
        return out
    lo = _safe_float(bounds.get("min"))
    hi = _safe_float(bounds.get("max"))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return out
    return float(out * (hi - lo) + lo)


def _reconstruct_full_values(
    truth_norm,
    pred_norm,
    target_name: str,
    output_column: str,
    prev_state_norm,
    truth_full_direct,
    norm_bounds: dict,
) -> tuple[float, float, float, float, float]:
    if output_column.endswith("_res"):
        state_column = f"{target_name}_state"
        prev_state_full = _denormalize_value(prev_state_norm, state_column, norm_bounds)
        truth_res = _denormalize_value(truth_norm, output_column, norm_bounds)
        pred_res = _denormalize_value(pred_norm, output_column, norm_bounds)
        truth_full = _safe_float(truth_full_direct)
        if not np.isfinite(truth_full):
            truth_full = prev_state_full + truth_res if np.isfinite(prev_state_full) and np.isfinite(truth_res) else np.nan
        pred_full = prev_state_full + pred_res if np.isfinite(prev_state_full) and np.isfinite(pred_res) else np.nan
        return truth_full, pred_full, prev_state_full, truth_res, pred_res
    truth_full = _denormalize_value(truth_norm, output_column, norm_bounds)
    pred_full = _denormalize_value(pred_norm, output_column, norm_bounds)
    return truth_full, pred_full, np.nan, truth_full, pred_full


def _prediction_series_from_df(pred_df: pd.DataFrame, base_label: str) -> pd.Series | None:
    exact = [col for col in pred_df.columns if col == base_label]
    if exact:
        return pd.to_numeric(pred_df[exact[0]], errors="coerce")
    matched = [col for col in pred_df.columns if col.startswith(f"{base_label}_")]
    if not matched:
        return None
    return pred_df[matched].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _target_series_from_df(pred_df: pd.DataFrame) -> pd.Series | None:
    if "target" in pred_df.columns:
        return pd.to_numeric(pred_df["target"], errors="coerce")
    matched = [col for col in pred_df.columns if col.startswith("target_")]
    if not matched:
        return None
    return pred_df[matched].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _find_eval_config(artifact_dir: Path) -> Path | None:
    configs = sorted(artifact_dir.glob("config_evaluate_*.yml"))
    return configs[0] if configs else None


def _resolve_eval_paths(config: dict) -> tuple[dict, dict]:
    config_dir = config["__config_dir"]
    data_cfg = dict(config["data"])
    eval_cfg = eval_module.merge_eval_config(config)
    data_cfg["data_dir"], data_cfg["sample_subdir"] = _resolve_data_paths(data_cfg, config_dir)
    for key in ["historic_path", "thresholds_path", "normalization_path"]:
        if eval_cfg.get(key):
            eval_cfg[key] = str(_resolve_path_from_config(eval_cfg[key], config_dir))
    if eval_cfg.get("baseline_split_source"):
        eval_cfg["baseline_split_source"] = str(_resolve_path_from_config(eval_cfg["baseline_split_source"], config_dir))
    return data_cfg, eval_cfg


def _load_eval_context(artifact_dir: Path, evaluation_scope: str) -> dict:
    config_path = _find_eval_config(artifact_dir)
    if config_path is None:
        raise FileNotFoundError(f"No config_evaluate_*.yml found in {artifact_dir}")

    config = load_config(str(config_path))
    data_cfg, eval_cfg = _resolve_eval_paths(config)
    model_type = str(config["model_type"])
    model_name = str(config.get("model_name", ""))
    model_config = eval_module.load_model_config(data_cfg["data_dir"], data_cfg["forecast_name"], model_name, fallback_data=data_cfg)
    input_columns = list(model_config["input_columns"])
    output_columns = list(model_config["output_columns"])
    input_rows = slice(model_config["input_row_1"], model_config["input_row_2"])
    output_rows = model_config["output_rows"]
    input_aggregation = str(model_config.get("input_aggregation", data_cfg.get("input_aggregation", "none"))).lower()
    split_cfg = config.get("data_split", {"random_state": 42})
    split_fault_tolerant = bool(split_cfg.get("fault_tolerant", False))
    collapse_mc = bool(eval_cfg.get("collapse_mc_replicates_for_eval", False))

    split_base_dir = Path(data_cfg.get("forecast_dir", config_path.parent))
    test_split_files = eval_module._read_split_files(split_base_dir, "test_files.txt")
    if collapse_mc:
        test_split_files = eval_module._dedupe_split_files_by_base_sample(test_split_files)
    test_samples = eval_module.load_split_samples(
        data_cfg["data_dir"],
        data_cfg["sample_subdir"],
        data_cfg["forecast_name"],
        input_columns,
        output_columns,
        input_rows,
        output_rows,
        "test_files.txt",
        split_source_dir=split_base_dir,
        split_files_override=test_split_files,
        fault_tolerant=split_fault_tolerant,
        input_aggregation=input_aggregation,
    )

    train_split_files = []
    train_samples = []
    if evaluation_scope == "combined":
        train_split_files = eval_module._read_split_files(split_base_dir, "train_files.txt")
        if collapse_mc:
            train_split_files = eval_module._dedupe_split_files_by_base_sample(train_split_files)
        train_samples = eval_module.load_split_samples(
            data_cfg["data_dir"],
            data_cfg["sample_subdir"],
            data_cfg["forecast_name"],
            input_columns,
            output_columns,
            input_rows,
            output_rows,
            "train_files.txt",
            split_source_dir=split_base_dir,
            split_files_override=train_split_files,
            fault_tolerant=split_fault_tolerant,
            input_aggregation=input_aggregation,
        )

    if evaluation_scope == "combined":
        eval_samples = list(train_samples) + list(test_samples)
        eval_split_files = list(train_split_files) + list(test_split_files)
        eval_kinds = (["train"] * len(train_samples)) + (["test"] * len(test_samples))
    else:
        eval_samples = list(test_samples)
        eval_split_files = list(test_split_files)
        eval_kinds = ["test"] * len(test_samples)

    return {
        "config_path": config_path,
        "config": config,
        "data_cfg": data_cfg,
        "eval_cfg": eval_cfg,
        "split_cfg": split_cfg,
        "model_type": model_type,
        "model_name": model_name,
        "model_config": model_config,
        "input_columns": input_columns,
        "output_columns": output_columns,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "input_aggregation": input_aggregation,
        "test_samples": test_samples,
        "test_split_files": test_split_files,
        "train_samples": train_samples,
        "train_split_files": train_split_files,
        "eval_samples": eval_samples,
        "eval_split_files": eval_split_files,
        "eval_kinds": eval_kinds,
    }


def _predict_mlr_from_samples(samples: list, input_columns: list[str], raw_model: str, model_config: dict) -> np.ndarray:
    meta_rows = list(model_config.get("per_target_meta", []))
    if not meta_rows:
        return np.empty((0, 1), dtype=float)
    meta = meta_rows[0]
    selected_features = list(meta.get("selected_features", []))
    coefficients = np.asarray(meta.get("coefficients", []), dtype=float)
    intercept = _safe_float(meta.get("intercept", 0.0))
    aggregation_mode = _MLR_AGGREGATION_BY_MODEL.get(raw_model, "last")

    preds = np.full((len(samples), 1), np.nan, dtype=float)
    if len(selected_features) == 0 or coefficients.size == 0 or not np.isfinite(intercept):
        if np.isfinite(intercept):
            preds[:] = intercept
        return preds

    col_index = {name: idx for idx, name in enumerate(input_columns)}
    sel_idx = [col_index[name] for name in selected_features if name in col_index]
    if not sel_idx or len(sel_idx) != len(coefficients):
        return preds

    for idx, sample in enumerate(samples):
        x_sample = np.asarray(sample[0], dtype=float)
        if x_sample.ndim == 1:
            x_sample = x_sample.reshape(1, -1)
        if aggregation_mode == "avg12":
            agg = np.nanmean(x_sample[-12:] if len(x_sample) >= 12 else x_sample, axis=0)
        elif aggregation_mode == "avgall":
            agg = np.nanmean(x_sample, axis=0)
        else:
            agg = x_sample[-1]
        selected = agg[sel_idx]
        if np.all(np.isfinite(selected)):
            preds[idx, 0] = float(intercept + np.dot(selected, coefficients))
    return preds


def _aligned_target_array(samples: list) -> np.ndarray:
    if not samples:
        return np.empty((0, 1), dtype=float)
    y = np.asarray([np.asarray(sample[1], dtype=float).reshape(-1) for sample in samples], dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return y


def _build_grouped_predictions_df(preds, targets, split_files, kinds) -> pd.DataFrame:
    grouped_rows, _ = eval_module._group_independent_prediction_stats(preds, targets, split_files)
    kind_by_sample = {}
    for sample_file, kind in zip(split_files, kinds):
        kind_by_sample.setdefault(_base_sample_file(sample_file), str(kind))
    rows = []
    for grouped in grouped_rows:
        rows.append(
            {
                "kind": kind_by_sample.get(grouped["sample_file"], "test"),
                "sample_file": grouped["sample_file"],
                "target_norm": _nanmean_or_nan(grouped.get("target_mean", [])),
                "pred_norm": _nanmean_or_nan(grouped.get("pred_mean", [])),
                "n_replicates": int(grouped.get("n_replicates", 1)),
            }
        )
    return pd.DataFrame(rows)


def _prediction_rows_from_saved_csv(predictions_csv: Path, prediction_column: str, evaluation_scope: str) -> pd.DataFrame:
    pred_df = pd.read_csv(predictions_csv)
    if "kind" in pred_df.columns:
        pred_df = pred_df[pred_df["kind"].astype(str).str.lower() == "test"].copy()
    target_series = _target_series_from_df(pred_df)
    pred_series = _prediction_series_from_df(pred_df, prediction_column)
    if target_series is None or pred_series is None or "sample_file" not in pred_df.columns:
        raise ValueError("predictions.csv missing required target/model/sample_file columns.")
    rows = pred_df[["sample_file"]].copy()
    rows["kind"] = "test"
    rows["target_norm"] = pd.to_numeric(target_series, errors="coerce")
    rows["pred_norm"] = pd.to_numeric(pred_series, errors="coerce")
    rows["n_replicates"] = pd.to_numeric(pred_df.get("mc_n_replicates"), errors="coerce")
    rows["evaluation_scope"] = evaluation_scope
    return rows


def _prediction_rows_from_recomputed_scope(context: dict, raw_model: str, evaluation_scope: str) -> pd.DataFrame:
    data_cfg = context["data_cfg"]
    eval_cfg = context["eval_cfg"]
    split_cfg = context["split_cfg"]
    model_config = context["model_config"]
    eval_samples = context["eval_samples"]
    eval_split_files = context["eval_split_files"]
    eval_kinds = context["eval_kinds"]
    raw_model = _normalize_model_name(raw_model)

    if raw_model in _BASELINE_MODELS:
        output_columns = context["output_columns"]
        baseline_output_rows = eval_module._baseline_output_rows_start(context["output_rows"])
        secondary, baseline_window_hours = eval_module.load_secondary(output_columns, int(eval_cfg.get("window_hours", 340)))
        if raw_model == "naive":
            preds, targets = eval_module.evaluate_naive(
                eval_samples,
                eval_cfg["historic_path"],
                output_columns,
                data_cfg["data_dir"],
                output_rows=baseline_output_rows,
                gap_hours=int(eval_cfg.get("gap_hours", 5)),
                sample_subdir=data_cfg.get("sample_subdir", "samples"),
            )
        elif raw_model == "seasonal":
            preds, targets = eval_module.evaluate_seasonal(
                eval_samples,
                eval_cfg["historic_path"],
                output_columns,
                data_cfg["data_dir"],
                output_rows=baseline_output_rows,
                diurnal_window=int(eval_cfg.get("diurnal_window", 2)),
                secondary=secondary,
                sample_subdir=data_cfg.get("sample_subdir", "samples"),
            )
        else:
            preds, targets = eval_module.evaluate_linear(
                data_cfg["data_dir"],
                data_cfg["forecast_name"],
                eval_samples,
                eval_cfg["historic_path"],
                output_columns,
                output_rows=baseline_output_rows,
                window_hours=int(baseline_window_hours),
                gap_hours=int(eval_cfg.get("gap_hours", 0)),
                debug_plot=False,
                examples=int(eval_cfg.get("debug_examples", 10)),
                sample_subdir=data_cfg.get("sample_subdir", "samples"),
            )
        rows = _build_grouped_predictions_df(preds, targets, eval_split_files, eval_kinds)
        rows["evaluation_scope"] = evaluation_scope
        return rows

    if raw_model in _MLR_AGGREGATION_BY_MODEL:
        preds = _predict_mlr_from_samples(eval_samples, context["input_columns"], raw_model, model_config)
        targets = _aligned_target_array(eval_samples)
        rows = _build_grouped_predictions_df(preds, targets, eval_split_files, eval_kinds)
        rows["evaluation_scope"] = evaluation_scope
        return rows

    torch_mod = getattr(eval_module, "torch", None)
    device = torch_mod.device("cuda" if torch_mod is not None and torch_mod.cuda.is_available() else "cpu")
    train_samples_for_model = context["train_samples"] if raw_model == "gp_regressor" else (context["train_samples"] or None)
    model = eval_module.load_model(
        raw_model,
        data_cfg,
        split_cfg,
        context["model_name"],
        model_config,
        device,
        train_samples=train_samples_for_model,
        config_dir=context["config"]["__config_dir"],
    )

    if raw_model == "transformer":
        X = np.asarray([np.asarray(sample[0], dtype=float) for sample in eval_samples], dtype=float)
        preds = model(torch_mod.tensor(X, dtype=torch_mod.float32, device=device)).detach().cpu().numpy()
    elif raw_model == "gp_regressor":
        X = np.asarray([np.asarray(sample[0], dtype=float).flatten() for sample in eval_samples], dtype=float)
        preds = eval_module._predict_gp_bundle(model, X, device)["mean"]
    elif raw_model == "xgb_regressor":
        X = np.asarray([np.asarray(sample[0], dtype=float).flatten() for sample in eval_samples], dtype=float)
        y = _aligned_target_array(eval_samples)
        preds = model.predict(X).reshape(-1, y.shape[1] if y.ndim > 1 else 1)
    else:
        raise ValueError(f"Unsupported model for recomputed scope: {raw_model}")

    targets = _aligned_target_array(eval_samples)
    rows = _build_grouped_predictions_df(preds, targets, eval_split_files, eval_kinds)
    rows["evaluation_scope"] = evaluation_scope
    return rows


def _build_point_rows(
    prediction_rows: pd.DataFrame,
    target_dir: Path,
    sample_subdir: str,
    output_rows,
    target_name: str,
    sample_target_name: str,
    output_column: str,
    model_type: str,
    evaluation_scope: str,
    threshold_upper,
    threshold_lower,
    norm_bounds: dict,
    sample_cache: dict,
) -> pd.DataFrame:
    rows = []
    for _, pred_row in prediction_rows.iterrows():
        sample_file = _base_sample_file(pred_row.get("sample_file", ""))
        metadata = _extract_sample_metadata(
            data_dir=target_dir,
            sample_subdir=sample_subdir,
            sample_file=sample_file,
            output_rows=output_rows,
            output_column=output_column,
            target_name=sample_target_name,
            sample_cache=sample_cache,
        )
        truth_norm = _safe_float(pred_row.get("target_norm"))
        if not np.isfinite(truth_norm):
            truth_norm = _safe_float(metadata.get("target_norm"))
        truth_full, pred_full, prev_state_full, truth_res, pred_res = _reconstruct_full_values(
            truth_norm=truth_norm,
            pred_norm=_safe_float(pred_row.get("pred_norm")),
            target_name=sample_target_name,
            output_column=output_column,
            prev_state_norm=metadata.get("prev_state_norm"),
            truth_full_direct=metadata.get("target_full"),
            norm_bounds=norm_bounds,
        )
        truth_out_of_limit = bool(
            np.isfinite(truth_full)
            and limit_exceedance_mask(
                np.asarray([truth_full], dtype=float),
                upper=threshold_upper,
                lower=threshold_lower,
            )[0]
        )
        rows.append(
            {
                "target": target_name,
                "model_type": model_type,
                "evaluation_scope": evaluation_scope,
                "kind": str(pred_row.get("kind", "test")),
                "sample_file": sample_file,
                "timestamp": metadata.get("timestamp", pd.NaT),
                "sample_path": metadata.get("sample_path", ""),
                "truth_norm": truth_norm,
                "pred_norm": _safe_float(pred_row.get("pred_norm")),
                "state_norm": _safe_float(metadata.get("state_norm")),
                "prev_state_norm": _safe_float(metadata.get("prev_state_norm")),
                "truth_full_value": truth_full,
                "pred_full_value": pred_full,
                "state_full_value": _safe_float(metadata.get("target_full")),
                "prev_state_full_value": prev_state_full,
                "truth_res_value": truth_res,
                "pred_res_value": pred_res,
                "truth_out_of_limit": truth_out_of_limit,
                "threshold_upper": threshold_upper,
                "threshold_lower": threshold_lower,
                "n_replicates": _safe_int(pred_row.get("n_replicates")),
            }
        )
    points_df = pd.DataFrame(rows)
    if not points_df.empty:
        points_df["timestamp"] = pd.to_datetime(points_df["timestamp"], errors="coerce")
        points_df = points_df.sort_values(["timestamp", "sample_file"]).reset_index(drop=True)
    return points_df


def _evaluate_selected_row(
    target_dir: Path,
    target_name: str,
    target_label: str,
    sweep_namespace: str,
    row: pd.Series,
    metrics_df: pd.DataFrame,
    limit_spec: dict | None,
    evaluation_scope: str,
    sample_cache: dict,
) -> tuple[dict, pd.DataFrame]:
    model_group = str(row.get("model_group"))
    raw_model = _normalize_model_name(row.get("model", ""))
    prediction_column = _PREDICTION_COLUMN_BY_GROUP.get(model_group, "")
    artifact_dir = _artifact_dir_for_row(target_dir, sweep_namespace, row, metrics_df)
    out = {
        "dataset": target_dir.name,
        "target": target_name,
        "target_label": target_label,
        "model_type": model_group,
        "raw_model": raw_model,
        "prediction_column": prediction_column,
        "source_subset_label": row.get("subset_label", ""),
        "source_feature_tag": row.get("feature_tag", ""),
        "source_subset_rank": row.get("subset_rank", ""),
        "source_row_count": row.get("row_count", ""),
        "source_r2": _safe_float(row.get("r2")),
        "source_rmse": _safe_float(row.get("rmse")),
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else "",
        "threshold_upper": np.nan,
        "threshold_lower": np.nan,
        "threshold_mode": "missing",
        "threshold_status": "missing",
        "threshold_match_candidate": "",
        "evaluation_scope": evaluation_scope,
        "excluded_reason": "",
        "n_scope_positive": 0,
        "n_scope_negative": 0,
    }

    if limit_spec is None:
        out.update(_empty_metric_payload("missing_threshold", "Target has no matched threshold in Limits.csv."))
        return out, pd.DataFrame()

    upper = limit_spec.get("upper")
    lower = limit_spec.get("lower")
    threshold_mode = _threshold_mode(upper, lower)
    out["threshold_upper"] = upper if upper is not None else np.nan
    out["threshold_lower"] = lower if lower is not None else np.nan
    out["threshold_mode"] = threshold_mode
    out["threshold_status"] = "ok" if threshold_mode != "missing" else "missing"
    out["threshold_match_candidate"] = limit_spec.get("matched_candidate", "")

    if artifact_dir is None:
        out.update(_empty_metric_payload("missing_predictions", "Could not resolve forecast artifact directory."))
        return out, pd.DataFrame()

    try:
        context = _load_eval_context(artifact_dir, evaluation_scope)
    except Exception as exc:
        out.update(_empty_metric_payload("missing_predictions", f"Could not load evaluation config/context: {exc}"))
        return out, pd.DataFrame()

    output_columns = list(context.get("output_columns", []))
    output_column = output_columns[0] if output_columns else target_name
    sample_target_name = output_column[:-4] if output_column.endswith("_res") else output_column
    norm_bounds = _load_normalization_bounds(target_dir)

    try:
        if evaluation_scope == "test":
            prediction_rows = _prediction_rows_from_saved_csv(artifact_dir / "predictions.csv", prediction_column, evaluation_scope)
        else:
            prediction_rows = _prediction_rows_from_recomputed_scope(context, raw_model, evaluation_scope)
    except Exception as exc:
        out.update(_empty_metric_payload("missing_predictions", f"Could not load or compute predictions: {exc}"))
        return out, pd.DataFrame()

    if prediction_rows.empty:
        out.update(_empty_metric_payload("missing_predictions", "No prediction rows available for the selected scope."))
        return out, pd.DataFrame()

    points_df = _build_point_rows(
        prediction_rows=prediction_rows,
        target_dir=target_dir,
        sample_subdir=str(context["data_cfg"].get("sample_subdir", "samples")),
        output_rows=context["output_rows"],
        target_name=target_name,
        sample_target_name=sample_target_name,
        output_column=output_column,
        model_type=model_group,
        evaluation_scope=evaluation_scope,
        threshold_upper=upper,
        threshold_lower=lower,
        norm_bounds=norm_bounds,
        sample_cache=sample_cache,
    )

    if points_df.empty:
        out.update(_empty_metric_payload("missing_predictions", "No evaluation points could be reconstructed for the selected scope."))
        return out, points_df

    metrics = _compute_classification_metrics(
        truth_values=points_df["truth_full_value"].to_numpy(),
        pred_values=points_df["pred_full_value"].to_numpy(),
        threshold_upper=upper,
        threshold_lower=lower,
    )
    out.update(metrics)
    out["n_scope_positive"] = int(metrics.get("n_ground_truth_positive", 0))
    out["n_scope_negative"] = int(metrics.get("n_ground_truth_negative", 0))
    return out, points_df


def _build_threshold_coverage_rows(class_df: pd.DataFrame, all_targets_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if class_df.empty and all_targets_df.empty:
        return pd.DataFrame()

    class_groups = {key: group for key, group in class_df.groupby("target", sort=False)}
    target_order = all_targets_df["target"].tolist() if not all_targets_df.empty else list(class_groups.keys())
    for target_name in dict.fromkeys(target_order):
        group = class_groups.get(target_name)
        if group is not None and not group.empty:
            valid_rows = group[group["n_valid"].fillna(0) > 0]
            base_row = valid_rows.iloc[0] if not valid_rows.empty else group.iloc[0]
        else:
            fallback = all_targets_df[all_targets_df["target"] == target_name]
            if fallback.empty:
                continue
            base_row = fallback.iloc[0]
        rows.append(
            {
                "dataset": base_row.get("dataset", ""),
                "target": target_name,
                "target_label": base_row.get("target_label", ""),
                "evaluation_scope": base_row.get("evaluation_scope", ""),
                "threshold_upper": base_row.get("threshold_upper", np.nan),
                "threshold_lower": base_row.get("threshold_lower", np.nan),
                "threshold_mode": base_row.get("threshold_mode", "missing"),
                "threshold_status": base_row.get("threshold_status", "missing"),
                "threshold_match_candidate": base_row.get("threshold_match_candidate", ""),
                "coverage_source_model_type": base_row.get("model_type", ""),
                "n_total": base_row.get("n_total", 0),
                "n_valid": base_row.get("n_valid", 0),
                "n_scope_positive": base_row.get("n_scope_positive", base_row.get("n_ground_truth_positive", 0)),
                "n_scope_negative": base_row.get("n_scope_negative", base_row.get("n_ground_truth_negative", 0)),
                "prevalence_out": base_row.get("prevalence_out", np.nan),
                "excluded_reason": base_row.get("excluded_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def _annotate_grouped_bars(ax, bars, values):
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    for bar, val in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        if np.isfinite(val):
            label = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
        else:
            label = "NA"
            y = 0.0
        ax.text(x, y + ymax * 0.01, label, ha="center", va="bottom", rotation=90, fontsize=7)


def _plot_grouped_metric(ax, df: pd.DataFrame, metric: str, title: str, ylabel: str):
    targets = list(dict.fromkeys(df["target"].tolist()))
    if not targets:
        ax.set_axis_off()
        return
    x = np.arange(len(targets), dtype=float)
    width = 0.8 / max(len(_MODEL_ORDER), 1)
    for idx, model_type in enumerate(_MODEL_ORDER):
        subset = df[df["model_type"] == model_type]
        values = []
        for target_name in targets:
            rows = subset[subset["target"] == target_name]
            values.append(_safe_float(rows.iloc[0][metric]) if not rows.empty else np.nan)
        offsets = x + (idx - (len(_MODEL_ORDER) - 1) / 2.0) * width
        bars = ax.bar(
            offsets,
            [0.0 if not np.isfinite(v) else v for v in values],
            width=width,
            color=_MODEL_COLORS[model_type],
            label=model_type,
        )
        _annotate_grouped_bars(ax, bars, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([clean_target_label(t, "MC") or t for t in targets], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)


def _write_metric_figures(class_df: pd.DataFrame, out_dir: Path, evaluation_scope: str) -> None:
    plot_df = class_df.copy()
    plot_df = plot_df[plot_df["threshold_status"].astype(str) == "ok"].copy()
    if plot_df.empty:
        print("[INFO] No threshold-matched rows available for classification figures.")
        return

    fig, ax = plt.subplots(figsize=(max(12, 0.9 * plot_df["target"].nunique()), 7))
    _plot_grouped_metric(ax, plot_df, "accuracy", "Classification Accuracy", "Accuracy")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
    fig.text(0.5, 0.01, f"Evaluation scope: {evaluation_scope}", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.savefig(out_dir / "classification_accuracy.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(max(12, 0.9 * plot_df["target"].nunique()), 16), sharex=True)
    _plot_grouped_metric(axes[0], plot_df, "precision_out", "Precision: Out of Limit", "Precision")
    _plot_grouped_metric(axes[1], plot_df, "recall_out", "Recall: Out of Limit", "Recall")
    _plot_grouped_metric(axes[2], plot_df, "f1_out", "F1: Out of Limit", "F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
    fig.text(0.5, 0.01, "NA indicates an undefined metric due to zero support in the denominator.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_dir / "classification_precision_recall_f1.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(max(12, 0.9 * plot_df["target"].nunique()), 14), sharex=True)
    _plot_grouped_metric(axes[0, 0], plot_df, "tp", "True Positives", "Count")
    _plot_grouped_metric(axes[0, 1], plot_df, "fp", "False Positives", "Count")
    _plot_grouped_metric(axes[1, 0], plot_df, "fn", "False Negatives", "Count")
    _plot_grouped_metric(axes[1, 1], plot_df, "tn", "True Negatives", "Count")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
    fig.text(0.5, 0.01, "Support counts expose targets with zero positive or zero negative ground-truth cases.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_dir / "classification_confusion_support.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _build_scope_spans(group: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    spans = []
    for kind, kind_df in group.groupby("kind", sort=False):
        valid = kind_df["timestamp"].dropna().sort_values()
        if valid.empty:
            continue
        spans.append((str(kind), valid.iloc[0], valid.iloc[-1]))
    return spans


def _write_overlay_figure(
    points_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    out_dir: Path,
    evaluation_scope: str,
    truth_value_column: str,
    pred_value_column: str,
    output_filename: str,
    show_limit_lines: bool,
) -> None:
    plot_df = points_df.copy()
    if plot_df.empty:
        print("[INFO] No evaluation points available for overlay figure.")
        return

    font_size = 10
    y_label_font_size = int(round(font_size * 1.5))
    x_label_font_size = int(round(font_size * 1.5))
    y_value_font_size = font_size

    target_order = coverage_df["target"].tolist() if not coverage_df.empty else list(dict.fromkeys(plot_df["target"].tolist()))
    targets = [target for target in target_order if target in set(plot_df["target"])]
    if not targets:
        print("[INFO] No included targets available for overlay figure.")
        return

    fig, axes = plt.subplots(len(targets), 1, sharex=False, figsize=(17, max(4.5, 2.9 * len(targets))), gridspec_kw={"hspace": 0.28})
    if len(targets) == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []
    for ax, target in zip(axes, targets):
        target_points = plot_df[plot_df["target"] == target].copy().sort_values(["timestamp", "sample_file", "model_type"])
        base_row = coverage_df[coverage_df["target"] == target].iloc[0] if not coverage_df.empty and target in set(coverage_df["target"]) else None
        target_label = clean_target_label(target, "MC") or target
        upper = _safe_float(base_row.get("threshold_upper")) if base_row is not None else np.nan
        lower = _safe_float(base_row.get("threshold_lower")) if base_row is not None else np.nan

        for kind, start, end in _build_scope_spans(target_points):
            if kind == "train":
                color = "#66bb66"
                label = "Training set"
            else:
                color = "#f4a3c2"
                label = "Test set"
            ax.axvspan(start, end, ymin=0.03, ymax=0.97, color=color, alpha=0.20, zorder=0)
            if label not in legend_labels:
                legend_handles.append(Patch(facecolor=color, alpha=0.20, edgecolor="none"))
                legend_labels.append(label)

        truth = target_points.drop_duplicates(subset=["sample_file", "kind"]).sort_values("timestamp")
        truth_within = truth[~truth["truth_out_of_limit"].fillna(False)].copy()
        truth_out = truth[truth["truth_out_of_limit"].fillna(False)].copy()
        if not truth_within.empty:
            ax.scatter(
                truth_within["timestamp"],
                truth_within[truth_value_column],
                color="black",
                s=18,
                alpha=0.95,
                linewidths=0.25,
                edgecolors="black",
                zorder=3,
            )
        if not truth_out.empty:
            ax.scatter(
                truth_out["timestamp"],
                truth_out[truth_value_column],
                color="#d00000",
                s=28,
                alpha=0.95,
                marker="x",
                linewidths=1.0,
                zorder=4,
            )
        if "Measured value (within limit)" not in legend_labels:
            legend_handles.append(plt.Line2D([0], [0], color="black", linestyle="", marker="o", markersize=6))
            legend_labels.append("Measured value (within limit)")
        if "Measured value (out of limit)" not in legend_labels:
            legend_handles.append(plt.Line2D([0], [0], color="#d00000", linestyle="", marker="x", markersize=7))
            legend_labels.append("Measured value (out of limit)")

        for model_type in _MODEL_ORDER:
            model_points = target_points[target_points["model_type"] == model_type].copy()
            if model_points.empty:
                continue
            model_points = model_points.sort_values("timestamp")
            pred_out_mask = limit_exceedance_mask(
                pd.to_numeric(model_points["pred_full_value"], errors="coerce").to_numpy(dtype=float),
                upper=upper,
                lower=lower,
            ) if (np.isfinite(upper) or np.isfinite(lower)) else np.zeros(len(model_points), dtype=bool)
            model_within = model_points.loc[~pred_out_mask].copy()
            model_out = model_points.loc[pred_out_mask].copy()
            ax.plot(
                model_points["timestamp"],
                model_points[pred_value_column],
                color=_MODEL_COLORS[model_type],
                linewidth=1.2,
                alpha=0.78,
                zorder=1.8,
            )
            if not model_within.empty:
                ax.scatter(
                    model_within["timestamp"],
                    model_within[pred_value_column],
                    color=_MODEL_COLORS[model_type],
                    s=12,
                    alpha=0.90,
                    marker="o",
                    linewidths=0.0,
                    zorder=2.2,
                )
            if not model_out.empty:
                ax.scatter(
                    model_out["timestamp"],
                    model_out[pred_value_column],
                    color=_MODEL_COLORS[model_type],
                    s=24,
                    alpha=0.95,
                    marker="x",
                    linewidths=1.0,
                    zorder=2.6,
                )
            if model_type not in legend_labels:
                legend_handles.append(plt.Line2D([0], [0], color=_MODEL_COLORS[model_type], linewidth=1.4, marker="o", markersize=5))
                legend_labels.append(model_type)

        if show_limit_lines and np.isfinite(upper):
            ax.axhline(upper, color="#ff0000", linewidth=0.8, linestyle="-", alpha=0.9)
            if "Upper limit" not in legend_labels:
                legend_handles.append(plt.Line2D([0], [0], color="#ff0000", linewidth=0.8))
                legend_labels.append("Upper limit")
        if show_limit_lines and np.isfinite(lower):
            ax.axhline(lower, color="#2ca02c", linewidth=0.8, linestyle="-", alpha=0.9)
            if "Lower limit" not in legend_labels:
                legend_handles.append(plt.Line2D([0], [0], color="#2ca02c", linewidth=0.8))
                legend_labels.append("Lower limit")

        ax.set_ylabel(target_label, rotation=0, ha="right", va="center", labelpad=24, fontsize=y_label_font_size)
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.5)
        ax.tick_params(axis="x", rotation=25, labelsize=x_label_font_size)
        ax.tick_params(axis="y", labelsize=y_value_font_size)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.988), ncol=min(max(len(legend_labels), 1), 6), framealpha=0.85, fontsize=y_label_font_size, borderaxespad=0.06)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.942, bottom=0.06, hspace=0.34)
    fig.savefig(out_dir / output_filename, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _write_figures(class_df: pd.DataFrame, points_df: pd.DataFrame, coverage_df: pd.DataFrame, out_dir: Path, evaluation_scope: str) -> None:
    _write_metric_figures(class_df, out_dir, evaluation_scope)
    _write_overlay_figure(
        points_df,
        coverage_df,
        out_dir,
        evaluation_scope,
        truth_value_column="truth_full_value",
        pred_value_column="pred_full_value",
        output_filename="classification_timeseries_overlay.png",
        show_limit_lines=True,
    )
    _write_overlay_figure(
        points_df,
        coverage_df,
        out_dir,
        evaluation_scope,
        truth_value_column="truth_res_value",
        pred_value_column="pred_res_value",
        output_filename="classification_timeseries_overlay_res.png",
        show_limit_lines=False,
    )


def post_process_run(
    run_dir: Path,
    summaries_subdir: str = "classification",
    sweep_namespace: str = "feature_sweeps",
    targets: list[str] | None = None,
    make_figures: bool = True,
    evaluation_scope: str = "test",
    exclude_no_out_of_limit: bool = False,
) -> int:
    if not run_dir.is_dir():
        print(f"[ERROR] --run-dir does not exist: {run_dir}")
        return 1
    if evaluation_scope not in {"test", "combined"}:
        print(f"[ERROR] Unsupported evaluation scope: {evaluation_scope}")
        return 1

    limits_csv = Path(__file__).resolve().parent.parent / "data" / "input" / "Limits.csv"
    limits_records = load_limits_records(limits_csv)
    out_dir = run_dir / "summaries" / summaries_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    points_frames = []
    target_dirs = _list_target_dirs(run_dir, targets)
    if not target_dirs:
        print("[ERROR] No target directories found.")
        return 1

    sample_cache: dict = {}
    for target_dir in target_dirs:
        metrics_df = _read_metrics_csv(target_dir, sweep_namespace)
        if metrics_df is None or metrics_df.empty:
            print(f"[WARN] No feature_sweep_final_metrics.csv for {target_dir.name}; skipping.")
            continue

        filtered_df = _filter_candidate_rows(metrics_df)
        if filtered_df.empty:
            print(f"[WARN] No usable candidate rows for {target_dir.name}; skipping.")
            continue

        target_name = str(filtered_df.iloc[0].get("target", target_dir.name))
        target_label = clean_target_label(target_name, "MC") or clean_target_label(target_dir.name, "MC") or target_name
        target_name = _normalize_dataset_name(target_name) if str(target_name).endswith("_res") else target_name
        limit_spec = _build_target_limit_spec(target_name, target_dir.name, limits_records)

        for selected_row in _select_best_rows(filtered_df):
            summary_row, points_df = _evaluate_selected_row(
                target_dir=target_dir,
                target_name=target_name,
                target_label=target_label,
                sweep_namespace=sweep_namespace,
                row=selected_row,
                metrics_df=filtered_df,
                limit_spec=limit_spec,
                evaluation_scope=evaluation_scope,
                sample_cache=sample_cache,
            )
            rows.append(summary_row)
            if not points_df.empty:
                points_df["dataset"] = target_dir.name
                points_df["target_label"] = target_label
                points_frames.append(points_df)

    if not rows:
        print("[ERROR] No classification rows were generated.")
        return 1

    all_class_df = pd.DataFrame(rows)
    all_class_df["model_type"] = pd.Categorical(all_class_df["model_type"], categories=_MODEL_ORDER, ordered=True)
    all_class_df = all_class_df.sort_values(["target_label", "model_type"]).reset_index(drop=True)
    all_points_df = pd.concat(points_frames, ignore_index=True) if points_frames else pd.DataFrame()

    exclusion_map = {}
    for target_name, group in all_class_df.groupby("target", sort=False):
        group_pos = pd.to_numeric(group["n_scope_positive"], errors="coerce").fillna(0)
        excluded_reason = ""
        if exclude_no_out_of_limit and float(group_pos.max()) <= 0:
            excluded_reason = f"excluded_no_out_of_limit_{evaluation_scope}"
        exclusion_map[target_name] = excluded_reason

    all_class_df["excluded_reason"] = all_class_df["target"].map(exclusion_map).fillna("")
    if not all_points_df.empty:
        all_points_df["excluded_reason"] = all_points_df["target"].map(exclusion_map).fillna("")

    class_df = all_class_df[all_class_df["excluded_reason"].astype(str) == ""].copy()
    points_df = all_points_df[all_points_df["excluded_reason"].astype(str) == ""].copy() if not all_points_df.empty else pd.DataFrame()
    coverage_df = _build_threshold_coverage_rows(class_df, all_class_df)
    if not coverage_df.empty:
        coverage_df = coverage_df.sort_values(["target_label"]).reset_index(drop=True)

    by_target_model_path = out_dir / "summary_classification_by_target_model.csv"
    best_per_type_path = out_dir / "summary_classification_best_per_type.csv"
    coverage_path = out_dir / "summary_classification_threshold_coverage.csv"
    points_path = out_dir / "classification_overlay_points.csv"

    class_df.to_csv(by_target_model_path, index=False)
    class_df.to_csv(best_per_type_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    (points_df if not points_df.empty else pd.DataFrame()).to_csv(points_path, index=False)

    print(f"[INFO] Wrote {by_target_model_path}")
    print(f"[INFO] Wrote {best_per_type_path}")
    print(f"[INFO] Wrote {coverage_path}")
    print(f"[INFO] Wrote {points_path}")

    if make_figures:
        _write_figures(class_df, points_df, coverage_df[coverage_df["excluded_reason"].astype(str) == ""].copy(), out_dir, evaluation_scope)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-process threshold-based classification metrics from feature-sweep predictions.")
    parser.add_argument("--run-dir", required=True, help="Run directory such as data/output/CV14")
    parser.add_argument("--summaries-subdir", default="classification", help="Subdirectory under <run-dir>/summaries for output files (default: classification).")
    parser.add_argument("--targets", nargs="*", default=None, help="Optional target directory names to limit processing.")
    parser.add_argument("--sweep-namespace", default="feature_sweeps", help="Forecast sweep subdirectory under forecasts/ (default: feature_sweeps).")
    parser.add_argument("--evaluation-scope", choices=["test", "combined"], default="test", help="Evaluate saved test predictions only, or recompute train+test combined metrics.")
    parser.add_argument("--exclude-no-out-of-limit", action="store_true", help="Exclude targets whose selected evaluation scope has zero out-of-limit ground-truth values.")
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG figure generation.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_dir = _resolve(args.run_dir)
    print(f"[INFO] run_dir              : {run_dir}")
    print(f"[INFO] summaries_subdir    : {args.summaries_subdir}")
    print(f"[INFO] sweep_namespace     : {args.sweep_namespace}")
    print(f"[INFO] evaluation_scope    : {args.evaluation_scope}")
    print(f"[INFO] exclude_no_out_limit: {bool(args.exclude_no_out_of_limit)}")
    return post_process_run(
        run_dir=run_dir,
        summaries_subdir=str(args.summaries_subdir),
        sweep_namespace=str(args.sweep_namespace),
        targets=args.targets,
        make_figures=not bool(args.no_figures),
        evaluation_scope=str(args.evaluation_scope),
        exclude_no_out_of_limit=bool(args.exclude_no_out_of_limit),
    )


if __name__ == "__main__":
    raise SystemExit(main())

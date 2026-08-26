"""
Shared configuration and data-path utilities used by both e_Train.py and f_Evaluate.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import yaml


def _effective_sample_size(n_samples: int, n_mc_replicates: int = 1) -> int:
    """Effective independent sample count, discounting MC replicates.

    When samples are Monte Carlo replicates of the same underlying
    observations, divides by the replicate count to estimate the number
    of truly independent data points.  Returns at least 1.
    """
    return max(1, n_samples // max(1, n_mc_replicates))


NORMALIZATION_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "output"
    / "sensors"
    / "normalization.json"
)


def _default_aggregate_offset_csv():
    root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        root / "data" / "output" / "calibration" / "aggregate" / "offset_gain_model_results.csv",
        root / "data" / "output" / "calibration" / "summaries" / "aggregate" / "offset_gain_model_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_aggregate_sensor_name(name):
    text = str(name).strip().replace("Âµ", "µ")
    if "Sp Cond" in text:
        return "Sp Cond (microS_cm)"
    return text


def _load_aggregate_offset_map(aggregate_csv_path, verbose=True):
    """
    Load aggregate calibration offsets by canonical sensor name.

    Returns dict: canonical_sensor_name -> np.ndarray(offsets)
    """
    if aggregate_csv_path is None:
        return {}

    agg_path = Path(aggregate_csv_path)
    if not agg_path.exists():
        if verbose:
            print(f"[WARN] Aggregate offset CSV not found: {agg_path}")
        return {}

    try:
        agg_df = pd.read_csv(agg_path)
    except Exception as exc:
        if verbose:
            print(f"[WARN] Could not read aggregate offset CSV {agg_path}: {exc}")
        return {}

    if "Sensor" not in agg_df.columns or "Offset" not in agg_df.columns:
        if verbose:
            print(f"[WARN] Aggregate offset CSV missing Sensor/Offset columns: {agg_path}")
        return {}

    agg_df = agg_df.copy()
    agg_df["Sensor_Normalized"] = agg_df["Sensor"].map(_normalize_aggregate_sensor_name)
    agg_df["Sensor_Canonical"] = agg_df["Sensor_Normalized"].map(_canonical_feature_name)
    agg_df["Offset"] = pd.to_numeric(agg_df["Offset"], errors="coerce")

    out = {}
    for canonical_name, group in agg_df.groupby("Sensor_Canonical"):
        offsets = group["Offset"].dropna().to_numpy(dtype=float)
        if offsets.size >= 3:
            out[str(canonical_name)] = offsets

    return out


def _normalize_std_for_feature(raw_std, feature_name, norm_params):
    """Scale a raw-space standard deviation into normalized space when possible."""
    std = float(raw_std)
    if std <= 0:
        return 0.0, False
    if feature_name not in norm_params:
        return std, False

    v_min = norm_params[feature_name].get("min", 0)
    v_max = norm_params[feature_name].get("max", 1)
    v_range = v_max - v_min
    if v_range in (0, 0.0):
        return std, False
    return std / v_range, True


def _fit_t_or_fallback_std(offsets):
    """Return an estimated standard deviation from fitted Student's t or empirical fallback."""
    offsets = np.asarray(offsets, dtype=float)
    if offsets.size < 3:
        return float(np.std(offsets, ddof=0)) if offsets.size > 0 else 0.0

    try:
        t_df, _t_loc, t_scale = stats.t.fit(offsets)
        if np.isfinite(t_df) and np.isfinite(t_scale) and t_scale > 0 and t_df > 2:
            return float(np.sqrt((t_scale ** 2) * (t_df / (t_df - 2.0))))
    except Exception:
        pass

    return float(np.std(offsets, ddof=0))


def _sample_offset_deltas(
    offsets,
    n_samples,
    rng,
    mode,
):
    """
    Sample delta offsets (e1 - e2) used by uncertain-input MC kernel expectation.
    """
    offsets = np.asarray(offsets, dtype=float)
    n_samples = int(max(1, n_samples))

    if offsets.size == 0:
        return np.zeros(n_samples, dtype=np.float32)

    if mode == "aggregate_t":
        try:
            t_df, t_loc, t_scale = stats.t.fit(offsets)
            if np.isfinite(t_df) and np.isfinite(t_loc) and np.isfinite(t_scale) and t_scale > 0:
                e1 = stats.t.rvs(df=t_df, loc=t_loc, scale=t_scale, size=n_samples, random_state=rng)
                e2 = stats.t.rvs(df=t_df, loc=t_loc, scale=t_scale, size=n_samples, random_state=rng)
                return (e1 - e2).astype(np.float32)
        except Exception:
            pass

    if mode in {"aggregate_t", "aggregate_empirical"}:
        e1 = rng.choice(offsets, size=n_samples, replace=True)
        e2 = rng.choice(offsets, size=n_samples, replace=True)
        return (e1 - e2).astype(np.float32)

    std = float(np.std(offsets, ddof=0))
    if std <= 0:
        return np.zeros(n_samples, dtype=np.float32)
    return rng.normal(loc=0.0, scale=np.sqrt(2.0) * std, size=n_samples).astype(np.float32)


def load_config(config_path):
    """Load configuration from YAML or JSON file, storing the config directory."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1252") as f:
            raw_text = f.read()

    if path.suffix in (".yaml", ".yml"):
        config = yaml.safe_load(raw_text)
    elif path.suffix == ".json":
        config = json.loads(raw_text)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")

    config["__config_dir"] = str(path.resolve().parent)
    return config


def _resolve_path_from_config(path_value, config_dir):
    """Resolve a path value relative to the config file directory."""
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (Path(config_dir) / path_obj).resolve()


def _resolve_data_paths(data_cfg, config_dir):
    """Resolve base data directory and sample subdirectory with backward compatibility."""
    configured_subdir = data_cfg.get("sample_subdir")
    data_dir_path = _resolve_path_from_config(data_cfg["data_dir"], config_dir)

    if configured_subdir:
        return str(data_dir_path), configured_subdir

    # Backward compatibility: if data_dir points directly to the samples folder,
    # infer parent + subdir.
    if data_dir_path.name in {"samples", "mc_replicates"}:
        return str(data_dir_path.parent), data_dir_path.name

    return str(data_dir_path), "samples"


def _canonical_feature_name(name):
    """Normalise a sensor/feature name to a canonical lowercase form for matching."""
    text = str(name).strip().lower().replace("µ", "u")
    text = text.replace("micro", "u")
    text = text.replace("_", " ")
    if " - " in text:
        text = text.split(" - ", 1)[1].strip()
    for token in ("(", ")", "/", "%", "°", "-", ".", ","):
        text = text.replace(token, " ")
    return " ".join(text.split())


def _resolve_summary_dir(hyper_cfg, config_dir):
    """Return the uncertainty summary directory, resolving relative to config if set."""
    if hyper_cfg.get("uncertainty_summary_dir"):
        return _resolve_path_from_config(hyper_cfg["uncertainty_summary_dir"], config_dir)
    # Default: <project_root>/data/output/calibration/summaries
    # __file__ is utils/config_utils.py → parent = utils/ → parent = src/ → parent = project root
    return Path(__file__).parent.parent.parent / "data" / "output" / "calibration" / "summaries"


def _load_uncertainty_std_map(summary_dir, verbose=True):
    """
    Load per-sensor offset-std values from *_uncertainty_summary.csv files.

    Returns a dict mapping canonical sensor name → dict with keys
    ``"offset_std"``, ``"sensor_name"``, ``"source_file"``.
    """
    if not summary_dir.exists():
        if verbose:
            print(f"[WARN] Uncertainty summary directory not found: {summary_dir}")
        return {}

    summary_map = {}
    for file_path in summary_dir.rglob("*_uncertainty_summary.csv"):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            row = df.iloc[0]
            sensor_name = row.get("Sensor")
            if pd.isna(sensor_name):
                continue
            offset_std = row.get("Offset_Std", 0.0)
            if pd.isna(offset_std):
                offset_std = 0.0
            canonical_name = _canonical_feature_name(sensor_name)
            if canonical_name in summary_map:
                if verbose:
                    print(
                        f"[WARN] Duplicate uncertainty entry for '{sensor_name}' "
                        f"(canonical='{canonical_name}'). Keeping first source: "
                        f"{summary_map[canonical_name]['source_file']}"
                    )
                continue
            summary_map[canonical_name] = {
                "offset_std": float(offset_std),
                "sensor_name": str(sensor_name),
                "source_file": str(file_path),
            }
        except Exception as exc:
            if verbose:
                print(f"[WARN] Could not parse uncertainty summary file {file_path}: {exc}")
    return summary_map


def _build_feature_uncertainty_variance(data_cfg, hyper_cfg, config_dir, verbose=True):
    """
    Compute per-feature input uncertainty variances for the GP uncertain-input kernel.

    Variances are derived from sensor offset_std values in the uncertainty summary
    directory, scaled by the normalization range when a normalization.json is present.
    Returns a 1-D float32 array of length ``n_features * seq_len``.
    """
    bundle = _build_feature_uncertainty_bundle(data_cfg, hyper_cfg, config_dir, verbose=verbose)
    return bundle["feature_variances"]


def _build_feature_uncertainty_bundle(data_cfg, hyper_cfg, config_dir, verbose=True):
    """
    Build uncertainty information for uncertain-input GP kernels.

    Returns dict with keys:
    - feature_variances: flattened per-feature variance for each input timestep
    - noise_delta_samples: MC delta offsets (e1 - e2) aligned to flattened input shape
    - source_mode_effective: applied source mode string
    - source_details: per-feature source diagnostics
    """
    input_columns = data_cfg["input_columns"]
    seq_len = data_cfg["input_row_2"] - data_cfg["input_row_1"]
    # The per-feature variance is tiled to match the flattened input, so it has to
    # follow the aggregation. When the window is averaged to a single row before
    # training, the flattened input holds one value per predictor rather than one
    # per predictor per timestep, and tiling across timesteps would leave the
    # variance vector seq_len times too long for the data it describes.
    if str(data_cfg.get("input_aggregation", "none")).lower() == "mean":
        seq_len = 1

    n_mc_samples = int(hyper_cfg.get("uncertain_kernel_mc_samples", 64))
    mc_seed = int(hyper_cfg.get("uncertain_kernel_mc_seed", 0))
    source_mode_requested = str(hyper_cfg.get("uncertainty_source_mode", "aggregate_t")).lower()
    if source_mode_requested not in {"aggregate_t", "aggregate_empirical", "summary_std"}:
        source_mode_requested = "aggregate_t"

    summary_dir = _resolve_summary_dir(hyper_cfg, config_dir)
    summary_std_map = _load_uncertainty_std_map(summary_dir, verbose=verbose)
    aggregate_csv_path = hyper_cfg.get("uncertainty_aggregate_csv")
    if aggregate_csv_path:
        aggregate_csv_path = _resolve_path_from_config(aggregate_csv_path, config_dir)
    else:
        aggregate_csv_path = _default_aggregate_offset_csv()
    aggregate_offsets_map = _load_aggregate_offset_map(aggregate_csv_path, verbose=verbose)

    if verbose:
        print(f"[INFO] Uncertainty summary directory: {summary_dir}")
        print(f"[INFO] Uncertainty aggregate CSV: {aggregate_csv_path}")

    source_mode_effective = source_mode_requested
    if source_mode_requested in {"aggregate_t", "aggregate_empirical"} and not aggregate_offsets_map:
        source_mode_effective = "summary_std"
        if verbose:
            print(
                "[WARN] Aggregate offsets unavailable; falling back to summary_std uncertainty mode for GP kernel."
            )

    norm_path = NORMALIZATION_OUTPUT_PATH
    norm_params = {}
    if norm_path.exists():
        try:
            with open(norm_path, "r") as f:
                norm_params = json.load(f)
        except Exception as exc:
            print(f"[WARN] Could not read normalization.json at {norm_path}: {exc}")

    rng = np.random.default_rng(mc_seed)
    feature_variances = []
    feature_deltas = []
    source_details = []

    for feature in input_columns:
        candidates = [_canonical_feature_name(feature)]
        if " - " in feature:
            candidates.append(_canonical_feature_name(feature.split(" - ", 1)[1]))

        matched_key = None
        for candidate in candidates:
            if candidate in aggregate_offsets_map or candidate in summary_std_map:
                matched_key = candidate
                break

        raw_std = 0.0
        sensor_source = "none"
        scaling_applied = False
        deltas = np.zeros(n_mc_samples, dtype=np.float32)

        if matched_key is not None and source_mode_effective in {"aggregate_t", "aggregate_empirical"}:
            offsets = aggregate_offsets_map.get(matched_key)
            if offsets is not None and offsets.size >= 3:
                raw_std = _fit_t_or_fallback_std(offsets)
                deltas = _sample_offset_deltas(offsets, n_mc_samples, rng, source_mode_effective)
                sensor_source = f"aggregate:{source_mode_effective}"

        if sensor_source == "none":
            matched_entry = summary_std_map.get(matched_key) if matched_key is not None else None
            raw_std = 0.0 if matched_entry is None else float(matched_entry["offset_std"])
            if raw_std > 0:
                deltas = rng.normal(loc=0.0, scale=np.sqrt(2.0) * raw_std, size=n_mc_samples).astype(np.float32)
            sensor_source = "summary_std" if matched_entry is not None else "none"

        applied_std, scaling_applied = _normalize_std_for_feature(raw_std, feature, norm_params)
        if scaling_applied:
            v_min = norm_params[feature].get("min", 0)
            v_max = norm_params[feature].get("max", 1)
            v_range = v_max - v_min
            if v_range not in (0, 0.0):
                deltas = (deltas / float(v_range)).astype(np.float32)

        variance = float(applied_std ** 2)
        feature_variances.append(variance)
        feature_deltas.append(deltas)

        source_details.append(
            {
                "feature": feature,
                "matched_key": matched_key,
                "source": sensor_source,
                "raw_std": float(raw_std),
                "applied_std": float(applied_std),
                "variance": variance,
                "normalization_scaled": bool(scaling_applied),
            }
        )

        if verbose:
            print(
                f"  [UNCERTAINTY] feature='{feature}' | matched_key='{matched_key}' | "
                f"source='{sensor_source}' | raw_std={raw_std:.6g} | "
                f"applied_std={applied_std:.6g} | var={variance:.6g} | "
                f"scaled={'yes' if scaling_applied else 'no'}"
            )

    feature_variances_arr = np.array(feature_variances, dtype=np.float32)
    feature_delta_samples = np.stack(feature_deltas, axis=1) if feature_deltas else np.zeros((n_mc_samples, 0), dtype=np.float32)

    return {
        "feature_variances": np.tile(feature_variances_arr, seq_len),
        "noise_delta_samples": np.tile(feature_delta_samples, (1, seq_len)),
        "source_mode_requested": source_mode_requested,
        "source_mode_effective": source_mode_effective,
        "aggregate_csv_path": str(aggregate_csv_path) if aggregate_csv_path is not None else None,
        "summary_dir": str(summary_dir),
        "source_details": source_details,
    }


def select_best_model_row(df: "pd.DataFrame") -> "pd.Series":
    """Return the best-model row from a pre-filtered metrics DataFrame.

    Selection criterion (applied in order):
    1. Among rows with r2 > 0 AND min_skill_rmse > 0 (beats baseline),
       pick the one with the highest r2.
    2. If no row clears both gates (or min_skill_rmse is absent), fall back
       to the row with the highest r2 unconditionally.

    The caller is responsible for pre-filtering: baseline rows, rows with
    non-finite r2, and any minimum-sample-count requirements should be
    removed before calling this function.
    """
    r2_vals = pd.to_numeric(df["r2"], errors="coerce")
    if "min_skill_rmse" in df.columns:
        skill_vals = pd.to_numeric(df["min_skill_rmse"], errors="coerce")
        valid_mask = (skill_vals > 0) & (r2_vals > 0)
        if valid_mask.any():
            return df.loc[r2_vals[valid_mask].idxmax()]
    return df.loc[r2_vals.idxmax()]

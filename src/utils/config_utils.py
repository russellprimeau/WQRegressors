"""
Shared configuration and data-path utilities used by both e_Train.py and f_Evaluate.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


NORMALIZATION_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "output"
    / "sensors"
    / "normalization.json"
)


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
    input_columns = data_cfg["input_columns"]
    seq_len = data_cfg["input_row_2"] - data_cfg["input_row_1"]

    summary_dir = _resolve_summary_dir(hyper_cfg, config_dir)
    summary_std_map = _load_uncertainty_std_map(summary_dir, verbose=verbose)
    if verbose:
        print(f"[INFO] Uncertainty source directory: {summary_dir}")

    norm_path = NORMALIZATION_OUTPUT_PATH
    norm_params = {}
    if norm_path.exists():
        try:
            with open(norm_path, "r") as f:
                norm_params = json.load(f)
        except Exception as exc:
            print(f"[WARN] Could not read normalization.json at {norm_path}: {exc}")

    feature_variances = []
    for feature in input_columns:
        candidates = [_canonical_feature_name(feature)]
        if " - " in feature:
            candidates.append(_canonical_feature_name(feature.split(" - ", 1)[1]))

        matched_entry = None
        matched_key = None
        for candidate in candidates:
            if candidate in summary_std_map:
                matched_entry = summary_std_map[candidate]
                matched_key = candidate
                break

        raw_std = 0.0 if matched_entry is None else float(matched_entry["offset_std"])
        applied_std = raw_std
        scaling_applied = False

        if applied_std > 0 and feature in norm_params:
            v_min = norm_params[feature].get("min", 0)
            v_max = norm_params[feature].get("max", 1)
            v_range = v_max - v_min
            if v_range not in (0, 0.0):
                applied_std = applied_std / v_range
                scaling_applied = True

        variance = float(applied_std ** 2)
        feature_variances.append(variance)

        if verbose:
            if matched_entry is None:
                print(
                    f"  [UNCERTAINTY] feature='{feature}' | source='none' | "
                    f"matched_key='none' | raw_offset_std=0 | applied_std=0 | var=0"
                )
            else:
                scaling_note = "scaled_by_normalization" if scaling_applied else "no_scaling"
                print(
                    f"  [UNCERTAINTY] feature='{feature}' | "
                    f"source='{matched_entry['source_file']}' | "
                    f"sensor='{matched_entry['sensor_name']}' | "
                    f"matched_key='{matched_key}' | "
                    f"raw_offset_std={raw_std:.6g} | applied_std={applied_std:.6g} | "
                    f"var={variance:.6g} | {scaling_note}"
                )

    return np.tile(np.array(feature_variances, dtype=np.float32), seq_len)

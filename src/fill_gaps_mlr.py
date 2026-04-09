"""Fill gaps in Consolidated_sparse.csv using Multiple Linear Regression.

For each of the 14 Eurofins target columns, trains 3 MLR variants on the first
70% of observed (non-blank) rows:
  - "mlr"       : single row at sample time
  - "mlr_avg12" : nanmean of last 12 rows up to sample time
  - "mlr_avgall": nanmean of all rows since the previous observed sample

An additional predictor is included for each target: the lagged value of the
target's own _state column, offset back by a per-target number of rows equal
to the median inter-observation gap (default: 168 rows).  This gives the model
the most recently known target value at a realistic lag.  For avg12 the lagged
_state is averaged over the same 12-row window ending at (i - offset); for
avgall it is averaged over the inter-measurement window ending at (i - offset).

Records model metadata (predictors, weights, R², RMSE) in
  data/output/regression/mlr_model_summary.csv

Writes predictions for blank target rows into the corresponding <target>_state
column of a copy called
  data/output/regression/Consolidated_filled.csv

The 30% holdout is reserved and never touched by this script.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

# Ensure src/ is on the path so utils.mlr is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.mlr import fit_and_predict, select_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "output", "regression", "Consolidated_sparse.csv"
)
OUTPUT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "output", "regression", "Consolidated_filled.csv"
)
SUMMARY_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "output", "regression", "mlr_model_summary.csv"
)

TARGET_COLUMNS = [
    "Color",
    "Turbidity (FNU)",
    "E.coli (CFU/100mL)",
    "Intestinal enterococci (CFU/100mL)",
    "Colony Count 22°C (CFU/mL)",
    "Total coliforms 37°C (CFU/100mL)",
    "Arsenic (µg/L)",
    "Lead (µg/L)",
    "Cadmium (µg/L)",
    "Copper filtered (mg/L)",
    "Chromium (µg/L)",
    "Nickel (µg/L)",
    "Zinc (µg/L)",
    "pH",
]

PREDICTOR_COLUMNS = [
    "Pfl - Water temperature (°C)",
    "Pfl - Sp Cond (microS_cm)",
    "Pfl - pH",
    "Pfl - DO (% Sat)",
    "Pfl - Turbidity (FNU)",
    "Pfl - fDOM (RFU)",
    "Pfl - fDOM (QSU)",
    "Wind speed x (m/s)",
    "Wind speed y (m/s)",
    "Atmospheric pressure (mBar)",
    "Longwave (IR) radiation (W/m2)",
    "Shortwave (solar) radiation (W/m2)",
    "24hr precipitation total (mm)",
    "Air temperature (°C)",
    "Humidity (%)",
    "SCADA - pH",
    "SCADA - Temperature (°C)",
]

VARIANTS = [
    {"name": "mlr",        "mode": "last"},
    {"name": "mlr_avg12",  "mode": "avg12"},
    {"name": "mlr_avgall", "mode": "avgall"},
]

TRAIN_FRACTION = 0.70
MIN_OBSERVED = 5
DEFAULT_STATE_OFFSET = 168   # rows; fallback when gap cannot be computed

# ---------------------------------------------------------------------------
# Predictor group exclusion
# ---------------------------------------------------------------------------
# Set of column-name prefixes whose columns are excluded from predictors.
# Supported group names (keys) and their prefixes:
#   "surface"  -> columns starting with "Pfl - "
#   "scada"    -> columns starting with "SCADA - "
# Set to an empty set to use all PREDICTOR_COLUMNS.
EXCLUDE_PREDICTOR_GROUPS: set = set()
# EXCLUDE_PREDICTOR_GROUPS = {"surface"}
# Examples:
#   EXCLUDE_PREDICTOR_GROUPS = {"surface"}
#   EXCLUDE_PREDICTOR_GROUPS = {"scada"}
#   EXCLUDE_PREDICTOR_GROUPS = {"surface", "scada"}

_GROUP_PREFIXES = {
    "surface": "Pfl - ",
    "scada":   "SCADA - ",
}


def _active_predictor_columns() -> list:
    """Return PREDICTOR_COLUMNS with any excluded groups removed."""
    excluded_prefixes = tuple(
        _GROUP_PREFIXES[g] for g in EXCLUDE_PREDICTOR_GROUPS if g in _GROUP_PREFIXES
    )
    if not excluded_prefixes:
        return list(PREDICTOR_COLUMNS)
    return [c for c in PREDICTOR_COLUMNS if not c.startswith(excluded_prefixes)]


# ---------------------------------------------------------------------------
# Monte Carlo uncertainty perturbation
# ---------------------------------------------------------------------------
# Set USE_MC_PERTURBATION = True to augment training data with K perturbed
# copies of each training sample, using the calibration-derived Student's t
# offset distributions for the six surface sensor (Pfl - *) columns.
# Only the surface columns are perturbed; all other predictors are unchanged.
USE_MC_PERTURBATION: bool = True
MC_REPLICATES: int = 10   # number of perturbed copies per training sample

# Mapping from calibration sensor name -> Pfl - column name
_SURFACE_SENSOR_MAP = {
    "Sp Cond (microS_cm)": "Pfl - Sp Cond (microS_cm)",
    "pH":                   "Pfl - pH",
    "DO (% Sat)":           "Pfl - DO (% Sat)",
    "Turbidity (FNU)":      "Pfl - Turbidity (FNU)",
    "fDOM (RFU)":           "Pfl - fDOM (RFU)",
    "fDOM (QSU)":           "Pfl - fDOM (QSU)",
}


def _default_aggregate_offset_csv() -> Path | None:
    """Return path to offset_gain_model_results.csv, checking both known locations."""
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "data" / "output" / "calibration" / "aggregate" / "offset_gain_model_results.csv",
        root / "data" / "output" / "calibration" / "summaries" / "aggregate" / "offset_gain_model_results.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_surface_uncertainty_specs() -> dict:
    """Load Student's t offset parameters for the six surface sensor columns.

    Returns a dict mapping Pfl column name -> {t_df, t_loc, t_scale}, or an
    empty dict if the calibration data cannot be found.  Mirrors the logic in
    d_RunResample._fit_offset_t_params_from_aggregate() and
    _load_and_prepare_sensor_uncertainties(), but operates in raw (physical)
    units — no normalization is needed because MLR works in raw space.
    """
    agg_csv = _default_aggregate_offset_csv()
    if agg_csv is None:
        warnings.warn(
            "MC perturbation requested but offset_gain_model_results.csv not found; "
            "USE_MC_PERTURBATION will have no effect.",
            RuntimeWarning, stacklevel=2,
        )
        return {}

    agg_df = pd.read_csv(agg_csv)
    if "Sensor" not in agg_df.columns or "Offset" not in agg_df.columns:
        warnings.warn(
            f"offset_gain_model_results.csv missing Sensor/Offset columns ({agg_csv}); "
            "MC perturbation disabled.",
            RuntimeWarning, stacklevel=2,
        )
        return {}

    def _norm(name: str) -> str:
        text = str(name).strip().replace("Âµ", "µ")
        return "Sp Cond (microS_cm)" if "Sp Cond" in text else text

    agg_df = agg_df.copy()
    agg_df["_sensor"] = agg_df["Sensor"].map(_norm)
    agg_df["Offset"] = pd.to_numeric(agg_df["Offset"], errors="coerce")

    specs = {}
    for sensor_name, col_name in _SURFACE_SENSOR_MAP.items():
        offsets = agg_df.loc[agg_df["_sensor"] == sensor_name, "Offset"].dropna().to_numpy(float)
        if offsets.size < 3:
            warnings.warn(
                f"Insufficient calibration data for {sensor_name} (n={offsets.size}); "
                "column will not be perturbed.",
                RuntimeWarning, stacklevel=2,
            )
            continue
        try:
            t_df, t_loc, t_scale = stats.t.fit(offsets)
        except Exception as exc:
            warnings.warn(f"Student's t fit failed for {sensor_name}: {exc}", RuntimeWarning, stacklevel=2)
            continue
        if not (np.isfinite(t_df) and np.isfinite(t_loc) and np.isfinite(t_scale) and t_scale > 0):
            warnings.warn(f"Invalid t params for {sensor_name}; column will not be perturbed.", RuntimeWarning, stacklevel=2)
            continue
        specs[col_name] = {"t_df": float(t_df), "t_loc": float(t_loc), "t_scale": float(t_scale)}

    return specs


def _perturb_predictor_row(row: np.ndarray, predictor_cols: list,
                            uncertainty_specs: dict, rng: np.random.Generator) -> np.ndarray:
    """Return a single perturbed copy of an aggregated predictor row (1-D array).

    For each surface column present in *uncertainty_specs*, draws one additive
    offset from its Student's t distribution and adds it to the corresponding
    element.  All other elements are unchanged.
    """
    perturbed = row.copy()
    for col_name, spec in uncertainty_specs.items():
        if col_name not in predictor_cols:
            continue
        idx = predictor_cols.index(col_name)
        if not np.isfinite(perturbed[idx]):
            continue
        offset = float(stats.t.rvs(
            df=spec["t_df"], loc=spec["t_loc"], scale=spec["t_scale"],
            random_state=rng,
        ))
        perturbed[idx] += offset
    return perturbed


# ---------------------------------------------------------------------------
# Per-target offset computation
# ---------------------------------------------------------------------------

def _compute_state_offset(df, target_col):
    """Return the median inter-observation row gap for *target_col*.

    This is used as the lag (in rows) applied to the _state predictor so that
    the model always sees the previous measurement at a realistic temporal
    distance.  Falls back to DEFAULT_STATE_OFFSET when fewer than 2 valid
    observations exist.
    """
    obs_ilocs = np.where(df[target_col].notna().values)[0]
    if len(obs_ilocs) < 2:
        return DEFAULT_STATE_OFFSET
    gaps = np.diff(obs_ilocs)
    offset = int(round(float(np.median(gaps))))
    return max(1, offset)


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def _build_window(df_pred, row_idx, mode, prev_obs_idx, state_offset=0):
    """Return a 2D predictor array X for one sample.

    Parameters
    ----------
    df_pred : DataFrame — rows × predictor columns (float values)
    row_idx : int — positional (iloc) index of the current sample row
    mode : str — "last", "avg12", or "avgall"
    prev_obs_idx : int or None — retained for API compatibility; unused by avgall
    state_offset : int — window width for avgall (median inter-observation gap)

    Returns
    -------
    X : ndarray shape (window_rows, n_cols)
    """
    if mode == "last":
        return df_pred.iloc[[row_idx]].values.astype(float)

    if mode == "avg12":
        start = max(0, row_idx - 11)
        return df_pred.iloc[start: row_idx + 1].values.astype(float)

    if mode == "avgall":
        start = max(0, row_idx - state_offset)
        return df_pred.iloc[start: row_idx + 1].values.astype(float)

    raise ValueError(f"Unknown mode: {mode!r}")


def _lagged_state_value(state_series, row_idx, mode, offset, prev_obs_idx=None):
    """Return the lagged _state scalar to append as an extra predictor.

    The lag is applied by shifting the same window used for the sensor
    predictors back by *offset* rows:

      last   : state_series.iloc[row_idx - offset]
      avg12  : nanmean(state_series.iloc[row_idx-offset-11 : row_idx-offset+1])
      avgall : nanmean(state_series.iloc[row_idx-2*offset : row_idx-offset+1])

    prev_obs_idx is retained for API compatibility but is no longer used.

    Returns np.nan when the lagged position is before the start of the series.
    """
    lagged_end = row_idx - offset  # inclusive upper bound of the lagged window
    if lagged_end < 0:
        return np.nan

    if mode == "last":
        return float(state_series.iat[lagged_end])

    if mode == "avg12":
        lagged_start = max(0, lagged_end - 11)
        window = state_series.iloc[lagged_start: lagged_end + 1].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            val = float(np.nanmean(window))
        return val

    if mode == "avgall":
        lagged_start = max(0, lagged_end - offset)
        window = state_series.iloc[lagged_start: lagged_end + 1].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            val = float(np.nanmean(window))
        return val

    raise ValueError(f"Unknown mode: {mode!r}")


def _build_samples(df, target_col, mode, obs_indices, state_offset,
                   state_series=None, predictor_cols=None):
    """Build (X, y, label) sample tuples for all observed rows of target_col.

    Each sample's predictor array X has shape (window_rows, n_sensor_cols + 1):
    the last column is a constant filled with the lagged _state scalar so that
    _extract_aligned_predictor_row() in mlr.py appends it correctly after
    aggregation.

    Parameters
    ----------
    df : DataFrame — full Consolidated_sparse DataFrame
    target_col : str — name of the target column
    mode : str — aggregation mode
    obs_indices : array-like of int — iloc positions of rows with observed values
    state_offset : int — rows to offset back for the _state predictor
    state_series : pd.Series or None — the _state column; defaults to
                   df[f"{target_col}_state"] when None
    predictor_cols : list[str] or None — sensor/weather columns to use;
                     defaults to _active_predictor_columns()

    Returns
    -------
    samples : list of (X_2d, y_1d, label)
    """
    if predictor_cols is None:
        predictor_cols = _active_predictor_columns()
    if state_series is None:
        state_col = f"{target_col}_state"
        state_series = pd.to_numeric(df[state_col], errors="coerce") if state_col in df.columns else None

    df_pred = df[predictor_cols]
    samples = []
    prev_obs_iloc = None

    for k, iloc_idx in enumerate(obs_indices):
        y_val = float(df[target_col].iat[iloc_idx])
        X_sensor = _build_window(df_pred, iloc_idx, mode, prev_obs_iloc, state_offset)

        # Append lagged _state as an extra constant column to every row of X
        if state_series is not None:
            lagged_val = _lagged_state_value(
                state_series, iloc_idx, mode, state_offset, prev_obs_iloc
            )
        else:
            lagged_val = np.nan
        extra_col = np.full((X_sensor.shape[0], 1), lagged_val, dtype=float)
        X = np.concatenate([X_sensor, extra_col], axis=1)

        samples.append((X, np.array([y_val], dtype=float), f"obs_{k}"))
        prev_obs_iloc = iloc_idx

    return samples


# ---------------------------------------------------------------------------
# Compute R² and RMSE from evaluate output
# ---------------------------------------------------------------------------

def _r2_rmse(predictions, targets):
    """Compute R² and RMSE ignoring NaN."""
    preds = np.asarray(predictions).flatten()
    targs = np.asarray(targets).flatten()
    mask = np.isfinite(preds) & np.isfinite(targs)
    if mask.sum() < 2:
        return np.nan, np.nan
    r2 = r2_score(targs[mask], preds[mask])
    rmse = float(np.sqrt(mean_squared_error(targs[mask], preds[mask])))
    return float(r2), rmse


# ---------------------------------------------------------------------------
# Inference: build predictor matrix for all blank rows
# ---------------------------------------------------------------------------

def _build_all_blank_predictor_matrix(df, target_col, mode, state_offset,
                                      state_series=None, predictor_cols=None):
    """Build aggregated predictor rows for every blank-target row in df.

    The returned matrix has shape (n_blank, len(predictor_cols) + 1); the
    last column is the lagged _state scalar for that row.

    Parameters
    ----------
    df : DataFrame — source data (may be df_out with predictions already written)
    target_col : str
    mode : str — "last", "avg12", or "avgall"
    state_offset : int — row offset for the lagged _state predictor
    state_series : pd.Series or None — the _state column to use for the lagged
                   predictor.  When None, read from df[f"{target_col}_state"].
    predictor_cols : list[str] or None — sensor/weather columns to use;
                     defaults to _active_predictor_columns()

    Returns
    -------
    blank_ilocs : list[int]
    X_blank     : ndarray [n_blank, len(predictor_cols) + 1]
    """
    if predictor_cols is None:
        predictor_cols = _active_predictor_columns()
    if state_series is None:
        state_col = f"{target_col}_state"
        state_series = (
            pd.to_numeric(df[state_col], errors="coerce")
            if state_col in df.columns else None
        )

    df_pred = df[predictor_cols]
    target_series = df[target_col]

    last_obs_iloc = None
    blank_ilocs = []
    agg_rows = []

    for iloc_idx in range(len(df)):
        val = target_series.iat[iloc_idx]
        if pd.notna(val) and np.isfinite(float(val)):
            last_obs_iloc = iloc_idx
            continue

        # Blank row — build aggregated sensor predictor vector
        X = _build_window(df_pred, iloc_idx, mode, last_obs_iloc, state_offset)
        if mode == "last":
            agg_sensor = X[-1, :]
        elif mode == "avg12":
            with np.errstate(all="ignore"):
                agg_sensor = np.nanmean(X[-12:, :], axis=0)
        else:  # avgall
            if X.shape[0] == 0:
                continue
            with np.errstate(all="ignore"):
                agg_sensor = np.nanmean(X, axis=0)

        # Append lagged _state scalar
        if state_series is not None:
            lagged_val = _lagged_state_value(
                state_series, iloc_idx, mode, state_offset, last_obs_iloc
            )
        else:
            lagged_val = np.nan

        blank_ilocs.append(iloc_idx)
        agg_rows.append(np.append(agg_sensor, lagged_val))

    n_cols = len(predictor_cols) + 1
    if not agg_rows:
        return [], np.empty((0, n_cols), dtype=float)

    return blank_ilocs, np.array(agg_rows, dtype=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _aggregate_samples(samples, mode, predictor_cols=None,
                       uncertainty_specs=None, n_replicates=0, rng=None):
    """Convert a list of (X_2d, y_1d, label) into stacked (X_agg, y) arrays.

    X_2d already has the lagged _state appended as the last column; averaging
    a constant column preserves the scalar value, so the same aggregation logic
    applies uniformly to both sensor columns and the _state column.

    When *uncertainty_specs* is provided and *n_replicates* > 0, each nominal
    aggregated row is followed by *n_replicates* perturbed copies (same y).
    The lagged _state column (last column, not in predictor_cols) is never
    perturbed — it represents the previously measured value, not a live sensor
    reading.
    """
    X_rows = []
    y_vals = []
    for X_s, y_s, _ in samples:
        if mode == "last":
            agg = X_s[-1, :]
        elif mode == "avg12":
            with np.errstate(all="ignore"):
                agg = np.nanmean(X_s[-12:, :], axis=0)
        else:  # avgall
            if X_s.shape[0] == 0:
                continue
            with np.errstate(all="ignore"):
                agg = np.nanmean(X_s, axis=0)
        X_rows.append(agg)
        y_vals.append(float(y_s[0]))

        # MC replicates — perturb only the sensor columns (not the appended
        # _state column, which is the last element of agg)
        if uncertainty_specs and n_replicates > 0 and predictor_cols is not None:
            for _ in range(n_replicates):
                rep = _perturb_predictor_row(agg, predictor_cols, uncertainty_specs, rng)
                X_rows.append(rep)
                y_vals.append(float(y_s[0]))

    if not X_rows:
        return None, None
    return np.array(X_rows, dtype=float), np.array(y_vals, dtype=float)


def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["TIMESTAMP"], low_memory=False)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Resolve active predictor columns once (respects EXCLUDE_PREDICTOR_GROUPS)
    active_cols = _active_predictor_columns()
    excluded = [c for c in PREDICTOR_COLUMNS if c not in active_cols]
    if excluded:
        print(f"  Excluded predictor groups {sorted(EXCLUDE_PREDICTOR_GROUPS)}: "
              f"{len(excluded)} columns removed, {len(active_cols)} remaining.")

    # MC perturbation setup
    uncertainty_specs: dict = {}
    rng = np.random.default_rng(seed=42)
    if USE_MC_PERTURBATION:
        uncertainty_specs = _load_surface_uncertainty_specs()
        # Only keep specs for columns that are actually active predictors
        uncertainty_specs = {k: v for k, v in uncertainty_specs.items() if k in active_cols}
        if uncertainty_specs:
            print(f"  MC perturbation enabled: {MC_REPLICATES} replicates, "
                  f"{len(uncertainty_specs)} surface column(s) perturbed: "
                  f"{list(uncertainty_specs)}")
        else:
            print("  MC perturbation requested but no matching surface columns are active.")

    summary_rows = []

    # Best model per target: {target_col: info_dict}
    best_models = {}

    for target_col in TARGET_COLUMNS:
        if target_col not in df.columns:
            print(f"[WARN] Target column '{target_col}' not found in CSV, skipping.")
            continue

        state_col = f"{target_col}_state"

        # Per-target state offset (median inter-observation gap)
        state_offset = _compute_state_offset(df, target_col)

        # Extended feature names: active sensor predictors + lagged _state
        feature_names_ext = active_cols + [f"{target_col}_state (lag={state_offset})"]

        # Find observed (non-blank, finite) rows
        obs_mask = df[target_col].notna() & df[target_col].apply(
            lambda v: np.isfinite(float(v)) if pd.notna(v) else False
        )
        obs_ilocs = np.where(obs_mask.values)[0]

        if len(obs_ilocs) < MIN_OBSERVED:
            print(
                f"[SKIP] '{target_col}': only {len(obs_ilocs)} observed values "
                f"(minimum {MIN_OBSERVED})."
            )
            continue

        n_train = int(np.floor(len(obs_ilocs) * TRAIN_FRACTION))
        train_ilocs = obs_ilocs[:n_train]

        print(
            f"\n[{target_col}] {len(obs_ilocs)} obs -> "
            f"{n_train} train / {len(obs_ilocs) - n_train} holdout  "
            f"(state_offset={state_offset})"
        )

        # Use the original _state series from the input CSV for training
        state_series_input = (
            pd.to_numeric(df[state_col], errors="coerce")
            if state_col in df.columns else None
        )

        best_models[target_col] = {}
        best_r2 = -np.inf
        best_variant_name = None

        for variant in VARIANTS:
            vname = variant["name"]
            mode = variant["mode"]

            train_samples = _build_samples(
                df, target_col, mode, train_ilocs,
                state_offset=state_offset,
                state_series=state_series_input,
                predictor_cols=active_cols,
            )

            if len(train_samples) < 2:
                print(f"  [{vname}] Not enough training samples, skipping.")
                summary_rows.append({
                    "target": target_col, "variant": vname,
                    "state_offset": state_offset,
                    "n_train": len(train_samples), "n_test": 0,
                    "r2": np.nan, "rmse": np.nan,
                    "selected_features": "", "coefficients": "",
                    "intercept": np.nan, "failure_reason": "insufficient_train_samples",
                })
                continue

            X_train, y_train = _aggregate_samples(
                train_samples, mode,
                predictor_cols=active_cols,
                uncertainty_specs=uncertainty_specs,
                n_replicates=MC_REPLICATES if uncertainty_specs else 0,
                rng=rng,
            )

            if X_train is None or len(X_train) < 2:
                print(f"  [{vname}] Aggregation produced < 2 rows, skipping.")
                summary_rows.append({
                    "target": target_col, "variant": vname,
                    "state_offset": state_offset,
                    "n_train": 0, "n_test": 0,
                    "r2": np.nan, "rmse": np.nan,
                    "selected_features": "", "coefficients": "",
                    "intercept": np.nan, "failure_reason": "insufficient_aggregated_rows",
                })
                continue

            # In-sample evaluation: pass train as both train and test
            preds, meta = fit_and_predict(
                X_train, y_train, X_train,
                feature_names=feature_names_ext,
            )
            r2, rmse = _r2_rmse(preds, y_train)

            print(
                f"  [{vname}] {meta['n_selected']} features selected, "
                f"n_train={meta['n_train']}, R2={r2:.3f}, RMSE={rmse:.4g}"
                + (f", failure={meta['failure_reason']}" if meta["failure_reason"] else "")
            )

            summary_rows.append({
                "target": target_col,
                "variant": vname,
                "state_offset": state_offset,
                "n_train": meta["n_train"],
                "n_test": meta["n_train"],
                "r2": r2,
                "rmse": rmse,
                "selected_features": ";".join(meta["selected_features"]),
                "coefficients": ";".join(str(c) for c in meta["coefficients"]),
                "intercept": meta["intercept"],
                "failure_reason": meta["failure_reason"] or "",
            })

            if np.isfinite(r2) and r2 > best_r2:
                best_r2 = r2
                best_variant_name = vname
                best_models[target_col]["best_variant"] = vname
                best_models[target_col]["best_mode"] = mode
                best_models[target_col]["state_offset"] = state_offset
                best_models[target_col]["feature_names_ext"] = feature_names_ext

        print(f"  -> Best variant: {best_variant_name} (R2={best_r2:.3f})")

    # --- Write summary artifact ---
    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nModel summary written to {SUMMARY_CSV} ({len(summary_df)} rows)")

    # --- Build Consolidated_filled.csv ---
    # df_out starts as a copy; as we fill each target's _state column the
    # updated values become available to later rows' lagged _state predictor.
    print(f"\nBuilding {OUTPUT_CSV} ...")
    df_out = df.copy()

    for target_col, info in best_models.items():
        if "best_variant" not in info:
            print(f"[SKIP fill] '{target_col}': no valid model found.")
            continue

        state_col = f"{target_col}_state"
        if state_col not in df_out.columns:
            print(f"[WARN fill] '{target_col}': _state column '{state_col}' not found.")
            continue

        mode = info["best_mode"]
        state_offset = info["state_offset"]
        feature_names_ext = info["feature_names_ext"]

        # Observed iloc positions (from input CSV — unchanged)
        all_obs_ilocs = np.where(
            df[target_col].notna() & df[target_col].apply(
                lambda v: np.isfinite(float(v)) if pd.notna(v) else False
            )
        )[0]

        # For training the final model, use the original _state series
        state_series_input = pd.to_numeric(df[state_col], errors="coerce")

        all_samples = _build_samples(
            df, target_col, mode, all_obs_ilocs,
            state_offset=state_offset,
            state_series=state_series_input,
            predictor_cols=active_cols,
        )
        X_all, y_all = _aggregate_samples(
            all_samples, mode,
            predictor_cols=active_cols,
            uncertainty_specs=uncertainty_specs,
            n_replicates=MC_REPLICATES if uncertainty_specs else 0,
            rng=rng,
        )

        if X_all is None or len(X_all) < 2:
            print(f"[SKIP fill] '{target_col}': fewer than 2 samples for full refit.")
            continue

        # Build blank-row predictor matrix using df_out (includes any predictions
        # already written to _state in earlier iterations)
        state_series_out = pd.to_numeric(df_out[state_col], errors="coerce")
        blank_ilocs, X_blank = _build_all_blank_predictor_matrix(
            df_out, target_col, mode,
            state_offset=state_offset,
            state_series=state_series_out,
            predictor_cols=active_cols,
        )

        if len(blank_ilocs) == 0:
            print(f"[{target_col}] No blank rows to fill.")
            continue

        # Refit on full observed set, predict blank rows
        preds_blank, _ = fit_and_predict(
            X_all, y_all, X_blank,
            feature_names=feature_names_ext,
        )

        # Write predictions into df_out _state column (blank-target rows only)
        n_filled = 0
        for local_k, iloc_idx in enumerate(blank_ilocs):
            pred_val = preds_blank[local_k]
            if np.isfinite(pred_val):
                df_out.at[df_out.index[iloc_idx], state_col] = pred_val
                n_filled += 1

        print(
            f"  [{target_col}] Filled {n_filled}/{len(blank_ilocs)} blank rows "
            f"using '{info['best_variant']}' (mode={mode}, offset={state_offset})"
        )

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nConsolidated_filled written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

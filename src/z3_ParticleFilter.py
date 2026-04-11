"""Particle filter state estimator for water quality targets.

For each of the 14 lab target columns, the script fits an MLR process model on
the first 70% of observed samples, then runs a sequential bootstrap particle
filter across the full timeseries.

The MLR predicts absolute target values from the high-frequency sensor
predictors at each timestep. The particle filter blends those MLR priors with
the carried-forward posterior state, injects Gaussian process noise, and
updates particle weights whenever a sparse lab measurement arrives.

By default, the two uncertainty terms are estimated separately:
  - Process noise (sigma_Q) is inferred from Monte Carlo perturbations of the
    calibrated profiler predictor columns using the same data-driven
    uncertainty distributions used elsewhere in the pipeline.
  - Measurement noise (sqrt(R)) is inferred from train-split lab-vs-model
    residuals.

Both can still be overridden via CLI.

Input
-----
  Reads data/output/regression/Consolidated_sparse.csv by default (override
  with --input).  This file must contain:
    - TIMESTAMP column
    - The raw target columns (e.g. "pH", "Turbidity (FNU)", ...)
    - The corresponding <target>_state columns (lagged state predictors)
    - All high-frequency sensor and weather predictor columns

  Note: this script does NOT read Consolidated_filled.csv.  The sparse CSV
  is used directly so that the particle filter processes the unmodified
  observation record.

Outputs
-------
  Without --run-name, all files are written to --output-dir
  (default: data/output/regression/).

  With --run-name NAME, all files are written to data/output/<NAME>/,
  keeping each run's outputs isolated.

  pf_mlr_model_log.csv
      One row per target. Columns include target, variant_chosen, mode,
      state_offset, selected_features (semicolon-separated),
      n_selected_features, coefficients (semicolon-separated), intercept,
      r2_train, rmse_train, n_train, n_obs_total, process_noise_std,
      process_noise_source, measurement_noise_std, measurement_noise_source,
      prior_blend_mode, prior_blend_auto, prior_blend, signal_std_train,
      residual_std_train, state_output_column, failure_reason.
      Written after MLR fitting and before the filter runs, so models can
      be inspected without waiting for the full filter pass.

  Consolidated_particle_filter.csv
      All columns from the input CSV plus <target>_pf_mean,
      <target>_pf_std, and <target>_state_pf for each processed target.
      The original <target>_state column is preserved by default; use
      --overwrite-state-output to also replace it with the PF state estimate.

  pf_jump_detail.csv
      One row per observation update (all targets combined).  Columns:
      target, row_idx, timestamp, z (observed value), prior_mean,
      prior_std, error (prior_mean - z), abs_error, z_score.
      Captures the filter state immediately before each measurement
      update, characterising how far off the running estimate was at
      the moment each new measurement arrived.

  pf_jump_stats.csv
      One row per target.  Summary statistics of the jump errors:
      n_updates, mean_error, std_error, mae, rmse, median_error,
      median_abs_error, p10/p90 of signed and absolute errors,
      mean_abs_z_score, frac_within_1sigma, frac_within_2sigma.

  pf_tuning_detail.csv  (only when --tune-filter is supplied)
      One row per target. Columns include target, best_sigma_q,
      best_meas_std, best_prior_blend, and the cross-validated score
      used to select the best hyperparameter combination.

  evaluation_summary_<target>.csv  (only when --cv-dir is supplied)
      Per-target evaluation summary in the 26-column f_Evaluate format,
      with two rows:
        "particle_filter (absolute)"  — metrics in raw physical units
        "particle_filter (res_norm)"  — metrics in normalised _res units,
            directly comparable to existing XGB/GP/Transformer results.
      The prior_mean at each held-out observation update is used as the
      prediction. Metrics are evaluated only on the held-out 30% of observed
      samples, not on the train split.

  evaluation_summary_all_targets.csv  (only when --cv-dir is supplied)
      All per-target rows combined into one file.

  Target_timeseries_pf.png
      One subplot per target: continuous posterior mean line with ±1σ
      shaded band, scatter of actual measured values, and horizontal legal
      limit lines (red upper, green lower).

  pf_jump_stats.png
      One row of three panels per target:
        1. Histogram of signed errors (prior_mean − z) with mean and ±1σ
        2. Signed error vs. time (scatter)
        3. |error| vs. prior σ scatter with y=x reference (calibration)

  pf_comparison_rmse.png  (only when --cv-dir is supplied)
  pf_comparison_nrmse.png
  pf_comparison_r2.png
  pf_comparison_skill_vs_best.png
      Four clustered bar charts comparing the particle filter against all
      model types (XGB, GP, Transformer, MLR, MLR12, MLRall) from the
      feature_sweep_final_metrics.csv files in each target's dataset
      directory.  One cluster per target, one bar per model type plus
      "PF" for the particle filter.  All metrics are in normalised _res
      units, directly comparable to z1_FeaturePostProcess ML comparison
      figures.  Skill is computed as 1 - RMSE / best_baseline_RMSE.

  Target_timeseries_pf_vs_model.png  (only when --predictions-csv is supplied)
      One subplot per target overlaying:
        - Existing model reconstruction (dashed grey, from _reconstructed column)
        - Particle filter posterior mean (solid coloured line)
        - Particle filter ±1σ shaded band
        - Scatter of actual measured values
        - Legal limit lines

CLI arguments
-------------
  --input PATH
      Path to the input CSV.
      Default: data/output/regression/Consolidated_sparse.csv

  --output-dir PATH
      Directory where all output files are written.
      Default: data/output/regression/

  --run-name NAME
      Write all outputs into data/output/<NAME>/ instead of --output-dir.
      Keeps each run's outputs isolated without filename clutter.
      E.g. --run-name test01 → data/output/test01/pf_mlr_model_log.csv etc.
      Default: outputs go directly into --output-dir.

  --targets TARGET [TARGET ...]
      One or more target column names to process, separated by spaces.
      Names containing spaces or special characters must be quoted.
      Must match exact column names in the input CSV.
      Targets are processed in the canonical order defined in
      fill_gaps_mlr.TARGET_COLUMNS; unrecognised names generate a warning.
      Default: all 14 standard targets.

  --n-particles N
      Number of particles for the bootstrap filter.
      Default: 500

  --process-model {mlr,best_ml_gap_residual,best_mlr_gap_residual}
      Process-model family used inside the particle filter.
        mlr                    — fits a fresh MLR on the training split
                                 (default; no external inputs required).
        best_ml_gap_residual   — uses the best ML model's gap-residual
                                 predictions as the process signal.
        best_mlr_gap_residual  — uses the best MLR model's gap-residual
                                 predictions as the process signal.
      The two non-default choices require --cv-dir to locate the fitted
      model outputs.
      Default: mlr

  --process-noise SIGMA_Q
      Override the process noise standard deviation σ_Q for all targets.
      When omitted, σ_Q is inferred per target from Monte Carlo perturbations
      of the calibrated profiler predictor columns (sensor_uncertainty_mc).
      Falls back to std(training residuals) if no uncertainty specs are
      available for the target's active predictors.

  --measurement-noise SIGMA_R
      Override the measurement noise standard deviation √R for all targets.
      When omitted, √R is inferred per target as std(train-split
      lab-vs-model residuals).  This differs from the σ_Q default, which
      uses Monte Carlo sensor-uncertainty propagation rather than residuals.

  --prior-blend W
      Weight in [0, 1] that pulls each particle toward the MLR prior
      prediction at every timestep.  0 keeps a pure carried-forward state
      (no prior injection); 1 fully resets to the MLR prior.
      When omitted, a conservative value is inferred automatically from
      the train-only MLR skill and residual noise.
      Default: auto-inferred per target.

  --seed INT
      Integer random seed for the particle RNG.
      Default: 42

  --overwrite-state-output
      Also overwrite each <target>_state column in the output CSV with
      the PF state estimate.  By default the original _state columns are
      preserved and the PF result is written to <target>_state_pf.
      Default: off.

  --cv-dir PATH
      Path to a CV output directory (e.g. data/output/CV14).  Enables
      two additional outputs: per-target evaluation_summary.csv files in
      the 26-column f_Evaluate format (for direct comparison with existing
      model results), and a combined evaluation_summary_all_targets.csv.
      Metrics are computed in both absolute and normalised _res units.
      The normalization.json in each target's dataset subdirectory is used
      for the _res normalisation.  If omitted, no evaluation_summary is
      written.

  --predictions-csv PATH
      Path to a Consolidated_predictions.csv produced by
      z2_PredictionTimeseries.py, containing <target>_reconstructed columns.
      When supplied, a second figure is generated:
        Target_timeseries_pf_vs_model.png
      Each subplot overlays the particle filter mean (solid coloured line)
      and ±1σ band on top of the existing model's reconstructed timeseries
      (dashed grey line), with measured values scattered on top.
      This provides a direct visual comparison of the two approaches.
      Default: no comparison figure.

  --no-figure
      Skip figure generation.  The CSV outputs are still written.

  --tune-filter
      Enable per-target hyperparameter tuning of σ_Q, measurement noise,
      and prior_blend.  A coarse grid search is run on sequential
      train-only folds (see --tune-folds); the best combination is then
      used for the full filter pass.  When active, also writes
      pf_tuning_detail.csv (one row per target with the chosen values).
      Default: off.

  --tune-folds N
      Number of sequential train-only validation folds used by
      --tune-filter.  More folds give a more stable estimate but
      increase runtime linearly.
      Default: 3

  --tune-grid-scale N
      Controls the width of the coarse noise grid searched by
      --tune-filter.
        1  — compact grid: multipliers [0.25, 0.5, 1.0, 2.0, 4.0]
             around the inferred default σ_Q and σ_R  (25 pairs).
        >1 — wider grid: multipliers [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
             (49 pairs; useful when the default σ_Q/σ_R are far from
             optimal and the compact grid misses the minimum).
      Only the noise candidates are affected; prior_blend candidates
      are not scaled.
      Default: 1

Usage examples
--------------
  # Run all targets with defaults
  python src/z3_ParticleFilter.py

  # Run only two targets, isolated in their own output directory
  python src/z3_ParticleFilter.py --targets "pH" "Turbidity (FNU)" --run-name ph_turbidity

  # Increase particle count; outputs go to data/output/n2000/
  python src/z3_ParticleFilter.py --n-particles 2000 --run-name n2000

  # Manually tune noise parameters (useful when inferred values produce
  # over- or under-smoothed estimates)
  python src/z3_ParticleFilter.py --process-noise 1.5 --measurement-noise 0.5

  # Quick smoke test: few particles, no figure, custom output directory
  python src/z3_ParticleFilter.py --n-particles 50 --no-figure --output-dir data/output/pf_test

  # Full run with CV comparison — produces evaluation_summary files
  # directly comparable to XGB/GP/Transformer results in CV14
  python src/z3_ParticleFilter.py --run-name pf_vs_cv14 --cv-dir data/output/CV14

  # Specify a non-default input file
  python src/z3_ParticleFilter.py --input data/output/regression/Consolidated_filled.csv

Modularity notes
----------------
ProcessModel and ObservationModel are typing.Protocol classes.  Adding a new
process model (e.g. ResidualProcessModel for XGB/GP/Transformer) only requires
implementing those two methods; run_particle_filter() is unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from fill_gaps_mlr import (
    TARGET_COLUMNS,
    TRAIN_FRACTION,
    MIN_OBSERVED,
    VARIANTS,
    _active_predictor_columns,
    _compute_state_offset,
    _build_samples,
    _aggregate_samples,
    _lagged_state_value,
    _build_window,
    _r2_rmse,
    _load_surface_uncertainty_specs,
    _perturb_predictor_row,
)
from utils.mlr import fit_and_predict
from utils.limits import load_limits_records, map_limits_to_columns
from utils.config_utils import select_best_model_row
from utils.evaluation import visualizer, boxplot_from_error_rows
from plot_mlr_fill_timeseries import (
    _strip_common_affixes,
    _safe_series_colors,
    _limit_bounds,
)
from z2_PredictionTimeseries import (
    _load_model_bundle,
    _predict_all_rows,
    _denormalize_value,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUT      = str(_REPO_ROOT / "data" / "output" / "regression" / "Consolidated_sparse.csv")
_DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "data" / "output" / "regression")
_ML_PROCESS_MODELS = {"xgb_regressor", "gp_regressor", "transformer"}
_SAVED_MLR_PROCESS_MODELS = {"mlr", "mlr_avg12", "mlr_avgall"}


# ---------------------------------------------------------------------------
# Protocols (abstract interfaces for process and observation models)
# ---------------------------------------------------------------------------

@runtime_checkable
class ProcessModel(Protocol):
    """Protocol for particle filter process models.

    A process model predicts the deterministic prior mean at a given timestep.
    The filter adds Gaussian noise (sigma_Q) around this mean to form the
    particle cloud for that step.
    """

    def predict(
        self,
        df: pd.DataFrame,
        row_idx: int,
        state_history: np.ndarray | None = None,
    ) -> float:
        """Return the deterministic mean state prediction for row_idx.

        Returns np.nan when predictor data is unavailable; the filter then
        falls back to a random-walk propagation.
        """
        ...

    @property
    def noise_std(self) -> float:
        """Process noise standard deviation sigma_Q."""
        ...


@runtime_checkable
class ObservationModel(Protocol):
    """Protocol for particle filter observation (likelihood) models."""

    def log_likelihood(self, z: float, x_particle: float) -> float:
        """Return log p(z | x_particle)."""
        ...


# ---------------------------------------------------------------------------
# MLR process model
# ---------------------------------------------------------------------------

class MLRProcessModel:
    """Process model using a pre-fitted MLR (absolute value prediction).

    At each timestep the model returns the OLS prediction from the current
    high-frequency sensor predictors.  Each particle is then drawn as:

        x_particle[t] = mu_MLR(t) + N(0, sigma_Q^2)

    When mu_MLR(t) is NaN (missing sensor data), the filter falls back to
    additive random-walk propagation.

    Parameters
    ----------
    df : full input DataFrame (must contain all predictor columns).
    target_col : name of the target variable.
    fit_result : dict returned by fit_mlr_for_target().
    noise_std_override : if given, overrides the inferred sigma_Q.
    precomputed_preds : optional ndarray [n_rows] of pre-computed MLR
        predictions at every row.  When provided, predict() is an O(1)
        array lookup; otherwise it builds the predictor window on the fly.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        fit_result: dict,
        noise_std_override: float | None = None,
        precomputed_preds: np.ndarray | None = None,
    ):
        self._precomputed = precomputed_preds

        meta = fit_result["meta"]
        self._selected_features: list[str] = list(meta["selected_features"])
        self._coef = np.array(meta["coefficients"], dtype=float)
        self._intercept = float(meta["intercept"])

        self._mode = fit_result["mode"]
        self._state_offset = fit_result["state_offset"]
        self._feature_names_ext: list[str] = list(fit_result["feature_names_ext"])
        self._active_cols: list[str] = _active_predictor_columns()
        self._state_feature_name = f"{target_col}_state (lag={self._state_offset})"
        self._uses_state_feature = self._state_feature_name in self._selected_features

        # Raw sparse _state series from the input CSV. This seeds the lagged-state
        # predictor until the filter has generated enough posterior history to use
        # its own state estimate instead.
        state_col = f"{target_col}_state"
        self._state_series = (
            pd.to_numeric(df[state_col], errors="coerce")
            if state_col in df.columns else None
        )

        # Map selected feature names to indices in the extended feature list
        self._sel_indices = [
            self._feature_names_ext.index(fn)
            for fn in self._selected_features
            if fn in self._feature_names_ext
        ]

        self._noise_std_val = (
            float(noise_std_override)
            if noise_std_override is not None
            else float(fit_result["process_noise_std"])
        )

        # Keep a reference to the predictor columns of df for the row-by-row path
        self._df_pred = df[self._active_cols]

    def _resolve_lagged_state_value(
        self,
        row_idx: int,
        state_history: np.ndarray | None,
    ) -> float:
        """Return lagged-state predictor value, preferring PF history when available."""
        raw_val = (
            _lagged_state_value(
                self._state_series, row_idx, self._mode, self._state_offset, None
            )
            if self._state_series is not None else np.nan
        )
        if state_history is None:
            return raw_val

        hist = np.asarray(state_history, dtype=float)
        if hist.ndim != 1 or row_idx <= 0:
            return raw_val

        usable = hist[:row_idx].copy()
        if self._state_series is not None:
            raw_prefix = self._state_series.iloc[:row_idx].to_numpy(dtype=float, copy=True)
            if raw_prefix.shape == usable.shape:
                replace_mask = ~np.isfinite(usable) & np.isfinite(raw_prefix)
                usable[replace_mask] = raw_prefix[replace_mask]

        if not np.isfinite(usable).any():
            return raw_val

        hist_series = pd.Series(usable)
        hist_val = _lagged_state_value(
            hist_series, row_idx, self._mode, self._state_offset, None
        )
        return float(hist_val) if np.isfinite(hist_val) else raw_val

    def predict(
        self,
        df: pd.DataFrame,
        row_idx: int,
        state_history: np.ndarray | None = None,
    ) -> float:  # noqa: ARG002
        """Return the MLR prediction at row_idx (O(1) if precomputed)."""
        if self._precomputed is not None and not self._uses_state_feature:
            val = self._precomputed[row_idx]
            return float(val) if np.isfinite(val) else np.nan

        # --- Row-by-row fallback ---
        X_win = _build_window(
            self._df_pred, row_idx, self._mode, None, self._state_offset
        )
        if self._mode == "last":
            agg_sensor = X_win[-1, :]
        elif self._mode == "avg12":
            with np.errstate(all="ignore"):
                agg_sensor = np.nanmean(X_win[-12:, :], axis=0)
        else:  # avgall
            with np.errstate(all="ignore"):
                agg_sensor = np.nanmean(X_win, axis=0)

        lagged_val = self._resolve_lagged_state_value(row_idx, state_history)
        row_full = np.append(agg_sensor, lagged_val)

        if not self._sel_indices:
            return float(self._intercept)

        x_sel = row_full[self._sel_indices]
        if not np.all(np.isfinite(x_sel)):
            return np.nan
        return float(self._intercept + np.dot(self._coef, x_sel))

    @property
    def noise_std(self) -> float:
        return self._noise_std_val


class MLGapResidualProcessModel:
    """Process model using saved ML residual predictions over the long lookback L.

    The loaded ML model predicts a residual delta over the target-specific
    lookback interval ``L = state_offset`` in normalized ``_res`` units.
    At row ``i`` this process model converts that into an absolute target-state
    goal:

        x_ml_target(i) = filtered_state[i-L] + delta_raw(i)

    where ``filtered_state[i-L]`` comes from PF posterior history when
    available, falling back to the raw observed/state series only to bootstrap
    early rows before enough PF history exists.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        output_col: str,
        state_offset: int,
        norm_bounds: dict,
        pred_res_norm: np.ndarray,
        noise_std: float,
        model_type: str,
        dataset_dir: Path,
        forecast_namespace: str,
    ):
        self._target_col = str(target_col)
        self._output_col = str(output_col)
        self._state_offset = int(state_offset)
        self._norm_bounds = dict(norm_bounds or {})
        self._pred_res_norm = np.asarray(pred_res_norm, dtype=float)
        self._noise_std_val = max(float(noise_std), 1e-6)
        self._model_type = str(model_type)
        self._dataset_dir = Path(dataset_dir)
        self._forecast_namespace = str(forecast_namespace)
        self._observed = (
            pd.to_numeric(df[target_col], errors="coerce")
            if target_col in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
        )
        state_col = f"{target_col}_state"
        self._state_series = (
            pd.to_numeric(df[state_col], errors="coerce")
            if state_col in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
        )

    def _resolve_base_state(self, lookback_idx: int, state_history: np.ndarray | None) -> float:
        """Return the filtered lookback state, with raw-state bootstrap fallback."""
        if lookback_idx < 0:
            return np.nan

        if state_history is not None:
            hist = np.asarray(state_history, dtype=float)
            if hist.ndim == 1 and 0 <= lookback_idx < len(hist):
                hist_val = hist[lookback_idx]
                if np.isfinite(hist_val):
                    return float(hist_val)

        if 0 <= lookback_idx < len(self._observed):
            obs_val = self._observed.iat[lookback_idx]
            if np.isfinite(obs_val):
                return float(obs_val)

        if 0 <= lookback_idx < len(self._state_series):
            state_val = self._state_series.iat[lookback_idx]
            if np.isfinite(state_val):
                return float(state_val)

        return np.nan

    def predict(
        self,
        df: pd.DataFrame,
        row_idx: int,
        state_history: np.ndarray | None = None,
    ) -> float:  # noqa: ARG002
        if row_idx < 0 or row_idx >= len(self._pred_res_norm):
            return np.nan

        pred_norm = self._pred_res_norm[row_idx]
        if not np.isfinite(pred_norm):
            return np.nan

        delta_raw = _denormalize_value(float(pred_norm), self._output_col, self._norm_bounds)
        if not np.isfinite(delta_raw):
            return np.nan

        lookback_idx = row_idx - self._state_offset
        base_state = self._resolve_base_state(lookback_idx, state_history)
        if not np.isfinite(base_state):
            return np.nan
        return float(base_state + delta_raw)

    @property
    def noise_std(self) -> float:
        return self._noise_std_val


class MLRGapResidualProcessModel(MLGapResidualProcessModel):
    """Process model using saved MLR residual predictions over the long lookback L."""


class NoiseOverrideProcessModel:
    """Wrap a process model while overriding only its process-noise std."""

    def __init__(self, base_model: ProcessModel, noise_std: float):
        self._base_model = base_model
        self._noise_std_val = max(float(noise_std), 1e-6)

    def predict(
        self,
        df: pd.DataFrame,
        row_idx: int,
        state_history: np.ndarray | None = None,
    ) -> float:
        return self._base_model.predict(df, row_idx, state_history=state_history)

    @property
    def noise_std(self) -> float:
        return self._noise_std_val


# ---------------------------------------------------------------------------
# Gaussian observation model
# ---------------------------------------------------------------------------

class GaussianObservationModel:
    """Observation model: log N(z; x_particle, R).

    R is the measurement noise variance, inferred from training residuals
    (variance of MLR in-sample errors on the 70% training split).

    Parameters
    ----------
    measurement_noise_var : R > 0.  Clamped to 1e-12 if non-positive.
    """

    def __init__(self, measurement_noise_var: float):
        if not np.isfinite(measurement_noise_var) or measurement_noise_var <= 0:
            measurement_noise_var = 1.0
        self._R = float(measurement_noise_var)
        self._log_norm = np.log(2.0 * np.pi * self._R)

    def log_likelihood(self, z: float, x_particle: float) -> float:
        diff = z - x_particle
        return -0.5 * (diff * diff / self._R + self._log_norm)

    @property
    def R(self) -> float:
        return self._R


# ---------------------------------------------------------------------------
# Resampling utilities
# ---------------------------------------------------------------------------

def _systematic_resample(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Systematic (low-variance) resampling — O(N).

    Parameters
    ----------
    particles : shape (N,)
    weights   : shape (N,), normalised (sum == 1)

    Returns
    -------
    resampled : shape (N,)
    """
    N = len(particles)
    cumsum = np.cumsum(weights)
    u0 = rng.uniform(0.0, 1.0 / N)
    positions = u0 + np.arange(N) / N
    indices = np.clip(np.searchsorted(cumsum, positions), 0, N - 1)
    return particles[indices]


def _effective_sample_size(weights: np.ndarray) -> float:
    """ESS = 1 / sum(w_i^2), with weights assumed normalised."""
    return 1.0 / float(np.sum(weights ** 2))


def _suggest_prior_blend(
    r2_train: float,
    measurement_noise_var: float,
    y_train: np.ndarray,
) -> float:
    """Return a conservative train-only blend weight for the MLR prior.

    The weight is intentionally capped well below 0.5 so that the carried
    posterior remains dominant unless the MLR is both strong and low-noise.
    Holdout/evaluation metrics are intentionally excluded here to avoid
    using future information inside the online filter.
    """
    finite_y = np.asarray(y_train, dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size >= 2:
        signal_var = float(np.var(finite_y, ddof=1))
    else:
        signal_var = np.nan

    if not np.isfinite(signal_var) or signal_var <= 0:
        return 0.05

    skill = float(np.clip(r2_train, 0.0, 1.0)) if np.isfinite(r2_train) else 0.0
    noise_var = float(measurement_noise_var) if np.isfinite(measurement_noise_var) else signal_var
    reliability = signal_var / (signal_var + max(noise_var, 1e-12))

    # Range: [0.05, 0.35]. A poor/noisy MLR gets a minimal nudge; a strong
    # low-noise MLR can pull harder, but still does not dominate the posterior.
    blend = 0.05 + 0.30 * skill * reliability
    return float(np.clip(blend, 0.05, 0.35))


def _estimate_process_noise_from_sensor_uncertainty(
    X_train: np.ndarray,
    meta: dict,
    feature_names_ext: list[str],
    active_cols: list[str],
    rng: np.random.Generator,
    uncertainty_specs: dict | None = None,
    n_mc_replicates: int = 20,
) -> tuple[float, dict]:
    """Estimate sigma_Q from calibrated sensor-uncertainty perturbations.

    The training predictor rows are perturbed using the profiler-column
    calibration distributions. The resulting spread in the fitted MLR
    predictions is used as a data-driven estimate of process-model uncertainty.
    """
    if uncertainty_specs is None:
        uncertainty_specs = _load_surface_uncertainty_specs()
    uncertainty_specs = {
        str(k): v for k, v in dict(uncertainty_specs or {}).items()
        if str(k) in active_cols
    }
    if not uncertainty_specs or X_train is None or len(X_train) < 1:
        return np.nan, {
            "process_noise_source": "sensor_uncertainty_unavailable",
            "process_noise_mc_samples": 0,
        }

    coef = np.asarray(meta.get("coefficients", []), dtype=float)
    intercept = float(meta.get("intercept", np.nan))
    selected_features = list(meta.get("selected_features", []))
    sel_idx = [feature_names_ext.index(fn) for fn in selected_features if fn in feature_names_ext]
    if not sel_idx or coef.size != len(sel_idx) or not np.isfinite(intercept):
        return np.nan, {
            "process_noise_source": "sensor_uncertainty_unavailable",
            "process_noise_mc_samples": 0,
        }

    delta_samples: list[float] = []
    sensor_width = len(active_cols)
    for row in np.asarray(X_train, dtype=float):
        if row.shape[0] < sensor_width:
            continue
        sensor_row = row[:sensor_width]
        if not np.isfinite(sensor_row).any():
            continue
        nominal_full = row.copy()
        nominal_selected = nominal_full[sel_idx]
        if not np.all(np.isfinite(nominal_selected)):
            continue
        nominal_pred = float(intercept + np.dot(coef, nominal_selected))
        for _ in range(int(max(0, n_mc_replicates))):
            perturbed_sensor = _perturb_predictor_row(
                sensor_row,
                active_cols,
                uncertainty_specs,
                rng,
            )
            perturbed_full = nominal_full.copy()
            perturbed_full[:sensor_width] = perturbed_sensor
            perturbed_selected = perturbed_full[sel_idx]
            if not np.all(np.isfinite(perturbed_selected)):
                continue
            perturbed_pred = float(intercept + np.dot(coef, perturbed_selected))
            delta = perturbed_pred - nominal_pred
            if np.isfinite(delta):
                delta_samples.append(delta)

    if len(delta_samples) < 2:
        return np.nan, {
            "process_noise_source": "sensor_uncertainty_unavailable",
            "process_noise_mc_samples": len(delta_samples),
        }

    sigma_q = float(np.std(np.asarray(delta_samples, dtype=float), ddof=1))
    return max(sigma_q, 1e-6), {
        "process_noise_source": "sensor_uncertainty_mc",
        "process_noise_mc_samples": len(delta_samples),
    }


def _observed_split_ilocs(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return observed, train, and holdout ilocs for target_col using the 70/30 split."""
    target_series = pd.to_numeric(df[target_col], errors="coerce")
    obs_ilocs = np.where(np.isfinite(target_series.values))[0]
    if len(obs_ilocs) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty
    n_train = int(np.floor(len(obs_ilocs) * TRAIN_FRACTION))
    train_ilocs = obs_ilocs[:n_train].astype(int)
    holdout_ilocs = obs_ilocs[n_train:].astype(int)
    return obs_ilocs.astype(int), train_ilocs, holdout_ilocs


def _resolve_ml_forecast_namespace(best_row: pd.Series) -> str:
    """Infer forecast namespace from the selected metrics row."""
    subset_label = str(best_row.get("subset_label", "")).strip().lower()
    return "Shapley_sweeps" if subset_label.startswith("shap_") else "feature_sweeps"


# ---------------------------------------------------------------------------
# MLR fitting helper
# ---------------------------------------------------------------------------

def fit_mlr_for_target(
    df: pd.DataFrame,
    target_col: str,
    active_cols: list[str],
    rng: np.random.Generator,
) -> dict | None:
    """Fit the best MLR variant for target_col and return a result dict.

    Replicates the per-target fitting logic from fill_gaps_mlr.main() and
    plot_mlr_fill_timeseries._build_mlr_estimates(), but additionally computes
    training residuals to derive noise parameters.

    Parameters
    ----------
    df : full input DataFrame.
    target_col : name of the target column.
    active_cols : list of active predictor column names.
    rng : random generator (reserved; not used by the pure MLR path).

    Returns
    -------
    dict with keys:
        mode, variant_name, state_offset, feature_names_ext,
        X_train, y_train, meta,
        r2, rmse, n_train, n_obs_total, train_ilocs, holdout_ilocs,
        process_noise_std, process_noise_source, process_noise_mc_samples,
        measurement_noise_var, measurement_noise_source,
        residual_std_train, signal_std_train, prior_blend_auto
    None if the target has fewer than MIN_OBSERVED valid observations.
    """
    state_col = f"{target_col}_state"
    state_offset = _compute_state_offset(df, target_col)
    feature_names_ext = active_cols + [f"{target_col}_state (lag={state_offset})"]

    # Observed (non-blank, finite) row positions
    obs_mask = df[target_col].notna() & df[target_col].apply(
        lambda v: np.isfinite(float(v)) if pd.notna(v) else False
    )
    obs_ilocs = np.where(obs_mask.values)[0]

    if len(obs_ilocs) < MIN_OBSERVED:
        return None

    n_train = int(np.floor(len(obs_ilocs) * TRAIN_FRACTION))
    train_ilocs = obs_ilocs[:n_train]

    state_series = (
        pd.to_numeric(df[state_col], errors="coerce")
        if state_col in df.columns else None
    )

    # ------------------------------------------------------------------
    # Select best variant by in-sample R²
    # ------------------------------------------------------------------
    best_r2 = -np.inf
    best_mode = None
    best_variant_name = None

    for variant in VARIANTS:
        vname = variant["name"]
        mode = variant["mode"]

        train_samples = _build_samples(
            df, target_col, mode, train_ilocs,
            state_offset=state_offset,
            state_series=state_series,
            predictor_cols=active_cols,
        )
        if len(train_samples) < 2:
            continue

        X_tr, y_tr = _aggregate_samples(train_samples, mode, predictor_cols=active_cols)
        if X_tr is None or len(X_tr) < 2:
            continue

        preds, _ = fit_and_predict(X_tr, y_tr, X_tr, feature_names=feature_names_ext)
        r2, _ = _r2_rmse(preds, y_tr)

        if np.isfinite(r2) and r2 > best_r2:
            best_r2 = r2
            best_mode = mode
            best_variant_name = vname

    if best_mode is None:
        return None

    # ------------------------------------------------------------------
    # Fit final model on the 70% training split only
    # ------------------------------------------------------------------
    train_samples_final = _build_samples(
        df, target_col, best_mode, train_ilocs,
        state_offset=state_offset,
        state_series=state_series,
        predictor_cols=active_cols,
    )
    X_train, y_train = _aggregate_samples(
        train_samples_final, best_mode, predictor_cols=active_cols
    )
    if X_train is None or len(X_train) < 2:
        return None

    # Final model (trained on training split only): get coefficients via fit_and_predict
    _, meta = fit_and_predict(X_train, y_train, X_train, feature_names=feature_names_ext)

    # ------------------------------------------------------------------
    # In-sample predictions on the 70% training split (for residual noise
    # inference). Use the final train-only model to predict on the same
    # training subset; this is still in-sample, but it avoids leaking holdout
    # observations into either Q or R.
    # ------------------------------------------------------------------
    train_samples_eval = _build_samples(
        df, target_col, best_mode, train_ilocs,
        state_offset=state_offset,
        state_series=state_series,
        predictor_cols=active_cols,
    )
    X_tr_eval, y_tr_eval = _aggregate_samples(
        train_samples_eval, best_mode, predictor_cols=active_cols
    )

    if X_tr_eval is not None and len(X_tr_eval) >= 2:
        train_preds, _ = fit_and_predict(
            X_train, y_train, X_tr_eval, feature_names=feature_names_ext
        )
        residuals = train_preds - y_tr_eval
        finite_mask = np.isfinite(residuals)
        if finite_mask.sum() >= 2:
            meas_noise_var = float(np.var(residuals[finite_mask], ddof=1))
            residual_std_train = float(np.std(residuals[finite_mask], ddof=1))
            measurement_noise_source = "train_residual"
        else:
            meas_noise_var = 1.0
            residual_std_train = 1.0
            measurement_noise_source = "fallback_default"
    else:
        meas_noise_var = 1.0
        residual_std_train = 1.0
        measurement_noise_source = "fallback_default"

    proc_noise_std, proc_meta = _estimate_process_noise_from_sensor_uncertainty(
        X_train=X_train,
        meta=meta,
        feature_names_ext=feature_names_ext,
        active_cols=active_cols,
        rng=rng,
    )
    if not np.isfinite(proc_noise_std) or proc_noise_std <= 0:
        proc_noise_std = max(float(residual_std_train), 1e-6)
        process_noise_source = "train_residual_fallback"
        process_noise_mc_samples = int(proc_meta.get("process_noise_mc_samples", 0))
    else:
        process_noise_source = str(proc_meta.get("process_noise_source", "sensor_uncertainty_mc"))
        process_noise_mc_samples = int(proc_meta.get("process_noise_mc_samples", 0))

    # Guard against non-positive variance (can happen when Lasso collapses
    # the model to an intercept and residuals are all identical)
    meas_noise_var = max(meas_noise_var, 1e-12)
    proc_noise_std = max(proc_noise_std, 1e-6)

    r2, rmse = _r2_rmse(
        fit_and_predict(X_train, y_train, X_train, feature_names=feature_names_ext)[0],
        y_train,
    )
    prior_blend_auto = _suggest_prior_blend(r2, meas_noise_var, y_train)
    signal_std_train = float(np.nanstd(y_train, ddof=1)) if len(y_train) >= 2 else np.nan

    return {
        "mode":                best_mode,
        "variant_name":        best_variant_name,
        "state_offset":        state_offset,
        "feature_names_ext":   feature_names_ext,
        "X_train":             X_train,
        "y_train":             y_train,
        "meta":                meta,
        "r2":                  float(r2),
        "rmse":                float(rmse),
        "n_train":             int(n_train),
        "n_obs_total":         int(len(obs_ilocs)),
        "train_ilocs":         train_ilocs.astype(int).tolist(),
        "holdout_ilocs":       obs_ilocs[n_train:].astype(int).tolist(),
        "process_noise_std":   proc_noise_std,
        "process_noise_source": process_noise_source,
        "process_noise_mc_samples": process_noise_mc_samples,
        "measurement_noise_var": meas_noise_var,
        "measurement_noise_source": measurement_noise_source,
        "residual_std_train":  residual_std_train,
        "signal_std_train":    signal_std_train,
        "prior_blend_auto":    prior_blend_auto,
    }


# ---------------------------------------------------------------------------
# Vectorised MLR inference over all rows
# ---------------------------------------------------------------------------

def _precompute_mlr_predictions(
    df: pd.DataFrame,
    target_col: str,
    fit_result: dict,
    active_cols: list[str],
) -> np.ndarray:
    """Build the MLR deterministic prediction at every row of df.

    Mirrors the all-rows loop in plot_mlr_fill_timeseries._build_mlr_estimates
    but uses the pre-fitted coefficients from fit_result rather than
    calling fit_and_predict() again.

    Returns
    -------
    ndarray [n_rows] with NaN where any selected predictor is NaN.
    """
    mode           = fit_result["mode"]
    state_offset   = fit_result["state_offset"]
    feature_names_ext = fit_result["feature_names_ext"]
    meta           = fit_result["meta"]
    coef           = np.array(meta["coefficients"], dtype=float)
    intercept      = float(meta["intercept"])
    selected_features: list[str] = list(meta["selected_features"])

    state_col = f"{target_col}_state"
    state_series = (
        pd.to_numeric(df[state_col], errors="coerce")
        if state_col in df.columns else None
    )

    # Map selected feature names to column indices in the extended feature list
    sel_idx = []
    for fn in selected_features:
        try:
            sel_idx.append(feature_names_ext.index(fn))
        except ValueError:
            pass  # feature not in extended list — skip

    obs_mask = df[target_col].notna() & df[target_col].apply(
        lambda v: np.isfinite(float(v)) if pd.notna(v) else False
    )

    df_pred = df[active_cols]
    n_rows = len(df)
    last_obs_iloc = None
    agg_rows = []

    for iloc_idx in range(n_rows):
        if mode == "last":
            agg_sensor = df_pred.iloc[iloc_idx].values.astype(float)
        elif mode == "avg12":
            start = max(0, iloc_idx - 11)
            X_win = df_pred.iloc[start: iloc_idx + 1].values.astype(float)
            with np.errstate(all="ignore"):
                agg_sensor = np.nanmean(X_win, axis=0)
        else:  # avgall
            start = max(0, iloc_idx - state_offset)
            X_win = df_pred.iloc[start: iloc_idx + 1].values.astype(float)
            with np.errstate(all="ignore"):
                agg_sensor = np.nanmean(X_win, axis=0)

        lagged_val = (
            _lagged_state_value(
                state_series, iloc_idx, mode, state_offset, last_obs_iloc
            )
            if state_series is not None else np.nan
        )
        row_full = np.append(agg_sensor, lagged_val)
        agg_rows.append(row_full)

        if obs_mask.iat[iloc_idx]:
            last_obs_iloc = iloc_idx

    X_all_rows = np.array(agg_rows, dtype=float)   # (n_rows, n_ext_features)

    preds = np.full(n_rows, np.nan, dtype=float)
    if not sel_idx:
        preds[:] = intercept
        return preds

    X_sel = X_all_rows[:, sel_idx]                  # (n_rows, n_selected)
    finite_mask = np.all(np.isfinite(X_sel), axis=1)
    preds[finite_mask] = intercept + X_sel[finite_mask] @ coef
    return preds


def build_ml_gap_process_model_for_target(
    df: pd.DataFrame,
    target_col: str,
    cv_dir: Path,
    process_noise_override: float | None = None,
) -> dict | None:
    """Build an ML long-gap residual process model from saved CV artifacts."""
    dataset_dir = _dataset_dir_for_target(cv_dir, target_col)
    if dataset_dir is None:
        print(f"  [WARN] No dataset directory found in {cv_dir} for '{target_col}'.")
        return None

    fm_csv = dataset_dir / "forecasts" / "feature_sweeps" / "feature_sweep_final_metrics.csv"
    if not fm_csv.exists():
        print(f"  [WARN] feature_sweep_final_metrics.csv not found: {fm_csv}")
        return None

    try:
        df_fm = pd.read_csv(fm_csv)
    except Exception as exc:
        print(f"  [WARN] Could not read {fm_csv}: {exc}")
        return None
    if df_fm.empty:
        print(f"  [WARN] Empty feature_sweep_final_metrics.csv: {fm_csv}")
        return None

    valid_df = _filter_sweep_rows(df_fm)
    ml_df = valid_df[
        (~valid_df["model"].apply(_is_baseline_model))
        & (valid_df["model"].astype(str).str.lower().isin(_ML_PROCESS_MODELS))
    ].copy()
    if ml_df.empty:
        print(f"  [WARN] No valid non-MLR ML rows found for '{target_col}' in {fm_csv}.")
        return None

    best_row = select_best_model_row(ml_df)
    if best_row is None or getattr(best_row, "empty", False):
        print(f"  [WARN] Could not select a best ML model row for '{target_col}'.")
        return None

    forecast_namespace = _resolve_ml_forecast_namespace(best_row)
    bundle = _load_model_bundle(dataset_dir, "feature_sweeps", best_row)
    if bundle is None:
        print(f"  [WARN] Could not load best ML model bundle for '{target_col}'.")
        return None

    if str(bundle.model_type).strip().lower() not in _ML_PROCESS_MODELS:
        print(
            f"  [WARN] Best loaded bundle for '{target_col}' is '{bundle.model_type}', "
            "not an eligible non-MLR ML model."
        )
        return None

    pred_res_norm = _predict_all_rows(bundle, df)
    state_offset = _compute_state_offset(df, target_col) if target_col in df.columns else 1
    rmse_norm = float(pd.to_numeric(best_row.get("rmse", np.nan), errors="coerce"))
    output_col = str(bundle.output_column)
    rmse_raw = _denormalize_value(rmse_norm, output_col, bundle.norm_bounds) if np.isfinite(rmse_norm) else np.nan
    rmse_raw = abs(float(rmse_raw)) if np.isfinite(rmse_raw) else np.nan
    process_noise_std = (
        float(process_noise_override)
        if process_noise_override is not None
        else (rmse_raw if np.isfinite(rmse_raw) and rmse_raw > 0 else 1.0)
    )
    measurement_noise_var = (
        rmse_raw ** 2 if np.isfinite(rmse_raw) and rmse_raw > 0 else 1.0
    )
    obs_ilocs, train_ilocs, holdout_ilocs = _observed_split_ilocs(df, target_col)
    prior_blend_auto = _suggest_prior_blend(
        float(pd.to_numeric(best_row.get("r2", np.nan), errors="coerce")),
        measurement_noise_var,
        pd.to_numeric(df[target_col], errors="coerce").iloc[train_ilocs].to_numpy(dtype=float)
        if target_col in df.columns and len(train_ilocs) > 0 else np.array([], dtype=float),
    )

    proc_model = MLGapResidualProcessModel(
        df=df,
        target_col=target_col,
        output_col=output_col,
        state_offset=state_offset,
        norm_bounds=bundle.norm_bounds,
        pred_res_norm=pred_res_norm,
        noise_std=process_noise_std,
        model_type=str(bundle.model_type),
        dataset_dir=dataset_dir,
        forecast_namespace=forecast_namespace,
    )

    return {
        "process_model_type": "best_ml_gap_residual",
        "variant_name": str(best_row.get("model", bundle.model_type)),
        "mode": "gap_residual_ml",
        "state_offset": state_offset,
        "feature_names_ext": list(bundle.input_columns),
        "meta": {
            "selected_features": list(bundle.input_columns),
            "coefficients": [],
            "intercept": np.nan,
            "failure_reason": None,
        },
        "r2": float(pd.to_numeric(best_row.get("r2", np.nan), errors="coerce")),
        "rmse": rmse_raw,
        "n_train": int(len(train_ilocs)),
        "n_obs_total": int(len(obs_ilocs)),
        "train_ilocs": train_ilocs.astype(int).tolist(),
        "holdout_ilocs": holdout_ilocs.astype(int).tolist(),
        "process_noise_std": float(max(process_noise_std, 1e-6)),
        "process_noise_source": "saved_ml_rmse",
        "process_noise_mc_samples": np.nan,
        "measurement_noise_var": float(max(measurement_noise_var, 1e-12)),
        "measurement_noise_source": "saved_ml_rmse",
        "residual_std_train": rmse_raw,
        "signal_std_train": float(np.nanstd(pd.to_numeric(df[target_col], errors='coerce').iloc[train_ilocs], ddof=1))
        if target_col in df.columns and len(train_ilocs) >= 2 else np.nan,
        "prior_blend_auto": prior_blend_auto,
        "proc_model": proc_model,
        "pred_res_norm": pred_res_norm,
        "ml_model_type": str(bundle.model_type),
        "ml_output_column": output_col,
        "ml_dataset_dir": str(dataset_dir),
        "ml_forecast_namespace": forecast_namespace,
        "n_valid_process_preds": int(np.isfinite(pred_res_norm).sum()),
    }


def build_mlr_gap_process_model_for_target(
    df: pd.DataFrame,
    target_col: str,
    cv_dir: Path,
    process_noise_override: float | None = None,
) -> dict | None:
    """Build a saved-MLR long-gap residual process model from CV artifacts."""
    dataset_dir = _dataset_dir_for_target(cv_dir, target_col)
    if dataset_dir is None:
        print(f"  [WARN] No dataset directory found in {cv_dir} for '{target_col}'.")
        return None

    fm_csv = dataset_dir / "forecasts" / "feature_sweeps" / "feature_sweep_final_metrics.csv"
    if not fm_csv.exists():
        print(f"  [WARN] feature_sweep_final_metrics.csv not found: {fm_csv}")
        return None

    try:
        df_fm = pd.read_csv(fm_csv)
    except Exception as exc:
        print(f"  [WARN] Could not read {fm_csv}: {exc}")
        return None
    if df_fm.empty:
        print(f"  [WARN] Empty feature_sweep_final_metrics.csv: {fm_csv}")
        return None

    valid_df = _filter_sweep_rows(df_fm)
    mlr_df = valid_df[
        (~valid_df["model"].apply(_is_baseline_model))
        & (valid_df["model"].astype(str).str.lower().isin(_SAVED_MLR_PROCESS_MODELS))
    ].copy()
    if mlr_df.empty:
        print(f"  [WARN] No valid saved MLR rows found for '{target_col}' in {fm_csv}.")
        return None

    best_row = select_best_model_row(mlr_df)
    if best_row is None or getattr(best_row, "empty", False):
        print(f"  [WARN] Could not select a best saved MLR row for '{target_col}'.")
        return None

    forecast_namespace = _resolve_ml_forecast_namespace(best_row)
    bundle = _load_model_bundle(dataset_dir, "feature_sweeps", best_row)
    if bundle is None:
        print(f"  [WARN] Could not load best saved MLR bundle for '{target_col}'.")
        return None

    if str(bundle.model_type).strip().lower() not in _SAVED_MLR_PROCESS_MODELS:
        print(
            f"  [WARN] Best loaded bundle for '{target_col}' is '{bundle.model_type}', "
            "not an eligible saved MLR model."
        )
        return None

    pred_res_norm = _predict_all_rows(bundle, df)
    state_offset = _compute_state_offset(df, target_col) if target_col in df.columns else 1
    rmse_norm = float(pd.to_numeric(best_row.get("rmse", np.nan), errors="coerce"))
    output_col = str(bundle.output_column)
    rmse_raw = _denormalize_value(rmse_norm, output_col, bundle.norm_bounds) if np.isfinite(rmse_norm) else np.nan
    rmse_raw = abs(float(rmse_raw)) if np.isfinite(rmse_raw) else np.nan
    process_noise_std = (
        float(process_noise_override)
        if process_noise_override is not None
        else (rmse_raw if np.isfinite(rmse_raw) and rmse_raw > 0 else 1.0)
    )
    measurement_noise_var = (
        rmse_raw ** 2 if np.isfinite(rmse_raw) and rmse_raw > 0 else 1.0
    )
    obs_ilocs, train_ilocs, holdout_ilocs = _observed_split_ilocs(df, target_col)
    prior_blend_auto = _suggest_prior_blend(
        float(pd.to_numeric(best_row.get("r2", np.nan), errors="coerce")),
        measurement_noise_var,
        pd.to_numeric(df[target_col], errors="coerce").iloc[train_ilocs].to_numpy(dtype=float)
        if target_col in df.columns and len(train_ilocs) > 0 else np.array([], dtype=float),
    )

    proc_model = MLRGapResidualProcessModel(
        df=df,
        target_col=target_col,
        output_col=output_col,
        state_offset=state_offset,
        norm_bounds=bundle.norm_bounds,
        pred_res_norm=pred_res_norm,
        noise_std=process_noise_std,
        model_type=str(bundle.model_type),
        dataset_dir=dataset_dir,
        forecast_namespace=forecast_namespace,
    )

    meta_selected = []
    meta_coeffs = []
    meta_intercept = np.nan
    if getattr(bundle, "mlr_selected_features", None) is not None:
        meta_selected = list(bundle.mlr_selected_features or [])
    if getattr(bundle, "mlr_coefficients", None) is not None:
        meta_coeffs = [float(c) for c in np.asarray(bundle.mlr_coefficients, dtype=float).tolist()]
    if getattr(bundle, "mlr_intercept", None) is not None:
        try:
            meta_intercept = float(bundle.mlr_intercept)
        except Exception:
            meta_intercept = np.nan

    return {
        "process_model_type": "best_mlr_gap_residual",
        "variant_name": str(best_row.get("model", bundle.model_type)),
        "mode": "gap_residual_saved_mlr",
        "state_offset": state_offset,
        "feature_names_ext": list(bundle.input_columns),
        "meta": {
            "selected_features": meta_selected,
            "coefficients": meta_coeffs,
            "intercept": meta_intercept,
            "failure_reason": None,
        },
        "r2": float(pd.to_numeric(best_row.get("r2", np.nan), errors="coerce")),
        "rmse": rmse_raw,
        "n_train": int(len(train_ilocs)),
        "n_obs_total": int(len(obs_ilocs)),
        "train_ilocs": train_ilocs.astype(int).tolist(),
        "holdout_ilocs": holdout_ilocs.astype(int).tolist(),
        "process_noise_std": float(max(process_noise_std, 1e-6)),
        "process_noise_source": "saved_mlr_rmse",
        "process_noise_mc_samples": np.nan,
        "measurement_noise_var": float(max(measurement_noise_var, 1e-12)),
        "measurement_noise_source": "saved_mlr_rmse",
        "residual_std_train": rmse_raw,
        "signal_std_train": float(np.nanstd(pd.to_numeric(df[target_col], errors='coerce').iloc[train_ilocs], ddof=1))
        if target_col in df.columns and len(train_ilocs) >= 2 else np.nan,
        "prior_blend_auto": prior_blend_auto,
        "proc_model": proc_model,
        "pred_res_norm": pred_res_norm,
        "ml_model_type": str(bundle.model_type),
        "ml_output_column": output_col,
        "ml_dataset_dir": str(dataset_dir),
        "ml_forecast_namespace": forecast_namespace,
        "n_valid_process_preds": int(np.isfinite(pred_res_norm).sum()),
    }


# ---------------------------------------------------------------------------
# Core particle filter
# ---------------------------------------------------------------------------

def run_particle_filter(
    df: pd.DataFrame,
    target_col: str,
    process_model: ProcessModel,
    obs_model: ObservationModel,
    n_particles: int,
    prior_blend: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run a bootstrap particle filter for one target over all rows of df.

    Parameters
    ----------
    df : full input DataFrame (index = integer positional).
    target_col : name of the target column.
    process_model : implements ProcessModel protocol.
    obs_model : implements ObservationModel protocol.
    n_particles : number of particles N.
    prior_blend : float in [0, 1]; weight pulling particles toward the MLR prior.
    rng : random generator.

    Returns
    -------
    pf_mean : ndarray [n_rows] — posterior mean at each row (NaN before first obs).
    pf_std  : ndarray [n_rows] — posterior std  at each row (NaN before first obs).
    update_records : list of dicts, one per observation row (excluding the first,
        which has no prior estimate to compare against).  Each dict has:
          row_idx       — integer positional index in df
          timestamp     — TIMESTAMP value at that row (or NaT)
          z             — the observed measurement value
          prior_mean    — particle mean immediately after propagation, before update
          prior_std     — particle std immediately after propagation, before update
          error         — prior_mean - z  (signed; positive = filter overestimated)
          abs_error     — |prior_mean - z|
          z_score       — error / prior_std  (NaN if prior_std == 0)
    """
    N = int(n_particles)
    if N < 2:
        raise ValueError(f"n_particles must be >= 2, got {n_particles!r}")
    if not np.isfinite(prior_blend) or not (0.0 <= float(prior_blend) <= 1.0):
        raise ValueError(f"prior_blend must be between 0 and 1, got {prior_blend!r}")
    prior_blend = float(prior_blend)
    n_rows = len(df)
    pf_mean = np.full(n_rows, np.nan, dtype=float)
    pf_std  = np.full(n_rows, np.nan, dtype=float)
    update_records: list[dict] = []

    target_series = pd.to_numeric(df[target_col], errors="coerce")
    obs_values = target_series.values  # fast array access

    ts_values = df["TIMESTAMP"].values if "TIMESTAMP" in df.columns else None

    # Sorted integer positions of actual measurements
    obs_ilocs = np.where(np.isfinite(obs_values))[0]
    if len(obs_ilocs) == 0:
        return pf_mean, pf_std, update_records

    first_obs_iloc = int(obs_ilocs[0])
    first_obs_val  = float(obs_values[first_obs_iloc])
    sigma_q = process_model.noise_std

    # ------------------------------------------------------------------
    # Initialise particles at the first measurement row
    # ------------------------------------------------------------------
    mu0 = process_model.predict(df, first_obs_iloc, state_history=pf_mean)
    if not np.isfinite(mu0):
        mu0 = first_obs_val

    particles = rng.normal(loc=mu0, scale=sigma_q, size=N)
    weights   = np.ones(N, dtype=float) / N

    # Observation update at first measurement — no prior estimate to record
    log_w = np.array(
        [obs_model.log_likelihood(first_obs_val, p) for p in particles]
    )
    log_w -= np.max(log_w)   # numerical stability
    weights = np.exp(log_w)
    w_sum = weights.sum()
    if w_sum > 0:
        weights /= w_sum
    else:
        weights = np.ones(N, dtype=float) / N

    # Always resample after the first measurement
    particles = _systematic_resample(particles, weights, rng)
    weights   = np.ones(N, dtype=float) / N

    pf_mean[first_obs_iloc] = float(np.dot(weights, particles))
    pf_std[first_obs_iloc]  = float(
        np.sqrt(np.dot(weights, (particles - pf_mean[first_obs_iloc]) ** 2))
    )

    # ------------------------------------------------------------------
    # Main sequential filter loop
    # ------------------------------------------------------------------
    for row_idx in range(first_obs_iloc + 1, n_rows):

        # 1. Propagation
        mu_t = process_model.predict(df, row_idx, state_history=pf_mean)
        if np.isfinite(mu_t):
            # Blend the previous posterior particles toward the MLR prior so
            # that dense predictors guide the state without fully erasing the
            # last measurement update.
            particles = (
                (1.0 - prior_blend) * particles
                + prior_blend * mu_t
                + rng.normal(0.0, sigma_q, N)
            )
        else:
            # No predictor data: carry forward with pure Gaussian diffusion
            particles = particles + rng.normal(0.0, sigma_q, N)

        # 2. Observation update (if a measurement is available)
        z = obs_values[row_idx]
        if np.isfinite(z):
            # Capture prior (pre-update) statistics for jump analysis
            prior_mu  = float(np.dot(weights, particles))
            prior_sig = float(np.sqrt(np.dot(weights, (particles - prior_mu) ** 2)))
            err = prior_mu - float(z)
            update_records.append({
                "row_idx":    row_idx,
                "timestamp":  ts_values[row_idx] if ts_values is not None else None,
                "z":          float(z),
                "prior_mean": prior_mu,
                "prior_std":  prior_sig,
                "error":      err,
                "abs_error":  abs(err),
                "z_score":    err / prior_sig if prior_sig > 0 else np.nan,
            })

            log_w = np.array(
                [obs_model.log_likelihood(float(z), p) for p in particles]
            )
            log_w -= np.max(log_w)
            weights = np.exp(log_w)
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum
            else:
                weights = np.ones(N, dtype=float) / N

            # 3. Resample if ESS drops below N/2
            if _effective_sample_size(weights) < N / 2:
                particles = _systematic_resample(particles, weights, rng)
                weights   = np.ones(N, dtype=float) / N

        # 4. Record posterior statistics
        mu = float(np.dot(weights, particles))
        pf_mean[row_idx] = mu
        pf_std[row_idx]  = float(
            np.sqrt(np.dot(weights, (particles - mu) ** 2))
        )

    return pf_mean, pf_std, update_records


def _build_tuning_validation_ilocs(
    train_ilocs: list[int] | np.ndarray,
    all_obs_ilocs: list[int] | np.ndarray,
    n_folds: int,
) -> tuple[list[np.ndarray], set[int]]:
    """Return sequential validation folds within the train observations.

    The startup interval ending at the second overall observation is excluded
    because that segment does not yet use the usual recursive-lookback behavior.
    """
    train_arr = np.asarray(train_ilocs, dtype=int)
    obs_arr = np.asarray(all_obs_ilocs, dtype=int)
    if train_arr.size == 0:
        return [], set()

    excluded = set()
    if obs_arr.size >= 2:
        excluded.add(int(obs_arr[1]))
    candidates = np.array([i for i in train_arr if int(i) not in excluded], dtype=int)
    if candidates.size == 0:
        return [], set()

    n_parts = max(1, min(int(n_folds), int(candidates.size)))
    folds = [arr.astype(int) for arr in np.array_split(candidates, n_parts) if len(arr) > 0]
    scored = {int(i) for fold in folds for i in fold.tolist()}
    return folds, scored


def _compute_tuning_metrics(
    update_records: list[dict],
    scored_ilocs: set[int],
) -> dict:
    """Compute RMSE-primary tuning metrics on a scored subset of update rows."""
    records = [r for r in update_records if int(r.get("row_idx", -1)) in scored_ilocs]
    if not records:
        return {
            "n_scored": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "calib_penalty": np.nan,
            "nll": np.nan,
            "score": np.inf,
        }

    y_true = np.array([r["z"] for r in records], dtype=float)
    y_pred = np.array([r["prior_mean"] for r in records], dtype=float)
    prior_std = np.array([r["prior_std"] for r in records], dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.count_nonzero(mask)) < 1:
        return {
            "n_scored": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "calib_penalty": np.nan,
            "nll": np.nan,
            "score": np.inf,
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    ps = prior_std[mask]
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))

    z_mask = np.isfinite(ps) & (ps > 1e-12)
    if int(np.count_nonzero(z_mask)) >= 1:
        innov = (yp[z_mask] - yt[z_mask]) / ps[z_mask]
        innov_mean = float(np.mean(innov))
        innov_var = float(np.var(innov, ddof=1)) if innov.size >= 2 else float(np.var(innov))
        frac1 = float(np.mean(np.abs(innov) <= 1.0))
        frac2 = float(np.mean(np.abs(innov) <= 2.0))
        calib_penalty = (
            abs(innov_mean)
            + abs(np.log(max(innov_var, 1e-12)))
            + max(0.0, 0.68 - frac1)
            + max(0.0, 0.95 - frac2)
        )
        nll = float(np.mean(0.5 * np.log(2.0 * np.pi * (ps[z_mask] ** 2)) + 0.5 * (innov ** 2)))
    else:
        calib_penalty = 10.0
        nll = np.inf

    scale = float(np.nanstd(yt, ddof=1)) if yt.size >= 2 else float(np.nanstd(yt))
    scale = max(scale, 1e-3)
    score = rmse + 0.05 * scale * calib_penalty + 0.01 * scale * (nll if np.isfinite(nll) else 100.0)
    return {
        "n_scored": int(mask.sum()),
        "rmse": rmse,
        "mae": mae,
        "calib_penalty": float(calib_penalty),
        "nll": float(nll) if np.isfinite(nll) else np.inf,
        "score": float(score),
    }


def _candidate_values_around_default(default: float, scales: list[float], floor: float) -> list[float]:
    vals = []
    for s in scales:
        val = max(floor, float(default) * float(s))
        vals.append(val)
    vals.append(max(floor, float(default)))
    return sorted({float(v) for v in vals if np.isfinite(v)})


def tune_filter_hyperparameters_for_target(
    df: pd.DataFrame,
    target_col: str,
    process_model: ProcessModel,
    default_sigma_q: float,
    default_measurement_noise_var: float,
    default_prior_blend: float,
    n_particles: int,
    seed: int,
    train_ilocs: list[int] | np.ndarray,
    all_obs_ilocs: list[int] | np.ndarray,
    n_folds: int = 3,
    grid_scale: int = 1,
) -> tuple[dict, list[dict]]:
    """Tune sigma_Q, R, and prior_blend on train observations only."""
    folds, scored_ilocs = _build_tuning_validation_ilocs(train_ilocs, all_obs_ilocs, n_folds)
    if not scored_ilocs:
        baseline = {
            "sigma_q": float(default_sigma_q),
            "measurement_noise_var": float(default_measurement_noise_var),
            "measurement_noise_std": float(np.sqrt(default_measurement_noise_var)),
            "prior_blend": float(default_prior_blend),
            "score": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "n_scored": 0,
            "tuning_enabled": False,
            "tuning_n_folds": len(folds),
        }
        return baseline, []

    scale_mult = [0.25, 0.5, 1.0, 2.0, 4.0] if int(grid_scale) <= 1 else [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    blend_candidates = np.clip(
        np.array([
            default_prior_blend * 0.5,
            default_prior_blend * 0.75,
            default_prior_blend,
            min(1.0, default_prior_blend * 1.25),
            min(1.0, default_prior_blend * 1.5),
        ], dtype=float),
        0.0, 1.0,
    )
    blend_candidates = sorted({float(v) for v in blend_candidates if np.isfinite(v)})

    sigma_candidates = _candidate_values_around_default(default_sigma_q, scale_mult, 1e-6)
    meas_std_default = float(np.sqrt(max(default_measurement_noise_var, 1e-12)))
    meas_std_candidates = _candidate_values_around_default(meas_std_default, scale_mult, 1e-6)

    tried_rows: list[dict] = []

    def _evaluate_candidate(sig_q: float, meas_std: float, blend: float) -> dict:
        proc = NoiseOverrideProcessModel(process_model, sig_q)
        obs = GaussianObservationModel(measurement_noise_var=float(meas_std) ** 2)
        rng_local = np.random.default_rng(int(seed))
        _, _, records = run_particle_filter(
            df=df,
            target_col=target_col,
            process_model=proc,
            obs_model=obs,
            n_particles=n_particles,
            prior_blend=blend,
            rng=rng_local,
        )
        metrics = _compute_tuning_metrics(records, scored_ilocs)
        row = {
            "target": target_col,
            "sigma_q": float(sig_q),
            "measurement_noise_std": float(meas_std),
            "measurement_noise_var": float(meas_std) ** 2,
            "prior_blend": float(blend),
            "score": metrics["score"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "calib_penalty": metrics["calib_penalty"],
            "nll": metrics["nll"],
            "n_scored": metrics["n_scored"],
            "n_folds": len(folds),
        }
        tried_rows.append(row)
        return row

    best = _evaluate_candidate(default_sigma_q, meas_std_default, default_prior_blend)
    baseline = dict(best)

    for blend in blend_candidates:
        cand = _evaluate_candidate(default_sigma_q, meas_std_default, blend)
        if cand["score"] < best["score"]:
            best = cand

    for sig_q in sigma_candidates:
        for meas_std in meas_std_candidates:
            cand = _evaluate_candidate(sig_q, meas_std, best["prior_blend"])
            if cand["score"] < best["score"]:
                best = cand

    local_sigma = _candidate_values_around_default(best["sigma_q"], [0.5, 0.75, 1.0, 1.5, 2.0], 1e-6)
    local_meas = _candidate_values_around_default(best["measurement_noise_std"], [0.5, 0.75, 1.0, 1.5, 2.0], 1e-6)
    local_blend = sorted({
        float(np.clip(v, 0.0, 1.0)) for v in [
            best["prior_blend"] * 0.8,
            best["prior_blend"] * 0.9,
            best["prior_blend"],
            min(1.0, best["prior_blend"] * 1.1),
            min(1.0, best["prior_blend"] * 1.2),
        ]
    })
    for blend in local_blend:
        for sig_q in local_sigma:
            for meas_std in local_meas:
                cand = _evaluate_candidate(sig_q, meas_std, blend)
                if cand["score"] < best["score"]:
                    best = cand

    result = {
        "sigma_q": float(best["sigma_q"]),
        "measurement_noise_var": float(best["measurement_noise_var"]),
        "measurement_noise_std": float(best["measurement_noise_std"]),
        "prior_blend": float(best["prior_blend"]),
        "score": float(best["score"]),
        "rmse": float(best["rmse"]),
        "mae": float(best["mae"]),
        "n_scored": int(best["n_scored"]),
        "baseline_score": float(baseline["score"]),
        "baseline_rmse": float(baseline["rmse"]),
        "baseline_mae": float(baseline["mae"]),
        "tuning_enabled": True,
        "tuning_n_folds": len(folds),
    }
    return result, tried_rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _compute_jump_stats(update_records: list[dict]) -> dict:
    """Summarise pre-update prediction errors across all observation updates.

    Parameters
    ----------
    update_records : list of dicts returned by run_particle_filter() —
        one entry per observation row (excluding the initialisation row).

    Returns
    -------
    dict with keys:
        n_updates       — number of observation updates recorded
        mean_error      — mean signed error (prior_mean - z); bias indicator
        std_error       — std of signed errors; spread indicator
        mae             — mean absolute error
        rmse            — root mean squared error
        median_error    — median signed error
        median_abs_error — median absolute error
        p10_error / p90_error — 10th/90th percentiles of signed error
        p10_abs / p90_abs     — 10th/90th percentiles of |error|
        mean_z_score    — mean |z-score| (how many prior-σ off was the filter)
        frac_within_1sigma — fraction of updates where |z_score| <= 1
        frac_within_2sigma — fraction of updates where |z_score| <= 2
    """
    if not update_records:
        return {"n_updates": 0}

    errors   = np.array([r["error"]     for r in update_records], dtype=float)
    abs_errs = np.array([r["abs_error"] for r in update_records], dtype=float)
    zscores  = np.array([r["z_score"]   for r in update_records], dtype=float)

    finite_e = errors[np.isfinite(errors)]
    finite_a = abs_errs[np.isfinite(abs_errs)]
    finite_z = zscores[np.isfinite(zscores)]

    def _safe(fn, arr, fallback=np.nan):
        return float(fn(arr)) if len(arr) > 0 else fallback

    stats: dict = {
        "n_updates":          len(update_records),
        "mean_error":         _safe(np.mean,   finite_e),
        "std_error":          _safe(np.std,    finite_e),
        "mae":                _safe(np.mean,   finite_a),
        "rmse":               _safe(lambda a: np.sqrt(np.mean(a ** 2)), finite_e),
        "median_error":       _safe(np.median, finite_e),
        "median_abs_error":   _safe(np.median, finite_a),
        "p10_error":          _safe(lambda a: np.percentile(a, 10), finite_e),
        "p90_error":          _safe(lambda a: np.percentile(a, 90), finite_e),
        "p10_abs":            _safe(lambda a: np.percentile(a, 10), finite_a),
        "p90_abs":            _safe(lambda a: np.percentile(a, 90), finite_a),
        "mean_abs_z_score":   _safe(lambda a: np.mean(np.abs(a)), finite_z),
        "frac_within_1sigma": _safe(lambda a: np.mean(np.abs(a) <= 1.0), finite_z),
        "frac_within_2sigma": _safe(lambda a: np.mean(np.abs(a) <= 2.0), finite_z),
    }
    return stats


# ---------------------------------------------------------------------------
# CV-dir integration: normalization lookup and evaluation_summary writer
# ---------------------------------------------------------------------------

# Dataset directory name prefixes for each target — mirrors the naming
# convention used by h_RunMCFeatureSelectionSweep (e.g. "MC_pH_res").
# Built lazily from the cv_dir at runtime; see _find_normalization().
_NORM_CACHE: dict[str, dict] = {}   # target_col -> {col: {min, max}}


def _sanitise_for_dirname(name: str) -> str:
    """Sanitise a column name to the form used in dataset directory names.

    Mirrors the convention used by the pipeline: spaces, parentheses, dots,
    slashes, colons, and degree symbols are replaced with underscores, then
    consecutive underscores are collapsed.
    """
    import re as _re
    s = name
    for ch in (" ", "(", ")", ".", "/", ":", "\u00b0"):
        s = s.replace(ch, "_")
    s = _re.sub(r"_+", "_", s)
    return s.strip("_")


def _dataset_dir_for_target(cv_dir: Path, target_col: str) -> Path | None:
    """Return the MC_<target>_res dataset directory for *target_col*, or None.

    Matches directory names against a sanitised version of the target name.
    E.g. target_col="Arsenic (µg/L)" → sanitised "Arsenic__µg_L_" → matches
    directory "MC_Arsenic__µg_L__res".
    """
    if not cv_dir.is_dir():
        return None
    sanitised = _sanitise_for_dirname(target_col)
    # Look for MC_<sanitised>_res (exact) or a directory whose name contains
    # the sanitised target and ends with _res.
    for d in sorted(cv_dir.iterdir()):
        if not d.is_dir() or d.name == "summaries":
            continue
        # Strip the MC_ prefix for matching
        dname = d.name
        if dname.startswith("MC_"):
            dname_core = dname[3:]  # e.g. "Arsenic__µg_L__res"
        else:
            dname_core = dname
        if not dname_core.endswith("_res"):
            continue
        dname_target = dname_core[:-4]  # e.g. "Arsenic__µg_L_"
        # Compare sanitised forms
        if _sanitise_for_dirname(dname_target) == sanitised:
            return d
    return None


def _find_normalization(cv_dir: Path, target_col: str) -> dict:
    """Load normalization.json for *target_col* from *cv_dir*.

    Returns a dict {col: {"min": float, "max": float}} or {} if not found.
    Results are cached for the lifetime of the process.
    """
    if target_col in _NORM_CACHE:
        return _NORM_CACHE[target_col]
    d = _dataset_dir_for_target(cv_dir, target_col)
    if d is None:
        _NORM_CACHE[target_col] = {}
        return {}
    norm_path = d / "normalization.json"
    try:
        import json as _json
        norm = _json.loads(norm_path.read_text(encoding="utf-8"))
        _NORM_CACHE[target_col] = norm
        return norm
    except Exception:
        _NORM_CACHE[target_col] = {}
        return {}


def _compute_evaluation_rows(
    target_col: str,
    update_records: list[dict],
    df: pd.DataFrame,
    norm_bounds: dict,
    eval_row_indices: set[int] | None = None,
) -> list[dict]:
    """Build evaluation_summary.csv rows for the particle filter predictions.

    At each observation update the filter had a prior mean (pf_mean before
    the update was applied).  These priors are treated as predictions.

    Two rows are produced:
      label="particle_filter (absolute)"    — raw physical units
      label="particle_filter (res_norm)"    — normalised _res units,
          comparable to existing pipeline metrics

    Parameters
    ----------
    update_records : list[dict] from run_particle_filter() — each entry has
        z (observed value), prior_mean, and timestamp.
    df : full input DataFrame (must have a forward-filled _state column for
        computing _res).
    norm_bounds : dict from normalization.json — must contain the _res key
        for this target (e.g. "pH_res") to produce the normalised row.

    Returns
    -------
    List of 1 or 2 row dicts matching the f_Evaluate evaluation_summary schema.
    """
    from scipy.stats import pearsonr as _pearsonr

    if eval_row_indices is not None:
        update_records = [
            r for r in update_records
            if int(r.get("row_idx", -1)) in eval_row_indices
        ]

    if not update_records:
        return []

    # ---- Absolute-unit vectors ----
    y_pred_abs = np.array([r["prior_mean"] for r in update_records], dtype=float)
    y_true_abs = np.array([r["z"]          for r in update_records], dtype=float)
    mask_abs   = np.isfinite(y_pred_abs) & np.isfinite(y_true_abs)

    def _metrics(y_true, y_pred):
        """Return mae, rmse, r2, pearson_r or all-nan on failure."""
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        if m.sum() < 2:
            return np.nan, np.nan, np.nan, np.nan
        yt, yp = y_true[m], y_pred[m]
        mae  = float(np.mean(np.abs(yp - yt)))
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        try:
            pr, _ = _pearsonr(yt, yp)
        except Exception:
            pr = np.nan
        return mae, rmse, float(r2), float(pr)

    mae_a, rmse_a, r2_a, pr_a = _metrics(y_true_abs, y_pred_abs)
    n_finite = int(mask_abs.sum())

    _SCHEMA = {
        "mae_replicate": np.nan, "rmse_replicate": np.nan,
        "r2_replicate": np.nan, "pearson_r_replicate": np.nan,
        "skill_v_naive": np.nan,
        "kind": "test",
        "gp_uncertainty_mode": "not_gp",
        "n_test_independent": n_finite,
        "n_test_valid": n_finite,
        "n_test_evals": len(update_records),
        "n_train_samples": np.nan,
        "n_test_samples": n_finite,
        "n_eval_rows": len(update_records),
        "n_eval_outputs": 1,
        "n_eval_points_finite": n_finite,
        "n_eval_points_finite_replicate": np.nan,
        "metric_semantics": "independent_sample_primary",
        "metric_contract_version": 1,
        "input_dim": np.nan,
        "target_dim": 1,
        "data_dir": "",
    }

    row_abs = {
        "target":    target_col,
        "label": "particle_filter (absolute)",
        "mae":       mae_a,
        "rmse":      rmse_a,
        "r2":        r2_a,
        "pearson_r": pr_a,
        "std_target": float(np.nanstd(y_true_abs)) if n_finite >= 2 else np.nan,
        **_SCHEMA,
    }
    rows = [row_abs]

    # ---- Normalised _res row ----
    res_col = f"{target_col}_res"
    state_col = f"{target_col}_state"
    res_bounds = norm_bounds.get(res_col)

    if res_bounds is not None and state_col in df.columns:
        res_min  = float(res_bounds.get("min", 0.0))
        res_max  = float(res_bounds.get("max", 1.0))
        res_span = res_max - res_min
        if res_span > 0:
            # Forward-fill the _state column to get the most recent prior
            # measurement at each update row (same as how _res is computed
            # in preprocessing.add_res).
            state_ff = pd.to_numeric(df[state_col], errors="coerce").ffill().values

            pred_res_norm = np.full(len(update_records), np.nan)
            true_res_norm = np.full(len(update_records), np.nan)

            for k, rec in enumerate(update_records):
                ri = rec["row_idx"]
                # state at the previous timestep (before this observation)
                prev_state = state_ff[ri - 1] if ri > 0 else np.nan
                if not np.isfinite(prev_state):
                    continue
                pred_res = rec["prior_mean"] - prev_state
                true_res = rec["z"]          - prev_state
                pred_res_norm[k] = pred_res / res_span
                true_res_norm[k] = true_res / res_span

            mae_n, rmse_n, r2_n, pr_n = _metrics(true_res_norm, pred_res_norm)
            mask_n = np.isfinite(pred_res_norm) & np.isfinite(true_res_norm)
            n_finite_n = int(mask_n.sum())

            row_norm = {
                "target":    target_col,
                "label": "particle_filter (res_norm)",
                "mae":       mae_n,
                "rmse":      rmse_n,
                "r2":        r2_n,
                "pearson_r": pr_n,
                "std_target": float(np.nanstd(true_res_norm[mask_n])) if n_finite_n >= 2 else np.nan,
                **{**_SCHEMA,
                   "n_test_independent": n_finite_n,
                   "n_test_valid": n_finite_n,
                   "n_test_samples": n_finite_n,
                   "n_eval_points_finite": n_finite_n},
            }
            rows.append(row_norm)

    return rows


def _write_evaluation_summary(rows: list[dict], output_path: Path) -> None:
    """Write evaluation_summary.csv in f_Evaluate's 26-column format.

    A leading ``target`` column is prepended so that each row in the combined
    file can be traced back to its target variable.
    """
    ordered_columns = [
        "target", "label", "mae", "rmse", "r2", "pearson_r", "std_target",
        "mae_replicate", "rmse_replicate", "r2_replicate", "pearson_r_replicate",
        "skill_v_naive", "kind", "gp_uncertainty_mode",
        "n_test_independent", "n_test_valid", "n_test_evals",
        "n_train_samples", "n_test_samples", "n_eval_rows", "n_eval_outputs",
        "n_eval_points_finite", "n_eval_points_finite_replicate",
        "metric_semantics", "metric_contract_version", "input_dim",
        "target_dim", "data_dir",
    ]
    if not rows:
        return
    df_out = pd.DataFrame(rows)
    for col in ordered_columns:
        if col not in df_out.columns:
            df_out[col] = np.nan
    df_out = df_out[[c for c in ordered_columns if c in df_out.columns]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"  evaluation_summary written to: {output_path}")


def _pf_forecast_name(
    process_model: str,
    run_name: str | None = None,
    tuned: bool = False,
) -> str:
    """Return a stable forecast directory name for PF artifact export."""
    parts = ["particle_filter", str(process_model).strip().lower()]
    if tuned:
        parts.append("tuned")
    if run_name:
        parts.append(_sanitise_for_dirname(str(run_name)))
    return "_".join(p for p in parts if p)


def _build_pf_predictions_rows(
    target_col: str,
    update_records: list[dict],
    df: pd.DataFrame,
    norm_bounds: dict,
    eval_row_indices: set[int] | None = None,
) -> tuple[list[dict], list[str]]:
    """Build PF predictions.csv rows in normalized _res units.

    Each held-out observation update is treated as one independent test sample,
    matching the semantics used by the existing model-evaluation artifacts.
    """
    from f_Evaluate import _write_predictions_csv  # imported lazily for parity

    del _write_predictions_csv  # suppress lint-style unused warning

    if eval_row_indices is not None:
        update_records = [
            r for r in update_records
            if int(r.get("row_idx", -1)) in eval_row_indices
        ]
    if not update_records:
        return [], []

    res_col = f"{target_col}_res"
    state_col = f"{target_col}_state"
    res_bounds = norm_bounds.get(res_col)
    if res_bounds is None or state_col not in df.columns:
        return [], []

    res_min = float(res_bounds.get("min", 0.0))
    res_max = float(res_bounds.get("max", 1.0))
    res_span = res_max - res_min
    if not np.isfinite(res_span) or res_span <= 0:
        return [], []

    state_ff = pd.to_numeric(df[state_col], errors="coerce").ffill().to_numpy(dtype=float)

    rows: list[dict] = []
    split_files: list[str] = []
    for rec in update_records:
        row_idx = int(rec.get("row_idx", -1))
        if row_idx <= 0 or row_idx >= len(state_ff):
            continue
        prev_state = state_ff[row_idx - 1]
        pred_abs = float(rec.get("prior_mean", np.nan))
        true_abs = float(rec.get("z", np.nan))
        if not (np.isfinite(prev_state) and np.isfinite(pred_abs) and np.isfinite(true_abs)):
            continue

        pred_res_norm = (pred_abs - prev_state) / res_span
        true_res_norm = (true_abs - prev_state) / res_span
        if not (np.isfinite(pred_res_norm) and np.isfinite(true_res_norm)):
            continue

        sample_file = f"pf_row_{row_idx:05d}.csv"
        split_files.append(sample_file)
        rows.append({
            "kind": "test",
            "sample_file": sample_file,
            "gp_uncertainty_mode": "not_gp",
            "metric_semantics": "independent_sample_primary",
            "metric_contract_version": 1,
            "target": true_res_norm,
            "Particle Filter": pred_res_norm,
            "pf_prior_std_abs": float(rec.get("prior_std", np.nan)),
            "timestamp": rec.get("timestamp"),
            "row_idx": row_idx,
        })

    return rows, split_files


def _write_pf_forecast_artifacts(
    *,
    cv_dir: Path,
    target_col: str,
    df: pd.DataFrame,
    update_records: list[dict],
    eval_rows: list[dict],
    eval_row_indices: set[int],
    process_model: str,
    run_name: str | None,
    tuned: bool,
    make_plots: bool,
) -> Path | None:
    """Write PF evaluation artifacts under the target dataset forecast dir."""
    dataset_dir = _dataset_dir_for_target(cv_dir, target_col)
    if dataset_dir is None:
        print(f"  [WARN] Could not locate dataset directory for '{target_col}'; skipping PF forecast artifacts.")
        return None

    forecast_name = _pf_forecast_name(
        process_model=process_model,
        run_name=run_name,
        tuned=tuned,
    )
    forecast_dir = dataset_dir / "forecasts" / forecast_name
    forecast_dir.mkdir(parents=True, exist_ok=True)

    eval_summary_path = forecast_dir / "evaluation_summary.csv"
    _write_evaluation_summary(eval_rows, eval_summary_path)

    norm_bounds = _find_normalization(cv_dir, target_col)
    pred_rows, split_files = _build_pf_predictions_rows(
        target_col=target_col,
        update_records=update_records,
        df=df,
        norm_bounds=norm_bounds,
        eval_row_indices=eval_row_indices,
    )
    if not pred_rows:
        print(f"  [WARN] No PF prediction rows available for '{target_col}'; skipping predictions.csv/plots.")
        return forecast_dir

    from f_Evaluate import (
        _write_predictions_csv,
        _build_boxplot_error_rows_from_predictions,
    )

    ordered_columns = [
        "kind",
        "sample_file",
        "gp_uncertainty_mode",
        "metric_semantics",
        "metric_contract_version",
        "target",
        "Particle Filter",
        "pf_prior_std_abs",
        "timestamp",
        "row_idx",
    ]
    predictions_path = forecast_dir / "predictions.csv"
    _write_predictions_csv(pred_rows, predictions_path, ordered_columns)

    if make_plots:
        pred_arr = np.array([[float(r["Particle Filter"])] for r in pred_rows], dtype=float)
        target_arr = np.array([[float(r["target"])] for r in pred_rows], dtype=float)
        visualizer(
            (pred_arr, target_arr),
            labels=["Particle Filter"],
            directory=str(dataset_dir),
            forecast_name=forecast_name,
            split_files_by_pair=[split_files],
            collapse_error_points_by_pair=[False],
        )
        boxplot_rows = _build_boxplot_error_rows_from_predictions(
            pred_rows,
            model_label="Particle Filter",
            baseline_labels=[],
        )
        boxplot_from_error_rows(
            boxplot_rows,
            directory=str(dataset_dir),
            forecast_name=forecast_name,
        )

    return forecast_dir


# ---------------------------------------------------------------------------
# Jump-statistics figure
# ---------------------------------------------------------------------------

def _plot_jump_stats(
    all_update_records: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """Three-panel jump-statistics figure, one row of panels per target.

    Panels (left to right):
      1. Histogram of signed errors (prior_mean - z) with mean and ±1σ marked
      2. Signed error vs. time (scatter)
      3. |error| vs. prior_std scatter with y=x reference line

    Parameters
    ----------
    all_update_records : {target_col: [update_record, ...]}
    out_path : full path for the saved PNG
    """
    target_cols = [t for t, recs in all_update_records.items() if recs]
    if not target_cols:
        print("[WARN] No update records to plot for jump stats.")
        return

    n_targets = len(target_cols)
    fig_width  = 15.0
    row_height = 3.2
    fig_height = max(3.2, row_height * n_targets)
    font_size  = 8

    fig, axes_grid = plt.subplots(
        n_targets, 3,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    raw_labels  = list(target_cols)
    stripped    = _strip_common_affixes(raw_labels)
    palette     = _safe_series_colors(n_targets)

    for i, (target_col, label) in enumerate(zip(target_cols, stripped)):
        records = all_update_records[target_col]
        color   = palette[i]

        errors    = np.array([r["error"]     for r in records], dtype=float)
        abs_errs  = np.array([r["abs_error"] for r in records], dtype=float)
        prior_std = np.array([r["prior_std"] for r in records], dtype=float)
        ts_raw    = [r["timestamp"] for r in records]

        try:
            timestamps = pd.to_datetime(ts_raw, errors="coerce")
        except Exception:
            timestamps = pd.array([None] * len(errors))

        fin = np.isfinite(errors)

        ax_hist, ax_time, ax_cal = axes_grid[i]

        # --- Panel 1: error histogram ---
        if fin.any():
            mu_e  = float(np.nanmean(errors[fin]))
            sig_e = float(np.nanstd(errors[fin]))
            ax_hist.hist(errors[fin], bins=min(20, max(5, fin.sum() // 2)),
                         color=color, alpha=0.7, edgecolor="white", linewidth=0.4)
            ax_hist.axvline(mu_e,           color="black",  linewidth=1.2, linestyle="-",
                            label=f"mean={mu_e:.3g}")
            ax_hist.axvline(mu_e + sig_e,   color="black",  linewidth=0.8, linestyle="--")
            ax_hist.axvline(mu_e - sig_e,   color="black",  linewidth=0.8, linestyle="--")
            ax_hist.axvline(0,              color="#888888", linewidth=0.7, linestyle=":")
            ax_hist.legend(fontsize=font_size - 1, loc="upper right")
        ax_hist.set_ylabel(label, rotation=0, ha="right", va="center",
                           fontsize=font_size + 1, labelpad=6)
        ax_hist.set_xlabel("error (prior − z)" if i == n_targets - 1 else "",
                           fontsize=font_size)
        ax_hist.tick_params(labelsize=font_size)
        ax_hist.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.4)
        if i == 0:
            ax_hist.set_title("Error distribution", fontsize=font_size + 1)

        # --- Panel 2: error vs. time ---
        ts_valid = [(t, e) for t, e in zip(timestamps, errors)
                    if t is not None and pd.notna(t) and np.isfinite(e)]
        if ts_valid:
            ts_arr, e_arr = zip(*ts_valid)
            ax_time.scatter(ts_arr, e_arr, s=14, color=color, alpha=0.7,
                            edgecolors="none", zorder=2)
            ax_time.axhline(0, color="#888888", linewidth=0.7, linestyle=":")
            ax_time.tick_params(axis="x", labelsize=font_size - 1, rotation=20)
        ax_time.set_xlabel("time" if i == n_targets - 1 else "", fontsize=font_size)
        ax_time.tick_params(labelsize=font_size)
        ax_time.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.4)
        if i == 0:
            ax_time.set_title("Error vs. time", fontsize=font_size + 1)

        # --- Panel 3: |error| vs. prior_std (calibration) ---
        cal_mask = np.isfinite(abs_errs) & np.isfinite(prior_std) & (prior_std > 0)
        if cal_mask.any():
            ax_cal.scatter(prior_std[cal_mask], abs_errs[cal_mask],
                           s=14, color=color, alpha=0.7, edgecolors="none", zorder=2)
            lim = max(float(prior_std[cal_mask].max()), float(abs_errs[cal_mask].max()))
            ax_cal.plot([0, lim], [0, lim], color="#888888", linewidth=0.8,
                        linestyle="--", zorder=1, label="y = x")
            ax_cal.legend(fontsize=font_size - 1, loc="upper left")
        ax_cal.set_xlabel("prior σ" if i == n_targets - 1 else "", fontsize=font_size)
        ax_cal.set_ylabel("|error|" if i == 0 else "", fontsize=font_size)
        ax_cal.tick_params(labelsize=font_size)
        ax_cal.grid(linestyle="--", alpha=0.3, linewidth=0.4)
        if i == 0:
            ax_cal.set_title("|error| vs. prior σ", fontsize=font_size + 1)

    fig.tight_layout(pad=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Jump stats figure written to: {out_path}")


def _write_mlr_model_log(log_rows: list[dict], output_path: Path) -> None:
    """Write pf_mlr_model_log.csv — one row per target."""
    df_log = pd.DataFrame(log_rows)
    ordered_columns = [
        "target", "process_model_type", "variant_chosen", "mode", "state_offset",
        "selected_features", "n_selected_features", "coefficients", "intercept",
        "r2_train", "rmse_train", "n_train", "n_obs_total",
        "process_noise_std", "process_noise_source", "process_noise_mc_samples",
        "measurement_noise_std", "measurement_noise_source",
        "prior_blend_mode", "prior_blend_auto", "prior_blend",
        "signal_std_train", "residual_std_train",
        "ml_model_type", "ml_output_column", "ml_dataset_dir",
        "ml_forecast_namespace", "n_valid_process_preds",
        "tuning_enabled", "tuned_sigma_Q", "tuned_measurement_noise_std",
        "tuned_prior_blend", "tuning_score_best", "tuning_score_baseline",
        "tuning_rmse_best", "tuning_rmse_baseline", "tuning_n_folds",
        "tuning_n_updates_scored",
        "state_output_column",
        "failure_reason",
    ]
    for col in ordered_columns:
        if col not in df_log.columns:
            df_log[col] = np.nan
    df_log = df_log[[c for c in ordered_columns if c in df_log.columns]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_log.to_csv(output_path, index=False)
    print(f"MLR model log written to: {output_path}")


def _plot_pf_timeseries(
    df: pd.DataFrame,
    pf_results: dict,
    out_path: Path,
) -> None:
    """Generate the particle filter timeseries figure.

    Each subplot shows:
      - Horizontal legal limit lines (red upper, green lower)
      - Shaded ±1σ uncertainty band around the posterior mean
      - Continuous posterior mean line
      - Scatter of actual measured values

    Parameters
    ----------
    out_path : full path (including filename) for the saved PNG.
    """
    limits_csv = _REPO_ROOT / "data" / "input" / "Limits.csv"
    target_cols = [t for t in TARGET_COLUMNS if t in pf_results]
    if not target_cols:
        print("[WARN] No particle filter results to plot.")
        return

    limits_records = load_limits_records(limits_csv) if limits_csv.exists() else []
    all_limit_specs = map_limits_to_columns(target_cols, limits_records)

    ts = pd.to_datetime(df.get("TIMESTAMP"), errors="coerce")

    # Layout constants — mirror plot_mlr_fill_timeseries.py
    fig_width        = 13.0
    row_height       = 0.88
    min_fig_height   = 2.8
    font_size        = 10
    y_label_font_size  = int(round(font_size * 1.5))
    x_label_font_size  = int(round(font_size * 1.5))
    y_value_font_size  = font_size
    y_label_pad        = 8
    shared_hspace      = 0.08
    top_margin_in    = 0.10
    bottom_margin_in = 0.90
    shared_right_margin = 0.995

    raw_labels     = list(target_cols)
    stripped       = _strip_common_affixes(raw_labels)
    wrapped_labels = [textwrap.fill(lbl, width=17) for lbl in stripped]

    global_max_label_line_len = max(
        (max(len(line) for line in lbl.splitlines()) if lbl else 0)
        for lbl in wrapped_labels
    )
    est_char_width_in = max(0.045, 0.0055 * y_label_font_size)
    left_margin_in = global_max_label_line_len * est_char_width_in + 0.55
    shared_left_margin = max(0.14, min(0.26, left_margin_in / fig_width))

    n_rows  = len(target_cols)
    fig_h   = max(min_fig_height, row_height * n_rows)
    palette = _safe_series_colors(n_rows)

    fig, axes_raw = plt.subplots(
        n_rows, 1, sharex=True,
        figsize=(fig_width, fig_h),
        gridspec_kw={"hspace": shared_hspace},
    )
    axes = [axes_raw] if n_rows == 1 else list(axes_raw)

    ts_valid = ts[ts.notna()]
    start_date = ts_valid.min() if len(ts_valid) else pd.Timestamp("2021-12-15")
    end_date   = ts_valid.max() if len(ts_valid) else pd.Timestamp("2025-03-01")
    quarter_ticks = pd.date_range(start=start_date, end=end_date, freq="QS-JAN")

    for i, (ax, col, label) in enumerate(zip(axes, target_cols, wrapped_labels)):
        color    = palette[i]
        res      = pf_results[col]
        mean_arr = res["pf_mean"]
        std_arr  = res["pf_std"]
        observed = pd.to_numeric(df[col], errors="coerce")
        obs_mask = observed.notna() & np.isfinite(observed.values)

        # Limit lines (behind everything)
        if col in all_limit_specs:
            upper_lim, lower_lim = _limit_bounds(all_limit_specs[col])
            if upper_lim is not None:
                ax.axhline(y=upper_lim, color="#ff0000", linewidth=0.7,
                           linestyle="-", zorder=1.2)
            if lower_lim is not None:
                ax.axhline(y=lower_lim, color="#2ca02c", linewidth=0.7,
                           linestyle="-", zorder=1.2)

        finite = np.isfinite(mean_arr) & np.isfinite(std_arr)

        # Break the visual series at each new measurement update so the gap
        # prediction error is easier to see. The updated observation row itself
        # becomes the first point of the next segment.
        segment_mask = finite.copy()
        obs_ilocs = np.flatnonzero(obs_mask.to_numpy(dtype=bool))
        if obs_ilocs.size > 1:
            segment_mask[obs_ilocs[1:]] = False

        seg_start = None
        for row_idx in range(len(segment_mask)):
            if bool(segment_mask[row_idx]):
                if seg_start is None:
                    seg_start = row_idx
                continue
            if seg_start is None:
                continue
            seg_slice = slice(seg_start, row_idx)
            seg_ts = ts.iloc[seg_slice]
            seg_mean = mean_arr[seg_slice]
            seg_std = std_arr[seg_slice]
            if np.isfinite(seg_mean).any() and seg_ts.notna().any():
                ax.fill_between(
                    seg_ts,
                    seg_mean - seg_std,
                    seg_mean + seg_std,
                    alpha=0.25, color=color, zorder=1.3,
                )
                ax.plot(
                    seg_ts, seg_mean,
                    linestyle="-", linewidth=0.7, color=color, alpha=0.8,
                    zorder=1.5,
                )
            seg_start = None
        if seg_start is not None:
            seg_slice = slice(seg_start, len(segment_mask))
            seg_ts = ts.iloc[seg_slice]
            seg_mean = mean_arr[seg_slice]
            seg_std = std_arr[seg_slice]
            if np.isfinite(seg_mean).any() and seg_ts.notna().any():
                ax.fill_between(
                    seg_ts,
                    seg_mean - seg_std,
                    seg_mean + seg_std,
                    alpha=0.25, color=color, zorder=1.3,
                )
                ax.plot(
                    seg_ts, seg_mean,
                    linestyle="-", linewidth=0.7, color=color, alpha=0.8,
                    zorder=1.5,
                )

        # Observed scatter
        if obs_mask.any():
            ax.plot(
                ts[obs_mask], observed.values[obs_mask],
                linestyle="", marker="o", markersize=4.5,
                color=color, markeredgecolor="black", markeredgewidth=0.5,
                alpha=0.95, zorder=2.5,
            )

        # Y-axis range should reflect the measured values only, so extreme
        # predictions do not flatten the useful visual scale around the target.
        measured_vals = observed.values[obs_mask] if obs_mask.any() else np.array([])
        finite_measured = measured_vals[np.isfinite(measured_vals)]
        if finite_measured.size:
            y_low = float(finite_measured.min())
            y_high = float(finite_measured.max())
            y_span = y_high - y_low
            if y_span <= 0:
                y_pad = 0.12 * max(abs(y_low), 1.0)
            else:
                y_pad = max(0.08 * y_span, 0.03 * max(abs(y_low), abs(y_high), 1.0))
            ax.set_ylim(y_low - y_pad, y_high + y_pad)
        elif finite.any():
            all_vals = []
            for arr in (mean_arr[finite], mean_arr[finite] + std_arr[finite], mean_arr[finite] - std_arr[finite]):
                fin = arr[np.isfinite(arr)]
                if fin.size:
                    all_vals.extend([float(fin.min()), float(fin.max())])
            if all_vals:
                y_low, y_high = min(all_vals), max(all_vals)
                y_span = y_high - y_low
                if y_span <= 0:
                    y_pad = 0.12 * max(abs(y_low), 1.0)
                else:
                    y_pad = max(0.08 * y_span, 0.03 * max(abs(y_low), abs(y_high), 1.0))
                ax.set_ylim(y_low - y_pad, y_high + y_pad)

        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontsize=y_label_font_size, labelpad=y_label_pad)
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.4)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
        ax.tick_params(axis="y", labelsize=y_value_font_size)
        ax.tick_params(axis="x", labelsize=x_label_font_size)
        if i < n_rows - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    # X-axis on bottom subplot only
    axes[-1].set_xticks(quarter_ticks)
    axes[-1].set_xticklabels(
        [dt.strftime("%Y-%m-%d") for dt in quarter_ticks],
        rotation=25, ha="right", fontsize=x_label_font_size,
    )
    axes[-1].set_xlim(start_date, end_date)

    shared_top_margin    = max(0.86, min(0.995, 1.0 - (top_margin_in    / fig_h)))
    shared_bottom_margin = max(0.08, min(0.30,  bottom_margin_in / fig_h))
    fig.subplots_adjust(
        left=shared_left_margin, right=shared_right_margin,
        top=shared_top_margin,   bottom=shared_bottom_margin,
        hspace=shared_hspace,
    )

    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=0.7, linestyle="-",
                   alpha=0.8, label="Particle filter mean"),
        mpatches.Patch(facecolor="grey", alpha=0.25, label="±1σ band"),
        plt.Line2D([0], [0], color="black", linestyle="", marker="o",
                   markersize=4.5, markeredgecolor="black", markeredgewidth=0.5,
                   label="Measured value"),
        plt.Line2D([0], [0], color="#ff0000", linewidth=0.7, linestyle="-",
                   label="Legal threshold (upper)"),
        plt.Line2D([0], [0], color="#2ca02c", linewidth=0.7, linestyle="-",
                   label="Legal threshold (lower)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        framealpha=0.85,
        fontsize=font_size,
        borderaxespad=0.12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Figure written to: {out_path}")


# ---------------------------------------------------------------------------
# Clustered bar chart: PF vs. all CV14 methods (summary metrics)
# ---------------------------------------------------------------------------

# Model display names and colours — mirrors z1_FeaturePostProcess.py
_ML_MODEL_TYPES = ['XGB', 'Trans.', 'GP', 'MLR', 'MLR12', 'MLRall']
_ML_COLORS = {
    'GP': 'tab:blue', 'Trans.': 'tab:orange', 'XGB': 'tab:green',
    'MLR': 'tab:red', 'MLR12': 'tab:purple', 'MLRall': 'tab:brown',
}
_PF_DISPLAY = 'PF'
_PF_COLOR   = '#e07b39'
_BASELINE_MODEL_IDS = {'naive', 'seasonal', 'linear'}
_MIN_REQUIRED_VALID = 5


def _normalize_model_display(val: str) -> str | None:
    """Map raw model string to a display key (mirrors z1)."""
    key = str(val).strip().lower()
    if 'xgb' in key:
        return 'XGB'
    if 'transformer' in key:
        return 'Trans.'
    if 'gp' in key:
        return 'GP'
    if key == 'mlr':
        return 'MLR'
    if key == 'mlr_avg12':
        return 'MLR12'
    if key == 'mlr_avgall':
        return 'MLRall'
    if 'mlr' in key:
        return 'MLR'
    return None


def _is_baseline_model(val: str) -> bool:
    return str(val).strip().lower() in _BASELINE_MODEL_IDS


def _filter_sweep_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same validity filters as z1_FeaturePostProcess."""
    out = df.copy()
    if 'std_target' in out.columns:
        out = out[(out['std_target'].notna()) & (out['std_target'] > 0)]
    if 'r2' in out.columns:
        out = out[out['r2'].notna() & np.isfinite(out['r2'])]
    if 'std_target' in out.columns:
        out['nrmse'] = out['rmse'] / out['std_target']
    else:
        out['nrmse'] = np.nan
    # Minimum valid independent test count
    count_col = None
    if 'n_test_valid' in out.columns:
        count_col = 'n_test_valid'
    elif 'n_test_independent' in out.columns:
        count_col = 'n_test_independent'
    if count_col is not None:
        vals = pd.to_numeric(out[count_col], errors='coerce')
        out = out[np.isfinite(vals) & (vals >= _MIN_REQUIRED_VALID)].copy()
    return out


def _plot_model_comparison(
    cv_dir: Path,
    pf_eval_all_targets_path: Path,
    target_cols: list[str],
    output_dir: Path,
) -> None:
    """Four clustered bar charts comparing PF against all CV14 model types.

    Reads ``feature_sweep_final_metrics.csv`` from each target's dataset
    directory in cv_dir (the same source as z1_FeaturePostProcess), applies
    identical filtering and best-per-model-type selection (best R2), then adds
    the particle filter as an extra bar.

    Output files (in output_dir):
      pf_comparison_rmse.png
      pf_comparison_nrmse.png
      pf_comparison_r2.png
      pf_comparison_skill_vs_best.png
    """
    import matplotlib.patches as _mpatches

    # ---- Load PF evaluation rows ----
    if not pf_eval_all_targets_path.exists():
        print(f"[WARN] {pf_eval_all_targets_path} not found; skipping comparison figures.")
        return
    try:
        df_pf_all = pd.read_csv(pf_eval_all_targets_path)
    except Exception as exc:
        print(f"[WARN] Could not read {pf_eval_all_targets_path}: {exc}; skipping comparison figures.")
        return
    df_pf_norm = df_pf_all[df_pf_all["label"] == "particle_filter (res_norm)"].copy()
    if df_pf_norm.empty:
        print("[WARN] No 'particle_filter (res_norm)' rows; skipping comparison figures.")
        return

    # ---- Build comparison records from feature_sweep_final_metrics.csv ----
    records: list[dict] = []

    # Iterate dataset directories directly — each one IS a target
    for d in sorted(cv_dir.iterdir()):
        if not d.is_dir() or d.name == "summaries":
            continue
        fm_csv = d / "forecasts" / "feature_sweeps" / "feature_sweep_final_metrics.csv"
        if not fm_csv.exists():
            continue
        try:
            df_fm = pd.read_csv(fm_csv)
        except Exception:
            continue
        if df_fm.empty:
            continue

        # Identify target_col from the CSV's target column (e.g. "pH_res",
        # "Arsenic__µg_L__res").  Match by checking if sanitising the real
        # target name + "_res" yields the same string.
        csv_target = str(df_fm["target"].iloc[0])
        target_col = None
        for tc in target_cols:
            sanitised = _sanitise_for_dirname(tc + "_res")
            if sanitised == csv_target or sanitised == _sanitise_for_dirname(csv_target):
                target_col = tc
                break
        if target_col is None:
            continue

        # Filter and exclude baselines for ML selection
        valid_df = _filter_sweep_rows(df_fm)
        ml_df = valid_df[~valid_df['model'].apply(_is_baseline_model)].copy()

        # n_samples column
        n_col = 'n_test_valid' if 'n_test_valid' in valid_df.columns else (
            'n_test_independent' if 'n_test_independent' in valid_df.columns else None)

        # Best baseline RMSE from evaluation_summary.csv (same approach as z1)
        eval_csv = d / "evaluation_summary.csv"
        baseline_rmse: dict[str, float] = {}
        if eval_csv.exists():
            try:
                df_eval = pd.read_csv(eval_csv)
                for kind in ('naive', 'seasonal', 'linear'):
                    match = df_eval[df_eval['label'].str.lower().str.contains(kind)]
                    if not match.empty:
                        v = pd.to_numeric(match.iloc[0].get('rmse'), errors='coerce')
                        baseline_rmse[kind] = float(v) if np.isfinite(v) else np.nan
                    else:
                        baseline_rmse[kind] = np.nan
            except Exception:
                pass
        best_baseline_rmse = min(
            (v for v in baseline_rmse.values() if np.isfinite(v) and v > 0),
            default=np.nan,
        )

        # Best row per display model type (by R2)
        ml_df = ml_df.copy()
        ml_df['_display'] = ml_df['model'].apply(_normalize_model_display)

        for model_display in _ML_MODEL_TYPES:
            subset = ml_df[ml_df['_display'] == model_display]
            if subset.empty:
                continue
            best_idx = subset['r2'].idxmax()
            row = subset.loc[best_idx]
            rmse_val  = float(row.get('rmse', np.nan))
            nrmse_val = float(row.get('nrmse', np.nan))
            r2_val    = float(row.get('r2', np.nan))
            n_val     = float(row.get(n_col, np.nan)) if n_col else np.nan

            if np.isfinite(rmse_val) and np.isfinite(best_baseline_rmse) and best_baseline_rmse > 0:
                skill_val = 1.0 - rmse_val / best_baseline_rmse
            else:
                skill_val = np.nan

            records.append({
                'target':        target_col,
                'model_display': model_display,
                'rmse':          rmse_val,
                'nrmse':         nrmse_val,
                'r2':            r2_val,
                'skill_vs_best': skill_val,
                'n_samples':     n_val,
            })

        # ---- PF row for this target ----
        pf_row = df_pf_norm[df_pf_norm['target'] == target_col]
        if not pf_row.empty:
            pf_rmse = float(pf_row['rmse'].iloc[0])
            pf_r2   = float(pf_row['r2'].iloc[0])
            pf_std  = float(pf_row.get('std_target', pd.Series([np.nan])).iloc[0])
            pf_nrmse = pf_rmse / pf_std if np.isfinite(pf_std) and pf_std > 0 else np.nan
            if np.isfinite(pf_rmse) and np.isfinite(best_baseline_rmse) and best_baseline_rmse > 0:
                pf_skill = 1.0 - pf_rmse / best_baseline_rmse
            else:
                pf_skill = np.nan

            records.append({
                'target':        target_col,
                'model_display': _PF_DISPLAY,
                'rmse':          pf_rmse,
                'nrmse':         pf_nrmse,
                'r2':            pf_r2,
                'skill_vs_best': pf_skill,
                'n_samples':     np.nan,
            })

    if not records:
        print("[WARN] No comparison records; skipping comparison figures.")
        return

    comp_df = pd.DataFrame(records)

    # Target ordering: best skill_vs_best descending (matches z1)
    best_skill_per_target = (
        comp_df.groupby('target')['skill_vs_best'].max().sort_values(ascending=False)
    )
    ordered_targets = [
        t for t in best_skill_per_target.index if t in target_cols
    ]
    # Add any targets not yet seen (e.g. all skill=NaN)
    for t in target_cols:
        if t not in ordered_targets:
            ordered_targets.append(t)
    # Filter to targets that actually have data
    ordered_targets = [t for t in ordered_targets if t in comp_df['target'].values]

    # Model ordering: mean skill descending (matches z1)
    all_display_types = _ML_MODEL_TYPES + [_PF_DISPLAY]
    mean_skill_per_model = (
        comp_df.groupby('model_display')['skill_vs_best']
        .mean()
        .reindex(all_display_types)
        .fillna(-np.inf)
        .sort_values(ascending=False)
    )
    ordered_models = list(mean_skill_per_model.index)

    # Colours
    color_map = {**_ML_COLORS, _PF_DISPLAY: _PF_COLOR}

    n_targets = len(ordered_targets)
    n_models  = len(ordered_models)
    x         = np.arange(n_targets, dtype=float)
    width     = min(0.22, 0.85 / max(n_models, 1))
    offsets   = np.array([(i - (n_models - 1) / 2) for i in range(n_models)])
    _FS       = 12

    # Stripped target labels for x-axis
    stripped_targets = _strip_common_affixes(ordered_targets)

    metric_specs = [
        ('rmse',          'RMSE',                    '.2e', False),
        ('nrmse',         'nRMSE',                   '.2e', False),
        ('r2',            'R2',                       '.2f', False),
        ('skill_vs_best', 'Skill vs. Best Baseline', '.2f', True),
    ]
    file_names = {
        'rmse':          'pf_comparison_rmse.png',
        'nrmse':         'pf_comparison_nrmse.png',
        'r2':            'pf_comparison_r2.png',
        'skill_vs_best': 'pf_comparison_skill_vs_best.png',
    }

    for metric_key, ylabel, fmt, add_hline in metric_specs:
        fig, ax = plt.subplots(figsize=(max(10, n_targets * (n_models * 0.35 + 0.5)), 6))

        # Build legend handles upfront so all models always appear
        legend_handles = [
            _mpatches.Patch(facecolor=color_map.get(m, '#888888'), label=m)
            for m in ordered_models
        ]

        pending_annotations: list[tuple] = []

        for mi, model_display in enumerate(ordered_models):
            color = color_map.get(model_display, '#888888')
            vals = []
            ns   = []
            bar_x = []

            for ti, tgt in enumerate(ordered_targets):
                row_match = comp_df[
                    (comp_df['target'] == tgt) & (comp_df['model_display'] == model_display)
                ]
                if row_match.empty:
                    continue
                v = float(row_match.iloc[0][metric_key])
                if not np.isfinite(v):
                    continue
                bar_x.append(x[ti] + offsets[mi] * width)
                vals.append(v)
                ns.append(float(row_match.iloc[0].get('n_samples', np.nan)))

            if not bar_x:
                continue

            bars = ax.bar(bar_x, vals, width, color=color)
            pending_annotations.append((bars, vals, ns, fmt))

        if add_hline:
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

        if metric_key in {'r2', 'skill_vs_best'}:
            ax.set_ylim(-1.0, 1.0)
        else:
            ymin_cur, ymax_cur = ax.get_ylim()
            all_metric_vals = comp_df[metric_key].dropna().to_numpy(dtype=float)
            all_metric_vals = all_metric_vals[np.isfinite(all_metric_vals)]
            min_val = float(np.min(all_metric_vals)) if all_metric_vals.size else 0.0
            if min_val >= 0.0:
                ax.set_ylim(bottom=0.0)
            else:
                ax.set_ylim(bottom=max(ymin_cur, -1.0))

        # Annotate bars with value and sample count
        for bars, vals, ns, fmt_str in pending_annotations:
            ymin_ax, ymax_ax = ax.get_ylim()
            for bar, val, n in zip(bars, vals, ns):
                h = bar.get_height()
                bar_top = bar.get_y() + h
                if not (ymin_ax <= bar_top <= ymax_ax):
                    continue
                n_str = f", n={int(n)}" if np.isfinite(n) else ""
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar_top + 0.008 * (ymax_ax - ymin_ax),
                    f"{val:{fmt_str}}{n_str}",
                    ha='center', va='bottom', fontsize=7,
                    rotation=90, clip_on=True,
                )

        # Tight horizontal bounds
        cluster_half = (n_models - 1) / 2 * width + width / 2
        if n_targets > 0:
            ax.set_xlim(x[0] - cluster_half - 0.07, x[-1] + cluster_half + 0.07)

        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.tick_params(axis='y', labelsize=_FS)
        ax.set_xticks(x)
        ax.set_xticklabels(stripped_targets, rotation=45, ha='right', fontsize=_FS)
        ax.grid(axis='y', alpha=0.3)

        ax.legend(
            handles=legend_handles,
            loc='lower center', bbox_to_anchor=(0.5, 1.02),
            ncol=len(ordered_models), frameon=False, fontsize=_FS,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out_path = output_dir / file_names[metric_key]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"Comparison figure written to: {out_path}")


# ---------------------------------------------------------------------------
# Comparison figure: PF vs. existing model predictions
# ---------------------------------------------------------------------------

def _plot_comparison_timeseries(
    df: pd.DataFrame,
    pf_results: dict,
    predictions_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Overlay particle filter and existing model predictions on a shared figure.

    For each target one subplot shows:
      - Horizontal legal limit lines (behind data)
      - Existing model reconstruction (``<target>_reconstructed``) as a thin
        dashed line in a neutral grey-brown
      - Particle filter ±1σ shaded band
      - Particle filter posterior mean as a solid coloured line
      - Scatter of actual measured values

    The two DataFrames are joined on TIMESTAMP so that rows align correctly
    even if the two CSVs have different lengths or row orderings.

    Parameters
    ----------
    df : the PF input DataFrame (Consolidated_sparse.csv), must have TIMESTAMP.
    pf_results : dict {target_col: {"pf_mean": ndarray, "pf_std": ndarray}}.
    predictions_df : DataFrame loaded from Consolidated_predictions.csv (or
        similar); must have TIMESTAMP and ``<target>_reconstructed`` columns.
    out_path : full path (including filename) for the saved PNG.
    """
    target_cols = [t for t in TARGET_COLUMNS if t in pf_results]
    if not target_cols:
        print("[WARN] No particle filter results to plot for comparison figure.")
        return

    limits_csv = _REPO_ROOT / "data" / "input" / "Limits.csv"
    limits_records = load_limits_records(limits_csv) if limits_csv.exists() else []
    all_limit_specs = map_limits_to_columns(target_cols, limits_records)

    # Join the two DataFrames on TIMESTAMP so arrays align
    ts_pf   = pd.to_datetime(df.get("TIMESTAMP"), errors="coerce")
    ts_pred = pd.to_datetime(predictions_df.get("TIMESTAMP"), errors="coerce")

    # Build a merged index: for each target we need the reconstructed column
    # aligned to the same rows as the PF output.
    # Strategy: use TIMESTAMP as a join key; fall back to positional if no TIMESTAMP.
    use_ts_join = ts_pf.notna().any() and ts_pred.notna().any()
    if use_ts_join:
        df_pf_ts   = df.copy(); df_pf_ts["_ts"] = ts_pf
        df_pred_ts = predictions_df.copy(); df_pred_ts["_ts"] = ts_pred
        merged = df_pf_ts.merge(df_pred_ts, on="_ts", how="left", suffixes=("", "_z2"))
        ts_merged = pd.to_datetime(merged["_ts"], errors="coerce")
    else:
        # Positional fallback
        merged = df.copy()
        n_pad = len(df) - len(predictions_df)
        if n_pad > 0:
            pad = pd.DataFrame(np.nan, index=range(n_pad), columns=predictions_df.columns)
            predictions_df = pd.concat([predictions_df, pad], ignore_index=True)
        for col in predictions_df.columns:
            if col not in merged.columns:
                merged[col] = predictions_df[col].values[: len(df)]
        ts_merged = ts_pf

    # Layout constants — mirror _plot_pf_timeseries
    fig_width          = 13.0
    row_height         = 0.88
    min_fig_height     = 2.8
    font_size          = 10
    y_label_font_size  = int(round(font_size * 1.5))
    x_label_font_size  = int(round(font_size * 1.5))
    y_value_font_size  = font_size
    y_label_pad        = 8
    shared_hspace      = 0.08
    top_margin_in      = 0.10
    bottom_margin_in   = 0.90
    shared_right_margin = 0.995

    raw_labels     = list(target_cols)
    stripped       = _strip_common_affixes(raw_labels)
    wrapped_labels = [textwrap.fill(lbl, width=17) for lbl in stripped]

    global_max_label_line_len = max(
        (max(len(line) for line in lbl.splitlines()) if lbl else 0)
        for lbl in wrapped_labels
    )
    est_char_width_in = max(0.045, 0.0055 * y_label_font_size)
    left_margin_in    = global_max_label_line_len * est_char_width_in + 0.55
    shared_left_margin = max(0.14, min(0.26, left_margin_in / fig_width))

    n_rows  = len(target_cols)
    fig_h   = max(min_fig_height, row_height * n_rows)
    palette = _safe_series_colors(n_rows)

    fig, axes_raw = plt.subplots(
        n_rows, 1, sharex=True,
        figsize=(fig_width, fig_h),
        gridspec_kw={"hspace": shared_hspace},
    )
    axes = [axes_raw] if n_rows == 1 else list(axes_raw)

    ts_valid   = ts_merged[ts_merged.notna()]
    start_date = ts_valid.min() if len(ts_valid) else pd.Timestamp("2021-12-15")
    end_date   = ts_valid.max() if len(ts_valid) else pd.Timestamp("2025-03-01")
    quarter_ticks = pd.date_range(start=start_date, end=end_date, freq="QS-JAN")

    # Colour for existing-model line — neutral to avoid clashing with targets
    _existing_color = "#7f7f7f"

    for i, (ax, col, label) in enumerate(zip(axes, target_cols, wrapped_labels)):
        color    = palette[i]
        res      = pf_results[col]
        mean_arr = res["pf_mean"]
        std_arr  = res["pf_std"]

        # Limit lines (behind everything)
        if col in all_limit_specs:
            upper_lim, lower_lim = _limit_bounds(all_limit_specs[col])
            if upper_lim is not None:
                ax.axhline(y=upper_lim, color="#ff0000", linewidth=0.7,
                           linestyle="-", zorder=1.0)
            if lower_lim is not None:
                ax.axhline(y=lower_lim, color="#2ca02c", linewidth=0.7,
                           linestyle="-", zorder=1.0)

        finite_pf = np.isfinite(mean_arr) & np.isfinite(std_arr)

        # --- Existing model reconstruction (dashed, behind PF) ---
        recon_col = f"{col}_reconstructed"
        if recon_col in merged.columns:
            recon = pd.to_numeric(merged[recon_col], errors="coerce").values
            recon_finite = np.isfinite(recon)
            if recon_finite.any():
                ax.plot(
                    ts_merged[recon_finite], recon[recon_finite],
                    linestyle="--", linewidth=0.9, color=_existing_color,
                    alpha=0.75, zorder=1.4, label="Existing model",
                )

        # --- PF ±1σ band ---
        if finite_pf.any():
            ax.fill_between(
                ts_merged[finite_pf],
                mean_arr[finite_pf] - std_arr[finite_pf],
                mean_arr[finite_pf] + std_arr[finite_pf],
                alpha=0.20, color=color, zorder=1.5,
            )

        # --- PF posterior mean ---
        if finite_pf.any():
            ax.plot(
                ts_merged[finite_pf], mean_arr[finite_pf],
                linestyle="-", linewidth=0.85, color=color, alpha=0.9,
                zorder=1.6,
            )

        # --- Observed scatter ---
        observed = pd.to_numeric(df[col], errors="coerce")
        obs_mask = observed.notna() & np.isfinite(observed.values)
        if obs_mask.any():
            ax.plot(
                ts_pf[obs_mask], observed.values[obs_mask],
                linestyle="", marker="o", markersize=4.5,
                color=color, markeredgecolor="black", markeredgewidth=0.5,
                alpha=0.95, zorder=2.5,
            )

        # Y-axis range with padding — include both PF and existing-model extents
        all_vals: list[float] = []
        for arr in (
            mean_arr[finite_pf],
            mean_arr[finite_pf] + std_arr[finite_pf],
            mean_arr[finite_pf] - std_arr[finite_pf],
            observed.values[obs_mask] if obs_mask.any() else np.array([]),
        ):
            fin = arr[np.isfinite(arr)]
            if fin.size:
                all_vals.extend([float(fin.min()), float(fin.max())])
        if recon_col in merged.columns:
            recon_fin = recon[np.isfinite(recon)]
            if recon_fin.size:
                all_vals.extend([float(recon_fin.min()), float(recon_fin.max())])
        if all_vals:
            y_low, y_high = min(all_vals), max(all_vals)
            y_span = y_high - y_low
            if y_span <= 0:
                y_pad = 0.12 * max(abs(y_low), 1.0)
            else:
                y_pad = max(0.08 * y_span, 0.03 * max(abs(y_low), abs(y_high), 1.0))
            ax.set_ylim(y_low - y_pad, y_high + y_pad)

        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontsize=y_label_font_size, labelpad=y_label_pad)
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.4)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
        ax.tick_params(axis="y", labelsize=y_value_font_size)
        ax.tick_params(axis="x", labelsize=x_label_font_size)
        if i < n_rows - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    # X-axis ticks on bottom subplot only
    axes[-1].set_xticks(quarter_ticks)
    axes[-1].set_xticklabels(
        [dt.strftime("%Y-%m-%d") for dt in quarter_ticks],
        rotation=25, ha="right", fontsize=x_label_font_size,
    )
    axes[-1].set_xlim(start_date, end_date)

    shared_top_margin    = max(0.86, min(0.995, 1.0 - (top_margin_in    / fig_h)))
    shared_bottom_margin = max(0.08, min(0.30,  bottom_margin_in / fig_h))
    fig.subplots_adjust(
        left=shared_left_margin, right=shared_right_margin,
        top=shared_top_margin,   bottom=shared_bottom_margin,
        hspace=shared_hspace,
    )

    legend_handles = [
        plt.Line2D([0], [0], color=_existing_color, linewidth=0.9,
                   linestyle="--", alpha=0.75, label="Existing model (best)"),
        plt.Line2D([0], [0], color="black", linewidth=0.85, linestyle="-",
                   alpha=0.9, label="Particle filter mean"),
        mpatches.Patch(facecolor="grey", alpha=0.20, label="PF ±1σ band"),
        plt.Line2D([0], [0], color="black", linestyle="", marker="o",
                   markersize=4.5, markeredgecolor="black", markeredgewidth=0.5,
                   label="Measured value"),
        plt.Line2D([0], [0], color="#ff0000", linewidth=0.7, linestyle="-",
                   label="Legal threshold (upper)"),
        plt.Line2D([0], [0], color="#2ca02c", linewidth=0.7, linestyle="-",
                   label="Legal threshold (lower)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        framealpha=0.85,
        fontsize=font_size,
        borderaxespad=0.12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Comparison figure written to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Particle filter state estimator for water quality targets."
    )
    parser.add_argument(
        "--input", default=_DEFAULT_INPUT,
        help="Path to Consolidated_sparse.csv "
             f"(default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir", default=_DEFAULT_OUTPUT_DIR,
        help="Directory for output files "
             f"(default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--n-particles", type=int, default=500,
        help="Number of particles (default: 500)",
    )
    parser.add_argument(
        "--process-model", default="mlr",
        choices=["mlr", "best_ml_gap_residual", "best_mlr_gap_residual"],
        help="Process-model family to use inside the particle filter "
             "(default: mlr).",
    )
    parser.add_argument(
        "--process-noise", type=float, default=None,
        metavar="SIGMA_Q",
        help="Override process noise std σ_Q for all targets "
        "(default: inferred from calibrated profiler predictor uncertainty, "
             "falling back to train residual spread)",
    )
    parser.add_argument(
        "--measurement-noise", type=float, default=None,
        metavar="SIGMA_R",
        help="Override measurement noise std √R for all targets "
             "(default: inferred from train-split lab-vs-model residuals)",
        )
    parser.add_argument(
        "--prior-blend", type=float, default=None,
        metavar="W",
        help="Weight in [0, 1] pulling particles toward the MLR prior at each step. "
             "0 keeps a pure carried-forward state; 1 resets fully to the MLR prior. "
             "Default: conservative auto value inferred from train-only MLR skill and residual noise.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--overwrite-state-output", action="store_true",
        help="Also overwrite each <target>_state column in the output CSV with the "
             "PF state estimate. By default the original _state columns are preserved "
             "and the PF state is written to <target>_state_pf.",
    )
    parser.add_argument(
        "--targets", nargs="+", default=None,
        metavar="TARGET",
        help="One or more target column names to process.  Must match the exact "
             "column names in the input CSV (quotes required when names contain "
             "spaces or special characters).  Default: all 14 standard targets.  "
             "Example: --targets \"pH\" \"Turbidity (FNU)\"",
    )
    parser.add_argument(
        "--run-name", default=None,
        metavar="NAME",
        help="Create a subdirectory data/output/<NAME>/ and write all outputs "
             "there.  Prevents overwriting previous runs.  "
             "E.g. --run-name test01 writes to data/output/test01/.  "
             "Default: outputs go directly into --output-dir.",
    )
    parser.add_argument(
        "--cv-dir", default=None,
        metavar="PATH",
        help="Path to a CV output directory (e.g. data/output/CV14).  "
             "When supplied, the script locates each target's normalization.json "
             "there, produces an evaluation_summary.csv in the run-name output "
             "directory (in the same 26-column format as f_Evaluate.py), and "
             "writes jump-stat figures.  Metrics are produced in both absolute "
             "physical units and normalised _res units for direct comparison with "
             "existing pipeline results.",
    )
    parser.add_argument(
        "--predictions-csv", default=None,
        metavar="PATH",
        help="Path to a Consolidated_predictions.csv produced by "
             "z2_PredictionTimeseries.py (or a similar file with "
             "<target>_reconstructed columns).  When supplied, a second "
             "timeseries figure is written that overlays the particle filter "
             "mean and ±1σ band on top of the existing model's reconstructed "
             "timeseries, one subplot per target.  "
             "Output: Target_timeseries_pf_vs_model.png in the output directory.  "
             "E.g. --predictions-csv data/output/CV14/summaries/predictions/"
             "Consolidated_predictions.csv",
    )
    parser.add_argument(
        "--no-figure", action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--tune-filter", action="store_true",
        help="Tune sigma_Q, measurement noise, and prior_blend per target on training observations only.",
    )
    parser.add_argument(
        "--tune-folds", type=int, default=3,
        help="Number of sequential train-only validation folds used when --tune-filter is enabled (default: 3).",
    )
    parser.add_argument(
        "--tune-grid-scale", type=int, default=1,
        help="Coarse search size for --tune-filter. 1 = default compact search, >1 = wider search (default: 1).",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Raise ValueError for invalid CLI argument combinations/values."""
    if int(args.n_particles) < 2:
        raise ValueError("--n-particles must be >= 2.")
    if args.process_noise is not None and float(args.process_noise) < 0:
        raise ValueError("--process-noise must be >= 0.")
    if args.measurement_noise is not None and float(args.measurement_noise) < 0:
        raise ValueError("--measurement-noise must be >= 0.")
    if args.prior_blend is not None and not (0.0 <= float(args.prior_blend) <= 1.0):
        raise ValueError("--prior-blend must be between 0 and 1.")
    if str(args.process_model) in {"best_ml_gap_residual", "best_mlr_gap_residual"} and not args.cv_dir:
        raise ValueError(
            "--cv-dir is required when --process-model best_ml_gap_residual "
            "or best_mlr_gap_residual is used."
        )
    if int(args.tune_folds) < 1:
        raise ValueError("--tune-folds must be >= 1.")
    if int(args.tune_grid_scale) < 1:
        raise ValueError("--tune-grid-scale must be >= 1.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)

    args = build_parser().parse_args()
    _validate_args(args)
    rng = np.random.default_rng(args.seed)

    # When --run-name is given, outputs go into data/output/<run-name>/.
    # Without it, outputs go directly into --output-dir (default: data/output/regression/).
    base_output_dir = Path(args.output_dir)
    if args.run_name:
        output_dir = base_output_dir.parent / args.run_name
    else:
        output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, parse_dates=["TIMESTAMP"], low_memory=False)
    df = df.reset_index(drop=True)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    active_cols = _active_predictor_columns()
    print(f"  Active predictor columns: {len(active_cols)}")

    # Resolve the set of targets to process
    if args.targets is not None:
        unknown = [t for t in args.targets if t not in TARGET_COLUMNS]
        if unknown:
            print(
                f"[WARN] The following --targets are not in the standard TARGET_COLUMNS "
                f"list and will be skipped if absent from the CSV: {unknown}"
            )
        targets_to_run = [t for t in TARGET_COLUMNS if t in args.targets] + [
            t for t in args.targets if t not in TARGET_COLUMNS
        ]
        print(f"  Targets selected: {targets_to_run}")
    else:
        targets_to_run = list(TARGET_COLUMNS)
        print(f"  Targets: all {len(targets_to_run)} standard targets")

    log_rows: list[dict] = []
    tuning_detail_rows: list[dict] = []
    # Store (proc_model, obs_model) per target for Phase 2
    model_store: dict[str, dict] = {}
    holdout_ilocs_by_target: dict[str, set[int]] = {}

    # ------------------------------------------------------------------
    # Phase 1: build process/observation models per target
    # ------------------------------------------------------------------
    print(f"\n--- Phase 1: Process-model setup ({args.process_model}) ---")
    for target_col in targets_to_run:
        if target_col not in df.columns:
            print(f"[WARN] '{target_col}' not in CSV, skipping.")
            continue

        if args.process_model == "mlr":
            print(f"\n[{target_col}] Fitting MLR ...")
            fit_result = fit_mlr_for_target(df, target_col, active_cols, rng)
        elif args.process_model == "best_ml_gap_residual":
            print(f"\n[{target_col}] Loading best saved ML gap-residual model ...")
            fit_result = build_ml_gap_process_model_for_target(
                df=df,
                target_col=target_col,
                cv_dir=Path(args.cv_dir),
                process_noise_override=args.process_noise,
            )
        else:
            print(f"\n[{target_col}] Loading best saved MLR gap-residual model ...")
            fit_result = build_mlr_gap_process_model_for_target(
                df=df,
                target_col=target_col,
                cv_dir=Path(args.cv_dir),
                process_noise_override=args.process_noise,
            )

        if fit_result is None:
            print(f"  [SKIP] Insufficient observations (< {MIN_OBSERVED}).")
            continue

        meta = fit_result["meta"]

        # Apply CLI noise overrides
        proc_noise_std = (
            float(args.process_noise)
            if args.process_noise is not None
            else fit_result["process_noise_std"]
        )
        process_noise_source = (
            "cli_override"
            if args.process_noise is not None
            else fit_result["process_noise_source"]
        )
        meas_noise_var = (
            float(args.measurement_noise) ** 2
            if args.measurement_noise is not None
            else fit_result["measurement_noise_var"]
        )
        measurement_noise_source = (
            "cli_override"
            if args.measurement_noise is not None
            else fit_result["measurement_noise_source"]
        )
        prior_blend = (
            float(args.prior_blend)
            if args.prior_blend is not None
            else float(fit_result["prior_blend_auto"])
        )
        prior_blend_mode = "manual" if args.prior_blend is not None else "auto"
        tuning_meta = {
            "tuning_enabled": False,
            "tuned_sigma_Q": np.nan,
            "tuned_measurement_noise_std": np.nan,
            "tuned_prior_blend": np.nan,
            "tuning_score_best": np.nan,
            "tuning_score_baseline": np.nan,
            "tuning_rmse_best": np.nan,
            "tuning_rmse_baseline": np.nan,
            "tuning_n_folds": 0,
            "tuning_n_updates_scored": 0,
        }

        print(
            f"  Variant: {fit_result['variant_name']}, "
            f"R²={fit_result['r2']:.3f}, RMSE={fit_result['rmse']:.4g}  |  "
            f"sigma_Q={proc_noise_std:.4g}, sqrt_R={np.sqrt(meas_noise_var):.4g}, "
            f"prior_blend={prior_blend:.3f}  |  "
            f"features: {meta['selected_features']}"
        )
        if len(fit_result.get("holdout_ilocs", [])) < 5:
            print("  [WARN] Fewer than 5 holdout observations; evaluation metrics may be unstable.")
        if not meta.get("selected_features"):
            print("  [WARN] Intercept-only MLR fallback selected for this target.")

        if args.process_model == "mlr":
            # Pre-compute MLR predictions at every row for a fast filter loop
            print(f"  Pre-computing MLR predictions at all {len(df)} rows ...")
            mlr_preds_all = _precompute_mlr_predictions(df, target_col, fit_result, active_cols)
            proc_model = MLRProcessModel(
                df=df,
                target_col=target_col,
                fit_result=fit_result,
                noise_std_override=proc_noise_std,
                precomputed_preds=mlr_preds_all,
            )
        else:
            proc_model = fit_result["proc_model"]

        if args.tune_filter:
            obs_ilocs_all = np.where(np.isfinite(pd.to_numeric(df[target_col], errors="coerce").values))[0]
            tuned, detail_rows = tune_filter_hyperparameters_for_target(
                df=df,
                target_col=target_col,
                process_model=proc_model,
                default_sigma_q=proc_noise_std,
                default_measurement_noise_var=meas_noise_var,
                default_prior_blend=prior_blend,
                n_particles=args.n_particles,
                seed=args.seed,
                train_ilocs=fit_result.get("train_ilocs", []),
                all_obs_ilocs=obs_ilocs_all,
                n_folds=args.tune_folds,
                grid_scale=args.tune_grid_scale,
            )
            tuning_detail_rows.extend(detail_rows)
            if tuned.get("tuning_enabled"):
                proc_noise_std = float(tuned["sigma_q"])
                meas_noise_var = float(tuned["measurement_noise_var"])
                prior_blend = float(tuned["prior_blend"])
                prior_blend_mode = "tuned"
                proc_model = NoiseOverrideProcessModel(proc_model, proc_noise_std)
                tuning_meta = {
                    "tuning_enabled": True,
                    "tuned_sigma_Q": proc_noise_std,
                    "tuned_measurement_noise_std": float(np.sqrt(meas_noise_var)),
                    "tuned_prior_blend": prior_blend,
                    "tuning_score_best": float(tuned["score"]),
                    "tuning_score_baseline": float(tuned["baseline_score"]),
                    "tuning_rmse_best": float(tuned["rmse"]),
                    "tuning_rmse_baseline": float(tuned["baseline_rmse"]),
                    "tuning_n_folds": int(tuned["tuning_n_folds"]),
                    "tuning_n_updates_scored": int(tuned["n_scored"]),
                }
                print(
                    f"  Tuned -> sigma_Q={proc_noise_std:.4g}, sqrt_R={np.sqrt(meas_noise_var):.4g}, "
                    f"prior_blend={prior_blend:.3f}, RMSE={tuned['rmse']:.4g} "
                    f"(baseline {tuned['baseline_rmse']:.4g})"
                )
            else:
                print("  [WARN] Tuning skipped: insufficient scored training updates.")

        # Collect artifact log row
        log_rows.append({
            "target":               target_col,
            "process_model_type":   fit_result.get("process_model_type", args.process_model),
            "variant_chosen":       fit_result["variant_name"],
            "mode":                 fit_result["mode"],
            "state_offset":         fit_result["state_offset"],
            "selected_features":    ";".join(meta["selected_features"]),
            "n_selected_features":  len(meta["selected_features"]),
            "coefficients":         ";".join(str(c) for c in meta["coefficients"]),
            "intercept":            meta["intercept"],
            "r2_train":             fit_result["r2"],
            "rmse_train":           fit_result["rmse"],
            "n_train":              fit_result["n_train"],
            "n_obs_total":          fit_result["n_obs_total"],
            "process_noise_std":    proc_noise_std,
            "process_noise_source": process_noise_source,
            "process_noise_mc_samples": fit_result.get("process_noise_mc_samples", np.nan),
            "measurement_noise_std": np.sqrt(meas_noise_var),
            "measurement_noise_source": measurement_noise_source,
            "prior_blend_mode":     prior_blend_mode,
            "prior_blend_auto":     fit_result.get("prior_blend_auto", np.nan),
            "prior_blend":          prior_blend,
            "signal_std_train":     fit_result.get("signal_std_train", np.nan),
            "residual_std_train":   fit_result.get("residual_std_train", np.nan),
            "ml_model_type":        fit_result.get("ml_model_type", ""),
            "ml_output_column":     fit_result.get("ml_output_column", ""),
            "ml_dataset_dir":       fit_result.get("ml_dataset_dir", ""),
            "ml_forecast_namespace": fit_result.get("ml_forecast_namespace", ""),
            "n_valid_process_preds": fit_result.get("n_valid_process_preds", np.nan),
            **tuning_meta,
            "state_output_column":  (
                f"{target_col}_state and {target_col}_state_pf"
                if args.overwrite_state_output
                else f"{target_col}_state_pf"
            ),
            "failure_reason":       meta.get("failure_reason") or "",
        })
        obs_model = GaussianObservationModel(measurement_noise_var=meas_noise_var)

        model_store[target_col] = {
            "proc_model": proc_model,
            "obs_model":  obs_model,
            "prior_blend": prior_blend,
        }
        holdout_ilocs_by_target[target_col] = {
            int(i) for i in fit_result.get("holdout_ilocs", [])
        }

    # Write model log before running the (potentially slow) filter
    log_path = output_dir / "pf_mlr_model_log.csv"
    _write_mlr_model_log(log_rows, log_path)
    if args.tune_filter and tuning_detail_rows:
        tuning_path = output_dir / "pf_tuning_detail.csv"
        pd.DataFrame(tuning_detail_rows).to_csv(tuning_path, index=False)
        print(f"Tuning detail written to: {tuning_path}")

    # ------------------------------------------------------------------
    # Phase 2: run particle filter per target
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Particle filter ---")
    pf_results: dict[str, dict] = {}
    all_update_records: dict[str, list[dict]] = {}

    for target_col, models in model_store.items():
        print(f"\n[{target_col}] Running filter ({args.n_particles} particles) ...")
        pf_mean, pf_std, update_records = run_particle_filter(
            df=df,
            target_col=target_col,
            process_model=models["proc_model"],
            obs_model=models["obs_model"],
            n_particles=args.n_particles,
            prior_blend=models["prior_blend"],
            rng=rng,
        )
        finite_count = int(np.isfinite(pf_mean).sum())
        jump = _compute_jump_stats(update_records)
        print(
            f"  Done - {finite_count}/{len(df)} rows have finite estimates  "
            f"(mean range: [{np.nanmin(pf_mean):.4g}, {np.nanmax(pf_mean):.4g}])  |  "
            f"jump MAE={jump.get('mae', np.nan):.4g}, "
            f"RMSE={jump.get('rmse', np.nan):.4g}, "
            f"within 1sigma={jump.get('frac_within_1sigma', np.nan):.1%}"
        )
        pf_results[target_col] = {"pf_mean": pf_mean, "pf_std": pf_std}
        all_update_records[target_col] = update_records

    # ------------------------------------------------------------------
    # Phase 3: write output CSV
    # ------------------------------------------------------------------
    df_out = df.copy()
    for target_col, res in pf_results.items():
        state_col = f"{target_col}_state"
        state_pf_col = f"{target_col}_state_pf"
        df_out[f"{target_col}_pf_mean"] = res["pf_mean"]
        df_out[f"{target_col}_pf_std"]  = res["pf_std"]
        pf_vals = np.asarray(res["pf_mean"], dtype=float)
        df_out[state_pf_col] = pf_vals
        if args.overwrite_state_output and state_col in df_out.columns:
            state_vals = pd.to_numeric(df_out[state_col], errors="coerce").to_numpy(dtype=float, copy=True)
            fill_mask = np.isfinite(pf_vals)
            state_vals[fill_mask] = pf_vals[fill_mask]
            df_out[state_col] = state_vals

    out_csv = output_dir / "Consolidated_particle_filter.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nParticle filter CSV written to: {out_csv}")

    # ------------------------------------------------------------------
    # Phase 4: jump statistics CSVs
    # ------------------------------------------------------------------

    # Per-update detail: one row per observation update per target
    detail_rows = []
    for target_col, records in all_update_records.items():
        for rec in records:
            detail_rows.append({"target": target_col, **rec})
    detail_path = output_dir / "pf_jump_detail.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    print(f"Jump detail written to: {detail_path}")

    # Per-target summary: one row per target
    summary_rows = []
    for target_col, records in all_update_records.items():
        stats = _compute_jump_stats(records)
        summary_rows.append({"target": target_col, **stats})
    summary_path = output_dir / "pf_jump_stats.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Jump summary written to: {summary_path}")

    # ------------------------------------------------------------------
    # Phase 5: evaluation_summary.csv (when --cv-dir supplied)
    # ------------------------------------------------------------------
    cv_dir = Path(args.cv_dir) if args.cv_dir else None

    if cv_dir is not None:
        print(f"\n--- Phase 5: Evaluation summary (cv-dir: {cv_dir}) ---")
        eval_rows_all: list[dict] = []

        for target_col, records in all_update_records.items():
            norm_bounds = _find_normalization(cv_dir, target_col)
            if not norm_bounds:
                print(f"  [WARN] normalization.json not found for '{target_col}'; "
                      f"skipping evaluation summary rows for this target.")
                continue

            rows = _compute_evaluation_rows(
                target_col,
                records,
                df,
                norm_bounds,
                eval_row_indices=holdout_ilocs_by_target.get(target_col, set()),
            )
            if not rows:
                print(f"  [WARN] No evaluation rows available for '{target_col}'; skipping PF forecast artifact export.")
                continue
            for row in rows:
                row["data_dir"] = str(cv_dir)
            eval_rows_all.extend(rows)

            # Also write a per-target evaluation_summary.csv for easy diff
            per_target_path = output_dir / f"evaluation_summary_{target_col}.csv".replace(
                "/", "_").replace(" ", "_")
            _write_evaluation_summary(rows, per_target_path)

            forecast_dir = _write_pf_forecast_artifacts(
                cv_dir=cv_dir,
                target_col=target_col,
                df=df,
                update_records=records,
                eval_rows=rows,
                eval_row_indices=holdout_ilocs_by_target.get(target_col, set()),
                process_model=args.process_model,
                run_name=args.run_name,
                tuned=args.tune_filter,
                make_plots=not args.no_figure,
            )
            if forecast_dir is not None:
                print(f"  PF forecast artifacts written under: {forecast_dir}")

        # Combined file covering all processed targets
        combined_eval_path = output_dir / "evaluation_summary_all_targets.csv"
        _write_evaluation_summary(eval_rows_all, combined_eval_path)
        print(f"Combined evaluation summary written to: {combined_eval_path}")

        if not args.no_figure:
            print("\nGenerating model comparison figures ...")
            _plot_model_comparison(
                cv_dir=cv_dir,
                pf_eval_all_targets_path=combined_eval_path,
                target_cols=list(all_update_records.keys()),
                output_dir=output_dir,
            )

    # ------------------------------------------------------------------
    # Phase 6: figures
    # ------------------------------------------------------------------
    if not args.no_figure:
        print("\nGenerating timeseries figure ...")
        fig_path = output_dir / "Target_timeseries_pf.png"
        _plot_pf_timeseries(df, pf_results, fig_path)

        print("Generating jump stats figure ...")
        jump_fig_path = output_dir / "pf_jump_stats.png"
        _plot_jump_stats(all_update_records, jump_fig_path)

        if args.predictions_csv is not None:
            pred_csv_path = Path(args.predictions_csv)
            if pred_csv_path.exists():
                print(f"\nLoading predictions CSV: {pred_csv_path} ...")
                predictions_df = pd.read_csv(
                    pred_csv_path, parse_dates=["TIMESTAMP"], low_memory=False
                )
                print(f"  {len(predictions_df)} rows, {len(predictions_df.columns)} columns")
                print("Generating comparison figure ...")
                cmp_fig_path = output_dir / "Target_timeseries_pf_vs_model.png"
                _plot_comparison_timeseries(df, pf_results, predictions_df, cmp_fig_path)
            else:
                print(f"[WARN] --predictions-csv path not found: {pred_csv_path}; "
                      f"skipping comparison figure.")


if __name__ == "__main__":
    main()

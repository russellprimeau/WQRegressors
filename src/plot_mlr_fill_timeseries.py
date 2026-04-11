"""Plot MLR gap-fill timeseries for each target column.

Generates data/output/regression/Target_timeseries_mlr_fill.png —
a variant of the standard Target_timeseries figure that overlays:
  - A continuous line of the <target>_state MLR estimates (filled + observed)
  - Scatter points for the measured (observed) target values on top

At rows where measured values exist the estimate line shows MLR predictions
(not the stored _state values), so the line is fully continuous and unbiased
by the measured data.  These predictions are computed in memory only and are
never written to disk.

Layout and styling reuse helpers from b_ExploreData.py.
"""

import sys
import os
import re
import textwrap
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# Make fill_gaps_mlr importable (it lives in the same src/ directory)
sys.path.insert(0, os.path.dirname(__file__))

from utils.limits import load_limits_records, map_limits_to_columns, limit_exceedance_mask
from fill_gaps_mlr import (
    TARGET_COLUMNS,
    VARIANTS,
    TRAIN_FRACTION,
    MIN_OBSERVED,
    _active_predictor_columns,
    _compute_state_offset,
    _lagged_state_value,
    _build_samples,
    _aggregate_samples,
)
from utils.mlr import fit_and_predict

# ---------------------------------------------------------------------------
# Helpers copied/adapted from b_ExploreData.py
# ---------------------------------------------------------------------------

def _limit_bounds(limit_spec) -> tuple:
    if not isinstance(limit_spec, dict):
        return None, None
    upper = limit_spec.get("upper")
    lower = limit_spec.get("lower")
    upper = float(upper) if upper is not None and pd.notna(upper) else None
    lower = float(lower) if lower is not None and pd.notna(lower) else None
    return upper, lower


def _strip_common_affixes(labels: list) -> list:
    """Remove shared prefix/suffix across labels, preserving uniqueness."""
    if not labels:
        return labels
    if len(labels) == 1:
        return [labels[0].strip()]

    def _common_prefix(strings):
        pref = strings[0]
        for s in strings[1:]:
            i = 0
            max_i = min(len(pref), len(s))
            while i < max_i and pref[i] == s[i]:
                i += 1
            pref = pref[:i]
            if not pref:
                break
        return pref

    def _common_suffix(strings):
        return _common_prefix([s[::-1] for s in strings])[::-1]

    prefix = _common_prefix(labels)
    suffix = _common_suffix(labels)
    if suffix.strip() == ")":
        suffix = ""
    cleaned = []
    for label in labels:
        start = len(prefix) if prefix else 0
        end = len(label) - len(suffix) if suffix else len(label)
        new_label = label[start:end].strip()
        cleaned.append(new_label if new_label else label.strip())
    return cleaned


def _safe_series_colors(n: int) -> list:
    base = [
        "#1f77b4", "#2ca02c", "#17becf", "#9467bd", "#7f7f7f",
        "#bcbd22", "#aec7e8", "#98df8a", "#c5b0d5", "#9edae5",
        "#8c564b", "#c7c7c7",
    ]
    if n <= len(base):
        return base[:n]
    repeats = (n // len(base)) + 1
    return (base * repeats)[:n]


def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


# ---------------------------------------------------------------------------
# MLR prediction helper
# ---------------------------------------------------------------------------

def _build_mlr_estimates(df: pd.DataFrame) -> dict:
    """Return MLR predictions at every row (including observed) for each target.

    Replicates the fit logic from fill_gaps_mlr.main() but builds a predictor
    matrix for ALL rows — not just blank ones — so the estimate series is
    continuous and never uses observed values at observed rows.

    Parameters
    ----------
    df : DataFrame — full Consolidated_filled DataFrame with index=TIMESTAMP,
         including both target columns and _state columns.

    Returns
    -------
    estimates : dict[str, pd.Series]
        Maps target column name -> Series (same index as df) of MLR predictions.
        NaN where prediction was not possible.
    """
    active_cols = _active_predictor_columns()
    estimates = {}

    for target_col in TARGET_COLUMNS:
        if target_col not in df.columns:
            continue

        state_col = f"{target_col}_state"
        state_offset = _compute_state_offset(df, target_col)
        feature_names_ext = active_cols + [f"{target_col}_state (lag={state_offset})"]
        state_series = (
            pd.to_numeric(df[state_col], errors="coerce")
            if state_col in df.columns else None
        )

        obs_mask = df[target_col].notna() & df[target_col].apply(
            lambda v: np.isfinite(float(v)) if pd.notna(v) else False
        )
        obs_ilocs = np.where(obs_mask.values)[0]

        if len(obs_ilocs) < MIN_OBSERVED:
            continue

        n_train = int(np.floor(len(obs_ilocs) * TRAIN_FRACTION))
        train_ilocs = obs_ilocs[:n_train]

        # Determine best variant (highest in-sample R²)
        best_r2 = -np.inf
        best_mode = None

        for variant in VARIANTS:
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
            finite = np.isfinite(preds) & np.isfinite(y_tr)
            if finite.sum() < 2:
                continue
            from sklearn.metrics import r2_score as _r2
            r2 = float(_r2(y_tr[finite], preds[finite]))
            if r2 > best_r2:
                best_r2 = r2
                best_mode = mode

        if best_mode is None:
            continue

        # Refit on ALL observed rows with best mode
        all_samples = _build_samples(
            df, target_col, best_mode, obs_ilocs,
            state_offset=state_offset,
            state_series=state_series,
            predictor_cols=active_cols,
        )
        X_all, y_all = _aggregate_samples(all_samples, best_mode, predictor_cols=active_cols)
        if X_all is None or len(X_all) < 2:
            continue

        # Build predictor matrix for ALL rows (observed + blank alike).
        # Mirrors _build_all_blank_predictor_matrix but does not skip observed rows.
        df_pred = df[active_cols]
        last_obs_iloc = None
        all_ilocs = []
        agg_rows = []

        for iloc_idx in range(len(df)):
            if best_mode == "last":
                agg_sensor = df_pred.iloc[iloc_idx].values.astype(float)
            elif best_mode == "avg12":
                start = max(0, iloc_idx - 11)
                X_win = df_pred.iloc[start: iloc_idx + 1].values.astype(float)
                with np.errstate(all="ignore"):
                    agg_sensor = np.nanmean(X_win, axis=0)
            else:  # avgall — fixed window of state_offset rows (corrected semantics)
                start = max(0, iloc_idx - state_offset)
                X_win = df_pred.iloc[start: iloc_idx + 1].values.astype(float)
                with np.errstate(all="ignore"):
                    agg_sensor = np.nanmean(X_win, axis=0)

            lagged_val = (
                _lagged_state_value(state_series, iloc_idx, best_mode, state_offset, last_obs_iloc)
                if state_series is not None else np.nan
            )

            all_ilocs.append(iloc_idx)
            agg_rows.append(np.append(agg_sensor, lagged_val))

            if obs_mask.values[iloc_idx]:
                last_obs_iloc = iloc_idx

        if not agg_rows:
            continue

        X_all_rows = np.array(agg_rows, dtype=float)
        preds_all, _ = fit_and_predict(X_all, y_all, X_all_rows, feature_names=feature_names_ext)

        est = pd.Series(np.nan, index=df.index, dtype=float)
        for k, iloc_idx in enumerate(all_ilocs):
            est.iat[iloc_idx] = preds_all[k]
        estimates[target_col] = est

    return estimates


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_mlr_fill_timeseries(repo_root: Path) -> None:
    filled_csv  = repo_root / "data" / "output" / "regression" / "Consolidated_filled.csv"
    limits_csv  = repo_root / "data" / "input"  / "Limits.csv"
    out_path    = repo_root / "data" / "output" / "regression" / "Target_timeseries_mlr_fill.png"

    if not filled_csv.exists():
        raise FileNotFoundError(f"Consolidated_filled.csv not found: {filled_csv}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(filled_csv, low_memory=False)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df[df["TIMESTAMP"].notna()].sort_values("TIMESTAMP").copy()

    start_date = pd.Timestamp("2021-12-15")
    end_date   = pd.Timestamp("2025-03-01")
    df = df[(df["TIMESTAMP"] >= start_date) & (df["TIMESTAMP"] <= end_date)].copy()
    df = df.set_index("TIMESTAMP")

    all_cols = list(df.columns)

    # ------------------------------------------------------------------
    # Identify target columns (same logic as b_ExploreData.py)
    # ------------------------------------------------------------------
    weather_cols_norm = {
        _norm_col(c) for c in [
            "Wind speed x (m/s)", "Wind speed y (m/s)",
            "Maximum 3s wind gust (m/s)", "Atmospheric pressure (mBar)",
            "Longwave (IR) radiation (W/m2)", "Shortwave (solar) radiation (W/m2)",
            "24hr precipitation total (mm)", "Air temperature (°C)",
            "Humidity (%)", "Air Temperature (C)", "Air temperature (C)",
        ]
    }
    non_target = set()
    for c in all_cols:
        cn = _norm_col(c)
        if c.startswith("Pfl - ") or c.startswith("SCADA - ") or cn in weather_cols_norm:
            non_target.add(c)
        if c == "Interpolated":
            non_target.add(c)

    target_cols = [
        c for c in all_cols
        if c not in non_target
        and not c.endswith("_state")
        and not c.endswith("_res")
    ]

    # Sort by descending valid-count (matches b_ExploreData.py ordering)
    target_cols = sorted(
        target_cols,
        key=lambda c: (-int(pd.to_numeric(df[c], errors="coerce").notna().sum()), c.lower()),
    )

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------
    limits_records = load_limits_records(limits_csv)
    all_limit_specs = map_limits_to_columns(target_cols, limits_records)

    # ------------------------------------------------------------------
    # Compute MLR predictions at every row (including observed rows)
    # ------------------------------------------------------------------
    print("Computing MLR predictions at all rows (including observed) ...")
    mlr_estimates = _build_mlr_estimates(df)

    # ------------------------------------------------------------------
    # Layout constants (mirrored from b_ExploreData.py)
    # ------------------------------------------------------------------
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

    # Build wrapped labels (same as b_ExploreData.py)
    raw_labels = list(target_cols)
    raw_labels_stripped = _strip_common_affixes(raw_labels)
    wrapped_labels = [textwrap.fill(lbl, width=17) for lbl in raw_labels_stripped]

    global_max_label_line_len = max(
        (max(len(line) for line in lbl.splitlines()) if lbl else 0)
        for lbl in wrapped_labels
    )
    est_char_width_in = max(0.045, 0.0055 * y_label_font_size)
    left_margin_in = global_max_label_line_len * est_char_width_in + 0.55
    shared_left_margin = max(0.14, min(0.26, left_margin_in / fig_width))

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    n_rows  = len(target_cols)
    fig_h   = max(min_fig_height, row_height * n_rows)
    palette = _safe_series_colors(n_rows)

    fig, axes_raw = plt.subplots(
        n_rows, 1, sharex=True,
        figsize=(fig_width, fig_h),
        gridspec_kw={"hspace": shared_hspace},
    )
    axes = [axes_raw] if n_rows == 1 else list(axes_raw)

    quarter_ticks = pd.date_range(start=start_date, end=end_date, freq="QS-JAN")

    for i, (ax, col, label) in enumerate(zip(axes, target_cols, wrapped_labels)):
        color = palette[i]

        # --- Observed values (raw target column) ---
        observed = pd.to_numeric(df[col], errors="coerce")
        obs_mask = observed.notna()

        # --- MLR estimate (in-memory predictions only; no _state fallback) ---
        estimate = mlr_estimates.get(col)

        # --- Limit lines (before scatter so they sit behind data) ---
        if col in all_limit_specs:
            upper_limit, lower_limit = _limit_bounds(all_limit_specs[col])
            if upper_limit is not None:
                ax.axhline(y=upper_limit, color="#ff0000", linewidth=0.7, linestyle="-", zorder=1.2)
            if lower_limit is not None:
                ax.axhline(y=lower_limit, color="#2ca02c", linewidth=0.7, linestyle="-", zorder=1.2)

        # --- Line: MLR estimates, broken at each measurement row after the first ---
        if estimate is not None and estimate.notna().any():
            est_vals = estimate.to_numpy(dtype=float)
            segment_mask = np.isfinite(est_vals)
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
                ax.plot(
                    estimate.index[seg_slice], est_vals[seg_slice],
                    linestyle="-", linewidth=0.7,
                    color=color, alpha=0.65,
                    zorder=1.5, label="MLR estimate" if seg_start == 0 else None,
                )
                seg_start = None
            if seg_start is not None:
                seg_slice = slice(seg_start, len(segment_mask))
                ax.plot(
                    estimate.index[seg_slice], est_vals[seg_slice],
                    linestyle="-", linewidth=0.7,
                    color=color, alpha=0.65,
                    zorder=1.5, label="MLR estimate" if seg_start == 0 else None,
                )

        # --- Scatter: observed measured values ---
        if obs_mask.any():
            ax.plot(
                observed.index[obs_mask], observed.values[obs_mask],
                linestyle="", marker="o", markersize=4.5,
                color=color,
                markeredgecolor="black", markeredgewidth=0.5,
                alpha=0.95, zorder=2.5, label="Measured",
            )

        # Y-axis limits with padding, based on measurements only. If a target
        # has no measured values, fall back to the estimate range.
        y_low = y_high = None
        observed_vals = observed.to_numpy(dtype=float)
        finite_obs = observed_vals[np.isfinite(observed_vals)]
        if finite_obs.size > 0:
            y_low = float(np.min(finite_obs))
            y_high = float(np.max(finite_obs))
        elif estimate is not None:
            est_vals = estimate.to_numpy(dtype=float)
            finite_est = est_vals[np.isfinite(est_vals)]
            if finite_est.size > 0:
                y_low = float(np.min(finite_est))
                y_high = float(np.max(finite_est))

        if y_low is not None and y_high is not None:
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

    # Margins
    shared_top_margin    = max(0.86, min(0.995, 1.0 - (top_margin_in    / fig_h)))
    shared_bottom_margin = max(0.08, min(0.30,  bottom_margin_in / fig_h))
    fig.subplots_adjust(
        left=shared_left_margin, right=shared_right_margin,
        top=shared_top_margin,   bottom=shared_bottom_margin,
        hspace=shared_hspace,
    )

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=0.7, linestyle="-",
                   alpha=0.65, label="MLR estimate (_state)"),
        plt.Line2D([0], [0], color="black", linestyle="", marker="o",
                   markersize=4.5, markeredgecolor="black", markeredgewidth=0.5,
                   label="Measured value"),
        plt.Line2D([0], [0], color="#ff0000", linewidth=0.7, linestyle="-",
                   label="Strictest legal threshold (upper bound)"),
        plt.Line2D([0], [0], color="#2ca02c", linewidth=0.7, linestyle="-",
                   label="Strictest legal threshold (lower bound)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        framealpha=0.85,
        fontsize=font_size,
        borderaxespad=0.12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote chart to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    plot_mlr_fill_timeseries(repo_root)

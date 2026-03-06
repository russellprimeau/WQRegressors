"""
Generate sensor availability summary tables for all raw data sources.

Writes CSV summaries to data/sensors/summaries/:
  - FullHourly_summary.csv  (profiler sensor data)
  - Weather_summary.csv     (weather station parameters)
  - SCADA_summary.csv       (SCADA parameters)
  - Eurofins_summary.csv    (Eurofins lab measurements)

Run directly to regenerate all four summaries:
    python src/b_ExploreData.py
"""
import re
import textwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from utils.limits import (
    load_limits_records,
    map_limits_to_columns,
    limit_exceedance_mask,
    normalize_limit_name,
)


def _summary_theme_dirs(repo_root: Path) -> dict:
    """Return themed output directories under data/output/sensors."""
    base = repo_root / "data" / "output" / "sensors"
    dirs = {
        "root": base,
        "tables": base / "tables",
        "availability_charts": base / "availability_charts",
        "timeseries_charts": base / "timeseries_charts",
        "split_comparison": base / "split_comparison",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _valid_nonblank_mask(series: pd.Series) -> pd.Series:
    """Treat NaN, blanks, and #N/A as invalid values."""
    as_str = series.astype(str).str.strip()
    return series.notna() & (as_str != "") & (as_str.str.upper() != "#N/A")


def _mean_std_valid_numeric(series: pd.Series, valid_mask: pd.Series) -> tuple:
    """Return mean/std for valid numeric values only."""
    numeric = pd.to_numeric(series.where(valid_mask), errors="coerce").dropna()
    if numeric.empty:
        return None, None
    return float(numeric.mean()), float(numeric.std(ddof=0))


def _mean_std_valid_numeric_normalized_01(series: pd.Series, valid_mask: pd.Series) -> tuple:
    """Return mean/std after min-max normalization to [0, 1]; constant columns return NaN stats."""
    numeric = pd.to_numeric(series.where(valid_mask), errors="coerce").dropna()
    if numeric.empty:
        return None, None
    col_min = numeric.min()
    col_max = numeric.max()
    if col_max != col_min:
        normalized = (numeric - col_min) / (col_max - col_min)
    else:
        return None, None
    return float(normalized.mean()), float(normalized.std(ddof=0))


def _limit_bounds(limit_spec) -> tuple:
    if not isinstance(limit_spec, dict):
        return None, None
    upper = limit_spec.get("upper")
    lower = limit_spec.get("lower")
    upper = float(upper) if upper is not None and pd.notna(upper) else None
    lower = float(lower) if lower is not None and pd.notna(lower) else None
    return upper, lower


def _write_clustered_bar_chart(
    summary_df: pd.DataFrame,
    out_dir: Path,
    dataset_name: str,
    include_exceed_bar: bool = False,
) -> None:
    """
    Write a clustered bar chart with:
      - left axis: count bars
      - right axis: normalized std bar
    """
    label_col = "parameter" if "parameter" in summary_df.columns else "sensor"
    if label_col not in summary_df.columns or "valid_count_consolidated" not in summary_df.columns:
        return
    if "std_valid_consolidated_norm01" not in summary_df.columns:
        return

    plot_df = summary_df[[label_col, "valid_count_consolidated", "std_valid_consolidated_norm01"]].copy()
    if include_exceed_bar and "count_exceed_limit" in summary_df.columns:
        plot_df["count_exceed_limit"] = summary_df["count_exceed_limit"]

    plot_df["valid_count_consolidated"] = pd.to_numeric(plot_df["valid_count_consolidated"], errors="coerce")
    plot_df["std_valid_consolidated_norm01"] = pd.to_numeric(plot_df["std_valid_consolidated_norm01"], errors="coerce")
    if "count_exceed_limit" in plot_df.columns:
        plot_df["count_exceed_limit"] = pd.to_numeric(plot_df["count_exceed_limit"], errors="coerce")

    plot_df["_label_sort"] = plot_df[label_col].astype(str).str.lower()
    plot_df = plot_df.sort_values(
        ["valid_count_consolidated", "_label_sort"],
        ascending=[False, True]
    ).reset_index(drop=True)
    plot_df = plot_df.drop(columns=["_label_sort"])
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    width = 0.25 if "count_exceed_limit" in plot_df.columns else 0.3

    fig, ax_left = plt.subplots(figsize=(max(13, len(plot_df) * 0.6), 6))

    bars_count = ax_left.bar(
        x - width if "count_exceed_limit" in plot_df.columns else x - width / 2,
        plot_df["valid_count_consolidated"].fillna(0),
        width=width,
        label="valid_count_consolidated",
        color="#4C78A8",
    )
    left_handles = [bars_count]
    left_labels = ["Valid count"]

    if "count_exceed_limit" in plot_df.columns:
        bars_exceed = ax_left.bar(
            x,
            plot_df["count_exceed_limit"].fillna(0),
            width=width,
            label="count_exceed_limit",
            color="#F58518",
        )
        left_handles.append(bars_exceed)
        left_labels.append("Exceed limit count")

    ax_right = ax_left.twinx()
    right_x = x + (width if "count_exceed_limit" in plot_df.columns else width / 2)
    bars_std = ax_right.bar(
        right_x,
        plot_df["std_valid_consolidated_norm01"].fillna(0),
        width=width,
        label="std_valid_consolidated_norm01",
        color="#54A24B",
    )

    ax_left.set_ylabel("Count")
    ax_right.set_ylabel("Std (normalized [0,1])")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(plot_df[label_col].astype(str), rotation=45, ha="right")
    ax_left.grid(axis="y", linestyle="--", alpha=0.35)

    handles = left_handles + [bars_std]
    labels = left_labels + ["Std norm [0,1]"]
    ax_left.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    fig_path = out_dir / f"{dataset_name}_clustered_bar.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote chart to: {fig_path}")


def _write_eurofins_interval_clustered_bar(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Write Eurofins interval chart with:
      - left axis: mean/median hours between valid measurements
      - right axis: count of intervals greater than median
    """
    required = {
        "parameter",
        "valid_count_consolidated",
        "mean_hours_between_measurements",
        "median_hours_between_measurements",
        "count_gt_median_hours",
    }
    if summary_df.empty or not required.issubset(set(summary_df.columns)):
        return

    plot_df = summary_df[
        [
            "parameter",
            "valid_count_consolidated",
            "mean_hours_between_measurements",
            "median_hours_between_measurements",
            "count_gt_median_hours",
        ]
    ].copy()
    plot_df["valid_count_consolidated"] = pd.to_numeric(plot_df["valid_count_consolidated"], errors="coerce")
    plot_df["mean_hours_between_measurements"] = pd.to_numeric(plot_df["mean_hours_between_measurements"], errors="coerce")
    plot_df["median_hours_between_measurements"] = pd.to_numeric(plot_df["median_hours_between_measurements"], errors="coerce")
    plot_df["count_gt_median_hours"] = pd.to_numeric(plot_df["count_gt_median_hours"], errors="coerce")

    plot_df["_label_sort"] = plot_df["parameter"].astype(str).str.lower()
    plot_df = plot_df.sort_values(
        ["median_hours_between_measurements", "_label_sort"],
        ascending=[True, True],
    ).reset_index(drop=True)
    plot_df = plot_df.drop(columns=["_label_sort"])
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    width = 0.26
    font_size = 14  # ~40% larger than Matplotlib default (10)

    def _format_label(value, decimals=1):
        """Format label: if rounds to 0, show '0', else show with decimals."""
        rounded = round(value, decimals)
        if rounded == 0:
            return "0"
        return f"{value:.{decimals}f}"

    fig, ax_left = plt.subplots(figsize=(max(13, len(plot_df) * 0.65), 6))
    bars_median = ax_left.bar(
        x,
        plot_df["median_hours_between_measurements"].fillna(0),
        width=width,
        color="#54A24B",
        label="Median hours",
    )
    bars_mean = ax_left.bar(
        x - width,
        plot_df["mean_hours_between_measurements"].fillna(0),
        width=width,
        color="#4C78A8",
        label="Mean hours",
    )
    
    labels_mean = [_format_label(v, 1) for v in plot_df["mean_hours_between_measurements"].fillna(0)]
    ax_left.bar_label(bars_mean, labels=labels_mean, rotation=90, padding=3, fontsize=font_size)
    labels_median = [_format_label(v, 1) for v in plot_df["median_hours_between_measurements"].fillna(0)]
    ax_left.bar_label(bars_median, labels=labels_median, rotation=90, padding=3, fontsize=font_size)

    ax_right = ax_left.twinx()
    bars_gt = ax_right.bar(
        x + width,
        plot_df["count_gt_median_hours"].fillna(0),
        width=width,
        color="#F58518",
        label="Count > median",
    )

    labels_gt = [_format_label(v, 0) for v in plot_df["count_gt_median_hours"].fillna(0)]
    ax_right.bar_label(bars_gt, labels=labels_gt, rotation=90, padding=3, fontsize=font_size)

    # Add headroom so vertical bar labels do not clip at the top edge.
    left_max = float(
        max(
            pd.to_numeric(plot_df["mean_hours_between_measurements"], errors="coerce").fillna(0).max(),
            pd.to_numeric(plot_df["median_hours_between_measurements"], errors="coerce").fillna(0).max(),
        )
    )
    right_max = float(pd.to_numeric(plot_df["count_gt_median_hours"], errors="coerce").fillna(0).max())
    ax_left.set_ylim(0, (left_max * 1.18) if left_max > 0 else 1)
    ax_right.set_ylim(0, (right_max * 1.23) if right_max > 0 else 1)

    ax_left.set_ylabel("Hours between measurements", fontsize=font_size)
    ax_right.set_ylabel("Count > median", fontsize=font_size)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(plot_df["parameter"].astype(str), rotation=45, ha="right", fontsize=font_size)
    ax_left.tick_params(axis="y", labelsize=font_size)
    ax_right.tick_params(axis="y", labelsize=font_size)
    ax_left.grid(axis="y", linestyle="--", alpha=0.35)

    handles = [bars_median, bars_mean, bars_gt]
    labels = ["Median hours", "Mean hours", "Count > median"]
    ax_left.legend(handles, labels, loc="upper left", fontsize=font_size)

    fig.tight_layout()
    out_path = out_dir / "Eurofins_interval_clustered_bar.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote chart to: {out_path}")


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic without SciPy dependency."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    x = np.sort(np.unique(np.concatenate([a, b])))
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    cdf_a = np.searchsorted(a_sorted, x, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, x, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _split_index_from_cumulative_counts(valid_counts: list, train_fraction: float = 0.8) -> tuple:
    """
    Mirror temporal split logic from utils.training._split_index_from_cumulative_valid_counts.
    Returns (split_idx, total_valid, train_valid) where split_idx is count of train samples.
    """
    if not valid_counts:
        return 0, 0, 0

    counts = [max(0, int(v)) for v in valid_counts]
    n = len(counts)
    total_valid = int(np.sum(counts))
    if total_valid > 0:
        cutoff = float(train_fraction) * total_valid
        cumulative = np.cumsum(counts)
        split_idx = int(np.searchsorted(cumulative, cutoff, side="left") + 1)
    else:
        split_idx = int(n * float(train_fraction))

    if n > 1:
        split_idx = max(1, min(n - 1, split_idx))
    else:
        split_idx = n

    train_valid = int(np.sum(counts[:split_idx]))
    return split_idx, total_valid, train_valid


def _write_split_clustered_bars(
    split_df: pd.DataFrame,
    out_dir: Path,
    category_name: str,
    metric_col: str,
    y_label: str,
) -> None:
    """Write clustered bars where each feature has train/test/combined bars."""
    if split_df.empty or metric_col not in split_df.columns:
        return

    plot_df = split_df.copy()
    if "series_display" not in plot_df.columns:
        plot_df["series_display"] = plot_df["series"]
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df["n_samples"] = pd.to_numeric(plot_df["n_samples"], errors="coerce")

    combined_rows = plot_df[plot_df["split"] == "combined"].copy()
    if not combined_rows.empty:
        combined_rows["_label_sort"] = combined_rows["series_display"].astype(str).str.lower()
        combined_order = (
            combined_rows
            .sort_values(["n_samples", "_label_sort"], ascending=[False, True])["series_display"]
            .tolist()
        )
    else:
        combined_order = []
    if not combined_order:
        combined_order = sorted(plot_df["series_display"].dropna().unique().tolist())

    piv = (
        plot_df.pivot_table(index="series_display", columns="split", values=metric_col, aggfunc="first")
        .reindex(combined_order)
    )
    if piv.empty:
        return

    split_order = ["train", "test", "combined"]
    x = np.arange(len(piv.index))
    width = 0.24
    colors = {"train": "#4C78A8", "test": "#F58518", "combined": "#54A24B"}

    fig, ax = plt.subplots(figsize=(max(13, len(piv.index) * 0.7), 6))
    for i, split_name in enumerate(split_order):
        vals = piv[split_name].to_numpy() if split_name in piv.columns else np.full(len(piv.index), np.nan)
        ax.bar(
            x + (i - 1) * width,
            np.nan_to_num(vals, nan=0.0),
            width=width,
            label=split_name.capitalize(),
            color=colors[split_name],
            alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(piv.index.tolist(), rotation=45, ha="right")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = out_dir / f"{category_name}_split_{metric_col}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote chart to: {out_path}")


def write_target_split_comparison(repo_root: Path) -> None:
    """
    Compare time-based train/test/combined statistics for Target and Target_diff series.
    Saves CSV summaries and clustered bar charts.
    """
    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    limits_csv = repo_root / "data" / "input" / "Limits.csv"
    out_dir = _summary_theme_dirs(repo_root)["split_comparison"]
    if not consolidated_csv.exists():
        return

    df = pd.read_csv(consolidated_csv, low_memory=False)
    if df.empty or "TIMESTAMP" not in df.columns:
        return
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df[df["TIMESTAMP"].notna()].sort_values("TIMESTAMP").reset_index(drop=True)
    if df.empty:
        return

    def _norm_col(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    all_cols = [c for c in df.columns if c not in {"TIMESTAMP", "Interpolated"} and not c.endswith("_state")]
    all_cols_by_norm = {_norm_col(c): c for c in all_cols}

    weather_cols_expected = [
        "Wind speed x (m/s)",
        "Wind speed y (m/s)",
        "Maximum 3s wind gust (m/s)",
        "Atmospheric pressure (mBar)",
        "Longwave (IR) radiation (W/m2)",
        "Shortwave (solar) radiation (W/m2)",
        "24hr precipitation total (mm)",
        "Air temperature (°C)",
        "Air temperature (C)",
        "Humidity (%)",
    ]
    weather_norm = {_norm_col(c) for c in weather_cols_expected}
    weather_cols = [all_cols_by_norm[k] for k in all_cols_by_norm if k in weather_norm]

    surface_cols = [c for c in all_cols if c.startswith("Pfl - ")]
    scada_cols = [c for c in all_cols if c.startswith("SCADA - ")]
    target_cols = [c for c in all_cols if (not c.endswith("_res") and c not in weather_cols and not c.startswith("Pfl - ") and not c.startswith("SCADA - "))]
    target_diff_cols = [c for c in all_cols if c.endswith("_res")]
    predictor_cols = [c for c in all_cols if c not in target_cols and c not in target_diff_cols]
    # Threshold map for base target columns.
    limits_records = load_limits_records(limits_csv)
    all_limit_specs = map_limits_to_columns(all_cols, limits_records)

    n_rows = len(df)
    window = 168  # target timestep + previous 167
    first_valid_idx = window - 1

    if not predictor_cols:
        predictor_valid = pd.DataFrame(index=df.index)
    else:
        predictor_valid = pd.DataFrame(
            {c: _valid_nonblank_mask(df[c]) for c in predictor_cols},
            index=df.index,
        )

    def _predictor_valid_count_for_idx(idx: int) -> int:
        if predictor_valid.empty or idx < first_valid_idx:
            return 0
        w = predictor_valid.iloc[idx - (window - 1): idx + 1].to_numpy(dtype=bool)
        return int(np.count_nonzero(w))

    def _compute_category_stats(category_name: str, series_cols: list):
        rows = []
        shift_rows = []
        for col in series_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            base_target_col = col[:-4] if col.endswith("_res") else col
            base_target_col = all_cols_by_norm.get(_norm_col(base_target_col), base_target_col)
            base_series = pd.to_numeric(df[base_target_col], errors="coerce") if base_target_col in df.columns else series
            limit_spec = all_limit_specs.get(base_target_col)
            upper_limit, lower_limit = _limit_bounds(limit_spec)

            # Per-series split by cumulative predictor-valid counts over the 168-step sample window.
            valid_idx = np.where(series.notna().to_numpy() & (np.arange(n_rows) >= first_valid_idx))[0]
            if valid_idx.size == 0:
                continue

            valid_counts = [_predictor_valid_count_for_idx(int(i)) for i in valid_idx]
            split_cut, _, _ = _split_index_from_cumulative_counts(valid_counts, train_fraction=0.8)
            split_cut = max(1, min(split_cut, int(valid_idx.size)))

            eligible_vals = series.iloc[valid_idx].dropna()
            col_min = float(eligible_vals.min()) if not eligible_vals.empty else np.nan
            col_max = float(eligible_vals.max()) if not eligible_vals.empty else np.nan
            use_norm = np.isfinite(col_min) and np.isfinite(col_max) and col_max > col_min

            split_map = {
                "train": valid_idx[:split_cut],
                "test": valid_idx[split_cut:],
                "combined": valid_idx,
            }

            norm_train = np.array([])
            norm_test = np.array([])
            for split_name, idxs in split_map.items():
                vals = pd.to_numeric(series.iloc[idxs], errors="coerce").dropna().to_numpy(dtype=float)
                n_samples = int(len(idxs))

                if use_norm and vals.size > 0:
                    vals_norm = (vals - col_min) / (col_max - col_min)
                    mean_norm = float(np.mean(vals_norm))
                    std_norm = float(np.std(vals_norm, ddof=0))
                else:
                    vals_norm = np.array([])
                    mean_norm = np.nan
                    std_norm = np.nan

                if split_name == "train":
                    norm_train = vals_norm
                if split_name == "test":
                    norm_test = vals_norm

                exceed_count = np.nan
                exceed_pct = np.nan
                if (upper_limit is not None) or (lower_limit is not None):
                    base_vals = pd.to_numeric(base_series.iloc[idxs], errors="coerce").to_numpy(dtype=float)
                    valid_base = np.isfinite(base_vals)
                    if valid_base.any():
                        exceed = limit_exceedance_mask(
                            base_vals[valid_base],
                            upper=upper_limit,
                            lower=lower_limit,
                        )
                        exceed_count = int(np.sum(exceed))
                        exceed_pct = float(np.mean(exceed) * 100.0)
                    else:
                        exceed_count = 0
                        exceed_pct = 0.0

                window_valid_pcts = []
                full_window_count = 0
                window_eligible_count = 0
                if n_samples > 0 and not predictor_valid.empty:
                    for idx in idxs:
                        window_eligible_count += 1
                        w = predictor_valid.iloc[idx - (window - 1): idx + 1].to_numpy(dtype=bool)
                        if w.size == 0:
                            continue
                        window_valid_pcts.append(float(w.mean() * 100.0))
                        if bool(w.all()):
                            full_window_count += 1

                avg_window_valid_pct = float(np.mean(window_valid_pcts)) if window_valid_pcts else np.nan
                full_window_pct = (
                    100.0 * full_window_count / window_eligible_count
                    if window_eligible_count > 0 else np.nan
                )

                disp = col
                if category_name == "Target_diff" and disp.endswith("_res"):
                    disp = disp[:-4]

                rows.append({
                    "category": category_name,
                    "series": col,
                    "series_display": disp,
                    "split": split_name,
                    "n_samples": n_samples,
                    "threshold": upper_limit,
                    "threshold_upper": upper_limit,
                    "threshold_lower": lower_limit,
                    "exceed_count": exceed_count,
                    "exceed_pct": exceed_pct,
                    "mean_norm01": mean_norm,
                    "std_norm01": std_norm,
                    "avg_predictor_window_valid_pct": avg_window_valid_pct,
                    "window_eligible_samples": window_eligible_count,
                    "full_predictor_window_count": full_window_count,
                    "full_predictor_window_pct": full_window_pct,
                })

            shift_rows.append({
                "category": category_name,
                "series": col,
                "series_display": (col[:-4] if (category_name == "Target_diff" and col.endswith("_res")) else col),
                "n_train": int(len(split_map["train"])),
                "n_test": int(len(split_map["test"])),
                "ks_train_test_norm01": _ks_statistic(norm_train, norm_test),
                "mean_shift_abs_norm01": (
                    float(abs(np.mean(norm_train) - np.mean(norm_test)))
                    if norm_train.size > 0 and norm_test.size > 0 else np.nan
                ),
            })

        return pd.DataFrame(rows), pd.DataFrame(shift_rows)

    target_stats_df, target_shift_df = _compute_category_stats("Target", target_cols)
    target_diff_stats_df, target_diff_shift_df = _compute_category_stats("Target_diff", target_diff_cols)

    if not target_stats_df.empty:
        out_csv = out_dir / "Target_split_comparison.csv"
        target_stats_df.to_csv(out_csv, index=False)
        print(f"Wrote summary to: {out_csv}")
    if not target_diff_stats_df.empty:
        out_csv = out_dir / "Target_diff_split_comparison.csv"
        target_diff_stats_df.to_csv(out_csv, index=False)
        print(f"Wrote summary to: {out_csv}")
    if not target_shift_df.empty:
        out_csv = out_dir / "Target_split_shift_summary.csv"
        target_shift_df.to_csv(out_csv, index=False)
        print(f"Wrote summary to: {out_csv}")
    if not target_diff_shift_df.empty:
        out_csv = out_dir / "Target_diff_split_shift_summary.csv"
        target_diff_shift_df.to_csv(out_csv, index=False)
        print(f"Wrote summary to: {out_csv}")

    metric_specs = [
        ("n_samples", "Sample count"),
        ("exceed_pct", "Percent outside limits (%)"),
        ("std_norm01", "Std (normalized [0,1])"),
        ("mean_norm01", "Mean (normalized [0,1])"),
        ("avg_predictor_window_valid_pct", "Avg predictor validity over 168h window (%)"),
        ("full_predictor_window_count", "Count with fully valid 168h predictor window"),
        ("full_predictor_window_pct", "Percent with fully valid 168h predictor window (%)"),
    ]

    for metric_col, y_label in metric_specs:
        if not target_stats_df.empty:
            _write_split_clustered_bars(target_stats_df, out_dir, "Target", metric_col, y_label)
        if not target_diff_stats_df.empty:
            _write_split_clustered_bars(target_diff_stats_df, out_dir, "Target_diff", metric_col, y_label)


def _write_coverage_timeline_raster(repo_root: Path, coverage_entries: list) -> None:
    """Write condensed coverage timeline: proxy rasters + Eurofins marker rows."""
    if not coverage_entries:
        return

    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    out_dir = _summary_theme_dirs(repo_root)["availability_charts"]
    if not consolidated_csv.exists():
        return

    consolidated_df = pd.read_csv(consolidated_csv, low_memory=False)
    if consolidated_df.empty:
        return

    valid_entries = [e for e in coverage_entries if e.get("column") in consolidated_df.columns]
    if not valid_entries:
        return

    # Count window for label statistics.
    count_window_mask = np.ones(len(consolidated_df), dtype=bool)
    if "TIMESTAMP" in consolidated_df.columns:
        ts_for_counts = pd.to_datetime(consolidated_df["TIMESTAMP"], errors="coerce")
        if ts_for_counts.notna().any():
            count_start = pd.Timestamp("2021-12-28 00:00:00")
            count_end = pd.Timestamp("2025-02-11 23:59:59")
            count_window_mask = (
                ts_for_counts.notna()
                & (ts_for_counts >= count_start)
                & (ts_for_counts <= count_end)
            ).to_numpy()

    # De-duplicate entries while preserving first occurrence order.
    dedup = {}
    for e in valid_entries:
        key = (e.get("dataset"), e.get("label"), e.get("column"))
        if key not in dedup:
            dedup[key] = e
    valid_entries = list(dedup.values())

    # Pick one proxy series for each high-frequency source.
    proxy_rows = []
    for dataset, display in [("FullHourly", "Surface"), ("Weather", "Weather"), ("SCADA", "SCADA")]:
        dataset_entries = [e for e in valid_entries if e.get("dataset") == dataset]
        if not dataset_entries:
            continue
        proxy_col = dataset_entries[0]["column"]
        proxy_mask = _valid_nonblank_mask(consolidated_df[proxy_col]).astype(int).to_numpy()
        proxy_count = int((proxy_mask == 1)[count_window_mask].sum())
        proxy_rows.append({
            "row_type": "proxy",
            "label": display,
            "mask": proxy_mask,
            "count": proxy_count,
            "feature_count": len(dataset_entries),
        })

    euro_entries = [e for e in valid_entries if e.get("dataset") == "Eurofins"]
    euro_rows = []
    for e in euro_entries:
        euro_mask = _valid_nonblank_mask(consolidated_df[e["column"]]).astype(int).to_numpy()
        euro_count = int((euro_mask == 1)[count_window_mask].sum())
        euro_rows.append({
            "row_type": "target",
            "label": e["label"],  # no "Eurofins: " prefix
            "column": e["column"],
            "limit_value": e.get("limit_value"),
            "limit_upper": e.get("limit_upper"),
            "limit_lower": e.get("limit_lower"),
            "mask": euro_mask,
            "count": euro_count,
        })

    if not proxy_rows and not euro_rows:
        return

    all_rows = proxy_rows + euro_rows
    all_rows.sort(key=lambda r: (-r["count"], str(r["label"]).lower()))

    total_rows = len(all_rows)
    fig_h = max(5, min(22, 0.30 * total_rows + 1.2))
    fig, ax = plt.subplots(figsize=(13, fig_h))

    # Use one consistent text size across all figure text elements.
    text_fs = 11

    y_labels = []
    y_positions = []

    for y, row in enumerate(all_rows):
        if row["row_type"] == "proxy":
            y_labels.append(f"{row['label']} ({row['count']} timesteps x {row['feature_count']} features)")
            y_positions.append(y)
            ax.hlines(y, 0, len(row["mask"]) - 1, color="#e8e8e8", linewidth=7, alpha=1.0, zorder=1)
            present_x = np.where(row["mask"] == 1)[0]
            if present_x.size > 0:
                ax.scatter(present_x, np.full(present_x.shape, y), s=8, marker="s", color="#1f77b4", zorder=2)
            continue

        y_labels.append(f"{row['label']} ({row['count']} samples)")
        y_positions.append(y)
        ax.hlines(y, 0, len(row["mask"]) - 1, color="#efefef", linewidth=1.2, alpha=0.95, zorder=1)
        present_x = np.where(row["mask"] == 1)[0]
        if present_x.size > 0:
            sample_series = consolidated_df.iloc[present_x][row["column"]]
            limit_upper = row.get("limit_upper")
            limit_lower = row.get("limit_lower")
            if (limit_upper is None) and (limit_lower is None):
                limit_upper = row.get("limit_value")
            if (limit_upper is not None) or (limit_lower is not None):
                numeric_vals = pd.to_numeric(sample_series, errors="coerce")
                exceed_mask = limit_exceedance_mask(
                    numeric_vals,
                    upper=limit_upper,
                    lower=limit_lower,
                )
                point_colors = np.where(exceed_mask, "#d62728", "#2ca02c")
            else:
                point_colors = "#2ca02c"
            raw_preview = sample_series.head(5).tolist()
            numeric_preview = pd.to_numeric(sample_series, errors="coerce").head(5).tolist()
            print(
                f"[LIMIT CHECK] {row['label']}: upper={limit_upper}, lower={limit_lower}, "
                f"first_raw_values={raw_preview}, first_numeric_values={numeric_preview}"
            )
            ax.scatter(present_x, np.full(present_x.shape, y), s=14, marker="o", c=point_colors, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=text_fs)
    # ax.set_ylabel("Feature", fontsize=text_fs)
    # ax.set_xlabel("Time", fontsize=text_fs)
    ax.tick_params(axis="both", labelsize=text_fs)
    ax.set_ylim(total_rows - 0.5, -0.5)

    # Limit to requested date range and build readable x ticks from timestamp if available.
    x_start = 0
    x_end = len(consolidated_df) - 1
    if "TIMESTAMP" in consolidated_df.columns:
        ts = pd.to_datetime(consolidated_df["TIMESTAMP"], errors="coerce")
        if ts.notna().any():
            start_date = pd.Timestamp("2021-12-15")
            end_date = pd.Timestamp("2025-03-01")

            valid_idx = np.where(ts.notna())[0]
            ts_valid = ts.iloc[valid_idx]
            in_range_idx = valid_idx[(ts_valid >= start_date) & (ts_valid <= end_date)]

            if len(in_range_idx) > 0:
                x_start = int(in_range_idx[0])
                x_end = int(in_range_idx[-1])

            quarter_starts = pd.date_range(start=start_date, end=end_date, freq="QS-JAN")
            tick_idx = []
            tick_labels = []
            for dt in quarter_starts:
                match_positions = np.where((ts >= dt) & (np.arange(len(ts)) >= x_start) & (np.arange(len(ts)) <= x_end))[0]
                if match_positions.size > 0:
                    tick_idx.append(int(match_positions[0]))
                    tick_labels.append(dt.strftime("%Y-%m-%d"))

            if tick_idx:
                ax.set_xticks(tick_idx)
                ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=text_fs)
    ax.set_xlim(x_start, x_end)

    eurofins_legend = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#2ca02c", markeredgecolor="#2ca02c", markersize=6, label="Within limits"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#d62728", markeredgecolor="#d62728", markersize=6, label="Outside limits"),
    ]
    ax.legend(handles=eurofins_legend, loc="center right", bbox_to_anchor=(0.985, 0.5), borderaxespad=0.0, fontsize=text_fs)

    fig.tight_layout()
    out_path = out_dir / "Coverage_timeline_raster.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote chart to: {out_path}")


def _strip_common_affixes(labels: list) -> list:
    """Remove shared prefix/suffix across labels, preserving uniqueness where possible."""
    if not labels:
        return labels
    if len(labels) == 1:
        return [labels[0].strip()]

    def _common_prefix(strings: list) -> str:
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

    def _common_suffix(strings: list) -> str:
        rev = [s[::-1] for s in strings]
        return _common_prefix(rev)[::-1]

    prefix = _common_prefix(labels)
    suffix = _common_suffix(labels)
    # Preserve unit-closing parenthesis when it is the only shared suffix.
    if suffix.strip() == ")":
        suffix = ""

    cleaned = []
    for label in labels:
        start = len(prefix) if prefix else 0
        end = len(label) - len(suffix) if suffix else len(label)
        new_label = label[start:end].strip()
        cleaned.append(new_label if new_label else label.strip())
    return cleaned


def _series_with_gap_breaks(series: pd.Series, max_gap_hours: int = 1) -> pd.Series:
    """Insert NaN after temporal jumps so line plots do not bridge across large gaps."""
    s = pd.to_numeric(series, errors="coerce").copy()
    valid_idx = s.dropna().index
    if len(valid_idx) < 2:
        return s

    diffs = valid_idx[1:] - valid_idx[:-1]
    gap_positions = np.where(diffs > pd.Timedelta(hours=max_gap_hours))[0]
    if gap_positions.size == 0:
        return s

    for pos in gap_positions:
        prev_t = valid_idx[pos]
        next_t = valid_idx[pos + 1]
        delta_hours = (next_t - prev_t) / pd.Timedelta(hours=1)
        if delta_hours > max_gap_hours + 1 and (prev_t + pd.Timedelta(hours=1)) in s.index:
            s.loc[prev_t + pd.Timedelta(hours=1)] = np.nan
        else:
            s.loc[prev_t] = np.nan
    return s


def _safe_series_colors(n: int) -> list:
    """Return n colors that intentionally exclude red and nearby hues."""
    base = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#17becf",  # cyan
        "#9467bd",  # purple
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#aec7e8",  # light blue
        "#98df8a",  # light green
        "#c5b0d5",  # light purple
        "#9edae5",  # pale cyan
        "#8c564b",  # brown
        "#c7c7c7",  # light gray
    ]
    if n <= len(base):
        return base[:n]
    repeats = (n // len(base)) + 1
    return (base * repeats)[:n]


def _load_surface_offset_uncertainty_specs(repo_root: Path) -> dict:
    """
    Build per-sensor uncertainty specs for Surface plotting from offset distributions.
    Returns mapping: normalized_sensor_name -> {"mu": float, "sigma": float, "preferred": str}
    """
    summaries_root = repo_root / "data" / "output" / "calibration" / "summaries"
    if not summaries_root.exists():
        return {}

    def _norm_key(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    dist_candidates = [
        repo_root / "data" / "output" / "calibration" / "summaries" / "aggregate" / "distribution_goodness_of_fit.csv",
        repo_root / "data" / "output" / "calibration" / "aggregate" / "distribution_goodness_of_fit.csv",
    ]
    dist_csv = next((p for p in dist_candidates if p.exists()), None)
    dist_by_sensor = {}
    if dist_csv is not None:
        try:
            dist_df = pd.read_csv(dist_csv, low_memory=False)
            if "component" in dist_df.columns and "Sensor" in dist_df.columns:
                offset_df = dist_df[dist_df["component"].astype(str).str.lower() == "offset"].copy()
                for _, row in offset_df.iterrows():
                    sensor_key = _norm_key(row.get("Sensor", ""))
                    if sensor_key:
                        dist_by_sensor[sensor_key] = row
        except Exception:
            dist_by_sensor = {}

    specs = {}
    for summary_path in summaries_root.glob("*/*_uncertainty_summary.csv"):
        try:
            sdf = pd.read_csv(summary_path, low_memory=False)
            if sdf.empty:
                continue
            row = sdf.iloc[0]
            sensor = str(row.get("Sensor", summary_path.parent.name)).strip()
            sensor_key = _norm_key(sensor)
            if not sensor_key:
                continue

            mu = pd.to_numeric(row.get("Offset_Mean"), errors="coerce")
            sigma = pd.to_numeric(row.get("Offset_Std"), errors="coerce")
            preferred = str(row.get("Offset_Distribution", "normal")).strip().lower()

            fit_row = dist_by_sensor.get(sensor_key)
            if fit_row is not None:
                fit_pref = str(fit_row.get("preferred", preferred)).strip().lower()
                if fit_pref == "t":
                    t_df = pd.to_numeric(fit_row.get("t_df"), errors="coerce")
                    t_loc = pd.to_numeric(fit_row.get("t_loc"), errors="coerce")
                    t_scale = pd.to_numeric(fit_row.get("t_scale"), errors="coerce")
                    if pd.notna(t_loc):
                        mu = float(t_loc)
                    if pd.notna(t_df) and pd.notna(t_scale) and t_df > 2 and t_scale > 0:
                        sigma = float(t_scale * np.sqrt(t_df / (t_df - 2.0)))
                    preferred = "t"
                elif fit_pref in {"norm", "normal", "equivalent"}:
                    n_loc = pd.to_numeric(fit_row.get("norm_loc"), errors="coerce")
                    n_scale = pd.to_numeric(fit_row.get("norm_scale"), errors="coerce")
                    if pd.notna(n_loc):
                        mu = float(n_loc)
                    if pd.notna(n_scale) and n_scale > 0:
                        sigma = float(n_scale)
                    preferred = "normal"

            if pd.isna(mu):
                mu = 0.0
            if pd.isna(sigma) or sigma <= 0:
                continue

            specs[sensor_key] = {
                "mu": float(mu),
                "sigma": float(sigma),
                "preferred": preferred,
            }
        except Exception:
            continue
    print(f"[UNCERTAINTY] Loaded {len(specs)} surface uncertainty specs from calibration summaries.")
    return specs


def write_category_timeseries_columns(repo_root: Path) -> None:
    """
    Write one figure per category with a single column of subplots (one subplot per feature).
    Categories: Surface, Weather, SCADA, Target, Target_diff.
    """
    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    limits_csv = repo_root / "data" / "input" / "Limits.csv"
    out_dir = _summary_theme_dirs(repo_root)["timeseries_charts"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not consolidated_csv.exists():
        raise FileNotFoundError(f"Consolidated_sparse.csv not found: {consolidated_csv}")

    df = pd.read_csv(consolidated_csv, low_memory=False)
    if "TIMESTAMP" not in df.columns:
        raise KeyError("Expected TIMESTAMP column in Consolidated_sparse.csv")

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df[df["TIMESTAMP"].notna()].sort_values("TIMESTAMP").copy()
    if df.empty:
        return

    start_date = pd.Timestamp("2021-12-15")
    end_date = pd.Timestamp("2025-03-01")
    df = df[(df["TIMESTAMP"] >= start_date) & (df["TIMESTAMP"] <= end_date)].copy()
    if df.empty:
        return

    df = df.set_index("TIMESTAMP")
    all_cols = [c for c in df.columns if c != "Interpolated" and not c.endswith("_state")]

    def _norm_col(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    all_cols_by_norm = {_norm_col(c): c for c in all_cols}
    surface_uncertainty_specs = _load_surface_offset_uncertainty_specs(repo_root)
    limits_records = load_limits_records(limits_csv)
    all_limit_specs = map_limits_to_columns(all_cols, limits_records)

    weather_cols_expected = [
        "Wind speed x (m/s)",
        "Wind speed y (m/s)",
        "Maximum 3s wind gust (m/s)",
        "Atmospheric pressure (mBar)",
        "Longwave (IR) radiation (W/m2)",
        "Shortwave (solar) radiation (W/m2)",
        "24hr precipitation total (mm)",
        "Air temperature (Â°C)",
        "Humidity (%)",
    ]

    # Match weather columns robustly to minor case/encoding/unit-text differences.
    weather_norm_candidates = {_norm_col(c) for c in weather_cols_expected}
    weather_norm_candidates.update({
        _norm_col("Air Temperature (C)"),
        _norm_col("Air Temperature (°C)"),
        _norm_col("Air temperature (C)"),
    })
    weather_cols = [all_cols_by_norm[k] for k in all_cols_by_norm if k in weather_norm_candidates]

    surface_cols = [c for c in all_cols if c.startswith("Pfl - ")]
    scada_cols = [c for c in all_cols if c.startswith("SCADA - ")]
    target_cols = [c for c in all_cols if (not c.endswith("_res") and c not in weather_cols and not c.startswith("Pfl - ") and not c.startswith("SCADA - "))]
    target_diff_cols = [c for c in all_cols if c.endswith("_res")]

    # Enforce identical ordering between Target and Target_diff:
    # 1) descending valid count
    # 2) alphabetical by label (base target name) as tie-break.
    target_by_base = {c: c for c in target_cols}
    target_diff_by_base = {(c[:-4] if c.endswith("_res") else c): c for c in target_diff_cols}
    base_names = sorted(set(target_by_base.keys()).union(set(target_diff_by_base.keys())))

    base_valid_counts = {}
    for base in base_names:
        src_col = target_by_base.get(base, target_diff_by_base.get(base))
        if src_col is None:
            base_valid_counts[base] = 0
        else:
            base_valid_counts[base] = int(pd.to_numeric(df[src_col], errors="coerce").notna().sum())

    ordered_base_names = sorted(base_names, key=lambda b: (-base_valid_counts.get(b, 0), b.lower()))
    ordered_target_cols = [target_by_base[b] for b in ordered_base_names if b in target_by_base]
    ordered_target_diff_cols = [target_diff_by_base[b] for b in ordered_base_names if b in target_diff_by_base]

    category_columns = {
        "Surface": surface_cols,
        "Weather": weather_cols,
        "SCADA": scada_cols,
        "Target": ordered_target_cols,
        "Target_diff": ordered_target_diff_cols,
    }
    predictor_cols_for_split = [c for c in all_cols if c not in ordered_target_cols and c not in ordered_target_diff_cols]
    window = 168
    first_valid_idx = window - 1
    if predictor_cols_for_split:
        predictor_valid_for_split = pd.DataFrame(
            {c: _valid_nonblank_mask(df[c]) for c in predictor_cols_for_split},
            index=df.index,
        )
    else:
        predictor_valid_for_split = pd.DataFrame(index=df.index)

    def _predictor_valid_count_for_overlay(idx: int) -> int:
        if predictor_valid_for_split.empty or idx < first_valid_idx:
            return 0
        w = predictor_valid_for_split.iloc[idx - (window - 1): idx + 1].to_numpy(dtype=bool)
        return int(np.count_nonzero(w))

    label_map = {}
    global_max_label_line_len = 0
    for figure_name, cols in category_columns.items():
        if not cols:
            continue
        labels = cols.copy()
        if figure_name == "Surface":
            labels = [l[len("Pfl - "):] if l.startswith("Pfl - ") else l for l in labels]
        if figure_name == "SCADA":
            labels = [l[len("SCADA - "):] if l.startswith("SCADA - ") else l for l in labels]
        if figure_name == "Target_diff":
            labels = [l[:-4] if l.endswith("_res") else l for l in labels]
        labels = _strip_common_affixes(labels)
        wrapped_labels = [textwrap.fill(lbl, width=17) for lbl in labels]
        label_map[figure_name] = wrapped_labels
        for lbl in wrapped_labels:
            line_max = max(len(line) for line in lbl.splitlines()) if lbl else 0
            if line_max > global_max_label_line_len:
                global_max_label_line_len = line_max

    # Shared layout constants across all category figures:
    # fixed width + fixed left/right margins => identical absolute subplot width.
    fig_width = 13.0
    row_height = 0.88
    min_fig_height = 2.8
    font_size = 10
    y_label_font_size = int(round(font_size * 1.5))
    x_label_font_size = int(round(font_size * 1.5))
    y_value_font_size = font_size
    y_label_pad = 8
    # Shared figure margin: estimate required left inches from wrapped label width.
    est_char_width_in = max(0.045, 0.0055 * y_label_font_size)
    left_margin_in = (global_max_label_line_len * est_char_width_in) + 0.55
    shared_left_margin = left_margin_in / fig_width
    shared_left_margin = max(0.14, min(0.26, shared_left_margin))
    shared_right_margin = 0.995
    top_margin_in = 0.10
    bottom_margin_in = 0.90
    shared_hspace = 0.08

    def _write_two_col_target_variant(
        figure_name: str,
        cols: list,
        wrapped_labels: list,
        with_overlay: bool,
        out_path: Path,
    ) -> None:
        if not cols:
            return
        n_series = len(cols)
        n_per_col = int(np.ceil(n_series / 2.0))
        fig_h = max(min_fig_height, row_height * n_per_col)
        fig_w = fig_width * 2.0
        fig = plt.figure(figsize=(fig_w, fig_h))

        # Two independent column panels in one canvas.
        panel_gap = 0.07
        left_margin_in = shared_left_margin * fig_width
        right_margin_in = (1.0 - shared_right_margin) * fig_width
        left_margin = max(0.04, min(0.18, left_margin_in / fig_w))
        right_margin = max(0.90, min(0.995, 1.0 - (right_margin_in / fig_w)))
        top_margin = max(0.86, min(0.995, 1.0 - (top_margin_in / fig_h)))
        bottom_margin = max(0.08, min(0.30, bottom_margin_in / fig_h))

        outer = fig.add_gridspec(
            1, 2,
            left=left_margin,
            right=right_margin,
            top=top_margin,
            bottom=bottom_margin,
            wspace=panel_gap,
        )
        gs_left = outer[0, 0].subgridspec(n_per_col, 1, hspace=shared_hspace)
        gs_right = outer[0, 1].subgridspec(n_per_col, 1, hspace=shared_hspace)

        axes_left = [fig.add_subplot(gs_left[r, 0]) for r in range(n_per_col)]
        axes_right = [fig.add_subplot(gs_right[r, 0]) for r in range(n_per_col)]
        all_axes = [axes_left, axes_right]

        palette = _safe_series_colors(n_series)
        overlay_spans = {}

        for k, (col, label) in enumerate(zip(cols, wrapped_labels)):
            panel_idx = 0 if k < n_per_col else 1
            row_idx = k if k < n_per_col else (k - n_per_col)
            ax = all_axes[panel_idx][row_idx]

            series = pd.to_numeric(df[col], errors="coerce")
            series_color = palette[k]
            y_low = None
            y_high = None
            finite_vals = series.to_numpy(dtype=float)
            finite_vals = finite_vals[np.isfinite(finite_vals)]
            if finite_vals.size > 0:
                y_low = float(np.min(finite_vals))
                y_high = float(np.max(finite_vals))

            if figure_name == "Target_diff":
                base_target_col = col[:-4] if col.endswith("_res") else col
                target_col_match = all_cols_by_norm.get(_norm_col(base_target_col))
                exceed_mask = pd.Series(False, index=series.index)
                if target_col_match in all_limit_specs:
                    target_series = pd.to_numeric(df[target_col_match], errors="coerce")
                    limit_spec = all_limit_specs.get(target_col_match)
                    upper_limit, lower_limit = _limit_bounds(limit_spec)
                    exceed_mask = pd.Series(
                        limit_exceedance_mask(target_series, upper=upper_limit, lower=lower_limit),
                        index=series.index,
                    )

                valid_mask = series.notna()
                normal_mask = valid_mask & (~exceed_mask.reindex(series.index, fill_value=False))
                exceed_plot_mask = valid_mask & exceed_mask.reindex(series.index, fill_value=False)
                ax.plot(
                    series.index[normal_mask], series.values[normal_mask],
                    linestyle="", marker="o", markersize=4.5, color=series_color,
                    markeredgecolor=("black" if with_overlay else series_color),
                    markeredgewidth=(0.35 if with_overlay else 0.0), alpha=0.95,
                )
                ax.plot(
                    series.index[exceed_plot_mask], series.values[exceed_plot_mask],
                    linestyle="", marker="x", markersize=4.0, markeredgewidth=0.9, color="#ff0000",
                )
            else:
                ax.plot(
                    series.index, series.values,
                    linestyle="", marker="o", markersize=4.5, color=series_color,
                    markeredgecolor=("black" if with_overlay else series_color),
                    markeredgewidth=(0.35 if with_overlay else 0.0), alpha=0.95,
                )

            if figure_name == "Target" and col in all_limit_specs:
                upper_limit, lower_limit = _limit_bounds(all_limit_specs[col])
                if upper_limit is not None:
                    ax.axhline(y=upper_limit, color="#ff0000", linewidth=0.7, linestyle="-", zorder=1.2)
                    if y_low is None or y_high is None:
                        y_low = upper_limit
                        y_high = upper_limit
                    else:
                        y_low = min(y_low, upper_limit)
                        y_high = max(y_high, upper_limit)
                if lower_limit is not None:
                    ax.axhline(y=lower_limit, color="#2ca02c", linewidth=0.7, linestyle="-", zorder=1.2)
                    if y_low is None or y_high is None:
                        y_low = lower_limit
                        y_high = lower_limit
                    else:
                        y_low = min(y_low, lower_limit)
                        y_high = max(y_high, lower_limit)

            if y_low is not None and y_high is not None:
                y_span = y_high - y_low
                if y_span <= 0:
                    y_pad = 0.12 * max(abs(y_low), 1.0)
                else:
                    y_pad = max(0.08 * y_span, 0.03 * max(abs(y_low), abs(y_high), 1.0))
                ax.set_ylim(y_low - y_pad, y_high + y_pad)

            if with_overlay:
                valid_idx = np.where(series.notna().to_numpy() & (np.arange(len(series)) >= first_valid_idx))[0]
                n_valid = len(valid_idx)
                if n_valid > 0:
                    valid_counts = [_predictor_valid_count_for_overlay(int(i)) for i in valid_idx]
                    split_cut, _, _ = _split_index_from_cumulative_counts(valid_counts, train_fraction=0.8)
                    split_cut = max(1, min(split_cut, n_valid))
                    first_start = series.index[valid_idx[0]] - pd.Timedelta(minutes=30)
                    first_end = series.index[valid_idx[split_cut - 1]]
                    second_start = None
                    second_end = None
                    if split_cut < n_valid:
                        second_start = series.index[valid_idx[split_cut]] - pd.Timedelta(minutes=30)
                        second_end = series.index[valid_idx[-1]] + pd.Timedelta(minutes=30)
                    overlay_spans[(panel_idx, row_idx)] = (first_start, first_end, second_start, second_end)

            ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=y_label_font_size, labelpad=y_label_pad)
            ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.4)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
            ax.tick_params(axis="y", labelsize=font_size)
            ax.tick_params(axis="x", labelsize=font_size)

        # Hide empty rows in right panel (if odd count)
        for r in range(n_per_col):
            k = n_per_col + r
            if k >= n_series:
                axes_right[r].axis("off")

        # X-axis formatting within each independent panel
        quarter_ticks = pd.date_range(start=start_date, end=end_date, freq="QS-JAN")
        for panel_idx, panel_axes in enumerate(all_axes):
            last_active = -1
            for r in range(n_per_col):
                k = r if panel_idx == 0 else (n_per_col + r)
                if k < n_series:
                    last_active = r
            if last_active < 0:
                continue
            for r, ax in enumerate(panel_axes):
                k = r if panel_idx == 0 else (n_per_col + r)
                if k >= n_series:
                    continue
                if r < last_active:
                    ax.tick_params(axis="x", which="both", labelbottom=False)
                else:
                    ax.set_xticks(quarter_ticks)
                    ax.set_xticklabels(
                        [dt.strftime("%Y-%m-%d") for dt in quarter_ticks],
                        rotation=25, ha="right", fontsize=font_size
                    )
                    ax.set_xlim(start_date, end_date)

        if with_overlay:
            for (panel_idx, row_idx), span_info in overlay_spans.items():
                ax = all_axes[panel_idx][row_idx]
                first_start, first_end, second_start, second_end = span_info
                ax.axvspan(first_start, first_end, ymin=0.03, ymax=0.97, color="#66bb66", alpha=0.24, zorder=0.3)
                if second_start is not None and second_end is not None:
                    ax.axvspan(second_start, second_end, ymin=0.03, ymax=0.97, color="#f4a3c2", alpha=0.24, zorder=0.3)

        # Center divider like two figures stitched together.
        fig.add_artist(plt.Line2D([0.5, 0.5], [bottom_margin, top_margin], color="#d0d0d0", linewidth=1.0, alpha=0.9))
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Wrote chart to: {out_path}")

    for figure_name, cols in category_columns.items():
        if not cols:
            continue
        wrapped_labels = label_map[figure_name]

        variant_specs = [("timeseries", False)]
        if figure_name == "Surface":
            variant_specs.append(("timeseries_uncertainty", True))

        for variant_suffix, include_surface_uncertainty in variant_specs:
            n_rows = len(cols)
            fig_h = max(min_fig_height, row_height * n_rows)
            fig, axes = plt.subplots(
                n_rows,
                1,
                sharex=True,
                figsize=(fig_width, fig_h),
                gridspec_kw={"hspace": shared_hspace},
            )
            if n_rows == 1:
                axes = [axes]
            palette = _safe_series_colors(n_rows)
            overlay_spans = []

            for i, (ax, col, label) in enumerate(zip(axes, cols, wrapped_labels)):
                series = pd.to_numeric(df[col], errors="coerce")
                series_color = palette[i]
                y_low = None
                y_high = None
                numeric_vals = series.to_numpy(dtype=float)
                finite_vals = numeric_vals[np.isfinite(numeric_vals)]
                if finite_vals.size > 0:
                    y_low = float(np.min(finite_vals))
                    y_high = float(np.max(finite_vals))
                if figure_name in {"Target", "Target_diff"}:
                    if figure_name == "Target_diff":
                        base_target_col = col[:-4] if col.endswith("_res") else col
                        target_col_match = all_cols_by_norm.get(_norm_col(base_target_col))
                        exceed_mask = pd.Series(False, index=series.index)
                        if target_col_match in all_limit_specs:
                            target_series = pd.to_numeric(df[target_col_match], errors="coerce")
                            limit_spec = all_limit_specs.get(target_col_match)
                            upper_limit, lower_limit = _limit_bounds(limit_spec)
                            exceed_mask = pd.Series(
                                limit_exceedance_mask(target_series, upper=upper_limit, lower=lower_limit),
                                index=series.index,
                            )

                        valid_mask = series.notna()
                        normal_mask = valid_mask & (~exceed_mask.reindex(series.index, fill_value=False))
                        exceed_plot_mask = valid_mask & exceed_mask.reindex(series.index, fill_value=False)

                        ax.plot(
                            series.index[normal_mask],
                            series.values[normal_mask],
                            linestyle="",
                            marker="o",
                            markersize=4.5,
                            color=series_color,
                            markeredgecolor=series_color,
                            markeredgewidth=0.0,
                            alpha=0.95,
                        )
                        ax.plot(
                            series.index[exceed_plot_mask],
                            series.values[exceed_plot_mask],
                            linestyle="",
                            marker="x",
                            markersize=4.0,
                            markeredgewidth=0.9,
                            color="#ff0000",
                        )
                    else:
                        ax.plot(
                            series.index,
                            series.values,
                            linestyle="",
                            marker="o",
                            markersize=4.5,
                            color=series_color,
                            markeredgecolor=series_color,
                            markeredgewidth=0.0,
                            alpha=0.95,
                        )
                    if figure_name == "Target" and col in all_limit_specs:
                        upper_limit, lower_limit = _limit_bounds(all_limit_specs[col])
                        if upper_limit is not None:
                            ax.axhline(
                                y=upper_limit,
                                color="#ff0000",
                                linewidth=0.7,
                                linestyle="-",
                                zorder=1.2,
                            )
                            if y_low is None or y_high is None:
                                y_low = upper_limit
                                y_high = upper_limit
                            else:
                                y_low = min(y_low, upper_limit)
                                y_high = max(y_high, upper_limit)
                        if lower_limit is not None:
                            ax.axhline(
                                y=lower_limit,
                                color="#2ca02c",
                                linewidth=0.7,
                                linestyle="-",
                                zorder=1.2,
                            )
                            if y_low is None or y_high is None:
                                y_low = lower_limit
                                y_high = lower_limit
                            else:
                                y_low = min(y_low, lower_limit)
                                y_high = max(y_high, lower_limit)

                    # Ensure threshold (if present) is always in-frame and add clear margin.
                    if y_low is not None and y_high is not None:
                        y_span = y_high - y_low
                        if y_span <= 0:
                            base = max(abs(y_low), 1.0)
                            y_pad = 0.12 * base
                        else:
                            y_pad = max(0.08 * y_span, 0.03 * max(abs(y_low), abs(y_high), 1.0))
                        ax.set_ylim(y_low - y_pad, y_high + y_pad)
                else:
                    broken = _series_with_gap_breaks(series, max_gap_hours=1)
                    if figure_name == "Surface" and include_surface_uncertainty:
                        base_surface_name = col[len("Pfl - "):] if col.startswith("Pfl - ") else col
                        spec = surface_uncertainty_specs.get(_norm_col(base_surface_name))
                        if spec is not None:
                            mu = float(spec["mu"])
                            sigma = float(spec["sigma"])
                            center = broken + mu
                            finite_center = pd.to_numeric(center, errors="coerce").to_numpy(dtype=float)
                            finite_center = finite_center[np.isfinite(finite_center)]
                            center_span = float(np.nanmax(finite_center) - np.nanmin(finite_center)) if finite_center.size > 0 else np.nan
                            rel_2sigma = (2.0 * sigma / center_span) if np.isfinite(center_span) and center_span > 0 else np.nan
                            print(
                                f"[UNCERTAINTY] Surface '{base_surface_name}': mu={mu:.6g}, sigma={sigma:.6g}, "
                                f"2sigma={2.0 * sigma:.6g}, center_span={center_span:.6g}, rel_2sigma={rel_2sigma:.6g}"
                            )
                            ax.fill_between(
                                broken.index,
                                center - (2.0 * sigma),
                                center + (2.0 * sigma),
                                color=series_color,
                                alpha=0.24,
                                linewidth=0.0,
                                zorder=0.6,
                            )
                            ax.fill_between(
                                broken.index,
                                center - sigma,
                                center + sigma,
                                color=series_color,
                                alpha=0.34,
                                linewidth=0.0,
                                zorder=0.7,
                            )
                        else:
                            print(f"[UNCERTAINTY] Surface '{base_surface_name}': no matching uncertainty spec found.")
                    ax.plot(
                        broken.index,
                        broken.values,
                        linestyle="-",
                        linewidth=0.45,
                        color=series_color,
                        alpha=0.85,
                        zorder=1.4,
                    )

                span_info = None
                if figure_name in {"Target", "Target_diff"}:
                    valid_idx = np.where(series.notna().to_numpy() & (np.arange(len(series)) >= first_valid_idx))[0]
                    n_valid = len(valid_idx)
                    if n_valid > 0:
                        valid_counts = [_predictor_valid_count_for_overlay(int(i)) for i in valid_idx]
                        split_cut, _, _ = _split_index_from_cumulative_counts(valid_counts, train_fraction=0.8)
                        split_cut = max(1, min(split_cut, n_valid))
                        train_last_pos = split_cut - 1
                        first_start = series.index[valid_idx[0]] - pd.Timedelta(minutes=30)
                        first_end = series.index[valid_idx[train_last_pos]]
                        second_start = None
                        second_end = None
                        if split_cut < n_valid:
                            second_start = series.index[valid_idx[split_cut]] - pd.Timedelta(minutes=30)
                            second_end = series.index[valid_idx[-1]] + pd.Timedelta(minutes=30)
                        span_info = (first_start, first_end, second_start, second_end)
                overlay_spans.append(span_info)

                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=y_label_font_size, labelpad=y_label_pad)
                ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.4)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
                if i < n_rows - 1:
                    ax.tick_params(axis="x", which="both", labelbottom=False)

            quarter_ticks = pd.date_range(start=start_date, end=end_date, freq="QS-JAN")
            axes[-1].set_xticks(quarter_ticks)
            axes[-1].set_xticklabels([dt.strftime("%Y-%m-%d") for dt in quarter_ticks], rotation=25, ha="right", fontsize=x_label_font_size)
            axes[-1].set_xlim(start_date, end_date)

            for ax in axes:
                ax.tick_params(axis="y", labelsize=y_value_font_size)
                ax.tick_params(axis="x", labelsize=x_label_font_size)

            # Use absolute-inch margins so short figures get enough label room
            # while taller figures avoid excessive whitespace.
            shared_top_margin = max(0.86, min(0.995, 1.0 - (top_margin_in / fig_h)))
            shared_bottom_margin = max(0.08, min(0.30, bottom_margin_in / fig_h))

            fig.subplots_adjust(
                left=shared_left_margin,
                right=shared_right_margin,
                top=shared_top_margin,
                bottom=shared_bottom_margin,
                hspace=shared_hspace,
            )
            out_path = out_dir / f"{figure_name}_{variant_suffix}.png"
            fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
            if figure_name in {"Target", "Target_diff"} and variant_suffix == "timeseries":
                # Overlay variant: restore marker borders for readability on shaded backgrounds.
                for ax in axes:
                    for line in ax.get_lines():
                        if str(line.get_marker()) == "o":
                            line.set_markeredgecolor("black")
                            line.set_markeredgewidth(0.35)
                for ax, span_info in zip(axes, overlay_spans):
                    if span_info is None:
                        continue
                    first_start, first_end, second_start, second_end = span_info
                    ax.axvspan(
                        first_start,
                        first_end,
                        ymin=0.03,
                        ymax=0.97,
                        color="#66bb66",
                        alpha=0.24,
                        zorder=0.3,
                    )
                    if second_start is not None and second_end is not None:
                        ax.axvspan(
                            second_start,
                            second_end,
                            ymin=0.03,
                            ymax=0.97,
                            color="#f4a3c2",
                            alpha=0.24,
                            zorder=0.3,
                        )
                overlay_out_path = out_dir / f"{figure_name}_timeseries_overlay.png"
                fig.savefig(overlay_out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            print(f"Wrote chart to: {out_path}")
            if figure_name in {"Target", "Target_diff"} and variant_suffix == "timeseries":
                print(f"Wrote chart to: {overlay_out_path}")


def generate_sensor_summary(repo_root: Path) -> list:
    """Generate FullHourly_summary.csv from profiler sensor (FullHourly.csv) data."""
    input_csv = repo_root / "data" / "input" / "sensors" / "FullHourly.csv"
    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    dirs = _summary_theme_dirs(repo_root)
    out_tables = dirs["tables"]
    out_charts = dirs["availability_charts"]

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    if not consolidated_csv.exists():
        raise FileNotFoundError(f"Consolidated_sparse.csv not found: {consolidated_csv}")

    sensor_map = {
        "sensorParms(1)": "Pfl - Water temperature (°C)",
        "sensorParms(2)": "Pfl - Cond (microS_cm)",
        "sensorParms(3)": "Pfl - Sp Cond (microS_cm)",
        "sensorParms(4)": "Pfl - Salinity (ppt)",
        "sensorParms(5)": "Pfl - pH",
        "sensorParms(6)": "Pfl - DO (% Sat)",
        "sensorParms(7)": "Pfl - Turbidity (NTU)",
        "sensorParms(8)": "Pfl - Turbidity (FNU)",
        "sensorParms(9)": "Pfl - Vertical position (m)",
        "sensorParms(10)": "Pfl - fDOM (RFU)",
        "sensorParms(11)": "Pfl - fDOM (QSU)",
    }

    df = pd.read_csv(input_csv, low_memory=False).rename(columns=sensor_map)
    consolidated_df = pd.read_csv(consolidated_csv, low_memory=False)

    rows = []
    coverage_entries = []
    for key, sensor_name in sensor_map.items():
        name = sensor_name[len("Pfl - "):] if sensor_name.startswith("Pfl - ") else sensor_name
        m = re.match(r"(.+?)\s*\(([^)]+)\)$", name)
        base_name, unit = (m.group(1).strip(), m.group(2).strip()) if m else (name.strip(), "")

        col_candidates = [
            c for c in consolidated_df.columns
            if c.strip() in (sensor_name, name, base_name)
        ]
        if not col_candidates:
            col_candidates = [
                c for c in consolidated_df.columns
                if c.strip().lower().replace(" ", "") == base_name.lower().replace(" ", "")
            ]

        if not col_candidates:
            continue

        col = col_candidates[0]
        coverage_entries.append({"dataset": "FullHourly", "label": base_name, "column": col})
        valid_mask = _valid_nonblank_mask(consolidated_df[col])
        valid_count = valid_mask.sum()
        total_rows = len(consolidated_df)
        pct = (valid_count / total_rows * 100) if total_rows > 0 else 0.0
        mean_cons, std_cons = _mean_std_valid_numeric(consolidated_df[col], valid_mask)
        mean_cons_norm01, std_cons_norm01 = _mean_std_valid_numeric_normalized_01(consolidated_df[col], valid_mask)

        valid_count_full = 0
        mean_full = None
        std_full = None
        if sensor_name in df.columns:
            valid_mask_full = _valid_nonblank_mask(df[sensor_name])
            valid_count_full = valid_mask_full.sum()
            mean_full, std_full = _mean_std_valid_numeric(df[sensor_name], valid_mask_full)

        rows.append({
            "sensor": base_name,
            "unit": unit,
            "valid_count_consolidated": int(valid_count),
            "percent_valid_consolidated": round(pct, 2),
            "avg_valid_consolidated": round(mean_cons, 4) if mean_cons is not None else None,
            "std_valid_consolidated": round(std_cons, 4) if std_cons is not None else None,
            "avg_valid_consolidated_norm01": round(mean_cons_norm01, 4) if mean_cons_norm01 is not None else None,
            "std_valid_consolidated_norm01": round(std_cons_norm01, 4) if std_cons_norm01 is not None else None,
            "valid_count_fullhourly": int(valid_count_full),
            "avg_valid_fullhourly": round(mean_full, 4) if mean_full is not None else None,
            "std_valid_fullhourly": round(std_full, 4) if std_full is not None else None,
        })

    out_path = out_tables / "FullHourly_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")
    _write_clustered_bar_chart(summary_df, out_charts, "FullHourly", include_exceed_bar=False)
    return coverage_entries


def generate_weather_summary(repo_root: Path) -> list:
    """Generate Weather_summary.csv from weather station (Weather.csv) data."""
    weather_csv = repo_root / "data" / "input" / "sensors" / "Weather.csv"
    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    dirs = _summary_theme_dirs(repo_root)
    out_tables = dirs["tables"]
    out_charts = dirs["availability_charts"]

    if not weather_csv.exists():
        raise FileNotFoundError(f"Weather.csv not found: {weather_csv}")
    if not consolidated_csv.exists():
        raise FileNotFoundError(f"Consolidated_sparse.csv not found: {consolidated_csv}")

    full_weather_columns = {
        "1818_time: AA[mBar]": "Instantaneous atmospheric pressure (mBar)",
        "1818_time: DD Retning[°]": "Wind direction 10minRollingAvg (°)",
        "1818_time: DX_l[°]": "Hourly average wind direction (°)",
        "1818_time: FF Hastighet[m/s]": "Average wind speed (m/s)",
        "1818_time: FG_l[m/s]": "Maximum sustained wind speed, 3-second span (m/s)",
        "1818_time: FG_tid_l[N/A]": "Time of maximum 3s Gust",
        "1818_time: FX Kast[m/s]": "Maximum sustained wind speed, 10-minute span (m/s)",
        "1818_time: FX_tid_l[N/A]": "Time of maximum 10 minute gust",
        "1818_time: PO Trykk stasjonshøyde[mBar]": "Hourly average atmospheric pressure at station (mBar)",
        "1818_time: PP[mBar]": "Maximum pressure differential, 3-hour span (mBar)",
        "1818_time: PR Trykk redusert til havnivå[mBar]": "Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)",
        "1818_time: QLI Langbølget[W/m2]": "Longwave (IR) radiation (W/m2)",
        "1818_time: QNH[mBar]": "Instantaneous sea-level atmospheric pressure (mBar)",
        "1818_time: QSI Kortbølget[W/m2]": "Shortwave (solar) radiation (W/m2)",
        "1818_time: RR_1[mm]": "Precipitation (mm/hr)",
        "1818_time: TA Middel[°C]": "Instantaneous temperature (°C)",
        "1818_time: TA_a_Max[°C]": "Maximum temperature (°C)",
        "1818_time: TA_a_Min[°C]": "Minimum temperature (°C)",
        "1818_time: UU Luftfuktighet[%RH]": "Average humidity (% relative humidity)",
    }

    # Columns in Consolidated_sparse.csv that are derived from weather sources
    derived_weather_mappings = {
        "Wind speed x (m/s)": ["1818_time: DX_l[°]", "1818_time: FF Hastighet[m/s]"],
        "Wind speed y (m/s)": ["1818_time: DX_l[°]", "1818_time: FF Hastighet[m/s]"],
        "Maximum 3s wind gust (m/s)": ["1818_time: FG_l[m/s]"],
        "Atmospheric pressure (mBar)": ["1818_time: PR Trykk redusert til havnivå[mBar]"],
        "24hr precipitation total (mm)": ["1818_time: RR_1[mm]"],
        "Air temperature (°C)": ["1818_time: TA Middel[°C]"],
        "Humidity (%)": ["1818_time: UU Luftfuktighet[%RH]"],
    }

    weather_df = pd.read_csv(weather_csv, sep=";", decimal=",", low_memory=False)
    weather_cols = [c for c in weather_df.columns if c != "Time"]
    consolidated_df = pd.read_csv(consolidated_csv, low_memory=False)
    consolidated_english_names = {c.strip(): c for c in consolidated_df.columns}

    rows = []
    coverage_entries = []
    mapped_consolidated_cols = set()

    for col in weather_cols:
        english_name = full_weather_columns.get(col, col).replace('"', "").replace("'", "")
        m = re.match(r"(.+?)\s*\(([^)]+)\)$", english_name)
        base_name, unit = (m.group(1).strip(), m.group(2).strip()) if m else (english_name.strip(), "")

        valid_mask_weather = _valid_nonblank_mask(weather_df[col])
        valid_count_weather = valid_mask_weather.sum()
        total_weather = len(weather_df)
        pct_weather = (valid_count_weather / total_weather * 100) if total_weather > 0 else 0.0
        mean_weather, std_weather = _mean_std_valid_numeric(weather_df[col], valid_mask_weather)

        col_candidates = [
            c for c in consolidated_df.columns
            if c.strip() in (base_name, english_name)
        ]
        if not col_candidates:
            col_candidates = [
                c for c in consolidated_df.columns
                if c.strip().lower().replace(" ", "") == base_name.lower().replace(" ", "")
            ]
        for derived_name, sources in derived_weather_mappings.items():
            if col in sources and derived_name in consolidated_english_names:
                col_candidates.append(derived_name)
        col_candidates = list(dict.fromkeys(col_candidates))

        for c in col_candidates:
            if c in mapped_consolidated_cols:
                continue
            mapped_consolidated_cols.add(c)
            label = base_name if c in (base_name, english_name) else c
            coverage_entries.append({"dataset": "Weather", "label": label, "column": c})
            valid_mask_consolidated = _valid_nonblank_mask(consolidated_df[c])
            valid_count_consolidated = valid_mask_consolidated.sum()
            total_consolidated = len(consolidated_df)
            pct_consolidated = (valid_count_consolidated / total_consolidated * 100) if total_consolidated > 0 else 0.0
            mean_cons, std_cons = _mean_std_valid_numeric(consolidated_df[c], valid_mask_consolidated)
            mean_cons_norm01, std_cons_norm01 = _mean_std_valid_numeric_normalized_01(consolidated_df[c], valid_mask_consolidated)
            rows.append({
                "parameter": label,
                "unit": unit,
                "valid_count_weather": int(valid_count_weather),
                "percent_valid_weather": round(pct_weather, 2),
                "avg_valid_weather": round(mean_weather, 4) if mean_weather is not None else None,
                "std_valid_weather": round(std_weather, 4) if std_weather is not None else None,
                "valid_count_consolidated": int(valid_count_consolidated),
                "percent_valid_consolidated": round(pct_consolidated, 2),
                "avg_valid_consolidated": round(mean_cons, 4) if mean_cons is not None else None,
                "std_valid_consolidated": round(std_cons, 4) if std_cons is not None else None,
                "avg_valid_consolidated_norm01": round(mean_cons_norm01, 4) if mean_cons_norm01 is not None else None,
                "std_valid_consolidated_norm01": round(std_cons_norm01, 4) if std_cons_norm01 is not None else None,
            })

    # Warn about any consolidated weather columns not captured
    col_list = list(consolidated_df.columns)
    try:
        start_idx = col_list.index("Wind speed x (m/s)")
        end_idx = col_list.index("Humidity (%)")
        for c in col_list[start_idx:end_idx + 1]:
            if c not in mapped_consolidated_cols:
                print(f"[WARN] No mapping found for consolidated column '{c}' in Weather_summary.csv.")
    except ValueError:
        pass

    out_path = out_tables / "Weather_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")
    _write_clustered_bar_chart(summary_df, out_charts, "Weather", include_exceed_bar=False)
    return coverage_entries


def generate_scada_summary(repo_root: Path) -> list:
    """Generate SCADA_summary.csv from SCADA sensor (SCADA.csv) data."""
    scada_csv = repo_root / "data" / "input" / "sensors" / "SCADA.csv"
    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    dirs = _summary_theme_dirs(repo_root)
    out_tables = dirs["tables"]
    out_charts = dirs["availability_charts"]

    if not scada_csv.exists():
        raise FileNotFoundError(f"SCADA.csv not found: {scada_csv}")
    if not consolidated_csv.exists():
        raise FileNotFoundError(f"Consolidated_sparse.csv not found: {consolidated_csv}")

    scada_df = pd.read_csv(scada_csv, sep=";", decimal=".", low_memory=False)
    scada_cols = [c for c in scada_df.columns if c != "Time"]
    consolidated_df = pd.read_csv(consolidated_csv, low_memory=False)

    rows = []
    coverage_entries = []
    mapped_consolidated_cols = set()

    for col in scada_cols:
        base_name = col[len("SCADA - "):] if col.startswith("SCADA - ") else col
        base_name = base_name.replace('"', "").replace("'", "").strip()
        m = re.match(r"(.+?)\s*\(([^)]+)\)$", base_name)
        param_name, unit = (m.group(1).strip(), m.group(2).strip()) if m else (base_name, "")

        valid_mask_scada = _valid_nonblank_mask(scada_df[col])
        valid_count_scada = valid_mask_scada.sum()
        total_scada = len(scada_df)
        pct_scada = (valid_count_scada / total_scada * 100) if total_scada > 0 else 0.0
        mean_scada, std_scada = _mean_std_valid_numeric(scada_df[col], valid_mask_scada)

        col_candidates = [
            c for c in consolidated_df.columns
            if c.strip() in (param_name, base_name, col)
        ]
        if not col_candidates:
            col_candidates = [
                c for c in consolidated_df.columns
                if c.strip().lower().replace(" ", "") in (
                    param_name.lower().replace(" ", ""),
                    base_name.lower().replace(" ", ""),
                    col.lower().replace(" ", ""),
                )
            ]

        if not col_candidates:
            continue

        c = col_candidates[0]
        mapped_consolidated_cols.add(c)
        coverage_entries.append({"dataset": "SCADA", "label": param_name, "column": c})
        valid_mask_consolidated = _valid_nonblank_mask(consolidated_df[c])
        valid_count_consolidated = valid_mask_consolidated.sum()
        total_consolidated = len(consolidated_df)
        pct_consolidated = (valid_count_consolidated / total_consolidated * 100) if total_consolidated > 0 else 0.0
        mean_cons, std_cons = _mean_std_valid_numeric(consolidated_df[c], valid_mask_consolidated)
        mean_cons_norm01, std_cons_norm01 = _mean_std_valid_numeric_normalized_01(consolidated_df[c], valid_mask_consolidated)

        rows.append({
            "parameter": param_name,
            "unit": unit,
            "valid_count_scada": int(valid_count_scada),
            "percent_valid_scada": round(pct_scada, 2),
            "avg_valid_scada": round(mean_scada, 4) if mean_scada is not None else None,
            "std_valid_scada": round(std_scada, 4) if std_scada is not None else None,
            "valid_count_consolidated": int(valid_count_consolidated),
            "percent_valid_consolidated": round(pct_consolidated, 2),
            "avg_valid_consolidated": round(mean_cons, 4) if mean_cons is not None else None,
            "std_valid_consolidated": round(std_cons, 4) if std_cons is not None else None,
            "avg_valid_consolidated_norm01": round(mean_cons_norm01, 4) if mean_cons_norm01 is not None else None,
            "std_valid_consolidated_norm01": round(std_cons_norm01, 4) if std_cons_norm01 is not None else None,
        })

    out_path = out_tables / "SCADA_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")
    _write_clustered_bar_chart(summary_df, out_charts, "SCADA", include_exceed_bar=False)
    return coverage_entries


def generate_eurofins_summary(repo_root: Path) -> list:
    """Generate Eurofins_summary.csv from lab measurement (Eurofins.csv) data."""
    eurofins_csv = repo_root / "data" / "input" / "sensors" / "Eurofins.csv"
    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    limits_csv = repo_root / "data" / "input" / "Limits.csv"
    dirs = _summary_theme_dirs(repo_root)
    out_tables = dirs["tables"]
    out_charts = dirs["availability_charts"]

    if not eurofins_csv.exists():
        raise FileNotFoundError(f"Eurofins.csv not found: {eurofins_csv}")
    if not consolidated_csv.exists():
        raise FileNotFoundError(f"Consolidated_sparse.csv not found: {consolidated_csv}")
    if not limits_csv.exists():
        raise FileNotFoundError(f"Limits.csv not found: {limits_csv}")

    eurofins_df = pd.read_csv(eurofins_csv, sep=";", decimal=".", low_memory=False)
    consolidated_df = pd.read_csv(consolidated_csv, low_memory=False)
    consolidated_timestamps = pd.to_datetime(consolidated_df.get("TIMESTAMP"), errors="coerce")
    limits_records = load_limits_records(limits_csv)

    eurofins_cols = [c for c in eurofins_df.columns if c not in {"Time", "TIMESTAMP"}]
    consolidated_cols = [
        c for c in consolidated_df.columns
        if c not in {"Time", "TIMESTAMP", "Interpolated"} and not c.endswith("_state")
    ]
    eurofins_limits = map_limits_to_columns(eurofins_cols, limits_records)
    consolidated_limits = map_limits_to_columns(consolidated_cols, limits_records)

    def _name_variants_like_limits(name: str) -> set:
        raw = "" if name is None else str(name).strip()
        if not raw:
            return set()
        variants = {normalize_limit_name(raw)}
        no_paren = re.sub(r"\s*\([^)]*\)", "", raw).strip()
        if no_paren:
            variants.add(normalize_limit_name(no_paren))
        return {v for v in variants if v}

    def _pick_col_by_names(columns: list, names: list):
        name_keys = set()
        for n in names:
            name_keys.update(_name_variants_like_limits(n))
        for col in columns:
            col_keys = _name_variants_like_limits(col)
            if name_keys.intersection(col_keys):
                return col
        return None

    ordered_limit_records = [r for r in limits_records if r.get("names")]

    rows = []
    coverage_entries = []
    for rec in ordered_limit_records:
        display_name = rec.get("translated_name") or rec.get("original_name") or rec["names"][0]
        names = rec.get("names", [])
        eurofins_col = _pick_col_by_names(eurofins_cols, names)
        cons_col = _pick_col_by_names(consolidated_cols, names)
        if cons_col is None:
            cons_col = _pick_col_by_names(consolidated_cols, [display_name])

        limit_spec = consolidated_limits.get(cons_col) if cons_col is not None else None
        if limit_spec is None and eurofins_col is not None:
            limit_spec = eurofins_limits.get(eurofins_col)
        upper_limit, lower_limit = _limit_bounds(limit_spec if limit_spec is not None else rec)

        m = re.match(r"(.+?)\s*\(([^)]+)\)$", display_name)
        param_name, unit = (m.group(1).strip(), m.group(2).strip()) if m else (display_name, "")

        valid_count_eurofins = 0
        pct_eurofins = 0.0
        total_eurofins = len(eurofins_df)
        mean_eurofins = None
        std_eurofins = None
        if eurofins_col and eurofins_col in eurofins_df.columns:
            valid_mask = _valid_nonblank_mask(eurofins_df[eurofins_col])
            valid_count_eurofins = valid_mask.sum()
            pct_eurofins = (valid_count_eurofins / total_eurofins * 100) if total_eurofins > 0 else 0.0
            mean_eurofins, std_eurofins = _mean_std_valid_numeric(eurofins_df[eurofins_col], valid_mask)

        if cons_col is not None:
            coverage_entries.append({
                "dataset": "Eurofins",
                "label": param_name,
                "column": cons_col,
                "limit_value": upper_limit,
                "limit_upper": upper_limit,
                "limit_lower": lower_limit,
            })
            valid_mask_cons = _valid_nonblank_mask(consolidated_df[cons_col])
            valid_count_consolidated = valid_mask_cons.sum()
            total_consolidated = len(consolidated_df)
            pct_consolidated = (valid_count_consolidated / total_consolidated * 100) if total_consolidated > 0 else 0.0
            mean_cons, std_cons = _mean_std_valid_numeric(consolidated_df[cons_col], valid_mask_cons)
            mean_cons_norm01, std_cons_norm01 = _mean_std_valid_numeric_normalized_01(consolidated_df[cons_col], valid_mask_cons)
            mean_hours_between = None
            median_hours_between = None
            count_gt_median_hours = None
            valid_ts = consolidated_timestamps[valid_mask_cons & consolidated_timestamps.notna()]
            if not valid_ts.empty:
                valid_ts = valid_ts.sort_values()
                deltas_hours = valid_ts.diff().dt.total_seconds().div(3600.0).dropna()
                if not deltas_hours.empty:
                    mean_hours_between = float(deltas_hours.mean())
                    median_hours_between = float(deltas_hours.median())
                    count_gt_median_hours = int((deltas_hours > median_hours_between).sum())
            split_idx = int(total_consolidated * 0.8)
            train_valid_count = int(valid_mask_cons.iloc[:split_idx].sum())
            test_valid_count = int(valid_mask_cons.iloc[split_idx:].sum())
            exceed_count = 0
            test_exceed_count = 0
            if (upper_limit is not None) or (lower_limit is not None):
                try:
                    numeric_cons = pd.to_numeric(consolidated_df[cons_col], errors="coerce")
                    exceed_all = limit_exceedance_mask(numeric_cons, upper=upper_limit, lower=lower_limit)
                    exceed_count = int((valid_mask_cons.to_numpy() & exceed_all).sum())
                    exceed_test = limit_exceedance_mask(numeric_cons.iloc[split_idx:], upper=upper_limit, lower=lower_limit)
                    test_exceed_count = int(exceed_test.sum())
                except Exception:
                    exceed_count = 0
                    test_exceed_count = 0
        else:
            valid_count_consolidated = 0
            pct_consolidated = 0.0
            mean_cons = None
            std_cons = None
            mean_cons_norm01 = None
            std_cons_norm01 = None
            train_valid_count = 0
            test_valid_count = 0
            exceed_count = 0
            test_exceed_count = 0
            mean_hours_between = None
            median_hours_between = None
            count_gt_median_hours = None

        rows.append({
            "parameter": param_name,
            "unit": unit,
            "limit_value": upper_limit,
            "limit_upper": upper_limit,
            "limit_lower": lower_limit,
            "valid_count_eurofins": int(valid_count_eurofins),
            "percent_valid_eurofins": round(pct_eurofins, 2),
            "avg_valid_eurofins": round(mean_eurofins, 4) if mean_eurofins is not None else None,
            "std_valid_eurofins": round(std_eurofins, 4) if std_eurofins is not None else None,
            "valid_count_consolidated": int(valid_count_consolidated),
            "percent_valid_consolidated": round(pct_consolidated, 2),
            "avg_valid_consolidated": round(mean_cons, 4) if mean_cons is not None else None,
            "std_valid_consolidated": round(std_cons, 4) if std_cons is not None else None,
            "avg_valid_consolidated_norm01": round(mean_cons_norm01, 4) if mean_cons_norm01 is not None else None,
            "std_valid_consolidated_norm01": round(std_cons_norm01, 4) if std_cons_norm01 is not None else None,
            "count_train_valid_consolidated": int(train_valid_count),
            "count_test_valid_consolidated": int(test_valid_count),
            "count_exceed_limit": exceed_count,
            "count_test_exceed_limit": int(test_exceed_count),
            "mean_hours_between_measurements": round(mean_hours_between, 4) if mean_hours_between is not None else None,
            "median_hours_between_measurements": round(median_hours_between, 4) if median_hours_between is not None else None,
            "count_gt_median_hours": int(count_gt_median_hours) if count_gt_median_hours is not None else None,
        })

    out_path = out_tables / "Eurofins_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")
    _write_clustered_bar_chart(summary_df, out_charts, "Eurofins", include_exceed_bar=True)
    _write_eurofins_interval_clustered_bar(summary_df, out_charts)
    return coverage_entries


def main():
    repo_root = Path(__file__).resolve().parents[1]
    coverage_entries = []
    coverage_entries.extend(generate_sensor_summary(repo_root))
    coverage_entries.extend(generate_weather_summary(repo_root))
    coverage_entries.extend(generate_scada_summary(repo_root))
    coverage_entries.extend(generate_eurofins_summary(repo_root))
    _write_coverage_timeline_raster(repo_root, coverage_entries)
    write_category_timeseries_columns(repo_root)
    write_target_split_comparison(repo_root)


if __name__ == "__main__":
    main()




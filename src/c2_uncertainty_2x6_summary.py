"""
Build a single 2x6 uncertainty summary figure for Monte Carlo documentation.

Row 1: raw correction errors by calibration point (box + jitter).
Row 2: offset-error histogram per sensor with fitted normal overlay.
Row 3: per-sensor goodness-of-fit/statistics table.
Row 4: Q-Q plot for recommended best-fit distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from c2_uncertainty import apply_filtering_logic, test_distribution_fit


SENSOR_ORDER = [
    "Sp Cond (microS_cm)",
    "pH",
    "DO (% Sat)",
    "Turbidity (FNU)",
    "fDOM (RFU)",
    "fDOM (QSU)",
]

FONT_SIZE = 15
TIMELINE_FIG_WIDTH = 13.0
TIMELINE_ROW_HEIGHT = 0.92
TIMELINE_RASTER_HEIGHT = 0.45
TIMELINE_MARKER_SIZE = 28
TIMELINE_START_DATE = pd.Timestamp("2020-04-01")
TIMELINE_END_DATE = pd.Timestamp("2025-04-01")
PROFILE_GAP_HOURS = 13.0
POINT_COUNT_STYLES = {
    1: ("#4C78A8", "1-point"),
    2: ("#F58518", "2-point"),
    3: ("#54A24B", "3-point"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_aggregate_csv() -> Path:
    root = _repo_root()
    candidates = [
        root / "data" / "output" / "calibration" / "aggregate" / "offset_gain_model_results.csv",
        root / "data" / "output" / "calibration" / "summaries" / "aggregate" / "offset_gain_model_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _default_output_png(aggregate_csv: Path) -> Path:
    return aggregate_csv.parent / "offset_error_distribution_2x6.png"


def normalize_sensor_name(name: str) -> str:
    text = str(name).strip()
    text = text.replace("Âµ", "µ")
    if "Sp Cond" in text:
        return "Sp Cond (microS_cm)"
    return text


def display_sensor_name(sensor_name: str) -> str:
    if str(sensor_name).strip() == "Sp Cond (microS_cm)":
        return "Specific Cond. (µS/cm)"
    return str(sensor_name).strip()


def _calibration_output_dir() -> Path:
    return _repo_root() / "data" / "output" / "calibration"


def _regression_output_dir() -> Path:
    return _repo_root() / "data" / "output" / "regression"


def _sensor_input_dir() -> Path:
    return _repo_root() / "data" / "input" / "sensors"


def _valid_nonblank_mask(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip()
    return series.notna() & (as_str != "") & (as_str.str.upper() != "#N/A")


def _load_raw_point_errors(sensor_name: str) -> dict[int, np.ndarray]:
    """
    Load raw calibration correction errors per point from data/output/calibration/<sensor>.csv.
    Returns {point_number: np.ndarray(errors)}.
    """
    base_dir = _calibration_output_dir()
    if not base_dir.exists():
        return {}

    sensor_csv = None
    for path in sorted(base_dir.glob("*.csv")):
        if normalize_sensor_name(path.stem) == sensor_name:
            sensor_csv = path
            break
    if sensor_csv is None:
        return {}

    try:
        raw_df = pd.read_csv(sensor_csv)
    except Exception:
        return {}

    out: dict[int, np.ndarray] = {}
    for idx, col in enumerate(["Correction1", "Correction2", "Correction3"], start=1):
        if col in raw_df.columns:
            vals = pd.to_numeric(raw_df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size > 0:
                out[idx] = vals
    return out


def fmt(value: float | str | None, digits: int = 4, scientific: bool = False) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        return "NA"
    if scientific:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _most_common_points(points: np.ndarray) -> str:
    if points.size == 0:
        return "NA"
    valid = points[np.isfinite(points)]
    if valid.size == 0:
        return "NA"
    # Use integer point counts and take modal value across events.
    values, counts = np.unique(valid.astype(int), return_counts=True)
    return str(values[np.argmax(counts)])


def _preferred_fit_label(fit: dict) -> str:
    preferred = fit.get("preferred", "NA")
    if preferred == "t":
        return "Student t"
    elif preferred == "norm":
        return "Normal"
    return str(preferred)


def _sensor_offset_label(sensor_name: str) -> str:
    text = display_sensor_name(sensor_name)
    if text.endswith(")") and " (" in text:
        base, units = text.rsplit(" (", 1)
        return f"{base} Offset\n({units}"
    return f"{text} Offset"


def _sensor_timeline_label(sensor_name: str, calibration_count: int) -> str:
    return f"{_sensor_offset_label(sensor_name)} - {int(calibration_count)} events"


def _profiler_window_label(day_count: int) -> str:
    return f"Profiler Uptime - {int(day_count)} days"


def _timeline_sensor_order() -> list[str]:
    return sorted(SENSOR_ORDER, key=lambda sensor: _sensor_offset_label(sensor).lower())


def _event_index_column(df: pd.DataFrame) -> str:
    for candidate in ["Event_ID", "Event_Index"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Aggregate calibration file must include either 'Event_ID' or 'Event_Index'."
    )


def _load_filtered_calibration_rows() -> dict[str, pd.DataFrame]:
    base_dir = _calibration_output_dir()
    if not base_dir.exists():
        raise FileNotFoundError(f"Calibration output directory not found: {base_dir}")

    by_sensor: dict[str, pd.DataFrame] = {}
    for path in sorted(base_dir.glob("*.csv")):
        sensor_name = normalize_sensor_name(path.stem)
        raw_df = pd.read_csv(path, low_memory=False)
        filtered_df = apply_filtering_logic(raw_df)
        filtered_df = filtered_df.reset_index(drop=True).copy()
        filtered_df["_Filtered_Event_Index"] = np.arange(len(filtered_df), dtype=int)
        by_sensor[sensor_name] = filtered_df
    return by_sensor


def _enrich_with_calibration_end_time(df: pd.DataFrame) -> pd.DataFrame:
    event_col = _event_index_column(df)
    enriched = df.copy()
    enriched["Sensor_Normalized"] = enriched["Sensor"].map(normalize_sensor_name)
    event_index = pd.to_numeric(enriched[event_col], errors="coerce")
    if event_index.isna().all():
        raise ValueError(f"Column '{event_col}' does not contain any usable event indices.")
    enriched["_Event_Index"] = event_index.astype("Int64")

    filtered_rows = _load_filtered_calibration_rows()
    calibration_times: list[pd.Timestamp | pd.NaT] = []

    for _, row in enriched.iterrows():
        sensor_name = row["Sensor_Normalized"]
        event_idx = row["_Event_Index"]
        if pd.isna(event_idx) or sensor_name not in filtered_rows:
            calibration_times.append(pd.NaT)
            continue

        sensor_df = filtered_rows[sensor_name]
        event_idx_int = int(event_idx)
        if event_idx_int < 0 or event_idx_int >= len(sensor_df):
            calibration_times.append(pd.NaT)
            continue

        end_value = sensor_df.iloc[event_idx_int].get("Calibration End Time")
        calibration_times.append(pd.to_datetime(end_value, errors="coerce"))

    enriched["Calibration_End_Time"] = pd.to_datetime(calibration_times, errors="coerce")
    return enriched


def _load_surface_raster_data() -> tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp, int]:
    profiles_csv = _sensor_input_dir() / "Profiles.csv"
    if not profiles_csv.exists():
        raise FileNotFoundError(f"Profiles.csv not found: {profiles_csv}")

    df = pd.read_csv(profiles_csv, low_memory=False)
    if "TIMESTAMP" not in df.columns:
        raise ValueError(f"Missing required column 'TIMESTAMP' in {profiles_csv}")

    sensor_cols = [c for c in df.columns if str(c).startswith("sensorParms(")]
    if not sensor_cols:
        raise ValueError(
            f"No profiler sensor columns matching 'sensorParms(n)' found in {profiles_csv}"
        )

    timestamps = pd.to_datetime(df["TIMESTAMP"], errors="coerce").dropna().drop_duplicates().sort_values()
    if timestamps.empty:
        raise ValueError(f"No valid TIMESTAMP values found in {profiles_csv}")
    day_count = int(pd.DatetimeIndex(timestamps).normalize().nunique())

    x_edges, values = _build_profile_operational_window(
        pd.DatetimeIndex(timestamps),
        start_date=TIMELINE_START_DATE,
        end_date=TIMELINE_END_DATE,
        gap_hours=PROFILE_GAP_HOURS,
    )
    return x_edges, values.reshape(1, -1), TIMELINE_START_DATE, TIMELINE_END_DATE, day_count


def _build_profile_operational_window(
    timestamps: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    gap_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    if start_date >= end_date:
        raise ValueError("Timeline start must be before timeline end.")

    gap_threshold = pd.Timedelta(hours=float(gap_hours))
    edges = [mdates.date2num(start_date.to_pydatetime())]
    values: list[float] = []
    current = start_date

    if len(timestamps) >= 2:
        for left, right in zip(timestamps[:-1], timestamps[1:]):
            if right <= start_date or left >= end_date:
                continue

            seg_start = max(pd.Timestamp(left), start_date)
            seg_end = min(pd.Timestamp(right), end_date)
            if seg_end <= seg_start:
                continue

            is_operational = (pd.Timestamp(right) - pd.Timestamp(left)) <= gap_threshold
            if seg_start > current:
                edges.append(mdates.date2num(seg_start.to_pydatetime()))
                values.append(0.0)
                current = seg_start

            if seg_end > current:
                edges.append(mdates.date2num(seg_end.to_pydatetime()))
                values.append(1.0 if is_operational else 0.0)
                current = seg_end

    if current < end_date:
        edges.append(mdates.date2num(end_date.to_pydatetime()))
        values.append(0.0)

    if len(edges) == 1:
        edges.append(mdates.date2num(end_date.to_pydatetime()))
        values.append(0.0)

    return np.asarray(edges, dtype=float), np.asarray(values, dtype=float)


def _datetime_edges(timestamps: pd.DatetimeIndex) -> np.ndarray:
    t_float = mdates.date2num(timestamps.to_pydatetime())
    if len(t_float) == 1:
        return np.array([t_float[0] - 1.0 / 48.0, t_float[0] + 1.0 / 48.0], dtype=float)

    mids = (t_float[:-1] + t_float[1:]) / 2.0
    start_edge = t_float[0] - (t_float[1] - t_float[0]) / 2.0
    end_edge = t_float[-1] + (t_float[-1] - t_float[-2]) / 2.0
    return np.concatenate([[start_edge], mids, [end_edge]])


def _quarter_ticks(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    ticks = list(pd.date_range(start=start_date.normalize(), end=end_date.normalize(), freq="QS-JAN"))
    if not ticks:
        return [start_date, end_date]
    return ticks


def _set_timeline_xticks(ax: plt.Axes, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    ticks = _quarter_ticks(start_date, end_date)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [pd.Timestamp(tick).strftime("%Y-%m-%d") for tick in ticks],
        rotation=25,
        ha="right",
        fontsize=FONT_SIZE - 1,
    )
    ax.set_xlim(start_date, end_date)


def _point_count_legend_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=7,
            label=label,
        )
        for _, (color, label) in POINT_COUNT_STYLES.items()
    ]


def _create_timeline_axes(
    dpi: int,
) -> tuple[plt.Figure, plt.Axes, list[plt.Axes]]:
    timeline_sensors = _timeline_sensor_order()
    fig_h = TIMELINE_RASTER_HEIGHT + TIMELINE_ROW_HEIGHT * len(timeline_sensors)
    fig, all_axes = plt.subplots(
        len(timeline_sensors) + 1,
        1,
        sharex=True,
        figsize=(TIMELINE_FIG_WIDTH, fig_h),
        dpi=dpi,
        gridspec_kw={
            "hspace": 0.08,
            "height_ratios": [TIMELINE_RASTER_HEIGHT / TIMELINE_ROW_HEIGHT] + [1.0] * len(timeline_sensors),
        },
    )
    return fig, all_axes[0], list(all_axes[1:])


def _draw_surface_raster(
    ax: plt.Axes,
    x_edges: np.ndarray,
    count_values: np.ndarray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    day_count: int,
) -> None:
    vmax = max(int(np.nanmax(count_values)), 1)
    ax.pcolormesh(
        x_edges,
        [0, 1],
        count_values,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        rasterized=True,
    )

    ax.set_ylabel(
        _profiler_window_label(day_count),
        rotation=0,
        ha="right",
        va="center",
        fontsize=FONT_SIZE,
        labelpad=24,
    )
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.set_xlim(start_date, end_date)


def create_offset_event_timeseries_figure(df: pd.DataFrame, output_png: Path, dpi: int = 300) -> None:
    event_df = df.copy()
    event_df["Offset"] = pd.to_numeric(event_df["Offset"], errors="coerce")
    event_df["N_Points"] = pd.to_numeric(event_df["N_Points"], errors="coerce")
    event_df = event_df.dropna(subset=["Calibration_End_Time", "Offset"])
    event_df["N_Points_Plot"] = event_df["N_Points"].clip(lower=1).fillna(1).astype(int).clip(upper=3)

    x_edges, count_values, start_date, end_date, day_count = _load_surface_raster_data()
    timeline_sensors = _timeline_sensor_order()
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, raster_ax, axes = _create_timeline_axes(dpi=dpi)
    _draw_surface_raster(raster_ax, x_edges, count_values, start_date, end_date, day_count)

    for ax, sensor in zip(axes, timeline_sensors):
        sensor_df = event_df.loc[event_df["Sensor_Normalized"] == sensor].sort_values("Calibration_End_Time")
        if sensor_df.empty:
            ax.text(0.5, 0.5, "No events", ha="center", va="center", fontsize=FONT_SIZE, transform=ax.transAxes)
            ax.set_yticks([])
        else:
            for point_count, (color, _) in POINT_COUNT_STYLES.items():
                subset = sensor_df.loc[sensor_df["N_Points_Plot"] == point_count]
                if subset.empty:
                    continue
                ax.scatter(
                    subset["Calibration_End_Time"],
                    subset["Offset"],
                    s=TIMELINE_MARKER_SIZE,
                    color=color,
                    alpha=0.9,
                    edgecolors="none",
                )
            ax.tick_params(axis="y", labelsize=FONT_SIZE - 1)
        ax.set_ylabel(
            _sensor_timeline_label(sensor, len(sensor_df)),
            rotation=0,
            ha="right",
            va="center",
            fontsize=FONT_SIZE,
            labelpad=18,
        )
        ax.grid(True, alpha=0.25, axis="y")

    fig.legend(
        handles=_point_count_legend_handles(),
        loc="lower left",
        bbox_to_anchor=(0.004, 0.006),
        borderaxespad=0.0,
        frameon=True,
        fontsize=FONT_SIZE - 2,
    )
    _set_timeline_xticks(axes[-1], start_date, end_date)
    axes[-1].set_xlabel("Calibration End Time", fontsize=FONT_SIZE)
    fig.subplots_adjust(top=0.985, bottom=0.15, left=0.19, right=0.93, hspace=0.1)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def create_calibration_pointcount_timeline_figure(
    df: pd.DataFrame,
    output_png: Path,
    dpi: int = 300,
) -> None:
    event_df = df.copy()
    event_df["N_Points"] = pd.to_numeric(event_df["N_Points"], errors="coerce")
    event_df = event_df.dropna(subset=["Calibration_End_Time", "N_Points"])

    x_edges, count_values, start_date, end_date, day_count = _load_surface_raster_data()
    timeline_sensors = _timeline_sensor_order()
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig_h = max(5.4, 0.65 * (len(timeline_sensors) + 1) + 1.0)
    fig, ax = plt.subplots(figsize=(TIMELINE_FIG_WIDTH, fig_h), dpi=dpi)

    raster_bottom = -0.35
    raster_top = 0.35
    vmax = max(int(np.nanmax(count_values)), 1)
    ax.pcolormesh(
        x_edges,
        [raster_bottom, raster_top],
        count_values,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        rasterized=True,
        zorder=1,
    )

    y_positions = [0]
    y_labels = [_profiler_window_label(day_count)]
    for row_idx, sensor in enumerate(timeline_sensors, start=1):
        sensor_df = event_df.loc[event_df["Sensor_Normalized"] == sensor].sort_values("Calibration_End_Time").copy()
        sensor_df["N_Points_Plot"] = sensor_df["N_Points"].clip(lower=1).fillna(1).astype(int).clip(upper=3)

        ax.hlines(row_idx, start_date, end_date, color="#E6E6E6", linewidth=1.0, zorder=1)
        for point_count, (color, _) in POINT_COUNT_STYLES.items():
            subset = sensor_df.loc[sensor_df["N_Points_Plot"] == point_count]
            if subset.empty:
                continue
            ax.plot(
                subset["Calibration_End_Time"],
                np.full(len(subset), row_idx, dtype=float),
                linestyle="",
                marker="o",
                markersize=6,
                color=color,
                alpha=0.95,
                zorder=2,
            )

        y_positions.append(row_idx)
        y_labels.append(_sensor_timeline_label(sensor, len(sensor_df)))

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=FONT_SIZE - 1)
    ax.tick_params(axis="y", left=False, labelleft=True)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(len(timeline_sensors) + 0.5, -0.5)

    fig.legend(
        handles=_point_count_legend_handles(),
        loc="lower left",
        bbox_to_anchor=(0.004, 0.006),
        borderaxespad=0.0,
        frameon=True,
        fontsize=FONT_SIZE - 2,
    )
    _set_timeline_xticks(ax, start_date, end_date)
    ax.set_xlabel("Calibration End Time", fontsize=FONT_SIZE)
    fig.subplots_adjust(top=0.985, bottom=0.17, left=0.28, right=0.98)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_sensor_table_rows(fit: dict, errors: np.ndarray, points_per_event: np.ndarray) -> list[list[str]]:
    preferred = _preferred_fit_label(fit)

    rows = [
        ["n calibrations", str(int(fit.get("n_clean", len(errors))))],
        ["mean", fmt(np.mean(errors))],
        ["std", fmt(np.std(errors))],
        ["Shapiro p", fmt(fit.get("shapiro_p"), scientific=True)],
        ["Anderson", fmt(fit.get("anderson_stat"))],
        ["AD crit (5%)", fmt(fit.get("anderson_critical_5pct"))],
        ["Normal AIC", fmt(fit.get("norm_aic"), digits=2)],
        ["Student t AIC", fmt(fit.get("t_aic"), digits=2)],
        ["t df", fmt(fit.get("t_df"))],
        ["t loc", fmt(fit.get("t_loc"))],
        ["t scale", fmt(fit.get("t_scale"))],
        ["AIC delta", fmt(fit.get("aic_diff"), digits=2)],
        ["Best fit", preferred],
    ]
    return rows


def build_quantile_rows(fit: dict, errors: np.ndarray) -> list[list[str]]:
    probs = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
    preferred = fit.get("preferred", "equivalent")
    dist_used = "Normal"

    q_values = None
    if preferred == "t":
        t_df = fit.get("t_df")
        t_loc = fit.get("t_loc")
        t_scale = fit.get("t_scale")
        if all(v is not None and np.isfinite(v) for v in [t_df, t_loc, t_scale]) and t_scale > 0:
            q_values = stats.t.ppf(probs, df=t_df, loc=t_loc, scale=t_scale)
            dist_used = "Student t"

    if q_values is None:
        mu = fit.get("norm_loc", float(np.mean(errors)))
        sigma = fit.get("norm_scale", float(np.std(errors, ddof=0)))
        if sigma <= 0 or not np.isfinite(sigma):
            q_values = np.full_like(probs, np.nan, dtype=float)
        else:
            q_values = stats.norm.ppf(probs, loc=mu, scale=sigma)
        if preferred == "equivalent":
            dist_used = "Equivalent->Normal"

    return [
        ["Best fit used", dist_used],
        ["q05", fmt(q_values[0])],
        ["q25", fmt(q_values[1])],
        ["q50", fmt(q_values[2])],
        ["q75", fmt(q_values[3])],
        ["q95", fmt(q_values[4])],
    ]


def plot_histogram(ax: plt.Axes, errors: np.ndarray, title: str, fit: dict) -> None:
    bins = max(6, min(14, len(errors) // 2))
    counts, bin_edges, _ = ax.hist(
        errors,
        bins=bins,
        color="#4C78A8",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.7,
    )

    mu = float(np.mean(errors))
    sigma = float(np.std(errors, ddof=0))
    preferred = fit.get("preferred", "equivalent")
    normal_lw = 3.0 if preferred == "norm" else 1.8
    t_lw = 3.0 if preferred == "t" else 1.8

    if sigma > 0 and len(bin_edges) > 1:
        x = np.linspace(bin_edges[0], bin_edges[-1], 300)
        bin_width = bin_edges[1] - bin_edges[0]
        y = stats.norm.pdf(x, loc=mu, scale=sigma) * len(errors) * bin_width
        ax.plot(x, y, color="#E45756", linewidth=normal_lw, label="Normal fit")

        try:
            t_df, t_loc, t_scale = stats.t.fit(errors)
            if np.isfinite(t_df) and np.isfinite(t_loc) and np.isfinite(t_scale) and t_scale > 0:
                y_t = stats.t.pdf(x, df=t_df, loc=t_loc, scale=t_scale) * len(errors) * bin_width
                ax.plot(x, y_t, color="#54A24B", linewidth=t_lw, label="Student t fit")
        except Exception:
            pass

        ax.legend(loc="upper right", frameon=True, fontsize=FONT_SIZE)

    ax.axvline(mu, color="#F58518", linestyle="--", linewidth=1.5)
    ax.set_title("")
    ax.set_xlabel("Offset", fontsize=FONT_SIZE)
    ax.set_ylabel("Count", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.25, axis="y")


def plot_point_box_jitter(ax: plt.Axes, sensor_name: str, point_errors: dict[int, np.ndarray]) -> None:
    title_text = display_sensor_name(sensor_name)
    if not point_errors:
        ax.text(0.5, 0.5, "No point errors", ha="center", va="center", fontsize=FONT_SIZE, transform=ax.transAxes)
        ax.set_title(title_text, fontsize=FONT_SIZE, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    points = sorted(point_errors.keys())
    data = [point_errors[p] for p in points]

    ax.boxplot(
        data,
        positions=points,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#BFD7EA", alpha=0.75, edgecolor="black"),
        medianprops=dict(color="#E45756", linewidth=2.0),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    rng = np.random.default_rng(42)
    for p, vals in zip(points, data):
        x_jitter = p + rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x_jitter, vals, s=20, alpha=0.65, color="#2E5D8A", edgecolors="none")

    ax.set_title(title_text, fontsize=FONT_SIZE, fontweight="bold")
    ax.set_xlabel("Calibration point", fontsize=FONT_SIZE)
    ax.set_ylabel("Raw error", fontsize=FONT_SIZE)
    ax.set_xticks(points)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.25, axis="y")


def plot_best_fit_qq(ax: plt.Axes, errors: np.ndarray, fit: dict) -> None:
    if errors.size < 3:
        ax.text(0.5, 0.5, "n < 3", ha="center", va="center", fontsize=FONT_SIZE, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    probs = (np.arange(1, errors.size + 1) - 0.5) / errors.size
    sample_q = np.sort(errors)
    preferred = fit.get("preferred", "equivalent")

    if preferred == "t":
        t_df = fit.get("t_df")
        t_loc = fit.get("t_loc")
        t_scale = fit.get("t_scale")
        if all(v is not None and np.isfinite(v) for v in [t_df, t_loc, t_scale]) and t_scale > 0:
            theoretical_q = stats.t.ppf(probs, df=t_df, loc=t_loc, scale=t_scale)
            dist_label = "Student t"
        else:
            mu = fit.get("norm_loc", float(np.mean(errors)))
            sigma = fit.get("norm_scale", float(np.std(errors, ddof=0)))
            theoretical_q = stats.norm.ppf(probs, loc=mu, scale=sigma)
            dist_label = "Normal"
    else:
        mu = fit.get("norm_loc", float(np.mean(errors)))
        sigma = fit.get("norm_scale", float(np.std(errors, ddof=0)))
        theoretical_q = stats.norm.ppf(probs, loc=mu, scale=sigma)
        dist_label = "Normal"

    ax.scatter(theoretical_q, sample_q, color="#4C78A8", s=36, alpha=0.85, edgecolors="none")
    lo = min(np.nanmin(theoretical_q), np.nanmin(sample_q))
    hi = max(np.nanmax(theoretical_q), np.nanmax(sample_q))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#E45756", linewidth=2.0)
    ax.set_title(f"Q-Q ({dist_label})", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_xlabel("Theoretical quantiles", fontsize=FONT_SIZE)
    ax.set_ylabel("Sample quantiles", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.25)


def create_figure(df: pd.DataFrame, output_png: Path, dpi: int = 300) -> None:
    df = df.copy()
    df["Sensor_Normalized"] = df["Sensor"].map(normalize_sensor_name)
    df["Offset"] = pd.to_numeric(df["Offset"], errors="coerce")

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(4, 6, figsize=(30, 17), dpi=dpi)
    for col, sensor in enumerate(SENSOR_ORDER):
        sensor_errors = (
            df.loc[df["Sensor_Normalized"] == sensor, "Offset"]
            .dropna()
            .to_numpy(dtype=float)
        )
        sensor_point_errors = _load_raw_point_errors(sensor)
        sensor_points = (
            pd.to_numeric(df.loc[df["Sensor_Normalized"] == sensor, "N_Points"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
            if "N_Points" in df.columns
            else np.array([])
        )

        fit = test_distribution_fit(sensor_errors, "Offset")
        box_ax = axes[0, col]
        hist_ax = axes[1, col]
        table_ax = axes[2, col]
        quant_ax = axes[3, col]

        if sensor_errors.size < 3:
            plot_point_box_jitter(box_ax, sensor, sensor_point_errors)
            hist_ax.text(
                0.5, 0.5, "Insufficient data", ha="center", va="center",
                fontsize=FONT_SIZE, transform=hist_ax.transAxes
            )
            hist_ax.set_title("")
            hist_ax.set_xlabel("Offset", fontsize=FONT_SIZE)
            hist_ax.set_xticks([])
            hist_ax.set_yticks([])

            table_ax.axis("off")
            table_ax.text(
                0.5, 0.5, "n < 3", ha="center", va="center",
                fontsize=FONT_SIZE, transform=table_ax.transAxes
            )
            quant_ax.axis("off")
            quant_ax.text(
                0.5, 0.5, "n < 3", ha="center", va="center",
                fontsize=FONT_SIZE, transform=quant_ax.transAxes
            )
            continue

        plot_point_box_jitter(box_ax, sensor, sensor_point_errors)
        plot_histogram(hist_ax, sensor_errors, "", fit)

        table_ax.axis("off")
        table_rows = build_sensor_table_rows(fit, sensor_errors, sensor_points)
        table = table_ax.table(
            cellText=table_rows,
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(FONT_SIZE)
        table.scale(1.0, 1.25)

        plot_best_fit_qq(quant_ax, sensor_errors, fit)

    fig.subplots_adjust(top=0.985, bottom=0.03, left=0.03, right=0.995, hspace=0.15, wspace=0.24)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 2x6 figure with per-sensor offset-error histograms and fit-stat tables "
            "for MC uncertainty documentation."
        )
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=_default_aggregate_csv(),
        help="Path to aggregate offset_gain_model_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: aggregate dir / offset_error_distribution_2x6.png)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_csv = args.aggregate_csv
    output_png = args.output or _default_output_png(aggregate_csv)

    if not aggregate_csv.exists():
        raise FileNotFoundError(
            f"Could not find aggregate file: {aggregate_csv}. Run c2_uncertainty.py first."
        )

    df = pd.read_csv(aggregate_csv)
    required_cols = {"Sensor", "Offset"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {aggregate_csv}: {sorted(missing)}")

    timeline_df = _enrich_with_calibration_end_time(df)
    valid_timeline_df = timeline_df.dropna(subset=["Calibration_End_Time"]).copy()
    if valid_timeline_df.empty:
        raise ValueError("Could not join any aggregate events to valid Calibration End Time values.")

    create_figure(df, output_png, dpi=args.dpi)
    offset_timeline_png = output_png.parent / "offset_event_timeseries.png"
    pointcount_timeline_png = output_png.parent / "calibration_event_pointcount_timeline.png"
    create_offset_event_timeseries_figure(valid_timeline_df, offset_timeline_png, dpi=args.dpi)
    create_calibration_pointcount_timeline_figure(valid_timeline_df, pointcount_timeline_png, dpi=args.dpi)
    print(f"Saved: {output_png}")
    print(f"Saved: {offset_timeline_png}")
    print(f"Saved: {pointcount_timeline_png}")


if __name__ == "__main__":
    main()

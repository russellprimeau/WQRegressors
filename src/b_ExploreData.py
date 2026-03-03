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
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

    plot_df = plot_df.sort_values("valid_count_consolidated", ascending=False).reset_index(drop=True)
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
    ax_left.set_title(f"{dataset_name}: Data Availability and Normalized Variability")
    ax_left.grid(axis="y", linestyle="--", alpha=0.35)

    handles = left_handles + [bars_std]
    labels = left_labels + ["Std norm [0,1]"]
    ax_left.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    fig_path = out_dir / f"{dataset_name}_clustered_bar.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote chart to: {fig_path}")


def _write_coverage_timeline_raster(repo_root: Path, coverage_entries: list) -> None:
    """Write condensed coverage timeline: proxy rasters + Eurofins marker rows."""
    if not coverage_entries:
        return

    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    out_dir = repo_root / "data" / "sensors" / "summaries"
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
            "mask": euro_mask,
            "count": euro_count,
        })

    if not proxy_rows and not euro_rows:
        return

    all_rows = proxy_rows + euro_rows
    all_rows.sort(key=lambda r: (-r["count"], r["label"]))

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
            limit_val = row.get("limit_value")
            if limit_val is not None:
                numeric_vals = pd.to_numeric(sample_series, errors="coerce")
                if np.isclose(limit_val, 0.0):
                    exceed_mask = numeric_vals.to_numpy() > limit_val
                else:
                    exceed_mask = numeric_vals.to_numpy() >= limit_val
                point_colors = np.where(exceed_mask, "#d62728", "#2ca02c")
            else:
                point_colors = "#2ca02c"
            raw_preview = sample_series.head(5).tolist()
            numeric_preview = pd.to_numeric(sample_series, errors="coerce").head(5).tolist()
            print(
                f"[LIMIT CHECK] {row['label']}: limit={limit_val}, "
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
            start_date = pd.Timestamp("2022-01-01")
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
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#2ca02c", markeredgecolor="#2ca02c", markersize=6, label="Target < limit"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#d62728", markeredgecolor="#d62728", markersize=6, label="Target ≥ limit"),
    ]
    ax.legend(handles=eurofins_legend, loc="center right", bbox_to_anchor=(0.985, 0.5), borderaxespad=0.0, fontsize=text_fs)

    fig.tight_layout()
    out_path = out_dir / "Coverage_timeline_raster.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote chart to: {out_path}")


def generate_sensor_summary(repo_root: Path) -> list:
    """Generate FullHourly_summary.csv from profiler sensor (FullHourly.csv) data."""
    input_csv = repo_root / "data" / "input" / "sensors" / "FullHourly.csv"
    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    out_dir = repo_root / "data" / "sensors" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    out_path = out_dir / "FullHourly_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")
    _write_clustered_bar_chart(summary_df, out_dir, "FullHourly", include_exceed_bar=False)
    return coverage_entries


def generate_weather_summary(repo_root: Path) -> list:
    """Generate Weather_summary.csv from weather station (Weather.csv) data."""
    weather_csv = repo_root / "data" / "input" / "sensors" / "Weather.csv"
    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    out_dir = repo_root / "data" / "sensors" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    out_path = out_dir / "Weather_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")
    _write_clustered_bar_chart(summary_df, out_dir, "Weather", include_exceed_bar=False)
    return coverage_entries


def generate_scada_summary(repo_root: Path) -> list:
    """Generate SCADA_summary.csv from SCADA sensor (SCADA.csv) data."""
    scada_csv = repo_root / "data" / "input" / "sensors" / "SCADA.csv"
    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    out_dir = repo_root / "data" / "sensors" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    out_path = out_dir / "SCADA_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")
    _write_clustered_bar_chart(summary_df, out_dir, "SCADA", include_exceed_bar=False)
    return coverage_entries


def generate_eurofins_summary(repo_root: Path) -> list:
    """Generate Eurofins_summary.csv from lab measurement (Eurofins.csv) data."""
    eurofins_csv = repo_root / "data" / "input" / "sensors" / "Eurofins.csv"
    consolidated_csv = repo_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    limits_csv = repo_root / "data" / "input" / "Limits.csv"
    out_dir = repo_root / "data" / "sensors" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not eurofins_csv.exists():
        raise FileNotFoundError(f"Eurofins.csv not found: {eurofins_csv}")
    if not consolidated_csv.exists():
        raise FileNotFoundError(f"Consolidated_sparse.csv not found: {consolidated_csv}")
    if not limits_csv.exists():
        raise FileNotFoundError(f"Limits.csv not found: {limits_csv}")

    eurofins_mapping = {
        "01-Farge": "Color",
        "04-Turbiditet": "Turbidity (FNU)",
        "06-E.coli": "E.coli (CFU/100mL)",
        "07-Intestinale enterokokker": "Intestinal enterococci (CFU/100mL)",
        "08-Kimtall 22°C": "Colony Count 22°C (CFU/mL)",
        "09-Koliforme bakterier 37°C": "Total coliforms 37°C (CFU/100mL)",
        "21-Arsen": "Arsenic (µg/L)",
        "24-Bly": "Lead (µg/L)",
        "32-Kadmium": "Cadmium (µg/L)",
        "36-Kopper filtrert": "Copper filtered (mg/L)",
        "37-Krom": "Chromium (µg/L)",
        "41-Nikkel": "Nickel (µg/L)",
        "Sink (Zn)": "Zinc (µg/L)",
        "44-pH, surhetsgrad": "pH",
    }

    summary_params = [
        "Color", "Turbidity (FNU)", "E.coli (CFU/100mL)", "Intestinal enterococci (CFU/100mL)",
        "Colony Count 22°C (CFU/mL)", "Total coliforms 37°C (CFU/100mL)", "Arsenic (µg/L)",
        "Lead (µg/L)", "Cadmium (µg/L)", "Copper filtered (mg/L)", "Chromium (µg/L)",
        "Nickel (µg/L)", "Zinc (µg/L)", "pH",
    ]

    eurofins_df = pd.read_csv(eurofins_csv, sep=";", decimal=".", low_memory=False)
    consolidated_df = pd.read_csv(consolidated_csv, low_memory=False)
    limits_df = pd.read_csv(limits_csv, sep=";", decimal=".", low_memory=False)
    limits_mapping = {col: limits_df.iloc[0][col] for col in limits_df.columns}

    rows = []
    coverage_entries = []
    for param in summary_params:
        eurofins_col = next((k for k, v in eurofins_mapping.items() if v == param), None)
        m = re.match(r"(.+?)\s*\(([^)]+)\)$", param)
        param_name, unit = (m.group(1).strip(), m.group(2).strip()) if m else (param, "")

        limit_val = None
        if eurofins_col and eurofins_col in limits_mapping:
            try:
                raw_limit = limits_mapping[eurofins_col]
                if pd.notna(raw_limit):
                    if isinstance(raw_limit, str):
                        cleaned = raw_limit.strip().replace(",", ".")
                        numeric_match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
                        limit_val = float(numeric_match.group(0)) if numeric_match else None
                    else:
                        limit_val = float(raw_limit)
            except (ValueError, TypeError):
                limit_val = None

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

        col_candidates = [
            c for c in consolidated_df.columns
            if c.strip() == param or c.strip().lower().replace(" ", "") == param.lower().replace(" ", "")
        ]

        if col_candidates:
            c = col_candidates[0]
            coverage_entries.append({
                "dataset": "Eurofins",
                "label": param_name,
                "column": c,
                "limit_value": limit_val,
            })
            valid_mask_cons = _valid_nonblank_mask(consolidated_df[c])
            valid_count_consolidated = valid_mask_cons.sum()
            total_consolidated = len(consolidated_df)
            pct_consolidated = (valid_count_consolidated / total_consolidated * 100) if total_consolidated > 0 else 0.0
            mean_cons, std_cons = _mean_std_valid_numeric(consolidated_df[c], valid_mask_cons)
            mean_cons_norm01, std_cons_norm01 = _mean_std_valid_numeric_normalized_01(consolidated_df[c], valid_mask_cons)
            split_idx = int(total_consolidated * 0.8)
            train_valid_count = int(valid_mask_cons.iloc[:split_idx].sum())
            test_valid_count = int(valid_mask_cons.iloc[split_idx:].sum())
            exceed_count = 0
            test_exceed_count = 0
            if limit_val is not None:
                try:
                    numeric_cons = pd.to_numeric(consolidated_df[c], errors="coerce")
                    exceed_count = int((valid_mask_cons & (numeric_cons > limit_val)).sum())
                    test_exceed_count = int((numeric_cons.iloc[split_idx:] > limit_val).sum())
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

        rows.append({
            "parameter": param_name,
            "unit": unit,
            "limit_value": limit_val,
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
        })

    out_path = out_dir / "Eurofins_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")
    _write_clustered_bar_chart(summary_df, out_dir, "Eurofins", include_exceed_bar=True)
    return coverage_entries


def main():
    repo_root = Path(__file__).resolve().parents[1]
    coverage_entries = []
    coverage_entries.extend(generate_sensor_summary(repo_root))
    coverage_entries.extend(generate_weather_summary(repo_root))
    coverage_entries.extend(generate_scada_summary(repo_root))
    coverage_entries.extend(generate_eurofins_summary(repo_root))
    _write_coverage_timeline_raster(repo_root, coverage_entries)


if __name__ == "__main__":
    main()

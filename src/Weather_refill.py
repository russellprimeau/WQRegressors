"""
Create a refill template for weather backfilling.

Reads:
  data/input/sensors/Weather.csv

Writes:
  data/input/sensors/Filler.csv

Output contains:
  - The same columns as Weather.csv
  - One row per timestamp from Weather.csv within the requested date window
  - Blank values for all non-Time columns

Run:

python src/Weather_refill.py make-template
python src/Weather_refill.py query-api --endpoint "https://your.api/endpoint" --dry-run
python src/Weather_refill.py query-api --endpoint "https://your.api/endpoint" --params-per-query 6 --times-per-query 72

python src/Weather_refill.py query-thredds
python src/Weather_refill.py query-thredds `
  --weather-filename "Weather.csv" `
  --output-filename "Filler_thredds.csv" `
  --start "2022-01-01 00:00:00" `
  --end "2025-03-01 00:00:00" `
  --lat 62.484785020758075 `
  --lon 6.479653454212095 `
  --base-url "https://thredds.met.no/thredds/dodsC/metpparchive" `
  --filename-prefix "met_analysis_1_0km_nordic"

python src/Weather_refill.py plot-thredds-overlay

python src/Weather_refill.py combine-weather  

python src/Weather_refill.py combine-weather `
  --weather-filename Weather.csv `
  --filler-thredds-filename Filler_thredds.csv `
  --output-filename Weather_combo.csv
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import netCDF4 as nc


def create_weather_refill_template(
    repo_root: Path,
    start: str = "2020-01-01 00:00:00",
    end: str = "2025-04-01 00:00:00",
) -> None:
    weather_path = repo_root / "data" / "input" / "sensors" / "Weather.csv"
    filler_path = repo_root / "data" / "input" / "sensors" / "Filler.csv"

    if not weather_path.exists():
        raise FileNotFoundError(f"Weather.csv not found: {weather_path}")

    weather_df = pd.read_csv(weather_path, sep=";", low_memory=False)
    if "Time" not in weather_df.columns:
        raise ValueError("Expected a 'Time' column in Weather.csv")

    ts = pd.to_datetime(weather_df["Time"], errors="coerce")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    in_window = (ts >= start_ts) & (ts <= end_ts)
    times = ts[in_window].dropna().drop_duplicates().sort_values()

    filler_df = pd.DataFrame({"Time": times.dt.strftime("%Y-%m-%dT%H:%M:%S")})
    for col in weather_df.columns:
        if col != "Time":
            filler_df[col] = pd.NA

    filler_df = filler_df[weather_df.columns]
    filler_df.to_csv(filler_path, sep=";", index=False)
    print(f"Wrote refill template to: {filler_path}")
    print(f"Rows written: {len(filler_df)}")


def chunked(values: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(values), n):
        yield values[i:i + n]


def default_param_transform(param: str, value: object) -> object:
    """
    Hook for unit conversions/transforms.
    Keep identity until API semantics are known.
    """
    return value


def extract_values_default(payload: object) -> List[Tuple[str, str, object]]:
    """
    Try to parse common API response shapes into (time, parameter, value).
    Adjust this once response format is known.
    """
    out: List[Tuple[str, str, object]] = []
    if not isinstance(payload, dict):
        return out

    data = payload.get("data", payload)
    # Shape A: {"data": [{"time": "...", "parameter": "...", "value": ...}, ...]}
    if isinstance(data, list):
        for rec in data:
            if not isinstance(rec, dict):
                continue
            t = rec.get("time") or rec.get("timestamp")
            p = rec.get("parameter") or rec.get("name")
            v = rec.get("value")
            if t is not None and p is not None:
                out.append((str(t), str(p), v))
        if out:
            return out

    # Shape B: {"data": {"ParamA": [{"time":"...","value":...}], ...}}
    if isinstance(data, dict):
        for p, series in data.items():
            if isinstance(series, list):
                for rec in series:
                    if not isinstance(rec, dict):
                        continue
                    t = rec.get("time") or rec.get("timestamp")
                    v = rec.get("value")
                    if t is not None:
                        out.append((str(t), str(p), v))
    return out


def run_query_api(
    repo_root: Path,
    endpoint: str,
    filler_filename: str = "Filler.csv",
    output_filename: str = "Filler_populated.csv",
    time_col: str = "Time",
    params_per_query: int = 6,
    times_per_query: int = 72,
    request_timeout_s: int = 60,
    dry_run: bool = False,
    extra_payload: Optional[Dict] = None,
) -> None:
    sensors_dir = repo_root / "data" / "input" / "sensors"
    debug_dir = repo_root / "data" / "output" / "api_debug" / "weather_refill"
    debug_dir.mkdir(parents=True, exist_ok=True)

    filler_path = sensors_dir / filler_filename
    if not filler_path.exists():
        raise FileNotFoundError(f"Filler file not found: {filler_path}")

    df = pd.read_csv(filler_path, sep=";", low_memory=False)
    if time_col not in df.columns:
        raise ValueError(f"Expected '{time_col}' column in {filler_path.name}")

    # Rows/times where any parameter is still missing.
    param_cols = [c for c in df.columns if c != time_col]
    if not param_cols:
        raise ValueError("No parameter columns found in filler file.")

    time_series = pd.to_datetime(df[time_col], errors="coerce")
    missing_mask = df[param_cols].isna().any(axis=1)
    missing_times = (
        time_series[missing_mask]
        .dropna()
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
        .drop_duplicates()
        .tolist()
    )

    if not missing_times:
        print("No missing rows detected; nothing to query.")
        return

    client_id = os.getenv("WEATHER_API_CLIENT_ID")
    token = os.getenv("WEATHER_API_TOKEN")
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if client_id:
        headers["X-Client-Id"] = client_id
    if token:
        headers["Authorization"] = f"Bearer {token}"

    query_idx = 0
    session = requests.Session()
    total_filled = 0

    for t_batch in chunked(missing_times, max(1, times_per_query)):
        for p_batch in chunked(param_cols, max(1, params_per_query)):
            query_idx += 1
            payload = {
                "times": t_batch,
                "parameters": p_batch,
            }
            if extra_payload:
                payload.update(extra_payload)

            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            req_file = debug_dir / f"query_{query_idx:04d}_{stamp}_request.json"
            with req_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            response_json: object = {}
            response_status: object = "dry_run"
            response_error = None

            if not dry_run:
                try:
                    response = session.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=request_timeout_s,
                    )
                    response_status = response.status_code
                    try:
                        response_json = response.json()
                    except Exception:
                        response_json = {"raw_text": response.text}
                except Exception as exc:
                    response_error = str(exc)

            resp_file = debug_dir / f"query_{query_idx:04d}_{stamp}_response.json"
            with resp_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "status": response_status,
                        "error": response_error,
                        "response": response_json,
                    },
                    f,
                    indent=2,
                )

            if dry_run or response_error is not None:
                continue

            triples = extract_values_default(response_json)
            if not triples:
                continue

            for time_raw, param, value in triples:
                if param not in df.columns:
                    continue
                time_norm = pd.to_datetime(time_raw, errors="coerce")
                if pd.isna(time_norm):
                    continue
                time_key = time_norm.strftime("%Y-%m-%dT%H:%M:%S")
                value_t = default_param_transform(param, value)
                idx = df.index[df[time_col] == time_key]
                if len(idx) == 0:
                    continue
                if pd.isna(df.at[idx[0], param]):
                    df.at[idx[0], param] = value_t
                    total_filled += 1

    out_path = sensors_dir / output_filename
    df.to_csv(out_path, sep=";", index=False)
    print(f"Wrote populated file: {out_path}")
    print(f"Filled values: {total_filled}")
    print(f"Saved request/response JSON files in: {debug_dir}")


def build_thredds_url(base_url: str, filename_prefix: str, ts: pd.Timestamp) -> str:
    return (
        f"{base_url.rstrip('/')}/"
        f"{ts.strftime('%Y/%m/%d')}/"
        f"{filename_prefix}_{ts.strftime('%Y%m%dT%HZ')}.nc"
    )


def build_thredds_fallback_urls(primary_base_url: str, filename_prefix: str, ts: pd.Timestamp) -> List[str]:
    """Return ordered OPeNDAP URLs: LongTerm, MainArchive, v4, v3."""
    base_candidates = [
        ("https://thredds.met.no/thredds/dodsC/metppltcarchivev1", "met_analysis_ltc_1_0km_nordic"),
        (primary_base_url.rstrip("/"), filename_prefix),
        ("https://thredds.met.no/thredds/dodsC/metpparchivev4", filename_prefix),
        ("https://thredds.met.no/thredds/dodsC/metpparchivev3", filename_prefix),
    ]
    urls = [build_thredds_url(base_url=b, filename_prefix=p, ts=ts) for b, p in base_candidates]
    # De-duplicate while preserving order.
    return list(dict.fromkeys(urls))


def archive_label_from_url(url: str) -> str:
    if "metppltcarchivev1" in url:
        return "LongTerm"
    if "metpparchivev4" in url:
        return "RerunVersion4"
    if "metpparchivev3" in url:
        return "RerunVersion3"
    if "metpparchive" in url:
        return "MainArchive"
    return "UnknownArchive"


def _get_var_name(variables: Dict, candidates: List[str]) -> Optional[str]:
    lower_map = {name.lower(): name for name in variables.keys()}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def find_nearest_grid_index(
    ds,
    lat_target: float,
    lon_target: float,
    lat_var_candidates: List[str],
    lon_var_candidates: List[str],
) -> Tuple[int, int]:
    lat_name = _get_var_name(ds.variables, lat_var_candidates)
    lon_name = _get_var_name(ds.variables, lon_var_candidates)
    if lat_name is None or lon_name is None:
        raise KeyError("Could not find latitude/longitude variables in dataset.")

    lat_vals = np.array(ds.variables[lat_name][:], dtype=float)
    lon_vals = np.array(ds.variables[lon_name][:], dtype=float)

    lon_t = float(lon_target)
    if np.nanmin(lon_vals) >= 0 and lon_t < 0:
        lon_t = lon_t + 360.0

    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    elif lat_vals.shape == lon_vals.shape:
        lat_grid, lon_grid = lat_vals, lon_vals
    else:
        raise ValueError(
            f"Unsupported lat/lon shapes: lat={lat_vals.shape}, lon={lon_vals.shape}"
        )

    dist2 = (lat_grid - lat_target) ** 2 + (lon_grid - lon_t) ** 2
    y_idx, x_idx = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return int(y_idx), int(x_idx)


def extract_point_value(var, y_idx: int, x_idx: int) -> Optional[float]:
    dims = [d.lower() for d in var.dimensions]
    sel = []
    for i, dim in enumerate(dims):
        if "time" in dim:
            sel.append(0)
        elif ("lat" in dim) or (dim in {"y", "yc", "rlat"}):
            sel.append(y_idx)
        elif ("lon" in dim) or (dim in {"x", "xc", "rlon"}):
            sel.append(x_idx)
        else:
            size_i = var.shape[i] if i < len(var.shape) else 1
            sel.append(0 if size_i > 0 else slice(None))

    value = var[tuple(sel)]
    arr = np.ma.asarray(value).squeeze()
    if np.ma.is_masked(arr):
        if np.all(arr.mask):
            return None
        arr = arr.filled(np.nan)
    arr = np.asarray(arr)
    if arr.size == 0:
        return None
    scalar = arr.flat[0]
    if pd.isna(scalar):
        return None
    try:
        return float(scalar)
    except Exception:
        return None


def interpolate_short_gaps_timewise(
    df: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    max_consecutive_missing_to_fill: int = 2,
) -> pd.DataFrame:
    """
    Replace invalid markers with missing and interpolate short internal gaps.

    With hourly sampling and max_consecutive_missing_to_fill=2:
      - 1-2 consecutive missing rows are interpolated
      - 3+ consecutive missing rows remain missing
    """
    invalid_markers = {
        "",
        " ",
        "NA",
        "N/A",
        "NaN",
        "nan",
        "#N/A",
        "-999",
        "-999.0",
        "-9999",
        "-9999.0",
        "-99",
        "-99.0",
        "-99.9",
    }

    out = df.copy()
    out["_time_tmp_"] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.sort_values("_time_tmp_").set_index("_time_tmp_")

    for col in value_cols:
        if col not in out.columns:
            continue
        s = out[col].replace(list(invalid_markers), np.nan)
        s = pd.to_numeric(s, errors="coerce")
        s = s.interpolate(
            method="time",
            limit=max_consecutive_missing_to_fill,
            limit_direction="both",
            limit_area="inside",
        )
        out[col] = s

    out = out.reset_index(drop=True)
    return out


def get_thredds_weather_mappings() -> List[Tuple[str, str]]:
    """(THREDDS element, Weather.csv semantic key) mappings confirmed for refill."""
    return [
        ("air_pressure_at_sea_level", "pr_trykk_redusert"),
        ("air_temperature_2m", "ta_middel"),
        ("integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time", "qsi_kortbolget"),
        ("integral_of_surface_downwelling_longwave_flux_in_air_wrt_time", "qli_langbolget"),
        ("precipitation_amount", "rr_1"),
        ("relative_humidity_2m", "uu_luftfuktighet"),
        ("wind_direction_10m", "dx_l"),
        ("wind_speed_10m", "ff_hastighet"),
    ]


def resolve_weather_columns(weather_columns: List[str]) -> Dict[str, str]:
    """Resolve target Weather.csv columns by robust token matching."""
    patterns = {
        "pr_trykk_redusert": ["pr trykk redusert"],
        "ta_middel": ["ta middel"],
        "qsi_kortbolget": ["qsi kort"],
        "qli_langbolget": ["qli lang", "qli "],
        "rr_1": ["rr_1"],
        "uu_luftfuktighet": ["uu luftfuktighet"],
        "dx_l": ["dx_l"],
        "ff_hastighet": ["ff hastighet"],
    }
    resolved: Dict[str, str] = {}
    lower_cols = {c.lower(): c for c in weather_columns}
    for key, tokens in patterns.items():
        chosen = None
        for col_lower, original in lower_cols.items():
            if any(tok in col_lower for tok in tokens):
                chosen = original
                break
        if chosen is not None:
            resolved[key] = chosen
    return resolved


def transform_thredds_series_for_weather_compare(element: str, series: pd.Series) -> pd.Series:
    """
    Apply unit conversions with guards to avoid double-converting already converted files.
    """
    s = pd.to_numeric(series, errors="coerce")
    s_non_na = s.dropna()
    if s_non_na.empty:
        return s

    median_val = float(s_non_na.median())
    p95_val = float(s_non_na.quantile(0.95))

    # Kelvin -> Celsius (only if values look like Kelvin)
    if element == "air_temperature_2m":
        if median_val > 150.0:
            return s - 273.15
        return s

    # Pascal -> mBar (hPa) (only if values look like Pa)
    if element == "air_pressure_at_sea_level":
        if median_val > 2000.0:
            return s / 100.0
        return s

    # Fraction -> percent (only if values look fractional)
    if element == "relative_humidity_2m":
        if p95_val <= 1.5:
            return s * 100.0
        return s

    # J/m2 over hour -> W/m2 (only if values look integral-scale)
    if element == "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time":
        if median_val > 2000.0:
            return s / 3600.0
        return s
    if element == "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time":
        if median_val > 2000.0:
            return s / 3600.0
        return s

    return s


def apply_thredds_unit_conversions_for_write(
    df: pd.DataFrame,
    out_cols: Dict[str, str],
) -> pd.DataFrame:
    """
    Deterministically convert raw THREDDS columns before writing to Filler_thredds.csv.
    """
    out = df.copy()
    for element, col in out_cols.items():
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        if element == "air_temperature_2m":
            out[col] = s - 273.15
        elif element == "air_pressure_at_sea_level":
            out[col] = s / 100.0
        elif element == "relative_humidity_2m":
            out[col] = s * 100.0
        elif element == "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time":
            out[col] = s / 3600.0
        elif element == "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time":
            out[col] = s / 3600.0
        else:
            out[col] = s
    return out


def infer_decimal_places_by_column(weather_raw_df: pd.DataFrame) -> Dict[str, int]:
    """
    Infer decimal places used by each original Weather.csv column from raw string values.
    """
    invalid_tokens = {"", " ", "NA", "N/A", "NaN", "nan", "#N/A", "-99,9", "-99.9", "-99", "-999", "-9999"}
    decimal_map: Dict[str, int] = {}
    for col in weather_raw_df.columns:
        if col == "Time":
            continue
        counts: Dict[int, int] = {}
        for v in weather_raw_df[col].dropna():
            s = str(v).strip()
            if s in invalid_tokens:
                continue
            if "," in s:
                dec = len(s.split(",")[-1])
            elif "." in s:
                dec = len(s.split(".")[-1])
            else:
                dec = 0
            counts[dec] = counts.get(dec, 0) + 1
        if counts:
            # Most common decimal precision; break ties toward fewer decimals.
            decimal_map[col] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return decimal_map


def format_numeric_no_scientific(value: object, decimals: int) -> object:
    if pd.isna(value):
        return pd.NA
    try:
        f = float(value)
    except Exception:
        return value
    txt = f"{round(f, max(0, decimals)):.{max(0, decimals)}f}"
    return txt.replace(".", ",")


def build_series_with_hourly_gap_breaks(
    times: pd.Series,
    values: pd.Series,
    max_gap_hours: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """
    Return time/value series where gaps > max_gap_hours are represented as NaN breaks.
    """
    tmp = pd.DataFrame(
        {
            "Time": pd.to_datetime(times, errors="coerce"),
            "Value": pd.to_numeric(values, errors="coerce"),
        }
    ).dropna(subset=["Time"])
    if tmp.empty:
        return pd.Series(dtype="datetime64[ns]"), pd.Series(dtype="float64")

    tmp = tmp.sort_values("Time").drop_duplicates(subset=["Time"], keep="first").set_index("Time")
    # Reindex to hourly grid so missing timestamps create NaN and break plotted lines.
    full_index = pd.date_range(start=tmp.index.min(), end=tmp.index.max(), freq="h")
    aligned = tmp.reindex(full_index)
    if max_gap_hours > 1:
        # For this script we only need hourly break logic; keep simple for now.
        pass
    return pd.Series(aligned.index), aligned["Value"]


def generate_thredds_overlay_figure(
    repo_root: Path,
    filler_thredds_filename: str = "Filler_thredds.csv",
    weather_filename: str = "Weather.csv",
    output_figure: str = "Weather_thredds_overlay.png",
) -> None:
    sensors_dir = repo_root / "data" / "input" / "sensors"
    summaries_dir = repo_root / "data" / "sensors" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    filler_path = sensors_dir / filler_thredds_filename
    weather_path = sensors_dir / weather_filename
    if not filler_path.exists():
        raise FileNotFoundError(f"THREDDS output file not found: {filler_path}")
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather file not found: {weather_path}")

    filler_df = pd.read_csv(filler_path, sep=";", low_memory=False)
    weather_df = pd.read_csv(weather_path, sep=";", decimal=",", low_memory=False)
    if "Time" not in filler_df.columns or "Time" not in weather_df.columns:
        raise ValueError("Expected 'Time' in both files.")

    filler_df["Time"] = pd.to_datetime(filler_df["Time"], errors="coerce")
    weather_df["Time"] = pd.to_datetime(weather_df["Time"], errors="coerce")

    mappings = get_thredds_weather_mappings()
    resolved_cols = resolve_weather_columns(list(weather_df.columns))
    n = len(mappings)
    fig, axes = plt.subplots(n, 1, figsize=(16, max(10, n * 2.6)), sharex=True)
    if n == 1:
        axes = [axes]
    legend_handles = None

    thredds_time_min = None
    thredds_time_max = None
    thredds_cols_present = [f"thredds_{el}" for el, _ in mappings if f"thredds_{el}" in filler_df.columns]
    if thredds_cols_present:
        th_valid_any = pd.Series(False, index=filler_df.index)
        for c in thredds_cols_present:
            th_valid_any = th_valid_any | pd.to_numeric(filler_df[c], errors="coerce").notna()
        if th_valid_any.any():
            th_times = filler_df.loc[th_valid_any, "Time"].dropna()
            if not th_times.empty:
                thredds_time_min = th_times.min()
                thredds_time_max = th_times.max()

    for ax, (element, weather_key) in zip(axes, mappings):
        weather_col = resolved_cols.get(weather_key)
        th_col = f"thredds_{element}"
        if th_col not in filler_df.columns or weather_col is None:
            ax.text(0.01, 0.5, f"Missing columns for mapping:\n{th_col}\n{weather_key}", transform=ax.transAxes)
            ax.set_title(f"{element} | Weather: {weather_key}", fontsize=9, loc="left")
            ax.grid(alpha=0.2, linestyle="--")
            continue

        th_t_raw = filler_df["Time"]
        th_v_raw = transform_thredds_series_for_weather_compare(element, filler_df[th_col])
        w_t_raw = weather_df["Time"]
        w_v_raw = pd.to_numeric(weather_df[weather_col], errors="coerce")
        th_t, th_v = build_series_with_hourly_gap_breaks(th_t_raw, th_v_raw, max_gap_hours=1)
        w_t, w_v = build_series_with_hourly_gap_breaks(w_t_raw, w_v_raw, max_gap_hours=1)

        line_weather, = ax.plot(w_t, w_v, color="#4C78A8", linewidth=1.0, alpha=0.8, label="NTNU weather station")
        line_thredds, = ax.plot(th_t, th_v, color="#F58518", linewidth=0.9, alpha=0.9, label="MET Analysis")
        if legend_handles is None:
            legend_handles = (line_weather, line_thredds)
        ax.set_title(f"{element} | Weather: {weather_col}", fontsize=9, loc="left")
        ax.grid(alpha=0.25, linestyle="--")

    if legend_handles is not None:
        axes[0].legend(list(legend_handles), ["NTNU weather station", "MET Analysis"], loc="upper right")
    axes[-1].set_xlabel("Time")
    if (thredds_time_min is not None) and (thredds_time_max is not None):
        x_min = thredds_time_min - pd.DateOffset(months=1)
        x_max = thredds_time_max + pd.DateOffset(months=1)
        for ax in axes:
            ax.set_xlim(x_min, x_max)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.98, bottom=0.07, hspace=0.5)
    fig.tight_layout()

    out_path = summaries_dir / output_figure
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote overlay figure: {out_path}")


def combine_weather_with_thredds(
    repo_root: Path,
    weather_filename: str = "Weather.csv",
    filler_thredds_filename: str = "Filler_thredds.csv",
    output_filename: str = "Weather_combo.csv",
) -> None:
    """
    Combine Weather.csv with locally downloaded THREDDS values.

    - Keeps Weather.csv schema/columns.
    - Fills mapped target columns where Weather.csv is missing.
    - Allows additional timestamps from THREDDS file (other columns stay blank).
    """
    sensors_dir = repo_root / "data" / "input" / "sensors"
    weather_path = sensors_dir / weather_filename
    thredds_path = sensors_dir / filler_thredds_filename
    out_path = sensors_dir / output_filename

    if not weather_path.exists():
        raise FileNotFoundError(f"Weather file not found: {weather_path}")
    if not thredds_path.exists():
        raise FileNotFoundError(f"THREDDS file not found: {thredds_path}")

    weather_raw_df = pd.read_csv(weather_path, sep=";", dtype=str, low_memory=False)
    weather_df = pd.read_csv(weather_path, sep=";", decimal=",", low_memory=False)
    thredds_df = pd.read_csv(thredds_path, sep=";", low_memory=False)
    if "Time" not in weather_df.columns or "Time" not in thredds_df.columns:
        raise ValueError("Expected 'Time' column in both weather and THREDDS files.")

    weather_df["Time"] = pd.to_datetime(weather_df["Time"], errors="coerce")
    thredds_df["Time"] = pd.to_datetime(thredds_df["Time"], errors="coerce")
    weather_df = weather_df.dropna(subset=["Time"]).drop_duplicates(subset=["Time"]).set_index("Time").sort_index()
    thredds_df = thredds_df.dropna(subset=["Time"]).drop_duplicates(subset=["Time"]).set_index("Time").sort_index()

    # Include union of timestamps so THREDDS-only times can be retained.
    all_times = weather_df.index.union(thredds_df.index).sort_values()
    combo_df = weather_df.reindex(all_times)

    mappings = get_thredds_weather_mappings()
    resolved_cols = resolve_weather_columns(list(weather_df.columns))
    decimal_places = infer_decimal_places_by_column(weather_raw_df)
    invalid_numeric_markers = {-9999.0, -999.0, -99.9, -99.0}
    fill_report = []
    for element, weather_key in mappings:
        th_col = f"thredds_{element}"
        weather_col = resolved_cols.get(weather_key)
        if weather_col is None or weather_col not in combo_df.columns:
            fill_report.append((weather_key, th_col, 0, "weather column missing"))
            continue
        if th_col not in thredds_df.columns:
            fill_report.append((weather_col, th_col, 0, "thredds column missing"))
            continue

        weather_vals = pd.to_numeric(combo_df[weather_col], errors="coerce")
        weather_vals = weather_vals.mask(weather_vals.isin(list(invalid_numeric_markers)), np.nan)
        th_vals = transform_thredds_series_for_weather_compare(element, thredds_df[th_col]).reindex(all_times)
        before_missing = int(weather_vals.isna().sum())
        filled_series = weather_vals.fillna(th_vals)
        after_missing = int(pd.to_numeric(filled_series, errors="coerce").isna().sum())
        decimals = decimal_places.get(weather_col, 2)
        combo_df[weather_col] = filled_series.map(lambda x: format_numeric_no_scientific(x, decimals))
        fill_report.append((weather_col, th_col, before_missing - after_missing, "ok"))

    combo_df = combo_df.reset_index().rename(columns={"index": "Time"})
    combo_df["Time"] = combo_df["Time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    combo_df = combo_df[weather_df.reset_index().columns]
    combo_df.to_csv(out_path, sep=";", decimal=",", index=False)

    print(f"Wrote combined weather file: {out_path}")
    print("Fill summary (weather_col <- thredds_col):")
    for weather_col, th_col, filled, status in fill_report:
        print(f"  - {weather_col} <- {th_col}: filled={filled} ({status})")


def run_query_thredds(
    repo_root: Path,
    weather_filename: str = "Weather.csv",
    output_filename: str = "Filler_thredds.csv",
    time_col: str = "Time",
    start: str = "2022-01-01 00:00:00",
    end: str = "2025-03-01 00:00:00",
    lat: float = 62.484785020758075,
    lon: float = 6.479653454212095,
    base_url: str = "https://thredds.met.no/thredds/dodsC/metpparchive",
    filename_prefix: str = "met_analysis_1_0km_nordic",
    elements: Optional[List[str]] = None,
    checkpoint_every: int = 168,
) -> None:
    if nc is None:
        raise ImportError(
            "netCDF4 is required for THREDDS mode. Install with: pip install netCDF4"
        )

    default_elements = [
        "air_pressure_at_sea_level",
        "air_temperature_2m",
        "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time",
        "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time",
        "precipitation_amount",
        "relative_humidity_2m",
        "wind_direction_10m",
        "wind_speed_10m",
    ]
    elements = elements or default_elements

    sensors_dir = repo_root / "data" / "input" / "sensors"
    debug_dir = repo_root / "data" / "output" / "api_debug" / "weather_refill"
    debug_dir.mkdir(parents=True, exist_ok=True)

    weather_path = sensors_dir / weather_filename
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather file not found: {weather_path}")

    weather_df = pd.read_csv(weather_path, sep=";", decimal=",", low_memory=False)
    if time_col not in weather_df.columns:
        raise ValueError(f"Expected '{time_col}' column in {weather_path.name}")

    # Determine which rows need refill: mapped Weather columns missing/invalid within the date window.
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    weather_df["_time_tmp_"] = pd.to_datetime(weather_df[time_col], errors="coerce")
    in_window = (
        weather_df["_time_tmp_"].notna()
        & (weather_df["_time_tmp_"] >= start_ts)
        & (weather_df["_time_tmp_"] <= end_ts)
    )

    invalid_numeric_markers = {-9999.0, -999.0, -99.9, -99.0}
    selected_elements = set(elements)
    resolved_cols = resolve_weather_columns(list(weather_df.columns))
    mapped_weather_cols = [
        resolved_cols[w_key]
        for el, w_key in get_thredds_weather_mappings()
        if (el in selected_elements) and (w_key in resolved_cols)
    ]
    if not mapped_weather_cols:
        raise ValueError("None of the mapped Weather columns were found in Weather.csv")

    missing_mask = pd.Series(False, index=weather_df.index)
    for w_col in mapped_weather_cols:
        s = pd.to_numeric(weather_df[w_col], errors="coerce")
        s = s.mask(s.isin(list(invalid_numeric_markers)), np.nan)
        missing_mask = missing_mask | s.isna()

    query_df = weather_df.loc[in_window & missing_mask, [time_col, "_time_tmp_"]].copy()
    query_df = query_df.drop_duplicates(subset=[time_col]).sort_values("_time_tmp_").reset_index(drop=True)
    missing_value_rows = len(query_df)

    # Also include timestamps entirely absent from Weather.csv in the selected window.
    existing_times = weather_df.loc[in_window, "_time_tmp_"].dropna().drop_duplicates().sort_values()
    expected_times = pd.date_range(start=start_ts, end=end_ts, freq="h")
    missing_times = expected_times.difference(existing_times)
    if len(missing_times) > 0:
        missing_df = pd.DataFrame({"_time_tmp_": missing_times})
        missing_df[time_col] = missing_df["_time_tmp_"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        query_df = pd.concat([query_df, missing_df[[time_col, "_time_tmp_"]]], ignore_index=True)
        query_df = query_df.drop_duplicates(subset=[time_col]).sort_values("_time_tmp_").reset_index(drop=True)
    if query_df.empty:
        print(
            f"No missing mapped weather rows found in window "
            f"{start_ts.strftime('%Y-%m-%d %H:%M:%S')} to {end_ts.strftime('%Y-%m-%d %H:%M:%S')}."
        )
        return

    out_cols = {el: f"thredds_{el}" for el in elements}
    for c in out_cols.values():
        query_df[c] = pd.NA

    # Cache nearest grid index after first successful open.
    grid_idx: Optional[Tuple[int, int]] = None
    errors = []
    files_opened = 0
    values_filled = 0
    archive_hits = {"MainArchive": 0, "RerunVersion4": 0, "RerunVersion3": 0, "LongTerm": 0, "UnknownArchive": 0}
    total_rows = len(query_df)
    out_path = sensors_dir / output_filename
    checkpoint_every = max(1, int(checkpoint_every))
    print(
        f"Starting THREDDS lookup for {total_rows} samples "
        f"in window {start_ts.strftime('%Y-%m-%d')} to {end_ts.strftime('%Y-%m-%d')} "
        f"(checkpoint/log every {checkpoint_every} samples)."
    )
    print(
        f"  - missing mapped values in existing Weather rows: {missing_value_rows}\n"
        f"  - missing hourly rows in window: {len(missing_times)}"
    )
    max_retries_per_source = 3
    retry_backoff_seconds = 1.0

    for sample_num, (row_idx, row) in enumerate(query_df.iterrows(), start=1):
        ts = pd.to_datetime(row["_time_tmp_"], errors="coerce")
        if pd.isna(ts):
            errors.append({"row": int(row_idx), "time": row.get(time_col), "error": "Invalid timestamp"})
            continue

        candidate_urls = build_thredds_fallback_urls(
            primary_base_url=base_url,
            filename_prefix=filename_prefix,
            ts=ts,
        )
        url = candidate_urls[0]
        if sample_num % checkpoint_every == 0:
            print(
                f"[progress] sample {sample_num}/{total_rows} | "
                f"time={ts.strftime('%Y-%m-%dT%H:%M:%S')} | url={url}"
            )
        open_cache: Dict[str, Optional[object]] = {}
        attempt_errors: Dict[str, str] = {}

        def get_dataset(url_key: str, force_reopen: bool = False):
            nonlocal files_opened
            if force_reopen and (url_key in open_cache) and (open_cache[url_key] is not None):
                try:
                    open_cache[url_key].close()
                except Exception:
                    pass
                open_cache.pop(url_key, None)
            if url_key in open_cache:
                return open_cache[url_key]
            try:
                ds_obj = nc.Dataset(url_key)
                files_opened += 1
                archive_hits[archive_label_from_url(url_key)] += 1
                open_cache[url_key] = ds_obj
                return ds_obj
            except Exception as exc:
                attempt_errors[url_key] = str(exc)
                open_cache[url_key] = None
                return None

        # Ensure we have grid coordinates from the first reachable source.
        if grid_idx is None:
            for candidate_url in candidate_urls:
                ds_probe = get_dataset(candidate_url)
                if ds_probe is None:
                    continue
                grid_idx = find_nearest_grid_index(
                    ds=ds_probe,
                    lat_target=lat,
                    lon_target=lon,
                    lat_var_candidates=["latitude", "lat", "y"],
                    lon_var_candidates=["longitude", "lon", "x"],
                )
                print(f"Using grid index y={grid_idx[0]}, x={grid_idx[1]} for lat={lat}, lon={lon}")
                break

        if grid_idx is None:
            errors.append(
                {
                    "row": int(row_idx),
                    "time": ts.strftime("%Y-%m-%dT%H:%M:%S") if not pd.isna(ts) else str(row.get(time_col)),
                    "attempts": [{"url": u, "error": attempt_errors.get(u, "untried")} for u in candidate_urls],
                    "error": "All archive fallbacks failed (no dataset opened for grid lookup)",
                }
            )
            for ds_obj in open_cache.values():
                if ds_obj is not None:
                    try:
                        ds_obj.close()
                    except Exception:
                        pass
            continue

        y_idx, x_idx = grid_idx
        for el in elements:
            out_col = out_cols[el]
            if not pd.isna(query_df.at[row_idx, out_col]):
                continue

            filled_this_element = False
            per_element_attempts = []
            for candidate_url in candidate_urls:
                # Retry open/read on each source to handle transient OPeNDAP parser/network failures.
                for attempt_idx in range(1, max_retries_per_source + 1):
                    ds_use = get_dataset(candidate_url, force_reopen=(attempt_idx > 1))
                    if ds_use is None:
                        err_text = attempt_errors.get(candidate_url, "open failed")
                        per_element_attempts.append(
                            {
                                "url": candidate_url,
                                "attempt": attempt_idx,
                                "stage": "open",
                                "error": err_text,
                            }
                        )
                        errors.append(
                            {
                                "row": int(row_idx),
                                "time": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                                "element": el,
                                "url": candidate_url,
                                "attempt": attempt_idx,
                                "stage": "open",
                                "error": err_text,
                            }
                        )
                        if attempt_idx < max_retries_per_source:
                            time.sleep(retry_backoff_seconds * attempt_idx)
                        continue

                    if el not in ds_use.variables:
                        per_element_attempts.append(
                            {
                                "url": candidate_url,
                                "attempt": attempt_idx,
                                "stage": "read",
                                "error": "Missing variable",
                            }
                        )
                        # No point retrying same source if variable does not exist there.
                        break

                    try:
                        val = extract_point_value(ds_use.variables[el], y_idx=y_idx, x_idx=x_idx)
                        if val is None:
                            per_element_attempts.append(
                                {
                                    "url": candidate_url,
                                    "attempt": attempt_idx,
                                    "stage": "read",
                                    "error": "Variable present but value missing",
                                }
                            )
                            # Missing value is usually not transient; move to next source.
                            break
                        query_df.at[row_idx, out_col] = val
                        values_filled += 1
                        filled_this_element = True
                        break
                    except Exception as exc:
                        err_text = str(exc)
                        per_element_attempts.append(
                            {
                                "url": candidate_url,
                                "attempt": attempt_idx,
                                "stage": "read",
                                "error": err_text,
                            }
                        )
                        errors.append(
                            {
                                "row": int(row_idx),
                                "time": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                                "element": el,
                                "url": candidate_url,
                                "attempt": attempt_idx,
                                "stage": "read",
                                "error": err_text,
                            }
                        )
                        if attempt_idx < max_retries_per_source:
                            time.sleep(retry_backoff_seconds * attempt_idx)
                            continue
                        # Exhausted retries on this source, try next fallback source.
                        break

                if filled_this_element:
                    break

            if not filled_this_element:
                errors.append(
                    {
                        "row": int(row_idx),
                        "time": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "element": el,
                        "attempts": per_element_attempts,
                        "error": "All archive fallbacks failed for element",
                    }
                )

        for ds_obj in open_cache.values():
            if ds_obj is not None:
                try:
                    ds_obj.close()
                except Exception:
                    pass

        if sample_num % checkpoint_every == 0:
            checkpoint_df = query_df.drop(columns=["_time_tmp_"], errors="ignore")
            checkpoint_df = apply_thredds_unit_conversions_for_write(checkpoint_df, out_cols)
            checkpoint_df.to_csv(out_path, sep=";", index=False)
            print(f"[checkpoint] wrote partial output: {out_path} at sample {sample_num}/{total_rows}")

    thredds_cols = list(out_cols.values())
    query_df = apply_thredds_unit_conversions_for_write(
        query_df.drop(columns=["_time_tmp_"], errors="ignore"),
        out_cols,
    )
    query_df = interpolate_short_gaps_timewise(
        df=query_df,
        time_col=time_col,
        value_cols=thredds_cols,
        max_consecutive_missing_to_fill=2,
    )
    query_df.to_csv(out_path, sep=";", index=False)
    print(f"Wrote THREDDS output file: {out_path}")
    print(f"Files opened: {files_opened}")
    print(f"Values filled: {values_filled}")
    print(
        "Archive usage: "
        f"MainArchive={archive_hits['MainArchive']}, "
        f"RerunVersion4={archive_hits['RerunVersion4']}, "
        f"RerunVersion3={archive_hits['RerunVersion3']}, "
        f"LongTerm={archive_hits['LongTerm']}"
    )

    error_path = debug_dir / "thredds_lookup_errors.json"
    with error_path.open("w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
    print(f"Wrote THREDDS error log: {error_path} (entries: {len(errors)})")
    generate_thredds_overlay_figure(
        repo_root=repo_root,
        filler_thredds_filename=output_filename,
        weather_filename="Weather.csv",
        output_figure="Weather_thredds_overlay.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weather refill helper.")
    sub = parser.add_subparsers(dest="command", required=False)

    p_template = sub.add_parser("make-template", help="Create Filler.csv from Weather.csv")
    p_template.add_argument("--start", default="2022-01-01 00:00:00")
    p_template.add_argument("--end", default="2025-03-01 00:00:00")

    p_query = sub.add_parser("query-api", help="Run batched API queries and optionally populate filler data")
    p_query.add_argument("--endpoint", required=True, help="API endpoint URL")
    p_query.add_argument("--filler-filename", default="Filler.csv")
    p_query.add_argument("--output-filename", default="Filler_populated.csv")
    p_query.add_argument("--params-per-query", type=int, default=6)
    p_query.add_argument("--times-per-query", type=int, default=72)
    p_query.add_argument("--timeout", type=int, default=60)
    p_query.add_argument("--dry-run", action="store_true")
    p_query.add_argument(
        "--extra-payload-json",
        default="",
        help="Path to JSON file with additional payload fields",
    )

    p_thredds = sub.add_parser("query-thredds", help="Read point data from THREDDS OPeNDAP files")
    p_thredds.add_argument("--weather-filename", default="Weather.csv")
    p_thredds.add_argument("--output-filename", default="Filler_thredds.csv")
    p_thredds.add_argument("--start", default="2022-01-01 00:00:00")
    p_thredds.add_argument("--end", default="2025-03-01 00:00:00")
    p_thredds.add_argument("--lat", type=float, default=62.484785020758075)
    p_thredds.add_argument("--lon", type=float, default=6.479653454212095)
    p_thredds.add_argument("--base-url", default="https://thredds.met.no/thredds/dodsC/metpparchive")
    p_thredds.add_argument("--filename-prefix", default="met_analysis_1_0km_nordic")
    p_thredds.add_argument("--checkpoint-every", type=int, default=168)
    p_thredds.add_argument(
        "--elements",
        default="air_pressure_at_sea_level,air_temperature_2m,integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time,integral_of_surface_downwelling_longwave_flux_in_air_wrt_time,precipitation_amount,relative_humidity_2m,wind_direction_10m,wind_speed_10m",
        help="Comma-separated list of variable names to extract",
    )

    p_plot = sub.add_parser("plot-thredds-overlay", help="Plot THREDDS columns over Weather.csv equivalents")
    p_plot.add_argument("--filler-thredds-filename", default="Filler_thredds.csv")
    p_plot.add_argument("--weather-filename", default="Weather.csv")
    p_plot.add_argument("--output-figure", default="Weather_thredds_overlay.png")

    p_combo = sub.add_parser("combine-weather", help="Combine Weather.csv with local THREDDS values")
    p_combo.add_argument("--weather-filename", default="Weather.csv")
    p_combo.add_argument("--filler-thredds-filename", default="Filler_thredds.csv")
    p_combo.add_argument("--output-filename", default="Weather_combo.csv")

    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args()
    cmd = args.command or "make-template"

    if cmd == "make-template":
        create_weather_refill_template(repo_root, start=args.start, end=args.end)
        return

    if cmd == "query-api":
        extra_payload = None
        if args.extra_payload_json:
            extra_path = Path(args.extra_payload_json)
            if not extra_path.exists():
                raise FileNotFoundError(f"Extra payload JSON file not found: {extra_path}")
            with extra_path.open("r", encoding="utf-8") as f:
                extra_payload = json.load(f)

        run_query_api(
            repo_root=repo_root,
            endpoint=args.endpoint,
            filler_filename=args.filler_filename,
            output_filename=args.output_filename,
            params_per_query=args.params_per_query,
            times_per_query=args.times_per_query,
            request_timeout_s=args.timeout,
            dry_run=args.dry_run,
            extra_payload=extra_payload,
        )
        return

    if cmd == "query-thredds":
        elements = [e.strip() for e in args.elements.split(",") if e.strip()]
        run_query_thredds(
            repo_root=repo_root,
            weather_filename=args.weather_filename,
            output_filename=args.output_filename,
            start=args.start,
            end=args.end,
            lat=args.lat,
            lon=args.lon,
            base_url=args.base_url,
            filename_prefix=args.filename_prefix,
            elements=elements,
            checkpoint_every=args.checkpoint_every,
        )
        return

    if cmd == "plot-thredds-overlay":
        generate_thredds_overlay_figure(
            repo_root=repo_root,
            filler_thredds_filename=args.filler_thredds_filename,
            weather_filename=args.weather_filename,
            output_figure=args.output_figure,
        )
        return

    if cmd == "combine-weather":
        combine_weather_with_thredds(
            repo_root=repo_root,
            weather_filename=args.weather_filename,
            filler_thredds_filename=args.filler_thredds_filename,
            output_filename=args.output_filename,
        )
        return

    raise ValueError(f"Unsupported command: {cmd}")


if __name__ == "__main__":
    main()

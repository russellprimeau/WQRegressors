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

import pandas as pd


def generate_sensor_summary(repo_root: Path) -> None:
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
        valid_mask = consolidated_df[col].notna()
        valid_count = valid_mask.sum()
        total_rows = len(consolidated_df)
        pct = (valid_count / total_rows * 100) if total_rows > 0 else 0.0

        valid_count_full = (
            df[sensor_name].notna().sum() if sensor_name in df.columns else 0
        )

        rows.append({
            "sensor": base_name,
            "unit": unit,
            "valid_count_consolidated": int(valid_count),
            "percent_valid_consolidated": round(pct, 2),
            "valid_count_fullhourly": int(valid_count_full),
        })

    out_path = out_dir / "FullHourly_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")


def generate_weather_summary(repo_root: Path) -> None:
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
    mapped_consolidated_cols = set()

    for col in weather_cols:
        english_name = full_weather_columns.get(col, col).replace('"', "").replace("'", "")
        m = re.match(r"(.+?)\s*\(([^)]+)\)$", english_name)
        base_name, unit = (m.group(1).strip(), m.group(2).strip()) if m else (english_name.strip(), "")

        valid_count_weather = weather_df[col].notna().sum()
        total_weather = len(weather_df)
        pct_weather = (valid_count_weather / total_weather * 100) if total_weather > 0 else 0.0

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
            valid_count_consolidated = consolidated_df[c].notna().sum()
            total_consolidated = len(consolidated_df)
            pct_consolidated = (valid_count_consolidated / total_consolidated * 100) if total_consolidated > 0 else 0.0
            rows.append({
                "parameter": base_name if c in (base_name, english_name) else c,
                "unit": unit,
                "valid_count_weather": int(valid_count_weather),
                "percent_valid_weather": round(pct_weather, 2),
                "valid_count_consolidated": int(valid_count_consolidated),
                "percent_valid_consolidated": round(pct_consolidated, 2),
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
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")


def generate_scada_summary(repo_root: Path) -> None:
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
    mapped_consolidated_cols = set()

    for col in scada_cols:
        base_name = col[len("SCADA - "):] if col.startswith("SCADA - ") else col
        base_name = base_name.replace('"', "").replace("'", "").strip()
        m = re.match(r"(.+?)\s*\(([^)]+)\)$", base_name)
        param_name, unit = (m.group(1).strip(), m.group(2).strip()) if m else (base_name, "")

        valid_count_scada = scada_df[col].notna().sum()
        total_scada = len(scada_df)
        pct_scada = (valid_count_scada / total_scada * 100) if total_scada > 0 else 0.0

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
        valid_count_consolidated = consolidated_df[c].notna().sum()
        total_consolidated = len(consolidated_df)
        pct_consolidated = (valid_count_consolidated / total_consolidated * 100) if total_consolidated > 0 else 0.0

        rows.append({
            "parameter": param_name,
            "unit": unit,
            "valid_count_scada": int(valid_count_scada),
            "percent_valid_scada": round(pct_scada, 2),
            "valid_count_consolidated": int(valid_count_consolidated),
            "percent_valid_consolidated": round(pct_consolidated, 2),
        })

    out_path = out_dir / "SCADA_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")


def generate_eurofins_summary(repo_root: Path) -> None:
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
    for param in summary_params:
        eurofins_col = next((k for k, v in eurofins_mapping.items() if v == param), None)
        m = re.match(r"(.+?)\s*\(([^)]+)\)$", param)
        param_name, unit = (m.group(1).strip(), m.group(2).strip()) if m else (param, "")

        limit_val = None
        if eurofins_col and eurofins_col in limits_mapping:
            try:
                limit_val = float(limits_mapping[eurofins_col])
            except (ValueError, TypeError):
                limit_val = None

        valid_count_eurofins = 0
        pct_eurofins = 0.0
        total_eurofins = len(eurofins_df)
        if eurofins_col and eurofins_col in eurofins_df.columns:
            valid_mask = (
                eurofins_df[eurofins_col].notna()
                & (eurofins_df[eurofins_col] != "#N/A")
            )
            valid_count_eurofins = valid_mask.sum()
            pct_eurofins = (valid_count_eurofins / total_eurofins * 100) if total_eurofins > 0 else 0.0

        col_candidates = [
            c for c in consolidated_df.columns
            if c.strip() == param or c.strip().lower().replace(" ", "") == param.lower().replace(" ", "")
        ]

        if col_candidates:
            c = col_candidates[0]
            valid_mask_cons = consolidated_df[c].notna()
            valid_count_consolidated = valid_mask_cons.sum()
            total_consolidated = len(consolidated_df)
            pct_consolidated = (valid_count_consolidated / total_consolidated * 100) if total_consolidated > 0 else 0.0
            exceed_count = 0
            if limit_val is not None:
                try:
                    exceed_count = int((valid_mask_cons & (consolidated_df[c] > limit_val)).sum())
                except Exception:
                    exceed_count = 0
        else:
            valid_count_consolidated = 0
            pct_consolidated = 0.0
            exceed_count = 0

        rows.append({
            "parameter": param_name,
            "unit": unit,
            "limit_value": limit_val,
            "valid_count_eurofins": int(valid_count_eurofins),
            "percent_valid_eurofins": round(pct_eurofins, 2),
            "valid_count_consolidated": int(valid_count_consolidated),
            "percent_valid_consolidated": round(pct_consolidated, 2),
            "count_exceed_limit": exceed_count,
        })

    out_path = out_dir / "Eurofins_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote summary to: {out_path}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    generate_sensor_summary(repo_root)
    generate_weather_summary(repo_root)
    generate_scada_summary(repo_root)
    generate_eurofins_summary(repo_root)


if __name__ == "__main__":
    main()

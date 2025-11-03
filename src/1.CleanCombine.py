"""
Combines datasets from the profiler, weather station, SCADA system and Eurofins reports in a single table.

Small gaps (below a specified threshold, default of 6 hours) are filled by linear interpolation.
When there is a gap in any one column, either all rows can be dropped, or "NaN" can be retained for that column.
"""

import pandas as pd
import numpy as np


def clean_profiler(full_df):
    """
    Clean profiler hourly surface data. Fill small gaps by interpolation.
    :param df: dataframe of profiler hourly surface data
    :return:
    """

    # Drop extraneous metadata columns
    sensor_columns = [col for col in full_df.columns if col.startswith("sensorParms")]
    df = full_df[["TIMESTAMP"] + sensor_columns].copy()

    # Convert TIMESTAMP to datetime, sort, and drop duplicates
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.sort_values("TIMESTAMP").drop_duplicates(subset="TIMESTAMP")

    # Replace -9999 and NaN with NaN for interpolation
    df[sensor_columns] = df[sensor_columns].replace([-9999, "NaN"], np.nan)

    # Round TIMESTAMP to the nearest hour
    df["TIMESTAMP"] = df["TIMESTAMP"].dt.round("h")

    # Fill gaps and track interpolated rows
    df = df.reset_index(drop=True)
    new_rows = []
    df["Interpolated"] = 0

    for i in range(1, len(df)):
        current_time = df.loc[i, "TIMESTAMP"]
        previous_time = df.loc[i - 1, "TIMESTAMP"]
        time_diff = (current_time - previous_time).total_seconds() / 3600

        if 1 < time_diff <= 6:
            for h in range(1, int(time_diff)):
                new_time = previous_time + pd.Timedelta(hours=h)
                new_row = {"TIMESTAMP": new_time, "Interpolated": 1}
                for col in sensor_columns:
                    new_row[col] = np.nan
                new_rows.append(new_row)

    # Append new rows and sort again
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)

    # Interpolate missing values and round to 2 decimals
    df[sensor_columns] = df[sensor_columns].interpolate(method="linear")
    df[sensor_columns] = df[sensor_columns].round(2)

    # Apply column labels
    column_names = {
        "TIMESTAMP": "TIMESTAMP",
        "RECORD": "Pfl - Record Number",
        "PFL_Counter": "Pfl - Day",
        "CntRS232": "Pfl - CntRS232",
        "RS232Dpt": "Pfl - Vertical Position1 (m)",
        "sensorParms(1)": "Pfl - Temp (C)",
        "sensorParms(2)": "Pfl - Cond (microS_cm)",
        "sensorParms(3)": "Pfl - Sp Cond (microS_cm)",
        "sensorParms(4)": "Pfl - Salinity (ppt)",
        "sensorParms(5)": "Pfl - pH",
        "sensorParms(6)": "Pfl - DO (% Sat)",
        "sensorParms(7)": "Pfl - Turbidity (NTU)",
        "sensorParms(8)": "Pfl - Turbidity (FNU)",
        "sensorParms(9)": "Pfl - Vertical Position (m)",
        "sensorParms(10)": "Pfl - fDOM (RFU)",
        "sensorParms(11)": "Pfl - fDOM (QSU)",
        "lat": "Latitude",
        "lon": "Longitude",
    }
    df = df.rename(columns=column_names)
    keepers = ["TIMESTAMP", "Interpolated", "Pfl - Temp (C)", "Pfl - Sp Cond (microS_cm)", "Pfl - pH",
               "Pfl - DO (% Sat)",
               "Pfl - Turbidity (FNU)", "Pfl - fDOM (RFU)", "Pfl - fDOM (QSU)"]
    df = df[keepers]
    return df

def add_source(df, secondary_df, include_NAs=False, max_gap=6):
    # Rename 'Time' in secondary_df to match 'TIMESTAMP' in primary_df
    secondary_df.rename(columns={"Time": "TIMESTAMP"}, inplace=True)
    # === MERGE ON TIMESTAMP ===
    merged_df = pd.merge(df, secondary_df, on="TIMESTAMP", how="left")


    # Identify new columns from secondary data
    new_columns = [col for col in merged_df.columns if col not in df.columns and col != "TIMESTAMP"]

    # Replace empty strings or whitespace with NaN
    merged_df[new_columns] = merged_df[new_columns].replace([r'^\s*$', "NA"], np.nan, regex=True)

    # Convert to numeric where possible
    merged_df[new_columns] = merged_df[new_columns].apply(pd.to_numeric, errors='coerce')

    # Create a mask where all new columns are NaN
    missing_mask = merged_df[new_columns].isna().all(axis=1)

    # Initialize list to collect indices of rows to drop
    rows_to_drop = []

    # Track consecutive missing groups
    start_idx = None
    for idx, is_missing in missing_mask.items():
        if is_missing:
            if start_idx is None:
                start_idx = idx
        else:
            if start_idx is not None:
                group = list(range(start_idx, idx))
                if len(group) > max_gap:
                    rows_to_drop.extend(group)
                start_idx = None

    # Handle case where missing group is at the end
    if start_idx is not None:
        group = list(range(start_idx, missing_mask.index[-1] + 1))
        if len(group) > max_gap:
            rows_to_drop.extend(group)

    # Drop rows
    if not include_NAs:
        merged_df.drop(index=rows_to_drop, inplace=True)
        # Interpolate remaining missing values in new columns
        merged_df[new_columns] = merged_df[new_columns].interpolate(method="linear", limit=max_gap, limit_direction="both")

    return merged_df

def count_segs(df):
    time_diff = df["TIMESTAMP"].diff()  # Calculate time difference between consecutive rows
    breaks = time_diff > pd.Timedelta(hours=1)  # Identify breaks (gaps greater than 1 hour)
    df["Segment"] = breaks.cumsum() + 1  # Start segments from 1  # Cumulatively sum the breaks

    # Move metadata columns to front
    cols = list(df.columns)
    cols.remove("Interpolated")
    cols.remove("Segment")
    cols.insert(1, "Interpolated")
    cols.insert(1, "Segment")
    df = df[cols]  # Reorder the DataFrame
    return df


if __name__ == '__main__':
    # Load data files
    df = pd.read_csv("../data/input/sensors/FullHourly.csv")  # Profiler data
    weather_df = pd.read_csv("../data/input/sensors/Weather.csv", sep=";", decimal=",", parse_dates=["Time"])
    scada_df = pd.read_csv("../data/input/sensors/SCADA.csv", sep=";", decimal=".", parse_dates=["Time"])
    eurofins_df = pd.read_csv("../data/input/sensors/Eurofins.csv", sep=";", decimal=",", parse_dates=["Time"])

    # Clean profiler dataset, which is foundation for other sources
    clean_df = clean_profiler(df)

    # Merge data from other sources into the dataset.
    merge1_df = add_source(clean_df, weather_df, include_NAs=False, max_gap=6)

    weather_columns = {"1818_time: AA[mBar]": "Instantaneous atmospheric pressure (mBar)",
                    "1818_time: DD Retning[°]": "Wind direction 10minRollingAvg (°)",
                    "1818_time: DX_l[°]": "Hourly average wind direction (°)",
                    "1818_time: FF Hastighet[m/s]": "Average wind speed (m/s)",
                    "1818_time: FG_l[m/s]": "Maximum sustained wind speed, 3-second span (m/s)",
                    "1818_time: FG_tid_l[N/A]": "Time of maximum 3s Gust",
                    "1818_time: FX Kast[m/s]": "Maximum sustained wind speed, 10-minute span (m/s)",
                    "1818_time: FX_tid_l[N/A]": "Time of maximum 10 minute gust",
                    "1818_time: PO Trykk stasjonshøyde[mBar]":
                        "Hourly average atmospheric pressure at station (mBar)",
                    "1818_time: PP[mBar]": "Maximum pressure differential, 3-hour span (mBar)",
                    "1818_time: PR Trykk redusert til havnivå[mBar]":
                        "Instantaneous atmospheric pressure compensated for temperature, humidity and station "
                        "elevation (mBar)",
                    "1818_time: QLI Langbølget[W/m2]": "Longwave (IR) radiation (W/m2)",
                    "1818_time: QNH[mBar]": "Instantaneous sea-level atmospheric pressure (mBar)",
                    "1818_time: QSI Kortbølget[W/m2]": "Shortwave (solar) radiation (W/m2)",
                    "1818_time: RR_1[mm]": "Precipitation (mm/hr)",
                    "1818_time: TA Middel[°C]": "Instantaneous temperature (°C)",
                    "1818_time: TA_a_Max[°C]": "Maximum temperature (°C)",
                    "1818_time: TA_a_Min[°C]": "Minimum temperature (°C)",
                    "1818_time: UU Luftfuktighet[%RH]": "Average humidity (% relative humidity)"
                    }
    merge1_df.rename(columns=weather_columns, inplace=True)
    merge2_df = add_source(merge1_df, scada_df, include_NAs=True, max_gap=6)
    merge3_df = add_source(merge2_df, eurofins_df, include_NAs=True, max_gap=6)

    segmented_df = count_segs(merge3_df)  # Add column with index for continuous segments

    # Save the cleaned and merged dataset
    segmented_df.to_csv("../data/output/for_regression/Combined_Cleaned.csv", index=False)

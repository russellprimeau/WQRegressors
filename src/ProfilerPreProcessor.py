'''
Combines datasets from the profiler, weather station, SCADA system and Eurofins reports in a single table.

Small gaps are filled by linear interpolation.
When there is a gap in any one column, either all rows can be dropped, or "NaN" can be retained for that column.
'''

import pandas as pd
import numpy as np

#######################################################################################################################
# Clean profiler hourly surface data. Fill small gaps by interpolation.

# Load the CSV file
df = pd.read_csv("../data/HourlyDemo.csv")

# Remove extraneous metadata
sensor_columns = [col for col in df.columns if col.startswith("sensorParms")]
df = df[["TIMESTAMP"] + sensor_columns]

# Convert TIMESTAMP to datetime, sort, and drop duplicates
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
df = df.sort_values("TIMESTAMP").drop_duplicates(subset="TIMESTAMP")

# Step 3: Replace -9999 and NaN with NaN for interpolation
df[sensor_columns] = df[sensor_columns].replace([-9999, "NaN"], np.nan)

# Step 4: Round TIMESTAMP to the nearest hour
df["TIMESTAMP"] = df["TIMESTAMP"].dt.round("h")

# Step 5: Add Segment column initialized to 1
df["Segment"] = 1

# Step 6: Fill gaps and track interpolated rows
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

# Step 7: Interpolate missing values and round to 2 decimals
df[sensor_columns] = df[sensor_columns].interpolate(method="linear")
df[sensor_columns] = df[sensor_columns].round(2)

#######################################################################################################################
# Add weather station data to the dataset.

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
    if include_NAs is False:
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

weather_file = "../data/Weather.csv"
weather_df = pd.read_csv(weather_file, sep=";", decimal=",", parse_dates=["Time"])
merge1_df = add_source(df, weather_df, include_NAs=False, max_gap=6)

scada_file = "../data/SCADA.csv"
scada_df = pd.read_csv(scada_file, sep=";", decimal=".", parse_dates=["Time"])
merge2_df = add_source(merge1_df, scada_df, include_NAs=True, max_gap=6)

eurofins_file = "../data/Eurofins.csv"
eurofins_df = pd.read_csv(eurofins_file, sep=";", decimal=",", parse_dates=["Time"])
merge3_df = add_source(merge2_df, eurofins_df, include_NAs=True, max_gap=6)

segmented_df = count_segs(merge3_df)

# Save the cleaned and merged dataset
segmented_df.to_csv("../data/Combined_Cleaned.csv", index=False)
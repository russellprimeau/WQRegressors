'''
Processes log files from the calibration of the YSI EXO sensors.
Creates a summary table for each sensor, with calibration events and the correction factors
(multiple factors for multi-point calibrations).

Plots the correction factors against the time between calibrations and/or temperature at calibration.

Apply a linear correction to each of the sensor columns based on the observed error at the time of calibration.
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import seaborn as sns
from datetime import timedelta

def sanitize_filename(name):
    """
    Sanitize text data: replace all non-alphanumeric characters with underscores.
    :param name:
    :return:
    """
    return re.sub(r'[\\/*?:"<>|]', "_", name)

# def parse_calibration_file(filepath):
#     """
#     Parse a single calibration log file
#     :param filepath:
#     :return:
#     """
#     with open(filepath, 'r', encoding='latin-1') as f:
#         content = f.read()
#
#     # Remove header lines like 'KorEXO Calibration File Export'
#     lines = content.strip().splitlines()
#     lines = [line for line in lines if not line.strip().startswith("KorEXO Calibration File Export")]
#     content = "\n".join(lines)
#
#     chapters = content.split("----------")
#     for chapter in chapters:
#         paragraphs = [p.strip() for p in chapter.strip().split('\n\n') if p.strip()]
#         if not paragraphs:
#             continue
#
#         # Extract metadata from the first paragraph
#         metadata_lines = paragraphs[0].split('\n')
#         metadata = {}
#         for line in metadata_lines:
#             if '=,' in line:
#                 key, value = map(str.strip, line.split('=,', 1))
#                 metadata[key] = value
#
#         # Extract data from subsequent paragraphs
#         data_entry = metadata.copy()
#         field_counts = defaultdict(int)
#         for para in paragraphs[1:]:
#             lines = para.strip().split('\n')
#             for line in lines[1:]:  # Skip first line (e.g., [Cal Point 1])
#                 if '=,' in line:
#                     key, value = map(str.strip, line.split('=,', 1))
#                     field_counts[key] += 1
#                     if field_counts[key] == 1 and key not in data_entry:
#                         data_entry[key] = value
#                     else:
#                         data_entry[f"{key} {field_counts[key]}"] = value
#
#             # Determine chapter name
#             chapter_name = metadata.get("Parameter Type", "Unknown")
#             chapter_dataframes[chapter_name].append(data_entry)

def parse_calibration_files(directory):
    chapter_dataframes = defaultdict(list)

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()

            # Remove header lines like 'KorEXO Calibration File Export'
            lines = content.strip().splitlines()
            lines = [line for line in lines if not line.strip().startswith("KorEXO Calibration File Export")]
            content = "\n".join(lines)

            # Split into chapters using '----------'
            chapters = content.split("----------")

            for chapter in chapters:
                paragraphs = [p.strip() for p in chapter.strip().split('\n\n') if p.strip()]
                if not paragraphs:
                    continue

                # Extract metadata from the first paragraph
                metadata_lines = paragraphs[0].splitlines()
                metadata = {}
                for line in metadata_lines:
                    if '=,' in line:
                        key, value = map(str.strip, line.split('=,', 1))
                        metadata[key] = value

                # Extract chapter name
                chapter_name = metadata.get("Parameter Type", "Unknown")
                print(f"Creating chapter '{chapter_name}' from file: {filename}")

                # Extract data paragraphs - combine all Cal Points into single entry
                data_entry = metadata.copy()
                field_counter = defaultdict(int)
                for para in paragraphs[1:]:
                    lines = para.strip().splitlines()
                    for line in lines[1:]:  # Skip first line (e.g., [Cal Point 1])
                        if '=,' in line:
                            key, value = map(str.strip, line.split('=,', 1))
                            field_counter[key] += 1
                            if field_counter[key] == 1:
                                data_entry[key] = value
                            else:
                                data_entry[f"{key} {field_counter[key]}"] = value
                chapter_dataframes[chapter_name].append(data_entry)

    # Convert lists of dicts to DataFrames
    for chapter in chapter_dataframes:
        chapter_dataframes[chapter] = pd.DataFrame(chapter_dataframes[chapter])

    return chapter_dataframes

def summarize(relative_dir):
    """

    :param relative_dir:
    :return:
    """
    csv_files = [f for f in os.listdir(relative_dir) if f.endswith('.csv')]

    # Dictionary to hold dataframes grouped by Parameter Type
    chapter_dataframes = defaultdict(list)



    # Example usage
    # directory = "."  # Current directory
    chapter_dfs = parse_calibration_files(relative_dir)

    # # Display the chapter names and first few rows of each DataFrame
    # for chapter, df in chapter_dfs.items():
    #     print(f"\nChapter: {chapter}")
    #     print(df.head())
    #


    # # Parse all CSV files in the directory
    # for file in csv_files:
    #     parse_calibration_files(Path(relative_dir, file))

    # Convert lists of dicts to DataFrames and process datetime columns
    final_dataframes = {}
    for chapter, entries in chapter_dfs.items():
        df = pd.DataFrame(entries)

        for col in ['Calibration End Time', 'Last Calibration Time', 'Calibration Start Time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        if 'Calibration End Time' in df.columns and 'Last Calibration Time' in df.columns:
            df['Timespan'] = df['Calibration End Time'] - df['Last Calibration Time']
        if 'Post Calibration Value' in df.columns and 'Pre Calibration Value' in df.columns:
            df['Post Calibration Value'] = df['Post Calibration Value'].str.split(expand=True)[0].astype(float)
            df['Pre Calibration Value'] = df['Pre Calibration Value'].str.split(expand=True)[0].astype(float)
            df['Correction1'] = (
                    df['Post Calibration Value'] - df['Pre Calibration Value'])  # /df['Post Calibration Value']
        if 'Post Calibration Value 2' in df.columns and 'Pre Calibration Value 2' in df.columns:
            df['Post Calibration Value 2'] = df['Post Calibration Value 2'].str.split(expand=True)[0].astype(float)
            df['Pre Calibration Value 2'] = df['Pre Calibration Value 2'].str.split(expand=True)[0].astype(float)
            df['Correction2'] = (df['Post Calibration Value 2'] - df[
                'Pre Calibration Value 2'])  # /df['Post Calibration Value 2']
        if 'Post Calibration Value 3' in df.columns and 'Pre Calibration Value 2' in df.columns:
            df['Post Calibration Value 3'] = df['Post Calibration Value 3'].str.split(expand=True)[0].astype(float)
            df['Pre Calibration Value 3'] = df['Pre Calibration Value 3'].str.split(expand=True)[0].astype(float)
            df['Correction3'] = (df['Post Calibration Value 3'] - df[
                'Pre Calibration Value 3'])  # /df['Post Calibration Value 3']
        # chapter_dataframes[chapter] = df.sort_values(by='Calibration End Time')
        final_dataframes[chapter] = df.sort_values(by='Calibration End Time')

    # Print preview of each dataframe
    for chapter, df in final_dataframes.items():
        # Skip sensors not of interest
        if (chapter.startswith("Cond") and "Sp Cond" not in chapter) or chapter == "Depth (m)":
            print(f"Skipping {chapter} (not of interest)")
            continue
        
        # print(f"\nChapter: {chapter}")
        # print(df.shape)
        # print(df.columns)
        # print(df.head(20))

        output_dir = '../data/output/calibration/summaries'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        sanitized_name = sanitize_filename(chapter)
        filename = os.path.join(output_dir, f"{sanitized_name}.csv")
        df.to_csv(filename, index=False)

        # Plotting removed - now handled by c2_uncertainty.py

if __name__ == '__main__':
    # Define the relative path to the directory containing CSV files
    relative_dir = '../data/input/calibrationlogs'
    summarize(relative_dir)

################################################################################################################
# Apply Corrections

def correct(sensor_df, summary_dir):
    """
    Calculate the correction for each column of sensor data based on the errors observed at time of calibration.
    :param sensor_df:
    :param summary_dir:
    :return:
    """
    summaries = [f for f in os.listdir(summary_dir) if f.endswith('.csv')]

    for file in summaries:
        param = file.replace(".csv", "")
        profiler_var = (f"Pfl - {param}")
        
        # Load the calibration summary CSV
        try:
            calibration_df = pd.read_csv(f"../data/output/calibration/summaries/{file}", sep=',', decimal='.')
            
            # Check if required datetime columns exist before parsing
            required_cols = ["Last Calibration Time", "Calibration Start Time", "Calibration End Time"]
            if not all(col in calibration_df.columns for col in required_cols):
                # Skip files that don't have the required calibration columns (e.g., statistics outputs)
                continue
            
            # Parse the datetime columns
            for col in required_cols:
                calibration_df[col] = pd.to_datetime(calibration_df[col], errors='coerce')
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue
        
        # Check if the profiler variable exists in sensor_df
        if profiler_var not in sensor_df.columns:
            # Try alternative spelling: replace µS_cm with microS_cm
            profiler_var_alt = profiler_var.replace("µS_cm", "microS_cm")
            if profiler_var_alt in sensor_df.columns:
                profiler_var = profiler_var_alt
            else:
                print(f"Skipping {file}: sensor column '{profiler_var}' not found in sensor data")
                continue
        
        print(f"Processing {file}")

        # Initialize lists to store results
        before_values = []
        after_values = []
        before_time_deltas = []
        after_time_deltas = []
        differences = []
        between_counts = []

        # Iterate over each calibration time
        previous_time = None
        for calibration_time in calibration_df["Calibration End Time"]:
            # Filter for timestamps within x hours before and after
            time_window_start = calibration_time - timedelta(hours=24)
            time_window_end = calibration_time + timedelta(hours=24)

            # Find the closest timestamp before/after calibration_time and within the cutoff
            before = sensor_df[(sensor_df["TIMESTAMP"] < calibration_time) &
                               (sensor_df["TIMESTAMP"] >= time_window_start)]
            after = sensor_df[(sensor_df["TIMESTAMP"] > calibration_time) &
                              (sensor_df["TIMESTAMP"] <= time_window_end)]

            # # Check if both before and after exist
            # if not before.empty and not after.empty:
            #     before_val = before.iloc[-1][profiler_var]
            #     after_val = after.iloc[0][profiler_var]
            #     diff = abs(after_val - before_val)
            # else:
            #     before_val = after_val = diff = None  # or np.nan

            # Count rows between previous and current calibration_time
            if previous_time is not None:
                between_rows = sensor_df[
                    (sensor_df["TIMESTAMP"] > previous_time) &
                    (sensor_df["TIMESTAMP"] <= calibration_time)
                    ]
                if len(between_rows) < 13:
                    before_val = after_val = diff = np.nan
                    before_time_delta = after_time_delta = pd.NaT
                else:
                    if not before.empty and not after.empty:
                        before_row = before.iloc[-1]
                        after_row = after.iloc[0]
                        before_val = before_row[profiler_var]
                        after_val = after_row[profiler_var]
                        diff = abs(after_val - before_val)
                        # Calculate time differences
                        before_time_delta = calibration_time - before_row["TIMESTAMP"]
                        after_time_delta = after_row["TIMESTAMP"] - calibration_time
                    else:
                        before_val = after_val = diff = np.nan
                        before_time_delta = after_time_delta = pd.NaT
            else:
                between_count = 0
                before_val = after_val = diff = np.nan
                before_time_delta = after_time_delta = pd.NaT

            previous_time = calibration_time
            before_values.append(before_val)
            after_values.append(after_val)
            before_time_deltas.append(before_time_delta)
            after_time_deltas.append(after_time_delta)
            differences.append(diff)
            between_counts.append(between_count)

        # Add new columns to the summary dataframe
        calibration_df["Before Value"] = before_values
        calibration_df["After Value"] = after_values
        calibration_df["Before Time Delta"] = before_time_deltas
        calibration_df["After Time Delta"] = after_time_deltas
        calibration_df["Difference"] = differences
        calibration_df["Between Count"] = between_counts
        keepers = ["Last Calibration Time","Calibration Start Time","Calibration End Time","Calibration Status",
                   "Standard","Pre Calibration Value","Post Calibration Value","Raw Value","Temperature","Timespan",
                   "Correction1","Before Value","After Value","Before Time Delta","After Time Delta","Difference", "Between Count"]

        calibration_df = calibration_df[keepers]
        calibration_df.to_csv(f"../data/output/calibration/corrections/{param}.csv", index=False)

        # Plotting removed - now handled by c2_uncertainty.py

    # Save the updated dataframe to a new CSV file
    # calibration_df.to_csv("../data/calibration_data_updated.csv", index=False)

if __name__ == '__main__':
    # Load the raw sensor data (not cleaned/adjusted)
    sensor_df = pd.read_csv("../data/input/sensors/FullHourly.csv")
    
    # Apply column renaming to match calibration parameter names
    column_names = {
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
    sensor_df = sensor_df.rename(columns=column_names)
    sensor_df["TIMESTAMP"] = pd.to_datetime(sensor_df["TIMESTAMP"])
    sensor_df = sensor_df.sort_values("TIMESTAMP")

    # Point to the directory with the calibration log summary tables
    summary_dir = "../data/output/calibration/summaries"
    correct(sensor_df, summary_dir)
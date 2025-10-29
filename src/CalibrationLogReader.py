'''
Processes log files from the calibration of the YSI EXO sensors.
Creates a summary table for each sensor, with calibration events and the correction factors
(multiple factors for multi-point calibrations).

Plots the correction factors against the time between calibrations and/or temperature at calibration.
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict


# Define the relative path to the directory containing CSV files
relative_dir = '../data/calibrationlogs'
csv_files = [f for f in os.listdir(relative_dir) if f.endswith('.csv')]

# Dictionary to hold dataframes grouped by Parameter Type
chapter_dataframes = defaultdict(list)


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)


# Function to parse a single file
def parse_calibration_file(filepath):
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    # Remove header lines like 'KorEXO Calibration File Export'
    lines = content.strip().splitlines()
    lines = [line for line in lines if not line.strip().startswith("KorEXO Calibration File Export")]
    content = "\n".join(lines)

    chapters = content.split("----------")
    for chapter in chapters:
        paragraphs = [p.strip() for p in chapter.strip().split('\n\n') if p.strip()]
        if not paragraphs:
            continue

        # Extract metadata from the first paragraph
        metadata_lines = paragraphs[0].split('\n')
        metadata = {}
        for line in metadata_lines:
            if '=,' in line:
                key, value = map(str.strip, line.split('=,', 1))
                metadata[key] = value

        # Extract data from subsequent paragraphs
        data_entry = metadata.copy()
        field_counts = defaultdict(int)
        for para in paragraphs[1:]:
            lines = para.strip().split('\n')
            for line in lines[1:]:  # Skip first line (e.g., [Cal Point 1])
                if '=,' in line:
                    key, value = map(str.strip, line.split('=,', 1))
                    field_counts[key] += 1
                    if field_counts[key] == 1 and key not in data_entry:
                        data_entry[key] = value
                    else:
                        data_entry[f"{key} {field_counts[key]}"] = value

        # Determine chapter name
        chapter_name = metadata.get("Parameter Type", "Unknown")
        chapter_dataframes[chapter_name].append(data_entry)

# Parse all CSV files in the directory
for file in csv_files:
    parse_calibration_file(os.path.join(relative_dir, file))

# Convert lists of dicts to DataFrames and process datetime columns
final_dataframes = {}
for chapter, entries in chapter_dataframes.items():
    df = pd.DataFrame(entries)

    for col in ['Calibration End Time', 'Last Calibration Time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if 'Calibration End Time' in df.columns and 'Last Calibration Time' in df.columns:
        df['Timespan'] = df['Calibration End Time'] - df['Last Calibration Time']
    if 'Post Calibration Value' in df.columns and 'Pre Calibration Value' in df.columns:
        df['Post Calibration Value'] = df['Post Calibration Value'].str.split(expand=True)[0].astype(float)
        df['Pre Calibration Value'] = df['Pre Calibration Value'].str.split(expand=True)[0].astype(float)
        df['Correction1'] = (df['Post Calibration Value'] - df['Pre Calibration Value'])#/df['Post Calibration Value']
    if 'Post Calibration Value 2' in df.columns and 'Pre Calibration Value 2' in df.columns:
        df['Post Calibration Value 2'] = df['Post Calibration Value 2'].str.split(expand=True)[0].astype(float)
        df['Pre Calibration Value 2'] = df['Pre Calibration Value 2'].str.split(expand=True)[0].astype(float)
        df['Correction2'] = (df['Post Calibration Value 2'] - df['Pre Calibration Value 2'])#/df['Post Calibration Value 2']
    if 'Post Calibration Value 3' in df.columns and 'Pre Calibration Value 2' in df.columns:
        df['Post Calibration Value 3'] = df['Post Calibration Value 3'].str.split(expand=True)[0].astype(float)
        df['Pre Calibration Value 3'] = df['Pre Calibration Value 3'].str.split(expand=True)[0].astype(float)
        df['Correction3'] = (df['Post Calibration Value 3'] - df['Pre Calibration Value 3'])#/df['Post Calibration Value 3']
    # chapter_dataframes[chapter] = df.sort_values(by='Calibration End Time')
    final_dataframes[chapter] = df.sort_values(by='Calibration End Time')

# Print preview of each dataframe
for chapter, df in final_dataframes.items():
    print(f"\nChapter: {chapter}")
    print(df.shape)
    print(df.columns)
    print(df.head(20))

    output_dir = '../data/calibrationlogs/summaries'
    sanitized_name = sanitize_filename(chapter)
    filename = os.path.join(output_dir, f"{sanitized_name}.csv")
    df.to_csv(filename, index=False)

    if 'Timespan' in df.columns:
        # Convert Timespan to total seconds for plotting
        df['Timespan_seconds'] = df['Timespan'].dt.total_seconds()

        # if 'Correction2' not in df.columns:
        #     y_columns = ['Correction1']
        # if 'Correction3' not in df.columns:
        #     y_columns = ['Correction1', 'Correction2']
        # else:
        #     y_columns = ['Correction1', 'Correction2', 'Correction3']

        y_columns = ['Correction1']
        if 'Correction2' in df.columns:
            y_columns = ['Correction1', 'Correction2']
        if 'Correction3' in df.columns:
            y_columns = ['Correction1', 'Correction2', 'Correction3']

        melted_df = df[['Timespan_seconds'] + y_columns].melt(id_vars='Timespan_seconds',
                                                                  value_vars=y_columns,
                                                                  var_name='Series',
                                                                  value_name='Value')
        # Convert to numeric
        melted_df['Value'] = pd.to_numeric(melted_df['Value'], errors='coerce')

        plt.figure(figsize=(10, 6))
        for col in y_columns:
            sns.scatterplot(x='Temperature', y=col, data=df, label=col)

        plt.title(f"{chapter} - Timespan vs Selected Values")
        plt.xlabel("Timespan (seconds)")
        plt.ylabel("Value")
        # plt.xscale('log')
        # plt.yscale('log')
        plt.grid
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Conclusions: Limited expalanatory power of timespan or temperature on correction values.
        # Could fit linear function in certain cases, but would need to ignore lots of examples.
        # Therefore, prefer to use simple linear correction based on error at time of calibration.




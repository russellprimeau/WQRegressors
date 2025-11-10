'''
Analyze the combined dataset to identify commonly-sized chunks of data which can be used for forecasting over
different horizons.
Then, split the dataset into files for each equivalent sample.
'''

import os
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path


def clean_directory(directory_path):
    """
    Deletes all files within the specified directory.
    Subdirectories and their contents are not affected.
    """
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f"Deleted file: {file_path}")
    except OSError as e:
        print(f"Error deleting files in {directory_path}: {e}")

def gapless(df, target_columns, name="length_v_count_analysis"):
    """
    Evaluate # of samples with time series leading directly into the Eurofin sample time (for "nowcast").
    Plot results for each parameter.
    :return:
    """

    # Initialize a dictionary to store results
    results = {col: [] for col in target_columns}

    # Loop through each variable and segment length
    for col in target_columns:
        for seg_len in range(10, 169, 25):
            count = 0
            for i in range(seg_len - 1, len(df)):
                segment = df.iloc[i - seg_len + 1:i + 1]
                if pd.notnull(segment.iloc[-1][col]) and segment["Segment"].nunique() == 1:
                    count += 1
            results[col].append((seg_len, count))

    # Plotting
    plt.figure(figsize=(12, 8))
    for col in target_columns:
        lengths, counts = zip(*results[col])
        plt.plot(lengths, counts, label=col)

    plt.xlabel("Segment Length")
    plt.ylabel("Number of Valid Segments")
    plt.title("Segment Length vs. Number of Valid Segments for Each Variable")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("../data/output/regression/availability", name + ".png"))
    # plt.show()

def gapped(df, target_columns, seg_length, name="length_v_count_analysis"):
    """
    Evaluate a range of gaps from time series to Eurofins values.
    :return:
    """
    os.makedirs(os.path.join("..data/output/regression/availability"), exist_ok=True)

    results_df = pd.DataFrame({"Gap Hours", "Valid Segments", "Variable Name"})
    for column in target_columns:
        # Initialize a dictionary to store results
        gap_results = {gap: 0 for gap in range(0, 167, 5)}
        # print(column)
        # Iterate over each row with a non-null value in the last column
        for idx, row in df[df[column].notna()].iterrows():
            current_time = row["TIMESTAMP"]
            current_segment = row["Segment"]
            for gap in range(0, 167, 5):
                target_time = current_time - pd.Timedelta(hours=gap)
                # Find the first row before current_time that matches target_time
                preceding_rows = df[(df["TIMESTAMP"] <= target_time)].sort_values(by="TIMESTAMP", ascending=False)
                if not preceding_rows.empty:
                    first_row = preceding_rows.iloc[0]
                    segment_value = first_row["Segment"]
                    segment_rows = df[(df["TIMESTAMP"] <= first_row["TIMESTAMP"]) & (df["Segment"] == segment_value)]
                    if len(segment_rows) >= seg_length:
                        gap_results[gap] += 1

        result = pd.DataFrame({
            "Gap Hours": list(gap_results.keys()),
            "Valid Segments": list(gap_results.values()),
            "Variable Name": column,
        })

        # Append results to DataFrame for plotting
        results_df = pd.concat([results_df, result], ignore_index=True)

    fig = px.line(results_df, x="Gap Hours", y="Valid Segments", color="Variable Name",
                  title=f"Effect of Gap on Valid Segments", markers=True)
    fig.write_image(os.path.join("../data/output/regression/availability", name + ".png"))

def normalize_columns(df, columns, min=0, max=1, save=False, directory="../data/output/regression"):
    """
    Normalize specified columns in a DataFrame to a given range and save original min/max values.

    Parameters:
    - df: pandas.DataFrame
    - columns: list of column names to normalize
    - min: minimum value of target range
    - max: maximum value of target range
    - save_path: path to save normalization parameters

    Returns:
    - A copy of the DataFrame with normalized columns.
    """
    df_normalized = df.copy()
    min_val, max_val = min, max
    normalization_params = {}

    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        normalization_params[col] = {"min": col_min, "max": col_max}
        if col_max != col_min:
            df_normalized[col] = ((df[col] - col_min) / (col_max - col_min)) * (max_val - min_val) + min_val
        else:
            df_normalized[col] = (min_val + max_val) / 2

    # Save normalization parameters to file if selected:
    if save:
        file = Path(directory, "normalized.json")
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as f:
            json.dump(normalization_params, f)

    return df_normalized

def split(df, output_dir, target_columns=['06-E.coli', '08-Kimtall 22°C', '21-Arsen', '24-Bly', '32-Kadmium',
                                          '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)',
                                          '09-Koliforme bakterier 37°C', '07-Intestinale enterokokker', '01-Farge',
                                          '04-Turbiditet', '44-pH, surhetsgrad'],
          length=1, to_normalize=[], offset=0):
    """
    Break up a dataset which contains gaps into many files of standard size, which do not contain gaps
    :param df: dataframe of consolidated dataset to be broken up
    :param output_dir: where to save the output files
    :param length: chunk size (threshold for consecutive rows to include in each sample)
    :return:
    """

    df = normalize_columns(df, to_normalize, 0, 1, save=True, directory=output_dir)

    os.makedirs(output_dir + '/samples', exist_ok=True)  # Create a directory to store the output files
    clean_directory(output_dir + '/samples')  #

    # Initialize a counter for naming output files
    segment_counter = 1

    # Iterate through the dataframe to find valid segments
    for i in range(len(df) - (length-1)):
        segment = df.iloc[i:i+length]
        last_row = segment.iloc[-1]
        preceding_rows = segment.iloc[:-1]

        # Check if the last row has any non-null value in the last ten columns
        if last_row[target_columns].notnull().all():

            # Check if the 'Segment' column has a constant value in all rows
            if preceding_rows['Segment'].nunique() == 1:
                # Save the segment to a CSV file
                output_file = os.path.join(output_dir, 'samples', f"segment_{segment_counter}.csv")
                segment.to_csv(output_file, index=False)
                segment_counter += 1


if __name__ == '__main__':
    matplotlib.use('Agg')  # Non-interactive backend for file output to handle remote machine installation errors
    ## Load sensor data
    ## For binary classification (of Eurofins parameters):
    df = pd.read_csv("../data/output/classification/Consolidated_binarized.csv",
                     parse_dates=["TIMESTAMP"])
    ## For regression (of any parameters:
    # df = pd.read_csv("../data/output/regression/Consolidated.csv",parse_dates=["TIMESTAMP"])
    df = df.sort_values("TIMESTAMP")

    ## Identify prediction target columns. Output will only include samples with valid value in last row.
    target_columns = ['06-E.coli']
    # target_columns = ['06-E.coli', '08-Kimtall 22°C', '21-Arsen', '24-Bly', '32-Kadmium',
    #     '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)', '09-Koliforme bakterier 37°C',
    #     '07-Intestinale enterokokker']  # alternative 1: name-based selection
    # target_columns = df.columns[-9:]  # alternative 2: index-based selection

    ## To analyze the impact of sample dimensions on the # of available samples:
    gapless(df, target_columns, name="Eurofins_availability")  # Analysis function #1
    # seg_length = 24  # fixed segment length for evaluating range of lengths of gap betweeen input and output
    # gapped(df, target_columns, seg_length)  # Analysis function #2

    ## Name the dataset and select the size of each sample (# of timesteps/rows)
    set_name  = "Ecoli24hr"  # Name of subdirectory where samples will be organized
    length = 24  # Hours of contiguous data per sample
    output_dir = os.path.join("../data/output/classification", set_name)

    ## Select columns where values in samples will be normalized, which helps with calculating loss accurately
    # to_normalize = df.columns[3:]
    to_normalize = ['Pfl - Temp (C)', 'Pfl - Sp Cond (microS_cm)',
        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)', 'Pfl - fDOM (QSU)',
        'Instantaneous atmospheric pressure (mBar)', 'Wind direction 10minRollingAvg (°)_x',
        'Wind direction 10minRollingAvg (°)_y',
        'Hourly average wind direction (°)_x', 'Hourly average wind direction (°)_y', 'Average wind speed (m/s)',
        'Maximum sustained wind speed, 3-second span (m/s)', 'Time of maximum 3s Gust',
        'Maximum sustained wind speed, 10-minute span (m/s)', 'Time of maximum 10 minute gust',
        'Hourly average atmospheric pressure at station (mBar)', 'Maximum pressure differential, 3-hour span (mBar)',
        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
        'Longwave (IR) radiation (W/m2)', 'Instantaneous sea-level atmospheric pressure (mBar)',
        'Shortwave (solar) radiation (W/m2)', 'Precipitation (mm/hr)', 'Instantaneous temperature (°C)',
        'Maximum temperature (°C)', 'Minimum temperature (°C)', 'Average humidity (% relative humidity)',
        'SCADA - pH', 'SCADA - Temperature (°C)', '06-E.coli', '08-Kimtall 22°C', '21-Arsen', '24-Bly', '32-Kadmium',
        '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)', '09-Koliforme bakterier 37°C',
        '07-Intestinale enterokokker', '01-Farge', '04-Turbiditet', '44-pH, surhetsgrad']
    split(df, output_dir, target_columns, length, to_normalize, 0)
#
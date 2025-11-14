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
from a1_CleanCombine import normalize_columns

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

def find_valid(df, targets, predictors, span, valid):
    valid_indices = []

    for i in range(len(df)):
        # Check targets in current row
        if df.loc[i, targets].isna().any():
            continue

        # Define window for previous rows
        start = max(0, i - span)
        window = df.iloc[start:i]

        if window.empty:
            continue

        # Count non-NaN predictor values in the window
        total_values = len(window) * len(predictors)
        non_nan_values = window[predictors].notna().sum().sum()

        # Check if proportion meets threshold
        if total_values > 0 and 1 - (non_nan_values / total_values) <= valid:
            valid_indices.append(i)

    return valid_indices

def analyze_valid(df, targets, predictors, span, valid, name="FaultTolerantSampleSize"):
    plt.figure(figsize=(12, 8))

    if len(targets) > 1:  # If multiple target columns specified, check each across a range of fault tolerances
        results = {col: [] for col in target_columns}
        for col in targets:
            for val in range(0,101,10):
                count = len(find_valid(df, [col], predictors, 96, val/100))
                results[col].append((val, count))
            fraction, counts = zip(*results[col])
            plt.plot(fraction, counts, '--', label=col)
        plt.xlabel(f"Fault tolerance (Maximum % missing values) in input set")
        plt.ylabel(f"Number of {span}-hour Samples")
        plt.title("Sample size vs. fault tolerance")
    elif len(span) > 1:  # If multiple window sizes specified, check each across a range of fault tolerances
        results = {duration: [] for duration in span}
        for duration in span:
            for val in range(0, 101, 10):
                count = len(find_valid(df, targets, predictors, duration, val/100))
                results[duration].append((val, count))
            val, counts = zip(*results[duration])
            plt.plot(val, counts, '--', label=f"{duration}-hour samples")
        plt.xlabel(f"Fault tolerance (Maximum % missing values) in input set")
        plt.ylabel(f"Number of Samples")
        plt.title("Sample size vs. fault tolerance")
    elif len(valid) > 1:  # If fault tolerances specified, check only at these values
        results = {tol: [] for tol in valid}
        for tol in valid:
            for duration in range(0, 169, 12):
                count = len(find_valid(df, targets, predictors, duration, tol/100))
                results[duration].append((duration, count))
            duration, counts = zip(*results[tol])
            plt.plot(duration, counts, '--', label=f"{tol}% fault tolerance")
        plt.xlabel(f"% faults (NaN) in input set")
        plt.ylabel(f"Number of Samples")
        plt.title("Sample size vs. fault tolerance")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("../data/output/regression/availability", name + ".png"))
    # plt.show()

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
        plt.plot(lengths, counts, '--', label=col)
        print(col, max(counts))

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

def split(df, output_dir, target_columns=['01-Farge', '04-Turbiditet', '06-E.coli',
                        '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen',
                        '24-Bly', '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)'],
          length=1, valid=1, to_normalize=[],fault_tolerant=False, offset=0):
    """
    Break up a dataset which contains gaps into many files of standard size, which do not contain gaps
    :param df: dataframe of consolidated dataset to be broken up
    :param output_dir: where to save the output files
    :param length: chunk size (threshold for consecutive rows to include in each sample)
    :return:
    """

    df = normalize_columns(df, to_normalize, param_file=None, min_val=0, max_val=1, save=True, directory=output_dir)

    os.makedirs(output_dir + '/samples', exist_ok=True)  # Create a directory to store the output files
    clean_directory(output_dir + '/samples')  #

    # Initialize a counter for naming output files
    segment_counter = 1

    ## Iterate through the dataframe to find valid segments
    if fault_tolerant:
        indices = find_valid(df, target_columns, predictor_cols, length, valid)
        for i, idx in enumerate(indices):
            # Compute start and end of the segment
            start = max(0, idx - length)
            end = idx + 1  # include the current row

            # Slice the DataFrame
            segment = df.iloc[start:end]

            # Build filename
            filename = os.path.join(output_dir, 'samples', f"Segment_{i}.csv")

            # Write to CSV
            segment.to_csv(filename, index=False)  # index=False since you want columns only
        pass
    else:
        for i in range(len(df) - (length-1)):
            segment = df.iloc[i:i+length]
            last_row = segment.iloc[-1]
            preceding_rows = segment.iloc[:-1]

            # Check if the last row has any non-null value in the target columns
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
    # df = pd.read_csv("../data/output/classification/Consolidated_binarized.csv",
    #                  parse_dates=["TIMESTAMP"])
    ## For regression (of any parameters:
    df = pd.read_csv("../data/output/classification/Consolidated_sparse_binarized.csv",parse_dates=["TIMESTAMP"])
    df = df.sort_values("TIMESTAMP")

    ## Identify prediction target columns. Output will only include samples with valid value in last row.
    predictor_cols  = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
                        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                        'Pfl - fDOM (QSU)', "Wind speed, x (m/s)", "Wind speed, y (m/s)",
                        'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                        'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                        '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)']
    target_columns = ['01-Farge', '04-Turbiditet', '06-E.coli',
                        '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen',
                        '24-Bly', '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']  # alternative 1: name-based selection
    # target_columns = df.columns[-9:]  # alternative: index-based selection

    ## Alternative with better coverage
    predictor_cols_max = ["Wind speed, x (m/s)", "Wind speed, y (m/s)",
                      'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                      'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                      '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
                      'SCADA - Temperature (°C)']
    target_columns_max = ['06-E.coli', '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C']



    ## To analyze the impact of sample dimensions on the # of available samples:
    # gapless(df, target_columns, name="Sparse_Eurofins_availability")  # Analysis function #1
    # seg_length = 24  # fixed segment length for evaluating range of lengths of gap betweeen input and output
    # gapped(df, target_columns, seg_length)  # Analysis function #2
    # analyze_valid(df, target_columns, predictor_cols, 96, 1, name="Max_Input_96hr_Set")
    # analyze_valid(df, target_columns_max, predictor_cols_max, 96, 1, name="Max_Coverage_96hr_Set")
    # analyze_valid(df, ['09-Koliforme bakterier 37°C'], predictor_cols, [12, 24, 48, 96, 168], 1, name="Koli_Max_Input_Set_v_Length")

    ## Name the dataset and select the size of each sample (# of timesteps/rows)
    set_name  = "Koliforms96Sparse"  # Name of subdirectory where samples will be organized
    length = 96  # Hours of contiguous data per sample
    output_dir = os.path.join("../data/output/classification", set_name)

    ## Select columns where values in samples will be normalized, which helps with calculating loss accurately
    # to_normalize = df.columns[3:]
    to_normalize = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
                        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                        'Pfl - fDOM (QSU)', "Wind speed, x (m/s)", "Wind speed, y (m/s)",
                        'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                        'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                        '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
        'SCADA - pH', 'SCADA - Temperature (°C)', '01-Farge', '04-Turbiditet', '06-E.coli',
        '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen',
        '24-Bly', '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']

    # to_normalize = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
    #                     'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
    #                     'Pfl - fDOM (QSU)', "Wind speed, x (m/s)", "Wind speed, y (m/s)",
    #                     'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
    #                     'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
    #                     '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
    #     'SCADA - pH', 'SCADA - Temperature (°C)']

    # split(df, output_dir, target_columns, length, to_normalize, 0)

    split(df, output_dir, ['09-Koliforme bakterier 37°C'], length, 90, to_normalize, True)
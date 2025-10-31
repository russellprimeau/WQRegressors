'''
Analyze the combined dataset to identify commonly-sized chunks of data which can be used for forecasting over
different horizons.
Then, split the dataset into files for each equivalent sample.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def gapless(df, target_columns):
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
    plt.savefig("../data/output/for_regression/analysis/segment_analysis.png")
    plt.show()

def gapped(df, target_columns):
    """
    Evaluate a range of gaps from time series to Eurofins values.
    :return:
    """
    # Initialize a dictionary to store results
    gap_results = {gap: 0 for gap in range(0, 167, 5)}

    results_df = pd.DataFrame({"Gap Hours", "Valid Segments", "Variable Name"})

    for column in target_columns:
        print(column)
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
                    if len(segment_rows) >= 10:
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
    fig.write_image("../data/output/analysis/gap_vs_segments.png")

def split(df, output_dir, length):

    # Create a directory to store the output files
    os.makedirs(output_dir, exist_ok=True)

    # Initialize a counter for naming output files
    segment_counter = 1

    # Iterate through the dataframe to find valid segments
    for i in range(len(df) - 23):
        segment = df.iloc[i:i+24]
        last_row = segment.iloc[-1]
        preceding_rows = segment.iloc[:-1]

        # Check if the last row has any non-null value in the last ten columns
        if last_row[target_columns].notnull().any():
            # Check if the 'Segment' column has a constant value in the preceding 167 rows
            if preceding_rows['Segment'].nunique() == 1:
                # Save the segment to a CSV file
                output_file = os.path.join(output_dir, f"segment_{segment_counter}.csv")
                segment.to_csv(output_file, index=False)
                segment_counter += 1


if __name__ == '__main__':
    # Load the sensor data
    df = pd.read_csv("../data/output/for_regression/Combined_Cleaned.csv", parse_dates=["TIMESTAMP"])
    df = df.sort_values("TIMESTAMP")

    # Identify prediction targets (Eurofins data)
    target_columns = df.columns[-9:]

    gapless(df, target_columns)
    # gapped(df, target_columns)

    output_dir = "../data/output/for_regression/partial_Eurofins"
    length = 24
    # split(df, output_dir, length)

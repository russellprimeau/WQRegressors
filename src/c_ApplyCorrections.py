'''
Apply a linear correction to each of the sensor columns based on the observed error at the time of calibration.
'''

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

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
        # Load the second CSV file with calibration times
        print("param", param)
        calibration_df = pd.read_csv(f"../data/output/calibration/summaries/{file}", sep=',', decimal='.', parse_dates=["Last Calibration Time", "Calibration Start Time", "Calibration End Time"])

        # Initialize lists to store results
        before_values = []
        after_values = []
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
                else:
                    if not before.empty and not after.empty:
                        before_val = before.iloc[-1][profiler_var]
                        after_val = after.iloc[0][profiler_var]
                        diff = abs(after_val - before_val)
                    else:
                        before_val = after_val = diff = np.nan
            else:
                between_count = 0
                before_val = after_val = diff = np.nan

            previous_time = calibration_time
            before_values.append(before_val)
            after_values.append(after_val)
            differences.append(diff)
            between_counts.append(between_count)

        # Add new columns to the summary dataframe
        calibration_df["Before Value"] = before_values
        calibration_df["After Value"] = after_values
        calibration_df["Difference"] = differences
        calibration_df["Between Count"] = between_counts
        keepers = ["Last Calibration Time","Calibration Start Time","Calibration End Time","Calibration Status",
                   "Standard","Pre Calibration Value","Post Calibration Value","Raw Value","Temperature","Timespan",
                   "Correction1","Before Value","After Value","Difference", "Between Count"]

        calibration_df = calibration_df[keepers]
        calibration_df.to_csv(f"../data/output/calibration/corrections/{param}.csv", index=False)

        calibration_df["Span"] = calibration_df["Calibration Start Time"] - calibration_df["Last Calibration Time"]

        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(x="Span", y="Difference", data=calibration_df, label=profiler_var)
        ax.grid()

        # for i, row in calibration_df.iterrows():
        #     ax.annotate(f"{row['Calibration End Time'].strftime('%Y-%m-%d %H:%M')}\nCount: {row['Points']}",
        #                 row["Span"], row["Difference"])


        plt.title("")
        plt.xlabel("Timespan")
        plt.ylabel(f"Absolute Error, {profiler_var}")
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
        plt.show()


        # calibration_df = calibration_df["Calibration End Time", "Last Calibration Time", "Before Value", "After Value", "Difference"]
        # calibration_df = calibration_df["Calibration End Time"]

    # Save the updated dataframe to a new CSV file
    # calibration_df.to_csv("../data/calibration_data_updated.csv", index=False)

if __name__ == '__main__':
    # Load the sensor data
    sensor_df = pd.read_csv("../data/output/regression/Consolidated.csv", parse_dates=["TIMESTAMP"])
    sensor_df = sensor_df.sort_values("TIMESTAMP")

    # Point to the directory with the calibration log summary tables
    summary_dir = "../data/output/calibration/summaries"
    correct(sensor_df, summary_dir)
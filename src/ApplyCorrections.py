'''
Apply a linear correction to each of the sensor columns based on the observed error at the time of calibration.
'''

import os
import pandas as pd
from datetime import timedelta

# Load the first CSV file with sensor data
sensor_df = pd.read_csv("../data/Combined_Cleaned.csv", parse_dates=["TIMESTAMP"])
sensor_df = sensor_df.sort_values("TIMESTAMP")

summary_dir = "../data/calibrationlogs/summaries"
summaries = [f for f in os.listdir(summary_dir) if f.endswith('.csv')]
summaries = ["Turbidity (FNU).csv"]

for file in summaries:
    param = file.replace(".csv", "")
    print(f"Pfl - {param}")
    # Load the second CSV file with calibration times
    calibration_df = pd.read_csv(f"../data/calibrationlogs/summaries/{file}", sep=',', decimal='.', parse_dates=["Calibration End Time"])

    # Initialize lists to store results
    before_values = []
    after_values = []
    differences = []

    # Iterate over each calibration time
    for calibration_time in calibration_df["Calibration End Time"]:
        print("calibration_time", calibration_time)
        # Filter for timestamps within 12 hours before and after
        time_window_start = calibration_time - timedelta(hours=24)
        time_window_end = calibration_time + timedelta(hours=24)

        # Find the closest timestamp before/after calibration_time and within the cutoff
        before = sensor_df[(sensor_df["TIMESTAMP"] < calibration_time) &
                           (sensor_df["TIMESTAMP"] >= time_window_start)]
        print("before", before)
        after = sensor_df[(sensor_df["TIMESTAMP"] > calibration_time) &
                          (sensor_df["TIMESTAMP"] <= time_window_end)]


        # Check if both before and after exist
        if not before.empty and not after.empty:
            before_val = before.iloc[-1][f"Pfl - {param}"]
            after_val = after.iloc[0][f"Pfl - {param}"]
            diff = after_val - before_val
            print("before", before.iloc[-1]["TIMESTAMP"])
            print("after", after.iloc[0]["TIMESTAMP"])
        else:
            before_val = after_val = diff = None  # or np.nan

        before_values.append(before_val)
        after_values.append(after_val)
        differences.append(diff)

    # Add new columns to the summary dataframe
    calibration_df["Before Value"] = before_values
    calibration_df["After Value"] = after_values
    calibration_df["Difference"] = differences
    calibration_df.to_csv(f"../data/calibrationlogs/corrections/{param}.csv", index=False)

    # calibration_df = calibration_df["Calibration End Time", "Last Calibration Time", "Before Value", "After Value", "Difference"]
    # calibration_df = calibration_df["Calibration End Time"]
    # , "Pre Calibration Value","Post Calibration Value","Raw Value","Temperature","Pre Calibration Value 2","Post Calibration Value 2","Raw Value 2","Temperature 2","Stability Achieved 2","Pre Calibration Value 3","Post Calibration Value 3""Timespan","Correction1","Correction2", "Correction3"]

# Save the updated dataframe to a new CSV file
calibration_df.to_csv("../data/calibration_data_updated.csv", index=False)
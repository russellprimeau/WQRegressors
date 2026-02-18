"""
Combines datasets from the profiler, weather station, SCADA system and Eurofins reports in a single table.

Small gaps (below a specified threshold, default of 6 hours) are filled by linear interpolation.
When there is a gap in any one column, either all rows can be dropped, or "NaN" can be retained for that column.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from utils.preprocessing import (clean_profiler, add_source, decompose_direction, rolling_sum,
                                 explore_data, forward_fill_columns, add_res, count_segs)

## Configuration Parameters
# Whether to keep all timesteps (True) or drop rows with missing values (False)
sparse = True  # Options: True, False

# Output mode: 'regression' for continuous values, 'classification' for binarized values
mode = 'regression'  # Options: 'regression', 'classification'

# Whether to forward fill Eurofins columns (only applies when sparse=True)
forward_fill = True  # Options: True, False

# Whether to add residual columns (only applies when sparse=True and forward_fill=True)
add_res_cols = True  # Options: True, False

# Whether to add a column with index for continuous segments
count_segments = False  # Options: True, False


if __name__ == '__main__':
    ## Load and pre-process data from source files
    ## Profiler data
    df = pd.read_csv("../data/input/sensors/FullHourly.csv")
    clean_df = clean_profiler(df, max_gap=6)  # Clean profiler dataset, since certain errors are easy to detect

    ## Weather data
    weather_df = pd.read_csv("../data/input/sensors/Weather.csv", sep=";", decimal=",", parse_dates=["Time"])
    weather_df["Time"] = weather_df["Time"].dt.round("h")
    full_weather_columns = {"1818_time: AA[mBar]": "Instantaneous atmospheric pressure (mBar)",
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

    weather_df.rename(columns=full_weather_columns, inplace=True)

    # Set negative shortwave values to 0 (this is very common and appears to represent a calibration issue)
    weather_df['Shortwave (solar) radiation (W/m2)'] = (
        np.where(weather_df['Shortwave (solar) radiation (W/m2)'] < 0, 0, weather_df['Shortwave (solar) radiation (W/m2)']))

    # Define conditions for each parameter which indicate errors in the data
    weather_error_conditions = {
        "Time": (weather_df['Time'] < pd.to_datetime('2000-01-01')) | (weather_df['Time'] > pd.to_datetime('2099-12-31')),
        'Hourly average wind direction (°)': (weather_df['Hourly average wind direction (°)'] < 0) | (weather_df['Hourly average wind direction (°)'] > 360),
        "Average wind speed (m/s)": (weather_df["Average wind speed (m/s)"] < 0) | (weather_df["Average wind speed (m/s)"] > 100),
        'Maximum sustained wind speed, 3-second span (m/s)': (weather_df['Maximum sustained wind speed, 3-second span (m/s)'] < 0) | (weather_df['Maximum sustained wind speed, 3-second span (m/s)'] > 100),
        'Maximum sustained wind speed, 10-minute span (m/s)': (weather_df['Maximum sustained wind speed, 10-minute span (m/s)'] < 0) |(weather_df['Maximum sustained wind speed, 10-minute span (m/s)'] > 100),
        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)': (weather_df['Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)'] < 860) | (weather_df['Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)'] > 1080),
        'Maximum pressure differential, 3-hour span (mBar)': (weather_df['Maximum pressure differential, 3-hour span (mBar)'] < 0) | (weather_df['Maximum pressure differential, 3-hour span (mBar)'] > 50),
        'Longwave (IR) radiation (W/m2)': (weather_df['Longwave (IR) radiation (W/m2)'] < 0) | (weather_df['Longwave (IR) radiation (W/m2)'] > 750),
        'Shortwave (solar) radiation (W/m2)': (weather_df['Shortwave (solar) radiation (W/m2)'] < 0) | (weather_df['Shortwave (solar) radiation (W/m2)'] > 900),
        'Precipitation (mm/hr)': (weather_df['Precipitation (mm/hr)'] < 0) | ( weather_df['Precipitation (mm/hr)'] > 50),
        'Maximum temperature (°C)': (weather_df['Maximum temperature (°C)'] < -40) | ( weather_df['Maximum temperature (°C)'] > 40),
        'Minimum temperature (°C)': (weather_df['Minimum temperature (°C)'] < -40) | ( weather_df['Minimum temperature (°C)'] > 40),
        'Average humidity (% relative humidity)': (weather_df['Average humidity (% relative humidity)'] < 0) | ( weather_df['Average humidity (% relative humidity)'] > 100)
    }

    # Replace values meeting the error conditions with np.nan using boolean indexing
    for col, condition in weather_error_conditions.items():
        weather_df.loc[condition, col] = np.nan
    decomp_df = decompose_direction(weather_df, "Hourly average wind direction (°)",
                                    "Average wind speed (m/s)")
    weather_roll_df = rolling_sum(decomp_df, "Time", 'Precipitation (mm/hr)', 24)
    simplified_weather_set = ['Time', 'Hourly average wind direction (°)_x',
        'Hourly average wind direction (°)_y',
        "Maximum sustained wind speed, 3-second span (m/s)",
        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
        'Longwave (IR) radiation (W/m2)',
        'Shortwave (solar) radiation (W/m2)',
        'rolling Precipitation (mm/hr)',
        'Instantaneous temperature (°C)',
        'Average humidity (% relative humidity)']
    weather_simp_df = weather_roll_df[simplified_weather_set].copy()
    simplified_weather_names = {"Hourly average wind direction (°)_x":"Wind speed x (m/s)",
                                "Hourly average wind direction (°)_y": "Wind speed y (m/s)",
                                "Maximum sustained wind speed, 3-second span (m/s)": 'Maximum 3s wind gust (m/s)',
                                "Instantaneous atmospheric pressure compensated for temperature, humidity and station "
                                "elevation (mBar)": "Atmospheric pressure (mBar)",
                                'Instantaneous temperature (°C)': 'Air temperature (°C)',
                                'Average humidity (% relative humidity)': 'Humidity (%)',
                                'rolling Precipitation (mm/hr)': '24hr precipitation total (mm)'}
    weather_simp_df.rename(columns=simplified_weather_names, inplace=True)

    ## SCADA data
    scada_df = pd.read_csv("../data/input/sensors/SCADA.csv", sep=";", decimal=".", parse_dates=["Time"])
    scada_df["Time"] = scada_df["Time"].dt.round("h")

    ## Eurofins data
    eurofins_cols = ['01-Farge', '04-Turbiditet', '44-pH, surhetsgrad', '06-E.coli',
               '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen', '24-Bly',
               '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']
    simplified_Eurofins = {'01-Farge': 'Color',
                            '04-Turbiditet': 'Turbidity (FNU)',
                            '44-pH, surhetsgrad': 'pH',
                            '06-E.coli': 'E.coli (CFU/100mL)', 
                            '07-Intestinale enterokokker': 'Intestinal enterococci (CFU/100mL)',
                            '08-Kimtall 22°C': 'Colony Count 22°C (CFU/mL)',
                            '09-Koliforme bakterier 37°C': 'Total coliforms 37°C (CFU/100mL)',
                            '21-Arsen': 'Arsenic (µg/L)',
                            '24-Bly': 'Lead (µg/L)',
                            '32-Kadmium': 'Cadmium (µg/L)',
                            '36-Kopper filtrert': 'Copper filtered (mg/L)',
                            '37-Krom': 'Chromium (µg/L)',
                            '41-Nikkel': 'Nickel (µg/L)',
                            'Sink (Zn)': 'Zinc (µg/L)'}

    eurofins_df = pd.read_csv("../data/input/sensors/Eurofins.csv", sep=";", decimal=",", parse_dates=["Time"])
    eurofins_df["Time"] = eurofins_df["Time"].dt.round("h")
    eurofins_df.rename(columns=simplified_Eurofins, inplace=True)

    # Determine binarize parameter based on mode
    binarize = (mode == 'classification')

    if sparse:
        # Sparse mode: Create continuous time index and align all dataframes
        dfs = [clean_df, weather_simp_df, scada_df, eurofins_df]

        ## Create continuous time index
        min_time, max_time = None, None
        for i, df in enumerate(dfs):
            if 'TIMESTAMP' in df.columns:
                df.set_index('TIMESTAMP', inplace=True)
            elif 'Time' in df.columns:
                df.set_index('Time', inplace=True)
            else:
                print(f"No TIMESTAMP or Time column in df {i}")
            if min_time is None or df.index.min() < min_time:
                min_time = df.index.min()
            if max_time is None or df.index.max() > max_time:
                max_time = df.index.max()
            dup_counts = df.index.value_counts()
            dup_counts = dup_counts[dup_counts > 1]
            if not dup_counts.empty:
                print(f"File {i} duplicate timestamps:")
                print(dup_counts)

        full_index = pd.date_range(start=min_time, end=max_time, freq='h')
        aligned_dfs = [df.reindex(full_index) for df in dfs]

        # Merge data from other sources into the dataset
        merge1_df = add_source(aligned_dfs[0], aligned_dfs[1], include_NAs=True, max_gap=6, sparse=True)
        merge2_df = add_source(merge1_df, aligned_dfs[2], include_NAs=True, max_gap=6, sparse=True)
        merge3_df = add_source(merge2_df, aligned_dfs[3], include_NAs=True, max_gap=6, sparse=True, binarize=binarize)
        
        # Apply forward fill and add_res if requested
        if forward_fill:
            merge3_plain_df = merge3_df.reset_index(names="TIMESTAMP")
            eurofins_cols_renamed = list(simplified_Eurofins.values())
            merge3_prior_df = forward_fill_columns(merge3_plain_df, eurofins_cols_renamed)
            if add_res_cols:
                final_df = add_res(merge3_prior_df, eurofins_cols_renamed)
            else:
                final_df = merge3_prior_df
        else:
            final_df = merge3_df.reset_index(names="TIMESTAMP")
    else:
        # Dense mode: Drop rows with missing values
        merge1_df = add_source(clean_df, weather_simp_df, include_NAs=False, max_gap=6)
        merge2_df = add_source(merge1_df, scada_df, include_NAs=True, max_gap=6)
        final_df = add_source(merge2_df, eurofins_df, include_NAs=True, max_gap=6, binarize=binarize)

    # Add segment counting if requested
    if count_segments:
        final_df = count_segs(final_df)

    ## Determine output directory and filename
    if mode == 'regression':
        output_dir = "../data/output/regression"
        if sparse:
            filename = "Consolidated_sparse.csv"
        else:
            filename = "Consolidated.csv"
    else:  # classification
        output_dir = "../data/output/classification"
        if sparse:
            filename = "Consolidated_binarized_sparse.csv"
        else:
            filename = "Consolidated_binarized.csv"

    ## Write the combined dataset to file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    final_df.to_csv(Path(output_dir, filename), index=False)

    print(f"Output saved to: {Path(output_dir, filename)}")
    print(f"Configuration: sparse={sparse}, mode={mode}, forward_fill={forward_fill}, add_res={add_res_cols}, count_segments={count_segments}")

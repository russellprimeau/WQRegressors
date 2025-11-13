"""
Combines datasets from the profiler, weather station, SCADA system and Eurofins reports in a single table.

Small gaps (below a specified threshold, default of 6 hours) are filled by linear interpolation.
When there is a gap in any one column, either all rows can be dropped, or "NaN" can be retained for that column.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go


def clean_profiler(full_df, max_gap=6):
    """
    Clean profiler hourly surface data. Fill small gaps by interpolation.
    :return:
    """
    full_df["Interpolated"] = 0
    # Apply column labels
    column_names = {
        "TIMESTAMP": "TIMESTAMP",
        "RECORD": "Pfl - Record number",
        "PFL_Counter": "Pfl - Day",
        "CntRS232": "Pfl - CntRS232",
        "RS232Dpt": "Pfl - Vertical position1 (m)",
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
        "lat": "Latitude",
        "lon": "Longitude",
    }
    df = full_df.rename(columns=column_names)

    error_conditions = {
        "Pfl - Water temperature (°C)": (df['Pfl - Water temperature (°C)'] < 1) | (df['Pfl - Water temperature (°C)'] > 25),
        "Pfl - Cond (microS_cm)": (df['Pfl - Cond (microS_cm)'] < 0) |(df['Pfl - Cond (microS_cm)'] > 45),
        "Pfl - Sp Cond (microS_cm)": (df['Pfl - Sp Cond (microS_cm)'] < 1),
        "Pfl - Salinity (ppt)": (df['Pfl - Salinity (ppt)'] < 0) | (df['Pfl - Salinity (ppt)'] > .03),
        "Pfl - pH": (df['Pfl - pH'] < 2) | (df['Pfl - pH'] > 12),
        "Pfl - DO (% Sat)": (df['Pfl - DO (% Sat)'] < 10) | (df['Pfl - DO (% Sat)'] > 120),
        "Pfl - Turbidity (NTU)": (df['Pfl - Turbidity (NTU)'] < 0),
        "Pfl - Turbidity (FNU)": (df['Pfl - Turbidity (FNU)'] < 0),
        "Pfl - fDOM (RFU)": (df['Pfl - fDOM (RFU)'] < 0) | (df['Pfl - fDOM (RFU)'] > 100),
        "Pfl - fDOM (QSU)": (df['Pfl - fDOM (QSU)'] < 0) | (df['Pfl - fDOM (QSU)'] > 300),
    }

    # Replace values meeting the error conditions with np.nan using boolean indexing
    for col, condition in error_conditions.items():
        full_df.loc[condition, col] = np.nan

    sensor_columns = ["Pfl - Water temperature (°C)", "Pfl - Sp Cond (microS_cm)", "Pfl - pH",
                      "Pfl - DO (% Sat)",
                      "Pfl - Turbidity (FNU)", "Pfl - fDOM (RFU)", "Pfl - fDOM (QSU)"]
    keepers = ["TIMESTAMP", "Interpolated"] + sensor_columns
    df = df[keepers].copy()

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

    for i in range(1, len(df)):
        current_time = df.loc[i, "TIMESTAMP"]
        previous_time = df.loc[i - 1, "TIMESTAMP"]
        time_diff = (current_time - previous_time).total_seconds() / 3600

        if 1 < time_diff <= max_gap:
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
    return df

def add_source(df, secondary_df, include_NAs=False, max_gap=6, binarize=False):
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

    if binarize:
        thresholds_df = pd.read_csv(Path('../data/input', "Limits.csv"), sep=';', decimal='.')
        merged_df = binarize_dataframe(merged_df, output_columns=new_columns, thresholds_df=thresholds_df)

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
        # Interpolate remaining missing values in new_columns
        merged_df[new_columns] = merged_df[new_columns].interpolate(method="linear", limit=max_gap, limit_direction="both")

    return merged_df

def binarize_dataframe(df, output_columns, thresholds_df):
    """
    Convert values in specified columns of a DataFrame to binary (0 or 1)
    based on thresholds provided in a single-row thresholds_df.
    NaN values are preserved.
    """
    binary_df = df.copy()
    for col in output_columns:
        if col not in thresholds_df.columns:
            raise ValueError(f"Threshold for column '{col}' not found in thresholds_df.")
        threshold = thresholds_df.iloc[0][col]
        # Apply threshold only to non-NaN values
        binary_df[col] = binary_df[col].where(binary_df[col].isna(), (binary_df[col] > threshold).astype(int))
    binary_df["anomaly"] = np.where(
        binary_df[output_columns].notna().any(axis=1),  # At least one non-NaN
        np.where(binary_df[output_columns].eq(1).any(axis=1), 1, 0),  # If any 1 → 1 else 0
        np.nan  # If all NaN → NaN
    )
    return binary_df

def decompose_direction(df, directional, magnitude=None):
    """
    Replace column (directional) storing values of direction in degrees with two new columns representing
    the x and y components, optionally scaled by values from a magnitude column.
    The new columns are inserted at the same position as the original column.

    Parameters:
    - df: pandas.DataFrame
    - directional: str, name of the column containing degree values

    Returns:
    - Modified DataFrame with x and y components replacing the original column
    """
    df_copy = df.copy()
    radians = np.deg2rad(df_copy[directional])
    x_component = np.cos(radians)
    y_component = np.sin(radians)
    if magnitude is not None:
        x_component = x_component * df_copy[magnitude].values
        y_component = y_component * df_copy[magnitude].values

    insert_position = df_copy.columns.get_loc(directional)
    df_copy.drop(columns=[directional], inplace=True)
    df_copy.insert(insert_position, f"{directional}_x", x_component)
    df_copy.insert(insert_position + 1, f"{directional}_y", y_component)

    return df_copy

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

def normalize_columns(df, columns, param_file=None, min_val=0, max_val=1, save=False, directory="../data/output",
                    filename="normalization.json"):
    '''
    Normalize values in columns (columns) from dataframe (df) either on to interval (min_val, max_val),
    or optionally, by applying a previously defined normalization written to (param_file). Optionally, write the
    applied normalization to (directory/filename) for later reuse.
    :param df:
    :param columns:
    :param param_file:
    :param min_val:
    :param max_val:
    :param save:
    :param directory:
    :param filename:
    :return:
    '''
    df_normalized = df.copy()
    if param_file is not None:
        try:
            with open(param_file, 'r') as f:
                normalization_params = json.load(f)
        except Exception as e:
            # print(e)
            normalization_params = {}
    else:
        normalization_params = {}

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if (param_file is not None) and (col in normalization_params):
                col_min = normalization_params[col]["min"]
                col_max = normalization_params[col]["max"]
                if col_max != col_min:
                    df_normalized[col] = ((df[col] - col_min) / (col_max - col_min)) * (max_val - min_val) + min_val
                else:
                    df_normalized[col] = (min_val + max_val) / 2
            else:
                # print(f"Column {col} not found in {param_file}.")
                col_min = df[col].min()
                col_max = df[col].max()
                normalization_params[col] = {"min": col_min, "max": col_max}
                if col_max != col_min:
                    df_normalized[col] = ((df[col] - col_min) / (col_max - col_min)) * (max_val - min_val) + min_val
                else:
                    df_normalized[col] = (min_val + max_val) / 2
        else:
            # print(f"Column {col} not found in dataframe.")
            pass

    # Save normalization parameters to file if selected:
    if save:
        file = Path(directory, filename)
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(file, 'w') as f:
                pass
        except FileNotFoundError:
            print(f"Error: File not found at {file}")
            pass
        with open(file, "w") as f:
            json.dump(normalization_params, f)
    return df_normalized

def rolling_sum(df, time_col, target_col, interval_hours):
    """
    Calculate rolling sum over a time-based interval.

    Args:
        df (pd.DataFrame): Input dataframe sorted by time.
        time_col (str): Name of the timestamp column.
        target_col (str): Column to sum.
        interval_hours (int): Interval in hours.

    Returns:
        pd.DataFrame: Original dataframe with an added 'rolling_sum' column.
    """
    # Ensure time column is datetime
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Initialize result column
    rolling_sums = []

    for i in range(len(df)):
        current_time = df.loc[i, time_col]
        window_start = current_time - pd.Timedelta(hours=interval_hours)

        # Filter rows within the interval
        mask = (df[time_col] >= window_start) & (df[time_col] <= current_time)
        rolling_sums.append(df.loc[mask, target_col].sum())

    df["rolling " + target_col] = rolling_sums
    return df

def explore_data(full, *raw):
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H3("Configure plat area"),
        html.H4("Upstream datasets:"),
        dcc.Checklist(
            id='in_sets',
            options=[
                {'label': 'Profiler', 'value': 'profiler'},
                {'label': 'Weather', 'value': 'weather'},
            ],
            value=['profiler', 'weather'],
            inline=True
        ),
        html.H4("Downstream (treatment plant):"),
        dcc.Checklist(
            id='out_sets',
            options=[
                {'label': 'SCADA', 'value': 'scada'},
                {'label': 'Samples (Physical)', 'value': 'eurofins_phys'},
                {'label': 'Samples (Biological)', 'value': 'eurofins_bio'},
                {'label': 'Samples (Metals)', 'value': 'eurofins_metal'},
            ],
            value=['scada', 'eurofins_phys', 'eurofins_bio', 'eurofins_metal'],
            inline=True
        ),
        dcc.RadioItems(
            id='source-select',
            options=[
                {'label': 'Filtered', 'value': 'cleaned'},
                {'label': 'Full Extent', 'value': 'sources'},
            ],
            value='cleaned',
            inline=True
        ),
        dcc.RadioItems(
            id='normalize',
            options=[
                {'label': 'Normalized on (0,1)', 'value': 'normalized'},
                {'label': 'Original Units', 'value': 'original'},
            ],
            value='normalized',
            inline=True
        ),
        dcc.RadioItems(
            id='plot-mode',
            options=[
                {'label': 'Continuous Time', 'value': 'continuous'},
                {'label': 'Seasonality', 'value': 'seasonality'}
            ],
            value='continuous',
            inline=True
        ),
        dcc.RadioItems(
            id='thresholds',
            options=[
                {'label': 'Hide limit values', 'value': 'off'},
                {'label': 'Show limit values', 'value': 'on'}
            ],
            value='on',
            inline=True
        ),
        dcc.DatePickerRange(
            id='date-range',
            start_date=None,
            end_date=None,
            min_date_allowed=None,
            max_date_allowed=None,
            display_format='YYYY-MM-DD',
        ),
        dcc.Graph(id='timeseries-plot', style={'height': '90vh', 'width': '100%'})
    ],
        style={'height': '100vh', 'width': '100%', 'padding': '10px', 'box-sizing': 'border-box'})

    @app.callback(
        Output('timeseries-plot', 'figure'),
        [Input('in_sets', 'value'),
         Input('out_sets', 'value'),
         Input('source-select', 'value'),
         Input('normalize', 'value'),
         Input('plot-mode', 'value'),
         Input('thresholds', 'value'),
         Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_plot(input_key, output_key, source_key, normalize_key, mode_key, thresholds_key, start_date_key,
                    end_date_key):
        profiler_cols = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
                        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                        'Pfl - fDOM (QSU)']
        weather_cols = ["Wind speed, x (m/s)", "Wind speed, y (m/s)",
                        'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                        'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                        '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)']
        SCADA_cols = ['SCADA - pH', 'SCADA - Temperature (°C)']
        Eurofins_phys = ['01-Farge', '04-Turbiditet']
        Eurofins_bio = ['06-E.coli', '07-Intestinale enterokokker', '08-Kimtall 22°C',
                         '09-Koliforme bakterier 37°C']
        Eurofins_metal = ['21-Arsen', '24-Bly', '32-Kadmium', '36-Kopper filtrert',
                         '37-Krom', '41-Nikkel', 'Sink (Zn)']
        included_cols = []
        if "profiler" in input_key:
            included_cols += profiler_cols
        if "weather" in input_key:
            included_cols += weather_cols
        if "scada" in output_key:
            included_cols += SCADA_cols
        if "eurofins_phys" in output_key:
            included_cols += Eurofins_phys
        if "eurofins_bio" in output_key:
            included_cols += Eurofins_bio
        if "eurofins_metal" in output_key:
            included_cols += Eurofins_metal
        # Convert dict to go.Figure
        fig = go.Figure()
        normalization_param_file = 'normalization.json'

        if source_key == 'cleaned':
            if normalize_key == 'normalized':
                filepath = Path("../data/input/" + normalization_param_file)
                try:
                    with open(filepath, 'w') as f:
                        pass
                except FileNotFoundError:
                    print(f"Error: File not found at {filepath}")
                    pass
                full_df = normalize_columns(full, included_cols, save=True, directory="../data/input",
                                            filename=normalization_param_file)
                title = "Cleaned, Normalized Dataset (for models)"
            else:
                full_df = full
                title = "Cleaned Dataset (for models)"
            dfs = [full_df]
        else:
            dfs = []
            if normalize_key == 'normalized':
                filepath = Path("../data/input/" + normalization_param_file)
                try:
                    with open(filepath, 'w') as f:
                        pass
                except FileNotFoundError:
                    print(f"Error: File not found at {filepath}")
                    pass
                for df in raw:
                    dfs.append(normalize_columns(df, included_cols, param_file=Path('../data/input', normalization_param_file), save=True,
                                                 directory="../data/input", filename=normalization_param_file))
                title = "Original Datasets, Normalized"
            else:
                dfs = raw
                title = "Original Datasets, Original Values"

        # Dash patterns for years
        dash_styles = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

        # Add data from sources to plot as traces

        if mode_key == 'seasonality':  # Seasonality plot (use single year as x-range, overlay source data by year)
            for plot_df in dfs:
                # Extract years and time from start of year
                plot_df['year'] = plot_df['TIMESTAMP'].dt.year
                plot_df['time'] = (plot_df['TIMESTAMP'].dt.dayofyear - 1) * 24 + plot_df['TIMESTAMP'].dt.hour
                for col in plot_df.columns:
                    if col in included_cols:
                        for i, year in enumerate(sorted(plot_df['year'].unique())):
                            subset = plot_df[plot_df['year'] == year]
                            fig.add_trace(go.Scatter(
                                x=subset['time'], y=subset[col],mode='lines',name=f"{col} ({year})", connectgaps=True,
                                line = dict(dash=dash_styles[i % len(dash_styles)]))
                            )
                fig.update_xaxes(title='Month', tickmode='array', tickvals=[24,768,1464,2208,2928,3672,4392,5136,5880,6600,7344,8064,8808],
                                 ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                                           'Dec', 'Jan'])
        else:  # Use the x-axis to show the entire range of source data continuously
            for df in dfs:
                for col in df.columns:
                    if col in included_cols:
                        fig.add_trace(go.Scatter(x=df["TIMESTAMP"], y=df[col], mode="lines", name=col, connectgaps=True))

        if thresholds_key == 'on':
            raw_thresholds_df = pd.read_csv('../data/input/Limits.csv', sep=';', decimal='.')
            raw_thresholds_df = raw_thresholds_df.astype(float)
            if normalize_key == 'normalized':
                thresholds_df = normalize_columns(raw_thresholds_df, included_cols,
                                                  param_file=Path('../data/input',normalization_param_file),
                                                  save=False)
                for col in thresholds_df.columns:
                    if col in included_cols:
                        if thresholds_df[col].max() > 1:
                            thresholds_df[col] = 1
                        if thresholds_df[col].min() < 0:
                            thresholds_df[col] = 0
                    else:
                        thresholds_df.drop(columns=[col], inplace=True)
            else:
                thresholds_df = raw_thresholds_df[raw_thresholds_df.columns.intersection(included_cols)]
            thresholds_dict = {col: thresholds_df[col].iloc[0] for col in thresholds_df.columns}

            # Extract x-axis range
            all_x = []
            for trace in fig.data:
                if hasattr(trace, 'x'):
                    all_x.extend(trace.x)
            x_min = pd.to_datetime(min(all_x))
            x_max = pd.to_datetime(max(all_x))

            if mode_key == 'continuous':
                for label, val in thresholds_dict.items():
                    fig.add_trace(go.Scatter(
                        x=[x_min, x_max],
                        y=[val, val],
                        mode='lines',
                        name=label+' limit', connectgaps=True
                    ))
                fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value",
                                  xaxis_range=[pd.to_datetime(start_date_key), pd.to_datetime(end_date_key)])
            elif mode_key == 'seasonality':
                for label, val in thresholds_dict.items():
                    fig.add_trace(go.Scatter(
                        x=[0, 8760],
                        y=[val, val],
                        mode='lines',
                        name=label+' limit', connectgaps=True
                    ))
                fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value",
                                  xaxis_range=[0, 8760])
        else:
            if mode_key == 'continuous':
                fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value",
                                      xaxis_range=[pd.to_datetime(start_date_key), pd.to_datetime(end_date_key)])
            else:
                fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value",
                                  xaxis_range=[0, 8760])
        return fig
    app.run(debug=True)


if __name__ == '__main__':
    # Load data files
    df = pd.read_csv("../data/input/sensors/FullHourly.csv")  # Profiler data
    weather_df = pd.read_csv("../data/input/sensors/Weather.csv", sep=";", decimal=",", parse_dates=["Time"])
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
    simplified_weather_names = {"Hourly average wind direction (°)_x":"Wind speed, x (m/s)",
                                "Hourly average wind direction (°)_y": "Wind speed, y (m/s)",
                                "Maximum sustained wind speed, 3-second span (m/s)": 'Maximum 3s wind gust (m/s)',
                                "Instantaneous atmospheric pressure compensated for temperature, humidity and station "
                                "elevation (mBar)": "Atmospheric pressure (mBar)",
                                'Instantaneous temperature (°C)': 'Air temperature (°C)',
                                'Average humidity (% relative humidity)': 'Humidity (%)',
                                'rolling Precipitation (mm/hr)': '24hr precipitation total (mm)'}
    weather_simp_df.rename(columns=simplified_weather_names, inplace=True)


    scada_df = pd.read_csv("../data/input/sensors/SCADA.csv", sep=";", decimal=".", parse_dates=["Time"])
    eurofins_df = pd.read_csv("../data/input/sensors/Eurofins.csv", sep=";", decimal=",", parse_dates=["Time"])

    # Clean profiler dataset, which is foundation for other sources
    clean_df = clean_profiler(df, max_gap=6)

    # Merge data from other sources into the dataset.
    merge1_df = add_source(clean_df, weather_simp_df, include_NAs=False, max_gap=6)
    merge2_df = add_source(merge1_df, scada_df, include_NAs=True, max_gap=6)
    # merge3_df = add_source(merge2_df, eurofins_df, include_NAs=True, max_gap=6, binarize=False)  # For regression
    merge3_df = add_source(merge2_df, eurofins_df, include_NAs=True, max_gap=6, binarize=True)  # For classification

    segmented_df = count_segs(merge3_df)  # Add column with index for continuous segments

    ## Save the cleaned and merged dataset
    # For regression (with optional post-process classification)
    # output_dir = "../data/output/regression"
    # filename = "Consolidated.csv"

    # For classification only:
    output_dir = "../data/output/classification"
    filename = "Consolidated_binarized.csv"


    ## Write a combined dataset to file (either regression or classification)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    segmented_df.to_csv(Path(output_dir, filename), index=False)

    ## Visualize datasets in a browser window:
    # explore_data(merge3_df, clean_df, weather_simp_df, scada_df, eurofins_df)

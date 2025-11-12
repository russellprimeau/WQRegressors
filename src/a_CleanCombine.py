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
from dash import dcc, html, Output, Input, State
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
        "RECORD": "Pfl - Record Number",
        "PFL_Counter": "Pfl - Day",
        "CntRS232": "Pfl - CntRS232",
        "RS232Dpt": "Pfl - Vertical Position1 (m)",
        "sensorParms(1)": "Pfl - Temp (C)",
        "sensorParms(2)": "Pfl - Cond (microS_cm)",
        "sensorParms(3)": "Pfl - Sp Cond (microS_cm)",
        "sensorParms(4)": "Pfl - Salinity (ppt)",
        "sensorParms(5)": "Pfl - pH",
        "sensorParms(6)": "Pfl - DO (% Sat)",
        "sensorParms(7)": "Pfl - Turbidity (NTU)",
        "sensorParms(8)": "Pfl - Turbidity (FNU)",
        "sensorParms(9)": "Pfl - Vertical Position (m)",
        "sensorParms(10)": "Pfl - fDOM (RFU)",
        "sensorParms(11)": "Pfl - fDOM (QSU)",
        "lat": "Latitude",
        "lon": "Longitude",
    }
    df = full_df.rename(columns=column_names)
    sensor_columns = ["Pfl - Temp (C)", "Pfl - Sp Cond (microS_cm)", "Pfl - pH",
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
    binary_df["anomaly"] = binary_df[output_columns].bool(axis=1)
    return binary_df

def decompose_direction(df, column_name):
    """
    Replace a column containing degree values with two new columns representing
    the x and y components using cosine and sine. The new columns are inserted
    at the same position as the original column.

    Parameters:
    - df: pandas.DataFrame
    - column_name: str, name of the column containing degree values

    Returns:
    - Modified DataFrame with x and y components replacing the original column
    """
    df_copy = df.copy()
    radians = np.deg2rad(df_copy[column_name])
    x_component = np.cos(radians)
    y_component = np.sin(radians)

    insert_position = df_copy.columns.get_loc(column_name)
    df_copy.drop(columns=[column_name], inplace=True)
    df_copy.insert(insert_position, f"{column_name}_x", x_component)
    df_copy.insert(insert_position + 1, f"{column_name}_y", y_component)

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
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
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

def explore_data(full, *raw):
    app = dash.Dash(__name__)
    fig = go.Figure()
    app.layout = html.Div([
        html.H3("Toggle Data Sources"),
        dcc.RadioItems(
            id='source-select',
            options=[
                {'label': 'Cleaned', 'value': 'Cleaned'},
                {'label': 'Original Sources', 'value': 'Original Sources'},
            ],
            value='Cleaned',
            inline=True
        ),
        dcc.RadioItems(
            id='normalize',
            options=[
                {'label': 'Normalized on (0,1)', 'value': 'Normalized'},
                {'label': 'Original Units', 'value': 'Original Values'},
            ],
            value='Normalized',
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
            value='off',
            inline=True
        ),
        html.Button('Toggle All Traces', id='toggle-btn', n_clicks=0),
        dcc.Graph(id='timeseries-plot', figure=fig, style={'height': '90vh', 'width': '100%'})
    ],
        style={'height': '100vh', 'width': '100%', 'padding': '10px', 'box-sizing': 'border-box'})

    @app.callback(
        Output('timeseries-plot', 'figure'),
        [Input('source-select', 'value'),
        Input('normalize', 'value'),
        Input('toggle-btn', 'n_clicks'),
         Input('plot-mode', 'value'),
         Input('thresholds', 'value')],
        State('timeseries-plot', 'figure')
    )

    def update_plot(source_key, normalize_key, n_clicks, mode, thresholds, current_fig):
        print(f"Before update: {len(current_fig['data'])} traces")
        ctx = dash.callback_context
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]

        if triggered == 'toggle-btn':
            # Toggle visibility only
            fig = go.Figure(current_fig)
            all_visible = all(trace.visible == True for trace in fig.data)
            for trace in fig.data:
                trace.visible = 'legendonly' if all_visible else True
            return fig

        to_normalize = ['Pfl - Temp (C)', 'Pfl - Sp Cond (microS_cm)',
                        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                        'Pfl - fDOM (QSU)',
                        'Instantaneous atmospheric pressure (mBar)', 'Wind direction 10minRollingAvg (°)_x',
                        'Wind direction 10minRollingAvg (°)_y',
                        'Hourly average wind direction (°)_x', 'Hourly average wind direction (°)_y',
                        'Average wind speed (m/s)',
                        'Maximum sustained wind speed, 3-second span (m/s)', 'Time of maximum 3s Gust',
                        'Maximum sustained wind speed, 10-minute span (m/s)', 'Time of maximum 10 minute gust',
                        'Hourly average atmospheric pressure at station (mBar)',
                        'Maximum pressure differential, 3-hour span (mBar)',
                        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
                        'Longwave (IR) radiation (W/m2)', 'Instantaneous sea-level atmospheric pressure (mBar)',
                        'Shortwave (solar) radiation (W/m2)', 'Precipitation (mm/hr)',
                        'Instantaneous temperature (°C)',
                        'Maximum temperature (°C)', 'Minimum temperature (°C)',
                        'Average humidity (% relative humidity)',
                        'SCADA - pH', 'SCADA - Temperature (°C)', '01-Farge', '04-Turbiditet', '06-E.coli',
                        '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen',
                        '24-Bly', '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)',
                        '44-pH, surhetsgrad']
        # Convert dict to go.Figure
        fig = go.Figure()

        if source_key == 'Cleaned':
            if normalize_key == 'Normalized':
                full_df = normalize_columns(full, to_normalize)
                title = "Cleaned, Normalized Dataset (for models)"
            else:
                full_df = full
                title = "Cleaned Dataset (for models)"
            dfs = [full_df]
        else:
            dfs = []
            if normalize_key == 'Normalized':
                for df in raw:
                    dfs.append(normalize_columns(df, to_normalize))
                title = "Original Datasets, Normalized"
            else:
                dfs = raw
                title = "Original Datasets, Original Values"

        # Dash patterns for years
        dash_styles = ['solid', 'dash', 'dot', 'dashdot']

        # Add traces
        if mode == 'continuous':
            for df in dfs:
                for col in df.columns:
                    if col not in ["TIMESTAMP", "TIME", "Interpolated", "Segment", "year", "time"]:
                        fig.add_trace(go.Scatter(x=df["TIMESTAMP"], y=df[col], mode="lines", name=col, connectgaps=True))
        else:
            for df in dfs:
                # Extract years and months
                df['year'] = df['TIMESTAMP'].dt.year
                df['time'] = (df['TIMESTAMP'].dt.dayofyear - 1) * 24 + df['TIMESTAMP'].dt.hour
                for col in df.columns:
                    if col not in ["TIMESTAMP", "TIME", "Interpolated", "Segment", "year", "time"]:
                        for i, year in enumerate(sorted(df['year'].unique())):
                            subset = df[df['year'] == year]
                            fig.add_trace(go.Scatter(
                                x=subset['time'], y=subset[col],mode='lines',name=f"{col} ({year})",
                                line=dict(dash=dash_styles[i % len(dash_styles)]))
                            )

                fig.update_xaxes(title='Month', tickmode='array', tickvals=list(range(1, 8760, 720)),
                                 ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                                           'Dec'])

        print(f"Before thresholds update: {len(fig.data)} traces")
        if thresholds == 'on':
            raw_thresholds_df = pd.read_csv('../data/input/Limits.csv', sep=';', decimal='.')
            raw_thresholds_df = raw_thresholds_df.astype(float)
            print(raw_thresholds_df.dtypes)
            Eurofins_columns = ['01-Farge', '04-Turbiditet', '06-E.coli',
                        '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen',
                        '24-Bly', '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)',
                        '44-pH, surhetsgrad']
            if normalize_key == 'Normalized':
                thresholds_df = normalize_columns(raw_thresholds_df, Eurofins_columns)
            else:
                thresholds_df = raw_thresholds_df
            thresholds_dict = {col + ' limit': thresholds_df[col].iloc[0] for col in thresholds_df.columns}
            print(thresholds_dict)

            # Extract x-axis range
            x_range = []
            all_x = []
            for trace in fig.data:
                if hasattr(trace, 'x'):
                    all_x.extend(trace.x)
            x_range = [min(all_x), max(all_x)]

            for label, val in thresholds_dict.items():
                print(f"Limit for {label}: {val} on {x_range}")
                fig.add_trace(go.Scatter(
                    x=(x_range[0], x_range[1]),
                    y=[val, val],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name=label
                ))

        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value")
        print(f"After update: {len(fig.data)} traces")

        # Toggle logic
        if n_clicks > 0:
            all_visible = all(trace.visible == True for trace in fig.data)
            for trace in fig.data:
                trace.visible = 'legendonly' if all_visible else True

        return fig

    app.run(debug=True)


if __name__ == '__main__':
    # Load data files
    df = pd.read_csv("../data/input/sensors/FullHourly.csv")  # Profiler data
    weather_df = pd.read_csv("../data/input/sensors/Weather.csv", sep=";", decimal=",", parse_dates=["Time"])
    weather_columns = {"1818_time: AA[mBar]": "Instantaneous atmospheric pressure (mBar)",
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
    weather_df.rename(columns=weather_columns, inplace=True)
    decomp_df = decompose_direction(weather_df, "Wind direction 10minRollingAvg (°)")
    decomp2_df = decompose_direction(decomp_df, "Hourly average wind direction (°)")

    scada_df = pd.read_csv("../data/input/sensors/SCADA.csv", sep=";", decimal=".", parse_dates=["Time"])
    eurofins_df = pd.read_csv("../data/input/sensors/Eurofins.csv", sep=";", decimal=",", parse_dates=["Time"])

    # Clean profiler dataset, which is foundation for other sources
    clean_df = clean_profiler(df, max_gap=6)

    # Merge data from other sources into the dataset.
    merge1_df = add_source(clean_df, decomp2_df, include_NAs=False, max_gap=6)
    merge2_df = add_source(merge1_df, scada_df, include_NAs=True, max_gap=6)
    merge3_df = add_source(merge2_df, eurofins_df, include_NAs=True, max_gap=6, binarize=False)
    # merge3_df = add_source(merge2_df, eurofins_df, include_NAs=True, max_gap=6, binarize=True)

    segmented_df = count_segs(merge3_df)  # Add column with index for continuous segments

    ## Save the cleaned and merged dataset
    # For regression (with optional post-process classification)
    output_dir = "../data/output/regression"
    filename = "Consolidated.csv"

    # For classification only:
    # output_dir = "../data/output/classification"
    # filename = "Consolidated_binarized.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # segmented_df.to_csv(Path(output_dir, filename), index=False)

    explore_data(merge3_df, clean_df, decomp2_df, scada_df, eurofins_df)

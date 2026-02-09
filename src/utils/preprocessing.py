import json
import numpy as np
import pandas as pd
from pathlib import Path
import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

def add_source(df, secondary_df, include_NAs=False, max_gap=6, sparse=False, binarize=False):
    # Rename 'Time' in secondary_df to match 'TIMESTAMP' in primary_df
    secondary_df.rename(columns={"Time": "TIMESTAMP"}, inplace=True)

    # === MERGE ON TIMESTAMP ===
    if sparse:
        merged_df = pd.concat([df, secondary_df], axis=1)
    else:
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

    if not sparse:
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
        html.H3("Configure plot area"),
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
        Eurofins_phys = ['01-Farge', '04-Turbiditet', '44-pH']
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


import pandas as pd


def forward_fill_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    For each column in 'columns' that exists in 'df', create a new column
    where NaN values are replaced by the last non-NaN value (forward fill).
    The new column will have the suffix '_filled'.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        columns (list): List of column names to forward fill.

    Returns:
        pd.DataFrame: Modified dataframe with new forward-filled columns.
    """
    final_df = df.copy()

    for col in columns:
        if col in final_df.columns:
            final_df[f"{col}_state"] = final_df[col].ffill()

    return final_df


import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


def add_res(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    For each column in 'columns' that exists in 'df', create a new column 'col_res'.
    Rules:
      - If the value in col is NaN, col_res is NaN.
      - For the first non-NaN value in col, col_res is NaN.
      - Otherwise, col_res is the difference between the current value and the previous non-NaN value.
    Values in the new column are rounded to 3 decimal places.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        columns (list): List of column names to process.

    Returns:
        pd.DataFrame: Modified dataframe with new difference columns.
    """
    final_df = df.copy()

    for col in columns:
        if col in final_df.columns:
            # Forward fill to get previous non-NaN values
            prev_vals = final_df[col].ffill()

            # Compute difference
            diffs = final_df[col] - prev_vals.shift(1)

            # Ensure first non-NaN gets NaN
            first_non_nan_idx = final_df[col].first_valid_index()
            if first_non_nan_idx is not None:
                diffs.iloc[first_non_nan_idx] = np.nan

            # Round to 3 decimal places
            final_df[f"{col}_res"] = diffs.round(3)

    return final_df


def prepare_tsne_plot(
    df_or_path: pd.DataFrame | str | Path,
    feature_columns: list | None = None,
    timestamp_col: str = "TIMESTAMP",
    perplexity: int = 30,
    learning_rate: str | float = "auto",
    n_iter: int = 1000,
    random_state: int = 42,
    init: str = "pca",
    scale: bool = True,
    drop_all_nan_cols: bool = True,
    return_fig: bool = True,
    include_time_features: bool = False,
):
    """
    Prepare a t-SNE embedding (and optional Plotly scatter) for multivariate datasets
    such as Eurofins.csv. Handles missing values, numeric coercion, and scaling.
    
    Args:
        include_time_features: If True, extract temporal features (month, day_of_year) 
                              from timestamp and include them in the t-SNE analysis.

    Returns:
        embedding_df (pd.DataFrame): Contains tsne_1, tsne_2, original_index, timestamp,
                                     and all original feature values for traceability.
        fig (plotly.graph_objects.Figure | None): Interactive plot if return_fig=True.
    """
    if isinstance(df_or_path, (str, Path)):
        work_df = pd.read_csv(df_or_path, sep=";", decimal=".")
    else:
        work_df = df_or_path.copy()

    if timestamp_col not in work_df.columns and "Time" in work_df.columns:
        work_df = work_df.rename(columns={"Time": timestamp_col})

    work_df = work_df.replace([r"^\s*$", "NA", "#N/A"], np.nan, regex=True)

    if feature_columns is None:
        numeric_df = work_df.apply(pd.to_numeric, errors="coerce")
        if timestamp_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[timestamp_col])
        feature_columns = numeric_df.columns.tolist()
        work_df[feature_columns] = numeric_df
    else:
        work_df[feature_columns] = work_df[feature_columns].apply(pd.to_numeric, errors="coerce")

    if drop_all_nan_cols:
        all_nan_cols = [col for col in feature_columns if work_df[col].isna().all()]
        if all_nan_cols:
            work_df = work_df.drop(columns=all_nan_cols)
            feature_columns = [col for col in feature_columns if col not in all_nan_cols]

    if not feature_columns:
        raise ValueError("No valid numeric feature columns available for t-SNE.")

    # Store original index before filtering
    work_df = work_df.reset_index(drop=False).rename(columns={'index': 'original_index'})
    
    # Add temporal features if requested
    time_features = []
    if include_time_features and timestamp_col in work_df.columns:
        work_df[timestamp_col] = pd.to_datetime(work_df[timestamp_col], errors='coerce')
        work_df['month'] = work_df[timestamp_col].dt.month
        work_df['day_of_year'] = work_df[timestamp_col].dt.dayofyear
        time_features = ['month', 'day_of_year']
        feature_columns = feature_columns + time_features
    
    work_df = work_df.dropna(subset=feature_columns, how="any")

    X = work_df[feature_columns].to_numpy()

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("t-SNE requires at least 2 samples.")
    max_perplexity = max(1, (n_samples - 1) // 3)
    effective_perplexity = min(perplexity, max_perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        random_state=random_state,
        init=init,
    )
    embedding = tsne.fit_transform(X)

    embedding_df = pd.DataFrame(embedding, columns=["tsne_1", "tsne_2"])
    embedding_df['original_index'] = work_df['original_index'].values
    
    if timestamp_col in work_df.columns:
        embedding_df[timestamp_col] = pd.to_datetime(work_df[timestamp_col].values, errors="coerce")
    
    # Include all original feature values for full traceability
    for col in feature_columns:
        embedding_df[col] = work_df[col].values

    fig = None
    if return_fig:
        # Build comprehensive hover text
        hover_text = []
        for idx, row in embedding_df.iterrows():
            text = f"Index: {row['original_index']}<br>"
            if timestamp_col in embedding_df.columns and pd.notna(row[timestamp_col]):
                text += f"Time: {row[timestamp_col]}<br>"
            text += f"t-SNE 1: {row['tsne_1']:.3f}<br>t-SNE 2: {row['tsne_2']:.3f}"
            hover_text.append(text)
        
        fig = go.Figure(
            data=go.Scatter(
                x=embedding_df["tsne_1"],
                y=embedding_df["tsne_2"],
                mode="markers",
                marker=dict(size=6, opacity=0.75),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        fig.update_layout(title="t-SNE embedding", xaxis_title="t-SNE 1", yaxis_title="t-SNE 2")

    return embedding_df, fig

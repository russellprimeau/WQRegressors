data_columns = ['Pfl - Temp (C)', 'Pfl - Sp Cond (microS_cm)',
    'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)', 'Pfl - fDOM (QSU)',
    'Instantaneous atmospheric pressure (mBar)', 'Wind direction 10minRollingAvg (°)_x',
    'Wind direction 10minRollingAvg (°)_y', 'Hourly average wind direction (°)_x',
    'Hourly average wind direction (°)_y', 'Average wind speed (m/s)',
    'Maximum sustained wind speed, 3-second span (m/s)', 'Time of maximum 3s Gust',
    'Maximum sustained wind speed, 10-minute span (m/s)', 'Time of maximum 10 minute gust',
    'Hourly average atmospheric pressure at station (mBar)', 'Maximum pressure differential, 3-hour span (mBar)',
    'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
    'Longwave (IR) radiation (W/m2)', 'Instantaneous sea-level atmospheric pressure (mBar)',
    'Shortwave (solar) radiation (W/m2)', 'Precipitation (mm/hr)', 'Instantaneous temperature (°C)',
    'Maximum temperature (°C)', 'Minimum temperature (°C)', 'Average humidity (% relative humidity)',
    'SCADA - pH', 'SCADA - Temperature (°C)', '01-Farge', '04-Turbiditet', '06-E.coli',
    '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen', '24-Bly',
    '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']

all_columns = ['TIMESTAMP', 'Segment', 'Interpolated'] + data_columns

outputs = ['01-Farge', '04-Turbiditet', '06-E.coli',
    '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen', '24-Bly',
    '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']

state = ['01-Farge_state', '04-Turbiditet_state', '06-E.coli_state',
    '07-Intestinale enterokokker_state', '08-Kimtall 22°C_state', '09-Koliforme bakterier 37°C_state', '21-Arsen_state',
             '24-Bly_state', '32-Kadmium_state', '36-Kopper filtrert_state', '37-Krom_state', '41-Nikkel_state', 'Sink (Zn)_state']


residuals = ['01-Farge_res', '04-Turbiditet_res', '06-E.coli_res',
    '07-Intestinale enterokokker_res', '08-Kimtall 22°C_res', '09-Koliforme bakterier 37°C_res', '21-Arsen_res',
             '24-Bly_res', '32-Kadmium_res', '36-Kopper filtrert_res', '37-Krom_res', '41-Nikkel_res', 'Sink (Zn)_res']
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

diffs = ['01-Farge_diff', '04-Turbiditet_diff', '06-E.coli_diff',
    '07-Intestinale enterokokker_diff', '08-Kimtall 22°C_diff', '09-Koliforme bakterier 37°C_diff', '21-Arsen_diff',
             '24-Bly_diff', '32-Kadmium_diff', '36-Kopper filtrert_diff', '37-Krom_diff', '41-Nikkel_diff', 'Sink (Zn)_diff']


import re as _re

# Standalone tokens that represent units and should be dropped from display labels.
# Evaluated after digits and the characters °, µ, / have been removed.
_UNIT_TOKENS = {"g", "mg", "mL", "L", "CFU", "C", "FNU"}


def clean_target_label(dataset_name: str, prefix: str = "MC") -> str:
    """Return a clean display label for a dataset directory name.

    Strips (in order):
      1. Dataset prefix + underscore (e.g. ``MC_``)
      2. Leading ``ex`` marker
      3. Trailing ``_diff``, ``_res`` and ``_state`` suffixes
      4. Trailing parenthesised unit block ``_(...)``
      5. Underscores → spaces
      6. Digits and the characters °, µ, /
      7. Standalone unit-word tokens (g, mg, mL, L, CFU, C, FNU)
      8. Normalises runs of whitespace; strips leading/trailing spaces

    Examples::

        clean_target_label('MC_exArsenic_(µg_L)_res')
        # 'Arsenic'
        clean_target_label('MC_exColony_Count_22°C_(CFU_mL)_res')
        # 'Colony Count'
        clean_target_label('MC_exIntestinal_enterococci_(CFU_100mL)_res')
        # 'Intestinal enterococci'
    """
    lbl = dataset_name
    bare = prefix.rstrip("_")
    if lbl.startswith(bare + "_"):
        lbl = lbl[len(bare) + 1:]
    if lbl.startswith("ex"):
        lbl = lbl[2:]
    for sfx in ("_diff", "_res", "_state"):
        if lbl.endswith(sfx):
            lbl = lbl[: -len(sfx)]
    lbl = _re.sub(r"_\([^)]*\)$", "", lbl)        # trailing _(unit) block
    lbl = lbl.replace("_", " ")
    lbl = _re.sub(r"\d+", "", lbl)                 # digits
    lbl = lbl.translate(str.maketrans("", "", "°µ/"))  # unit characters
    words = [w for w in lbl.split() if w not in _UNIT_TOKENS]
    return " ".join(words).strip()

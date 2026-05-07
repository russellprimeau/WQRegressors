"""Physical-plausibility bounds for sensor columns.

Mirrors the per-column min/max exclusion rules applied during dataset
consolidation (see clean_profiler in preprocessing.py and the inline
weather rules in a_ConsolidateDatasets.py). Centralised here so other
utilities (e.g. correlation analysis) can re-apply the same rules to
the consolidated CSV, which contains interpolated values that may fall
outside the original sensor's plausible range.

Bounds are expressed as (min_inclusive, max_inclusive). Either side may
be None to indicate no bound on that side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PROFILER_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "Pfl - Water temperature (°C)": (1, 25),
    "Pfl - Cond (microS_cm)": (0, 45),
    "Pfl - Sp Cond (microS_cm)": (1, None),
    "Pfl - Salinity (ppt)": (0, 0.03),
    "Pfl - pH": (2, 12),
    "Pfl - DO (% Sat)": (10, 120),
    "Pfl - Turbidity (NTU)": (0, None),
    "Pfl - Turbidity (FNU)": (0, None),
    "Pfl - fDOM (RFU)": (0, 100),
    "Pfl - fDOM (QSU)": (0, 300),
}

WEATHER_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "Hourly average wind direction (°)": (0, 360),
    "Average wind speed (m/s)": (0, 100),
    "Maximum sustained wind speed, 3-second span (m/s)": (0, 100),
    "Maximum sustained wind speed, 10-minute span (m/s)": (0, 100),
    "Atmospheric pressure (mBar)": (860, 1080),
    "Maximum pressure differential, 3-hour span (mBar)": (0, 50),
    "Longwave (IR) radiation (W/m2)": (0, 750),
    "Shortwave (solar) radiation (W/m2)": (0, 900),
    "Precipitation (mm/hr)": (0, 50),
    "24hr precipitation total (mm)": (0, 24 * 50),
    "Maximum temperature (°C)": (-40, 40),
    "Minimum temperature (°C)": (-40, 40),
    "Air temperature (°C)": (-40, 40),
    "Humidity (%)": (0, 100),
}

ALL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    **PROFILER_BOUNDS,
    **WEATHER_BOUNDS,
}


def apply_plausibility_bounds(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float | None, float | None]] = ALL_BOUNDS,
) -> pd.DataFrame:
    """Return a copy of df with values outside per-column bounds set to NaN.

    Columns absent from df are silently skipped. Per-cell masking only
    nullifies the offending value and leaves other columns at the same
    timestamp untouched.
    """
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        mask = pd.Series(False, index=out.index)
        if lo is not None:
            mask |= series < lo
        if hi is not None:
            mask |= series > hi
        out.loc[mask, col] = np.nan
    return out

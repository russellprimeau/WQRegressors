"""Canonical display labels for the 31 study parameters.

The column names stored in ``Consolidated_sparse.csv`` are the *data contract*: the
resample YAMLs under ``data/input/splitting/``, the generated per-dataset training
configs, ``data/input/normalization.json`` and ``utils/plausibility.py`` all key on them.
They are therefore not renamed.  They do, however, carry logger-era compromises that must
not reach a printed figure -- ``microS_cm`` for micro-siemens per centimetre, ``W/m2`` for
an exponent, ``mBar`` for millibar.

This module is the single display layer between the two.  Plotting code asks for a label;
it never derives one by string-munging a column name.

Two rules the registry exists to enforce:

*Units belong to values, not to names.*  A label carries its unit when the figure plots
measured values in that unit, and drops it when the figure plots a statistic *about* a
model of that parameter (an R^2, a z-score, an inclusion flag).  That is the
``with_unit`` argument, not a per-call-site judgement.

*Source qualifiers appear only where they disambiguate.*  pH, turbidity and water
temperature are each measured by more than one dataset, so those carry ``(Surface)`` /
``(SCADA)`` / ``(lab)``.  Everything else is unique and needs no prefix -- which is why
``Pfl -`` is dropped rather than expanded.
"""
from __future__ import annotations

import re as _re
from dataclasses import dataclass

__all__ = [
    "Param",
    "PARAMS",
    "SURFACE",
    "SCADA",
    "WEATHER",
    "SAMPLES",
    "label",
    "unit",
    "axis_label",
    "clean_target_label",
    "slug",
]

SURFACE = "Surface"
SCADA = "SCADA"
WEATHER = "Weather"
SAMPLES = "Samples"


@dataclass(frozen=True)
class Param:
    """Display metadata for one stored column.

    ``short`` is for tick labels and legends; ``long`` for axis labels.  ``unit`` is
    matplotlib mathtext and is ``None`` for dimensionless quantities (pH, colour index),
    which is different from "unit not yet decided".

    ``qualifier`` is the source disambiguator (``Surface`` / ``SCADA`` / ``lab``) and is
    set only on the three quantities measured by more than one dataset.  It is applied by
    :func:`label` on request rather than baked into ``short``, because whether it is
    needed depends on the figure: a figure mixing Surface and SCADA pH must distinguish
    them, whereas one whose every entry is a laboratory target would only add noise.

    ``unit_is_identity`` marks the case where the unit is not a property of the values but
    the only thing distinguishing two otherwise identically-named series -- the two fDOM
    calibrations.  Such a unit is shown even when ``with_unit=False``, because dropping it
    would merge two distinct series into one label.
    """

    short: str
    long: str
    unit: str | None
    group: str
    qualifier: str | None = None
    unit_is_identity: bool = False


def _p(short, unit, group, long=None, qualifier=None, unit_is_identity=False):
    return Param(short=short, long=long or short, unit=unit, group=group,
                 qualifier=qualifier, unit_is_identity=unit_is_identity)


# --- Surface: EXO sonde on the vertical profiler, ~2.3 m depth --------------------
# Stored with a "Pfl - " prefix from the datalogger channel map in preprocessing.py.
_SURFACE: dict[str, Param] = {
    "Pfl - Water temperature (°C)": _p("Water temp.", "°C", SURFACE,
                                       "Water temperature", qualifier=SURFACE),
    "Pfl - Sp Cond (microS_cm)": _p("Specific conductance", r"$\mu$S/cm", SURFACE),
    "Pfl - pH": _p("pH", None, SURFACE, qualifier=SURFACE),
    "Pfl - DO (% Sat)": _p("Dissolved oxygen", "% sat.", SURFACE),
    "Pfl - Turbidity (FNU)": _p("Turbidity", "FNU", SURFACE, qualifier=SURFACE),
    # The two fDOM series are the same quantity in two calibrations, so the unit is what
    # tells them apart and is never dropped -- see Param.unit_is_identity.
    "Pfl - fDOM (RFU)": _p("fDOM", "RFU", SURFACE, unit_is_identity=True),
    "Pfl - fDOM (QSU)": _p("fDOM", "QSU", SURFACE, unit_is_identity=True),
    # Present in the raw profiler feed but not in the consolidated predictor set.
    "Pfl - Cond (microS_cm)": _p("Conductivity", r"$\mu$S/cm", SURFACE),
    "Pfl - Salinity (ppt)": _p("Salinity", "ppt", SURFACE),
    "Pfl - Turbidity (NTU)": _p("Turbidity, NTU", "NTU", SURFACE, qualifier=SURFACE),
    "Pfl - Vertical position (m)": _p("Sonde depth", "m", SURFACE),
}

# --- SCADA: treatment-plant raw-water intake --------------------------------------
_SCADA: dict[str, Param] = {
    "SCADA - pH": _p("pH", None, SCADA, qualifier=SCADA),
    "SCADA - Temperature (°C)": _p("Water temp.", "°C", SCADA,
                                   "Water temperature", qualifier=SCADA),
}

# --- Weather: local station, gap-filled from NORA3 --------------------------------
_WEATHER: dict[str, Param] = {
    # x = cos(bearing)*speed, y = sin(bearing)*speed on the meteorological "from"
    # bearing (utils/preprocessing.py:decompose_direction).  That is not the standard
    # eastward/northward u/v convention, so the components are named neutrally.
    "Wind speed x (m/s)": _p("Wind speed, x", "m/s", WEATHER, "Wind speed, x-component"),
    "Wind speed y (m/s)": _p("Wind speed, y", "m/s", WEATHER, "Wind speed, y-component"),
    "Atmospheric pressure (mBar)": _p("Atmospheric pressure", "mbar", WEATHER),
    "Longwave (IR) radiation (W/m2)": _p("Longwave (IR) irradiance", r"W/m$^2$", WEATHER),
    "Shortwave (solar) radiation (W/m2)": _p("Shortwave (solar) irradiance", r"W/m$^2$",
                                             WEATHER),
    # Cumulative, unlike the instantaneous 'Precipitation (mm/hr)' it is derived from.
    "24hr precipitation total (mm)": _p("Precipitation, 24 h total", "mm", WEATHER),
    "Precipitation (mm/hr)": _p("Precipitation rate", "mm/h", WEATHER),
    "Air temperature (°C)": _p("Air temperature", "°C", WEATHER),
    "Humidity (%)": _p("Relative humidity", "%", WEATHER),
    "Maximum 3s wind gust (m/s)": _p("Wind gust, 3 s", "m/s", WEATHER),
}

# --- Samples: Eurofins laboratory analyses (the forecast targets) -----------------
# Incubation temperatures (22 °C / 37 °C) are part of the method and distinguish the two
# culture counts; they are never stripped.
_SAMPLES: dict[str, Param] = {
    # NOTE: the colour index is reported without a unit anywhere in the pipeline
    # (data/input/Limits.csv gives a limit of 20 with a blank unit).  Norwegian 'Farge'
    # for drinking water is conventionally mg Pt/L; flagged for author confirmation.
    "Color": _p("Color", "mg Pt/L", SAMPLES),
    "Turbidity (FNU)": _p("Turbidity", "FNU", SAMPLES, qualifier="lab"),
    "pH": _p("pH", None, SAMPLES, qualifier="lab"),
    "E.coli (CFU/100mL)": _p("$\\it{E.\\ coli}$", "CFU/100 mL", SAMPLES),
    "Intestinal enterococci (CFU/100mL)": _p("Intestinal enterococci", "CFU/100 mL",
                                             SAMPLES),
    "Colony Count 22°C (CFU/mL)": _p("Colony count, 22 °C", "CFU/mL", SAMPLES),
    "Total coliforms 37°C (CFU/100mL)": _p("Total coliforms, 37 °C", "CFU/100 mL",
                                           SAMPLES),
    "Arsenic (µg/L)": _p("Arsenic", r"$\mu$g/L", SAMPLES),
    "Lead (µg/L)": _p("Lead", r"$\mu$g/L", SAMPLES),
    "Cadmium (µg/L)": _p("Cadmium", r"$\mu$g/L", SAMPLES),
    # mg/L, unlike every other metal here.  A real difference, not a typo.
    "Copper filtered (mg/L)": _p("Copper, filtered", "mg/L", SAMPLES),
    "Chromium (µg/L)": _p("Chromium", r"$\mu$g/L", SAMPLES),
    "Nickel (µg/L)": _p("Nickel", r"$\mu$g/L", SAMPLES),
    "Zinc (µg/L)": _p("Zinc", r"$\mu$g/L", SAMPLES),
}

PARAMS: dict[str, Param] = {**_SURFACE, **_SCADA, **_WEATHER, **_SAMPLES}


# --- Target representation suffixes -----------------------------------------------
# A target column may appear as the previous measured value (_state), the change since
# that value (_diff), or the residual from a reference model (_res).  These change what
# the number means, so they are rendered as words rather than passed through raw.
_SUFFIX_TEMPLATES = {
    "_state": "{}, previous value",
    "_diff": "Δ{}",
    "_res": "{}, residual",
}

# Suffixes that change the quantity enough that the base unit no longer applies as-is.
# A change and a residual are still in the base unit; a state value certainly is.
_SUFFIX_KEEPS_UNIT = {"_state": True, "_diff": True, "_res": True}


def _split_suffix(col: str) -> tuple[str, str | None]:
    for sfx in _SUFFIX_TEMPLATES:
        if col.endswith(sfx):
            return col[: -len(sfx)], sfx
    return col, None


def slug(col: str) -> str:
    """Filesystem slug for a column, matching ``d_RunResample.py``.

    Kept here so the reverse lookup in :func:`clean_target_label` cannot drift from the
    rule that produced the directory names on disk.
    """
    return _re.sub(r"[^\w]", "_", col)


# slug -> stored column, built forwards from the same rule that created the directories.
_SLUG_TO_COL: dict[str, str] = {slug(c): c for c in PARAMS}


def _lookup(col: str) -> tuple[Param | None, str | None]:
    base, sfx = _split_suffix(str(col))
    param = PARAMS.get(base)
    if param is None:
        param = PARAMS.get(_SLUG_TO_COL.get(slug(base), ""))
    return param, sfx


def unit(col: str) -> str | None:
    """Unit string for a column, or ``None`` if the quantity is dimensionless."""
    param, sfx = _lookup(col)
    if param is None:
        return None
    if sfx is not None and not _SUFFIX_KEEPS_UNIT.get(sfx, True):
        return None
    return param.unit


def label(
    col: str,
    *,
    with_unit: bool = True,
    long: bool = False,
    qualified: bool = True,
    with_suffix: bool = True,
) -> str:
    """Display label for a stored column name.

    ``with_unit=False`` when the figure plots a statistic *about* a model of this
    parameter rather than values of the parameter itself -- an R^2, an importance
    z-score, a 0/1 feature-inclusion flag.  Attaching a concentration unit to such a
    number states something false.

    ``qualified=False`` when every entry in the figure comes from the same source, so
    that the ``(Surface)`` / ``(lab)`` disambiguator would be noise rather than
    information.

    ``with_suffix=False`` when every entry in the figure shares the same representation
    (all residuals, all differences), so that repeating it on each tick adds nothing.

    Unknown columns are returned unchanged rather than mangled, so a gap in the registry
    is visible in the output instead of silently producing a wrong label.
    """
    param, sfx = _lookup(col)
    if param is None:
        return str(col)

    text = param.long if long else param.short
    if qualified and param.qualifier:
        text = f"{text} ({param.qualifier})"
    if with_suffix and sfx is not None:
        text = _SUFFIX_TEMPLATES[sfx].format(text)

    if with_unit or param.unit_is_identity:
        u = unit(col)
        if u:
            text = f"{text} ({u})"
    return text


def axis_label(col: str, *, with_unit: bool = True, qualified: bool = True) -> str:
    """Long-form label, for axis titles."""
    return label(col, with_unit=with_unit, long=True, qualified=qualified)


def clean_target_label(
    dataset_name: str,
    prefix: str = "MC",
    *,
    with_suffix: bool = False,
    qualified: bool = False,
) -> str:
    """Display label for a per-target output directory name.

    Directory names are produced by ``d_RunResample.py`` as
    ``{prefix}_{slug(target)}``, optionally with an ``ex`` marker, e.g.
    ``MC_Colony_Count_22_C__CFU_mL__res``.  This resolves them by applying the same slug
    rule *forwards* to the registry and matching, rather than by stripping characters out
    of the directory name.

    The previous implementation deleted all digits and the characters ``°µ/``, which
    turned ``Colony Count 22°C (CFU/mL)`` into ``Colony Count`` -- discarding the
    incubation temperature that distinguishes it from ``Total coliforms 37°C``.

    Labels are returned without units, because this function names a *target* in
    model-performance figures, where the plotted quantity is a metric rather than a
    concentration.  For the same reason the representation suffix and the source
    qualifier are off by default: every target in such a figure shares them, so they
    would repeat on every tick without distinguishing anything.
    """
    name = str(dataset_name)
    bare = prefix.rstrip("_")
    if bare and name.startswith(bare + "_"):
        name = name[len(bare) + 1:]
    if name.startswith("ex"):
        name = name[2:]

    # Directory names encode the suffix through the slug rule, so '_res' survives intact
    # but '(µg/L)' has become '__µg_L_'.  Strip the representation suffix first.
    sfx = None
    for candidate in _SUFFIX_TEMPLATES:
        if name.endswith(candidate):
            name, sfx = name[: -len(candidate)], candidate
            break

    col = _SLUG_TO_COL.get(name)
    if col is None:
        # Fall back to a slug comparison that ignores runs of separator underscores,
        # which differ between '(µg/L)' -> '__µg_L_' and hand-written directory names.
        squashed = _re.sub(r"_+", "_", name).strip("_")
        for cand_slug, cand_col in _SLUG_TO_COL.items():
            if _re.sub(r"_+", "_", cand_slug).strip("_") == squashed:
                col = cand_col
                break
    if col is None:
        return _re.sub(r"_+", " ", name).strip()

    param = PARAMS[col]
    text = param.short
    if qualified and param.qualifier:
        text = f"{text} ({param.qualifier})"
    if with_suffix and sfx is not None:
        text = _SUFFIX_TEMPLATES[sfx].format(text)
    return text

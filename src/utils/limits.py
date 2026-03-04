import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def _as_float(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def normalize_limit_name(name: str) -> str:
    text = "" if name is None else str(name)
    replacements = {
        "Ã‚Â°C": "C",
        "Â°C": "C",
        "°C": "C",
        "ºC": "C",
        "Ã‚Âµ": "u",
        "Âµ": "u",
        "µ": "u",
        "μ": "u",
        "Ã¸": "o",
        "ø": "o",
        "Ã…": "A",
        "å": "a",
        "Ã†": "Ae",
        "æ": "ae",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^a-z0-9]+", "", text)


def _name_variants(name: str) -> set:
    raw = "" if name is None else str(name).strip()
    if not raw:
        return set()
    variants = {normalize_limit_name(raw)}
    no_paren = re.sub(r"\s*\([^)]*\)", "", raw).strip()
    if no_paren:
        variants.add(normalize_limit_name(no_paren))
    return {v for v in variants if v}


def load_limits_records(limits_csv: Path) -> list:
    limits_csv = Path(limits_csv)
    if not limits_csv.exists():
        return []

    try:
        df = pd.read_csv(limits_csv, low_memory=False)
    except Exception:
        df = pd.read_csv(limits_csv, sep=";", decimal=".", low_memory=False)
    if len(df.columns) == 1:
        try:
            df_alt = pd.read_csv(limits_csv, sep=";", decimal=".", low_memory=False)
            if len(df_alt.columns) > len(df.columns):
                df = df_alt
        except Exception:
            pass
    if df.empty:
        return []

    cols_norm = {normalize_limit_name(c): c for c in df.columns}
    key_param_nor = cols_norm.get(normalize_limit_name("Param_nor"))
    key_param_eng = cols_norm.get(normalize_limit_name("Param_eng"))
    key_upper = cols_norm.get(normalize_limit_name("Upper"))
    key_lower = cols_norm.get(normalize_limit_name("Lower"))

    records = []
    if key_param_nor or key_param_eng:
        for _, row in df.iterrows():
            original_name = row.get(key_param_nor) if key_param_nor else None
            translated_name = row.get(key_param_eng) if key_param_eng else None
            names = []
            if pd.notna(original_name):
                names.append(str(original_name).strip())
            if pd.notna(translated_name):
                names.append(str(translated_name).strip())
            names = [n for n in names if n]
            if not names:
                continue
            records.append(
                {
                    "names": names,
                    "original_name": str(original_name).strip() if pd.notna(original_name) else None,
                    "translated_name": str(translated_name).strip() if pd.notna(translated_name) else None,
                    "upper": _as_float(row.get(key_upper)) if key_upper else None,
                    "lower": _as_float(row.get(key_lower)) if key_lower else None,
                }
            )
        return records

    # Legacy fallback: one-row wide table, upper thresholds only.
    first_row = df.iloc[0]
    for col in df.columns:
        records.append(
            {
                "names": [str(col)],
                "original_name": str(col),
                "translated_name": None,
                "upper": _as_float(first_row[col]),
                "lower": None,
            }
        )
    return records


def map_limits_to_columns(columns: list, limits_records: list) -> dict:
    if not columns or not limits_records:
        return {}

    indexed = {}
    for rec in limits_records:
        upper = rec.get("upper")
        lower = rec.get("lower")
        if upper is None and lower is None:
            continue
        for name in rec.get("names", []):
            for key in _name_variants(name):
                if key not in indexed:
                    indexed[key] = {
                        "upper": upper,
                        "lower": lower,
                        "record": rec,
                    }

    out = {}
    for col in columns:
        for key in _name_variants(col):
            if key in indexed:
                out[col] = {
                    "upper": indexed[key]["upper"],
                    "lower": indexed[key]["lower"],
                    "record": indexed[key]["record"],
                }
                break
    return out


def limit_exceedance_mask(values, upper=None, lower=None):
    arr = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    mask = np.zeros(arr.shape, dtype=bool)
    valid = np.isfinite(arr)
    if upper is not None:
        upper_val = float(upper)
        if np.isclose(upper_val, 0.0):
            mask |= valid & (arr > upper_val)
        else:
            mask |= valid & (arr >= upper_val)
    if lower is not None:
        lower_val = float(lower)
        if np.isclose(lower_val, 0.0):
            mask |= valid & (arr < lower_val)
        else:
            mask |= valid & (arr <= lower_val)
    return mask

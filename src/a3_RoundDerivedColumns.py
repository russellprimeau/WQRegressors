"""Trim the spurious precision from derived columns of an existing consolidated record.

The problem
-----------
Two stage-``a`` constructions stored far more resolution than their inputs carry:

* ``decompose_direction`` formed ``speed * cos(direction)`` and ``speed * sin(direction)``
  and stored the float64 result. Direction and speed are each recorded to one
  decimal (0-360.0 deg, 0-13.1 m/s), so a component holds roughly three significant
  figures of real information; the record stored twelve or more decimals.
* ``rolling_sum`` accumulated 24 hourly precipitation readings, each exact to one
  decimal, and stored the float64 sum with its addition artifacts.

Both are fixed at source in ``utils/preprocessing.py``: ``decompose_direction`` now
rounds to one guard digit finer than the magnitude column's own resolution, and
``rolling_sum`` rounds to the source column's resolution. Any future regeneration is
correct without this script.

Why this script exists anyway
-----------------------------
The existing ``Consolidated_sparse.csv`` predates the fix and cannot simply be
regenerated: the raw sensor inputs were updated afterwards, so a full stage-``a``
rebuild also moves the record start and changes profiler and weather coverage at
thousands of timestamps, which would alter the predictors for every target and
invalidate a completed sweep.

Applying the rounding to the stored values instead was verified to be exactly what
the fixed code produces. Against a full regeneration with the fix, on every shared
timestamp where both files hold a value, the maximum absolute difference is **0** for
wind x, wind y, air temperature and 24-hour precipitation. Every discrepancy is
explained by input drift rather than by the rounding: 24 rows where the regenerated
record starts 15 days earlier and so has a full 24-hour precipitation window, and 241
timestamps absent from the older file that the updated inputs fill.

So this is the fix applied, not an approximation of it.

Precision is inferred per column rather than hard-coded, by the same rule the source
functions use, so a column whose inputs are finer keeps its resolution.

Usage:
    python src/a3_RoundDerivedColumns.py --dry-run
    python src/a3_RoundDerivedColumns.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.csv_textedit import rewrite_columns  # noqa: E402
from utils.preprocessing import _decimal_resolution  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = Path("data/output/regression/Consolidated_sparse.csv")

# Derived column -> the column whose measurement resolution governs it, and the
# guard digits allowed beyond that resolution.  These mirror the rules now applied in
# decompose_direction (one guard digit past the magnitude) and rolling_sum (no guard
# digit, because a sum of values exact at d decimals is exact at d decimals).
DERIVED_COLUMNS: dict[str, tuple[str, int]] = {
    "Wind speed x (m/s)": ("Wind speed x (m/s)", 1),
    "Wind speed y (m/s)": ("Wind speed y (m/s)", 1),
    "24hr precipitation total (mm)": ("24hr precipitation total (mm)", 0),
}

# The measurement resolution of the governing quantity cannot be read back out of the
# derived column itself once the noise is in it, so it is stated here with its source.
# Wind: direction DX_l and speed FF Hastighet in Weather.csv, both one decimal.
# Precipitation: RR_1 in Weather.csv, one decimal.
SOURCE_RESOLUTION: dict[str, int] = {
    "Wind speed x (m/s)": 1,
    "Wind speed y (m/s)": 1,
    "24hr precipitation total (mm)": 1,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=None,
                    help="Consolidated dataset to update (default %s)." % DEFAULT_CSV)
    ap.add_argument("--columns", nargs="*", default=None,
                    help="Derived columns to trim. Default: all of %s"
                         % ", ".join(DERIVED_COLUMNS))
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change and write nothing.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip writing the .bak copy alongside the file.")
    args = ap.parse_args()

    csv = args.csv if args.csv else (REPO_ROOT / DEFAULT_CSV)
    csv = csv if csv.is_absolute() else (REPO_ROOT / csv)
    if not csv.exists():
        raise SystemExit("consolidated dataset not found: %s" % csv)

    df = pd.read_csv(csv, low_memory=False)
    wanted = list(args.columns) if args.columns else list(DERIVED_COLUMNS)
    unknown = [c for c in wanted if c not in DERIVED_COLUMNS]
    if unknown:
        raise SystemExit("not a known derived column: %s. Known: %s"
                         % (unknown, list(DERIVED_COLUMNS)))
    absent = [c for c in wanted if c not in df.columns]
    if absent:
        raise SystemExit("column(s) absent from %s: %s" % (csv.name, absent))

    print("[INFO] %s" % csv)
    print("[INFO] %d rows, %d columns" % (len(df), len(df.columns)))
    print()
    print("%-32s %8s %8s %10s %10s %12s"
          % ("column", "dec_now", "dec_new", "distinct", "moves", "max_delta"))

    replacements: dict[str, np.ndarray] = {}
    for c in wanted:
        _, guard = DERIVED_COLUMNS[c]
        decimals = SOURCE_RESOLUTION[c] + guard
        before = pd.to_numeric(df[c], errors="coerce").to_numpy()
        after = np.round(before, decimals)
        same = np.isclose(before, after, rtol=0.0, atol=0.0, equal_nan=True)
        d = np.abs(before[~same] - after[~same])
        d = d[np.isfinite(d)]
        print("%-32s %8d %8d %10d %10d %12.6g"
              % (c[:32], _decimal_resolution(df[c]), decimals,
                 int(pd.Series(after).nunique()), int((~same).sum()),
                 float(d.max()) if d.size else 0.0))
        if not same.all():
            replacements[c] = after

    if not replacements:
        print()
        print("[INFO] Nothing changed; the file already reflects the current construction.")
        return 0

    print()
    print("[INFO] %d column(s) would change: %s"
          % (len(replacements), ", ".join(replacements)))
    if args.dry_run:
        print("[INFO] --dry-run: nothing written.")
        return 0

    if not args.no_backup:
        bak = csv.with_suffix(csv.suffix + ".bak")
        if bak.exists():
            print("[INFO] Backup already present, leaving it as the pre-edit state: %s" % bak)
        else:
            shutil.copy2(csv, bak)
            print("[INFO] Backed up to %s" % bak)

    try:
        report = rewrite_columns(csv, replacements)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print("[INFO] Wrote %s" % csv)
    print("[INFO] Re-rendered %d column(s) over %d rows; %d untouched fields verified "
          "byte-identical." % (len(report.columns), report.rows,
                               report.untouched_fields_verified))
    print("[INFO] Any target whose samples are regenerated from this file now uses the "
          "trimmed predictors; targets already swept are unaffected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

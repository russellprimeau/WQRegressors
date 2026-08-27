"""Re-apply the fixed target-difference construction to the consolidated dataset.

Why this exists
---------------
The ``_diff`` and ``_res`` target columns are built by ``add_diff`` / ``add_res``
in stage ``a`` (``a_ConsolidateDatasets.py``), which writes
``Consolidated_sparse.csv``. Both functions used to round their output to a fixed
three decimals; they now round to the source column's own decimal resolution,
because a fixed precision quantizes any target reported in a coarse unit.
Filtered copper in mg/L spans 0.004 in total, so three decimals reduced its
target to five levels with 75 percent of them exactly zero.

The fix lives in stage ``a``, but re-running stage ``a`` is not the same as
re-running the fix: the raw sensor inputs (``FullHourly.csv``, ``Weather.csv``,
``Profiles.csv``) were updated after the consolidated file was last built, so a
full regeneration also moves the record start and changes profiler and weather
coverage at thousands of timestamps. That would alter the predictors for every
target and invalidate a completed sweep.

This script re-executes only the target-difference step, on the existing
consolidated file, using the same ``add_diff`` / ``add_res`` functions stage ``a``
calls. Verified equivalent: on the 2026-04-21 consolidated file the columns it
produces are bit-identical to those from a full stage-``a`` regeneration (48908 of
48908 rows, maximum absolute difference 0.0), while every other column is left
untouched.

It is idempotent, and safe to run over every target: a target whose unit is fine
at three decimals does not move. The report names the columns that changed, so a
target moving unexpectedly is visible rather than silent, and the script refuses
to write if anything outside the target set moves.

Why the write is a text substitution
------------------------------------
Rewriting the file through ``DataFrame.to_csv`` perturbs the last bit of floats it
was never asked to touch: the existing file carries 17 significant digits, the
CSV writer emits fewer, and the two are not always the same double. Measured on
this file, that moved 472 wind-speed values by one to two units in the last place
-- numerically irrelevant, but a silent edit to columns outside the requested
scope, and enough to make "nothing else changed" untrue as stated.

So the write goes through ``utils.csv_textedit.rewrite_columns``, which copies the
untouched fields as text, byte for byte, re-renders only the requested target
columns via ``repr`` (which round-trips a float64 exactly), and then verifies the
untouched columns as raw text rather than as parsed numbers -- the strongest check
available, since it catches encoding drift and float reformatting alike.

That writer requires the file to have no quoted fields and a uniform field count;
both are asserted before anything is written.

Downstream: the affected target's samples must be regenerated with
``d_RunResample.py`` and its sweep re-run. A target's ``_diff`` column appears
only in its own sample files, so no other target is affected.

Usage:
    python src/a2_RefreshTargetDiffs.py --dry-run
    python src/a2_RefreshTargetDiffs.py --targets "Copper filtered (mg/L)"
    python src/a2_RefreshTargetDiffs.py
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
from utils.preprocessing import _decimal_resolution, add_diff, add_res  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = Path("data/output/regression/Consolidated_sparse.csv")


def _base_targets(df: pd.DataFrame) -> list[str]:
    """Base target names that already carry a _diff or _res column."""
    out: list[str] = []
    for c in df.columns:
        for suf in ("_diff", "_res"):
            if c.endswith(suf):
                base = c[: -len(suf)]
                if base in df.columns and base not in out:
                    out.append(base)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=None,
                    help="Consolidated dataset to update (default %s)." % DEFAULT_CSV)
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Base target column names. Default: every target that already "
                         "has a _diff or _res column.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change and write nothing.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip writing the .bak copy alongside the file.")
    args = ap.parse_args()

    csv = args.csv if args.csv else (REPO_ROOT / DEFAULT_CSV)
    csv = csv if csv.is_absolute() else (REPO_ROOT / csv)
    if not csv.exists():
        raise SystemExit("consolidated dataset not found: %s" % csv)

    original = pd.read_csv(csv, low_memory=False)
    targets = list(args.targets) if args.targets else _base_targets(original)
    missing = [t for t in targets if t not in original.columns]
    if missing:
        raise SystemExit("target column(s) absent from %s: %s" % (csv.name, missing))
    print("[INFO] %s" % csv)
    print("[INFO] %d rows, %d columns, %d target(s) to refresh"
          % (len(original), len(original.columns), len(targets)))

    updated = add_res(original.copy(), targets)
    updated = add_diff(updated, targets)

    if list(updated.columns) != list(original.columns):
        raise SystemExit("column set changed; refusing to write. This script must only "
                         "recompute existing columns.")

    changed: list[tuple[str, int, float]] = []
    for c in original.columns:
        a = pd.to_numeric(original[c], errors="coerce").to_numpy()
        b = pd.to_numeric(updated[c], errors="coerce").to_numpy()
        if a.dtype.kind not in "fc" or b.dtype.kind not in "fc":
            continue
        same = np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True)
        if not same.all():
            delta = np.abs(a[~same] - b[~same])
            worst = float(np.nanmax(delta)) if np.isfinite(delta).any() else float("nan")
            changed.append((c, int((~same).sum()), worst))

    touched_all = {"%s%s" % (t, s) for t in targets for s in ("_diff", "_res")}
    stray = [c for c, _, _ in changed if c not in touched_all]
    if stray:
        raise SystemExit("columns outside the target set changed: %s. Refusing to write; "
                         "investigate before proceeding." % stray)

    print()
    print("%-46s %10s %11s %11s %9s %10s"
          % ("column", "resolution", "rows_moved", "max_delta", "distinct", "frac_zero"))
    for t in targets:
        res = _decimal_resolution(original[t])
        for suf in ("_diff", "_res"):
            c = "%s%s" % (t, suf)
            if c not in original.columns:
                continue
            k, m = next(((k, m) for cc, k, m in changed if cc == c), (0, 0.0))
            s = pd.to_numeric(updated[c], errors="coerce").dropna()
            print("%-46s %10d %11d %11.6g %9d %10.2f"
                  % (c[:46], res, k, m, s.nunique(), float(np.mean(s == 0))))

    if not changed:
        print()
        print("[INFO] Nothing changed; the file already reflects the current construction.")
        return 0

    print()
    print("[INFO] %d column(s) would change: %s"
          % (len(changed), ", ".join(c for c, _, _ in changed)))
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

    # Only the columns actually recomputed are re-rendered, so a target that did
    # not move keeps its original text as well.
    touched = [c for c, _, _ in changed]
    try:
        report = rewrite_columns(
            csv, {c: pd.to_numeric(updated[c], errors="coerce").to_numpy() for c in touched})
    except ValueError as exc:
        raise SystemExit(str(exc))
    print("[INFO] Wrote %s" % csv)
    print("[INFO] Re-rendered %d column(s) over %d rows; %d untouched fields verified "
          "byte-identical." % (len(report.columns), report.rows,
                               report.untouched_fields_verified))
    print("[INFO] Regenerate the affected target's samples with d_RunResample.py and "
          "re-run its sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

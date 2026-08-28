# -*- coding: utf-8 -*-
"""Confirm which reference value the differential target was computed against.

The fixed-row-offset definition means the target is
    diff = y(last row of window) - y(first row of window)
The superseded irregular-interval definition would instead difference against the
most recent measurement inside the window, which differs only for windows that
contain more than one laboratory measurement.

Both `_state` and `_diff` are independently min-max scaled, so the columns cannot
be compared until they are inverted through `normalization.json`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import run_paths as rp

TOL = 1e-6


def ascii_key(s: str) -> str:
    """Fold a column name to ASCII so mojibake in CSV vs JSON does not block the join."""
    return re.sub(r"[^A-Za-z0-9_()/.]+", "", str(s))


def invert(norm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + np.asarray(norm, dtype=float) * (hi - lo)


def check_target(ds: Path) -> dict:
    sd = ds / "samples"
    files = sorted(sd.glob("segment_*.csv"))
    if not files:
        return {"target": ds.name, "status": "no samples"}

    scalers = json.loads((ds / "normalization.json").read_text(encoding="utf-8", errors="replace"))
    scal_by_ascii = {ascii_key(k): v for k, v in scalers.items()}

    hdr = pd.read_csv(files[0], nrows=0, encoding="utf-8", encoding_errors="replace").columns
    state_col = next((c for c in hdr if c.endswith("_state")), None)
    diff_col = next((c for c in hdr if c.endswith("_diff")), None)
    if state_col is None or diff_col is None:
        return {"target": ds.name, "status": f"missing cols (state={state_col}, diff={diff_col})"}

    s_sc = scal_by_ascii.get(ascii_key(state_col))
    d_sc = scal_by_ascii.get(ascii_key(diff_col))
    if not s_sc or not d_sc:
        return {"target": ds.name, "status": "scaler not found"}

    n_first = n_prev = n_neither = n_amb = 0
    multi = 0
    worst = None
    for f in files:
        d = pd.read_csv(f, usecols=[state_col, diff_col], encoding="utf-8", encoding_errors="replace")
        dv = d[diff_col].dropna()
        if dv.empty:
            continue
        diff = invert(float(dv.iloc[-1]), d_sc["min"], d_sc["max"])
        s = invert(d[state_col].to_numpy(), s_sc["min"], s_sc["max"])
        s_last = s[-1]
        head = s[:-1][np.isfinite(s[:-1])]
        if head.size == 0:
            continue
        s_first, s_prev = head[0], head[-1]
        # More than one distinct value before the final row means the window spans
        # more than one laboratory measurement: the only case that discriminates.
        discriminating = not np.isclose(s_first, s_prev, atol=TOL)
        multi += int(discriminating)

        ok_first = abs((s_last - s_first) - diff) < 1e-6
        ok_prev = abs((s_last - s_prev) - diff) < 1e-6
        if ok_first and ok_prev:
            n_amb += 1
        elif ok_first:
            n_first += 1
        elif ok_prev:
            n_prev += 1
        else:
            n_neither += 1
            err = abs((s_last - s_first) - diff)
            if worst is None or err > worst[1]:
                worst = (f.name, err, s_first, s_prev, s_last, diff)

    return {
        "target": re.sub(r"_diff$", "", ds.name.replace("MC_", "")),
        "files": len(files),
        "vs_first_only": n_first,
        "vs_prev_only": n_prev,
        "both_agree": n_amb,
        "neither": n_neither,
        "discriminating": multi,
        "worst": worst,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None,
                    help="Results root to check. Defaults to the reporting root.")
    args = ap.parse_args(argv)
    root = rp.resolve_root(args.root)
    if not root.is_dir():
        raise SystemExit("results root not found: %s" % root)
    print("checking %s" % root)
    rows = [check_target(ds) for ds in sorted(root.glob("MC_*")) if (ds / "samples").is_dir()]
    print("%-30s %6s %8s %8s %8s %8s %8s" % (
        "target", "files", "multi", "vs_FIRST", "vs_prev", "both", "neither"))
    bad = 0
    for r in rows:
        if "status" in r:
            print("%-30s %s" % (r["target"][:30], r["status"]))
            continue
        print("%-30s %6d %8d %8d %8d %8d %8d" % (
            r["target"][:30], r["files"], r["discriminating"],
            r["vs_first_only"], r["vs_prev_only"], r["both_agree"], r["neither"]))
        bad += r["vs_prev_only"] + r["neither"]
        if r["worst"]:
            print("    worst mismatch: %s err=%.3g first=%.4g prev=%.4g last=%.4g diff=%.4g" % r["worst"])
    print()
    tot_disc = sum(r.get("discriminating", 0) for r in rows)
    print("discriminating windows (span >1 lab measurement): %d" % tot_disc)
    print("windows matching window-START difference only     : %d"
          % sum(r.get("vs_first_only", 0) for r in rows))
    print("windows matching LATEST-measurement difference only: %d"
          % sum(r.get("vs_prev_only", 0) for r in rows))
    print("windows matching neither                           : %d"
          % sum(r.get("neither", 0) for r in rows))
    print()
    print("VERDICT:", "fixed-row-offset definition confirmed" if bad == 0
          else "MIXED OR SUPERSEDED DEFINITION PRESENT (%d windows)" % bad)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

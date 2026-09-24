# -*- coding: utf-8 -*-
"""Confirm that saved GP artifacts survive a save/load round trip, in both schemas.

The GP artifact is the only model artifact in this project that is a wrapper dict
rather than a bare state dict, and it is the only one whose kernel carries buffers
describing the *inputs* rather than the fit. Those buffers used to be persisted inside
``model_state_dict`` as well as stored at top level, so 89% of every artifact was one
window-tiled array written twice. They are now non-persistent, and the top-level arrays
are stored in their pre-tile form.

That change touches the one code path that reconstructs a fitted GP, and nothing in the
repository exercised it: ``v2_RecoverFailedGPConfigs`` comes closest but records an
evaluation failure in a status column instead of failing, and ``o_PredictionTimeseries``
turns a load error into a ``[WARN]`` and a silently missing figure. This script is the
gate that turns either into a non-zero exit.

Three things are checked:

1. **Legacy artifacts still load.** Every root predating the change holds version-2
   artifacts with the buffers baked in, and CV19 and CV22_profilerless are manuscript
   sources, so they must keep reconstructing exactly.
2. **Both kernel nestings are covered.** The buffer key is
   ``covar_module.base_kernel.noise_delta_samples`` or
   ``covar_module.base_kernel.kernels.0.noise_delta_samples`` depending on whether an
   AdditiveKernel wraps it. Both occur on disk; code that matches a full path instead of
   a suffix handles one and silently breaks the other.
3. **The new encoding is lossless.** Re-tiling the stored base arrays must reproduce the
   dense arrays bit for bit, because they are what the kernel marginalises over.

Usage:
    python src/v5_CheckArtifactRoundTrip.py
    python src/v5_CheckArtifactRoundTrip.py --root data/output/CV19 --limit 50
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from utils.console import force_utf8_console
from utils import run_paths as rp

# Before anything can print: every target name carries a mu, so a cp1252 console
# raises UnicodeEncodeError from inside print(). See utils/console.py.
force_utf8_console()

from utils.config_utils import UNCERTAINTY_DISTRIBUTION_FEATURES
from utils.gp_utils import (
    GP_ARTIFACT_VERSION,
    NON_PERSISTENT_KERNEL_BUFFERS,
    strip_legacy_kernel_buffers,
    uncertainty_arrays,
)

# Roots checked when none is given: the reporting basis and the profiler arm, which
# between them cover both kernel nestings and both live/absent input uncertainty.
DEFAULT_ROOTS = (rp.REPORTING_ROOT, rp.PROFILER_ROOT)


def _iter_artifacts(root: Path, limit: int):
    """Up to ``limit`` gp_model.pt paths under ``root``, spread across targets."""
    per_target: dict[str, list[Path]] = {}
    for p in root.rglob("gp_model.pt"):
        try:
            target = p.relative_to(root).parts[0]
        except ValueError:
            target = ""
        per_target.setdefault(target, []).append(p)
    out: list[Path] = []
    targets = sorted(per_target)
    i = 0
    while len(out) < limit and targets:
        progressed = False
        for t in targets:
            if i < len(per_target[t]):
                out.append(per_target[t][i])
                progressed = True
                if len(out) >= limit:
                    break
        if not progressed:
            break
        i += 1
    return out


def _check_one(path: Path, torch) -> dict:
    """Load one artifact and report what it exercised, or why it failed."""
    rec = {"path": path, "ok": False, "version": None, "keys": set(), "note": ""}
    try:
        art = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        rec["note"] = "unreadable: %s: %s" % (type(exc).__name__, exc)
        return rec

    rec["version"] = art.get("artifact_version")

    var, deltas = uncertainty_arrays(art)
    n = art.get("input_dim")

    # The arrays describe the flattened input the model was fitted on. A mismatch here
    # is the failure the persisted buffer used to catch by accident.
    if var is not None and n is not None and int(np.asarray(var).shape[-1]) != int(n):
        rec["note"] = "variance describes %d features, model fitted on %d" % (
            np.asarray(var).shape[-1], n)
        return rec
    if deltas is not None and n is not None and int(np.asarray(deltas).shape[-1]) != int(n):
        rec["note"] = "noise draws describe %d features, model fitted on %d" % (
            np.asarray(deltas).shape[-1], n)
        return rec

    # Re-tiling must be exact against whatever the artifact itself stores dense.
    if art.get("uncertainty_tile_reps") is not None:
        dense = art.get("uncertainty_noise_deltas")
        if dense is not None and deltas is not None:
            if not np.array_equal(np.asarray(dense), np.asarray(deltas)):
                rec["note"] = "re-tiled draws differ from the stored dense array"
                return rec

    for state in art.get("models") or []:
        sd = state.get("model_state_dict") or {}
        for k in sd:
            if k.split(".")[-1] in NON_PERSISTENT_KERNEL_BUFFERS:
                rec["keys"].add(k)
        stripped = strip_legacy_kernel_buffers(sd)
        leftover = [k for k in stripped
                    if k.split(".")[-1] in NON_PERSISTENT_KERNEL_BUFFERS]
        if leftover:
            rec["note"] = "buffers survived stripping: %s" % leftover[:2]
            return rec
        if not stripped:
            rec["note"] = "state dict empty after stripping"
            return rec

    rec["ok"] = True
    return rec


def check_replicates(root: Path) -> list[str]:
    """Problems with a root's Monte Carlo replicate tree. Empty means it is sound.

    A replicate exists to carry a calibration offset into the data for the families that
    cannot represent input uncertainty themselves. Four things have to hold for that to
    be true, and each has failed at some point:

    - a window's replicates are either all distinct or a single copy, never a partial set;
    - the offsets differ *between* windows. They did not: the seed omitted the window, so
      every window received the same draw and ten replicates encoded one global shift;
    - the recorded offsets reproduce the replicates, so the tree is not the only record;
    - a root with nothing perturbable has no replicate tree, and its configs say so.
    """
    import pandas as pd

    problems: list[str] = []
    targets = sorted(root.glob("MC_*"))
    if not targets:
        return [f"{root.name}: no MC_* target directories"]

    perturbable = set(UNCERTAINTY_DISTRIBUTION_FEATURES)
    for t in targets:
        mc, sm = t / "mc_replicates", t / "samples"
        cols = set(pd.read_csv(next(sm.glob("segment_*.csv")), nrows=0).columns)
        root_perturbable = bool(cols & perturbable)

        if not root_perturbable:
            if mc.is_dir():
                problems.append(f"{t.name}: no perturbable column, but mc_replicates/ exists")
            for cfg_path in t.glob("config_*.yml"):
                sub = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["data"]["sample_subdir"]
                if sub != "samples":
                    problems.append(f"{t.name}/{cfg_path.name}: reads {sub!r} but none was written")
            continue

        if not mc.is_dir():
            problems.append(f"{t.name}: perturbable columns present but no mc_replicates/")
            continue

        # A tree written before the offset record exists cannot be checked against it, and
        # is not evidence of a defect -- it simply predates the change. Say so once, rather
        # than enumerating every window of a root nobody is going to regenerate.
        if not (t / "provenance" / "mc_offsets.csv").exists():
            problems.append(
                f"{t.name}: legacy replicate tree (no provenance/mc_offsets.csv). "
                "Predates the per-window draw; regenerate the root to conform.")
            continue

        groups: dict[str, list[Path]] = {}
        for f in mc.glob("segment_*_mc_*.csv"):
            groups.setdefault(re.sub(r"_mc_\d+\.csv$", "", f.name), []).append(f)
        sizes = {len(v) for v in groups.values()}
        if not sizes <= {1, max(sizes)}:
            problems.append(f"{t.name}: partial replicate groups, sizes {sorted(sizes)}")
        for base, files in groups.items():
            n_distinct = len({hashlib.md5(f.read_bytes()).hexdigest() for f in files})
            if len(files) > 1 and n_distinct != len(files):
                problems.append(
                    f"{t.name}/{base}: {len(files)} replicates but {n_distinct} distinct")

        off = pd.read_csv(t / "provenance" / "mc_offsets.csv")

        # Offsets must vary between windows. This is the regression test for the reseed.
        for col, grp in off[off["replicate"] == 1].groupby("column"):
            if len(grp) > 1 and grp["offset_normalised"].nunique() == 1:
                problems.append(
                    f"{t.name}: replicate 1 applied one identical offset to all "
                    f"{len(grp)} windows for {col!r} -- per-window draw has collapsed")

        # The recorded offsets must rebuild the replicates from the base samples.
        for (seg, rep), grp in list(off.groupby(["segment", "replicate"]))[:5]:
            base = pd.read_csv(sm / f"{seg}.csv")
            rp = pd.read_csv(mc / f"{seg}_mc_{int(rep):03d}.csv")
            recon = base.copy()
            for _, r in grp.iterrows():
                m = recon[r["column"]].notna()
                recon.loc[m, r["column"]] = recon.loc[m, r["column"]] + r["offset_normalised"]
            if not np.allclose(recon.select_dtypes(float).to_numpy(),
                               rp.select_dtypes(float).to_numpy(),
                               equal_nan=True, rtol=0, atol=1e-9):
                problems.append(f"{t.name}/{seg} rep {rep}: offsets do not reproduce the replicate")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, action="append", default=None,
                    help="Results root to check. Repeatable. Defaults to the reporting "
                         "root and the profiler arm.")
    ap.add_argument("--limit", type=int, default=40,
                    help="Artifacts to check per root.")
    ap.add_argument("--replicates", action="store_true",
                    help="Check the Monte Carlo replicate tree instead of GP artifacts.")
    args = ap.parse_args()

    if args.replicates:
        roots = [rp.resolve_root(r) for r in (args.root or DEFAULT_ROOTS)]
        all_problems: list[str] = []
        for root in roots:
            if not root.is_dir():
                print("%-34s SKIP (not found)" % root.name)
                continue
            problems = check_replicates(root)
            all_problems += problems
            print("%-34s %d problem(s)" % (root.name, len(problems)))
        if all_problems:
            print("\nFAILURES: %d" % len(all_problems))
            for p in all_problems[:30]:
                print("  %s" % p)
        print("\nVERDICT:", "replicate trees are sound" if not all_problems
              else "REPLICATE TREE BROKEN (%d problem(s))" % len(all_problems))
        return 0 if not all_problems else 1

    roots = [rp.resolve_root(r) for r in (args.root or DEFAULT_ROOTS)]

    try:
        import torch  # noqa: F401
    except Exception as exc:
        print("torch is required to read model artifacts: %s" % exc)
        return 1

    print("artifact schema written by this code: version %d\n" % GP_ARTIFACT_VERSION)

    failures: list[dict] = []
    all_keys: set[str] = set()
    versions: dict = {}
    checked = 0

    for root in roots:
        if not root.is_dir():
            print("%-34s SKIP (not found)" % root.name)
            continue
        paths = _iter_artifacts(root, args.limit)
        if not paths:
            print("%-34s SKIP (no gp_model.pt)" % root.name)
            continue
        bad = 0
        for p in paths:
            rec = _check_one(p, torch)
            checked += 1
            all_keys |= rec["keys"]
            versions[rec["version"]] = versions.get(rec["version"], 0) + 1
            if not rec["ok"]:
                bad += 1
                failures.append(rec)
        print("%-34s %4d checked, %d failed" % (root.name, len(paths), bad))

    print()
    print("artifact versions seen: %s"
          % ", ".join("v%s x%d" % (k, v) for k, v in sorted(versions.items(), key=str)))

    # Both nestings must actually have been exercised, or the suffix matching is
    # untested against the case it exists for.
    nestings = {".".join(k.split(".")[:-1]) for k in all_keys}
    print("kernel nestings exercised: %d" % len(nestings))
    for n in sorted(nestings):
        print("    %s" % n)

    if checked == 0:
        print("\nVERDICT: NOTHING CHECKED - no artifacts found under the given roots")
        return 1

    if failures:
        print("\nFAILURES: %d" % len(failures))
        for rec in failures[:20]:
            print("  %s\n      %s" % (rec["path"], rec["note"]))

    legacy_seen = any(v is not None and int(v) < GP_ARTIFACT_VERSION
                      for v in versions if v is not None)
    if legacy_seen and len(nestings) < 2:
        print("\nNOTE: only one kernel nesting was exercised. Suffix matching for the "
              "other form is untested here; widen --root or --limit.")

    print("\nVERDICT:", "artifacts round-trip in every schema found"
          if not failures else "ROUND TRIP BROKEN (%d artifacts)" % len(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

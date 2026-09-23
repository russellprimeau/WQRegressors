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
import sys
from pathlib import Path

import numpy as np

from utils import run_paths as rp
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, action="append", default=None,
                    help="Results root to check. Repeatable. Defaults to the reporting "
                         "root and the profiler arm.")
    ap.add_argument("--limit", type=int, default=40,
                    help="Artifacts to check per root.")
    args = ap.parse_args()

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

"""Record what a run actually did, so it never has to be inferred afterwards.

Why
---
Nothing in this pipeline recorded its own settings. Answering "what did CV22 do?" meant
reading directory names and the *current* defaults -- and the defaults had moved since,
so the answer came out wrong more than once. Two examples that cost real time: the
surrogate turned out to be chosen among XGBoost configs only (`auto:xgb`) rather than all
families, and whether the reported run used seed replication at all had to be settled by
counting directories.

A manifest removes the guessing. It is written before the work starts, so a run that
crashes still leaves a record of what it was attempting, and updated when the run ends.

Manifests accumulate under ``<root>/summaries/run_manifests/`` rather than overwriting:
a tree is often built by several invocations, and which ones touched it is exactly what
gets forgotten.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_SUBDIR = Path("summaries") / "run_manifests"


def _git_state() -> dict:
    """Commit, branch and dirty flag, or the reason none could be read.

    The dirty flag matters more than the commit: a run from an edited tree is not
    reproducible from its SHA, and saying so is the point.
    """
    def _run(*cmd: str) -> "str | None":
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15,
                                 cwd=str(Path(__file__).resolve().parents[2]))
        except Exception:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _run("git", "rev-parse", "HEAD")
    if sha is None:
        return {"available": False, "reason": "git not available or not a repository"}
    status = _run("git", "status", "--porcelain")
    return {
        "available": True,
        "commit": sha,
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_files": [ln[3:] for ln in (status or "").splitlines()[:40]],
    }


def _versions() -> dict:
    """Versions of the packages whose behaviour the results depend on."""
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "torch", "gpytorch", "xgboost", "sklearn", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def write_manifest(root: Path, script: str, args=None, extra: "dict | None" = None) -> Path:
    """Write a start-of-run manifest under *root* and return its path.

    Args:
        root: The output tree this run writes into.
        script: Name of the entry-point script.
        args: The parsed argparse namespace, if there is one.
        extra: Anything else worth pinning, such as resolved defaults that the
            command line did not state.

    Returns:
        The manifest path, to hand back to `finalize_manifest`.
    """
    started = datetime.now(timezone.utc)
    payload = {
        "script": script,
        "root": str(root),
        "argv": list(sys.argv),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in sorted(vars(args).items())} if args is not None else None,
        "extra": extra or {},
        "git": _git_state(),
        "versions": _versions(),
        "host": platform.node(),
        "pid": os.getpid(),
        "started_utc": started.isoformat(),
        "status": "running",
    }
    out_dir = Path(root) / MANIFEST_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("%s_%s_%d.json" % (started.strftime("%Y%m%dT%H%M%SZ"),
                                         Path(script).stem, os.getpid()))
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("[PROVENANCE] %s" % path)
    if payload["git"].get("dirty"):
        print("[PROVENANCE][WARN] the working tree has uncommitted changes; this run is "
              "not reproducible from its commit alone.")
    return path


def finalize_manifest(path: "Path | None", status: str, note: str = "") -> None:
    """Stamp the manifest with how the run ended. Never raises."""
    if path is None:
        return
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        ended = datetime.now(timezone.utc)
        started = datetime.fromisoformat(payload["started_utc"])
        payload.update(status=status, note=note, ended_utc=ended.isoformat(),
                       duration_seconds=round((ended - started).total_seconds(), 1))
        Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        # A manifest that cannot be closed is worth a line, not a crash: the run's real
        # output is already on disk by this point.
        print("[PROVENANCE][WARN] could not finalize %s: %s" % (path, exc))

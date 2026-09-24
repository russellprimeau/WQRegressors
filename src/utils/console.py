"""Console encoding for the pipeline's entry points.

On Windows the console defaults to cp1252, which cannot represent most of what this
project prints: the target names alone carry ``µ`` (``MC_Chromium__µg_L__diff``), and the
reporting code adds ``σ``, ``²`` and arrows. Writing one of those to a strict cp1252
stream raises ``UnicodeEncodeError`` from inside ``print``.

That is not a cosmetic failure. The sweep already lost a whole stage to it once -- the
exception surfaced inside a broad ``except Exception`` and the MLR k-cluster integration
was skipped with only a warning -- and ``z2_HorizonPostProcess --help`` aborted outright
on the ``σ`` in its own help text. Reconfiguring the streams removes the failure mode
rather than the characters, so no stage can be lost to a glyph, and ``errors="replace"``
keeps output flowing on any stream that still cannot represent something.

Call :func:`force_utf8_console` at import time in every entry point.
"""
from __future__ import annotations

import sys

__all__ = ["force_utf8_console"]


def force_utf8_console() -> None:
    """Make stdout and stderr UTF-8, replacing anything they still cannot encode."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Streams that are not reconfigurable (a pipe wrapper, a captured buffer)
            # are left alone: failing to adjust the console must never stop the run.
            pass

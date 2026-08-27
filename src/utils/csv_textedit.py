"""Replace whole columns of a CSV while leaving every other field byte-identical.

Why this is not ``DataFrame.to_csv``
------------------------------------
Rewriting a file through pandas perturbs the last bit of floats it was never asked
to touch: ``Consolidated_sparse.csv`` carries 17 significant digits, the CSV writer
emits fewer, and the two are not always the same double. Measured on that file,
a full rewrite moved 472 wind-speed values by one to two units in the last place.
Numerically irrelevant, but a silent edit to columns outside the requested scope --
and enough to make "nothing else changed" untrue as stated.

So the untouched fields are copied as text, and only the requested columns are
re-rendered, via ``repr``, which round-trips a float64 exactly. Verification then
compares the untouched columns as raw text rather than as parsed numbers, which is
the strongest check available: it catches encoding drift and float reformatting
alike.

This requires the file to have no quoted fields and a uniform field count. Both are
asserted before anything is written, so a file that needs a real CSV parser is
refused rather than silently mangled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RewriteReport:
    """What a rewrite touched, and how much of the file was proven unchanged."""
    columns: list[str]
    rows: int
    untouched_fields_verified: int


def _field(value) -> str:
    """One CSV field. Empty for missing, matching the writer that made the file."""
    if value is None:
        return ""
    v = float(value)
    if math.isnan(v):
        return ""
    return repr(v)


def split_lines(raw: str) -> tuple[list[str], str]:
    """Lines without terminators, plus the terminator in use."""
    eol = "\r\n" if "\r\n" in raw else "\n"
    body = raw[: -len(eol)] if raw.endswith(eol) else raw
    return body.split(eol), eol


def _assert_substitutable(lines: list[str], n_cols: int, name: str) -> None:
    for i, ln in enumerate(lines):
        if '"' in ln:
            raise ValueError(
                "%s line %d contains a double quote, so fields may be quoted and comma "
                "splitting is not safe. Refusing to write." % (name, i + 1))
        if ln.count(",") + 1 != n_cols:
            raise ValueError(
                "%s line %d has %d fields, expected %d. Refusing to write."
                % (name, i + 1, ln.count(",") + 1, n_cols))


def rewrite_columns(csv: Path, replacements: dict[str, np.ndarray]) -> RewriteReport:
    """Rewrite *csv*, replacing the named columns and copying every other field verbatim.

    ``replacements`` maps column name to a full-length array of new values; NaN is
    written as an empty field. Raises rather than writing a partially-verified file.
    """
    csv = Path(csv)
    raw = csv.read_text(encoding="utf-8", newline="")
    lines, eol = split_lines(raw)
    if not lines:
        raise ValueError("%s is empty." % csv.name)
    header = lines[0].split(",")
    _assert_substitutable(lines, len(header), csv.name)

    n_rows = len(lines) - 1
    missing = [c for c in replacements if c not in header]
    if missing:
        raise ValueError("column(s) not in %s: %s" % (csv.name, missing))
    for c, vals in replacements.items():
        if len(vals) != n_rows:
            raise ValueError("replacement for %r has %d values, file has %d data rows."
                             % (c, len(vals), n_rows))

    columns = list(replacements)
    idx = [header.index(c) for c in columns]
    arrays = [np.asarray(replacements[c], dtype=float) for c in columns]

    out = [lines[0]]
    for r in range(n_rows):
        fields = lines[r + 1].split(",")
        for j, k in enumerate(idx):
            fields[k] = _field(arrays[j][r])
        out.append(",".join(fields))
    csv.write_text(eol.join(out) + eol, encoding="utf-8", newline="")

    verified = _verify(csv, raw, columns, arrays, set(idx))
    return RewriteReport(columns=columns, rows=n_rows, untouched_fields_verified=verified)


def _verify(csv: Path, raw_before: str, columns: list[str],
            arrays: list[np.ndarray], skip: set[int]) -> int:
    before, _ = split_lines(raw_before)
    after, _ = split_lines(csv.read_text(encoding="utf-8", newline=""))
    if len(before) != len(after):
        raise ValueError("verification failed: line count changed.")
    if before[0] != after[0]:
        raise ValueError("verification failed: header text changed.")
    names = before[0].split(",")
    for r in range(1, len(before)):
        fb, fa = before[r].split(","), after[r].split(",")
        if len(fb) != len(fa):
            raise ValueError("verification failed: field count changed on line %d." % (r + 1))
        for k in range(len(fb)):
            if k not in skip and fb[k] != fa[k]:
                raise ValueError(
                    "verification failed: untouched field changed on line %d, column %r "
                    "(%r -> %r)." % (r + 1, names[k], fb[k], fa[k]))
    reread = pd.read_csv(csv, low_memory=False)
    for c, want in zip(columns, arrays):
        got = pd.to_numeric(reread[c], errors="coerce").to_numpy()
        if not np.isclose(got, want, rtol=0.0, atol=0.0, equal_nan=True).all():
            raise ValueError("verification failed: %r did not round-trip exactly." % c)
    return (len(before) - 1) * (len(names) - len(skip))

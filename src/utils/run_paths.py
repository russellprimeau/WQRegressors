"""Derive post-processing output paths from the run root actually being read.

Every ``z*`` reporting script defaults to the current results root. When those
defaults are written as two independent constants -- a default root and a
default output file that happens to sit under it -- pointing the script at a
second tree with ``--root`` leaves the output constant untouched, so the new
tree's analysis is written into the old tree's summaries under the old tree's
name. That is not hypothetical: it is how ``CV19/summaries/common_set_metrics.csv``
came to hold the profiler-free analysis.

The rule here is that an output path is a *function of the root* unless the
caller states otherwise, so a script can only ever write where it read.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The root the manuscript is built from. Declared once, here, so that changing
# which arm the paper reports is a one-line edit rather than a shell argument
# every caller has to remember. The profiler-bearing arm remains on disk and is
# still analysed, by passing --root explicitly.
REPORTING_ROOT = Path("data/output/CV22_profilerless")

# The alternative arm, retained for the profiler contrast and the supplementary
# table.
PROFILER_ROOT = Path("data/output/CV19")

# The results root used when a script is invoked with no arguments at all.
DEFAULT_ROOT = REPORTING_ROOT


def is_reporting_root(root: Path | str | None) -> bool:
    """True when ``root`` is the arm the manuscript reports from.

    Lets a script that writes into the manuscript refuse to do so silently for
    any other tree.
    """
    return resolve_root(root) == resolve_root(REPORTING_ROOT)


def resolve_root(root: Path | str | None) -> Path:
    """Absolute path to the results root, defaulting to ``DEFAULT_ROOT``."""
    r = Path(root) if root else DEFAULT_ROOT
    return r if r.is_absolute() else (REPO_ROOT / r).resolve()


def summaries_dir(root: Path | str | None) -> Path:
    """The ``summaries`` directory belonging to ``root``."""
    return resolve_root(root) / "summaries"


def summary_path(root: Path | str | None, name: str) -> Path:
    """Path to one artifact inside ``root``'s summaries directory."""
    return summaries_dir(root) / name


def resolve_output(explicit: Path | str | None, root: Path | str | None,
                   name: str) -> Path:
    """An explicit ``--output`` if given, otherwise ``name`` under ``root``.

    Used so that a reporting script writes beside the tree it read rather than
    beside a hard-coded default one.
    """
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()
    return summary_path(root, name)


def root_of_summary(summary: Path | str) -> Path:
    """The results root that owns a ``summaries/<file>`` path.

    Lets a script that takes ``--summary`` rather than ``--root`` still place
    its own output beside the summary it was given.
    """
    p = Path(summary).resolve()
    return p.parent.parent if p.parent.name == "summaries" else p.parent

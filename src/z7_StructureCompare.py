"""Compare model input/output structures on equal terms.

``z4_Compare.py`` reads each root's ``summary_best_model_performance.csv``, which
records the best candidate model under whatever candidate rules that root was
post-processed with.  That is not a fair basis for the structure comparison in
Section 3.1: the CV19 root treats multiple linear regression as a *reference
forecast* and excludes it from the candidate pool, while the older comparison
roots still carry MLR as a candidate.  Six of the fourteen best-model entries in
those roots are an MLR variant, so a figure built from the summary CSVs compares
"best of {GP, XGB, Transformer}" against "best of {GP, XGB, Transformer, MLR}"
and is biased against the structure it is meant to support.

This script goes back to the per-model ``feature_sweep_final_metrics.csv`` in
each dataset directory and applies one exclusion rule to every root, so the
structures are compared over an identical candidate pool.

Usage:
    python src/z7_StructureCompare.py
    python src/z7_StructureCompare.py --stat r2 --output <path.png>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.names import clean_target_label
from utils.plotstyle import (
    PAGE_WIDTH_IN,
    apply_paper_style,
    legend_above,
    save_figure,
)

# Paths are resolved against the repository root so the script runs identically
# from the repo root or from src/.
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ROOTS = [
    ("data/output/CV19", "State in, differential out"),
    ("data/output/CV16stateless", "No state in, differential out"),
    ("data/output/CV18_raw", "State in, state out"),
]
DEFAULT_OUTPUT = Path("data/output/comparisons/all/structure_r2.png")


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)

# Candidate families excluded from every root so the comparison is like-for-like.
DEFAULT_EXCLUDE = ("mlr",)


def _target_key(dataset: str) -> str:
    """Strip the root-specific prefix and target-construction suffix.

    The same target is named ``MC_Lead__ug_L__diff`` in the differential roots and
    ``MC_Lead__ug_L_`` in the absolute root, so the suffix has to come off before
    the roots can be joined.
    """
    key = re.sub(r"^MC_", "", str(dataset))
    return re.sub(r"_(diff|res)$", "", key)


def collect(root: Path, stat: str, exclude: tuple[str, ...]) -> dict[str, float]:
    """Best value of *stat* per target in *root*, over the filtered candidate pool."""
    best: dict[str, float] = {}
    paths = sorted(root.glob("*/forecasts/feature_sweeps/feature_sweep_final_metrics.csv"))
    if not paths:
        raise SystemExit(f"No feature_sweep_final_metrics.csv found under {root}")
    for path in paths:
        dataset = path.relative_to(root).parts[0]
        df = pd.read_csv(path)
        if stat not in df.columns or "model" not in df.columns:
            continue
        models = df["model"].astype(str).str.strip().str.lower()
        keep = ~models.str.startswith(exclude) if exclude else pd.Series(True, index=df.index)
        vals = pd.to_numeric(df.loc[keep, stat], errors="coerce").dropna()
        if len(vals):
            best[_target_key(dataset)] = float(vals.max())
    return best


def build_frame(roots, labels, stat, exclude) -> pd.DataFrame:
    cols = {}
    for root, label in zip(roots, labels):
        cols[label] = collect(_resolve(root), stat, exclude)
    keys = sorted(set().union(*(set(c) for c in cols.values())))
    frame = pd.DataFrame({lab: [cols[lab].get(k, np.nan) for k in keys] for lab in labels},
                         index=keys)
    # Keep only targets present in every root, so the bars compare like with like.
    complete = frame.dropna(how="any")
    dropped = [k for k in frame.index if k not in complete.index]
    if dropped:
        print(f"[WARN] Dropped {len(dropped)} target(s) missing from at least one root: "
              f"{', '.join(dropped)}")
    return complete


def plot(frame: pd.DataFrame, stat: str, output: Path, exclude: tuple[str, ...]) -> Path:
    apply_paper_style()
    labels = list(frame.columns)
    targets = [clean_target_label(k, "MC") for k in frame.index]
    n_t, n_r = len(frame), len(labels)

    height = max(3.0, 0.30 * n_t + 1.1)
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, height))

    y = np.arange(n_t)
    bar_h = 0.8 / n_r
    for i, lab in enumerate(labels):
        offset = (i - (n_r - 1) / 2) * bar_h
        ax.barh(y + offset, frame[lab].to_numpy(), height=bar_h, label=lab,
                edgecolor="none")

    ax.set_yticks(y)
    ax.set_yticklabels(targets)
    ax.invert_yaxis()
    ax.set_xlabel({"r2": "$R^2$ of best candidate model"}.get(stat, stat))
    ax.axvline(0.0, color="0.35", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    legend_above(ax, ncol=min(3, n_r))

    path = save_figure(fig, output)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", dest="roots", action="append", default=[])
    ap.add_argument("--label", dest="labels", action="append", default=[])
    ap.add_argument("--stat", default="r2")
    ap.add_argument("--exclude-model", dest="exclude", action="append", default=None,
                    help="Model-name prefix excluded from every root (default: mlr). "
                         "Repeat to exclude more; pass 'none' to exclude nothing.")
    ap.add_argument("--output", type=Path, default=None,
                    help=f"Output PNG (default: {DEFAULT_OUTPUT} under the repo root). "
                         "An explicit relative path is taken relative to the current "
                         "working directory.")
    args = ap.parse_args()

    if args.roots:
        roots = args.roots
        labels = args.labels or [Path(r).name for r in args.roots]
        if len(labels) != len(roots):
            raise SystemExit("--label must be given once per --root")
    else:
        roots = [r for r, _ in DEFAULT_ROOTS]
        labels = [l for _, l in DEFAULT_ROOTS]

    exclude = tuple(DEFAULT_EXCLUDE if args.exclude is None else
                    () if args.exclude == ["none"] else
                    [e.strip().lower() for e in args.exclude])

    # A default path belongs to the repo; an explicit one belongs to the caller's
    # working directory, so that "--output ../docs/..." means what it looks like.
    output = _resolve(DEFAULT_OUTPUT) if args.output is None else Path(args.output).resolve()

    frame = build_frame(roots, labels, args.stat, exclude)
    path = plot(frame, args.stat, output, exclude)

    print(f"[INFO] Wrote {path}")
    print(f"[INFO] Candidate models excluded from every root: "
          f"{', '.join(exclude) if exclude else '(none)'}")
    print(f"[INFO] {len(frame)} targets compared\n")
    summary = pd.DataFrame({
        "mean": frame.mean(), "median": frame.median(),
        "min": frame.min(), "wins": frame.idxmax(axis=1).value_counts(),
        "n>0": (frame > 0).sum(),
    })
    print(summary.fillna(0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

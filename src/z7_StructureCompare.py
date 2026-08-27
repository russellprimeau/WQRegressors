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

# The middle root was previously labelled "no state in, differential out". Its
# sample files carry a plain target column and its target values are byte-identical
# to CV18_raw's, so it targets the absolute value, not a difference. No
# configuration combining a differential target with no state input was ever run.
DEFAULT_ROOTS = [
    ("data/output/CV19", "State in, differential out"),
    ("data/output/CV16stateless", "No state in, absolute out"),
    ("data/output/CV18_raw", "State in, absolute out"),
]
DEFAULT_OUTPUT = Path("data/output/comparisons/all/structure_r2.png")


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)

# No family is excluded by default. The question this figure answers is whether
# any method finds a relationship under each structure, so the reference forecasts
# compete on the same terms as the learned models. Pass --exclude-model to
# restrict the pool for a diagnostic view.
DEFAULT_EXCLUDE = ()


def _target_key(dataset: str) -> str:
    """Strip the root-specific prefix and target-construction suffix.

    The same target is named ``MC_Lead__ug_L__diff`` in the differential roots and
    ``MC_Lead__ug_L_`` in the absolute root, so the suffix has to come off before
    the roots can be joined.
    """
    key = re.sub(r"^MC_", "", str(dataset))
    return re.sub(r"_(diff|res)$", "", key)


def _plain_label(label: str) -> str:
    """Readable form of a display label, for console output only.

    clean_target_label returns LaTeX so figures can italicise species names and
    set degree signs. Commands are mapped rather than stripped by pattern,
    because a command can run directly into the text that follows it.
    """
    s = str(label)
    for cmd, text in ((r"\upmu", "\u00b5"), (r"\circ", "\u00b0"),
                      (r"\it", ""), (r"\ ", " ")):
        s = s.replace(cmd, text)
    for ch in "${}^":
        s = s.replace(ch, "")
    s = s.replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def target_construction(root: Path) -> str:
    """Whether a root's targets are differences, residuals, or absolute values.

    The three roots do not share a target construction, and that governs how the
    figure may be read: an absolute series is far more autocorrelated than its
    differences, so R^2 is not comparable between them. Detecting it here means a
    future root swap cannot silently change what is being compared.
    """
    for ds in sorted(root.glob("MC_*")):
        files = sorted((ds / "samples").glob("segment_*.csv"))
        if not files:
            continue
        try:
            hdr = pd.read_csv(files[0], nrows=0, encoding="utf-8",
                              encoding_errors="replace").columns
        except Exception:
            continue
        if any(str(c).endswith("_diff") for c in hdr):
            return "differential"
        if any(str(c).endswith("_res") for c in hdr):
            return "residual"
        return "absolute"
    return "unknown"


def collect(root: Path, stat: str, exclude: tuple[str, ...],
            only: tuple[str, ...] = ()) -> dict[str, float]:
    """Best value of *stat* per target in *root*, over the filtered candidate pool.

    *only* restricts the pool to the named model-name prefixes. Its purpose is the
    structure comparison across roots built at different times: a family whose
    implementation changed between them cannot be compared across them, whereas a
    family that did not change can. Restricting to the common surrogate keeps the
    comparison honest at the cost of narrowing the claim, and the caption has to
    say so.
    """
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
        if only:
            keep &= models.str.startswith(only)
        vals = pd.to_numeric(df.loc[keep, stat], errors="coerce").dropna()
        if len(vals):
            best[_target_key(dataset)] = float(vals.max())
    return best


def build_frame(roots, labels, stat, exclude, only=()) -> pd.DataFrame:
    cols = {}
    for root, label in zip(roots, labels):
        cols[label] = collect(_resolve(root), stat, exclude, only)
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


def plot(frame: pd.DataFrame, stat: str, output: Path, exclude: tuple[str, ...],
         constructions: dict | None = None) -> Path:
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
    label_text = {"r2": "$R^2$ of best candidate model"}.get(stat, stat)
    if constructions and len(set(constructions.values())) > 1:
        label_text += "  (target construction differs between structures; see caption)"
    ax.set_xlabel(label_text)
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
    ap.add_argument("--only-model", dest="only", action="append", default=None,
                    help="Restrict every root to these model-name prefixes (repeatable). Use "
                         "when the roots were built at different times and only some families "
                         "are comparable across them.")
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

    # Report the target construction of each root before anything is compared, and
    # say plainly when they differ: R^2 on an absolute target and R^2 on a
    # differential target are not on the same footing.
    constructions = {label: target_construction(_resolve(root))
                     for root, label in zip(roots, labels)}
    print("[INFO] target construction by structure:")
    for label, kind in constructions.items():
        print(f"         {label:<34} {kind}")
    if len(set(constructions.values())) > 1:
        print("[WARN] These structures do not share a target construction. An absolute "
              "series is far more autocorrelated than its differences, so the R^2 values "
              "below are not directly comparable between them; the differential structure "
              "is scored on the harder quantity.")

    only = tuple(o.strip().lower() for o in (args.only or []))
    frame = build_frame(roots, labels, args.stat, exclude, only)
    path = plot(frame, args.stat, output, exclude, constructions)

    print(f"[INFO] Wrote {path}")
    if only:
        print(f"[INFO] Candidate models restricted to: {', '.join(only)}")
    print(f"[INFO] Candidate models excluded from every root: "
          f"{', '.join(exclude) if exclude else '(none)'}")
    print(f"[INFO] {len(frame)} targets compared\n")
    # Counts only. The targets have unrelated dynamics, so a mean or median across
    # them describes no quantity of interest.
    wins = frame.idxmax(axis=1).value_counts()
    summary = pd.DataFrame({
        "targets won": wins,
        "targets with R2 > 0": (frame > 0).sum(),
        "targets with R2 > 0.5": (frame > 0.5).sum(),
    }).fillna(0).astype(int)
    print(summary.to_string())
    print()
    for label in frame.columns:
        won = [t for t in frame.index if frame.loc[t].idxmax() == label]
        if won:
            names = [_plain_label(clean_target_label(t, "MC")) for t in won]
            print(f"  {label}: best for {len(won)} of {len(frame)} targets "
                  f"({', '.join(names)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

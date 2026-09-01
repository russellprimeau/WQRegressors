"""What the in-lake profiler channels are worth, separated from what they cost.

The two differential runs on disk differ in exactly one factor: the
profiler-bearing arm carries the seven ``Pfl - `` channels alongside the ten
weather and SCADA predictors, the profiler-free arm carries the ten alone.
Everything else -- targets, target construction, window lengths, splits -- is
identical.

Two quantities are wanted, and they must not be confused with one another:

**Cost.** The profiler is recorded for only part of the year, so a window using
it is dropped wherever a channel is absent. The families that require complete
predictor rows therefore lose most of their laboratory samples, which shrinks
the common evaluation set for every family compared against them. This is
reported as the per-family segment count in each arm.

**Value.** Whether the channels carry predictive information at all. Measuring
this needs a comparison the coverage loss cannot contaminate, so it is
restricted to **XGBoost**, the one family that tolerates missing predictors
natively and therefore scores the *same segments in both arms*. Each arm's best
retained XGBoost configuration is scored on the intersection of the two arms'
segments.

Restricting the value comparison to a single family narrows the claim to
"under a common model, these predictors are worth this much", and that
restriction belongs in the caption rather than being left implicit. It is the
same restriction, and for the same reason, that the structure comparison
applies.

Nothing is recomputed from raw predictions that ``z8`` does not already define:
this module imports z8's run loader so that a number here and a number in the
results table describe the same objects.

Usage:
    python src/z13_ProfilerContrast.py
    python src/z13_ProfilerContrast.py --root-with data/output/CV19
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import z8_CommonSetMetrics as z8
from utils.names import clean_target_label
from utils.plotstyle import (
    PAGE_WIDTH_IN, apply_paper_style, legend_above, save_figure)
from utils import run_paths as rp

# The probe family: the only one whose evaluation coverage is unaffected by
# profiler availability, which is what makes the accuracy comparison matched.
PROBE = "XGB"

# Below this, a difference in R^2 is not an interpretable direction at the 11 to
# 48 segments available, so it is counted as indistinguishable rather than as a
# win for whichever side happens to carry the sign.
MEANINGFUL_DELTA = 0.01

OUTPUT_CSV = "profiler_contrast.csv"
OUTPUT_PNG = "profiler_contrast.png"


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    if ss <= 0:
        return float("nan")
    return 1.0 - float(((y - p) ** 2).sum()) / ss


def _fam_key(family: str) -> str:
    return family.lower().replace("-", "")


def _retained_runs(root: Path, dataset: str) -> list:
    sweeps = root / dataset / "forecasts" / "feature_sweeps"
    if not sweeps.is_dir():
        return []
    return [r for r in z8.load_runs(sweeps) if r.retained]


def _best_on(runs: list, segments: list) -> tuple:
    """Best R^2 among runs covering every segment, with the run that achieved it.

    A run that cannot score the whole set is excluded rather than scored on a
    subset, because scoring on a subset is the confound this comparison exists
    to remove.
    """
    need = set(segments)
    best, who = float("nan"), ""
    for r in runs:
        if not need <= r.segments:
            continue
        y = np.array([r.targets[s] for s in segments], dtype=float)
        p = np.array([r.preds[s] for s in segments], dtype=float)
        v = _r2(y, p)
        if np.isfinite(v) and (not np.isfinite(best) or v > best):
            best, who = v, r.run
    return best, who


def _coverage(runs: list) -> dict:
    """Per family, the most segments any one of its retained runs can score."""
    out = {}
    for fam in sorted({r.family for r in runs}):
        cand = [r for r in runs if r.family == fam]
        out[fam] = max(len(r.segments) for r in cand) if cand else 0
    return out


def analyse(root_with: Path, root_without: Path, prefix: str) -> pd.DataFrame:
    rows = []
    for d in sorted(root_with.glob("MC_*")):
        if not d.is_dir():
            continue
        name = d.name
        if not (root_without / name).is_dir():
            print("[WARN] %s: no counterpart in %s; skipped." % (name, root_without.name))
            continue

        runs_w = _retained_runs(root_with, name)
        runs_o = _retained_runs(root_without, name)
        probe_w = [r for r in runs_w if r.family == PROBE]
        probe_o = [r for r in runs_o if r.family == PROBE]
        if not probe_w or not probe_o:
            print("[WARN] %s: no retained %s run in one arm; skipped." % (name, PROBE))
            continue

        segs_w = set().union(*[r.segments for r in probe_w])
        segs_o = set().union(*[r.segments for r in probe_o])
        shared = sorted(segs_w & segs_o, key=z8.segment_index)
        if len(shared) < 3:
            print("[WARN] %s: only %d shared segment(s); skipped." % (name, len(shared)))
            continue

        r2_w, run_w = _best_on(probe_w, shared)
        r2_o, run_o = _best_on(probe_o, shared)

        cov_w, cov_o = _coverage(runs_w), _coverage(runs_o)
        row = dict(
            dataset=name,
            target=clean_target_label(name, prefix),
            n_shared=len(shared),
            probe=PROBE,
            r2_with=r2_w,
            r2_without=r2_o,
            delta=r2_o - r2_w,
            run_with=run_w,
            run_without=run_o,
        )
        for fam in sorted(set(cov_w) | set(cov_o)):
            row["n_" + _fam_key(fam) + "_with"] = cov_w.get(fam)
            row["n_" + _fam_key(fam) + "_without"] = cov_o.get(fam)
        rows.append(row)

    if not rows:
        raise SystemExit("no target could be compared across the two arms.")
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


IMPROVED_COLOUR = "#1f77b4"
WORSE_COLOUR = "#d62728"
NEUTRAL_COLOUR = "0.55"


def render(df: pd.DataFrame, output: Path) -> Path:
    apply_paper_style()
    n = len(df)

    # Row pitch and grid match the other stacked per-target figures; every font size
    # comes from apply_paper_style rather than being set here, so this figure prints at
    # the same sizes as the method comparison and the retention counts.
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 0.30 * n + 1.5))

    # One slope per target: a line running right means dropping the profiler
    # channels improved the fit on segments both predictor sets score. Colour
    # follows the same threshold as the reported counts, so a difference too
    # small to interpret is not drawn as a direction. The blue is the one the
    # prediction time-series figures use for a model series, and the red is the
    # "worse" end of the method-comparison colour scale.
    for i, r in df.iterrows():
        a, b = float(r["r2_with"]), float(r["r2_without"])
        d = b - a
        colour = (IMPROVED_COLOUR if d > MEANINGFUL_DELTA
                  else WORSE_COLOUR if d < -MEANINGFUL_DELTA else NEUTRAL_COLOUR)
        ax.plot([a, b], [i, i], color=colour, lw=1.4, zorder=2,
                solid_capstyle="round")
        ax.scatter([b], [i], s=20, color=colour, zorder=3)
        # Drawn last so that it stays visible where the two arms coincide;
        # otherwise those targets read as having only one result.
        ax.scatter([a], [i], s=22, facecolor="none", edgecolor="0.25",
                   linewidth=1.0, zorder=4)

    ax.axvline(0.0, color="0.75", lw=0.8, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(["%s  (%d)" % (r["target"], int(r["n_shared"]))
                        for _, r in df.iterrows()])
    ax.invert_yaxis()
    ax.set_xlabel("$R^2$ on the segments both predictor sets score")
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Above the axes, not inside them: at lower right the legend sat on top of the
    # bottom-ranked target's marker.
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                   markeredgecolor="0.25", markeredgewidth=1.0, markersize=4.6),
        plt.Line2D([0], [0], marker="o", linestyle="none", color=IMPROVED_COLOUR,
                   markersize=4.4),
    ]
    legend_above(ax, handles, ["profiler-bearing", "profiler-free"], ncol=2)

    return save_figure(fig, output)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root-with", type=Path, default=None,
                    help="Profiler-bearing arm. Defaults to the alternative arm.")
    ap.add_argument("--root-without", type=Path, default=None,
                    help="Profiler-free arm. Defaults to the reporting root.")
    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--output-png", type=Path, default=None)
    ap.add_argument("--dataset-prefix", type=str, default="MC")
    args = ap.parse_args()

    root_with = rp.resolve_root(args.root_with or rp.PROFILER_ROOT)
    root_without = rp.resolve_root(args.root_without or rp.REPORTING_ROOT)
    for r in (root_with, root_without):
        if not r.is_dir():
            raise SystemExit("results root not found: %s" % r)

    # Both artifacts land beside the arm the paper reports from, whose results
    # section carries this figure.
    csv_path = rp.resolve_output(args.output_csv, root_without, OUTPUT_CSV)
    png_path = rp.resolve_output(args.output_png, root_without, OUTPUT_PNG)

    df = analyse(root_with, root_without, args.dataset_prefix)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    render(df, png_path)
    print("[INFO] Wrote %s" % csv_path)
    print("[INFO] Wrote %s" % png_path)

    print("\n[INFO] %s on the segments both arms score (%s -> %s):"
          % (PROBE, root_with.name, root_without.name))
    print("  %-26s %5s %8s %9s %8s" % ("target", "segs", "with", "without", "delta"))
    for _, r in df.iterrows():
        print("  %-26s %5d %+8.3f %+9.3f %+8.3f"
              % (str(r["target"])[:26], int(r["n_shared"]),
                 r["r2_with"], r["r2_without"], r["delta"]))

    # Counts only; the targets have unrelated dynamics and nothing is averaged.
    # A threshold rather than an exact sign: at 11 to 48 segments a difference of
    # a few thousandths of R^2 is not an interpretable direction, and counting it
    # as one would report a null as a result.
    better = int((df["delta"] > MEANINGFUL_DELTA).sum())
    worse = int((df["delta"] < -MEANINGFUL_DELTA).sum())
    same = len(df) - better - worse
    print("\n[INFO] dropping the profiler channels, at a %.2f R^2 threshold: better on "
          "%d target(s), indistinguishable on %d, worse on %d."
          % (MEANINGFUL_DELTA, better, same, worse))

    # The coverage cost, which is the other half of the question.
    with_cols = [c for c in df.columns if c.startswith("n_") and c.endswith("_with")]
    print("[INFO] segments each family can score, profiler-bearing -> profiler-free:")
    for _, r in df.iterrows():
        parts = []
        for c in with_cols:
            fam = c[2:-5]
            a, b = r.get(c), r.get("n_" + fam + "_without")
            if pd.notna(a) and pd.notna(b) and int(a) != int(b):
                parts.append("%s %d->%d" % (fam, int(a), int(b)))
        if parts:
            print("  %-26s %s" % (str(r["target"])[:26], ", ".join(parts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

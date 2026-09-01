"""Explained variance against forecast horizon, one panel per target.

Section 3.4 needs a single figure showing how far ahead each target stays predictable.
``z2_HorizonPostProcess`` already computes the underlying numbers and draws its own
version of this plot, but at its own sizes and colours; this redraws the same quantity
in the house style so that it sits beside the other manuscript figures without a change
of typeface or scale. Nothing is recomputed here.

The y-axis is shared across panels, unlike the per-target figures elsewhere in this
paper: R^2 is already a normalized quantity, so a common scale is what makes the decay
rates comparable between targets. It is clipped at -1, because an R^2 below that says
only "much worse than the evaluation-period mean" and one target reaching -9.5 would
otherwise flatten every other panel. Points outside the range are marked rather than
dropped.

Usage:
    python src/z16_HorizonCurves.py
    python src/z16_HorizonCurves.py --root data/output/CV22_profilerless
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.names import clean_target_label  # noqa: E402
from utils.plotstyle import apply_paper_style, legend_above, save_figure  # noqa: E402
from utils import run_paths as rp  # noqa: E402

AGGREGATE = Path("summaries") / "horizons" / "eval_test" / "best" / "lookahead_aggregate.csv"
OUTPUT_NAME = "horizon_r2.png"
ENSEMBLE_NAME = "lookahead_ensemble.csv"

FIG_WIDTH_IN = 13.0
ROW_HEIGHT_IN = 0.72
BASE_FONT_PT = 14
ROW_LABEL_PT = int(round(BASE_FONT_PT * 1.1))
X_LABEL_PT = int(round(BASE_FONT_PT * 1.3))

Y_FLOOR = -1.0
Y_CEIL = 1.0

MEAN_STYLE = {"color": "#1f77b4", "marker": "o", "ms": 4.0, "lw": 1.2, "zorder": 3}
REP_STYLE = {"color": "#1f77b4", "marker": ".", "ms": 2.6, "alpha": 0.45,
             "linestyle": "none", "zorder": 2}


def load(root: Path) -> pd.DataFrame:
    """Per-horizon scores, preferring the seed ensemble where one exists.

    z2's aggregate holds one row per replicate, and averaging their R^2 is not the
    quantity Table 3 reports. Squared error is convex in the prediction, so the R^2 of
    the mean prediction exceeds the mean of the R^2s by the across-seed variance term
    -- 0.035 on Cadmium at horizon 0, which left this figure's leftmost point
    disagreeing with the results table for every target whose winner is stochastic.
    z18 writes the ensemble; it is used when present, with the replicate mean as a
    fallback that warns.
    """
    ens = sorted(root.glob("MC_*/horizons/lookahead_sweeps/" + ENSEMBLE_NAME))
    if ens:
        frames = []
        for f in ens:
            t = pd.read_csv(f, encoding="utf-8", encoding_errors="replace")
            # z18 names the column `horizon`; draw() and z2's aggregate use `lookahead`.
            t = t.rename(columns={"horizon": "lookahead"})
            t["replicate"] = "0"
            t["dataset"] = [clean_target_label(str(x), "MC") for x in t["dataset"]]
            frames.append(t)
        print("[INFO] using the seed ensemble for %d target(s)" % len(ens))
        return pd.concat(frames, ignore_index=True)
    path = root / AGGREGATE
    if not path.is_file():
        raise SystemExit(
            "Not found: %s\nRun:\n    python src/z2_HorizonPostProcess.py --data-root %s"
            % (path, root))
    print("[WARN] no %s found; falling back to the mean of the replicates' R^2, which "
          "does not match Table 3 for stochastic winners. Run z18_HorizonEnsembles.py."
          % ENSEMBLE_NAME)
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def draw(df: pd.DataFrame, out_path: Path) -> Path:
    apply_paper_style()
    reps = df[df["replicate"].astype(str) != "mean"]
    order = (reps[reps["lookahead"] == reps["lookahead"].min()]
             .groupby("dataset")["r2"].mean().sort_values(ascending=False).index.tolist())
    horizons = sorted(reps["lookahead"].unique())
    # Evenly spaced positions: the sweep is denser at short horizons, and a linear time
    # axis would compress 0-48 h into the left eighth of the panel.
    pos = {h: i for i, h in enumerate(horizons)}

    fig, axes_raw = plt.subplots(
        len(order), 1, sharex=True,
        figsize=(FIG_WIDTH_IN, max(2.8, ROW_HEIGHT_IN * len(order))),
        gridspec_kw={"hspace": 0.10})
    axes = [axes_raw] if len(order) == 1 else list(axes_raw)

    clipped = []
    for ax, name in zip(axes, order):
        g = reps[reps["dataset"] == name]
        x = [pos[h] for h in g["lookahead"]]
        ax.plot(x, np.clip(g["r2"], Y_FLOOR, Y_CEIL), **REP_STYLE)
        m = g.groupby("lookahead")["r2"].mean().reindex(horizons)
        ax.plot([pos[h] for h in horizons], np.clip(m.to_numpy(), Y_FLOOR, Y_CEIL),
                label="Mean over replicates", **MEAN_STYLE)

        # An out-of-range point is marked where it leaves the axis, so a clipped curve
        # cannot be read as one that merely flattened out.
        out = [(pos[h], v) for h, v in m.items() if np.isfinite(v) and v < Y_FLOOR]
        if out:
            ax.plot([p for p, _ in out], [Y_FLOOR] * len(out), marker="v", ms=4.5,
                    linestyle="none", color="#d62728", zorder=4,
                    label="Below the axis range")
            clipped.append((name, min(v for _, v in out)))

        ax.axhline(0.0, color="#999999", lw=0.6, ls="--", zorder=1)
        ax.set_ylim(Y_FLOOR - 0.08, Y_CEIL + 0.08)
        ax.set_yticks([-1, 0, 1])
        ax.set_ylabel(name, rotation=0, ha="right", va="center",
                      fontsize=ROW_LABEL_PT, labelpad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.4)
        ax.yaxis.set_major_locator(mticker.FixedLocator([-1, 0, 1]))
        ax.tick_params(axis="y", labelsize=BASE_FONT_PT)

    axes[-1].set_xticks(list(pos.values()))
    axes[-1].set_xticklabels(["%d" % h for h in horizons], fontsize=X_LABEL_PT)
    axes[-1].set_xlabel("Forecast horizon (hours)", fontsize=X_LABEL_PT)
    axes[-1].set_xlim(-0.25, len(horizons) - 0.75)

    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    legend_above(axes[0], handles, labels, ncol=len(labels), fontsize=X_LABEL_PT)

    widest = max(len(n) for n in order)
    left = (widest * max(0.045, 0.0055 * ROW_LABEL_PT) + 0.55) / FIG_WIDTH_IN
    fig.subplots_adjust(left=max(0.14, min(0.26, left)), right=0.995, hspace=0.10)

    for name, worst in clipped:
        print("  [NOTE] %s falls to R2 = %+.2f, below the plotted range." % (name, worst))
    return save_figure(fig, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    root = rp.resolve_root(args.root)
    df = load(root)
    out = args.output or (root / "summaries" / OUTPUT_NAME)
    print("Wrote %s" % draw(df, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

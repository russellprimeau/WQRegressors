"""Normalized error for every method on every target, on the common evaluation set.

Why this exists rather than the ML-comparison plot from ``z1``
-------------------------------------------------------------
``z1`` builds its comparison from each run's own metrics, so a cell there is
scored on whatever segments that configuration happened to retain, and the four
statistical models are filtered out of it entirely when MLR is treated as a
baseline. Both are wrong for this paper: the reported comparison basis is the
common evaluation set, and the statistical models are candidates for "which
method predicts this target best" rather than a separate class held aside.

``z8`` already scores all nine methods per target on that one set with a single
sigma, so this reads its output and draws it. Nothing is recomputed here.

The layout is a matrix rather than grouped bars. Nine methods across 14 targets
is 126 bars, which is unreadable at page width, and the quantity plotted is
ordinal, so a sequential colormap carries information rather than merely
labelling categories.

Usage:
    python src/z12_MethodComparison.py
    python src/z12_MethodComparison.py --stat r2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.names import clean_target_label
from utils.plotstyle import PAGE_WIDTH_IN, apply_paper_style, save_figure

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = Path("data/output/CV19/summaries/common_set_metrics.csv")
DEFAULT_OUTPUT = Path("data/output/CV19/summaries/method_comparison.png")

# Column prefix in common_set_metrics.csv -> display label. Machine-learning
# models first, then the four statistical models, so the split is visible in the
# figure without needing a legend to explain it.
ML = [("xgb", "XGB"), ("gp", "GP"), ("transformer", "Trans.")]
STAT = [("mlr", "MLR"), ("mlr12", "MLR-12"), ("mlrall", "MLR-All"),
        ("naive", "Naive"), ("seasonal", "Seasonal"), ("linear", "Linear")]

STATS = {
    "nrmse": dict(label="nRMSE", cmap="RdYlGn_r", fmt="{:.2f}"),
    "r2": dict(label="$R^2$", cmap="RdYlGn", fmt="{:.2f}"),
}


def build(df: pd.DataFrame, stat: str, prefix: str) -> pd.DataFrame:
    """Targets (rows, ordered by best R^2) by method (columns)."""
    df = df.sort_values("best_r2", ascending=False)
    cols = ML + STAT
    out = {}
    for key, label in cols:
        out[label] = [_f(r.get(f"{key}_{stat}")) for _, r in df.iterrows()]
    index = [clean_target_label(str(d), prefix) for d in df["dataset"]]
    return pd.DataFrame(out, index=index)


def _f(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def render(mat: pd.DataFrame, stat: str, output: Path) -> Path:
    apply_paper_style()
    spec = STATS[stat]
    n_rows, n_cols = mat.shape

    # Clip the colour scale to the central mass so that one very poor method does
    # not flatten the contrast across every other cell. The printed number is
    # never clipped, so an out-of-scale value is still readable.
    finite = mat.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    lo, hi = np.percentile(finite, [5, 95]) if finite.size else (0.0, 1.0)

    fig_h = 0.30 * n_rows + 1.35
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, fig_h))
    ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap=spec["cmap"],
              vmin=lo, vmax=hi)

    for i in range(n_rows):
        for j in range(n_cols):
            v = mat.iat[i, j]
            ax.text(j, i, "---" if not np.isfinite(v) else spec["fmt"].format(v),
                    ha="center", va="center", fontsize=7.0)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(mat.columns, fontsize=7.0, rotation=30, ha="left")
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(mat.index, fontsize=7.0)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # The boundary between the machine-learning and statistical blocks.
    ax.axvline(len(ML) - 0.5, color="white", lw=2.0)
    ax.text((len(ML) - 1) / 2, n_rows - 0.35, "machine-learning",
            ha="center", va="top", fontsize=7.0, style="italic")
    ax.text(len(ML) + (len(STAT) - 1) / 2, n_rows - 0.35, "statistical",
            ha="center", va="top", fontsize=7.0, style="italic")

    return save_figure(fig, output)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--stat", choices=sorted(STATS), default="nrmse")
    ap.add_argument("--dataset-prefix", default="MC")
    args = ap.parse_args()

    summary = args.summary or (REPO_ROOT / DEFAULT_SUMMARY)
    if not summary.is_file():
        raise SystemExit("common-set metrics not found: %s. Run z8 first." % summary)
    output = args.output or (REPO_ROOT / DEFAULT_OUTPUT)
    if args.stat != "nrmse" and args.output is None:
        output = output.with_name("method_comparison_%s.png" % args.stat)

    df = pd.read_csv(summary)
    mat = build(df, args.stat, args.dataset_prefix)
    render(mat, args.stat, output)
    print("[INFO] Wrote %s" % output)
    missing = int(mat.isna().to_numpy().sum())
    print("[INFO] %d targets x %d methods; %d cell(s) with no result"
          % (mat.shape[0], mat.shape[1], missing))
    for c in mat.columns:
        n = int(mat[c].isna().sum())
        if n:
            print("[INFO]   %s: no result for %d target(s)" % (c, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

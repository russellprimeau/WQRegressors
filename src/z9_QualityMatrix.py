"""Render the per-target quality matrix from the common evaluation set.

Replaces the version built inside ``z1_FeaturePostProcess.py``, which had three
problems the figure could not survive review with:

* It led with the best *machine-learning* model and named the reference only as
  a label, so a reader could not see that a reference forecast achieved a higher
  R^2 on six of fourteen targets. The study asks which method predicts each
  target, so the winner leads regardless of family.
* Where the best model and the best reference were the same row it wrote a skill
  of exactly ``0.00``, which reads as a measured null rather than as
  "not applicable".
* Every column was normalized independently for colour, so an R^2 of 0.46 and an
  R^2 of 0.42 could differ by half the colour range. R^2 now uses a fixed scale.

The statistics come from ``common_set_metrics.csv``, where every method for a
target is scored on the same test segments, so the accuracy and skill columns
finally describe one comparison. The skill block is left blank wherever a
reference forecast won, because there the reference *is* the result.

Usage:
    python src/z9_QualityMatrix.py
    python src/z9_QualityMatrix.py --summary <path.csv> --output <path.png>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

from utils.names import clean_target_label
from utils.plotstyle import PAGE_WIDTH_IN, apply_paper_style, save_figure
from utils import run_paths as rp

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_NAME = "common_set_metrics.csv"
OUTPUT_NAME = "summary_model_quality_matrix.png"

ML_FAMILIES = {"GP", "XGB", "Transformer"}

VERDICT_RANK = {"supported": 3, "directional": 2, "underpowered": 1, "not_supported": 0}
VERDICT_SHORT = {"supported": "Supported", "directional": "Directional",
                 "underpowered": "Underpowered", "not_supported": "Not supported"}

# Accuracy columns describe the target; the skill block describes whether the
# result stands out from the alternatives, which is a question for every target.
# Family name and value are separate columns: combined into one cell they
# overflow at any font size that satisfies the printed-size rule.
ACCURACY_COLS = ["n", "Best", "$R^2$", "nRMSE",
                 "ML", "ML $R^2$", "Ref.", "Ref. $R^2$"]
SKILL_COLS = ["Skill", "95% LB", "Win rate", "$p$", "$p_{\\min}$", "Verdict"]


# "Transformer" is the widest label in the table and appears in two columns; at
# the printed-size floor it is the difference between fitting the page and not.
_SHORT_FAMILY = {"Transformer": "Trans."}


def _short(name: str) -> str:
    name = str(name or "").strip()
    return _SHORT_FAMILY.get(name, name) or "---"


def _fmt(v, places=2, dash="---"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return dash
    return f"{v:.{places}f}"


def build_frame(df: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (display strings, colour values) indexed by target label."""
    df = df.sort_values("best_r2", ascending=False)
    text_rows, norm_rows, index = [], [], []

    for _, r in df.iterrows():
        family = str(r.get("best_family", "") or "").strip()
        is_ml = family in ML_FAMILIES
        ml_fam = str(r.get("best_ml_family", "") or "").strip()
        ref_fam = str(r.get("best_ref_family", "") or "").strip()
        verdict = str(r.get("aligned_verdict", "") or "").strip()

        n = r.get("n_common")
        best_r2 = r.get("best_r2")
        key = family.lower().replace("-", "")
        nrmse = r.get(f"{key}_nrmse")

        text = {
            "n": _fmt(n, 0),
            "Best": _short(family),
            "$R^2$": _fmt(best_r2),
            "nRMSE": _fmt(nrmse),
            "ML": _short(ml_fam),
            "ML $R^2$": _fmt(r.get("best_ml_r2")),
            "Ref.": _short(ref_fam),
            "Ref. $R^2$": _fmt(r.get("best_ref_r2")),
        }
        norm = {
            "n": n,
            "Best": np.nan,
            "$R^2$": best_r2,
            "nRMSE": nrmse,
            "ML": np.nan,
            "ML $R^2$": r.get("best_ml_r2"),
            "Ref.": np.nan,
            "Ref. $R^2$": r.get("best_ref_r2"),
        }

        # The skill block says whether the winning result stands out from the
        # alternatives or is merely the best of several comparable ones, so it is
        # populated for every target. Where a reference forecast won, the skill of
        # the best learned model against it is negative, and its interval says
        # whether that deficit is resolvable at the available sampling.
        text.update({
            "Skill": _fmt(r.get("skill_vs_best_ref")),
            "95% LB": _fmt(r.get("aligned_skill_ci05")),
            "Win rate": _fmt(r.get("aligned_sign_win_rate")),
            "$p$": _fmt(r.get("aligned_sign_p")),
            "$p_{\\min}$": _fmt(r.get("aligned_min_attainable_p")),
            "Verdict": VERDICT_SHORT.get(verdict, "---"),
        })
        norm.update({
            "Skill": r.get("skill_vs_best_ref"),
            "95% LB": r.get("aligned_skill_ci05"),
            "Win rate": r.get("aligned_sign_win_rate"),
            "$p$": r.get("aligned_sign_p"),
            "$p_{\\min}$": r.get("aligned_min_attainable_p"),
            "Verdict": VERDICT_RANK.get(verdict, np.nan),
        })

        text_rows.append(text)
        norm_rows.append(norm)
        index.append(clean_target_label(str(r.get("dataset", "")), prefix))

    cols = ACCURACY_COLS + SKILL_COLS
    return (pd.DataFrame(text_rows, index=index)[cols],
            pd.DataFrame(norm_rows, index=index)[cols])


def _column_norm(col: str, values: np.ndarray):
    """Colour scaling per column, fixed where the quantity has a natural range."""
    finite = values[np.isfinite(values)]
    if col in ("$R^2$", "Best ML", "Best reference"):
        # Fixed, so equal R^2 values look equal across columns and targets. R^2 is
        # unbounded below, so anything negative saturates at the floor.
        return Normalize(vmin=0.0, vmax=1.0), False
    if col in ("ML $R^2$", "Ref. $R^2$"):
        return Normalize(vmin=0.0, vmax=1.0), False
    if col == "Win rate":
        return Normalize(vmin=0.0, vmax=1.0), False
    if col in ("$p$", "$p_{\\min}$"):
        return Normalize(vmin=0.0, vmax=1.0), True
    if col == "nRMSE":
        return Normalize(vmin=0.0, vmax=max(1.5, float(finite.max()) if finite.size else 1.5)), True
    if col in ("Skill", "95% LB"):
        lim = max(0.5, float(np.nanmax(np.abs(finite))) if finite.size else 0.5)
        return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), False
    if col == "Verdict":
        return Normalize(vmin=0, vmax=3), False
    if col == "n":
        return Normalize(vmin=0.0, vmax=float(finite.max()) if finite.size else 1.0), False
    return Normalize(vmin=0.0, vmax=1.0), False


def render(text: pd.DataFrame, norm: pd.DataFrame, output: Path) -> Path:
    apply_paper_style()
    n_rows = len(text)
    gap_at = len(ACCURACY_COLS)

    # Columns are sized to the widest string they must hold, including the
    # header. Uniform widths force either overflow or a font below the printed
    # minimum, which is how the previous version rendered "GP 0.82" on top of
    # "MLR-All 0.62".
    def _visible(s: str) -> int:
        return len(str(s).replace("$", "").replace("\\", "").replace("_{", "").replace("}", ""))

    widths = []
    for col in text.columns:
        longest = max([_visible(col)] + [_visible(v) for v in text[col]])
        widths.append(max(3.0, longest + 1.0))
    gap_w = 1.4
    total = sum(widths) + gap_w

    xs, acc = [], 0.0
    for j, w in enumerate(widths):
        if j == gap_at:
            acc += gap_w
        xs.append(acc)
        acc += w

    fig_h = max(3.2, 0.30 * n_rows + 1.9)
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, fig_h))
    ax.set_xlim(0, total)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    cmap = plt.get_cmap("RdYlGn")
    for j, col in enumerate(text.columns):
        x, w = xs[j], widths[j]
        vals = pd.to_numeric(norm[col], errors="coerce").to_numpy(dtype=float)
        cnorm, invert = _column_norm(col, vals)
        for i in range(n_rows):
            label = str(text[col].iloc[i])
            if label == "":
                continue
            v = vals[i]
            if np.isfinite(v):
                frac = float(np.clip(cnorm(v), 0.0, 1.0))
                colour = cmap(1.0 - frac if invert else frac)
            else:
                colour = (0.92, 0.92, 0.92, 1.0)
            ax.add_patch(plt.Rectangle((x, i), w, 1, facecolor=colour,
                                       edgecolor="white", linewidth=0.6))
            ax.text(x + w / 2, i + 0.5, label, ha="center", va="center", fontsize=7.0)
        ax.text(x + w / 2, -0.2, col, ha="left", va="bottom", fontsize=7.0,
                rotation=40, rotation_mode="anchor")

    for i, name in enumerate(text.index):
        ax.text(-0.4, i + 0.5, name, ha="right", va="center", fontsize=7.0)

    mid_acc = (xs[0] + xs[gap_at - 1] + widths[gap_at - 1]) / 2
    mid_skill = (xs[gap_at] + xs[-1] + widths[-1]) / 2
    ax.text(mid_acc, n_rows + 0.45, "accuracy: which method predicts the target",
            ha="center", va="top", fontsize=7.0, style="italic")
    ax.text(mid_skill, n_rows + 0.45, "standing: does that result stand out",
            ha="center", va="top", fontsize=7.0, style="italic")

    return save_figure(fig, output)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--dataset-prefix", type=str, default="MC")
    args = ap.parse_args()

    # The output follows the summary that was actually read, so pointing
    # --summary at a second results tree cannot overwrite the first tree's figure.
    summary = rp.resolve_output(args.summary, None, SUMMARY_NAME)
    output = rp.resolve_output(args.output, rp.root_of_summary(summary), OUTPUT_NAME)
    if not summary.exists():
        raise SystemExit(f"summary CSV not found: {summary}. Run z8_CommonSetMetrics.py first.")

    df = pd.read_csv(summary)
    text, norm = build_frame(df, args.dataset_prefix)
    path = render(text, norm, output)
    print(f"[INFO] Wrote {path}")
    print(f"[INFO] {len(text)} targets; skill block populated for "
          f"{int((text['Verdict'] != '').sum())} of them.")
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

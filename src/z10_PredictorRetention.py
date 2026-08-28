"""How often each predictor was retained, counted across targets.

Replaces the cross-target average of Shapley attributions. Averaging an
importance score over targets asserts that a predictor has a single study-level
importance, which the data do not support: the targets have unrelated dynamics,
and a score that is large for one and negligible for another has no meaningful
mean. A count does not make that claim. "Retained in the best model for 9 of 14
targets" is a statement about this study that survives the objection.

The feature list is read from each run's ``model_config.json``, so it is the set
the model was actually fitted on, after the multiple-linear-regression variants
have applied their own internal selection.

Target-specific state columns are pooled into a single "prior target value"
entry: counted literally they would appear as fourteen distinct predictors of
count one each, which is the opposite of what the count is for.

Usage:
    python src/z10_PredictorRetention.py
    python src/z10_PredictorRetention.py --family ml --output <path.png>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plotstyle import PAGE_WIDTH_IN, apply_paper_style, save_figure
from utils import run_paths as rp

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = rp.DEFAULT_ROOT
SUMMARY_NAME = "common_set_metrics.csv"
OUTPUT_NAME = "predictor_retention_counts.png"

STATE_LABEL = "Prior target value"
PROFILER_PREFIX = "Pfl - "
ML_FAMILIES = {"GP", "XGB", "Transformer"}


def _run_features(root: Path, dataset: str, run: str) -> list[str] | None:
    cfg = root / dataset / "forecasts" / "feature_sweeps" / run / "model_config.json"
    if not cfg.exists():
        return None
    try:
        payload = json.loads(cfg.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    cols = payload.get("input_columns")
    return [str(c) for c in cols] if isinstance(cols, list) else None


def _normalize(name: str) -> str:
    return STATE_LABEL if name.endswith("_state") else name


def collect(root: Path, summary: pd.DataFrame, family_mode: str) -> tuple[Counter, list[str]]:
    counts: Counter = Counter()
    missing: list[str] = []
    n_targets = 0
    for _, r in summary.iterrows():
        dataset = str(r.get("dataset", ""))
        if family_mode == "ml":
            family = str(r.get("best_ml_family", "") or "").strip()
        else:
            family = str(r.get("best_family", "") or "").strip()
        if not family:
            missing.append(f"{dataset}: no {family_mode} winner recorded")
            continue
        run = r.get(f"{family.lower().replace('-', '')}_run")
        if not isinstance(run, str) or not run:
            # naive, seasonal and trend forecasts use no predictors, so there is
            # nothing to retain; that is a result, not a gap.
            missing.append(f"{dataset}: {family} uses no predictors")
            continue
        feats = _run_features(root, dataset, run)
        if feats is None:
            missing.append(f"{dataset}: no feature list for {run}")
            continue
        n_targets += 1
        for f in {_normalize(x) for x in feats}:
            counts[f] += 1
    return counts, missing


def render(counts: Counter, n_targets: int, output: Path, family_mode: str) -> Path:
    apply_paper_style()
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    height = max(3.0, 0.26 * len(labels) + 1.4)
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, height))
    y = np.arange(len(labels))
    colours = ["#4c72b0" if not l.startswith(PROFILER_PREFIX) else "#dd8452" for l in labels]
    ax.barh(y, values, color=colours, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Targets retaining the predictor (of {n_targets})")
    ax.set_xlim(0, max(n_targets, max(values) if values else 1))
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for yi, v in zip(y, values):
        ax.text(v + 0.08, yi, str(v), va="center", fontsize=7)

    handles = [plt.Rectangle((0, 0), 1, 1, color="#4c72b0"),
               plt.Rectangle((0, 0), 1, 1, color="#dd8452")]
    ax.legend(handles, ["Continuous coverage", "Lake profiler (seasonal coverage)"],
              loc="lower right", frameon=False, fontsize=7)

    path = save_figure(fig, output)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None,
                    help="Defaults to <root>/summaries/%s." % OUTPUT_NAME)
    ap.add_argument("--family", choices=["best", "ml"], default="best",
                    help="'best' counts the winning method of any family; 'ml' counts the "
                         "best learned model even where a reference forecast won.")
    args = ap.parse_args()

    root = rp.resolve_root(args.root)
    summary_path = rp.resolve_output(args.summary, root, SUMMARY_NAME)
    output = rp.resolve_output(args.output, root, OUTPUT_NAME)
    if not summary_path.exists():
        raise SystemExit(f"summary not found: {summary_path}. Run z8_CommonSetMetrics.py first.")

    summary = pd.read_csv(summary_path)
    counts, missing = collect(root, summary, args.family)
    if not counts:
        raise SystemExit("No feature lists could be read; nothing to count.")

    n_targets = len(summary) - len(missing)
    path = render(counts, n_targets, output, args.family)
    print(f"[INFO] Wrote {path}")
    print(f"[INFO] counted the '{args.family}' model for {n_targets} of {len(summary)} targets")
    for note in missing:
        print(f"[INFO]   excluded {note}")
    print()
    for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:2d} / {n_targets}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

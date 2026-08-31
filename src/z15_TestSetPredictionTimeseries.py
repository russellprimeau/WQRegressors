"""Predictions against ground truth on the pinned test set, one panel per target.

Table 3 reduces each target to a single number. That number says how close the
predictions were on average; it cannot say whether a model tracks the target through
time, misses one excursion badly, or merely reproduces the mean. This draws the
comparison the table summarises: the measured value, the best machine-learning model,
and the best reference forecast, on the segments each target was actually scored on.

Two versions are written, because the models predict a *change* rather than a level and
neither framing is complete on its own:

    ..._differential.png   the quantity the models were trained and scored on, so the
                           panel corresponds exactly to the R2 in Table 3.
    ..._absolute.png       that change added back to the preceding observation, in the
                           units of the existing target time-series figure. Easier to
                           read, but every series inherits the same prior value, so the
                           three curves agree more closely than the models' skill on the
                           differential warrants. Read the differential for performance
                           and this one for context.

The reference series is the best of the naive, seasonal and linear forecasts only. The
multiple linear regressions are excluded deliberately: this study treats them as
competing statistical models rather than as baselines, and putting one here would
present it as the thing to be beaten.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.names import clean_target_label, label as names_label  # noqa: E402
from utils.plotstyle import apply_paper_style, legend_above, save_figure  # noqa: E402
from utils.limits import load_limits_records, map_limits_to_columns  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
_LIMIT_RECORDS = load_limits_records(REPO_ROOT / "data" / "input" / "Limits.csv")
DEFAULT_ROOT = REPO_ROOT / "data" / "output" / "CV22_profilerless"

BASELINE_COLUMNS = {"naive": "Naive", "seasonal": "Seasonal", "linear": "Linear"}
META_COLUMNS = {
    "kind", "sample_file", "gp_uncertainty_mode", "metric_semantics",
    "metric_contract_version", "target",
}

# Drawn on a 13 in canvas that LaTeX scales to the 6.5 in text block, so every size
# here prints at half its value -- the convention the other target figures use.
TRUTH_STYLE = {"color": "#111111", "marker": "o", "ms": 4.0, "lw": 0.8, "zorder": 2.0}
ML_STYLE = {"color": "#1f77b4", "marker": "s", "ms": 3.6, "lw": 0.8, "zorder": 2.5}
# Magenta: black, blue and the two threshold colours (red, green) are already taken, and
# magenta is the one remaining hue that stays dark enough on white to read at this line
# width. The dashed style and triangle marker separate it from the others in greyscale.
REF_STYLE = {"color": "#c51b8a", "marker": "^", "ms": 3.6, "lw": 0.8,
             "ls": "--", "zorder": 3.0}

FIG_WIDTH_IN = 13.0
ROW_HEIGHT_IN = 0.88
BASE_FONT_PT = 14
ROW_LABEL_PT = int(round(BASE_FONT_PT * 1.1))
X_LABEL_PT = int(round(BASE_FONT_PT * 1.3))
LABEL_WRAP = 17
MAX_LABEL_LINES = 2


def _wrap_to_max_lines(text: str, width: int = LABEL_WRAP,
                       max_lines: int = MAX_LABEL_LINES) -> str:
    """Wrap a row label, widening rather than spilling past *max_lines*."""
    wrapped = textwrap.fill(text, width=width)
    while len(wrapped.splitlines()) > max_lines and width < 60:
        width += 1
        wrapped = textwrap.fill(text, width=width)
    return wrapped


def _denorm(values, spec) -> np.ndarray:
    """Map normalised values back to physical units."""
    v = np.asarray(values, dtype=float)
    if not spec:
        return v
    lo, hi = float(spec["min"]), float(spec["max"])
    return v * (hi - lo) + lo


def _prediction_column(columns) -> str | None:
    """The model's own prediction column, which sits immediately after ``target``."""
    cols = list(columns)
    after = cols[cols.index("target") + 1:] if "target" in cols else cols
    for c in after:
        if (c not in META_COLUMNS and c not in BASELINE_COLUMNS.values()
                and not str(c).endswith(("_std", "_var"))):
            return c
    return None


def _segment_timestamps(dataset_dir: Path, output_row: int) -> dict:
    """``{segment file: timestamp of the predicted observation}``."""
    out = {}
    for f in sorted((dataset_dir / "samples").glob("segment_*.csv")):
        try:
            ts = pd.read_csv(f, usecols=["TIMESTAMP"], parse_dates=["TIMESTAMP"],
                             encoding="utf-8", encoding_errors="replace")["TIMESTAMP"]
        except Exception:
            continue
        if len(ts) > output_row:
            out[f.name] = ts.iloc[output_row]
    return out


def _find_baseline_predictions(sweeps: Path, column: str, wanted: set):
    """Baseline predictions covering exactly *wanted*, from whichever run recorded them.

    A run only carries the reference columns if baselines were evaluated alongside it,
    and the best machine-learning run is often not one of those. Any run scored on the
    same segments gives the same reference forecast, so the first match is sufficient.
    """
    for run in sorted(p for p in sweeps.iterdir() if p.is_dir()):
        pred = run / "predictions.csv"
        if not pred.is_file():
            continue
        try:
            df = pd.read_csv(pred)
        except Exception:
            continue
        if column not in df.columns or "sample_file" not in df.columns:
            continue
        df = df[df.get("kind", "test").astype(str) == "test"]
        if set(df["sample_file"].astype(str)) != wanted:
            continue
        return df.set_index(df["sample_file"].astype(str))[column]
    return None


def collect_target(root: Path, row: pd.Series) -> dict | None:
    """Truth, best-ML and best-reference series for one target, in physical units."""
    dataset_dir = root / str(row["dataset"])
    sweeps = dataset_dir / "forecasts" / "feature_sweeps"
    if not sweeps.is_dir():
        return None

    family = str(row.get("best_ml_family", "") or "")
    run_col = {"GP": "gp_run", "XGB": "xgb_run", "Transformer": "transformer_run",
               "MLR": "mlr_run", "MLR-12": "mlr12_run", "MLR-All": "mlrall_run"}.get(family)
    run_name = str(row.get(run_col, "") or "") if run_col else ""
    pred_csv = sweeps / run_name / "predictions.csv"
    if not run_name or not pred_csv.is_file():
        print(f"[WARN] {row['target']}: no predictions for best model {family!r} ({run_name!r}).")
        return None

    df = pd.read_csv(pred_csv)
    df = df[df.get("kind", "test").astype(str) == "test"]
    ml_col = _prediction_column(df.columns)
    if ml_col is None or df.empty:
        return None
    seg = df["sample_file"].astype(str)

    # Best of the three reference forecasts, by the R2 already computed for them.
    scored = {k: float(row.get(f"{k}_r2", np.nan)) for k in BASELINE_COLUMNS}
    scored = {k: v for k, v in scored.items() if np.isfinite(v)}
    if not scored:
        print(f"[WARN] {row['target']}: no reference forecast scored; skipping.")
        return None
    ref_key = max(scored, key=scored.get)
    ref_series = _find_baseline_predictions(sweeps, BASELINE_COLUMNS[ref_key], set(seg))
    if ref_series is None:
        print(f"[WARN] {row['target']}: no run carries {BASELINE_COLUMNS[ref_key]!r} "
              "on the scored segments; skipping.")
        return None

    cfg = yaml.safe_load(open(sweeps / run_name /
                              f"config_evaluate_{run_name}.yml", encoding="utf-8"))
    data_cfg = cfg["data"]
    target_col = list(data_cfg["output_columns"])[0]
    output_row = int(list(data_cfg["output_rows"])[0])

    norms = json.load(open(dataset_dir / "normalization.json", encoding="utf-8"))
    diff_spec = norms.get(target_col)
    state_col = re.sub(r"_diff$", "_state", target_col)
    state_spec = norms.get(state_col)

    stamps = _segment_timestamps(dataset_dir, output_row)
    keep = [i for i, s in enumerate(seg) if s in stamps]
    if not keep:
        return None
    order = sorted(keep, key=lambda i: stamps[seg.iloc[i]])
    seg_ord = [seg.iloc[i] for i in order]

    truth = _denorm(df["target"].to_numpy()[order], diff_spec)
    ml = _denorm(df[ml_col].to_numpy()[order], diff_spec)
    ref = _denorm(ref_series.reindex(seg_ord).to_numpy(), diff_spec)

    # The preceding observation each change is measured from, for the absolute version.
    # The state column is a forward fill of the observed series: at the target row it
    # already carries the NEW measurement, and only the row before it holds the value the
    # change is taken from. That row is also the last one the model sees -- the input
    # window is slice(input_row_1, input_row_2), which excludes the target row -- so it
    # is the baseline the forecast is actually issued from.
    prior = np.full(len(seg_ord), np.nan)
    if state_spec is not None and output_row > 0:
        vals = []
        for s in seg_ord:
            t = pd.read_csv(dataset_dir / "samples" / s, encoding="utf-8",
                            encoding_errors="replace")
            vals.append(t[state_col].iloc[output_row - 1]
                        if state_col in t.columns else np.nan)
        prior = _denorm(vals, state_spec)

    short = clean_target_label(str(row["dataset"]), with_suffix=False)
    full = names_label(target_col, with_unit=True, qualified=False, with_suffix=False)
    # Thresholds are defined on the measured level, so they are looked up on the base
    # column, not the differential the models are trained on.
    base_col = re.sub(r"_diff$", "", target_col)
    spec = map_limits_to_columns([base_col], _LIMIT_RECORDS).get(base_col) or {}
    upper = spec.get("upper")
    lower = spec.get("lower")
    m = re.search(r"\(([^()]*)\)\s*$", str(full))
    unit_only = m.group(1) if m else ""
    return {
        "label": short,
        "short": short,
        "unit_only": unit_only,
        "unit_label": full,
        "times": [stamps[s] for s in seg_ord],
        "truth": truth, "ml": ml, "ref": ref, "prior": prior,
        "ml_family": family, "ref_name": BASELINE_COLUMNS[ref_key],
        "n": len(seg_ord),
        "row_label": _wrap_to_max_lines(str(full)),
        "limit_upper": float(upper) if upper is not None and pd.notna(upper) else None,
        "limit_lower": float(lower) if lower is not None and pd.notna(lower) else None,
    }


# The metals are sampled over one span of the record and everything else over another,
# and their concentrations differ from the microbiological counts by orders of magnitude.
# Drawn together, each group wastes most of its axes on the other group's range.
METAL_TARGETS = ("arsenic", "cadmium", "chromium", "lead", "nickel", "copper", "zinc")


def _is_metal(s: dict) -> bool:
    text = str(s["label"]).lower()
    return any(m in text for m in METAL_TARGETS)


def draw(series: list, out_path: Path, *, absolute: bool) -> Path:
    """One stacked panel per target, sharing the layout of the target time-series figure.

    Geometry, fonts, row labels, tick format and threshold lines are deliberately those
    of ``Target_timeseries_no_raster``; only the plotted series differ, so the two
    figures can be read against each other without re-learning the axes.
    """
    n_rows = len(series)
    fig, axes_raw = plt.subplots(
        n_rows, 1, sharex=True,
        figsize=(FIG_WIDTH_IN, max(2.8, ROW_HEIGHT_IN * n_rows)),
        gridspec_kw={"hspace": 0.08},
    )
    axes = [axes_raw] if n_rows == 1 else list(axes_raw)
    limits_in_view: set = set()

    for i, (ax, s) in enumerate(zip(axes, series)):
        off = s["prior"] if absolute else 0.0
        drawn = []
        if absolute and not np.isfinite(np.asarray(off, dtype=float)).any():
            ax.text(0.5, 0.5, "no prior observation recorded", transform=ax.transAxes,
                    ha="center", va="center", fontsize=BASE_FONT_PT)
            ax.set_yticks([])
        else:
            # Measured first, so the two forecasts overlay it rather than hide it.
            ax.plot(s["times"], s["truth"] + off, label="Measured value", **TRUTH_STYLE)
            ax.plot(s["times"], s["ml"] + off, label="Best machine-learning model",
                    **ML_STYLE)
            ax.plot(s["times"], s["ref"] + off, label="Best reference forecast",
                    **REF_STYLE)
            drawn = [s["truth"] + off, s["ml"] + off, s["ref"] + off]
            if not absolute:
                ax.axhline(0.0, color="#999999", linewidth=0.6, zorder=1.1)

        y_low = y_high = None
        finite = np.concatenate([np.asarray(v, dtype=float).ravel() for v in drawn])             if drawn else np.asarray([])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            y_low, y_high = float(finite.min()), float(finite.max())

        if y_low is not None and y_high is not None:
            span = y_high - y_low
            # The target figure floors this at an absolute 0.03 because it plots levels
            # of order 1. The differentials here run to 0.003, where that floor is five
            # times the whole span and flattens the series, so the pad stays relative.
            pad = (0.12 * max(abs(y_low), 1e-9) if span <= 0
                   else max(0.08 * span, 0.03 * max(abs(y_low), abs(y_high))))
            ax.set_ylim(y_low - pad, y_high + pad)

        # Thresholds apply to the measured level, so they go only on the absolute panels;
        # a change has no threshold defined on it. They are drawn after the limits are
        # fixed and deliberately do not widen them: most of these sit orders of magnitude
        # above the observed values, and letting them set the range compresses the data
        # into a flat line. A threshold outside a panel's range simply does not appear.
        if absolute and drawn:
            lo_v, hi_v = ax.get_ylim()
            for value, colour in ((s["limit_upper"], "#ff0000"),
                                  (s["limit_lower"], "#2ca02c")):
                if value is None:
                    continue
                ax.axhline(y=value, color=colour, linewidth=0.7, linestyle="-",
                           zorder=1.2)
                if lo_v <= value <= hi_v:
                    limits_in_view.add(colour)
            ax.set_ylim(lo_v, hi_v)

        ax.set_ylabel(s["row_label"], rotation=0, ha="right", va="center",
                      fontsize=ROW_LABEL_PT, labelpad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.4)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
        ax.tick_params(axis="y", labelsize=BASE_FONT_PT)
        ax.tick_params(axis="x", labelsize=X_LABEL_PT)
        if i < n_rows - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    stamps = [t for s in series for t in s["times"]]
    lo, hi = min(stamps), max(stamps)
    # The metals occupy about five months of the record and everything else about
    # thirteen. Quarterly ticks, which suit the full-record target figure, leave the
    # metals panel with a single label, so the interval follows the span on show.
    span_days = (pd.Timestamp(hi) - pd.Timestamp(lo)).days
    freq = "QS-JAN" if span_days > 400 else "MS"
    ticks = pd.date_range(start=pd.Timestamp(lo).to_period("Q").start_time,
                          end=hi, freq=freq)
    if len(ticks) >= 2:
        axes[-1].set_xticks(ticks)
        axes[-1].set_xticklabels([d.strftime("%Y-%m-%d") for d in ticks],
                                 rotation=25, ha="right", fontsize=X_LABEL_PT)
    # Without a pad the first and last markers sit on the frame and are clipped in half.
    x_pad = max((pd.Timestamp(hi) - pd.Timestamp(lo)) * 0.02, pd.Timedelta(days=1))
    axes[-1].set_xlim(pd.Timestamp(lo) - x_pad, pd.Timestamp(hi) + x_pad)

    handles, labels = axes[0].get_legend_handles_labels()
    for colour, text in (("#ff0000", "Strictest legal threshold (upper bound)"),
                         ("#2ca02c", "Strictest legal threshold (lower bound)")):
        if colour in limits_in_view:
            handles.append(plt.Line2D([0], [0], color=colour, linewidth=0.7,
                                      linestyle="-"))
            labels.append(text)
    # Three columns of the threshold labels overflow the 13 in canvas, and the tight
    # bounding box then grows with them -- which would print this figure at a different
    # scale from its companion. Two columns keep both bounding boxes at the canvas width.
    legend_above(axes[0], handles, labels, ncol=2 if len(handles) > 3 else 3,
                 fontsize=X_LABEL_PT)

    # Row labels sit outside the axes, so the left margin is sized from their width,
    # by the same rule the other stacked target figures use.
    widest = max((max(len(ln) for ln in s["row_label"].splitlines())
                  for s in series), default=1)
    left = (widest * max(0.045, 0.0055 * ROW_LABEL_PT) + 0.55) / FIG_WIDTH_IN
    fig.subplots_adjust(left=max(0.14, min(0.26, left)), right=0.995, hspace=0.08)
    return save_figure(fig, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--summary", type=Path, default=None,
                    help="common_set_metrics.csv; defaults to <root>/summaries/.")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Defaults to <root>/summaries/.")
    args = ap.parse_args()

    root = Path(args.root)
    summary = Path(args.summary) if args.summary else root / "summaries" / "common_set_metrics.csv"
    outdir = Path(args.outdir) if args.outdir else root / "summaries"
    if not summary.is_file():
        raise SystemExit(f"Summary not found: {summary}")

    apply_paper_style()
    df = pd.read_csv(summary)

    series = []
    for _, row in df.iterrows():
        s = collect_target(root, row)
        if s is not None:
            series.append(s)
            print("  %-34s n=%-3d best=%-12s reference=%s"
                  % (s["label"][:34], s["n"], s["ml_family"], s["ref_name"]))
    if not series:
        raise SystemExit("No target produced a usable series.")

    # Largest evaluation sets first, so the densest panels are read before the sparsest.
    series.sort(key=lambda s: -s["n"])

    outdir.mkdir(parents=True, exist_ok=True)
    groups = {
        "metals": [s for s in series if _is_metal(s)],
        "other": [s for s in series if not _is_metal(s)],
    }
    print()
    for group, members in groups.items():
        if not members:
            continue
        for absolute in (False, True):
            kind = "absolute" if absolute else "differential"
            out = draw(members,
                       outdir / f"test_set_predictions_{group}_{kind}.png",
                       absolute=absolute)
            print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

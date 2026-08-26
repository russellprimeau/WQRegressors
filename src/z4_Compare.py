"""
Generates a clustered bar chart comparing a summary statistic (e.g. skill vs.
best baseline) across matching datasets from two or more output directories.

Each output directory must contain a ``summaries/summary_best_model_performance.csv``
(written by ``z1_FeaturePostProcess.py``).  One group of bars is drawn per
dataset; each bar in a group corresponds to one supplied directory.

Default statistic is ``skill_vs_best_baseline``, computed on the fly as the
row-wise maximum of ``skill_vs_naive``, ``skill_vs_seasonal``, and
``skill_vs_linear``.  Any column present in the summary CSV can be specified
via ``--stat``.

By default only datasets that appear in **all** supplied directories are shown.
Pass ``--all-datasets`` to include datasets missing from some directories
(missing bars are simply omitted from those groups).

Outputs a single PNG to ``--output`` (default: ``compare_<stat>.png`` in the
current working directory).

CLI arguments:
    --root PATH          Output directory to include (repeat for each directory).
                         At least two must be supplied.
    --label STR          Display label for the corresponding --root (repeat in
                         the same order).  Defaults to the directory's basename.
    --stat STR           Column to compare.  Use ``skill_vs_best_baseline`` for
                         the row-wise max across the three baseline skills
                         (default), or any column present in the summary CSV.
    --dataset-prefix STR Strip this prefix when building dataset display labels
                         (default: MC).
    --summaries-subdir   Subdirectory under a root that holds the summary CSV
                         (default: summaries).  Repeat to give a different
                         subdirectory per --root, in the same order.
    --summary-file       Filename of the summary CSV (default:
                         summary_best_model_performance.csv).
    --sort               Sort datasets by the chosen --sort-by key (descending).
                         Default: alphabetical.
    --all-datasets       Include datasets missing from some roots (their bars
                         are omitted; intersection-only is the default).
    --output PATH        Output PNG path.  Defaults to compare_<stat>.png in
                         the current directory.
    --output-dir PATH    Directory in which to write the output PNG (ignored
                         when --output is an absolute path).

Examples:
python src/z3_Compare.py --root data/output/CV14 --root data/output/profileless
python src/z3_Compare.py --root data/output/CV14 --label "With profiles" --root data/output/CV15profileless --label "No profiles" --stat skill_vs_best_baseline --sort --output data/output/comparisons/profile/skill_vs_best_baseline.png
python src/z3_Compare.py --root data/output/CV14 --label "With state" --root data/output/CV16statelessless --label "No state" --stat skill_vs_best_baseline --sort --output data/output/comparisons/state/skill_vs_best_baseline.png
python src/z4_Compare.py --root data/output/CV14 --label "With state" --root data/output/CV16stateless --label "No state" --root data/output/CV15profileless --label "No Surface" --root data/output/CV18_raw --label "No residual" --stat r2 --sort --output data/output/comparisons/all/r2.png
"""
from __future__ import annotations

import argparse
import re as _re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# Ensure src/ is on the path when run as ``python src/z3_Compare.py``
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.names import clean_target_label
from utils.plotstyle import legend_above

# Colour palette matching the other z-scripts (avoids red/orange).
_SERIES_COLORS = [
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#8c564b",  # brown
    "#bcbd22",  # olive
    "#7f7f7f",  # gray
    "#aec7e8",  # light blue
    "#98df8a",  # light green
    "#c5b0d5",  # light purple
]

_SKILL_COLS = ("skill_vs_naive", "skill_vs_seasonal", "skill_vs_linear")
_SKILL_VS_BEST = "skill_vs_best_baseline"

# Suffixes stripped when normalising dataset names for cross-directory matching.
_NORM_SUFFIX_RE = _re.compile(r"(_diff|_res|_state)+$", _re.IGNORECASE)


def _normalize_dataset_name(name: str) -> str:
    """Return a canonical key for *name* used when matching across directories.

    Strips trailing ``_diff`` / ``_res`` / ``_state`` suffixes (any number of repetitions)
    and then strips any remaining trailing underscores, so that e.g.
    ``MC_Arsenic__µg_L__res`` and ``MC_Arsenic__µg_L_`` both map to
    ``MC_Arsenic__µg_L``.
    """
    return _NORM_SUFFIX_RE.sub("", name).rstrip("_")

# Display labels for known summary columns.  Falls back to underscore→space for
# anything not listed here.
_STAT_LABELS: dict[str, str] = {
    "skill_vs_best_baseline":        "Skill vs. Best Baseline",
    "skill_vs_naive":                "Skill vs. naive",
    "skill_vs_seasonal":             "Skill vs. seasonal",
    "skill_vs_linear":               "Skill vs. linear",
    "bootstrap_skill_mean_vs_naive":  "Bootstrap skill vs. naive (mean)",
    "bootstrap_skill_mean_vs_seasonal": "Bootstrap skill vs. seasonal (mean)",
    "bootstrap_skill_mean_vs_linear": "Bootstrap skill vs. linear (mean)",
    "lcb95_skill_vs_naive":          "Skill vs. naive (95 % LCB)",
    "lcb95_skill_vs_seasonal":       "Skill vs. seasonal (95 % LCB)",
    "lcb95_skill_vs_linear":         "Skill vs. linear (95 % LCB)",
    "r2":                            "R²",
    "nrmse":                         "nRMSE",
    "rmse":                          "RMSE",
    "pearson_r":                     "Pearson r",
    "rolling_cv_r2":                 "Rolling-CV R²",
    "rolling_cv_r2_median":          "Rolling-CV R² (median)",
    "evidence_score_overall_mean":   "Evidence score (mean)",
    "evidence_score_overall_min":    "Evidence score (min)",
}


def _load_summary(
    root: Path,
    summaries_subdir: str,
    summary_file: str,
) -> pd.DataFrame:
    """Load the best-model summary CSV from *root*.

    Raises ``FileNotFoundError`` when the file is absent.
    """
    path = root / summaries_subdir / summary_file
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        raise ValueError(f"'dataset' column missing in {path}")
    return df


def _add_skill_vs_best(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``skill_vs_best_baseline`` as the row-wise max of the three skills.

    Operates only on the columns that are actually present; returns *df*
    unchanged if none of the skill columns exist.
    """
    present = [c for c in _SKILL_COLS if c in df.columns]
    if not present:
        return df
    df = df.copy()
    df[_SKILL_VS_BEST] = df[present].max(axis=1)
    return df


def _format_bar_annotation(value: float, scientific: bool) -> str:
    """Return a compact annotation string with chart-wide scientific formatting when needed."""
    if scientific:
        s = f"{value:.2e}"
        mantissa, exp = s.split("e")
        return f"{mantissa}e{int(exp)}"
    return f"{value:.2f}"


def _expand_ylim_to_fit_annotations(ax: plt.Axes, pad_pixels: float = 2.0, max_passes: int = 3) -> None:
    """Expand y-limits just enough so existing annotation texts fit inside the axes box."""
    texts = [txt for txt in ax.texts if txt.get_visible()]
    if not texts:
        return
    fig = ax.figure
    for _ in range(max_passes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax.get_window_extent(renderer=renderer)
        top_over = 0.0
        bottom_over = 0.0
        for txt in texts:
            bbox = txt.get_window_extent(renderer=renderer)
            top_over = max(top_over, bbox.y1 - ax_bbox.y1)
            bottom_over = max(bottom_over, ax_bbox.y0 - bbox.y0)
        if top_over <= 0 and bottom_over <= 0:
            break
        x_ref = 0.5 * (ax_bbox.x0 + ax_bbox.x1)
        y_lo, y_hi = ax.get_ylim()
        new_y_lo = y_lo
        new_y_hi = y_hi
        if bottom_over > 0:
            new_y_lo = ax.transData.inverted().transform((x_ref, ax_bbox.y0 - bottom_over - pad_pixels))[1]
        if top_over > 0:
            new_y_hi = ax.transData.inverted().transform((x_ref, ax_bbox.y1 + top_over + pad_pixels))[1]
        if new_y_lo == y_lo and new_y_hi == y_hi:
            break
        ax.set_ylim(new_y_lo, new_y_hi)


def generate_figure(
    roots: list[Path],
    labels: list[str],
    stat_col: str,
    dataset_prefix: str,
    summaries_subdirs: list[str],
    summary_file: str,
    sort: bool,
    sort_by: str,
    intersection_only: bool,
    output_path: Path,
) -> int:
    """Load summaries, build the chart, and write *output_path*.

    Returns 0 on success, 1 on failure.
    """
    if len(roots) < 2:
        print("[ERROR] At least two --root directories are required.")
        return 1

    # ------------------------------------------------------------------
    # Load data — index by normalised name for cross-directory matching
    # ------------------------------------------------------------------
    # frames[label]: dict mapping normalised_name → stat value
    # norm_to_display: first-seen original name for each normalised key (for labels)
    frames: dict[str, dict[str, float]] = {}
    norm_to_original: dict[str, str] = {}

    for root, label, subdir in zip(roots, labels, summaries_subdirs):
        try:
            df = _load_summary(root, subdir, summary_file)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1

        # Auto-compute skill_vs_best_baseline if needed
        if stat_col == _SKILL_VS_BEST or stat_col not in df.columns:
            df = _add_skill_vs_best(df)

        if stat_col not in df.columns:
            available = ", ".join(df.columns.tolist())
            print(
                f"[ERROR] Column '{stat_col}' not found in summary for '{label}'.\n"
                f"        Available columns: {available}"
            )
            return 1

        # Keep only the best row per dataset (subset_rank == 1 when present)
        if "subset_rank" in df.columns:
            df = df.sort_values("subset_rank").groupby("dataset", as_index=False).first()
        else:
            df = df.groupby("dataset", as_index=False).first()

        mapping: dict[str, float] = {}
        for _, row in df.iterrows():
            orig = str(row["dataset"])
            norm = _normalize_dataset_name(orig)
            v = row[stat_col]
            mapping[norm] = float(v) if pd.notna(v) and np.isfinite(float(v)) else np.nan
            norm_to_original.setdefault(norm, orig)

        frames[label] = mapping
        print(f"[INFO] '{label}': {len(mapping)} dataset(s) loaded from {root}")

    # ------------------------------------------------------------------
    # Determine dataset set (on normalised keys)
    # ------------------------------------------------------------------
    all_norm_sets = [set(m.keys()) for m in frames.values()]
    if intersection_only:
        norm_pool = sorted(all_norm_sets[0].intersection(*all_norm_sets[1:]))
        if not norm_pool:
            print("[ERROR] No datasets appear in all supplied directories after name normalisation.")
            return 1
        excluded = all_norm_sets[0].union(*all_norm_sets[1:]) - set(norm_pool)
        if excluded:
            print(f"[INFO] {len(excluded)} dataset(s) excluded (not in all roots): "
                  f"{', '.join(sorted(excluded))}")
    else:
        norm_pool = sorted(all_norm_sets[0].union(*all_norm_sets[1:]))

    print(f"[INFO] Plotting {len(norm_pool)} dataset(s) across {len(roots)} root(s).")

    # Display labels derived from the first-seen original name per normalised key
    display_labels = [
        clean_target_label(norm_to_original.get(n, n), dataset_prefix)
        for n in norm_pool
    ]

    # Build value matrix: rows = datasets (normalised), cols = roots
    value_matrix = np.full((len(norm_pool), len(labels)), np.nan)
    for j, label in enumerate(labels):
        mapping = frames[label]
        for i, norm in enumerate(norm_pool):
            if norm in mapping:
                value_matrix[i, j] = mapping[norm]

    # ------------------------------------------------------------------
    # Optional sort
    # ------------------------------------------------------------------
    if sort:
        if sort_by == "first":
            sort_vals = value_matrix[:, 0]
        elif sort_by == "diff":
            sort_vals = value_matrix[:, -1] - value_matrix[:, 0]
        else:
            sort_vals = np.nanmean(value_matrix, axis=1)
        order = np.argsort(sort_vals)[::-1]
        norm_pool      = [norm_pool[k]      for k in order]
        display_labels = [display_labels[k] for k in order]
        value_matrix   = value_matrix[order]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    n_datasets = len(norm_pool)
    n_roots    = len(labels)
    # Use contiguous bars within each cluster, with an inter-cluster gap equal
    # to half of one bar width.  Since dataset centres are one x-unit apart,
    # solve n_roots * bar_w + 0.5 * bar_w = 1 for bar_w.
    bar_w      = 1.0 / (n_roots + 0.5)
    font_size  = 9
    x          = np.arange(n_datasets)

    fig_w = max(7.0, 0.85 * n_datasets + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    finite_vals = value_matrix[np.isfinite(value_matrix)]
    use_scientific = bool(finite_vals.size and np.any(np.abs(finite_vals) < 0.01))

    for j, (label, color) in enumerate(zip(labels, _SERIES_COLORS)):
        offsets = x + (j - (n_roots - 1) / 2.0) * bar_w
        vals = value_matrix[:, j]
        # Draw bars; NaN values produce zero-height bars, so skip them explicitly
        for xi, v in zip(offsets, vals):
            if np.isfinite(v):
                ax.bar(xi, v, width=bar_w, color=color, label=label if xi == offsets[0] else "")
                # Value label: above positive bars, below negative bars
                va = "bottom" if v >= 0 else "top"
                y_offset = 0.005 * (1 if v >= 0 else -1)
                ax.text(xi, v + y_offset, _format_bar_annotation(v, use_scientific), ha="center", va=va,
                        fontsize=font_size, rotation=90)

    # Zero reference line
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    # Keep edge margins small and proportional to the inter-cluster spacing.
    half_cluster = n_roots * bar_w / 2
    ax.set_xlim(x[0] - half_cluster - 0.25 * bar_w, x[-1] + half_cluster + 0.25 * bar_w)
    stat_ylabel = _STAT_LABELS.get(stat_col, stat_col.replace("_", " "))
    ax.set_ylabel(stat_ylabel, fontsize=font_size)
    ax.grid(axis="y", alpha=0.3)

    # Legend — deduplicate entries; it is placed in the margin above the axes rather than
    # inside them, so it can never sit on top of a bar or its value annotation.
    handles, leg_labels = ax.get_legend_handles_labels()
    seen: dict[str, int] = {}
    unique_h, unique_l = [], []
    for h, l in zip(handles, leg_labels):
        if l not in seen:
            seen[l] = 1
            unique_h.append(h)
            unique_l.append(l)

    fig.tight_layout()
    _expand_ylim_to_fit_annotations(ax)
    legend_above(ax, unique_h, unique_l, ncol=min(len(unique_h), 4), fontsize=font_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[INFO] Wrote comparison chart: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clustered bar chart comparing a summary statistic across "
            "datasets in two or more output directories."
        )
    )
    parser.add_argument(
        "--root",
        dest="roots",
        action="append",
        metavar="PATH",
        default=[],
        help=(
            "Output directory containing a summaries/ subdirectory. "
            "Repeat this flag for each directory to compare."
        ),
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        metavar="STR",
        default=[],
        help=(
            "Display label for the corresponding --root (repeat in the same order). "
            "Defaults to the directory basename."
        ),
    )
    parser.add_argument(
        "--stat",
        type=str,
        default=_SKILL_VS_BEST,
        help=(
            f"Column to compare (default: {_SKILL_VS_BEST}, computed as the "
            "row-wise max of skill_vs_naive / seasonal / linear). "
            "Any column present in the summary CSV is accepted."
        ),
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="MC",
        help="Prefix stripped when building dataset display labels (default: MC).",
    )
    parser.add_argument(
        "--summaries-subdir",
        dest="summaries_subdirs",
        action="append",
        type=str,
        metavar="NAME",
        help=(
            "Subdirectory under a root that contains the summary CSV (default: summaries). "
            "Repeat to give a different subdirectory per --root, in the same order; roots "
            "beyond the last value given fall back to the default. Supply once to apply "
            "the same subdirectory to every root."
        ),
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="summary_best_model_performance.csv",
        help="Filename of the summary CSV (default: summary_best_model_performance.csv).",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort datasets by the chosen --sort-by key (descending).",
    )
    parser.add_argument(
        "--sort-by",
        choices=["first", "diff", "mean"],
        default="first",
        help=(
            "Key used when --sort is active.  "
            "'first' (default) sorts by the stat value of the first --root; "
            "'diff' sorts by root[-1] minus root[0]; "
            "'mean' sorts by the mean across all roots."
        ),
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help=(
            "Include datasets missing from some roots (their bars are omitted). "
            "Default: show only datasets present in every root."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output PNG path.  Defaults to compare_<stat>.png in --output-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for the output PNG when --output is not an absolute path "
            "(default: current working directory)."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if len(args.roots) < 2:
        print("[ERROR] Supply at least two --root directories.")
        build_parser().print_help()
        return 1

    workspace_root = Path(__file__).resolve().parent.parent

    def _resolve(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = (workspace_root / path).resolve()
        return path

    roots  = [_resolve(r) for r in args.roots]
    labels = list(args.labels)
    # Pad missing labels with directory basenames
    while len(labels) < len(roots):
        labels.append(roots[len(labels)].name)

    # Summary subdirectory, per root.  Roots do not always agree: a partial re-run can
    # leave one root's ``summaries/`` holding a subset of the datasets while the complete
    # set survives alongside it, and the comparison then needs a different subdirectory
    # for that root alone.
    supplied_subdirs = list(args.summaries_subdirs or [])
    if len(supplied_subdirs) == 1:
        summaries_subdirs = supplied_subdirs * len(roots)
    else:
        summaries_subdirs = supplied_subdirs[: len(roots)]
        summaries_subdirs += ["summaries"] * (len(roots) - len(summaries_subdirs))

    # Resolve output path
    stat_slug = args.stat.replace(" ", "_")
    default_filename = f"compare_{stat_slug}.png"
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            base_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
            output_path = base_dir / output_path
    else:
        base_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd()
        output_path = base_dir / default_filename

    for root, label, subdir in zip(roots, labels, summaries_subdirs):
        print(f"[INFO] root '{label}': {root} ({subdir}/)")
    print(f"[INFO] stat  : {args.stat}")
    print(f"[INFO] output: {output_path}")

    return generate_figure(
        roots=roots,
        labels=labels,
        stat_col=args.stat,
        dataset_prefix=args.dataset_prefix,
        summaries_subdirs=summaries_subdirs,
        summary_file=args.summary_file,
        sort=args.sort,
        sort_by=args.sort_by,
        intersection_only=not args.all_datasets,
        output_path=output_path,
    )


if __name__ == "__main__":
    sys.exit(main())

"""Compute and visualise correlations between predictor parameters (profiler,
weather, SCADA) and Eurofins lab-analysis target parameters from
Consolidated_sparse.csv.

Produces two figures:
  1. Annotated heatmap of Pearson correlations
  2. Grid of scatterplots (one per predictor-target pair)

Usage:
    python src/utils/correlate_profiler_targets.py
"""

import sys
from pathlib import Path

# Allow `from utils.X import ...` regardless of the current working directory
# when this script is invoked directly.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from utils.plausibility import apply_plausibility_bounds

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CSV_PATH = _PROJECT_ROOT / "data" / "output" / "regression" / "Consolidated_sparse.csv"
_OUT_DIR = _PROJECT_ROOT / "data" / "output" / "sensors" / "correlation"
_HEATMAP_PEARSON_PATH = _OUT_DIR / "predictor_target_correlations.png"
_HEATMAP_SPEARMAN_PATH = _OUT_DIR / "predictor_target_correlations_spearman.png"
_PROFILER_PEARSON_PATH = _OUT_DIR / "profiler_correlations_pearson.png"
_PROFILER_SPEARMAN_PATH = _OUT_DIR / "profiler_correlations_spearman.png"

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
PROFILER_COLS = [
    "Pfl - Water temperature (°C)",
    "Pfl - Sp Cond (microS_cm)",
    "Pfl - pH",
    "Pfl - DO (% Sat)",
    "Pfl - Turbidity (FNU)",
    "Pfl - fDOM (RFU)",
    "Pfl - fDOM (QSU)",
]

WEATHER_COLS = [
    "Wind speed x (m/s)",
    "Wind speed y (m/s)",
    "Atmospheric pressure (mBar)",
    "Longwave (IR) radiation (W/m2)",
    "Shortwave (solar) radiation (W/m2)",
    "24hr precipitation total (mm)",
    "Air temperature (°C)",
    "Humidity (%)",
]

SCADA_COLS = [
    "SCADA - pH",
    "SCADA - Temperature (°C)",
]

ALL_PREDICTORS = PROFILER_COLS + WEATHER_COLS + SCADA_COLS

PREDICTOR_GROUPS = {
    "profiler": ("Profiler", PROFILER_COLS),
    "weather": ("Weather", WEATHER_COLS),
    "scada": ("SCADA", SCADA_COLS),
}

EUROFINS_TARGETS = [
    "Color",
    "Turbidity (FNU)",
    "E.coli (CFU/100mL)",
    "Intestinal enterococci (CFU/100mL)",
    "Colony Count 22°C (CFU/mL)",
    "Total coliforms 37°C (CFU/100mL)",
    "Arsenic (µg/L)",
    "Lead (µg/L)",
    "Cadmium (µg/L)",
    "Copper filtered (mg/L)",
    "Chromium (µg/L)",
    "Nickel (µg/L)",
    "Zinc (µg/L)",
    "pH",
]

# Short display labels
_PREDICTOR_LABELS = {
    c: c.replace("Pfl - ", "").replace("SCADA - ", "SCADA ") for c in ALL_PREDICTORS
}
_TARGET_LABELS = EUROFINS_TARGETS


def _short_label(col: str) -> str:
    return _PREDICTOR_LABELS.get(col, col)


def compute_correlations(df: pd.DataFrame):
    """Compute Pearson and Spearman correlations with p-values.

    Returns
    -------
    pearson_r : DataFrame  — Pearson correlation coefficients
    pearson_p : DataFrame  — two-sided p-values for the Pearson t-test
    spearman_r : DataFrame — Spearman rank correlation coefficients
    spearman_p : DataFrame — two-sided p-values for the Spearman test
    """
    empty = lambda: pd.DataFrame(index=ALL_PREDICTORS, columns=EUROFINS_TARGETS, dtype=float)
    pearson_r, pearson_p = empty(), empty()
    spearman_r, spearman_p = empty(), empty()

    for target in EUROFINS_TARGETS:
        target_valid = df[target].dropna().index
        for pred in ALL_PREDICTORS:
            pair = df.loc[target_valid, [pred, target]].dropna()
            if len(pair) >= 3:
                # Correlation is undefined when either input has no variation.
                if pair[pred].nunique(dropna=True) < 2 or pair[target].nunique(dropna=True) < 2:
                    continue
                r, p = stats.pearsonr(pair[pred], pair[target])
                pearson_r.loc[pred, target] = r
                pearson_p.loc[pred, target] = p
                rho, sp = stats.spearmanr(pair[pred], pair[target])
                spearman_r.loc[pred, target] = rho
                spearman_p.loc[pred, target] = sp

    return pearson_r, pearson_p, spearman_r, spearman_p


def compute_profiler_autocorrelation(df: pd.DataFrame):
    """Compute Pearson and Spearman correlations among PROFILER_COLS.

    Returns
    -------
    pearson_r, pearson_p, spearman_r, spearman_p : DataFrame
        7x7 matrices indexed and columned by PROFILER_COLS. Diagonal is 1.0
        (and p=0). Off-diagonal cells use pairwise dropna; cells without
        enough data or without variation in either input remain NaN.
    """
    empty = lambda: pd.DataFrame(index=PROFILER_COLS, columns=PROFILER_COLS, dtype=float)
    pearson_r, pearson_p = empty(), empty()
    spearman_r, spearman_p = empty(), empty()

    for i, a in enumerate(PROFILER_COLS):
        for j, b in enumerate(PROFILER_COLS):
            if i == j:
                pearson_r.loc[a, b] = 1.0
                pearson_p.loc[a, b] = 0.0
                spearman_r.loc[a, b] = 1.0
                spearman_p.loc[a, b] = 0.0
                continue
            if j < i:
                # Symmetric — copy from the other triangle
                pearson_r.loc[a, b] = pearson_r.loc[b, a]
                pearson_p.loc[a, b] = pearson_p.loc[b, a]
                spearman_r.loc[a, b] = spearman_r.loc[b, a]
                spearman_p.loc[a, b] = spearman_p.loc[b, a]
                continue

            pair = df[[a, b]].dropna()
            if len(pair) < 3:
                continue
            if pair[a].nunique(dropna=True) < 2 or pair[b].nunique(dropna=True) < 2:
                continue
            r, p = stats.pearsonr(pair[a], pair[b])
            pearson_r.loc[a, b] = r
            pearson_p.loc[a, b] = p
            rho, sp = stats.spearmanr(pair[a], pair[b])
            spearman_r.loc[a, b] = rho
            spearman_p.loc[a, b] = sp

    return pearson_r, pearson_p, spearman_r, spearman_p


def _significance_stars(p: float) -> str:
    """Return asterisk string for a p-value: *** < 0.001, ** < 0.01, * < 0.05."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _build_annot_array(corr: pd.DataFrame, pvals: pd.DataFrame) -> np.ndarray:
    """Build a string annotation array: correlation value + significance stars."""
    annot = np.empty(corr.shape, dtype=object)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            r = corr.iloc[i, j]
            p = pvals.iloc[i, j]
            if np.isnan(r):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{r:.2f}{_significance_stars(p)}"
    return annot


def plot_correlation_matrix(corr: pd.DataFrame, pvals: pd.DataFrame,
                           out_path: Path) -> None:
    """Save an annotated heatmap of predictor-target correlations.

    Cell annotations show the correlation coefficient plus significance
    asterisks derived from *pvals* (* p<0.05, ** p<0.01, *** p<0.001).
    """
    sns.set_style("whitegrid")
    n_pred = len(ALL_PREDICTORS)
    fig, ax = plt.subplots(figsize=(16, max(8, n_pred * 0.5)))

    y_labels = [_short_label(c) for c in corr.index]
    annot = _build_annot_array(corr, pvals)

    sns.heatmap(
        corr.astype(float),
        annot=annot,
        fmt="",
        cmap=_CORR_CMAP,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        xticklabels=_TARGET_LABELS,
        yticklabels=y_labels,
        ax=ax,
    )

    # Draw horizontal lines to separate predictor groups
    group_boundaries = [len(PROFILER_COLS), len(PROFILER_COLS) + len(WEATHER_COLS)]
    for y in group_boundaries:
        ax.axhline(y, color="black", linewidth=2)

    ax.set_xlabel("Eurofins Target Parameter")
    ax.set_ylabel("Predictor Parameter")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved heatmap to {out_path}")
    # plt.show()


# Custom diverging colormap: deep blue (-1) → warm amber (0) → deep red (+1).
# Every value maps to a saturated, clearly visible colour.
_CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr_blue_amber_red",
    [(0.0, "#1b4f72"),    # -1  dark blue
     (0.25, "#2e86c1"),   # -0.5 mid blue
     (0.5, "#d4a017"),    #  0  amber
     (0.75, "#cb4335"),   #  0.5 mid red
     (1.0, "#78281f")],   #  1  dark red
    N=256,
)
_CORR_NORM = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)


def plot_scatter_group(df: pd.DataFrame, corr: pd.DataFrame,
                       group_key: str, out_dir: Path) -> None:
    """Save a scatterplot grid for one predictor group (targets on y-axis)."""
    _, pred_cols = PREDICTOR_GROUPS[group_key]
    n_pred = len(pred_cols)
    n_tgt = len(EUROFINS_TARGETS)
    cell = 1.8  # side length per subplot (square)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        n_tgt, n_pred, figsize=(n_pred * cell, n_tgt * cell),
        squeeze=False,
    )

    for j, pred in enumerate(pred_cols):
        for i, target in enumerate(EUROFINS_TARGETS):
            ax = axes[i, j]
            ax.set_aspect("auto")
            pair = df[[pred, target]].dropna()
            n = len(pair)

            r = corr.loc[pred, target] if pred in corr.index else np.nan
            colour = _CORR_CMAP(_CORR_NORM(r)) if np.isfinite(r) else "0.55"

            if n >= 2:
                ax.scatter(pair[pred], pair[target], s=10, alpha=0.7,
                           color=colour, edgecolors="black", linewidths=0.3)
            ax.text(0.97, 0.95, f"n={n}", ha="right", va="top",
                    transform=ax.transAxes, fontsize=6, color="grey")

            # Column headers (predictor) on top row only
            if i == 0:
                ax.set_title(_short_label(pred), fontsize=8)
            # Row labels (target) on left column only
            if j == 0:
                ax.set_ylabel(_TARGET_LABELS[i], fontsize=7)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            # X tick labels on bottom row only
            if i < n_tgt - 1:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=5)

            # Force square aspect after data limits are set
            ax.set_box_aspect(1)

    plt.tight_layout()
    out_path = out_dir / f"scatter_{group_key}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {group_key} scatterplots to {out_path}")
    # plt.show()


def plot_profiler_pairplot(df: pd.DataFrame, corr: pd.DataFrame,
                           pvals: pd.DataFrame, out_path: Path,
                           method_label: str) -> None:
    """Save a profiler pair plot: scatter (lower) / hist (diag) / r (upper).

    Lower triangle: scatterplots of column pairs in a neutral color.
    Diagonal: histograms of each profiler variable.
    Upper triangle: cells filled by the diverging colormap and annotated
    with the correlation coefficient + significance stars.
    """
    sns.set_style("white")
    n = len(PROFILER_COLS)
    cell = 1.5
    label_fontsize = 9
    fig, axes = plt.subplots(n, n, figsize=(n * cell, n * cell), squeeze=False)
    fig.subplots_adjust(wspace=0, hspace=0, left=0.08, right=0.98,
                        top=0.95, bottom=0.08)

    for i, row_col in enumerate(PROFILER_COLS):
        for j, col_col in enumerate(PROFILER_COLS):
            ax = axes[i, j]

            if i == j:
                series = df[row_col].dropna()
                if len(series) > 0:
                    ax.hist(series, bins=30, color="0.45", edgecolor="black",
                            linewidth=0.3)
                ax.set_yticks([])
                ax.tick_params(labelsize=label_fontsize)
            elif i > j:
                # Lower triangle: scatter (x=col_col, y=row_col)
                pair = df[[col_col, row_col]].dropna()
                if len(pair) >= 2:
                    ax.scatter(pair[col_col], pair[row_col], s=10, alpha=0.7,
                               color="0.3", edgecolors="black", linewidths=0.3)
                ax.tick_params(labelsize=label_fontsize)
            else:
                # Upper triangle: filled cell with r + stars
                r = corr.loc[row_col, col_col]
                p = pvals.loc[row_col, col_col]
                if np.isfinite(r):
                    ax.set_facecolor(_CORR_CMAP(_CORR_NORM(r)))
                    ax.text(0.5, 0.5, f"{r:.2f}{_significance_stars(p)}",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=label_fontsize + 2, color="white",
                            fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])

            # Edge labels: only on outer rim, hide interior tick labels so
            # there is no whitespace eaten by tick marks between cells.
            if j == 0:
                ax.set_ylabel(_short_label(row_col), fontsize=label_fontsize)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(_short_label(col_col), fontsize=label_fontsize)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved profiler {method_label} pair plot to {out_path}")


def main() -> None:
    df = pd.read_csv(_CSV_PATH, parse_dates=["TIMESTAMP"], index_col="TIMESTAMP")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify expected columns are present
    missing_pred = [c for c in ALL_PREDICTORS if c not in df.columns]
    missing_tgt = [c for c in EUROFINS_TARGETS if c not in df.columns]
    if missing_pred:
        raise ValueError(f"Missing predictor columns: {missing_pred}")
    if missing_tgt:
        raise ValueError(f"Missing target columns: {missing_tgt}")

    # Apply per-column plausibility bounds. The consolidated CSV has been
    # filtered once during consolidation, but linear-interpolated rows in
    # clean_profiler can still fall outside sensor ranges; re-applying here
    # also catches any future drift in the upstream cleaning logic.
    before_nans = df.isna().sum().sum()
    df = apply_plausibility_bounds(df)
    after_nans = df.isna().sum().sum()
    if after_nans > before_nans:
        print(f"Plausibility filter nulled {after_nans - before_nans} additional cells.")

    pearson_r, pearson_p, spearman_r, spearman_p = compute_correlations(df)
    print(pearson_r.to_string())
    plot_correlation_matrix(pearson_r, pearson_p, _HEATMAP_PEARSON_PATH)
    plot_correlation_matrix(spearman_r, spearman_p, _HEATMAP_SPEARMAN_PATH)
    for group_key in PREDICTOR_GROUPS:
        plot_scatter_group(df, pearson_r, group_key, _OUT_DIR)

    pr, pp, sr, sp = compute_profiler_autocorrelation(df)
    plot_profiler_pairplot(df, pr, pp, _PROFILER_PEARSON_PATH, "Pearson")
    plot_profiler_pairplot(df, sr, sp, _PROFILER_SPEARMAN_PATH, "Spearman")


if __name__ == "__main__":
    main()

"""Shared figure styling for report-grade output.

Every figure in this repository is ultimately included in ``docs/report/draft/manuscript.tex``
with ``\\includegraphics[width=1\\linewidth]{...}``.  The MDPI single-column body is
``PAGE_WIDTH_IN`` inches wide, so a figure drawn wider than that is *downscaled* by LaTeX,
and every point of text in it shrinks by the same factor::

    printed_pt = font_pt * (PAGE_WIDTH_IN / fig_width_in)

This is the single reason report figures come out illegible: not the font choice, but a
figure drawn 36 inches wide with 16 pt text, which lands on the page at 2.9 pt.
:func:`check_printed_font_size` turns that arithmetic into a check the code performs on
itself rather than something to re-audit by eye.

Use::

    from utils.plotstyle import apply_paper_style, save_figure

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.2))
    ...
    save_figure(fig, out_path)      # strips titles, checks font size, saves
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt

__all__ = [
    "PAGE_WIDTH_IN",
    "FIGURE_DPI",
    "MIN_PRINTED_PT",
    "apply_paper_style",
    "clear_figure_titles",
    "timeseries_figsize",
    "timeseries_font_sizes",
    "annotate_bars",
    "LEGEND_GAP_IN",
    "legend_above",
    "check_printed_font_size",
    "save_figure",
]

# MDPI single-column text block; figures are included at width=1\linewidth.
PAGE_WIDTH_IN = 6.5

# Standardised output resolution.  Previously 300 in the calibration/summary scripts,
# 220 in the report timeseries, 180 in the sweep diagnostics and 150-200 in older
# utilities; there is no reason for report figures to differ.
FIGURE_DPI = 300

# Smallest text size considered legible in print.
MIN_PRINTED_PT = 7.0

# Clear vertical gap between the top of the plotting area and the bottom of the legend.
LEGEND_GAP_IN = 0.12

# Base sizes, chosen for a figure drawn at PAGE_WIDTH_IN (i.e. no LaTeX downscaling).
_BASE_FONT_PT = 8
_BASE_LABEL_PT = 9


def apply_paper_style() -> None:
    """Install the shared rcParams.

    Deliberately sets ``axes.titlesize`` small rather than removing titles: titles are
    stripped at save time by :func:`clear_figure_titles`, because captions live in LaTeX.
    """
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": _BASE_FONT_PT,
        "axes.labelsize": _BASE_LABEL_PT,
        "axes.titlesize": _BASE_LABEL_PT,
        "xtick.labelsize": _BASE_FONT_PT,
        "ytick.labelsize": _BASE_FONT_PT,
        "legend.fontsize": _BASE_FONT_PT,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,
        "axes.grid": False,
        "figure.autolayout": False,
    })


def clear_figure_titles(fig) -> None:
    """Remove all titles from a figure and its axes.

    Lifted from ``c2_uncertainty.py`` so that the whole repository shares one
    implementation.  Figure captions belong in the LaTeX document, so in-figure titles
    would duplicate them.
    """
    fig.suptitle("")
    for ax in fig.axes:
        ax.set_title("")


def timeseries_figsize(n_rows: int, *, width: float = PAGE_WIDTH_IN) -> tuple[float, float]:
    """Figure size for the stacked single-column timeseries figures.

    Preserves the proportions established in ``b_ExploreData.py`` (``row_height = 0.88``,
    ``min_fig_height = 2.8`` at ``fig_width = 13.0``) while rendering at page width, so
    the text is drawn at the size it will actually print at.
    """
    scale = width / 13.0
    row_height = 0.88 * scale
    min_height = 2.8 * scale
    return (width, max(min_height, row_height * max(1, n_rows)))


def timeseries_font_sizes() -> dict[str, int]:
    """Font sizes for the stacked timeseries figures, in the proportions used previously.

    ``b_ExploreData.py`` drew axis labels at 1.5x the base size and tick values at 1.0x.
    That relationship is kept; only the absolute values change, because the figure is no
    longer downscaled by LaTeX.
    """
    return {
        "axis_label": int(round(_BASE_LABEL_PT * 1.15)),
        "tick_value": _BASE_FONT_PT,
        "legend": _BASE_FONT_PT,
    }


def annotate_bars(
    ax,
    bars,
    values,
    *,
    fmt: str = ".2f",
    fontsize: int | None = None,
    rotation: int = 90,
) -> None:
    """Label bars with their values, in the style ``r2.png`` already uses successfully.

    Fixed-decimal rather than scientific notation, rotated so the label is no wider than
    the bar, and placed inside the bar when it would otherwise overflow the axes.
    """
    if fontsize is None:
        fontsize = plt.rcParams["font.size"] - 1
    ymin, ymax = ax.get_ylim()
    span = float(ymax - ymin)
    if not (span > 0):
        span = 1.0
    pad = 0.02 * span

    for bar, val in zip(bars, values):
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if fval != fval:  # NaN
            continue
        height = bar.get_height()
        if height < ymin or height > ymax:
            continue
        y_txt = height + pad
        va = "bottom"
        if y_txt > (ymax - pad):
            y_txt = height - pad
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_txt,
            format(fval, fmt),
            ha="center",
            va=va,
            fontsize=fontsize,
            rotation=rotation,
            clip_on=False,
        )


def legend_above(
    target,
    handles=None,
    labels=None,
    *,
    ncol: int | None = None,
    fontsize=None,
    gap_in: float = LEGEND_GAP_IN,
    frameon: bool = False,
    **kwargs,
):
    """Place a legend in the margin above the plotting area.

    The repository's legends were anchored by their *top* edge
    (``loc="upper center", bbox_to_anchor=(0.5, 1.03)``), so a legend grew *downwards*
    into the axes as soon as it gained a row or a larger font — which is why nearly every
    figure had its legend sitting on top of the first subplot.  Anchoring the *bottom*
    edge a fixed distance above the topmost axes inverts that: the legend grows away from
    the plot, and the clearance is the same no matter how tall the legend turns out to be.

    ``target`` may be a Figure or an Axes; either way the legend is attached to the figure
    so ``bbox_inches="tight"`` keeps it.  Call this *after* ``subplots_adjust`` /
    ``tight_layout``, since the anchor is read from the final axes positions.
    """
    fig = getattr(target, "figure", target)
    positions = [ax.get_position().y1 for ax in fig.axes if ax.get_visible()]
    top = max(positions) if positions else 1.0

    fig_height_in = float(fig.get_size_inches()[1]) or 1.0
    y_anchor = top + (gap_in / fig_height_in)

    if handles is None:
        source = target if target is not fig else (fig.axes[0] if fig.axes else fig)
        handles, auto_labels = source.get_legend_handles_labels()
        if labels is None:
            labels = auto_labels
    if ncol is None:
        ncol = max(1, len(handles))
    if fontsize is not None:
        kwargs["fontsize"] = fontsize

    # Passed by keyword: a lone positional list of Artists is read as a list of labels.
    kwargs["handles"] = handles
    if labels is not None:
        kwargs["labels"] = labels
    return fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=ncol,
        frameon=frameon,
        borderaxespad=0.0,
        **kwargs,
    )


def check_printed_font_size(fig, *, min_pt: float = MIN_PRINTED_PT) -> float:
    """Warn if the figure's text would print below ``min_pt``.

    Returns the smallest printed point size found.  A figure narrower than the text block
    is not upscaled by ``width=1\\linewidth`` in any way that hurts, so the scale factor is
    capped at 1.
    """
    fig_width_in = float(fig.get_size_inches()[0])
    if fig_width_in <= 0:
        return float("inf")
    scale = min(1.0, PAGE_WIDTH_IN / fig_width_in)

    sizes = [t.get_fontsize() for t in fig.findobj(plt.Text) if t.get_text().strip()]
    if not sizes:
        return float("inf")

    smallest_printed = min(sizes) * scale
    if smallest_printed < min_pt:
        warnings.warn(
            f"Figure is {fig_width_in:.1f} in wide and will be downscaled to "
            f"{PAGE_WIDTH_IN} in; its smallest text would print at "
            f"{smallest_printed:.1f} pt (minimum {min_pt} pt). "
            f"Either narrow the figure or raise its font sizes.",
            stacklevel=2,
        )
    return smallest_printed


def save_figure(fig, path, *, dpi: int = FIGURE_DPI, check: bool = True, **kwargs):
    """Strip titles, check legibility, and save.

    The single exit point report figures should use, so that the no-titles rule and the
    printed-size rule cannot be forgotten at an individual call site.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clear_figure_titles(fig)
    if check:
        check_printed_font_size(fig)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.02)
    fig.savefig(path, dpi=dpi, **kwargs)
    return path

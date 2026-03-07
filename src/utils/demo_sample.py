"""
Create schematic sample-layout rasters for explaining d_Resample.py outputs.

Outputs two figures to data/output:
  - demo_sample_full_coverage.png
  - demo_sample_missing_predictors.png
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _build_labels() -> tuple[list[str], list[str], int, int]:
    y_labels = ["Predictor 1", "Predictor 2", "Predictor 3", "...", "Predictor n", "Target State", "Target", "Target Diff"]
    x_labels = ["Timestep 1", "Timestep 2", "Timestep 3", "...", "Timestep m-1", "Target Timestep"]
    predictor_rows = 5  # includes the ellipsis row
    history_cols = 5  # includes the ellipsis column
    return y_labels, x_labels, predictor_rows, history_cols


def _is_ellipsis_index(idx: int) -> bool:
    return idx == 3


def _is_predictor_row(row_idx: int, predictor_rows: int) -> bool:
    return row_idx < predictor_rows and not _is_ellipsis_index(row_idx)


def _is_history_col(col_idx: int, history_cols: int) -> bool:
    return col_idx < history_cols and not _is_ellipsis_index(col_idx)


def _is_target_state_row(row_idx: int, predictor_rows: int) -> bool:
    return row_idx == predictor_rows


def _is_target_row(row_idx: int, predictor_rows: int) -> bool:
    return row_idx == predictor_rows + 1


def _is_target_diff_row(row_idx: int, predictor_rows: int) -> bool:
    return row_idx == predictor_rows + 2


def _is_target_timestep_col(col_idx: int, history_cols: int) -> bool:
    return col_idx == history_cols


def _should_mark_cell(row_idx: int, col_idx: int, predictor_rows: int, history_cols: int) -> bool:
    if _is_predictor_row(row_idx, predictor_rows):
        return _is_history_col(col_idx, history_cols) or _is_target_timestep_col(col_idx, history_cols)
    if _is_target_state_row(row_idx, predictor_rows):
        return _is_history_col(col_idx, history_cols) or _is_target_timestep_col(col_idx, history_cols)
    if _is_target_row(row_idx, predictor_rows) or _is_target_diff_row(row_idx, predictor_rows):
        return _is_target_timestep_col(col_idx, history_cols)
    return False


def _marker_style_for_row(row_idx: int, predictor_rows: int) -> tuple[str, int, str]:
    if _is_target_row(row_idx, predictor_rows):
        return "o", 240, "#2ca02c"
    if _is_target_diff_row(row_idx, predictor_rows):
        return "o", 240, "#d62728"
    if _is_target_state_row(row_idx, predictor_rows):
        return "s", 170, "#ff7f0e"
    return "s", 170, "#1f77b4"


def _build_presence_matrix(n_rows: int, n_cols: int, predictor_rows: int, history_cols: int, missing_fraction: float) -> np.ndarray:
    """
    Build presence matrix where:
      - Predictor rows and Target State have timeline markers.
      - Target and Target Diff only have marker at final timestep.
      - Ellipsis row/column are labels only (no markers).
    """
    present = np.zeros((n_rows, n_cols), dtype=bool)

    for r in range(n_rows):
        for c in range(n_cols):
            present[r, c] = _should_mark_cell(r, c, predictor_rows, history_cols)

    if missing_fraction <= 0:
        return present

    rng = np.random.default_rng(42)
    # Keep Predictor 2 fully missing in the partial-coverage schematic.
    predictor_2_row = 1
    if predictor_2_row < predictor_rows:
        present[predictor_2_row, :] = False

    predictor_mask = np.zeros_like(present, dtype=bool)
    for r in range(n_rows):
        if not _is_predictor_row(r, predictor_rows):
            continue
        if r == predictor_2_row:
            continue
        for c in range(n_cols):
            if _is_history_col(c, history_cols) or _is_target_timestep_col(c, history_cols):
                predictor_mask[r, c] = True

    predictor_positions = np.argwhere(predictor_mask & present)
    n_drop = int(round(len(predictor_positions) * missing_fraction))
    if n_drop > 0:
        drop_idx = rng.choice(len(predictor_positions), size=n_drop, replace=False)
        to_drop = predictor_positions[drop_idx]
        present[to_drop[:, 0], to_drop[:, 1]] = False

    return present


def _build_symbolic_matrix(n_rows: int, n_cols: int, predictor_rows: int, history_cols: int) -> np.ndarray:
    """
    Build symbolic (ellipsis) markers:
      - Entire ellipsis predictor row
      - Entire ellipsis timestep column
    """
    symbolic = np.zeros((n_rows, n_cols), dtype=bool)
    ellipsis_row = 3
    ellipsis_col = 3

    symbolic[ellipsis_row, :] = True
    symbolic[:, ellipsis_col] = True

    # Do not imply extra target/target-diff historic values.
    symbolic[predictor_rows + 1, :history_cols] = False
    symbolic[predictor_rows + 2, :history_cols] = False

    return symbolic


def _draw_raster(
    y_labels: list[str],
    x_labels: list[str],
    presence: np.ndarray,
    symbolic: np.ndarray,
    out_path: Path,
    predictor_rows: int,
) -> None:
    n_rows, n_cols = presence.shape
    row_step = 0.5
    # Keep row spacing mostly unchanged while compressing columns.
    # Use a near-square canvas so horizontal spacing is tighter.
    fig_h = max(5.0, n_rows * 0.4)
    fig_w = max(6.8, min(fig_h * 1.03, n_cols * 0.82))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    text_fs = 18

    for y in range(n_rows):
        y_pos = y * row_step
        ax.hlines(y_pos, 0, n_cols - 1, color="#e8e8e8", linewidth=5, zorder=1)
        symbolic_x = np.where(symbolic[y])[0]
        if symbolic_x.size > 0:
            ax.scatter(
                symbolic_x,
                np.full(symbolic_x.shape, y_pos),
                s=145,
                marker="s",
                facecolor="#efefef",
                edgecolor="#c9c9c9",
                linewidth=1.0,
                zorder=1.8,
            )

        present_x = np.where(presence[y])[0]
        if present_x.size == 0:
            continue

        marker, size, color = _marker_style_for_row(y, predictor_rows)
        ax.scatter(present_x, np.full(present_x.shape, y_pos), s=size, marker=marker, color=color, zorder=2)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=text_fs)
    ax.set_yticks(np.arange(n_rows) * row_step)
    ax.set_yticklabels(y_labels, fontsize=text_fs)
    for tick in ax.get_xticklabels():
        if tick.get_text() == "...":
            tick.set_fontsize(text_fs + 8)
            tick.set_fontweight("bold")
            tick.set_rotation(0)
            tick.set_ha("center")
    for tick in ax.get_yticklabels():
        if tick.get_text() == "...":
            tick.set_fontsize(text_fs + 8)
            tick.set_fontweight("bold")
            tick.set_ha("right")
            tick.set_x(-0.08)
    ax.set_ylim((n_rows - 1) * row_step + 0.25, -0.25)
    edge_pad = 0.42
    ax.set_xlim(-edge_pad, (n_cols - 1) + edge_pad)
    ax.tick_params(axis="both", labelsize=text_fs)
    ax.grid(axis="x", linestyle="--", linewidth=1.0, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock sample-layout rasters.")
    return parser.parse_args()


def main() -> None:
    _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "data" / "output"

    y_labels, x_labels, predictor_rows, history_cols = _build_labels()
    n_rows = len(y_labels)
    n_cols = len(x_labels)

    full_presence = _build_presence_matrix(
        n_rows=n_rows,
        n_cols=n_cols,
        predictor_rows=predictor_rows,
        history_cols=history_cols,
        missing_fraction=0.0,
    )
    missing_presence = _build_presence_matrix(
        n_rows=n_rows,
        n_cols=n_cols,
        predictor_rows=predictor_rows,
        history_cols=history_cols,
        missing_fraction=0.5,
    )
    symbolic = _build_symbolic_matrix(
        n_rows=n_rows,
        n_cols=n_cols,
        predictor_rows=predictor_rows,
        history_cols=history_cols,
    )

    _draw_raster(
        y_labels=y_labels,
        x_labels=x_labels,
        presence=full_presence,
        symbolic=symbolic,
        out_path=out_dir / "demo_sample_full_coverage.png",
        predictor_rows=predictor_rows,
    )
    _draw_raster(
        y_labels=y_labels,
        x_labels=x_labels,
        presence=missing_presence,
        symbolic=symbolic,
        out_path=out_dir / "demo_sample_missing_predictors.png",
        predictor_rows=predictor_rows,
    )

    print(f"Wrote: {out_dir / 'demo_sample_full_coverage.png'}")
    print(f"Wrote: {out_dir / 'demo_sample_missing_predictors.png'}")


if __name__ == "__main__":
    main()

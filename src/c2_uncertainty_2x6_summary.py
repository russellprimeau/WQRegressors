"""
Build a single 2x6 uncertainty summary figure for Monte Carlo documentation.

Top row: offset-error histogram per sensor with fitted normal overlay.
Bottom row: per-sensor goodness-of-fit/statistics table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from c2_uncertainty import test_distribution_fit


SENSOR_ORDER = [
    "Sp Cond (microS_cm)",
    "pH",
    "DO (% Sat)",
    "Turbidity (FNU)",
    "fDOM (RFU)",
    "fDOM (QSU)",
]

FONT_SIZE = 15


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_aggregate_csv() -> Path:
    root = _repo_root()
    candidates = [
        root / "data" / "output" / "calibration" / "aggregate" / "offset_gain_model_results.csv",
        root / "data" / "output" / "calibration" / "summaries" / "aggregate" / "offset_gain_model_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _default_output_png(aggregate_csv: Path) -> Path:
    return aggregate_csv.parent / "offset_error_distribution_2x6.png"


def normalize_sensor_name(name: str) -> str:
    text = str(name).strip()
    if "Sp Cond" in text:
        return "Sp Cond (microS_cm)"
    return text


def fmt(value: float | str | None, digits: int = 4, scientific: bool = False) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        return "NA"
    if scientific:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _most_common_points(points: np.ndarray) -> str:
    if points.size == 0:
        return "NA"
    valid = points[np.isfinite(points)]
    if valid.size == 0:
        return "NA"
    # Use integer point counts and take modal value across events.
    values, counts = np.unique(valid.astype(int), return_counts=True)
    return str(values[np.argmax(counts)])


def build_sensor_table_rows(errors: np.ndarray, points_per_event: np.ndarray) -> list[list[str]]:
    fit = test_distribution_fit(errors, "Offset")
    preferred = fit.get("preferred", "NA")
    if preferred == "t":
        preferred = "Student t"
    elif preferred == "norm":
        preferred = "Normal"

    rows = [
        ["Points per calibration", _most_common_points(points_per_event)],
        ["n", str(int(fit.get("n_clean", len(errors))))],
        ["mean", fmt(np.mean(errors))],
        ["std", fmt(np.std(errors))],
        ["Shapiro p", fmt(fit.get("shapiro_p"), scientific=True)],
        ["Anderson", fmt(fit.get("anderson_stat"))],
        ["AD crit (5%)", fmt(fit.get("anderson_critical_5pct"))],
        ["Normal AIC", fmt(fit.get("norm_aic"), digits=2)],
        ["Student t AIC", fmt(fit.get("t_aic"), digits=2)],
        ["t df", fmt(fit.get("t_df"))],
        ["t loc", fmt(fit.get("t_loc"))],
        ["t scale", fmt(fit.get("t_scale"))],
        ["AIC delta", fmt(fit.get("aic_diff"), digits=2)],
        ["Best fit", preferred],
    ]
    return rows


def plot_histogram(ax: plt.Axes, errors: np.ndarray, title: str) -> None:
    bins = max(6, min(14, len(errors) // 2))
    counts, bin_edges, _ = ax.hist(
        errors,
        bins=bins,
        color="#4C78A8",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.7,
    )

    mu = float(np.mean(errors))
    sigma = float(np.std(errors, ddof=0))
    if sigma > 0 and len(bin_edges) > 1:
        x = np.linspace(bin_edges[0], bin_edges[-1], 300)
        bin_width = bin_edges[1] - bin_edges[0]
        y = stats.norm.pdf(x, loc=mu, scale=sigma) * len(errors) * bin_width
        ax.plot(x, y, color="#E45756", linewidth=2.0, label="Normal fit")

        try:
            t_df, t_loc, t_scale = stats.t.fit(errors)
            if np.isfinite(t_df) and np.isfinite(t_loc) and np.isfinite(t_scale) and t_scale > 0:
                y_t = stats.t.pdf(x, df=t_df, loc=t_loc, scale=t_scale) * len(errors) * bin_width
                ax.plot(x, y_t, color="#54A24B", linewidth=2.0, label="Student t fit")
        except Exception:
            pass

        ax.legend(loc="upper right", frameon=True, fontsize=FONT_SIZE)

    ax.axvline(mu, color="#F58518", linestyle="--", linewidth=1.5)
    ax.set_title(title, fontsize=FONT_SIZE, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.25, axis="y")


def create_figure(df: pd.DataFrame, output_png: Path, dpi: int = 300) -> None:
    df = df.copy()
    df["Sensor_Normalized"] = df["Sensor"].map(normalize_sensor_name)
    df["Offset"] = pd.to_numeric(df["Offset"], errors="coerce")

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(2, 6, figsize=(30, 10), dpi=dpi)
    for col, sensor in enumerate(SENSOR_ORDER):
        sensor_errors = (
            df.loc[df["Sensor_Normalized"] == sensor, "Offset"]
            .dropna()
            .to_numpy(dtype=float)
        )
        sensor_points = (
            pd.to_numeric(df.loc[df["Sensor_Normalized"] == sensor, "N_Points"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
            if "N_Points" in df.columns
            else np.array([])
        )

        top_ax = axes[0, col]
        bottom_ax = axes[1, col]

        if sensor_errors.size < 3:
            top_ax.text(
                0.5, 0.5, "Insufficient data", ha="center", va="center",
                fontsize=FONT_SIZE, transform=top_ax.transAxes
            )
            top_ax.set_title(sensor, fontsize=FONT_SIZE, fontweight="bold")
            top_ax.set_xticks([])
            top_ax.set_yticks([])

            bottom_ax.axis("off")
            bottom_ax.text(
                0.5, 0.5, "n < 3", ha="center", va="center",
                fontsize=FONT_SIZE, transform=bottom_ax.transAxes
            )
            continue

        plot_histogram(top_ax, sensor_errors, sensor)

        bottom_ax.axis("off")
        table_rows = build_sensor_table_rows(sensor_errors, sensor_points)
        table = bottom_ax.table(
            cellText=table_rows,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(FONT_SIZE)
        table.scale(1.0, 1.25)

    fig.subplots_adjust(top=0.97, bottom=0.05, left=0.03, right=0.995, hspace=0.03, wspace=0.20)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 2x6 figure with per-sensor offset-error histograms and fit-stat tables "
            "for MC uncertainty documentation."
        )
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=_default_aggregate_csv(),
        help="Path to aggregate offset_gain_model_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: aggregate dir / offset_error_distribution_2x6.png)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_csv = args.aggregate_csv
    output_png = args.output or _default_output_png(aggregate_csv)

    if not aggregate_csv.exists():
        raise FileNotFoundError(
            f"Could not find aggregate file: {aggregate_csv}. Run c2_uncertainty.py first."
        )

    df = pd.read_csv(aggregate_csv)
    required_cols = {"Sensor", "Offset"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {aggregate_csv}: {sorted(missing)}")

    create_figure(df, output_png, dpi=args.dpi)
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    main()

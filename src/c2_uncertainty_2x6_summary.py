"""
Build a single 2x6 uncertainty summary figure for Monte Carlo documentation.

Row 1: raw correction errors by calibration point (box + jitter).
Row 2: offset-error histogram per sensor with fitted normal overlay.
Row 3: per-sensor goodness-of-fit/statistics table.
Row 4: Q-Q plot for recommended best-fit distribution.
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
    text = text.replace("Âµ", "µ")
    if "Sp Cond" in text:
        return "Sp Cond (microS_cm)"
    return text


def _calibration_output_dir() -> Path:
    return _repo_root() / "data" / "output" / "calibration"


def _load_raw_point_errors(sensor_name: str) -> dict[int, np.ndarray]:
    """
    Load raw calibration correction errors per point from data/output/calibration/<sensor>.csv.
    Returns {point_number: np.ndarray(errors)}.
    """
    base_dir = _calibration_output_dir()
    if not base_dir.exists():
        return {}

    sensor_csv = None
    for path in sorted(base_dir.glob("*.csv")):
        if normalize_sensor_name(path.stem) == sensor_name:
            sensor_csv = path
            break
    if sensor_csv is None:
        return {}

    try:
        raw_df = pd.read_csv(sensor_csv)
    except Exception:
        return {}

    out: dict[int, np.ndarray] = {}
    for idx, col in enumerate(["Correction1", "Correction2", "Correction3"], start=1):
        if col in raw_df.columns:
            vals = pd.to_numeric(raw_df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size > 0:
                out[idx] = vals
    return out


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


def _preferred_fit_label(fit: dict) -> str:
    preferred = fit.get("preferred", "NA")
    if preferred == "t":
        return "Student t"
    elif preferred == "norm":
        return "Normal"
    return str(preferred)


def build_sensor_table_rows(fit: dict, errors: np.ndarray, points_per_event: np.ndarray) -> list[list[str]]:
    preferred = _preferred_fit_label(fit)

    rows = [
        ["n calibrations", str(int(fit.get("n_clean", len(errors))))],
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


def build_quantile_rows(fit: dict, errors: np.ndarray) -> list[list[str]]:
    probs = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
    preferred = fit.get("preferred", "equivalent")
    dist_used = "Normal"

    q_values = None
    if preferred == "t":
        t_df = fit.get("t_df")
        t_loc = fit.get("t_loc")
        t_scale = fit.get("t_scale")
        if all(v is not None and np.isfinite(v) for v in [t_df, t_loc, t_scale]) and t_scale > 0:
            q_values = stats.t.ppf(probs, df=t_df, loc=t_loc, scale=t_scale)
            dist_used = "Student t"

    if q_values is None:
        mu = fit.get("norm_loc", float(np.mean(errors)))
        sigma = fit.get("norm_scale", float(np.std(errors, ddof=0)))
        if sigma <= 0 or not np.isfinite(sigma):
            q_values = np.full_like(probs, np.nan, dtype=float)
        else:
            q_values = stats.norm.ppf(probs, loc=mu, scale=sigma)
        if preferred == "equivalent":
            dist_used = "Equivalent->Normal"

    return [
        ["Best fit used", dist_used],
        ["q05", fmt(q_values[0])],
        ["q25", fmt(q_values[1])],
        ["q50", fmt(q_values[2])],
        ["q75", fmt(q_values[3])],
        ["q95", fmt(q_values[4])],
    ]


def plot_histogram(ax: plt.Axes, errors: np.ndarray, title: str, fit: dict) -> None:
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
    preferred = fit.get("preferred", "equivalent")
    normal_lw = 3.0 if preferred == "norm" else 1.8
    t_lw = 3.0 if preferred == "t" else 1.8

    if sigma > 0 and len(bin_edges) > 1:
        x = np.linspace(bin_edges[0], bin_edges[-1], 300)
        bin_width = bin_edges[1] - bin_edges[0]
        y = stats.norm.pdf(x, loc=mu, scale=sigma) * len(errors) * bin_width
        ax.plot(x, y, color="#E45756", linewidth=normal_lw, label="Normal fit")

        try:
            t_df, t_loc, t_scale = stats.t.fit(errors)
            if np.isfinite(t_df) and np.isfinite(t_loc) and np.isfinite(t_scale) and t_scale > 0:
                y_t = stats.t.pdf(x, df=t_df, loc=t_loc, scale=t_scale) * len(errors) * bin_width
                ax.plot(x, y_t, color="#54A24B", linewidth=t_lw, label="Student t fit")
        except Exception:
            pass

        ax.legend(loc="upper right", frameon=True, fontsize=FONT_SIZE)

    ax.axvline(mu, color="#F58518", linestyle="--", linewidth=1.5)
    ax.set_title("")
    ax.set_xlabel("Offset", fontsize=FONT_SIZE)
    ax.set_ylabel("Count", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.25, axis="y")


def plot_point_box_jitter(ax: plt.Axes, sensor_name: str, point_errors: dict[int, np.ndarray]) -> None:
    if not point_errors:
        ax.text(0.5, 0.5, "No point errors", ha="center", va="center", fontsize=FONT_SIZE, transform=ax.transAxes)
        ax.set_title(sensor_name, fontsize=FONT_SIZE, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    points = sorted(point_errors.keys())
    data = [point_errors[p] for p in points]

    ax.boxplot(
        data,
        positions=points,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#BFD7EA", alpha=0.75, edgecolor="black"),
        medianprops=dict(color="#E45756", linewidth=2.0),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    rng = np.random.default_rng(42)
    for p, vals in zip(points, data):
        x_jitter = p + rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x_jitter, vals, s=20, alpha=0.65, color="#2E5D8A", edgecolors="none")

    ax.set_title(sensor_name, fontsize=FONT_SIZE, fontweight="bold")
    ax.set_xlabel("Calibration point", fontsize=FONT_SIZE)
    ax.set_ylabel("Raw error", fontsize=FONT_SIZE)
    ax.set_xticks(points)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.25, axis="y")


def plot_best_fit_qq(ax: plt.Axes, errors: np.ndarray, fit: dict) -> None:
    if errors.size < 3:
        ax.text(0.5, 0.5, "n < 3", ha="center", va="center", fontsize=FONT_SIZE, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    probs = (np.arange(1, errors.size + 1) - 0.5) / errors.size
    sample_q = np.sort(errors)
    preferred = fit.get("preferred", "equivalent")

    if preferred == "t":
        t_df = fit.get("t_df")
        t_loc = fit.get("t_loc")
        t_scale = fit.get("t_scale")
        if all(v is not None and np.isfinite(v) for v in [t_df, t_loc, t_scale]) and t_scale > 0:
            theoretical_q = stats.t.ppf(probs, df=t_df, loc=t_loc, scale=t_scale)
            dist_label = "Student t"
        else:
            mu = fit.get("norm_loc", float(np.mean(errors)))
            sigma = fit.get("norm_scale", float(np.std(errors, ddof=0)))
            theoretical_q = stats.norm.ppf(probs, loc=mu, scale=sigma)
            dist_label = "Normal"
    else:
        mu = fit.get("norm_loc", float(np.mean(errors)))
        sigma = fit.get("norm_scale", float(np.std(errors, ddof=0)))
        theoretical_q = stats.norm.ppf(probs, loc=mu, scale=sigma)
        dist_label = "Normal"

    ax.scatter(theoretical_q, sample_q, color="#4C78A8", s=36, alpha=0.85, edgecolors="none")
    lo = min(np.nanmin(theoretical_q), np.nanmin(sample_q))
    hi = max(np.nanmax(theoretical_q), np.nanmax(sample_q))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#E45756", linewidth=2.0)
    ax.set_title(f"Q-Q ({dist_label})", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_xlabel("Theoretical quantiles", fontsize=FONT_SIZE)
    ax.set_ylabel("Sample quantiles", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.25)


def create_figure(df: pd.DataFrame, output_png: Path, dpi: int = 300) -> None:
    df = df.copy()
    df["Sensor_Normalized"] = df["Sensor"].map(normalize_sensor_name)
    df["Offset"] = pd.to_numeric(df["Offset"], errors="coerce")

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(4, 6, figsize=(30, 17), dpi=dpi)
    for col, sensor in enumerate(SENSOR_ORDER):
        sensor_errors = (
            df.loc[df["Sensor_Normalized"] == sensor, "Offset"]
            .dropna()
            .to_numpy(dtype=float)
        )
        sensor_point_errors = _load_raw_point_errors(sensor)
        sensor_points = (
            pd.to_numeric(df.loc[df["Sensor_Normalized"] == sensor, "N_Points"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
            if "N_Points" in df.columns
            else np.array([])
        )

        fit = test_distribution_fit(sensor_errors, "Offset")
        box_ax = axes[0, col]
        hist_ax = axes[1, col]
        table_ax = axes[2, col]
        quant_ax = axes[3, col]

        if sensor_errors.size < 3:
            plot_point_box_jitter(box_ax, sensor, sensor_point_errors)
            hist_ax.text(
                0.5, 0.5, "Insufficient data", ha="center", va="center",
                fontsize=FONT_SIZE, transform=hist_ax.transAxes
            )
            hist_ax.set_title("")
            hist_ax.set_xlabel("Offset", fontsize=FONT_SIZE)
            hist_ax.set_xticks([])
            hist_ax.set_yticks([])

            table_ax.axis("off")
            table_ax.text(
                0.5, 0.5, "n < 3", ha="center", va="center",
                fontsize=FONT_SIZE, transform=table_ax.transAxes
            )
            quant_ax.axis("off")
            quant_ax.text(
                0.5, 0.5, "n < 3", ha="center", va="center",
                fontsize=FONT_SIZE, transform=quant_ax.transAxes
            )
            continue

        plot_point_box_jitter(box_ax, sensor, sensor_point_errors)
        plot_histogram(hist_ax, sensor_errors, "", fit)

        table_ax.axis("off")
        table_rows = build_sensor_table_rows(fit, sensor_errors, sensor_points)
        table = table_ax.table(
            cellText=table_rows,
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(FONT_SIZE)
        table.scale(1.0, 1.25)

        plot_best_fit_qq(quant_ax, sensor_errors, fit)

    fig.subplots_adjust(top=0.985, bottom=0.03, left=0.03, right=0.995, hspace=0.15, wspace=0.24)
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

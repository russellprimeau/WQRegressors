"""
Generates horizon-sweep comparison figures across all MC datasets.

Reads ``lookahead_metrics.csv`` from each dataset's ``forecasts/lookahead_sweeps/``
directory (written by ``k_RunHorizonSweep.py``) and produces two summary figures:

  1. ``lookahead_r2_comparison.png``    – R² vs forecast horizon (hours)
  2. ``lookahead_nrmse_comparison.png`` – nRMSE vs forecast horizon (hours)

nRMSE (= RMSE / std_target) is used instead of raw RMSE so that datasets with
very different target magnitudes can be compared on the same axis.  The
``std_target`` value is read from each dataset's
``forecasts/feature_sweeps/feature_sweep_final_metrics.csv`` (populated by
``z1_PostProcess.py``).

Both figures are written to the ``summaries/`` subdirectory of the data root,
the same location used by ``z1_PostProcess.py``.

The script supports metrics CSVs produced by both ``k_RunHorizonSweep.py``
(column ``horizon``) and the legacy ``k_lookahead_sweep.py`` (column
``lookahead``); both are handled transparently.

CLI arguments:
    --data-root PATH        Root directory containing MC_* dataset subdirectories.
                            Default: data/output/regression
    --dataset-prefix STR    Only include datasets whose name starts with this
                            prefix.  Default: MC

Examples:
    python src/z2_horizon_post.py
    python src/z2_horizon_post.py --data-root data/output/regression
    python src/z2_horizon_post.py --data-root data/output/regression --dataset-prefix MC
"""
from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


def _clean_label(dataset_name: str, prefix: str) -> str:
    """Strip the dataset prefix (+ underscore separator) and trailing ``_res``."""
    label = dataset_name
    bare_prefix = prefix.rstrip("_")
    if label.startswith(bare_prefix + "_"):
        label = label[len(bare_prefix) + 1:]
    if label.endswith("_res"):
        label = label[:-4]
    return label.replace("_", " ")


def _load_std_target(dataset_dir: Path) -> float | None:
    """Return ``std_target`` for *dataset_dir* from the feature-sweep metrics CSV.

    ``std_target`` is a property of the dataset (the standard deviation of the
    target variable across all samples) and is constant across every row in
    ``feature_sweep_final_metrics.csv``.  Returns ``None`` when the file is
    absent or contains no valid value.
    """
    csv_path = dataset_dir / "forecasts" / "feature_sweeps" / "feature_sweep_final_metrics.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if "std_target" not in df.columns:
            return None
        valid = df["std_target"].dropna()
        valid = valid[valid > 0]
        if valid.empty:
            return None
        return float(valid.iloc[0])
    except Exception:
        return None


def _discover_datasets(data_root: Path, prefix: str) -> list[tuple[str, Path, Path]]:
    """Return ``(dataset_name, dataset_dir, lookahead_metrics_path)`` for every qualifying dataset."""
    hits: list[tuple[str, Path, Path]] = []
    if not data_root.exists():
        print(f"[WARN] data_root does not exist: {data_root}")
        return hits
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        metrics_csv = child / "forecasts" / "lookahead_sweeps" / "lookahead_metrics.csv"
        if metrics_csv.exists():
            hits.append((child.name, child, metrics_csv))
        else:
            print(f"[SKIP] No lookahead_metrics.csv for {child.name}")
    return hits


def generate_figures(data_root: Path, prefix: str, summaries_dir: Path) -> int:
    datasets = _discover_datasets(data_root, prefix)
    if not datasets:
        print(f"[WARN] No datasets with lookahead metrics found under {data_root}.")
        return 1

    print(f"[INFO] Found {len(datasets)} dataset(s) with lookahead metrics.")
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Each record: (label, dataframe-with-nrmse-column)
    records: list[tuple[str, pd.DataFrame]] = []
    for name, dataset_dir, csv_path in datasets:
        try:
            df = pd.read_csv(csv_path)
            # Normalise column name: k_RunHorizonSweep uses "horizon",
            # legacy k_lookahead_sweep used "lookahead".
            if "horizon" in df.columns and "lookahead" not in df.columns:
                df = df.rename(columns={"horizon": "lookahead"})
            df = df.sort_values("lookahead").reset_index(drop=True)
            std_target = _load_std_target(dataset_dir)
            if std_target is not None:
                df["nrmse"] = df["rmse"] / std_target
                std_note = f"std_target={std_target:.4g}"
            else:
                df["nrmse"] = float("nan")
                std_note = "std_target not found – nRMSE will be NaN"
            label = _clean_label(name, prefix)
            records.append((label, df))
            print(f"[INFO]  {name}: {len(df)} horizon rows, {std_note}")
        except Exception:
            print(f"[WARN] Could not load {csv_path}:")
            traceback.print_exc()

    if not records:
        print("[WARN] No data loaded; aborting.")
        return 1

    # Collect all x-values across datasets for consistent tick marks
    all_x = sorted({v for _, df in records for v in df["lookahead"].dropna().tolist()})

    def _make_figure(metric: str, ylabel: str, filename: str, hline_zero: bool = False) -> Path:
        fig_w = max(10, len(records) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_w, 5), constrained_layout=True)
        for label, df in records:
            if metric not in df.columns or df[metric].isnull().all():
                continue
            ax.plot(
                df["lookahead"],
                df[metric],
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=label,
            )
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(all_x)
        ax.set_xticklabels([])  # labels drawn manually below to allow vertical staggering
        # Stagger tick labels at two depths so closely-spaced values don't overlap.
        # get_xaxis_transform(): x in data coords, y in axes fraction (0=bottom, negative=below).
        trans = ax.get_xaxis_transform()
        for i, val in enumerate(all_x):
            y_offset = -0.05 if i % 2 == 0 else -0.12
            ax.text(val, y_offset, str(int(val)), transform=trans,
                    ha="center", va="top", fontsize=9)
        ax.grid(axis="both", alpha=0.3)
        if hline_zero:
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        # Place legend outside the plot area so it never overlaps the lines
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            fontsize=7,
            framealpha=0.8,
        )
        out = summaries_dir / filename
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    r2_path = _make_figure("r2", "R²", "lookahead_r2_comparison.png", hline_zero=True)
    print(f"[INFO] Wrote R² figure:    {r2_path}")

    nrmse_path = _make_figure("nrmse", "nRMSE (RMSE / σ_target)", "lookahead_nrmse_comparison.png")
    print(f"[INFO] Wrote nRMSE figure: {nrmse_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate lookahead-sweep R² and nRMSE comparison figures across MC datasets."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/output/regression",
        help="Root directory containing dataset subdirectories (default: data/output/regression).",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="MC",
        help="Only include dataset directories whose name starts with this prefix (default: MC).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()
    summaries_dir = (data_root / "summaries").resolve()
    print(f"[INFO] data_root : {data_root}")
    print(f"[INFO] summaries : {summaries_dir}")
    return generate_figures(data_root=data_root, prefix=args.dataset_prefix, summaries_dir=summaries_dir)


if __name__ == "__main__":
    sys.exit(main())

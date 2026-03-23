"""
Generates horizon-sweep comparison figures across all MC datasets.

Reads ``lookahead_metrics.csv`` from each dataset's ``horizons/lookahead_sweeps/``
directory (written by ``k_RunHorizonSweep.py``) and produces:

  1. ``lookahead_r2_comparison.png``    – R² vs forecast horizon (hours)
  2. ``lookahead_nrmse_comparison.png`` – nRMSE vs forecast horizon (hours)
  3. ``lookahead_skill_comparison.png``     – skill vs. best baseline per horizon
  4. ``lookahead_time_to_zero_skill.png``   – initial skill / skill rate (hours to zero skill)
  5. ``lookahead_aggregate.csv``            – combined table of all datasets × horizons × replicates

nRMSE (= RMSE / std_target) is used instead of raw RMSE so that datasets with
very different target magnitudes can be compared on the same axis.  The
``std_target`` value is read from each dataset's
``forecasts/feature_sweeps/feature_sweep_final_metrics.csv`` (populated by
``z1_PostProcess.py``).

All outputs are written to the ``summaries/horizons/`` subdirectory of the data root.

Uncertainty bands (replicates > 1):
    When ``k_RunHorizonSweep.py`` is run with ``--replicates M > 1``,
    ``lookahead_metrics.csv`` contains a ``replicate`` column with M rows per
    horizon.  This script detects that and plots each dataset as:
      - Solid mean line with markers
      - Dashed ±1σ boundary lines (linewidth=0.8, alpha=0.7)
      - Dotted ±2σ boundary lines (linewidth=0.6, alpha=0.55)
    This matches the styling of ``Surface_timeseries_uncertainty.png`` produced
    by ``b_ExploreData.py``.  When only one replicate is present the standard
    single-line plot is drawn.

CSV column compatibility:
    Supports CSVs produced by both ``k_RunHorizonSweep.py`` (column ``horizon``,
    optional ``replicate``) and the legacy ``k_lookahead_sweep.py`` (column
    ``lookahead``, no ``replicate``); both are handled transparently.

CLI arguments:
    --data-root PATH        Root directory containing MC_* dataset subdirectories.
                            Default: data/output/regression
    --dataset-prefix STR    Only include datasets whose name starts with this
                            prefix.  Default: MC

Examples:
python src/z2_HorizonPostProcess.py
python src/z2_HorizonPostProcess.py --data-root data/output/regression
python src/z2_HorizonPostProcess.py --data-root data/output/CV11_horizons --dataset-prefix MC
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
matplotlib.use("Agg")
# Ensure src/ is on the path so utils can be imported when the script is run
# directly (python src/z2_...) or from the workspace root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.names import clean_target_label

# Color palette matching b_ExploreData.py _safe_series_colors — avoids red/orange hues
# so uncertainty bands (same color, lower alpha) stay visually distinct from alarm states.
_SERIES_COLORS = [
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#17becf",  # cyan
    "#9467bd",  # purple
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#aec7e8",  # light blue
    "#98df8a",  # light green
    "#c5b0d5",  # light purple
    "#9edae5",  # pale cyan
    "#8c564b",  # brown
    "#c7c7c7",  # light gray
]


def _find_horizon_forecast_dirs(dataset_dir: Path) -> list[tuple[int, int, Path]]:
    """Discover ``(horizon, replicate, forecast_dir)`` tuples under *dataset_dir*/horizons/.

    Supports two on-disk layouts:

    **Shared-samples layout** (k_RunHorizonSweep with data_dir=horizon_dir)::

        horizons/horizon_NNNhr/forecasts/horizon_NNNhr_rep_RRR/

    **Per-replicate layout** (original k_RunHorizonSweep)::

        horizons/horizon_NNNhr/rep_RRR/forecasts/horizon_NNNhr_rep_RRR/
    """
    _horizon_re = re.compile(r"^horizon_(\d+)hr$")
    _rep_re = re.compile(r"_rep_(\d+)$")
    hits: list[tuple[int, int, Path]] = []
    horizons_root = dataset_dir / "horizons"
    if not horizons_root.is_dir():
        return hits
    for h_dir in sorted(horizons_root.iterdir()):
        m_h = _horizon_re.match(h_dir.name)
        if not m_h:
            continue
        horizon = int(m_h.group(1))
        # Shared-samples layout: horizon_dir/forecasts/horizon_NNNhr_rep_RRR/
        forecasts_root = h_dir / "forecasts"
        if forecasts_root.is_dir():
            for fc_dir in sorted(forecasts_root.iterdir()):
                m_r = _rep_re.search(fc_dir.name)
                if m_r and fc_dir.is_dir():
                    hits.append((horizon, int(m_r.group(1)), fc_dir))
        # Per-replicate layout: horizon_dir/rep_RRR/forecasts/horizon_NNNhr_rep_RRR/
        for rep_dir in sorted(h_dir.iterdir()):
            if not rep_dir.is_dir() or not rep_dir.name.startswith("rep_"):
                continue
            rep_fc_root = rep_dir / "forecasts"
            if not rep_fc_root.is_dir():
                continue
            for fc_dir in sorted(rep_fc_root.iterdir()):
                m_r = _rep_re.search(fc_dir.name)
                if m_r and fc_dir.is_dir():
                    # Avoid duplicates when both layouts coexist
                    entry = (horizon, int(m_r.group(1)), fc_dir)
                    if entry not in hits:
                        hits.append(entry)
    return hits


def _run_pending_evaluations(dataset_dir: Path) -> int:
    """Run ``f_Evaluate.py`` for forecast dirs that have an eval config but no evaluation_summary.csv.

    Returns the number of evaluations successfully completed.
    """
    completed = 0
    for horizon, rep_idx, fc_dir in _find_horizon_forecast_dirs(dataset_dir):
        summary_csv = fc_dir / "evaluation_summary.csv"
        if summary_csv.exists():
            continue
        # Find eval config
        eval_cfgs = list(fc_dir.glob("config_evaluate_*.yml"))
        if not eval_cfgs:
            continue
        eval_cfg = eval_cfgs[0]
        print(f"  [EVAL] Running pending evaluation: horizon {horizon}hr rep {rep_idx}")
        try:
            subprocess.run(
                [sys.executable, "src/f_Evaluate.py", "--config", str(eval_cfg)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if summary_csv.exists():
                completed += 1
            else:
                print(f"  [WARN] Evaluation completed but no summary produced: {fc_dir.name}")
        except subprocess.CalledProcessError as exc:
            print(f"  [WARN] Evaluation failed for {fc_dir.name}:")
            stderr_text = exc.stderr.decode(errors="replace") if exc.stderr else ""
            if stderr_text:
                for line in stderr_text.strip().splitlines()[-3:]:
                    print(f"         {line}")
    return completed


def _reconstruct_lookahead_metrics(dataset_dir: Path) -> pd.DataFrame | None:
    """Build a ``lookahead_metrics.csv``-equivalent DataFrame by scanning evaluation_summary.csv files.

    Mimics the filtering logic in ``k_RunHorizonSweep.py``: keeps only test-set
    rows, drops the ``label``/``kind`` columns, and adds ``horizon`` / ``replicate``
    columns derived from the directory structure.
    """
    frames: list[pd.DataFrame] = []
    for horizon, rep_idx, fc_dir in _find_horizon_forecast_dirs(dataset_dir):
        summary_csv = fc_dir / "evaluation_summary.csv"
        if not summary_csv.exists():
            continue
        try:
            df = pd.read_csv(summary_csv)
        except Exception:
            continue
        # Filter to test rows (same logic as k_RunHorizonSweep.py)
        if "kind" in df.columns:
            df = df[df["kind"] == "test"]
        elif "label" in df.columns:
            df = df[df["label"].str.contains("test", case=False, na=False)]
        df = df.drop(columns=[c for c in ("label", "kind") if c in df.columns], errors="ignore")
        df["horizon"] = horizon
        df["replicate"] = rep_idx
        # Derive input_rows from the eval config if available
        eval_cfgs = list(fc_dir.glob("config_evaluate_*.yml"))
        if eval_cfgs:
            try:
                with open(eval_cfgs[0], "r", encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh)
                data_cfg = cfg.get("data", {})
                row1 = int(data_cfg.get("input_row_1", 0))
                row2 = int(data_cfg.get("input_row_2", 0))
                df["input_rows_included"] = row2 - row1 + 1
                df["input_rows_excluded"] = horizon
            except Exception:
                pass
        frames.append(df)
    if not frames:
        return None
    result = pd.concat(frames, ignore_index=True)
    # Put horizon and replicate first
    front = [c for c in ("horizon", "replicate") if c in result.columns]
    rest = [c for c in result.columns if c not in front]
    return result[front + rest]


def _clean_label(dataset_name: str, prefix: str) -> str:
    """Delegate to the shared ``clean_target_label`` in ``utils.names``."""
    return clean_target_label(dataset_name, prefix)


def _load_baseline_rmses(dataset_dir: Path, horizon_hr: int, replicate: int) -> dict[str, float]:
    """Return ``{baseline_label: rmse}`` from the evaluation_summary.csv for one horizon/replicate.

    Returns an empty dict when the file is absent or unreadable.
    Searches both the shared-samples layout (``horizon_dir/forecasts/rep_name/``)
    and the per-replicate layout (``horizon_dir/rep_NNN/forecasts/rep_name/``).
    """
    rep_name = f"horizon_{horizon_hr:03d}hr_rep_{replicate:03d}"
    h_dir = dataset_dir / "horizons" / f"horizon_{horizon_hr:03d}hr"
    candidates = [
        h_dir / "forecasts" / rep_name / "evaluation_summary.csv",
        h_dir / f"rep_{replicate:03d}" / "forecasts" / rep_name / "evaluation_summary.csv",
    ]
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break
    if csv_path is None:
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "rmse" not in df.columns:
            return {}
        # Accept rows whose kind == "baseline" or whose label contains a known baseline name
        _known = {"naive", "seasonal", "linear"}
        if "kind" in df.columns:
            mask = df["kind"].str.lower() == "baseline"
        else:
            mask = df["label"].str.lower().apply(lambda s: any(k in s for k in _known))
        baselines = df[mask].dropna(subset=["rmse"])
        result: dict[str, float] = {}
        for _, row in baselines.iterrows():
            lbl = str(row.get("label", "")).lower()
            rmse_val = float(row["rmse"])
            if np.isfinite(rmse_val) and rmse_val > 0:
                result[lbl] = rmse_val
        return result
    except Exception:
        return {}


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


def _rate_table(records: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """One row per dataset: linear slope of each metric vs. horizon (units per hour)."""
    rows = []
    for label, df in records:
        grp = df.groupby("lookahead")
        means = grp[["rmse", "nrmse", "r2", "skill_v_best_baseline"]].mean()
        has_reps = "replicate" in df.columns and df["replicate"].nunique() > 1
        std_rmse  = grp["rmse"].std()               if has_reps else pd.Series(dtype=float)
        std_r2    = grp["r2"].std()                 if has_reps else pd.Series(dtype=float)
        std_skill = grp["skill_v_best_baseline"].std() if has_reps else pd.Series(dtype=float)

        def _slope(series: pd.Series) -> float:
            s = series.dropna()
            if len(s) < 2:
                return float("nan")
            return float(np.polyfit(s.index.astype(float), s.values, 1)[0])

        def _std_slope(series: pd.Series) -> float:
            return _slope(series) if not series.empty else float("nan")

        skill_by_h = df.groupby("lookahead")["skill_v_best_baseline"].mean()
        _min_h = skill_by_h.index.min()
        initial_skill = float(skill_by_h.loc[_min_h]) if pd.notna(skill_by_h.loc[_min_h]) else float("nan")

        rows.append({
            "dataset":         label,
            "rmse_rate":       _slope(means["rmse"]),
            "nrmse_rate":      _slope(means["nrmse"]),
            "r2_rate":         _slope(means["r2"]),
            "std_rate":        _std_slope(std_rmse),
            "skill_rate":      _slope(means["skill_v_best_baseline"]),
            "std_r2_rate":     _std_slope(std_r2),
            "std_skill_rate":  _std_slope(std_skill),
            "initial_skill":   initial_skill,
        })
    return pd.DataFrame(rows)


def _discover_datasets(data_root: Path, prefix: str) -> list[tuple[str, Path, Path]]:
    """Return ``(dataset_name, dataset_dir, lookahead_metrics_path)`` for every qualifying dataset.

    When ``lookahead_metrics.csv`` is missing (e.g. because the evaluation step
    in ``k_RunHorizonSweep.py`` failed), this function:

    1. Runs any pending evaluations whose eval config exists but whose
       ``evaluation_summary.csv`` is absent.
    2. Reconstructs ``lookahead_metrics.csv`` from the per-horizon
       ``evaluation_summary.csv`` files found on disk.
    """
    hits: list[tuple[str, Path, Path]] = []
    if not data_root.exists():
        print(f"[WARN] data_root does not exist: {data_root}")
        return hits
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        metrics_csv = child / "horizons" / "lookahead_sweeps" / "lookahead_metrics.csv"
        if metrics_csv.exists():
            hits.append((child.name, child, metrics_csv))
            continue

        # --- Fallback: run pending evaluations and reconstruct metrics ---
        n_completed = _run_pending_evaluations(child)
        if n_completed:
            print(f"[INFO] Completed {n_completed} pending evaluation(s) for {child.name}")
        reconstructed = _reconstruct_lookahead_metrics(child)
        if reconstructed is not None and not reconstructed.empty:
            sweep_dir = child / "horizons" / "lookahead_sweeps"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            reconstructed.to_csv(metrics_csv, index=False)
            print(f"[INFO] Reconstructed lookahead_metrics.csv for {child.name} "
                  f"({len(reconstructed)} rows)")
            hits.append((child.name, child, metrics_csv))
        else:
            print(f"[SKIP] No lookahead_metrics.csv for {child.name}")
    return hits


def _bar_fmt(v: float) -> str:
    """Scientific notation without padding or explicit positive sign (e.g. '1.23e3')."""
    s = f"{v:.3g}"
    if "e" in s:
        mantissa, exp = s.split("e")
        return f"{mantissa}e{int(exp)}"
    return s


def _annotate_bars(ax: plt.Axes, fontsize: int = 9) -> None:
    """Annotate each bar patch with its numeric value (rotated 90°)."""
    for rect in ax.patches:
        h = rect.get_height()
        if not np.isfinite(h) or h == 0:
            continue
        x = rect.get_x() + rect.get_width() / 2
        va = "bottom" if h >= 0 else "top"
        ax.text(x, h, _bar_fmt(h), ha="center", va=va, fontsize=fontsize,
                rotation=90, clip_on=False)


def _plot_rate_bar(
    rate_df: pd.DataFrame,
    col: str,
    ylabel: str,
    filename: str,
    summaries_dir: Path,
    ascending: bool = False,
    color: str = _SERIES_COLORS[0],
    std_col: str | None = None,
    std_label: str = "σ rate (/hr)",
    std_color: str = _SERIES_COLORS[3],
) -> Path:
    """Bar chart for one rate column, with an optional paired std bar.

    When *std_col* is provided the bars are clustered (±bar_w/2 offset);
    otherwise a single centred bar is drawn.
    """
    df = rate_df.dropna(subset=[col]).copy()
    df = df.sort_values(col, ascending=ascending).reset_index(drop=True)

    n = len(df)
    x = np.arange(n)
    clustered = std_col is not None and df[std_col].notna().any()
    bar_w = 0.42 if clustered else 0.72

    fig_w = max(6.0, 0.9 * n + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    x_main = x - bar_w / 2 if clustered else x
    ax.bar(x_main, df[col], width=bar_w, color=color,
           label=col.replace("_", " "))
    if clustered:
        ax.bar(x + bar_w / 2, df[std_col], width=bar_w,
               color=std_color, label=std_label)
        ax.legend(fontsize=11, framealpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=45, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel(textwrap.fill(ylabel, width=20), fontsize=14)
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(axis="y", alpha=0.3)
    _annotate_bars(ax)
    _ylo, _yhi = ax.get_ylim()
    _span = _yhi - _ylo
    ax.set_ylim(_ylo - 0.12 * _span, _yhi + 0.2 * _span)

    fig.tight_layout()
    out = summaries_dir / filename
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def _plot_rates(rate_df: pd.DataFrame, summaries_dir: Path, show_std: bool = True) -> Path:
    """Clustered bar chart of nrmse_rate and std_rate per dataset.

    Clusters are ordered by decreasing nrmse_rate.  std_rate bars are omitted
    (rendered as NaN-height, i.e. absent) for datasets with only one replicate,
    or when *show_std* is False.
    """
    df = rate_df.dropna(subset=["nrmse_rate"]).copy()
    df = df.sort_values("nrmse_rate", ascending=False).reset_index(drop=True)

    n = len(df)
    x = np.arange(n)
    bar_w = 0.42

    fig_w = max(6.0, 0.9 * n + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    rmse_color = _SERIES_COLORS[0]   # blue
    std_color  = _SERIES_COLORS[3]   # purple

    ax.bar(x - bar_w / 2, df["nrmse_rate"], width=bar_w,
           color=rmse_color, label="nRMSE rate (/hr)")

    std_vals = df["std_rate"]
    if show_std and std_vals.notna().any():
        ax.bar(x + bar_w / 2, std_vals, width=bar_w,
               color=std_color, label="σ(RMSE) rate (units/hr)")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=45, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel(textwrap.fill("nRMSE avg. rate of change (/hr)", width=20), fontsize=14)
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11, framealpha=0.85)
    _annotate_bars(ax)
    _ylo, _yhi = ax.get_ylim()
    _span = _yhi - _ylo
    ax.set_ylim(_ylo - 0.12 * _span, _yhi + 0.12 * _span)

    fig.tight_layout()
    out = summaries_dir / "lookahead_rates_bar.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def generate_figures(data_root: Path, prefix: str, summaries_dir: Path, show_std: bool = True) -> int:
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

            # Skill vs. each baseline — requires per-horizon evaluation_summary.csv.
            # Cache lookups so each (horizon, replicate) pair is read at most once.
            rep_col = "replicate" if "replicate" in df.columns else None
            _baseline_cache: dict[tuple[int, int], dict[str, float]] = {}
            for _, _row in df.iterrows():
                _h = int(_row["lookahead"])
                _r = int(_row[rep_col]) if rep_col else 0
                _key = (_h, _r)
                if _key not in _baseline_cache:
                    _baseline_cache[_key] = _load_baseline_rmses(dataset_dir, _h, _r)

            def _skill(model_rmse: float, baseline_rmse: float) -> float:
                if np.isfinite(model_rmse) and np.isfinite(baseline_rmse) and baseline_rmse > 0:
                    return 1.0 - model_rmse / baseline_rmse
                return float("nan")

            def _get_skills(row: pd.Series) -> pd.Series:
                h = int(row["lookahead"])
                r = int(row[rep_col]) if rep_col else 0
                bl = _baseline_cache.get((h, r), {})
                m = float(row["rmse"])
                return pd.Series({
                    "skill_v_naive":    _skill(m, bl.get("naive",    float("nan"))),
                    "skill_v_seasonal": _skill(m, bl.get("seasonal", float("nan"))),
                    "skill_v_linear":   _skill(m, bl.get("linear",   float("nan"))),
                })

            _skills = df.apply(_get_skills, axis=1)
            df["skill_v_naive"]    = _skills["skill_v_naive"]
            df["skill_v_seasonal"] = _skills["skill_v_seasonal"]
            df["skill_v_linear"]   = _skills["skill_v_linear"]
            df["skill_v_best_baseline"] = _skills[
                ["skill_v_naive", "skill_v_seasonal", "skill_v_linear"]
            ].max(axis=1)

            label = _clean_label(name, prefix)
            records.append((label, df))
            n_horizons = df["lookahead"].nunique()
            n_reps = df["replicate"].nunique() if "replicate" in df.columns else 1
            print(f"[INFO]  {name}: {n_horizons} horizons × {n_reps} replicate(s), {std_note}")
        except Exception:
            print(f"[WARN] Could not load {csv_path}:")
            traceback.print_exc()

    if not records:
        print("[WARN] No data loaded; aborting.")
        return 1

    # Collect all x-values across datasets for consistent tick marks
    all_x = sorted({v for _, df in records for v in df["lookahead"].dropna().tolist()})

    # Detect whether any dataset has multiple replicates (drives σ-band and legend content).
    # Suppressed entirely when show_std=False.
    any_replicates = show_std and any(
        "replicate" in df.columns and df["replicate"].nunique() > 1
        for _, df in records
    )

    # Layout constants matching b_ExploreData.py Target_timeseries style
    _FIG_WIDTH  = 13.0
    _ROW_HEIGHT = 0.88
    _MIN_FIG_H  = 2.8
    _HSPACE     = 0.08
    _TOP_IN     = 0.45   # inches reserved at top — must exceed legend height (~0.35 in) + gap
    _BOTTOM_IN  = 0.90   # inches reserved at bottom (staggered tick labels + supxlabel)

    def _make_figure(
        metric: str,
        ylabel: str,
        filename: str,
        hline_zero: bool = False,
        any_replicates: bool = False,
        ylim: tuple | None = None,
        yticks: list | None = None,
    ) -> Path:
        n_rows = len(records)
        fig_h = max(_MIN_FIG_H, _ROW_HEIGHT * n_rows)
        fig, axes = plt.subplots(
            n_rows, 1, sharex=True,
            figsize=(_FIG_WIDTH, fig_h),
            gridspec_kw={"hspace": _HSPACE},
        )
        if n_rows == 1:
            axes = [axes]

        for i, (label, df) in enumerate(records):
            ax = axes[i]
            color = _SERIES_COLORS[i % len(_SERIES_COLORS)]

            if metric not in df.columns or df[metric].isnull().all():
                ax.set_visible(False)
                continue

            has_replicates = show_std and "replicate" in df.columns and df["replicate"].nunique() > 1
            if has_replicates:
                grp = df.groupby("lookahead")[metric].agg(["mean", "std"]).reset_index()
                x = grp["lookahead"]
                mu = grp["mean"]
                sigma = grp["std"].fillna(0)
                # Mean line (solid with markers)
                ax.plot(x, mu, marker="o", markersize=4, linewidth=1.5,
                        color=color, zorder=3)
                # ±1σ — dashed, matching b_ExploreData.py Surface_timeseries_uncertainty style
                ax.plot(x, mu + sigma,     linestyle="--", linewidth=0.8, alpha=0.7,
                        color=color, zorder=2)
                ax.plot(x, mu - sigma,     linestyle="--", linewidth=0.8, alpha=0.7,
                        color=color, zorder=2)
                # ±2σ — dotted
                ax.plot(x, mu + 2 * sigma, linestyle=":",  linewidth=0.6, alpha=0.55,
                        color=color, zorder=1)
                ax.plot(x, mu - 2 * sigma, linestyle=":",  linewidth=0.6, alpha=0.55,
                        color=color, zorder=1)
            else:
                # Single replicate or legacy CSV without replicate column
                plot_df = (
                    df.groupby("lookahead")[metric].mean().reset_index()
                    if "replicate" in df.columns else df
                )
                ax.plot(plot_df["lookahead"], plot_df[metric],
                        marker="o", markersize=4, linewidth=1.5, color=color)

            wrapped = "\n".join(textwrap.wrap(label, width=15))
            ax.set_ylabel(wrapped, rotation=0, ha="right", va="center",
                          fontsize=15, labelpad=8)
            ax.grid(axis="both", alpha=0.3)
            if yticks is not None:
                ax.set_yticks(yticks)
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            if ylim is not None:
                ax.set_ylim(*ylim)
            if hline_zero:
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            if i < n_rows - 1:
                ax.tick_params(axis="x", which="both", labelbottom=False)

        # X-axis: staggered tick labels on bottom subplot only.
        # get_xaxis_transform(): x in data coords, y in axes fraction (negative = below axis).
        axes[-1].set_xticks(all_x)
        axes[-1].set_xticklabels([])
        trans = axes[-1].get_xaxis_transform()
        for j, val in enumerate(all_x):
            y_offset = -0.1 if j % 2 == 0 else -0.12
            axes[-1].text(val, y_offset, str(int(val)), transform=trans,
                          ha="center", va="top", fontsize=10)

        # Margins — computed here so axes height is known for xlabel labelpad.
        top_frac    = max(0.86, min(0.995, 1.0 - (_TOP_IN    / fig_h)))
        bottom_frac = max(0.08, min(0.30,          _BOTTOM_IN / fig_h))
        fig.subplots_adjust(
            left=0.18, right=0.995,
            top=top_frac, bottom=bottom_frac,
            hspace=_HSPACE,
        )

        # X-axis label — positioned below the deepest stagger text.
        # The stagger sits at -0.12 axes fraction; convert to points to derive labelpad.
        _axes_h_frac = (top_frac - bottom_frac) / (n_rows + max(n_rows - 1, 0) * _HSPACE)
        _axes_h_pts  = _axes_h_frac * fig_h * 72
        _xlabel_pad  = max(18, int(0.12 * _axes_h_pts + 15))
        axes[-1].set_xlabel("Forecast horizon (hours)", fontsize=15, labelpad=_xlabel_pad)

        # Single figure-level legend above top subplot with generic black lines
        legend_handles = [
            plt.Line2D([0], [0], color="black", linewidth=1.5,
                       marker="o", markersize=4, label=f"Mean {ylabel}"),
        ]
        if any_replicates:
            legend_handles += [
                plt.Line2D([0], [0], color="black", linestyle="--",
                           linewidth=0.8, alpha=0.7,  label="±1σ"),
                plt.Line2D([0], [0], color="black", linestyle=":",
                           linewidth=0.6, alpha=0.55, label="±2σ"),
            ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=len(legend_handles),
            framealpha=0.85,
            fontsize=15,
            borderaxespad=0.12,
        )

        out = summaries_dir / filename
        fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return out

    r2_path = _make_figure("r2", "R²", "lookahead_r2_comparison.png",
                           hline_zero=True, any_replicates=any_replicates,
                           ylim=(-1.2, 1.2), yticks=[-1, 0, 1])
    print(f"[INFO] Wrote R² figure:    {r2_path}")

    # Shared nRMSE y-axis limits: global min/max across all datasets and replicates.
    _nrmse_all = pd.concat(
        [df["nrmse"] for _, df in records if "nrmse" in df.columns],
        ignore_index=True,
    ).dropna()
    if not _nrmse_all.empty:
        _nrmse_min, _nrmse_max = float(_nrmse_all.min()), float(_nrmse_all.max())
        _nrmse_margin = 0.1 * (_nrmse_max - _nrmse_min)
        nrmse_ylim = (_nrmse_min - _nrmse_margin, _nrmse_max + _nrmse_margin)
    else:
        nrmse_ylim = None

    nrmse_path = _make_figure("nrmse", "nRMSE (RMSE / σ_target)", "lookahead_nrmse_comparison.png",
                              any_replicates=any_replicates, ylim=nrmse_ylim)
    print(f"[INFO] Wrote nRMSE figure: {nrmse_path}")

    # Skill vs. best-baseline figure
    _skill_all = pd.concat(
        [df["skill_v_best_baseline"] for _, df in records if "skill_v_best_baseline" in df.columns],
        ignore_index=True,
    ).dropna()
    if not _skill_all.empty:
        _s_min, _s_max = float(_skill_all.min()), float(_skill_all.max())
        _s_margin = max(0.05, 0.1 * (_s_max - _s_min))
        skill_ylim: tuple | None = (max(-2.0, _s_min - _s_margin), min(1.05, _s_max + _s_margin))
    else:
        skill_ylim = None

    skill_path = _make_figure(
        "skill_v_best_baseline",
        "Skill vs. Best Baseline",
        "lookahead_skill_comparison.png",
        hline_zero=True,
        any_replicates=any_replicates,
        ylim=skill_ylim,
    )
    print(f"[INFO] Wrote skill figure:  {skill_path}")

    # Aggregate CSV — all datasets × horizons × replicates in one table
    _drop_suffix = "_replicate"   # these columns repeat per-replicate values; redundant here
    _front_cols = [
        "dataset", "lookahead", "replicate",
        "mae", "rmse", "nrmse", "r2", "pearson_r",
        "skill_v_naive", "skill_v_seasonal", "skill_v_linear", "skill_v_best_baseline",
    ]
    agg_frames = []
    for lbl, df in records:
        frame = df.copy()
        # Drop redundant *_replicate aggregate columns
        frame = frame.drop(columns=[c for c in frame.columns if c.endswith(_drop_suffix)],
                           errors="ignore")
        frame.insert(0, "dataset", lbl)
        # Append a replicate=mean row for each lookahead value
        _numeric_means = (
            frame.groupby("lookahead", sort=True)
                 .mean(numeric_only=True)
                 .reset_index()
        )
        _numeric_means["dataset"]   = lbl
        _numeric_means["replicate"] = "mean"
        frame = pd.concat([frame, _numeric_means], ignore_index=True)
        agg_frames.append(frame)
    agg_df = pd.concat(agg_frames, ignore_index=True)
    # Reorder: metric/skill columns first, then remaining metadata
    _rest_cols = [c for c in agg_df.columns if c not in _front_cols]
    agg_df = agg_df[[c for c in _front_cols if c in agg_df.columns] + _rest_cols]
    agg_path = summaries_dir / "lookahead_aggregate.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"[INFO] Wrote aggregate CSV: {agg_path}")

    rate_df = _rate_table(records)
    rate_path = summaries_dir / "lookahead_rates.csv"
    rate_df.to_csv(rate_path, index=False)
    print(f"[INFO] Wrote rates CSV:     {rate_path}")

    rates_path = _plot_rates(rate_df, summaries_dir, show_std=show_std)
    print(f"[INFO] Wrote rates figure:  {rates_path}")

    skill_bar_path = _plot_rate_bar(
        rate_df, "skill_rate", "Skill avg. rate of change (/hr)",
        "lookahead_skill_rate_bar.png", summaries_dir,
        ascending=True, color=_SERIES_COLORS[1],
        std_col="std_skill_rate" if show_std else None,
        std_label="σ(skill) rate (/hr)",
    )
    print(f"[INFO] Wrote skill rate bar: {skill_bar_path}")

    r2_bar_path = _plot_rate_bar(
        rate_df, "r2_rate", "$R^2$ avg. rate of change (/hr)",
        "lookahead_r2_rate_bar.png", summaries_dir,
        ascending=True, color=_SERIES_COLORS[2],
        std_col="std_r2_rate" if show_std else None,
        std_label="σ(R²) rate (/hr)",
    )
    print(f"[INFO] Wrote R² rate bar:   {r2_bar_path}")

    # Time-to-zero-skill: initial skill / skill rate (hours at which linear
    # extrapolation of skill reaches 0).  Larger positive values mean the model
    # retains useful skill over a longer forecast horizon.
    rate_df["time_to_zero_skill"] = rate_df["initial_skill"] / (-24*rate_df["skill_rate"])
    tzs_path = _plot_rate_bar(
        rate_df, "time_to_zero_skill",
        "Forecast Horizon (days)",
        "lookahead_time_to_zero_skill.png", summaries_dir,
        ascending=False, color=_SERIES_COLORS[1],
    )
    print(f"[INFO] Wrote time-to-zero-skill bar: {tzs_path}")

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
    parser.add_argument(
        "--std",
        action="store_true",
        help=(
            "Include σ-band uncertainty series in line plots and σ-rate columns in bar "
            "charts.  Omitted by default because σ is not comparable across datasets "
            "that use different model types."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()
    summaries_dir = (data_root / "summaries" / "horizons").resolve()
    print(f"[INFO] data_root : {data_root}")
    print(f"[INFO] summaries : {summaries_dir}")
    return generate_figures(
        data_root=data_root,
        prefix=args.dataset_prefix,
        summaries_dir=summaries_dir,
        show_std=args.std,
    )


if __name__ == "__main__":
    sys.exit(main())

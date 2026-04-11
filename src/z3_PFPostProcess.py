"""Post-process particle filter runs produced by z3_ParticleFilter.py.

Reads the incremental artifacts written by a particle-filter run (per-target
forecasts/<name>/evaluation_summary.csv + predictions.csv, pf_mlr_model_log.csv,
pf_run_metadata.json) and regenerates the cheap reporting outputs without
re-running the expensive MLR fitting / particle filter sweep.

Outputs written under ``<run-dir>/summaries/``:
  summary_best_model_performance.csv   -- one row per target, schema matching
                                          z1_FeaturePostProcess so this file is
                                          a drop-in input to z3_Compare.py.
  summary_best_model_performance.png   -- 3-panel (skill / nRMSE / R2) bar chart.
  pf_mlr_model_log.csv                 -- copy of the training-side log.

And, under ``<run-dir>/`` (overwriting the files already produced by z3_ParticleFilter):
  pf_comparison_{rmse,nrmse,r2,skill_vs_best}.png  -- requires --cv-dir.

CLI:
    python src/z3_ParticleFilterPostProcess.py --run-dir data/output/<pf-run>
    python src/z3_ParticleFilterPostProcess.py --run-dir data/output/<pf-run> --cv-dir data/output/CV14

Use ``z3_Compare.py`` afterwards to compare the resulting summary against other
runs, e.g.::

    python src/z3_Compare.py --root data/output/CV14 --root data/output/<pf-run> \
        --stat skill_vs_best_baseline --sort
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# Ensure src/ is on path when run as `python src/z3_ParticleFilterPostProcess.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.names import clean_target_label  # noqa: E402

# Reuse the particle-filter-side helpers that are pure functions over on-disk data.
from n_ParticleFilter import (  # noqa: E402
    _plot_model_comparison,
    _pf_forecast_name,
    _sanitise_for_dirname,
)

# Reuse the z1 plotting primitives and baseline constants so the summary figure
# matches summary_best_model_performance.png from the feature-sweep pipeline.
from z1_FeaturePostProcess import (  # noqa: E402
    BASELINE_ORDER,
    BASELINE_PLOT_COLORS,
    BASELINE_PLOT_LABELS,
    _annotate_bars_within_ylim,
    _draw_bar_group,
)

_PF_MODEL_LABEL = "particle_filter"
_PF_DISPLAY_LABEL = "Particle Filter"


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------

def _load_run_metadata(run_dir: Path) -> dict:
    """Load pf_run_metadata.json if present; return an empty dict otherwise.

    Older runs (written before the metadata dump was added) return an empty
    dict and the caller will fall back to scanning the filesystem.
    """
    path = run_dir / "pf_run_metadata.json"
    if not path.exists():
        print(f"[INFO] {path.name} not found; will infer forecast_name by scanning target dirs.")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not parse {path}: {exc}")
        return {}


def _infer_forecast_name_from_disk(run_dir: Path) -> str | None:
    """Scan <run_dir>/<target>/forecasts/ for the single PF forecast subdir.

    Used only when pf_run_metadata.json is missing (older runs). Returns the
    first forecast directory name that starts with "particle_filter".
    """
    for target_dir in sorted(run_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        forecasts_dir = target_dir / "forecasts"
        if not forecasts_dir.is_dir():
            continue
        for fc in sorted(forecasts_dir.iterdir()):
            if fc.is_dir() and fc.name.startswith("particle_filter"):
                return fc.name
    return None


def _discover_targets(run_dir: Path, forecast_name: str) -> list[str]:
    """Return the list of target names that have evaluation artifacts under run_dir.

    Walks <run_dir>/<sanitised_target>/forecasts/<forecast_name>/evaluation_summary.csv
    and reads the 'target' column of each to recover the original (pre-sanitisation)
    target name. This is the list we iterate over when building the summary CSV.
    """
    targets: list[str] = []
    for target_dir in sorted(run_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        eval_csv = target_dir / "forecasts" / forecast_name / "evaluation_summary.csv"
        if not eval_csv.exists():
            continue
        try:
            df = pd.read_csv(eval_csv)
        except Exception as exc:
            print(f"[WARN] Could not read {eval_csv}: {exc}")
            continue
        if df.empty or "target" not in df.columns:
            continue
        targets.append(str(df["target"].iloc[0]))
    return targets


# ---------------------------------------------------------------------------
# Per-target metric assembly
# ---------------------------------------------------------------------------

def _metrics_from_vectors(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, R2, and Pearson r from paired vectors.

    Mirrors the semantics used in z3_ParticleFilter._compute_evaluation_rows so
    baseline metrics recomputed here line up exactly with what the training
    script writes. Returns all-NaN on < 2 finite pairs.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson_r": np.nan}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if np.std(yt) > 0 and np.std(yp) > 0:
        pr = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pr = np.nan
    return {"mae": mae, "rmse": rmse, "r2": float(r2), "pearson_r": pr}


def _build_perf_row(
    target_col: str,
    run_dir: Path,
    forecast_name: str,
    dataset_prefix: str,
    mlr_log_by_target: dict[str, dict],
) -> dict | None:
    """Assemble one summary row for *target_col* from its on-disk PF artifacts.

    Reads the per-target evaluation_summary.csv (for the PF's own normalised
    metrics) and predictions.csv (to recompute baseline metrics in the same
    units), then joins the MLR log metadata. Returns None if required artifacts
    are missing.
    """
    target_dir = run_dir / _sanitise_for_dirname(target_col)
    forecast_dir = target_dir / "forecasts" / forecast_name
    eval_csv = forecast_dir / "evaluation_summary.csv"
    pred_csv = forecast_dir / "predictions.csv"

    if not eval_csv.exists():
        print(f"[WARN] Missing {eval_csv}; skipping target '{target_col}'.")
        return None

    try:
        eval_df = pd.read_csv(eval_csv)
    except Exception as exc:
        print(f"[WARN] Could not read {eval_csv}: {exc}; skipping target '{target_col}'.")
        return None

    norm_rows = eval_df[eval_df["label"].astype(str).str.contains("res_norm", na=False)]
    if norm_rows.empty:
        print(f"[WARN] No 'particle_filter (res_norm)' row in {eval_csv}; skipping.")
        return None
    pf_row = norm_rows.iloc[0]

    dataset_label = f"{dataset_prefix}_{_sanitise_for_dirname(target_col)}_res"

    row: dict = {
        "dataset":    dataset_label,
        "target":     target_col,
        "model":      _PF_MODEL_LABEL,
        "subset_rank": 1,
        "feature_tag": "pf",
        "forecast_name": forecast_name,
        "n_test_independent": float(pf_row.get("n_test_independent", np.nan)),
        "n_test_valid":       float(pf_row.get("n_test_valid", np.nan)),
        "n_test_valid_source": float(pf_row.get("n_test_valid", np.nan)),
        "mae":        float(pd.to_numeric(pf_row.get("mae"), errors="coerce")),
        "rmse":       float(pd.to_numeric(pf_row.get("rmse"), errors="coerce")),
        "r2":         float(pd.to_numeric(pf_row.get("r2"), errors="coerce")),
        "pearson_r":  float(pd.to_numeric(pf_row.get("pearson_r"), errors="coerce")),
        "std_target": float(pd.to_numeric(pf_row.get("std_target"), errors="coerce")),
    }
    if np.isfinite(row["std_target"]) and row["std_target"] > 0:
        row["nrmse"] = row["rmse"] / row["std_target"]
    else:
        row["nrmse"] = np.nan

    # ---- Baseline metrics from predictions.csv ----
    for kind in BASELINE_ORDER:
        for stat in ("mae", "rmse", "r2", "pearson_r"):
            row[f"{kind}_{stat}"] = np.nan

    if pred_csv.exists():
        try:
            pred_df = pd.read_csv(pred_csv)
        except Exception as exc:
            print(f"[WARN] Could not read {pred_csv}: {exc}")
            pred_df = pd.DataFrame()
        if not pred_df.empty and "target" in pred_df.columns:
            y_true = pd.to_numeric(pred_df["target"], errors="coerce").to_numpy(dtype=float)
            label_lookup = {
                "naive":    "Naive",
                "seasonal": "Seasonal",
                "linear":   "Linear",
            }
            for kind in BASELINE_ORDER:
                col = label_lookup[kind]
                if col not in pred_df.columns:
                    continue
                y_pred = pd.to_numeric(pred_df[col], errors="coerce").to_numpy(dtype=float)
                metrics = _metrics_from_vectors(y_true, y_pred)
                for stat, val in metrics.items():
                    row[f"{kind}_{stat}"] = val
    else:
        print(f"[WARN] {pred_csv} not found; baseline metrics will be NaN for '{target_col}'.")

    # ---- Skill scores ----
    model_rmse = row["rmse"]
    skill_vals: list[float] = []
    for kind in BASELINE_ORDER:
        base_rmse = row.get(f"{kind}_rmse", np.nan)
        if np.isfinite(model_rmse) and np.isfinite(base_rmse) and base_rmse > 0:
            skill = 1.0 - model_rmse / base_rmse
        else:
            skill = np.nan
        row[f"skill_vs_{kind}"] = skill
        if np.isfinite(skill):
            skill_vals.append(float(skill))
    row["skill_vs_best_baseline"] = max(skill_vals) if skill_vals else np.nan

    # ---- Compliance status (mirrors z1; PF has no evidence score so just
    # ---- gate on n_test_valid >= 5) ----
    n_valid = row.get("n_test_valid_source", np.nan)
    if np.isfinite(n_valid) and n_valid >= 5:
        row["compliance_status"] = "ok"
        row["compliance_reason"] = ""
    else:
        row["compliance_status"] = "failed"
        row["compliance_reason"] = "n_test_valid_source_below_5"

    # ---- MLR log metadata ----
    log_row = mlr_log_by_target.get(target_col, {})
    for key in (
        "variant_chosen", "process_noise_std", "measurement_noise_std",
        "prior_blend", "prior_blend_mode", "selected_features", "n_selected_features",
        "r2_train", "rmse_train", "n_train", "tuning_enabled",
    ):
        if key in log_row:
            row[f"pf_{key}"] = log_row[key]

    return row


# ---------------------------------------------------------------------------
# Summary-figure drawing (z1-style 3-panel bar chart)
# ---------------------------------------------------------------------------

def _write_summary_figure(perf_df: pd.DataFrame, output_path: Path) -> None:
    """Draw the 3-panel skill/nRMSE/R2 bar chart matching z1's layout."""
    if perf_df.empty:
        return

    # Sort by descending skill so the strongest targets appear first.
    perf_df = perf_df.sort_values(
        "skill_vs_best_baseline", ascending=False, na_position="last"
    ).reset_index(drop=True)

    labels = [
        clean_target_label(str(t), "") for t in perf_df.get("target", perf_df["dataset"])
    ]
    x = np.arange(len(perf_df))
    width = 0.20

    methods = [_PF_DISPLAY_LABEL] + [BASELINE_PLOT_LABELS[name] for name in BASELINE_ORDER]
    colors = ["tab:blue"] + [BASELINE_PLOT_COLORS[name] for name in BASELINE_ORDER]

    std_target_col = pd.to_numeric(perf_df["std_target"], errors="coerce").replace(0, np.nan)
    nrmse_data = [
        pd.to_numeric(perf_df["nrmse"], errors="coerce"),
        pd.to_numeric(perf_df["naive_rmse"], errors="coerce") / std_target_col,
        pd.to_numeric(perf_df["seasonal_rmse"], errors="coerce") / std_target_col,
        pd.to_numeric(perf_df["linear_rmse"], errors="coerce") / std_target_col,
    ]
    r2_data = [
        pd.to_numeric(perf_df["r2"], errors="coerce"),
        pd.to_numeric(perf_df["naive_r2"], errors="coerce"),
        pd.to_numeric(perf_df["seasonal_r2"], errors="coerce"),
        pd.to_numeric(perf_df["linear_r2"], errors="coerce"),
    ]

    skill_data = [
        pd.to_numeric(perf_df["skill_vs_naive"], errors="coerce"),
        pd.to_numeric(perf_df["skill_vs_seasonal"], errors="coerce"),
        pd.to_numeric(perf_df["skill_vs_linear"], errors="coerce"),
    ]
    skill_methods = [
        "Compared with Naive Baseline",
        "Compared with Seasonal Baseline",
        "Compared with Linear Baseline",
    ]
    skill_colors = [
        BASELINE_PLOT_COLORS["naive"],
        BASELINE_PLOT_COLORS["seasonal"],
        BASELINE_PLOT_COLORS["linear"],
    ]

    fig, (ax_skill, ax_nrmse, ax_r2) = plt.subplots(
        3, 1, figsize=(max(12, len(perf_df) * 0.8), 13), sharex=True
    )

    _draw_bar_group(
        ax_skill, x, width, skill_data, skill_colors, skill_methods, ".2f",
        center_offset=0.5,
    )
    ax_skill.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_skill.set_ylabel("Skill Score")
    ax_skill.grid(axis="y", alpha=0.3)
    ax_skill.legend()

    _draw_bar_group(ax_nrmse, x, width, nrmse_data, colors, methods, ".2e")
    ax_nrmse.set_ylabel("nRMSE")
    ax_nrmse.grid(axis="y", alpha=0.3)
    ax_nrmse.legend()

    r2_bars = _draw_bar_group(
        ax_r2, x, width, r2_data, colors, methods, ".2f", annotate=False
    )
    ax_r2.set_ylabel("Coefficient of Determination")
    ax_r2.set_ylim(-0.1, 1.0)
    for bars in r2_bars:
        _annotate_bars_within_ylim(ax_r2, bars, ".2f")
    ax_r2.grid(axis="y", alpha=0.3)
    ax_r2.legend()
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote summary figure: {output_path}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def post_process_run(
    run_dir: Path,
    cv_dir: Path | None,
    dataset_prefix: str,
    summaries_subdir: str,
    target_subset: list[str] | None,
    make_figures: bool,
) -> int:
    """Execute the full post-process workflow for a single run directory."""
    if not run_dir.is_dir():
        print(f"[ERROR] --run-dir does not exist: {run_dir}")
        return 1

    metadata = _load_run_metadata(run_dir)
    forecast_name = metadata.get("forecast_name")
    if not forecast_name:
        forecast_name = _infer_forecast_name_from_disk(run_dir)
    if not forecast_name:
        print("[ERROR] Could not determine forecast_name; no particle_filter_* subdirs found.")
        return 1
    print(f"[INFO] forecast_name: {forecast_name}")

    # Load MLR log keyed by target
    mlr_log_path = run_dir / "pf_mlr_model_log.csv"
    mlr_log_by_target: dict[str, dict] = {}
    if mlr_log_path.exists():
        try:
            df_log = pd.read_csv(mlr_log_path)
            if "target" in df_log.columns:
                mlr_log_by_target = {
                    str(r["target"]): {k: r[k] for k in df_log.columns if k != "target"}
                    for _, r in df_log.iterrows()
                }
        except Exception as exc:
            print(f"[WARN] Could not read {mlr_log_path}: {exc}")
    else:
        print(f"[WARN] {mlr_log_path} not found; pf_* metadata columns will be empty.")

    # Discover targets from on-disk forecasts/<name>/ subdirs
    all_targets = _discover_targets(run_dir, forecast_name)
    if not all_targets:
        print(f"[ERROR] No targets with evaluation_summary.csv under {run_dir}/*/forecasts/{forecast_name}/")
        return 1
    if target_subset:
        targets = [t for t in all_targets if t in set(target_subset)]
        missing = [t for t in target_subset if t not in all_targets]
        if missing:
            print(f"[WARN] --targets not found on disk: {missing}")
    else:
        targets = all_targets
    print(f"[INFO] Processing {len(targets)} target(s).")

    # Build per-target summary rows
    summary_rows: list[dict] = []
    for target_col in targets:
        row = _build_perf_row(
            target_col=target_col,
            run_dir=run_dir,
            forecast_name=forecast_name,
            dataset_prefix=dataset_prefix,
            mlr_log_by_target=mlr_log_by_target,
        )
        if row is not None:
            summary_rows.append(row)

    if not summary_rows:
        print("[ERROR] No summary rows built; aborting.")
        return 1

    perf_df = pd.DataFrame(summary_rows).sort_values("r2", ascending=False).reset_index(drop=True)

    summaries_dir = run_dir / summaries_subdir
    summaries_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = summaries_dir / "summary_best_model_performance.csv"
    perf_df.to_csv(summary_csv, index=False)
    print(f"[INFO] Wrote summary CSV: {summary_csv}")

    # Copy the MLR log into summaries/ for convenience.
    if mlr_log_path.exists():
        shutil.copy2(mlr_log_path, summaries_dir / mlr_log_path.name)

    if make_figures:
        _write_summary_figure(perf_df, summaries_dir / "summary_best_model_performance.png")

        combined_eval_path = run_dir / "evaluation_summary_all_targets.csv"
        if cv_dir is not None and combined_eval_path.exists():
            print("[INFO] Regenerating pf_comparison_*.png ...")
            _plot_model_comparison(
                cv_dir=cv_dir,
                pf_eval_all_targets_path=combined_eval_path,
                target_cols=list(targets),
                output_dir=run_dir,
            )
        elif cv_dir is None:
            print("[INFO] --cv-dir not supplied; skipping pf_comparison_*.png.")
        else:
            print(f"[WARN] {combined_eval_path} missing; skipping pf_comparison_*.png.")

    print("[INFO] Done.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a particle-filter run produced by z3_ParticleFilter.py: "
            "reload incremental artifacts and regenerate summary CSVs and figures "
            "without re-running the filter."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Particle-filter run output directory (the --output-dir used by z3_ParticleFilter.py).",
    )
    parser.add_argument(
        "--cv-dir",
        type=str,
        default=None,
        help="Baseline CV output directory (needed for pf_comparison_*.png).",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="MC",
        help="Prefix used when building the dataset label column (default: MC).",
    )
    parser.add_argument(
        "--summaries-subdir",
        type=str,
        default="summaries",
        help="Subdirectory under --run-dir for summary outputs (default: summaries).",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=None,
        help="Subset of targets to post-process (default: all found on disk).",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip all figure regeneration.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    workspace_root = Path(__file__).resolve().parent.parent

    def _resolve(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = (workspace_root / path).resolve()
        return path

    run_dir = _resolve(args.run_dir)
    cv_dir = _resolve(args.cv_dir) if args.cv_dir else None

    print(f"[INFO] run_dir : {run_dir}")
    print(f"[INFO] cv_dir  : {cv_dir}")

    return post_process_run(
        run_dir=run_dir,
        cv_dir=cv_dir,
        dataset_prefix=args.dataset_prefix,
        summaries_subdir=args.summaries_subdir,
        target_subset=args.targets,
        make_figures=not args.no_figure,
    )


if __name__ == "__main__":
    sys.exit(main())

"""
Generates horizon-sweep comparison figures across all MC datasets.

Reads ``lookahead_metrics.csv`` from each dataset's ``horizons/lookahead_sweeps/``
directory (written by ``k_RunHorizonSweep.py``) and produces:

  1. ``lookahead_r2_comparison.png``    – R² vs forecast horizon (hours)
  2. ``lookahead_nrmse_comparison.png`` – nRMSE vs forecast horizon (hours)
  3. ``lookahead_skill_comparison.png``     – skill vs. best baseline per horizon
    4. ``lookahead_time_to_zero_skill.png``   – initial skill / skill rate (days to zero skill)
    5. ``lookahead_time_to_baseline_r2.png``  – time for R² to reach best baseline R² (days)
    6. ``lookahead_aggregate.csv``            – combined table of all datasets × horizons × replicates
    7. ``lookahead_time_to_baseline_r2_repro.csv`` – per-horizon values used to reproduce the timing calculation

nRMSE (= RMSE / std_target) is used instead of raw RMSE so that datasets with
very different target magnitudes can be compared on the same axis.  The
``std_target`` value is read from each dataset's
``forecasts/feature_sweeps/feature_sweep_final_metrics.csv`` (populated by
``z1_PostProcess.py``).

All outputs are written to the ``summaries/horizons/eval_test/`` and
``summaries/horizons/eval_all/`` subdirectories of the data root.

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
    --ml-selection MODE     Which ML-family horizon results to include. "best"
                            keeps the current behavior (best of XGB, GP,
                            Transformer). "xgb" restricts ML-family results to
                            XGB only. Default: best
    --treat-mlr-as-baseline Include MLR as an additional baseline candidate
                            when computing best-baseline skill summaries.
        --evaluate-all          Deprecated compatibility flag; no longer needed.
                                                        This script now always generates both:
                                                            - eval_test : test-set only evaluation
                                                            - eval_all  : combined train+test evaluation
                                                        For eval_all, when combined-set metrics are missing,
                                                        f_Evaluate.py is re-run with evaluate_all=true against
                                                        saved model weights (no retraining).

Examples:
python src/z2_HorizonPostProcess.py
python src/z2_HorizonPostProcess.py --data-root data/output/regression
python src/z2_HorizonPostProcess.py --ml-selection xgb
python src/z2_HorizonPostProcess.py --treat-mlr-as-baseline
python src/z2_HorizonPostProcess.py --data-root data/output/CV14 --dataset-prefix MC
python src/z2_HorizonPostProcess.py --data-root data/output/CV14 --dataset-prefix MC --evaluate-all
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
import tempfile
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
from utils.plotstyle import legend_above

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

_BAR_LABEL_FONTSIZE = 12
_BAR_ANNOTATION_FONTSIZE = 12
_BAR_LEGEND_FONTSIZE = 12
_CLUSTERED_BAR_WIDTH = 0.50
_SINGLE_BAR_WIDTH = 0.86
_MLR_MODEL_NAMES = {"mlr", "mlr_avg12", "mlr_avgall"}
_XGB_MODEL_NAMES = {"xgb", "xgbregressor", "xgb_regressor", "xgbclassifier", "xgb_classifier"}


def _normalize_model_id(value: object) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


_NORMALIZED_MLR_MODEL_NAMES = {_normalize_model_id(v) for v in _MLR_MODEL_NAMES}
_NORMALIZED_XGB_MODEL_NAMES = {_normalize_model_id(v) for v in _XGB_MODEL_NAMES}
_DEFAULT_BASELINE_LABELS = ("naive", "seasonal", "linear")


def _is_mlr_model_name(value: object) -> bool:
    return _normalize_model_id(value) in _NORMALIZED_MLR_MODEL_NAMES


def _is_xgb_model_name(value: object) -> bool:
    return _normalize_model_id(value) in _NORMALIZED_XGB_MODEL_NAMES


def _baseline_labels(include_mlr_baseline: bool = False) -> tuple[str, ...]:
    return _DEFAULT_BASELINE_LABELS + (("mlr",) if include_mlr_baseline else ())


def _bar_fig_width(n_bars: int, clustered: bool) -> float:
    """Return a compact figure width that still leaves labels readable."""
    per_dataset = 0.72 if clustered else 0.66
    return max(5.4, per_dataset * max(n_bars, 1) + 1.4)


def _set_bar_xlim(ax: plt.Axes, x: np.ndarray, bar_w: float, clustered: bool) -> None:
    """Keep edge margins small and proportional to the bar width."""
    if len(x) == 0:
        return
    half_span = bar_w if clustered else bar_w / 2
    margin = 0.20 * bar_w
    ax.set_xlim(x[0] - half_span - margin, x[-1] + half_span + margin)


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


def _find_horizon_forecast_dirs(dataset_dir: Path) -> list[tuple[int, int, Path, "str | None"]]:
    """Discover ``(horizon, replicate, forecast_dir, model_class)`` tuples under *dataset_dir*/horizons/.

    Searches the following layouts (most-specific first):

    * New per-class layout::

        horizons/NNNhr/ml/forecasts/rep_RRR/
        horizons/NNNhr/mlr/forecasts/rep_RRR/

    * Legacy flat layout (pre-migration)::

        horizons/NNNhr/forecasts/rep_RRR/     (model_class=None)
    """
    _horizon_re = re.compile(r"^(\d+)hr$")
    _rep_re = re.compile(r"^rep_(\d+)$")
    hits: list[tuple[int, int, Path, "str | None"]] = []
    horizons_root = dataset_dir / "horizons"
    if not horizons_root.is_dir():
        return hits
    for h_dir in sorted(horizons_root.iterdir()):
        m_h = _horizon_re.match(h_dir.name)
        if not m_h:
            continue
        horizon = int(m_h.group(1))
        found_class_dirs = False
        # New layout: ml/ and mlr/ subdirectories
        for model_class in ("ml", "mlr"):
            class_dir = h_dir / model_class
            forecasts_root = class_dir / "forecasts"
            if not forecasts_root.is_dir():
                continue
            found_class_dirs = True
            for fc_dir in sorted(forecasts_root.iterdir()):
                m_r = _rep_re.match(fc_dir.name)
                if m_r and fc_dir.is_dir():
                    hits.append((horizon, int(m_r.group(1)), fc_dir, model_class))
        # Legacy flat layout — only if no class subdirs found
        if not found_class_dirs:
            forecasts_root = h_dir / "forecasts"
            if forecasts_root.is_dir():
                for fc_dir in sorted(forecasts_root.iterdir()):
                    m_r = _rep_re.match(fc_dir.name)
                    if m_r and fc_dir.is_dir():
                        hits.append((horizon, int(m_r.group(1)), fc_dir, None))
    return hits


def _run_pending_evaluations(dataset_dir: Path) -> int:
    """Run ``f_Evaluate.py`` for forecast dirs that have an eval config but no evaluation_summary.csv.

    Returns the number of evaluations successfully completed.
    """
    completed = 0
    for horizon, rep_idx, fc_dir, _mc in _find_horizon_forecast_dirs(dataset_dir):
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


def _ensure_combined_metrics_mlr(fc_dir: Path) -> bool:
    """Compute combined (train+test) metrics for an MLR replicate directory.

    MLR reps have no ``config_evaluate_*.yml`` and no saved model weights.  Instead
    they store ``model_config.json`` (model_type, input_columns, output_columns),
    ``train_files.txt``, ``test_files.txt``, and derive the data-dimension config
    from ``config.yml`` at the class-directory level (``fc_dir.parent.parent``).

    The MLR model is re-fit on the original train samples and then evaluated on all
    samples (train+test combined).  A ``kind="combined"`` row is appended to
    ``evaluation_summary.csv``.  Combined baseline rows are also written to
    ``combined_baseline_summary.csv`` at the class directory level using
    ``_append_mlr_baseline_outputs`` from ``h_RunMCFeatureSelectionSweep``.

    Returns True on success, False on failure.
    """
    import json as _json

    summary_csv = fc_dir / "evaluation_summary.csv"
    model_cfg_path = fc_dir / "model_config.json"
    train_txt = fc_dir / "train_files.txt"
    test_txt = fc_dir / "test_files.txt"

    if not model_cfg_path.exists():
        print(f"  [WARN] MLR combined: model_config.json not found in {fc_dir}")
        return False
    if not train_txt.exists() or not test_txt.exists():
        print(f"  [WARN] MLR combined: train_files.txt / test_files.txt not found in {fc_dir}")
        return False

    try:
        with open(model_cfg_path, "r", encoding="utf-8") as fh:
            mcfg = _json.load(fh)
    except Exception as exc:
        print(f"  [WARN] MLR combined: could not read model_config.json: {exc}")
        return False

    model_type = str(mcfg.get("model_type", "mlr")).strip().lower()
    if model_type not in {"mlr", "mlr_avg12", "mlr_avgall"}:
        print(f"  [WARN] MLR combined: unexpected model_type '{model_type}' in {fc_dir}")
        return False

    input_columns = list(mcfg.get("input_columns", []))
    output_columns = list(mcfg.get("output_columns", []))
    if not input_columns or not output_columns:
        print(f"  [WARN] MLR combined: empty input/output columns in model_config.json")
        return False

    _MLR_AGG_MODE = {"mlr": "last", "mlr_avg12": "avg12", "mlr_avgall": "avgall"}
    aggregation_mode = _MLR_AGG_MODE.get(model_type, "last")

    # fc_dir = horizons/NNNhr/mlr/forecasts/rep_NNN/
    class_dir = fc_dir.parent.parent

    # Derive input_row_1 / input_row_2 / output_rows.  New artifacts store these
    # in model_config.json; older ones do not.  For older artifacts, read the row
    # count directly from a sample CSV: samples always span rows 0..N-1 where row
    # N-1 is the output row.
    input_row_1 = mcfg.get("input_row_1")
    input_row_2 = mcfg.get("input_row_2")
    output_rows = mcfg.get("output_rows")

    if input_row_1 is None or input_row_2 is None or output_rows is None:
        sample_dir = class_dir / "samples"
        _sample_files = sorted(sample_dir.glob("*.csv")) if sample_dir.is_dir() else []
        if not _sample_files:
            print(f"  [WARN] MLR combined: cannot determine row config — no samples in {sample_dir}")
            return False
        try:
            _n_rows = sum(1 for _ in open(_sample_files[0], encoding="utf-8")) - 1  # subtract header
        except Exception as exc:
            print(f"  [WARN] MLR combined: could not read sample file: {exc}")
            return False
        input_row_1 = 0
        input_row_2 = _n_rows - 1
        output_rows = [_n_rows - 1]

    input_row_1 = int(input_row_1)
    input_row_2 = int(input_row_2)
    output_rows = list(output_rows)
    input_rows = slice(input_row_1, input_row_2)
    base_cfg: dict = {}

    # Inject absolute paths for baseline evaluation into base_cfg so that
    # merge_eval_config in _append_mlr_baseline_outputs does not fall back to
    # DEFAULT_EVAL_CONFIG's relative defaults, which resolve incorrectly from
    # the class dir.  These files are at fixed locations relative to the project root.
    _project_root = Path(__file__).resolve().parent.parent
    _historic_abs = _project_root / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    base_cfg.setdefault("evaluation", {})["historic_path"] = str(_historic_abs)

    # Load samples
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from utils.training import load_samples as _load_samples
        from utils.mlr import evaluate_mlr as _eval_mlr
        import f_Evaluate as eval_module
    except Exception as exc:
        print(f"  [WARN] MLR combined: import failed: {exc}")
        return False

    sample_dir = class_dir / "samples"
    if not sample_dir.is_dir():
        print(f"  [WARN] MLR combined: samples dir not found: {sample_dir}")
        return False

    try:
        train_samples = _load_samples(
            sample_dir, input_columns=input_columns, output_columns=output_columns,
            input_rows=input_rows, output_rows=output_rows,
            source=train_txt, fault_tolerant=True, input_aggregation="none",
        )
        test_samples = _load_samples(
            sample_dir, input_columns=input_columns, output_columns=output_columns,
            input_rows=input_rows, output_rows=output_rows,
            source=test_txt, fault_tolerant=True, input_aggregation="none",
        )
    except Exception as exc:
        print(f"  [WARN] MLR combined: sample loading failed: {exc}")
        return False

    if len(train_samples) < 2:
        print(f"  [WARN] MLR combined: too few train samples ({len(train_samples)}) in {fc_dir}")
        return False

    all_samples = train_samples + test_samples
    if len(all_samples) < 2:
        print(f"  [WARN] MLR combined: no samples loaded for {fc_dir}")
        return False

    # Fit on train, predict on all (no rebalancing — combined evaluation is read-only)
    try:
        preds, targets, _ = _eval_mlr(
            train_samples=train_samples,
            test_samples=all_samples,
            feature_names=input_columns,
            selection_config=None,
            aggregation_mode=aggregation_mode,
            use_spearman_prefilter=True,
        )
    except Exception as exc:
        print(f"  [WARN] MLR combined: evaluate_mlr failed: {exc}")
        return False

    all_split_files = [str(s[2]) for s in all_samples]
    _model_label = model_type.upper().replace("_", "-")
    combined_row = eval_module._compute_regression_summary(
        f"{_model_label} (combined)",
        preds,
        targets,
        len(all_samples),
        metadata={"kind": "combined", "gp_uncertainty_mode": "not_gp"},
        split_files=all_split_files,
    )
    combined_row["n_train_samples"] = len(train_samples)
    combined_row["n_test_samples"] = len(test_samples)
    combined_row["input_dim"] = len(input_columns)
    combined_row["target_dim"] = len(output_columns)
    combined_row["data_dir"] = str(class_dir)

    # Append combined row to evaluation_summary.csv (preserve existing rows)
    try:
        if summary_csv.exists():
            existing_df = pd.read_csv(summary_csv)
            # Remove any stale combined rows before appending
            if "kind" in existing_df.columns:
                existing_df = existing_df[existing_df["kind"] != "combined"]
            existing_rows = existing_df.to_dict(orient="records")
        else:
            existing_rows = []
        eval_module._write_summary_csv(existing_rows + [combined_row], summary_csv)
    except Exception as exc:
        print(f"  [WARN] MLR combined: could not write evaluation_summary.csv: {exc}")
        return False

    # Write combined_baseline_summary.csv at class level if not already present.
    combined_bl_csv = class_dir / "combined_baseline_summary.csv"
    if not combined_bl_csv.exists():
        try:
            import h_RunMCFeatureSelectionSweep as _h
            # Build a minimal ref_cfg carrying only the fields _append_mlr_baseline_outputs
            # needs.  base_cfg may be empty for older artifacts that had no config_*.yml.
            _ref_cfg = {"evaluation": {"historic_path": base_cfg.get("evaluation", {}).get("historic_path", "")}}
            _ref_data_cfg = {"output_rows": output_rows, "output_columns": output_columns}
            # _append_mlr_baseline_outputs requires a real path for _resolve_path_from_config;
            # supply the project root so any remaining relative paths resolve correctly.
            _ref_cfg_path = Path(__file__).resolve().parent.parent / "src" / "z2_HorizonPostProcess.py"
            baseline_rows: list[dict] = []
            _h._append_mlr_baseline_outputs(
                baseline_rows,
                [],
                ref_cfg=_ref_cfg,
                ref_cfg_path=_ref_cfg_path,
                ref_data_cfg=_ref_data_cfg,
                data_dir=str(class_dir.resolve()),
                sample_subdir="samples",
                output_columns=output_columns,
                output_rows=output_rows,
                forecast_name="",
                test_samples=all_samples,
                test_split_files=all_split_files,
            )
            if baseline_rows:
                eval_module._write_summary_csv(baseline_rows, combined_bl_csv)
        except Exception as exc:
            print(f"  [WARN] MLR combined: could not write combined_baseline_summary.csv: {exc}")

    return True


def _ensure_combined_metrics(fc_dir: Path) -> bool:
    """Ensure ``evaluation_summary.csv`` in *fc_dir* contains a ``combined`` row.

    If a ``combined`` row is already present, returns True immediately.  Otherwise,
    locates the ``config_evaluate_*.yml`` in *fc_dir*, writes a temporary copy with
    ``evaluate_all: true`` and ``run_baselines: true``, runs ``f_Evaluate.py
    --no-plots`` against it, then removes the temporary config.

    For ML models (XGB, GP, Transformer): reads the eval config, re-runs
    ``f_Evaluate.py --no-plots`` with ``evaluate_all: true`` and
    ``run_baselines: true``.  After a successful re-run, the baseline rows
    (``kind="baseline"``) are extracted and cached as
    ``combined_baseline_summary.csv`` at the model-class directory level
    (``horizons/NNNhr/{model_class}/``).

    For MLR reps (no ``config_evaluate_*.yml``): delegates to
    ``_ensure_combined_metrics_mlr``, which re-fits on train samples and
    evaluates on all samples inline.

    Returns True when a ``combined`` row is present after the attempt, False on
    failure.
    """
    summary_csv = fc_dir / "evaluation_summary.csv"
    # Check whether combined row already exists.
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            kind_col = "kind" if "kind" in df.columns else ("label" if "label" in df.columns else None)
            if kind_col == "kind" and (df["kind"] == "combined").any():
                return True
            if kind_col == "label" and df["label"].str.contains("combined", case=False, na=False).any():
                return True
        except Exception:
            pass

    eval_cfgs = list(fc_dir.glob("config_evaluate_*.yml"))
    if not eval_cfgs:
        # MLR reps have no eval config — use inline MLR combined evaluation.
        return _ensure_combined_metrics_mlr(fc_dir)
    eval_cfg_path = eval_cfgs[0]
    try:
        with open(eval_cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        cfg.setdefault("evaluation", {})["evaluate_all"] = True
        cfg["evaluation"]["run_baselines"] = True
        # Overwrite the three reference paths with their fixed absolute locations.
        # Configs written by older e_Train.py versions may have the wrong number of
        # '..' levels; computing from the project root avoids any relative-path
        # ambiguity.
        _project_root = Path(__file__).resolve().parent.parent
        _eval_section = cfg.setdefault("evaluation", {})
        _eval_section["historic_path"]      = str(_project_root / "data" / "output" / "regression" / "Consolidated_sparse.csv")
        _eval_section["normalization_path"] = str(_project_root / "data" / "output" / "sensors"     / "normalization.json")
        _eval_section["thresholds_path"]    = str(_project_root / "data" / "input"                  / "Limits.csv")
        _data_section = cfg.get("data", {})
        _dval = _data_section.get("data_dir")
        if _dval and not Path(_dval).is_absolute():
            _data_section["data_dir"] = str((fc_dir / _dval).resolve())
        # Write a temporary config in the same directory so any remaining relative
        # paths (split files, etc.) still resolve correctly.
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            suffix=".yml", prefix="_tmp_evalall_", dir=fc_dir
        )
        tmp_path = Path(tmp_path_str)
        try:
            with open(tmp_fd, "w", encoding="utf-8") as fh:
                yaml.dump(cfg, fh, allow_unicode=True, default_flow_style=False)
            result = subprocess.run(
                [sys.executable, "src/f_Evaluate.py", "--config", str(tmp_path),
                 "--no-plots"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                stderr_text = result.stderr.decode(errors="replace") if result.stderr else ""
                print(f"  [WARN] evaluate_all re-run failed for {fc_dir.name}:")
                for line in stderr_text.strip().splitlines()[-3:]:
                    print(f"         {line}")
                return False
            # Cache combined baseline RMSEs at the class level so that
            # _load_baseline_rmses can find them in subsequent post-process runs.
            # fc_dir = horizons/NNNhr/{model_class}/forecasts/rep_NNN/
            class_dir = fc_dir.parent.parent
            combined_bl_csv = class_dir / "combined_baseline_summary.csv"
            if not combined_bl_csv.exists():
                try:
                    _df = pd.read_csv(summary_csv)
                    if "kind" in _df.columns:
                        _bl = _df[_df["kind"] == "baseline"]
                        if not _bl.empty:
                            _bl.to_csv(combined_bl_csv, index=False)
                except Exception:
                    pass
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"  [WARN] _ensure_combined_metrics failed for {fc_dir}: {exc}")
        return False

    # Verify
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            if "kind" in df.columns and (df["kind"] == "combined").any():
                return True
            if "label" in df.columns and df["label"].str.contains("combined", case=False, na=False).any():
                return True
        except Exception:
            pass
    return False


def _reconstruct_lookahead_metrics(
    dataset_dir: Path,
    evaluate_all: bool = False,
) -> pd.DataFrame | None:
    """Build a ``lookahead_metrics.csv``-equivalent DataFrame by scanning evaluation_summary.csv files.

    Mimics the filtering logic in ``k_RunHorizonSweep.py``: keeps only test-set
    rows (or ``combined`` rows when *evaluate_all* is True), drops the
    ``label``/``kind`` columns, and adds ``horizon`` / ``replicate`` columns
    derived from the directory structure.

    When *evaluate_all* is True and a rep dir lacks a ``combined`` row,
    ``_ensure_combined_metrics`` is called.  For ML models this re-runs
    ``f_Evaluate.py`` with ``evaluate_all: true``; for MLR reps it re-fits
    and evaluates inline.  Falls back to the test row only if that fails.
    """
    frames: list[pd.DataFrame] = []
    for horizon, rep_idx, fc_dir, model_class in _find_horizon_forecast_dirs(dataset_dir):
        summary_csv = fc_dir / "evaluation_summary.csv"
        if not summary_csv.exists():
            continue
        # When evaluate_all is requested, ensure combined rows exist (re-running
        # f_Evaluate.py against the saved model if needed).
        if evaluate_all:
            ok = _ensure_combined_metrics(fc_dir)
            if ok is False:
                print(f"  [WARN] Could not obtain combined metrics for {fc_dir}; falling back to test rows.")
        try:
            df = pd.read_csv(summary_csv)
        except Exception:
            continue
        # Filter to combined rows when evaluate_all is set; fall back to test rows
        # if no combined row is available.
        if evaluate_all:
            if "kind" in df.columns:
                combined = df[df["kind"] == "combined"]
                df = combined if not combined.empty else df[df["kind"] == "test"]
            elif "label" in df.columns:
                combined = df[df["label"].str.contains("combined", case=False, na=False)]
                df = combined if not combined.empty else df[df["label"].str.contains("test", case=False, na=False)]
        else:
            # Default: test rows only (original behaviour)
            if "kind" in df.columns:
                df = df[df["kind"] == "test"]
            elif "label" in df.columns:
                df = df[df["label"].str.contains("test", case=False, na=False)]
        df = df.drop(columns=[c for c in ("label", "kind") if c in df.columns], errors="ignore")
        df["horizon"] = horizon
        df["replicate"] = rep_idx
        # model_class and model_name — try model_config.json in the rep dir first.
        _mc = model_class  # may be None for legacy flat layout
        _mn: "str | None" = None
        model_cfg_path = fc_dir / "model_config.json"
        if model_cfg_path.exists():
            try:
                import json as _json
                with open(model_cfg_path, "r", encoding="utf-8") as _fh:
                    _mcfg = _json.load(_fh)
                _mt = str(_mcfg.get("model_type", "")).strip().lower()
                if _mt:
                    _mn = _mt
                    if _mc is None:
                        _mc = "mlr" if _mt in {"mlr", "mlr_avg12", "mlr_avgall"} else "ml"
            except Exception:
                pass
        if _mc is not None:
            df["model_class"] = _mc
        if _mn is not None:
            df["model_name"] = _mn
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
    # Put model_class, model_name, horizon, replicate first
    front = [c for c in ("model_class", "model_name", "horizon", "replicate") if c in result.columns]
    rest = [c for c in result.columns if c not in front]
    return result[front + rest]


def _clean_label(dataset_name: str, prefix: str) -> str:
    """Delegate to the shared ``clean_target_label`` in ``utils.names``."""
    return clean_target_label(dataset_name, prefix)


def _load_baseline_rmses(
    dataset_dir: Path,
    horizon_hr: int,
    replicate: int,
    model_class: "str | None" = None,
    include_mlr_baseline: bool = False,
    evaluate_all: bool = False,
) -> dict[str, float]:
    """Return ``{baseline_label: rmse}`` from a baseline summary CSV for one horizon.

    Baselines are shared across replicates; the *replicate* parameter is accepted
    for interface compatibility but ignored.

    When *evaluate_all* is True, ``combined_baseline_summary.csv`` (written by
    ``_ensure_combined_metrics``) is checked first.  This file contains baseline
    metrics computed over the combined train+test sample set, giving correct skill
    scores when the model is also evaluated over all samples.

    Search order when *evaluate_all* is True:
      1. ``horizons/NNNhr/{model_class}/combined_baseline_summary.csv``
      2. ``horizons/NNNhr/combined_baseline_summary.csv``
      3. ``horizons/NNNhr/{model_class}/baseline_summary.csv``
      4. ``horizons/NNNhr/baseline_summary.csv``

    Search order when *evaluate_all* is False (default):
      1. ``horizons/NNNhr/{model_class}/baseline_summary.csv``
      2. ``horizons/NNNhr/baseline_summary.csv``

    Returns an empty dict when no file is found or readable.
    """
    h_label = f"{horizon_hr:03d}hr"
    candidates: list[Path] = []
    if evaluate_all:
        if model_class is not None:
            candidates.append(dataset_dir / "horizons" / h_label / model_class / "combined_baseline_summary.csv")
        candidates.append(dataset_dir / "horizons" / h_label / "combined_baseline_summary.csv")
    if model_class is not None:
        candidates.append(dataset_dir / "horizons" / h_label / model_class / "baseline_summary.csv")
    # Always fall back to flat layout
    candidates.append(dataset_dir / "horizons" / h_label / "baseline_summary.csv")
    baseline_csv: "Path | None" = None
    for c in candidates:
        if c.exists():
            baseline_csv = c
            break
    if baseline_csv is None:
        return {}
    try:
        df = pd.read_csv(baseline_csv)
        if "rmse" not in df.columns:
            return {}
        _known = set(_DEFAULT_BASELINE_LABELS)
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
        if include_mlr_baseline:
            mlr_csv = (
                dataset_dir / "horizons" / "lookahead_sweeps" / "_combined_metrics.csv"
                if evaluate_all
                else dataset_dir / "horizons" / "lookahead_sweeps" / "lookahead_metrics.csv"
            )
            if mlr_csv.exists():
                try:
                    mlr_df = pd.read_csv(mlr_csv)
                    horizon_col = "horizon" if "horizon" in mlr_df.columns else "lookahead"
                    if horizon_col in mlr_df.columns and "rmse" in mlr_df.columns:
                        mlr_subset = mlr_df[pd.to_numeric(mlr_df[horizon_col], errors="coerce") == int(horizon_hr)].copy()
                        if "model_class" in mlr_subset.columns:
                            mlr_subset = mlr_subset[mlr_subset["model_class"].astype(str).str.lower() == "mlr"].copy()
                        elif "model_name" in mlr_subset.columns:
                            mlr_subset = mlr_subset[mlr_subset["model_name"].apply(_is_mlr_model_name)].copy()
                        if evaluate_all and "kind" in mlr_subset.columns:
                            combined = mlr_subset[mlr_subset["kind"] == "combined"]
                            mlr_subset = combined if not combined.empty else mlr_subset[mlr_subset["kind"] == "test"]
                        elif "kind" in mlr_subset.columns:
                            mlr_subset = mlr_subset[mlr_subset["kind"] == "test"]
                        mlr_rmse = pd.to_numeric(mlr_subset["rmse"], errors="coerce").dropna()
                        mlr_rmse = mlr_rmse[mlr_rmse > 0]
                        if not mlr_rmse.empty:
                            result["mlr"] = float(mlr_rmse.min())
                except Exception:
                    pass
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


def _rate_table(records: list[tuple[str, pd.DataFrame]], include_mlr_baseline: bool = False) -> pd.DataFrame:
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

        # Also get initial R2, RMSE, and baseline metrics at min horizon
        _nrmse_at_min_h = float(means["nrmse"].loc[_min_h]) if _min_h in means.index and pd.notna(means["nrmse"].loc[_min_h]) else float("nan")
        _r2_at_min_h = float(means["r2"].loc[_min_h]) if _min_h in means.index and pd.notna(means["r2"].loc[_min_h]) else float("nan")
        _rmse_at_min_h = float(means["rmse"].loc[_min_h]) if _min_h in means.index and pd.notna(means["rmse"].loc[_min_h]) else float("nan")
        
        # Get individual baseline skills at min horizon to determine which is best
        _df_at_min_h = df[df["lookahead"] == _min_h]
        _skill_v_naive_at_min = float(_df_at_min_h["skill_v_naive"].mean()) if "skill_v_naive" in _df_at_min_h.columns and not _df_at_min_h["skill_v_naive"].isnull().all() else float("nan")
        _skill_v_seasonal_at_min = float(_df_at_min_h["skill_v_seasonal"].mean()) if "skill_v_seasonal" in _df_at_min_h.columns and not _df_at_min_h["skill_v_seasonal"].isnull().all() else float("nan")
        _skill_v_linear_at_min = float(_df_at_min_h["skill_v_linear"].mean()) if "skill_v_linear" in _df_at_min_h.columns and not _df_at_min_h["skill_v_linear"].isnull().all() else float("nan")
        _skill_v_mlr_at_min = float(_df_at_min_h["skill_v_mlr"].mean()) if include_mlr_baseline and "skill_v_mlr" in _df_at_min_h.columns and not _df_at_min_h["skill_v_mlr"].isnull().all() else float("nan")

        baseline_candidates = {
            "naive": _skill_v_naive_at_min,
            "seasonal": _skill_v_seasonal_at_min,
            "linear": _skill_v_linear_at_min,
        }
        if include_mlr_baseline:
            baseline_candidates["mlr"] = _skill_v_mlr_at_min
        finite_baselines = {name: score for name, score in baseline_candidates.items() if np.isfinite(score)}
        if finite_baselines:
            best_baseline_label, best_skill = max(finite_baselines.items(), key=lambda item: item[1])
            if np.isfinite(_rmse_at_min_h) and _rmse_at_min_h > 0 and np.isfinite(_nrmse_at_min_h) and _nrmse_at_min_h > 0 and best_skill < 1.0:
                _baseline_rmse_at_min = _rmse_at_min_h / (1.0 - best_skill)
                _std_target_at_min = _rmse_at_min_h / _nrmse_at_min_h
                _raw_initial_baseline_r2 = 1.0 - (_baseline_rmse_at_min / _std_target_at_min) ** 2 if _std_target_at_min > 0 else float("nan")
                _initial_baseline_r2 = max(0.0, _raw_initial_baseline_r2) if np.isfinite(_raw_initial_baseline_r2) else float("nan")
            else:
                _raw_initial_baseline_r2 = float("nan")
                _initial_baseline_r2 = float("nan")
        else:
            best_baseline_label = ""
            _raw_initial_baseline_r2 = float("nan")
            _initial_baseline_r2 = float("nan")

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
            "initial_nrmse":   _nrmse_at_min_h,
            "initial_r2":      _r2_at_min_h,
            "raw_initial_baseline_r2": _raw_initial_baseline_r2,
            "initial_baseline_r2": _initial_baseline_r2,
            "best_baseline_label": best_baseline_label,
            "initial_rmse":    _rmse_at_min_h,
            "initial_skill_v_naive":    _skill_v_naive_at_min,
            "initial_skill_v_seasonal": _skill_v_seasonal_at_min,
            "initial_skill_v_linear":   _skill_v_linear_at_min,
            "initial_skill_v_mlr":      _skill_v_mlr_at_min,
        })
    return pd.DataFrame(rows)


def _time_to_baseline_r2_hours(rate_df: pd.DataFrame, include_mlr_baseline: bool = False) -> pd.Series:
    """Return hours needed for R² to reach the best baseline R² at horizon 0."""

    def _row_hours(row: pd.Series) -> float:
        initial_r2 = float(row.get("initial_r2", float("nan")))
        initial_rmse = float(row.get("initial_rmse", float("nan")))
        initial_nrmse = float(row.get("initial_nrmse", float("nan")))
        r2_rate = float(row.get("r2_rate", float("nan")))
        if not (
            np.isfinite(initial_r2)
            and np.isfinite(initial_rmse)
            and np.isfinite(initial_nrmse)
            and np.isfinite(r2_rate)
            and r2_rate != 0
            and initial_rmse > 0
            and initial_nrmse > 0
        ):
            return float("nan")

        skill_candidates = {
            "naive": float(row.get("initial_skill_v_naive", float("nan"))),
            "seasonal": float(row.get("initial_skill_v_seasonal", float("nan"))),
            "linear": float(row.get("initial_skill_v_linear", float("nan"))),
        }
        if include_mlr_baseline:
            skill_candidates["mlr"] = float(row.get("initial_skill_v_mlr", float("nan")))
        finite_candidates = {name: score for name, score in skill_candidates.items() if np.isfinite(score)}
        if not finite_candidates:
            return float("nan")
        best_baseline_label, best_skill = max(finite_candidates.items(), key=lambda item: item[1])
        if best_skill >= 1.0:
            return float("nan")

        baseline_rmse = initial_rmse / (1.0 - best_skill)
        std_target = initial_rmse / initial_nrmse if initial_nrmse > 0 else float("nan")
        if not np.isfinite(std_target) or std_target <= 0:
            return float("nan")

        baseline_r2 = 1.0 - (baseline_rmse / std_target) ** 2
        baseline_r2 = max(0.0, baseline_r2)
        return (baseline_r2 - initial_r2) / r2_rate

    return rate_df.apply(_row_hours, axis=1)


def _baseline_r2_reproducibility_table(records: list[tuple[str, pd.DataFrame]], rate_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-horizon values used to reproduce the baseline-R² timing calculation."""
    rate_lookup = rate_df.set_index("dataset") if not rate_df.empty and "dataset" in rate_df.columns else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for label, df in records:
        if rate_lookup.empty or label not in rate_lookup.index:
            continue
        rate_row = rate_lookup.loc[label]
        grouped_r2 = df.groupby("lookahead", sort=True)["r2"].mean().reset_index(name="mean_r2_used_for_rate")
        for _, row in grouped_r2.iterrows():
            rows.append({
                "dataset": label,
                "lookahead": int(row["lookahead"]),
                "mean_r2_used_for_rate": float(row["mean_r2_used_for_rate"]),
                "initial_r2": float(rate_row.get("initial_r2", float("nan"))),
                "raw_initial_baseline_r2": float(rate_row.get("raw_initial_baseline_r2", float("nan"))),
                "initial_baseline_r2": float(rate_row.get("initial_baseline_r2", float("nan"))),
                "best_baseline_label": str(rate_row.get("best_baseline_label", "")),
                "r2_rate": float(rate_row.get("r2_rate", float("nan"))),
                "time_to_baseline_r2_hours": float(rate_row.get("time_to_baseline_r2_hours", float("nan"))),
                "time_to_baseline_r2_days": float(rate_row.get("time_to_baseline_r2", float("nan"))),
            })
    return pd.DataFrame(rows)


def _discover_datasets(
    data_root: Path,
    prefix: str,
    evaluate_all: bool = False,
) -> list[tuple[str, Path, Path]]:
    """Return ``(dataset_name, dataset_dir, lookahead_metrics_path)`` for every qualifying dataset.

    When ``lookahead_metrics.csv`` is missing (e.g. because the evaluation step
    in ``k_RunHorizonSweep.py`` failed), this function:

    1. Runs any pending evaluations whose eval config exists but whose
       ``evaluation_summary.csv`` is absent.
    2. Reconstructs ``lookahead_metrics.csv`` from the per-horizon
       ``evaluation_summary.csv`` files found on disk.

    When *evaluate_all* is True the reconstruction step uses combined (train+test)
    rows, triggering re-evaluation against saved models where needed.  The cached
    ``lookahead_metrics.csv`` is bypassed so that combined rows are always freshly
    reconstructed from ``evaluation_summary.csv``.
    """
    hits: list[tuple[str, Path, Path]] = []
    if not data_root.exists():
        print(f"[WARN] data_root does not exist: {data_root}")
        return hits
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        metrics_csv = child / "horizons" / "lookahead_sweeps" / "lookahead_metrics.csv"

        # When evaluate_all is requested, always reconstruct from per-rep summaries
        # so combined rows are used regardless of whether a cached CSV exists.
        if not evaluate_all and metrics_csv.exists():
            hits.append((child.name, child, metrics_csv))
            continue

        # --- Fallback / evaluate_all: run pending evaluations and reconstruct metrics ---
        n_completed = _run_pending_evaluations(child)
        if n_completed:
            print(f"[INFO] Completed {n_completed} pending evaluation(s) for {child.name}")
        reconstructed = _reconstruct_lookahead_metrics(child, evaluate_all=evaluate_all)
        if reconstructed is not None and not reconstructed.empty:
            if not evaluate_all:
                # Cache only when not in evaluate_all mode (combined rows are not
                # written by k_RunHorizonSweep.py, so the cached CSV would be stale
                # on the next normal run).
                sweep_dir = child / "horizons" / "lookahead_sweeps"
                sweep_dir.mkdir(parents=True, exist_ok=True)
                reconstructed.to_csv(metrics_csv, index=False)
            mode_note = " [combined]" if evaluate_all else ""
            print(f"[INFO] Reconstructed lookahead_metrics{mode_note} for {child.name} "
                  f"({len(reconstructed)} rows)")
            # Use a temporary in-memory path token for evaluate_all (caller reads from
            # dataset_dir, not from this path, in that mode).
            hits.append((child.name, child, metrics_csv if not evaluate_all
                         else child / "horizons" / "lookahead_sweeps" / "_combined_metrics.csv"))
            if evaluate_all:
                # Write the combined reconstruction to a separate file so it doesn't
                # overwrite the normal lookahead_metrics.csv.
                _combined_path = child / "horizons" / "lookahead_sweeps" / "_combined_metrics.csv"
                _combined_path.parent.mkdir(parents=True, exist_ok=True)
                reconstructed.to_csv(_combined_path, index=False)
        else:
            print(f"[SKIP] No lookahead_metrics.csv for {child.name}")
    return hits


def _bar_fmt(v: float, scientific: bool = False) -> str:
    """Return a compact annotation string for bar values."""
    if scientific:
        s = f"{v:.2e}"
        mantissa, exp = s.split("e")
        return f"{mantissa}e{int(exp)}"
    return f"{v:.3g}"


def _annotate_bars(ax: plt.Axes, fontsize: int = _BAR_ANNOTATION_FONTSIZE) -> None:
    """Annotate each bar patch with its numeric value (rotated 90°), anchored at y=0."""
    heights = [
        rect.get_height()
        for rect in ax.patches
        if np.isfinite(rect.get_height()) and rect.get_height() != 0
    ]
    use_scientific = any(abs(h) < 0.01 for h in heights)
    for rect in ax.patches:
        h = rect.get_height()
        if not np.isfinite(h) or h == 0:
            continue
        x = rect.get_x() + rect.get_width() / 2
        va = "bottom" if h >= 0 else "top"
        ax.text(x, 0, _bar_fmt(h, scientific=use_scientific), ha="center", va=va, fontsize=fontsize,
                rotation=90, clip_on=True)


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
    bar_w = _CLUSTERED_BAR_WIDTH if clustered else _SINGLE_BAR_WIDTH

    fig_w = _bar_fig_width(n, clustered)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    x_main = x - bar_w / 2 if clustered else x
    ax.bar(x_main, df[col], width=bar_w, color=color,
           label=col.replace("_", " "))
    if clustered:
        ax.bar(x + bar_w / 2, df[std_col], width=bar_w,
               color=std_color, label=std_label)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=45, ha="right", fontsize=_BAR_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=_BAR_LABEL_FONTSIZE)
    ax.set_ylabel(textwrap.fill(ylabel, width=20), fontsize=_BAR_LABEL_FONTSIZE)
    _set_bar_xlim(ax, x, bar_w, clustered)
    ax.grid(axis="y", alpha=0.3)
    _annotate_bars(ax)

    fig.tight_layout()
    _expand_ylim_to_fit_annotations(ax)
    if clustered:
        legend_above(ax, fontsize=_BAR_LEGEND_FONTSIZE)
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
    bar_w = _CLUSTERED_BAR_WIDTH

    fig_w = _bar_fig_width(n, clustered=True)
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
    ax.set_xticklabels(df["dataset"], rotation=45, ha="right", fontsize=_BAR_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=_BAR_LABEL_FONTSIZE)
    ax.set_ylabel(textwrap.fill("nRMSE avg. rate of change (/hr)", width=20), fontsize=_BAR_LABEL_FONTSIZE)
    _set_bar_xlim(ax, x, bar_w, clustered=True)
    ax.grid(axis="y", alpha=0.3)
    _annotate_bars(ax)

    fig.tight_layout()
    _expand_ylim_to_fit_annotations(ax)
    legend_above(ax, fontsize=_BAR_LEGEND_FONTSIZE)
    out = summaries_dir / "lookahead_rates_bar.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def _build_records(
    datasets: list[tuple[str, Path, Path]],
    prefix: str,
    model_class_filter: "str | None",
    show_std: bool,
    ml_selection: str = "best",
    treat_mlr_as_baseline: bool = False,
    evaluate_all: bool = False,
) -> list[tuple[str, pd.DataFrame]]:
    """Load and enrich metrics for each dataset, filtered by model_class.

    *model_class_filter*: 'ml', 'mlr', or None (accept all rows).
    *evaluate_all*: when True, ``combined`` (train+test) rows are preferred over
        ``test``-only rows when computing statistics.  When reading from
        ``_combined_metrics.csv`` (produced by ``_discover_datasets`` in evaluate_all
        mode) the ``kind`` column has already been dropped by
        ``_reconstruct_lookahead_metrics``, so the kind filter below is a no-op in
        that path; it remains active when the CSV retains a ``kind`` column (e.g.
        future formats or direct reads of ``evaluation_summary.csv``).

        ``nrmse`` uses ``std_target`` from the combined row when the column is
        present (written by ``f_Evaluate.py`` as of the current version), falling
        back to ``sqrt(1 - r2)`` for older CSVs.  Skill scores use
        ``combined_baseline_summary.csv`` (written by ``_ensure_combined_metrics``)
        so baselines are also evaluated on the combined sample set.

    Returns a list of (clean_label, enriched_df) pairs with nrmse and skill columns added.
    Rows where model_class column exists and does not match *model_class_filter* are dropped.
    """
    records: list[tuple[str, pd.DataFrame]] = []
    for name, dataset_dir, csv_path in datasets:
        try:
            df = pd.read_csv(csv_path)
            if "horizon" in df.columns and "lookahead" not in df.columns:
                df = df.rename(columns={"horizon": "lookahead"})
            df = df.sort_values("lookahead").reset_index(drop=True)

            # Filter by model_class if requested and column is present.
            if model_class_filter is not None and "model_class" in df.columns:
                df = df[df["model_class"] == model_class_filter].copy()
            if df.empty:
                continue

            if ml_selection == "xgb" and "model_name" in df.columns:
                if model_class_filter == "ml":
                    df = df[df["model_name"].apply(_is_xgb_model_name)].copy()
                elif model_class_filter is None:
                    keep_mask = df["model_name"].apply(_is_xgb_model_name)
                    if "model_class" in df.columns:
                        keep_mask = keep_mask | df["model_class"].astype(str).str.lower().eq("mlr")
                    else:
                        keep_mask = keep_mask | df["model_name"].apply(_is_mlr_model_name)
                    df = df[keep_mask].copy()
            if df.empty:
                continue

            # Filter rows by kind: prefer combined (train+test) when evaluate_all is
            # set; fall back to test if no combined rows are present.
            if evaluate_all and "kind" in df.columns:
                combined = df[df["kind"] == "combined"]
                df = combined if not combined.empty else df[df["kind"] == "test"]
            elif "kind" in df.columns:
                df = df[df["kind"] == "test"]
            if df.empty:
                continue

            std_target = _load_std_target(dataset_dir)
            if evaluate_all and "std_target" in df.columns:
                # Use std_target written directly into evaluation_summary.csv by
                # f_Evaluate.py — the correct denominator for the combined sample set.
                # Falls back to the algebraic derivation for older CSVs that predate
                # the std_target column.
                df["nrmse"] = df.apply(
                    lambda row: float(row["rmse"]) / float(row["std_target"])
                    if (np.isfinite(float(row.get("std_target", float("nan"))))
                        and float(row["std_target"]) > 0)
                    else float("nan"),
                    axis=1,
                )
            elif evaluate_all:
                # Older CSVs without std_target: derive algebraically from r2.
                # nrmse = rmse / std  and  r2 = 1 - rmse²/std²  →  nrmse = sqrt(1 - r2)
                df["nrmse"] = df["r2"].apply(
                    lambda r2: np.sqrt(max(0.0, 1.0 - r2)) if np.isfinite(r2) else float("nan")
                )
            else:
                df["nrmse"] = df["rmse"] / std_target if std_target is not None else float("nan")

            # Skill vs. baselines — keyed by (horizon, replicate, model_class).
            rep_col = "replicate" if "replicate" in df.columns else None
            mc_col = "model_class" if "model_class" in df.columns else None
            _baseline_cache: dict[tuple, dict[str, float]] = {}
            for _, _row in df.iterrows():
                _h = int(_row["lookahead"])
                _r = int(_row[rep_col]) if rep_col else 0
                _mc = str(_row[mc_col]) if mc_col else None
                _key = (_h, _r, _mc)
                if _key not in _baseline_cache:
                    _baseline_cache[_key] = _load_baseline_rmses(dataset_dir, _h, _r, _mc,
                                                                  include_mlr_baseline=treat_mlr_as_baseline,
                                                                  evaluate_all=evaluate_all)

            def _skill(model_rmse: float, baseline_rmse: float) -> float:
                if np.isfinite(model_rmse) and np.isfinite(baseline_rmse) and baseline_rmse > 0:
                    return 1.0 - model_rmse / baseline_rmse
                return float("nan")

            def _get_skills(row: pd.Series) -> pd.Series:
                h = int(row["lookahead"])
                r = int(row[rep_col]) if rep_col else 0
                mc = str(row[mc_col]) if mc_col else None
                bl = _baseline_cache.get((h, r, mc), {})
                m = float(row["rmse"])
                return pd.Series({
                    "skill_v_naive":    _skill(m, bl.get("naive",    float("nan"))),
                    "skill_v_seasonal": _skill(m, bl.get("seasonal", float("nan"))),
                    "skill_v_linear":   _skill(m, bl.get("linear",   float("nan"))),
                    "skill_v_mlr":      _skill(m, bl.get("mlr",      float("nan"))),
                })

            _skills = df.apply(_get_skills, axis=1)
            df["skill_v_naive"]    = _skills["skill_v_naive"]
            df["skill_v_seasonal"] = _skills["skill_v_seasonal"]
            df["skill_v_linear"]   = _skills["skill_v_linear"]
            df["skill_v_mlr"]      = _skills["skill_v_mlr"]
            baseline_skill_cols = ["skill_v_naive", "skill_v_seasonal", "skill_v_linear"]
            if treat_mlr_as_baseline:
                baseline_skill_cols.append("skill_v_mlr")
            df["skill_v_best_baseline"] = _skills[baseline_skill_cols].max(axis=1)

            label = _clean_label(name, prefix)
            records.append((label, df))
            n_horizons = df["lookahead"].nunique()
            n_reps = df["replicate"].nunique() if "replicate" in df.columns else 1
            std_note = f"std_target={std_target:.4g}" if std_target is not None else "std_target not found"
            mc_note = f" [{model_class_filter}]" if model_class_filter else ""
            print(f"[INFO]  {name}{mc_note}: {n_horizons} horizons × {n_reps} replicate(s), {std_note}")
        except Exception:
            print(f"[WARN] Could not load {csv_path}:")
            traceback.print_exc()
    return records


def _pick_best_records(
    records_ml: list[tuple[str, pd.DataFrame]],
    records_mlr: list[tuple[str, pd.DataFrame]],
) -> list[tuple[str, pd.DataFrame]]:
    """Return one record per label: whichever model_class has higher initial_skill at min horizon.

    For labels that appear in only one class, that class's record is used.
    Falls back to lower mean RMSE when skill is NaN for both.
    """
    ml_map = {lbl: df for lbl, df in records_ml}
    mlr_map = {lbl: df for lbl, df in records_mlr}
    all_labels = list(dict.fromkeys([lbl for lbl, _ in records_ml] + [lbl for lbl, _ in records_mlr]))
    best: list[tuple[str, pd.DataFrame]] = []
    for lbl in all_labels:
        ml_df = ml_map.get(lbl)
        mlr_df = mlr_map.get(lbl)
        if ml_df is None:
            best.append((lbl, mlr_df))
            continue
        if mlr_df is None:
            best.append((lbl, ml_df))
            continue
        # Compare initial skill at minimum horizon
        def _initial_skill(df: pd.DataFrame) -> float:
            if "skill_v_best_baseline" not in df.columns:
                return float("nan")
            min_h = df["lookahead"].min()
            vals = df.loc[df["lookahead"] == min_h, "skill_v_best_baseline"].dropna()
            return float(vals.mean()) if not vals.empty else float("nan")

        s_ml = _initial_skill(ml_df)
        s_mlr = _initial_skill(mlr_df)
        if np.isfinite(s_ml) and np.isfinite(s_mlr):
            best.append((lbl, ml_df if s_ml >= s_mlr else mlr_df))
        elif np.isfinite(s_ml):
            best.append((lbl, ml_df))
        elif np.isfinite(s_mlr):
            best.append((lbl, mlr_df))
        else:
            # Fall back to lower mean RMSE
            rmse_ml = float(ml_df["rmse"].mean()) if "rmse" in ml_df.columns else float("inf")
            rmse_mlr = float(mlr_df["rmse"].mean()) if "rmse" in mlr_df.columns else float("inf")
            best.append((lbl, ml_df if rmse_ml <= rmse_mlr else mlr_df))
    return best


def _write_aggregate_csv(records: list[tuple[str, pd.DataFrame]], out_path: Path) -> None:
    """Write lookahead_aggregate.csv for the given records list."""
    _drop_suffix = "_replicate"
    _front_cols = [
        "dataset", "dataset_dir", "model_class", "model_name", "lookahead", "replicate",
        "mae", "rmse", "nrmse", "r2", "pearson_r",
        "skill_v_naive", "skill_v_seasonal", "skill_v_linear", "skill_v_best_baseline",
    ]
    agg_frames = []
    for lbl, df in records:
        frame = df.copy()
        frame = frame.drop(columns=[c for c in frame.columns if c.endswith(_drop_suffix)],
                           errors="ignore")
        # k_RunHorizonSweep writes its own "dataset" column holding the directory name,
        # while everything else in this script keys on the display label. They are
        # different strings for the same target, so the raw one is kept under a distinct
        # name rather than being overwritten or inserted alongside a duplicate.
        if "dataset" in frame.columns:
            frame = frame.rename(columns={"dataset": "dataset_dir"})
        frame.insert(0, "dataset", lbl)
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
    _rest_cols = [c for c in agg_df.columns if c not in _front_cols]
    agg_df = agg_df[[c for c in _front_cols if c in agg_df.columns] + _rest_cols]
    agg_df.to_csv(out_path, index=False)


def _generate_all_figures(
    records: list[tuple[str, pd.DataFrame]],
    out_dir: Path,
    show_std: bool = True,
    all_x: "list | None" = None,
    tag: str = "",
    treat_mlr_as_baseline: bool = False,
) -> None:
    """Generate the standard 8-figure + 2-CSV set for a single records list.

    *all_x*: shared x-axis tick values; computed from *records* if not provided.
    *tag*: short string for log messages (e.g. 'best', 'ml', 'mlr').
    """
    if not records:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = f"[{tag}] " if tag else ""

    if all_x is None:
        all_x = sorted({v for _, df in records for v in df["lookahead"].dropna().tolist()})

    any_replicates = show_std and any(
        "replicate" in df.columns and df["replicate"].nunique() > 1
        for _, df in records
    )

    _FIG_WIDTH  = 13.0
    _ROW_HEIGHT = 0.88
    _MIN_FIG_H  = 2.8
    _HSPACE     = 0.08
    _TOP_IN     = 0.45
    _BOTTOM_IN  = 0.90

    def _make_figure(
        metric: str,
        ylabel: str,
        filename: str,
        hline_zero: bool = False,
        any_replicates: bool = False,
        ylim: "tuple | None" = None,
        yticks: "list | None" = None,
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
                ax.plot(x, mu, marker="o", markersize=4, linewidth=1.5, color=color, zorder=3)
                ax.plot(x, mu + sigma, linestyle="--", linewidth=0.8, alpha=0.7, color=color, zorder=2)
                ax.plot(x, mu - sigma, linestyle="--", linewidth=0.8, alpha=0.7, color=color, zorder=2)
                ax.plot(x, mu + 2 * sigma, linestyle=":", linewidth=0.6, alpha=0.55, color=color, zorder=1)
                ax.plot(x, mu - 2 * sigma, linestyle=":", linewidth=0.6, alpha=0.55, color=color, zorder=1)
            else:
                plot_df = (
                    df.groupby("lookahead")[metric].mean().reset_index()
                    if "replicate" in df.columns else df
                )
                ax.plot(plot_df["lookahead"], plot_df[metric],
                        marker="o", markersize=4, linewidth=1.5, color=color)

            wrapped = "\n".join(textwrap.wrap(label, width=15))
            ax.set_ylabel(wrapped, rotation=0, ha="right", va="center", fontsize=15, labelpad=8)
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

        axes[-1].set_xticks(all_x)
        axes[-1].set_xticklabels([])
        trans = axes[-1].get_xaxis_transform()
        for j, val in enumerate(all_x):
            y_offset = -0.1 if j % 2 == 0 else -0.12
            axes[-1].text(val, y_offset, str(int(val)), transform=trans,
                          ha="center", va="top", fontsize=10)

        top_frac    = max(0.86, min(0.995, 1.0 - (_TOP_IN    / fig_h)))
        bottom_frac = max(0.08, min(0.30,          _BOTTOM_IN / fig_h))
        fig.subplots_adjust(left=0.18, right=0.995, top=top_frac, bottom=bottom_frac, hspace=_HSPACE)
        _axes_h_frac = (top_frac - bottom_frac) / (n_rows + max(n_rows - 1, 0) * _HSPACE)
        _axes_h_pts  = _axes_h_frac * fig_h * 72
        _xlabel_pad  = max(18, int(0.12 * _axes_h_pts + 15))
        axes[-1].set_xlabel("Forecast horizon (hours)", fontsize=15, labelpad=_xlabel_pad)

        # Each row's y-axis label names its target, so nothing on the axes says which
        # metric is plotted; "Mean" alone did not. Naming the metric in the legend is the
        # one place it fits without colliding with the per-row labels.
        legend_handles = [
            plt.Line2D([0], [0], color="black", linewidth=1.5,
                       marker="o", markersize=4, label=f"Mean {ylabel}"),
        ]
        if any_replicates:
            legend_handles += [
                plt.Line2D([0], [0], color="black", linestyle="--", linewidth=0.8, alpha=0.7, label="±1σ"),
                plt.Line2D([0], [0], color="black", linestyle=":", linewidth=0.6, alpha=0.55, label="±2σ"),
            ]
        fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   bbox_transform=axes[0].transAxes,
                   ncol=len(legend_handles), framealpha=0.85, fontsize=15, borderaxespad=0.12)

        out = out_dir / filename
        fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return out

    r2_path = _make_figure("r2", "R²", "lookahead_r2_comparison.png",
                           hline_zero=True, any_replicates=any_replicates,
                           ylim=(-1.2, 1.2), yticks=[-1, 0, 1])
    print(f"[INFO] {pfx}Wrote R²:    {r2_path}")

    _nrmse_all = pd.concat([df["nrmse"] for _, df in records if "nrmse" in df.columns],
                           ignore_index=True).dropna()
    if not _nrmse_all.empty:
        _nrmse_min, _nrmse_max = float(_nrmse_all.min()), float(_nrmse_all.max())
        _margin = 0.1 * (_nrmse_max - _nrmse_min)
        nrmse_ylim: "tuple | None" = (_nrmse_min - _margin, _nrmse_max + _margin)
    else:
        nrmse_ylim = None
    nrmse_path = _make_figure("nrmse", "nRMSE (RMSE / σ_target)", "lookahead_nrmse_comparison.png",
                              any_replicates=any_replicates, ylim=nrmse_ylim)
    print(f"[INFO] {pfx}Wrote nRMSE: {nrmse_path}")

    _skill_all = pd.concat(
        [df["skill_v_best_baseline"] for _, df in records if "skill_v_best_baseline" in df.columns],
        ignore_index=True).dropna()
    if not _skill_all.empty:
        _s_min, _s_max = float(_skill_all.min()), float(_skill_all.max())
        _s_margin = max(0.05, 0.1 * (_s_max - _s_min))
        skill_ylim: "tuple | None" = (max(-2.0, _s_min - _s_margin), min(1.05, _s_max + _s_margin))
    else:
        skill_ylim = None
    skill_path = _make_figure("skill_v_best_baseline", "Skill vs. Best Baseline",
                              "lookahead_skill_comparison.png", hline_zero=True,
                              any_replicates=any_replicates, ylim=skill_ylim)
    print(f"[INFO] {pfx}Wrote skill: {skill_path}")

    agg_path = out_dir / "lookahead_aggregate.csv"
    _write_aggregate_csv(records, agg_path)
    print(f"[INFO] {pfx}Wrote aggregate CSV: {agg_path}")

    rate_df = _rate_table(records, include_mlr_baseline=treat_mlr_as_baseline)
    rate_path = out_dir / "lookahead_rates.csv"
    rate_df.to_csv(rate_path, index=False)
    print(f"[INFO] {pfx}Wrote rates CSV: {rate_path}")

    rates_path = _plot_rates(rate_df, out_dir, show_std=show_std)
    print(f"[INFO] {pfx}Wrote rates bar: {rates_path}")

    skill_bar_path = _plot_rate_bar(
        rate_df, "skill_rate", "Skill avg. rate of change (/hr)",
        "lookahead_skill_rate_bar.png", out_dir,
        ascending=True, color=_SERIES_COLORS[1],
        std_col="std_skill_rate" if show_std else None,
        std_label="σ(skill) rate (/hr)",
    )
    print(f"[INFO] {pfx}Wrote skill rate bar: {skill_bar_path}")

    r2_bar_path = _plot_rate_bar(
        rate_df, "r2_rate", "$R^2$ avg. rate of change (/hr)",
        "lookahead_r2_rate_bar.png", out_dir,
        ascending=True, color=_SERIES_COLORS[2],
        std_col="std_r2_rate" if show_std else None,
        std_label="σ(R²) rate (/hr)",
    )
    print(f"[INFO] {pfx}Wrote R² rate bar: {r2_bar_path}")

    rate_df["time_to_zero_skill_hours"] = rate_df["initial_skill"] / (-rate_df["skill_rate"])
    rate_df["time_to_zero_skill"] = rate_df["time_to_zero_skill_hours"] / 24.0
    tzs_path = _plot_rate_bar(
        rate_df, "time_to_zero_skill", "Forecast Horizon (days)",
        "lookahead_time_to_zero_skill.png", out_dir,
        ascending=False, color=_SERIES_COLORS[1],
    )
    print(f"[INFO] {pfx}Wrote time-to-zero-skill: {tzs_path}")

    rate_df["time_to_baseline_r2_hours"] = _time_to_baseline_r2_hours(
        rate_df,
        include_mlr_baseline=treat_mlr_as_baseline,
    )
    rate_df["time_to_baseline_r2"] = rate_df["time_to_baseline_r2_hours"] / 24.0
    ttbr2_path = _plot_rate_bar(
        rate_df, "time_to_baseline_r2", "Forecast Horizon (days)",
        "lookahead_time_to_baseline_r2.png", out_dir,
        ascending=False, color=_SERIES_COLORS[1],
    )
    print(f"[INFO] {pfx}Wrote time-to-baseline-R²: {ttbr2_path}")

    repro_df = _baseline_r2_reproducibility_table(records, rate_df)
    repro_path = out_dir / "lookahead_time_to_baseline_r2_repro.csv"
    repro_df.to_csv(repro_path, index=False)
    print(f"[INFO] {pfx}Wrote baseline-R² reproducibility CSV: {repro_path}")


def _generate_combined_figures(
    records_ml: list[tuple[str, pd.DataFrame]],
    records_mlr: list[tuple[str, pd.DataFrame]],
    out_dir: Path,
    show_std: bool = True,
    all_x: "list | None" = None,
    treat_mlr_as_baseline: bool = False,
) -> None:
    """Generate combined figures with ML (solid) and MLR (dashed) series on same axes.

    One subplot per target label. If a target has no MLR data, that subplot shows
    only the ML series (and vice versa).  Labels that have neither are hidden.
    Combined aggregate CSV and rates CSV include model_class column.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ml_map  = {lbl: df for lbl, df in records_ml}
    mlr_map = {lbl: df for lbl, df in records_mlr}
    all_labels = list(dict.fromkeys(
        [lbl for lbl, _ in records_ml] + [lbl for lbl, _ in records_mlr]
    ))
    # Only include labels that have at least one of ML or MLR.
    active_labels = [lbl for lbl in all_labels if ml_map.get(lbl) is not None or mlr_map.get(lbl) is not None]
    if not active_labels:
        return

    if all_x is None:
        all_series = list(records_ml) + list(records_mlr)
        all_x = sorted({v for _, df in all_series for v in df["lookahead"].dropna().tolist()})

    _FIG_WIDTH  = 13.0
    _ROW_HEIGHT = 0.88
    _MIN_FIG_H  = 2.8
    _HSPACE     = 0.08
    _TOP_IN     = 0.55   # slightly more room for legend (ML/MLR entries)
    _BOTTOM_IN  = 0.90

    _ML_LS  = "-"    # solid for ML
    _MLR_LS = "--"   # dashed for MLR

    def _make_combined_figure(
        metric: str,
        ylabel: str,
        filename: str,
        hline_zero: bool = False,
        ylim: "tuple | None" = None,
        yticks: "list | None" = None,
    ) -> Path:
        n_rows = len(active_labels)
        fig_h = max(_MIN_FIG_H, _ROW_HEIGHT * n_rows)
        fig, axes = plt.subplots(
            n_rows, 1, sharex=True,
            figsize=(_FIG_WIDTH, fig_h),
            gridspec_kw={"hspace": _HSPACE},
        )
        if n_rows == 1:
            axes = [axes]

        for i, label in enumerate(active_labels):
            ax = axes[i]
            color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
            has_any = False

            for df_map, ls, class_name in [(ml_map, _ML_LS, "ML"), (mlr_map, _MLR_LS, "MLR")]:
                df = df_map.get(label)
                if df is None or metric not in df.columns or df[metric].isnull().all():
                    continue
                has_any = True
                plot_df = df.groupby("lookahead")[metric].mean().reset_index()
                ax.plot(plot_df["lookahead"], plot_df[metric],
                        marker="o" if ls == _ML_LS else None,
                        markersize=4, linewidth=1.5,
                        color=color, linestyle=ls, zorder=3)

            if not has_any:
                ax.set_visible(False)
                continue

            wrapped = "\n".join(textwrap.wrap(label, width=15))
            ax.set_ylabel(wrapped, rotation=0, ha="right", va="center", fontsize=15, labelpad=8)
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

        axes[-1].set_xticks(all_x)
        axes[-1].set_xticklabels([])
        trans = axes[-1].get_xaxis_transform()
        for j, val in enumerate(all_x):
            y_offset = -0.1 if j % 2 == 0 else -0.12
            axes[-1].text(val, y_offset, str(int(val)), transform=trans,
                          ha="center", va="top", fontsize=10)

        top_frac    = max(0.86, min(0.995, 1.0 - (_TOP_IN    / fig_h)))
        bottom_frac = max(0.08, min(0.30,          _BOTTOM_IN / fig_h))
        fig.subplots_adjust(left=0.18, right=0.995, top=top_frac, bottom=bottom_frac, hspace=_HSPACE)
        _axes_h_frac = (top_frac - bottom_frac) / (n_rows + max(n_rows - 1, 0) * _HSPACE)
        _axes_h_pts  = _axes_h_frac * fig_h * 72
        _xlabel_pad  = max(18, int(0.12 * _axes_h_pts + 15))
        axes[-1].set_xlabel("Forecast horizon (hours)", fontsize=15, labelpad=_xlabel_pad)

        legend_handles = [
            plt.Line2D([0], [0], color="black", linewidth=1.5, linestyle=_ML_LS,
                       marker="o", markersize=4, label="Machine Learning"),
            plt.Line2D([0], [0], color="black", linewidth=1.5, linestyle=_MLR_LS,
                       label="Multiple Linear Regression"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   bbox_transform=axes[0].transAxes,
                   ncol=2, framealpha=0.85, fontsize=15, borderaxespad=0.12)

        out = out_dir / filename
        fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return out

    r2_path = _make_combined_figure("r2", "R²", "lookahead_r2_comparison.png",
                                    hline_zero=True, ylim=(-1.2, 1.2), yticks=[-1, 0, 1])
    print(f"[INFO] [combined] Wrote R²: {r2_path}")

    all_nrmse = pd.concat(
        [df["nrmse"] for _, df in list(records_ml) + list(records_mlr) if "nrmse" in df.columns],
        ignore_index=True).dropna()
    if not all_nrmse.empty:
        _n_min, _n_max = float(all_nrmse.min()), float(all_nrmse.max())
        _n_margin = 0.1 * (_n_max - _n_min)
        nrmse_ylim: "tuple | None" = (_n_min - _n_margin, _n_max + _n_margin)
    else:
        nrmse_ylim = None
    nrmse_path = _make_combined_figure("nrmse", "nRMSE", "lookahead_nrmse_comparison.png",
                                       ylim=nrmse_ylim)
    print(f"[INFO] [combined] Wrote nRMSE: {nrmse_path}")

    all_skill = pd.concat(
        [df["skill_v_best_baseline"] for _, df in list(records_ml) + list(records_mlr)
         if "skill_v_best_baseline" in df.columns],
        ignore_index=True).dropna()
    if not all_skill.empty:
        _s_min, _s_max = float(all_skill.min()), float(all_skill.max())
        _s_margin = max(0.05, 0.1 * (_s_max - _s_min))
        skill_ylim: "tuple | None" = (max(-2.0, _s_min - _s_margin), min(1.05, _s_max + _s_margin))
    else:
        skill_ylim = None
    skill_path = _make_combined_figure("skill_v_best_baseline", "Skill", "lookahead_skill_comparison.png",
                                       hline_zero=True, ylim=skill_ylim)
    print(f"[INFO] [combined] Wrote skill: {skill_path}")

    # Combined aggregate CSV
    combined_records = [(lbl, df) for lbl, df in records_ml] + [(lbl, df) for lbl, df in records_mlr]
    agg_path = out_dir / "lookahead_aggregate.csv"
    _write_aggregate_csv(combined_records, agg_path)
    print(f"[INFO] [combined] Wrote aggregate CSV: {agg_path}")

    # Combined rates CSV (two rows per dataset: ML and MLR)
    rate_df_ml  = _rate_table(records_ml, include_mlr_baseline=treat_mlr_as_baseline)
    rate_df_mlr = _rate_table(records_mlr, include_mlr_baseline=treat_mlr_as_baseline)
    if not rate_df_ml.empty:
        rate_df_ml["model_class"] = "ml"
    if not rate_df_mlr.empty:
        rate_df_mlr["model_class"] = "mlr"
    combined_rate_df = pd.concat([rate_df_ml, rate_df_mlr], ignore_index=True)
    rate_path = out_dir / "lookahead_rates.csv"
    combined_rate_df.to_csv(rate_path, index=False)
    print(f"[INFO] [combined] Wrote rates CSV: {rate_path}")

    # Bar charts — one clustered pair of bars per dataset (ML blue, MLR orange)
    # Use rate_df_ml and rate_df_mlr aligned by dataset label.
    def _plot_combined_rate_bar(col: str, ylabel: str, filename: str, ascending: bool = False) -> None:
        all_datasets = list(dict.fromkeys(
            list(rate_df_ml["dataset"]) + list(rate_df_mlr["dataset"])
        ))
        ml_vals  = rate_df_ml.set_index("dataset")[col] if not rate_df_ml.empty else pd.Series(dtype=float)
        mlr_vals = rate_df_mlr.set_index("dataset")[col] if not rate_df_mlr.empty else pd.Series(dtype=float)
        # Order by ML value (NaN last)
        order = sorted(all_datasets, key=lambda d: (
            not np.isfinite(ml_vals.get(d, float("nan"))), ml_vals.get(d, float("nan"))
        ), reverse=not ascending)
        n = len(order)
        x = np.arange(n)
        bar_w = _CLUSTERED_BAR_WIDTH
        fig_w = _bar_fig_width(n, clustered=True)
        fig, ax = plt.subplots(figsize=(fig_w, 4.5))
        ax.bar(x - bar_w / 2, [ml_vals.get(d, float("nan")) for d in order],
               width=bar_w, color=_SERIES_COLORS[0], label="Machine Learning")
        ax.bar(x + bar_w / 2, [mlr_vals.get(d, float("nan")) for d in order],
               width=bar_w, color=_SERIES_COLORS[5], label="Multiple Linear Regression")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=_BAR_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=_BAR_LABEL_FONTSIZE)
        ax.set_ylabel(textwrap.fill(ylabel, width=20), fontsize=_BAR_LABEL_FONTSIZE)
        _set_bar_xlim(ax, x, bar_w, clustered=True)
        ax.grid(axis="y", alpha=0.3)
        _annotate_bars(ax)
        fig.tight_layout()
        _expand_ylim_to_fit_annotations(ax)
        legend_above(ax, fontsize=_BAR_LEGEND_FONTSIZE)
        out = out_dir / filename
        fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"[INFO] [combined] Wrote {filename}: {out}")

    _plot_combined_rate_bar("nrmse_rate", "nRMSE avg. rate of change (/hr)",
                            "lookahead_rates_bar.png", ascending=False)
    _plot_combined_rate_bar("skill_rate", "Skill avg. rate of change (/hr)",
                            "lookahead_skill_rate_bar.png", ascending=True)
    _plot_combined_rate_bar("r2_rate", "$R^2$ avg. rate of change (/hr)",
                            "lookahead_r2_rate_bar.png", ascending=True)

    # Time-to-zero-skill combined
    for _rdf in [rate_df_ml, rate_df_mlr]:
        _rdf["time_to_zero_skill_hours"] = _rdf["initial_skill"] / (-_rdf["skill_rate"])
        _rdf["time_to_zero_skill"] = _rdf["time_to_zero_skill_hours"] / 24.0
    _plot_combined_rate_bar("time_to_zero_skill", "Forecast Horizon (days)",
                            "lookahead_time_to_zero_skill.png", ascending=False)

    # Time-to-baseline-R² combined
    for _rdf in [rate_df_ml, rate_df_mlr]:
        _rdf["time_to_baseline_r2_hours"] = _time_to_baseline_r2_hours(
            _rdf,
            include_mlr_baseline=treat_mlr_as_baseline,
        )
        _rdf["time_to_baseline_r2"] = _rdf["time_to_baseline_r2_hours"] / 24.0
    _plot_combined_rate_bar("time_to_baseline_r2", "Forecast Horizon (days)",
                            "lookahead_time_to_baseline_r2.png", ascending=False)

    combined_repro_df = pd.concat([
        _baseline_r2_reproducibility_table(records_ml, rate_df_ml).assign(model_class="ml"),
        _baseline_r2_reproducibility_table(records_mlr, rate_df_mlr).assign(model_class="mlr"),
    ], ignore_index=True)
    combined_repro_path = out_dir / "lookahead_time_to_baseline_r2_repro.csv"
    combined_repro_df.to_csv(combined_repro_path, index=False)
    print(f"[INFO] [combined] Wrote baseline-R² reproducibility CSV: {combined_repro_path}")


def generate_figures(
    data_root: Path,
    prefix: str,
    summaries_dir: Path,
    show_std: bool = True,
    ml_selection: str = "best",
    treat_mlr_as_baseline: bool = False,
    evaluate_all: bool = False,
) -> int:
    datasets = _discover_datasets(data_root, prefix, evaluate_all=evaluate_all)
    if not datasets:
        print(f"[WARN] No datasets with lookahead metrics found under {data_root}.")
        return 1

    print(f"[INFO] Found {len(datasets)} dataset(s) with lookahead metrics.")
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Shared x-axis tick values across all classes and datasets
    all_x_set: set[float] = set()
    for _, _, csv_path in datasets:
        try:
            _df = pd.read_csv(csv_path)
            col = "horizon" if "horizon" in _df.columns else "lookahead"
            if col in _df.columns:
                all_x_set.update(_df[col].dropna().tolist())
        except Exception:
            pass
    all_x = sorted(all_x_set)

    print("[INFO] Building ML records...")
    records_ml   = _build_records(
        datasets,
        prefix,
        model_class_filter="ml",
        show_std=show_std,
        ml_selection=ml_selection,
        treat_mlr_as_baseline=treat_mlr_as_baseline,
        evaluate_all=evaluate_all,
    )
    print("[INFO] Building MLR records...")
    records_mlr  = _build_records(
        datasets,
        prefix,
        model_class_filter="mlr",
        show_std=show_std,
        ml_selection=ml_selection,
        treat_mlr_as_baseline=treat_mlr_as_baseline,
        evaluate_all=evaluate_all,
    )
    print("[INFO] Building combined (all) records for 'best' selection...")
    records_all  = _build_records(
        datasets,
        prefix,
        model_class_filter=None,
        show_std=show_std,
        ml_selection=ml_selection,
        treat_mlr_as_baseline=treat_mlr_as_baseline,
        evaluate_all=evaluate_all,
    )

    # 'best' = one record per label, whichever model_class has better initial_skill
    if records_ml or records_mlr:
        records_best = _pick_best_records(records_ml, records_mlr)
    else:
        records_best = records_all

    print(f"\n[INFO] Generating best/ figures ({len(records_best)} dataset(s))...")
    _generate_all_figures(
        records_best,
        summaries_dir / "best",
        show_std=show_std,
        all_x=all_x,
        tag="best",
        treat_mlr_as_baseline=treat_mlr_as_baseline,
    )

    print(f"\n[INFO] Generating ml/ figures ({len(records_ml)} dataset(s))...")
    _generate_all_figures(
        records_ml,
        summaries_dir / "ml",
        show_std=show_std,
        all_x=all_x,
        tag="ml",
        treat_mlr_as_baseline=treat_mlr_as_baseline,
    )

    print(f"\n[INFO] Generating mlr/ figures ({len(records_mlr)} dataset(s))...")
    _generate_all_figures(
        records_mlr,
        summaries_dir / "mlr",
        show_std=show_std,
        all_x=all_x,
        tag="mlr",
        treat_mlr_as_baseline=treat_mlr_as_baseline,
    )

    if records_ml and records_mlr:
        print(f"\n[INFO] Generating combined/ figures...")
        _generate_combined_figures(
            records_ml,
            records_mlr,
            summaries_dir / "combined",
            show_std=show_std,
            all_x=all_x,
            treat_mlr_as_baseline=treat_mlr_as_baseline,
        )
    else:
        print("[INFO] Skipping combined/ figures — fewer than 2 model classes have data.")

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
        "--ml-selection",
        choices=["best", "xgb"],
        default="best",
        help=(
            'How to choose ML-family horizon results: "best" uses the best of '
            'XGB/GP/Transformer, "xgb" restricts ML-family results to XGB only.'
        ),
    )
    parser.add_argument(
        "--treat-mlr-as-baseline",
        action="store_true",
        help=(
            "Include MLR as an additional baseline candidate in best-baseline "
            "skill summaries."
        ),
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
    parser.add_argument(
        "--evaluate-all",
        action="store_true",
        help=(
            "Deprecated compatibility flag.  The script now always generates both "
            "eval_test (test set only) and eval_all (combined train+test) outputs "
            "under summaries/horizons/."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()
    summaries_root = (data_root / "summaries" / "horizons").resolve()
    print(f"[INFO] data_root : {data_root}")
    print(f"[INFO] summaries : {summaries_root}")
    if args.evaluate_all:
        print("[INFO] --evaluate-all is deprecated; generating both eval_test and eval_all outputs.")

    # Always generate both evaluation scopes under separate directory levels.
    eval_test_dir = summaries_root / "eval_test"
    eval_all_dir = summaries_root / "eval_all"

    print("[INFO] Generating eval_test outputs (test set only)...")
    rc_test = generate_figures(
        data_root=data_root,
        prefix=args.dataset_prefix,
        summaries_dir=eval_test_dir,
        show_std=args.std,
        ml_selection=args.ml_selection,
        treat_mlr_as_baseline=args.treat_mlr_as_baseline,
        evaluate_all=False,
    )

    print("[INFO] Generating eval_all outputs (train + test combined)...")
    rc_all = generate_figures(
        data_root=data_root,
        prefix=args.dataset_prefix,
        summaries_dir=eval_all_dir,
        show_std=args.std,
        ml_selection=args.ml_selection,
        treat_mlr_as_baseline=args.treat_mlr_as_baseline,
        evaluate_all=True,
    )

    return 0 if (rc_test == 0 and rc_all == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

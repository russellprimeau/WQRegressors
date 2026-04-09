"""Repair baseline_summary.csv files for all datasets and horizons under data_root.

Reads existing eval configs and sample splits already on disk — no model training
or evaluation is performed.  Only writes baseline_summary.csv files that are
missing or contain no valid RMSE values.

For GP/XGB/Transformer datasets: uses the per-horizon rep_000 eval config
(config_evaluate_rep_000.yml), which carries historic_path.

For MLR datasets: no per-horizon eval config exists, so falls back to any
feature-sweep eval config under the dataset directory that carries historic_path.
The split is derived from the rep_000 test_files.txt that was written by
k_RunHorizonSweep._run_mlr_horizon_rep.

Usage:
    python src/z2_repair_baselines.py --data-root data/output/CV14 --dataset-prefix MC
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _is_valid(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
        return "rmse" in df.columns and df["rmse"].notna().any()
    except Exception:
        return False


def _find_eval_config_with_historic(dataset_dir: Path) -> "Path | None":
    """Return the first eval config under dataset_dir that contains historic_path.

    Searches feature_sweeps first (most stable configs), then anywhere under
    the dataset directory.  Used as a fallback for MLR datasets.
    """
    feature_sweeps = dataset_dir / "forecasts" / "feature_sweeps"
    candidates: list[Path] = []
    if feature_sweeps.is_dir():
        candidates.extend(sorted(feature_sweeps.rglob("config_evaluate_*.yml")))
    candidates.extend(
        p for p in sorted(dataset_dir.rglob("config_evaluate_*.yml"))
        if p not in candidates
    )
    for cfg_path in candidates:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
            if doc.get("evaluation", {}).get("historic_path"):
                return cfg_path
        except Exception:
            continue
    return None


def _iter_repair_targets(h_dir: Path) -> "list[tuple[Path, Path, str]]":
    """Return (class_dir, rep0_dir, layout_label) triples for one horizon directory.

    Searches in order:
      1. h_dir/ml/   (new per-class layout)
      2. h_dir/mlr/  (new per-class layout)
      3. h_dir/      (legacy flat layout — baseline lives at horizon level)

    The legacy flat layout is included only when neither ml/ nor mlr/ exist.
    Returns a list so all matching subdirectories are repaired.
    """
    targets = []
    found_class = False
    for model_class in ("ml", "mlr"):
        class_dir = h_dir / model_class
        rep0_dir = class_dir / "forecasts" / "rep_000"
        if rep0_dir.is_dir():
            found_class = True
            targets.append((class_dir, rep0_dir, model_class))
    if not found_class:
        rep0_dir = h_dir / "forecasts" / "rep_000"
        if rep0_dir.is_dir():
            targets.append((h_dir, rep0_dir, "flat"))
    return targets


def repair(data_root: Path, prefix: str) -> None:
    import f_Evaluate as eval_module
    import h_RunMCFeatureSelectionSweep as _h
    from utils.training import splitter as _splitter

    horizon_re = re.compile(r"^(\d+)hr$")

    for dataset_dir in sorted(data_root.iterdir()):
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith(prefix):
            continue
        horizons_root = dataset_dir / "horizons"
        if not horizons_root.is_dir():
            continue

        # Find the best available eval config with historic_path for this dataset.
        # Used as fallback when no per-horizon eval config exists (MLR datasets).
        dataset_eval_cfg = _find_eval_config_with_historic(dataset_dir)

        for h_dir in sorted(horizons_root.iterdir()):
            m = horizon_re.match(h_dir.name)
            if not m:
                continue

            for class_dir, rep0_dir, layout_label in _iter_repair_targets(h_dir):
                baseline_csv = class_dir / "baseline_summary.csv"
                loc_label = f"{dataset_dir.name} / {h_dir.name} [{layout_label}]"

                if _is_valid(baseline_csv):
                    print(f"[SKIP] {loc_label} — already valid")
                    continue

                rep0_eval_cfg = rep0_dir / "config_evaluate_rep_000.yml"

                # Determine which eval config to use for historic_path / evaluation params.
                # Priority: per-horizon rep_000 eval config > dataset-level feature sweep config.
                if rep0_eval_cfg.exists():
                    ref_eval_cfg = rep0_eval_cfg
                    reuse_split = True
                    split_source = rep0_dir
                elif dataset_eval_cfg is not None:
                    ref_eval_cfg = dataset_eval_cfg
                    reuse_split = True
                    split_source = rep0_dir  # test_files.txt written by _run_mlr_horizon_rep
                else:
                    print(f"[WARN] {loc_label} — no eval config with historic_path found, skipping")
                    continue

                try:
                    with open(ref_eval_cfg, "r", encoding="utf-8") as f:
                        ref_cfg = yaml.safe_load(f)

                    # Data dimensions: prefer the per-horizon rep_000 eval config.
                    # For MLR (no eval config), read input_columns from model_config.json
                    # in rep_000 — this reflects only the columns that actually had data,
                    # since MLR's fault_tolerant mode drops all-NaN columns at evaluation time.
                    # Using the full GP config column list causes the splitter to reject all
                    # samples when some sensors were absent for this dataset.
                    if rep0_eval_cfg.exists():
                        ref_data_cfg = ref_cfg.get("data", {})
                        _inp_cols = list(ref_data_cfg.get("input_columns", []))
                    else:
                        # MLR path: get output_columns/output_rows from any base training config.
                        ref_data_cfg = ref_cfg.get("data", {})
                        base_cfgs = sorted(dataset_dir.glob("config_*.yml"))
                        for bc in base_cfgs:
                            try:
                                with open(bc, "r", encoding="utf-8") as f:
                                    bc_doc = yaml.safe_load(f)
                                if bc_doc.get("data", {}).get("output_columns"):
                                    ref_data_cfg = bc_doc["data"]
                                    break
                            except Exception:
                                continue
                        # Read actual input_columns from the MLR model_config.json (may differ
                        # from the training config after fault_tolerant column dropping).
                        model_cfg_path = rep0_dir / "model_config.json"
                        if model_cfg_path.exists():
                            import json
                            with open(model_cfg_path, "r", encoding="utf-8") as f:
                                model_cfg_json = json.load(f)
                            _inp_cols = list(model_cfg_json.get("input_columns", ref_data_cfg.get("input_columns", [])))
                        else:
                            _inp_cols = list(ref_data_cfg.get("input_columns", []))

                    _in_r1 = int(ref_data_cfg.get("input_row_1", 0))
                    _in_r2 = int(ref_data_cfg.get("input_row_2", 0))
                    _out_cols = list(ref_data_cfg.get("output_columns", []))
                    _out_rows = list(ref_data_cfg.get("output_rows", []))
                    _sample_subdir = str(ref_data_cfg.get("sample_subdir", "samples"))

                    _, test_samples = _splitter(
                        str(class_dir.resolve()),
                        "rep_000",
                        _inp_cols,
                        slice(_in_r1, _in_r2),
                        _out_cols,
                        _out_rows,
                        fault_tolerant=True,
                        reuse_split=reuse_split,
                        split_source=split_source,
                        sample_subdir=_sample_subdir,
                        input_aggregation="none",
                    )
                    test_split_files = [str(s[2]) for s in test_samples]

                    summary_rows: list[dict] = []
                    _h._append_mlr_baseline_outputs(
                        summary_rows,
                        [],
                        ref_cfg=ref_cfg,
                        ref_cfg_path=ref_eval_cfg,
                        ref_data_cfg=ref_data_cfg,
                        data_dir=str(class_dir.resolve()),
                        sample_subdir=_sample_subdir,
                        output_columns=_out_cols,
                        output_rows=_out_rows,
                        forecast_name="",
                        test_samples=test_samples,
                        test_split_files=test_split_files,
                    )

                    if not summary_rows:
                        print(f"[WARN] {loc_label} — no baseline rows produced")
                        continue

                    eval_module._write_summary_csv(summary_rows, baseline_csv)
                    rmses = {r["label"]: f"{r['rmse']:.4g}" for r in summary_rows if "rmse" in r}
                    print(f"[OK]   {loc_label} — {rmses}")

                except Exception as exc:
                    import traceback
                    print(f"[ERR]  {loc_label} — {exc}")
                    traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair baseline_summary.csv files.")
    parser.add_argument("--data-root", default="data/output/regression")
    parser.add_argument("--dataset-prefix", default="MC")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()

    print(f"[INFO] data_root: {data_root}")
    repair(data_root, args.dataset_prefix)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

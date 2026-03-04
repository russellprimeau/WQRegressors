"""
Quick standalone split verifier for MC_ex datasets.

Scans data/output/regression for dataset directories starting with "MC_ex",
loads each model config (config_*.yml/json), and runs the exact split path used
from h_RunMCFeatureSelectionSweep.py:

    load_config -> merge_with_defaults -> load_and_split_data

Outputs:
- Console table of split sizes and key split settings.
- CSV report with per-config diagnostics.

Usage:
    python src/z_VerifyMCExSplits.py
    python src/z_VerifyMCExSplits.py --data-root data/output/regression --dataset-prefix MC_ex
"""

from __future__ import annotations

import argparse
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import e_Train as train_module


SUPPORTED_CONFIG_SUFFIXES = {".yml", ".yaml", ".json"}


def _raw_segment_key(filename: str) -> str:
    return re.sub(r"_mc_\d+(?=\.csv$)", "", str(filename))


def _collect_split_stats(train_samples: list, test_samples: list) -> dict[str, float]:
    train_files = [str(s[2]) for s in train_samples]
    test_files = [str(s[2]) for s in test_samples]
    all_files = train_files + test_files

    train_segments = {_raw_segment_key(name) for name in train_files}
    test_segments = {_raw_segment_key(name) for name in test_files}
    overlap_segments = train_segments.intersection(test_segments)

    n_total = len(all_files)
    n_train = len(train_files)
    n_test = len(test_files)
    n_train_segments = len(train_segments)
    n_test_segments = len(test_segments)
    n_total_segments = len({_raw_segment_key(name) for name in all_files})

    test_fraction = (n_test / n_total) if n_total > 0 else np.nan
    train_fraction = (n_train / n_total) if n_total > 0 else np.nan
    test_fraction_segments = (
        (n_test_segments / n_total_segments) if n_total_segments > 0 else np.nan
    )

    return {
        "n_total_samples": int(n_total),
        "n_train_samples": int(n_train),
        "n_test_samples": int(n_test),
        "train_fraction_samples": float(train_fraction),
        "test_fraction_samples": float(test_fraction),
        "n_total_raw_segments": int(n_total_segments),
        "n_train_raw_segments": int(n_train_segments),
        "n_test_raw_segments": int(n_test_segments),
        "test_fraction_segments": float(test_fraction_segments),
        "n_segment_leakage_overlap": int(len(overlap_segments)),
    }


def _load_and_split(config_path: Path) -> tuple[dict, list, list]:
    cfg = train_module.load_config(str(config_path))
    model_type = cfg.get("model_type", "")
    cfg = train_module.merge_with_defaults(cfg, model_type)
    train_samples, test_samples = train_module.load_and_split_data(cfg)
    return cfg, train_samples, test_samples


def _find_dataset_configs(dataset_dir: Path) -> list[Path]:
    configs = []
    for p in sorted(dataset_dir.glob("config_*")):
        if p.suffix.lower() in SUPPORTED_CONFIG_SUFFIXES:
            configs.append(p)
    return configs


def run_verification(data_root: Path, dataset_prefix: str, output_csv: Path) -> int:
    dataset_dirs = [
        p for p in sorted(data_root.iterdir()) if p.is_dir() and p.name.startswith(dataset_prefix)
    ]

    if not dataset_dirs:
        print(f"[ERROR] No datasets found with prefix '{dataset_prefix}' under: {data_root}")
        return 1

    rows: list[dict] = []
    for dataset_dir in dataset_dirs:
        configs = _find_dataset_configs(dataset_dir)
        if not configs:
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "config_file": "",
                    "model_type": "",
                    "status": "no_configs",
                    "error": "No config_*.yml/yaml/json found",
                }
            )
            continue

        for config_path in configs:
            print(f"\n[VERIFY] dataset={dataset_dir.name} config={config_path.name}")
            try:
                cfg, train_samples, test_samples = _load_and_split(config_path)
                split_cfg = cfg.get("data_split", {})
                data_cfg = cfg.get("data", {})
                stats = _collect_split_stats(train_samples, test_samples)

                row = {
                    "dataset": dataset_dir.name,
                    "config_file": config_path.name,
                    "model_type": cfg.get("model_type", ""),
                    "sample_subdir": data_cfg.get("sample_subdir", ""),
                    "forecast_name": data_cfg.get("forecast_name", ""),
                    "split_type_requested": split_cfg.get("split_type", ""),
                    "fault_tolerant": bool(split_cfg.get("fault_tolerant", False)),
                    "nan_tolerance": split_cfg.get("nan_tolerance", np.nan),
                    "test_size_target": split_cfg.get("test_size", np.nan),
                    "status": "ok",
                    "error": "",
                }
                row.update(stats)

                print(
                    "  "
                    f"train={row['n_train_samples']} test={row['n_test_samples']} total={row['n_total_samples']} | "
                    f"train_raw={row['n_train_raw_segments']} test_raw={row['n_test_raw_segments']} "
                    f"overlap={row['n_segment_leakage_overlap']}"
                )
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "config_file": config_path.name,
                        "model_type": "",
                        "sample_subdir": "",
                        "forecast_name": "",
                        "split_type_requested": "",
                        "fault_tolerant": "",
                        "nan_tolerance": "",
                        "test_size_target": "",
                        "n_total_samples": "",
                        "n_train_samples": "",
                        "n_test_samples": "",
                        "train_fraction_samples": "",
                        "test_fraction_samples": "",
                        "n_total_raw_segments": "",
                        "n_train_raw_segments": "",
                        "n_test_raw_segments": "",
                        "test_fraction_segments": "",
                        "n_segment_leakage_overlap": "",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"  [ERROR] {type(exc).__name__}: {exc}")
                traceback.print_exc()

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[INFO] Verification report written: {output_csv}")

    ok_rows = df[df["status"] == "ok"] if "status" in df.columns else pd.DataFrame()
    if not ok_rows.empty:
        print("\nSummary (ok rows only)")
        by_model = (
            ok_rows.groupby("model_type", dropna=False)[["n_train_samples", "n_test_samples"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        print(by_model.to_string(index=False))

        leakage_rows = ok_rows[ok_rows["n_segment_leakage_overlap"] > 0]
        if not leakage_rows.empty:
            print("\n[WARN] Segment leakage detected in these rows:")
            cols = ["dataset", "config_file", "n_segment_leakage_overlap"]
            print(leakage_rows[cols].to_string(index=False))
    else:
        print("\n[WARN] No successful rows to summarize.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify train/test split behavior for MC_ex datasets using e_Train split logic."
    )
    parser.add_argument("--data-root", type=str, default="data/output/regression")
    parser.add_argument("--dataset-prefix", type=str, default="MC_ex")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/output/regression/_split_verification/mc_ex_split_verification.csv",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (workspace_root / data_root).resolve()

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = (workspace_root / output_csv).resolve()

    return run_verification(
        data_root=data_root,
        dataset_prefix=str(args.dataset_prefix),
        output_csv=output_csv,
    )


if __name__ == "__main__":
    raise SystemExit(main())


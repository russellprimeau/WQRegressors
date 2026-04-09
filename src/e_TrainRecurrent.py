"""
Standalone training script for recurrent forecasting models.

Supported model types:
  recurrent_transformer  — RecurrentTransformerModel
  lstm                   — LSTMForecasterModel

Usage:
  .venv/Scripts/python src/e_TrainRecurrent.py --config path/to/config.yaml

Config structure mirrors e_Train.py. Key differences:
  - data.state_column: name of the column that is both a predictor and the target
  - No separate output target column needed beyond the state column itself
  - hyperparameters.cv_tuning.enabled: true to use Optuna CV tuning

Example config (YAML):

  model_type: "recurrent_transformer"
  model_name: "recurrent_transformer_01"
  device: "cpu"

  data:
    data_dir: "data/output/regression/MC_exArsenic"
    forecast_name: "recurrent_transformer_01"
    sample_subdir: "samples"
    state_column: "Arsenic (µg/L)_state"
    input_columns:
      - "Pfl - Water temperature (C)"
      - "Arsenic (µg/L)_state"
    output_columns:
      - "Arsenic (µg/L)_state"
    input_row_1: 0
    input_row_2: 168
    output_rows:
      - 167

  data_split:
    split_type: "temporal"
    test_size: 0.3
    random_state: 42
    fault_tolerant: false
    reuse_split: false
    split_source: null
    nan_tolerance: null

  hyperparameters:
    num_epochs: 200
    batch_size: 16
    learning_rate: 0.001
    patience: 20
    model_dim: 64        # RecurrentTransformer only
    num_heads: 4         # RecurrentTransformer only
    num_layers: 2
    dropout: 0.1
    hidden_dim: 64       # LSTM only
    cv_tuning:
      enabled: false
      n_folds: 5
      n_trials: 20
      cv_epochs: 50
      cv_patience: 10
      param_space:
        model_dim: [64, 128, 256]
        num_heads: [2, 4, 8]
        num_layers: {low: 2, high: 6, type: "int"}
        dropout: {low: 0.05, high: 0.4}
        learning_rate: {low: 0.0005, high: 0.05, log: true}
        batch_size: [8, 16, 32]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow running from project root or from src/
_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from utils.config_utils import load_config
from utils.training import write_config, splitter
from utils.recurrent_models import (
    RecurrentTransformerModel,
    LSTMForecasterModel,
    train_recurrent_model,
)


# ---------------------------------------------------------------------------
# Default config sections
# ---------------------------------------------------------------------------

DEFAULT_DATA_SPLIT = {
    "split_type": "temporal",
    "test_size": 0.3,
    "random_state": 42,
    "fault_tolerant": False,
    "reuse_split": False,
    "split_source": None,
    "nan_tolerance": None,
}

DEFAULT_HYPERPARAMETERS = {
    "num_epochs": 100,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "patience": 20,
    "model_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "hidden_dim": 64,
    "grad_clip_norm": 1.0,
    "warmup_epochs": 10,
    "cv_tuning": {
        "enabled": False,
        "n_folds": 5,
        "n_trials": 20,
        "cv_epochs": 50,
        "cv_patience": 10,
        "param_space": {},
    },
}


def _merge_defaults(config: dict) -> dict:
    config.setdefault("device", "cpu")
    split_cfg = config.setdefault("data_split", {})
    for k, v in DEFAULT_DATA_SPLIT.items():
        split_cfg.setdefault(k, v)
    hyper_cfg = config.setdefault("hyperparameters", {})
    for k, v in DEFAULT_HYPERPARAMETERS.items():
        if k == "cv_tuning":
            cv = hyper_cfg.setdefault("cv_tuning", {})
            for ck, cv_v in DEFAULT_HYPERPARAMETERS["cv_tuning"].items():
                cv.setdefault(ck, cv_v)
        else:
            hyper_cfg.setdefault(k, v)
    return config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_and_split(config: dict):
    data_cfg = config["data"]
    split_cfg = config["data_split"]

    data_dir = str(Path(config["__config_dir"]) / data_cfg["data_dir"])
    data_dir = str(Path(data_dir).resolve())
    data_cfg["data_dir"] = data_dir

    sample_subdir = data_cfg.get("sample_subdir", "samples")
    input_rows = slice(data_cfg["input_row_1"], data_cfg["input_row_2"])

    train_samples, test_samples = splitter(
        data_dir,
        data_cfg["forecast_name"],
        data_cfg["input_columns"],
        input_rows,
        data_cfg["output_columns"],
        data_cfg["output_rows"],
        split_cfg["fault_tolerant"],
        split_cfg["reuse_split"],
        split_cfg["split_source"],
        split_cfg["split_type"],
        split_cfg["test_size"],
        split_cfg["random_state"],
        sample_subdir,
        split_cfg["nan_tolerance"],
    )
    print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")
    return train_samples, test_samples


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _save_outputs(config: dict, result: dict, model_cls):
    data_cfg = config["data"]
    out_dir = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model checkpoint
    ckpt_name = (
        "recurrent_transformer_model.pt"
        if model_cls is RecurrentTransformerModel
        else "lstm_model.pt"
    )
    torch.save(result["model"].state_dict(), out_dir / ckpt_name)
    print(f"Model saved: {out_dir / ckpt_name}")

    # Model config JSON
    write_config(result["final_model_config"], data_cfg["data_dir"], data_cfg["forecast_name"], "")

    # Best params JSON (alongside model config)
    best_params_path = out_dir / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(result["best_params"], f, indent=2)

    # Predictions CSVs
    def _save_predictions(preds, targets, fnames, split_name):
        df = pd.DataFrame({
            "filename": fnames,
            "prediction": preds,
            "target": targets,
        })
        path = out_dir / f"predictions_{split_name}.csv"
        df.to_csv(path, index=False)
        print(f"Predictions saved: {path}  ({len(df)} rows)")

    _save_predictions(
        result["predictions_train"], result["targets_train"], result["filenames_train"], "train"
    )
    _save_predictions(
        result["predictions_test"], result["targets_test"], result["filenames_test"], "test"
    )


# ---------------------------------------------------------------------------
# Per-model-type wrappers
# ---------------------------------------------------------------------------

def train_recurrent_transformer(config: dict, train_samples, test_samples):
    result = train_recurrent_model(config, train_samples, test_samples, RecurrentTransformerModel)
    _save_outputs(config, result, RecurrentTransformerModel)


def train_lstm(config: dict, train_samples, test_samples):
    result = train_recurrent_model(config, train_samples, test_samples, LSTMForecasterModel)
    _save_outputs(config, result, LSTMForecasterModel)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Training script for recurrent forecasting models (RecurrentTransformer, LSTM)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("LOADING CONFIGURATION")
    print("=" * 80)
    print(f"Config file: {args.config}")

    config = load_config(args.config)

    required = ["model_type", "model_name", "data"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    if "state_column" not in config["data"]:
        raise ValueError("data.state_column is required for recurrent models")

    model_type = config["model_type"]
    print(f"Model type: {model_type}")

    config = _merge_defaults(config)

    device = torch.device(config["device"])
    print(f"Using device: {device}")

    print("\n" + "=" * 80)
    print("LOADING AND SPLITTING DATA")
    print("=" * 80)
    train_samples, test_samples = _load_and_split(config)

    if model_type == "recurrent_transformer":
        train_recurrent_transformer(config, train_samples, test_samples)
    elif model_type == "lstm":
        train_lstm(config, train_samples, test_samples)
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            "Expected 'recurrent_transformer' or 'lstm'."
        )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

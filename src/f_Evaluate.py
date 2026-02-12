"""
Unified evaluation script for regression, transformer, and classification models.
Uses a YAML/JSON config file to control what gets evaluated.
"""

import os
import json
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
import torch
import xgboost as xgb

from utils.training import load_samples
from utils.transformer import TimeSeriesTargetDataset, TimeSeriesTransformer
from utils.evaluation import (
    load_secondary,
    evaluate_naive,
    evaluate_seasonal,
    evaluate_linear,
    evaluate_transformer,
    visualizer,
    classification_visualizer,
    reverse_normalize,
    binarize_predictions,
    apply_saved_normalize,
)


DEFAULT_EVAL_CONFIG = {
    "run_regression": True,
    "run_threshold_classification": False,
    "run_pure_classification": False,
    "run_baselines": False,
    "num_samples": 200,
    "debug_plot": False,
    "debug_examples": 10,
    "gap_hours": 1,
    "window_hours": 550,
    "diurnal_window": 1,
    "historic_path": "../data/output/regression/Consolidated_sparse.csv",
    "thresholds_path": "../data/input/Limits.csv",
    "normalization_path": "../data/input/normalization.json",
    "use_normalized_thresholds": False,
}


def load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if path.suffix in [".yaml", ".yml"]:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config file format: {path.suffix}")


def merge_eval_config(cfg):
    eval_cfg = cfg.get("evaluation", {})
    merged = DEFAULT_EVAL_CONFIG.copy()
    merged.update(eval_cfg)
    return merged


def load_model_config(data_dir, forecast_name, model_name, fallback_data=None):
    config_path = Path(data_dir, "forecasts", forecast_name, model_name, "model_config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)

    if fallback_data is None:
        raise FileNotFoundError(f"model_config.json not found at {config_path}")

    return {
        "input_columns": fallback_data["input_columns"],
        "output_columns": fallback_data["output_columns"],
        "input_row_1": fallback_data["input_row_1"],
        "input_row_2": fallback_data["input_row_2"],
        "output_rows": fallback_data["output_rows"],
    }


def load_test_samples(data_dir, forecast_name, input_columns, output_columns, input_rows, output_rows):
    reloadset = Path(data_dir, "forecasts", forecast_name, "test_files.txt")
    with open(reloadset) as f:
        test_files = [line.strip() for line in f]

    test_samples = load_samples(
        os.path.join(data_dir, "samples"),
        input_columns=input_columns,
        output_columns=output_columns,
        input_rows=input_rows,
        output_rows=output_rows,
        file_list=test_files,
        fault_tolerant=True,
    )
    return test_samples


def get_output_dim(data_dir, output_columns, output_rows):
    sample_files = sorted(os.listdir(Path(data_dir, "samples")))
    sample_df = pd.read_csv(Path(data_dir, "samples", sample_files[0]))
    return len(output_columns) * len(sample_df.iloc[output_rows:])


def load_model(model_type, data_dir, forecast_name, model_name, model_config, device):
    model_path = Path(data_dir, "forecasts", forecast_name, model_name)

    if model_type == "transformer":
        model = TimeSeriesTransformer(model_config).to(device)
        model.load_state_dict(torch.load(model_path / "transformer_model.pt", map_location=device))
        model.eval()
        return model

    if model_type == "xgb_regressor":
        model = xgb.XGBRegressor()
        model.load_model(model_path / "xgboost_model.json")
        return model

    if model_type == "xgb_classifier":
        model = xgb.XGBClassifier()
        model.load_model(model_path / "xgboost_model.json")
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


def load_thresholds(eval_cfg):
    thresholds_df = pd.read_csv(Path(eval_cfg["thresholds_path"]), sep=";", decimal=".")
    if eval_cfg["use_normalized_thresholds"]:
        thresholds_df = apply_saved_normalize(
            thresholds_df, param_file=Path(eval_cfg["normalization_path"])
        )
    return thresholds_df


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()

    config = load_config(args.config)

    model_type = config["model_type"]
    model_name = config["model_name"]
    data_cfg = config["data"]
    eval_cfg = merge_eval_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    matplotlib.use("Agg")

    model_config = load_model_config(
        data_cfg["data_dir"],
        data_cfg["forecast_name"],
        model_name,
        fallback_data=data_cfg,
    )

    input_columns = model_config["input_columns"]
    output_columns = model_config["output_columns"]
    input_rows = slice(model_config["input_row_1"], model_config["input_row_2"])
    output_rows = model_config["output_rows"]

    test_samples = load_test_samples(
        data_cfg["data_dir"],
        data_cfg["forecast_name"],
        input_columns,
        output_columns,
        input_rows,
        output_rows,
    )
    test_dataset = TimeSeriesTargetDataset(test_samples)

    X_test = np.array([s[0].flatten() for s in test_samples])
    y_test = np.array([s[1].flatten()[0] for s in test_samples])

    output_dim = get_output_dim(data_cfg["data_dir"], output_columns, output_rows)

    model = load_model(model_type, data_cfg["data_dir"], data_cfg["forecast_name"], model_name, model_config, device)

    regression_pairs = []
    regression_labels = []

    if eval_cfg["run_regression"]:
        if model_type == "transformer":
            preds, targets = evaluate_transformer(model, test_dataset, device)
            regression_pairs.append((preds, targets))
            regression_labels.append("Transformer")
        elif model_type == "xgb_regressor":
            preds_flat = model.predict(X_test)
            preds = preds_flat.reshape(-1, output_dim)
            targets = y_test.reshape(-1, output_dim)
            regression_pairs.append((preds, targets))
            regression_labels.append("XGBRegressor")
        elif model_type == "xgb_classifier":
            print("Skipping regression evaluation for xgb_classifier model_type")

    if eval_cfg["run_baselines"]:
        secondary, eval_cfg["window_hours"] = load_secondary(output_columns, eval_cfg["window_hours"])
        naive_preds, naive_targets = evaluate_naive(
            test_dataset,
            eval_cfg["historic_path"],
            output_columns,
            data_cfg["data_dir"],
            output_rows=output_rows,
            gap_hours=eval_cfg["gap_hours"],
        )
        linear_preds, linear_targets = evaluate_linear(
            data_cfg["data_dir"],
            data_cfg["forecast_name"],
            test_dataset,
            eval_cfg["historic_path"],
            output_columns,
            output_rows=output_rows,
            window_hours=eval_cfg["window_hours"],
            gap_hours=eval_cfg["gap_hours"],
            debug_plot=eval_cfg["debug_plot"],
            examples=eval_cfg["debug_examples"],
        )
        seasonal_preds, seasonal_targets = evaluate_seasonal(
            test_dataset,
            eval_cfg["historic_path"],
            output_columns,
            data_cfg["data_dir"],
            data_cfg["forecast_name"],
            output_rows=output_rows,
            diurnal_window=eval_cfg["diurnal_window"],
            secondary=secondary,
        )

        regression_pairs.extend(
            [(naive_preds, naive_targets), (linear_preds, linear_targets), (seasonal_preds, seasonal_targets)]
        )
        regression_labels.extend(["Naive", "Linear", "Seasonal"])

    if eval_cfg["run_regression"] and regression_pairs:
        visualizer(
            *regression_pairs,
            labels=regression_labels,
            forecast_name=data_cfg["forecast_name"],
            directory=data_cfg["data_dir"],
            num_samples=eval_cfg["num_samples"],
        )

    if eval_cfg["run_threshold_classification"] and regression_pairs:
        thresholds_df = load_thresholds(eval_cfg)
        class_results = []
        for preds, targets in regression_pairs:
            if eval_cfg["use_normalized_thresholds"]:
                preds_eval = preds
                targets_eval = targets
            else:
                preds_eval = reverse_normalize(preds, output_columns, Path(eval_cfg["normalization_path"]))
                targets_eval = reverse_normalize(targets, output_columns, Path(eval_cfg["normalization_path"]))

            bin_preds = binarize_predictions(preds_eval, output_columns=output_columns, thresholds_df=thresholds_df)
            bin_targets = binarize_predictions(targets_eval, output_columns=output_columns, thresholds_df=thresholds_df)
            class_results.append((bin_preds, bin_targets))

        classification_visualizer(
            *class_results,
            labels=regression_labels,
            directory=data_cfg["data_dir"],
            forecast_name=data_cfg["forecast_name"],
            num_samples=eval_cfg["num_samples"],
        )

    if eval_cfg["run_pure_classification"]:
        if model_type != "xgb_classifier":
            print("Skipping pure classification: model_type is not xgb_classifier")
        else:
            preds_flat = model.predict(X_test)
            preds = preds_flat.reshape(-1, output_dim)
            targets = np.array([np.rint(s[1].flatten()) for s in test_samples]).reshape(-1, output_dim)

            classification_visualizer(
                (preds, targets),
                labels=["XGBClassifier"],
                directory=data_cfg["data_dir"],
                forecast_name=data_cfg["forecast_name"],
                num_samples=eval_cfg["num_samples"],
            )


if __name__ == "__main__":
    main()

"""
Consolidated training script for Transformer, XGBoost Regressor, and XGBoost Classifier models.
Supports configuration via YAML/JSON config files.
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
from torch.utils.data import DataLoader

from utils.training import write_config, splitter
from utils.transformer import train_model as train_transformer, TimeSeriesTransformer, TimeSeriesTargetDataset


# ===========================================================================================
# DEFAULT CONFIGURATIONS
# ===========================================================================================

DEFAULT_COMMON_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "matplotlib_backend": "Agg",
}

DEFAULT_TRANSFORMER_CONFIG = {
    "model_dim": 128,
    "num_heads": 4,
    "num_layers": 8,
    "dropout": 0.1,
    "batch_size": 10,
    "num_epochs": 100,
    "loss_threshold": 0.000001,
    "learning_rate": 1e-4,
    "patience": 10,
}

DEFAULT_XGB_REGRESSOR_CONFIG = {
    "metric": "rmse",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_estimators": 1100,
    "max_depth": 10,
    "subsample": 0.2,
    "colsample_bytree": 0.8,
    "learning_rate": 0.01,
    "n_jobs": -1,
    "early_stopping_rounds": 200,
}

DEFAULT_XGB_CLASSIFIER_CONFIG = {
    "eval_metric": "logloss",
    "tree_method": "hist",
    "objective": "binary:logistic",
    "n_estimators": 1500,
    "max_depth": 10,
    "subsample": 0.2,
    "colsample_bytree": 0.8,
    "learning_rate": 0.01,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

DEFAULT_DATA_SPLIT_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "reuse_split": False,
    "split_source": None,
    "split_type": "random",
}


# ===========================================================================================
# CONFIG LOADING AND MERGING
# ===========================================================================================

def load_config(config_path):
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if path.suffix in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    return config


def merge_with_defaults(config, model_type):
    """Merge provided config with defaults, printing when defaults are applied."""
    merged_config = config.copy()
    
    # Get model-specific defaults
    if model_type == "transformer":
        defaults = DEFAULT_TRANSFORMER_CONFIG
    elif model_type == "xgb_regressor":
        defaults = DEFAULT_XGB_REGRESSOR_CONFIG
    elif model_type == "xgb_classifier":
        defaults = DEFAULT_XGB_CLASSIFIER_CONFIG
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Merge in data split defaults
    if "data_split" not in merged_config:
        merged_config["data_split"] = {}
    
    for key, default_value in DEFAULT_DATA_SPLIT_CONFIG.items():
        if key not in merged_config["data_split"]:
            merged_config["data_split"][key] = default_value
            print(f"  [DEFAULT] data_split.{key} = {default_value}")
    
    # Merge in model-specific defaults
    if "hyperparameters" not in merged_config:
        merged_config["hyperparameters"] = {}
    
    for key, default_value in defaults.items():
        if key not in merged_config["hyperparameters"]:
            merged_config["hyperparameters"][key] = default_value
            print(f"  [DEFAULT] hyperparameters.{key} = {default_value}")
    
    # Add device if not specified
    if "device" not in merged_config:
        merged_config["device"] = DEFAULT_COMMON_CONFIG["device"]
        print(f"  [DEFAULT] device = {merged_config['device']}")
    
    # Add matplotlib backend if not specified
    if "matplotlib_backend" not in merged_config:
        merged_config["matplotlib_backend"] = DEFAULT_COMMON_CONFIG["matplotlib_backend"]
    
    return merged_config


# ===========================================================================================
# DATA LOADING
# ===========================================================================================

def load_and_split_data(config):
    """Load and split data according to configuration."""
    data_cfg = config["data"]
    split_cfg = config["data_split"]
    
    input_rows = slice(data_cfg["input_row_1"], data_cfg["input_row_2"])
    
    train_samples, test_samples = splitter(
        data_cfg["data_dir"],
        data_cfg["forecast_name"],
        data_cfg["input_columns"],
        input_rows,
        data_cfg["output_columns"],
        data_cfg["output_rows"],
        False,  # normalize - set based on model type if needed
        split_cfg["reuse_split"],
        split_cfg["split_source"],
        split_cfg["split_type"],
        split_cfg["test_size"],
        split_cfg["random_state"]
    )
    
    return train_samples, test_samples, input_rows


# ===========================================================================================
# TRANSFORMER TRAINING
# ===========================================================================================

def train_transformer_model(config, train_samples, test_samples):
    """Train Transformer model."""
    print("\n" + "="*80)
    print("TRAINING TRANSFORMER MODEL")
    print("="*80)
    
    device = torch.device(config["device"])
    data_cfg = config["data"]
    hyper_cfg = config["hyperparameters"]
    
    # Prepare data
    train_dataset = TimeSeriesTargetDataset(train_samples)
    test_dataset = TimeSeriesTargetDataset(test_samples)
    trainloader = DataLoader(train_dataset, batch_size=hyper_cfg["batch_size"], shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=hyper_cfg["batch_size"], shuffle=True)
    
    # Create config dictionary for model
    input_rows = slice(data_cfg["input_row_1"], data_cfg["input_row_2"])
    files = [f for f in os.listdir(Path(data_cfg["data_dir"], 'samples')) if
             os.path.isfile(Path(data_cfg["data_dir"], 'samples', f))]
    sample_df = pd.read_csv(Path(data_cfg["data_dir"], 'samples', sorted(files)[0]))
    
    model_config = {
        'input_dim': len(data_cfg["input_columns"]),
        'model_dim': hyper_cfg["model_dim"],
        'num_heads': hyper_cfg["num_heads"],
        'num_layers': hyper_cfg["num_layers"],
        'dropout': hyper_cfg["dropout"],
        'output_dim': len(data_cfg["output_columns"]) * len(sample_df.iloc[data_cfg["output_rows"]:]),
        'seq_len': data_cfg["input_row_2"] - data_cfg["input_row_1"],
        'input_columns': data_cfg["input_columns"],
        'input_row_1': data_cfg["input_row_1"],
        'input_row_2': data_cfg["input_row_2"],
        'output_columns': data_cfg["output_columns"],
        'output_rows': data_cfg["output_rows"],
    }
    
    # Write config
    write_config(model_config, data_cfg["data_dir"], data_cfg["forecast_name"], config["model_name"])
    
    # Create and train model
    model = TimeSeriesTransformer(model_config).to(device)
    
    train_transformer(
        data_cfg["data_dir"],
        model,
        data_cfg["forecast_name"],
        trainloader,
        testloader,
        device,
        hyper_cfg["num_epochs"],
        hyper_cfg["learning_rate"],
        hyper_cfg["loss_threshold"],
        hyper_cfg["patience"]
    )
    
    # Save model
    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"], config["model_name"])
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), save_path / "transformer_model.pt")
    print(f"\nModel saved to: {save_path / 'transformer_model.pt'}")


# ===========================================================================================
# XGBOOST REGRESSOR TRAINING
# ===========================================================================================

def train_xgb_regressor_model(config, train_samples, test_samples):
    """Train XGBoost Regressor model."""
    print("\n" + "="*80)
    print("TRAINING XGBOOST REGRESSOR MODEL")
    print("="*80)
    
    data_cfg = config["data"]
    hyper_cfg = config["hyperparameters"]
    
    # Prepare data
    X_train = np.array([s[0].flatten() for s in train_samples])
    y_train = np.array([s[1].flatten()[0] for s in train_samples])
    X_test = np.array([s[0].flatten() for s in test_samples])
    y_test = np.array([s[1].flatten()[0] for s in test_samples])
    
    # Create config dictionary
    input_rows = slice(data_cfg["input_row_1"], data_cfg["input_row_2"])
    files = [f for f in os.listdir(Path(data_cfg["data_dir"], 'samples')) if
             os.path.isfile(Path(data_cfg["data_dir"], 'samples', f))]
    sample_df = pd.read_csv(Path(data_cfg["data_dir"], 'samples', sorted(files)[0]))
    
    model_config = {
        'input_dim': len(data_cfg["input_columns"]),
        'output_dim': len(data_cfg["output_columns"]) * len(sample_df.iloc[data_cfg["output_rows"]:]),
        'seq_len': data_cfg["input_row_2"] - data_cfg["input_row_1"],
        'input_columns': data_cfg["input_columns"],
        'input_row_1': data_cfg["input_row_1"],
        'input_row_2': data_cfg["input_row_2"],
        'output_columns': data_cfg["output_columns"],
        'output_rows': data_cfg["output_rows"],
    }
    
    write_config(model_config, data_cfg["data_dir"], data_cfg["forecast_name"], config["model_name"])
    
    # Create and train model
    model = xgb.XGBRegressor(
        tree_method=hyper_cfg["tree_method"],
        objective=hyper_cfg["objective"],
        n_estimators=hyper_cfg["n_estimators"],
        max_depth=hyper_cfg["max_depth"],
        subsample=hyper_cfg["subsample"],
        colsample_bytree=hyper_cfg["colsample_bytree"],
        learning_rate=hyper_cfg["learning_rate"],
        n_jobs=hyper_cfg["n_jobs"],
        early_stopping_rounds=hyper_cfg["early_stopping_rounds"]
    )
    
    print("Training XGBRegressor...")
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
    
    # Save model
    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"], config["model_name"])
    os.makedirs(save_path, exist_ok=True)
    model.save_model(save_path / "xgboost_model.json")
    print(f"\nModel saved to: {save_path / 'xgboost_model.json'}")
    
    # Plot results
    results = model.evals_result()
    metric = hyper_cfg["metric"]
    epochs = len(results['validation_0'][metric])
    plt.figure(figsize=(8, 5))
    plt.loglog(range(epochs), results['validation_0'][metric], label='Training Loss')
    plt.loglog(range(epochs), results['validation_1'][metric], label='Validation Loss')
    plt.xlabel('Boosting Rounds')
    plt.ylabel(metric)
    plt.grid(True, which="both", ls="--")
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(save_path / "loss_plot.png")
    plt.close()
    print(f"Loss plot saved to: {save_path / 'loss_plot.png'}")


# ===========================================================================================
# XGBOOST CLASSIFIER TRAINING
# ===========================================================================================

def train_xgb_classifier_model(config, train_samples, test_samples):
    """Train XGBoost Classifier model."""
    print("\n" + "="*80)
    print("TRAINING XGBOOST CLASSIFIER MODEL")
    print("="*80)
    
    data_cfg = config["data"]
    hyper_cfg = config["hyperparameters"]
    
    # Prepare data and ensure binary outputs
    X_train = np.array([s[0].flatten() for s in train_samples])
    y_train = np.array([int(round(s[1].flatten()[0])) for s in train_samples])
    X_test = np.array([s[0].flatten() for s in test_samples])
    y_test = np.array([int(round(s[1].flatten()[0])) for s in test_samples])
    
    # Create config dictionary
    input_rows = slice(data_cfg["input_row_1"], data_cfg["input_row_2"])
    files = [f for f in os.listdir(Path(data_cfg["data_dir"], 'samples')) if
             os.path.isfile(Path(data_cfg["data_dir"], 'samples', f))]
    sample_df = pd.read_csv(Path(data_cfg["data_dir"], 'samples', sorted(files)[0]))
    
    model_config = {
        'input_dim': len(data_cfg["input_columns"]),
        'output_dim': len(data_cfg["output_columns"]) * len(sample_df.iloc[data_cfg["output_rows"]:]),
        'seq_len': data_cfg["input_row_2"] - data_cfg["input_row_1"],
        'input_columns': data_cfg["input_columns"],
        'input_row_1': data_cfg["input_row_1"],
        'input_row_2': data_cfg["input_row_2"],
        'output_columns': data_cfg["output_columns"],
        'output_rows': data_cfg["output_rows"],
    }
    
    write_config(model_config, data_cfg["data_dir"], data_cfg["forecast_name"], config["model_name"])
    
    # Create and train model
    model = xgb.XGBClassifier(
        eval_metric=hyper_cfg["eval_metric"],
        tree_method=hyper_cfg["tree_method"],
        objective=hyper_cfg["objective"],
        n_estimators=hyper_cfg["n_estimators"],
        max_depth=hyper_cfg["max_depth"],
        subsample=hyper_cfg["subsample"],
        colsample_bytree=hyper_cfg["colsample_bytree"],
        learning_rate=hyper_cfg["learning_rate"],
        n_jobs=hyper_cfg["n_jobs"],
        early_stopping_rounds=hyper_cfg["early_stopping_rounds"],
    )
    
    print("Training XGBClassifier...")
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
    
    # Save model
    save_path = Path(data_cfg["data_dir"], "forecasts", data_cfg["forecast_name"], config["model_name"])
    os.makedirs(save_path, exist_ok=True)
    model.save_model(save_path / "xgboost_model.json")
    print(f"\nModel saved to: {save_path / 'xgboost_model.json'}")
    
    # Plot results
    results = model.evals_result()
    metric = hyper_cfg["eval_metric"]
    epochs = len(results['validation_0'][metric])
    plt.figure(figsize=(8, 5))
    plt.loglog(range(epochs), results['validation_0'][metric], label='Train logloss')
    plt.loglog(range(epochs), results['validation_1'][metric], label='Validation logloss')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Logloss')
    plt.grid(True, which="both", ls="--")
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(save_path / "loss_plot.png")
    plt.close()
    print(f"Loss plot saved to: {save_path / 'loss_plot.png'}")


# ===========================================================================================
# MAIN ORCHESTRATION
# ===========================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Consolidated training script for Transformer, XGBoost Regressor, and XGBoost Classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML or JSON)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("LOADING CONFIGURATION")
    print("="*80)
    print(f"Config file: {args.config}")
    
    # Load config
    config = load_config(args.config)
    
    # Validate required fields
    required_fields = ["model_type", "model_name", "data"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    model_type = config["model_type"]
    print(f"Model type: {model_type}")
    
    # Merge with defaults and print applied defaults
    print("\nApplying defaults:")
    config = merge_with_defaults(config, model_type)
    
    # Configure device and backend
    device = torch.device(config["device"])
    print(f"\nUsing device: {device}")
    matplotlib.use(config["matplotlib_backend"])
    
    # Load and split data
    print("\n" + "="*80)
    print("LOADING AND SPLITTING DATA")
    print("="*80)
    train_samples, test_samples, input_rows = load_and_split_data(config)
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Train appropriate model
    if model_type == "transformer":
        train_transformer_model(config, train_samples, test_samples)
    elif model_type == "xgb_regressor":
        train_xgb_regressor_model(config, train_samples, test_samples)
    elif model_type == "xgb_classifier":
        train_xgb_classifier_model(config, train_samples, test_samples)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

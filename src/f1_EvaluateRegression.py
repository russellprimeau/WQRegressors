"""
Compares accuracy of Time Series Forecasting using various forecasts.
"""

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
import torch
import xgboost as xgb
from utils.training import load_samples
from utils.transformer import TimeSeriesTargetDataset, TimeSeriesTransformer
from utils.evaluation import (load_secondary, evaluate_naive, evaluate_seasonal, evaluate_linear, evaluate_transformer,
                              visualizer, classification_visualizer, reverse_normalize, binarize_predictions)

if __name__ == '__main__':
    ## Configure execution space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    matplotlib.use('Agg')  # Non-interactive backend for file output to handle remote machine installation errors

    ##################################################################################################################
    ## Load input, output and model hyperparameters from data_dir
    # data_dir = "../data/output/regression/Kimtall12hr"  # Parent directory of test/train sample folder
    data_dir = "../data/output/regression/Farge"
    forecast_name = "nowcast"
    model_name = "xgbregressor"

    with open(Path(data_dir, 'forecasts', forecast_name, model_name, 'model_config.json'), 'r') as f:
        config = json.load(f)

    input_columns = config["input_columns"]
    output_columns = config["output_columns"]
    input_rows = slice(config["input_row_1"], config["input_row_2"])
    output_rows = config["output_rows"]

    ## Configure simple non-ML ("baseline") model calculation methods
    historic = "../data/output/regression/Consolidated_sparse.csv"  # Path to file with baseline model input
    gap_hours = 1  # Period before first forecast value from which input data is not used in baseline forecasts
    window_hours = 550  # Length of period for linear regression training (min. ~530 hrs for Eurofins params)
    diurnal_window = 1  # Number of hours before/after target time to include in diurnal average for seasonality model
    secondary, window_hours = load_secondary(output_columns, window_hours)  # Check output_columns and automatically adjust some baseline forecasts

    ################################################################################################################
    ## Prepare data for evaluation in various forecasts
    ## Run evaluation using samples excluded from training
    reloadset = Path(data_dir, "forecasts", forecast_name, "test_files.txt")
    with open(reloadset) as f:
        test_files = [line.strip() for line in f]
    test_samples = load_samples(os.path.join(data_dir,"samples"),input_columns=input_columns,output_columns=output_columns,
        input_rows=input_rows, output_rows=output_rows, file_list=test_files, fault_tolerant=True)
    test_dataset = TimeSeriesTargetDataset(test_samples)

    # # Alternative: for full-coverage plotting of sparse data, evaluate forecasts on complete sample set (train + test)
    # samples = load_samples(os.path.join(data_dir, 'samples'), input_columns=input_columns,
    #                        output_columns=output_columns,
    #                        input_rows=input_rows, output_rows=output_rows, fault_tolerant=True)
    # test_dataset = TimeSeriesTargetDataset(samples)
    # test_samples = samples

    ## Flatten sample arrays for XGB methods
    X_test = np.array([s[0].flatten() for s in test_samples])
    y_test = np.array([s[1].flatten()[0] for s in test_samples])

    ################################################################################################################
    # ## Prepare transformer model for evaluation
    # transformer_model = TimeSeriesTransformer(config).to(device)
    # transformer_model.load_state_dict(torch.load(os.path.join(data_dir, "forecasts", forecast_name, "transformer",
    #                                               "transformer_model.pt"), map_location=device))
    # transformer_model.eval()  # Set to evaluation mode

    ################################################################################################################
    ## Prepare XGBRegresssor model for evaluation
    xgbr_model = xgb.XGBRegressor()
    xgbr_path = Path(data_dir, "forecasts", forecast_name, "XGBRegressor", "xgboost_model.json")
    xgbr_model.load_model(xgbr_path)

    ##################################################################################################################
    # Evaluate regression forecasts
    # transformer_preds, transformer_targets = evaluate_transformer(transformer_model, test_dataset, device)
    naive_preds, naive_targets = evaluate_naive(test_dataset, historic, output_columns, data_dir,
                                                output_rows=output_rows, gap_hours=gap_hours)
    linear_preds, linear_targets = evaluate_linear(data_dir, forecast_name, test_dataset, historic, output_columns,
                                                   output_rows=output_rows, window_hours=window_hours,
                                                   gap_hours=gap_hours,
                                                   debug_plot=True, examples=100)

    seasonal_preds, seasonal_targets = evaluate_seasonal(test_dataset, historic, output_columns, data_dir, forecast_name,
                                                         output_rows=output_rows, diurnal_window=diurnal_window,
                                                         secondary=secondary)
    xgbr_pred_flat = xgbr_model.predict(X_test)

    # Compute output_dim dynamically
    sample_df = pd.read_csv(Path(data_dir, 'samples', sorted(os.listdir(Path(data_dir, 'samples')))[0]))
    output_dim = len(output_columns) * len(sample_df.iloc[output_rows:])

    ## Reshape y_pred to [num_samples, output_dim]
    xgbr_pred = xgbr_pred_flat.reshape(-1, output_dim)
    xgbr_target = y_test.reshape(-1, output_dim)

    # alternatives = [(transformer_preds, transformer_targets), (xgbr_pred, xgbr_target), (naive_preds, naive_targets),
    #                 (linear_preds, linear_targets), (seasonal_preds, seasonal_targets)]
    # labels = ["Transformer", "XGBRegressor", "Naive", "Linear", "Seasonal"]

    alternatives = [(xgbr_pred, xgbr_target), (naive_preds, naive_targets),
                    (linear_preds, linear_targets), (seasonal_preds, seasonal_targets)]
    labels = ["XGBRegressor", "Naive", "Linear", "Seasonal"]

    # alternatives = [(xgbr_pred, xgbr_target)]
    # labels = ["XGBRegressor"]

    visualizer(*alternatives, labels=labels, forecast_name=forecast_name, directory=data_dir, num_samples=200)

    #################################################################################################################
    ## Convert regression model outputs to classes based on thresholds for each output column, and
    ## evaluate success of regressors on classification problem

    reconstituted = []
    for preds, targets in alternatives:
        preds_original = reverse_normalize(preds, output_columns, Path('../data/input', "normalization.json"))
        targets_original = reverse_normalize(targets, output_columns, Path('../data/input', "normalization.json"))
        reconstituted.append((preds_original, targets_original))

    ## If working in real (not normalized) values:
    thresholds_df = pd.read_csv(Path('../data/input', "Limits.csv"), sep=';', decimal='.')

    ## Alternative for retaining normalized values:
    # raw_thresholds_df = pd.read_csv(Path('../data/input', "Limits.csv"), sep=';', decimal='.')
    # thresholds_df = apply_saved_normalize(raw_thresholds_df, param_file=Path('../data/input', "normalization.json"))

    # class_pred_target_pairs = []
    # class_results = []
    # for preds, targets in reconstituted:
    #     bin_preds = binarize_predictions(preds, output_columns=output_columns, thresholds_df=thresholds_df)
    #     bin_targets = binarize_predictions(targets, output_columns=output_columns, thresholds_df=thresholds_df)
    #     class_results.append((bin_preds, bin_targets))
    #
    # classification_visualizer(*class_results, labels=labels, directory=data_dir, forecast_name=forecast_name,
    #                           num_samples=200)

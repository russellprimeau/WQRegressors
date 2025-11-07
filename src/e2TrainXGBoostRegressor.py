import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import torch
from e1_TrainTransformer import load_samples

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")

    matplotlib.use('Agg')  # Non-interactive backend for file output to handle remote machine installation errors


    # Configuration parameters (same as original script)
    data_dir = "../data/output/regression/Kimtall12hr"
    forecast_name = "nowcast"
    model_name = "xgbregressor"
    input_columns = ['Pfl - Temp (C)', 'Pfl - Sp Cond (microS_cm)', 'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)',
                     'Pfl - fDOM (QSU)', 'Instantaneous atmospheric pressure (mBar)',
                     'Hourly average wind direction (°)_x', 'Hourly average wind direction (°)_y',
                     'Average wind speed (m/s)',
                     'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
                     'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                     'Precipitation (mm/hr)', 'Instantaneous temperature (°C)', 'Average humidity (% relative humidity)']
    output_columns = ['08-Kimtall 22°C']
    output_rows = -1
    input_row_1 = 0
    input_row_2 = 11
    input_rows = slice(input_row_1, input_row_2)
    random_state = 30
    test_size = 0.2

    ## If no previous training completed
    # # Load and split samples
    # samples = load_samples(os.path.join(data_dir, 'samples'), input_columns=input_columns,
    #                        output_columns=output_columns, input_rows=input_rows, output_rows=output_rows)
    # train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=random_state)

    ## Run training set used for transformer, to avoid data leakage in evaluation
    reloadset = Path(data_dir, "forecasts", forecast_name, "train_files.txt")
    with open(reloadset) as f:
        train_files = [line.strip() for line in f]

    train_samples = load_samples(os.path.join(data_dir, "samples"), input_columns=input_columns,
                                output_columns=output_columns,
                                input_rows=input_rows, output_rows=output_rows, file_list=train_files)

    # Prepare training and test data
    X_train = np.array([s[0].flatten() for s in train_samples])
    y_train = np.array([s[1].flatten()[0] for s in train_samples])

    # Initialize and train XGBoost Regressor
    model = xgb.XGBRegressor(tree_method='hist', objective='reg:squarederror', n_estimators=10000, max_depth=10,
                             subsample=0.5, colsample_bytree=0.8, learning_rate=0.01, n_jobs=1)
    # model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.join(data_dir, "forecasts", forecast_name, model_name), exist_ok=True)
    model.save_model(Path(data_dir, "forecasts", forecast_name, model_name, "xgboost_model.json"))

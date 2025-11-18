import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
from utils.training import write_config, splitter


if __name__ == "__main__":
    ## Configure execution space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")
    matplotlib.use('Agg')  # Non-interactive backend for file output to handle remote machine installation errors

    ##################################################################################################################
    ## Configure model and dataset (hyperparameters)

    ## Dataset selection
    data_dir = "../data/output/regression/Koliforms96Sparse"
    forecast_name = "nowcast"
    model_name = "xgbregressor"
    input_columns = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
                     'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                     'Pfl - fDOM (QSU)', "Wind speed, x (m/s)", "Wind speed, y (m/s)",
                     'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                     'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                     '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
                     'SCADA - Temperature (°C)']  # Default: all different-dimensioned profiler and weather params, no SCADA
    output_columns = ['09-Koliforme bakterier 37°C']
    output_rows = -1
    input_row_1 = 0
    input_row_2 = 95

    ## Test/train split parameters
    random_state = 35  # Random seed which deterministically sets the test/train split
    test_size = 0.2  # Fraction of samples saved for evaluation after training
    reuse_split = True  # Reuse a previous train/test split from .txt files in data_dir/forecasts_name/
    split_source = None  # Or optionally, specify a different directory with train/test split here
    split_type = 'random'  # 'temporal' takes the first test_size fraction of samples for train, remainder for test

    ## Training hyperparameters
    metric = 'rmse'
    tree_method = 'hist'
    objective = 'reg:squarederror'
    n_estimators = 1100
    max_depth = 5
    subsample = 0.2
    colsample_bytree = 0.8
    learning_rate = 0.01
    n_jobs = -1
    early_stopping_rounds = 10

    ## Generate additional model dimensions parametrically based on selection
    input_rows = slice(input_row_1, input_row_2)
    files = [f for f in os.listdir(Path(data_dir, 'samples')) if
             os.path.isfile(Path(data_dir, 'samples', f))]
    sample_df = pd.read_csv(Path(data_dir, 'samples', sorted(files)[0]))

    ## Encapsulate model parameters which can be used to configure other model types for the same data in a dictionary
    config = {
        'input_dim': len(input_columns),
        'output_dim': len(output_columns) * len(sample_df.iloc[output_rows:]),
        'seq_len': input_row_2 - input_row_1,
        'input_columns': input_columns,
        'input_row_1': input_row_1,
        'input_row_2': input_row_2,
        'output_columns': output_columns,
        'output_rows': output_rows,
    }

    ## Write the configuration dictionary to a .json file
    write_config(config, data_dir, forecast_name, model_name)

    ##################################################################################################################
    ## Load and split a dataset according to configuration settings
    train_samples, test_samples = splitter(data_dir, forecast_name, input_columns, input_rows, output_columns,
                                           output_rows, True, reuse_split, split_source, split_type,
                                           test_size, random_state)

    ## Flatten train and test sets (required for XGB structure)
    X_train = np.array([s[0].flatten() for s in train_samples])
    y_train = np.array([s[1].flatten()[0] for s in train_samples])
    X_test = np.array([s[0].flatten() for s in test_samples])
    y_test = np.array([s[1].flatten()[0] for s in test_samples])

    ## Initialize and train XGBoost Regressor
    model = xgb.XGBRegressor(tree_method=tree_method, objective=objective, n_estimators=n_estimators,
                             max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree,
                             learning_rate=learning_rate, n_jobs=n_jobs, early_stopping_rounds=early_stopping_rounds)

    ## Train model
    model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)],verbose=True)

    ## Save trained model
    os.makedirs(os.path.join(data_dir, "forecasts", forecast_name, model_name), exist_ok=True)
    model.save_model(Path(data_dir, "forecasts", forecast_name, model_name, "xgboost_model.json"))

    ## Get evaluation results
    results = model.evals_result()

    ## Plot training vs validation loss
    epochs = len(results['validation_0'][metric])
    plt.figure(figsize=(8, 5))
    plt.loglog(range(epochs), results['validation_0'][metric], label=f'Training Loss')
    plt.loglog(range(epochs), results['validation_1'][metric], label=f'Validation Loss')
    plt.xlabel('Boosting Rounds')
    plt.ylabel(metric)
    plt.grid(True, which="both", ls="--")
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(data_dir, "forecasts", forecast_name, model_name, "loss_plot.png"))
    plt.close()
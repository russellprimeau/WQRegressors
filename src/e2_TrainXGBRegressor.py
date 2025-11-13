import os
import json
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
    data_dir = "../data/output/regression/Koliforms96Full"
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
    input_rows = slice(input_row_1, input_row_2)
    random_state = 30
    test_size = 0.2

    # Generate additional model dimensions parametrically based on selection
    input_rows = slice(input_row_1, input_row_2)
    files = [f for f in os.listdir(Path(data_dir, 'samples')) if
             os.path.isfile(Path(data_dir, 'samples', f))]
    sample_df = pd.read_csv(Path(data_dir, 'samples', sorted(files)[0]))

    # Encapsulate model configuration in a dictionary
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

    # Write model configuration dictionary to file so it can be re-run and re-used for other model types
    filepath = Path(data_dir, 'forecasts', forecast_name, model_name, 'model_config.json')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f)

    # If no previous training completed
    # Load and split samples
    # samples = load_samples(os.path.join(data_dir, 'samples'), input_columns=input_columns,
    #                        output_columns=output_columns, input_rows=input_rows, output_rows=output_rows, fault_tolerant=True)
    # train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=random_state)
    # file1 = Path(data_dir, "forecasts", forecast_name, "train_files.txt")
    # with open(file1, "w") as f:
    #     f.writelines(f"{s[2]}\n" for s in train_samples)
    # file2 = Path(data_dir, "forecasts", forecast_name, "test_files.txt")
    # with open(file2, "w") as f:
    #     f.writelines(f"{s[2]}\n" for s in test_samples)

    ## Run training set used for transformer, to avoid data leakage in evaluation
    reloadset = Path(data_dir, "forecasts", forecast_name, "train_files.txt")
    with open(reloadset) as f:
        train_files = [line.strip() for line in f]

    train_samples = load_samples(os.path.join(data_dir, "samples"), input_columns=input_columns,
                                output_columns=output_columns,
                                input_rows=input_rows, output_rows=output_rows, file_list=train_files)

    reloadset = Path(data_dir, "forecasts", forecast_name, "test_files.txt")
    with open(reloadset) as f:
        test_files = [line.strip() for line in f]
    test_samples = load_samples(os.path.join(data_dir, "samples"), input_columns=input_columns,
                                output_columns=output_columns,
                                input_rows=input_rows, output_rows=output_rows, file_list=test_files,
                                fault_tolerant=True)

    # Prepare training and test data
    X_train = np.array([s[0].flatten() for s in train_samples])
    y_train = np.array([s[1].flatten()[0] for s in train_samples])
    X_test = np.array([s[0].flatten() for s in test_samples])
    y_test = np.array([s[1].flatten()[0] for s in test_samples])

    # Initialize and train XGBoost Regressor
    model = xgb.XGBRegressor(tree_method='hist', objective='reg:squarederror', n_estimators=2000, max_depth=5,
                             subsample=0.2, colsample_bytree=0.8, learning_rate=0.01, n_jobs=-1)
    # model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=True)

    # Save model
    os.makedirs(os.path.join(data_dir, "forecasts", forecast_name, model_name), exist_ok=True)
    model.save_model(Path(data_dir, "forecasts", forecast_name, model_name, "xgboost_model.json"))

    # Get evaluation results
    results = model.evals_result()

    # Plot training vs validation loss
    epochs = len(results['validation_0']['rmse'])
    plt.figure(figsize=(8, 5))
    plt.loglog(range(epochs), results['validation_0']['rmse'], label='Train RMSE')
    plt.loglog(range(epochs), results['validation_1']['rmse'], label='Validation RMSE')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('RMSE')
    plt.grid(True, which="both", ls="--")
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(data_dir, "forecasts", forecast_name, model_name, "loss_plot.png"))
    plt.close()


import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import torch
from e1_TrainTransformer import load_samples

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    matplotlib.use('Agg')  # Non-interactive backend

    # Configuration parameters
    data_dir = "../data/output/classification/Anomaly24hr"
    forecast_name = "nowcast"
    model_name = "xgbclassifier"
    input_columns = [
        'Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
                        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                        'Pfl - fDOM (QSU)', "Wind speed, x (m/s)", "Wind speed, y (m/s)",
                        'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                        'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                        '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)'
    ]
    output_columns = ['anomaly']
    output_rows = -1
    input_row_1 = 0
    input_row_2 = 23
    input_rows = slice(input_row_1, input_row_2)
    random_state = 32
    test_size = 0.22

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

    # ## Load training set used for existing model(to avoid data leakage)


    # Pre-process dataset
    samples = load_samples(os.path.join(data_dir, 'samples'), input_columns=input_columns,
                           output_columns=output_columns,
                           input_rows=input_rows, output_rows=output_rows)
    all_filenames = sorted([f for f in os.listdir(os.path.join(data_dir, 'samples')) if f.endswith(".csv")])
    train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=random_state)
    os.makedirs(os.path.join(data_dir, "forecasts", forecast_name), exist_ok=True)
    file1 = Path(data_dir, "forecasts", forecast_name, "train_files.txt")
    with open(file1, "w") as f:
        f.writelines(f"{s[2]}\n" for s in train_samples)
    file2 = Path(data_dir, "forecasts", forecast_name, "test_files.txt")
    with open(file2, "w") as f:
        f.writelines(f"{s[2]}\n" for s in test_samples)
    #
    # reloadset = Path(data_dir, "forecasts", forecast_name, "train_files.txt")
    # with open(reloadset) as f:
    #     train_files = [line.strip() for line in f]
    #
    # train_samples = load_samples(
    #     os.path.join(data_dir, "samples"),
    #     input_columns=input_columns,
    #     output_columns=output_columns,
    #     input_rows=input_rows,
    #     output_rows=output_rows,
    #     file_list=train_files
    # )

    # Prepare training data
    X_train = np.array([s[0].flatten() for s in train_samples])
    y_train = np.array([int(round(s[1].flatten()[0])) for s in train_samples])  # ensure 0/1 ints

    # Initialize and train XGBoost Classifier
    model = xgb.XGBClassifier(
        tree_method='hist',
        objective='binary:logistic',
        n_estimators=100000,
        max_depth=10,
        subsample=0.5,
        colsample_bytree=0.8,
        learning_rate=0.01,
        n_jobs=1,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Save model
    save_path = Path(data_dir, "forecasts", forecast_name, model_name)
    os.makedirs(save_path, exist_ok=True)
    model.save_model(save_path / "xgboost_model.json")

    print("Model training complete. Model saved to:", save_path)

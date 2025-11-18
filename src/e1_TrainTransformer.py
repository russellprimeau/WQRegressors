"""
Train a PyTorch Transformer Model for time series forecasting, and write the model weights to file.
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib
import torch
from torch.utils.data import DataLoader
from utils.transformer import train_model, TimeSeriesTransformer, TimeSeriesTargetDataset
from utils.training import write_config, splitter


if __name__ == '__main__':
    ## Configure execution space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    matplotlib.use('Agg')  # Non-interactive backend for file output to handle remote machine installation errors

    ##################################################################################################################
    ## Configure model and dataset (hyperparameters)

    ## Dataset selection
    data_dir = "../data/output/regression/Koliforms96Full"  # Parent directory of test/train sample folder
    historic = "../data/output/regression/Consolidated.csv"  # Path to file with baseline model input
    input_columns = ['Pfl - Water temperature (°C)', 'Pfl - Sp Cond (microS_cm)',
                        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)',
                        'Pfl - fDOM (QSU)', "Wind speed, x (m/s)", "Wind speed, y (m/s)",
                        'Maximum 3s wind gust (m/s)', "Atmospheric pressure (mBar)",
                        'Longwave (IR) radiation (W/m2)', 'Shortwave (solar) radiation (W/m2)',
                        '24hr precipitation total (mm)', 'Air temperature (°C)', 'Humidity (%)',
                     'SCADA - Temperature (°C)']  # Default: all different-dimensioned profiler and weather params, no SCADA
    output_columns = ['09-Koliforme bakterier 37°C']
    forecast_name = "nowcast"
    model_name = "transformer"
    output_rows = -1  # Default: -1 (increase value to increase forecast length, but decrease input_row_2 accordingly)
    input_row_1 = 0  # Default: 0
    input_row_2 = 95  # Default: len(sample) - abs(output_rows)

    ## Test/train split parameters
    random_state = 35  # Random seed which deterministically sets the test/train split
    test_size = 0.2  # Fraction of samples saved for evaluation after training
    reuse_split = True  # Reuse a previous train/test split from .txt files in data_dir/forecasts_name/
    split_source = None  # Or optionally, specify a different directory with train/test split here
    split_type = 'random'  # 'temporal' takes the first test_size fraction of samples for train, remainder for test

    ## Model hyperparameters
    model_dim = 128  # Model size
    num_heads = 4  # Parallel attention heads
    num_layers = 8  # Depth of NN
    dropout = 0.1  # % neurons to randomly remove each epoch to prevent overfitting (regularization)

    ## Training hyperparameters
    batch_size = 1  # Minibatch size. Smaller batches -> noisier, but escapes local minima quicker
    num_epochs = 100  # Training duration (excessive epochs can cause overfitting to training data)
    loss_threshold = 0.000001  # Threshold of acceptably small loss to terminate training early
    learning_rate = 1e-4  # Limit on parameter adjustment size per epoch
    patience = 10 # Limit on how many epochs can fail to improve loss on validation set before early stopping

    ## Generate additional model dimensions parametrically based on selection
    input_rows = slice(input_row_1, input_row_2)
    files = [f for f in os.listdir(Path(data_dir, 'samples')) if
             os.path.isfile(Path(data_dir, 'samples', f))]
    sample_df = pd.read_csv(Path(data_dir, 'samples', sorted(files)[0]))

    ## Encapsulate model parameters which can be used to configure other model types for the same data in a dictionary
    config = {
        'input_dim' : len(input_columns),
        'model_dim' : model_dim,
        'num_heads' : num_heads,
        'num_layers' : num_layers,
        'dropout' : dropout,
        'output_dim' : len(output_columns) * len(sample_df.iloc[output_rows:]),
        'seq_len' : input_row_2 - input_row_1,
        'input_columns' : input_columns,
        'input_row_1': input_row_1,
        'input_row_2': input_row_2,
        'output_columns' : output_columns,
        'output_rows' : output_rows,
    }

    ## Write model configuration dictionary to file so it can be re-run and re-used for other model types
    write_config(config, data_dir, forecast_name, model_name)

    ##################################################################################################################
    ## Load and split samples
    train_samples, test_samples = splitter(data_dir, forecast_name, input_columns, input_rows, output_columns,
                                           output_rows, False, reuse_split, split_source, split_type,
                                           test_size, random_state)

    ## Restructure samples for transformer
    train_dataset = TimeSeriesTargetDataset(train_samples)
    test_dataset = TimeSeriesTargetDataset(test_samples)
    trainloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    model = TimeSeriesTransformer(config).to(device)

    ## Train and evaluate model
    train_model(data_dir, model, forecast_name, trainloader, testloader, device, num_epochs, learning_rate, loss_threshold, patience)
    os.makedirs(os.path.join(data_dir, "forecasts", forecast_name, model_name), exist_ok=True)

    ## Save trained model to file
    torch.save(model.state_dict(), Path(data_dir, "forecasts", forecast_name, model_name, "transformer_model.pt"))
"""
Train a PyTorch Transformer Model for time series forecasting, and write the model weights to file.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def load_samples(directory, input_columns, output_columns, input_rows, output_rows, file_list=None):
    samples = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".csv"):
            continue
        if file_list is not None and filename not in file_list:
            continue  # Skip files not in the provided list
        df = pd.read_csv(os.path.join(directory, filename))
        if not set(input_columns + output_columns).issubset(df.columns):
            continue  # skip files with missing columns
        if len(df) < input_rows.stop:
            print(f"Sample {filename} skipped — not enough rows ({len(df)} < {input_rows.stop})")
            continue  # skip files without enough rows

        input_seq = df.iloc[input_rows, :][input_columns].values
        output_seq = df.iloc[output_rows:, :][output_columns].values

        if np.isnan(input_seq).any() or np.isnan(output_seq).any():
            print(f"Sample {filename} skipped - contains NaN values")
            continue  # skip invalid samples
        samples.append((input_seq, output_seq, filename))
    print("Samples loaded")
    return samples

def train_model(directory, model, forecast_name, dataloader, num_epochs=100, learning_rate=1e-3, loss_threshold=1e-3):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}")

        # Early stopping condition
        if loss_threshold is not None and avg_loss <= loss_threshold:
            print(f"Stopping early at epoch {epoch + 1} because loss reached {avg_loss:.6f}")
            break

        # Plotting loss vs. epochs on log-log scale
        plt.figure(figsize=(8, 6))
        x_vals = list(range(1, len(epoch_losses) + 1))
        y_vals = epoch_losses
        plt.loglog(x_vals, y_vals, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss vs. Epochs (Log-Log Scale)")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(directory, "forecasts", forecast_name, "loss_plot.png"))
        plt.close()

class TimeSeriesTransformer(nn.Module):
    def __init__(self, config):
        super(TimeSeriesTransformer, self).__init__()
        self.model_dim = config['model_dim']

        # Project input features to model dimension
        self.input_proj = nn.Linear(config['input_dim'], config['model_dim'])

        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, config['seq_len'], config['model_dim']))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['model_dim'],
            nhead=config['num_heads'],
            dropout=config['dropout'],
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        # Output projection to scalar
        self.output_proj = nn.Linear(config['model_dim'], config['output_dim'])

    def forward(self, x):
        """
        x: [batch_size, seq_len=24, input_dim=28]
        returns: [batch_size, output_dim=1]
        """
        batch_size, seq_len, _ = x.size()

        # Project input features
        x = self.input_proj(x)  # [batch_size, seq_len, model_dim]

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        # Encode
        encoded = self.transformer_encoder(x)

        # Use last timestep's encoding
        last_encoding = encoded[:, -1, :]  # [batch_size, model_dim]

        # Project to output
        output = self.output_proj(last_encoding)  # [batch_size, 1]

        return output

class TimeSeriesTargetDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq, filename = self.samples[idx]
        x = torch.tensor(input_seq, dtype=torch.float32)
        y = torch.tensor(target_seq, dtype=torch.float32).flatten()
        return x, y, filename


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")

    matplotlib.use('Agg')  # Non-interactive backend for file output to handle remote machine installation errors

    ##################################################################################################################
    # Configure input, output and model hyperparameters
    all_columns = ['TIMESTAMP', 'Segment', 'Interpolated', 'Pfl - Temp (C)', 'Pfl - Sp Cond (microS_cm)',
        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)', 'Pfl - fDOM (QSU)',
        'Instantaneous atmospheric pressure (mBar)', 'Wind direction 10minRollingAvg (°)_x',
        'Wind direction 10minRollingAvg (°)_y', 'Hourly average wind direction (°)_x',
        'Hourly average wind direction (°)_y', 'Average wind speed (m/s)',
        'Maximum sustained wind speed, 3-second span (m/s)', 'Time of maximum 3s Gust',
        'Maximum sustained wind speed, 10-minute span (m/s)', 'Time of maximum 10 minute gust',
        'Hourly average atmospheric pressure at station (mBar)', 'Maximum pressure differential, 3-hour span (mBar)',
        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
        'Longwave (IR) radiation (W/m2)', 'Instantaneous sea-level atmospheric pressure (mBar)',
        'Shortwave (solar) radiation (W/m2)', 'Precipitation (mm/hr)', 'Instantaneous temperature (°C)',
        'Maximum temperature (°C)', 'Minimum temperature (°C)', 'Average humidity (% relative humidity)',
        'SCADA - pH', 'SCADA - Temperature (°C)', '01-Farge', '04-Turbiditet', '06-E.coli',
        '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen', '24-Bly',
        '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)', '44-pH, surhetsgrad']
    data_columns = ['Pfl - Temp (C)', 'Pfl - Sp Cond (microS_cm)',
        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)', 'Pfl - fDOM (QSU)',
        'Instantaneous atmospheric pressure (mBar)', 'Wind direction 10minRollingAvg (°)_x',
        'Wind direction 10minRollingAvg (°)_y', 'Hourly average wind direction (°)_x',
        'Hourly average wind direction (°)_y', 'Average wind speed (m/s)',
        'Maximum sustained wind speed, 3-second span (m/s)', 'Time of maximum 3s Gust',
        'Maximum sustained wind speed, 10-minute span (m/s)', 'Time of maximum 10 minute gust',
        'Hourly average atmospheric pressure at station (mBar)', 'Maximum pressure differential, 3-hour span (mBar)',
        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
        'Longwave (IR) radiation (W/m2)', 'Instantaneous sea-level atmospheric pressure (mBar)',
        'Shortwave (solar) radiation (W/m2)', 'Precipitation (mm/hr)', 'Instantaneous temperature (°C)',
        'Maximum temperature (°C)', 'Minimum temperature (°C)', 'Average humidity (% relative humidity)',
        'SCADA - pH', 'SCADA - Temperature (°C)', '01-Farge', '04-Turbiditet', '06-E.coli',
        '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen', '24-Bly',
        '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)', '44-pH, surhetsgrad']

    data_dir = "../data/output/regression/Kimtall12hr"  # Parent directory of test/train sample folder
    historic = "../data/output/regression/Consolidated.csv"  # Path to file with baseline model input
    input_columns = ['Pfl - Temp (C)',
        'Pfl - Sp Cond (microS_cm)',
        'Pfl - pH',
        'Pfl - DO (% Sat)',
        'Pfl - Turbidity (FNU)',
        'Pfl - fDOM (QSU)',
        'Instantaneous atmospheric pressure (mBar)',
        'Hourly average wind direction (°)_x',
        'Hourly average wind direction (°)_y',
        'Average wind speed (m/s)',
        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
        'Longwave (IR) radiation (W/m2)',
        'Shortwave (solar) radiation (W/m2)',
        'Precipitation (mm/hr)',
        'Instantaneous temperature (°C)',
        'Average humidity (% relative humidity)'
                     ]  # Default: all different-dimensioned profiler and weather params, no SCADA
    output_columns = ['08-Kimtall 22°C']
    forecast_name = "nowcast"
    output_rows = -1  # Default: -1 (increase value to increase forecast length, but decrease input_row_2 accordingly)
    input_row_1 = 0  # Default: 0
    input_row_2 = 11  # Default: len(sample) - abs(output_rows)

    # Model hyperparameters
    model_dim = 128  # Model size
    num_heads = 4  # Parallel attention heads
    num_layers = 8  # Depth of NN
    dropout = 0.1  # % neurons to randomly remove each epoch to prevent overfitting (regularization)

    # Training hyperparameters
    random_state = 35  # Random seed which deterministically sets the test/train split
    test_size = 0.2  # Fraction of samples saved for evaluation after training
    batch_size = 1  # Minibatch size. Smaller batches -> noisier, but escapes local minima quicker
    num_epochs = 12  # Training duration (excessive epochs can cause overfitting to training data)
    loss_threshold = 0.000001  # Threshold of acceptably small loss to terminate training early
    learning_rate = 1e-4  # Limit on parameter adjustment size per epoch

    # Generate additional model dimensions parametrically based on selection
    input_rows = slice(input_row_1, input_row_2)
    files = [f for f in os.listdir(Path(data_dir, 'samples')) if
             os.path.isfile(Path(data_dir, 'samples', f))]
    sample_df = pd.read_csv(Path(data_dir, 'samples', sorted(files)[0]))

    # Encapsulate model configuration in a dictionary
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

    # Write model configuration dictionary to file so it can be re-run and re-used for other model types
    with open(Path(data_dir, 'forecasts', forecast_name, 'model_config.json'), 'w') as f:
        json.dump(config, f)

    ##################################################################################################################
    # Pre-process dataset
    samples = load_samples(os.path.join(data_dir,'samples'), input_columns=input_columns, output_columns=output_columns,
                                          input_rows=input_rows, output_rows=output_rows)
    all_filenames = sorted([f for f in os.listdir(os.path.join(data_dir,'samples')) if f.endswith(".csv")])
    train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=random_state)
    file1 = Path(data_dir, "forecasts", forecast_name, "train_files.txt")
    with open(file1, "w") as f:
        f.writelines(f"{s[2]}\n" for s in train_samples)
    file2 = Path(data_dir, "forecasts", forecast_name, "test_files.txt")
    with open(file2, "w") as f:
        f.writelines(f"{s[2]}\n" for s in test_samples)

    ##################################################################################################################
    ## Train transformer model on 'train' portion of dataset
    train_dataset = TimeSeriesTargetDataset(train_samples)
    dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    # model = TimeSeriesTransformer(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers,
    #                               dropout=dropout, output_dim=output_dim, seq_len=seq_len).to(device)
    model = TimeSeriesTransformer(config).to(device)

    train_model(data_dir, model, forecast_name, dataloader, num_epochs, learning_rate, loss_threshold)
    os.makedirs(os.path.join(data_dir, "forecasts", forecast_name, "transformer"), exist_ok=True)
    torch.save(model.state_dict(), Path(data_dir, "forecasts", forecast_name, "transformer", "transformer_model.pt"))
"""
Time Series Forecasting using Transformer Model in PyTorch.
"""


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

        input_seq = df.loc[input_rows, input_columns].values  # shape: [timesteps, input_dim]
        output_seq = df.iloc[output_rows:, :][output_columns].values

        if np.isnan(input_seq).any() or np.isnan(output_seq).any():
            print(f"Sample {filename} skipped - contains NaN values")
            continue  # skip invalid samples
        samples.append((input_seq, output_seq, filename))
    print("Samples loaded")
    return samples

def train_model(model, dataloader, num_epochs=100, learning_rate=1e-3, loss_threshold=1e-3):

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
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
        plt.savefig("../data/output/for_regression/loss_plot.png")
        plt.close()

def evaluate_model(model, dataset):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y, filename = dataset[i]
            x = x.unsqueeze(0).to(device)  # Add batch dimension
            pred = model(x).squeeze().item()
            true = y.item()
            # print(f"{filename}: Prediction = {pred:.4f}, Target = {true:.4f}")
            predictions.append(pred)
            targets.append(true)

    # Convert to NumPy arrays and filter out NaN/inf
    predictions = np.array(predictions)
    targets = np.array(targets)
    mask = np.isfinite(predictions) & np.isfinite(targets)
    predictions = predictions[mask]
    targets = targets[mask]
    return predictions, targets

def evaluate_baseline(dataset, historical_df, output_columns, data_dir, output_rows=-1):
    predictions, targets = [], []
    for i in range(len(dataset)):
        _, y, filename = dataset[i]
        sample_df = pd.read_csv(os.path.join(data_dir, filename), parse_dates=["TIMESTAMP"])
        sample_time = sample_df["TIMESTAMP"].iloc[output_rows]
        earlier_values = historical_df[historical_df["TIMESTAMP"] < sample_time][output_columns]
        # Drop NaN values before selecting the last one
        valid_values = earlier_values.dropna()
        baseline_pred = valid_values.iloc[-1].values if not valid_values.empty else np.full(len(output_columns), np.nan)
        predictions.append(baseline_pred)
        targets.append(y.item())
        print("Baseline: ", sample_time," in ",filename, ". Naive prediction: ", baseline_pred, "Ground truth: ", y.item())
    return np.array(predictions), np.array(targets)

def visualizer(*pred_target_pairs, labels=None, num_samples=100):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = sns.color_palette("husl", len(pred_target_pairs))
    min_val, max_val = float("inf"), float("-inf")

    # Compute and print statistics
    for i, (preds, targets) in enumerate(pred_target_pairs):
        preds = preds[:min(len(preds),num_samples)]
        targets = targets[:min(len(targets),num_samples)]
        label = labels[i] if labels else f"Model {i+1}"
        mae = mean_absolute_error(targets, preds)
        rmse = mean_squared_error(targets, preds)
        r2 = r2_score(targets, preds)
        print(f"{label} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        # Plot predictions
        ax.scatter(targets, preds, label=f"{label}", alpha=0.7, color=colors[i])
        min_val = min(min_val, targets.min(), preds.min())
        max_val = max(max_val, targets.max(), preds.max())

    # Diagonal reference line
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Predicted vs Actual Values")
    ax.set_xlim(0.95 * min_val, 1.05 * max_val)
    ax.set_ylim(0.95 * min_val, 1.05 * max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../data/output/for_regression/predictions.png")
    # plt.show()  # Tk/Tcl issues preventing interactivity

def normalize_columns(df, columns, min=0, max=1):
    """
    Normalize specified columns in a DataFrame to a given range.

    Parameters:
    - df: pandas.DataFrame
    - columns: list of column names to normalize
    - target_range: tuple (min, max) for the normalization range

    Returns:
    - A copy of the DataFrame with normalized columns.
    """
    df_normalized = df.copy()
    min_val, max_val = min, max
    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max != col_min:
            df_normalized[col] = ((df[col] - col_min) / (col_max - col_min)) * (max_val - min_val) + min_val
        else:
            # If all values are the same, set them to the midpoint of the target range
            df_normalized[col] = (min_val + max_val) / 2

    return df_normalized


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=4, dropout=0.1, output_dim=1, seq_len=72):
        super(TimeSeriesTransformer, self).__init__()
        self.model_dim = model_dim

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, model_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # <-- Add this
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to scalar
        self.output_proj = nn.Linear(model_dim, output_dim)

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

    matplotlib.use('Agg')  # Non-interactive backend for file output

    ## Configure input, output and model hyperparameters
    all_columns = ['TIMESTAMP', 'Segment', 'Interpolated', 'Pfl - Temp (C)', 'Pfl - Sp Cond (microS_cm)',
        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)', 'Pfl - fDOM (QSU)',
        'Instantaneous atmospheric pressure (mBar)', 'Wind direction 10minRollingAvg (°)',
        'Hourly average wind direction (°)', 'Average wind speed (m/s)',
        'Maximum sustained wind speed, 3-second span (m/s)', 'Time of maximum 3s Gust',
        'Maximum sustained wind speed, 10-minute span (m/s)', 'Time of maximum 10 minute gust',
        'Hourly average atmospheric pressure at station (mBar)', 'Maximum pressure differential, 3-hour span (mBar)',
        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
        'Longwave (IR) radiation (W/m2)', 'Instantaneous sea-level atmospheric pressure (mBar)',
        'Shortwave (solar) radiation (W/m2)', 'Precipitation (mm/hr)', 'Instantaneous temperature (°C)',
        'Maximum temperature (°C)', 'Minimum temperature (°C)', 'Average humidity (% relative humidity)',
        'SCADA - pH', 'SCADA - Temperature (°C)', '06-E.coli', '08-Kimtall 22°C', '21-Arsen', '24-Bly',
        '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']

    data_columns = ['Pfl - Temp (C)', 'Pfl - Sp Cond (microS_cm)',
        'Pfl - pH', 'Pfl - DO (% Sat)', 'Pfl - Turbidity (FNU)', 'Pfl - fDOM (RFU)', 'Pfl - fDOM (QSU)',
        'Instantaneous atmospheric pressure (mBar)', 'Wind direction 10minRollingAvg (°)',
        'Hourly average wind direction (°)', 'Average wind speed (m/s)',
        'Maximum sustained wind speed, 3-second span (m/s)', 'Time of maximum 3s Gust',
        'Maximum sustained wind speed, 10-minute span (m/s)', 'Time of maximum 10 minute gust',
        'Hourly average atmospheric pressure at station (mBar)', 'Maximum pressure differential, 3-hour span (mBar)',
        'Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)',
        'Longwave (IR) radiation (W/m2)', 'Instantaneous sea-level atmospheric pressure (mBar)',
        'Shortwave (solar) radiation (W/m2)', 'Precipitation (mm/hr)', 'Instantaneous temperature (°C)',
        'Maximum temperature (°C)', 'Minimum temperature (°C)', 'Average humidity (% relative humidity)',
        'SCADA - pH', 'SCADA - Temperature (°C)', '06-E.coli', '08-Kimtall 22°C', '21-Arsen', '24-Bly',
        '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']

    data_dir = "../data/output/for_regression/SCADATemp96hr"
    input_columns = ['Pfl - Temp (C)', 'SCADA - Temperature (°C)']
    output_columns = ['SCADA - Temperature (°C)']
    input_rows = slice(0, 96)
    output_rows = -1
    random_state = 40  # Random seed which deterministically sets the test/train split
    test_size = 0.15  # Fraction of samples saved for evaluation after training
    batch_size = 32  # Minibatch size. Smaller batches -> noisier, but escapes local minima quicker
    num_epochs = 200  # Training duration (excessive epochs can cause overfitting to training data)
    loss_threshold = 0.00001  # Threshold of acceptably small loss to terminate training early
    learning_rate = 1e-4  # Limit on parameter adjustment size per epoch
    model_dim = 128  # Model size
    num_heads = 4  # Parallel attention heads
    num_layers = 4  # Depth of NN
    dropout = 0.2  # Regularization technique to prevent overtraining by randomly removing some neurons each epoch

    # Generate parameters from selection
    input_dim = len(input_columns)
    sample_df = pd.read_csv(os.path.join(data_dir, sorted(os.listdir(data_dir))[0]))
    output_dim = len(output_columns) * len(sample_df.iloc[output_rows:])
    seq_len = input_rows.stop - input_rows.start  #

    # Pre-process dataset
    samples = load_samples(data_dir, input_columns=input_columns, output_columns=output_columns,
                                          input_rows=input_rows, output_rows=output_rows)
    all_filenames = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=random_state)
    with open("../data/output/for_regression/train_files.txt", "w") as f:
        f.writelines(f"{s[2]}\n" for s in train_samples)
    with open("../data/output/for_regression/test_files.txt", "w") as f:
        f.writelines(f"{s[2]}\n" for s in test_samples)

    ## Train
    train_dataset = TimeSeriesTargetDataset(train_samples)
    dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    model = TimeSeriesTransformer(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers,
                                  dropout=dropout, output_dim=output_dim, seq_len=seq_len).to(device)
    train_model(model, dataloader, num_epochs, learning_rate, loss_threshold)
    torch.save(model.state_dict(), "../data/output/for_regression/transformer_model.pt")

    ## Post-processing
    model = TimeSeriesTransformer(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers,
                                  dropout=dropout, output_dim=output_dim, seq_len=seq_len).to(device)
    model.load_state_dict(torch.load("../data/output/for_regression/transformer_model.pt", map_location=device))
    model.eval()  # Set to evaluation mode

    with open("../data/output/for_regression/test_files.txt") as f:
        test_files = [line.strip() for line in f]
    test_samples = load_samples(data_dir,input_columns=input_columns,output_columns=output_columns,
        input_rows=input_rows, output_rows=output_rows, file_list=test_files)
    test_dataset = TimeSeriesTargetDataset(test_samples)

    model_preds, targets = evaluate_model(model, test_dataset)

    historic_df = pd.read_csv("../data/output/for_regression/Combined_Cleaned.csv",
                              parse_dates=["TIMESTAMP"])
    sort_df = historic_df.sort_values("TIMESTAMP")
    norm_df = normalize_columns(sort_df, data_columns)

    baseline_preds, targets_baseline = evaluate_baseline(test_dataset, norm_df, output_columns, data_dir,
                                                         output_rows=output_rows)

    visualizer((model_preds, targets), (baseline_preds, targets_baseline),
                     labels=["Transformer", "Baseline"], num_samples=200)
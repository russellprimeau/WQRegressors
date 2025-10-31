'''
Time Series Forecasting using Transformer Model in PyTorch.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Transformer model definition
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=3, output_window=48, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_dim = model_dim
        self.output_window = output_window
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 168, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(model_dim, input_dim * output_window)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = x.transpose(0, 1)
        encoded = self.transformer_encoder(x)
        encoded = encoded.transpose(0, 1)
        last_encoding = encoded[:, -1, :]
        output = self.output_proj(last_encoding)
        output = output.view(batch_size, self.output_window, self.input_dim)
        return output

# Custom dataset for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window=168, output_window=48):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.samples = []
        self.prepare_samples()

    def prepare_samples(self):
        total_length = len(self.data)
        for i in range(total_length - self.input_window - self.output_window):
            input_seq = self.data[i:i+self.input_window]
            output_seq = self.data[i+self.input_window:i+self.input_window+self.output_window]
            self.samples.append((input_seq, output_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, output_seq = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

# Load data from CSV
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    data = df.values  # shape: [num_timesteps, num_features]
    return data

# Training loop
def train_model(model, dataloader, num_epochs=10, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

# Visualize predictions
def visualize_predictions(model, dataset, num_samples=3):
    model.eval()
    for i in range(num_samples):
        inputs, targets = dataset[i]
        inputs = inputs.unsqueeze(0)
        with torch.no_grad():
            predictions = model(inputs).squeeze(0).numpy()
        targets = targets.numpy()
        for feature in range(inputs.shape[2]):
            plt.figure(figsize=(10, 4))
            sns.lineplot(x=np.arange(48), y=targets[:, feature], label='Target')
            sns.lineplot(x=np.arange(48), y=predictions[:, feature], label='Prediction')
            plt.title(f'Sample {i+1} - Feature {feature+1}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'prediction_sample_{i+1}_feature_{feature+1}.png')
            plt.close()

# Main pipeline
def main():
    csv_file = '../data/output/for_regression/Combined_Cleaned.csv'  # Replace with actual file name
    data = load_data(csv_file)
    dataset = TimeSeriesDataset(data, input_window=168, output_window=48)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = data.shape[1]
    model = TimeSeriesTransformer(input_dim=input_dim)

    train_model(model, dataloader, num_epochs=10)
    visualize_predictions(model, dataset)

main()
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


def train_model(directory, model, forecast_name, trainloader, testloader, device, num_epochs=100, learning_rate=1e-3,
                loss_threshold=1e-3, patience=5):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets, _ in trainloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(trainloader)
        train_losses.append(avg_loss)

        # Validation (for early stopping condition)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets, _ in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(testloader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss / len(trainloader):.6f}, Validation Loss: {val_loss / len(testloader):.6f}")

        # Early stopping condition
        if loss_threshold is not None and avg_loss <= loss_threshold:
            print(f"Stopping early at epoch {epoch + 1} because loss reached {avg_loss:.6f}")
            break

        # Early stopping checks
        if loss_threshold and avg_loss <= loss_threshold:
            print("Stopping early due to training loss threshold.")
            break
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Stopping early due to validation loss plateau.")
                break

        # Plotting loss vs. epochs on log-log scale
        plt.figure(figsize=(8, 6))
        x_vals = list(range(1, len(train_losses) + 1))
        plt.loglog(x_vals, train_losses, marker='o', label='Training Loss')
        plt.loglog(x_vals, val_losses, marker='s', label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss vs. Epochs (Log-Log Scale)")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        filepath = Path(directory, "forecasts", forecast_name, "transformer")
        os.makedirs(filepath, exist_ok=True)
        plt.savefig(Path(directory, "forecasts", forecast_name, "transformer", "loss_plot.png"))
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
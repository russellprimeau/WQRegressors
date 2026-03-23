import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


def _compact_stop_reason_for_plot(stop_reason_text: str) -> str:
    """Shorten stop annotations for plotting without dropping numeric details."""
    compact_text = " ".join(str(stop_reason_text).split())
    replacements = (
        ("Early stopping:", "Early stop:"),
        ("Scheduled stop:", "Scheduled:"),
        ("all configured epochs exhausted", "epochs exhausted"),
        ("round(s)", "rounds"),
        ("epoch(s)", "epochs"),
    )
    for old, new in replacements:
        compact_text = compact_text.replace(old, new)
    return compact_text


def _batch_pearson_corr(y_pred, y_true, eps=1e-8, clip=True):
    """Compute mean Pearson correlation across flattened output dimensions."""
    pred = y_pred.reshape(y_pred.shape[0], -1)
    target = y_true.reshape(y_true.shape[0], -1)

    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    numerator = torch.sum(pred_centered * target_centered, dim=0)
    pred_norm = torch.sqrt(torch.sum(pred_centered ** 2, dim=0) + eps)
    target_norm = torch.sqrt(torch.sum(target_centered ** 2, dim=0) + eps)
    corr = numerator / (pred_norm * target_norm + eps)

    if clip:
        corr = torch.clamp(corr, -1.0, 1.0)

    return torch.mean(corr)


def train_model(directory, model, forecast_name, trainloader, testloader, device, num_epochs=100, learning_rate=1e-3,
                loss_threshold=1e-3, patience=5, model_subdir='transformer',
                corr_lambda=0.1, corr_eps=1e-8, corr_clip=True,
                max_epochs_override=None, skip_plot=False):
    """Train transformer model.

    Parameters
    ----------
    max_epochs_override : int or None
        When set, train for exactly this many epochs with no patience-based
        early stopping on the validation set.  The loss_threshold stop rule
        on training combined loss is still honoured.  This is used when the
        epoch budget has been estimated externally (e.g. via internal CV) to
        avoid leaking test-set information into the training process.
    """

    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    train_mse_losses = []
    train_corr_terms = []
    train_combined_losses = []
    val_mse_losses = []
    val_corr_terms = []
    val_combined_losses = []
    best_val_combined = float('inf')
    best_val_epoch = None
    patience_counter = 0
    stop_reason_code = None
    stop_reason_text = None
    stop_epoch = None
    use_fixed_budget = max_epochs_override is not None
    effective_num_epochs = max_epochs_override if use_fixed_budget else num_epochs

    for epoch in range(effective_num_epochs):
        model.train()
        epoch_train_mse = 0.0
        epoch_train_corr = 0.0
        epoch_train_combined = 0.0
        for inputs, targets, _ in trainloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            mse_loss = mse_criterion(outputs, targets)
            corr_term = _batch_pearson_corr(outputs, targets, eps=corr_eps, clip=corr_clip)
            combined_loss = mse_loss - float(corr_lambda) * corr_term
            combined_loss.backward()
            optimizer.step()

            epoch_train_mse += mse_loss.item()
            epoch_train_corr += corr_term.item()
            epoch_train_combined += combined_loss.item()

        avg_train_mse = epoch_train_mse / len(trainloader)
        avg_train_corr = epoch_train_corr / len(trainloader)
        avg_train_combined = epoch_train_combined / len(trainloader)
        train_mse_losses.append(avg_train_mse)
        train_corr_terms.append(avg_train_corr)
        train_combined_losses.append(avg_train_combined)

        # Validation (for early stopping condition)
        model.eval()
        epoch_val_mse = 0.0
        epoch_val_corr = 0.0
        epoch_val_combined = 0.0
        with torch.no_grad():
            for inputs, targets, _ in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                mse_loss = mse_criterion(outputs, targets)
                corr_term = _batch_pearson_corr(outputs, targets, eps=corr_eps, clip=corr_clip)
                combined_loss = mse_loss - float(corr_lambda) * corr_term

                epoch_val_mse += mse_loss.item()
                epoch_val_corr += corr_term.item()
                epoch_val_combined += combined_loss.item()

        avg_val_mse = epoch_val_mse / len(testloader)
        avg_val_corr = epoch_val_corr / len(testloader)
        avg_val_combined = epoch_val_combined / len(testloader)
        val_mse_losses.append(avg_val_mse)
        val_corr_terms.append(avg_val_corr)
        val_combined_losses.append(avg_val_combined)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_mse={avg_train_mse:.6f}, train_corr={avg_train_corr:.6f}, train_combined={avg_train_combined:.6f} | "
            f"val_mse={avg_val_mse:.6f}, val_corr={avg_val_corr:.6f}, val_combined={avg_val_combined:.6f}"
        )

        # Early stopping condition
        if loss_threshold is not None and avg_train_combined <= loss_threshold:
            stop_reason_code = "train_combined_threshold_reached"
            stop_epoch = int(epoch + 1)
            stop_reason_text = (
                "Early stopping: train_combined <= loss_threshold "
                f"({avg_train_combined:.6g} <= {float(loss_threshold):.6g}) "
                f"at epoch {stop_epoch}/{int(num_epochs)}."
            )
            print(
                f"Stopping early at epoch {epoch + 1} because combined loss reached "
                f"{avg_train_combined:.6f}"
            )
            break

        if avg_val_combined < best_val_combined:
            best_val_combined = avg_val_combined
            best_val_epoch = int(epoch + 1)
            patience_counter = 0
        elif not use_fixed_budget:
            patience_counter += 1
            if patience_counter >= patience:
                stop_reason_code = "validation_combined_patience_exhausted"
                stop_epoch = int(epoch + 1)
                stop_reason_text = (
                    "Early stopping: no decrease in best validation combined loss "
                    f"for {int(patience)} consecutive epoch(s) "
                    f"(best={best_val_combined:.6g} at epoch {int(best_val_epoch or 1)}) "
                    f"at epoch {stop_epoch}/{int(num_epochs)}."
                )
                print("Stopping early due to validation combined-loss plateau.")
                break

    if stop_reason_code is None:
        stop_reason_code = "cv_epoch_budget_exhausted" if use_fixed_budget else "max_epochs_exhausted"
        stop_epoch = int(len(train_combined_losses))
        stop_reason_text = (
            f"Scheduled stop: {'CV epoch budget' if use_fixed_budget else 'all configured epochs'} exhausted "
            f"({int(effective_num_epochs)}/{int(effective_num_epochs)} epochs)."
        )

    # Plot all objective terms for train/validation to inspect optimization behavior.
    if not skip_plot:
        filepath = Path(directory, "forecasts", forecast_name, model_subdir)
        os.makedirs(filepath, exist_ok=True)
        plt.figure(figsize=(8, 6))
        x_vals = list(range(1, len(train_combined_losses) + 1))
        plt.plot(x_vals, train_combined_losses, marker='o', label='Train Combined')
        plt.plot(x_vals, val_combined_losses, marker='s', label='Val Combined')
        plt.plot(x_vals, train_mse_losses, marker='o', linestyle='--', label='Train MSE')
        plt.plot(x_vals, val_mse_losses, marker='s', linestyle='--', label='Val MSE')
        plt.plot(x_vals, train_corr_terms, marker='o', linestyle=':', label='Train Corr')
        plt.plot(x_vals, val_corr_terms, marker='s', linestyle=':', label='Val Corr')
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True, ls="--")
        plt.legend()
        plt.gcf().text(
            0.01,
            0.99,
            _compact_stop_reason_for_plot(stop_reason_text),
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#666666", "boxstyle": "round,pad=0.15"},
        )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
        plt.savefig(filepath / "loss_plot.png")
        plt.close()

    return {
        "model_type": "transformer",
        "stop_reason_code": str(stop_reason_code),
        "stop_reason_text": str(stop_reason_text),
        "stop_epoch": int(stop_epoch),
        "stopped_early": bool(str(stop_reason_code) != "max_epochs_exhausted"),
        "configured": {
            "num_epochs": int(num_epochs),
            "loss_threshold": None if loss_threshold is None else float(loss_threshold),
            "patience": int(patience),
        },
        "observed": {
            "best_val_combined": None if not val_combined_losses else float(min(val_combined_losses)),
            "best_val_epoch": None if best_val_epoch is None else int(best_val_epoch),
            "final_train_combined": None if not train_combined_losses else float(train_combined_losses[-1]),
            "final_val_combined": None if not val_combined_losses else float(val_combined_losses[-1]),
            "epochs_executed": int(len(train_combined_losses)),
        },
    }


class TimeSeriesLSTM(nn.Module):
    def __init__(self, config):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=config['input_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'] if config['num_layers'] > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )
        self.output_proj = nn.Linear(config['hidden_dim'], config['output_dim'])

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]
        output = self.output_proj(last_timestep)
        return output

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
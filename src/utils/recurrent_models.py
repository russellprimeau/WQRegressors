"""
Recurrent forecasting models: RecurrentTransformerModel and LSTMForecasterModel.

Both models take a sequence of feature vectors [B, T, F] and predict a scalar target:
the final-row value of the state column. The state column is zeroed in the last input row
by RecurrentTimeSeriesDataset to prevent data leakage.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RecurrentTimeSeriesDataset(Dataset):
    """Dataset for recurrent models.

    Mirrors TimeSeriesTargetDataset but:
    - Zeros the state column in the last row of each sample (leakage guard).
    - Uses that zeroed-out value as the regression target y.

    Parameters
    ----------
    samples : list of (input_seq, target_seq, filename)
        input_seq : ndarray [T, F]
        target_seq : ndarray [1, 1] or [1] — the state value at the last row,
                     as returned by load_samples with output_rows=[T-1].
        filename : str
    state_col_idx : int
        Index of the state column within input_seq's feature axis.
    """

    def __init__(self, samples, state_col_idx: int):
        self.samples = samples
        self.state_col_idx = state_col_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq, filename = self.samples[idx]
        x = torch.tensor(input_seq, dtype=torch.float32)          # [T, F]
        y = torch.tensor(target_seq, dtype=torch.float32).flatten()  # [1]

        # Zero the state column in the last row to prevent the model from
        # reading the answer it is supposed to predict.
        x = x.clone()
        x[-1, self.state_col_idx] = 0.0

        return x, y, filename


# ---------------------------------------------------------------------------
# Model: Recurrent Transformer
# ---------------------------------------------------------------------------

class RecurrentTransformerModel(nn.Module):
    """Transformer encoder that reads a full feature sequence and predicts a scalar.

    Config keys
    -----------
    input_dim : int   — number of input features F
    model_dim : int   — internal embedding dimension
    num_heads : int   — attention heads (must divide model_dim)
    num_layers : int  — number of encoder layers
    dropout : float
    seq_len : int     — sequence length T (used for positional embedding)
    output_dim : int  — output dimension (default 1)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.input_proj = nn.Linear(config["input_dim"], config["model_dim"])
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config["seq_len"], config["model_dim"])
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["model_dim"],
            nhead=config["num_heads"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config["num_layers"]
        )
        self.output_proj = nn.Linear(config["model_dim"], config.get("output_dim", 1))

    def forward(self, x):
        """x: [B, T, F] -> [B, 1]"""
        batch_size, seq_len, _ = x.size()
        x = self.input_proj(x)                           # [B, T, model_dim]
        x = x + self.pos_embedding[:, :seq_len, :]       # add positional encoding
        # Causal mask: each position attends only to itself and earlier positions.
        # This breaks the uniform-attention fixed point that causes mean-collapse.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        encoded = self.transformer_encoder(x, mask=causal_mask)  # [B, T, model_dim]
        last_encoding = encoded[:, -1, :]                # [B, model_dim]
        return self.output_proj(last_encoding)           # [B, 1]


# ---------------------------------------------------------------------------
# Model: LSTM Forecaster
# ---------------------------------------------------------------------------

class LSTMForecasterModel(nn.Module):
    """LSTM that reads a full feature sequence and predicts a scalar.

    Config keys
    -----------
    input_dim : int   — number of input features F
    hidden_dim : int  — LSTM hidden state size
    num_layers : int  — number of stacked LSTM layers
    dropout : float   — applied between LSTM layers (ignored if num_layers == 1)
    output_dim : int  — output dimension (default 1)
    """

    def __init__(self, config: dict):
        super().__init__()
        lstm_dropout = config["dropout"] if config["num_layers"] > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config["input_dim"],
            hidden_size=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(config["hidden_dim"], config.get("output_dim", 1))

    def forward(self, x):
        """x: [B, T, F] -> [B, 1]"""
        _, (h_n, _) = self.lstm(x)   # h_n: [num_layers, B, hidden_dim]
        last_hidden = h_n[-1]        # [B, hidden_dim]  (top layer)
        return self.output_proj(last_hidden)  # [B, 1]


# ---------------------------------------------------------------------------
# Optuna parameter suggestion helper (mirrors e_Train._optuna_suggest_param)
# ---------------------------------------------------------------------------

def _optuna_suggest_param(trial, spec, param_name: str):
    if isinstance(spec, dict):
        low = spec.get("low")
        high = spec.get("high")
        mode = str(spec.get("type", "float")).lower()
        if spec.get("log", False):
            mode = "log"
        if low is None or high is None:
            return spec.get("value")
        if mode == "int":
            return trial.suggest_int(param_name, int(low), int(high))
        if mode == "log":
            low = max(float(low), 1e-12)
            high = max(float(high), low * 1.000001)
            return trial.suggest_float(param_name, float(low), float(high), log=True)
        return trial.suggest_float(param_name, float(low), float(high))
    if isinstance(spec, (list, tuple)):
        return trial.suggest_categorical(param_name, list(spec))
    return spec


# ---------------------------------------------------------------------------
# Internal: single CV fold score
# ---------------------------------------------------------------------------

def _recurrent_cv_fold_score(
    model_cls,
    base_model_config: dict,
    trial_hyper: dict,
    train_fold_samples,
    val_fold_samples,
    state_col_idx: int,
    device: torch.device,
    num_epochs: int,
    patience: int,
    grad_clip_norm: float = 1.0,
) -> float:
    """Train model_cls for one CV fold and return validation RMSE."""
    fold_model_config = dict(base_model_config)
    fold_model_config.update({
        k: trial_hyper[k]
        for k in ("model_dim", "num_heads", "num_layers", "dropout", "hidden_dim")
        if k in trial_hyper
    })

    lr = trial_hyper.get("learning_rate", 1e-3)
    warmup_epochs = min(10, num_epochs // 5)

    model = model_cls(fold_model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr / 10.0)
    criterion = nn.MSELoss()

    batch_size = int(trial_hyper.get("batch_size", 16))

    train_ds = RecurrentTimeSeriesDataset(train_fold_samples, state_col_idx)
    val_ds = RecurrentTimeSeriesDataset(val_fold_samples, state_col_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_rmse = float("inf")
    no_improve = 0

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr / 10.0 + (lr - lr / 10.0) * scale

        model.train()
        for x_batch, y_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch).squeeze(-1)
            loss = criterion(pred, y_batch.squeeze(-1))
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        model.eval()
        val_sq_errors = []
        with torch.no_grad():
            for x_batch, y_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch).squeeze(-1)
                sq = ((pred - y_batch.squeeze(-1)) ** 2).cpu().numpy()
                val_sq_errors.extend(sq.tolist())

        val_rmse = float(np.sqrt(np.mean(val_sq_errors)))
        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val_rmse


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _build_trial_hyper(trial, param_space: dict, base_hyper: dict) -> dict:
    """Merge base hyperparameters with Optuna-suggested values from param_space."""
    hyper = dict(base_hyper)
    for param_name, spec in param_space.items():
        hyper[param_name] = _optuna_suggest_param(trial, spec, param_name)
    return hyper


def _make_optuna_objective(
    model_cls,
    base_model_config: dict,
    base_hyper: dict,
    param_space: dict,
    train_samples,
    state_col_idx: int,
    n_folds: int,
    device: torch.device,
    num_epochs: int,
    patience: int,
    seed: int,
    grad_clip_norm: float = 1.0,
):
    def objective(trial):
        trial_hyper = _build_trial_hyper(trial, param_space, base_hyper)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        indices = np.arange(len(train_samples))
        fold_rmses = []
        for train_idx, val_idx in kf.split(indices):
            fold_train = [train_samples[i] for i in train_idx]
            fold_val = [train_samples[i] for i in val_idx]
            rmse = _recurrent_cv_fold_score(
                model_cls,
                base_model_config,
                trial_hyper,
                fold_train,
                fold_val,
                state_col_idx,
                device,
                num_epochs,
                patience,
                grad_clip_norm=grad_clip_norm,
            )
            fold_rmses.append(rmse)
        return float(np.mean(fold_rmses))

    return objective


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_recurrent_model(config: dict, train_samples: list, test_samples: list, model_cls):
    """Train a recurrent model (RecurrentTransformerModel or LSTMForecasterModel).

    Parameters
    ----------
    config : dict
        Full config as loaded from YAML.
    train_samples, test_samples : list of (input_seq, output_seq, filename)
    model_cls : class
        RecurrentTransformerModel or LSTMForecasterModel.

    Returns
    -------
    dict with keys: 'predictions_train', 'targets_train', 'filenames_train',
                    'predictions_test', 'targets_test', 'filenames_test',
                    'best_params'
    """
    print("\n" + "=" * 80)
    print(f"TRAINING {model_cls.__name__.upper()}")
    print("=" * 80)

    device = torch.device(config["device"])
    data_cfg = config["data"]
    hyper_cfg = config["hyperparameters"]

    input_columns = data_cfg["input_columns"]
    state_column = data_cfg["state_column"]
    if state_column not in input_columns:
        raise ValueError(
            f"state_column '{state_column}' must be in data.input_columns"
        )
    state_col_idx = input_columns.index(state_column)

    seq_len = data_cfg["input_row_2"] - data_cfg["input_row_1"]
    input_dim = len(input_columns)

    base_model_config = {
        "input_dim": input_dim,
        "seq_len": seq_len,
        "output_dim": 1,
        # Transformer defaults (ignored by LSTM)
        "model_dim": hyper_cfg.get("model_dim", 64),
        "num_heads": hyper_cfg.get("num_heads", 4),
        "num_layers": hyper_cfg.get("num_layers", 2),
        "dropout": hyper_cfg.get("dropout", 0.1),
        # LSTM default (ignored by Transformer)
        "hidden_dim": hyper_cfg.get("hidden_dim", 64),
    }

    base_hyper = {
        "batch_size": hyper_cfg.get("batch_size", 16),
        "learning_rate": hyper_cfg.get("learning_rate", 1e-3),
        "num_epochs": hyper_cfg.get("num_epochs", 100),
        "patience": hyper_cfg.get("patience", 20),
        **{k: hyper_cfg[k] for k in ("model_dim", "num_heads", "num_layers",
                                      "dropout", "hidden_dim")
           if k in hyper_cfg},
    }

    # ------------------------------------------------------------------
    # Regularisation settings
    # ------------------------------------------------------------------
    grad_clip_norm = hyper_cfg.get("grad_clip_norm", 1.0)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    # ------------------------------------------------------------------
    # Optuna hyperparameter tuning
    # ------------------------------------------------------------------
    cv_cfg = hyper_cfg.get("cv_tuning", {})
    best_params = dict(base_hyper)

    if cv_cfg.get("enabled", False):
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("optuna is required for cv_tuning. Install it with: pip install optuna")

        n_trials = int(cv_cfg.get("n_trials", 20))
        n_folds = int(cv_cfg.get("n_folds", 5))
        param_space = cv_cfg.get("param_space", {})
        seed = int(config.get("data_split", {}).get("random_state", 42))
        cv_epochs = int(cv_cfg.get("cv_epochs", base_hyper["num_epochs"]))
        cv_patience = int(cv_cfg.get("cv_patience", base_hyper["patience"]))

        print(f"\n[INFO] Optuna CV tuning: {n_trials} trials, {n_folds} folds")

        objective = _make_optuna_objective(
            model_cls,
            base_model_config,
            base_hyper,
            param_space,
            train_samples,
            state_col_idx,
            n_folds,
            device,
            cv_epochs,
            cv_patience,
            seed,
            grad_clip_norm=grad_clip_norm,
        )

        def _logged_objective(trial):
            value = objective(trial)
            print(
                f"  Trial {trial.number + 1:3d}/{n_trials}  RMSE={value:.6f}"
                f"  params={trial.params}"
            )
            return value

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(_logged_objective, n_trials=n_trials, show_progress_bar=False)

        best_trial_params = study.best_trial.params
        best_params = _build_trial_hyper(study.best_trial, param_space, base_hyper)
        print(f"[INFO] Best trial RMSE: {study.best_trial.value:.6f}")
        print(f"[INFO] Best params: {best_trial_params}")
    else:
        print("[INFO] CV tuning disabled; using config hyperparameters.")

    # ------------------------------------------------------------------
    # Build final model config from best params
    # ------------------------------------------------------------------
    final_model_config = dict(base_model_config)
    for k in ("model_dim", "num_heads", "num_layers", "dropout", "hidden_dim"):
        if k in best_params:
            final_model_config[k] = best_params[k]

    # ------------------------------------------------------------------
    # Train final model on all training data
    # ------------------------------------------------------------------
    num_epochs = int(best_params.get("num_epochs", hyper_cfg.get("num_epochs", 100)))
    patience = int(best_params.get("patience", hyper_cfg.get("patience", 20)))
    learning_rate = float(best_params.get("learning_rate", hyper_cfg.get("learning_rate", 1e-3)))
    batch_size = int(best_params.get("batch_size", hyper_cfg.get("batch_size", 16)))

    # LR warmup: ramp from lr/10 to lr over the first warmup_epochs epochs.
    # Prevents large early gradient steps from locking attention weights into
    # a uniform-attention fixed point before the causal mask can guide specialization.
    warmup_epochs = int(hyper_cfg.get("warmup_epochs", min(10, num_epochs // 5)))

    print(f"\n[INFO] Training final model: epochs={num_epochs}, lr={learning_rate}, "
          f"batch={batch_size}, patience={patience}, grad_clip_norm={grad_clip_norm}, "
          f"warmup_epochs={warmup_epochs}")

    train_ds = RecurrentTimeSeriesDataset(train_samples, state_col_idx)
    test_ds = RecurrentTimeSeriesDataset(test_samples, state_col_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = model_cls(final_model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate / 10.0)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(num_epochs):
        # Linear LR warmup
        if epoch < warmup_epochs:
            scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = learning_rate / 10.0 + (learning_rate - learning_rate / 10.0) * scale

        model.train()
        train_losses = []
        for x_batch, y_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch).squeeze(-1)
            loss = criterion(pred, y_batch.squeeze(-1))
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation pass for early stopping
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch, _ in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch).squeeze(-1)
                val_losses.append(criterion(pred, y_batch.squeeze(-1)).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:4d}/{num_epochs} | "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={current_lr:.6f}"
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1} (patience={patience})")
                break

    # ------------------------------------------------------------------
    # Collect predictions
    # ------------------------------------------------------------------
    def collect_predictions(loader):
        model.eval()
        preds, targets, fnames = [], [], []
        with torch.no_grad():
            for x_batch, y_batch, fname_batch in loader:
                x_batch = x_batch.to(device)
                pred = model(x_batch).squeeze(-1).cpu().numpy()
                preds.extend(pred.tolist())
                targets.extend(y_batch.squeeze(-1).numpy().tolist())
                fnames.extend(list(fname_batch))
        return np.array(preds), np.array(targets), fnames

    preds_train, targets_train, fnames_train = collect_predictions(train_loader)
    preds_test, targets_test, fnames_test = collect_predictions(test_loader)

    return {
        "model": model,
        "final_model_config": final_model_config,
        "best_params": best_params,
        "predictions_train": preds_train,
        "targets_train": targets_train,
        "filenames_train": fnames_train,
        "predictions_test": preds_test,
        "targets_test": targets_test,
        "filenames_test": fnames_test,
    }

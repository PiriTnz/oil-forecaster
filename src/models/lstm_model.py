"""
LSTM-based oil price forecaster.

LSTM/GRU are well-suited to oil because:
  - Sequential patterns matter (momentum, mean-reversion at multiple horizons)
  - Regime persistence (once in war regime, tends to stay)
  - Multi-feature time series input is natural

Architecture:
  Input: (batch, sequence_len, n_features)
  -> LSTM layer(s) with dropout
  -> Dense head
  -> Output: h-day forward return

Walk-forward validation, same as XGB, ensures no leakage.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    horizon_days: int = 5
    sequence_length: int = 60  # lookback window
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    patience: int = 8
    random_state: int = 42
    device: str = "cpu"  # set to "cuda" if GPU available


class LSTMForecaster:
    """Pytorch LSTM forecaster for forward returns"""

    def __init__(self, config: Optional[LSTMConfig] = None):
        self.config = config or LSTMConfig()
        self.model = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.feature_cols: list[str] = []
        self.target_col: str = ""
        self.history: dict = {}

    def fit(self, df: pd.DataFrame, feature_cols: list[str],
            val_fraction: float = 0.2) -> "LSTMForecaster":
        """Train LSTM with chronological train/val split"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        self.feature_cols = feature_cols
        self.target_col = f"target_logret_{self.config.horizon_days}d"

        data = df[feature_cols + [self.target_col]].dropna().reset_index(drop=True)
        X_raw = data[feature_cols].values.astype(np.float32)
        y = data[self.target_col].values.astype(np.float32)

        # Standardize features using TRAIN ONLY stats (avoid leakage)
        split = int(len(X_raw) * (1 - val_fraction))
        self.scaler_mean = X_raw[:split].mean(axis=0)
        self.scaler_std = X_raw[:split].std(axis=0) + 1e-8
        X = (X_raw - self.scaler_mean) / self.scaler_std

        # Build sequences
        X_seq, y_seq = self._build_sequences(X, y)
        split_seq = split - self.config.sequence_length
        X_tr, X_val = X_seq[:split_seq], X_seq[split_seq:]
        y_tr, y_val = y_seq[:split_seq], y_seq[split_seq:]

        logger.info(f"Train: {X_tr.shape}, Val: {X_val.shape}")

        device = torch.device(self.config.device)
        self.model = _LSTMNet(
            n_features=len(feature_cols),
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

        best_val = float("inf")
        best_state = None
        no_improve = 0
        history = {"train_loss": [], "val_loss": [], "val_dir_acc": []}

        for epoch in range(self.config.epochs):
            # Train
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = self.model(xb).squeeze()
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                train_losses.append(loss.item())

            # Validate
            self.model.eval()
            val_losses = []
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    p = self.model(xb).squeeze()
                    val_losses.append(loss_fn(p, yb).item())
                    preds.append(p.cpu().numpy())
                    trues.append(yb.cpu().numpy())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            preds_arr = np.concatenate(preds)
            trues_arr = np.concatenate(trues)
            dir_acc = float(np.mean(np.sign(preds_arr) == np.sign(trues_arr)))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_dir_acc"].append(dir_acc)

            logger.info(f"Epoch {epoch+1}/{self.config.epochs} "
                        f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                        f"val_dir_acc={dir_acc:.3f}")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.history = history
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        import torch
        if self.model is None:
            raise RuntimeError("Call .fit() first")

        data = df[self.feature_cols].dropna()
        X = (data.values.astype(np.float32) - self.scaler_mean) / self.scaler_std

        if len(X) < self.config.sequence_length:
            raise ValueError(f"Need at least {self.config.sequence_length} rows")

        X_seq, _ = self._build_sequences(X, np.zeros(len(X)))

        device = torch.device(self.config.device)
        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X_seq).to(device)
            pred = self.model(xb).squeeze().cpu().numpy()

        # Pad the beginning (no prediction for first sequence_length-1 rows)
        out = np.full(len(data), np.nan)
        out[self.config.sequence_length - 1:] = pred
        return out

    def _build_sequences(self, X: np.ndarray, y: np.ndarray):
        seq_len = self.config.sequence_length
        n = len(X) - seq_len + 1
        X_seq = np.zeros((n, seq_len, X.shape[1]), dtype=np.float32)
        for i in range(n):
            X_seq[i] = X[i:i + seq_len]
        y_seq = y[seq_len - 1:]
        return X_seq, y_seq

    def save(self, path: str) -> None:
        import torch
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": asdict(self.config),
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "history": self.history,
        }, p / "lstm_model.pt")
        logger.info(f"Saved LSTM model to {p}")

    @classmethod
    def load(cls, path: str) -> "LSTMForecaster":
        import torch
        p = Path(path)
        checkpoint = torch.load(p / "lstm_model.pt", map_location="cpu")
        config = LSTMConfig(**checkpoint["config"])
        inst = cls(config)
        inst.feature_cols = checkpoint["feature_cols"]
        inst.target_col = checkpoint["target_col"]
        inst.scaler_mean = checkpoint["scaler_mean"]
        inst.scaler_std = checkpoint["scaler_std"]
        inst.history = checkpoint["history"]
        inst.model = _LSTMNet(
            n_features=len(inst.feature_cols),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        inst.model.load_state_dict(checkpoint["model_state"])
        inst.model.eval()
        return inst


# ---------------------------------------------------------------------
# Internal network
# ---------------------------------------------------------------------
def _make_lstm_class():
    import torch.nn as nn

    class LSTMNet(nn.Module):
        def __init__(self, n_features: int, hidden_size: int,
                     num_layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last)

    return LSTMNet


# Lazy-load torch only when needed (avoid import cost when only using XGB)
class _LSTMNet:
    """Proxy that materializes the real module on first instantiation"""
    def __new__(cls, *args, **kwargs):
        RealClass = _make_lstm_class()
        return RealClass(*args, **kwargs)

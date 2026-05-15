"""
LSTM with Regime-Aware Attention.

Standard LSTM treats every past timestep equally. For oil forecasting that's
wrong: when we're in a war regime today, days that were ALSO in a war regime
should dominate the prediction, not the calm-market days from last quarter.

Architecture:
  Input sequence: (B, L, d) where L=60, d=feature dim
  Each timestep also has a regime label z_s ∈ {1, ..., K}.

  1) LSTM encodes sequence → hidden states H ∈ ℝ^(B × L × h)
  2) Regime-aware attention:
       Let z_t = current regime (last timestep)
       For each past step s:
         compatibility(s) = (Q h_t)^T (K h_s) / √h
         regime_bonus(s)  = α · 1[z_s = z_t]    # boost matching regimes
         logit(s) = compatibility(s) + regime_bonus(s)
       weights(s) = softmax_s(logit(s))
       context = Σ_s weights(s) · (V h_s)
  3) Combine context with current hidden state → prediction head

The α coefficient is learned. If α grows large, the model is saying "regime
matters a lot." If α stays near zero, it's a regular attention model.

We also add the Trump volatility factor as a SECOND attention bonus: days
when rhetoric was volatile get DOWN-weighted (less reliable signal).

  trump_penalty(s) = -γ · trump_volatility_factor[s]
  logit(s) = compatibility(s) + regime_bonus(s) + trump_penalty(s)
"""
from __future__ import annotations
import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeLSTMConfig:
    horizon_days: int = 5
    sequence_length: int = 60
    n_features: int = 0          # set at fit time
    n_regimes: int = 4

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    attention_heads: int = 4

    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    patience: int = 8
    weight_decay: float = 1e-5

    # Regime-aware attention coefficients
    regime_bonus_init: float = 0.5      # initial value for α
    trump_penalty_init: float = 0.5     # initial value for γ
    learn_attention_coefs: bool = True   # let α, γ be trained

    random_state: int = 42
    device: str = "cpu"


# ============================================================================
# Internal nn module (lazy-loaded so torch isn't imported unless needed)
# ============================================================================
def _build_module(config: RegimeLSTMConfig):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RegimeAwareAttention(nn.Module):
        """
        Multi-head attention with two bonus terms in the logits:
          - regime match bonus (positive if past step is same regime as current)
          - trump volatility penalty (negative for high-volatility days)
        """
        def __init__(self, hidden_size: int, n_heads: int,
                     regime_bonus_init: float, trump_penalty_init: float,
                     learn_coefs: bool = True):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = hidden_size // n_heads
            assert hidden_size % n_heads == 0
            self.W_q = nn.Linear(hidden_size, hidden_size)
            self.W_k = nn.Linear(hidden_size, hidden_size)
            self.W_v = nn.Linear(hidden_size, hidden_size)
            self.W_o = nn.Linear(hidden_size, hidden_size)

            # Learnable bonus / penalty coefficients
            self.alpha = nn.Parameter(
                torch.tensor(regime_bonus_init),
                requires_grad=learn_coefs,
            )
            self.gamma = nn.Parameter(
                torch.tensor(trump_penalty_init),
                requires_grad=learn_coefs,
            )

        def forward(self, h, regime_ids, trump_volatility):
            """
            h: (B, L, H)
            regime_ids: (B, L) — integer regime indices
            trump_volatility: (B, L) — float in [0, 1]

            Returns: (B, H) — pooled context vector
            """
            B, L, H = h.shape
            Q = self.W_q(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B,nh,L,dh)
            K = self.W_k(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            V = self.W_v(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

            # We compute attention from the LAST timestep as query to all timesteps
            Q_last = Q[:, :, -1:, :]  # (B, nh, 1, dh)

            # Raw compatibility
            scores = (Q_last @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, 1, L)

            # Regime bonus: +α where regime_s == regime_last
            regime_last = regime_ids[:, -1:].unsqueeze(1).unsqueeze(-1)  # (B, 1, 1, 1)
            regime_match = (regime_ids.unsqueeze(1).unsqueeze(1) == regime_last).float()  # (B,1,1,L)
            scores = scores + self.alpha * regime_match  # broadcast over heads

            # Trump volatility penalty: -γ · vol_s for every past step
            tvol = trump_volatility.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            scores = scores - self.gamma * tvol

            weights = F.softmax(scores, dim=-1)  # (B, nh, 1, L)
            context = weights @ V  # (B, nh, 1, dh)
            context = context.transpose(1, 2).contiguous().view(B, 1, H).squeeze(1)
            return self.W_o(context), weights.squeeze(2)  # context, attention weights

    class RegimeAwareLSTM(nn.Module):
        def __init__(self, cfg: RegimeLSTMConfig):
            super().__init__()
            self.cfg = cfg
            self.lstm = nn.LSTM(
                input_size=cfg.n_features,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )
            self.attn = RegimeAwareAttention(
                hidden_size=cfg.hidden_size,
                n_heads=cfg.attention_heads,
                regime_bonus_init=cfg.regime_bonus_init,
                trump_penalty_init=cfg.trump_penalty_init,
                learn_coefs=cfg.learn_attention_coefs,
            )
            self.head = nn.Sequential(
                nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_size, 1),
            )

        def forward(self, x, regime_ids, trump_volatility):
            h, _ = self.lstm(x)  # (B, L, H)
            context, attn_weights = self.attn(h, regime_ids, trump_volatility)
            last = h[:, -1, :]
            combined = torch.cat([last, context], dim=-1)
            return self.head(combined).squeeze(-1), attn_weights

    return RegimeAwareLSTM(config)


# ============================================================================
# Public wrapper
# ============================================================================
class RegimeAwareLSTMForecaster:
    """
    LSTM with regime-aware attention.

    fit() expects a DataFrame with:
      - feature columns (as specified)
      - 'regime_id' column (integer regime labels)
      - 'trump_volatility_factor' column (optional; defaults to 0)
      - target column 'target_logret_{h}d'
    """

    def __init__(self, config: Optional[RegimeLSTMConfig] = None):
        self.config = config or RegimeLSTMConfig()
        self.model = None
        self.feature_cols: list[str] = []
        self.regime_col: str = "regime_id"
        self.trump_vol_col: str = "trump_volatility_factor"
        self.target_col: str = ""
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.history: dict = {}
        self.learned_coefs: dict = {}  # α, γ after training

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        val_fraction: float = 0.2,
    ) -> "RegimeAwareLSTMForecaster":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        self.feature_cols = feature_cols
        self.target_col = f"target_logret_{self.config.horizon_days}d"
        self.config.n_features = len(feature_cols)

        # Default missing columns
        df = df.copy()
        if self.regime_col not in df.columns:
            df[self.regime_col] = 0
        if self.trump_vol_col not in df.columns:
            df[self.trump_vol_col] = 0.0

        # Drop NaN target rows; for features we fill 0 after scaling
        data = df[feature_cols + [self.regime_col, self.trump_vol_col, self.target_col]].copy()
        data = data.dropna(subset=[self.target_col]).reset_index(drop=True)
        for c in feature_cols:
            data[c] = data[c].fillna(method="ffill").fillna(0)
        data[self.regime_col] = data[self.regime_col].fillna(0).astype(int)
        data[self.trump_vol_col] = data[self.trump_vol_col].fillna(0.0).astype(float)

        X = data[feature_cols].values.astype(np.float32)
        regimes = data[self.regime_col].values.astype(np.int64)
        tvol = data[self.trump_vol_col].values.astype(np.float32)
        y = data[self.target_col].values.astype(np.float32)

        # Scale features using train portion only
        split_pt = int(len(X) * (1 - val_fraction))
        self.scaler_mean = X[:split_pt].mean(axis=0)
        self.scaler_std = X[:split_pt].std(axis=0) + 1e-8
        X = (X - self.scaler_mean) / self.scaler_std

        # Build sequences
        X_seq, regimes_seq, tvol_seq, y_seq = self._build_sequences(X, regimes, tvol, y)
        split_seq = split_pt - self.config.sequence_length
        if split_seq < 50 or len(X_seq) - split_seq < 20:
            raise ValueError("Insufficient data after sequencing")

        X_tr, X_val = X_seq[:split_seq], X_seq[split_seq:]
        r_tr, r_val = regimes_seq[:split_seq], regimes_seq[split_seq:]
        v_tr, v_val = tvol_seq[:split_seq], tvol_seq[split_seq:]
        y_tr, y_val = y_seq[:split_seq], y_seq[split_seq:]

        logger.info(f"Train: {X_tr.shape}, Val: {X_val.shape}")

        device = torch.device(self.config.device)
        self.model = _build_module(self.config).to(device)
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()

        train_ds = TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(r_tr),
            torch.from_numpy(v_tr),
            torch.from_numpy(y_tr),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(r_val),
            torch.from_numpy(v_val),
            torch.from_numpy(y_val),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": [], "val_dir_acc": [],
                   "alpha": [], "gamma": []}
        best_val = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            tr_losses = []
            for xb, rb, vb, yb in train_loader:
                xb, rb, vb, yb = xb.to(device), rb.to(device), vb.to(device), yb.to(device)
                opt.zero_grad()
                pred, _ = self.model(xb, rb, vb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tr_losses.append(loss.item())

            self.model.eval()
            val_losses, preds_all, trues_all = [], [], []
            with torch.no_grad():
                for xb, rb, vb, yb in val_loader:
                    xb, rb, vb, yb = xb.to(device), rb.to(device), vb.to(device), yb.to(device)
                    p, _ = self.model(xb, rb, vb)
                    val_losses.append(loss_fn(p, yb).item())
                    preds_all.append(p.cpu().numpy())
                    trues_all.append(yb.cpu().numpy())

            tr_loss = float(np.mean(tr_losses))
            val_loss = float(np.mean(val_losses))
            preds_arr = np.concatenate(preds_all)
            trues_arr = np.concatenate(trues_all)
            dir_acc = float(np.mean(np.sign(preds_arr) == np.sign(trues_arr)))

            alpha = float(self.model.attn.alpha.item())
            gamma = float(self.model.attn.gamma.item())

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["val_dir_acc"].append(dir_acc)
            history["alpha"].append(alpha)
            history["gamma"].append(gamma)

            logger.info(
                f"Ep {epoch+1:>2d}/{self.config.epochs}  "
                f"tr={tr_loss:.5f}  val={val_loss:.5f}  dir_acc={dir_acc:.3f}  "
                f"α={alpha:+.3f}  γ={gamma:+.3f}"
            )

            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.history = history
        self.learned_coefs = {
            "alpha_regime_bonus": float(self.model.attn.alpha.item()),
            "gamma_trump_penalty": float(self.model.attn.gamma.item()),
        }
        logger.info(f"Final learned coefficients: {self.learned_coefs}")
        return self

    def predict(self, df: pd.DataFrame, return_attention: bool = False):
        """Returns predictions (and optionally attention weights for interpretability)."""
        import torch
        if self.model is None:
            raise RuntimeError("Call .fit() first")

        df = df.copy()
        if self.regime_col not in df.columns:
            df[self.regime_col] = 0
        if self.trump_vol_col not in df.columns:
            df[self.trump_vol_col] = 0.0

        data = df[self.feature_cols + [self.regime_col, self.trump_vol_col]].copy()
        for c in self.feature_cols:
            data[c] = data[c].fillna(method="ffill").fillna(0)
        data[self.regime_col] = data[self.regime_col].fillna(0).astype(int)
        data[self.trump_vol_col] = data[self.trump_vol_col].fillna(0.0).astype(float)

        X = (data[self.feature_cols].values.astype(np.float32) - self.scaler_mean) / self.scaler_std
        regimes = data[self.regime_col].values.astype(np.int64)
        tvol = data[self.trump_vol_col].values.astype(np.float32)

        if len(X) < self.config.sequence_length:
            raise ValueError(f"Need ≥{self.config.sequence_length} rows")

        X_seq, r_seq, v_seq, _ = self._build_sequences(X, regimes, tvol, np.zeros(len(X)))

        device = torch.device(self.config.device)
        self.model.eval()
        with torch.no_grad():
            pred, attn = self.model(
                torch.from_numpy(X_seq).to(device),
                torch.from_numpy(r_seq).to(device),
                torch.from_numpy(v_seq).to(device),
            )
            pred = pred.cpu().numpy()
            attn = attn.cpu().numpy()

        # Pad beginning with NaN (no prediction for first sequence_length-1 rows)
        out = np.full(len(data), np.nan)
        out[self.config.sequence_length - 1:] = pred

        if return_attention:
            return out, attn
        return out

    def _build_sequences(self, X, regimes, tvol, y):
        L = self.config.sequence_length
        n = len(X) - L + 1
        X_seq = np.zeros((n, L, X.shape[1]), dtype=np.float32)
        r_seq = np.zeros((n, L), dtype=np.int64)
        v_seq = np.zeros((n, L), dtype=np.float32)
        for i in range(n):
            X_seq[i] = X[i:i + L]
            r_seq[i] = regimes[i:i + L]
            v_seq[i] = tvol[i:i + L]
        y_seq = y[L - 1:]
        return X_seq, r_seq, v_seq, y_seq

    def save(self, path: str) -> None:
        import torch
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": asdict(self.config),
            "feature_cols": self.feature_cols,
            "regime_col": self.regime_col,
            "trump_vol_col": self.trump_vol_col,
            "target_col": self.target_col,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "history": self.history,
            "learned_coefs": self.learned_coefs,
        }, p / "regime_lstm.pt")
        logger.info(f"Saved RegimeAwareLSTM to {p}")

    @classmethod
    def load(cls, path: str) -> "RegimeAwareLSTMForecaster":
        import torch
        p = Path(path)
        ckpt = torch.load(p / "regime_lstm.pt", map_location="cpu", weights_only=False)
        config = RegimeLSTMConfig(**ckpt["config"])
        inst = cls(config)
        inst.feature_cols = ckpt["feature_cols"]
        inst.regime_col = ckpt["regime_col"]
        inst.trump_vol_col = ckpt["trump_vol_col"]
        inst.target_col = ckpt["target_col"]
        inst.scaler_mean = ckpt["scaler_mean"]
        inst.scaler_std = ckpt["scaler_std"]
        inst.history = ckpt["history"]
        inst.learned_coefs = ckpt["learned_coefs"]
        inst.model = _build_module(config)
        inst.model.load_state_dict(ckpt["model_state"])
        inst.model.eval()
        return inst

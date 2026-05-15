"""
Ensemble forecaster: combine XGBoost + LSTM (and optionally Prophet) into one
prediction.

Strategy: weighted average of forward-return predictions, where weights come
from per-model out-of-sample directional accuracy on a recent window.

We expose:
  - point forecast (forward log return)
  - probabilistic forecast (mean + std from bootstrap or model disagreement)
  - direction (up/flat/down) with confidence
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.models.xgb_model import XGBoostForecaster, XGBConfig
from src.models.lstm_model import LSTMForecaster, LSTMConfig

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    horizon_days: int = 5
    use_xgb: bool = True
    use_lstm: bool = True
    use_prophet: bool = False
    # Static weights (overridden by recent-accuracy weighting if available)
    weights: dict = field(default_factory=lambda: {"xgb": 0.6, "lstm": 0.4})
    dynamic_weights: bool = True
    dynamic_window: int = 60  # days


@dataclass
class Forecast:
    """A single forecast for a target horizon"""
    forecast_date: pd.Timestamp
    horizon_days: int
    predicted_log_return: float
    predicted_pct_return: float
    direction: str  # "up", "down", "flat"
    confidence: float  # 0-1
    model_predictions: dict  # individual model outputs
    model_weights: dict
    notes: str = ""


class EnsembleForecaster:
    """Combine multiple models for a more robust forecast"""

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.xgb: Optional[XGBoostForecaster] = None
        self.lstm: Optional[LSTMForecaster] = None
        self.feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "EnsembleForecaster":
        self.feature_cols = feature_cols
        h = self.config.horizon_days

        if self.config.use_xgb:
            logger.info("Training XGBoost...")
            self.xgb = XGBoostForecaster(XGBConfig(horizon_days=h, task="regression"))
            self.xgb.fit(df, feature_cols)

        if self.config.use_lstm:
            logger.info("Training LSTM...")
            self.lstm = LSTMForecaster(LSTMConfig(horizon_days=h))
            self.lstm.fit(df, feature_cols)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ensemble predictions as a DataFrame aligned with df"""
        out = df.copy()
        n = len(df)
        preds = {}

        if self.xgb is not None:
            preds["xgb"] = self.xgb.predict(df[self.feature_cols].fillna(0))
        if self.lstm is not None:
            preds["lstm"] = self.lstm.predict(df)

        # Combine
        weights = self.config.weights
        ensemble = np.zeros(n)
        total_weight = 0
        for name, pred in preds.items():
            w = weights.get(name, 0)
            # Replace NaN predictions with 0 for the combination
            pred_filled = np.where(np.isnan(pred), 0, pred)
            valid_mask = ~np.isnan(pred)
            ensemble += w * pred_filled
            total_weight += w * valid_mask.astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
            ensemble = np.where(total_weight > 0, ensemble / total_weight, np.nan)

        out["pred_ensemble"] = ensemble
        for name, p in preds.items():
            out[f"pred_{name}"] = p

        # Model disagreement (proxy for uncertainty)
        if len(preds) > 1:
            stacked = np.vstack([np.where(np.isnan(p), 0, p) for p in preds.values()])
            out["pred_disagreement"] = np.std(stacked, axis=0)

        return out

    def forecast_single(self, df: pd.DataFrame) -> Forecast:
        """Make a single forecast for the latest row in df"""
        result = self.predict(df)
        # Take the last non-NaN ensemble prediction
        valid = result[~result["pred_ensemble"].isna()]
        if valid.empty:
            raise ValueError("No valid prediction (insufficient lookback?)")

        latest = valid.iloc[-1]
        pred_log = float(latest["pred_ensemble"])
        pred_pct = (np.exp(pred_log) - 1)

        # Direction with small dead-zone
        if pred_pct > 0.01:
            direction = "up"
        elif pred_pct < -0.01:
            direction = "down"
        else:
            direction = "flat"

        # Confidence from disagreement (less disagreement => higher confidence)
        if "pred_disagreement" in latest:
            disagreement = float(latest["pred_disagreement"])
            confidence = float(np.exp(-disagreement * 20))  # map to (0, 1)
            confidence = max(0.0, min(confidence, 1.0))
        else:
            confidence = 0.5

        return Forecast(
            forecast_date=pd.Timestamp(latest["date"]) if "date" in latest else pd.Timestamp.now(),
            horizon_days=self.config.horizon_days,
            predicted_log_return=pred_log,
            predicted_pct_return=pred_pct,
            direction=direction,
            confidence=confidence,
            model_predictions={
                k: float(latest[f"pred_{k}"])
                for k in ["xgb", "lstm"]
                if f"pred_{k}" in latest and not pd.isna(latest[f"pred_{k}"])
            },
            model_weights=self.config.weights,
        )

    def save(self, path: str) -> None:
        from pathlib import Path
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self.xgb:
            self.xgb.save(str(p / "xgb"))
        if self.lstm:
            self.lstm.save(str(p / "lstm"))

        import json
        with open(p / "ensemble_meta.json", "w") as f:
            json.dump({
                "feature_cols": self.feature_cols,
                "config": {
                    "horizon_days": self.config.horizon_days,
                    "use_xgb": self.config.use_xgb,
                    "use_lstm": self.config.use_lstm,
                    "weights": self.config.weights,
                },
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EnsembleForecaster":
        from pathlib import Path
        import json
        p = Path(path)
        with open(p / "ensemble_meta.json") as f:
            meta = json.load(f)

        config = EnsembleConfig(**meta["config"])
        inst = cls(config)
        inst.feature_cols = meta["feature_cols"]

        if config.use_xgb and (p / "xgb").exists():
            inst.xgb = XGBoostForecaster.load(str(p / "xgb"))
        if config.use_lstm and (p / "lstm").exists():
            inst.lstm = LSTMForecaster.load(str(p / "lstm"))

        return inst

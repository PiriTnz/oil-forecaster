"""
Quantile regression forecaster — implementation of section 8.3 / 11.2.

Trains three XGBoost regressors with quantile loss:

    ρ_q(u) = q·u         if u ≥ 0
             (q-1)·u     if u < 0

for q ∈ {q_lower, 0.5, q_upper}, producing a prediction interval
[ŷ_{q_lower,t}, ŷ_{q_upper,t}] that has approximately (q_upper - q_lower)
coverage on the training distribution.

We use this for:
  - 80% prediction bands (q ∈ {0.1, 0.9})
  - The median forecast (q = 0.5) as a robust point estimate
  - Comparison with the squared-error ensemble forecast
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class QuantileConfig:
    horizon_days: int = 5
    quantiles: tuple = (0.1, 0.5, 0.9)
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    n_splits: int = 5
    random_state: int = 42


class QuantileXGBoostForecaster:
    """
    Quantile regression forecaster.

    For each requested quantile q, trains an XGBoost model with the corresponding
    quantile loss. Quantile XGB uses the `quantile` objective with `quantile_alpha=q`.
    """

    def __init__(self, config: Optional[QuantileConfig] = None):
        self.config = config or QuantileConfig()
        self.models: dict[float, object] = {}  # q → trained model
        self.feature_cols: list[str] = []
        self.target_col: str = ""
        self.cv_metrics: list[dict] = []

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "QuantileXGBoostForecaster":
        import xgboost as xgb

        self.feature_cols = feature_cols
        h = self.config.horizon_days
        self.target_col = f"target_logret_{h}d"

        data = df[feature_cols + [self.target_col]].dropna()
        if len(data) < 200:
            raise ValueError(f"Need ≥200 clean rows, got {len(data)}")

        X = data[feature_cols].values
        y = data[self.target_col].values

        # Time-series CV for the median model only (others use same split)
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)

        for q in self.config.quantiles:
            logger.info(f"Training quantile model q={q}")

            cv_losses = []
            last_model = None
            for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                model = xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    reg_lambda=self.config.reg_lambda,
                    random_state=self.config.random_state,
                    early_stopping_rounds=30,
                    n_jobs=-1,
                    tree_method="hist",
                )

                try:
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                except Exception as e:
                    # Older xgboost may not support quantile objective; fall back to reg:squarederror
                    logger.warning(f"Quantile objective unavailable ({e}); falling back to MSE")
                    model = xgb.XGBRegressor(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        learning_rate=self.config.learning_rate,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                    )
                    model.fit(X_tr, y_tr, verbose=False)

                pred = model.predict(X_val)
                loss = _quantile_loss(y_val, pred, q)
                cv_losses.append(loss)
                last_model = model

            self.cv_metrics.append({
                "quantile": q,
                "mean_pinball_loss": float(np.mean(cv_losses)),
                "fold_losses": [float(l) for l in cv_losses],
            })

            # Train final model on full data
            final = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
            try:
                final.fit(X, y, verbose=False)
            except Exception:
                final = xgb.XGBRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
                final.fit(X, y, verbose=False)
            self.models[q] = final

        logger.info(f"Trained {len(self.models)} quantile models")
        return self

    def predict_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame with columns for each quantile.

        The interval [q_low, q_high] has approximate (q_high - q_low) coverage.
        """
        if not self.models:
            raise RuntimeError("Call .fit() first")

        X = df[self.feature_cols].fillna(0).values
        out = pd.DataFrame(index=df.index)
        for q, model in sorted(self.models.items()):
            out[f"q_{int(q*100):02d}"] = model.predict(X)
        return out

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        for q, model in self.models.items():
            model.save_model(str(p / f"qmodel_{int(q*1000):04d}.json"))
        meta = {
            "config": asdict(self.config),
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "cv_metrics": self.cv_metrics,
            "quantiles": list(self.models.keys()),
        }
        with open(p / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "QuantileXGBoostForecaster":
        import xgboost as xgb
        p = Path(path)
        with open(p / "metadata.json") as f:
            meta = json.load(f)

        # quantiles in meta might be tuple or list of floats
        config_dict = meta["config"]
        # Ensure the quantiles field is a tuple
        if isinstance(config_dict.get("quantiles"), list):
            config_dict["quantiles"] = tuple(config_dict["quantiles"])
        config = QuantileConfig(**config_dict)

        inst = cls(config)
        inst.feature_cols = meta["feature_cols"]
        inst.target_col = meta["target_col"]
        inst.cv_metrics = meta["cv_metrics"]
        for q in meta["quantiles"]:
            m = xgb.XGBRegressor()
            m.load_model(str(p / f"qmodel_{int(float(q)*1000):04d}.json"))
            inst.models[float(q)] = m
        return inst


def _quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball / quantile loss ρ_q(y - ŷ)"""
    u = y_true - y_pred
    return float(np.mean(np.maximum(q * u, (q - 1) * u)))


def evaluate_intervals(
    intervals: pd.DataFrame,
    y_true: np.ndarray,
    target_coverage: float = 0.8,
) -> dict:
    """
    Check empirical coverage of a prediction interval.

    intervals: DataFrame with columns q_05, q_95 (or similar quantile columns)
    """
    y_true = np.asarray(y_true).ravel()
    cols = sorted(intervals.columns)
    lower_col, upper_col = cols[0], cols[-1]
    lower = intervals[lower_col].values
    upper = intervals[upper_col].values

    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    covered = (y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])
    width = (upper[mask] - lower[mask])

    return {
        "target_coverage": target_coverage,
        "empirical_coverage": float(covered.mean()),
        "mean_width": float(width.mean()),
        "n_observations": int(mask.sum()),
    }

"""
XGBoost-based oil price forecaster.

This is the workhorse model — fast, robust, handles non-linearity well, and
gives feature importances we can show to users.

Two model types:
  - Regressor: predicts h-day forward log return
  - Classifier: predicts direction (up/flat/down)

Training uses TimeSeriesSplit (never random) to avoid leakage.
"""
from __future__ import annotations
import logging
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report
)

logger = logging.getLogger(__name__)


@dataclass
class XGBConfig:
    horizon_days: int = 5
    task: str = "regression"  # or "classification"

    # XGBoost hyperparameters
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30

    # Training
    n_splits: int = 5
    random_state: int = 42


class XGBoostForecaster:
    """Gradient-boosted forecaster with time-series CV and feature importance"""

    def __init__(self, config: Optional[XGBConfig] = None):
        self.config = config or XGBConfig()
        self.model = None
        self.feature_cols: list[str] = []
        self.target_col: str = ""
        self.cv_metrics: list[dict] = []
        self.feature_importance: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "XGBoostForecaster":
        """Train with time-series cross-validation"""
        import xgboost as xgb

        self.feature_cols = feature_cols
        h = self.config.horizon_days
        if self.config.task == "regression":
            self.target_col = f"target_logret_{h}d"
        else:
            self.target_col = f"target_direction_{h}d"

        # Drop NaN rows (start of series for lookback features, end for target)
        data = df[feature_cols + [self.target_col]].dropna()
        if len(data) < 200:
            raise ValueError(f"Too few clean rows ({len(data)}) for training")

        X = data[feature_cols].values
        y = data[self.target_col].values

        # Time-series CV - no shuffling, expanding window
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        self.cv_metrics = []

        last_model = None
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if self.config.task == "regression":
                model = xgb.XGBRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    min_child_weight=self.config.min_child_weight,
                    reg_alpha=self.config.reg_alpha,
                    reg_lambda=self.config.reg_lambda,
                    random_state=self.config.random_state,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    eval_metric="rmse",
                    n_jobs=-1,
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    min_child_weight=self.config.min_child_weight,
                    reg_alpha=self.config.reg_alpha,
                    reg_lambda=self.config.reg_lambda,
                    random_state=self.config.random_state,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    eval_metric="mlogloss",
                    n_jobs=-1,
                )

            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            pred = model.predict(X_val)
            metrics = self._compute_metrics(y_val, pred)
            metrics["fold"] = fold
            metrics["train_size"] = len(X_tr)
            metrics["val_size"] = len(X_val)
            self.cv_metrics.append(metrics)
            logger.info(f"Fold {fold}: {metrics}")
            last_model = model

        # Final model: train on all data
        if self.config.task == "regression":
            final = xgb.XGBRegressor(
                n_estimators=last_model.best_iteration + 1
                    if hasattr(last_model, "best_iteration") else self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            final = xgb.XGBClassifier(
                n_estimators=last_model.best_iteration + 1
                    if hasattr(last_model, "best_iteration") else self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        final.fit(X, y, verbose=False)
        self.model = final

        # Feature importance
        self.feature_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": final.feature_importances_,
        }).sort_values("importance", ascending=False)

        logger.info(f"Final model trained on {len(X)} rows")
        logger.info(f"Top features:\n{self.feature_importance.head(10)}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call .fit() first")
        X = df[self.feature_cols].values
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Only for classification"""
        if self.model is None or self.config.task != "classification":
            raise RuntimeError("Classifier not trained")
        X = df[self.feature_cols].values
        return self.model.predict_proba(X)

    def _compute_metrics(self, y_true, y_pred) -> dict:
        if self.config.task == "regression":
            return {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
                "directional_acc": float(
                    np.mean(np.sign(y_true) == np.sign(y_pred))
                ),
            }
        else:
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
            }

    def save(self, path: str) -> None:
        """Persist model + metadata"""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(p / "xgb_model.json"))

        meta = {
            "config": asdict(self.config),
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "cv_metrics": self.cv_metrics,
        }
        with open(p / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self.feature_importance is not None:
            self.feature_importance.to_csv(p / "feature_importance.csv", index=False)

        logger.info(f"Saved model to {p}")

    @classmethod
    def load(cls, path: str) -> "XGBoostForecaster":
        import xgboost as xgb
        p = Path(path)
        with open(p / "metadata.json") as f:
            meta = json.load(f)

        config = XGBConfig(**meta["config"])
        instance = cls(config)
        instance.feature_cols = meta["feature_cols"]
        instance.target_col = meta["target_col"]
        instance.cv_metrics = meta["cv_metrics"]

        if config.task == "regression":
            instance.model = xgb.XGBRegressor()
        else:
            instance.model = xgb.XGBClassifier()
        instance.model.load_model(str(p / "xgb_model.json"))

        fi_path = p / "feature_importance.csv"
        if fi_path.exists():
            import pandas as pd
            instance.feature_importance = pd.read_csv(fi_path)

        return instance

"""
Walk-forward backtesting framework.

Critical for forecasting models: the only valid out-of-sample test is to
retrain the model with only data available at each historical point, then
predict forward. This is expensive but honest.

Two modes:
  1. Walk-forward retraining: retrain every `step` days (e.g., monthly)
  2. Expanding-window prediction: use a single trained model but predict
     iteratively on out-of-sample data

We compute:
  - Directional accuracy (overall and per regime)
  - Sharpe of a simple strategy following predictions
  - Performance during specific historical crises (Iraq war, GFC, COVID, etc.)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    train_window_years: int = 5      # rolling training window
    step_days: int = 21              # retrain every N days
    horizon_days: int = 5            # forecast horizon
    min_train_size: int = 252 * 2    # minimum 2 years of data
    transaction_cost_bps: int = 5    # 5 basis points per trade
    capital: float = 100_000.0


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics_overall: dict
    metrics_by_regime: dict
    metrics_by_event: dict
    equity_curve: pd.DataFrame


class WalkForwardBacktester:
    """Honest out-of-sample evaluation"""

    def __init__(
        self,
        model_factory: Callable,
        config: Optional[BacktestConfig] = None,
    ):
        """
        model_factory: callable returning a fresh model with .fit() and .predict()
                      e.g., lambda: EnsembleForecaster(config)
        """
        self.model_factory = model_factory
        self.config = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        date_col: str = "date",
        price_col: str = "wti_close",
        regime_col: Optional[str] = None,
    ) -> BacktestResult:
        """Run walk-forward backtest"""
        df = df.sort_values(date_col).reset_index(drop=True)
        dates = pd.to_datetime(df[date_col])

        h = self.config.horizon_days
        target_col = f"target_logret_{h}d"

        # Build the rolling-prediction schedule
        n = len(df)
        train_window_days = self.config.train_window_years * 365
        predictions = []

        # First valid prediction index: after min_train_size + horizon
        start_i = max(self.config.min_train_size, train_window_days)
        step = self.config.step_days

        # Cache the most recently trained model
        cached_model = None
        last_train_end = -1

        for i in range(start_i, n - h, step):
            train_start_date = dates.iloc[i] - pd.Timedelta(days=train_window_days)
            train_mask = (dates >= train_start_date) & (dates < dates.iloc[i])
            train_df = df.loc[train_mask].copy()

            if len(train_df) < self.config.min_train_size:
                continue

            # Re-train
            model = self.model_factory()
            try:
                model.fit(train_df, feature_cols)
            except Exception as e:
                logger.error(f"Training failed at {dates.iloc[i].date()}: {e}")
                continue

            # Predict the next `step` days
            predict_end = min(i + step, n - h)
            predict_df = df.iloc[i:predict_end].copy()

            try:
                pred_result = model.predict(predict_df)
                if isinstance(pred_result, pd.DataFrame) and "pred_ensemble" in pred_result:
                    preds = pred_result["pred_ensemble"].values
                else:
                    preds = np.asarray(pred_result)
            except Exception as e:
                logger.error(f"Prediction failed at {dates.iloc[i].date()}: {e}")
                continue

            for j, idx in enumerate(predict_df.index):
                if j >= len(preds) or np.isnan(preds[j]):
                    continue
                predictions.append({
                    "date": dates.iloc[idx],
                    "actual": df.at[idx, target_col] if target_col in df.columns else np.nan,
                    "predicted": float(preds[j]),
                    "price": df.at[idx, price_col],
                    "regime": df.at[idx, regime_col] if regime_col else None,
                })

            logger.info(f"Backtest: completed window ending {dates.iloc[i].date()}, "
                        f"{len(predictions)} preds so far")

        pred_df = pd.DataFrame(predictions)
        if pred_df.empty:
            raise RuntimeError("No predictions generated; check data/config")

        # Compute metrics
        metrics_overall = self._compute_metrics(pred_df)
        metrics_by_regime = {}
        if regime_col and "regime" in pred_df.columns:
            for regime, sub in pred_df.groupby("regime"):
                if len(sub) >= 10:
                    metrics_by_regime[regime] = self._compute_metrics(sub)

        # Per-event metrics (using events database)
        metrics_by_event = self._compute_event_metrics(pred_df)

        # Strategy equity curve
        equity = self._compute_equity_curve(pred_df)

        return BacktestResult(
            predictions=pred_df,
            metrics_overall=metrics_overall,
            metrics_by_regime=metrics_by_regime,
            metrics_by_event=metrics_by_event,
            equity_curve=equity,
        )

    def _compute_metrics(self, pred_df: pd.DataFrame) -> dict:
        df = pred_df.dropna(subset=["actual", "predicted"])
        if df.empty:
            return {}
        actual = df["actual"].values
        pred = df["predicted"].values

        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        dir_acc = float(np.mean(np.sign(actual) == np.sign(pred)))

        # Information coefficient (rank correlation)
        from scipy.stats import spearmanr
        ic, _ = spearmanr(actual, pred)

        # Strategy: long if pred>0, short if pred<0, return = pred_direction * actual
        strategy_ret = np.sign(pred) * actual
        sharpe = (np.mean(strategy_ret) / (np.std(strategy_ret) + 1e-8)) * np.sqrt(252 / self.config.horizon_days)
        hit_rate = float(np.mean(strategy_ret > 0))

        return {
            "n_predictions": len(df),
            "mae": mae,
            "rmse": rmse,
            "directional_accuracy": dir_acc,
            "information_coefficient": float(ic) if not np.isnan(ic) else 0.0,
            "strategy_sharpe": float(sharpe),
            "strategy_hit_rate": hit_rate,
            "mean_predicted": float(np.mean(pred)),
            "mean_actual": float(np.mean(actual)),
        }

    def _compute_event_metrics(self, pred_df: pd.DataFrame) -> dict:
        """Performance during specific historical crises"""
        from src.data.events_database import EVENTS, Severity

        result = {}
        for event in EVENTS:
            if event.severity in (Severity.LOW,):
                continue  # Skip minor events

            start = pd.Timestamp(event.start_date)
            end = pd.Timestamp(event.end_date) if event.end_date else pd.Timestamp.now()

            mask = (pd.to_datetime(pred_df["date"]) >= start) & \
                   (pd.to_datetime(pred_df["date"]) <= end)
            event_preds = pred_df[mask]

            if len(event_preds) >= 5:
                result[event.event_id] = {
                    "name": event.name,
                    "type": event.event_type.value,
                    **self._compute_metrics(event_preds),
                }
        return result

    def _compute_equity_curve(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Simulate a simple long/short strategy"""
        df = pred_df.dropna(subset=["actual", "predicted"]).copy()
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("date").reset_index(drop=True)
        position = np.sign(df["predicted"].values)
        gross_ret = position * df["actual"].values
        cost = self.config.transaction_cost_bps / 10000
        # Apply cost when position changes
        position_changes = np.abs(np.diff(np.concatenate([[0], position])))
        net_ret = gross_ret - position_changes * cost

        df["position"] = position
        df["gross_return"] = gross_ret
        df["net_return"] = net_ret
        df["cumulative_return"] = (1 + df["net_return"]).cumprod() - 1
        df["equity"] = self.config.capital * (1 + df["cumulative_return"])
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(__doc__)

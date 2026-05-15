"""
Inference pipeline.

Loads a trained ensemble model + feature dataset, computes the latest features,
and produces a forecast with LLM interpretation.
"""
from __future__ import annotations
import logging
from datetime import date
from pathlib import Path
from typing import Optional
import json

import pandas as pd
import numpy as np

from src.models.ensemble import EnsembleForecaster, Forecast
from src.features.builder import build_feature_matrix, get_feature_columns
from src.data.loader import OilDataLoader, DataConfig
from src.llm.interpreter import ForecastInterpreter, LLMInterpretation
from src.data.events_database import EVENTS, get_events_in_range

logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end inference: data -> features -> forecast -> LLM"""

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts = Path(artifacts_dir)
        self.ensemble: Optional[EnsembleForecaster] = None
        self.metadata: dict = {}
        self.feature_dataset: Optional[pd.DataFrame] = None
        self.interpreter = ForecastInterpreter()

    def load(self) -> "InferencePipeline":
        """Load trained model and metadata"""
        if not (self.artifacts / "ensemble").exists():
            raise FileNotFoundError(f"No trained model at {self.artifacts}/ensemble")

        self.ensemble = EnsembleForecaster.load(str(self.artifacts / "ensemble"))

        meta_path = self.artifacts / "training_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)

        ds_path = self.artifacts / "feature_dataset.parquet"
        if ds_path.exists():
            self.feature_dataset = pd.read_parquet(ds_path)

        logger.info(f"Loaded model + metadata from {self.artifacts}")
        return self

    def refresh_data(self) -> pd.DataFrame:
        """Pull latest data and rebuild features for inference"""
        loader = OilDataLoader(DataConfig(start_date=date(2000, 1, 1)))
        raw = loader.build_master_dataset(refresh=True)

        horizon = self.metadata.get("horizon_days", 5)
        feats = build_feature_matrix(raw, horizon_days=horizon, include_events=True)
        feats = feats.dropna(subset=["wti_close"]).reset_index(drop=True)

        # Use the same feature columns as training
        for c in self.metadata.get("feature_cols", []):
            if c not in feats.columns:
                feats[c] = 0
            feats[c] = feats[c].fillna(method="ffill").fillna(0)

        self.feature_dataset = feats
        return feats

    def predict_latest(self, with_llm: bool = True) -> dict:
        """Generate a forecast for the latest available date"""
        if self.ensemble is None:
            raise RuntimeError("Call .load() first")

        df = self.feature_dataset
        if df is None or df.empty:
            df = self.refresh_data()

        # Forecast on the latest row
        forecast = self.ensemble.forecast_single(df.tail(100))
        forecast_dict = {
            "forecast_date": forecast.forecast_date.isoformat() if hasattr(forecast.forecast_date, "isoformat") else str(forecast.forecast_date),
            "horizon_days": forecast.horizon_days,
            "predicted_log_return": forecast.predicted_log_return,
            "predicted_pct_return": forecast.predicted_pct_return,
            "direction": forecast.direction,
            "confidence": forecast.confidence,
            "model_predictions": forecast.model_predictions,
        }

        # Get context for LLM
        latest = df.iloc[-1]
        current_price = float(latest.get("wti_close", 0))
        regime = str(latest.get("regime_name", "unknown"))

        # Active events today
        active = get_events_in_range(date.today(), date.today())
        active_events = [e.to_dict() for e in active]

        # Top features
        importance = []
        if self.ensemble.xgb and self.ensemble.xgb.feature_importance is not None:
            top = self.ensemble.xgb.feature_importance.head(10)
            importance = top.to_dict("records")

        # Project current row features into a simple dict
        feature_snapshot = {
            k: float(v) if not pd.isna(v) else None
            for k, v in latest.items()
            if k in {"wti_close", "vol_21d", "drawdown_1y", "rsi_14",
                     "vix_level", "dxy_ret_21d"}
        }

        result = {
            "forecast": forecast_dict,
            "current_price": current_price,
            "regime": regime,
            "active_events": active_events,
            "feature_snapshot": feature_snapshot,
            "model_top_features": importance,
        }

        if with_llm:
            logger.info("Calling LLM for interpretation...")
            interp = self.interpreter.interpret(
                forecast=forecast_dict,
                current_features=feature_snapshot,
                active_events=active_events,
                recent_regime=regime,
                feature_importance_top10=importance,
            )
            result["interpretation"] = {
                "forecast_summary": interp.forecast_summary,
                "primary_drivers": interp.primary_drivers,
                "historical_analogs": interp.historical_analogs,
                "risk_factors": interp.risk_factors,
                "confidence_qualifier": interp.confidence_qualifier,
            }

        return result

    def stress_test(self, scenario: str) -> str:
        if self.feature_dataset is None or self.feature_dataset.empty:
            self.refresh_data()
        latest = self.feature_dataset.iloc[-1]

        active = get_events_in_range(date.today(), date.today())
        return self.interpreter.stress_test(
            scenario=scenario,
            current_price=float(latest.get("wti_close", 0)),
            current_features={"vol_21d": float(latest.get("vol_21d", 0))},
            active_events=[e.to_dict() for e in active],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = InferencePipeline()
    pipeline.load()
    result = pipeline.predict_latest()
    print(json.dumps(result, indent=2, default=str))

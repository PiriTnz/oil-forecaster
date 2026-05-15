"""
End-to-end training pipeline.

Steps:
  1. Load historical data (2000-now)
  2. Build feature matrix with events
  3. Fit regime detector
  4. Train ensemble forecaster with walk-forward CV
  5. Backtest on historical crisis events
  6. Log everything to MLflow
  7. Save artifacts for serving

Run: python -m src.pipelines.train
"""
from __future__ import annotations
import logging
import json
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.data.loader import OilDataLoader, DataConfig
from src.features.builder import build_feature_matrix, get_feature_columns
from src.models.regime_detector import HMMRegimeDetector, RegimeConfig, EventBasedRegimeLabeler
from src.models.ensemble import EnsembleForecaster, EnsembleConfig
from src.evaluation.backtester import WalkForwardBacktester, BacktestConfig

logger = logging.getLogger(__name__)


def run_training_pipeline(
    horizon_days: int = 5,
    artifacts_dir: str = "artifacts",
    use_mlflow: bool = True,
    skip_backtest: bool = False,
) -> dict:
    """Run the full training pipeline and return metrics"""
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Setup MLflow if available
    mlflow_run = None
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("oil-forecaster")
            mlflow_run = mlflow.start_run()
            mlflow.log_params({
                "horizon_days": horizon_days,
                "data_start": "2000-01-01",
                "data_end": date.today().isoformat(),
            })
        except Exception as e:
            logger.warning(f"MLflow disabled: {e}")

    metrics = {}

    try:
        # 1. Data
        logger.info("=== Step 1: Load data ===")
        loader = OilDataLoader(DataConfig(start_date=date(2000, 1, 1)))
        df_raw = loader.build_master_dataset()
        logger.info(f"Loaded {len(df_raw)} rows, {df_raw['date'].min().date()} to {df_raw['date'].max().date()}")

        if use_mlflow and mlflow_run:
            import mlflow
            mlflow.log_metric("data_rows", len(df_raw))

        # 2. Features
        logger.info("=== Step 2: Build features ===")
        df = build_feature_matrix(df_raw, horizon_days=horizon_days, include_events=True)
        df = df.dropna(subset=["wti_close"]).reset_index(drop=True)
        feature_cols = get_feature_columns(df)
        logger.info(f"Features: {len(feature_cols)} columns, {len(df)} rows")

        # 3. Regime detection
        logger.info("=== Step 3: Fit regime detector ===")
        regime_features = ("ret_1d", "vol_21d", "drawdown_1y", "vix_level")
        # Drop any missing in regime features
        regime_features_present = tuple(f for f in regime_features if f in df.columns)
        if len(regime_features_present) >= 2:
            regime_clean = df.dropna(subset=list(regime_features_present))
            detector = HMMRegimeDetector(RegimeConfig(
                n_regimes=4,
                features=regime_features_present,
            ))
            detector.fit(regime_clean)
            df_regimes = detector.predict(df)
            df["regime_name"] = df_regimes["regime_name"]
            df["regime_id"] = df_regimes["regime_id"]

            # Also add event-based labels
            df = EventBasedRegimeLabeler().label(df)

            logger.info(f"Regime distribution:\n{df['regime_name'].value_counts()}")
        else:
            logger.warning("Insufficient features for regime detection")
            df["regime_name"] = "unknown"

        # 4. Train ensemble
        logger.info("=== Step 4: Train ensemble ===")
        # Drop rows with NaN target or critical features
        train_data = df.dropna(subset=[f"target_logret_{horizon_days}d"])
        # Sane fill for any remaining NaN features
        for c in feature_cols:
            if c in train_data.columns:
                train_data[c] = train_data[c].fillna(method="ffill").fillna(0)

        ensemble_cfg = EnsembleConfig(
            horizon_days=horizon_days,
            use_xgb=True,
            use_lstm=False,  # disable for fast first training; enable later
        )
        ensemble = EnsembleForecaster(ensemble_cfg)
        ensemble.fit(train_data, feature_cols)
        ensemble.save(str(artifacts / "ensemble"))
        logger.info("Ensemble trained and saved")

        if ensemble.xgb and ensemble.xgb.cv_metrics:
            cv = ensemble.xgb.cv_metrics[-1]
            metrics["xgb_final_fold"] = cv
            if use_mlflow and mlflow_run:
                import mlflow
                for k, v in cv.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"xgb_cv_{k}", v)

        # 5. Backtest
        if not skip_backtest:
            logger.info("=== Step 5: Walk-forward backtest ===")
            # For speed, limit backtest data; in production run full
            bt_data = train_data.tail(252 * 8).reset_index(drop=True)  # last 8 years
            backtester = WalkForwardBacktester(
                model_factory=lambda: EnsembleForecaster(ensemble_cfg),
                config=BacktestConfig(
                    train_window_years=4,
                    step_days=63,  # retrain quarterly for speed
                    horizon_days=horizon_days,
                ),
            )
            try:
                bt_result = backtester.run(
                    bt_data, feature_cols,
                    regime_col="regime_name",
                )
                metrics["backtest_overall"] = bt_result.metrics_overall
                metrics["backtest_by_regime"] = bt_result.metrics_by_regime
                metrics["backtest_by_event"] = bt_result.metrics_by_event

                # Save artifacts
                bt_result.predictions.to_csv(artifacts / "backtest_predictions.csv", index=False)
                bt_result.equity_curve.to_csv(artifacts / "backtest_equity.csv", index=False)

                with open(artifacts / "backtest_metrics.json", "w") as f:
                    json.dump({
                        "overall": metrics["backtest_overall"],
                        "by_regime": metrics["backtest_by_regime"],
                        "by_event": metrics["backtest_by_event"],
                    }, f, indent=2, default=str)

                logger.info(f"Backtest overall: {metrics['backtest_overall']}")

                if use_mlflow and mlflow_run:
                    import mlflow
                    for k, v in metrics["backtest_overall"].items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"bt_{k}", v)
            except Exception as e:
                logger.error(f"Backtest failed: {e}", exc_info=True)

        # Save the feature dataset for inference
        df.to_parquet(artifacts / "feature_dataset.parquet", index=False)

        # Save metadata
        with open(artifacts / "training_metadata.json", "w") as f:
            json.dump({
                "trained_at": date.today().isoformat(),
                "horizon_days": horizon_days,
                "n_train_rows": len(train_data),
                "feature_cols": feature_cols,
                "metrics": metrics,
            }, f, indent=2, default=str)

        logger.info(f"=== Pipeline complete. Artifacts in {artifacts}/ ===")
        return metrics

    finally:
        if use_mlflow and mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in days")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-backtest", action="store_true")
    args = parser.parse_args()

    run_training_pipeline(
        horizon_days=args.horizon,
        artifacts_dir=args.artifacts,
        use_mlflow=not args.no_mlflow,
        skip_backtest=args.no_backtest,
    )

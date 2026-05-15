"""
QUICKSTART: end-to-end demo

Run this to verify everything works locally before deploying.

Usage:
    python scripts/quickstart_demo.py
"""
import logging
import sys
from pathlib import Path

# Allow running without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.events_database import EVENTS, EventType
from src.data.loader import OilDataLoader, DataConfig
from src.features.builder import build_feature_matrix, get_feature_columns
from src.models.regime_detector import HMMRegimeDetector, RegimeConfig
from src.models.xgb_model import XGBoostForecaster, XGBConfig
from datetime import date


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 70)
    print(" OIL FORECASTER - QUICKSTART DEMO")
    print("=" * 70)

    # 1. Events database
    print(f"\n[1] Events database loaded: {len(EVENTS)} events")
    war_events = [e for e in EVENTS if e.event_type == EventType.WAR]
    print(f"    Wars tracked: {len(war_events)}")
    for e in war_events[:5]:
        print(f"      • {e.start_date}: {e.name} ({e.severity.value})")

    # 2. Load historical data
    print("\n[2] Loading historical data (2000-now)...")
    print("    (This may take a minute on first run...)")
    loader = OilDataLoader(DataConfig(start_date=date(2010, 1, 1)))  # shorter for demo
    df = loader.build_master_dataset()
    print(f"    Loaded: {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"    Columns: {list(df.columns)[:10]}...")

    # 3. Feature engineering
    print("\n[3] Building features (this includes event labeling)...")
    feats = build_feature_matrix(df, horizon_days=5, include_events=True)
    feats = feats.dropna(subset=["wti_close"]).reset_index(drop=True)
    cols = get_feature_columns(feats)
    print(f"    Feature columns: {len(cols)}")
    print(f"    Sample features: {cols[:10]}")

    # 4. Regime detection
    print("\n[4] Detecting market regimes...")
    regime_features = ("ret_1d", "vol_21d", "drawdown_1y", "vix_level")
    detector = HMMRegimeDetector(RegimeConfig(n_regimes=4, features=regime_features))
    clean_for_regime = feats.dropna(subset=list(regime_features))
    detector.fit(clean_for_regime)
    pred = detector.predict(feats)
    print(f"    Regime distribution:")
    print(pred["regime_name"].value_counts().to_string())

    # 5. Train forecaster
    print("\n[5] Training XGBoost forecaster (this takes ~30 seconds)...")
    train_data = feats.dropna(subset=["target_logret_5d"]).copy()
    for c in cols:
        if c in train_data.columns:
            train_data[c] = train_data[c].fillna(method="ffill").fillna(0)

    model = XGBoostForecaster(XGBConfig(
        horizon_days=5, task="regression",
        n_estimators=200, max_depth=5, n_splits=3,
    ))
    model.fit(train_data, cols)

    print("\n    Cross-validation metrics:")
    for fold in model.cv_metrics:
        print(f"      Fold {fold['fold']}: "
              f"MAE={fold.get('mae', 0):.5f}, "
              f"directional_acc={fold.get('directional_acc', 0):.3f}, "
              f"R²={fold.get('r2', 0):.3f}")

    print("\n    Top 10 most important features:")
    print(model.feature_importance.head(10).to_string(index=False))

    # 6. Make a forecast
    print("\n[6] Making forecast for the latest available date...")
    latest = train_data.tail(1)
    pred_val = float(model.predict(latest)[0])
    current_price = float(latest["wti_close"].iloc[0])
    projected = current_price * (1 + pred_val)
    direction = "UP" if pred_val > 0 else "DOWN"

    print(f"\n    Latest date: {latest['date'].iloc[0]}")
    print(f"    Current WTI: ${current_price:.2f}")
    print(f"    Predicted 5-day return: {pred_val*100:+.2f}%")
    print(f"    Projected price in 5 days: ${projected:.2f} ({direction})")

    print("\n" + "=" * 70)
    print(" Demo complete!")
    print("\n Next steps:")
    print("   • Run full training: python -m src.pipelines.train")
    print("   • Start API:         uvicorn src.api.main:app --reload")
    print("   • View dashboard:    http://localhost:8000/static/index.html")
    print("=" * 70)


if __name__ == "__main__":
    main()

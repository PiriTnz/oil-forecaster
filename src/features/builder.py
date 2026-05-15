"""
Feature engineering for oil price forecasting.

Three families of features:
  1. PRICE features    -- returns, momentum, volatility, technical indicators
  2. MACRO features    -- USD strength, equity risk-on/off, yield environment
  3. REGIME features   -- event-based labels from geopolitical database

All features are constructed to be CAUSAL (no leakage from the future).
"""
from __future__ import annotations
import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_return_features(df: pd.DataFrame, price_col: str = "wti_close") -> pd.DataFrame:
    """Add log-return and percentage-return features at multiple horizons"""
    out = df.copy()
    log_p = np.log(out[price_col])

    # Single-day returns
    out["ret_1d"] = log_p.diff()
    out["ret_pct_1d"] = out[price_col].pct_change()

    # Multi-day cumulative returns
    for n in [5, 10, 21, 63, 252]:  # week, 2w, month, quarter, year
        out[f"ret_{n}d"] = log_p.diff(n)

    return out


def add_volatility_features(df: pd.DataFrame, price_col: str = "wti_close") -> pd.DataFrame:
    """Realized volatility at multiple windows + GARCH-like signals"""
    out = df.copy()
    log_p = np.log(out[price_col])
    rets = log_p.diff()

    # Rolling realized volatility (annualized)
    for window in [5, 10, 21, 63]:
        out[f"vol_{window}d"] = rets.rolling(window).std() * np.sqrt(252)

    # Vol-of-vol (instability indicator - high during crises)
    out["vol_of_vol_21d"] = out["vol_21d"].rolling(21).std()

    # Drawdown from rolling max
    rolling_max = out[price_col].rolling(252, min_periods=1).max()
    out["drawdown_1y"] = out[price_col] / rolling_max - 1

    return out


def add_momentum_features(df: pd.DataFrame, price_col: str = "wti_close") -> pd.DataFrame:
    """Momentum and mean-reversion signals"""
    out = df.copy()

    # SMA crosses
    out["sma_20"] = out[price_col].rolling(20).mean()
    out["sma_50"] = out[price_col].rolling(50).mean()
    out["sma_200"] = out[price_col].rolling(200).mean()
    out["sma_20_50_diff"] = (out["sma_20"] - out["sma_50"]) / out["sma_50"]
    out["sma_50_200_diff"] = (out["sma_50"] - out["sma_200"]) / out["sma_200"]
    out["price_vs_sma200"] = out[price_col] / out["sma_200"] - 1

    # RSI (Wilder's, 14-day)
    delta = out[price_col].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Band position (-1 to +1)
    mid = out[price_col].rolling(20).mean()
    sd = out[price_col].rolling(20).std()
    out["bb_position"] = (out[price_col] - mid) / (2 * sd)

    return out


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-asset features capturing macro regime"""
    out = df.copy()

    # USD strength (oil is priced in USD - inverse relationship typical)
    if "dxy_close" in out.columns:
        out["dxy_ret_5d"] = np.log(out["dxy_close"]).diff(5)
        out["dxy_ret_21d"] = np.log(out["dxy_close"]).diff(21)

    # Equity risk-on/off
    if "spx_close" in out.columns:
        out["spx_ret_5d"] = np.log(out["spx_close"]).diff(5)
        out["spx_vol_21d"] = np.log(out["spx_close"]).diff().rolling(21).std() * np.sqrt(252)

    # VIX - fear gauge
    if "vix_close" in out.columns:
        out["vix_level"] = out["vix_close"]
        out["vix_change_5d"] = out["vix_close"].diff(5)
        out["vix_regime_high"] = (out["vix_close"] > 25).astype(int)

    # Gold/Oil ratio (safe-haven shift)
    if "gold_close" in out.columns and "wti_close" in out.columns:
        out["gold_oil_ratio"] = out["gold_close"] / out["wti_close"]
        out["gold_oil_ratio_z"] = (
            out["gold_oil_ratio"] - out["gold_oil_ratio"].rolling(252).mean()
        ) / out["gold_oil_ratio"].rolling(252).std()

    # Yield curve proxy
    if "tnx_close" in out.columns:
        out["tnx_level"] = out["tnx_close"]
        out["tnx_change_21d"] = out["tnx_close"].diff(21)

    return out


def add_event_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add features from the geopolitical events database.

    For each date, we encode:
      - Whether each event type is currently active
      - Days since last major event of each type
      - Aggregate severity score of active events
    """
    from src.data.events_database import EVENTS, EventType, Severity

    out = df.copy()
    dates = pd.to_datetime(out[date_col]).dt.date.tolist()

    # Severity numeric mapping
    sev_score = {Severity.LOW: 1, Severity.MEDIUM: 2,
                 Severity.HIGH: 3, Severity.EXTREME: 4}

    event_types_to_track = [
        EventType.WAR, EventType.SANCTIONS, EventType.SUPPLY_DISRUPTION,
        EventType.OPEC_ACTION, EventType.FINANCIAL_CRISIS, EventType.PANDEMIC,
    ]

    # Initialize columns
    for et in event_types_to_track:
        out[f"event_active_{et.value}"] = 0
        out[f"days_since_{et.value}"] = 9999

    out["event_severity_score"] = 0
    out["event_count_active"] = 0
    out["event_bullish_count"] = 0
    out["event_bearish_count"] = 0

    for i, d in enumerate(dates):
        active_severity = 0
        n_active = 0
        bullish = 0
        bearish = 0

        for e in EVENTS:
            e_end = e.end_date or date.today()
            # Active on this date?
            if e.start_date <= d <= e_end:
                col = f"event_active_{e.event_type.value}"
                if col in out.columns:
                    out.at[out.index[i], col] = 1
                active_severity = max(active_severity, sev_score.get(e.severity, 0))
                n_active += 1
                if e.impact_direction.value == "bullish":
                    bullish += 1
                elif e.impact_direction.value == "bearish":
                    bearish += 1

            # Days since this event of this type
            if e.start_date <= d:
                col = f"days_since_{e.event_type.value}"
                if col in out.columns:
                    days = (d - e.start_date).days
                    out.at[out.index[i], col] = min(out.at[out.index[i], col], days)

        out.at[out.index[i], "event_severity_score"] = active_severity
        out.at[out.index[i], "event_count_active"] = n_active
        out.at[out.index[i], "event_bullish_count"] = bullish
        out.at[out.index[i], "event_bearish_count"] = bearish

    return out


def add_seasonality_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Calendar features: seasonality matters for oil (summer driving, winter heating)"""
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["month"] = dt.dt.month
    out["dayofweek"] = dt.dt.dayofweek
    out["quarter"] = dt.dt.quarter
    out["year"] = dt.dt.year

    # Cyclical encoding (better than raw int for ML)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 5)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 5)

    return out


def make_target(
    df: pd.DataFrame,
    price_col: str = "wti_close",
    horizon_days: int = 5,
    log_return: bool = True,
) -> pd.DataFrame:
    """
    Create the prediction target: forward return over `horizon_days`.

    CRITICAL: target uses future data. When training, drop the last `horizon_days`
    rows. When predicting, this column is NaN for the current observation.
    """
    out = df.copy()
    if log_return:
        out[f"target_logret_{horizon_days}d"] = (
            np.log(out[price_col].shift(-horizon_days)) - np.log(out[price_col])
        )
    else:
        out[f"target_pctret_{horizon_days}d"] = (
            out[price_col].shift(-horizon_days) / out[price_col] - 1
        )

    # Classification target: direction (-1, 0, +1) with a small dead zone
    fwd_ret = out[price_col].shift(-horizon_days) / out[price_col] - 1
    out[f"target_direction_{horizon_days}d"] = np.where(
        fwd_ret > 0.01, 1,  # >1% up
        np.where(fwd_ret < -0.01, -1, 0)
    )

    return out


def build_feature_matrix(
    df: pd.DataFrame,
    horizon_days: int = 5,
    include_events: bool = True,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Returns a DataFrame with all features + target. Drop NaN rows before training.
    """
    logger.info(f"Building features for {len(df)} rows")
    out = df.copy()
    out = add_return_features(out)
    out = add_volatility_features(out)
    out = add_momentum_features(out)
    out = add_macro_features(out)
    out = add_seasonality_features(out)
    if include_events:
        out = add_event_features(out)
    out = make_target(out, horizon_days=horizon_days)

    logger.info(f"Feature matrix: {out.shape}, {len(out.columns)} columns")
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of column names that are features (exclude target/identifiers)"""
    exclude = {"date", "symbol"}
    feature_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("target_"):
            continue
        # Raw OHLCV are inputs to features, not features themselves;
        # we keep the close as a level reference but the model uses returns
        if c.endswith(("_open", "_high", "_low", "_vol")):
            continue
        feature_cols.append(c)
    return feature_cols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Demo with a synthetic example
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="B")
    n = len(dates)
    prices = 50 + np.cumsum(np.random.normal(0, 1, n))
    df = pd.DataFrame({
        "date": dates,
        "wti_close": prices,
        "brent_close": prices + 3,
        "dxy_close": 95 + np.cumsum(np.random.normal(0, 0.1, n)),
        "spx_close": 3000 + np.cumsum(np.random.normal(1, 5, n)),
        "vix_close": np.abs(20 + np.random.normal(0, 5, n)),
        "gold_close": 1500 + np.cumsum(np.random.normal(0.5, 2, n)),
        "tnx_close": 2 + np.random.normal(0, 0.05, n),
    })
    feat = build_feature_matrix(df)
    print(f"Feature shape: {feat.shape}")
    print(f"Features: {get_feature_columns(feat)[:20]}...")

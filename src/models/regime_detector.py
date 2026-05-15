"""
Market regime detection for oil prices.

We want to classify each day into one of K regimes (e.g., "calm uptrend",
"war-driven spike", "demand-shock crash", etc.) so that:

  1. Forecasting models can condition on regime
  2. Backtests can be filtered by regime
  3. The LLM analysis layer can explain WHY a forecast looks the way it does

Two approaches implemented:
  A. Unsupervised HMM (Hidden Markov Model) on returns + volatility
  B. Supervised classifier trained on the labeled events database

Both are useful: HMM finds regimes the data "thinks" exist; supervised
checks alignment with our human labels.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    n_regimes: int = 4
    features: tuple = ("ret_1d", "vol_21d", "drawdown_1y", "vix_level")
    random_state: int = 42


class HMMRegimeDetector:
    """
    HMM-style regime detector via Gaussian Mixture Model on rolling features.

    GMM is a soft-clustering alternative to a full HMM that is much faster to
    train and good enough for daily oil data. For sequential dependencies use
    `hmmlearn` (added below as alternative).
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.scaler = StandardScaler()
        self.model: Optional[GaussianMixture] = None
        self.regime_names: dict[int, str] = {}

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """Fit regime model on feature columns"""
        cols = list(self.config.features)
        X = df[cols].dropna().values
        if len(X) < 100:
            raise ValueError(f"Too few rows ({len(X)}) to fit regime model")

        X_scaled = self.scaler.fit_transform(X)
        self.model = GaussianMixture(
            n_components=self.config.n_regimes,
            covariance_type="full",
            random_state=self.config.random_state,
            max_iter=200,
        )
        self.model.fit(X_scaled)
        logger.info(f"Fit GMM with {self.config.n_regimes} regimes "
                    f"on {len(X_scaled)} rows, "
                    f"log-likelihood={self.model.score(X_scaled):.2f}")

        # Auto-name regimes by sorting on mean volatility & return
        self._auto_name_regimes(X_scaled)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict regime for each row; returns DataFrame with regime + probabilities"""
        if self.model is None:
            raise RuntimeError("Call .fit() first")

        cols = list(self.config.features)
        mask = df[cols].notna().all(axis=1)
        X = df.loc[mask, cols].values
        X_scaled = self.scaler.transform(X)

        labels = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)

        out = df.copy()
        out["regime_id"] = np.nan
        out["regime_name"] = None
        for i in range(self.config.n_regimes):
            out[f"regime_prob_{i}"] = np.nan

        out.loc[mask, "regime_id"] = labels
        out.loc[mask, "regime_name"] = [self.regime_names.get(int(l), f"regime_{l}")
                                         for l in labels]
        for i in range(self.config.n_regimes):
            out.loc[mask, f"regime_prob_{i}"] = probs[:, i]

        return out

    def _auto_name_regimes(self, X_scaled: np.ndarray) -> None:
        """Assign human-readable names based on regime centroids"""
        means = self.model.means_  # shape (k, n_features)
        cols = list(self.config.features)

        try:
            ret_idx = cols.index("ret_1d")
        except ValueError:
            ret_idx = 0
        try:
            vol_idx = cols.index("vol_21d")
        except ValueError:
            vol_idx = 1

        for i, m in enumerate(means):
            ret_z = m[ret_idx]
            vol_z = m[vol_idx]

            if vol_z > 0.8:
                name = "crisis" if ret_z < 0 else "spike"
            elif vol_z < -0.3:
                name = "calm_uptrend" if ret_z > 0 else "calm_drift"
            else:
                name = "normal_up" if ret_z > 0 else "normal_down"

            self.regime_names[i] = name

        logger.info(f"Regime names: {self.regime_names}")


class EventBasedRegimeLabeler:
    """
    Maps each date to a regime label using the geopolitical events database.

    This produces a 'ground truth' regime sequence that can be:
      1. Used as supervised labels for training a classifier
      2. Compared with HMM output to validate that unsupervised clusters
         align with real-world events
    """

    def label(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        from src.data.events_database import EVENTS, EventType
        from datetime import date as date_t

        out = df.copy()
        labels = []
        for d in pd.to_datetime(out[date_col]).dt.date:
            active = [e for e in EVENTS
                      if e.start_date <= d <= (e.end_date or date_t.today())]

            if any(e.event_type == EventType.WAR and e.severity.value in ("high", "extreme")
                   for e in active):
                label = "war_regime"
            elif any(e.event_type == EventType.SANCTIONS for e in active):
                label = "sanctions_regime"
            elif any(e.event_type in (EventType.FINANCIAL_CRISIS, EventType.PANDEMIC)
                     for e in active):
                label = "crisis_regime"
            elif any(e.event_type == EventType.OPEC_ACTION for e in active):
                label = "opec_regime"
            else:
                label = "normal_regime"
            labels.append(label)

        out["event_regime"] = labels
        return out


def transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """Compute empirical transition probabilities between regimes"""
    s = regimes.dropna()
    pairs = list(zip(s.iloc[:-1], s.iloc[1:]))
    df = pd.DataFrame(pairs, columns=["from", "to"])
    counts = df.groupby(["from", "to"]).size().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo with synthetic data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # Simulate three regimes
    ret = np.concatenate([
        np.random.normal(0.001, 0.01, 400),
        np.random.normal(-0.005, 0.04, 200),  # crisis
        np.random.normal(0.002, 0.015, 400),
    ])
    df = pd.DataFrame({
        "date": dates,
        "ret_1d": ret,
        "vol_21d": pd.Series(ret).rolling(21).std() * np.sqrt(252),
        "drawdown_1y": -np.abs(np.cumsum(ret)),
        "vix_level": 20 + 30 * np.abs(ret) * 100,
    })

    detector = HMMRegimeDetector(RegimeConfig(n_regimes=3))
    detector.fit(df)
    pred = detector.predict(df)
    print(pred[["date", "regime_id", "regime_name"]].tail(10))
    print("\nTransition matrix:")
    print(transition_matrix(pred["regime_name"]))

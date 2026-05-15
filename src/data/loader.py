"""
Historical oil price data loader.

Sources (in order of preference):
  1. EIA (US Energy Information Administration) - free, authoritative, daily since 1986
  2. Yahoo Finance - free, daily futures
  3. FRED (Federal Reserve) - free, weekly/monthly
  4. CSV cache - local persistence

We pull ~25 years of daily data for WTI (CL=F) and Brent (BZ=F) plus related
indicators (USD index, S&P 500, VIX, US 10Y yield) used as features.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Symbol catalog
# =============================================================================
PRICE_SYMBOLS = {
    "WTI": "CL=F",       # WTI Crude Futures
    "BRENT": "BZ=F",     # Brent Crude Futures
}

MACRO_SYMBOLS = {
    "DXY": "DX-Y.NYB",   # US Dollar Index
    "SPX": "^GSPC",      # S&P 500
    "VIX": "^VIX",       # Volatility index
    "GOLD": "GC=F",      # Gold (alternative safe haven)
    "TNX": "^TNX",       # 10-year Treasury yield
}

EIA_SERIES = {
    "WTI_SPOT": "PET.RWTC.D",       # Cushing WTI spot price
    "BRENT_SPOT": "PET.RBRTE.D",    # Europe Brent spot
    "US_INVENTORY": "PET.WCRSTUS1.W",  # US crude inventory (weekly)
    "OPEC_PROD": "STEO.COPR_OPEC.M",   # OPEC production (monthly)
}


@dataclass
class DataConfig:
    start_date: date = date(2000, 1, 1)
    end_date: date = date.today()
    cache_dir: Path = Path("data/raw")
    cache_max_age_hours: int = 24


class OilDataLoader:
    """Load and cache historical price data from multiple sources"""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Yahoo Finance
    # ------------------------------------------------------------------
    def fetch_yahoo(self, symbol: str, refresh: bool = False) -> pd.DataFrame:
        """Fetch daily OHLCV from Yahoo Finance with disk caching"""
        cache_path = self.config.cache_dir / f"yahoo_{symbol.replace('=', '_').replace('^', '')}.parquet"

        if not refresh and self._cache_is_fresh(cache_path):
            logger.info(f"Loading {symbol} from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        try:
            import yfinance as yf
            logger.info(f"Fetching {symbol} from Yahoo Finance "
                        f"({self.config.start_date} to {self.config.end_date})")
            df = yf.download(
                symbol,
                start=self.config.start_date,
                end=self.config.end_date,
                progress=False,
                auto_adjust=False,
            )
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Flatten MultiIndex columns if present (newer yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df["symbol"] = symbol
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Yahoo fetch failed for {symbol}: {e}")
            if cache_path.exists():
                logger.info("Falling back to stale cache")
                return pd.read_parquet(cache_path)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # EIA (most authoritative for oil)
    # ------------------------------------------------------------------
    def fetch_eia(self, series_id: str, api_key: Optional[str] = None,
                  refresh: bool = False) -> pd.DataFrame:
        """Fetch from EIA API (https://www.eia.gov/opendata/)"""
        cache_path = self.config.cache_dir / f"eia_{series_id.replace('.', '_')}.parquet"

        if not refresh and self._cache_is_fresh(cache_path):
            return pd.read_parquet(cache_path)

        if not api_key:
            import os
            api_key = os.getenv("EIA_API_KEY", "")

        if not api_key:
            logger.warning(f"No EIA API key, skipping {series_id}")
            return pd.DataFrame()

        try:
            import requests
            url = f"https://api.eia.gov/v2/seriesid/{series_id}"
            params = {"api_key": api_key}
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            rows = data.get("response", {}).get("data", [])
            if not rows:
                logger.warning(f"EIA returned no data for {series_id}")
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df["period"] = pd.to_datetime(df["period"])
            df = df.rename(columns={"period": "date", "value": "price"})
            df = df.sort_values("date").reset_index(drop=True)
            df["series_id"] = series_id
            df.to_parquet(cache_path, index=False)
            logger.info(f"EIA: cached {len(df)} rows for {series_id}")
            return df

        except Exception as e:
            logger.error(f"EIA fetch failed for {series_id}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # FRED (macro)
    # ------------------------------------------------------------------
    def fetch_fred(self, series_id: str, api_key: Optional[str] = None,
                   refresh: bool = False) -> pd.DataFrame:
        """Fetch from FRED API (Federal Reserve)"""
        cache_path = self.config.cache_dir / f"fred_{series_id}.parquet"

        if not refresh and self._cache_is_fresh(cache_path):
            return pd.read_parquet(cache_path)

        if not api_key:
            import os
            api_key = os.getenv("FRED_API_KEY", "")

        if not api_key:
            logger.warning(f"No FRED API key, skipping {series_id}")
            return pd.DataFrame()

        try:
            import requests
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": self.config.start_date.isoformat(),
                "observation_end": self.config.end_date.isoformat(),
            }
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            rows = data.get("observations", [])
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", "value"]].dropna()
            df["series_id"] = series_id
            df.to_parquet(cache_path, index=False)
            return df

        except Exception as e:
            logger.error(f"FRED fetch failed for {series_id}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Combined dataset
    # ------------------------------------------------------------------
    def build_master_dataset(self, refresh: bool = False) -> pd.DataFrame:
        """Build the full feature dataset combining all sources, aligned by date"""
        cache_path = self.config.cache_dir / "master_dataset.parquet"
        if not refresh and self._cache_is_fresh(cache_path):
            logger.info("Loading master dataset from cache")
            return pd.read_parquet(cache_path)

        frames = []

        # Oil prices (primary target)
        for name, sym in PRICE_SYMBOLS.items():
            df = self.fetch_yahoo(sym, refresh=refresh)
            if df.empty:
                continue
            df = df[["date", "close", "high", "low", "open", "volume"]].copy()
            df.columns = ["date", f"{name.lower()}_close", f"{name.lower()}_high",
                          f"{name.lower()}_low", f"{name.lower()}_open", f"{name.lower()}_vol"]
            frames.append(df)

        # Macro features
        for name, sym in MACRO_SYMBOLS.items():
            df = self.fetch_yahoo(sym, refresh=refresh)
            if df.empty:
                continue
            df = df[["date", "close"]].copy()
            df.columns = ["date", f"{name.lower()}_close"]
            frames.append(df)

        if not frames:
            logger.error("No data sources available")
            return pd.DataFrame()

        # Merge on date
        master = frames[0]
        for df in frames[1:]:
            master = master.merge(df, on="date", how="outer")
        master = master.sort_values("date").reset_index(drop=True)

        # Forward-fill non-oil columns (markets close on different schedules)
        macro_cols = [c for c in master.columns
                      if any(c.startswith(m.lower()) for m in MACRO_SYMBOLS)]
        master[macro_cols] = master[macro_cols].ffill()

        # Drop rows before we have oil prices
        master = master.dropna(subset=["wti_close"]).reset_index(drop=True)

        master.to_parquet(cache_path, index=False)
        logger.info(f"Master dataset: {len(master)} rows, "
                    f"{master['date'].min()} to {master['date'].max()}, "
                    f"{len(master.columns)} columns")
        return master

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _cache_is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
        return age_hours < self.config.cache_max_age_hours


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = OilDataLoader()
    df = loader.build_master_dataset()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Missing values per column:\n{df.isna().sum()}")

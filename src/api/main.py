"""
FastAPI serving layer for oil forecasts.

Endpoints:
  GET  /api/v1/health                       - Liveness
  GET  /api/v1/ready                        - Model loaded?
  GET  /api/v1/forecast                     - Latest forecast + LLM interpretation
  POST /api/v1/forecast/stress-test         - Scenario analysis
  GET  /api/v1/data/price-history           - Historical prices
  GET  /api/v1/data/events                  - Events database
  GET  /api/v1/data/regimes                 - Recent regime classifications
  GET  /api/v1/backtest/summary             - Backtest metrics
  GET  /metrics                             - Prometheus
"""
from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from src.pipelines.inference import InferencePipeline
from src.data.events_database import EVENTS, get_events_in_range

logger = logging.getLogger(__name__)

# Global pipeline instance (loaded on startup)
_pipeline: Optional[InferencePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    try:
        _pipeline = InferencePipeline(artifacts_dir=artifacts_dir).load()
        logger.info("Inference pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        _pipeline = None
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Oil Forecaster API",
    description="LLM + ML-powered oil price forecasting with regime awareness",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
Instrumentator().instrument(app).expose(app)

# Mount static frontend if present
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ============================================================================
# Models
# ============================================================================
class StressTestRequest(BaseModel):
    scenario: str
    horizon_days: int = 30


class ForecastResponse(BaseModel):
    forecast: dict
    current_price: float
    regime: str
    active_events: list
    interpretation: Optional[dict] = None
    feature_snapshot: dict
    model_top_features: list


# ============================================================================
# Health
# ============================================================================
@app.get("/")
async def root():
    return {
        "service": "oil-forecaster",
        "version": "1.0.0",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _pipeline is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/ready")
async def ready():
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")
    return {"ready": True}


# ============================================================================
# Forecasting
# ============================================================================
@app.get("/api/v1/forecast", response_model=ForecastResponse)
async def get_forecast(with_llm: bool = Query(True, description="Include LLM interpretation")):
    """Get the latest forecast with optional LLM narrative"""
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded - run training pipeline first")
    try:
        result = _pipeline.predict_latest(with_llm=with_llm)
        return result
    except Exception as e:
        logger.error(f"Forecast failed: {e}", exc_info=True)
        raise HTTPException(500, f"Forecast failed: {e}")


@app.post("/api/v1/forecast/stress-test")
async def stress_test(req: StressTestRequest):
    """Run a hypothetical scenario through the LLM analyst"""
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")
    try:
        analysis = _pipeline.stress_test(req.scenario)
        return {"scenario": req.scenario, "analysis": analysis}
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/v1/forecast/refresh")
async def refresh_data():
    """Re-pull latest data and rebuild features"""
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")
    try:
        df = _pipeline.refresh_data()
        return {
            "refreshed": True,
            "rows": len(df),
            "latest_date": df["date"].max().isoformat() if "date" in df.columns else None,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================================
# Data endpoints
# ============================================================================
@app.get("/api/v1/data/price-history")
async def price_history(
    days: int = Query(365, ge=30, le=9000),
    symbol: str = Query("wti", regex="^(wti|brent)$"),
):
    """Return historical prices for charting"""
    if _pipeline is None or _pipeline.feature_dataset is None:
        raise HTTPException(503, "Data not available")
    df = _pipeline.feature_dataset.tail(days)
    col = f"{symbol}_close"
    if col not in df.columns:
        raise HTTPException(404, f"No data for {symbol}")
    return {
        "symbol": symbol,
        "data": [
            {"date": d.isoformat() if hasattr(d, "isoformat") else str(d), "price": float(p)}
            for d, p in zip(df["date"], df[col])
            if not (isinstance(p, float) and (p != p))  # filter NaN
        ],
    }


@app.get("/api/v1/data/events")
async def list_events(
    start_year: int = Query(2000, ge=1990, le=2100),
    end_year: int = Query(2030, ge=1990, le=2100),
    event_type: Optional[str] = None,
):
    """Return events from the database with optional filters"""
    events = [e.to_dict() for e in EVENTS
              if start_year <= e.start_date.year <= end_year]
    if event_type:
        events = [e for e in events if e["event_type"] == event_type]
    return {"count": len(events), "events": events}


@app.get("/api/v1/data/events/active")
async def active_events():
    """Currently active events"""
    today = date.today()
    active = get_events_in_range(today, today)
    return {"count": len(active), "events": [e.to_dict() for e in active]}


@app.get("/api/v1/data/regimes")
async def regimes(days: int = Query(252, ge=30, le=2520)):
    """Recent regime classifications"""
    if _pipeline is None or _pipeline.feature_dataset is None:
        raise HTTPException(503, "Data not available")
    df = _pipeline.feature_dataset.tail(days)
    if "regime_name" not in df.columns:
        raise HTTPException(404, "No regime data available")
    return {
        "data": [
            {
                "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
                "regime": str(r),
                "price": float(p) if not (isinstance(p, float) and (p != p)) else None,
            }
            for d, r, p in zip(df["date"], df["regime_name"], df.get("wti_close", []))
        ],
    }


# ============================================================================
# Backtest
# ============================================================================
@app.get("/api/v1/backtest/summary")
async def backtest_summary():
    """Return saved backtest metrics"""
    import json
    bt_path = Path(os.getenv("ARTIFACTS_DIR", "artifacts")) / "backtest_metrics.json"
    if not bt_path.exists():
        raise HTTPException(404, "No backtest results available")
    with open(bt_path) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

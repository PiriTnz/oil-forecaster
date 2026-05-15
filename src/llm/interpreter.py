"""
LLM-powered narrative layer.

The numerical models produce forecasts; the LLM does three things:

  1. INTERPRET: explain WHY the model is predicting what it is, based on
     active geopolitical events, current regime, and feature importances.

  2. CONTEXTUALIZE: compare current setup to historical analogs (e.g.,
     "this looks like Sep 2019 - drone strike on Saudi Aramco - which led
     to a 14% spike that reversed within 2 weeks").

  3. STRESS-TEST: given a hypothetical scenario from the user (e.g., "what
     if Strait of Hormuz closes for a week"), generate an analytical view.

The LLM never replaces the quantitative forecast; it explains and contextualizes.
"""
from __future__ import annotations
import logging
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMInterpretation:
    forecast_summary: str
    primary_drivers: list[str]
    historical_analogs: list[dict]  # [{"event": ..., "similarity": ..., "outcome": ...}]
    risk_factors: list[str]
    confidence_qualifier: str
    scenario_analysis: Optional[str] = None
    raw_response: str = ""


SYSTEM_PROMPT = """You are a senior commodity strategist specializing in oil markets. You analyze numerical model output, geopolitical context, and historical analogs to produce concise, professional market commentary.

CRITICAL RULES:
1. You are explaining a model's forecast, NOT making your own. Anchor your narrative to the model's prediction.
2. Be specific. Reference exact events from the provided database.
3. Acknowledge uncertainty. Models are wrong frequently in oil markets.
4. NEVER give investment advice. Frame everything as analysis.
5. Respond with ONLY valid JSON matching the schema, no preamble.

JSON Schema:
{
  "forecast_summary": "2-3 sentence summary anchored to the model's prediction",
  "primary_drivers": ["3-5 specific drivers, each a single sentence"],
  "historical_analogs": [
    {"event": "name", "year": YYYY, "similarity_basis": "why analogous", "outcome": "what happened"}
  ],
  "risk_factors": ["2-4 specific risks that could invalidate the forecast"],
  "confidence_qualifier": "1 sentence on why model confidence is high/medium/low here"
}"""


class ForecastInterpreter:
    """Wraps Anthropic Claude for forecast explanation"""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-5"):
        import os
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.enabled = bool(self.api_key)

        if self.enabled:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("ANTHROPIC_API_KEY not set - LLM interpretation disabled")

    def interpret(
        self,
        forecast: dict,
        current_features: dict,
        active_events: list[dict],
        recent_regime: str,
        feature_importance_top10: list[dict],
        historical_context: Optional[list[dict]] = None,
    ) -> LLMInterpretation:
        """Generate narrative interpretation of a forecast"""

        if not self.enabled:
            return self._fallback_interpretation(forecast, active_events)

        user_prompt = self._build_prompt(
            forecast=forecast,
            features=current_features,
            events=active_events,
            regime=recent_regime,
            importance=feature_importance_top10,
            history=historical_context or [],
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = response.content[0].text.strip()

            # Strip markdown fences
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)
            return LLMInterpretation(
                forecast_summary=parsed["forecast_summary"],
                primary_drivers=parsed["primary_drivers"],
                historical_analogs=parsed.get("historical_analogs", []),
                risk_factors=parsed["risk_factors"],
                confidence_qualifier=parsed["confidence_qualifier"],
                raw_response=response.content[0].text,
            )

        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}", exc_info=True)
            return self._fallback_interpretation(forecast, active_events)

    def stress_test(
        self,
        scenario: str,
        current_price: float,
        current_features: dict,
        active_events: list[dict],
    ) -> str:
        """Generate scenario analysis for a hypothetical event"""
        if not self.enabled:
            return "LLM stress-test unavailable (no API key configured)"

        prompt = f"""Hypothetical scenario: {scenario}

Current WTI price: ${current_price:.2f}
Currently active major events:
{json.dumps(active_events, indent=2)}

Provide a 2-paragraph analysis: (1) likely immediate price impact based on
historical analogs, (2) factors that would amplify or dampen the impact.
Reference specific past events when relevant."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system="You are a senior commodity analyst. Be specific and reference history.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return f"Stress test failed: {e}"

    def _build_prompt(self, forecast, features, events, regime, importance, history):
        """Build the analysis prompt"""
        return f"""Today: {datetime.utcnow().date().isoformat()}

MODEL FORECAST ({forecast['horizon_days']}-day horizon):
  - Predicted return: {forecast['predicted_pct_return']*100:+.2f}%
  - Direction: {forecast['direction']}
  - Model confidence: {forecast['confidence']:.2f}
  - Individual model outputs: {json.dumps(forecast.get('model_predictions', {}), indent=2)}

CURRENT MARKET STATE:
  - WTI price: ${features.get('wti_close', 'N/A')}
  - 21-day vol: {features.get('vol_21d', 'N/A')}
  - 1-year drawdown: {features.get('drawdown_1y', 'N/A')}
  - RSI(14): {features.get('rsi_14', 'N/A')}
  - VIX level: {features.get('vix_level', 'N/A')}
  - DXY 21-day change: {features.get('dxy_ret_21d', 'N/A')}

DETECTED REGIME: {regime}

ACTIVE GEOPOLITICAL EVENTS:
{json.dumps(events, indent=2) if events else "  None"}

TOP MODEL FEATURES (by importance):
{json.dumps(importance[:10], indent=2)}

HISTORICAL CONTEXT FROM SIMILAR PERIODS:
{json.dumps(history[:5], indent=2) if history else "  None provided"}

Provide your analysis as JSON per the schema."""

    def _fallback_interpretation(self, forecast, active_events) -> LLMInterpretation:
        direction = forecast.get("direction", "unknown")
        pct = forecast.get("predicted_pct_return", 0) * 100

        drivers = []
        if active_events:
            for e in active_events[:3]:
                drivers.append(f"Active event: {e.get('name', 'unknown')} ({e.get('event_type', '?')})")
        if not drivers:
            drivers.append("No major geopolitical events active")

        return LLMInterpretation(
            forecast_summary=f"Model predicts {direction} move of {pct:+.2f}% over the forecast horizon. (LLM narrative unavailable.)",
            primary_drivers=drivers,
            historical_analogs=[],
            risk_factors=["LLM interpretation unavailable - manual review recommended"],
            confidence_qualifier="Cannot assess without LLM narrative",
        )

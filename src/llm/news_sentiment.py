"""
News Sentiment Analyzer.

Daily news → structured signals for the forecasting model.

Special design considerations:
  1. Trump-class actors have a HUGE rhetoric-action gap. A bombastic statement
     ("we'll destroy their civilization") may be followed within 48 hours by
     "let's negotiate" — and vice versa. We track these as SEPARATE features
     so the model can learn the (un)reliability of statements.
  2. Iranian officials' statements about Hormuz/retaliation also follow patterns:
     general threats vs specific operational announcements have very different
     market impact.
  3. OPEC: sourced statements vs official communiqués differ in reliability.

For each batch of news (typically 1 day), the LLM produces a structured JSON
with sentiment, intensity, actor-specific signals, and explicit rhetoric-action
flags.
"""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data classes
# =============================================================================
@dataclass
class NewsItem:
    """A single news article"""
    title: str
    source: str
    published_at: datetime
    description: str = ""
    url: str = ""

    def to_compact(self) -> str:
        """Compact one-line representation for LLM input"""
        ts = self.published_at.strftime("%Y-%m-%d %H:%M")
        return f"[{ts}] [{self.source}] {self.title}" + (
            f" -- {self.description[:200]}" if self.description else ""
        )


@dataclass
class DailySentimentSignals:
    """
    Structured sentiment signals for one trading day.

    These become FEATURES in the forecasting model. Each value is a continuous
    score; the model learns how to weight them.
    """
    date: date

    # ---- Aggregate sentiment (signed) ----
    overall_sentiment: float = 0.0      # [-1, +1] — bearish to bullish for oil
    geopolitical_tension: float = 0.0   # [0, 10] — global tension index
    supply_risk: float = 0.0            # [0, 10] — supply disruption risk
    demand_outlook: float = 0.0         # [-1, +1] — demand expectations

    # ---- Actor-specific signals ----
    trump_oil_rhetoric: float = 0.0        # [-1, +1] — Trump's stated stance
    trump_action_alignment: float = 0.0    # [-1, +1] — does action match rhetoric?
    trump_volatility_factor: float = 0.0   # [0, 1] — how erratic today's signals
    iran_threat_level: float = 0.0          # [0, 10]
    iran_diplomatic_openness: float = 0.0   # [-1, +1]
    saudi_opec_stance: float = 0.0          # [-1, +1] — hawkish (cuts) to dovish (open taps)
    china_iran_engagement: float = 0.0      # [-1, +1]
    russia_oil_policy: float = 0.0          # [-1, +1]

    # ---- Specific event flags (binary) ----
    hormuz_mentions: int = 0
    hormuz_closure_threat: bool = False
    direct_military_action: bool = False
    sanctions_news: bool = False
    opec_meeting_news: bool = False

    # ---- News volume / attention ----
    news_volume: int = 0
    relevant_news_volume: int = 0

    # ---- Metadata ----
    llm_confidence: float = 0.5
    notes: str = ""
    rationale: str = ""

    def to_features_dict(self) -> dict:
        """Return a flat dict suitable for merging into the feature matrix"""
        d = asdict(self)
        d.pop("notes", None)
        d.pop("rationale", None)
        d["date"] = self.date.isoformat() if isinstance(self.date, date) else str(self.date)
        # Convert booleans to ints for ML
        for k, v in list(d.items()):
            if isinstance(v, bool):
                d[k] = int(v)
        return d


# =============================================================================
# LLM-based sentiment analyzer
# =============================================================================
SENTIMENT_SYSTEM_PROMPT = """You are an expert oil market sentiment analyst. Your job is to read a batch of news from one day and produce a STRUCTURED sentiment signal that a forecasting model will use as features.

CRITICAL INSIGHT - the Trump rhetoric-action gap:
Donald Trump has a documented pattern of bombastic statements followed by softer actions, OR vice versa. Examples in the 2025-2026 Iran war:
- March 23, 2026: said he'd postpone attacks "for talks" → oil fell sharply, then he resumed strikes within days
- March 27, 2026: threatened to seize the Strait of Hormuz → no immediate action
- April 8, 2026: announced ceasefire → broken within 2 weeks
- "Their whole civilization will die tonight" → followed by negotiations

When you score `trump_oil_rhetoric`, capture WHAT HE SAID today.
When you score `trump_action_alignment`, capture how closely his stated position matches his concurrent actions (or recent past actions). Use:
  +1.0 = saying X and doing X (aligned)
   0.0 = saying X with no observable action
  -1.0 = saying X but doing opposite (e.g., threatening then negotiating)
When you score `trump_volatility_factor`, capture today's contradictoriness:
   0.0 = consistent messaging
   1.0 = mutually contradictory statements within hours

You will respond with ONLY valid JSON matching this schema, no preamble:
{
  "overall_sentiment": float in [-1, 1],
  "geopolitical_tension": float in [0, 10],
  "supply_risk": float in [0, 10],
  "demand_outlook": float in [-1, 1],
  "trump_oil_rhetoric": float in [-1, 1],
  "trump_action_alignment": float in [-1, 1],
  "trump_volatility_factor": float in [0, 1],
  "iran_threat_level": float in [0, 10],
  "iran_diplomatic_openness": float in [-1, 1],
  "saudi_opec_stance": float in [-1, 1],
  "china_iran_engagement": float in [-1, 1],
  "russia_oil_policy": float in [-1, 1],
  "hormuz_mentions": int,
  "hormuz_closure_threat": bool,
  "direct_military_action": bool,
  "sanctions_news": bool,
  "opec_meeting_news": bool,
  "llm_confidence": float in [0, 1],
  "rationale": "1-2 sentence explanation of the day's dominant signal"
}

Sentiment sign conventions for OIL:
  +1 (bullish for oil) = supply threats, war, sanctions on producers, OPEC cuts
  -1 (bearish for oil) = supply increases, demand destruction, ceasefires, OPEC opening taps

If there is NO oil-relevant news, return zeros across the board with llm_confidence=0.1."""


@dataclass
class SentimentConfig:
    model: str = "claude-sonnet-4-5"
    max_tokens: int = 1024
    batch_window_days: int = 1   # Aggregate news per day
    cache_dir: Optional[Path] = None


class LLMNewsAnalyzer:
    """
    Analyze a batch of daily news with an LLM, producing structured signals.

    Caches results to disk to avoid re-calling for the same date.
    """

    def __init__(self, api_key: Optional[str] = None,
                 config: Optional[SentimentConfig] = None):
        import os
        self.config = config or SentimentConfig()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.enabled = bool(self.api_key)

        if self.enabled:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("ANTHROPIC_API_KEY not set; using rule-based fallback")

        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze_day(
        self,
        target_date: date,
        news_items: list[NewsItem],
        prior_context: Optional[str] = None,
    ) -> DailySentimentSignals:
        """
        Score one day of news.

        prior_context: optional string with summary of last 3-7 days
                       (helps the LLM detect rhetoric-action gap)
        """
        # Check cache first
        cached = self._load_from_cache(target_date)
        if cached is not None:
            return cached

        if not news_items:
            return DailySentimentSignals(date=target_date, llm_confidence=0.0)

        if not self.enabled:
            return self._rule_based_fallback(target_date, news_items)

        # Build the prompt
        news_block = "\n".join(it.to_compact() for it in news_items[:50])
        context_block = f"\nPRIOR 3-7 DAY CONTEXT:\n{prior_context}\n" if prior_context else ""

        user_prompt = f"""Date: {target_date.isoformat()}
Number of news items: {len(news_items)}
{context_block}
NEWS:
{news_block}

Analyze the day's oil-market sentiment and return JSON per the schema."""

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=SENTIMENT_SYSTEM_PROMPT,
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
            parsed["date"] = target_date
            parsed["news_volume"] = len(news_items)
            parsed["relevant_news_volume"] = len(news_items)
            parsed["notes"] = ""

            # Drop any extra keys before instantiation
            valid_keys = {f for f in DailySentimentSignals.__dataclass_fields__}
            filtered = {k: v for k, v in parsed.items() if k in valid_keys}

            signals = DailySentimentSignals(**filtered)
            self._save_to_cache(signals)
            return signals

        except Exception as e:
            logger.error(f"LLM sentiment failed for {target_date}: {e}")
            return self._rule_based_fallback(target_date, news_items)

    # ------------------------------------------------------------------
    # Bulk processing for historical data
    # ------------------------------------------------------------------
    def analyze_range(
        self,
        news_by_date: dict[date, list[NewsItem]],
        with_context: bool = True,
    ) -> pd.DataFrame:
        """Score every date in news_by_date, returning a feature DataFrame."""
        rows = []
        sorted_dates = sorted(news_by_date.keys())
        recent_signals: list[DailySentimentSignals] = []

        for d in sorted_dates:
            items = news_by_date[d]
            context = self._build_context_string(recent_signals[-7:]) if with_context else None
            signals = self.analyze_day(d, items, prior_context=context)
            recent_signals.append(signals)
            rows.append(signals.to_features_dict())

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------
    def _rule_based_fallback(
        self, target_date: date, news_items: list[NewsItem]
    ) -> DailySentimentSignals:
        """Keyword-based heuristics when LLM unavailable. Deliberately conservative."""
        text = " ".join((it.title + " " + (it.description or "")).lower()
                        for it in news_items)

        bullish_terms = ["war", "attack", "strike", "sanctions", "cut",
                         "disruption", "block", "missile", "explosion"]
        bearish_terms = ["ceasefire", "deal", "agreement", "increase production",
                         "release reserves", "negotiat", "open the strait"]

        b_count = sum(text.count(w) for w in bullish_terms)
        bear_count = sum(text.count(w) for w in bearish_terms)
        total = b_count + bear_count + 1
        overall = (b_count - bear_count) / total

        return DailySentimentSignals(
            date=target_date,
            overall_sentiment=float(np.clip(overall, -1, 1)),
            geopolitical_tension=float(min(b_count * 0.5, 10.0)),
            supply_risk=float(min(text.count("supply") + text.count("disrupt"), 10.0)),
            hormuz_mentions=text.count("hormuz"),
            hormuz_closure_threat="strait" in text and ("clos" in text or "block" in text),
            direct_military_action="strike" in text or "missile" in text,
            sanctions_news="sanction" in text,
            opec_meeting_news="opec" in text,
            news_volume=len(news_items),
            relevant_news_volume=len(news_items),
            llm_confidence=0.3,
            notes="rule_based_fallback",
        )

    # ------------------------------------------------------------------
    # Context building for rhetoric-action detection
    # ------------------------------------------------------------------
    def _build_context_string(self, recent: list[DailySentimentSignals]) -> str:
        """
        Construct a short summary of the past few days to help the LLM detect
        rhetoric-action gaps and trend continuity.
        """
        if not recent:
            return ""
        lines = []
        for s in recent:
            lines.append(
                f"{s.date}: tension={s.geopolitical_tension:.1f}, "
                f"trump_rhet={s.trump_oil_rhetoric:+.2f}, "
                f"trump_action={s.trump_action_alignment:+.2f}, "
                f"iran_threat={s.iran_threat_level:.1f}, "
                f"summary={s.rationale[:120]}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _cache_path(self, target_date: date) -> Optional[Path]:
        if not self.config.cache_dir:
            return None
        return self.config.cache_dir / f"sentiment_{target_date.isoformat()}.json"

    def _load_from_cache(self, target_date: date) -> Optional[DailySentimentSignals]:
        p = self._cache_path(target_date)
        if p is None or not p.exists():
            return None
        try:
            with open(p) as f:
                data = json.load(f)
            data["date"] = date.fromisoformat(data["date"])
            valid_keys = {f for f in DailySentimentSignals.__dataclass_fields__}
            return DailySentimentSignals(**{k: v for k, v in data.items() if k in valid_keys})
        except Exception as e:
            logger.warning(f"Failed to load cached {p}: {e}")
            return None

    def _save_to_cache(self, signals: DailySentimentSignals) -> None:
        p = self._cache_path(signals.date)
        if p is None:
            return
        try:
            with open(p, "w") as f:
                json.dump(signals.to_features_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to cache sentiment: {e}")


# =============================================================================
# Trump rhetoric-action gap detector
# =============================================================================
class RhetoricActionDetector:
    """
    Tracks whether Trump's recent statements match his subsequent actions.

    Maintains a rolling history of (statement_intensity, action_intensity) pairs.
    Produces a "reliability score" feature: 1.0 = statements predict actions,
    -1.0 = statements are anti-predictive of actions (counter-signal).

    Mathematical formulation:
        Let R_t = rhetoric intensity at day t  (signed, in [-1, +1])
        Let A_{t+k} = action intensity over days [t+1, t+k]  (signed)

        Reliability = Corr( {R_t}, {A_{t+k}} )  over rolling window
        Volatility  = Var( {R_t} ) over rolling window

    A high-volatility, low-reliability actor produces NOISE that the model
    should learn to discount.
    """

    def __init__(self, window: int = 20, lookahead: int = 5):
        """
        window: how many past days to use for correlation
        lookahead: how many days after a statement to measure actions
        """
        self.window = window
        self.lookahead = lookahead
        self.history: list[dict] = []  # [{date, rhetoric, action_observed}]

    def add_observation(
        self,
        d: date,
        rhetoric_score: float,
        action_score: Optional[float] = None,
    ) -> None:
        """
        Add a new day's rhetoric observation. Action is filled in `lookahead` days later.
        """
        self.history.append({
            "date": d,
            "rhetoric": float(rhetoric_score),
            "action": float(action_score) if action_score is not None else None,
        })

    def update_actions(self, current_date: date, observed_action: float) -> None:
        """
        Once we observe the actual action `lookahead` days after a rhetoric
        observation, back-fill it.
        """
        # Find rhetoric observations that are now `lookahead` days old
        for entry in self.history:
            if entry["action"] is None:
                gap = (current_date - entry["date"]).days
                if 1 <= gap <= self.lookahead:
                    # Aggregate the action into the rhetoric's expected window
                    entry["action"] = (entry.get("action") or 0.0) + observed_action / self.lookahead

    def current_reliability(self) -> float:
        """
        Spearman-like correlation between rhetoric and subsequent actions over
        the rolling window. Returns 0.0 if not enough data.
        """
        recent = [h for h in self.history[-self.window:]
                  if h["action"] is not None]
        if len(recent) < 5:
            return 0.0
        r = np.array([h["rhetoric"] for h in recent])
        a = np.array([h["action"] for h in recent])
        if np.std(r) < 1e-6 or np.std(a) < 1e-6:
            return 0.0
        return float(np.corrcoef(r, a)[0, 1])

    def current_volatility(self) -> float:
        """Standard deviation of rhetoric over the window."""
        recent = self.history[-self.window:]
        if len(recent) < 5:
            return 0.0
        return float(np.std([h["rhetoric"] for h in recent]))

    def discount_factor(self) -> float:
        """
        How much should the model trust today's rhetoric?
        Returns a multiplier in [0, 1]:
          1.0 = highly reliable (rhetoric matches actions)
          0.0 = pure noise (rhetoric uncorrelated with actions)
          <0 = anti-predictive (we'd use the OPPOSITE of stated position)
        """
        rel = self.current_reliability()
        # Map reliability ∈ [-1, +1] to discount ∈ [0, 1] with anti-predictive cap
        if rel >= 0:
            return rel
        # If anti-predictive, treat as "discount to zero" rather than flipping the sign
        # (we don't want the model to chase Trump's anti-signal blindly)
        return max(0.0, 0.5 + rel * 0.5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo
    print("=" * 70)
    print(" NEWS SENTIMENT DEMO")
    print("=" * 70)

    # Synthetic news for March 23, 2026 (Trump postpones attacks day)
    items = [
        NewsItem(
            title="Trump announces 2-week postponement of Iran attacks for talks",
            source="Reuters", published_at=datetime(2026, 3, 23, 14, 0),
            description="President Trump said he would pause US military operations against Iran for two weeks to allow for diplomatic talks."
        ),
        NewsItem(
            title="Iran foreign minister: 'open to genuine negotiations'",
            source="Bloomberg", published_at=datetime(2026, 3, 23, 16, 30),
            description="Iranian FM Araghchi welcomed the pause but warned attacks would resume tit-for-tat."
        ),
        NewsItem(
            title="Suspicious $580M short bets on oil futures placed minutes before announcement",
            source="Financial Times", published_at=datetime(2026, 3, 23, 18, 0),
            description="Investigation underway into potential insider trading around Trump's postponement statement."
        ),
        NewsItem(
            title="Brent crude falls 8% as Hormuz closure fears ease",
            source="Reuters", published_at=datetime(2026, 3, 23, 20, 0),
            description=""
        ),
    ]

    analyzer = LLMNewsAnalyzer()
    signals = analyzer.analyze_day(date(2026, 3, 23), items)
    print(f"\nSentiment for {signals.date}:")
    for k, v in signals.to_features_dict().items():
        if k != "date":
            print(f"  {k:<35s} {v}")

    # Rhetoric-action detector demo
    print("\n" + "=" * 70)
    print(" RHETORIC-ACTION DETECTOR DEMO")
    print("=" * 70)
    detector = RhetoricActionDetector(window=10, lookahead=3)
    # Simulate Trump-like erratic pattern
    for i, (rhet, act) in enumerate([
        (+0.9, -0.3),  # threatens, then negotiates
        (+0.8, -0.2),
        (-0.6, +0.7),  # talks peace, then attacks
        (+0.7, -0.4),
        (-0.5, +0.5),
        (+0.8, -0.1),
        (-0.7, +0.6),
    ]):
        d = date(2026, 3, 1) + timedelta(days=i)
        detector.add_observation(d, rhet, act)

    print(f"\nReliability of rhetoric: {detector.current_reliability():+.3f}")
    print(f"Volatility of rhetoric:  {detector.current_volatility():.3f}")
    print(f"Discount factor:         {detector.discount_factor():.3f}")
    print(f"\n(reliability < 0 means rhetoric is ANTI-predictive of actions)")

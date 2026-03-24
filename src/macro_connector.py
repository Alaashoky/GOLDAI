from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DXY / Gold inverse relationship explanation
# ---------------------------------------------------------------------------
# The US Dollar Index (DXY) measures USD strength against a basket of
# currencies.  Gold (XAUUSD) is priced in USD so:
#   • DXY rising  → USD stronger → gold price falls (bearish bias for gold)
#   • DXY falling → USD weaker  → gold price rises  (bullish bias for gold)
#
# US 10-year Treasury yields share a similar inverse relationship:
#   • Yields rising  → risk-free rate up → gold (non-yielding) less attractive
#   • Yields falling → lower opportunity cost → gold more attractive
# ---------------------------------------------------------------------------


@dataclass
class MacroData:
    """Snapshot of macro data used to derive gold bias.

    Attributes:
        dxy_value: Latest DXY index value (or NaN if unavailable).
        dxy_trend: DXY trend direction (+1 up, -1 down, 0 flat).
        yield_value: US 10Y yield in percent.
        yield_trend: Yield trend direction (+1 up, -1 down, 0 flat).
        risk_sentiment: Risk-on/off score [-1 risk-off, +1 risk-on].
        macro_score: Aggregated gold bias in [-1, +1].
        source: Data source label ('live' or 'mock').
        timestamp: UTC timestamp of the data snapshot.
    """

    dxy_value: float = float("nan")
    dxy_trend: int = 0
    yield_value: float = float("nan")
    yield_trend: int = 0
    risk_sentiment: float = 0.0
    macro_score: float = 0.0
    source: str = "mock"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MacroConnector:
    """Fetch and aggregate macro-economic data relevant to XAUUSD.

    When live data sources are unavailable (no API key / network error) the
    connector falls back to cached or mock data so that the bot can continue
    operating with a neutral macro bias rather than hard-failing.

    Args:
        use_mock: Force mock data regardless of API availability.
        dxy_weight: Weight for DXY contribution to macro score (default 0.5).
        yield_weight: Weight for yields contribution (default 0.3).
        sentiment_weight: Weight for risk sentiment (default 0.2).
    """

    def __init__(
        self,
        use_mock: bool = True,
        dxy_weight: float = 0.5,
        yield_weight: float = 0.3,
        sentiment_weight: float = 0.2,
    ) -> None:
        self.use_mock = use_mock
        self.dxy_weight = dxy_weight
        self.yield_weight = yield_weight
        self.sentiment_weight = sentiment_weight
        self._last_data: Optional[MacroData] = None

    # ------------------------------------------------------------------
    # Mock / fallback helpers
    # ------------------------------------------------------------------

    def _mock_dxy(self) -> tuple[float, int]:
        """Return plausible mock DXY data.

        Returns:
            Tuple of (dxy_value, trend).
        """
        value = round(103.0 + random.gauss(0, 0.5), 2)
        trend = random.choice([-1, 0, 0, 1])
        return value, trend

    def _mock_yield(self) -> tuple[float, int]:
        """Return plausible mock 10Y yield data.

        Returns:
            Tuple of (yield_percent, trend).
        """
        value = round(4.2 + random.gauss(0, 0.1), 3)
        trend = random.choice([-1, 0, 0, 1])
        return value, trend

    def _mock_risk_sentiment(self) -> float:
        """Return plausible mock risk sentiment score.

        Returns:
            Score in [-1, 1].
        """
        return round(random.gauss(0, 0.3), 3)

    # ------------------------------------------------------------------
    # Trend fetch methods (async; can be overridden for live data)
    # ------------------------------------------------------------------

    async def fetch_dxy_trend(self) -> tuple[float, int]:
        """Fetch DXY value and trend direction.

        Returns:
            Tuple of (dxy_value, trend) where trend is +1/0/-1.
        """
        if self.use_mock:
            return self._mock_dxy()
        # Placeholder for live API integration
        logger.warning("Live DXY API not configured; using mock")
        return self._mock_dxy()

    async def fetch_yields_trend(self) -> tuple[float, int]:
        """Fetch US 10-year yield value and trend.

        Returns:
            Tuple of (yield_percent, trend) where trend is +1/0/-1.
        """
        if self.use_mock:
            return self._mock_yield()
        logger.warning("Live yields API not configured; using mock")
        return self._mock_yield()

    async def fetch_risk_sentiment(self) -> float:
        """Fetch overall market risk sentiment.

        Returns:
            Score in [-1, 1]; +1 = strong risk-on, -1 = strong risk-off.
        """
        if self.use_mock:
            return self._mock_risk_sentiment()
        logger.warning("Live risk-sentiment API not configured; using mock")
        return self._mock_risk_sentiment()

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def calculate_gold_macro_bias(
        self,
        dxy_trend: int,
        yields_trend: int,
        risk_sentiment: float,
    ) -> float:
        """Combine macro inputs into a single gold bias score.

        DXY and yields are *inversely* related to gold, so their contribution
        is negated.  Risk-on sentiment is mildly negative for gold (safe haven
        demand weakens), so it is also inverted.

        Args:
            dxy_trend: DXY trend +1/0/-1.
            yields_trend: 10Y yield trend +1/0/-1.
            risk_sentiment: Risk sentiment in [-1, 1].

        Returns:
            Aggregated gold macro bias in [-1, +1]; positive = bullish gold.
        """
        # Invert: DXY up → gold down
        dxy_contribution = -float(dxy_trend) * self.dxy_weight
        # Invert: yields up → gold down
        yield_contribution = -float(yields_trend) * self.yield_weight
        # Risk-on → slightly negative for gold (less safe-haven demand)
        sentiment_contribution = -risk_sentiment * self.sentiment_weight

        score = dxy_contribution + yield_contribution + sentiment_contribution
        # Normalise to [-1, 1]
        normaliser = self.dxy_weight + self.yield_weight + self.sentiment_weight
        if normaliser > 0:
            score = score / normaliser
        return round(max(-1.0, min(1.0, score)), 4)

    async def get_macro_sentiment(self) -> float:
        """Fetch all macro inputs and return aggregated gold bias score.

        Returns:
            Gold macro bias in [-1, +1]; +1 = strongly bullish for gold.
        """
        try:
            dxy_value, dxy_trend = await self.fetch_dxy_trend()
            yield_value, yield_trend = await self.fetch_yields_trend()
            risk_sentiment = await self.fetch_risk_sentiment()

            score = self.calculate_gold_macro_bias(dxy_trend, yield_trend, risk_sentiment)

            self._last_data = MacroData(
                dxy_value=dxy_value,
                dxy_trend=dxy_trend,
                yield_value=yield_value,
                yield_trend=yield_trend,
                risk_sentiment=risk_sentiment,
                macro_score=score,
                source="mock" if self.use_mock else "live",
                timestamp=__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ),
            )
            logger.debug("Macro sentiment: score=%.4f dxy=%d yields=%d risk=%.3f",
                         score, dxy_trend, yield_trend, risk_sentiment)
            return score

        except Exception as exc:
            logger.error("MacroConnector error: %s; returning neutral bias", exc)
            return 0.0

    def get_last_data(self) -> Optional[MacroData]:
        """Return the most recently fetched MacroData snapshot.

        Returns:
            Last MacroData or None if not yet fetched.
        """
        return self._last_data

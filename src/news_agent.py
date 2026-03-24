"""Economic news agent for XAUUSD impact assessment.

Fetches upcoming high-impact economic events, determines blackout windows,
and estimates each event's likely effect on gold prices.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from src.utils import utc_now

logger = logging.getLogger(__name__)

# High-impact keywords that affect XAUUSD
HIGH_IMPACT_KEYWORDS = frozenset([
    "NFP", "FOMC", "CPI", "PPI", "GDP", "Fed", "Powell",
    "Rate", "Inflation", "Employment", "Payroll", "Interest",
    "Treasury", "Dollar", "DXY",
])

# Minutes to avoid trading before and after each event category
BLACKOUT_MINUTES: dict[str, int] = {
    "NFP":     60,
    "FOMC":    90,
    "CPI":     45,
    "PPI":     30,
    "GDP":     30,
    "Fed":     45,
    "Powell":  45,
    "default": 30,
}

# Estimated impact score for each keyword (0 = no impact, 1 = maximum)
IMPACT_SCORES: dict[str, float] = {
    "NFP":        1.0,
    "FOMC":       1.0,
    "CPI":        0.9,
    "PPI":        0.7,
    "GDP":        0.8,
    "Fed":        0.85,
    "Powell":     0.85,
    "Rate":       0.75,
    "Inflation":  0.8,
    "Employment": 0.75,
    "Payroll":    0.85,
    "Interest":   0.7,
    "Treasury":   0.6,
    "Dollar":     0.55,
    "DXY":        0.55,
}


@dataclass
class NewsEvent:
    """A single economic news event.

    Attributes:
        title: Event title or description.
        event_time: Scheduled UTC release time.
        impact: Impact category: "HIGH", "MEDIUM", or "LOW".
        currency: Affected currency code (e.g. "USD").
        forecast: Analyst forecast value as string.
        previous: Previous period value as string.
        gold_impact_score: Estimated effect on XAUUSD (0.0–1.0).
    """

    title: str
    event_time: datetime
    impact: str = "LOW"
    currency: str = "USD"
    forecast: str = ""
    previous: str = ""
    gold_impact_score: float = 0.0


class NewsAgent:
    """Tracks high-impact economic events and manages news blackout windows.

    Caches events to avoid repeated network calls, and provides helpers
    to determine whether gold trading should be paused near an event.

    Attributes:
        blackout_minutes_before: Minutes to pause before an event.
        blackout_minutes_after: Minutes to pause after an event.
        cache_ttl_minutes: How long fetched event lists remain valid.
    """

    def __init__(
        self,
        blackout_minutes_before: int = 30,
        blackout_minutes_after: int = 30,
        cache_ttl_minutes: int = 60,
    ) -> None:
        """Initialise the news agent.

        Args:
            blackout_minutes_before: Minutes before an event to block trading.
            blackout_minutes_after: Minutes after an event to block trading.
            cache_ttl_minutes: Cache lifetime in minutes for fetched events.
        """
        self.blackout_minutes_before = blackout_minutes_before
        self.blackout_minutes_after = blackout_minutes_after
        self.cache_ttl_minutes = cache_ttl_minutes
        self._cached_events: List[NewsEvent] = []
        self._cache_fetched_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_news_events(self, hours_ahead: int = 24) -> List[NewsEvent]:
        """Retrieve upcoming economic events within the given horizon.

        Attempts to fetch from a public economic calendar. Falls back to
        returning an empty list when the network is unavailable.

        Args:
            hours_ahead: Number of hours ahead to look for events.

        Returns:
            List[NewsEvent]: Upcoming events sorted by scheduled time.
        """
        if self._is_cache_valid():
            return self._cached_events

        events: List[NewsEvent] = []
        try:
            events = await self._fetch_from_calendar(hours_ahead)
        except Exception as exc:
            logger.warning("News calendar fetch failed (%s) — using empty event list.", exc)
            events = self._mock_events(hours_ahead)

        for event in events:
            event.gold_impact_score = self.assess_gold_impact(event)

        self._cached_events = sorted(events, key=lambda e: e.event_time)
        self._cache_fetched_at = utc_now()
        logger.info("Loaded %d upcoming news events (next %dh).", len(events), hours_ahead)
        return self._cached_events

    def is_high_impact_event(self, now: Optional[datetime] = None) -> bool:
        """Check whether a high-impact event is imminent or just occurred.

        Args:
            now: UTC datetime to evaluate. Uses current UTC time if None.

        Returns:
            bool: True when a high-impact event falls inside the blackout window.
        """
        now = now or utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        for event in self._cached_events:
            if event.impact != "HIGH":
                continue
            event_time = event.event_time
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
            window_start = event_time - timedelta(minutes=self.blackout_minutes_before)
            window_end   = event_time + timedelta(minutes=self.blackout_minutes_after)
            if window_start <= now <= window_end:
                logger.info("High-impact event active: %s at %s", event.title, event_time)
                return True
        return False

    def get_news_blackout_minutes(self, event_title: str = "") -> int:
        """Return the total blackout duration for a given event type.

        Args:
            event_title: Event title to look up. Falls back to "default".

        Returns:
            int: Total blackout minutes (before + after).
        """
        for keyword, minutes in BLACKOUT_MINUTES.items():
            if keyword.lower() in event_title.lower():
                return minutes
        return BLACKOUT_MINUTES["default"]

    def assess_gold_impact(self, event: NewsEvent) -> float:
        """Estimate how strongly an event will move XAUUSD prices.

        Args:
            event: The news event to assess.

        Returns:
            float: Gold impact score in [0.0, 1.0].
        """
        if event.currency not in ("USD", "XAU"):
            return 0.1

        score = 0.0
        title_upper = event.title.upper()
        for keyword, impact in IMPACT_SCORES.items():
            if keyword.upper() in title_upper:
                score = max(score, impact)

        # Scale down for non-high-impact calendar events
        if event.impact == "MEDIUM":
            score *= 0.6
        elif event.impact == "LOW":
            score *= 0.3

        return round(min(score, 1.0), 2)

    def get_next_high_impact_event(self, now: Optional[datetime] = None) -> Optional[NewsEvent]:
        """Return the next upcoming high-impact event.

        Args:
            now: UTC datetime reference. Uses current UTC time if None.

        Returns:
            Optional[NewsEvent]: The soonest upcoming high-impact event, or None.
        """
        now = now or utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        for event in self._cached_events:
            event_time = event.event_time
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
            if event.impact == "HIGH" and event_time > now:
                return event
        return None

    def minutes_to_next_event(self, now: Optional[datetime] = None) -> Optional[float]:
        """Minutes until the next high-impact event starts.

        Args:
            now: UTC datetime reference. Uses current UTC time if None.

        Returns:
            Optional[float]: Minutes until next event, or None if no event pending.
        """
        now = now or utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        event = self.get_next_high_impact_event(now)
        if event is None:
            return None
        event_time = event.event_time
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        delta = (event_time - now).total_seconds() / 60
        return round(delta, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_cache_valid(self) -> bool:
        """Check whether the cached event list is still fresh."""
        if not self._cache_fetched_at:
            return False
        age = (utc_now() - self._cache_fetched_at).total_seconds() / 60
        return age < self.cache_ttl_minutes

    async def _fetch_from_calendar(self, hours_ahead: int) -> List[NewsEvent]:
        """Attempt to pull events from an external economic calendar.

        Currently uses a lightweight approach to parse a public JSON feed.
        Returns empty list if the request times out or the feed is unavailable.

        Args:
            hours_ahead: Hours ahead to include.

        Returns:
            List[NewsEvent]: Parsed events.
        """
        try:
            import aiohttp  # type: ignore
        except ImportError:
            logger.debug("aiohttp not installed — skipping live news fetch.")
            return []

        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.debug("Calendar request failed: %s", exc)
            return []

        now = utc_now()
        horizon = now + timedelta(hours=hours_ahead)
        events: List[NewsEvent] = []

        for item in data:
            try:
                raw_time = item.get("date", "")
                event_time = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
                if not (now <= event_time <= horizon):
                    continue

                impact_map = {"High": "HIGH", "Medium": "MEDIUM", "Low": "LOW"}
                impact = impact_map.get(item.get("impact", "Low"), "LOW")

                events.append(
                    NewsEvent(
                        title=item.get("title", ""),
                        event_time=event_time,
                        impact=impact,
                        currency=item.get("country", "USD"),
                        forecast=str(item.get("forecast", "")),
                        previous=str(item.get("previous", "")),
                    )
                )
            except Exception:
                continue

        return events

    def _mock_events(self, hours_ahead: int) -> List[NewsEvent]:
        """Return mock events for testing/offline use.

        Args:
            hours_ahead: Hours ahead for mock event scheduling.

        Returns:
            List[NewsEvent]: Representative mock events.
        """
        now = utc_now()
        return [
            NewsEvent(
                title="Non-Farm Payrolls (NFP)",
                event_time=now + timedelta(hours=hours_ahead // 2),
                impact="HIGH",
                currency="USD",
                forecast="200K",
                previous="185K",
            ),
        ]

"""Trading session filter for XAUUSD.

Determines whether trading is allowed based on the current UTC time,
active Forex session, London/NY overlap windows, and news blackout periods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.utils import utc_now

logger = logging.getLogger(__name__)

# Session window definitions: (start_hour_utc, end_hour_utc)
TOKYO_SESSION:  Tuple[int, int] = (0,  9)
LONDON_SESSION: Tuple[int, int] = (7, 16)
NY_SESSION:     Tuple[int, int] = (13, 22)
OVERLAP_LONDON_NY: Tuple[int, int] = (13, 16)

# Volatility multipliers per session
SESSION_VOLATILITY: Dict[str, float] = {
    "TOKYO":           0.7,
    "LONDON":          1.2,
    "NEW_YORK":        1.2,
    "LONDON_NY_OVERLAP": 1.5,
    "OFF_HOURS":       0.4,
}


@dataclass
class NewsEvent:
    """A single high-impact news blackout window.

    Attributes:
        event_time: UTC datetime when the news releases.
        duration_minutes: Total blackout window in minutes (split before/after).
        description: Human-readable event description.
    """

    event_time: datetime
    duration_minutes: int = 60
    description: str = ""

    @property
    def blackout_start(self) -> datetime:
        """Start of the blackout window (half duration before event)."""
        return self.event_time - timedelta(minutes=self.duration_minutes // 2)

    @property
    def blackout_end(self) -> datetime:
        """End of the blackout window (half duration after event)."""
        return self.event_time + timedelta(minutes=self.duration_minutes // 2)


class SessionFilter:
    """Determines trading eligibility based on session and news rules.

    Trading is only permitted during active Forex sessions on weekdays,
    with enhanced conditions during the London/NY overlap. News events
    create automatic blackout windows around their scheduled times.

    Attributes:
        allowed_sessions: Set of session names permitted for trading.
        news_events: Registered news blackout windows.
    """

    def __init__(
        self,
        allowed_sessions: Optional[List[str]] = None,
        allow_off_hours: bool = False,
    ) -> None:
        """Initialise the session filter.

        Args:
            allowed_sessions: Session names to trade in. Defaults to LONDON,
                NEW_YORK, and LONDON_NY_OVERLAP.
            allow_off_hours: Whether to permit trading outside named sessions.
        """
        self.allowed_sessions: List[str] = allowed_sessions or [
            "LONDON", "NEW_YORK", "LONDON_NY_OVERLAP"
        ]
        self.allow_off_hours = allow_off_hours
        self.news_events: List[NewsEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_session(self, now: Optional[datetime] = None) -> str:
        """Identify the dominant Forex trading session at the given time.

        The London/NY overlap takes precedence when both sessions are active.

        Args:
            now: UTC datetime to evaluate. Uses current UTC time if None.

        Returns:
            str: Session name: one of TOKYO, LONDON, NEW_YORK,
                LONDON_NY_OVERLAP, or OFF_HOURS.
        """
        now = now or utc_now()
        hour = now.hour

        in_london = LONDON_SESSION[0] <= hour < LONDON_SESSION[1]
        in_ny = NY_SESSION[0] <= hour < NY_SESSION[1]
        in_overlap = OVERLAP_LONDON_NY[0] <= hour < OVERLAP_LONDON_NY[1]
        in_tokyo = TOKYO_SESSION[0] <= hour < TOKYO_SESSION[1]

        if in_overlap:
            return "LONDON_NY_OVERLAP"
        if in_london:
            return "LONDON"
        if in_ny:
            return "NEW_YORK"
        if in_tokyo:
            return "TOKYO"
        return "OFF_HOURS"

    def is_trading_allowed(self, now: Optional[datetime] = None) -> bool:
        """Return whether a new trade entry is currently permitted.

        A trade is blocked if:
        - It is the weekend
        - The session is not in ``allowed_sessions``
        - A news blackout window is active

        Args:
            now: UTC datetime to evaluate. Uses current UTC time if None.

        Returns:
            bool: True when trading is allowed.
        """
        now = now or utc_now()

        if self.is_weekend(now):
            logger.debug("Trading blocked: weekend.")
            return False

        session = self.get_current_session(now)
        if session == "OFF_HOURS" and not self.allow_off_hours:
            logger.debug("Trading blocked: off-hours.")
            return False

        if session not in self.allowed_sessions and session != "LONDON_NY_OVERLAP":
            logger.debug("Trading blocked: session %s not in allowed list.", session)
            return False

        if self.is_news_blackout(now):
            logger.debug("Trading blocked: news blackout.")
            return False

        return True

    def get_session_volatility_multiplier(self, now: Optional[datetime] = None) -> float:
        """Return a volatility scaling factor for the current session.

        Higher values indicate more volatile sessions where stop-loss
        distances should be widened proportionally.

        Args:
            now: UTC datetime to evaluate. Uses current UTC time if None.

        Returns:
            float: Volatility multiplier (0.4 – 1.5).
        """
        session = self.get_current_session(now or utc_now())
        return SESSION_VOLATILITY.get(session, 0.4)

    def is_news_blackout(self, now: Optional[datetime] = None) -> bool:
        """Check whether a registered news event blackout is active.

        Args:
            now: UTC datetime to evaluate. Uses current UTC time if None.

        Returns:
            bool: True when currently inside a news blackout window.
        """
        now = now or utc_now()
        # Make now timezone-aware if not already
        if now.tzinfo is None:
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)
        for event in self.news_events:
            start = event.blackout_start
            end = event.blackout_end
            # Ensure start/end are timezone-aware
            if start.tzinfo is None:
                from datetime import timezone
                start = start.replace(tzinfo=timezone.utc)
                end = end.replace(tzinfo=timezone.utc)
            if start <= now <= end:
                logger.info("News blackout active: %s", event.description)
                return True
        return False

    def is_weekend(self, now: Optional[datetime] = None) -> bool:
        """Check whether the current day is a weekend (Sat/Sun UTC).

        Args:
            now: UTC datetime to evaluate. Uses current UTC time if None.

        Returns:
            bool: True on Saturday or Sunday.
        """
        now = now or utc_now()
        return now.weekday() >= 5  # 5=Saturday, 6=Sunday

    def add_news_event(
        self,
        event_time: datetime,
        duration_minutes: int = 60,
        description: str = "",
    ) -> None:
        """Register a news event to create an automatic blackout window.

        The blackout window spans ``duration_minutes / 2`` before and after
        the scheduled event time.

        Args:
            event_time: UTC datetime of the news release.
            duration_minutes: Total blackout duration in minutes.
            description: Human-readable description of the event.
        """
        event = NewsEvent(
            event_time=event_time,
            duration_minutes=duration_minutes,
            description=description,
        )
        self.news_events.append(event)
        logger.info(
            "News event added: %s at %s (blackout ±%d min)",
            description, event_time.isoformat(), duration_minutes // 2,
        )

    def remove_expired_events(self, now: Optional[datetime] = None) -> int:
        """Purge news events whose blackout windows have already passed.

        Args:
            now: UTC datetime reference. Uses current UTC time if None.

        Returns:
            int: Number of events removed.
        """
        from datetime import timezone as _tz
        now = now or utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=_tz.utc)
        before = len(self.news_events)
        self.news_events = [e for e in self.news_events if self._aware(e.blackout_end) >= now]
        removed = before - len(self.news_events)
        if removed:
            logger.debug("Removed %d expired news events.", removed)
        return removed

    @staticmethod
    def _aware(dt: datetime) -> datetime:
        """Ensure a datetime is timezone-aware (UTC).

        Args:
            dt: Datetime to normalise.

        Returns:
            datetime: Timezone-aware UTC datetime.
        """
        from datetime import timezone as _tz
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=_tz.utc)

    def get_session_summary(self, now: Optional[datetime] = None) -> Dict[str, object]:
        """Return a human-readable summary of the current session state.

        Args:
            now: UTC datetime reference. Uses current UTC time if None.

        Returns:
            Dict[str, object]: Session state dictionary.
        """
        now = now or utc_now()
        session = self.get_current_session(now)
        return {
            "session":              session,
            "is_weekend":           self.is_weekend(now),
            "is_trading_allowed":   self.is_trading_allowed(now),
            "is_news_blackout":     self.is_news_blackout(now),
            "volatility_multiplier": self.get_session_volatility_multiplier(now),
            "pending_news_events":  len(self.news_events),
            "utc_hour":             now.hour,
        }

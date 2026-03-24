from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

logger = logging.getLogger(__name__)

_LOOKBACK = 10          # number of recent trades for score calculation
_HOT_STREAK_MIN = 3     # consecutive wins to be "hot"
_COLD_STREAK_MIN = 3    # consecutive losses to be "cold"


@dataclass
class StreakInfo:
    """Current streak state for the profit momentum tracker.

    Attributes:
        win_streak: Consecutive winning trades.
        loss_streak: Consecutive losing trades.
        total_trades: Total trades recorded.
        recent_pnl: Deque of recent P&L values (most recent at right).
        is_hot: True when on a hot winning streak.
        is_cold: True when on a cold losing streak.
        momentum_score: Composite performance score in [0, 1].
        risk_multiplier: Recommended position size multiplier.
    """

    win_streak: int = 0
    loss_streak: int = 0
    total_trades: int = 0
    recent_pnl: Deque[float] = field(default_factory=lambda: deque(maxlen=_LOOKBACK))
    is_hot: bool = False
    is_cold: bool = False
    momentum_score: float = 0.5
    risk_multiplier: float = 1.0


class ProfitMomentumTracker:
    """Track profit/loss streaks and scale risk based on recent performance.

    Hot streaks allow a modest risk increase (up to 1.5×); cold streaks trigger
    a risk reduction (down to 0.5×).  Changes are gradual to prevent over-
    reaction to short runs of luck.

    Args:
        hot_streak_min: Consecutive wins to trigger hot streak (default 3).
        cold_streak_min: Consecutive losses to trigger cold streak (default 3).
        lookback: Number of recent trades for momentum score (default 10).
    """

    def __init__(
        self,
        hot_streak_min: int = _HOT_STREAK_MIN,
        cold_streak_min: int = _COLD_STREAK_MIN,
        lookback: int = _LOOKBACK,
    ) -> None:
        self.hot_streak_min = hot_streak_min
        self.cold_streak_min = cold_streak_min
        self._state = StreakInfo(recent_pnl=deque(maxlen=lookback))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, trade_profit: float) -> None:
        """Record a completed trade and refresh streak state.

        Args:
            trade_profit: P&L of the completed trade (positive = win).
        """
        self._state.total_trades += 1
        self._state.recent_pnl.append(trade_profit)

        if trade_profit > 0:
            self._state.win_streak += 1
            self._state.loss_streak = 0
        else:
            self._state.loss_streak += 1
            self._state.win_streak = 0

        self._state.is_hot = self._state.win_streak >= self.hot_streak_min
        self._state.is_cold = self._state.loss_streak >= self.cold_streak_min
        self._state.momentum_score = self._compute_momentum_score()
        self._state.risk_multiplier = self._compute_risk_multiplier()

        logger.debug(
            "ProfitMomentum: win_streak=%d loss_streak=%d hot=%s cold=%s score=%.3f",
            self._state.win_streak, self._state.loss_streak,
            self._state.is_hot, self._state.is_cold, self._state.momentum_score,
        )

    # ------------------------------------------------------------------
    # Internal computations
    # ------------------------------------------------------------------

    def _compute_momentum_score(self) -> float:
        """Compute composite momentum score from recent P&L.

        Returns:
            Score in [0, 1]; 0.5 = neutral, > 0.5 = positive momentum.
        """
        pnl = list(self._state.recent_pnl)
        if not pnl:
            return 0.5

        win_rate = sum(1 for p in pnl if p > 0) / len(pnl)

        # Recency-weighted win rate (more weight on recent trades)
        weights = [i + 1 for i in range(len(pnl))]
        weighted_wins = sum(w * float(p > 0) for w, p in zip(weights, pnl))
        total_w = sum(weights)
        weighted_wr = weighted_wins / total_w if total_w else 0.5

        # Blend equal and weighted win rates
        score = 0.4 * win_rate + 0.6 * weighted_wr
        return round(min(1.0, max(0.0, score)), 4)

    def _compute_risk_multiplier(self) -> float:
        """Derive risk multiplier from streak and momentum score.

        Returns:
            Multiplier in [0.5, 1.5].
        """
        score = self._state.momentum_score

        if self._state.is_hot:
            # Increase by up to 50% when hot, proportional to win streak
            extra = min(0.5, (self._state.win_streak - self.hot_streak_min) * 0.1)
            multiplier = 1.0 + extra
        elif self._state.is_cold:
            # Decrease proportionally to loss streak depth
            reduction = min(0.5, (self._state.loss_streak - self.cold_streak_min) * 0.1)
            multiplier = 1.0 - reduction
        else:
            # Neutral – scale linearly from 0.8 to 1.2 based on score
            multiplier = 0.8 + score * 0.4

        return round(max(0.5, min(1.5, multiplier)), 3)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_streak_info(self) -> StreakInfo:
        """Return current streak state snapshot.

        Returns:
            StreakInfo dataclass with all current streak metrics.
        """
        return self._state

    def get_risk_multiplier(self) -> float:
        """Return recommended position size multiplier.

        Returns:
            Multiplier in [0.5, 1.5].
        """
        return self._state.risk_multiplier

    def is_hot_streak(self) -> bool:
        """Return True when on a hot (winning) streak.

        Returns:
            True when win streak >= hot_streak_min.
        """
        return self._state.is_hot

    def is_cold_streak(self) -> bool:
        """Return True when on a cold (losing) streak.

        Returns:
            True when loss streak >= cold_streak_min.
        """
        return self._state.is_cold

    def get_momentum_score(self) -> float:
        """Return recent performance momentum score.

        Returns:
            Score in [0, 1]; 0.5 = neutral.
        """
        return self._state.momentum_score

    def reset(self) -> None:
        """Reset all streak and momentum state."""
        lookback = self._state.recent_pnl.maxlen or _LOOKBACK
        self._state = StreakInfo(recent_pnl=deque(maxlen=lookback))
        logger.info("ProfitMomentumTracker reset")

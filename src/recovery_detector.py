from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

_MAX_STREAK_LOSS = 10     # cap for normalisation
_MAX_STREAK_WIN = 5       # win streak cap before increasing risk


@dataclass
class RecoveryState:
    """Tracks loss/win streaks for the recovery detector.

    Attributes:
        loss_streak: Consecutive losing trades.
        win_streak: Consecutive winning trades since last loss.
        total_trades: Total trades recorded.
        in_recovery: True when actively recovering from a loss streak.
        last_result: Last trade result (positive = win).
    """

    loss_streak: int = 0
    win_streak: int = 0
    total_trades: int = 0
    in_recovery: bool = False
    last_result: float = 0.0


class RecoveryDetector:
    """Anti-revenge-trading position sizing and setup quality filter.

    After a loss streak the risk multiplier is reduced (never increased).
    Risk is only restored gradually once a winning streak is confirmed,
    preventing the psychological trap of doubling down to recoup losses.

    Args:
        recovery_win_streak: Consecutive wins needed to start restoring risk.
        max_risk_reduction: Floor for risk multiplier (default 0.5 = half size).
        setup_quality_threshold: Minimum setup score for recovery trades.
    """

    def __init__(
        self,
        recovery_win_streak: int = 3,
        max_risk_reduction: float = 0.5,
        setup_quality_threshold: float = 0.65,
    ) -> None:
        self.recovery_win_streak = recovery_win_streak
        self.max_risk_reduction = max_risk_reduction
        self.setup_quality_threshold = setup_quality_threshold
        self._state = RecoveryState()

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_loss_streak(self, trade_result: float) -> None:
        """Record a completed trade and update streak counters.

        Args:
            trade_result: P&L of the completed trade (positive = win).
        """
        self._state.total_trades += 1
        self._state.last_result = trade_result

        if trade_result < 0:
            self._state.loss_streak += 1
            self._state.win_streak = 0
            if self._state.loss_streak >= 2:
                self._state.in_recovery = True
            logger.info(
                "Loss recorded – streak=%d, in_recovery=%s",
                self._state.loss_streak,
                self._state.in_recovery,
            )
        else:
            self._state.win_streak += 1
            if self._state.win_streak >= self.recovery_win_streak:
                # Streak criterion met – begin exiting recovery
                if self._state.in_recovery:
                    logger.info(
                        "Recovery win streak (%d) reached – restoring risk",
                        self._state.win_streak,
                    )
                self._state.loss_streak = max(0, self._state.loss_streak - 1)
                if self._state.loss_streak == 0:
                    self._state.in_recovery = False

    # ------------------------------------------------------------------
    # Risk sizing
    # ------------------------------------------------------------------

    def get_recovery_factor(self) -> float:
        """Return a [0, 1] scale factor representing the severity of the drawdown.

        Higher value = more severe → applied as a multiplier to reduce risk.

        Returns:
            Factor in [0, 1]; 0 = no losses, 1 = maximum streak.
        """
        return min(self._state.loss_streak / _MAX_STREAK_LOSS, 1.0)

    def get_recommended_risk_multiplier(self) -> float:
        """Return the recommended position size multiplier.

        Anti-revenge logic:
          - No losses → 1.0 (normal size)
          - Loss streak → linearly reduced down to max_risk_reduction floor
          - The multiplier NEVER exceeds 1.0; it only reduces risk.

        Returns:
            Risk multiplier in [max_risk_reduction, 1.0].
        """
        factor = self.get_recovery_factor()
        # Linear decay from 1.0 to max_risk_reduction
        multiplier = 1.0 - factor * (1.0 - self.max_risk_reduction)
        multiplier = max(self.max_risk_reduction, min(1.0, multiplier))
        logger.debug(
            "Risk multiplier: %.3f (loss_streak=%d)", multiplier, self._state.loss_streak
        )
        return round(multiplier, 3)

    def should_increase_risk(self) -> bool:
        """Return True only when recovery conditions are fully met.

        Risk increase is only permitted after the win streak threshold has been
        reached AND the loss streak has been cleared to zero.

        Returns:
            True when it is safe to return to normal sizing.
        """
        return (
            not self._state.in_recovery
            and self._state.loss_streak == 0
            and self._state.win_streak >= self.recovery_win_streak
        )

    # ------------------------------------------------------------------
    # Setup quality gate
    # ------------------------------------------------------------------

    def detect_recovery_setup(
        self,
        df: pl.DataFrame,
        direction: int,
    ) -> bool:
        """Assess if the current chart pattern is high-quality enough during recovery.

        During a loss streak we apply a stricter quality gate to avoid
        low-probability setups.  This is a simplified heuristic that scores
        how clean the candle structure is on the most recent bars.

        Args:
            df: OHLCV polars DataFrame.
            direction: Intended trade direction (+1 long, -1 short).

        Returns:
            True when the setup meets the recovery quality threshold.
        """
        if not self._state.in_recovery:
            return True  # Not in recovery – use normal rules

        if len(df) < 5:
            return False

        tail = df.tail(5)
        closes = tail["close"].to_numpy()
        opens = tail["open"].to_numpy()

        # Count candles aligned with direction
        aligned = sum(
            1
            for o, c in zip(opens, closes)
            if (direction == 1 and c > o) or (direction == -1 and c < o)
        )
        alignment_score = aligned / 5.0

        # Price trending in the right direction over last 5 bars
        trend_ok = (direction == 1 and closes[-1] > closes[0]) or (
            direction == -1 and closes[-1] < closes[0]
        )

        quality_score = alignment_score * 0.6 + float(trend_ok) * 0.4
        meets_threshold = quality_score >= self.setup_quality_threshold

        logger.debug(
            "Recovery setup score=%.3f threshold=%.3f ok=%s",
            quality_score, self.setup_quality_threshold, meets_threshold,
        )
        return meets_threshold

    @property
    def state(self) -> RecoveryState:
        """Read-only view of the current recovery state."""
        return self._state

    def reset(self) -> None:
        """Reset all state to initial values."""
        self._state = RecoveryState()
        logger.info("RecoveryDetector state reset")

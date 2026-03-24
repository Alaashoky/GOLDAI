from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_MIN_SAMPLE_SIZE = 20  # Minimum trades before Kelly is trusted


@dataclass
class KellyStats:
    """Running statistics for Kelly criterion calculation.

    Attributes:
        total_trades: Number of trades recorded.
        wins: Number of winning trades.
        losses: Number of losing trades.
        total_win_amount: Cumulative win P&L (positive).
        total_loss_amount: Cumulative loss P&L (positive absolute value).
    """

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_win_amount: float = 0.0
    total_loss_amount: float = 0.0

    @property
    def win_rate(self) -> float:
        """Win rate as a fraction in [0, 1]."""
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def avg_win(self) -> float:
        """Average win amount (positive)."""
        if self.wins == 0:
            return 0.0
        return self.total_win_amount / self.wins

    @property
    def avg_loss(self) -> float:
        """Average loss amount (positive absolute value)."""
        if self.losses == 0:
            return 1.0  # prevent division by zero; treated as 1-unit loss
        return self.total_loss_amount / self.losses

    @property
    def has_sufficient_data(self) -> bool:
        """True when sample size meets the minimum requirement."""
        return self.total_trades >= _MIN_SAMPLE_SIZE


class KellyPositionScaler:
    """Kelly-criterion-based position sizing with safety caps.

    The full Kelly fraction is calculated as:

        f* = (W * R - L) / R

    where W = win_rate, L = loss_rate = (1 - W), R = avg_win / avg_loss.

    A fractional Kelly multiplier (default 0.25) prevents over-betting and
    protects against parameter estimation error, which is critical in live
    trading.

    Args:
        max_fractional_kelly: Cap on the Kelly fraction (default 0.25 = quarter-Kelly).
        min_sample_size: Minimum trades before Kelly is used (default 20).
    """

    def __init__(
        self,
        max_fractional_kelly: float = 0.25,
        min_sample_size: int = _MIN_SAMPLE_SIZE,
    ) -> None:
        self.max_fractional_kelly = max_fractional_kelly
        self.min_sample_size = min_sample_size
        self._stats = KellyStats()

    # ------------------------------------------------------------------
    # Static calculations
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_kelly_fraction(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Compute the full Kelly fraction.

        Args:
            win_rate: Fraction of trades that are winners, in [0, 1].
            avg_win: Average profit per winning trade (positive).
            avg_loss: Average loss per losing trade (positive absolute value).

        Returns:
            Full Kelly fraction in [0, 1] (clamped; negative fractions → 0).
        """
        if avg_loss <= 0:
            logger.warning("avg_loss must be > 0; returning 0")
            return 0.0
        if win_rate <= 0:
            return 0.0
        if win_rate >= 1.0:
            return 1.0

        loss_rate = 1.0 - win_rate
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (loss_rate / win_loss_ratio)
        return max(0.0, min(1.0, kelly))

    def calculate_lot_size(
        self,
        balance: float,
        kelly_fraction: float,
        risk_per_point: float,
        fractional_kelly: float = 0.25,
    ) -> float:
        """Convert Kelly fraction to a concrete lot size.

        Lot size = (balance × kelly_fraction × fractional_kelly) / risk_per_point

        Args:
            balance: Account equity in account currency.
            kelly_fraction: Full Kelly fraction from calculate_kelly_fraction.
            risk_per_point: Dollar risk per pip / point for 1 standard lot.
            fractional_kelly: Fractional Kelly multiplier (default 0.25).

        Returns:
            Recommended lot size rounded to 2 decimal places, minimum 0.01.
        """
        if risk_per_point <= 0 or balance <= 0:
            logger.warning("Invalid balance or risk_per_point for lot sizing")
            return 0.01

        effective_fraction = kelly_fraction * min(fractional_kelly, self.max_fractional_kelly)
        risk_capital = balance * effective_fraction
        lots = risk_capital / risk_per_point
        lots = max(0.01, round(lots, 2))
        logger.debug(
            "Lot size: balance=%.2f kelly=%.4f frac=%.2f risk_per_pt=%.2f → lots=%.2f",
            balance, kelly_fraction, fractional_kelly, risk_per_point, lots,
        )
        return lots

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def update_stats(self, trade_result: float) -> None:
        """Record a completed trade result.

        Args:
            trade_result: P&L of the trade (positive = win, negative = loss).
        """
        self._stats.total_trades += 1
        if trade_result > 0:
            self._stats.wins += 1
            self._stats.total_win_amount += trade_result
        else:
            self._stats.losses += 1
            self._stats.total_loss_amount += abs(trade_result)
        logger.debug(
            "KellyStats updated: trades=%d wins=%d losses=%d",
            self._stats.total_trades, self._stats.wins, self._stats.losses,
        )

    def get_recommended_fraction(self) -> float:
        """Return the current recommended (fractional) Kelly fraction.

        Returns 0.0 when insufficient data is available.

        Returns:
            Fractional Kelly value in [0, max_fractional_kelly].
        """
        if not self._stats.has_sufficient_data:
            logger.info(
                "Insufficient data (%d trades); returning 0 Kelly fraction",
                self._stats.total_trades,
            )
            return 0.0

        full_kelly = self.calculate_kelly_fraction(
            self._stats.win_rate,
            self._stats.avg_win,
            self._stats.avg_loss,
        )
        return round(full_kelly * self.max_fractional_kelly, 4)

    @property
    def stats(self) -> KellyStats:
        """Expose internal statistics (read-only view)."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset all tracked statistics."""
        self._stats = KellyStats()
        logger.info("KellyPositionScaler statistics reset")

"""Comprehensive risk management system for XAUUSD trading.

Provides dynamic lot sizing, drawdown protection, circuit breakers,
margin monitoring, and streak detection for adaptive risk control.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Deque, Dict, List, Optional

from src.utils import utc_now

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Snapshot of current risk state.

    Attributes:
        daily_drawdown_pct: Drawdown since session start as a fraction.
        weekly_drawdown_pct: Drawdown since week start as a fraction.
        consecutive_losses: Current run of losing trades.
        consecutive_wins: Current run of winning trades.
        total_exposure_lots: Sum of open lot sizes.
        margin_level_pct: Current margin level percentage.
        is_circuit_broken: Whether the circuit breaker has tripped.
    """

    daily_drawdown_pct: float = 0.0
    weekly_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    total_exposure_lots: float = 0.0
    margin_level_pct: float = 0.0
    is_circuit_broken: bool = False


@dataclass
class DrawdownState:
    """Tracks equity high-water marks for drawdown calculations.

    Attributes:
        daily_start_equity: Equity at the beginning of the trading day.
        weekly_start_equity: Equity at the beginning of the trading week.
        peak_equity: All-time high equity seen in current session.
        last_reset_date: Date of the last daily reset.
    """

    daily_start_equity: float = 0.0
    weekly_start_equity: float = 0.0
    peak_equity: float = 0.0
    last_reset_date: Optional[date] = None


class SmartRiskManager:
    """Manages position sizing and risk constraints for the trading bot.

    Applies dynamic lot sizing, daily/weekly loss limits, margin checks,
    and circuit breakers to protect the trading account.

    Attributes:
        max_daily_loss_pct: Maximum allowed daily drawdown fraction.
        max_weekly_loss_pct: Maximum allowed weekly drawdown fraction.
        max_consecutive_losses: Circuit-breaker loss streak threshold.
        max_margin_level_warning: Margin level that triggers a warning.
        circuit_breaker_pct: Equity drop fraction that halts trading.
        max_positions: Maximum concurrent open positions.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,
        max_weekly_loss_pct: float = 0.10,
        max_consecutive_losses: int = 5,
        max_margin_level_warning: float = 200.0,
        circuit_breaker_pct: float = 0.10,
        max_positions: int = 5,
        equity_history_size: int = 200,
    ) -> None:
        """Initialise the risk manager with configurable limits.

        Args:
            max_daily_loss_pct: Maximum daily loss as a fraction of starting equity.
            max_weekly_loss_pct: Maximum weekly loss as a fraction of starting equity.
            max_consecutive_losses: Number of consecutive losses before halting.
            max_margin_level_warning: Margin level (%) below which warnings fire.
            circuit_breaker_pct: Equity drop fraction that activates circuit breaker.
            max_positions: Maximum number of simultaneous open positions allowed.
            equity_history_size: Rolling window size for equity history tracking.
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_margin_level_warning = max_margin_level_warning
        self.circuit_breaker_pct = circuit_breaker_pct
        self.max_positions = max_positions

        self._metrics: RiskMetrics = RiskMetrics()
        self._drawdown: DrawdownState = DrawdownState()
        self._equity_history: Deque[float] = deque(maxlen=equity_history_size)
        self._trade_results: Deque[float] = deque(maxlen=100)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_lot_size(
        self,
        balance: float,
        risk_pct: float,
        sl_points: float,
        symbol_info: object,
        min_lot: float = 0.01,
        max_lot: float = 5.0,
    ) -> float:
        """Calculate position size using fixed fractional risk.

        The formula is: lot = (balance * risk_pct) / (sl_points * point_value)
        where point_value = contract_size * point.

        Args:
            balance: Current account balance.
            risk_pct: Risk fraction per trade, e.g. 0.02 for 2 %.
            sl_points: Stop-loss distance in points.
            symbol_info: Object with attributes ``point`` and ``contract_size``.
            min_lot: Minimum allowed lot size.
            max_lot: Maximum allowed lot size.

        Returns:
            float: Calculated lot size, clamped to [min_lot, max_lot].
        """
        if sl_points <= 0 or balance <= 0:
            return min_lot

        risk_amount = balance * risk_pct
        point = getattr(symbol_info, "point", 0.01)
        contract_size = getattr(symbol_info, "contract_size", 100.0)
        point_value = contract_size * point

        if point_value <= 0:
            return min_lot

        lot = risk_amount / (sl_points * point_value)
        lot = max(min_lot, min(max_lot, lot))

        # Snap to volume_step
        volume_step = getattr(symbol_info, "volume_step", 0.01)
        if volume_step > 0:
            lot = round(round(lot / volume_step) * volume_step, 8)

        logger.debug(
            "Lot size: balance=%.2f risk_pct=%.3f sl_pts=%.1f -> lot=%.2f",
            balance, risk_pct, sl_points, lot,
        )
        return lot

    def check_daily_drawdown(self, equity: float, daily_start: float) -> bool:
        """Check whether daily drawdown is within the configured limit.

        Args:
            equity: Current account equity.
            daily_start: Equity value at the start of the trading day.

        Returns:
            bool: True when trading is still allowed (drawdown is within limit).
        """
        if daily_start <= 0:
            return True
        drawdown = (daily_start - equity) / daily_start
        self._metrics.daily_drawdown_pct = max(0.0, drawdown)
        allowed = drawdown < self.max_daily_loss_pct
        if not allowed:
            logger.warning("Daily drawdown limit hit: %.2f%% >= %.2f%%", drawdown * 100, self.max_daily_loss_pct * 100)
        return allowed

    def check_weekly_drawdown(self, equity: float, weekly_start: float) -> bool:
        """Check whether weekly drawdown is within the configured limit.

        Args:
            equity: Current account equity.
            weekly_start: Equity value at the start of the trading week.

        Returns:
            bool: True when trading is still allowed.
        """
        if weekly_start <= 0:
            return True
        drawdown = (weekly_start - equity) / weekly_start
        self._metrics.weekly_drawdown_pct = max(0.0, drawdown)
        allowed = drawdown < self.max_weekly_loss_pct
        if not allowed:
            logger.warning("Weekly drawdown limit hit: %.2f%%", drawdown * 100)
        return allowed

    def check_correlation_risk(self, positions: List[object]) -> bool:
        """Prevent over-exposure by checking total open positions count.

        Args:
            positions: List of open position objects (any type with optional ``lot`` attribute).

        Returns:
            bool: True when adding another position is acceptable.
        """
        count = len(positions)
        if count >= self.max_positions:
            logger.warning("Correlation risk: max positions (%d) already open.", self.max_positions)
            return False
        total_lots = sum(getattr(p, "lot", 0.0) for p in positions)
        self._metrics.total_exposure_lots = total_lots
        return True

    def check_margin_level(self, margin_level: float) -> bool:
        """Verify the margin level is safely above the warning threshold.

        Args:
            margin_level: Current margin level percentage (0 means no open positions).

        Returns:
            bool: True when margin level is acceptable for trading.
        """
        self._metrics.margin_level_pct = margin_level
        if margin_level == 0:
            return True
        if margin_level < self.max_margin_level_warning:
            logger.warning("Low margin level: %.1f%%", margin_level)
            return False
        return True

    def circuit_breaker_check(self, equity: float, balance: float) -> bool:
        """Detect extreme equity loss and halt trading if necessary.

        Args:
            equity: Current account equity.
            balance: Current account balance (used as reference when no history).

        Returns:
            bool: True when trading should be halted (circuit breaker active).
        """
        reference = self._drawdown.peak_equity if self._drawdown.peak_equity > 0 else balance
        if reference <= 0:
            return False
        drop = (reference - equity) / reference
        if drop >= self.circuit_breaker_pct:
            self._metrics.is_circuit_broken = True
            logger.critical("CIRCUIT BREAKER ACTIVE: equity drop %.2f%% from peak.", drop * 100)
            return True
        return False

    def get_position_exposure(self, positions: List[object]) -> Dict[str, float]:
        """Calculate total and per-symbol lot exposure.

        Args:
            positions: List of open position objects with ``symbol`` and ``lot`` attributes.

        Returns:
            Dict[str, float]: Mapping of symbol -> total lots, plus "total" key.
        """
        exposure: Dict[str, float] = {}
        for pos in positions:
            symbol = getattr(pos, "symbol", "UNKNOWN")
            lot = getattr(pos, "lot", 0.0)
            exposure[symbol] = exposure.get(symbol, 0.0) + lot
        exposure["total"] = sum(v for k, v in exposure.items())
        self._metrics.total_exposure_lots = exposure["total"]
        return exposure

    def update_metrics(self, equity: float) -> None:
        """Record current equity and update drawdown high-water mark.

        Args:
            equity: Current account equity.
        """
        self._equity_history.append(equity)
        if equity > self._drawdown.peak_equity:
            self._drawdown.peak_equity = equity

        today = utc_now().date()
        if self._drawdown.last_reset_date != today:
            self._drawdown.daily_start_equity = equity
            self._drawdown.last_reset_date = today
            logger.info("Daily equity reset: %.2f", equity)

        if self._drawdown.weekly_start_equity == 0:
            self._drawdown.weekly_start_equity = equity

    def record_trade_result(self, profit: float) -> None:
        """Record a closed trade result and update streak counters.

        Args:
            profit: Net profit/loss of the closed trade.
        """
        self._trade_results.append(profit)
        if profit > 0:
            self._metrics.consecutive_wins += 1
            self._metrics.consecutive_losses = 0
        else:
            self._metrics.consecutive_losses += 1
            self._metrics.consecutive_wins = 0

        if self._metrics.consecutive_losses >= self.max_consecutive_losses:
            logger.warning("Max consecutive losses reached: %d", self._metrics.consecutive_losses)

    def is_on_hot_streak(self, min_wins: int = 3) -> bool:
        """Detect a winning hot streak.

        Args:
            min_wins: Minimum consecutive wins required.

        Returns:
            bool: True when currently on a hot streak.
        """
        return self._metrics.consecutive_wins >= min_wins

    def is_on_cold_streak(self, min_losses: int = 3) -> bool:
        """Detect a losing cold streak.

        Args:
            min_losses: Minimum consecutive losses required.

        Returns:
            bool: True when currently on a cold streak.
        """
        return self._metrics.consecutive_losses >= min_losses

    def get_metrics(self) -> RiskMetrics:
        """Return the current risk metrics snapshot.

        Returns:
            RiskMetrics: Current risk state.
        """
        return self._metrics

    def get_drawdown_state(self) -> DrawdownState:
        """Return the current drawdown tracking state.

        Returns:
            DrawdownState: Current drawdown state.
        """
        return self._drawdown

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (requires operator confirmation)."""
        self._metrics.is_circuit_broken = False
        logger.info("Circuit breaker reset by operator.")

"""Risk management engine.

Provides position sizing, drawdown tracking, risk-reward validation,
and kill switch functionality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class DrawdownTracker:
    start_balance: float = 0.0
    peak_balance: float = 0.0
    current_equity: float = 0.0
    daily_start: float = 0.0
    weekly_start: float = 0.0
    monthly_start: float = 0.0
    last_reset_day: int = 0
    last_reset_week: int = 0
    last_reset_month: int = 0

    @property
    def daily_drawdown(self) -> float:
        if self.daily_start <= 0:
            return 0.0
        return (self.daily_start - self.current_equity) / self.daily_start

    @property
    def weekly_drawdown(self) -> float:
        if self.weekly_start <= 0:
            return 0.0
        return (self.weekly_start - self.current_equity) / self.weekly_start

    @property
    def monthly_drawdown(self) -> float:
        if self.monthly_start <= 0:
            return 0.0
        return (self.monthly_start - self.current_equity) / self.monthly_start

    @property
    def max_drawdown(self) -> float:
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_equity) / self.peak_balance

@dataclass
class RiskCheck:
    allowed: bool = True
    reason: str = ""
    position_size: float = 0.0
    risk_amount: float = 0.0


class RiskEngine:
    """Risk management engine for position sizing and drawdown control.

    Attributes:
        max_risk_per_trade: Maximum risk per trade as decimal.
        max_daily_dd: Maximum daily drawdown as decimal.
        max_weekly_dd: Maximum weekly drawdown.
        max_monthly_dd: Maximum monthly drawdown.
        max_positions: Maximum concurrent positions.
        min_rr: Minimum risk-reward ratio.
    """

    def __init__(
        self,
        max_risk_per_trade: float = 0.02,
        max_daily_dd: float = 0.05,
        max_weekly_dd: float = 0.08,
        max_monthly_dd: float = 0.12,
        max_positions: int = 3,
        min_rr: float = 1.5,
    ) -> None:
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_dd = max_daily_dd
        self.max_weekly_dd = max_weekly_dd
        self.max_monthly_dd = max_monthly_dd
        self.max_positions = max_positions
        self.min_rr = min_rr
        self._kill_switch = False
        self.tracker = DrawdownTracker()

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch

    def initialize(self, balance: float) -> None:
        """Initialize the risk engine with account balance.

        Args:
            balance: Current account balance.
        """
        self.tracker.start_balance = balance
        self.tracker.peak_balance = balance
        self.tracker.current_equity = balance
        self.tracker.daily_start = balance
        self.tracker.weekly_start = balance
        self.tracker.monthly_start = balance
        now = datetime.now(timezone.utc)
        self.tracker.last_reset_day = now.day
        self.tracker.last_reset_week = now.isocalendar()[1]
        self.tracker.last_reset_month = now.month
        self._kill_switch = False
        logger.info("Risk engine initialized: balance=%.2f", balance)

    def update_equity(self, equity: float) -> None:
        """Update current equity and check drawdown levels.

        Args:
            equity: Current account equity.
        """
        self.tracker.current_equity = equity
        if equity > self.tracker.peak_balance:
            self.tracker.peak_balance = equity

        now = datetime.now(timezone.utc)
        if now.day != self.tracker.last_reset_day:
            self.tracker.daily_start = equity
            self.tracker.last_reset_day = now.day
        if now.isocalendar()[1] != self.tracker.last_reset_week:
            self.tracker.weekly_start = equity
            self.tracker.last_reset_week = now.isocalendar()[1]
        if now.month != self.tracker.last_reset_month:
            self.tracker.monthly_start = equity
            self.tracker.last_reset_month = now.month

        if self.tracker.daily_drawdown >= self.max_daily_dd:
            self._kill_switch = True
            logger.critical("KILL SWITCH: Daily DD %.2f%% >= %.2f%%",
                          self.tracker.daily_drawdown * 100, self.max_daily_dd * 100)

    def calculate_position_size(
        self,
        balance: float,
        risk_pct: Optional[float] = None,
        sl_points: float = 0.0,
        point_value: float = 0.01,
        contract_size: float = 100.0,
        min_lot: float = 0.01,
        max_lot: float = 1.0,
    ) -> float:
        """Calculate position size based on risk parameters.

        Args:
            balance: Account balance.
            risk_pct: Risk percentage (decimal). None uses default.
            sl_points: Stop loss in points.
            point_value: Value per point.
            contract_size: Contract size.
            min_lot: Minimum lot size.
            max_lot: Maximum lot size.

        Returns:
            float: Position size in lots.
        """
        risk = risk_pct or self.max_risk_per_trade
        risk_amount = balance * risk

        if sl_points <= 0:
            logger.warning("Invalid SL points: %.2f", sl_points)
            return min_lot

        pip_value_per_lot = contract_size * point_value
        sl_value = sl_points * pip_value_per_lot

        if sl_value <= 0:
            return min_lot

        lot_size = risk_amount / sl_value
        lot_size = max(min_lot, min(lot_size, max_lot))
        lot_size = round(lot_size, 2)

        logger.debug(
            "Position size: balance=%.2f risk=%.2f%% SL=%.0f pts -> %.2f lots",
            balance, risk * 100, sl_points, lot_size,
        )
        return lot_size

    def check_daily_drawdown(
        self,
        current_equity: float,
        start_of_day_balance: Optional[float] = None,
    ) -> bool:
        """Check if daily drawdown limit is exceeded.

        Args:
            current_equity: Current account equity.
            start_of_day_balance: Balance at start of trading day.

        Returns:
            bool: True if within limits.
        """
        start = start_of_day_balance or self.tracker.daily_start
        if start <= 0:
            return True
        dd = (start - current_equity) / start
        return dd < self.max_daily_dd

    def validate_risk_reward(
        self,
        sl_points: float,
        tp_points: float,
        min_rr: Optional[float] = None,
    ) -> bool:
        """Validate risk-reward ratio.

        Args:
            sl_points: Stop loss in points.
            tp_points: Take profit in points.
            min_rr: Minimum RR ratio. None uses default.

        Returns:
            bool: True if RR meets minimum requirement.
        """
        if sl_points <= 0:
            return False
        rr = tp_points / sl_points
        min_ratio = min_rr or self.min_rr
        is_valid = rr >= min_ratio
        if not is_valid:
            logger.debug("RR %.2f < min %.2f", rr, min_ratio)
        return is_valid

    def check_max_positions(self, current_positions: int) -> bool:
        """Check if max positions limit allows a new trade.

        Args:
            current_positions: Number of currently open positions.

        Returns:
            bool: True if new position is allowed.
        """
        return current_positions < self.max_positions

    def check_margin(
        self,
        free_margin: float,
        required_margin: float,
        buffer: float = 1.5,
    ) -> bool:
        """Check if there is enough free margin.

        Args:
            free_margin: Available free margin.
            required_margin: Required margin for the trade.
            buffer: Safety buffer multiplier.

        Returns:
            bool: True if enough margin.
        """
        return free_margin >= required_margin * buffer

    def full_check(
        self,
        balance: float,
        equity: float,
        free_margin: float,
        current_positions: int,
        sl_points: float,
        tp_points: float,
        required_margin: float = 0.0,
    ) -> RiskCheck:
        """Perform complete risk check before opening a trade.

        Args:
            balance: Account balance.
            equity: Account equity.
            free_margin: Free margin.
            current_positions: Number of open positions.
            sl_points: Stop loss in points.
            tp_points: Take profit in points.
            required_margin: Required margin for trade.

        Returns:
            RiskCheck: Complete risk assessment.
        """
        self.update_equity(equity)

        if self._kill_switch:
            return RiskCheck(allowed=False, reason="Kill switch activated")

        if not self.check_daily_drawdown(equity):
            return RiskCheck(allowed=False, reason="Daily drawdown exceeded")

        if not self.check_max_positions(current_positions):
            return RiskCheck(allowed=False, reason=f"Max positions ({self.max_positions}) reached")

        if not self.validate_risk_reward(sl_points, tp_points):
            return RiskCheck(allowed=False, reason=f"RR < {self.min_rr}")

        if required_margin > 0 and not self.check_margin(free_margin, required_margin):
            return RiskCheck(allowed=False, reason="Insufficient margin")

        pos_size = self.calculate_position_size(balance, sl_points=sl_points)
        risk_amount = balance * self.max_risk_per_trade

        return RiskCheck(
            allowed=True,
            position_size=pos_size,
            risk_amount=risk_amount,
        )

    def reset_kill_switch(self) -> None:
        """Manually reset the kill switch."""
        self._kill_switch = False
        logger.info("Kill switch reset")

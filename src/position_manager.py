"""Position management system for XAUUSD trading.

Handles trailing stops, breakeven moves, partial closes,
dynamic take-profit calculation, and early exit logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.utils import utc_now

logger = logging.getLogger(__name__)

DIRECTION_BUY  = "BUY"
DIRECTION_SELL = "SELL"


@dataclass
class PositionInfo:
    """All information about a single open position.

    Attributes:
        ticket: Unique position identifier.
        symbol: Trading instrument.
        type: "BUY" or "SELL".
        entry: Entry price.
        sl: Current stop-loss price.
        tp: Current take-profit price.
        lot: Position size in lots.
        profit: Unrealised P&L in account currency.
        open_time: UTC datetime when the position was opened.
        comment: Optional order comment.
    """

    ticket: int
    symbol: str
    type: str
    entry: float
    sl: float
    tp: float
    lot: float
    profit: float = 0.0
    open_time: datetime = field(default_factory=utc_now)
    comment: str = ""


@dataclass
class ManagementAction:
    """Describes an action to take on a position.

    Attributes:
        ticket: Target position ticket.
        action: One of "MODIFY_SL_TP", "PARTIAL_CLOSE", "CLOSE".
        new_sl: Updated stop-loss level (for MODIFY_SL_TP).
        new_tp: Updated take-profit level (for MODIFY_SL_TP).
        close_lot: Lot size to close (for PARTIAL_CLOSE).
        reason: Human-readable reason for the action.
    """

    ticket: int
    action: str
    new_sl: float = 0.0
    new_tp: float = 0.0
    close_lot: float = 0.0
    reason: str = ""


class PositionManager:
    """Manages open position lifecycle: trailing stops, breakeven, and exits.

    Attributes:
        trailing_atr_multiplier: ATR multiple used for trailing-stop distance.
        breakeven_buffer_points: Extra points added above entry for breakeven SL.
        partial_close_fraction: Fraction of the position to close partially.
        tp_atr_multiplier: ATR multiple used for dynamic take-profit.
    """

    def __init__(
        self,
        trailing_atr_multiplier: float = 1.5,
        breakeven_buffer_points: float = 5.0,
        partial_close_fraction: float = 0.5,
        tp_atr_multiplier: float = 2.5,
    ) -> None:
        """Initialise position manager with configurable parameters.

        Args:
            trailing_atr_multiplier: ATR multiplier for trailing stop distance.
            breakeven_buffer_points: Minimum profit points before moving to breakeven.
            partial_close_fraction: Fraction of lot closed on partial-close trigger.
            tp_atr_multiplier: ATR multiplier used when calculating dynamic TP.
        """
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.breakeven_buffer_points = breakeven_buffer_points
        self.partial_close_fraction = partial_close_fraction
        self.tp_atr_multiplier = tp_atr_multiplier
        self._breakeven_moved: Dict[int, bool] = {}
        self._partially_closed: Dict[int, bool] = {}

    # ------------------------------------------------------------------
    # Core management methods
    # ------------------------------------------------------------------

    def update_trailing_stop(
        self,
        position: PositionInfo,
        current_price: float,
        atr: float,
    ) -> Optional[float]:
        """Calculate a new trailing stop price for an open position.

        The trailing stop maintains a fixed ATR-based distance behind the
        current market price in the direction of the trade.

        Args:
            position: Open position details.
            current_price: Latest market price (bid for BUY, ask for SELL).
            atr: Average True Range value for the active timeframe.

        Returns:
            Optional[float]: New SL price if it improves the current SL,
                otherwise None (meaning no change is needed).
        """
        trail_distance = atr * self.trailing_atr_multiplier

        if position.type == DIRECTION_BUY:
            new_sl = current_price - trail_distance
            if new_sl > position.sl:
                logger.debug(
                    "Trailing BUY ticket=%d: SL %.2f -> %.2f",
                    position.ticket, position.sl, new_sl,
                )
                return round(new_sl, 2)
        else:  # SELL
            new_sl = current_price + trail_distance
            if new_sl < position.sl or position.sl == 0:
                logger.debug(
                    "Trailing SELL ticket=%d: SL %.2f -> %.2f",
                    position.ticket, position.sl, new_sl,
                )
                return round(new_sl, 2)

        return None

    def move_to_breakeven(
        self,
        position: PositionInfo,
        current_price: float,
        min_profit_points: float = 20.0,
    ) -> Optional[float]:
        """Shift the stop-loss to the entry price once minimum profit is reached.

        The SL is placed ``breakeven_buffer_points`` beyond the entry so the
        trade cannot close at a loss even if price briefly retraces.

        Args:
            position: Open position details.
            current_price: Latest market price.
            min_profit_points: Minimum distance in price points before moving.

        Returns:
            Optional[float]: Breakeven SL price if the move should be executed,
                otherwise None.
        """
        if self._breakeven_moved.get(position.ticket):
            return None

        buffer = self.breakeven_buffer_points

        if position.type == DIRECTION_BUY:
            profit_points = current_price - position.entry
            if profit_points >= min_profit_points:
                be_sl = position.entry + buffer
                if be_sl > position.sl:
                    self._breakeven_moved[position.ticket] = True
                    logger.info("Breakeven BUY ticket=%d: SL -> %.2f", position.ticket, be_sl)
                    return round(be_sl, 2)
        else:  # SELL
            profit_points = position.entry - current_price
            if profit_points >= min_profit_points:
                be_sl = position.entry - buffer
                if be_sl < position.sl or position.sl == 0:
                    self._breakeven_moved[position.ticket] = True
                    logger.info("Breakeven SELL ticket=%d: SL -> %.2f", position.ticket, be_sl)
                    return round(be_sl, 2)

        return None

    def check_partial_close(
        self,
        position: PositionInfo,
        current_price: float,
        threshold: float = 50.0,
    ) -> Optional[float]:
        """Determine whether a partial close should be executed.

        Args:
            position: Open position details.
            current_price: Latest market price.
            threshold: Profit in points required to trigger a partial close.

        Returns:
            Optional[float]: Lot size to close if a partial close is warranted,
                otherwise None.
        """
        if self._partially_closed.get(position.ticket):
            return None

        if position.type == DIRECTION_BUY:
            profit_points = current_price - position.entry
        else:
            profit_points = position.entry - current_price

        if profit_points >= threshold:
            close_lot = round(position.lot * self.partial_close_fraction, 2)
            close_lot = max(0.01, close_lot)
            self._partially_closed[position.ticket] = True
            logger.info(
                "Partial close ticket=%d: %.2f lots at profit_pts=%.1f",
                position.ticket, close_lot, profit_points,
            )
            return close_lot

        return None

    def monitor_positions(
        self,
        positions: List[PositionInfo],
        current_price: float,
        atr: float,
    ) -> List[ManagementAction]:
        """Evaluate all open positions and generate management actions.

        Checks trailing stops, breakeven moves, and partial closes for each
        position and returns a consolidated list of actions to execute.

        Args:
            positions: All currently open positions.
            current_price: Latest market price.
            atr: Average True Range for position sizing reference.

        Returns:
            List[ManagementAction]: Actions to execute (may be empty).
        """
        actions: List[ManagementAction] = []

        for pos in positions:
            new_sl: Optional[float] = None
            new_tp: Optional[float] = None

            # Breakeven takes priority over trailing
            be_sl = self.move_to_breakeven(pos, current_price)
            if be_sl is not None:
                new_sl = be_sl
                new_tp = pos.tp
            else:
                trail_sl = self.update_trailing_stop(pos, current_price, atr)
                if trail_sl is not None:
                    new_sl = trail_sl
                    new_tp = pos.tp

            if new_sl is not None:
                actions.append(
                    ManagementAction(
                        ticket=pos.ticket,
                        action="MODIFY_SL_TP",
                        new_sl=new_sl,
                        new_tp=new_tp or pos.tp,
                        reason="trailing/breakeven",
                    )
                )

            # Partial close
            close_lot = self.check_partial_close(pos, current_price)
            if close_lot is not None:
                actions.append(
                    ManagementAction(
                        ticket=pos.ticket,
                        action="PARTIAL_CLOSE",
                        close_lot=close_lot,
                        reason="partial_close_threshold",
                    )
                )

        return actions

    def calculate_dynamic_tp(
        self,
        entry: float,
        direction: str,
        atr: float,
        regime: str = "TRENDING",
    ) -> float:
        """Calculate a dynamic take-profit level based on ATR and market regime.

        Trending markets use a larger TP multiplier to capture extended moves,
        while ranging markets use a tighter TP to lock in profits quickly.

        Args:
            entry: Position entry price.
            direction: "BUY" or "SELL".
            atr: Average True Range value.
            regime: Market regime label: "TRENDING", "RANGING", or "VOLATILE".

        Returns:
            float: Calculated take-profit price.
        """
        regime_multipliers: Dict[str, float] = {
            "TRENDING": self.tp_atr_multiplier,
            "RANGING":  self.tp_atr_multiplier * 0.6,
            "VOLATILE": self.tp_atr_multiplier * 0.8,
        }
        multiplier = regime_multipliers.get(regime, self.tp_atr_multiplier)
        distance = atr * multiplier

        if direction == DIRECTION_BUY:
            tp = entry + distance
        else:
            tp = entry - distance

        return round(tp, 2)

    def should_exit_early(
        self,
        position: PositionInfo,
        smc_signal: str,
        fuzzy_score: float,
    ) -> Tuple[bool, str]:
        """Determine whether to exit a trade ahead of its take-profit.

        Exits early when the SMC signal has reversed against the open position
        and the fuzzy confidence for that reversal is sufficiently high.

        Args:
            position: Open position details.
            smc_signal: Current SMC signal direction ("BUY", "SELL", or "HOLD").
            fuzzy_score: Confidence score of the SMC signal (0.0–1.0).

        Returns:
            Tuple[bool, str]: (should_exit, reason_string).
        """
        opposite = DIRECTION_SELL if position.type == DIRECTION_BUY else DIRECTION_BUY

        if smc_signal == opposite and fuzzy_score >= 0.65:
            reason = f"SMC reversal signal {smc_signal} with confidence {fuzzy_score:.2f}"
            logger.info("Early exit ticket=%d: %s", position.ticket, reason)
            return True, reason

        # Also exit if the position has been in significant loss (risk management fallback)
        if position.profit < 0 and abs(position.profit) > 0:
            loss_pct = abs(position.profit)
            if loss_pct > 0 and fuzzy_score < 0.3 and smc_signal == opposite:
                reason = "weak confluence with counter-direction SMC"
                return True, reason

        return False, ""

    def cleanup_closed(self, ticket: int) -> None:
        """Remove tracking state for a closed position.

        Args:
            ticket: Ticket number of the closed position.
        """
        self._breakeven_moved.pop(ticket, None)
        self._partially_closed.pop(ticket, None)

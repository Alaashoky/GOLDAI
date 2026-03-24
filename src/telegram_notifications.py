from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

logger = logging.getLogger(__name__)


class TelegramNotifications:
    """Rich Telegram notification templates for GOLDAI.

    All messages use HTML parse mode (Telegram ``parse_mode="HTML"``).
    Emojis and bold/italic tags provide quick visual scanning on mobile.
    """

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def format_trade_entry(
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        lot: float,
        confidence: float,
        regime: str,
    ) -> str:
        """Format a trade-entry notification.

        Args:
            symbol: Instrument symbol (e.g. 'XAUUSD').
            direction: 'LONG' or 'SHORT'.
            entry: Entry price.
            sl: Stop-loss price.
            tp: Take-profit price.
            lot: Lot size.
            confidence: Signal confidence in [0, 1].
            regime: Market regime string ('trending', 'ranging', 'volatile').

        Returns:
            HTML-formatted Telegram message.
        """
        dir_icon = "🟢" if direction.upper() == "LONG" else "🔴"
        sl_pips = round(abs(entry - sl) * 10, 1)
        tp_pips = round(abs(tp - entry) * 10, 1)
        rr = round(tp_pips / sl_pips, 2) if sl_pips else 0.0

        return (
            f"{dir_icon} <b>NEW TRADE – {symbol}</b>\n"
            "━━━━━━━━━━━━━━━\n"
            f"Direction:   <b>{direction.upper()}</b>\n"
            f"Entry:       <b>{entry:.4f}</b>\n"
            f"Stop Loss:   <b>{sl:.4f}</b>  ({sl_pips} pips)\n"
            f"Take Profit: <b>{tp:.4f}</b>  ({tp_pips} pips)\n"
            f"R:R Ratio:   <b>1 : {rr}</b>\n"
            f"Lot Size:    <b>{lot:.2f}</b>\n"
            f"Confidence:  <b>{confidence:.0%}</b>\n"
            f"Regime:      <i>{regime.capitalize()}</i>\n"
        )

    @staticmethod
    def format_trade_exit(
        symbol: str,
        direction: str,
        profit_pips: float,
        profit_usd: float,
        duration: timedelta | float,
        reason: str,
    ) -> str:
        """Format a trade-exit notification.

        Args:
            symbol: Instrument symbol.
            direction: 'LONG' or 'SHORT'.
            profit_pips: Profit/loss in pips (negative = loss).
            profit_usd: Profit/loss in USD.
            duration: Trade duration as timedelta or hours (float).
            reason: Exit reason string.

        Returns:
            HTML-formatted Telegram message.
        """
        win = profit_usd >= 0
        result_icon = "✅" if win else "❌"
        pnl_icon = "📈" if win else "📉"

        if isinstance(duration, timedelta):
            hours = duration.total_seconds() / 3600
        else:
            hours = float(duration)
        dur_str = f"{hours:.1f}h"

        return (
            f"{result_icon} <b>TRADE CLOSED – {symbol}</b>\n"
            "━━━━━━━━━━━━━━━\n"
            f"Direction:  <b>{direction.upper()}</b>\n"
            f"Result:     {pnl_icon} <b>{profit_pips:+.1f} pips</b>  "
            f"(<b>${profit_usd:+,.2f}</b>)\n"
            f"Duration:   <b>{dur_str}</b>\n"
            f"Reason:     <i>{reason}</i>\n"
        )

    # ------------------------------------------------------------------
    # Periodic reports
    # ------------------------------------------------------------------

    @staticmethod
    def format_daily_summary(
        stats: dict[str, Any],
        equity: float,
        trades_today: int,
    ) -> str:
        """Format the end-of-day summary notification.

        Args:
            stats: Dict with keys: wins, losses, net_pnl, win_rate, best_trade,
                   worst_trade.
            equity: Current account equity.
            trades_today: Number of trades taken today.

        Returns:
            HTML-formatted daily summary message.
        """
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        net_pnl = stats.get("net_pnl", 0.0)
        win_rate = stats.get("win_rate", 0.0)
        best = stats.get("best_trade", 0.0)
        worst = stats.get("worst_trade", 0.0)
        pnl_icon = "📈" if net_pnl >= 0 else "📉"

        return (
            "📅 <b>Daily Summary</b>\n"
            "━━━━━━━━━━━━━━━\n"
            f"Trades:      <b>{trades_today}</b>  "
            f"(✅ {wins} / ❌ {losses})\n"
            f"Win Rate:    <b>{win_rate:.1%}</b>\n"
            f"Net P&L:     {pnl_icon} <b>${net_pnl:+,.2f}</b>\n"
            f"Best Trade:  📈 <b>${best:+,.2f}</b>\n"
            f"Worst Trade: 📉 <b>${worst:+,.2f}</b>\n"
            f"Equity:      💰 <b>${equity:,.2f}</b>\n"
        )

    @staticmethod
    def format_weekly_summary(
        stats: dict[str, Any],
        weekly_pnl: float,
    ) -> str:
        """Format the end-of-week summary notification.

        Args:
            stats: Dict with keys: total_trades, win_rate, sharpe, max_drawdown.
            weekly_pnl: Total P&L for the week.

        Returns:
            HTML-formatted weekly summary message.
        """
        total = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0.0)
        sharpe = stats.get("sharpe", 0.0)
        max_dd = stats.get("max_drawdown", 0.0)
        pnl_icon = "📈" if weekly_pnl >= 0 else "📉"

        return (
            "📆 <b>Weekly Summary</b>\n"
            "━━━━━━━━━━━━━━━\n"
            f"Total Trades:  <b>{total}</b>\n"
            f"Win Rate:      <b>{win_rate:.1%}</b>\n"
            f"Weekly P&L:    {pnl_icon} <b>${weekly_pnl:+,.2f}</b>\n"
            f"Sharpe Ratio:  <b>{sharpe:.2f}</b>\n"
            f"Max Drawdown:  ⚠️ <b>{max_dd:.2%}</b>\n"
        )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    @staticmethod
    def format_error_alert(error_type: str, message: str) -> str:
        """Format an error/alert notification.

        Args:
            error_type: Short error category (e.g. 'ConnectionError').
            message: Detailed error message.

        Returns:
            HTML-formatted error alert message.
        """
        return (
            "⚠️ <b>GOLDAI Alert</b>\n"
            "━━━━━━━━━━━━━━━\n"
            f"Type:    <b>{error_type}</b>\n"
            f"Message: <i>{message}</i>\n"
        )

    @staticmethod
    def format_risk_alert(
        alert_type: str,
        current_value: float,
        threshold: float,
    ) -> str:
        """Format a risk limit breach notification.

        Args:
            alert_type: What limit was breached (e.g. 'Daily Loss').
            current_value: Current value that breached the limit.
            threshold: The configured limit value.

        Returns:
            HTML-formatted risk alert.
        """
        return (
            "🚨 <b>Risk Alert</b>\n"
            "━━━━━━━━━━━━━━━\n"
            f"Limit:   <b>{alert_type}</b>\n"
            f"Current: <b>{current_value:.2%}</b>\n"
            f"Limit:   <b>{threshold:.2%}</b>\n"
            "Trading suspended for this session.\n"
        )

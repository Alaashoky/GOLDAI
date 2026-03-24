from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Command constants
CMD_STATUS = "/status"
CMD_POSITIONS = "/positions"
CMD_BALANCE = "/balance"
CMD_PERFORMANCE = "/performance"
CMD_PAUSE = "/pause"
CMD_RESUME = "/resume"
CMD_HELP = "/help"

_PAUSED = False  # module-level pause flag


class TelegramCommands:
    """Handler for Telegram bot commands.

    Each handler receives the relevant application object(s) and returns a
    formatted string ready to send back to the user via the Telegram API.

    Usage::

        cmds = TelegramCommands()
        text = cmds.parse_command("/status")
        # Route to the appropriate handler with injected dependencies.
    """

    # ------------------------------------------------------------------
    # Pause / resume state
    # ------------------------------------------------------------------

    @staticmethod
    def handle_pause() -> str:
        """Handle /pause command – suspend new trade entries.

        Returns:
            Confirmation message string.
        """
        global _PAUSED
        _PAUSED = True
        logger.info("Bot paused via Telegram command")
        return "⏸️ <b>Bot Paused</b>\nNew trade entries are suspended. Use /resume to restart."

    @staticmethod
    def handle_resume() -> str:
        """Handle /resume command – re-enable trade entries.

        Returns:
            Confirmation message string.
        """
        global _PAUSED
        _PAUSED = False
        logger.info("Bot resumed via Telegram command")
        return "▶️ <b>Bot Resumed</b>\nTrade entries are now active."

    @staticmethod
    def is_paused() -> bool:
        """Return current pause state.

        Returns:
            True when the bot is paused.
        """
        return _PAUSED

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @staticmethod
    def handle_status(connector: Any, risk_manager: Any) -> str:
        """Handle /status command – show bot health and key metrics.

        Args:
            connector: MT5 / broker connector object with account info methods.
            risk_manager: Risk manager with daily loss and drawdown metrics.

        Returns:
            Formatted HTML status string.
        """
        try:
            balance = getattr(connector, "get_balance", lambda: "N/A")()
            equity = getattr(connector, "get_equity", lambda: "N/A")()
            daily_loss = getattr(risk_manager, "get_daily_loss_pct", lambda: 0.0)()
            status_icon = "⏸️" if _PAUSED else "🟢"
            mode = "PAUSED" if _PAUSED else "ACTIVE"

            return (
                f"{status_icon} <b>GOLDAI Status</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"Mode: <b>{mode}</b>\n"
                f"Balance: <b>${balance:,.2f}</b>\n"
                f"Equity: <b>${equity:,.2f}</b>\n"
                f"Daily Loss: <b>{daily_loss:.2%}</b>\n"
            )
        except Exception as exc:
            logger.error("handle_status error: %s", exc)
            return f"⚠️ Status unavailable: {exc}"

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    @staticmethod
    def handle_positions(connector: Any) -> str:
        """Handle /positions command – list open positions.

        Args:
            connector: Broker connector with get_open_positions() method.

        Returns:
            Formatted HTML positions list.
        """
        try:
            positions = getattr(connector, "get_open_positions", lambda: [])()
            if not positions:
                return "📋 <b>Open Positions</b>\n━━━━━━━━━━━━━━━\nNo open positions."

            lines = ["📋 <b>Open Positions</b>", "━━━━━━━━━━━━━━━"]
            for pos in positions:
                direction = "🟢 LONG" if getattr(pos, "type", 0) == 0 else "🔴 SHORT"
                symbol = getattr(pos, "symbol", "XAUUSD")
                lots = getattr(pos, "volume", 0.0)
                profit = getattr(pos, "profit", 0.0)
                profit_icon = "📈" if profit >= 0 else "📉"
                lines.append(
                    f"{direction} {symbol} | {lots:.2f} lots | "
                    f"{profit_icon} ${profit:.2f}"
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.error("handle_positions error: %s", exc)
            return f"⚠️ Positions unavailable: {exc}"

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    @staticmethod
    def handle_balance(connector: Any) -> str:
        """Handle /balance command – show account balance details.

        Args:
            connector: Broker connector with account info methods.

        Returns:
            Formatted HTML balance string.
        """
        try:
            balance = getattr(connector, "get_balance", lambda: 0.0)()
            equity = getattr(connector, "get_equity", lambda: 0.0)()
            margin = getattr(connector, "get_margin", lambda: 0.0)()
            free_margin = getattr(connector, "get_free_margin", lambda: 0.0)()

            return (
                "💰 <b>Account Balance</b>\n"
                "━━━━━━━━━━━━━━━\n"
                f"Balance:     <b>${balance:,.2f}</b>\n"
                f"Equity:      <b>${equity:,.2f}</b>\n"
                f"Margin:      <b>${margin:,.2f}</b>\n"
                f"Free Margin: <b>${free_margin:,.2f}</b>\n"
            )
        except Exception as exc:
            logger.error("handle_balance error: %s", exc)
            return f"⚠️ Balance unavailable: {exc}"

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    @staticmethod
    def handle_performance(trade_logger: Any) -> str:
        """Handle /performance command – show trading performance summary.

        Args:
            trade_logger: Trade logger / analytics object with summary methods.

        Returns:
            Formatted HTML performance string.
        """
        try:
            summary = getattr(trade_logger, "get_summary", lambda: {})()
            wins = summary.get("wins", 0)
            losses = summary.get("losses", 0)
            total = wins + losses
            win_rate = wins / total if total else 0.0
            net_pnl = summary.get("net_pnl", 0.0)
            pnl_icon = "📈" if net_pnl >= 0 else "📉"

            return (
                f"📊 <b>Performance Summary</b>\n"
                "━━━━━━━━━━━━━━━\n"
                f"Total Trades: <b>{total}</b>\n"
                f"Wins: <b>{wins}</b> | Losses: <b>{losses}</b>\n"
                f"Win Rate: <b>{win_rate:.1%}</b>\n"
                f"Net P&L: {pnl_icon} <b>${net_pnl:,.2f}</b>\n"
            )
        except Exception as exc:
            logger.error("handle_performance error: %s", exc)
            return f"⚠️ Performance data unavailable: {exc}"

    # ------------------------------------------------------------------
    # Command parser
    # ------------------------------------------------------------------

    @staticmethod
    def parse_command(text: str) -> Optional[str]:
        """Extract the command keyword from a Telegram message.

        Args:
            text: Raw message text from the Telegram update.

        Returns:
            The command string (e.g. '/status') or None if not a command.
        """
        if not text or not text.startswith("/"):
            return None
        # Strip @botname suffix if present and extract first word
        parts = text.strip().split()
        cmd = parts[0].split("@")[0].lower()
        known = {
            CMD_STATUS, CMD_POSITIONS, CMD_BALANCE,
            CMD_PERFORMANCE, CMD_PAUSE, CMD_RESUME, CMD_HELP,
        }
        return cmd if cmd in known else None

    @staticmethod
    def handle_help() -> str:
        """Return the help text listing all commands.

        Returns:
            Formatted HTML help string.
        """
        return (
            "🤖 <b>GOLDAI Commands</b>\n"
            "━━━━━━━━━━━━━━━\n"
            "/status      – Bot health and key metrics\n"
            "/positions   – Open positions list\n"
            "/balance     – Account balance details\n"
            "/performance – Trading performance summary\n"
            "/pause       – Suspend new entries\n"
            "/resume      – Re-enable entries\n"
            "/help        – This message\n"
        )

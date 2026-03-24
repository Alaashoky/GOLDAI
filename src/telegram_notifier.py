"""Telegram notification system for GOLDAI trading alerts.

Provides async message delivery with an internal queue, rate limiting,
emoji-rich formatting, and automatic retry on transient failures.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Optional

from src.utils import utc_now

logger = logging.getLogger(__name__)

# Telegram API allows ~30 messages per minute per bot
_RATE_LIMIT_PER_MINUTE = 30
_RATE_WINDOW_SECONDS = 60.0


@dataclass
class TelegramMessage:
    """A message waiting in the delivery queue.

    Attributes:
        text: Message body (Markdown-safe).
        enqueued_at: UTC time the message entered the queue.
        attempts: Number of delivery attempts so far.
    """

    text: str
    enqueued_at: datetime = field(default_factory=utc_now)
    attempts: int = 0


class TelegramNotifier:
    """Sends structured trade notifications via the Telegram Bot API.

    Messages are queued internally and dispatched respecting Telegram's
    rate limit of 30 messages per minute. Each send is retried up to
    three times on transient network or API errors.

    Attributes:
        bot_token: Telegram bot API token.
        chat_id: Target chat / channel identifier.
        max_retries: Maximum delivery attempts per message.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        max_retries: int = 3,
        parse_mode: str = "HTML",
    ) -> None:
        """Initialise the notifier.

        Args:
            bot_token: Telegram Bot API token (from BotFather).
            chat_id: Destination chat or channel ID.
            max_retries: Number of send attempts before a message is dropped.
            parse_mode: Telegram parse mode; "HTML" or "Markdown".
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.max_retries = max_retries
        self.parse_mode = parse_mode
        self._queue: Deque[TelegramMessage] = deque()
        self._sent_this_window: int = 0
        self._window_start: float = 0.0
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_message(self, text: str) -> bool:
        """Enqueue a plain-text message for delivery.

        Args:
            text: Message body (HTML or Markdown depending on parse_mode).

        Returns:
            bool: True when the message was delivered successfully.
        """
        return await self._enqueue_and_send(text)

    async def send_trade_alert(
        self,
        direction: str,
        symbol: str,
        entry: float,
        sl: float,
        tp: float,
        lot: float,
        confidence: float,
    ) -> bool:
        """Send a formatted trade entry alert.

        Args:
            direction: "BUY" or "SELL".
            symbol: Trading instrument, e.g. "XAUUSD".
            entry: Entry price.
            sl: Stop-loss price.
            tp: Take-profit price.
            lot: Position size in lots.
            confidence: Signal confidence score (0.0–1.0).

        Returns:
            bool: True on successful delivery.
        """
        arrow = "📈" if direction == "BUY" else "📉"
        rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0.0
        text = (
            f"{arrow} <b>NEW TRADE ALERT</b>\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"🪙 <b>Symbol:</b>     {symbol}\n"
            f"📊 <b>Direction:</b>  {direction}\n"
            f"💵 <b>Entry:</b>      {entry:.2f}\n"
            f"🛑 <b>Stop Loss:</b>  {sl:.2f}\n"
            f"🎯 <b>Take Profit:</b> {tp:.2f}\n"
            f"📦 <b>Lot Size:</b>   {lot:.2f}\n"
            f"⚖️  <b>R:R Ratio:</b>  1:{rr:.1f}\n"
            f"🔥 <b>Confidence:</b> {confidence * 100:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"🕐 {utc_now().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return await self.send_message(text)

    async def send_error(self, error_msg: str) -> bool:
        """Send an error notification.

        Args:
            error_msg: Error description.

        Returns:
            bool: True on successful delivery.
        """
        text = (
            f"🚨 <b>ERROR</b>\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"{error_msg}\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"🕐 {utc_now().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return await self.send_message(text)

    async def send_status_update(
        self,
        equity: float,
        positions_count: int,
        daily_pnl: float,
    ) -> bool:
        """Send a periodic account status summary.

        Args:
            equity: Current account equity.
            positions_count: Number of open positions.
            daily_pnl: Today's profit/loss in account currency.

        Returns:
            bool: True on successful delivery.
        """
        pnl_icon = "💚" if daily_pnl >= 0 else "🔴"
        text = (
            f"📊 <b>BOT STATUS UPDATE</b>\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Equity:</b>     ${equity:,.2f}\n"
            f"📂 <b>Positions:</b>  {positions_count}\n"
            f"{pnl_icon} <b>Daily P&amp;L:</b>  ${daily_pnl:+,.2f}\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"🕐 {utc_now().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return await self.send_message(text)

    async def send_position_closed(
        self,
        ticket: int,
        profit: float,
        duration: str,
    ) -> bool:
        """Send a position-closed notification.

        Args:
            ticket: Closed position ticket number.
            profit: Net profit/loss in account currency.
            duration: Human-readable trade duration string (e.g. "2h 15m").

        Returns:
            bool: True on successful delivery.
        """
        icon = "✅" if profit >= 0 else "❌"
        text = (
            f"{icon} <b>POSITION CLOSED</b>\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"🎫 <b>Ticket:</b>   #{ticket}\n"
            f"{'💰' if profit >= 0 else '💸'} <b>Profit:</b>   ${profit:+,.2f}\n"
            f"⏱️  <b>Duration:</b> {duration}\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"🕐 {utc_now().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return await self.send_message(text)

    # ------------------------------------------------------------------
    # Internal queue and HTTP delivery
    # ------------------------------------------------------------------

    async def _enqueue_and_send(self, text: str) -> bool:
        """Add a message to the queue and attempt immediate delivery.

        Args:
            text: Formatted message text.

        Returns:
            bool: True when the message was delivered.
        """
        msg = TelegramMessage(text=text)
        self._queue.append(msg)
        return await self._flush_queue()

    async def _flush_queue(self) -> bool:
        """Deliver all queued messages, respecting the rate limit.

        Returns:
            bool: True if the most recently enqueued message was delivered.
        """
        last_result = False
        while self._queue:
            await self._wait_for_rate_limit()
            msg = self._queue.popleft()
            success = await self._deliver(msg)
            if success:
                last_result = True
            else:
                if msg.attempts < self.max_retries:
                    self._queue.appendleft(msg)
                else:
                    logger.error("Message dropped after %d attempts: %s", msg.attempts, msg.text[:80])
        return last_result

    async def _wait_for_rate_limit(self) -> None:
        """Pause execution if the rate window is exhausted."""
        import time
        now = time.monotonic()
        if now - self._window_start >= _RATE_WINDOW_SECONDS:
            self._window_start = now
            self._sent_this_window = 0

        if self._sent_this_window >= _RATE_LIMIT_PER_MINUTE:
            sleep_secs = _RATE_WINDOW_SECONDS - (now - self._window_start) + 0.1
            logger.debug("Rate limit reached — sleeping %.1fs.", sleep_secs)
            await asyncio.sleep(sleep_secs)
            self._window_start = time.monotonic()
            self._sent_this_window = 0

    async def _deliver(self, msg: TelegramMessage) -> bool:
        """Send a single message to the Telegram API.

        Args:
            msg: Message object with text and attempt counter.

        Returns:
            bool: True on HTTP 200 with ok=true response.
        """
        msg.attempts += 1

        if not self.bot_token or not self.chat_id:
            logger.debug("Telegram not configured — message suppressed.")
            return True  # Silently succeed to avoid blocking bot logic

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id":    self.chat_id,
            "text":       msg.text,
            "parse_mode": self.parse_mode,
        }

        try:
            import aiohttp  # type: ignore
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        self._sent_this_window += 1
                        logger.debug("Telegram message sent (attempt %d).", msg.attempts)
                        return True
                    body = await resp.text()
                    logger.warning("Telegram API error %d: %s", resp.status, body[:200])
                    return False
        except ImportError:
            # aiohttp not installed; fall back to stdlib
            return await self._deliver_stdlib(msg, url, payload)
        except Exception as exc:
            logger.warning("Telegram send failed (attempt %d): %s", msg.attempts, exc)
            if msg.attempts < self.max_retries:
                await asyncio.sleep(2 ** msg.attempts)
            return False

    async def _deliver_stdlib(
        self,
        msg: TelegramMessage,
        url: str,
        payload: dict,
    ) -> bool:
        """Fallback delivery using urllib when aiohttp is unavailable.

        Args:
            msg: Message being delivered.
            url: Telegram API endpoint.
            payload: JSON-serialisable request body.

        Returns:
            bool: True on success.
        """
        import json
        import urllib.request
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    self._sent_this_window += 1
                    return True
        except Exception as exc:
            logger.warning("urllib Telegram fallback failed: %s", exc)
        return False

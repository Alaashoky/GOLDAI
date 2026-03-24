"""Trade logging system for GOLDAI.

Records every trade entry and exit to CSV files and a JSON summary,
and computes performance statistics including Sharpe ratio, win rate,
and profit factor.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils import utc_now

logger = logging.getLogger(__name__)

_CSV_FIELDNAMES = [
    "ticket", "symbol", "direction", "entry_price", "exit_price",
    "sl", "tp", "lot", "confidence", "profit", "exit_reason",
    "entry_time", "exit_time", "duration_minutes",
]


@dataclass
class TradeRecord:
    """Complete record of a single round-trip trade.

    Attributes:
        ticket: Unique position identifier.
        symbol: Trading instrument.
        direction: "BUY" or "SELL".
        entry_price: Fill price at open.
        sl: Stop-loss level at entry.
        tp: Take-profit level at entry.
        lot: Position size in lots.
        confidence: Signal confidence score at entry (0.0–1.0).
        entry_time: UTC datetime the position was opened.
        exit_price: Fill price at close (0 if still open).
        profit: Realised profit/loss in account currency.
        exit_reason: Why the trade closed, e.g. "TP", "SL", "MANUAL".
        exit_time: UTC datetime the position was closed (None if open).
    """

    ticket: int
    symbol: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    lot: float
    confidence: float = 0.0
    entry_time: datetime = field(default_factory=utc_now)
    exit_price: float = 0.0
    profit: float = 0.0
    exit_reason: str = ""
    exit_time: Optional[datetime] = None

    @property
    def is_closed(self) -> bool:
        """Whether the trade has been closed."""
        return self.exit_time is not None

    @property
    def duration_minutes(self) -> float:
        """Trade duration in minutes, or 0 if still open."""
        if not self.exit_time:
            return 0.0
        return (self.exit_time - self.entry_time).total_seconds() / 60.0

    @property
    def is_winner(self) -> bool:
        """Whether the trade made a profit."""
        return self.profit > 0


class TradeLogger:
    """Persists trade records to CSV and JSON, and computes statistics.

    All closed trades are appended to ``trades.csv`` inside ``log_dir``.
    An aggregated ``summary.json`` is rewritten after each update.

    Attributes:
        log_dir: Directory where log files are written.
        csv_path: Full path to the CSV trade log.
        summary_path: Full path to the JSON summary file.
    """

    def __init__(self, log_dir: str = "data/trades") -> None:
        """Initialise the trade logger.

        Args:
            log_dir: Directory for log files (created automatically).
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "trades.csv")
        self.summary_path = os.path.join(log_dir, "summary.json")
        self._open_trades: Dict[int, TradeRecord] = {}
        self._closed_trades: List[TradeRecord] = []
        self._ensure_csv_header()
        self._load_existing_csv()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_entry(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        lot: float,
        confidence: float = 0.0,
    ) -> None:
        """Record a new trade entry.

        Args:
            ticket: Unique position ticket.
            symbol: Trading instrument.
            direction: "BUY" or "SELL".
            entry_price: Execution entry price.
            sl: Stop-loss level.
            tp: Take-profit level.
            lot: Position size in lots.
            confidence: Signal confidence at entry.
        """
        record = TradeRecord(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            lot=lot,
            confidence=confidence,
            entry_time=utc_now(),
        )
        self._open_trades[ticket] = record
        logger.info(
            "Trade ENTRY logged: ticket=%d %s %s @ %.2f lot=%.2f conf=%.2f",
            ticket, direction, symbol, entry_price, lot, confidence,
        )

    def log_exit(
        self,
        ticket: int,
        exit_price: float,
        profit: float,
        reason: str = "UNKNOWN",
    ) -> Optional[TradeRecord]:
        """Record the close of an existing trade.

        Args:
            ticket: Position ticket to close.
            exit_price: Execution exit price.
            profit: Realised profit/loss.
            reason: Exit reason, e.g. "TP", "SL", "MANUAL", "EARLY_EXIT".

        Returns:
            Optional[TradeRecord]: The completed record, or None if ticket unknown.
        """
        record = self._open_trades.pop(ticket, None)
        if record is None:
            logger.warning("log_exit: no open trade with ticket=%d", ticket)
            return None

        record.exit_price = exit_price
        record.profit = profit
        record.exit_reason = reason
        record.exit_time = utc_now()

        self._closed_trades.append(record)
        self._append_csv(record)
        self._save_summary()

        icon = "✅" if profit > 0 else "❌"
        logger.info(
            "%s Trade EXIT logged: ticket=%d profit=%.2f reason=%s",
            icon, ticket, profit, reason,
        )
        return record

    def get_performance_stats(self) -> Dict[str, Any]:
        """Compute aggregate trading performance statistics.

        Returns:
            Dict[str, Any]: Statistics including win_rate, avg_profit,
                total_trades, profit_factor, sharpe_ratio, and more.
        """
        trades = self._closed_trades
        if not trades:
            return self._empty_stats()

        profits = [t.profit for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]

        total = len(trades)
        win_rate = len(wins) / total if total else 0.0
        avg_profit = sum(profits) / total if total else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        sharpe = self._calculate_sharpe(profits)
        max_dd = self._calculate_max_drawdown(profits)

        return {
            "total_trades":    total,
            "win_rate":        round(win_rate, 4),
            "avg_profit":      round(avg_profit, 2),
            "gross_profit":    round(gross_profit, 2),
            "gross_loss":      round(gross_loss, 2),
            "profit_factor":   round(profit_factor, 3),
            "sharpe_ratio":    round(sharpe, 3),
            "max_drawdown":    round(max_dd, 2),
            "net_profit":      round(sum(profits), 2),
            "best_trade":      round(max(profits), 2),
            "worst_trade":     round(min(profits), 2),
            "avg_win":         round(sum(wins) / len(wins), 2) if wins else 0.0,
            "avg_loss":        round(sum(losses) / len(losses), 2) if losses else 0.0,
            "open_trades":     len(self._open_trades),
        }

    def get_recent_trades(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent N closed trades as plain dictionaries.

        Args:
            n: Number of recent trades to return.

        Returns:
            List[Dict[str, Any]]: List of trade dictionaries, newest first.
        """
        recent = self._closed_trades[-n:][::-1]
        result = []
        for t in recent:
            d = asdict(t)
            d["entry_time"] = t.entry_time.isoformat()
            d["exit_time"] = t.exit_time.isoformat() if t.exit_time else None
            d["duration_minutes"] = round(t.duration_minutes, 1)
            d["is_winner"] = t.is_winner
            result.append(d)
        return result

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Return all currently open (unexited) trades.

        Returns:
            List[Dict[str, Any]]: Open trade records.
        """
        result = []
        for t in self._open_trades.values():
            d = asdict(t)
            d["entry_time"] = t.entry_time.isoformat()
            d["exit_time"] = None
            d["duration_minutes"] = round(
                (utc_now() - t.entry_time).total_seconds() / 60, 1
            )
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_csv_header(self) -> None:
        """Write CSV header row if the file does not exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
                writer.writeheader()

    def _append_csv(self, trade: TradeRecord) -> None:
        """Append a single closed trade to the CSV log.

        Args:
            trade: The closed trade record.
        """
        row = {
            "ticket":           trade.ticket,
            "symbol":           trade.symbol,
            "direction":        trade.direction,
            "entry_price":      trade.entry_price,
            "exit_price":       trade.exit_price,
            "sl":               trade.sl,
            "tp":               trade.tp,
            "lot":              trade.lot,
            "confidence":       round(trade.confidence, 4),
            "profit":           round(trade.profit, 2),
            "exit_reason":      trade.exit_reason,
            "entry_time":       trade.entry_time.isoformat(),
            "exit_time":        trade.exit_time.isoformat() if trade.exit_time else "",
            "duration_minutes": round(trade.duration_minutes, 1),
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            writer.writerow(row)

    def _save_summary(self) -> None:
        """Rewrite the JSON summary file with current statistics."""
        stats = self.get_performance_stats()
        stats["last_updated"] = utc_now().isoformat()
        try:
            with open(self.summary_path, "w") as f:
                json.dump(stats, f, indent=2)
        except OSError as exc:
            logger.warning("Could not write summary JSON: %s", exc)

    def _load_existing_csv(self) -> None:
        """Reload closed trades from the CSV on startup."""
        if not os.path.exists(self.csv_path):
            return
        try:
            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        record = TradeRecord(
                            ticket=int(row["ticket"]),
                            symbol=row["symbol"],
                            direction=row["direction"],
                            entry_price=float(row["entry_price"]),
                            exit_price=float(row["exit_price"]),
                            sl=float(row["sl"]),
                            tp=float(row["tp"]),
                            lot=float(row["lot"]),
                            confidence=float(row["confidence"]),
                            profit=float(row["profit"]),
                            exit_reason=row["exit_reason"],
                            entry_time=datetime.fromisoformat(row["entry_time"]),
                            exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
                        )
                        self._closed_trades.append(record)
                    except Exception:
                        continue
            logger.info("Loaded %d historical trades from CSV.", len(self._closed_trades))
        except OSError as exc:
            logger.warning("Could not load historical trades: %s", exc)

    @staticmethod
    def _calculate_sharpe(profits: List[float], risk_free: float = 0.0) -> float:
        """Compute the Sharpe ratio for a profit series.

        Args:
            profits: List of trade profits.
            risk_free: Risk-free return per trade (default 0).

        Returns:
            float: Sharpe ratio, or 0.0 when undefined.
        """
        n = len(profits)
        if n < 2:
            return 0.0
        mean = sum(profits) / n - risk_free
        variance = sum((p - mean) ** 2 for p in profits) / (n - 1)
        std = math.sqrt(variance)
        return mean / std if std > 0 else 0.0

    @staticmethod
    def _calculate_max_drawdown(profits: List[float]) -> float:
        """Compute peak-to-trough maximum drawdown from the profit series.

        Args:
            profits: Sequential list of trade profits.

        Returns:
            float: Maximum drawdown value (as a positive loss amount).
        """
        peak = 0.0
        max_dd = 0.0
        equity = 0.0
        for p in profits:
            equity += p
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _empty_stats() -> Dict[str, Any]:
        """Return a zero-filled statistics dictionary."""
        return {
            "total_trades": 0, "win_rate": 0.0, "avg_profit": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 0.0,
            "sharpe_ratio": 0.0, "max_drawdown": 0.0, "net_profit": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "open_trades": 0,
        }

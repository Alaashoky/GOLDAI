"""MetaTrader 5 connection manager for XAUUSD trading.

Provides connection lifecycle management, market data retrieval,
and order execution through the MT5 terminal API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import polars as pl

from src.utils import retry, utc_now

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_AVAILABLE = False

logger = logging.getLogger(__name__)

TIMEFRAME_MAP: Dict[str, Any] = {}
if MT5_AVAILABLE:
    TIMEFRAME_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }
else:
    TIMEFRAME_MAP = {k: k for k in ("M1", "M5", "M15", "M30", "H1", "H4", "D1")}

ORDER_TYPE_BUY = "BUY"
ORDER_TYPE_SELL = "SELL"


@dataclass
class TickData:
    """Current bid/ask tick for a symbol."""

    symbol: str
    bid: float
    ask: float
    time: datetime = field(default_factory=utc_now)

    @property
    def spread(self) -> float:
        """Spread in price units."""
        return round(self.ask - self.bid, 5)


@dataclass
class AccountInfo:
    """MT5 account snapshot."""

    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str
    leverage: int
    login: int


@dataclass
class SymbolInfo:
    """Contract specification for a trading symbol."""

    symbol: str
    digits: int
    point: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_stops_level: int


@dataclass
class PositionData:
    """Open position information."""

    ticket: int
    symbol: str
    order_type: str
    lot: float
    entry_price: float
    sl: float
    tp: float
    profit: float
    comment: str
    open_time: datetime


class MT5Connector:
    """Manages the MetaTrader 5 terminal connection and all trading operations.

    Attributes:
        login: MT5 account login number.
        password: MT5 account password.
        server: MT5 broker server name.
        path: Path to the MT5 terminal executable.
        _connected: Current connection state.
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str = r"C:\Program Files\MetaTrader 5\terminal64.exe",
    ) -> None:
        """Initialize the connector with credentials.

        Args:
            login: MT5 account number.
            password: MT5 account password.
            server: Broker server name.
            path: Full path to the MT5 terminal executable.
        """
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @retry(max_retries=3, base_delay=2.0)
    def connect(self) -> bool:
        """Initialize the MT5 terminal and authenticate.

        Returns:
            bool: True when connection is established.

        Raises:
            RuntimeError: If MT5 library is unavailable or connection fails.
        """
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package is not installed on this system.")

        if not mt5.initialize(path=self.path, login=self.login, password=self.password, server=self.server):
            error = mt5.last_error()
            raise RuntimeError(f"MT5 initialize failed: {error}")

        info = mt5.account_info()
        if info is None:
            mt5.shutdown()
            raise RuntimeError("MT5 connected but account_info returned None.")

        self._connected = True
        logger.info("MT5 connected: login=%d server=%s balance=%.2f", info.login, info.server, info.balance)
        return True

    def disconnect(self) -> None:
        """Shut down the MT5 terminal connection."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected.")

    def is_connected(self) -> bool:
        """Check whether the terminal is currently connected.

        Returns:
            bool: Connection status.
        """
        if not MT5_AVAILABLE or not self._connected:
            return False
        return mt5.terminal_info() is not None

    def _ensure_connected(self) -> None:
        """Reconnect if the terminal connection has been lost."""
        if not self.is_connected():
            logger.warning("MT5 not connected — attempting reconnect.")
            self.connect()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @retry(max_retries=3, base_delay=1.0)
    def get_ohlcv(self, symbol: str, timeframe: str, count: int = 500) -> pl.DataFrame:
        """Fetch OHLCV bars from MT5 as a Polars DataFrame.

        Args:
            symbol: Trading symbol, e.g. "XAUUSD".
            timeframe: Timeframe string, e.g. "M5", "H1".
            count: Number of bars to retrieve.

        Returns:
            pl.DataFrame: DataFrame with columns time, open, high, low, close, volume.

        Raises:
            RuntimeError: If MT5 is unavailable or data fetch fails.
        """
        if not MT5_AVAILABLE:
            return _mock_ohlcv(count)

        self._ensure_connected()
        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos returned empty for {symbol} {timeframe}: {mt5.last_error()}")

        df = pl.DataFrame(
            {
                "time":   [datetime.fromtimestamp(r["time"], tz=timezone.utc) for r in rates],
                "open":   [float(r["open"])  for r in rates],
                "high":   [float(r["high"])  for r in rates],
                "low":    [float(r["low"])   for r in rates],
                "close":  [float(r["close"]) for r in rates],
                "volume": [int(r["tick_volume"]) for r in rates],
            }
        )
        logger.debug("Fetched %d bars for %s %s", len(df), symbol, timeframe)
        return df

    @retry(max_retries=3, base_delay=1.0)
    def get_tick(self, symbol: str) -> TickData:
        """Return the current bid/ask for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            TickData: Current tick with bid and ask prices.

        Raises:
            RuntimeError: If tick data cannot be retrieved.
        """
        if not MT5_AVAILABLE:
            return TickData(symbol=symbol, bid=1950.00, ask=1950.20)

        self._ensure_connected()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick failed for {symbol}: {mt5.last_error()}")

        return TickData(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            time=datetime.fromtimestamp(tick.time, tz=timezone.utc),
        )

    @retry(max_retries=3, base_delay=1.0)
    def get_account_info(self) -> AccountInfo:
        """Retrieve current account snapshot.

        Returns:
            AccountInfo: Account balance, equity and margin details.

        Raises:
            RuntimeError: If account info cannot be retrieved.
        """
        if not MT5_AVAILABLE:
            return AccountInfo(
                balance=10000.0, equity=10000.0, margin=0.0,
                free_margin=10000.0, margin_level=0.0,
                currency="USD", leverage=100, login=0,
            )

        self._ensure_connected()
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info failed: {mt5.last_error()}")

        return AccountInfo(
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            margin_level=info.margin_level if info.margin > 0 else 0.0,
            currency=info.currency,
            leverage=info.leverage,
            login=info.login,
        )

    @retry(max_retries=3, base_delay=1.0)
    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Retrieve contract specifications for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            SymbolInfo: Contract details.

        Raises:
            RuntimeError: If symbol info cannot be retrieved.
        """
        if not MT5_AVAILABLE:
            return SymbolInfo(
                symbol=symbol, digits=2, point=0.01,
                contract_size=100.0, volume_min=0.01,
                volume_max=500.0, volume_step=0.01,
                trade_stops_level=0,
            )

        self._ensure_connected()
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info failed for {symbol}: {mt5.last_error()}")

        return SymbolInfo(
            symbol=symbol,
            digits=info.digits,
            point=info.point,
            contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            trade_stops_level=info.trade_stops_level,
        )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @retry(max_retries=2, base_delay=1.0)
    def place_order(
        self,
        symbol: str,
        order_type: str,
        lot: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = "GOLDAI",
    ) -> int:
        """Place a market BUY or SELL order.

        Args:
            symbol: Trading symbol.
            order_type: "BUY" or "SELL".
            lot: Position size in lots.
            price: Expected execution price (slippage tolerance applied).
            sl: Stop-loss price level.
            tp: Take-profit price level.
            comment: Order comment string.

        Returns:
            int: Order ticket number on success.

        Raises:
            RuntimeError: If MT5 is unavailable or order is rejected.
        """
        if not MT5_AVAILABLE:
            ticket = int(time.time() * 1000) % 1_000_000
            logger.info("MOCK place_order %s %s lot=%.2f ticket=%d", order_type, symbol, lot, ticket)
            return ticket

        self._ensure_connected()
        mt5_type = mt5.ORDER_TYPE_BUY if order_type == ORDER_TYPE_BUY else mt5.ORDER_TYPE_SELL

        request: Dict[str, Any] = {
            "action":     mt5.TRADE_ACTION_DEAL,
            "symbol":     symbol,
            "volume":     lot,
            "type":       mt5_type,
            "price":      price,
            "sl":         sl,
            "tp":         tp,
            "deviation":  20,
            "magic":      20240101,
            "comment":    comment,
            "type_time":  mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment_err = result.comment if result else ""
            raise RuntimeError(f"order_send failed retcode={retcode} comment={comment_err}")

        logger.info("Order placed: ticket=%d %s %s lot=%.2f", result.order, order_type, symbol, lot)
        return result.order

    @retry(max_retries=2, base_delay=1.0)
    def modify_order(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify the SL/TP of an open position.

        Args:
            ticket: Position ticket number.
            sl: New stop-loss level.
            tp: New take-profit level.

        Returns:
            bool: True on success.

        Raises:
            RuntimeError: If modification fails.
        """
        if not MT5_AVAILABLE:
            logger.info("MOCK modify_order ticket=%d sl=%.2f tp=%.2f", ticket, sl, tp)
            return True

        self._ensure_connected()
        request: Dict[str, Any] = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       sl,
            "tp":       tp,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            raise RuntimeError(f"modify_order failed retcode={retcode}")

        logger.info("Order modified: ticket=%d sl=%.2f tp=%.2f", ticket, sl, tp)
        return True

    @retry(max_retries=2, base_delay=1.0)
    def close_order(self, ticket: int) -> bool:
        """Close an open position by ticket.

        Args:
            ticket: Position ticket number.

        Returns:
            bool: True on success.

        Raises:
            RuntimeError: If the close operation fails.
        """
        if not MT5_AVAILABLE:
            logger.info("MOCK close_order ticket=%d", ticket)
            return True

        self._ensure_connected()
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning("close_order: position %d not found.", ticket)
            return False

        pos = position[0]
        symbol = pos.symbol
        lot = pos.volume
        price_field = "ask" if pos.type == mt5.ORDER_TYPE_BUY else "bid"
        tick = mt5.symbol_info_tick(symbol)
        close_price = getattr(tick, price_field)
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

        request: Dict[str, Any] = {
            "action":     mt5.TRADE_ACTION_DEAL,
            "position":   ticket,
            "symbol":     symbol,
            "volume":     lot,
            "type":       close_type,
            "price":      close_price,
            "deviation":  20,
            "magic":      20240101,
            "comment":    "GOLDAI_CLOSE",
            "type_time":  mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            raise RuntimeError(f"close_order failed retcode={retcode}")

        logger.info("Position closed: ticket=%d", ticket)
        return True

    def get_positions(self, symbol: Optional[str] = None) -> List[PositionData]:
        """Retrieve open positions, optionally filtered by symbol.

        Args:
            symbol: If provided, filter to this symbol only.

        Returns:
            List[PositionData]: List of open positions.
        """
        if not MT5_AVAILABLE:
            return []

        self._ensure_connected()
        if symbol:
            raw = mt5.positions_get(symbol=symbol)
        else:
            raw = mt5.positions_get()

        if raw is None:
            return []

        positions: List[PositionData] = []
        for p in raw:
            order_type = ORDER_TYPE_BUY if p.type == mt5.ORDER_TYPE_BUY else ORDER_TYPE_SELL
            positions.append(
                PositionData(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    order_type=order_type,
                    lot=p.volume,
                    entry_price=p.price_open,
                    sl=p.sl,
                    tp=p.tp,
                    profit=p.profit,
                    comment=p.comment,
                    open_time=datetime.fromtimestamp(p.time, tz=timezone.utc),
                )
            )
        return positions

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> MT5Connector:
        """Connect when used as context manager."""
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        """Disconnect when exiting context."""
        self.disconnect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_ohlcv(count: int) -> pl.DataFrame:
    """Generate placeholder OHLCV data for systems without MT5."""
    import random
    price = 1950.0
    rows = []
    now = utc_now()
    for i in range(count):
        o = price + random.uniform(-5, 5)
        h = o + random.uniform(0, 10)
        lo = o - random.uniform(0, 10)
        c = (o + h + lo) / 3
        rows.append((now.replace(second=0, microsecond=0), o, h, lo, c, random.randint(100, 1000)))
        price = c
    times, opens, highs, lows, closes, vols = zip(*rows)
    return pl.DataFrame(
        {"time": list(times), "open": list(opens), "high": list(highs),
         "low": list(lows), "close": list(closes), "volume": list(vols)}
    )

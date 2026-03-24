"""Utility functions for the GOLDAI trading bot.

Provides logging setup, retry decorator, timezone utilities,
and price rounding for XAUUSD.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from datetime import datetime, timezone, timedelta
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    name: str = "goldai",
) -> logging.Logger:
    """Set up logging with console and rotating file handlers.

    Args:
        log_dir: Directory for log files.
        log_level: Logging level string.
        name: Logger name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, f"{name}.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger

def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        exceptions: Tuple of exception types to catch.

    Returns:
        Callable: Decorated function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logging.getLogger("goldai").warning(
                            "Retry %d/%d for %s after %.1fs: %s",
                            attempt + 1, max_retries, func.__name__, delay, e,
                        )
                        await asyncio.sleep(delay)
            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logging.getLogger("goldai").warning(
                            "Retry %d/%d for %s after %.1fs: %s",
                            attempt + 1, max_retries, func.__name__, delay, e,
                        )
                        import time
                        time.sleep(delay)
            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator

def utc_now() -> datetime:
    """Get current UTC datetime.

    Returns:
        datetime: Current UTC datetime.
    """
    return datetime.now(timezone.utc)

def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC.

    Args:
        dt: Datetime to convert.

    Returns:
        datetime: UTC datetime.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def round_price(price: float, digits: int = 2) -> float:
    """Round price for XAUUSD (2 decimal places).

    Args:
        price: Price value to round.
        digits: Number of decimal places.

    Returns:
        float: Rounded price.
    """
    return round(price, digits)

def calculate_pip_value(
    lot_size: float,
    point_value: float = 0.01,
    contract_size: float = 100.0,
) -> float:
    """Calculate pip value for gold trading.

    Args:
        lot_size: Position lot size.
        point_value: Point value (0.01 for XAUUSD).
        contract_size: Contract size (100 oz for gold).

    Returns:
        float: Pip value in account currency.
    """
    return lot_size * contract_size * point_value

def points_to_price(points: float, point_size: float = 0.01) -> float:
    """Convert points to price distance.

    Args:
        points: Number of points.
        point_size: Size of one point.

    Returns:
        float: Price distance.
    """
    return points * point_size

def price_to_points(price_distance: float, point_size: float = 0.01) -> float:
    """Convert price distance to points.

    Args:
        price_distance: Price distance.
        point_size: Size of one point.

    Returns:
        float: Number of points.
    """
    return price_distance / point_size if point_size != 0 else 0.0

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency string.

    Args:
        amount: Amount to format.
        currency: Currency code.

    Returns:
        str: Formatted currency string.
    """
    return f"${amount:,.2f} {currency}"

def time_until(target_hour: int, target_minute: int = 0) -> timedelta:
    """Calculate time until a specific UTC hour.

    Args:
        target_hour: Target hour in UTC.
        target_minute: Target minute.

    Returns:
        timedelta: Time until target.
    """
    now = utc_now()
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target - now

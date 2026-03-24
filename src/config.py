"""Trading configuration and capital modes.

This module provides centralized configuration using Pydantic BaseSettings,
supporting multiple capital modes and environment variable loading.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class CapitalMode(str, Enum):
    """Trading capital modes with corresponding risk profiles."""
    MICRO = "MICRO"
    MINI = "MINI"
    STANDARD = "STANDARD"
    PRO = "PRO"


class MarketRegime(str, Enum):
    """Market regime states."""
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"


class TradeDirection(str, Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


CAPITAL_MODE_CONFIG: Dict[CapitalMode, Dict] = {
    CapitalMode.MICRO: {
        "min_balance": 100,
        "max_balance": 500,
        "default_risk": 0.01,
        "max_lot": 0.05,
        "max_positions": 2,
    },
    CapitalMode.MINI: {
        "min_balance": 500,
        "max_balance": 2000,
        "default_risk": 0.02,
        "max_lot": 0.2,
        "max_positions": 3,
    },
    CapitalMode.STANDARD: {
        "min_balance": 2000,
        "max_balance": 10000,
        "default_risk": 0.02,
        "max_lot": 1.0,
        "max_positions": 5,
    },
    CapitalMode.PRO: {
        "min_balance": 10000,
        "max_balance": float("inf"),
        "default_risk": 0.03,
        "max_lot": 5.0,
        "max_positions": 10,
    },
}


SESSION_TIMES: Dict[str, Tuple[int, int]] = {
    "sydney": (22, 7),
    "tokyo": (0, 9),
    "london": (7, 16),
    "new_york": (13, 22),
}


TIMEFRAMES: List[str] = ["M5", "M15", "H1", "H4"]


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""

    # MetaTrader 5
    mt5_login: int = Field(default=0, description="MT5 account login")
    mt5_password: str = Field(default="", description="MT5 account password")
    mt5_server: str = Field(default="", description="MT5 broker server")
    mt5_path: str = Field(
        default=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        description="MT5 terminal path",
    )

    # Telegram
    telegram_bot_token: str = Field(default="", description="Telegram bot token")
    telegram_chat_id: str = Field(default="", description="Telegram chat ID")

    # Trading
    capital_mode: CapitalMode = Field(default=CapitalMode.MINI, description="Capital mode")
    symbol: str = Field(default="XAUUSD", description="Trading symbol")
    max_risk_per_trade: float = Field(default=0.02, ge=0.01, le=0.03, description="Max risk per trade")
    max_daily_drawdown: float = Field(default=0.05, ge=0.01, le=0.10, description="Max daily drawdown")
    max_open_positions: int = Field(default=3, ge=1, le=10, description="Max open positions")
    scan_interval_seconds: int = Field(default=300, description="Scan interval in seconds")
    min_risk_reward: float = Field(default=1.5, description="Minimum risk-reward ratio")

    # Model
    model_path: str = Field(default="models/xgb_model.pkl", description="Model file path")
    confidence_threshold: float = Field(default=0.65, ge=0.5, le=0.95, description="Confidence threshold")
    retrain_interval_hours: int = Field(default=24, ge=1, description="Retrain interval in hours")

    # Database
    db_path: str = Field(default="data/trades.db", description="SQLite database path")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: str = Field(default="logs", description="Log directory")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_capital_config(self) -> Dict:
        """Get configuration for the current capital mode.

        Returns:
            Dict: Capital mode specific configuration.
        """
        return CAPITAL_MODE_CONFIG[self.capital_mode]

    def get_max_lot(self) -> float:
        """Get maximum lot size for current capital mode.

        Returns:
            float: Maximum allowed lot size.
        """
        return self.get_capital_config()["max_lot"]


def get_settings() -> Settings:
    """Create and return a Settings instance.

    Returns:
        Settings: Application settings.
    """
    settings = Settings()
    logger.info("Settings loaded: mode=%s, symbol=%s", settings.capital_mode, settings.symbol)
    return settings
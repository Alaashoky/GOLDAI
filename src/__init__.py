"""GOLDAI - AI-Powered XAUUSD Trading Bot."""

from src.version import VERSION as __version__

__author__ = "GOLDAI Team"

__all__ = [
    # Core
    "config",
    "utils",
    "feature_eng",
    "ml_model",
    "regime_detector",
    "risk_engine",
    "smc_polars",
    # Connectors & Infrastructure
    "mt5_connector",
    "trade_logger",
    "telegram_notifier",
    "telegram_commands",
    "telegram_notifications",
    "news_agent",
    # Risk & Position Management
    "smart_risk_manager",
    "position_manager",
    "session_filter",
    "dynamic_confidence",
    # Advanced AI Modules
    "fuzzy_exit_logic",
    "kalman_filter",
    "kelly_position_scaler",
    "m5_confirmation",
    "macro_connector",
    "momentum_persistence",
    "recovery_detector",
    "risk_metrics",
    "trajectory_predictor",
    "profit_momentum_tracker",
    # Configuration & Training
    "filter_config",
    "auto_trainer",
    "version",
]

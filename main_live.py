"""GOLDAI - Live Trading Bot Main Entry Point.

Orchestrates all modules for live XAUUSD trading on MetaTrader 5.
"""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging setup (before any other imports so modules see the root config)
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """Configure rotating file + stream logging.

    Args:
        log_dir: Directory to write log files.
        log_level: Python logging level string.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "goldai.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


_setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module imports (all wrapped in try/except for portability)
# ---------------------------------------------------------------------------
try:
    from src.config import Settings, get_settings
    from src.feature_eng import compute_features, create_labels
    from src.ml_model import TradingModel, PredictionResult
    from src.smc_polars import SMCAnalyzer, SMCSignal
    from src.regime_detector import RegimeDetector
    from src.mt5_connector import MT5Connector
    from src.smart_risk_manager import SmartRiskManager
    from src.session_filter import SessionFilter
    from src.position_manager import PositionManager
    from src.dynamic_confidence import DynamicConfidence
    from src.auto_trainer import AutoTrainer
    from src.news_agent import NewsAgent
    from src.telegram_notifier import TelegramNotifier
    from src.trade_logger import TradeLogger
    from src.kalman_filter import KalmanFilter
    from src.kelly_position_scaler import KellyPositionScaler
    from src.m5_confirmation import M5Confirmation
    from src.macro_connector import MacroConnector
    from src.momentum_persistence import MomentumPersistence
    from src.recovery_detector import RecoveryDetector
    from src.fuzzy_exit_logic import FuzzyExitLogic
    from src.trajectory_predictor import TrajectoryPredictor
    from src.profit_momentum_tracker import ProfitMomentumTracker
    from src.telegram_commands import TelegramCommands
    from src.risk_metrics import RiskMetrics
    IMPORTS_OK = True
except ImportError as _ie:
    logger.warning("Some imports failed: %s — running in degraded mode.", _ie)
    IMPORTS_OK = False

# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

import json

def _write_status(status: dict, path: str = "data/status.json") -> None:
    """Persist bot status to JSON.

    Args:
        status: Status dict to write.
        path: File path for the JSON status file.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(status, fh, indent=2, default=str)
    except OSError as exc:
        logger.warning("Could not write status: %s", exc)


# ---------------------------------------------------------------------------
# GoldAIBot
# ---------------------------------------------------------------------------

class GoldAIBot:
    """Top-level orchestrator for the GOLDAI live trading bot.

    Initialises every subsystem and provides the main async event loop
    that scans markets, filters signals, and manages positions.

    Attributes:
        settings: Application configuration loaded from .env.
        _running: Whether the main loop should continue.
        _shutdown_event: asyncio.Event signalled when shutdown is requested.
    """

    def __init__(self) -> None:
        """Initialise all subsystems using settings from .env."""
        self.settings: Settings = get_settings()
        _setup_logging(self.settings.log_dir, self.settings.log_level)

        logger.info("Initialising GOLDAI bot (version: %s)", _bot_version())

        # Core market-data / order execution
        self.connector = MT5Connector(
            login=self.settings.mt5_login,
            password=self.settings.mt5_password,
            server=self.settings.mt5_server,
            path=self.settings.mt5_path,
        )

        # Risk & sizing
        self.risk_manager = SmartRiskManager(
            max_risk_per_trade=self.settings.max_risk_per_trade,
            max_daily_drawdown=self.settings.max_daily_drawdown,
        )
        self.kelly_scaler = KellyPositionScaler(
            max_risk_pct=self.settings.max_risk_per_trade,
        )
        self.risk_metrics = RiskMetrics()
        self.recovery_detector = RecoveryDetector()
        self.profit_momentum = ProfitMomentumTracker()

        # Signal & regime
        self.smc_analyzer = SMCAnalyzer()
        self.regime_detector = RegimeDetector()
        self.kalman = KalmanFilter()
        self.momentum = MomentumPersistence()
        self.trajectory = TrajectoryPredictor()
        self.fuzzy_exit = FuzzyExitLogic()
        self.macro = MacroConnector()

        # Filters
        self.session_filter = SessionFilter()
        self.news_agent = NewsAgent()
        self.m5_confirm = M5Confirmation()

        # ML model
        self.model = TradingModel(model_path=self.settings.model_path)
        self._load_model_if_exists()

        # Position management
        self.position_manager = PositionManager(connector=self.connector)

        # Confidence scoring
        self.confidence_engine = DynamicConfidence(
            base_threshold=self.settings.confidence_threshold,
        )

        # Auto-trainer
        self.auto_trainer = AutoTrainer(
            model_path=self.settings.model_path,
            retrain_interval_hours=self.settings.retrain_interval_hours,
        )

        # Notifications & logging
        self.notifier = TelegramNotifier(
            token=self.settings.telegram_bot_token,
            chat_id=self.settings.telegram_chat_id,
        )
        self.trade_logger = TradeLogger(log_dir="data/trades")
        self.telegram_commands = TelegramCommands()

        # Runtime state
        self._running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._backoff_seconds: float = 5.0
        self._daily_start_equity: float = 0.0
        self._weekly_start_equity: float = 0.0

        logger.info("All subsystems initialised.")

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def _load_model_if_exists(self) -> None:
        """Load pre-trained model from disk if the file exists."""
        if os.path.exists(self.settings.model_path):
            try:
                self.model.load(self.settings.model_path)
                logger.info("Model loaded from %s", self.settings.model_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load model: %s", exc)
        else:
            logger.warning("Model file not found at %s — running without ML signal.", self.settings.model_path)

    def _register_signals(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown_signal)
            except NotImplementedError:
                # Windows does not support add_signal_handler
                signal.signal(sig, lambda *_: self._handle_shutdown_signal())

    def _handle_shutdown_signal(self) -> None:
        """Mark the bot for graceful shutdown."""
        logger.info("Shutdown signal received — stopping after current iteration.")
        self._running = False
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Market analysis
    # ------------------------------------------------------------------

    def analyze_market(self) -> tuple[PredictionResult, SMCSignal]:
        """Fetch H1 OHLCV data, run features, ML predict, and SMC analysis.

        Returns:
            Tuple of (PredictionResult, SMCSignal) for the current bar.
        """
        symbol = self.settings.symbol
        h1_df = self.connector.get_ohlcv(symbol, "H1", count=200)
        if h1_df is None or h1_df.is_empty():
            logger.warning("No H1 data returned for %s", symbol)
            default_pred = PredictionResult(direction="HOLD", confidence=0.0)
            default_smc = SMCSignal()
            return default_pred, default_smc

        # Feature engineering
        features_df = compute_features(h1_df)
        feature_cols = [c for c in features_df.columns if c not in ("time", "open", "high", "low", "close", "tick_volume")]
        if features_df.is_empty() or not feature_cols:
            logger.warning("Feature engineering produced no features.")
            return PredictionResult(), SMCSignal()

        # Kalman filter on close prices
        closes = h1_df["close"].to_list()
        self.kalman.get_filtered_series(closes)

        # ML prediction on last row
        X = features_df[feature_cols].tail(1).to_numpy()
        prediction = self.model.predict(X)[0] if hasattr(self.model, "predict") and self.model._model is not None else PredictionResult()

        # SMC analysis
        smc_signal = self.smc_analyzer.analyze(h1_df)

        return prediction, smc_signal

    # ------------------------------------------------------------------
    # Trade filters
    # ------------------------------------------------------------------

    def should_trade(self, confidence: float, prediction: PredictionResult, smc: SMCSignal) -> bool:
        """Check all pre-trade filters.

        Args:
            confidence: Combined confidence score.
            prediction: ML model prediction result.
            smc: SMC signal from chart analysis.

        Returns:
            True if all filters pass and a trade may be placed.
        """
        if TelegramCommands.is_paused():
            logger.info("Trading paused via Telegram command.")
            return False

        if not self.session_filter.is_trading_allowed():
            logger.debug("Outside allowed session.")
            return False

        if self.news_agent.is_high_impact_event():
            logger.info("High-impact news blackout active — skipping trade.")
            return False

        account = self.connector.get_account_info()
        if account is None:
            return False

        if not self.risk_manager.check_daily_drawdown(account.equity, self._daily_start_equity):
            logger.warning("Daily drawdown limit hit — no new trades.")
            return False

        if not self.risk_manager.circuit_breaker_check(account.equity, account.balance):
            logger.warning("Circuit breaker active.")
            return False

        if len(self.connector.get_positions(self.settings.symbol)) >= self.settings.max_open_positions:
            logger.debug("Max open positions reached.")
            return False

        if confidence < self.confidence_engine.get_current_threshold():
            logger.debug("Confidence %.2f below threshold %.2f.", confidence, self.confidence_engine.get_current_threshold())
            return False

        if prediction.direction == "HOLD":
            return False

        return True

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_trade(self, direction: str, confidence: float, smc: SMCSignal) -> bool:
        """Place a trade order with all confirmations and sizing logic.

        Args:
            direction: "BUY" or "SELL".
            confidence: Combined confidence score.
            smc: SMC signal used to derive SL/TP levels.

        Returns:
            True if the order was placed successfully.
        """
        symbol = self.settings.symbol
        account = self.connector.get_account_info()
        if account is None:
            return False

        tick = self.connector.get_tick(symbol)
        if tick is None:
            return False

        sym_info = self.connector.get_symbol_info(symbol)
        if sym_info is None:
            return False

        # Fetch M5 data for confirmation
        m5_df = self.connector.get_ohlcv(symbol, "M5", count=50)
        if m5_df is not None and not m5_df.is_empty():
            confirm = self.m5_confirm.confirm_entry(m5_df, direction)
            if not confirm.confirmed:
                logger.info("M5 confirmation rejected entry (score=%.2f).", confirm.score)
                return False

        # Macro filter
        macro_data = self.macro.get_last_data()
        if macro_data is not None:
            macro_bias = self.macro.calculate_gold_macro_bias(macro_data)
            if direction == "BUY" and macro_bias < -0.3:
                logger.info("Macro bias bearish (%.2f) — skip BUY.", macro_bias)
                return False
            if direction == "SELL" and macro_bias > 0.3:
                logger.info("Macro bias bullish (%.2f) — skip SELL.", macro_bias)
                return False

        # Lot sizing via Kelly
        recovery_mult = self.recovery_detector.get_recommended_risk_multiplier()
        momentum_mult = self.profit_momentum.get_risk_multiplier()
        base_risk = self.settings.max_risk_per_trade * recovery_mult * momentum_mult

        lot = self.kelly_scaler.calculate_lot_size(
            account_balance=account.balance,
            risk_pct=base_risk,
            sl_pips=200.0,  # Will be refined per symbol_info
        )
        lot = min(lot, self.settings.get_max_lot())
        lot = max(lot, sym_info.volume_min)

        # SL/TP derivation (ATR-based fallback)
        atr_pips = 150.0
        if sym_info.point > 0:
            sl_distance = atr_pips * sym_info.point * 10
        else:
            sl_distance = 1.5

        price = tick.ask if direction == "BUY" else tick.bid
        sl = (price - sl_distance) if direction == "BUY" else (price + sl_distance)
        tp = (price + sl_distance * self.settings.min_risk_reward) if direction == "BUY" else (price - sl_distance * self.settings.min_risk_reward)

        result = self.connector.place_order(
            symbol=symbol,
            order_type=direction,
            volume=round(lot, 2),
            price=price,
            sl=round(sl, 2),
            tp=round(tp, 2),
            comment=f"GOLDAI c={confidence:.2f}",
        )

        if result is not None and result.retcode == 10009:
            logger.info("Order placed: ticket=%s dir=%s lot=%.2f", result.order, direction, lot)
            self.trade_logger.log_entry(
                ticket=result.order,
                symbol=symbol,
                direction=direction,
                entry_price=price,
                lot=lot,
                sl=sl,
                tp=tp,
                confidence=confidence,
            )
            self.notifier.send_sync(
                f"✅ Trade Opened\n{direction} {symbol}\nLot: {lot:.2f} | Confidence: {confidence:.0%}\nSL: {sl:.2f} | TP: {tp:.2f}"
            )
            return True

        logger.warning("Order failed: %s", result)
        return False

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def manage_positions(self) -> None:
        """Run position management loop — trailing stop, BE, partial close."""
        positions = self.connector.get_positions(self.settings.symbol)
        if not positions:
            return

        h1_df = self.connector.get_ohlcv(self.settings.symbol, "H1", count=50)
        account = self.connector.get_account_info()

        for pos in positions:
            self.position_manager.update_trailing_stop(pos, h1_df)
            self.position_manager.move_to_breakeven(pos)
            close_result = self.position_manager.check_partial_close(pos)
            if close_result:
                logger.info("Partial close executed for ticket %s", pos.ticket)

            # Fuzzy exit
            if h1_df is not None and not h1_df.is_empty():
                momentum_result = self.momentum.get_momentum_score(h1_df)
                open_time = getattr(pos, "time", None)
                hours_open = 0.0
                if open_time:
                    hours_open = (datetime.now(timezone.utc) - open_time).total_seconds() / 3600
                exit_score = self.fuzzy_exit.calculate_exit_score(
                    profit_pips=getattr(pos, "profit", 0.0),
                    hours_open=hours_open,
                    momentum_score=momentum_result.score if hasattr(momentum_result, "score") else 0.0,
                    volatility=getattr(h1_df, "height", 50) / 50.0,
                    smc_signal=0.0,
                )
                if self.fuzzy_exit.should_exit(exit_score):
                    closed = self.connector.close_order(pos.ticket)
                    if closed:
                        logger.info("Fuzzy exit triggered for ticket %s (score=%.2f)", pos.ticket, exit_score.score)
                        profit = getattr(pos, "profit", 0.0)
                        self.trade_logger.log_exit(ticket=pos.ticket, exit_price=0.0, profit=profit)
                        self.recovery_detector.update_loss_streak(profit)
                        self.profit_momentum.update(profit)
                        self.kelly_scaler.update_stats(profit)
                        self.confidence_engine.record_outcome(0.7, profit > 0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Async main loop — entry point for the bot.

        Connects to MT5, starts Telegram notifier, then runs the main
        scan loop until a shutdown signal is received.
        """
        self._running = True
        self._register_signals()

        logger.info("Connecting to MetaTrader 5 …")
        if not self.connector.connect():
            logger.error("MT5 connection failed — aborting.")
            return

        account = self.connector.get_account_info()
        if account:
            self._daily_start_equity = account.equity
            self._weekly_start_equity = account.equity
            logger.info("Account: #%s | Balance: %.2f | Equity: %.2f", account.login, account.balance, account.equity)

        await self.notifier.send("🚀 GOLDAI Bot started.")

        consecutive_errors = 0

        while self._running:
            try:
                loop_start = time.monotonic()

                # Persist bot status
                _write_status({
                    "last_run": datetime.now(timezone.utc).isoformat(),
                    "running": True,
                    "positions": len(self.connector.get_positions(self.settings.symbol)),
                })

                # 1. Manage open positions
                self.manage_positions()

                # 2. Session / news pre-check
                if not self.session_filter.is_trading_allowed():
                    logger.debug("Outside trading hours — sleeping.")
                    await self._sleep(self.settings.scan_interval_seconds)
                    continue

                if self.news_agent.is_high_impact_event():
                    mins = self.news_agent.minutes_to_next_event() or 30
                    logger.info("News blackout — waiting %.0f min.", mins)
                    await self._sleep(min(int(mins * 60), self.settings.scan_interval_seconds))
                    continue

                # 3. Market analysis
                prediction, smc_signal = self.analyze_market()

                # 4. Confidence scoring
                h1_df = self.connector.get_ohlcv(self.settings.symbol, "H1", count=50)
                momentum_result = self.momentum.get_momentum_score(h1_df) if (h1_df is not None and not h1_df.is_empty()) else None
                signals_dict = {
                    "ml_confidence": prediction.confidence,
                    "smc_confidence": smc_signal.confidence,
                    "direction_match": 1.0 if (prediction.direction != "HOLD" and
                        ((prediction.direction == "BUY" and smc_signal.direction.value == "BULLISH") or
                         (prediction.direction == "SELL" and smc_signal.direction.value == "BEARISH"))) else 0.0,
                    "momentum": getattr(momentum_result, "score", 0.5) if momentum_result else 0.5,
                }
                confidence = self.confidence_engine.get_trade_confidence(signals_dict)

                # 5. Trade decision
                if self.should_trade(confidence, prediction, smc_signal):
                    direction = prediction.direction
                    logger.info("Signal: %s | Confidence: %.2f", direction, confidence)
                    self.execute_trade(direction, confidence, smc_signal)

                # 6. Auto-trainer check
                if self.auto_trainer.should_retrain():
                    logger.info("Auto-trainer: retraining model …")
                    data = self.auto_trainer.get_training_data(self.connector)
                    if data is not None:
                        self.auto_trainer.retrain(self.model, data)

                consecutive_errors = 0
                self._backoff_seconds = 5.0

                elapsed = time.monotonic() - loop_start
                sleep_time = max(0, self.settings.scan_interval_seconds - elapsed)
                await self._sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                consecutive_errors += 1
                logger.exception("Unhandled error in main loop: %s", exc)
                self._backoff_seconds = min(self._backoff_seconds * 2, 300)
                logger.info("Backoff: sleeping %.0f s (errors=%d)", self._backoff_seconds, consecutive_errors)
                await self._sleep(self._backoff_seconds)

        await self._shutdown()

    async def _sleep(self, seconds: float) -> None:
        """Sleep for `seconds`, waking early on shutdown event.

        Args:
            seconds: How long to sleep.
        """
        try:
            await asyncio.wait_for(asyncio.shield(self._shutdown_event.wait()), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _shutdown(self) -> None:
        """Graceful shutdown: disconnect MT5 and notify Telegram."""
        logger.info("Shutting down GOLDAI bot …")
        _write_status({"last_run": datetime.now(timezone.utc).isoformat(), "running": False})
        try:
            await self.notifier.send("🛑 GOLDAI Bot stopped.")
        except Exception:  # noqa: BLE001
            pass
        self.connector.disconnect()
        logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Version helper
# ---------------------------------------------------------------------------

def _bot_version() -> str:
    """Return the bot version string.

    Returns:
        Version string from src.version if available, else 'unknown'.
    """
    try:
        from src.version import __version__
        return __version__
    except ImportError:
        return "unknown"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Configure and run the async event loop."""
    for directory in ("data", "models", "logs"):
        Path(directory).mkdir(parents=True, exist_ok=True)

    if not IMPORTS_OK:
        logger.error("Critical imports failed — cannot start bot.")
        sys.exit(1)

    bot = GoldAIBot()

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — exiting.")


if __name__ == "__main__":
    main()

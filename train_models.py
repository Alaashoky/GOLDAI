#!/usr/bin/env python3
"""
GOLDAI Model Training Script — Date-Based Clean Split
=====================================================
Trains models on a fixed historical period with zero overlap
with the backtesting period.

Training period  : 2020-01-01 → 2023-12-31  (4 years)
Validation period: 2024-01-01 → 2024-06-30  (6 months — for early stopping)

Out-of-sample backtest (run separately via backtests/backtest_v4.py):
Test period      : 2024-07-01 → present      (completely held-out)

Usage:
    python train_models.py [--symbol XAUUSD.m] [--timeframe M15]

Output:
    - models/hmm_regime.pkl
    - models/xgboost_model.pkl
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, date
import polars as pl
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)
logger.add(
    "logs/training_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    rotation="1 day",
    level="DEBUG",
)

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config import get_config
from src.mt5_connector import MT5Connector
from src.smc_polars import SMCAnalyzer
from src.feature_eng import FeatureEngineer
from src.regime_detector import MarketRegimeDetector
from src.ml_model import TradingModel, get_default_feature_columns

# ---------------------------------------------------------------------------
# Fixed date boundaries (change only here if you need a different split)
# ---------------------------------------------------------------------------
TRAIN_START = datetime(2020, 1, 1)
TRAIN_END   = datetime(2023, 12, 31, 23, 59, 59)
VAL_START   = datetime(2024, 1, 1)
VAL_END     = datetime(2024, 6, 30, 23, 59, 59)

# M15 bars per year ≈ 52 weeks × 5 days × 24 h × 4 bars = 24,960
# Fetch enough bars to cover TRAIN_START → VAL_END with a large safety margin.
# From 2020-01-01 to 2024-06-30 ≈ 4.5 years ≈ 112,000 M15 bars.
# We request 200,000 to guarantee full coverage even if the broker
# starts slightly later than 2020-01-01.
FETCH_BARS_M15 = 200_000


def fetch_and_filter(
    connector: MT5Connector,
    symbol: str,
    timeframe: str,
    bars: int,
    date_from: datetime,
    date_to: datetime,
) -> pl.DataFrame:
    """
    Fetch `bars` recent bars from MT5 and keep only those whose
    timestamp falls within [date_from, date_to].
    """
    logger.info(f"Fetching {bars:,} {timeframe} bars for {symbol} ...")
    df = connector.get_market_data(symbol, timeframe, bars)

    if df is None or len(df) == 0:
        raise ValueError("No data received from MT5")

    logger.info(f"Received {len(df):,} bars  ({df['time'].min()} → {df['time'].max()})")

    # Filter to the requested date window
    df = df.filter(
        (pl.col("time") >= date_from) & (pl.col("time") <= date_to)
    )

    if len(df) == 0:
        raise ValueError(
            f"No bars found in window {date_from.date()} → {date_to.date()}. "
            "Try increasing FETCH_BARS_M15 or check broker data availability."
        )

    logger.info(
        f"After date filter [{date_from.date()} → {date_to.date()}]: "
        f"{len(df):,} bars"
    )
    return df


def prepare_features(df: pl.DataFrame) -> pl.DataFrame:
    """Apply feature engineering and SMC indicators, then create the target."""
    logger.info("Applying feature engineering + SMC indicators ...")
    fe = FeatureEngineer()
    df = fe.calculate_all(df, include_ml_features=True)
    smc = SMCAnalyzer(swing_length=5)
    df = smc.calculate_all(df)
    df = fe.create_target(df, lookahead=1)
    logger.info(f"Feature columns created: {len(df.columns)}")
    return df


def train_hmm_model(
    df_train: pl.DataFrame,
    model_path: str = "models/hmm_regime.pkl",
) -> MarketRegimeDetector:
    """Train HMM regime model on training data only."""
    logger.info("=" * 60)
    logger.info("Training HMM Regime Model")
    logger.info(f"  Data  : {df_train['time'].min()} → {df_train['time'].max()}")
    logger.info(f"  Bars  : {len(df_train):,}")
    logger.info("=" * 60)

    detector = MarketRegimeDetector(
        n_regimes=3,
        lookback_periods=500,
        model_path=model_path,
    )
    detector.fit(df_train)

    if detector.fitted:
        df_pred = detector.predict(df_train)
        regime_counts = df_pred.group_by("regime_name").len()
        logger.info("Regime Distribution:")
        for row in regime_counts.iter_rows(named=True):
            if row["regime_name"]:
                logger.info(f"  {row['regime_name']}: {row['len']} bars")

        logger.info("Transition Matrix:")
        transmat = detector.get_transition_matrix()
        for i, regime in detector.regime_mapping.items():
            probs = [f"{p:.2f}" for p in transmat[i]]
            logger.info(f"  {regime.value}: {probs}")

    return detector


def train_xgboost_model(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    model_path: str = "models/xgboost_model.pkl",
    confidence_threshold: float = 0.60,
) -> TradingModel:
    """
    Train XGBoost on training data, using the validation set for early
    stopping via the train_ratio parameter in TradingModel.fit().

    Strategy: concatenate train + val, then set train_ratio so that
    fit()'s internal split lands exactly at the train/val boundary.
    This lets XGBoost's early stopping be driven by the held-out
    validation period (2024-01 → 2024-06).
    """
    logger.info("=" * 60)
    logger.info("Training XGBoost Model")
    logger.info(f"  Train : {df_train['time'].min()} → {df_train['time'].max()}  ({len(df_train):,} bars)")
    logger.info(f"  Val   : {df_val['time'].min()} → {df_val['time'].max()}  ({len(df_val):,} bars)")
    logger.info("=" * 60)

    default_features = get_default_feature_columns()

    # Combine train + val in chronological order
    df_combined = pl.concat([df_train, df_val])

    available_features = [f for f in default_features if f in df_combined.columns]
    logger.info(f"Available features: {len(available_features)}/{len(default_features)}")

    # Calculate train_ratio so that fit()'s split lands at the train/val boundary.
    # drop_nulls() inside fit() may shrink the row count slightly; using the
    # ratio of bar counts gives a good approximation.
    n_train = len(df_train)
    n_total = len(df_combined)
    train_ratio = n_train / n_total
    logger.info(f"train_ratio = {n_train}/{n_total} = {train_ratio:.4f}")

    model = TradingModel(
        confidence_threshold=confidence_threshold,
        model_path=model_path,
    )

    model.fit(
        df_combined,
        available_features,
        target_col="target",
        train_ratio=train_ratio,
        num_boost_round=200,
        early_stopping_rounds=15,
    )

    if model.fitted:
        logger.info("Top 10 Feature Importance:")
        for feat, imp in model.get_feature_importance(10).items():
            logger.info(f"  {feat}: {imp:.4f}")

        # Walk-forward validation within the training window only
        total_bars = len(df_train)
        train_window = max(500, total_bars // 20)
        test_window  = max(50,  total_bars // 200)
        step         = max(50,  total_bars // 200)

        logger.info("Running walk-forward validation (training split only) ...")
        available_train = [f for f in default_features if f in df_train.columns]
        results = model.walk_forward_train(
            df_train,
            available_train,
            "target",
            train_window=train_window,
            test_window=test_window,
            step=step,
        )

        if results:
            avg_train = np.mean([r[0] for r in results])
            avg_test  = np.mean([r[1] for r in results])
            logger.info("Walk-forward Results:")
            logger.info(f"  Avg Train AUC: {avg_train:.4f}")
            logger.info(f"  Avg Test AUC:  {avg_test:.4f}")
            if avg_test > 0:
                logger.info(f"  Overfitting ratio: {avg_train / avg_test:.2f}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GOLDAI Model Training — date-based clean split"
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Trading symbol (default: from config)"
    )
    parser.add_argument(
        "--timeframe", type=str, default=None,
        help="Timeframe (default: from config, e.g. M15)"
    )
    parser.add_argument(
        "--train-start", type=str, default="2020-01-01",
        help="Training start date YYYY-MM-DD (default: 2020-01-01)"
    )
    parser.add_argument(
        "--train-end", type=str, default="2023-12-31",
        help="Training end date YYYY-MM-DD (default: 2023-12-31)"
    )
    parser.add_argument(
        "--val-start", type=str, default="2024-01-01",
        help="Validation start date YYYY-MM-DD (default: 2024-01-01)"
    )
    parser.add_argument(
        "--val-end", type=str, default="2024-06-30",
        help="Validation end date YYYY-MM-DD (default: 2024-06-30)"
    )
    args = parser.parse_args()

    # Parse dates
    train_start = datetime.strptime(args.train_start, "%Y-%m-%d")
    train_end   = datetime.strptime(args.train_end,   "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    val_start   = datetime.strptime(args.val_start,   "%Y-%m-%d")
    val_end     = datetime.strptime(args.val_end,     "%Y-%m-%d").replace(hour=23, minute=59, second=59)

    logger.info("=" * 60)
    logger.info("GOLDAI — MODEL TRAINING (DATE-BASED CLEAN SPLIT)")
    logger.info("=" * 60)
    logger.info(f"  Training period  : {train_start.date()} → {train_end.date()}")
    logger.info(f"  Validation period: {val_start.date()} → {val_end.date()}")
    logger.info(f"  Out-of-sample test (backtest_v4.py): after {val_end.date()}")
    logger.info("=" * 60)

    # Load config
    config = get_config()
    symbol    = args.symbol    or config.symbol
    timeframe = args.timeframe or config.execution_timeframe

    logger.info(f"Symbol   : {symbol}")
    logger.info(f"Timeframe: {timeframe}")

    # Connect to MT5
    logger.info("Connecting to MT5 ...")
    connector = MT5Connector(
        login=config.mt5_login,
        password=config.mt5_password,
        server=config.mt5_server,
        path=config.mt5_path,
    )

    try:
        connector.connect()
        logger.info("MT5 connected successfully")
        logger.info(f"Account balance: ${connector.account_balance:,.2f}")
    except Exception as e:
        logger.error(f"MT5 connection failed: {e}")
        logger.info("Ensure MT5 terminal is running with auto-trading enabled.")
        return

    try:
        # ------------------------------------------------------------------
        # Fetch data covering training + validation window
        # ------------------------------------------------------------------
        df_full = fetch_and_filter(
            connector, symbol, timeframe,
            bars=FETCH_BARS_M15,
            date_from=train_start,
            date_to=val_end,
        )

        # ------------------------------------------------------------------
        # Date-based split
        # ------------------------------------------------------------------
        df_train_raw = df_full.filter(
            (pl.col("time") >= train_start) & (pl.col("time") <= train_end)
        )
        df_val_raw = df_full.filter(
            (pl.col("time") >= val_start) & (pl.col("time") <= val_end)
        )

        if len(df_train_raw) < 500:
            raise ValueError(
                f"Training window too small ({len(df_train_raw)} bars). "
                "Check dates or increase FETCH_BARS_M15."
            )
        if len(df_val_raw) < 100:
            logger.warning(
                f"Validation window small ({len(df_val_raw)} bars). "
                "Results may be less reliable."
            )

        logger.info(
            f"Date-split → Train: {len(df_train_raw):,} bars | "
            f"Val: {len(df_val_raw):,} bars"
        )

        # ------------------------------------------------------------------
        # Feature engineering on training data (HMM + XGBoost train split)
        # ------------------------------------------------------------------
        logger.info("Preparing features for TRAINING window ...")
        df_train = prepare_features(df_train_raw)

        # ------------------------------------------------------------------
        # Feature engineering on validation data
        # ------------------------------------------------------------------
        logger.info("Preparing features for VALIDATION window ...")
        df_val = prepare_features(df_val_raw)

        # ------------------------------------------------------------------
        # Save combined training+validation data for reference
        # ------------------------------------------------------------------
        combined_save = pl.concat([df_train, df_val])
        save_path = "data/training_data.parquet"
        combined_save.write_parquet(save_path)
        logger.info(f"Training data saved to {save_path}")

        # ------------------------------------------------------------------
        # Train HMM on training data ONLY
        # ------------------------------------------------------------------
        hmm_model = train_hmm_model(df_train)

        # Add regime predictions to training data (for XGBoost features)
        if hmm_model.fitted:
            df_train = hmm_model.predict(df_train)
            df_val   = hmm_model.predict(df_val)

        # ------------------------------------------------------------------
        # Train XGBoost (train → val split via train_ratio)
        # ------------------------------------------------------------------
        xgb_model = train_xgboost_model(df_train, df_val)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"HMM Model    : {'SAVED' if hmm_model.fitted else 'FAILED'}")
        logger.info(f"XGBoost Model: {'SAVED' if xgb_model.fitted else 'FAILED'}")
        logger.info(f"Models saved : models/")
        logger.info(f"Training data: {train_start.date()} → {train_end.date()}  ({len(df_train):,} bars)")
        logger.info(f"Validation   : {val_start.date()} → {val_end.date()}  ({len(df_val):,} bars)")
        logger.info(f"  ⚠  These dates are NEVER shown to the backtester:")
        logger.info(f"     Test period starts at {val_end.date()} + 1 day")
        logger.info("=" * 60)
        logger.info(
            "Next step: python backtests/backtest_v4.py "
            "--balance 439 --lot 0.01"
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        connector.disconnect()
        logger.info("MT5 disconnected")


if __name__ == "__main__":
    main()

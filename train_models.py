"""
Model Training Script
=====================
Fetches historical data from MT5 and trains all models.

Usage:
    python train_models.py [--bars 140000] [--symbol XAUUSD] [--timeframe M15]

Output:
    - models/xgboost_model.pkl
    - models/hmm_regime.pkl
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import polars as pl
import numpy as np
from loguru import logger

# Configure logging
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

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Import modules
from src.config import TradingConfig, get_config
from src.mt5_connector import MT5Connector
from src.smc_polars import SMCAnalyzer
from src.feature_eng import FeatureEngineer
from src.regime_detector import MarketRegimeDetector
from src.ml_model import TradingModel, get_default_feature_columns


def fetch_training_data(
    connector: MT5Connector,
    symbol: str,
    timeframe: str,
    bars: int = 5000,
) -> pl.DataFrame:
    """Fetch historical data for training."""
    logger.info(f"Fetching {bars} bars of {symbol} {timeframe} data...")
    
    df = connector.get_market_data(symbol, timeframe, bars)
    
    if len(df) == 0:
        raise ValueError("No data received from MT5")
    
    logger.info(f"Received {len(df)} bars")
    logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
    
    return df


def prepare_features(df: pl.DataFrame) -> pl.DataFrame:
    """Apply all feature engineering."""
    logger.info("Applying feature engineering...")
    
    # Technical indicators
    fe = FeatureEngineer()
    df = fe.calculate_all(df, include_ml_features=True)
    
    # SMC indicators
    smc = SMCAnalyzer(swing_length=5)
    df = smc.calculate_all(df)
    
    # Create target variable
    df = fe.create_target(df, lookahead=1)
    
    logger.info(f"Total features created: {len(df.columns)}")
    
    return df


def train_hmm_model(
    df: pl.DataFrame,
    model_path: str = "models/hmm_regime.pkl",
) -> MarketRegimeDetector:
    """Train HMM regime detection model."""
    logger.info("=" * 60)
    logger.info("Training HMM Regime Model")
    logger.info("=" * 60)
    
    detector = MarketRegimeDetector(
        n_regimes=3,
        lookback_periods=500,
        model_path=model_path,
    )
    
    detector.fit(df)
    
    if detector.fitted:
        # Add regime predictions to df
        df_with_regime = detector.predict(df)
        
        # Show regime distribution
        regime_counts = df_with_regime.group_by("regime_name").len()
        logger.info("Regime Distribution:")
        for row in regime_counts.iter_rows(named=True):
            if row["regime_name"]:
                logger.info(f"  {row['regime_name']}: {row['len']} bars")
        
        # Show transition matrix
        logger.info("Transition Matrix:")
        transmat = detector.get_transition_matrix()
        for i, regime in detector.regime_mapping.items():
            probs = [f"{p:.2f}" for p in transmat[i]]
            logger.info(f"  {regime.value}: {probs}")
    
    return detector


def train_xgboost_model(
    df: pl.DataFrame,
    model_path: str = "models/xgboost_model.pkl",
) -> TradingModel:
    """Train XGBoost prediction model with anti-overfitting measures."""
    logger.info("=" * 60)
    logger.info("Training XGBoost Model (Anti-Overfit Config)")
    logger.info("=" * 60)

    # Get feature columns that exist in df
    default_features = get_default_feature_columns()
    available_features = [f for f in default_features if f in df.columns]

    logger.info(f"Available features: {len(available_features)}/{len(default_features)}")

    # Create model with anti-overfitting parameters
    model = TradingModel(
        confidence_threshold=0.60,  # Lowered from 0.65 for more signals
        model_path=model_path,
    )

    total_bars = len(df)

    # Scale walk-forward windows proportionally to data size
    train_window = max(500, total_bars // 20)
    test_window = max(50, total_bars // 200)
    step = max(50, total_bars // 200)

    logger.info(f"Total bars: {total_bars:,}")
    logger.info(f"Walk-forward windows — train: {train_window}, test: {test_window}, step: {step}")

    # Train with settings scaled for dataset size
    model.fit(
        df,
        available_features,
        target_col="target",
        train_ratio=0.7,              # 70% train, 30% test
        num_boost_round=150,          # More rounds for larger dataset
        early_stopping_rounds=15,     # Generous stopping window
    )
    
    if model.fitted:
        # Show feature importance
        logger.info("Top 10 Feature Importance:")
        for feat, imp in model.get_feature_importance(10).items():
            logger.info(f"  {feat}: {imp:.4f}")
        
        # Walk-forward validation with scaled windows
        logger.info("Running walk-forward validation...")
        results = model.walk_forward_train(
            df,
            available_features,
            "target",
            train_window=train_window,
            test_window=test_window,
            step=step,
        )
        
        if results:
            avg_train = np.mean([r[0] for r in results])
            avg_test = np.mean([r[1] for r in results])
            logger.info(f"Walk-forward Results:")
            logger.info(f"  Avg Train AUC: {avg_train:.4f}")
            logger.info(f"  Avg Test AUC: {avg_test:.4f}")
            logger.info(f"  Overfitting ratio: {avg_train/avg_test:.2f}")
    
    return model


def save_training_data(df: pl.DataFrame, path: str = "data/training_data.parquet"):
    """Save training data for future reference."""
    df.write_parquet(path)
    logger.info(f"Training data saved to {path}")


def main():
    """Main training pipeline."""
    # ---------------------------------------------------------------
    # CLI arguments
    # ---------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="GOLDAI Model Training Script"
    )
    parser.add_argument(
        "--bars", type=int, default=140000,
        help="Number of M15 bars to fetch (140000 ≈ 5 years of M15 data, default: 140000)",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Trading symbol (default: from config, e.g. XAUUSD)",
    )
    parser.add_argument(
        "--timeframe", type=str, default=None,
        help="Timeframe (default: from config, e.g. M15)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SMART TRADING BOT - MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load config
    config = get_config()

    # Override symbol/timeframe from CLI if provided
    symbol = args.symbol or config.symbol
    timeframe = args.timeframe or config.execution_timeframe

    logger.info(f"Symbol   : {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Bars     : {args.bars:,} (~{args.bars / (96 * 260):.1f} years of M15 data)")
    logger.info(f"Capital  : ${config.capital:,.2f}")
    logger.info(f"Mode     : {config.capital_mode.value}")
    
    # Connect to MT5
    logger.info("Connecting to MT5...")
    connector = MT5Connector(
        login=config.mt5_login,
        password=config.mt5_password,
        server=config.mt5_server,
        path=config.mt5_path,
    )
    
    try:
        connector.connect()
        logger.info("MT5 connected successfully!")
        
        # Get account info
        balance = connector.account_balance
        equity = connector.account_equity
        logger.info(f"Account Balance: ${balance:,.2f}")
        logger.info(f"Account Equity: ${equity:,.2f}")
        
    except Exception as e:
        logger.error(f"MT5 connection failed: {e}")
        logger.info("Please ensure:")
        logger.info("  1. MT5 terminal is running")
        logger.info("  2. Auto-trading is enabled")
        logger.info("  3. Login credentials are correct")
        return
    
    try:
        # Fetch data
        df = fetch_training_data(
            connector,
            symbol,
            timeframe,
            bars=args.bars,
        )

        # Training summary: date range and data statistics
        date_min = df["time"].min()
        date_max = df["time"].max()
        logger.info("=" * 60)
        logger.info("TRAINING DATA SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Date range : {date_min} → {date_max}")
        logger.info(f"  Total bars : {len(df):,}")
        logger.info(f"  Symbol     : {symbol}  Timeframe: {timeframe}")
        if "close" in df.columns:
            logger.info(f"  Price range: {df['close'].min():.2f} – {df['close'].max():.2f}")
        logger.info("=" * 60)
        
        # Prepare features
        df = prepare_features(df)
        
        # Save raw data
        save_training_data(df)
        
        # Train HMM
        hmm_model = train_hmm_model(df)
        
        # Add regime to features
        if hmm_model.fitted:
            df = hmm_model.predict(df)
        
        # Train XGBoost
        xgb_model = train_xgboost_model(df)
        
        # Summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"HMM Model   : {'SAVED' if hmm_model.fitted else 'FAILED'}")
        logger.info(f"XGBoost Model: {'SAVED' if xgb_model.fitted else 'FAILED'}")
        logger.info(f"Models saved in : models/")
        logger.info(f"Training data   : data/")
        logger.info(f"Date range      : {date_min} → {date_max}")
        logger.info(f"Bars trained on : {len(df):,}")
        logger.info("=" * 60)
        logger.info("Next step: python backtests/backtest_v3.py --bars 140000 --balance 439 --lot 0.01")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        connector.disconnect()
        logger.info("MT5 disconnected")


if __name__ == "__main__":
    main()

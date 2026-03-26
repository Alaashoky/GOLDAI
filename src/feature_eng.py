"""
Feature Engineering Module - Pure Polars
=========================================
Technical indicators and ML features using Polars expressions.

NO PANDAS. NO TA-Lib.

Implements:
- RSI (Wilder's Smoothing)
- ATR (Average True Range)
- MACD
- Bollinger Bands
- EMA/SMA
- Volume Profile
- ML-ready features
"""

import polars as pl
import numpy as np
from typing import List, Optional
from loguru import logger


class FeatureEngineer:
    """
    Feature engineering using pure Polars expressions.
    
    All calculations are vectorized for maximum performance.
    No loops, no external TA libraries.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        pass
    
    def calculate_all(
        self,
        df: pl.DataFrame,
        include_ml_features: bool = True,
        include_smc_features: bool = True,
    ) -> pl.DataFrame:
        """
        Calculate all technical indicators and features.

        Args:
            df: Polars DataFrame with OHLCV data
            include_ml_features: Include ML-specific features
            include_smc_features: Include SMC context features (requires SMC cols)

        Returns:
            DataFrame with all features added
        """
        df = self.calculate_rsi(df)
        df = self.calculate_atr(df)
        df = self.calculate_macd(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_ema_crossover(df)
        df = self.calculate_volume_features(df)

        if include_ml_features:
            df = self.calculate_ml_features(df)

        if include_smc_features:
            df = self.calculate_smc_features(df)

        return df
    
    def calculate_rsi(
        self,
        df: pl.DataFrame,
        period: int = 14,
        column: str = "close",
    ) -> pl.DataFrame:
        """
        Calculate RSI using Wilder's Smoothing.
        
        Wilder's Smoothing uses alpha = 1/n for ewm_mean.
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        
        Args:
            df: DataFrame with price data
            period: RSI period (default 14)
            column: Price column to use
            
        Returns:
            DataFrame with RSI column added
        """
        # Wilder's smoothing alpha
        alpha = 1.0 / period
        
        df = df.with_columns([
            # Calculate price changes
            pl.col(column).diff().alias("_delta"),
        ])
        
        df = df.with_columns([
            # Separate gains (positive changes) and losses (negative changes)
            pl.when(pl.col("_delta") > 0)
                .then(pl.col("_delta"))
                .otherwise(0.0)
                .alias("_gains"),
            
            pl.when(pl.col("_delta") < 0)
                .then(-pl.col("_delta"))
                .otherwise(0.0)
                .alias("_losses"),
        ])
        
        df = df.with_columns([
            # Apply Wilder's smoothing (EWM with alpha=1/period, adjust=False)
            pl.col("_gains")
                .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
                .alias("_avg_gain"),
            
            pl.col("_losses")
                .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
                .alias("_avg_loss"),
        ])
        
        df = df.with_columns([
            # Calculate RSI
            pl.when(pl.col("_avg_loss") == 0)
                .then(100.0)
                .otherwise(
                    100.0 - (100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss")))
                )
                .alias("rsi"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_delta", "_gains", "_losses", "_avg_gain", "_avg_loss"])
        
        logger.debug(f"RSI calculated (period={period})")
        return df
    
    def calculate_atr(
        self,
        df: pl.DataFrame,
        period: int = 14,
    ) -> pl.DataFrame:
        """
        Calculate ATR (Average True Range) using Wilder's Smoothing.
        
        True Range = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
        ATR = Wilder's smoothing of True Range
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period (default 14)
            
        Returns:
            DataFrame with ATR column added
        """
        alpha = 1.0 / period
        
        df = df.with_columns([
            # Previous close for True Range calculation
            pl.col("close").shift(1).alias("_prev_close"),
        ])
        
        df = df.with_columns([
            # Three components of True Range
            (pl.col("high") - pl.col("low")).alias("_hl"),
            (pl.col("high") - pl.col("_prev_close")).abs().alias("_hpc"),
            (pl.col("low") - pl.col("_prev_close")).abs().alias("_lpc"),
        ])
        
        df = df.with_columns([
            # True Range = maximum of three components
            pl.max_horizontal("_hl", "_hpc", "_lpc").alias("_tr"),
        ])
        
        df = df.with_columns([
            # Apply Wilder's smoothing to get ATR
            pl.col("_tr")
                .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
                .alias("atr"),
        ])
        
        # Calculate ATR percentage (ATR / Close)
        df = df.with_columns([
            (pl.col("atr") / pl.col("close") * 100).alias("atr_percent"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_prev_close", "_hl", "_hpc", "_lpc", "_tr"])
        
        logger.debug(f"ATR calculated (period={period})")
        return df
    
    def calculate_macd(
        self,
        df: pl.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ) -> pl.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line)
        Histogram = MACD Line - Signal Line
        
        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line period (default 9)
            column: Price column to use
            
        Returns:
            DataFrame with MACD columns added
        """
        df = df.with_columns([
            # Calculate EMAs
            pl.col(column)
                .ewm_mean(span=fast_period, adjust=False)
                .alias("_ema_fast"),
            pl.col(column)
                .ewm_mean(span=slow_period, adjust=False)
                .alias("_ema_slow"),
        ])
        
        df = df.with_columns([
            # MACD line
            (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd"),
        ])
        
        df = df.with_columns([
            # Signal line
            pl.col("macd")
                .ewm_mean(span=signal_period, adjust=False)
                .alias("macd_signal"),
        ])
        
        df = df.with_columns([
            # Histogram
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_ema_fast", "_ema_slow"])
        
        logger.debug(f"MACD calculated ({fast_period}/{slow_period}/{signal_period})")
        return df
    
    def calculate_bollinger_bands(
        self,
        df: pl.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = "close",
    ) -> pl.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Middle Band = SMA(period)
        Upper Band = Middle + (std_dev * StdDev)
        Lower Band = Middle - (std_dev * StdDev)
        
        Args:
            df: DataFrame with price data
            period: SMA period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
            column: Price column to use
            
        Returns:
            DataFrame with Bollinger Band columns added
        """
        df = df.with_columns([
            # Middle band (SMA)
            pl.col(column)
                .rolling_mean(window_size=period)
                .alias("bb_middle"),
            
            # Rolling standard deviation
            pl.col(column)
                .rolling_std(window_size=period)
                .alias("_bb_std"),
        ])
        
        df = df.with_columns([
            # Upper and lower bands
            (pl.col("bb_middle") + std_dev * pl.col("_bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - std_dev * pl.col("_bb_std")).alias("bb_lower"),
        ])
        
        df = df.with_columns([
            # Bollinger Band Width (volatility indicator)
            ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle"))
                .alias("bb_width"),
            
            # %B (position within bands, 0-1 normally)
            ((pl.col(column) - pl.col("bb_lower")) / 
             (pl.col("bb_upper") - pl.col("bb_lower")))
                .alias("bb_percent_b"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_bb_std"])
        
        logger.debug(f"Bollinger Bands calculated (period={period}, std={std_dev})")
        return df
    
    def calculate_ema_crossover(
        self,
        df: pl.DataFrame,
        fast_period: int = 9,
        slow_period: int = 21,
        column: str = "close",
    ) -> pl.DataFrame:
        """
        Calculate EMA crossover signals.
        
        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            column: Price column
            
        Returns:
            DataFrame with EMA and crossover columns
        """
        df = df.with_columns([
            pl.col(column)
                .ewm_mean(span=fast_period, adjust=False)
                .alias(f"ema_{fast_period}"),
            pl.col(column)
                .ewm_mean(span=slow_period, adjust=False)
                .alias(f"ema_{slow_period}"),
        ])
        
        # EMA crossover detection
        df = df.with_columns([
            (pl.col(f"ema_{fast_period}") > pl.col(f"ema_{slow_period}"))
                .alias("_ema_above"),
        ])
        
        df = df.with_columns([
            pl.col("_ema_above").shift(1).alias("_ema_above_prev"),
        ])
        
        df = df.with_columns([
            # Bullish crossover: fast crosses above slow
            (pl.col("_ema_above") & ~pl.col("_ema_above_prev").fill_null(False))
                .cast(pl.Int8)
                .alias("ema_cross_bull"),
            
            # Bearish crossover: fast crosses below slow
            (~pl.col("_ema_above") & pl.col("_ema_above_prev").fill_null(False))
                .cast(pl.Int8)
                .alias("ema_cross_bear"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_ema_above", "_ema_above_prev"])
        
        logger.debug(f"EMA crossover calculated ({fast_period}/{slow_period})")
        return df
    
    def calculate_volume_features(
        self,
        df: pl.DataFrame,
        period: int = 20,
    ) -> pl.DataFrame:
        """
        Calculate volume-based features.
        
        Args:
            df: DataFrame with volume data
            period: Period for volume analysis
            
        Returns:
            DataFrame with volume features
        """
        if "volume" not in df.columns:
            logger.warning("Volume column not found, skipping volume features")
            return df
        
        df = df.with_columns([
            # Volume SMA
            pl.col("volume")
                .rolling_mean(window_size=period)
                .alias("volume_sma"),
        ])
        
        df = df.with_columns([
            # Volume ratio (current / average)
            (pl.col("volume") / pl.col("volume_sma")).alias("volume_ratio"),
            
            # Volume trend (increasing or decreasing)
            (pl.col("volume") > pl.col("volume").shift(1))
                .cast(pl.Int8)
                .alias("volume_increasing"),
        ])
        
        # High volume bars (> 1.5x average)
        df = df.with_columns([
            (pl.col("volume_ratio") > 1.5)
                .cast(pl.Int8)
                .alias("high_volume"),
        ])

        # === ADVANCED: Order Flow Imbalance (Pseudo-OFI) ===
        # Phase 4 - Advanced Exit Strategies
        # Directional volume classification
        df = df.with_columns([
            # Buy volume: close > open (bullish candle)
            pl.when(pl.col("close") > pl.col("open"))
                .then(pl.col("volume"))
                .otherwise(0)
                .alias("buy_volume"),

            # Sell volume: close < open (bearish candle)
            pl.when(pl.col("close") < pl.col("open"))
                .then(pl.col("volume"))
                .otherwise(0)
                .alias("sell_volume"),
        ])

        # Pseudo-OFI calculation
        df = df.with_columns([
            (
                (pl.col("buy_volume") - pl.col("sell_volume")) /
                (pl.col("buy_volume") + pl.col("sell_volume") + 1e-9)
            )
            .alias("ofi_pseudo")
            .fill_nan(0)
            .fill_null(0)
        ])

        # OFI trend and divergence
        df = df.with_columns([
            # Rolling OFI mean (20 bars)
            pl.col("ofi_pseudo").rolling_mean(20).alias("ofi_trend"),

            # Rolling OFI std (for normalization)
            pl.col("ofi_pseudo").rolling_std(20).alias("ofi_std"),
        ])

        df = df.with_columns([
            # OFI divergence (current vs trend)
            (pl.col("ofi_pseudo") - pl.col("ofi_trend"))
            .alias("ofi_divergence")
            .fill_nan(0)
            .fill_null(0)
        ])

        # Volume momentum (acceleration)
        df = df.with_columns([
            # Volume ratio change (1st derivative)
            (pl.col("volume_ratio") / pl.col("volume_ratio").shift(1) - 1)
            .alias("volume_momentum")
            .fill_nan(0)
            .fill_null(0),
        ])

        # Volume toxicity metric
        # Combines: volume acceleration + OFI divergence + spread expansion
        if "spread" in df.columns:
            df = df.with_columns([
                # Toxicity score (0-5+)
                (
                    pl.col("volume_momentum").abs() +
                    pl.col("ofi_divergence").abs() * 2 +
                    (pl.col("spread") / pl.col("spread").rolling_mean(20) - 1).abs()
                )
                .alias("toxicity")
                .fill_nan(0)
                .fill_null(0)
            ])
        else:
            # Simplified toxicity without spread
            df = df.with_columns([
                (
                    pl.col("volume_momentum").abs() +
                    pl.col("ofi_divergence").abs() * 2
                )
                .alias("toxicity")
                .fill_nan(0)
                .fill_null(0)
            ])

        logger.debug(f"Volume features calculated (period={period}, includes OFI & toxicity)")
        return df
    
    def calculate_ml_features(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calculate ML-specific features for XGBoost.
        
        Includes:
        - Returns and momentum
        - Price position features
        - Volatility features
        - Lag features
        - Time-based features
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            DataFrame with ML features
        """
        # Returns and momentum
        df = df.with_columns([
            # Simple returns
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("returns_1"),
            (pl.col("close") / pl.col("close").shift(5) - 1).alias("returns_5"),
            (pl.col("close") / pl.col("close").shift(20) - 1).alias("returns_20"),
            
            # Log returns
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_returns"),
        ])
        
        # Price position features
        df = df.with_columns([
            # Price position within day's range
            ((pl.col("close") - pl.col("low")) / 
             (pl.col("high") - pl.col("low")))
                .alias("price_position"),
            
            # Distance from SMA
            pl.col("close")
                .rolling_mean(window_size=20)
                .alias("_sma_20"),
        ])
        
        df = df.with_columns([
            (pl.col("close") / pl.col("_sma_20") - 1).alias("dist_from_sma_20"),
        ])
        
        # Volatility features
        df = df.with_columns([
            # Realized volatility (rolling std of returns)
            pl.col("log_returns")
                .rolling_std(window_size=20)
                .alias("volatility_20"),
            
            # Normalized range
            ((pl.col("high") - pl.col("low")) / pl.col("close"))
                .alias("normalized_range"),
            
            # Average normalized range
            ((pl.col("high") - pl.col("low")) / pl.col("close"))
                .rolling_mean(window_size=14)
                .alias("avg_normalized_range"),
        ])
        
        # Lag features
        df = df.with_columns([
            pl.col("close").shift(1).alias("close_lag_1"),
            pl.col("close").shift(2).alias("close_lag_2"),
            pl.col("close").shift(3).alias("close_lag_3"),
            pl.col("close").shift(5).alias("close_lag_5"),
        ])
        
        # Trend features
        df = df.with_columns([
            # Higher high / lower low sequences
            (pl.col("high") > pl.col("high").shift(1))
                .cast(pl.Int8)
                .alias("higher_high"),
            (pl.col("low") < pl.col("low").shift(1))
                .cast(pl.Int8)
                .alias("lower_low"),
        ])
        
        # Rolling trend strength
        df = df.with_columns([
            pl.col("higher_high")
                .rolling_sum(window_size=5)
                .alias("hh_count_5"),
            pl.col("lower_low")
                .rolling_sum(window_size=5)
                .alias("ll_count_5"),
        ])
        
        # Time-based features (if datetime column exists)
        if "time" in df.columns and df["time"].dtype == pl.Datetime:
            df = df.with_columns([
                pl.col("time").dt.hour().alias("hour"),
                pl.col("time").dt.weekday().alias("weekday"),
                
                # Trading session indicators
                ((pl.col("time").dt.hour() >= 8) & (pl.col("time").dt.hour() < 16))
                    .cast(pl.Int8)
                    .alias("london_session"),
                ((pl.col("time").dt.hour() >= 13) & (pl.col("time").dt.hour() < 21))
                    .cast(pl.Int8)
                    .alias("ny_session"),
            ])
        
        # Drop temporary columns
        df = df.drop(["_sma_20"])
        
        logger.debug("ML features calculated")
        return df

    def calculate_smc_features(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calculate SMC-context features for ML model.

        These features give the model richer context about the
        quality and strength of each SMC setup — not just 0/1 signals.

        Requires columns: bos, choch, fvg_signal, ob, market_structure, atr, close, high, low
        Returns: DataFrame with additional SMC-context feature columns.
        """
        # ── BOS / CHoCH rolling counts ───────────────────────────────────────
        if "bos" in df.columns:
            df = df.with_columns([
                # How many bullish BOS in last 20 bars (trend strength)
                pl.col("bos")
                    .rolling_sum(window_size=20)
                    .alias("bos_count_20"),
                # Most-recent BOS direction (signed, faded over distance)
                pl.col("bos").alias("bos_recent"),
            ])
        else:
            df = df.with_columns([
                pl.lit(0).cast(pl.Int8).alias("bos_count_20"),
                pl.lit(0).cast(pl.Int8).alias("bos_recent"),
            ])

        if "choch" in df.columns:
            df = df.with_columns([
                # Was there a CHoCH in the last 5 bars? (fresh reversal signal)
                pl.col("choch")
                    .abs()
                    .rolling_max(window_size=5)
                    .cast(pl.Int8)
                    .alias("choch_recent_5"),
            ])
        else:
            df = df.with_columns([
                pl.lit(0).cast(pl.Int8).alias("choch_recent_5"),
            ])

        # ── FVG size (bigger gap = stronger imbalance) ────────────────────────
        if all(c in df.columns for c in ["fvg_top", "fvg_bottom", "close"]):
            df = df.with_columns([
                ((pl.col("fvg_top") - pl.col("fvg_bottom")).abs() / pl.col("close"))
                    .fill_nan(0)
                    .fill_null(0)
                    .alias("fvg_size"),
            ])
        elif "fvg_signal" in df.columns:
            # fallback: proxy via ATR-normalised dummy when price levels unavailable
            if "atr" in df.columns:
                df = df.with_columns([
                    (pl.col("fvg_signal").abs().cast(pl.Float64) *
                     pl.col("atr") / pl.col("close"))
                        .fill_nan(0).fill_null(0)
                        .alias("fvg_size"),
                ])
            else:
                df = df.with_columns([
                    pl.col("fvg_signal").abs().cast(pl.Float64).alias("fvg_size"),
                ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("fvg_size")])

        # ── OB distance from current price ───────────────────────────────────
        if all(c in df.columns for c in ["ob_top", "ob_bottom", "close"]):
            df = df.with_columns([
                (
                    (pl.col("close") - (pl.col("ob_top") + pl.col("ob_bottom")) / 2).abs()
                    / pl.col("close")
                )
                .fill_nan(0).fill_null(0)
                .alias("ob_distance"),
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("ob_distance")])

        # ── Swing strength (how many confirming bars each swing has) ──────────
        if "swing_high" in df.columns:
            df = df.with_columns([
                pl.col("swing_high")
                    .rolling_sum(window_size=10)
                    .alias("swing_high_count_10"),
            ])
        else:
            df = df.with_columns([pl.lit(0).cast(pl.Int8).alias("swing_high_count_10")])

        if "swing_low" in df.columns:
            df = df.with_columns([
                pl.col("swing_low")
                    .rolling_sum(window_size=10)
                    .alias("swing_low_count_10"),
            ])
        else:
            df = df.with_columns([pl.lit(0).cast(pl.Int8).alias("swing_low_count_10")])

        # ── Market structure age (bars since last BOS/CHoCH) ─────────────────
        if "bos" in df.columns and "choch" in df.columns:
            df = df.with_columns([
                (pl.col("bos").abs() + pl.col("choch").abs())
                    .cast(pl.Int8)
                    .alias("_structure_event"),
            ])
            # Cumulative count reset at each event → bars since last event
            structure_arr = df["_structure_event"].to_numpy()
            ages = np.zeros(len(structure_arr), dtype=np.int32)
            counter = 0
            for i, val in enumerate(structure_arr):
                if val != 0:
                    counter = 0
                else:
                    counter += 1
                ages[i] = counter
            df = df.with_columns([
                pl.Series("structure_age", ages),
            ])
            df = df.drop(["_structure_event"])
        else:
            df = df.with_columns([pl.lit(0).cast(pl.Int32).alias("structure_age")])

        # ── FVG freshness: is price currently inside the FVG zone? ───────────
        if all(c in df.columns for c in ["fvg_top", "fvg_bottom", "close"]):
            df = df.with_columns([
                (
                    (pl.col("close") >= pl.col("fvg_bottom")) &
                    (pl.col("close") <= pl.col("fvg_top"))
                )
                .cast(pl.Int8)
                .alias("fvg_in_zone"),
            ])
        else:
            df = df.with_columns([pl.lit(0).cast(pl.Int8).alias("fvg_in_zone")])

        logger.debug("SMC context features calculated")
        return df

    def create_target(
        self,
        df: pl.DataFrame,
        lookahead: int = 8,
        threshold: float = 0.0003,
        tp_atr_mult: float = 1.5,
        sl_atr_mult: float = 2.0,
        use_smc_aware: bool = True,
        drop_no_setup: bool = False,
    ) -> pl.DataFrame:
        """
        Create target variable for ML training.

        Two modes:
        1. SMC-aware (use_smc_aware=True, default):
           target = 1  if there is an active SMC BUY setup at bar t
                        AND close[t + lookahead] >= close[t] + tp_atr_mult * ATR[t]
           target = 0  if there is an active SMC SELL setup at bar t
                        AND close[t + lookahead] <= close[t] - tp_atr_mult * ATR[t]
           Bars without a clear setup are labelled NaN (dropped later by drop_nulls).

           Requires columns: market_structure (or bos/choch), atr

        2. Generic (use_smc_aware=False):
           Identical to the previous behaviour — binary 1/0 based on a
           simple percentage threshold.  Kept for backwards compatibility.

        Args:
            df: DataFrame with features already calculated
            lookahead: Bars ahead to check target price
            threshold: Generic-mode minimum return (ignored in SMC-aware mode)
            tp_atr_mult: ATR multiplier for TP in SMC-aware mode
            sl_atr_mult: ATR multiplier for SL in SMC-aware mode (unused in
                         labelling but logged for reference)
            use_smc_aware: Use SMC-context labelling (recommended)
            drop_no_setup: When True (and use_smc_aware=True), rows where
                           target IS NULL are dropped before returning.
                           Default False keeps the full time series intact so
                           callers such as fit() can handle filtering themselves.

        Returns:
            DataFrame with 'target' and 'target_return' columns added
        """
        # Future close is always needed
        df = df.with_columns([
            pl.col("close").shift(-lookahead).alias("_future_close"),
        ])

        df = df.with_columns([
            # Return target (for regression / diagnostics)
            (pl.col("_future_close") / pl.col("close") - 1).alias("target_return"),
        ])

        if use_smc_aware and "atr" in df.columns:
            # ── determine BUY / SELL bias from SMC structure ─────────────────
            has_market_structure = "market_structure" in df.columns
            has_bos = "bos" in df.columns
            has_choch = "choch" in df.columns

            if has_market_structure:
                buy_bias  = pl.col("market_structure") > 0
                sell_bias = pl.col("market_structure") < 0
            elif has_bos or has_choch:
                bos_col   = pl.col("bos")   if has_bos   else pl.lit(0)
                choch_col = pl.col("choch") if has_choch else pl.lit(0)
                buy_bias  = (bos_col > 0) | (choch_col > 0)
                sell_bias = (bos_col < 0) | (choch_col < 0)
            else:
                # No SMC structure available — fall back to generic mode
                logger.warning(
                    "SMC-aware target requested but no market_structure/bos/choch columns "
                    "found. Falling back to generic threshold-based target."
                )
                df = df.with_columns([
                    (pl.col("target_return") > threshold)
                        .cast(pl.Int32)
                        .alias("target"),
                ])
                df = df.drop(["_future_close"])
                logger.debug(
                    f"Target created (generic fallback, lookahead={lookahead}, "
                    f"threshold={threshold})"
                )
                return df

            # TP distance = tp_atr_mult × ATR
            tp_dist = pl.col("atr") * tp_atr_mult

            # SMC-aware labelling:
            #   BUY setup that hit TP  → 1
            #   SELL setup that hit TP → 0
            #   Anything else          → null (will be dropped by drop_nulls)
            df = df.with_columns([
                pl.when(buy_bias & (pl.col("_future_close") >= pl.col("close") + tp_dist))
                    .then(pl.lit(1))
                    .when(sell_bias & (pl.col("_future_close") <= pl.col("close") - tp_dist))
                    .then(pl.lit(0))
                    .otherwise(pl.lit(None))
                    .cast(pl.Int32)
                    .alias("target"),
            ])

            n_total    = len(df)
            n_labelled = df["target"].drop_nulls().len()
            n_buy      = (df["target"] == 1).sum()
            n_sell     = (df["target"] == 0).sum()
            logger.info(
                f"SMC-aware target: {n_labelled}/{n_total} bars labelled "
                f"({n_buy} BUY-wins, {n_sell} SELL-wins, "
                f"TP_mult={tp_atr_mult}×ATR, lookahead={lookahead})"
            )
        else:
            # ── Generic mode (original behaviour) ────────────────────────────
            df = df.with_columns([
                (pl.col("target_return") > threshold)
                    .cast(pl.Int32)
                    .alias("target"),
            ])
            logger.debug(
                f"Target created (generic, lookahead={lookahead}, threshold={threshold})"
            )

        df = df.drop(["_future_close"])

        if use_smc_aware and drop_no_setup:
            before = len(df)
            df = df.filter(pl.col("target").is_not_null())
            logger.info(f"drop_no_setup: kept {len(df)}/{before} labelled bars")

        return df
    
    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Get list of feature columns for ML training.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            List of feature column names
        """
        # Exclude non-feature columns
        exclude_cols = {
            "time", "open", "high", "low", "close", "volume",
            "spread", "real_volume", "target", "target_return",
            # SMC columns that are signals, not features
            "swing_high_level", "swing_low_level",
            "fvg_top", "fvg_bottom", "fvg_mid",
            "ob_top", "ob_bottom",
            "bos_level", "choch_level",
            "bsl_level", "ssl_level",
            "last_swing_high", "last_swing_low",
        }
        
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and not col.startswith("_")  # Temporary columns
        ]
        
        return feature_cols


def get_default_feature_engineer() -> FeatureEngineer:
    """Get default configured feature engineer."""
    return FeatureEngineer()


if __name__ == "__main__":
    # Test feature engineering with synthetic data
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create synthetic OHLCV data
    np.random.seed(42)
    n = 500
    
    base_price = 2000.0
    returns = np.random.randn(n) * 0.002
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pl.DataFrame({
        "time": [datetime.now() - timedelta(minutes=15*i) for i in range(n-1, -1, -1)],
        "open": prices,
        "high": prices * (1 + np.abs(np.random.randn(n)) * 0.001),
        "low": prices * (1 - np.abs(np.random.randn(n)) * 0.001),
        "close": prices * (1 + np.random.randn(n) * 0.0005),
        "volume": np.random.randint(1000, 10000, n),
    })
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Calculate all features
    df = fe.calculate_all(df, include_ml_features=True)
    
    # Create target
    df = fe.create_target(df, lookahead=1)
    
    # Get feature columns
    feature_cols = fe.get_feature_columns(df)
    
    print("\n=== Feature Engineering Test ===")
    print(f"Total columns: {len(df.columns)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"\nFeatures: {feature_cols}")
    
    # Show sample with key indicators
    print("\n=== Sample Data (Last 5 Rows) ===")
    display_cols = ["time", "close", "rsi", "atr", "macd", "bb_percent_b", "returns_1", "target"]
    available_cols = [c for c in display_cols if c in df.columns]
    print(df.select(available_cols).tail(5))
    
    # Stats for key indicators
    print("\n=== Indicator Statistics ===")
    for col in ["rsi", "atr", "macd", "bb_percent_b"]:
        if col in df.columns:
            stats = df[col].describe()
            print(f"{col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

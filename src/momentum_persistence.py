from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_MIN_BARS = 14


@dataclass
class MomentumResult:
    """Output of the MomentumPersistence analyser.

    Attributes:
        score: Combined momentum persistence score in [0, 1].
        adx_strength: ADX value (0–100; >25 = trending).
        macd_slope: Normalised MACD slope direction in [-1, 1].
        divergence_detected: True when RSI divergence is present.
        exhaustion_detected: True when momentum exhaustion is detected.
        direction: Net momentum direction (+1 bullish, -1 bearish, 0 neutral).
    """

    score: float
    adx_strength: float
    macd_slope: float
    divergence_detected: bool
    exhaustion_detected: bool
    direction: int


class MomentumPersistence:
    """Detect and score momentum persistence using multiple indicators.

    Combines ADX trend strength, MACD slope, RSI divergence, and volume-based
    exhaustion signals into a single composite score that represents how likely
    the current momentum is to continue.

    Args:
        adx_period: Period for ADX calculation (default 14).
        divergence_lookback: Bars to look back for divergence (default 10).
    """

    def __init__(
        self,
        adx_period: int = 14,
        divergence_lookback: int = 10,
    ) -> None:
        self.adx_period = adx_period
        self.divergence_lookback = divergence_lookback

    # ------------------------------------------------------------------
    # RSI divergence
    # ------------------------------------------------------------------

    def calculate_rsi_divergence(
        self,
        prices: np.ndarray,
        rsi_values: np.ndarray,
    ) -> bool:
        """Detect regular RSI divergence (price vs RSI extremes disagree).

        Bullish divergence: lower price low but higher RSI low.
        Bearish divergence: higher price high but lower RSI high.

        Args:
            prices: Array of close prices.
            rsi_values: Corresponding RSI values (same length).

        Returns:
            True when any divergence is detected in the lookback window.
        """
        lb = self.divergence_lookback
        if len(prices) < lb + 2 or len(rsi_values) < lb + 2:
            return False

        p = np.asarray(prices[-lb:], dtype=float)
        r = np.asarray(rsi_values[-lb:], dtype=float)

        # Valid RSI mask (non-nan)
        valid = ~np.isnan(r)
        if valid.sum() < 4:
            return False

        p_valid = p[valid]
        r_valid = r[valid]

        # Bearish divergence: price higher high but RSI lower high
        price_higher = p_valid[-1] > p_valid[0]
        rsi_lower = r_valid[-1] < r_valid[0]
        bearish_div = price_higher and rsi_lower

        # Bullish divergence: price lower low but RSI higher low
        price_lower = p_valid[-1] < p_valid[0]
        rsi_higher = r_valid[-1] > r_valid[0]
        bullish_div = price_lower and rsi_higher

        result = bearish_div or bullish_div
        if result:
            logger.debug("RSI divergence detected (bearish=%s bullish=%s)",
                         bearish_div, bullish_div)
        return result

    # ------------------------------------------------------------------
    # MACD slope
    # ------------------------------------------------------------------

    def calculate_macd_slope(
        self,
        macd_line: np.ndarray,
        signal_line: np.ndarray,
    ) -> float:
        """Compute normalised MACD momentum direction.

        Args:
            macd_line: MACD histogram line values.
            signal_line: MACD signal line values.

        Returns:
            Score in [-1, 1]; positive = bullish momentum.
        """
        if len(macd_line) < 3:
            return 0.0

        macd = np.asarray(macd_line, dtype=float)
        signal = np.asarray(signal_line, dtype=float)

        histogram = macd - signal
        recent = histogram[-3:]
        valid = recent[~np.isnan(recent)]
        if len(valid) < 2:
            return 0.0

        slope = float(np.polyfit(range(len(valid)), valid, 1)[0])

        # Normalise using rolling std
        std = float(np.std(histogram[~np.isnan(histogram)])) or 1.0
        normalised = np.tanh(slope / std)
        return round(float(normalised), 4)

    # ------------------------------------------------------------------
    # ADX
    # ------------------------------------------------------------------

    def get_adx_strength(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> float:
        """Compute ADX trend strength.

        Args:
            high: High price array.
            low: Low price array.
            close: Close price array.

        Returns:
            ADX value in [0, 100]; higher = stronger trend.
        """
        n = self.adx_period
        if len(close) < n + 1:
            return 0.0

        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        prev_close = close[:-1]
        curr_high = high[1:]
        curr_low = low[1:]

        # True range
        tr = np.maximum(
            curr_high - curr_low,
            np.maximum(
                np.abs(curr_high - prev_close),
                np.abs(curr_low - prev_close),
            ),
        )

        # Directional movements
        up_move = curr_high - high[:-1]
        down_move = low[:-1] - curr_low

        dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smooth
        def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
            result = np.empty_like(arr)
            result[: period - 1] = np.nan
            result[period - 1] = arr[:period].sum()
            for i in range(period, len(arr)):
                result[i] = result[i - 1] - result[i - 1] / period + arr[i]
            return result

        tr_s = _wilder_smooth(tr, n)
        dm_plus_s = _wilder_smooth(dm_plus, n)
        dm_minus_s = _wilder_smooth(dm_minus, n)

        with np.errstate(divide="ignore", invalid="ignore"):
            di_plus = 100.0 * dm_plus_s / tr_s
            di_minus = 100.0 * dm_minus_s / tr_s
            dx = 100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)

        adx_arr = _wilder_smooth(np.nan_to_num(dx), n)
        # Return last valid value
        valid = adx_arr[~np.isnan(adx_arr)]
        return float(valid[-1]) if len(valid) else 0.0

    # ------------------------------------------------------------------
    # Exhaustion detection
    # ------------------------------------------------------------------

    def detect_exhaustion(
        self,
        prices: np.ndarray,
        volume: np.ndarray,
    ) -> bool:
        """Detect momentum exhaustion via price-volume divergence.

        Exhaustion signal: price continuing to new extremes while volume
        is declining (momentum running out of fuel).

        Args:
            prices: Recent close prices.
            volume: Corresponding volume / tick-volume.

        Returns:
            True when exhaustion is likely.
        """
        if len(prices) < 5 or len(volume) < 5:
            return False

        p = np.asarray(prices[-5:], dtype=float)
        v = np.asarray(volume[-5:], dtype=float)

        price_slope = float(np.polyfit(range(len(p)), p, 1)[0])
        vol_slope = float(np.polyfit(range(len(v)), v, 1)[0])

        # Exhaustion: price trending strongly but volume declining
        price_trending = abs(price_slope) > 0.0
        volume_declining = vol_slope < 0

        result = price_trending and volume_declining
        if result:
            logger.debug("Exhaustion detected: price_slope=%.4f vol_slope=%.4f",
                         price_slope, vol_slope)
        return result

    # ------------------------------------------------------------------
    # Combined score
    # ------------------------------------------------------------------

    def get_momentum_score(self, df: pl.DataFrame) -> MomentumResult:
        """Compute combined momentum persistence score from OHLCV DataFrame.

        Args:
            df: Polars DataFrame with columns: open, high, low, close, volume
                (or tick_volume).  Minimum ``_MIN_BARS`` rows required.

        Returns:
            MomentumResult with composite score and component diagnostics.
        """
        if len(df) < _MIN_BARS:
            logger.warning("Insufficient bars for momentum score (%d)", len(df))
            return MomentumResult(
                score=0.0,
                adx_strength=0.0,
                macd_slope=0.0,
                divergence_detected=False,
                exhaustion_detected=False,
                direction=0,
            )

        close_arr = df["close"].to_numpy().astype(float)
        high_arr = df["high"].to_numpy().astype(float)
        low_arr = df["low"].to_numpy().astype(float)
        vol_col = "volume" if "volume" in df.columns else "tick_volume"
        vol_arr = df[vol_col].to_numpy().astype(float) if vol_col in df.columns else np.ones(len(df))

        # RSI via numpy
        delta = np.diff(close_arr)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.convolve(gain, np.ones(14) / 14, mode="valid")
        avg_loss = np.convolve(loss, np.ones(14) / 14, mode="valid")
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        # Pad to match close length
        rsi_full = np.full(len(close_arr), np.nan)
        rsi_full[14:] = rsi

        # MACD
        def _ema(arr: np.ndarray, span: int) -> np.ndarray:
            k = 2.0 / (span + 1)
            out = np.empty_like(arr)
            out[0] = arr[0]
            for i in range(1, len(arr)):
                out[i] = arr[i] * k + out[i - 1] * (1 - k)
            return out

        ema12 = _ema(close_arr, 12)
        ema26 = _ema(close_arr, 26)
        macd_line = ema12 - ema26
        signal_line = _ema(macd_line, 9)

        adx = self.get_adx_strength(high_arr, low_arr, close_arr)
        macd_slope = self.calculate_macd_slope(macd_line, signal_line)
        divergence = self.calculate_rsi_divergence(close_arr, rsi_full)
        exhaustion = self.detect_exhaustion(close_arr, vol_arr)

        # ADX normalised contribution: 25 = threshold, 50 = strong
        adx_norm = min(adx / 50.0, 1.0)

        # Combine: ADX 40%, MACD slope 40%, penalise for exhaustion/divergence
        base_score = 0.40 * adx_norm + 0.40 * (abs(macd_slope))
        penalty = 0.15 * float(divergence) + 0.10 * float(exhaustion)
        score = max(0.0, min(1.0, base_score - penalty + 0.20))  # +0.20 base floor

        # Direction from MACD slope sign
        if macd_slope > 0.05:
            direction = 1
        elif macd_slope < -0.05:
            direction = -1
        else:
            direction = 0

        return MomentumResult(
            score=round(score, 4),
            adx_strength=round(adx, 2),
            macd_slope=macd_slope,
            divergence_detected=divergence,
            exhaustion_detected=exhaustion,
            direction=direction,
        )

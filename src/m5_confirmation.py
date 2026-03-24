from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

_RSI_PERIOD = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_MIN_BARS = 30


@dataclass
class M5ConfirmationResult:
    """Result from the M5 confirmation system.

    Attributes:
        confirmed: True when all required checks pass.
        score: Composite confirmation score in [0, 1].
        reason: Human-readable explanation of the decision.
        momentum_ok: Momentum alignment result.
        pattern_ok: Candle pattern check result.
        volume_ok: Volume confirmation result.
    """

    confirmed: bool
    score: float
    reason: str
    momentum_ok: bool = False
    pattern_ok: bool = False
    volume_ok: bool = False


class M5Confirmation:
    """M5 timeframe confirmation layer for H1 trade signals.

    Validates that the 5-minute chart supports an H1 directional bias before
    trade entry is authorised.  Three independent checks are combined into a
    weighted confirmation score:
      - Momentum alignment (RSI + MACD):  50% weight
      - Candle pattern:                   30% weight
      - Volume confirmation:              20% weight

    Args:
        score_threshold: Minimum composite score to confirm entry (default 0.55).
        require_all: When True, ALL sub-checks must pass (default False).
    """

    def __init__(
        self,
        score_threshold: float = 0.55,
        require_all: bool = False,
    ) -> None:
        self.score_threshold = score_threshold
        self.require_all = require_all

    # ------------------------------------------------------------------
    # Internal indicator helpers
    # ------------------------------------------------------------------

    def _compute_rsi(self, closes: pl.Series, period: int = _RSI_PERIOD) -> pl.Series:
        """Compute RSI using Wilder smoothing approximation.

        Args:
            closes: Close price series.
            period: Lookback period.

        Returns:
            RSI series (same length as closes; NaN for initial bars).
        """
        delta = closes.diff()
        gain = delta.clip(lower_bound=0)
        loss = (-delta).clip(lower_bound=0)

        avg_gain = gain.rolling_mean(window_size=period)
        avg_loss = loss.rolling_mean(window_size=period)

        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _compute_macd(
        self,
        closes: pl.Series,
        fast: int = _MACD_FAST,
        slow: int = _MACD_SLOW,
        signal: int = _MACD_SIGNAL,
    ) -> tuple[pl.Series, pl.Series]:
        """Compute MACD line and signal line.

        Args:
            closes: Close price series.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal EMA period.

        Returns:
            Tuple of (macd_line, signal_line) polars Series.
        """
        ema_fast = closes.ewm_mean(span=fast)
        ema_slow = closes.ewm_mean(span=slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm_mean(span=signal)
        return macd_line, signal_line

    # ------------------------------------------------------------------
    # Public checks
    # ------------------------------------------------------------------

    def check_momentum_alignment(
        self,
        m5_df: pl.DataFrame,
        direction: int,
    ) -> bool:
        """Check RSI and MACD are aligned with the intended direction.

        Args:
            m5_df: M5 OHLCV DataFrame with 'close' column.
            direction: +1 for long, -1 for short.

        Returns:
            True when both RSI and MACD support the direction.
        """
        if len(m5_df) < _MIN_BARS:
            logger.warning("Insufficient M5 bars for momentum check (%d)", len(m5_df))
            return False

        closes = m5_df["close"]
        rsi = self._compute_rsi(closes)
        macd_line, signal_line = self._compute_macd(closes)

        last_rsi = rsi[-1]
        last_macd = macd_line[-1]
        last_signal = signal_line[-1]

        if last_rsi is None or last_macd is None or last_signal is None:
            return False

        if direction == 1:
            rsi_ok = 40.0 < float(last_rsi) < 75.0   # not overbought, above neutral
            macd_ok = float(last_macd) > float(last_signal)  # bullish crossover
        else:
            rsi_ok = 25.0 < float(last_rsi) < 60.0   # not oversold, below neutral
            macd_ok = float(last_macd) < float(last_signal)  # bearish crossover

        result = rsi_ok and macd_ok
        logger.debug(
            "Momentum check dir=%d rsi=%.1f macd=%.4f signal=%.4f → %s",
            direction, float(last_rsi), float(last_macd), float(last_signal), result,
        )
        return result

    def check_candle_pattern(
        self,
        m5_df: pl.DataFrame,
        direction: int,
    ) -> bool:
        """Detect bullish/bearish engulfing or strong directional candle.

        Args:
            m5_df: M5 OHLCV DataFrame with open/high/low/close columns.
            direction: +1 for long, -1 for short.

        Returns:
            True when a confirming candle pattern is detected.
        """
        if len(m5_df) < 2:
            return False

        rows = m5_df.tail(2)
        prev_open = float(rows["open"][0])
        prev_close = float(rows["close"][0])
        curr_open = float(rows["open"][1])
        curr_close = float(rows["close"][1])

        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)

        # Engulfing pattern
        if direction == 1:
            bullish_engulf = (
                prev_close < prev_open         # prev candle bearish
                and curr_close > curr_open     # curr candle bullish
                and curr_close > prev_open     # curr body engulfs prev
                and curr_open < prev_close
            )
            strong_bull = curr_close > curr_open and curr_body > prev_body * 0.8
            result = bullish_engulf or strong_bull
        else:
            bearish_engulf = (
                prev_close > prev_open
                and curr_close < curr_open
                and curr_close < prev_open
                and curr_open > prev_close
            )
            strong_bear = curr_close < curr_open and curr_body > prev_body * 0.8
            result = bearish_engulf or strong_bear

        logger.debug("Candle pattern check dir=%d → %s", direction, result)
        return result

    def check_volume_confirmation(self, m5_df: pl.DataFrame) -> bool:
        """Check that volume is increasing on the signal bar vs recent average.

        Args:
            m5_df: M5 OHLCV DataFrame with 'volume' (or 'tick_volume') column.

        Returns:
            True when current volume exceeds recent average.
        """
        vol_col = "volume" if "volume" in m5_df.columns else "tick_volume"
        if vol_col not in m5_df.columns or len(m5_df) < 5:
            return True  # volume data absent; pass by default

        recent = m5_df[vol_col].tail(10).head(9)
        current = float(m5_df[vol_col][-1])
        avg = float(recent.mean())

        result = current > avg * 1.0  # at least at average
        logger.debug("Volume check: current=%.0f avg=%.0f → %s", current, avg, result)
        return result

    def get_entry_score(
        self,
        m5_df: pl.DataFrame,
        direction: int,
    ) -> float:
        """Compute a weighted composite confirmation score.

        Weights: momentum 50%, candle pattern 30%, volume 20%.

        Args:
            m5_df: M5 OHLCV DataFrame.
            direction: +1 for long, -1 for short.

        Returns:
            Composite score in [0.0, 1.0].
        """
        momentum_ok = self.check_momentum_alignment(m5_df, direction)
        pattern_ok = self.check_candle_pattern(m5_df, direction)
        volume_ok = self.check_volume_confirmation(m5_df)

        score = (
            0.50 * float(momentum_ok)
            + 0.30 * float(pattern_ok)
            + 0.20 * float(volume_ok)
        )
        return round(score, 4)

    def confirm_entry(
        self,
        m5_df: pl.DataFrame,
        direction: int,
        h1_signal: float = 1.0,
    ) -> M5ConfirmationResult:
        """Determine if M5 data confirms the H1 trade signal.

        Args:
            m5_df: M5 OHLCV DataFrame with at least 30 rows.
            direction: +1 for long, -1 for short.
            h1_signal: H1 signal strength in [0, 1] (default 1.0).

        Returns:
            M5ConfirmationResult with score, decision and component flags.
        """
        momentum_ok = self.check_momentum_alignment(m5_df, direction)
        pattern_ok = self.check_candle_pattern(m5_df, direction)
        volume_ok = self.check_volume_confirmation(m5_df)

        base_score = (
            0.50 * float(momentum_ok)
            + 0.30 * float(pattern_ok)
            + 0.20 * float(volume_ok)
        )
        # Scale by H1 signal quality
        score = round(base_score * max(0.5, min(1.0, h1_signal)), 4)

        if self.require_all:
            confirmed = momentum_ok and pattern_ok and volume_ok
        else:
            confirmed = score >= self.score_threshold

        reasons = []
        if not momentum_ok:
            reasons.append("momentum misaligned")
        if not pattern_ok:
            reasons.append("no candle pattern")
        if not volume_ok:
            reasons.append("volume weak")
        reason = (
            "M5 confirms entry"
            if confirmed
            else "M5 rejects entry: " + ", ".join(reasons)
        )

        logger.debug("M5 confirm dir=%d score=%.3f confirmed=%s", direction, score, confirmed)
        return M5ConfirmationResult(
            confirmed=confirmed,
            score=score,
            reason=reason,
            momentum_ok=momentum_ok,
            pattern_ok=pattern_ok,
            volume_ok=volume_ok,
        )

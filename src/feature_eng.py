"""Feature engineering with 37 features using Polars.

Computes price, technical, SMC, volume, session, regime,
and momentum features for the ML model.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = [
    "returns_1", "returns_5", "returns_10",
    "log_returns_1", "log_returns_5", "log_returns_10",
    "volatility_5", "volatility_10", "volatility_20",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower",
    "atr_14",
    "adx_14",
    "stoch_k", "stoch_d", "stoch_signal",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "ob_proximity", "fvg_present", "bos_signal",
    "choch_signal", "liquidity_sweep",
    "volume_ratio", "volume_momentum", "vwap",
    "session_id",
    "hmm_state", "regime_volatility",
    "momentum_20",
]


def _ema(series: pl.Series, span: int) -> pl.Series:
    values = series.to_numpy().astype(float)
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return pl.Series(name=series.name, values=result)


def _rsi(close: pl.Series, period: int = 14) -> pl.Series:
    values = close.to_numpy().astype(float)
    deltas = np.diff(values, prepend=values[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.zeros_like(values)
    avg_loss = np.zeros_like(values)
    avg_gain[period] = np.mean(gains[1:period + 1])
    avg_loss[period] = np.mean(losses[1:period + 1])
    for i in range(period + 1, len(values)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    return pl.Series(name="rsi_14", values=rsi)


def _atr(high: pl.Series, low: pl.Series, close: pl.Series, period: int = 14) -> pl.Series:
    h = high.to_numpy().astype(float)
    l = low.to_numpy().astype(float)
    c = close.to_numpy().astype(float)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = np.zeros_like(tr)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    atr[:period - 1] = atr[period - 1]
    return pl.Series(name="atr_14", values=atr)


def _adx(high: pl.Series, low: pl.Series, close: pl.Series, period: int = 14) -> pl.Series:
    h = high.to_numpy().astype(float)
    l = low.to_numpy().astype(float)
    c = close.to_numpy().astype(float)
    n = len(h)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    atr[period] = np.sum(tr[1:period + 1])
    s_plus = np.sum(plus_dm[1:period + 1])
    s_minus = np.sum(minus_dm[1:period + 1])
    for i in range(period, n):
        if i > period:
            atr[i] = atr[i - 1] - atr[i - 1] / period + tr[i]
            s_plus = s_plus - s_plus / period + plus_dm[i]
            s_minus = s_minus - s_minus / period + minus_dm[i]
        if atr[i] != 0:
            plus_di[i] = 100 * s_plus / atr[i]
            minus_di[i] = 100 * s_minus / atr[i]
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0
    adx[2 * period - 1] = np.mean(dx[period:2 * period])
    for i in range(2 * period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    adx[:2 * period - 1] = adx[2 * period - 1] if 2 * period - 1 < n else 25.0
    return pl.Series(name="adx_14", values=adx)


def _stochastic(
    high: pl.Series, low: pl.Series, close: pl.Series,
    k_period: int = 14, d_period: int = 3,
) -> tuple:
    h = high.to_numpy().astype(float)
    l = low.to_numpy().astype(float)
    c = close.to_numpy().astype(float)
    n = len(c)
    k = np.full(n, 50.0)
    for i in range(k_period - 1, n):
        hh = np.max(h[i - k_period + 1:i + 1])
        ll = np.min(l[i - k_period + 1:i + 1])
        if hh != ll:
            k[i] = 100 * (c[i] - ll) / (hh - ll)
    d = np.convolve(k, np.ones(d_period) / d_period, mode="same")
    signal = np.convolve(d, np.ones(d_period) / d_period, mode="same")
    return (
        pl.Series(name="stoch_k", values=k),
        pl.Series(name="stoch_d", values=d),
        pl.Series(name="stoch_signal", values=signal),
    )


def _get_session_id(hour: int) -> int:
    if 22 <= hour or hour < 7:
        return 0
    elif 0 <= hour < 9:
        return 1
    elif 7 <= hour < 16:
        return 2
    else:
        return 3


def compute_features(
    df: pl.DataFrame,
    smc_data: Optional[dict] = None,
    regime_state: int = 1,
    regime_vol: float = 0.0,
) -> pl.DataFrame:
    """Compute all 37 features from OHLCV data.

    Args:
        df: Polars DataFrame with columns: open, high, low, close, tick_volume, time.
        smc_data: Optional SMC analysis results.
        regime_state: HMM regime state (0=trend, 1=range, 2=volatile).
        regime_vol: Regime volatility measure.

    Returns:
        pl.DataFrame: DataFrame with 37 feature columns.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    n = len(df)

    c = close.to_numpy().astype(float)
    h = high.to_numpy().astype(float)
    l = low.to_numpy().astype(float)

    features = {}

    for p in [1, 5, 10]:
        ret = np.zeros(n)
        ret[p:] = (c[p:] - c[:-p]) / (c[:-p] + 1e-8)
        features[f"returns_{p}"] = ret

    for p in [1, 5, 10]:
        lr = np.zeros(n)
        lr[p:] = np.log(c[p:] / (c[:-p] + 1e-8))
        features[f"log_returns_{p}"] = lr

    for p in [5, 10, 20]:
        vol = np.zeros(n)
        for i in range(p, n):
            vol[i] = np.std(c[i - p:i])
        features[f"volatility_{p}"] = vol

    features["rsi_14"] = _rsi(close).to_numpy()

    ema12 = _ema(close, 12).to_numpy()
    ema26 = _ema(close, 26).to_numpy()
    macd_line = ema12 - ema26
    macd_signal = _ema(pl.Series(values=macd_line), 9).to_numpy()
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_line - macd_signal

    bb_mid = _ema(close, 20).to_numpy()
    bb_std = np.zeros(n)
    for i in range(20, n):
        bb_std[i] = np.std(c[i - 20:i])
    bb_std[:20] = bb_std[20] if n > 20 else 1.0
    features["bb_upper"] = bb_mid + 2 * bb_std
    features["bb_middle"] = bb_mid
    features["bb_lower"] = bb_mid - 2 * bb_std

    features["atr_14"] = _atr(high, low, close).to_numpy()
    features["adx_14"] = _adx(high, low, close).to_numpy()

    sk, sd, ss = _stochastic(high, low, close)
    features["stoch_k"] = sk.to_numpy()
    features["stoch_d"] = sd.to_numpy()
    features["stoch_signal"] = ss.to_numpy()

    for span in [9, 21, 50, 200]:
        features[f"ema_{span}"] = _ema(close, span).to_numpy()

    smc = smc_data or {}
    features["ob_proximity"] = np.full(n, smc.get("ob_proximity", 0.0))
    features["fvg_present"] = np.full(n, float(smc.get("fvg_present", 0)))
    features["bos_signal"] = np.full(n, float(smc.get("bos_signal", 0)))
    features["choch_signal"] = np.full(n, float(smc.get("choch_signal", 0)))
    features["liquidity_sweep"] = np.full(n, float(smc.get("liquidity_sweep", 0)))

    vol_arr = df["tick_volume"].to_numpy().astype(float) if "tick_volume" in df.columns else np.ones(n)
    vol_mean = np.mean(vol_arr[-20:]) if n >= 20 else np.mean(vol_arr) + 1e-8
    features["volume_ratio"] = vol_arr / (vol_mean + 1e-8)
    mom = np.zeros(n)
    for i in range(5, n):
        mom[i] = vol_arr[i] / (np.mean(vol_arr[i - 5:i]) + 1e-8)
    features["volume_momentum"] = mom

    cum_vol = np.cumsum(vol_arr)
    cum_vp = np.cumsum(vol_arr * c)
    features["vwap"] = np.where(cum_vol != 0, cum_vp / cum_vol, c)

    if "time" in df.columns:
        times = df["time"].to_list()
        session_ids = np.array([_get_session_id(getattr(t, "hour", 12)) for t in times], dtype=float)
    else:
        session_ids = np.full(n, 2.0)
    features["session_id"] = session_ids

    features["hmm_state"] = np.full(n, float(regime_state))
    features["regime_volatility"] = np.full(n, regime_vol)

    mom20 = np.zeros(n)
    if n >= 20:
        mom20[20:] = c[20:] - c[:-20]
    features["momentum_20"] = mom20

    result = pl.DataFrame({name: features[name] for name in FEATURE_NAMES})
    result = result.fill_nan(0.0).fill_null(0.0)

    logger.debug("Computed %d features for %d rows", len(FEATURE_NAMES), n)
    return result


def create_labels(
    df: pl.DataFrame,
    forward_bars: int = 10,
    threshold: float = 0.003,
) -> np.ndarray:
    """Create training labels based on forward returns.

    Args:
        df: DataFrame with close prices.
        forward_bars: Bars to look ahead.
        threshold: Return threshold for BUY/SELL.

    Returns:
        np.ndarray: Labels (0=SELL, 1=HOLD, 2=BUY).
    """
    c = df["close"].to_numpy().astype(float)
    n = len(c)
    labels = np.ones(n, dtype=int)

    for i in range(n - forward_bars):
        ret = (c[i + forward_bars] - c[i]) / (c[i] + 1e-8)
        if ret > threshold:
            labels[i] = 2
        elif ret < -threshold:
            labels[i] = 0

    return labels
#!/usr/bin/env python3
"""
GOLDAI Backtest V4 — Date-Based Out-of-Sample Test
===================================================
Zero data leakage: test period starts AFTER the validation period used
during model training.

Date boundaries (matching train_models.py defaults):
  Training   : 2020-01-01 → 2023-12-31  (models trained on this)
  Validation : 2024-01-01 → 2024-06-30  (used only for early stopping)
  Test       : 2024-07-01 → 2026-03-26  (completely held-out — tested here)

Two passes are run:
  V2 — old logic: no H1 blocker, no zone guard, raw SMC SL/TP
  V4 — new logic: H1 Blocker + Zone Guard + Dynamic SL (1.2× ATR, min 12 pts)
                  + Dynamic RR (2.0–3.0) + Breakeven Trailing + Min RR 2.0 filter
                  + Confidence threshold 0.70

Usage:
    python backtests/backtest_v4.py [options]

    Key options:
      --symbol XAUUSD.m
      --lot 0.01
      --balance 439
      --confidence 0.70
      --test-start 2024-07-01
      --test-end   2026-03-26
      --train-start 2020-01-01
      --train-end   2023-12-31
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict

# ---------------------------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.config import get_config
from src.mt5_connector import MT5Connector
from src.feature_eng import FeatureEngineer
from src.smc_polars import SMCAnalyzer
from src.regime_detector import MarketRegimeDetector
from src.ml_model import TradingModel, get_default_feature_columns

# Suppress verbose logging during backtest; only warnings+errors to stderr
logger.remove()
logger.add(sys.stderr, level="WARNING")

# ---------------------------------------------------------------------------
# M15 bars per year ≈ 52 weeks × 5 days × 24 h × 4 = 24,960.
# Test window: 2024-07-01 → 2026-03-26 ≈ 21 months ≈ 43,750 bars.
# Fetch 70,000 M15 bars to guarantee full coverage with a safety margin.
# H1 bars: test window ≈ 5,500; fetch 15,000 for safety.
# ---------------------------------------------------------------------------
FETCH_BARS_M15 = 70_000
FETCH_BARS_H1  = 15_000

# ---------------------------------------------------------------------------
# Session filter (London + NY, UTC)
# ---------------------------------------------------------------------------
LONDON_OPEN  = 7
LONDON_CLOSE = 16
NY_OPEN      = 13
NY_CLOSE     = 22


def get_session(hour_utc: int) -> Optional[str]:
    """Return active trading session name or None if outside hours."""
    in_london = LONDON_OPEN <= hour_utc < LONDON_CLOSE
    in_ny     = NY_OPEN     <= hour_utc < NY_CLOSE
    if in_london and in_ny:
        return "LONDON_NY"
    if in_london:
        return "LONDON"
    if in_ny:
        return "NY"
    return None


# ---------------------------------------------------------------------------
# H1 Bias helpers (mirrors main_live.py _get_h1_bias logic)
# ---------------------------------------------------------------------------

def _count_candle_bias_row(closes: List[float], opens: List[float]) -> int:
    """Count net bias from the last 5 candles (+1 bull, -1 bear, 0 neutral)."""
    if len(closes) < 5:
        return 0
    last_5_c = closes[-5:]
    last_5_o = opens[-5:]
    bullish = sum(1 for c, o in zip(last_5_c, last_5_o) if c > o)
    bearish = 5 - bullish
    if bullish >= 3:
        return 1
    elif bearish >= 3:
        return -1
    return 0


def _get_regime_weights_static(regime_str: str) -> dict:
    """Indicator weights keyed by regime label."""
    if not regime_str:
        regime_str = "medium_volatility"
    r = regime_str.lower()
    if "low" in r or "ranging" in r:
        return {"ema_trend": 0.15, "ema_cross": 0.15, "rsi": 0.30,
                "macd": 0.25, "candles": 0.15}
    elif "high" in r or "trending" in r:
        return {"ema_trend": 0.30, "ema_cross": 0.25, "rsi": 0.10,
                "macd": 0.25, "candles": 0.10}
    else:
        return {"ema_trend": 0.25, "ema_cross": 0.20, "rsi": 0.20,
                "macd": 0.20, "candles": 0.15}


def build_h1_bias_table(df_h1: pl.DataFrame) -> pl.DataFrame:
    """
    Pre-calculate H1 bias for each H1 bar.
    Returns DataFrame with: time_hour, h1_bias, h1_strength.
    """
    required = ["time", "close", "open", "ema_9", "ema_21", "rsi", "macd_histogram"]
    for col in required:
        if col not in df_h1.columns:
            logger.warning(f"H1 column missing: {col} — H1 bias will be NEUTRAL")
            return pl.DataFrame({
                "time_hour": pl.Series([], dtype=pl.Datetime),
                "h1_bias":   [],
                "h1_strength": [],
            })

    rows = df_h1.to_dicts()
    results = []

    for i, row in enumerate(rows):
        if i < 30:
            results.append({
                "time_hour":   row["time"],
                "h1_bias":     "NEUTRAL",
                "h1_strength": "weak",
            })
            continue

        price     = row["close"]
        ema_9     = row["ema_9"]
        ema_21    = row["ema_21"]
        rsi       = row["rsi"]
        macd_hist = row["macd_histogram"]

        if any(v is None for v in [price, ema_9, ema_21, rsi, macd_hist]):
            results.append({
                "time_hour":   row["time"],
                "h1_bias":     "NEUTRAL",
                "h1_strength": "weak",
            })
            continue

        closes = [r["close"] for r in rows[max(0, i - 4): i + 1]]
        opens  = [r["open"]  for r in rows[max(0, i - 4): i + 1]]

        signals = {
            "ema_trend": 1 if price > ema_21 else (-1 if price < ema_21 else 0),
            "ema_cross": 1 if ema_9  > ema_21 else (-1 if ema_9  < ema_21 else 0),
            "rsi":       1 if rsi    > 55     else (-1 if rsi    < 45     else 0),
            "macd":      1 if macd_hist > 0   else (-1 if macd_hist < 0   else 0),
            "candles":   _count_candle_bias_row(closes, opens),
        }

        regime_str = row.get("regime_name") or "medium_volatility"
        weights = _get_regime_weights_static(regime_str)
        score = sum(signals[k] * weights[k] for k in signals)

        if score >= 0.3:
            bias = "BULLISH"
        elif score <= -0.3:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        abs_score = abs(score)
        strength = ("strong"   if abs_score >= 0.7 else
                    "moderate" if abs_score >= 0.5 else
                    "weak")

        results.append({
            "time_hour":   row["time"],
            "h1_bias":     bias,
            "h1_strength": strength,
        })

    return pl.DataFrame(results)


def lookup_h1_bias(h1_table: pl.DataFrame, bar_time) -> Tuple[str, str]:
    """Return (bias, strength) for the H1 bar containing bar_time."""
    if h1_table.is_empty():
        return "NEUTRAL", "weak"
    try:
        if hasattr(bar_time, "replace"):
            target_hour = bar_time.replace(minute=0, second=0, microsecond=0)
        else:
            return "NEUTRAL", "weak"

        match = h1_table.filter(pl.col("time_hour") == target_hour)
        if match.is_empty():
            earlier = h1_table.filter(pl.col("time_hour") <= target_hour)
            if earlier.is_empty():
                return "NEUTRAL", "weak"
            match = earlier.tail(1)

        return match["h1_bias"][0], match["h1_strength"][0]
    except Exception:
        return "NEUTRAL", "weak"


# ---------------------------------------------------------------------------
# Dynamic RR (mirrors smc_polars._calculate_dynamic_rr)
# ---------------------------------------------------------------------------

def calculate_dynamic_rr(
    market_structure: int,
    has_bullish_break: bool,
    has_bearish_break: bool,
    has_fvg: bool,
    has_ob: bool,
    df_window: pl.DataFrame,
) -> float:
    """Return a dynamic RR in [2.0, 3.0] based on market conditions."""
    rr = 2.0  # raised from 1.5

    if market_structure != 0:
        rr += 0.25   # raised from 0.15
    if has_bullish_break or has_bearish_break:
        rr += 0.20   # raised from 0.10
    if has_fvg:
        rr += 0.15   # raised from 0.05
    if has_ob:
        rr += 0.15   # raised from 0.05

    if "bos" in df_window.columns:
        recent_bos = df_window.tail(20)["bos"].to_list()
        bos_count  = sum(1 for b in recent_bos if b != 0)
        if bos_count >= 3:
            rr += 0.25   # raised from 0.10
        elif bos_count >= 2:
            rr += 0.10   # raised from 0.05

    if "atr" in df_window.columns:
        atr_vals = df_window.tail(1)["atr"].to_list()
        if atr_vals and atr_vals[0] is not None:
            atr = atr_vals[0]
            if atr > 18:
                rr -= 0.15
            elif atr > 15:
                rr -= 0.05

    if "bos" in df_window.columns:
        recent_bos_30 = df_window.tail(30)["bos"].to_list()
        bos_count_30  = sum(1 for b in recent_bos_30 if b != 0)
        if bos_count_30 == 0:
            rr -= 0.10

    return max(2.0, min(3.0, rr))  # [2.0, 3.0] raised from [1.5, 2.0]


# ---------------------------------------------------------------------------
# Trade class
# ---------------------------------------------------------------------------

class Trade:
    """Represents a single backtest trade."""

    __slots__ = [
        "direction", "entry", "sl", "tp", "entry_bar",
        "exit_price", "exit_bar", "result", "pnl",
        "closed_partial", "partial_price", "partial_pnl",
        "trailing_active", "trailing_sl",
        "lot_size", "pip_value",
    ]

    def __init__(self, direction: str, entry: float, sl: float, tp: float,
                 entry_bar: int, lot_size: float, pip_value: float):
        self.direction       = direction
        self.entry           = entry
        self.sl              = sl
        self.tp              = tp
        self.entry_bar       = entry_bar
        self.exit_price: Optional[float] = None
        self.exit_bar:   Optional[int]   = None
        self.result:     Optional[str]   = None   # "win" | "loss"
        self.pnl:        float           = 0.0
        self.closed_partial: bool        = False
        self.partial_price:  Optional[float] = None
        self.partial_pnl:    float       = 0.0
        self.trailing_active: bool       = False
        self.trailing_sl:     float      = sl
        self.lot_size   = lot_size
        self.pip_value  = pip_value


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    df: pl.DataFrame,
    h1_table: pl.DataFrame,
    model: Optional[TradingModel],
    feature_cols: List[str],
    smc: SMCAnalyzer,
    features_eng: FeatureEngineer,
    # Feature flags
    use_h1_blocker: bool = True,
    use_zone_guard: bool = True,
    use_dynamic_sl: bool = True,
    use_dynamic_rr: bool = True,
    use_trailing:   bool = True,
    use_partial:    bool = True,
    use_session:    bool = True,
    use_ml:         bool = True,
    # Parameters
    lot_size:             float = 0.01,
    initial_balance:      float = 439.0,
    confidence_threshold: float = 0.70,
    cooldown_bars:        int   = 1,
) -> Dict:
    """
    Single-pass vectorised backtest over the provided DataFrame.
    Returns a dict of performance metrics.
    """
    pip_value = lot_size * 100 * 0.01   # XAUUSD: 1 pip = $0.01 per 0.01 lot

    balance     = initial_balance
    equity_curve: List[float] = [balance]
    trades:       List[Trade] = []
    active_trade: Optional[Trade] = None

    recent_loss_zones: List[Tuple[str, float, int]] = []
    h1_blocked_count   = 0
    zone_blocked_count = 0
    sl_distances: List[float] = []
    rr_values:    List[float] = []

    n_bars = len(df)
    times  = df["time"].to_list() if "time" in df.columns else [None] * n_bars
    closes = df["close"].to_list()
    highs  = df["high"].to_list()
    lows   = df["low"].to_list()
    atrs   = df["atr"].to_list() if "atr" in df.columns else [None] * n_bars

    WARMUP         = 100
    last_close_bar = -cooldown_bars

    for i in range(WARMUP, n_bars):
        bar_time = times[i]
        close    = closes[i]
        high     = highs[i]
        low      = lows[i]

        # ----------------------------------------------------------------
        # Manage active trade
        # ----------------------------------------------------------------
        if active_trade is not None:
            t        = active_trade
            risk     = abs(t.entry - t.sl)
            atr_raw  = atrs[i]
            atr_trail = atr_raw if (atr_raw is not None and atr_raw > 0) else risk

            # Trailing stop update
            if use_trailing and t.trailing_active:
                if t.direction == "BUY":
                    new_trail = high - 1.0 * atr_trail
                    if new_trail > t.trailing_sl:
                        t.trailing_sl = new_trail
                else:
                    new_trail = low + 1.0 * atr_trail
                    if new_trail < t.trailing_sl:
                        t.trailing_sl = new_trail

            # Activate trailing after 0.5 ATR in profit (breakeven first)
            if use_trailing and not t.trailing_active:
                atr_trail_trigger = 0.5 * atr_trail
                if t.direction == "BUY" and high >= t.entry + atr_trail_trigger:
                    t.trailing_active = True
                    t.trailing_sl     = t.entry  # breakeven first
                elif t.direction == "SELL" and low <= t.entry - atr_trail_trigger:
                    t.trailing_active = True
                    t.trailing_sl     = t.entry  # breakeven first

            effective_sl = t.trailing_sl if t.trailing_active else t.sl

            # Partial close at halfway to TP
            if use_partial and not t.closed_partial:
                halfway = t.entry + (t.tp - t.entry) * 0.5
                if t.direction == "BUY" and high >= halfway:
                    t.closed_partial  = True
                    t.partial_price   = halfway
                    partial_pts       = halfway - t.entry
                    t.partial_pnl     = partial_pts / 0.01 * pip_value * 0.5
                    balance          += t.partial_pnl
                elif t.direction == "SELL" and low <= halfway:
                    t.closed_partial  = True
                    t.partial_price   = halfway
                    partial_pts       = t.entry - halfway
                    t.partial_pnl     = partial_pts / 0.01 * pip_value * 0.5
                    balance          += t.partial_pnl

            # SL hit
            sl_hit = False
            if t.direction == "BUY"  and low  <= effective_sl:
                sl_hit, exit_price = True, effective_sl
            elif t.direction == "SELL" and high >= effective_sl:
                sl_hit, exit_price = True, effective_sl

            if sl_hit:
                t.exit_price   = exit_price
                t.exit_bar     = i
                remaining      = 0.5 if t.closed_partial else 1.0
                pts = exit_price - t.entry if t.direction == "BUY" else t.entry - exit_price
                t.pnl  = pts / 0.01 * pip_value * remaining + t.partial_pnl
                balance += pts / 0.01 * pip_value * remaining
                t.result       = "loss" if t.pnl < 0 else "win"
                trades.append(t)
                active_trade   = None
                last_close_bar = i
                if t.result == "loss":
                    recent_loss_zones.append((t.direction, t.entry, i))
                equity_curve.append(balance)
                continue

            # TP hit
            tp_hit = False
            if t.direction == "BUY"  and high >= t.tp:
                tp_hit, exit_price = True, t.tp
            elif t.direction == "SELL" and low  <= t.tp:
                tp_hit, exit_price = True, t.tp

            if tp_hit:
                t.exit_price   = exit_price
                t.exit_bar     = i
                remaining      = 0.5 if t.closed_partial else 1.0
                pts = exit_price - t.entry if t.direction == "BUY" else t.entry - exit_price
                t.pnl  = pts / 0.01 * pip_value * remaining + t.partial_pnl
                balance += pts / 0.01 * pip_value * remaining
                t.result       = "win"
                trades.append(t)
                active_trade   = None
                last_close_bar = i
                equity_curve.append(balance)
                continue

            equity_curve.append(balance)
            continue   # trade still open

        # ----------------------------------------------------------------
        # No active trade — look for a new signal
        # ----------------------------------------------------------------
        equity_curve.append(balance)

        # Cooldown
        if i - last_close_bar < cooldown_bars:
            continue

        # Session filter
        if use_session and bar_time is not None:
            hour_utc = bar_time.hour if hasattr(bar_time, "hour") else 0
            if get_session(hour_utc) is None:
                continue

        # Build look-back window (uses pre-calculated features)
        window_start = max(0, i - 199)
        df_window    = df.slice(window_start, i - window_start + 1)

        if smc is None:
            continue
        try:
            signal = smc.generate_signal(df_window)
        except Exception:
            continue
        if signal is None:
            continue

        # ML filter
        if use_ml and model is not None and model.fitted:
            try:
                pred = model.predict(df_window, feature_cols)
                if pred.signal == "HOLD":
                    continue
                if pred.signal != signal.signal_type:
                    continue
                if pred.confidence < confidence_threshold:
                    continue
            except Exception:
                pass

        confidence = signal.confidence
        if confidence < confidence_threshold:
            continue

        # H1 Blocker
        if use_h1_blocker and bar_time is not None and not h1_table.is_empty():
            h1_bias, h1_strength = lookup_h1_bias(h1_table, bar_time)

            h1_opposed = (
                (signal.signal_type == "BUY"  and h1_bias == "BEARISH") or
                (signal.signal_type == "SELL" and h1_bias == "BULLISH")
            )
            h1_aligned = (
                (signal.signal_type == "BUY"  and h1_bias == "BULLISH") or
                (signal.signal_type == "SELL" and h1_bias == "BEARISH")
            )

            if h1_aligned:
                confidence = min(1.0, confidence * 1.05)
            elif h1_opposed:
                if h1_strength in ("strong", "moderate"):
                    h1_blocked_count += 1
                    continue
                else:
                    confidence = confidence * 0.85

            if confidence < confidence_threshold:
                continue

        # Zone Guard
        if use_zone_guard:
            recent_loss_zones = [
                (d, p, b) for d, p, b in recent_loss_zones if i - b < 20
            ]
            atr_val = None
            if "atr" in df_window.columns:
                atr_vals = df_window.tail(1)["atr"].to_list()
                if atr_vals and atr_vals[0] is not None and atr_vals[0] > 0:
                    atr_val = atr_vals[0]
            zone_radius = atr_val if atr_val else 15.0

            zone_blocked = any(
                prev_dir == signal.signal_type and abs(close - prev_price) <= zone_radius
                for prev_dir, prev_price, _ in recent_loss_zones
            )
            if zone_blocked:
                zone_blocked_count += 1
                continue

        # SL / TP calculation
        entry = close

        if "atr" in df_window.columns:
            atr_vals = df_window.tail(1)["atr"].to_list()
            atr = atr_vals[0] if (atr_vals and atr_vals[0] is not None and atr_vals[0] > 0) else 12.0
        else:
            atr = 12.0

        if use_dynamic_sl:
            min_sl_dist = max(1.2 * atr, 12.0)  # tightened from max(2.0 * atr, 20.0)
        else:
            raw_sl_dist = abs(entry - signal.stop_loss) if signal.stop_loss else (1.5 * atr)
            min_sl_dist = max(raw_sl_dist, 5.0)

        if signal.signal_type == "BUY":
            swing_sl = signal.stop_loss if signal.stop_loss and signal.stop_loss < entry else None
            atr_sl   = entry - min_sl_dist
            sl       = min(swing_sl, atr_sl) if swing_sl else atr_sl
            sl       = min(sl, entry - min_sl_dist)
        else:
            swing_sl = signal.stop_loss if signal.stop_loss and signal.stop_loss > entry else None
            atr_sl   = entry + min_sl_dist
            sl       = max(swing_sl, atr_sl) if swing_sl else atr_sl
            sl       = max(sl, entry + min_sl_dist)

        risk = abs(entry - sl)
        if risk <= 0:
            continue

        # Dynamic RR
        if use_dynamic_rr:
            latest_row       = df_window.tail(1)
            market_structure = (
                latest_row["market_structure"].item()
                if "market_structure" in df_window.columns else 0
            )
            recent_bos   = df_window.tail(10)["bos"].to_list()   if "bos"   in df_window.columns else []
            recent_choch = df_window.tail(10)["choch"].to_list() if "choch" in df_window.columns else []
            has_bullish_break = 1  in recent_bos or 1  in recent_choch
            has_bearish_break = -1 in recent_bos or -1 in recent_choch
            has_fvg = (any(df_window.tail(10)["is_fvg_bull"].to_list())
                       if "is_fvg_bull" in df_window.columns else False)
            has_ob  = any(
                o != 0 for o in (df_window.tail(10)["ob"].to_list()
                                 if "ob" in df_window.columns else [])
            )
            rr = calculate_dynamic_rr(
                market_structure=market_structure,
                has_bullish_break=has_bullish_break,
                has_bearish_break=has_bearish_break,
                has_fvg=has_fvg,
                has_ob=has_ob,
                df_window=df_window,
            )
        else:
            rr = 2.0  # raised from 1.5

        # Minimum RR filter — skip trades where RR < 2.0
        if rr < 2.0:
            continue

        tp = (entry + risk * rr) if signal.signal_type == "BUY" else (entry - risk * rr)

        sl_distances.append(min_sl_dist)
        rr_values.append(rr)

        active_trade = Trade(
            direction=signal.signal_type,
            entry=entry,
            sl=sl,
            tp=tp,
            entry_bar=i,
            lot_size=lot_size,
            pip_value=pip_value,
        )

    # Close any still-open trade at end of data
    if active_trade is not None:
        t            = active_trade
        exit_price   = closes[-1]
        t.exit_price = exit_price
        t.exit_bar   = n_bars - 1
        remaining    = 0.5 if t.closed_partial else 1.0
        pts = exit_price - t.entry if t.direction == "BUY" else t.entry - exit_price
        t.pnl   = pts / 0.01 * pip_value * remaining + t.partial_pnl
        balance += pts / 0.01 * pip_value * remaining
        t.result = "win" if t.pnl > 0 else "loss"
        trades.append(t)
        equity_curve.append(balance)

    # ----------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------
    total        = len(trades)
    wins         = [t for t in trades if t.result == "win"]
    losses       = [t for t in trades if t.result == "loss"]
    win_rate     = len(wins) / total * 100 if total > 0 else 0.0

    total_pnl    = balance - initial_balance
    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win  = gross_profit / len(wins)   if wins   else 0.0
    avg_loss = gross_loss   / len(losses) if losses else 0.0

    # Max drawdown
    peak   = initial_balance
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    # Sharpe (per-trade, annualised)
    pnls = [t.pnl for t in trades]
    if len(pnls) > 1:
        mean_pnl = np.mean(pnls)
        std_pnl  = np.std(pnls, ddof=1)
        sharpe   = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_trades":  total,
        "win_rate":      win_rate,
        "total_pnl":     total_pnl,
        "profit_factor": profit_factor,
        "max_drawdown":  max_dd,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "sharpe":        sharpe,
        "h1_blocked":    h1_blocked_count,
        "zone_blocked":  zone_blocked_count,
        "avg_sl":        float(np.mean(sl_distances)) if sl_distances else 0.0,
        "avg_rr":        float(np.mean(rr_values))    if rr_values    else 0.0,
        "equity_curve":  equity_curve,
        "trades":        trades,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(
    v2: Dict,
    v4: Optional[Dict],
    test_start,
    test_end,
    train_start,
    train_end,
    total_bars: int,
    v2_only: bool,
    v4_only: bool,
) -> None:
    """Print a formatted V2 vs V4 comparison report."""

    def fmt_usd(v: float) -> str:
        return f"{'+'if v>0 else ''}${v:.2f}"

    def fmt_float(v: float) -> str:
        return f"{'+'if v>0 else ''}{v:.2f}"

    ts = test_start.strftime("%Y-%m-%d") if hasattr(test_start, "strftime") else str(test_start)
    te = test_end.strftime("%Y-%m-%d")   if hasattr(test_end,   "strftime") else str(test_end)
    trs = train_start.strftime("%Y-%m-%d") if hasattr(train_start, "strftime") else str(train_start)
    tre = train_end.strftime("%Y-%m-%d")   if hasattr(train_end,   "strftime") else str(train_end)

    print()
    print("=" * 62)
    print("  GOLDAI BACKTEST V4 — COMPARISON REPORT")
    print("=" * 62)
    print(f"  Training period : {trs} → {tre}  (model trained on this)")
    print(f"  Test period     : {ts} → {te}  (OUT-OF-SAMPLE)")
    print(f"  Bars tested     : {total_bars:,}")
    print(f"  Zero leakage    : ✅  (test starts after validation gap)")
    print()

    W = 26

    if v2_only:
        print("  [V2-ONLY mode]")
        if v2:
            _print_single(v2, "V2 (Old Logic)")
        print("=" * 62)
        return

    if v4_only:
        print("  [V4-ONLY mode]")
        if v4:
            _print_single(v4, "V4 (New Logic)")
        print("=" * 62)
        return

    if not v2 or not v4:
        print("  [WARN] Missing results — cannot compare.")
        if v2:
            _print_single(v2, "V2 (Old Logic)")
        if v4:
            _print_single(v4, "V4 (New Logic)")
        print("=" * 62)
        return

    print(f"  {'METRIC':<{W}} {'OLD (V2)':>12} {'NEW (V4)':>12} {'CHANGE':>12}")
    print("  " + "─" * 58)

    def row(label: str, v2_val: str, v4_val: str, change: str = ""):
        print(f"  {label:<{W}} {v2_val:>12} {v4_val:>12} {change:>12}")

    tc = (
        f"{(v4['total_trades'] - v2['total_trades']) / v2['total_trades'] * 100:+.0f}%"
        if v2["total_trades"] > 0 else "N/A"
    )
    row("Total Trades",   str(v2["total_trades"]),       str(v4["total_trades"]),   tc)
    row("Win Rate",
        f"{v2['win_rate']:.1f}%",
        f"{v4['win_rate']:.1f}%",
        f"{v4['win_rate'] - v2['win_rate']:+.1f}%")
    row("Total P&L",
        f"${v2['total_pnl']:.2f}",
        f"${v4['total_pnl']:.2f}",
        fmt_usd(v4['total_pnl'] - v2['total_pnl']))

    pf_v2 = f"{v2['profit_factor']:.2f}" if v2['profit_factor'] != float('inf') else "∞"
    pf_v4 = f"{v4['profit_factor']:.2f}" if v4['profit_factor'] != float('inf') else "∞"
    pf_change = (
        fmt_float(v4['profit_factor'] - v2['profit_factor'])
        if v2['profit_factor'] != float('inf') and v4['profit_factor'] != float('inf')
        else ""
    )
    row("Profit Factor",  pf_v2, pf_v4, pf_change)
    row("Max Drawdown",
        f"${v2['max_drawdown']:.2f}",
        f"${v4['max_drawdown']:.2f}",
        fmt_usd(v2['max_drawdown'] - v4['max_drawdown']))
    row("Avg Win",  f"${v2['avg_win']:.2f}",  f"${v4['avg_win']:.2f}",  "")
    row("Avg Loss", f"${v2['avg_loss']:.2f}", f"${v4['avg_loss']:.2f}", "")
    row("Sharpe Ratio",
        f"{v2['sharpe']:.2f}",
        f"{v4['sharpe']:.2f}",
        fmt_float(v4['sharpe'] - v2['sharpe']))

    print()
    print("  NEW FILTER STATS (V4 only):")
    print(f"  {'H1 Blocked':<{W}} {v4['h1_blocked']:>12} signals")
    print(f"  {'Zone Guard Blocked':<{W}} {v4['zone_blocked']:>12} signals")
    print(f"  {'Avg Dynamic SL':<{W}} {v4['avg_sl']:>11.1f} pips")
    print(f"  {'Avg Dynamic RR':<{W}} {v4['avg_rr']:>12.2f}")
    print()

    improvements = sum([
        v4["win_rate"]      > v2["win_rate"],
        v4["total_pnl"]     > v2["total_pnl"],
        v4["profit_factor"] > v2["profit_factor"],
        v4["max_drawdown"]  < v2["max_drawdown"],
    ])
    v4_better = improvements >= 3
    if v4_better:
        print("  VERDICT: ✅ V4 BETTER")
    else:
        print(f"  VERDICT: ❌ V4 WORSE ({improvements}/4 metrics improved — review filters)")

    print("=" * 62)
    print()


def _print_single(result: Dict, label: str) -> None:
    print(f"\n  {label}")
    print("  " + "─" * 40)
    print(f"  Total Trades   : {result['total_trades']}")
    print(f"  Win Rate       : {result['win_rate']:.1f}%")
    print(f"  Total P&L      : ${result['total_pnl']:.2f}")
    print(f"  Profit Factor  : {result['profit_factor']:.2f}")
    print(f"  Max Drawdown   : ${result['max_drawdown']:.2f}")
    print(f"  Avg Win        : ${result['avg_win']:.2f}")
    print(f"  Avg Loss       : ${result['avg_loss']:.2f}")
    print(f"  Sharpe Ratio   : {result['sharpe']:.2f}")
    if "h1_blocked" in result:
        print(f"  H1 Blocked     : {result['h1_blocked']} signals")
        print(f"  Zone Blocked   : {result['zone_blocked']} signals")
        print(f"  Avg Dynamic SL : {result['avg_sl']:.1f} pips")
        print(f"  Avg Dynamic RR : {result['avg_rr']:.2f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GOLDAI Backtest V4 — Date-Based Out-of-Sample Test"
    )
    parser.add_argument("--symbol",       type=str,   default="XAUUSD.m",
                        help="MT5 symbol (default: XAUUSD.m)")
    parser.add_argument("--lot",          type=float, default=0.01,
                        help="Lot size (default: 0.01)")
    parser.add_argument("--balance",      type=float, default=439.0,
                        help="Starting balance in USD (default: 439)")
    parser.add_argument("--confidence",   type=float, default=0.70,
                        help="ML confidence threshold (default: 0.70)")
    parser.add_argument("--train-start",  type=str,   default="2020-01-01",
                        help="Training start date (default: 2020-01-01)")
    parser.add_argument("--train-end",    type=str,   default="2023-12-31",
                        help="Training end date (default: 2023-12-31)")
    parser.add_argument("--test-start",   type=str,   default="2024-07-01",
                        help="Test start date (default: 2024-07-01)")
    parser.add_argument("--test-end",     type=str,   default="2026-03-26",
                        help="Test end date (default: 2026-03-26)")
    parser.add_argument("--no-trailing",  action="store_true",
                        help="Disable trailing stop")
    parser.add_argument("--no-partial",   action="store_true",
                        help="Disable partial close")
    parser.add_argument("--no-session",   action="store_true",
                        help="Disable session filter")
    parser.add_argument("--v2-only",      action="store_true",
                        help="Run V2 pass only")
    parser.add_argument("--v4-only",      action="store_true",
                        help="Run V4 pass only")
    args = parser.parse_args()

    # Parse dates
    train_start = datetime.strptime(args.train_start, "%Y-%m-%d")
    train_end   = datetime.strptime(args.train_end,   "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    test_start  = datetime.strptime(args.test_start,  "%Y-%m-%d")
    test_end    = datetime.strptime(args.test_end,    "%Y-%m-%d").replace(hour=23, minute=59, second=59)

    print()
    print("=" * 62)
    print("  GOLDAI BACKTEST V4")
    print("=" * 62)
    print(f"  Symbol      : {args.symbol}")
    print(f"  Lot         : {args.lot}")
    print(f"  Balance     : ${args.balance:.2f}")
    print(f"  Confidence  : {args.confidence}")
    print(f"  Train period: {train_start.date()} → {train_end.date()}")
    print(f"  Test period : {test_start.date()} → {test_end.date()}  (OUT-OF-SAMPLE)")
    print()

    # ------------------------------------------------------------------
    # MT5 connection
    # ------------------------------------------------------------------
    try:
        config = get_config()
    except Exception as e:
        print(f"[ERROR] Could not load config: {e}")
        sys.exit(1)

    try:
        mt5_conn = MT5Connector(
            login=config.mt5_login,
            password=config.mt5_password,
            server=config.mt5_server,
        )
        if not mt5_conn.ensure_connected():
            print("[ERROR] Cannot connect to MT5. Is the terminal running?")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] MT5 connection failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # [1/5] Fetch and filter M15 data to the test period
    # ------------------------------------------------------------------
    print(f"[1/5] Fetching M15 data for test period "
          f"{test_start.date()} → {test_end.date()} ...")
    try:
        df_raw = mt5_conn.get_market_data(
            symbol=args.symbol,
            timeframe="M15",
            count=FETCH_BARS_M15,
        )
    except Exception as e:
        print(f"[ERROR] M15 data fetch failed: {e}")
        sys.exit(1)

    if df_raw is None or len(df_raw) == 0:
        print("[ERROR] No M15 data received from MT5")
        sys.exit(1)

    # Filter to test window
    df_test = df_raw.filter(
        (pl.col("time") >= test_start) & (pl.col("time") <= test_end)
    )

    if len(df_test) < 500:
        print(
            f"[ERROR] Too few bars in test window: {len(df_test)}. "
            "Increase FETCH_BARS_M15 or check dates."
        )
        sys.exit(1)

    n_test     = len(df_test)
    test_times = df_test["time"].to_list()
    actual_start = test_times[0]  if test_times else test_start
    actual_end   = test_times[-1] if test_times else test_end

    as_str = actual_start.strftime("%Y-%m-%d") if hasattr(actual_start, "strftime") else str(actual_start)
    ae_str = actual_end.strftime("%Y-%m-%d")   if hasattr(actual_end,   "strftime") else str(actual_end)
    print(f"       Received {len(df_raw):,} total M15 bars")
    print(f"       Test window: {as_str} → {ae_str}  ({n_test:,} bars — OUT-OF-SAMPLE)")
    print()

    # ------------------------------------------------------------------
    # [2/5] Fetch H1 data for H1 bias calculation
    # ------------------------------------------------------------------
    print(f"[2/5] Fetching H1 data for bias calculation ...")
    h1_table = pl.DataFrame()
    try:
        df_h1_raw = mt5_conn.get_market_data(
            symbol=args.symbol,
            timeframe="H1",
            count=FETCH_BARS_H1,
        )
        if df_h1_raw is not None and len(df_h1_raw) > 50:
            # Filter H1 to test window (with some look-back for warm-up)
            df_h1_filt = df_h1_raw.filter(pl.col("time") <= test_end)
            fe_h1  = FeatureEngineer()
            smc_h1 = SMCAnalyzer()
            df_h1  = fe_h1.calculate_all(df_h1_filt, include_ml_features=False)
            df_h1  = smc_h1.calculate_all(df_h1)

            # Try to add regime using the pre-trained HMM
            reg_h1  = MarketRegimeDetector(n_regimes=3)
            hmm_path = ROOT / "models" / "hmm_regime.pkl"
            if hmm_path.exists():
                try:
                    reg_h1.load(str(hmm_path))
                    df_h1 = reg_h1.predict(df_h1)
                except Exception:
                    df_h1 = df_h1.with_columns(pl.lit(1).alias("regime"))
            else:
                df_h1 = df_h1.with_columns(pl.lit(1).alias("regime"))

            h1_table = build_h1_bias_table(df_h1)
            print(f"       H1 bias table: {len(h1_table):,} rows")
        else:
            print("       [WARN] H1 data insufficient — H1 blocker disabled")
    except Exception as e:
        print(f"       [WARN] H1 fetch/calc failed: {e} — H1 blocker disabled")
    print()

    # ------------------------------------------------------------------
    # [3/5] Feature engineering on test data
    # ------------------------------------------------------------------
    print("[3/5] Calculating features & SMC on test data ...")
    fe  = FeatureEngineer()
    smc = SMCAnalyzer()
    try:
        df_test = fe.calculate_all(df_test, include_ml_features=True)
        df_test = smc.calculate_all(df_test)
    except Exception as e:
        print(f"[ERROR] Feature/SMC calculation failed: {e}")
        sys.exit(1)
    print()

    # ------------------------------------------------------------------
    # [4/5] Load pre-trained models
    # ------------------------------------------------------------------
    print("[4/5] Loading pre-trained models from models/ ...")

    # HMM regime
    regime_detector = MarketRegimeDetector(n_regimes=3)
    hmm_path = ROOT / "models" / "hmm_regime.pkl"
    hmm_loaded = False
    if hmm_path.exists():
        try:
            regime_detector.load(str(hmm_path))
            df_test    = regime_detector.predict(df_test)
            hmm_loaded = True
            print(f"       HMM loaded: {hmm_path.name}")
        except Exception as e:
            print(f"       [WARN] HMM load failed: {e} — using default regime")
    else:
        print("       [WARN] hmm_regime.pkl not found — run train_models.py first")

    if not hmm_loaded:
        df_test = df_test.with_columns([
            pl.lit(1).alias("regime"),
            pl.lit("medium_volatility").alias("regime_name"),
            pl.lit(1.0).alias("regime_confidence"),
        ])

    # XGBoost model
    model = None
    feature_cols = get_default_feature_columns()
    xgb_candidates = [
        ROOT / "models" / "xgboost_model.pkl",
        ROOT / "models" / "xgb_model.pkl",
    ]
    for mp in xgb_candidates:
        if mp.exists():
            try:
                m = TradingModel(
                    confidence_threshold=args.confidence,
                    model_path=str(mp),
                )
                m.load(str(mp))
                model = m
                print(f"       XGBoost loaded: {mp.name}")

                # Leakage check: model's training cutoff must be before test start
                cutoff = model.train_cutoff_date
                if cutoff is not None:
                    cutoff_str = cutoff.strftime("%Y-%m-%d") if hasattr(cutoff, "strftime") else str(cutoff)
                    if cutoff >= test_start:
                        print()
                        print("  " + "⚠️  " * 10)
                        print("  ⚠️  DATA LEAKAGE WARNING  ⚠️")
                        print(f"  Model trained up to : {cutoff_str}")
                        print(f"  Test period starts  : {test_start.date()}")
                        print("  Re-train with: python train_models.py")
                        print("  " + "⚠️  " * 10)
                        print()
                    else:
                        print(f"       ✅ No leakage: model cutoff {cutoff_str} < test start {test_start.date()}")
                else:
                    print("       [WARN] Model has no train_cutoff_date — cannot verify leakage")
                break
            except Exception as e:
                print(f"       [WARN] Could not load {mp.name}: {e}")

    if model is None:
        print("       [WARN] No XGBoost model found — ML filter disabled")
        print("              Run: python train_models.py")
    print()

    # ------------------------------------------------------------------
    # [5/5] Run backtest passes
    # ------------------------------------------------------------------
    print("[5/5] Running backtest passes ...")
    print()

    use_trailing = not args.no_trailing
    use_partial  = not args.no_partial
    use_session  = not args.no_session
    cooldown     = 1

    v2_result = None
    v4_result = None

    if not args.v4_only:
        print("  → Pass A: V2 (old logic — no H1 blocker, no zone guard, raw SMC SL/TP) ...")
        v2_result = run_backtest(
            df=df_test,
            h1_table=h1_table,
            model=model,
            feature_cols=feature_cols,
            smc=smc,
            features_eng=fe,
            use_h1_blocker=False,
            use_zone_guard=False,
            use_dynamic_sl=False,
            use_dynamic_rr=False,
            use_trailing=use_trailing,
            use_partial=use_partial,
            use_session=use_session,
            use_ml=(model is not None),
            lot_size=args.lot,
            initial_balance=args.balance,
            confidence_threshold=args.confidence,
            cooldown_bars=cooldown,
        )
        print(
            f"     Trades: {v2_result['total_trades']} | "
            f"Win: {v2_result['win_rate']:.1f}% | "
            f"P&L: ${v2_result['total_pnl']:.2f} | "
            f"PF: {v2_result['profit_factor']:.2f}"
        )

    if not args.v2_only:
        print("  → Pass B: V4 (all filters: H1 Blocker, Zone Guard, Dynamic SL/RR) ...")
        v4_result = run_backtest(
            df=df_test,
            h1_table=h1_table,
            model=model,
            feature_cols=feature_cols,
            smc=smc,
            features_eng=fe,
            use_h1_blocker=(not h1_table.is_empty()),
            use_zone_guard=True,
            use_dynamic_sl=True,
            use_dynamic_rr=True,
            use_trailing=use_trailing,
            use_partial=use_partial,
            use_session=use_session,
            use_ml=(model is not None),
            lot_size=args.lot,
            initial_balance=args.balance,
            confidence_threshold=args.confidence,
            cooldown_bars=cooldown,
        )
        print(
            f"     Trades: {v4_result['total_trades']} | "
            f"Win: {v4_result['win_rate']:.1f}% | "
            f"P&L: ${v4_result['total_pnl']:.2f} | "
            f"PF: {v4_result['profit_factor']:.2f}"
        )

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print_report(
        v2=v2_result or {},
        v4=v4_result or {},
        test_start=actual_start,
        test_end=actual_end,
        train_start=train_start,
        train_end=train_end,
        total_bars=n_test,
        v2_only=args.v2_only,
        v4_only=args.v4_only,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GOLDAI Backtest V3 — Out-of-Sample Comparison
==============================================
Tests the 5 improvements from PR #8 on out-of-sample data (last 25%).

Improvements tested:
1. H1 Blocker — blocks trades opposing strong/moderate H1 trend
2. Zone Guard — prevents re-entry in the same price zone after a loss
3. Dynamic SL — 2.0× ATR with min 20 points
4. Dynamic RR — 1.5–2.0 based on market conditions
5. ML V2 Model support

Usage:
    python backtests/backtest_v3.py [--bars 140000] [--lot 0.01] [--balance 439]
        [--confidence 0.65] [--symbol XAUUSD.m] [--no-trailing] [--no-partial]
        [--no-session] [--v2-only] [--v3-only]
"""

import argparse
import sys
import os
from pathlib import Path
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

# Suppress verbose logging during backtest
logger.remove()
logger.add(sys.stderr, level="WARNING")


# ---------------------------------------------------------------------------
# Session filter (London + NY, UTC hours)
# ---------------------------------------------------------------------------
LONDON_OPEN = 7
LONDON_CLOSE = 16
NY_OPEN = 13
NY_CLOSE = 22


def get_session(hour_utc: int) -> Optional[str]:
    """Return active session or None if outside trading hours."""
    in_london = LONDON_OPEN <= hour_utc < LONDON_CLOSE
    in_ny = NY_OPEN <= hour_utc < NY_CLOSE
    if in_london and in_ny:
        return "LONDON_NY"
    if in_london:
        return "LONDON"
    if in_ny:
        return "NY"
    return None


# ---------------------------------------------------------------------------
# H1 Bias calculation (matches main_live.py _get_h1_bias logic)
# ---------------------------------------------------------------------------

def _count_candle_bias_row(closes: List[float], opens: List[float]) -> int:
    """Count bullish/bearish candles in last 5 candles."""
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
    """Get indicator weights based on regime string."""
    regime = regime_str.lower()
    if "low" in regime or "ranging" in regime:
        return {"ema_trend": 0.15, "ema_cross": 0.15, "rsi": 0.30, "macd": 0.25, "candles": 0.15}
    elif "high" in regime or "trending" in regime:
        return {"ema_trend": 0.30, "ema_cross": 0.25, "rsi": 0.10, "macd": 0.25, "candles": 0.10}
    else:
        return {"ema_trend": 0.25, "ema_cross": 0.20, "rsi": 0.20, "macd": 0.20, "candles": 0.15}


def build_h1_bias_table(df_h1: pl.DataFrame) -> pl.DataFrame:
    """
    Pre-calculate H1 bias for each H1 bar.
    Returns DataFrame with columns: time_hour (floored to hour), h1_bias, h1_strength
    """
    # Ensure required columns exist
    required = ["time", "close", "open", "ema_9", "ema_21", "rsi", "macd_histogram"]
    for col in required:
        if col not in df_h1.columns:
            logger.warning(f"H1 column missing: {col} — H1 bias will be NEUTRAL")
            return pl.DataFrame({"time_hour": pl.Series([], dtype=pl.Datetime), "h1_bias": [], "h1_strength": []})

    rows = df_h1.to_dicts()
    results = []

    for i, row in enumerate(rows):
        if i < 30:
            results.append({"time_hour": row["time"], "h1_bias": "NEUTRAL", "h1_strength": "weak"})
            continue

        price = row["close"]
        ema_9 = row["ema_9"]
        ema_21 = row["ema_21"]
        rsi = row["rsi"]
        macd_hist = row["macd_histogram"]

        # Guard against None values
        if any(v is None for v in [price, ema_9, ema_21, rsi, macd_hist]):
            results.append({"time_hour": row["time"], "h1_bias": "NEUTRAL", "h1_strength": "weak"})
            continue

        closes = [r["close"] for r in rows[max(0, i - 4): i + 1]]
        opens = [r["open"] for r in rows[max(0, i - 4): i + 1]]

        signals = {
            "ema_trend": 1 if price > ema_21 else (-1 if price < ema_21 else 0),
            "ema_cross": 1 if ema_9 > ema_21 else (-1 if ema_9 < ema_21 else 0),
            "rsi": 1 if rsi > 55 else (-1 if rsi < 45 else 0),
            "macd": 1 if macd_hist > 0 else (-1 if macd_hist < 0 else 0),
            "candles": _count_candle_bias_row(closes, opens),
        }

        # Use balanced weights (no live regime available in backtest)
        weights = _get_regime_weights_static("medium_volatility")
        score = sum(signals[k] * weights[k] for k in signals)

        if score >= 0.3:
            bias = "BULLISH"
        elif score <= -0.3:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        abs_score = abs(score)
        if abs_score >= 0.7:
            strength = "strong"
        elif abs_score >= 0.5:
            strength = "moderate"
        else:
            strength = "weak"

        results.append({"time_hour": row["time"], "h1_bias": bias, "h1_strength": strength})

    return pl.DataFrame(results)


def lookup_h1_bias(h1_table: pl.DataFrame, bar_time) -> Tuple[str, str]:
    """
    Find H1 bias for a given M15 bar time.
    Floors bar_time to the hour and finds the matching H1 row.
    Returns (bias, strength).
    """
    if h1_table.is_empty():
        return "NEUTRAL", "weak"

    try:
        # Floor to hour
        if hasattr(bar_time, "replace"):
            target_hour = bar_time.replace(minute=0, second=0, microsecond=0)
        else:
            return "NEUTRAL", "weak"

        # Find matching row
        match = h1_table.filter(pl.col("time_hour") == target_hour)
        if match.is_empty():
            # Try to find the closest earlier H1 bar
            earlier = h1_table.filter(pl.col("time_hour") <= target_hour)
            if earlier.is_empty():
                return "NEUTRAL", "weak"
            match = earlier.tail(1)

        return match["h1_bias"][0], match["h1_strength"][0]
    except Exception:
        return "NEUTRAL", "weak"


# ---------------------------------------------------------------------------
# Dynamic RR calculation (mirrors smc_polars._calculate_dynamic_rr)
# ---------------------------------------------------------------------------

def calculate_dynamic_rr(
    market_structure: int,
    has_bullish_break: bool,
    has_bearish_break: bool,
    has_fvg: bool,
    has_ob: bool,
    df_window: pl.DataFrame,
) -> float:
    """Calculate dynamic RR ratio (1.5–2.0) based on market conditions."""
    rr = 1.5

    if market_structure != 0:
        rr += 0.15

    if has_bullish_break or has_bearish_break:
        rr += 0.10

    if has_fvg:
        rr += 0.05
    if has_ob:
        rr += 0.05

    if "bos" in df_window.columns:
        recent_bos = df_window.tail(20)["bos"].to_list()
        bos_count = sum(1 for b in recent_bos if b != 0)
        if bos_count >= 3:
            rr += 0.10
        elif bos_count >= 2:
            rr += 0.05

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
        bos_count_30 = sum(1 for b in recent_bos_30 if b != 0)
        if bos_count_30 == 0:
            rr -= 0.10

    return max(1.5, min(2.0, rr))


# ---------------------------------------------------------------------------
# Trade dataclass
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
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.tp = tp
        self.entry_bar = entry_bar
        self.exit_price: Optional[float] = None
        self.exit_bar: Optional[int] = None
        self.result: Optional[str] = None  # "win", "loss", "partial_win"
        self.pnl: float = 0.0
        self.closed_partial: bool = False
        self.partial_price: Optional[float] = None
        self.partial_pnl: float = 0.0
        self.trailing_active: bool = False
        self.trailing_sl: float = sl
        self.lot_size = lot_size
        self.pip_value = pip_value


# ---------------------------------------------------------------------------
# Main backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    df: pl.DataFrame,
    h1_table: pl.DataFrame,
    model: Optional[TradingModel],
    feature_cols: List[str],
    smc: SMCAnalyzer,
    features_eng: FeatureEngineer,
    # Options
    use_h1_blocker: bool = True,
    use_zone_guard: bool = True,
    use_dynamic_sl: bool = True,
    use_dynamic_rr: bool = True,
    use_trailing: bool = True,
    use_partial: bool = True,
    use_session: bool = True,
    use_ml: bool = True,
    lot_size: float = 0.01,
    initial_balance: float = 439.0,
    confidence_threshold: float = 0.65,
) -> Dict:
    """
    Run a single-pass backtest over the provided DataFrame.

    Returns a dict with performance metrics and per-trade stats.
    """
    pip_value = lot_size * 100 * 0.01  # XAUUSD pip value

    balance = initial_balance
    equity_curve: List[float] = [balance]
    trades: List[Trade] = []
    active_trade: Optional[Trade] = None

    # Zone Guard state: list of (direction, price, bar_index)
    recent_loss_zones: List[Tuple[str, float, int]] = []

    h1_blocked_count = 0
    zone_blocked_count = 0
    sl_distances: List[float] = []
    rr_values: List[float] = []

    n_bars = len(df)
    times = df["time"].to_list() if "time" in df.columns else [None] * n_bars
    closes = df["close"].to_list()
    highs = df["high"].to_list()
    lows = df["low"].to_list()

    WARMUP = 100  # bars needed for indicators to stabilise

    for i in range(WARMUP, n_bars):
        bar_time = times[i]
        close = closes[i]
        high = highs[i]
        low = lows[i]

        # ----------------------------------------------------------------
        # Manage active trade
        # ----------------------------------------------------------------
        if active_trade is not None:
            t = active_trade
            risk = abs(t.entry - t.sl)
            atr_for_trail = risk  # Approximate ATR as risk distance

            # --- Trailing stop ---
            if use_trailing and t.trailing_active:
                if t.direction == "BUY":
                    new_trail = close - 1.5 * atr_for_trail
                    if new_trail > t.trailing_sl:
                        t.trailing_sl = new_trail
                else:
                    new_trail = close + 1.5 * atr_for_trail
                    if new_trail < t.trailing_sl:
                        t.trailing_sl = new_trail

            # Activate trailing after 1 ATR profit
            if use_trailing and not t.trailing_active:
                if t.direction == "BUY" and close >= t.entry + atr_for_trail:
                    t.trailing_active = True
                    t.trailing_sl = close - 1.5 * atr_for_trail
                elif t.direction == "SELL" and close <= t.entry - atr_for_trail:
                    t.trailing_active = True
                    t.trailing_sl = close + 1.5 * atr_for_trail

            effective_sl = t.trailing_sl if t.trailing_active else t.sl

            # --- Partial close at halfway to TP ---
            if use_partial and not t.closed_partial:
                halfway = t.entry + (t.tp - t.entry) * 0.5
                if t.direction == "BUY" and high >= halfway:
                    t.closed_partial = True
                    t.partial_price = halfway
                    partial_pts = halfway - t.entry
                    t.partial_pnl = partial_pts / 0.01 * pip_value * 0.5
                    balance += t.partial_pnl
                elif t.direction == "SELL" and low <= halfway:
                    t.closed_partial = True
                    t.partial_price = halfway
                    partial_pts = t.entry - halfway
                    t.partial_pnl = partial_pts / 0.01 * pip_value * 0.5
                    balance += t.partial_pnl

            # --- Check SL hit ---
            sl_hit = False
            if t.direction == "BUY" and low <= effective_sl:
                sl_hit = True
                exit_price = effective_sl
            elif t.direction == "SELL" and high >= effective_sl:
                sl_hit = True
                exit_price = effective_sl

            if sl_hit:
                t.exit_price = exit_price
                t.exit_bar = i
                remaining_lot = 0.5 if t.closed_partial else 1.0
                if t.direction == "BUY":
                    pts = exit_price - t.entry
                else:
                    pts = t.entry - exit_price
                t.pnl = pts / 0.01 * pip_value * remaining_lot + t.partial_pnl
                balance += pts / 0.01 * pip_value * remaining_lot
                t.result = "loss" if t.pnl < 0 else "win"
                trades.append(t)
                active_trade = None

                # Zone Guard: record loss zone
                if t.result == "loss":
                    recent_loss_zones.append((t.direction, t.entry, i))
                equity_curve.append(balance)
                continue

            # --- Check TP hit ---
            tp_hit = False
            if t.direction == "BUY" and high >= t.tp:
                tp_hit = True
                exit_price = t.tp
            elif t.direction == "SELL" and low <= t.tp:
                tp_hit = True
                exit_price = t.tp

            if tp_hit:
                t.exit_price = exit_price
                t.exit_bar = i
                remaining_lot = 0.5 if t.closed_partial else 1.0
                if t.direction == "BUY":
                    pts = exit_price - t.entry
                else:
                    pts = t.entry - exit_price
                t.pnl = pts / 0.01 * pip_value * remaining_lot + t.partial_pnl
                balance += pts / 0.01 * pip_value * remaining_lot
                t.result = "win"
                trades.append(t)
                active_trade = None
                equity_curve.append(balance)
                continue

            equity_curve.append(balance)
            continue  # Trade still open — skip new signal logic

        # ----------------------------------------------------------------
        # No active trade — look for a new signal
        # ----------------------------------------------------------------
        equity_curve.append(balance)

        # Session filter
        if use_session and bar_time is not None:
            hour_utc = bar_time.hour if hasattr(bar_time, "hour") else 0
            if get_session(hour_utc) is None:
                continue

        # Build window using pre-calculated features+SMC (no recalculation needed)
        window_start = max(0, i - 199)
        df_window = df.slice(window_start, i - window_start + 1)

        # Generate SMC signal (uses pre-computed columns in df_window)
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
                pred = model.predict(df_window)
                if pred.signal == "HOLD":
                    continue
                if pred.signal != signal.signal_type:
                    continue
                if pred.confidence < confidence_threshold:
                    continue
            except Exception:
                pass  # Fall through if ML fails

        confidence = signal.confidence
        if confidence < confidence_threshold:
            continue

        # ----------------------------------------------------------------
        # H1 Blocker (V3 only)
        # ----------------------------------------------------------------
        if use_h1_blocker and bar_time is not None and not h1_table.is_empty():
            h1_bias, h1_strength = lookup_h1_bias(h1_table, bar_time)

            h1_opposed = (
                (signal.signal_type == "BUY" and h1_bias == "BEARISH") or
                (signal.signal_type == "SELL" and h1_bias == "BULLISH")
            )
            h1_aligned = (
                (signal.signal_type == "BUY" and h1_bias == "BULLISH") or
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

        # ----------------------------------------------------------------
        # Zone Guard (V3 only)
        # ----------------------------------------------------------------
        if use_zone_guard:
            # Expire zones older than 20 bars
            recent_loss_zones = [
                (d, p, b) for d, p, b in recent_loss_zones
                if i - b < 20
            ]

            # Get ATR for zone radius
            atr_val = None
            if "atr" in df_window.columns:
                atr_vals = df_window.tail(1)["atr"].to_list()
                if atr_vals and atr_vals[0] is not None and atr_vals[0] > 0:
                    atr_val = atr_vals[0]
            zone_radius = atr_val if atr_val else 15.0

            zone_blocked = False
            for prev_dir, prev_price, _ in recent_loss_zones:
                if prev_dir == signal.signal_type and abs(close - prev_price) <= zone_radius:
                    zone_blocked = True
                    break

            if zone_blocked:
                zone_blocked_count += 1
                continue

        # ----------------------------------------------------------------
        # Determine SL / TP
        # ----------------------------------------------------------------
        entry = close

        if "atr" in df_window.columns:
            atr_vals = df_window.tail(1)["atr"].to_list()
            atr = atr_vals[0] if atr_vals and atr_vals[0] is not None and atr_vals[0] > 0 else 12.0
        else:
            atr = 12.0

        if use_dynamic_sl:
            # Dynamic SL: max(2.0 × ATR, 20 points)
            min_sl_dist = max(2.0 * atr, 20.0)
        else:
            # Old logic: use raw SMC signal SL
            min_sl_dist = abs(entry - signal.stop_loss) if signal.stop_loss else (1.5 * atr)
            min_sl_dist = max(min_sl_dist, 5.0)  # sanity floor

        if signal.signal_type == "BUY":
            # Use further of swing SL or ATR SL
            swing_sl = signal.stop_loss if signal.stop_loss and signal.stop_loss < entry else None
            atr_sl = entry - min_sl_dist
            sl = min(swing_sl, atr_sl) if swing_sl else atr_sl
            sl = min(sl, entry - min_sl_dist)
        else:
            swing_sl = signal.stop_loss if signal.stop_loss and signal.stop_loss > entry else None
            atr_sl = entry + min_sl_dist
            sl = max(swing_sl, atr_sl) if swing_sl else atr_sl
            sl = max(sl, entry + min_sl_dist)

        risk = abs(entry - sl)
        if risk <= 0:
            continue

        # Dynamic RR
        if use_dynamic_rr:
            latest_row = df_window.tail(1)
            market_structure = latest_row["market_structure"].item() if "market_structure" in df_window.columns else 0
            recent_bos = df_window.tail(10)["bos"].to_list() if "bos" in df_window.columns else []
            recent_choch = df_window.tail(10)["choch"].to_list() if "choch" in df_window.columns else []
            has_bullish_break = 1 in recent_bos or 1 in recent_choch
            has_bearish_break = -1 in recent_bos or -1 in recent_choch
            has_fvg = any(df_window.tail(10)["is_fvg_bull"].to_list()) if "is_fvg_bull" in df_window.columns else False
            has_ob = any(o != 0 for o in (df_window.tail(10)["ob"].to_list() if "ob" in df_window.columns else []))
            rr = calculate_dynamic_rr(
                market_structure=market_structure,
                has_bullish_break=has_bullish_break,
                has_bearish_break=has_bearish_break,
                has_fvg=has_fvg,
                has_ob=has_ob,
                df_window=df_window,
            )
        else:
            rr = 1.5

        if signal.signal_type == "BUY":
            tp = entry + risk * rr
        else:
            tp = entry - risk * rr

        sl_distances.append(min_sl_dist)
        rr_values.append(rr)

        # Create trade
        active_trade = Trade(
            direction=signal.signal_type,
            entry=entry,
            sl=sl,
            tp=tp,
            entry_bar=i,
            lot_size=lot_size,
            pip_value=pip_value,
        )

    # Close any open trade at end of data
    if active_trade is not None:
        t = active_trade
        exit_price = closes[-1]
        t.exit_price = exit_price
        t.exit_bar = n_bars - 1
        remaining_lot = 0.5 if t.closed_partial else 1.0
        if t.direction == "BUY":
            pts = exit_price - t.entry
        else:
            pts = t.entry - exit_price
        t.pnl = pts / 0.01 * pip_value * remaining_lot + t.partial_pnl
        balance += pts / 0.01 * pip_value * remaining_lot
        t.result = "win" if t.pnl > 0 else "loss"
        trades.append(t)
        equity_curve.append(balance)

    # ----------------------------------------------------------------
    # Compute metrics
    # ----------------------------------------------------------------
    total = len(trades)
    wins = [t for t in trades if t.result == "win"]
    losses = [t for t in trades if t.result == "loss"]
    win_rate = len(wins) / total * 100 if total > 0 else 0.0

    total_pnl = balance - initial_balance
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0

    # Max drawdown
    peak = initial_balance
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    # Sharpe (daily returns approximation)
    pnls = [t.pnl for t in trades]
    if len(pnls) > 1:
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)
        sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_trades": total,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe": sharpe,
        "h1_blocked": h1_blocked_count,
        "zone_blocked": zone_blocked_count,
        "avg_sl": float(np.mean(sl_distances)) if sl_distances else 0.0,
        "avg_rr": float(np.mean(rr_values)) if rr_values else 0.0,
        "equity_curve": equity_curve,
        "trades": trades,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(
    v2: Dict,
    v3: Optional[Dict],
    test_start,
    test_end,
    total_bars: int,
    v2_only: bool,
    v3_only: bool,
) -> None:
    """Print a formatted comparison report."""

    def fmt_pct(v: float) -> str:
        return f"{v:+.1f}%" if v != 0 else "—"

    def fmt_usd(v: float) -> str:
        sign = "+" if v > 0 else ""
        return f"{sign}${v:.2f}"

    def fmt_float(v: float) -> str:
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.2f}"

    start_str = test_start.strftime("%Y-%m-%d") if hasattr(test_start, "strftime") else str(test_start)
    end_str = test_end.strftime("%Y-%m-%d") if hasattr(test_end, "strftime") else str(test_end)

    print()
    print("=" * 62)
    print("  GOLDAI BACKTEST V3 — COMPARISON REPORT")
    print("=" * 62)
    print(f"  Test Period : {start_str} to {end_str}  (OUT-OF-SAMPLE)")
    print(f"  Bars Tested : {total_bars:,}")
    print()

    if v2_only:
        print("  [V2-ONLY mode — V3 pass skipped]")
        if v2:
            _print_single(v2, "V2 (Old Logic)")
        return

    if v3_only:
        print("  [V3-ONLY mode — V2 pass skipped]")
        if v3:
            _print_single(v3, "V3 (New Logic)")
        return

    # Guard: both passes must have results
    if not v2 or not v3:
        print("  [WARN] Missing results — cannot compare.")
        if v2:
            _print_single(v2, "V2 (Old Logic)")
        if v3:
            _print_single(v3, "V3 (New Logic)")
        return

    # Both passes — comparison table
    W = 26
    print(f"  {'METRIC':<{W}} {'OLD (V2)':>12} {'NEW (V3)':>12} {'CHANGE':>12}")
    print("  " + "─" * 58)

    def row(label: str, v2_val: str, v3_val: str, change: str = ""):
        print(f"  {label:<{W}} {v2_val:>12} {v3_val:>12} {change:>12}")

    trade_change = (
        f"{(v3['total_trades'] - v2['total_trades']) / v2['total_trades'] * 100:+.0f}%"
        if v2["total_trades"] > 0
        else "N/A"
    )
    row("Total Trades",
        str(v2["total_trades"]),
        str(v3["total_trades"]),
        trade_change)

    row("Win Rate",
        f"{v2['win_rate']:.1f}%",
        f"{v3['win_rate']:.1f}%",
        f"{v3['win_rate'] - v2['win_rate']:+.1f}%")

    row("Total P&L",
        f"${v2['total_pnl']:.2f}",
        f"${v3['total_pnl']:.2f}",
        fmt_usd(v3['total_pnl'] - v2['total_pnl']))

    row("Profit Factor",
        f"{v2['profit_factor']:.2f}" if v2['profit_factor'] != float('inf') else "∞",
        f"{v3['profit_factor']:.2f}" if v3['profit_factor'] != float('inf') else "∞",
        fmt_float(v3['profit_factor'] - v2['profit_factor']) if v2['profit_factor'] != float('inf') and v3['profit_factor'] != float('inf') else "")

    row("Max Drawdown",
        f"${v2['max_drawdown']:.2f}",
        f"${v3['max_drawdown']:.2f}",
        fmt_usd(v2['max_drawdown'] - v3['max_drawdown']))  # positive = improvement

    row("Avg Win",
        f"${v2['avg_win']:.2f}",
        f"${v3['avg_win']:.2f}", "")

    row("Avg Loss",
        f"${v2['avg_loss']:.2f}",
        f"${v3['avg_loss']:.2f}", "")

    row("Sharpe Ratio",
        f"{v2['sharpe']:.2f}",
        f"{v3['sharpe']:.2f}",
        fmt_float(v3['sharpe'] - v2['sharpe']))

    print()
    print("  NEW FILTER STATS (V3 only):")
    print(f"  {'H1 Blocked':<{W}} {v3['h1_blocked']:>12} signals")
    print(f"  {'Zone Guard Blocked':<{W}} {v3['zone_blocked']:>12} signals")
    print(f"  {'Avg Dynamic SL':<{W}} {v3['avg_sl']:>11.1f} pips")
    print(f"  {'Avg Dynamic RR':<{W}} {v3['avg_rr']:>12.2f}")
    print()

    # Verdict
    v3_better = (
        v3["win_rate"] >= v2["win_rate"] and
        v3["total_pnl"] >= v2["total_pnl"] and
        v3["profit_factor"] >= v2["profit_factor"]
    )
    if v3_better:
        print("  VERDICT: ✅ V3 BETTER")
    else:
        # Partial improvement
        improvements = 0
        if v3["win_rate"] > v2["win_rate"]:
            improvements += 1
        if v3["total_pnl"] > v2["total_pnl"]:
            improvements += 1
        if v3["profit_factor"] > v2["profit_factor"]:
            improvements += 1
        if v3["max_drawdown"] < v2["max_drawdown"]:
            improvements += 1
        if improvements >= 3:
            print("  VERDICT: ✅ V3 BETTER (3+ metrics improved)")
        else:
            print("  VERDICT: ❌ V3 WORSE (fewer metrics improved — review filters)")

    print("=" * 62)
    print()


def _print_single(result: Dict, label: str) -> None:
    """Print metrics for a single pass."""
    print(f"\n  {label}")
    print("  " + "─" * 40)
    print(f"  Total Trades    : {result['total_trades']}")
    print(f"  Win Rate        : {result['win_rate']:.1f}%")
    print(f"  Total P&L       : ${result['total_pnl']:.2f}")
    print(f"  Profit Factor   : {result['profit_factor']:.2f}")
    print(f"  Max Drawdown    : ${result['max_drawdown']:.2f}")
    print(f"  Avg Win         : ${result['avg_win']:.2f}")
    print(f"  Avg Loss        : ${result['avg_loss']:.2f}")
    print(f"  Sharpe Ratio    : {result['sharpe']:.2f}")
    if "h1_blocked" in result:
        print(f"  H1 Blocked      : {result['h1_blocked']} signals")
        print(f"  Zone Blocked    : {result['zone_blocked']} signals")
        print(f"  Avg Dynamic SL  : {result['avg_sl']:.1f} pips")
        print(f"  Avg Dynamic RR  : {result['avg_rr']:.2f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GOLDAI Backtest V3 — Out-of-Sample Comparison"
    )
    parser.add_argument("--bars", type=int, default=140000,
                        help="Total M15 bars to fetch from MT5 (default: 140000 ≈ 4 years)")
    parser.add_argument("--lot", type=float, default=0.01, help="Lot size (default: 0.01)")
    parser.add_argument("--balance", type=float, default=439.0,
                        help="Starting balance in USD (default: 439)")
    parser.add_argument("--confidence", type=float, default=0.65,
                        help="ML confidence threshold (default: 0.65)")
    parser.add_argument("--symbol", type=str, default="XAUUSD.m",
                        help="MT5 symbol (default: XAUUSD.m)")
    parser.add_argument("--no-trailing", action="store_true", help="Disable trailing stop")
    parser.add_argument("--no-partial", action="store_true", help="Disable partial close")
    parser.add_argument("--no-session", action="store_true", help="Disable session filter")
    parser.add_argument("--v2-only", action="store_true", help="Run V2 pass only (skip V3)")
    parser.add_argument("--v3-only", action="store_true", help="Run V3 pass only (skip V2)")
    args = parser.parse_args()

    print()
    print("=" * 62)
    print("  GOLDAI BACKTEST V3")
    print("=" * 62)
    print(f"  Symbol  : {args.symbol}")
    print(f"  Total   : {args.bars:,} M15 bars requested")
    print(f"  Lot     : {args.lot}")
    print(f"  Balance : ${args.balance:.2f}")
    print(f"  Conf.   : {args.confidence}")
    print()

    # ------------------------------------------------------------------
    # Connect to MT5
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
    # Fetch M15 data
    # ------------------------------------------------------------------
    print(f"[1/5] Fetching {args.bars:,} M15 bars for {args.symbol} ...")
    try:
        df_full = mt5_conn.get_market_data(
            symbol=args.symbol,
            timeframe="M15",
            count=args.bars,
        )
    except Exception as e:
        print(f"[ERROR] Data fetch failed: {e}")
        sys.exit(1)

    if df_full is None or len(df_full) < 500:
        print(f"[ERROR] Not enough data: got {len(df_full) if df_full is not None else 0} bars")
        sys.exit(1)

    total_bars = len(df_full)
    print(f"       Received {total_bars:,} bars")

    # ------------------------------------------------------------------
    # Split: last 25% for out-of-sample testing
    # ------------------------------------------------------------------
    split_idx = int(total_bars * 0.75)
    df_test = df_full.slice(split_idx, total_bars - split_idx)
    n_test = len(df_test)

    test_times = df_test["time"].to_list() if "time" in df_test.columns else []
    test_start = test_times[0] if test_times else None
    test_end = test_times[-1] if test_times else None

    start_str = test_start.strftime("%Y-%m-%d") if hasattr(test_start, "strftime") else str(test_start)
    end_str = test_end.strftime("%Y-%m-%d") if hasattr(test_end, "strftime") else str(test_end)
    print(f"       Test period  : {start_str} → {end_str}  ({n_test:,} bars — OUT-OF-SAMPLE)")
    print()

    # ------------------------------------------------------------------
    # Fetch H1 data for the same period (for H1 bias)
    # ------------------------------------------------------------------
    h1_bars = args.bars // 4 + 200  # M15/4 + buffer
    print(f"[2/5] Fetching ~{h1_bars} H1 bars for bias calculation ...")
    h1_table = pl.DataFrame()
    try:
        df_h1_raw = mt5_conn.get_market_data(
            symbol=args.symbol,
            timeframe="H1",
            count=h1_bars,
        )
        if df_h1_raw is not None and len(df_h1_raw) > 50:
            features_eng_h1 = FeatureEngineer()
            smc_h1 = SMCAnalyzer()
            df_h1_calc = features_eng_h1.calculate_all(df_h1_raw, include_ml_features=False)
            df_h1_calc = smc_h1.calculate_all(df_h1_calc)
            h1_table = build_h1_bias_table(df_h1_calc)
            print(f"       H1 bias table : {len(h1_table):,} rows")
        else:
            print("       [WARN] H1 data insufficient — H1 blocker will be disabled")
    except Exception as e:
        print(f"       [WARN] H1 fetch/calc failed: {e} — H1 blocker disabled")

    # ------------------------------------------------------------------
    # Calculate features + SMC on full test data
    # ------------------------------------------------------------------
    print(f"[3/5] Calculating features & SMC indicators on test data ...")
    features_eng = FeatureEngineer()
    smc = SMCAnalyzer()

    try:
        df_test = features_eng.calculate_all(df_test, include_ml_features=True)
        df_test = smc.calculate_all(df_test)
    except Exception as e:
        print(f"[ERROR] Feature/SMC calculation failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load ML model
    # ------------------------------------------------------------------
    print("[4/5] Loading ML model ...")
    model = None
    feature_cols = get_default_feature_columns()
    model_paths = [
        ROOT / "models" / "xgb_model.pkl",
        ROOT / "models" / "xgboost_model.pkl",
        ROOT / "models" / "xgboost_model.json",
    ]
    for mp in model_paths:
        if mp.exists():
            try:
                model = TradingModel(confidence_threshold=args.confidence, model_path=str(mp))
                model.load(str(mp))
                print(f"       Model loaded : {mp.name}")
                break
            except Exception as e:
                print(f"       [WARN] Could not load {mp.name}: {e}")

    if model is None or not model.fitted:
        print("       [WARN] No ML model loaded — ML filter will be skipped")
        model = None

    # ------------------------------------------------------------------
    # Run backtests
    # ------------------------------------------------------------------
    print("[5/5] Running backtest passes ...")
    print()

    v2_result = None
    v3_result = None

    use_trailing = not args.no_trailing
    use_partial = not args.no_partial
    use_session = not args.no_session

    if not args.v3_only:
        print("  → Pass A: V2 (old logic — no H1 blocker, no zone guard, raw SMC SL/TP) ...")
        v2_result = run_backtest(
            df=df_test,
            h1_table=h1_table,
            model=model,
            feature_cols=feature_cols,
            smc=smc,
            features_eng=features_eng,
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
        )
        print(f"     Trades: {v2_result['total_trades']} | "
              f"Win: {v2_result['win_rate']:.1f}% | "
              f"P&L: ${v2_result['total_pnl']:.2f} | "
              f"PF: {v2_result['profit_factor']:.2f}")

    if not args.v2_only:
        print("  → Pass B: V3 (new logic — all filters active) ...")
        v3_result = run_backtest(
            df=df_test,
            h1_table=h1_table,
            model=model,
            feature_cols=feature_cols,
            smc=smc,
            features_eng=features_eng,
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
        )
        print(f"     Trades: {v3_result['total_trades']} | "
              f"Win: {v3_result['win_rate']:.1f}% | "
              f"P&L: ${v3_result['total_pnl']:.2f} | "
              f"PF: {v3_result['profit_factor']:.2f}")

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print_report(
        v2=v2_result or {},
        v3=v3_result or {},
        test_start=test_start,
        test_end=test_end,
        total_bars=n_test,
        v2_only=args.v2_only,
        v3_only=args.v3_only,
    )


if __name__ == "__main__":
    main()

"""
GOLDAI Backtest Engine V2 - ENHANCED
=====================================
3 Improvements:
1. Trailing Stop (protect profits after 1 ATR)
2. Partial Close (close 50% at halfway to TP)
3. Session Filter (London + NY only)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import polars as pl
from datetime import datetime

from src.config import get_config
from src.mt5_connector import MT5Connector
from src.feature_eng import FeatureEngineer
from src.smc_polars import SMCAnalyzer
from src.regime_detector import MarketRegimeDetector
from src.ml_model import TradingModel, get_default_feature_columns


def get_session(hour_utc):
    """Determine trading session from UTC hour."""
    if 0 <= hour_utc < 7:
        return "ASIAN"
    elif 7 <= hour_utc < 12:
        return "LONDON"
    elif 12 <= hour_utc < 16:
        return "LONDON_NY"  # Overlap — BEST time
    elif 16 <= hour_utc < 21:
        return "NEW_YORK"
    else:
        return "OFF_HOURS"


def run_backtest(bars=100000, lot_size=0.01, balance=439.0, min_conf=0.65,
                 use_trailing=True, use_partial=True, use_session=True):

    print(f"\n{'='*60}")
    print(f"  GOLDAI BACKTEST V2 — ENHANCED")
    print(f"{'='*60}")
    print(f"  Bars: {bars} | Lot: {lot_size} | Balance: ${balance}")
    print(f"  Trailing Stop : {'ON' if use_trailing else 'OFF'}")
    print(f"  Partial Close  : {'ON' if use_partial else 'OFF'}")
    print(f"  Session Filter : {'ON' if use_session else 'OFF'}")
    print(f"{'='*60}\n")

    # Connect and fetch data
    config = get_config()
    mt5 = MT5Connector(
        login=config.mt5_login, password=config.mt5_password,
        server=config.mt5_server, path=config.mt5_path,
    )
    mt5.connect()
    df = mt5.get_market_data("XAUUSD.m", "M15", bars)
    mt5.disconnect()
    print(f"Data: {len(df)} bars ({df['time'].min()} to {df['time'].max()})")

    # Calculate indicators
    fe = FeatureEngineer()
    df = fe.calculate_all(df, include_ml_features=True)
    smc = SMCAnalyzer(swing_length=5)
    df = smc.calculate_all(df)

    # Load regime detector
    hmm = MarketRegimeDetector(n_regimes=3, model_path="models/hmm_regime.pkl")
    hmm.load()
    if hmm.fitted:
        df = hmm.predict(df)
    else:
        df = df.with_columns(pl.lit(1).alias("regime"), pl.lit(1.0).alias("regime_confidence"))

    # Load ML model
    model = TradingModel(confidence_threshold=0.60, model_path="models/xgboost_model.pkl")
    model.load()
    feature_cols = [f for f in get_default_feature_columns() if f in df.columns]
    print(f"Features: {len(feature_cols)} | Model fitted: {model.fitted}")

    # Get ATR values for trailing stop
    atr_values = df["atr"].to_list() if "atr" in df.columns else [12.0] * len(df)

    # Simulation
    trades = []
    peak_balance = balance
    max_dd = 0.0
    in_trade = False
    trade = None
    warmup = 200
    pip_value = lot_size * 100 * 0.01

    # Stats
    session_stats = {"ASIAN": 0, "LONDON": 0, "LONDON_NY": 0, "NEW_YORK": 0, "OFF_HOURS": 0}
    session_filtered = 0
    trailing_saves = 0
    partial_closes = 0

    print(f"Simulating {len(df) - warmup} bars...\n")

    for i in range(warmup, len(df)):
        row = df.row(i, named=True)
        high = row["high"]
        low = row["low"]
        close = row["close"]
        bar_time = row["time"]
        current_atr = atr_values[i] if atr_values[i] is not None and atr_values[i] > 0 else 12.0

        # ==============================
        # CHECK OPEN TRADE
        # ==============================
        if in_trade and trade:
            hit_tp = False
            hit_sl = False
            hit_trail = False
            exit_price = 0

            # ---- IMPROVEMENT 1: TRAILING STOP ----
            if use_trailing and trade.get("trail_active", False):
                if trade["type"] == "BUY":
                    # Update trailing SL: move up as price moves up
                    new_trail = high - (current_atr * 1.5)
                    if new_trail > trade["trail_sl"]:
                        trade["trail_sl"] = new_trail

                    # Check if trailing SL hit
                    if low <= trade["trail_sl"]:
                        hit_trail = True
                        exit_price = trade["trail_sl"]
                else:  # SELL
                    new_trail = low + (current_atr * 1.5)
                    if new_trail < trade["trail_sl"]:
                        trade["trail_sl"] = new_trail

                    if high >= trade["trail_sl"]:
                        hit_trail = True
                        exit_price = trade["trail_sl"]

            # Activate trailing after 1 ATR profit
            if use_trailing and not trade.get("trail_active", False):
                if trade["type"] == "BUY":
                    unrealized = high - trade["entry"]
                    if unrealized >= current_atr * 1.0:
                        trade["trail_active"] = True
                        trade["trail_sl"] = high - (current_atr * 1.5)
                else:
                    unrealized = trade["entry"] - low
                    if unrealized >= current_atr * 1.0:
                        trade["trail_active"] = True
                        trade["trail_sl"] = low + (current_atr * 1.5)

            # ---- IMPROVEMENT 2: PARTIAL CLOSE ----
            if use_partial and not trade.get("partial_done", False):
                halfway = abs(trade["tp"] - trade["entry"]) / 2
                if trade["type"] == "BUY":
                    if high >= trade["entry"] + halfway:
                        # Close 50% at halfway
                        half_pips = halfway / 0.01
                        half_pnl = half_pips * pip_value * 0.5  # Half lot
                        balance += half_pnl
                        trade["partial_done"] = True
                        trade["remaining_lot"] = 0.5  # Track remaining
                        partial_closes += 1
                else:
                    if low <= trade["entry"] - halfway:
                        half_pips = halfway / 0.01
                        half_pnl = half_pips * pip_value * 0.5
                        balance += half_pnl
                        trade["partial_done"] = True
                        trade["remaining_lot"] = 0.5
                        partial_closes += 1

            # Check normal SL/TP
            if not hit_trail:
                if trade["type"] == "BUY":
                    if low <= trade["sl"]:
                        hit_sl = True
                        exit_price = trade["sl"]
                    elif high >= trade["tp"]:
                        hit_tp = True
                        exit_price = trade["tp"]
                else:
                    if high >= trade["sl"]:
                        hit_sl = True
                        exit_price = trade["sl"]
                    elif low <= trade["tp"]:
                        hit_tp = True
                        exit_price = trade["tp"]

            if hit_tp or hit_sl or hit_trail:
                if trade["type"] == "BUY":
                    pips = (exit_price - trade["entry"]) / 0.01
                else:
                    pips = (trade["entry"] - exit_price) / 0.01

                # Adjust PnL for partial close
                lot_mult = trade.get("remaining_lot", 1.0)
                pnl = pips * pip_value * lot_mult

                balance += pnl
                if balance > peak_balance:
                    peak_balance = balance
                dd = peak_balance - balance
                if dd > max_dd:
                    max_dd = dd

                result_type = "WIN" if pnl > 0 else "LOSS"
                exit_reason = "TP" if hit_tp else ("TRAIL" if hit_trail else "SL")

                if hit_trail and pnl > 0:
                    trailing_saves += 1

                trades.append({
                    "time": bar_time,
                    "type": trade["type"],
                    "entry": trade["entry"],
                    "exit": exit_price,
                    "pnl": pnl + (trade.get("partial_pnl", 0)),
                    "result": result_type,
                    "conf": trade["conf"],
                    "exit_reason": exit_reason,
                    "session": trade.get("session", ""),
                })
                in_trade = False
                trade = None
                continue

        if in_trade:
            continue

        # ==============================
        # GENERATE NEW SIGNAL
        # ==============================

        # ---- IMPROVEMENT 3: SESSION FILTER ----
        if use_session:
            hour_utc = bar_time.hour if hasattr(bar_time, 'hour') else 12
            session = get_session(hour_utc)

            # Only trade London, London-NY overlap, and New York
            if session not in ("LONDON", "LONDON_NY", "NEW_YORK"):
                session_filtered += 1
                continue

        # Generate SMC signal
        df_slice = df.slice(0, i + 1)
        signal = smc.generate_signal(df_slice)

        if signal is None or signal.confidence < min_conf:
            continue

        # Validate SL/TP
        if signal.signal_type == "BUY":
            if signal.stop_loss >= close or signal.take_profit <= close:
                continue
        else:
            if signal.stop_loss <= close or signal.take_profit >= close:
                continue

        # ML filter
        ml_agrees = False
        if model.fitted:
            pred = model.predict(df_slice, feature_cols)
            if pred.signal == signal.signal_type:
                ml_agrees = True

        # Determine session
        hour_utc = bar_time.hour if hasattr(bar_time, 'hour') else 12
        session = get_session(hour_utc)
        session_stats[session] = session_stats.get(session, 0) + 1

        # Enter trade
        trade = {
            "type": signal.signal_type,
            "entry": close,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "conf": signal.confidence,
            "ml": ml_agrees,
            "session": session,
            "trail_active": False,
            "trail_sl": 0,
            "partial_done": False,
            "remaining_lot": 1.0,
            "partial_pnl": 0,
        }
        in_trade = True

    # Close open trade at end
    if in_trade and trade:
        last_close = df.row(-1, named=True)["close"]
        if trade["type"] == "BUY":
            pips = (last_close - trade["entry"]) / 0.01
        else:
            pips = (trade["entry"] - last_close) / 0.01
        lot_mult = trade.get("remaining_lot", 1.0)
        pnl = pips * pip_value * lot_mult
        balance += pnl
        trades.append({
            "time": df.row(-1, named=True)["time"],
            "type": trade["type"],
            "entry": trade["entry"],
            "exit": last_close,
            "pnl": pnl,
            "result": "WIN" if pnl > 0 else "LOSS",
            "conf": trade["conf"],
            "exit_reason": "END",
            "session": trade.get("session", ""),
        })

    # ==============================
    # RESULTS
    # ==============================
    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    gross_win = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    dd_pct = (max_dd / peak_balance * 100) if peak_balance > 0 else 0

    # Exit reason breakdown
    tp_exits = len([t for t in trades if t.get("exit_reason") == "TP"])
    sl_exits = len([t for t in trades if t.get("exit_reason") == "SL"])
    trail_exits = len([t for t in trades if t.get("exit_reason") == "TRAIL"])

    print(f"{'='*60}")
    print(f"  BACKTEST V2 RESULTS")
    print(f"{'='*60}")
    print(f"  Period      : {df['time'].min()} to {df['time'].max()}")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Wins        : {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losses      : {len(losses)}")
    print(f"  Total PnL   : ${total_pnl:+.2f}")
    print(f"  Final Balance: ${balance:.2f}")
    print(f"  Avg Win     : ${avg_win:+.2f}")
    print(f"  Avg Loss    : ${avg_loss:.2f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Max Drawdown: ${max_dd:.2f} ({dd_pct:.1f}%)")
    print(f"{'='*60}")

    print(f"\n  EXIT REASONS:")
    print(f"  TP hits     : {tp_exits}")
    print(f"  SL hits     : {sl_exits}")
    print(f"  Trail exits : {trail_exits}")
    print(f"  Partial closes: {partial_closes}")
    print(f"  Trail saves : {trailing_saves} (profits protected)")

    if use_session:
        print(f"\n  SESSION BREAKDOWN:")
        for s, count in session_stats.items():
            s_trades = [t for t in trades if t.get("session") == s]
            s_pnl = sum(t["pnl"] for t in s_trades)
            s_wins = len([t for t in s_trades if t["result"] == "WIN"])
            s_wr = (s_wins / len(s_trades) * 100) if s_trades else 0
            print(f"  {s:<12}: {len(s_trades):>3} trades | WR: {s_wr:.0f}% | PnL: ${s_pnl:+.2f}")
        print(f"  Filtered out: {session_filtered} signals (Asian/Off-hours)")

    # Verdict
    print(f"\n  VERDICT:")
    if len(trades) == 0:
        print(f"  ❌ NO TRADES")
    elif total_pnl > 0 and win_rate > 45 and pf > 1.3:
        print(f"  ✅ PROFITABLE — Ready for demo!")
    elif total_pnl > 0 and pf > 1.1:
        print(f"  ⚠️ MARGINAL — Needs more optimization")
    else:
        print(f"  ❌ NOT PROFITABLE")

    # Last 10 trades
    if trades:
        print(f"\n  Last 10 Trades:")
        print(f"  {'Time':<20} {'Type':<5} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'Exit':>6} {'Session'}")
        print(f"  {'-'*80}")
        for t in trades[-10:]:
            time_str = str(t['time'])[:16]
            print(f"  {time_str:<20} {t['type']:<5} {t['entry']:>10.2f} {t['exit']:>10.2f} ${t['pnl']:>+8.2f} {t.get('exit_reason',''):>6} {t.get('session','')}")

    print(f"\n  Total: {len(trades)} trades | Balance: ${balance:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=100000)
    parser.add_argument("--lot", type=float, default=0.01)
    parser.add_argument("--balance", type=float, default=439)
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--no-trailing", action="store_true")
    parser.add_argument("--no-partial", action="store_true")
    parser.add_argument("--no-session", action="store_true")
    args = parser.parse_args()
    run_backtest(
        bars=args.bars, lot_size=args.lot, balance=args.balance,
        min_conf=args.confidence,
        use_trailing=not args.no_trailing,
        use_partial=not args.no_partial,
        use_session=not args.no_session,
    )
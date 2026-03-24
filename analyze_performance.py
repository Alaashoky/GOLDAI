"""Trading performance analysis.

Loads trade history from data/trades/, calculates comprehensive
performance metrics, and prints equity curve statistics plus win
rate breakdowns by session, trade direction, and market regime.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

try:
    from src.risk_metrics import RiskMetrics
    RISK_OK = True
except ImportError:
    RISK_OK = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_trades(data_dir: str = "data/trades") -> List[dict]:
    """Load all closed trade records from CSV files.

    Args:
        data_dir: Directory containing trade log CSV files.

    Returns:
        List of trade dicts, sorted by entry_time ascending.
    """
    trades: List[dict] = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    profit_str = row.get("profit", "")
                    if profit_str:
                        try:
                            row["profit"] = float(profit_str)
                            trades.append(row)
                        except ValueError:
                            pass
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)

    trades.sort(key=lambda r: r.get("entry_time", ""))
    return trades


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _equity_stats(profits: "np.ndarray") -> dict:
    """Compute basic equity curve statistics.

    Args:
        profits: Array of per-trade profits.

    Returns:
        Dict with total, min, max, std, cumulative high, and end equity.
    """
    equity = np.cumsum(profits)
    return {
        "total_pnl": float(profits.sum()),
        "std": float(profits.std()),
        "best_trade": float(profits.max()),
        "worst_trade": float(profits.min()),
        "peak_equity": float(equity.max()),
        "final_equity": float(equity[-1]),
    }


def _win_rate_by(trades: List[dict], key: str) -> Dict[str, dict]:
    """Compute win rate and average profit per group value.

    Args:
        trades: List of trade records.
        key: Field name to group by (e.g. 'session', 'direction').

    Returns:
        Dict mapping group value → stats dict.
    """
    groups: Dict[str, List[float]] = defaultdict(list)
    for t in trades:
        val = t.get(key) or "unknown"
        groups[val].append(t["profit"])

    result: Dict[str, dict] = {}
    for val, profits in groups.items():
        arr = profits
        wins = sum(1 for p in arr if p > 0)
        result[val] = {
            "count": len(arr),
            "win_rate": wins / len(arr),
            "avg_profit": sum(arr) / len(arr),
            "total": sum(arr),
        }
    return dict(sorted(result.items()))


def _monthly_pnl(trades: List[dict]) -> Dict[str, float]:
    """Aggregate profit by calendar month.

    Args:
        trades: List of trade records.

    Returns:
        Dict mapping 'YYYY-MM' strings to total profit.
    """
    monthly: Dict[str, float] = defaultdict(float)
    for t in trades:
        entry = t.get("entry_time", "")
        if entry:
            month = entry[:7]  # YYYY-MM
            monthly[month] += t["profit"]
    return dict(sorted(monthly.items()))


def _streaks(profits: "np.ndarray") -> tuple[int, int]:
    """Find the longest winning and losing streaks.

    Args:
        profits: Array of per-trade profits.

    Returns:
        Tuple of (max_win_streak, max_loss_streak).
    """
    max_win = max_loss = cur_win = cur_loss = 0
    for p in profits:
        if p > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(trades: List[dict]) -> None:
    """Print the full performance report to stdout.

    Args:
        trades: Loaded trade records.
    """
    if not trades:
        print("No trade data found.")
        return

    profits = np.array([t["profit"] for t in trades])
    eq_stats = _equity_stats(profits)
    wins = sum(1 for p in profits if p > 0)
    losses = len(profits) - wins
    win_rate = wins / len(profits)
    avg_win = float(profits[profits > 0].mean()) if wins > 0 else 0.0
    avg_loss = float(profits[profits < 0].mean()) if losses > 0 else 0.0
    expectancy = float(profits.mean())
    profit_factor = (avg_win * wins / abs(avg_loss * losses)) if (avg_loss != 0 and losses > 0) else float("nan")
    max_win_streak, max_loss_streak = _streaks(profits)

    rm = RiskMetrics() if RISK_OK else None
    sharpe = rm.calculate_sharpe(profits.tolist()) if rm else float("nan")
    sortino = rm.calculate_sortino(profits.tolist()) if rm else float("nan")
    max_dd_result = rm.calculate_max_drawdown(profits.tolist()) if rm else {"max_drawdown": float("nan")}
    max_dd = max_dd_result["max_drawdown"] if isinstance(max_dd_result, dict) else float(max_dd_result)
    var_95 = rm.calculate_var(profits.tolist(), confidence=0.95) if rm else float("nan")
    cvar_95 = rm.calculate_cvar(profits.tolist(), confidence=0.95) if rm else float("nan")
    calmar = rm.calculate_calmar(profits.tolist(), max_dd) if rm else float("nan")

    print(f"\n{'=' * 62}")
    print("  GOLDAI Performance Report")
    print(f"  Generated: {date.today().isoformat()}")
    print(f"{'=' * 62}")
    print(f"  Total Trades    : {len(trades)}")
    print(f"  Wins / Losses   : {wins} / {losses}")
    print(f"  Win Rate        : {win_rate:.1%}")
    print(f"  Expectancy      : {expectancy:+.2f}")
    print(f"  Avg Win         : {avg_win:+.2f}")
    print(f"  Avg Loss        : {avg_loss:+.2f}")
    if not (profit_factor != profit_factor):
        print(f"  Profit Factor   : {profit_factor:.2f}")
    print(f"  Total P&L       : {eq_stats['total_pnl']:+.2f}")
    print(f"  Best Trade      : {eq_stats['best_trade']:+.2f}")
    print(f"  Worst Trade     : {eq_stats['worst_trade']:+.2f}")
    print(f"  Peak Equity     : {eq_stats['peak_equity']:+.2f}")
    print(f"  Win Streak      : {max_win_streak}")
    print(f"  Loss Streak     : {max_loss_streak}")
    print(f"\n  --- Risk Metrics ---")
    print(f"  Sharpe Ratio    : {sharpe:.3f}")
    print(f"  Sortino Ratio   : {sortino:.3f}")
    print(f"  Calmar Ratio    : {calmar:.3f}")
    print(f"  Max Drawdown    : {max_dd:.2f}")
    print(f"  VaR (95%)       : {var_95:.2f}")
    print(f"  CVaR (95%)      : {cvar_95:.2f}")

    # Breakdown tables
    for group_key, label in [("direction", "Direction"), ("session", "Session"), ("regime", "Regime")]:
        breakdown = _win_rate_by(trades, group_key)
        if breakdown:
            print(f"\n  --- Win Rate by {label} ---")
            print(f"  {'Group':<18} {'Trades':>7} {'Win%':>7} {'Avg':>8} {'Total':>10}")
            print(f"  {'─' * 54}")
            for grp, stats in breakdown.items():
                print(
                    f"  {grp:<18}"
                    f" {stats['count']:>7}"
                    f" {stats['win_rate']:>6.1%}"
                    f" {stats['avg_profit']:>+8.2f}"
                    f" {stats['total']:>+10.2f}"
                )

    # Monthly P&L
    monthly = _monthly_pnl(trades)
    if monthly:
        print(f"\n  --- Monthly P&L ---")
        for month, total in monthly.items():
            bar = "█" * min(int(abs(total) / 10), 30)
            sign = "+" if total >= 0 else "-"
            print(f"  {month}  {sign}{abs(total):>8.2f}  {bar}")

    print(f"\n{'=' * 62}\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Analyse GOLDAI trading performance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/trades", help="Trade log directory.")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    if not NUMPY_OK:
        print("[ERROR] numpy is required.")
        sys.exit(1)

    args = parse_args()
    trades = _load_trades(args.data_dir)
    print_report(trades)


if __name__ == "__main__":
    main()

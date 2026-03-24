"""Generate a comprehensive risk report from GOLDAI trade history.

Loads closed trade records from data/trades/, computes all standard
risk metrics (Sharpe, Sortino, Calmar, VaR, CVaR, max drawdown, win
rate, expectancy) and prints a formatted report.  Optionally saves the
report to a CSV file.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import date
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
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


def _load_trades(data_dir: str = "data/trades") -> List[dict]:
    """Load all closed trade records from CSV files.

    Args:
        data_dir: Directory containing trade CSV files.

    Returns:
        List of trade dicts with at least a 'profit' key.
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
    return trades


def _equity_curve(trades: List[dict]) -> "np.ndarray":
    """Build cumulative equity curve from trade profits.

    Args:
        trades: List of trade dicts containing 'profit'.

    Returns:
        Numpy array of cumulative profit values.
    """
    profits = [t["profit"] for t in trades]
    return np.cumsum(profits)


def _win_rate(trades: List[dict]) -> float:
    """Compute overall win rate.

    Args:
        trades: List of trade dicts.

    Returns:
        Win rate in [0, 1].
    """
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t["profit"] > 0)
    return wins / len(trades)


def _expectancy(trades: List[dict]) -> float:
    """Compute trade expectancy (mean profit per trade).

    Args:
        trades: List of trade dicts.

    Returns:
        Average profit per trade.
    """
    if not trades:
        return 0.0
    return sum(t["profit"] for t in trades) / len(trades)


def _group_by(trades: List[dict], key: str) -> dict:
    """Group trades by a field value and compute per-group win rate.

    Args:
        trades: Trade records.
        key: Field name to group by.

    Returns:
        Dict mapping group value to win rate float.
    """
    groups: dict = {}
    for t in trades:
        val = t.get(key, "unknown")
        groups.setdefault(val, []).append(t["profit"])
    return {k: (sum(1 for p in v if p > 0) / len(v)) for k, v in groups.items() if v}


def print_report(trades: List[dict], output_csv: str | None = None) -> None:
    """Print the full risk report and optionally save to CSV.

    Args:
        trades: Loaded trade records.
        output_csv: Optional path to save the report as CSV.
    """
    if not trades:
        print("No trade data found.")
        return

    profits = np.array([t["profit"] for t in trades])
    equity = _equity_curve(trades)
    rm = RiskMetrics() if RISK_OK else None

    total_pnl = float(profits.sum())
    win_rt = _win_rate(trades)
    exp = _expectancy(trades)
    total_trades = len(trades)

    sharpe = rm.calculate_sharpe(profits.tolist()) if rm else float("nan")
    sortino = rm.calculate_sortino(profits.tolist()) if rm else float("nan")
    max_dd_result = rm.calculate_max_drawdown(profits.tolist()) if rm else {"max_drawdown": float("nan")}
    max_dd = max_dd_result["max_drawdown"] if isinstance(max_dd_result, dict) else float(max_dd_result)
    var_95 = rm.calculate_var(profits.tolist(), confidence=0.95) if rm else float("nan")
    cvar_95 = rm.calculate_cvar(profits.tolist(), confidence=0.95) if rm else float("nan")
    calmar = rm.calculate_calmar(profits.tolist(), max_dd) if rm else float("nan")

    avg_win = float(profits[profits > 0].mean()) if (profits > 0).any() else 0.0
    avg_loss = float(profits[profits < 0].mean()) if (profits < 0).any() else 0.0
    profit_factor = abs(avg_win / avg_loss) * win_rt / (1 - win_rt) if (avg_loss != 0 and win_rt not in (0, 1)) else float("nan")

    print(f"\n{'=' * 60}")
    print("  GOLDAI Risk Report")
    print(f"  Generated: {date.today().isoformat()}")
    print(f"{'=' * 60}")
    print(f"  Total Trades    : {total_trades}")
    print(f"  Total P&L       : {total_pnl:+.2f}")
    print(f"  Win Rate        : {win_rt:.1%}")
    print(f"  Expectancy      : {exp:+.2f} per trade")
    print(f"  Avg Win         : {avg_win:+.2f}")
    print(f"  Avg Loss        : {avg_loss:+.2f}")
    print(f"  Profit Factor   : {profit_factor:.2f}" if not (profit_factor != profit_factor) else "  Profit Factor   : N/A")
    print(f"\n  --- Risk Metrics ---")
    print(f"  Sharpe Ratio    : {sharpe:.3f}")
    print(f"  Sortino Ratio   : {sortino:.3f}")
    print(f"  Calmar Ratio    : {calmar:.3f}")
    print(f"  Max Drawdown    : {max_dd:.2f}")
    print(f"  VaR (95%)       : {var_95:.2f}")
    print(f"  CVaR (95%)      : {cvar_95:.2f}")

    # Win rate by session / direction
    for group_key, label in [("session", "Session"), ("direction", "Direction")]:
        grp = _group_by(trades, group_key)
        if grp:
            print(f"\n  Win Rate by {label}:")
            for k, wr in sorted(grp.items()):
                print(f"    {k:<15}: {wr:.1%}")

    print(f"\n{'=' * 60}\n")

    if output_csv:
        rows = [
            {"metric": "total_trades", "value": total_trades},
            {"metric": "total_pnl", "value": total_pnl},
            {"metric": "win_rate", "value": win_rt},
            {"metric": "expectancy", "value": exp},
            {"metric": "sharpe", "value": sharpe},
            {"metric": "sortino", "value": sortino},
            {"metric": "calmar", "value": calmar},
            {"metric": "max_drawdown", "value": max_dd},
            {"metric": "var_95", "value": var_95},
            {"metric": "cvar_95", "value": cvar_95},
        ]
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Report saved to {output_csv}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate GOLDAI risk report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/trades", help="Trade log directory.")
    parser.add_argument("--output", default=None, help="Save report to CSV file.")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    if not NUMPY_OK:
        print("[ERROR] numpy is required.")
        sys.exit(1)

    args = parse_args()
    trades = _load_trades(args.data_dir)
    print_report(trades, output_csv=args.output)


if __name__ == "__main__":
    main()

"""View current open XAUUSD positions in MetaTrader 5.

Prints a formatted table with ticket, direction, entry price, SL, TP,
current profit, and a summary of total P&L and exposure.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    from src.config import get_settings
    from src.mt5_connector import MT5Connector
    IMPORTS_OK = True
except ImportError as _ie:
    print(f"[ERROR] Import failed: {_ie}")
    IMPORTS_OK = False


def _fmt(val: float, decimals: int = 2) -> str:
    """Format float to fixed decimal string.

    Args:
        val: Float value to format.
        decimals: Number of decimal places.

    Returns:
        Formatted string.
    """
    return f"{val:.{decimals}f}"


def main() -> None:
    """Print all open XAUUSD positions to stdout."""
    if not IMPORTS_OK:
        sys.exit(1)

    settings = get_settings()
    connector = MT5Connector(
        login=settings.mt5_login,
        password=settings.mt5_password,
        server=settings.mt5_server,
        path=settings.mt5_path,
    )

    if not connector.connect():
        print("[ERROR] Could not connect to MT5.")
        sys.exit(1)

    try:
        positions = connector.get_positions(settings.symbol)

        print(f"\n{'=' * 80}")
        print(f"  Open Positions — {settings.symbol}")
        print(f"{'=' * 80}")

        if not positions:
            print("  No open positions.\n")
            return

        col_w = [10, 6, 12, 12, 12, 12, 10]
        header = (
            f"{'Ticket':<{col_w[0]}}"
            f"{'Dir':<{col_w[1]}}"
            f"{'Entry':>{col_w[2]}}"
            f"{'SL':>{col_w[3]}}"
            f"{'TP':>{col_w[4]}}"
            f"{'Profit':>{col_w[5]}}"
            f"{'Lots':>{col_w[6]}}"
        )
        print(f"  {header}")
        print(f"  {'-' * sum(col_w)}")

        total_profit = 0.0
        total_lots = 0.0

        for pos in positions:
            ticket = str(getattr(pos, "ticket", "?"))
            direction = getattr(pos, "type", "?")
            if hasattr(direction, "name"):
                direction = "BUY" if direction.name in ("POSITION_TYPE_BUY", "BUY") else "SELL"
            entry = _fmt(getattr(pos, "price_open", 0.0))
            sl = _fmt(getattr(pos, "sl", 0.0))
            tp = _fmt(getattr(pos, "tp", 0.0))
            profit = getattr(pos, "profit", 0.0)
            lots = getattr(pos, "volume", 0.0)
            profit_str = f"{'+' if profit >= 0 else ''}{_fmt(profit)}"

            row = (
                f"{ticket:<{col_w[0]}}"
                f"{direction:<{col_w[1]}}"
                f"{entry:>{col_w[2]}}"
                f"{sl:>{col_w[3]}}"
                f"{tp:>{col_w[4]}}"
                f"{profit_str:>{col_w[5]}}"
                f"{_fmt(lots, 2):>{col_w[6]}}"
            )
            print(f"  {row}")
            total_profit += profit
            total_lots += lots

        print(f"  {'-' * sum(col_w)}")
        pnl_str = f"{'+' if total_profit >= 0 else ''}{_fmt(total_profit)}"
        print(f"  {'TOTAL':<{col_w[0]}} {'':>{col_w[1]}} {'':>{col_w[2]}} {'':>{col_w[3]}} {'':>{col_w[4]}} {pnl_str:>{col_w[5]}} {_fmt(total_lots, 2):>{col_w[6]}}")

        account = connector.get_account_info()
        if account:
            exposure_pct = (total_lots / account.balance * 100) if account.balance > 0 else 0.0
            print(f"\n  Total P&L   : {pnl_str}")
            print(f"  Positions   : {len(positions)}")
            print(f"  Total Lots  : {_fmt(total_lots, 2)}")
            print(f"  Balance     : {_fmt(account.balance)}")
            print(f"  Equity      : {_fmt(account.equity)}")
            print(f"  Exposure    : {exposure_pct:.1f}%")

        print(f"\n{'=' * 80}\n")

    finally:
        connector.disconnect()


if __name__ == "__main__":
    main()

"""Check GOLDAI bot status.

Reads the last status from data/status.json (if it exists), prints last
run time, positions count, and daily P&L, then checks whether the bot
process is currently running.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STATUS_FILE = Path("data/status.json")
BOT_PROCESS_NAME = "main_live.py"


def _read_status() -> dict:
    """Read status JSON from disk.

    Returns:
        Parsed status dict or empty dict if the file does not exist.
    """
    if not STATUS_FILE.exists():
        return {}
    try:
        with open(STATUS_FILE) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read status file: %s", exc)
        return {}


def _is_bot_running() -> tuple[bool, int]:
    """Check whether the bot process is alive.

    Returns:
        Tuple of (is_running, pid). pid is 0 when not found.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["pgrep", "-f", BOT_PROCESS_NAME],
            capture_output=True,
            text=True,
        )
        pids = [int(p) for p in result.stdout.strip().split() if p.isdigit()]
        if pids:
            return True, pids[0]
    except Exception:  # noqa: BLE001
        pass
    return False, 0


def _minutes_since(iso_str: str) -> float | None:
    """Return minutes elapsed since an ISO 8601 timestamp string.

    Args:
        iso_str: ISO 8601 datetime string.

    Returns:
        Minutes elapsed or None if parsing fails.
    """
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return delta.total_seconds() / 60.0
    except (ValueError, TypeError):
        return None


def _read_daily_pnl() -> float | None:
    """Attempt to read today's P&L from the trade log CSV.

    Returns:
        Today's aggregate profit or None if not available.
    """
    try:
        from pathlib import Path
        import csv

        today = datetime.now(timezone.utc).date().isoformat()
        log_paths = sorted(Path("data/trades").glob("*.csv"))
        if not log_paths:
            return None

        daily = 0.0
        found = False
        with open(log_paths[-1], newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                entry_time = row.get("entry_time", "")
                if entry_time.startswith(today):
                    profit_str = row.get("profit", "")
                    if profit_str:
                        daily += float(profit_str)
                        found = True
        return daily if found else None
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    """Print GOLDAI bot status summary."""
    status = _read_status()
    running, pid = _is_bot_running()
    daily_pnl = _read_daily_pnl()

    print(f"\n{'=' * 50}")
    print("  GOLDAI Bot Status")
    print(f"{'=' * 50}")

    if running:
        print(f"  Process     : ✅ Running (PID {pid})")
    else:
        print("  Process     : ❌ Not running")

    last_run = status.get("last_run")
    if last_run:
        mins = _minutes_since(last_run)
        age_str = f"{mins:.1f} min ago" if mins is not None else "unknown"
        print(f"  Last Run    : {last_run} ({age_str})")
    else:
        print("  Last Run    : No status file found")

    positions = status.get("positions", "N/A")
    print(f"  Positions   : {positions}")

    if daily_pnl is not None:
        sign = "+" if daily_pnl >= 0 else ""
        print(f"  Daily P&L   : {sign}{daily_pnl:.2f}")
    else:
        print("  Daily P&L   : N/A")

    # Stale warning
    if last_run:
        mins = _minutes_since(last_run)
        if mins is not None and mins > 15 and not running:
            print(f"\n  ⚠️  Bot has not run for {mins:.0f} minutes and is not detected as running.")

    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()

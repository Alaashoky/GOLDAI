"""Monitor GOLDAI bot health.

Watches the bot log file for errors, alerts if the bot has not run
within the configured threshold, and prints live health metrics.
Can run continuously as a daemon with --daemon.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOG_FILE = Path("logs/goldai.log")
STATUS_FILE = Path("data/status.json")
BOT_PROCESS_NAME = "main_live.py"

_ERROR_PATTERN = re.compile(r"\b(ERROR|CRITICAL|Exception|Traceback)\b", re.IGNORECASE)


def _tail(path: Path, lines: int = 50) -> list[str]:
    """Return the last *lines* lines of a file.

    Args:
        path: File to tail.
        lines: Number of lines from the end.

    Returns:
        List of line strings (without newlines).
    """
    if not path.exists():
        return []
    try:
        with open(path) as fh:
            all_lines = fh.readlines()
            return [l.rstrip() for l in all_lines[-lines:]]
    except OSError:
        return []


def _is_bot_running() -> tuple[bool, int]:
    """Detect whether the bot process is alive.

    Returns:
        Tuple (is_running, pid).
    """
    try:
        import subprocess
        result = subprocess.run(["pgrep", "-f", BOT_PROCESS_NAME], capture_output=True, text=True)
        pids = [int(p) for p in result.stdout.strip().split() if p.isdigit()]
        if pids:
            return True, pids[0]
    except Exception:  # noqa: BLE001
        pass
    return False, 0


def _read_status() -> dict:
    """Read data/status.json.

    Returns:
        Parsed dict or empty dict.
    """
    import json
    if not STATUS_FILE.exists():
        return {}
    try:
        with open(STATUS_FILE) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _minutes_since_last_run(status: dict) -> float | None:
    """Return minutes elapsed since the bot last ran.

    Args:
        status: Status dict from _read_status().

    Returns:
        Minutes elapsed or None.
    """
    last_run = status.get("last_run")
    if not last_run:
        return None
    try:
        dt = datetime.fromisoformat(last_run)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except ValueError:
        return None


def _count_recent_errors(log_lines: list[str]) -> int:
    """Count lines matching the error pattern.

    Args:
        log_lines: Lines from the log file.

    Returns:
        Number of error-looking lines.
    """
    return sum(1 for line in log_lines if _ERROR_PATTERN.search(line))


def print_health(stale_threshold_minutes: float = 10.0) -> bool:
    """Print a health report and return True if all checks pass.

    Args:
        stale_threshold_minutes: Minutes without activity before alert.

    Returns:
        True if healthy, False if any alert condition triggered.
    """
    running, pid = _is_bot_running()
    status = _read_status()
    mins = _minutes_since_last_run(status)
    recent_lines = _tail(LOG_FILE, 100)
    error_count = _count_recent_errors(recent_lines)

    healthy = True

    print(f"\n{'─' * 52}")
    print(f"  GOLDAI Health Monitor  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'─' * 52}")
    print(f"  Process : {'✅ Running (PID ' + str(pid) + ')' if running else '❌ Stopped'}")

    if mins is not None:
        age = f"{mins:.1f} min ago"
    else:
        age = "unknown"
    print(f"  Last Run: {status.get('last_run', 'N/A')} ({age})")
    print(f"  Positions: {status.get('positions', 'N/A')}")
    print(f"  Errors (last 100 lines): {error_count}")

    if not running:
        print("  🔴 ALERT: Bot process is NOT running!")
        healthy = False

    if mins is not None and mins > stale_threshold_minutes and running:
        print(f"  🟡 WARNING: No activity for {mins:.0f} min (threshold={stale_threshold_minutes:.0f}).")
        healthy = False

    if error_count > 5:
        print(f"  🟡 WARNING: {error_count} errors in recent log lines.")
        healthy = False

    if error_count > 0:
        print("\n  --- Recent errors ---")
        for line in recent_lines:
            if _ERROR_PATTERN.search(line):
                print(f"  {line}")

    print(f"{'─' * 52}")
    return healthy


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Monitor GOLDAI bot health.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--daemon", action="store_true", help="Run continuously.")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds (daemon mode).")
    parser.add_argument("--stale-threshold", type=float, default=10.0, help="Minutes without activity to trigger alert.")
    return parser.parse_args()


def main() -> None:
    """Entry point for the health monitor."""
    args = parse_args()

    if args.daemon:
        logger.info("Running in daemon mode (interval=%ds) — press Ctrl+C to stop.", args.interval)
        try:
            while True:
                print_health(args.stale_threshold)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Monitor stopped.")
    else:
        ok = print_health(args.stale_threshold)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

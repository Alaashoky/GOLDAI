"""Check current XAUUSD market conditions.

Connects to MetaTrader 5, fetches the latest H1 data, runs SMC analysis,
and prints market structure, trend, key levels, and current session info.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from src.config import get_settings
    from src.mt5_connector import MT5Connector
    from src.smc_polars import SMCAnalyzer
    from src.session_filter import SessionFilter
    from src.feature_eng import compute_features
    from src.regime_detector import RegimeDetector
    IMPORTS_OK = True
except ImportError as _ie:
    print(f"[ERROR] Import failed: {_ie}")
    IMPORTS_OK = False


def main() -> None:
    """Print a snapshot of current XAUUSD market conditions."""
    if not IMPORTS_OK:
        sys.exit(1)

    settings = get_settings()
    symbol = settings.symbol

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
        # --- Current price ---
        tick = connector.get_tick(symbol)
        if tick is None:
            print(f"[ERROR] Could not get tick for {symbol}.")
            return
        print(f"\n{'=' * 55}")
        print(f"  GOLDAI Market Check — {symbol}  [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}]")
        print(f"{'=' * 55}")
        print(f"  Bid : {tick.bid:.2f}   Ask : {tick.ask:.2f}   Spread : {tick.spread:.2f}")

        # --- H1 data & features ---
        df = connector.get_ohlcv(symbol, "H1", count=100)
        if df is None or df.is_empty():
            print("[WARN] No H1 data available.")
            return

        # --- SMC analysis ---
        analyzer = SMCAnalyzer()
        smc = analyzer.analyze(df)
        print(f"\n  SMC Signal  : {smc.direction.value}  (confidence={smc.confidence:.2f})")
        print(f"  BOS         : {'Yes' if smc.bos_detected else 'No'}")
        print(f"  CHoCH       : {'Yes' if smc.choch_detected else 'No'}")
        print(f"  FVGs        : {len(smc.fvgs)}")
        print(f"  Order Blocks: {len(smc.order_blocks)}")

        if smc.order_blocks:
            ob = smc.order_blocks[-1]
            print(f"\n  Latest OB   : {ob.type.value}  High={ob.high:.2f}  Low={ob.low:.2f}")

        if smc.fvgs:
            fvg = smc.fvgs[-1]
            print(f"  Latest FVG  : {fvg.type.value}  Top={fvg.top:.2f}  Bottom={fvg.bottom:.2f}")

        # --- Regime ---
        import numpy as np
        regime_det = RegimeDetector()
        closes = df["close"].to_numpy()
        if len(closes) >= 50:
            regime_det.fit(closes)
            regime_result = regime_det.detect(closes)
            print(f"\n  Regime      : {regime_result.regime.name}  (vol={regime_result.volatility:.4f})")

        # --- Features quick summary ---
        feat_df = compute_features(df)
        if feat_df is not None and not feat_df.is_empty():
            last = feat_df.tail(1)
            for col in ("rsi_14", "atr_14", "adx_14"):
                if col in last.columns:
                    print(f"  {col:<12}: {last[col][0]:.2f}")

        # --- Session ---
        sf = SessionFilter()
        active = sf.get_active_sessions()
        allowed = sf.is_trading_allowed()
        print(f"\n  Sessions    : {', '.join(active) if active else 'Off-hours'}")
        print(f"  Trading OK  : {'✅ Yes' if allowed else '❌ No'}")

        print(f"\n{'=' * 55}\n")

    finally:
        connector.disconnect()


if __name__ == "__main__":
    main()

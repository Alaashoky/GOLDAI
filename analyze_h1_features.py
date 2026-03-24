"""H1 feature importance and correlation analysis.

Loads historical H1 data (from MT5 or a saved Parquet/CSV file),
computes features via the GOLDAI feature engineering pipeline,
calculates inter-feature correlations, and prints the top features
ranked by importance from a trained XGBoost model if one is available.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import numpy as np
    import polars as pl
    POLARS_OK = True
except ImportError:
    POLARS_OK = False

try:
    from src.config import get_settings
    from src.feature_eng import compute_features, FEATURE_NAMES
    from src.ml_model import TradingModel
    IMPORTS_OK = True
except ImportError as _ie:
    logger.warning("Import error: %s", _ie)
    IMPORTS_OK = False

try:
    from src.mt5_connector import MT5Connector
    MT5_IMPORT_OK = True
except ImportError:
    MT5_IMPORT_OK = False


def _load_from_file(path: str) -> "pl.DataFrame | None":
    """Load OHLCV data from a Parquet or CSV file.

    Args:
        path: Path to the file.

    Returns:
        Polars DataFrame or None on failure.
    """
    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", path)
        return None
    if p.suffix == ".parquet":
        return pl.read_parquet(path)
    if p.suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    logger.error("Unsupported file format: %s", p.suffix)
    return None


def _load_from_mt5(symbol: str, bars: int) -> "pl.DataFrame | None":
    """Fetch H1 data from MetaTrader 5.

    Args:
        symbol: Trading symbol.
        bars: Number of bars to fetch.

    Returns:
        Polars DataFrame or None on failure.
    """
    if not MT5_IMPORT_OK:
        logger.error("MT5 connector not available.")
        return None
    settings = get_settings()
    connector = MT5Connector(
        login=settings.mt5_login,
        password=settings.mt5_password,
        server=settings.mt5_server,
        path=settings.mt5_path,
    )
    if not connector.connect():
        logger.error("MT5 connection failed.")
        return None
    df = connector.get_ohlcv(symbol, "H1", count=bars)
    connector.disconnect()
    return df


def _print_correlations(feat_df: "pl.DataFrame", top_n: int = 20) -> None:
    """Print the features most highly correlated with 1-bar future returns.

    Args:
        feat_df: Feature DataFrame that must contain a 'close' column.
        top_n: Number of top correlations to display.
    """
    if "close" not in feat_df.columns:
        logger.warning("'close' column not in features — skipping correlation.")
        return

    closes = feat_df["close"].to_numpy().astype(float)
    returns = np.diff(closes, prepend=closes[0]) / np.where(closes != 0, closes, 1.0)

    exclude = {"time", "open", "high", "low", "close", "tick_volume"}
    feature_cols = [c for c in feat_df.columns if c not in exclude]

    corrs: list[tuple[str, float]] = []
    for col in feature_cols:
        try:
            vals = feat_df[col].to_numpy().astype(float)
            if np.std(vals) == 0:
                continue
            corr = float(np.corrcoef(vals, returns)[0, 1])
            corrs.append((col, corr))
        except Exception:  # noqa: BLE001
            pass

    corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'─' * 50}")
    print(f"  Top {top_n} Features by |Correlation with Returns|")
    print(f"{'─' * 50}")
    for name, corr in corrs[:top_n]:
        bar = "█" * int(abs(corr) * 30)
        sign = "+" if corr >= 0 else "-"
        print(f"  {name:<28} {sign}{abs(corr):.4f}  {bar}")


def _print_model_importance(model: "TradingModel", feat_df: "pl.DataFrame", top_n: int = 20) -> None:
    """Print XGBoost feature importances.

    Args:
        model: Trained TradingModel instance.
        feat_df: Feature DataFrame (used to get column names).
        top_n: Number of top features to show.
    """
    importance = model.get_feature_importance()
    if not importance:
        logger.info("Model has no feature importance data.")
        return

    sorted_imp = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)

    print(f"\n{'─' * 50}")
    print(f"  Top {top_n} XGBoost Feature Importances")
    print(f"{'─' * 50}")
    max_score = sorted_imp[0][1] if sorted_imp else 1.0
    for name, score in sorted_imp[:top_n]:
        bar = "█" * int(score / max_score * 30)
        print(f"  {name:<28} {score:.4f}  {bar}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="H1 feature importance and correlation analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", default=None, help="Path to saved OHLCV file (Parquet/CSV).")
    parser.add_argument("--bars", type=int, default=2000, help="Bars to fetch from MT5 if --file not given.")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to fetch.")
    parser.add_argument("--model", default="models/xgb_model.pkl", help="Trained model path.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top features to display.")
    return parser.parse_args()


def main() -> None:
    """Run H1 feature analysis."""
    if not IMPORTS_OK or not POLARS_OK:
        logger.error("Required modules unavailable.")
        sys.exit(1)

    args = parse_args()

    if args.file:
        df = _load_from_file(args.file)
    else:
        df = _load_from_mt5(args.symbol, args.bars)

    if df is None or df.is_empty():
        logger.error("No data loaded.")
        sys.exit(1)

    logger.info("Loaded %d rows. Computing features …", len(df))
    feat_df = compute_features(df)
    if feat_df is None or feat_df.is_empty():
        logger.error("Feature computation failed.")
        sys.exit(1)

    print(f"\n{'=' * 50}")
    print(f"  GOLDAI H1 Feature Analysis")
    print(f"  Rows: {len(feat_df)}   Features: {len(FEATURE_NAMES)}")
    print(f"{'=' * 50}")

    # Need close for correlation — merge back
    if "close" in df.columns and "close" not in feat_df.columns:
        feat_df = feat_df.with_columns(df["close"].alias("close"))

    _print_correlations(feat_df, args.top_n)

    # Model importance
    model_path = args.model
    if Path(model_path).exists():
        model = TradingModel(model_path=model_path)
        model.load(model_path)
        _print_model_importance(model, feat_df, args.top_n)
    else:
        logger.info("No trained model found at %s — skipping importance.", model_path)

    print()


if __name__ == "__main__":
    main()

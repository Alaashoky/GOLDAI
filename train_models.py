"""GOLDAI - Model Training Script.

Fetches historical XAUUSD data from MetaTrader 5, runs the full feature
engineering pipeline, creates labelled training data, and trains an
XGBoost model which is then saved to the models/ directory.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from src.config import get_settings
    from src.mt5_connector import MT5Connector
    from src.feature_eng import compute_features, create_labels
    from src.ml_model import TradingModel
    IMPORTS_OK = True
except ImportError as _ie:
    logger.warning("Import error: %s", _ie)
    IMPORTS_OK = False

try:
    import polars as pl
    POLARS_OK = True
except ImportError:
    POLARS_OK = False


def _setup_logging() -> None:
    """Configure basic console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def fetch_data(connector: "MT5Connector", symbol: str, bars: int) -> "pl.DataFrame | None":
    """Fetch H1 OHLCV data from MT5.

    Args:
        connector: Active MT5Connector instance.
        symbol: Trading symbol, e.g. "XAUUSD".
        bars: Number of historical bars to fetch.

    Returns:
        Polars DataFrame of OHLCV data or None if fetch failed.
    """
    logger.info("Fetching %d H1 bars for %s …", bars, symbol)
    df = connector.get_ohlcv(symbol, "H1", count=bars)
    if df is None or df.is_empty():
        logger.error("No data returned for %s.", symbol)
        return None
    logger.info("Fetched %d rows.", len(df))
    return df


def prepare_dataset(df: "pl.DataFrame", forward_bars: int = 5, move_threshold: float = 0.002) -> "tuple[object, object] | None":
    """Run feature engineering and label generation.

    Args:
        df: Raw OHLCV DataFrame.
        forward_bars: Number of future bars for label look-ahead.
        move_threshold: Minimum price move fraction to assign BUY/SELL label.

    Returns:
        Tuple of (X numpy array, y numpy array) or None on failure.
    """
    import numpy as np

    logger.info("Computing features …")
    features_df = compute_features(df)
    if features_df is None or features_df.is_empty():
        logger.error("Feature computation returned empty DataFrame.")
        return None

    logger.info("Creating labels (forward_bars=%d, threshold=%.4f) …", forward_bars, move_threshold)
    labels = create_labels(df, forward_bars=forward_bars, threshold=move_threshold)
    if labels is None or len(labels) == 0:
        logger.error("Label creation returned empty series.")
        return None

    exclude = {"time", "open", "high", "low", "close", "tick_volume"}
    feature_cols = [c for c in features_df.columns if c not in exclude]
    if not feature_cols:
        logger.error("No feature columns found.")
        return None

    X = features_df[feature_cols].to_numpy()
    y = labels.to_numpy() if hasattr(labels, "to_numpy") else labels

    # Align lengths (labels may have fewer rows due to look-ahead)
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    # Remove rows with NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y.astype(float)))
    X, y = X[mask], y[mask]

    logger.info("Dataset: %d samples, %d features.", len(X), X.shape[1])
    return X, y


def train(X: object, y: object, model_path: str) -> "TradingModel":
    """Train XGBoost model and return the trained instance.

    Args:
        X: Feature matrix (numpy array).
        y: Label vector (numpy array).
        model_path: Where to save the serialised model.

    Returns:
        Trained TradingModel instance.
    """
    from sklearn.model_selection import train_test_split

    logger.info("Splitting data into train/test (80/20) …")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = TradingModel(model_path=model_path)
    logger.info("Training XGBoost model …")
    model.fit(X_train, y_train)

    logger.info("Evaluating on test set …")
    metrics = model.evaluate(X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy  : {metrics.accuracy:.4f}")
    print(f"Precision : {metrics.precision:.4f}")
    print(f"Recall    : {metrics.recall:.4f}")
    print(f"F1 Score  : {metrics.f1:.4f}")
    print("-" * 60)
    print(metrics.report)

    # Feature importance
    importance = model.get_feature_importance()
    if importance:
        print("Top 10 Feature Importances:")
        sorted_imp = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for feat, score in sorted_imp:
            print(f"  {feat:<30s}: {score:.4f}")

    return model


def save_model(model: "TradingModel", output_path: str) -> None:
    """Persist the trained model to disk.

    Args:
        model: Trained TradingModel instance.
        output_path: Destination file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    logger.info("Model saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Train the GOLDAI XGBoost trading model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bars", type=int, default=5000, help="Number of H1 bars to fetch.")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol.")
    parser.add_argument("--output", type=str, default="models/xgb_model.pkl", help="Output model path.")
    parser.add_argument("--forward-bars", type=int, default=5, help="Look-ahead bars for label creation.")
    parser.add_argument("--threshold", type=float, default=0.002, help="Price-move threshold for BUY/SELL label.")
    return parser.parse_args()


def main() -> None:
    """Main entry point for model training."""
    _setup_logging()

    if not IMPORTS_OK or not POLARS_OK:
        logger.error("Required modules not available. Install dependencies first.")
        sys.exit(1)

    args = parse_args()
    settings = get_settings()

    # Override symbol from args if provided explicitly
    symbol = args.symbol or settings.symbol
    output = args.output or settings.model_path

    connector = MT5Connector(
        login=settings.mt5_login,
        password=settings.mt5_password,
        server=settings.mt5_server,
        path=settings.mt5_path,
    )

    logger.info("Connecting to MetaTrader 5 …")
    if not connector.connect():
        logger.error("MT5 connection failed.")
        sys.exit(1)

    try:
        df = fetch_data(connector, symbol, args.bars)
        if df is None:
            sys.exit(1)

        result = prepare_dataset(df, forward_bars=args.forward_bars, move_threshold=args.threshold)
        if result is None:
            sys.exit(1)

        X, y = result
        model = train(X, y, output)
        save_model(model, output)
        print(f"\n✅ Training complete — model saved to {output}")

    finally:
        connector.disconnect()


if __name__ == "__main__":
    main()

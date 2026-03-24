"""ML model evaluation and backtest analysis.

Loads a trained XGBoost model and evaluates it against historical data,
printing accuracy, precision, recall, F1, a confusion matrix, and
feature importance rankings.
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
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

try:
    import polars as pl
    POLARS_OK = True
except ImportError:
    POLARS_OK = False

try:
    from src.config import get_settings
    from src.feature_eng import compute_features, create_labels
    from src.ml_model import TradingModel, LABEL_MAP
    IMPORTS_OK = True
except ImportError as _ie:
    logger.warning("Import error: %s", _ie)
    IMPORTS_OK = False

try:
    from src.mt5_connector import MT5Connector
    MT5_IMPORT_OK = True
except ImportError:
    MT5_IMPORT_OK = False


def _load_data_mt5(symbol: str, bars: int) -> "pl.DataFrame | None":
    """Fetch H1 data from MetaTrader 5.

    Args:
        symbol: Trading symbol.
        bars: Number of bars.

    Returns:
        Polars DataFrame or None.
    """
    if not MT5_IMPORT_OK:
        logger.error("MT5 connector unavailable.")
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


def _load_data_file(path: str) -> "pl.DataFrame | None":
    """Load OHLCV from Parquet or CSV.

    Args:
        path: File path.

    Returns:
        Polars DataFrame or None.
    """
    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", path)
        return None
    return pl.read_parquet(path) if p.suffix == ".parquet" else pl.read_csv(path, try_parse_dates=True)


def _confusion_matrix_str(y_true: "np.ndarray", y_pred: "np.ndarray", labels: list[int]) -> str:
    """Build a text-based confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: List of unique label values.

    Returns:
        Formatted string representation.
    """
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    label_idx = {lbl: i for i, lbl in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            matrix[label_idx[t]][label_idx[p]] += 1

    label_names = [LABEL_MAP.get(l, str(l)) for l in labels]
    col_w = 10
    header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in label_names)
    rows = [header, "─" * (col_w * (n + 1))]
    for i, true_name in enumerate(label_names):
        row = f"{true_name:>{col_w}}" + "".join(f"{matrix[i][j]:>{col_w}}" for j in range(n))
        rows.append(row)
    return "\n".join(rows)


def evaluate(model: "TradingModel", X: "np.ndarray", y: "np.ndarray") -> None:
    """Run evaluation and print metrics.

    Args:
        model: Trained TradingModel.
        X: Feature matrix.
        y: True labels.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    preds_raw = model._model.predict(X) if model._model is not None else []
    if len(preds_raw) == 0:
        logger.error("Model produced no predictions.")
        return

    # Map numeric predictions back
    y_pred = np.array(preds_raw)
    y_true = np.array(y)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=list(LABEL_MAP.values()), zero_division=0)

    labels_present = sorted(list(set(y_true.tolist() + y_pred.tolist())))
    cm_str = _confusion_matrix_str(y_true, y_pred, labels_present)

    print(f"\n{'=' * 60}")
    print("  GOLDAI ML Model Evaluation")
    print(f"{'=' * 60}")
    print(f"  Samples     : {len(y_true)}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"\n  --- Classification Report ---")
    print(report)
    print(f"  --- Confusion Matrix (true↓  pred→) ---")
    print(cm_str)

    # Feature importance
    importance = model.get_feature_importance()
    if importance:
        print(f"\n  --- Top 15 Feature Importances ---")
        sorted_imp = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:15]
        max_score = sorted_imp[0][1] if sorted_imp else 1.0
        for feat, score in sorted_imp:
            bar = "█" * int(score / max_score * 25)
            print(f"  {feat:<28} {score:.4f}  {bar}")

    print(f"\n{'=' * 60}\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the GOLDAI XGBoost trading model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="models/xgb_model.pkl", help="Trained model path.")
    parser.add_argument("--file", default=None, help="OHLCV data file (Parquet/CSV).")
    parser.add_argument("--bars", type=int, default=2000, help="Bars from MT5 if no --file.")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to fetch.")
    parser.add_argument("--forward-bars", type=int, default=5, help="Look-ahead bars for labels.")
    parser.add_argument("--threshold", type=float, default=0.002, help="Price-move threshold for labels.")
    return parser.parse_args()


def main() -> None:
    """Entry point for model evaluation."""
    if not IMPORTS_OK or not NUMPY_OK or not POLARS_OK:
        logger.error("Required modules unavailable.")
        sys.exit(1)

    args = parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        logger.error("Model not found at %s.", model_path)
        sys.exit(1)

    model = TradingModel(model_path=model_path)
    model.load(model_path)
    logger.info("Model loaded from %s", model_path)

    if args.file:
        df = _load_data_file(args.file)
    else:
        df = _load_data_mt5(args.symbol, args.bars)

    if df is None or df.is_empty():
        logger.error("No data loaded.")
        sys.exit(1)

    logger.info("Computing features for %d bars …", len(df))
    feat_df = compute_features(df)
    if feat_df is None or feat_df.is_empty():
        sys.exit(1)

    labels = create_labels(df, forward_bars=args.forward_bars, threshold=args.threshold)
    if labels is None or len(labels) == 0:
        logger.error("Label creation failed.")
        sys.exit(1)

    exclude = {"time", "open", "high", "low", "close", "tick_volume"}
    feature_cols = [c for c in feat_df.columns if c not in exclude]
    X = feat_df[feature_cols].to_numpy()
    y = labels.to_numpy() if hasattr(labels, "to_numpy") else np.array(labels)

    min_len = min(len(X), len(y))
    X, y = X[:min_len], y[:min_len]

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y.astype(float)))
    X, y = X[mask], y[mask]

    evaluate(model, X, y)


if __name__ == "__main__":
    main()

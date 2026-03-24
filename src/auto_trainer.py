"""Automatic model retraining for the GOLDAI trading bot.

Monitors model performance, fetches fresh training data, retrains the
XGBoost model when needed, and manages model versioning with backups.
"""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from src.utils import utc_now

logger = logging.getLogger(__name__)

_DEFAULT_RETRAIN_HOURS = 24
_MIN_ACCURACY_THRESHOLD = 0.52
_MIN_BARS_FOR_TRAINING = 2000


@dataclass
class ModelMetrics:
    """Performance metrics for a trained model.

    Attributes:
        accuracy: Classification accuracy (0.0–1.0).
        precision: Precision score.
        recall: Recall score.
        f1: F1 score.
        train_time: UTC datetime when the model was trained.
        bars_used: Number of bars used in training.
        version: Semantic version string.
    """

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    train_time: datetime = field(default_factory=utc_now)
    bars_used: int = 0
    version: str = "0.0.0"


class AutoTrainer:
    """Orchestrates scheduled and performance-triggered model retraining.

    Fetches fresh OHLCV data, builds features, retrains the model, and
    swaps it in only when the new model outperforms the incumbent.

    Attributes:
        model_path: Path to the active pickled model file.
        backup_dir: Directory where model backups are stored.
        retrain_interval_hours: Hours between scheduled retrains.
        min_accuracy: Minimum required accuracy to accept a new model.
        symbol: Trading symbol used for data collection.
        timeframe: Timeframe used for training bars.
    """

    def __init__(
        self,
        model_path: str = "models/xgb_model.pkl",
        backup_dir: str = "models/backups",
        retrain_interval_hours: int = _DEFAULT_RETRAIN_HOURS,
        min_accuracy: float = _MIN_ACCURACY_THRESHOLD,
        symbol: str = "XAUUSD",
        timeframe: str = "H1",
    ) -> None:
        """Initialise the auto-trainer.

        Args:
            model_path: File path for the active model pickle.
            backup_dir: Directory where backup models are stored.
            retrain_interval_hours: Minimum hours between automatic retrains.
            min_accuracy: Accuracy floor below which a model is rejected.
            symbol: Symbol to fetch training data for.
            timeframe: OHLCV timeframe for training data.
        """
        self.model_path = model_path
        self.backup_dir = backup_dir
        self.retrain_interval_hours = retrain_interval_hours
        self.min_accuracy = min_accuracy
        self.symbol = symbol
        self.timeframe = timeframe
        self._last_retrain_time: Optional[datetime] = None
        self._retrain_count: int = 0
        os.makedirs(backup_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_retrain(
        self,
        last_train_time: Optional[datetime],
        performance_score: float,
    ) -> bool:
        """Determine whether a retrain is warranted.

        Retraining is triggered when:
        - No model exists yet (``last_train_time`` is None).
        - The scheduled interval has elapsed.
        - The model accuracy has fallen below the minimum threshold.

        Args:
            last_train_time: UTC datetime of the last successful retrain.
            performance_score: Current model accuracy (0.0–1.0).

        Returns:
            bool: True when retraining should be initiated.
        """
        if last_train_time is None:
            logger.info("No previous training time found — triggering initial retrain.")
            return True

        elapsed = utc_now() - last_train_time
        if elapsed >= timedelta(hours=self.retrain_interval_hours):
            logger.info("Scheduled retrain due: %.1f h since last train.", elapsed.total_seconds() / 3600)
            return True

        if performance_score < self.min_accuracy:
            logger.warning(
                "Model performance %.3f below threshold %.3f — triggering retrain.",
                performance_score, self.min_accuracy,
            )
            return True

        return False

    def get_training_data(self, connector: Any, bars: int = _MIN_BARS_FOR_TRAINING) -> Any:
        """Fetch historical OHLCV data for model training.

        Args:
            connector: MT5Connector (or compatible) instance.
            bars: Number of historical bars to retrieve.

        Returns:
            polars.DataFrame: OHLCV data frame.

        Raises:
            RuntimeError: If insufficient data is returned.
        """
        logger.info("Fetching %d bars of %s %s for training.", bars, self.symbol, self.timeframe)
        df = connector.get_ohlcv(self.symbol, self.timeframe, bars)
        if len(df) < bars // 2:
            raise RuntimeError(f"Insufficient training data: got {len(df)} bars, need ≥ {bars // 2}.")
        logger.info("Training data ready: %d rows.", len(df))
        return df

    def retrain(
        self,
        connector: Any,
        feature_eng: Any,
        model: Any,
        bars: int = _MIN_BARS_FOR_TRAINING,
    ) -> Tuple[bool, ModelMetrics]:
        """Fetch data, build features, retrain model, and swap if improved.

        Args:
            connector: MT5Connector instance for data retrieval.
            feature_eng: FeatureEngineer instance to build features.
            model: Current MLModel instance (will be retrained in-place if accepted).
            bars: Number of historical bars to use.

        Returns:
            Tuple[bool, ModelMetrics]: (success, metrics_of_new_model).
        """
        start = time.time()
        try:
            raw_df = self.get_training_data(connector, bars)
            features_df = feature_eng.generate_features(raw_df)

            logger.info("Starting model retrain with %d feature rows.", len(features_df))
            new_metrics = model.train(features_df)

            if not isinstance(new_metrics, ModelMetrics):
                new_metrics = ModelMetrics(
                    accuracy=float(new_metrics.get("accuracy", 0.0)) if hasattr(new_metrics, "get") else 0.0,
                    bars_used=len(features_df),
                    train_time=utc_now(),
                )

            elapsed = time.time() - start
            logger.info("Retrain complete in %.1fs — accuracy=%.3f", elapsed, new_metrics.accuracy)

            if new_metrics.accuracy >= self.min_accuracy:
                self.save_model_backup(model, version=f"v{self._retrain_count}")
                model.save(self.model_path)
                self._last_retrain_time = utc_now()
                self._retrain_count += 1
                return True, new_metrics

            logger.warning(
                "New model accuracy %.3f below threshold %.3f — keeping old model.",
                new_metrics.accuracy, self.min_accuracy,
            )
            return False, new_metrics

        except Exception as exc:
            logger.exception("Retrain failed: %s", exc)
            return False, ModelMetrics()

    def compare_models(
        self,
        old_metrics: ModelMetrics,
        new_metrics: ModelMetrics,
    ) -> bool:
        """Decide whether the new model should replace the incumbent.

        The new model is accepted when it achieves higher accuracy AND higher
        F1 score (or is the only option).

        Args:
            old_metrics: Performance of the current production model.
            new_metrics: Performance of the newly trained model.

        Returns:
            bool: True when the new model should be deployed.
        """
        if old_metrics.accuracy == 0.0:
            return True  # No incumbent to compare against

        acc_gain = new_metrics.accuracy - old_metrics.accuracy
        f1_gain = new_metrics.f1 - old_metrics.f1

        # Accept if either score improves by at least 0.5 pp
        accept = (acc_gain >= 0.005) or (f1_gain >= 0.005 and acc_gain >= 0.0)
        logger.info(
            "Model comparison: old_acc=%.3f new_acc=%.3f Δacc=%.3f Δf1=%.3f -> %s",
            old_metrics.accuracy, new_metrics.accuracy, acc_gain, f1_gain,
            "ACCEPT" if accept else "REJECT",
        )
        return accept

    def save_model_backup(self, model: Any, version: str = "") -> str:
        """Pickle a backup of the model before overwriting.

        Args:
            model: Model object with a ``save(path)`` method or pickle-able.
            version: Version tag appended to the backup filename.

        Returns:
            str: Full path to the backup file.
        """
        ts = utc_now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{version}" if version else ""
        backup_path = os.path.join(self.backup_dir, f"model_{ts}{tag}.pkl")

        if os.path.exists(self.model_path):
            shutil.copy2(self.model_path, backup_path)
            logger.info("Model backup saved: %s", backup_path)
        else:
            # Pickle the in-memory model directly
            try:
                with open(backup_path, "wb") as fh:
                    pickle.dump(model, fh)
                logger.info("In-memory model backup: %s", backup_path)
            except Exception as exc:
                logger.warning("Could not save model backup: %s", exc)

        return backup_path

    def get_last_retrain_time(self) -> Optional[datetime]:
        """Return the UTC datetime of the most recent successful retrain.

        Returns:
            Optional[datetime]: Last retrain time, or None if never retrained.
        """
        return self._last_retrain_time

    def get_retrain_count(self) -> int:
        """Return the total number of successful retrains in this session.

        Returns:
            int: Retrain count.
        """
        return self._retrain_count

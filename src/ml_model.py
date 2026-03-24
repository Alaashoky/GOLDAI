"""XGBoost trading model with versioning and evaluation.

Provides BUY/SELL/HOLD classification using XGBoost with
model persistence, feature importance, and performance metrics.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
REVERSE_MAP = {"SELL": 0, "HOLD": 1, "BUY": 2}

@dataclass
class PredictionResult:
    direction: str = "HOLD"
    confidence: float = 0.0
    probabilities: Dict[str, float] = None

    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = {"SELL": 0.0, "HOLD": 1.0, "BUY": 0.0}

@dataclass
class ModelMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    report: str = ""


class TradingModel:
    """XGBoost-based trading signal classifier.

    Attributes:
        model_path: Path to save/load the model.
        version: Model version string.
    """

    def __init__(self,
        model_path: str = "models/xgb_model.pkl",
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_child_weight: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ) -> None:
        self.model_path = model_path
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
        self._is_trained = False
        self._feature_names: List[str] = []

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> ModelMetrics:
        """Train the model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            feature_names: Feature names.

        Returns:
            ModelMetrics: Training evaluation metrics.
        """
        if feature_names:
            self._feature_names = feature_names

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self._model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self._is_trained = True
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")

        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        else:
            metrics = self.evaluate(X_train, y_train)

        logger.info(
            "Model trained v%s: acc=%.3f f1=%.3f",
            self.version, metrics.accuracy, metrics.f1,
        )
        return metrics

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Predict trading signal for a single sample.

        Args:
            X: Feature array (1 sample).

        Returns:
            PredictionResult: Prediction with confidence.
        """
        if not self._is_trained:
            return PredictionResult()

        if X.ndim == 1:
            X = X.reshape(1, -1)

        probas = self._model.predict_proba(X)[0]
        pred_class = int(np.argmax(probas))
        direction = LABEL_MAP.get(pred_class, "HOLD")
        confidence = float(probas[pred_class])

        return PredictionResult(
            direction=direction,
            confidence=round(confidence, 4),
            probabilities={
                "SELL": round(float(probas[0]), 4),
                "HOLD": round(float(probas[1]), 4),
                "BUY": round(float(probas[2]), 4),
            },
        )

    def predict_batch(self, X: np.ndarray) -> List[PredictionResult]:
        """Predict trading signals for multiple samples.

        Args:
            X: Feature array (N samples).

        Returns:
            List[PredictionResult]: List of predictions.
        """
        if not self._is_trained:
            return [PredictionResult() for _ in range(len(X))]

        probas = self._model.predict_proba(X)
        results = []
        for p in probas:
            pred_class = int(np.argmax(p))
            results.append(PredictionResult(
                direction=LABEL_MAP.get(pred_class, "HOLD"),
                confidence=round(float(p[pred_class]), 4),
                probabilities={
                    "SELL": round(float(p[0]), 4),
                    "HOLD": round(float(p[1]), 4),
                    "BUY": round(float(p[2]), 4),
                },
            ))
        return results

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Evaluate model performance.

        Args:
            X: Feature array.
            y: True labels.

        Returns:
            ModelMetrics: Evaluation metrics.
        """
        if not self._is_trained:
            return ModelMetrics()

        y_pred = self._model.predict(X)
        return ModelMetrics(
            accuracy=round(accuracy_score(y, y_pred), 4),
            precision=round(precision_score(y, y_pred, average="weighted", zero_division=0), 4),
            recall=round(recall_score(y, y_pred, average="weighted", zero_division=0), 4),
            f1=round(f1_score(y, y_pred, average="weighted", zero_division=0), 4),
            report=classification_report(y, y_pred, target_names=["SELL", "HOLD", "BUY"], zero_division=0),
        )

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get top feature importances.

        Args:
            top_n: Number of top features to return.

        Returns:
            List[Tuple[str, float]]: Feature name and importance pairs.
        """
        if not self._is_trained:
            return []

        importances = self._model.feature_importances_
        if self._feature_names and len(self._feature_names) == len(importances):
            names = self._feature_names
        else:
            names = [f"f_{{i}}" for i in range(len(importances))]

        pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return pairs[:top_n]

    def save(self, path: Optional[str] = None) -> str:
        """Save model to disk.

        Args:
            path: Override save path.

        Returns:
            str: Path where model was saved.
        """
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        data = {
            "model": self._model,
            "version": self.version,
            "feature_names": self._feature_names,
            "is_trained": self._is_trained,
        }
        joblib.dump(data, save_path)
        logger.info("Model saved to %s (v%s)", save_path, self.version)
        return save_path

    def load(self, path: Optional[str] = None) -> bool:
        """Load model from disk.

        Args:
            path: Override load path.

        Returns:
            bool: True if loaded successfully.
        """
        load_path = path or self.model_path
        if not os.path.exists(load_path):
            logger.warning("Model file not found: %s", load_path)
            return False
        try:
            data = joblib.load(load_path)
            self._model = data["model"]
            self.version = data.get("version", "unknown")
            self._feature_names = data.get("feature_names", [])
            self._is_trained = data.get("is_trained", True)
            logger.info("Model loaded from %s (v%s)", load_path, self.version)
            return True
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

"""Dynamic confidence scoring for trade signal aggregation.

Combines ML model probability, SMC signal strength, market regime,
session quality, and macro score into a single weighted confidence value.
Thresholds adapt automatically based on recent accuracy history.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default component weights (must sum to 1.0)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "ml":      0.35,
    "smc":     0.30,
    "regime":  0.15,
    "session": 0.10,
    "macro":   0.10,
}

# Regime quality scores for confidence weighting
REGIME_SCORES: Dict[str, float] = {
    "TRENDING": 1.0,
    "VOLATILE": 0.6,
    "RANGING":  0.5,
}


@dataclass
class ConfidenceRecord:
    """A single historical confidence prediction with its outcome.

    Attributes:
        score: Confidence score at time of prediction (0.0–1.0).
        was_correct: Whether the trade was profitable.
        timestamp: Unix timestamp of the prediction.
    """

    score: float
    was_correct: bool
    timestamp: float = 0.0


class DynamicConfidence:
    """Aggregates signals into a single trade confidence score.

    Weights follow the scheme: ML 35%, SMC 30%, Regime 15%,
    Session 10%, Macro 10%. Thresholds self-adjust based on recent
    prediction accuracy to avoid overtrading in low-accuracy regimes.

    Attributes:
        weights: Component weight dictionary.
        base_threshold: Default confidence threshold for trade entry.
        history_size: Number of historical records to maintain.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        base_threshold: float = 0.65,
        history_size: int = 100,
    ) -> None:
        """Initialise the confidence scorer.

        Args:
            weights: Optional custom weight dictionary. Must contain keys
                ml, smc, regime, session, macro summing to 1.0.
            base_threshold: Starting confidence threshold for high-confidence check.
            history_size: Rolling window size for accuracy tracking.
        """
        self.weights: Dict[str, float] = weights or DEFAULT_WEIGHTS.copy()
        self.base_threshold: float = base_threshold
        self._threshold: float = base_threshold
        self._history: Deque[ConfidenceRecord] = deque(maxlen=history_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_confidence(
        self,
        ml_prob: float,
        smc_signal: float,
        regime: str,
        session_quality: float,
        macro_score: float,
    ) -> float:
        """Compute the weighted confidence score from raw component values.

        Args:
            ml_prob: ML model win probability (0.0–1.0).
            smc_signal: SMC signal strength (0.0–1.0, where 1.0 is strongest).
            regime: Market regime label; e.g. "TRENDING", "RANGING", "VOLATILE".
            session_quality: Session suitability score (0.0–1.0).
            macro_score: Macro/news alignment score (0.0–1.0).

        Returns:
            float: Aggregated confidence score in [0.0, 1.0].
        """
        regime_score = REGIME_SCORES.get(regime, 0.5)

        # Clamp all inputs to [0, 1]
        components: Dict[str, float] = {
            "ml":      max(0.0, min(1.0, ml_prob)),
            "smc":     max(0.0, min(1.0, smc_signal)),
            "regime":  max(0.0, min(1.0, regime_score)),
            "session": max(0.0, min(1.0, session_quality)),
            "macro":   max(0.0, min(1.0, macro_score)),
        }

        confidence = sum(self.weights[k] * v for k, v in components.items())
        confidence = round(max(0.0, min(1.0, confidence)), 4)

        logger.debug(
            "Confidence=%.3f [ml=%.2f smc=%.2f regime=%s session=%.2f macro=%.2f]",
            confidence, ml_prob, smc_signal, regime, session_quality, macro_score,
        )
        return confidence

    def update_thresholds(self, recent_accuracy: float) -> None:
        """Adapt the confidence threshold based on recent signal accuracy.

        If recent accuracy is below 50 %, the threshold is raised by 5 pp to
        reduce trade frequency. If accuracy exceeds 65 %, the threshold is
        relaxed toward the base value.

        Args:
            recent_accuracy: Win rate fraction from the most recent N trades
                (0.0–1.0).
        """
        if recent_accuracy < 0.50:
            self._threshold = min(0.85, self._threshold + 0.05)
            logger.info("Threshold raised to %.2f (accuracy=%.1f%%)", self._threshold, recent_accuracy * 100)
        elif recent_accuracy > 0.65:
            self._threshold = max(self.base_threshold, self._threshold - 0.02)
            logger.info("Threshold relaxed to %.2f (accuracy=%.1f%%)", self._threshold, recent_accuracy * 100)

    def get_trade_confidence(self, signals_dict: Dict[str, Any]) -> float:
        """Derive a confidence score from a generic signals dictionary.

        Expected keys (all optional, defaulting to neutral values):
        - ``ml_probability`` (float): ML win probability.
        - ``smc_strength``   (float): SMC signal strength.
        - ``regime``         (str):   Market regime label.
        - ``session_score``  (float): Session quality.
        - ``macro_score``    (float): Macro/news alignment.

        Args:
            signals_dict: Mapping of signal names to values.

        Returns:
            float: Aggregated confidence score.
        """
        return self.calculate_confidence(
            ml_prob=float(signals_dict.get("ml_probability", 0.5)),
            smc_signal=float(signals_dict.get("smc_strength", 0.5)),
            regime=str(signals_dict.get("regime", "RANGING")),
            session_quality=float(signals_dict.get("session_score", 0.5)),
            macro_score=float(signals_dict.get("macro_score", 0.5)),
        )

    def is_high_confidence(self, confidence_score: float) -> bool:
        """Check whether a score meets the current adaptive threshold.

        Args:
            confidence_score: Score returned by ``calculate_confidence``.

        Returns:
            bool: True when the score exceeds the current threshold.
        """
        return confidence_score >= self._threshold

    def record_outcome(self, score: float, was_correct: bool, timestamp: float = 0.0) -> None:
        """Store a trade outcome for accuracy tracking.

        Args:
            score: Confidence score at signal time.
            was_correct: Whether the subsequent trade was profitable.
            timestamp: Unix timestamp of the record.
        """
        self._history.append(
            ConfidenceRecord(score=score, was_correct=was_correct, timestamp=timestamp)
        )
        accuracy = self.get_recent_accuracy()
        self.update_thresholds(accuracy)

    def get_recent_accuracy(self, n: int = 20) -> float:
        """Compute the win rate over the most recent N predictions.

        Args:
            n: Number of recent records to include.

        Returns:
            float: Win rate fraction (0.0–1.0). Returns 0.5 when insufficient data.
        """
        recent = list(self._history)[-n:]
        if not recent:
            return 0.5
        return sum(1 for r in recent if r.was_correct) / len(recent)

    def get_current_threshold(self) -> float:
        """Return the current adaptive threshold.

        Returns:
            float: Current threshold value.
        """
        return self._threshold

    def get_component_scores(
        self,
        ml_prob: float,
        smc_signal: float,
        regime: str,
        session_quality: float,
        macro_score: float,
    ) -> Dict[str, float]:
        """Return individual weighted component contributions for debugging.

        Args:
            ml_prob: ML win probability.
            smc_signal: SMC signal strength.
            regime: Market regime label.
            session_quality: Session quality score.
            macro_score: Macro alignment score.

        Returns:
            Dict[str, float]: Weighted contribution of each component.
        """
        regime_score = REGIME_SCORES.get(regime, 0.5)
        raw: Dict[str, float] = {
            "ml":      ml_prob,
            "smc":     smc_signal,
            "regime":  regime_score,
            "session": session_quality,
            "macro":   macro_score,
        }
        return {k: round(self.weights[k] * v, 4) for k, v in raw.items()}

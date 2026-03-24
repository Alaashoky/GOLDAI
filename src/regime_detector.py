"""HMM-based market regime detector.

Detects 3 regimes: TRENDING, RANGING, VOLATILE using
Gaussian HMM from hmmlearn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None
    logger.warning("hmmlearn not installed. Regime detection will use fallback.")


class Regime(IntEnum):
    TRENDING = 0
    RANGING = 1
    VOLATILE = 2


@dataclass
class RegimeResult:
    current_regime: Regime = Regime.RANGING
    regime_probability: float = 0.0
    regime_volatility: float = 0.0
    state_sequence: Optional[np.ndarray] = None


class RegimeDetector:
    """Hidden Markov Model based market regime detector.

    Uses returns and volatility as observable features to classify
    market conditions into TRENDING, RANGING, or VOLATILE states.

    Attributes:
        n_states: Number of HMM states (3).
        lookback: Number of bars for feature calculation.
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback: int = 100,
        n_iter: int = 100,
        vol_window: int = 20,
    ) -> None:
        self.n_states = n_states
        self.lookback = lookback
        self.vol_window = vol_window
        self._model: Optional[GaussianHMM] = None
        self._is_fitted = False

        if GaussianHMM is not None:
            self._model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=n_iter,
                random_state=42,
            )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _prepare_features(self, close: np.ndarray) -> np.ndarray:
        """Prepare features for HMM: returns and rolling volatility.

        Args:
            close: Close price array.

        Returns:
            np.ndarray: Feature matrix (n_samples, 2).
        """
        returns = np.diff(np.log(close + 1e-8))
        n = len(returns)
        volatility = np.zeros(n)
        for i in range(self.vol_window, n):
            volatility[i] = np.std(returns[i - self.vol_window:i])
        volatility[:self.vol_window] = volatility[self.vol_window] if n > self.vol_window else 0.01

        features = np.column_stack([returns, volatility])
        return features

    def _map_states(self, means: np.ndarray) -> dict:
        """Map HMM states to regime labels based on volatility.

        Args:
            means: State means from the HMM.

        Returns:
            dict: Mapping from HMM state index to Regime.
        """
        vol_means = means[:, 1] if means.shape[1] > 1 else np.abs(means[:, 0])
        sorted_idx = np.argsort(vol_means)
        mapping = {}
        mapping[sorted_idx[0]] = Regime.RANGING
        mapping[sorted_idx[1]] = Regime.TRENDING
        mapping[sorted_idx[2]] = Regime.VOLATILE
        return mapping

    def fit(self, close: np.ndarray) -> bool:
        """Fit the HMM on historical close prices.

        Args:
            close: Array of close prices.

        Returns:
            bool: True if fitting was successful.
        """
        if self._model is None:
            logger.warning("HMM not available, using fallback")
            self._is_fitted = True
            return True

        if len(close) < self.lookback:
            logger.warning("Not enough data for HMM: %d < %d", len(close), self.lookback)
            return False

        try:
            features = self._prepare_features(close[-self.lookback:])
            self._model.fit(features)
            self._is_fitted = True
            logger.info("HMM fitted on %d samples", len(features))
            return True
        except Exception as e:
            logger.error("HMM fit error: %s", e)
            return False

    def detect(self, close: np.ndarray) -> RegimeResult:
        """Detect current market regime.

        Args:
            close: Array of close prices.

        Returns:
            RegimeResult: Current regime information.
        """
        if not self._is_fitted:
            self.fit(close)

        if self._model is None or not self._is_fitted:
            return self._fallback_detect(close)

        try:
            features = self._prepare_features(close[-self.lookback:])
            states = self._model.predict(features)
            proba = self._model.predict_proba(features)

            mapping = self._map_states(self._model.means_)
            current_hmm_state = states[-1]
            current_regime = mapping.get(current_hmm_state, Regime.RANGING)
            current_prob = float(proba[-1][current_hmm_state])

            returns = np.diff(np.log(close[-self.vol_window:] + 1e-8))
            regime_vol = float(np.std(returns)) if len(returns) > 0 else 0.0

            return RegimeResult(
                current_regime=current_regime,
                regime_probability=round(current_prob, 4),
                regime_volatility=round(regime_vol, 6),
                state_sequence=np.array([mapping.get(s, Regime.RANGING) for s in states]),
            )
        except Exception as e:
            logger.error("Regime detect error: %s", e)
            return self._fallback_detect(close)

    def _fallback_detect(self, close: np.ndarray) -> RegimeResult:
        """Fallback regime detection using simple statistics.

        Args:
            close: Array of close prices.

        Returns:
            RegimeResult: Regime result.
        """
        if len(close) < 20:
            return RegimeResult()

        returns = np.diff(np.log(close[-50:] + 1e-8))
        vol = float(np.std(returns))
        trend = abs(float(np.mean(returns)))

        if vol > 0.02:
            regime = Regime.VOLATILE
        elif trend > 0.001:
            regime = Regime.TRENDING
        else:
            regime = Regime.RANGING

        return RegimeResult(
            current_regime=regime,
            regime_probability=0.6,
            regime_volatility=round(vol, 6),
        )
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Internal state for the 1-D Kalman filter.

    State vector: [position, velocity]

    Attributes:
        x: State estimate vector [position, velocity].
        P: Error covariance matrix (2×2).
        initialized: Whether the filter has been seeded with the first measurement.
    """

    x: np.ndarray = field(default_factory=lambda: np.zeros(2))
    P: np.ndarray = field(default_factory=lambda: np.eye(2) * 1000.0)
    initialized: bool = False


class KalmanFilter:
    """1-D Kalman filter for smoothing financial price series.

    Models the state as [price, velocity] using a constant-velocity kinematic
    model.  The filter separates genuine price movement from measurement noise,
    yielding a smooth estimate and a velocity term that reflects the trend.

    Args:
        process_noise_q: Process noise variance Q (model uncertainty).
        measurement_noise_r: Measurement noise variance R (price tick noise).
        dt: Time step between observations (default 1 bar).
    """

    def __init__(
        self,
        process_noise_q: float = 0.1,
        measurement_noise_r: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        self.process_noise_q = process_noise_q
        self.measurement_noise_r = measurement_noise_r
        self.dt = dt

        # State-transition matrix F (constant-velocity model)
        self._F = np.array([[1.0, dt], [0.0, 1.0]])

        # Observation matrix H – we only observe position
        self._H = np.array([[1.0, 0.0]])

        # Process noise covariance Q
        self._Q = self._build_Q()

        # Measurement noise covariance R (scalar)
        self._R = np.array([[measurement_noise_r]])

        self._state = KalmanState()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_Q(self) -> np.ndarray:
        """Build process noise covariance matrix Q.

        Returns:
            2×2 numpy array representing process covariance.
        """
        dt = self.dt
        q = self.process_noise_q
        return np.array(
            [
                [q * dt**3 / 3.0, q * dt**2 / 2.0],
                [q * dt**2 / 2.0, q * dt],
            ]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset filter to uninitialised state."""
        self._state = KalmanState()
        logger.debug("KalmanFilter reset")

    def update(self, price: float) -> float:
        """Process a new price observation and return the smoothed value.

        Args:
            price: Raw price measurement.

        Returns:
            Kalman-smoothed price estimate.
        """
        z = np.array([[price]])

        if not self._state.initialized:
            self._state.x = np.array([price, 0.0])
            self._state.P = np.eye(2) * 1000.0
            self._state.initialized = True
            return price

        # --- Predict ---
        x_pred = self._F @ self._state.x
        P_pred = self._F @ self._state.P @ self._F.T + self._Q

        # --- Update ---
        S = self._H @ P_pred @ self._H.T + self._R          # Innovation covariance
        K = P_pred @ self._H.T @ np.linalg.inv(S)           # Kalman gain
        y = z - self._H @ x_pred                             # Innovation
        self._state.x = x_pred + (K @ y).flatten()
        self._state.P = (np.eye(2) - K @ self._H) @ P_pred

        smoothed = float(self._state.x[0])
        logger.debug("Kalman update: raw=%.4f smoothed=%.4f vel=%.6f",
                     price, smoothed, self._state.x[1])
        return smoothed

    def get_trend(self) -> int:
        """Return estimated trend direction based on velocity.

        Returns:
            +1 for uptrend, -1 for downtrend, 0 for flat.
        """
        if not self._state.initialized:
            return 0
        velocity = float(self._state.x[1])
        threshold = 0.01 * max(abs(float(self._state.x[0])), 1.0) * 0.001
        if velocity > threshold:
            return 1
        if velocity < -threshold:
            return -1
        return 0

    def get_filtered_series(self, prices: list[float] | np.ndarray) -> np.ndarray:
        """Apply the Kalman filter to an entire price series.

        The filter is reset before processing so that previous state does not
        bleed into the batch result.

        Args:
            prices: Sequence of raw price observations.

        Returns:
            Numpy array of smoothed prices with the same length.
        """
        self.reset()
        prices_arr = np.asarray(prices, dtype=float)
        smoothed = np.empty_like(prices_arr)
        for i, p in enumerate(prices_arr):
            smoothed[i] = self.update(float(p))
        return smoothed

    @property
    def current_estimate(self) -> float:
        """Return current position estimate without a new observation.

        Returns:
            Current smoothed price or 0.0 if not initialised.
        """
        return float(self._state.x[0]) if self._state.initialized else 0.0

    @property
    def current_velocity(self) -> float:
        """Return current velocity estimate.

        Returns:
            Estimated rate of price change per bar.
        """
        return float(self._state.x[1]) if self._state.initialized else 0.0

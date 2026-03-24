from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.kalman_filter import KalmanFilter

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryResult:
    """Result from the trajectory predictor.

    Attributes:
        direction: Predicted direction (+1 up, -1 down, 0 flat).
        confidence: Confidence in prediction in [0, 1].
        price_target: Estimated target price for the horizon.
        momentum_magnitude: Magnitude of current momentum vector.
        horizon_bars: Number of bars used for the prediction horizon.
    """

    direction: int
    confidence: float
    price_target: float
    momentum_magnitude: float
    horizon_bars: int


class TrajectoryPredictor:
    """Short-term price trajectory predictor using Kalman smoothing.

    Combines a Kalman filter velocity estimate with a simple linear momentum
    vector to project likely price direction and target over a short horizon.

    Args:
        process_noise_q: Kalman filter process noise (default 0.1).
        measurement_noise_r: Kalman filter measurement noise (default 1.0).
        confidence_lookback: Bars used to assess prediction confidence (default 20).
    """

    def __init__(
        self,
        process_noise_q: float = 0.1,
        measurement_noise_r: float = 1.0,
        confidence_lookback: int = 20,
    ) -> None:
        self._kf = KalmanFilter(
            process_noise_q=process_noise_q,
            measurement_noise_r=measurement_noise_r,
        )
        self.confidence_lookback = confidence_lookback

    # ------------------------------------------------------------------
    # Momentum vector
    # ------------------------------------------------------------------

    def calculate_momentum_vector(
        self,
        prices: np.ndarray | list[float],
    ) -> tuple[int, float]:
        """Estimate current momentum direction and magnitude.

        Fits a linear regression over the Kalman-smoothed prices to extract
        the dominant trend slope, then normalises it relative to recent ATR.

        Args:
            prices: Array of close prices (minimum 5 bars).

        Returns:
            Tuple of (direction, magnitude) where direction is +1/0/-1 and
            magnitude is the absolute normalised slope.
        """
        arr = np.asarray(prices, dtype=float)
        if len(arr) < 5:
            return 0, 0.0

        smoothed = self._kf.get_filtered_series(arr)
        n = len(smoothed)
        x = np.arange(n, dtype=float)
        slope, _ = np.polyfit(x, smoothed, 1)

        # Normalise by recent ATR proxy (std of raw diffs)
        diffs = np.diff(arr[-20:]) if len(arr) >= 20 else np.diff(arr)
        atr_proxy = float(np.std(diffs)) if len(diffs) > 1 else 1.0
        if atr_proxy == 0.0:
            atr_proxy = 1.0

        normalised_slope = slope / atr_proxy

        if normalised_slope > 0.1:
            direction = 1
        elif normalised_slope < -0.1:
            direction = -1
        else:
            direction = 0

        return direction, round(abs(normalised_slope), 4)

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def get_trajectory_confidence(
        self,
        prices: np.ndarray | list[float],
    ) -> float:
        """Estimate confidence in the predicted trajectory.

        Confidence is derived from:
          1. R-squared of the linear fit to smoothed prices (trend clarity).
          2. Consistency of directional bars in the lookback window.

        Args:
            prices: Array of close prices.

        Returns:
            Confidence score in [0, 1].
        """
        arr = np.asarray(prices, dtype=float)
        lb = min(self.confidence_lookback, len(arr))
        if lb < 5:
            return 0.0

        recent = arr[-lb:]
        smoothed = self._kf.get_filtered_series(recent)

        # R-squared of linear fit
        x = np.arange(len(smoothed), dtype=float)
        coeffs = np.polyfit(x, smoothed, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = float(np.sum((smoothed - y_pred) ** 2))
        ss_tot = float(np.sum((smoothed - np.mean(smoothed)) ** 2))
        r_squared = 1.0 - ss_res / (ss_tot + 1e-10)
        r_squared = max(0.0, min(1.0, r_squared))

        # Directional consistency
        diffs = np.diff(recent)
        if coeffs[0] > 0:
            consistency = float(np.mean(diffs > 0))
        elif coeffs[0] < 0:
            consistency = float(np.mean(diffs < 0))
        else:
            consistency = 0.5

        confidence = 0.60 * r_squared + 0.40 * consistency
        return round(min(1.0, max(0.0, confidence)), 4)

    # ------------------------------------------------------------------
    # Target pricing
    # ------------------------------------------------------------------

    def get_price_target(
        self,
        current_price: float,
        direction: int,
        atr: float,
    ) -> float:
        """Estimate a price target based on direction and ATR.

        Args:
            current_price: Latest price.
            direction: +1 long, -1 short, 0 flat.
            atr: Average True Range for the instrument.

        Returns:
            Estimated target price.
        """
        if direction == 0 or atr <= 0:
            return current_price
        target = current_price + direction * atr * 1.5
        return round(target, 5)

    # ------------------------------------------------------------------
    # Trajectory prediction
    # ------------------------------------------------------------------

    def predict_trajectory(
        self,
        prices: np.ndarray | list[float],
        horizon_bars: int = 5,
    ) -> TrajectoryResult:
        """Predict price trajectory over the next N bars.

        Args:
            prices: Historical close price array (minimum 10 bars).
            horizon_bars: Number of bars to project forward (default 5).

        Returns:
            TrajectoryResult with direction, confidence and price target.
        """
        arr = np.asarray(prices, dtype=float)
        if len(arr) < 10:
            return TrajectoryResult(
                direction=0,
                confidence=0.0,
                price_target=float(arr[-1]) if len(arr) else 0.0,
                momentum_magnitude=0.0,
                horizon_bars=horizon_bars,
            )

        direction, magnitude = self.calculate_momentum_vector(arr)
        confidence = self.get_trajectory_confidence(arr)

        # Estimate ATR over last 14 bars
        if len(arr) >= 15:
            diffs = np.abs(np.diff(arr[-15:]))
            atr = float(np.mean(diffs))
        else:
            atr = float(np.std(arr)) or 1.0

        current_price = float(arr[-1])
        # Scale ATR by sqrt of horizon for multi-bar projection
        scaled_atr = atr * math.sqrt(horizon_bars)
        target = self.get_price_target(current_price, direction, scaled_atr)

        logger.debug(
            "Trajectory: dir=%d conf=%.3f target=%.4f mag=%.4f",
            direction, confidence, target, magnitude,
        )
        return TrajectoryResult(
            direction=direction,
            confidence=confidence,
            price_target=target,
            momentum_magnitude=magnitude,
            horizon_bars=horizon_bars,
        )

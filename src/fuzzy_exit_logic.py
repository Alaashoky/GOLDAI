from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FuzzyExitScore:
    """Result of fuzzy exit logic evaluation.

    Attributes:
        score: Exit score in range [0.0, 1.0].
        should_exit: Whether exit is recommended.
        reason: Human-readable explanation.
    """

    score: float
    should_exit: bool
    reason: str


class FuzzyExitLogic:
    """Fuzzy logic system for trade exit decisions.

    Uses triangular and trapezoidal membership functions to evaluate
    multiple inputs and produce a defuzzified exit score via the
    centroid method.
    """

    def __init__(self, exit_threshold: float = 0.55) -> None:
        """Initialize with configurable exit threshold.

        Args:
            exit_threshold: Score above which we recommend exit (default 0.55).
        """
        self.exit_threshold = exit_threshold

    # ------------------------------------------------------------------
    # Membership functions
    # ------------------------------------------------------------------

    def _trimf(self, x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function.

        Args:
            x: Input value.
            a: Left foot.
            b: Peak.
            c: Right foot.

        Returns:
            Membership degree in [0, 1].
        """
        if x <= a or x >= c:
            return 0.0
        if x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        return (c - x) / (c - b) if c != b else 1.0

    def _trapmf(self, x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function.

        Args:
            x: Input value.
            a: Left foot.
            b: Left shoulder (start of plateau).
            c: Right shoulder (end of plateau).
            d: Right foot.

        Returns:
            Membership degree in [0, 1].
        """
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if x < b:
            return (x - a) / (b - a) if b != a else 1.0
        return (d - x) / (d - c) if d != c else 1.0

    # ------------------------------------------------------------------
    # Per-input fuzzy sets
    # ------------------------------------------------------------------

    def _profit_membership(self, profit_pips: float) -> dict[str, float]:
        """Classify profit level into fuzzy sets.

        Args:
            profit_pips: Current profit in pips (can be negative).

        Returns:
            Dict with keys 'loss', 'small', 'moderate', 'high'.
        """
        return {
            "loss": self._trapmf(profit_pips, -200.0, -50.0, 0.0, 5.0),
            "small": self._trimf(profit_pips, 0.0, 10.0, 30.0),
            "moderate": self._trimf(profit_pips, 20.0, 50.0, 100.0),
            "high": self._trapmf(profit_pips, 80.0, 120.0, 500.0, 500.0),
        }

    def _time_membership(self, hours: float) -> dict[str, float]:
        """Classify time in trade into fuzzy sets.

        Args:
            hours: Hours since trade entry.

        Returns:
            Dict with keys 'short', 'medium', 'long'.
        """
        return {
            "short": self._trapmf(hours, 0.0, 0.0, 1.0, 3.0),
            "medium": self._trimf(hours, 2.0, 6.0, 12.0),
            "long": self._trapmf(hours, 10.0, 18.0, 48.0, 48.0),
        }

    def _momentum_membership(self, momentum: float) -> dict[str, float]:
        """Classify momentum score into fuzzy sets.

        Args:
            momentum: Momentum score in [-1, 1]; positive = bullish.

        Returns:
            Dict with keys 'strong_against', 'weak', 'neutral', 'strong_with'.
        """
        return {
            "strong_against": self._trapmf(momentum, -1.0, -1.0, -0.5, -0.2),
            "weak": self._trimf(momentum, -0.4, -0.1, 0.2),
            "neutral": self._trimf(momentum, -0.2, 0.0, 0.2),
            "strong_with": self._trapmf(momentum, 0.2, 0.5, 1.0, 1.0),
        }

    def _volatility_membership(self, volatility: float) -> dict[str, float]:
        """Classify volatility level into fuzzy sets.

        Args:
            volatility: Normalised volatility score in [0, 1].

        Returns:
            Dict with keys 'low', 'normal', 'high'.
        """
        return {
            "low": self._trapmf(volatility, 0.0, 0.0, 0.2, 0.4),
            "normal": self._trimf(volatility, 0.3, 0.5, 0.7),
            "high": self._trapmf(volatility, 0.6, 0.8, 1.0, 1.0),
        }

    def _smc_membership(self, smc_signal: float) -> dict[str, float]:
        """Classify SMC signal into fuzzy sets.

        Args:
            smc_signal: SMC bias in [-1, 1]; positive = bullish structure.

        Returns:
            Dict with keys 'bearish', 'neutral', 'bullish'.
        """
        return {
            "bearish": self._trapmf(smc_signal, -1.0, -1.0, -0.3, 0.0),
            "neutral": self._trimf(smc_signal, -0.3, 0.0, 0.3),
            "bullish": self._trapmf(smc_signal, 0.0, 0.3, 1.0, 1.0),
        }

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rules(
        self,
        profit: dict[str, float],
        time: dict[str, float],
        momentum: dict[str, float],
        volatility: dict[str, float],
        smc: dict[str, float],
    ) -> list[tuple[float, float]]:
        """Evaluate fuzzy rules and return (strength, output_centroid) pairs.

        Rules are defined as (antecedent_strength, output_location) where
        output_location is the centre of the output singleton on [0, 1].

        Args:
            profit: Profit membership values.
            time: Time membership values.
            momentum: Momentum membership values.
            volatility: Volatility membership values.
            smc: SMC signal membership values.

        Returns:
            List of (firing_strength, output_value) tuples.
        """
        rules: list[tuple[float, float]] = []

        # Rule 1: High profit AND weakening/against momentum → EXIT (strong)
        r1 = min(profit["high"], momentum["strong_against"])
        rules.append((r1, 0.90))

        # Rule 2: High profit AND weak momentum → EXIT
        r2 = min(profit["high"], momentum["weak"])
        rules.append((r2, 0.80))

        # Rule 3: Moderate profit AND strong against momentum → EXIT
        r3 = min(profit["moderate"], momentum["strong_against"])
        rules.append((r3, 0.75))

        # Rule 4: Long time AND low volatility (ranging) → EXIT
        r4 = min(time["long"], volatility["low"])
        rules.append((r4, 0.70))

        # Rule 5: Long time AND neutral momentum → EXIT
        r5 = min(time["long"], momentum["neutral"])
        rules.append((r5, 0.65))

        # Rule 6: High profit AND bearish SMC (for a long) → EXIT
        r6 = min(profit["high"], smc["bearish"])
        rules.append((r6, 0.85))

        # Rule 7: Moderate profit AND bearish SMC → EXIT
        r7 = min(profit["moderate"], smc["bearish"])
        rules.append((r7, 0.60))

        # Rule 8: High volatility AND loss → HOLD / reduce exit desire
        r8 = min(volatility["high"], profit["loss"])
        rules.append((r8, 0.30))

        # Rule 9: Small profit AND strong momentum with trade → HOLD
        r9 = min(profit["small"], momentum["strong_with"])
        rules.append((r9, 0.20))

        # Rule 10: Short time AND strong momentum → HOLD
        r10 = min(time["short"], momentum["strong_with"])
        rules.append((r10, 0.15))

        return rules

    # ------------------------------------------------------------------
    # Defuzzification
    # ------------------------------------------------------------------

    def _centroid_defuzzify(self, rules: list[tuple[float, float]]) -> float:
        """Centroid defuzzification over singleton outputs.

        Args:
            rules: List of (firing_strength, output_value) from rule evaluation.

        Returns:
            Defuzzified crisp output in [0, 1].
        """
        numerator = sum(strength * value for strength, value in rules)
        denominator = sum(strength for strength, _ in rules)
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_exit_score(
        self,
        profit_pips: float,
        time_in_trade_hours: float,
        momentum_score: float,
        volatility: float,
        smc_signal: float,
    ) -> float:
        """Calculate fuzzy exit score.

        Args:
            profit_pips: Current trade profit in pips (negative = loss).
            time_in_trade_hours: Hours since entry.
            momentum_score: Directional momentum in [-1, 1].
            volatility: Normalised volatility in [0, 1].
            smc_signal: SMC structural bias in [-1, 1].

        Returns:
            Exit score in [0.0, 1.0]; higher means stronger exit signal.
        """
        profit = self._profit_membership(profit_pips)
        time = self._time_membership(time_in_trade_hours)
        momentum = self._momentum_membership(momentum_score)
        vol = self._volatility_membership(volatility)
        smc = self._smc_membership(smc_signal)

        rules = self._evaluate_rules(profit, time, momentum, vol, smc)
        score = self._centroid_defuzzify(rules)
        logger.debug("Fuzzy exit score: %.3f", score)
        return round(score, 4)

    def should_exit(
        self,
        profit_pips: float,
        time_hours: float,
        momentum: float,
        volatility: float,
        smc_score: float,
    ) -> bool:
        """Return True if fuzzy logic recommends exiting the trade.

        Args:
            profit_pips: Current profit in pips.
            time_hours: Hours in trade.
            momentum: Directional momentum score in [-1, 1].
            volatility: Normalised volatility in [0, 1].
            smc_score: SMC structural bias in [-1, 1].

        Returns:
            True if exit is recommended.
        """
        score = self.calculate_exit_score(
            profit_pips, time_hours, momentum, volatility, smc_score
        )
        return score >= self.exit_threshold

    def evaluate(
        self,
        profit_pips: float,
        time_in_trade_hours: float,
        momentum_score: float,
        volatility: float,
        smc_signal: float,
    ) -> FuzzyExitScore:
        """Full evaluation returning a structured result.

        Args:
            profit_pips: Current profit in pips.
            time_in_trade_hours: Hours since entry.
            momentum_score: Directional momentum in [-1, 1].
            volatility: Normalised volatility in [0, 1].
            smc_signal: SMC structural bias in [-1, 1].

        Returns:
            FuzzyExitScore with score, decision and reason.
        """
        score = self.calculate_exit_score(
            profit_pips, time_in_trade_hours, momentum_score, volatility, smc_signal
        )
        exit_flag = score >= self.exit_threshold

        if score >= 0.75:
            reason = "Strong exit signal – high score from multiple converging rules"
        elif score >= self.exit_threshold:
            reason = "Moderate exit signal – threshold crossed"
        else:
            reason = "Hold – insufficient exit signal"

        return FuzzyExitScore(score=score, should_exit=exit_flag, reason=reason)

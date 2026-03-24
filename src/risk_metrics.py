from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


class RiskMetrics:
    """Advanced risk and performance metrics calculator.

    All ratio calculations assume daily returns unless noted otherwise.
    Uses numpy for vectorised computation.
    """

    # ------------------------------------------------------------------
    # Return-based ratios
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_sharpe(
        returns: np.ndarray | list[float],
        risk_free_rate: float = 0.02,
    ) -> float:
        """Calculate annualised Sharpe ratio.

        Args:
            returns: Array of daily (or per-trade) returns.
            risk_free_rate: Annual risk-free rate (default 2%).

        Returns:
            Annualised Sharpe ratio; 0.0 if standard deviation is zero.
        """
        r = np.asarray(returns, dtype=float)
        if len(r) == 0:
            return 0.0
        daily_rfr = risk_free_rate / _TRADING_DAYS_PER_YEAR
        excess = r - daily_rfr
        std = float(np.std(excess, ddof=1))
        if std == 0.0:
            return 0.0
        sharpe = float(np.mean(excess)) / std * math.sqrt(_TRADING_DAYS_PER_YEAR)
        return round(sharpe, 4)

    @staticmethod
    def calculate_sortino(
        returns: np.ndarray | list[float],
        risk_free_rate: float = 0.02,
    ) -> float:
        """Calculate annualised Sortino ratio (downside deviation).

        Args:
            returns: Array of daily returns.
            risk_free_rate: Annual risk-free rate (default 2%).

        Returns:
            Annualised Sortino ratio; 0.0 if downside deviation is zero.
        """
        r = np.asarray(returns, dtype=float)
        if len(r) == 0:
            return 0.0
        daily_rfr = risk_free_rate / _TRADING_DAYS_PER_YEAR
        excess = r - daily_rfr
        downside = excess[excess < 0.0]
        if len(downside) == 0:
            return float("inf")
        downside_std = float(np.std(downside, ddof=1))
        if downside_std == 0.0:
            return 0.0
        sortino = float(np.mean(excess)) / downside_std * math.sqrt(_TRADING_DAYS_PER_YEAR)
        return round(sortino, 4)

    @staticmethod
    def calculate_calmar(
        returns: np.ndarray | list[float],
        max_drawdown: float,
    ) -> float:
        """Calculate Calmar ratio (annualised return / max drawdown).

        Args:
            returns: Array of daily returns.
            max_drawdown: Maximum drawdown expressed as a positive decimal (e.g. 0.20 = 20%).

        Returns:
            Calmar ratio; 0.0 if max_drawdown is zero.
        """
        r = np.asarray(returns, dtype=float)
        if len(r) == 0 or max_drawdown <= 0.0:
            return 0.0
        annualised_return = float(np.mean(r)) * _TRADING_DAYS_PER_YEAR
        calmar = annualised_return / max_drawdown
        return round(calmar, 4)

    # ------------------------------------------------------------------
    # Tail risk
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_var(
        returns: np.ndarray | list[float],
        confidence: float = 0.95,
    ) -> float:
        """Calculate historical Value at Risk (VaR).

        Returns the loss at the specified confidence level – a positive value
        representing the loss magnitude.

        Args:
            returns: Array of daily returns.
            confidence: Confidence level (default 0.95 = 95%).

        Returns:
            VaR as a positive loss value.
        """
        r = np.asarray(returns, dtype=float)
        if len(r) == 0:
            return 0.0
        percentile = (1.0 - confidence) * 100.0
        var = -float(np.percentile(r, percentile))
        return round(max(var, 0.0), 6)

    @staticmethod
    def calculate_cvar(
        returns: np.ndarray | list[float],
        confidence: float = 0.95,
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall).

        Average loss in the worst (1 - confidence) tail of the distribution.

        Args:
            returns: Array of daily returns.
            confidence: Confidence level (default 0.95).

        Returns:
            CVaR as a positive loss value.
        """
        r = np.asarray(returns, dtype=float)
        if len(r) == 0:
            return 0.0
        percentile = (1.0 - confidence) * 100.0
        var_threshold = float(np.percentile(r, percentile))
        tail = r[r <= var_threshold]
        if len(tail) == 0:
            return 0.0
        cvar = -float(np.mean(tail))
        return round(max(cvar, 0.0), 6)

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_max_drawdown(
        equity_curve: np.ndarray | list[float],
    ) -> dict[str, float]:
        """Calculate maximum drawdown and drawdown duration.

        Args:
            equity_curve: Sequence of equity values (e.g. account balance over time).

        Returns:
            Dict with keys:
              - ``max_drawdown``: Maximum drawdown as a positive fraction.
              - ``max_drawdown_duration``: Length of the longest drawdown period in bars.
              - ``current_drawdown``: Current drawdown from latest peak.
        """
        eq = np.asarray(equity_curve, dtype=float)
        if len(eq) < 2:
            return {"max_drawdown": 0.0, "max_drawdown_duration": 0, "current_drawdown": 0.0}

        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / (peak + 1e-10)

        max_dd = float(np.min(dd))  # most negative
        current_dd = float(dd[-1])

        # Duration: longest continuous drawdown period
        in_dd = dd < 0.0
        max_dur = 0
        cur_dur = 0
        for flag in in_dd:
            if flag:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0

        return {
            "max_drawdown": round(abs(max_dd), 6),
            "max_drawdown_duration": max_dur,
            "current_drawdown": round(abs(current_dd), 6),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_performance_summary(
        self,
        returns: np.ndarray | list[float],
        equity_curve: np.ndarray | list[float],
        risk_free_rate: float = 0.02,
    ) -> dict[str, float | int]:
        """Compute all performance metrics in a single call.

        Args:
            returns: Array of daily / per-trade returns.
            equity_curve: Account equity time series.
            risk_free_rate: Annual risk-free rate.

        Returns:
            Dictionary of metric name → value.
        """
        r = np.asarray(returns, dtype=float)
        dd_info = self.calculate_max_drawdown(equity_curve)
        max_dd = dd_info["max_drawdown"]

        return {
            "sharpe_ratio": self.calculate_sharpe(r, risk_free_rate),
            "sortino_ratio": self.calculate_sortino(r, risk_free_rate),
            "calmar_ratio": self.calculate_calmar(r, max_dd),
            "var_95": self.calculate_var(r, 0.95),
            "cvar_95": self.calculate_cvar(r, 0.95),
            "max_drawdown": max_dd,
            "max_drawdown_duration": dd_info["max_drawdown_duration"],
            "current_drawdown": dd_info["current_drawdown"],
            "total_return": round(float(r.sum()), 6),
            "annualised_return": round(float(r.mean()) * _TRADING_DAYS_PER_YEAR, 6),
            "volatility": round(float(np.std(r, ddof=1)) * math.sqrt(_TRADING_DAYS_PER_YEAR), 6),
            "win_rate": round(float(np.mean(r > 0)), 4),
            "num_trades": int(len(r)),
        }

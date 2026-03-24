from __future__ import annotations

from datetime import date

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

MAJOR: int = 1
MINOR: int = 0
PATCH: int = 0

VERSION: str = f"{MAJOR}.{MINOR}.{PATCH}"

__version__: str = VERSION

BUILD_DATE: str = "2025-01-01"


def get_version() -> str:
    """Return the current version string.

    Returns:
        Version string in MAJOR.MINOR.PATCH format.
    """
    return VERSION


# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------

CHANGELOG: dict[str, list[str]] = {
    "1.0.0": [
        "Initial production release",
        "SMC-based entry detection with H1 and M5 confirmation",
        "Kalman filter price smoothing",
        "Kelly criterion position sizing with fractional Kelly safety cap",
        "Fuzzy logic exit system with centroid defuzzification",
        "Macro connector for DXY and yields sentiment",
        "Momentum persistence scoring (ADX + MACD + RSI divergence)",
        "Recovery detector with anti-revenge-trading logic",
        "Advanced risk metrics: Sharpe, Sortino, Calmar, VaR, CVaR",
        "Trajectory predictor with Kalman smoothing",
        "Centralised filter configuration per regime",
        "Telegram command and notification system",
        "Profit momentum streak tracker",
    ],
}

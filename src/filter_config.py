from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FilterProfile:
    """Configuration profile for a specific market regime.

    Attributes:
        regime: Regime name ('trending', 'ranging', 'volatile').
        min_confidence: Minimum signal confidence required.
        min_volatility: Minimum normalised volatility to allow trades.
        max_spread_pips: Maximum acceptable spread in pips.
        max_risk_pct: Maximum risk per trade as fraction of balance.
        min_momentum_score: Minimum momentum persistence score.
        require_volume_confirmation: Whether volume check is mandatory.
        max_trades_per_session: Trade count cap for the session.
    """

    regime: str
    min_confidence: float
    min_volatility: float
    max_spread_pips: float
    max_risk_pct: float
    min_momentum_score: float
    require_volume_confirmation: bool
    max_trades_per_session: int


@dataclass
class FilterConfig:
    """Centralised filter configuration for all market regimes.

    Provides per-regime FilterProfile objects and global risk limits.
    Instantiate once and pass to strategy components.

    Attributes:
        trending_profile: Config when market is trending.
        ranging_profile: Config when market is ranging / consolidating.
        volatile_profile: Config during high-volatility / news regimes.
        global_max_daily_loss_pct: Hard daily loss limit as pct of balance.
        global_max_open_trades: Maximum simultaneous open positions.
        session_risk_limits: Per-session risk caps (keys: london, ny, asia).
        momentum_min_adx: ADX threshold for trend strength filter.
        momentum_macd_threshold: MACD threshold for momentum filter.
    """

    # Regime profiles
    trending_profile: FilterProfile = field(
        default_factory=lambda: FilterProfile(
            regime="trending",
            min_confidence=0.60,
            min_volatility=0.20,
            max_spread_pips=3.0,
            max_risk_pct=0.015,       # 1.5% per trade
            min_momentum_score=0.40,
            require_volume_confirmation=False,
            max_trades_per_session=4,
        )
    )

    ranging_profile: FilterProfile = field(
        default_factory=lambda: FilterProfile(
            regime="ranging",
            min_confidence=0.70,      # stricter – fewer false signals
            min_volatility=0.10,
            max_spread_pips=2.5,
            max_risk_pct=0.010,       # 1.0% – smaller size in chop
            min_momentum_score=0.50,
            require_volume_confirmation=True,
            max_trades_per_session=2,
        )
    )

    volatile_profile: FilterProfile = field(
        default_factory=lambda: FilterProfile(
            regime="volatile",
            min_confidence=0.75,      # highest bar – protect against spikes
            min_volatility=0.50,
            max_spread_pips=5.0,      # wider spreads allowed during news
            max_risk_pct=0.008,       # 0.8% – minimal risk during volatility
            min_momentum_score=0.55,
            require_volume_confirmation=True,
            max_trades_per_session=1,
        )
    )

    # Global limits
    global_max_daily_loss_pct: float = 0.03        # 3% daily stop
    global_max_open_trades: int = 2

    # Per-session risk multipliers (applied on top of profile max_risk_pct)
    session_risk_limits: dict[str, float] = field(
        default_factory=lambda: {
            "london": 1.0,   # full risk during London
            "ny": 1.0,       # full risk during NY
            "overlap": 1.0,  # London/NY overlap – most liquid
            "asia": 0.75,    # reduced risk – thinner market
            "off": 0.50,     # off-hours – minimal risk
        }
    )

    # Momentum thresholds
    momentum_min_adx: float = 20.0       # ADX below this = no clear trend
    momentum_macd_threshold: float = 0.0  # MACD sign flip threshold

    def get_profile(self, regime: str) -> FilterProfile:
        """Return the FilterProfile for the given regime.

        Args:
            regime: One of 'trending', 'ranging', 'volatile'.
                    Unknown regimes fall back to the volatile (strictest) profile.

        Returns:
            FilterProfile for the requested regime.
        """
        mapping = {
            "trending": self.trending_profile,
            "ranging": self.ranging_profile,
            "volatile": self.volatile_profile,
        }
        profile = mapping.get(regime.lower())
        if profile is None:
            logger.warning(
                "Unknown regime '%s'; using volatile profile (strictest)", regime
            )
            return self.volatile_profile
        return profile

    def get_session_risk_multiplier(self, session: str) -> float:
        """Return risk multiplier for the given trading session.

        Args:
            session: Session name (e.g. 'london', 'ny', 'asia').

        Returns:
            Risk multiplier in [0, 1]; defaults to 0.5 for unknown sessions.
        """
        return self.session_risk_limits.get(session.lower(), 0.5)

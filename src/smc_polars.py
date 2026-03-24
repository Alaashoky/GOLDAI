"""Smart Money Concepts (SMC) analyzer using Polars.

Detects Order Blocks, Fair Value Gaps, Break of Structure,
Change of Character, and liquidity pools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


class StructureType(str, Enum):
    HH = "HH"
    HL = "HL"
    LH = "LH"
    LL = "LL"


class SignalType(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class OrderBlock:
    index: int
    type: SignalType
    high: float
    low: float
    volume: float
    confidence: float = 0.0


@dataclass
class FairValueGap:
    index: int
    type: SignalType
    top: float
    bottom: float
    size: float


@dataclass
class SMCSignal:
    direction: SignalType = SignalType.NEUTRAL
    confidence: float = 0.0
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fvgs: List[FairValueGap] = field(default_factory=list)
    bos_detected: bool = False
    choch_detected: bool = False
    structure: List[StructureType] = field(default_factory=list)
    liquidity_above: float = 0.0
    liquidity_below: float = 0.0
    in_premium: bool = False
    in_discount: bool = False


class SMCAnalyzer:
    """Smart Money Concepts analyzer using Polars for performance."""

    def __init__(self, lookback: int = 50, ob_threshold: float = 1.5) -> None:
        self.lookback = lookback
        self.ob_threshold = ob_threshold

    def _find_swing_points(
        self, df: pl.DataFrame, window: int = 5
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        highs_col = df["high"].to_list()
        lows_col = df["low"].to_list()
        swing_highs: List[Tuple[int, float]] = []
        swing_lows: List[Tuple[int, float]] = []

        for i in range(window, len(highs_col) - window):
            is_high = all(
                highs_col[i] >= highs_col[i - j] and highs_col[i] >= highs_col[i + j]
                for j in range(1, window + 1)
            )
            if is_high:
                swing_highs.append((i, highs_col[i]))

            is_low = all(
                lows_col[i] <= lows_col[i - j] and lows_col[i] <= lows_col[i + j]
                for j in range(1, window + 1)
            )
            if is_low:
                swing_lows.append((i, lows_col[i]))

        return swing_highs, swing_lows

    def detect_order_blocks(self, df: pl.DataFrame) -> List[OrderBlock]:
        obs: List[OrderBlock] = []
        if len(df) < 3:
            return obs

        opens = df["open"].to_list()
        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        volumes = df["tick_volume"].to_list() if "tick_volume" in df.columns else [0.0] * len(df)

        avg_body = np.mean([abs(c - o) for o, c in zip(opens, closes)])

        for i in range(1, len(df) - 1):
            body = abs(closes[i] - opens[i])
            prev_body = abs(closes[i - 1] - opens[i - 1])

            if body > avg_body * self.ob_threshold:
                if closes[i - 1] < opens[i - 1] and closes[i] > opens[i]:
                    obs.append(OrderBlock(
                        index=i - 1,
                        type=SignalType.BULLISH,
                        high=highs[i - 1],
                        low=lows[i - 1],
                        volume=volumes[i - 1],
                        confidence=min(body / (avg_body + 1e-8), 3.0) / 3.0,
                    ))
                elif closes[i - 1] > opens[i - 1] and closes[i] < opens[i]:
                    obs.append(OrderBlock(
                        index=i - 1,
                        type=SignalType.BEARISH,
                        high=highs[i - 1],
                        low=lows[i - 1],
                        volume=volumes[i - 1],
                        confidence=min(body / (avg_body + 1e-8), 3.0) / 3.0,
                    ))
        return obs[-10:]

    def detect_fvg(self, df: pl.DataFrame) -> List[FairValueGap]:
        fvgs: List[FairValueGap] = []
        if len(df) < 3:
            return fvgs

        highs = df["high"].to_list()
        lows = df["low"].to_list()

        for i in range(2, len(df)):
            if lows[i] > highs[i - 2]:
                fvgs.append(FairValueGap(
                    index=i - 1,
                    type=SignalType.BULLISH,
                    top=lows[i],
                    bottom=highs[i - 2],
                    size=lows[i] - highs[i - 2],
                ))
            elif highs[i] < lows[i - 2]:
                fvgs.append(FairValueGap(
                    index=i - 1,
                    type=SignalType.BEARISH,
                    top=lows[i - 2],
                    bottom=highs[i],
                    size=lows[i - 2] - highs[i],
                ))
        return fvgs[-10:]

    def get_market_structure(
        self, df: pl.DataFrame
    ) -> List[StructureType]:
        swing_highs, swing_lows = self._find_swing_points(df)
        structure: List[StructureType] = []

        for i in range(1, len(swing_highs)):
            if swing_highs[i][1] > swing_highs[i - 1][1]:
                structure.append(StructureType.HH)
            else:
                structure.append(StructureType.LH)

        for i in range(1, len(swing_lows)):
            if swing_lows[i][1] > swing_lows[i - 1][1]:
                structure.append(StructureType.HL)
            else:
                structure.append(StructureType.LL)

        return structure

    def detect_bos(self, df: pl.DataFrame) -> bool:
        swing_highs, swing_lows = self._find_swing_points(df)
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return False

        closes = df["close"].to_list()
        last_close = closes[-1]
        last_high = swing_highs[-1][1]
        last_low = swing_lows[-1][1]

        return last_close > last_high or last_close < last_low

    def detect_choch(self, df: pl.DataFrame) -> bool:
        structure = self.get_market_structure(df)
        if len(structure) < 4:
            return False

        recent = structure[-4:]
        bullish_trend = all(s in (StructureType.HH, StructureType.HL) for s in recent[:2])
        bearish_shift = any(s in (StructureType.LH, StructureType.LL) for s in recent[2:])

        bearish_trend = all(s in (StructureType.LH, StructureType.LL) for s in recent[:2])
        bullish_shift = any(s in (StructureType.HH, StructureType.HL) for s in recent[2:])

        return (bullish_trend and bearish_shift) or (bearish_trend and bullish_shift)

    def find_liquidity_pools(self, df: pl.DataFrame) -> Tuple[float, float]:
        swing_highs, swing_lows = self._find_swing_points(df)
        liq_above = max((h[1] for h in swing_highs), default=0.0)
        liq_below = min((l[1] for l in swing_lows), default=0.0)
        return liq_above, liq_below

    def _check_premium_discount(
        self, df: pl.DataFrame
    ) -> Tuple[bool, bool]:
        swing_highs, swing_lows = self._find_swing_points(df)
        if not swing_highs or not swing_lows:
            return False, False

        range_high = max(h[1] for h in swing_highs[-5:])
        range_low = min(l[1] for l in swing_lows[-5:])
        midpoint = (range_high + range_low) / 2
        current_price = df["close"].to_list()[-1]

        in_premium = current_price > midpoint
        in_discount = current_price < midpoint
        return in_premium, in_discount

    def analyze(self, df: pl.DataFrame) -> SMCSignal:
        if len(df) < 10:
            return SMCSignal()

        obs = self.detect_order_blocks(df)
        fvgs = self.detect_fvg(df)
        bos = self.detect_bos(df)
        choch = self.detect_choch(df)
        structure = self.get_market_structure(df)
        liq_above, liq_below = self.find_liquidity_pools(df)
        in_premium, in_discount = self._check_premium_discount(df)

        bullish_score = 0.0
        bearish_score = 0.0

        bullish_obs = [ob for ob in obs if ob.type == SignalType.BULLISH]
        bearish_obs = [ob for ob in obs if ob.type == SignalType.BEARISH]
        bullish_score += len(bullish_obs) * 0.2
        bearish_score += len(bearish_obs) * 0.2

        bullish_fvgs = [f for f in fvgs if f.type == SignalType.BULLISH]
        bearish_fvgs = [f for f in fvgs if f.type == SignalType.BEARISH]
        bullish_score += len(bullish_fvgs) * 0.15
        bearish_score += len(bearish_fvgs) * 0.15

        hh_count = sum(1 for s in structure[-5:] if s in (StructureType.HH, StructureType.HL))
        ll_count = sum(1 for s in structure[-5:] if s in (StructureType.LH, StructureType.LL))
        bullish_score += hh_count * 0.1
        bearish_score += ll_count * 0.1

        if in_discount:
            bullish_score += 0.15
        if in_premium:
            bearish_score += 0.15

        if choch:
            if bullish_score > bearish_score:
                bullish_score += 0.2
            else:
                bearish_score += 0.2

        total = bullish_score + bearish_score
        if total == 0:
            direction = SignalType.NEUTRAL
            confidence = 0.0
        elif bullish_score > bearish_score:
            direction = SignalType.BULLISH
            confidence = min(bullish_score / (total + 0.5), 1.0)
        else:
            direction = SignalType.BEARISH
            confidence = min(bearish_score / (total + 0.5), 1.0)

        return SMCSignal(
            direction=direction,
            confidence=round(confidence, 3),
            order_blocks=obs,
            fvgs=fvgs,
            bos_detected=bos,
            choch_detected=choch,
            structure=structure,
            liquidity_above=liq_above,
            liquidity_below=liq_below,
            in_premium=in_premium,
            in_discount=in_discount,
        )

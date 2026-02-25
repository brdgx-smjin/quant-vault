"""Harmonic pattern detection (XABCD patterns)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class PatternType(Enum):
    GARTLEY = "gartley"
    BUTTERFLY = "butterfly"
    BAT = "bat"
    CRAB = "crab"
    SHARK = "shark"


@dataclass
class HarmonicPattern:
    """Detected harmonic pattern."""

    pattern_type: PatternType
    direction: str  # "bullish" or "bearish"
    points: dict[str, tuple[int, float]]  # X, A, B, C, D -> (index, price)
    score: float  # 0-1 completion score
    prz_low: float  # Potential Reversal Zone low
    prz_high: float  # Potential Reversal Zone high


# Ratio tolerances for each pattern
PATTERN_RATIOS = {
    PatternType.GARTLEY: {
        "XAB": (0.618, 0.05),  # (target_ratio, tolerance)
        "ABC": (0.382, 0.886, 0.05),  # (min, max, tolerance)
        "BCD": (1.272, 1.618, 0.05),
        "XAD": (0.786, 0.05),
    },
    PatternType.BUTTERFLY: {
        "XAB": (0.786, 0.05),
        "ABC": (0.382, 0.886, 0.05),
        "BCD": (1.618, 2.618, 0.05),
        "XAD": (1.272, 0.05),
    },
    PatternType.BAT: {
        "XAB": (0.382, 0.5, 0.05),
        "ABC": (0.382, 0.886, 0.05),
        "BCD": (1.618, 2.618, 0.05),
        "XAD": (0.886, 0.05),
    },
    PatternType.CRAB: {
        "XAB": (0.382, 0.618, 0.05),
        "ABC": (0.382, 0.886, 0.05),
        "BCD": (2.618, 3.618, 0.05),
        "XAD": (1.618, 0.05),
    },
}


class HarmonicPatternDetector:
    """Detects XABCD harmonic patterns in price data."""

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance

    def find_zigzag_points(
        self, df: pd.DataFrame, depth: int = 12, deviation: float = 5.0
    ) -> list[tuple[int, float]]:
        """Find significant swing points using zigzag algorithm.

        Args:
            df: OHLCV DataFrame.
            depth: Minimum bars between pivots.
            deviation: Minimum price change percentage.

        Returns:
            List of (index, price) tuples for swing points.
        """
        highs = df["high"].values
        lows = df["low"].values
        pivots: list[tuple[int, float]] = []

        last_pivot_type = 0  # 1 = high, -1 = low
        last_pivot_idx = 0
        last_pivot_val = highs[0]

        for i in range(depth, len(df)):
            window_high = np.max(highs[i - depth : i + 1])
            window_low = np.min(lows[i - depth : i + 1])

            if highs[i] == window_high and last_pivot_type != 1:
                if last_pivot_type == -1:
                    pct_change = abs(highs[i] - last_pivot_val) / last_pivot_val * 100
                    if pct_change >= deviation:
                        pivots.append((i, float(highs[i])))
                        last_pivot_type = 1
                        last_pivot_idx = i
                        last_pivot_val = highs[i]
                else:
                    pivots.append((i, float(highs[i])))
                    last_pivot_type = 1
                    last_pivot_idx = i
                    last_pivot_val = highs[i]

            elif lows[i] == window_low and last_pivot_type != -1:
                if last_pivot_type == 1:
                    pct_change = abs(lows[i] - last_pivot_val) / last_pivot_val * 100
                    if pct_change >= deviation:
                        pivots.append((i, float(lows[i])))
                        last_pivot_type = -1
                        last_pivot_idx = i
                        last_pivot_val = lows[i]
                else:
                    pivots.append((i, float(lows[i])))
                    last_pivot_type = -1
                    last_pivot_idx = i
                    last_pivot_val = lows[i]

        return pivots

    def _check_ratio(
        self, actual: float, target: float | tuple, tolerance: float
    ) -> float:
        """Check how well a ratio matches the target. Returns score 0-1."""
        if isinstance(target, tuple):
            low, high = target[0], target[1]
            if low - tolerance <= actual <= high + tolerance:
                mid = (low + high) / 2
                return max(0, 1 - abs(actual - mid) / (high - low + tolerance))
            return 0.0
        else:
            diff = abs(actual - target)
            if diff <= tolerance:
                return 1 - diff / tolerance
            return 0.0

    def detect_patterns(
        self, df: pd.DataFrame, lookback: int = 100
    ) -> list[HarmonicPattern]:
        """Detect all harmonic patterns in the data.

        Args:
            df: OHLCV DataFrame.
            lookback: Number of bars to analyze.

        Returns:
            List of detected HarmonicPattern objects.
        """
        recent = df.tail(lookback) if len(df) > lookback else df
        pivots = self.find_zigzag_points(recent)

        if len(pivots) < 5:
            return []

        patterns: list[HarmonicPattern] = []

        # Check last 5 pivots for pattern
        for i in range(len(pivots) - 4):
            x_idx, x_price = pivots[i]
            a_idx, a_price = pivots[i + 1]
            b_idx, b_price = pivots[i + 2]
            c_idx, c_price = pivots[i + 3]
            d_idx, d_price = pivots[i + 4]

            xa = abs(a_price - x_price)
            if xa == 0:
                continue

            ab = abs(b_price - a_price)
            bc = abs(c_price - b_price)
            cd = abs(d_price - c_price)

            xab_ratio = ab / xa
            abc_ratio = bc / ab if ab != 0 else 0
            bcd_ratio = cd / bc if bc != 0 else 0
            xad_ratio = abs(d_price - x_price) / xa

            direction = "bullish" if a_price < x_price else "bearish"

            for pattern_type, ratios in PATTERN_RATIOS.items():
                scores = []
                scores.append(self._check_ratio(
                    xab_ratio, ratios["XAB"][0] if len(ratios["XAB"]) == 2
                    else (ratios["XAB"][0], ratios["XAB"][1]),
                    self.tolerance,
                ))
                scores.append(self._check_ratio(
                    abc_ratio, (ratios["ABC"][0], ratios["ABC"][1]),
                    self.tolerance,
                ))
                scores.append(self._check_ratio(
                    bcd_ratio, (ratios["BCD"][0], ratios["BCD"][1]),
                    self.tolerance,
                ))
                scores.append(self._check_ratio(
                    xad_ratio, ratios["XAD"][0],
                    self.tolerance,
                ))

                avg_score = sum(scores) / len(scores)
                if avg_score > 0.5:
                    prz_mid = d_price
                    prz_range = xa * 0.05
                    patterns.append(HarmonicPattern(
                        pattern_type=pattern_type,
                        direction=direction,
                        points={
                            "X": (x_idx, x_price),
                            "A": (a_idx, a_price),
                            "B": (b_idx, b_price),
                            "C": (c_idx, c_price),
                            "D": (d_idx, d_price),
                        },
                        score=avg_score,
                        prz_low=prz_mid - prz_range,
                        prz_high=prz_mid + prz_range,
                    ))

        return sorted(patterns, key=lambda p: p.score, reverse=True)

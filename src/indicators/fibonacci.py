"""Fibonacci retracement and extension calculator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import FIB_EXTENSION, FIB_RETRACEMENT


@dataclass
class FibLevel:
    """A single Fibonacci level with price and ratio."""

    ratio: float
    price: float
    kind: str  # "retracement" or "extension"


@dataclass
class FibResult:
    """Result of Fibonacci analysis with all levels."""

    swing_high: float
    swing_low: float
    direction: str  # "up" (low→high) or "down" (high→low)
    levels: list[FibLevel]


class FibonacciCalculator:
    """Calculates Fibonacci retracement and extension levels."""

    def __init__(
        self,
        retracement_levels: list[float] = FIB_RETRACEMENT,
        extension_levels: list[float] = FIB_EXTENSION,
    ) -> None:
        self.retracement_levels = retracement_levels
        self.extension_levels = extension_levels

    def calculate_retracement(
        self, swing_high: float, swing_low: float, direction: str = "up"
    ) -> list[FibLevel]:
        """Calculate Fibonacci retracement levels.

        Args:
            swing_high: The swing high price.
            swing_low: The swing low price.
            direction: 'up' for bullish swing, 'down' for bearish swing.

        Returns:
            List of FibLevel objects.
        """
        diff = swing_high - swing_low
        levels = []
        for ratio in self.retracement_levels:
            if direction == "up":
                price = swing_high - diff * ratio
            else:
                price = swing_low + diff * ratio
            levels.append(FibLevel(ratio=ratio, price=price, kind="retracement"))
        return levels

    def calculate_extension(
        self, swing_high: float, swing_low: float, direction: str = "up"
    ) -> list[FibLevel]:
        """Calculate Fibonacci extension levels.

        Args:
            swing_high: The swing high price.
            swing_low: The swing low price.
            direction: 'up' for bullish target, 'down' for bearish target.

        Returns:
            List of FibLevel objects.
        """
        diff = swing_high - swing_low
        levels = []
        for ratio in self.extension_levels:
            if direction == "up":
                price = swing_high + diff * (ratio - 1.0)
            else:
                price = swing_low - diff * (ratio - 1.0)
            levels.append(FibLevel(ratio=ratio, price=price, kind="extension"))
        return levels

    def find_swing_points(
        self, df: pd.DataFrame, lookback: int = 20
    ) -> tuple[Optional[float], Optional[float]]:
        """Find the most recent swing high and swing low.

        Args:
            df: OHLCV DataFrame.
            lookback: Number of bars to look back.

        Returns:
            Tuple of (swing_high, swing_low) or (None, None).
        """
        if len(df) < lookback:
            return None, None
        recent = df.tail(lookback)
        swing_high = float(recent["high"].max())
        swing_low = float(recent["low"].min())
        return swing_high, swing_low

    def analyze(self, df: pd.DataFrame, lookback: int = 20) -> Optional[FibResult]:
        """Full Fibonacci analysis on recent price data.

        Args:
            df: OHLCV DataFrame.
            lookback: Lookback period for swing detection.

        Returns:
            FibResult with all levels, or None if insufficient data.
        """
        swing_high, swing_low = self.find_swing_points(df, lookback)
        if swing_high is None or swing_low is None:
            return None

        # Determine direction: if close is closer to high, trend is up
        last_close = float(df["close"].iloc[-1])
        mid = (swing_high + swing_low) / 2
        direction = "up" if last_close > mid else "down"

        retracements = self.calculate_retracement(swing_high, swing_low, direction)
        extensions = self.calculate_extension(swing_high, swing_low, direction)

        return FibResult(
            swing_high=swing_high,
            swing_low=swing_low,
            direction=direction,
            levels=retracements + extensions,
        )

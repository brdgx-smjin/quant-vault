"""Candlestick pattern recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CandlePattern:
    """Detected candlestick pattern."""

    name: str
    index: int
    signal: int  # 1 = bullish, -1 = bearish, 0 = neutral
    strength: float  # 0-1


class CandlestickPatterns:
    """Detects common candlestick patterns."""

    @staticmethod
    def detect_all(df: pd.DataFrame) -> list[CandlePattern]:
        """Run all pattern detectors on the last candle.

        Args:
            df: OHLCV DataFrame with at least 3 rows.

        Returns:
            List of detected patterns.
        """
        if len(df) < 3:
            return []

        patterns = []
        idx = len(df) - 1
        o, h, l, c = (
            df["open"].iloc[-1],
            df["high"].iloc[-1],
            df["low"].iloc[-1],
            df["close"].iloc[-1],
        )
        body = abs(c - o)
        full_range = h - l
        if full_range == 0:
            return []

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # Doji
        if body / full_range < 0.1:
            patterns.append(CandlePattern("doji", idx, 0, 0.7))

        # Hammer (bullish)
        if lower_wick > body * 2 and upper_wick < body * 0.5 and c > o:
            patterns.append(CandlePattern("hammer", idx, 1, 0.8))

        # Shooting Star (bearish)
        if upper_wick > body * 2 and lower_wick < body * 0.5 and c < o:
            patterns.append(CandlePattern("shooting_star", idx, -1, 0.8))

        # Engulfing
        if len(df) >= 2:
            prev_o = df["open"].iloc[-2]
            prev_c = df["close"].iloc[-2]
            # Bullish engulfing
            if prev_c < prev_o and c > o and o <= prev_c and c >= prev_o:
                patterns.append(CandlePattern("bullish_engulfing", idx, 1, 0.9))
            # Bearish engulfing
            if prev_c > prev_o and c < o and o >= prev_c and c <= prev_o:
                patterns.append(CandlePattern("bearish_engulfing", idx, -1, 0.9))

        return patterns

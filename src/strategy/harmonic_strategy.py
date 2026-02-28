"""DEPRECATED — Harmonic pattern trading strategy.

Never walk-forward validated. Initial scaffold from project setup.
Harmonic patterns produce too few trades on BTC 1h/15m for reliable WF validation.
Best portfolio: Cross-TF 1hRSI/1hDC/15mRSI/1hWillR 15/50/10/25 = 88% rob, +23.98% OOS (Phase 25).
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.indicators.harmonic import HarmonicPatternDetector
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class HarmonicPatternStrategy(BaseStrategy):
    """Trade based on harmonic pattern detection at PRZ."""

    name = "harmonic_pattern"

    def __init__(
        self,
        min_score: float = 0.6,
        lookback: int = 100,
        symbol: str = SYMBOL,
    ) -> None:
        self.min_score = min_score
        self.lookback = lookback
        self.symbol = symbol
        self.detector = HarmonicPatternDetector()

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate signal when price enters a harmonic PRZ.

        Args:
            df: OHLCV DataFrame.

        Returns:
            TradeSignal.
        """
        patterns = self.detector.detect_patterns(df, self.lookback)
        last_close = float(df["close"].iloc[-1])
        last_ts = df.index[-1]

        for pattern in patterns:
            if pattern.score < self.min_score:
                continue

            # Check if price is in PRZ
            if pattern.prz_low <= last_close <= pattern.prz_high:
                if pattern.direction == "bullish":
                    return TradeSignal(
                        signal=Signal.LONG,
                        symbol=self.symbol,
                        price=last_close,
                        timestamp=last_ts,
                        confidence=pattern.score,
                        stop_loss=pattern.prz_low * 0.99,
                        metadata={
                            "strategy": self.name,
                            "pattern": pattern.pattern_type.value,
                            "score": pattern.score,
                        },
                    )
                else:
                    return TradeSignal(
                        signal=Signal.SHORT,
                        symbol=self.symbol,
                        price=last_close,
                        timestamp=last_ts,
                        confidence=pattern.score,
                        stop_loss=pattern.prz_high * 1.01,
                        metadata={
                            "strategy": self.name,
                            "pattern": pattern.pattern_type.value,
                            "score": pattern.score,
                        },
                    )

        return TradeSignal(
            signal=Signal.HOLD,
            symbol=self.symbol,
            price=last_close,
            timestamp=last_ts,
        )

    def get_required_indicators(self) -> list[str]:
        return []  # Uses raw OHLCV + internal pattern detection

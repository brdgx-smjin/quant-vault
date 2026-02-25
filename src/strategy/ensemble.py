"""Ensemble strategy that combines multiple sub-strategies."""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class EnsembleStrategy(BaseStrategy):
    """Combines signals from multiple strategies via weighted voting."""

    name = "ensemble"

    def __init__(
        self,
        strategies: list[tuple[BaseStrategy, float]],
        threshold: float = 0.5,
        symbol: str = SYMBOL,
    ) -> None:
        """Initialize ensemble.

        Args:
            strategies: List of (strategy, weight) tuples.
            threshold: Minimum weighted score to trigger signal.
            symbol: Trading symbol.
        """
        self.strategies = strategies
        self.threshold = threshold
        self.symbol = symbol

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate combined signal from all sub-strategies.

        Args:
            df: OHLCV DataFrame.

        Returns:
            TradeSignal based on weighted voting.
        """
        last_close = float(df["close"].iloc[-1])
        last_ts = df.index[-1]
        weighted_score = 0.0
        total_weight = 0.0
        sub_signals: list[dict] = []

        for strategy, weight in self.strategies:
            sig = strategy.generate_signal(df)
            score = 0.0
            if sig.signal == Signal.LONG:
                score = sig.confidence
            elif sig.signal == Signal.SHORT:
                score = -sig.confidence

            weighted_score += score * weight
            total_weight += weight
            sub_signals.append({
                "strategy": strategy.name,
                "signal": sig.signal.value,
                "confidence": sig.confidence,
                "weight": weight,
            })

        if total_weight == 0:
            return TradeSignal(signal=Signal.HOLD, symbol=self.symbol,
                               price=last_close, timestamp=last_ts)

        normalized = weighted_score / total_weight

        if normalized > self.threshold:
            return TradeSignal(
                signal=Signal.LONG, symbol=self.symbol,
                price=last_close, timestamp=last_ts,
                confidence=normalized,
                metadata={"strategy": self.name, "sub_signals": sub_signals},
            )
        elif normalized < -self.threshold:
            return TradeSignal(
                signal=Signal.SHORT, symbol=self.symbol,
                price=last_close, timestamp=last_ts,
                confidence=abs(normalized),
                metadata={"strategy": self.name, "sub_signals": sub_signals},
            )

        return TradeSignal(signal=Signal.HOLD, symbol=self.symbol,
                           price=last_close, timestamp=last_ts)

    def get_required_indicators(self) -> list[str]:
        indicators: set[str] = set()
        for strategy, _ in self.strategies:
            indicators.update(strategy.get_required_indicators())
        return list(indicators)

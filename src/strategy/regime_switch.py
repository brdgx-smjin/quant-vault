"""Regime-switching meta-strategy.

Uses ADX to classify market regime (trending vs ranging) and delegates
to appropriate sub-strategies.

High ADX → trending → momentum/breakout strategy (e.g., BBSqueeze)
Low ADX  → ranging  → mean reversion strategy (e.g., RSI MR)

Walk-Forward results (1h, BB+RSI_MR sub-strategies):
  Phase 10 (5w):  80% robustness, OOS +5.48%, Full +6.67%, DD 15.8%
  Phase 10 (7w):  71% robustness, OOS +3.53%

Best config: adx_threshold=20. Higher thresholds (25-35) degrade OOS.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class RegimeSwitchStrategy(BaseStrategy):
    """Switch between trending and ranging strategies based on ADX.

    Walk-Forward results (Phase 10, 1h, BB+RSI_MR sub-strategies):
      5w: 80% robustness, OOS +5.48%, Full +6.67%, DD 15.8%, PF 1.09
      7w: 71% robustness, OOS +3.53%

    ADX20 threshold is optimal. Higher thresholds degrade full-period returns.
    Modest returns but consistent (80% robustness at 5w, 71% at 7w).
    ML Regime Classifier (Phase 10c) does NOT improve over simple ADX threshold.
    """

    name = "regime_switch"

    def __init__(
        self,
        trend_strategy: BaseStrategy,
        range_strategy: BaseStrategy,
        adx_threshold: float = 25.0,
        adx_column: str = "ADX_14",
        symbol: str = SYMBOL,
    ) -> None:
        """Initialize regime switcher.

        Args:
            trend_strategy: Strategy to use when ADX > threshold (trending).
            range_strategy: Strategy to use when ADX <= threshold (ranging).
            adx_threshold: ADX level separating trending from ranging.
            adx_column: DataFrame column name for ADX.
            symbol: Trading symbol.
        """
        self.trend_strategy = trend_strategy
        self.range_strategy = range_strategy
        self.adx_threshold = adx_threshold
        self.adx_column = adx_column
        self.symbol = symbol

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Delegate to trend or range strategy based on current ADX.

        Args:
            df: OHLCV DataFrame with ADX indicator.

        Returns:
            TradeSignal from the selected sub-strategy.
        """
        if len(df) < 30:
            return self._hold(df)

        adx_val = df[self.adx_column].iloc[-1]
        if pd.isna(adx_val):
            return self._hold(df)

        adx_val = float(adx_val)

        if adx_val > self.adx_threshold:
            signal = self.trend_strategy.generate_signal(df)
        else:
            signal = self.range_strategy.generate_signal(df)

        # Tag which regime produced the signal
        if signal.signal not in (Signal.HOLD,):
            signal.metadata["regime"] = "trend" if adx_val > self.adx_threshold else "range"
            signal.metadata["adx"] = adx_val

        return signal

    def _hold(self, df: pd.DataFrame) -> TradeSignal:
        return TradeSignal(
            signal=Signal.HOLD,
            symbol=self.symbol,
            price=float(df["close"].iloc[-1]),
            timestamp=df.index[-1],
        )

    def get_required_indicators(self) -> list[str]:
        indicators = set(self.trend_strategy.get_required_indicators())
        indicators.update(self.range_strategy.get_required_indicators())
        indicators.add(self.adx_column)
        return list(indicators)

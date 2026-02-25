"""Multi-Timeframe trend filter — decorator pattern for any BaseStrategy."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.strategy.base import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger(__name__)


class MultiTimeframeFilter(BaseStrategy):
    """Wraps a base strategy and blocks signals against the higher-TF trend.

    Uses 4h EMA_20 vs EMA_50 to determine trend direction:
    - Bullish (EMA_20 > EMA_50): only LONG signals pass through.
    - Bearish (EMA_20 < EMA_50): only SHORT signals pass through.
    - HOLD signals always pass through unchanged.

    The base strategy's code is never modified.
    """

    def __init__(self, base_strategy: BaseStrategy) -> None:
        self.base_strategy = base_strategy
        self.name = f"mtf_{base_strategy.name}"
        self.df_htf: Optional[pd.DataFrame] = None

    def set_htf_data(self, df_htf: pd.DataFrame) -> None:
        """Set the higher-timeframe DataFrame (e.g. 4h with ema_20, ema_50).

        Args:
            df_htf: OHLCV DataFrame at higher timeframe with EMA indicators.
        """
        self.df_htf = df_htf

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate signal from base strategy, then filter by HTF trend.

        Args:
            df: OHLCV DataFrame at the base timeframe (e.g. 30m).

        Returns:
            TradeSignal — original signal if aligned, HOLD if blocked.
        """
        signal = self.base_strategy.generate_signal(df)

        if self.df_htf is None or signal.signal == Signal.HOLD:
            return signal

        if len(self.df_htf) < 2:
            return signal

        ema20 = self.df_htf["ema_20"].iloc[-1]
        ema50 = self.df_htf["ema_50"].iloc[-1]

        if pd.isna(ema20) or pd.isna(ema50):
            return signal

        bullish = float(ema20) > float(ema50)

        # Block LONG in bearish trend
        if signal.signal == Signal.LONG and not bullish:
            logger.info(
                "[MTF] Blocked LONG — 4h trend bearish (EMA20=%.2f < EMA50=%.2f)",
                ema20, ema50,
            )
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=signal.symbol,
                price=signal.price,
                timestamp=signal.timestamp,
                metadata={"blocked_by": "mtf_filter", "original": "LONG"},
            )

        # Block SHORT in bullish trend
        if signal.signal == Signal.SHORT and bullish:
            logger.info(
                "[MTF] Blocked SHORT — 4h trend bullish (EMA20=%.2f > EMA50=%.2f)",
                ema20, ema50,
            )
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=signal.symbol,
                price=signal.price,
                timestamp=signal.timestamp,
                metadata={"blocked_by": "mtf_filter", "original": "SHORT"},
            )

        trend = "bullish" if bullish else "bearish"
        logger.debug("[MTF] 4h trend: %s — %s signal passed", trend, signal.signal.value)
        return signal

    def get_required_indicators(self) -> list[str]:
        """Delegate to base strategy."""
        return self.base_strategy.get_required_indicators()

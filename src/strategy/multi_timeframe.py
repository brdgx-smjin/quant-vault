"""DEPRECATED — Multi-timeframe analysis strategy.

Superseded by MultiTimeframeFilter in mtf_filter.py (decorator pattern).
Never walk-forward validated independently. mtf_filter.py wraps any BaseStrategy
with 4h EMA_20/50 trend filtering — used by all production components.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.indicators.basic import BasicIndicators
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class MultiTimeframeStrategy(BaseStrategy):
    """Combines signals from multiple timeframes for confirmation."""

    name = "multi_timeframe"

    def __init__(self, symbol: str = SYMBOL) -> None:
        self.symbol = symbol
        self.timeframe_data: dict[str, pd.DataFrame] = {}

    def set_timeframe_data(self, timeframe: str, df: pd.DataFrame) -> None:
        """Set OHLCV data for a specific timeframe.

        Args:
            timeframe: Timeframe string (e.g., '1h', '4h').
            df: OHLCV DataFrame for that timeframe.
        """
        self.timeframe_data[timeframe] = df

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate signal from multi-timeframe alignment.

        Args:
            df: Primary timeframe OHLCV data.

        Returns:
            TradeSignal.
        """
        last_close = float(df["close"].iloc[-1])
        last_ts = df.index[-1]

        signals: list[int] = []
        for tf, tf_df in self.timeframe_data.items():
            if len(tf_df) < 200:
                continue
            tf_df = BasicIndicators.add_all(tf_df)
            ema_20 = tf_df["ema_20"].iloc[-1]
            ema_50 = tf_df["ema_50"].iloc[-1]
            rsi = tf_df["rsi_14"].iloc[-1]

            if ema_20 > ema_50 and rsi < 70:
                signals.append(1)
            elif ema_20 < ema_50 and rsi > 30:
                signals.append(-1)
            else:
                signals.append(0)

        if not signals:
            return TradeSignal(signal=Signal.HOLD, symbol=self.symbol,
                               price=last_close, timestamp=last_ts)

        avg_signal = sum(signals) / len(signals)
        if avg_signal > 0.5:
            return TradeSignal(
                signal=Signal.LONG, symbol=self.symbol,
                price=last_close, timestamp=last_ts,
                confidence=avg_signal,
                metadata={"strategy": self.name, "tf_signals": signals},
            )
        elif avg_signal < -0.5:
            return TradeSignal(
                signal=Signal.SHORT, symbol=self.symbol,
                price=last_close, timestamp=last_ts,
                confidence=abs(avg_signal),
                metadata={"strategy": self.name, "tf_signals": signals},
            )

        return TradeSignal(signal=Signal.HOLD, symbol=self.symbol,
                           price=last_close, timestamp=last_ts)

    def get_required_indicators(self) -> list[str]:
        return ["ema_20", "ema_50", "rsi_14"]

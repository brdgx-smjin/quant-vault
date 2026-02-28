"""DEPRECATED — ROC (Rate of Change) Exhaustion mean-reversion strategy.

NOT recommended for production — suboptimal at 9-window WF (55%).
Use RSI_MR+MTF or DC+MTF instead.

Walk-Forward results (1h, with MTF 4h filter, ROC_6_3.0):
  Phase 15b (5w):  60% robustness, OOS +11.69%, Full +204%, DD 18.9%, PF 4.62
  Phase 15b (7w):  57% robustness, OOS +12.10%
  Phase 15b (9w):  55% robustness, OOS +10.84%, 12 trades

Standalone: marginal at 9w (55%). Too few trades for statistical confidence.
Portfolio value: ROC+RSI 50/50 = 77% rob (same ceiling as RSI+DC).
  Does NOT break the 77% ceiling — same W2/W6 negative structure.
Best config: roc_period=6, threshold=3.0, atr_sl=2.0, atr_tp=3.0.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class ROCExhaustionStrategy(BaseStrategy):
    """Fade ROC extremes: buy after extreme down-move, sell after extreme up-move.

    Args:
        roc_period: Lookback for ROC calculation.
        threshold: ROC level to trigger entry (e.g., 5.0 means ROC < -5%).
        atr_sl_mult: ATR multiplier for stop loss.
        atr_tp_mult: ATR multiplier for take profit.
        cooldown_bars: Minimum bars between entries.
    """

    name = "roc_exhaustion"

    def __init__(
        self,
        roc_period: int = 12,
        threshold: float = 5.0,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.roc_period = roc_period
        self.threshold = threshold
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate mean-reversion signal from ROC extremes.

        Args:
            df: OHLCV DataFrame with atr_14 column.

        Returns:
            TradeSignal.
        """
        min_bars = self.roc_period + 10
        if len(df) < min_bars:
            return self._hold(df)

        # Cooldown
        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        atr = last.get("atr_14")
        if pd.isna(atr):
            return self._hold(df)
        atr = float(atr)
        if atr <= 0:
            return self._hold(df)

        # ROC = (close - close_N) / close_N * 100
        close_n = float(df["close"].iloc[-self.roc_period - 1])
        if close_n <= 0:
            return self._hold(df)
        roc = (close - close_n) / close_n * 100

        # LONG: extreme negative ROC (price dropped too much too fast)
        if roc < -self.threshold:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult

            depth = abs(roc) - self.threshold
            confidence = min(1.0, 0.5 + depth / 10.0)

            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={
                    "strategy": self.name,
                    "roc": roc,
                    "roc_period": self.roc_period,
                    "threshold": self.threshold,
                },
            )

        # SHORT: extreme positive ROC (price rose too much too fast)
        if roc > self.threshold:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult

            depth = roc - self.threshold
            confidence = min(1.0, 0.5 + depth / 10.0)

            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={
                    "strategy": self.name,
                    "roc": roc,
                    "roc_period": self.roc_period,
                    "threshold": self.threshold,
                },
            )

        return self._hold(df)

    def _hold(self, df: pd.DataFrame) -> TradeSignal:
        return TradeSignal(
            signal=Signal.HOLD,
            symbol=self.symbol,
            price=float(df["close"].iloc[-1]),
            timestamp=df.index[-1],
        )

    def get_required_indicators(self) -> list[str]:
        return ["atr_14"]

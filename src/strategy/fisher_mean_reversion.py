"""DEPRECATED — Fisher Transform mean-reversion strategy with BB confirmation.

Fisher Transform applies arctanh to normalized price, amplifying extremes.

Walk-Forward results (9-window, date-aligned):
  Phase 28 standalone (1h + MTF):
    p5_t1.5:  22% rob, OOS -3.54%, 39 trades
    p5_t2.5:  44% rob, OOS -1.99%, 10 trades
    p9_t1.5:  33% rob, OOS -9.48%, 45 trades
    p13_t2.5: 66% rob, OOS +4.89%, 14 trades (BEST — too few trades)
  Phase 28 15m: 0 trades on ALL configs — NOT viable on 15m
  Phase 28 Z-Score MR was tested alongside, best = 66% rob, +7.62% OOS

VERDICT: Fisher does NOT improve portfolio. Best 66% rob with only 14 trades.
  Most configs 11-44% rob. DEPRECATED — use RSI, WillR, or CCI instead.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


def add_fisher(
    df: pd.DataFrame,
    length: int = 9,
) -> pd.DataFrame:
    """Add Fisher Transform columns to DataFrame.

    Args:
        df: OHLCV DataFrame.
        length: Lookback period for Fisher Transform.

    Returns:
        DataFrame with FISHERT_{length}_1 and FISHERTs_{length}_1 columns.
    """
    fisher_col = f"FISHERT_{length}_1"
    if fisher_col not in df.columns:
        result = ta.fisher(df["high"], df["low"], length=length)
        if result is not None:
            df = pd.concat([df, result], axis=1)
    return df


class FisherMeanReversionStrategy(BaseStrategy):
    """Fade extremes using Fisher Transform + Bollinger Band confirmation.

    Entry rules:
      LONG:  Fisher < -threshold AND close <= BB_lower
      SHORT: Fisher > +threshold AND close >= BB_upper

    Exit: ATR-based SL/TP.
    """

    name = "fisher_mean_reversion"

    def __init__(
        self,
        fisher_length: int = 9,
        threshold: float = 2.0,
        bb_column_lower: str = "BBL_20_2.0_2.0",
        bb_column_upper: str = "BBU_20_2.0_2.0",
        bb_proximity: float = 1.01,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.fisher_length = fisher_length
        self.threshold = threshold
        self.bb_lower = bb_column_lower
        self.bb_upper = bb_column_upper
        self.bb_proximity = bb_proximity
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self._last_entry_idx = -999
        self.symbol = symbol
        self._fisher_col = f"FISHERT_{fisher_length}_1"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate mean-reversion signal from Fisher Transform extremes + BB.

        Args:
            df: OHLCV DataFrame with Fisher Transform, BBL/BBU, atr_14.

        Returns:
            TradeSignal.
        """
        if len(df) < 30:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        fisher = last.get(self._fisher_col)
        atr = last.get("atr_14")
        bb_low = last.get(self.bb_lower)
        bb_up = last.get(self.bb_upper)

        if any(pd.isna(v) for v in [fisher, atr, bb_low, bb_up]):
            return self._hold(df)

        fisher = float(fisher)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        # Cooldown
        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        # LONG: Fisher oversold + near lower BB
        if fisher < -self.threshold and close <= bb_low * self.bb_proximity:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + abs(fisher + self.threshold) / 10.0)
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
                    "fisher": fisher,
                    "bb_touch": "lower",
                },
            )

        # SHORT: Fisher overbought + near upper BB
        if fisher > self.threshold and close >= bb_up / self.bb_proximity:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + abs(fisher - self.threshold) / 10.0)
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
                    "fisher": fisher,
                    "bb_touch": "upper",
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
        return [self._fisher_col, "atr_14", self.bb_lower, self.bb_upper]

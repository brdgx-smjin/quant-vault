"""DEPRECATED — EMA trend-following strategy with RSI and ATR.

Never walk-forward validated. Initial scaffold from project setup.
Use DonchianTrendStrategy+MTF for trend following instead (55% rob, 9w).
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class EMATrendStrategy(BaseStrategy):
    """Trade in the direction of the EMA trend with RSI confirmation.

    Entry rules:
      LONG:  EMA_fast > EMA_slow AND close > EMA_fast AND RSI < 65
      SHORT: EMA_fast < EMA_slow AND close < EMA_fast AND RSI > 35

    SL: entry +/- ATR * atr_sl_mult
    TP: entry +/- ATR * atr_tp_mult
    """

    name = "ema_trend"

    def __init__(
        self,
        fast_ema: int = 20,
        slow_ema: int = 50,
        rsi_long_max: float = 65.0,
        rsi_short_min: float = 35.0,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        require_pullback: bool = True,
        symbol: str = SYMBOL,
    ) -> None:
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.require_pullback = require_pullback
        self.symbol = symbol
        self._last_entry_idx = -999

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate trend-following signal.

        Args:
            df: OHLCV DataFrame with indicators (ema_20, ema_50, rsi_14, atr_14).

        Returns:
            TradeSignal.
        """
        if len(df) < self.slow_ema + 10:
            return self._hold(df)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(last["close"])
        ts = df.index[-1]

        ema_f = last.get(f"ema_{self.fast_ema}")
        ema_s = last.get(f"ema_{self.slow_ema}")
        rsi = last.get("rsi_14")
        atr = last.get("atr_14")

        if pd.isna(ema_f) or pd.isna(ema_s) or pd.isna(rsi) or pd.isna(atr):
            return self._hold(df)

        # Cooldown: minimum 10 bars between entries
        current_idx = len(df)
        if current_idx - self._last_entry_idx < 10:
            return self._hold(df)

        ema_f = float(ema_f)
        ema_s = float(ema_s)
        rsi = float(rsi)
        atr = float(atr)

        # Pullback check: price dipped toward fast EMA then bounced
        pullback_ok = True
        if self.require_pullback:
            prev_close = float(prev["close"])
            prev_low = float(prev["low"])
            prev_high = float(prev["high"])
            if ema_f > ema_s:
                # Bullish: previous bar low touched near fast EMA
                pullback_ok = prev_low <= ema_f * 1.005
            else:
                # Bearish: previous bar high touched near fast EMA
                pullback_ok = prev_high >= ema_f * 0.995

        # LONG signal
        if (
            ema_f > ema_s
            and close > ema_f
            and rsi < self.rsi_long_max
            and pullback_ok
        ):
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            confidence = min(1.0, 0.5 + (ema_f - ema_s) / ema_s * 10)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "rsi": rsi, "atr": atr},
            )

        # SHORT signal
        if (
            ema_f < ema_s
            and close < ema_f
            and rsi > self.rsi_short_min
            and pullback_ok
        ):
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            confidence = min(1.0, 0.5 + (ema_s - ema_f) / ema_s * 10)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "rsi": rsi, "atr": atr},
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
        return [f"ema_{self.fast_ema}", f"ema_{self.slow_ema}", "rsi_14", "atr_14"]

"""DEPRECATED — Stochastic Oscillator Mean Reversion strategy.

FAILED — Phase 11 results (1h):
  All variants have negative Full period returns (-15% to -70%).
  Best standalone: Stoch_21_20_80 — 80% rob (5w), OOS +9.18%, BUT Full -58%, PF 0.40
  Best +MTF: Stoch_21_20_80+MTF — 40% rob (5w), OOS +2.62%, Full -56%, PF 0.22
  7w/9w: Degrades to 42%/22% robustness — unreliable

  MTF filter HURTS Stochastic MR (opposite of RSI/VWAP):
    Stochastic crossover signals are too noisy for MTF to filter effectively.

  Conclusion: Stochastic Mean Reversion does NOT work on BTC/USDT 1h.
  RSI_MR+MTF (66% rob at 9w) and VWAP_MR+MTF (55% at 9w) are strictly superior.
  Kept for reference only.

Entry rules:
  LONG:  Slow %K < oversold AND %K crosses above %D
         AND close <= BB_lower * proximity
  SHORT: Slow %K > overbought AND %K crosses below %D
         AND close >= BB_upper / proximity
  SL: ATR-based. TP: ATR-based.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class StochMeanReversionStrategy(BaseStrategy):
    """Fade extremes using Stochastic Oscillator + BB confirmation.

    Uses Slow Stochastic with %K/%D crossover in extreme zones
    combined with Bollinger Band proximity for entry confirmation.
    """

    name = "stoch_mean_reversion"

    def __init__(
        self,
        stoch_k: int = 14,
        stoch_d: int = 3,
        stoch_smooth: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        bb_column_lower: str = "BBL_20_2.0_2.0",
        bb_column_upper: str = "BBU_20_2.0_2.0",
        bb_proximity: float = 1.01,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.stoch_smooth = stoch_smooth
        self.oversold = oversold
        self.overbought = overbought
        self.bb_lower = bb_column_lower
        self.bb_upper = bb_column_upper
        self.bb_proximity = bb_proximity
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999

        # pandas_ta stoch column names
        self._col_k = f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth}"
        self._col_d = f"STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate mean-reversion signal from Stochastic extremes + BB.

        Args:
            df: OHLCV DataFrame with stochastic and BB indicators.

        Returns:
            TradeSignal.
        """
        if len(df) < 30:
            return self._hold(df)

        # Cooldown
        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(last["close"])
        ts = df.index[-1]

        # Read indicators
        k_val = last.get(self._col_k)
        d_val = last.get(self._col_d)
        prev_k = prev.get(self._col_k)
        prev_d = prev.get(self._col_d)
        atr = last.get("atr_14")
        bb_low = last.get(self.bb_lower)
        bb_up = last.get(self.bb_upper)

        vals = [k_val, d_val, prev_k, prev_d, atr, bb_low, bb_up]
        if any(pd.isna(v) for v in vals):
            return self._hold(df)

        k_val = float(k_val)
        d_val = float(d_val)
        prev_k = float(prev_k)
        prev_d = float(prev_d)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        if atr <= 0:
            return self._hold(df)

        # Detect %K/%D crossover
        bullish_cross = prev_k <= prev_d and k_val > d_val
        bearish_cross = prev_k >= prev_d and k_val < d_val

        # LONG: oversold + bullish crossover + near lower BB
        if k_val < self.oversold and bullish_cross and close <= bb_low * self.bb_proximity:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + (self.oversold - k_val) / 100.0)
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
                    "stoch_k": k_val,
                    "stoch_d": d_val,
                    "bb_touch": "lower",
                },
            )

        # SHORT: overbought + bearish crossover + near upper BB
        if k_val > self.overbought and bearish_cross and close >= bb_up / self.bb_proximity:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + (k_val - self.overbought) / 100.0)
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
                    "stoch_k": k_val,
                    "stoch_d": d_val,
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
        return [self._col_k, self._col_d, "atr_14", self.bb_lower, self.bb_upper]

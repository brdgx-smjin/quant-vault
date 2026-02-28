"""DEPRECATED — MFI (Money Flow Index) Mean Reversion strategy.

Walk-Forward results (1h, with MTF 4h filter, MFI_20_20_80):
  Phase 15b (5w):  60% robustness, OOS +0.60% — barely positive
  Phase 15b (7w):  42% robustness, OOS -1.61%
  Phase 15b (9w):  22% robustness, OOS +0.17%, only 3 trades — FAILS

MFI signals too sparse and unreliable for BTC 1h. Volume-weighting
does not improve over pure RSI. Not recommended for production.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class MFIMeanReversionStrategy(BaseStrategy):
    """Fade MFI extremes: buy when volume-weighted momentum is oversold.

    Args:
        mfi_period: Lookback for MFI calculation.
        oversold_level: MFI below this triggers LONG.
        overbought_level: MFI above this triggers SHORT.
        atr_sl_mult: ATR multiplier for stop loss.
        atr_tp_mult: ATR multiplier for take profit.
        cooldown_bars: Minimum bars between entries.
    """

    name = "mfi_mean_reversion"

    def __init__(
        self,
        mfi_period: int = 14,
        oversold_level: float = 20.0,
        overbought_level: float = 80.0,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.mfi_period = mfi_period
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999
        self._mfi_col = f"MFI_{mfi_period}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate mean-reversion signal from MFI extremes.

        Args:
            df: OHLCV DataFrame with MFI_{period} and atr_14 columns.

        Returns:
            TradeSignal.
        """
        min_bars = self.mfi_period + 10
        if len(df) < min_bars:
            return self._hold(df)

        # Cooldown
        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        mfi = last.get(self._mfi_col)
        atr = last.get("atr_14")

        if pd.isna(mfi) or pd.isna(atr):
            return self._hold(df)

        mfi = float(mfi)
        atr = float(atr)
        if atr <= 0:
            return self._hold(df)

        # LONG: MFI deeply oversold
        if mfi < self.oversold_level:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult

            depth = self.oversold_level - mfi
            confidence = min(1.0, 0.5 + depth / 40.0)

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
                    "mfi": mfi,
                    "depth": depth,
                },
            )

        # SHORT: MFI deeply overbought
        if mfi > self.overbought_level:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult

            depth = mfi - self.overbought_level
            confidence = min(1.0, 0.5 + depth / 40.0)

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
                    "mfi": mfi,
                    "depth": depth,
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
        return [self._mfi_col, "atr_14"]

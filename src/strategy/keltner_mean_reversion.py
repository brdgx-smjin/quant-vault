"""DEPRECATED — Keltner Channel Mean Reversion strategy.

Walk-Forward results (1h, with MTF 4h filter, kc_mult=2.0):
  Phase 15 (5w):  80% robustness, OOS +14.37%, Full +150%, DD 13.8%, PF 3.22
  Phase 15 (7w):  57% robustness, OOS +6.23%  — degrades
  Phase 15 (9w):  44% robustness, OOS +10.18% — FAILS threshold

Same degradation pattern as BBSqueeze at high window counts.
Signals are too correlated with RSI/VWAP MR (same negative windows W2, W6).
NOT recommended for production — no diversification value over existing MR strategies.
Use RSI_MR+MTF or CCI_MR+MTF instead.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class KeltnerMeanReversionStrategy(BaseStrategy):
    """Fade Keltner Channel extremes: buy below lower band, sell above upper.

    Keltner Channel = EMA(center_period) +/- kc_mult * ATR(14).
    Entry when price trades beyond the outer band (overextended).

    Args:
        center_period: EMA period for channel center (default 20).
        kc_mult: ATR multiplier for band width (higher = fewer signals).
        atr_sl_mult: ATR multiplier for stop loss distance.
        atr_tp_mult: ATR multiplier for take profit distance.
        cooldown_bars: Minimum bars between entries.
    """

    name = "keltner_mean_reversion"

    def __init__(
        self,
        center_period: int = 20,
        kc_mult: float = 2.0,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.center_period = center_period
        self.kc_mult = kc_mult
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999
        self._ema_col = f"ema_{center_period}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate mean-reversion signal from Keltner Channel extremes.

        Args:
            df: OHLCV DataFrame with ema_{center_period} and atr_14 columns.

        Returns:
            TradeSignal.
        """
        min_bars = self.center_period + 10
        if len(df) < min_bars:
            return self._hold(df)

        # Cooldown
        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        ema = last.get(self._ema_col)
        atr = last.get("atr_14")

        if pd.isna(ema) or pd.isna(atr):
            return self._hold(df)

        ema = float(ema)
        atr = float(atr)

        if atr <= 0:
            return self._hold(df)

        # Keltner Channel bands
        upper_band = ema + self.kc_mult * atr
        lower_band = ema - self.kc_mult * atr

        # LONG: price below lower Keltner band (overextended to downside)
        if close < lower_band:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult

            # Confidence: how far beyond the band
            extension = (lower_band - close) / atr
            confidence = min(1.0, 0.5 + extension * 0.15)

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
                    "ema": ema,
                    "upper_band": upper_band,
                    "lower_band": lower_band,
                    "extension_atr": extension,
                },
            )

        # SHORT: price above upper Keltner band (overextended to upside)
        if close > upper_band:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult

            extension = (close - upper_band) / atr
            confidence = min(1.0, 0.5 + extension * 0.15)

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
                    "ema": ema,
                    "upper_band": upper_band,
                    "lower_band": lower_band,
                    "extension_atr": extension,
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
        return [self._ema_col, "atr_14"]

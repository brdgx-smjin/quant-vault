"""Williams %R mean-reversion strategy with Bollinger Band confirmation.

MUST use with MultiTimeframeFilter(4h EMA) for reliability.

Williams %R is a momentum oscillator ranging from -100 to 0:
  - Oversold:   %R < -80 (close near period low)
  - Overbought: %R > -20 (close near period high)

Similar to RSI mean-reversion but faster: responds more quickly to price
changes by directly comparing close to the high-low range. This may
capture different entry timings than RSI.

Walk-Forward results (9-window, date-aligned):
  Phase 25 standalone:
    1h p14_t90+MTF:  77% rob, +19.17% OOS (best 1h config)
    1h p14_t80+MTF:  55% rob
    15m p14_t90+MTF: 55% rob (15m not viable)
  Phase 25 as 4th Cross-TF component (1hRSI/1hDC/15mRSI/1hWR):
    Best weight 15/50/10/25: 88% rob, +23.98% OOS
    303/375 weight combos (80.8%) achieve 88% robustness
    11/12 param perturbations at 88% — STRONG stability
    +5.17% OOS improvement over 3-comp baseline (+18.81%)
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class WilliamsRMeanReversionStrategy(BaseStrategy):
    """Fade extremes using Williams %R + Bollinger Band confirmation.

    Entry rules:
      LONG:  %R < -oversold AND close <= BB_lower (oversold + at band)
      SHORT: %R > -overbought AND close >= BB_upper (overbought + at band)

    Exit: ATR-based SL/TP.
    """

    name = "willr_mean_reversion"

    def __init__(
        self,
        willr_period: int = 14,
        oversold_level: float = 80.0,
        overbought_level: float = 80.0,
        bb_column_lower: str = "BBL_20_2.0_2.0",
        bb_column_upper: str = "BBU_20_2.0_2.0",
        bb_proximity: float = 1.01,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.willr_period = willr_period
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        self.bb_lower = bb_column_lower
        self.bb_upper = bb_column_upper
        self.bb_proximity = bb_proximity
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self._last_entry_idx = -999
        self.symbol = symbol
        self._willr_col = f"WILLR_{willr_period}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate mean-reversion signal from Williams %R extremes + BB.

        Args:
            df: OHLCV DataFrame with Williams %R, BBL/BBU, atr_14.

        Returns:
            TradeSignal.
        """
        if len(df) < 30:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        willr = last.get(self._willr_col)
        atr = last.get("atr_14")
        bb_low = last.get(self.bb_lower)
        bb_up = last.get(self.bb_upper)

        if any(pd.isna(v) for v in [willr, atr, bb_low, bb_up]):
            return self._hold(df)

        willr = float(willr)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        # Cooldown
        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        # Williams %R divergence check
        bullish_div = False
        bearish_div = False
        if len(df) >= 10:
            recent_lows = df["low"].iloc[-10:]
            recent_willr = df[self._willr_col].iloc[-10:]
            if not recent_willr.isna().all():
                price_at_min = recent_lows.idxmin()
                if price_at_min != df.index[-1]:
                    old_willr = df[self._willr_col].loc[price_at_min]
                    if not pd.isna(old_willr):
                        # Price new low but %R higher (less oversold) = bullish div
                        if (close <= float(recent_lows.min()) * 1.005
                                and willr > float(old_willr)):
                            bullish_div = True

                recent_highs = df["high"].iloc[-10:]
                price_at_max = recent_highs.idxmax()
                if price_at_max != df.index[-1]:
                    old_willr_h = df[self._willr_col].loc[price_at_max]
                    if not pd.isna(old_willr_h):
                        # Price new high but %R lower (less overbought) = bearish div
                        if (close >= float(recent_highs.max()) * 0.995
                                and willr < float(old_willr_h)):
                            bearish_div = True

        # LONG: oversold (%R < -oversold_level) + near lower BB
        if willr < -self.oversold_level and close <= bb_low * self.bb_proximity:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            confidence = 0.55 + (-willr - self.oversold_level) / 200.0
            if bullish_div:
                confidence += 0.15
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=min(1.0, confidence),
                stop_loss=sl,
                take_profit=tp,
                metadata={
                    "strategy": self.name,
                    "willr": willr,
                    "bb_touch": "lower",
                    "divergence": bullish_div,
                },
            )

        # SHORT: overbought (%R > -overbought_level) + near upper BB
        if willr > -self.overbought_level and close >= bb_up / self.bb_proximity:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            confidence = 0.55 + (willr + self.overbought_level) / 200.0
            if bearish_div:
                confidence += 0.15
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=min(1.0, confidence),
                stop_loss=sl,
                take_profit=tp,
                metadata={
                    "strategy": self.name,
                    "willr": willr,
                    "bb_touch": "upper",
                    "divergence": bearish_div,
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
        return [self._willr_col, "atr_14", self.bb_lower, self.bb_upper]

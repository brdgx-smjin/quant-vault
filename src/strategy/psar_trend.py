"""Parabolic SAR trend-following strategy — DEPRECATED.

MUST use with MultiTimeframeFilter(4h EMA) for reliability.

Parabolic SAR (Stop and Reverse) uses an acceleration factor to create
a trailing stop that accelerates toward price. Key differences from
Supertrend and Donchian:
  - Acceleration factor speeds up as trend continues (vs fixed bands)
  - Auto-reverses position on flip (natural trend-following)
  - Better at capturing extended trends but whipsaws in range

Entry rules:
  LONG:  PSAR flips from above to below price (bearish → bullish)
         + volume > vol_mult * 20-bar average
  SHORT: PSAR flips from below to above price (bullish → bearish)
         + volume > vol_mult * 20-bar average

Exit: ATR-based SL and RR-based TP (or timeout via engine).

Walk-Forward results (9-window, date-aligned, 1h + MTF):
  Phase 36 standalone:
    AF0.01_MAF0.15_RR2.0: 77% rob, +27.62% OOS, 21 trades (BEST)
    8 configs at 77%, 8 at 66%. AF0.01 dominant (slow acceleration).
    max_af has minimal effect — PSAR barely accelerates with AF0.01.
  Phase 36 portfolio:
    5-comp (10/40/10/20/20): 77% rob, +25.07% OOS — DROPS from 88%
    Replace DC (15/50/10/25): 77% rob, +25.49% OOS — WORSE
    Replace WillR (15/50/10/25): 88% rob, +23.68% OOS — maintains 88%
      but -0.30% lower return than WillR (+23.98%).
      Closest replacement result of all tested indicators.
  15m: 44% rob, -7.47% OOS — NOT viable
  CONCLUSION: Strong standalone (+27.62%) but does NOT improve portfolio.
  13th indicator tested as 5th/replacement component — ALL fail.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class PSARTrendStrategy(BaseStrategy):
    """Trade Parabolic SAR reversals for trend following.

    Enters when the PSAR indicator flips direction, confirmed by volume.

    Args:
        af_step: Acceleration factor step size (also initial AF).
        max_af: Maximum acceleration factor.
        atr_sl_mult: ATR multiplier for stop loss.
        rr_ratio: Reward-to-risk ratio for take profit.
        vol_mult: Minimum volume vs 20-bar average.
        cooldown_bars: Minimum bars between entries.
    """

    name = "psar_trend"

    def __init__(
        self,
        af_step: float = 0.02,
        max_af: float = 0.20,
        atr_sl_mult: float = 2.0,
        rr_ratio: float = 2.0,
        vol_mult: float = 0.8,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.af_step = af_step
        self.max_af = max_af
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.vol_mult = vol_mult
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999
        self._psar_long_col = f"PSARl_{af_step}_{max_af}"
        self._psar_short_col = f"PSARs_{af_step}_{max_af}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate trend signal from Parabolic SAR flip.

        Args:
            df: OHLCV DataFrame with PSAR columns and atr_14.

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

        atr = last.get("atr_14")
        volume = last.get("volume")

        psar_long = last.get(self._psar_long_col)
        psar_short = last.get(self._psar_short_col)
        prev_psar_long = prev.get(self._psar_long_col)
        prev_psar_short = prev.get(self._psar_short_col)

        if pd.isna(atr) or pd.isna(volume):
            return self._hold(df)

        atr = float(atr)
        volume = float(volume)

        if atr <= 0:
            return self._hold(df)

        # Volume confirmation
        vol_avg = float(df["volume"].iloc[-20:].mean())
        if vol_avg <= 0 or volume < vol_avg * self.vol_mult:
            return self._hold(df)

        # Detect PSAR flip:
        # LONG flip: previous had short PSAR (above price), now has long PSAR (below price)
        long_flip = (
            pd.notna(psar_long) and pd.isna(prev_psar_long)
            and pd.isna(psar_short) and pd.notna(prev_psar_short)
        )

        # SHORT flip: previous had long PSAR (below price), now has short PSAR (above price)
        short_flip = (
            pd.notna(psar_short) and pd.isna(prev_psar_short)
            and pd.isna(psar_long) and pd.notna(prev_psar_long)
        )

        if long_flip:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_sl_mult * self.rr_ratio

            psar_dist = abs(close - float(psar_long)) / close * 100
            confidence = min(1.0, 0.5 + psar_dist * 0.2)

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
                    "psar_value": float(psar_long),
                    "psar_dist_pct": psar_dist,
                    "volume_ratio": volume / vol_avg,
                },
            )

        if short_flip:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_sl_mult * self.rr_ratio

            psar_dist = abs(close - float(psar_short)) / close * 100
            confidence = min(1.0, 0.5 + psar_dist * 0.2)

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
                    "psar_value": float(psar_short),
                    "psar_dist_pct": psar_dist,
                    "volume_ratio": volume / vol_avg,
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
        return [self._psar_long_col, self._psar_short_col, "atr_14"]

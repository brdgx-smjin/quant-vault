"""Supertrend trend-following strategy — DEPRECATED.

ATR-based dynamic support/resistance with direction flip entry.
Different from Donchian (fixed-period channel breakout):
  - Uses ATR for adaptive band width (volatile = wider bands)
  - Direction flip mechanism (price crosses ST line)
  - Tends to stay in trend longer than channel breakouts

Entry rules:
  LONG:  Supertrend direction flips from -1 to 1 (bullish)
         + volume > vol_mult * 20-bar average
  SHORT: Supertrend direction flips from 1 to -1 (bearish)
         + volume > vol_mult * 20-bar average

  Exit via ATR-based SL and RR-based TP (or timeout via engine).

Walk-Forward results (9-window, date-aligned, 1h + MTF):
  Phase 35 standalone:
    L14_M3.5_RR2.5: 77% rob, +16.15% OOS, 11 trades (BEST)
    L20_M2.0_RR1.5: 77% rob, +12.62% OOS, 18 trades
    L14_M3.5_RR2.0: 77% rob, +12.54% OOS, 11 trades
    6 configs at 77%, 15 at 66%, 11 at 55%, 16 at 44%
    Pattern: higher multiplier → fewer trades → higher robustness
  Phase 35 portfolio:
    5-comp (10/40/10/20/20): 77% rob, +22.35% OOS — DROPS from 88%
    Replace DC (15/50/10/25): 77% rob, +18.77% OOS — WORSE than DC
    Replace WillR (15/50/10/25): 77% rob, +20.31% OOS — WORSE than WillR
  15m: 22% rob, +0.28% OOS — NOT viable
  CONCLUSION: Decent standalone (77%) but degrades portfolio from 88% to 77%.
  Supertrend is the 12th indicator tested as 5th/replacement comp — ALL fail.
  Donchian remains the superior trend-following component for this portfolio.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class SupertrendTrendStrategy(BaseStrategy):
    """Trade Supertrend direction flips for trend following.

    Enters when the Supertrend indicator flips direction, confirmed by volume.

    Args:
        st_length: ATR lookback period for Supertrend calculation.
        st_multiplier: ATR multiplier for band distance.
        atr_sl_mult: ATR multiplier for stop loss.
        rr_ratio: Reward-to-risk ratio for take profit.
        vol_mult: Minimum volume vs 20-bar average.
        cooldown_bars: Minimum bars between entries.
    """

    name = "supertrend_trend"

    def __init__(
        self,
        st_length: int = 10,
        st_multiplier: float = 3.0,
        atr_sl_mult: float = 2.0,
        rr_ratio: float = 2.0,
        vol_mult: float = 0.8,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.st_length = st_length
        self.st_multiplier = st_multiplier
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.vol_mult = vol_mult
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999
        self._st_col = f"SUPERT_{st_length}_{st_multiplier}"
        self._std_col = f"SUPERTd_{st_length}_{st_multiplier}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate trend signal from Supertrend direction flip.

        Args:
            df: OHLCV DataFrame with Supertrend columns and atr_14.

        Returns:
            TradeSignal.
        """
        min_bars = max(self.st_length + 10, 30)
        if len(df) < min_bars:
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
        st_dir = last.get(self._std_col)
        st_dir_prev = prev.get(self._std_col)
        st_val = last.get(self._st_col)

        if any(pd.isna(v) for v in [atr, volume, st_dir, st_dir_prev, st_val]):
            return self._hold(df)

        atr = float(atr)
        volume = float(volume)
        st_dir = float(st_dir)
        st_dir_prev = float(st_dir_prev)
        st_val = float(st_val)

        if atr <= 0:
            return self._hold(df)

        # Volume confirmation
        vol_avg = float(df["volume"].iloc[-20:].mean())
        if vol_avg <= 0 or volume < vol_avg * self.vol_mult:
            return self._hold(df)

        # LONG: direction flips from -1 to 1 (bearish → bullish)
        if st_dir == 1.0 and st_dir_prev == -1.0:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_sl_mult * self.rr_ratio

            # Confidence based on distance from ST line
            dist_pct = abs(close - st_val) / close * 100
            confidence = min(1.0, 0.5 + dist_pct * 0.3)

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
                    "st_value": st_val,
                    "st_direction": st_dir,
                    "dist_pct": dist_pct,
                    "volume_ratio": volume / vol_avg,
                },
            )

        # SHORT: direction flips from 1 to -1 (bullish → bearish)
        if st_dir == -1.0 and st_dir_prev == 1.0:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_sl_mult * self.rr_ratio

            dist_pct = abs(close - st_val) / close * 100
            confidence = min(1.0, 0.5 + dist_pct * 0.3)

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
                    "st_value": st_val,
                    "st_direction": st_dir,
                    "dist_pct": dist_pct,
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
        return [self._st_col, self._std_col, "atr_14"]

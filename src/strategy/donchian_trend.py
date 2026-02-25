"""Donchian Channel Trend Following strategy.

Classic trend-following system: enter on N-bar channel breakouts.
Fundamentally different from existing strategies:
  - BBSqueeze: breakout from volatility compression (squeeze → expansion)
  - RSI/VWAP: mean reversion (fade extremes)
  - Donchian: pure trend following (ride sustained directional moves)

This provides low correlation with mean-reversion strategies, making it
a strong portfolio diversifier.

Entry rules:
  LONG:  close > highest high of last entry_period bars (upside breakout)
         + volume > vol_mult * 20-bar average
  SHORT: close < lowest low of last entry_period bars (downside breakout)
         + volume > vol_mult * 20-bar average

  Exit via ATR-based SL and RR-based TP (or trailing via engine).

Designed to be used with MultiTimeframeFilter(4h EMA) for best results.

Walk-Forward results: pending Phase 12 validation.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class DonchianTrendStrategy(BaseStrategy):
    """Trade N-bar channel breakouts for trend following.

    Enters when price breaks above/below the Donchian channel
    (highest high / lowest low of last N bars), confirmed by volume.

    Args:
        entry_period: Lookback for channel breakout (e.g., 24 = 1 day on 1h).
        exit_period: Lookback for exit channel (shorter, for faster exits).
            If 0, uses SL/TP only.
        atr_sl_mult: ATR multiplier for stop loss.
        rr_ratio: Reward-to-risk ratio for take profit.
        vol_mult: Minimum volume vs 20-bar average.
        cooldown_bars: Minimum bars between entries.
        require_close_break: If True, close must be beyond channel.
            If False, high/low touch is enough.
    """

    name = "donchian_trend"

    def __init__(
        self,
        entry_period: int = 24,
        exit_period: int = 0,
        atr_sl_mult: float = 2.0,
        rr_ratio: float = 2.0,
        vol_mult: float = 0.8,
        cooldown_bars: int = 6,
        require_close_break: bool = True,
        symbol: str = SYMBOL,
    ) -> None:
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.vol_mult = vol_mult
        self.cooldown_bars = cooldown_bars
        self.require_close_break = require_close_break
        self.symbol = symbol
        self._last_entry_idx = -999

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate trend signal from Donchian channel breakout.

        Args:
            df: OHLCV DataFrame with atr_14.

        Returns:
            TradeSignal.
        """
        min_bars = self.entry_period + 10
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
        volume = last.get("volume")

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

        # Donchian channel: exclude current bar to avoid look-ahead
        lookback = df.iloc[-(self.entry_period + 1):-1]
        channel_high = float(lookback["high"].max())
        channel_low = float(lookback["low"].min())

        # LONG: close breaks above channel high
        if close > channel_high:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_sl_mult * self.rr_ratio

            # Confidence based on breakout strength
            breakout_pct = (close - channel_high) / channel_high * 100
            confidence = min(1.0, 0.5 + breakout_pct * 0.5)

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
                    "channel_high": channel_high,
                    "channel_low": channel_low,
                    "breakout_pct": breakout_pct,
                    "volume_ratio": volume / vol_avg,
                },
            )

        # SHORT: close breaks below channel low
        if close < channel_low:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_sl_mult * self.rr_ratio

            breakout_pct = (channel_low - close) / channel_low * 100
            confidence = min(1.0, 0.5 + breakout_pct * 0.5)

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
                    "channel_high": channel_high,
                    "channel_low": channel_low,
                    "breakout_pct": breakout_pct,
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
        return ["atr_14"]

"""VWAP Mean Reversion strategy for crypto futures.

Uses rolling VWAP with standard deviation bands.
Entry when price deviates significantly from VWAP, targeting reversion.
MUST use with MultiTimeframeFilter(4h EMA) for reliability.

Walk-Forward results (1h, with MTF 4h filter):
  Phase 10 (5w):  80% robustness, OOS +8.91%, Full +152% (overfitted), DD 11%
  Phase 12 (7w):  71% robustness, OOS +3.43%
  Phase 14 (9w):  55% robustness, OOS +11.57%, Full +119.57%, DD 11.7%, PF 2.56
  Standalone (no MTF): 60% rob, OOS -1.48%, DD 40% — COLLAPSES

Phase 8 results (7w):
  VWAP_p24_b2.0:  71% rob, OOS +5.91%, Full +139.94%, DD 11.7%, PF 2.59
  VWAP_p24_b1.5:  71% rob, OOS +10.10%, Full +133.11%, DD 13.6%, PF 2.24
  VWAP_p48_b2.0:  57% rob, OOS +4.60%, Full +132.06%, DD 9.7%, PF 2.98
  VWAP+DC 50/50:  86% rob (7w), OOS +22.23% — strong portfolio component

Degrades at 9w (55%). Best as portfolio component: VWAP+DC 50/50 = 86% rob (7w).
Best config: vwap_period=24, band_mult=2.0, rsi=35, sl=2.0, cool=4.
Always use with MTF(4h EMA). Full return is overfitted — rely on WF OOS only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class VWAPMeanReversionStrategy(BaseStrategy):
    """Fade price deviations from rolling VWAP.

    Entry rules:
      LONG:  close < VWAP - band_mult * std  AND  RSI < rsi_threshold
      SHORT: close > VWAP + band_mult * std  AND  RSI > (100 - rsi_threshold)

    TP target: VWAP (mean reversion).
    SL: ATR-based.

    Walk-Forward results (Phase 10, 1h, with MTF 4h filter):
      5w: 80% robustness, OOS +8.91%, Full +152% (overfitted), DD 11%, PF 2.71
      7w: 71% robustness, OOS +3.28%
      9w: 67% robustness, OOS +11.55%
      Standalone (no MTF): collapses — 60% rob, OOS -1.48%, DD 40%

    Best config: vwap_period=24, band_mult=2.0, rsi=35, sl=2.0, cool=4.
    Always use with MTF(4h EMA). Full return (~152%) likely overfitted.
    """

    name = "vwap_mean_reversion"

    def __init__(
        self,
        vwap_period: int = 48,
        band_mult: float = 2.0,
        rsi_threshold: float = 35.0,
        atr_sl_mult: float = 2.0,
        tp_to_vwap_pct: float = 0.8,
        cooldown_bars: int = 6,
        min_volume_mult: float = 0.8,
        symbol: str = SYMBOL,
    ) -> None:
        """Initialize VWAP Mean Reversion.

        Args:
            vwap_period: Rolling window for VWAP calculation.
            band_mult: Std dev multiplier for entry bands.
            rsi_threshold: RSI threshold for oversold (symmetric for overbought).
            atr_sl_mult: ATR multiplier for stop loss.
            tp_to_vwap_pct: Fraction of distance to VWAP for TP (1.0 = full VWAP).
            cooldown_bars: Minimum bars between entries.
            min_volume_mult: Minimum volume vs 20-bar avg to confirm.
            symbol: Trading symbol.
        """
        self.vwap_period = vwap_period
        self.band_mult = band_mult
        self.rsi_threshold = rsi_threshold
        self.atr_sl_mult = atr_sl_mult
        self.tp_to_vwap_pct = tp_to_vwap_pct
        self.cooldown_bars = cooldown_bars
        self.min_volume_mult = min_volume_mult
        self.symbol = symbol
        self._last_entry_idx = -999
        self._vwap_cache: pd.Series | None = None
        self._vwap_std_cache: pd.Series | None = None

    def _compute_vwap(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Compute rolling VWAP and its standard deviation.

        Returns:
            Tuple of (vwap, vwap_std) Series.
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"]

        # Rolling VWAP = sum(TP * V) / sum(V)
        tp_vol = typical_price * vol
        vwap = tp_vol.rolling(self.vwap_period).sum() / vol.rolling(self.vwap_period).sum()

        # Standard deviation of price from VWAP
        deviation = df["close"] - vwap
        vwap_std = deviation.rolling(self.vwap_period).std()

        return vwap, vwap_std

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate mean-reversion signal from VWAP deviation.

        Args:
            df: OHLCV DataFrame with rsi_14, atr_14, volume.

        Returns:
            TradeSignal.
        """
        min_bars = max(self.vwap_period + 10, 60)
        if len(df) < min_bars:
            return self._hold(df)

        # Compute VWAP
        vwap, vwap_std = self._compute_vwap(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        rsi = last.get("rsi_14")
        atr = last.get("atr_14")
        volume = float(last["volume"])

        if any(pd.isna(v) for v in [rsi, atr, vwap.iloc[-1], vwap_std.iloc[-1]]):
            return self._hold(df)

        rsi = float(rsi)
        atr = float(atr)
        current_vwap = float(vwap.iloc[-1])
        current_std = float(vwap_std.iloc[-1])

        if current_std <= 0 or atr <= 0:
            return self._hold(df)

        # Cooldown check
        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        # Volume confirmation: must have reasonable volume
        vol_avg = df["volume"].iloc[-20:].mean()
        if volume < vol_avg * self.min_volume_mult:
            return self._hold(df)

        # Z-score: how many std devs from VWAP
        z_score = (close - current_vwap) / current_std

        lower_band = current_vwap - self.band_mult * current_std
        upper_band = current_vwap + self.band_mult * current_std

        # LONG: price below lower band + RSI oversold
        if close < lower_band and rsi < self.rsi_threshold:
            sl = close - atr * self.atr_sl_mult
            # TP: partial reversion to VWAP
            tp_distance = (current_vwap - close) * self.tp_to_vwap_pct
            tp = close + tp_distance

            confidence = min(1.0, 0.5 + abs(z_score) * 0.1)
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
                    "z_score": z_score,
                    "vwap": current_vwap,
                    "rsi": rsi,
                },
            )

        # SHORT: price above upper band + RSI overbought
        if close > upper_band and rsi > (100 - self.rsi_threshold):
            sl = close + atr * self.atr_sl_mult
            tp_distance = (close - current_vwap) * self.tp_to_vwap_pct
            tp = close - tp_distance

            confidence = min(1.0, 0.5 + abs(z_score) * 0.1)
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
                    "z_score": z_score,
                    "vwap": current_vwap,
                    "rsi": rsi,
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
        return ["rsi_14", "atr_14"]

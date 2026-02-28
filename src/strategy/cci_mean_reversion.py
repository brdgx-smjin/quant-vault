"""CCI (Commodity Channel Index) mean-reversion strategy.

Fundamentally different from RSI mean reversion:
  - RSI uses close-to-close magnitude of gains vs losses, bounded 0-100
  - CCI uses typical price deviation from SMA, UNBOUNDED (-300 to +300 common)
  - CCI responds more sharply to price spikes → faster entry/exit signals
  - CCI can stay extended longer in trends → better for catching reversals

This provides LOW CORRELATION with RSI, making it a portfolio diversifier.

MUST use with MultiTimeframeFilter(4h EMA) — standalone is catastrophic
(Full -40% to -73% without MTF).

Entry rules:
  LONG:  CCI < -oversold_level (extreme oversold)
         + volume > vol_mult * 20-bar average (optional)
  SHORT: CCI > +overbought_level (extreme overbought)
         + volume > vol_mult * 20-bar average (optional)

Exit via ATR-based SL and TP, or max_hold_bars in engine.

Walk-Forward results (1h, best config CCI_20_200+MTF):
  Phase 13 (5w):  80% robustness, OOS +16.24%, Full +84.77%, DD 16.4%, PF 1.80
  Phase 13 (7w):  57% robustness, OOS +11.65%
  Phase 13 (9w):  66% robustness, OOS +13.48%, 27 trades

Phase 18: Tested as 4th cross-TF portfolio component (25/25/25/25).
  W2 CCI = -2.09% → still negative. Result: 88% rob, +16.93% OOS.
  Same robustness as 3-component, but LOWER return. Not worth the complexity.

Best config: cci_period=20, oversold/overbought=200, atr_sl=2.0, atr_tp=3.0.
Note: High threshold (200) is key — CCI=100 is too noisy.

Portfolio value (7w):
  DC+CCI 50/50: 85% robustness, OOS +24.46% — matches VWAP+DC as best portfolio

Phase 23 — 15m CCI (9w):
  Standalone: 66% rob, +7.85% OOS, 127 trades
  As 4th component (30/20/30/20): 88% rob but +15.96% OOS (lower than baseline)

Phase 24 — 15m CCI as REPLACEMENT for 15m RSI (3-component):
  RSI/DC/CCI15 30/30/40: 88% rob, +14.65% OOS — 88% achievable but lower return
  RSI/DC/CCI15 20/50/30: 77% rob, +17.67% OOS
  RSI/DC/CCI15 33/33/34: 77% rob, +15.29% OOS
  Conclusion: 15m CCI can achieve 88% rob but with LOWER return than 15m RSI.
  15m RSI remains the superior 15m component for the cross-TF portfolio.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class CCIMeanReversionStrategy(BaseStrategy):
    """Fade CCI extremes: buy when deeply oversold, sell when deeply overbought.

    Args:
        cci_period: Lookback for CCI calculation.
        oversold_level: CCI below -this triggers LONG (e.g., 100 → CCI < -100).
        overbought_level: CCI above +this triggers SHORT (e.g., 100 → CCI > +100).
        atr_sl_mult: ATR multiplier for stop loss.
        atr_tp_mult: ATR multiplier for take profit.
        vol_mult: Minimum volume vs 20-bar average (0 = disabled).
        cooldown_bars: Minimum bars between entries.
    """

    name = "cci_mean_reversion"

    def __init__(
        self,
        cci_period: int = 20,
        oversold_level: float = 100.0,
        overbought_level: float = 100.0,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        vol_mult: float = 0.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.cci_period = cci_period
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.vol_mult = vol_mult
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999
        self._cci_col = f"CCI_{cci_period}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate mean-reversion signal from CCI extremes.

        Args:
            df: OHLCV DataFrame with CCI_{period} and atr_14 columns.

        Returns:
            TradeSignal.
        """
        min_bars = self.cci_period + 10
        if len(df) < min_bars:
            return self._hold(df)

        # Cooldown
        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        cci = last.get(self._cci_col)
        atr = last.get("atr_14")

        if pd.isna(cci) or pd.isna(atr):
            return self._hold(df)

        cci = float(cci)
        atr = float(atr)

        if atr <= 0:
            return self._hold(df)

        # Volume filter (optional)
        if self.vol_mult > 0:
            volume = float(last.get("volume", 0))
            vol_avg = float(df["volume"].iloc[-20:].mean())
            if vol_avg <= 0 or volume < vol_avg * self.vol_mult:
                return self._hold(df)

        # LONG: CCI deeply oversold (e.g., CCI < -100)
        if cci < -self.oversold_level:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult

            # Confidence: deeper oversold → higher confidence
            depth = abs(cci) - self.oversold_level
            confidence = min(1.0, 0.5 + depth / 200.0)

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
                    "cci": cci,
                    "cci_period": self.cci_period,
                    "depth": depth,
                },
            )

        # SHORT: CCI deeply overbought (e.g., CCI > +100)
        if cci > self.overbought_level:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult

            depth = cci - self.overbought_level
            confidence = min(1.0, 0.5 + depth / 200.0)

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
                    "cci": cci,
                    "cci_period": self.cci_period,
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
        return [self._cci_col, "atr_14"]

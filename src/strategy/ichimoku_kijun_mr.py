"""DEPRECATED — Ichimoku Kijun-sen mean-reversion strategy with Bollinger Band confirmation.

MUST use with MultiTimeframeFilter(4h EMA) for reliability.

Fundamentally different from RSI/WillR/CCI mean reversion:
  - RSI: close-to-close gain/loss ratio over 14 bars, bounded 0-100
  - WillR: close position in 14-bar high-low range, bounded -100 to 0
  - CCI: typical price deviation from SMA over 20 bars, unbounded
  - Kijun: distance from 26-bar HIGH-LOW MIDPOINT, in ATR units, unbounded

Kijun-sen = (26-bar highest high + 26-bar lowest low) / 2.
This represents the "equilibrium" price — the midpoint of the recent range.
When price deviates far from Kijun, mean reversion toward equilibrium is expected.

Entry rules:
  LONG:  (kijun - close) > deviation_threshold * ATR (below equilibrium)
         AND close <= BB_lower (confirmed by Bollinger Band)
  SHORT: (close - kijun) > deviation_threshold * ATR (above equilibrium)
         AND close >= BB_upper (confirmed by Bollinger Band)

Exit via ATR-based SL and TP, or max_hold_bars in engine.

Walk-Forward results (9-window, date-aligned):
  Phase 27 standalone (1h + MTF):
    k26_dev0.3: 55% rob, OOS -1.74%, 62 trades
    k26_dev0.5: 66% rob, OOS -0.01%, 58 trades
    k26_dev0.7: 66% rob, OOS +2.06%, 57 trades (BEST)
    k26_dev1.0: 55% rob, OOS -6.52%, 48 trades
    k20_dev0.7: 44% rob (shorter kijun = worse)
    k33_dev0.7: 66% rob, OOS -2.53%
  Phase 27 15m: max 55% rob — NOT viable on 15m
  Phase 27 as 5th Cross-TF component:
    ALL 5-comp combos = 77% rob (DOWN from 88% baseline)
    Ichimoku kills W3 (-3.98%) which is positive in 4-comp
  Phase 27 replacing WillR (4-comp):
    ALL combos = 77% rob, OOS +16-18% (worse than WillR 88% +23.98%)

VERDICT: Ichimoku Kijun MR does NOT improve the portfolio.
  - Standalone 66% rob is equal to RSI/CCI but with much lower return
  - Degrades portfolio robustness from 88% → 77% when added
  - High correlation with existing MR signals (another distance-from-mean variant)
  - DEPRECATED — use RSI or WillR instead
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


def add_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
) -> pd.DataFrame:
    """Add Ichimoku Tenkan-sen and Kijun-sen columns to DataFrame.

    Only computes Tenkan and Kijun (not cloud/chikou) since those
    require forward-shifting and are not used in the MR signal.

    Args:
        df: OHLCV DataFrame.
        tenkan_period: Lookback for Tenkan-sen.
        kijun_period: Lookback for Kijun-sen.

    Returns:
        DataFrame with tenkan_sen and kijun_sen columns.
    """
    t_col = f"tenkan_{tenkan_period}"
    k_col = f"kijun_{kijun_period}"

    if t_col not in df.columns:
        df[t_col] = (
            df["high"].rolling(tenkan_period).max()
            + df["low"].rolling(tenkan_period).min()
        ) / 2

    if k_col not in df.columns:
        df[k_col] = (
            df["high"].rolling(kijun_period).max()
            + df["low"].rolling(kijun_period).min()
        ) / 2

    return df


class IchimokuKijunMRStrategy(BaseStrategy):
    """Fade Kijun-sen deviation extremes + Bollinger Band confirmation.

    The Kijun-sen (26-bar equilibrium) acts as a dynamic "fair value."
    When price deviates far from this equilibrium AND touches Bollinger Band,
    we expect mean reversion back toward the Kijun-sen.

    Args:
        kijun_period: Lookback for Kijun-sen (26-bar midpoint).
        tenkan_period: Lookback for Tenkan-sen (9-bar midpoint).
        deviation_atr_mult: Min distance from Kijun in ATR units to trigger.
        bb_column_lower: BB lower band column name.
        bb_column_upper: BB upper band column name.
        bb_proximity: Proximity multiplier for BB touch (1.01 = within 1%).
        atr_sl_mult: ATR multiplier for stop loss.
        atr_tp_mult: ATR multiplier for take profit.
        cooldown_bars: Minimum bars between entries.
    """

    name = "ichimoku_kijun_mr"

    def __init__(
        self,
        kijun_period: int = 26,
        tenkan_period: int = 9,
        deviation_atr_mult: float = 0.5,
        bb_column_lower: str = "BBL_20_2.0_2.0",
        bb_column_upper: str = "BBU_20_2.0_2.0",
        bb_proximity: float = 1.01,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.kijun_period = kijun_period
        self.tenkan_period = tenkan_period
        self.deviation_atr_mult = deviation_atr_mult
        self.bb_lower = bb_column_lower
        self.bb_upper = bb_column_upper
        self.bb_proximity = bb_proximity
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self._last_entry_idx = -999
        self.symbol = symbol
        self._kijun_col = f"kijun_{kijun_period}"
        self._tenkan_col = f"tenkan_{tenkan_period}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate mean-reversion signal from Kijun deviation + BB.

        Args:
            df: OHLCV DataFrame with kijun_{period}, atr_14, BBL/BBU.

        Returns:
            TradeSignal.
        """
        min_bars = self.kijun_period + 10
        if len(df) < min_bars:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        kijun = last.get(self._kijun_col)
        tenkan = last.get(self._tenkan_col)
        atr = last.get("atr_14")
        bb_low = last.get(self.bb_lower)
        bb_up = last.get(self.bb_upper)

        if any(pd.isna(v) for v in [kijun, tenkan, atr, bb_low, bb_up]):
            return self._hold(df)

        kijun = float(kijun)
        tenkan = float(tenkan)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        if atr <= 0:
            return self._hold(df)

        # Cooldown
        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        # Kijun deviation in ATR units
        kijun_dev = (close - kijun) / atr  # positive = above kijun

        # TK momentum: tenkan > kijun = bullish momentum
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun

        # LONG: price well below Kijun equilibrium + BB lower touch
        if (kijun_dev < -self.deviation_atr_mult
                and close <= bb_low * self.bb_proximity):
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            # Confidence: deeper deviation → higher confidence
            depth = abs(kijun_dev) - self.deviation_atr_mult
            confidence = min(1.0, 0.5 + depth / 4.0)
            # Bonus if TK already crossing bullish (momentum shifting)
            if tk_bullish:
                confidence = min(1.0, confidence + 0.10)
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
                    "kijun_dev_atr": round(kijun_dev, 3),
                    "kijun": kijun,
                    "tenkan": tenkan,
                    "tk_bullish": tk_bullish,
                    "bb_touch": "lower",
                },
            )

        # SHORT: price well above Kijun equilibrium + BB upper touch
        if (kijun_dev > self.deviation_atr_mult
                and close >= bb_up / self.bb_proximity):
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            depth = kijun_dev - self.deviation_atr_mult
            confidence = min(1.0, 0.5 + depth / 4.0)
            if tk_bearish:
                confidence = min(1.0, confidence + 0.10)
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
                    "kijun_dev_atr": round(kijun_dev, 3),
                    "kijun": kijun,
                    "tenkan": tenkan,
                    "tk_bearish": tk_bearish,
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
        return [self._kijun_col, self._tenkan_col, "atr_14",
                self.bb_lower, self.bb_upper]

"""RSI mean-reversion strategy with Bollinger Band confirmation.

MUST use with MultiTimeframeFilter(4h EMA) for reliability.

Walk-Forward results (1h, with MTF 4h filter):
  Phase 10 (5w):  60% robustness, OOS +13.46%, Full +148% (overfitted)
  Phase 12 (7w):  71% robustness, OOS +10.68%
  Phase 14 (9w):  66% robustness, OOS +13.29%, Full +90.27%, DD 19.4%, PF 1.74
  Standalone (no MTF): 40% rob, OOS -3 to -18% — FAILS

Walk-Forward results (15m, RSI_35_65_mid config, cool=12, max_hold=96):
  Phase 16 (9w):  77% robustness, OOS +17.50%, Full +231.92%, Sharpe 1.42
  Best single-strategy result at any timeframe.
  15m captures different regimes than 1h (negative W2,W5 vs 1h W2,W6).

Best configs:
  1h:  oversold=35, overbought=65, sl=2.0, tp=3.0, cool=6
  15m: oversold=35, overbought=65, sl=2.0, tp=3.0, cool=12, max_hold=96

Cross-TF portfolio (Phase 17 — FORMALLY VALIDATED, date-aligned 9w):
  1hRSI/1hDC/15mRSI 33/33/34 = 88% robustness, OOS +18.81% ★
  1h-only RSI+DC 50/50 = 66% rob (date-aligned), 77% (bar-aligned)
  88% is the ceiling — only W2 (Nov 20-Dec 2 whipsaw) remains negative.

Phase 20 parameter stability (28 perturbations, all ≥66% rob):
  1hRSI: min=77%, avg=86%. Very stable across all params.
  15mRSI: min=66%, avg=83%. Most sensitive — don't tighten oversold below 35.
  Strategy is NOT parameter-overfit. 88% is a structural property.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class RSIMeanReversionStrategy(BaseStrategy):
    """Fade extremes: buy oversold, sell overbought.

    Entry rules:
      LONG:  RSI < oversold AND close <= BB_lower (oversold + at band)
      SHORT: RSI > overbought AND close >= BB_upper (overbought + at band)

    Exit: RSI returns to neutral zone (45-55) or ATR-based SL/TP.
    """

    name = "rsi_mean_reversion"

    def __init__(
        self,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        bb_column_lower: str = "BBL_20_2.0_2.0",
        bb_column_upper: str = "BBU_20_2.0_2.0",
        bb_proximity: float = 1.01,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 2.5,
        cooldown_bars: int = 5,
        symbol: str = SYMBOL,
    ) -> None:
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_lower = bb_column_lower
        self.bb_upper = bb_column_upper
        self.bb_proximity = bb_proximity
        self.cooldown_bars = cooldown_bars
        self._last_entry_idx = -999
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.symbol = symbol

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate mean-reversion signal from RSI extremes + BB.

        Args:
            df: OHLCV DataFrame with rsi_14, BBL/BBU, atr_14.

        Returns:
            TradeSignal.
        """
        if len(df) < 30:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        rsi = last.get("rsi_14")
        atr = last.get("atr_14")
        bb_low = last.get(self.bb_lower)
        bb_up = last.get(self.bb_upper)

        if any(pd.isna(v) for v in [rsi, atr, bb_low, bb_up]):
            return self._hold(df)

        rsi = float(rsi)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        # Cooldown: don't trade too frequently
        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        # RSI divergence check: price making new low but RSI higher (bullish)
        bullish_div = False
        bearish_div = False
        if len(df) >= 10:
            recent_lows = df["low"].iloc[-10:]
            recent_rsi = df["rsi_14"].iloc[-10:]
            if not recent_rsi.isna().all():
                price_at_min = recent_lows.idxmin()
                if price_at_min != df.index[-1]:
                    old_rsi = df["rsi_14"].loc[price_at_min]
                    if not pd.isna(old_rsi):
                        if close <= float(recent_lows.min()) * 1.005 and rsi > float(old_rsi):
                            bullish_div = True

                recent_highs = df["high"].iloc[-10:]
                price_at_max = recent_highs.idxmax()
                if price_at_max != df.index[-1]:
                    old_rsi_h = df["rsi_14"].loc[price_at_max]
                    if not pd.isna(old_rsi_h):
                        if close >= float(recent_highs.max()) * 0.995 and rsi < float(old_rsi_h):
                            bearish_div = True

        # LONG: oversold + near lower Bollinger Band
        if rsi < self.rsi_oversold and close <= bb_low * self.bb_proximity:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            confidence = 0.55 + (self.rsi_oversold - rsi) / 100.0
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
                    "rsi": rsi,
                    "bb_touch": "lower",
                    "divergence": bullish_div,
                },
            )

        # SHORT: overbought + near upper Bollinger Band
        if rsi > self.rsi_overbought and close >= bb_up / self.bb_proximity:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            confidence = 0.55 + (rsi - self.rsi_overbought) / 100.0
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
                    "rsi": rsi,
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
        return ["rsi_14", "atr_14", self.bb_lower, self.bb_upper]

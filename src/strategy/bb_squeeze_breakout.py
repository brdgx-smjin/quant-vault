"""Bollinger Band Squeeze Breakout strategy with volume and trend confirmation.

The canonical BBSqueeze strategy. Best single strategy across all phases.

Walk-Forward results (1h, with MTF 4h filter, require_trend=False):
  Phase 4-6 (5w): 60% robustness, OOS +1.14%, Full +15.54%, DD 16%, PF 1.42
  Phase 6 (7w):   57% robustness, OOS +9.58%, Full +15.54%
  Phase 9 (5w):   60% robustness, OOS +5.71%, Full +25.97%, DD 10.3%, PF 2.31

Optimal params: squeeze_pctile=25, vol_mult=1.1, atr_sl_mult=3.0,
    rr_ratio=2.0, cooldown_bars=6, require_trend=False.
Always use with MultiTimeframeFilter(4h EMA) for best robustness.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class BBSqueezeBreakoutStrategy(BaseStrategy):
    """Enter on breakout from a Bollinger Band squeeze with confirmation.

    Detects periods of low volatility (narrow BB width) and enters when
    price breaks out with volume and momentum confirmation. Squeeze
    breakouts tend to produce strong directional moves.

    Entry (LONG):
      - BB width in bottom squeeze_pctile of lookback period (squeeze)
      - Close > upper BB (breakout)
      - Volume > vol_mult * 20-bar average volume
      - RSI > 50 (momentum confirms direction)
      - EMA_20 > EMA_50 (trend alignment, if require_trend=True)

    Entry (SHORT):
      - BB width in bottom squeeze_pctile (squeeze)
      - Close < lower BB (breakout)
      - Volume > vol_mult * 20-bar average
      - RSI < 50
      - EMA_20 < EMA_50 (if require_trend=True)

    SL: Middle BB (SMA_20) or ATR-based, whichever is more protective.
    TP: Entry +/- rr_ratio * risk distance.
    """

    name = "bb_squeeze_breakout"

    def __init__(
        self,
        squeeze_lookback: int = 100,
        squeeze_pctile: float = 25.0,
        vol_mult: float = 1.3,
        atr_sl_mult: float = 2.0,
        rr_ratio: float = 2.5,
        require_trend: bool = True,
        cooldown_bars: int = 8,
        symbol: str = SYMBOL,
    ) -> None:
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_pctile = squeeze_pctile
        self.vol_mult = vol_mult
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.require_trend = require_trend
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate breakout signal from BB squeeze.

        Args:
            df: OHLCV DataFrame with rsi_14, atr_14, ema_20, ema_50,
                BBL_20_2.0_2.0, BBU_20_2.0_2.0, BBM_20_2.0_2.0,
                BBB_20_2.0_2.0 (BB width).

        Returns:
            TradeSignal.
        """
        min_bars = max(self.squeeze_lookback + 20, 60)
        if len(df) < min_bars:
            return self._hold(df)

        # Cooldown
        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        # Read indicators
        rsi = last.get("rsi_14")
        atr = last.get("atr_14")
        ema_20 = last.get("ema_20")
        ema_50 = last.get("ema_50")
        bb_upper = last.get("BBU_20_2.0_2.0")
        bb_lower = last.get("BBL_20_2.0_2.0")
        bb_mid = last.get("BBM_20_2.0_2.0")
        bb_width = last.get("BBB_20_2.0_2.0")
        volume = last.get("volume")

        vals = [rsi, atr, ema_20, ema_50, bb_upper, bb_lower, bb_mid, bb_width, volume]
        if any(pd.isna(v) for v in vals):
            return self._hold(df)

        rsi = float(rsi)
        atr = float(atr)
        ema_20 = float(ema_20)
        ema_50 = float(ema_50)
        bb_upper = float(bb_upper)
        bb_lower = float(bb_lower)
        bb_mid = float(bb_mid)
        bb_width = float(bb_width)
        volume = float(volume)

        # --- Squeeze detection ---
        # BB width (BBB) percentile over lookback
        bbw_col = "BBB_20_2.0_2.0"
        if bbw_col not in df.columns:
            return self._hold(df)

        recent_bbw = df[bbw_col].iloc[-self.squeeze_lookback:]
        recent_bbw = recent_bbw.dropna()
        if len(recent_bbw) < 20:
            return self._hold(df)

        threshold = recent_bbw.quantile(self.squeeze_pctile / 100.0)
        in_squeeze = bb_width <= threshold

        if not in_squeeze:
            return self._hold(df)

        # --- Volume confirmation ---
        vol_col = "volume"
        recent_vol = df[vol_col].iloc[-20:]
        avg_vol = float(recent_vol.mean())
        if avg_vol <= 0 or volume < avg_vol * self.vol_mult:
            return self._hold(df)

        # --- Trend filter ---
        if self.require_trend:
            uptrend = ema_20 > ema_50
            downtrend = ema_20 < ema_50
        else:
            uptrend = downtrend = True

        # --- LONG breakout ---
        if close > bb_upper and rsi > 50.0 and uptrend:
            risk = max(close - bb_mid, atr * self.atr_sl_mult)
            sl = close - risk
            tp = close + risk * self.rr_ratio
            confidence = min(1.0, 0.55 + (volume / avg_vol - 1.0) * 0.1)

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
                    "bb_width_pctile": float(
                        (recent_bbw < bb_width).sum() / len(recent_bbw) * 100
                    ),
                    "volume_ratio": volume / avg_vol,
                    "rsi": rsi,
                    "trend": "up",
                },
            )

        # --- SHORT breakout ---
        if close < bb_lower and rsi < 50.0 and downtrend:
            risk = max(bb_mid - close, atr * self.atr_sl_mult)
            sl = close + risk
            tp = close - risk * self.rr_ratio
            confidence = min(1.0, 0.55 + (volume / avg_vol - 1.0) * 0.1)

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
                    "bb_width_pctile": float(
                        (recent_bbw < bb_width).sum() / len(recent_bbw) * 100
                    ),
                    "volume_ratio": volume / avg_vol,
                    "rsi": rsi,
                    "trend": "down",
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
        return [
            "rsi_14",
            "atr_14",
            "ema_20",
            "ema_50",
            "BBL_20_2.0_2.0",
            "BBU_20_2.0_2.0",
            "BBM_20_2.0_2.0",
            "BBB_20_2.0_2.0",
        ]

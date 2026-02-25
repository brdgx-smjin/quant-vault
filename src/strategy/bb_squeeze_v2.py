"""BB Squeeze Breakout v2 — adds ADX filter and trailing-stop-aware TP.

Walk-Forward results (1h, with MTF 4h filter, require_trend=False):
  Baseline (ADX=0): 60% robustness, OOS +5.71% (Phase 9, 5w)
                     42-57% robustness (7w, varies by data window)
  ADX>15: 60% robustness but OOS -1.09% (Phase 5)
  ADX>20: 40% robustness (Phase 5)
  ADX>25-30: 40% robustness (Phase 5)
  Trailing stop (all variants): 40% robustness (Phase 5)

Conclusion: ADX filter does not improve OOS. Use with ADX=0 (disabled).
Optimal params: squeeze_pctile=25, vol_mult=1.1, atr_sl_mult=3.0,
    rr_ratio=2.0, cooldown_bars=6, require_trend=False + MTF(4h EMA).
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class BBSqueezeV2Strategy(BaseStrategy):
    """BBSqueeze Breakout with ADX trend strength filter.

    Improvements over v1:
      - ADX filter: only enter when ADX > threshold (trending market).
      - Optional TP disable: when using engine trailing stop, set TP=None
        so the trailing stop governs the exit instead of a fixed target.
      - Same squeeze/volume/EMA logic as v1.

    Entry (LONG):
      - BB width in bottom squeeze_pctile (squeeze)
      - Close > upper BB (breakout)
      - Volume > vol_mult * 20-bar average
      - RSI > 50
      - ADX > adx_threshold (if enabled)
      - EMA_20 > EMA_50 (if require_trend=True)

    Entry (SHORT): mirror conditions.
    """

    name = "bb_squeeze_v2"

    def __init__(
        self,
        squeeze_lookback: int = 100,
        squeeze_pctile: float = 25.0,
        vol_mult: float = 1.1,
        atr_sl_mult: float = 3.0,
        rr_ratio: float = 2.0,
        require_trend: bool = True,
        cooldown_bars: int = 6,
        adx_threshold: float = 0.0,  # 0 = disabled; e.g. 20 or 25
        disable_tp: bool = False,     # True when using engine trailing stop
        symbol: str = SYMBOL,
    ) -> None:
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_pctile = squeeze_pctile
        self.vol_mult = vol_mult
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.require_trend = require_trend
        self.cooldown_bars = cooldown_bars
        self.adx_threshold = adx_threshold
        self.disable_tp = disable_tp
        self.symbol = symbol
        self._last_entry_idx = -999

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate breakout signal from BB squeeze with ADX filter."""
        min_bars = max(self.squeeze_lookback + 20, 60)
        if len(df) < min_bars:
            return self._hold(df)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

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

        # --- ADX filter ---
        if self.adx_threshold > 0:
            adx_val = last.get("ADX_14")
            if pd.isna(adx_val):
                return self._hold(df)
            if float(adx_val) < self.adx_threshold:
                return self._hold(df)

        # --- Squeeze detection ---
        bbw_col = "BBB_20_2.0_2.0"
        if bbw_col not in df.columns:
            return self._hold(df)

        recent_bbw = df[bbw_col].iloc[-self.squeeze_lookback:]
        recent_bbw = recent_bbw.dropna()
        if len(recent_bbw) < 20:
            return self._hold(df)

        threshold = recent_bbw.quantile(self.squeeze_pctile / 100.0)
        if bb_width > threshold:
            return self._hold(df)

        # --- Volume confirmation ---
        recent_vol = df["volume"].iloc[-20:]
        avg_vol = float(recent_vol.mean())
        if avg_vol <= 0 or volume < avg_vol * self.vol_mult:
            return self._hold(df)

        # --- Trend filter ---
        if self.require_trend:
            uptrend = ema_20 > ema_50
            downtrend = ema_20 < ema_50
        else:
            uptrend = downtrend = True

        # --- Compute SL/TP ---
        def _make_signal(side: Signal, risk: float) -> TradeSignal:
            if side == Signal.LONG:
                sl = close - risk
                tp = close + risk * self.rr_ratio if not self.disable_tp else None
            else:
                sl = close + risk
                tp = close - risk * self.rr_ratio if not self.disable_tp else None

            confidence = min(1.0, 0.55 + (volume / avg_vol - 1.0) * 0.1)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=side,
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
                    "trend": "up" if side == Signal.LONG else "down",
                },
            )

        # --- LONG breakout ---
        if close > bb_upper and rsi > 50.0 and uptrend:
            risk = max(close - bb_mid, atr * self.atr_sl_mult)
            return _make_signal(Signal.LONG, risk)

        # --- SHORT breakout ---
        if close < bb_lower and rsi < 50.0 and downtrend:
            risk = max(bb_mid - close, atr * self.atr_sl_mult)
            return _make_signal(Signal.SHORT, risk)

        return self._hold(df)

    def _hold(self, df: pd.DataFrame) -> TradeSignal:
        return TradeSignal(
            signal=Signal.HOLD,
            symbol=self.symbol,
            price=float(df["close"].iloc[-1]),
            timestamp=df.index[-1],
        )

    def get_required_indicators(self) -> list[str]:
        indicators = [
            "rsi_14", "atr_14", "ema_20", "ema_50",
            "BBL_20_2.0_2.0", "BBU_20_2.0_2.0", "BBM_20_2.0_2.0", "BBB_20_2.0_2.0",
        ]
        if self.adx_threshold > 0:
            indicators.append("ADX_14")
        return indicators

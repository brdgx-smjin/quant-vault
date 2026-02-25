"""BB Squeeze Breakout v3 — candle body filter + session time filter.

DEPRECATED: All v3 filters degraded OOS performance vs baseline v1/v2.
Phase 6 WF results (5-window, 1h):
  - Body>40%: OOS -0.70%, Robustness 40% (worse than baseline +1.14%, 60%)
  - Body>50%: OOS -3.10%, Robustness 20%
  - Body>60%: OOS +1.27%, Robustness 20%
  - NoAsianLate: OOS +1.01%, Robustness 60% (same as baseline, not better)
  - USEuroOnly: OOS +0.41%, Robustness 60%
  - Combined Body+Session+BE: all worse than baseline

Use BBSqueezeBreakoutStrategy (v1) or BBSqueezeV2Strategy instead.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class BBSqueezeV3Strategy(BaseStrategy):
    """BBSqueeze Breakout with entry quality filters.

    Improvements over v2:
      - Candle body ratio: only enter when breakout candle body > min_body_ratio
        of total range (filters doji/spinning top fake breakouts).
      - Session filter: optionally block entries during low-liquidity hours.
      - All other logic identical to v1/v2 BBSqueeze.

    Entry (LONG):
      - BB width in bottom squeeze_pctile (squeeze)
      - Close > upper BB (breakout)
      - Volume > vol_mult * 20-bar average
      - RSI > 50
      - Candle body / range > min_body_ratio (strong breakout candle)
      - Hour NOT in blocked_hours (if set)
      - EMA_20 > EMA_50 (if require_trend=True)

    Entry (SHORT): mirror conditions.
    """

    name = "bb_squeeze_v3"

    def __init__(
        self,
        squeeze_lookback: int = 100,
        squeeze_pctile: float = 25.0,
        vol_mult: float = 1.1,
        atr_sl_mult: float = 3.0,
        rr_ratio: float = 2.0,
        require_trend: bool = True,
        cooldown_bars: int = 6,
        min_body_ratio: float = 0.0,  # 0 = disabled; e.g. 0.5 = 50%
        blocked_hours: tuple[int, ...] = (),  # UTC hours to block entries
        symbol: str = SYMBOL,
    ) -> None:
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_pctile = squeeze_pctile
        self.vol_mult = vol_mult
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.require_trend = require_trend
        self.cooldown_bars = cooldown_bars
        self.min_body_ratio = min_body_ratio
        self.blocked_hours = blocked_hours
        self.symbol = symbol
        self._last_entry_idx = -999

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate breakout signal with entry quality filters."""
        min_bars = max(self.squeeze_lookback + 20, 60)
        if len(df) < min_bars:
            return self._hold(df)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        high = float(last["high"])
        low = float(last["low"])
        open_ = float(last["open"])
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

        # --- Session filter ---
        if self.blocked_hours and hasattr(ts, "hour"):
            if ts.hour in self.blocked_hours:
                return self._hold(df)

        # --- Candle body ratio filter ---
        if self.min_body_ratio > 0:
            candle_range = high - low
            if candle_range <= 0:
                return self._hold(df)
            body = abs(close - open_)
            body_ratio = body / candle_range
            if body_ratio < self.min_body_ratio:
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
                tp = close + risk * self.rr_ratio
            else:
                sl = close + risk
                tp = close - risk * self.rr_ratio

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
                    "body_ratio": abs(close - open_) / (high - low) if high > low else 0,
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
        return [
            "rsi_14", "atr_14", "ema_20", "ema_50",
            "BBL_20_2.0_2.0", "BBU_20_2.0_2.0", "BBM_20_2.0_2.0", "BBB_20_2.0_2.0",
        ]

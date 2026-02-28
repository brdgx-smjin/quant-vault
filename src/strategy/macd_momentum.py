"""DEPRECATED — MACD Momentum strategy.

FAILED — Phase 10 results (1h):
  All variants deeply negative (OOS -3% to -49%, robustness 0-40%).
  Best: MACD_Tight(8_21)+MTF — OOS -3.41%, 40% robustness (still negative).
  Full period returns -55% to -90%.
  MACD momentum does NOT work on BTC/USDT 1h. Do not use in production.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class MACDMomentumStrategy(BaseStrategy):
    """Trade MACD crossovers that signal early momentum shifts.

    Enters when MACD line crosses its signal line in the opposite zone
    (bullish crossover while MACD negative, bearish while positive),
    confirmed by volume and RSI guard.

    Args:
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal line smoothing period.
        require_zero_cross: If True, MACD must be in opposite zone for entry.
        rsi_guard: Max RSI for LONG (min 100-rsi_guard for SHORT).
        atr_sl_mult: ATR multiplier for stop loss.
        rr_ratio: Reward-to-risk ratio for take profit.
        vol_mult: Minimum volume vs 20-bar average.
        cooldown_bars: Minimum bars between entries.
    """

    name = "macd_momentum"

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        require_zero_cross: bool = True,
        rsi_guard: float = 70.0,
        atr_sl_mult: float = 2.0,
        rr_ratio: float = 2.0,
        vol_mult: float = 0.8,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.require_zero_cross = require_zero_cross
        self.rsi_guard = rsi_guard
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.vol_mult = vol_mult
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999

        # pandas_ta column names: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        self._col_macd = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
        self._col_signal = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"
        self._col_hist = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate momentum signal from MACD crossover.

        Args:
            df: OHLCV DataFrame with MACD, MACD_signal, rsi_14, atr_14.

        Returns:
            TradeSignal.
        """
        min_bars = max(self.macd_slow + self.macd_signal + 5, 60)
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

        # Read indicators (pandas_ta naming: MACD_f_s_sig, MACDs_f_s_sig, MACDh_f_s_sig)
        macd_val = last.get(self._col_macd)
        macd_sig = last.get(self._col_signal)
        macd_hist = last.get(self._col_hist)
        prev_macd = prev.get(self._col_macd)
        prev_sig = prev.get(self._col_signal)
        rsi = last.get("rsi_14")
        atr = last.get("atr_14")
        volume = last.get("volume")

        vals = [macd_val, macd_sig, macd_hist, prev_macd, prev_sig, rsi, atr, volume]
        if any(pd.isna(v) for v in vals):
            return self._hold(df)

        macd_val = float(macd_val)
        macd_sig = float(macd_sig)
        prev_macd = float(prev_macd)
        prev_sig = float(prev_sig)
        rsi = float(rsi)
        atr = float(atr)
        volume = float(volume)

        if atr <= 0:
            return self._hold(df)

        # Volume confirmation
        vol_avg = float(df["volume"].iloc[-20:].mean())
        if vol_avg <= 0 or volume < vol_avg * self.vol_mult:
            return self._hold(df)

        # Detect MACD crossover
        bullish_cross = prev_macd <= prev_sig and macd_val > macd_sig
        bearish_cross = prev_macd >= prev_sig and macd_val < macd_sig

        # Zero-zone condition: enter early (MACD still in opposite territory)
        if self.require_zero_cross:
            long_zone = macd_val < 0  # Catching bullish shift from negative zone
            short_zone = macd_val > 0  # Catching bearish shift from positive zone
        else:
            long_zone = True
            short_zone = True

        # LONG: bullish crossover in negative zone, RSI not overbought
        if bullish_cross and long_zone and rsi < self.rsi_guard:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_sl_mult * self.rr_ratio

            # Confidence based on histogram magnitude
            hist_strength = abs(float(last[self._col_hist]))
            recent_hist = df[self._col_hist].iloc[-20:].abs()
            avg_hist = float(recent_hist.mean()) if len(recent_hist) > 0 else 1.0
            confidence = min(1.0, 0.5 + (hist_strength / max(avg_hist, 1e-8)) * 0.15)

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
                    "macd": macd_val,
                    "macd_signal": macd_sig,
                    "macd_hist": float(last[self._col_hist]),
                    "rsi": rsi,
                    "volume_ratio": volume / vol_avg,
                },
            )

        # SHORT: bearish crossover in positive zone, RSI not oversold
        if bearish_cross and short_zone and rsi > (100 - self.rsi_guard):
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_sl_mult * self.rr_ratio

            hist_strength = abs(float(last[self._col_hist]))
            recent_hist = df[self._col_hist].iloc[-20:].abs()
            avg_hist = float(recent_hist.mean()) if len(recent_hist) > 0 else 1.0
            confidence = min(1.0, 0.5 + (hist_strength / max(avg_hist, 1e-8)) * 0.15)

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
                    "macd": macd_val,
                    "macd_signal": macd_sig,
                    "macd_hist": float(last[self._col_hist]),
                    "rsi": rsi,
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
        return [self._col_macd, self._col_signal, self._col_hist, "rsi_14", "atr_14"]

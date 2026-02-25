"""Fibonacci retracement strategy with EMA trend filter and ATR stops.

Walk-Forward results (4h, standalone):
  Phase 10 (5w):  40% robustness, OOS +0.89%, Full +15.56%, only 7 OOS trades
  Phase 10 (7w):  14% robustness, OOS +0.86%, only 3 OOS trades

  BB+Fib 50/50 portfolio:
    5w: 80% rob, OOS +3.29% — decent robustness but low OOS
    7w: 42% rob, OOS +3.37% — FAILS below 60% threshold

Warning: Fib generates very few trades on 4h BTC — low statistical significance.
Not recommended for production. BB+RSI_MTF or BB+VWAP_MTF portfolios
are strictly superior at all window counts.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.indicators.fibonacci import FibonacciCalculator
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class FibonacciRetracementStrategy(BaseStrategy):
    """Enter at Fibonacci retracement levels WITH trend confirmation.

    Only takes longs when EMA_20 > EMA_50 (uptrend),
    only takes shorts when EMA_20 < EMA_50 (downtrend).
    Uses ATR-based stop-loss and Fibonacci extension for take-profit.
    """

    name = "fibonacci_retracement"

    def __init__(
        self,
        entry_levels: tuple[float, ...] = (0.5, 0.618),
        tp_extension: float = 1.618,
        tolerance_pct: float = 0.05,
        lookback: int = 50,
        atr_sl_mult: float = 1.5,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        require_trend: bool = True,
        symbol: str = SYMBOL,
    ) -> None:
        self.entry_levels = entry_levels
        self.tp_extension = tp_extension
        self.tolerance_pct = tolerance_pct
        self.lookback = lookback
        self.atr_sl_mult = atr_sl_mult
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.require_trend = require_trend
        self.symbol = symbol
        self.fib = FibonacciCalculator()

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate signal at Fibonacci levels in trend direction only.

        Args:
            df: OHLCV DataFrame with ema_20, ema_50, rsi_14, atr_14.

        Returns:
            TradeSignal.
        """
        if len(df) < self.lookback + 10:
            return self._hold(df)

        fib_result = self.fib.analyze(df, self.lookback)
        if fib_result is None:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]
        swing_range = abs(fib_result.swing_high - fib_result.swing_low)
        tolerance = swing_range * self.tolerance_pct

        # Read indicators
        ema_20 = last.get("ema_20")
        ema_50 = last.get("ema_50")
        rsi = last.get("rsi_14")
        atr = last.get("atr_14")

        has_indicators = not (
            pd.isna(ema_20) or pd.isna(ema_50) or pd.isna(rsi) or pd.isna(atr)
        )
        if not has_indicators:
            return self._hold(df)

        ema_20 = float(ema_20)
        ema_50 = float(ema_50)
        rsi = float(rsi)
        atr = float(atr)

        # Trend filter
        if self.require_trend:
            uptrend = ema_20 > ema_50
            downtrend = ema_20 < ema_50
        else:
            uptrend = downtrend = True

        # TP from Fibonacci extension
        tp_levels = [
            lv for lv in fib_result.levels
            if lv.kind == "extension" and abs(lv.ratio - self.tp_extension) < 0.01
        ]

        for target_ratio in self.entry_levels:
            entry_levels = [
                lv for lv in fib_result.levels
                if lv.kind == "retracement" and abs(lv.ratio - target_ratio) < 0.01
            ]
            if not entry_levels:
                continue

            entry_price = entry_levels[0].price
            if abs(close - entry_price) > tolerance:
                continue

            # Confidence: 0.618 is the golden ratio — higher confidence
            base_conf = 0.65 if target_ratio == 0.618 else 0.55

            # LONG: uptrend + price at fib support + RSI not overbought
            if uptrend and fib_result.direction == "up" and rsi < self.rsi_overbought:
                sl = close - atr * self.atr_sl_mult
                tp = tp_levels[0].price if tp_levels else close + atr * 3.0
                trend_strength = min(0.2, (ema_20 - ema_50) / ema_50 * 10)
                return TradeSignal(
                    signal=Signal.LONG,
                    symbol=self.symbol,
                    price=close,
                    timestamp=ts,
                    confidence=min(1.0, base_conf + trend_strength),
                    stop_loss=sl,
                    take_profit=tp,
                    metadata={
                        "strategy": self.name,
                        "fib_level": target_ratio,
                        "rsi": rsi,
                        "trend": "up",
                    },
                )

            # SHORT: downtrend + price at fib resistance + RSI not oversold
            if downtrend and fib_result.direction == "down" and rsi > self.rsi_oversold:
                sl = close + atr * self.atr_sl_mult
                tp = tp_levels[0].price if tp_levels else close - atr * 3.0
                trend_strength = min(0.2, (ema_50 - ema_20) / ema_50 * 10)
                return TradeSignal(
                    signal=Signal.SHORT,
                    symbol=self.symbol,
                    price=close,
                    timestamp=ts,
                    confidence=min(1.0, base_conf + trend_strength),
                    stop_loss=sl,
                    take_profit=tp,
                    metadata={
                        "strategy": self.name,
                        "fib_level": target_ratio,
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
        return ["ema_20", "ema_50", "rsi_14", "atr_14"]

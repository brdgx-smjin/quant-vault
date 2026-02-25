"""Cross-Timeframe Portfolio — combines 15m and 1h strategies.

Phase 16 validated (9w): 89% robustness (8/9 windows), OOS +19.61%.
Only W2 is negative across all timeframes.

Architecture:
    - TradingEngine streams 15m candles (base TF)
    - Engine resamples 15m → 1h, passes via set_htf_data()
    - This class resamples 1h → 4h for all strategies' MTF filters
    - 15m strategies: evaluated every 15m bar
    - 1h strategies: evaluated only at 1h boundaries (minute==45)

Validated config (Phase 16, 33/33/34):
    1h RSI (35/65, SL=2.0, TP=3.0, cool=6) + MTF(4h)     → 33%
    1h DC  (24, SL=2.0, RR=2.0, vol=0.8, cool=6) + MTF(4h) → 33%
    15m RSI (35/65, SL=2.0, TP=3.0, cool=12) + MTF(4h)    → 34%

Usage:
    from src.strategy.cross_tf_portfolio import CrossTimeframePortfolio
    from src.strategy.mtf_filter import MultiTimeframeFilter
    from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
    from src.strategy.donchian_trend import DonchianTrendStrategy

    rsi_1h = MultiTimeframeFilter(RSIMeanReversionStrategy(...))
    dc_1h = MultiTimeframeFilter(DonchianTrendStrategy(...))
    rsi_15m = MultiTimeframeFilter(RSIMeanReversionStrategy(...))

    portfolio = CrossTimeframePortfolio(
        strategies_15m=[(rsi_15m, 0.34)],
        strategies_1h=[(rsi_1h, 0.33), (dc_1h, 0.33)],
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.data.preprocessor import DataPreprocessor
from src.indicators.basic import BasicIndicators
from src.strategy.base import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger(__name__)


class CrossTimeframePortfolio(BaseStrategy):
    """Portfolio combining strategies from different timeframes.

    15m strategies are evaluated on every bar.
    1h strategies are evaluated only when a 1h candle completes
    (15m candle timestamp minute == 45).
    All strategies use 4h EMA trend filter (resampled from 1h data).
    """

    name = "cross_tf_portfolio"

    def __init__(
        self,
        strategies_15m: list[tuple[BaseStrategy, float]],
        strategies_1h: list[tuple[BaseStrategy, float]],
    ) -> None:
        """Initialize cross-timeframe portfolio.

        Args:
            strategies_15m: List of (strategy, weight) for 15m evaluation.
            strategies_1h: List of (strategy, weight) for 1h evaluation.
        """
        self.strategies_15m = strategies_15m
        self.strategies_1h = strategies_1h

        all_names = (
            [f"15m_{s.name}" for s, _ in strategies_15m]
            + [f"1h_{s.name}" for s, _ in strategies_1h]
        )
        self.name = "cross_tf_" + "+".join(all_names)

        self.df_1h: Optional[pd.DataFrame] = None
        self.df_4h: Optional[pd.DataFrame] = None

    def set_htf_data(self, df_htf: pd.DataFrame) -> None:
        """Receive 1h data from TradingEngine, resample to 4h for MTF filters.

        Args:
            df_htf: 1h OHLCV DataFrame with indicators (from engine resample).
        """
        self.df_1h = df_htf

        # Resample 1h → 4h for MTF filters
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in ohlcv_cols if c in df_htf.columns]
        df_4h = DataPreprocessor.resample(df_htf[available], "4h")
        if df_4h.index.duplicated().any():
            df_4h = df_4h[~df_4h.index.duplicated(keep="last")]
        self.df_4h = BasicIndicators.add_all(df_4h)

        # Set 4h data to all MTF-wrapped strategies
        for strategy, _ in self.strategies_15m + self.strategies_1h:
            if hasattr(strategy, "set_htf_data"):
                strategy.set_htf_data(self.df_4h)

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate signal from cross-timeframe strategies.

        Args:
            df: 15m OHLCV DataFrame with indicators.

        Returns:
            TradeSignal — best signal from active strategies.
        """
        candidates: list[tuple[TradeSignal, float]] = []

        # Always evaluate 15m strategies
        for strategy, weight in self.strategies_15m:
            sig = strategy.generate_signal(df)
            if sig.signal not in (Signal.HOLD,):
                sig.metadata["timeframe"] = "15m"
                candidates.append((sig, weight))

        # Evaluate 1h strategies only at 1h boundaries
        # 15m candle with minute==45 means the 1h candle just closed
        current_ts = df.index[-1]
        is_1h_boundary = current_ts.minute == 45

        if is_1h_boundary and self.df_1h is not None and len(self.df_1h) > 30:
            for strategy, weight in self.strategies_1h:
                sig = strategy.generate_signal(self.df_1h)
                if sig.signal not in (Signal.HOLD,):
                    sig.metadata["timeframe"] = "1h"
                    candidates.append((sig, weight))
            logger.info(
                "[CROSS-TF] 1h boundary — evaluated %d 1h strategies",
                len(self.strategies_1h),
            )

        if not candidates:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=df.attrs.get("symbol", ""),
                price=float(df["close"].iloc[-1]),
                timestamp=df.index[-1],
            )

        # Check for conflicting directions
        longs = [s for s, _ in candidates if s.signal == Signal.LONG]
        shorts = [s for s, _ in candidates if s.signal == Signal.SHORT]

        if longs and shorts:
            logger.info(
                "[CROSS-TF] Conflicting signals: %d LONG vs %d SHORT → HOLD",
                len(longs), len(shorts),
            )
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=candidates[0][0].symbol,
                price=float(df["close"].iloc[-1]),
                timestamp=df.index[-1],
                metadata={"blocked_by": "cross_tf_conflict"},
            )

        # Pick highest confidence signal
        best_sig, best_weight = max(candidates, key=lambda x: x[0].confidence)
        best_sig.metadata["portfolio_weight"] = best_weight
        best_sig.metadata["portfolio_name"] = self.name

        logger.info(
            "[CROSS-TF] %s signal from %s@%s (weight=%.0f%%, conf=%.3f)",
            best_sig.signal.value,
            best_sig.metadata.get("strategy", "?"),
            best_sig.metadata.get("timeframe", "?"),
            best_weight * 100,
            best_sig.confidence,
        )

        return best_sig

    def get_required_indicators(self) -> list[str]:
        indicators: set[str] = set()
        for s, _ in self.strategies_15m + self.strategies_1h:
            indicators.update(s.get_required_indicators())
        return list(indicators)

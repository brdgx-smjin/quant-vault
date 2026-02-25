"""ML Regime Classifier — meta-model for strategy selection.

Instead of predicting returns directly (which failed with XGBoost),
this classifies the market REGIME and selects the best strategy:
  - Trending regime: use BBSqueeze (breakout works in trends)
  - Ranging regime: use RSI_MR (mean reversion works in ranges)

Tested as improvement over RegimeSwitchStrategy(ADX=20) by using multiple
features (ADX, volatility, BB_width, trend strength) instead of just ADX.

Phase 10c results — DOES NOT IMPROVE over simple ADX threshold:
  MLRegime_NoVol+MTF (5w): 80% rob, OOS +6.30%, but Full -2.63% (overfitting)
  MLRegime_NoVol (7w): 57% rob — degrades vs Regime_ADX20's 71%
  More features = more noise = more overfitting.

Conclusion: Use RegimeSwitchStrategy(ADX=20) instead — simpler and more robust.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.settings import SYMBOL
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class MLRegimeStrategy(BaseStrategy):
    """Select strategy based on multi-feature regime classification.

    Uses ADX + ATR_norm + BB_width_percentile + EMA_trend_strength
    to classify market state, then delegates to the appropriate sub-strategy.

    Compared to RegimeSwitchStrategy (ADX-only threshold), this uses
    a richer feature set for regime detection.

    Args:
        trend_strategy: Strategy for trending markets (e.g., BBSqueeze).
        range_strategy: Strategy for ranging markets (e.g., RSI_MR).
        adx_trend_threshold: ADX above this = trending.
        adx_range_threshold: ADX below this = ranging. Between thresholds = neutral.
        volatility_filter: If True, block signals when ATR is extreme (>2x median).
        bb_squeeze_regime: If True, low BB_width → prefer range strategy.
        lookback: Lookback for computing regime features.
    """

    name = "ml_regime"

    def __init__(
        self,
        trend_strategy: BaseStrategy,
        range_strategy: BaseStrategy,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 18.0,
        volatility_filter: bool = True,
        vol_extreme_mult: float = 2.0,
        bb_squeeze_regime: bool = True,
        bb_squeeze_pctile: float = 20.0,
        lookback: int = 100,
        symbol: str = SYMBOL,
    ) -> None:
        self.trend_strategy = trend_strategy
        self.range_strategy = range_strategy
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.volatility_filter = volatility_filter
        self.vol_extreme_mult = vol_extreme_mult
        self.bb_squeeze_regime = bb_squeeze_regime
        self.bb_squeeze_pctile = bb_squeeze_pctile
        self.lookback = lookback
        self.symbol = symbol

    def _classify_regime(self, df: pd.DataFrame) -> str:
        """Classify current market regime using multiple features.

        Returns:
            One of: "trending", "ranging", "volatile", "neutral"
        """
        last = df.iloc[-1]

        adx = last.get("ADX_14")
        atr = last.get("atr_14")
        bb_width = last.get("BBB_20_2.0_2.0")
        ema_20 = last.get("ema_20")
        ema_50 = last.get("ema_50")

        if any(pd.isna(v) for v in [adx, atr, bb_width, ema_20, ema_50]):
            return "neutral"

        adx = float(adx)
        atr = float(atr)
        bb_width = float(bb_width)
        ema_20 = float(ema_20)
        ema_50 = float(ema_50)

        # Volatility check: extreme ATR = volatile regime
        if self.volatility_filter and "atr_14" in df.columns:
            atr_median = float(df["atr_14"].iloc[-self.lookback:].median())
            if atr_median > 0 and atr > atr_median * self.vol_extreme_mult:
                return "volatile"

        # BB squeeze: very low BB width = ranging (squeeze)
        if self.bb_squeeze_regime and "BBB_20_2.0_2.0" in df.columns:
            recent_bbw = df["BBB_20_2.0_2.0"].iloc[-self.lookback:].dropna()
            if len(recent_bbw) >= 20:
                threshold = recent_bbw.quantile(self.bb_squeeze_pctile / 100.0)
                if bb_width <= threshold:
                    # In a squeeze — could break out either way
                    # If ADX is also high, it's trending out of squeeze
                    if adx > self.adx_trend_threshold:
                        return "trending"
                    return "ranging"

        # ADX-based regime detection
        # EMA trend strength adds confidence
        ema_gap = abs(ema_20 - ema_50) / ema_50 * 100 if ema_50 > 0 else 0
        strong_trend = ema_gap > 1.0  # >1% gap between EMAs

        if adx > self.adx_trend_threshold:
            return "trending"
        elif adx < self.adx_range_threshold:
            return "ranging"
        elif strong_trend:
            # ADX between thresholds but EMAs diverged → leaning trending
            return "trending"
        else:
            return "neutral"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate signal by classifying regime and delegating.

        Args:
            df: OHLCV DataFrame with indicators.

        Returns:
            TradeSignal from the selected sub-strategy.
        """
        if len(df) < max(self.lookback, 60):
            return self._hold(df)

        regime = self._classify_regime(df)

        if regime == "trending":
            sig = self.trend_strategy.generate_signal(df)
        elif regime == "ranging":
            sig = self.range_strategy.generate_signal(df)
        elif regime == "volatile":
            # In extreme volatility, stay out
            return self._hold(df)
        else:
            # Neutral: use range strategy (mean reversion with MTF is safer)
            sig = self.range_strategy.generate_signal(df)

        # Enrich metadata with regime info
        if sig.signal != Signal.HOLD:
            sig.metadata["regime"] = regime
            sig.metadata["delegated_to"] = (
                "trend_strategy" if regime == "trending" else "range_strategy"
            )

        return sig

    def _hold(self, df: pd.DataFrame) -> TradeSignal:
        return TradeSignal(
            signal=Signal.HOLD,
            symbol=self.symbol,
            price=float(df["close"].iloc[-1]),
            timestamp=df.index[-1],
        )

    def get_required_indicators(self) -> list[str]:
        indicators = set()
        indicators.update(self.trend_strategy.get_required_indicators())
        indicators.update(self.range_strategy.get_required_indicators())
        indicators.add("ADX_14")
        indicators.add("BBB_20_2.0_2.0")
        return list(indicators)

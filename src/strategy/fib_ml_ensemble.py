"""DEPRECATED — Fibonacci + ML ensemble strategy: Fib generates signal, ML confirms.

Never walk-forward validated as ensemble. Component WF results:
  - Fibonacci (4h): 67% robustness but only 4 OOS trades — unreliable.
  - ML XGBoost filter: 0% robustness (Phase 4) — no predictive value.
  - Combined: not tested (both components failed independently).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config.settings import SYMBOL
from src.ml.features import build_features
from src.ml.models import SignalPredictor
from src.strategy.base import BaseStrategy, Signal, TradeSignal
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy

logger = logging.getLogger(__name__)


class FibMLEnsembleStrategy(BaseStrategy):
    """Fib generates primary signal, ML confirms direction.

    Rules:
        LONG  if Fib=LONG  AND ML probability > ml_long_threshold
        SHORT if Fib=SHORT AND ML probability < ml_short_threshold
        HOLD  otherwise

    If ML model fails (error, not loaded), falls through to raw Fib signal
    as graceful degradation.
    """

    name = "fib_ml_ensemble"

    def __init__(
        self,
        predictor: Optional[SignalPredictor] = None,
        ml_long_threshold: float = 0.55,
        ml_short_threshold: float = 0.45,
        entry_levels: tuple[float, ...] = (0.5, 0.618),
        tolerance_pct: float = 0.05,
        lookback: int = 50,
        require_trend: bool = True,
        future_bars: int = 12,
        symbol: str = SYMBOL,
    ) -> None:
        self.predictor = predictor
        self.ml_long_threshold = ml_long_threshold
        self.ml_short_threshold = ml_short_threshold
        self.future_bars = future_bars
        self.symbol = symbol

        self.fib_strategy = FibonacciRetracementStrategy(
            entry_levels=entry_levels,
            tolerance_pct=tolerance_pct,
            lookback=lookback,
            require_trend=require_trend,
            symbol=symbol,
        )

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate signal: Fib primary + ML confirmation.

        Args:
            df: OHLCV DataFrame with indicators.

        Returns:
            TradeSignal.
        """
        # Step 1: Get Fib signal
        fib_signal = self.fib_strategy.generate_signal(df)

        if fib_signal.signal == Signal.HOLD:
            return fib_signal

        # Step 2: Get ML probability for confirmation
        ml_prob = self._get_ml_prob(df)

        if ml_prob is None:
            # Graceful degradation: use raw Fib signal
            logger.debug("ML unavailable — using raw Fib signal")
            fib_signal.metadata["ml_status"] = "fallback"
            return fib_signal

        # Step 3: Confirm with ML
        if fib_signal.signal == Signal.LONG and ml_prob > self.ml_long_threshold:
            fib_signal.metadata["ml_prob"] = ml_prob
            fib_signal.metadata["ml_status"] = "confirmed"
            return fib_signal

        if fib_signal.signal == Signal.SHORT and ml_prob < self.ml_short_threshold:
            fib_signal.metadata["ml_prob"] = ml_prob
            fib_signal.metadata["ml_status"] = "confirmed"
            return fib_signal

        # ML disagrees — no trade
        logger.debug("ML rejected Fib %s signal (prob=%.3f)",
                     fib_signal.signal.value, ml_prob)
        return TradeSignal(
            signal=Signal.HOLD,
            symbol=self.symbol,
            price=fib_signal.price,
            timestamp=fib_signal.timestamp,
            metadata={"ml_prob": ml_prob, "ml_status": "rejected",
                       "fib_signal": fib_signal.signal.value},
        )

    def _get_ml_prob(self, df: pd.DataFrame) -> Optional[float]:
        """Get ML prediction probability, with error handling.

        Returns:
            Probability of profitable long, or None if ML unavailable.
        """
        if self.predictor is None or self.predictor.model is None:
            return None

        try:
            feat = build_features(df, future_bars=self.future_bars, include_target=False)
            if len(feat) < 1:
                return None
            last_feat = feat.iloc[[-1]]
            return float(self.predictor.predict_proba(last_feat)[0])
        except Exception:
            logger.debug("ML prediction failed", exc_info=True)
            return None

    def get_required_indicators(self) -> list[str]:
        return self.fib_strategy.get_required_indicators()

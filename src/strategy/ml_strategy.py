"""DEPRECATED — ML-based trading strategy using XGBoost predictions.

Walk-Forward results: 0% robustness at all window counts.
ML feature-based prediction does not work as a standalone strategy.
See ml_regime_strategy.py for the meta-model approach (also failed).
"""

from __future__ import annotations

import pandas as pd

from config.settings import SYMBOL
from src.ml.features import build_features
from src.ml.models import SignalPredictor
from src.strategy.base import BaseStrategy, Signal, TradeSignal


class MLStrategy(BaseStrategy):
    """Trade based on XGBoost probability predictions.

    Uses a trained model to predict probability of a profitable long.
    LONG when P(profit) > long_threshold.
    SHORT when P(profit) < short_threshold.
    """

    name = "ml_xgboost"

    def __init__(
        self,
        predictor: SignalPredictor,
        long_threshold: float = 0.60,
        short_threshold: float = 0.35,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        future_bars: int = 12,
        symbol: str = SYMBOL,
    ) -> None:
        self.predictor = predictor
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.future_bars = future_bars
        self.symbol = symbol
        self._last_entry_idx = -999

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate signal from ML model prediction.

        Args:
            df: OHLCV DataFrame (raw — features built internally).

        Returns:
            TradeSignal.
        """
        if len(df) < 60:
            return self._hold(df)

        # Cooldown
        current_idx = len(df)
        if current_idx - self._last_entry_idx < 6:
            return self._hold(df)

        # Build features for current bar (no target needed for inference)
        try:
            feat = build_features(df, future_bars=self.future_bars, include_target=False)
            if len(feat) < 1:
                return self._hold(df)
        except Exception:
            return self._hold(df)

        last_feat = feat.iloc[[-1]]
        prob = float(self.predictor.predict_proba(last_feat)[0])

        close = float(df["close"].iloc[-1])
        ts = df.index[-1]
        atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else close * 0.01

        if prob > self.long_threshold:
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=prob,
                stop_loss=close - atr * self.atr_sl_mult,
                take_profit=close + atr * self.atr_tp_mult,
                metadata={"strategy": self.name, "ml_prob": prob},
            )

        if prob < self.short_threshold:
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=1 - prob,
                stop_loss=close + atr * self.atr_sl_mult,
                take_profit=close - atr * self.atr_tp_mult,
                metadata={"strategy": self.name, "ml_prob": prob},
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
        return ["rsi_14", "atr_14", "ema_20", "ema_50"]

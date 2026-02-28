"""Portfolio strategy — runs multiple sub-strategies with weighted allocation.

INFRASTRUCTURE — Used by backtest walk-forward and live trading to combine
multiple strategies into a single portfolio with weighted capital allocation.

Walk-Forward Validated Portfolios (Phase 14-15, 9-window):
    RSI+DC 50/50       → 77% robustness, OOS +20.27%, 51 trades  ★ BEST
    VWAP+DC 50/50      → 77% robustness, OOS +19.31%, 51 trades
    RSI+VWAP+DC equal  → 77% robustness, OOS +17.54%, 76 trades
    4-strat equal 25%  → 77% robustness, OOS +16.56%, 103 trades (dilution)

Walk-Forward Validated Portfolios (Phase 14, 7-window):
    DC+CCI 50/50       → 85% robustness, OOS +24.46%, 58 trades  ★ BEST 7w
    VWAP+DC 50/50      → 85% robustness, OOS +19.64%, 51 trades
    RSI+VWAP+DC equal  → 85% robustness, OOS +17.04%, 77 trades

Key Findings (Phase 14-15):
    - 77% is the STRUCTURAL CEILING at 9 windows (W2 and W6 negative everywhere)
    - All weight variants (50/50, 60/40, 40/20/40) hit the same 77% ceiling
    - 50/50 is optimal: best return at max robustness
    - 4-strategy ensemble adds no value: same 77% but lower return (dilution)
    - Adding new strategies (ROC, Keltner, MFI) reaches 77% but never exceeds it

Usage:
    from src.strategy.portfolio import PortfolioStrategy
    from src.strategy.mtf_filter import MultiTimeframeFilter
    from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
    from src.strategy.donchian_trend import DonchianTrendStrategy

    rsi = MultiTimeframeFilter(RSIMeanReversionStrategy(...))
    dc = MultiTimeframeFilter(DonchianTrendStrategy(...))
    portfolio = PortfolioStrategy([rsi, dc], weights=[0.5, 0.5])
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.strategy.base import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger(__name__)


class PortfolioStrategy(BaseStrategy):
    """Run multiple strategies and allocate capital by weight.

    When a sub-strategy signals, its position is scaled by its weight.
    If multiple strategies signal the same direction, the highest-confidence
    signal is used. Conflicting directions (LONG + SHORT) result in HOLD.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        weights: list[float],
    ) -> None:
        if len(strategies) != len(weights):
            raise ValueError("strategies and weights must have same length")
        self.strategies = strategies
        self.weights = weights
        self.name = "portfolio_" + "+".join(s.name for s in strategies)

    def set_htf_data(self, df_htf: pd.DataFrame) -> None:
        """Pass HTF data to all sub-strategies that support it."""
        for s in self.strategies:
            if hasattr(s, "set_htf_data"):
                s.set_htf_data(df_htf)

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate signal from sub-strategies.

        Returns the highest-confidence non-HOLD signal with portfolio_weight
        in metadata. Conflicting directions produce HOLD.
        """
        candidates: list[tuple[TradeSignal, float]] = []

        for strategy, weight in zip(self.strategies, self.weights):
            sig = strategy.generate_signal(df)
            if sig.signal not in (Signal.HOLD,):
                candidates.append((sig, weight))

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
                "[PORTFOLIO] Conflicting signals: %d LONG vs %d SHORT → HOLD",
                len(longs), len(shorts),
            )
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=candidates[0][0].symbol,
                price=float(df["close"].iloc[-1]),
                timestamp=df.index[-1],
                metadata={"blocked_by": "portfolio_conflict"},
            )

        # Pick highest confidence signal
        best_sig, best_weight = max(candidates, key=lambda x: x[0].confidence)
        best_sig.metadata["portfolio_weight"] = best_weight
        best_sig.metadata["portfolio_name"] = self.name

        logger.info(
            "[PORTFOLIO] %s signal from %s (weight=%.0f%%, conf=%.3f)",
            best_sig.signal.value,
            best_sig.metadata.get("strategy", "?"),
            best_weight * 100,
            best_sig.confidence,
        )

        return best_sig

    def get_required_indicators(self) -> list[str]:
        indicators: set[str] = set()
        for s in self.strategies:
            indicators.update(s.get_required_indicators())
        return list(indicators)

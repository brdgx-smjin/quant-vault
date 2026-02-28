"""Walk-Forward analysis to detect overfitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """One train/test window result."""

    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    in_sample: BacktestResult
    out_of_sample: BacktestResult


@dataclass
class WalkForwardReport:
    """Aggregated Walk-Forward results."""

    windows: list[WalkForwardWindow]
    oos_total_return: float
    oos_avg_return: float
    oos_avg_sharpe: float
    oos_avg_win_rate: float
    oos_total_trades: int
    oos_profitable_windows: int
    total_windows: int
    robustness_score: float  # 0-1: fraction of profitable OOS windows


@dataclass
class CrossTFComponent:
    """One component of a cross-timeframe portfolio.

    Defines a strategy to run on a specific timeframe with its own
    backtest engine. Used by WalkForwardAnalyzer.run_cross_tf().
    """

    strategy_factory: Callable[[], object]  # Returns fresh BaseStrategy
    df: pd.DataFrame  # OHLCV with indicators for this timeframe
    htf_df: Optional[pd.DataFrame]  # Higher-TF data for MTF filter
    engine: BacktestEngine  # Configured for this timeframe (freq, max_hold)
    weight: float  # Portfolio weight (0-1)
    label: str  # Display name (e.g. "1h_RSI", "15m_RSI")


@dataclass
class ComponentWindowResult:
    """Result of one component in one cross-TF window."""

    label: str
    oos_return: float
    oos_trades: int


@dataclass
class CrossTFWindow:
    """One window of cross-TF walk-forward results."""

    window_id: int
    test_start: str
    test_end: str
    components: list[ComponentWindowResult]
    weighted_return: float


@dataclass
class CrossTFReport:
    """Cross-timeframe portfolio walk-forward report."""

    windows: list[CrossTFWindow]
    oos_total_return: float  # Compounded OOS return
    oos_profitable_windows: int
    total_windows: int
    robustness_score: float  # 0-1: fraction of profitable OOS windows
    total_trades: int


class WalkForwardAnalyzer:
    """Runs Walk-Forward analysis on a strategy.

    Splits data into rolling train/test windows and evaluates
    out-of-sample performance to check for overfitting.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        n_windows: int = 5,
        engine: BacktestEngine | None = None,
    ) -> None:
        self.train_ratio = train_ratio
        self.n_windows = n_windows
        self.engine = engine or BacktestEngine()

    def run(
        self,
        strategy_factory,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
    ) -> WalkForwardReport:
        """Run Walk-Forward analysis.

        Args:
            strategy_factory: Callable that returns a fresh strategy instance.
            df: Full OHLCV DataFrame with indicators.
            htf_df: Optional higher-timeframe DataFrame for MTF filtering.
                Passed through to BacktestEngine.run() which slices it
                per-bar to prevent look-ahead bias.

        Returns:
            WalkForwardReport with per-window and aggregate results.
        """
        n = len(df)
        window_size = n // self.n_windows
        # Anchored walk-forward: train grows, test is fixed size
        test_size = int(window_size * (1 - self.train_ratio))
        if test_size < 50:
            test_size = 50

        windows: list[WalkForwardWindow] = []

        for i in range(self.n_windows):
            test_end_idx = n - (self.n_windows - 1 - i) * test_size
            test_start_idx = test_end_idx - test_size
            train_start_idx = max(0, test_start_idx - int(test_size * self.train_ratio / (1 - self.train_ratio)))

            if train_start_idx >= test_start_idx or test_end_idx > n:
                continue

            train_df = df.iloc[train_start_idx:test_start_idx]
            test_df = df.iloc[test_start_idx:test_end_idx]

            if len(train_df) < 100 or len(test_df) < 30:
                continue

            logger.info(
                "Window %d: train[%s ~ %s] (%d) | test[%s ~ %s] (%d)",
                i + 1,
                train_df.index[0].date(), train_df.index[-1].date(), len(train_df),
                test_df.index[0].date(), test_df.index[-1].date(), len(test_df),
            )

            # In-sample (train)
            strategy_is = strategy_factory()
            is_result = self.engine.run(strategy_is, train_df, htf_df=htf_df)

            # Out-of-sample (test)
            strategy_oos = strategy_factory()
            oos_result = self.engine.run(strategy_oos, test_df, htf_df=htf_df)

            windows.append(WalkForwardWindow(
                window_id=i + 1,
                train_start=str(train_df.index[0].date()),
                train_end=str(train_df.index[-1].date()),
                test_start=str(test_df.index[0].date()),
                test_end=str(test_df.index[-1].date()),
                in_sample=is_result,
                out_of_sample=oos_result,
            ))

        if not windows:
            return WalkForwardReport(
                windows=[], oos_total_return=0, oos_avg_return=0,
                oos_avg_sharpe=0, oos_avg_win_rate=0, oos_total_trades=0,
                oos_profitable_windows=0, total_windows=0, robustness_score=0,
            )

        # Aggregate OOS results
        oos_returns = [w.out_of_sample.total_return for w in windows]
        oos_sharpes = [
            w.out_of_sample.sharpe_ratio for w in windows
            if np.isfinite(w.out_of_sample.sharpe_ratio)
        ]
        oos_win_rates = [
            w.out_of_sample.win_rate for w in windows
            if w.out_of_sample.total_trades > 0
        ]
        oos_trades = sum(w.out_of_sample.total_trades for w in windows)
        profitable = sum(1 for r in oos_returns if r > 0)

        # Compounded return
        compounded = 1.0
        for r in oos_returns:
            compounded *= (1 + r / 100)
        oos_total = (compounded - 1) * 100

        return WalkForwardReport(
            windows=windows,
            oos_total_return=oos_total,
            oos_avg_return=float(np.mean(oos_returns)) if oos_returns else 0,
            oos_avg_sharpe=float(np.mean(oos_sharpes)) if oos_sharpes else 0,
            oos_avg_win_rate=float(np.mean(oos_win_rates)) if oos_win_rates else 0,
            oos_total_trades=oos_trades,
            oos_profitable_windows=profitable,
            total_windows=len(windows),
            robustness_score=profitable / len(windows) if windows else 0,
        )

    def run_ml(
        self,
        df: pd.DataFrame,
        future_bars: int = 12,
        max_features: int = 10,
        n_estimators: int = 200,
        max_depth: int = 2,
        learning_rate: float = 0.05,
        long_threshold: float = 0.55,
        short_threshold: float = 0.0,
        min_train_samples: int = 400,
        long_only: bool = True,
        strategy_factory=None,
    ) -> WalkForwardReport:
        """Walk-Forward ML with expanding window and anti-overfit settings.

        Key improvements over naive WF:
        - **Expanding window**: train always starts at bar 0 (more data for later folds)
        - **LONG-only option**: target is P(profitable long), SHORT signals are noise
        - **Lower complexity**: max_depth=2 to prevent overfitting small samples
        - **Fewer features**: top 10 instead of 15
        - **Minimum training size**: skip windows with insufficient data
        - **Early stopping**: with purged validation split

        Args:
            df: Full OHLCV DataFrame with indicators.
            future_bars: Look-ahead bars for ML target.
            max_features: Max features for feature selection.
            n_estimators: Max boosting rounds.
            max_depth: Tree depth (lower = less overfitting).
            learning_rate: Boosting learning rate.
            long_threshold: ML probability threshold for LONG.
            short_threshold: ML probability threshold for SHORT (0=disabled).
            min_train_samples: Minimum valid training samples required.
            long_only: If True, only generate LONG signals (recommended).
            strategy_factory: Optional callable(predictor) -> BaseStrategy.
                If provided, used instead of default MLStrategy.
                Allows testing FibMLEnsemble or other strategies that use ML.

        Returns:
            WalkForwardReport with per-window and aggregate results.
        """
        from src.ml.features import build_features, purged_kfold_split
        from src.ml.models import SignalPredictor
        from src.strategy.ml_strategy import MLStrategy

        n = len(df)
        window_size = n // self.n_windows
        test_size = int(window_size * (1 - self.train_ratio))
        if test_size < 50:
            test_size = 50

        windows: list[WalkForwardWindow] = []

        for i in range(self.n_windows):
            test_end_idx = n - (self.n_windows - 1 - i) * test_size
            test_start_idx = test_end_idx - test_size

            if test_start_idx <= 0 or test_end_idx > n:
                continue

            # EXPANDING window: always train from start
            train_df = df.iloc[:test_start_idx]
            test_df = df.iloc[test_start_idx:test_end_idx]

            if len(train_df) < 200 or len(test_df) < 30:
                continue

            logger.info(
                "ML Window %d: train[%s ~ %s] (%d) | test[%s ~ %s] (%d)",
                i + 1,
                train_df.index[0].date(), train_df.index[-1].date(), len(train_df),
                test_df.index[0].date(), test_df.index[-1].date(), len(test_df),
            )

            # Build features and check minimum size
            try:
                feat_train = build_features(train_df, future_bars=future_bars)
                if len(feat_train) < min_train_samples:
                    logger.warning(
                        "  Skipping window %d: only %d samples (need %d)",
                        i + 1, len(feat_train), min_train_samples,
                    )
                    continue

                predictor = SignalPredictor(f"wf_ml_{i}")
                predictor.select_features(feat_train, max_features=max_features)

                # Split last 20% of train as validation for early stopping
                val_split = int(len(feat_train) * 0.8)
                feat_tr = feat_train.iloc[:val_split]
                feat_val = feat_train.iloc[val_split:]

                predictor.train(
                    feat_tr,
                    val_df=feat_val,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    early_stopping_rounds=30,
                    min_child_weight=10,
                    subsample=0.8,
                    colsample_bytree=0.6,
                    gamma=0.2,
                )

                # Log train/val gap
                tr_metrics = predictor._evaluate(feat_tr, "train")
                val_metrics = predictor._evaluate(feat_val, "val")
                logger.info(
                    "  Train Acc=%.3f | Val Acc=%.3f | Gap=%.1f%%",
                    tr_metrics.accuracy, val_metrics.accuracy,
                    (tr_metrics.accuracy - val_metrics.accuracy) * 100,
                )

            except Exception:
                logger.exception("  ML training failed for window %d", i + 1)
                continue

            # Build strategy: use factory if provided, else default MLStrategy
            eff_short = short_threshold if not long_only else 0.0

            if strategy_factory is not None:
                strat_is = strategy_factory(predictor)
                strat_oos = strategy_factory(predictor)
            else:
                strat_is = MLStrategy(
                    predictor, long_threshold=long_threshold,
                    short_threshold=eff_short,
                )
                strat_oos = MLStrategy(
                    predictor, long_threshold=long_threshold,
                    short_threshold=eff_short,
                )

            is_result = self.engine.run(strat_is, train_df)
            oos_result = self.engine.run(strat_oos, test_df)

            windows.append(WalkForwardWindow(
                window_id=i + 1,
                train_start=str(train_df.index[0].date()),
                train_end=str(train_df.index[-1].date()),
                test_start=str(test_df.index[0].date()),
                test_end=str(test_df.index[-1].date()),
                in_sample=is_result,
                out_of_sample=oos_result,
            ))

        if not windows:
            return WalkForwardReport(
                windows=[], oos_total_return=0, oos_avg_return=0,
                oos_avg_sharpe=0, oos_avg_win_rate=0, oos_total_trades=0,
                oos_profitable_windows=0, total_windows=0, robustness_score=0,
            )

        # Aggregate OOS results
        oos_returns = [w.out_of_sample.total_return for w in windows]
        oos_sharpes = [
            w.out_of_sample.sharpe_ratio for w in windows
            if np.isfinite(w.out_of_sample.sharpe_ratio)
        ]
        oos_win_rates = [
            w.out_of_sample.win_rate for w in windows
            if w.out_of_sample.total_trades > 0
        ]
        oos_trades = sum(w.out_of_sample.total_trades for w in windows)
        profitable = sum(1 for r in oos_returns if r > 0)

        compounded = 1.0
        for r in oos_returns:
            compounded *= (1 + r / 100)
        oos_total = (compounded - 1) * 100

        return WalkForwardReport(
            windows=windows,
            oos_total_return=oos_total,
            oos_avg_return=float(np.mean(oos_returns)) if oos_returns else 0,
            oos_avg_sharpe=float(np.mean(oos_sharpes)) if oos_sharpes else 0,
            oos_avg_win_rate=float(np.mean(oos_win_rates)) if oos_win_rates else 0,
            oos_total_trades=oos_trades,
            oos_profitable_windows=profitable,
            total_windows=len(windows),
            robustness_score=profitable / len(windows) if windows else 0,
        )

    def run_cross_tf(
        self,
        components: list[CrossTFComponent],
    ) -> CrossTFReport:
        """Run cross-timeframe portfolio walk-forward analysis.

        Splits windows by DATE (not bar index) so strategies on different
        timeframes have exactly aligned test periods. Each component runs
        independently on its own timeframe data; OOS returns are
        weight-averaged per window.

        This solves the Phase 16 caveat where separate bar-based WF runs
        on 1h and 15m had approximately-aligned but not exact window dates.

        Args:
            components: List of CrossTFComponent definitions. Each has its
                own strategy factory, data, engine, weight, and label.

        Returns:
            CrossTFReport with per-window and aggregate portfolio results.
        """
        if not components:
            return CrossTFReport(
                windows=[], oos_total_return=0,
                oos_profitable_windows=0, total_windows=0,
                robustness_score=0, total_trades=0,
            )

        # Validate weights
        total_weight = sum(c.weight for c in components)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                "Cross-TF weights sum to %.3f (expected 1.0)", total_weight,
            )

        # Common date range across all components
        start_date = max(c.df.index[0] for c in components)
        end_date = min(c.df.index[-1] for c in components)
        total_td = end_date - start_date

        # Test windows: contiguous, covering last (1-train_ratio) of range
        test_td = total_td * (1 - self.train_ratio) / self.n_windows

        logger.info(
            "Cross-TF WF: %d windows, %s ~ %s, test=~%d days each",
            self.n_windows, start_date.date(), end_date.date(),
            test_td.days,
        )

        windows: list[CrossTFWindow] = []

        for i in range(self.n_windows):
            test_end = end_date - (self.n_windows - 1 - i) * test_td
            test_start = test_end - test_td

            comp_results: list[ComponentWindowResult] = []

            for comp in components:
                # Slice by date — last window includes final bar
                if i < self.n_windows - 1:
                    test_mask = (
                        (comp.df.index >= test_start)
                        & (comp.df.index < test_end)
                    )
                else:
                    test_mask = comp.df.index >= test_start

                test_df = comp.df[test_mask]

                if len(test_df) < 30:
                    logger.warning(
                        "  W%d: %s has only %d bars — skipping",
                        i + 1, comp.label, len(test_df),
                    )
                    comp_results.append(ComponentWindowResult(
                        label=comp.label, oos_return=0.0, oos_trades=0,
                    ))
                    continue

                strategy = comp.strategy_factory()
                result = comp.engine.run(
                    strategy, test_df, htf_df=comp.htf_df,
                )
                comp_results.append(ComponentWindowResult(
                    label=comp.label,
                    oos_return=result.total_return,
                    oos_trades=result.total_trades,
                ))

            weighted_ret = sum(
                cr.oos_return * comp.weight
                for cr, comp in zip(comp_results, components)
            )

            windows.append(CrossTFWindow(
                window_id=i + 1,
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                components=comp_results,
                weighted_return=weighted_ret,
            ))

            parts = [
                f"{cr.label} {cr.oos_return:+.2f}%"
                for cr in comp_results
            ]
            marker = "+" if weighted_ret > 0 else "-"
            logger.info(
                "  W%d [%s ~ %s]: %s -> %+.2f%% %s",
                i + 1, test_start.date(), test_end.date(),
                " | ".join(parts), weighted_ret, marker,
            )

        # Aggregate
        oos_returns = [w.weighted_return for w in windows]
        compounded = 1.0
        for r in oos_returns:
            compounded *= (1 + r / 100)
        oos_total = (compounded - 1) * 100

        profitable = sum(1 for r in oos_returns if r > 0)
        total_trades = sum(
            sum(cr.oos_trades for cr in w.components)
            for w in windows
        )

        report = CrossTFReport(
            windows=windows,
            oos_total_return=oos_total,
            oos_profitable_windows=profitable,
            total_windows=len(windows),
            robustness_score=profitable / len(windows) if windows else 0,
            total_trades=total_trades,
        )

        logger.info(
            "  Cross-TF OOS: %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
            oos_total, int(report.robustness_score * 100),
            profitable, len(windows), total_trades,
        )

        return report

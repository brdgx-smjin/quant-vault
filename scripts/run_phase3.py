#!/usr/bin/env python3
"""Phase 3: Walk-Forward validation + improved ML model training + comprehensive backtest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import pandas as pd

from config.settings import DATA_DIR, SYMBOL
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.indicators.basic import BasicIndicators
from src.ml.features import build_features, split_train_test
from src.ml.models import SignalPredictor
from src.ml.training import train_with_cv
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategy.ml_strategy import MLStrategy
from src.strategy.ensemble import EnsembleStrategy
from src.strategy.fib_ml_ensemble import FibMLEnsembleStrategy
from src.monitoring.logger import setup_logging

logger = setup_logging("phase3")

# Enable backtest engine trade logging
logging.getLogger("src.backtest.engine").setLevel(logging.INFO)
logging.getLogger("src.backtest.engine").addHandler(
    logging.FileHandler("logs/backtest_trades.log")
)
# Enable ML logging
logging.getLogger("src.ml").setLevel(logging.INFO)

SYMBOL_FILE = SYMBOL.replace("/", "_").replace(":", "_")


def load_data(timeframe: str) -> pd.DataFrame:
    path = f"{DATA_DIR}/processed/{SYMBOL_FILE}_{timeframe}.parquet"
    df = pd.read_parquet(path)
    return BasicIndicators.add_all(df)


def print_result(name: str, r: BacktestResult) -> None:
    logger.info("  %-30s  %+7.2f%% | Sharpe %5.2f | WR %5.1f%% | %3d trades | PF %5.2f | DD %5.1f%%",
                name, r.total_return, r.sharpe_ratio, r.win_rate * 100,
                r.total_trades, r.profit_factor, r.max_drawdown)


def main() -> None:
    logger.info("=" * 72)
    logger.info("  PHASE 3 — Walk-Forward + Improved ML | %s | 4h", SYMBOL)
    logger.info("=" * 72)
    logger.info("")

    df_4h = load_data("4h")
    logger.info("Loaded 4h data: %d bars (%s ~ %s)",
                len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine = BacktestEngine(initial_capital=10_000, max_hold_bars=48)

    # ================================================================
    # PART 1: Walk-Forward on RSI MeanRev (best Phase 2 strategy)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 1: Walk-Forward Analysis — RSI_MeanRev 4h")
    logger.info("─" * 72)

    def rsi_factory():
        return RSIMeanReversionStrategy(
            rsi_oversold=35.0, rsi_overbought=65.0,
            atr_sl_mult=1.5, atr_tp_mult=2.5,
        )

    wf = WalkForwardAnalyzer(train_ratio=0.7, n_windows=5, engine=engine)
    wf_report = wf.run(rsi_factory, df_4h)

    logger.info("")
    for w in wf_report.windows:
        logger.info("  Window %d: IS %+6.2f%% (WR %.0f%%, %d tr) | OOS %+6.2f%% (WR %.0f%%, %d tr)",
                    w.window_id,
                    w.in_sample.total_return, w.in_sample.win_rate * 100, w.in_sample.total_trades,
                    w.out_of_sample.total_return, w.out_of_sample.win_rate * 100, w.out_of_sample.total_trades)

    logger.info("")
    logger.info("  Walk-Forward Summary:")
    logger.info("    OOS Compounded Return: %+.2f%%", wf_report.oos_total_return)
    logger.info("    OOS Avg Return/Window: %+.2f%%", wf_report.oos_avg_return)
    logger.info("    OOS Avg Sharpe:        %.2f", wf_report.oos_avg_sharpe)
    logger.info("    OOS Avg Win Rate:      %.1f%%", wf_report.oos_avg_win_rate * 100)
    logger.info("    OOS Total Trades:      %d", wf_report.oos_total_trades)
    logger.info("    Profitable Windows:    %d / %d", wf_report.oos_profitable_windows, wf_report.total_windows)
    logger.info("    Robustness Score:      %.0f%%", wf_report.robustness_score * 100)
    logger.info("")

    # ================================================================
    # PART 2: Walk-Forward on Fib+TrendFilter
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 2: Walk-Forward Analysis — Fib+TrendFilter 4h")
    logger.info("─" * 72)

    def fib_factory():
        return FibonacciRetracementStrategy(
            entry_levels=(0.5, 0.618), tolerance_pct=0.05,
            lookback=50, require_trend=True,
        )

    wf_fib = wf.run(fib_factory, df_4h)

    logger.info("")
    for w in wf_fib.windows:
        logger.info("  Window %d: IS %+6.2f%% (WR %.0f%%, %d tr) | OOS %+6.2f%% (WR %.0f%%, %d tr)",
                    w.window_id,
                    w.in_sample.total_return, w.in_sample.win_rate * 100, w.in_sample.total_trades,
                    w.out_of_sample.total_return, w.out_of_sample.win_rate * 100, w.out_of_sample.total_trades)
    logger.info("")
    logger.info("  Fib WF: OOS Return %+.2f%% | Robustness %.0f%%",
                wf_fib.oos_total_return, wf_fib.robustness_score * 100)
    logger.info("")

    # Also run Fib with 3 windows for fair comparison with FibML
    wf3_fib = WalkForwardAnalyzer(train_ratio=0.7, n_windows=3, engine=engine)
    wf_fib_3w = wf3_fib.run(fib_factory, df_4h)
    logger.info("  Fib WF (3w): OOS Return %+.2f%% | Robustness %.0f%%",
                wf_fib_3w.oos_total_return, wf_fib_3w.robustness_score * 100)
    for w in wf_fib_3w.windows:
        logger.info("    Window %d: IS %+6.2f%% (%d tr) | OOS %+6.2f%% (%d tr)",
                    w.window_id, w.in_sample.total_return, w.in_sample.total_trades,
                    w.out_of_sample.total_return, w.out_of_sample.total_trades)
    logger.info("")

    # ================================================================
    # PART 3: Improved XGBoost with Purged CV Pipeline
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 3: Improved XGBoost (Purged CV + Regularization)")
    logger.info("─" * 72)

    predictor, cv_report = train_with_cv(
        df_4h,
        model_name="xgb_signal_4h",
        future_bars=12,
        n_folds=5,
        purge_bars=12,
        max_features=15,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
    )

    logger.info("")
    logger.info("  CV Results:")
    logger.info("    Mean Accuracy: %.3f (+/- %.3f)", cv_report.mean_accuracy, cv_report.std_f1)
    logger.info("    Mean F1:       %.3f (+/- %.3f)", cv_report.mean_f1, cv_report.std_f1)
    logger.info("    Selected Features: %s", cv_report.selected_features)
    logger.info("    Avg Best Iteration: %d", cv_report.avg_best_iteration)
    if cv_report.final_train_metrics and cv_report.final_test_metrics:
        gap = cv_report.final_train_metrics.accuracy - cv_report.final_test_metrics.accuracy
        logger.info("    Final Train Acc: %.3f | Test Acc: %.3f | Gap: %.1f%%",
                    cv_report.final_train_metrics.accuracy,
                    cv_report.final_test_metrics.accuracy, gap * 100)
    logger.info("")

    # ================================================================
    # PART 3b: Walk-Forward ML Evaluation
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 3b: Walk-Forward ML (retrain per window)")
    logger.info("─" * 72)

    wf_ml = wf.run_ml(
        df_4h,
        future_bars=12,
        max_features=10,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        long_threshold=0.53,
        min_train_samples=300,
        long_only=True,
    )

    logger.info("")
    for w in wf_ml.windows:
        logger.info("  Window %d: IS %+6.2f%% (WR %.0f%%, %d tr) | OOS %+6.2f%% (WR %.0f%%, %d tr)",
                    w.window_id,
                    w.in_sample.total_return, w.in_sample.win_rate * 100, w.in_sample.total_trades,
                    w.out_of_sample.total_return, w.out_of_sample.win_rate * 100, w.out_of_sample.total_trades)
    logger.info("")
    logger.info("  ML WF: OOS Return %+.2f%% | Robustness %.0f%%",
                wf_ml.oos_total_return, wf_ml.robustness_score * 100)
    logger.info("")

    # ================================================================
    # PART 3c: Walk-Forward FibMLEnsemble (Fib primary + ML filter)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 3c: Walk-Forward FibMLEnsemble (Fib + ML confirmation)")
    logger.info("─" * 72)

    def fib_ml_factory(predictor):
        return FibMLEnsembleStrategy(
            predictor=predictor,
            ml_long_threshold=0.50,   # low threshold — Fib is primary filter
            ml_short_threshold=0.50,
            entry_levels=(0.5, 0.618),
            tolerance_pct=0.05,
            lookback=50,
            require_trend=True,
        )

    # Use 3 windows (more data per fold) for FibML
    wf3 = WalkForwardAnalyzer(train_ratio=0.7, n_windows=3, engine=engine)
    wf_fib_ml = wf3.run_ml(
        df_4h,
        future_bars=12,
        max_features=10,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        long_threshold=0.50,
        min_train_samples=300,
        long_only=True,
        strategy_factory=fib_ml_factory,
    )

    logger.info("")
    for w in wf_fib_ml.windows:
        logger.info("  Window %d: IS %+6.2f%% (WR %.0f%%, %d tr) | OOS %+6.2f%% (WR %.0f%%, %d tr)",
                    w.window_id,
                    w.in_sample.total_return, w.in_sample.win_rate * 100, w.in_sample.total_trades,
                    w.out_of_sample.total_return, w.out_of_sample.win_rate * 100, w.out_of_sample.total_trades)
    logger.info("")
    logger.info("  FibML WF: OOS Return %+.2f%% | Robustness %.0f%%",
                wf_fib_ml.oos_total_return, wf_fib_ml.robustness_score * 100)
    logger.info("")

    # ================================================================
    # PART 4: Strategy Comparison Backtest
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 4: Strategy Comparison Backtest (4h)")
    logger.info("─" * 72)

    strategies = [
        ("RSI_MeanRev", rsi_factory()),
        ("Fib+TrendFilter", fib_factory()),
        ("ML_LongOnly", MLStrategy(predictor, long_threshold=0.55, short_threshold=0.0)),
        ("ML_LongShort", MLStrategy(predictor, long_threshold=0.55, short_threshold=0.45)),
        ("ML+RSI_Ensemble", EnsembleStrategy(
            strategies=[(MLStrategy(predictor, long_threshold=0.55, short_threshold=0.0), 0.50),
                         (rsi_factory(), 0.50)],
            threshold=0.15,
        )),
        ("Fib+ML_Ensemble", EnsembleStrategy(
            strategies=[(MLStrategy(predictor, long_threshold=0.55, short_threshold=0.0), 0.40),
                         (fib_factory(), 0.60)],
            threshold=0.15,
        )),
        ("FibML_Confirm", FibMLEnsembleStrategy(
            predictor=predictor,
            ml_long_threshold=0.52,
            ml_short_threshold=0.48,
        )),
    ]

    logger.info("")
    results = []
    for name, strat in strategies:
        r = engine.run(strat, df_4h)
        print_result(name, r)
        results.append((name, r))

    # ================================================================
    # FINAL RANKING
    # ================================================================
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 3 FINAL RANKING (4h)")
    logger.info("=" * 72)
    results.sort(key=lambda x: x[1].sharpe_ratio if x[1].total_trades > 0 else -999, reverse=True)
    logger.info("  %-30s %8s %7s %6s %6s %6s %6s",
                "Strategy", "Return", "Sharpe", "MaxDD", "WR%", "#", "PF")
    logger.info("  " + "-" * 68)
    for name, r in results:
        if r.total_trades > 0:
            logger.info("  %-30s %+7.2f%% %7.2f %5.1f%% %5.1f%% %5d %6.2f",
                        name, r.total_return, r.sharpe_ratio, r.max_drawdown,
                        r.win_rate * 100, r.total_trades, r.profit_factor)
        else:
            logger.info("  %-30s       — (no trades)", name)

    logger.info("")
    logger.info("  Walk-Forward Robustness:")
    logger.info("    RSI_MeanRev:     %.0f%% (%d/%d windows profitable)",
                wf_report.robustness_score * 100, wf_report.oos_profitable_windows, wf_report.total_windows)
    logger.info("    Fib+TrendFilter: %.0f%% (%d/%d windows profitable)",
                wf_fib.robustness_score * 100, wf_fib.oos_profitable_windows, wf_fib.total_windows)
    logger.info("    ML (Walk-Fwd):   %.0f%% (%d/%d windows profitable)",
                wf_ml.robustness_score * 100, wf_ml.oos_profitable_windows, wf_ml.total_windows)
    logger.info("    FibML Confirm:   %.0f%% (%d/%d windows profitable)",
                wf_fib_ml.robustness_score * 100, wf_fib_ml.oos_profitable_windows, wf_fib_ml.total_windows)
    logger.info("    Fib (3-window):  %.0f%% (%d/%d windows profitable)",
                wf_fib_3w.robustness_score * 100, wf_fib_3w.oos_profitable_windows, wf_fib_3w.total_windows)
    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 3 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

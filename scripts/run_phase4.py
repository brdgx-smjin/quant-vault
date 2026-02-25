#!/usr/bin/env python3
"""Phase 4: BB Squeeze Breakout on 1h + comparison with existing strategies.

Tests the new BBSqueezeBreakout strategy on 1h timeframe (8769 bars) for
better statistical significance, then compares Walk-Forward robustness
against the proven Fib+TrendFilter and RSI_MeanRev strategies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import pandas as pd

from config.settings import DATA_DIR, SYMBOL
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.indicators.basic import BasicIndicators
from src.strategy.bb_squeeze_breakout import BBSqueezeBreakoutStrategy
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategy.ema_trend import EMATrendStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.monitoring.logger import setup_logging

logger = setup_logging("phase4")

# Enable backtest engine trade logging
logging.getLogger("src.backtest.engine").setLevel(logging.INFO)
logging.getLogger("src.backtest.engine").addHandler(
    logging.FileHandler("logs/backtest_trades.log")
)

SYMBOL_FILE = SYMBOL.replace("/", "_").replace(":", "_")


def load_data(timeframe: str) -> pd.DataFrame:
    """Load and add indicators to OHLCV data."""
    path = f"{DATA_DIR}/processed/{SYMBOL_FILE}_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    df.dropna(inplace=True)
    return df


def print_result(name: str, r: BacktestResult) -> None:
    logger.info(
        "  %-35s  %+8.2f%% | Sharpe %6.2f | WR %5.1f%% | %4d trades | PF %5.2f | DD %5.1f%%",
        name, r.total_return, r.sharpe_ratio, r.win_rate * 100,
        r.total_trades, r.profit_factor, r.max_drawdown,
    )


def print_wf_report(name: str, report) -> None:
    """Print Walk-Forward report summary."""
    logger.info("")
    for w in report.windows:
        logger.info(
            "  Window %d: IS %+7.2f%% (WR %.0f%%, %d tr) | OOS %+7.2f%% (WR %.0f%%, %d tr)",
            w.window_id,
            w.in_sample.total_return, w.in_sample.win_rate * 100, w.in_sample.total_trades,
            w.out_of_sample.total_return, w.out_of_sample.win_rate * 100, w.out_of_sample.total_trades,
        )
    logger.info("")
    logger.info("  %s Walk-Forward Summary:", name)
    logger.info("    OOS Compounded Return: %+.2f%%", report.oos_total_return)
    logger.info("    OOS Avg Return/Window: %+.2f%%", report.oos_avg_return)
    logger.info("    OOS Avg Sharpe:        %.2f", report.oos_avg_sharpe)
    logger.info("    OOS Avg Win Rate:      %.1f%%", report.oos_avg_win_rate * 100)
    logger.info("    OOS Total Trades:      %d", report.oos_total_trades)
    logger.info("    Profitable Windows:    %d / %d", report.oos_profitable_windows, report.total_windows)
    logger.info("    Robustness Score:      %.0f%%", report.robustness_score * 100)
    logger.info("")


def main() -> None:
    logger.info("=" * 72)
    logger.info("  PHASE 4 — BB Squeeze Breakout + Multi-TF | %s", SYMBOL)
    logger.info("=" * 72)
    logger.info("")

    # Load data
    df_1h = load_data("1h")
    df_4h = load_data("4h")
    logger.info("Loaded 1h data: %d bars (%s ~ %s)",
                len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("Loaded 4h data: %d bars (%s ~ %s)",
                len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(initial_capital=10_000, max_hold_bars=72)
    engine_4h = BacktestEngine(initial_capital=10_000, max_hold_bars=48)

    # ================================================================
    # PART 1: BB Squeeze Breakout — Walk-Forward on 1h
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 1: Walk-Forward — BBSqueeze Breakout (1h)")
    logger.info("─" * 72)

    def bb_squeeze_factory():
        return BBSqueezeBreakoutStrategy(
            squeeze_lookback=100,
            squeeze_pctile=25.0,
            vol_mult=1.3,
            atr_sl_mult=2.0,
            rr_ratio=2.5,
            require_trend=True,
            cooldown_bars=8,
        )

    wf_1h = WalkForwardAnalyzer(train_ratio=0.7, n_windows=5, engine=engine_1h)
    wf_bb = wf_1h.run(bb_squeeze_factory, df_1h)
    print_wf_report("BBSqueeze_1h", wf_bb)

    # ================================================================
    # PART 1b: BB Squeeze — no trend filter variant
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 1b: Walk-Forward — BBSqueeze NoTrend (1h)")
    logger.info("─" * 72)

    def bb_squeeze_notrend_factory():
        return BBSqueezeBreakoutStrategy(
            squeeze_lookback=100,
            squeeze_pctile=25.0,
            vol_mult=1.3,
            atr_sl_mult=2.0,
            rr_ratio=2.5,
            require_trend=False,
            cooldown_bars=8,
        )

    wf_bb_nt = wf_1h.run(bb_squeeze_notrend_factory, df_1h)
    print_wf_report("BBSqueeze_NoTrend_1h", wf_bb_nt)

    # ================================================================
    # PART 1c: BB Squeeze with MTF (4h) trend filter
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 1c: Walk-Forward — BBSqueeze + 4h MTF Filter (1h)")
    logger.info("─" * 72)

    def bb_squeeze_mtf_factory():
        base = BBSqueezeBreakoutStrategy(
            squeeze_lookback=100,
            squeeze_pctile=25.0,
            vol_mult=1.3,
            atr_sl_mult=2.0,
            rr_ratio=2.5,
            require_trend=False,  # MTF filter handles trend
            cooldown_bars=8,
        )
        return MultiTimeframeFilter(base)

    wf_bb_mtf = wf_1h.run(bb_squeeze_mtf_factory, df_1h, htf_df=df_4h)
    print_wf_report("BBSqueeze_MTF_1h", wf_bb_mtf)

    # ================================================================
    # PART 2: Parameter sensitivity — tighter/wider squeeze
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 2: Parameter Sensitivity — Squeeze Percentile")
    logger.info("─" * 72)

    for pctile in [15.0, 20.0, 30.0, 35.0]:
        def make_factory(p=pctile):
            def factory():
                return BBSqueezeBreakoutStrategy(
                    squeeze_lookback=100,
                    squeeze_pctile=p,
                    vol_mult=1.3,
                    atr_sl_mult=2.0,
                    rr_ratio=2.5,
                    require_trend=True,
                    cooldown_bars=8,
                )
            return factory

        wf_test = wf_1h.run(make_factory(), df_1h)
        logger.info(
            "  Pctile=%.0f: OOS %+7.2f%% | Robustness %.0f%% (%d/%d) | %d trades",
            pctile, wf_test.oos_total_return, wf_test.robustness_score * 100,
            wf_test.oos_profitable_windows, wf_test.total_windows,
            wf_test.oos_total_trades,
        )
    logger.info("")

    # ================================================================
    # PART 3: Comparison — existing strategies on 1h
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 3: Walk-Forward Comparison — All Strategies on 1h")
    logger.info("─" * 72)

    # RSI MeanRev on 1h
    def rsi_factory_1h():
        return RSIMeanReversionStrategy(
            rsi_oversold=35.0, rsi_overbought=65.0,
            atr_sl_mult=1.5, atr_tp_mult=2.5,
        )

    wf_rsi_1h = wf_1h.run(rsi_factory_1h, df_1h)

    # EMA Trend on 1h
    def ema_factory_1h():
        return EMATrendStrategy(
            atr_sl_mult=2.0, atr_tp_mult=3.0,
            require_pullback=True,
        )

    wf_ema_1h = wf_1h.run(ema_factory_1h, df_1h)

    # Fib on 1h
    def fib_factory_1h():
        return FibonacciRetracementStrategy(
            entry_levels=(0.5, 0.618), tolerance_pct=0.05,
            lookback=50, require_trend=True,
        )

    wf_fib_1h = wf_1h.run(fib_factory_1h, df_1h)

    # EMA + MTF on 1h
    def ema_mtf_factory_1h():
        base = EMATrendStrategy(
            atr_sl_mult=2.0, atr_tp_mult=3.0,
            require_pullback=True,
        )
        return MultiTimeframeFilter(base)

    wf_ema_mtf_1h = wf_1h.run(ema_mtf_factory_1h, df_1h, htf_df=df_4h)

    logger.info("")
    logger.info("  %-35s %10s %11s %8s",
                "Strategy", "OOS Return", "Robustness", "Trades")
    logger.info("  " + "-" * 68)
    wf_results = [
        ("BBSqueeze_Trend_1h", wf_bb),
        ("BBSqueeze_NoTrend_1h", wf_bb_nt),
        ("BBSqueeze_MTF_1h", wf_bb_mtf),
        ("RSI_MeanRev_1h", wf_rsi_1h),
        ("EMA_Trend_1h", wf_ema_1h),
        ("EMA_MTF_1h", wf_ema_mtf_1h),
        ("Fib_TrendFilter_1h", wf_fib_1h),
    ]
    wf_results.sort(key=lambda x: x[1].robustness_score, reverse=True)
    for name, rpt in wf_results:
        logger.info(
            "  %-35s %+9.2f%% %10.0f%% %8d",
            name, rpt.oos_total_return, rpt.robustness_score * 100,
            rpt.oos_total_trades,
        )
    logger.info("")

    # ================================================================
    # PART 4: Full-period backtest comparison on 1h
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 4: Full-Period Backtest Comparison (1h)")
    logger.info("─" * 72)
    logger.info("")

    strategies = [
        ("BBSqueeze_Trend", bb_squeeze_factory()),
        ("BBSqueeze_NoTrend", bb_squeeze_notrend_factory()),
        ("RSI_MeanRev", rsi_factory_1h()),
        ("EMA_Trend", ema_factory_1h()),
        ("Fib_TrendFilter", fib_factory_1h()),
    ]

    full_results = []
    for name, strat in strategies:
        r = engine_1h.run(strat, df_1h)
        print_result(name, r)
        full_results.append((name, r))

    # Also test BBSqueeze+MTF on full period
    bb_mtf_strat = bb_squeeze_mtf_factory()
    bb_mtf_strat.set_htf_data(df_4h)
    r_mtf = engine_1h.run(bb_mtf_strat, df_1h, htf_df=df_4h)
    print_result("BBSqueeze_MTF", r_mtf)
    full_results.append(("BBSqueeze_MTF", r_mtf))

    ema_mtf_strat = ema_mtf_factory_1h()
    ema_mtf_strat.set_htf_data(df_4h)
    r_ema_mtf = engine_1h.run(ema_mtf_strat, df_1h, htf_df=df_4h)
    print_result("EMA_MTF", r_ema_mtf)
    full_results.append(("EMA_MTF", r_ema_mtf))

    # ================================================================
    # PART 5: Best strategy on 4h for comparison
    # ================================================================
    logger.info("")
    logger.info("─" * 72)
    logger.info("  PART 5: BBSqueeze on 4h (for comparison with Phase 3)")
    logger.info("─" * 72)

    def bb_squeeze_4h_factory():
        return BBSqueezeBreakoutStrategy(
            squeeze_lookback=50,  # shorter lookback for fewer bars
            squeeze_pctile=25.0,
            vol_mult=1.3,
            atr_sl_mult=2.0,
            rr_ratio=2.5,
            require_trend=True,
            cooldown_bars=6,
        )

    wf_4h = WalkForwardAnalyzer(train_ratio=0.7, n_windows=5, engine=engine_4h)
    wf_bb_4h = wf_4h.run(bb_squeeze_4h_factory, df_4h)
    print_wf_report("BBSqueeze_4h", wf_bb_4h)

    r_bb_4h = engine_4h.run(bb_squeeze_4h_factory(), df_4h)
    print_result("BBSqueeze_4h_full", r_bb_4h)
    logger.info("")

    # ================================================================
    # FINAL RANKING
    # ================================================================
    logger.info("=" * 72)
    logger.info("  PHASE 4 FINAL RANKING")
    logger.info("=" * 72)

    logger.info("")
    logger.info("  Walk-Forward Robustness (1h, 5 windows):")
    for name, rpt in wf_results:
        logger.info(
            "    %-30s %.0f%% (%d/%d profitable) | OOS %+.2f%%",
            name, rpt.robustness_score * 100,
            rpt.oos_profitable_windows, rpt.total_windows,
            rpt.oos_total_return,
        )

    logger.info("")
    logger.info("  Walk-Forward Robustness (4h, 5 windows):")
    logger.info(
        "    %-30s %.0f%% (%d/%d profitable) | OOS %+.2f%%",
        "BBSqueeze_4h", wf_bb_4h.robustness_score * 100,
        wf_bb_4h.oos_profitable_windows, wf_bb_4h.total_windows,
        wf_bb_4h.oos_total_return,
    )

    logger.info("")
    logger.info("  Full-Period Backtest (1h):")
    logger.info("  %-35s %8s %7s %6s %6s %6s %6s",
                "Strategy", "Return", "Sharpe", "MaxDD", "WR%", "#", "PF")
    logger.info("  " + "-" * 74)
    full_results.sort(key=lambda x: x[1].sharpe_ratio if x[1].total_trades > 0 else -999, reverse=True)
    for name, r in full_results:
        if r.total_trades > 0:
            logger.info(
                "  %-35s %+7.2f%% %7.2f %5.1f%% %5.1f%% %5d %6.2f",
                name, r.total_return, r.sharpe_ratio, r.max_drawdown,
                r.win_rate * 100, r.total_trades, r.profit_factor,
            )
        else:
            logger.info("  %-35s       — (no trades)", name)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 4 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

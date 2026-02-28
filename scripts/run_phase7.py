#!/usr/bin/env python3
"""Phase 7: Portfolio Deep Validation + New Strategy Candidates.

Task 1: BBSqueeze(1h)+Fib(4h) portfolio 7-window WF validation
Task 2: RSI Mean Reversion (1h) — new candidate from Phase 6 analysis
Task 3: Donchian Trend (1h) — pure trend following, portfolio diversifier
Task 4: RSI+DC 50/50 portfolio
Task 5: Cross-TF portfolio: 1hRSI/1hDC/15mRSI 33/33/34

All tested with 7-window Walk-Forward to improve statistical robustness
over Phase 6's 5-window results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, SYMBOL
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import (
    CrossTFComponent,
    WalkForwardAnalyzer,
)
from src.indicators.basic import BasicIndicators
from src.strategy.bb_squeeze_breakout import BBSqueezeBreakoutStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.monitoring.logger import setup_logging

logger = setup_logging("phase7")

# Suppress per-bar engine logging
logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)

SYMBOL_FILE = SYMBOL.replace("/", "_").replace(":", "_")
N_WINDOWS = 7


def load_data(timeframe: str) -> pd.DataFrame:
    """Load and add indicators to OHLCV data."""
    path = f"{DATA_DIR}/processed/{SYMBOL_FILE}_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    df.dropna(inplace=True)
    return df


def print_result(name: str, r: BacktestResult) -> None:
    logger.info(
        "  %-40s %+8.2f%% | Shp %5.2f | DD %5.1f%% | WR %5.1f%% | %4d tr | PF %5.2f",
        name, r.total_return, r.sharpe_ratio, r.max_drawdown,
        r.win_rate * 100, r.total_trades, r.profit_factor,
    )


def print_wf(name: str, report) -> None:
    for w in report.windows:
        logger.info(
            "  W%d: IS %+7.2f%% (WR %.0f%%, %d tr) | OOS %+7.2f%% (WR %.0f%%, %d tr)",
            w.window_id,
            w.in_sample.total_return, w.in_sample.win_rate * 100,
            w.in_sample.total_trades,
            w.out_of_sample.total_return, w.out_of_sample.win_rate * 100,
            w.out_of_sample.total_trades,
        )
    logger.info(
        "  OOS: %+.2f%% | Robustness: %.0f%% (%d/%d) | Trades: %d",
        report.oos_total_return, report.robustness_score * 100,
        report.oos_profitable_windows, report.total_windows,
        report.oos_total_trades,
    )


def print_cross_tf(report) -> None:
    for w in report.windows:
        parts = [
            f"{cr.label} {cr.oos_return:+.2f}%"
            for cr in w.components
        ]
        marker = "+" if w.weighted_return > 0 else "-"
        logger.info(
            "  W%d [%s ~ %s]: %s -> %+.2f%% %s",
            w.window_id, w.test_start, w.test_end,
            " | ".join(parts), w.weighted_return, marker,
        )
    logger.info(
        "  OOS: %+.2f%% | Robustness: %.0f%% (%d/%d) | Trades: %d",
        report.oos_total_return, int(report.robustness_score * 100),
        report.oos_profitable_windows, report.total_windows,
        report.total_trades,
    )


def main() -> None:
    logger.info("=" * 72)
    logger.info("  PHASE 7 — Portfolio Deep Validation + New Candidates")
    logger.info("=" * 72)
    logger.info("")

    # Load data
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")

    logger.info("1h data:  %d bars (%s ~ %s)",
                len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("15m data: %d bars (%s ~ %s)",
                len(df_15m), df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("4h data:  %d bars (%s ~ %s)",
                len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(initial_capital=10_000, max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(initial_capital=10_000, max_hold_bars=96, freq="15m")
    wf_1h = WalkForwardAnalyzer(train_ratio=0.7, n_windows=N_WINDOWS, engine=engine_1h)

    all_results = []

    # ================================================================
    # PART 1: BBSqueeze+MTF Baseline (7w)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 1: BBSqueeze+MTF Baseline (%dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    def bb_factory():
        base = BBSqueezeBreakoutStrategy(
            squeeze_lookback=100, squeeze_pctile=25.0,
            vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
            require_trend=False, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    wf_bb = wf_1h.run(bb_factory, df_1h, htf_df=df_4h)
    r_bb = engine_1h.run(bb_factory(), df_1h, htf_df=df_4h)
    print_wf("BBSqueeze+MTF", wf_bb)
    print_result("BBSqueeze+MTF Full", r_bb)
    logger.info("")
    all_results.append(("BBSqueeze+MTF_1h", wf_bb, r_bb))

    # ================================================================
    # PART 2: Fib 4h Standalone (7w)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 2: Fibonacci Retracement 4h (%dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    engine_4h = BacktestEngine(initial_capital=10_000, max_hold_bars=24)
    wf_4h = WalkForwardAnalyzer(train_ratio=0.7, n_windows=N_WINDOWS, engine=engine_4h)

    def fib_factory():
        return FibonacciRetracementStrategy(
            entry_levels=(0.5, 0.618),
            tp_extension=1.618,
            tolerance_pct=0.05,
            lookback=50,
            atr_sl_mult=1.5,
            require_trend=True,
        )

    wf_fib = wf_4h.run(fib_factory, df_4h)
    r_fib = engine_4h.run(fib_factory(), df_4h)
    print_wf("Fib_4h", wf_fib)
    print_result("Fib_4h Full", r_fib)
    logger.info("")
    all_results.append(("Fib_4h", wf_fib, r_fib))

    # ================================================================
    # PART 3: BBSqueeze+Fib Portfolio — Deep Validation (7w)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 3: Portfolio 50%% BBSqueeze(1h) + 50%% Fib(4h) (%dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    if wf_bb.windows and wf_fib.windows:
        n_win = min(len(wf_bb.windows), len(wf_fib.windows))
        port_oos_returns = []
        port_profitable = 0
        port_trades = 0
        for i in range(n_win):
            bb_ret = wf_bb.windows[i].out_of_sample.total_return
            fib_ret = wf_fib.windows[i].out_of_sample.total_return if i < len(wf_fib.windows) else 0.0
            combined = 0.5 * bb_ret + 0.5 * fib_ret
            port_oos_returns.append(combined)
            bb_tr = wf_bb.windows[i].out_of_sample.total_trades
            fib_tr = wf_fib.windows[i].out_of_sample.total_trades if i < len(wf_fib.windows) else 0
            port_trades += bb_tr + fib_tr
            logger.info(
                "  W%d: BB OOS %+6.2f%% + Fib OOS %+6.2f%% -> Port %+6.2f%%",
                i + 1, bb_ret, fib_ret, combined,
            )
            if combined > 0:
                port_profitable += 1

        compounded = 1.0
        for r in port_oos_returns:
            compounded *= (1 + r / 100)
        port_total = (compounded - 1) * 100
        port_robustness = port_profitable / n_win if n_win > 0 else 0

        logger.info(
            "  Portfolio OOS: %+.2f%% | Robustness: %.0f%% (%d/%d) | Trades: %d",
            port_total, port_robustness * 100, port_profitable, n_win, port_trades,
        )
    logger.info("")

    # ================================================================
    # PART 4: RSI Mean Reversion (1h) — New Candidate
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 4: RSI Mean Reversion + MTF (%dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    def rsi_1h_factory():
        base = RSIMeanReversionStrategy(
            rsi_oversold=35.0, rsi_overbought=65.0,
            atr_sl_mult=2.0, atr_tp_mult=3.0,
            cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    wf_rsi = wf_1h.run(rsi_1h_factory, df_1h, htf_df=df_4h)
    r_rsi = engine_1h.run(rsi_1h_factory(), df_1h, htf_df=df_4h)
    print_wf("RSI_MR+MTF_1h", wf_rsi)
    print_result("RSI_MR+MTF_1h Full", r_rsi)
    logger.info("")
    all_results.append(("RSI_MR+MTF_1h", wf_rsi, r_rsi))

    # ================================================================
    # PART 5: Donchian Trend (1h) — New Candidate
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 5: Donchian Trend + MTF (%dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    def dc_1h_factory():
        base = DonchianTrendStrategy(
            entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
            vol_mult=0.8, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    wf_dc = wf_1h.run(dc_1h_factory, df_1h, htf_df=df_4h)
    r_dc = engine_1h.run(dc_1h_factory(), df_1h, htf_df=df_4h)
    print_wf("DC+MTF_1h", wf_dc)
    print_result("DC+MTF_1h Full", r_dc)
    logger.info("")
    all_results.append(("DC+MTF_1h", wf_dc, r_dc))

    # ================================================================
    # PART 6: RSI+DC 50/50 Portfolio (1h) — Diversification Test
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 6: RSI+DC 50/50 Portfolio (%dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    if wf_rsi.windows and wf_dc.windows:
        n_win = min(len(wf_rsi.windows), len(wf_dc.windows))
        rsi_dc_oos = []
        rsi_dc_prof = 0
        rsi_dc_trades = 0
        for i in range(n_win):
            rsi_ret = wf_rsi.windows[i].out_of_sample.total_return
            dc_ret = wf_dc.windows[i].out_of_sample.total_return
            combined = 0.5 * rsi_ret + 0.5 * dc_ret
            rsi_dc_oos.append(combined)
            rsi_dc_trades += (
                wf_rsi.windows[i].out_of_sample.total_trades
                + wf_dc.windows[i].out_of_sample.total_trades
            )
            logger.info(
                "  W%d: RSI OOS %+6.2f%% + DC OOS %+6.2f%% -> Port %+6.2f%%",
                i + 1, rsi_ret, dc_ret, combined,
            )
            if combined > 0:
                rsi_dc_prof += 1

        compounded = 1.0
        for r in rsi_dc_oos:
            compounded *= (1 + r / 100)
        rsi_dc_total = (compounded - 1) * 100
        rsi_dc_robustness = rsi_dc_prof / n_win if n_win > 0 else 0

        logger.info(
            "  RSI+DC Portfolio OOS: %+.2f%% | Robustness: %.0f%% (%d/%d) | Trades: %d",
            rsi_dc_total, rsi_dc_robustness * 100, rsi_dc_prof, n_win, rsi_dc_trades,
        )
    logger.info("")

    # ================================================================
    # PART 7: Cross-TF Portfolio 1hRSI/1hDC/15mRSI 33/33/34
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 7: Cross-TF 1hRSI/1hDC/15mRSI 33/33/34 (9w)")
    logger.info("─" * 72)
    logger.info("")

    wf_ctf = WalkForwardAnalyzer(train_ratio=0.7, n_windows=9)

    components = [
        CrossTFComponent(
            strategy_factory=lambda: MultiTimeframeFilter(
                RSIMeanReversionStrategy(
                    rsi_oversold=35.0, rsi_overbought=65.0,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
            ),
            df=df_1h,
            htf_df=df_4h,
            engine=BacktestEngine(initial_capital=10_000, max_hold_bars=48, freq="1h"),
            weight=0.33,
            label="1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=lambda: MultiTimeframeFilter(
                DonchianTrendStrategy(
                    entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
                    vol_mult=0.8, cooldown_bars=6,
                )
            ),
            df=df_1h,
            htf_df=df_4h,
            engine=BacktestEngine(initial_capital=10_000, max_hold_bars=48, freq="1h"),
            weight=0.33,
            label="1hDC",
        ),
        CrossTFComponent(
            strategy_factory=lambda: MultiTimeframeFilter(
                RSIMeanReversionStrategy(
                    rsi_oversold=35.0, rsi_overbought=65.0,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
                )
            ),
            df=df_15m,
            htf_df=df_4h,
            engine=BacktestEngine(initial_capital=10_000, max_hold_bars=96, freq="15m"),
            weight=0.34,
            label="15mRSI",
        ),
    ]

    ctf_report = wf_ctf.run_cross_tf(components)
    print_cross_tf(ctf_report)
    logger.info("")

    # ================================================================
    # FINAL RANKING
    # ================================================================
    logger.info("=" * 72)
    logger.info("  PHASE 7 — FINAL RANKING")
    logger.info("=" * 72)
    logger.info("")

    # Sort by robustness first, then OOS return
    all_results.sort(
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    )

    logger.info(
        "  %-35s %8s %6s %7s %8s %6s %5s %5s",
        "Strategy", "OOS Ret", "WF Rob", "Tr(OOS)",
        "Full Ret", "MaxDD", "PF", "Shp",
    )
    logger.info("  " + "-" * 90)

    for name, rpt, r_full in all_results:
        logger.info(
            "  %-35s %+7.2f%% %5.0f%% %6d   %+7.2f%% %5.1f%% %5.2f %5.2f",
            name, rpt.oos_total_return, rpt.robustness_score * 100,
            rpt.oos_total_trades,
            r_full.total_return, r_full.max_drawdown,
            r_full.profit_factor, r_full.sharpe_ratio,
        )

    logger.info("")

    # Portfolio summaries
    logger.info("  Portfolio summaries:")
    logger.info("  BB+Fib 50/50 (7w):  OOS %+.2f%% | Robustness: %.0f%%",
                port_total, port_robustness * 100)
    logger.info("  RSI+DC 50/50 (7w):  OOS %+.2f%% | Robustness: %.0f%%",
                rsi_dc_total, rsi_dc_robustness * 100)
    logger.info("  Cross-TF 33/33/34 (9w): OOS %+.2f%% | Robustness: %.0f%%",
                ctf_report.oos_total_return, ctf_report.robustness_score * 100)
    logger.info("")

    # Recommendation
    if all_results:
        best_name, best_rpt, best_full = all_results[0]
        logger.info("  Best single strategy: %s", best_name)
        logger.info("    WF Robustness: %.0f%% (%d/%d)",
                    best_rpt.robustness_score * 100,
                    best_rpt.oos_profitable_windows, best_rpt.total_windows)
        logger.info("    OOS Return: %+.2f%%", best_rpt.oos_total_return)
        logger.info("    Full Return: %+.2f%% | DD %.1f%% | PF %.2f",
                    best_full.total_return, best_full.max_drawdown,
                    best_full.profit_factor)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 7 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

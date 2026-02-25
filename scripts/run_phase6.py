#!/usr/bin/env python3
"""Phase 6: Entry Quality Filters + Breakeven SL + Portfolio.

Tests qualitatively different improvements over Phase 4/5 baseline:
1. Candle body ratio filter — reject weak breakout candles (doji/spinning top)
2. Session time filter — avoid low-liquidity UTC hours
3. Breakeven SL at 1R — move SL to entry after reaching 1R profit
4. Combined filters — best candle + session + breakeven
5. Multi-strategy portfolio — BBSqueeze (1h) + Fib (4h) diversification

All variants undergo Walk-Forward analysis (5 windows) to validate OOS.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, SYMBOL
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.indicators.basic import BasicIndicators
from src.strategy.bb_squeeze_breakout import BBSqueezeBreakoutStrategy
from src.strategy.bb_squeeze_v3 import BBSqueezeV3Strategy
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.monitoring.logger import setup_logging

logger = setup_logging("phase6")

logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)

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


def main() -> None:
    logger.info("=" * 72)
    logger.info("  PHASE 6 — Entry Quality + Breakeven SL + Portfolio")
    logger.info("=" * 72)
    logger.info("")

    # Load data
    df_1h = load_data("1h")
    df_4h = load_data("4h")
    logger.info("1h data: %d bars (%s ~ %s)",
                len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("4h data: %d bars (%s ~ %s)",
                len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    # Quick data stats for session analysis
    hours = df_1h.index.hour
    logger.info("1h bar distribution by UTC hour (sample):")
    for h in [0, 3, 6, 9, 12, 15, 18, 21]:
        logger.info("  %02d:00 UTC — %d bars", h, (hours == h).sum())
    logger.info("")

    engine_1h = BacktestEngine(initial_capital=10_000, max_hold_bars=72)
    wf_1h = WalkForwardAnalyzer(train_ratio=0.7, n_windows=5, engine=engine_1h)

    all_results = []

    # ================================================================
    # PART 0: Phase 4 Baseline reproduction
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 0: Phase 4 Baseline — BBSqueeze+MTF Conservative (1h)")
    logger.info("─" * 72)

    def baseline_factory():
        base = BBSqueezeBreakoutStrategy(
            squeeze_lookback=100, squeeze_pctile=25.0,
            vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
            require_trend=False, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    wf_baseline = wf_1h.run(baseline_factory, df_1h, htf_df=df_4h)
    print_wf("Baseline_1h", wf_baseline)
    r_baseline = engine_1h.run(baseline_factory(), df_1h, htf_df=df_4h)
    print_result("Baseline Full", r_baseline)
    logger.info("")
    all_results.append(("Baseline_1h", wf_baseline, r_baseline))

    # ================================================================
    # PART 1: Candle Body Ratio filter
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 1: Candle Body Ratio Filter — BBSqueeze+MTF (1h)")
    logger.info("─" * 72)
    logger.info("")

    for body_ratio in [0.4, 0.5, 0.6]:
        def body_factory(br=body_ratio):
            base = BBSqueezeV3Strategy(
                squeeze_lookback=100, squeeze_pctile=25.0,
                vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
                require_trend=False, cooldown_bars=6,
                min_body_ratio=br,
            )
            return MultiTimeframeFilter(base)

        rpt = wf_1h.run(body_factory, df_1h, htf_df=df_4h)
        r_full = engine_1h.run(body_factory(), df_1h, htf_df=df_4h)

        label = f"Body>{body_ratio:.0%}"
        logger.info("  --- %s ---", label)
        print_wf(label, rpt)
        print_result(label + " Full", r_full)
        logger.info("")
        all_results.append((label, rpt, r_full))

    # ================================================================
    # PART 2: Session Time Filter
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 2: Session Time Filter — BBSqueeze+MTF (1h)")
    logger.info("─" * 72)
    logger.info("")

    session_configs = [
        ("NoAsianLate", (0, 1, 2, 3, 4, 5)),  # Block 0-5 UTC
        ("NoAsian", (0, 1, 2, 3, 4, 5, 6, 7)),  # Block 0-7 UTC
        ("USEuroOnly", tuple(range(0, 8)) + tuple(range(22, 24))),  # Only 8-21 UTC
    ]

    for label, blocked in session_configs:
        def session_factory(bh=blocked):
            base = BBSqueezeV3Strategy(
                squeeze_lookback=100, squeeze_pctile=25.0,
                vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
                require_trend=False, cooldown_bars=6,
                blocked_hours=bh,
            )
            return MultiTimeframeFilter(base)

        rpt = wf_1h.run(session_factory, df_1h, htf_df=df_4h)
        r_full = engine_1h.run(session_factory(), df_1h, htf_df=df_4h)

        logger.info("  --- %s ---", label)
        print_wf(label, rpt)
        print_result(label + " Full", r_full)
        logger.info("")
        all_results.append((label, rpt, r_full))

    # ================================================================
    # PART 3: Breakeven SL at 1R
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 3: Breakeven SL — BBSqueeze+MTF (1h)")
    logger.info("─" * 72)
    logger.info("")

    for be_r in [0.5, 1.0, 1.5]:
        engine_be = BacktestEngine(
            initial_capital=10_000, max_hold_bars=72,
            breakeven_at_r=be_r,
        )
        wf_be = WalkForwardAnalyzer(train_ratio=0.7, n_windows=5, engine=engine_be)

        rpt = wf_be.run(baseline_factory, df_1h, htf_df=df_4h)
        r_full = engine_be.run(baseline_factory(), df_1h, htf_df=df_4h)

        label = f"BE@{be_r}R"
        logger.info("  --- %s ---", label)
        print_wf(label, rpt)
        print_result(label + " Full", r_full)
        logger.info("")
        all_results.append((label, rpt, r_full))

    # ================================================================
    # PART 4: Combined — Best body + session + breakeven
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 4: Combined Filters — BBSqueeze+MTF (1h)")
    logger.info("─" * 72)
    logger.info("")

    combo_configs = [
        ("Body50+BE1R", 0.5, (), 1.0),
        ("Body40+BE1R", 0.4, (), 1.0),
        ("Body50+NoAsianLate", 0.5, (0, 1, 2, 3, 4, 5), 0.0),
        ("Body40+NoAsianLate+BE1R", 0.4, (0, 1, 2, 3, 4, 5), 1.0),
        ("Body50+NoAsianLate+BE1R", 0.5, (0, 1, 2, 3, 4, 5), 1.0),
    ]

    for label, body_r, blocked, be_r in combo_configs:
        engine_combo = BacktestEngine(
            initial_capital=10_000, max_hold_bars=72,
            breakeven_at_r=be_r,
        )
        wf_combo = WalkForwardAnalyzer(
            train_ratio=0.7, n_windows=5, engine=engine_combo,
        )

        def combo_factory(br=body_r, bh=blocked):
            base = BBSqueezeV3Strategy(
                squeeze_lookback=100, squeeze_pctile=25.0,
                vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
                require_trend=False, cooldown_bars=6,
                min_body_ratio=br,
                blocked_hours=bh,
            )
            return MultiTimeframeFilter(base)

        rpt = wf_combo.run(combo_factory, df_1h, htf_df=df_4h)
        r_full = engine_combo.run(combo_factory(), df_1h, htf_df=df_4h)

        logger.info("  --- %s ---", label)
        print_wf(label, rpt)
        print_result(label + " Full", r_full)
        logger.info("")
        all_results.append((label, rpt, r_full))

    # ================================================================
    # PART 5: Multi-Strategy Portfolio (BBSqueeze + Fib)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 5: Multi-Strategy Portfolio — BBSqueeze(1h) + Fib(4h)")
    logger.info("─" * 72)
    logger.info("")

    # Run Fib WF separately on 4h
    engine_4h = BacktestEngine(initial_capital=10_000, max_hold_bars=24)
    wf_4h = WalkForwardAnalyzer(train_ratio=0.7, n_windows=5, engine=engine_4h)

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
    r_fib_full = engine_4h.run(fib_factory(), df_4h)

    logger.info("  --- Fib 4h standalone ---")
    print_wf("Fib_4h", wf_fib)
    print_result("Fib_4h Full", r_fib_full)
    logger.info("")

    # Portfolio: equal-weight combination of OOS returns
    # Simulated by averaging window-level returns
    logger.info("  --- Portfolio: 50%% BBSqueeze(1h) + 50%% Fib(4h) ---")
    if wf_baseline.windows and wf_fib.windows:
        n_windows = min(len(wf_baseline.windows), len(wf_fib.windows))
        port_oos_returns = []
        port_profitable = 0
        port_trades = 0
        for i in range(n_windows):
            bb_ret = wf_baseline.windows[i].out_of_sample.total_return
            fib_ret = wf_fib.windows[i].out_of_sample.total_return if i < len(wf_fib.windows) else 0.0
            combined_ret = 0.5 * bb_ret + 0.5 * fib_ret
            port_oos_returns.append(combined_ret)
            bb_tr = wf_baseline.windows[i].out_of_sample.total_trades
            fib_tr = wf_fib.windows[i].out_of_sample.total_trades if i < len(wf_fib.windows) else 0
            port_trades += bb_tr + fib_tr
            logger.info(
                "  W%d: BB OOS %+6.2f%% + Fib OOS %+6.2f%% → Port %+6.2f%%",
                i + 1, bb_ret, fib_ret, combined_ret,
            )
            if combined_ret > 0:
                port_profitable += 1

        compounded = 1.0
        for r in port_oos_returns:
            compounded *= (1 + r / 100)
        port_total = (compounded - 1) * 100

        port_robustness = port_profitable / n_windows if n_windows > 0 else 0
        logger.info(
            "  Portfolio OOS: %+.2f%% | Robustness: %.0f%% (%d/%d) | Trades: %d",
            port_total, port_robustness * 100, port_profitable, n_windows, port_trades,
        )

        # Full backtest portfolio
        port_full_ret = 0.5 * r_baseline.total_return + 0.5 * r_fib_full.total_return
        port_full_dd = max(r_baseline.max_drawdown, r_fib_full.max_drawdown) * 0.7  # Diversification benefit estimate
        logger.info(
            "  Portfolio Full: %+.2f%% | Est DD: %.1f%%",
            port_full_ret, port_full_dd,
        )
    logger.info("")

    # ================================================================
    # PART 6: Extended WF (7 windows) on baseline for statistical robustness
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 6: Extended WF (7 windows) — BBSqueeze+MTF (1h)")
    logger.info("─" * 72)
    logger.info("")

    engine_7w = BacktestEngine(initial_capital=10_000, max_hold_bars=72)
    wf_7w = WalkForwardAnalyzer(train_ratio=0.7, n_windows=7, engine=engine_7w)
    wf_7w_rpt = wf_7w.run(baseline_factory, df_1h, htf_df=df_4h)
    r_7w_full = engine_7w.run(baseline_factory(), df_1h, htf_df=df_4h)

    logger.info("  --- Baseline 7-window WF ---")
    print_wf("Baseline_7w", wf_7w_rpt)
    print_result("Baseline_7w Full", r_7w_full)
    logger.info("")
    all_results.append(("Baseline_7w", wf_7w_rpt, r_7w_full))

    # ================================================================
    # FINAL RANKING
    # ================================================================
    logger.info("=" * 72)
    logger.info("  PHASE 6 — FINAL RANKING")
    logger.info("=" * 72)
    logger.info("")

    # Sort by robustness first, then by OOS return
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

    # Recommendation
    if all_results:
        best_name, best_rpt, best_full = all_results[0]
        logger.info("  Recommendation: %s", best_name)
        logger.info("    - WF Robustness: %.0f%% (%d/%d windows profitable)",
                    best_rpt.robustness_score * 100,
                    best_rpt.oos_profitable_windows, best_rpt.total_windows)
        logger.info("    - OOS Return: %+.2f%%", best_rpt.oos_total_return)
        logger.info("    - Full Return: %+.2f%% | DD %.1f%% | PF %.2f",
                    best_full.total_return, best_full.max_drawdown,
                    best_full.profit_factor)

        # Compare vs baseline
        baseline_oos = wf_baseline.oos_total_return
        logger.info("")
        logger.info("  vs Baseline: OOS %+.2f%% → %+.2f%% (%+.2f%%)",
                    baseline_oos, best_rpt.oos_total_return,
                    best_rpt.oos_total_return - baseline_oos)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 6 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

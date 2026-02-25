#!/usr/bin/env python3
"""Phase 5: BBSqueeze v2 — Trailing Stop + ADX Filter + 30m Timeframe.

Tests three key improvements over Phase 4:
1. Trailing stop (ATR-based) instead of fixed TP — lets winners run
2. ADX trend strength filter — avoids choppy/ranging markets
3. 30m timeframe — 17k bars for better WF statistical significance

All variants undergo Walk-Forward analysis (5 windows) to validate OOS.
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
from src.strategy.bb_squeeze_v2 import BBSqueezeV2Strategy
from src.strategy.bb_squeeze_breakout import BBSqueezeBreakoutStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.monitoring.logger import setup_logging

logger = setup_logging("phase5")

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
    logger.info("  PHASE 5 — BBSqueeze v2: Trailing Stop + ADX + 30m")
    logger.info("=" * 72)
    logger.info("")

    # Load data
    df_30m = load_data("30m")
    df_1h = load_data("1h")
    df_4h = load_data("4h")
    logger.info("30m data: %d bars (%s ~ %s)",
                len(df_30m), df_30m.index[0].date(), df_30m.index[-1].date())
    logger.info("1h data: %d bars (%s ~ %s)",
                len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("4h data: %d bars (%s ~ %s)",
                len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    # Verify ADX column exists
    assert "ADX_14" in df_1h.columns, "ADX_14 not found in indicators"
    logger.info("ADX_14 range: %.1f ~ %.1f (mean %.1f)",
                df_1h["ADX_14"].min(), df_1h["ADX_14"].max(), df_1h["ADX_14"].mean())
    logger.info("")

    # ================================================================
    # PART 0: Phase 4 baseline reproduction (BBSqueeze+MTF Conservative)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 0: Phase 4 Baseline — BBSqueeze+MTF Conservative (1h)")
    logger.info("─" * 72)

    engine_1h = BacktestEngine(initial_capital=10_000, max_hold_bars=72)

    def baseline_factory():
        base = BBSqueezeBreakoutStrategy(
            squeeze_lookback=100, squeeze_pctile=25.0,
            vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
            require_trend=False, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    wf_1h = WalkForwardAnalyzer(train_ratio=0.7, n_windows=5, engine=engine_1h)
    wf_baseline = wf_1h.run(baseline_factory, df_1h, htf_df=df_4h)
    print_wf("Baseline_1h", wf_baseline)

    r_baseline = engine_1h.run(baseline_factory(), df_1h, htf_df=df_4h)
    print_result("Baseline Full", r_baseline)
    logger.info("")

    # ================================================================
    # PART 1: Trailing Stop variants (1h)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 1: Trailing Stop — BBSqueeze+MTF (1h)")
    logger.info("─" * 72)
    logger.info("")

    trailing_results = []

    for trail_mult in [2.5, 3.0, 3.5, 4.0]:
        engine_trail = BacktestEngine(
            initial_capital=10_000, max_hold_bars=96,
            trailing_atr_mult=trail_mult,
        )

        def trail_factory(tm=trail_mult):
            base = BBSqueezeV2Strategy(
                squeeze_lookback=100, squeeze_pctile=25.0,
                vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
                require_trend=False, cooldown_bars=6,
                disable_tp=True,  # Trailing stop handles exit
            )
            return MultiTimeframeFilter(base)

        wf_trail = WalkForwardAnalyzer(
            train_ratio=0.7, n_windows=5, engine=engine_trail
        )
        rpt = wf_trail.run(trail_factory, df_1h, htf_df=df_4h)
        r_full = engine_trail.run(trail_factory(), df_1h, htf_df=df_4h)

        label = f"Trail_{trail_mult}ATR"
        logger.info("  --- %s ---", label)
        print_wf(label, rpt)
        print_result(label + " Full", r_full)
        logger.info("")

        trailing_results.append((label, rpt, r_full))

    # Also test trailing WITH fixed TP (hybrid: trail catches big moves, TP caps at RR)
    for trail_mult in [3.0, 3.5]:
        engine_hybrid = BacktestEngine(
            initial_capital=10_000, max_hold_bars=96,
            trailing_atr_mult=trail_mult,
        )

        def hybrid_factory(tm=trail_mult):
            base = BBSqueezeV2Strategy(
                squeeze_lookback=100, squeeze_pctile=25.0,
                vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=3.0,
                require_trend=False, cooldown_bars=6,
                disable_tp=False,  # Keep TP as backstop
            )
            return MultiTimeframeFilter(base)

        wf_hybrid = WalkForwardAnalyzer(
            train_ratio=0.7, n_windows=5, engine=engine_hybrid
        )
        rpt = wf_hybrid.run(hybrid_factory, df_1h, htf_df=df_4h)
        r_full = engine_hybrid.run(hybrid_factory(), df_1h, htf_df=df_4h)

        label = f"Hybrid_Trail{trail_mult}_TP3.0"
        logger.info("  --- %s ---", label)
        print_wf(label, rpt)
        print_result(label + " Full", r_full)
        logger.info("")

        trailing_results.append((label, rpt, r_full))

    # ================================================================
    # PART 2: ADX Filter variants (1h)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 2: ADX Filter — BBSqueeze+MTF (1h)")
    logger.info("─" * 72)
    logger.info("")

    adx_results = []

    for adx_thresh in [15.0, 20.0, 25.0, 30.0]:
        def adx_factory(at=adx_thresh):
            base = BBSqueezeV2Strategy(
                squeeze_lookback=100, squeeze_pctile=25.0,
                vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
                require_trend=False, cooldown_bars=6,
                adx_threshold=at,
            )
            return MultiTimeframeFilter(base)

        rpt = wf_1h.run(adx_factory, df_1h, htf_df=df_4h)
        r_full = engine_1h.run(adx_factory(), df_1h, htf_df=df_4h)

        label = f"ADX>{adx_thresh:.0f}"
        logger.info("  --- %s ---", label)
        print_wf(label, rpt)
        print_result(label + " Full", r_full)
        logger.info("")

        adx_results.append((label, rpt, r_full))

    # ================================================================
    # PART 3: Combined — best trailing + best ADX (1h)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 3: Combined — Trail + ADX + MTF (1h)")
    logger.info("─" * 72)
    logger.info("")

    combined_results = []

    for trail_mult, adx_thresh in [(3.0, 20.0), (3.0, 25.0), (3.5, 20.0), (3.5, 25.0)]:
        engine_combo = BacktestEngine(
            initial_capital=10_000, max_hold_bars=96,
            trailing_atr_mult=trail_mult,
        )

        def combo_factory(tm=trail_mult, at=adx_thresh):
            base = BBSqueezeV2Strategy(
                squeeze_lookback=100, squeeze_pctile=25.0,
                vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
                require_trend=False, cooldown_bars=6,
                adx_threshold=at, disable_tp=True,
            )
            return MultiTimeframeFilter(base)

        wf_combo = WalkForwardAnalyzer(
            train_ratio=0.7, n_windows=5, engine=engine_combo,
        )
        rpt = wf_combo.run(combo_factory, df_1h, htf_df=df_4h)
        r_full = engine_combo.run(combo_factory(), df_1h, htf_df=df_4h)

        label = f"Trail{trail_mult}+ADX{adx_thresh:.0f}"
        logger.info("  --- %s ---", label)
        print_wf(label, rpt)
        print_result(label + " Full", r_full)
        logger.info("")

        combined_results.append((label, rpt, r_full))

    # ================================================================
    # PART 4: 30m Timeframe — BBSqueeze+MTF (more data)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 4: 30m Timeframe — BBSqueeze+MTF (17k bars)")
    logger.info("─" * 72)
    logger.info("")

    engine_30m = BacktestEngine(initial_capital=10_000, max_hold_bars=144)
    wf_30m = WalkForwardAnalyzer(train_ratio=0.7, n_windows=5, engine=engine_30m)

    tf30m_results = []

    # 30m baseline (fixed TP)
    def bb30m_factory():
        base = BBSqueezeV2Strategy(
            squeeze_lookback=200, squeeze_pctile=25.0,
            vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
            require_trend=False, cooldown_bars=12,
        )
        return MultiTimeframeFilter(base)

    rpt = wf_30m.run(bb30m_factory, df_30m, htf_df=df_4h)
    r_full = engine_30m.run(bb30m_factory(), df_30m, htf_df=df_4h)
    logger.info("  --- 30m Baseline (FixedTP RR=2.0) ---")
    print_wf("30m_Baseline", rpt)
    print_result("30m_Baseline Full", r_full)
    logger.info("")
    tf30m_results.append(("30m_Baseline", rpt, r_full))

    # 30m trailing stop
    for trail_mult in [3.0, 3.5, 4.0]:
        engine_30m_trail = BacktestEngine(
            initial_capital=10_000, max_hold_bars=144,
            trailing_atr_mult=trail_mult,
        )

        def bb30m_trail_factory(tm=trail_mult):
            base = BBSqueezeV2Strategy(
                squeeze_lookback=200, squeeze_pctile=25.0,
                vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
                require_trend=False, cooldown_bars=12,
                disable_tp=True,
            )
            return MultiTimeframeFilter(base)

        wf_30m_trail = WalkForwardAnalyzer(
            train_ratio=0.7, n_windows=5, engine=engine_30m_trail,
        )
        rpt = wf_30m_trail.run(bb30m_trail_factory, df_30m, htf_df=df_4h)
        r_full = engine_30m_trail.run(bb30m_trail_factory(), df_30m, htf_df=df_4h)

        label = f"30m_Trail_{trail_mult}ATR"
        logger.info("  --- %s ---", label)
        print_wf(label, rpt)
        print_result(label + " Full", r_full)
        logger.info("")
        tf30m_results.append((label, rpt, r_full))

    # 30m with ADX
    def bb30m_adx_factory():
        base = BBSqueezeV2Strategy(
            squeeze_lookback=200, squeeze_pctile=25.0,
            vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
            require_trend=False, cooldown_bars=12,
            adx_threshold=20.0,
        )
        return MultiTimeframeFilter(base)

    rpt = wf_30m.run(bb30m_adx_factory, df_30m, htf_df=df_4h)
    r_full = engine_30m.run(bb30m_adx_factory(), df_30m, htf_df=df_4h)
    logger.info("  --- 30m ADX>20 ---")
    print_wf("30m_ADX20", rpt)
    print_result("30m_ADX20 Full", r_full)
    logger.info("")
    tf30m_results.append(("30m_ADX20", rpt, r_full))

    # ================================================================
    # FINAL RANKING
    # ================================================================
    logger.info("=" * 72)
    logger.info("  PHASE 5 — FINAL RANKING")
    logger.info("=" * 72)
    logger.info("")

    all_results = (
        [("Baseline_1h", wf_baseline, r_baseline)]
        + trailing_results
        + adx_results
        + combined_results
        + tf30m_results
    )

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

    # Top 3
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

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 5 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 23 — New 15m Components & Weight Optimization.

Phase 22 conclusion: 88% robustness ceiling confirmed across all approaches.
  Cross-TF 1hRSI/1hDC/15mRSI 33/33/34 = 88% rob, +18.81% OOS.
  Multi-asset (ETH/SOL) failed. BTC-only is optimal.
  All 4-component portfolios: 88% rob but LOWER OOS return.

Gap identified: 15m VWAP and 15m CCI were NEVER tested as components.
Only 1h VWAP (55% rob) and 1h CCI (66% rob) were tested at their timeframe.
15m RSI is the best single strategy (77% rob) — 15m captures different signals.

Phase 23 tests:
  PART 1: 15m VWAP standalone (9w WF)
  PART 2: 15m CCI standalone (9w WF)
  PART 3: 4-component portfolios with 15m VWAP / 15m CCI
  PART 4: Exhaustive 3-component weight grid (10% increments)
  PART 5: Summary
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pandas_ta as ta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.basic import BasicIndicators
from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import (
    CrossTFComponent,
    CrossTFReport,
    WalkForwardAnalyzer,
)
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.cci_mean_reversion import CCIMeanReversionStrategy
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase23")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase23.log", mode="w")
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

for name in [
    "src.backtest.engine",
    "src.strategy.mtf_filter",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

wf_logger = logging.getLogger("src.backtest.walk_forward")
wf_logger.setLevel(logging.INFO)
wf_logger.handlers.clear()
wf_logger.addHandler(fh)
wf_logger.addHandler(sh)


# ─── Data ─────────────────────────────────────────────────────────

def load_data(timeframe: str) -> pd.DataFrame:
    """Load and add indicators to OHLCV data."""
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add CCI indicator to DataFrame."""
    col = f"CCI_{period}"
    if col not in df.columns:
        df[col] = ta.cci(df["high"], df["low"], df["close"], length=period)
    return df


# ─── Strategy Factories ──────────────────────────────────────────

def make_rsi_1h() -> MultiTimeframeFilter:
    """1h RSI MR — Phase 14 optimal params."""
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_dc_1h() -> MultiTimeframeFilter:
    """1h Donchian — Phase 14 optimal params."""
    base = DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_rsi_15m() -> MultiTimeframeFilter:
    """15m RSI MR — Phase 16 optimal params."""
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
    )
    return MultiTimeframeFilter(base)


def make_vwap_15m() -> MultiTimeframeFilter:
    """15m VWAP MR — adapted from 1h best config.

    1h best: vwap_period=24, band_mult=2.0, rsi=35, sl=2.0, cool=4
    15m adaptation: vwap_period=96 (24*4=24h), cool=16 (4h in 15m bars)
    """
    base = VWAPMeanReversionStrategy(
        vwap_period=96,
        band_mult=2.0,
        rsi_threshold=35.0,
        atr_sl_mult=2.0,
        tp_to_vwap_pct=0.8,
        cooldown_bars=16,
    )
    return MultiTimeframeFilter(base)


def make_cci_15m() -> MultiTimeframeFilter:
    """15m CCI MR — adapted from 1h best config.

    1h best: period=20, threshold=200, sl=2.0, tp=3.0, cool=6
    15m adaptation: period=20 (CCI is scale-independent), cool=12
    """
    base = CCIMeanReversionStrategy(
        cci_period=20,
        oversold_level=200,
        overbought_level=200,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=12,
    )
    return MultiTimeframeFilter(base)


def log_cross_tf(name: str, report: CrossTFReport) -> None:
    """Log cross-TF report summary."""
    logger.info(
        "  %s: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        name, report.oos_total_return,
        int(report.robustness_score * 100),
        report.oos_profitable_windows, report.total_windows,
        report.total_trades,
    )


def log_cross_tf_detail(name: str, report: CrossTFReport) -> None:
    """Log cross-TF with per-window breakdown."""
    for w in report.windows:
        parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
        marker = "+" if w.weighted_return > 0 else "-"
        logger.info(
            "    W%d [%s ~ %s]: %s -> %+.2f%% %s",
            w.window_id, w.test_start, w.test_end,
            " | ".join(parts), w.weighted_return, marker,
        )
    log_cross_tf(name, report)


def log_wf_report(
    name: str,
    report,
    engine: BacktestEngine,
    factory,
    df: pd.DataFrame,
    htf_df: pd.DataFrame | None = None,
) -> None:
    """Log WF window-by-window details and full-period backtest."""
    for w in report.windows:
        oos = w.out_of_sample
        is_ = w.in_sample
        logger.info(
            "  W%d: IS %+6.2f%% (WR %d%%, %d tr) | OOS %+6.2f%% (WR %d%%, %d tr)",
            w.window_id,
            is_.total_return, int(is_.win_rate * 100), is_.total_trades,
            oos.total_return, int(oos.win_rate * 100), oos.total_trades,
        )
    logger.info(
        "  OOS: %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        report.oos_total_return,
        int(report.robustness_score * 100),
        report.oos_profitable_windows, report.total_windows,
        report.oos_total_trades,
    )
    full = engine.run(factory(), df, htf_df=htf_df)
    logger.info(
        "  %s Full %+8.2f%% | Shp %5.2f | DD %5.1f%% | WR %5.1f%% | %4d tr | PF %5.2f",
        name, full.total_return, full.sharpe_ratio, full.max_drawdown,
        full.win_rate * 100, full.total_trades, full.profit_factor,
    )


def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 23 — New 15m Components & Weight Optimization")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Baseline: 1hRSI/1hDC/15mRSI 33/33/34 = 88%% rob, +18.81%% OOS")
    logger.info("  Goal: Test untried 15m VWAP & 15m CCI as components")
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")

    # Add CCI to 15m data
    df_15m = add_cci(df_15m, period=20)

    logger.info("1h data:  %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("15m data: %d bars (%s ~ %s)", len(df_15m),
                df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("4h data:  %d bars (%s ~ %s)", len(df_4h),
                df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")
    wf = WalkForwardAnalyzer(n_windows=9)

    # ═════════════════════════════════════════════════════════════
    #   PART 0: Baseline Reproduction
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 0: Baseline — 1hRSI/1hDC/15mRSI 33/33/34 (9w)")
    logger.info("-" * 72)
    logger.info("")

    baseline_report = wf.run_cross_tf([
        CrossTFComponent(
            strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.33, label="1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.33, label="1hDC",
        ),
        CrossTFComponent(
            strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
            engine=engine_15m, weight=0.34, label="15mRSI",
        ),
    ])
    log_cross_tf_detail("Baseline 33/33/34", baseline_report)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 1: 15m VWAP Standalone (9w)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: 15m VWAP Standalone (9w)")
    logger.info("-" * 72)
    logger.info("  1h VWAP: 55%% rob, +11.57%% OOS — does 15m capture different signals?")
    logger.info("")

    wf_15m = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
    vwap_15m_report = wf_15m.run(make_vwap_15m, df_15m, htf_df=df_4h)
    log_wf_report("VWAP_15m+MTF", vwap_15m_report, engine_15m, make_vwap_15m, df_15m, df_4h)
    logger.info("")

    # Also test with shorter VWAP period
    def make_vwap_15m_short() -> MultiTimeframeFilter:
        base = VWAPMeanReversionStrategy(
            vwap_period=48,  # 12h at 15m
            band_mult=2.0, rsi_threshold=35.0,
            atr_sl_mult=2.0, tp_to_vwap_pct=0.8,
            cooldown_bars=12,
        )
        return MultiTimeframeFilter(base)

    logger.info("  --- 15m VWAP period=48 (12h) ---")
    vwap_15m_short_report = wf_15m.run(make_vwap_15m_short, df_15m, htf_df=df_4h)
    log_wf_report("VWAP_15m_48+MTF", vwap_15m_short_report, engine_15m, make_vwap_15m_short, df_15m, df_4h)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: 15m CCI Standalone (9w)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: 15m CCI Standalone (9w)")
    logger.info("-" * 72)
    logger.info("  1h CCI: 66%% rob, +13.48%% OOS — does 15m CCI work differently?")
    logger.info("")

    cci_15m_report = wf_15m.run(make_cci_15m, df_15m, htf_df=df_4h)
    log_wf_report("CCI_15m+MTF", cci_15m_report, engine_15m, make_cci_15m, df_15m, df_4h)
    logger.info("")

    # Test with higher threshold
    def make_cci_15m_tight() -> MultiTimeframeFilter:
        base = CCIMeanReversionStrategy(
            cci_period=20, oversold_level=250, overbought_level=250,
            atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
        )
        return MultiTimeframeFilter(base)

    logger.info("  --- 15m CCI threshold=250 ---")
    cci_15m_tight_report = wf_15m.run(make_cci_15m_tight, df_15m, htf_df=df_4h)
    log_wf_report("CCI_15m_250+MTF", cci_15m_tight_report, engine_15m, make_cci_15m_tight, df_15m, df_4h)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: 4-Component Portfolios (9w)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: 4-Component Portfolios (9w)")
    logger.info("-" * 72)
    logger.info("  Phase 18: 1h CCI as 4th → 88%% rob but lower return.")
    logger.info("  Testing 15m VWAP and 15m CCI as 4th components.")
    logger.info("")

    # Choose best VWAP/CCI configs from Parts 1-2
    best_vwap_factory = make_vwap_15m
    best_cci_factory = make_cci_15m

    # 4-comp with 15m VWAP
    four_comp_configs = [
        # (name, w_rsi1h, w_dc1h, w_rsi15m, w_4th, 4th_factory, 4th_label)
        ("RSI/DC/RSI15/VWAP15 25/25/25/25", 0.25, 0.25, 0.25, 0.25, best_vwap_factory, "15mVWAP"),
        ("RSI/DC/RSI15/VWAP15 30/20/30/20", 0.30, 0.20, 0.30, 0.20, best_vwap_factory, "15mVWAP"),
        ("RSI/DC/RSI15/CCI15 25/25/25/25", 0.25, 0.25, 0.25, 0.25, best_cci_factory, "15mCCI"),
        ("RSI/DC/RSI15/CCI15 30/20/30/20", 0.30, 0.20, 0.30, 0.20, best_cci_factory, "15mCCI"),
    ]

    four_comp_results: dict[str, CrossTFReport] = {}
    for name, w1, w2, w3, w4, factory_4th, label_4th in four_comp_configs:
        logger.info("  --- %s ---", name)
        components = [
            CrossTFComponent(
                strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w1, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w2, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w3, label="15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=factory_4th, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w4, label=label_4th,
            ),
        ]
        report = wf.run_cross_tf(components)
        four_comp_results[name] = report
        log_cross_tf_detail(name, report)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Exhaustive 3-Component Weight Grid (10% steps)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 4: Exhaustive 3-Component Weight Grid (10%% steps)")
    logger.info("-" * 72)
    logger.info("  Phase 17 tested 7 combos. Now testing ALL valid 10%% combinations.")
    logger.info("")

    grid_results: dict[str, CrossTFReport] = {}

    # Generate all weight combos: w_rsi + w_dc + w_15m = 100, min 10%
    step = 10
    for w_rsi in range(10, 81, step):
        for w_dc in range(10, 81 - w_rsi + 1, step):
            w_15m = 100 - w_rsi - w_dc
            if w_15m < 10:
                continue

            name = f"{w_rsi}/{w_dc}/{w_15m}"
            components = [
                CrossTFComponent(
                    strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_rsi / 100, label="1hRSI",
                ),
                CrossTFComponent(
                    strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_dc / 100, label="1hDC",
                ),
                CrossTFComponent(
                    strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
                    engine=engine_15m, weight=w_15m / 100, label="15mRSI",
                ),
            ]
            report = wf.run_cross_tf(components)
            grid_results[name] = report

    # Sort by robustness then OOS return
    sorted_grid = sorted(
        grid_results.items(),
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    )

    logger.info("  %-15s %8s %6s %6s", "Weights", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 40)
    for name, report in sorted_grid:
        marker = " ***" if report.robustness_score > baseline_report.robustness_score else ""
        if not marker and report.robustness_score == baseline_report.robustness_score:
            if report.oos_total_return > baseline_report.oos_total_return:
                marker = " +"
        logger.info(
            "  %-15s %+7.2f%% %5d%% %6d%s",
            name, report.oos_total_return,
            int(report.robustness_score * 100),
            report.total_trades, marker,
        )
    logger.info("")

    # Show top 5 details
    logger.info("  Top 5 weight combinations (detailed):")
    for name, report in sorted_grid[:5]:
        logger.info("")
        log_cross_tf_detail(f"  {name}", report)

    # ═════════════════════════════════════════════════════════════
    #   PART 5: Summary
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 23 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    logger.info("  15m Standalone Results (9w):")
    logger.info("  %-25s %8s %6s %6s", "Strategy", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 50)
    standalone_results = [
        ("15m VWAP (period=96)", vwap_15m_report),
        ("15m VWAP (period=48)", vwap_15m_short_report),
        ("15m CCI (threshold=200)", cci_15m_report),
        ("15m CCI (threshold=250)", cci_15m_tight_report),
    ]
    for name, rpt in standalone_results:
        logger.info(
            "  %-25s %+7.2f%% %5d%% %6d",
            name, rpt.oos_total_return,
            int(rpt.robustness_score * 100), rpt.oos_total_trades,
        )

    logger.info("")
    logger.info("  4-Component Portfolio Results (9w):")
    logger.info("  %-40s %8s %6s %6s", "Portfolio", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 60)
    for name, report in sorted(
        four_comp_results.items(),
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    ):
        logger.info(
            "  %-40s %+7.2f%% %5d%% %6d",
            name, report.oos_total_return,
            int(report.robustness_score * 100), report.total_trades,
        )

    logger.info("")
    logger.info("  Best 3-Component Weight vs Baseline:")
    if sorted_grid:
        best_name, best_report = sorted_grid[0]
        logger.info("    Baseline 33/33/34: OOS %+.2f%%, Rob %d%%",
                     baseline_report.oos_total_return,
                     int(baseline_report.robustness_score * 100))
        logger.info("    Best %s:     OOS %+.2f%%, Rob %d%%",
                     best_name, best_report.oos_total_return,
                     int(best_report.robustness_score * 100))

        delta = best_report.oos_total_return - baseline_report.oos_total_return
        if best_report.robustness_score > baseline_report.robustness_score:
            logger.info("    ==> NEW CEILING BROKEN! Weight %s beats 88%%", best_name)
        elif delta > 0:
            logger.info("    ==> Same robustness, better return by %+.2f%%", delta)
        else:
            logger.info("    ==> 33/33/34 remains optimal")

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 23 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

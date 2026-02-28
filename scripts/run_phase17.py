#!/usr/bin/env python3
"""Phase 17 — Formal Cross-Timeframe Portfolio Walk-Forward Validation.

Phase 16 findings (manual analysis, approximate window alignment):
  - 1hRSI/1hDC/15mRSI 33/33/34 = 89% robustness, +19.61% OOS
  - Only W2 (Nov 20-Dec 2) is universally negative
  - Cross-TF diversification breaks the 77% ceiling from 1h-only portfolios

Phase 17 goal: FORMAL validation using date-aligned WF windows.
  - Uses new run_cross_tf() method for exact date alignment across timeframes
  - Tests multiple weight combinations
  - Compares with single-TF and 1h-only baselines
  - All results reproducible via the formal engine

Strategy components (optimal parameters from Phase 14-16):
  - 1h RSI MR: oversold=35, overbought=65, sl=2.0, tp=3.0, cool=6
  - 1h Donchian: period=24, sl=2.0, rr=2.0, vol=0.8, cool=6
  - 15m RSI MR: oversold=35, overbought=65, sl=2.0, tp=3.0, cool=12, hold=96

Data: 1h (~8760 bars), 15m (~35040 bars), 4h MTF filter (~2190 bars).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

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

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase17")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase17.log", mode="w")
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

# Suppress noisy sub-loggers
for name in [
    "src.backtest.engine",
    "src.strategy.mtf_filter",
    "src.backtest.walk_forward",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

# Redirect WF cross-TF logs to our logger
wf_logger = logging.getLogger("src.backtest.walk_forward")
wf_logger.setLevel(logging.INFO)
wf_logger.handlers.clear()
wf_logger.addHandler(fh)
wf_logger.addHandler(sh)


def load_data(timeframe: str) -> pd.DataFrame:
    """Load and add indicators to OHLCV data."""
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


# ─── Strategy Factories ──────────────────────────────────────────

def make_rsi_1h() -> MultiTimeframeFilter:
    """1h RSI MR with optimal Phase 14 params."""
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_dc_1h() -> MultiTimeframeFilter:
    """1h Donchian with optimal Phase 14 params."""
    base = DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_rsi_15m() -> MultiTimeframeFilter:
    """15m RSI MR mid-config from Phase 16."""
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
    )
    return MultiTimeframeFilter(base)


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


def log_cross_tf(name: str, report: CrossTFReport) -> None:
    """Log cross-TF report summary."""
    logger.info(
        "  %s: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        name, report.oos_total_return,
        int(report.robustness_score * 100),
        report.oos_profitable_windows, report.total_windows,
        report.total_trades,
    )


def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 17 — Cross-Timeframe Portfolio Formal Validation")
    logger.info("=" * 72)
    logger.info("")

    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")

    logger.info("1h data:  %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("15m data: %d bars (%s ~ %s)", len(df_15m),
                df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("4h data:  %d bars (%s ~ %s)", len(df_4h),
                df_4h.index[0].date(), df_4h.index[-1].date())

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")

    # ═════════════════════════════════════════════════════════════
    #   PART 1: Individual Strategy Baselines (9w)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("-" * 72)
    logger.info("  PART 1: Individual Strategy Baselines (9w)")
    logger.info("-" * 72)
    logger.info("")

    # 1h RSI
    logger.info("  --- 1h RSI_35_65+MTF (9w) ---")
    wf9 = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
    rsi_1h_report = wf9.run(make_rsi_1h, df_1h, htf_df=df_4h)
    log_wf_report("RSI_1h+MTF", rsi_1h_report, engine_1h, make_rsi_1h, df_1h, df_4h)
    logger.info("")

    # 1h DC
    logger.info("  --- 1h DC_24+MTF (9w) ---")
    wf9_dc = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
    dc_1h_report = wf9_dc.run(make_dc_1h, df_1h, htf_df=df_4h)
    log_wf_report("DC_1h+MTF", dc_1h_report, engine_1h, make_dc_1h, df_1h, df_4h)
    logger.info("")

    # 15m RSI mid
    logger.info("  --- 15m RSI_35_65_mid+MTF (9w, cool=12, hold=96) ---")
    wf9_15m = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
    rsi_15m_report = wf9_15m.run(make_rsi_15m, df_15m, htf_df=df_4h)
    log_wf_report("RSI_15m_mid+MTF", rsi_15m_report, engine_15m, make_rsi_15m, df_15m, df_4h)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: 1h-Only Portfolio Baseline via Cross-TF Engine
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("-" * 72)
    logger.info("  PART 2: 1h-Only Portfolio Baseline (9w, date-aligned)")
    logger.info("-" * 72)
    logger.info("  (Validation: should match ~77%% robustness from Phase 14)")
    logger.info("")

    wf_cross = WalkForwardAnalyzer(n_windows=9)

    logger.info("  --- RSI+DC 50/50 (1h only) ---")
    baseline_report = wf_cross.run_cross_tf([
        CrossTFComponent(
            strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.5, label="1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.5, label="1hDC",
        ),
    ])
    log_cross_tf("1h RSI+DC 50/50", baseline_report)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Cross-TF Portfolio Weight Sweep (9w)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("-" * 72)
    logger.info("  PART 3: Cross-TF Portfolio Weight Sweep (9w)")
    logger.info("-" * 72)
    logger.info("  Phase 16 manual best: 33/33/34 = 89%% rob, +19.61%% OOS")
    logger.info("")

    # Weight combos: (1hRSI, 1hDC, 15mRSI)
    weight_combos = [
        ("33/33/34 (equal)", 0.33, 0.33, 0.34),
        ("50/0/50 (1hRSI+15mRSI)", 0.50, 0.00, 0.50),
        ("40/20/40", 0.40, 0.20, 0.40),
        ("30/30/40", 0.30, 0.30, 0.40),
        ("0/50/50 (1hDC+15mRSI)", 0.00, 0.50, 0.50),
        ("25/25/50", 0.25, 0.25, 0.50),
        ("20/40/40", 0.20, 0.40, 0.40),
    ]

    cross_tf_results: dict[str, CrossTFReport] = {}

    for combo_name, w_rsi_1h, w_dc_1h, w_rsi_15m in weight_combos:
        label = f"1hRSI/1hDC/15mRSI {combo_name}"
        logger.info("  --- %s ---", label)

        components: list[CrossTFComponent] = []
        if w_rsi_1h > 0:
            components.append(CrossTFComponent(
                strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_rsi_1h, label="1hRSI",
            ))
        if w_dc_1h > 0:
            components.append(CrossTFComponent(
                strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_dc_1h, label="1hDC",
            ))
        if w_rsi_15m > 0:
            components.append(CrossTFComponent(
                strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w_rsi_15m, label="15mRSI",
            ))

        report = wf_cross.run_cross_tf(components)
        cross_tf_results[label] = report
        log_cross_tf(combo_name, report)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Summary
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 17 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    logger.info("  Individual Baselines (9w, bar-based WF):")
    logger.info("  %-30s %8s %6s %6s", "Strategy", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 55)
    for name, report in [
        ("RSI_1h+MTF", rsi_1h_report),
        ("DC_1h+MTF", dc_1h_report),
        ("RSI_15m_mid+MTF", rsi_15m_report),
    ]:
        logger.info(
            "  %-30s %+7.2f%% %5d%% %6d",
            name, report.oos_total_return,
            int(report.robustness_score * 100), report.oos_total_trades,
        )

    logger.info("")
    logger.info("  1h-Only Baseline (date-aligned):")
    logger.info("  %-30s %+7.2f%% %5d%% %6d  (Phase 14: 77%%, +20.27%%)",
                "RSI+DC 50/50",
                baseline_report.oos_total_return,
                int(baseline_report.robustness_score * 100),
                baseline_report.total_trades)

    logger.info("")
    logger.info("  Cross-TF Portfolio (date-aligned, 9w):")
    logger.info("  %-40s %8s %6s %6s", "Weights", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 65)

    best_report = None
    best_label = ""

    for label, report in sorted(
        cross_tf_results.items(),
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    ):
        marker = " ***" if report.robustness_score >= 0.85 else ""
        logger.info(
            "  %-40s %+7.2f%% %5d%% %6d%s",
            label, report.oos_total_return,
            int(report.robustness_score * 100),
            report.total_trades, marker,
        )
        if best_report is None or (
            report.robustness_score, report.oos_total_return
        ) > (best_report.robustness_score, best_report.oos_total_return):
            best_report = report
            best_label = label

    if best_report:
        logger.info("")
        logger.info("  BEST: %s", best_label)
        logger.info("    OOS: %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
                     best_report.oos_total_return,
                     int(best_report.robustness_score * 100),
                     best_report.oos_profitable_windows,
                     best_report.total_windows,
                     best_report.total_trades)
        logger.info("")
        logger.info("  Per-window breakdown:")
        for w in best_report.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("    W%d [%s ~ %s]: %s -> %+.2f%% %s",
                         w.window_id, w.test_start, w.test_end,
                         " | ".join(parts), w.weighted_return, marker)

    logger.info("")
    logger.info("  Comparison with Phase 16 manual analysis:")
    logger.info("    Phase 16 (approximate): 33/33/34 = 89%% rob, +19.61%% OOS")
    if "1hRSI/1hDC/15mRSI 33/33/34 (equal)" in cross_tf_results:
        r = cross_tf_results["1hRSI/1hDC/15mRSI 33/33/34 (equal)"]
        logger.info(
            "    Phase 17 (formal):      33/33/34 = %d%%%% rob, %+.2f%%%% OOS",
            int(r.robustness_score * 100), r.oos_total_return,
        )

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 17 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

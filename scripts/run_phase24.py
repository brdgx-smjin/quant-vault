#!/usr/bin/env python3
"""Phase 24 — 15m Donchian & Component Replacement Tests.

Phase 23 conclusion: 88% robustness ceiling confirmed. 13/28 weight combos = 88%.
  Best: 20/50/30 = +20.83% OOS. Production: 33/33/34 = +18.81% OOS.
  15m CCI standalone = 66% rob, 15m VWAP = 55% rob.
  4-comp portfolios: 88% rob but LOWER OOS return.

Gap identified: 15m Donchian was NEVER formally tested with WF.
  Docstring says "DC does NOT work on 15m" but no WF data exists.
  Also: 15m CCI was tested as 4th component but never as REPLACEMENT
  for 15m RSI in the 3-component portfolio.

Phase 24 tests:
  PART 1: 15m Donchian standalone (9w WF, multiple periods)
  PART 2: Component replacement — 15m CCI instead of 15m RSI (3-comp)
  PART 3: Component replacement — 15m DC instead of 15m RSI (3-comp)
  PART 4: Donchian-heavy portfolios — 1h DC + 15m DC + 1h RSI
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
    WalkForwardReport,
)
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.cci_mean_reversion import CCIMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase24")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase24.log", mode="w")
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


def make_dc_15m(entry_period: int = 96, cooldown: int = 12) -> MultiTimeframeFilter:
    """15m Donchian — adapted from 1h best config.

    1h best: period=24 (24h), sl=2.0, rr=2.0, vol=0.8, cool=6
    15m adaptation: period=96 (24*4=24h), cool=12 (3h in 15m bars)
    """
    base = DonchianTrendStrategy(
        entry_period=entry_period,
        atr_sl_mult=2.0,
        rr_ratio=2.0,
        vol_mult=0.8,
        cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


def make_cci_15m() -> MultiTimeframeFilter:
    """15m CCI MR — Phase 23 config."""
    base = CCIMeanReversionStrategy(
        cci_period=20,
        oversold_level=200,
        overbought_level=200,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=12,
    )
    return MultiTimeframeFilter(base)


# ─── Logging Helpers ──────────────────────────────────────────────

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
    report: WalkForwardReport,
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
    logger.info("  PHASE 24 — 15m Donchian & Component Replacement Tests")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Baseline: 1hRSI/1hDC/15mRSI 33/33/34 = 88%% rob, +18.81%% OOS")
    logger.info("  Gap: 15m DC never WF-tested. 15m CCI only as 4th comp, not replacement.")
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
    wf_15m = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)

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
    #   PART 1: 15m Donchian Standalone (9w)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: 15m Donchian Standalone (9w)")
    logger.info("-" * 72)
    logger.info("  1h DC: 55%% rob standalone. 15m DC: NEVER formally tested.")
    logger.info("  Testing multiple entry_period configurations.")
    logger.info("")

    dc_15m_configs = [
        ("DC_15m_p96_c12", 96, 12),    # 24h lookback, 3h cooldown
        ("DC_15m_p48_c12", 48, 12),    # 12h lookback, 3h cooldown
        ("DC_15m_p192_c24", 192, 24),  # 48h lookback, 6h cooldown
        ("DC_15m_p96_c24", 96, 24),    # 24h lookback, 6h cooldown
    ]

    dc_15m_results: dict[str, WalkForwardReport] = {}

    for name, period, cooldown in dc_15m_configs:
        logger.info("  --- %s (period=%d, cooldown=%d) ---", name, period, cooldown)
        factory = lambda p=period, c=cooldown: make_dc_15m(p, c)
        report = wf_15m.run(factory, df_15m, htf_df=df_4h)
        dc_15m_results[name] = report
        log_wf_report(name, report, engine_15m, factory, df_15m, df_4h)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Component Replacement — 15m CCI replaces 15m RSI
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: Replace 15m RSI with 15m CCI (3-component)")
    logger.info("-" * 72)
    logger.info("  Phase 23: 15m CCI standalone = 66%% rob, +7.85%% OOS.")
    logger.info("  Testing as REPLACEMENT for 15m RSI, not as 4th component.")
    logger.info("")

    # Weight grid for 1hRSI / 1hDC / 15mCCI
    cci_replace_results: dict[str, CrossTFReport] = {}
    weight_combos = [
        (33, 33, 34),  # Equal
        (30, 40, 30),  # DC-heavy
        (40, 30, 30),  # RSI-heavy
        (20, 50, 30),  # Best weights from Phase 23 baseline
        (30, 30, 40),  # CCI-heavy
    ]

    for w_rsi, w_dc, w_cci in weight_combos:
        name = f"RSI/DC/CCI15 {w_rsi}/{w_dc}/{w_cci}"
        logger.info("  --- %s ---", name)
        report = wf.run_cross_tf([
            CrossTFComponent(
                strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_rsi / 100, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_dc / 100, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=make_cci_15m, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w_cci / 100, label="15mCCI",
            ),
        ])
        cci_replace_results[name] = report
        log_cross_tf_detail(name, report)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Component Replacement — 15m DC replaces 15m RSI
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: Replace 15m RSI with 15m DC (3-component)")
    logger.info("-" * 72)
    logger.info("  Testing best 15m DC config from Part 1 as RSI replacement.")
    logger.info("")

    # Find best 15m DC from Part 1
    best_dc_name = max(
        dc_15m_results,
        key=lambda k: (dc_15m_results[k].robustness_score, dc_15m_results[k].oos_total_return),
    )
    best_dc_report = dc_15m_results[best_dc_name]
    logger.info(
        "  Best 15m DC: %s (Rob %d%%, OOS %+.2f%%)",
        best_dc_name, int(best_dc_report.robustness_score * 100),
        best_dc_report.oos_total_return,
    )
    logger.info("")

    # Extract period/cooldown from best config name
    best_dc_config = next(
        (p, c) for n, p, c in dc_15m_configs if n == best_dc_name
    )
    best_dc_period, best_dc_cooldown = best_dc_config

    dc_replace_results: dict[str, CrossTFReport] = {}

    for w_rsi, w_dc, w_dc15 in weight_combos:
        name = f"RSI/DC/DC15 {w_rsi}/{w_dc}/{w_dc15}"
        logger.info("  --- %s ---", name)
        factory_dc15 = lambda p=best_dc_period, c=best_dc_cooldown: make_dc_15m(p, c)
        report = wf.run_cross_tf([
            CrossTFComponent(
                strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_rsi / 100, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_dc / 100, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=factory_dc15, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w_dc15 / 100, label="15mDC",
            ),
        ])
        dc_replace_results[name] = report
        log_cross_tf_detail(name, report)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Donchian-heavy Portfolios
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 4: Donchian-heavy Portfolios (1hDC + 15mDC + 1hRSI)")
    logger.info("-" * 72)
    logger.info("  DC is the strongest diversifier. Testing DC on both TFs.")
    logger.info("")

    dc_heavy_results: dict[str, CrossTFReport] = {}
    dc_heavy_combos = [
        (34, 33, 33),  # RSI-balanced
        (20, 50, 30),  # DC-heavy
        (30, 40, 30),  # Mild DC-heavy
        (40, 30, 30),  # RSI-heavy
    ]

    for w_rsi, w_dc1h, w_dc15 in dc_heavy_combos:
        name = f"RSI/DC1h/DC15m {w_rsi}/{w_dc1h}/{w_dc15}"
        logger.info("  --- %s ---", name)
        factory_dc15 = lambda p=best_dc_period, c=best_dc_cooldown: make_dc_15m(p, c)
        report = wf.run_cross_tf([
            CrossTFComponent(
                strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_rsi / 100, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_dc1h / 100, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=factory_dc15, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w_dc15 / 100, label="15mDC",
            ),
        ])
        dc_heavy_results[name] = report
        log_cross_tf_detail(name, report)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 5: Summary
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 24 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    # Part 1: 15m DC standalone
    logger.info("  15m Donchian Standalone Results (9w):")
    logger.info("  %-25s %8s %6s %6s", "Strategy", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 50)
    for name in sorted(
        dc_15m_results,
        key=lambda k: (dc_15m_results[k].robustness_score, dc_15m_results[k].oos_total_return),
        reverse=True,
    ):
        rpt = dc_15m_results[name]
        logger.info(
            "  %-25s %+7.2f%% %5d%% %6d",
            name, rpt.oos_total_return,
            int(rpt.robustness_score * 100), rpt.oos_total_trades,
        )

    logger.info("")
    logger.info("  Reference: 15m RSI = 77%% rob, +17.50%% OOS")
    logger.info("  Reference: 1h DC  = 55%% rob, +26.50%% OOS")
    logger.info("")

    # Part 2: CCI replacement
    logger.info("  CCI Replacement Portfolios (1hRSI / 1hDC / 15mCCI):")
    logger.info("  %-35s %8s %6s %6s", "Portfolio", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 60)
    for name, report in sorted(
        cci_replace_results.items(),
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    ):
        marker = " ***" if report.robustness_score > baseline_report.robustness_score else ""
        logger.info(
            "  %-35s %+7.2f%% %5d%% %6d%s",
            name, report.oos_total_return,
            int(report.robustness_score * 100),
            report.total_trades, marker,
        )

    logger.info("")

    # Part 3: DC replacement
    logger.info("  DC Replacement Portfolios (1hRSI / 1hDC / 15mDC):")
    logger.info("  %-35s %8s %6s %6s", "Portfolio", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 60)
    for name, report in sorted(
        dc_replace_results.items(),
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    ):
        marker = " ***" if report.robustness_score > baseline_report.robustness_score else ""
        logger.info(
            "  %-35s %+7.2f%% %5d%% %6d%s",
            name, report.oos_total_return,
            int(report.robustness_score * 100),
            report.total_trades, marker,
        )

    logger.info("")

    # Part 4: DC-heavy
    logger.info("  Donchian-heavy Portfolios (1hRSI / 1hDC / 15mDC):")
    logger.info("  %-35s %8s %6s %6s", "Portfolio", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 60)
    for name, report in sorted(
        dc_heavy_results.items(),
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    ):
        marker = " ***" if report.robustness_score > baseline_report.robustness_score else ""
        logger.info(
            "  %-35s %+7.2f%% %5d%% %6d%s",
            name, report.oos_total_return,
            int(report.robustness_score * 100),
            report.total_trades, marker,
        )

    logger.info("")

    # Final comparison
    all_results: dict[str, CrossTFReport] = {}
    all_results["Baseline 33/33/34"] = baseline_report
    all_results.update(cci_replace_results)
    all_results.update(dc_replace_results)
    all_results.update(dc_heavy_results)

    best_name = max(
        all_results,
        key=lambda k: (all_results[k].robustness_score, all_results[k].oos_total_return),
    )
    best_report = all_results[best_name]

    logger.info("  FINAL COMPARISON:")
    logger.info("    Baseline: 1hRSI/1hDC/15mRSI 33/33/34 = OOS %+.2f%%, Rob %d%%",
                 baseline_report.oos_total_return,
                 int(baseline_report.robustness_score * 100))
    logger.info("    Best:     %s = OOS %+.2f%%, Rob %d%%",
                 best_name, best_report.oos_total_return,
                 int(best_report.robustness_score * 100))

    if best_report.robustness_score > baseline_report.robustness_score:
        logger.info("    ==> NEW CEILING BROKEN!")
    elif best_name != "Baseline 33/33/34":
        delta = best_report.oos_total_return - baseline_report.oos_total_return
        logger.info("    ==> Delta: %+.2f%% (same robustness)", delta)
    else:
        logger.info("    ==> Baseline remains best")

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 24 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

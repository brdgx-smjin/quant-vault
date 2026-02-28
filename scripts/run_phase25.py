#!/usr/bin/env python3
"""Phase 25 — Williams %R Mean Reversion & Cross-TF Portfolio Tests.

Phase 24 conclusion: 88% robustness ceiling confirmed. Baseline remains best.
  1hRSI/1hDC/15mRSI 33/33/34 = 88% rob, +18.81% OOS.
  All component replacements tested (15m DC, 15m CCI) — inferior.

Gap identified: Williams %R was NEVER tested. It's a faster oscillator than RSI
  that measures close relative to high-low range. Similar mean-reversion logic
  but may capture different entry timings.

Phase 25 tests:
  PART 0: Baseline reproduction (sanity check)
  PART 1: Williams %R standalone (1h, 9w) — multiple threshold configs
  PART 2: Williams %R standalone (15m, 9w) — multiple configs
  PART 3: Cross-TF replacement — Williams %R replaces RSI in portfolio
  PART 4: Cross-TF 4th component — Williams %R added to 3-component portfolio
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
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase25")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase25.log", mode="w")
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


def add_willr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Williams %R indicator to DataFrame."""
    col = f"WILLR_{period}"
    if col not in df.columns:
        df[col] = ta.willr(df["high"], df["low"], df["close"], length=period)
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


def make_willr_1h(
    period: int = 14,
    oversold: float = 80.0,
    overbought: float = 80.0,
    cooldown: int = 6,
) -> MultiTimeframeFilter:
    """1h Williams %R MR."""
    base = WilliamsRMeanReversionStrategy(
        willr_period=period,
        oversold_level=oversold,
        overbought_level=overbought,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


def make_willr_15m(
    period: int = 14,
    oversold: float = 80.0,
    overbought: float = 80.0,
    cooldown: int = 12,
) -> MultiTimeframeFilter:
    """15m Williams %R MR."""
    base = WilliamsRMeanReversionStrategy(
        willr_period=period,
        oversold_level=oversold,
        overbought_level=overbought,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=cooldown,
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
    logger.info("  PHASE 25 — Williams %%R Mean Reversion Tests")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Baseline: 1hRSI/1hDC/15mRSI 33/33/34 = 88%% rob, +18.81%% OOS")
    logger.info("  Gap: Williams %%R never tested. Faster oscillator than RSI.")
    logger.info("  %%R range: -100 to 0. Oversold < -80, Overbought > -20.")
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")

    # Add Williams %R to data (multiple periods)
    for period in [14, 21, 7]:
        df_1h = add_willr(df_1h, period)
        df_15m = add_willr(df_15m, period)

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
    #   PART 1: Williams %R Standalone — 1h (9w)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: Williams %%R Standalone — 1h (9w)")
    logger.info("-" * 72)
    logger.info("  Testing multiple period/threshold configs.")
    logger.info("  Reference: 1h RSI = 66%% rob, +13.29%% OOS")
    logger.info("")

    willr_1h_configs = [
        ("WillR_1h_p14_t80", 14, 80, 80, 6),
        ("WillR_1h_p14_t85", 14, 85, 85, 6),
        ("WillR_1h_p14_t90", 14, 90, 90, 6),
        ("WillR_1h_p21_t80", 21, 80, 80, 6),
        ("WillR_1h_p7_t80", 7, 80, 80, 6),
        ("WillR_1h_p14_t80_c8", 14, 80, 80, 8),
    ]

    willr_1h_results: dict[str, WalkForwardReport] = {}

    for name, period, os_level, ob_level, cool in willr_1h_configs:
        logger.info("  --- %s (period=%d, thresholds=%d/%d, cool=%d) ---",
                     name, period, os_level, ob_level, cool)
        factory = lambda p=period, o=os_level, b=ob_level, c=cool: make_willr_1h(p, o, b, c)
        report = wf.run(factory, df_1h, htf_df=df_4h)
        willr_1h_results[name] = report
        log_wf_report(name, report, engine_1h, factory, df_1h, df_4h)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Williams %R Standalone — 15m (9w)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: Williams %%R Standalone — 15m (9w)")
    logger.info("-" * 72)
    logger.info("  Testing multiple configs. Reference: 15m RSI = 77%% rob, +17.50%% OOS")
    logger.info("")

    willr_15m_configs = [
        ("WillR_15m_p14_t80", 14, 80, 80, 12),
        ("WillR_15m_p14_t85", 14, 85, 85, 12),
        ("WillR_15m_p14_t90", 14, 90, 90, 12),
        ("WillR_15m_p21_t80", 21, 80, 80, 12),
        ("WillR_15m_p7_t80", 7, 80, 80, 12),
        ("WillR_15m_p14_t80_c16", 14, 80, 80, 16),
    ]

    willr_15m_results: dict[str, WalkForwardReport] = {}

    for name, period, os_level, ob_level, cool in willr_15m_configs:
        logger.info("  --- %s (period=%d, thresholds=%d/%d, cool=%d) ---",
                     name, period, os_level, ob_level, cool)
        factory = lambda p=period, o=os_level, b=ob_level, c=cool: make_willr_15m(p, o, b, c)
        report = wf_15m.run(factory, df_15m, htf_df=df_4h)
        willr_15m_results[name] = report
        log_wf_report(name, report, engine_15m, factory, df_15m, df_4h)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Cross-TF — Williams %R replaces RSI
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: Cross-TF — Williams %%R replaces RSI")
    logger.info("-" * 72)
    logger.info("")

    # Find best 1h and 15m Williams %R configs
    best_1h_name = max(
        willr_1h_results,
        key=lambda k: (willr_1h_results[k].robustness_score, willr_1h_results[k].oos_total_return),
    )
    best_1h = willr_1h_results[best_1h_name]

    best_15m_name = max(
        willr_15m_results,
        key=lambda k: (willr_15m_results[k].robustness_score, willr_15m_results[k].oos_total_return),
    )
    best_15m = willr_15m_results[best_15m_name]

    logger.info("  Best 1h Williams %%R:  %s (Rob %d%%, OOS %+.2f%%)",
                 best_1h_name, int(best_1h.robustness_score * 100), best_1h.oos_total_return)
    logger.info("  Best 15m Williams %%R: %s (Rob %d%%, OOS %+.2f%%)",
                 best_15m_name, int(best_15m.robustness_score * 100), best_15m.oos_total_return)
    logger.info("")

    # Extract params from best config names
    best_1h_cfg = next(
        (p, o, b, c) for n, p, o, b, c in willr_1h_configs if n == best_1h_name
    )
    best_15m_cfg = next(
        (p, o, b, c) for n, p, o, b, c in willr_15m_configs if n == best_15m_name
    )

    # Test 3a: Replace BOTH RSI with Williams %R
    logger.info("  --- 3a: 1hWillR/1hDC/15mWillR (replace both RSI) ---")
    replace_both_results: dict[str, CrossTFReport] = {}
    weight_combos = [
        (33, 33, 34),
        (30, 40, 30),
        (20, 50, 30),
        (40, 30, 30),
    ]

    for w_wr, w_dc, w_wr15 in weight_combos:
        name = f"WR/DC/WR15 {w_wr}/{w_dc}/{w_wr15}"
        logger.info("  --- %s ---", name)
        factory_1h = lambda p=best_1h_cfg[0], o=best_1h_cfg[1], b=best_1h_cfg[2], c=best_1h_cfg[3]: make_willr_1h(p, o, b, c)
        factory_15m = lambda p=best_15m_cfg[0], o=best_15m_cfg[1], b=best_15m_cfg[2], c=best_15m_cfg[3]: make_willr_15m(p, o, b, c)
        report = wf.run_cross_tf([
            CrossTFComponent(
                strategy_factory=factory_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_wr / 100, label="1hWillR",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_dc / 100, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=factory_15m, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w_wr15 / 100, label="15mWillR",
            ),
        ])
        replace_both_results[name] = report
        log_cross_tf_detail(name, report)
        logger.info("")

    # Test 3b: Replace only 15m RSI with Williams %R (keep 1h RSI)
    logger.info("  --- 3b: 1hRSI/1hDC/15mWillR (replace 15m RSI only) ---")
    replace_15m_results: dict[str, CrossTFReport] = {}

    for w_rsi, w_dc, w_wr15 in weight_combos:
        name = f"RSI/DC/WR15 {w_rsi}/{w_dc}/{w_wr15}"
        logger.info("  --- %s ---", name)
        factory_15m = lambda p=best_15m_cfg[0], o=best_15m_cfg[1], b=best_15m_cfg[2], c=best_15m_cfg[3]: make_willr_15m(p, o, b, c)
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
                strategy_factory=factory_15m, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w_wr15 / 100, label="15mWillR",
            ),
        ])
        replace_15m_results[name] = report
        log_cross_tf_detail(name, report)
        logger.info("")

    # Test 3c: Replace only 1h RSI with Williams %R (keep 15m RSI)
    logger.info("  --- 3c: 1hWillR/1hDC/15mRSI (replace 1h RSI only) ---")
    replace_1h_results: dict[str, CrossTFReport] = {}

    for w_wr, w_dc, w_rsi15 in weight_combos:
        name = f"WR/DC/RSI15 {w_wr}/{w_dc}/{w_rsi15}"
        logger.info("  --- %s ---", name)
        factory_1h = lambda p=best_1h_cfg[0], o=best_1h_cfg[1], b=best_1h_cfg[2], c=best_1h_cfg[3]: make_willr_1h(p, o, b, c)
        report = wf.run_cross_tf([
            CrossTFComponent(
                strategy_factory=factory_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_wr / 100, label="1hWillR",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_dc / 100, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w_rsi15 / 100, label="15mRSI",
            ),
        ])
        replace_1h_results[name] = report
        log_cross_tf_detail(name, report)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Cross-TF — Williams %R as 4th component
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 4: Cross-TF — Williams %%R as 4th component")
    logger.info("-" * 72)
    logger.info("  Phase 18/23 showed 4th component = 88%% rob but LOWER return.")
    logger.info("  Testing if Williams %%R as 4th does better than CCI as 4th.")
    logger.info("")

    four_comp_results: dict[str, CrossTFReport] = {}
    four_comp_combos = [
        (25, 25, 25, 25),  # Equal
        (25, 30, 30, 15),  # Williams %R minor
        (30, 25, 25, 20),  # RSI-heavy
        (20, 35, 25, 20),  # DC-heavy
    ]

    for w_rsi, w_dc, w_rsi15, w_wr in four_comp_combos:
        # Test with best 1h Williams %R
        name = f"RSI/DC/RSI15/WR1h {w_rsi}/{w_dc}/{w_rsi15}/{w_wr}"
        logger.info("  --- %s ---", name)
        factory_1h_wr = lambda p=best_1h_cfg[0], o=best_1h_cfg[1], b=best_1h_cfg[2], c=best_1h_cfg[3]: make_willr_1h(p, o, b, c)
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
                strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w_rsi15 / 100, label="15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=factory_1h_wr, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_wr / 100, label="1hWillR",
            ),
        ])
        four_comp_results[name] = report
        log_cross_tf_detail(name, report)
        logger.info("")

    # Also test with 15m Williams %R as 4th
    for w_rsi, w_dc, w_rsi15, w_wr in four_comp_combos:
        name = f"RSI/DC/RSI15/WR15 {w_rsi}/{w_dc}/{w_rsi15}/{w_wr}"
        logger.info("  --- %s ---", name)
        factory_15m_wr = lambda p=best_15m_cfg[0], o=best_15m_cfg[1], b=best_15m_cfg[2], c=best_15m_cfg[3]: make_willr_15m(p, o, b, c)
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
                strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w_rsi15 / 100, label="15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=factory_15m_wr, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w_wr / 100, label="15mWillR",
            ),
        ])
        four_comp_results[name] = report
        log_cross_tf_detail(name, report)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 5: Summary
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 25 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    # Part 1: 1h Williams %R standalone
    logger.info("  1h Williams %%R Standalone Results (9w):")
    logger.info("  %-30s %8s %6s %6s", "Strategy", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 55)
    for name in sorted(
        willr_1h_results,
        key=lambda k: (willr_1h_results[k].robustness_score, willr_1h_results[k].oos_total_return),
        reverse=True,
    ):
        rpt = willr_1h_results[name]
        logger.info(
            "  %-30s %+7.2f%% %5d%% %6d",
            name, rpt.oos_total_return,
            int(rpt.robustness_score * 100), rpt.oos_total_trades,
        )
    logger.info("  Reference: 1h RSI = 66%% rob, +13.29%% OOS")
    logger.info("")

    # Part 2: 15m Williams %R standalone
    logger.info("  15m Williams %%R Standalone Results (9w):")
    logger.info("  %-30s %8s %6s %6s", "Strategy", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 55)
    for name in sorted(
        willr_15m_results,
        key=lambda k: (willr_15m_results[k].robustness_score, willr_15m_results[k].oos_total_return),
        reverse=True,
    ):
        rpt = willr_15m_results[name]
        logger.info(
            "  %-30s %+7.2f%% %5d%% %6d",
            name, rpt.oos_total_return,
            int(rpt.robustness_score * 100), rpt.oos_total_trades,
        )
    logger.info("  Reference: 15m RSI = 77%% rob, +17.50%% OOS")
    logger.info("")

    # Part 3: Replacement portfolios
    all_replacements: dict[str, CrossTFReport] = {}
    all_replacements.update(replace_both_results)
    all_replacements.update(replace_15m_results)
    all_replacements.update(replace_1h_results)

    logger.info("  Replacement Portfolio Results (3-component):")
    logger.info("  %-40s %8s %6s %6s", "Portfolio", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 65)
    for name in sorted(
        all_replacements,
        key=lambda k: (all_replacements[k].robustness_score, all_replacements[k].oos_total_return),
        reverse=True,
    ):
        rpt = all_replacements[name]
        marker = " ***" if rpt.robustness_score > baseline_report.robustness_score else ""
        logger.info(
            "  %-40s %+7.2f%% %5d%% %6d%s",
            name, rpt.oos_total_return,
            int(rpt.robustness_score * 100),
            rpt.total_trades, marker,
        )
    logger.info("  Reference: Baseline 33/33/34 = %+.2f%%, %d%% rob",
                 baseline_report.oos_total_return,
                 int(baseline_report.robustness_score * 100))
    logger.info("")

    # Part 4: 4-component portfolios
    logger.info("  4-Component Portfolio Results:")
    logger.info("  %-45s %8s %6s %6s", "Portfolio", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 70)
    for name in sorted(
        four_comp_results,
        key=lambda k: (four_comp_results[k].robustness_score, four_comp_results[k].oos_total_return),
        reverse=True,
    ):
        rpt = four_comp_results[name]
        marker = " ***" if rpt.robustness_score > baseline_report.robustness_score else ""
        logger.info(
            "  %-45s %+7.2f%% %5d%% %6d%s",
            name, rpt.oos_total_return,
            int(rpt.robustness_score * 100),
            rpt.total_trades, marker,
        )
    logger.info("")

    # Final comparison
    all_results: dict[str, CrossTFReport] = {"Baseline 33/33/34": baseline_report}
    all_results.update(all_replacements)
    all_results.update(four_comp_results)

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
        if best_report.robustness_score == baseline_report.robustness_score:
            logger.info("    ==> SAME robustness, delta: %+.2f%%", delta)
        else:
            logger.info("    ==> Lower robustness, delta: %+.2f%%", delta)
    else:
        logger.info("    ==> Baseline remains best")

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 25 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

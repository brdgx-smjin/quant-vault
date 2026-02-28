#!/usr/bin/env python3
"""Phase 37 — Production Strategy Re-Validation.

Re-runs the current production 4-comp Cross-TF portfolio with latest data
to confirm results match Phase 25 benchmarks.

Expected results (Phase 25):
  4-comp 1hRSI/1hDC/15mRSI/1hWillR 15/50/10/25 = 88% rob, +23.98% OOS
  3-comp 1hRSI/1hDC/15mRSI 33/33/34 = 88% rob, +18.81% OOS (fallback)
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
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase37")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase37.log", mode="w")
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

for name in ["src.backtest.engine", "src.strategy.mtf_filter"]:
    logging.getLogger(name).setLevel(logging.WARNING)

wf_logger = logging.getLogger("src.backtest.walk_forward")
wf_logger.setLevel(logging.INFO)
wf_logger.handlers.clear()
wf_logger.addHandler(fh)
wf_logger.addHandler(sh)


# ─── Data ─────────────────────────────────────────────────────────

def load_data(timeframe: str) -> pd.DataFrame:
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


def add_willr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    col = f"WILLR_{period}"
    if col not in df.columns:
        df[col] = ta.willr(df["high"], df["low"], df["close"], length=period)
    return df


# ─── Strategy Factories ──────────────────────────────────────────

def make_rsi_1h() -> MultiTimeframeFilter:
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_dc_1h() -> MultiTimeframeFilter:
    base = DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_rsi_15m() -> MultiTimeframeFilter:
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
    )
    return MultiTimeframeFilter(base)


def make_willr_1h() -> MultiTimeframeFilter:
    base = WilliamsRMeanReversionStrategy(
        willr_period=14,
        oversold_level=90,
        overbought_level=90,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


# ─── Logging Helpers ──────────────────────────────────────────────

def log_cross_tf_detail(name: str, report: CrossTFReport) -> None:
    for w in report.windows:
        parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
        marker = "+" if w.weighted_return > 0 else "-"
        logger.info(
            "    W%d [%s ~ %s]: %s -> %+.2f%% %s",
            w.window_id, w.test_start, w.test_end,
            " | ".join(parts), w.weighted_return, marker,
        )
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
    logger.info("  PHASE 37 — Production Strategy Re-Validation")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Expected: 4-comp 15/50/10/25 = 88%% rob, +23.98%% OOS")
    logger.info("  Expected: 3-comp 33/33/34 = 88%% rob, +18.81%% OOS")
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")
    df_1h = add_willr(df_1h, 14)

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
    #   TEST 1: 4-comp Production Config (15/50/10/25)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  TEST 1: 4-comp 1hRSI/1hDC/15mRSI/1hWillR (15/50/10/25)")
    logger.info("-" * 72)
    logger.info("")

    prod_report = wf.run_cross_tf([
        CrossTFComponent(
            strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.15, label="1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.50, label="1hDC",
        ),
        CrossTFComponent(
            strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
            engine=engine_15m, weight=0.10, label="15mRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_willr_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.25, label="1hWillR",
        ),
    ])
    log_cross_tf_detail("4-comp 15/50/10/25", prod_report)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   TEST 2: 3-comp Fallback Config (33/33/34)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  TEST 2: 3-comp 1hRSI/1hDC/15mRSI (33/33/34)")
    logger.info("-" * 72)
    logger.info("")

    fallback_report = wf.run_cross_tf([
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
    log_cross_tf_detail("3-comp 33/33/34", fallback_report)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   TEST 3: Individual Components (standalone)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  TEST 3: Individual Component Standalone (9w)")
    logger.info("-" * 72)
    logger.info("")

    components = [
        ("1h RSI MR", make_rsi_1h, engine_1h, df_1h, df_4h),
        ("1h Donchian", make_dc_1h, engine_1h, df_1h, df_4h),
        ("15m RSI MR", make_rsi_15m, engine_15m, df_15m, df_4h),
        ("1h WillR MR", make_willr_1h, engine_1h, df_1h, df_4h),
    ]

    for name, factory, engine, df, htf_df in components:
        wf_eng = WalkForwardAnalyzer(n_windows=9, engine=engine)
        report = wf_eng.run(factory, df, htf_df=htf_df)
        logger.info(
            "  %s: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
            name, report.oos_total_return,
            int(report.robustness_score * 100),
            report.oos_profitable_windows, report.total_windows,
            report.oos_total_trades,
        )
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   SUMMARY
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 37 — VALIDATION SUMMARY")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  4-comp 15/50/10/25:")
    logger.info("    OOS Return:  %+.2f%% (expected: +23.98%%)",
                prod_report.oos_total_return)
    logger.info("    Robustness:  %d%% (expected: 88%%)",
                int(prod_report.robustness_score * 100))
    logger.info("    Trades:      %d (expected: ~236)", prod_report.total_trades)
    logger.info("")
    logger.info("  3-comp 33/33/34:")
    logger.info("    OOS Return:  %+.2f%% (expected: +18.81%%)",
                fallback_report.oos_total_return)
    logger.info("    Robustness:  %d%% (expected: 88%%)",
                int(fallback_report.robustness_score * 100))
    logger.info("    Trades:      %d", fallback_report.total_trades)
    logger.info("")

    # Validation checks
    prod_ok = (
        int(prod_report.robustness_score * 100) == 88
        and abs(prod_report.oos_total_return - 23.98) < 5.0  # Allow some data drift
    )
    fallback_ok = (
        int(fallback_report.robustness_score * 100) == 88
        and abs(fallback_report.oos_total_return - 18.81) < 5.0
    )

    if prod_ok and fallback_ok:
        logger.info("  VALIDATION: PASS — Both configs match Phase 25 benchmarks.")
    elif prod_ok:
        logger.info("  VALIDATION: PARTIAL — Production passes, fallback drifted.")
    elif fallback_ok:
        logger.info("  VALIDATION: PARTIAL — Fallback passes, production drifted.")
    else:
        logger.info("  VALIDATION: FAIL — Results diverged from Phase 25.")
        logger.info("  Check data updates or engine changes.")

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 37 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

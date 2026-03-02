#!/usr/bin/env python3
"""Phase 40 — MTF Extreme Override Testing.

Allow extreme oversold/overbought signals to bypass the MTF 4h trend filter.
Different from Phase 38 (directional asymmetry = permanently suppress one
direction). This is CONDITIONAL — only the most extreme indicator values
bypass the filter.

Live observation: RSI=21 at $63K was blocked by MTF BEARISH, missing +11%
rally to $70K.  The override targets exactly these high-conviction entries.

Tests:
  Part 0: Baseline reproduction (4-comp 15/50/10/25)
  Part 1: RSI standalone with oversold override grid (RSI < 20/25/30)
  Part 2: WillR standalone with oversold override grid (WillR < -97/-95/-93)
  Part 3: Symmetric overbought override tests
  Part 4: 4-comp portfolio with best override thresholds
  Part 5: Summary

Baseline: 4-comp 15/50/10/25, 88% rob, +23.98% OOS, 236 trades.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas_ta as ta
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.basic import BasicIndicators
from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import (
    CrossTFComponent,
    WalkForwardAnalyzer,
)
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase40")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase40.log", mode="w")
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

def make_rsi_1h(
    os_rsi: float = 0.0,
    ob_rsi: float = 100.0,
) -> MultiTimeframeFilter:
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(
        base,
        extreme_oversold_rsi=os_rsi,
        extreme_overbought_rsi=ob_rsi,
    )


def make_dc_1h() -> MultiTimeframeFilter:
    """Donchian — NO override. Trend-following respects MTF strictly."""
    base = DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_rsi_15m(
    os_rsi: float = 0.0,
    ob_rsi: float = 100.0,
) -> MultiTimeframeFilter:
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
    )
    return MultiTimeframeFilter(
        base,
        extreme_oversold_rsi=os_rsi,
        extreme_overbought_rsi=ob_rsi,
    )


def make_willr_1h(
    os_willr: float = -100.0,
    ob_willr: float = 0.0,
) -> MultiTimeframeFilter:
    base = WilliamsRMeanReversionStrategy(
        willr_period=14, oversold_level=90.0, overbought_level=90.0,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(
        base,
        extreme_oversold_willr=os_willr,
        extreme_overbought_willr=ob_willr,
    )


def log_window_details(report, logger_fn=logger.info):
    """Log per-window breakdown."""
    for w in report.windows:
        if hasattr(w, "components") and w.components:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger_fn("    W%d: %s -> %+.2f%% %s",
                       w.window_id, " | ".join(parts), w.weighted_return, marker)
        else:
            ret = w.oos_return if hasattr(w, "oos_return") else w.weighted_return
            marker = "+" if ret > 0 else "-"
            logger_fn("    W%d: %+.2f%% %s", w.window_id, ret, marker)


# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 40 — MTF Extreme Override Testing")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Allow extreme oversold/overbought to bypass MTF trend filter.")
    logger.info("  Live evidence: RSI=21 at $63K blocked → missed +11%% rally.")
    logger.info("  Baseline: 4-comp 15/50/10/25, 88%% rob, +23.98%% OOS")
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")
    df_1h = add_willr(df_1h, 14)

    logger.info("  1h data:  %d bars (%s ~ %s)",
                len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("  15m data: %d bars (%s ~ %s)",
                len(df_15m), df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")

    # ═════════════════════════════════════════════════════════════
    #   PART 0: Baseline Reproduction
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 0: Baseline (no override)")
    logger.info("-" * 72)
    logger.info("")

    wf = WalkForwardAnalyzer(n_windows=9)
    baseline_components = [
        CrossTFComponent(
            strategy_factory=make_rsi_1h,
            df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.15, label="1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h,
            df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.50, label="1hDC",
        ),
        CrossTFComponent(
            strategy_factory=make_rsi_15m,
            df=df_15m, htf_df=df_4h,
            engine=engine_15m, weight=0.10, label="15mRSI",
        ),
        CrossTFComponent(
            strategy_factory=lambda: make_willr_1h(),
            df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.25, label="1hWillR",
        ),
    ]

    baseline_report = wf.run_cross_tf(baseline_components)
    baseline_rob = int(baseline_report.robustness_score * 100)
    logger.info("  Baseline 4-comp (15/50/10/25):")
    logger.info("    Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                baseline_rob, baseline_report.oos_total_return,
                baseline_report.total_trades)
    log_window_details(baseline_report)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 1: RSI Standalone with Oversold Override (1h)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: RSI 1h Standalone — Oversold Override Grid")
    logger.info("-" * 72)
    logger.info("  Grid: extreme_oversold_rsi in [20, 25, 30]")
    logger.info("  Reference: RSI standalone w/o override")
    logger.info("")

    # First: RSI standalone without override (reference)
    wf_ref = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
    rsi_ref_report = wf_ref.run(make_rsi_1h, df_1h, htf_df=df_4h)
    rsi_ref_rob = int(rsi_ref_report.robustness_score * 100)
    logger.info("    RSI_1h (no override): Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                rsi_ref_rob, rsi_ref_report.oos_total_return,
                rsi_ref_report.oos_total_trades)
    logger.info("")

    rsi_os_results = []
    os_rsi_grid = [20.0, 25.0, 30.0]

    for os_rsi in os_rsi_grid:
        def factory(t=os_rsi):
            return make_rsi_1h(os_rsi=t)

        wf_r = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
        report = wf_r.run(factory, df_1h, htf_df=df_4h)
        rob = int(report.robustness_score * 100)
        rsi_os_results.append({
            "threshold": os_rsi, "rob": rob,
            "oos": report.oos_total_return,
            "trades": report.oos_total_trades,
        })
        marker = "★" if rob > rsi_ref_rob else ""
        logger.info("    RSI_1h (os_rsi<%.0f): Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                    os_rsi, rob, report.oos_total_return,
                    report.oos_total_trades, marker)

    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 1b: RSI 15m Standalone with Oversold Override
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1b: RSI 15m Standalone — Oversold Override Grid")
    logger.info("-" * 72)
    logger.info("")

    wf_ref15 = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
    rsi15_ref_report = wf_ref15.run(make_rsi_15m, df_15m, htf_df=df_4h)
    rsi15_ref_rob = int(rsi15_ref_report.robustness_score * 100)
    logger.info("    RSI_15m (no override): Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                rsi15_ref_rob, rsi15_ref_report.oos_total_return,
                rsi15_ref_report.oos_total_trades)
    logger.info("")

    rsi15_os_results = []
    for os_rsi in os_rsi_grid:
        def factory(t=os_rsi):
            return make_rsi_15m(os_rsi=t)

        wf_r = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
        report = wf_r.run(factory, df_15m, htf_df=df_4h)
        rob = int(report.robustness_score * 100)
        rsi15_os_results.append({
            "threshold": os_rsi, "rob": rob,
            "oos": report.oos_total_return,
            "trades": report.oos_total_trades,
        })
        marker = "★" if rob > rsi15_ref_rob else ""
        logger.info("    RSI_15m (os_rsi<%.0f): Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                    os_rsi, rob, report.oos_total_return,
                    report.oos_total_trades, marker)

    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: WillR Standalone with Oversold Override (1h)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: WillR 1h Standalone — Oversold Override Grid")
    logger.info("-" * 72)
    logger.info("  Grid: extreme_oversold_willr in [-97, -95, -93]")
    logger.info("")

    wf_wr_ref = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
    willr_ref_report = wf_wr_ref.run(lambda: make_willr_1h(), df_1h, htf_df=df_4h)
    willr_ref_rob = int(willr_ref_report.robustness_score * 100)
    logger.info("    WillR_1h (no override): Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                willr_ref_rob, willr_ref_report.oos_total_return,
                willr_ref_report.oos_total_trades)
    logger.info("")

    willr_os_results = []
    os_willr_grid = [-97.0, -95.0, -93.0]

    for os_willr in os_willr_grid:
        def factory(t=os_willr):
            return make_willr_1h(os_willr=t)

        wf_r = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
        report = wf_r.run(factory, df_1h, htf_df=df_4h)
        rob = int(report.robustness_score * 100)
        willr_os_results.append({
            "threshold": os_willr, "rob": rob,
            "oos": report.oos_total_return,
            "trades": report.oos_total_trades,
        })
        marker = "★" if rob > willr_ref_rob else ""
        logger.info("    WillR_1h (os_willr<%.0f): Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                    os_willr, rob, report.oos_total_return,
                    report.oos_total_trades, marker)

    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Symmetric Overbought Override Tests
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: Overbought Override (SHORT in bullish)")
    logger.info("-" * 72)
    logger.info("")

    ob_rsi_grid = [70.0, 75.0, 80.0]
    rsi_ob_results = []

    for ob_rsi in ob_rsi_grid:
        def factory(t=ob_rsi):
            return make_rsi_1h(ob_rsi=t)

        wf_r = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
        report = wf_r.run(factory, df_1h, htf_df=df_4h)
        rob = int(report.robustness_score * 100)
        rsi_ob_results.append({
            "threshold": ob_rsi, "rob": rob,
            "oos": report.oos_total_return,
            "trades": report.oos_total_trades,
        })
        marker = "★" if rob > rsi_ref_rob else ""
        logger.info("    RSI_1h (ob_rsi>%.0f): Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                    ob_rsi, rob, report.oos_total_return,
                    report.oos_total_trades, marker)

    logger.info("")

    ob_willr_grid = [-10.0, -5.0, -3.0]
    willr_ob_results = []

    for ob_willr in ob_willr_grid:
        def factory(t=ob_willr):
            return make_willr_1h(ob_willr=t)

        wf_r = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
        report = wf_r.run(factory, df_1h, htf_df=df_4h)
        rob = int(report.robustness_score * 100)
        willr_ob_results.append({
            "threshold": ob_willr, "rob": rob,
            "oos": report.oos_total_return,
            "trades": report.oos_total_trades,
        })
        marker = "★" if rob > willr_ref_rob else ""
        logger.info("    WillR_1h (ob_willr>%.0f): Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                    ob_willr, rob, report.oos_total_return,
                    report.oos_total_trades, marker)

    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Portfolio Integration (best thresholds)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 4: 4-comp Portfolio with Override")
    logger.info("-" * 72)
    logger.info("")

    # Collect best oversold thresholds
    all_rsi_os = rsi_os_results + rsi15_os_results
    best_rsi_os = max(all_rsi_os, key=lambda x: (x["rob"], x["oos"]))
    best_willr_os = max(willr_os_results, key=lambda x: (x["rob"], x["oos"]))

    # Collect best overbought thresholds
    best_rsi_ob = max(rsi_ob_results, key=lambda x: (x["rob"], x["oos"]))
    best_willr_ob = max(willr_ob_results, key=lambda x: (x["rob"], x["oos"]))

    logger.info("  Best oversold:  RSI<%.0f (%d%% rob), WillR<%.0f (%d%% rob)",
                best_rsi_os["threshold"], best_rsi_os["rob"],
                best_willr_os["threshold"], best_willr_os["rob"])
    logger.info("  Best overbought: RSI>%.0f (%d%% rob), WillR>%.0f (%d%% rob)",
                best_rsi_ob["threshold"], best_rsi_ob["rob"],
                best_willr_ob["threshold"], best_willr_ob["rob"])
    logger.info("")

    # Test configurations
    portfolio_configs = []

    # 4a: Override only on RSI components (oversold only)
    best_os_t = best_rsi_os["threshold"]
    def config_4a():
        return [
            CrossTFComponent(
                strategy_factory=lambda: make_rsi_1h(os_rsi=best_os_t),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.15, label="1hRSI_OVR",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.50, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=lambda: make_rsi_15m(os_rsi=best_os_t),
                df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=0.10, label="15mRSI_OVR",
            ),
            CrossTFComponent(
                strategy_factory=lambda: make_willr_1h(),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hWillR",
            ),
        ]

    # 4b: Override only on WillR
    best_os_w = best_willr_os["threshold"]
    def config_4b():
        return [
            CrossTFComponent(
                strategy_factory=make_rsi_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.15, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.50, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=make_rsi_15m,
                df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=0.10, label="15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=lambda: make_willr_1h(os_willr=best_os_w),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hWillR_OVR",
            ),
        ]

    # 4c: Override on ALL MR components (RSI + WillR, both oversold)
    def config_4c():
        return [
            CrossTFComponent(
                strategy_factory=lambda: make_rsi_1h(os_rsi=best_os_t),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.15, label="1hRSI_OVR",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.50, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=lambda: make_rsi_15m(os_rsi=best_os_t),
                df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=0.10, label="15mRSI_OVR",
            ),
            CrossTFComponent(
                strategy_factory=lambda: make_willr_1h(os_willr=best_os_w),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hWillR_OVR",
            ),
        ]

    # 4d: Override on ALL MR + both directions (oversold + overbought)
    best_ob_t = best_rsi_ob["threshold"]
    best_ob_w = best_willr_ob["threshold"]
    def config_4d():
        return [
            CrossTFComponent(
                strategy_factory=lambda: make_rsi_1h(os_rsi=best_os_t, ob_rsi=best_ob_t),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.15, label="1hRSI_FULL",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.50, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=lambda: make_rsi_15m(os_rsi=best_os_t, ob_rsi=best_ob_t),
                df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=0.10, label="15mRSI_FULL",
            ),
            CrossTFComponent(
                strategy_factory=lambda: make_willr_1h(os_willr=best_os_w, ob_willr=best_ob_w),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hWillR_FULL",
            ),
        ]

    configs = [
        ("4a: RSI oversold override only", config_4a),
        ("4b: WillR oversold override only", config_4b),
        ("4c: All MR oversold override", config_4c),
        ("4d: All MR both directions", config_4d),
    ]

    portfolio_results = []
    for name, config_fn in configs:
        wf_p = WalkForwardAnalyzer(n_windows=9)
        components = config_fn()
        report = wf_p.run_cross_tf(components)
        rob = int(report.robustness_score * 100)
        portfolio_results.append({
            "name": name, "rob": rob,
            "oos": report.oos_total_return,
            "trades": report.total_trades,
        })
        logger.info("  %s:", name)
        logger.info("    Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    rob, report.oos_total_return, report.total_trades)
        log_window_details(report)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 5: Summary
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 40 SUMMARY")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Baseline: Rob=%d%%, OOS=%+.2f%%", baseline_rob,
                baseline_report.oos_total_return)
    logger.info("")

    logger.info("  RSI 1h oversold override:")
    for r in rsi_os_results:
        delta = r["oos"] - rsi_ref_report.oos_total_return
        logger.info("    RSI<%.0f: Rob=%d%%, OOS=%+.2f%% (delta=%+.2f%%)",
                    r["threshold"], r["rob"], r["oos"], delta)

    logger.info("  RSI 15m oversold override:")
    for r in rsi15_os_results:
        delta = r["oos"] - rsi15_ref_report.oos_total_return
        logger.info("    RSI<%.0f: Rob=%d%%, OOS=%+.2f%% (delta=%+.2f%%)",
                    r["threshold"], r["rob"], r["oos"], delta)

    logger.info("  WillR 1h oversold override:")
    for r in willr_os_results:
        delta = r["oos"] - willr_ref_report.oos_total_return
        logger.info("    WillR<%.0f: Rob=%d%%, OOS=%+.2f%% (delta=%+.2f%%)",
                    r["threshold"], r["rob"], r["oos"], delta)

    logger.info("")
    logger.info("  Portfolio results:")
    for r in portfolio_results:
        delta = r["oos"] - baseline_report.oos_total_return
        status = "IMPROVED" if r["rob"] >= baseline_rob and delta > 0 else \
                 "MAINTAINED" if r["rob"] >= baseline_rob else "DEGRADED"
        logger.info("    %s: Rob=%d%%, OOS=%+.2f%% (delta=%+.2f%%) [%s]",
                    r["name"], r["rob"], r["oos"], delta, status)

    logger.info("")
    best_portfolio = max(portfolio_results, key=lambda x: (x["rob"], x["oos"]))
    logger.info("  BEST: %s → Rob=%d%%, OOS=%+.2f%%",
                best_portfolio["name"], best_portfolio["rob"],
                best_portfolio["oos"])

    if best_portfolio["rob"] >= baseline_rob:
        logger.info("  ✓ Robustness maintained or improved — DEPLOY CANDIDATE")
    else:
        logger.info("  ✗ Robustness degraded — DO NOT DEPLOY")

    logger.info("")
    logger.info("  Phase 40 complete.")


if __name__ == "__main__":
    main()

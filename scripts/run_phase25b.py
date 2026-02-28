#!/usr/bin/env python3
"""Phase 25b — Exhaustive Weight Grid & Parameter Stability for 4-comp WillR.

Phase 25 finding: 4-comp 1hRSI/1hDC/15mRSI/1hWillR achieves 88% rob with
  HIGHER OOS return than baseline. Best: 20/35/25/20 = +22.08% vs +18.81%.
  All 4 tested weight combos hit 88%.

Phase 25b tests:
  PART 1: Exhaustive weight grid (5% increments, sum=100)
  PART 2: Williams %R parameter stability (perturbations of p14_t90)
  PART 3: Summary — identify production-ready 4-comp portfolio
"""

from __future__ import annotations

import logging
import sys
from itertools import product
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

logger = logging.getLogger("phase25b")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase25b.log", mode="w")
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


def make_willr_1h(
    period: int = 14,
    oversold: float = 90.0,
    overbought: float = 90.0,
    cooldown: int = 6,
) -> MultiTimeframeFilter:
    base = WilliamsRMeanReversionStrategy(
        willr_period=period,
        oversold_level=oversold,
        overbought_level=overbought,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


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
    logger.info("  PHASE 25b — Exhaustive Weight Grid & Parameter Stability")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Phase 25 finding: 4-comp RSI/DC/RSI15/WR1h at 88%% rob")
    logger.info("  Best: 20/35/25/20 = +22.08%%, baseline 33/33/34 = +18.81%%")
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")

    # Add all Williams %R periods for stability test
    for period in [14, 10, 21, 7]:
        df_1h = add_willr(df_1h, period)

    logger.info("1h data:  %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("15m data: %d bars (%s ~ %s)", len(df_15m),
                df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")
    wf = WalkForwardAnalyzer(n_windows=9)

    # ═════════════════════════════════════════════════════════════
    #   PART 1: Exhaustive Weight Grid (5% increments)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: Exhaustive Weight Grid (5%% increments)")
    logger.info("-" * 72)
    logger.info("  Components: 1hRSI / 1hDC / 15mRSI / 1hWillR(p14_t90)")
    logger.info("  Constraint: each >= 10%%, sum = 100%%")
    logger.info("")

    # Generate all weight combos with 5% steps, min 10% each, sum=100%
    weight_combos: list[tuple[int, int, int, int]] = []
    for w1 in range(10, 55, 5):
        for w2 in range(10, 55, 5):
            for w3 in range(10, 55, 5):
                w4 = 100 - w1 - w2 - w3
                if 10 <= w4 <= 50:
                    weight_combos.append((w1, w2, w3, w4))

    logger.info("  Total weight combinations: %d", len(weight_combos))
    logger.info("")

    grid_results: dict[str, CrossTFReport] = {}
    n_88 = 0

    for idx, (w_rsi, w_dc, w_rsi15, w_wr) in enumerate(weight_combos):
        name = f"{w_rsi}/{w_dc}/{w_rsi15}/{w_wr}"
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
                strategy_factory=make_willr_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_wr / 100, label="1hWillR",
            ),
        ])
        grid_results[name] = report

        rob_pct = int(report.robustness_score * 100)
        if rob_pct >= 88:
            n_88 += 1
            logger.info(
                "  [%d/%d] %s: OOS %+.2f%% | Rob %d%% | Trades %d  ★",
                idx + 1, len(weight_combos), name,
                report.oos_total_return, rob_pct, report.total_trades,
            )
        elif (idx + 1) % 20 == 0:
            logger.info(
                "  [%d/%d] %s: OOS %+.2f%% | Rob %d%%",
                idx + 1, len(weight_combos), name,
                report.oos_total_return, rob_pct,
            )

    logger.info("")
    logger.info("  Weight grid complete: %d/%d combos at 88%% robustness",
                 n_88, len(weight_combos))
    logger.info("")

    # Rank 88% combos by OOS return
    combos_88 = {
        k: v for k, v in grid_results.items()
        if int(v.robustness_score * 100) >= 88
    }

    if combos_88:
        logger.info("  All 88%% robustness combos (sorted by OOS return):")
        logger.info("  %-20s %8s %6s", "Weights", "OOS Ret", "Trades")
        logger.info("  " + "-" * 40)
        for name in sorted(
            combos_88,
            key=lambda k: combos_88[k].oos_total_return,
            reverse=True,
        ):
            rpt = combos_88[name]
            logger.info(
                "  %-20s %+7.2f%% %6d",
                name, rpt.oos_total_return, rpt.total_trades,
            )
        logger.info("")

        # Show window breakdown of top 3
        top3 = sorted(
            combos_88,
            key=lambda k: combos_88[k].oos_total_return,
            reverse=True,
        )[:3]
        for name in top3:
            logger.info("  Detail for %s:", name)
            log_cross_tf_detail(name, combos_88[name])
            logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Williams %R Parameter Stability
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: Williams %%R Parameter Stability")
    logger.info("-" * 72)
    logger.info("  Base: p14, t90, cool=6. Testing perturbations.")
    logger.info("  Using best weight from Part 1 (or 20/35/25/20 as default)")
    logger.info("")

    # Use best 88% combo from Part 1, or default
    if combos_88:
        best_weight_name = max(combos_88, key=lambda k: combos_88[k].oos_total_return)
        parts = best_weight_name.split("/")
        best_w = tuple(int(p) for p in parts)
    else:
        best_w = (20, 35, 25, 20)
        best_weight_name = "20/35/25/20"

    logger.info("  Using weight: %s", best_weight_name)
    logger.info("")

    perturbations = [
        ("base_p14_t90_c6", 14, 90, 90, 6),
        ("p14_t88_c6", 14, 88, 88, 6),
        ("p14_t92_c6", 14, 92, 92, 6),
        ("p14_t85_c6", 14, 85, 85, 6),
        ("p14_t95_c6", 14, 95, 95, 6),
        ("p14_t90_c4", 14, 90, 90, 4),
        ("p14_t90_c8", 14, 90, 90, 8),
        ("p10_t90_c6", 10, 90, 90, 6),
        ("p21_t90_c6", 21, 90, 90, 6),
        ("p7_t90_c6", 7, 90, 90, 6),
        # Asymmetric thresholds
        ("p14_os90_ob85_c6", 14, 90, 85, 6),
        ("p14_os85_ob90_c6", 14, 85, 90, 6),
    ]

    stability_results: dict[str, CrossTFReport] = {}

    for name, period, os_level, ob_level, cool in perturbations:
        df_1h = add_willr(df_1h, period)
        factory = lambda p=period, o=os_level, b=ob_level, c=cool: make_willr_1h(p, o, b, c)
        report = wf.run_cross_tf([
            CrossTFComponent(
                strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=best_w[0] / 100, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=best_w[1] / 100, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=best_w[2] / 100, label="15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=factory, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=best_w[3] / 100, label="1hWillR",
            ),
        ])
        stability_results[name] = report

        rob_pct = int(report.robustness_score * 100)
        marker = "★" if rob_pct >= 88 else ""
        logger.info(
            "  %s: OOS %+.2f%% | Rob %d%% | Trades %d  %s",
            name, report.oos_total_return, rob_pct,
            report.total_trades, marker,
        )

    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Summary
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 25b — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    # Weight grid summary
    total_combos = len(weight_combos)
    rob_counts: dict[int, int] = {}
    for rpt in grid_results.values():
        r = int(rpt.robustness_score * 100)
        rob_counts[r] = rob_counts.get(r, 0) + 1

    logger.info("  Weight Grid Robustness Distribution (%d combos):", total_combos)
    for r in sorted(rob_counts, reverse=True):
        logger.info("    %d%%: %d combos (%.1f%%)",
                     r, rob_counts[r], rob_counts[r] / total_combos * 100)
    logger.info("")

    if combos_88:
        best_combo_name = max(combos_88, key=lambda k: combos_88[k].oos_total_return)
        worst_combo_name = min(combos_88, key=lambda k: combos_88[k].oos_total_return)
        best_rpt = combos_88[best_combo_name]
        worst_rpt = combos_88[worst_combo_name]

        logger.info("  88%% combos: %d/%d", len(combos_88), total_combos)
        logger.info("    Best:  %s = OOS %+.2f%%", best_combo_name, best_rpt.oos_total_return)
        logger.info("    Worst: %s = OOS %+.2f%%", worst_combo_name, worst_rpt.oos_total_return)
        logger.info("")

    # Parameter stability summary
    stable_count = sum(
        1 for rpt in stability_results.values()
        if int(rpt.robustness_score * 100) >= 88
    )
    logger.info("  Parameter Stability: %d/%d perturbations at 88%% rob",
                 stable_count, len(perturbations))
    logger.info("  %-25s %8s %6s %6s", "Config", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 50)
    for name in sorted(
        stability_results,
        key=lambda k: (stability_results[k].robustness_score, stability_results[k].oos_total_return),
        reverse=True,
    ):
        rpt = stability_results[name]
        logger.info(
            "  %-25s %+7.2f%% %5d%% %6d",
            name, rpt.oos_total_return,
            int(rpt.robustness_score * 100), rpt.total_trades,
        )
    logger.info("")

    # Final verdict
    logger.info("  FINAL VERDICT:")
    logger.info("    3-comp baseline: 1hRSI/1hDC/15mRSI 33/33/34 = +18.81%%, 88%% rob")
    if combos_88:
        logger.info("    4-comp best:     %s = %+.2f%%, 88%% rob",
                     best_combo_name, best_rpt.oos_total_return)
        delta = best_rpt.oos_total_return - 18.81
        logger.info("    Delta: %+.2f%% OOS return improvement", delta)
        logger.info("    Parameter stable: %d/%d configs at 88%% rob",
                     stable_count, len(perturbations))
        if stable_count >= 8:
            logger.info("    ==> STRONG parameter stability. Production candidate!")
        elif stable_count >= 5:
            logger.info("    ==> MODERATE stability. Consider with caution.")
        else:
            logger.info("    ==> WEAK stability. NOT production-ready.")
    else:
        logger.info("    4-comp: NO combos at 88%%. Williams %%R adds no value.")

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 25b complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

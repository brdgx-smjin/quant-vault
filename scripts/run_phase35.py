#!/usr/bin/env python3
"""Phase 35 — Supertrend Trend-Following Strategy.

Previous 34 phases tested 30+ indicators. Supertrend is GENUINELY UNTESTED:
  - ATR-based dynamic support/resistance (different from Donchian fixed channel)
  - Direction flip mechanism (price crosses ST line)
  - Popular in crypto, never validated in this system

Phase 35 tests:
  Part 1: Supertrend standalone on 1h + MTF (grid: length × multiplier)
  Part 2: If ≥66% rob, as 5th component in 4-comp portfolio
  Part 3: If ≥66% rob, as Donchian replacement in 4-comp portfolio
  Part 4: 15m test if ≥55% rob on 1h

Baseline: 4-comp 15/50/10/25, 88% rob, +23.98% OOS, 236 trades.
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
    WalkForwardAnalyzer,
)
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy
from src.strategy.supertrend_trend import SupertrendTrendStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase35")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase35.log", mode="w")
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


def add_supertrend(
    df: pd.DataFrame, length: int = 10, multiplier: float = 3.0,
) -> pd.DataFrame:
    """Add Supertrend columns to DataFrame."""
    st_col = f"SUPERT_{length}_{multiplier}"
    if st_col not in df.columns:
        st = ta.supertrend(
            df["high"], df["low"], df["close"],
            length=length, multiplier=multiplier,
        )
        if st is not None:
            df[st_col] = st.iloc[:, 0]              # ST line value
            df[f"SUPERTd_{length}_{multiplier}"] = st.iloc[:, 1]   # Direction
            df[f"SUPERTl_{length}_{multiplier}"] = st.iloc[:, 2]   # Lower band
            df[f"SUPERTs_{length}_{multiplier}"] = st.iloc[:, 3]   # Upper band
    return df


# ─── Strategy Factories (baseline) ──────────────────────────────

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
        willr_period=14, oversold_level=90.0, overbought_level=90.0,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 35 — Supertrend Trend-Following Strategy")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Supertrend: ATR-based dynamic S/R, direction flip entry.")
    logger.info("  Different from Donchian: adaptive band width, different timing.")
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
    #   PART 1: Supertrend Standalone — Grid Search (1h + MTF)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: Supertrend Trend-Following (1h + MTF)")
    logger.info("-" * 72)
    logger.info("  Entry: direction flips from -1→1 (LONG) or 1→-1 (SHORT)")
    logger.info("  Volume confirmation: vol > 0.8 * 20-bar avg")
    logger.info("  Grid: length × multiplier × RR ratio")
    logger.info("")

    st_results = []
    st_lengths = [7, 10, 14, 20]
    st_multipliers = [2.0, 2.5, 3.0, 3.5]
    st_rr_ratios = [1.5, 2.0, 2.5]

    total = len(st_lengths) * len(st_multipliers) * len(st_rr_ratios)
    idx = 0

    for length in st_lengths:
        for mult in st_multipliers:
            df_1h_st = add_supertrend(df_1h.copy(), length=length, multiplier=mult)

            for rr in st_rr_ratios:
                idx += 1

                def make_st_1h(l=length, m=mult, r=rr):
                    base = SupertrendTrendStrategy(
                        st_length=l, st_multiplier=m,
                        atr_sl_mult=2.0, rr_ratio=r,
                        vol_mult=0.8, cooldown_bars=6,
                    )
                    return MultiTimeframeFilter(base)

                wf = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
                report = wf.run(make_st_1h, df_1h_st, htf_df=df_4h)

                rob = int(report.robustness_score * 100)
                st_results.append({
                    "length": length, "mult": mult, "rr": rr,
                    "rob": rob, "oos": report.oos_total_return,
                    "trades": report.oos_total_trades,
                })

                marker = "★" if rob >= 66 else ""
                logger.info("    [%d/%d] L%d_M%.1f_RR%.1f: Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                            idx, total, length, mult, rr, rob,
                            report.oos_total_return, report.oos_total_trades, marker)

    logger.info("")
    best_st = max(st_results, key=lambda x: (x["rob"], x["oos"]))
    logger.info("  Best Supertrend: L%d_M%.1f_RR%.1f → %d%% rob, %+.2f%% OOS, %d trades",
                best_st["length"], best_st["mult"], best_st["rr"],
                best_st["rob"], best_st["oos"], best_st["trades"])
    logger.info("")

    # Show top 5
    st_sorted = sorted(st_results, key=lambda x: (x["rob"], x["oos"]), reverse=True)
    logger.info("  Top 5 configs:")
    for i, r in enumerate(st_sorted[:5]):
        logger.info("    %d. L%d_M%.1f_RR%.1f: %d%% rob, %+.2f%% OOS, %d trades",
                    i + 1, r["length"], r["mult"], r["rr"],
                    r["rob"], r["oos"], r["trades"])
    logger.info("")

    # Robustness distribution
    rob_dist = {}
    for r in st_results:
        rb = r["rob"]
        rob_dist[rb] = rob_dist.get(rb, 0) + 1
    logger.info("  Robustness distribution:")
    for rb in sorted(rob_dist.keys(), reverse=True):
        logger.info("    %d%%: %d configs", rb, rob_dist[rb])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Cross-TF Portfolio — 5th Component
    # ═════════════════════════════════════════════════════════════
    if best_st["rob"] >= 66:
        logger.info("-" * 72)
        logger.info("  PART 2: Supertrend as 5th Component in 4-comp Portfolio")
        logger.info("-" * 72)
        logger.info("")

        bl = best_st["length"]
        bm = best_st["mult"]
        br = best_st["rr"]
        df_1h_best = add_supertrend(df_1h.copy(), length=bl, multiplier=bm)

        def make_st_best_1h(l=bl, m=bm, r=br):
            base = SupertrendTrendStrategy(
                st_length=l, st_multiplier=m,
                atr_sl_mult=2.0, rr_ratio=r,
                vol_mult=0.8, cooldown_bars=6,
            )
            return MultiTimeframeFilter(base)

        # 5-comp: 10/40/10/20/20 (reduce DC 50→40, WillR 25→20)
        wf = WalkForwardAnalyzer(n_windows=9)
        components_5comp = [
            CrossTFComponent(
                strategy_factory=make_rsi_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.10, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.40, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=make_rsi_15m,
                df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=0.10, label="15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_willr_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.20, label="1hWillR",
            ),
            CrossTFComponent(
                strategy_factory=make_st_best_1h,
                df=df_1h_best, htf_df=df_4h,
                engine=engine_1h, weight=0.20, label="1hST",
            ),
        ]

        report_5c = wf.run_cross_tf(components_5comp)
        rob_5c = int(report_5c.robustness_score * 100)
        logger.info("  5-comp (10/40/10/20/20): Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    rob_5c, report_5c.oos_total_return, report_5c.total_trades)
        for w in report_5c.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("    W%d: %s -> %+.2f%% %s",
                        w.window_id, " | ".join(parts), w.weighted_return, marker)
        logger.info("")

        # ═══════════════════════════════════════════════════════════
        #   PART 3: Supertrend Replacing Donchian
        # ═══════════════════════════════════════════════════════════
        logger.info("-" * 72)
        logger.info("  PART 3: Supertrend Replacing Donchian (15/50/10/25 weights)")
        logger.info("-" * 72)
        logger.info("  Both are trend-following — test if ST is better diversifier.")
        logger.info("")

        components_replace_dc = [
            CrossTFComponent(
                strategy_factory=make_rsi_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.15, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_st_best_1h,
                df=df_1h_best, htf_df=df_4h,
                engine=engine_1h, weight=0.50, label="1hST",
            ),
            CrossTFComponent(
                strategy_factory=make_rsi_15m,
                df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=0.10, label="15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_willr_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hWillR",
            ),
        ]

        report_replace_dc = wf.run_cross_tf(components_replace_dc)
        rob_rdc = int(report_replace_dc.robustness_score * 100)
        logger.info("  Replace DC (15/50/10/25): Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    rob_rdc, report_replace_dc.oos_total_return,
                    report_replace_dc.total_trades)
        for w in report_replace_dc.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("    W%d: %s -> %+.2f%% %s",
                        w.window_id, " | ".join(parts), w.weighted_return, marker)
        logger.info("")

        # Also test replacing WillR
        logger.info("  Supertrend Replacing WillR (15/50/10/25 weights):")
        components_replace_wr = [
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
                strategy_factory=make_st_best_1h,
                df=df_1h_best, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hST",
            ),
        ]

        report_replace_wr = wf.run_cross_tf(components_replace_wr)
        rob_rwr = int(report_replace_wr.robustness_score * 100)
        logger.info("  Replace WillR (15/50/10/25): Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    rob_rwr, report_replace_wr.oos_total_return,
                    report_replace_wr.total_trades)
        for w in report_replace_wr.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("    W%d: %s -> %+.2f%% %s",
                        w.window_id, " | ".join(parts), w.weighted_return, marker)
        logger.info("")

    else:
        logger.info("  Supertrend did not reach 66%% robustness standalone.")
        logger.info("  Skipping portfolio integration tests (Parts 2 & 3).")
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: 15m Test (if ≥55% rob on 1h)
    # ═════════════════════════════════════════════════════════════
    if best_st["rob"] >= 55:
        logger.info("-" * 72)
        logger.info("  PART 4: Supertrend on 15m")
        logger.info("-" * 72)
        logger.info("")

        bl = best_st["length"]
        bm = best_st["mult"]
        br = best_st["rr"]

        df_15m_st = add_supertrend(df_15m.copy(), length=bl, multiplier=bm)

        def make_st_15m(l=bl, m=bm, r=br):
            base = SupertrendTrendStrategy(
                st_length=l, st_multiplier=m,
                atr_sl_mult=2.0, rr_ratio=r,
                vol_mult=0.8, cooldown_bars=12,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
        report_15m = wf.run(make_st_15m, df_15m_st, htf_df=df_4h)
        rob_15m = int(report_15m.robustness_score * 100)

        logger.info("  15m: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    rob_15m, report_15m.oos_total_return,
                    report_15m.oos_total_trades)
        logger.info("")
    else:
        logger.info("  Supertrend too weak on 1h (<55%% rob). Skipping 15m test.")
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   SUMMARY
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 35 SUMMARY")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Baseline: 4-comp Fixed (15/50/10/25)")
    logger.info("    OOS Return: +23.98%% | Robustness: 88%% | Trades: 236")
    logger.info("")
    logger.info("  Best Supertrend standalone (1h + MTF):")
    logger.info("    L%d_M%.1f_RR%.1f: %d%% rob, %+.2f%% OOS, %d trades",
                best_st["length"], best_st["mult"], best_st["rr"],
                best_st["rob"], best_st["oos"], best_st["trades"])
    logger.info("")

    # Full grid results
    logger.info("  Full Grid Search Results (sorted by rob, then OOS):")
    for r in st_sorted:
        m = "★" if r["rob"] >= 66 else ""
        logger.info("    L%d_M%.1f_RR%.1f: %d%% rob, %+.2f%% OOS, %d trades %s",
                    r["length"], r["mult"], r["rr"],
                    r["rob"], r["oos"], r["trades"], m)
    logger.info("")

    if best_st["rob"] >= 66:
        logger.info("  Portfolio integration results:")
        logger.info("    5-comp (10/40/10/20/20): %d%% rob, %+.2f%% OOS",
                    rob_5c, report_5c.oos_total_return)
        logger.info("    Replace DC (15/50/10/25): %d%% rob, %+.2f%% OOS",
                    rob_rdc, report_replace_dc.oos_total_return)
        logger.info("    Replace WillR (15/50/10/25): %d%% rob, %+.2f%% OOS",
                    rob_rwr, report_replace_wr.oos_total_return)
        logger.info("")

        beats_baseline = (
            (rob_5c >= 88 and report_5c.oos_total_return > 23.98)
            or (rob_rdc >= 88 and report_replace_dc.oos_total_return > 23.98)
            or (rob_rwr >= 88 and report_replace_wr.oos_total_return > 23.98)
        )

        if beats_baseline:
            logger.info("  ★★★ BREAKTHROUGH: Supertrend IMPROVES portfolio! ★★★")
            logger.info("  ACTION: Update production weights and integrate Supertrend.")
        else:
            logger.info("  CONCLUSION: Supertrend does NOT improve 4-comp portfolio.")
            logger.info("  Same pattern as all prior 5th/replacement components.")
    else:
        logger.info("  CONCLUSION: Supertrend NOT viable standalone (<66%% rob).")
        logger.info("  Trend-following via Donchian remains superior for this dataset.")

    logger.info("")
    logger.info("  Phase 35 complete.")


if __name__ == "__main__":
    main()

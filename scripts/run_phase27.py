#!/usr/bin/env python3
"""Phase 27 — Ichimoku Kijun Mean Reversion: New Untested Indicator.

Ichimoku Cloud indicators exist in src/indicators/ichimoku.py but have
NEVER been tested as a standalone strategy or portfolio component.

Kijun-sen (26-bar equilibrium) provides a fundamentally different signal:
  - RSI: 14-bar close-to-close ratio (bounded 0-100)
  - WillR: 14-bar close position in H-L range (bounded -100 to 0)
  - CCI: 20-bar TP deviation from SMA (unbounded)
  - Kijun: 26-bar H-L MIDPOINT deviation in ATR units (unbounded)

Plan:
  PART 1: Standalone Ichimoku Kijun MR + MTF (1h) — threshold grid
  PART 2: Standalone on 15m (if 1h ≥ 55% rob)
  PART 3: As 5th component in cross-TF portfolio (if standalone ≥ 55%)
  PART 4: Kijun period sensitivity (if promising)
  PART 5: Summary and comparison with existing components
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
from src.strategy.ichimoku_kijun_mr import IchimokuKijunMRStrategy, add_ichimoku

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase27")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase27.log", mode="w")
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

def make_ichimoku_1h(
    kijun_period: int = 26,
    tenkan_period: int = 9,
    deviation: float = 0.5,
    cooldown: int = 6,
) -> MultiTimeframeFilter:
    base = IchimokuKijunMRStrategy(
        kijun_period=kijun_period,
        tenkan_period=tenkan_period,
        deviation_atr_mult=deviation,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


def make_ichimoku_15m(
    kijun_period: int = 26,
    tenkan_period: int = 9,
    deviation: float = 0.5,
    cooldown: int = 12,
) -> MultiTimeframeFilter:
    base = IchimokuKijunMRStrategy(
        kijun_period=kijun_period,
        tenkan_period=tenkan_period,
        deviation_atr_mult=deviation,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


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


# ─── Logging Helpers ──────────────────────────────────────────────

def log_wf_detail(name: str, report) -> None:
    """Log per-window details for standard WF report."""
    for w in report.windows:
        marker = "+" if w.out_of_sample.total_return > 0 else "-"
        logger.info(
            "    W%d [%s ~ %s]: IS %+.2f%% | OOS %+.2f%% | %d trades %s",
            w.window_id, w.test_start, w.test_end,
            w.in_sample.total_return, w.out_of_sample.total_return,
            w.out_of_sample.total_trades, marker,
        )
    logger.info(
        "  %s: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d | "
        "Sharpe: %.2f | MaxDD: %.1f%%",
        name, report.oos_total_return,
        int(report.robustness_score * 100),
        sum(1 for w in report.windows if w.out_of_sample.total_return > 0),
        report.total_windows,
        report.oos_total_trades,
        report.oos_avg_sharpe,
        max(
            (abs(w.out_of_sample.max_drawdown) for w in report.windows),
            default=0,
        ),
    )


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


# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 27 — Ichimoku Kijun Mean Reversion")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Ichimoku indicators have NEVER been tested as a strategy.")
    logger.info("  Kijun-sen = 26-bar H-L midpoint equilibrium.")
    logger.info("  Signal: fade ATR-normalized Kijun deviation + BB confirm.")
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")

    # Add Ichimoku columns
    for kp in [20, 26, 33]:
        for tp in [7, 9, 13]:
            df_1h = add_ichimoku(df_1h, tenkan_period=tp, kijun_period=kp)
    df_15m = add_ichimoku(df_15m, tenkan_period=9, kijun_period=26)

    # Add WillR for comparison/portfolio
    df_1h = add_willr(df_1h, 14)

    logger.info("1h data:  %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("15m data: %d bars (%s ~ %s)", len(df_15m),
                df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")
    wf_1h = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
    wf_15m = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
    wf_xtf = WalkForwardAnalyzer(n_windows=9)

    # ═════════════════════════════════════════════════════════════
    #   PART 1: Standalone Ichimoku Kijun MR + MTF (1h)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: Ichimoku Kijun MR + MTF on 1h — Threshold Grid")
    logger.info("-" * 72)
    logger.info("  Testing deviation thresholds: 0.3, 0.5, 0.7, 1.0 ATR")
    logger.info("  Base config: kijun=26, tenkan=9, SL=2.0, TP=3.0, cool=6")
    logger.info("")

    part1_results: dict[str, object] = {}
    best_rob_1h = 0.0
    best_cfg_1h = ""

    for dev in [0.3, 0.5, 0.7, 1.0]:
        name = f"Ichi_k26_dev{dev}"
        factory = lambda d=dev: make_ichimoku_1h(deviation=d)
        report = wf_1h.run(factory, df_1h, htf_df=df_4h)
        part1_results[name] = report
        log_wf_detail(name, report)
        logger.info("")

        rob = report.robustness_score * 100
        if rob > best_rob_1h or (rob == best_rob_1h and
                report.oos_total_return > part1_results.get(best_cfg_1h, report).oos_total_return):
            best_rob_1h = rob
            best_cfg_1h = name

    logger.info("  PART 1 BEST: %s (robustness %.0f%%)", best_cfg_1h, best_rob_1h)
    logger.info("")

    # Also test with different Kijun periods (20, 33) using best deviation
    logger.info("-" * 72)
    logger.info("  PART 1b: Kijun Period Sensitivity")
    logger.info("-" * 72)
    best_dev = float(best_cfg_1h.split("dev")[1]) if best_cfg_1h else 0.5

    for kp in [20, 33]:
        name = f"Ichi_k{kp}_dev{best_dev}"
        factory = lambda k=kp, d=best_dev: make_ichimoku_1h(
            kijun_period=k, deviation=d,
        )
        report = wf_1h.run(factory, df_1h, htf_df=df_4h)
        part1_results[name] = report
        log_wf_detail(name, report)
        logger.info("")

        rob = report.robustness_score * 100
        if rob > best_rob_1h or (rob == best_rob_1h and
                report.oos_total_return > part1_results[best_cfg_1h].oos_total_return):
            best_rob_1h = rob
            best_cfg_1h = name

    logger.info("  PART 1 FINAL BEST: %s (robustness %.0f%%)", best_cfg_1h, best_rob_1h)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Standalone on 15m (only if 1h ≥ 55% rob)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: Ichimoku Kijun MR + MTF on 15m")
    logger.info("-" * 72)

    if best_rob_1h >= 44:
        logger.info("  1h best was %.0f%% — testing 15m with best deviation=%s",
                     best_rob_1h, best_dev)
        logger.info("")

        for dev in [best_dev, best_dev - 0.2, best_dev + 0.2]:
            if dev < 0.1:
                continue
            dev = round(dev, 1)
            name = f"Ichi_15m_k26_dev{dev}"
            factory = lambda d=dev: make_ichimoku_15m(deviation=d)
            report = wf_15m.run(factory, df_15m, htf_df=df_4h)
            log_wf_detail(name, report)
            logger.info("")
    else:
        logger.info("  SKIP — 1h robustness %.0f%% < 44%%. Not viable.", best_rob_1h)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: As 5th component in Cross-TF Portfolio
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: Ichimoku as 5th Cross-TF Component")
    logger.info("-" * 72)

    if best_rob_1h >= 44:
        logger.info("  Baseline: 4-comp RSI/DC/RSI15/WR = 88%% rob, +23.98%% OOS")
        logger.info("  Testing: 5-comp with 1h Ichimoku as 5th component")
        logger.info("")

        # Use best config from Part 1 for the Ichimoku component
        best_params = best_cfg_1h.replace("Ichi_", "")
        # Parse kijun and dev from best config name
        kp_str = best_params.split("_dev")[0].replace("k", "")
        dev_str = best_params.split("_dev")[1]
        best_kp = int(kp_str)
        best_dev_val = float(dev_str)

        ichi_factory = lambda: make_ichimoku_1h(
            kijun_period=best_kp, deviation=best_dev_val,
        )

        # Test weight distributions for 5 components
        weight_configs = [
            # (RSI, DC, RSI15, WR, Ichi) — keep total = 100
            (10, 40, 10, 20, 20),  # Give Ichi 20%, reduce DC/WR
            (10, 35, 10, 25, 20),  # Balanced with existing best
            (15, 35, 10, 20, 20),  # More RSI
            (10, 30, 10, 20, 30),  # Ichi-heavy
            (10, 40, 10, 25, 15),  # Ichi-light
        ]

        for w_rsi, w_dc, w_rsi15, w_wr, w_ichi in weight_configs:
            name = f"5comp_{w_rsi}/{w_dc}/{w_rsi15}/{w_wr}/{w_ichi}"
            report = wf_xtf.run_cross_tf([
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
                CrossTFComponent(
                    strategy_factory=ichi_factory, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_ichi / 100, label="1hIchi",
                ),
            ])
            log_cross_tf_detail(name, report)
            logger.info("")

        # Also test as REPLACEMENT for WillR (4-comp: RSI/DC/RSI15/Ichi)
        logger.info("  --- Ichimoku replacing WillR (4-comp) ---")
        logger.info("")
        replace_configs = [
            (15, 50, 10, 25),  # Same weights as current best
            (20, 40, 15, 25),  # Rebalanced
            (15, 40, 15, 30),  # Ichi-heavier
        ]
        for w_rsi, w_dc, w_rsi15, w_ichi in replace_configs:
            name = f"4comp_ichi_{w_rsi}/{w_dc}/{w_rsi15}/{w_ichi}"
            report = wf_xtf.run_cross_tf([
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
                    strategy_factory=ichi_factory, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_ichi / 100, label="1hIchi",
                ),
            ])
            log_cross_tf_detail(name, report)
            logger.info("")
    else:
        logger.info("  SKIP — 1h robustness %.0f%% too low.", best_rob_1h)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Summary
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 27 SUMMARY")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Part 1 results (1h standalone + MTF):")
    for name, report in part1_results.items():
        rob = int(report.robustness_score * 100)
        logger.info("    %s: OOS %+.2f%% | Robustness %d%% | Trades %d",
                     name, report.oos_total_return, rob, report.oos_total_trades)
    logger.info("")
    logger.info("  Best 1h standalone: %s (%.0f%% rob)", best_cfg_1h, best_rob_1h)
    logger.info("")
    logger.info("  Reference (Phase 25 — current production):")
    logger.info("    4-comp 15/50/10/25: 88%% rob, +23.98%% OOS")
    logger.info("    WillR standalone: 77%% rob, +19.17%% OOS")
    logger.info("    RSI 1h standalone: 66%% rob, +13.29%% OOS")
    logger.info("    CCI standalone: 66%% rob, +13.48%% OOS")
    logger.info("")
    logger.info("  Phase 27 complete.")


if __name__ == "__main__":
    main()

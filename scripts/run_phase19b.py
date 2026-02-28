#!/usr/bin/env python3
"""Phase 19b — 30m RSI+MTF Sweep + Cross-TF Portfolio Test.

Phase 19a findings:
  - 2h RSI+MTF: 33% rob, FAILS — too few trades (44 per year, 0 per OOS window)
  - Time-of-day: small sample, no actionable filter

Phase 19b hypothesis:
  30m RSI_mid (35/65) + MTF may work. 30m was previously tested in early
  phases with different configs, but NOT with the current optimal RSI_mid
  approach (cool=24 for 30m = 12h cooldown, max_hold=48 for 24h hold).

  If 30m has different negative windows than 1h/15m, it could be a
  diversification leg.

Test plan:
  PART 1: 30m RSI_mid + MTF sweep (cooldown and max_hold variants)
  PART 2: If 30m works (>=55% rob), test as 4th component in cross-TF
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
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

logger = logging.getLogger("phase19b")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase19b.log", mode="w")
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


# ─── Strategy Factories ──────────────────────────────────────────

def make_rsi_30m(cooldown: int = 24) -> MultiTimeframeFilter:
    """RSI MR on 30m with configurable cooldown.

    30m-specific params:
      - cool=24 bars = 12h cooldown (same as 15m cool=12 = 3h? No: 24*30m = 12h)
      - Actually let's test multiple cooldowns
    """
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=cooldown,
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


# ─── PART 1: 30m RSI+MTF Sweep ───────────────────────────────────

def part1_30m_sweep() -> list[dict]:
    """Test RSI MR + MTF on 30m with multiple configs."""
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PART 1: 30m RSI_35_65+MTF Sweep (9 windows)")
    logger.info("=" * 72)
    logger.info("")

    df_30m = load_data("30m")
    df_4h = load_data("4h")

    logger.info("  30m data: %d bars (%s ~ %s)", len(df_30m),
                df_30m.index[0].date(), df_30m.index[-1].date())
    logger.info("  4h data:  %d bars", len(df_4h))
    logger.info("")

    # Configs: (cooldown, max_hold_bars, label)
    # 30m bar equivalents:
    #   cool=6 = 3h, cool=12 = 6h, cool=24 = 12h
    #   hold=48 = 24h, hold=96 = 48h, hold=192 = 96h
    configs = [
        (6,  48,  "cool6_hold48"),    # 3h cooldown, 24h max hold
        (12, 48,  "cool12_hold48"),   # 6h cooldown, 24h max hold
        (12, 96,  "cool12_hold96"),   # 6h cooldown, 48h max hold
        (24, 96,  "cool24_hold96"),   # 12h cooldown, 48h max hold
        (24, 192, "cool24_hold192"),  # 12h cooldown, 96h max hold
    ]

    results = []

    for cooldown, max_hold, label in configs:
        logger.info("  --- 30m RSI_35_65+MTF (%s) ---", label)

        engine = BacktestEngine(max_hold_bars=max_hold, freq="30m")
        wf = WalkForwardAnalyzer(train_ratio=0.7, n_windows=9, engine=engine)

        def factory(cd=cooldown):
            return make_rsi_30m(cd)

        report = wf.run(factory, df_30m, htf_df=df_4h)

        for w in report.windows:
            oos = w.out_of_sample
            logger.info(
                "  W%d: OOS %+6.2f%% (WR %d%%, %d tr)",
                w.window_id, oos.total_return,
                int(oos.win_rate * 100) if oos.total_trades > 0 else 0,
                oos.total_trades,
            )

        logger.info(
            "  OOS: %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
            report.oos_total_return,
            int(report.robustness_score * 100),
            report.oos_profitable_windows,
            report.total_windows,
            report.oos_total_trades,
        )

        # Full backtest
        strat_full = factory()
        full_result = engine.run(strat_full, df_30m, htf_df=df_4h)
        logger.info(
            "  Full %+.2f%% | Shp %5.2f | DD %5.1f%% | WR %5.1f%% | %3d tr | PF %5.2f",
            full_result.total_return, full_result.sharpe_ratio,
            full_result.max_drawdown * 100, full_result.win_rate * 100,
            full_result.total_trades, full_result.profit_factor,
        )
        logger.info("")

        results.append({
            "label": label,
            "cooldown": cooldown,
            "max_hold": max_hold,
            "oos_return": report.oos_total_return,
            "robustness": report.robustness_score,
            "rob_pct": int(report.robustness_score * 100),
            "trades": report.oos_total_trades,
            "full_return": full_result.total_return,
            "sharpe": full_result.sharpe_ratio,
            "dd": full_result.max_drawdown * 100,
            "report": report,
        })

    # Summary
    logger.info("  ─── 30m RSI+MTF Summary ───")
    logger.info("  %-20s  %8s  %8s  %7s  %8s  %6s", "Config", "OOS Ret", "Rob", "Trades", "Full", "Sharpe")
    for r in sorted(results, key=lambda x: (-x["robustness"], -x["oos_return"])):
        logger.info(
            "  %-20s  %+7.2f%%  %5d%%  %6d  %+7.2f%%  %5.2f",
            r["label"], r["oos_return"], r["rob_pct"],
            r["trades"], r["full_return"], r["sharpe"],
        )
    logger.info("")

    return results


# ─── PART 2: Cross-TF with 30m ───────────────────────────────────

def part2_cross_tf_with_30m(best_config: dict) -> None:
    """Add 30m RSI as 4th component to cross-TF portfolio."""
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PART 2: Cross-TF Portfolio with 30m RSI (9 windows)")
    logger.info("=" * 72)
    logger.info("")

    cooldown = best_config["cooldown"]
    max_hold = best_config["max_hold"]

    logger.info("  Best 30m config: cool=%d, hold=%d, rob=%d%%",
                cooldown, max_hold, best_config["rob_pct"])
    logger.info("")

    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_30m = load_data("30m")
    df_4h = load_data("4h")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")
    engine_30m = BacktestEngine(max_hold_bars=max_hold, freq="30m")

    wf = WalkForwardAnalyzer(train_ratio=0.7, n_windows=9)

    def factory_30m(cd=cooldown):
        return make_rsi_30m(cd)

    weight_schemes = [
        (0.25, 0.25, 0.25, 0.25, "equal_25"),
        (0.30, 0.25, 0.30, 0.15, "15m30m_light"),
        (0.25, 0.25, 0.30, 0.20, "15m_heavy"),
    ]

    for w1h_rsi, w1h_dc, w15m, w30m, label in weight_schemes:
        logger.info("  --- 4-comp %s: 1hRSI/1hDC/15mRSI/30mRSI = %.0f/%.0f/%.0f/%.0f ---",
                    label, w1h_rsi * 100, w1h_dc * 100, w15m * 100, w30m * 100)

        components = [
            CrossTFComponent(
                strategy_factory=make_rsi_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w1h_rsi, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_dc_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w1h_dc, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=make_rsi_15m,
                df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w15m, label="15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=factory_30m,
                df=df_30m, htf_df=df_4h,
                engine=engine_30m, weight=w30m, label="30mRSI",
            ),
        ]

        report = wf.run_cross_tf(components)

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
            label, report.oos_total_return,
            int(report.robustness_score * 100),
            report.oos_profitable_windows,
            report.total_windows,
            report.total_trades,
        )
        logger.info("")

    # Baseline
    logger.info("  --- Baseline 3-comp: 1hRSI/1hDC/15mRSI = 33/33/34 ---")
    components_baseline = [
        CrossTFComponent(
            strategy_factory=make_rsi_1h,
            df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.33, label="1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h,
            df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.33, label="1hDC",
        ),
        CrossTFComponent(
            strategy_factory=make_rsi_15m,
            df=df_15m, htf_df=df_4h,
            engine=engine_15m, weight=0.34, label="15mRSI",
        ),
    ]
    baseline = wf.run_cross_tf(components_baseline)
    for w in baseline.windows:
        parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
        marker = "+" if w.weighted_return > 0 else "-"
        logger.info(
            "    W%d [%s ~ %s]: %s -> %+.2f%% %s",
            w.window_id, w.test_start, w.test_end,
            " | ".join(parts), w.weighted_return, marker,
        )
    logger.info(
        "  Baseline 33/33/34: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        baseline.oos_total_return,
        int(baseline.robustness_score * 100),
        baseline.oos_profitable_windows,
        baseline.total_windows,
        baseline.total_trades,
    )
    logger.info("")


# ─── Main ─────────────────────────────────────────────────────────

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 19b — 30m RSI+MTF Exploration")
    logger.info("=" * 72)
    logger.info("")

    results = part1_30m_sweep()

    if results:
        best = max(results, key=lambda x: (x["robustness"], x["oos_return"]))
        logger.info("  Best 30m config: %s (rob=%d%%, OOS=%+.2f%%)",
                    best["label"], best["rob_pct"], best["oos_return"])
        logger.info("")

        if best["robustness"] >= 0.55:
            part2_cross_tf_with_30m(best)
        else:
            logger.info("  30m RSI robustness < 55%% — skipping cross-TF test.")
            logger.info("")

    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 19b — COMPLETE")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

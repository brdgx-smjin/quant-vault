#!/usr/bin/env python3
"""Phase 19 — 2h Timeframe Exploration + Time-of-Day Analysis.

Phase 18 conclusion: 88% robustness is the ceiling for BTC/USDT with
existing strategy types on 1h + 15m. W2 (Nov 20-Dec 2) is unsolvable.

Phase 19 hypothesis:
  A) 2h RSI+MTF may work as a standalone strategy AND may have different
     negative windows than 1h/15m, enabling further cross-TF diversification.
  B) Time-of-day patterns during W2 could reveal session-specific weaknesses.

Test plan:
  PART 1: Resample 1m → 2h, test RSI MR + MTF (sweep cooldown/hold combos)
  PART 2: If 2h works, test as 4th component in cross-TF portfolio
  PART 3: Time-of-day analysis of all WF windows (which sessions lose money?)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
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

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase19")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase19.log", mode="w")
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


def resample_to_2h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h data to 2h OHLCV bars and add indicators.

    Uses 1h data (full year) rather than 1m (only ~3 months available).
    """
    ohlcv = df_1h.resample("2h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    ohlcv = BasicIndicators.add_all(ohlcv)
    return ohlcv


# ─── Strategy Factories ──────────────────────────────────────────

def make_rsi_2h(cooldown: int = 3) -> MultiTimeframeFilter:
    """RSI MR on 2h with configurable cooldown."""
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


# ─── PART 1: 2h RSI+MTF Sweep ────────────────────────────────────

def part1_2h_sweep() -> None:
    """Test RSI MR + MTF on 2h with multiple cooldown/hold configs."""
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PART 1: 2h Timeframe RSI+MTF Sweep (9 windows)")
    logger.info("=" * 72)
    logger.info("")

    # Load 1h data and resample to 2h (1m only has ~3 months)
    logger.info("Loading 1h data and resampling to 2h...")
    df_1h_raw = pd.read_parquet(ROOT / "data/processed/BTC_USDT_USDT_1h.parquet")
    df_2h = resample_to_2h(df_1h_raw)
    df_4h = load_data("4h")

    logger.info("  2h data: %d bars (%s ~ %s)", len(df_2h),
                df_2h.index[0].date(), df_2h.index[-1].date())
    logger.info("  4h data: %d bars", len(df_4h))
    logger.info("")

    # Configs to test: (cooldown, max_hold_bars, label)
    configs = [
        (3, 24, "cool3_hold24"),   # 6h cooldown, 48h max hold
        (3, 36, "cool3_hold36"),   # 6h cooldown, 72h max hold
        (4, 24, "cool4_hold24"),   # 8h cooldown, 48h max hold
        (6, 24, "cool6_hold24"),   # 12h cooldown, 48h max hold
        (4, 36, "cool4_hold36"),   # 8h cooldown, 72h max hold
    ]

    results = []

    for cooldown, max_hold, label in configs:
        logger.info("  --- 2h RSI_35_65+MTF (%s) ---", label)

        engine = BacktestEngine(max_hold_bars=max_hold, freq="2h")
        wf = WalkForwardAnalyzer(train_ratio=0.7, n_windows=9, engine=engine)

        def factory(cd=cooldown):
            return make_rsi_2h(cd)

        report = wf.run(factory, df_2h, htf_df=df_4h)

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
        full_result = engine.run(strat_full, df_2h, htf_df=df_4h)
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

    # Summary table
    logger.info("  ─── 2h RSI+MTF Summary ───")
    logger.info("  %-20s  %8s  %8s  %7s  %8s  %6s", "Config", "OOS Ret", "Rob", "Trades", "Full", "Sharpe")
    for r in sorted(results, key=lambda x: -x["robustness"]):
        logger.info(
            "  %-20s  %+7.2f%%  %5d%%  %6d  %+7.2f%%  %5.2f",
            r["label"], r["oos_return"], r["rob_pct"],
            r["trades"], r["full_return"], r["sharpe"],
        )
    logger.info("")

    return results


# ─── PART 2: Cross-TF with 2h (if PART 1 is promising) ──────────

def part2_cross_tf_with_2h(best_2h_config: dict) -> None:
    """Add 2h RSI as 4th component to the cross-TF portfolio."""
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PART 2: Cross-TF Portfolio with 2h RSI (9 windows)")
    logger.info("=" * 72)
    logger.info("")

    cooldown = best_2h_config["cooldown"]
    max_hold = best_2h_config["max_hold"]
    rob_pct = best_2h_config["rob_pct"]

    logger.info("  Best 2h config: cool=%d, hold=%d, rob=%d%%",
                cooldown, max_hold, rob_pct)
    logger.info("")

    # Load data
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")

    df_1h_raw = pd.read_parquet(ROOT / "data/processed/BTC_USDT_USDT_1h.parquet")
    df_2h = resample_to_2h(df_1h_raw)

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")
    engine_2h = BacktestEngine(max_hold_bars=max_hold, freq="2h")

    wf = WalkForwardAnalyzer(train_ratio=0.7, n_windows=9)

    def factory_2h(cd=cooldown):
        return make_rsi_2h(cd)

    # Weight schemes to test
    weight_schemes = [
        # (1hRSI, 1hDC, 15mRSI, 2hRSI, label)
        (0.25, 0.25, 0.25, 0.25, "equal_25"),
        (0.30, 0.25, 0.25, 0.20, "1hRSI_heavy"),
        (0.25, 0.20, 0.35, 0.20, "15m_heavy"),
        (0.20, 0.20, 0.30, 0.30, "2h_heavy"),
    ]

    for w1h_rsi, w1h_dc, w15m, w2h, label in weight_schemes:
        logger.info("  --- 4-comp %s: 1hRSI/1hDC/15mRSI/2hRSI = %.0f/%.0f/%.0f/%.0f ---",
                    label, w1h_rsi * 100, w1h_dc * 100, w15m * 100, w2h * 100)

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
                strategy_factory=factory_2h,
                df=df_2h, htf_df=df_4h,
                engine=engine_2h, weight=w2h, label="2hRSI",
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

    # Also test baseline 3-comp for comparison
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


# ─── PART 3: Time-of-Day Analysis ────────────────────────────────

def part3_time_analysis() -> None:
    """Analyze trade performance by time-of-day and day-of-week."""
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PART 3: Time-of-Day / Day-of-Week Analysis")
    logger.info("=" * 72)
    logger.info("")

    df_1h = load_data("1h")
    df_4h = load_data("4h")

    engine = BacktestEngine(max_hold_bars=48, freq="1h")

    # Run full backtest for RSI+MTF (1h) and extract trade logs
    strat = make_rsi_1h()
    result = engine.run(strat, df_1h, htf_df=df_4h)

    if not result.trade_logs:
        logger.info("  No trade logs available for analysis.")
        return

    trades = pd.DataFrame([{
        "entry_time": pd.Timestamp(t.entry_time),
        "exit_time": pd.Timestamp(t.exit_time),
        "side": t.side,
        "pnl": t.pnl,
        "return_pct": t.return_pct,
        "bars_held": t.bars_held,
        "exit_reason": t.exit_reason,
    } for t in result.trade_logs])

    trades["entry_hour"] = trades["entry_time"].dt.hour
    trades["entry_dow"] = trades["entry_time"].dt.dayofweek
    trades["entry_date"] = trades["entry_time"].dt.date

    # By hour
    logger.info("  RSI+MTF (1h) — Trade PnL by Entry Hour (UTC):")
    logger.info("  %-6s  %6s  %8s  %8s", "Hour", "Trades", "Avg Ret%", "Total%")
    for hour in range(24):
        mask = trades["entry_hour"] == hour
        subset = trades[mask]
        if len(subset) > 0:
            logger.info(
                "  %02d:00  %6d  %+7.2f%%  %+7.2f%%",
                hour, len(subset),
                subset["return_pct"].mean(),
                subset["return_pct"].sum(),
            )
    logger.info("")

    # By day of week
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    logger.info("  RSI+MTF (1h) — Trade PnL by Day of Week:")
    logger.info("  %-6s  %6s  %8s  %8s", "Day", "Trades", "Avg Ret%", "Total%")
    for dow in range(7):
        mask = trades["entry_dow"] == dow
        subset = trades[mask]
        if len(subset) > 0:
            logger.info(
                "  %-6s  %6d  %+7.2f%%  %+7.2f%%",
                days[dow], len(subset),
                subset["return_pct"].mean(),
                subset["return_pct"].sum(),
            )
    logger.info("")

    # Focus on W2 period (Nov 20 - Dec 2)
    w2_start = pd.Timestamp("2025-11-20")
    w2_end = pd.Timestamp("2025-12-02")
    w2_mask = (trades["entry_time"] >= w2_start) & (trades["entry_time"] < w2_end)
    w2_trades = trades[w2_mask]

    logger.info("  W2 [Nov 20 - Dec 2] trades detail:")
    logger.info("  %-20s  %-5s  %8s  %8s  %-8s  %s",
                "Entry", "Side", "Ret%", "PnL", "Exit", "Held")
    for _, t in w2_trades.iterrows():
        logger.info(
            "  %-20s  %-5s  %+7.2f%%  %+8.1f  %-8s  %d bars",
            str(t["entry_time"])[:16], t["side"],
            t["return_pct"], t["pnl"],
            t["exit_reason"], t["bars_held"],
        )
    logger.info("")

    # W2 by hour
    if len(w2_trades) > 0:
        logger.info("  W2 trades by entry hour:")
        logger.info("  %-6s  %6s  %8s", "Hour", "Trades", "Avg Ret%")
        for hour in sorted(w2_trades["entry_hour"].unique()):
            subset = w2_trades[w2_trades["entry_hour"] == hour]
            logger.info(
                "  %02d:00  %6d  %+7.2f%%",
                hour, len(subset), subset["return_pct"].mean(),
            )
        logger.info("")


# ─── Main ─────────────────────────────────────────────────────────

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 19 — 2h Timeframe + Time-of-Day Analysis")
    logger.info("=" * 72)
    logger.info("")

    # PART 1: 2h sweep
    results = part1_2h_sweep()

    # Find best config by robustness, then OOS return
    if results:
        best = max(results, key=lambda x: (x["robustness"], x["oos_return"]))
        logger.info("  Best 2h config: %s (rob=%d%%, OOS=%+.2f%%)",
                    best["label"], best["rob_pct"], best["oos_return"])
        logger.info("")

        # Only proceed to PART 2 if robustness >= 55% (worth testing as component)
        if best["robustness"] >= 0.55:
            part2_cross_tf_with_2h(best)
        else:
            logger.info("  2h RSI robustness < 55%% — skipping cross-TF portfolio test.")
            logger.info("")

    # PART 3: Time analysis (always run)
    part3_time_analysis()

    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 19 — COMPLETE")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

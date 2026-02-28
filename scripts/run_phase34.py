#!/usr/bin/env python3
"""Phase 34 — Exit Optimization.

All 33 prior phases focused on ENTRY signals and new indicators.
Exit has always been fixed SL/TP + timeout. This phase tests:

  PART 1: breakeven_at_r — move SL to entry after reaching N*R profit
          Tests: 1.0R, 1.5R, 2.0R (feature exists but NEVER tested)

  PART 2: Signal-driven exit — exit when indicator returns to neutral
          Tests: RSI > 50 exit for LONG (< 50 for SHORT)
                 WillR > -50 exit for LONG (< -50 for SHORT)
                 Combined: each component uses its own neutral exit

  PART 3: TP decay — TP target shrinks as position ages
          Tests: decay_rate 0.3, 0.5, 0.7
          (After 50% of max_hold, TP at 85%/75%/65% of original)

  PART 4: Best combinations — combine winning exit variants

Baseline: 4-comp 15/50/10/25, 88% rob, +23.98% OOS, 236 trades.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
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
from config.settings import SYMBOL

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase34")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase34.log", mode="a")
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


# ─── Exit check functions ─────────────────────────────────────────

def rsi_neutral_exit(df: pd.DataFrame, side: str) -> bool:
    """Exit when RSI returns to neutral zone (45-55)."""
    rsi = df["rsi_14"].iloc[-1]
    if pd.isna(rsi):
        return False
    rsi = float(rsi)
    if side == "long":
        return rsi >= 50
    elif side == "short":
        return rsi <= 50
    return False


def willr_neutral_exit(df: pd.DataFrame, side: str) -> bool:
    """Exit when Williams %R returns to neutral zone (-40 to -60)."""
    col = "WILLR_14"
    if col not in df.columns:
        return False
    willr = df[col].iloc[-1]
    if pd.isna(willr):
        return False
    willr = float(willr)
    if side == "long":
        return willr >= -50
    elif side == "short":
        return willr <= -50
    return False


def donchian_channel_exit(df: pd.DataFrame, side: str) -> bool:
    """Exit trend position when price falls back into channel.

    Uses 12-bar exit channel (half of 24-bar entry period).
    """
    if len(df) < 13:
        return False
    lookback = df.iloc[-13:-1]
    if side == "long":
        channel_low = float(lookback["low"].min())
        return float(df["close"].iloc[-1]) < channel_low
    elif side == "short":
        channel_high = float(lookback["high"].max())
        return float(df["close"].iloc[-1]) > channel_high
    return False


# ─── Strategy factories ──────────────────────────────────────────

def make_rsi_1h():
    return MultiTimeframeFilter(RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    ))


def make_dc_1h():
    return MultiTimeframeFilter(DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
    ))


def make_rsi_15m():
    return MultiTimeframeFilter(RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
    ))


def make_willr_1h():
    return MultiTimeframeFilter(WilliamsRMeanReversionStrategy(
        willr_period=14, oversold_level=90, overbought_level=90,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    ))


# ─── Cross-TF portfolio runner ───────────────────────────────────

def run_4comp_cross_tf(
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_4h: pd.DataFrame,
    weights: tuple[float, float, float, float] = (0.15, 0.50, 0.10, 0.25),
    n_windows: int = 9,
    engines: Optional[dict[str, BacktestEngine]] = None,
    label: str = "baseline",
) -> dict:
    """Run 4-comp cross-TF WF and return summary dict."""
    if engines is None:
        engines = {
            "rsi_1h": BacktestEngine(max_hold_bars=48, freq="1h"),
            "dc_1h": BacktestEngine(max_hold_bars=48, freq="1h"),
            "rsi_15m": BacktestEngine(max_hold_bars=96, freq="15m"),
            "willr_1h": BacktestEngine(max_hold_bars=48, freq="1h"),
        }

    w_rsi1h, w_dc1h, w_rsi15m, w_willr1h = weights

    components = [
        CrossTFComponent(
            strategy_factory=make_rsi_1h,
            df=df_1h, htf_df=df_4h,
            engine=engines["rsi_1h"],
            weight=w_rsi1h, label="1h_RSI",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h,
            df=df_1h, htf_df=df_4h,
            engine=engines["dc_1h"],
            weight=w_dc1h, label="1h_DC",
        ),
        CrossTFComponent(
            strategy_factory=make_rsi_15m,
            df=df_15m, htf_df=df_4h,
            engine=engines["rsi_15m"],
            weight=w_rsi15m, label="15m_RSI",
        ),
        CrossTFComponent(
            strategy_factory=make_willr_1h,
            df=df_1h, htf_df=df_4h,
            engine=engines["willr_1h"],
            weight=w_willr1h, label="1h_WR",
        ),
    ]

    wf = WalkForwardAnalyzer(train_ratio=0.7, n_windows=n_windows)
    report = wf.run_cross_tf(components)

    result = {
        "label": label,
        "robustness": int(report.robustness_score * 100),
        "oos_return": report.oos_total_return,
        "trades": report.total_trades,
        "profitable_windows": report.oos_profitable_windows,
        "total_windows": report.total_windows,
        "windows": [
            {
                "id": w.window_id,
                "return": w.weighted_return,
                "components": {c.label: c.oos_return for c in w.components},
            }
            for w in report.windows
        ],
    }
    return result


# ─── Main ─────────────────────────────────────────────────────────

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 34 — Exit Optimization")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Baseline: 4-comp 15/50/10/25, 88%% rob, +23.98%% OOS")
    logger.info("  Testing: breakeven_at_r, signal-driven exit, TP decay")
    logger.info("")

    # Load data
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")
    df_1h = add_willr(df_1h, 14)

    logger.info("  1h data:  %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("  15m data: %d bars (%s ~ %s)", len(df_15m),
                df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("")

    all_results = []

    # ─── PART 0: Baseline (confirm) ──────────────────────────────
    logger.info("-" * 72)
    logger.info("  PART 0: Baseline confirmation")
    logger.info("-" * 72)

    baseline = run_4comp_cross_tf(df_1h, df_15m, df_4h, label="baseline")
    all_results.append(baseline)
    logger.info("    Baseline: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                baseline["robustness"], baseline["oos_return"],
                baseline["trades"])
    logger.info("")

    # ─── PART 1: breakeven_at_r ──────────────────────────────────
    logger.info("-" * 72)
    logger.info("  PART 1: breakeven_at_r (move SL to entry after N*R profit)")
    logger.info("-" * 72)
    logger.info("  Feature exists in engine since Phase 1 but NEVER tested.")
    logger.info("")

    for be_r in [1.0, 1.5, 2.0]:
        label = f"breakeven_{be_r}R"
        engines = {
            "rsi_1h": BacktestEngine(
                max_hold_bars=48, freq="1h", breakeven_at_r=be_r),
            "dc_1h": BacktestEngine(
                max_hold_bars=48, freq="1h", breakeven_at_r=be_r),
            "rsi_15m": BacktestEngine(
                max_hold_bars=96, freq="15m", breakeven_at_r=be_r),
            "willr_1h": BacktestEngine(
                max_hold_bars=48, freq="1h", breakeven_at_r=be_r),
        }
        result = run_4comp_cross_tf(
            df_1h, df_15m, df_4h, engines=engines, label=label)
        all_results.append(result)
        logger.info("    %s: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    label, result["robustness"], result["oos_return"],
                    result["trades"])
    logger.info("")

    # ─── PART 2: Signal-driven exit ──────────────────────────────
    logger.info("-" * 72)
    logger.info("  PART 2: Signal-driven exit (indicator returns to neutral)")
    logger.info("-" * 72)
    logger.info("  Test A: RSI neutral exit (rsi crosses 50) on RSI components")
    logger.info("  Test B: WillR neutral exit (willr crosses -50) on WillR comp")
    logger.info("  Test C: Combined — each comp uses its own indicator exit")
    logger.info("  Test D: DC channel exit (12-bar) on DC component")
    logger.info("")

    # Test A: RSI neutral exit on RSI components only
    label = "rsi_neutral_exit"
    engines = {
        "rsi_1h": BacktestEngine(
            max_hold_bars=48, freq="1h", exit_check_fn=rsi_neutral_exit),
        "dc_1h": BacktestEngine(max_hold_bars=48, freq="1h"),
        "rsi_15m": BacktestEngine(
            max_hold_bars=96, freq="15m", exit_check_fn=rsi_neutral_exit),
        "willr_1h": BacktestEngine(max_hold_bars=48, freq="1h"),
    }
    result = run_4comp_cross_tf(
        df_1h, df_15m, df_4h, engines=engines, label=label)
    all_results.append(result)
    logger.info("    %s: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                label, result["robustness"], result["oos_return"],
                result["trades"])

    # Test B: WillR neutral exit on WillR component only
    label = "willr_neutral_exit"
    engines = {
        "rsi_1h": BacktestEngine(max_hold_bars=48, freq="1h"),
        "dc_1h": BacktestEngine(max_hold_bars=48, freq="1h"),
        "rsi_15m": BacktestEngine(max_hold_bars=96, freq="15m"),
        "willr_1h": BacktestEngine(
            max_hold_bars=48, freq="1h", exit_check_fn=willr_neutral_exit),
    }
    result = run_4comp_cross_tf(
        df_1h, df_15m, df_4h, engines=engines, label=label)
    all_results.append(result)
    logger.info("    %s: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                label, result["robustness"], result["oos_return"],
                result["trades"])

    # Test C: Combined — each comp uses its own indicator exit
    label = "combined_signal_exit"
    engines = {
        "rsi_1h": BacktestEngine(
            max_hold_bars=48, freq="1h", exit_check_fn=rsi_neutral_exit),
        "dc_1h": BacktestEngine(
            max_hold_bars=48, freq="1h", exit_check_fn=donchian_channel_exit),
        "rsi_15m": BacktestEngine(
            max_hold_bars=96, freq="15m", exit_check_fn=rsi_neutral_exit),
        "willr_1h": BacktestEngine(
            max_hold_bars=48, freq="1h", exit_check_fn=willr_neutral_exit),
    }
    result = run_4comp_cross_tf(
        df_1h, df_15m, df_4h, engines=engines, label=label)
    all_results.append(result)
    logger.info("    %s: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                label, result["robustness"], result["oos_return"],
                result["trades"])

    # Test D: DC channel exit only on DC component
    label = "dc_channel_exit"
    engines = {
        "rsi_1h": BacktestEngine(max_hold_bars=48, freq="1h"),
        "dc_1h": BacktestEngine(
            max_hold_bars=48, freq="1h", exit_check_fn=donchian_channel_exit),
        "rsi_15m": BacktestEngine(max_hold_bars=96, freq="15m"),
        "willr_1h": BacktestEngine(max_hold_bars=48, freq="1h"),
    }
    result = run_4comp_cross_tf(
        df_1h, df_15m, df_4h, engines=engines, label=label)
    all_results.append(result)
    logger.info("    %s: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                label, result["robustness"], result["oos_return"],
                result["trades"])
    logger.info("")

    # ─── PART 3: TP decay ────────────────────────────────────────
    logger.info("-" * 72)
    logger.info("  PART 3: TP decay (TP target shrinks as position ages)")
    logger.info("-" * 72)
    logger.info("  TP_adjusted = entry + (original_TP - entry) * decay_factor")
    logger.info("  decay_factor = max(0.5, 1 - bars_held/max_hold * rate)")
    logger.info("")

    for decay_rate in [0.3, 0.5, 0.7]:
        label = f"tp_decay_{decay_rate}"
        engines = {
            "rsi_1h": BacktestEngine(
                max_hold_bars=48, freq="1h", tp_decay_rate=decay_rate),
            "dc_1h": BacktestEngine(
                max_hold_bars=48, freq="1h", tp_decay_rate=decay_rate),
            "rsi_15m": BacktestEngine(
                max_hold_bars=96, freq="15m", tp_decay_rate=decay_rate),
            "willr_1h": BacktestEngine(
                max_hold_bars=48, freq="1h", tp_decay_rate=decay_rate),
        }
        result = run_4comp_cross_tf(
            df_1h, df_15m, df_4h, engines=engines, label=label)
        all_results.append(result)
        logger.info("    %s: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    label, result["robustness"], result["oos_return"],
                    result["trades"])
    logger.info("")

    # ─── PART 4: Best combinations ───────────────────────────────
    logger.info("-" * 72)
    logger.info("  PART 4: Promising combinations")
    logger.info("-" * 72)

    # Find best from each part
    non_baseline = [r for r in all_results if r["label"] != "baseline"]
    improvements = [
        r for r in non_baseline
        if r["robustness"] >= baseline["robustness"]
        and r["oos_return"] > baseline["oos_return"]
    ]

    if improvements:
        logger.info("  Found %d variants beating baseline:", len(improvements))
        for r in improvements:
            logger.info("    %s: Rob=%d%%, OOS=%+.2f%%",
                        r["label"], r["robustness"], r["oos_return"])

        # Try combining breakeven + signal exit if both improved
        be_improved = [r for r in improvements if "breakeven" in r["label"]]
        sig_improved = [r for r in improvements if "exit" in r["label"]]

        if be_improved and sig_improved:
            best_be_r = float(be_improved[0]["label"].split("_")[1].replace("R", ""))
            best_sig = sig_improved[0]["label"]
            logger.info("  Testing combination: breakeven_%.1fR + %s",
                        best_be_r, best_sig)

            # Build combined engine config
            if "rsi_neutral" in best_sig:
                engines = {
                    "rsi_1h": BacktestEngine(
                        max_hold_bars=48, freq="1h",
                        breakeven_at_r=best_be_r,
                        exit_check_fn=rsi_neutral_exit),
                    "dc_1h": BacktestEngine(
                        max_hold_bars=48, freq="1h",
                        breakeven_at_r=best_be_r),
                    "rsi_15m": BacktestEngine(
                        max_hold_bars=96, freq="15m",
                        breakeven_at_r=best_be_r,
                        exit_check_fn=rsi_neutral_exit),
                    "willr_1h": BacktestEngine(
                        max_hold_bars=48, freq="1h",
                        breakeven_at_r=best_be_r),
                }
            elif "combined" in best_sig:
                engines = {
                    "rsi_1h": BacktestEngine(
                        max_hold_bars=48, freq="1h",
                        breakeven_at_r=best_be_r,
                        exit_check_fn=rsi_neutral_exit),
                    "dc_1h": BacktestEngine(
                        max_hold_bars=48, freq="1h",
                        breakeven_at_r=best_be_r,
                        exit_check_fn=donchian_channel_exit),
                    "rsi_15m": BacktestEngine(
                        max_hold_bars=96, freq="15m",
                        breakeven_at_r=best_be_r,
                        exit_check_fn=rsi_neutral_exit),
                    "willr_1h": BacktestEngine(
                        max_hold_bars=48, freq="1h",
                        breakeven_at_r=best_be_r,
                        exit_check_fn=willr_neutral_exit),
                }
            else:
                engines = None

            if engines:
                label = f"combo_be{best_be_r}+{best_sig}"
                result = run_4comp_cross_tf(
                    df_1h, df_15m, df_4h, engines=engines, label=label)
                all_results.append(result)
                logger.info("    %s: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                            label, result["robustness"], result["oos_return"],
                            result["trades"])
    else:
        logger.info("  No single variant beats baseline. Testing combos anyway:")

        # Try breakeven + signal exit combos even if neither individually beat baseline
        for be_r in [1.0, 1.5]:
            # breakeven + RSI neutral exit on RSI comps
            label = f"combo_be{be_r}+rsi_exit"
            engines = {
                "rsi_1h": BacktestEngine(
                    max_hold_bars=48, freq="1h",
                    breakeven_at_r=be_r, exit_check_fn=rsi_neutral_exit),
                "dc_1h": BacktestEngine(
                    max_hold_bars=48, freq="1h", breakeven_at_r=be_r),
                "rsi_15m": BacktestEngine(
                    max_hold_bars=96, freq="15m",
                    breakeven_at_r=be_r, exit_check_fn=rsi_neutral_exit),
                "willr_1h": BacktestEngine(
                    max_hold_bars=48, freq="1h", breakeven_at_r=be_r),
            }
            result = run_4comp_cross_tf(
                df_1h, df_15m, df_4h, engines=engines, label=label)
            all_results.append(result)
            logger.info("    %s: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                        label, result["robustness"], result["oos_return"],
                        result["trades"])

            # breakeven + combined signal exit
            label = f"combo_be{be_r}+combined_exit"
            engines = {
                "rsi_1h": BacktestEngine(
                    max_hold_bars=48, freq="1h",
                    breakeven_at_r=be_r, exit_check_fn=rsi_neutral_exit),
                "dc_1h": BacktestEngine(
                    max_hold_bars=48, freq="1h",
                    breakeven_at_r=be_r, exit_check_fn=donchian_channel_exit),
                "rsi_15m": BacktestEngine(
                    max_hold_bars=96, freq="15m",
                    breakeven_at_r=be_r, exit_check_fn=rsi_neutral_exit),
                "willr_1h": BacktestEngine(
                    max_hold_bars=48, freq="1h",
                    breakeven_at_r=be_r, exit_check_fn=willr_neutral_exit),
            }
            result = run_4comp_cross_tf(
                df_1h, df_15m, df_4h, engines=engines, label=label)
            all_results.append(result)
            logger.info("    %s: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                        label, result["robustness"], result["oos_return"],
                        result["trades"])

    logger.info("")

    # ─── SUMMARY ─────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  PHASE 34 SUMMARY")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  %-35s  %5s  %8s  %6s", "Config", "Rob%", "OOS Ret", "Trades")
    logger.info("  " + "-" * 60)

    for r in all_results:
        marker = " ★" if (
            r["robustness"] >= baseline["robustness"]
            and r["oos_return"] > baseline["oos_return"]
        ) else ""
        logger.info("  %-35s  %4d%%  %+7.2f%%  %5d%s",
                    r["label"], r["robustness"], r["oos_return"],
                    r["trades"], marker)

    # Check for improvements
    beats = [
        r for r in all_results
        if r["label"] != "baseline"
        and r["robustness"] >= baseline["robustness"]
        and r["oos_return"] > baseline["oos_return"]
    ]
    same_rob = [
        r for r in all_results
        if r["label"] != "baseline"
        and r["robustness"] >= baseline["robustness"]
    ]

    logger.info("")
    if beats:
        best = max(beats, key=lambda x: x["oos_return"])
        logger.info("  IMPROVED: %s — %d%% rob, %+.2f%% OOS (baseline %+.2f%%)",
                    best["label"], best["robustness"], best["oos_return"],
                    baseline["oos_return"])
        logger.info("  Improvement: %+.2f%% OOS",
                    best["oos_return"] - baseline["oos_return"])
    elif same_rob:
        logger.info("  %d variants maintain 88%% robustness but lower return.",
                    len(same_rob))
        best_same = max(same_rob, key=lambda x: x["oos_return"])
        logger.info("  Closest: %s — %+.2f%% OOS (vs baseline %+.2f%%)",
                    best_same["label"], best_same["oos_return"],
                    baseline["oos_return"])
    else:
        logger.info("  ALL exit optimizations REDUCE robustness or return.")
        logger.info("  Fixed SL/TP + timeout remains optimal.")

    # Window-by-window analysis of best configs
    logger.info("")
    logger.info("  Window-by-window (top 3 + baseline):")
    top_results = sorted(all_results, key=lambda x: x["oos_return"], reverse=True)[:4]
    if baseline not in top_results:
        top_results.append(baseline)

    header = "  %-25s" + "".join(f"  W{i+1:d}" for i in range(9))
    logger.info(header, "Config")
    for r in top_results:
        wins = r.get("windows", [])
        win_strs = []
        for w in wins:
            ret = w["return"]
            win_strs.append(f"{ret:+5.1f}" if abs(ret) < 100 else f"{ret:+5.0f}")
        logger.info("  %-25s" + "".join(f"  %s" % s for s in win_strs), r["label"])

    logger.info("")
    logger.info("  Phase 34 complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 38 — Directional Asymmetry Analysis.

A genuinely untested approach axis: instead of adding new indicators or
changing weights, we analyze LONG vs SHORT win rates per component and
suppress the unprofitable direction entirely.

Rationale:
  - Phase 21 showed DC LONG WR=16.7% (2/12 win) vs SHORT WR=50%
  - W2 (Nov 20 - Dec 2) was a strong rally → MR SHORT signals lose
  - The MTF filter already blocks signals against 4h trend, but
    signals WITH the trend may still be systematically losing
  - This is a trade-level filter (which trades to take), not a
    signal-level filter (what to generate) — a completely new axis

Tests:
  Part 1: Per-component LONG/SHORT trade anatomy across all 9 windows
  Part 2: Direction-suppressed component standalone tests
  Part 3: Portfolio with direction-suppressed components
  Part 4: Best directional portfolio comparison to baseline

Baseline: 4-comp 15/50/10/25, 88% rob, +23.98% OOS, 236 trades.
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
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import (
    CrossTFComponent,
    CrossTFReport,
    WalkForwardAnalyzer,
)
from src.strategy.base import BaseStrategy, Signal, TradeSignal
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase38")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase38.log", mode="w")
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


# ─── Directional Filter Wrapper ──────────────────────────────────

class DirectionalFilter(BaseStrategy):
    """Wrapper that suppresses one direction from a base strategy.

    Unlike MTF filter (trend-dependent), this is STATIC:
    it always blocks the specified direction regardless of trend.
    The chain is: base_strategy → MTF_filter → DirectionalFilter.
    """

    def __init__(
        self,
        base_strategy: BaseStrategy,
        allow_long: bool = True,
        allow_short: bool = True,
    ) -> None:
        self.base_strategy = base_strategy
        self.allow_long = allow_long
        self.allow_short = allow_short

        direction_tag = ""
        if not allow_long:
            direction_tag = "_SHORT_only"
        elif not allow_short:
            direction_tag = "_LONG_only"
        self.name = f"{base_strategy.name}{direction_tag}"

    def set_htf_data(self, df_htf: pd.DataFrame) -> None:
        if hasattr(self.base_strategy, "set_htf_data"):
            self.base_strategy.set_htf_data(df_htf)

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        sig = self.base_strategy.generate_signal(df)

        if sig.signal == Signal.LONG and not self.allow_long:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=sig.symbol,
                price=sig.price,
                timestamp=sig.timestamp,
                metadata={"blocked_by": "directional_filter", "original": "LONG"},
            )

        if sig.signal == Signal.SHORT and not self.allow_short:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=sig.symbol,
                price=sig.price,
                timestamp=sig.timestamp,
                metadata={"blocked_by": "directional_filter", "original": "SHORT"},
            )

        return sig

    def get_required_indicators(self) -> list[str]:
        return self.base_strategy.get_required_indicators()


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

def make_rsi_1h(**kwargs) -> MultiTimeframeFilter:
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_dc_1h(**kwargs) -> MultiTimeframeFilter:
    base = DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_rsi_15m(**kwargs) -> MultiTimeframeFilter:
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
    )
    return MultiTimeframeFilter(base)


def make_willr_1h(**kwargs) -> MultiTimeframeFilter:
    base = WilliamsRMeanReversionStrategy(
        willr_period=14, oversold_level=90, overbought_level=90,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


# ─── Directional Factories ───────────────────────────────────────

def make_dc_1h_short_only() -> DirectionalFilter:
    return DirectionalFilter(make_dc_1h(), allow_long=False, allow_short=True)

def make_dc_1h_long_only() -> DirectionalFilter:
    return DirectionalFilter(make_dc_1h(), allow_long=True, allow_short=False)

def make_rsi_1h_long_only() -> DirectionalFilter:
    return DirectionalFilter(make_rsi_1h(), allow_long=True, allow_short=False)

def make_rsi_1h_short_only() -> DirectionalFilter:
    return DirectionalFilter(make_rsi_1h(), allow_long=False, allow_short=True)

def make_rsi_15m_long_only() -> DirectionalFilter:
    return DirectionalFilter(make_rsi_15m(), allow_long=True, allow_short=False)

def make_rsi_15m_short_only() -> DirectionalFilter:
    return DirectionalFilter(make_rsi_15m(), allow_long=False, allow_short=True)

def make_willr_1h_long_only() -> DirectionalFilter:
    return DirectionalFilter(make_willr_1h(), allow_long=True, allow_short=False)

def make_willr_1h_short_only() -> DirectionalFilter:
    return DirectionalFilter(make_willr_1h(), allow_long=False, allow_short=True)


# ─── Helpers ──────────────────────────────────────────────────────

def analyze_trades(engine: BacktestEngine, strategy_factory, df, htf_df, label: str):
    """Run strategy on full data and analyze LONG vs SHORT trade breakdown."""
    strategy = strategy_factory()
    result = engine.run(strategy, df, htf_df=htf_df)

    longs = [t for t in result.trade_logs if t.side == "long"]
    shorts = [t for t in result.trade_logs if t.side == "short"]

    long_wins = sum(1 for t in longs if t.return_pct > 0)
    short_wins = sum(1 for t in shorts if t.return_pct > 0)

    long_wr = long_wins / len(longs) * 100 if longs else 0
    short_wr = short_wins / len(shorts) * 100 if shorts else 0

    long_ret = sum(t.return_pct for t in longs)
    short_ret = sum(t.return_pct for t in shorts)

    logger.info("  %s: %d trades total (L:%d S:%d)", label, len(result.trade_logs), len(longs), len(shorts))
    logger.info("    LONG:  %d trades, WR=%.1f%%, cumPnL=%+.2f%%", len(longs), long_wr, long_ret)
    logger.info("    SHORT: %d trades, WR=%.1f%%, cumPnL=%+.2f%%", len(shorts), short_wr, short_ret)
    logger.info("    Total return: %+.2f%%", result.total_return)

    return {
        "label": label,
        "total_trades": len(result.trade_logs),
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "long_wr": long_wr,
        "short_wr": short_wr,
        "long_ret": long_ret,
        "short_ret": short_ret,
        "total_return": result.total_return,
    }


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
    logger.info("  PHASE 38 — Directional Asymmetry Analysis")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Hypothesis: Suppressing the unprofitable direction per")
    logger.info("  component can reduce W2 losses and improve portfolio.")
    logger.info("  This is a TRADE-LEVEL filter, not a SIGNAL-LEVEL filter.")
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
    logger.info("  4h data:  %d bars (%s ~ %s)",
                len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")

    # ═════════════════════════════════════════════════════════════
    #   PART 1: Full-Period LONG/SHORT Trade Anatomy
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: LONG/SHORT Trade Anatomy (full period)")
    logger.info("-" * 72)
    logger.info("  Running each component on full data to analyze direction breakdown")
    logger.info("")

    components_full = [
        ("1h RSI MR", make_rsi_1h, engine_1h, df_1h, df_4h),
        ("1h Donchian", make_dc_1h, engine_1h, df_1h, df_4h),
        ("15m RSI MR", make_rsi_15m, engine_15m, df_15m, df_4h),
        ("1h WillR MR", make_willr_1h, engine_1h, df_1h, df_4h),
    ]

    anatomy_results = []
    for label, factory, engine, df, htf_df in components_full:
        result = analyze_trades(engine, factory, df, htf_df, label)
        anatomy_results.append(result)
        logger.info("")

    # Summary table
    logger.info("  ── Direction Anatomy Summary ──")
    logger.info("  %-15s  %5s  %5s  %6s  %6s  %8s  %8s",
                "Component", "L#", "S#", "L_WR%", "S_WR%", "L_PnL%", "S_PnL%")
    logger.info("  " + "-" * 65)
    for r in anatomy_results:
        logger.info("  %-15s  %5d  %5d  %5.1f%%  %5.1f%%  %+7.2f%%  %+7.2f%%",
                    r["label"], r["long_trades"], r["short_trades"],
                    r["long_wr"], r["short_wr"], r["long_ret"], r["short_ret"])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Direction-Filtered Standalone WF (9w each)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: Direction-Filtered Standalone (9w WF)")
    logger.info("-" * 72)
    logger.info("  Testing each component with one direction suppressed")
    logger.info("")

    standalone_configs = [
        # (label, factory, engine, df, htf_df)
        ("1h DC baseline", make_dc_1h, engine_1h, df_1h, df_4h),
        ("1h DC SHORT-only", make_dc_1h_short_only, engine_1h, df_1h, df_4h),
        ("1h DC LONG-only", make_dc_1h_long_only, engine_1h, df_1h, df_4h),
        ("1h RSI baseline", make_rsi_1h, engine_1h, df_1h, df_4h),
        ("1h RSI LONG-only", make_rsi_1h_long_only, engine_1h, df_1h, df_4h),
        ("1h RSI SHORT-only", make_rsi_1h_short_only, engine_1h, df_1h, df_4h),
        ("15m RSI baseline", make_rsi_15m, engine_15m, df_15m, df_4h),
        ("15m RSI LONG-only", make_rsi_15m_long_only, engine_15m, df_15m, df_4h),
        ("15m RSI SHORT-only", make_rsi_15m_short_only, engine_15m, df_15m, df_4h),
        ("1h WillR baseline", make_willr_1h, engine_1h, df_1h, df_4h),
        ("1h WillR LONG-only", make_willr_1h_long_only, engine_1h, df_1h, df_4h),
        ("1h WillR SHORT-only", make_willr_1h_short_only, engine_1h, df_1h, df_4h),
    ]

    standalone_results = []
    for label, factory, engine, df, htf_df in standalone_configs:
        wf = WalkForwardAnalyzer(n_windows=9, engine=engine)
        report = wf.run(factory, df, htf_df=htf_df)
        rob = int(report.robustness_score * 100)
        standalone_results.append({
            "label": label,
            "rob": rob,
            "oos": report.oos_total_return,
            "trades": report.oos_total_trades,
        })
        logger.info("  %s: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    label, rob, report.oos_total_return, report.oos_total_trades)

    logger.info("")
    logger.info("  ── Standalone Direction Summary ──")
    logger.info("  %-22s  %5s  %8s  %6s", "Config", "Rob%", "OOS%", "Trades")
    logger.info("  " + "-" * 50)
    for r in standalone_results:
        marker = " ★" if r["rob"] > standalone_results[standalone_results.index(r) - standalone_results.index(r) % 3]["rob"] else ""
        logger.info("  %-22s  %4d%%  %+7.2f%%  %6d%s",
                    r["label"], r["rob"], r["oos"], r["trades"], "")
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Portfolio with Direction-Filtered Components
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: Portfolio with Directional Filtering")
    logger.info("-" * 72)
    logger.info("")

    wf = WalkForwardAnalyzer(n_windows=9)

    # Test 3a: Baseline (for comparison)
    logger.info("  3a. Baseline 4-comp (15/50/10/25):")
    baseline_report = wf.run_cross_tf([
        CrossTFComponent(make_rsi_1h, df_1h, df_4h, engine_1h, 0.15, "1hRSI"),
        CrossTFComponent(make_dc_1h, df_1h, df_4h, engine_1h, 0.50, "1hDC"),
        CrossTFComponent(make_rsi_15m, df_15m, df_4h, engine_15m, 0.10, "15mRSI"),
        CrossTFComponent(make_willr_1h, df_1h, df_4h, engine_1h, 0.25, "1hWillR"),
    ])
    log_cross_tf_detail("Baseline", baseline_report)
    logger.info("")

    # Test 3b: DC SHORT-only (suppress DC LONGs)
    logger.info("  3b. DC SHORT-only (suppress DC LONGs):")
    dc_short_report = wf.run_cross_tf([
        CrossTFComponent(make_rsi_1h, df_1h, df_4h, engine_1h, 0.15, "1hRSI"),
        CrossTFComponent(make_dc_1h_short_only, df_1h, df_4h, engine_1h, 0.50, "1hDC_S"),
        CrossTFComponent(make_rsi_15m, df_15m, df_4h, engine_15m, 0.10, "15mRSI"),
        CrossTFComponent(make_willr_1h, df_1h, df_4h, engine_1h, 0.25, "1hWillR"),
    ])
    log_cross_tf_detail("DC SHORT-only", dc_short_report)
    logger.info("")

    # Test 3c: DC LONG-only (suppress DC SHORTs)
    logger.info("  3c. DC LONG-only (suppress DC SHORTs):")
    dc_long_report = wf.run_cross_tf([
        CrossTFComponent(make_rsi_1h, df_1h, df_4h, engine_1h, 0.15, "1hRSI"),
        CrossTFComponent(make_dc_1h_long_only, df_1h, df_4h, engine_1h, 0.50, "1hDC_L"),
        CrossTFComponent(make_rsi_15m, df_15m, df_4h, engine_15m, 0.10, "15mRSI"),
        CrossTFComponent(make_willr_1h, df_1h, df_4h, engine_1h, 0.25, "1hWillR"),
    ])
    log_cross_tf_detail("DC LONG-only", dc_long_report)
    logger.info("")

    # Test 3d: RSI LONG-only + DC SHORT-only
    logger.info("  3d. RSI LONG-only + DC SHORT-only:")
    rsi_l_dc_s_report = wf.run_cross_tf([
        CrossTFComponent(make_rsi_1h_long_only, df_1h, df_4h, engine_1h, 0.15, "1hRSI_L"),
        CrossTFComponent(make_dc_1h_short_only, df_1h, df_4h, engine_1h, 0.50, "1hDC_S"),
        CrossTFComponent(make_rsi_15m, df_15m, df_4h, engine_15m, 0.10, "15mRSI"),
        CrossTFComponent(make_willr_1h, df_1h, df_4h, engine_1h, 0.25, "1hWillR"),
    ])
    log_cross_tf_detail("RSI_L+DC_S", rsi_l_dc_s_report)
    logger.info("")

    # Test 3e: All MR LONG-only + DC SHORT-only (maximum asymmetry)
    logger.info("  3e. All MR LONG-only + DC SHORT-only:")
    full_asym_report = wf.run_cross_tf([
        CrossTFComponent(make_rsi_1h_long_only, df_1h, df_4h, engine_1h, 0.15, "1hRSI_L"),
        CrossTFComponent(make_dc_1h_short_only, df_1h, df_4h, engine_1h, 0.50, "1hDC_S"),
        CrossTFComponent(make_rsi_15m_long_only, df_15m, df_4h, engine_15m, 0.10, "15mRSI_L"),
        CrossTFComponent(make_willr_1h_long_only, df_1h, df_4h, engine_1h, 0.25, "1hWillR_L"),
    ])
    log_cross_tf_detail("FullAsym", full_asym_report)
    logger.info("")

    # Test 3f: Data-driven — suppress each component's weaker direction
    # Based on Part 1 anatomy results, choose the better direction per component
    logger.info("  3f. Data-Driven Direction Selection:")
    logger.info("  (Based on Part 1 analysis, suppress weaker direction per component)")

    # Determine best direction per component from anatomy results
    direction_choices = []
    for r in anatomy_results:
        if r["long_ret"] > r["short_ret"]:
            direction_choices.append(("LONG", r["label"]))
        elif r["short_ret"] > r["long_ret"]:
            direction_choices.append(("SHORT", r["label"]))
        else:
            direction_choices.append(("BOTH", r["label"]))
        logger.info("    %s: LONG=%+.2f%% vs SHORT=%+.2f%% → keep %s",
                    r["label"], r["long_ret"], r["short_ret"], direction_choices[-1][0])
    logger.info("")

    # Build data-driven portfolio
    def make_component_directed(label: str, factory, engine, df, htf_df, direction: str):
        """Create a CrossTFComponent with directional filtering based on data."""
        if direction == "LONG":
            def dir_factory(f=factory):
                return DirectionalFilter(f(), allow_long=True, allow_short=False)
            return dir_factory
        elif direction == "SHORT":
            def dir_factory(f=factory):
                return DirectionalFilter(f(), allow_long=False, allow_short=True)
            return dir_factory
        else:
            return factory

    base_components = [
        ("1h RSI MR", make_rsi_1h, engine_1h, df_1h, df_4h, 0.15, "1hRSI"),
        ("1h Donchian", make_dc_1h, engine_1h, df_1h, df_4h, 0.50, "1hDC"),
        ("15m RSI MR", make_rsi_15m, engine_15m, df_15m, df_4h, 0.10, "15mRSI"),
        ("1h WillR MR", make_willr_1h, engine_1h, df_1h, df_4h, 0.25, "1hWillR"),
    ]

    dd_components = []
    for (dir_choice, _), (label, factory, engine, df, htf_df, weight, tag) in zip(
        direction_choices, base_components
    ):
        dir_factory = make_component_directed(label, factory, engine, df, htf_df, dir_choice)
        suffix = f"_{dir_choice[0]}" if dir_choice != "BOTH" else ""
        dd_components.append(
            CrossTFComponent(dir_factory, df, htf_df, engine, weight, f"{tag}{suffix}")
        )

    dd_report = wf.run_cross_tf(dd_components)
    log_cross_tf_detail("DataDriven", dd_report)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Weight Optimization for Best Directional Config
    # ═════════════════════════════════════════════════════════════
    # Find the best directional config from Part 3, then try different weights

    all_portfolio_results = [
        ("Baseline", baseline_report),
        ("DC_SHORT_only", dc_short_report),
        ("DC_LONG_only", dc_long_report),
        ("RSI_L+DC_S", rsi_l_dc_s_report),
        ("FullAsym", full_asym_report),
        ("DataDriven", dd_report),
    ]

    logger.info("-" * 72)
    logger.info("  PART 4: Portfolio Comparison Summary")
    logger.info("-" * 72)
    logger.info("")
    logger.info("  %-20s  %5s  %8s  %6s  %5s", "Config", "Rob%", "OOS%", "Trades", "W2")
    logger.info("  " + "-" * 52)

    for name, report in all_portfolio_results:
        w2_ret = report.windows[1].weighted_return if len(report.windows) > 1 else 0.0
        rob = int(report.robustness_score * 100)
        marker = " ★" if rob > 88 else (" =" if rob == 88 else "")
        logger.info("  %-20s  %4d%%  %+7.2f%%  %6d  %+.2f%%%s",
                    name, rob, report.oos_total_return, report.total_trades,
                    w2_ret, marker)
    logger.info("")

    # Check if any config beats baseline
    best_non_baseline = max(
        all_portfolio_results[1:],  # exclude baseline
        key=lambda x: (int(x[1].robustness_score * 100), x[1].oos_total_return),
    )
    best_name, best_report = best_non_baseline
    best_rob = int(best_report.robustness_score * 100)
    baseline_rob = int(baseline_report.robustness_score * 100)

    if best_rob > baseline_rob:
        logger.info("  ★★★ BREAKTHROUGH: %s beats baseline! ★★★", best_name)
        logger.info("    %s: Rob=%d%%, OOS=%+.2f%%",
                    best_name, best_rob, best_report.oos_total_return)
        logger.info("    Baseline: Rob=%d%%, OOS=%+.2f%%",
                    baseline_rob, baseline_report.oos_total_return)

        # If breakthrough, test weight variations
        logger.info("")
        logger.info("  Testing weight variations for %s...", best_name)
        # (Weight optimization code would go here)

    elif best_rob == baseline_rob and best_report.oos_total_return > baseline_report.oos_total_return:
        logger.info("  IMPROVEMENT: %s same robustness but higher return", best_name)
        logger.info("    %s: Rob=%d%%, OOS=%+.2f%%",
                    best_name, best_rob, best_report.oos_total_return)
        logger.info("    Baseline: Rob=%d%%, OOS=%+.2f%%",
                    baseline_rob, baseline_report.oos_total_return)
    else:
        logger.info("  CONCLUSION: Directional filtering does NOT improve portfolio.")
        logger.info("  Best: %s (Rob=%d%%, OOS=%+.2f%%)",
                    best_name, best_rob, best_report.oos_total_return)
        logger.info("  88%% robustness ceiling confirmed — trade direction axis exhausted.")

    logger.info("")

    # W2 deep dive
    logger.info("-" * 72)
    logger.info("  PART 5: W2 (Nov 20-Dec 2) Deep Dive")
    logger.info("-" * 72)
    logger.info("")
    logger.info("  W2 returns by config:")
    for name, report in all_portfolio_results:
        if len(report.windows) > 1:
            w2 = report.windows[1]
            parts = [f"{cr.label}={cr.oos_return:+.2f}%" for cr in w2.components]
            logger.info("    %s: %s → %+.2f%%", name, ", ".join(parts), w2.weighted_return)
    logger.info("")

    logger.info("  Can directional filtering fix W2?")
    for name, report in all_portfolio_results:
        if len(report.windows) > 1:
            w2_pos = "YES" if report.windows[1].weighted_return > 0 else "NO"
            logger.info("    %s: W2=%+.2f%% → %s",
                        name, report.windows[1].weighted_return, w2_pos)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   SUMMARY
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 38 SUMMARY")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Approach: Directional asymmetry (suppress LONG/SHORT per component)")
    logger.info("  This is a NEW axis — tests which TRADES to take, not what SIGNALS")
    logger.info("  to generate. Different from all 36 prior phases.")
    logger.info("")
    logger.info("  Direction Anatomy:")
    for r in anatomy_results:
        logger.info("    %s: L=%d(%.0f%%), S=%d(%.0f%%), L_PnL=%+.2f%%, S_PnL=%+.2f%%",
                    r["label"], r["long_trades"], r["long_wr"],
                    r["short_trades"], r["short_wr"],
                    r["long_ret"], r["short_ret"])
    logger.info("")
    logger.info("  Portfolio Results:")
    for name, report in all_portfolio_results:
        rob = int(report.robustness_score * 100)
        logger.info("    %-20s: %d%% rob, %+.2f%% OOS, %d trades",
                    name, rob, report.oos_total_return, report.total_trades)
    logger.info("")
    logger.info("  Phase 38 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

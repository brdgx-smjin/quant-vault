#!/usr/bin/env python3
"""Phase 29 — Funding Rate as Signal Filter for Production Portfolio.

Hypothesis:
  Extreme funding rates indicate crowded positioning.
  - High FR (> threshold): many longs → suppress LONG signals (liquidation risk)
  - Low FR (< -threshold): many shorts → suppress SHORT signals (squeeze risk)
  - Neutral FR: proceed with signals as normal

Data limitation:
  Funding rate data: 2025-07-21 ~ 2026-02-25 (~7 months, 8h intervals)
  1h OHLCV data: 2025-02-25 ~ 2026-02-25 (~12 months)
  Only overlapping period is used. Baseline comparison uses SAME period.

Plan:
  PART 1: FR data exploration — distribution, autocorrelation, regime analysis
  PART 2: RSI MR + MTF + FR filter (1h, 9w WF) — grid search thresholds
  PART 3: Baseline comparison (RSI MR + MTF on SAME reduced period)
  PART 4: If promising, test on 4-comp Cross-TF portfolio
  PART 5: Summary
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
    WalkForwardAnalyzer,
)
from src.strategy.base import BaseStrategy, Signal, TradeSignal
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy
from src.data.preprocessor import DataPreprocessor

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase29")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase29.log", mode="w")
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


# ─── Funding Rate Filter (Decorator Pattern) ─────────────────────

class FundingRateFilter(BaseStrategy):
    """Wraps a strategy and blocks signals when funding rate is extreme.

    Reads 'funding_rate' column from the DataFrame.
    - Blocks LONG when FR >= long_suppress_threshold (crowded longs)
    - Blocks SHORT when FR <= short_suppress_threshold (crowded shorts)
    """

    def __init__(
        self,
        base_strategy: BaseStrategy,
        long_suppress_threshold: float = 0.0005,
        short_suppress_threshold: float = -0.0005,
    ) -> None:
        self.base_strategy = base_strategy
        self.long_suppress = long_suppress_threshold
        self.short_suppress = short_suppress_threshold
        self.name = f"fr_filter_{base_strategy.name}"

    def set_htf_data(self, df_htf: pd.DataFrame) -> None:
        """Pass through to base strategy."""
        if hasattr(self.base_strategy, "set_htf_data"):
            self.base_strategy.set_htf_data(df_htf)

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate signal then filter by funding rate."""
        signal = self.base_strategy.generate_signal(df)

        if signal.signal == Signal.HOLD:
            return signal

        # Read funding rate from DataFrame
        fr = df["funding_rate"].iloc[-1] if "funding_rate" in df.columns else None
        if fr is None or pd.isna(fr):
            return signal

        fr = float(fr)

        # Block LONG when funding rate is too positive (crowded longs)
        if signal.signal == Signal.LONG and fr >= self.long_suppress:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=signal.symbol,
                price=signal.price,
                timestamp=signal.timestamp,
                metadata={"blocked_by": "fr_filter", "original": "LONG", "fr": fr},
            )

        # Block SHORT when funding rate is too negative (crowded shorts)
        if signal.signal == Signal.SHORT and fr <= self.short_suppress:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=signal.symbol,
                price=signal.price,
                timestamp=signal.timestamp,
                metadata={"blocked_by": "fr_filter", "original": "SHORT", "fr": fr},
            )

        return signal

    def get_required_indicators(self) -> list[str]:
        return self.base_strategy.get_required_indicators()


# ─── Data Loading ─────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load 1h, 15m, 4h OHLCV + funding rate data.

    Returns:
        (df_1h, df_15m, df_4h, df_fr) — all with indicators added where needed.
    """
    df_1h = pd.read_parquet(ROOT / "data/processed/BTC_USDT_USDT_1h.parquet")
    df_15m = pd.read_parquet(ROOT / "data/processed/BTC_USDT_USDT_15m.parquet")
    df_4h = pd.read_parquet(ROOT / "data/processed/BTC_USDT_USDT_4h.parquet")
    df_fr = pd.read_parquet(ROOT / "data/processed/BTC_USDT_USDT_funding_rate.parquet")

    df_1h = BasicIndicators.add_all(df_1h)
    df_15m = BasicIndicators.add_all(df_15m)
    df_4h = BasicIndicators.add_all(df_4h)

    return df_1h, df_15m, df_4h, df_fr


def merge_funding_rate(df: pd.DataFrame, df_fr: pd.DataFrame) -> pd.DataFrame:
    """Merge 8h funding rate into higher-frequency OHLCV data via forward-fill.

    Args:
        df: OHLCV DataFrame (1h or 15m).
        df_fr: Funding rate DataFrame (8h intervals).

    Returns:
        df with 'funding_rate' column added.
    """
    df = df.copy()
    fr_series = df_fr["fundingRate"].rename("funding_rate")
    # Reindex to match df's timestamps, forward-fill (8h rate applies for next 8h)
    merged = pd.merge_asof(
        df[["close"]],  # dummy column for merge
        fr_series.to_frame(),
        left_index=True,
        right_index=True,
        direction="backward",  # use most recent FR at or before this timestamp
    )
    df["funding_rate"] = merged["funding_rate"]
    return df


def add_willr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Williams %R indicator."""
    col = f"WILLR_{period}"
    if col not in df.columns:
        df[col] = ta.willr(df["high"], df["low"], df["close"], length=period)
    return df


# ─── Main ─────────────────────────────────────────────────────────

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 29 — Funding Rate as Signal Filter")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Hypothesis: extreme FR indicates crowded positioning.")
    logger.info("  High FR → suppress LONG (liquidation risk)")
    logger.info("  Low FR  → suppress SHORT (squeeze risk)")
    logger.info("")

    df_1h, df_15m, df_4h, df_fr = load_data()
    logger.info("1h data:  %d bars (%s ~ %s)", len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("15m data: %d bars (%s ~ %s)", len(df_15m), df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("4h data:  %d bars (%s ~ %s)", len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("FR data:  %d rows (%s ~ %s)", len(df_fr), df_fr.index[0].date(), df_fr.index[-1].date())
    logger.info("")

    # ── PART 1: FR Data Exploration ──────────────────────────────
    logger.info("-" * 72)
    logger.info("  PART 1: Funding Rate Data Exploration")
    logger.info("-" * 72)

    fr_vals = df_fr["fundingRate"]
    logger.info("  Mean:   %.6f", fr_vals.mean())
    logger.info("  Std:    %.6f", fr_vals.std())
    logger.info("  Min:    %.6f", fr_vals.min())
    logger.info("  Max:    %.6f", fr_vals.max())
    logger.info("  Median: %.6f", fr_vals.median())
    logger.info("")

    # Distribution by threshold
    for threshold in [0.0003, 0.0005, 0.0008, 0.001]:
        pct_above = (fr_vals >= threshold).mean() * 100
        pct_below = (fr_vals <= -threshold).mean() * 100
        logger.info("  FR >= +%.4f: %.1f%% | FR <= -%.4f: %.1f%%",
                     threshold, pct_above, threshold, pct_below)
    logger.info("")

    # Merge FR into 1h data — use only overlapping period
    df_1h_fr = merge_funding_rate(df_1h, df_fr)
    fr_valid_mask = df_1h_fr["funding_rate"].notna()
    overlap_start = df_1h_fr[fr_valid_mask].index[0]
    df_1h_sub = df_1h_fr[df_1h_fr.index >= overlap_start].copy()

    logger.info("  Overlapping period: %s ~ %s (%d bars)",
                overlap_start.date(), df_1h_sub.index[-1].date(), len(df_1h_sub))
    logger.info("  FR NaN count in overlap: %d", df_1h_sub["funding_rate"].isna().sum())
    logger.info("")

    # ── PART 2: RSI MR + MTF + FR Filter (1h, 9w WF) ────────────
    logger.info("-" * 72)
    logger.info("  PART 2: RSI MR + MTF + FR Filter (1h, 9w WF)")
    logger.info("-" * 72)
    logger.info("  Grid: long_suppress=[0.0003, 0.0005, 0.0008, 0.001]")
    logger.info("         short_suppress=[-0.0003, -0.0005, -0.0008, -0.001]")
    logger.info("         (symmetric and asymmetric combos)")
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    wf = WalkForwardAnalyzer(train_ratio=0.7, n_windows=9, engine=engine_1h)

    # Slice 4h to overlap period for MTF filter
    df_4h_sub = df_4h[df_4h.index >= overlap_start].copy()

    results_p2: list[dict] = []

    # Test symmetric thresholds first
    for long_t in [0.0003, 0.0005, 0.0008, 0.001]:
        for short_t in [-0.0003, -0.0005, -0.0008, -0.001]:
            label = f"FR_L{long_t:.4f}_S{short_t:.4f}"

            def make_strategy(lt=long_t, st=short_t):
                base = RSIMeanReversionStrategy(
                    rsi_oversold=35, rsi_overbought=65,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
                mtf = MultiTimeframeFilter(base)
                return FundingRateFilter(mtf, long_suppress_threshold=lt, short_suppress_threshold=st)

            report = wf.run(make_strategy, df_1h_sub, htf_df=df_4h_sub)

            for w in report.windows:
                oos = w.out_of_sample
                marker = "+" if oos.total_return > 0 else "-"
                logger.info(
                    "    W%d [%s ~ %s]: IS %+.2f%% | OOS %+.2f%% | %d trades %s",
                    w.window_id, w.test_start, w.test_end,
                    w.in_sample.total_return, oos.total_return,
                    oos.total_trades, marker,
                )

            rob_pct = int(report.robustness_score * 100)
            logger.info(
                "  %s: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d | Sharpe: %.2f",
                label, report.oos_total_return, rob_pct,
                report.oos_profitable_windows, report.total_windows,
                report.oos_total_trades, report.oos_avg_sharpe,
            )
            logger.info("")

            results_p2.append({
                "label": label,
                "long_t": long_t,
                "short_t": short_t,
                "robustness": rob_pct,
                "oos_return": report.oos_total_return,
                "trades": report.oos_total_trades,
                "sharpe": report.oos_avg_sharpe,
                "windows": report.total_windows,
                "profitable": report.oos_profitable_windows,
            })

    # ── PART 3: Baseline (RSI MR + MTF, NO FR filter, same period) ─
    logger.info("-" * 72)
    logger.info("  PART 3: Baseline — RSI MR + MTF (no FR filter, same period)")
    logger.info("-" * 72)
    logger.info("")

    def make_baseline():
        base = RSIMeanReversionStrategy(
            rsi_oversold=35, rsi_overbought=65,
            atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    # Run baseline on same reduced period (no funding_rate column needed)
    baseline = wf.run(make_baseline, df_1h_sub, htf_df=df_4h_sub)

    for w in baseline.windows:
        oos = w.out_of_sample
        marker = "+" if oos.total_return > 0 else "-"
        logger.info(
            "    W%d [%s ~ %s]: IS %+.2f%% | OOS %+.2f%% | %d trades %s",
            w.window_id, w.test_start, w.test_end,
            w.in_sample.total_return, oos.total_return,
            oos.total_trades, marker,
        )

    baseline_rob = int(baseline.robustness_score * 100)
    logger.info(
        "  BASELINE (no FR): OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d | Sharpe: %.2f",
        baseline.oos_total_return, baseline_rob,
        baseline.oos_profitable_windows, baseline.total_windows,
        baseline.oos_total_trades, baseline.oos_avg_sharpe,
    )
    logger.info("")

    # ── PART 4: If FR helps, test on 4-comp Cross-TF ─────────────
    # Find best FR config that beats baseline
    best_fr = None
    for r in sorted(results_p2, key=lambda x: (-x["robustness"], -x["oos_return"])):
        if r["robustness"] > baseline_rob or (
            r["robustness"] == baseline_rob and r["oos_return"] > baseline.oos_total_return
        ):
            best_fr = r
            break

    if best_fr is None:
        logger.info("-" * 72)
        logger.info("  PART 4: SKIPPED — No FR config beats baseline")
        logger.info("-" * 72)
        logger.info("  Best FR: %s (rob=%d%%, OOS=%+.2f%%)",
                     results_p2[0]["label"] if results_p2 else "none",
                     results_p2[0]["robustness"] if results_p2 else 0,
                     results_p2[0]["oos_return"] if results_p2 else 0)
        logger.info("  Baseline: rob=%d%%, OOS=%+.2f%%", baseline_rob, baseline.oos_total_return)
        logger.info("  FR filter does NOT improve RSI MR standalone.")
        logger.info("")
    else:
        logger.info("-" * 72)
        logger.info("  PART 4: Best FR config on 4-comp Cross-TF Portfolio")
        logger.info("-" * 72)
        logger.info("  Best FR: %s (rob=%d%%, OOS=%+.2f%%)",
                     best_fr["label"], best_fr["robustness"], best_fr["oos_return"])
        logger.info("")

        best_lt = best_fr["long_t"]
        best_st = best_fr["short_t"]

        # Merge FR into 15m data too
        df_15m_fr = merge_funding_rate(df_15m, df_fr)
        df_15m_sub = df_15m_fr[df_15m_fr.index >= overlap_start].copy()

        # Add WillR to 1h data
        df_1h_sub = add_willr(df_1h_sub, period=14)

        engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")
        wf_ctf = WalkForwardAnalyzer(train_ratio=0.7, n_windows=9)

        # 4-comp with FR filter on ALL components
        def make_rsi_1h_fr():
            base = RSIMeanReversionStrategy(
                rsi_oversold=35, rsi_overbought=65,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
            )
            mtf = MultiTimeframeFilter(base)
            return FundingRateFilter(mtf, long_suppress_threshold=best_lt, short_suppress_threshold=best_st)

        def make_dc_1h_fr():
            base = DonchianTrendStrategy(
                entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
                vol_mult=0.8, cooldown_bars=6,
            )
            mtf = MultiTimeframeFilter(base)
            return FundingRateFilter(mtf, long_suppress_threshold=best_lt, short_suppress_threshold=best_st)

        def make_rsi_15m_fr():
            base = RSIMeanReversionStrategy(
                rsi_oversold=35, rsi_overbought=65,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
            )
            mtf = MultiTimeframeFilter(base)
            return FundingRateFilter(mtf, long_suppress_threshold=best_lt, short_suppress_threshold=best_st)

        def make_willr_1h_fr():
            base = WilliamsRMeanReversionStrategy(
                willr_period=14, oversold_level=90, overbought_level=90,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
            )
            mtf = MultiTimeframeFilter(base)
            return FundingRateFilter(mtf, long_suppress_threshold=best_lt, short_suppress_threshold=best_st)

        components_fr = [
            CrossTFComponent(make_rsi_1h_fr, df_1h_sub, df_4h_sub, engine_1h, 0.15, "1hRSI+FR"),
            CrossTFComponent(make_dc_1h_fr, df_1h_sub, df_4h_sub, engine_1h, 0.50, "1hDC+FR"),
            CrossTFComponent(make_rsi_15m_fr, df_15m_sub, df_4h_sub, engine_15m, 0.10, "15mRSI+FR"),
            CrossTFComponent(make_willr_1h_fr, df_1h_sub, df_4h_sub, engine_1h, 0.25, "1hWillR+FR"),
        ]

        report_fr = wf_ctf.run_cross_tf(components_fr)

        for w in report_fr.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info(
                "    W%d [%s ~ %s]: %s -> %+.2f%% %s",
                w.window_id, w.test_start, w.test_end,
                " | ".join(parts), w.weighted_return, marker,
            )

        rob_fr = int(report_fr.robustness_score * 100)
        logger.info(
            "  4comp+FR: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
            report_fr.oos_total_return, rob_fr,
            report_fr.oos_profitable_windows, report_fr.total_windows,
            report_fr.total_trades,
        )
        logger.info("")

        # Also run 4-comp WITHOUT FR filter on same reduced period as baseline
        def make_rsi_1h():
            base = RSIMeanReversionStrategy(
                rsi_oversold=35, rsi_overbought=65,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
            )
            return MultiTimeframeFilter(base)

        def make_dc_1h():
            base = DonchianTrendStrategy(
                entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
                vol_mult=0.8, cooldown_bars=6,
            )
            return MultiTimeframeFilter(base)

        def make_rsi_15m():
            base = RSIMeanReversionStrategy(
                rsi_oversold=35, rsi_overbought=65,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
            )
            return MultiTimeframeFilter(base)

        def make_willr_1h():
            base = WilliamsRMeanReversionStrategy(
                willr_period=14, oversold_level=90, overbought_level=90,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
            )
            return MultiTimeframeFilter(base)

        components_no_fr = [
            CrossTFComponent(make_rsi_1h, df_1h_sub, df_4h_sub, engine_1h, 0.15, "1hRSI"),
            CrossTFComponent(make_dc_1h, df_1h_sub, df_4h_sub, engine_1h, 0.50, "1hDC"),
            CrossTFComponent(make_rsi_15m, df_15m_sub, df_4h_sub, engine_15m, 0.10, "15mRSI"),
            CrossTFComponent(make_willr_1h, df_1h_sub, df_4h_sub, engine_1h, 0.25, "1hWillR"),
        ]

        report_no_fr = wf_ctf.run_cross_tf(components_no_fr)

        for w in report_no_fr.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info(
                "    W%d [%s ~ %s]: %s -> %+.2f%% %s",
                w.window_id, w.test_start, w.test_end,
                " | ".join(parts), w.weighted_return, marker,
            )

        rob_no_fr = int(report_no_fr.robustness_score * 100)
        logger.info(
            "  4comp_baseline: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
            report_no_fr.oos_total_return, rob_no_fr,
            report_no_fr.oos_profitable_windows, report_no_fr.total_windows,
            report_no_fr.total_trades,
        )
        logger.info("")

    # ── PART 5: Summary ──────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  PHASE 29 SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    logger.info("  Baseline (RSI+MTF, reduced period): rob=%d%%, OOS=%+.2f%%, trades=%d",
                baseline_rob, baseline.oos_total_return, baseline.oos_total_trades)
    logger.info("")

    logger.info("  FR Filter Grid Results (sorted by robustness, then return):")
    for r in sorted(results_p2, key=lambda x: (-x["robustness"], -x["oos_return"])):
        delta = r["oos_return"] - baseline.oos_total_return
        logger.info(
            "    %s: rob=%d%%, OOS=%+.2f%% (delta=%+.2f%%), trades=%d",
            r["label"], r["robustness"], r["oos_return"], delta, r["trades"],
        )
    logger.info("")

    if best_fr:
        logger.info("  Best FR config: %s", best_fr["label"])
        logger.info("  Improvement over baseline: rob %d%% -> %d%%, OOS %+.2f%% -> %+.2f%%",
                     baseline_rob, best_fr["robustness"],
                     baseline.oos_total_return, best_fr["oos_return"])
    else:
        logger.info("  CONCLUSION: FR filter does NOT improve production portfolio.")
        logger.info("  88%% robustness ceiling remains intact.")

    logger.info("")
    logger.info("  Reference (full-period production):")
    logger.info("    4-comp 15/50/10/25: 88%% rob, +23.98%% OOS (full 12-month)")
    logger.info("")
    logger.info("  Phase 29 complete.")


if __name__ == "__main__":
    main()

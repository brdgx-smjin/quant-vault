#!/usr/bin/env python3
"""Phase 36 — Parabolic SAR & TRIX Oscillator.

Two genuinely untested indicators:
  1. Parabolic SAR: acceleration-based trend-following (different from Supertrend/DC)
  2. TRIX: triple-smoothed EMA momentum oscillator (mean-reversion)

Phase 36 tests:
  Part 1: PSAR standalone on 1h + MTF (grid: af_step × max_af × RR)
  Part 2: If PSAR ≥66% rob → as 5th component and DC/WillR replacement
  Part 3: TRIX mean-reversion on 1h + MTF (grid: length × threshold)
  Part 4: If TRIX ≥66% rob → as 5th component and RSI/WillR replacement
  Part 5: 15m test for any indicator reaching ≥55% on 1h

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
from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import (
    CrossTFComponent,
    WalkForwardAnalyzer,
)
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy
from src.strategy.psar_trend import PSARTrendStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase36")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase36.log", mode="w")
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


def add_psar(
    df: pd.DataFrame, af_step: float = 0.02, max_af: float = 0.20,
) -> pd.DataFrame:
    """Add Parabolic SAR columns to DataFrame."""
    long_col = f"PSARl_{af_step}_{max_af}"
    if long_col not in df.columns:
        psar = ta.psar(
            df["high"], df["low"], df["close"],
            af0=af_step, af=af_step, max_af=max_af,
        )
        if psar is not None:
            for col in psar.columns:
                df[col] = psar[col]
    return df


def add_trix(df: pd.DataFrame, length: int = 15) -> pd.DataFrame:
    """Add TRIX oscillator column to DataFrame."""
    col = f"TRIX_{length}"
    if col not in df.columns:
        trix_result = ta.trix(df["close"], length=length)
        if trix_result is not None:
            if isinstance(trix_result, pd.DataFrame):
                df[col] = trix_result.iloc[:, 0]
            else:
                df[col] = trix_result
    return df


# ─── TRIX Mean Reversion Strategy (inline) ───────────────────────

from src.strategy.base import BaseStrategy, Signal, TradeSignal
from config.settings import SYMBOL


class TRIXMeanReversionStrategy(BaseStrategy):
    """Mean-reversion using TRIX oscillator extremes.

    TRIX = 100 * rate of change of triple-smoothed EMA.
    Oscillates around zero. Extremes indicate overbought/oversold.

    Entry rules:
      LONG:  TRIX < -threshold (oversold momentum)
      SHORT: TRIX > +threshold (overbought momentum)
    """

    name = "trix_mean_reversion"

    def __init__(
        self,
        trix_length: int = 15,
        threshold: float = 0.10,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.trix_length = trix_length
        self.threshold = threshold
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999
        self._trix_col = f"TRIX_{trix_length}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 30:
            return self._hold(df)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        trix_val = last.get(self._trix_col)
        atr = last.get("atr_14")
        bb_low = last.get("BBL_20_2.0_2.0")
        bb_up = last.get("BBU_20_2.0_2.0")

        if any(pd.isna(v) for v in [trix_val, atr, bb_low, bb_up]):
            return self._hold(df)

        trix_val = float(trix_val)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        if atr <= 0:
            return self._hold(df)

        # LONG: TRIX extremely negative + near lower BB
        if trix_val < -self.threshold and close <= bb_low * 1.01:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            confidence = min(1.0, 0.5 + abs(trix_val) / self.threshold * 0.1)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={
                    "strategy": self.name,
                    "trix": trix_val,
                    "bb_touch": "lower",
                },
            )

        # SHORT: TRIX extremely positive + near upper BB
        if trix_val > self.threshold and close >= bb_up / 1.01:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            confidence = min(1.0, 0.5 + abs(trix_val) / self.threshold * 0.1)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={
                    "strategy": self.name,
                    "trix": trix_val,
                    "bb_touch": "upper",
                },
            )

        return self._hold(df)

    def _hold(self, df: pd.DataFrame) -> TradeSignal:
        return TradeSignal(
            signal=Signal.HOLD,
            symbol=self.symbol,
            price=float(df["close"].iloc[-1]),
            timestamp=df.index[-1],
        )

    def get_required_indicators(self) -> list[str]:
        return [self._trix_col, "atr_14", "BBL_20_2.0_2.0", "BBU_20_2.0_2.0"]


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
    logger.info("  PHASE 36 — Parabolic SAR & TRIX Oscillator")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Two genuinely untested indicators:")
    logger.info("  1. Parabolic SAR: acceleration-based trend-following")
    logger.info("  2. TRIX: triple-smoothed EMA momentum oscillator (MR)")
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
    #   PART 1: Parabolic SAR — Grid Search (1h + MTF)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: Parabolic SAR Trend-Following (1h + MTF)")
    logger.info("-" * 72)
    logger.info("  Entry: PSAR flips from above→below (LONG) or below→above (SHORT)")
    logger.info("  Volume confirmation: vol > 0.8 * 20-bar avg")
    logger.info("  Grid: af_step × max_af × RR ratio")
    logger.info("")

    psar_results = []
    af_steps = [0.01, 0.02, 0.03]
    max_afs = [0.15, 0.20, 0.25, 0.30]
    rr_ratios = [1.5, 2.0, 2.5]

    total = len(af_steps) * len(max_afs) * len(rr_ratios)
    idx = 0

    for af in af_steps:
        for maf in max_afs:
            df_1h_psar = add_psar(df_1h.copy(), af_step=af, max_af=maf)

            for rr in rr_ratios:
                idx += 1

                def make_psar_1h(a=af, m=maf, r=rr):
                    base = PSARTrendStrategy(
                        af_step=a, max_af=m,
                        atr_sl_mult=2.0, rr_ratio=r,
                        vol_mult=0.8, cooldown_bars=6,
                    )
                    return MultiTimeframeFilter(base)

                wf = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
                report = wf.run(make_psar_1h, df_1h_psar, htf_df=df_4h)

                rob = int(report.robustness_score * 100)
                psar_results.append({
                    "af": af, "maf": maf, "rr": rr,
                    "rob": rob, "oos": report.oos_total_return,
                    "trades": report.oos_total_trades,
                })

                marker = "★" if rob >= 66 else ""
                logger.info("    [%d/%d] AF%.2f_MAF%.2f_RR%.1f: Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                            idx, total, af, maf, rr, rob,
                            report.oos_total_return, report.oos_total_trades, marker)

    logger.info("")
    psar_sorted = sorted(psar_results, key=lambda x: (x["rob"], x["oos"]), reverse=True)
    best_psar = psar_sorted[0]
    logger.info("  Best PSAR: AF%.2f_MAF%.2f_RR%.1f → %d%% rob, %+.2f%% OOS, %d trades",
                best_psar["af"], best_psar["maf"], best_psar["rr"],
                best_psar["rob"], best_psar["oos"], best_psar["trades"])
    logger.info("")

    # Top results
    logger.info("  Top 10 configs:")
    for i, r in enumerate(psar_sorted[:10]):
        m = "★" if r["rob"] >= 66 else ""
        logger.info("    %d. AF%.2f_MAF%.2f_RR%.1f: %d%% rob, %+.2f%% OOS, %d trades %s",
                    i + 1, r["af"], r["maf"], r["rr"],
                    r["rob"], r["oos"], r["trades"], m)
    logger.info("")

    # Robustness distribution
    rob_dist = {}
    for r in psar_results:
        rb = r["rob"]
        rob_dist[rb] = rob_dist.get(rb, 0) + 1
    logger.info("  Robustness distribution:")
    for rb in sorted(rob_dist.keys(), reverse=True):
        logger.info("    %d%%: %d configs", rb, rob_dist[rb])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: PSAR Portfolio Integration (if ≥66% rob)
    # ═════════════════════════════════════════════════════════════
    if best_psar["rob"] >= 66:
        logger.info("-" * 72)
        logger.info("  PART 2: PSAR Portfolio Integration")
        logger.info("-" * 72)
        logger.info("")

        ba = best_psar["af"]
        bm = best_psar["maf"]
        br = best_psar["rr"]
        df_1h_best_psar = add_psar(df_1h.copy(), af_step=ba, max_af=bm)

        def make_psar_best_1h(a=ba, m=bm, r=br):
            base = PSARTrendStrategy(
                af_step=a, max_af=m,
                atr_sl_mult=2.0, rr_ratio=r,
                vol_mult=0.8, cooldown_bars=6,
            )
            return MultiTimeframeFilter(base)

        # 5-comp: 10/40/10/20/20
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
                strategy_factory=make_psar_best_1h,
                df=df_1h_best_psar, htf_df=df_4h,
                engine=engine_1h, weight=0.20, label="1hPSAR",
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

        # Replace Donchian
        logger.info("  PSAR Replacing Donchian (15/50/10/25 weights):")
        components_replace_dc = [
            CrossTFComponent(
                strategy_factory=make_rsi_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.15, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_psar_best_1h,
                df=df_1h_best_psar, htf_df=df_4h,
                engine=engine_1h, weight=0.50, label="1hPSAR",
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

        report_rdc = wf.run_cross_tf(components_replace_dc)
        rob_rdc = int(report_rdc.robustness_score * 100)
        logger.info("  Replace DC: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    rob_rdc, report_rdc.oos_total_return, report_rdc.total_trades)
        for w in report_rdc.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("    W%d: %s -> %+.2f%% %s",
                        w.window_id, " | ".join(parts), w.weighted_return, marker)
        logger.info("")

        # Replace WillR
        logger.info("  PSAR Replacing WillR (15/50/10/25 weights):")
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
                strategy_factory=make_psar_best_1h,
                df=df_1h_best_psar, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hPSAR",
            ),
        ]

        report_rwr = wf.run_cross_tf(components_replace_wr)
        rob_rwr = int(report_rwr.robustness_score * 100)
        logger.info("  Replace WillR: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    rob_rwr, report_rwr.oos_total_return, report_rwr.total_trades)
        for w in report_rwr.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("    W%d: %s -> %+.2f%% %s",
                        w.window_id, " | ".join(parts), w.weighted_return, marker)
        logger.info("")
    else:
        logger.info("  PSAR did not reach 66%% robustness. Skipping portfolio tests.")
        logger.info("")
        rob_5c = rob_rdc = rob_rwr = 0
        report_5c = report_rdc = report_rwr = None

    # ═════════════════════════════════════════════════════════════
    #   PART 3: TRIX Mean Reversion — Grid Search (1h + MTF)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: TRIX Mean Reversion (1h + MTF)")
    logger.info("-" * 72)
    logger.info("  TRIX = 100 * ROC(EMA(EMA(EMA(close))))")
    logger.info("  Entry: TRIX extreme + BB band touch (like RSI/WillR MR)")
    logger.info("  Grid: length × threshold")
    logger.info("")

    trix_results = []
    trix_lengths = [9, 12, 15, 18, 21]
    trix_thresholds = [0.05, 0.08, 0.10, 0.15, 0.20]

    total_trix = len(trix_lengths) * len(trix_thresholds)
    idx = 0

    for length in trix_lengths:
        df_1h_trix = add_trix(df_1h.copy(), length=length)

        for threshold in trix_thresholds:
            idx += 1

            def make_trix_1h(l=length, t=threshold):
                base = TRIXMeanReversionStrategy(
                    trix_length=l, threshold=t,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
                return MultiTimeframeFilter(base)

            wf = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
            report = wf.run(make_trix_1h, df_1h_trix, htf_df=df_4h)

            rob = int(report.robustness_score * 100)
            trix_results.append({
                "length": length, "threshold": threshold,
                "rob": rob, "oos": report.oos_total_return,
                "trades": report.oos_total_trades,
            })

            marker = "★" if rob >= 66 else ""
            logger.info("    [%d/%d] TRIX_%d_t%.2f: Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                        idx, total_trix, length, threshold, rob,
                        report.oos_total_return, report.oos_total_trades, marker)

    logger.info("")
    trix_sorted = sorted(trix_results, key=lambda x: (x["rob"], x["oos"]), reverse=True)
    best_trix = trix_sorted[0]
    logger.info("  Best TRIX: TRIX_%d_t%.2f → %d%% rob, %+.2f%% OOS, %d trades",
                best_trix["length"], best_trix["threshold"],
                best_trix["rob"], best_trix["oos"], best_trix["trades"])
    logger.info("")

    # Top results
    logger.info("  Top 10 configs:")
    for i, r in enumerate(trix_sorted[:10]):
        m = "★" if r["rob"] >= 66 else ""
        logger.info("    %d. TRIX_%d_t%.2f: %d%% rob, %+.2f%% OOS, %d trades %s",
                    i + 1, r["length"], r["threshold"],
                    r["rob"], r["oos"], r["trades"], m)
    logger.info("")

    # Robustness distribution
    rob_dist_trix = {}
    for r in trix_results:
        rb = r["rob"]
        rob_dist_trix[rb] = rob_dist_trix.get(rb, 0) + 1
    logger.info("  TRIX Robustness distribution:")
    for rb in sorted(rob_dist_trix.keys(), reverse=True):
        logger.info("    %d%%: %d configs", rb, rob_dist_trix[rb])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: TRIX Portfolio Integration (if ≥66% rob)
    # ═════════════════════════════════════════════════════════════
    trix_5c_rob = 0
    trix_rr_rob = 0
    trix_rw_rob = 0
    report_trix_5c = report_trix_rr = report_trix_rw = None

    if best_trix["rob"] >= 66:
        logger.info("-" * 72)
        logger.info("  PART 4: TRIX Portfolio Integration")
        logger.info("-" * 72)
        logger.info("")

        tl = best_trix["length"]
        tt = best_trix["threshold"]
        df_1h_best_trix = add_trix(df_1h.copy(), length=tl)

        def make_trix_best_1h(l=tl, t=tt):
            base = TRIXMeanReversionStrategy(
                trix_length=l, threshold=t,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
            )
            return MultiTimeframeFilter(base)

        # 5-comp
        wf = WalkForwardAnalyzer(n_windows=9)
        components_trix_5c = [
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
                strategy_factory=make_trix_best_1h,
                df=df_1h_best_trix, htf_df=df_4h,
                engine=engine_1h, weight=0.20, label="1hTRIX",
            ),
        ]

        report_trix_5c = wf.run_cross_tf(components_trix_5c)
        trix_5c_rob = int(report_trix_5c.robustness_score * 100)
        logger.info("  5-comp (10/40/10/20/20): Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    trix_5c_rob, report_trix_5c.oos_total_return,
                    report_trix_5c.total_trades)
        for w in report_trix_5c.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("    W%d: %s -> %+.2f%% %s",
                        w.window_id, " | ".join(parts), w.weighted_return, marker)
        logger.info("")

        # Replace RSI
        logger.info("  TRIX Replacing RSI 1h (15/50/10/25 → TRIX/DC/RSI15/WillR):")
        components_trix_rr = [
            CrossTFComponent(
                strategy_factory=make_trix_best_1h,
                df=df_1h_best_trix, htf_df=df_4h,
                engine=engine_1h, weight=0.15, label="1hTRIX",
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
                strategy_factory=make_willr_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hWillR",
            ),
        ]

        report_trix_rr = wf.run_cross_tf(components_trix_rr)
        trix_rr_rob = int(report_trix_rr.robustness_score * 100)
        logger.info("  Replace RSI: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    trix_rr_rob, report_trix_rr.oos_total_return,
                    report_trix_rr.total_trades)
        for w in report_trix_rr.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("    W%d: %s -> %+.2f%% %s",
                        w.window_id, " | ".join(parts), w.weighted_return, marker)
        logger.info("")

        # Replace WillR
        logger.info("  TRIX Replacing WillR (15/50/10/25 → RSI/DC/RSI15/TRIX):")
        components_trix_rw = [
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
                strategy_factory=make_trix_best_1h,
                df=df_1h_best_trix, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hTRIX",
            ),
        ]

        report_trix_rw = wf.run_cross_tf(components_trix_rw)
        trix_rw_rob = int(report_trix_rw.robustness_score * 100)
        logger.info("  Replace WillR: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    trix_rw_rob, report_trix_rw.oos_total_return,
                    report_trix_rw.total_trades)
        for w in report_trix_rw.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("    W%d: %s -> %+.2f%% %s",
                        w.window_id, " | ".join(parts), w.weighted_return, marker)
        logger.info("")
    else:
        logger.info("  TRIX did not reach 66%% robustness. Skipping portfolio tests.")
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 5: 15m Tests (for any ≥55% on 1h)
    # ═════════════════════════════════════════════════════════════
    tested_15m = False

    if best_psar["rob"] >= 55:
        logger.info("-" * 72)
        logger.info("  PART 5a: PSAR on 15m")
        logger.info("-" * 72)
        logger.info("")
        tested_15m = True

        ba = best_psar["af"]
        bm = best_psar["maf"]
        br = best_psar["rr"]
        df_15m_psar = add_psar(df_15m.copy(), af_step=ba, max_af=bm)

        def make_psar_15m(a=ba, m=bm, r=br):
            base = PSARTrendStrategy(
                af_step=a, max_af=m,
                atr_sl_mult=2.0, rr_ratio=r,
                vol_mult=0.8, cooldown_bars=12,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
        report_psar_15m = wf.run(make_psar_15m, df_15m_psar, htf_df=df_4h)
        rob_psar_15m = int(report_psar_15m.robustness_score * 100)
        logger.info("  PSAR 15m: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    rob_psar_15m, report_psar_15m.oos_total_return,
                    report_psar_15m.oos_total_trades)
        logger.info("")

    if best_trix["rob"] >= 55:
        logger.info("-" * 72)
        logger.info("  PART 5b: TRIX on 15m")
        logger.info("-" * 72)
        logger.info("")
        tested_15m = True

        tl = best_trix["length"]
        tt = best_trix["threshold"]
        df_15m_trix = add_trix(df_15m.copy(), length=tl)

        def make_trix_15m(l=tl, t=tt):
            base = TRIXMeanReversionStrategy(
                trix_length=l, threshold=t,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
        report_trix_15m = wf.run(make_trix_15m, df_15m_trix, htf_df=df_4h)
        rob_trix_15m = int(report_trix_15m.robustness_score * 100)
        logger.info("  TRIX 15m: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    rob_trix_15m, report_trix_15m.oos_total_return,
                    report_trix_15m.oos_total_trades)
        logger.info("")

    if not tested_15m:
        logger.info("  No indicators reached 55%% on 1h. Skipping 15m tests.")
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   SUMMARY
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 36 SUMMARY")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Baseline: 4-comp Fixed (15/50/10/25)")
    logger.info("    OOS Return: +23.98%% | Robustness: 88%% | Trades: 236")
    logger.info("")

    # PSAR summary
    logger.info("  ── Parabolic SAR ──")
    logger.info("  Best standalone (1h + MTF):")
    logger.info("    AF%.2f_MAF%.2f_RR%.1f: %d%% rob, %+.2f%% OOS, %d trades",
                best_psar["af"], best_psar["maf"], best_psar["rr"],
                best_psar["rob"], best_psar["oos"], best_psar["trades"])

    # Full PSAR grid
    logger.info("  Full PSAR Grid (sorted by rob, then OOS):")
    for r in psar_sorted:
        m = "★" if r["rob"] >= 66 else ""
        logger.info("    AF%.2f_MAF%.2f_RR%.1f: %d%% rob, %+.2f%% OOS, %d trades %s",
                    r["af"], r["maf"], r["rr"], r["rob"], r["oos"], r["trades"], m)

    if best_psar["rob"] >= 66 and report_5c is not None:
        logger.info("  PSAR portfolio:")
        logger.info("    5-comp: %d%% rob, %+.2f%% OOS", rob_5c, report_5c.oos_total_return)
        logger.info("    Replace DC: %d%% rob, %+.2f%% OOS", rob_rdc, report_rdc.oos_total_return)
        logger.info("    Replace WillR: %d%% rob, %+.2f%% OOS", rob_rwr, report_rwr.oos_total_return)
    logger.info("")

    # TRIX summary
    logger.info("  ── TRIX Oscillator ──")
    logger.info("  Best standalone (1h + MTF):")
    logger.info("    TRIX_%d_t%.2f: %d%% rob, %+.2f%% OOS, %d trades",
                best_trix["length"], best_trix["threshold"],
                best_trix["rob"], best_trix["oos"], best_trix["trades"])

    # Full TRIX grid
    logger.info("  Full TRIX Grid (sorted by rob, then OOS):")
    for r in trix_sorted:
        m = "★" if r["rob"] >= 66 else ""
        logger.info("    TRIX_%d_t%.2f: %d%% rob, %+.2f%% OOS, %d trades %s",
                    r["length"], r["threshold"], r["rob"], r["oos"], r["trades"], m)

    if best_trix["rob"] >= 66 and report_trix_5c is not None:
        logger.info("  TRIX portfolio:")
        logger.info("    5-comp: %d%% rob, %+.2f%% OOS",
                    trix_5c_rob, report_trix_5c.oos_total_return)
        logger.info("    Replace RSI: %d%% rob, %+.2f%% OOS",
                    trix_rr_rob, report_trix_rr.oos_total_return)
        logger.info("    Replace WillR: %d%% rob, %+.2f%% OOS",
                    trix_rw_rob, report_trix_rw.oos_total_return)
    logger.info("")

    # Conclusion
    psar_improves = (
        best_psar["rob"] >= 66
        and report_5c is not None
        and (
            (rob_5c >= 88 and report_5c.oos_total_return > 23.98)
            or (rob_rdc >= 88 and report_rdc.oos_total_return > 23.98)
            or (rob_rwr >= 88 and report_rwr.oos_total_return > 23.98)
        )
    )

    trix_improves = (
        best_trix["rob"] >= 66
        and report_trix_5c is not None
        and (
            (trix_5c_rob >= 88 and report_trix_5c.oos_total_return > 23.98)
            or (trix_rr_rob >= 88 and report_trix_rr.oos_total_return > 23.98)
            or (trix_rw_rob >= 88 and report_trix_rw.oos_total_return > 23.98)
        )
    )

    if psar_improves or trix_improves:
        logger.info("  ★★★ BREAKTHROUGH: Found improvement to portfolio! ★★★")
    else:
        logger.info("  CONCLUSION: Neither PSAR nor TRIX improve 4-comp portfolio.")
        logger.info("  This is the 13th-14th indicator tested as 5th/replacement.")
        logger.info("  88%% robustness ceiling confirmed as structural.")
    logger.info("")
    logger.info("  Phase 36 complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 33 — Untested Indicator Exploration: Aroon, DPO, TSI.

Previous 32 phases tested: RSI, Williams %R, CCI, VWAP, Fisher Transform,
Z-Score, CMF, StochRSI, EFI, Keltner, MFI, ROC, Ichimoku Kijun, MACD,
Stochastic, BB Squeeze, Donchian, Hurst, ER, Funding Rate.

Phase 33 tests three genuinely UNTESTED indicators as MR strategies:
  1. Aroon Oscillator — time since high/low (unique: time-based, not price-level)
  2. DPO (Detrended Price Oscillator) — removes trend to isolate cycles
  3. TSI (True Strength Index) — double-smoothed momentum

Each is tested:
  A. Standalone on 1h + MTF (grid search params)
  B. If ≥66% rob: as 5th component in 4-comp portfolio
  C. If ≥66% rob: as WillR replacement in 4-comp portfolio

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
from src.strategy.base import BaseStrategy, Signal, TradeSignal
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy
from config.settings import SYMBOL

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase33")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase33.log", mode="w")
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


# ─── Strategy 1: Aroon Mean Reversion ────────────────────────────

class AroonMeanReversionStrategy(BaseStrategy):
    """Mean reversion using Aroon Oscillator.

    Aroon measures how long since the most recent high/low:
      Aroon Up = ((period - bars_since_high) / period) * 100
      Aroon Down = ((period - bars_since_low) / period) * 100
      Aroon Osc = Aroon Up - Aroon Down  (range: -100 to +100)

    MR logic:
      LONG:  Aroon Osc < -threshold (bearish extreme → oversold)
             AND close <= BB_lower (price confirmation)
      SHORT: Aroon Osc > +threshold (bullish extreme → overbought)
             AND close >= BB_upper (price confirmation)
    """

    name = "aroon_mean_reversion"

    def __init__(
        self,
        aroon_period: int = 14,
        threshold: float = 80.0,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.aroon_period = aroon_period
        self.threshold = threshold
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self._last_entry_idx = -999
        self.symbol = symbol
        self._aroon_up_col = f"AROONU_{aroon_period}"
        self._aroon_down_col = f"AROOND_{aroon_period}"
        self._aroon_osc_col = f"AROONOSC_{aroon_period}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 30:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        aroon_osc = last.get(self._aroon_osc_col)
        atr = last.get("atr_14")
        bb_low = last.get("BBL_20_2.0_2.0")
        bb_up = last.get("BBU_20_2.0_2.0")

        if any(pd.isna(v) for v in [aroon_osc, atr, bb_low, bb_up]):
            return self._hold(df)

        aroon_osc = float(aroon_osc)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        # LONG: Aroon Osc deeply negative (bearish extreme) + BB lower
        if aroon_osc < -self.threshold and close <= bb_low * 1.01:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG, symbol=self.symbol, price=close,
                timestamp=ts, confidence=0.6, stop_loss=sl, take_profit=tp,
                metadata={"strategy": self.name, "aroon_osc": aroon_osc},
            )

        # SHORT: Aroon Osc deeply positive (bullish extreme) + BB upper
        if aroon_osc > self.threshold and close >= bb_up / 1.01:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT, symbol=self.symbol, price=close,
                timestamp=ts, confidence=0.6, stop_loss=sl, take_profit=tp,
                metadata={"strategy": self.name, "aroon_osc": aroon_osc},
            )

        return self._hold(df)

    def _hold(self, df: pd.DataFrame) -> TradeSignal:
        return TradeSignal(
            signal=Signal.HOLD, symbol=self.symbol,
            price=float(df["close"].iloc[-1]), timestamp=df.index[-1],
        )

    def get_required_indicators(self) -> list[str]:
        return [self._aroon_osc_col, "atr_14", "BBL_20_2.0_2.0", "BBU_20_2.0_2.0"]


# ─── Strategy 2: DPO Mean Reversion ─────────────────────────────

class DPOMeanReversionStrategy(BaseStrategy):
    """Mean reversion using Detrended Price Oscillator.

    DPO removes the trend component to isolate price cycles.
    DPO = close[t - (period/2 + 1)] - SMA(close, period)

    MR logic:
      LONG:  DPO < -threshold*ATR (price below detrended mean)
             AND close <= BB_lower
      SHORT: DPO > +threshold*ATR (price above detrended mean)
             AND close >= BB_upper
    """

    name = "dpo_mean_reversion"

    def __init__(
        self,
        dpo_period: int = 20,
        threshold_mult: float = 1.5,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.dpo_period = dpo_period
        self.threshold_mult = threshold_mult
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self._last_entry_idx = -999
        self.symbol = symbol
        self._dpo_col = f"DPO_{dpo_period}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 30:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        dpo = last.get(self._dpo_col)
        atr = last.get("atr_14")
        bb_low = last.get("BBL_20_2.0_2.0")
        bb_up = last.get("BBU_20_2.0_2.0")

        if any(pd.isna(v) for v in [dpo, atr, bb_low, bb_up]):
            return self._hold(df)

        dpo = float(dpo)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        threshold = atr * self.threshold_mult

        # LONG: DPO deeply negative (below detrended mean) + BB lower
        if dpo < -threshold and close <= bb_low * 1.01:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG, symbol=self.symbol, price=close,
                timestamp=ts, confidence=0.6, stop_loss=sl, take_profit=tp,
                metadata={"strategy": self.name, "dpo": dpo},
            )

        # SHORT: DPO deeply positive + BB upper
        if dpo > threshold and close >= bb_up / 1.01:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT, symbol=self.symbol, price=close,
                timestamp=ts, confidence=0.6, stop_loss=sl, take_profit=tp,
                metadata={"strategy": self.name, "dpo": dpo},
            )

        return self._hold(df)

    def _hold(self, df: pd.DataFrame) -> TradeSignal:
        return TradeSignal(
            signal=Signal.HOLD, symbol=self.symbol,
            price=float(df["close"].iloc[-1]), timestamp=df.index[-1],
        )

    def get_required_indicators(self) -> list[str]:
        return [self._dpo_col, "atr_14", "BBL_20_2.0_2.0", "BBU_20_2.0_2.0"]


# ─── Strategy 3: TSI Mean Reversion ─────────────────────────────

class TSIMeanReversionStrategy(BaseStrategy):
    """Mean reversion using True Strength Index.

    TSI = 100 * EMA(EMA(momentum, long), short) / EMA(EMA(|momentum|, long), short)
    Range: -100 to +100.

    MR logic:
      LONG:  TSI < -threshold (deeply oversold momentum)
             AND close <= BB_lower
      SHORT: TSI > +threshold (deeply overbought momentum)
             AND close >= BB_upper
    """

    name = "tsi_mean_reversion"

    def __init__(
        self,
        tsi_long: int = 25,
        tsi_short: int = 13,
        threshold: float = 20.0,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.tsi_long = tsi_long
        self.tsi_short = tsi_short
        self.threshold = threshold
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self._last_entry_idx = -999
        self.symbol = symbol
        self._tsi_col = f"TSI_{tsi_long}_{tsi_short}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 30:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        tsi_val = last.get(self._tsi_col)
        atr = last.get("atr_14")
        bb_low = last.get("BBL_20_2.0_2.0")
        bb_up = last.get("BBU_20_2.0_2.0")

        if any(pd.isna(v) for v in [tsi_val, atr, bb_low, bb_up]):
            return self._hold(df)

        tsi_val = float(tsi_val)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        # LONG: TSI deeply negative + BB lower
        if tsi_val < -self.threshold and close <= bb_low * 1.01:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG, symbol=self.symbol, price=close,
                timestamp=ts, confidence=0.6, stop_loss=sl, take_profit=tp,
                metadata={"strategy": self.name, "tsi": tsi_val},
            )

        # SHORT: TSI deeply positive + BB upper
        if tsi_val > self.threshold and close >= bb_up / 1.01:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT, symbol=self.symbol, price=close,
                timestamp=ts, confidence=0.6, stop_loss=sl, take_profit=tp,
                metadata={"strategy": self.name, "tsi": tsi_val},
            )

        return self._hold(df)

    def _hold(self, df: pd.DataFrame) -> TradeSignal:
        return TradeSignal(
            signal=Signal.HOLD, symbol=self.symbol,
            price=float(df["close"].iloc[-1]), timestamp=df.index[-1],
        )

    def get_required_indicators(self) -> list[str]:
        return [self._tsi_col, "atr_14", "BBL_20_2.0_2.0", "BBU_20_2.0_2.0"]


# ─── Indicator helpers ───────────────────────────────────────────

def add_aroon(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Aroon Up, Down, and Oscillator columns."""
    osc_col = f"AROONOSC_{period}"
    if osc_col not in df.columns:
        aroon = ta.aroon(df["high"], df["low"], length=period)
        if aroon is not None:
            df[f"AROONU_{period}"] = aroon.iloc[:, 0]
            df[f"AROOND_{period}"] = aroon.iloc[:, 1]
            df[osc_col] = aroon.iloc[:, 2]
    return df


def add_dpo(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Detrended Price Oscillator column."""
    col = f"DPO_{period}"
    if col not in df.columns:
        dpo = ta.dpo(df["close"], length=period)
        if dpo is not None:
            df[col] = dpo
    return df


def add_tsi(df: pd.DataFrame, long_period: int = 25, short_period: int = 13) -> pd.DataFrame:
    """Add True Strength Index column."""
    col = f"TSI_{long_period}_{short_period}"
    if col not in df.columns:
        tsi = ta.tsi(df["close"], fast=short_period, slow=long_period)
        if tsi is not None:
            # tsi returns a DataFrame with TSI and signal line
            df[col] = tsi.iloc[:, 0]
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
    logger.info("  PHASE 33 — Untested Indicator Exploration")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Testing: Aroon Oscillator, DPO, TSI")
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
    #   PART 1: Aroon Oscillator MR — Grid Search
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: Aroon Oscillator Mean Reversion (1h + MTF)")
    logger.info("-" * 72)
    logger.info("  Aroon measures time since recent high/low. Range: -100 to +100.")
    logger.info("  Unique: time-based oscillator (all prior indicators are price-level).")
    logger.info("")

    aroon_results = []
    aroon_periods = [10, 14, 20, 25]
    aroon_thresholds = [60, 70, 80, 90]

    for period in aroon_periods:
        df_1h_aroon = add_aroon(df_1h.copy(), period=period)

        for threshold in aroon_thresholds:
            def make_aroon_1h(p=period, t=threshold):
                base = AroonMeanReversionStrategy(
                    aroon_period=p, threshold=float(t),
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
                return MultiTimeframeFilter(base)

            wf = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
            report = wf.run(make_aroon_1h, df_1h_aroon, htf_df=df_4h)

            rob = int(report.robustness_score * 100)
            aroon_results.append({
                "period": period, "threshold": threshold,
                "rob": rob, "oos": report.oos_total_return,
                "trades": report.oos_total_trades,
            })

            marker = "★" if rob >= 66 else ""
            logger.info("    p%d_t%d: Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                        period, threshold, rob, report.oos_total_return,
                        report.oos_total_trades, marker)

    logger.info("")
    best_aroon = max(aroon_results, key=lambda x: (x["rob"], x["oos"]))
    logger.info("  Best Aroon: p%d_t%d → %d%% rob, %+.2f%% OOS, %d trades",
                best_aroon["period"], best_aroon["threshold"],
                best_aroon["rob"], best_aroon["oos"], best_aroon["trades"])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: DPO Mean Reversion — Grid Search
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: DPO (Detrended Price Oscillator) Mean Reversion (1h + MTF)")
    logger.info("-" * 72)
    logger.info("  DPO removes trend to isolate cycles. Threshold in ATR multiples.")
    logger.info("")

    dpo_results = []
    dpo_periods = [14, 20, 26, 30]
    dpo_thresholds = [1.0, 1.5, 2.0, 2.5]

    for period in dpo_periods:
        df_1h_dpo = add_dpo(df_1h.copy(), period=period)

        for threshold in dpo_thresholds:
            def make_dpo_1h(p=period, t=threshold):
                base = DPOMeanReversionStrategy(
                    dpo_period=p, threshold_mult=t,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
                return MultiTimeframeFilter(base)

            wf = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
            report = wf.run(make_dpo_1h, df_1h_dpo, htf_df=df_4h)

            rob = int(report.robustness_score * 100)
            dpo_results.append({
                "period": period, "threshold": threshold,
                "rob": rob, "oos": report.oos_total_return,
                "trades": report.oos_total_trades,
            })

            marker = "★" if rob >= 66 else ""
            logger.info("    p%d_t%.1f: Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                        period, threshold, rob, report.oos_total_return,
                        report.oos_total_trades, marker)

    logger.info("")
    best_dpo = max(dpo_results, key=lambda x: (x["rob"], x["oos"]))
    logger.info("  Best DPO: p%d_t%.1f → %d%% rob, %+.2f%% OOS, %d trades",
                best_dpo["period"], best_dpo["threshold"],
                best_dpo["rob"], best_dpo["oos"], best_dpo["trades"])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: TSI Mean Reversion — Grid Search
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: TSI (True Strength Index) Mean Reversion (1h + MTF)")
    logger.info("-" * 72)
    logger.info("  TSI double-smooths momentum. Range: -100 to +100.")
    logger.info("")

    tsi_results = []
    tsi_configs = [
        (25, 13),  # default
        (20, 10),  # faster
        (30, 15),  # slower
        (13, 7),   # very fast
    ]
    tsi_thresholds = [15, 20, 25, 30]

    for tsi_long, tsi_short in tsi_configs:
        df_1h_tsi = add_tsi(df_1h.copy(), long_period=tsi_long, short_period=tsi_short)

        for threshold in tsi_thresholds:
            def make_tsi_1h(tl=tsi_long, ts_p=tsi_short, t=threshold):
                base = TSIMeanReversionStrategy(
                    tsi_long=tl, tsi_short=ts_p, threshold=float(t),
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
                return MultiTimeframeFilter(base)

            wf = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
            report = wf.run(make_tsi_1h, df_1h_tsi, htf_df=df_4h)

            rob = int(report.robustness_score * 100)
            tsi_results.append({
                "long": tsi_long, "short": tsi_short, "threshold": threshold,
                "rob": rob, "oos": report.oos_total_return,
                "trades": report.oos_total_trades,
            })

            marker = "★" if rob >= 66 else ""
            logger.info("    TSI(%d,%d)_t%d: Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                        tsi_long, tsi_short, threshold, rob,
                        report.oos_total_return, report.oos_total_trades, marker)

    logger.info("")
    best_tsi = max(tsi_results, key=lambda x: (x["rob"], x["oos"]))
    logger.info("  Best TSI: (%d,%d)_t%d → %d%% rob, %+.2f%% OOS, %d trades",
                best_tsi["long"], best_tsi["short"], best_tsi["threshold"],
                best_tsi["rob"], best_tsi["oos"], best_tsi["trades"])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Cross-TF Portfolio Tests (if any indicator ≥66% rob)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 4: Cross-TF Portfolio Integration Tests")
    logger.info("-" * 72)
    logger.info("")

    viable_indicators = []
    if best_aroon["rob"] >= 66:
        viable_indicators.append(("Aroon", best_aroon))
    if best_dpo["rob"] >= 66:
        viable_indicators.append(("DPO", best_dpo))
    if best_tsi["rob"] >= 66:
        viable_indicators.append(("TSI", best_tsi))

    if not viable_indicators:
        logger.info("  No indicator reached 66%% robustness standalone.")
        logger.info("  Skipping portfolio integration tests.")
        logger.info("")
    else:
        for name, params in viable_indicators:
            logger.info("  Testing %s as 5th component in 4-comp portfolio:", name)

            # Build 5-comp: reduce each existing weight by 5% to give new comp 20%
            if name == "Aroon":
                period = params["period"]
                threshold = params["threshold"]
                df_1h_ind = add_aroon(df_1h.copy(), period=period)

                def make_new_1h(p=period, t=threshold):
                    base = AroonMeanReversionStrategy(
                        aroon_period=p, threshold=float(t),
                        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                    )
                    return MultiTimeframeFilter(base)

            elif name == "DPO":
                period = params["period"]
                threshold = params["threshold"]
                df_1h_ind = add_dpo(df_1h.copy(), period=period)

                def make_new_1h(p=period, t=threshold):
                    base = DPOMeanReversionStrategy(
                        dpo_period=p, threshold_mult=t,
                        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                    )
                    return MultiTimeframeFilter(base)

            elif name == "TSI":
                tl = params["long"]
                ts_p = params["short"]
                threshold = params["threshold"]
                df_1h_ind = add_tsi(df_1h.copy(), long_period=tl, short_period=ts_p)

                def make_new_1h(tl_=tl, ts_=ts_p, t=threshold):
                    base = TSIMeanReversionStrategy(
                        tsi_long=tl_, tsi_short=ts_, threshold=float(t),
                        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                    )
                    return MultiTimeframeFilter(base)

            # 5-comp: 10/40/10/20/20 (reduce DC from 50→40, WillR 25→20)
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
                    strategy_factory=make_new_1h,
                    df=df_1h_ind, htf_df=df_4h,
                    engine=engine_1h, weight=0.20, label=f"1h{name}",
                ),
            ]

            report_5c = wf.run_cross_tf(components_5comp)
            logger.info("    5-comp (10/40/10/20/20): Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                        int(report_5c.robustness_score * 100),
                        report_5c.oos_total_return, report_5c.total_trades)
            for w in report_5c.windows:
                parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
                marker = "+" if w.weighted_return > 0 else "-"
                logger.info("      W%d: %s -> %+.2f%% %s",
                            w.window_id, " | ".join(parts), w.weighted_return, marker)
            logger.info("")

            # Replace WillR with new indicator
            logger.info("  Testing %s replacing WillR (15/50/10/25):", name)
            components_replace = [
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
                    strategy_factory=make_new_1h,
                    df=df_1h_ind, htf_df=df_4h,
                    engine=engine_1h, weight=0.25, label=f"1h{name}",
                ),
            ]

            report_rep = wf.run_cross_tf(components_replace)
            logger.info("    Replace WillR: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                        int(report_rep.robustness_score * 100),
                        report_rep.oos_total_return, report_rep.total_trades)
            for w in report_rep.windows:
                parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
                marker = "+" if w.weighted_return > 0 else "-"
                logger.info("      W%d: %s -> %+.2f%% %s",
                            w.window_id, " | ".join(parts), w.weighted_return, marker)
            logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 5: 15m Tests for best indicators
    # ═════════════════════════════════════════════════════════════
    any_viable_15m = False
    for name, params in [("Aroon", best_aroon), ("DPO", best_dpo), ("TSI", best_tsi)]:
        if params["rob"] >= 55:  # lower bar for 15m testing
            if not any_viable_15m:
                logger.info("-" * 72)
                logger.info("  PART 5: 15m Tests for Promising Indicators")
                logger.info("-" * 72)
                logger.info("")
                any_viable_15m = True

            logger.info("  Testing %s on 15m:", name)

            if name == "Aroon":
                period = params["period"]
                threshold = params["threshold"]
                df_15m_ind = add_aroon(df_15m.copy(), period=period)

                def make_15m(p=period, t=threshold):
                    base = AroonMeanReversionStrategy(
                        aroon_period=p, threshold=float(t),
                        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
                    )
                    return MultiTimeframeFilter(base)

            elif name == "DPO":
                period = params["period"]
                threshold = params["threshold"]
                df_15m_ind = add_dpo(df_15m.copy(), period=period)

                def make_15m(p=period, t=threshold):
                    base = DPOMeanReversionStrategy(
                        dpo_period=p, threshold_mult=t,
                        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
                    )
                    return MultiTimeframeFilter(base)

            elif name == "TSI":
                tl = params["long"]
                ts_p = params["short"]
                threshold = params["threshold"]
                df_15m_ind = add_tsi(df_15m.copy(), long_period=tl, short_period=ts_p)

                def make_15m(tl_=tl, ts_=ts_p, t=threshold):
                    base = TSIMeanReversionStrategy(
                        tsi_long=tl_, tsi_short=ts_, threshold=float(t),
                        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
                    )
                    return MultiTimeframeFilter(base)

            wf = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
            report = wf.run(make_15m, df_15m_ind, htf_df=df_4h)

            logger.info("    15m: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                        int(report.robustness_score * 100),
                        report.oos_total_return, report.oos_total_trades)
            logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   SUMMARY
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 33 SUMMARY")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Baseline: 4-comp Fixed (15/50/10/25)")
    logger.info("    OOS Return: +23.98%% | Robustness: 88%% | Trades: 236")
    logger.info("")

    logger.info("  Standalone 1h+MTF Results (best per indicator):")
    logger.info("  %-20s  %6s  %10s  %7s",
                "Indicator", "Rob%", "OOS Ret", "Trades")
    logger.info("  " + "-" * 50)

    for name, params in [("Aroon", best_aroon), ("DPO", best_dpo), ("TSI", best_tsi)]:
        if name == "Aroon":
            label = f"Aroon p{params['period']}_t{params['threshold']}"
        elif name == "DPO":
            label = f"DPO p{params['period']}_t{params['threshold']}"
        else:
            label = f"TSI({params['long']},{params['short']})_t{params['threshold']}"

        logger.info("  %-20s  %5d%%  %+9.2f%%  %6d",
                    label, params["rob"], params["oos"], params["trades"])

    logger.info("")

    # All standalone results table
    logger.info("  Full Grid Search Results:")
    logger.info("")
    logger.info("  Aroon Oscillator:")
    for r in aroon_results:
        m = "★" if r["rob"] >= 66 else ""
        logger.info("    p%d_t%d: %d%% rob, %+.2f%% OOS, %d trades %s",
                    r["period"], r["threshold"], r["rob"], r["oos"], r["trades"], m)

    logger.info("")
    logger.info("  DPO:")
    for r in dpo_results:
        m = "★" if r["rob"] >= 66 else ""
        logger.info("    p%d_t%.1f: %d%% rob, %+.2f%% OOS, %d trades %s",
                    r["period"], r["threshold"], r["rob"], r["oos"], r["trades"], m)

    logger.info("")
    logger.info("  TSI:")
    for r in tsi_results:
        m = "★" if r["rob"] >= 66 else ""
        logger.info("    TSI(%d,%d)_t%d: %d%% rob, %+.2f%% OOS, %d trades %s",
                    r["long"], r["short"], r["threshold"], r["rob"], r["oos"], r["trades"], m)

    logger.info("")

    # Final conclusion
    any_improvement = any(
        params["rob"] >= 66
        for _, params in [("Aroon", best_aroon), ("DPO", best_dpo), ("TSI", best_tsi)]
    )

    if any_improvement:
        logger.info("  CONCLUSION: Some indicators reached ≥66%% standalone robustness.")
        logger.info("  See portfolio integration tests above for 4-comp impact.")
    else:
        logger.info("  CONCLUSION: No new indicator reached 66%% standalone robustness.")
        logger.info("  4-comp Cross-TF Portfolio (15/50/10/25) remains optimal at 88%%.")
        logger.info("  Aroon/DPO/TSI join the list of exhausted indicator options.")

    logger.info("")
    logger.info("  Phase 33 complete.")


if __name__ == "__main__":
    main()

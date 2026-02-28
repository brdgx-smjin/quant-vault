#!/usr/bin/env python3
"""Phase 39 — OBV (On-Balance Volume) Strategy Testing.

Last genuinely untested major indicator. All other volume indicators
(VWAP P23, MFI P15b, CMF P30, EFI P30) failed. OBV is structurally
different: cumulative volume signed by price direction.

Tests:
  Part 1: OBV Z-Score Mean Reversion (1h + MTF) — grid: period × z_thresh
  Part 2: OBV Donchian Breakout (1h + MTF) — grid: obv_period × dc_period
  Part 3: If any ≥66% rob → 5th component portfolio test
  Part 4: If any ≥55% rob → 15m test

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

logger = logging.getLogger("phase39")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase39.log", mode="w")
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


# ─── Data ─────────────────────────────────────────────────────────

def load_data(timeframe: str) -> pd.DataFrame:
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


def add_obv_zscore(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add OBV and its Z-score to DataFrame."""
    obv_col = "OBV"
    zscore_col = f"OBV_Z_{period}"
    if zscore_col not in df.columns:
        if obv_col not in df.columns:
            obv_raw = ta.obv(df["close"], df["volume"])
            if obv_raw is not None:
                df[obv_col] = obv_raw
        if obv_col in df.columns:
            obv_ma = df[obv_col].rolling(period).mean()
            obv_std = df[obv_col].rolling(period).std()
            df[zscore_col] = (df[obv_col] - obv_ma) / obv_std.replace(0, np.nan)
    return df


def add_willr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    col = f"WILLR_{period}"
    if col not in df.columns:
        df[col] = ta.willr(df["high"], df["low"], df["close"], length=period)
    return df


# ─── OBV Z-Score Mean Reversion Strategy ──────────────────────────

class OBVZScoreMRStrategy(BaseStrategy):
    """Mean-reversion using OBV Z-Score extremes + Bollinger Band confirmation.

    OBV (On-Balance Volume) accumulates volume on up-closes and subtracts
    on down-closes. Z-Score normalization creates a bounded oscillator.

    Entry rules:
      LONG:  OBV_Z < -threshold (extreme selling volume = accumulation) + BB lower
      SHORT: OBV_Z > +threshold (extreme buying volume = distribution) + BB upper
    """

    name = "obv_zscore_mr"

    def __init__(
        self,
        obv_period: int = 20,
        z_threshold: float = 2.0,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.obv_period = obv_period
        self.z_threshold = z_threshold
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999
        self._zscore_col = f"OBV_Z_{obv_period}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 30:
            return self._hold(df)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        z_val = last.get(self._zscore_col)
        atr = last.get("atr_14")
        bb_low = last.get("BBL_20_2.0_2.0")
        bb_up = last.get("BBU_20_2.0_2.0")

        if any(pd.isna(v) for v in [z_val, atr, bb_low, bb_up]):
            return self._hold(df)

        z_val = float(z_val)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        if atr <= 0:
            return self._hold(df)

        # LONG: OBV extremely negative (selling exhaustion) + near lower BB
        if z_val < -self.z_threshold and close <= bb_low * 1.01:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            confidence = min(1.0, 0.5 + abs(z_val) / self.z_threshold * 0.1)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "obv_z": z_val, "bb_touch": "lower"},
            )

        # SHORT: OBV extremely positive (buying exhaustion) + near upper BB
        if z_val > self.z_threshold and close >= bb_up / 1.01:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            confidence = min(1.0, 0.5 + abs(z_val) / self.z_threshold * 0.1)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "obv_z": z_val, "bb_touch": "upper"},
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
        return [self._zscore_col, "atr_14", "BBL_20_2.0_2.0", "BBU_20_2.0_2.0"]


# ─── OBV Donchian Breakout Strategy ──────────────────────────────

class OBVDonchianStrategy(BaseStrategy):
    """Trend-following using OBV Donchian Channel breakout.

    Instead of price breakout (standard Donchian), this uses VOLUME breakout:
    enter when OBV breaks its N-bar range. Volume breakout may lead price.

    Entry rules:
      LONG:  OBV > max(OBV[-dc_period:]) — volume accumulation breakout
             + volume > vol_mult * 20-bar average
      SHORT: OBV < min(OBV[-dc_period:]) — volume distribution breakout
             + volume > vol_mult * 20-bar average
    """

    name = "obv_donchian"

    def __init__(
        self,
        obv_period: int = 14,
        dc_period: int = 24,
        atr_sl_mult: float = 2.0,
        rr_ratio: float = 2.0,
        vol_mult: float = 0.8,
        cooldown_bars: int = 6,
        symbol: str = SYMBOL,
    ) -> None:
        self.obv_period = obv_period
        self.dc_period = dc_period
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.vol_mult = vol_mult
        self.cooldown_bars = cooldown_bars
        self.symbol = symbol
        self._last_entry_idx = -999

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        min_bars = self.dc_period + 10
        if len(df) < min_bars:
            return self._hold(df)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        obv = last.get("OBV")
        atr = last.get("atr_14")
        volume = last.get("volume")

        if any(pd.isna(v) for v in [obv, atr, volume]):
            return self._hold(df)

        obv = float(obv)
        atr = float(atr)
        volume = float(volume)

        if atr <= 0:
            return self._hold(df)

        # Volume confirmation
        vol_avg = float(df["volume"].iloc[-20:].mean())
        if vol_avg <= 0 or volume < vol_avg * self.vol_mult:
            return self._hold(df)

        # OBV Donchian channel (exclude current bar)
        obv_lookback = df["OBV"].iloc[-(self.dc_period + 1):-1]
        if obv_lookback.isna().all():
            return self._hold(df)

        obv_high = float(obv_lookback.max())
        obv_low = float(obv_lookback.min())

        # LONG: OBV breaks above channel high
        if obv > obv_high:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_sl_mult * self.rr_ratio
            breakout_strength = (obv - obv_high) / max(abs(obv_high), 1)
            confidence = min(1.0, 0.5 + breakout_strength * 0.5)
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
                    "obv": obv,
                    "obv_channel_high": obv_high,
                    "volume_ratio": volume / vol_avg,
                },
            )

        # SHORT: OBV breaks below channel low
        if obv < obv_low:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_sl_mult * self.rr_ratio
            breakout_strength = (obv_low - obv) / max(abs(obv_low), 1)
            confidence = min(1.0, 0.5 + breakout_strength * 0.5)
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
                    "obv": obv,
                    "obv_channel_low": obv_low,
                    "volume_ratio": volume / vol_avg,
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
        return ["OBV", "atr_14"]


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
    logger.info("  PHASE 39 — OBV (On-Balance Volume) Strategy Testing")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Last genuinely untested major indicator.")
    logger.info("  Other volume indicators (VWAP P23, MFI P15b, CMF P30, EFI P30)")
    logger.info("  all failed. OBV is structurally different: cumulative volume")
    logger.info("  signed by price direction.")
    logger.info("  Baseline: 4-comp 15/50/10/25, 88%% rob, +23.98%% OOS")
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")
    df_1h = add_willr(df_1h, 14)

    # Add raw OBV to both timeframes
    obv_1h = ta.obv(df_1h["close"], df_1h["volume"])
    if obv_1h is not None:
        df_1h["OBV"] = obv_1h
    obv_15m = ta.obv(df_15m["close"], df_15m["volume"])
    if obv_15m is not None:
        df_15m["OBV"] = obv_15m

    logger.info("  1h data:  %d bars (%s ~ %s)",
                len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("  15m data: %d bars (%s ~ %s)",
                len(df_15m), df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")

    # ═════════════════════════════════════════════════════════════
    #   PART 1: OBV Z-Score Mean Reversion (1h + MTF)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: OBV Z-Score Mean Reversion (1h + MTF)")
    logger.info("-" * 72)
    logger.info("  Entry: extreme OBV z-score + Bollinger Band touch")
    logger.info("  Grid: obv_period × z_threshold")
    logger.info("")

    obv_mr_results = []
    obv_periods = [10, 14, 20, 30]
    z_thresholds = [1.5, 2.0, 2.5, 3.0]
    total = len(obv_periods) * len(z_thresholds)
    idx = 0

    for period in obv_periods:
        df_1h_z = add_obv_zscore(df_1h.copy(), period=period)

        for z_thresh in z_thresholds:
            idx += 1

            def make_obv_mr(p=period, z=z_thresh):
                base = OBVZScoreMRStrategy(
                    obv_period=p, z_threshold=z,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
                return MultiTimeframeFilter(base)

            wf = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
            report = wf.run(make_obv_mr, df_1h_z, htf_df=df_4h)

            rob = int(report.robustness_score * 100)
            obv_mr_results.append({
                "period": period, "z_thresh": z_thresh,
                "rob": rob, "oos": report.oos_total_return,
                "trades": report.oos_total_trades,
            })

            marker = "★" if rob >= 66 else ""
            logger.info("    [%d/%d] OBV_Z_p%d_z%.1f: Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                        idx, total, period, z_thresh, rob,
                        report.oos_total_return, report.oos_total_trades, marker)

    logger.info("")
    mr_sorted = sorted(obv_mr_results, key=lambda x: (x["rob"], x["oos"]), reverse=True)
    best_mr = mr_sorted[0]
    logger.info("  Best OBV MR: p%d_z%.1f → %d%% rob, %+.2f%% OOS, %d trades",
                best_mr["period"], best_mr["z_thresh"],
                best_mr["rob"], best_mr["oos"], best_mr["trades"])

    # Robustness distribution
    rob_dist = {}
    for r in obv_mr_results:
        rb = r["rob"]
        rob_dist[rb] = rob_dist.get(rb, 0) + 1
    logger.info("  MR Robustness distribution:")
    for rb in sorted(rob_dist.keys(), reverse=True):
        logger.info("    %d%%: %d configs", rb, rob_dist[rb])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: OBV Donchian Breakout (1h + MTF)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: OBV Donchian Breakout (1h + MTF)")
    logger.info("-" * 72)
    logger.info("  Entry: OBV breaks N-bar channel + volume confirmation")
    logger.info("  Grid: dc_period × rr_ratio")
    logger.info("")

    obv_dc_results = []
    dc_periods = [12, 18, 24, 36]
    rr_ratios = [1.5, 2.0, 2.5]
    total = len(dc_periods) * len(rr_ratios)
    idx = 0

    for dc_p in dc_periods:
        for rr in rr_ratios:
            idx += 1

            def make_obv_dc(dp=dc_p, r=rr):
                base = OBVDonchianStrategy(
                    dc_period=dp, atr_sl_mult=2.0, rr_ratio=r,
                    vol_mult=0.8, cooldown_bars=6,
                )
                return MultiTimeframeFilter(base)

            wf = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
            report = wf.run(make_obv_dc, df_1h, htf_df=df_4h)

            rob = int(report.robustness_score * 100)
            obv_dc_results.append({
                "dc_period": dc_p, "rr": rr,
                "rob": rob, "oos": report.oos_total_return,
                "trades": report.oos_total_trades,
            })

            marker = "★" if rob >= 66 else ""
            logger.info("    [%d/%d] OBV_DC_p%d_rr%.1f: Rob=%d%%, OOS=%+.2f%%, Trades=%d %s",
                        idx, total, dc_p, rr, rob,
                        report.oos_total_return, report.oos_total_trades, marker)

    logger.info("")
    dc_sorted = sorted(obv_dc_results, key=lambda x: (x["rob"], x["oos"]), reverse=True)
    best_dc = dc_sorted[0]
    logger.info("  Best OBV DC: p%d_rr%.1f → %d%% rob, %+.2f%% OOS, %d trades",
                best_dc["dc_period"], best_dc["rr"],
                best_dc["rob"], best_dc["oos"], best_dc["trades"])

    # Robustness distribution
    rob_dist = {}
    for r in obv_dc_results:
        rb = r["rob"]
        rob_dist[rb] = rob_dist.get(rb, 0) + 1
    logger.info("  DC Robustness distribution:")
    for rb in sorted(rob_dist.keys(), reverse=True):
        logger.info("    %d%%: %d configs", rb, rob_dist[rb])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Portfolio Integration (if any ≥66% rob)
    # ═════════════════════════════════════════════════════════════
    overall_best = max(
        [best_mr, best_dc],
        key=lambda x: (x["rob"], x["oos"]),
    )

    if overall_best["rob"] >= 66:
        logger.info("-" * 72)
        logger.info("  PART 3: OBV Portfolio Integration (5th component)")
        logger.info("-" * 72)
        logger.info("")

        # Determine which strategy to use
        if overall_best == best_mr:
            bp, bz = best_mr["period"], best_mr["z_thresh"]
            df_1h_best = add_obv_zscore(df_1h.copy(), period=bp)

            def make_obv_best(p=bp, z=bz):
                base = OBVZScoreMRStrategy(
                    obv_period=p, z_threshold=z,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
                return MultiTimeframeFilter(base)

            label = f"1hOBV_MR"
        else:
            bdp, brr = best_dc["dc_period"], best_dc["rr"]
            df_1h_best = df_1h.copy()

            def make_obv_best(dp=bdp, r=brr):
                base = OBVDonchianStrategy(
                    dc_period=dp, atr_sl_mult=2.0, rr_ratio=r,
                    vol_mult=0.8, cooldown_bars=6,
                )
                return MultiTimeframeFilter(base)

            label = f"1hOBV_DC"

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
                strategy_factory=make_obv_best,
                df=df_1h_best, htf_df=df_4h,
                engine=engine_1h, weight=0.20, label=label,
            ),
        ]

        report_5c = wf.run_cross_tf(components_5comp)
        rob_5c = int(report_5c.robustness_score * 100)
        logger.info("  5-comp (10/40/10/20/20 + %s):", label)
        logger.info("    Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    rob_5c, report_5c.oos_total_return, report_5c.total_trades)
        for w in report_5c.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("    W%d: %s -> %+.2f%% %s",
                        w.window_id, " | ".join(parts), w.weighted_return, marker)
        logger.info("")

        # Also try as DC replacement (since OBV is volume-based, may complement MR better)
        wf2 = WalkForwardAnalyzer(n_windows=9)
        components_replace = [
            CrossTFComponent(
                strategy_factory=make_rsi_1h,
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.15, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_obv_best,
                df=df_1h_best, htf_df=df_4h,
                engine=engine_1h, weight=0.50, label=label,
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

        report_replace = wf2.run_cross_tf(components_replace)
        rob_rep = int(report_replace.robustness_score * 100)
        logger.info("  OBV replacing DC (15/50/10/25):")
        logger.info("    Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                    rob_rep, report_replace.oos_total_return, report_replace.total_trades)
        for w in report_replace.windows:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("    W%d: %s -> %+.2f%% %s",
                        w.window_id, " | ".join(parts), w.weighted_return, marker)
        logger.info("")

    else:
        logger.info("  Best OBV standalone rob=%d%% < 66%% → skipping portfolio integration.",
                    overall_best["rob"])
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: 15m OBV Test (if any 1h config ≥55%)
    # ═════════════════════════════════════════════════════════════
    if overall_best["rob"] >= 55:
        logger.info("-" * 72)
        logger.info("  PART 4: 15m OBV Test")
        logger.info("-" * 72)
        logger.info("")

        # Test best MR config on 15m
        if best_mr["rob"] >= 55:
            bp = best_mr["period"]
            bz = best_mr["z_thresh"]
            df_15m_z = add_obv_zscore(df_15m.copy(), period=bp)

            def make_obv_mr_15m(p=bp, z=bz):
                base = OBVZScoreMRStrategy(
                    obv_period=p, z_threshold=z,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
                )
                return MultiTimeframeFilter(base)

            wf = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
            report_15m = wf.run(make_obv_mr_15m, df_15m_z, htf_df=df_4h)
            rob_15m = int(report_15m.robustness_score * 100)
            logger.info("  OBV_MR 15m p%d_z%.1f: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                        bp, bz, rob_15m, report_15m.oos_total_return,
                        report_15m.oos_total_trades)

        # Test best DC config on 15m
        if best_dc["rob"] >= 55:
            bdp = best_dc["dc_period"]
            brr = best_dc["rr"]
            # Scale dc_period for 15m (4x more bars per unit time)
            dc_15m = bdp * 4

            def make_obv_dc_15m(dp=dc_15m, r=brr):
                base = OBVDonchianStrategy(
                    dc_period=dp, atr_sl_mult=2.0, rr_ratio=r,
                    vol_mult=0.8, cooldown_bars=12,
                )
                return MultiTimeframeFilter(base)

            wf = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
            report_15m_dc = wf.run(make_obv_dc_15m, df_15m, htf_df=df_4h)
            rob_15m_dc = int(report_15m_dc.robustness_score * 100)
            logger.info("  OBV_DC 15m p%d_rr%.1f: Rob=%d%%, OOS=%+.2f%%, Trades=%d",
                        dc_15m, brr, rob_15m_dc, report_15m_dc.oos_total_return,
                        report_15m_dc.oos_total_trades)

        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   SUMMARY
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 39 SUMMARY")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  OBV Z-Score MR best: p%d_z%.1f → %d%% rob, %+.2f%% OOS, %d trades",
                best_mr["period"], best_mr["z_thresh"],
                best_mr["rob"], best_mr["oos"], best_mr["trades"])
    logger.info("  OBV Donchian best:   p%d_rr%.1f → %d%% rob, %+.2f%% OOS, %d trades",
                best_dc["dc_period"], best_dc["rr"],
                best_dc["rob"], best_dc["oos"], best_dc["trades"])
    logger.info("")
    logger.info("  Baseline comparison: 88%% rob, +23.98%% OOS, 236 trades")
    logger.info("")

    if overall_best["rob"] < 66:
        logger.info("  CONCLUSION: OBV does NOT reach 66%% robustness threshold.")
        logger.info("  All volume-based indicators (VWAP, MFI, CMF, EFI, OBV) confirmed FAILED.")
        logger.info("  88%% robustness ceiling confirmed — no remaining untested indicators.")
    elif overall_best["rob"] < 88:
        logger.info("  CONCLUSION: OBV standalone viable (%d%% rob) but does NOT break 88%% ceiling.",
                    overall_best["rob"])
    else:
        logger.info("  CONCLUSION: OBV achieves %d%% robustness!", overall_best["rob"])

    logger.info("")
    logger.info("  Phase 39 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

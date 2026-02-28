#!/usr/bin/env python3
"""Phase 30 — Volume-Based Indicators & StochRSI: Untested Dimensions.

Previous phases exhaustively tested price-only oscillators:
  - RSI, WillR, CCI, Fisher, Z-Score, Ichimoku (all price-only)
  - VWAP (volume-weighted but only 55% rob)

Phase 30 explores genuinely different indicator families:
  1. CMF (Chaikin Money Flow): volume-weighted accumulation/distribution [-1,1]
  2. StochRSI: stochastic of RSI — more sensitive oscillator [0,100]
  3. EFI (Elder Force Index): volume × price change (momentum-volume hybrid)

Each indicator uses fundamentally different mathematics from all tested ones.

Plan:
  PART 1: CMF MR + MTF (1h) — period + threshold grid
  PART 2: StochRSI MR + MTF (1h) — period + threshold grid
  PART 3: EFI MR + MTF (1h) — period + threshold grid
  PART 4: Best on 15m (if 1h >= 55%)
  PART 5: Best as 5th component or WillR replacement in cross-TF
  PART 6: Summary
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
from src.strategy.base import BaseStrategy, Signal, TradeSignal
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase30")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase30.log", mode="w")
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


# ─── CMF Mean Reversion Strategy ─────────────────────────────────

class CMFMeanReversionStrategy(BaseStrategy):
    """Fade extremes using Chaikin Money Flow + Bollinger Band confirmation.

    CMF = sum(CLV * volume, N) / sum(volume, N)
    where CLV = ((close-low) - (high-close)) / (high-low)

    CMF is bounded [-1, 1] and measures buying/selling pressure via volume.
    This is fundamentally different from price-only oscillators (RSI, WillR, CCI).

    Entry rules:
      LONG:  CMF < -threshold AND close <= BB_lower (selling exhaustion)
      SHORT: CMF > +threshold AND close >= BB_upper (buying exhaustion)

    Exit: ATR-based SL/TP.
    """

    name = "cmf_mean_reversion"

    def __init__(
        self,
        cmf_period: int = 20,
        threshold: float = 0.20,
        bb_column_lower: str = "BBL_20_2.0_2.0",
        bb_column_upper: str = "BBU_20_2.0_2.0",
        bb_proximity: float = 1.01,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = "BTC/USDT:USDT",
    ) -> None:
        self.cmf_period = cmf_period
        self.threshold = threshold
        self.bb_lower = bb_column_lower
        self.bb_upper = bb_column_upper
        self.bb_proximity = bb_proximity
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self._last_entry_idx = -999
        self.symbol = symbol
        self._cmf_col = f"CMF_{cmf_period}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 30:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        cmf = last.get(self._cmf_col)
        atr = last.get("atr_14")
        bb_low = last.get(self.bb_lower)
        bb_up = last.get(self.bb_upper)

        if any(pd.isna(v) for v in [cmf, atr, bb_low, bb_up]):
            return self._hold(df)

        cmf = float(cmf)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        # LONG: CMF deeply negative (selling exhaustion) + near lower BB
        if cmf < -self.threshold and close <= bb_low * self.bb_proximity:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + abs(cmf + self.threshold) * 2.0)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "cmf": cmf},
            )

        # SHORT: CMF deeply positive (buying exhaustion) + near upper BB
        if cmf > self.threshold and close >= bb_up / self.bb_proximity:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + abs(cmf - self.threshold) * 2.0)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "cmf": cmf},
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
        return [self._cmf_col, "atr_14", self.bb_lower, self.bb_upper]


# ─── StochRSI Mean Reversion Strategy ────────────────────────────

class StochRSIMeanReversionStrategy(BaseStrategy):
    """Fade extremes using Stochastic RSI + Bollinger Band confirmation.

    StochRSI applies the Stochastic oscillator formula to RSI values:
      StochRSI = (RSI - min(RSI, N)) / (max(RSI, N) - min(RSI, N))

    More sensitive than plain RSI — detects oversold/overbought faster.
    Uses %K (smoothed StochRSI) for entries.

    Entry rules:
      LONG:  StochRSI_K < oversold AND close <= BB_lower
      SHORT: StochRSI_K > overbought AND close >= BB_upper

    Exit: ATR-based SL/TP.
    """

    name = "stochrsi_mean_reversion"

    def __init__(
        self,
        rsi_length: int = 14,
        stoch_length: int = 14,
        k_smooth: int = 3,
        d_smooth: int = 3,
        oversold: float = 15.0,
        overbought: float = 85.0,
        bb_column_lower: str = "BBL_20_2.0_2.0",
        bb_column_upper: str = "BBU_20_2.0_2.0",
        bb_proximity: float = 1.01,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = "BTC/USDT:USDT",
    ) -> None:
        self.rsi_length = rsi_length
        self.stoch_length = stoch_length
        self.k_smooth = k_smooth
        self.d_smooth = d_smooth
        self.oversold = oversold
        self.overbought = overbought
        self.bb_lower = bb_column_lower
        self.bb_upper = bb_column_upper
        self.bb_proximity = bb_proximity
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self._last_entry_idx = -999
        self.symbol = symbol
        self._k_col = f"STOCHRSIk_{rsi_length}_{stoch_length}_{k_smooth}_{d_smooth}"
        self._d_col = f"STOCHRSId_{rsi_length}_{stoch_length}_{k_smooth}_{d_smooth}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 30:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        k_val = last.get(self._k_col)
        atr = last.get("atr_14")
        bb_low = last.get(self.bb_lower)
        bb_up = last.get(self.bb_upper)

        if any(pd.isna(v) for v in [k_val, atr, bb_low, bb_up]):
            return self._hold(df)

        k_val = float(k_val)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        # LONG: StochRSI_K oversold + near lower BB
        if k_val < self.oversold and close <= bb_low * self.bb_proximity:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + (self.oversold - k_val) / 100.0)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "stochrsi_k": k_val},
            )

        # SHORT: StochRSI_K overbought + near upper BB
        if k_val > self.overbought and close >= bb_up / self.bb_proximity:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + (k_val - self.overbought) / 100.0)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "stochrsi_k": k_val},
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
        return [self._k_col, "atr_14", self.bb_lower, self.bb_upper]


# ─── EFI (Elder Force Index) Mean Reversion Strategy ─────────────

class EFIMeanReversionStrategy(BaseStrategy):
    """Fade extremes using Elder Force Index + Bollinger Band confirmation.

    EFI = EMA(close_change * volume, N)

    EFI combines price change with volume to measure force of moves.
    Extreme negative EFI = strong selling force (exhaustion candidate).
    Extreme positive EFI = strong buying force (exhaustion candidate).

    We normalize EFI by its rolling std to get z-score of force.

    Entry rules:
      LONG:  EFI_zscore < -threshold AND close <= BB_lower
      SHORT: EFI_zscore > +threshold AND close >= BB_upper

    Exit: ATR-based SL/TP.
    """

    name = "efi_mean_reversion"

    def __init__(
        self,
        efi_period: int = 13,
        zscore_lookback: int = 50,
        threshold: float = 2.0,
        bb_column_lower: str = "BBL_20_2.0_2.0",
        bb_column_upper: str = "BBU_20_2.0_2.0",
        bb_proximity: float = 1.01,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = "BTC/USDT:USDT",
    ) -> None:
        self.efi_period = efi_period
        self.zscore_lookback = zscore_lookback
        self.threshold = threshold
        self.bb_lower = bb_column_lower
        self.bb_upper = bb_column_upper
        self.bb_proximity = bb_proximity
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self._last_entry_idx = -999
        self.symbol = symbol
        self._efi_col = f"EFI_{efi_period}"
        self._efi_z_col = f"EFI_Z_{efi_period}_{zscore_lookback}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < self.zscore_lookback + 10:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        efi_z = last.get(self._efi_z_col)
        atr = last.get("atr_14")
        bb_low = last.get(self.bb_lower)
        bb_up = last.get(self.bb_upper)

        if any(pd.isna(v) for v in [efi_z, atr, bb_low, bb_up]):
            return self._hold(df)

        efi_z = float(efi_z)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        # LONG: extreme selling force (exhaustion) + near lower BB
        if efi_z < -self.threshold and close <= bb_low * self.bb_proximity:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + abs(efi_z + self.threshold) / 5.0)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "efi_z": efi_z},
            )

        # SHORT: extreme buying force (exhaustion) + near upper BB
        if efi_z > self.threshold and close >= bb_up / self.bb_proximity:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + abs(efi_z - self.threshold) / 5.0)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "efi_z": efi_z},
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
        return [self._efi_z_col, "atr_14", self.bb_lower, self.bb_upper]


# ─── Indicator Helpers ────────────────────────────────────────────

def add_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Chaikin Money Flow indicator."""
    col = f"CMF_{period}"
    if col not in df.columns:
        result = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=period)
        if result is not None:
            df[col] = result
    return df


def add_stochrsi(
    df: pd.DataFrame,
    rsi_length: int = 14,
    stoch_length: int = 14,
    k: int = 3,
    d: int = 3,
) -> pd.DataFrame:
    """Add Stochastic RSI indicator."""
    k_col = f"STOCHRSIk_{rsi_length}_{stoch_length}_{k}_{d}"
    if k_col not in df.columns:
        result = ta.stochrsi(
            df["close"], length=rsi_length, rsi_length=stoch_length,
            k=k, d=d,
        )
        if result is not None:
            df = pd.concat([df, result], axis=1)
    return df


def add_efi(
    df: pd.DataFrame, period: int = 13, zscore_lookback: int = 50,
) -> pd.DataFrame:
    """Add Elder Force Index and its z-score."""
    efi_col = f"EFI_{period}"
    z_col = f"EFI_Z_{period}_{zscore_lookback}"
    if efi_col not in df.columns:
        # EFI = EMA(close_change * volume, period)
        force = df["close"].diff() * df["volume"]
        df[efi_col] = force.ewm(span=period, adjust=False).mean()
    if z_col not in df.columns:
        efi = df[efi_col]
        rolling_mean = efi.rolling(zscore_lookback).mean()
        rolling_std = efi.rolling(zscore_lookback).std()
        df[z_col] = (efi - rolling_mean) / rolling_std.replace(0, np.nan)
    return df


def add_willr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Williams %R indicator."""
    col = f"WILLR_{period}"
    if col not in df.columns:
        df[col] = ta.willr(df["high"], df["low"], df["close"], length=period)
    return df


# ─── Data ─────────────────────────────────────────────────────────

def load_data(timeframe: str) -> pd.DataFrame:
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


# ─── Strategy Factories ──────────────────────────────────────────

def make_cmf_1h(
    period: int = 20, threshold: float = 0.20, cooldown: int = 6,
) -> MultiTimeframeFilter:
    base = CMFMeanReversionStrategy(
        cmf_period=period, threshold=threshold,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


def make_stochrsi_1h(
    rsi_length: int = 14, stoch_length: int = 14,
    k_smooth: int = 3, d_smooth: int = 3,
    oversold: float = 15.0, overbought: float = 85.0,
    cooldown: int = 6,
) -> MultiTimeframeFilter:
    base = StochRSIMeanReversionStrategy(
        rsi_length=rsi_length, stoch_length=stoch_length,
        k_smooth=k_smooth, d_smooth=d_smooth,
        oversold=oversold, overbought=overbought,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


def make_efi_1h(
    period: int = 13, zscore_lookback: int = 50,
    threshold: float = 2.0, cooldown: int = 6,
) -> MultiTimeframeFilter:
    base = EFIMeanReversionStrategy(
        efi_period=period, zscore_lookback=zscore_lookback,
        threshold=threshold,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


# 15m variants
def make_cmf_15m(
    period: int = 20, threshold: float = 0.20, cooldown: int = 12,
) -> MultiTimeframeFilter:
    base = CMFMeanReversionStrategy(
        cmf_period=period, threshold=threshold,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


def make_stochrsi_15m(
    rsi_length: int = 14, stoch_length: int = 14,
    k_smooth: int = 3, d_smooth: int = 3,
    oversold: float = 15.0, overbought: float = 85.0,
    cooldown: int = 12,
) -> MultiTimeframeFilter:
    base = StochRSIMeanReversionStrategy(
        rsi_length=rsi_length, stoch_length=stoch_length,
        k_smooth=k_smooth, d_smooth=d_smooth,
        oversold=oversold, overbought=overbought,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


def make_efi_15m(
    period: int = 13, zscore_lookback: int = 50,
    threshold: float = 2.0, cooldown: int = 12,
) -> MultiTimeframeFilter:
    base = EFIMeanReversionStrategy(
        efi_period=period, zscore_lookback=zscore_lookback,
        threshold=threshold,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


# Production baselines
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
        "Sharpe: %.2f",
        name, report.oos_total_return,
        int(report.robustness_score * 100),
        sum(1 for w in report.windows if w.out_of_sample.total_return > 0),
        report.total_windows,
        report.oos_total_trades,
        report.oos_avg_sharpe,
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
    logger.info("  PHASE 30 — Volume-Based Indicators & StochRSI")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  CMF: volume-weighted accumulation/distribution [-1,1]")
    logger.info("  StochRSI: stochastic of RSI — more sensitive [0,100]")
    logger.info("  EFI: Elder Force Index — volume × price change (z-scored)")
    logger.info("  All mathematically distinct from RSI/WillR/CCI/Fisher/Z-Score.")
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")

    # Add CMF for different periods
    for period in [10, 20]:
        df_1h = add_cmf(df_1h, period=period)
    df_15m = add_cmf(df_15m, period=20)

    # Add StochRSI
    df_1h = add_stochrsi(df_1h)
    df_15m = add_stochrsi(df_15m)

    # Add EFI for different periods/lookbacks
    for period in [13, 20]:
        for zl in [50]:
            df_1h = add_efi(df_1h, period=period, zscore_lookback=zl)
    df_15m = add_efi(df_15m, period=13, zscore_lookback=50)

    # Add WillR for portfolio comparison
    df_1h = add_willr(df_1h, 14)

    logger.info("1h data:  %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("15m data: %d bars (%s ~ %s)", len(df_15m),
                df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("")

    # Check volume data availability
    vol_1h = df_1h["volume"]
    logger.info("  Volume stats (1h): mean=%.0f, min=%.0f, max=%.0f, zeros=%d/%d",
                vol_1h.mean(), vol_1h.min(), vol_1h.max(),
                (vol_1h == 0).sum(), len(vol_1h))
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")
    wf_1h = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
    wf_15m = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
    wf_xtf = WalkForwardAnalyzer(n_windows=9)

    # Track results
    all_results: dict[str, tuple] = {}  # name -> (rob%, oos%, trades)
    best_cmf = ("", 0.0, None)  # (name, rob%, report)
    best_stochrsi = ("", 0.0, None)
    best_efi = ("", 0.0, None)

    # ═════════════════════════════════════════════════════════════
    #   PART 1: CMF Mean Reversion + MTF (1h) — Grid Search
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: CMF (Chaikin Money Flow) MR + MTF on 1h")
    logger.info("-" * 72)
    logger.info("  Testing: periods=[10,20] x thresholds=[0.10,0.15,0.20,0.25]")
    logger.info("  Config: SL=2.0ATR, TP=3.0ATR, cool=6, BB confirm")
    logger.info("")

    for period in [10, 20]:
        for threshold in [0.10, 0.15, 0.20, 0.25]:
            name = f"CMF_p{period}_t{threshold:.2f}"
            factory = lambda p=period, t=threshold: make_cmf_1h(
                period=p, threshold=t,
            )
            report = wf_1h.run(factory, df_1h, htf_df=df_4h)
            rob = report.robustness_score * 100
            log_wf_detail(name, report)
            logger.info("")

            all_results[name] = (rob, report.oos_total_return, report.oos_total_trades)
            if rob > best_cmf[1] or (
                rob == best_cmf[1] and report.oos_total_return >
                (best_cmf[2].oos_total_return if best_cmf[2] else -999)
            ):
                best_cmf = (name, rob, report)

    logger.info("  PART 1 BEST: %s (%.0f%% rob, %+.2f%% OOS, %d trades)",
                best_cmf[0], best_cmf[1],
                best_cmf[2].oos_total_return if best_cmf[2] else 0,
                best_cmf[2].oos_total_trades if best_cmf[2] else 0)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: StochRSI Mean Reversion + MTF (1h) — Grid Search
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: StochRSI MR + MTF on 1h")
    logger.info("-" * 72)
    logger.info("  Testing: oversold=[10,15,20] x overbought=[80,85,90]")
    logger.info("  Config: RSI=14, Stoch=14, K=3, D=3, SL=2.0ATR, TP=3.0ATR, cool=6")
    logger.info("")

    for oversold in [10.0, 15.0, 20.0]:
        for overbought in [80.0, 85.0, 90.0]:
            name = f"StochRSI_os{int(oversold)}_ob{int(overbought)}"
            factory = lambda os=oversold, ob=overbought: make_stochrsi_1h(
                oversold=os, overbought=ob,
            )
            report = wf_1h.run(factory, df_1h, htf_df=df_4h)
            rob = report.robustness_score * 100
            log_wf_detail(name, report)
            logger.info("")

            all_results[name] = (rob, report.oos_total_return, report.oos_total_trades)
            if rob > best_stochrsi[1] or (
                rob == best_stochrsi[1] and report.oos_total_return >
                (best_stochrsi[2].oos_total_return if best_stochrsi[2] else -999)
            ):
                best_stochrsi = (name, rob, report)

    logger.info("  PART 2 BEST: %s (%.0f%% rob, %+.2f%% OOS, %d trades)",
                best_stochrsi[0], best_stochrsi[1],
                best_stochrsi[2].oos_total_return if best_stochrsi[2] else 0,
                best_stochrsi[2].oos_total_trades if best_stochrsi[2] else 0)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: EFI Mean Reversion + MTF (1h) — Grid Search
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: Elder Force Index MR + MTF on 1h")
    logger.info("-" * 72)
    logger.info("  Testing: periods=[13,20] x thresholds=[1.5,2.0,2.5]")
    logger.info("  Config: zscore_lookback=50, SL=2.0ATR, TP=3.0ATR, cool=6")
    logger.info("")

    for period in [13, 20]:
        for threshold in [1.5, 2.0, 2.5]:
            name = f"EFI_p{period}_t{threshold}"
            factory = lambda p=period, t=threshold: make_efi_1h(
                period=p, threshold=t,
            )
            report = wf_1h.run(factory, df_1h, htf_df=df_4h)
            rob = report.robustness_score * 100
            log_wf_detail(name, report)
            logger.info("")

            all_results[name] = (rob, report.oos_total_return, report.oos_total_trades)
            if rob > best_efi[1] or (
                rob == best_efi[1] and report.oos_total_return >
                (best_efi[2].oos_total_return if best_efi[2] else -999)
            ):
                best_efi = (name, rob, report)

    logger.info("  PART 3 BEST: %s (%.0f%% rob, %+.2f%% OOS, %d trades)",
                best_efi[0], best_efi[1],
                best_efi[2].oos_total_return if best_efi[2] else 0,
                best_efi[2].oos_total_trades if best_efi[2] else 0)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Best indicators on 15m
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 4: Best indicators on 15m")
    logger.info("-" * 72)

    candidates = [
        ("CMF", best_cmf),
        ("StochRSI", best_stochrsi),
        ("EFI", best_efi),
    ]
    viable_15m = []

    for ind_name, (name, rob, report) in candidates:
        if rob >= 55:
            logger.info("  %s best 1h = %s (%.0f%% rob) — testing on 15m", ind_name, name, rob)
            logger.info("")

            if ind_name == "CMF":
                bp = int(name.split("_p")[1].split("_")[0])
                bt = float(name.split("_t")[1])
                for t_adj in [-0.05, 0, +0.05]:
                    t = round(bt + t_adj, 2)
                    if t < 0.05:
                        continue
                    nm = f"CMF_15m_p{bp}_t{t:.2f}"
                    fac = lambda p=bp, th=t: make_cmf_15m(period=p, threshold=th)
                    rep = wf_15m.run(fac, df_15m, htf_df=df_4h)
                    log_wf_detail(nm, rep)
                    logger.info("")
                    if rep.robustness_score * 100 >= 55:
                        viable_15m.append((nm, rep))

            elif ind_name == "StochRSI":
                os_val = int(name.split("_os")[1].split("_")[0])
                ob_val = int(name.split("_ob")[1])
                for os_adj in [-5, 0, +5]:
                    os_t = os_val + os_adj
                    if os_t < 5:
                        continue
                    nm = f"StochRSI_15m_os{os_t}_ob{ob_val}"
                    fac = lambda os=float(os_t), ob=float(ob_val): make_stochrsi_15m(
                        oversold=os, overbought=ob,
                    )
                    rep = wf_15m.run(fac, df_15m, htf_df=df_4h)
                    log_wf_detail(nm, rep)
                    logger.info("")
                    if rep.robustness_score * 100 >= 55:
                        viable_15m.append((nm, rep))

            elif ind_name == "EFI":
                bp = int(name.split("_p")[1].split("_")[0])
                bt = float(name.split("_t")[1])
                for t_adj in [-0.5, 0, +0.5]:
                    t = round(bt + t_adj, 1)
                    if t < 1.0:
                        continue
                    nm = f"EFI_15m_p{bp}_t{t}"
                    fac = lambda p=bp, th=t: make_efi_15m(period=p, threshold=th)
                    rep = wf_15m.run(fac, df_15m, htf_df=df_4h)
                    log_wf_detail(nm, rep)
                    logger.info("")
                    if rep.robustness_score * 100 >= 55:
                        viable_15m.append((nm, rep))

    if not any(rob >= 55 for _, (_, rob, _) in candidates):
        logger.info("  SKIP — no indicator reached 55%% robustness on 1h")
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 5: Cross-TF Portfolio Integration
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 5: Cross-TF Portfolio Integration")
    logger.info("-" * 72)

    # Find overall best indicator
    best_overall = max(
        [best_cmf, best_stochrsi, best_efi],
        key=lambda x: (x[1], x[2].oos_total_return if x[2] else -999),
    )

    if best_overall[1] >= 55:
        logger.info("  Baseline: 4-comp RSI/DC/RSI15/WR = 88%% rob, +23.98%% OOS")
        logger.info("  Testing: %s (%.0f%% rob) as 5th component / WillR replacement",
                     best_overall[0], best_overall[1])
        logger.info("")

        # Build factory for best indicator
        best_name = best_overall[0]
        if best_name.startswith("CMF"):
            bp = int(best_name.split("_p")[1].split("_")[0])
            bt = float(best_name.split("_t")[1])
            new_factory = lambda: make_cmf_1h(period=bp, threshold=bt)
            new_label = "1hCMF"
        elif best_name.startswith("StochRSI"):
            os_val = int(best_name.split("_os")[1].split("_")[0])
            ob_val = int(best_name.split("_ob")[1])
            new_factory = lambda: make_stochrsi_1h(oversold=float(os_val), overbought=float(ob_val))
            new_label = "1hStochRSI"
        else:
            bp = int(best_name.split("_p")[1].split("_")[0])
            bt = float(best_name.split("_t")[1])
            new_factory = lambda: make_efi_1h(period=bp, threshold=bt)
            new_label = "1hEFI"

        # 5-component test
        logger.info("  --- As 5th Component ---")
        logger.info("")

        weight_configs_5 = [
            (10, 40, 10, 20, 20),
            (10, 35, 10, 25, 20),
            (15, 35, 10, 20, 20),
        ]
        for w_rsi, w_dc, w_rsi15, w_wr, w_new in weight_configs_5:
            name = f"5comp_{w_rsi}/{w_dc}/{w_rsi15}/{w_wr}/{w_new}"
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
                    strategy_factory=new_factory, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_new / 100, label=new_label,
                ),
            ])
            log_cross_tf_detail(name, report)
            logger.info("")

        # WillR replacement test (4-comp)
        logger.info("  --- Replacing WillR (4-comp) ---")
        logger.info("")

        replace_configs = [
            (15, 50, 10, 25),
            (20, 40, 15, 25),
            (15, 40, 15, 30),
        ]
        for w_rsi, w_dc, w_rsi15, w_new in replace_configs:
            name = f"4comp_new_{w_rsi}/{w_dc}/{w_rsi15}/{w_new}"
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
                    strategy_factory=new_factory, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_new / 100, label=new_label,
                ),
            ])
            log_cross_tf_detail(name, report)
            logger.info("")
    else:
        logger.info("  SKIP — no indicator reached 55%% robustness on 1h")
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 6: Summary
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 30 SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    logger.info("  Part 1 — CMF (Chaikin Money Flow) MR:")
    for name, (rob, oos, trades) in sorted(all_results.items()):
        if name.startswith("CMF"):
            logger.info("    %s: OOS %+.2f%% | Rob %d%% | Trades %d",
                         name, oos, int(rob), trades)
    logger.info("  Best: %s (%.0f%% rob)", best_cmf[0], best_cmf[1])
    logger.info("")

    logger.info("  Part 2 — StochRSI MR:")
    for name, (rob, oos, trades) in sorted(all_results.items()):
        if name.startswith("StochRSI"):
            logger.info("    %s: OOS %+.2f%% | Rob %d%% | Trades %d",
                         name, oos, int(rob), trades)
    logger.info("  Best: %s (%.0f%% rob)", best_stochrsi[0], best_stochrsi[1])
    logger.info("")

    logger.info("  Part 3 — EFI (Elder Force Index) MR:")
    for name, (rob, oos, trades) in sorted(all_results.items()):
        if name.startswith("EFI"):
            logger.info("    %s: OOS %+.2f%% | Rob %d%% | Trades %d",
                         name, oos, int(rob), trades)
    logger.info("  Best: %s (%.0f%% rob)", best_efi[0], best_efi[1])
    logger.info("")

    logger.info("  Reference (current production):")
    logger.info("    4-comp 15/50/10/25: 88%% rob, +23.98%% OOS")
    logger.info("    WillR standalone:   77%% rob, +19.17%% OOS")
    logger.info("    RSI 1h standalone:  66%% rob, +13.29%% OOS")
    logger.info("    CCI standalone:     66%% rob, +13.48%% OOS")
    logger.info("")

    # Overall conclusion
    any_viable = any(x[1] >= 66 for x in [best_cmf, best_stochrsi, best_efi])
    if any_viable:
        best = max([best_cmf, best_stochrsi, best_efi],
                   key=lambda x: (x[1], x[2].oos_total_return if x[2] else -999))
        logger.info("  CONCLUSION: %s (%.0f%% rob, %+.2f%% OOS) worth investigating further.",
                     best[0], best[1], best[2].oos_total_return)
    else:
        logger.info("  CONCLUSION: Volume-based & StochRSI indicators do NOT break 88%% ceiling.")
        logger.info("  Further confirms: price-only vs volume-based makes no difference for BTC 1h MR.")

    logger.info("")
    logger.info("  Phase 30 complete.")


if __name__ == "__main__":
    main()

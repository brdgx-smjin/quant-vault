#!/usr/bin/env python3
"""Phase 32 — Hurst Exponent Regime Filter for Cross-TF Portfolio.

Previous phases established:
  - 4-comp Cross-TF (1hRSI/1hDC/15mRSI/1hWillR) = 88% rob, +23.98% OOS
  - 88% is structural ceiling (W2 Nov 20-Dec 2 unsolvable, ALL components negative)
  - ALL tested filters failed: ADX, chop, time-of-day, funding rate, session

Phase 32 Question: Can Hurst exponent (or Efficiency Ratio) detect the W2
trending regime and reduce losses there, without destroying other windows?

Approach:
  1. Compute rolling Hurst exponent on BTC 1h close prices
  2. Analyze Hurst distribution per walk-forward window
  3. Test Hurst as a regime-aware trade suppression filter
  4. Test Efficiency Ratio (Kaufman) as a simpler alternative
  5. Compare all filtered portfolios vs unfiltered baseline

Key insight: W2 was a strong trending period (BTC rallied). If Hurst > 0.5
during W2 but not during profitable windows, we can use it as a filter.
But if Hurst is high across ALL windows (typical for BTC), filter won't help.
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

logger = logging.getLogger("phase32")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase32.log", mode="w")
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


# ─── Hurst Exponent Computation ──────────────────────────────────

def hurst_rs(series: np.ndarray) -> float:
    """Estimate Hurst exponent using Rescaled Range (R/S) analysis.

    Args:
        series: 1D price or return array (at least 20 elements).

    Returns:
        Hurst exponent estimate. H > 0.5 = trending, H < 0.5 = mean-reverting.
    """
    n = len(series)
    if n < 20:
        return 0.5  # insufficient data

    # Use log-returns
    returns = np.diff(np.log(series))
    returns = returns[~np.isnan(returns)]
    if len(returns) < 16:
        return 0.5

    # Test multiple sub-series sizes
    sizes = []
    rs_values = []

    for size in [8, 16, 32, 64, 128, 256]:
        if size > len(returns) // 2:
            break

        n_subseries = len(returns) // size
        if n_subseries < 1:
            break

        rs_list = []
        for j in range(n_subseries):
            sub = returns[j * size:(j + 1) * size]
            mean_sub = sub.mean()
            cumdev = np.cumsum(sub - mean_sub)
            r = cumdev.max() - cumdev.min()
            s = sub.std(ddof=1)
            if s > 1e-10:
                rs_list.append(r / s)

        if rs_list:
            sizes.append(size)
            rs_values.append(np.mean(rs_list))

    if len(sizes) < 2:
        return 0.5

    # Linear regression: log(R/S) = H * log(n) + c
    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)

    # OLS
    x_mean = log_sizes.mean()
    y_mean = log_rs.mean()
    num = np.sum((log_sizes - x_mean) * (log_rs - y_mean))
    den = np.sum((log_sizes - x_mean) ** 2)

    if den < 1e-10:
        return 0.5

    h = num / den
    return float(np.clip(h, 0.0, 1.0))


def rolling_hurst(close: pd.Series, window: int = 168) -> pd.Series:
    """Compute rolling Hurst exponent over a window.

    Args:
        close: Close price series.
        window: Lookback window in bars.

    Returns:
        Series of Hurst exponent values.
    """
    values = close.values
    result = np.full(len(values), np.nan)

    for i in range(window, len(values)):
        result[i] = hurst_rs(values[i - window:i])

    return pd.Series(result, index=close.index, name=f"hurst_{window}")


def efficiency_ratio(close: pd.Series, window: int = 48) -> pd.Series:
    """Kaufman Efficiency Ratio (ER).

    ER = |direction| / volatility
    ER → 1.0 means trending, ER → 0.0 means choppy/mean-reverting.

    Args:
        close: Close price series.
        window: Lookback window in bars.

    Returns:
        Series of ER values [0, 1].
    """
    direction = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.clip(0.0, 1.0)
    return er.rename(f"ER_{window}")


# ─── Regime-Filtered Strategy Wrapper ─────────────────────────────

class HurstFilteredStrategy(BaseStrategy):
    """Wrapper that suppresses signals when Hurst > threshold (trending).

    When the regime is trending (Hurst > threshold), this wrapper
    returns HOLD instead of the base strategy's signal.
    """

    name = "hurst_filtered"

    def __init__(
        self,
        base_strategy: BaseStrategy,
        hurst_series: pd.Series,
        threshold: float = 0.55,
        suppress_in_trending: bool = True,
    ) -> None:
        self.base = base_strategy
        self.hurst_series = hurst_series
        self.threshold = threshold
        self.suppress_in_trending = suppress_in_trending
        self.name = f"hurst_filtered_{base_strategy.name}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        current_time = df.index[-1]

        # Get nearest Hurst value
        hurst_val = np.nan
        if current_time in self.hurst_series.index:
            hurst_val = self.hurst_series.loc[current_time]
        else:
            # Find nearest prior value
            valid = self.hurst_series[:current_time].dropna()
            if len(valid) > 0:
                hurst_val = valid.iloc[-1]

        # If Hurst indicates trending and we want to suppress MR signals
        if not np.isnan(hurst_val):
            if self.suppress_in_trending and hurst_val > self.threshold:
                return TradeSignal(
                    signal=Signal.HOLD,
                    symbol=getattr(self.base, "symbol", "BTC/USDT:USDT"),
                    price=float(df["close"].iloc[-1]),
                    timestamp=current_time,
                )

        return self.base.generate_signal(df)

    def get_required_indicators(self) -> list[str]:
        return self.base.get_required_indicators()


class ERFilteredStrategy(BaseStrategy):
    """Wrapper that suppresses signals when Efficiency Ratio > threshold."""

    name = "er_filtered"

    def __init__(
        self,
        base_strategy: BaseStrategy,
        er_series: pd.Series,
        threshold: float = 0.3,
    ) -> None:
        self.base = base_strategy
        self.er_series = er_series
        self.threshold = threshold
        self.name = f"er_filtered_{base_strategy.name}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        current_time = df.index[-1]

        er_val = np.nan
        if current_time in self.er_series.index:
            er_val = self.er_series.loc[current_time]
        else:
            valid = self.er_series[:current_time].dropna()
            if len(valid) > 0:
                er_val = valid.iloc[-1]

        if not np.isnan(er_val) and er_val > self.threshold:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=getattr(self.base, "symbol", "BTC/USDT:USDT"),
                price=float(df["close"].iloc[-1]),
                timestamp=current_time,
            )

        return self.base.generate_signal(df)

    def get_required_indicators(self) -> list[str]:
        return self.base.get_required_indicators()


# ─── Strategy Factories ──────────────────────────────────────────

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


def make_hurst_filtered_rsi_1h(hurst_1h: pd.Series, threshold: float):
    def factory():
        base = RSIMeanReversionStrategy(
            rsi_oversold=35, rsi_overbought=65,
            atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
        )
        filtered = HurstFilteredStrategy(base, hurst_1h, threshold)
        return MultiTimeframeFilter(filtered)
    return factory


def make_hurst_filtered_dc_1h(hurst_1h: pd.Series, threshold: float):
    """Note: DC is trend-following, so we DON'T filter it.
    We only filter MR strategies (RSI, WillR)."""
    def factory():
        base = DonchianTrendStrategy(
            entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
            vol_mult=0.8, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)
    return factory


def make_hurst_filtered_rsi_15m(hurst_15m: pd.Series, threshold: float):
    def factory():
        base = RSIMeanReversionStrategy(
            rsi_oversold=35, rsi_overbought=65,
            atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
        )
        filtered = HurstFilteredStrategy(base, hurst_15m, threshold)
        return MultiTimeframeFilter(filtered)
    return factory


def make_hurst_filtered_willr_1h(hurst_1h: pd.Series, threshold: float):
    def factory():
        base = WilliamsRMeanReversionStrategy(
            willr_period=14, oversold_level=90.0, overbought_level=90.0,
            atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
        )
        filtered = HurstFilteredStrategy(base, hurst_1h, threshold)
        return MultiTimeframeFilter(filtered)
    return factory


def make_er_filtered_factory(er_series, threshold, strategy_class, strategy_kwargs, is_15m=False):
    """Generic factory for ER-filtered strategies."""
    def factory():
        base = strategy_class(**strategy_kwargs)
        filtered = ERFilteredStrategy(base, er_series, threshold)
        return MultiTimeframeFilter(filtered)
    return factory


# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 32 — Hurst Exponent & Efficiency Ratio Regime Filter")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Goal: Can regime detection (Hurst/ER) improve W2 performance?")
    logger.info("  Baseline: 4-comp 15/50/10/25, 88%% rob, +23.98%% OOS")
    logger.info("  W2: ALL 4 components negative (-5.59/-2.46/-2.60/-4.06)")
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

    # ═════════════════════════════════════════════════════════════
    #   PART 1: Hurst Exponent & ER Analysis
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: Rolling Hurst Exponent & Efficiency Ratio Analysis")
    logger.info("-" * 72)
    logger.info("")

    # Compute rolling Hurst for multiple lookback windows
    hurst_windows = [96, 168, 336]  # 4d, 7d, 14d in 1h bars
    hurst_results_1h = {}
    for hw in hurst_windows:
        logger.info("  Computing rolling Hurst (1h, window=%d bars / %dd)...", hw, hw // 24)
        h = rolling_hurst(df_1h["close"], window=hw)
        hurst_results_1h[hw] = h
        valid = h.dropna()
        logger.info("    Mean=%.3f, Std=%.3f, Min=%.3f, Max=%.3f, N=%d",
                     valid.mean(), valid.std(), valid.min(), valid.max(), len(valid))

    # Compute for 15m too (scaled windows)
    hurst_15m_window = 672  # 7 days in 15m bars
    logger.info("  Computing rolling Hurst (15m, window=%d bars / 7d)...", hurst_15m_window)
    hurst_15m = rolling_hurst(df_15m["close"], window=hurst_15m_window)
    valid_15m = hurst_15m.dropna()
    logger.info("    Mean=%.3f, Std=%.3f, Min=%.3f, Max=%.3f",
                 valid_15m.mean(), valid_15m.std(), valid_15m.min(), valid_15m.max())
    logger.info("")

    # Compute Efficiency Ratio
    er_windows = [24, 48, 96]  # 1d, 2d, 4d
    er_results_1h = {}
    for ew in er_windows:
        logger.info("  Computing Efficiency Ratio (1h, window=%d bars)...", ew)
        e = efficiency_ratio(df_1h["close"], window=ew)
        er_results_1h[ew] = e
        valid_e = e.dropna()
        logger.info("    Mean=%.3f, Std=%.3f, Min=%.3f, Max=%.3f",
                     valid_e.mean(), valid_e.std(), valid_e.min(), valid_e.max())

    er_15m = efficiency_ratio(df_15m["close"], window=192)  # 2d in 15m
    logger.info("  Computing Efficiency Ratio (15m, window=192 bars / 2d)...")
    valid_er_15m = er_15m.dropna()
    logger.info("    Mean=%.3f, Std=%.3f, Min=%.3f, Max=%.3f",
                 valid_er_15m.mean(), valid_er_15m.std(), valid_er_15m.min(), valid_er_15m.max())
    logger.info("")

    # ─── Analyze per WF window ────────────────────────────────────
    # Date ranges for 9 windows (from Phase 31 log)
    common_start = max(df_1h.index[0], df_15m.index[0])
    common_end = min(df_1h.index[-1], df_15m.index[-1])
    total_td = common_end - common_start
    test_td = total_td * 0.3 / 9

    logger.info("  Hurst & ER per walk-forward window:")
    logger.info("  %8s  %12s  %12s  %8s  %8s  %8s  %8s",
                "Window", "Start", "End",
                "Hurst96", "Hurst168", "Hurst336", "ER48")

    window_hurst = {}
    window_er = {}
    for i in range(9):
        test_end = common_end - (8 - i) * test_td
        test_start = test_end - test_td

        h_vals = {}
        for hw in hurst_windows:
            h = hurst_results_1h[hw]
            mask = (h.index >= test_start) & (h.index < test_end)
            vals = h[mask].dropna()
            h_vals[hw] = vals.mean() if len(vals) > 0 else np.nan

        e_vals = {}
        for ew in er_windows:
            e = er_results_1h[ew]
            mask = (e.index >= test_start) & (e.index < test_end)
            vals = e[mask].dropna()
            e_vals[ew] = vals.mean() if len(vals) > 0 else np.nan

        window_hurst[i] = h_vals
        window_er[i] = e_vals

        w_marker = " ←W2" if i == 1 else ""
        logger.info("  W%d  %12s  %12s  %8.3f  %8.3f  %8.3f  %8.3f%s",
                     i + 1,
                     str(test_start.date()), str(test_end.date()),
                     h_vals.get(96, np.nan),
                     h_vals.get(168, np.nan),
                     h_vals.get(336, np.nan),
                     e_vals.get(48, np.nan),
                     w_marker)

    logger.info("")

    # Check if W2 Hurst is distinctly different
    w2_hurst_168 = window_hurst[1].get(168, np.nan)
    other_hurst_168 = [window_hurst[i].get(168, np.nan) for i in range(9) if i != 1]
    other_hurst_168 = [v for v in other_hurst_168 if not np.isnan(v)]

    if not np.isnan(w2_hurst_168) and other_hurst_168:
        other_mean = np.mean(other_hurst_168)
        other_std = np.std(other_hurst_168)
        z_score = (w2_hurst_168 - other_mean) / other_std if other_std > 0 else 0
        logger.info("  W2 Hurst(168) z-score vs other windows: %.2f", z_score)
        logger.info("    W2=%.3f, Others mean=%.3f, std=%.3f",
                     w2_hurst_168, other_mean, other_std)
        if abs(z_score) < 1.0:
            logger.info("    → W2 Hurst NOT significantly different from other windows.")
            logger.info("    → Hurst may NOT be able to discriminate W2 from other windows.")
        else:
            logger.info("    → W2 Hurst is %.1f std from mean — potentially discriminative.", z_score)
    logger.info("")

    # Check ER similarly
    w2_er_48 = window_er[1].get(48, np.nan)
    other_er_48 = [window_er[i].get(48, np.nan) for i in range(9) if i != 1]
    other_er_48 = [v for v in other_er_48 if not np.isnan(v)]

    if not np.isnan(w2_er_48) and other_er_48:
        other_mean_er = np.mean(other_er_48)
        other_std_er = np.std(other_er_48)
        z_score_er = (w2_er_48 - other_mean_er) / other_std_er if other_std_er > 0 else 0
        logger.info("  W2 ER(48) z-score vs other windows: %.2f", z_score_er)
        logger.info("    W2=%.3f, Others mean=%.3f, std=%.3f",
                     w2_er_48, other_mean_er, other_std_er)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Hurst-Filtered Cross-TF Portfolio WF
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: Hurst-Filtered Cross-TF Portfolio")
    logger.info("-" * 72)
    logger.info("  Filter: suppress MR signals when Hurst > threshold (trending).")
    logger.info("  DC (trend-following) is NOT filtered — it should benefit from trends.")
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")

    # Use Hurst(168) = 7-day window as primary indicator
    hurst_1h_168 = hurst_results_1h[168]

    # Test multiple thresholds
    hurst_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]
    hurst_filter_results = []

    for ht in hurst_thresholds:
        logger.info("  Testing Hurst threshold %.2f (suppress MR when Hurst > %.2f):", ht, ht)

        wf = WalkForwardAnalyzer(n_windows=9)
        components = [
            CrossTFComponent(
                strategy_factory=make_hurst_filtered_rsi_1h(hurst_1h_168, ht),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.15, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_hurst_filtered_dc_1h(hurst_1h_168, ht),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.50, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=make_hurst_filtered_rsi_15m(hurst_15m, ht),
                df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=0.10, label="15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_hurst_filtered_willr_1h(hurst_1h_168, ht),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hWillR",
            ),
        ]

        report = wf.run_cross_tf(components)

        # Extract per-window returns
        per_window = []
        for w in report.windows:
            per_window.append(w.weighted_return)

        hurst_filter_results.append({
            "threshold": ht,
            "oos_return": report.oos_total_return,
            "robustness": int(report.robustness_score * 100),
            "profitable_windows": report.oos_profitable_windows,
            "total_windows": report.total_windows,
            "trades": report.total_trades,
            "per_window": per_window,
        })

        logger.info("    OOS: %+.2f%% | Rob: %d%% (%d/%d) | Trades: %d",
                     report.oos_total_return,
                     int(report.robustness_score * 100),
                     report.oos_profitable_windows,
                     report.total_windows,
                     report.total_trades)
        for w_idx, w in enumerate(report.windows):
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("      W%d: %s -> %+.2f%% %s",
                         w_idx + 1, " | ".join(parts), w.weighted_return, marker)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Efficiency Ratio Filtered Portfolio
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: Efficiency Ratio Filtered Cross-TF Portfolio")
    logger.info("-" * 72)
    logger.info("  Filter: suppress MR signals when ER > threshold (trending).")
    logger.info("")

    er_1h_48 = er_results_1h[48]

    er_thresholds = [0.15, 0.20, 0.25, 0.30, 0.40]
    er_filter_results = []

    for et in er_thresholds:
        logger.info("  Testing ER threshold %.2f:", et)

        def make_er_rsi_1h(er_s=er_1h_48, t=et):
            def factory():
                base = RSIMeanReversionStrategy(
                    rsi_oversold=35, rsi_overbought=65,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
                filtered = ERFilteredStrategy(base, er_s, t)
                return MultiTimeframeFilter(filtered)
            return factory

        def make_er_dc_1h():
            # DC not filtered
            def factory():
                base = DonchianTrendStrategy(
                    entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
                    vol_mult=0.8, cooldown_bars=6,
                )
                return MultiTimeframeFilter(base)
            return factory

        def make_er_rsi_15m(er_s=er_15m, t=et):
            def factory():
                base = RSIMeanReversionStrategy(
                    rsi_oversold=35, rsi_overbought=65,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
                )
                filtered = ERFilteredStrategy(base, er_s, t)
                return MultiTimeframeFilter(filtered)
            return factory

        def make_er_willr_1h(er_s=er_1h_48, t=et):
            def factory():
                base = WilliamsRMeanReversionStrategy(
                    willr_period=14, oversold_level=90.0, overbought_level=90.0,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
                filtered = ERFilteredStrategy(base, er_s, t)
                return MultiTimeframeFilter(filtered)
            return factory

        wf = WalkForwardAnalyzer(n_windows=9)
        components = [
            CrossTFComponent(
                strategy_factory=make_er_rsi_1h(),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.15, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_er_dc_1h(),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.50, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=make_er_rsi_15m(),
                df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=0.10, label="15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_er_willr_1h(),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hWillR",
            ),
        ]

        report = wf.run_cross_tf(components)

        per_window = [w.weighted_return for w in report.windows]

        er_filter_results.append({
            "threshold": et,
            "oos_return": report.oos_total_return,
            "robustness": int(report.robustness_score * 100),
            "profitable_windows": report.oos_profitable_windows,
            "total_windows": report.total_windows,
            "trades": report.total_trades,
            "per_window": per_window,
        })

        logger.info("    OOS: %+.2f%% | Rob: %d%% (%d/%d) | Trades: %d",
                     report.oos_total_return,
                     int(report.robustness_score * 100),
                     report.oos_profitable_windows,
                     report.total_windows,
                     report.total_trades)
        for w_idx, w in enumerate(report.windows):
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("      W%d: %s -> %+.2f%% %s",
                         w_idx + 1, " | ".join(parts), w.weighted_return, marker)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Hurst-Filtered MR ONLY (no portfolio)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 4: Standalone Hurst Analysis — Filter ALL components")
    logger.info("-" * 72)
    logger.info("  Also filter DC when Hurst < 0.5 (suppress trend-following in MR regime)")
    logger.info("")

    # Test: filter MR when Hurst > 0.55, AND filter DC when Hurst < 0.45
    # This is the "full regime-switching" approach
    def make_inverse_hurst_dc(hurst_s, threshold):
        """DC filtered: suppress when Hurst < threshold (mean-reverting, bad for DC)."""
        def factory():
            base = DonchianTrendStrategy(
                entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
                vol_mult=0.8, cooldown_bars=6,
            )
            # Invert: suppress when Hurst < threshold (NOT trending)
            filtered = HurstFilteredStrategy(base, hurst_s, threshold, suppress_in_trending=False)
            return MultiTimeframeFilter(filtered)
        return factory

    class InverseHurstFilter(BaseStrategy):
        """Suppress signals when Hurst BELOW threshold (for trend strategies)."""
        name = "inv_hurst_filtered"

        def __init__(self, base_strategy, hurst_series, threshold):
            self.base = base_strategy
            self.hurst_series = hurst_series
            self.threshold = threshold
            self.name = f"inv_hurst_{base_strategy.name}"

        def generate_signal(self, df):
            current_time = df.index[-1]
            hurst_val = np.nan
            if current_time in self.hurst_series.index:
                hurst_val = self.hurst_series.loc[current_time]
            else:
                valid = self.hurst_series[:current_time].dropna()
                if len(valid) > 0:
                    hurst_val = valid.iloc[-1]

            # Suppress trend-following when market is mean-reverting
            if not np.isnan(hurst_val) and hurst_val < self.threshold:
                return TradeSignal(
                    signal=Signal.HOLD,
                    symbol=getattr(self.base, "symbol", "BTC/USDT:USDT"),
                    price=float(df["close"].iloc[-1]),
                    timestamp=current_time,
                )
            return self.base.generate_signal(df)

        def get_required_indicators(self):
            return self.base.get_required_indicators()

    # Full regime switch test
    for mr_thresh, dc_thresh in [(0.55, 0.45), (0.60, 0.40), (0.50, 0.50)]:
        logger.info("  Regime switch: suppress MR when Hurst > %.2f, DC when Hurst < %.2f:",
                     mr_thresh, dc_thresh)

        def make_rs_dc(h=hurst_1h_168, t=dc_thresh):
            def factory():
                base = DonchianTrendStrategy(
                    entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
                    vol_mult=0.8, cooldown_bars=6,
                )
                filtered = InverseHurstFilter(base, h, t)
                return MultiTimeframeFilter(filtered)
            return factory

        wf = WalkForwardAnalyzer(n_windows=9)
        components = [
            CrossTFComponent(
                strategy_factory=make_hurst_filtered_rsi_1h(hurst_1h_168, mr_thresh),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.15, label="1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_rs_dc(),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.50, label="1hDC",
            ),
            CrossTFComponent(
                strategy_factory=make_hurst_filtered_rsi_15m(hurst_15m, mr_thresh),
                df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=0.10, label="15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=make_hurst_filtered_willr_1h(hurst_1h_168, mr_thresh),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.25, label="1hWillR",
            ),
        ]

        report = wf.run_cross_tf(components)

        logger.info("    OOS: %+.2f%% | Rob: %d%% (%d/%d) | Trades: %d",
                     report.oos_total_return,
                     int(report.robustness_score * 100),
                     report.oos_profitable_windows,
                     report.total_windows,
                     report.total_trades)
        for w_idx, w in enumerate(report.windows):
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info("      W%d: %s -> %+.2f%% %s",
                         w_idx + 1, " | ".join(parts), w.weighted_return, marker)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 5: Summary & Comparison
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 32 SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    logger.info("  Baseline: 4-comp Fixed (15/50/10/25)")
    logger.info("    OOS Return: +23.98%% | Robustness: 88%% | Trades: 236")
    logger.info("")

    logger.info("  Hurst-Filtered (MR only, DC unfiltered):")
    logger.info("  %-20s  %10s  %6s  %7s",
                "Threshold", "OOS Ret", "Rob%", "Trades")
    logger.info("  " + "-" * 50)
    for r in hurst_filter_results:
        logger.info("  Hurst > %-10.2f  %+9.2f%%  %5d%%  %6d",
                     r["threshold"], r["oos_return"], r["robustness"], r["trades"])
    logger.info("")

    logger.info("  ER-Filtered (MR only, DC unfiltered):")
    logger.info("  %-20s  %10s  %6s  %7s",
                "Threshold", "OOS Ret", "Rob%", "Trades")
    logger.info("  " + "-" * 50)
    for r in er_filter_results:
        logger.info("  ER > %-13.2f  %+9.2f%%  %5d%%  %6d",
                     r["threshold"], r["oos_return"], r["robustness"], r["trades"])
    logger.info("")

    # Check if any filter improved robustness beyond 88%
    any_improvement = False
    for r in hurst_filter_results + er_filter_results:
        if r["robustness"] > 88:
            any_improvement = True
            logger.info("  ★ IMPROVEMENT: %s with threshold %.2f → %d%% robustness, %+.2f%% OOS",
                         "Hurst" if "threshold" in str(r) else "ER",
                         r["threshold"], r["robustness"], r["oos_return"])

    if not any_improvement:
        logger.info("  CONCLUSION: No regime filter breaks the 88%% robustness ceiling.")
        logger.info("  W2 is NOT detectable by Hurst or ER — it's a structural market event,")
        logger.info("  not a simple regime shift.")
    logger.info("")

    # Best achievable with filter
    best_hurst = max(hurst_filter_results, key=lambda x: x["oos_return"])
    best_er = max(er_filter_results, key=lambda x: x["oos_return"])
    logger.info("  Best Hurst: threshold=%.2f, OOS=%+.2f%%, Rob=%d%%",
                 best_hurst["threshold"], best_hurst["oos_return"], best_hurst["robustness"])
    logger.info("  Best ER:    threshold=%.2f, OOS=%+.2f%%, Rob=%d%%",
                 best_er["threshold"], best_er["oos_return"], best_er["robustness"])
    logger.info("")

    logger.info("  Phase 32 complete.")


if __name__ == "__main__":
    main()

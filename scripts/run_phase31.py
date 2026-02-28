#!/usr/bin/env python3
"""Phase 31 — Adaptive Weight Allocation for Cross-TF Portfolio.

Previous phases established:
  - 4-comp Cross-TF (1hRSI/1hDC/15mRSI/1hWillR) = 88% rob, +23.98% OOS
  - 88% is structural ceiling (W2 Nov 20-Dec 2 unsolvable, ALL components negative)
  - 303/375 static weight combos achieve 88%
  - Fixed best: 15/50/10/25

Phase 31 Question: Can DYNAMIC weight allocation improve OOS return or
risk-adjusted metrics while maintaining 88% robustness?

Approach:
  1. Collect per-component per-window OOS returns (9 windows × 4 components)
  2. Apply adaptive weighting schemes POST-HOC:
     - Scheme A: Equal weight (25/25/25/25)
     - Scheme B: Production fixed (15/50/10/25)
     - Scheme C: Momentum — weight ∝ cumulative return of last K windows
     - Scheme D: Inverse volatility — weight ∝ 1/std of last K returns
     - Scheme E: Best recent — overweight best performer from last window
     - Scheme F: Defensive — reduce exposure when trailing (all-negative recent)
  3. Also test: volatility-scaled POSITION SIZING (ATR-based sizing per window)
  4. Summary: compare all schemes on robustness, return, max drawdown

Key insight: Since component trades/returns are independent of weights,
we can collect returns once and test all weighting schemes post-hoc.
This is purely a PORTFOLIO ALLOCATION study, not a new strategy.
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

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase31")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase31.log", mode="w")
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


# ─── Adaptive Weighting Schemes ──────────────────────────────────

def normalize_weights(weights: np.ndarray, min_weight: float = 0.05) -> np.ndarray:
    """Normalize weights to sum to 1.0 with minimum allocation."""
    n = len(weights)
    weights = np.maximum(weights, min_weight)
    return weights / weights.sum()


def scheme_equal(
    component_returns: np.ndarray,  # shape: (n_windows, n_components)
    window_idx: int,
) -> np.ndarray:
    """Equal weight: 25/25/25/25."""
    n = component_returns.shape[1]
    return np.ones(n) / n


def scheme_fixed_production(
    component_returns: np.ndarray,
    window_idx: int,
) -> np.ndarray:
    """Fixed production weights: 15/50/10/25."""
    return np.array([0.15, 0.50, 0.10, 0.25])


def scheme_momentum(
    component_returns: np.ndarray,
    window_idx: int,
    lookback: int = 2,
) -> np.ndarray:
    """Weight proportional to cumulative return over last K windows.

    Positive recent performers get more weight.
    Uses softmax-like normalization to handle negative returns.
    """
    n = component_returns.shape[1]
    if window_idx < 1:
        return np.ones(n) / n

    start = max(0, window_idx - lookback)
    recent = component_returns[start:window_idx, :]  # shape: (K, n_components)
    cum_returns = recent.sum(axis=0)  # sum of recent returns per component

    # Softmax to convert returns to weights (handles negatives gracefully)
    # Temperature controls how aggressively we re-weight
    temp = 5.0
    exp_w = np.exp(cum_returns / temp)
    return normalize_weights(exp_w)


def scheme_momentum_aggressive(
    component_returns: np.ndarray,
    window_idx: int,
    lookback: int = 3,
) -> np.ndarray:
    """More aggressive momentum — higher temperature, longer lookback."""
    n = component_returns.shape[1]
    if window_idx < 1:
        return np.ones(n) / n

    start = max(0, window_idx - lookback)
    recent = component_returns[start:window_idx, :]
    cum_returns = recent.sum(axis=0)

    temp = 3.0
    exp_w = np.exp(cum_returns / temp)
    return normalize_weights(exp_w)


def scheme_inverse_vol(
    component_returns: np.ndarray,
    window_idx: int,
    lookback: int = 3,
) -> np.ndarray:
    """Weight inversely proportional to recent return volatility.

    More stable components get more weight — risk parity concept.
    """
    n = component_returns.shape[1]
    if window_idx < 2:
        return np.ones(n) / n

    start = max(0, window_idx - lookback)
    recent = component_returns[start:window_idx, :]

    if len(recent) < 2:
        return np.ones(n) / n

    stds = recent.std(axis=0)
    stds = np.maximum(stds, 0.01)  # floor to avoid division by zero
    inv_vol = 1.0 / stds
    return normalize_weights(inv_vol)


def scheme_best_recent(
    component_returns: np.ndarray,
    window_idx: int,
) -> np.ndarray:
    """Overweight the best performer from last window.

    Winner-take-most: best gets 40%, others split remaining 60%.
    """
    n = component_returns.shape[1]
    if window_idx < 1:
        return np.ones(n) / n

    last = component_returns[window_idx - 1, :]
    best_idx = np.argmax(last)

    weights = np.full(n, 0.60 / (n - 1))
    weights[best_idx] = 0.40
    return weights


def scheme_defensive(
    component_returns: np.ndarray,
    window_idx: int,
    lookback: int = 2,
    reduction: float = 0.5,
) -> np.ndarray:
    """Reduce total exposure when all components trailing.

    If ALL components had negative average return recently, reduce all
    weights by `reduction` factor (rest goes to cash). Otherwise, equal.

    This models a "risk-off" regime switch.
    """
    n = component_returns.shape[1]
    base = np.ones(n) / n

    if window_idx < 1:
        return base

    start = max(0, window_idx - lookback)
    recent = component_returns[start:window_idx, :]
    avg_returns = recent.mean(axis=0)

    # If ALL components negative recently → reduce exposure
    if np.all(avg_returns < 0):
        return base * reduction
    # If MOST components negative → mild reduction
    elif np.sum(avg_returns < 0) >= n - 1:
        return base * (1 - (1 - reduction) * 0.5)
    else:
        return base


def scheme_defensive_momentum(
    component_returns: np.ndarray,
    window_idx: int,
    lookback: int = 2,
    reduction: float = 0.5,
) -> np.ndarray:
    """Combination: momentum weights + defensive exposure reduction.

    Uses momentum for relative weights AND reduces total exposure
    when all components trailing.
    """
    n = component_returns.shape[1]
    if window_idx < 1:
        return np.ones(n) / n

    # Momentum weights
    start = max(0, window_idx - lookback)
    recent = component_returns[start:window_idx, :]
    cum_returns = recent.sum(axis=0)

    temp = 5.0
    exp_w = np.exp(cum_returns / temp)
    weights = normalize_weights(exp_w)

    # Defensive reduction
    avg_returns = recent.mean(axis=0)
    if np.all(avg_returns < 0):
        weights *= reduction
    elif np.sum(avg_returns < 0) >= n - 1:
        weights *= (1 - (1 - reduction) * 0.5)

    return weights


# ─── Adaptive Simulation ─────────────────────────────────────────

def simulate_adaptive(
    component_returns: np.ndarray,
    scheme_fn,
    scheme_name: str,
) -> dict:
    """Apply adaptive weighting scheme to collected component returns.

    Args:
        component_returns: shape (n_windows, n_components), OOS returns per window
        scheme_fn: callable(component_returns, window_idx) -> weights array
        scheme_name: display name

    Returns:
        dict with portfolio metrics
    """
    n_windows, n_comp = component_returns.shape
    portfolio_returns = []
    weight_history = []

    for w in range(n_windows):
        weights = scheme_fn(component_returns, w)
        weight_history.append(weights.copy())

        # Portfolio return = weighted sum (weights may sum to < 1 for defensive)
        port_ret = np.dot(weights, component_returns[w, :])
        # If total weight < 1, the "cash" portion earns 0
        portfolio_returns.append(port_ret)

    portfolio_returns = np.array(portfolio_returns)
    weight_history = np.array(weight_history)

    # Compounded return
    compounded = 1.0
    for r in portfolio_returns:
        compounded *= (1 + r / 100)
    total_return = (compounded - 1) * 100

    # Robustness
    profitable = int(np.sum(portfolio_returns > 0))
    robustness = profitable / n_windows

    # Max drawdown
    cumulative = np.cumprod(1 + portfolio_returns / 100)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak * 100
    max_dd = float(drawdowns.min())

    # Simple Sharpe approximation
    if portfolio_returns.std() > 0:
        sharpe = portfolio_returns.mean() / portfolio_returns.std()
    else:
        sharpe = 0.0

    return {
        "name": scheme_name,
        "total_return": total_return,
        "robustness": robustness,
        "profitable_windows": profitable,
        "total_windows": n_windows,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "mean_return": float(portfolio_returns.mean()),
        "std_return": float(portfolio_returns.std()),
        "window_returns": portfolio_returns,
        "weight_history": weight_history,
    }


# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 31 — Adaptive Weight Allocation")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Goal: Can dynamic weights improve the 4-comp portfolio?")
    logger.info("  88%% robustness ceiling assumed (W2 unsolvable).")
    logger.info("  Targets: higher OOS return, better Sharpe, lower MaxDD.")
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
    wf_xtf = WalkForwardAnalyzer(n_windows=9)

    # ═════════════════════════════════════════════════════════════
    #   PART 1: Collect Per-Component Per-Window OOS Returns
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: Collecting per-component OOS returns (9 windows)")
    logger.info("-" * 72)
    logger.info("")

    # Run with equal weights to collect individual component returns
    # (weights don't affect individual component results)
    components = [
        CrossTFComponent(
            strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.25, label="1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.25, label="1hDC",
        ),
        CrossTFComponent(
            strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
            engine=engine_15m, weight=0.25, label="15mRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_willr_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.25, label="1hWillR",
        ),
    ]

    report = wf_xtf.run_cross_tf(components)

    # Extract per-component per-window returns
    n_windows = len(report.windows)
    n_comp = 4
    comp_labels = ["1hRSI", "1hDC", "15mRSI", "1hWillR"]

    component_returns = np.zeros((n_windows, n_comp))
    for w_idx, window in enumerate(report.windows):
        for c_idx, cr in enumerate(window.components):
            component_returns[w_idx, c_idx] = cr.oos_return

    logger.info("")
    logger.info("  Per-component OOS returns (%%/window):")
    logger.info("  %8s  %8s  %8s  %8s  %8s", "Window", *comp_labels)
    for w_idx in range(n_windows):
        w = report.windows[w_idx]
        vals = [f"{component_returns[w_idx, c]:+7.2f}" for c in range(n_comp)]
        logger.info("  W%d [%s]:  %s", w_idx + 1, w.test_start, "  ".join(vals))

    logger.info("")
    logger.info("  Component means:  %s",
                "  ".join(f"{component_returns[:, c].mean():+6.2f}" for c in range(n_comp)))
    logger.info("  Component stds:   %s",
                "  ".join(f"{component_returns[:, c].std():6.2f}" for c in range(n_comp)))
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Apply Adaptive Weighting Schemes
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: Adaptive Weighting Schemes")
    logger.info("-" * 72)
    logger.info("")

    schemes = [
        (scheme_equal, "A. Equal (25/25/25/25)"),
        (scheme_fixed_production, "B. Fixed Production (15/50/10/25)"),
        (scheme_momentum, "C. Momentum (K=2, temp=5)"),
        (scheme_momentum_aggressive, "D. Momentum Aggressive (K=3, temp=3)"),
        (scheme_inverse_vol, "E. Inverse Volatility (K=3)"),
        (scheme_best_recent, "F. Best Recent Winner"),
        (scheme_defensive, "G. Defensive (reduce when trailing)"),
        (scheme_defensive_momentum, "H. Defensive + Momentum"),
    ]

    all_results: list[dict] = []

    for scheme_fn, scheme_name in schemes:
        result = simulate_adaptive(component_returns, scheme_fn, scheme_name)
        all_results.append(result)

        logger.info("  %s:", scheme_name)
        logger.info("    OOS Return: %+.2f%% | Robustness: %d%% (%d/%d)",
                     result["total_return"],
                     int(result["robustness"] * 100),
                     result["profitable_windows"],
                     result["total_windows"])
        logger.info("    MaxDD: %.2f%% | Sharpe: %.3f | Mean: %+.2f%% | Std: %.2f%%",
                     result["max_drawdown"], result["sharpe"],
                     result["mean_return"], result["std_return"])

        # Log per-window weights and returns
        for w_idx in range(n_windows):
            weights = result["weight_history"][w_idx]
            port_ret = result["window_returns"][w_idx]
            w_str = "  ".join(f"{weights[c]:.2f}" for c in range(n_comp))
            marker = "+" if port_ret > 0 else "-"
            logger.info("    W%d: w=[%s] -> %+.2f%% %s",
                         w_idx + 1, w_str, port_ret, marker)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Extended Static Weight Grid (top performers only)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: Fine-Grained Static Weight Grid")
    logger.info("-" * 72)
    logger.info("  Testing weights in 5%% steps (min 10%%), looking for")
    logger.info("  static combos that beat 15/50/10/25 on OOS return.")
    logger.info("")

    best_static = {"name": "15/50/10/25", "total_return": -999}
    static_count_88 = 0
    top_statics: list[tuple[str, float, float]] = []  # (name, return, sharpe)

    for w_rsi in range(10, 45, 5):
        for w_dc in range(10, 65, 5):
            for w_rsi15 in range(10, 45, 5):
                w_wr = 100 - w_rsi - w_dc - w_rsi15
                if w_wr < 10 or w_wr > 60:
                    continue

                weights = np.array([w_rsi, w_dc, w_rsi15, w_wr]) / 100.0
                port_returns = component_returns @ weights

                # Compound
                compounded = 1.0
                for r in port_returns:
                    compounded *= (1 + r / 100)
                total_ret = (compounded - 1) * 100

                profitable = int(np.sum(port_returns > 0))
                robustness = profitable / n_windows

                if robustness >= 0.87:  # 88% = 8/9
                    static_count_88 += 1
                    if port_returns.std() > 0:
                        sharpe = port_returns.mean() / port_returns.std()
                    else:
                        sharpe = 0.0
                    top_statics.append((
                        f"{w_rsi}/{w_dc}/{w_rsi15}/{w_wr}",
                        total_ret, sharpe,
                    ))
                    if total_ret > best_static["total_return"]:
                        best_static = {
                            "name": f"{w_rsi}/{w_dc}/{w_rsi15}/{w_wr}",
                            "total_return": total_ret,
                            "sharpe": sharpe,
                            "robustness": robustness,
                        }

    logger.info("  Static combos at 88%% robustness: %d", static_count_88)
    logger.info("")

    # Sort by return, show top 10
    top_statics.sort(key=lambda x: x[1], reverse=True)
    logger.info("  Top 10 static weights (88%% rob):")
    for i, (name, ret, sharpe) in enumerate(top_statics[:10]):
        logger.info("    %2d. %s: %+.2f%% OOS | Sharpe %.3f", i + 1, name, ret, sharpe)
    logger.info("")

    # Sort by Sharpe, show top 10
    top_by_sharpe = sorted(top_statics, key=lambda x: x[2], reverse=True)
    logger.info("  Top 10 by Sharpe (88%% rob):")
    for i, (name, ret, sharpe) in enumerate(top_by_sharpe[:10]):
        logger.info("    %2d. %s: Sharpe %.3f | %+.2f%% OOS", i + 1, name, sharpe, ret)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Volatility-Scaled Position Sizing
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 4: Volatility-Scaled Position Sizing")
    logger.info("-" * 72)
    logger.info("  Scale total portfolio exposure by inverse of recent ATR.")
    logger.info("  High vol → smaller positions, low vol → larger positions.")
    logger.info("")

    # Calculate average ATR per window period
    # Use 1h data as reference for overall market volatility
    common_start = max(df_1h.index[0], df_15m.index[0])
    common_end = min(df_1h.index[-1], df_15m.index[-1])
    total_td = common_end - common_start
    test_td = total_td * 0.3 / 9

    window_atrs = []
    for i in range(9):
        test_end = common_end - (8 - i) * test_td
        test_start = test_end - test_td
        mask = (df_1h.index >= test_start) & (df_1h.index < test_end)
        window_df = df_1h[mask]
        if len(window_df) > 0 and "atr_14" in window_df.columns:
            avg_atr = window_df["atr_14"].mean()
            # Normalize by close price to get ATR %
            avg_close = window_df["close"].mean()
            atr_pct = avg_atr / avg_close * 100 if avg_close > 0 else 2.0
            window_atrs.append(atr_pct)
        else:
            window_atrs.append(2.0)  # default

    window_atrs = np.array(window_atrs)
    logger.info("  Window ATR%% (annualized vol proxy):")
    for w_idx in range(n_windows):
        logger.info("    W%d: ATR = %.3f%%", w_idx + 1, window_atrs[w_idx])
    logger.info("")

    # Volatility scaling: target_vol / actual_vol
    target_vol = np.median(window_atrs)  # target = median ATR
    vol_scalers = target_vol / window_atrs
    vol_scalers = np.clip(vol_scalers, 0.5, 2.0)  # cap at 50%-200%

    logger.info("  Target ATR%%: %.3f%% (median)", target_vol)
    logger.info("  Vol scalers: %s", "  ".join(f"{s:.2f}" for s in vol_scalers))
    logger.info("")

    # Apply vol-scaling to best static weights
    best_w = np.array([0.15, 0.50, 0.10, 0.25])
    vol_scaled_returns = []
    for w_idx in range(n_windows):
        scaled_weights = best_w * vol_scalers[w_idx]
        port_ret = np.dot(scaled_weights, component_returns[w_idx, :])
        vol_scaled_returns.append(port_ret)

    vol_scaled_returns = np.array(vol_scaled_returns)
    compounded = 1.0
    for r in vol_scaled_returns:
        compounded *= (1 + r / 100)
    vol_total = (compounded - 1) * 100
    vol_profitable = int(np.sum(vol_scaled_returns > 0))

    cumulative = np.cumprod(1 + vol_scaled_returns / 100)
    peak = np.maximum.accumulate(cumulative)
    vol_max_dd = float(((cumulative - peak) / peak * 100).min())
    vol_sharpe = (vol_scaled_returns.mean() / vol_scaled_returns.std()
                  if vol_scaled_returns.std() > 0 else 0)

    logger.info("  Vol-Scaled (15/50/10/25):")
    logger.info("    OOS Return: %+.2f%% | Robustness: %d%% (%d/%d)",
                vol_total, int(vol_profitable / n_windows * 100),
                vol_profitable, n_windows)
    logger.info("    MaxDD: %.2f%% | Sharpe: %.3f",
                vol_max_dd, vol_sharpe)
    for w_idx in range(n_windows):
        marker = "+" if vol_scaled_returns[w_idx] > 0 else "-"
        logger.info("    W%d: scaler=%.2f -> %+.2f%% %s",
                     w_idx + 1, vol_scalers[w_idx],
                     vol_scaled_returns[w_idx], marker)
    logger.info("")

    # Also test vol-scaling with equal weights
    equal_w = np.ones(n_comp) / n_comp
    vol_eq_returns = []
    for w_idx in range(n_windows):
        scaled_weights = equal_w * vol_scalers[w_idx]
        port_ret = np.dot(scaled_weights, component_returns[w_idx, :])
        vol_eq_returns.append(port_ret)

    vol_eq_returns = np.array(vol_eq_returns)
    compounded = 1.0
    for r in vol_eq_returns:
        compounded *= (1 + r / 100)
    vol_eq_total = (compounded - 1) * 100
    vol_eq_profitable = int(np.sum(vol_eq_returns > 0))
    vol_eq_sharpe = (vol_eq_returns.mean() / vol_eq_returns.std()
                     if vol_eq_returns.std() > 0 else 0)

    logger.info("  Vol-Scaled (Equal):")
    logger.info("    OOS Return: %+.2f%% | Robustness: %d%% (%d/%d) | Sharpe: %.3f",
                vol_eq_total,
                int(vol_eq_profitable / n_windows * 100),
                vol_eq_profitable, n_windows, vol_eq_sharpe)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 5: Summary & Comparison
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 31 SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    logger.info("  %-35s  %8s  %5s  %7s  %7s",
                "Scheme", "OOS Ret", "Rob%", "MaxDD", "Sharpe")
    logger.info("  " + "-" * 68)

    for result in all_results:
        logger.info("  %-35s  %+7.2f%%  %4d%%  %6.2f%%  %6.3f",
                     result["name"],
                     result["total_return"],
                     int(result["robustness"] * 100),
                     result["max_drawdown"],
                     result["sharpe"])

    # Add vol-scaled results
    logger.info("  %-35s  %+7.2f%%  %4d%%  %6.2f%%  %6.3f",
                "I. Vol-Scaled (15/50/10/25)",
                vol_total,
                int(vol_profitable / n_windows * 100),
                vol_max_dd, vol_sharpe)
    logger.info("  %-35s  %+7.2f%%  %4d%%  %6.2f%%  %6.3f",
                "J. Vol-Scaled (Equal)",
                vol_eq_total,
                int(vol_eq_profitable / n_windows * 100),
                float(((np.cumprod(1 + vol_eq_returns / 100) -
                         np.maximum.accumulate(np.cumprod(1 + vol_eq_returns / 100))) /
                        np.maximum.accumulate(np.cumprod(1 + vol_eq_returns / 100)) * 100).min()),
                vol_eq_sharpe)

    logger.info("")
    logger.info("  Best static weights: %s (%+.2f%% OOS, Sharpe %.3f)",
                best_static["name"], best_static["total_return"],
                best_static.get("sharpe", 0))
    logger.info("")

    # Find overall best scheme
    all_comparable = all_results + [
        {"name": "I. Vol-Scaled (15/50/10/25)",
         "total_return": vol_total, "robustness": vol_profitable / n_windows,
         "sharpe": vol_sharpe},
    ]

    best_by_return = max(all_comparable, key=lambda x: x["total_return"])
    best_by_sharpe = max(all_comparable, key=lambda x: x["sharpe"])

    logger.info("  Best by OOS return: %s (%+.2f%%)",
                best_by_return["name"], best_by_return["total_return"])
    logger.info("  Best by Sharpe:     %s (%.3f)",
                best_by_sharpe["name"], best_by_sharpe["sharpe"])
    logger.info("")

    # Key insight: does any adaptive scheme beat production fixed weights?
    production = next(r for r in all_results if "Production" in r["name"])
    better_schemes = [
        r for r in all_comparable
        if r["total_return"] > production["total_return"]
        and r["robustness"] >= production["robustness"]
    ]

    if better_schemes:
        logger.info("  CONCLUSION: %d scheme(s) beat production 15/50/10/25:",
                     len(better_schemes))
        for s in better_schemes:
            logger.info("    %s: %+.2f%% vs %+.2f%% (%+.2f%% improvement)",
                         s["name"], s["total_return"],
                         production["total_return"],
                         s["total_return"] - production["total_return"])
    else:
        logger.info("  CONCLUSION: No adaptive scheme beats production (15/50/10/25)")
        logger.info("  Fixed weights remain optimal. Adaptive allocation adds")
        logger.info("  complexity without benefit — likely due to only 9 windows")
        logger.info("  being insufficient for reliable performance estimation.")

    logger.info("")

    # W2 analysis
    w2_returns = component_returns[1, :]  # W2 is index 1
    logger.info("  W2 Analysis (the unsolvable window):")
    logger.info("    Component returns: %s",
                "  ".join(f"{comp_labels[c]}={w2_returns[c]:+.2f}%" for c in range(n_comp)))
    logger.info("    All negative: %s", "YES" if np.all(w2_returns < 0) else "NO")
    if np.all(w2_returns < 0):
        # Best possible W2 outcome = least-negative component
        best_w2 = w2_returns.max()
        logger.info("    Best achievable W2: %+.2f%% (100%% in %s)",
                     best_w2, comp_labels[np.argmax(w2_returns)])
        logger.info("    → W2 loss is STRUCTURAL. No weight allocation can make W2 positive.")
    logger.info("")

    logger.info("  Phase 31 complete.")


if __name__ == "__main__":
    main()

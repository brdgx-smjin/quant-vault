#!/usr/bin/env python3
"""Phase 21 — Cross-TF Portfolio Risk Analytics.

Comprehensive risk analysis of the validated cross-TF portfolio
(1hRSI/1hDC/15mRSI 33/33/34, 88% robustness).

Analyses:
  1. Full-period backtest with detailed trade log collection
  2. Component return correlation (cross-window)
  3. Risk metrics: MaxDD, Calmar, Sortino, profit factor by component
  4. Trade anatomy: exit reasons, duration, LONG vs SHORT breakdown
  5. Rolling 3-window performance stability
  6. Consecutive loss / drawdown analysis
  7. Return distribution (skewness, kurtosis)

This is a pure analysis phase — no new strategies or parameter changes.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.basic import BasicIndicators
from src.backtest.engine import BacktestEngine, BacktestResult

from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase21")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase21.log", mode="w")
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
    "src.backtest.walk_forward",
]:
    logging.getLogger(name).setLevel(logging.WARNING)


# ─── Data ─────────────────────────────────────────────────────────

def load_data(timeframe: str) -> pd.DataFrame:
    """Load and add indicators to OHLCV data."""
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


# ─── Strategy Factories ──────────────────────────────────────────

def make_rsi_1h() -> MultiTimeframeFilter:
    return MultiTimeframeFilter(RSIMeanReversionStrategy(
        rsi_oversold=35.0, rsi_overbought=65.0,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    ))


def make_dc_1h() -> MultiTimeframeFilter:
    return MultiTimeframeFilter(DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
    ))


def make_rsi_15m() -> MultiTimeframeFilter:
    return MultiTimeframeFilter(RSIMeanReversionStrategy(
        rsi_oversold=35.0, rsi_overbought=65.0,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
    ))


# ─── Window Computation (matches run_cross_tf) ──────────────────

@dataclass
class WindowBounds:
    window_id: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    is_last: bool


def compute_windows(
    dfs: list[pd.DataFrame],
    n_windows: int = 9,
    train_ratio: float = 0.7,
) -> list[WindowBounds]:
    """Compute date-aligned window boundaries."""
    start_date = max(df.index[0] for df in dfs)
    end_date = min(df.index[-1] for df in dfs)
    total_td = end_date - start_date
    test_td = total_td * (1 - train_ratio) / n_windows

    windows = []
    for i in range(n_windows):
        test_end = end_date - (n_windows - 1 - i) * test_td
        test_start = test_end - test_td
        windows.append(WindowBounds(
            window_id=i + 1,
            test_start=test_start,
            test_end=test_end,
            is_last=(i == n_windows - 1),
        ))
    return windows


# ─── Component Runner (returns BacktestResult + trade logs) ──────

@dataclass
class ComponentResult:
    """Full results for one component across all windows."""
    label: str
    per_window_returns: list[float]
    per_window_trades: list[int]
    per_window_results: list[BacktestResult]
    all_trade_logs: list  # Flattened trade logs across all windows


def run_component_full(
    label: str,
    factory,
    df: pd.DataFrame,
    htf_df: pd.DataFrame,
    engine: BacktestEngine,
    windows: list[WindowBounds],
) -> ComponentResult:
    """Run component through all windows, collecting full results."""
    returns = []
    trades = []
    results = []
    all_logs = []

    for w in windows:
        if w.is_last:
            test_df = df[df.index >= w.test_start]
        else:
            test_df = df[(df.index >= w.test_start) & (df.index < w.test_end)]

        if len(test_df) < 30:
            returns.append(0.0)
            trades.append(0)
            results.append(None)
            continue

        strategy = factory()
        result = engine.run(strategy, test_df, htf_df=htf_df)
        returns.append(result.total_return)
        trades.append(result.total_trades)
        results.append(result)

        for tl in result.trade_logs:
            tl.metadata["window"] = w.window_id
            tl.metadata["component"] = label
            all_logs.append(tl)

    return ComponentResult(
        label=label,
        per_window_returns=returns,
        per_window_trades=trades,
        per_window_results=results,
        all_trade_logs=all_logs,
    )


# ─── Analysis Functions ──────────────────────────────────────────

def analyze_trade_anatomy(trade_logs: list, label: str) -> dict:
    """Analyze trade exit reasons, duration, and direction breakdown."""
    if not trade_logs:
        return {}

    longs = [t for t in trade_logs if t.side == "long"]
    shorts = [t for t in trade_logs if t.side == "short"]
    sl_trades = [t for t in trade_logs if t.exit_reason == "sl"]
    tp_trades = [t for t in trade_logs if t.exit_reason == "tp"]
    timeout_trades = [t for t in trade_logs if t.exit_reason == "timeout"]
    wins = [t for t in trade_logs if t.return_pct > 0]
    losses = [t for t in trade_logs if t.return_pct < 0]

    durations = [t.bars_held for t in trade_logs]
    returns = [t.return_pct for t in trade_logs]

    long_wins = [t for t in longs if t.return_pct > 0]
    short_wins = [t for t in shorts if t.return_pct > 0]

    return {
        "label": label,
        "total": len(trade_logs),
        "longs": len(longs),
        "shorts": len(shorts),
        "long_wr": len(long_wins) / len(longs) * 100 if longs else 0,
        "short_wr": len(short_wins) / len(shorts) * 100 if shorts else 0,
        "sl_exits": len(sl_trades),
        "tp_exits": len(tp_trades),
        "timeout_exits": len(timeout_trades),
        "avg_duration": np.mean(durations) if durations else 0,
        "med_duration": np.median(durations) if durations else 0,
        "max_duration": max(durations) if durations else 0,
        "avg_win": np.mean([t.return_pct for t in wins]) if wins else 0,
        "avg_loss": np.mean([t.return_pct for t in losses]) if losses else 0,
        "max_win": max([t.return_pct for t in wins]) if wins else 0,
        "max_loss": min([t.return_pct for t in losses]) if losses else 0,
        "returns": returns,
    }


def analyze_consecutive_losses(trade_logs: list) -> tuple[int, float]:
    """Find max consecutive losses and worst consecutive loss streak PnL.

    Returns:
        (max_consec_losses, worst_streak_pnl_pct)
    """
    if not trade_logs:
        return 0, 0.0

    max_streak = 0
    current_streak = 0
    worst_pnl = 0.0
    current_pnl = 0.0

    for t in trade_logs:
        if t.return_pct < 0:
            current_streak += 1
            current_pnl += t.return_pct
            if current_streak > max_streak:
                max_streak = current_streak
            if current_pnl < worst_pnl:
                worst_pnl = current_pnl
        else:
            current_streak = 0
            current_pnl = 0.0

    return max_streak, worst_pnl


def compute_sortino(returns: list[float]) -> float:
    """Compute Sortino ratio from a list of per-trade returns."""
    if not returns or len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    mean_ret = np.mean(arr)
    downside = arr[arr < 0]
    if len(downside) == 0:
        return float("inf") if mean_ret > 0 else 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(mean_ret / downside_std)


def compute_return_distribution(returns: list[float]) -> dict:
    """Compute distribution statistics for returns."""
    if len(returns) < 4:
        return {"skew": 0.0, "kurtosis": 0.0, "std": 0.0}
    arr = np.array(returns)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return {"skew": 0.0, "kurtosis": 0.0, "std": 0.0}
    skew = float(np.mean(((arr - mean) / std) ** 3))
    kurt = float(np.mean(((arr - mean) / std) ** 4) - 3)  # Excess kurtosis
    return {"skew": skew, "kurtosis": kurt, "std": float(std)}


# ─── Main ─────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 21 — Cross-TF Portfolio Risk Analytics")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Goal: Comprehensive risk profile of the 88%% cross-TF portfolio.")
    logger.info("  Portfolio: 1hRSI(33%%) + 1hDC(33%%) + 15mRSI(34%%)")
    logger.info("")

    # Load data
    logger.info("  Loading data...")
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")
    logger.info("  1h: %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("  15m: %d bars (%s ~ %s)", len(df_15m),
                df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("")

    # Engines
    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")

    # Windows
    n_windows = 9
    windows = compute_windows([df_1h, df_15m], n_windows=n_windows)
    logger.info("  Date-aligned OOS windows:")
    for w in windows:
        logger.info("    W%d: %s ~ %s%s", w.window_id,
                    w.test_start.date(), w.test_end.date(),
                    " (last)" if w.is_last else "")
    logger.info("")

    weights = {"1hRSI": 0.33, "1hDC": 0.33, "15mRSI": 0.34}

    # ══════════════════════════════════════════════════════════════
    #  STEP 1: Run all components with full trade logging
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  STEP 1: Component Backtests (full trade logs)")
    logger.info("=" * 72)
    logger.info("")

    comp_1h_rsi = run_component_full(
        "1hRSI", make_rsi_1h, df_1h, df_4h, engine_1h, windows,
    )
    logger.info("  1hRSI: %d trades across 9 windows", len(comp_1h_rsi.all_trade_logs))

    comp_1h_dc = run_component_full(
        "1hDC", make_dc_1h, df_1h, df_4h, engine_1h, windows,
    )
    logger.info("  1hDC: %d trades across 9 windows", len(comp_1h_dc.all_trade_logs))

    comp_15m_rsi = run_component_full(
        "15mRSI", make_rsi_15m, df_15m, df_4h, engine_15m, windows,
    )
    logger.info("  15mRSI: %d trades across 9 windows", len(comp_15m_rsi.all_trade_logs))

    components = [comp_1h_rsi, comp_1h_dc, comp_15m_rsi]

    # Portfolio weighted returns
    portfolio_returns = []
    for i in range(n_windows):
        wr = sum(
            comp.per_window_returns[i] * weights[comp.label]
            for comp in components
        )
        portfolio_returns.append(wr)

    logger.info("")
    logger.info("  Per-window returns:")
    logger.info("  %-4s  %8s  %8s  %8s  %10s", "Win", "1hRSI", "1hDC", "15mRSI", "Portfolio")
    logger.info("  " + "-" * 44)
    for i, w in enumerate(windows):
        marker = "+" if portfolio_returns[i] > 0 else "-"
        logger.info("  W%-3d  %+7.2f%%  %+7.2f%%  %+7.2f%%  %+9.2f%% %s",
                    w.window_id,
                    comp_1h_rsi.per_window_returns[i],
                    comp_1h_dc.per_window_returns[i],
                    comp_15m_rsi.per_window_returns[i],
                    portfolio_returns[i], marker)

    compounded = 1.0
    for r in portfolio_returns:
        compounded *= (1 + r / 100)
    total_oos = (compounded - 1) * 100
    profitable = sum(1 for r in portfolio_returns if r > 0)

    logger.info("  " + "-" * 44)
    logger.info("  OOS Total: %+.2f%% | Robustness: %d%% (%d/%d)",
                total_oos, int(profitable / n_windows * 100), profitable, n_windows)
    logger.info("")

    # ══════════════════════════════════════════════════════════════
    #  STEP 2: Component Return Correlation
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  STEP 2: Component Return Correlation (across 9 OOS windows)")
    logger.info("=" * 72)
    logger.info("")

    corr_df = pd.DataFrame({
        "1hRSI": comp_1h_rsi.per_window_returns,
        "1hDC": comp_1h_dc.per_window_returns,
        "15mRSI": comp_15m_rsi.per_window_returns,
    })
    corr_matrix = corr_df.corr()

    logger.info("  Correlation matrix:")
    logger.info("  %-8s  %8s  %8s  %8s", "", "1hRSI", "1hDC", "15mRSI")
    for label in ["1hRSI", "1hDC", "15mRSI"]:
        logger.info("  %-8s  %8.3f  %8.3f  %8.3f",
                    label,
                    corr_matrix.loc[label, "1hRSI"],
                    corr_matrix.loc[label, "1hDC"],
                    corr_matrix.loc[label, "15mRSI"])

    avg_corr = (
        corr_matrix.loc["1hRSI", "1hDC"]
        + corr_matrix.loc["1hRSI", "15mRSI"]
        + corr_matrix.loc["1hDC", "15mRSI"]
    ) / 3
    logger.info("")
    logger.info("  Average pairwise correlation: %.3f", avg_corr)

    # Check which windows each component is negative
    logger.info("")
    logger.info("  Negative windows per component:")
    for comp in components:
        neg_wins = [
            f"W{i+1}" for i, r in enumerate(comp.per_window_returns) if r < 0
        ]
        logger.info("    %s: %s", comp.label, ", ".join(neg_wins) if neg_wins else "None")

    # Diversification benefit
    # Compare: avg of individual robustness vs portfolio robustness
    individual_robs = []
    for comp in components:
        prof = sum(1 for r in comp.per_window_returns if r > 0)
        individual_robs.append(prof / n_windows * 100)
    avg_ind_rob = np.mean(individual_robs)
    portfolio_rob = profitable / n_windows * 100

    logger.info("")
    logger.info("  Diversification benefit:")
    for comp, rob in zip(components, individual_robs):
        logger.info("    %s standalone robustness: %d%%", comp.label, int(rob))
    logger.info("    Average individual robustness: %.0f%%", avg_ind_rob)
    logger.info("    Portfolio robustness: %d%%", int(portfolio_rob))
    logger.info("    → Diversification uplift: %+.0f%%", portfolio_rob - avg_ind_rob)
    logger.info("")

    # ══════════════════════════════════════════════════════════════
    #  STEP 3: Trade Anatomy
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  STEP 3: Trade Anatomy")
    logger.info("=" * 72)
    logger.info("")

    all_portfolio_trades = []
    for comp in components:
        all_portfolio_trades.extend(comp.all_trade_logs)

    for comp in components:
        anatomy = analyze_trade_anatomy(comp.all_trade_logs, comp.label)
        if not anatomy:
            continue

        logger.info("  ─── %s (%d trades) ───", anatomy["label"], anatomy["total"])
        logger.info("    Direction:  %d LONG / %d SHORT", anatomy["longs"], anatomy["shorts"])
        logger.info("    Win Rate:   LONG %.1f%% | SHORT %.1f%%", anatomy["long_wr"], anatomy["short_wr"])
        logger.info("    Exits:      SL=%d  TP=%d  Timeout=%d",
                    anatomy["sl_exits"], anatomy["tp_exits"], anatomy["timeout_exits"])
        logger.info("    Duration:   avg=%.1f bars, median=%.0f, max=%d",
                    anatomy["avg_duration"], anatomy["med_duration"], anatomy["max_duration"])
        logger.info("    Returns:    avg_win=%+.2f%%  avg_loss=%+.2f%%",
                    anatomy["avg_win"], anatomy["avg_loss"])
        logger.info("    Extremes:   best=%+.2f%%  worst=%+.2f%%",
                    anatomy["max_win"], anatomy["max_loss"])

        max_consec, worst_streak = analyze_consecutive_losses(comp.all_trade_logs)
        logger.info("    Consec Loss: max=%d trades, worst streak PnL=%+.2f%%",
                    max_consec, worst_streak)
        logger.info("")

    # Portfolio-level trade anatomy
    logger.info("  ─── PORTFOLIO AGGREGATE (%d trades) ───", len(all_portfolio_trades))
    port_anatomy = analyze_trade_anatomy(all_portfolio_trades, "Portfolio")
    if port_anatomy:
        logger.info("    Direction:  %d LONG / %d SHORT", port_anatomy["longs"], port_anatomy["shorts"])
        logger.info("    Win Rate:   LONG %.1f%% | SHORT %.1f%%",
                    port_anatomy["long_wr"], port_anatomy["short_wr"])
        logger.info("    Exits:      SL=%d  TP=%d  Timeout=%d",
                    port_anatomy["sl_exits"], port_anatomy["tp_exits"], port_anatomy["timeout_exits"])
        logger.info("    Duration:   avg=%.1f bars, median=%.0f",
                    port_anatomy["avg_duration"], port_anatomy["med_duration"])
        logger.info("    Returns:    avg_win=%+.2f%%  avg_loss=%+.2f%%",
                    port_anatomy["avg_win"], port_anatomy["avg_loss"])
    logger.info("")

    # ══════════════════════════════════════════════════════════════
    #  STEP 4: Risk Metrics
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  STEP 4: Risk Metrics")
    logger.info("=" * 72)
    logger.info("")

    # Per-component risk metrics from BacktestResult
    logger.info("  %-8s  %8s  %8s  %8s  %8s  %8s  %8s",
                "Comp", "Return", "Sharpe", "MaxDD", "WinRate", "PF", "Sortino")
    logger.info("  " + "-" * 60)

    for comp in components:
        # Aggregate across windows
        total_ret_list = comp.per_window_returns
        sharpes = [r.sharpe_ratio for r in comp.per_window_results if r and np.isfinite(r.sharpe_ratio)]
        max_dds = [r.max_drawdown for r in comp.per_window_results if r]
        win_rates = [r.win_rate for r in comp.per_window_results if r and r.total_trades > 0]
        pfs = [r.profit_factor for r in comp.per_window_results
               if r and np.isfinite(r.profit_factor) and r.total_trades > 0]

        comp_ret = 1.0
        for r in total_ret_list:
            comp_ret *= (1 + r / 100)
        comp_ret = (comp_ret - 1) * 100

        trade_returns = [t.return_pct for t in comp.all_trade_logs]
        sortino = compute_sortino(trade_returns)

        logger.info("  %-8s  %+7.2f%%  %8.2f  %7.1f%%  %7.1f%%  %8.2f  %8.2f",
                    comp.label, comp_ret,
                    np.mean(sharpes) if sharpes else 0,
                    np.mean(max_dds) if max_dds else 0,
                    np.mean(win_rates) * 100 if win_rates else 0,
                    np.mean(pfs) if pfs else 0,
                    sortino)

    # Portfolio-level Sortino
    port_trade_returns = [t.return_pct for t in all_portfolio_trades]
    port_sortino = compute_sortino(port_trade_returns)
    port_max_dd_per_window = []
    for comp in components:
        for r in comp.per_window_results:
            if r:
                port_max_dd_per_window.append(r.max_drawdown)

    logger.info("  " + "-" * 60)
    logger.info("  %-8s  %+7.2f%%  %8s  %7.1f%%  %7.1f%%  %8s  %8.2f",
                "PORTF", total_oos, "---",
                np.mean(port_max_dd_per_window) if port_max_dd_per_window else 0,
                len([t for t in all_portfolio_trades if t.return_pct > 0]) / len(all_portfolio_trades) * 100 if all_portfolio_trades else 0,
                "---", port_sortino)

    # Calmar ratio (annualized return / max drawdown)
    # Approximate: OOS period is ~30% of 365 days ≈ 110 days
    oos_days = sum((w.test_end - w.test_start).days for w in windows)
    annualized_ret = total_oos * (365 / oos_days) if oos_days > 0 else 0
    worst_window_dd = max(port_max_dd_per_window) if port_max_dd_per_window else 1
    calmar = annualized_ret / worst_window_dd if worst_window_dd > 0 else 0

    logger.info("")
    logger.info("  Portfolio Calmar Ratio: %.2f (annualized ret %.1f%% / worst DD %.1f%%)",
                calmar, annualized_ret, worst_window_dd)
    logger.info("  OOS period: %d days", oos_days)
    logger.info("")

    # ══════════════════════════════════════════════════════════════
    #  STEP 5: Return Distribution
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  STEP 5: Return Distribution")
    logger.info("=" * 72)
    logger.info("")

    for comp in components:
        trade_rets = [t.return_pct for t in comp.all_trade_logs]
        dist = compute_return_distribution(trade_rets)
        logger.info("  %s: std=%.2f%%, skew=%.2f, kurtosis=%.2f  (n=%d trades)",
                    comp.label, dist["std"], dist["skew"], dist["kurtosis"],
                    len(trade_rets))

    port_dist = compute_return_distribution(port_trade_returns)
    logger.info("  PORTF: std=%.2f%%, skew=%.2f, kurtosis=%.2f  (n=%d trades)",
                port_dist["std"], port_dist["skew"], port_dist["kurtosis"],
                len(port_trade_returns))

    logger.info("")
    logger.info("  Interpretation:")
    if port_dist["skew"] > 0:
        logger.info("    Positive skew (%.2f) → right tail (large wins) fatter than left", port_dist["skew"])
    else:
        logger.info("    Negative skew (%.2f) → left tail (large losses) fatter than right", port_dist["skew"])
    if port_dist["kurtosis"] > 0:
        logger.info("    Positive excess kurtosis (%.2f) → heavy tails, more extreme outcomes", port_dist["kurtosis"])
    else:
        logger.info("    Negative excess kurtosis (%.2f) → thin tails, fewer extreme outcomes", port_dist["kurtosis"])
    logger.info("")

    # Return quantiles
    if port_trade_returns:
        arr = np.array(port_trade_returns)
        logger.info("  Return quantiles:")
        for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            logger.info("    P%02d: %+.2f%%", q, np.percentile(arr, q))
        logger.info("")

    # ══════════════════════════════════════════════════════════════
    #  STEP 6: Rolling Performance Stability
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  STEP 6: Rolling 3-Window Performance")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Checks if portfolio performance degrades over time.")
    logger.info("")

    for start_i in range(n_windows - 2):
        end_i = start_i + 3
        window_slice = portfolio_returns[start_i:end_i]
        roll_comp = 1.0
        for r in window_slice:
            roll_comp *= (1 + r / 100)
        roll_ret = (roll_comp - 1) * 100
        roll_prof = sum(1 for r in window_slice if r > 0)
        roll_rob = roll_prof / 3 * 100

        logger.info("  W%d-W%d: %+.2f%% return, %d/3 profitable (%.0f%%)",
                    start_i + 1, end_i, roll_ret, roll_prof, roll_rob)

    # Trend test: is the last 3-window block worse than first?
    first_3 = portfolio_returns[:3]
    last_3 = portfolio_returns[-3:]
    first_3_ret = sum(first_3)
    last_3_ret = sum(last_3)

    logger.info("")
    logger.info("  First 3 windows avg: %+.2f%% per window", np.mean(first_3))
    logger.info("  Last 3 windows avg:  %+.2f%% per window", np.mean(last_3))
    if np.mean(last_3) >= np.mean(first_3) * 0.5:
        logger.info("  → No significant degradation detected")
    else:
        logger.info("  ⚠ Performance may be degrading over time")
    logger.info("")

    # ══════════════════════════════════════════════════════════════
    #  STEP 7: Worst-Case Scenario Analysis
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  STEP 7: Worst-Case Scenario Analysis")
    logger.info("=" * 72)
    logger.info("")

    # W2 deep dive (the only negative window)
    w2_idx = 1  # 0-indexed
    logger.info("  ─── W2 Deep Dive (only negative window) ───")
    logger.info("  Period: %s ~ %s", windows[w2_idx].test_start.date(),
                windows[w2_idx].test_end.date())
    logger.info("")

    for comp in components:
        w2_trades = [t for t in comp.all_trade_logs if t.metadata.get("window") == 2]
        logger.info("  %s in W2: %d trades, return %+.2f%%",
                    comp.label, len(w2_trades), comp.per_window_returns[w2_idx])
        for t in w2_trades:
            logger.info("    %s %s: entry=%.2f exit=%.2f %+.2f%% (%s, %d bars)",
                        t.side.upper(), t.entry_time[:10],
                        t.entry_price, t.exit_price,
                        t.return_pct, t.exit_reason, t.bars_held)

    logger.info("")
    logger.info("  Portfolio W2 return: %+.2f%%", portfolio_returns[w2_idx])
    logger.info("")

    # Max portfolio drawdown sequence
    logger.info("  ─── Portfolio Equity Curve (OOS windows) ───")
    equity = 100.0  # Start at 100
    peak = equity
    max_dd = 0.0
    max_dd_window = 0

    for i, r in enumerate(portfolio_returns):
        equity *= (1 + r / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
            max_dd_window = i + 1
        logger.info("    W%d: equity=%.2f  peak=%.2f  DD=%.2f%%",
                    i + 1, equity, peak, dd)

    logger.info("")
    logger.info("  Portfolio max drawdown (cumulative): %.2f%% at W%d", max_dd, max_dd_window)
    logger.info("  Final equity: %.2f (started at 100)", equity)
    logger.info("")

    # ══════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 21 — Risk Profile Summary")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Portfolio: 1hRSI(33%%) + 1hDC(33%%) + 15mRSI(34%%)")
    logger.info("  OOS Return: %+.2f%% | Robustness: %d%% (%d/%d)",
                total_oos, int(profitable / n_windows * 100), profitable, n_windows)
    logger.info("  Annualized Return: %+.1f%%", annualized_ret)
    logger.info("  Calmar Ratio: %.2f", calmar)
    logger.info("  Portfolio Sortino: %.2f", port_sortino)
    logger.info("  Avg Pairwise Correlation: %.3f", avg_corr)
    logger.info("  Diversification Uplift: %+.0f%% robustness", portfolio_rob - avg_ind_rob)
    logger.info("  Total OOS Trades: %d", len(all_portfolio_trades))
    logger.info("  Cumulative MaxDD: %.2f%%", max_dd)
    logger.info("  Worst Window: W2 (%+.2f%%)", portfolio_returns[w2_idx])

    if port_trade_returns:
        overall_wr = len([t for t in all_portfolio_trades if t.return_pct > 0]) / len(all_portfolio_trades) * 100
        logger.info("  Overall Win Rate: %.1f%%", overall_wr)
    logger.info("")

    elapsed = time.time() - t0
    logger.info("=" * 72)
    logger.info("  PHASE 21 — COMPLETE (%.1f seconds)", elapsed)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

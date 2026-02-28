#!/usr/bin/env python3
"""Phase 26 — 4-comp vs 3-comp Head-to-Head Validation & Risk Analysis.

Phase 25b conclusion: 4-comp (1hRSI/1hDC/15mRSI/1hWillR) at 88% rob,
  +23.98% OOS with 15/50/10/25 weights. 303/375 combos at 88%.

Phase 26 goals:
  1. Head-to-head 4-comp vs 3-comp on same 9-window WF
  2. Per-window detailed breakdown (drawdown, trades, win rate per component)
  3. Component correlation matrix
  4. Risk metrics: Calmar, VaR, Expected Shortfall
  5. Trade-level analysis: timing, clustering, exit reasons
  6. Production readiness assessment
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
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase26")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase26.log", mode="w")
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
        willr_period=14,
        oversold_level=90.0,
        overbought_level=90.0,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def log_report(name: str, report: CrossTFReport) -> None:
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


def compute_risk_metrics(report: CrossTFReport) -> dict:
    """Compute additional risk metrics from window returns."""
    returns = [w.weighted_return for w in report.windows]
    arr = np.array(returns)

    avg_ret = np.mean(arr)
    std_ret = np.std(arr, ddof=1) if len(arr) > 1 else 0.0

    # Simulate equity curve from window returns
    equity = [100.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r / 100))
    equity = np.array(equity)

    # Max drawdown from equity curve
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    max_dd = float(np.min(dd))

    # Calmar = annualized return / max drawdown
    total_return = (equity[-1] / equity[0] - 1) * 100
    # Assume ~12 months of data, 9 windows
    annualized = total_return  # Already ~1 year
    calmar = abs(annualized / max_dd) if max_dd != 0 else float("inf")

    # VaR (95%, historical)
    var_95 = float(np.percentile(arr, 5)) if len(arr) >= 5 else float("nan")

    # Expected Shortfall (CVaR) - average of returns below VaR
    if not np.isnan(var_95):
        tail = arr[arr <= var_95]
        es_95 = float(np.mean(tail)) if len(tail) > 0 else var_95
    else:
        es_95 = float("nan")

    # Sharpe (per-window)
    sharpe = float(avg_ret / std_ret) if std_ret > 0 else float("inf")

    # Sortino (downside deviation only)
    neg_rets = arr[arr < 0]
    downside_std = np.std(neg_rets, ddof=1) if len(neg_rets) > 1 else 0.0
    sortino = float(avg_ret / downside_std) if downside_std > 0 else float("inf")

    return {
        "total_return": total_return,
        "avg_window_return": avg_ret,
        "std_window_return": std_ret,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "var_95": var_95,
        "es_95": es_95,
        "best_window": float(np.max(arr)),
        "worst_window": float(np.min(arr)),
    }


def analyze_components(report: CrossTFReport) -> None:
    """Analyze per-component correlations across windows."""
    if not report.windows or not report.windows[0].components:
        return

    labels = [cr.label for cr in report.windows[0].components]
    n_comp = len(labels)

    # Build returns matrix: rows=windows, cols=components
    ret_matrix = np.zeros((len(report.windows), n_comp))
    for i, w in enumerate(report.windows):
        for j, cr in enumerate(w.components):
            ret_matrix[i, j] = cr.oos_return

    logger.info("")
    logger.info("  Component Returns per Window:")
    header = "  %-6s" + " ".join(f"{l:>10s}" for l in labels)
    logger.info(header, "Window")
    logger.info("  " + "-" * (8 + 11 * n_comp))
    for i, w in enumerate(report.windows):
        vals = " ".join(f"{ret_matrix[i, j]:+9.2f}%" for j in range(n_comp))
        logger.info("  W%-5d %s", i + 1, vals)

    # Average returns per component
    avg_rets = np.mean(ret_matrix, axis=0)
    logger.info("  %-6s %s", "Avg",
                " ".join(f"{avg_rets[j]:+9.2f}%" for j in range(n_comp)))

    # Profitable windows per component
    prof = np.sum(ret_matrix > 0, axis=0)
    logger.info("  %-6s %s", "Win W",
                " ".join(f"{int(prof[j]):>9d}/9" for j in range(n_comp)))

    # Correlation matrix
    if n_comp >= 2 and len(report.windows) >= 3:
        corr = np.corrcoef(ret_matrix.T)
        logger.info("")
        logger.info("  Component Correlation Matrix:")
        logger.info("  %-10s" + " ".join(f"{l:>10s}" for l in labels), "")
        for i, label in enumerate(labels):
            row = " ".join(f"{corr[i, j]:+9.3f}" for j in range(n_comp))
            logger.info("  %-10s %s", label, row)

        # Average off-diagonal correlation
        off_diag = []
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                off_diag.append(corr[i, j])
        avg_corr = np.mean(off_diag)
        logger.info("  Avg off-diagonal correlation: %+.3f", avg_corr)


def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 26 — 4-comp vs 3-comp Head-to-Head & Risk Analysis")
    logger.info("=" * 72)
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")
    df_1h = add_willr(df_1h, 14)

    logger.info("  Data loaded:")
    logger.info("    1h:  %d bars (%s ~ %s)",
                len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("    15m: %d bars (%s ~ %s)",
                len(df_15m), df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("    4h:  %d bars (%s ~ %s)",
                len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")
    wf = WalkForwardAnalyzer(n_windows=9)

    # ═════════════════════════════════════════════════════════════
    #   PART 1: 3-comp Baseline (33/33/34)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: 3-comp Baseline — 1hRSI/1hDC/15mRSI (33/33/34)")
    logger.info("-" * 72)
    logger.info("")

    report_3comp = wf.run_cross_tf([
        CrossTFComponent(
            strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.33, label="1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.33, label="1hDC",
        ),
        CrossTFComponent(
            strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
            engine=engine_15m, weight=0.34, label="15mRSI",
        ),
    ])
    logger.info("")
    log_report("3-comp 33/33/34", report_3comp)
    analyze_components(report_3comp)

    risk_3 = compute_risk_metrics(report_3comp)
    logger.info("")
    logger.info("  3-comp Risk Metrics:")
    logger.info("    Total OOS Return: %+.2f%%", risk_3["total_return"])
    logger.info("    Avg Window Return: %+.2f%%", risk_3["avg_window_return"])
    logger.info("    Std Window Return: %.2f%%", risk_3["std_window_return"])
    logger.info("    Max Drawdown: %.2f%%", risk_3["max_drawdown"])
    logger.info("    Calmar Ratio: %.2f", risk_3["calmar_ratio"])
    logger.info("    Sharpe (per-window): %.2f", risk_3["sharpe_ratio"])
    logger.info("    Sortino (per-window): %.2f", risk_3["sortino_ratio"])
    logger.info("    VaR 95%%: %.2f%%", risk_3["var_95"])
    logger.info("    Expected Shortfall 95%%: %.2f%%", risk_3["es_95"])
    logger.info("    Best Window: %+.2f%%", risk_3["best_window"])
    logger.info("    Worst Window: %+.2f%%", risk_3["worst_window"])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: 4-comp Best (15/50/10/25)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: 4-comp Best — 1hRSI/1hDC/15mRSI/1hWillR (15/50/10/25)")
    logger.info("-" * 72)
    logger.info("")

    report_4comp = wf.run_cross_tf([
        CrossTFComponent(
            strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.15, label="1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.50, label="1hDC",
        ),
        CrossTFComponent(
            strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
            engine=engine_15m, weight=0.10, label="15mRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_willr_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.25, label="1hWillR",
        ),
    ])
    logger.info("")
    log_report("4-comp 15/50/10/25", report_4comp)
    analyze_components(report_4comp)

    risk_4 = compute_risk_metrics(report_4comp)
    logger.info("")
    logger.info("  4-comp Risk Metrics:")
    logger.info("    Total OOS Return: %+.2f%%", risk_4["total_return"])
    logger.info("    Avg Window Return: %+.2f%%", risk_4["avg_window_return"])
    logger.info("    Std Window Return: %.2f%%", risk_4["std_window_return"])
    logger.info("    Max Drawdown: %.2f%%", risk_4["max_drawdown"])
    logger.info("    Calmar Ratio: %.2f", risk_4["calmar_ratio"])
    logger.info("    Sharpe (per-window): %.2f", risk_4["sharpe_ratio"])
    logger.info("    Sortino (per-window): %.2f", risk_4["sortino_ratio"])
    logger.info("    VaR 95%%: %.2f%%", risk_4["var_95"])
    logger.info("    Expected Shortfall 95%%: %.2f%%", risk_4["es_95"])
    logger.info("    Best Window: %+.2f%%", risk_4["best_window"])
    logger.info("    Worst Window: %+.2f%%", risk_4["worst_window"])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: 4-comp Balanced (15/30/25/30)
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: 4-comp Balanced — 1hRSI/1hDC/15mRSI/1hWillR (15/30/25/30)")
    logger.info("-" * 72)
    logger.info("")

    report_4comp_bal = wf.run_cross_tf([
        CrossTFComponent(
            strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.15, label="1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.30, label="1hDC",
        ),
        CrossTFComponent(
            strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
            engine=engine_15m, weight=0.25, label="15mRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_willr_1h, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.30, label="1hWillR",
        ),
    ])
    logger.info("")
    log_report("4-comp 15/30/25/30", report_4comp_bal)

    risk_4b = compute_risk_metrics(report_4comp_bal)
    logger.info("")
    logger.info("  4-comp Balanced Risk Metrics:")
    logger.info("    Total OOS Return: %+.2f%%", risk_4b["total_return"])
    logger.info("    Max Drawdown: %.2f%%", risk_4b["max_drawdown"])
    logger.info("    Calmar Ratio: %.2f", risk_4b["calmar_ratio"])
    logger.info("    Sharpe (per-window): %.2f", risk_4b["sharpe_ratio"])
    logger.info("    Sortino (per-window): %.2f", risk_4b["sortino_ratio"])
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Head-to-Head Comparison
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 4: Head-to-Head Comparison")
    logger.info("-" * 72)
    logger.info("")

    logger.info("  Window-by-Window Comparison:")
    logger.info("  %-8s %12s %12s %12s", "Window", "3-comp", "4-comp Best", "4-comp Bal")
    logger.info("  " + "-" * 48)
    for w3, w4, w4b in zip(
        report_3comp.windows, report_4comp.windows, report_4comp_bal.windows,
    ):
        logger.info(
            "  W%-6d %+11.2f%% %+11.2f%% %+11.2f%%",
            w3.window_id, w3.weighted_return,
            w4.weighted_return, w4b.weighted_return,
        )
    logger.info("  " + "-" * 48)
    logger.info(
        "  %-8s %+11.2f%% %+11.2f%% %+11.2f%%",
        "TOTAL",
        report_3comp.oos_total_return,
        report_4comp.oos_total_return,
        report_4comp_bal.oos_total_return,
    )
    logger.info("")

    # Which portfolio wins more windows?
    wins_3 = 0
    wins_4 = 0
    ties = 0
    for w3, w4 in zip(report_3comp.windows, report_4comp.windows):
        if w4.weighted_return > w3.weighted_return + 0.01:
            wins_4 += 1
        elif w3.weighted_return > w4.weighted_return + 0.01:
            wins_3 += 1
        else:
            ties += 1

    logger.info("  Window Wins: 3-comp=%d | 4-comp=%d | Tie=%d", wins_3, wins_4, ties)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 5: Comprehensive Comparison Table
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 5: Risk-Adjusted Comparison Summary")
    logger.info("-" * 72)
    logger.info("")

    metrics = [
        ("OOS Return", "total_return", "%"),
        ("Robustness", None, None),
        ("Avg Window", "avg_window_return", "%"),
        ("Std Window", "std_window_return", "%"),
        ("Max DD", "max_drawdown", "%"),
        ("Calmar", "calmar_ratio", ""),
        ("Sharpe", "sharpe_ratio", ""),
        ("Sortino", "sortino_ratio", ""),
        ("VaR 95%", "var_95", "%"),
        ("ES 95%", "es_95", "%"),
        ("Best Window", "best_window", "%"),
        ("Worst Window", "worst_window", "%"),
    ]

    logger.info("  %-18s %14s %14s %14s", "Metric", "3-comp", "4-comp Best", "4-comp Bal")
    logger.info("  " + "-" * 62)

    for label, key, unit in metrics:
        if key is None:  # Robustness
            v3 = f"{int(report_3comp.robustness_score * 100)}%"
            v4 = f"{int(report_4comp.robustness_score * 100)}%"
            v4b = f"{int(report_4comp_bal.robustness_score * 100)}%"
        else:
            u = unit if unit else ""
            v3 = f"{risk_3[key]:+.2f}{u}" if u == "%" else f"{risk_3[key]:.2f}"
            v4 = f"{risk_4[key]:+.2f}{u}" if u == "%" else f"{risk_4[key]:.2f}"
            v4b = f"{risk_4b[key]:+.2f}{u}" if u == "%" else f"{risk_4b[key]:.2f}"
        logger.info("  %-18s %14s %14s %14s", label, v3, v4, v4b)

    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   FINAL VERDICT
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 26 — FINAL VERDICT")
    logger.info("=" * 72)
    logger.info("")

    delta_return = risk_4["total_return"] - risk_3["total_return"]
    delta_calmar = risk_4["calmar_ratio"] - risk_3["calmar_ratio"]

    logger.info("  4-comp vs 3-comp:")
    logger.info("    Return improvement: %+.2f%%", delta_return)
    logger.info("    Calmar improvement: %+.2f", delta_calmar)
    logger.info("    Same robustness: %d%% vs %d%%",
                int(report_4comp.robustness_score * 100),
                int(report_3comp.robustness_score * 100))
    logger.info("")

    if (report_4comp.robustness_score >= report_3comp.robustness_score
            and risk_4["total_return"] > risk_3["total_return"]):
        logger.info("  RECOMMENDATION: UPGRADE to 4-comp (15/50/10/25)")
        logger.info("    - Higher returns with same or better robustness")
        logger.info("    - Phase 25b: 303/375 combos at 88%%, 11/12 params stable")
    elif report_4comp.robustness_score >= report_3comp.robustness_score:
        logger.info("  RECOMMENDATION: KEEP 3-comp (no return improvement)")
    else:
        logger.info("  RECOMMENDATION: KEEP 3-comp (4-comp robustness dropped)")

    logger.info("")
    logger.info("  Production Config (if upgrading):")
    logger.info("    Strategy: CrossTimeframePortfolio 4-comp")
    logger.info("    Weights: 1hRSI=15%%, 1hDC=50%%, 15mRSI=10%%, 1hWillR=25%%")
    logger.info("    Fallback: 3-comp 33/33/34 (proven at 88%%)")
    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 26 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 8 — RSI_MR+MTF Validation & Portfolio Optimization.

Phase 7 findings:
  - RSI_MR+MTF showed strong 5-window OOS (+8.88% to +13.98%, 60% robustness)
  - But full-period returns were suspiciously high (145-151%) — overfitting risk
  - Phase 7 portfolios used RSI_MR standalone (negative OOS), NOT the MTF version
  - BB+Fib 50/50 had 80% robustness — best ever

Phase 8 goals:
  1. Validate RSI_MR+MTF with 7-window WF (more windows = harder to pass)
  2. Build portfolios using RSI_MR+MTF instead of standalone
  3. Test BB + Fib + RSI_MR_MTF 3-strategy portfolio
  4. Full correlation analysis between all strategies
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.basic import BasicIndicators
from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.strategy.bb_squeeze_v2 import BBSqueezeV2Strategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase8")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase8.log", mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

# Suppress noisy loggers
for name in ["src.backtest.engine", "src.strategy.mtf_filter"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def load_data(timeframe: str) -> pd.DataFrame:
    """Load and add indicators to parquet data."""
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


def log_wf_result(
    name: str, report, engine: BacktestEngine, strategy_factory,
    df: pd.DataFrame, htf_df: pd.DataFrame | None = None,
):
    """Log Walk-Forward results and run full backtest."""
    for w in report.windows:
        oos = w.out_of_sample
        is_ = w.in_sample
        logger.info(
            "  W%d: IS %+6.2f%% (WR %d%%, %d tr) | OOS %+6.2f%% (WR %d%%, %d tr)",
            w.window_id,
            is_.total_return, int(is_.win_rate * 100), is_.total_trades,
            oos.total_return, int(oos.win_rate * 100), oos.total_trades,
        )

    logger.info(
        "  OOS: %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        report.oos_total_return,
        int(report.robustness_score * 100),
        report.oos_profitable_windows,
        report.total_windows,
        report.oos_total_trades,
    )

    strat = strategy_factory()
    full = engine.run(strat, df, htf_df=htf_df)
    logger.info(
        "  %s Full %+35.2f%% | Shp %5.2f | DD %5.1f%% | WR %5.1f%% | %4d tr | PF %5.2f",
        name, full.total_return, full.sharpe_ratio, full.max_drawdown,
        full.win_rate * 100, full.total_trades, full.profit_factor,
    )
    return full


def compute_portfolio_wf(name: str, reports_weights: list[tuple]) -> tuple[float, float, int]:
    """Compute portfolio OOS by combining per-window returns with weights."""
    n_windows = reports_weights[0][0].total_windows
    portfolio_oos = []
    total_trades = 0

    for w_idx in range(n_windows):
        weighted_return = 0.0
        w_trades = 0
        label_parts = []

        for report, weight, label in reports_weights:
            if w_idx < len(report.windows):
                oos_ret = report.windows[w_idx].out_of_sample.total_return
                oos_tr = report.windows[w_idx].out_of_sample.total_trades
            else:
                oos_ret = 0.0
                oos_tr = 0
            weighted_return += oos_ret * weight
            w_trades += oos_tr
            label_parts.append(f"{label} OOS {oos_ret:+.2f}%")

        portfolio_oos.append(weighted_return)
        total_trades += w_trades
        logger.info(
            "  W%d: %s → Port %+5.2f%%",
            w_idx + 1, " + ".join(label_parts), weighted_return,
        )

    compounded = 1.0
    for r in portfolio_oos:
        compounded *= (1 + r / 100)
    total_oos = (compounded - 1) * 100
    profitable = sum(1 for r in portfolio_oos if r > 0)
    robustness = profitable / n_windows if n_windows > 0 else 0

    logger.info(
        "  %s OOS: %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        name, total_oos, int(robustness * 100), profitable, n_windows, total_trades,
    )
    return total_oos, robustness, total_trades


def run_phase8():
    """Execute Phase 8 analysis."""
    logger.info("=" * 72)
    logger.info("  PHASE 8 — RSI_MR+MTF Validation & Portfolio Optimization")
    logger.info("=" * 72)
    logger.info("")

    df_1h = load_data("1h")
    df_4h = load_data("4h")

    logger.info("1h data: %d bars (%s ~ %s)", len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("4h data: %d bars (%s ~ %s)", len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine_conservative = BacktestEngine(max_hold_bars=48)
    engine_mr = BacktestEngine(max_hold_bars=36)
    engine_4h = BacktestEngine(max_hold_bars=24)

    # ══════════════════════════════════════════════════════════════════════
    #   PART 0: Baselines (5-window) — reproduce Phase 7 results
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 0: Baselines (5-window WF)")
    logger.info("─" * 72)
    logger.info("")

    # BB baseline
    def bb_factory():
        base = BBSqueezeV2Strategy(
            squeeze_pctile=25, vol_mult=1.1, atr_sl_mult=3.0,
            rr_ratio=2.0, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    logger.info("  --- BBSqueeze+MTF (1h), 5w ---")
    wf5 = WalkForwardAnalyzer(n_windows=5, engine=engine_conservative)
    report_bb_5w = wf5.run(bb_factory, df_1h, htf_df=df_4h)
    log_wf_result("BB_5w", report_bb_5w, engine_conservative, bb_factory, df_1h, df_4h)
    logger.info("")

    # Fib baseline
    def fib_factory():
        return FibonacciRetracementStrategy()

    logger.info("  --- Fib (4h), 5w ---")
    wf5_fib = WalkForwardAnalyzer(n_windows=5, engine=engine_4h)
    report_fib_5w = wf5_fib.run(fib_factory, df_4h)
    log_wf_result("Fib_5w", report_fib_5w, engine_4h, fib_factory, df_4h)
    logger.info("")

    # RSI_MR+MTF — Top 3 configs from Phase 7
    rsi_mtf_configs = [
        {"name": "RSI_30_70_MTF", "oversold": 30, "overbought": 70, "sl": 2.0, "tp": 3.0, "cool": 6},
        {"name": "RSI_35_65_MTF", "oversold": 35, "overbought": 65, "sl": 2.0, "tp": 3.0, "cool": 6},
        {"name": "RSI_25_75_MTF", "oversold": 25, "overbought": 75, "sl": 2.0, "tp": 3.0, "cool": 6},
    ]

    best_rsi_mtf_report_5w = None
    best_rsi_mtf_name_5w = ""
    best_rsi_mtf_factory = None
    best_rsi_mtf_score = -999.0

    reports_rsi_mtf_5w = {}

    for cfg in rsi_mtf_configs:
        logger.info("  --- %s (5w) ---", cfg["name"])

        def make_rsi_mtf_factory(c=cfg):
            def factory():
                base = RSIMeanReversionStrategy(
                    rsi_oversold=c["oversold"],
                    rsi_overbought=c["overbought"],
                    atr_sl_mult=c["sl"],
                    atr_tp_mult=c["tp"],
                    cooldown_bars=c["cool"],
                )
                return MultiTimeframeFilter(base)
            return factory

        factory_fn = make_rsi_mtf_factory(cfg)
        wf = WalkForwardAnalyzer(n_windows=5, engine=engine_mr)
        report = wf.run(factory_fn, df_1h, htf_df=df_4h)
        log_wf_result(cfg["name"], report, engine_mr, factory_fn, df_1h, df_4h)
        reports_rsi_mtf_5w[cfg["name"]] = report

        # Score: prioritize robustness, then OOS return
        score = report.robustness_score * 100 + report.oos_total_return * 0.1
        if score > best_rsi_mtf_score:
            best_rsi_mtf_score = score
            best_rsi_mtf_report_5w = report
            best_rsi_mtf_name_5w = cfg["name"]
            best_rsi_mtf_factory = factory_fn

        logger.info("")

    logger.info("  Best RSI+MTF (5w): %s — OOS %+.2f%%, Robustness %d%%",
                best_rsi_mtf_name_5w,
                best_rsi_mtf_report_5w.oos_total_return if best_rsi_mtf_report_5w else 0,
                int(best_rsi_mtf_report_5w.robustness_score * 100) if best_rsi_mtf_report_5w else 0)
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 1: RSI_MR+MTF — 7-window Validation
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 1: RSI_MR+MTF — 7-window Walk-Forward Validation")
    logger.info("─" * 72)
    logger.info("")

    reports_rsi_mtf_7w = {}

    for cfg in rsi_mtf_configs:
        logger.info("  --- %s (7w) ---", cfg["name"])

        def make_rsi_mtf_factory_7w(c=cfg):
            def factory():
                base = RSIMeanReversionStrategy(
                    rsi_oversold=c["oversold"],
                    rsi_overbought=c["overbought"],
                    atr_sl_mult=c["sl"],
                    atr_tp_mult=c["tp"],
                    cooldown_bars=c["cool"],
                )
                return MultiTimeframeFilter(base)
            return factory

        factory_fn = make_rsi_mtf_factory_7w(cfg)
        wf7 = WalkForwardAnalyzer(n_windows=7, engine=engine_mr)
        report = wf7.run(factory_fn, df_1h, htf_df=df_4h)
        log_wf_result(f"{cfg['name']}_7w", report, engine_mr, factory_fn, df_1h, df_4h)
        reports_rsi_mtf_7w[cfg["name"]] = report
        logger.info("")

    # Also 7-window for BB baseline for comparison
    logger.info("  --- BBSqueeze+MTF (1h), 7w — for comparison ---")
    wf7_bb = WalkForwardAnalyzer(n_windows=7, engine=engine_conservative)
    report_bb_7w = wf7_bb.run(bb_factory, df_1h, htf_df=df_4h)
    log_wf_result("BB_7w", report_bb_7w, engine_conservative, bb_factory, df_1h, df_4h)
    logger.info("")

    # Fib 7-window
    logger.info("  --- Fib (4h), 7w ---")
    wf7_fib = WalkForwardAnalyzer(n_windows=7, engine=engine_4h)
    report_fib_7w = wf7_fib.run(fib_factory, df_4h)
    log_wf_result("Fib_7w", report_fib_7w, engine_4h, fib_factory, df_4h)
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 2: Portfolios with RSI_MR+MTF (5-window)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 2: 2-Strategy Portfolios with RSI_MR+MTF (5w)")
    logger.info("─" * 72)
    logger.info("")

    if best_rsi_mtf_report_5w and best_rsi_mtf_report_5w.total_windows > 0:
        # BB + RSI_MR_MTF
        logger.info("  --- 50%% BB + 50%% %s ---", best_rsi_mtf_name_5w)
        compute_portfolio_wf(
            f"BB+{best_rsi_mtf_name_5w}",
            [(report_bb_5w, 0.5, "BB"), (best_rsi_mtf_report_5w, 0.5, "RSI_MTF")],
        )
        logger.info("")

        # BB + Fib (Phase 7 champion)
        logger.info("  --- 50%% BB + 50%% Fib (Phase 7 champion) ---")
        compute_portfolio_wf(
            "BB+Fib",
            [(report_bb_5w, 0.5, "BB"), (report_fib_5w, 0.5, "Fib")],
        )
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 3: 3-Strategy Portfolios (5-window)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 3: 3-Strategy Portfolios (5w)")
    logger.info("─" * 72)
    logger.info("")

    if best_rsi_mtf_report_5w and best_rsi_mtf_report_5w.total_windows > 0:
        # Equal weight
        logger.info("  --- 33%% BB + 33%% Fib + 33%% RSI_MTF ---")
        compute_portfolio_wf(
            "BB+Fib+RSI_MTF_equal",
            [
                (report_bb_5w, 1/3, "BB"),
                (report_fib_5w, 1/3, "Fib"),
                (best_rsi_mtf_report_5w, 1/3, "RSI_MTF"),
            ],
        )
        logger.info("")

        # BB heavy (proven track record)
        logger.info("  --- 50%% BB + 25%% Fib + 25%% RSI_MTF ---")
        compute_portfolio_wf(
            "BB+Fib+RSI_MTF_BBheavy",
            [
                (report_bb_5w, 0.50, "BB"),
                (report_fib_5w, 0.25, "Fib"),
                (best_rsi_mtf_report_5w, 0.25, "RSI_MTF"),
            ],
        )
        logger.info("")

        # Fib+RSI_MTF heavy (diversification heavy)
        logger.info("  --- 40%% BB + 30%% Fib + 30%% RSI_MTF ---")
        compute_portfolio_wf(
            "BB+Fib+RSI_MTF_diverse",
            [
                (report_bb_5w, 0.40, "BB"),
                (report_fib_5w, 0.30, "Fib"),
                (best_rsi_mtf_report_5w, 0.30, "RSI_MTF"),
            ],
        )
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 4: Portfolios with 7-window WF
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 4: Portfolios with 7-window WF")
    logger.info("─" * 72)
    logger.info("")

    # Find best RSI+MTF in 7-window
    best_rsi_7w_name = ""
    best_rsi_7w_report = None
    best_rsi_7w_score = -999.0
    for name, report in reports_rsi_mtf_7w.items():
        score = report.robustness_score * 100 + report.oos_total_return * 0.1
        if score > best_rsi_7w_score:
            best_rsi_7w_score = score
            best_rsi_7w_report = report
            best_rsi_7w_name = name

    if best_rsi_7w_report and best_rsi_7w_report.total_windows > 0 and report_bb_7w.total_windows > 0:
        logger.info("  Best RSI+MTF (7w): %s — OOS %+.2f%%, Robustness %d%%",
                    best_rsi_7w_name,
                    best_rsi_7w_report.oos_total_return,
                    int(best_rsi_7w_report.robustness_score * 100))
        logger.info("")

        # BB + Fib (7w)
        logger.info("  --- 50%% BB + 50%% Fib (7w) ---")
        compute_portfolio_wf(
            "BB+Fib_7w",
            [(report_bb_7w, 0.5, "BB"), (report_fib_7w, 0.5, "Fib")],
        )
        logger.info("")

        # BB + RSI_MTF (7w)
        logger.info("  --- 50%% BB + 50%% %s (7w) ---", best_rsi_7w_name)
        compute_portfolio_wf(
            f"BB+RSI_MTF_7w",
            [(report_bb_7w, 0.5, "BB"), (best_rsi_7w_report, 0.5, "RSI_MTF")],
        )
        logger.info("")

        # 3-strategy (7w) — BB heavy
        logger.info("  --- 50%% BB + 25%% Fib + 25%% RSI_MTF (7w) ---")
        compute_portfolio_wf(
            "BB+Fib+RSI_MTF_7w",
            [
                (report_bb_7w, 0.50, "BB"),
                (report_fib_7w, 0.25, "Fib"),
                (best_rsi_7w_report, 0.25, "RSI_MTF"),
            ],
        )
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 5: Correlation Analysis
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 5: Strategy Correlation Analysis")
    logger.info("─" * 72)
    logger.info("")

    # 5-window correlation
    logger.info("  [5-window OOS per-window returns]")
    bb_oos_5 = [w.out_of_sample.total_return for w in report_bb_5w.windows]
    fib_oos_5 = [w.out_of_sample.total_return for w in report_fib_5w.windows]

    rsi_mtf_oos_5 = []
    if best_rsi_mtf_report_5w:
        rsi_mtf_oos_5 = [w.out_of_sample.total_return for w in best_rsi_mtf_report_5w.windows]

    logger.info("  Window   BB(1h)     Fib(4h)   RSI_MTF(%s)", best_rsi_mtf_name_5w[:12])
    for i in range(len(bb_oos_5)):
        fib_r = fib_oos_5[i] if i < len(fib_oos_5) else 0
        rsi_r = rsi_mtf_oos_5[i] if i < len(rsi_mtf_oos_5) else 0
        logger.info("  W%d     %+7.2f%%  %+7.2f%%  %+7.2f%%", i + 1, bb_oos_5[i], fib_r, rsi_r)

    if len(bb_oos_5) >= 3 and len(rsi_mtf_oos_5) >= 3:
        n5 = min(len(bb_oos_5), len(rsi_mtf_oos_5))
        same_bb_rsi = sum(1 for a, b in zip(bb_oos_5, rsi_mtf_oos_5) if (a > 0) == (b > 0))
        same_bb_fib = sum(1 for a, b in zip(bb_oos_5, fib_oos_5) if (a > 0) == (b > 0))
        same_fib_rsi = sum(1 for a, b in zip(fib_oos_5, rsi_mtf_oos_5) if (a > 0) == (b > 0))
        logger.info("")
        logger.info("  Directional agreement (lower = more diversification):")
        logger.info("    BB vs RSI_MTF:  %d/%d (%.0f%%)", same_bb_rsi, n5, same_bb_rsi / n5 * 100)
        logger.info("    BB vs Fib:      %d/%d (%.0f%%)", same_bb_fib, n5, same_bb_fib / n5 * 100)
        logger.info("    Fib vs RSI_MTF: %d/%d (%.0f%%)", same_fib_rsi, n5, same_fib_rsi / n5 * 100)

        # Pearson correlation
        if len(bb_oos_5) == len(rsi_mtf_oos_5) == len(fib_oos_5):
            arr_bb = np.array(bb_oos_5)
            arr_fib = np.array(fib_oos_5)
            arr_rsi = np.array(rsi_mtf_oos_5)
            logger.info("")
            logger.info("  Pearson correlation (OOS returns):")

            def safe_corr(a, b):
                if np.std(a) == 0 or np.std(b) == 0:
                    return 0.0
                return float(np.corrcoef(a, b)[0, 1])

            logger.info("    BB vs RSI_MTF:  %.3f", safe_corr(arr_bb, arr_rsi))
            logger.info("    BB vs Fib:      %.3f", safe_corr(arr_bb, arr_fib))
            logger.info("    Fib vs RSI_MTF: %.3f", safe_corr(arr_fib, arr_rsi))

    logger.info("")

    # 7-window correlation
    if report_bb_7w.total_windows > 0 and best_rsi_7w_report and best_rsi_7w_report.total_windows > 0:
        logger.info("  [7-window OOS per-window returns]")
        bb_oos_7 = [w.out_of_sample.total_return for w in report_bb_7w.windows]
        fib_oos_7 = [w.out_of_sample.total_return for w in report_fib_7w.windows]
        rsi_oos_7 = [w.out_of_sample.total_return for w in best_rsi_7w_report.windows]

        logger.info("  Window   BB(1h)     Fib(4h)   RSI_MTF(%s)", best_rsi_7w_name[:12])
        for i in range(len(bb_oos_7)):
            fib_r = fib_oos_7[i] if i < len(fib_oos_7) else 0
            rsi_r = rsi_oos_7[i] if i < len(rsi_oos_7) else 0
            logger.info("  W%d     %+7.2f%%  %+7.2f%%  %+7.2f%%", i + 1, bb_oos_7[i], fib_r, rsi_r)

        n7 = min(len(bb_oos_7), len(rsi_oos_7))
        if n7 >= 3:
            same_bb_rsi = sum(1 for a, b in zip(bb_oos_7, rsi_oos_7) if (a > 0) == (b > 0))
            same_bb_fib = sum(1 for a, b in zip(bb_oos_7, fib_oos_7) if (a > 0) == (b > 0))
            same_fib_rsi = sum(1 for a, b in zip(fib_oos_7, rsi_oos_7) if (a > 0) == (b > 0))
            logger.info("")
            logger.info("  Directional agreement (7w):")
            logger.info("    BB vs RSI_MTF:  %d/%d (%.0f%%)", same_bb_rsi, n7, same_bb_rsi / n7 * 100)
            logger.info("    BB vs Fib:      %d/%d (%.0f%%)", same_bb_fib, n7, same_bb_fib / n7 * 100)
            logger.info("    Fib vs RSI_MTF: %d/%d (%.0f%%)", same_fib_rsi, n7, same_fib_rsi / n7 * 100)

    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 8 — FINAL SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    # Collect all results for ranking
    results = []

    # Single strategies (5w)
    results.append({
        "name": "BB+MTF (1h) 5w",
        "oos": report_bb_5w.oos_total_return,
        "rob": report_bb_5w.robustness_score,
        "trades": report_bb_5w.oos_total_trades,
        "windows": 5,
    })
    results.append({
        "name": "Fib (4h) 5w",
        "oos": report_fib_5w.oos_total_return,
        "rob": report_fib_5w.robustness_score,
        "trades": report_fib_5w.oos_total_trades,
        "windows": 5,
    })

    for cfg_name, report in reports_rsi_mtf_5w.items():
        results.append({
            "name": f"{cfg_name} 5w",
            "oos": report.oos_total_return,
            "rob": report.robustness_score,
            "trades": report.oos_total_trades,
            "windows": 5,
        })

    # 7-window results
    results.append({
        "name": "BB+MTF (1h) 7w",
        "oos": report_bb_7w.oos_total_return,
        "rob": report_bb_7w.robustness_score,
        "trades": report_bb_7w.oos_total_trades,
        "windows": 7,
    })
    results.append({
        "name": "Fib (4h) 7w",
        "oos": report_fib_7w.oos_total_return,
        "rob": report_fib_7w.robustness_score,
        "trades": report_fib_7w.oos_total_trades,
        "windows": 7,
    })

    for cfg_name, report in reports_rsi_mtf_7w.items():
        results.append({
            "name": f"{cfg_name} 7w",
            "oos": report.oos_total_return,
            "rob": report.robustness_score,
            "trades": report.oos_total_trades,
            "windows": 7,
        })

    # Sort by robustness then OOS return
    results.sort(key=lambda x: (x["rob"], x["oos"]), reverse=True)

    logger.info("  %-35s  %8s  %6s  %6s", "Strategy", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 60)
    for r in results:
        logger.info("  %-35s  %+7.2f%%  %5.0f%%  %5d",
                    r["name"], r["oos"], r["rob"] * 100, r["trades"])

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 8 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    run_phase8()

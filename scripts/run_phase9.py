#!/usr/bin/env python3
"""Phase 9 — New Strategy Exploration: VWAP MR + Regime Switching.

Phase 7/8 findings:
  - BB+RSI_MTF 50/50 (7w): 71% robustness, OOS +12.02% — best portfolio
  - BB+Fib+RSI_MTF 50/25/25 (7w): 71% robustness, OOS +8.28%
  - RSI_MR+MTF standalone: 57-60% robustness (suspicious Full return 145%+)
  - Fib collapses at 7w (14% robustness)

Phase 9 goals:
  1. VWAP Mean Reversion (1h) — new strategy, multiple param sets
  2. VWAP+MTF — with 4h trend filter
  3. Regime Switching (ADX-based BB<->RSI_MR) — meta strategy
  4. Portfolio combinations with VWAP and regime strategies
  5. All validated with 5w and 7w Walk-Forward
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.basic import BasicIndicators
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.strategy.bb_squeeze_breakout import BBSqueezeBreakoutStrategy
from src.strategy.bb_squeeze_v2 import BBSqueezeV2Strategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
from src.strategy.regime_switch import RegimeSwitchStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase9")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase9.log", mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

for name in ["src.backtest.engine", "src.strategy.mtf_filter"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def load_data(timeframe: str) -> pd.DataFrame:
    """Load parquet data and add indicators."""
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


def log_wf(name: str, report, engine: BacktestEngine,
           strategy_factory, df: pd.DataFrame, htf_df=None) -> BacktestResult:
    """Log WF results and run full backtest."""
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
        report.oos_profitable_windows, report.total_windows,
        report.oos_total_trades,
    )
    full = engine.run(strategy_factory(), df, htf_df=htf_df)
    logger.info(
        "  %s Full %+8.2f%% | Shp %5.2f | DD %5.1f%% | WR %5.1f%% | %4d tr | PF %5.2f",
        name, full.total_return, full.sharpe_ratio, full.max_drawdown,
        full.win_rate * 100, full.total_trades, full.profit_factor,
    )
    return full


def compute_portfolio(name: str, reports_weights: list[tuple]) -> tuple[float, float, int]:
    """Compute portfolio OOS from per-window weighted returns."""
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
            label_parts.append(f"{label} {oos_ret:+.2f}%")

        portfolio_oos.append(weighted_return)
        total_trades += w_trades
        logger.info("  W%d: %s → Port %+5.2f%%",
                     w_idx + 1, " + ".join(label_parts), weighted_return)

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


def run_phase9() -> None:
    logger.info("=" * 72)
    logger.info("  PHASE 9 — VWAP Mean Reversion + Regime Switching")
    logger.info("=" * 72)
    logger.info("")

    df_1h = load_data("1h")
    df_4h = load_data("4h")

    logger.info("1h data: %d bars (%s ~ %s)",
                len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("4h data: %d bars (%s ~ %s)",
                len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48)
    engine_mr = BacktestEngine(max_hold_bars=36)

    all_results: list[tuple[str, object, BacktestResult]] = []

    # ══════════════════════════════════════════════════════════════════════
    #   PART 0: Baselines (5w)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 0: Baselines (5-window WF)")
    logger.info("─" * 72)
    logger.info("")

    # BB+MTF baseline
    def bb_factory():
        base = BBSqueezeV2Strategy(
            squeeze_pctile=25, vol_mult=1.1, atr_sl_mult=3.0,
            rr_ratio=2.0, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    logger.info("  --- BBSqueeze+MTF (1h) ---")
    wf5 = WalkForwardAnalyzer(n_windows=5, engine=engine_1h)
    rpt_bb = wf5.run(bb_factory, df_1h, htf_df=df_4h)
    full_bb = log_wf("BB+MTF", rpt_bb, engine_1h, bb_factory, df_1h, df_4h)
    all_results.append(("BB+MTF_1h", rpt_bb, full_bb))
    logger.info("")

    # RSI_MR+MTF baseline (best from Phase 8: RSI_35_65)
    def rsi_mtf_factory():
        base = RSIMeanReversionStrategy(
            rsi_oversold=35, rsi_overbought=65,
            atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    logger.info("  --- RSI_35_65+MTF (1h) ---")
    rpt_rsi_mtf = wf5.run(rsi_mtf_factory, df_1h, htf_df=df_4h)
    full_rsi_mtf = log_wf("RSI_MTF", rpt_rsi_mtf, engine_mr, rsi_mtf_factory, df_1h, df_4h)
    all_results.append(("RSI_35_65_MTF", rpt_rsi_mtf, full_rsi_mtf))
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 1: VWAP Mean Reversion (1h) — Parameter Sweep
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 1: VWAP Mean Reversion (1h) — Parameter Sweep")
    logger.info("─" * 72)
    logger.info("")

    vwap_configs = [
        {"name": "VWAP_48_2.0", "period": 48, "band": 2.0, "rsi": 35, "sl": 2.0, "tp_pct": 0.8, "cool": 6},
        {"name": "VWAP_48_1.5", "period": 48, "band": 1.5, "rsi": 35, "sl": 2.0, "tp_pct": 0.8, "cool": 6},
        {"name": "VWAP_48_2.5", "period": 48, "band": 2.5, "rsi": 30, "sl": 2.5, "tp_pct": 0.8, "cool": 6},
        {"name": "VWAP_24_2.0", "period": 24, "band": 2.0, "rsi": 35, "sl": 2.0, "tp_pct": 0.8, "cool": 4},
        {"name": "VWAP_72_2.0", "period": 72, "band": 2.0, "rsi": 35, "sl": 2.0, "tp_pct": 0.8, "cool": 8},
        {"name": "VWAP_48_2.0_full", "period": 48, "band": 2.0, "rsi": 35, "sl": 2.0, "tp_pct": 1.0, "cool": 6},
    ]

    best_vwap_report = None
    best_vwap_name = ""
    best_vwap_factory = None
    best_vwap_score = -999.0

    for cfg in vwap_configs:
        logger.info("  --- %s ---", cfg["name"])

        def make_vwap_factory(c=cfg):
            def factory():
                return VWAPMeanReversionStrategy(
                    vwap_period=c["period"],
                    band_mult=c["band"],
                    rsi_threshold=c["rsi"],
                    atr_sl_mult=c["sl"],
                    tp_to_vwap_pct=c["tp_pct"],
                    cooldown_bars=c["cool"],
                )
            return factory

        factory_fn = make_vwap_factory(cfg)
        wf_vwap = WalkForwardAnalyzer(n_windows=5, engine=engine_mr)
        report = wf_vwap.run(factory_fn, df_1h)
        full = log_wf(cfg["name"], report, engine_mr, factory_fn, df_1h)
        all_results.append((cfg["name"], report, full))

        score = report.robustness_score * 100 + report.oos_total_return * 0.1
        if score > best_vwap_score:
            best_vwap_score = score
            best_vwap_report = report
            best_vwap_name = cfg["name"]
            best_vwap_factory = factory_fn

        logger.info("")

    logger.info("  Best VWAP: %s — OOS %+.2f%%, Robustness %d%%",
                best_vwap_name,
                best_vwap_report.oos_total_return if best_vwap_report else 0,
                int(best_vwap_report.robustness_score * 100) if best_vwap_report else 0)
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 2: VWAP + MTF (4h trend filter)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 2: VWAP + MTF (4h trend filter)")
    logger.info("─" * 72)
    logger.info("")

    # Top 3 VWAP configs with MTF
    top_vwap_cfgs = sorted(
        vwap_configs,
        key=lambda c: next(
            (r[1].robustness_score * 100 + r[1].oos_total_return * 0.1
             for r in all_results if r[0] == c["name"]),
            -999,
        ),
        reverse=True,
    )[:3]

    best_vwap_mtf_report = None
    best_vwap_mtf_name = ""
    best_vwap_mtf_factory = None
    best_vwap_mtf_score = -999.0

    for cfg in top_vwap_cfgs:
        label = f"{cfg['name']}_MTF"
        logger.info("  --- %s ---", label)

        def make_vwap_mtf_factory(c=cfg):
            def factory():
                base = VWAPMeanReversionStrategy(
                    vwap_period=c["period"],
                    band_mult=c["band"],
                    rsi_threshold=c["rsi"],
                    atr_sl_mult=c["sl"],
                    tp_to_vwap_pct=c["tp_pct"],
                    cooldown_bars=c["cool"],
                )
                return MultiTimeframeFilter(base)
            return factory

        factory_fn = make_vwap_mtf_factory(cfg)
        wf = WalkForwardAnalyzer(n_windows=5, engine=engine_mr)
        report = wf.run(factory_fn, df_1h, htf_df=df_4h)
        full = log_wf(label, report, engine_mr, factory_fn, df_1h, df_4h)
        all_results.append((label, report, full))

        score = report.robustness_score * 100 + report.oos_total_return * 0.1
        if score > best_vwap_mtf_score:
            best_vwap_mtf_score = score
            best_vwap_mtf_report = report
            best_vwap_mtf_name = label
            best_vwap_mtf_factory = factory_fn

        logger.info("")

    logger.info("  Best VWAP+MTF: %s — OOS %+.2f%%, Robustness %d%%",
                best_vwap_mtf_name,
                best_vwap_mtf_report.oos_total_return if best_vwap_mtf_report else 0,
                int(best_vwap_mtf_report.robustness_score * 100) if best_vwap_mtf_report else 0)
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 3: Regime Switching — ADX-based BB <-> RSI_MR
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 3: Regime Switching (ADX-based)")
    logger.info("─" * 72)
    logger.info("")

    regime_configs = [
        {"name": "Regime_ADX20", "threshold": 20},
        {"name": "Regime_ADX25", "threshold": 25},
        {"name": "Regime_ADX30", "threshold": 30},
        {"name": "Regime_ADX35", "threshold": 35},
    ]

    best_regime_report = None
    best_regime_name = ""
    best_regime_factory = None
    best_regime_score = -999.0

    for cfg in regime_configs:
        label = cfg["name"]
        logger.info("  --- %s ---", label)

        def make_regime_factory(c=cfg):
            def factory():
                trend_strat = BBSqueezeBreakoutStrategy(
                    squeeze_lookback=100, squeeze_pctile=25.0,
                    vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
                    require_trend=False, cooldown_bars=6,
                )
                range_strat = RSIMeanReversionStrategy(
                    rsi_oversold=35, rsi_overbought=65,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
                return RegimeSwitchStrategy(
                    trend_strategy=trend_strat,
                    range_strategy=range_strat,
                    adx_threshold=c["threshold"],
                )
            return factory

        factory_fn = make_regime_factory(cfg)
        wf = WalkForwardAnalyzer(n_windows=5, engine=engine_mr)
        report = wf.run(factory_fn, df_1h)
        full = log_wf(label, report, engine_mr, factory_fn, df_1h)
        all_results.append((label, report, full))

        score = report.robustness_score * 100 + report.oos_total_return * 0.1
        if score > best_regime_score:
            best_regime_score = score
            best_regime_report = report
            best_regime_name = label
            best_regime_factory = factory_fn

        logger.info("")

    # Also test Regime+MTF for best threshold
    if best_regime_report and best_regime_report.robustness_score >= 0.4:
        threshold = int(best_regime_name.split("ADX")[1])
        label = f"Regime_ADX{threshold}_MTF"
        logger.info("  --- %s ---", label)

        def make_regime_mtf_factory(th=threshold):
            def factory():
                trend_strat = BBSqueezeBreakoutStrategy(
                    squeeze_lookback=100, squeeze_pctile=25.0,
                    vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
                    require_trend=False, cooldown_bars=6,
                )
                range_strat = RSIMeanReversionStrategy(
                    rsi_oversold=35, rsi_overbought=65,
                    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
                )
                base = RegimeSwitchStrategy(
                    trend_strategy=trend_strat,
                    range_strategy=range_strat,
                    adx_threshold=th,
                )
                return MultiTimeframeFilter(base)
            return factory

        factory_fn = make_regime_mtf_factory(threshold)
        wf = WalkForwardAnalyzer(n_windows=5, engine=engine_mr)
        report = wf.run(factory_fn, df_1h, htf_df=df_4h)
        full = log_wf(label, report, engine_mr, factory_fn, df_1h, df_4h)
        all_results.append((label, report, full))

        if report.robustness_score * 100 + report.oos_total_return * 0.1 > best_regime_score:
            best_regime_report = report
            best_regime_name = label
            best_regime_factory = factory_fn

        logger.info("")

    logger.info("  Best Regime: %s — OOS %+.2f%%, Robustness %d%%",
                best_regime_name,
                best_regime_report.oos_total_return if best_regime_report else 0,
                int(best_regime_report.robustness_score * 100) if best_regime_report else 0)
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 4: 7-window WF Validation on best strategies
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 4: 7-window WF Validation")
    logger.info("─" * 72)
    logger.info("")

    # BB+MTF 7w
    logger.info("  --- BB+MTF (1h) 7w ---")
    wf7 = WalkForwardAnalyzer(n_windows=7, engine=engine_1h)
    rpt_bb_7w = wf7.run(bb_factory, df_1h, htf_df=df_4h)
    log_wf("BB+MTF_7w", rpt_bb_7w, engine_1h, bb_factory, df_1h, df_4h)
    logger.info("")

    # Best VWAP 7w (if robustness >= 40%)
    rpt_vwap_7w = None
    if best_vwap_report and best_vwap_report.robustness_score >= 0.4:
        logger.info("  --- %s (7w) ---", best_vwap_name)
        wf7v = WalkForwardAnalyzer(n_windows=7, engine=engine_mr)
        rpt_vwap_7w = wf7v.run(best_vwap_factory, df_1h)
        log_wf(f"{best_vwap_name}_7w", rpt_vwap_7w, engine_mr, best_vwap_factory, df_1h)
        logger.info("")

    # Best VWAP+MTF 7w
    rpt_vwap_mtf_7w = None
    if best_vwap_mtf_report and best_vwap_mtf_report.robustness_score >= 0.4:
        logger.info("  --- %s (7w) ---", best_vwap_mtf_name)
        wf7vm = WalkForwardAnalyzer(n_windows=7, engine=engine_mr)
        rpt_vwap_mtf_7w = wf7vm.run(best_vwap_mtf_factory, df_1h, htf_df=df_4h)
        log_wf(f"{best_vwap_mtf_name}_7w", rpt_vwap_mtf_7w, engine_mr,
               best_vwap_mtf_factory, df_1h, df_4h)
        logger.info("")

    # Best Regime 7w
    rpt_regime_7w = None
    if best_regime_report and best_regime_report.robustness_score >= 0.4:
        logger.info("  --- %s (7w) ---", best_regime_name)
        # Determine engine and htf_df based on whether MTF is used
        is_mtf = "MTF" in best_regime_name
        wf7r = WalkForwardAnalyzer(n_windows=7, engine=engine_mr)
        rpt_regime_7w = wf7r.run(
            best_regime_factory, df_1h,
            htf_df=df_4h if is_mtf else None,
        )
        log_wf(f"{best_regime_name}_7w", rpt_regime_7w, engine_mr,
               best_regime_factory, df_1h, df_4h if is_mtf else None)
        logger.info("")

    # RSI_MTF 7w
    logger.info("  --- RSI_35_65+MTF (7w) ---")
    wf7r_rsi = WalkForwardAnalyzer(n_windows=7, engine=engine_mr)
    rpt_rsi_mtf_7w = wf7r_rsi.run(rsi_mtf_factory, df_1h, htf_df=df_4h)
    log_wf("RSI_MTF_7w", rpt_rsi_mtf_7w, engine_mr, rsi_mtf_factory, df_1h, df_4h)
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 5: Portfolio Combinations (5w and 7w)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 5: Portfolio Combinations")
    logger.info("─" * 72)
    logger.info("")

    # 5w portfolios
    logger.info("  [5-window portfolios]")
    logger.info("")

    # BB + best VWAP (if available)
    if best_vwap_report and best_vwap_report.total_windows == rpt_bb.total_windows:
        logger.info("  --- 50%% BB + 50%% %s ---", best_vwap_name)
        compute_portfolio("BB+VWAP", [
            (rpt_bb, 0.5, "BB"),
            (best_vwap_report, 0.5, "VWAP"),
        ])
        logger.info("")

    # BB + best VWAP+MTF
    if best_vwap_mtf_report and best_vwap_mtf_report.total_windows == rpt_bb.total_windows:
        logger.info("  --- 50%% BB + 50%% %s ---", best_vwap_mtf_name)
        compute_portfolio("BB+VWAP_MTF", [
            (rpt_bb, 0.5, "BB"),
            (best_vwap_mtf_report, 0.5, "VWAP_MTF"),
        ])
        logger.info("")

    # BB + RSI_MTF (Phase 8 champion for comparison)
    logger.info("  --- 50%% BB + 50%% RSI_MTF (Phase 8 ref) ---")
    compute_portfolio("BB+RSI_MTF", [
        (rpt_bb, 0.5, "BB"),
        (rpt_rsi_mtf, 0.5, "RSI_MTF"),
    ])
    logger.info("")

    # BB + VWAP + RSI (3-way if both available)
    if (best_vwap_mtf_report
        and best_vwap_mtf_report.total_windows == rpt_bb.total_windows):
        logger.info("  --- 40%% BB + 30%% RSI_MTF + 30%% %s ---", best_vwap_mtf_name)
        compute_portfolio("BB+RSI+VWAP_3way", [
            (rpt_bb, 0.40, "BB"),
            (rpt_rsi_mtf, 0.30, "RSI_MTF"),
            (best_vwap_mtf_report, 0.30, "VWAP_MTF"),
        ])
        logger.info("")

    # 7w portfolios
    logger.info("  [7-window portfolios]")
    logger.info("")

    logger.info("  --- 50%% BB + 50%% RSI_MTF (7w) ---")
    compute_portfolio("BB+RSI_MTF_7w", [
        (rpt_bb_7w, 0.5, "BB"),
        (rpt_rsi_mtf_7w, 0.5, "RSI_MTF"),
    ])
    logger.info("")

    if rpt_vwap_mtf_7w and rpt_vwap_mtf_7w.total_windows == rpt_bb_7w.total_windows:
        logger.info("  --- 50%% BB + 50%% VWAP_MTF (7w) ---")
        compute_portfolio("BB+VWAP_MTF_7w", [
            (rpt_bb_7w, 0.5, "BB"),
            (rpt_vwap_mtf_7w, 0.5, "VWAP_MTF"),
        ])
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   FINAL RANKING
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 9 — FINAL RANKING (Single Strategies, 5w)")
    logger.info("=" * 72)
    logger.info("")

    all_results.sort(
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    )

    logger.info(
        "  %-30s %8s %6s %7s %8s %6s %5s %5s",
        "Strategy", "OOS Ret", "WF Rob", "Tr(OOS)",
        "Full Ret", "MaxDD", "PF", "Shp",
    )
    logger.info("  " + "-" * 85)

    for name, rpt, r_full in all_results:
        logger.info(
            "  %-30s %+7.2f%% %5.0f%% %6d   %+7.2f%% %5.1f%% %5.2f %5.2f",
            name, rpt.oos_total_return, rpt.robustness_score * 100,
            rpt.oos_total_trades,
            r_full.total_return, r_full.max_drawdown,
            r_full.profit_factor, r_full.sharpe_ratio,
        )

    logger.info("")

    # Best single strategy
    if all_results:
        best_name, best_rpt, best_full = all_results[0]
        logger.info("  Best single: %s", best_name)
        logger.info("    OOS: %+.2f%% | Robustness: %d%% | Full: %+.2f%% | DD: %.1f%%",
                     best_rpt.oos_total_return,
                     int(best_rpt.robustness_score * 100),
                     best_full.total_return, best_full.max_drawdown)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 9 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    run_phase9()

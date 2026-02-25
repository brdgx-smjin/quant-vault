#!/usr/bin/env python3
"""Phase 10 — Portfolio Deep Validation + MACD Momentum Strategy.

Phase 9 findings:
  - VWAP_24_2.0+MTF (5w): 80% robustness, OOS +8.91%
  - Regime_ADX20 (5w): 80% robustness, OOS +5.48%
  - BB+RSI_MTF 50/50 (7w): 71% robustness, OOS +8.19% — best portfolio
  - BB+VWAP_MTF 50/50 (7w): 71% robustness, OOS +4.49%

Phase 10 goals:
  1. Re-validate top single strategies (5w baselines)
  2. New strategy: MACD Momentum + MTF (pure momentum, not MR or breakout)
  3. Extended 9-window WF for extra robustness validation
  4. Comprehensive portfolio optimization (weights + combinations)
  5. BB+Fib portfolio validation (as requested — expect low performance)
  6. Final ranking across all strategies and portfolios
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
from src.strategy.bb_squeeze_v2 import BBSqueezeV2Strategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
from src.strategy.regime_switch import RegimeSwitchStrategy
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategy.macd_momentum import MACDMomentumStrategy
from src.strategy.bb_squeeze_breakout import BBSqueezeBreakoutStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase10")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase10.log", mode="w")
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


# ─── Strategy Factories ──────────────────────────────────────────

def bb_mtf_factory():
    """BB+MTF: Proven baseline (Phase 9: 60% rob, OOS +5.71%)."""
    base = BBSqueezeV2Strategy(
        squeeze_pctile=25, vol_mult=1.1, atr_sl_mult=3.0,
        rr_ratio=2.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def rsi_mtf_factory():
    """RSI_35_65+MTF: Strong mean reversion (Phase 9: 60% rob, OOS +6.27%)."""
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def vwap_mtf_factory():
    """VWAP_24_2.0+MTF: Best single Phase 9 (80% rob, OOS +8.91%)."""
    base = VWAPMeanReversionStrategy(
        vwap_period=24, band_mult=2.0, rsi_threshold=35,
        atr_sl_mult=2.0, tp_to_vwap_pct=0.8, cooldown_bars=4,
    )
    return MultiTimeframeFilter(base)


def regime_factory():
    """Regime_ADX20: Meta-strategy (Phase 9: 80% rob, OOS +5.48%)."""
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
        adx_threshold=20,
    )


def fib_factory():
    """Fibonacci (4h): Low-volume diversifier."""
    return FibonacciRetracementStrategy(
        entry_levels=(0.5, 0.618),
        tp_extension=1.618,
        tolerance_pct=0.05,
        lookback=50,
        atr_sl_mult=1.5,
        require_trend=True,
    )


# ─── MACD Factories ──────────────────────────────────────────────

def macd_factory():
    """MACD Momentum: zero-cross + crossover."""
    return MACDMomentumStrategy(
        macd_fast=12, macd_slow=26, macd_signal=9,
        require_zero_cross=True, rsi_guard=70,
        atr_sl_mult=2.0, rr_ratio=2.0, vol_mult=0.8, cooldown_bars=6,
    )


def macd_nozero_factory():
    """MACD Momentum: crossover only (no zero-cross requirement)."""
    return MACDMomentumStrategy(
        macd_fast=12, macd_slow=26, macd_signal=9,
        require_zero_cross=False, rsi_guard=70,
        atr_sl_mult=2.0, rr_ratio=2.0, vol_mult=0.8, cooldown_bars=6,
    )


def macd_mtf_factory():
    """MACD Momentum + MTF: zero-cross with 4h trend filter."""
    return MultiTimeframeFilter(macd_factory())


def macd_nozero_mtf_factory():
    """MACD Momentum + MTF: crossover only with 4h trend filter."""
    return MultiTimeframeFilter(macd_nozero_factory())


def macd_tight_factory():
    """MACD Momentum: tighter params (fast=8, slow=21)."""
    return MACDMomentumStrategy(
        macd_fast=8, macd_slow=21, macd_signal=9,
        require_zero_cross=True, rsi_guard=65,
        atr_sl_mult=2.5, rr_ratio=2.5, vol_mult=0.8, cooldown_bars=8,
    )


def macd_tight_mtf_factory():
    """MACD tight + MTF."""
    return MultiTimeframeFilter(macd_tight_factory())


def run_phase10() -> None:
    logger.info("=" * 72)
    logger.info("  PHASE 10 — Portfolio Deep Validation + MACD Momentum")
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
    engine_4h = BacktestEngine(max_hold_bars=12)

    # Store all single strategy results: (name, report_5w, report_7w, full)
    single_results_5w: list[tuple[str, object, BacktestResult]] = []
    reports_5w: dict[str, object] = {}

    # ══════════════════════════════════════════════════════════════════════
    #   PART 0: Baselines — 5-window WF (re-confirm)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 0: Baselines (5-window WF)")
    logger.info("─" * 72)
    logger.info("")

    wf5 = WalkForwardAnalyzer(n_windows=5, engine=engine_1h)
    wf5_mr = WalkForwardAnalyzer(n_windows=5, engine=engine_mr)

    baselines = [
        ("BB+MTF", bb_mtf_factory, wf5, engine_1h, df_1h, df_4h),
        ("RSI_35_65+MTF", rsi_mtf_factory, wf5_mr, engine_mr, df_1h, df_4h),
        ("VWAP_24_2.0+MTF", vwap_mtf_factory, wf5_mr, engine_mr, df_1h, df_4h),
        ("Regime_ADX20", regime_factory,
         WalkForwardAnalyzer(n_windows=5, engine=engine_mr), engine_mr, df_1h, None),
    ]

    for name, factory_fn, wf_obj, eng, df, htf in baselines:
        logger.info("  --- %s ---", name)
        rpt = wf_obj.run(factory_fn, df, htf_df=htf)
        full = log_wf(name, rpt, eng, factory_fn, df, htf)
        single_results_5w.append((name, rpt, full))
        reports_5w[name] = rpt
        logger.info("")

    # Fib (4h) baseline
    logger.info("  --- Fib (4h) ---")
    wf5_4h = WalkForwardAnalyzer(n_windows=5, engine=engine_4h)
    rpt_fib = wf5_4h.run(fib_factory, df_4h)
    full_fib = log_wf("Fib_4h", rpt_fib, engine_4h, fib_factory, df_4h)
    single_results_5w.append(("Fib_4h", rpt_fib, full_fib))
    reports_5w["Fib_4h"] = rpt_fib
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 1: MACD Momentum — New Strategy
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 1: MACD Momentum Strategy (1h)")
    logger.info("─" * 72)
    logger.info("")

    macd_configs = [
        ("MACD_ZeroCross", macd_factory, False),
        ("MACD_NoZero", macd_nozero_factory, False),
        ("MACD_Tight", macd_tight_factory, False),
        ("MACD_ZeroCross+MTF", macd_mtf_factory, True),
        ("MACD_NoZero+MTF", macd_nozero_mtf_factory, True),
        ("MACD_Tight+MTF", macd_tight_mtf_factory, True),
    ]

    best_macd_report = None
    best_macd_name = ""
    best_macd_score = -999.0

    for name, factory_fn, uses_mtf in macd_configs:
        logger.info("  --- %s ---", name)
        htf = df_4h if uses_mtf else None
        wf = WalkForwardAnalyzer(n_windows=5, engine=engine_1h)
        rpt = wf.run(factory_fn, df_1h, htf_df=htf)
        full = log_wf(name, rpt, engine_1h, factory_fn, df_1h, htf)
        single_results_5w.append((name, rpt, full))
        reports_5w[name] = rpt

        score = rpt.robustness_score * 100 + rpt.oos_total_return * 0.1
        if score > best_macd_score:
            best_macd_score = score
            best_macd_report = rpt
            best_macd_name = name

        logger.info("")

    logger.info("  Best MACD: %s — OOS %+.2f%%, Robustness %d%%",
                best_macd_name,
                best_macd_report.oos_total_return if best_macd_report else 0,
                int(best_macd_report.robustness_score * 100) if best_macd_report else 0)
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 2: 7-window WF Validation (top strategies)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 2: 7-window WF Validation")
    logger.info("─" * 72)
    logger.info("")

    reports_7w: dict[str, object] = {}

    top_7w = [
        ("BB+MTF_7w", bb_mtf_factory, engine_1h, df_1h, df_4h),
        ("RSI_35_65+MTF_7w", rsi_mtf_factory, engine_mr, df_1h, df_4h),
        ("VWAP_24_2.0+MTF_7w", vwap_mtf_factory, engine_mr, df_1h, df_4h),
        ("Regime_ADX20_7w", regime_factory, engine_mr, df_1h, None),
    ]

    for name, factory_fn, eng, df, htf in top_7w:
        logger.info("  --- %s ---", name)
        wf7 = WalkForwardAnalyzer(n_windows=7, engine=eng)
        rpt = wf7.run(factory_fn, df, htf_df=htf)
        log_wf(name, rpt, eng, factory_fn, df, htf)
        reports_7w[name] = rpt
        logger.info("")

    # Fib 7w
    logger.info("  --- Fib_4h_7w ---")
    wf7_4h = WalkForwardAnalyzer(n_windows=7, engine=engine_4h)
    rpt_fib_7w = wf7_4h.run(fib_factory, df_4h)
    log_wf("Fib_4h_7w", rpt_fib_7w, engine_4h, fib_factory, df_4h)
    reports_7w["Fib_4h_7w"] = rpt_fib_7w
    logger.info("")

    # Best MACD at 7w (if robustness >= 40%)
    if best_macd_report and best_macd_report.robustness_score >= 0.4:
        macd_7w_name = f"{best_macd_name}_7w"
        logger.info("  --- %s ---", macd_7w_name)
        uses_mtf = "MTF" in best_macd_name

        # Find the factory function for best MACD
        macd_7w_factory = None
        for cfg_name, cfg_factory, _ in macd_configs:
            if cfg_name == best_macd_name:
                macd_7w_factory = cfg_factory
                break

        if macd_7w_factory:
            wf7m = WalkForwardAnalyzer(n_windows=7, engine=engine_1h)
            rpt_macd_7w = wf7m.run(
                macd_7w_factory, df_1h,
                htf_df=df_4h if uses_mtf else None,
            )
            log_wf(macd_7w_name, rpt_macd_7w, engine_1h, macd_7w_factory,
                   df_1h, df_4h if uses_mtf else None)
            reports_7w[macd_7w_name] = rpt_macd_7w
            logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 3: 9-window WF (extra rigor for top 3 strategies)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 3: 9-window WF (Ultra Validation)")
    logger.info("─" * 72)
    logger.info("")

    reports_9w: dict[str, object] = {}

    top_9w = [
        ("BB+MTF_9w", bb_mtf_factory, engine_1h, df_1h, df_4h),
        ("VWAP_24_2.0+MTF_9w", vwap_mtf_factory, engine_mr, df_1h, df_4h),
        ("RSI_35_65+MTF_9w", rsi_mtf_factory, engine_mr, df_1h, df_4h),
    ]

    for name, factory_fn, eng, df, htf in top_9w:
        logger.info("  --- %s ---", name)
        wf9 = WalkForwardAnalyzer(n_windows=9, engine=eng)
        rpt = wf9.run(factory_fn, df, htf_df=htf)
        log_wf(name, rpt, eng, factory_fn, df, htf)
        reports_9w[name] = rpt
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 4: Portfolio Combinations (5w)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 4: Portfolio Combinations (5-window)")
    logger.info("─" * 72)
    logger.info("")

    rpt_bb = reports_5w["BB+MTF"]
    rpt_rsi = reports_5w["RSI_35_65+MTF"]
    rpt_vwap = reports_5w["VWAP_24_2.0+MTF"]
    rpt_fib_5w = reports_5w["Fib_4h"]

    # --- Requested: BB+Fib 50/50 (5w) ---
    logger.info("  --- 50%% BB + 50%% Fib (5w) — Requested Validation ---")
    compute_portfolio("BB+Fib_50_50", [
        (rpt_bb, 0.5, "BB"), (rpt_fib_5w, 0.5, "Fib"),
    ])
    logger.info("")

    # --- BB + RSI_MTF 50/50 ---
    logger.info("  --- 50%% BB + 50%% RSI_MTF (5w) ---")
    compute_portfolio("BB+RSI_50_50", [
        (rpt_bb, 0.5, "BB"), (rpt_rsi, 0.5, "RSI"),
    ])
    logger.info("")

    # --- BB + VWAP_MTF 50/50 ---
    logger.info("  --- 50%% BB + 50%% VWAP_MTF (5w) ---")
    compute_portfolio("BB+VWAP_50_50", [
        (rpt_bb, 0.5, "BB"), (rpt_vwap, 0.5, "VWAP"),
    ])
    logger.info("")

    # --- BB + RSI + VWAP 3-way (equal weight) ---
    logger.info("  --- 33%% BB + 33%% RSI_MTF + 33%% VWAP_MTF (5w) ---")
    compute_portfolio("BB+RSI+VWAP_equal", [
        (rpt_bb, 0.333, "BB"), (rpt_rsi, 0.333, "RSI"), (rpt_vwap, 0.334, "VWAP"),
    ])
    logger.info("")

    # --- BB + RSI + VWAP weighted (40/30/30) ---
    logger.info("  --- 40%% BB + 30%% RSI_MTF + 30%% VWAP_MTF (5w) ---")
    compute_portfolio("BB+RSI+VWAP_40_30_30", [
        (rpt_bb, 0.40, "BB"), (rpt_rsi, 0.30, "RSI"), (rpt_vwap, 0.30, "VWAP"),
    ])
    logger.info("")

    # --- BB + RSI + VWAP weighted (20/40/40) — overweight MR ---
    logger.info("  --- 20%% BB + 40%% RSI_MTF + 40%% VWAP_MTF (5w) ---")
    compute_portfolio("BB+RSI+VWAP_20_40_40", [
        (rpt_bb, 0.20, "BB"), (rpt_rsi, 0.40, "RSI"), (rpt_vwap, 0.40, "VWAP"),
    ])
    logger.info("")

    # --- BB + VWAP + Regime 3-way ---
    rpt_regime = reports_5w["Regime_ADX20"]
    logger.info("  --- 33%% BB + 33%% VWAP_MTF + 33%% Regime (5w) ---")
    compute_portfolio("BB+VWAP+Regime_equal", [
        (rpt_bb, 0.333, "BB"), (rpt_vwap, 0.333, "VWAP"), (rpt_regime, 0.334, "Regime"),
    ])
    logger.info("")

    # --- If best MACD has decent robustness, include in portfolio ---
    if best_macd_report and best_macd_report.robustness_score >= 0.4:
        rpt_macd_best = reports_5w[best_macd_name]
        logger.info("  --- 40%% BB + 30%% VWAP_MTF + 30%% %s (5w) ---", best_macd_name)
        compute_portfolio("BB+VWAP+MACD", [
            (rpt_bb, 0.40, "BB"), (rpt_vwap, 0.30, "VWAP"),
            (rpt_macd_best, 0.30, "MACD"),
        ])
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 5: Portfolio Combinations (7w)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 5: Portfolio Combinations (7-window)")
    logger.info("─" * 72)
    logger.info("")

    rpt_bb_7w = reports_7w["BB+MTF_7w"]
    rpt_rsi_7w = reports_7w["RSI_35_65+MTF_7w"]
    rpt_vwap_7w = reports_7w["VWAP_24_2.0+MTF_7w"]
    rpt_fib_7w_val = reports_7w["Fib_4h_7w"]

    # --- Requested: BB+Fib 50/50 (7w) ---
    logger.info("  --- 50%% BB + 50%% Fib (7w) — Requested Validation ---")
    compute_portfolio("BB+Fib_50_50_7w", [
        (rpt_bb_7w, 0.5, "BB"), (rpt_fib_7w_val, 0.5, "Fib"),
    ])
    logger.info("")

    # --- BB + RSI_MTF 50/50 (7w) ---
    logger.info("  --- 50%% BB + 50%% RSI_MTF (7w) ---")
    compute_portfolio("BB+RSI_50_50_7w", [
        (rpt_bb_7w, 0.5, "BB"), (rpt_rsi_7w, 0.5, "RSI"),
    ])
    logger.info("")

    # --- BB + VWAP_MTF 50/50 (7w) ---
    logger.info("  --- 50%% BB + 50%% VWAP_MTF (7w) ---")
    compute_portfolio("BB+VWAP_50_50_7w", [
        (rpt_bb_7w, 0.5, "BB"), (rpt_vwap_7w, 0.5, "VWAP"),
    ])
    logger.info("")

    # --- 3-way equal (7w) ---
    logger.info("  --- 33%% BB + 33%% RSI_MTF + 33%% VWAP_MTF (7w) ---")
    compute_portfolio("BB+RSI+VWAP_equal_7w", [
        (rpt_bb_7w, 0.333, "BB"), (rpt_rsi_7w, 0.333, "RSI"),
        (rpt_vwap_7w, 0.334, "VWAP"),
    ])
    logger.info("")

    # --- 3-way 40/30/30 (7w) ---
    logger.info("  --- 40%% BB + 30%% RSI_MTF + 30%% VWAP_MTF (7w) ---")
    compute_portfolio("BB+RSI+VWAP_40_30_30_7w", [
        (rpt_bb_7w, 0.40, "BB"), (rpt_rsi_7w, 0.30, "RSI"),
        (rpt_vwap_7w, 0.30, "VWAP"),
    ])
    logger.info("")

    # --- 3-way 20/40/40 (7w) ---
    logger.info("  --- 20%% BB + 40%% RSI_MTF + 40%% VWAP_MTF (7w) ---")
    compute_portfolio("BB+RSI+VWAP_20_40_40_7w", [
        (rpt_bb_7w, 0.20, "BB"), (rpt_rsi_7w, 0.40, "RSI"),
        (rpt_vwap_7w, 0.40, "VWAP"),
    ])
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   FINAL RANKING
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 10 — FINAL SINGLE STRATEGY RANKING (5w)")
    logger.info("=" * 72)
    logger.info("")

    single_results_5w.sort(
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    )

    logger.info(
        "  %-30s %8s %6s %7s %8s %6s %5s %5s",
        "Strategy", "OOS Ret", "WF Rob", "Tr(OOS)",
        "Full Ret", "MaxDD", "PF", "Shp",
    )
    logger.info("  " + "-" * 85)

    for name, rpt, r_full in single_results_5w:
        logger.info(
            "  %-30s %+7.2f%% %5.0f%% %6d   %+7.2f%% %5.1f%% %5.2f %5.2f",
            name, rpt.oos_total_return, rpt.robustness_score * 100,
            rpt.oos_total_trades,
            r_full.total_return, r_full.max_drawdown,
            r_full.profit_factor, r_full.sharpe_ratio,
        )

    logger.info("")

    # 7w summary table
    logger.info("=" * 72)
    logger.info("  7-WINDOW WF RESULTS")
    logger.info("=" * 72)
    logger.info("")
    logger.info(
        "  %-35s %8s %6s %7s",
        "Strategy", "OOS Ret", "WF Rob", "Tr(OOS)",
    )
    logger.info("  " + "-" * 60)

    for name, rpt in sorted(
        reports_7w.items(),
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    ):
        logger.info(
            "  %-35s %+7.2f%% %5.0f%% %6d",
            name, rpt.oos_total_return, rpt.robustness_score * 100,
            rpt.oos_total_trades,
        )

    logger.info("")

    # 9w summary table
    if reports_9w:
        logger.info("=" * 72)
        logger.info("  9-WINDOW WF RESULTS (Ultra Validation)")
        logger.info("=" * 72)
        logger.info("")
        logger.info(
            "  %-35s %8s %6s %7s",
            "Strategy", "OOS Ret", "WF Rob", "Tr(OOS)",
        )
        logger.info("  " + "-" * 60)

        for name, rpt in sorted(
            reports_9w.items(),
            key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
            reverse=True,
        ):
            logger.info(
                "  %-35s %+7.2f%% %5.0f%% %6d",
                name, rpt.oos_total_return, rpt.robustness_score * 100,
                rpt.oos_total_trades,
            )

        logger.info("")

    logger.info("=" * 72)
    logger.info("  Phase 10 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    run_phase10()

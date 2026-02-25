#!/usr/bin/env python3
"""Phase 7 — Portfolio Diversification with Mean Reversion Strategies.

Goals:
  1. WF-test RSI Mean Reversion (existing strategy, untested)
  2. Develop & WF-test VWAP Mean Reversion (new strategy)
  3. Build 3-strategy portfolio: BBSqueeze(1h) + Fib(4h) + MeanReversion
  4. Compare portfolio vs baseline

Rationale:
  - BBSqueeze = momentum/breakout (profits in trending markets)
  - Fib = pullback entry (profits in trending markets with retracements)
  - Mean Reversion = fade extremes (profits in ranging/choppy markets)
  → Low correlation → better risk-adjusted returns
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

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

logger = logging.getLogger("phase7")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase7.log", mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)


def load_data(timeframe: str) -> pd.DataFrame:
    """Load and add indicators to parquet data."""
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


def log_wf_result(name: str, report, engine: BacktestEngine, strategy_factory, df, htf_df=None):
    """Log Walk-Forward results in standard format."""
    for w in report.windows:
        oos = w.out_of_sample
        is_ = w.in_sample
        logger.info(
            "  W%d: IS  %+5.2f%% (WR %d%%, %d tr) | OOS  %+5.2f%% (WR %d%%, %d tr)",
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

    # Full backtest
    strat = strategy_factory()
    if htf_df is not None and hasattr(strat, "set_htf_data"):
        pass  # Engine handles HTF slicing
    full = engine.run(strat, df, htf_df=htf_df)
    logger.info(
        "  %s Full %+35.2f%% | Shp %5.2f | DD %5.1f%% | WR %5.1f%% | %4d tr | PF %5.2f",
        name, full.total_return, full.sharpe_ratio, full.max_drawdown,
        full.win_rate * 100, full.total_trades, full.profit_factor,
    )
    return full


def run_phase7():
    """Execute Phase 7 analysis."""
    logger.info("=" * 72)
    logger.info("  PHASE 7 — Portfolio Diversification: Mean Reversion Strategies")
    logger.info("=" * 72)
    logger.info("")

    # Load data
    df_1h = load_data("1h")
    df_4h = load_data("4h")

    logger.info("1h data: %d bars (%s ~ %s)", len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("4h data: %d bars (%s ~ %s)", len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    # ── Engine configs ──
    engine_conservative = BacktestEngine(max_hold_bars=48)
    engine_mr = BacktestEngine(max_hold_bars=36)  # MR exits faster

    # ══════════════════════════════════════════════════════════════════════
    #   PART 0: Baseline — BBSqueeze+MTF Conservative (1h), same as Phase 4-6
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 0: Phase 4-6 Baseline — BBSqueeze+MTF Conservative (1h)")
    logger.info("─" * 72)

    def bb_factory():
        base = BBSqueezeV2Strategy(
            squeeze_pctile=25, vol_mult=1.1, atr_sl_mult=3.0,
            rr_ratio=2.0, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    wf_bb = WalkForwardAnalyzer(n_windows=5, engine=engine_conservative)
    report_bb = wf_bb.run(bb_factory, df_1h, htf_df=df_4h)
    full_bb = log_wf_result("Baseline_1h", report_bb, engine_conservative, bb_factory, df_1h, df_4h)
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 1: RSI Mean Reversion — standalone WF (1h)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 1: RSI Mean Reversion — 1h standalone")
    logger.info("─" * 72)
    logger.info("")

    # Test multiple RSI parameter combinations
    rsi_configs = [
        {"name": "RSI_30_70", "oversold": 30, "overbought": 70, "sl": 2.0, "tp": 3.0, "cool": 6},
        {"name": "RSI_35_65", "oversold": 35, "overbought": 65, "sl": 2.0, "tp": 3.0, "cool": 6},
        {"name": "RSI_25_75", "oversold": 25, "overbought": 75, "sl": 2.0, "tp": 3.0, "cool": 6},
        {"name": "RSI_30_70_wide", "oversold": 30, "overbought": 70, "sl": 2.5, "tp": 2.5, "cool": 8},
        {"name": "RSI_30_70_tight", "oversold": 30, "overbought": 70, "sl": 1.5, "tp": 2.0, "cool": 4},
    ]

    best_rsi_report = None
    best_rsi_name = ""
    best_rsi_factory = None
    best_rsi_robustness = -1

    for cfg in rsi_configs:
        logger.info("  --- %s ---", cfg["name"])

        def make_rsi_factory(c=cfg):
            def factory():
                return RSIMeanReversionStrategy(
                    rsi_oversold=c["oversold"],
                    rsi_overbought=c["overbought"],
                    atr_sl_mult=c["sl"],
                    atr_tp_mult=c["tp"],
                    cooldown_bars=c["cool"],
                )
            return factory

        factory_fn = make_rsi_factory(cfg)
        wf_rsi = WalkForwardAnalyzer(n_windows=5, engine=engine_mr)
        report = wf_rsi.run(factory_fn, df_1h)
        log_wf_result(cfg["name"], report, engine_mr, factory_fn, df_1h)

        if report.robustness_score > best_rsi_robustness or (
            report.robustness_score == best_rsi_robustness
            and report.oos_total_return > (best_rsi_report.oos_total_return if best_rsi_report else -999)
        ):
            best_rsi_robustness = report.robustness_score
            best_rsi_report = report
            best_rsi_name = cfg["name"]
            best_rsi_factory = factory_fn

        logger.info("")

    logger.info("  Best RSI config: %s (Robustness: %d%%, OOS: %+.2f%%)",
                best_rsi_name, int(best_rsi_robustness * 100),
                best_rsi_report.oos_total_return if best_rsi_report else 0)
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 2: RSI Mean Reversion + MTF Filter (1h)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 2: RSI Mean Reversion + MTF (4h trend filter)")
    logger.info("─" * 72)
    logger.info("")

    # RSI MR is counter-trend by nature.
    # Test WITH MTF filter (only trade MR in trend direction)
    # and WITHOUT (pure counter-trend).
    # Also test REVERSE MTF: only trade MR AGAINST the 4h trend
    # (fade overextensions in counter-trend direction).

    for cfg in rsi_configs[:3]:  # Top 3 configs only
        logger.info("  --- %s + MTF ---", cfg["name"])

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
        log_wf_result(f"{cfg['name']}_MTF", report, engine_mr, factory_fn, df_1h, df_4h)
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 3: RSI Mean Reversion — 4h standalone
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 3: RSI Mean Reversion — 4h standalone")
    logger.info("─" * 72)
    logger.info("")

    engine_4h = BacktestEngine(max_hold_bars=24)  # 24 bars = 4 days on 4h

    for cfg in rsi_configs[:3]:
        logger.info("  --- %s (4h) ---", cfg["name"])

        def make_rsi_4h_factory(c=cfg):
            def factory():
                return RSIMeanReversionStrategy(
                    rsi_oversold=c["oversold"],
                    rsi_overbought=c["overbought"],
                    atr_sl_mult=c["sl"],
                    atr_tp_mult=c["tp"],
                    cooldown_bars=max(2, c["cool"] // 4),  # Scale cooldown for 4h
                )
            return factory

        factory_fn = make_rsi_4h_factory(cfg)
        wf = WalkForwardAnalyzer(n_windows=5, engine=engine_4h)
        report = wf.run(factory_fn, df_4h)
        log_wf_result(f"{cfg['name']}_4h", report, engine_4h, factory_fn, df_4h)
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 4: Fib Standalone (4h) — Portfolio component
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 4: Fib Standalone (4h) — for portfolio baseline")
    logger.info("─" * 72)
    logger.info("")

    def fib_factory():
        return FibonacciRetracementStrategy()

    wf_fib = WalkForwardAnalyzer(n_windows=5, engine=engine_4h)
    report_fib = wf_fib.run(fib_factory, df_4h)
    full_fib = log_wf_result("Fib_4h", report_fib, engine_4h, fib_factory, df_4h)
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 5: 2-Strategy Portfolios
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 5: 2-Strategy Portfolios")
    logger.info("─" * 72)
    logger.info("")

    def compute_portfolio_wf(name, reports_weights):
        """Compute portfolio OOS by combining per-window returns."""
        # All reports must have same n_windows
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
                label_parts.append(f"{label} OOS {oos_ret:+5.2f}%")

            portfolio_oos.append(weighted_return)
            total_trades += w_trades
            logger.info(
                "  W%d: %s → Port %+5.2f%%",
                w_idx + 1, " + ".join(label_parts), weighted_return,
            )

        # Compound OOS
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

    # BB + Fib (Phase 6 baseline portfolio)
    logger.info("  --- Portfolio: 50%% BB(1h) + 50%% Fib(4h) ---")
    compute_portfolio_wf(
        "BB+Fib",
        [(report_bb, 0.5, "BB"), (report_fib, 0.5, "Fib")],
    )
    logger.info("")

    # BB + RSI MR
    if best_rsi_report and best_rsi_report.total_windows > 0:
        logger.info("  --- Portfolio: 50%% BB(1h) + 50%% RSI_MR(1h) ---")
        compute_portfolio_wf(
            "BB+RSI_MR",
            [(report_bb, 0.5, "BB"), (best_rsi_report, 0.5, "RSI_MR")],
        )
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 6: 3-Strategy Portfolio — BB + Fib + RSI MR
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 6: 3-Strategy Portfolios")
    logger.info("─" * 72)
    logger.info("")

    if best_rsi_report and best_rsi_report.total_windows > 0:
        # Equal weight
        logger.info("  --- Portfolio: 33%% BB + 33%% Fib + 33%% RSI_MR ---")
        compute_portfolio_wf(
            "BB+Fib+RSI_equal",
            [
                (report_bb, 1/3, "BB"),
                (report_fib, 1/3, "Fib"),
                (best_rsi_report, 1/3, "RSI_MR"),
            ],
        )
        logger.info("")

        # Overweight BB (it has best track record)
        logger.info("  --- Portfolio: 50%% BB + 25%% Fib + 25%% RSI_MR ---")
        compute_portfolio_wf(
            "BB+Fib+RSI_BBheavy",
            [
                (report_bb, 0.50, "BB"),
                (report_fib, 0.25, "Fib"),
                (best_rsi_report, 0.25, "RSI_MR"),
            ],
        )
        logger.info("")

        # Overweight MR (if it adds diversification value)
        logger.info("  --- Portfolio: 40%% BB + 20%% Fib + 40%% RSI_MR ---")
        compute_portfolio_wf(
            "BB+Fib+RSI_MRheavy",
            [
                (report_bb, 0.40, "BB"),
                (report_fib, 0.20, "Fib"),
                (best_rsi_report, 0.40, "RSI_MR"),
            ],
        )
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   PART 7: Correlation Analysis
    # ══════════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 7: Strategy Correlation Analysis")
    logger.info("─" * 72)
    logger.info("")

    # Compare per-window returns to assess correlation
    if best_rsi_report and report_bb.total_windows == best_rsi_report.total_windows:
        bb_oos = [w.out_of_sample.total_return for w in report_bb.windows]
        fib_oos = [w.out_of_sample.total_return for w in report_fib.windows]
        rsi_oos = [w.out_of_sample.total_return for w in best_rsi_report.windows]

        logger.info("  Per-window OOS returns:")
        logger.info("  Window   BB(1h)    Fib(4h)   RSI_MR")
        for i in range(len(bb_oos)):
            fib_r = fib_oos[i] if i < len(fib_oos) else 0
            rsi_r = rsi_oos[i] if i < len(rsi_oos) else 0
            logger.info("  W%d     %+6.2f%%   %+6.2f%%   %+6.2f%%", i + 1, bb_oos[i], fib_r, rsi_r)

        # Simple correlation: count windows where both positive or both negative
        if len(bb_oos) >= 3 and len(rsi_oos) >= 3:
            same_sign_bb_rsi = sum(
                1 for a, b in zip(bb_oos, rsi_oos) if (a > 0) == (b > 0)
            )
            same_sign_bb_fib = sum(
                1 for a, b in zip(bb_oos, fib_oos) if (a > 0) == (b > 0)
            )
            n = min(len(bb_oos), len(rsi_oos))
            logger.info("")
            logger.info("  Directional agreement (same sign):")
            logger.info("    BB vs RSI_MR: %d/%d (%.0f%%) — lower = more diversification",
                        same_sign_bb_rsi, n, same_sign_bb_rsi / n * 100)
            logger.info("    BB vs Fib:    %d/%d (%.0f%%)",
                        same_sign_bb_fib, min(len(bb_oos), len(fib_oos)),
                        same_sign_bb_fib / min(len(bb_oos), len(fib_oos)) * 100)

    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #   FINAL RANKING
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 7 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Best single: BBSqueeze+MTF (1h) — OOS %+.2f%%, Robustness %d%%",
                report_bb.oos_total_return, int(report_bb.robustness_score * 100))
    logger.info("  Best RSI MR: %s — OOS %+.2f%%, Robustness %d%%",
                best_rsi_name,
                best_rsi_report.oos_total_return if best_rsi_report else 0,
                int(best_rsi_robustness * 100))
    logger.info("  Fib (4h):   OOS %+.2f%%, Robustness %d%%",
                report_fib.oos_total_return, int(report_fib.robustness_score * 100))
    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 7 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    run_phase7()

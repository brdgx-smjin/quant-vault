#!/usr/bin/env python3
"""Phase 11 — Stochastic Mean Reversion Strategy + Portfolio Diversification.

Phase 10 findings:
  - RSI_35_65+MTF (9w): 78% robustness, OOS +20.59% — BEST single
  - VWAP_24_2.0+MTF (9w): 67% robustness, OOS +11.55%
  - BB+RSI_MTF 50/50 (7w): 71% robustness, OOS +8.19% — BEST portfolio

Phase 11 goals:
  1. Test Stochastic Mean Reversion + MTF (different oscillator from RSI)
  2. Parameter grid search for Stochastic MR
  3. 5w/7w/9w WF validation
  4. Portfolio combos: RSI+Stoch, VWAP+Stoch, BB+RSI+Stoch
  5. Compare Stoch MR against RSI MR and VWAP MR
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pandas_ta as ta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.basic import BasicIndicators
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.strategy.bb_squeeze_v2 import BBSqueezeV2Strategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
from src.strategy.stoch_mean_reversion import StochMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase11")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase11.log", mode="w")
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


def add_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3) -> pd.DataFrame:
    """Add Stochastic Oscillator columns to DataFrame."""
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=k, d=d, smooth_k=smooth)
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)
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
        logger.info("  W%d: %s → Port %+.2f%%", w_idx + 1, " + ".join(label_parts), weighted_return)

    # Compounded OOS
    compounded = 1.0
    for r in portfolio_oos:
        compounded *= (1 + r / 100)
    oos_total = (compounded - 1) * 100
    profitable = sum(1 for r in portfolio_oos if r > 0)
    rob = profitable / n_windows if n_windows > 0 else 0
    logger.info(
        "  %s OOS: %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        name, oos_total, int(rob * 100), profitable, n_windows, total_trades,
    )
    return oos_total, rob, total_trades


def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 11 — Stochastic Mean Reversion + Portfolio")
    logger.info("=" * 72)
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_4h = load_data("4h")

    # Add stochastic indicators with different params
    for k, d, smooth in [(14, 3, 3), (10, 3, 3), (21, 5, 3)]:
        df_1h = add_stochastic(df_1h, k=k, d=d, smooth=smooth)

    logger.info("1h data: %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("4h data: %d bars (%s ~ %s)", len(df_4h),
                df_4h.index[0].date(), df_4h.index[-1].date())

    # Check stochastic columns
    stoch_cols = [c for c in df_1h.columns if "STOCH" in c]
    logger.info("Stochastic columns: %s", stoch_cols)
    logger.info("")

    engine = BacktestEngine(max_hold_bars=36)  # MR strategies use shorter hold

    # ─────────────────────────────────────────────────────────────
    #   PART 0: Stochastic MR Parameter Grid (5w)
    # ─────────────────────────────────────────────────────────────
    logger.info("─" * 72)
    logger.info("  PART 0: Stochastic MR Parameter Grid (5-window WF)")
    logger.info("─" * 72)
    logger.info("")

    configs = [
        # (name, stoch_k, stoch_d, stoch_smooth, oversold, overbought, sl, tp, cool)
        ("Stoch_14_20_80",  14, 3, 3, 20.0, 80.0, 2.0, 3.0, 6),
        ("Stoch_14_15_85",  14, 3, 3, 15.0, 85.0, 2.0, 3.0, 6),
        ("Stoch_14_25_75",  14, 3, 3, 25.0, 75.0, 2.0, 3.0, 6),
        ("Stoch_10_20_80",  10, 3, 3, 20.0, 80.0, 2.0, 3.0, 6),
        ("Stoch_21_20_80",  21, 5, 3, 20.0, 80.0, 2.0, 3.0, 6),
        ("Stoch_14_20_80_sl15", 14, 3, 3, 20.0, 80.0, 1.5, 2.5, 6),
        ("Stoch_14_20_80_cool4", 14, 3, 3, 20.0, 80.0, 2.0, 3.0, 4),
    ]

    grid_results = []

    for cfg_name, sk, sd, ss, os_val, ob_val, sl, tp, cool in configs:
        logger.info("  --- %s ---", cfg_name)

        def make_stoch(sk=sk, sd=sd, ss=ss, os_val=os_val, ob_val=ob_val,
                       sl=sl, tp=tp, cool=cool):
            return StochMeanReversionStrategy(
                stoch_k=sk, stoch_d=sd, stoch_smooth=ss,
                oversold=os_val, overbought=ob_val,
                atr_sl_mult=sl, atr_tp_mult=tp, cooldown_bars=cool,
            )

        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(make_stoch, df_1h)
        full = log_wf(cfg_name, report, engine, make_stoch, df_1h)
        logger.info("")

        grid_results.append((cfg_name, report, full))

    # Also test with MTF
    logger.info("  --- Stochastic + MTF variants ---")
    logger.info("")

    mtf_results = []
    for cfg_name, sk, sd, ss, os_val, ob_val, sl, tp, cool in configs:
        mtf_name = f"{cfg_name}+MTF"
        logger.info("  --- %s ---", mtf_name)

        def make_stoch_mtf(sk=sk, sd=sd, ss=ss, os_val=os_val, ob_val=ob_val,
                           sl=sl, tp=tp, cool=cool):
            base = StochMeanReversionStrategy(
                stoch_k=sk, stoch_d=sd, stoch_smooth=ss,
                oversold=os_val, overbought=ob_val,
                atr_sl_mult=sl, atr_tp_mult=tp, cooldown_bars=cool,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(make_stoch_mtf, df_1h, htf_df=df_4h)
        full = log_wf(mtf_name, report, engine, make_stoch_mtf, df_1h, htf_df=df_4h)
        logger.info("")

        mtf_results.append((mtf_name, report, full))

    # Find best standalone and +MTF config
    all_results = grid_results + mtf_results
    best = max(all_results, key=lambda x: (x[1].robustness_score, x[1].oos_total_return))
    logger.info("  Best Stochastic config: %s — OOS %+.2f%%, Robustness %d%%",
                best[0], best[1].oos_total_return, int(best[1].robustness_score * 100))
    logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   PART 1: Best Stochastic MR — 7w and 9w WF
    # ─────────────────────────────────────────────────────────────
    # Find best +MTF config for deeper validation
    best_mtf = max(mtf_results, key=lambda x: (x[1].robustness_score, x[1].oos_total_return))
    logger.info("─" * 72)
    logger.info("  PART 1: Extended WF — %s", best_mtf[0])
    logger.info("─" * 72)
    logger.info("")

    # Parse best config params from the name
    best_mtf_name = best_mtf[0]
    # Find the matching config
    best_cfg = None
    for cfg_name, sk, sd, ss, os_val, ob_val, sl, tp, cool in configs:
        if f"{cfg_name}+MTF" == best_mtf_name:
            best_cfg = (sk, sd, ss, os_val, ob_val, sl, tp, cool)
            break

    if best_cfg is None:
        # fallback to default
        best_cfg = (14, 3, 3, 20.0, 80.0, 2.0, 3.0, 6)

    sk, sd, ss, os_val, ob_val, sl_m, tp_m, cool = best_cfg

    def make_best_stoch_mtf():
        base = StochMeanReversionStrategy(
            stoch_k=sk, stoch_d=sd, stoch_smooth=ss,
            oversold=os_val, overbought=ob_val,
            atr_sl_mult=sl_m, atr_tp_mult=tp_m, cooldown_bars=cool,
        )
        return MultiTimeframeFilter(base)

    # Also define baseline factories for comparison
    def make_rsi_mtf():
        base = RSIMeanReversionStrategy(
            rsi_oversold=35, rsi_overbought=65,
            atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    def make_vwap_mtf():
        base = VWAPMeanReversionStrategy(
            vwap_period=24, band_mult=2.0, rsi_threshold=35.0,
            atr_sl_mult=2.0, cooldown_bars=4,
        )
        return MultiTimeframeFilter(base)

    def make_bb_mtf():
        base = BBSqueezeV2Strategy(
            squeeze_pctile=25, vol_mult=1.1, atr_sl_mult=3.0,
            rr_ratio=2.0, require_trend=False, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    # 7-window WF
    for n_win, label in [(7, "7w"), (9, "9w")]:
        logger.info("  --- %s (%s) ---", best_mtf_name.replace("+MTF", f"+MTF_{label}"), label)
        wf = WalkForwardAnalyzer(n_windows=n_win, engine=engine)
        report = wf.run(make_best_stoch_mtf, df_1h, htf_df=df_4h)
        full = log_wf(f"{best_mtf_name}_{label}",
                      report, engine, make_best_stoch_mtf, df_1h, htf_df=df_4h)
        logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   PART 2: Baselines for comparison (7w)
    # ─────────────────────────────────────────────────────────────
    logger.info("─" * 72)
    logger.info("  PART 2: Baselines for comparison (7-window WF)")
    logger.info("─" * 72)
    logger.info("")

    baseline_reports_7w = {}

    for name, factory, htf in [
        ("RSI_35_65+MTF_7w", make_rsi_mtf, df_4h),
        ("VWAP_24_2.0+MTF_7w", make_vwap_mtf, df_4h),
        ("BB+MTF_7w", make_bb_mtf, df_4h),
    ]:
        logger.info("  --- %s ---", name)
        wf = WalkForwardAnalyzer(n_windows=7, engine=engine)
        report = wf.run(factory, df_1h, htf_df=htf)
        full = log_wf(name, report, engine, factory, df_1h, htf_df=htf)
        baseline_reports_7w[name] = report
        logger.info("")

    # Run best stoch at 7w for portfolio
    logger.info("  --- %s_7w ---", best_mtf_name)
    wf = WalkForwardAnalyzer(n_windows=7, engine=engine)
    stoch_report_7w = wf.run(make_best_stoch_mtf, df_1h, htf_df=df_4h)
    log_wf(f"{best_mtf_name}_7w", stoch_report_7w, engine, make_best_stoch_mtf,
           df_1h, htf_df=df_4h)
    logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   PART 3: Portfolio Combinations (7w)
    # ─────────────────────────────────────────────────────────────
    logger.info("─" * 72)
    logger.info("  PART 3: Portfolio Combinations (7-window)")
    logger.info("─" * 72)
    logger.info("")

    rsi_7w = baseline_reports_7w.get("RSI_35_65+MTF_7w")
    vwap_7w = baseline_reports_7w.get("VWAP_24_2.0+MTF_7w")
    bb_7w = baseline_reports_7w.get("BB+MTF_7w")

    if rsi_7w and stoch_report_7w.total_windows > 0:
        logger.info("  --- 50%% RSI + 50%% Stoch (7w) ---")
        compute_portfolio("RSI+Stoch_50_50_7w", [
            (rsi_7w, 0.5, "RSI"),
            (stoch_report_7w, 0.5, "Stoch"),
        ])
        logger.info("")

    if vwap_7w and stoch_report_7w.total_windows > 0:
        logger.info("  --- 50%% VWAP + 50%% Stoch (7w) ---")
        compute_portfolio("VWAP+Stoch_50_50_7w", [
            (vwap_7w, 0.5, "VWAP"),
            (stoch_report_7w, 0.5, "Stoch"),
        ])
        logger.info("")

    if bb_7w and rsi_7w and stoch_report_7w.total_windows > 0:
        logger.info("  --- 33%% BB + 33%% RSI + 33%% Stoch (7w) ---")
        compute_portfolio("BB+RSI+Stoch_equal_7w", [
            (bb_7w, 0.33, "BB"),
            (rsi_7w, 0.33, "RSI"),
            (stoch_report_7w, 0.34, "Stoch"),
        ])
        logger.info("")

    if rsi_7w and vwap_7w and stoch_report_7w.total_windows > 0:
        logger.info("  --- 33%% RSI + 33%% VWAP + 33%% Stoch (7w) ---")
        compute_portfolio("RSI+VWAP+Stoch_equal_7w", [
            (rsi_7w, 0.33, "RSI"),
            (vwap_7w, 0.33, "VWAP"),
            (stoch_report_7w, 0.34, "Stoch"),
        ])
        logger.info("")

    if bb_7w and rsi_7w and vwap_7w and stoch_report_7w.total_windows > 0:
        logger.info("  --- 25%% BB + 25%% RSI + 25%% VWAP + 25%% Stoch (7w) ---")
        compute_portfolio("BB+RSI+VWAP+Stoch_equal_7w", [
            (bb_7w, 0.25, "BB"),
            (rsi_7w, 0.25, "RSI"),
            (vwap_7w, 0.25, "VWAP"),
            (stoch_report_7w, 0.25, "Stoch"),
        ])
        logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   SUMMARY
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  PHASE 11 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    # Rank all grid results
    logger.info("  Stochastic MR Grid Results (5w):")
    logger.info("  %-30s %8s %6s %6s %8s %6s %6s %5s",
                "Strategy", "OOS Ret", "WF Rob", "Trades", "Full Ret", "MaxDD", "PF", "Shp")
    logger.info("  " + "-" * 95)

    sorted_results = sorted(all_results,
                            key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
                            reverse=True)
    for name, report, full in sorted_results:
        logger.info(
            "  %-30s %+7.2f%% %5d%% %6d %+7.2f%% %5.1f%% %5.2f %5.2f",
            name, report.oos_total_return,
            int(report.robustness_score * 100), report.oos_total_trades,
            full.total_return, full.max_drawdown, full.profit_factor,
            full.sharpe_ratio,
        )

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 11 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

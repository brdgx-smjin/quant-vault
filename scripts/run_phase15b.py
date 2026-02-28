#!/usr/bin/env python3
"""Phase 15b — ROC Exhaustion + MFI Mean Reversion exploration.

Phase 15 findings:
  - Keltner Channel MR: 80% at 5w, degrades to 44% at 9w → FAILS
  - Same negative windows (W2, W6) as existing strategies → no diversification

Phase 15b goals:
  1. ROC Exhaustion MR: raw % change → mathematically different from RSI/CCI
  2. MFI MR: volume-weighted RSI → incorporates volume in signal generation
  3. If either meets 55%+ at 9w → test in portfolio
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
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.roc_exhaustion import ROCExhaustionStrategy
from src.strategy.mfi_mean_reversion import MFIMeanReversionStrategy
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase15b")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase15b.log", mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

for name in ["src.backtest.engine", "src.strategy.mtf_filter"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def load_data(timeframe: str) -> pd.DataFrame:
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    col = f"MFI_{period}"
    if col not in df.columns:
        df[col] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=period)
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    col = f"CCI_{period}"
    if col not in df.columns:
        df[col] = ta.cci(df["high"], df["low"], df["close"], length=period)
    return df


def log_wf(name: str, report, engine: BacktestEngine,
           strategy_factory, df: pd.DataFrame, htf_df=None) -> BacktestResult:
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
        logger.info("  W%d: %s → Port %+.2f%%",
                     w_idx + 1, " + ".join(label_parts), weighted_return)
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
    logger.info("  PHASE 15b — ROC Exhaustion + MFI Mean Reversion")
    logger.info("=" * 72)
    logger.info("")

    df_1h = load_data("1h")
    df_4h = load_data("4h")
    df_1h = add_mfi(df_1h, 14)
    df_1h = add_cci(df_1h, 20)

    logger.info("1h data: %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("")

    engine = BacktestEngine(max_hold_bars=48)

    # ═════════════════════════════════════════════════════════════
    #   PART 1: ROC Exhaustion parameter sweep (5w)
    # ═════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 1: ROC Exhaustion Parameter Sweep (5w)")
    logger.info("─" * 72)
    logger.info("")

    roc_results_5w = {}

    roc_configs = [
        # period, threshold, sl, tp, cool, label
        (12, 4.0, 2.0, 3.0, 6, "ROC_12_4.0"),
        (12, 5.0, 2.0, 3.0, 6, "ROC_12_5.0"),
        (12, 6.0, 2.0, 3.0, 6, "ROC_12_6.0"),
        (24, 5.0, 2.0, 3.0, 6, "ROC_24_5.0"),
        (24, 7.0, 2.0, 3.0, 6, "ROC_24_7.0"),
        (24, 8.0, 2.0, 3.0, 6, "ROC_24_8.0"),
        (6,  3.0, 2.0, 3.0, 6, "ROC_6_3.0"),
        (6,  4.0, 2.0, 3.0, 6, "ROC_6_4.0"),
    ]

    for period, threshold, sl, tp, cool, label in roc_configs:
        full_label = f"{label}+MTF_5w"
        logger.info("  --- %s ---", full_label)

        def factory(p=period, t=threshold, s=sl, tp_=tp, c=cool):
            base = ROCExhaustionStrategy(
                roc_period=p, threshold=t,
                atr_sl_mult=s, atr_tp_mult=tp_, cooldown_bars=c,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(factory, df_1h, htf_df=df_4h)
        full = log_wf(full_label, report, engine, factory, df_1h, htf_df=df_4h)
        roc_results_5w[full_label] = (
            report.oos_total_return, report.robustness_score,
            report.oos_total_trades, full.total_return,
            full.sharpe_ratio, full.max_drawdown, report,
        )
        logger.info("")

    logger.info("  ═══ ROC 5w Summary ═══")
    logger.info("  %-25s %8s %6s %6s %8s %5s %5s", "Config", "OOS Ret", "Rob",
                "Trades", "FullRet", "Sharpe", "DD")
    logger.info("  " + "-" * 70)
    for name, (oos, rob, tr, fret, shp, dd, _) in sorted(
            roc_results_5w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-25s %+7.2f%% %5d%% %6d %+7.2f%% %5.2f %5.1f%%%s",
                     name, oos, int(rob * 100), tr, fret, shp, dd, marker)

    # ═════════════════════════════════════════════════════════════
    #   PART 2: MFI Mean Reversion parameter sweep (5w)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─" * 72)
    logger.info("  PART 2: MFI Mean Reversion Parameter Sweep (5w)")
    logger.info("─" * 72)
    logger.info("")

    mfi_results_5w = {}

    mfi_configs = [
        # period, oversold, overbought, sl, tp, cool, label
        (14, 20, 80, 2.0, 3.0, 6, "MFI_14_20_80"),
        (14, 15, 85, 2.0, 3.0, 6, "MFI_14_15_85"),
        (14, 10, 90, 2.0, 3.0, 6, "MFI_14_10_90"),
        (20, 20, 80, 2.0, 3.0, 6, "MFI_20_20_80"),
        (20, 15, 85, 2.0, 3.0, 6, "MFI_20_15_85"),
        (10, 20, 80, 2.0, 3.0, 6, "MFI_10_20_80"),
        (10, 15, 85, 2.0, 3.0, 6, "MFI_10_15_85"),
    ]

    for period, oversold, overbought, sl, tp, cool, label in mfi_configs:
        full_label = f"{label}+MTF_5w"
        logger.info("  --- %s ---", full_label)

        # Add MFI column if needed
        mfi_col = f"MFI_{period}"
        if mfi_col not in df_1h.columns:
            df_1h[mfi_col] = ta.mfi(
                df_1h["high"], df_1h["low"], df_1h["close"], df_1h["volume"],
                length=period,
            )

        def factory(p=period, os_=oversold, ob=overbought, s=sl, tp_=tp, c=cool):
            base = MFIMeanReversionStrategy(
                mfi_period=p, oversold_level=os_, overbought_level=ob,
                atr_sl_mult=s, atr_tp_mult=tp_, cooldown_bars=c,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(factory, df_1h, htf_df=df_4h)
        full = log_wf(full_label, report, engine, factory, df_1h, htf_df=df_4h)
        mfi_results_5w[full_label] = (
            report.oos_total_return, report.robustness_score,
            report.oos_total_trades, full.total_return,
            full.sharpe_ratio, full.max_drawdown, report,
        )
        logger.info("")

    logger.info("  ═══ MFI 5w Summary ═══")
    logger.info("  %-25s %8s %6s %6s %8s %5s %5s", "Config", "OOS Ret", "Rob",
                "Trades", "FullRet", "Sharpe", "DD")
    logger.info("  " + "-" * 70)
    for name, (oos, rob, tr, fret, shp, dd, _) in sorted(
            mfi_results_5w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-25s %+7.2f%% %5d%% %6d %+7.2f%% %5.2f %5.1f%%%s",
                     name, oos, int(rob * 100), tr, fret, shp, dd, marker)

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Best variants at 7w and 9w
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─" * 72)
    logger.info("  PART 3: Best ROC + MFI at 7w and 9w")
    logger.info("─" * 72)
    logger.info("")

    # Pick top configs with >= 60% robustness at 5w
    promising = {}

    for name, vals in roc_results_5w.items():
        if vals[1] >= 0.6:
            promising[name] = vals

    for name, vals in mfi_results_5w.items():
        if vals[1] >= 0.6:
            promising[name] = vals

    if not promising:
        # Fall back to best from each
        best_roc = max(roc_results_5w.items(), key=lambda x: (x[1][1], x[1][0]))
        best_mfi = max(mfi_results_5w.items(), key=lambda x: (x[1][1], x[1][0]))
        promising[best_roc[0]] = best_roc[1]
        promising[best_mfi[0]] = best_mfi[1]
        logger.info("  No configs met 60%% threshold. Testing best from each:")
        logger.info("    ROC: %s (%.0f%%)", best_roc[0], best_roc[1][1] * 100)
        logger.info("    MFI: %s (%.0f%%)", best_mfi[0], best_mfi[1][1] * 100)
        logger.info("")

    results_7w = {}
    results_9w = {}
    best_report_9w = {}

    # Build factory from label
    all_configs = {}
    for p, t, s, tp_, c, label in roc_configs:
        all_configs[f"{label}+MTF_5w"] = ("roc", p, t, s, tp_, c)
    for p, os_, ob, s, tp_, c, label in mfi_configs:
        all_configs[f"{label}+MTF_5w"] = ("mfi", p, os_, ob, s, tp_, c)

    for name_5w in promising:
        if name_5w not in all_configs:
            continue

        cfg = all_configs[name_5w]
        base_label = name_5w.replace("+MTF_5w", "")

        if cfg[0] == "roc":
            _, p, t, s, tp_, c = cfg
            def factory(p_=p, t_=t, s_=s, tp__=tp_, c_=c):
                base = ROCExhaustionStrategy(
                    roc_period=p_, threshold=t_,
                    atr_sl_mult=s_, atr_tp_mult=tp__, cooldown_bars=c_,
                )
                return MultiTimeframeFilter(base)
        else:
            _, p, os_, ob, s, tp_, c = cfg
            mfi_col = f"MFI_{p}"
            if mfi_col not in df_1h.columns:
                df_1h[mfi_col] = ta.mfi(
                    df_1h["high"], df_1h["low"], df_1h["close"], df_1h["volume"],
                    length=p,
                )
            def factory(p_=p, os__=os_, ob_=ob, s_=s, tp__=tp_, c_=c):
                base = MFIMeanReversionStrategy(
                    mfi_period=p_, oversold_level=os__, overbought_level=ob_,
                    atr_sl_mult=s_, atr_tp_mult=tp__, cooldown_bars=c_,
                )
                return MultiTimeframeFilter(base)

        # 7w
        label_7w = f"{base_label}+MTF_7w"
        logger.info("  --- %s ---", label_7w)
        wf7 = WalkForwardAnalyzer(n_windows=7, engine=engine)
        report7 = wf7.run(factory, df_1h, htf_df=df_4h)
        full7 = log_wf(label_7w, report7, engine, factory, df_1h, htf_df=df_4h)
        results_7w[label_7w] = (
            report7.oos_total_return, report7.robustness_score,
            report7.oos_total_trades,
        )
        logger.info("")

        # 9w
        label_9w = f"{base_label}+MTF_9w"
        logger.info("  --- %s ---", label_9w)
        wf9 = WalkForwardAnalyzer(n_windows=9, engine=engine)
        report9 = wf9.run(factory, df_1h, htf_df=df_4h)
        full9 = log_wf(label_9w, report9, engine, factory, df_1h, htf_df=df_4h)
        results_9w[label_9w] = (
            report9.oos_total_return, report9.robustness_score,
            report9.oos_total_trades,
        )
        best_report_9w[label_9w] = report9
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Portfolio tests if any 9w variant is decent
    # ═════════════════════════════════════════════════════════════
    best_9w_rob = max((v[1] for v in results_9w.values()), default=0)

    if best_9w_rob >= 0.55:
        logger.info("")
        logger.info("─" * 72)
        logger.info("  PART 4: Portfolio Tests (9w)")
        logger.info("─" * 72)
        logger.info("")

        # Get best new strategy report
        best_new_name = max(results_9w.items(), key=lambda x: (x[1][1], x[1][0]))[0]
        best_new_report = best_report_9w[best_new_name]

        # Run RSI and DC at 9w
        def make_rsi():
            base = RSIMeanReversionStrategy(
                rsi_oversold=35, rsi_overbought=65,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
            )
            return MultiTimeframeFilter(base)

        def make_dc():
            base = DonchianTrendStrategy(
                entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
                vol_mult=0.8, cooldown_bars=6,
            )
            return MultiTimeframeFilter(base)

        logger.info("  --- RSI+MTF_9w (baseline) ---")
        wf_rsi = WalkForwardAnalyzer(n_windows=9, engine=engine)
        rsi_report = wf_rsi.run(make_rsi, df_1h, htf_df=df_4h)
        log_wf("RSI+MTF_9w", rsi_report, engine, make_rsi, df_1h, htf_df=df_4h)
        logger.info("")

        logger.info("  --- DC+MTF_9w (baseline) ---")
        wf_dc = WalkForwardAnalyzer(n_windows=9, engine=engine)
        dc_report = wf_dc.run(make_dc, df_1h, htf_df=df_4h)
        log_wf("DC+MTF_9w", dc_report, engine, make_dc, df_1h, htf_df=df_4h)
        logger.info("")

        short_name = best_new_name.replace("+MTF_9w", "").split("_")[0]
        portfolio_results = {}

        combos = [
            (f"{short_name}+RSI_50_50_9w",
             [(best_new_report, 0.5, short_name), (rsi_report, 0.5, "RSI")]),
            (f"{short_name}+DC_50_50_9w",
             [(best_new_report, 0.5, short_name), (dc_report, 0.5, "DC")]),
            (f"{short_name}+RSI+DC_equal_9w",
             [(best_new_report, 0.33, short_name), (rsi_report, 0.33, "RSI"),
              (dc_report, 0.34, "DC")]),
            ("RSI+DC_50_50_9w_baseline",
             [(rsi_report, 0.5, "RSI"), (dc_report, 0.5, "DC")]),
        ]

        for pname, components in combos:
            logger.info("  --- %s ---", pname)
            oos, rob, trades = compute_portfolio(pname, components)
            portfolio_results[pname] = (oos, rob, trades)
            logger.info("")
    else:
        logger.info("")
        logger.info("  !!! No ROC/MFI variant meets 55%% robustness at 9w.")
        logger.info("  !!! Skipping portfolio tests.")
        portfolio_results = {}

    # ═════════════════════════════════════════════════════════════
    #   SUMMARY
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 15b — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    logger.info("  ROC Exhaustion (5w):")
    logger.info("  %-25s %8s %6s %6s", "Config", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 50)
    for name, (oos, rob, tr, *_) in sorted(
            roc_results_5w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-25s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    logger.info("")
    logger.info("  MFI Mean Reversion (5w):")
    logger.info("  %-25s %8s %6s %6s", "Config", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 50)
    for name, (oos, rob, tr, *_) in sorted(
            mfi_results_5w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-25s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    if results_7w:
        logger.info("")
        logger.info("  Best variants (7w):")
        for name, (oos, rob, tr) in sorted(results_7w.items(),
                                             key=lambda x: (x[1][1], x[1][0]), reverse=True):
            marker = " ★" if rob >= 0.6 else ""
            logger.info("  %-30s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    if results_9w:
        logger.info("")
        logger.info("  Best variants (9w):")
        for name, (oos, rob, tr) in sorted(results_9w.items(),
                                             key=lambda x: (x[1][1], x[1][0]), reverse=True):
            marker = " ★" if rob >= 0.55 else ""
            logger.info("  %-30s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    if portfolio_results:
        logger.info("")
        logger.info("  Portfolio Tests (9w):")
        for name, (oos, rob, tr) in sorted(portfolio_results.items(),
                                             key=lambda x: (x[1][1], x[1][0]), reverse=True):
            marker = " ★" if rob >= 0.77 else ""
            logger.info("  %-35s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 15b complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

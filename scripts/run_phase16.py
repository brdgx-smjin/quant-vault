#!/usr/bin/env python3
"""Phase 16 — 15m Timeframe Cross-Validation.

Phase 15b findings (1h):
  - 77% robustness is the structural ceiling at 9 windows
  - W2 (Nov 20-Dec 2) and W6 (Jan 8-20) are negative across ALL strategies
  - New strategies (ROC, Keltner, MFI) reach 77% but never exceed it

Phase 16 hypothesis:
  15m data has 4x more bars → different WF window boundaries →
  potentially different structural negative periods → different ceiling.

Strategy approach:
  1. Test RSI_MR and Donchian on 15m with two parameter sets:
     a) Time-scaled (4x bar counts to match 1h duration)
     b) Native 15m (same bar counts, shorter duration)
  2. Use 4h MTF filter (proven on 1h, should work on 15m)
  3. Quick 5w screen → 7w/9w for promising configs
  4. Portfolio tests if individual strategies pass threshold

Data: 15m has ~35K bars (1 year). MTF: 4h (~2189 bars).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.basic import BasicIndicators
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase16")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase16.log", mode="w")
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
    logger.info("  PHASE 16 — 15m Timeframe Cross-Validation")
    logger.info("=" * 72)
    logger.info("")

    df_15m = load_data("15m")
    df_4h = load_data("4h")

    logger.info("15m data: %d bars (%s ~ %s)", len(df_15m),
                df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("4h data:  %d bars (%s ~ %s)", len(df_4h),
                df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 1: RSI Mean Reversion on 15m (5w quick screen)
    # ═════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 1: RSI Mean Reversion on 15m (5w)")
    logger.info("─" * 72)
    logger.info("")

    rsi_results_5w = {}

    # 15m parameter configs:
    # "ts" = time-scaled (4x cooldown, 4x max_hold to match 1h time duration)
    # "native" = same bar counts (captures shorter-duration patterns)
    rsi_configs = [
        # label, oversold, overbought, sl, tp, cool, max_hold
        ("RSI_35_65_ts",  35, 65, 2.0, 3.0, 24, 192),  # time-scaled (6h cool, 48h hold)
        ("RSI_35_65_mid", 35, 65, 2.0, 3.0, 12, 96),   # intermediate (3h cool, 24h hold)
        ("RSI_35_65_nat", 35, 65, 2.0, 3.0, 6,  48),   # native 15m (1.5h cool, 12h hold)
        ("RSI_30_70_ts",  30, 70, 2.0, 3.0, 24, 192),  # wider bands, time-scaled
        ("RSI_30_70_nat", 30, 70, 2.0, 3.0, 6,  48),   # wider bands, native
        ("RSI_35_65_ts2", 35, 65, 1.5, 2.5, 24, 192),  # tighter SL/TP, time-scaled
    ]

    for label, os_, ob, sl, tp, cool, mh in rsi_configs:
        full_label = f"{label}+MTF_5w"
        logger.info("  --- %s (cool=%d, hold=%d) ---", full_label, cool, mh)

        engine = BacktestEngine(max_hold_bars=mh, freq="15m")

        def factory(os_=os_, ob_=ob, sl_=sl, tp_=tp, cool_=cool):
            base = RSIMeanReversionStrategy(
                rsi_oversold=os_, rsi_overbought=ob_,
                atr_sl_mult=sl_, atr_tp_mult=tp_, cooldown_bars=cool_,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(factory, df_15m, htf_df=df_4h)
        full = log_wf(full_label, report, engine, factory, df_15m, htf_df=df_4h)
        rsi_results_5w[full_label] = (
            report.oos_total_return, report.robustness_score,
            report.oos_total_trades, full.total_return,
            full.sharpe_ratio, full.max_drawdown, report, cool, mh,
        )
        logger.info("")

    logger.info("  ═══ RSI 15m 5w Summary ═══")
    logger.info("  %-28s %8s %6s %6s %8s %5s %5s",
                "Config", "OOS Ret", "Rob", "Trades", "FullRet", "Sharpe", "DD")
    logger.info("  " + "-" * 70)
    for name, (oos, rob, tr, fret, shp, dd, *_) in sorted(
            rsi_results_5w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-28s %+7.2f%% %5d%% %6d %+7.2f%% %5.2f %5.1f%%%s",
                     name, oos, int(rob * 100), tr, fret, shp, dd, marker)

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Donchian Trend on 15m (5w quick screen)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─" * 72)
    logger.info("  PART 2: Donchian Trend on 15m (5w)")
    logger.info("─" * 72)
    logger.info("")

    dc_results_5w = {}

    dc_configs = [
        # label, entry_period, sl, rr, vol, cool, max_hold
        ("DC_96_ts",  96, 2.0, 2.0, 0.8, 24, 192),  # time-scaled (24h channel)
        ("DC_48_mid", 48, 2.0, 2.0, 0.8, 12, 96),   # intermediate (12h channel)
        ("DC_24_nat", 24, 2.0, 2.0, 0.8, 6,  48),   # native (6h channel)
        ("DC_72_ts",  72, 2.0, 2.0, 0.8, 24, 192),  # 18h channel, time-scaled
        ("DC_96_ts2", 96, 1.5, 2.5, 0.8, 24, 192),  # 24h channel, tighter SL wider RR
        ("DC_48_nat", 48, 2.0, 2.0, 0.8, 6,  48),   # 12h channel, native cooldown
    ]

    for label, period, sl, rr, vol, cool, mh in dc_configs:
        full_label = f"{label}+MTF_5w"
        logger.info("  --- %s (period=%d, cool=%d, hold=%d) ---",
                     full_label, period, cool, mh)

        engine = BacktestEngine(max_hold_bars=mh, freq="15m")

        def factory(p_=period, sl_=sl, rr_=rr, vol_=vol, cool_=cool):
            base = DonchianTrendStrategy(
                entry_period=p_, atr_sl_mult=sl_, rr_ratio=rr_,
                vol_mult=vol_, cooldown_bars=cool_,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(factory, df_15m, htf_df=df_4h)
        full = log_wf(full_label, report, engine, factory, df_15m, htf_df=df_4h)
        dc_results_5w[full_label] = (
            report.oos_total_return, report.robustness_score,
            report.oos_total_trades, full.total_return,
            full.sharpe_ratio, full.max_drawdown, report, cool, mh,
        )
        logger.info("")

    logger.info("  ═══ DC 15m 5w Summary ═══")
    logger.info("  %-28s %8s %6s %6s %8s %5s %5s",
                "Config", "OOS Ret", "Rob", "Trades", "FullRet", "Sharpe", "DD")
    logger.info("  " + "-" * 70)
    for name, (oos, rob, tr, fret, shp, dd, *_) in sorted(
            dc_results_5w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-28s %+7.2f%% %5d%% %6d %+7.2f%% %5.2f %5.1f%%%s",
                     name, oos, int(rob * 100), tr, fret, shp, dd, marker)

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Best RSI + DC at 7w and 9w
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─" * 72)
    logger.info("  PART 3: Best 15m configs at 7w and 9w")
    logger.info("─" * 72)
    logger.info("")

    # Pick configs with >= 60% robustness at 5w
    promising_rsi = {k: v for k, v in rsi_results_5w.items() if v[1] >= 0.6}
    promising_dc = {k: v for k, v in dc_results_5w.items() if v[1] >= 0.6}

    # Fallback: take best from each if none pass
    if not promising_rsi:
        best = max(rsi_results_5w.items(), key=lambda x: (x[1][1], x[1][0]))
        promising_rsi[best[0]] = best[1]
        logger.info("  No RSI config met 60%% at 5w. Taking best: %s (%.0f%%)",
                     best[0], best[1][1] * 100)
    if not promising_dc:
        best = max(dc_results_5w.items(), key=lambda x: (x[1][1], x[1][0]))
        promising_dc[best[0]] = best[1]
        logger.info("  No DC config met 60%% at 5w. Taking best: %s (%.0f%%)",
                     best[0], best[1][1] * 100)

    results_7w = {}
    results_9w = {}
    best_reports_9w = {}

    def parse_rsi_config(label_5w):
        """Reconstruct factory params from label."""
        for label, os_, ob, sl, tp, cool, mh in rsi_configs:
            if f"{label}+MTF_5w" == label_5w:
                return os_, ob, sl, tp, cool, mh
        return None

    def parse_dc_config(label_5w):
        """Reconstruct factory params from label."""
        for label, period, sl, rr, vol, cool, mh in dc_configs:
            if f"{label}+MTF_5w" == label_5w:
                return period, sl, rr, vol, cool, mh
        return None

    all_promising = {}
    for k, v in promising_rsi.items():
        all_promising[k] = ("rsi", v)
    for k, v in promising_dc.items():
        all_promising[k] = ("dc", v)

    for name_5w, (strat_type, vals) in all_promising.items():
        base_label = name_5w.replace("+MTF_5w", "")

        if strat_type == "rsi":
            params = parse_rsi_config(name_5w)
            if not params:
                continue
            os_, ob, sl, tp, cool, mh = params

            def factory(os_=os_, ob_=ob, sl_=sl, tp_=tp, cool_=cool):
                base = RSIMeanReversionStrategy(
                    rsi_oversold=os_, rsi_overbought=ob_,
                    atr_sl_mult=sl_, atr_tp_mult=tp_, cooldown_bars=cool_,
                )
                return MultiTimeframeFilter(base)
        else:
            params = parse_dc_config(name_5w)
            if not params:
                continue
            period, sl, rr, vol, cool, mh = params

            def factory(p_=period, sl_=sl, rr_=rr, vol_=vol, cool_=cool):
                base = DonchianTrendStrategy(
                    entry_period=p_, atr_sl_mult=sl_, rr_ratio=rr_,
                    vol_mult=vol_, cooldown_bars=cool_,
                )
                return MultiTimeframeFilter(base)

        engine = BacktestEngine(max_hold_bars=mh, freq="15m")

        # 7w
        label_7w = f"{base_label}+MTF_7w"
        logger.info("  --- %s ---", label_7w)
        wf7 = WalkForwardAnalyzer(n_windows=7, engine=engine)
        report7 = wf7.run(factory, df_15m, htf_df=df_4h)
        log_wf(label_7w, report7, engine, factory, df_15m, htf_df=df_4h)
        results_7w[label_7w] = (
            report7.oos_total_return, report7.robustness_score,
            report7.oos_total_trades,
        )
        logger.info("")

        # 9w
        label_9w = f"{base_label}+MTF_9w"
        logger.info("  --- %s ---", label_9w)
        wf9 = WalkForwardAnalyzer(n_windows=9, engine=engine)
        report9 = wf9.run(factory, df_15m, htf_df=df_4h)
        log_wf(label_9w, report9, engine, factory, df_15m, htf_df=df_4h)
        results_9w[label_9w] = (
            report9.oos_total_return, report9.robustness_score,
            report9.oos_total_trades,
        )
        best_reports_9w[label_9w] = (report9, strat_type, mh)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Portfolio tests at 9w
    # ═════════════════════════════════════════════════════════════
    # Pick best RSI and DC at 9w for portfolio tests
    best_rsi_9w = None
    best_dc_9w = None

    for name, (oos, rob, tr) in results_9w.items():
        info = best_reports_9w[name]
        if info[1] == "rsi" and (best_rsi_9w is None or (rob, oos) > (best_rsi_9w[1], best_rsi_9w[0])):
            best_rsi_9w = (oos, rob, tr, name, info[0])
        if info[1] == "dc" and (best_dc_9w is None or (rob, oos) > (best_dc_9w[1], best_dc_9w[0])):
            best_dc_9w = (oos, rob, tr, name, info[0])

    can_portfolio = (best_rsi_9w is not None and best_dc_9w is not None
                     and (best_rsi_9w[1] >= 0.55 or best_dc_9w[1] >= 0.55))

    portfolio_results = {}

    if can_portfolio:
        logger.info("")
        logger.info("─" * 72)
        logger.info("  PART 4: 15m Portfolio Tests (9w)")
        logger.info("─" * 72)
        logger.info("")

        rsi_report = best_rsi_9w[4]
        dc_report = best_dc_9w[4]
        rsi_name = best_rsi_9w[3].replace("+MTF_9w", "")
        dc_name = best_dc_9w[3].replace("+MTF_9w", "")

        combos = [
            (f"15m_{rsi_name}+{dc_name}_50_50_9w",
             [(rsi_report, 0.5, f"RSI({rsi_name})"), (dc_report, 0.5, f"DC({dc_name})")]),
        ]

        for pname, components in combos:
            logger.info("  --- %s ---", pname)
            oos, rob, trades = compute_portfolio(pname, components)
            portfolio_results[pname] = (oos, rob, trades)
            logger.info("")

        # Also: cross-timeframe portfolio — 15m best + 1h RSI+DC baseline
        # This requires running 1h strategies at 9w too
        logger.info("")
        logger.info("─" * 72)
        logger.info("  PART 5: Cross-Timeframe Portfolio (15m + 1h, 9w)")
        logger.info("─" * 72)
        logger.info("")
        logger.info("  Running 1h baselines at 9w for cross-TF comparison...")

        df_1h = load_data("1h")
        engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")

        def make_rsi_1h():
            base = RSIMeanReversionStrategy(
                rsi_oversold=35, rsi_overbought=65,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
            )
            return MultiTimeframeFilter(base)

        def make_dc_1h():
            base = DonchianTrendStrategy(
                entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
                vol_mult=0.8, cooldown_bars=6,
            )
            return MultiTimeframeFilter(base)

        logger.info("  --- RSI_1h+MTF_9w (baseline) ---")
        wf_rsi_1h = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
        rsi_1h_report = wf_rsi_1h.run(make_rsi_1h, df_1h, htf_df=df_4h)
        log_wf("RSI_1h+MTF_9w", rsi_1h_report, engine_1h, make_rsi_1h, df_1h, htf_df=df_4h)
        logger.info("")

        logger.info("  --- DC_1h+MTF_9w (baseline) ---")
        wf_dc_1h = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
        dc_1h_report = wf_dc_1h.run(make_dc_1h, df_1h, htf_df=df_4h)
        log_wf("DC_1h+MTF_9w", dc_1h_report, engine_1h, make_dc_1h, df_1h, htf_df=df_4h)
        logger.info("")

        # Cross-TF portfolios: 15m RSI+DC + 1h RSI+DC
        # Since different timeframes have different WF window dates,
        # we can't directly combine reports. Instead, just log for comparison.
        logger.info("  (Note: Cross-TF portfolio can't be computed directly because")
        logger.info("   15m and 1h have different WF window boundaries.)")
        logger.info("   Compare 9w robustness directly:")
        logger.info("   1h RSI+DC 50/50: known 77%% robustness, +20.27%% OOS")
        if portfolio_results:
            for pname, (oos, rob, tr) in portfolio_results.items():
                logger.info("   15m %s: %.0f%% robustness, %+.2f%% OOS",
                             pname, rob * 100, oos)
    else:
        logger.info("")
        logger.info("  !!! Neither RSI nor DC met 55%% robustness at 9w on 15m.")
        logger.info("  !!! Skipping portfolio tests.")

    # ═════════════════════════════════════════════════════════════
    #   SUMMARY
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 16 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    logger.info("  RSI Mean Reversion 15m (5w):")
    logger.info("  %-28s %8s %6s %6s", "Config", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 55)
    for name, (oos, rob, tr, *_) in sorted(
            rsi_results_5w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-28s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    logger.info("")
    logger.info("  Donchian Trend 15m (5w):")
    logger.info("  %-28s %8s %6s %6s", "Config", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 55)
    for name, (oos, rob, tr, *_) in sorted(
            dc_results_5w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-28s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    if results_7w:
        logger.info("")
        logger.info("  Promising configs at 7w:")
        for name, (oos, rob, tr) in sorted(results_7w.items(),
                                             key=lambda x: (x[1][1], x[1][0]), reverse=True):
            marker = " ★" if rob >= 0.6 else ""
            logger.info("  %-30s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    if results_9w:
        logger.info("")
        logger.info("  Promising configs at 9w:")
        for name, (oos, rob, tr) in sorted(results_9w.items(),
                                             key=lambda x: (x[1][1], x[1][0]), reverse=True):
            marker = " ★" if rob >= 0.55 else ""
            logger.info("  %-30s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    if portfolio_results:
        logger.info("")
        logger.info("  15m Portfolio Tests (9w):")
        for name, (oos, rob, tr) in sorted(portfolio_results.items(),
                                             key=lambda x: (x[1][1], x[1][0]), reverse=True):
            marker = " ★" if rob >= 0.77 else ""
            logger.info("  %-40s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    logger.info("")
    logger.info("  Reference (1h, Phase 14): RSI+DC 50/50 = 77%% rob, +20.27%% OOS")
    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 16 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""5m Timeframe Backtest — VWAP + Donchian Channel Grid Search.

Objective:
  - Test existing VWAP and DC strategies on 5m data (51k+ bars, 180 days)
  - MTF filter uses 1h EMA_20 vs EMA_50 (instead of 4h for 1h data)
  - 7w WF primary validation, 5w/9w secondary
  - Portfolio VWAP+DC 50/50
  - Compare vs 1h baseline

Engine settings for 5m:
  - freq="5min" for correct Sharpe annualization
  - max_hold_bars=192 (16 hours at 5m)
  - commission=0.0004, slippage=0.0001

NOTE: Full backtests skipped on 5m (51k bars → O(n^2) too slow).
      WF OOS results are the primary validation metric anyway.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.basic import BasicIndicators
from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("backtest_5m")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "backtest_5m.log", mode="w")
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


def log_wf(name: str, report) -> None:
    """Log WF results."""
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
        logger.info("  W%d: %s → Port %+.2f%%",
                     w_idx + 1, " + ".join(label_parts), weighted_return)

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
    logger.info("  5m TIMEFRAME BACKTEST — VWAP + Donchian Channel Grid")
    logger.info("=" * 72)
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_5m = load_data("5m")
    df_1h = load_data("1h")   # HTF for MTF filter (replaces 4h in 1h setup)

    logger.info("5m data: %d bars (%s ~ %s)", len(df_5m),
                df_5m.index[0].date(), df_5m.index[-1].date())
    logger.info("1h data: %d bars (%s ~ %s) [HTF for MTF filter]", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("")

    # 5m engine: freq="5min", max_hold=192 bars (16h)
    engine = BacktestEngine(
        freq="5min",
        max_hold_bars=192,
        commission=0.0004,
        slippage=0.0001,
    )

    # ─────────────────────────────────────────────────────────────
    #   PART 0: VWAP Grid (7-window WF, MTF=1h)
    # ─────────────────────────────────────────────────────────────
    logger.info("─" * 72)
    logger.info("  PART 0: VWAP Mean Reversion Parameter Grid (7-window WF)")
    logger.info("─" * 72)
    logger.info("")

    vwap_configs = [
        # (name, vwap_period, band_mult, rsi_thresh, atr_sl_mult, cooldown)
        ("VWAP_36_1.5_35_c6",  36, 1.5, 35.0, 2.0, 6),
        ("VWAP_36_2.0_35_c6",  36, 2.0, 35.0, 2.0, 6),
        ("VWAP_48_1.5_35_c6",  48, 1.5, 35.0, 2.0, 6),
        ("VWAP_48_2.0_35_c6",  48, 2.0, 35.0, 2.0, 6),
        ("VWAP_48_2.0_40_c6",  48, 2.0, 40.0, 2.0, 6),
        ("VWAP_72_1.5_35_c6",  72, 1.5, 35.0, 2.0, 6),
        ("VWAP_72_2.0_35_c6",  72, 2.0, 35.0, 2.0, 6),
        ("VWAP_48_2.0_35_c12", 48, 2.0, 35.0, 2.0, 12),
    ]

    vwap_results = []

    for cfg_name, period, band, rsi, sl, cool in vwap_configs:
        mtf_name = f"{cfg_name}+MTF"
        logger.info("  --- %s ---", mtf_name)

        def make_vwap_mtf(period=period, band=band, rsi=rsi, sl=sl, cool=cool):
            base = VWAPMeanReversionStrategy(
                vwap_period=period, band_mult=band, rsi_threshold=rsi,
                atr_sl_mult=sl, cooldown_bars=cool,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=7, engine=engine)
        report = wf.run(make_vwap_mtf, df_5m, htf_df=df_1h)
        log_wf(mtf_name, report)
        logger.info("")

        vwap_results.append((mtf_name, report, make_vwap_mtf))

    # Sort and find best VWAP configs
    vwap_sorted = sorted(vwap_results,
                         key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
                         reverse=True)
    best_vwap = vwap_sorted[0]
    logger.info("  Best VWAP: %s — OOS %+.2f%%, Robustness %d%%",
                best_vwap[0], best_vwap[1].oos_total_return,
                int(best_vwap[1].robustness_score * 100))
    logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   PART 1: DC Grid (7-window WF, MTF=1h)
    # ─────────────────────────────────────────────────────────────
    logger.info("─" * 72)
    logger.info("  PART 1: Donchian Channel Parameter Grid (7-window WF)")
    logger.info("─" * 72)
    logger.info("")

    dc_configs = [
        # (name, entry_period, atr_sl_mult, rr_ratio, vol_mult, cooldown)
        ("DC_36_1.5_2.0_c6",  36, 1.5, 2.0, 0.8, 6),
        ("DC_36_2.0_2.0_c6",  36, 2.0, 2.0, 0.8, 6),
        ("DC_48_1.5_2.0_c6",  48, 1.5, 2.0, 0.8, 6),
        ("DC_48_2.0_2.0_c6",  48, 2.0, 2.0, 0.8, 6),
        ("DC_48_2.0_3.0_c6",  48, 2.0, 3.0, 0.8, 6),
        ("DC_72_1.5_2.0_c6",  72, 1.5, 2.0, 0.8, 6),
        ("DC_72_2.0_2.0_c6",  72, 2.0, 2.0, 0.8, 6),
        ("DC_48_2.0_2.0_c12", 48, 2.0, 2.0, 0.8, 12),
    ]

    dc_results = []

    for cfg_name, period, sl, rr, vol, cool in dc_configs:
        mtf_name = f"{cfg_name}+MTF"
        logger.info("  --- %s ---", mtf_name)

        def make_dc_mtf(period=period, sl=sl, rr=rr, vol=vol, cool=cool):
            base = DonchianTrendStrategy(
                entry_period=period, atr_sl_mult=sl, rr_ratio=rr,
                vol_mult=vol, cooldown_bars=cool,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=7, engine=engine)
        report = wf.run(make_dc_mtf, df_5m, htf_df=df_1h)
        log_wf(mtf_name, report)
        logger.info("")

        dc_results.append((mtf_name, report, make_dc_mtf))

    # Sort and find best DC configs
    dc_sorted = sorted(dc_results,
                       key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
                       reverse=True)
    best_dc = dc_sorted[0]
    logger.info("  Best DC: %s — OOS %+.2f%%, Robustness %d%%",
                best_dc[0], best_dc[1].oos_total_return,
                int(best_dc[1].robustness_score * 100))
    logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   PART 2: Top 3 — 5w and 9w Extended Validation
    # ─────────────────────────────────────────────────────────────
    logger.info("─" * 72)
    logger.info("  PART 2: Extended WF — Top 3 VWAP + Top 3 DC (5w/9w)")
    logger.info("─" * 72)
    logger.info("")

    top_vwap = vwap_sorted[:3]
    top_dc = dc_sorted[:3]

    extended_reports = {}

    for candidates, label in [(top_vwap, "VWAP"), (top_dc, "DC")]:
        for cfg_name, report_7w, factory in candidates:
            for n_win, win_label in [(5, "5w"), (9, "9w")]:
                ext_name = f"{cfg_name}_{win_label}"
                logger.info("  --- %s ---", ext_name)
                wf = WalkForwardAnalyzer(n_windows=n_win, engine=engine)
                report = wf.run(factory, df_5m, htf_df=df_1h)
                log_wf(ext_name, report)
                extended_reports[ext_name] = report
                logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   PART 3: Portfolio Combinations (7w)
    # ─────────────────────────────────────────────────────────────
    logger.info("─" * 72)
    logger.info("  PART 3: Portfolio Combinations (7-window)")
    logger.info("─" * 72)
    logger.info("")

    best_vwap_report = best_vwap[1]
    best_dc_report = best_dc[1]

    # VWAP+DC 50/50
    logger.info("  --- VWAP+DC 50/50 (7w) ---")
    logger.info("  Using: %s + %s", best_vwap[0], best_dc[0])
    port_oos, port_rob, port_trades = compute_portfolio("VWAP+DC_50_50_7w", [
        (best_vwap_report, 0.5, "VWAP"), (best_dc_report, 0.5, "DC"),
    ])
    logger.info("")

    # VWAP+DC 60/40
    logger.info("  --- VWAP+DC 60/40 (7w) ---")
    compute_portfolio("VWAP+DC_60_40_7w", [
        (best_vwap_report, 0.6, "VWAP"), (best_dc_report, 0.4, "DC"),
    ])
    logger.info("")

    # VWAP+DC 40/60
    logger.info("  --- VWAP+DC 40/60 (7w) ---")
    compute_portfolio("VWAP+DC_40_60_7w", [
        (best_vwap_report, 0.4, "VWAP"), (best_dc_report, 0.6, "DC"),
    ])
    logger.info("")

    # Also try 2nd and 3rd best combos
    if len(vwap_sorted) >= 2 and len(dc_sorted) >= 2:
        logger.info("  --- 2nd best VWAP + 2nd best DC 50/50 (7w) ---")
        logger.info("  Using: %s + %s", vwap_sorted[1][0], dc_sorted[1][0])
        compute_portfolio("VWAP2+DC2_50_50_7w", [
            (vwap_sorted[1][1], 0.5, "VWAP2"), (dc_sorted[1][1], 0.5, "DC2"),
        ])
        logger.info("")

    # Best VWAP + Best DC portfolio at 9w (if we have 9w reports)
    best_vwap_9w_key = f"{best_vwap[0]}_9w"
    best_dc_9w_key = f"{best_dc[0]}_9w"
    if best_vwap_9w_key in extended_reports and best_dc_9w_key in extended_reports:
        logger.info("  --- VWAP+DC 50/50 (9w) ---")
        logger.info("  Using: %s + %s", best_vwap[0], best_dc[0])
        compute_portfolio("VWAP+DC_50_50_9w", [
            (extended_reports[best_vwap_9w_key], 0.5, "VWAP"),
            (extended_reports[best_dc_9w_key], 0.5, "DC"),
        ])
        logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   PART 4: 1h Baseline Comparison
    # ─────────────────────────────────────────────────────────────
    logger.info("─" * 72)
    logger.info("  PART 4: 1h Baseline Comparison (7-window)")
    logger.info("─" * 72)
    logger.info("")

    # 1h engine for baseline
    engine_1h = BacktestEngine(freq="1h", max_hold_bars=48)
    df_4h = load_data("4h")

    logger.info("4h data: %d bars (%s ~ %s) [HTF for 1h baseline]", len(df_4h),
                df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    baseline_reports = {}

    def make_vwap_1h_mtf():
        base = VWAPMeanReversionStrategy(
            vwap_period=24, band_mult=2.0, rsi_threshold=35.0,
            atr_sl_mult=2.0, cooldown_bars=4,
        )
        return MultiTimeframeFilter(base)

    def make_dc_1h_mtf():
        base = DonchianTrendStrategy(
            entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
            vol_mult=0.8, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    for bname, factory, htf in [
        ("VWAP_1h+MTF_7w", make_vwap_1h_mtf, df_4h),
        ("DC_1h+MTF_7w", make_dc_1h_mtf, df_4h),
    ]:
        logger.info("  --- %s ---", bname)
        wf = WalkForwardAnalyzer(n_windows=7, engine=engine_1h)
        report = wf.run(factory, df_1h, htf_df=htf)
        log_wf(bname, report)
        baseline_reports[bname] = report
        logger.info("")

    # 1h portfolio baseline
    vwap_1h_rep = baseline_reports.get("VWAP_1h+MTF_7w")
    dc_1h_rep = baseline_reports.get("DC_1h+MTF_7w")
    if vwap_1h_rep and dc_1h_rep:
        logger.info("  --- VWAP+DC 50/50 1h Baseline (7w) ---")
        compute_portfolio("VWAP+DC_1h_50_50_7w", [
            (vwap_1h_rep, 0.5, "VWAP_1h"), (dc_1h_rep, 0.5, "DC_1h"),
        ])
        logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   SUMMARY
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  5m BACKTEST — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    # VWAP grid ranking
    logger.info("  VWAP Grid Results (7w):")
    logger.info("  %-32s %8s %6s %6s",
                "Strategy", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 60)
    for rname, report, _ in vwap_sorted:
        logger.info(
            "  %-32s %+7.2f%% %5d%% %6d",
            rname, report.oos_total_return,
            int(report.robustness_score * 100), report.oos_total_trades,
        )
    logger.info("")

    # DC grid ranking
    logger.info("  DC Grid Results (7w):")
    logger.info("  %-32s %8s %6s %6s",
                "Strategy", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 60)
    for rname, report, _ in dc_sorted:
        logger.info(
            "  %-32s %+7.2f%% %5d%% %6d",
            rname, report.oos_total_return,
            int(report.robustness_score * 100), report.oos_total_trades,
        )
    logger.info("")

    # Extended validation summary
    logger.info("  Extended Validation (5w/9w):")
    logger.info("  %-38s %8s %6s %6s",
                "Strategy", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 65)
    for ext_name in sorted(extended_reports.keys()):
        rep = extended_reports[ext_name]
        logger.info(
            "  %-38s %+7.2f%% %5d%% %6d",
            ext_name, rep.oos_total_return,
            int(rep.robustness_score * 100), rep.oos_total_trades,
        )
    logger.info("")

    # 1h Baseline comparison
    logger.info("  1h Baseline Comparison (7w):")
    logger.info("  %-32s %8s %6s %6s",
                "Strategy", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 60)
    for bname, report in baseline_reports.items():
        logger.info(
            "  %-32s %+7.2f%% %5d%% %6d",
            bname, report.oos_total_return,
            int(report.robustness_score * 100), report.oos_total_trades,
        )
    logger.info("")

    # Trade frequency comparison
    logger.info("  Trade Frequency Comparison:")
    best_5m_vwap_trades = best_vwap[1].oos_total_trades
    best_5m_dc_trades = best_dc[1].oos_total_trades
    if vwap_1h_rep and dc_1h_rep:
        vwap_1h_trades = vwap_1h_rep.oos_total_trades
        dc_1h_trades = dc_1h_rep.oos_total_trades
        if vwap_1h_trades > 0:
            logger.info("  VWAP: 5m=%d vs 1h=%d trades → %.1fx more",
                        best_5m_vwap_trades, vwap_1h_trades,
                        best_5m_vwap_trades / vwap_1h_trades)
        if dc_1h_trades > 0:
            logger.info("  DC:   5m=%d vs 1h=%d trades → %.1fx more",
                        best_5m_dc_trades, dc_1h_trades,
                        best_5m_dc_trades / dc_1h_trades)
    logger.info("")

    logger.info("=" * 72)
    logger.info("  5m Backtest complete. Review logs/backtest_5m.log")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

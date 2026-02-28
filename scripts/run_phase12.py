#!/usr/bin/env python3
"""Phase 12 — Donchian Channel Trend Following + Portfolio Diversification.

Phase 10/11 findings:
  - RSI_35_65+MTF (9w): 78% robustness, OOS +20.59% — BEST single
  - VWAP_24_2.0+MTF (9w): 67% robustness, OOS +11.55%
  - BB+RSI_MTF 50/50 (7w): 71% robustness, OOS +8.19% — BEST portfolio
  - Stochastic MR: FAILED (degrades at 7w/9w, Full always negative)
  - MACD Momentum: FAILED (OOS -3% to -49%)

Phase 12 goals:
  1. Test Donchian Channel Trend Following (pure trend system)
     - Fundamentally different from existing strategies
     - Low correlation with mean-reversion → good diversifier
  2. Parameter grid: entry_period (20, 24, 48), SL, RR
  3. 5w/7w/9w WF validation
  4. Portfolio combos with RSI+MTF, VWAP+MTF, BB+MTF
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
from src.strategy.bb_squeeze_breakout import BBSqueezeBreakoutStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase12")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase12.log", mode="w")
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
    logger.info("  PHASE 12 — Donchian Channel Trend Following + Portfolio")
    logger.info("=" * 72)
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_4h = load_data("4h")

    logger.info("1h data: %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("4h data: %d bars (%s ~ %s)", len(df_4h),
                df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine = BacktestEngine(max_hold_bars=48)  # Trend following needs longer hold

    # ─────────────────────────────────────────────────────────────
    #   PART 0: Donchian Parameter Grid (5-window WF)
    # ─────────────────────────────────────────────────────────────
    logger.info("─" * 72)
    logger.info("  PART 0: Donchian Channel Parameter Grid (5-window WF)")
    logger.info("─" * 72)
    logger.info("")

    configs = [
        # (name, entry_period, atr_sl_mult, rr_ratio, vol_mult, cooldown)
        ("DC_20_2.0_2.0",    20, 2.0, 2.0, 0.8, 6),
        ("DC_24_2.0_2.0",    24, 2.0, 2.0, 0.8, 6),
        ("DC_48_2.0_2.0",    48, 2.0, 2.0, 0.8, 6),
        ("DC_24_3.0_2.0",    24, 3.0, 2.0, 0.8, 6),
        ("DC_24_2.0_3.0",    24, 2.0, 3.0, 0.8, 6),
        ("DC_24_3.0_3.0",    24, 3.0, 3.0, 0.8, 6),
        ("DC_20_2.0_2.0_nv", 20, 2.0, 2.0, 0.0, 6),  # no volume filter
        ("DC_48_3.0_2.0",    48, 3.0, 2.0, 0.8, 6),
        ("DC_24_2.0_2.0_c4", 24, 2.0, 2.0, 0.8, 4),
    ]

    grid_results = []

    for cfg_name, period, sl, rr, vol, cool in configs:
        logger.info("  --- %s ---", cfg_name)

        def make_dc(period=period, sl=sl, rr=rr, vol=vol, cool=cool):
            return DonchianTrendStrategy(
                entry_period=period, atr_sl_mult=sl, rr_ratio=rr,
                vol_mult=vol, cooldown_bars=cool,
            )

        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(make_dc, df_1h)
        full = log_wf(cfg_name, report, engine, make_dc, df_1h)
        logger.info("")

        grid_results.append((cfg_name, report, full))

    # Also test with MTF
    logger.info("  --- Donchian + MTF variants ---")
    logger.info("")

    mtf_results = []
    for cfg_name, period, sl, rr, vol, cool in configs:
        mtf_name = f"{cfg_name}+MTF"
        logger.info("  --- %s ---", mtf_name)

        def make_dc_mtf(period=period, sl=sl, rr=rr, vol=vol, cool=cool):
            base = DonchianTrendStrategy(
                entry_period=period, atr_sl_mult=sl, rr_ratio=rr,
                vol_mult=vol, cooldown_bars=cool,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(make_dc_mtf, df_1h, htf_df=df_4h)
        full = log_wf(mtf_name, report, engine, make_dc_mtf, df_1h, htf_df=df_4h)
        logger.info("")

        mtf_results.append((mtf_name, report, full))

    # Find best configs
    all_results = grid_results + mtf_results
    best = max(all_results, key=lambda x: (x[1].robustness_score, x[1].oos_total_return))
    logger.info("  Best Donchian config: %s — OOS %+.2f%%, Robustness %d%%",
                best[0], best[1].oos_total_return, int(best[1].robustness_score * 100))
    logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   PART 1: Best Donchian — 7w and 9w WF
    # ─────────────────────────────────────────────────────────────
    # Find best standalone and best +MTF for extended validation
    best_standalone = max(grid_results, key=lambda x: (x[1].robustness_score, x[1].oos_total_return))
    best_mtf = max(mtf_results, key=lambda x: (x[1].robustness_score, x[1].oos_total_return))

    logger.info("─" * 72)
    logger.info("  PART 1: Extended WF — Best Configs")
    logger.info("─" * 72)
    logger.info("")

    # Parse best standalone config
    best_s_name = best_standalone[0]
    best_s_cfg = None
    for cfg_name, period, sl, rr, vol, cool in configs:
        if cfg_name == best_s_name:
            best_s_cfg = (period, sl, rr, vol, cool)
            break

    # Parse best MTF config
    best_m_name = best_mtf[0]
    best_m_cfg = None
    for cfg_name, period, sl, rr, vol, cool in configs:
        if f"{cfg_name}+MTF" == best_m_name:
            best_m_cfg = (period, sl, rr, vol, cool)
            break

    if best_s_cfg is None:
        best_s_cfg = (24, 2.0, 2.0, 0.8, 6)
    if best_m_cfg is None:
        best_m_cfg = (24, 2.0, 2.0, 0.8, 6)

    def make_best_standalone():
        p, s, r, v, c = best_s_cfg
        return DonchianTrendStrategy(
            entry_period=p, atr_sl_mult=s, rr_ratio=r,
            vol_mult=v, cooldown_bars=c,
        )

    def make_best_mtf():
        p, s, r, v, c = best_m_cfg
        base = DonchianTrendStrategy(
            entry_period=p, atr_sl_mult=s, rr_ratio=r,
            vol_mult=v, cooldown_bars=c,
        )
        return MultiTimeframeFilter(base)

    extended_reports = {}

    for n_win, label in [(7, "7w"), (9, "9w")]:
        # Standalone
        ext_name = f"{best_s_name}_{label}"
        logger.info("  --- %s ---", ext_name)
        wf = WalkForwardAnalyzer(n_windows=n_win, engine=engine)
        report = wf.run(make_best_standalone, df_1h)
        log_wf(ext_name, report, engine, make_best_standalone, df_1h)
        extended_reports[ext_name] = report
        logger.info("")

        # MTF
        ext_name_mtf = f"{best_m_name}_{label}"
        logger.info("  --- %s ---", ext_name_mtf)
        wf = WalkForwardAnalyzer(n_windows=n_win, engine=engine)
        report = wf.run(make_best_mtf, df_1h, htf_df=df_4h)
        log_wf(ext_name_mtf, report, engine, make_best_mtf, df_1h, htf_df=df_4h)
        extended_reports[ext_name_mtf] = report
        logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   PART 2: Baselines for comparison (7w)
    # ─────────────────────────────────────────────────────────────
    logger.info("─" * 72)
    logger.info("  PART 2: Baselines for comparison (7-window WF)")
    logger.info("─" * 72)
    logger.info("")

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
        base = BBSqueezeBreakoutStrategy(
            squeeze_pctile=25, vol_mult=1.1, atr_sl_mult=3.0,
            rr_ratio=2.0, require_trend=False, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    baseline_reports_7w = {}
    for bname, factory, htf in [
        ("RSI_35_65+MTF_7w", make_rsi_mtf, df_4h),
        ("VWAP_24_2.0+MTF_7w", make_vwap_mtf, df_4h),
        ("BB+MTF_7w", make_bb_mtf, df_4h),
    ]:
        logger.info("  --- %s ---", bname)
        wf = WalkForwardAnalyzer(n_windows=7, engine=engine)
        report = wf.run(factory, df_1h, htf_df=htf)
        full = log_wf(bname, report, engine, factory, df_1h, htf_df=htf)
        baseline_reports_7w[bname] = report
        logger.info("")

    # Run best Donchian+MTF at 7w for portfolio
    logger.info("  --- %s_7w ---", best_m_name)
    wf = WalkForwardAnalyzer(n_windows=7, engine=engine)
    dc_report_7w = wf.run(make_best_mtf, df_1h, htf_df=df_4h)
    log_wf(f"{best_m_name}_7w", dc_report_7w, engine, make_best_mtf,
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

    if dc_report_7w.total_windows > 0:
        if rsi_7w:
            logger.info("  --- 50%% RSI + 50%% DC (7w) ---")
            compute_portfolio("RSI+DC_50_50_7w", [
                (rsi_7w, 0.5, "RSI"), (dc_report_7w, 0.5, "DC"),
            ])
            logger.info("")

        if vwap_7w:
            logger.info("  --- 50%% VWAP + 50%% DC (7w) ---")
            compute_portfolio("VWAP+DC_50_50_7w", [
                (vwap_7w, 0.5, "VWAP"), (dc_report_7w, 0.5, "DC"),
            ])
            logger.info("")

        if bb_7w:
            logger.info("  --- 50%% BB + 50%% DC (7w) ---")
            compute_portfolio("BB+DC_50_50_7w", [
                (bb_7w, 0.5, "BB"), (dc_report_7w, 0.5, "DC"),
            ])
            logger.info("")

        if rsi_7w and bb_7w:
            logger.info("  --- 33%% BB + 33%% RSI + 33%% DC (7w) ---")
            compute_portfolio("BB+RSI+DC_equal_7w", [
                (bb_7w, 0.33, "BB"), (rsi_7w, 0.33, "RSI"),
                (dc_report_7w, 0.34, "DC"),
            ])
            logger.info("")

        if rsi_7w and vwap_7w:
            logger.info("  --- 33%% RSI + 33%% VWAP + 33%% DC (7w) ---")
            compute_portfolio("RSI+VWAP+DC_equal_7w", [
                (rsi_7w, 0.33, "RSI"), (vwap_7w, 0.33, "VWAP"),
                (dc_report_7w, 0.34, "DC"),
            ])
            logger.info("")

    # ─────────────────────────────────────────────────────────────
    #   SUMMARY
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  PHASE 12 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    # Rank all grid results
    logger.info("  Donchian Grid Results (5w):")
    logger.info("  %-30s %8s %6s %6s %8s %6s %6s %5s",
                "Strategy", "OOS Ret", "WF Rob", "Trades", "Full Ret", "MaxDD", "PF", "Shp")
    logger.info("  " + "-" * 95)

    sorted_results = sorted(all_results,
                            key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
                            reverse=True)
    for rname, report, full in sorted_results:
        logger.info(
            "  %-30s %+7.2f%% %5d%% %6d %+7.2f%% %5.1f%% %5.2f %5.2f",
            rname, report.oos_total_return,
            int(report.robustness_score * 100), report.oos_total_trades,
            full.total_return, full.max_drawdown, full.profit_factor,
            full.sharpe_ratio,
        )

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 12 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

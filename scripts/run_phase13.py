#!/usr/bin/env python3
"""Phase 13 — 9w Portfolio Validation + CCI Mean Reversion.

Phase 12 findings:
  - RSI_35_65+MTF (9w): 78% robustness, OOS +20.59% — BEST single
  - VWAP+DC 50/50 (7w): 85% robustness, OOS +19.64% — BEST portfolio
  - BB+DC 50/50 (7w): 85% robustness, OOS +19.18%
  - RSI+VWAP+DC equal (7w): 85% robustness, OOS +17.04%
  - DC+MTF degrades at 9w (55% rob) like BB

Phase 13 goals:
  1. 9-window portfolio validation — does 85% robustness hold?
  2. CCI Mean Reversion (new, uncorrelated with RSI)
     - CCI uses typical price deviation from SMA (unbounded)
     - RSI uses close-to-close gains/losses (bounded 0-100)
     - Potentially low correlation → portfolio diversifier
  3. Best CCI configs at 7w/9w + portfolio combos
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
from src.strategy.bb_squeeze_breakout import BBSqueezeBreakoutStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.cci_mean_reversion import CCIMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase13")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase13.log", mode="w")
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


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add CCI indicator to DataFrame."""
    col = f"CCI_{period}"
    if col not in df.columns:
        df[col] = ta.cci(df["high"], df["low"], df["close"], length=period)
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
    logger.info("  PHASE 13 — 9w Portfolio Validation + CCI Mean Reversion")
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

    engine = BacktestEngine(max_hold_bars=48)

    # ═════════════════════════════════════════════════════════════
    #   PART 0: 9-window Portfolio Validation
    # ═════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 0: 9-window Portfolio Validation")
    logger.info("─" * 72)
    logger.info("")

    # Run all individual strategies at 9w
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

    def make_dc_mtf():
        base = DonchianTrendStrategy(
            entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
            vol_mult=0.8, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    reports_9w = {}
    for bname, factory in [
        ("RSI_35_65+MTF", make_rsi_mtf),
        ("VWAP_24_2.0+MTF", make_vwap_mtf),
        ("BB+MTF", make_bb_mtf),
        ("DC_24+MTF", make_dc_mtf),
    ]:
        label = f"{bname}_9w"
        logger.info("  --- %s ---", label)
        wf = WalkForwardAnalyzer(n_windows=9, engine=engine)
        report = wf.run(factory, df_1h, htf_df=df_4h)
        log_wf(label, report, engine, factory, df_1h, htf_df=df_4h)
        reports_9w[bname] = report
        logger.info("")

    # 9w Portfolio combinations
    logger.info("─" * 72)
    logger.info("  9-window Portfolio Combinations")
    logger.info("─" * 72)
    logger.info("")

    rsi_9w = reports_9w["RSI_35_65+MTF"]
    vwap_9w = reports_9w["VWAP_24_2.0+MTF"]
    bb_9w = reports_9w["BB+MTF"]
    dc_9w = reports_9w["DC_24+MTF"]

    portfolios_9w = [
        ("VWAP+DC_50_50_9w", [(vwap_9w, 0.5, "VWAP"), (dc_9w, 0.5, "DC")]),
        ("BB+DC_50_50_9w", [(bb_9w, 0.5, "BB"), (dc_9w, 0.5, "DC")]),
        ("RSI+DC_50_50_9w", [(rsi_9w, 0.5, "RSI"), (dc_9w, 0.5, "DC")]),
        ("RSI+VWAP+DC_equal_9w", [(rsi_9w, 0.33, "RSI"), (vwap_9w, 0.33, "VWAP"), (dc_9w, 0.34, "DC")]),
        ("RSI+VWAP_50_50_9w", [(rsi_9w, 0.5, "RSI"), (vwap_9w, 0.5, "VWAP")]),
        ("BB+RSI+DC_equal_9w", [(bb_9w, 0.33, "BB"), (rsi_9w, 0.33, "RSI"), (dc_9w, 0.34, "DC")]),
    ]

    port_results_9w = {}
    for pname, components in portfolios_9w:
        logger.info("  --- %s ---", pname)
        oos, rob, trades = compute_portfolio(pname, components)
        port_results_9w[pname] = (oos, rob, trades)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 1: CCI Mean Reversion Parameter Grid (5-window WF)
    # ═════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 1: CCI Mean Reversion Parameter Grid (5-window WF)")
    logger.info("─" * 72)
    logger.info("")

    cci_configs = [
        # (name, cci_period, oversold, overbought, sl, tp, vol, cool)
        ("CCI_20_100",     20, 100, 100, 2.0, 3.0, 0.0, 6),
        ("CCI_20_150",     20, 150, 150, 2.0, 3.0, 0.0, 6),
        ("CCI_20_200",     20, 200, 200, 2.0, 3.0, 0.0, 6),
        ("CCI_14_100",     14, 100, 100, 2.0, 3.0, 0.0, 6),
        ("CCI_14_150",     14, 150, 150, 2.0, 3.0, 0.0, 6),
        ("CCI_20_100_s15", 20, 100, 100, 1.5, 2.5, 0.0, 6),
        ("CCI_20_100_s25", 20, 100, 100, 2.5, 3.5, 0.0, 6),
        ("CCI_20_100_v08", 20, 100, 100, 2.0, 3.0, 0.8, 6),
        ("CCI_20_100_c4",  20, 100, 100, 2.0, 3.0, 0.0, 4),
    ]

    # Add CCI indicators
    for _, period, *_ in cci_configs:
        df_1h = add_cci(df_1h, period)

    cci_standalone = []
    cci_mtf = []

    for cfg_name, period, osold, obought, sl, tp, vol, cool in cci_configs:
        # Standalone
        logger.info("  --- %s ---", cfg_name)

        def make_cci(period=period, osold=osold, obought=obought,
                     sl=sl, tp=tp, vol=vol, cool=cool):
            return CCIMeanReversionStrategy(
                cci_period=period, oversold_level=osold, overbought_level=obought,
                atr_sl_mult=sl, atr_tp_mult=tp, vol_mult=vol, cooldown_bars=cool,
            )

        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(make_cci, df_1h)
        full = log_wf(cfg_name, report, engine, make_cci, df_1h)
        cci_standalone.append((cfg_name, report, full))
        logger.info("")

    # CCI + MTF variants
    logger.info("  --- CCI + MTF variants ---")
    logger.info("")

    for cfg_name, period, osold, obought, sl, tp, vol, cool in cci_configs:
        mtf_name = f"{cfg_name}+MTF"
        logger.info("  --- %s ---", mtf_name)

        def make_cci_mtf(period=period, osold=osold, obought=obought,
                         sl=sl, tp=tp, vol=vol, cool=cool):
            base = CCIMeanReversionStrategy(
                cci_period=period, oversold_level=osold, overbought_level=obought,
                atr_sl_mult=sl, atr_tp_mult=tp, vol_mult=vol, cooldown_bars=cool,
            )
            return MultiTimeframeFilter(base)

        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(make_cci_mtf, df_1h, htf_df=df_4h)
        full = log_wf(mtf_name, report, engine, make_cci_mtf, df_1h, htf_df=df_4h)
        cci_mtf.append((mtf_name, report, full))
        logger.info("")

    # Find best CCI configs
    all_cci = cci_standalone + cci_mtf
    best_cci = max(all_cci, key=lambda x: (x[1].robustness_score, x[1].oos_total_return))
    logger.info("  Best CCI config: %s — OOS %+.2f%%, Robustness %d%%",
                best_cci[0], best_cci[1].oos_total_return,
                int(best_cci[1].robustness_score * 100))
    logger.info("")

    # Find best standalone and best +MTF separately
    best_cci_sa = max(cci_standalone, key=lambda x: (x[1].robustness_score, x[1].oos_total_return))
    best_cci_mf = max(cci_mtf, key=lambda x: (x[1].robustness_score, x[1].oos_total_return))

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Best CCI — Extended WF (7w, 9w)
    # ═════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 2: Best CCI Configs — Extended WF (7w, 9w)")
    logger.info("─" * 72)
    logger.info("")

    # Parse best configs from names
    def parse_cci_cfg(name):
        for cfg_name, period, osold, obought, sl, tp, vol, cool in cci_configs:
            if cfg_name == name or f"{cfg_name}+MTF" == name:
                return (period, osold, obought, sl, tp, vol, cool)
        return (20, 100, 100, 2.0, 3.0, 0.0, 6)

    sa_cfg = parse_cci_cfg(best_cci_sa[0])
    mf_cfg = parse_cci_cfg(best_cci_mf[0])

    def make_best_cci_sa():
        p, o, ob, sl, tp, v, c = sa_cfg
        return CCIMeanReversionStrategy(
            cci_period=p, oversold_level=o, overbought_level=ob,
            atr_sl_mult=sl, atr_tp_mult=tp, vol_mult=v, cooldown_bars=c,
        )

    def make_best_cci_mtf():
        p, o, ob, sl, tp, v, c = mf_cfg
        base = CCIMeanReversionStrategy(
            cci_period=p, oversold_level=o, overbought_level=ob,
            atr_sl_mult=sl, atr_tp_mult=tp, vol_mult=v, cooldown_bars=c,
        )
        return MultiTimeframeFilter(base)

    cci_extended = {}

    for n_win, label in [(7, "7w"), (9, "9w")]:
        # Standalone
        ext_name = f"{best_cci_sa[0]}_{label}"
        logger.info("  --- %s ---", ext_name)
        wf = WalkForwardAnalyzer(n_windows=n_win, engine=engine)
        report = wf.run(make_best_cci_sa, df_1h)
        log_wf(ext_name, report, engine, make_best_cci_sa, df_1h)
        cci_extended[ext_name] = report
        logger.info("")

        # MTF
        ext_name_mtf = f"{best_cci_mf[0]}_{label}"
        logger.info("  --- %s ---", ext_name_mtf)
        wf = WalkForwardAnalyzer(n_windows=n_win, engine=engine)
        report = wf.run(make_best_cci_mtf, df_1h, htf_df=df_4h)
        log_wf(ext_name_mtf, report, engine, make_best_cci_mtf, df_1h, htf_df=df_4h)
        cci_extended[ext_name_mtf] = report
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Portfolio Combos with CCI (7w)
    # ═════════════════════════════════════════════════════════════
    # Only if CCI shows promise (>=60% robustness at 5w)
    if best_cci_mf[1].robustness_score >= 0.6:
        logger.info("─" * 72)
        logger.info("  PART 3: Portfolio Combos with CCI (7-window)")
        logger.info("─" * 72)
        logger.info("")

        # Run baselines at 7w for portfolio computation
        baseline_7w = {}
        for bname, factory in [
            ("RSI+MTF", make_rsi_mtf),
            ("VWAP+MTF", make_vwap_mtf),
            ("DC+MTF", make_dc_mtf),
        ]:
            label = f"{bname}_7w"
            logger.info("  --- %s ---", label)
            wf = WalkForwardAnalyzer(n_windows=7, engine=engine)
            report = wf.run(factory, df_1h, htf_df=df_4h)
            log_wf(label, report, engine, factory, df_1h, htf_df=df_4h)
            baseline_7w[bname] = report
            logger.info("")

        # CCI+MTF at 7w
        cci_mtf_7w_name = f"{best_cci_mf[0]}_7w"
        cci_7w = cci_extended.get(cci_mtf_7w_name)

        if cci_7w and cci_7w.total_windows > 0:
            rsi_7w = baseline_7w["RSI+MTF"]
            vwap_7w = baseline_7w["VWAP+MTF"]
            dc_7w = baseline_7w["DC+MTF"]

            cci_portfolios = [
                ("RSI+CCI_50_50_7w", [(rsi_7w, 0.5, "RSI"), (cci_7w, 0.5, "CCI")]),
                ("VWAP+CCI_50_50_7w", [(vwap_7w, 0.5, "VWAP"), (cci_7w, 0.5, "CCI")]),
                ("DC+CCI_50_50_7w", [(dc_7w, 0.5, "DC"), (cci_7w, 0.5, "CCI")]),
                ("RSI+VWAP+CCI_equal_7w", [(rsi_7w, 0.33, "RSI"), (vwap_7w, 0.33, "VWAP"), (cci_7w, 0.34, "CCI")]),
                ("VWAP+DC+CCI_equal_7w", [(vwap_7w, 0.33, "VWAP"), (dc_7w, 0.33, "DC"), (cci_7w, 0.34, "CCI")]),
            ]

            for pname, components in cci_portfolios:
                logger.info("  --- %s ---", pname)
                compute_portfolio(pname, components)
                logger.info("")
    else:
        logger.info("")
        logger.info("  CCI best MTF robustness < 60%% — skipping portfolio combos.")
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   SUMMARY
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 13 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    # 9w individual strategies
    logger.info("  Individual Strategies (9w):")
    logger.info("  %-25s %8s %6s %6s", "Strategy", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 50)
    for bname, report in sorted(reports_9w.items(),
                                 key=lambda x: x[1].robustness_score, reverse=True):
        logger.info("  %-25s %+7.2f%% %5d%% %6d",
                     bname, report.oos_total_return,
                     int(report.robustness_score * 100), report.oos_total_trades)

    # 9w portfolios
    logger.info("")
    logger.info("  Portfolios (9w):")
    logger.info("  %-30s %8s %6s %6s", "Portfolio", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 55)
    for pname, (oos, rob, trades) in sorted(port_results_9w.items(),
                                             key=lambda x: x[1][1], reverse=True):
        logger.info("  %-30s %+7.2f%% %5d%% %6d",
                     pname, oos, int(rob * 100), trades)

    # CCI grid results
    logger.info("")
    logger.info("  CCI Grid Results (5w):")
    logger.info("  %-30s %8s %6s %6s %8s %6s %6s %5s",
                "Strategy", "OOS Ret", "WF Rob", "Trades", "Full Ret", "MaxDD", "PF", "Shp")
    logger.info("  " + "-" * 95)

    sorted_cci = sorted(all_cci,
                        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
                        reverse=True)
    for rname, report, full in sorted_cci:
        logger.info(
            "  %-30s %+7.2f%% %5d%% %6d %+7.2f%% %5.1f%% %5.2f %5.2f",
            rname, report.oos_total_return,
            int(report.robustness_score * 100), report.oos_total_trades,
            full.total_return, full.max_drawdown, full.profit_factor,
            full.sharpe_ratio,
        )

    # CCI extended
    if cci_extended:
        logger.info("")
        logger.info("  CCI Extended WF:")
        for ename, report in cci_extended.items():
            logger.info("  %-30s OOS %+7.2f%% | Rob %d%% | Trades %d",
                        ename, report.oos_total_return,
                        int(report.robustness_score * 100), report.oos_total_trades)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 13 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

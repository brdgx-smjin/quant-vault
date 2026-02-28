#!/usr/bin/env python3
"""Phase 14 — Portfolio Weight Optimization + 4-Strategy Ensemble + Funding Rate Analysis.

Phase 13 findings:
  - Best 9w single: RSI_35_65+MTF (66%), CCI_20_200+MTF (66%)
  - Best 9w portfolio: RSI+DC 50/50 (77% rob, +20.27% OOS)
  - Best 7w portfolio: DC+CCI 50/50 (85% rob, +24.46% OOS)
  - CCI portfolios only tested at 7w — need 9w validation

Phase 14 goals:
  1. Weight optimization for top 9w portfolios (60/40, 40/60, 70/30)
  2. CCI portfolio combos at 9w (missing from Phase 13)
  3. 4-strategy ensemble: RSI+VWAP+DC+CCI at 9w
  4. Funding Rate signal analysis — correlation with future returns
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
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

logger = logging.getLogger("phase14")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase14.log", mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

for name in ["src.backtest.engine", "src.strategy.mtf_filter", "src.strategy.portfolio"]:
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


# ─── Strategy factories ────────────────────────────────────────────
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


def make_dc_mtf():
    base = DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_cci_mtf():
    base = CCIMeanReversionStrategy(
        cci_period=20, oversold_level=200, overbought_level=200,
        atr_sl_mult=2.0, atr_tp_mult=3.0, vol_mult=0.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 14 — Portfolio Weight Optimization + 4-Strategy Ensemble")
    logger.info("=" * 72)
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_4h = load_data("4h")
    df_1h = add_cci(df_1h, 20)

    logger.info("1h data: %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("4h data: %d bars (%s ~ %s)", len(df_4h),
                df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine = BacktestEngine(max_hold_bars=48)

    # ═════════════════════════════════════════════════════════════
    #   PART 0: Run all 5 strategies at 9w to get report objects
    # ═════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 0: Individual Strategies at 9-window WF")
    logger.info("─" * 72)
    logger.info("")

    reports_9w = {}
    for bname, factory in [
        ("RSI", make_rsi_mtf),
        ("VWAP", make_vwap_mtf),
        ("DC", make_dc_mtf),
        ("CCI", make_cci_mtf),
    ]:
        label = f"{bname}+MTF_9w"
        logger.info("  --- %s ---", label)
        wf = WalkForwardAnalyzer(n_windows=9, engine=engine)
        report = wf.run(factory, df_1h, htf_df=df_4h)
        log_wf(label, report, engine, factory, df_1h, htf_df=df_4h)
        reports_9w[bname] = report
        logger.info("")

    rsi_9w = reports_9w["RSI"]
    vwap_9w = reports_9w["VWAP"]
    dc_9w = reports_9w["DC"]
    cci_9w = reports_9w["CCI"]

    # ═════════════════════════════════════════════════════════════
    #   PART 1: Weight Optimization for Top 9w Portfolios
    # ═════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 1: Weight Optimization (9w)")
    logger.info("─" * 72)
    logger.info("")

    weight_results = {}

    # RSI+DC weight variants
    for w_rsi, w_dc in [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6), (0.7, 0.3), (0.3, 0.7)]:
        pname = f"RSI+DC_{int(w_rsi*100)}_{int(w_dc*100)}_9w"
        logger.info("  --- %s ---", pname)
        oos, rob, trades = compute_portfolio(
            pname,
            [(rsi_9w, w_rsi, "RSI"), (dc_9w, w_dc, "DC")],
        )
        weight_results[pname] = (oos, rob, trades)
        logger.info("")

    # VWAP+DC weight variants
    for w_vwap, w_dc in [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6), (0.7, 0.3), (0.3, 0.7)]:
        pname = f"VWAP+DC_{int(w_vwap*100)}_{int(w_dc*100)}_9w"
        logger.info("  --- %s ---", pname)
        oos, rob, trades = compute_portfolio(
            pname,
            [(vwap_9w, w_vwap, "VWAP"), (dc_9w, w_dc, "DC")],
        )
        weight_results[pname] = (oos, rob, trades)
        logger.info("")

    # RSI+VWAP+DC weight variants
    for w_rsi, w_vwap, w_dc in [
        (0.33, 0.33, 0.34),  # equal
        (0.4, 0.2, 0.4),     # RSI+DC heavy
        (0.2, 0.4, 0.4),     # VWAP+DC heavy
        (0.4, 0.4, 0.2),     # MR heavy (RSI+VWAP)
        (0.5, 0.25, 0.25),   # RSI dominant
        (0.25, 0.25, 0.5),   # DC dominant
    ]:
        pname = f"RSI+VWAP+DC_{int(w_rsi*100)}_{int(w_vwap*100)}_{int(w_dc*100)}_9w"
        logger.info("  --- %s ---", pname)
        oos, rob, trades = compute_portfolio(
            pname,
            [(rsi_9w, w_rsi, "RSI"), (vwap_9w, w_vwap, "VWAP"), (dc_9w, w_dc, "DC")],
        )
        weight_results[pname] = (oos, rob, trades)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: CCI Portfolio Combos at 9w
    # ═════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 2: CCI Portfolio Combos at 9w")
    logger.info("─" * 72)
    logger.info("")

    cci_port_results = {}

    cci_portfolios = [
        ("DC+CCI_50_50_9w", [(dc_9w, 0.5, "DC"), (cci_9w, 0.5, "CCI")]),
        ("RSI+CCI_50_50_9w", [(rsi_9w, 0.5, "RSI"), (cci_9w, 0.5, "CCI")]),
        ("VWAP+CCI_50_50_9w", [(vwap_9w, 0.5, "VWAP"), (cci_9w, 0.5, "CCI")]),
        ("DC+CCI_60_40_9w", [(dc_9w, 0.6, "DC"), (cci_9w, 0.4, "CCI")]),
        ("DC+CCI_40_60_9w", [(dc_9w, 0.4, "DC"), (cci_9w, 0.6, "CCI")]),
        ("RSI+DC+CCI_equal_9w", [(rsi_9w, 0.33, "RSI"), (dc_9w, 0.33, "DC"), (cci_9w, 0.34, "CCI")]),
        ("VWAP+DC+CCI_equal_9w", [(vwap_9w, 0.33, "VWAP"), (dc_9w, 0.33, "DC"), (cci_9w, 0.34, "CCI")]),
    ]

    for pname, components in cci_portfolios:
        logger.info("  --- %s ---", pname)
        oos, rob, trades = compute_portfolio(pname, components)
        cci_port_results[pname] = (oos, rob, trades)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: 4-Strategy Ensemble at 9w
    # ═════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 3: 4-Strategy Ensemble (RSI+VWAP+DC+CCI) at 9w")
    logger.info("─" * 72)
    logger.info("")

    ensemble_results = {}

    four_strat_configs = [
        ("equal_25", 0.25, 0.25, 0.25, 0.25),
        ("MR_heavy_30_30_20_20", 0.30, 0.30, 0.20, 0.20),
        ("trend_heavy_20_20_30_30", 0.20, 0.20, 0.30, 0.30),
        ("RSI_DC_focus_30_15_30_25", 0.30, 0.15, 0.30, 0.25),
        ("balanced_30_20_30_20", 0.30, 0.20, 0.30, 0.20),
    ]

    for config_name, w_rsi, w_vwap, w_dc, w_cci in four_strat_configs:
        pname = f"4strat_{config_name}_9w"
        logger.info("  --- %s ---", pname)
        logger.info("  Weights: RSI=%.0f%% VWAP=%.0f%% DC=%.0f%% CCI=%.0f%%",
                     w_rsi*100, w_vwap*100, w_dc*100, w_cci*100)
        oos, rob, trades = compute_portfolio(
            pname,
            [
                (rsi_9w, w_rsi, "RSI"),
                (vwap_9w, w_vwap, "VWAP"),
                (dc_9w, w_dc, "DC"),
                (cci_9w, w_cci, "CCI"),
            ],
        )
        ensemble_results[pname] = (oos, rob, trades)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Funding Rate Signal Analysis
    # ═════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 4: Funding Rate Signal Analysis")
    logger.info("─" * 72)
    logger.info("")

    try:
        fr = pd.read_parquet(ROOT / "data/processed/BTC_USDT_USDT_funding_rate.parquet")
        logger.info("  Funding rate data: %d bars (%s ~ %s)",
                     len(fr), fr.index[0].date(), fr.index[-1].date())

        # Resample to 1h (forward fill)
        fr_1h = fr.resample("1h").ffill()
        logger.info("  Resampled to 1h: %d bars", len(fr_1h))

        # Merge with 1h OHLCV
        df_merged = df_1h.join(fr_1h, how="inner")
        logger.info("  Merged data: %d bars (%s ~ %s)",
                     len(df_merged), df_merged.index[0].date(), df_merged.index[-1].date())

        # Compute forward returns
        for horizon in [1, 4, 8, 24]:
            col = f"fwd_ret_{horizon}h"
            df_merged[col] = df_merged["close"].pct_change(horizon).shift(-horizon) * 100

        # Funding rate quintile analysis
        df_valid = df_merged.dropna(subset=["fundingRate", "fwd_ret_8h"])

        if len(df_valid) > 100:
            df_valid["fr_quintile"] = pd.qcut(
                df_valid["fundingRate"], q=5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
                duplicates="drop",
            )

            logger.info("")
            logger.info("  Funding Rate Quintile → Forward Returns:")
            logger.info("  %-10s %8s %8s %8s %8s %8s",
                         "Quintile", "FR_mean", "1h_ret", "4h_ret", "8h_ret", "24h_ret")
            logger.info("  " + "-" * 60)

            for q in df_valid["fr_quintile"].cat.categories:
                mask = df_valid["fr_quintile"] == q
                subset = df_valid[mask]
                logger.info(
                    "  %-10s %+7.4f%% %+7.3f%% %+7.3f%% %+7.3f%% %+7.3f%%",
                    q,
                    subset["fundingRate"].mean() * 100,
                    subset["fwd_ret_1h"].mean(),
                    subset["fwd_ret_4h"].mean(),
                    subset["fwd_ret_8h"].mean(),
                    subset["fwd_ret_24h"].mean(),
                )

            # Extreme funding analysis
            logger.info("")
            logger.info("  Extreme Funding Rate Analysis:")
            for threshold in [0.03, 0.05, 0.1]:
                high_mask = df_valid["fundingRate"] > threshold / 100
                low_mask = df_valid["fundingRate"] < -threshold / 100

                high_count = high_mask.sum()
                low_count = low_mask.sum()

                if high_count > 5:
                    high_8h = df_valid.loc[high_mask, "fwd_ret_8h"].mean()
                    high_24h = df_valid.loc[high_mask, "fwd_ret_24h"].mean()
                    logger.info(
                        "  FR > +%.2f%% (%d bars): 8h_ret %+.3f%%, 24h_ret %+.3f%%",
                        threshold, high_count, high_8h, high_24h,
                    )

                if low_count > 5:
                    low_8h = df_valid.loc[low_mask, "fwd_ret_8h"].mean()
                    low_24h = df_valid.loc[low_mask, "fwd_ret_24h"].mean()
                    logger.info(
                        "  FR < -%.2f%% (%d bars): 8h_ret %+.3f%%, 24h_ret %+.3f%%",
                        threshold, low_count, low_8h, low_24h,
                    )

            # Correlation
            logger.info("")
            logger.info("  Correlation (fundingRate vs forward returns):")
            for horizon in [1, 4, 8, 24]:
                col = f"fwd_ret_{horizon}h"
                corr = df_valid["fundingRate"].corr(df_valid[col])
                logger.info("  %dh: %.4f", horizon, corr)
        else:
            logger.info("  Not enough merged data for analysis (%d bars)", len(df_valid))
    except Exception as e:
        logger.info("  Funding rate analysis failed: %s", e)

    # ═════════════════════════════════════════════════════════════
    #   SUMMARY
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 14 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    # Individual strategies recap
    logger.info("  Individual Strategies (9w):")
    logger.info("  %-20s %8s %6s %6s", "Strategy", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 45)
    for bname, report in sorted(reports_9w.items(),
                                 key=lambda x: x[1].robustness_score, reverse=True):
        logger.info("  %-20s %+7.2f%% %5d%% %6d",
                     bname + "+MTF", report.oos_total_return,
                     int(report.robustness_score * 100), report.oos_total_trades)

    # Weight optimization results
    logger.info("")
    logger.info("  Weight Optimization (9w):")
    logger.info("  %-35s %8s %6s %6s", "Portfolio", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 60)
    for pname, (oos, rob, trades) in sorted(weight_results.items(),
                                             key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.77 else ""
        logger.info("  %-35s %+7.2f%% %5d%% %6d%s",
                     pname, oos, int(rob * 100), trades, marker)

    # CCI portfolios at 9w
    logger.info("")
    logger.info("  CCI Portfolios (9w):")
    logger.info("  %-35s %8s %6s %6s", "Portfolio", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 60)
    for pname, (oos, rob, trades) in sorted(cci_port_results.items(),
                                             key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.77 else ""
        logger.info("  %-35s %+7.2f%% %5d%% %6d%s",
                     pname, oos, int(rob * 100), trades, marker)

    # 4-strategy ensemble
    logger.info("")
    logger.info("  4-Strategy Ensemble (9w):")
    logger.info("  %-40s %8s %6s %6s", "Portfolio", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 65)
    for pname, (oos, rob, trades) in sorted(ensemble_results.items(),
                                             key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.77 else ""
        logger.info("  %-40s %+7.2f%% %5d%% %6d%s",
                     pname, oos, int(rob * 100), trades, marker)

    # Best overall
    logger.info("")
    logger.info("  " + "─" * 40)
    all_results = {}
    all_results.update(weight_results)
    all_results.update(cci_port_results)
    all_results.update(ensemble_results)

    best = max(all_results.items(), key=lambda x: (x[1][1], x[1][0]))
    logger.info("  BEST OVERALL: %s", best[0])
    logger.info("    OOS: %+.2f%% | Robustness: %d%% | Trades: %d",
                best[1][0], int(best[1][1] * 100), best[1][2])

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 14 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

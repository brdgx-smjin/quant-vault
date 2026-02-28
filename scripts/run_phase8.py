#!/usr/bin/env python3
"""Phase 8: New Strategy Exploration — VWAP + Regime Switching.

Tests new strategy ideas from system prompt task 2:
1. VWAP Mean Reversion (1h) standalone
2. VWAP+DC portfolio (documented as 85% at 7w in donchian docstring)
3. VWAP+RSI+DC triple portfolio
4. Regime Switch (DC for trending + RSI MR for ranging)

All tested with 7-window Walk-Forward. Minimum 60% robustness to adopt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, SYMBOL
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import (
    CrossTFComponent,
    WalkForwardAnalyzer,
)
from src.indicators.basic import BasicIndicators
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.regime_switch import RegimeSwitchStrategy
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
from src.monitoring.logger import setup_logging

logger = setup_logging("phase8")

logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)

SYMBOL_FILE = SYMBOL.replace("/", "_").replace(":", "_")
N_WINDOWS = 7


def load_data(timeframe: str) -> pd.DataFrame:
    path = f"{DATA_DIR}/processed/{SYMBOL_FILE}_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    df.dropna(inplace=True)
    return df


def print_result(name: str, r: BacktestResult) -> None:
    logger.info(
        "  %-40s %+8.2f%% | Shp %5.2f | DD %5.1f%% | WR %5.1f%% | %4d tr | PF %5.2f",
        name, r.total_return, r.sharpe_ratio, r.max_drawdown,
        r.win_rate * 100, r.total_trades, r.profit_factor,
    )


def print_wf(name: str, report) -> None:
    for w in report.windows:
        logger.info(
            "  W%d: IS %+7.2f%% (WR %.0f%%, %d tr) | OOS %+7.2f%% (WR %.0f%%, %d tr)",
            w.window_id,
            w.in_sample.total_return, w.in_sample.win_rate * 100,
            w.in_sample.total_trades,
            w.out_of_sample.total_return, w.out_of_sample.win_rate * 100,
            w.out_of_sample.total_trades,
        )
    logger.info(
        "  OOS: %+.2f%% | Robustness: %.0f%% (%d/%d) | Trades: %d",
        report.oos_total_return, report.robustness_score * 100,
        report.oos_profitable_windows, report.total_windows,
        report.oos_total_trades,
    )


def main() -> None:
    logger.info("=" * 72)
    logger.info("  PHASE 8 — New Strategy Exploration: VWAP + Regime Switch")
    logger.info("=" * 72)
    logger.info("")

    df_1h = load_data("1h")
    df_4h = load_data("4h")

    logger.info("1h data: %d bars (%s ~ %s)",
                len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("4h data: %d bars (%s ~ %s)",
                len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine = BacktestEngine(initial_capital=10_000, max_hold_bars=48, freq="1h")
    wf = WalkForwardAnalyzer(train_ratio=0.7, n_windows=N_WINDOWS, engine=engine)

    all_results = []

    # ================================================================
    # PART 1: VWAP Mean Reversion (1h) — Best config from docstring
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 1: VWAP Mean Reversion + MTF (1h, %dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    vwap_configs = [
        ("VWAP_p24_b2.0", 24, 2.0, 35.0, 2.0, 4),
        ("VWAP_p48_b2.0", 48, 2.0, 35.0, 2.0, 6),
        ("VWAP_p24_b1.5", 24, 1.5, 35.0, 2.0, 4),
    ]

    vwap_results = {}
    for label, vp, bm, rsi_t, sl, cool in vwap_configs:
        def vwap_factory(vp=vp, bm=bm, rsi_t=rsi_t, sl=sl, cool=cool):
            base = VWAPMeanReversionStrategy(
                vwap_period=vp, band_mult=bm, rsi_threshold=rsi_t,
                atr_sl_mult=sl, cooldown_bars=cool,
            )
            return MultiTimeframeFilter(base)

        rpt = wf.run(vwap_factory, df_1h, htf_df=df_4h)
        r_full = engine.run(vwap_factory(), df_1h, htf_df=df_4h)

        logger.info("  --- %s ---", label)
        print_wf(label, rpt)
        print_result(label + " Full", r_full)
        logger.info("")
        all_results.append((label, rpt, r_full))
        vwap_results[label] = rpt

    # ================================================================
    # PART 2: RSI MR + DC baselines for portfolio comparison
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 2: RSI + DC Baselines (%dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    def rsi_factory():
        base = RSIMeanReversionStrategy(
            rsi_oversold=35.0, rsi_overbought=65.0,
            atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    def dc_factory():
        base = DonchianTrendStrategy(
            entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
            vol_mult=0.8, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    wf_rsi = wf.run(rsi_factory, df_1h, htf_df=df_4h)
    r_rsi = engine.run(rsi_factory(), df_1h, htf_df=df_4h)
    logger.info("  --- RSI_MR+MTF ---")
    print_wf("RSI_MR", wf_rsi)
    print_result("RSI_MR Full", r_rsi)
    logger.info("")
    all_results.append(("RSI_MR+MTF_1h", wf_rsi, r_rsi))

    wf_dc = wf.run(dc_factory, df_1h, htf_df=df_4h)
    r_dc = engine.run(dc_factory(), df_1h, htf_df=df_4h)
    logger.info("  --- DC+MTF ---")
    print_wf("DC_MTF", wf_dc)
    print_result("DC_MTF Full", r_dc)
    logger.info("")
    all_results.append(("DC+MTF_1h", wf_dc, r_dc))

    # ================================================================
    # PART 3: Portfolio combinations
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 3: Portfolio Combinations (%dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    # Pick best VWAP config
    best_vwap_label = max(vwap_results, key=lambda k: vwap_results[k].robustness_score)
    best_vwap = vwap_results[best_vwap_label]

    # VWAP + DC 50/50
    if best_vwap.windows and wf_dc.windows:
        n_win = min(len(best_vwap.windows), len(wf_dc.windows))
        logger.info("  --- VWAP+DC 50/50 (best VWAP: %s) ---", best_vwap_label)
        vwap_dc_oos = []
        vwap_dc_prof = 0
        for i in range(n_win):
            v_ret = best_vwap.windows[i].out_of_sample.total_return
            dc_ret = wf_dc.windows[i].out_of_sample.total_return
            combined = 0.5 * v_ret + 0.5 * dc_ret
            vwap_dc_oos.append(combined)
            logger.info("  W%d: VWAP %+6.2f%% + DC %+6.2f%% -> %+6.2f%%",
                        i + 1, v_ret, dc_ret, combined)
            if combined > 0:
                vwap_dc_prof += 1

        compounded = 1.0
        for r in vwap_dc_oos:
            compounded *= (1 + r / 100)
        vwap_dc_total = (compounded - 1) * 100
        vwap_dc_rob = vwap_dc_prof / n_win

        logger.info("  VWAP+DC 50/50: OOS %+.2f%% | Robustness: %.0f%% (%d/%d)",
                    vwap_dc_total, vwap_dc_rob * 100, vwap_dc_prof, n_win)
    logger.info("")

    # RSI + DC 50/50
    if wf_rsi.windows and wf_dc.windows:
        n_win = min(len(wf_rsi.windows), len(wf_dc.windows))
        logger.info("  --- RSI+DC 50/50 ---")
        rsi_dc_oos = []
        rsi_dc_prof = 0
        for i in range(n_win):
            r_ret = wf_rsi.windows[i].out_of_sample.total_return
            dc_ret = wf_dc.windows[i].out_of_sample.total_return
            combined = 0.5 * r_ret + 0.5 * dc_ret
            rsi_dc_oos.append(combined)
            logger.info("  W%d: RSI %+6.2f%% + DC %+6.2f%% -> %+6.2f%%",
                        i + 1, r_ret, dc_ret, combined)
            if combined > 0:
                rsi_dc_prof += 1

        compounded = 1.0
        for r in rsi_dc_oos:
            compounded *= (1 + r / 100)
        rsi_dc_total = (compounded - 1) * 100
        rsi_dc_rob = rsi_dc_prof / n_win

        logger.info("  RSI+DC 50/50: OOS %+.2f%% | Robustness: %.0f%% (%d/%d)",
                    rsi_dc_total, rsi_dc_rob * 100, rsi_dc_prof, n_win)
    logger.info("")

    # VWAP + RSI + DC equal weight
    if best_vwap.windows and wf_rsi.windows and wf_dc.windows:
        n_win = min(len(best_vwap.windows), len(wf_rsi.windows), len(wf_dc.windows))
        logger.info("  --- VWAP+RSI+DC 33/33/34 ---")
        triple_oos = []
        triple_prof = 0
        for i in range(n_win):
            v_ret = best_vwap.windows[i].out_of_sample.total_return
            r_ret = wf_rsi.windows[i].out_of_sample.total_return
            d_ret = wf_dc.windows[i].out_of_sample.total_return
            combined = 0.33 * v_ret + 0.33 * r_ret + 0.34 * d_ret
            triple_oos.append(combined)
            logger.info("  W%d: VWAP %+6.2f%% + RSI %+6.2f%% + DC %+6.2f%% -> %+6.2f%%",
                        i + 1, v_ret, r_ret, d_ret, combined)
            if combined > 0:
                triple_prof += 1

        compounded = 1.0
        for r in triple_oos:
            compounded *= (1 + r / 100)
        triple_total = (compounded - 1) * 100
        triple_rob = triple_prof / n_win

        logger.info("  VWAP+RSI+DC: OOS %+.2f%% | Robustness: %.0f%% (%d/%d)",
                    triple_total, triple_rob * 100, triple_prof, n_win)
    logger.info("")

    # ================================================================
    # PART 4: Regime Switching — DC(trend) + RSI_MR(range)
    # ================================================================
    logger.info("─" * 72)
    logger.info("  PART 4: Regime Switch — DC(trend) + RSI_MR(range) (%dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    for adx_thresh in [20, 25, 30]:
        def regime_factory(thresh=adx_thresh):
            trend = DonchianTrendStrategy(
                entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
                vol_mult=0.8, cooldown_bars=6,
            )
            ranging = RSIMeanReversionStrategy(
                rsi_oversold=35.0, rsi_overbought=65.0,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
            )
            regime = RegimeSwitchStrategy(
                trend_strategy=trend,
                range_strategy=ranging,
                adx_threshold=thresh,
            )
            return MultiTimeframeFilter(regime)

        rpt = wf.run(regime_factory, df_1h, htf_df=df_4h)
        r_full = engine.run(regime_factory(), df_1h, htf_df=df_4h)

        label = f"Regime_ADX{adx_thresh}"
        logger.info("  --- %s ---", label)
        print_wf(label, rpt)
        print_result(label + " Full", r_full)
        logger.info("")
        all_results.append((label, rpt, r_full))

    # ================================================================
    # FINAL RANKING
    # ================================================================
    logger.info("=" * 72)
    logger.info("  PHASE 8 — FINAL RANKING")
    logger.info("=" * 72)
    logger.info("")

    all_results.sort(
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    )

    logger.info(
        "  %-35s %8s %6s %7s %8s %6s %5s %5s",
        "Strategy", "OOS Ret", "WF Rob", "Tr(OOS)",
        "Full Ret", "MaxDD", "PF", "Shp",
    )
    logger.info("  " + "-" * 90)

    for name, rpt, r_full in all_results:
        logger.info(
            "  %-35s %+7.2f%% %5.0f%% %6d   %+7.2f%% %5.1f%% %5.2f %5.2f",
            name, rpt.oos_total_return, rpt.robustness_score * 100,
            rpt.oos_total_trades,
            r_full.total_return, r_full.max_drawdown,
            r_full.profit_factor, r_full.sharpe_ratio,
        )

    logger.info("")

    # Portfolio summaries
    logger.info("  Portfolio summaries:")
    logger.info("  VWAP+DC 50/50: OOS %+.2f%% | Robustness: %.0f%%",
                vwap_dc_total, vwap_dc_rob * 100)
    logger.info("  RSI+DC 50/50:  OOS %+.2f%% | Robustness: %.0f%%",
                rsi_dc_total, rsi_dc_rob * 100)
    logger.info("  VWAP+RSI+DC:   OOS %+.2f%% | Robustness: %.0f%%",
                triple_total, triple_rob * 100)
    logger.info("")

    logger.info("  Reference: Cross-TF 1hRSI/1hDC/15mRSI 33/33/34 (9w) = 88%% rob")
    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 8 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

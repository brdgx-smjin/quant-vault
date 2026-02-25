#!/usr/bin/env python3
"""Phase 10c — ML Regime Classifier vs Rule-Based Regime Switch.

Compare:
  1. RegimeSwitchStrategy(ADX=20) — existing rule-based (80% rob, +5.48% OOS)
  2. MLRegimeStrategy — multi-feature regime detection
  3. Both with and without MTF filter

Goal: Does adding volatility filter, BB_width, and EMA_gap features
improve regime classification over pure ADX threshold?
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
from src.strategy.bb_squeeze_v2 import BBSqueezeV2Strategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
from src.strategy.regime_switch import RegimeSwitchStrategy
from src.strategy.ml_regime_strategy import MLRegimeStrategy

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase10c")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase10.log", mode="a")  # append
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


def _make_bb():
    return BBSqueezeBreakoutStrategy(
        squeeze_lookback=100, squeeze_pctile=25.0,
        vol_mult=1.1, atr_sl_mult=3.0, rr_ratio=2.0,
        require_trend=False, cooldown_bars=6,
    )


def _make_rsi():
    return RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )


def _make_vwap():
    return VWAPMeanReversionStrategy(
        vwap_period=24, band_mult=2.0, rsi_threshold=35,
        atr_sl_mult=2.0, tp_to_vwap_pct=0.8, cooldown_bars=4,
    )


def run():
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 10c — ML Regime Classifier Comparison")
    logger.info("=" * 72)
    logger.info("")

    df_1h = load_data("1h")
    df_4h = load_data("4h")

    engine_mr = BacktestEngine(max_hold_bars=36)
    all_results = []

    # ═══════════════════════════════════════════════════════════════════
    #   Control: Rule-Based Regime (ADX-only)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  Control: RegimeSwitchStrategy (ADX-only)")
    logger.info("─" * 72)
    logger.info("")

    def regime_adx20_factory():
        return RegimeSwitchStrategy(
            trend_strategy=_make_bb(),
            range_strategy=_make_rsi(),
            adx_threshold=20,
        )

    logger.info("  --- Regime_ADX20 (5w) ---")
    wf5 = WalkForwardAnalyzer(n_windows=5, engine=engine_mr)
    rpt = wf5.run(regime_adx20_factory, df_1h)
    full = log_wf("Regime_ADX20", rpt, engine_mr, regime_adx20_factory, df_1h)
    all_results.append(("Regime_ADX20", rpt, full))
    logger.info("")

    # Regime_ADX20 + MTF
    def regime_adx20_mtf_factory():
        return MultiTimeframeFilter(regime_adx20_factory())

    logger.info("  --- Regime_ADX20+MTF (5w) ---")
    wf5m = WalkForwardAnalyzer(n_windows=5, engine=engine_mr)
    rpt = wf5m.run(regime_adx20_mtf_factory, df_1h, htf_df=df_4h)
    full = log_wf("Regime_ADX20+MTF", rpt, engine_mr, regime_adx20_mtf_factory, df_1h, df_4h)
    all_results.append(("Regime_ADX20+MTF", rpt, full))
    logger.info("")

    # ═══════════════════════════════════════════════════════════════════
    #   Test: Multi-Feature Regime (MLRegimeStrategy)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  Test: MLRegimeStrategy (Multi-Feature)")
    logger.info("─" * 72)
    logger.info("")

    configs = [
        {
            "name": "MLRegime_BB_RSI",
            "trend": _make_bb, "range": _make_rsi,
            "adx_t": 25, "adx_r": 18,
            "vol_filter": True, "vol_mult": 2.0,
            "bb_squeeze": True, "bb_pctile": 20,
        },
        {
            "name": "MLRegime_BB_RSI_strict",
            "trend": _make_bb, "range": _make_rsi,
            "adx_t": 30, "adx_r": 15,
            "vol_filter": True, "vol_mult": 1.5,
            "bb_squeeze": True, "bb_pctile": 25,
        },
        {
            "name": "MLRegime_BB_RSI_loose",
            "trend": _make_bb, "range": _make_rsi,
            "adx_t": 22, "adx_r": 18,
            "vol_filter": False, "vol_mult": 2.0,
            "bb_squeeze": False, "bb_pctile": 20,
        },
        {
            "name": "MLRegime_BB_VWAP",
            "trend": _make_bb, "range": _make_vwap,
            "adx_t": 25, "adx_r": 18,
            "vol_filter": True, "vol_mult": 2.0,
            "bb_squeeze": True, "bb_pctile": 20,
        },
        {
            "name": "MLRegime_NoVol",
            "trend": _make_bb, "range": _make_rsi,
            "adx_t": 25, "adx_r": 18,
            "vol_filter": False, "vol_mult": 2.0,
            "bb_squeeze": True, "bb_pctile": 20,
        },
    ]

    for cfg in configs:
        logger.info("  --- %s (5w) ---", cfg["name"])

        def make_factory(c=cfg):
            def factory():
                return MLRegimeStrategy(
                    trend_strategy=c["trend"](),
                    range_strategy=c["range"](),
                    adx_trend_threshold=c["adx_t"],
                    adx_range_threshold=c["adx_r"],
                    volatility_filter=c["vol_filter"],
                    vol_extreme_mult=c["vol_mult"],
                    bb_squeeze_regime=c["bb_squeeze"],
                    bb_squeeze_pctile=c["bb_pctile"],
                )
            return factory

        factory_fn = make_factory(cfg)
        wf = WalkForwardAnalyzer(n_windows=5, engine=engine_mr)
        rpt = wf.run(factory_fn, df_1h)
        full = log_wf(cfg["name"], rpt, engine_mr, factory_fn, df_1h)
        all_results.append((cfg["name"], rpt, full))
        logger.info("")

    # Best MLRegime + MTF
    best_ml = max(
        [(n, r, f) for n, r, f in all_results if n.startswith("MLRegime")],
        key=lambda x: x[1].robustness_score * 100 + x[1].oos_total_return * 0.1,
    )
    best_name = best_ml[0]
    logger.info("  Best MLRegime: %s — OOS %+.2f%%, Robustness %d%%",
                best_name, best_ml[1].oos_total_return,
                int(best_ml[1].robustness_score * 100))
    logger.info("")

    # Find the config for best
    best_cfg = next(c for c in configs if c["name"] == best_name)

    def best_ml_mtf_factory(c=best_cfg):
        def factory():
            base = MLRegimeStrategy(
                trend_strategy=c["trend"](),
                range_strategy=c["range"](),
                adx_trend_threshold=c["adx_t"],
                adx_range_threshold=c["adx_r"],
                volatility_filter=c["vol_filter"],
                vol_extreme_mult=c["vol_mult"],
                bb_squeeze_regime=c["bb_squeeze"],
                bb_squeeze_pctile=c["bb_pctile"],
            )
            return MultiTimeframeFilter(base)
        return factory

    mtf_factory = best_ml_mtf_factory(best_cfg)
    logger.info("  --- %s+MTF (5w) ---", best_name)
    wf_mtf = WalkForwardAnalyzer(n_windows=5, engine=engine_mr)
    rpt = wf_mtf.run(mtf_factory, df_1h, htf_df=df_4h)
    full = log_wf(f"{best_name}+MTF", rpt, engine_mr, mtf_factory, df_1h, df_4h)
    all_results.append((f"{best_name}+MTF", rpt, full))
    logger.info("")

    # ═══════════════════════════════════════════════════════════════════
    #   7-window Validation for best strategies
    # ═══════════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  7-window WF Validation")
    logger.info("─" * 72)
    logger.info("")

    # Regime_ADX20 7w
    logger.info("  --- Regime_ADX20 (7w) ---")
    wf7 = WalkForwardAnalyzer(n_windows=7, engine=engine_mr)
    rpt7_adx = wf7.run(regime_adx20_factory, df_1h)
    log_wf("Regime_ADX20_7w", rpt7_adx, engine_mr, regime_adx20_factory, df_1h)
    logger.info("")

    # Best MLRegime 7w
    def best_ml_factory(c=best_cfg):
        def factory():
            return MLRegimeStrategy(
                trend_strategy=c["trend"](),
                range_strategy=c["range"](),
                adx_trend_threshold=c["adx_t"],
                adx_range_threshold=c["adx_r"],
                volatility_filter=c["vol_filter"],
                vol_extreme_mult=c["vol_mult"],
                bb_squeeze_regime=c["bb_squeeze"],
                bb_squeeze_pctile=c["bb_pctile"],
            )
        return factory

    logger.info("  --- %s (7w) ---", best_name)
    wf7ml = WalkForwardAnalyzer(n_windows=7, engine=engine_mr)
    rpt7_ml = wf7ml.run(best_ml_factory(best_cfg), df_1h)
    log_wf(f"{best_name}_7w", rpt7_ml, engine_mr, best_ml_factory(best_cfg), df_1h)
    logger.info("")

    # ═══════════════════════════════════════════════════════════════════
    #   Final Ranking
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  REGIME STRATEGY RANKING (5w)")
    logger.info("=" * 72)
    logger.info("")

    all_results.sort(
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    )

    logger.info(
        "  %-35s %8s %6s %7s %8s %6s %5s",
        "Strategy", "OOS Ret", "WF Rob", "Tr(OOS)",
        "Full Ret", "MaxDD", "PF",
    )
    logger.info("  " + "-" * 80)

    for name, rpt, r_full in all_results:
        logger.info(
            "  %-35s %+7.2f%% %5.0f%% %6d   %+7.2f%% %5.1f%% %5.2f",
            name, rpt.oos_total_return, rpt.robustness_score * 100,
            rpt.oos_total_trades,
            r_full.total_return, r_full.max_drawdown, r_full.profit_factor,
        )

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 10c complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    run()

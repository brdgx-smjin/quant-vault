#!/usr/bin/env python3
"""Phase 10b — MACD Momentum re-test after column name fix."""

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
from src.strategy.macd_momentum import MACDMomentumStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.bb_squeeze_v2 import BBSqueezeV2Strategy

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase10b")
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
        logger.info("  W%d: %s → Port %+5.2f%%",
                     w_idx + 1, " + ".join(label_parts), weighted_return)

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


def run():
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 10b — MACD Momentum (column name fix)")
    logger.info("=" * 72)
    logger.info("")

    df_1h = load_data("1h")
    df_4h = load_data("4h")

    # Verify MACD columns exist
    macd_cols = [c for c in df_1h.columns if "MACD" in c]
    logger.info("MACD columns found: %s", macd_cols)
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48)

    macd_configs = [
        # (name, factory, uses_mtf)
        ("MACD_ZeroCross",
         lambda: MACDMomentumStrategy(require_zero_cross=True, rsi_guard=70,
                                       atr_sl_mult=2.0, rr_ratio=2.0, cooldown_bars=6),
         False),
        ("MACD_NoZero",
         lambda: MACDMomentumStrategy(require_zero_cross=False, rsi_guard=70,
                                       atr_sl_mult=2.0, rr_ratio=2.0, cooldown_bars=6),
         False),
        ("MACD_ZeroCross+MTF",
         lambda: MultiTimeframeFilter(
             MACDMomentumStrategy(require_zero_cross=True, rsi_guard=70,
                                   atr_sl_mult=2.0, rr_ratio=2.0, cooldown_bars=6)),
         True),
        ("MACD_NoZero+MTF",
         lambda: MultiTimeframeFilter(
             MACDMomentumStrategy(require_zero_cross=False, rsi_guard=70,
                                   atr_sl_mult=2.0, rr_ratio=2.0, cooldown_bars=6)),
         True),
        ("MACD_Tight(8_21)",
         lambda: MACDMomentumStrategy(macd_fast=8, macd_slow=21, macd_signal=9,
                                       require_zero_cross=True, rsi_guard=65,
                                       atr_sl_mult=2.5, rr_ratio=2.5, cooldown_bars=8),
         False),
        ("MACD_Tight(8_21)+MTF",
         lambda: MultiTimeframeFilter(
             MACDMomentumStrategy(macd_fast=8, macd_slow=21, macd_signal=9,
                                   require_zero_cross=True, rsi_guard=65,
                                   atr_sl_mult=2.5, rr_ratio=2.5, cooldown_bars=8)),
         True),
        # Relaxed RSI guard
        ("MACD_Relaxed",
         lambda: MACDMomentumStrategy(require_zero_cross=False, rsi_guard=80,
                                       atr_sl_mult=2.0, rr_ratio=2.0, cooldown_bars=4,
                                       vol_mult=0.5),
         False),
        ("MACD_Relaxed+MTF",
         lambda: MultiTimeframeFilter(
             MACDMomentumStrategy(require_zero_cross=False, rsi_guard=80,
                                   atr_sl_mult=2.0, rr_ratio=2.0, cooldown_bars=4,
                                   vol_mult=0.5)),
         True),
    ]

    best_macd_report = None
    best_macd_name = ""
    best_macd_factory = None
    best_macd_score = -999.0
    all_results = []

    for name, factory_fn, uses_mtf in macd_configs:
        logger.info("  --- %s ---", name)
        htf = df_4h if uses_mtf else None

        # For tight MACD with different params, need custom indicators
        if "8_21" in name:
            # Check if MACD_8_21_9 columns exist, if not compute them
            col_macd_custom = "MACD_8_21_9"
            if col_macd_custom not in df_1h.columns:
                import pandas_ta as pta
                macd_custom = pta.macd(df_1h["close"], fast=8, slow=21, signal=9)
                if macd_custom is not None:
                    for c in macd_custom.columns:
                        df_1h[c] = macd_custom[c]
                logger.info("  Added MACD_8_21_9 columns")

        wf = WalkForwardAnalyzer(n_windows=5, engine=engine_1h)
        rpt = wf.run(factory_fn, df_1h, htf_df=htf)
        full = log_wf(name, rpt, engine_1h, factory_fn, df_1h, htf)
        all_results.append((name, rpt, full))

        score = rpt.robustness_score * 100 + rpt.oos_total_return * 0.1
        if score > best_macd_score:
            best_macd_score = score
            best_macd_report = rpt
            best_macd_name = name
            best_macd_factory = factory_fn

        logger.info("")

    logger.info("  Best MACD: %s — OOS %+.2f%%, Robustness %d%%",
                best_macd_name,
                best_macd_report.oos_total_return if best_macd_report else 0,
                int(best_macd_report.robustness_score * 100) if best_macd_report else 0)
    logger.info("")

    # 7w for best MACD if it has decent performance
    if best_macd_report and best_macd_report.robustness_score >= 0.4:
        logger.info("  --- %s (7w) ---", best_macd_name)
        uses_mtf = "MTF" in best_macd_name
        wf7 = WalkForwardAnalyzer(n_windows=7, engine=engine_1h)
        rpt_7w = wf7.run(best_macd_factory, df_1h, htf_df=df_4h if uses_mtf else None)
        log_wf(f"{best_macd_name}_7w", rpt_7w, engine_1h, best_macd_factory,
               df_1h, df_4h if uses_mtf else None)
        logger.info("")

        # Portfolio: BB+MACD
        if rpt_7w.robustness_score >= 0.4:
            # Also get BB baseline for 7w portfolio
            def bb_mtf_factory():
                base = BBSqueezeV2Strategy(
                    squeeze_pctile=25, vol_mult=1.1, atr_sl_mult=3.0,
                    rr_ratio=2.0, cooldown_bars=6,
                )
                return MultiTimeframeFilter(base)

            wf7_bb = WalkForwardAnalyzer(n_windows=7, engine=engine_1h)
            rpt_bb_7w = wf7_bb.run(bb_mtf_factory, df_1h, htf_df=df_4h)

            logger.info("  --- 50%% BB + 50%% %s (7w) ---", best_macd_name)
            compute_portfolio(f"BB+MACD_7w", [
                (rpt_bb_7w, 0.5, "BB"),
                (rpt_7w, 0.5, "MACD"),
            ])
            logger.info("")

    # MACD ranking
    logger.info("=" * 72)
    logger.info("  MACD RANKING (5w)")
    logger.info("=" * 72)
    logger.info("")
    logger.info(
        "  %-30s %8s %6s %7s %8s %6s %5s %5s",
        "Strategy", "OOS Ret", "WF Rob", "Tr(OOS)",
        "Full Ret", "MaxDD", "PF", "Shp",
    )
    logger.info("  " + "-" * 85)

    all_results.sort(
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    )

    for name, rpt, r_full in all_results:
        logger.info(
            "  %-30s %+7.2f%% %5.0f%% %6d   %+7.2f%% %5.1f%% %5.2f %5.2f",
            name, rpt.oos_total_return, rpt.robustness_score * 100,
            rpt.oos_total_trades,
            r_full.total_return, r_full.max_drawdown,
            r_full.profit_factor, r_full.sharpe_ratio,
        )

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 10b complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    run()

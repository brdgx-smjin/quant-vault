#!/usr/bin/env python3
"""Walk-Forward grid test for 4h BTC/USDT:USDT — ATR SL x MTF filter.

Same grid as 30m but on 4h data:
- ATR SL multiplier: [1.5, 2.0, 2.5]
- Multi-TF filter: [off, on]  (MTF uses 1d EMA trend for 4h)
Each combination runs WF with 3 and 5 windows.

4h parameters:
- max_hold_bars=48 (8 days)
- HTF = 1d (daily) for MTF filter
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DATA_DIR, SYMBOL
from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.data.preprocessor import DataPreprocessor
from src.indicators.basic import BasicIndicators
from src.monitoring.logger import setup_logging
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter

logger = setup_logging("wf_4h_grid")

# Configure root logger for sub-module visibility
root = logging.getLogger()
root.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_h = logging.StreamHandler(sys.stdout)
console_h.setFormatter(fmt)
root.addHandler(console_h)
file_h = logging.FileHandler("logs/wf_4h_grid.log")
file_h.setFormatter(fmt)
root.addHandler(file_h)
logger.propagate = False

# Grid parameters
ATR_SL_MULTS = [1.5, 2.0, 2.5]
MTF_OPTIONS = [False, True]


def print_report(label: str, report) -> None:
    """Log Walk-Forward report summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  %s", label)
    logger.info("=" * 60)

    for w in report.windows:
        logger.info(
            "  Window %d: IS %+.2f%% (trades=%d) | OOS %+.2f%% (trades=%d)",
            w.window_id,
            w.in_sample.total_return, w.in_sample.total_trades,
            w.out_of_sample.total_return, w.out_of_sample.total_trades,
        )

    logger.info("-" * 60)
    logger.info("  OOS Total Return: %+.2f%%", report.oos_total_return)
    logger.info("  OOS Avg Return:   %+.2f%%", report.oos_avg_return)
    logger.info("  OOS Avg Sharpe:   %.2f", report.oos_avg_sharpe)
    logger.info("  OOS Avg Win Rate: %.1f%%", report.oos_avg_win_rate * 100)
    logger.info("  OOS Total Trades: %d", report.oos_total_trades)
    logger.info("  Profitable Windows: %d / %d", report.oos_profitable_windows, report.total_windows)
    logger.info("  Robustness Score: %.0f%%", report.robustness_score * 100)
    logger.info("=" * 60)


def main() -> None:
    """Run Walk-Forward grid on 4h data."""
    parquet_path = f"{DATA_DIR}/processed/{SYMBOL.replace('/', '_').replace(':', '_')}_4h.parquet"
    if not Path(parquet_path).exists():
        logger.error("4h data not found at %s", parquet_path)
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d bars from %s", len(df), parquet_path)
    logger.info("Date range: %s ~ %s", df.index[0], df.index[-1])

    # Add indicators to 4h
    df = BasicIndicators.add_all(df)
    df = df.dropna()
    logger.info("After indicators + dropna: %d bars", len(df))

    # Resample 4h -> 1d for MTF filter
    df_1d = DataPreprocessor.resample(df, "1d")
    df_1d = BasicIndicators.add_all(df_1d)
    df_1d = df_1d.dropna()
    logger.info("1d resampled: %d bars (%s ~ %s)",
                len(df_1d), df_1d.index[0], df_1d.index[-1])

    engine = BacktestEngine(initial_capital=10_000, max_hold_bars=48)

    results: list[dict] = []

    for atr_mult in ATR_SL_MULTS:
        for use_mtf in MTF_OPTIONS:
            mtf_label = "MTF_ON" if use_mtf else "MTF_OFF"
            combo_label = f"ATR_{atr_mult}x_{mtf_label}"

            logger.info("")
            logger.info("#" * 60)
            logger.info("  COMBO: %s", combo_label)
            logger.info("#" * 60)

            def make_factory(mult: float, mtf: bool):
                def factory() -> FibonacciRetracementStrategy | MultiTimeframeFilter:
                    base = FibonacciRetracementStrategy(
                        entry_levels=(0.5, 0.618),
                        tolerance_pct=0.05,
                        lookback=50,
                        atr_sl_mult=mult,
                        require_trend=True,
                    )
                    if mtf:
                        return MultiTimeframeFilter(base)
                    return base
                return factory

            factory = make_factory(atr_mult, use_mtf)
            htf_arg = df_1d if use_mtf else None

            # WF 3 windows
            wf3 = WalkForwardAnalyzer(train_ratio=0.7, n_windows=3, engine=engine)
            report_3w = wf3.run(factory, df, htf_df=htf_arg)
            print_report(f"{combo_label} — 3 Windows", report_3w)

            # WF 5 windows
            wf5 = WalkForwardAnalyzer(train_ratio=0.7, n_windows=5, engine=engine)
            report_5w = wf5.run(factory, df, htf_df=htf_arg)
            print_report(f"{combo_label} — 5 Windows", report_5w)

            best = report_3w if report_3w.robustness_score >= report_5w.robustness_score else report_5w
            best_nw = "3w" if best is report_3w else "5w"

            results.append({
                "combo": combo_label,
                "atr_mult": atr_mult,
                "mtf": use_mtf,
                "best_nw": best_nw,
                "robustness": best.robustness_score,
                "oos_return": best.oos_total_return,
                "oos_sharpe": best.oos_avg_sharpe,
                "oos_trades": best.oos_total_trades,
                "oos_wr": best.oos_avg_win_rate,
            })

    # Final summary table
    logger.info("")
    logger.info("=" * 80)
    logger.info("  GRID RESULTS SUMMARY (4h)")
    logger.info("=" * 80)
    logger.info(
        "  %-25s  %4s  %10s  %10s  %8s  %6s  %6s",
        "Combo", "NW", "Robustness", "OOS Return", "Sharpe", "Trades", "WR",
    )
    logger.info("  " + "-" * 75)

    best_result = None
    for r in results:
        logger.info(
            "  %-25s  %4s  %9.0f%%  %+9.2f%%  %8.2f  %6d  %5.1f%%",
            r["combo"], r["best_nw"],
            r["robustness"] * 100, r["oos_return"],
            r["oos_sharpe"], r["oos_trades"], r["oos_wr"] * 100,
        )
        if best_result is None or r["robustness"] > best_result["robustness"]:
            best_result = r
        elif r["robustness"] == best_result["robustness"] and r["oos_return"] > best_result["oos_return"]:
            best_result = r

    logger.info("  " + "-" * 75)
    if best_result:
        logger.info(
            "  BEST: %s (%s) | Robustness %.0f%% | OOS %+.2f%%",
            best_result["combo"], best_result["best_nw"],
            best_result["robustness"] * 100, best_result["oos_return"],
        )
        logger.info("")
        logger.info("  Recommended settings:")
        logger.info("    atr_sl_mult=%.1f", best_result["atr_mult"])
        logger.info("    use_mtf=%s", best_result["mtf"])
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

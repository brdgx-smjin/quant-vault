#!/usr/bin/env python3
"""Run comprehensive backtest: individual strategies + ensemble."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DATA_DIR, SYMBOL
from src.backtest.engine import BacktestEngine, BacktestResult
from src.indicators.basic import BasicIndicators
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategy.ema_trend import EMATrendStrategy
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.ensemble import EnsembleStrategy
from src.monitoring.logger import setup_logging

logger = setup_logging("backtest")

SYMBOL_FILE = SYMBOL.replace("/", "_").replace(":", "_")


def load_data(timeframe: str) -> pd.DataFrame:
    path = f"{DATA_DIR}/processed/{SYMBOL_FILE}_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


def print_result(name: str, tf: str, r: BacktestResult) -> None:
    logger.info("─" * 60)
    logger.info("  %-30s | %s", name, tf)
    logger.info("─" * 60)
    logger.info("  Total Return:   %+8.2f%%", r.total_return)
    logger.info("  Sharpe Ratio:   %8.2f", r.sharpe_ratio)
    logger.info("  Max Drawdown:   %8.2f%%", r.max_drawdown)
    logger.info("  Win Rate:       %7.1f%%  (%d trades)", r.win_rate * 100, r.total_trades)
    logger.info("  Profit Factor:  %8.2f", r.profit_factor)
    logger.info("  Avg Trade PnL:  $%7.2f", r.avg_trade_return)
    logger.info("")


def make_strategies():
    """Create fresh strategy instances (resets internal state)."""
    fib = FibonacciRetracementStrategy(
        entry_levels=(0.5, 0.618),
        tolerance_pct=0.05,
        lookback=50,
        require_trend=True,
    )
    ema = EMATrendStrategy(
        fast_ema=20,
        slow_ema=50,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        require_pullback=True,
    )
    rsi_mr = RSIMeanReversionStrategy(
        rsi_oversold=35.0,
        rsi_overbought=65.0,
        atr_sl_mult=1.5,
        atr_tp_mult=2.5,
    )
    ensemble = EnsembleStrategy(
        strategies=[
            (ema, 0.40),
            (fib, 0.35),
            (rsi_mr, 0.25),
        ],
        threshold=0.12,
    )
    return [
        ("Fib+TrendFilter", fib),
        ("EMA_Trend_20/50", ema),
        ("RSI_MeanRev", rsi_mr),
        ("Ensemble(40/35/25)", ensemble),
    ]


def main() -> None:
    engine = BacktestEngine(initial_capital=10_000, max_hold_bars=48)
    timeframes = ["15m", "1h", "4h"]

    logger.info("=" * 60)
    logger.info("  PHASE 2 BACKTEST REPORT — %s", SYMBOL)
    logger.info("  Capital: $10,000 | Fee: 0.04%% | Max Hold: 48 bars")
    logger.info("=" * 60)
    logger.info("")

    summary: list[dict] = []

    for tf in timeframes:
        try:
            df = load_data(tf)
            logger.info("Loaded %s: %d bars (%s ~ %s)",
                        tf, len(df), df.index[0].date(), df.index[-1].date())
            logger.info("")
        except FileNotFoundError:
            logger.warning("No data for %s, skipping", tf)
            continue

        # Fresh strategy instances per timeframe (resets cooldown state)
        all_strategies = make_strategies()

        for name, strategy in all_strategies:
            logger.info("Running %s on %s ...", name, tf)
            result = engine.run(strategy, df)
            print_result(name, tf, result)
            summary.append({
                "strategy": name,
                "tf": tf,
                "return": result.total_return,
                "sharpe": result.sharpe_ratio,
                "max_dd": result.max_drawdown,
                "win_rate": result.win_rate,
                "trades": result.total_trades,
                "pf": result.profit_factor,
            })

    # --- Summary table ---
    logger.info("=" * 60)
    logger.info("  RANKING (by Sharpe Ratio)")
    logger.info("=" * 60)
    summary.sort(key=lambda x: x["sharpe"], reverse=True)
    logger.info("  %-26s %4s %8s %7s %6s %5s %5s",
                "Strategy", "TF", "Return", "Sharpe", "MaxDD", "WR%", "#")
    logger.info("  " + "-" * 56)
    for s in summary:
        logger.info("  %-26s %4s %+7.1f%% %7.2f %5.1f%% %4.1f%% %5d",
                    s["strategy"], s["tf"], s["return"], s["sharpe"],
                    s["max_dd"], s["win_rate"] * 100, s["trades"])

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Backtest complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

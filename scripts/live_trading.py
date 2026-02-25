#!/usr/bin/env python3
"""Live paper trading bot entry point.

Runs the TradingEngine with Fib+TrendFilter strategy on Binance testnet.
Fib+TrendFilter was selected as it has the highest Walk-Forward robustness (67%).
Hard safety abort if TESTNET=false to prevent accidental real-money trading.
"""

import os
import asyncio
import logging as _logging
import signal
import sys
from pathlib import Path

os.environ.setdefault("SSL_CERT_FILE", str(Path(__file__).resolve().parent.parent / ".venv/lib/python3.12/site-packages/certifi/cacert.pem"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import SYMBOL, TESTNET
from src.execution.trading_engine import TradingEngine
from src.monitoring.logger import setup_logging
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter

logger = setup_logging("live")

# Configure root logger so all sub-module logs (src.execution, src.data, etc.)
# are captured with the same format and go to console + file
root = _logging.getLogger()
root.setLevel(_logging.INFO)
fmt = _logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_h = _logging.StreamHandler(sys.stdout)
console_h.setFormatter(fmt)
root.addHandler(console_h)
file_h = _logging.FileHandler("logs/live.log")
file_h.setFormatter(fmt)
root.addHandler(file_h)
# Prevent duplicate output on the 'live' logger
logger.propagate = False


def build_strategy() -> MultiTimeframeFilter:
    """Build the Fib+TrendFilter strategy with 4h MTF filter.

    Returns:
        MultiTimeframeFilter wrapping FibonacciRetracementStrategy.
    """
    base = FibonacciRetracementStrategy(
        entry_levels=(0.5, 0.618),
        tolerance_pct=0.05,
        lookback=50,
        atr_sl_mult=2.0,
        require_trend=True,
    )
    strategy = MultiTimeframeFilter(base)
    logger.info("Strategy: %s (ATR SL=2.0x, 4h MTF filter)", strategy.name)
    return strategy


async def main() -> None:
    """Main live trading entry point."""
    # Hard safety check
    if not TESTNET:
        logger.critical("ABORT: BINANCE_TESTNET is not 'true'!")
        logger.critical("This script is for paper trading only.")
        logger.critical("Set BINANCE_TESTNET=true in .env and retry.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  LIVE PAPER TRADING — %s | 30m + 4h MTF", SYMBOL)
    logger.info("  Mode: TESTNET (paper trading)")
    logger.info("  Strategy: Fib+TrendFilter + MTF (ATR SL=2.0x)")
    logger.info("=" * 60)
    logger.info("")

    strategy = build_strategy()
    engine = TradingEngine(
        strategy=strategy,
        symbol=SYMBOL,
        timeframe="30m",
        testnet=True,
        warmup_bars=200,
        htf_timeframe="4h",
    )

    # Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def handle_signal(sig: int) -> None:
        sig_name = signal.Signals(sig).name
        logger.info("Received %s — initiating graceful shutdown...", sig_name)
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal, sig)

    try:
        await engine.initialize()

        # Run engine and shutdown watcher concurrently
        engine_task = asyncio.create_task(engine.run())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            [engine_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

    except Exception:
        logger.exception("Fatal error in trading engine")
    finally:
        await engine.shutdown()
        logger.info("Live trading bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Live paper trading bot entry point.

Runs the TradingEngine with 15m Cross-TF portfolio on Binance testnet.
Phase 16 validated (9w): 89% robustness (8/9 windows), OOS +19.61%.
  - 1h RSI_35_65+MTF(4h):  mean reversion (66% robustness at 9w)  → 33%
  - 1h DC_24+MTF(4h):      trend following (55% robustness at 9w) → 33%
  - 15m RSI_35_65+MTF(4h): mean reversion (77% robustness at 9w) → 34%
  - Cross-TF: decorrelates negative windows (15m W5 vs 1h W6) → 89%
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
from src.strategy.cross_tf_portfolio import CrossTimeframePortfolio
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy

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


def build_strategy() -> CrossTimeframePortfolio:
    """Build cross-TF portfolio: 1h RSI + 1h DC + 15m RSI, all with 4h MTF.

    Phase 16 validated (9w): 89% robustness, OOS +19.61%, only W2 negative.

    Returns:
        CrossTimeframePortfolio with 33/33/34 weights.
    """
    # 1h RSI Mean Reversion + MTF(4h)
    rsi_1h = RSIMeanReversionStrategy(
        rsi_oversold=35,
        rsi_overbought=65,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=6,
    )
    rsi_1h_mtf = MultiTimeframeFilter(rsi_1h)

    # 1h Donchian Trend Following + MTF(4h)
    dc_1h = DonchianTrendStrategy(
        entry_period=24,
        atr_sl_mult=2.0,
        rr_ratio=2.0,
        vol_mult=0.8,
        cooldown_bars=6,
    )
    dc_1h_mtf = MultiTimeframeFilter(dc_1h)

    # 15m RSI Mean Reversion + MTF(4h)
    rsi_15m = RSIMeanReversionStrategy(
        rsi_oversold=35,
        rsi_overbought=65,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=12,
    )
    rsi_15m_mtf = MultiTimeframeFilter(rsi_15m)

    # Cross-TF portfolio: 33/33/34 weights
    portfolio = CrossTimeframePortfolio(
        strategies_15m=[(rsi_15m_mtf, 0.34)],
        strategies_1h=[(rsi_1h_mtf, 0.33), (dc_1h_mtf, 0.33)],
    )

    logger.info("Strategy: %s", portfolio.name)
    logger.info("  1h RSI:  35/65, SL=2.0ATR, TP=3.0ATR, cool=6 + 4h MTF (33%%)")
    logger.info("  1h DC:   24-bar, SL=2.0ATR, RR=2.0, vol=0.8x + 4h MTF (33%%)")
    logger.info("  15m RSI: 35/65, SL=2.0ATR, TP=3.0ATR, cool=12 + 4h MTF (34%%)")
    logger.info("  Allocation: 33/33/34 (cross-TF)")
    return portfolio


async def main() -> None:
    """Main live trading entry point."""
    # Hard safety check
    if not TESTNET:
        logger.critical("ABORT: BINANCE_TESTNET is not 'true'!")
        logger.critical("This script is for paper trading only.")
        logger.critical("Set BINANCE_TESTNET=true in .env and retry.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  LIVE PAPER TRADING — %s | 15m + 1h Cross-TF", SYMBOL)
    logger.info("  Mode: TESTNET (paper trading)")
    logger.info("  Strategy: Cross-TF Portfolio (1hRSI + 1hDC + 15mRSI)")
    logger.info("  Phase 16: 89%% robustness, OOS +19.61%%")
    logger.info("=" * 60)
    logger.info("")

    strategy = build_strategy()
    engine = TradingEngine(
        strategy=strategy,
        symbol=SYMBOL,
        timeframe="15m",
        testnet=True,
        warmup_bars=2000,
        htf_timeframe="1h",
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

#!/usr/bin/env python3
"""Live paper trading bot entry point.

Runs the TradingEngine with 15m Cross-TF portfolio on Binance testnet.
Phase 25 validated (9w): 88% robustness (8/9 windows), OOS +23.98%.
  - 1h RSI_35_65+MTF(4h):  mean reversion  → 15%
  - 1h DC_24+MTF(4h):      trend following  → 50%
  - 15m RSI_35_65+MTF(4h): mean reversion   → 10%
  - 1h WillR_14_90+MTF(4h): mean reversion  → 25%
  - Cross-TF: decorrelates negative windows → 88%
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
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy

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
    """Build cross-TF portfolio: 1h RSI + 1h DC + 15m RSI + 1h WillR, all with 4h MTF.

    Phase 41: MTF switched from EMA_20 vs EMA_50 to close > EMA_20 (price_vs_ema).
    Faster trend reaction: 62K→73K rally no longer stuck in BEARISH for 2 weeks.
    WF result: 77% rob, +21.39% OOS (vs +12.55% baseline), W2 nearly breakeven.

    Returns:
        CrossTimeframePortfolio with 15/50/10/25 weights.
    """
    # 1h RSI Mean Reversion + MTF(4h) — 15%
    rsi_1h = RSIMeanReversionStrategy(
        rsi_oversold=35,
        rsi_overbought=65,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=6,
    )
    rsi_1h_mtf = MultiTimeframeFilter(rsi_1h, trend_mode="price_vs_ema")

    # 1h Donchian Trend Following + MTF(4h) — 50%
    dc_1h = DonchianTrendStrategy(
        entry_period=24,
        atr_sl_mult=2.0,
        rr_ratio=2.0,
        vol_mult=0.8,
        cooldown_bars=6,
    )
    dc_1h_mtf = MultiTimeframeFilter(dc_1h, trend_mode="price_vs_ema")

    # 15m RSI Mean Reversion + MTF(4h) — 10%
    rsi_15m = RSIMeanReversionStrategy(
        rsi_oversold=35,
        rsi_overbought=65,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=12,
    )
    rsi_15m_mtf = MultiTimeframeFilter(rsi_15m, trend_mode="price_vs_ema")

    # 1h Williams %R Mean Reversion + MTF(4h) — 25%
    willr_1h = WilliamsRMeanReversionStrategy(
        willr_period=14,
        oversold_level=90.0,
        overbought_level=90.0,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        cooldown_bars=6,
    )
    willr_1h_mtf = MultiTimeframeFilter(willr_1h, trend_mode="price_vs_ema")

    # Cross-TF portfolio: 15/50/10/25 weights (Phase 25 optimal)
    portfolio = CrossTimeframePortfolio(
        strategies_15m=[(rsi_15m_mtf, 0.10)],
        strategies_1h=[(rsi_1h_mtf, 0.15), (dc_1h_mtf, 0.50), (willr_1h_mtf, 0.25)],
    )

    logger.info("Strategy: %s", portfolio.name)
    logger.info("  1h RSI:  35/65, SL=2.0ATR, TP=3.0ATR, cool=6 + 4h MTF (15%%)")
    logger.info("  1h DC:   24-bar, SL=2.0ATR, RR=2.0, vol=0.8x + 4h MTF (50%%)")
    logger.info("  15m RSI: 35/65, SL=2.0ATR, TP=3.0ATR, cool=12 + 4h MTF (10%%)")
    logger.info("  1h WillR: p14/t90, SL=2.0ATR, TP=3.0ATR, cool=6 + 4h MTF (25%%)")
    logger.info("  Allocation: 15/50/10/25 (Phase 25 optimal)")
    logger.info("  MTF: price_vs_ema (close > EMA_20) — Phase 41")
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
    logger.info("  Strategy: 4-comp Cross-TF (1hRSI + 1hDC + 15mRSI + 1hWillR)")
    logger.info("  Phase 25: 88%% robustness, OOS +23.98%%")
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

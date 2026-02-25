#!/usr/bin/env python3
"""Fetch 30m historical OHLCV data for BTC/USDT:USDT from Binance.

Collects ~1 year of 30m candles (~17,520 bars) and saves as parquet.
Uses BinanceCollector.fetch_historical() with 1500-bar pagination.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import SYMBOL, DATA_DIR
from src.data.collector import BinanceCollector
from src.data.preprocessor import DataPreprocessor
from src.monitoring.logger import setup_logging

logger = setup_logging("fetch_30m")

TIMEFRAME = "30m"
DAYS = 365


async def main() -> None:
    """Fetch 30m historical data and save to parquet."""
    collector = BinanceCollector()
    preprocessor = DataPreprocessor()

    Path(f"{DATA_DIR}/processed").mkdir(parents=True, exist_ok=True)

    logger.info("Fetching %s %s data (%d days)...", SYMBOL, TIMEFRAME, DAYS)

    try:
        df = await collector.fetch_historical(timeframe=TIMEFRAME, days=DAYS)
        df = preprocessor.clean_ohlcv(df)
        df = preprocessor.add_returns(df)

        path = f"{DATA_DIR}/processed/{SYMBOL.replace('/', '_').replace(':', '_')}_{TIMEFRAME}.parquet"
        df.to_parquet(path)
        logger.info("Saved %d bars to %s", len(df), path)
        logger.info("Date range: %s ~ %s", df.index[0], df.index[-1])
    except Exception:
        logger.exception("Failed to fetch %s data", TIMEFRAME)
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Collect BTC/USDT derivatives data: funding rate and open interest.

Usage:
    # Collect both funding rate and open interest
    python scripts/collect_derivatives.py

    # Funding rate only
    python scripts/collect_derivatives.py --funding-rate

    # Open interest only
    python scripts/collect_derivatives.py --open-interest

    # Incremental update (default if files exist)
    python scripts/collect_derivatives.py

    # Full re-collection for N days
    python scripts/collect_derivatives.py --full --days 730
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR, SYMBOL
from src.data.collector import BinanceCollector
from src.monitoring.logger import setup_logging

logger = setup_logging("data_engineer")

FUNDING_RATE_PATH = f"{DATA_DIR}/processed/{SYMBOL.replace('/', '_').replace(':', '_')}_funding_rate.parquet"
OI_PATH = f"{DATA_DIR}/processed/{SYMBOL.replace('/', '_').replace(':', '_')}_open_interest_1h.parquet"


async def collect_funding_rate(
    collector: BinanceCollector, days: int, incremental: bool
) -> bool:
    """Collect funding rate data.

    Returns:
        True if successful, False otherwise.
    """
    logger.info("=== Collecting funding rate data ===")
    try:
        existing = pd.DataFrame()
        since_ms = None

        if incremental and Path(FUNDING_RATE_PATH).exists():
            existing = pd.read_parquet(FUNDING_RATE_PATH)
            logger.info(
                "[BEFORE] funding_rate: rows=%d, start=%s, end=%s",
                len(existing), existing.index.min(), existing.index.max(),
            )
            since_ms = int(existing.index.max().timestamp() * 1000)
        else:
            since_ms = int(
                (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
            )

        new_df = await collector.fetch_funding_rate(since=since_ms)
        if new_df.empty:
            logger.info("No new funding rate data")
            return True

        logger.info("Fetched %d new funding rate records", len(new_df))

        if not existing.empty:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new_df

        combined.to_parquet(FUNDING_RATE_PATH)
        logger.info(
            "[AFTER] funding_rate: rows=%d, start=%s, end=%s",
            len(combined), combined.index.min(), combined.index.max(),
        )
        return True

    except Exception:
        logger.exception("Failed to collect funding rate data")
        return False


async def collect_open_interest(
    collector: BinanceCollector, days: int, incremental: bool
) -> bool:
    """Collect open interest data (1h timeframe).

    Returns:
        True if successful, False otherwise.
    """
    logger.info("=== Collecting open interest data (1h) ===")
    try:
        existing = pd.DataFrame()
        since_ms = None

        if incremental and Path(OI_PATH).exists():
            existing = pd.read_parquet(OI_PATH)
            logger.info(
                "[BEFORE] open_interest: rows=%d, start=%s, end=%s",
                len(existing), existing.index.min(), existing.index.max(),
            )
            since_ms = int(existing.index.max().timestamp() * 1000)
        else:
            since_ms = int(
                (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
            )

        new_df = await collector.fetch_open_interest(timeframe="1h", since=since_ms)
        if new_df.empty:
            logger.info("No new open interest data")
            return True

        logger.info("Fetched %d new open interest records", len(new_df))

        if not existing.empty:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new_df

        combined.to_parquet(OI_PATH)
        logger.info(
            "[AFTER] open_interest: rows=%d, start=%s, end=%s",
            len(combined), combined.index.min(), combined.index.max(),
        )
        return True

    except Exception:
        logger.exception("Failed to collect open interest data")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect BTC/USDT derivatives data")
    parser.add_argument("--funding-rate", action="store_true", help="Collect funding rate only")
    parser.add_argument("--open-interest", action="store_true", help="Collect open interest only")
    parser.add_argument("--full", action="store_true", help="Full re-collection (not incremental)")
    parser.add_argument("--days", type=int, default=730, help="Days of history (default: 730)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    incremental = not args.full
    collect_both = not args.funding_rate and not args.open_interest

    logger.info("Started derivatives collection at %s", datetime.now(timezone.utc).isoformat())
    logger.info("Mode: %s, Days: %d", "incremental" if incremental else "full", args.days)

    collector = BinanceCollector()
    success = True

    try:
        if collect_both or args.funding_rate:
            if not await collect_funding_rate(collector, args.days, incremental):
                success = False

        if collect_both or args.open_interest:
            if not await collect_open_interest(collector, args.days, incremental):
                success = False
    finally:
        await collector.close()

    logger.info("Finished at %s", datetime.now(timezone.utc).isoformat())
    if not success:
        logger.error("Some collections failed")
        sys.exit(1)
    else:
        logger.info("All derivatives data collected successfully")


if __name__ == "__main__":
    asyncio.run(main())

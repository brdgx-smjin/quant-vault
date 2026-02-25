#!/usr/bin/env python3
"""Historical data collection script with incremental update support.

Usage:
    # Incremental update (default): fetch only new candles since last saved timestamp
    python scripts/collect_data.py

    # Full re-collection for N days
    python scripts/collect_data.py --full --days 365

    # Extend history to 2 years
    python scripts/collect_data.py --full --days 730

    # Cron mode: minimal output, exit code 1 on error
    python scripts/collect_data.py --cron

    # Specific timeframes only
    python scripts/collect_data.py --timeframes 1h 4h
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR, SYMBOL, TIMEFRAMES
from src.data.collector import BinanceCollector
from src.data.preprocessor import DataPreprocessor, RETENTION_DAYS
from src.monitoring.logger import setup_logging


def _parquet_path(tf: str) -> str:
    return f"{DATA_DIR}/processed/{SYMBOL.replace('/', '_').replace(':', '_')}_{tf}.parquet"


def _log_stats(logger, label: str, df: pd.DataFrame, tf: str) -> None:
    if df.empty:
        logger.info("[%s] %s: empty", label, tf)
    else:
        logger.info(
            "[%s] %s: rows=%d, start=%s, end=%s",
            label, tf, len(df), df.index.min(), df.index.max(),
        )


async def incremental_update(
    collector: BinanceCollector,
    preprocessor: DataPreprocessor,
    timeframes: list[str],
    logger,
) -> bool:
    """Fetch only new candles since last saved timestamp for each timeframe.

    Returns:
        True if all timeframes updated successfully, False otherwise.
    """
    success = True
    for tf in timeframes:
        path = _parquet_path(tf)
        logger.info("=== Incremental update: %s %s ===", SYMBOL, tf)
        try:
            if Path(path).exists():
                existing = pd.read_parquet(path)
                _log_stats(logger, "BEFORE", existing, tf)
                last_ts = existing.index.max()
                since_ms = int(last_ts.timestamp() * 1000)
                logger.info("Fetching new candles since %s", last_ts)
            else:
                existing = pd.DataFrame()
                since_ms = None
                logger.info("No existing data, fetching full history (365 days)")

            if since_ms is not None:
                new_df = await collector.fetch_since(timeframe=tf, since_ms=since_ms)
            else:
                new_df = await collector.fetch_historical(timeframe=tf, days=365)

            if new_df.empty:
                logger.info("No new candles for %s", tf)
                continue

            logger.info("Fetched %d new candles for %s", len(new_df), tf)

            if not existing.empty:
                combined = pd.concat([existing.drop(columns=["pct_return", "log_return"], errors="ignore"), new_df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = new_df

            combined = preprocessor.clean_ohlcv(combined)

            # Rolling window trim
            max_days = RETENTION_DAYS.get(tf)
            if max_days:
                combined = preprocessor.trim_to_period(combined, max_days)

            combined = preprocessor.add_returns(combined)

            combined.to_parquet(path)
            _log_stats(logger, "AFTER", combined, tf)

        except Exception:
            logger.exception("Failed to update %s data", tf)
            success = False

    return success


async def full_collection(
    collector: BinanceCollector,
    preprocessor: DataPreprocessor,
    timeframes: list[str],
    days: int,
    logger,
) -> bool:
    """Full re-collection of historical data.

    Returns:
        True if all timeframes collected successfully, False otherwise.
    """
    success = True
    for tf in timeframes:
        path = _parquet_path(tf)
        logger.info("=== Full collection: %s %s (%d days) ===", SYMBOL, tf, days)
        try:
            # Log existing stats before overwrite
            if Path(path).exists():
                existing = pd.read_parquet(path)
                _log_stats(logger, "BEFORE", existing, tf)

            df = await collector.fetch_historical(timeframe=tf, days=days)
            df = preprocessor.clean_ohlcv(df)

            # Rolling window trim
            max_days = RETENTION_DAYS.get(tf)
            if max_days:
                df = preprocessor.trim_to_period(df, max_days)

            df = preprocessor.add_returns(df)

            df.to_parquet(path)
            _log_stats(logger, "AFTER", df, tf)

        except Exception:
            logger.exception("Failed to collect %s data", tf)
            success = False

    return success


async def extend_history(
    collector: BinanceCollector,
    preprocessor: DataPreprocessor,
    timeframes: list[str],
    days: int,
    logger,
) -> bool:
    """Extend existing data backwards to cover more history.

    Fetches from (now - days) and merges with existing data, keeping all candles.

    Returns:
        True if all timeframes extended successfully, False otherwise.
    """
    success = True
    for tf in timeframes:
        path = _parquet_path(tf)
        logger.info("=== Extend history: %s %s (%d days) ===", SYMBOL, tf, days)
        try:
            existing = pd.DataFrame()
            if Path(path).exists():
                existing = pd.read_parquet(path)
                _log_stats(logger, "BEFORE", existing, tf)

            df = await collector.fetch_historical(timeframe=tf, days=days)

            if not existing.empty:
                combined = pd.concat([
                    df,
                    existing.drop(columns=["pct_return", "log_return"], errors="ignore"),
                ])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = df

            combined = preprocessor.clean_ohlcv(combined)

            # Rolling window trim
            max_days = RETENTION_DAYS.get(tf)
            if max_days:
                combined = preprocessor.trim_to_period(combined, max_days)

            combined = preprocessor.add_returns(combined)

            combined.to_parquet(path)
            _log_stats(logger, "AFTER", combined, tf)

        except Exception:
            logger.exception("Failed to extend %s data", tf)
            success = False

    return success


async def collect_derivatives(
    collector: BinanceCollector,
    logger,
) -> bool:
    """Collect funding rate and open interest data.

    Returns:
        True if all derivatives data collected successfully, False otherwise.
    """
    success = True
    symbol_safe = SYMBOL.replace("/", "_").replace(":", "_")

    # --- Funding Rate ---
    fr_path = f"{DATA_DIR}/processed/{symbol_safe}_funding_rate.parquet"
    logger.info("=== Collecting funding rate data ===")
    try:
        since_ms = None
        existing_fr = pd.DataFrame()
        if Path(fr_path).exists():
            existing_fr = pd.read_parquet(fr_path)
            if not existing_fr.empty:
                since_ms = int(existing_fr.index.max().timestamp() * 1000)
                logger.info("Funding rate: existing %d rows, last=%s", len(existing_fr), existing_fr.index.max())

        new_fr = await collector.fetch_funding_rate(since=since_ms)
        if not new_fr.empty:
            if not existing_fr.empty:
                combined = pd.concat([existing_fr, new_fr])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = new_fr
            combined.to_parquet(fr_path)
            logger.info("Funding rate: saved %d rows, start=%s, end=%s",
                        len(combined), combined.index.min(), combined.index.max())
        else:
            logger.info("No new funding rate data")
    except Exception:
        logger.exception("Failed to collect funding rate data")
        success = False

    # --- Open Interest ---
    oi_path = f"{DATA_DIR}/processed/{symbol_safe}_open_interest_1h.parquet"
    logger.info("=== Collecting open interest data ===")
    try:
        since_ms = None
        existing_oi = pd.DataFrame()
        if Path(oi_path).exists():
            existing_oi = pd.read_parquet(oi_path)
            if not existing_oi.empty:
                since_ms = int(existing_oi.index.max().timestamp() * 1000)
                logger.info("Open interest: existing %d rows, last=%s", len(existing_oi), existing_oi.index.max())

        new_oi = await collector.fetch_open_interest(timeframe="1h", since=since_ms)
        if not new_oi.empty:
            if not existing_oi.empty:
                combined = pd.concat([existing_oi, new_oi])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = new_oi
            combined.to_parquet(oi_path)
            logger.info("Open interest: saved %d rows, start=%s, end=%s",
                        len(combined), combined.index.min(), combined.index.max())
        else:
            logger.info("No new open interest data")
    except Exception:
        logger.exception("Failed to collect open interest data")
        success = False

    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC/USDT OHLCV data collector")
    parser.add_argument(
        "--full", action="store_true",
        help="Full re-collection instead of incremental update",
    )
    parser.add_argument(
        "--extend", action="store_true",
        help="Extend history backwards (fetch full range and merge with existing)",
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="Number of days of history (default: 365)",
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=None,
        help="Specific timeframes to collect (default: all)",
    )
    parser.add_argument(
        "--cron", action="store_true",
        help="Cron mode: log only, exit code 1 on error",
    )
    parser.add_argument(
        "--no-derivatives", action="store_true",
        help="Skip funding rate and open interest collection",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    log_name = "data_engineer" if not args.cron else "data_engineer_cron"
    logger = setup_logging(log_name)

    if args.cron:
        # Suppress console output in cron mode
        for handler in logger.handlers:
            if hasattr(handler, "stream") and handler.stream == sys.stdout:
                handler.setLevel("WARNING")

    timeframes = args.timeframes or TIMEFRAMES
    # Validate timeframes
    for tf in timeframes:
        if tf not in TIMEFRAMES:
            logger.error("Invalid timeframe: %s (valid: %s)", tf, TIMEFRAMES)
            sys.exit(1)

    logger.info("Started at %s", datetime.now(timezone.utc).isoformat())
    logger.info("Mode: %s, Days: %d, Timeframes: %s",
                "extend" if args.extend else ("full" if args.full else "incremental"),
                args.days, timeframes)

    collector = BinanceCollector()
    preprocessor = DataPreprocessor()

    Path(f"{DATA_DIR}/processed").mkdir(parents=True, exist_ok=True)

    try:
        if args.extend:
            success = await extend_history(collector, preprocessor, timeframes, args.days, logger)
        elif args.full:
            success = await full_collection(collector, preprocessor, timeframes, args.days, logger)
        else:
            success = await incremental_update(collector, preprocessor, timeframes, logger)

        # Collect derivatives data (funding rate, open interest)
        if not args.no_derivatives:
            deriv_success = await collect_derivatives(collector, logger)
            success = success and deriv_success
    finally:
        await collector.close()

    logger.info("Finished at %s", datetime.now(timezone.utc).isoformat())

    if not success:
        logger.error("Some timeframes failed to update")
        sys.exit(1)
    else:
        logger.info("All timeframes updated successfully")


if __name__ == "__main__":
    asyncio.run(main())

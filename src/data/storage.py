"""Database storage layer for OHLCV data."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config.settings import DATABASE_URL

logger = logging.getLogger(__name__)


class OHLCVStorage:
    """Stores and retrieves OHLCV data from PostgreSQL."""

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self.database_url = database_url

    def save_ohlcv(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> None:
        """Save OHLCV DataFrame to database.

        Args:
            df: OHLCV DataFrame with timestamp index.
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
        """
        # TODO: Implement PostgreSQL/TimescaleDB storage
        # For now, save to local parquet as fallback
        from config.settings import DATA_DIR
        path = f"{DATA_DIR}/processed/{symbol.replace('/', '_')}_{timeframe}.parquet"
        df.to_parquet(path)
        logger.info("Saved %d rows to %s", len(df), path)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from storage.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
            start: Start date string.
            end: End date string.

        Returns:
            OHLCV DataFrame.
        """
        from config.settings import DATA_DIR
        path = f"{DATA_DIR}/processed/{symbol.replace('/', '_')}_{timeframe}.parquet"
        try:
            df = pd.read_parquet(path)
            if start:
                df = df[df.index >= start]
            if end:
                df = df[df.index <= end]
            return df
        except FileNotFoundError:
            logger.warning("No data file found at %s", path)
            return pd.DataFrame()

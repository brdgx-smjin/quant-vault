"""Binance OHLCV data collector using ccxt."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd

from config.settings import (
    API_KEY,
    EXCHANGE_ID,
    SECRET_KEY,
    SYMBOL,
    TESTNET,
    TIMEFRAMES,
)

logger = logging.getLogger(__name__)


class BinanceCollector:
    """Collects OHLCV data from Binance Futures."""

    def __init__(self, symbol: str = SYMBOL, testnet: bool = TESTNET) -> None:
        self.symbol = symbol
        self.testnet = testnet
        self._exchange: Optional[ccxt.Exchange] = None

    async def _get_exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            exchange_class = getattr(ccxt, EXCHANGE_ID)
            self._exchange = exchange_class({
                "apiKey": API_KEY,
                "secret": SECRET_KEY,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            })
            if self.testnet:
                self._exchange.enable_demo_trading(True)
        return self._exchange

    async def fetch_ohlcv(
        self,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles from Binance.

        Args:
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d).
            since: Start timestamp in milliseconds.
            limit: Number of candles to fetch (max 1500).

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        exchange = await self._get_exchange()
        ohlcv = await exchange.fetch_ohlcv(
            self.symbol, timeframe=timeframe, since=since, limit=limit
        )
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        return df

    async def fetch_historical(
        self,
        timeframe: str = "1h",
        days: int = 365,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data by paginating through time.

        Args:
            timeframe: Candle timeframe.
            days: Number of days of history to fetch.

        Returns:
            DataFrame with full historical OHLCV data.
        """
        exchange = await self._get_exchange()
        all_candles: list[list] = []
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

        while True:
            candles = await exchange.fetch_ohlcv(
                self.symbol, timeframe=timeframe, since=since, limit=1500
            )
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            logger.info(
                "Fetched %d candles, last: %s",
                len(candles),
                datetime.utcfromtimestamp(candles[-1][0] / 1000),
            )
            await asyncio.sleep(exchange.rateLimit / 1000)

        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
        return df

    async def fetch_since(
        self,
        timeframe: str = "1h",
        since_ms: int = 0,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from a specific timestamp to now by paginating.

        Args:
            timeframe: Candle timeframe.
            since_ms: Start timestamp in milliseconds (inclusive).

        Returns:
            DataFrame with OHLCV data from since_ms to present.
        """
        exchange = await self._get_exchange()
        all_candles: list[list] = []
        since = since_ms

        while True:
            candles = await exchange.fetch_ohlcv(
                self.symbol, timeframe=timeframe, since=since, limit=1500
            )
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            logger.info(
                "Fetched %d candles, last: %s",
                len(candles),
                datetime.utcfromtimestamp(candles[-1][0] / 1000),
            )
            if len(candles) < 1500:
                break
            await asyncio.sleep(exchange.rateLimit / 1000)

        if not all_candles:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
        return df

    async def fetch_funding_rate(
        self,
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch funding rate history from Binance Futures.

        Args:
            since: Start timestamp in milliseconds.
            limit: Number of records per request.

        Returns:
            DataFrame with funding rate data.
        """
        exchange = await self._get_exchange()
        all_records: list[dict] = []
        current_since = since

        while True:
            params = {"symbol": self.symbol.replace("/", "").replace(":USDT", "")}
            if current_since:
                params["startTime"] = current_since
            params["limit"] = limit

            records = await exchange.fapiPublicGetFundingRate(params)
            if not records:
                break
            all_records.extend(records)
            current_since = int(records[-1]["fundingTime"]) + 1
            logger.info("Fetched %d funding rate records", len(records))
            if len(records) < limit:
                break
            await asyncio.sleep(exchange.rateLimit / 1000)

        if not all_records:
            return pd.DataFrame(columns=["fundingRate"])

        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms")
        df["fundingRate"] = df["fundingRate"].astype(float)
        df = df[["timestamp", "fundingRate"]].set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df

    async def fetch_open_interest(
        self,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch open interest history from Binance Futures.

        Uses a separate non-testnet exchange instance since testnet does not
        support fapiData endpoints. Open interest is public data.

        Note: Binance OI API limits startTime to ~30 days in the past.
        This method paginates within that window.

        Args:
            timeframe: Period for open interest data (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d).
            since: Start timestamp in milliseconds (max ~30 days ago).
            limit: Number of records per request (max 500).

        Returns:
            DataFrame with open interest data.
        """
        tf_map = {
            "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h",
            "6h": "6h", "12h": "12h", "1d": "1d",
        }
        period = tf_map.get(timeframe, "1h")
        symbol = self.symbol.replace("/", "").replace(":USDT", "")

        # Clamp since to ~29 days ago (API limit)
        max_lookback_ms = int((datetime.utcnow() - timedelta(days=29)).timestamp() * 1000)
        if since is None or since < max_lookback_ms:
            since = max_lookback_ms
            logger.info("OI API limited to ~30 days lookback, clamping startTime")

        # Create a non-testnet exchange for public data
        prod_exchange = ccxt.binanceusdm({"enableRateLimit": True})

        all_records: list[dict] = []
        current_since = since

        try:
            while True:
                params: dict = {
                    "symbol": symbol,
                    "period": period,
                    "limit": min(limit, 500),
                    "startTime": current_since,
                }

                records = await prod_exchange.fapiDataGetOpenInterestHist(params)
                if not records:
                    break
                all_records.extend(records)
                current_since = int(records[-1]["timestamp"]) + 1
                logger.info("Fetched %d open interest records", len(records))
                if len(records) < limit:
                    break
                await asyncio.sleep(prod_exchange.rateLimit / 1000)
        finally:
            await prod_exchange.close()

        if not all_records:
            return pd.DataFrame(columns=["sumOpenInterest", "sumOpenInterestValue"])

        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
        df = df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]].set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df

    async def close(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

"""WebSocket real-time data stream from Binance."""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from typing import Callable, Optional

import certifi
import websockets

from config.settings import SYMBOL, TESTNET

logger = logging.getLogger(__name__)

# Binance WebSocket endpoints
WS_MAINNET = "wss://fstream.binance.com/ws"
WS_TESTNET = "wss://stream.binancefuture.com/ws"


class BinanceStream:
    """Real-time WebSocket stream for Binance Futures."""

    _BASE_RECONNECT_DELAY = 5
    _MAX_RECONNECT_DELAY = 60

    def __init__(
        self,
        symbol: str = SYMBOL,
        testnet: bool = TESTNET,
        on_candle: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
    ) -> None:
        self.symbol = symbol.replace("/", "").replace(":USDT", "").lower()
        # Always use mainnet WebSocket for real market data
        self.ws_url = WS_MAINNET
        self.on_candle = on_candle
        self.on_trade = on_trade
        self._running = False
        self._reconnect_count = 0

    async def start_kline_stream(self, timeframe: str = "1m") -> None:
        """Subscribe to kline/candlestick stream.

        Args:
            timeframe: Candle interval (1m, 5m, 15m, 1h, etc.).
        """
        stream_name = f"{self.symbol}@kline_{timeframe}"
        url = f"{self.ws_url}/{stream_name}"
        self._running = True

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        while self._running:
            try:
                async with websockets.connect(url, ssl=ssl_ctx) as ws:
                    logger.info("Connected to %s (reconnects=%d)",
                                url, self._reconnect_count)
                    self._reconnect_count = 0  # Reset on successful connection
                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        if self.on_candle and data.get("e") == "kline":
                            await self.on_candle(data["k"])
            except asyncio.CancelledError:
                break
            except Exception:
                self._reconnect_count += 1
                delay = min(
                    self._BASE_RECONNECT_DELAY * (2 ** min(self._reconnect_count - 1, 4)),
                    self._MAX_RECONNECT_DELAY,
                )
                logger.exception(
                    "WebSocket error (attempt #%d), reconnecting in %ds",
                    self._reconnect_count, delay,
                )
                await asyncio.sleep(delay)

    def stop(self) -> None:
        """Stop the WebSocket stream."""
        self._running = False

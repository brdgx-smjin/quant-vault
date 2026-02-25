"""Binance Futures order executor using ccxt."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

import ccxt.async_support as ccxt

from config.settings import API_KEY, EXCHANGE_ID, SECRET_KEY, TESTNET
from src.execution.order_manager import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


class BinanceExecutor:
    """Executes orders on Binance Futures via ccxt."""

    def __init__(self, testnet: bool = TESTNET) -> None:
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

    async def execute(self, order: Order) -> dict:
        """Execute an order on Binance.

        Args:
            order: Order to execute.

        Returns:
            Exchange response dict.
        """
        exchange = await self._get_exchange()

        # Set leverage
        try:
            await exchange.set_leverage(order.leverage, order.symbol)
        except Exception:
            logger.warning("Could not set leverage for %s", order.symbol)

        side = order.side.value
        amount = float(order.amount)
        params = {}

        if order.reduce_only:
            params["reduceOnly"] = True

        if order.order_type == OrderType.MARKET:
            result = await exchange.create_order(
                order.symbol, "market", side, amount, params=params,
            )
        elif order.order_type == OrderType.LIMIT:
            result = await exchange.create_order(
                order.symbol, "limit", side, amount,
                float(order.price), params=params,
            )
        elif order.order_type == OrderType.STOP_LOSS:
            params["stopPrice"] = float(order.stop_price)
            result = await exchange.create_order(
                order.symbol, "stop_market", side, amount, params=params,
            )
        else:
            raise ValueError(f"Unknown order type: {order.order_type}")

        logger.info("Order executed: %s", result.get("id"))
        return result

    async def get_balance(self) -> Decimal:
        """Get USDT futures balance."""
        exchange = await self._get_exchange()
        balance = await exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        return Decimal(str(usdt.get("free", 0)))

    async def get_positions(self) -> list[dict]:
        """Get open positions."""
        exchange = await self._get_exchange()
        positions = await exchange.fetch_positions()
        return [p for p in positions if float(p.get("contracts", 0)) > 0]

    async def close(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

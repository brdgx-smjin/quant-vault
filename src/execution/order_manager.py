"""Order creation and management for Binance Futures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    leverage: int = 5
    reduce_only: bool = False


class OrderManager:
    """Creates and tracks orders."""

    def __init__(self) -> None:
        self.pending_orders: list[Order] = []
        self.filled_orders: list[dict] = []

    def create_market_order(
        self, symbol: str, side: OrderSide, amount: Decimal, leverage: int = 5
    ) -> Order:
        """Create a market order."""
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            amount=amount,
            leverage=leverage,
        )
        self.pending_orders.append(order)
        logger.info("Market order created: %s %s %s", side.value, amount, symbol)
        return order

    def create_limit_order(
        self, symbol: str, side: OrderSide, amount: Decimal,
        price: Decimal, leverage: int = 5,
    ) -> Order:
        """Create a limit order."""
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            amount=amount,
            price=price,
            leverage=leverage,
        )
        self.pending_orders.append(order)
        logger.info("Limit order created: %s %s %s @ %s", side.value, amount, symbol, price)
        return order

    def create_stop_loss(
        self, symbol: str, side: OrderSide, amount: Decimal, stop_price: Decimal
    ) -> Order:
        """Create a stop-loss order."""
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_LOSS,
            amount=amount,
            stop_price=stop_price,
            reduce_only=True,
        )
        self.pending_orders.append(order)
        logger.info("Stop-loss created: %s %s %s @ %s", side.value, amount, symbol, stop_price)
        return order

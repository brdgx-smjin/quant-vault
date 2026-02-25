"""Position tracking and management."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""

    symbol: str
    side: str  # "long" or "short"
    entry_price: Decimal
    amount: Decimal
    leverage: int
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal("0")
    # Partial close tracking: list of (tp_price, close_fraction) tuples
    tp_levels: list = field(default_factory=list)
    next_tp_idx: int = 0


class PositionManager:
    """Tracks and manages open positions."""

    def __init__(self) -> None:
        self.positions: dict[str, Position] = {}

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        amount: Decimal,
        leverage: int = 5,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> Position:
        """Record a new open position."""
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            amount=amount,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.positions[symbol] = pos
        logger.info("Opened %s %s @ %s, size=%s", side, symbol, entry_price, amount)
        return pos

    def close_position(self, symbol: str, exit_price: Decimal) -> Decimal:
        """Close a position fully and return realized PnL."""
        if symbol not in self.positions:
            return Decimal("0")

        pos = self.positions.pop(symbol)
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.amount
        else:
            pnl = (pos.entry_price - exit_price) * pos.amount

        logger.info("Closed %s %s @ %s, PnL=%s", pos.side, symbol, exit_price, pnl)
        return pnl

    def partial_close(self, symbol: str, exit_price: Decimal, fraction: Decimal) -> Decimal:
        """Close a fraction of a position and return realized PnL on that portion.

        Args:
            symbol: Trading pair symbol.
            exit_price: Current exit price.
            fraction: Fraction of remaining position to close (0-1).

        Returns:
            Realized PnL for the closed portion.
        """
        if symbol not in self.positions:
            return Decimal("0")

        pos = self.positions[symbol]
        close_amount = pos.amount * fraction

        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * close_amount
        else:
            pnl = (pos.entry_price - exit_price) * close_amount

        pos.amount -= close_amount
        logger.info(
            "Partial close %s %s @ %s, fraction=%.0f%%, PnL=%s, remaining=%s",
            pos.side, symbol, exit_price,
            float(fraction) * 100, pnl, pos.amount,
        )

        # Remove position if fully closed
        if pos.amount <= 0:
            self.positions.pop(symbol)

        return pnl

    def update_unrealized_pnl(self, symbol: str, current_price: Decimal) -> None:
        """Update unrealized PnL for a position."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        if pos.side == "long":
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.amount
        else:
            pos.unrealized_pnl = (pos.entry_price - current_price) * pos.amount

    def get_total_exposure(self) -> Decimal:
        """Get total position exposure in USDT."""
        return sum(
            pos.entry_price * pos.amount for pos in self.positions.values()
        )

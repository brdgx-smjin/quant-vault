"""Math utilities for trading calculations."""

from __future__ import annotations

from decimal import Decimal


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Calculate Kelly criterion position size fraction.

    Args:
        win_rate: Win probability (0-1).
        avg_win: Average winning trade return.
        avg_loss: Average losing trade return (positive number).

    Returns:
        Optimal fraction of capital to risk.
    """
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    return max(0, (b * win_rate - (1 - win_rate)) / b)


def round_to_tick(price: Decimal, tick_size: Decimal) -> Decimal:
    """Round price to nearest tick size."""
    return (price / tick_size).quantize(Decimal("1")) * tick_size


def calculate_pnl(
    entry_price: Decimal,
    exit_price: Decimal,
    amount: Decimal,
    side: str,
) -> Decimal:
    """Calculate PnL for a trade.

    Args:
        entry_price: Entry price.
        exit_price: Exit price.
        amount: Position size.
        side: 'long' or 'short'.

    Returns:
        Realized PnL.
    """
    if side == "long":
        return (exit_price - entry_price) * amount
    else:
        return (entry_price - exit_price) * amount

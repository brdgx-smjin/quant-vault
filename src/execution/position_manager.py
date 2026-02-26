"""Position tracking and management."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILE = Path("data/position_state.json")


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

    def sync_from_exchange(self, exchange_positions: list[dict]) -> list[str]:
        """Sync local state with actual Binance positions.

        Args:
            exchange_positions: List of position dicts from BinanceExecutor.get_positions().

        Returns:
            List of human-readable change descriptions.
        """
        changes: list[str] = []

        # Build map of exchange positions: symbol -> position dict
        remote: dict[str, dict] = {}
        for p in exchange_positions:
            sym = p.get("symbol", "")
            remote[sym] = p

        # 1. Remote exists but local doesn't → add to local
        for sym, p in remote.items():
            contracts = Decimal(str(p.get("contracts", 0)))
            if contracts <= 0:
                continue
            side = (p.get("side") or "long").lower()  # "long" or "short"
            entry = Decimal(str(p.get("entryPrice") or 0))
            leverage = int(p.get("leverage") or 5)

            if sym not in self.positions:
                self.positions[sym] = Position(
                    symbol=sym,
                    side=side,
                    entry_price=entry,
                    amount=contracts,
                    leverage=leverage,
                )
                changes.append(
                    f"외부 포지션 감지: {side.upper()} {sym} {contracts} @ {entry}"
                )
                logger.info("Synced external position: %s %s %s @ %s", side, sym, contracts, entry)
            else:
                local = self.positions[sym]
                # Update amount if it changed (partial close externally, etc.)
                if local.amount != contracts or local.side != side:
                    old_desc = f"{local.side.upper()} {local.amount}"
                    local.side = side
                    local.amount = contracts
                    local.entry_price = entry
                    local.leverage = leverage
                    changes.append(
                        f"포지션 변경 감지: {sym} {old_desc} → {side.upper()} {contracts} @ {entry}"
                    )
                    logger.info("Synced position change: %s -> %s %s @ %s", sym, side, contracts, entry)

        # 2. Local exists but remote doesn't → position was closed externally
        closed_symbols = [sym for sym in self.positions if sym not in remote]
        for sym in closed_symbols:
            pos = self.positions.pop(sym)
            changes.append(
                f"외부 청산 감지: {pos.side.upper()} {sym} {pos.amount} @ {pos.entry_price}"
            )
            logger.info("Synced external close: %s %s", pos.side, sym)

        return changes

    def get_total_exposure(self) -> Decimal:
        """Get total position exposure in USDT."""
        return sum(
            pos.entry_price * pos.amount for pos in self.positions.values()
        )

    def _save_state(self) -> None:
        """Persist SL/TP/tp_levels to JSON for crash recovery."""
        state: dict[str, dict] = {}
        for sym, pos in self.positions.items():
            tp_levels_ser = [
                [str(price), str(frac)] for price, frac in pos.tp_levels
            ]
            state[sym] = {
                "side": pos.side,
                "entry_price": str(pos.entry_price),
                "amount": str(pos.amount),
                "leverage": pos.leverage,
                "stop_loss": str(pos.stop_loss) if pos.stop_loss is not None else None,
                "take_profit": str(pos.take_profit) if pos.take_profit is not None else None,
                "tp_levels": tp_levels_ser,
                "next_tp_idx": pos.next_tp_idx,
            }
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(json.dumps(state, indent=2))
            logger.debug("[STATE] Saved position state: %d positions", len(state))
        except Exception:
            logger.exception("[STATE] Failed to save position state")

    def _load_state(self) -> dict:
        """Load position metadata from JSON."""
        if not STATE_FILE.exists():
            return {}
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            logger.exception("[STATE] Failed to load position state")
            return {}

    def restore_sl_tp(self) -> list[str]:
        """Restore SL/TP/tp_levels from JSON after sync_from_exchange().

        Returns:
            List of human-readable restoration descriptions.
        """
        saved = self._load_state()
        if not saved:
            return []

        changes: list[str] = []
        for sym, pos in self.positions.items():
            if sym not in saved:
                continue
            s = saved[sym]

            # Only restore if sides match (position hasn't flipped)
            if s.get("side") != pos.side:
                logger.warning(
                    "[RESTORE] Side mismatch for %s: saved=%s, current=%s — skipping",
                    sym, s.get("side"), pos.side,
                )
                continue

            restored_fields: list[str] = []

            if s.get("stop_loss") is not None and pos.stop_loss is None:
                pos.stop_loss = Decimal(s["stop_loss"])
                restored_fields.append(f"SL={s['stop_loss']}")

            if s.get("take_profit") is not None and pos.take_profit is None:
                pos.take_profit = Decimal(s["take_profit"])
                restored_fields.append(f"TP={s['take_profit']}")

            if s.get("tp_levels") and not pos.tp_levels:
                pos.tp_levels = [
                    (Decimal(p), Decimal(f)) for p, f in s["tp_levels"]
                ]
                pos.next_tp_idx = s.get("next_tp_idx", 0)
                restored_fields.append(f"tp_levels={len(pos.tp_levels)}")

            if restored_fields:
                desc = f"[RESTORE] {sym}: {', '.join(restored_fields)}"
                changes.append(desc)
                logger.info(desc)

        # Clean up saved state for positions that no longer exist
        stale = [sym for sym in saved if sym not in self.positions]
        if stale:
            for sym in stale:
                del saved[sym]
            try:
                STATE_FILE.write_text(json.dumps(saved, indent=2))
                logger.info("[STATE] Cleaned stale entries: %s", stale)
            except Exception:
                logger.exception("[STATE] Failed to clean stale state")

        return changes

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
    component_id: str = ""  # e.g. "15m_mtf_rsi_mean_reversion"


class PositionManager:
    """Tracks and manages open positions."""

    def __init__(self) -> None:
        self.positions: dict[str, Position] = {}

    def get_symbol_positions(self, symbol: str) -> dict[str, Position]:
        """Return all component positions for a symbol.

        Args:
            symbol: Trading pair symbol (e.g. "BTC/USDT:USDT").

        Returns:
            Dict of position_key -> Position for the symbol.
        """
        return {k: v for k, v in self.positions.items() if v.symbol == symbol}

    def get_net_amount(self, symbol: str) -> Decimal:
        """Calculate NET position amount for a symbol (signed: + long, - short).

        Args:
            symbol: Trading pair symbol.

        Returns:
            Net amount (positive = net long, negative = net short).
        """
        net = Decimal("0")
        for pos in self.positions.values():
            if pos.symbol == symbol:
                if pos.side == "long":
                    net += pos.amount
                else:
                    net -= pos.amount
        return net

    @staticmethod
    def make_position_key(symbol: str, component_id: str = "") -> str:
        """Build the position dict key.

        Args:
            symbol: Trading pair symbol.
            component_id: Optional component identifier.

        Returns:
            "symbol:component_id" if component_id, else "symbol".
        """
        if component_id:
            return f"{symbol}:{component_id}"
        return symbol

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        amount: Decimal,
        leverage: int = 5,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        component_id: str = "",
    ) -> Position:
        """Record a new open position.

        Args:
            symbol: Trading pair symbol.
            side: "long" or "short".
            entry_price: Entry price.
            amount: Position size.
            leverage: Leverage multiplier.
            stop_loss: Stop-loss price.
            take_profit: Take-profit price.
            component_id: Component identifier for multi-position tracking.

        Returns:
            The created Position.
        """
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            amount=amount,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            component_id=component_id,
        )
        key = self.make_position_key(symbol, component_id)
        self.positions[key] = pos
        logger.info("Opened %s %s [%s] @ %s, size=%s", side, symbol, component_id or "default", entry_price, amount)
        return pos

    def close_position(self, position_key: str, exit_price: Decimal) -> Decimal:
        """Close a position fully and return realized PnL.

        Args:
            position_key: Position dict key (symbol or symbol:component_id).
            exit_price: Exit price.

        Returns:
            Realized PnL.
        """
        if position_key not in self.positions:
            return Decimal("0")

        pos = self.positions.pop(position_key)
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.amount
        else:
            pnl = (pos.entry_price - exit_price) * pos.amount

        logger.info("Closed %s %s [%s] @ %s, PnL=%s", pos.side, pos.symbol, pos.component_id or "default", exit_price, pnl)
        return pnl

    def partial_close(self, position_key: str, exit_price: Decimal, fraction: Decimal) -> Decimal:
        """Close a fraction of a position and return realized PnL on that portion.

        Args:
            position_key: Position dict key (symbol or symbol:component_id).
            exit_price: Current exit price.
            fraction: Fraction of remaining position to close (0-1).

        Returns:
            Realized PnL for the closed portion.
        """
        if position_key not in self.positions:
            return Decimal("0")

        pos = self.positions[position_key]
        close_amount = pos.amount * fraction

        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * close_amount
        else:
            pnl = (pos.entry_price - exit_price) * close_amount

        pos.amount -= close_amount
        logger.info(
            "Partial close %s %s [%s] @ %s, fraction=%.0f%%, PnL=%s, remaining=%s",
            pos.side, pos.symbol, pos.component_id or "default", exit_price,
            float(fraction) * 100, pnl, pos.amount,
        )

        # Remove position if fully closed
        if pos.amount <= 0:
            self.positions.pop(position_key)

        return pnl

    def update_unrealized_pnl(self, symbol: str, current_price: Decimal) -> None:
        """Update unrealized PnL for all positions of a symbol."""
        for pos in self.positions.values():
            if pos.symbol == symbol:
                if pos.side == "long":
                    pos.unrealized_pnl = (current_price - pos.entry_price) * pos.amount
                else:
                    pos.unrealized_pnl = (pos.entry_price - current_price) * pos.amount

    def sync_from_exchange(self, exchange_positions: list[dict]) -> list[str]:
        """Sync local state with actual Binance positions.

        Multi-position aware:
        - No local positions → restore from saved state if available, else create legacy
        - Local positions exist → compare NET amounts, reconcile on mismatch
        - Exchange position gone → clear all local positions for that symbol

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

        # Collect all symbols tracked locally
        local_symbols: set[str] = {pos.symbol for pos in self.positions.values()}

        # 1. Remote exists — check against local
        for sym, p in remote.items():
            contracts = Decimal(str(p.get("contracts", 0)))
            if contracts <= 0:
                continue
            side = (p.get("side") or "long").lower()
            entry = Decimal(str(p.get("entryPrice") or 0))
            leverage = int(p.get("leverage") or 5)

            local_positions = self.get_symbol_positions(sym)

            if not local_positions:
                # No local positions — try to restore from saved state
                restored = self._restore_from_saved(sym, side, contracts, entry, leverage)
                if restored:
                    changes.extend(restored)
                else:
                    # No saved state — create legacy entry
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
                # Local positions exist — compare NET amount
                # Use tolerance >= Binance step size (0.001 for BTC)
                local_net = self.get_net_amount(sym)
                remote_signed = contracts if side == "long" else -contracts

                diff = abs(local_net - remote_signed)
                if diff > Decimal("0.001"):
                    # Significant mismatch — reconcile by scaling local amounts
                    logger.warning(
                        "[SYNC] %s NET mismatch: local=%s, exchange=%s %s — reconciling",
                        sym, local_net, side.upper(), contracts,
                    )
                    self._reconcile_amounts(sym, side, contracts)
                    changes.append(
                        f"[SYNC] {sym} 수량 보정: local={local_net} → exchange={side.upper()} {contracts}"
                    )

        # 2. Local exists but remote doesn't → externally closed
        for sym in local_symbols:
            if sym not in remote or Decimal(str(remote[sym].get("contracts", 0))) <= 0:
                # Remove ALL local positions for this symbol
                keys_to_remove = [k for k, v in self.positions.items() if v.symbol == sym]
                for key in keys_to_remove:
                    pos = self.positions.pop(key)
                    changes.append(
                        f"외부 청산 감지: {pos.side.upper()} {sym} [{pos.component_id or 'default'}] {pos.amount}"
                    )
                    logger.info("Synced external close: %s %s [%s]", pos.side, sym, pos.component_id or "default")

        return changes

    def _restore_from_saved(
        self,
        symbol: str,
        exchange_side: str,
        exchange_amount: Decimal,
        exchange_entry: Decimal,
        exchange_leverage: int,
    ) -> list[str]:
        """Try to restore component positions from saved state file.

        Args:
            symbol: Trading pair symbol.
            exchange_side: Position side from exchange.
            exchange_amount: Position amount from exchange.
            exchange_entry: Entry price from exchange.
            exchange_leverage: Leverage from exchange.

        Returns:
            List of change descriptions, or empty list if no saved state found.
        """
        saved = self._load_state()
        # Find saved components matching symbol and side
        saved_components = {
            k: v for k, v in saved.items()
            if v.get("symbol") == symbol and v.get("side") == exchange_side
        }

        if not saved_components:
            return []

        changes: list[str] = []
        saved_total = sum(
            Decimal(v["amount"]) for v in saved_components.values()
        )

        # Scale amounts proportionally to match exchange total
        ratio = exchange_amount / saved_total if saved_total > 0 else Decimal("1")

        for saved_key, s in saved_components.items():
            amount = Decimal(s["amount"]) * ratio
            pos = Position(
                symbol=symbol,
                side=s.get("side", exchange_side),
                entry_price=Decimal(s.get("entry_price", str(exchange_entry))),
                amount=amount,
                leverage=int(s.get("leverage", exchange_leverage)),
                stop_loss=Decimal(s["stop_loss"]) if s.get("stop_loss") else None,
                take_profit=Decimal(s["take_profit"]) if s.get("take_profit") else None,
                component_id=s.get("component_id", ""),
            )
            if s.get("tp_levels"):
                pos.tp_levels = [
                    (Decimal(p_), Decimal(f_)) for p_, f_ in s["tp_levels"]
                ]
                pos.next_tp_idx = s.get("next_tp_idx", 0)

            self.positions[saved_key] = pos
            changes.append(
                f"포지션 복원: {pos.side.upper()} {symbol} [{pos.component_id or 'default'}] "
                f"{pos.amount} @ {pos.entry_price} "
                f"SL={pos.stop_loss or 'N/A'} TP={pos.take_profit or 'N/A'}"
            )
            logger.info(
                "Restored component position: %s %s [%s] @ %s, size=%s, SL=%s, TP=%s",
                pos.side, symbol, pos.component_id or "default",
                pos.entry_price, pos.amount, pos.stop_loss, pos.take_profit,
            )

        return changes

    def _reconcile_amounts(self, symbol: str, exchange_side: str, exchange_amount: Decimal) -> None:
        """Reconcile local position amounts to match exchange.

        If local NET direction matches exchange, scale proportionally.
        If direction flipped, clear all local and recreate from exchange/saved.

        Args:
            symbol: Trading pair symbol.
            exchange_side: Position side from exchange.
            exchange_amount: Absolute position amount from exchange.
        """
        local_net = self.get_net_amount(symbol)
        local_side = "long" if local_net > 0 else "short"

        if local_side != exchange_side:
            # Direction flipped — clear all local, recreate from saved or legacy
            logger.warning(
                "[SYNC] %s direction flipped: local=%s, exchange=%s — clearing local",
                symbol, local_side.upper(), exchange_side.upper(),
            )
            keys_to_remove = [k for k, v in self.positions.items() if v.symbol == symbol]
            for key in keys_to_remove:
                self.positions.pop(key)
            # Try to restore from saved state
            restored = self._restore_from_saved(
                symbol, exchange_side, exchange_amount, Decimal("0"), 5,
            )
            if not restored:
                # Fallback: can't restore without entry price, will be picked up next sync
                logger.warning("[SYNC] %s no saved state for flipped direction", symbol)
        else:
            # Same direction — scale proportionally
            local_total = abs(local_net)
            if local_total > 0:
                scale = exchange_amount / local_total
                for pos in self.positions.values():
                    if pos.symbol == symbol:
                        old = pos.amount
                        pos.amount = pos.amount * scale
                        logger.info(
                            "[SYNC] Adjusted %s [%s]: %s → %s",
                            symbol, pos.component_id or "default", old, pos.amount,
                        )

    def get_total_exposure(self) -> Decimal:
        """Get total position exposure in USDT."""
        return sum(
            pos.entry_price * pos.amount for pos in self.positions.values()
        )

    def _save_state(self) -> None:
        """Persist SL/TP/tp_levels to JSON for crash recovery."""
        state: dict[str, dict] = {}
        for key, pos in self.positions.items():
            tp_levels_ser = [
                [str(price), str(frac)] for price, frac in pos.tp_levels
            ]
            state[key] = {
                "symbol": pos.symbol,
                "side": pos.side,
                "entry_price": str(pos.entry_price),
                "amount": str(pos.amount),
                "leverage": pos.leverage,
                "stop_loss": str(pos.stop_loss) if pos.stop_loss is not None else None,
                "take_profit": str(pos.take_profit) if pos.take_profit is not None else None,
                "tp_levels": tp_levels_ser,
                "next_tp_idx": pos.next_tp_idx,
                "component_id": pos.component_id,
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

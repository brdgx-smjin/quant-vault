"""Risk management module."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional

import yaml

from config.settings import PROJECT_ROOT
from src.strategy.base import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Result of risk validation."""

    approved: bool
    position_size: Decimal
    leverage: int
    reason: str = ""


class RiskManager:
    """Validates trades against risk parameters before execution."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            config_path = str(PROJECT_ROOT / "config" / "risk.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.risk = self.config["risk_management"]
        self.sizing = self.config["position_sizing"]
        self.daily_pnl = Decimal("0")
        self.open_positions = 0

    def check_trade(
        self,
        signal: TradeSignal,
        account_balance: Decimal,
    ) -> RiskCheckResult:
        """Validate a trade signal against risk rules.

        Args:
            signal: The proposed trade signal.
            account_balance: Current account balance in USDT.

        Returns:
            RiskCheckResult with approval status and position size.
        """
        # Check daily max loss
        daily_max = account_balance * Decimal(str(self.risk["daily_max_loss_pct"]))
        if self.daily_pnl < -daily_max:
            return RiskCheckResult(
                approved=False,
                position_size=Decimal("0"),
                leverage=0,
                reason=f"Daily loss limit reached: {self.daily_pnl}",
            )

        # Check max open positions
        if self.open_positions >= self.risk["max_open_positions"]:
            return RiskCheckResult(
                approved=False,
                position_size=Decimal("0"),
                leverage=0,
                reason=f"Max positions reached: {self.open_positions}",
            )

        # Calculate position size
        max_loss = account_balance * Decimal(str(self.risk["max_loss_per_trade_pct"]))
        leverage = min(
            self.risk["default_leverage"],
            self.risk["max_leverage"],
        )

        price = Decimal(str(signal.price))
        if price <= 0:
            return RiskCheckResult(
                approved=False,
                position_size=Decimal("0"),
                leverage=0,
                reason=f"Invalid signal price: {price}",
            )

        if signal.stop_loss is not None:
            risk_per_unit = abs(price - Decimal(str(signal.stop_loss)))
            if risk_per_unit > 0:
                position_size = max_loss / risk_per_unit
            else:
                # Fallback: convert fixed USDT amount to asset units
                position_size = Decimal(str(self.sizing["fixed_amount_usdt"])) / price
        else:
            # No stop loss: convert fixed USDT amount to asset units
            position_size = Decimal(str(self.sizing["fixed_amount_usdt"])) / price

        # Apply Kelly fraction if configured
        if self.sizing.get("method") == "kelly":
            kelly_fraction = Decimal(str(self.sizing.get("kelly_fraction", "0.25")))
            position_size = position_size * kelly_fraction

        # Cap position size by account balance * leverage (convert to asset units)
        max_notional = account_balance * Decimal("0.1") * leverage
        max_position = max_notional / price
        position_size = min(position_size, max_position)

        # Binance minimum notional check (~$5 USDT)
        min_notional = Decimal("5")
        notional_value = position_size * price
        if notional_value < min_notional:
            return RiskCheckResult(
                approved=False,
                position_size=Decimal("0"),
                leverage=0,
                reason=f"Position too small: notional={notional_value:.2f} < min {min_notional}",
            )

        logger.info(
            "Risk check APPROVED: size=%.4f, leverage=%d, max_loss=%.2f, notional=%.2f",
            float(position_size), leverage, float(max_loss), float(notional_value),
        )

        return RiskCheckResult(
            approved=True,
            position_size=position_size,
            leverage=leverage,
        )

    def update_daily_pnl(self, pnl: Decimal) -> None:
        """Update running daily PnL."""
        self.daily_pnl += pnl

    def get_daily_loss_level(self, account_balance: Decimal) -> str:
        """Check daily loss severity level.

        Args:
            account_balance: Current account balance.

        Returns:
            "ok", "warn" (>3% loss), or "critical" (>5% loss).
        """
        if account_balance <= 0:
            return "ok"
        loss_pct = -self.daily_pnl / account_balance
        if loss_pct >= Decimal("0.05"):
            return "critical"
        if loss_pct >= Decimal("0.03"):
            return "warn"
        return "ok"

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of each trading day)."""
        self.daily_pnl = Decimal("0")

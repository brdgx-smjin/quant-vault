"""Performance dashboard data provider."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional


@dataclass
class DashboardMetrics:
    """Real-time trading performance metrics."""

    total_pnl: Decimal
    daily_pnl: Decimal
    win_rate: float
    total_trades: int
    open_positions: int
    max_drawdown: float
    sharpe_ratio: float
    current_equity: float = 0.0
    current_drawdown: float = 0.0
    consecutive_losses: int = 0
    daily_trade_count: int = 0


class DashboardProvider:
    """Provides metrics for monitoring dashboard (Grafana / CLI)."""

    def __init__(self, initial_equity: Decimal = Decimal("10000")) -> None:
        self.trades: list[dict] = []
        self.daily_pnl = Decimal("0")
        self._daily_trade_count = 0

        # Equity curve tracking
        self._initial_equity = initial_equity
        self._equity_curve: list[float] = [float(initial_equity)]
        self._peak_equity: float = float(initial_equity)
        self._max_drawdown: float = 0.0
        self._returns: list[float] = []
        self._consecutive_losses = 0

    def record_trade(self, pnl: Decimal, metadata: dict) -> None:
        """Record a completed trade and update equity curve."""
        self.trades.append({"pnl": pnl, **metadata})
        self.daily_pnl += pnl
        self._daily_trade_count += 1

        # Track consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Update equity curve
        current_equity = self._equity_curve[-1] + float(pnl)
        self._equity_curve.append(current_equity)

        # Update peak and max drawdown
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        drawdown = (self._peak_equity - current_equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

        # Track returns for Sharpe calculation
        prev_equity = self._equity_curve[-2]
        if prev_equity > 0:
            self._returns.append(float(pnl) / prev_equity)

    def reset_daily(self) -> None:
        """Reset daily PnL counter (call at UTC midnight)."""
        self.daily_pnl = Decimal("0")
        self._daily_trade_count = 0

    def get_metrics(self) -> DashboardMetrics:
        """Calculate current performance metrics."""
        if not self.trades:
            return DashboardMetrics(
                total_pnl=Decimal("0"), daily_pnl=Decimal("0"),
                win_rate=0.0, total_trades=0, open_positions=0,
                max_drawdown=0.0, sharpe_ratio=0.0,
                current_equity=self.current_equity,
                current_drawdown=self.current_drawdown,
                consecutive_losses=0,
                daily_trade_count=0,
            )

        total_pnl = sum(t["pnl"] for t in self.trades)
        wins = [t for t in self.trades if t["pnl"] > 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0

        return DashboardMetrics(
            total_pnl=total_pnl,
            daily_pnl=self.daily_pnl,
            win_rate=win_rate,
            total_trades=len(self.trades),
            open_positions=0,  # Updated by position manager
            max_drawdown=self._max_drawdown,
            sharpe_ratio=self._calculate_sharpe(),
            current_equity=self.current_equity,
            current_drawdown=self.current_drawdown,
            consecutive_losses=self._consecutive_losses,
            daily_trade_count=self._daily_trade_count,
        )

    def _calculate_sharpe(self, annualization_factor: float = 365 * 6) -> float:
        """Calculate Sharpe ratio from trade returns.

        Args:
            annualization_factor: Number of trading periods per year.
                Default assumes 4h candles (6 per day * 365).

        Returns:
            Annualized Sharpe ratio, or 0.0 if insufficient data.
        """
        if len(self._returns) < 2:
            return 0.0

        mean_ret = sum(self._returns) / len(self._returns)
        variance = sum((r - mean_ret) ** 2 for r in self._returns) / (len(self._returns) - 1)
        std_ret = math.sqrt(variance)

        if std_ret == 0:
            return 0.0

        return (mean_ret / std_ret) * math.sqrt(annualization_factor)

    @property
    def equity_curve(self) -> list[float]:
        """Current equity curve values."""
        return list(self._equity_curve)

    @property
    def current_equity(self) -> float:
        """Current equity value."""
        return self._equity_curve[-1] if self._equity_curve else float(self._initial_equity)

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak (0-1)."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self.current_equity) / self._peak_equity

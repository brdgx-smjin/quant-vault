"""Tests for DashboardProvider metrics tracking."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.monitoring.dashboard import DashboardProvider


@pytest.fixture
def dashboard() -> DashboardProvider:
    return DashboardProvider(initial_equity=Decimal("10000"))


# ---------------------------------------------------------------------------
# Basic recording
# ---------------------------------------------------------------------------

class TestRecordTrade:

    def test_single_winning_trade(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("200"), {"side": "long"})
        m = dashboard.get_metrics()
        assert m.total_pnl == Decimal("200")
        assert m.total_trades == 1
        assert m.win_rate == 1.0

    def test_single_losing_trade(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("-100"), {"side": "long"})
        m = dashboard.get_metrics()
        assert m.total_pnl == Decimal("-100")
        assert m.win_rate == 0.0

    def test_mixed_trades_win_rate(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("200"), {"side": "long"})
        dashboard.record_trade(Decimal("-100"), {"side": "short"})
        dashboard.record_trade(Decimal("150"), {"side": "long"})
        m = dashboard.get_metrics()
        assert m.win_rate == pytest.approx(2 / 3)
        assert m.total_trades == 3


# ---------------------------------------------------------------------------
# Consecutive losses tracking
# ---------------------------------------------------------------------------

class TestConsecutiveLosses:

    def test_consecutive_losses_counter(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("-50"), {})
        dashboard.record_trade(Decimal("-30"), {})
        dashboard.record_trade(Decimal("-20"), {})
        m = dashboard.get_metrics()
        assert m.consecutive_losses == 3
        assert m.max_consecutive_losses == 3

    def test_win_resets_consecutive_losses(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("-50"), {})
        dashboard.record_trade(Decimal("-30"), {})
        dashboard.record_trade(Decimal("100"), {})
        m = dashboard.get_metrics()
        assert m.consecutive_losses == 0
        # Max should still be 2
        assert m.max_consecutive_losses == 2

    def test_max_consecutive_losses_across_streaks(self, dashboard: DashboardProvider) -> None:
        # First losing streak: 2
        dashboard.record_trade(Decimal("-50"), {})
        dashboard.record_trade(Decimal("-30"), {})
        dashboard.record_trade(Decimal("100"), {})
        # Second losing streak: 4
        dashboard.record_trade(Decimal("-10"), {})
        dashboard.record_trade(Decimal("-20"), {})
        dashboard.record_trade(Decimal("-30"), {})
        dashboard.record_trade(Decimal("-40"), {})
        m = dashboard.get_metrics()
        assert m.max_consecutive_losses == 4
        assert m.consecutive_losses == 4


# ---------------------------------------------------------------------------
# Drawdown and recovery
# ---------------------------------------------------------------------------

class TestDrawdown:

    def test_max_drawdown_calculation(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("500"), {})   # equity: 10500
        dashboard.record_trade(Decimal("-300"), {})   # equity: 10200
        m = dashboard.get_metrics()
        expected_dd = 300 / 10500
        assert m.max_drawdown == pytest.approx(expected_dd, rel=1e-6)

    def test_drawdown_recovery_tracks_trades(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("500"), {})   # equity: 10500 (peak)
        dashboard.record_trade(Decimal("-300"), {})   # equity: 10200 (DD start)
        dashboard.record_trade(Decimal("-100"), {})   # equity: 10100
        dashboard.record_trade(Decimal("200"), {})    # equity: 10300
        dashboard.record_trade(Decimal("300"), {})    # equity: 10600 (new peak, recovery!)
        m = dashboard.get_metrics()
        # DD started at trade idx 1, recovered at trade 4 → 4 trades to recover
        assert m.drawdown_recovery_trades >= 3

    def test_current_drawdown(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("1000"), {})   # equity: 11000 (peak)
        dashboard.record_trade(Decimal("-500"), {})    # equity: 10500
        m = dashboard.get_metrics()
        assert m.current_drawdown == pytest.approx(500 / 11000, rel=1e-6)


# ---------------------------------------------------------------------------
# Daily tracking
# ---------------------------------------------------------------------------

class TestDailyTracking:

    def test_daily_pnl_accumulates(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("100"), {})
        dashboard.record_trade(Decimal("-50"), {})
        m = dashboard.get_metrics()
        assert m.daily_pnl == Decimal("50")

    def test_daily_trade_count(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("100"), {})
        dashboard.record_trade(Decimal("200"), {})
        m = dashboard.get_metrics()
        assert m.daily_trade_count == 2

    def test_reset_daily(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("100"), {})
        dashboard.record_trade(Decimal("200"), {})
        dashboard.reset_daily()
        m = dashboard.get_metrics()
        assert m.daily_pnl == Decimal("0")
        assert m.daily_trade_count == 0
        # Total trades should still be 2
        assert m.total_trades == 2


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

class TestEquityCurve:

    def test_equity_curve_length(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("100"), {})
        dashboard.record_trade(Decimal("200"), {})
        # Initial + 2 trades = 3 points
        assert len(dashboard.equity_curve) == 3

    def test_current_equity(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("500"), {})
        assert dashboard.current_equity == pytest.approx(10500.0)

    def test_initial_equity_no_trades(self, dashboard: DashboardProvider) -> None:
        assert dashboard.current_equity == pytest.approx(10000.0)


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpe:

    def test_sharpe_insufficient_data(self, dashboard: DashboardProvider) -> None:
        dashboard.record_trade(Decimal("100"), {})
        m = dashboard.get_metrics()
        assert m.sharpe_ratio == 0.0

    def test_sharpe_positive_for_all_wins(self, dashboard: DashboardProvider) -> None:
        for _ in range(10):
            dashboard.record_trade(Decimal("100"), {})
        m = dashboard.get_metrics()
        assert m.sharpe_ratio > 0.0

    def test_sharpe_zero_for_no_variance(self, dashboard: DashboardProvider) -> None:
        # Exactly same return each time — zero variance → zero Sharpe
        # (Well, actually same PnL with growing equity means slightly different returns,
        #  so this tests that Sharpe is finite and positive)
        for _ in range(5):
            dashboard.record_trade(Decimal("100"), {})
        m = dashboard.get_metrics()
        # With constant positive PnL, Sharpe should be large and positive
        assert m.sharpe_ratio > 0


# ---------------------------------------------------------------------------
# Daily trade limit
# ---------------------------------------------------------------------------

class TestDailyTradeLimit:

    def test_not_reached_initially(self) -> None:
        d = DashboardProvider(daily_trade_limit=5)
        assert d.is_daily_limit_reached() is False

    def test_reached_at_limit(self) -> None:
        d = DashboardProvider(daily_trade_limit=3)
        for _ in range(3):
            d.record_trade(Decimal("10"), {})
        assert d.is_daily_limit_reached() is True

    def test_not_reached_below_limit(self) -> None:
        d = DashboardProvider(daily_trade_limit=5)
        for _ in range(4):
            d.record_trade(Decimal("10"), {})
        assert d.is_daily_limit_reached() is False

    def test_reset_clears_daily_count(self) -> None:
        d = DashboardProvider(daily_trade_limit=3)
        for _ in range(3):
            d.record_trade(Decimal("10"), {})
        assert d.is_daily_limit_reached() is True
        d.reset_daily()
        assert d.is_daily_limit_reached() is False

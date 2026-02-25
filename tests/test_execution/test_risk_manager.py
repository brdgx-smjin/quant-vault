"""Tests for risk manager."""

from decimal import Decimal

import pandas as pd
import pytest

from src.execution.risk_manager import RiskManager
from src.strategy.base import Signal, TradeSignal


@pytest.fixture
def risk_mgr():
    return RiskManager()


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------

def test_approve_normal_trade(risk_mgr: RiskManager) -> None:
    signal = TradeSignal(
        signal=Signal.LONG,
        symbol="BTC/USDT:USDT",
        price=50000.0,
        timestamp=pd.Timestamp.now(),
        stop_loss=49000.0,
    )
    result = risk_mgr.check_trade(signal, Decimal("10000"))
    assert result.approved is True
    assert result.position_size > 0


def test_reject_at_daily_limit(risk_mgr: RiskManager) -> None:
    risk_mgr.daily_pnl = Decimal("-600")  # Over 5% of 10000
    signal = TradeSignal(
        signal=Signal.LONG,
        symbol="BTC/USDT:USDT",
        price=50000.0,
        timestamp=pd.Timestamp.now(),
    )
    result = risk_mgr.check_trade(signal, Decimal("10000"))
    assert result.approved is False


def test_reject_max_positions(risk_mgr: RiskManager) -> None:
    risk_mgr.open_positions = 3
    signal = TradeSignal(
        signal=Signal.LONG,
        symbol="BTC/USDT:USDT",
        price=50000.0,
        timestamp=pd.Timestamp.now(),
    )
    result = risk_mgr.check_trade(signal, Decimal("10000"))
    assert result.approved is False


# ---------------------------------------------------------------------------
# Kelly fraction
# ---------------------------------------------------------------------------

def test_kelly_fraction_reduces_position_size(risk_mgr: RiskManager) -> None:
    """Kelly fraction (0.25) should reduce position size by 75%."""
    assert risk_mgr.sizing.get("method") == "kelly"
    signal = TradeSignal(
        signal=Signal.LONG,
        symbol="BTC/USDT:USDT",
        price=50000.0,
        timestamp=pd.Timestamp.now(),
        stop_loss=49000.0,
    )
    result = risk_mgr.check_trade(signal, Decimal("10000"))
    assert result.approved is True

    # Without Kelly: max_loss=200 / risk_per_unit=1000 -> 0.2 BTC
    # With Kelly 0.25: 0.2 * 0.25 = 0.05 BTC
    # But capped by max_notional = 10000 * 0.1 * 5 / 50000 = 0.1
    # 0.05 < 0.1 so Kelly is the binding constraint
    assert result.position_size == Decimal("0.05")


def test_kelly_fraction_value_from_config(risk_mgr: RiskManager) -> None:
    """Kelly fraction should be 0.25 from config."""
    assert risk_mgr.sizing.get("kelly_fraction") == 0.25


# ---------------------------------------------------------------------------
# Binance minimum notional
# ---------------------------------------------------------------------------

def test_reject_below_minimum_notional(risk_mgr: RiskManager) -> None:
    """Position with notional < $5 should be rejected."""
    signal = TradeSignal(
        signal=Signal.LONG,
        symbol="BTC/USDT:USDT",
        price=50000.0,
        timestamp=pd.Timestamp.now(),
        stop_loss=49999.0,  # Very tight SL -> tiny position
    )
    # Tiny balance -> tiny position
    result = risk_mgr.check_trade(signal, Decimal("1"))
    assert result.approved is False
    assert "too small" in result.reason.lower()


# ---------------------------------------------------------------------------
# Leverage capping
# ---------------------------------------------------------------------------

def test_leverage_capped_at_max(risk_mgr: RiskManager) -> None:
    """Leverage should be min(default, max) from config."""
    signal = TradeSignal(
        signal=Signal.LONG,
        symbol="BTC/USDT:USDT",
        price=50000.0,
        timestamp=pd.Timestamp.now(),
        stop_loss=49000.0,
    )
    result = risk_mgr.check_trade(signal, Decimal("10000"))
    assert result.leverage <= risk_mgr.risk["max_leverage"]
    assert result.leverage == min(
        risk_mgr.risk["default_leverage"],
        risk_mgr.risk["max_leverage"],
    )


# ---------------------------------------------------------------------------
# Invalid price
# ---------------------------------------------------------------------------

def test_reject_zero_price(risk_mgr: RiskManager) -> None:
    """Zero price signal should be rejected."""
    signal = TradeSignal(
        signal=Signal.LONG,
        symbol="BTC/USDT:USDT",
        price=0.0,
        timestamp=pd.Timestamp.now(),
    )
    result = risk_mgr.check_trade(signal, Decimal("10000"))
    assert result.approved is False
    assert "invalid" in result.reason.lower()


# ---------------------------------------------------------------------------
# Daily PnL tracking and loss levels
# ---------------------------------------------------------------------------

def test_update_daily_pnl(risk_mgr: RiskManager) -> None:
    """update_daily_pnl should accumulate."""
    risk_mgr.update_daily_pnl(Decimal("-100"))
    risk_mgr.update_daily_pnl(Decimal("-50"))
    assert risk_mgr.daily_pnl == Decimal("-150")


def test_reset_daily(risk_mgr: RiskManager) -> None:
    """reset_daily should zero out daily PnL."""
    risk_mgr.update_daily_pnl(Decimal("-200"))
    risk_mgr.reset_daily()
    assert risk_mgr.daily_pnl == Decimal("0")


def test_daily_loss_level_ok(risk_mgr: RiskManager) -> None:
    """Loss < 3% should return 'ok'."""
    risk_mgr.daily_pnl = Decimal("-200")  # 2% of 10000
    assert risk_mgr.get_daily_loss_level(Decimal("10000")) == "ok"


def test_daily_loss_level_warn(risk_mgr: RiskManager) -> None:
    """Loss 3-5% should return 'warn'."""
    risk_mgr.daily_pnl = Decimal("-350")  # 3.5% of 10000
    assert risk_mgr.get_daily_loss_level(Decimal("10000")) == "warn"


def test_daily_loss_level_critical(risk_mgr: RiskManager) -> None:
    """Loss >= 5% should return 'critical'."""
    risk_mgr.daily_pnl = Decimal("-500")  # 5% of 10000
    assert risk_mgr.get_daily_loss_level(Decimal("10000")) == "critical"


def test_daily_loss_level_profit_is_ok(risk_mgr: RiskManager) -> None:
    """Positive PnL should always be 'ok'."""
    risk_mgr.daily_pnl = Decimal("500")
    assert risk_mgr.get_daily_loss_level(Decimal("10000")) == "ok"

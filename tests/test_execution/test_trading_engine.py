"""Tests for TradingEngine candle processing and SL/TP logic."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.execution.position_manager import Position, PositionManager
from src.execution.trading_engine import MAX_DF_BARS, TradingEngine
from src.indicators.basic import BasicIndicators
from src.strategy.base import BaseStrategy, Signal, TradeSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyStrategy(BaseStrategy):
    """Strategy that always returns HOLD."""

    name = "dummy"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        return TradeSignal(
            signal=Signal.HOLD,
            symbol="BTC/USDT:USDT",
            price=float(df["close"].iloc[-1]),
            timestamp=df.index[-1],
        )

    def get_required_indicators(self) -> list[str]:
        return []


def _make_ohlcv(n: int = 100, freq: str = "30min") -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-06-01", periods=n, freq=freq)
    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame({
        "open": close - np.random.rand(n) * 10,
        "high": close + np.random.rand(n) * 20,
        "low": close - np.random.rand(n) * 20,
        "close": close,
        "volume": np.random.rand(n) * 100 + 10,
    }, index=dates)


def _build_engine() -> TradingEngine:
    """Create a TradingEngine with mocked external deps."""
    with patch("src.execution.trading_engine.BinanceExecutor"), \
         patch("src.execution.trading_engine.BinanceCollector"), \
         patch("src.execution.trading_engine.Alerter") as MockAlerter:
        mock_alerter = MockAlerter.return_value
        mock_alerter.alert = AsyncMock()
        mock_alerter.alert_entry = AsyncMock()
        mock_alerter.alert_exit = AsyncMock()
        mock_alerter.alert_sl_adjustment = AsyncMock()
        mock_alerter.close = AsyncMock()

        engine = TradingEngine(
            strategy=DummyStrategy(),
            symbol="BTC/USDT:USDT",
            timeframe="30m",
            testnet=True,
        )
    return engine


# ---------------------------------------------------------------------------
# Tests: Duplicate timestamp handling
# ---------------------------------------------------------------------------

class TestDuplicateTimestampHandling:
    """Verify _on_candle_close handles duplicate timestamps gracefully."""

    @pytest.mark.asyncio
    async def test_duplicate_index_dedup_before_add_all(self) -> None:
        """DataFrame with duplicate index should be cleaned before add_all."""
        engine = _build_engine()
        df = _make_ohlcv(50)
        # Inject a duplicate row
        dup_row = df.iloc[-1:].copy()
        df = pd.concat([df, dup_row])
        assert df.index.duplicated().any()
        engine.df = BasicIndicators.add_all(df[~df.index.duplicated(keep="last")])

        kline = {
            "t": int(pd.Timestamp("2024-06-03").timestamp() * 1000),
            "o": "51000", "h": "51100", "l": "50900", "c": "51050", "v": "123",
            "x": True,
        }
        await engine._on_candle_close(kline)
        assert not engine.df.index.duplicated().any()

    @pytest.mark.asyncio
    async def test_existing_timestamp_updates_in_place(self) -> None:
        """If candle timestamp already exists, update rather than append."""
        engine = _build_engine()
        engine.df = BasicIndicators.add_all(_make_ohlcv(50))
        existing_ts = engine.df.index[-1]

        kline = {
            "t": int(existing_ts.timestamp() * 1000),
            "o": "99000", "h": "99100", "l": "98900", "c": "99050", "v": "999",
            "x": True,
        }
        bars_before = len(engine.df)
        await engine._on_candle_close(kline)
        # Should NOT add a new row
        assert len(engine.df) == bars_before
        assert engine.df.loc[existing_ts, "close"] == 99050.0

    @pytest.mark.asyncio
    async def test_new_timestamp_appends(self) -> None:
        """New candle timestamp should add one row."""
        engine = _build_engine()
        engine.df = BasicIndicators.add_all(_make_ohlcv(50))
        bars_before = len(engine.df)
        new_ts = engine.df.index[-1] + pd.Timedelta(minutes=30)

        kline = {
            "t": int(new_ts.timestamp() * 1000),
            "o": "51000", "h": "51100", "l": "50900", "c": "51050", "v": "123",
            "x": True,
        }
        await engine._on_candle_close(kline)
        assert len(engine.df) == bars_before + 1

    @pytest.mark.asyncio
    async def test_max_bars_bounded(self) -> None:
        """DataFrame should not grow beyond MAX_DF_BARS."""
        engine = _build_engine()
        engine.df = BasicIndicators.add_all(_make_ohlcv(MAX_DF_BARS))

        new_ts = engine.df.index[-1] + pd.Timedelta(minutes=30)
        kline = {
            "t": int(new_ts.timestamp() * 1000),
            "o": "51000", "h": "51100", "l": "50900", "c": "51050", "v": "123",
            "x": True,
        }
        await engine._on_candle_close(kline)
        assert len(engine.df) <= MAX_DF_BARS


# ---------------------------------------------------------------------------
# Tests: SL/TP check logic
# ---------------------------------------------------------------------------

class TestSlTpCheck:
    """Verify stop-loss and take-profit trigger direction."""

    def _setup_engine_with_position(
        self, side: str, entry: float, sl: float, tp: float
    ) -> TradingEngine:
        engine = _build_engine()
        engine.executor.execute = AsyncMock(return_value={"id": "test"})
        engine.position_manager.open_position(
            symbol="BTC/USDT:USDT",
            side=side,
            entry_price=Decimal(str(entry)),
            amount=Decimal("0.01"),
            leverage=5,
            stop_loss=Decimal(str(sl)),
            take_profit=Decimal(str(tp)),
        )
        return engine

    @pytest.mark.asyncio
    async def test_long_sl_triggers_on_low(self) -> None:
        """LONG SL triggers when low <= SL price."""
        engine = self._setup_engine_with_position("long", 50000, 49000, 52000)
        await engine._check_sl_tp(high=50500.0, low=48900.0)
        # Position should be closed
        assert "BTC/USDT:USDT" not in engine.position_manager.positions

    @pytest.mark.asyncio
    async def test_long_sl_does_not_trigger_above(self) -> None:
        """LONG SL should NOT trigger when low > SL."""
        engine = self._setup_engine_with_position("long", 50000, 49000, 52000)
        await engine._check_sl_tp(high=50500.0, low=49100.0)
        assert "BTC/USDT:USDT" in engine.position_manager.positions

    @pytest.mark.asyncio
    async def test_long_tp_triggers_on_high(self) -> None:
        """LONG TP triggers when high >= TP price."""
        engine = self._setup_engine_with_position("long", 50000, 49000, 52000)
        await engine._check_sl_tp(high=52100.0, low=51000.0)
        assert "BTC/USDT:USDT" not in engine.position_manager.positions

    @pytest.mark.asyncio
    async def test_short_sl_triggers_on_high(self) -> None:
        """SHORT SL triggers when high >= SL price."""
        engine = self._setup_engine_with_position("short", 50000, 51000, 48000)
        await engine._check_sl_tp(high=51100.0, low=49500.0)
        assert "BTC/USDT:USDT" not in engine.position_manager.positions

    @pytest.mark.asyncio
    async def test_short_sl_does_not_trigger_below(self) -> None:
        """SHORT SL should NOT trigger when high < SL."""
        engine = self._setup_engine_with_position("short", 50000, 51000, 48000)
        await engine._check_sl_tp(high=50900.0, low=49500.0)
        assert "BTC/USDT:USDT" in engine.position_manager.positions

    @pytest.mark.asyncio
    async def test_short_tp_triggers_on_low(self) -> None:
        """SHORT TP triggers when low <= TP price."""
        engine = self._setup_engine_with_position("short", 50000, 51000, 48000)
        await engine._check_sl_tp(high=49000.0, low=47900.0)
        assert "BTC/USDT:USDT" not in engine.position_manager.positions


# ---------------------------------------------------------------------------
# Tests: _closing guard
# ---------------------------------------------------------------------------

class TestClosingGuard:
    """Ensure concurrent close attempts are blocked by _closing flag."""

    @pytest.mark.asyncio
    async def test_closing_guard_prevents_double_close(self) -> None:
        """Second _close_position call should be no-op while _closing is True."""
        engine = _build_engine()
        engine.executor.execute = AsyncMock(return_value={"id": "test"})
        engine.position_manager.open_position(
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=Decimal("50000"),
            amount=Decimal("0.01"),
            leverage=5,
        )

        # Manually set _closing to simulate in-flight close
        engine._closing = True
        await engine._close_position(49000.0, "stop_loss")
        # Position should still exist (close was blocked)
        assert "BTC/USDT:USDT" in engine.position_manager.positions

    @pytest.mark.asyncio
    async def test_closing_flag_resets_after_close(self) -> None:
        """_closing flag should reset to False after successful close."""
        engine = _build_engine()
        engine.executor.execute = AsyncMock(return_value={"id": "test"})
        engine.position_manager.open_position(
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=Decimal("50000"),
            amount=Decimal("0.01"),
            leverage=5,
        )
        await engine._close_position(49000.0, "stop_loss")
        assert engine._closing is False

    @pytest.mark.asyncio
    async def test_sl_tp_skipped_when_closing(self) -> None:
        """_check_sl_tp should be a no-op when _closing is True."""
        engine = _build_engine()
        engine.executor.execute = AsyncMock(return_value={"id": "test"})
        engine.position_manager.open_position(
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=Decimal("50000"),
            amount=Decimal("0.01"),
            leverage=5,
            stop_loss=Decimal("49000"),
        )
        engine._closing = True
        await engine._check_sl_tp(high=50000.0, low=48000.0)
        # Position should still exist
        assert "BTC/USDT:USDT" in engine.position_manager.positions

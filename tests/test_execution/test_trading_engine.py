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
from src.monitoring.dashboard import DashboardProvider


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

    @pytest.mark.asyncio
    async def test_multiple_duplicates_all_cleaned(self) -> None:
        """Multiple duplicate timestamps should all be resolved."""
        engine = _build_engine()
        df = _make_ohlcv(50)
        # Inject 3 copies of the same row
        dup1 = df.iloc[-1:].copy()
        dup2 = df.iloc[-2:-1].copy()
        df = pd.concat([df, dup1, dup2, dup1])
        engine.df = df  # Purposely don't clean — let _on_candle_close handle it

        new_ts = df.index[-1] + pd.Timedelta(minutes=60)
        kline = {
            "t": int(new_ts.timestamp() * 1000),
            "o": "51000", "h": "51100", "l": "50900", "c": "51050", "v": "123",
            "x": True,
        }
        await engine._on_candle_close(kline)
        assert not engine.df.index.duplicated().any()

    @pytest.mark.asyncio
    async def test_out_of_order_timestamp_sorted(self) -> None:
        """Out-of-order timestamps should be sorted after append."""
        engine = _build_engine()
        engine.df = BasicIndicators.add_all(_make_ohlcv(50))

        # Insert a timestamp that's before the last bar (out of order)
        early_ts = engine.df.index[10] + pd.Timedelta(seconds=1)
        kline = {
            "t": int(early_ts.timestamp() * 1000),
            "o": "51000", "h": "51100", "l": "50900", "c": "51050", "v": "123",
            "x": True,
        }
        await engine._on_candle_close(kline)
        assert engine.df.index.is_monotonic_increasing

    @pytest.mark.asyncio
    async def test_indicators_present_after_candle_close(self) -> None:
        """BasicIndicators columns should be present after processing."""
        engine = _build_engine()
        engine.df = BasicIndicators.add_all(_make_ohlcv(50))
        new_ts = engine.df.index[-1] + pd.Timedelta(minutes=30)

        kline = {
            "t": int(new_ts.timestamp() * 1000),
            "o": "51000", "h": "51100", "l": "50900", "c": "51050", "v": "123",
            "x": True,
        }
        await engine._on_candle_close(kline)
        expected_cols = ["rsi_14", "atr_14", "ema_20", "ema_50"]
        for col in expected_cols:
            assert col in engine.df.columns, f"Missing indicator column: {col}"


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

    @pytest.mark.asyncio
    async def test_exact_sl_price_triggers(self) -> None:
        """SL should trigger at exactly the SL price (boundary)."""
        engine = self._setup_engine_with_position("long", 50000, 49000, 52000)
        await engine._check_sl_tp(high=50500.0, low=49000.0)
        assert "BTC/USDT:USDT" not in engine.position_manager.positions

    @pytest.mark.asyncio
    async def test_exact_tp_price_triggers(self) -> None:
        """TP should trigger at exactly the TP price (boundary)."""
        engine = self._setup_engine_with_position("long", 50000, 49000, 52000)
        await engine._check_sl_tp(high=52000.0, low=51000.0)
        assert "BTC/USDT:USDT" not in engine.position_manager.positions

    @pytest.mark.asyncio
    async def test_no_position_is_noop(self) -> None:
        """_check_sl_tp should do nothing when no position exists."""
        engine = _build_engine()
        # Should not raise
        await engine._check_sl_tp(high=50000.0, low=49000.0)


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

    @pytest.mark.asyncio
    async def test_closing_resets_on_execution_failure(self) -> None:
        """_closing should reset to False if executor raises."""
        engine = _build_engine()
        engine.executor.execute = AsyncMock(side_effect=Exception("API error"))
        engine.position_manager.open_position(
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=Decimal("50000"),
            amount=Decimal("0.01"),
            leverage=5,
        )
        await engine._close_position(49000.0, "stop_loss")
        assert engine._closing is False
        # Position should still exist (close failed)
        assert "BTC/USDT:USDT" in engine.position_manager.positions


# ---------------------------------------------------------------------------
# Tests: Partial TP logic
# ---------------------------------------------------------------------------

class TestPartialTakeProfit:
    """Verify partial take-profit behavior."""

    def _setup_engine_with_partial_tp(self) -> TradingEngine:
        """Build engine with a LONG position and 3-level partial TP."""
        engine = _build_engine()
        engine.executor.execute = AsyncMock(return_value={"id": "test"})
        engine.position_manager.open_position(
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=Decimal("50000"),
            amount=Decimal("0.10"),
            leverage=5,
            stop_loss=Decimal("49000"),
            take_profit=Decimal("53000"),
        )
        pos = engine.position_manager.positions["BTC/USDT:USDT"]
        # 3 TP levels: 51000 (50%), 52000 (30%), 53000 (20%)
        pos.tp_levels = [
            (Decimal("51000"), Decimal("0.5")),
            (Decimal("52000"), Decimal("0.3")),
            (Decimal("53000"), Decimal("0.2")),
        ]
        pos.next_tp_idx = 0
        return engine

    @pytest.mark.asyncio
    async def test_partial_tp1_closes_fraction(self) -> None:
        """TP1 hit should close 50% and advance next_tp_idx."""
        engine = self._setup_engine_with_partial_tp()
        await engine._check_sl_tp(high=51100.0, low=50500.0)

        pos = engine.position_manager.positions.get("BTC/USDT:USDT")
        assert pos is not None, "Position should still exist (partial close)"
        assert pos.next_tp_idx == 1
        # 50% of 0.10 = 0.05 closed, remaining = 0.05
        assert pos.amount == Decimal("0.05")

    @pytest.mark.asyncio
    async def test_partial_tp1_moves_sl_to_breakeven(self) -> None:
        """After TP1, SL should move to entry price (breakeven)."""
        engine = self._setup_engine_with_partial_tp()
        await engine._check_sl_tp(high=51100.0, low=50500.0)

        pos = engine.position_manager.positions["BTC/USDT:USDT"]
        assert pos.stop_loss == Decimal("50000")

    @pytest.mark.asyncio
    async def test_final_tp_closes_remaining(self) -> None:
        """Last TP level should close the entire remaining position."""
        engine = self._setup_engine_with_partial_tp()
        pos = engine.position_manager.positions["BTC/USDT:USDT"]
        # Simulate TP1 and TP2 already hit
        pos.next_tp_idx = 2
        pos.amount = Decimal("0.02")

        await engine._check_sl_tp(high=53100.0, low=52500.0)
        # Position fully closed
        assert "BTC/USDT:USDT" not in engine.position_manager.positions

    @pytest.mark.asyncio
    async def test_partial_close_guard(self) -> None:
        """_partial_close should be blocked when _closing is True."""
        engine = self._setup_engine_with_partial_tp()
        engine._closing = True
        await engine._check_sl_tp(high=51100.0, low=50500.0)
        # Nothing should change
        pos = engine.position_manager.positions["BTC/USDT:USDT"]
        assert pos.amount == Decimal("0.10")
        assert pos.next_tp_idx == 0


# ---------------------------------------------------------------------------
# Tests: Signal reason enrichment
# ---------------------------------------------------------------------------

class TestSignalReasonEnrichment:
    """Verify _build_signal_reason generates reason from metadata."""

    def test_vwap_metadata_generates_reason(self) -> None:
        """VWAP signal metadata should produce z-score + RSI reason."""
        signal = TradeSignal(
            signal=Signal.LONG,
            symbol="BTC/USDT:USDT",
            price=50000.0,
            timestamp=pd.Timestamp.now(),
            metadata={"strategy": "vwap_mean_reversion", "z_score": -2.5, "rsi": 28.3},
        )
        reason = TradingEngine._build_signal_reason(signal)
        assert "VWAP" in reason
        assert "2.5" in reason
        assert "하단" in reason
        assert "RSI" in reason
        assert "28.3" in reason

    def test_short_vwap_shows_upper(self) -> None:
        """SHORT VWAP signal should show upper band touch."""
        signal = TradeSignal(
            signal=Signal.SHORT,
            symbol="BTC/USDT:USDT",
            price=50000.0,
            timestamp=pd.Timestamp.now(),
            metadata={"strategy": "vwap_mean_reversion", "z_score": 2.1, "rsi": 72.0},
        )
        reason = TradingEngine._build_signal_reason(signal)
        assert "상단" in reason

    def test_existing_reason_preserved(self) -> None:
        """If signal already has a reason, it should be returned as-is."""
        signal = TradeSignal(
            signal=Signal.LONG,
            symbol="BTC/USDT:USDT",
            price=50000.0,
            timestamp=pd.Timestamp.now(),
            reason="Custom reason",
            metadata={"z_score": -2.0, "rsi": 30.0},
        )
        reason = TradingEngine._build_signal_reason(signal)
        assert reason == "Custom reason"

    def test_empty_metadata_returns_strategy_name(self) -> None:
        """Empty metadata should fall back to strategy name."""
        signal = TradeSignal(
            signal=Signal.LONG,
            symbol="BTC/USDT:USDT",
            price=50000.0,
            timestamp=pd.Timestamp.now(),
            metadata={"strategy": "some_strat"},
        )
        reason = TradingEngine._build_signal_reason(signal)
        assert reason == "some_strat"

    def test_no_metadata_returns_empty(self) -> None:
        """No metadata and no reason should return empty string."""
        signal = TradeSignal(
            signal=Signal.LONG,
            symbol="BTC/USDT:USDT",
            price=50000.0,
            timestamp=pd.Timestamp.now(),
        )
        reason = TradingEngine._build_signal_reason(signal)
        assert reason == ""


# ---------------------------------------------------------------------------
# Tests: Daily trade limit integration
# ---------------------------------------------------------------------------

class TestDailyTradeLimit:
    """Verify engine respects daily trade limit from dashboard."""

    @pytest.mark.asyncio
    async def test_signal_skipped_at_daily_limit(self) -> None:
        """Signal should be skipped when daily trade limit reached."""
        engine = _build_engine()
        engine.df = BasicIndicators.add_all(_make_ohlcv(50))
        engine.dashboard = DashboardProvider(
            initial_equity=Decimal("5000"), daily_trade_limit=2,
        )
        # Record 2 trades to hit limit
        engine.dashboard.record_trade(Decimal("10"), {})
        engine.dashboard.record_trade(Decimal("20"), {})
        assert engine.dashboard.is_daily_limit_reached()

        signal = TradeSignal(
            signal=Signal.LONG,
            symbol="BTC/USDT:USDT",
            price=50000.0,
            timestamp=pd.Timestamp.now(),
            stop_loss=49000.0,
            take_profit=52000.0,
            metadata={"strategy": "test"},
        )
        # Should NOT execute (daily limit reached)
        await engine._execute_signal(signal)
        assert "BTC/USDT:USDT" not in engine.position_manager.positions

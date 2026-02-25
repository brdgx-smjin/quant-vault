"""Tests for strategy base and signal generation."""

import pandas as pd
import pytest

from src.strategy.base import Signal, TradeSignal
from src.strategy.fibonacci_retracement import FibonacciRetracementStrategy


def test_fibonacci_strategy_returns_signal(sample_ohlcv: pd.DataFrame) -> None:
    strategy = FibonacciRetracementStrategy()
    signal = strategy.generate_signal(sample_ohlcv)
    assert isinstance(signal, TradeSignal)
    assert signal.signal in Signal


def test_fibonacci_strategy_name() -> None:
    strategy = FibonacciRetracementStrategy()
    assert strategy.name == "fibonacci_retracement"

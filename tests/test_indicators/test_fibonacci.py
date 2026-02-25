"""Tests for Fibonacci calculator."""

import pandas as pd
import pytest

from src.indicators.fibonacci import FibonacciCalculator


@pytest.fixture
def fib():
    return FibonacciCalculator()


def test_retracement_levels(fib: FibonacciCalculator) -> None:
    levels = fib.calculate_retracement(50000, 40000, direction="up")
    assert len(levels) == 5
    # 61.8% retracement from 50000→40000 swing up = 50000 - 10000*0.618 = 43820
    l618 = [lv for lv in levels if abs(lv.ratio - 0.618) < 0.001][0]
    assert abs(l618.price - 43820) < 1


def test_extension_levels(fib: FibonacciCalculator) -> None:
    levels = fib.calculate_extension(50000, 40000, direction="up")
    assert len(levels) == 5
    # 1.618 extension = 50000 + 10000 * 0.618 = 56180
    l1618 = [lv for lv in levels if abs(lv.ratio - 1.618) < 0.001][0]
    assert abs(l1618.price - 56180) < 1


def test_find_swing_points(fib: FibonacciCalculator, sample_ohlcv: pd.DataFrame) -> None:
    high, low = fib.find_swing_points(sample_ohlcv, 20)
    assert high is not None
    assert low is not None
    assert high > low


def test_analyze(fib: FibonacciCalculator, sample_ohlcv: pd.DataFrame) -> None:
    result = fib.analyze(sample_ohlcv, 50)
    assert result is not None
    assert result.swing_high > result.swing_low
    assert len(result.levels) == 10  # 5 retracement + 5 extension

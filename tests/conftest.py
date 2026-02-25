"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    # Simulate random walk price
    returns = np.random.normal(0, 0.002, n)
    close = 40000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.random.uniform(0, 0.005, n))
    low = close * (1 - np.random.uniform(0, 0.005, n))
    open_ = close * (1 + np.random.normal(0, 0.001, n))
    volume = np.random.uniform(100, 1000, n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    return df

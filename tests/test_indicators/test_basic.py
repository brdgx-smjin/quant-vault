"""Tests for basic technical indicators."""

import pandas as pd
import pytest

from src.indicators.basic import BasicIndicators


def test_rsi(sample_ohlcv: pd.DataFrame) -> None:
    result = BasicIndicators.rsi(sample_ohlcv)
    assert result is not None
    assert len(result) == len(sample_ohlcv)
    # RSI should be between 0-100 (excluding NaN warmup)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_macd(sample_ohlcv: pd.DataFrame) -> None:
    result = BasicIndicators.macd(sample_ohlcv)
    assert result is not None
    assert len(result) == len(sample_ohlcv)


def test_bollinger_bands(sample_ohlcv: pd.DataFrame) -> None:
    result = BasicIndicators.bollinger_bands(sample_ohlcv)
    assert result is not None


def test_ema(sample_ohlcv: pd.DataFrame) -> None:
    result = BasicIndicators.ema(sample_ohlcv, 20)
    assert result is not None
    valid = result.dropna()
    assert len(valid) > 0


def test_atr(sample_ohlcv: pd.DataFrame) -> None:
    result = BasicIndicators.atr(sample_ohlcv)
    assert result is not None
    valid = result.dropna()
    assert (valid > 0).all()


def test_add_all(sample_ohlcv: pd.DataFrame) -> None:
    result = BasicIndicators.add_all(sample_ohlcv)
    assert "rsi_14" in result.columns
    assert "atr_14" in result.columns
    assert "ema_20" in result.columns

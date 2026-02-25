"""Data preprocessing utilities."""

from __future__ import annotations

import pandas as pd
import numpy as np


class DataPreprocessor:
    """Cleans and transforms raw OHLCV data."""

    @staticmethod
    def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, fill gaps, and validate OHLCV data.

        Args:
            df: Raw OHLCV DataFrame with timestamp index.

        Returns:
            Cleaned DataFrame.
        """
        df = df.copy()
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        # Drop rows with zero or negative values
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=numeric_cols)
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]

        return df

    @staticmethod
    def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to a different timeframe.

        Args:
            df: OHLCV DataFrame with timestamp index.
            timeframe: Target timeframe (e.g., '1h', '4h', '1d').

        Returns:
            Resampled DataFrame.
        """
        tf_map = {
            "1m": "1min", "5m": "5min", "15m": "15min",
            "1h": "1h", "4h": "4h", "1d": "1D",
        }
        rule = tf_map.get(timeframe, timeframe)
        return df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Add log returns and percentage returns columns.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with added return columns.
        """
        df = df.copy()
        df["pct_return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        return df

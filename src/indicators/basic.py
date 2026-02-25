"""Basic technical indicators: RSI, MACD, Bollinger Bands, EMA, ATR."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


class BasicIndicators:
    """Calculates standard technical indicators on OHLCV data."""

    @staticmethod
    def rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Relative Strength Index."""
        return ta.rsi(df["close"], length=length)

    @staticmethod
    def macd(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """MACD with signal line and histogram."""
        return ta.macd(df["close"], fast=fast, slow=slow, signal=signal)

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame, length: int = 20, std: float = 2.0
    ) -> pd.DataFrame:
        """Bollinger Bands (upper, mid, lower)."""
        return ta.bbands(df["close"], length=length, std=std)

    @staticmethod
    def ema(df: pd.DataFrame, length: int = 20) -> pd.Series:
        """Exponential Moving Average."""
        return ta.ema(df["close"], length=length)

    @staticmethod
    def sma(df: pd.DataFrame, length: int = 20) -> pd.Series:
        """Simple Moving Average."""
        return ta.sma(df["close"], length=length)

    @staticmethod
    def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Average True Range."""
        return ta.atr(df["high"], df["low"], df["close"], length=length)

    @staticmethod
    def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """Average Directional Index (ADX, +DI, -DI)."""
        return ta.adx(df["high"], df["low"], df["close"], length=length)

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """Add all basic indicators to the DataFrame.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with indicator columns added.
        """
        df = df.copy()
        df["rsi_14"] = BasicIndicators.rsi(df)
        df["atr_14"] = BasicIndicators.atr(df)
        df["ema_20"] = BasicIndicators.ema(df, 20)
        df["ema_50"] = BasicIndicators.ema(df, 50)
        df["ema_200"] = BasicIndicators.ema(df, 200)

        macd_df = BasicIndicators.macd(df)
        if macd_df is not None:
            df = pd.concat([df, macd_df], axis=1)

        bb_df = BasicIndicators.bollinger_bands(df)
        if bb_df is not None:
            df = pd.concat([df, bb_df], axis=1)

        adx_df = BasicIndicators.adx(df)
        if adx_df is not None:
            df = pd.concat([df, adx_df], axis=1)

        return df

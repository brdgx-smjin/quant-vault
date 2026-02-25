"""Ichimoku Cloud indicator."""

from __future__ import annotations

import pandas as pd


class IchimokuCalculator:
    """Calculates Ichimoku Kinko Hyo (Ichimoku Cloud) indicator."""

    def __init__(
        self,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
        chikou: int = 26,
    ) -> None:
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b
        self.chikou = chikou

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all Ichimoku components.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with Ichimoku columns added.
        """
        df = df.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Tenkan-sen (Conversion Line)
        df["tenkan_sen"] = (
            high.rolling(self.tenkan).max() + low.rolling(self.tenkan).min()
        ) / 2

        # Kijun-sen (Base Line)
        df["kijun_sen"] = (
            high.rolling(self.kijun).max() + low.rolling(self.kijun).min()
        ) / 2

        # Senkou Span A (Leading Span A) — shifted forward
        df["senkou_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(self.kijun)

        # Senkou Span B (Leading Span B) — shifted forward
        df["senkou_b"] = (
            (high.rolling(self.senkou_b).max() + low.rolling(self.senkou_b).min()) / 2
        ).shift(self.kijun)

        # Chikou Span (Lagging Span) — shifted backward
        df["chikou_span"] = close.shift(-self.chikou)

        return df

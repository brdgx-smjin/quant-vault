"""Volume Profile analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolumeProfileResult:
    """Volume profile analysis result."""

    poc: float  # Point of Control (highest volume price)
    value_area_high: float
    value_area_low: float
    profile: pd.Series  # price_level -> volume


class VolumeProfileCalculator:
    """Calculates Volume Profile (volume at price)."""

    def __init__(self, num_bins: int = 50, value_area_pct: float = 0.70) -> None:
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct

    def calculate(self, df: pd.DataFrame) -> VolumeProfileResult:
        """Calculate volume profile for the given data.

        Args:
            df: OHLCV DataFrame.

        Returns:
            VolumeProfileResult with POC and value area.
        """
        price_range = np.linspace(df["low"].min(), df["high"].max(), self.num_bins)
        volume_at_price = pd.Series(0.0, index=price_range[:-1])

        for _, row in df.iterrows():
            mask = (price_range[:-1] >= row["low"]) & (price_range[:-1] <= row["high"])
            matching_bins = mask.sum()
            if matching_bins > 0:
                volume_at_price[mask] += row["volume"] / matching_bins

        poc = float(volume_at_price.idxmax())
        total_volume = volume_at_price.sum()
        target_volume = total_volume * self.value_area_pct

        sorted_profile = volume_at_price.sort_values(ascending=False)
        cumulative = 0.0
        value_area_prices = []
        for price, vol in sorted_profile.items():
            cumulative += vol
            value_area_prices.append(price)
            if cumulative >= target_volume:
                break

        return VolumeProfileResult(
            poc=poc,
            value_area_high=max(value_area_prices),
            value_area_low=min(value_area_prices),
            profile=volume_at_price,
        )

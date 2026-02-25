"""Feature engineering for ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.basic import BasicIndicators


def build_features(
    df: pd.DataFrame,
    future_bars: int = 12,
    include_target: bool = True,
) -> pd.DataFrame:
    """Build feature matrix and optionally target from OHLCV data.

    Features:
      - Price-based: returns, volatility, range ratios
      - Indicator-based: RSI, MACD, BB position, EMA ratios
      - Volume-based: relative volume, volume trend
      - Pattern-based: higher highs/lows, candle body ratio

    Target (when include_target=True):
        1 if price rises > 0.5% in next `future_bars`, else 0.

    Args:
        df: OHLCV DataFrame.
        future_bars: Look-ahead bars for target label.
        include_target: If False, skip target creation (for live inference).
            This allows features to be computed up to the latest bar
            instead of dropping the last `future_bars` rows.

    Returns:
        DataFrame with features (and optionally 'target' column).
    """
    df = df.copy()
    if "rsi_14" not in df.columns:
        df = BasicIndicators.add_all(df)
    # Deduplicate columns if add_all was called multiple times
    df = df.loc[:, ~df.columns.duplicated()]
    feat = pd.DataFrame(index=df.index)

    # --- Price features ---
    feat["return_1"] = df["close"].pct_change(1)
    feat["return_3"] = df["close"].pct_change(3)
    feat["return_6"] = df["close"].pct_change(6)
    feat["return_12"] = df["close"].pct_change(12)

    feat["volatility_12"] = df["close"].pct_change().rolling(12).std()
    feat["volatility_24"] = df["close"].pct_change().rolling(24).std()

    feat["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    feat["body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)
    feat["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-10)
    feat["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-10)

    # --- Indicator features ---
    feat["rsi"] = df["rsi_14"]
    feat["rsi_delta"] = df["rsi_14"].diff(3)

    # MACD
    if "MACD_12_26_9" in df.columns:
        feat["macd"] = df["MACD_12_26_9"] / df["close"]
    if "MACDh_12_26_9" in df.columns:
        feat["macd_hist"] = df["MACDh_12_26_9"] / df["close"]
        feat["macd_hist_delta"] = feat["macd_hist"].diff(1)

    # Bollinger Band position
    bbl = [c for c in df.columns if c.startswith("BBL_")]
    bbu = [c for c in df.columns if c.startswith("BBU_")]
    if bbl and bbu:
        bb_range = df[bbu[0]] - df[bbl[0]]
        feat["bb_position"] = (df["close"] - df[bbl[0]]) / (bb_range + 1e-10)
        feat["bb_width"] = bb_range / df["close"]

    # EMA ratios
    feat["ema_20_ratio"] = df["close"] / df["ema_20"] - 1
    feat["ema_50_ratio"] = df["close"] / df["ema_50"] - 1
    feat["ema_cross"] = (df["ema_20"] - df["ema_50"]) / df["close"]

    # ATR normalized
    feat["atr_norm"] = df["atr_14"] / df["close"]

    # --- Volume features ---
    feat["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    feat["volume_trend"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()

    # --- Pattern features ---
    feat["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
    feat["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)
    feat["higher_close"] = (df["close"] > df["close"].shift(1)).astype(int)

    # Consecutive up/down
    up = (df["close"] > df["close"].shift(1)).astype(int)
    feat["consec_up"] = up.groupby((up != up.shift()).cumsum()).cumcount() * up
    down = (df["close"] < df["close"].shift(1)).astype(int)
    feat["consec_down"] = down.groupby((down != down.shift()).cumsum()).cumcount() * down

    # --- Target ---
    if include_target:
        future_return = df["close"].shift(-future_bars) / df["close"] - 1
        feat["target"] = (future_return > 0.005).astype(int)  # >0.5% = profitable long
        feat["future_return"] = future_return

    # Drop rows where FEATURES are NaN (rolling warmup).
    # When include_target=True, also drops last future_bars rows (NaN target).
    # When include_target=False, keeps rows up to the latest bar.
    feature_cols = [c for c in feat.columns if c not in ("target", "future_return")]
    feat = feat.dropna(subset=feature_cols)
    if include_target:
        feat = feat.dropna(subset=["target"])

    return feat


def split_train_test(
    feat: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split (no shuffle).

    Args:
        feat: Feature DataFrame with 'target' column.
        train_ratio: Fraction for training.

    Returns:
        (train_df, test_df) tuple.
    """
    split_idx = int(len(feat) * train_ratio)
    return feat.iloc[:split_idx], feat.iloc[split_idx:]


def purged_kfold_split(
    feat: pd.DataFrame,
    n_folds: int = 5,
    purge_bars: int = 12,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Purged expanding-window time-series cross-validation.

    Train window grows across folds (anchored at start).
    A gap of `purge_bars` between train and test eliminates target leakage
    caused by overlapping look-ahead labels.

    Args:
        feat: Feature DataFrame with 'target' column.
        n_folds: Number of CV folds.
        purge_bars: Gap between train end and test start (should match
            ``future_bars`` used when building features).

    Returns:
        List of (train_df, test_df) tuples.
    """
    n = len(feat)
    fold_size = n // (n_folds + 1)  # reserve space for expanding train

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for i in range(n_folds):
        # Expanding train: always starts at 0, grows each fold
        train_end = fold_size * (i + 1)
        # Purge gap eliminates look-ahead contamination
        test_start = train_end + purge_bars
        test_end = test_start + fold_size

        if test_end > n:
            test_end = n
        if test_start >= n:
            break

        train_df = feat.iloc[:train_end]
        test_df = feat.iloc[test_start:test_end]

        if len(train_df) < 100 or len(test_df) < 30:
            continue

        folds.append((train_df, test_df))

    return folds

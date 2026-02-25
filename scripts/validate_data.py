#!/usr/bin/env python3
"""Data quality validation script for BTC/USDT OHLCV parquet files.

Checks:
  1. Timestamp continuity (gap detection)
  2. OHLC logic (high >= max(open,close), low <= min(open,close))
  3. Zero-volume candle ratio
  4. Extreme moves (>10% single candle)
  5. Cross-timeframe consistency (1h close matches 4h candle close at aligned hours)

Results are written to logs/data_validation.log.

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --timeframes 1h 4h
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR, SYMBOL, TIMEFRAMES
from src.monitoring.logger import setup_logging

logger = setup_logging("data_validation")

# Expected interval per timeframe in minutes
TF_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}


def _parquet_path(tf: str) -> str:
    return f"{DATA_DIR}/processed/{SYMBOL.replace('/', '_').replace(':', '_')}_{tf}.parquet"


def check_timestamp_gaps(df: pd.DataFrame, tf: str) -> list[dict]:
    """Detect missing timestamps in the data.

    Returns:
        List of gap records with start, end, and missing count.
    """
    expected_freq = pd.Timedelta(minutes=TF_MINUTES[tf])
    diffs = df.index.to_series().diff()
    gaps = diffs[diffs > expected_freq * 1.5]

    gap_records = []
    for ts, delta in gaps.items():
        expected_count = int(delta / expected_freq) - 1
        prev_ts = ts - delta
        gap_records.append({
            "gap_start": prev_ts,
            "gap_end": ts,
            "missing_candles": expected_count,
            "duration": str(delta),
        })

    return gap_records


def check_ohlc_logic(df: pd.DataFrame) -> pd.DataFrame:
    """Verify OHLC relationships: high >= max(open,close), low <= min(open,close).

    Returns:
        DataFrame of rows that violate OHLC logic.
    """
    high_violation = df["high"] < np.maximum(df["open"], df["close"])
    low_violation = df["low"] > np.minimum(df["open"], df["close"])
    violations = df[high_violation | low_violation].copy()
    violations["high_violation"] = high_violation[violations.index]
    violations["low_violation"] = low_violation[violations.index]
    return violations


def check_zero_volume(df: pd.DataFrame) -> tuple[int, float]:
    """Count zero-volume candles.

    Returns:
        Tuple of (count, ratio).
    """
    zero_count = (df["volume"] == 0).sum()
    ratio = zero_count / len(df) if len(df) > 0 else 0.0
    return int(zero_count), float(ratio)


def check_extreme_moves(df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Detect candles with >threshold (default 10%) absolute price move.

    Returns:
        DataFrame of extreme move candles with move_pct column.
    """
    move = (df["close"] - df["open"]).abs() / df["open"]
    extreme = df[move > threshold].copy()
    extreme["move_pct"] = (move[extreme.index] * 100).round(2)
    return extreme


def check_cross_timeframe_consistency(
    df_lower: pd.DataFrame,
    df_upper: pd.DataFrame,
    lower_tf: str,
    upper_tf: str,
) -> list[dict]:
    """Check that lower timeframe close matches upper timeframe close.

    Candle timestamps represent the OPEN time. So a 4h candle at 04:00 covers
    04:00-08:00, and its close should match the last 1h candle's close within
    that period (i.e., the 1h candle at 07:00).

    For each upper-TF candle at time T, we compare its close with the lower-TF
    candle at T + (upper_interval - lower_interval).

    Returns:
        List of inconsistency records.
    """
    upper_minutes = TF_MINUTES[upper_tf]
    lower_minutes = TF_MINUTES[lower_tf]
    offset = pd.Timedelta(minutes=upper_minutes - lower_minutes)

    inconsistencies = []
    for upper_ts in df_upper.index:
        # The last lower-TF candle within this upper-TF period
        lower_ts = upper_ts + offset
        if lower_ts not in df_lower.index:
            continue

        lower_close = df_lower.loc[lower_ts, "close"]
        upper_close = df_upper.loc[upper_ts, "close"]
        if not np.isclose(lower_close, upper_close, rtol=1e-6):
            inconsistencies.append({
                "upper_timestamp": upper_ts,
                "lower_timestamp": lower_ts,
                "lower_close": lower_close,
                "upper_close": upper_close,
                "diff_pct": abs(lower_close - upper_close) / upper_close * 100,
            })

    return inconsistencies


def validate_timeframe(tf: str) -> dict:
    """Run all validations for a single timeframe.

    Returns:
        Dictionary with validation results.
    """
    path = _parquet_path(tf)
    if not Path(path).exists():
        logger.warning("File not found: %s", path)
        return {"status": "missing"}

    df = pd.read_parquet(path)
    results: dict = {
        "timeframe": tf,
        "rows": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
    }

    logger.info("=" * 60)
    logger.info("Validating %s: %d rows (%s to %s)", tf, len(df), df.index.min(), df.index.max())

    # 1. Timestamp gaps
    gaps = check_timestamp_gaps(df, tf)
    results["gap_count"] = len(gaps)
    total_missing = sum(g["missing_candles"] for g in gaps)
    results["total_missing_candles"] = total_missing
    if gaps:
        logger.warning("[%s] Found %d gaps (%d missing candles)", tf, len(gaps), total_missing)
        for g in gaps[:10]:  # Log first 10 gaps
            logger.warning(
                "  Gap: %s -> %s (%s, ~%d missing)",
                g["gap_start"], g["gap_end"], g["duration"], g["missing_candles"],
            )
        if len(gaps) > 10:
            logger.warning("  ... and %d more gaps", len(gaps) - 10)
    else:
        logger.info("[%s] No timestamp gaps found", tf)

    # 2. OHLC logic
    violations = check_ohlc_logic(df)
    results["ohlc_violations"] = len(violations)
    if len(violations) > 0:
        logger.warning("[%s] %d OHLC logic violations", tf, len(violations))
        for ts, row in violations.head(5).iterrows():
            logger.warning(
                "  %s: O=%.2f H=%.2f L=%.2f C=%.2f (high_viol=%s, low_viol=%s)",
                ts, row["open"], row["high"], row["low"], row["close"],
                row["high_violation"], row["low_violation"],
            )
    else:
        logger.info("[%s] OHLC logic OK", tf)

    # 3. Zero volume
    zero_count, zero_ratio = check_zero_volume(df)
    results["zero_volume_count"] = zero_count
    results["zero_volume_ratio"] = round(zero_ratio, 6)
    if zero_count > 0:
        logger.warning("[%s] %d zero-volume candles (%.4f%%)", tf, zero_count, zero_ratio * 100)
    else:
        logger.info("[%s] No zero-volume candles", tf)

    # 4. Extreme moves
    extreme = check_extreme_moves(df)
    results["extreme_move_count"] = len(extreme)
    if len(extreme) > 0:
        logger.warning("[%s] %d extreme moves (>10%%)", tf, len(extreme))
        for ts, row in extreme.head(5).iterrows():
            logger.warning(
                "  %s: O=%.2f C=%.2f (%.2f%% move)",
                ts, row["open"], row["close"], row["move_pct"],
            )
        if len(extreme) > 5:
            logger.warning("  ... and %d more", len(extreme) - 5)
    else:
        logger.info("[%s] No extreme moves (>10%%)", tf)

    return results


def validate_cross_timeframe() -> None:
    """Check 1h vs 4h cross-timeframe consistency."""
    path_1h = _parquet_path("1h")
    path_4h = _parquet_path("4h")

    if not Path(path_1h).exists() or not Path(path_4h).exists():
        logger.warning("Cannot check cross-timeframe: 1h or 4h data missing")
        return

    df_1h = pd.read_parquet(path_1h)
    df_4h = pd.read_parquet(path_4h)

    logger.info("=" * 60)
    logger.info("Cross-timeframe consistency check: 1h vs 4h")

    inconsistencies = check_cross_timeframe_consistency(df_1h, df_4h, "1h", "4h")
    if inconsistencies:
        logger.warning("Found %d cross-timeframe inconsistencies", len(inconsistencies))
        for inc in inconsistencies[:10]:
            logger.warning(
                "  4h@%s -> 1h@%s: 1h_close=%.2f, 4h_close=%.2f (diff=%.6f%%)",
                inc["upper_timestamp"], inc["lower_timestamp"],
                inc["lower_close"], inc["upper_close"], inc["diff_pct"],
            )
    else:
        logger.info("1h vs 4h cross-timeframe consistency OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BTC/USDT OHLCV data quality")
    parser.add_argument(
        "--timeframes", nargs="+", default=None,
        help="Specific timeframes to validate (default: all)",
    )
    args = parser.parse_args()

    timeframes = args.timeframes or TIMEFRAMES

    logger.info("Starting data validation for %s", SYMBOL)
    logger.info("Timeframes: %s", timeframes)

    all_results = []
    for tf in timeframes:
        result = validate_timeframe(tf)
        all_results.append(result)

    # Cross-timeframe check
    if "1h" in timeframes and "4h" in timeframes:
        validate_cross_timeframe()

    # Summary
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    for r in all_results:
        if r.get("status") == "missing":
            logger.info("  %s: FILE MISSING", r.get("timeframe", "?"))
            continue
        issues = []
        if r.get("gap_count", 0) > 0:
            issues.append(f"{r['gap_count']} gaps ({r['total_missing_candles']} missing)")
        if r.get("ohlc_violations", 0) > 0:
            issues.append(f"{r['ohlc_violations']} OHLC violations")
        if r.get("zero_volume_count", 0) > 0:
            issues.append(f"{r['zero_volume_count']} zero-vol")
        if r.get("extreme_move_count", 0) > 0:
            issues.append(f"{r['extreme_move_count']} extreme moves")

        status = "PASS" if not issues else "WARN"
        issue_str = "; ".join(issues) if issues else "all checks passed"
        logger.info("  %s [%s]: %d rows | %s", r["timeframe"], status, r["rows"], issue_str)

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()

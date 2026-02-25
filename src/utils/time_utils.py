"""Time-related utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def now_utc() -> datetime:
    """Current UTC datetime."""
    return datetime.now(timezone.utc)


def ms_to_datetime(ms: int) -> datetime:
    """Convert millisecond timestamp to datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """Convert datetime to millisecond timestamp."""
    return int(dt.timestamp() * 1000)


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Convert timeframe string to timedelta.

    Args:
        timeframe: e.g., '1m', '5m', '1h', '4h', '1d'.

    Returns:
        Corresponding timedelta.
    """
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    mapping = {
        "m": timedelta(minutes=value),
        "h": timedelta(hours=value),
        "d": timedelta(days=value),
        "w": timedelta(weeks=value),
    }
    return mapping.get(unit, timedelta(hours=1))

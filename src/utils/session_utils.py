"""
Session detection and regime classification utilities.
Institutional alignment: New York (America/New_York) timezone.
"""
from __future__ import annotations

import datetime
import zoneinfo
from typing import Any
import pandas as pd


ET_TZ = zoneinfo.ZoneInfo("America/New_York")


def parse_timestamp_to_utc(raw_ts: Any, fallback_to_now: bool = False) -> datetime.datetime | None:
    """
    Parses a raw timestamp (ISO string, datetime, epoch int/float in s/ms/us/ns) to UTC datetime.
    If raw_ts is None or NaN, defaults to current UTC datetime.
    If raw_ts is an invalid non-timestamp number or unparseable object, returns None (or now_utc if fallback_to_now=True).
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if raw_ts is None:
        return now_utc

    # Check for NaN float / pandas NA
    try:
        if pd.isna(raw_ts):
            return now_utc
    except (TypeError, ValueError):
        pass

    # Check if already a datetime
    if isinstance(raw_ts, datetime.datetime):
        if raw_ts.tzinfo is None:
            return raw_ts.replace(tzinfo=datetime.timezone.utc)
        return raw_ts.astimezone(datetime.timezone.utc)

    # Check if epoch numeric
    val = None
    try:
        val = float(raw_ts)
    except (ValueError, TypeError):
        pass

    if val is not None:
        # Guard against zero or small synthetic indices (e.g. index 0, 100)
        if val <= 1e7:
            return now_utc if fallback_to_now else None


        if val > 1e16:
            target_ts = pd.to_datetime(val, unit="ns", utc=True)
        elif val > 1e13:
            target_ts = pd.to_datetime(val, unit="us", utc=True)
        elif val > 1e10:
            target_ts = pd.to_datetime(val, unit="ms", utc=True)
        else:
            target_ts = pd.to_datetime(val, unit="s", utc=True)

        if target_ts is not None:
            dt = target_ts.floor("us").to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt
        return now_utc if fallback_to_now else None

    # String ISO format
    try:
        ts = pd.to_datetime(str(raw_ts).replace("Z", "+00:00"))
        if ts is not None and not pd.isna(ts):
            dt = ts.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt
    except Exception:
        pass

    return now_utc if fallback_to_now else None


def get_session_regime(
    raw_ts: Any = None,
    fallback_to_now: bool = False,
    include_london: bool = False,
) -> str:
    """
    Classifies a timestamp into a market session regime (America/New_York).
    
    Regimes:
    - NY RTH: 09:30 - 16:00 ET (Mon-Fri)
    - NY PRE: 08:00 - 09:30 ET (Mon-Fri)
    - LONDON: 03:00 - 08:00 ET (Mon-Fri, if include_london=True)
    - ASIAN: 18:00 - 02:00 ET (Sun evening - Fri morning)
    - GLOBEX: Off-hours electronic trading (e.g. 02:00 - 08:00 ET if include_london=False, or 16:00 - 17:00 ET)
    - MAINTENANCE: Mon-Thu 17:00 - 18:00 ET daily CME halt
    - CLOSED: Fri 17:00 ET through Sun 18:00 ET weekend closure
    - UNKNOWN: When timestamp cannot be parsed and fallback_to_now=False
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    target_dt = parse_timestamp_to_utc(raw_ts, fallback_to_now=fallback_to_now)

    if target_dt is None:
        return "UNKNOWN"

    # If fallback_to_now is enabled and bar timestamp is older than 30 minutes,
    # evaluate the session using current clock time.
    if fallback_to_now and abs((now_utc - target_dt).total_seconds()) > 1800:
        target_dt = now_utc

    try:
        et = target_dt.astimezone(ET_TZ)
        if et.year < 2000:
            return "UNKNOWN"

        weekday = et.weekday()  # 0=Mon, ..., 4=Fri, 5=Sat, 6=Sun
        mins = et.hour * 60 + et.minute

        # Weekend closure: Friday 17:00 ET through Sunday 18:00 ET
        if weekday == 4 and mins >= 1020:
            return "CLOSED"
        if weekday == 5:
            return "CLOSED"
        if weekday == 6 and mins < 1080:
            return "CLOSED"

        # Daily maintenance window (Mon-Thu 17:00-18:00 ET)
        if weekday in (0, 1, 2, 3) and 1020 <= mins < 1080:
            return "MAINTENANCE"

        if 570 <= mins < 960:    # 09:30 - 16:00 ET
            return "NY RTH"
        elif 480 <= mins < 570:  # 08:00 - 09:30 ET
            return "NY PRE"
        elif include_london and (180 <= mins < 480):  # 03:00 - 08:00 ET
            return "LONDON"
        elif 1080 <= mins or mins < 120:  # 18:00 - 02:00 ET
            return "ASIAN"
        return "GLOBEX"
    except Exception:
        return "UNKNOWN"


def get_trade_session(raw_ts: Any = None) -> str:
    """
    Helper for trade logging: returns the session the trade occurred in
    (including LONDON for European morning hours).
    Evaluates the timestamp of the trade without overriding with current clock time.
    """
    res = get_session_regime(raw_ts, fallback_to_now=False, include_london=True)
    if res in ("UNKNOWN", "CLOSED", "MAINTENANCE"):
        return "GLOBEX"
    return res


from __future__ import annotations

import math
from typing import Any

import pandas as pd

TIMEFRAME_RULES = {
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


def normalize_frame(frame: str) -> str:
    normalized = str(frame or "1m").strip().lower()
    aliases = {
        "1min": "1m",
        "2min": "2m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "1hr": "1h",
        "4hr": "4h",
        "1day": "1d",
        "daily": "1d",
    }
    return aliases.get(normalized, normalized)


def chart_time(value: Any) -> int:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return 0
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())


def bars_to_chart_records(frame: pd.DataFrame, *, limit: int = 500) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    out = []
    required = ["timestamp", "open", "high", "low", "close"]
    clean = frame.copy()
    if any(col not in clean.columns for col in required):
        return []
    clean = clean.dropna(subset=required).tail(int(limit or 500))
    for row in clean.to_dict(orient="records"):
        item = {
            "time": chart_time(row.get("timestamp")),
            "open": _finite_float(row.get("open")),
            "high": _finite_float(row.get("high")),
            "low": _finite_float(row.get("low")),
            "close": _finite_float(row.get("close")),
            "volume": _finite_float(row.get("volume", 0.0)),
        }
        if item["time"] > 0 and all(item[key] is not None for key in ("open", "high", "low", "close")):
            out.append(item)
    return out


def resample_timeframe(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    normalized = normalize_frame(timeframe)
    if normalized == "1m":
        return _normalize_bars(frame)
    if normalized not in TIMEFRAME_RULES:
        raise ValueError(f"Unsupported time frame '{timeframe}'")
    base = _normalize_bars(frame)
    if base.empty:
        return base
    indexed = base.set_index("timestamp")
    resampled = indexed.resample(TIMEFRAME_RULES[normalized]).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "price": "last",
        }
    )
    return resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()


def _normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "price"])
    clean = frame.copy()
    clean.columns = [str(col).lower() for col in clean.columns]
    if "timestamp" not in clean.columns:
        clean.loc[:, "timestamp"] = pd.RangeIndex(start=0, stop=len(clean), step=1)
    clean.loc[:, "timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
    clean = clean.dropna(subset=["timestamp"]).sort_values("timestamp")
    for col in ("open", "high", "low", "close"):
        if col not in clean.columns:
            clean.loc[:, col] = clean.get("price", 0.0)
        clean.loc[:, col] = pd.to_numeric(clean[col], errors="coerce")
    if "price" not in clean.columns:
        clean.loc[:, "price"] = clean["close"]
    clean.loc[:, "price"] = pd.to_numeric(clean["price"], errors="coerce")
    if "volume" not in clean.columns:
        clean.loc[:, "volume"] = 0.0
    clean.loc[:, "volume"] = pd.to_numeric(clean["volume"], errors="coerce").fillna(0.0)
    return clean.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None

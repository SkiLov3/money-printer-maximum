from __future__ import annotations

import html
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


TIMEFRAME_RULES = {
    "1m": None,
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1D": "1D",
    "1W": "1W",
}


def build_trade_key(symbol: str, order_id: str | None, opened_at: float | str | None) -> str:
    if order_id:
        return str(order_id)
    raw = f"{symbol}-{opened_at or datetime.now(timezone.utc).isoformat()}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")


def capture_trade_context_artifacts(
    *,
    bars: pd.DataFrame,
    trade_key: str,
    symbol: str,
    opened_at: datetime,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    if not bool(cfg.get("enabled", True)):
        return {"enabled": False, "timeframes": {}}

    timeframes = cfg.get("timeframes") or ["1m", "5m", "15m", "30m", "1h", "4h", "1D", "1W"]
    min_chart_bars = int(cfg.get("min_chart_bars", 8) or 8)
    max_chart_bars = int(cfg.get("max_chart_bars", 120) or 120)
    artifact_dir = Path(str(cfg.get("artifact_dir") or "reports/learning_review/trade_contexts"))
    safe_trade_key = re.sub(r"[^A-Za-z0-9_.-]+", "-", trade_key).strip("-") or "trade"
    out_dir = artifact_dir / safe_trade_key
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _normalize_bars(bars)
    artifacts: dict[str, Any] = {
        "enabled": True,
        "symbol": symbol,
        "trade_key": trade_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timeframes": {},
    }
    for timeframe in timeframes:
        tf = str(timeframe)
        try:
            frame = _resample_timeframe(base, tf)
            if len(frame) < min_chart_bars:
                artifacts["timeframes"][tf] = {
                    "status": "insufficient_data",
                    "bars": len(frame),
                    "minimum_bars": min_chart_bars,
                }
                continue
            window = frame.tail(max_chart_bars)
            svg_path = out_dir / f"{tf}.svg"
            json_path = out_dir / f"{tf}.json"
            _write_svg_chart(window, svg_path, title=f"{symbol} {tf}", marker_time=opened_at)
            json_path.write_text(
                json.dumps(
                    {
                        "symbol": symbol,
                        "trade_key": trade_key,
                        "timeframe": tf,
                        "bars": _bars_to_records(window),
                        "opened_at": opened_at.isoformat(),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            artifacts["timeframes"][tf] = {
                "status": "captured",
                "bars": len(window),
                "svg_path": str(svg_path),
                "data_path": str(json_path),
            }
        except Exception as exc:
            logger.debug("Learning chart capture failed for %s %s: %s", symbol, tf, exc)
            artifacts["timeframes"][tf] = {"status": "error", "error": str(exc)}
    return artifacts


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                "open": pd.Series(dtype="float64"),
                "high": pd.Series(dtype="float64"),
                "low": pd.Series(dtype="float64"),
                "close": pd.Series(dtype="float64"),
                "volume": pd.Series(dtype="float64"),
            }
        )
    frame = bars.copy()
    frame.columns = [str(col).lower() for col in frame.columns]
    if "timestamp" not in frame.columns:
        frame.loc[:, "timestamp"] = pd.RangeIndex(start=0, stop=len(frame), step=1)
    frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    for col in ("open", "high", "low", "close"):
        if col not in frame.columns:
            frame.loc[:, col] = frame.get("price", 0.0)
        frame.loc[:, col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" not in frame.columns:
        frame.loc[:, "volume"] = 0.0
    frame.loc[:, "volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    return frame.dropna(subset=["open", "high", "low", "close"])


def _resample_timeframe(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = TIMEFRAME_RULES.get(timeframe)
    if rule is None:
        return frame.reset_index(drop=True)
    if frame.empty:
        return frame.reset_index(drop=True)
    indexed = frame.set_index("timestamp")
    out = indexed.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def _bars_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in frame.tail(120).to_dict(orient="records"):
        item = dict(row)
        if hasattr(item.get("timestamp"), "isoformat"):
            item["timestamp"] = item["timestamp"].isoformat()
        records.append(item)
    return records


def _write_svg_chart(frame: pd.DataFrame, path: Path, *, title: str, marker_time: datetime) -> None:
    frame = frame.reset_index(drop=True)
    width, height = 980, 520
    pad_left, pad_right, pad_top, pad_bottom = 58, 24, 38, 36
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom
    lows = frame["low"].astype(float)
    highs = frame["high"].astype(float)
    min_price = float(lows.min())
    max_price = float(highs.max())
    if max_price <= min_price:
        max_price = min_price + 1.0
    price_span = max_price - min_price

    def y_for(price: float) -> float:
        return pad_top + (max_price - price) / price_span * chart_h

    n = max(len(frame), 1)
    step = chart_w / max(n, 1)
    body_w = max(2.0, min(10.0, step * 0.55))
    candles = []
    for idx, row in enumerate(frame.itertuples(index=False)):
        x = pad_left + idx * step + step / 2
        open_y = y_for(float(row.open))
        close_y = y_for(float(row.close))
        high_y = y_for(float(row.high))
        low_y = y_for(float(row.low))
        up = float(row.close) >= float(row.open)
        color = "#16a34a" if up else "#dc2626"
        body_y = min(open_y, close_y)
        body_h = max(1.0, abs(open_y - close_y))
        candles.append(f'<line x1="{x:.2f}" y1="{high_y:.2f}" x2="{x:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="1.4"/>')
        candles.append(
            f'<rect x="{x - body_w / 2:.2f}" y="{body_y:.2f}" width="{body_w:.2f}" height="{body_h:.2f}" '
            f'fill="{color}" opacity="0.9"/>'
        )

    marker_svg = ""
    marker_ts = pd.to_datetime(marker_time)
    if getattr(marker_ts, "tzinfo", None) is not None:
        marker_ts = marker_ts.tz_convert("UTC").tz_localize(None)
    timestamps = pd.to_datetime(frame["timestamp"])
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    if not timestamps.empty and timestamps.min() <= marker_ts <= timestamps.max():
        idx = int((timestamps - marker_ts).abs().idxmin())
        x = pad_left + idx * step + step / 2
        marker_svg = f'<line x1="{x:.2f}" y1="{pad_top}" x2="{x:.2f}" y2="{height - pad_bottom}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="5 5"/>'

    axis_labels = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        price = max_price - frac * price_span
        y = pad_top + frac * chart_h
        axis_labels.append(f'<line x1="{pad_left}" y1="{y:.2f}" x2="{width - pad_right}" y2="{y:.2f}" stroke="#1f2937" stroke-width="1"/>')
        axis_labels.append(f'<text x="8" y="{y + 4:.2f}" fill="#cbd5e1" font-size="12">{price:.2f}</text>')

    escaped_title = html.escape(title)
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#0b1220"/>',
            f'<text x="{pad_left}" y="24" fill="#f8fafc" font-family="Arial, sans-serif" font-size="18" font-weight="700">{escaped_title}</text>',
            *axis_labels,
            f'<rect x="{pad_left}" y="{pad_top}" width="{chart_w}" height="{chart_h}" fill="none" stroke="#334155" stroke-width="1"/>',
            *candles,
            marker_svg,
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScreenshotConfig:
    enabled: bool = False
    every_signal: bool = True
    every_n_bars: int = 0
    max_captures: int = 12
    out_dir: str = "reports/backtest/optimizer_screenshots"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ScreenshotConfig:
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            every_signal=bool(data.get("every_signal", True)),
            every_n_bars=int(data.get("every_n_bars", 0) or 0),
            max_captures=int(data.get("max_captures", 12) or 12),
            out_dir=str(data.get("out_dir") or "reports/backtest/optimizer_screenshots"),
        )


class ScreenshotCapture:
    def __init__(self, config: ScreenshotConfig, run_id: str = "run") -> None:
        self._cfg = config
        self._run_id = run_id
        self._out_dir: Path | None = None
        self._bars_since_last_capture = 0
        self._capture_count = 0

    def capture(
        self,
        candles: Any,
        timestamp: datetime,
        strategy_name: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self._cfg.enabled:
            return None
        if self._capture_count >= self._cfg.max_captures:
            return None
        if event_type == "signal_generated" and not self._cfg.every_signal:
            return None
        try:
            import mplfinance as mpf  # type: ignore[import-untyped]
        except Exception as exc:
            logger.warning("Screenshot capture unavailable: %s", exc)
            return None

        try:
            out_dir = self._ensure_out_dir()
            filename = self._filename(timestamp, strategy_name, event_type)
            png_path = out_dir / f"{filename}.png"
            json_path = out_dir / f"{filename}.json"
            window = candles.tail(100) if hasattr(candles, "tail") else candles
            mpf.plot(
                window,
                type="candle",
                style="nightclouds",
                volume="volume" in getattr(window, "columns", []),
                savefig={"fname": str(png_path), "dpi": 80},
            )
            sidecar = {
                "run_id": self._run_id,
                "timestamp_utc": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
                "strategy_name": strategy_name,
                "event_type": event_type,
                "window_bars": len(window) if hasattr(window, "__len__") else 0,
                **(metadata or {}),
            }
            json_path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
            self._capture_count += 1
            return png_path
        except Exception as exc:
            logger.warning("Screenshot capture failed: %s", exc)
            return None

    def on_bar_end(self) -> bool:
        if not self._cfg.enabled or self._cfg.every_n_bars <= 0:
            return False
        self._bars_since_last_capture += 1
        if self._bars_since_last_capture >= self._cfg.every_n_bars:
            self._bars_since_last_capture = 0
            return True
        return False

    def _ensure_out_dir(self) -> Path:
        if self._out_dir is None:
            self._out_dir = Path(self._cfg.out_dir) / self._run_id
            self._out_dir.mkdir(parents=True, exist_ok=True)
        return self._out_dir

    def _filename(self, timestamp: datetime, strategy_name: str, event_type: str) -> str:
        ts = timestamp.strftime("%Y%m%dT%H%M%S") if hasattr(timestamp, "strftime") else str(timestamp)
        ts = ts.replace(":", "").replace(" ", "_")
        return f"{strategy_name}_{ts}_{event_type}"

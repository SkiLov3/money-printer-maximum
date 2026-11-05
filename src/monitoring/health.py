from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any


class StrategyHealthMonitor:
    def __init__(self, config: dict[str, Any] | None = None, database=None):
        self.config = config or {}
        self.database = database
        self.rejections = defaultdict(int)
        self.accepts = defaultdict(int)
        self.last_signal_at: dict[str, float] = {}
        self.last_accept_at: dict[str, float] = {}
        self.events = deque(maxlen=int(self.config.get("max_events", 50)))

    def initialize_symbols(self, symbols: list[str]) -> None:
        now = datetime.now(timezone.utc).timestamp()
        for sym in symbols:
            if sym not in self.last_signal_at:
                self.last_signal_at[sym] = now

    def record_signal(self, symbol: str, accepted: bool, reason: str = "") -> None:
        now = datetime.now(timezone.utc).timestamp()
        self.last_signal_at[symbol] = now
        if accepted:
            self.accepts[symbol] += 1
            self.last_accept_at[symbol] = now
        else:
            self.rejections[symbol] += 1
        event = {
            "symbol": symbol,
            "accepted": accepted,
            "reason": reason,
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        }
        self.events.append(event)
        if self.database and getattr(self.database, "connected", False):
            try:
                self.database.persist_health_event(symbol, accepted, reason)
            except Exception:
                pass

    def record_event(self, event_type: str, detail: str = "") -> None:
        event = {
            "symbol": event_type,
            "accepted": True,
            "reason": detail,
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        }
        self.events.append(event)
        if self.database and getattr(self.database, "connected", False):
            try:
                self.database.persist_health_event(event_type, True, detail)
            except Exception:
                pass

    def get_relaxation_recommendation(self, symbol: str) -> dict[str, Any]:
        total = self.accepts[symbol] + self.rejections[symbol]
        rejection_rate = (self.rejections[symbol] / total) if total else 0.0
        now = datetime.now(timezone.utc).timestamp()
        last_ts = self.last_signal_at.get(symbol, 0.0)
        drought_seconds = round(now - last_ts, 1) if last_ts else 0.0
        window = float(self.config.get("drought_threshold_seconds", 1800))
        action = "none"
        if drought_seconds >= window and rejection_rate >= float(self.config.get("rejection_rate_threshold", 0.7)):
            action = "relax_filters"
        elif rejection_rate < 0.3 and self.accepts[symbol] >= int(self.config.get("tighten_after_accepts", 5)):
            action = "tighten_filters"
        return {
            "symbol": symbol,
            "action": action,
            "rejection_rate": round(rejection_rate, 3),
            "drought_seconds": drought_seconds,
        }

    def get_status(self) -> dict[str, Any]:
        drought_seconds = {}
        acceptance_ratio = {}
        now = datetime.now(timezone.utc).timestamp()
        symbols = set(self.last_signal_at.keys()) | set(self.accepts.keys()) | set(self.rejections.keys())
        for symbol in symbols:
            last_ts = self.last_signal_at.get(symbol, 0.0)
            drought_seconds[symbol] = round(now - last_ts, 1) if last_ts else 0.0
            total = self.accepts[symbol] + self.rejections[symbol]
            acceptance_ratio[symbol] = round((self.accepts[symbol] / total), 3) if total else 0.0
        recommendations = {symbol: self.get_relaxation_recommendation(symbol) for symbol in symbols}
        return {
            "accepts": dict(self.accepts),
            "rejections": dict(self.rejections),
            "acceptance_ratio": acceptance_ratio,
            "drought_seconds": drought_seconds,
            "recommendations": recommendations,
            "recent_events": list(self.events),
        }

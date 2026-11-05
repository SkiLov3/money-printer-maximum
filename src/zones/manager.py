from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _ManagedZone:
    zone_id: int
    zone: Any
    zone_family: str
    invalidation_reason: str | None = None


class ZoneManager:
    def __init__(self, merge_overlap: bool = True, max_invalidated_age_bars: int = 500):
        self.merge_overlap = merge_overlap
        self.max_invalidated_age_bars = max_invalidated_age_bars
        self._zones: list[_ManagedZone] = []
        self._next_id = 1
        self._invalidated_ages: dict[int, int] = {}

    @property
    def zones(self) -> list[Any]:
        return [managed.zone for managed in self._zones]

    @property
    def active_zones(self) -> list[Any]:
        return [managed.zone for managed in self._zones if getattr(managed.zone, "status", "active") in ("active", "tested")]

    def add_zone(self, zone: Any) -> int:
        if self.merge_overlap:
            merged = self._try_merge(zone)
            if merged is not None:
                return merged
        zone_id = self._next_id
        self._next_id += 1
        self._zones.append(_ManagedZone(zone_id=zone_id, zone=zone, zone_family=self._classify_family(zone)))
        return zone_id

    def invalidate(self, zone_id: int, reason: str = "") -> None:
        managed = self._find(zone_id)
        if managed is None:
            return
        if hasattr(managed.zone, "status"):
            managed.zone.status = "invalidated"
        managed.invalidation_reason = reason
        self._invalidated_ages[zone_id] = 0

    def get_nearby_zones(self, price: float, atr: float, max_distance_atr: float = 2.0) -> list[Any]:
        proximity = max_distance_atr * atr
        nearby: list[tuple[float, Any]] = []
        for managed in self._zones:
            zone = managed.zone
            if getattr(zone, "status", "active") == "invalidated":
                continue
            z_high = float(zone.price_high)
            z_low = float(zone.price_low)
            if z_low <= price <= z_high:
                dist = 0.0
            else:
                dist = min(abs(price - z_high), abs(price - z_low))
            if dist <= proximity:
                nearby.append((dist, zone))
        nearby.sort(key=lambda item: item[0])
        return [zone for _, zone in nearby]

    def maintenance(self) -> None:
        stale_ids = []
        for zone_id in list(self._invalidated_ages.keys()):
            self._invalidated_ages[zone_id] += 1
            if self._invalidated_ages[zone_id] > self.max_invalidated_age_bars:
                stale_ids.append(zone_id)
        if stale_ids:
            self._zones = [managed for managed in self._zones if managed.zone_id not in stale_ids]
            for zone_id in stale_ids:
                self._invalidated_ages.pop(zone_id, None)

    def _try_merge(self, zone: Any) -> int | None:
        zone_type = getattr(zone, "zone_type", "")
        for managed in self._zones:
            existing = managed.zone
            if getattr(existing, "status", "active") == "invalidated":
                continue
            if getattr(existing, "zone_type", "") != zone_type:
                continue
            if float(zone.price_low) <= float(existing.price_high) and float(zone.price_high) >= float(existing.price_low):
                existing.price_low = min(float(existing.price_low), float(zone.price_low))
                existing.price_high = max(float(existing.price_high), float(zone.price_high))
                return managed.zone_id
        return None

    def _classify_family(self, zone: Any) -> str:
        zone_type = getattr(zone, "zone_type", "generic")
        if zone_type in ("support", "resistance"):
            return "sr"
        if zone_type in ("supply", "demand"):
            return "sd"
        if zone_type == "pivot":
            return "pivot"
        return "generic"

    def _find(self, zone_id: int) -> _ManagedZone | None:
        for managed in self._zones:
            if managed.zone_id == zone_id:
                return managed
        return None

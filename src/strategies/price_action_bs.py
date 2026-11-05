from __future__ import annotations

import math
import zoneinfo
from collections.abc import Iterable
from datetime import timezone
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
import yaml

from src.charting import chart_time

from .base import BaseStrategy, StrategySignal


class PriceActionBSStrategy(BaseStrategy):
    name = "price_action_bs"
    warmup_bars = 80

    DEFAULT_SOURCE_FILES = [
        "pa_strat/playbook.md",
        "pa_strat/setups.yaml",
        "pa_strat/glossary.md",
        "pa_strat/open_questions.md",
    ]

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.params = self._apply_symbol_overrides(self.params)
        self.repo_root = Path(__file__).resolve().parents[2]
        self.source_files = list(self.params.get("source_files", self.DEFAULT_SOURCE_FILES))
        self.setups_path = self._resolve_path(str(self.params.get("setups_path", "pa_strat/setups.yaml")))
        self.spec = self._load_spec(self.setups_path)
        self.metadata = dict(self.spec.get("metadata") or {})
        self.setup_cards = dict(self.spec.get("setups") or {})
        locks = dict(self.spec.get("universal_locks") or {})
        guardrails = dict(self.spec.get("guardrails_locked") or {})

        self.enabled_symbols = [str(s).upper() for s in self.params.get("enabled_symbols", [])]
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.ema_period = int(self.params.get("ema_period", self.metadata.get("ema_period", 21)))
        self.ema_column = f"ema_{self.ema_period}"
        self.scalp_target_ticks = int(self.params.get("scalp_target_ticks", locks.get("scalp_target_ticks", 10)))
        self.target_mode = str(self.params.get("target_mode", "fixed_ticks")).lower()
        self.risk_reward = float(self.params.get("risk_reward", self.params.get("target_r_multiple", 3.0)))
        self.min_target_ticks = int(self.params.get("min_target_ticks", self.scalp_target_ticks))
        self.max_target_ticks = int(self.params.get("max_target_ticks", 0) or 0)
        self.target_ticks_by_setup = {
            str(key): int(value)
            for key, value in dict(self.params.get("target_ticks_by_setup", {}) or {}).items()
            if value is not None
        }
        self.use_candidate_price_targets = bool(self.params.get("use_candidate_price_targets", False))
        self.enabled_setups = {
            str(item)
            for item in self.params.get(
                "enabled_setups",
                ["E-01", "E-02", "E-03", "E-04", "E-05", "E-06", "E-07", "E-08"],
            )
        }
        self.disabled_setup_sides = self._parse_disabled_setup_sides(self.params.get("disabled_setup_sides", []))
        self.max_stop_ticks = int(self.params.get("max_stop_ticks", 0) or 0)
        self.min_stop_ticks = int(self.params.get("min_stop_ticks", 0) or 0)
        self.min_stop_atr_mult = float(self.params.get("min_stop_atr_mult", 0.0) or 0.0)
        self.skip_large_signal_bars = bool(self.params.get("skip_large_signal_bars", False))
        self.min_signal_bar_atr_ratio = float(self.params.get("min_signal_bar_atr_ratio", 0.0) or 0.0)
        self.min_body_ratio = float(self.params.get("min_body_ratio", 0.15) or 0.15)
        self.min_close_ema_distance_ticks = float(self.params.get("min_close_ema_distance_ticks", 0.0) or 0.0)
        self.ema_distance_ticks = int(self.params.get("ema_distance_filter_ticks", locks.get("ema_distance_filter_ticks", 10)))
        self.large_signal_bar_threshold_ticks = int(
            self.params.get("large_signal_bar_threshold_ticks", locks.get("large_signal_bar_threshold_ticks", 12))
        )
        self.fractal_length = int(
            self.params.get(
                "fractal_length",
                (((self.spec.get("primitives") or {}).get("primitives") or {}).get("swing_pivot") or {}).get(
                    "fractal_length",
                    2,
                ),
            )
        )
        self.min_leg_ticks = int(self.params.get("min_leg_ticks", 2)) # [GOLDILOCKS] Relaxed from 4
        self.min_leg_bars = int(self.params.get("min_leg_bars", 1)) # [GOLDILOCKS] Relaxed from 2
        self.pullback_lookback_bars = int(self.params.get("pullback_lookback_bars", 40))
        self.signal_after_pivot_bars = int(self.params.get("signal_after_pivot_bars", 10))
        self.failure_window_bars = int(self.params.get("failure_window_bars", 5))
        self.range_detection_bars = int(self.params.get("range_detection_bars", 60))
        self.range_cluster_tolerance_ticks = int(self.params.get("range_cluster_tolerance_ticks", 2))
        self.range_min_width_ticks = int(self.params.get("range_min_width_ticks", 8))
        self.range_max_width_ticks = int(self.params.get("range_max_width_ticks", 80))
        self.range_double_test_tolerance_ticks = int(self.params.get("range_double_test_tolerance_ticks", 2))
        self.range_double_test_lookback_bars = int(self.params.get("range_double_test_lookback_bars", 20))
        self.range_double_test_min_separation_bars = int(self.params.get("range_double_test_min_separation_bars", 3))
        self.range_double_test_breakout_ticks = int(self.params.get("range_double_test_breakout_ticks", 1))
        self.range_double_test_use_visual_range = bool(self.params.get("range_double_test_use_visual_range", True))
        self.range_double_test_visual_bars = int(
            self.params.get("range_double_test_visual_bars", max(self.range_detection_bars, 120))
        )
        self.show_range_double_test_overlay = bool(self.params.get("show_range_double_test_overlay", True))
        self.chop_overlap_threshold = float(self.params.get("chop_overlap_threshold", 0.60))
        self.chop_ema_flat_ticks = float(self.params.get("chop_ema_flat_ticks", 2.0))
        self.vol_spike_atr_mult = float(self.params.get("vol_spike_atr_mult", 3.0))
        self.vol_spike_release_mult = float(self.params.get("vol_spike_release_mult", 2.0))
        self.vol_spike_release_bars = int(self.params.get("vol_spike_release_bars", 3))
        self.session_cutoff_enabled = bool(self.params.get("session_cutoff_enabled", True))
        self.allowed_entry_hours_et = self._parse_hour_set(self.params.get("allowed_entry_hours_et"))
        self.excluded_entry_hours_et = self._parse_hour_set(self.params.get("excluded_entry_hours_et"))
        self.no_new_entries_after_et = str(
            self.params.get(
                "no_new_entries_after_et",
                ((self.spec.get("session_locks") or {}).get("F-08_end_of_session_cutoff") or {}).get(
                    "rule",
                    "no new entries after 15:30 ET",
                ),
            )
        )
        self.daily_loss_guardrail = dict(guardrails.get("G-02_daily_loss_cap") or {})
        self.reset()

    def _apply_symbol_overrides(self, params: dict[str, Any]) -> dict[str, Any]:
        merged = dict(params or {})
        overrides = merged.get("symbol_params", {}) or {}
        if not isinstance(overrides, dict):
            return merged
        symbol_upper = self.symbol.upper()
        keys = [symbol_upper, self._base_symbol(symbol_upper)]
        for key in keys:
            override = overrides.get(key)
            if isinstance(override, dict):
                merged.update(override)
        return merged

    def _base_symbol(self, symbol_upper: str) -> str:
        known = ("MNQ", "MES", "MCL", "MGC", "MYM", "NQ", "ES", "CL", "GC", "YM")
        for prefix in known:
            if symbol_upper.startswith(prefix):
                return prefix
        return symbol_upper

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.repo_root / path

    def _parse_disabled_setup_sides(self, raw: Any) -> set[tuple[str, str, str]]:
        disabled: set[tuple[str, str, str]] = set()
        if raw is None:
            return disabled
        if isinstance(raw, dict):
            for symbol, setup_map in raw.items():
                if not isinstance(setup_map, dict):
                    continue
                for setup_id, sides in setup_map.items():
                    side_items = sides if isinstance(sides, list) else [sides]
                    for side in side_items:
                        disabled.add((str(symbol).upper(), str(setup_id), str(side).lower()))
            return disabled
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            parts = str(item).split(":")
            if len(parts) != 3:
                continue
            symbol, setup_id, side = parts
            disabled.add((symbol.strip().upper(), setup_id.strip(), side.strip().lower()))
        return disabled

    def _parse_hour_set(self, raw: Any) -> set[int]:
        if raw is None or raw == "":
            return set()
        items: Iterable[Any]
        if isinstance(raw, str):
            items = [item.strip() for item in raw.split(",")]
        elif isinstance(raw, Iterable):
            items = raw
        else:
            items = [raw]
        hours: set[int] = set()
        for item in items:
            try:
                hour = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= hour <= 23:
                hours.add(hour)
        return hours

    def _load_spec(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {
                "metadata": {"instrument": "ES", "ema_period": 21, "lock_in_version": "fallback"},
                "universal_locks": {
                    "scalp_target_ticks": 10,
                    "ema_distance_filter_ticks": 10,
                    "large_signal_bar_threshold_ticks": 12,
                },
                "setups": {},
            }
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def reset(self) -> None:
        self.swing_highs: list[dict[str, Any]] = []
        self.swing_lows: list[dict[str, Any]] = []
        self._last_confirmed_index = -1
        self._pending_candidates: list[dict[str, Any]] = []
        self._last_signal_key = None
        self._range_state: dict[str, Any] | None = None
        self._volatility_cooldown_active = False
        self._volatility_release_count = 0
        self._range_double_test_armed_keys: set[tuple[Any, ...]] = set()
        self._chart_events: list[dict[str, Any]] = []
        self._chart_event_keys: set[tuple[Any, ...]] = set()

    def get_description(self) -> str:
        version = self.metadata.get("lock_in_version", "price-action playbook")
        return (
            "Price Action BS: Thomas Wade-style EMA21 second entries, failed second entries, "
            f"range rejections, and failed breakouts from pa_strat sources ({version})."
        )

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        bias = self._main_bias(index, bars, indicators) if index < len(indicators) else "neutral"
        range_state = dict(self._range_state) if self._range_state else None
        
        # [EXPERT FIX] Expose current session regime to the dashboard
        current_regime = "OUT"
        if index < len(indicators):
             current_regime = self._get_session_regime(indicators.iloc[index])
             
        return {
            "status": "Active" if index + 1 >= self.warmup_bars else f"Warming up ({index + 1}/{self.warmup_bars})",
            "session_regime": current_regime,
            "strategy": self.name,
            "source_files": list(self.source_files),
            "setups_path": str(self.setups_path),
            "ema_period": self.ema_period,
            "main_bias": bias,
            "swing_highs": len(self.swing_highs),
            "swing_lows": len(self.swing_lows),
            "pending_candidates": len(self._pending_candidates),
            "range_state": range_state,
            "volatility_cooldown_active": self._volatility_cooldown_active,
            "enabled_setups": sorted(self.enabled_setups),
            "disabled_setup_sides": sorted(":".join(item) for item in self.disabled_setup_sides),
            "max_stop_ticks": self.max_stop_ticks,
            "scalp_target_ticks": self.scalp_target_ticks,
            "target_mode": self.target_mode,
            "risk_reward": self.risk_reward,
            "min_target_ticks": self.min_target_ticks,
            "max_target_ticks": self.max_target_ticks,
            "use_candidate_price_targets": self.use_candidate_price_targets,
            "min_signal_bar_atr_ratio": self.min_signal_bar_atr_ratio,
            "min_body_ratio": self.min_body_ratio,
            "min_close_ema_distance_ticks": self.min_close_ema_distance_ticks,
            "allowed_entry_hours_et": sorted(self.allowed_entry_hours_et),
            "excluded_entry_hours_et": sorted(self.excluded_entry_hours_et),
            "chart_event_count": len(self._chart_events),
        }

    def get_chart_overlays(self, index, bars, indicators) -> dict[str, Any]:
        if bars is None or indicators is None or len(bars) == 0 or len(indicators) == 0:
            return {"lines": [], "levels": [], "markers": [], "events": []}
        index = min(int(index), len(bars) - 1, len(indicators) - 1)
        start_index = max(0, index - 500)
        start_time = self._bar_time(bars, start_index)
        end_time = self._bar_time(bars, index)

        ema_points = []
        if self.ema_column in indicators.columns:
            for i in range(start_index, index + 1):
                value = indicators.iloc[i].get(self.ema_column)
                if value is not None and not pd.isna(value):
                    ema_points.append({"time": self._bar_time(bars, i), "value": float(value)})

        lines = []
        if ema_points:
            lines.append({"id": "ema21", "label": "EMA21", "color": "#f59e0b", "style": "solid", "points": ema_points})

        high_trendline = self._trendline_from_pivots("swing_high_trendline", self.swing_highs, bars, "#ef4444", index)
        low_trendline = self._trendline_from_pivots("swing_low_trendline", self.swing_lows, bars, "#10b981", index)
        lines.extend([line for line in (high_trendline, low_trendline) if line])
        lines.extend(self._visual_channel_lines(bars, index, start_index))
        if self.show_range_double_test_overlay:
            lines.extend(self._range_double_test_overlay_lines(index, bars))

        levels = []
        if self._range_state:
            range_start = self._bar_time(bars, int(self._range_state.get("detected_index", start_index)))
            range_mid = (float(self._range_state["high"]) + float(self._range_state["low"])) / 2.0
            levels.extend(
                [
                    {
                        "id": "range_high",
                        "label": "Range High",
                        "price": float(self._range_state["high"]),
                        "start_time": range_start,
                        "end_time": end_time,
                        "color": "#ef4444",
                        "style": "dashed",
                        "width": 2,
                    },
                    {
                        "id": "range_low",
                        "label": "Range Low",
                        "price": float(self._range_state["low"]),
                        "start_time": range_start,
                        "end_time": end_time,
                        "color": "#10b981",
                        "style": "dashed",
                        "width": 2,
                    },
                    {
                        "id": "range_mid",
                        "label": "Range Mid",
                        "price": range_mid,
                        "start_time": range_start,
                        "end_time": end_time,
                        "color": "#94a3b8",
                        "style": "dotted",
                        "width": 1,
                    },
                ]
            )
        else:
            levels.extend(self._visual_range_levels(index, bars, start_index, end_time))

        for idx, candidate in enumerate(self._pending_candidates[-12:]):
            signal_index = int(candidate.get("signal_index", index))
            candidate_start = self._bar_time(bars, signal_index)
            side = str(candidate.get("side", ""))
            color = "#10b981" if side == "Buy" else "#ef4444"
            levels.extend(
                [
                    {
                        "id": f"pending_entry_{idx}",
                        "label": f"{candidate.get('setup_id', '')} {side} Trigger",
                        "price": float(candidate["entry_trigger"]),
                        "start_time": candidate_start,
                        "end_time": end_time,
                        "color": color,
                        "style": "solid",
                    },
                    {
                        "id": f"pending_failure_{idx}",
                        "label": f"{candidate.get('setup_id', '')} Failure",
                        "price": float(candidate["failure_trigger"]),
                        "start_time": candidate_start,
                        "end_time": end_time,
                        "color": "#f59e0b",
                        "style": "dotted",
                    },
                ]
            )

        markers = self._pivot_markers(self.swing_highs[-25:], bars, "swing_high") + self._pivot_markers(self.swing_lows[-25:], bars, "swing_low")
        return {
            "lines": lines,
            "levels": levels,
            "markers": markers,
            "events": list(self._chart_events[-80:]),
            "state": {
                "main_bias": self._main_bias(index, bars, indicators),
                "pending_candidates": len(self._pending_candidates),
                "volatility_cooldown_active": bool(self._volatility_cooldown_active),
                "range_state": dict(self._range_state or {}),
                "start_time": start_time,
                "end_time": end_time,
            },
        }

    def on_bar(self, index, bars, indicators):
        if self.enabled_symbols and self.symbol.upper() not in self.enabled_symbols and self._base_symbol(self.symbol.upper()) not in self.enabled_symbols:
            return None

        self._confirm_swings(index, bars)
        self._update_volatility_guardrail(index, indicators)
        self._update_range_state(index, bars, indicators)

        if index < self.warmup_bars:
            return None

        # [EXPERT FIX] Regime-Aware Session Logic
        if self._volatility_cooldown_active:
            return None
        regime = self._get_session_regime(indicators.iloc[index])
        if regime == "OUT":
            return None

        # Apply regime-aware filter overrides
        orig_min_atr_ratio = self.min_signal_bar_atr_ratio
        if regime == "ASIAN":
            # Tighten filters for lower-liquidity Asian session
            self.min_signal_bar_atr_ratio = float(self.params.get("asian_min_signal_bar_atr_ratio", 0.40))
        
        try:
            pending_signal = self._evaluate_pending_candidates(index, bars, indicators)
            if pending_signal is not None:
                return pending_signal

            range_signal = self._evaluate_range_setups(index, bars, indicators)
            if range_signal is not None:
                return range_signal

            self._track_continuation_candidates(index, bars, indicators)
            return None
        finally:
            # Restore original ratio so we don't leak Asian settings into NY turn
            self.min_signal_bar_atr_ratio = orig_min_atr_ratio

    def _append_chart_event(
        self,
        *,
        key: tuple[Any, ...],
        index: int,
        bars: pd.DataFrame,
        kind: str,
        label: str,
        price: float | None = None,
        side: str | None = None,
        setup_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        if key in self._chart_event_keys:
            return
        self._chart_event_keys.add(key)
        if len(self._chart_event_keys) > 400:
            self._chart_event_keys = set(list(self._chart_event_keys)[-250:])
        event = {
            "time": self._bar_time(bars, index),
            "kind": kind,
            "label": label,
            "price": float(price) if price is not None else None,
            "side": side,
            "setup_id": setup_id,
            "detail": detail,
        }
        self._chart_events.append({k: v for k, v in event.items() if v is not None})
        self._chart_events = self._chart_events[-160:]

    def _bar_time(self, bars: pd.DataFrame, index: int) -> int:
        if bars is None or len(bars) == 0:
            return 0
        idx = max(0, min(int(index), len(bars) - 1))
        return chart_time(bars.iloc[idx].get("timestamp", idx))

    def _pivot_markers(self, pivots: list[dict[str, Any]], bars: pd.DataFrame, kind: str) -> list[dict[str, Any]]:
        markers = []
        for pivot in pivots:
            idx = int(pivot.get("index", 0))
            markers.append(
                {
                    "time": self._bar_time(bars, idx),
                    "price": float(pivot.get("price", 0.0)),
                    "kind": kind,
                    "label": "SH" if kind == "swing_high" else "SL",
                    "color": "#ef4444" if kind == "swing_high" else "#10b981",
                }
            )
        return markers

    def _trendline_from_pivots(
        self,
        line_id: str,
        pivots: list[dict[str, Any]],
        bars: pd.DataFrame,
        color: str,
        end_index: int,
    ) -> dict[str, Any] | None:
        usable = [pivot for pivot in pivots if int(pivot.get("index", -1)) < len(bars)]
        if len(usable) < 2:
            return None
        first, second = usable[-2], usable[-1]
        first_idx = int(first["index"])
        second_idx = int(second["index"])
        if second_idx <= first_idx:
            return None
        first_price = float(first["price"])
        second_price = float(second["price"])
        slope = (second_price - first_price) / float(second_idx - first_idx)
        projected = second_price + slope * float(max(int(end_index), second_idx) - second_idx)
        points = [
            {"time": self._bar_time(bars, first_idx), "value": first_price},
            {"time": self._bar_time(bars, second_idx), "value": second_price},
        ]
        if end_index > second_idx:
            points.append({"time": self._bar_time(bars, end_index), "value": projected})
        return {
            "id": line_id,
            "label": "Swing High Trendline" if "high" in line_id else "Swing Low Trendline",
            "color": color,
            "style": "dashed",
            "width": 2,
            "points": points,
        }

    def _visual_channel_lines(self, bars: pd.DataFrame, end_index: int, start_index: int) -> list[dict[str, Any]]:
        base_pivots = self.swing_lows if len(self.swing_lows) >= 2 else self.swing_highs
        usable = [pivot for pivot in base_pivots if int(pivot.get("index", -1)) < len(bars)]
        if len(usable) < 2:
            return []
        first, second = usable[-2], usable[-1]
        first_idx = int(first["index"])
        second_idx = int(second["index"])
        if second_idx <= first_idx:
            return []
        first_price = float(first["price"])
        second_price = float(second["price"])
        slope = (second_price - first_price) / float(second_idx - first_idx)
        window_start = max(start_index, end_index - 180, 0)
        offsets_high = []
        offsets_low = []
        for i in range(window_start, end_index + 1):
            projected = second_price + slope * float(i - second_idx)
            row = bars.iloc[i]
            offsets_high.append(float(row["high"]) - projected)
            offsets_low.append(float(row["low"]) - projected)
        if not offsets_high or not offsets_low:
            return []
        upper_offset = max(offsets_high)
        lower_offset = min(offsets_low)
        if not math.isfinite(upper_offset) or not math.isfinite(lower_offset) or upper_offset <= lower_offset:
            return []

        def point(idx: int, offset: float) -> dict[str, Any]:
            return {
                "time": self._bar_time(bars, idx),
                "value": second_price + slope * float(idx - second_idx) + offset,
            }

        mid_offset = (upper_offset + lower_offset) / 2.0
        return [
            {
                "id": "visual_channel_upper",
                "label": "Visual Channel Upper",
                "color": "#38bdf8",
                "style": "dotted",
                "width": 2,
                "points": [point(window_start, upper_offset), point(end_index, upper_offset)],
            },
            {
                "id": "visual_channel_mid",
                "label": "Visual Channel Mid",
                "color": "#64748b",
                "style": "dotted",
                "width": 1,
                "points": [point(window_start, mid_offset), point(end_index, mid_offset)],
            },
            {
                "id": "visual_channel_lower",
                "label": "Visual Channel Lower",
                "color": "#38bdf8",
                "style": "dotted",
                "width": 2,
                "points": [point(window_start, lower_offset), point(end_index, lower_offset)],
            },
        ]

    def _visual_range_levels(self, index: int, bars: pd.DataFrame, start_index: int, end_time: int) -> list[dict[str, Any]]:
        window_start = max(start_index, index - max(self.range_detection_bars, 120), 0)
        recent = bars.iloc[window_start : index + 1]
        if recent.empty:
            return []
        high = float(recent["high"].max())
        low = float(recent["low"].min())
        if not math.isfinite(high) or not math.isfinite(low) or high <= low:
            return []
        mid = (high + low) / 2.0
        range_start = self._bar_time(bars, window_start)
        return [
            {
                "id": "visual_range_high",
                "label": "Visual Range High",
                "price": high,
                "start_time": range_start,
                "end_time": end_time,
                "color": "#a855f7",
                "style": "dotted",
                "width": 2,
            },
            {
                "id": "visual_range_mid",
                "label": "Visual Range Mid",
                "price": mid,
                "start_time": range_start,
                "end_time": end_time,
                "color": "#94a3b8",
                "style": "dotted",
                "width": 1,
            },
            {
                "id": "visual_range_low",
                "label": "Visual Range Low",
                "price": low,
                "start_time": range_start,
                "end_time": end_time,
                "color": "#a855f7",
                "style": "dotted",
                "width": 2,
            },
        ]

    def _confirm_swings(self, index: int, bars: pd.DataFrame) -> None:
        lookback = self.fractal_length
        if index < lookback * 2:
            return
        candidate_index = index - lookback
        if candidate_index <= self._last_confirmed_index:
            return

        start = candidate_index - lookback
        end = candidate_index + lookback + 1
        window = bars.iloc[start:end]
        candidate = bars.iloc[candidate_index]
        low = float(candidate["low"])
        high = float(candidate["high"])

        if low < float(window.drop(window.index[lookback])["low"].min()):
            self._append_pivot(self.swing_lows, {"index": candidate_index, "price": low, "kind": "low"})
        if high > float(window.drop(window.index[lookback])["high"].max()):
            self._append_pivot(self.swing_highs, {"index": candidate_index, "price": high, "kind": "high"})
        self._last_confirmed_index = candidate_index

    def _append_pivot(self, collection: list[dict[str, Any]], pivot: dict[str, Any]) -> None:
        if collection and pivot["index"] <= collection[-1]["index"]:
            return
        collection.append(pivot)
        if len(collection) > 80:
            del collection[0]

    def _update_volatility_guardrail(self, index: int, indicators: pd.DataFrame) -> None:
        row = indicators.iloc[index]
        atr = float(row.get("atr", 0.0) or 0.0)
        if math.isnan(atr) or atr <= 0 or index < 30:
            return
            
        # 1. Bar Spike Check: Single bar range relative to current ATR
        bar_range = float(row["high"]) - float(row["low"])
        ratio = bar_range / atr
        
        # 2. Sustained Volatility Check: Current ATR relative to 30-bar mean ATR
        recent_atr_mean = indicators["atr"].iloc[max(0, index-30):index].mean()
        atr_expansion = atr / recent_atr_mean if recent_atr_mean > 0 else 1.0
        
        spike_detected = (ratio > self.vol_spike_atr_mult) or (atr_expansion > 2.5)
        
        if spike_detected:
            self._append_chart_event(
                key=("vol_spike", index),
                index=index,
                bars=indicators,
                kind="volatility",
                label="Volatility cooldown",
                price=float(row["close"]),
                detail=f"bar_ratio={ratio:.1f}, atr_exp={atr_expansion:.1f}",
            )
            self._volatility_cooldown_active = True
            self._volatility_release_count = 0
            return
            
        if not self._volatility_cooldown_active:
            return
            
        # Release if bar range returns to normal relative to ATR
        if ratio < self.vol_spike_release_mult:
            self._volatility_release_count += 1
            if self._volatility_release_count >= self.vol_spike_release_bars:
                self._append_chart_event(
                    key=("vol_release", index),
                    index=index,
                    bars=indicators,
                    kind="volatility",
                    label="Volatility released",
                    price=float(row["close"]),
                    detail=f"normalized_at_{ratio:.2f}",
                )
                self._volatility_cooldown_active = False
                self._volatility_release_count = 0
        else:
            self._volatility_release_count = 0

    def _update_range_state(self, index: int, bars: pd.DataFrame, indicators: pd.DataFrame) -> None:
        row = indicators.iloc[index]
        close = float(row["close"])
        if self._range_state:
            high = float(self._range_state["high"])
            low = float(self._range_state["low"])
            if close >= high + self.tick_size or close <= low - self.tick_size:
                self._append_chart_event(
                    key=("range_clear", index),
                    index=index,
                    bars=bars,
                    kind="range",
                    label="Range cleared",
                    price=close,
                    detail=f"{low:.2f}-{high:.2f}",
                )
                self._range_state = None
            return

        if not self._ema_is_flat(index, indicators):
            return

        recent_highs = self._recent_pivots(self.swing_highs, index, self.range_detection_bars)
        recent_lows = self._recent_pivots(self.swing_lows, index, self.range_detection_bars)
        high_cluster = self._clustered_prices(recent_highs, self.range_cluster_tolerance_ticks, minimum=2)
        low_cluster = self._clustered_prices(recent_lows, self.range_cluster_tolerance_ticks, minimum=2)
        if not high_cluster or not low_cluster:
            return
        range_high = float(median(high_cluster))
        range_low = float(median(low_cluster))
        width_ticks = int(round((range_high - range_low) / self.tick_size))
        if self.range_min_width_ticks <= width_ticks <= self.range_max_width_ticks:
            self._range_state = {"high": range_high, "low": range_low, "width_ticks": width_ticks, "detected_index": index}
            self._append_chart_event(
                key=("range_detect", index, round(range_high, 8), round(range_low, 8)),
                index=index,
                bars=bars,
                kind="range",
                label="Range detected",
                price=close,
                detail=f"{range_low:.2f}-{range_high:.2f} ({width_ticks} ticks)",
            )

    def _clustered_prices(
        self,
        pivots: Iterable[dict[str, Any]],
        tolerance_ticks: int,
        minimum: int,
    ) -> list[float]:
        prices = [float(p["price"]) for p in pivots]
        tolerance = tolerance_ticks * self.tick_size
        best: list[float] = []
        for price in prices:
            cluster = [candidate for candidate in prices if abs(candidate - price) <= tolerance]
            if len(cluster) > len(best):
                best = cluster
        return best if len(best) >= minimum else []

    def _evaluate_pending_candidates(self, index: int, bars: pd.DataFrame, indicators: pd.DataFrame):
        if not self._pending_candidates:
            return None

        row = indicators.iloc[index]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        open_ = float(row["open"])
        ema = self._ema_value(row, close)
        remaining = []

        for cand_idx, candidate in enumerate(self._pending_candidates):
            side = candidate["side"]
            age = index - int(candidate["signal_index"])
            if age <= 0:
                remaining.append(candidate)
                continue
            if age > self.failure_window_bars:
                self._append_chart_event(
                    key=("candidate_expired", candidate.get("key"), index),
                    index=index,
                    bars=bars,
                    kind="failed",
                    label="Setup expired",
                    price=close,
                    side=side,
                    setup_id=candidate.get("setup_id"),
                    detail=f"age={age}",
                )
                continue

            # Use a tiny epsilon (0.1 ticks) to ensure we don't miss entries due to float noise
            epsilon = 0.1 * self.tick_size
            continuation_hit = (high >= candidate["entry_trigger"] - epsilon) if side == "Buy" else (low <= candidate["entry_trigger"] + epsilon)
            failure_hit = (low <= candidate["failure_trigger"] + epsilon) if side == "Buy" else (high >= candidate["failure_trigger"] - epsilon)

            if continuation_hit and failure_hit:
                self._append_chart_event(
                    key=("candidate_both_hit", candidate.get("key"), index),
                    index=index,
                    bars=bars,
                    kind="failed",
                    label="Trigger and failure hit same bar",
                    price=close,
                    side=side,
                    setup_id=candidate.get("setup_id"),
                )
                continue
            if continuation_hit:
                self._append_chart_event(
                    key=("candidate_trigger", candidate.get("key"), index),
                    index=index,
                    bars=bars,
                    kind="signal",
                    label="Continuation triggered",
                    price=float(candidate["entry_trigger"]),
                    side=side,
                    setup_id=candidate.get("setup_id"),
                )
                self._pending_candidates = remaining + self._pending_candidates[cand_idx + 1 :]
                return self._build_signal(index, bars, indicators, candidate["setup_id"], side, candidate)
            if failure_hit:
                failed_side = "Sell" if side == "Buy" else "Buy"
                failure_setup = "E-02" if side == "Buy" else "E-04"
                if failure_setup not in self.enabled_setups:
                    continue
                valid_failure_bar = close < open_ and close < ema if failed_side == "Sell" else close > open_ and close > ema
                if valid_failure_bar and self._signal_bar_is_clean(index, bars, failed_side, indicators):
                    failure_candidate = dict(candidate)
                    failure_candidate.update(
                        {
                            "setup_id": failure_setup,
                            "side": failed_side,
                            "entry_trigger": low - self.tick_size if failed_side == "Sell" else high + self.tick_size,
                            "failure_of": candidate["setup_id"],
                            "signal_high": high,
                            "signal_low": low,
                            "signal_index": index,
                        }
                    )
                    self._pending_candidates = remaining + self._pending_candidates[cand_idx + 1 :]
                    self._append_chart_event(
                        key=("failure_trigger", candidate.get("key"), index),
                        index=index,
                        bars=bars,
                        kind="signal",
                        label="Failed second entry triggered",
                        price=float(failure_candidate["entry_trigger"]),
                        side=failed_side,
                        setup_id=failure_setup,
                    )
                    return self._build_signal(index, bars, indicators, failure_setup, failed_side, failure_candidate)
                self._append_chart_event(
                    key=("invalid_failure", candidate.get("key"), index),
                    index=index,
                    bars=bars,
                    kind="failed",
                    label="Failure bar invalid",
                    price=close,
                    side=failed_side,
                    setup_id=failure_setup,
                )
                continue

            remaining.append(candidate)

        self._pending_candidates = remaining
        return None

    def _evaluate_range_setups(self, index: int, bars: pd.DataFrame, indicators: pd.DataFrame):
        if index < 1:
            return None

        row = indicators.iloc[index]
        close = float(row["close"])
        open_ = float(row["open"])

        double_test_signal = self._evaluate_range_double_test_setup(index, bars, indicators)
        if double_test_signal is not None:
            return double_test_signal

        if not self._range_state:
            return None

        prev = indicators.iloc[index - 1]
        high = float(row["high"])
        low = float(row["low"])
        range_high = float(self._range_state["high"])
        range_low = float(self._range_state["low"])
        tolerance = self.range_cluster_tolerance_ticks * self.tick_size

        prev_close = float(prev["close"])
        failed_breakout_short = prev_close > range_high + self.tick_size and close < range_high and close < open_
        failed_breakout_long = prev_close < range_low - self.tick_size and close > range_low and close > open_
        if failed_breakout_short:
            return self._direct_setup_signal(index, bars, indicators, "E-06", "Sell", "failed breakout above range")
        if failed_breakout_long:
            return self._direct_setup_signal(index, bars, indicators, "E-06", "Buy", "failed breakout below range")

        tests_high = self._level_tests(self.swing_highs, range_high, tolerance, index, self.range_detection_bars)
        tests_low = self._level_tests(self.swing_lows, range_low, tolerance, index, self.range_detection_bars)
        if tests_high >= 3 and high >= range_high - tolerance and close < open_:
            return self._direct_setup_signal(index, bars, indicators, "E-07", "Sell", "triple test at range high")
        if tests_low >= 3 and low <= range_low + tolerance and close > open_:
            return self._direct_setup_signal(index, bars, indicators, "E-07", "Buy", "triple test at range low")
        return None

    def _track_continuation_candidates(self, index: int, bars: pd.DataFrame, indicators: pd.DataFrame) -> None:
        if index < 1 or self._volatility_cooldown_active or self._get_session_regime(indicators.iloc[index]) == "OUT":
            return
        if self._is_choppy(index, bars, indicators):
            return

        row = indicators.iloc[index]
        close = float(row["close"])
        open_ = float(row["open"])
        bias = self._main_bias(index, bars, indicators)

        if "E-01" in self.enabled_setups and bias == "bullish" and close > open_ and self._near_ema(row) and self._has_two_pullback_legs(index, "down"):
            if self._signal_bar_is_clean(index, bars, "Buy", indicators):
                self._add_pending_candidate(index, row, "E-01", "Buy")

        if "E-03" in self.enabled_setups and bias == "bearish" and close < open_ and self._near_ema(row) and self._has_two_pullback_legs(index, "up"):
            if self._signal_bar_is_clean(index, bars, "Sell", indicators):
                self._add_pending_candidate(index, row, "E-03", "Sell")

        if "E-05" in self.enabled_setups and bias == "bullish" and close > open_ and self._near_ema(row) and self._has_single_pullback_confirmation(index, "down"):
            if self._signal_bar_is_clean(index, bars, "Buy", indicators):
                self._add_pending_candidate(index, row, "E-05", "Buy")

        if "E-05" in self.enabled_setups and bias == "bearish" and close < open_ and self._near_ema(row) and self._has_single_pullback_confirmation(index, "up"):
            if self._signal_bar_is_clean(index, bars, "Sell", indicators):
                self._add_pending_candidate(index, row, "E-05", "Sell")

    def _evaluate_range_double_test_setup(self, index: int, bars: pd.DataFrame, indicators: pd.DataFrame):
        if "E-11" not in self.enabled_setups:
            return

        row = indicators.iloc[index]
        close = float(row["close"])
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        breakout = self.range_double_test_breakout_ticks * self.tick_size

        for side in ("Buy", "Sell"):
            context = self._range_double_test_context(index, bars, side)
            if not context:
                continue

            pattern_key = (
                side,
                int(context["first_test_index"]),
                int(context["second_test_index"]),
                int(context["line_second_index"]),
            )
            if pattern_key in self._range_double_test_armed_keys:
                continue
            if index - int(context["second_test_index"]) > self.signal_after_pivot_bars:
                continue

            line_value = float(context["line_value"])
            prev_line_value = float(context["previous_line_value"])
            target_price = float(context["range_mid"])

            if side == "Buy":
                attempted_break = high >= line_value
                confirmed_break = close >= line_value + breakout
                direction_ok = close > open_
            else:
                attempted_break = low <= line_value
                confirmed_break = close <= line_value - breakout
                direction_ok = close < open_

            if not confirmed_break:
                if attempted_break:
                    self._append_chart_event(
                        key=("e11_reject_no_close", side, index),
                        index=index,
                        bars=bars,
                        kind="failed",
                        label="E-11 rejected: no close beyond trendline",
                        price=close,
                        side=side,
                        setup_id="E-11",
                        detail=f"line={line_value:.2f}",
                    )
                continue

            if not direction_ok:
                self._append_chart_event(
                    key=("e11_reject_color", side, index),
                    index=index,
                    bars=bars,
                    kind="failed",
                    label="E-11 rejected: signal bar color",
                    price=close,
                    side=side,
                    setup_id="E-11",
                )
                continue

            if not self._signal_bar_is_clean(index, bars, side, indicators):
                self._append_chart_event(
                    key=("e11_reject_unclean", side, index),
                    index=index,
                    bars=bars,
                    kind="failed",
                    label="E-11 rejected: unclean signal bar",
                    price=close,
                    side=side,
                    setup_id="E-11",
                )
                continue

            reason = "Range Double Bottom Trendline Break" if side == "Buy" else "Range Double Top Trendline Break"
            added = self._add_pending_candidate(
                index,
                row,
                "E-11",
                side,
                extra_metadata={
                    "range_state": dict(context["range_context"]),
                    "range_context": dict(context["range_context"]),
                    "range_mid_target_price": target_price,
                    "target_price": target_price,
                    "direct_reason": reason,
                    "e11_context": {
                        "edge": context["edge"],
                        "first_test_index": context["first_test_index"],
                        "second_test_index": context["second_test_index"],
                        "trigger_line_value": line_value,
                        "previous_trigger_line_value": prev_line_value,
                    },
                },
            )
            if added:
                self._range_double_test_armed_keys.add(pattern_key)
                self._append_chart_event(
                    key=("e11_confirmed", side, index),
                    index=index,
                    bars=bars,
                    kind="pending",
                    label=reason,
                    price=close,
                    side=side,
                    setup_id="E-11",
                    detail=f"target_mid={target_price:.2f}",
                )
        return

    def _range_double_test_context(self, index: int, bars: pd.DataFrame, side: str) -> dict[str, Any] | None:
        if index < 1:
            return None

        range_context = self._range_double_test_range_context(index, bars)
        if not range_context:
            return None
        tolerance = self.range_double_test_tolerance_ticks * self.tick_size
        range_high = float(range_context["high"])
        range_low = float(range_context["low"])
        range_mid = (range_high + range_low) / 2.0
        if side == "Buy":
            edge = range_low
            recent_test_pivots = self._recent_pivots(self.swing_lows, index, self.range_double_test_lookback_bars)
            line_pivots = self.swing_highs
        else:
            edge = range_high
            recent_test_pivots = self._recent_pivots(self.swing_highs, index, self.range_double_test_lookback_bars)
            line_pivots = self.swing_lows

        test_pivots = [pivot for pivot in recent_test_pivots if abs(float(pivot["price"]) - edge) <= tolerance]
        if len(test_pivots) < 2:
            return None
        first_test, second_test = test_pivots[-2], test_pivots[-1]
        first_idx = int(first_test["index"])
        second_idx = int(second_test["index"])
        if second_idx <= first_idx:
            return None
        if second_idx - first_idx < self.range_double_test_min_separation_bars:
            return None
        if abs(float(first_test["price"]) - float(second_test["price"])) > tolerance:
            return None

        trigger_pivots = [pivot for pivot in line_pivots if first_idx < int(pivot.get("index", -1)) < index]
        if len(trigger_pivots) < 2:
            return None
        line_first, line_second = trigger_pivots[-2], trigger_pivots[-1]
        line = self._line_from_two_pivots(line_first, line_second)
        if line is None:
            return None
        slope = float(line["slope"])
        if side == "Buy" and slope > 0:
            return None
        if side == "Sell" and slope < 0:
            return None

        return {
            "side": side,
            "edge": edge,
            "range_mid": range_mid,
            "first_test_index": first_idx,
            "second_test_index": second_idx,
            "first_test_price": float(first_test["price"]),
            "second_test_price": float(second_test["price"]),
            "line_first": line_first,
            "line_second": line_second,
            "line_first_index": int(line_first["index"]),
            "line_second_index": int(line_second["index"]),
            "range_context": range_context,
            "line_value": self._line_value(line, index),
            "previous_line_value": self._line_value(line, index - 1),
            "line": line,
        }

    def _range_double_test_range_context(self, index: int, bars: pd.DataFrame) -> dict[str, Any] | None:
        if self._range_state:
            return {
                "source": "validated",
                "high": float(self._range_state["high"]),
                "low": float(self._range_state["low"]),
                "width_ticks": int(self._range_state.get("width_ticks", 0) or 0),
                "detected_index": int(self._range_state.get("detected_index", index)),
            }
        if not self.range_double_test_use_visual_range:
            return None
        if bars is None or len(bars) == 0:
            return None
        window_start = max(0, int(index) - max(1, self.range_double_test_visual_bars) + 1)
        recent = bars.iloc[window_start : int(index) + 1]
        if recent.empty:
            return None
        range_high = float(recent["high"].max())
        range_low = float(recent["low"].min())
        if not math.isfinite(range_high) or not math.isfinite(range_low) or range_high <= range_low:
            return None
        width_ticks = int(round((range_high - range_low) / self.tick_size))
        if width_ticks < self.range_min_width_ticks:
            return None
        return {
            "source": "visual_window",
            "high": range_high,
            "low": range_low,
            "width_ticks": width_ticks,
            "detected_index": window_start,
        }

    def _line_from_two_pivots(self, first: dict[str, Any], second: dict[str, Any]) -> dict[str, float] | None:
        first_idx = int(first.get("index", -1))
        second_idx = int(second.get("index", -1))
        if second_idx <= first_idx:
            return None
        first_price = float(first["price"])
        second_price = float(second["price"])
        slope = (second_price - first_price) / float(second_idx - first_idx)
        return {"first_index": first_idx, "first_price": first_price, "second_index": second_idx, "second_price": second_price, "slope": slope}

    def _line_value(self, line: dict[str, float], index: int) -> float:
        return float(line["second_price"]) + float(line["slope"]) * float(int(index) - int(line["second_index"]))

    def _range_double_test_overlay_lines(self, index: int, bars: pd.DataFrame) -> list[dict[str, Any]]:
        lines = []
        for side, color, label in (
            ("Buy", "#22c55e", "E-11 Buy Break Line"),
            ("Sell", "#ef4444", "E-11 Sell Break Line"),
        ):
            context = self._range_double_test_context(index, bars, side)
            if not context:
                continue
            line = context["line"]
            first_idx = int(line["first_index"])
            second_idx = int(line["second_index"])
            projected = self._line_value(line, index)
            lines.append(
                {
                    "id": f"e11_{side.lower()}_break_line",
                    "label": label,
                    "color": color,
                    "style": "dashed",
                    "width": 2,
                    "points": [
                        {"time": self._bar_time(bars, first_idx), "value": float(line["first_price"])},
                        {"time": self._bar_time(bars, second_idx), "value": float(line["second_price"])},
                        {"time": self._bar_time(bars, index), "value": projected},
                    ],
                }
            )
        return lines

    def _add_pending_candidate(
        self,
        index: int,
        row,
        setup_id: str,
        side: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> bool:
        if setup_id not in self.enabled_setups:
            return False
        if not self._setup_side_enabled(setup_id, side):
            return False
        if self.skip_large_signal_bars and self._range_ticks(row) > self.large_signal_bar_threshold_ticks:
            self._append_chart_event(
                key=("large_signal_skip", setup_id, side, index),
                index=index,
                bars=pd.DataFrame([row]),
                kind="failed",
                label="Signal bar too large",
                price=float(row["close"]),
                side=side,
                setup_id=setup_id,
            )
            return False
        key = (setup_id, side, index)
        if key == self._last_signal_key:
            return False
        if any(c["key"] == key for c in self._pending_candidates):
            return False
        high = float(row["high"])
        low = float(row["low"])
        candidate = {
            "key": key,
            "setup_id": setup_id,
            "side": side,
            "signal_index": index,
            "signal_high": high,
            "signal_low": low,
            "entry_trigger": high + self.tick_size if side == "Buy" else low - self.tick_size,
            "failure_trigger": low - self.tick_size if side == "Buy" else high + self.tick_size,
            "entry_rule": self._entry_rule_for_signal_bar(row),
            "ema21": self._ema_value(row, float(row["close"])),
        }
        if extra_metadata:
            candidate.update(extra_metadata)
        self._pending_candidates.append(candidate)
        self._append_chart_event(
            key=("pending_added", setup_id, side, index),
            index=index,
            bars=pd.DataFrame([row]),
            kind="pending",
            label="Setup armed",
            price=float(row["close"]),
            side=side,
            setup_id=setup_id,
            detail=str(candidate.get("entry_rule") or ""),
        )
        return True

    def _direct_setup_signal(self, index: int, bars, indicators, setup_id: str, side: str, reason: str):
        if setup_id not in self.enabled_setups:
            return None
        if not self._setup_side_enabled(setup_id, side):
            return None
        if not self._signal_bar_is_clean(index, bars, side, indicators):
            self._append_chart_event(
                key=("direct_unclean", setup_id, side, index),
                index=index,
                bars=bars,
                kind="failed",
                label="Direct setup unclean",
                price=float(bars.iloc[index]["close"]),
                side=side,
                setup_id=setup_id,
                detail=reason,
            )
            return None
        row = indicators.iloc[index]
        candidate = {
            "key": (setup_id, side, index),
            "setup_id": setup_id,
            "side": side,
            "signal_index": index,
            "signal_high": float(row["high"]),
            "signal_low": float(row["low"]),
            "entry_trigger": float(row["close"]),
            "entry_rule": "market_on_confirmed_rejection",
            "range_state": dict(self._range_state or {}),
            "direct_reason": reason,
            "ema21": self._ema_value(row, float(row["close"])),
        }
        return self._build_signal(index, bars, indicators, setup_id, side, candidate)

    def _build_signal(self, index: int, bars, indicators, setup_id: str, side: str, candidate: dict[str, Any]):
        row = indicators.iloc[index]
        close = float(row["close"])
        setup = dict(self.setup_cards.get(setup_id) or {})
        if not self._entry_hour_allowed(row):
            self._append_chart_event(
                key=("entry_hour_reject", setup_id, side, int(candidate.get("signal_index", index)), index),
                index=index,
                bars=bars,
                kind="failed",
                label="Rejected: session hour",
                price=close,
                side=side,
                setup_id=setup_id,
                detail="outside allowed hours",
            )
            return None
        signal_high = float(candidate.get("signal_high", row["high"]))
        signal_low = float(candidate.get("signal_low", row["low"]))
        stop_price = signal_low - self.tick_size if side == "Buy" else signal_high + self.tick_size
        entry_reference = float(candidate.get("entry_trigger", close))
        sl_ticks = max(1, int(math.ceil(round(abs(entry_reference - stop_price) / self.tick_size, 4))))
        
        # [EXPERT FIX] Enforce dynamic and absolute minimum stop loss floors
        atr = float(row.get("atr", 0.0) or 0.0)
        min_sl_atr = int((atr * self.min_stop_atr_mult) / self.tick_size) if atr > 0 else 0
        floor_sl_ticks = max(self.min_stop_ticks, min_sl_atr)
        
        if sl_ticks < floor_sl_ticks:
            sl_ticks = floor_sl_ticks
            # Push the stop price further away to match the new sl_ticks
            stop_price = entry_reference - (sl_ticks * self.tick_size) if side == "Buy" else entry_reference + (sl_ticks * self.tick_size)

        if self.max_stop_ticks > 0 and sl_ticks > self.max_stop_ticks:
            self._append_chart_event(
                key=("max_stop_reject", setup_id, side, int(candidate.get("signal_index", index)), index),
                index=index,
                bars=bars,
                kind="failed",
                label="Rejected: stop too wide",
                price=close,
                side=side,
                setup_id=setup_id,
                detail=f"{sl_ticks}>{self.max_stop_ticks} ticks",
            )
            return None
        candidate_target_price = candidate.get("target_price")
        target_price = None
        if self.use_candidate_price_targets and candidate_target_price is not None:
            target_price = float(candidate_target_price)
            target_distance_ticks = (
                (target_price - entry_reference) / self.tick_size
                if side == "Buy"
                else (entry_reference - target_price) / self.tick_size
            )
            if target_distance_ticks < 1.0:
                self._append_chart_event(
                    key=("target_reject", setup_id, side, int(candidate.get("signal_index", index)), index),
                    index=index,
                    bars=bars,
                    kind="failed",
                    label="Rejected: target too close",
                    price=close,
                    side=side,
                    setup_id=setup_id,
                    detail=f"target={target_price:.2f}",
                )
                return None
            tp_ticks = max(1, int(math.floor(target_distance_ticks + 1e-9)))
        else:
            tp_ticks = self._target_ticks_for_setup(setup_id, sl_ticks)
        execution_target_price = (
            entry_reference + (tp_ticks * self.tick_size)
            if side == "Buy"
            else entry_reference - (tp_ticks * self.tick_size)
        )
        key = (setup_id, side, int(candidate.get("signal_index", index)), index)
        if key == self._last_signal_key:
            return None
        self._last_signal_key = key

        confidence = {
            "E-01": 0.74,
            "E-02": 0.78,
            "E-03": 0.74,
            "E-04": 0.78,
            "E-05": 0.66,
            "E-06": 0.68,
            "E-07": 0.65,
            "E-08": 0.66,
            "E-11": 0.62,
        }.get(setup_id, 0.60)
        reason = candidate.get("direct_reason") or setup.get("title") or setup.get("canonical_name") or setup_id
        metadata = {
            "setup_id": setup_id,
            "setup_name": setup.get("title") or setup.get("canonical_name") or setup_id,
            "source_files": list(self.source_files),
            "setups_path": str(self.setups_path),
            "lock_in_version": self.metadata.get("lock_in_version"),
            "primary_chart": self.metadata.get("primary_chart"),
            "instrument_source": self.metadata.get("instrument"),
            "entry_rule": candidate.get("entry_rule"),
            "entry_reference": entry_reference,
            "signal_bar_index": int(candidate.get("signal_index", index)),
            "signal_bar_high": signal_high,
            "signal_bar_low": signal_low,
            "stop_price": stop_price,
            "stop_distance_ticks": sl_ticks,
            "target_ticks": tp_ticks,
            "target_mode": self.target_mode,
            "risk_reward": self.risk_reward,
            "target_price": round(float(candidate_target_price), 6) if candidate_target_price is not None else round(execution_target_price, 6),
            "candidate_target_price": candidate_target_price,
            "execution_target_price": round(execution_target_price, 6),
            "use_candidate_price_targets": self.use_candidate_price_targets,
            "range_mid_target_price": candidate.get("range_mid_target_price"),
            "ema21": candidate.get("ema21", self._ema_value(row, close)),
            "main_bias": self._main_bias(index, bars, indicators),
            "playbook_position_sizing": setup.get("position_sizing"),
            "failure_of": candidate.get("failure_of"),
            "range_state": candidate.get("range_state"),
            "e11_context": candidate.get("e11_context"),
        }
        self._append_chart_event(
            key=("signal_emitted", setup_id, side, int(candidate.get("signal_index", index)), index),
            index=index,
            bars=bars,
            kind="signal",
            label=str(reason),
            price=entry_reference,
            side=side,
            setup_id=setup_id,
            detail=f"SL {sl_ticks} / TP {tp_ticks}",
        )
        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
            confidence=confidence,
            reason=str(reason),
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    def _target_ticks_for_setup(self, setup_id: str, sl_ticks: int) -> int:
        if setup_id in self.target_ticks_by_setup:
            return max(1, int(self.target_ticks_by_setup[setup_id]))
        if self.target_mode in {"risk_multiple", "r_multiple", "rr", "risk_reward"}:
            ticks = int(math.ceil(max(1, sl_ticks) * max(0.1, self.risk_reward)))
            ticks = max(1, self.min_target_ticks, ticks)
            if self.max_target_ticks > 0:
                ticks = min(ticks, self.max_target_ticks)
            return max(1, ticks)
        return max(1, int(self.scalp_target_ticks))

    def _signal_bar_is_clean(self, index: int, bars: pd.DataFrame, side: str, indicators: pd.DataFrame | None = None) -> bool:
        if index < 1:
            return False
        bar = bars.iloc[index]
        prev = bars.iloc[index - 1]
        if float(bar["high"]) <= float(prev["high"]) and float(bar["low"]) >= float(prev["low"]):
            return False
        range_ticks = self._range_ticks(bar)
        if range_ticks <= 0:
            return False
        if self.min_signal_bar_atr_ratio > 0.0 and indicators is not None and index < len(indicators):
            atr = indicators.iloc[index].get("atr")
            if atr is not None and not pd.isna(atr) and float(atr) > 0:
                range_atr_ratio = (float(bar["high"]) - float(bar["low"])) / float(atr)
                if range_atr_ratio < self.min_signal_bar_atr_ratio:
                    return False
        if self.min_close_ema_distance_ticks > 0.0:
            ema_row = indicators.iloc[index] if indicators is not None and index < len(indicators) else bar
            close = float(bar["close"])
            ema = self._ema_value(ema_row, close)
            if abs(close - ema) / self.tick_size < self.min_close_ema_distance_ticks:
                return False
        body_ticks = abs(float(bar["close"]) - float(bar["open"])) / self.tick_size
        # [GOLDILOCKS] Configurable body ratio (defaults to 0.15, tightened to 0.30 or similar via config to filter dojis)
        if body_ticks / max(1.0, range_ticks) < self.min_body_ratio:
            return False
        upper_wick = (float(bar["high"]) - max(float(bar["open"]), float(bar["close"]))) / self.tick_size
        lower_wick = (min(float(bar["open"]), float(bar["close"])) - float(bar["low"])) / self.tick_size
        wrong_side = upper_wick if side == "Buy" else lower_wick
        return wrong_side / max(1.0, range_ticks) < 0.40

    def _entry_rule_for_signal_bar(self, row) -> str:
        if self._range_ticks(row) > self.large_signal_bar_threshold_ticks:
            return "limit_at_signal_close_large_bar"
        return "stop_one_tick_beyond_signal_bar"

    def _main_bias(self, index: int, bars: pd.DataFrame, indicators: pd.DataFrame) -> str:
        row = indicators.iloc[index]
        close = float(row["close"])
        ema = self._ema_value(row, close)
        
        # [GOLDILOCKS] Use ATR-normalized EMA slope to measure trend strength consistently across volatility regimes
        ema_slope_norm = self._ema_slope_ticks(index, indicators, lookback=5, normalized=True)
        min_slope_norm = float(self.params.get("ema_slope_min", 0.15))
        
        # [AUDITOR FIX] Restored price-EMA confluence. Slope alone is lagging noise on 1m.
        # A scalper MUST buy pullbacks when price is > EMA, not when it has dumped far below a lagging rising EMA.
        if close > ema and ema_slope_norm >= min_slope_norm:
            return "bullish"
        if close < ema and ema_slope_norm <= -min_slope_norm:
            return "bearish"
        return "neutral"

    def _has_two_pullback_legs(self, index: int, direction: str) -> bool:
        pivots = self._recent_pivots(self.swing_lows if direction == "down" else self.swing_highs, index, self.pullback_lookback_bars)
        if len(pivots) < 2:
            return False
        # [SWARM FIX] Ensure BOTH pivots are relatively recent to ensure momentum is fresh
        if index - int(pivots[-1]["index"]) > self.signal_after_pivot_bars:
            return False
        if index - int(pivots[-2]["index"]) > self.signal_after_pivot_bars + 15: # allow slightly more age for 2nd leg
            return False
        return self._pivot_leg_is_valid(pivots[-1], direction) and self._pivot_leg_is_valid(pivots[-2], direction)

    def _has_single_pullback_confirmation(self, index: int, direction: str) -> bool:
        pivots = self._recent_pivots(self.swing_lows if direction == "down" else self.swing_highs, index, self.pullback_lookback_bars)
        return bool(pivots and index - int(pivots[-1]["index"]) <= self.signal_after_pivot_bars and self._pivot_leg_is_valid(pivots[-1], direction))

    def _pivot_leg_is_valid(self, pivot: dict[str, Any], direction: str) -> bool:
        opposite = self.swing_highs if direction == "down" else self.swing_lows
        prior = [p for p in opposite if p["index"] < pivot["index"]]
        if not prior:
            return True
        anchor = prior[-1]
        duration = int(pivot["index"]) - int(anchor["index"])
        if duration < self.min_leg_bars:
            return False
        move_ticks = abs(float(anchor["price"]) - float(pivot["price"])) / self.tick_size
        return move_ticks >= self.min_leg_ticks

    def _near_ema(self, row) -> bool:
        ema = self._ema_value(row, float(row["close"]))
        distances = [
            abs(float(row["high"]) - ema),
            abs(float(row["low"]) - ema),
            abs(float(row["close"]) - ema),
        ]
        return min(distances) / self.tick_size <= self.ema_distance_ticks

    def _setup_side_enabled(self, setup_id: str, side: str) -> bool:
        side_key = str(side).lower()
        symbol_keys = {self.symbol.upper(), self._base_symbol(self.symbol.upper()), "*"}
        setup_keys = {str(setup_id), "*"}
        return not any(
            (symbol, setup, side_key) in self.disabled_setup_sides
            for symbol in symbol_keys
            for setup in setup_keys
        )

    def _is_choppy(self, index: int, bars: pd.DataFrame, indicators: pd.DataFrame) -> bool:
        if not self._ema_is_flat(index, indicators):
            return False
        start = max(1, index - 8 + 1)
        ratios = []
        for idx in range(start, index + 1):
            current = bars.iloc[idx]
            prev = bars.iloc[idx - 1]
            overlap = max(0.0, min(float(current["high"]), float(prev["high"])) - max(float(current["low"]), float(prev["low"])))
            current_range = max(self.tick_size, float(current["high"]) - float(current["low"]))
            ratios.append(overlap / current_range)
        return bool(ratios and sum(ratios) / len(ratios) >= self.chop_overlap_threshold)

    def _ema_is_flat(self, index: int, indicators: pd.DataFrame) -> bool:
        return abs(self._ema_slope_ticks(index, indicators, lookback=8)) <= self.chop_ema_flat_ticks

    def _ema_slope_ticks(self, index: int, indicators: pd.DataFrame, lookback: int, normalized: bool = False) -> float:
        if index <= lookback:
            return 0.0
        current = self._ema_value(indicators.iloc[index], float(indicators.iloc[index]["close"]))
        prior = self._ema_value(indicators.iloc[index - lookback], float(indicators.iloc[index - lookback]["close"]))
        raw_slope = current - prior
        
        if not normalized:
            return raw_slope / self.tick_size
            
        # [EXPERT FIX] Normalize trend strength to ATR to ensure cross-regime consistency
        atr = float(indicators.iloc[index].get("atr", 0.0) or 0.0)
        if atr <= 0:
            return 0.0
        return raw_slope / atr

    def _ema_value(self, row, fallback: float) -> float:
        value = row.get(self.ema_column)
        if value is None or pd.isna(value):
            value = row.get("ema_21", fallback)
        if value is None or pd.isna(value):
            return fallback
        return float(value)

    def _range_ticks(self, row) -> int:
        return int(round((float(row["high"]) - float(row["low"])) / self.tick_size))

    def _recent_pivots(self, pivots: list[dict[str, Any]], index: int, lookback: int) -> list[dict[str, Any]]:
        return [p for p in pivots if 0 <= index - int(p["index"]) <= lookback]

    def _level_tests(self, pivots: list[dict[str, Any]], level: float, tolerance: float, index: int, lookback: int) -> int:
        return sum(1 for p in self._recent_pivots(pivots, index, lookback) if abs(float(p["price"]) - level) <= tolerance)

    def _get_session_regime(self, row) -> str:
        """
        Stateful, timezone-aware session detector for NY and ASIAN windows.
        Handles midnight wraparound and Sunday Open thin-liquidity cooldowns.
        """
        raw_ts = row.get("timestamp")
        if raw_ts is None or pd.isna(raw_ts):
            return "OUT"
            
        try:
            val = None
            try:
                val = float(raw_ts)
            except (ValueError, TypeError):
                pass
            if val is not None:
                if val > 1e16:
                    ts = pd.to_datetime(val, unit='ns', utc=True)
                elif val > 1e13:
                    ts = pd.to_datetime(val, unit='us', utc=True)
                elif val > 1e10:
                    ts = pd.to_datetime(val, unit='ms', utc=True)
                elif val > 1e8:
                    ts = pd.to_datetime(val, unit='s', utc=True)
                else:
                    return "OUT"
            else:
                ts = pd.to_datetime(raw_ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # Standardize to New York for institutional session alignment
            et = ts.astimezone(zoneinfo.ZoneInfo("America/New_York"))
            if et.year < 2000:
                return "OUT"
            
            # Use modular 'minutes from midnight' to handle wraparound (18:30 - 02:00)
            mins = et.hour * 60 + et.minute
            
            # NY Window: 08:00 - 15:30 ET
            if 480 <= mins < 930:
                return "NY"
                
            # Asian Window: 18:30 - 02:00 ET (Next Day)
            # 1110 = 18:30, 120 = 02:00
            if mins >= 1110 or mins < 120:
                # [EXPERT FIX] Sunday Open Trap: Avoid first 60m of Sunday evening liquidity
                if et.weekday() == 6 and mins < 1170: # 1170 = 19:30 ET
                    return "OUT"
                return "ASIAN"
                
            return "OUT"
        except (TypeError, ValueError):
            return "OUT"

    def _entry_hour_allowed(self, row) -> bool:
        """Legacy helper maintained for backward compatibility with specific excluded_entry_hours configs."""
        regime = self._get_session_regime(row)
        if regime == "OUT":
            return False
        raw_ts = row.get("timestamp")
        if raw_ts is not None and not pd.isna(raw_ts):
            try:
                ts = pd.to_datetime(raw_ts)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                et = ts.astimezone(zoneinfo.ZoneInfo("America/New_York"))
                if self.allowed_entry_hours_et and et.hour not in self.allowed_entry_hours_et:
                    return False
                if self.excluded_entry_hours_et and et.hour in self.excluded_entry_hours_et:
                    return False
            except (TypeError, ValueError):
                pass
        return True

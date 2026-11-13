from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from src.strategies.always_in_15m_ema200 import AlwaysIn15mEMA200Strategy
from src.strategies.always_in_vote import AlwaysInVoteStrategy
from src.strategies.asian_breakout import AsianBreakoutStrategy
from src.strategies.blender import SignalBlender
from src.strategies.ema_pullback import EMAPullbackStrategy
from src.strategies.fx_at_one_glance import FXAtOneGlanceStrategy
from src.strategies.golden_zone_scalper import GoldenZoneScalperStrategy
from src.strategies.htf_level_retest import HTFLevelRetestStrategy
from src.strategies.ichimoku import IchimokuStrategy
from src.strategies.indicators import prepare_indicator_frame
from src.strategies.mtf_trend_scalper import MTFTrendScalperStrategy
from src.strategies.orb_vwap import ORBVWAPStrategy
from src.strategies.otz_continuation import OTZContinuationStrategy
from src.strategies.previous_bar_breakout import PreviousBarBreakoutStrategy
from src.strategies.price_action_bs import PriceActionBSStrategy
from src.strategies.prior_extreme_vwap_reversion import (
    PriorExtremeVWAPReversionStrategy,
)
from src.strategies.resting_level_breakout import RestingLevelBreakoutStrategy
from src.strategies.session_pivot_retest import SessionPivotRetestStrategy
from src.strategies.simple_market_metrics import SimpleMarketMetricsStrategy
from src.strategies.sss import SSSStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.volatility_squeeze_breakout import VolatilitySqueezeBreakoutStrategy
from src.telemetry.store import TelemetryStore


class StrategyEngine:
    DEFAULT_CONFIG = {
        "blender": {"mode": "confluence", "multi_agree_bonus": 2.0},
        "direction_filters": {},
        "strategies": [
            {"name": "ichimoku", "enabled": True, "priority": 100, "params": {}},
            {"name": "fx_at_one_glance", "enabled": True, "priority": 95, "params": {}},
            {"name": "ema_pullback", "enabled": True, "priority": 90, "params": {}},
            {"name": "asian_breakout", "enabled": True, "priority": 85, "params": {}},
            {"name": "sss", "enabled": True, "priority": 80, "params": {}},
            {"name": "trend_following", "enabled": False, "priority": 70, "params": {}},
            {"name": "orb_vwap", "enabled": False, "priority": 65, "params": {}},
            {"name": "previous_bar_breakout", "enabled": False, "priority": 60, "params": {}},
            {"name": "prior_extreme_vwap_reversion", "enabled": False, "priority": 55, "params": {}},
            {"name": "resting_level_breakout", "enabled": False, "priority": 50, "params": {}},
            {"name": "volatility_squeeze_breakout", "enabled": False, "priority": 45, "params": {}},
            {"name": "htf_level_retest", "enabled": False, "priority": 40, "params": {}},
            {"name": "session_pivot_retest", "enabled": False, "priority": 35, "params": {}},
            {"name": "simple_market_metrics", "enabled": False, "priority": 30, "params": {}},
            {"name": "otz_continuation", "enabled": False, "priority": 25, "params": {}},
        ],
    }

    def __init__(
        self,
        symbol: str,
        default_qty: int,
        bracket: dict[str, Any],
        telemetry: TelemetryStore | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.symbol = symbol
        self.default_qty = default_qty
        self.bracket = bracket
        self.tick_size = float(bracket.get("tick_size", 0.25))
        self.telemetry = telemetry or TelemetryStore(enabled=False)
        self.config = self._merge_config(config or {})
        self.blender_config = dict(self.config.get("blender", {}))
        self.direction_filters = {
            name: {side.lower() for side in sides}
            for name, sides in self.config.get("direction_filters", {}).items()
        }
        self.strategy_specs = self.config.get("strategies", [])
        self.strategies = self._build_strategies()
        self.blender = SignalBlender(float(self.blender_config.get("multi_agree_bonus", 2.0)))
        self._last_processed_timestamp = None
        self._last_snapshot = {}
        self._last_candidates = []
        self._last_selected = None
        self._cached_indicators_key = None
        self._cached_indicators_df = None
        self._cached_summary = None

    def _merge_config(self, override: dict[str, Any]) -> dict[str, Any]:
        merged = copy.deepcopy(self.DEFAULT_CONFIG)
        if "blender" in override:
            merged["blender"] = copy.deepcopy(override["blender"])
        if "direction_filters" in override:
            merged["direction_filters"] = copy.deepcopy(override["direction_filters"])
        if "strategies" in override:
            merged["strategies"] = copy.deepcopy(override["strategies"])
        return merged

    def _build_strategies(self):
        registry = {
            "ichimoku": IchimokuStrategy,
            "fx_at_one_glance": FXAtOneGlanceStrategy,
            "golden_zone_scalper": GoldenZoneScalperStrategy,
            "ema_pullback": EMAPullbackStrategy,
            "price_action_bs": PriceActionBSStrategy,
            "trend_following": TrendFollowingStrategy,
            "asian_breakout": AsianBreakoutStrategy,
            "sss": SSSStrategy,
            "orb_vwap": ORBVWAPStrategy,
            "previous_bar_breakout": PreviousBarBreakoutStrategy,
            "prior_extreme_vwap_reversion": PriorExtremeVWAPReversionStrategy,
            "resting_level_breakout": RestingLevelBreakoutStrategy,
            "volatility_squeeze_breakout": VolatilitySqueezeBreakoutStrategy,
            "htf_level_retest": HTFLevelRetestStrategy,
            "session_pivot_retest": SessionPivotRetestStrategy,
            "simple_market_metrics": SimpleMarketMetricsStrategy,
            "otz_continuation": OTZContinuationStrategy,
            "always_in_15m_ema200": AlwaysIn15mEMA200Strategy,
            "always_in_vote": AlwaysInVoteStrategy,
            "mtf_trend_scalper": MTFTrendScalperStrategy,
        }
        strategies = []
        for spec in self.strategy_specs:
            strategy_name = spec.get("name")
            cls = registry.get(strategy_name)
            if cls is None or not spec.get("enabled", True):
                continue
            params = dict(spec.get("params", {}))
            params.update(
                {
                    "priority": spec.get("priority", 0),
                    "enabled": spec.get("enabled", True),
                    "quantity": params.get("quantity", self.default_qty),
                    "sl_ticks": params.get("sl_ticks", self.bracket.get("sl", 40)),
                    "tp_ticks": params.get("tp_ticks", self.bracket.get("tp", 80)),
                    "tick_size": params.get("tick_size", self.tick_size),
                    "min_atr": params.get("min_atr", self.bracket.get("min_atr", 1.0)),
                    "max_atr": params.get("max_atr", self.bracket.get("max_atr", 100.0)),
                }
            )
            if strategy_name == "trend_following":
                base_adx = 25 if self.symbol.upper().startswith("MCL") else 20
                params["adx_threshold"] = params.get("adx_threshold", base_adx + float(params.pop("adx_threshold_delta", 0)))
            strategy_instance = cls(self.symbol, params)
            strategies.append(strategy_instance)
        strategies.sort(key=lambda strategy: strategy.priority, reverse=True)
        return strategies

    def reset(self):
        self._last_processed_timestamp = None
        self._last_snapshot = {}
        self._cached_indicators_key = None
        self._cached_indicators_df = None
        self._cached_summary = None
        for strategy in self.strategies:
            strategy.reset()

    def prepare_data(self, df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        bars = df.copy()
        bars.columns = [c.lower() for c in bars.columns]
        if "timestamp" in bars.columns:
            bars.loc[:, "timestamp"] = pd.to_datetime(bars["timestamp"])
        if "price" not in bars.columns and "close" in bars.columns:
            bars.loc[:, "price"] = bars["close"]

        last_ts = bars["timestamp"].iloc[-1] if "timestamp" in bars.columns else None
        raw_close = bars["close"].iloc[-1] if "close" in bars.columns else (bars["price"].iloc[-1] if "price" in bars.columns else None)
        try:
            close_val = float(raw_close) if raw_close is not None and pd.notna(raw_close) else None
        except (ValueError, TypeError):
            close_val = None

        cache_key = (len(bars), str(last_ts), close_val, limit)
        if self._cached_indicators_key == cache_key and self._cached_indicators_df is not None:
            return self._cached_indicators_df.copy()

        prepared = prepare_indicator_frame(bars, self.tick_size, limit=limit)
        self._cached_indicators_key = cache_key
        self._cached_indicators_df = prepared
        return prepared.copy()

    def _apply_direction_filter(self, signal) -> bool:
        allowed = self.direction_filters.get(signal.strategy_name)
        if not allowed:
            return True
        return signal.side.lower() in allowed

    def _snapshot(self, bars, indicators, index):
        strategy_views = {}
        for strategy in self.strategies:
            strategy_views[strategy.name] = strategy.get_snapshot(index, bars, indicators)
        self._last_snapshot = strategy_views
        return strategy_views

    def evaluate_index(self, bars: pd.DataFrame, indicators: pd.DataFrame, index: int):
        accepted = None
        candidates = []
        candidate_signals = []
        
        row = indicators.iloc[index]
        timestamp = row.get("timestamp")
        if timestamp is not None and self._last_processed_timestamp is not None and timestamp <= self._last_processed_timestamp:
            return None, []
            
        price = float(row.get("close", row.get("price", 0)))
        self._last_candidates = []
        self._last_selected = None
        is_priming = any(getattr(s, "is_priming", False) for s in self.strategies)
        
        for strategy in self.strategies:
            if not getattr(strategy, "enabled", True):
                continue
            signal = strategy.on_bar(index, bars, indicators)
            if signal is None:
                continue
            if not self._apply_direction_filter(signal):
                if not is_priming:
                    self.telemetry.record_signal(
                        self.symbol,
                        signal.strategy_name,
                        "blocked_direction",
                        signal.side,
                        timestamp,
                        price,
                        signal.reason,
                        signal.metadata,
                    )
                continue

            intent = signal.to_trade_intent(self.symbol)
            candidates.append(intent)
            candidate_signals.append(signal)
            if not is_priming:
                self.telemetry.record_signal(
                    self.symbol,
                    signal.strategy_name,
                    "candidate",
                    signal.side,
                    timestamp,
                    price,
                    signal.reason,
                    signal.metadata,
                )

        if candidates:
            selected_index, _, candidate_blend_data = self._select_candidate_index(candidate_signals)
            for idx, candidate in enumerate(candidates):
                candidate.setdefault("metadata", {})
                if idx < len(candidate_blend_data) and candidate_blend_data[idx]:
                    candidate["metadata"]["blender"] = candidate_blend_data[idx]
            if selected_index is not None:
                accepted = dict(candidates[selected_index])
                self._last_selected = dict(accepted)
            self._last_candidates = [dict(candidate) for candidate in candidates]

        self._snapshot(bars, indicators, index)
        if accepted is not None and not is_priming:
            self.telemetry.record_signal(
                self.symbol,
                accepted["strategy_name"],
                "selected",
                accepted["side"],
                timestamp,
                price,
                accepted.get("reason", ""),
                accepted.get("metadata", {}),
            )
        return accepted, candidates

    def _select_candidate_index(self, candidate_signals):
        if not candidate_signals:
            return None, None, []

        mode = str(self.blender_config.get("mode", "confluence")).strip().lower()
        if mode == "priority_first":
            metadata = {
                "mode": mode,
                "base_score": None,
                "effective_score": None,
                "agreement_count": 1,
                "multi_agree_bonus": float(self.blender_config.get("multi_agree_bonus", 2.0)),
            }
            return 0, metadata, [dict(metadata) for _ in candidate_signals]

        scored = self.blender.score(candidate_signals)
        if not scored:
            return 0, None, []

        candidate_blend_data = []
        for signal in candidate_signals:
            match = next((item for item in scored if item.signal is signal), None)
            if match is None:
                candidate_blend_data.append({})
                continue
            candidate_blend_data.append(
                {
                    "mode": mode,
                    "base_score": match.base_score,
                    "effective_score": match.effective_score,
                    "agreement_count": match.agreement_count,
                    "multi_agree_bonus": float(self.blender_config.get("multi_agree_bonus", 2.0)),
                }
            )

        blended = scored[0]
        selected_index = candidate_signals.index(blended.signal)
        return selected_index, {
            "mode": mode,
            "base_score": blended.base_score,
            "effective_score": blended.effective_score,
            "agreement_count": blended.agreement_count,
            "multi_agree_bonus": float(self.blender_config.get("multi_agree_bonus", 2.0)),
        }, candidate_blend_data

    def prime(self, bars: pd.DataFrame, indicators: pd.DataFrame, end_index: int):
        for strategy in self.strategies:
            strategy.is_priming = True
        try:
            for idx in range(end_index + 1):
                self.evaluate_index(bars, indicators, idx)
        finally:
            for strategy in self.strategies:
                strategy.is_priming = False
        
        if not indicators.empty:
            self._last_processed_timestamp = indicators.iloc[end_index].get("timestamp")

    def analyze(self, data: pd.DataFrame):
        if data.empty:
            return None

        indicators = self.prepare_data(data)
        bars = pd.DataFrame(indicators[["timestamp", "open", "high", "low", "close", "volume", "price"]]).copy()

        signal = None
        last_idx = len(indicators) - 1
        start_idx = 0
        if self._last_processed_timestamp is not None and "timestamp" in indicators.columns:
            ts_index = pd.Index(indicators["timestamp"])
            if ts_index.is_monotonic_increasing:
                start_idx = int(ts_index.searchsorted(self._last_processed_timestamp, side="right"))
                if start_idx > last_idx:
                    return None
        elif last_idx > 0:
            # Cold analyze() call without prior prime(): prime historical bars silently
            # so stateful strategies build state without emitting warmup telemetry.
            self.prime(bars, indicators, last_idx - 1)
            start_idx = last_idx

        for idx in range(start_idx, last_idx + 1):
            res, _ = self.evaluate_index(bars, indicators, idx)
            if idx == last_idx:
                signal = res

        if not indicators.empty:
            self._last_processed_timestamp = indicators.iloc[-1].get("timestamp")

        return signal

    def get_summary(self, data: pd.DataFrame):
        if data.empty:
            return {"description": "No data", "strategy_details": {}}
        indicators = self.prepare_data(data)
        bars = pd.DataFrame(indicators[["timestamp", "open", "high", "low", "close", "volume", "price"]]).copy()
        index = len(indicators) - 1
        self._snapshot(bars, indicators, index)
        row = indicators.iloc[index]
        
        # Helper to clean numeric values for JSON serialization
        def _c(v):
            import pandas as pd
            return float(v) if v is not None and not pd.isna(v) else None

        import datetime
        bar_ts = row.get("timestamp")
        data_stale = False
        if bar_ts is not None and not (isinstance(bar_ts, float) and pd.isna(bar_ts)):
            try:
                val = None
                try:
                    val = float(bar_ts)
                except (ValueError, TypeError):
                    pass
                if val is not None:
                    if val > 1e16:
                        b_dt = pd.to_datetime(val, unit='ns', utc=True)
                    elif val > 1e13:
                        b_dt = pd.to_datetime(val, unit='us', utc=True)
                    elif val > 1e10:
                        b_dt = pd.to_datetime(val, unit='ms', utc=True)
                    elif val > 1e8:
                        b_dt = pd.to_datetime(val, unit='s', utc=True)
                    else:
                        b_dt = None
                else:
                    b_dt = pd.to_datetime(bar_ts)
                    if b_dt is not None and not pd.isna(b_dt):
                        if b_dt.tzinfo is None:
                            b_dt = b_dt.replace(tzinfo=datetime.timezone.utc)
                        else:
                            b_dt = b_dt.tz_convert(datetime.timezone.utc)

                if b_dt is not None and not pd.isna(b_dt):
                    regime = self._compute_session_regime(fallback_to_now=True)
                    if regime not in ("CLOSED", "MAINTENANCE") and abs((datetime.datetime.now(datetime.timezone.utc) - b_dt).total_seconds()) > 300:
                        data_stale = True
            except Exception:
                pass

        # Always extract global indicators that the dashboard expects
        global_indicators = {
            "price": _c(row.get("price")),
            "vwap": _c(row.get("vwap")),
            "ema_200": _c(row.get("ema_200")),
            "rsi": _c(row.get("rsi")),
            "adx": _c(row.get("adx")),
            "atr": _c(row.get("atr")),
            "session_regime": self._compute_session_regime(row.get("timestamp"), fallback_to_now=True),
            "data_stale": data_stale,
            "last_bar_timestamp": str(row.get("timestamp")) if row.get("timestamp") is not None and not pd.isna(row.get("timestamp")) else None,
            "bar_count": len(indicators)
        }

        aggregate = copy.deepcopy(self._last_snapshot.get("trend_following", {}))
        if not aggregate and self._last_snapshot:
            first_key = next(iter(self._last_snapshot))
            aggregate = copy.deepcopy(self._last_snapshot.get(first_key, {}))
            
        # Merge global indicators so they are always present
        aggregate.update(global_indicators)

        return {
            "description": self.get_strategy_description(),
            "strategy_details": copy.deepcopy(self._last_snapshot),
            "aggregate_indicators": aggregate,
            "blender": {
                "candidates": copy.deepcopy(self._last_candidates),
                "selected": copy.deepcopy(self._last_selected),
                "mode": str(self.blender_config.get("mode", "confluence")),
                "multi_agree_bonus": float(self.blender_config.get("multi_agree_bonus", 2.0)),
            },
        }

    def get_chart_overlays(self, data: pd.DataFrame):
        if data.empty:
            return {"lines": [], "levels": [], "markers": [], "events": []}
        indicators = self.prepare_data(data)
        bars = indicators[["timestamp", "open", "high", "low", "close", "volume", "price"]].copy()
        index = len(indicators) - 1
        overlays = {"lines": [], "levels": [], "markers": [], "events": [], "state": {}}
        for strategy in self.strategies:
            getter = getattr(strategy, "get_chart_overlays", None)
            if getter is None:
                continue
            current = getter(index, bars, indicators) or {}
            for key in ("lines", "levels", "markers", "events"):
                overlays[key].extend(current.get(key) or [])
            if current.get("state"):
                overlays["state"][strategy.name] = current["state"]
        return overlays

    def get_strategy_description(self):
        if not self.strategies:
            return "No active strategies."
        return " | ".join([strategy.get_description() for strategy in self.strategies])

    def _compute_session_regime(self, raw_ts=None, fallback_to_now=False) -> str:
        from src.utils.session_utils import get_session_regime
        return get_session_regime(raw_ts, fallback_to_now=fallback_to_now, include_london=False)


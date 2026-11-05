import logging
from typing import Any

import pandas as pd

from src.config.settings import get_symbol_brackets, load_stack_config
from src.strategies.engine import StrategyEngine
from src.strategies.indicators import prepare_indicator_frame
from src.telemetry.store import TelemetryStore


class StrategyAgent:
    MICRO_PREFIXES = {"MNQ", "MES", "MYM", "MCL", "MGC", "M2K", "M6E", "M6A", "M6B", "M6J", "MNK"}

    def __init__(self, symbol: str, strategy_config: dict[str, Any] | None = None, telemetry: TelemetryStore | None = None):
        self.symbol = symbol
        self.logger = logging.getLogger(f"StrategyAgent-{symbol}")
        
        # FIX: Dynamically load brackets from YAML instead of hardcoded dict
        stack_config = load_stack_config()
        self.all_brackets = get_symbol_brackets(stack_config)
        self.base_symbol = self._extract_base(symbol)
        bracket = self.all_brackets.get(self.base_symbol, {})

        self.sl_ticks = bracket.get("sl", 40)
        self.tp_ticks = bracket.get("tp", 80)
        self.min_atr = bracket.get("min_atr", 1.0)
        self.max_atr = bracket.get("max_atr", 100.0)
        self.tick_size = bracket.get("tick_size", 0.25)

        self.is_micro = any(self.base_symbol.startswith(prefix) for prefix in self.MICRO_PREFIXES)
        self.default_qty = 3 if self.is_micro else 1
        self.telemetry = telemetry or TelemetryStore()
        self.engine = StrategyEngine(
            symbol=self.symbol,
            default_qty=self.default_qty,
            bracket=bracket,
            telemetry=self.telemetry,
            config=strategy_config,
        )

        self.logger.info(
            f"Strategy engine initialized for {symbol} (base={self.base_symbol}, "
            f"micro={self.is_micro}, qty={self.default_qty}, strategies={len(self.engine.strategies)})"
        )

    @staticmethod
    def _extract_base_static(symbol: str) -> str:
        # Fallback for when we don't have an instance but need base ticker
        # This is a bit duplicative but safer than passing instances around
        # during early resolution.
        from src.config.settings import get_symbol_brackets, load_stack_config
        brackets = get_symbol_brackets(load_stack_config())
        
        symbol_upper = symbol.upper()
        if symbol_upper in brackets:
            return symbol_upper
        best_match = None
        for known in brackets:
            if symbol_upper.startswith(known):
                if best_match is None or len(known) > len(best_match):
                    best_match = known
        if best_match:
            return best_match
        import re

        match = re.match(r"^([A-Z0-9]+?)[FGHJKMNQUVXZ]\d{1,2}$", symbol_upper)
        if match:
            return match.group(1)
        return symbol_upper

    def _extract_base(self, symbol: str) -> str:
        symbol_upper = symbol.upper()
        if symbol_upper in self.all_brackets:
            return symbol_upper
        best_match = None
        for known in self.all_brackets:
            if symbol_upper.startswith(known):
                if best_match is None or len(known) > len(best_match):
                    best_match = known
        if best_match:
            return best_match
        import re

        match = re.match(r"^([A-Z0-9]+?)[FGHJKMNQUVXZ]\d{1,2}$", symbol_upper)
        if match:
            return match.group(1)
        return symbol_upper

    def _calculate_indicators(self, df):
        return prepare_indicator_frame(df.copy(), self.tick_size)

    def reset(self):
        self.engine.reset()

    def analyze(self, data: pd.DataFrame):
        # Keep compatibility with previous contract: don't emit signals before 200 bars.
        if len(data) < 200:
            if len(data) % 20 == 0: # Log every 20 bars to avoid spam
                self.logger.info(f"Warming up indicators: {len(data)}/200 bars collected...")
            return None
        return self.engine.analyze(data)

    def get_strategy_description(self):
        return self.engine.get_strategy_description()

    def get_latest_indicators(self, data: pd.DataFrame):
        if len(data) == 0:
            return {"status": "Waiting for data..."}
        summary = self.engine.get_summary(data)
        aggregate = summary.get("aggregate_indicators", {})
        aggregate["strategy_details"] = summary.get("strategy_details", {})
        aggregate["blender"] = summary.get("blender", {})
        aggregate["status"] = aggregate.get("status", "Active")
        return aggregate

    def get_chart_overlays(self, data: pd.DataFrame):
        return self.engine.get_chart_overlays(data)

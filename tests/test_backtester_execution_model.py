# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
import pandas as pd

from src.utils.backtester import Backtester


class _OneSignalEngine:
    config = {"strategies": []}

    def __init__(self, signal_index: int):
        self.signal_index = signal_index

    def reset(self):
        pass

    def evaluate_index(self, bars, indicators, index):
        if index != self.signal_index:
            return None, []
        return (
            {
                "side": "Buy",
                "quantity": 1,
                "symbol": "MNQM6",
                "sl_ticks": 100,
                "tp_ticks": 100,
                "strategy_name": "unit_signal",
                "confidence": 1.0,
                "reason": "test signal",
                "metadata": {},
            },
            [],
        )


class _MultiSignalEngine:
    config = {"strategies": []}

    def __init__(self, signal_indexes, sl_ticks=100, tp_ticks=1):
        self.signal_indexes = set(signal_indexes)
        self.sl_ticks = sl_ticks
        self.tp_ticks = tp_ticks

    def reset(self):
        pass

    def evaluate_index(self, bars, indicators, index):
        if index not in self.signal_indexes:
            return None, []
        return (
            {
                "side": "Buy",
                "quantity": 1,
                "symbol": "MNQM6",
                "sl_ticks": self.sl_ticks,
                "tp_ticks": self.tp_ticks,
                "strategy_name": "unit_signal",
                "confidence": 1.0,
                "reason": "test signal",
                "metadata": {},
            },
            [],
        )


def _bars():
    return pd.DataFrame(
        [
            {"timestamp": "2026-04-01T14:30:00Z", "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10},
            {"timestamp": "2026-04-01T14:31:00Z", "open": 101.0, "high": 101.5, "low": 100.5, "close": 101.0, "volume": 10},
            {"timestamp": "2026-04-01T14:32:00Z", "open": 110.0, "high": 110.5, "low": 109.5, "close": 110.0, "volume": 10},
        ]
    )


def _backtester(execution_model: str):
    bt = Backtester(
        "MNQM6",
        strategy_config={"strategies": []},
        telemetry_enabled=False,
        screenshot_config={"enabled": False},
        routing_mode="optimizer_fast",
        execution_model=execution_model,
    )
    bt.strategy.engine = _OneSignalEngine(signal_index=1)
    bt.strategy._calculate_indicators = lambda frame: frame.assign(price=frame["close"])
    bt.slippage_ticks = 1
    return bt


def test_next_bar_open_enters_after_signal_bar_open():
    bt = _backtester("next_bar_open")

    bt.run_dataframe(_bars())

    assert len(bt.active_trades) == 1
    assert bt.active_trades[0]["entry"] == 110.25
    assert bt.active_trades[0]["time_open"] == "2026-04-01T14:32:00+00:00"


def test_same_bar_open_compatibility_mode_enters_signal_bar_open():
    bt = _backtester("same_bar_open")

    bt.run_dataframe(_bars())

    assert len(bt.active_trades) == 1
    assert bt.active_trades[0]["entry"] == 101.25
    assert bt.active_trades[0]["time_open"] == "2026-04-01T14:31:00+00:00"


def test_daily_profit_lock_skips_new_entries_after_realized_goal():
    bt = Backtester(
        "MNQM6",
        strategy_config={"risk": {"daily_profit_lock": 0.1}, "strategies": []},
        telemetry_enabled=False,
        screenshot_config={"enabled": False},
        routing_mode="optimizer_fast",
        execution_model="next_bar_open",
    )
    bt.strategy.engine = _MultiSignalEngine(signal_indexes=[0, 2])
    bt.strategy._calculate_indicators = lambda frame: frame.assign(price=frame["close"])
    bt.slippage_ticks = 0
    bt.commission_per_contract = 0
    bars = pd.DataFrame(
        [
            {"timestamp": "2026-04-01T14:30:00Z", "open": 100.0, "high": 100.25, "low": 99.5, "close": 100.0, "volume": 10},
            {"timestamp": "2026-04-01T14:31:00Z", "open": 101.0, "high": 101.5, "low": 100.5, "close": 101.25, "volume": 10},
            {"timestamp": "2026-04-01T14:32:00Z", "open": 102.0, "high": 102.5, "low": 101.5, "close": 102.0, "volume": 10},
            {"timestamp": "2026-04-01T14:33:00Z", "open": 103.0, "high": 103.5, "low": 102.5, "close": 103.0, "volume": 10},
        ]
    )

    results = bt.run_dataframe(bars)

    assert len(results["trades"]) == 1
    assert not bt.active_trades


def test_daily_profit_lock_flattens_open_trade_next_bar():
    bt = Backtester(
        "MNQM6",
        strategy_config={"risk": {"daily_profit_lock": 0.1}, "strategies": []},
        telemetry_enabled=False,
        screenshot_config={"enabled": False},
        routing_mode="optimizer_fast",
        execution_model="next_bar_open",
    )
    bt.strategy.engine = _MultiSignalEngine(signal_indexes=[0, 2], tp_ticks=1000)
    bt.strategy._calculate_indicators = lambda frame: frame.assign(price=frame["close"])
    bt.slippage_ticks = 0
    bt.commission_per_contract = 0
    bars = pd.DataFrame(
        [
            {"timestamp": "2026-04-01T14:30:00Z", "open": 100.0, "high": 100.25, "low": 99.5, "close": 100.0, "volume": 10},
            {"timestamp": "2026-04-01T14:31:00Z", "open": 101.0, "high": 101.5, "low": 100.5, "close": 101.25, "volume": 10},
            {"timestamp": "2026-04-01T14:32:00Z", "open": 102.0, "high": 102.5, "low": 101.5, "close": 102.0, "volume": 10},
            {"timestamp": "2026-04-01T14:33:00Z", "open": 103.0, "high": 103.5, "low": 102.5, "close": 103.0, "volume": 10},
        ]
    )

    results = bt.run_dataframe(bars)

    assert len(results["trades"]) == 1
    assert results["trades"][0]["result"] == "DAILY-PROFIT-LOCK"
    assert results["trades"][0]["exit"] == 102.0
    assert not bt.active_trades


def test_backtester_session_reset_and_consecutive_loss_halts():
    bt = Backtester(
        "MNQM6",
        strategy_config={"strategies": []},
        telemetry_enabled=False,
        screenshot_config={"enabled": False},
        routing_mode="optimizer_fast",
        execution_model="same_bar_open",
    )
    bt.strategy.engine = _MultiSignalEngine(signal_indexes=[0, 1, 2, 4], sl_ticks=10, tp_ticks=100)
    bt.strategy._calculate_indicators = lambda frame: frame.assign(price=frame["close"])
    bt.slippage_ticks = 0
    bt.commission_per_contract = 0

    bars = pd.DataFrame(
        [
            # Bar 0 (Session 1): open 100, drops to 90 -> SL hit (-10 ticks)
            {"timestamp": "2026-04-01T15:00:00Z", "open": 100.0, "high": 100.0, "low": 90.0, "close": 90.0, "volume": 10},
            # Bar 1 (Session 1): open 90, drops to 80 -> SL hit (-10 ticks). Consecutive losses = 2 -> Halted!
            {"timestamp": "2026-04-01T15:01:00Z", "open": 90.0, "high": 90.0, "low": 80.0, "close": 80.0, "volume": 10},
            # Bar 2 (Session 1): signal rejected due to consecutive loss halt!
            {"timestamp": "2026-04-01T15:02:00Z", "open": 80.0, "high": 85.0, "low": 80.0, "close": 85.0, "volume": 10},
            # Bar 3: Maintenance / roll boundary: 2026-04-01 22:00 UTC = 18:00 ET. New Globex session begins!
            {"timestamp": "2026-04-01T22:00:00Z", "open": 85.0, "high": 85.0, "low": 85.0, "close": 85.0, "volume": 10},
            # Bar 4 (Session 2): New session allows trade!
            {"timestamp": "2026-04-01T22:01:00Z", "open": 85.0, "high": 100.0, "low": 84.0, "close": 95.0, "volume": 10},
        ]
    )

    results = bt.run_dataframe(bars)
    assert len(results["trades"]) == 2
    assert len(bt.active_trades) == 1
    assert bt.active_trades[0]["entry"] == 85.0


class _ReversalSignalEngine:
    config = {"strategies": []}

    def reset(self):
        pass

    def evaluate_index(self, bars, indicators, index):
        if index == 0:
            return (
                {
                    "side": "Buy",
                    "quantity": 1,
                    "symbol": "MNQM6",
                    "sl_ticks": 100,
                    "tp_ticks": 100,
                    "strategy_name": "unit_buy",
                    "confidence": 1.0,
                    "reason": "test buy",
                    "metadata": {},
                },
                [],
            )
        elif index == 1:
            return (
                {
                    "side": "Sell",
                    "quantity": 1,
                    "symbol": "MNQM6",
                    "sl_ticks": 100,
                    "tp_ticks": 100,
                    "strategy_name": "unit_sell",
                    "confidence": 1.0,
                    "reason": "test reversal sell",
                    "metadata": {},
                },
                [],
            )
        return None, []


def test_backtester_reversal_trade_recorded():
    bt = Backtester(
        "MNQM6",
        strategy_config={"strategies": []},
        telemetry_enabled=False,
        screenshot_config={"enabled": False},
        routing_mode="optimizer_fast",
        execution_model="same_bar_open",
    )
    bt.routing_config.setdefault("entry_filters", {})["allow_flips"] = True
    bt.strategy.engine = _ReversalSignalEngine()
    bt.strategy._calculate_indicators = lambda frame: frame.assign(price=frame["close"])
    bt.slippage_ticks = 0
    bt.commission_per_contract = 0

    bars = pd.DataFrame(
        [
            {"timestamp": "2026-04-01T14:30:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10},
            {"timestamp": "2026-04-01T14:31:00Z", "open": 105.0, "high": 106.0, "low": 104.0, "close": 105.0, "volume": 10},
        ]
    )

    results = bt.run_dataframe(bars)
    assert len(results["trades"]) == 1
    assert results["trades"][0]["result"] == "REVERSAL"
    assert results["trades"][0]["entry"] == 100.0
    assert results["trades"][0]["exit"] == 105.0
    assert results["trades"][0]["pnl"] > 0

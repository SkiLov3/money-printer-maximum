# pyright: reportAttributeAccessIssue=false
from pathlib import Path

import pandas as pd
import yaml

from src.agents.risk_agent import RiskAgent
from src.optimization.combine_rules import evaluate_combine_rules
from src.optimization.optimizer import StrategyOptimizer
from src.strategies.engine import StrategyEngine
from src.strategies.indicators import prepare_indicator_frame
from src.strategies.price_action_bs import PriceActionBSStrategy
from src.trading.filters import FilterManager
from src.trading.router import TradeRouter


class _ExecutionStub:
    def execute_trade(self, trade_intent):
        return {"status": "simulation_fill", "avgPrice": 100.0, "orderId": "SIM-PRICE-ACTION"}


def _stack_config():
    return yaml.safe_load(Path("config/trading_stack.example.yaml").read_text(encoding="utf-8"))


def _e11_bars() -> pd.DataFrame:
    count = 130
    ts = pd.date_range("2026-04-30 16:00:00", periods=count, freq="1min")
    base = pd.Series([102.0] * count, dtype="float64")
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "volume": [100] * count,
            "price": base,
        }
    )
    bars.loc[100, "low"] = 100.0
    bars.loc[106, "low"] = 100.25
    bars.loc[109, ["open", "high", "low", "close", "price"]] = [102.0, 102.75, 101.75, 102.0, 102.0]
    bars.loc[110, ["open", "high", "low", "close", "price"]] = [102.5, 103.25, 102.25, 103.0, 103.0]
    bars.loc[111, ["open", "high", "low", "close", "price"]] = [103.25, 103.75, 103.0, 103.6, 103.6]
    return bars


def test_branch_config_enables_gz_legacy_stack():
    stack = _stack_config()
    strategy_block = stack["strategies"]
    enabled = [item["name"] for item in strategy_block["strategies"] if item.get("enabled", True)]

    if "price_action_bs" not in enabled:
        import pytest
        pytest.skip("price_action_bs is not enabled in the example stack config; skipping branch default config checks.")

    assert "price_action_bs" in enabled
    assert strategy_block["blender"] == {"mode": "priority_first", "multi_agree_bonus": 0.0}
    assert strategy_block["direction_filters"]["price_action_bs"] == ["buy", "sell"]

    params = next(s for s in strategy_block["strategies"] if s["name"] == "price_action_bs")["params"]
    assert params["ema_period"] == 21
    assert params["symbol_params"]["MNQ"]["max_stop_ticks"] == 150
    assert params["symbol_params"]["MGC"]["max_stop_ticks"] in (50, 60)
    assert params["symbol_params"]["MCL"]["max_stop_ticks"] == 60

    assert stack["routing"]["risk_per_trade_pct"] >= 0.18
    assert stack["routing"]["daily_profit_target"] >= 1500
    assert stack["routing"]["daily_loss_halt"] >= 300
    assert stack["feature_flags"]["routing_enabled"] is True
    assert stack["feature_flags"]["instrument_sizing_enabled"] is True
    assert stack["feature_flags"]["zone_engine_enabled"] is False
    assert stack["feature_flags"].get("meta_optimizer_enabled", False) is False


def test_e11_range_double_test_is_not_live_enabled_by_default():
    strategy = PriceActionBSStrategy("MNQM6", {"tick_size": 0.25})

    assert "E-11" not in strategy.enabled_setups


def test_signal_quality_params_reject_low_atr_displacement():
    strategy = PriceActionBSStrategy("MNQM6", {"tick_size": 0.25, "min_signal_bar_atr_ratio": 1.0})
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-30 10:00:00", periods=20, freq="1min"),
            "open": [100.0] * 20,
            "high": [100.5] * 19 + [100.75],
            "low": [99.5] * 19 + [100.5],
            "close": [100.25] * 19 + [100.75],
            "volume": [100] * 20,
            "price": [100.25] * 19 + [100.75],
        }
    )
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    assert strategy._signal_bar_is_clean(19, bars, "Buy", indicators) is False


def test_signal_quality_params_reject_close_still_on_ema():
    strategy = PriceActionBSStrategy("MNQM6", {"tick_size": 0.25, "min_close_ema_distance_ticks": 4})
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-30 10:00:00", periods=3, freq="1min"),
            "open": [100.0, 100.0, 100.0],
            "high": [100.5, 100.5, 101.25],
            "low": [99.5, 99.5, 99.75],
            "close": [100.25, 100.25, 100.5],
            "volume": [100, 100, 100],
            "price": [100.25, 100.25, 100.5],
        }
    )
    indicators = prepare_indicator_frame(bars, tick_size=0.25)
    indicators.loc[2, "ema_21"] = 100.25

    assert strategy._signal_bar_is_clean(2, bars, "Buy", indicators) is False


def test_disabled_setup_sides_support_symbol_and_base_symbol_keys():
    strategy = PriceActionBSStrategy(
        "MGCM6",
        {
            "tick_size": 0.1,
            "disabled_setup_sides": ["MGC:E-01:Buy", "MNQ:E-05:Sell"],
        },
    )

    assert strategy._setup_side_enabled("E-01", "Buy") is False
    assert strategy._setup_side_enabled("E-01", "Sell") is True
    assert strategy._setup_side_enabled("E-05", "Sell") is True


def test_strategy_engine_builds_gz_legacy_stack_from_branch_config():
    stack = _stack_config()
    engine = StrategyEngine(
        symbol="MNQM6",
        default_qty=3,
        bracket={"sl": 65, "tp": 130, "tick_size": 0.25, "min_atr": 5.0, "max_atr": 80.0},
        config=stack["strategies"],
    )

    sorted_enabled = sorted([s for s in stack["strategies"]["strategies"] if s.get("enabled", True)], key=lambda x: x.get("priority", 0), reverse=True)
    assert [strategy.name for strategy in engine.strategies] == [s["name"] for s in sorted_enabled]
    description = engine.get_strategy_description()
    assert len(description) > 0


def test_e11_detects_double_bottom_breakout_context():
    strategy = PriceActionBSStrategy("MNQM6", {"tick_size": 0.25, "enabled_setups": ["E-11"]})
    bars = _e11_bars()
    strategy._range_state = {"high": 112.0, "low": 100.0, "width_ticks": 48, "detected_index": 95}
    strategy.swing_lows = [{"index": 100, "price": 100.0}, {"index": 106, "price": 100.25}]
    strategy.swing_highs = [{"index": 102, "price": 104.0}, {"index": 108, "price": 103.0}]

    context = strategy._range_double_test_context(110, bars, "Buy")

    assert context is not None
    assert context["edge"] == 100.0
    assert context["first_test_index"] == 100
    assert context["second_test_index"] == 106
    assert context["range_mid"] == 106.0
    assert context["line_value"] < context["previous_line_value"]


def test_e11_can_use_deterministic_visual_range_fallback():
    strategy = PriceActionBSStrategy("MNQM6", {"tick_size": 0.25, "enabled_setups": ["E-11"]})
    bars = _e11_bars()
    strategy.swing_lows = [{"index": 100, "price": 100.0}, {"index": 106, "price": 100.25}]
    strategy.swing_highs = [{"index": 102, "price": 104.0}, {"index": 108, "price": 103.0}]

    context = strategy._range_double_test_context(110, bars, "Buy")

    assert context is not None
    assert context["range_context"]["source"] == "visual_window"
    assert context["edge"] == 100.0


def test_e11_arms_with_current_entry_logic_and_targets_range_mid():
    strategy = PriceActionBSStrategy("MNQM6", {"tick_size": 0.25, "enabled_setups": ["E-11"]})
    bars = _e11_bars()
    indicators = prepare_indicator_frame(bars, tick_size=0.25)
    strategy._range_state = {"high": 112.0, "low": 100.0, "width_ticks": 48, "detected_index": 95}
    strategy.swing_lows = [{"index": 100, "price": 100.0}, {"index": 106, "price": 100.25}]
    strategy.swing_highs = [{"index": 102, "price": 104.0}, {"index": 108, "price": 103.0}]

    assert strategy._evaluate_range_double_test_setup(110, bars, indicators) is None
    assert len(strategy._pending_candidates) == 1

    candidate = strategy._pending_candidates[0]
    assert candidate["setup_id"] == "E-11"
    assert candidate["side"] == "Buy"
    assert candidate["entry_trigger"] == 103.5
    assert candidate["target_price"] == 106.0

    signal = strategy._evaluate_pending_candidates(111, bars, indicators)

    assert signal is not None
    assert signal.side == "Buy"
    assert signal.tp_ticks == 10
    assert signal.metadata["target_price"] == 106.0
    assert signal.metadata["range_mid_target_price"] == 106.0


def test_visual_target_metadata_does_not_override_r_multiple_target():
    strategy = PriceActionBSStrategy(
        "MNQM6",
        {"tick_size": 0.25, "scalp_target_ticks": 10, "target_mode": "risk_multiple", "risk_reward": 3.0},
    )
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-30 14:00:00", periods=3, freq="1min"),
            "open": [100.0, 100.5, 101.0],
            "high": [100.5, 101.0, 101.5],
            "low": [99.5, 100.0, 100.5],
            "close": [100.25, 100.75, 101.25],
            "volume": [100, 100, 100],
            "price": [100.25, 100.75, 101.25],
        }
    )
    indicators = prepare_indicator_frame(bars, tick_size=0.25)
    signal = strategy._build_signal(
        2,
        bars,
        indicators,
        "E-01",
        "Buy",
        {
            "signal_index": 2,
            "signal_high": 101.5,
            "signal_low": 100.5,
            "entry_trigger": 101.25,
            "target_price": 106.25,
            "range_mid_target_price": 106.25,
        },
    )

    assert signal is not None
    assert signal.sl_ticks == 4
    assert signal.tp_ticks == 12
    assert signal.metadata["target_price"] == 106.25
    assert signal.metadata["execution_target_price"] == 104.25


def test_e11_overlay_draws_observational_break_line():
    strategy = PriceActionBSStrategy("MNQM6", {"tick_size": 0.25, "enabled_setups": ["E-11"]})
    bars = _e11_bars()
    indicators = prepare_indicator_frame(bars, tick_size=0.25)
    strategy._range_state = {"high": 112.0, "low": 100.0, "width_ticks": 48, "detected_index": 95}
    strategy.swing_lows = [{"index": 100, "price": 100.0}, {"index": 106, "price": 100.25}]
    strategy.swing_highs = [{"index": 102, "price": 104.0}, {"index": 108, "price": 103.0}]

    overlays = strategy.get_chart_overlays(110, bars, indicators)

    line_ids = {line["id"] for line in overlays["lines"]}
    assert "e11_buy_break_line" in line_ids


def test_optimizer_candidates_remain_valid_offline_search_space():
    # The offline optimizer search space (config.optimizer.strategy_candidates)
    # is intentionally decoupled from the promoted live stack.
    optimizer_cfg = {
        "strategy_candidates": [
            {
                "direction_filters": {"price_action_bs": ["buy", "sell"]},
                "strategies": [
                    {
                        "name": "price_action_bs",
                        "enabled": True,
                        "priority": 100,
                        "params": {"scalp_target_ticks": 10, "risk_reward": 2.0},
                    }
                ],
            }
        ]
    }
    optimizer = StrategyOptimizer("MNQM6", optimizer_config=optimizer_cfg)
    candidates = optimizer.generate_candidates()

    assert len(candidates) >= 1
    candidate = candidates[0]
    assert "direction_filters" in candidate
    assert "strategies" in candidate
    assert isinstance(candidate["strategies"], list) and candidate["strategies"]


def test_price_action_trade_intent_still_uses_instrument_sizing():
    risk_agent = RiskAgent("TEST")
    risk_agent.register_contract(
        "MNQM6",
        {
            "tickSize": 0.25,
            "tickValue": 0.5,
            "description": "Micro NASDAQ",
            "symbolId": "F.US.MNQ",
        },
    )
    risk_agent.update_account_state(50000, 0, 0)
    router = TradeRouter(
        risk_agent,
        _ExecutionStub(),
        filter_manager=FilterManager(
            {
                "enabled": True,
                "entry_filters": {
                    "market_hours": False,
                    "news_standdown": False,
                    "near_close": False,
                    "cooldown": False,
                    "max_positions": True,
                    "pending_order": False,
                },
            }
        ),
    )

    result = router.route_trade(
        {
            "symbol": "MNQM6",
            "side": "Buy",
            "quantity": 99,
            "sl_ticks": 20,
            "tp_ticks": 10,
            "strategy_name": "price_action_bs",
        },
        {
            "active_count": 0,
            "max_positions": 1,
            "pending_order": False,
            "last_signal_time": 0.0,
            "cooldown_seconds": 0,
            "current_price": 18000.0,
        },
    )

    assert result.accepted is True
    assert result.quantity == risk_agent.calculate_instrument_quantity("MNQM6", sl_ticks=20)


def test_combine_rules_model_topstep_trailing_max_loss():
    results = {
        "summary": {"total_pnl": -501},
        "trades": [
            {"time_close": "2026-04-01T20:00:00Z", "pnl": 1500},
            {"time_close": "2026-04-02T14:00:00Z", "pnl": -2001},
        ],
        "equity_curve": [
            {"timestamp": "2026-04-01T20:00:00Z", "equity": 51500},
            {"timestamp": "2026-04-02T14:00:00Z", "equity": 49499},
        ],
    }

    evaluated = evaluate_combine_rules(
        results,
        {
            "starting_balance": 50000,
            "profit_target": 3000,
            "max_loss_limit": 2000,
            "trailing_max_loss": True,
            "consistency_max_best_day_pct": 50,
        },
    )

    max_loss = evaluated["checks"]["max_loss_limit"]
    assert evaluated["passed"] is False
    assert max_loss["ok"] is False
    assert max_loss["breach"]["mll"] == 49500


def test_combine_consistency_allows_exact_fifty_percent():
    results = {
        "summary": {"total_pnl": 3000},
        "trades": [
            {"time_close": "2026-04-01T20:00:00Z", "pnl": 1500},
            {"time_close": "2026-04-02T20:00:00Z", "pnl": 1500},
        ],
        "equity_history": [50000, 51500, 53000],
    }

    evaluated = evaluate_combine_rules(
        results,
        {
            "starting_balance": 50000,
            "profit_target": 3000,
            "max_loss_limit": 2000,
            "consistency_max_best_day_pct": 50,
        },
    )

    assert evaluated["checks"]["consistency"]["ok"] is True


def test_combine_rules_rejects_50k_micro_position_cap_breach():
    evaluated = evaluate_combine_rules(
        {
            "summary": {"total_pnl": 3200},
            "trades": [
                {
                    "symbol": "MNQM6",
                    "qty": 51,
                    "pnl": 3200,
                    "time_open": "2026-04-01T14:30:00Z",
                    "time_close": "2026-04-01T15:30:00Z",
                }
            ],
            "equity_curve": [
                {"timestamp": "2026-04-01T15:30:00Z", "equity": 53200},
            ],
        },
        {
            "starting_balance": 50000,
            "profit_target": 3000,
            "max_loss_limit": 2000,
            "trailing_max_loss": True,
            "consistency_max_best_day_pct": 100,
            "max_contracts": 5,
            "max_micros": 50,
        },
    )

    assert evaluated["passed"] is False
    assert evaluated["checks"]["position_size"]["ok"] is False
    assert evaluated["checks"]["position_size"]["max_open_micros_seen"] == 51


def test_strategies_override_parameters_from_yaml_correctly():
    stack = _stack_config()
    sss_config = next(s for s in stack["strategies"]["strategies"] if s["name"] == "sss")
    ema_config = next(s for s in stack["strategies"]["strategies"] if s["name"] == "ema_pullback")

    from src.strategies.ema_pullback import EMAPullbackStrategy
    from src.strategies.sss import SSSStrategy

    sss_mnq = SSSStrategy("MNQM6", sss_config["params"])
    sss_mcl = SSSStrategy("MCLM6", sss_config["params"])

    assert sss_mnq.min_sl_ticks == 100

    assert sss_mcl.min_sl_ticks == 40

    ema_mnq = EMAPullbackStrategy("MNQM6", ema_config["params"])
    ema_mcl = EMAPullbackStrategy("MCLM6", ema_config["params"])

    assert ema_mnq.min_sl_ticks == 100

    assert ema_mcl.min_sl_ticks in (30, 40)

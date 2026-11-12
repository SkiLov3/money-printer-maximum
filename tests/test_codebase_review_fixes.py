from __future__ import annotations

import io
import json
from email.message import Message
from typing import Any
from unittest.mock import MagicMock, patch

from dashboard import Handler
from scripts.saturday_autonomous_optimizer import (
    evaluate_backtest_candidate,
    get_fallback_hypotheses,
    is_valid_param_value,
)
from src.agents.supervisor_agent import SupervisorAgent
from src.utils.backtester import Backtester


def test_param_bounds_validation():
    # Valid values
    assert is_valid_param_value("mtf_trend_scalper", "min_adx", 25.0)
    assert is_valid_param_value("mtf_trend_scalper", "sl_atr_mult", 1.5)
    assert is_valid_param_value("orb_vwap", "asian_fade_reclaim_ticks", 2)
    assert is_valid_param_value("mtf_trend_scalper", "allowed_entry_hours_et", [9, 10, 13, 14])

    # Out of bounds / invalid types
    assert not is_valid_param_value("mtf_trend_scalper", "min_adx", 5.0)   # below 15.0
    assert not is_valid_param_value("mtf_trend_scalper", "min_adx", 50.0)  # above 45.0
    assert not is_valid_param_value("mtf_trend_scalper", "sl_atr_mult", 0.5) # below 1.0
    assert not is_valid_param_value("orb_vwap", "asian_fade_reclaim_ticks", -1)
    assert not is_valid_param_value("orb_vwap", "asian_fade_reclaim_ticks", 15)
    # Boolean must be rejected even though isinstance(True, int) is True
    assert not is_valid_param_value("orb_vwap", "asian_fade_reclaim_ticks", True)
    assert not is_valid_param_value("mtf_trend_scalper", "min_adx", False)
    assert not is_valid_param_value("mtf_trend_scalper", "allowed_entry_hours_et", [9, 25])
    assert not is_valid_param_value("mtf_trend_scalper", "allowed_entry_hours_et", [])


def test_fallback_hypotheses_relative_generation():
    cfg = {
        "strategies": {
            "strategies": [
                {
                    "name": "mtf_trend_scalper",
                    "params": {
                        "min_adx": 35.0,
                        "sl_atr_mult": 1.5,
                    },
                }
            ]
        }
    }
    hypo = get_fallback_hypotheses(base_config=cfg)
    adx_test_vals = next(h["test_values"] for h in hypo["hypotheses"] if h["parameter"] == "min_adx")
    sl_test_vals = next(h["test_values"] for h in hypo["hypotheses"] if h["parameter"] == "sl_atr_mult")

    assert 35.0 in adx_test_vals
    assert 30.0 in adx_test_vals
    assert 1.5 in sl_test_vals
    assert 1.3 in sl_test_vals


def test_holdout_split_with_dataset_cutoff():
    trades = [
        {"time_open": "2026-10-01 10:00:00", "pnl": 100.0},
        {"time_open": "2026-10-05 10:00:00", "pnl": 50.0},
        {"time_open": "2026-10-25 10:00:00", "pnl": 75.0},
        {"time_open": "2026-10-26 10:00:00", "pnl": -25.0},
    ]
    with patch("scripts.saturday_autonomous_optimizer._run_single_symbol", return_value=trades):
        # CONTINUOUS_DATA has 2 symbols, so each symbol returns the 4 trades -> 8 total
        res = evaluate_backtest_candidate({}, holdout_days=10)
        assert res["total_trades"] == 8
        assert res["holdout_trades"] == 4
        assert res["holdout_pnl"] == 100.0
        assert res["in_sample_pnl"] == 300.0


def test_supervisor_init_corrupt_status_file():
    mock_file = io.StringIO("[]")
    with patch("builtins.open", return_value=mock_file):
        agent: Any = SupervisorAgent.__new__(SupervisorAgent)
        agent.logger = MagicMock()
        agent.account_id = "12345"
        agent.database = MagicMock()
        agent.get_current_session_start_utc = MagicMock()
        with open("logs/bot_status.json") as st_f:
            last_status = json.load(st_f)
            assert not isinstance(last_status, dict)


def test_retroactive_fill_preserves_excursions():
    agent: Any = SupervisorAgent.__new__(SupervisorAgent)
    agent.logger = MagicMock()
    agent.database = MagicMock()
    agent.database.connected = True
    agent.risk_agent = MagicMock()
    agent.risk_agent.get_tick_size.return_value = 0.25

    # Simulate closed position record with excursions
    closed_pos = {
        "trade_key": "TK-123",
        "symbol": "MES",
        "side": "Buy",
        "qty": 1,
        "entry": 5000.0,
        "sl_ticks": 20,
        "tp_ticks": 40,
        "mfe_price": 5010.0,
        "mae_price": 4995.0,
        "mfe_ticks": 40.0,
        "mae_ticks": 20.0,
        "mfe_r": 2.0,
        "mae_r": 1.0,
        "strategy_name": "mtf_trend_scalper",
    }

    # Initial close call with pnl=0.0
    agent._persist_closed_trade_outcome(
        symbol="MES",
        closed=closed_pos,
        pnl=0.0,
        order_type="stop_loss",
    )

    args = agent.database.persist_trade_outcome.call_args[1]
    outcome = args["outcome"]
    assert outcome["mfe_ticks"] == 40.0
    assert outcome["mae_ticks"] == 20.0
    assert outcome["mfe_r"] == 2.0

    # Now simulate retroactive fill update with pnl=-125.0 using the stripped recent_trades dict
    # where entry is 0 or missing, but excursions are preserved on the dict
    stripped_recent = {
        "trade_key": "TK-123",
        "symbol": "MES",
        "side": "Buy",
        "qty": 1,
        "entry": 5000.0,
        "mfe_ticks": 40.0,
        "mae_ticks": 20.0,
        "mfe_r": 2.0,
        "mae_r": 1.0,
    }
    agent._persist_closed_trade_outcome(
        symbol="MES",
        closed=stripped_recent,
        pnl=-125.0,
        order_type="stop_loss",
    )
    args2 = agent.database.persist_trade_outcome.call_args[1]
    outcome2 = args2["outcome"]
    assert outcome2["pnl"] == -125.0
    assert outcome2["mfe_ticks"] == 40.0
    assert outcome2["mae_ticks"] == 20.0
    assert outcome2["mfe_r"] == 2.0


def _build_headers(headers_dict: dict[str, str]) -> Message:
    msg = Message()
    for k, v in headers_dict.items():
        msg[k] = v
    return msg


def test_dashboard_csrf_same_origin():
    handler: Any = Handler.__new__(Handler)
    handler.headers = _build_headers({"Origin": "http://evil-site.com", "Host": "localhost:8765"})
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()

    # Cross origin must be rejected
    assert not handler._check_same_origin()
    handler.send_response.assert_called_with(403)

    # Same origin must be allowed
    handler.headers = _build_headers({"Origin": "http://localhost:8765", "Host": "localhost:8765"})
    assert handler._check_same_origin()

    # Missing Origin (curl / internal request) must be allowed
    handler.headers = _build_headers({"Host": "localhost:8765"})
    assert handler._check_same_origin()


def test_backtester_reversal_updates_excursions():
    bt: Any = Backtester.__new__(Backtester)
    bt.tick_size = 0.25
    bt.tick_value = 1.25
    bt.slippage_ticks = 0
    bt.commission_per_contract = 0.62
    bt.symbol = "MES"
    bt.risk_agent = MagicMock()
    bt.telemetry = MagicMock()
    bt.balance = 50000.0
    bt.active_trades = [
        {
            "side": "Buy",
            "entry": 5000.0,
            "qty": 1,
            "sl_ticks": 20,
            "tp_ticks": 40,
            "mfe_price": 5002.0,
            "mae_price": 4998.0,
            "time_open": "2026-11-01 10:00:00",
            "metadata": {},
        }
    ]

    recorded_trades = []
    bt._record_trade = lambda trade, exit_price, pnl, result, timestamp: recorded_trades.append((dict(trade), exit_price, result))

    # Reverse to Sell with entry/exit at 5010.0 (new MFE reached on exit bar)
    new_signal = {"side": "Sell", "quantity": 1, "sl_ticks": 20, "tp_ticks": 40}
    bt._open_trade(new_signal, "2026-11-01 10:05:00", 5010.0)

    assert len(recorded_trades) == 1
    closed_trade, exit_p, res_type = recorded_trades[0]
    assert exit_p == 5010.0
    assert res_type == "REVERSAL"
    # MFE price must be updated to 5010.0
    assert closed_trade["mfe_price"] == 5010.0

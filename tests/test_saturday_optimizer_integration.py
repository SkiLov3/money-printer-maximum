"""Integration tests for SupervisorAgent Saturday Autonomous Optimizer integration.

Verifies:
- Supervisor triggers Saturday optimization at/after 08:00 ET when flat
- Supervisor skips if market is open, not Saturday, before 08:00, or positions are open
- Supervisor reload_configuration reloads stack and strategy agents cleanly
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from src.agents.supervisor_agent import SupervisorAgent


def create_mock_supervisor():
    with patch("src.agents.supervisor_agent.ProjectXClient"), \
         patch("src.agents.supervisor_agent.RiskAgent"), \
         patch("src.agents.supervisor_agent.ExecutionAgent"), \
         patch("src.agents.supervisor_agent.NewsAgent"), \
         patch("src.agents.supervisor_agent.TelemetryStore"), \
         patch("src.agents.supervisor_agent.DatabaseService"), \
         patch("src.agents.supervisor_agent.DiscoveryService"), \
         patch("src.agents.supervisor_agent.StrategyHealthMonitor"), \
         patch("src.agents.supervisor_agent.FilterManager"), \
         patch("src.agents.supervisor_agent.TradeRouter"):
        sup = SupervisorAgent(symbols=["MES", "MCL"], account_id="12345")
        return sup


@pytest.mark.anyio
async def test_check_saturday_optimization_triggers_on_saturday_morning():
    mock_supervisor = create_mock_supervisor()
    mock_supervisor.loop = asyncio.get_running_loop()

    # Simulate Saturday 09:30 ET
    sat_dt = datetime(2026, 10, 17, 9, 30, tzinfo=pytz.timezone("US/Eastern"))

    with patch("src.agents.supervisor_agent.datetime") as mock_dt, \
         patch.object(mock_supervisor, "_run_saturday_optimizer_async", new_callable=AsyncMock):

        mock_now = MagicMock()
        mock_now.astimezone.return_value = sat_dt
        mock_now.strftime.return_value = "2026-10-17"
        mock_now.weekday.return_value = 5
        mock_now.hour = 9
        mock_dt.now.return_value = mock_now

        mock_supervisor.active_positions = {}
        mock_supervisor._last_saturday_optimizer_run = None

        await mock_supervisor._check_saturday_optimization()

        assert mock_supervisor._last_saturday_optimizer_run == "2026-10-17"
        assert mock_supervisor._optimizer_task is not None


@pytest.mark.anyio
async def test_check_saturday_optimization_skips_when_positions_open():
    mock_supervisor = create_mock_supervisor()
    mock_supervisor.loop = asyncio.get_running_loop()

    sat_dt = datetime(2026, 10, 17, 9, 30, tzinfo=pytz.timezone("US/Eastern"))

    with patch("src.agents.supervisor_agent.datetime") as mock_dt:
        mock_now = MagicMock()
        mock_now.astimezone.return_value = sat_dt
        mock_now.strftime.return_value = "2026-10-17"
        mock_now.weekday.return_value = 5
        mock_now.hour = 9
        mock_dt.now.return_value = mock_now

        mock_supervisor.active_positions = {"MES": [{"side": "Buy", "qty": 1}]}
        mock_supervisor._last_saturday_optimizer_run = None

        await mock_supervisor._check_saturday_optimization()

        assert mock_supervisor._last_saturday_optimizer_run is None
        assert mock_supervisor._optimizer_task is None


@pytest.mark.anyio
async def test_check_saturday_optimization_skips_on_weekdays():
    mock_supervisor = create_mock_supervisor()
    mock_supervisor.loop = asyncio.get_running_loop()

    # Tuesday 09:30 ET
    tue_dt = datetime(2026, 10, 13, 9, 30, tzinfo=pytz.timezone("US/Eastern"))

    with patch("src.agents.supervisor_agent.datetime") as mock_dt:
        mock_now = MagicMock()
        mock_now.astimezone.return_value = tue_dt
        mock_now.strftime.return_value = "2026-10-13"
        mock_now.weekday.return_value = 1  # Tuesday
        mock_now.hour = 9
        mock_dt.now.return_value = mock_now

        mock_supervisor.active_positions = {}
        mock_supervisor._last_saturday_optimizer_run = None

        await mock_supervisor._check_saturday_optimization()

        assert mock_supervisor._last_saturday_optimizer_run is None


def test_reload_configuration():
    import pandas as pd

    mock_supervisor = create_mock_supervisor()
    with patch("src.agents.supervisor_agent.load_stack_config") as mock_load:
        mock_load.return_value = {"strategies": {"strategies": []}}
        mock_supervisor.official_symbols = ["MES"]
        mock_supervisor.strategy_agents = {"MES": MagicMock()}  # type: ignore
        fake_da = MagicMock()
        fake_da.get_latest_data.return_value = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-10-20", periods=5, freq="1min"),
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [10.0] * 5,
                "price": [100.5] * 5,
            }
        )
        mock_supervisor.data_agents = {"MES": fake_da}  # type: ignore

        assert mock_supervisor.reload_configuration() is True
        assert mock_supervisor.stack_config == {"strategies": {"strategies": []}}
        assert getattr(mock_supervisor.strategy_agents["MES"].engine, "_last_processed_timestamp", None) is not None


@pytest.mark.anyio
async def test_run_saturday_optimizer_async_resets_run_flag_on_failure():
    mock_supervisor = create_mock_supervisor()
    mock_supervisor._last_saturday_optimizer_run = "2026-10-17"

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"Error occurred"))
    mock_proc.returncode = 1

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        await mock_supervisor._run_saturday_optimizer_async("2026-10-17")
        assert mock_supervisor._last_saturday_optimizer_run is None



def test_supervisor_agent_imports_sys():
    import src.agents.supervisor_agent as sa
    assert hasattr(sa, "sys")
    assert sa.sys.executable


def test_optimizer_retry_backoff_and_dry_run_guard():
    import inspect

    from src.agents.supervisor_agent import SupervisorAgent

    src = inspect.getsource(SupervisorAgent._check_saturday_optimization)
    assert "_optimizer_failures >= 3" in src and "_optimizer_next_retry_at" in src
    run_src = inspect.getsource(SupervisorAgent._run_saturday_optimizer_async)
    assert 'state_data.get("dry_run")' in run_src
    assert "_optimizer_next_retry_at = time.time() + 1800" in run_src


def test_optimizer_run_id_uses_eastern_date():
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "scripts" / "saturday_autonomous_optimizer.py"
    assert 'ZoneInfo("America/New_York")' in p.read_text()


def test_candidate_param_validation_rejects_fake_params():
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "scripts" / "saturday_autonomous_optimizer.py"
    spec = importlib.util.spec_from_file_location("sat_opt", p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert not m.is_valid_candidate_param("orb_vwap", "asian_volume_ratio_min")
    assert not m.is_valid_candidate_param("orb_vwap", "sl_atr_mult")
    assert m.is_valid_candidate_param("mtf_trend_scalper", "min_adx")
    assert m.is_valid_candidate_param("orb_vwap", "min_volume_ratio")


def test_completed_state_from_prior_day_starts_fresh(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "scripts" / "saturday_autonomous_optimizer.py"
    spec = importlib.util.spec_from_file_location("sat_opt2", p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    f = tmp_path / "state.json"
    f.write_text(json.dumps({"run_id": "2000-01-01", "phase": "COMPLETED"}))
    assert m.StateManager(f).state["phase"] == "INITIAL"


def test_apply_param_override_mutates_config_in_place():
    import importlib.util
    from pathlib import Path

    from src.config.settings import get_strategy_config, load_stack_config

    p = Path(__file__).resolve().parent.parent / "scripts" / "saturday_autonomous_optimizer.py"
    spec = importlib.util.spec_from_file_location("sat_opt3", p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = load_stack_config("config/trading_stack.example.yaml")
    assert m.apply_param_override(cfg, "mtf_trend_scalper", "min_adx", 60.0)
    mtf = [s for s in get_strategy_config(cfg)["strategies"] if s["name"] == "mtf_trend_scalper"][0]
    assert mtf["params"]["min_adx"] == 60.0
    assert not m.apply_param_override(cfg, "nonexistent", "x", 1)

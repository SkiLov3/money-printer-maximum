from __future__ import annotations

import re
from types import SimpleNamespace

import pandas as pd
import pytest

from src.agents.data_agent import DataAgent
from src.agents.risk_agent import RiskAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.charting import bars_to_chart_records, resample_timeframe
from src.strategies.indicators import prepare_indicator_frame
from src.strategies.price_action_bs import PriceActionBSStrategy


def test_dashboard_chart_refresh_preserves_manual_zoom():
    from dashboard import HTML

    assert 'onclick="fitChartContent()"' in HTML
    assert 'onclick="jumpChartLatest()"' in HTML
    assert 'id="chart-zoom"' in HTML
    assert 'onclick="stepChartZoom(-1)"' in HTML
    assert 'onclick="stepChartZoom(1)"' in HTML
    assert "setChartZoomPercent(this.value)" in HTML
    assert "chartZoomPercents" in HTML
    assert "const chartFrames = ['1m','5m','15m','30m','1h','4h','1d']" in HTML
    assert "let activeChartFrame = '1m'" in HTML
    assert "'1000t'" not in HTML
    assert "'2000t'" not in HTML
    assert '<option value="2000">2000%</option>' in HTML
    assert "minBarSpacing: 0.1" in HTML
    assert "chartUserInteractingUntil" in HTML
    assert "Date.now() < chartUserInteractingUntil" in HTML
    assert "function formatChartTimestamp" in HTML
    assert "function diagnosticRows" in HTML
    assert "Swing High" in HTML
    assert "Swing Low" in HTML
    assert "event-time" in HTML
    assert "Last</span><span style=\"text-align:right\">P&L" in HTML
    assert "fmtPx(p.last_price)" in HTML
    assert "fmt$(p.open_pnl)" in HTML
    assert "setInterval(refresh, 1000)" in HTML
    assert "/api/chart/live" in HTML
    assert "candleSeries.update(bar)" in HTML
    assert "setInterval(() => refreshLiveChart(), 750)" in HTML
    assert "getVisibleLogicalRange()" in HTML
    assert "setVisibleLogicalRange(previousRange)" in HTML
    assert "scrollToRealTime" in HTML

    live_body = re.search(
        r"async function refreshLiveChart\(\) \{(?P<body>.*?)\n\}\n\nfunction renderChartDiagnostics",
        HTML,
        re.S,
    )
    assert live_body is not None
    assert "candleSeries.update(bar)" in live_body.group("body")
    assert "setVisibleLogicalRange" not in live_body.group("body")
    assert "scrollToRealTime" not in live_body.group("body")
    assert "applyPriceAutoscale" not in live_body.group("body")


def _bars(count=240):
    ts = pd.date_range("2026-04-30 14:00:00", periods=count, freq="1min")
    base = pd.Series(range(count), dtype="float64") * 0.25 + 100.0
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.25,
            "volume": [100] * count,
            "price": base + 0.25,
        }
    )


def test_resample_and_chart_records_are_lightweight_chart_ready():
    frame = resample_timeframe(_bars(60), "15m")
    records = bars_to_chart_records(frame, limit=10)

    assert len(records) == 4
    assert {"time", "open", "high", "low", "close", "volume"} <= set(records[0])
    assert isinstance(records[0]["time"], int)


def test_price_action_chart_overlays_include_visual_state():
    strategy = PriceActionBSStrategy("MNQM6", {"tick_size": 0.25})
    bars = _bars(120)
    indicators = prepare_indicator_frame(bars, tick_size=0.25)
    strategy.swing_highs = [{"index": 20, "price": 107.0}, {"index": 80, "price": 112.0}]
    strategy.swing_lows = [{"index": 25, "price": 101.0}, {"index": 90, "price": 106.0}]
    strategy._range_state = {"high": 112.0, "low": 106.0, "width_ticks": 24, "detected_index": 90}
    strategy._pending_candidates = [
        {
            "key": ("E-01", "Buy", 100),
            "setup_id": "E-01",
            "side": "Buy",
            "signal_index": 100,
            "entry_trigger": 110.25,
            "failure_trigger": 108.75,
        }
    ]
    strategy._append_chart_event(
        key=("test", 1),
        index=110,
        bars=bars,
        kind="failed",
        label="Failure bar invalid",
        price=109.0,
        setup_id="E-02",
    )

    overlays = strategy.get_chart_overlays(len(bars) - 1, bars, indicators)

    line_ids = {line["id"] for line in overlays["lines"]}
    level_ids = {level["id"] for level in overlays["levels"]}
    assert any(line["id"] == "ema21" for line in overlays["lines"])
    assert "swing_high_trendline" in line_ids
    assert "swing_low_trendline" in line_ids
    assert "visual_channel_upper" in line_ids
    assert "visual_channel_lower" in line_ids
    assert "range_high" in level_ids
    assert "range_mid" in level_ids
    assert any(level["id"].startswith("pending_entry") for level in overlays["levels"])
    assert len(next(line for line in overlays["lines"] if line["id"] == "swing_high_trendline")["points"]) >= 3
    assert any(marker["kind"] == "swing_high" for marker in overlays["markers"])
    assert overlays["events"][0]["kind"] == "failed"


def test_price_action_chart_overlays_emit_visual_range_without_validated_range():
    strategy = PriceActionBSStrategy("MNQM6", {"tick_size": 0.25})
    bars = _bars(160)
    indicators = prepare_indicator_frame(bars, tick_size=0.25)
    strategy.swing_highs = [{"index": 80, "price": 122.0}, {"index": 130, "price": 128.0}]
    strategy.swing_lows = [{"index": 85, "price": 110.0}, {"index": 135, "price": 116.0}]

    overlays = strategy.get_chart_overlays(len(bars) - 1, bars, indicators)

    level_ids = {level["id"] for level in overlays["levels"]}
    assert {"visual_range_high", "visual_range_mid", "visual_range_low"} <= level_ids
    assert any(line["id"] == "visual_channel_upper" for line in overlays["lines"])


def test_supervisor_chart_payload_is_observational_and_serializable():
    data_agent = DataAgent("MNQM6", max_bars=500)
    data_agent.load_historical_bars(_bars(240).to_dict(orient="records"))
    strategy_agent = StrategyAgent("MNQM6")
    risk_agent = RiskAgent("TEST")
    risk_agent.register_contract("MNQM6", {"tickSize": 0.25, "tickValue": 0.5, "description": "Micro NASDAQ"})

    supervisor = object.__new__(SupervisorAgent)
    supervisor.official_symbols = ["MNQM6"]
    supervisor.data_agents = {"MNQM6": data_agent}
    supervisor.strategy_agents = {"MNQM6": strategy_agent}
    supervisor.active_positions = {}
    supervisor._chart_trade_events = []
    supervisor.risk_agent = risk_agent

    payload = SupervisorAgent.get_chart_payload(supervisor, symbol="MNQM6", frame="5m", limit=100)

    assert payload["symbol"] == "MNQM6"
    assert payload["frame"] == "5m"
    assert payload["frames"] == ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    assert payload["bars"]
    assert {"lines", "levels", "markers", "events", "state"} <= set(payload["overlays"])

    with pytest.raises(ValueError, match="Unsupported chart frame"):
        SupervisorAgent.get_chart_payload(supervisor, symbol="MNQM6", frame="1000t", limit=100)


def test_supervisor_live_chart_payload_returns_latest_forming_bar():
    data_agent = DataAgent("MNQM6", max_bars=500)
    data_agent.load_historical_bars(_bars(3).to_dict(orient="records"))
    data_agent.handle_market_data({"symbol": "MNQM6", "price": 101.5, "volume": 2})

    supervisor = object.__new__(SupervisorAgent)
    supervisor.official_symbols = ["MNQM6"]
    supervisor.data_agents = {"MNQM6": data_agent}
    supervisor.client = SimpleNamespace(last_tick_at=123.0, is_connected=True)  # type: ignore[assignment]

    payload = SupervisorAgent.get_chart_live_payload(supervisor, symbol="MNQM6", frame="1m")

    assert payload["symbol"] == "MNQM6"
    assert payload["frame"] == "1m"
    assert payload["bar"]["close"] == 101.5
    assert payload["last_price"] == 101.5
    assert payload["signalr_connected"] is True


def test_supervisor_position_status_entry_computes_live_open_pnl():
    supervisor = object.__new__(SupervisorAgent)
    supervisor.risk_agent = SimpleNamespace(  # type: ignore[assignment]
        contract_meta={"MCLM6": {"tickSize": 0.01, "tickValue": 1.0}}
    )

    item = SupervisorAgent._position_status_entry(
        supervisor,
        "MCLM6",
        side="Buy",
        qty=32,
        entry=105.48,
        time_text="(broker)",
        latest_prices={"MCLM6": 105.51},
    )

    assert item["last_price"] == 105.51
    assert item["open_ticks"] == 3
    assert item["open_pnl"] == 96.0


def test_dashboard_saturday_optimizer_widget_and_pruned_elements():
    from dashboard import HTML

    # Saturday Autonomous Optimizer Widget Elements
    assert 'id="sat-body"' in HTML
    assert 'id="sat-phase-pill"' in HTML
    assert 'id="sat-last-run"' in HTML
    assert 'id="sat-run-btn"' in HTML
    assert 'id="sat-body-tab"' in HTML
    assert 'id="sat-phase-pill-tab"' in HTML
    assert 'id="sat-last-run-tab"' in HTML
    assert 'id="sat-run-btn-tab"' in HTML
    assert 'id="optimizer-tab"' in HTML
    assert 'id="optimizer-view"' in HTML
    assert 'id="badge-sat-opt"' in HTML
    from src.version import VERSION
    assert f"v{VERSION}" in HTML
    assert "function renderSaturdayOptimizer" in HTML
    assert "function triggerSaturdayOptimizer" in HTML
    assert "/api/saturday-optimizer/run" in HTML
    assert "Saturday Autonomous Optimizer (Qwen 3.5 122B)" in HTML
    assert "getNextSaturdayET" in HTML
    assert "formatLastRunET" in HTML
    assert "LAST:" in HTML and "NEXT:" in HTML
    assert "latestStatus?.active_positions" in HTML

    # Verified Pruned Elements
    assert 'id="mega-body"' not in HTML
    assert 'id="zone-count"' not in HTML
    assert 'id="discovery-body"' not in HTML
    assert 'id="validation-body"' not in HTML
    assert 'id="meta-body"' not in HTML
    assert 'id="meta-settings"' not in HTML
    assert 'id="health-body"' not in HTML
    assert 'id="subsystems-body"' not in HTML
    assert 'id="risk-dll"' not in HTML
    assert 'id="pnl-chart"' not in HTML
    assert 'id="session-trades-body"' not in HTML
    assert 'id="recent-trades-body"' not in HTML

    # Modernized Institutional Elements
    assert 'id="trades-body"' in HTML
    assert 'id="trade-count"' in HTML
    assert 'id="positions-body"' in HTML
    assert 'id="strategy-body"' in HTML


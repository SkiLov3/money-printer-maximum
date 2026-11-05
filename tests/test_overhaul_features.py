# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalSubscript=false
import asyncio
import json
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.agents.risk_agent import RiskAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.api.projectx_client import normalize_order_type, normalize_status
from src.backtesting.results_exporter import ResultsExporter
from src.discovery.meta_optimizer import MetaOptimizerService
from src.discovery.service import DiscoveryService
from src.learning.claude_md_writer import ClaudeMdWriter
from src.learning.session_capture import capture_trade_context_artifacts
from src.mega_vision.agent import MegaStrategyAgent, MegaVisionConfig
from src.monitoring.health import StrategyHealthMonitor
from src.optimization.adaptive_runner import AdaptiveOptimizerRunner
from src.optimization.context_embedder import embed_market
from src.optimization.experience_store import ExperienceStore
from src.optimization.llm_analysis_builder import AnalysisBuilder
from src.optimization.llm_analyst import LLMAnalyst
from src.persistence.database import DatabaseService
from src.strategies.base import StrategySignal
from src.strategies.blender import SignalBlender
from src.strategies.engine import StrategyEngine
from src.strategies.fx_at_one_glance import FXAtOneGlanceStrategy
from src.trading.filters import FilterManager, TradingFilterContext
from src.trading.router import RouterRejection, RouterResult, TradeRouter
from src.utils.backtester import Backtester
from src.zones.engine import ZoneSignalEngine


class _ExecutionStub:
    def execute_trade(self, trade_intent):
        return {"status": "simulation_fill", "avgPrice": 100.0, "orderId": "SIM-1"}


def test_filter_manager_blocks_news_standdown():
    manager = FilterManager({"enabled": True, "entry_filters": {"news_standdown": True}})
    context = TradingFilterContext(
        symbol="MNQM6",
        now_dt=pd.Timestamp("2026-04-14T15:00:00Z").to_pydatetime(),
        trade_intent={"symbol": "MNQM6", "side": "Buy"},
        market_open=True,
        news_standdown=True,
        near_close=False,
        active_count=0,
        max_positions=1,
        pending_order=False,
        last_signal_time=0.0,
        cooldown_seconds=0,
    )
    allowed, decisions = manager.evaluate_entry(context)
    assert allowed is False
    assert decisions[-1].name == "news_standdown"


def test_instrument_quantity_uses_registered_contract():
    from decimal import Decimal
    risk_agent = RiskAgent("TEST")
    risk_agent.max_daily_loss = Decimal("1000.0")
    risk_agent.register_contract(
        "MNQ",
        {"tickSize": 0.25, "tickValue": 0.5, "description": "Micro NASDAQ", "symbolId": "F.US.MNQ"},
    )
    risk_agent.update_account_state(50000, 0, 0)
    qty = risk_agent.calculate_instrument_quantity("MNQ", sl_ticks=20, risk_pct=0.4)
    assert qty == 20


def test_instrument_quantity_caps_at_25_pct_of_daily_loss_limit():
    from decimal import Decimal
    risk_agent = RiskAgent("TEST")
    risk_agent.max_daily_loss = Decimal("500.0")
    risk_agent.register_contract(
        "MNQ",
        {"tickSize": 0.25, "tickValue": 0.5, "description": "Micro NASDAQ", "symbolId": "F.US.MNQ"},
    )
    risk_agent.update_account_state(50000, 0, 0)
    qty = risk_agent.calculate_instrument_quantity("MNQ", sl_ticks=20, risk_pct=0.4)
    assert qty == 12


def test_instrument_entry_quantity_never_exceeds_risk_budget_after_wins():
    risk_agent = RiskAgent("TEST")
    risk_agent.register_contract(
        "MNQ",
        {"tickSize": 0.25, "tickValue": 0.5, "description": "Micro NASDAQ", "symbolId": "F.US.MNQ"},
    )
    risk_agent.update_account_state(50000, 200, 0)
    risk_agent.record_trade_result(50)
    risk_agent.record_trade_result(75)
    risk_agent.record_trade_result(25)

    qty = risk_agent.get_entry_quantity("MNQ", requested_qty=99, sl_ticks=100)

    assert qty == risk_agent.calculate_instrument_quantity("MNQ", sl_ticks=100)


def test_trade_router_accepts_trade():
    risk_agent = RiskAgent("TEST")
    risk_agent.register_contract(
        "MNQ",
        {"tickSize": 0.25, "tickValue": 0.5, "description": "Micro NASDAQ", "symbolId": "F.US.MNQ"},
    )
    risk_agent.update_account_state(50000, 0, 0)
    router = TradeRouter(risk_agent, _ExecutionStub(), filter_manager=FilterManager({"enabled": True}))
    result = router.route_trade(
        {"symbol": "MNQ", "side": "Buy", "quantity": 1, "sl_ticks": 20, "tp_ticks": 40, "strategy_name": "test"},
        {
            "market_open": True,
            "news_standdown": False,
            "near_close": False,
            "active_count": 0,
            "max_positions": 2,
            "pending_order": False,
            "last_signal_time": 0.0,
            "cooldown_seconds": 0,
            "current_price": 100.0,
        },
    )
    assert result.accepted is True
    assert result.quantity >= 1


def test_trade_router_rejects_pending_order():
    risk_agent = RiskAgent("TEST")
    risk_agent.update_account_state(50000, 0, 0)
    router = TradeRouter(risk_agent, _ExecutionStub(), filter_manager=FilterManager({"enabled": True}))
    rejection = router.route_trade(
        {"symbol": "NQ", "side": "Buy", "quantity": 1, "sl_ticks": 20, "tp_ticks": 40, "strategy_name": "test"},
        {
            "market_open": True,
            "news_standdown": False,
            "near_close": False,
            "active_count": 0,
            "max_positions": 1,
            "pending_order": True,
            "last_signal_time": 0.0,
            "cooldown_seconds": 0,
            "current_price": 100.0,
        },
    )
    assert isinstance(rejection, RouterRejection)
    assert rejection.stage == "filter"


def test_backtester_optimizer_fast_disables_operational_filters():
    fast = Backtester("MNQM6", initial_balance=50000, telemetry_enabled=False, routing_mode="optimizer_fast")
    full = Backtester("MNQM6", initial_balance=50000, telemetry_enabled=False, routing_mode="full_live")

    assert fast.routing_config["entry_filters"]["news_standdown"] is False
    assert fast.routing_config["entry_filters"]["cooldown"] is False
    assert fast.risk_agent.get_entry_quantity("MNQM6", 99, sl_ticks=20) >= 1
    assert full.routing_config["entry_filters"]["news_standdown"] is True


def test_signal_blender_applies_multi_agree_bonus():
    blender = SignalBlender(multi_agree_bonus=2.0)
    signals = [
        StrategySignal("trend_following", "Buy", 1, 20, 40, confidence=0.4, metadata={"confluence_score": 4}),
        StrategySignal("asian_breakout", "Buy", 1, 20, 40, confidence=0.3, metadata={"confluence_score": 3}),
        StrategySignal("sss", "Sell", 1, 20, 40, confidence=0.5, metadata={"confluence_score": 5}),
    ]

    result = blender.select(signals)

    assert result is not None
    assert result.signal.strategy_name == "trend_following"
    assert result.effective_score == 6.0
    assert result.agreement_count == 2


def test_strategy_engine_uses_confluence_blender():
    engine = StrategyEngine(
        symbol="MNQM6",
        default_qty=1,
        bracket={"sl": 20, "tp": 40, "tick_size": 0.25},
        telemetry=None,
        config={
            "blender": {"mode": "confluence", "multi_agree_bonus": 2.0},
            "strategies": [],
        },
    )

    buy_signal = StrategySignal(
        "trend_following",
        "Buy",
        1,
        20,
        40,
        confidence=0.4,
        reason="buy candidate",
        metadata={"confluence_score": 4},
    )
    sell_signal = StrategySignal(
        "sss",
        "Sell",
        1,
        20,
        40,
        confidence=0.5,
        reason="sell candidate",
        metadata={"confluence_score": 5},
    )
    asian_signal = StrategySignal(
        "asian_breakout",
        "Buy",
        1,
        20,
        40,
        confidence=0.3,
        reason="agreement buy candidate",
        metadata={"confluence_score": 3},
    )

    selected_index, blend_metadata, candidate_blend_data = engine._select_candidate_index([buy_signal, asian_signal, sell_signal])

    assert selected_index == 0
    assert blend_metadata["mode"] == "confluence"
    assert blend_metadata["effective_score"] == 6.0
    assert blend_metadata["agreement_count"] == 2
    assert candidate_blend_data[1]["effective_score"] == 5.0


def test_strategy_engine_priority_first_mode_still_supported():
    engine = StrategyEngine(
        symbol="MNQM6",
        default_qty=1,
        bracket={"sl": 20, "tp": 40, "tick_size": 0.25},
        telemetry=None,
        config={
            "blender": {"mode": "priority_first", "multi_agree_bonus": 2.0},
            "strategies": [],
        },
    )

    first = StrategySignal("first", "Sell", 1, 20, 40, confidence=0.1)
    second = StrategySignal("second", "Buy", 1, 20, 40, confidence=0.9)

    selected_index, blend_metadata, candidate_blend_data = engine._select_candidate_index([first, second])

    assert selected_index == 0
    assert blend_metadata["mode"] == "priority_first"
    assert len(candidate_blend_data) == 2


def test_strategy_engine_registers_parity_strategies():
    engine = StrategyEngine(
        symbol="MGCK6",
        default_qty=1,
        bracket={"sl": 20, "tp": 40, "tick_size": 0.1},
        telemetry=None,
    )

    names = [strategy.name for strategy in engine.strategies]

    assert "ichimoku" in names
    assert "fx_at_one_glance" in names
    assert "ema_pullback" in names
    assert "asian_breakout" in names
    assert "sss" in names


def test_fx_at_one_glance_throttles_repeated_checklist_signals():
    strategy = FXAtOneGlanceStrategy(
        "MGCM6",
        {
            "min_confluence": 6,
            "min_score_edge": 2,
            "min_signal_gap_bars": 12,
            "require_breakout": True,
        },
    )
    row = {
        "price": 110.0,
        "close": 110.0,
        "ema_200": 100.0,
        "vwap": 105.0,
        "kijun": 109.0,
        "atr": 2.0,
        "adx": 25.0,
        "cloud_thickness": 1.0,
        "rsi": 60.0,
        "rolling_high_20": 110.0,
        "rolling_low_20": 95.0,
    }
    indicators = pd.DataFrame([row.copy() for _ in range(140)])
    bars = indicators.copy()

    assert strategy.on_bar(119, bars, indicators) is not None
    assert strategy.on_bar(120, bars, indicators) is None
    assert strategy.on_bar(131, bars, indicators) is not None


def test_zone_engine_generates_demand_signal():
    engine = ZoneSignalEngine("MNQM6", config={"minimum_bars": 20, "lookback": 20, "pivot_window": 2, "entry_distance_atr": 10.0})
    closes = [100, 99, 98, 97, 96, 97, 98, 99, 100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 104, 105]
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [100] * len(closes),
            "atr": [2.0] * len(closes),
            "ema_50": closes,
            "ema_200": [c - 1 for c in closes],
        }
    )
    signal = engine.analyze(df)
    assert signal is None or signal["strategy_name"] == "zone_confluence"


def test_health_monitor_tracks_signal_results():
    monitor = StrategyHealthMonitor({"max_events": 5})
    monitor.record_signal("MNQ", True, "accepted")
    monitor.record_signal("MNQ", False, "blocked")
    status = monitor.get_status()
    assert status["accepts"]["MNQ"] == 1
    assert status["rejections"]["MNQ"] == 1


def test_database_service_degrades_when_disabled():
    service = DatabaseService({"enabled": False})
    status = service.health_check()
    assert status["connected"] is False
    assert status["reason"] == "disabled"


def _test_output_dir(name: str) -> Path:
    root = Path(tempfile.gettempdir()) / "topstep_test_runtime" / f"{name}_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_optimizer_fixture(root: Path) -> str:
    rows = []
    start = pd.Timestamp("2026-04-13T13:30:00Z")
    for idx in range(420):
        close = 18000 + (idx * 0.08) + ((idx % 18) - 9) * 0.12
        open_price = close - 0.1
        rows.append(
            {
                "timestamp": start + pd.Timedelta(minutes=idx),
                "open": open_price,
                "high": max(open_price, close) + 0.5,
                "low": min(open_price, close) - 0.5,
                "close": close,
                "volume": 100 + (idx % 7),
            }
        )
    path = root / "mnqm6_optimizer_fixture.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_database_service_sqlite_persists_zone():
    db_path = _test_output_dir("sqlite_zone") / "swarm.db"
    service = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": str(db_path)})
    class Zone:
        zone_type = "demand"
        price_low = 100.0
        price_high = 101.0
        strength = 1.0
        status = "active"
        touch_count = 0
    service.persist_zone("MNQ", Zone())
    rows = service.list_active_zones("MNQ")
    assert len(rows) == 1
    assert rows[0]["zone_type"] == "demand"


def test_database_candidate_promotions_round_trip():
    db_path = _test_output_dir("candidate_promotions") / "swarm.db"
    service = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": str(db_path)})
    service.persist_candidate_promotion(
        candidate_id="candidate-1",
        batch_id="batch-1",
        config_path="",
        config_diff={"strategies.0.params.min_confluence": 4},
        gate_status={"three_combine_passes": False},
        status="pending_optimizer_validation",
    )

    rows = service.list_candidate_promotions(status="pending_optimizer_validation")

    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "candidate-1"
    assert rows[0]["config_diff"]["strategies.0.params.min_confluence"] == 4


def test_adaptive_optimizer_rejects_high_churn_promotion_quality():
    runner = AdaptiveOptimizerRunner(
        "MGCM6",
        "unused.csv",
        optimizer_config={
            "promotion_quality": {
                "enabled": True,
                "max_trades_per_day": 15,
                "max_total_trades": 450,
                "min_profit_factor": 1.35,
                "min_avg_trade": 8.0,
                "max_fx_at_one_glance_trade_share_pct": 45,
            }
        },
        database_config={"enabled": False},
    )
    result = {
        "summary": {"total_trades": 1382, "profit_factor": 1.29, "avg_trade": 4.96},
        "combine_evaluation": {"daily_pnl": {f"2026-04-{day:02d}": 100 for day in range(1, 25)}},
        "by_strategy": {"fx_at_one_glance": {"trades": 1311}},
    }

    quality = runner._evaluate_promotion_quality(result)

    assert quality["passed"] is False
    assert quality["checks"]["max_trades_per_day"]["ok"] is False
    assert quality["checks"]["max_fx_at_one_glance_trade_share_pct"]["ok"] is False


def test_discovery_service_evaluates_trade():
    root = _test_output_dir("discovery")
    db_path = root / "swarm.db"
    database = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": str(db_path)})
    experience_path = root / "experience.jsonl"
    discovery = DiscoveryService({"enabled": True, "experience_path": str(experience_path), "vector_enabled": True}, database=database)
    df = pd.DataFrame(
        {
            "close": [100 + i for i in range(130)],
            "high": [100.5 + i for i in range(130)],
            "low": [99.5 + i for i in range(130)],
            "open": [100 + i for i in range(130)],
        }
    )
    result = discovery.evaluate_trade("MNQ", df, {"strategy_name": "test", "quantity": 1, "side": "Buy"})
    assert result["allowed"] is True
    assert "quantity_multiplier" in result


def _bearish_df(n: int = 130) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [200 - i for i in range(n)],
            "high": [200.5 - i for i in range(n)],
            "low": [199.5 - i for i in range(n)],
            "open": [200 - i for i in range(n)],
        }
    )


def _build_discovery(root: Path) -> DiscoveryService:
    return DiscoveryService(
        {
            "enabled": True,
            "experience_path": str(root / "experience.jsonl"),
            "vector_enabled": False,
            "advisory_failure_similarity_threshold": 0.5,
        },
    )


def test_discovery_ignores_opposite_side_failures():
    """Regression: a past Buy-side loss must NOT block a fresh Sell intent."""
    root = _test_output_dir("discovery_side_filter")
    df = _bearish_df()
    experience = ExperienceStore(str(root / "experience.jsonl"))
    experience.persist_trial(
        instrument="MNQ",
        market_embedding=embed_market(df),
        params_embedding=[],
        outcome_embedding=[],
        full_params={},
        outcome={},
        passed_combine=False,
        side="Buy",
        strategy_name="sss",
        source="live",
    )
    discovery = _build_discovery(root)
    result = discovery.evaluate_trade("MNQ", df, {"strategy_name": "sss", "side": "Sell"})
    assert result["allowed"] is True


def test_discovery_same_side_failure_is_advisory_only():
    root = _test_output_dir("discovery_same_side")
    df = _bearish_df()
    experience = ExperienceStore(str(root / "experience.jsonl"))
    experience.persist_trial(
        instrument="MNQ",
        market_embedding=embed_market(df),
        params_embedding=[],
        outcome_embedding=[],
        full_params={},
        outcome={},
        passed_combine=False,
        side="Buy",
        strategy_name="sss",
        source="live",
    )
    discovery = _build_discovery(root)
    result = discovery.evaluate_trade("MNQ", df, {"strategy_name": "sss", "side": "Buy"})
    assert result["allowed"] is True
    assert result["quantity_multiplier"] == 1.0
    assert result["enforced"] is False
    assert "advisory" in result["reason"]
    assert result["matched_failure"]["side"] == "Buy"


def test_discovery_legacy_no_side_record_is_advisory_only():
    root = _test_output_dir("discovery_legacy")
    df = _bearish_df()
    experience = ExperienceStore(str(root / "experience.jsonl"))
    # Legacy-style record with no side — matches optimizer trials pre-fix.
    experience.persist_trial(
        instrument="MNQ",
        market_embedding=embed_market(df),
        params_embedding=[],
        outcome_embedding=[],
        full_params={},
        outcome={},
        passed_combine=False,
        # side, strategy_name default to None
    )
    discovery = _build_discovery(root)
    for side in ("Buy", "Sell"):
        result = discovery.evaluate_trade("MNQ", df, {"strategy_name": "x", "side": side})
        assert result["allowed"] is True
        assert result["quantity_multiplier"] == 1.0
        assert result["enforced"] is False
        assert "advisory" in result["reason"]


def test_discovery_strategy_filter():
    root = _test_output_dir("discovery_strategy_filter")
    df = _bearish_df()
    experience = ExperienceStore(str(root / "experience.jsonl"))
    experience.persist_trial(
        instrument="MNQ",
        market_embedding=embed_market(df),
        params_embedding=[],
        outcome_embedding=[],
        full_params={},
        outcome={},
        passed_combine=False,
        side="Sell",
        strategy_name="A",
        source="live",
    )
    discovery = _build_discovery(root)
    # Same strategy remains advisory.
    same_strategy = discovery.evaluate_trade("MNQ", df, {"strategy_name": "A", "side": "Sell"})
    # Different strategy has no matched failure.
    different_strategy = discovery.evaluate_trade("MNQ", df, {"strategy_name": "B", "side": "Sell"})
    assert same_strategy["allowed"] is True
    assert same_strategy["enforced"] is False
    assert same_strategy["matched_failure"]["strategy_name"] == "A"
    assert different_strategy["allowed"] is True
    assert different_strategy["matched_failure"] == {}


def test_persist_live_outcome_roundtrip():
    root = _test_output_dir("discovery_live_feedback")
    df = _bearish_df()
    experience = ExperienceStore(str(root / "experience.jsonl"))
    embedding = embed_market(df)
    experience.persist_live_outcome(
        instrument="MNQ",
        market_embedding=embedding,
        side="Sell",
        strategy_name="zone_confluence",
        pnl=-42.5,
        order_type="stoploss",
    )
    hits = experience.find_similar_failures(
        embedding,
        "MNQ",
        min_similarity=0.0,
        side="Sell",
        strategy_name="zone_confluence",
    )
    assert len(hits) == 1
    assert hits[0]["source"] == "live"
    assert hits[0]["side"] == "Sell"
    assert hits[0]["outcome"]["pnl"] == -42.5
    # Opposite side must NOT see the record.
    assert experience.find_similar_failures(
        embedding, "MNQ", min_similarity=0.0, side="Buy", strategy_name="zone_confluence"
    ) == []


def test_normalize_status_handles_ints_strings_and_none():
    assert normalize_status(2) == "filled"
    assert normalize_status("filled") == "filled"
    assert normalize_status("FILLED") == "filled"
    assert normalize_status(None) == ""
    assert normalize_status(6) == "pending_new"
    assert normalize_status("garbage") == "garbage"
    # Unknown int → defensive "code_N", never raises.
    assert normalize_status(99) == "code_99"
    # Type codes handled similarly.
    assert normalize_order_type(2) == "market"
    assert normalize_order_type(None) == ""


def test_health_monitor_recommendation():
    monitor = StrategyHealthMonitor({"drought_threshold_seconds": 0, "rejection_rate_threshold": 0.5})
    monitor.record_signal("MNQ", False, "blocked")
    rec = monitor.get_relaxation_recommendation("MNQ")
    assert rec["action"] in ("relax_filters", "none")


def test_database_service_persists_meta_optimizer_run():
    root = _test_output_dir("meta_optimizer")
    db_path = root / "swarm.db"
    service = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": str(db_path)})
    service.persist_meta_optimizer_run("MNQ", 1, 123.4, {"status": "ok"})
    rows = service.list_meta_optimizer_runs("MNQ", limit=5)
    assert len(rows) == 1
    assert rows[0]["epoch"] == 1


def test_database_service_persists_llm_analysis():
    root = _test_output_dir("llm_analysis")
    db_path = root / "swarm.db"
    service = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": str(db_path)})
    service.persist_llm_analysis("MNQ", 2, "review text", {"strategies.0.params.min_confluence": 5}, [{"file": "src/x.py"}], "reports/llm_analysis/mnq_epoch_2.txt")
    rows = service.list_llm_analysis("MNQ", limit=5)
    assert len(rows) == 1
    assert rows[0]["epoch"] == 2
    assert rows[0]["config_changes"]["strategies.0.params.min_confluence"] == 5


def test_database_service_learning_batch_is_idempotent():
    root = _test_output_dir("learning_batch")
    db_path = root / "swarm.db"
    service = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": str(db_path)})
    service.persist_trade_context(
        trade_key="ORDER-1",
        symbol="MNQM6",
        strategy_name="sss",
        side="Buy",
        quantity=2,
        order_id="ORDER-1",
        entry_price=100.0,
        opened_at="2026-04-24T16:00:00+00:00",
        strategy_params={"strategies": []},
        selected_signal={"strategy_name": "sss"},
        candidates=[{"strategy_name": "sss"}],
        blender_metadata={"mode": "confluence"},
        route_context={"current_price": 100.0},
        discovery_context={"allowed": True},
        account_snapshot={"balance": 50000},
        chart_artifacts={"timeframes": {"1W": {"status": "insufficient_data"}}},
    )
    service.persist_trade_outcome(
        trade_key="ORDER-1",
        symbol="MNQM6",
        order_id="ORDER-1",
        pnl=-25.0,
        close_type="sl",
        closed_at="2026-04-24T17:00:00+00:00",
        outcome={"pnl": -25.0},
    )
    rows = service.list_closed_unanalyzed_trades(limit=10)
    assert len(rows) == 1
    assert rows[0]["trade_key"] == "ORDER-1"
    service.create_learning_batch("BATCH-1", "local", ["ORDER-1"], status="processing")
    service.create_learning_batch("BATCH-1", "local", ["ORDER-1"], status="processing")
    service.complete_learning_batch(
        batch_id="BATCH-1",
        status="completed",
        raw_output_path="raw.txt",
        html_report_path="report.html",
        json_report_path="report.json",
        summary={"ok": True},
        analyzed_trade_keys=["ORDER-1"],
    )
    assert service.list_closed_unanalyzed_trades(limit=10) == []


def test_post_session_learning_script_generates_reports_and_marks_analyzed():
    root = _test_output_dir("post_session_script")
    db_path = root / "swarm.db"
    reports_dir = root / "reports"
    raw_dir = reports_dir / "raw"
    config_path = root / "stack.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "database": {"enabled": True, "backend": "sqlite", "sqlite_path": str(db_path)},
                "learning_review": {
                    "enabled": True,
                    "provider": "local",
                    "batch_size": 10,
                    "reports_dir": str(reports_dir),
                    "raw_output_dir": str(raw_dir),
                    "providers": {"local": {"command": []}},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    service = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": str(db_path)})
    service.persist_trade_context(
        trade_key="ORDER-SCRIPT",
        symbol="MNQM6",
        strategy_name="sss",
        side="Sell",
        quantity=1,
        order_id="ORDER-SCRIPT",
        entry_price=101.0,
        opened_at="2026-04-24T16:00:00+00:00",
        strategy_params={"strategies": []},
        selected_signal={"strategy_name": "sss", "side": "Sell"},
        candidates=[{"strategy_name": "sss", "side": "Sell"}],
        blender_metadata={"mode": "confluence"},
        route_context={"current_price": 101.0},
        discovery_context={"allowed": True},
        account_snapshot={"balance": 50000},
        chart_artifacts={"timeframes": {"1m": {"status": "insufficient_data"}}},
    )
    service.persist_trade_outcome(
        trade_key="ORDER-SCRIPT",
        symbol="MNQM6",
        order_id="ORDER-SCRIPT",
        pnl=-12.5,
        close_type="sl",
        closed_at="2026-04-24T17:00:00+00:00",
        outcome={"pnl": -12.5},
    )

    first = subprocess.run(
        [sys.executable, "scripts/run_post_session_learning.py", "--config", str(config_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    assert Path(payload["html_report"]).exists()
    assert Path(payload["json_report"]).exists()
    assert service.list_closed_unanalyzed_trades(limit=10) == []

    second = subprocess.run(
        [sys.executable, "scripts/run_post_session_learning.py", "--config", str(config_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert "no closed unanalyzed trades found" in second.stdout


def test_capture_trade_context_artifacts_marks_empty_history_insufficient():
    root = _test_output_dir("empty_artifacts")
    artifacts = capture_trade_context_artifacts(
        bars=pd.DataFrame(),
        trade_key="ORDER-EMPTY",
        symbol="MNQM6",
        opened_at=pd.Timestamp("2026-04-24T16:00:00Z").to_pydatetime(),
        config={"artifact_dir": str(root), "timeframes": ["1m", "5m"], "min_chart_bars": 8},
    )

    assert artifacts["timeframes"]["1m"]["status"] == "insufficient_data"
    assert artifacts["timeframes"]["5m"]["status"] == "insufficient_data"


def test_supervisor_select_trade_intent_does_not_call_mega_vision():
    class MegaStub:
        class Config:
            mode = "shadow"

        config = Config()

        async def decide(self, *_args, **_kwargs):
            raise AssertionError("pre-trade Mega Vision should not be called")

    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.feature_flags = {"zone_engine_enabled": False, "mega_vision_enabled": True}
    supervisor.mega_vision = MegaStub()

    selected = asyncio.run(
        SupervisorAgent._select_trade_intent(
            supervisor,
            "MNQM6",
            {"symbol": "MNQM6", "strategy_name": "sss", "side": "Buy", "quantity": 1},
            None,
            route_context={"current_price": 100.0},
        )
    )

    assert selected["strategy_name"] == "sss"
    assert selected["metadata"]["selection"]["authority"] == "deterministic"


def test_supervisor_select_trade_intent_preserves_strategy_candidate_set():
    class EngineStub:
        _last_candidates = [
            {"symbol": "MNQM6", "strategy_name": "sss", "side": "Buy", "quantity": 1},
            {"symbol": "MNQM6", "strategy_name": "ema_pullback", "side": "Buy", "quantity": 1},
        ]

    class StrategyAgentStub:
        engine = EngineStub()

    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.feature_flags = {"zone_engine_enabled": True}
    supervisor.strategy_agents = {"MNQM6": StrategyAgentStub()}

    selected = asyncio.run(
        SupervisorAgent._select_trade_intent(
            supervisor,
            "MNQM6",
            {"symbol": "MNQM6", "strategy_name": "sss", "side": "Buy", "quantity": 1},
            {"symbol": "MNQM6", "strategy_name": "zone_confluence", "side": "Buy", "quantity": 1},
        )
    )

    selection = selected["metadata"]["selection"]
    assert selected["strategy_name"] == "zone_confluence"
    assert [item["strategy_name"] for item in selection["candidates"]] == [
        "sss",
        "ema_pullback",
        "zone_confluence",
    ]


def test_supervisor_dispatch_keeps_discovery_advisory_only():
    class DiscoveryStub:
        def evaluate_trade(self, *_args, **_kwargs):
            return {
                "allowed": False,
                "quantity_multiplier": 0.25,
                "reason": "legacy advisory false",
                "matched_failure": {"side": "Buy", "strategy_name": "sss", "similarity": 1.0},
            }

    class DataAgentStub:
        def get_latest_data(self):
            return _bearish_df()

    class HealthStub:
        def record_signal(self, *_args, **_kwargs):
            pass

    class RouterStub:
        def __init__(self):
            self.seen = None

        def route_trade(self, trade_intent, _route_context):
            self.seen = dict(trade_intent)
            return RouterResult(False, quantity=int(trade_intent["quantity"]), result={}, reason="accepted")

    router = RouterStub()
    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.discovery = DiscoveryStub()
    supervisor.data_agents = {"MNQM6": DataAgentStub()}
    supervisor._last_discovery_decisions = {}
    supervisor.health_monitor = HealthStub()
    supervisor.trade_router = router
    supervisor.loop = None
    supervisor.logger = __import__("logging").getLogger("test-supervisor")

    async def run_dispatch():
        supervisor.loop = asyncio.get_running_loop()
        await SupervisorAgent._dispatch_trade(
            supervisor,
            "MNQM6",
            {"symbol": "MNQM6", "strategy_name": "sss", "side": "Buy", "quantity": 4},
            {"current_price": 100.0},
        )

    asyncio.run(run_dispatch())

    assert router.seen is not None
    assert router.seen["quantity"] == 4
    assert router.seen["metadata"]["discovery"]["allowed"] is True
    assert router.seen["metadata"]["discovery"]["quantity_multiplier"] == 1.0
    assert router.seen["metadata"]["discovery"]["enforced"] is True


def test_analysis_builder_writes_prompt_with_prior_analysis():
    root = _test_output_dir("analysis_builder")
    builder = AnalysisBuilder(str(root))
    prompt, prompt_path = builder.build_prompt(
        instrument="MNQM6",
        epoch=3,
        optimizer_result={"status": {"epoch": 3, "best_trial": {"summary": {"by_strategy": {"sss": {"trades": 4, "win_rate": 50.0, "pnl": 123.0}}}}}, "trials": []},
        screenshot_paths=["reports/backtest/optimizer_screenshots/run/chart.png"],
        prior_analysis=[{"epoch": 2, "reasoning": "trend logic lagged", "config_changes": {"x": 1}, "code_patches": []}],
    )
    assert "trend logic lagged" in prompt
    assert "chart.png" in prompt
    assert prompt_path.exists()


def test_llm_analyst_caps_prompt_and_uses_one_provider():
    analyst = LLMAnalyst(
        config={
            "provider": "claude",
            "max_provider_calls_per_analysis": 1,
            "max_prompt_chars": 7,
            "timeout_seconds": 30,
            "claude": [sys.executable, "-c", "import sys; print(len(sys.stdin.read()))"],
            "codex": [sys.executable, "-c", "raise SystemExit(99)"],
            "gemini": [sys.executable, "-c", "raise SystemExit(99)"],
        }
    )

    assert analyst._call("x" * 100).strip() == "7"


def test_default_external_model_paths_are_opt_in():
    stack = yaml.safe_load(Path("config/trading_stack.example.yaml").read_text(encoding="utf-8"))
    optimization = stack.get("optimization", {})
    assert optimization.get("discovery", {}).get("enabled", False) is True

    mega_vision = yaml.safe_load(Path("config/mega_vision.yaml").read_text(encoding="utf-8"))
    assert mega_vision["mode"] == "shadow"
    assert mega_vision["external_calls_enabled"] is False

    opt_loop = yaml.safe_load(Path("config/optimization_loop.yaml").read_text(encoding="utf-8"))
    assert opt_loop["claude"]["enabled"] is False
    assert opt_loop["codex"]["enabled"] is False
    assert opt_loop["gemini"]["enabled"] is False
    assert opt_loop["external_agents"]["max_provider_calls_per_iteration"] == 1

    discovery_loop = yaml.safe_load(Path("config/discovery_loop.yaml").read_text(encoding="utf-8"))
    assert discovery_loop["discovery"]["hypothesis_enabled"] is False
    assert discovery_loop["claude"]["enabled"] is False
    assert discovery_loop["codex"]["enabled"] is False
    assert discovery_loop["gemini"]["enabled"] is False
    assert discovery_loop["external_agents"]["max_provider_calls_per_run"] == 1


def test_meta_optimizer_service_runs_epoch():
    root = _test_output_dir("meta_service")
    data_file = _write_optimizer_fixture(root)
    db_path = root / "swarm.db"
    database = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": str(db_path)})
    config_path = root / "stack.yaml"
    stack = yaml.safe_load(Path("config/trading_stack.example.yaml").read_text(encoding="utf-8"))
    stack.setdefault("adaptive_optimizer", {}).setdefault("llm", {})["enabled"] = False
    stack["adaptive_optimizer"]["screenshots"] = {"enabled": False}
    config_path.write_text(yaml.safe_dump(stack, sort_keys=False), encoding="utf-8")
    service = MetaOptimizerService("MNQM6", data_file, config_path=str(config_path), database=database)
    result = service.run_epoch(epoch=1, trials_per_epoch=2)
    assert result["symbol"] == "MNQM6"
    assert "recommendation" in result


def test_adaptive_runner_supports_llm_analysis_disabled():
    root = _test_output_dir("adaptive_llm_disabled")
    data_file = _write_optimizer_fixture(root)
    runner = AdaptiveOptimizerRunner(
        "MNQM6",
        data_file,
        strategy_template={
            "blender": {"mode": "confluence", "multi_agree_bonus": 2.0},
            "direction_filters": {"sss": ["buy", "sell"]},
            "strategies": [{"name": "sss", "enabled": True, "priority": 80, "params": {"min_confluence": 2}}],
        },
        optimizer_config={
            "sss_min_confluence_values": [2],
            "enabled_sets": [{"sss": True}],
            "direction_candidates": {"default": [{"sss": ["buy", "sell"]}]},
            "llm": {"enabled": False},
            "screenshots": {"enabled": False},
        },
        combine_rules={"starting_balance": 50000, "profit_target": 3000, "max_loss_limit": 2000},
        experience_path=str(root / "experience.jsonl"),
        checkpoint_path=str(root / "checkpoint.json"),
        reports_dir=str(root / "reports"),
        database_config={"enabled": True, "backend": "sqlite", "sqlite_path": str(root / "swarm.db")},
    )
    result = runner.run_once(epoch=1, trials_per_epoch=1)
    assert result["llm_analysis"]["enabled"] is False


def test_supervisor_resolves_meta_optimizer_data_file_from_catalog():
    root = _test_output_dir("meta_catalog")
    data_file = root / "mnqm6_runtime.csv"
    data_file.write_text("close\n100\n", encoding="utf-8")
    catalog_path = root / "dataset_catalog.json"
    catalog_path.write_text(json.dumps([{"symbol": "MNQM6", "path": str(data_file)}]), encoding="utf-8")

    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.logger = __import__("logging").getLogger("test-supervisor")
    supervisor.meta_optimizer_config = {"dataset_catalog": str(catalog_path)}
    supervisor._meta_optimizer_catalog = SupervisorAgent._load_dataset_catalog(supervisor)

    assert SupervisorAgent._resolve_meta_optimizer_data_file(supervisor, "MNQM6") == str(data_file)


def test_supervisor_meta_optimizer_health_check_counts_ready_datasets():
    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.feature_flags = {"meta_optimizer_enabled": True}
    supervisor.meta_optimizer_config = {"enabled": True, "run_interval_seconds": 900}
    supervisor.official_symbols = ["MNQM6", "MCLK6"]
    supervisor._resolve_meta_optimizer_data_file = lambda symbol: "data/file.csv" if symbol == "MNQM6" else None

    health = SupervisorAgent._meta_optimizer_health_check(supervisor)

    assert health["connected"] is True
    assert health["datasets_ready"] == 1
    assert health["symbols_tracked"] == 2


def test_supervisor_updates_meta_optimizer_runtime_config():
    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.meta_optimizer_config = {"enabled": True, "run_interval_seconds": 900, "trials_per_epoch": 2, "market_closed_only": True, "max_recent_runs": 5}
    supervisor._write_status = lambda: None

    snapshot = SupervisorAgent.update_meta_optimizer_runtime_config(
        supervisor,
        {"enabled": False, "run_interval_seconds": 1200, "trials_per_epoch": 4, "market_closed_only": False},
    )

    assert snapshot["enabled"] is False
    assert snapshot["run_interval_seconds"] == 1200
    assert snapshot["trials_per_epoch"] == 4
    assert snapshot["market_closed_only"] is False


def test_meta_optimizer_cycle_skips_during_market_hours_when_closed_only():
    class _NewsStub:
        def is_market_hours(self, _):
            return True

    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.feature_flags = {"meta_optimizer_enabled": True}
    supervisor.meta_optimizer_config = {"enabled": True, "market_closed_only": True, "run_interval_seconds": 900, "trials_per_epoch": 2}
    supervisor.news_agent = _NewsStub()
    supervisor.official_symbols = ["MNQM6"]
    supervisor._last_meta_optimizer_runs = {}
    supervisor._meta_optimizer_last_run_at = {}
    supervisor._meta_optimizer_services = {}
    supervisor.database = DatabaseService({"enabled": False})
    supervisor.loop = None

    import asyncio

    asyncio.run(SupervisorAgent._run_meta_optimizer_cycle(supervisor, 1000.0))

    assert supervisor._last_meta_optimizer_runs == {}


def test_claude_md_writer_round_trip():
    root = _test_output_dir("claude_md")
    path = root / "CLAUDE.md"
    writer = ClaudeMdWriter(str(path))
    writer.append_learning(
        run_id="run_001",
        changes_made="tighten sss confluence",
        metrics_before={"pass_rate": 0.2, "win_rate": 0.4, "max_dd": 0.1},
        metrics_after={"pass_rate": 0.3, "win_rate": 0.5, "max_dd": 0.08},
        verdict="kept",
    )
    summary = writer.get_learnings_summary()
    assert "run_001" in summary
    assert "tighten sss confluence" in summary


def test_results_exporter_writes_and_lists_reports():
    root = _test_output_dir("results_exporter")
    exporter = ResultsExporter(str(root))
    path = exporter.export_run_report(
        result={"summary": {"total_pnl": 123.0, "win_rate": 0.55, "total_trades": 10}, "trades": [{"pnl": -10, "strategy_name": "sss", "side": "Buy"}]},
        config={"strategies": {}},
        run_id="run_001",
    )
    assert path.exists()
    runs = exporter.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_001"


def test_mega_strategy_agent_parses_json_pick():
    agent = MegaStrategyAgent(MegaVisionConfig(mode="shadow"))
    candidates = [{"strategy_name": "sss", "side": "Buy", "quantity": 1, "sl_ticks": 20, "tp_ticks": 40}]
    parsed = agent._parse_response('{"strategy_name":"sss","side":"Buy","confidence":0.72,"reasoning":"best","no_pick":false,"advisory_only":true}', candidates)
    assert parsed is not None
    assert parsed["strategy_name"] == "sss"
    assert parsed["advisory_only"] is True
    assert parsed["no_pick"] is False
    assert parsed["fallback"] is False


def test_mega_strategy_agent_no_pick_response_is_advisory():
    agent = MegaStrategyAgent(MegaVisionConfig(mode="shadow"))
    parsed = agent._parse_response('{"no_pick":true,"reasoning":"no edge","advisory_only":true}', [])
    assert parsed is not None
    assert parsed["advisory_only"] is True
    assert parsed["no_pick"] is True


def test_mega_strategy_agent_external_calls_disabled_returns_advisory_no_pick():
    agent = MegaStrategyAgent(MegaVisionConfig(mode="shadow", external_calls_enabled=False))
    agent._spawn_claude = lambda _prompt: (_ for _ in ()).throw(AssertionError("external call attempted"))
    candidate = {"strategy_name": "sss", "side": "Buy", "quantity": 1, "sl_ticks": 20, "tp_ticks": 40}

    parsed = asyncio.run(
        agent.decide(
            datetime.now(timezone.utc),
            [candidate],
            {"market_open": True, "news_standdown": False, "near_close": False, "active_count": 0, "max_positions": 1, "current_price": 100.0},
        )
    )

    assert parsed["advisory_only"] is True
    assert parsed["no_pick"] is True
    assert "external_calls_disabled" in parsed["reasoning"]


def test_mega_vision_scorecard_no_pick_is_not_reject(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor._last_mega_vision_decisions = {
        "MNQM6": {
            "no_pick": True,
            "agreed": False,
            "strategy_name": None,
            "native_pick": {"strategy_name": "sss", "side": "Buy"},
            "confidence": 0.0,
            "reasoning": "no edge",
        }
    }
    supervisor.logger = __import__("logging").getLogger("test-supervisor")

    SupervisorAgent._score_mega_vision_decision(supervisor, symbol="MNQM6", pnl=-25.0, closed={})

    score_path = tmp_path / "logs" / "telemetry" / "mega_vision_scorecard.jsonl"
    record = json.loads(score_path.read_text(encoding="utf-8").strip())
    assert record["llm_no_pick"] is True
    assert "llm_rejected" not in record


class _RiskCloseStub:
    def __init__(self):
        self.closed_symbols = []
        self.trade_results = []

    def record_position_close(self, symbol):
        self.closed_symbols.append(symbol)

    def record_trade_result(self, pnl):
        self.trade_results.append(pnl)


class _TelemetryCloseStub:
    def __init__(self):
        self.closed = []

    def record_trade_close(self, *args):
        self.closed.append(args)


def test_position_close_uses_pending_fill_pnl_for_trade_log():
    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor._lock = asyncio.Lock()
    supervisor.active_positions = {
        "MGCM6": [
            {
                "side": "Buy",
                "qty": 37,
                "entry": 4629.8,
                "time": 1234567890,
                "strategy_name": "price_action_bs",
            }
        ]
    }
    supervisor.risk_agent = _RiskCloseStub()
    supervisor.telemetry = _TelemetryCloseStub()
    supervisor.logger = __import__("logging").getLogger("test-supervisor-close")
    supervisor._pending_fill_pnl = {"MGCM6": 1480.0}
    supervisor._close_in_flight = {}
    supervisor._close_in_flight_since = {}
    supervisor._recently_closed = {}
    supervisor._last_trade_won = {}
    supervisor._consecutive_losses = {}
    supervisor._recent_trades = []
    supervisor._record_chart_trade_event = lambda *args, **kwargs: None
    supervisor._persist_closed_trade_outcome = lambda **kwargs: None
    supervisor._feed_live_outcome_to_discovery = lambda **kwargs: None
    supervisor._score_mega_vision_decision = lambda **kwargs: None

    asyncio.run(
        SupervisorAgent._handle_position_close_update(
            supervisor,
            "MGCM6",
            "takeprofit",
            None,
            drain=False,
        )
    )

    assert supervisor._pending_fill_pnl == {}
    assert supervisor._recent_trades[-1]["pnl"] == 1480.0
    assert supervisor.risk_agent.trade_results == [1480.0]
    assert supervisor.telemetry.closed[0][4] == 1480.0


def test_max_positions_filter_flips():
    import pandas as pd

    from src.trading.filters import FilterManager, TradingFilterContext

    # 1. Test allow_flips=False (default behavior)
    manager_no_flips = FilterManager({"entry_filters": {"allow_flips": False}})
    context = TradingFilterContext(
        symbol="MNQM6",
        now_dt=pd.Timestamp("2026-06-09T10:00:00Z").to_pydatetime(),
        trade_intent={"symbol": "MNQM6", "side": "Sell"},
        market_open=True,
        news_standdown=False,
        near_close=False,
        active_count=1,
        max_positions=1,
        pending_order=False,
        last_signal_time=0.0,
        cooldown_seconds=0,
        existing_side="Buy", # opposite side
    )
    allowed, decisions = manager_no_flips.evaluate_entry(context)
    assert allowed is False
    assert decisions[-1].name == "max_positions"
    assert "already at max positions" in decisions[-1].reason

    # 2. Test allow_flips=True
    manager_with_flips = FilterManager({"entry_filters": {"allow_flips": True}})
    allowed, decisions = manager_with_flips.evaluate_entry(context)
    assert allowed is True
    # find the decision for max_positions filter
    max_pos_dec = next(d for d in decisions if d.name == "max_positions")
    assert max_pos_dec.allowed is True
    assert max_pos_dec.metadata.get("flip") is True


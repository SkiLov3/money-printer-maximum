"""Unit tests for Saturday Autonomous Optimizer.

Verifies:
- Market hours safety lock (Fri 17:00 ET to Sun 17:00 ET)
- StateManager persistence and crash recovery
- Atomic configuration staging, backup creation, and rollback safety
- Fallback hypotheses when LLM is unavailable
"""
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.saturday_autonomous_optimizer import (
    StateManager,
    call_llm_stage,
    generate_optimization_report,
    get_fallback_hypotheses,
    is_market_closed,
    query_llm_hypotheses,
    stage_atomic_config_update,
    stage_codestral_certify,
    stage_llama_redteam,
)


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_state_manager_lifecycle(temp_dir):
    state_file = temp_dir / "test_state.json"
    mgr = StateManager(state_file)

    assert mgr.state["phase"] == "INITIAL"
    mgr.update_phase("DIAGNOSTIC_AUDIT")

    # Verify persisted to disk
    with open(state_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["phase"] == "DIAGNOSTIC_AUDIT"

    # Simulate process restart
    mgr2 = StateManager(state_file)
    assert mgr2.state["phase"] == "DIAGNOSTIC_AUDIT"


def test_state_manager_corrupted_file_recovery(temp_dir):
    state_file = temp_dir / "corrupted_state.json"
    state_file.write_text("{bad json...", encoding="utf-8")

    mgr = StateManager(state_file)
    assert mgr.state["phase"] == "INITIAL"


def test_market_closed_schedule():
    # Friday 18:00 ET (Closed)
    with patch("scripts.saturday_autonomous_optimizer.datetime") as mock_dt:
        mock_now = MagicMock()
        mock_now.weekday.return_value = 4  # Friday
        mock_now.hour = 18
        mock_dt.now.return_value = mock_now
        assert is_market_closed() is True

    # Saturday 12:00 ET (Closed)
    with patch("scripts.saturday_autonomous_optimizer.datetime") as mock_dt:
        mock_now = MagicMock()
        mock_now.weekday.return_value = 5  # Saturday
        mock_now.hour = 12
        mock_dt.now.return_value = mock_now
        assert is_market_closed() is True

    # Sunday 16:00 ET (Closed)
    with patch("scripts.saturday_autonomous_optimizer.datetime") as mock_dt:
        mock_now = MagicMock()
        mock_now.weekday.return_value = 6  # Sunday
        mock_now.hour = 16
        mock_dt.now.return_value = mock_now
        assert is_market_closed() is True

    # Tuesday 10:00 ET (Open)
    with patch("scripts.saturday_autonomous_optimizer.datetime") as mock_dt:
        mock_now = MagicMock()
        mock_now.weekday.return_value = 1  # Tuesday
        mock_now.hour = 10
        mock_dt.now.return_value = mock_now
        assert is_market_closed() is False


def test_atomic_config_staging(temp_dir):
    config_file = temp_dir / "trading_stack.yaml"
    initial_content = {"strategies": {"mode": "priority_first"}}
    with open(config_file, "w") as f:
        yaml.safe_dump(initial_content, f)

    backup_dir = temp_dir / "backups"

    with patch("scripts.saturday_autonomous_optimizer.CONFIG_FILE", config_file), \
         patch("scripts.saturday_autonomous_optimizer.BACKUP_DIR", backup_dir):

        # 1. Test Dry Run
        success = stage_atomic_config_update(
            {"strategies": {"mode": "dry_run"}}, "Dry run candidate", dry_run=True
        )
        assert success is True
        # Original remains untouched
        with open(config_file) as f:
            assert yaml.safe_load(f)["strategies"]["mode"] == "priority_first"

        # 2. Test Live Atomic Replacement
        new_content = {"strategies": {"mode": "promoted_new"}}
        success = stage_atomic_config_update(new_content, "Live promotion", dry_run=False)
        assert success is True

        # Check backup created
        backups = list(backup_dir.glob("*.yaml"))
        assert len(backups) == 1
        with open(backups[0]) as f:
            assert yaml.safe_load(f)["strategies"]["mode"] == "priority_first"

        # Check active config replaced cleanly
        with open(config_file) as f:
            assert yaml.safe_load(f)["strategies"]["mode"] == "promoted_new"


def test_fallback_hypotheses_when_ollama_offline():
    with patch("scripts.saturday_autonomous_optimizer.ping_ollama", return_value=False):
        res = query_llm_hypotheses([])
        assert "hypotheses" in res
        assert len(res["hypotheses"]) > 0
        assert res["hypotheses"][0]["strategy"] == "mtf_trend_scalper"


def test_stage_llama_redteam_offline_fallback():
    hypotheses = [{"strategy": "mtf_trend_scalper", "parameter": "sl_atr_mult", "test_values": [1.5, 2.0]}]
    with patch("scripts.saturday_autonomous_optimizer.ping_ollama", return_value=False):
        approved, audit = stage_llama_redteam(hypotheses)
        assert approved == hypotheses
        assert "offline" in audit.get("redteam_summary", "").lower()


def test_stage_llama_redteam_filtering():
    hypotheses = [
        {"strategy": "mtf_trend_scalper", "parameter": "sl_atr_mult", "test_values": [1.5]},
        {"strategy": "mtf_trend_scalper", "parameter": "sl_atr_mult", "test_values": [0.2]},
    ]
    mock_response = {
        "redteam_summary": "Filtered out dangerous 0.2 stop loss.",
        "approved_hypotheses": [
            {"strategy": "mtf_trend_scalper", "parameter": "sl_atr_mult", "test_values": [1.5], "rationale": "Valid"}
        ],
        "rejected_hypotheses": [
            {"strategy": "mtf_trend_scalper", "parameter": "sl_atr_mult", "rejection_reason": "Below 1.0"}
        ],
    }
    with patch("scripts.saturday_autonomous_optimizer.ping_ollama", return_value=True), \
         patch("scripts.saturday_autonomous_optimizer.call_llm_stage", return_value=mock_response):
        approved, audit = stage_llama_redteam(hypotheses)
        assert len(approved) == 1
        assert approved[0]["test_values"] == [1.5]
        assert len(audit["rejected_hypotheses"]) == 1


def test_stage_codestral_certify_rejection_of_unsupported_param():
    candidate = {"strategy": "mtf_trend_scalper", "parameter": "fake_parameter_xyz", "value": 1.5}
    cert = stage_codestral_certify(candidate)
    assert cert["certified"] is False
    assert "not whitelisted" in cert["reason"]


def test_stage_codestral_certify_offline_fallback():
    candidate = {"strategy": "mtf_trend_scalper", "parameter": "sl_atr_mult", "value": 1.5}
    with patch("scripts.saturday_autonomous_optimizer.ping_ollama", return_value=False):
        cert = stage_codestral_certify(candidate)
        assert cert["certified"] is True
        assert cert["code_signature_verified"] is True


def test_generate_optimization_report(temp_dir):
    state = {
        "run_id": "2026-10-18",
        "extracted_trades_count": 42,
        "analysis_summary": "Weekly audit summary test",
        "identified_flaws": ["High slippage around NY open"],
        "evaluation_results": [
            {
                "candidate_id": "mtf_trend_scalper.sl_atr_mult=1.5",
                "total_trades": 85,
                "win_rate": 0.58,
                "total_pnl": 2450.0,
                "profit_factor": 1.35,
                "holdout_pnl": 650.0,
                "holdout_win_rate": 0.60,
                "max_drawdown": 420.0,
            }
        ],
    }
    base_metrics = {
        "total_trades": 80,
        "win_rate": 0.55,
        "total_pnl": 1800.0,
        "profit_factor": 1.25,
        "holdout_pnl": 350.0,
        "holdout_win_rate": 0.54,
        "max_drawdown": 480.0,
    }
    winner = state["evaluation_results"][0]
    redteam = {"redteam_summary": "All bounds checked", "approved_hypotheses": [], "rejected_hypotheses": []}
    codestral_cert = {"certified": True, "strategy": "mtf_trend_scalper", "parameter": "sl_atr_mult", "value": 1.5, "reason": "AST verified", "code_signature_verified": True}

    with patch("scripts.saturday_autonomous_optimizer.REPORTS_DIR", temp_dir):
        report_path = generate_optimization_report(state, base_metrics, winner, redteam, codestral_cert, dry_run=False)
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Saturday Autonomous 3-LLM Optimization Report" in content
        assert "Stage 1: Diagnosis & Hypothesis Generation (Qwen 3.5 122B)" in content
        assert "Stage 2: Adversarial Red-Team & Risk Audit (Llama 3.3 70B)" in content
        assert "Stage 4: Strategy Code Contract Certification (Codestral 22B)" in content
        assert "WINNER" in content


def test_stage_llama_redteam_unanimous_rejection():
    """Fail-safe test: when red-team rejects all hypotheses, returns empty list rather than raw hypotheses."""
    hypotheses = [
        {"strategy": "mtf_trend_scalper", "parameter": "sl_atr_mult", "test_values": [0.1]},
    ]
    mock_response = {
        "redteam_summary": "Violates minimum 1.0 ATR stop rule. Rejecting all.",
        "approved_hypotheses": [],
        "rejected_hypotheses": [
            {"strategy": "mtf_trend_scalper", "parameter": "sl_atr_mult", "rejection_reason": "Stop loss too tight"}
        ],
    }
    with patch("scripts.saturday_autonomous_optimizer.ping_ollama", return_value=True), \
         patch("scripts.saturday_autonomous_optimizer.call_llm_stage", return_value=mock_response):
        approved, audit = stage_llama_redteam(hypotheses)
        # MUST be empty list, NOT falling back to raw hypotheses
        assert approved == []
        assert len(audit.get("rejected_hypotheses", [])) == 1


def test_stage_llama_redteam_api_failure_fallback_to_conservative_grid():
    """Fail-safe test: when LLM call fails, falls back to conservative grid rather than untrusted raw hypotheses."""
    hypotheses = [
        {"strategy": "mtf_trend_scalper", "parameter": "sl_atr_mult", "test_values": [999.0]},
    ]
    with patch("scripts.saturday_autonomous_optimizer.ping_ollama", return_value=True), \
         patch("scripts.saturday_autonomous_optimizer.call_llm_stage", return_value=None):
        approved, audit = stage_llama_redteam(hypotheses)
        fallback_hypo = get_fallback_hypotheses()["hypotheses"]
        assert approved == fallback_hypo
        assert "fallback" in audit.get("redteam_summary", "").lower()


def test_call_llm_stage_enforces_root_keep_alive():
    """Verifies that call_llm_stage places keep_alive at top level of request body per Ollama API spec."""
    captured_body = []

    def mock_urlopen(req, timeout=300):
        data = json.loads(req.data.decode("utf-8"))
        captured_body.append(data)
        resp = MagicMock()
        resp.read.return_value = json.dumps({"message": {"content": "{}"}}).encode("utf-8")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        return cm

    with patch("scripts.saturday_autonomous_optimizer.wait_for_gpu_free"), \
         patch("scripts.saturday_autonomous_optimizer.evict_model_from_gpu"), \
         patch("urllib.request.urlopen", side_effect=mock_urlopen):
        res = call_llm_stage("codestral:22b", "sys", "user")
        assert res == {}
        assert len(captured_body) == 1
        body = captured_body[0]
        # Crucial: keep_alive must be at root level, not inside options
        assert body.get("keep_alive") == "0s"
        assert "keep_alive" not in body.get("options", {})


def test_schema_enums_include_all_tunable_params():
    """Verifies that SCHEMA_HYPOTHESIS and SCHEMA_REDTEAM allow all TUNABLE_PARAMS (including orb_vwap)."""
    from scripts.saturday_autonomous_optimizer import (
        SCHEMA_HYPOTHESIS,
        SCHEMA_REDTEAM,
        TUNABLE_PARAMS,
    )

    hypo_enum = set(
        SCHEMA_HYPOTHESIS["properties"]["hypotheses"]["items"]["properties"]["parameter"]["enum"]
    )
    redteam_enum = set(
        SCHEMA_REDTEAM["properties"]["approved_hypotheses"]["items"]["properties"]["parameter"]["enum"]
    )
    expected = {p for params in TUNABLE_PARAMS.values() for p in params}
    assert hypo_enum == expected
    assert redteam_enum == expected
    assert "min_volume_ratio" in hypo_enum
    assert "asian_volume_ratio_min" not in hypo_enum


def test_atomic_config_update_aborts_if_bot_status_has_active_positions(temp_dir):
    """Verifies pre-promotion safety check aborts live swap if bot_status.json shows open positions."""
    config_file = temp_dir / "trading_stack.yaml"
    backup_dir = temp_dir / "backups"
    status_file = temp_dir / "bot_status.json"

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump({"strategies": {"mode": "original"}}, f)
    status_file.write_text(
        json.dumps({"active_positions": {"MCLZ6": [{"side": "Sell", "qty": 1}]}}),
        encoding="utf-8",
    )

    with (
        patch("scripts.saturday_autonomous_optimizer.CONFIG_FILE", config_file),
        patch("scripts.saturday_autonomous_optimizer.BACKUP_DIR", backup_dir),
        patch("scripts.saturday_autonomous_optimizer.BOT_STATUS_FILE", status_file),
    ):
        ok = stage_atomic_config_update({"strategies": {"mode": "new"}}, "Promote", dry_run=False)
        assert ok is False
        with open(config_file, encoding="utf-8") as f:
            assert yaml.safe_load(f)["strategies"]["mode"] == "original"


def test_backfill_broker_trades_dynamic_symbol_and_partial_fills():
    """Verifies dynamic contract symbol extraction and partial fill FIFO pairing."""
    from scripts.backfill_broker_trades import _symbol_for, pair_round_trips

    assert _symbol_for("CON.F.US.MCLE.Z26") == "MCLZ6"
    assert _symbol_for("CON.F.US.MCLE.X26") == "MCLX6"
    assert _symbol_for("CON.F.US.MES.H27") == "MESH7"

    fills = [
        {
            "orderId": 1001,
            "contractId": "CON.F.US.MES.Z26",
            "side": 0,
            "size": 2,
            "price": 5800.0,
            "creationTimestamp": "2026-10-20T14:00:00+00:00",
            "profitAndLoss": None,
            "fees": 2.0,
        },
        {
            "orderId": 1002,
            "contractId": "CON.F.US.MES.Z26",
            "side": 1,
            "size": 1,
            "price": 5805.0,
            "creationTimestamp": "2026-10-20T14:05:00+00:00",
            "profitAndLoss": 25.0,
            "fees": 1.0,
        },
        {
            "orderId": 1003,
            "contractId": "CON.F.US.MES.Z26",
            "side": 1,
            "size": 1,
            "price": 5810.0,
            "creationTimestamp": "2026-10-20T14:10:00+00:00",
            "profitAndLoss": 50.0,
            "fees": 1.0,
        },
    ]
    trips = pair_round_trips(fills)
    assert len(trips) == 2
    assert trips[0]["quantity"] == 1
    assert trips[0]["pnl"] == 25.0
    assert trips[0]["fees"] == 2.0  # 1.0 prorated entry + 1.0 exit
    assert trips[1]["quantity"] == 1
    assert trips[1]["pnl"] == 50.0
    assert trips[1]["fees"] == 2.0


def test_extract_recent_trades_includes_outcome_without_context(temp_dir):
    """Verifies LEFT JOIN in extract_recent_trades retains outcomes whose trade_contexts row is absent."""
    from datetime import datetime, timezone

    from scripts.saturday_autonomous_optimizer import extract_recent_trades
    from src.persistence.database import DatabaseService

    db_file = temp_dir / "test_left_join.db"
    db = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": str(db_file)})
    now_iso = datetime.now(timezone.utc).isoformat()
    db.persist_trade_outcome(
        trade_key="orphan-outcome-1",  # gitleaks:allow
        symbol="MCLZ6",
        order_id="999",
        account_id="12345678",
        session="ASIAN",
        pnl=63.0,
        close_type="limit",
        closed_at=now_iso,
        outcome={"closed_position": {"strategy_name": "position_update_sync", "side": "Sell", "qty": 1, "entry": 90.5}},
    )
    db.close()

    trades = extract_recent_trades(db_file, days=7)
    assert len(trades) == 1
    assert trades[0]["trade_key"] == "orphan-outcome-1"
    assert trades[0]["pnl"] == 63.0
    assert trades[0]["strategy_name"] == "position_update_sync"
    assert trades[0]["side"] == "Sell"


def test_apply_param_override_mirrors_asian_orb_and_symbol_params():
    """Verifies apply_param_override mirrors orb_vwap params to asian_* when Asian-only and syncs symbol_params."""
    from scripts.saturday_autonomous_optimizer import apply_param_override
    from src.strategies.orb_vwap import ORBVWAPStrategy

    cfg = {
        "strategies": {
            "strategies": [
                {
                    "name": "orb_vwap",
                    "enabled": True,
                    "params": {
                        "opening_range_enabled": False,
                        "asian_enabled": True,
                        "min_volume_ratio": 0.0,
                        "risk_reward": 2.0,
                        "asian_risk_reward": 2.0,
                        "symbol_params": {
                            "MCL": {"asian_min_or_ticks": 10, "min_volume_ratio": 0.0},
                        },
                    },
                },
                {
                    "name": "mtf_trend_scalper",
                    "enabled": True,
                    "params": {
                        "min_adx": 35.0,
                        "symbol_params": {
                            "MES": {"min_adx": 35.0},
                        },
                    },
                },
            ]
        }
    }

    assert apply_param_override(cfg, "orb_vwap", "min_or_ticks", 25) is True
    assert apply_param_override(cfg, "orb_vwap", "min_volume_ratio", 1.5) is True
    assert apply_param_override(cfg, "orb_vwap", "risk_reward", 2.5) is True
    assert apply_param_override(cfg, "mtf_trend_scalper", "min_adx", 30.0) is True

    orb_params = cfg["strategies"]["strategies"][0]["params"]
    assert orb_params["min_or_ticks"] == 25
    assert orb_params["asian_min_or_ticks"] == 25
    assert orb_params["min_volume_ratio"] == 1.5
    assert orb_params["asian_min_volume_ratio"] == 1.5
    assert orb_params["risk_reward"] == 2.5
    assert orb_params["asian_risk_reward"] == 2.5
    assert orb_params["symbol_params"]["MCL"]["asian_min_or_ticks"] == 25
    assert orb_params["symbol_params"]["MCL"]["min_volume_ratio"] == 1.5

    assert apply_param_override(cfg, "orb_vwap", "asian_min_or_ticks", 20) is True
    assert orb_params["min_or_ticks"] == 20
    assert orb_params["asian_min_or_ticks"] == 20
    assert orb_params["symbol_params"]["MCL"]["asian_min_or_ticks"] == 20

    strategy_obj = ORBVWAPStrategy("MCLZ6", orb_params)
    assert strategy_obj.asian_min_or_ticks == 20
    assert strategy_obj.asian_min_volume_ratio == 1.5
    assert strategy_obj.asian_risk_reward == 2.5

    mtf_params = cfg["strategies"]["strategies"][1]["params"]
    assert mtf_params["min_adx"] == 30.0
    assert mtf_params["symbol_params"]["MES"]["min_adx"] == 30.0


def test_reclaim_ticks_tunable_and_mfe_mae_excursion_telemetry(temp_dir):
    """Verifies fade_reclaim_ticks/asian_fade_reclaim_ticks whitelist and MFE/MAE telemetry extraction."""
    from datetime import datetime, timezone
    from unittest.mock import patch

    from scripts.saturday_autonomous_optimizer import (
        apply_param_override,
        extract_recent_trades,
        is_valid_candidate_param,
        query_llm_hypotheses,
    )
    from src.persistence.database import DatabaseService

    assert is_valid_candidate_param("orb_vwap", "fade_reclaim_ticks") is True
    assert is_valid_candidate_param("orb_vwap", "asian_fade_reclaim_ticks") is True

    cfg = {
        "strategies": {
            "strategies": [
                {
                    "name": "orb_vwap",
                    "enabled": True,
                    "params": {
                        "opening_range_enabled": False,
                        "asian_enabled": True,
                        "fade_reclaim_ticks": 0,
                        "asian_fade_reclaim_ticks": 0,
                    },
                }
            ]
        }
    }
    assert apply_param_override(cfg, "orb_vwap", "asian_fade_reclaim_ticks", 2) is True
    orb_p = cfg["strategies"]["strategies"][0]["params"]
    assert orb_p["asian_fade_reclaim_ticks"] == 2
    assert orb_p["fade_reclaim_ticks"] == 2

    db_file = temp_dir / "test_excursions.db"
    db = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": str(db_file)})
    now_iso = datetime.now(timezone.utc).isoformat()
    db.persist_trade_outcome(
        trade_key="mfe-mae-1",  # gitleaks:allow
        symbol="MCLZ6",
        order_id="1001",
        account_id="12345678",
        session="ASIAN",
        pnl=-50.0,
        close_type="stop",
        closed_at=now_iso,
        outcome={
            "closed_position": {"strategy_name": "orb_vwap", "side": "Sell", "qty": 1, "entry": 70.00, "sl_ticks": 20},
            "mfe_ticks": 30.0,
            "mae_ticks": 20.0,
            "mfe_r": 1.5,
            "mae_r": 1.0,
        },
    )
    db.close()

    trades = extract_recent_trades(db_file, days=7)
    assert len(trades) == 1
    captured_payload = {}

    def _fake_call(model, sys_p, user_c, schema=None, timeout=180):
        captured_payload["data"] = json.loads(user_c)
        return {"analysis_summary": "ok", "identified_flaws": [], "hypotheses": []}

    with (
        patch("scripts.saturday_autonomous_optimizer.ping_ollama", return_value=True),
        patch("scripts.saturday_autonomous_optimizer.call_llm_stage", side_effect=_fake_call),
    ):
        query_llm_hypotheses(trades)

    loss_item = captured_payload["data"]["sample_losses"][0]
    assert loss_item["mfe_ticks"] == 30.0
    assert loss_item["mae_ticks"] == 20.0
    assert loss_item["mfe_r"] == 1.5
    assert loss_item["mae_r"] == 1.0







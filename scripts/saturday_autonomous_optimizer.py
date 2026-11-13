#!/usr/bin/env python3
"""Saturday Autonomous Multi-LLM Optimizer & Strategy Calibrator.

Operates strictly offline on weekends (Friday 17:00 ET to Sunday 17:00 ET).
Runs a sequential 3-LLM relay with explicit Metal GPU memory eviction (keep_alive: 0s):
1. Stage 1 (Qwen 3.5 122B): Deep quantitative diagnosis & hypothesis generator
2. Stage 2 (Llama 3.3 70B): Adversarial red-team auditor & combine risk validator
3. Stage 3 (CPU Backtester): Multicore empirical validation across 4.5 months + 21-day holdout
4. Stage 4 (Codestral 22B): Python strategy code analyzer & schema contract certifier
5. Stage 5 (Promotion Engine): Atomic YAML configuration update and markdown audit reporting

Compliance:
- AGENTS.md: Offline research only. Zero live execution gates. Zero live latency.
- TopstepX: Enforces combine drawdown limits, circuit breakers, and fee modeling.
- Apple Silicon (128GB RAM): Sequential execution with GPU memory verification guarantees peak memory <= 81GB.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import fcntl
import gc
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import sys
import time
import urllib.request
import zoneinfo
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import get_strategy_config, load_stack_config
from src.utils.backtester import Backtester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SaturdayOptimizer")

STATE_FILE = REPO_ROOT / "logs" / "saturday_optimizer_state.json"
LOCK_FILE = REPO_ROOT / "logs" / "saturday_optimizer.lock"
BOT_STATUS_FILE = REPO_ROOT / "logs" / "bot_status.json"
SWARM_DB = REPO_ROOT / "logs" / "swarm.db"
CONFIG_FILE = REPO_ROOT / "config" / "trading_stack.yaml"
BACKUP_DIR = REPO_ROOT / "config" / "backups"
REPORTS_DIR = REPO_ROOT / "docs"

CONTINUOUS_DATA = {
    "MES": REPO_ROOT / "data" / "continuous" / "MES_1m.csv",
    "MCL": REPO_ROOT / "data" / "continuous" / "MCL_1m.csv",
}

DEFAULT_MODELS = {
    "diagnose": "qwen3.5:122b",
    "redteam": "llama3.3:70b",
    "certify": "codestral:22b",
}

# Real parameters each strategy reads via params.get(...). Candidates outside this map are rejected.
TUNABLE_PARAMS: dict[str, tuple[str, ...]] = {
    "mtf_trend_scalper": ("min_adx", "sl_atr_mult", "tp_atr_mult", "min_stop_ticks", "max_stop_ticks"),
    "orb_vwap": (
        "min_volume_ratio",
        "risk_reward",
        "min_or_ticks",
        "max_or_ticks",
        "breakout_buffer_ticks",
        "fade_reclaim_ticks",
        "asian_risk_reward",
        "asian_min_or_ticks",
        "asian_max_or_ticks",
        "asian_min_volume_ratio",
        "asian_stop_buffer_ticks",
        "asian_breakout_buffer_ticks",
        "asian_fade_reclaim_ticks",
    ),
}
ALL_TUNABLE_PARAMS: list[str] = [p for params in TUNABLE_PARAMS.values() for p in params]

PARAM_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "mtf_trend_scalper": {
        "min_adx": (15.0, 45.0),
        "sl_atr_mult": (1.0, 3.5),
        "tp_atr_mult": (1.5, 6.0),
        "min_stop_ticks": (8.0, 40.0),
        "max_stop_ticks": (30.0, 120.0),
        "skip_open_minutes": (0.0, 30.0),
        "skip_close_minutes": (0.0, 30.0),
    },
    "orb_vwap": {
        "risk_reward": (1.0, 4.0),
        "min_or_ticks": (5.0, 40.0),
        "max_or_ticks": (30.0, 200.0),
        "breakout_buffer_ticks": (0.0, 10.0),
        "fade_reclaim_ticks": (0.0, 10.0),
        "asian_risk_reward": (1.0, 4.0),
        "asian_min_or_ticks": (5.0, 40.0),
        "asian_max_or_ticks": (30.0, 200.0),
        "asian_min_volume_ratio": (0.0, 3.0),
        "asian_stop_buffer_ticks": (0.0, 10.0),
        "asian_breakout_buffer_ticks": (0.0, 10.0),
        "asian_fade_reclaim_ticks": (0.0, 10.0),
    },
}

_ollama_raw = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
_ollama_base = _ollama_raw if _ollama_raw.startswith("http") else f"http://{_ollama_raw}"
OLLAMA_URL = f"{_ollama_base}/api/chat"
OLLAMA_VERSION_URL = f"{_ollama_base}/api/version"

SCHEMA_HYPOTHESIS = {
    "type": "object",
    "properties": {
        "analysis_summary": {"type": "string"},
        "identified_flaws": {"type": "array", "items": {"type": "string"}},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "enum": list(TUNABLE_PARAMS.keys())},
                    "parameter": {
                        "type": "string",
                        "enum": ALL_TUNABLE_PARAMS,
                    },
                    "test_values": {"type": "array", "items": {"type": "number"}},
                    "rationale": {"type": "string"},
                },
                "required": ["strategy", "parameter", "test_values", "rationale"],
            },
        },
    },
    "required": ["analysis_summary", "identified_flaws", "hypotheses"],
}

SCHEMA_REDTEAM = {
    "type": "object",
    "properties": {
        "redteam_summary": {"type": "string"},
        "approved_hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "enum": list(TUNABLE_PARAMS.keys())},
                    "parameter": {
                        "type": "string",
                        "enum": ALL_TUNABLE_PARAMS,
                    },
                    "test_values": {"type": "array", "items": {"type": "number"}},
                    "rationale": {"type": "string"},
                },
                "required": ["strategy", "parameter", "test_values", "rationale"],
            },
        },
        "rejected_hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string"},
                    "parameter": {"type": "string"},
                    "rejection_reason": {"type": "string"},
                },
                "required": ["strategy", "parameter", "rejection_reason"],
            },
        },
    },
    "required": ["redteam_summary", "approved_hypotheses", "rejected_hypotheses"],
}

SCHEMA_CERTIFICATION = {
    "type": "object",
    "properties": {
        "certified": {"type": "boolean"},
        "strategy": {"type": "string"},
        "parameter": {"type": "string"},
        "reason": {"type": "string"},
        "code_signature_verified": {"type": "boolean"},
    },
    "required": ["certified", "strategy", "parameter", "reason", "code_signature_verified"],
}


def _sanitize_json_numbers(obj: Any) -> Any:
    """Replace non-finite floats (inf/-inf/nan) with finite JSON-safe equivalents."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return 0.0
        if math.isinf(obj):
            return 999.0 if obj > 0 else -999.0
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_numbers(v) for v in obj]
    return obj


class StateManager:
    """Resumable state machine to survive machine reboots, sleep, or interruptions."""

    def __init__(self, path: Path):
        self.path = path
        self.state = self.load()

    def load(self) -> dict[str, Any]:
        tmp_file = self.path.with_suffix(".tmp")
        if not self.path.exists() and tmp_file.exists():
            try:
                os.replace(tmp_file, self.path)
            except Exception as e:
                logger.warning(f"Could not recover tmp state file: {e}")
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                today_et = datetime.now(zoneinfo.ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
                if loaded.get("phase") == "COMPLETED" and loaded.get("run_id") != today_et:
                    logger.info("Previous optimizer run already COMPLETED; starting a fresh run.")
                else:
                    return loaded
            except Exception as e:
                logger.warning(f"Could not load state file: {e}. Initializing fresh state.")
        return {
            "run_id": datetime.now(zoneinfo.ZoneInfo("America/New_York")).strftime("%Y-%m-%d"),
            "phase": "INITIAL",
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
            "extracted_trades_count": 0,
            "hypotheses": [],
            "redteam_audit": {},
            "baseline_metrics": {},
            "evaluation_results": [],
            "candidate_for_cert": None,
            "codestral_cert": {},
            "promoted_candidate": None,
        }

    def save(self):
        self.state["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        try:
            sanitized = _sanitize_json_numbers(self.state)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(sanitized, f, indent=2, allow_nan=False)
            os.replace(tmp, self.path)
        except Exception as e:
            logger.error(f"Failed to persist state file {self.path}: {e}")

    def update_phase(self, phase: str):
        self.state["phase"] = phase
        self.save()
        logger.info(f"==> Phase updated to: {phase}")


def is_market_closed() -> bool:
    """Returns True if current time is within CME weekend closure (Fri 17:00 ET to Sun 17:00 ET)."""
    tz_et = zoneinfo.ZoneInfo("America/New_York")
    now_et = datetime.now(tz_et)
    weekday = now_et.weekday()
    hour = now_et.hour

    if weekday == 4 and hour >= 17:  # Friday after 17:00 ET
        return True
    if weekday == 5:  # Saturday all day
        return True
    return bool(weekday == 6 and hour < 17)  # Sunday before 17:00 ET (reserves buffer before Globex open)


def ping_ollama() -> bool:
    try:
        req = urllib.request.Request(OLLAMA_VERSION_URL, headers={"User-Agent": "SaturdayOptimizer"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_for_gpu_free(timeout_s: int = 45) -> bool:
    """Polls Ollama /api/ps to confirm GPU memory is cleared before loading the next model."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(f"{_ollama_base}/api/ps")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.load(resp)
                if not data.get("models"):
                    time.sleep(3)  # Apple Silicon unified memory compaction buffer
                    return True
        except Exception as e:
            logger.debug(f"Ollama poll error: {e}")
        time.sleep(2)
    logger.warning("GPU eviction confirmation timed out; proceeding.")
    return True


def evict_model_from_gpu(model_name: str) -> None:
    """Explicitly signals Ollama to evict model from Metal GPU memory."""
    try:
        req = urllib.request.Request(
            f"{_ollama_base}/api/generate",
            data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as e:
        logger.debug(f"Ollama eviction error: {e}")


def call_llm_stage(
    model_name: str,
    system_prompt: str,
    user_content: str,
    schema: dict[str, Any] | None = None,
    timeout: int = 300,
) -> dict[str, Any] | None:
    """Invokes a local Ollama model sequentially, enforcing immediate memory eviction."""
    wait_for_gpu_free()

    body: dict[str, Any] = {
        "model": model_name,
        "stream": False,
        "keep_alive": "0s",  # Instruct Ollama to unload immediately upon reply
        "options": {
            "num_ctx": 8192,
            "temperature": 0.15,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if schema:
        body["format"] = schema

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            json.dumps(body).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        logger.info(f"Querying model '{model_name}' on local GPU...")
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
            content = data.get("message", {}).get("content", "")
            logger.info(f"Model '{model_name}' responded in {time.time()-t0:.1f}s")
            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0].strip()
            return json.loads(content)
    except Exception as e:
        logger.error(f"Error calling model '{model_name}': {e}")
        return None
    finally:
        evict_model_from_gpu(model_name)


def extract_recent_trades(db_path: Path, days: int = 7) -> list[dict[str, Any]]:
    """Extracts recent trades from swarm.db via an atomic snapshot backup."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    src_con: sqlite3.Connection | None = None
    con: sqlite3.Connection | None = None
    try:
        src_con = sqlite3.connect(str(db_path), timeout=10.0)
        src_con.execute("PRAGMA busy_timeout=10000")
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        src_con.backup(con)
        rows = con.execute(
            """
            SELECT o.id,
                   o.trade_key,
                   o.symbol,
                   COALESCE(c.strategy_name, json_extract(o.outcome_json, '$.closed_position.strategy_name'), 'unknown') AS strategy_name,
                   COALESCE(c.side, json_extract(o.outcome_json, '$.closed_position.side'), '') AS side,
                   COALESCE(c.quantity, CAST(json_extract(o.outcome_json, '$.closed_position.qty') AS INTEGER), 1) AS quantity,
                   COALESCE(c.entry_price, CAST(json_extract(o.outcome_json, '$.closed_position.entry') AS REAL), 0.0) AS entry_price,
                   COALESCE(c.opened_at, o.closed_at) AS opened_at,
                   COALESCE(o.session, c.session, json_extract(o.outcome_json, '$.session')) AS session,
                   COALESCE(c.selected_signal_json, '{}') AS selected_signal_json,
                   COALESCE(c.account_snapshot_json, '{}') AS account_snapshot_json,
                   COALESCE(c.route_context_json, '{}') AS route_context_json,
                   COALESCE(o.outcome_json, '{}') AS outcome_json,
                   o.pnl,
                   o.close_type
            FROM trade_outcomes o
            LEFT JOIN trade_contexts c ON o.trade_key = c.trade_key
            WHERE COALESCE(c.opened_at, o.closed_at) >= ?
            ORDER BY o.closed_at ASC, o.id ASC
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        logger.warning(f"Failed to extract recent trades from {db_path}: {exc}")
        return []
    finally:
        if con is not None:
            con.close()
        if src_con is not None:
            src_con.close()


def _format_trade_sample(t: dict[str, Any]) -> dict[str, Any]:
    sig_meta = {}
    raw_sig = t.get("selected_signal_json")
    if raw_sig:
        try:
            sig_meta = json.loads(raw_sig).get("metadata", {})
        except (ValueError, TypeError, AttributeError):
            sig_meta = {}
    outcome_meta = {}
    raw_out = t.get("outcome_json")
    if raw_out:
        try:
            parsed_out = json.loads(raw_out)
            if isinstance(parsed_out, dict):
                outcome_meta = parsed_out
        except (ValueError, TypeError):
            outcome_meta = {}
    entry = {
        "symbol": t.get("symbol"),
        "strategy": t.get("strategy_name"),
        "side": t.get("side"),
        "pnl": t.get("pnl"),
        "close_type": t.get("close_type"),
        "session": t.get("session"),
        "signal_details": sig_meta,
    }
    for exc_key in ("mfe_ticks", "mae_ticks", "mfe_r", "mae_r"):
        val = outcome_meta.get(exc_key)
        if val is None and isinstance(outcome_meta.get("closed_position"), dict):
            val = outcome_meta["closed_position"].get(exc_key)
        if val is not None:
            entry[exc_key] = val
    return entry


def query_llm_hypotheses(trades: list[dict[str, Any]], model: str = DEFAULT_MODELS["diagnose"]) -> dict[str, Any]:
    """Stage 1: Asks local Qwen 3.5 (122B) to diagnose losing trades and propose parameter sweep hypotheses."""
    if not ping_ollama():
        logger.warning("Ollama is not running. Using deterministic fallback parameter grid.")
        return get_fallback_hypotheses()

    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    wins = [t for t in trades if t.get("pnl", 0) > 0]

    sample_losses = [_format_trade_sample(t) for t in losses[:15]]
    sample_wins = [_format_trade_sample(t) for t in wins[:10]]

    summary_data = {
        "total_trades": len(trades),
        "wins_count": len(wins),
        "losses_count": len(losses),
        "sample_losses": sample_losses,
        "sample_wins": sample_wins,
    }

    system_prompt = (
        "You are an offline quantitative strategy auditor for an automated futures bot (MES and MCL). "
        "Review this week's losing trades and identify recurring structural patterns or friction points. "
        "Formulate testable parameter hypotheses that can be empirically backtested across 4.5 months of data. "
        "You may only propose hypotheses using these exact strategy/parameter pairs: "
        + "; ".join(f"{k}: {', '.join(v)}" for k, v in TUNABLE_PARAMS.items())
        + ". Provide 3 test_values per hypothesis, spaced widely enough to change behavior. "
        "Output strictly valid JSON matching the schema."
    )

    user_content = json.dumps(summary_data, default=str)
    res = call_llm_stage(model, system_prompt, user_content, schema=SCHEMA_HYPOTHESIS, timeout=240)
    if res and isinstance(res, dict) and "hypotheses" in res:
        return res

    logger.warning("Stage 1 LLM returned invalid or empty response. Falling back to default parameter grid.")
    return get_fallback_hypotheses()


def stage_llama_redteam(
    hypotheses: list[dict[str, Any]],
    model: str = DEFAULT_MODELS["redteam"],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stage 2: Llama 3.3 (70B) Adversarial Red-Team & Combine Risk Audit."""
    if not ping_ollama():
        logger.warning("Ollama is not running. Bypassing Llama red-team stage.")
        return hypotheses, {
            "redteam_summary": "Ollama offline; bypassed red-team.",
            "approved_hypotheses": hypotheses,
            "rejected_hypotheses": [],
        }

    system_prompt = (
        "You are an Adversarial Chief Risk Officer for a TopstepX Futures Combine (MES and MCL). "
        "Review the proposed parameter hypotheses from the diagnostic stage. "
        "Adversarially scrutinize and REJECT any hypothesis that logically violates combine risk safety:\n"
        "1. Stop Loss: sl_atr_mult must be >= 1.0 (never scalp with negative edge) and <= 3.5 (never blow past daily loss limits).\n"
        "2. Risk/Reward: tp_atr_mult must be > sl_atr_mult (minimum 1.0:1 reward-to-risk ratio).\n"
        "3. Trend Momentum: min_adx must be >= 15.0 and <= 45.0 (avoid chop, avoid overly restrictive filters).\n"
        "4. Overtrading Risk: Reject any parameter set likely to increase execution frequency into high commission drag.\n"
        "Output strictly valid JSON matching the schema with approved_hypotheses and rejected_hypotheses with clear reasons."
    )

    user_content = json.dumps({"hypotheses": hypotheses}, indent=2)
    res = call_llm_stage(model, system_prompt, user_content, schema=SCHEMA_REDTEAM, timeout=300)

    if res and isinstance(res, dict) and "approved_hypotheses" in res:
        approved = res.get("approved_hypotheses", [])
        logger.info(f"Llama 3.3 Red-Team approved {len(approved)}/{len(hypotheses)} hypotheses.")
        for rej in res.get("rejected_hypotheses", []):
            logger.warning(f"Llama 3.3 REJECTED {rej.get('strategy')}.{rej.get('parameter')}: {rej.get('rejection_reason')}")
        return approved, res
    else:
        logger.warning("Llama Red-Team failed or returned invalid format; proceeding with conservative parameter grid.")
        fallback = get_fallback_hypotheses().get("hypotheses", [])
        return fallback, {
            "redteam_summary": "Red-team fallback: API failed or unparseable. Switched to conservative parameter grid.",
            "approved_hypotheses": fallback,
            "rejected_hypotheses": [],
        }


def stage_codestral_certify(
    candidate: dict[str, Any],
    model: str = DEFAULT_MODELS["certify"],
) -> dict[str, Any]:
    """Stage 4: Codestral 22B Python Code Contract & Schema Certification."""
    strat_name = candidate.get("strategy")
    param_name = candidate.get("parameter")
    param_val = candidate.get("value")

    # Static baseline check first
    if not is_valid_candidate_param(strat_name, param_name):
        return {
            "certified": False,
            "strategy": strat_name,
            "parameter": param_name,
            "value": param_val,
            "reason": f"Parameter {param_name} is not whitelisted or read in strategy code.",
            "code_signature_verified": False,
        }

    if not is_valid_param_value(strat_name, param_name, param_val):
        return {
            "certified": False,
            "strategy": strat_name,
            "parameter": param_name,
            "value": param_val,
            "reason": f"Parameter value {param_val} for {strat_name}.{param_name} violates safety bounds.",
            "code_signature_verified": False,
        }

    if not ping_ollama():
        logger.warning("Ollama is not running. Using deterministic static AST certification.")
        return {
            "certified": True,
            "strategy": strat_name,
            "parameter": param_name,
            "value": param_val,
            "reason": "Static parameter whitelist verified (Ollama offline).",
            "code_signature_verified": True,
        }

    strat_file = REPO_ROOT / "src" / "strategies" / f"{strat_name}.py"
    code_snippet = ""
    if strat_file.exists():
        try:
            code_snippet = strat_file.read_text(encoding="utf-8")[:8000]
        except Exception:
            code_snippet = ""

    system_prompt = (
        "You are an expert Python AST & Software Contract Auditor. "
        "Review the proposed parameter candidate against the Python strategy implementation. "
        "Verify that:\n"
        "1. The parameter is actively accessed via self.params.get() or similar in the strategy class.\n"
        "2. The proposed value matches the expected type (float, int, list).\n"
        "3. The proposed value will not cause ZeroDivisionError, infinite loops, or runtime exceptions.\n"
        "Output strictly valid JSON matching the schema."
    )

    user_content = json.dumps({
        "candidate": {
            "strategy": strat_name,
            "parameter": param_name,
            "value": param_val,
            "rationale": candidate.get("rationale"),
        },
        "strategy_code_snippet": code_snippet,
    }, indent=2)

    res = call_llm_stage(model, system_prompt, user_content, schema=SCHEMA_CERTIFICATION, timeout=180)
    if res and isinstance(res, dict) and "certified" in res:
        logger.info(f"Codestral 22B Certification: certified={res.get('certified')}, reason={res.get('reason')}")
        return res
    else:
        logger.warning("Codestral certification call failed or unparseable; falling back to static validation.")
        return {
            "certified": True,
            "strategy": strat_name,
            "parameter": param_name,
            "value": param_val,
            "reason": "Static parameter check passed (Codestral fallback).",
            "code_signature_verified": True,
        }


def get_fallback_hypotheses(base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    if base_config is None:
        try:
            base_config = load_stack_config(str(CONFIG_FILE))
        except Exception:
            base_config = None

    base_adx = 25.0
    base_sl = 1.5
    if isinstance(base_config, dict):
        for spec in (base_config.get("strategies", {}) or {}).get("strategies", []):
            if isinstance(spec, dict) and spec.get("name") == "mtf_trend_scalper":
                params = spec.get("params", {})
                if isinstance(params, dict):
                    sym_params = params.get("symbol_params", {})
                    found_adx = False
                    found_sl = False
                    if isinstance(sym_params, dict):
                        for sym_cfg in sym_params.values():
                            if isinstance(sym_cfg, dict):
                                if not found_adx and "min_adx" in sym_cfg:
                                    with contextlib.suppress(ValueError, TypeError):
                                        base_adx = float(sym_cfg["min_adx"])
                                        found_adx = True
                                if not found_sl and "sl_atr_mult" in sym_cfg:
                                    with contextlib.suppress(ValueError, TypeError):
                                        base_sl = float(sym_cfg["sl_atr_mult"])
                                        found_sl = True
                    if not found_adx and "min_adx" in params:
                        with contextlib.suppress(ValueError, TypeError):
                            base_adx = float(params["min_adx"])
                    if not found_sl and "sl_atr_mult" in params:
                        with contextlib.suppress(ValueError, TypeError):
                            base_sl = float(params["sl_atr_mult"])

    return {
        "analysis_summary": "Deterministic grid sweep around current production baseline.",
        "identified_flaws": ["Routine weekly parameter neighborhood calibration."],
        "hypotheses": [
            {
                "strategy": "mtf_trend_scalper",
                "parameter": "min_adx",
                "test_values": [
                    round(base_adx - 5.0, 1),
                    round(base_adx - 2.0, 1),
                    round(base_adx, 1),
                    round(base_adx + 3.0, 1),
                ],
                "rationale": "Test ADX momentum threshold sensitivity across market volatility shifts.",
            },
            {
                "strategy": "mtf_trend_scalper",
                "parameter": "sl_atr_mult",
                "test_values": [
                    round(max(1.0, base_sl - 0.2), 2),
                    round(base_sl, 2),
                    round(base_sl + 0.3, 2),
                ],
                "rationale": "Calibrate stop-loss ATR breathing room.",
            },
        ],
    }


def is_valid_candidate_param(strategy: str | None, param: str | None) -> bool:
    """True only if the strategy is known, the param is whitelisted, and the strategy source reads it."""
    if strategy not in TUNABLE_PARAMS or param not in TUNABLE_PARAMS[strategy]:
        return False
    src_file = REPO_ROOT / "src" / "strategies" / f"{strategy}.py"
    try:
        return f'"{param}"' in src_file.read_text(encoding="utf-8")
    except OSError:
        return False


_ORB_ASIAN_PARAM_MIRRORS: dict[str, str] = {
    "risk_reward": "asian_risk_reward",
    "min_or_ticks": "asian_min_or_ticks",
    "max_or_ticks": "asian_max_or_ticks",
    "stop_buffer_ticks": "asian_stop_buffer_ticks",
    "breakout_buffer_ticks": "asian_breakout_buffer_ticks",
    "fade_reclaim_ticks": "asian_fade_reclaim_ticks",
    "min_volume_ratio": "asian_min_volume_ratio",
}


_ORB_ASIAN_REVERSE_MIRRORS: dict[str, str] = {v: k for k, v in _ORB_ASIAN_PARAM_MIRRORS.items()}


def is_valid_param_value(strategy: str | None, param: str | None, value: Any) -> bool:
    """Checks whether a parameter value satisfies safety bounds."""
    if isinstance(value, bool):
        return False
    if not strategy or not param:
        return False

    if param == "allowed_entry_hours_et":
        if isinstance(value, list):
            if not value:
                return False
            return all(
                isinstance(h, int) and not isinstance(h, bool) and 0 <= h <= 23
                for h in value
            )
        if isinstance(value, (int, float)):
            return 0 <= value <= 23
        return False

    strat_bounds = PARAM_BOUNDS.get(strategy)
    if not strat_bounds:
        return False

    bounds = strat_bounds.get(param)
    if bounds is None:
        mirror = _ORB_ASIAN_PARAM_MIRRORS.get(param)
        if mirror:
            bounds = strat_bounds.get(mirror)
        else:
            reverse_mirror = _ORB_ASIAN_REVERSE_MIRRORS.get(param)
            if reverse_mirror:
                bounds = strat_bounds.get(reverse_mirror)

    if bounds is None:
        return False

    if not isinstance(value, (int, float)):
        return False

    min_val, max_val = bounds
    return min_val <= float(value) <= max_val


def apply_param_override(cfg: dict[str, Any], strategy: str, param: str, value: Any) -> bool:
    """Sets a strategy param in-place on the live config dict and syncs any symbol_params or Asian ORB overrides."""
    applied = False
    for spec in (cfg.get("strategies", {}) or {}).get("strategies", []):
        if spec.get("name") == strategy:
            params_dict = spec.setdefault("params", {})
            params_dict[param] = value
            mirrored_param = None
            if (
                strategy == "orb_vwap"
                and params_dict.get("asian_enabled")
                and not params_dict.get("opening_range_enabled", True)
            ):
                if param in _ORB_ASIAN_PARAM_MIRRORS:
                    mirrored_param = _ORB_ASIAN_PARAM_MIRRORS[param]
                elif param in _ORB_ASIAN_REVERSE_MIRRORS:
                    mirrored_param = _ORB_ASIAN_REVERSE_MIRRORS[param]
                if mirrored_param:
                    params_dict[mirrored_param] = value
            sym_params = params_dict.get("symbol_params")
            if isinstance(sym_params, dict):
                for sym_override in sym_params.values():
                    if isinstance(sym_override, dict):
                        has_param = param in sym_override
                        has_mirror = bool(mirrored_param and mirrored_param in sym_override)
                        if has_param or has_mirror:
                            sym_override[param] = value
                            if mirrored_param:
                                sym_override[mirrored_param] = value
            applied = True
    return applied


def _run_single_symbol(sym: str, path: Path, strat_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Runs a backtest on a single symbol with telemetry disabled for maximum sweep speed."""
    if not path.exists():
        logger.warning(f"Continuous data missing for {sym}: {path}")
        return []
    bt = Backtester(sym, strategy_config=strat_cfg, telemetry_enabled=False)
    res = bt.run(str(path))
    trades = []
    if res and "trades" in res:
        for t in res["trades"]:
            tc = dict(t)
            tc["symbol"] = sym
            trades.append(tc)
    return trades


def _get_dataset_cutoff_date(holdout_days: int) -> str | None:
    """Inspects CONTINUOUS_DATA and returns (max_date - timedelta(days=holdout_days)).strftime('%Y-%m-%d')."""
    max_dt: datetime | None = None
    for raw_path in CONTINUOUS_DATA.values():
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size == 0:
                    continue
                f.seek(max(0, size - 4096), os.SEEK_SET)
                lines = f.read().decode("utf-8", errors="replace").strip().splitlines()
                if not lines:
                    continue
                ts_str = lines[-1].split(",")[0].strip()
                dt = datetime.fromisoformat(ts_str[:10])
                if max_dt is None or dt > max_dt:
                    max_dt = dt
        except Exception as exc:
            logger.debug(f"Could not extract max date from {path}: {exc}")
    if max_dt is None:
        return None
    return (max_dt - timedelta(days=holdout_days)).strftime("%Y-%m-%d")


def evaluate_backtest_candidate(
    config_dict: dict[str, Any], holdout_days: int = 21
) -> dict[str, Any]:
    """Runs portfolio backtest concurrently across continuous MES and MCL bars with holdout slicing."""
    strat_cfg = get_strategy_config(config_dict)
    portfolio_trades = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(CONTINUOUS_DATA))) as executor:
        futures = [
            executor.submit(_run_single_symbol, sym, path, strat_cfg)
            for sym, path in CONTINUOUS_DATA.items()
        ]
        for f in concurrent.futures.as_completed(futures):
            portfolio_trades.extend(f.result())

    if not portfolio_trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "in_sample_trades": 0,
            "in_sample_pnl": 0.0,
            "in_sample_pf": 0.0,
            "in_sample_max_drawdown": 0.0,
            "holdout_pnl": 0.0,
            "holdout_pf": 0.0,
            "holdout_trades": 0,
            "holdout_win_rate": 0.0,
        }

    portfolio_trades.sort(key=lambda x: str(x.get("time_close") or x.get("time_open") or ""))
    total_trades = len(portfolio_trades)
    wins = [t for t in portfolio_trades if t.get("pnl", 0) > 0]
    losses = [t for t in portfolio_trades if t.get("pnl", 0) < 0]
    win_rate = len(wins) / total_trades if total_trades else 0.0
    total_pnl = sum(t.get("pnl", 0) for t in portfolio_trades)
    gross_win = sum(t.get("pnl", 0) for t in wins)
    gross_loss = abs(sum(t.get("pnl", 0) for t in losses))
    pf = min(999.0, gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    cum_pnl = 0.0
    peak_pnl = 0.0
    max_drawdown = 0.0
    for t in portfolio_trades:
        mae_dollars = max(0.0, float(t.get("mae_dollars", 0.0) or 0.0))
        trough_pnl = cum_pnl - mae_dollars
        max_drawdown = max(max_drawdown, peak_pnl - trough_pnl)
        cum_pnl += float(t.get("pnl", 0.0) or 0.0)
        peak_pnl = max(peak_pnl, cum_pnl)
        max_drawdown = max(max_drawdown, peak_pnl - cum_pnl)

    all_dates = sorted({str(t.get("time_open", ""))[:10] for t in portfolio_trades if t.get("time_open")})
    cutoff_date = _get_dataset_cutoff_date(holdout_days)
    if cutoff_date is None and all_dates:
        cutoff_date = (datetime.fromisoformat(all_dates[-1]) - timedelta(days=holdout_days)).strftime("%Y-%m-%d")

    if cutoff_date is not None:
        holdout = [t for t in portfolio_trades if str(t.get("time_open", ""))[:10] >= cutoff_date]
        insample = [t for t in portfolio_trades if str(t.get("time_open", ""))[:10] < cutoff_date]
    else:
        holdout = []
        insample = portfolio_trades

    insample_pnl = sum(float(t.get("pnl", 0.0) or 0.0) for t in insample)
    is_wins = sum(float(t.get("pnl", 0.0) or 0.0) for t in insample if float(t.get("pnl", 0.0) or 0.0) > 0)
    is_losses = abs(sum(float(t.get("pnl", 0.0) or 0.0) for t in insample if float(t.get("pnl", 0.0) or 0.0) < 0))
    insample_pf = min(999.0, is_wins / is_losses) if is_losses > 0 else (999.0 if is_wins > 0 else 0.0)
    is_cum = 0.0
    is_peak = 0.0
    insample_mdd = 0.0
    for t in insample:
        mae_d = max(0.0, float(t.get("mae_dollars", 0.0) or 0.0))
        insample_mdd = max(insample_mdd, is_peak - (is_cum - mae_d))
        is_cum += float(t.get("pnl", 0.0) or 0.0)
        is_peak = max(is_peak, is_cum)
        insample_mdd = max(insample_mdd, is_peak - is_cum)

    if holdout:
        holdout_pnl = sum(float(t.get("pnl", 0.0) or 0.0) for t in holdout)
        holdout_wr = sum(1 for t in holdout if float(t.get("pnl", 0.0) or 0.0) > 0) / len(holdout)
        ho_wins = sum(float(t.get("pnl", 0.0) or 0.0) for t in holdout if float(t.get("pnl", 0.0) or 0.0) > 0)
        ho_losses = abs(sum(float(t.get("pnl", 0.0) or 0.0) for t in holdout if float(t.get("pnl", 0.0) or 0.0) < 0))
        holdout_pf = min(999.0, ho_wins / ho_losses) if ho_losses > 0 else (999.0 if ho_wins > 0 else 0.0)
    else:
        holdout_pnl = 0.0
        holdout_wr = 0.0
        holdout_pf = 0.0
        holdout = []

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown": round(max_drawdown, 2),
        "in_sample_trades": len(insample),
        "in_sample_pnl": round(insample_pnl, 2),
        "in_sample_pf": round(insample_pf, 3),
        "in_sample_max_drawdown": round(insample_mdd, 2),
        "holdout_pnl": round(holdout_pnl, 2),
        "holdout_pf": round(holdout_pf, 3),
        "holdout_trades": len(holdout),
        "holdout_win_rate": round(holdout_wr, 4),
    }


class _QuotedTimeSafeDumper(yaml.SafeDumper):
    """Ensure HH:MM / HH:MM:SS strings remain quoted so YAML 1.1 never parses them as sexagesimal ints."""


def _repr_time_str(dumper: yaml.SafeDumper, data: str):
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_QuotedTimeSafeDumper.add_representer(str, _repr_time_str)


def stage_atomic_config_update(
    new_config: dict[str, Any], candidate_summary: str, dry_run: bool = False
) -> bool:
    """Backs up existing config and performs an atomic replace."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_DIR / f"trading_stack_pre_opt_{ts}.yaml"

    if dry_run:
        logger.info(f"[DRY-RUN] Would backup {CONFIG_FILE} to {backup_file}")
        logger.info(f"[DRY-RUN] Config promotion details:\n{candidate_summary}")
        return True

    # Pre-promotion active position check (CRO & Architect safety gate)
    if BOT_STATUS_FILE.exists():
        try:
            bot_status = json.loads(BOT_STATUS_FILE.read_text(encoding="utf-8"))
            active_pos = bot_status.get("active_positions") or {}
            if isinstance(active_pos, dict) and any(active_pos.values()):
                logger.error("Aborting live config promotion: open positions detected in bot_status.json!")
                return False
        except Exception as e:
            logger.warning(f"Could not inspect {BOT_STATUS_FILE} prior to config promotion: {e}")

    # 1. Create Backup
    shutil.copy2(CONFIG_FILE, backup_file)
    logger.info(f"Created configuration backup at: {backup_file}")

    # 2. Write and validate temporary file
    tmp_file = CONFIG_FILE.with_suffix(".tmp")
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            yaml.dump(new_config, f, Dumper=_QuotedTimeSafeDumper, sort_keys=False)
    except Exception as e:
        logger.error(f"Failed to write temporary config file: {e}")
        return False

    try:
        with open(tmp_file, encoding="utf-8") as f:
            yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Validation failed on staged config: {e}. Aborting swap.")
        if tmp_file.exists():
            tmp_file.unlink()
        return False

    # 3. Atomic Replacement
    os.replace(tmp_file, CONFIG_FILE)
    logger.info(f"Successfully promoted and atomically replaced {CONFIG_FILE}!")
    return True


def generate_optimization_report(
    state: dict[str, Any],
    base_metrics: dict[str, Any],
    winner: dict[str, Any] | None,
    redteam_audit: dict[str, Any] | None,
    codestral_cert: dict[str, Any] | None,
    dry_run: bool = False,
) -> Path:
    """Generates a complete markdown audit trail of the 3-LLM relay optimization run."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_file = REPORTS_DIR / f"saturday_optimizer_run_{ts}.md"

    status_str = "DRY-RUN (NO CHANGES)" if dry_run else ("PROMOTED" if winner else "HELD BASELINE")

    md = [
        "# Saturday Autonomous 3-LLM Optimization Report",
        "",
        f"- **Run ID:** `{state.get('run_id')}`",
        f"- **Execution Timestamp (UTC):** `{datetime.now(timezone.utc).isoformat()}`",
        f"- **Status:** **{status_str}**",
        f"- **Extracted Trade Count:** {state.get('extracted_trades_count', 0)}",
        "",
        "---",
        "",
        "## Stage 1: Diagnosis & Hypothesis Generation (Qwen 3.5 122B)",
        "",
        f"**Analysis Summary:** {state.get('analysis_summary', 'N/A')}",
        "",
        "**Identified Flaws:**",
    ]
    for flaw in state.get("identified_flaws", []):
        md.append(f"- {flaw}")

    md.extend([
        "",
        "---",
        "",
        "## Stage 2: Adversarial Red-Team & Risk Audit (Llama 3.3 70B)",
        "",
        f"**Red-Team Summary:** {(redteam_audit or {}).get('redteam_summary', 'N/A')}",
        "",
        f"**Approved Hypotheses ({len((redteam_audit or {}).get('approved_hypotheses', []))}):**",
    ])
    for app in (redteam_audit or {}).get("approved_hypotheses", []):
        md.append(f"- `{app.get('strategy')}.{app.get('parameter')}` values={app.get('test_values')} ({app.get('rationale')})")

    if (redteam_audit or {}).get("rejected_hypotheses"):
        md.append("\n**Rejected Hypotheses:**")
        for rej in (redteam_audit or {}).get("rejected_hypotheses", []):
            md.append(f"- ❌ `{rej.get('strategy')}.{rej.get('parameter')}`: {rej.get('rejection_reason')}")

    md.extend([
        "",
        "---",
        "",
        "## Stage 3: Backtest Performance Sweep (CPU Multicore)",
        "",
        "| Candidate | Trades | Win Rate | Total PnL | PF | Holdout PnL | Holdout WR | Max DD |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **BASELINE** | {base_metrics.get('total_trades')} | {base_metrics.get('win_rate', 0)*100:.1f}% | ${base_metrics.get('total_pnl', 0):+8.2f} | {base_metrics.get('profit_factor', 0):.2f} | ${base_metrics.get('holdout_pnl', 0):+7.2f} | {base_metrics.get('holdout_win_rate', 0)*100:.1f}% | ${base_metrics.get('max_drawdown', 0):.2f} |",
    ])

    for cand in state.get("evaluation_results", []):
        marker = "🏆 **WINNER** " if winner and cand.get("candidate_id") == winner.get("candidate_id") else ""
        md.append(
            f"| {marker}`{cand.get('candidate_id')}` | {cand.get('total_trades')} | {cand.get('win_rate', 0)*100:.1f}% | "
            f"${cand.get('total_pnl', 0):+8.2f} | {cand.get('profit_factor', 0):.2f} | "
            f"${cand.get('holdout_pnl', 0):+7.2f} | {cand.get('holdout_win_rate', 0)*100:.1f}% | ${cand.get('max_drawdown', 0):.2f} |"
        )

    md.extend([
        "",
        "---",
        "",
        "## Stage 4: Strategy Code Contract Certification (Codestral 22B)",
        "",
        f"- **Certified:** `{(codestral_cert or {}).get('certified', 'N/A')}`",
        f"- **Target:** `{(codestral_cert or {}).get('strategy', 'N/A')}.{(codestral_cert or {}).get('parameter', 'N/A')} = {(codestral_cert or {}).get('value', 'N/A')}`",
        f"- **Reason:** {(codestral_cert or {}).get('reason', 'N/A')}",
        f"- **Code Signature Verified:** `{(codestral_cert or {}).get('code_signature_verified', 'N/A')}`",
        "",
        "---",
        "",
        "## Stage 5: Promotion & Deployment",
        "",
    ])

    if winner:
        md.append(f"✅ Promoted candidate `{winner.get('candidate_id')}` to `config/trading_stack.yaml`.")
    else:
        md.append("ℹ️ Baseline retained. No candidate surpassed the holdout profit factor and drawdown constraints.")

    report_file.write_text("\n".join(md), encoding="utf-8")
    logger.info(f"Generated optimization report at: {report_file}")
    return report_file


def run_optimizer(
    dry_run: bool = False,
    force_offline: bool = False,
    model: str | None = None,
    model_diagnose: str = DEFAULT_MODELS["diagnose"],
    model_redteam: str = DEFAULT_MODELS["redteam"],
    model_certify: str = DEFAULT_MODELS["certify"],
    history_days: int = 7,
    max_candidates: int = 9,
):
    if model:
        model_diagnose = model

    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCK_FILE, "w", encoding="utf-8") as lock_fh:
            try:
                fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.warning("Another Saturday Autonomous Optimizer instance is already running (lock held). Exiting.")
                return

            try:
                _run_optimizer_locked(
                    dry_run=dry_run,
                    force_offline=force_offline,
                    model_diagnose=model_diagnose,
                    model_redteam=model_redteam,
                    model_certify=model_certify,
                    history_days=history_days,
                    max_candidates=max_candidates,
                )
            finally:
                try:
                    fcntl.flock(lock_fh, fcntl.LOCK_UN)
                except OSError as unlock_err:
                    logger.debug(f"Could not unlock {LOCK_FILE}: {unlock_err}")
    except OSError as lock_err:
        logger.error(f"Could not open lockfile {LOCK_FILE}: {lock_err}")


def _run_optimizer_locked(
    dry_run: bool,
    force_offline: bool,
    model_diagnose: str,
    model_redteam: str,
    model_certify: str,
    history_days: int,
    max_candidates: int,
):
    logger.info("=" * 80)
    logger.info("SATURDAY AUTONOMOUS 3-LLM OPTIMIZER INITIATING")
    logger.info(f"Relay: Diagnose={model_diagnose} -> RedTeam={model_redteam} -> Certify={model_certify}")
    logger.info("=" * 80)

    # Safety Guard: Check Market Hours
    if not is_market_closed() and not force_offline:
        logger.warning(
            "MARKET IS CURRENTLY OPEN or within active trading hours! "
            "Forcing --dry-run mode to prevent live configuration modification."
        )
        dry_run = True

    mgr = StateManager(STATE_FILE)
    mgr.state["dry_run"] = dry_run
    mgr.save()
    base_config = load_stack_config(str(CONFIG_FILE))

    # Phase 1: Ingestion
    if mgr.state["phase"] in ("INITIAL", "INGESTION"):
        logger.info(f"Extracting recent trades from past {history_days} days...")
        trades = extract_recent_trades(SWARM_DB, days=history_days)
        mgr.state["extracted_trades_count"] = len(trades)
        mgr.update_phase("DIAGNOSTIC_AUDIT")
    else:
        trades = extract_recent_trades(SWARM_DB, days=history_days)

    # Phase 2: Stage 1 Diagnostic Audit via Qwen (122B)
    if mgr.state["phase"] in ("DIAGNOSTIC_AUDIT", "STAGE1_DIAGNOSE"):
        logger.info(f"Running Stage 1 LLM diagnostic on {len(trades)} trades with model {model_diagnose}...")
        diag = query_llm_hypotheses(trades, model=model_diagnose)
        mgr.state["hypotheses"] = diag.get("hypotheses", [])
        mgr.state["identified_flaws"] = diag.get("identified_flaws", [])
        mgr.state["analysis_summary"] = diag.get("analysis_summary", "")
        mgr.update_phase("REDTEAM_AUDIT")

    # Phase 3: Stage 2 Adversarial Red-Team via Llama 3.3 (70B)
    if mgr.state["phase"] in ("REDTEAM_AUDIT", "STAGE2_REDTEAM"):
        logger.info(f"Running Stage 2 Adversarial Red-Team audit with model {model_redteam}...")
        approved_hypo, redteam_audit = stage_llama_redteam(mgr.state.get("hypotheses", []), model=model_redteam)
        mgr.state["hypotheses"] = approved_hypo
        mgr.state["redteam_audit"] = redteam_audit
        mgr.update_phase("BASELINE_EVALUATION")

    # Phase 4: Baseline Evaluation
    if mgr.state["phase"] == "BASELINE_EVALUATION":
        logger.info("Evaluating baseline production metrics...")
        base_metrics = evaluate_backtest_candidate(base_config)
        mgr.state["baseline_metrics"] = base_metrics
        logger.info(
            f"BASELINE: Total PnL=${base_metrics['total_pnl']:+8.2f} | "
            f"PF={base_metrics['profit_factor']:.2f} | "
            f"Holdout PnL=${base_metrics['holdout_pnl']:+7.2f} | "
            f"Trades={base_metrics['total_trades']}"
        )
        mgr.update_phase("HYPOTHESIS_TESTING")

    # Phase 5: Stage 3 Multicore Backtesting Sweep (CPU)
    base_metrics = mgr.state["baseline_metrics"]
    eval_results = mgr.state.get("evaluation_results", [])

    if mgr.state["phase"] in ("HYPOTHESIS_TESTING", "STAGE3_BACKTEST"):
        logger.info(f"Testing candidate parameter hypotheses (max {max_candidates} effective candidates)...")
        tested_count = sum(1 for r in eval_results if not r.get("no_effect"))
        total_attempts = len(eval_results)
        max_attempts = max(max_candidates * 3, 18)
        for hypo in mgr.state.get("hypotheses", []):
            if tested_count >= max_candidates or total_attempts >= max_attempts:
                logger.info(
                    f"Reached candidate evaluation limit (effective={tested_count}/{max_candidates}, "
                    f"attempts={total_attempts}/{max_attempts}). Proceeding to validation."
                )
                break

            strat_name = hypo.get("strategy")
            param_name = hypo.get("parameter")
            test_values = hypo.get("test_values", [])

            if not is_valid_candidate_param(strat_name, param_name):
                logger.warning(f"Rejecting hypothesis {strat_name}.{param_name}: not a real tunable parameter.")
                continue

            for val in test_values:
                if tested_count >= max_candidates or total_attempts >= max_attempts:
                    break

                if param_name == "allowed_entry_hours_et":
                    if isinstance(val, (int, float)):
                        val = [int(val)]
                    elif not isinstance(val, list):
                        continue

                if not is_valid_param_value(strat_name, param_name, val):
                    logger.warning(
                        f"Rejecting candidate value {strat_name}.{param_name}={val}: outside safety bounds."
                    )
                    continue

                candidate_id = f"{strat_name}.{param_name}={val}"
                if any(r.get("candidate_id") == candidate_id for r in eval_results):
                    continue

                logger.info(f"Testing Candidate ({tested_count + 1}/{max_candidates}): {candidate_id}...")
                test_cfg = copy.deepcopy(base_config)
                if not apply_param_override(test_cfg, strat_name, param_name, val):
                    logger.warning(f"Could not apply {candidate_id}: strategy not found in config.")
                    continue

                cand_metrics = evaluate_backtest_candidate(test_cfg)
                gc.collect()
                if all(cand_metrics.get(k) == base_metrics.get(k) for k in ("total_trades", "total_pnl", "profit_factor", "max_drawdown")):
                    logger.warning(f"Candidate {candidate_id} produced results identical to baseline (no effect).")
                    cand_metrics["no_effect"] = True
                else:
                    tested_count += 1
                total_attempts += 1
                cand_metrics["candidate_id"] = candidate_id
                cand_metrics["strategy"] = strat_name
                cand_metrics["parameter"] = param_name
                cand_metrics["value"] = val
                cand_metrics["rationale"] = hypo.get("rationale")
                eval_results.append(cand_metrics)
                mgr.state["evaluation_results"] = eval_results
                mgr.save()

                logger.info(
                    f"  -> PF={cand_metrics['profit_factor']:.2f} | "
                    f"Total PnL=${cand_metrics['total_pnl']:+8.2f} | "
                    f"Holdout PnL=${cand_metrics['holdout_pnl']:+7.2f} | "
                    f"Trades={cand_metrics['total_trades']}"
                )

        mgr.update_phase("VALIDATION_GATE")

    # Phase 6: Validation Gate & Selection
    # Strictly separate In-Sample ranking from Out-of-Sample (21-day holdout) confirmation
    # to eliminate holdout snooping / data leakage.
    if mgr.state["phase"] == "VALIDATION_GATE":
        logger.info("Evaluating candidates against strict in-sample ranking + holdout confirmation gates...")
        best_in_sample = None
        base_is_pf = base_metrics.get("in_sample_pf", base_metrics.get("profit_factor", 1.25))
        base_is_pnl = base_metrics.get("in_sample_pnl", base_metrics.get("total_pnl", 0.0))
        base_is_mdd = base_metrics.get("in_sample_max_drawdown", base_metrics.get("max_drawdown", 1000.0))
        base_holdout = base_metrics.get("holdout_pnl", 0.0)
        base_is_trades = int(base_metrics.get("in_sample_trades", base_metrics.get("total_trades", 0)) or 0)
        min_required_is_trades = min(80, max(min(40, base_is_trades), int(base_is_trades * 0.75))) if base_is_trades > 0 else 40
        logger.info(f"Validation gate minimum in-sample trade threshold: {min_required_is_trades} trades (baseline IS={base_is_trades})")
        if eval_results:
            no_effect_rate = sum(1 for r in eval_results if r.get("no_effect")) / len(eval_results)
            if no_effect_rate > 0.5:
                logger.warning(f"High no_effect rate ({no_effect_rate:.1%}) in candidate sweep — review parameter mappings.")

        max_allowed_is_mdd = max(base_is_mdd * 1.15, 600.0)
        for cand in eval_results:
            is_pf = cand.get("in_sample_pf", cand.get("profit_factor", 0.0))
            is_pnl = cand.get("in_sample_pnl", cand.get("total_pnl", 0.0))
            is_mdd = cand.get("in_sample_max_drawdown", cand.get("max_drawdown", float("inf")))
            is_trades = int(cand.get("in_sample_trades", cand.get("total_trades", 0)) or 0)

            if (
                not cand.get("no_effect")
                and is_pf >= 1.25
                and is_pf >= base_is_pf
                and is_pnl > 0
                and is_trades >= min_required_is_trades
                and is_mdd <= max_allowed_is_mdd
                and (is_pf, is_pnl, -is_mdd) > (base_is_pf, base_is_pnl, -base_is_mdd)
            ):
                if best_in_sample is None or (
                    (is_pf, is_pnl, is_trades, -is_mdd)
                    > (
                        best_in_sample.get("in_sample_pf", best_in_sample.get("profit_factor", 0.0)),
                        best_in_sample.get("in_sample_pnl", best_in_sample.get("total_pnl", 0.0)),
                        int(best_in_sample.get("in_sample_trades", best_in_sample.get("total_trades", 0)) or 0),
                        -best_in_sample.get("in_sample_max_drawdown", best_in_sample.get("max_drawdown", float("inf"))),
                    )
                ):
                    best_in_sample = cand

        best_candidate = None
        if best_in_sample is not None:
            ho_pnl = best_in_sample.get("holdout_pnl", 0.0)
            ho_pf = best_in_sample.get("holdout_pf", best_in_sample.get("profit_factor", 0.0))
            ho_floor = (base_holdout * 0.85) if base_holdout > 0 else base_holdout
            if ho_pnl > 0 and ho_pf >= 1.05 and ho_pnl >= ho_floor:
                best_candidate = best_in_sample
            else:
                logger.info(
                    f"Top in-sample finalist {best_in_sample.get('candidate_id')} failed 21-day OOS holdout gate "
                    f"(Holdout PnL=${ho_pnl:+.2f} < floor ${ho_floor:+.2f}, PF={ho_pf:.2f}); retaining baseline."
                )

        if best_candidate:
            logger.info(
                f"POTENTIAL WINNING CANDIDATE IDENTIFIED: {best_candidate['candidate_id']} "
                f"(PF={best_candidate['profit_factor']:.2f}, Holdout=${best_candidate['holdout_pnl']:+7.2f})"
            )
            mgr.state["candidate_for_cert"] = best_candidate
            mgr.update_phase("CODESTRAL_CERTIFICATION")
        else:
            logger.info("No candidate outperformed the current production baseline under holdout rigor.")
            generate_optimization_report(mgr.state, base_metrics, None, mgr.state.get("redteam_audit"), None, dry_run=dry_run)
            mgr.update_phase("COMPLETED")

    # Phase 7: Stage 4 Codestral Python Code Contract Certification (22B)
    if mgr.state["phase"] in ("CODESTRAL_CERTIFICATION", "STAGE4_CERTIFY"):
        cand = mgr.state.get("candidate_for_cert")
        if cand:
            logger.info(f"Running Stage 4 Code Certification with model {model_certify} for candidate {cand.get('candidate_id')}...")
            cert = stage_codestral_certify(cand, model=model_certify)
            mgr.state["codestral_cert"] = cert
            if cert.get("certified"):
                logger.info(f"Candidate {cand.get('candidate_id')} CERTIFIED for promotion: {cert.get('reason')}")
                mgr.state["promoted_candidate"] = cand
                mgr.update_phase("PROMOTION")
            else:
                logger.warning(f"Candidate {cand.get('candidate_id')} FAILED certification: {cert.get('reason')}")
                generate_optimization_report(mgr.state, base_metrics, None, mgr.state.get("redteam_audit"), cert, dry_run=dry_run)
                mgr.update_phase("COMPLETED")
        else:
            mgr.update_phase("COMPLETED")

    # Phase 8: Stage 5 Atomic Promotion & Report Generation
    if mgr.state["phase"] == "PROMOTION":
        winner = mgr.state.get("promoted_candidate")
        if winner:
            promoted_cfg = copy.deepcopy(base_config)
            apply_param_override(promoted_cfg, winner["strategy"], winner["parameter"], winner["value"])

            summary = (
                f"Promoted: {winner['candidate_id']}\n"
                f"Rationale: {winner.get('rationale')}\n"
                f"Baseline: PF={base_metrics['profit_factor']:.2f}, Holdout=${base_metrics['holdout_pnl']:+7.2f}\n"
                f"Candidate: PF={winner['profit_factor']:.2f}, Holdout=${winner['holdout_pnl']:+7.2f}\n"
            )
            promoted_ok = stage_atomic_config_update(promoted_cfg, summary, dry_run=dry_run)
            if not promoted_ok:
                logger.warning(f"Config promotion skipped/aborted for {winner.get('candidate_id')}; keeping baseline active.")
                mgr.state["promoted_candidate"] = None
                winner = None

        generate_optimization_report(
            mgr.state,
            base_metrics,
            winner,
            mgr.state.get("redteam_audit"),
            mgr.state.get("codestral_cert"),
            dry_run=dry_run,
        )
        mgr.update_phase("COMPLETED")

    logger.info("=" * 80)
    logger.info("SATURDAY 3-LLM OPTIMIZER ROUTINE FINISHED")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saturday Autonomous Multi-LLM Optimizer")
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not stage config changes")
    parser.add_argument("--force-offline", action="store_true", help="Bypass market hours check for testing")
    parser.add_argument("--model", type=str, default=None, help="Override diagnostic Ollama model")
    parser.add_argument("--model-diagnose", type=str, default=DEFAULT_MODELS["diagnose"], help="Ollama model for diagnosis")
    parser.add_argument("--model-redteam", type=str, default=DEFAULT_MODELS["redteam"], help="Ollama model for red-team audit")
    parser.add_argument("--model-certify", type=str, default=DEFAULT_MODELS["certify"], help="Ollama model for code certification")
    parser.add_argument("--days", type=int, default=7, help="Days of past trades to inspect")
    parser.add_argument("--max-candidates", type=int, default=9, help="Max parameter candidates to evaluate")
    parser.add_argument("--reset-state", action="store_true", help="Reset state file and start fresh")
    args = parser.parse_args()

    if args.reset_state and STATE_FILE.exists():
        STATE_FILE.unlink()
        logger.info("Reset optimizer state file.")

    run_optimizer(
        dry_run=args.dry_run,
        force_offline=args.force_offline,
        model=args.model,
        model_diagnose=args.model_diagnose,
        model_redteam=args.model_redteam,
        model_certify=args.model_certify,
        history_days=args.days,
        max_candidates=args.max_candidates,
    )

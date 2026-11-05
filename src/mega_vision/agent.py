from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import (  # type: ignore[import-not-found]
        ClaudeAgentOptions,
        query,  # pyright: ignore[reportAssignmentType]
    )

    _HAS_SDK = True
except Exception:
    ClaudeAgentOptions = None  # type: ignore[assignment]
    _HAS_SDK = False

    async def query(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        if False:
            yield None


@dataclass
class MegaVisionConfig:
    mode: str = "shadow"
    external_calls_enabled: bool = False
    shadow_model: str = "claude-opus-4-1"
    live_model: str = "claude-sonnet-4-0"
    subscription_mode: bool = True
    kill_switch_env_var: str = "MEGA_VISION_KILL_SWITCH"
    max_retries_per_decision: int = 2
    fallback_on_timeout_seconds: int = 30
    claude_command: list[str] | None = None
    codex_command: list[str] | None = None
    gemini_command: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MegaVisionConfig:
        data = data or {}
        return cls(
            mode=str(data.get("mode", "shadow")),
            external_calls_enabled=bool(data.get("external_calls_enabled", False)),
            shadow_model=str(data.get("shadow_model", "claude-opus-4-1")),
            live_model=str(data.get("live_model", "claude-sonnet-4-0")),
            subscription_mode=bool(data.get("subscription_mode", True)),
            kill_switch_env_var=str(data.get("kill_switch_env_var", "MEGA_VISION_KILL_SWITCH")),
            max_retries_per_decision=int(data.get("max_retries_per_decision", 2)),
            fallback_on_timeout_seconds=int(data.get("fallback_on_timeout_seconds", 30)),
            claude_command=list(data.get("claude", {}).get("command", ["claude", "-p"])),
            codex_command=list(data.get("codex", {}).get("command", ["codex", "exec", "--full-auto"])),
            gemini_command=list(data.get("gemini", {}).get("command", ["gemini", "-p"])),
        )


class MegaStrategyAgent:
    def __init__(self, config: MegaVisionConfig, prompts_dir: Path | str | None = None) -> None:
        self.config = config
        self._prompts_dir = Path(prompts_dir or Path(__file__).resolve().parents[2] / "prompts")
        self._system_prompt: str | None = None
        self._user_template: str | None = None
        self._missing_commands: set[str] = set()

    async def decide(self, ts: datetime, candidate_signals: list[dict[str, Any]], market_context: dict[str, Any]) -> dict[str, Any]:
        if self.config.mode == "disabled":
            return self._fallback("disabled")
        if os.environ.get(self.config.kill_switch_env_var, ""):
            return self._fallback("kill_switch")
        if not candidate_signals:
            return self._fallback("no_candidates")
        if not self.config.external_calls_enabled:
            return self._fallback("external_calls_disabled")

        prompt = self._render_prompt(ts, candidate_signals, market_context)
        if _HAS_SDK:
            result = await self._query_sdk(prompt, shadow=self.config.mode == "shadow")
            if result:
                return result
        for runner in (self._spawn_claude, self._spawn_codex, self._spawn_gemini):
            result = await asyncio.to_thread(runner, prompt)
            parsed = self._parse_response(result, candidate_signals)
            if parsed:
                return parsed
        return self._fallback("no_valid_pick")

    async def _query_sdk(self, prompt: str, *, shadow: bool) -> dict[str, Any] | None:
        try:
            model = self.config.shadow_model if shadow else self.config.live_model
            options = ClaudeAgentOptions(  # type: ignore[misc]
                system_prompt=self._load_system_prompt(),
                model=model,
                max_turns=4,
                permission_mode="bypassPermissions",
            )
            chunks: list[str] = []
            async for message in query(prompt=prompt, options=options):
                text = self._extract_message_text(message)
                if text:
                    chunks.append(text)
            return self._parse_response("\n".join(chunks), [])
        except Exception as exc:
            logger.warning("MegaStrategyAgent SDK query failed: %s", exc)
            return None

    def _spawn_claude(self, prompt: str) -> str:
        return self._spawn_subprocess(self.config.claude_command or ["claude", "-p"], prompt)

    def _spawn_codex(self, prompt: str) -> str:
        command = list(self.config.codex_command or ["codex", "exec", "--full-auto"])
        binary_name = Path(command[0]).stem.lower() if command else ""
        if binary_name == "codex" and len(command) >= 2 and command[1] == "exec":
            command.append(prompt[:8000])
            prompt = ""
        return self._spawn_subprocess(command, prompt)

    def _spawn_gemini(self, prompt: str) -> str:
        return self._spawn_subprocess(self.config.gemini_command or ["gemini", "-p"], prompt)

    def _spawn_subprocess(self, command: list[str], prompt: str) -> str:
        command = self._resolve_command(command)
        binary = command[0] if command else ""
        if not binary:
            return ""
        if binary in self._missing_commands:
            return ""
        if shutil.which(binary) is None:
            self._missing_commands.add(binary)
            logger.warning("MegaStrategyAgent command not found: %s", binary)
            return ""
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
                timeout=max(15, int(self.config.fallback_on_timeout_seconds)),
                encoding="utf-8",
            )
            return completed.stdout or ""
        except Exception as exc:
            logger.warning("MegaStrategyAgent subprocess failed for %s: %s", command[:2], exc)
            return ""

    def _resolve_command(self, command: list[str]) -> list[str]:
        if not command:
            return []
        binary = command[0]
        resolved_path = shutil.which(binary)
        if resolved_path is not None:
            resolved = list(command)
            resolved[0] = resolved_path
            return resolved
        if sys.platform.startswith("win") and "." not in Path(binary).name:
            for suffix in (".exe", ".cmd", ".bat"):
                candidate = f"{binary}{suffix}"
                resolved_path = shutil.which(candidate)
                if resolved_path is not None:
                    resolved = list(command)
                    resolved[0] = resolved_path
                    return resolved
        return command

    def _render_prompt(self, ts: datetime, candidate_signals: list[dict[str, Any]], market_context: dict[str, Any]) -> str:
        template = self._load_user_template()
        candidates = []
        for signal in candidate_signals:
            candidates.append(
                f"- {signal.get('strategy_name', 'unknown')} | {signal.get('side', '?')} | "
                f"qty={signal.get('quantity', 1)} | sl={signal.get('sl_ticks')} | tp={signal.get('tp_ticks')} | "
                f"reason={signal.get('reason', '')}"
            )
        return template.format(
            timestamp_utc=ts.isoformat(),
            candidate_signals_summary="\n".join(candidates),
            market_open=market_context.get("market_open"),
            news_standdown=market_context.get("news_standdown"),
            near_close=market_context.get("near_close"),
            active_count=market_context.get("active_count"),
            max_positions=market_context.get("max_positions"),
            current_price=market_context.get("current_price"),
        )

    def _load_system_prompt(self) -> str:
        if self._system_prompt is not None:
            return self._system_prompt
        path = self._prompts_dir / "mega_vision_system.md"
        self._system_prompt = path.read_text(encoding="utf-8") if path.exists() else (
            "You are the Mega Vision futures trading analyst. Provide advisory context only. "
            "Recommend one candidate for review or return no_pick. "
            "Return compact JSON with keys strategy_name, side, confidence, reasoning, no_pick, advisory_only."
        )
        return self._system_prompt

    def _load_user_template(self) -> str:
        if self._user_template is not None:
            return self._user_template
        path = self._prompts_dir / "mega_vision_user_template.md"
        self._user_template = path.read_text(encoding="utf-8") if path.exists() else (
            "Decision request at {timestamp_utc}.\n\nCandidates:\n{candidate_signals_summary}\n\n"
            "State: market_open={market_open} news_standdown={news_standdown} near_close={near_close} "
            "positions={active_count}/{max_positions} price={current_price}\n\n"
            "Respond with JSON only."
        )
        return self._user_template

    def _extract_message_text(self, message: Any) -> str:
        for attr in ("text", "content", "message"):
            value = getattr(message, attr, None)
            if isinstance(value, str):
                return value
        return str(message)

    def _parse_response(self, raw_text: str, candidate_signals: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not raw_text.strip():
            return None
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if payload.get("no_pick"):
            return self._fallback(payload.get("reasoning", "no_pick"))
        strategy_name = payload.get("strategy_name")
        if candidate_signals:
            for signal in candidate_signals:
                if signal.get("strategy_name") == strategy_name:
                    decision = dict(signal)
                    decision["confidence"] = float(payload.get("confidence", signal.get("confidence", 0.0)))
                    decision["reasoning"] = payload.get("reasoning", "")
                    decision["advisory_only"] = True
                    decision["no_pick"] = False
                    decision["fallback"] = False
                    decision["ts"] = datetime.now(timezone.utc).isoformat()
                    return decision
        if strategy_name:
            payload["fallback"] = False
            payload["advisory_only"] = True
            payload["no_pick"] = False
            payload["ts"] = datetime.now(timezone.utc).isoformat()
            return payload
        return None

    def _fallback(self, reason: str) -> dict[str, Any]:
        return {
            "strategy_name": None,
            "confidence": 0.0,
            "reasoning": f"FALLBACK: {reason}",
            "advisory_only": True,
            "no_pick": True,
            "fallback": True,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

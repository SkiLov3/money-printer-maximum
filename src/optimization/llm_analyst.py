from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.persistence.database import DatabaseService

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "llm_analysis"


@dataclass
class AnalysisResult:
    reasoning: str = ""
    config_changes: dict[str, Any] = field(default_factory=dict)
    code_patches: list[dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""
    output_path: str = ""


class LLMAnalyst:
    def __init__(self, database: DatabaseService | None = None, config: dict[str, Any] | None = None) -> None:
        self.database = database or DatabaseService({"enabled": False})
        self.config = config or {}

    def analyze(self, instrument: str, epoch: int, prompt: str) -> AnalysisResult:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / f"{instrument}_epoch_{epoch}.txt"
        raw_output = self._call(prompt)
        output_path.write_text(raw_output, encoding="utf-8")
        result = self._parse(raw_output)
        result.raw_output = raw_output
        result.output_path = str(output_path)
        if self.database.connected:
            self.database.persist_llm_analysis(
                instrument,
                epoch,
                result.reasoning,
                result.config_changes,
                result.code_patches,
                result.output_path,
            )
        return result

    def _call(self, prompt: str) -> str:
        provider = str(self.config.get("provider", "claude")).strip().lower()
        runners = [
            ("claude", self.config.get("claude", ["claude", "-p"])),
            ("codex", self.config.get("codex", ["codex", "exec", "--full-auto"])),
            ("gemini", self.config.get("gemini", ["gemini", "-p"])),
        ]
        if provider:
            runners = [runner for runner in runners if runner[0] == provider]
        timeout = int(self.config.get("timeout_seconds", 300))
        max_calls = max(1, int(self.config.get("max_provider_calls_per_analysis", 1)))
        for calls, (key, command) in enumerate(runners):
            if calls >= max_calls:
                break
            output = self._run_command(list(command), prompt, timeout=timeout, key=key)
            if output.strip():
                return output
        return ""

    def _run_command(self, command: list[str], prompt: str, *, timeout: int, key: str) -> str:
        if not command:
            return ""
        prompt = prompt[: int(self.config.get("max_prompt_chars", 8000))]
        command = self._resolve_command(command)
        if not command:
            return ""
        try:
            binary_name = Path(command[0]).stem.lower()
            if key == "codex" and binary_name == "codex" and len(command) >= 2 and command[1] == "exec":
                command = list(command) + [prompt[:8000]]
                prompt = ""
            proc = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                encoding="utf-8",
            )
            return proc.stdout or proc.stderr or ""
        except Exception as exc:
            logger.warning("LLM analyst %s runner failed: %s", key, exc)
            return ""

    def _resolve_command(self, command: list[str]) -> list[str]:
        binary = command[0]
        resolved = shutil.which(binary)
        if resolved:
            out = list(command)
            out[0] = resolved
            return out
        if sys.platform.startswith("win") and "." not in Path(binary).name:
            for suffix in (".exe", ".cmd", ".bat"):
                resolved = shutil.which(f"{binary}{suffix}")
                if resolved:
                    out = list(command)
                    out[0] = resolved
                    return out
        return []

    def _parse(self, raw: str) -> AnalysisResult:
        result = AnalysisResult()
        reasoning_match = re.search(r"###?\s*REASONING\s*\n(.*?)(?=###?\s*CONFIG_CHANGES|$)", raw, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            result.reasoning = reasoning_match.group(1).strip()

        json_blocks = re.findall(r"```json\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
        for block in json_blocks:
            try:
                parsed = json.loads(block.strip())
            except json.JSONDecodeError:
                continue
            if "config_changes" in parsed:
                result.config_changes = parsed.get("config_changes", {})
            if "code_patches" in parsed:
                result.code_patches = parsed.get("code_patches", [])
        return result

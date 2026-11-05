from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SECTION_HEADING = "## Strategy Learnings"
_LINE_PATTERN = re.compile(
    r"^- Run (?P<run_id>\S+): (?P<changes_made>.+?) -> (?P<metrics>.+?) \((?P<verdict>kept|reverted)\)\s*$"
)


def _fmt_metric(label: str, before: float | None, after: float | None, *, pct: bool = False) -> str | None:
    if before is None or after is None:
        return None
    if pct:
        return f"{label} {before:.1%}->{after:.1%}"
    return f"{label} {before:.2f}->{after:.2f}"


class ClaudeMdWriter:
    def __init__(self, claude_md_path: str = "CLAUDE.md") -> None:
        self._path = Path(claude_md_path)

    def append_learning(
        self,
        *,
        run_id: str,
        changes_made: str,
        metrics_before: dict,
        metrics_after: dict,
        verdict: str,
    ) -> None:
        parts: list[str] = []
        sharpe = _fmt_metric("Sharpe", metrics_before.get("sharpe"), metrics_after.get("sharpe"))
        if sharpe:
            parts.append(sharpe)
        win_rate = _fmt_metric("WinRate", metrics_before.get("win_rate"), metrics_after.get("win_rate"), pct=True)
        if win_rate:
            parts.append(win_rate)
        drawdown = _fmt_metric("MaxDD", metrics_before.get("max_dd"), metrics_after.get("max_dd"), pct=True)
        if drawdown:
            parts.append(drawdown)
        pass_rate = _fmt_metric("PassRate", metrics_before.get("pass_rate"), metrics_after.get("pass_rate"), pct=True)
        if pass_rate:
            parts.append(pass_rate)

        metrics_text = " | ".join(parts) if parts else "no metrics"
        changes_text = changes_made.strip() or "(no changes)"
        line = f"- Run {run_id}: {changes_text} -> {metrics_text} ({verdict})\n"

        content = self._read_file()
        updated = self._ensure_section(content)
        updated = self._append_line(updated, line)
        self._write_file(updated)

    def read_learnings(self) -> list[dict]:
        content = self._read_file()
        lines = self._extract_section_lines(content)
        result: list[dict] = []
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped.startswith("- "):
                continue
            match = _LINE_PATTERN.match(stripped)
            if not match:
                continue
            result.append(
                {
                    "run_id": match.group("run_id"),
                    "changes_made": match.group("changes_made"),
                    "metrics_text": match.group("metrics"),
                    "verdict": match.group("verdict"),
                }
            )
        return result

    def get_learnings_summary(self) -> str:
        learnings = self.read_learnings()
        if not learnings:
            return "No strategy learnings recorded yet."
        lines = ["Strategy Learnings (most recent last):"]
        for entry in learnings:
            lines.append(
                f"  Run {entry['run_id']}: {entry['changes_made']} -> {entry['metrics_text']} ({entry['verdict']})"
            )
        return "\n".join(lines)

    def _read_file(self) -> str:
        try:
            return self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except OSError as exc:
            logger.error("ClaudeMdWriter read error: %s", exc)
            return ""

    def _write_file(self, content: str) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.error("ClaudeMdWriter write error: %s", exc)

    def _ensure_section(self, content: str) -> str:
        if _SECTION_HEADING in content:
            return content
        separator = "\n\n" if content.rstrip() else ""
        return content.rstrip() + f"{separator}{_SECTION_HEADING}\n"

    def _append_line(self, content: str, line: str) -> str:
        lines = content.splitlines(keepends=True)
        start_index = None
        for index, raw_line in enumerate(lines):
            if raw_line.strip() == _SECTION_HEADING:
                start_index = index
                break
        if start_index is None:
            return content + "\n" + line
        insert_index = len(lines)
        for index in range(start_index + 1, len(lines)):
            if lines[index].startswith("## "):
                insert_index = index
                break
        lines.insert(insert_index, line)
        return "".join(lines)

    def _extract_section_lines(self, content: str) -> list[str]:
        lines = content.splitlines()
        start_index = None
        for index, raw_line in enumerate(lines):
            if raw_line.strip() == _SECTION_HEADING:
                start_index = index + 1
                break
        if start_index is None:
            return []
        result = []
        for raw_line in lines[start_index:]:
            if raw_line.startswith("## "):
                break
            result.append(raw_line)
        return result

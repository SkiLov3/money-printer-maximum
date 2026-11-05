from __future__ import annotations

import http.client
import json
import logging
import math
import os
import urllib.parse
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def embed_market(df: pd.DataFrame) -> list[float]:
    frame = df.copy()
    frame.columns = [c.lower() for c in frame.columns]
    if "close" not in frame.columns and "price" in frame.columns:
        frame["close"] = frame["price"]

    close = frame["close"].astype(float)
    returns = close.pct_change().fillna(0.0)
    rolling = returns.rolling(30).std().fillna(0.0)
    trend = (close.iloc[-1] - close.iloc[0]) / max(abs(close.iloc[0]), 1e-9)
    vol = float(returns.std())
    skew = float(returns.skew()) if len(returns) > 2 else 0.0
    kurt = float(returns.kurt()) if len(returns) > 3 else 0.0
    tail = float(returns.quantile(0.95) - returns.quantile(0.05))
    realized = float(rolling.mean())
    max_move = float(returns.abs().max())
    return [
        _safe_float(trend),
        _safe_float(vol),
        _safe_float(skew),
        _safe_float(kurt),
        _safe_float(tail),
        _safe_float(realized),
        _safe_float(max_move),
    ]


def embed_params(config: dict[str, Any]) -> list[float]:
    strategies = config.get("strategies", [])
    direction_filters = config.get("direction_filters", {})
    score = 0.0
    for item in strategies:
        if item.get("enabled", True):
            score += 1.0
        score += float(item.get("priority", 0)) / 100.0
        params = item.get("params", {})
        for value in params.values():
            if isinstance(value, (int, float)):
                score += float(value) * 0.01
    side_total = sum(len(v) for v in direction_filters.values()) if direction_filters else 0
    return [score, float(side_total), float(len(strategies))]


def embed_outcome(summary: dict[str, Any]) -> list[float]:
    return [
        _safe_float(summary.get("total_pnl", 0.0)),
        _safe_float(summary.get("win_rate", 0.0)),
        abs(_safe_float(summary.get("max_drawdown", 0.0))),
        _safe_float(summary.get("profit_factor", 0.0) if summary.get("profit_factor") != "Inf" else 5.0),
        _safe_float(summary.get("total_trades", 0.0)),
    ]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def embed_text(
    text: str | list[str],
    model: str = "nomic-embed-text",
    host: str | None = None,
    timeout: int = 30,
) -> list[float] | list[list[float]]:
    """Generate high-dimensional vector embeddings via Ollama (default: nomic-embed-text).

    Args:
        text: A single string or a list of strings to embed.
        model: Embedding model name in Ollama (defaults to 'nomic-embed-text', 768-d).
        host: Ollama host URL (defaults to OLLAMA_HOST or 'http://127.0.0.1:11434').
        timeout: Request timeout in seconds.

    Returns:
        A 768-d float vector if input is a single string, or a list of vectors if input is a list.
    """
    ollama_host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    parsed_url = urllib.parse.urlparse(ollama_host)
    if parsed_url.scheme not in ("http", "https"):
        raise ValueError(f"Disallowed URL scheme: {parsed_url.scheme}")

    is_single = isinstance(text, str)
    inputs = [text] if is_single else list(text)

    if not inputs or (is_single and not text.strip()):
        return [] if is_single else []

    payload = {"model": model, "input": inputs}
    data = json.dumps(payload).encode("utf-8")

    hostname = parsed_url.hostname or "127.0.0.1"
    port = parsed_url.port or (443 if parsed_url.scheme == "https" else 11434)
    path = (parsed_url.path.rstrip("/") + "/api/embed") if parsed_url.path else "/api/embed"

    try:
        connection_cls = http.client.HTTPSConnection if parsed_url.scheme == "https" else http.client.HTTPConnection
        conn = connection_cls(hostname, port, timeout=timeout)
        conn.request("POST", path, body=data, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            logger.warning(f"Ollama returned HTTP {resp.status} for {path}")
            conn.close()
            return [] if is_single else []
        result = json.loads(resp.read().decode("utf-8"))
        conn.close()
        embeddings = result.get("embeddings", [])
        if is_single:
            return embeddings[0] if embeddings else []
        return embeddings
    except Exception as e:
        logger.warning(f"Failed to generate text embedding via Ollama at {ollama_host}: {e}")
        return [] if is_single else []

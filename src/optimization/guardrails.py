from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def check_consecutive_passes(
    *,
    data: pd.DataFrame,
    config: dict[str, Any],
    backtest_fn: Callable[[pd.DataFrame, dict[str, Any]], dict[str, Any] | None],
    n_required: int = 3,
    offset_days: list[int] | None = None,
    bars_per_day: int = 60 * 23,
    min_bars: int = 5000,
) -> dict[str, Any]:
    offsets = offset_days if offset_days is not None else [0, 3, 7]
    passes = []
    attempts = []
    for day_offset in offsets:
        bar_offset = max(0, int(day_offset) * int(bars_per_day))
        window = data.iloc[bar_offset:]
        if len(window) < min_bars:
            window = data

        result = backtest_fn(window, config)
        if result is None:
            passes.append(False)
            attempts.append(
                {
                    "offset_days": day_offset,
                    "offset_bars": bar_offset,
                    "passed": False,
                }
            )
            continue
        combine_eval = result.get("combine_evaluation")
        if isinstance(combine_eval, dict):
            passed = bool(combine_eval.get("passed", False))
        else:
            passed = bool(result.get("passed", False))
        passes.append(passed)
        attempts.append(
            {
                "offset_days": day_offset,
                "offset_bars": bar_offset,
                "passed": passed,
                "summary": result.get("summary", {}),
            }
        )

    return {
        "all_passed": sum(1 for p in passes if p) >= n_required,
        "passes": passes,
        "attempts": attempts,
    }


def _permute_ohlc(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    frame = data.copy().reset_index(drop=True)
    frame.columns = [c.lower() for c in frame.columns]
    if "close" not in frame.columns:
        return frame

    closes = frame["close"].astype(float).to_numpy()
    if len(closes) < 2:
        return frame
    log_returns = np.diff(np.log(np.maximum(closes, 1e-12)))
    rng = np.random.default_rng(seed)
    rng.shuffle(log_returns)

    new_close = np.empty_like(closes)
    new_close[0] = closes[0]
    for idx, log_ret in enumerate(log_returns, start=1):
        new_close[idx] = new_close[idx - 1] * float(np.exp(log_ret))
    new_close = np.maximum(new_close, 1e-8)
    frame.loc[:, "close"] = new_close

    ratio = new_close / np.maximum(closes, 1e-12)
    for col in ("open", "high", "low"):
        if col in frame.columns:
            frame.loc[:, col] = frame[col].astype(float).to_numpy() * ratio

    if "open" in frame.columns and "high" in frame.columns and "low" in frame.columns:
        frame.loc[:, "high"] = frame[["open", "close", "high"]].max(axis=1)
        frame.loc[:, "low"] = frame[["open", "close", "low"]].min(axis=1)
    if "price" in frame.columns:
        frame.loc[:, "price"] = frame["close"]
    return frame


def check_permutation_significance(
    *,
    real_return: float,
    data: pd.DataFrame,
    config: dict[str, Any],
    backtest_fn: Callable[[pd.DataFrame, dict[str, Any]], dict[str, Any] | None],
    n_permutations: int = 20,
    p_threshold: float = 0.05,
) -> dict[str, Any]:
    permuted_returns = []
    ge_count = 0

    for idx in range(n_permutations):
        permuted = _permute_ohlc(data, seed=42 + idx)
        result = backtest_fn(permuted, config)
        if result is None:
            score = float("-inf")
        elif "summary" in result and isinstance(result["summary"], dict):
            score = float(result["summary"].get("total_pnl", float("-inf")))
        else:
            score = float(result.get("total_return", result.get("total_pnl", float("-inf"))))
        permuted_returns.append(score)
        if score >= real_return:
            ge_count += 1

    p_value = ge_count / max(n_permutations, 1)
    return {
        "p_value": p_value,
        "significant": p_value < p_threshold,
        "n_permutations": n_permutations,
        "real_return": real_return,
        "mean_permuted": float(np.mean(permuted_returns)) if permuted_returns else 0.0,
        "permuted_returns": permuted_returns,
    }

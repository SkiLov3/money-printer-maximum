from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .base import StrategySignal


@dataclass
class BlendResult:
    signal: StrategySignal
    base_score: float
    effective_score: float
    agreement_count: int


class SignalBlender:
    """Select the highest-confluence strategy signal.

    This ports the native itchy-tradebot blender behavior into the swarm
    strategy engine: highest score wins, and signals that agree on direction
    receive a multi-agree bonus.
    """

    def __init__(self, multi_agree_bonus: float = 2.0):
        self._bonus = float(multi_agree_bonus)

    def select(self, signals: list[StrategySignal]) -> BlendResult | None:
        scored = self.score(signals)
        if not scored:
            return None
        return scored[0]

    def score(self, signals: list[StrategySignal]) -> list[BlendResult]:
        if not signals:
            return []

        direction_counts = Counter(signal.side.lower() for signal in signals)
        scored: list[BlendResult] = []
        for signal in signals:
            agreement_count = int(direction_counts[signal.side.lower()])
            base_score = self._base_score(signal)
            effective_score = base_score + (self._bonus if agreement_count >= 2 else 0.0)
            scored.append(
                BlendResult(
                    signal=signal,
                    base_score=base_score,
                    effective_score=effective_score,
                    agreement_count=agreement_count,
                )
            )

        scored.sort(key=lambda item: item.effective_score, reverse=True)
        return scored

    @staticmethod
    def _base_score(signal: StrategySignal) -> float:
        metadata = signal.metadata or {}
        if "confluence_score" in metadata:
            try:
                score = float(metadata["confluence_score"])
                if math.isfinite(score):
                    return score
            except (TypeError, ValueError):
                pass
        try:
            conf = float(signal.confidence or 0.0)
            return conf * 10.0 if math.isfinite(conf) else 0.0
        except (TypeError, ValueError):
            return 0.0

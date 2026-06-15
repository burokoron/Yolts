from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationGameResult:
    game_id: str
    baseline_engine: str
    result: str

from __future__ import annotations

from dataclasses import dataclass

from tamanobori.versions import MOVE_LABEL_SIZE, OUTCOME_CAUSE_ORDER


@dataclass(frozen=True)
class NetworkOutputShape:
    policy_normal: tuple[int, int]
    policy_style: tuple[int, int]
    q_normal_result: tuple[int, int]
    q_nyugyoku: tuple[int, int]
    outcome_cause: tuple[int, int]
    moves_left: tuple[int, int]

    @classmethod
    def for_batch(cls, batch_size: int) -> NetworkOutputShape:
        return cls(
            policy_normal=(batch_size, MOVE_LABEL_SIZE),
            policy_style=(batch_size, MOVE_LABEL_SIZE),
            q_normal_result=(batch_size, MOVE_LABEL_SIZE),
            q_nyugyoku=(batch_size, MOVE_LABEL_SIZE),
            outcome_cause=(batch_size, len(OUTCOME_CAUSE_ORDER)),
            moves_left=(batch_size, 1),
        )

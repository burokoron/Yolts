from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointMetadata:
    checkpoint_id: str
    model_config_id: str
    dataset_ids: tuple[str, ...]
    optimizer_step: int

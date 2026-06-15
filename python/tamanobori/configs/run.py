from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    config_id: str
    run_name: str
    output_root: Path = Path("workspace/runs")
    random_seed: int = 0

from __future__ import annotations

from dataclasses import dataclass

FORMAL_NAME_JA = "爆裂玉登太郎"
ROMANIZED_NAME = "Bakuretsu Tamanobori Taro"
ENGINE_ID = "bakuretsu_tamanobori_taro"
SHORT_ID = "tamanobori"
ENGINE_FAMILY_ID = "bakuretsu"
USI_ID_NAME = "BakuretsuTamanoboriTaro"

RULE_VERSION = "rule_v0"
FEATURE_VERSION = "feature_v0"
MOVE_LABEL_VERSION = "move_label_v1_dlshogi27_current_side_plane_major"
NETWORK_OUTPUT_VERSION = "network_output_v0"
TARGET_VERSION = "target_v0"
SELFPLAY_LOG_VERSION = "selfplay_log_v0"
DATASET_VERSION = "dataset_v0"
CHECKPOINT_VERSION = "checkpoint_v0"

BOARD_SIZE = 9
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE
MOVE_LABEL_PLANES = 27
MOVE_LABEL_SIZE = BOARD_SQUARES * MOVE_LABEL_PLANES

OUTCOME_CAUSE_ORDER = (
    "win_normal",
    "win_nyugyoku",
    "draw",
    "loss_nyugyoku",
    "loss_normal",
)


@dataclass(frozen=True)
class VersionSet:
    rule: str = RULE_VERSION
    feature: str = FEATURE_VERSION
    move_label: str = MOVE_LABEL_VERSION
    network_output: str = NETWORK_OUTPUT_VERSION
    target: str = TARGET_VERSION
    selfplay_log: str = SELFPLAY_LOG_VERSION
    dataset: str = DATASET_VERSION
    checkpoint: str = CHECKPOINT_VERSION

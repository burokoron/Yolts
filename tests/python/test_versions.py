from pathlib import Path

from tamanobori import MOVE_LABEL_SIZE, USI_ID_NAME, VersionSet
from tamanobori.network import NetworkOutputShape
from tamanobori.versions import (
    BOARD_SIZE,
    BOARD_SQUARES,
    ENGINE_FAMILY_ID,
    ENGINE_ID,
    FORMAL_NAME_JA,
    MOVE_LABEL_PLANES,
    OUTCOME_CAUSE_ORDER,
    ROMANIZED_NAME,
    SHORT_ID,
)


def contract_values() -> dict[str, str]:
    contract_path = Path(__file__).parents[2] / "contracts" / "tamanobori_constants.env"
    values: dict[str, str] = {}
    for line in contract_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", maxsplit=1)
        values[key] = value
    return values


def assert_contract_value(contract: dict[str, str], key: str, actual: object) -> None:
    assert str(actual) == contract[key]


def test_move_label_size_matches_spec() -> None:
    contract = contract_values()

    assert_contract_value(contract, "BOARD_SIZE", BOARD_SIZE)
    assert_contract_value(contract, "BOARD_SQUARES", BOARD_SQUARES)
    assert_contract_value(contract, "MOVE_LABEL_PLANES", MOVE_LABEL_PLANES)
    assert_contract_value(contract, "MOVE_LABEL_SIZE", MOVE_LABEL_SIZE)


def test_outcome_cause_order_is_fixed() -> None:
    contract = contract_values()

    assert_contract_value(
        contract, "OUTCOME_CAUSE_ORDER", ",".join(OUTCOME_CAUSE_ORDER)
    )


def test_network_output_shape_uses_move_label_size() -> None:
    shape = NetworkOutputShape.for_batch(batch_size=2)

    assert shape.policy_normal == (2, MOVE_LABEL_SIZE)
    assert shape.q_nyugyoku == (2, MOVE_LABEL_SIZE)
    assert shape.outcome_cause == (2, 5)
    assert shape.moves_left == (2, 1)


def test_version_set_defaults_match_shared_contract() -> None:
    contract = contract_values()
    versions = VersionSet()

    assert_contract_value(contract, "FORMAL_NAME_JA", FORMAL_NAME_JA)
    assert_contract_value(contract, "ROMANIZED_NAME", ROMANIZED_NAME)
    assert_contract_value(contract, "ENGINE_ID", ENGINE_ID)
    assert_contract_value(contract, "SHORT_ID", SHORT_ID)
    assert_contract_value(contract, "ENGINE_FAMILY_ID", ENGINE_FAMILY_ID)
    assert_contract_value(contract, "USI_ID_NAME", USI_ID_NAME)
    assert_contract_value(contract, "RULE_VERSION", versions.rule)
    assert_contract_value(contract, "FEATURE_VERSION", versions.feature)
    assert_contract_value(contract, "MOVE_LABEL_VERSION", versions.move_label)
    assert_contract_value(contract, "NETWORK_OUTPUT_VERSION", versions.network_output)
    assert_contract_value(contract, "TARGET_VERSION", versions.target)
    assert_contract_value(contract, "SELFPLAY_LOG_VERSION", versions.selfplay_log)
    assert_contract_value(contract, "DATASET_VERSION", versions.dataset)
    assert_contract_value(contract, "CHECKPOINT_VERSION", versions.checkpoint)

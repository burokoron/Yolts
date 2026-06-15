pub const FORMAL_NAME_JA: &str = "爆裂玉登太郎";
pub const ROMANIZED_NAME: &str = "Bakuretsu Tamanobori Taro";
pub const ENGINE_ID: &str = "bakuretsu_tamanobori_taro";
pub const SHORT_ID: &str = "tamanobori";
pub const ENGINE_FAMILY_ID: &str = "bakuretsu";
pub const USI_ID_NAME: &str = "BakuretsuTamanoboriTaro";

pub const RULE_VERSION: &str = "rule_v0";
pub const FEATURE_VERSION: &str = "feature_v0";
pub const MOVE_LABEL_VERSION: &str = "move_label_v0";
pub const NETWORK_OUTPUT_VERSION: &str = "network_output_v0";
pub const TARGET_VERSION: &str = "target_v0";
pub const SELFPLAY_LOG_VERSION: &str = "selfplay_log_v0";
pub const DATASET_VERSION: &str = "dataset_v0";
pub const CHECKPOINT_VERSION: &str = "checkpoint_v0";

pub const BOARD_SIZE: usize = 9;
pub const BOARD_SQUARES: usize = BOARD_SIZE * BOARD_SIZE;
pub const MOVE_LABEL_PLANES: usize = 27;
pub const MOVE_LABEL_SIZE: usize = BOARD_SQUARES * MOVE_LABEL_PLANES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VersionSet {
    pub rule: &'static str,
    pub feature: &'static str,
    pub move_label: &'static str,
    pub network_output: &'static str,
    pub target: &'static str,
    pub selfplay_log: &'static str,
    pub dataset: &'static str,
    pub checkpoint: &'static str,
}

pub const VERSION_SET: VersionSet = VersionSet {
    rule: RULE_VERSION,
    feature: FEATURE_VERSION,
    move_label: MOVE_LABEL_VERSION,
    network_output: NETWORK_OUTPUT_VERSION,
    target: TARGET_VERSION,
    selfplay_log: SELFPLAY_LOG_VERSION,
    dataset: DATASET_VERSION,
    checkpoint: CHECKPOINT_VERSION,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutcomeCause {
    WinNormal,
    WinNyugyoku,
    Draw,
    LossNyugyoku,
    LossNormal,
}

pub const OUTCOME_CAUSE_ORDER: [OutcomeCause; 5] = [
    OutcomeCause::WinNormal,
    OutcomeCause::WinNyugyoku,
    OutcomeCause::Draw,
    OutcomeCause::LossNyugyoku,
    OutcomeCause::LossNormal,
];

#[must_use]
pub const fn move_label_size() -> usize {
    MOVE_LABEL_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTRACT: &str = include_str!("../../../contracts/tamanobori_constants.env");

    fn contract_value(key: &str) -> &'static str {
        for line in CONTRACT.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let Some((candidate, value)) = line.split_once('=') else {
                continue;
            };
            if candidate == key {
                return value;
            }
        }

        panic!("missing contract key: {key}");
    }

    fn outcome_cause_key(outcome: OutcomeCause) -> &'static str {
        match outcome {
            OutcomeCause::WinNormal => "win_normal",
            OutcomeCause::WinNyugyoku => "win_nyugyoku",
            OutcomeCause::Draw => "draw",
            OutcomeCause::LossNyugyoku => "loss_nyugyoku",
            OutcomeCause::LossNormal => "loss_normal",
        }
    }

    #[test]
    fn move_label_size_matches_spec() {
        assert_eq!(BOARD_SIZE.to_string(), contract_value("BOARD_SIZE"));
        assert_eq!(BOARD_SQUARES.to_string(), contract_value("BOARD_SQUARES"));
        assert_eq!(
            MOVE_LABEL_PLANES.to_string(),
            contract_value("MOVE_LABEL_PLANES")
        );
        assert_eq!(
            MOVE_LABEL_SIZE.to_string(),
            contract_value("MOVE_LABEL_SIZE")
        );
    }

    #[test]
    fn outcome_order_is_fixed() {
        let order = OUTCOME_CAUSE_ORDER
            .map(outcome_cause_key)
            .as_slice()
            .join(",");

        assert_eq!(order, contract_value("OUTCOME_CAUSE_ORDER"));
        assert_eq!(
            OUTCOME_CAUSE_ORDER,
            [
                OutcomeCause::WinNormal,
                OutcomeCause::WinNyugyoku,
                OutcomeCause::Draw,
                OutcomeCause::LossNyugyoku,
                OutcomeCause::LossNormal,
            ]
        );
    }

    #[test]
    fn constants_match_shared_contract() {
        assert_eq!(FORMAL_NAME_JA, contract_value("FORMAL_NAME_JA"));
        assert_eq!(ROMANIZED_NAME, contract_value("ROMANIZED_NAME"));
        assert_eq!(ENGINE_ID, contract_value("ENGINE_ID"));
        assert_eq!(SHORT_ID, contract_value("SHORT_ID"));
        assert_eq!(ENGINE_FAMILY_ID, contract_value("ENGINE_FAMILY_ID"));
        assert_eq!(USI_ID_NAME, contract_value("USI_ID_NAME"));
        assert_eq!(VERSION_SET.rule, contract_value("RULE_VERSION"));
        assert_eq!(VERSION_SET.feature, contract_value("FEATURE_VERSION"));
        assert_eq!(VERSION_SET.move_label, contract_value("MOVE_LABEL_VERSION"));
        assert_eq!(
            VERSION_SET.network_output,
            contract_value("NETWORK_OUTPUT_VERSION")
        );
        assert_eq!(VERSION_SET.target, contract_value("TARGET_VERSION"));
        assert_eq!(
            VERSION_SET.selfplay_log,
            contract_value("SELFPLAY_LOG_VERSION")
        );
        assert_eq!(VERSION_SET.dataset, contract_value("DATASET_VERSION"));
        assert_eq!(VERSION_SET.checkpoint, contract_value("CHECKPOINT_VERSION"));
    }
}

mod feature;
mod move_label;
mod position;
mod view;

#[cfg(test)]
mod tests;

pub use feature::{
    attack_count_feature_plane, attack_feature_plane, encode_attack_features,
    encode_check_features, encode_hand_features, encode_piece_placement_features,
    hand_feature_plane, piece_feature_plane, AttackFeatures, CheckFeatures, HandFeatures,
    PieceFeatureSide, PiecePlacementFeatures, ATTACK_FEATURE_PLANES, CHECK_FEATURE_PLANES,
    HAND_FEATURE_PLANES, HAND_FEATURE_PLANES_PER_SIDE, PIECE_FEATURE_PLANES,
};
pub use move_label::{
    decode_move_label, encode_legal_moves, encode_move_label, make_legal_move_mask,
    move_label_from_parts, move_label_parts, MoveLabelError, MoveLabelParts, MoveLabelPlane,
};
pub use position::{
    DeclarationRule, NyugyokuStats, Position, RepetitionOutcome, RuleTerminalOutcome,
};
pub use shogi_core::{Color, Hand, Move, Piece, PieceKind, Square};

pub const FORMAL_NAME_JA: &str = "爆裂玉登太郎";
pub const ROMANIZED_NAME: &str = "Bakuretsu Tamanobori Taro";
pub const ENGINE_ID: &str = "bakuretsu_tamanobori_taro";
pub const SHORT_ID: &str = "tamanobori";
pub const ENGINE_FAMILY_ID: &str = "bakuretsu";
pub const USI_ID_NAME: &str = "BakuretsuTamanoboriTaro";

pub const RULE_VERSION: &str = "rule_v0";
pub const FEATURE_VERSION: &str = "feature_v0";
pub const MOVE_LABEL_VERSION: &str = "move_label_v1_dlshogi27_current_side_plane_major";
pub const NETWORK_OUTPUT_VERSION: &str = "network_output_v0";
pub const TARGET_VERSION: &str = "target_v0";
pub const SELFPLAY_LOG_VERSION: &str = "selfplay_log_v0";
pub const DATASET_VERSION: &str = "dataset_v0";
pub const CHECKPOINT_VERSION: &str = "checkpoint_v0";

pub const BOARD_SIZE: usize = 9;
pub const BOARD_SQUARES: usize = BOARD_SIZE * BOARD_SIZE;
pub const MOVE_LABEL_PLANES: usize = 27;
pub const MOVE_LABEL_SIZE: usize = BOARD_SQUARES * MOVE_LABEL_PLANES;
pub const PROMOTE_OFFSET: usize = 10;

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

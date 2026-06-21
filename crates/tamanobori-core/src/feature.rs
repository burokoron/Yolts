use crate::view::{current_view, plane_square_index};
use crate::{Hand, PieceKind, Position, Square, BOARD_SQUARES};

/// 手番側14 planeと相手側14 planeを合わせた盤上駒特徴数。
pub const PIECE_FEATURE_PLANES: usize = PieceKind::NUM * 2;

/// 現在手番から見た駒の所有者。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PieceFeatureSide {
    Own,
    Opponent,
}

impl PieceFeatureSide {
    const fn offset(self) -> usize {
        match self {
            Self::Own => 0,
            Self::Opponent => PieceKind::NUM,
        }
    }
}

/// plane-major形式の盤上駒特徴。
pub type PiecePlacementFeatures = [[bool; BOARD_SQUARES]; PIECE_FEATURE_PLANES];

/// 現在手番視点の所有者と駒種から盤上駒特徴のplane indexを返す。
#[must_use]
pub const fn piece_feature_plane(side: PieceFeatureSide, kind: PieceKind) -> usize {
    side.offset() + kind.array_index()
}

/// 局面を現在手番視点の盤上駒特徴へ符号化する。
///
/// 手番側の14駒種を先頭に、相手側の14駒種を続ける。各planeのマス順は
/// 指し手ラベルと同じ現在手番視点のrank-major形式である。
#[must_use]
pub fn encode_piece_placement_features(position: &Position) -> PiecePlacementFeatures {
    let position: &yasai::Position = position.as_ref();
    let side_to_move = position.side_to_move();
    let mut features = [[false; BOARD_SQUARES]; PIECE_FEATURE_PLANES];

    for square in Square::all() {
        let Some(piece) = position.piece_at(square) else {
            continue;
        };
        let side = if piece.color() == side_to_move {
            PieceFeatureSide::Own
        } else {
            PieceFeatureSide::Opponent
        };
        let plane = piece_feature_plane(side, piece.piece_kind());
        let current_square = current_view(square, side_to_move);
        features[plane][plane_square_index(current_square)] = true;
    }

    features
}

/// 各サイド（手番側または相手側）ごとの持ち駒特徴数。
pub const HAND_FEATURE_PLANES_PER_SIDE: usize = 28;

/// 手番側と相手側を合わせた持ち駒特徴数。
pub const HAND_FEATURE_PLANES: usize = HAND_FEATURE_PLANES_PER_SIDE * 2;

/// plane-major形式の持ち駒特徴。
pub type HandFeatures = [[bool; BOARD_SQUARES]; HAND_FEATURE_PLANES];

pub(crate) const HAND_PIECES_WITH_MAX_COUNT: &[(PieceKind, usize)] = &[
    (PieceKind::Pawn, 8),
    (PieceKind::Lance, 4),
    (PieceKind::Knight, 4),
    (PieceKind::Silver, 4),
    (PieceKind::Gold, 4),
    (PieceKind::Bishop, 2),
    (PieceKind::Rook, 2),
];

/// 現在手番視点の所有者、駒種、および閾値（`>=1`）から持ち駒特徴のplane indexを返す。
///
/// 持ち駒に存在し得ない駒種（王や成駒など）や、閾値が有効範囲（例: 歩は `1..=8`）外の
/// 場合は `None` を返す。
#[must_use]
pub const fn hand_feature_plane(
    side: PieceFeatureSide,
    kind: PieceKind,
    threshold: usize,
) -> Option<usize> {
    let side_offset = match side {
        PieceFeatureSide::Own => 0,
        PieceFeatureSide::Opponent => HAND_FEATURE_PLANES_PER_SIDE,
    };

    // 各駒種に対応するオフセットと、最大閾値
    let (piece_offset, max_count) = match kind {
        PieceKind::Pawn => (0, 8),
        PieceKind::Lance => (8, 4),
        PieceKind::Knight => (12, 4),
        PieceKind::Silver => (16, 4),
        PieceKind::Gold => (20, 4),
        PieceKind::Bishop => (24, 2),
        PieceKind::Rook => (26, 2),
        _ => return None,
    };

    if threshold >= 1 && threshold <= max_count {
        Some(side_offset + piece_offset + (threshold - 1))
    } else {
        None
    }
}

/// 局面を現在手番視点の持ち駒特徴へ符号化する。
///
/// 手番側の持ち駒を先頭 28 plane に、相手側の持ち駒を後半 28 plane に配置する。
/// 各駒種のplane順は枚数閾値の昇順（`>=1`から最大値まで）である。
/// 条件を満たすplaneは81マスすべてが `true` になり、満たさない場合はすべて `false` になる。
#[must_use]
pub fn encode_hand_features(position: &Position) -> HandFeatures {
    let position: &yasai::Position = position.as_ref();
    let side_to_move = position.side_to_move();
    let mut features = [[false; BOARD_SQUARES]; HAND_FEATURE_PLANES];

    // 手番側の持ち駒をエンコード
    encode_one_side_hand(
        position.hand(side_to_move),
        &mut features[0..HAND_FEATURE_PLANES_PER_SIDE],
    );
    // 相手側の持ち駒をエンコード
    encode_one_side_hand(
        position.hand(side_to_move.flip()),
        &mut features[HAND_FEATURE_PLANES_PER_SIDE..HAND_FEATURE_PLANES],
    );

    features
}

fn encode_one_side_hand(hand: Hand, planes: &mut [[bool; BOARD_SQUARES]]) {
    let mut plane_idx = 0;
    for &(kind, max_count) in HAND_PIECES_WITH_MAX_COUNT {
        let count = usize::from(hand.count(kind).unwrap_or_default()).min(max_count);
        for plane in &mut planes[plane_idx..plane_idx + count] {
            plane.fill(true);
        }
        plane_idx += max_count;
    }
}

/// 手番側と相手側それぞれの利き数しきい値面数。
const ATTACK_COUNT_FEATURE_PLANES_PER_SIDE: usize = 3;

/// 手番側14面と相手側14面を合わせた駒種別利き特徴数。
const ATTACK_PIECE_FEATURE_PLANES: usize = PieceKind::NUM * 2;

/// 駒種別利き28面と利き数しきい値6面を合わせた利き特徴数。
pub const ATTACK_FEATURE_PLANES: usize =
    ATTACK_PIECE_FEATURE_PLANES + ATTACK_COUNT_FEATURE_PLANES_PER_SIDE * 2;

/// plane-major形式の利き特徴。
pub type AttackFeatures = [[bool; BOARD_SQUARES]; ATTACK_FEATURE_PLANES];

/// 駒種別利き特徴における手番側/相手側のplane offset。
///
/// 値は盤上駒特徴の `PieceFeatureSide::offset()` と同じ 14+14 構成だが、
/// 利き特徴は独立した plane 領域のためあえて別関数とする。
const fn attack_piece_feature_side_offset(side: PieceFeatureSide) -> usize {
    match side {
        PieceFeatureSide::Own => 0,
        PieceFeatureSide::Opponent => PieceKind::NUM,
    }
}

/// 現在手番視点の所有者と駒種から駒種別利き特徴のplane indexを返す。
#[must_use]
pub const fn attack_feature_plane(side: PieceFeatureSide, kind: PieceKind) -> usize {
    attack_piece_feature_side_offset(side) + kind.array_index()
}

/// 現在手番視点の所有者と利き数しきい値（`1..=3`）から利き数特徴のplane indexを返す。
///
/// しきい値が有効範囲外の場合は `None` を返す。
#[must_use]
pub const fn attack_count_feature_plane(side: PieceFeatureSide, threshold: usize) -> Option<usize> {
    if threshold == 0 || threshold > ATTACK_COUNT_FEATURE_PLANES_PER_SIDE {
        return None;
    }

    let side_offset = match side {
        PieceFeatureSide::Own => ATTACK_PIECE_FEATURE_PLANES,
        PieceFeatureSide::Opponent => {
            ATTACK_PIECE_FEATURE_PLANES + ATTACK_COUNT_FEATURE_PLANES_PER_SIDE
        }
    };
    Some(side_offset + (threshold - 1))
}

/// 局面を現在手番視点の利き特徴へ符号化する。
///
/// plane順は次のとおり固定する。
/// 1. 手番側の駒種別利き14面
/// 2. 相手側の駒種別利き14面
/// 3. 手番側の利き数 `>=1`, `>=2`, `>=3`
/// 4. 相手側の利き数 `>=1`, `>=2`, `>=3`
///
/// 各planeのマス順は指し手ラベルと同じ現在手番視点のrank-major形式である。
/// 利きはYasaiの幾何学的利きをそのまま採用し、長距離駒は最初の占有マスを含む。
///
/// `FEATURE_VERSION` は aggregate `FeatureEncoder` 統合時に bump する。
#[must_use]
pub fn encode_attack_features(position: &Position) -> AttackFeatures {
    let position: &yasai::Position = position.as_ref();
    let side_to_move = position.side_to_move();
    let attack_map = position.attack_map();
    let mut features = [[false; BOARD_SQUARES]; ATTACK_FEATURE_PLANES];

    for (side, color) in [
        (PieceFeatureSide::Own, side_to_move),
        (PieceFeatureSide::Opponent, side_to_move.flip()),
    ] {
        for kind in PieceKind::all() {
            let plane = attack_feature_plane(side, kind);
            for square in attack_map.attacks(color, kind).squares() {
                let current_square = current_view(square, side_to_move);
                features[plane][plane_square_index(current_square)] = true;
            }
        }

        let Some(count_plane_base) = attack_count_feature_plane(side, 1) else {
            continue;
        };
        for square in Square::all() {
            let count = usize::from(attack_map.attack_count(color, square));
            if count == 0 {
                continue;
            }
            let index = plane_square_index(current_view(square, side_to_move));
            let upto = count.min(ATTACK_COUNT_FEATURE_PLANES_PER_SIDE);
            for threshold in 1..=upto {
                features[count_plane_base + (threshold - 1)][index] = true;
            }
        }
    }

    features
}

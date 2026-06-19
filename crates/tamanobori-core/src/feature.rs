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

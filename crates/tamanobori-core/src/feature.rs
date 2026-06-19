use crate::view::{current_view, plane_square_index};
use crate::{PieceKind, Position, Square, BOARD_SQUARES};

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

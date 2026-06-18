use shogi_core::{
    Color as BackendColor, Hand as BackendHand, Move as BackendMove, PartialPosition,
    Piece as BackendPiece, PieceKind as BackendPieceKind, Square as BackendSquare,
};

use crate::{
    Color, HandPiece, Move, Piece, PieceKind, Position, Square, BOARD_SIZE, HAND_PIECE_ORDER,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuleError {
    InvalidPly {
        ply: u16,
    },
    MissingKing {
        color: Color,
    },
    MultipleKings {
        color: Color,
    },
    InvalidHandCount {
        color: Color,
        piece: HandPiece,
        count: u8,
    },
    InvalidPieceCount {
        kind: PieceKind,
        count: u8,
        maximum: u8,
    },
    OpponentKingInCheck {
        color: Color,
    },
}

/// 指定局面の合法手を生成する。
///
/// # Errors
///
/// `ply`、玉数、駒総数、持ち駒数が不正な場合、または非手番側の玉が
/// 王手されている場合は [`RuleError`] を返す。
pub fn generate_legal_moves(position: &Position) -> Result<Vec<Move>, RuleError> {
    let backend = backend_position(position)?;
    Ok(backend
        .legal_moves()
        .into_iter()
        .map(move_from_backend)
        .collect())
}

fn backend_position(position: &Position) -> Result<yasai::Position, RuleError> {
    validate_position(position)?;

    let mut partial = PartialPosition::empty();
    partial.side_to_move_set(color_to_backend(position.side_to_move));
    let ply_was_set = partial.ply_set(position.ply);
    debug_assert!(ply_was_set, "validate_position rejects zero ply");

    for rank in 1..=BOARD_SIZE {
        for file in 1..=BOARD_SIZE {
            let square = Square::new(file, rank).expect("board coordinates are in range");
            let Some(piece) = position.board.get(square) else {
                continue;
            };
            partial.piece_set(square_to_backend(square), Some(piece_to_backend(piece)));
        }
    }

    for color in [Color::Black, Color::White] {
        let mut backend_hand = BackendHand::default();
        for piece in HAND_PIECE_ORDER {
            let count = position.hand.count(color, piece);
            for _ in 0..count {
                backend_hand = backend_hand
                    .added(piece_kind_to_backend(piece.piece_kind()))
                    .ok_or(RuleError::InvalidHandCount {
                        color,
                        piece,
                        count,
                    })?;
            }
        }
        *partial.hand_of_a_player_mut(color_to_backend(color)) = backend_hand;
    }

    let opponent = position.side_to_move.opponent();
    let mut opponent_partial = partial.clone();
    opponent_partial.side_to_move_set(color_to_backend(opponent));
    if yasai::Position::new(opponent_partial).in_check() {
        return Err(RuleError::OpponentKingInCheck { color: opponent });
    }

    Ok(yasai::Position::new(partial))
}

fn validate_position(position: &Position) -> Result<(), RuleError> {
    if position.ply == 0 {
        return Err(RuleError::InvalidPly { ply: position.ply });
    }

    let mut king_counts = [0_u8; 2];
    let mut piece_counts = [0_u8; 8];
    for rank in 1..=BOARD_SIZE {
        for file in 1..=BOARD_SIZE {
            let square = Square::new(file, rank).expect("board coordinates are in range");
            let Some(piece) = position.board.get(square) else {
                continue;
            };
            if piece.kind == PieceKind::King {
                king_counts[piece.color.index()] += 1;
            }
            piece_counts[material_index(piece.kind)] += 1;
        }
    }
    validate_king_count(Color::Black, king_counts[Color::Black.index()])?;
    validate_king_count(Color::White, king_counts[Color::White.index()])?;

    for color in [Color::Black, Color::White] {
        for piece in HAND_PIECE_ORDER {
            let count = position.hand.count(color, piece);
            if count > maximum_hand_count(piece) {
                return Err(RuleError::InvalidHandCount {
                    color,
                    piece,
                    count,
                });
            }
            piece_counts[material_index(piece.piece_kind())] += count;
        }
    }

    for kind in [
        PieceKind::Pawn,
        PieceKind::Lance,
        PieceKind::Knight,
        PieceKind::Silver,
        PieceKind::Gold,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::King,
    ] {
        let count = piece_counts[material_index(kind)];
        let maximum = maximum_material_count(kind);
        if count > maximum {
            return Err(RuleError::InvalidPieceCount {
                kind,
                count,
                maximum,
            });
        }
    }

    Ok(())
}

const fn maximum_hand_count(piece: HandPiece) -> u8 {
    match piece {
        HandPiece::Pawn => 18,
        HandPiece::Lance | HandPiece::Knight | HandPiece::Silver | HandPiece::Gold => 4,
        HandPiece::Bishop | HandPiece::Rook => 2,
    }
}

const fn material_index(kind: PieceKind) -> usize {
    match kind {
        PieceKind::Pawn | PieceKind::ProPawn => 0,
        PieceKind::Lance | PieceKind::ProLance => 1,
        PieceKind::Knight | PieceKind::ProKnight => 2,
        PieceKind::Silver | PieceKind::ProSilver => 3,
        PieceKind::Gold => 4,
        PieceKind::Bishop | PieceKind::Horse => 5,
        PieceKind::Rook | PieceKind::Dragon => 6,
        PieceKind::King => 7,
    }
}

const fn maximum_material_count(kind: PieceKind) -> u8 {
    match kind {
        PieceKind::Pawn => 18,
        PieceKind::Lance | PieceKind::Knight | PieceKind::Silver | PieceKind::Gold => 4,
        PieceKind::Bishop | PieceKind::Rook | PieceKind::King => 2,
        PieceKind::ProPawn
        | PieceKind::ProLance
        | PieceKind::ProKnight
        | PieceKind::ProSilver
        | PieceKind::Horse
        | PieceKind::Dragon => unreachable!(),
    }
}

fn validate_king_count(color: Color, count: u8) -> Result<(), RuleError> {
    match count {
        1 => Ok(()),
        0 => Err(RuleError::MissingKing { color }),
        _ => Err(RuleError::MultipleKings { color }),
    }
}

const fn color_to_backend(color: Color) -> BackendColor {
    match color {
        Color::Black => BackendColor::Black,
        Color::White => BackendColor::White,
    }
}

const fn piece_kind_to_backend(kind: PieceKind) -> BackendPieceKind {
    match kind {
        PieceKind::Pawn => BackendPieceKind::Pawn,
        PieceKind::Lance => BackendPieceKind::Lance,
        PieceKind::Knight => BackendPieceKind::Knight,
        PieceKind::Silver => BackendPieceKind::Silver,
        PieceKind::Gold => BackendPieceKind::Gold,
        PieceKind::Bishop => BackendPieceKind::Bishop,
        PieceKind::Rook => BackendPieceKind::Rook,
        PieceKind::King => BackendPieceKind::King,
        PieceKind::ProPawn => BackendPieceKind::ProPawn,
        PieceKind::ProLance => BackendPieceKind::ProLance,
        PieceKind::ProKnight => BackendPieceKind::ProKnight,
        PieceKind::ProSilver => BackendPieceKind::ProSilver,
        PieceKind::Horse => BackendPieceKind::ProBishop,
        PieceKind::Dragon => BackendPieceKind::ProRook,
    }
}

const fn piece_kind_from_backend(kind: BackendPieceKind) -> PieceKind {
    match kind {
        BackendPieceKind::Pawn => PieceKind::Pawn,
        BackendPieceKind::Lance => PieceKind::Lance,
        BackendPieceKind::Knight => PieceKind::Knight,
        BackendPieceKind::Silver => PieceKind::Silver,
        BackendPieceKind::Gold => PieceKind::Gold,
        BackendPieceKind::Bishop => PieceKind::Bishop,
        BackendPieceKind::Rook => PieceKind::Rook,
        BackendPieceKind::King => PieceKind::King,
        BackendPieceKind::ProPawn => PieceKind::ProPawn,
        BackendPieceKind::ProLance => PieceKind::ProLance,
        BackendPieceKind::ProKnight => PieceKind::ProKnight,
        BackendPieceKind::ProSilver => PieceKind::ProSilver,
        BackendPieceKind::ProBishop => PieceKind::Horse,
        BackendPieceKind::ProRook => PieceKind::Dragon,
    }
}

fn piece_to_backend(piece: Piece) -> BackendPiece {
    BackendPiece::new(
        piece_kind_to_backend(piece.kind),
        color_to_backend(piece.color),
    )
}

fn square_to_backend(square: Square) -> BackendSquare {
    BackendSquare::new(
        u8::try_from(square.file).expect("square file fits in u8"),
        u8::try_from(square.rank).expect("square rank fits in u8"),
    )
    .expect("validated square converts to shogi_core")
}

fn square_from_backend(square: BackendSquare) -> Square {
    Square::new(usize::from(square.file()), usize::from(square.rank()))
        .expect("shogi_core square is in range")
}

fn move_from_backend(value: BackendMove) -> Move {
    match value {
        BackendMove::Normal { from, to, promote } => Move::Normal {
            from: square_from_backend(from),
            to: square_from_backend(to),
            promote,
        },
        BackendMove::Drop { piece, to } => {
            let kind = piece_kind_from_backend(piece.piece_kind());
            let piece = HandPiece::try_from(kind)
                .expect("yasai generates drops only for droppable piece kinds");
            Move::Drop {
                piece,
                to: square_from_backend(to),
            }
        }
    }
}

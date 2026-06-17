use crate::{Color, HandPiece, Move, Square, BOARD_SQUARES, MOVE_LABEL_PLANES};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoveLabelPlane {
    Up,
    UpLeft,
    UpRight,
    Left,
    Right,
    Down,
    DownLeft,
    DownRight,
    Up2Left,
    Up2Right,
    PromoteUp,
    PromoteUpLeft,
    PromoteUpRight,
    PromoteLeft,
    PromoteRight,
    PromoteDown,
    PromoteDownLeft,
    PromoteDownRight,
    PromoteUp2Left,
    PromoteUp2Right,
    DropPawn,
    DropLance,
    DropKnight,
    DropSilver,
    DropGold,
    DropBishop,
    DropRook,
}

impl MoveLabelPlane {
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Up => 0,
            Self::UpLeft => 1,
            Self::UpRight => 2,
            Self::Left => 3,
            Self::Right => 4,
            Self::Down => 5,
            Self::DownLeft => 6,
            Self::DownRight => 7,
            Self::Up2Left => 8,
            Self::Up2Right => 9,
            Self::PromoteUp => 10,
            Self::PromoteUpLeft => 11,
            Self::PromoteUpRight => 12,
            Self::PromoteLeft => 13,
            Self::PromoteRight => 14,
            Self::PromoteDown => 15,
            Self::PromoteDownLeft => 16,
            Self::PromoteDownRight => 17,
            Self::PromoteUp2Left => 18,
            Self::PromoteUp2Right => 19,
            Self::DropPawn => 20,
            Self::DropLance => 21,
            Self::DropKnight => 22,
            Self::DropSilver => 23,
            Self::DropGold => 24,
            Self::DropBishop => 25,
            Self::DropRook => 26,
        }
    }

    #[must_use]
    pub const fn from_drop_piece(piece: HandPiece) -> Self {
        match piece {
            HandPiece::Pawn => Self::DropPawn,
            HandPiece::Lance => Self::DropLance,
            HandPiece::Knight => Self::DropKnight,
            HandPiece::Silver => Self::DropSilver,
            HandPiece::Gold => Self::DropGold,
            HandPiece::Bishop => Self::DropBishop,
            HandPiece::Rook => Self::DropRook,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoveLabelError {
    PlaneOutOfRange { plane: usize },
    SquareIndexOutOfRange { index: usize },
    ZeroLengthMove { square: Square },
    UnrepresentableMove { from: Square, to: Square },
}

/// Builds a plane-major move label from raw label parts.
///
/// # Errors
///
/// Returns [`MoveLabelError`] when `plane >= 27` or `to_square_index >= 81`.
pub const fn move_label_from_parts(
    plane: usize,
    to_square_index: usize,
) -> Result<usize, MoveLabelError> {
    if plane >= MOVE_LABEL_PLANES {
        return Err(MoveLabelError::PlaneOutOfRange { plane });
    }
    if to_square_index >= BOARD_SQUARES {
        return Err(MoveLabelError::SquareIndexOutOfRange {
            index: to_square_index,
        });
    }

    Ok(plane * BOARD_SQUARES + to_square_index)
}

/// Encodes a move into the fixed 27-plane, plane-major move label space.
///
/// # Errors
///
/// Returns [`MoveLabelError`] when a normal move cannot be represented by the
/// current-side direction planes.
pub fn encode_move_label(value: Move, side_to_move: Color) -> Result<usize, MoveLabelError> {
    let (plane, to_square) = match value {
        Move::Normal { from, to, promote } => {
            let base_plane = normal_move_base_plane(from, to, side_to_move)?;
            let plane = if promote {
                promote_base_plane(base_plane)
            } else {
                base_plane
            };
            (plane, to.to_current_view(side_to_move))
        }
        Move::Drop { piece, to } => (
            MoveLabelPlane::from_drop_piece(piece),
            to.to_current_view(side_to_move),
        ),
    };

    move_label_from_parts(plane.index(), to_square.index())
}

fn normal_move_base_plane(
    from: Square,
    to: Square,
    side_to_move: Color,
) -> Result<MoveLabelPlane, MoveLabelError> {
    if from == to {
        return Err(MoveLabelError::ZeroLengthMove { square: from });
    }

    let from_current = from.to_current_view(side_to_move);
    let to_current = to.to_current_view(side_to_move);
    let dfile = square_coordinate_delta(to_current.file, from_current.file);
    let drank = square_coordinate_delta(to_current.rank, from_current.rank);

    match (dfile, drank) {
        (0, delta) if delta < 0 => Ok(MoveLabelPlane::Up),
        (delta, 0) if delta > 0 => Ok(MoveLabelPlane::Left),
        (delta, 0) if delta < 0 => Ok(MoveLabelPlane::Right),
        (0, delta) if delta > 0 => Ok(MoveLabelPlane::Down),
        (1, -2) => Ok(MoveLabelPlane::Up2Left),
        (-1, -2) => Ok(MoveLabelPlane::Up2Right),
        (file_delta, rank_delta) if file_delta.abs() == rank_delta.abs() => {
            match (file_delta.signum(), rank_delta.signum()) {
                (1, -1) => Ok(MoveLabelPlane::UpLeft),
                (-1, -1) => Ok(MoveLabelPlane::UpRight),
                (1, 1) => Ok(MoveLabelPlane::DownLeft),
                (-1, 1) => Ok(MoveLabelPlane::DownRight),
                _ => Err(MoveLabelError::UnrepresentableMove { from, to }),
            }
        }
        _ => Err(MoveLabelError::UnrepresentableMove { from, to }),
    }
}

fn promote_base_plane(plane: MoveLabelPlane) -> MoveLabelPlane {
    match plane {
        MoveLabelPlane::Up => MoveLabelPlane::PromoteUp,
        MoveLabelPlane::UpLeft => MoveLabelPlane::PromoteUpLeft,
        MoveLabelPlane::UpRight => MoveLabelPlane::PromoteUpRight,
        MoveLabelPlane::Left => MoveLabelPlane::PromoteLeft,
        MoveLabelPlane::Right => MoveLabelPlane::PromoteRight,
        MoveLabelPlane::Down => MoveLabelPlane::PromoteDown,
        MoveLabelPlane::DownLeft => MoveLabelPlane::PromoteDownLeft,
        MoveLabelPlane::DownRight => MoveLabelPlane::PromoteDownRight,
        MoveLabelPlane::Up2Left => MoveLabelPlane::PromoteUp2Left,
        MoveLabelPlane::Up2Right => MoveLabelPlane::PromoteUp2Right,
        MoveLabelPlane::PromoteUp
        | MoveLabelPlane::PromoteUpLeft
        | MoveLabelPlane::PromoteUpRight
        | MoveLabelPlane::PromoteLeft
        | MoveLabelPlane::PromoteRight
        | MoveLabelPlane::PromoteDown
        | MoveLabelPlane::PromoteDownLeft
        | MoveLabelPlane::PromoteDownRight
        | MoveLabelPlane::PromoteUp2Left
        | MoveLabelPlane::PromoteUp2Right
        | MoveLabelPlane::DropPawn
        | MoveLabelPlane::DropLance
        | MoveLabelPlane::DropKnight
        | MoveLabelPlane::DropSilver
        | MoveLabelPlane::DropGold
        | MoveLabelPlane::DropBishop
        | MoveLabelPlane::DropRook => {
            unreachable!("normal_move_base_plane returns only base direction planes")
        }
    }
}

fn square_coordinate_delta(to: usize, from: usize) -> i16 {
    i16::try_from(to).expect("square coordinate fits in i16")
        - i16::try_from(from).expect("square coordinate fits in i16")
}

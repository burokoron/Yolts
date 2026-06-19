use crate::{
    Color, Move, PieceKind, Position, Square, BOARD_SQUARES, MOVE_LABEL_PLANES, MOVE_LABEL_SIZE,
};

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
        self as usize
    }

    /// 固定されたplane indexから指し手ラベルplaneを作る。
    ///
    /// # Errors
    ///
    /// `index >= 27`の場合に返す。
    pub const fn from_index(index: usize) -> Result<Self, MoveLabelError> {
        match index {
            0 => Ok(Self::Up),
            1 => Ok(Self::UpLeft),
            2 => Ok(Self::UpRight),
            3 => Ok(Self::Left),
            4 => Ok(Self::Right),
            5 => Ok(Self::Down),
            6 => Ok(Self::DownLeft),
            7 => Ok(Self::DownRight),
            8 => Ok(Self::Up2Left),
            9 => Ok(Self::Up2Right),
            10 => Ok(Self::PromoteUp),
            11 => Ok(Self::PromoteUpLeft),
            12 => Ok(Self::PromoteUpRight),
            13 => Ok(Self::PromoteLeft),
            14 => Ok(Self::PromoteRight),
            15 => Ok(Self::PromoteDown),
            16 => Ok(Self::PromoteDownLeft),
            17 => Ok(Self::PromoteDownRight),
            18 => Ok(Self::PromoteUp2Left),
            19 => Ok(Self::PromoteUp2Right),
            20 => Ok(Self::DropPawn),
            21 => Ok(Self::DropLance),
            22 => Ok(Self::DropKnight),
            23 => Ok(Self::DropSilver),
            24 => Ok(Self::DropGold),
            25 => Ok(Self::DropBishop),
            26 => Ok(Self::DropRook),
            _ => Err(MoveLabelError::PlaneOutOfRange { plane: index }),
        }
    }

    const fn from_drop_piece(kind: PieceKind) -> Result<Self, MoveLabelError> {
        match kind {
            PieceKind::Pawn => Ok(Self::DropPawn),
            PieceKind::Lance => Ok(Self::DropLance),
            PieceKind::Knight => Ok(Self::DropKnight),
            PieceKind::Silver => Ok(Self::DropSilver),
            PieceKind::Gold => Ok(Self::DropGold),
            PieceKind::Bishop => Ok(Self::DropBishop),
            PieceKind::Rook => Ok(Self::DropRook),
            _ => Err(MoveLabelError::UndroppablePiece { kind }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoveLabelParts {
    pub plane: MoveLabelPlane,
    pub to_square_current_view: Square,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoveLabelError {
    LabelOutOfRange { label: usize },
    PlaneOutOfRange { plane: usize },
    SquareIndexOutOfRange { index: usize },
    ZeroLengthMove { square: Square },
    UnrepresentableMove { from: Square, to: Square },
    UndroppablePiece { kind: PieceKind },
    NotLegalInPosition { label: usize },
    DuplicateLegalLabel { label: usize },
}

/// plane-major形式の指し手ラベルを作る。
///
/// # Errors
///
/// planeまたは着地点indexが範囲外の場合に返す。
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

/// 指し手ラベルをplaneと現在手番視点の着地点へ分解する。
///
/// # Errors
///
/// ラベルが固定空間の範囲外の場合に返す。
pub fn move_label_parts(label: usize) -> Result<MoveLabelParts, MoveLabelError> {
    if label >= MOVE_LABEL_PLANES * BOARD_SQUARES {
        return Err(MoveLabelError::LabelOutOfRange { label });
    }
    let plane = MoveLabelPlane::from_index(label / BOARD_SQUARES)?;
    let index = label % BOARD_SQUARES;
    let file =
        u8::try_from(index % 9 + 1).map_err(|_| MoveLabelError::SquareIndexOutOfRange { index })?;
    let rank =
        u8::try_from(index / 9 + 1).map_err(|_| MoveLabelError::SquareIndexOutOfRange { index })?;
    let to_square_current_view =
        Square::new(file, rank).ok_or(MoveLabelError::SquareIndexOutOfRange { index })?;
    Ok(MoveLabelParts {
        plane,
        to_square_current_view,
    })
}

/// 指し手を固定27 planeのラベルへ符号化する。
///
/// # Errors
///
/// 通常着手を方向planeで表せない場合、または打てない駒種の場合に返す。
pub fn encode_move_label(value: Move, side_to_move: Color) -> Result<usize, MoveLabelError> {
    let (plane, to) = match value {
        Move::Normal { from, to, promote } => {
            let base = normal_move_base_plane(from, to, side_to_move)?;
            (if promote { promoted_plane(base) } else { base }, to)
        }
        Move::Drop { piece, to } => (MoveLabelPlane::from_drop_piece(piece.piece_kind())?, to),
    };
    let current_to = current_view(to, side_to_move);
    move_label_from_parts(plane.index(), plane_square_index(current_to))
}

/// 現在局面の合法手を固定27 planeのラベルへ符号化する。
///
/// 返却順は[`Position::legal_moves`]と一致する。合法手がない場合は空のリストを返す。
///
/// # Errors
///
/// 合法手をラベルで表せない場合、または複数の合法手が同じラベルになる場合に返す。
pub fn encode_legal_moves(position: &Position) -> Result<Vec<usize>, MoveLabelError> {
    let moves = position.legal_moves();
    let mut labels = Vec::with_capacity(moves.len());
    let mut seen = [false; MOVE_LABEL_SIZE];
    for value in moves {
        let label = encode_move_label(value, position.side_to_move())?;
        if seen[label] {
            return Err(MoveLabelError::DuplicateLegalLabel { label });
        }
        seen[label] = true;
        labels.push(label);
    }
    Ok(labels)
}

/// 現在局面の合法手ラベルだけを`true`にした固定長maskを作る。
///
/// 合法手がない場合は全要素が`false`のmaskを返す。
///
/// # Errors
///
/// 合法手をラベルで表せない場合、または複数の合法手が同じラベルになる場合に返す。
pub fn make_legal_move_mask(
    position: &Position,
) -> Result<[bool; MOVE_LABEL_SIZE], MoveLabelError> {
    let mut mask = [false; MOVE_LABEL_SIZE];
    for value in position.legal_moves() {
        let label = encode_move_label(value, position.side_to_move())?;
        if mask[label] {
            return Err(MoveLabelError::DuplicateLegalLabel { label });
        }
        mask[label] = true;
    }
    Ok(mask)
}

/// ラベルに一致する合法手をYasaiの合法手集合から復号する。
/// 同じ着地点と方向に複数の駒が並ぶ場合、手前の駒が奥の駒を遮るため、
/// 合法手集合では1つのラベルに最大1手だけが対応する。
///
/// # Errors
///
/// ラベルが不正または現在局面で非合法の場合に返す。
pub fn decode_move_label(label: usize, position: &Position) -> Result<Move, MoveLabelError> {
    if label >= MOVE_LABEL_PLANES * BOARD_SQUARES {
        return Err(MoveLabelError::LabelOutOfRange { label });
    }
    position
        .legal_moves()
        .find(|&value| encode_move_label(value, position.side_to_move()) == Ok(label))
        .ok_or(MoveLabelError::NotLegalInPosition { label })
}

fn normal_move_base_plane(
    from: Square,
    to: Square,
    side_to_move: Color,
) -> Result<MoveLabelPlane, MoveLabelError> {
    if from == to {
        return Err(MoveLabelError::ZeroLengthMove { square: from });
    }
    let from = current_view(from, side_to_move);
    let to = current_view(to, side_to_move);
    let dfile = i16::from(to.file()) - i16::from(from.file());
    let drank = i16::from(to.rank()) - i16::from(from.rank());
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

const fn promoted_plane(plane: MoveLabelPlane) -> MoveLabelPlane {
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
        _ => unreachable!(),
    }
}

fn current_view(square: Square, side_to_move: Color) -> Square {
    match side_to_move {
        Color::Black => square,
        Color::White => square.flip(),
    }
}

fn plane_square_index(square: Square) -> usize {
    usize::from(square.rank() - 1) * 9 + usize::from(square.file() - 1)
}

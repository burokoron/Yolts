use crate::{
    Color, HandPiece, Move, PieceKind, Position, Square, BOARD_SIZE, BOARD_SQUARES,
    MOVE_LABEL_PLANES, MOVE_LABEL_SIZE,
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

    /// 固定された plane index から指し手ラベル plane を作る。
    ///
    /// # Errors
    ///
    /// `index >= 27` の場合は [`MoveLabelError::PlaneOutOfRange`] を返す。
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

    const fn drop_piece(self) -> Option<HandPiece> {
        match self {
            Self::DropPawn => Some(HandPiece::Pawn),
            Self::DropLance => Some(HandPiece::Lance),
            Self::DropKnight => Some(HandPiece::Knight),
            Self::DropSilver => Some(HandPiece::Silver),
            Self::DropGold => Some(HandPiece::Gold),
            Self::DropBishop => Some(HandPiece::Bishop),
            Self::DropRook => Some(HandPiece::Rook),
            Self::Up
            | Self::UpLeft
            | Self::UpRight
            | Self::Left
            | Self::Right
            | Self::Down
            | Self::DownLeft
            | Self::DownRight
            | Self::Up2Left
            | Self::Up2Right
            | Self::PromoteUp
            | Self::PromoteUpLeft
            | Self::PromoteUpRight
            | Self::PromoteLeft
            | Self::PromoteRight
            | Self::PromoteDown
            | Self::PromoteDownLeft
            | Self::PromoteDownRight
            | Self::PromoteUp2Left
            | Self::PromoteUp2Right => None,
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
    OccupiedDropDestination { square: Square },
    OwnPieceOnDestination { square: Square },
    MissingDropPiece { color: Color, piece: HandPiece },
    NoMatchingSource { plane: MoveLabelPlane, to: Square },
}

/// plane-major 形式の指し手ラベルを plane と着地点 index から作る。
///
/// # Errors
///
/// `plane >= 27` または `to_square_index >= 81` の場合は [`MoveLabelError`] を返す。
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

/// 指し手ラベルを固定 plane と現在手番視点の着地点 square に分解する。
///
/// # Errors
///
/// `label >= MOVE_LABEL_SIZE` の場合は [`MoveLabelError::LabelOutOfRange`] を返す。
pub fn move_label_parts(label: usize) -> Result<MoveLabelParts, MoveLabelError> {
    if label >= MOVE_LABEL_SIZE {
        return Err(MoveLabelError::LabelOutOfRange { label });
    }

    let plane = MoveLabelPlane::from_index(label / BOARD_SQUARES)?;
    let square_index = label % BOARD_SQUARES;
    let to_square_current_view =
        Square::from_index(square_index).map_err(|_| MoveLabelError::SquareIndexOutOfRange {
            index: square_index,
        })?;

    Ok(MoveLabelParts {
        plane,
        to_square_current_view,
    })
}

/// 指し手を固定 27 plane、plane-major order の指し手ラベル空間へ符号化する。
///
/// # Errors
///
/// 通常着手を現在手番視点の方向 plane で表せない場合は [`MoveLabelError`] を返す。
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

/// 指し手ラベルを、指定局面上の通常着手候補へ復号する。
///
/// この関数は、現在局面でそのラベルを表せる最初の盤上着手候補を復元する。
/// 自玉の王手放置、二歩、打ち歩詰めなどの完全な合法手判定は行わない。
///
/// # Errors
///
/// ラベルが範囲外の場合、持ち駒から打てない場合、またはその plane と着地点に
/// 対応する移動元 square が局面上に見つからない場合は [`MoveLabelError`] を返す。
pub fn decode_move_label(label: usize, position: &Position) -> Result<Move, MoveLabelError> {
    let parts = move_label_parts(label)?;
    let side_to_move = position.side_to_move;
    let to = parts.to_square_current_view.from_current_view(side_to_move);

    if let Some(piece) = parts.plane.drop_piece() {
        return decode_drop(piece, to, position);
    }

    if position
        .board
        .get(to)
        .is_some_and(|piece| piece.color == side_to_move)
    {
        return Err(MoveLabelError::OwnPieceOnDestination { square: to });
    }

    let promote = is_promote_plane(parts.plane);
    let base_plane = unpromote_plane(parts.plane);
    let Some(from) = find_matching_source(base_plane, to, promote, position) else {
        return Err(MoveLabelError::NoMatchingSource {
            plane: parts.plane,
            to,
        });
    };

    Ok(Move::Normal { from, to, promote })
}

fn decode_drop(piece: HandPiece, to: Square, position: &Position) -> Result<Move, MoveLabelError> {
    if position.board.get(to).is_some() {
        return Err(MoveLabelError::OccupiedDropDestination { square: to });
    }
    if position.hand.count(position.side_to_move, piece) == 0 {
        return Err(MoveLabelError::MissingDropPiece {
            color: position.side_to_move,
            piece,
        });
    }

    Ok(Move::Drop { piece, to })
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

fn is_promote_plane(plane: MoveLabelPlane) -> bool {
    matches!(
        plane,
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
    )
}

fn unpromote_plane(plane: MoveLabelPlane) -> MoveLabelPlane {
    match plane {
        MoveLabelPlane::PromoteUp => MoveLabelPlane::Up,
        MoveLabelPlane::PromoteUpLeft => MoveLabelPlane::UpLeft,
        MoveLabelPlane::PromoteUpRight => MoveLabelPlane::UpRight,
        MoveLabelPlane::PromoteLeft => MoveLabelPlane::Left,
        MoveLabelPlane::PromoteRight => MoveLabelPlane::Right,
        MoveLabelPlane::PromoteDown => MoveLabelPlane::Down,
        MoveLabelPlane::PromoteDownLeft => MoveLabelPlane::DownLeft,
        MoveLabelPlane::PromoteDownRight => MoveLabelPlane::DownRight,
        MoveLabelPlane::PromoteUp2Left => MoveLabelPlane::Up2Left,
        MoveLabelPlane::PromoteUp2Right => MoveLabelPlane::Up2Right,
        base_plane => base_plane,
    }
}

fn find_matching_source(
    base_plane: MoveLabelPlane,
    to: Square,
    promote: bool,
    position: &Position,
) -> Option<Square> {
    let to_current = to.to_current_view(position.side_to_move);
    let (dfile, drank) = base_plane_delta(base_plane);

    if matches!(
        base_plane,
        MoveLabelPlane::Up2Left | MoveLabelPlane::Up2Right
    ) {
        let source_current = offset_square(to_current, -dfile, -drank)?;
        let source = source_current.from_current_view(position.side_to_move);
        let piece = position.board.get(source)?;

        return (piece.color == position.side_to_move
            && piece_matches_label(
                piece.kind,
                base_plane,
                1,
                promote,
                source_current,
                to_current,
            ))
        .then_some(source);
    }

    let mut distance = 1;
    let mut source_current = offset_square(to_current, -dfile, -drank);
    while let Some(current) = source_current {
        let source = current.from_current_view(position.side_to_move);
        if let Some(piece) = position.board.get(source) {
            if piece.color != position.side_to_move {
                return None;
            }

            return piece_matches_label(
                piece.kind, base_plane, distance, promote, current, to_current,
            )
            .then_some(source);
        }

        distance += 1;
        source_current = offset_square(current, -dfile, -drank);
    }

    None
}

fn piece_matches_label(
    kind: PieceKind,
    base_plane: MoveLabelPlane,
    distance: usize,
    promote: bool,
    from_current: Square,
    to_current: Square,
) -> bool {
    if promote {
        return kind.promoted().is_some()
            && is_promotion_zone_move(from_current, to_current)
            && piece_can_move_on_base(kind, base_plane, distance);
    }

    piece_can_move_on_base(kind, base_plane, distance)
}

fn is_promotion_zone_move(from_current: Square, to_current: Square) -> bool {
    from_current.rank <= 3 || to_current.rank <= 3
}

fn piece_can_move_on_base(kind: PieceKind, base_plane: MoveLabelPlane, distance: usize) -> bool {
    use MoveLabelPlane::{
        Down, DownLeft, DownRight, Left, Right, Up, Up2Left, Up2Right, UpLeft, UpRight,
    };
    use PieceKind::{
        Bishop, Dragon, Gold, Horse, King, Knight, Lance, Pawn, ProKnight, ProLance, ProPawn,
        ProSilver, Rook, Silver,
    };

    match kind {
        Pawn => distance == 1 && base_plane == Up,
        Lance => base_plane == Up,
        Knight => distance == 1 && matches!(base_plane, Up2Left | Up2Right),
        Silver => {
            distance == 1 && matches!(base_plane, Up | UpLeft | UpRight | DownLeft | DownRight)
        }
        Gold | ProPawn | ProLance | ProKnight | ProSilver => {
            distance == 1 && matches!(base_plane, Up | UpLeft | UpRight | Left | Right | Down)
        }
        Bishop => matches!(base_plane, UpLeft | UpRight | DownLeft | DownRight),
        Rook => matches!(base_plane, Up | Left | Right | Down),
        King => distance == 1,
        Horse => {
            matches!(base_plane, UpLeft | UpRight | DownLeft | DownRight)
                || (distance == 1 && matches!(base_plane, Up | Left | Right | Down))
        }
        Dragon => {
            matches!(base_plane, Up | Left | Right | Down)
                || (distance == 1 && matches!(base_plane, UpLeft | UpRight | DownLeft | DownRight))
        }
    }
}

fn base_plane_delta(plane: MoveLabelPlane) -> (i16, i16) {
    match plane {
        MoveLabelPlane::Up => (0, -1),
        MoveLabelPlane::UpLeft => (1, -1),
        MoveLabelPlane::UpRight => (-1, -1),
        MoveLabelPlane::Left => (1, 0),
        MoveLabelPlane::Right => (-1, 0),
        MoveLabelPlane::Down => (0, 1),
        MoveLabelPlane::DownLeft => (1, 1),
        MoveLabelPlane::DownRight => (-1, 1),
        MoveLabelPlane::Up2Left => (1, -2),
        MoveLabelPlane::Up2Right => (-1, -2),
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
            unreachable!("decode uses only unpromoted normal movement planes")
        }
    }
}

fn offset_square(square: Square, dfile: i16, drank: i16) -> Option<Square> {
    let file = i16::try_from(square.file).expect("square coordinate fits in i16") + dfile;
    let rank = i16::try_from(square.rank).expect("square coordinate fits in i16") + drank;

    if !(1..=i16::try_from(BOARD_SIZE).expect("board size fits in i16")).contains(&file)
        || !(1..=i16::try_from(BOARD_SIZE).expect("board size fits in i16")).contains(&rank)
    {
        return None;
    }

    Square::new(
        usize::try_from(file).expect("bounded file is positive"),
        usize::try_from(rank).expect("bounded rank is positive"),
    )
    .ok()
}

fn square_coordinate_delta(to: usize, from: usize) -> i16 {
    i16::try_from(to).expect("square coordinate fits in i16")
        - i16::try_from(from).expect("square coordinate fits in i16")
}

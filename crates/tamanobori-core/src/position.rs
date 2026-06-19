use shogi_core::{Color, Hand, Move, PartialPosition, Piece, PieceKind, Square};
use shogi_usi_parser::FromUsi;

pub const STARTPOS_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

const HAND_PIECE_KINDS: [PieceKind; 7] = [
    PieceKind::Rook,
    PieceKind::Bishop,
    PieceKind::Gold,
    PieceKind::Silver,
    PieceKind::Knight,
    PieceKind::Lance,
    PieceKind::Pawn,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepetitionOutcome {
    None,
    Draw,
    Win,
    Loss,
}

#[derive(Debug)]
pub enum PositionError {
    InvalidSfen(shogi_usi_parser::Error),
    InvalidSfenFieldCount {
        fields: usize,
    },
    MissingKing {
        color: Color,
    },
    MultipleKings {
        color: Color,
    },
    InvalidHandCount {
        color: Color,
        kind: PieceKind,
        count: u8,
        maximum: u8,
    },
    InvalidPieceCount {
        kind: PieceKind,
        count: u16,
        maximum: u16,
    },
    DeadPiece {
        piece: Piece,
        square: Square,
    },
    OpponentKingInCheck {
        color: Color,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Transition {
    Initial,
    Move(Move),
    NullMove,
}

#[derive(Clone, Copy, Debug)]
struct HistoryEntry {
    keys: (u64, u64),
    side_to_move: Color,
    in_check: bool,
    transition: Transition,
}

#[derive(Clone, Debug)]
pub struct Position {
    inner: yasai::Position,
    history: Vec<HistoryEntry>,
}

impl Position {
    /// 平手初期局面を構築する。
    ///
    /// # Panics
    ///
    /// `shogi_core`の組み込み初期局面がYasaiの前提を満たさない場合にのみpanicする。
    #[must_use]
    pub fn startpos() -> Self {
        Self::from_partial(PartialPosition::startpos())
            .expect("shogi_core start position must be valid")
    }

    /// SFENから局面を構築する。
    ///
    /// # Errors
    ///
    /// `SFEN`の構文またはYasaiが前提とする局面条件が不正な場合に返す。
    pub fn from_sfen(sfen: &str) -> Result<Self, PositionError> {
        let fields = sfen.split_ascii_whitespace().count();
        if fields != 4 {
            return Err(PositionError::InvalidSfenFieldCount { fields });
        }
        let partial = PartialPosition::from_usi(&format!("sfen {sfen}"))
            .map_err(PositionError::InvalidSfen)?;
        Self::from_partial(partial)
    }

    fn from_partial(partial: PartialPosition) -> Result<Self, PositionError> {
        validate_position(&partial)?;
        let opponent = partial.side_to_move().flip();
        let mut opponent_partial = partial.clone();
        opponent_partial.side_to_move_set(opponent);
        if yasai::Position::new(opponent_partial).in_check() {
            return Err(PositionError::OpponentKingInCheck { color: opponent });
        }

        let inner = yasai::Position::new(partial);
        let mut position = Self {
            inner,
            history: Vec::new(),
        };
        position.push_history(Transition::Initial);
        Ok(position)
    }

    #[must_use]
    pub fn to_sfen(&self) -> String {
        self.to_partial_position().to_sfen_owned()
    }

    #[must_use]
    pub fn side_to_move(&self) -> Color {
        self.inner.side_to_move()
    }

    #[must_use]
    pub fn ply(&self) -> u16 {
        self.inner.ply()
    }

    #[must_use]
    pub fn hand(&self, color: Color) -> Hand {
        self.inner.hand(color)
    }

    #[must_use]
    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        self.inner.piece_at(square)
    }

    #[must_use]
    pub fn key(&self) -> u64 {
        self.inner.key()
    }

    #[must_use]
    pub fn keys(&self) -> (u64, u64) {
        self.inner.keys()
    }

    #[must_use]
    pub fn in_check(&self) -> bool {
        self.inner.in_check()
    }

    #[must_use]
    pub fn legal_moves(&self) -> impl ExactSizeIterator<Item = Move> {
        self.inner.legal_moves().into_iter()
    }

    /// 合法手を進める。
    ///
    /// # Panics
    ///
    /// `value`が現在局面の合法手でない場合にpanicする可能性がある。
    pub fn do_move(&mut self, value: Move) {
        self.inner.do_move(value);
        self.push_history(Transition::Move(value));
    }

    /// 直前の着手を戻す。
    ///
    /// # Panics
    ///
    /// `value`が直前に進めた着手と一致しない場合にpanicする。
    pub fn undo_move(&mut self, value: Move) {
        assert_eq!(
            self.history.last().map(|entry| entry.transition),
            Some(Transition::Move(value)),
            "undo_move must match the latest move"
        );
        self.inner.undo_move(value);
        self.history.pop();
    }

    /// 王手されていない局面でnull moveを進める。
    ///
    /// # Panics
    ///
    /// 王手中に呼んだ場合にpanicする。Yasaiのnull moveは王手情報を引き継がないため、
    /// 呼び出し側は探索のnull move pruning条件で王手中を除外する必要がある。
    pub fn do_null_move(&mut self) {
        assert!(
            !self.in_check(),
            "null move must not be played while in check"
        );
        self.inner.do_null_move();
        self.push_history(Transition::NullMove);
    }

    /// 直前のnull moveを戻す。
    ///
    /// # Panics
    ///
    /// 直前の操作がnull moveでない場合にpanicする。
    pub fn undo_null_move(&mut self) {
        assert_eq!(
            self.history.last().map(|entry| entry.transition),
            Some(Transition::NullMove),
            "undo_null_move must match the latest move"
        );
        self.inner.undo_null_move();
        self.history.pop();
    }

    #[must_use]
    pub fn repetition_outcome(&self) -> RepetitionOutcome {
        let start = self
            .history
            .iter()
            .rposition(|entry| entry.transition == Transition::NullMove)
            .map_or(0, |index| index + 1);
        let current_keys = self.keys();
        let repetitions = self.history[start..]
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, entry)| entry.keys == current_keys)
            .map(|(index, _)| start + index)
            .take(4)
            .collect::<Vec<_>>();
        if repetitions.len() < 4 {
            return RepetitionOutcome::None;
        }

        let oldest = repetitions[3];
        for checker in Color::all() {
            let mut moved = false;
            let continuous = self.history[oldest + 1..].iter().all(|entry| {
                if !matches!(entry.transition, Transition::Move(_))
                    || entry.side_to_move.flip() != checker
                {
                    return true;
                }
                moved = true;
                entry.in_check
            });
            if moved && continuous {
                return if self.side_to_move() == checker {
                    RepetitionOutcome::Loss
                } else {
                    RepetitionOutcome::Win
                };
            }
        }
        RepetitionOutcome::Draw
    }

    #[must_use]
    pub fn has_no_legal_moves(&self) -> bool {
        self.inner.legal_moves().is_empty()
    }

    /// 現在手番側が王手され、合法な応手がないかを返す。
    /// 王手されていないが合法手がないステイルメイトは含めない。
    #[must_use]
    pub fn is_mated(&self) -> bool {
        self.in_check() && self.has_no_legal_moves()
    }

    pub fn mate_in_one_moves(&mut self) -> Vec<Move> {
        let moves = self.legal_moves().collect::<Vec<_>>();
        let mut mates = Vec::new();
        for value in moves {
            self.do_move(value);
            if self.is_mated() {
                mates.push(value);
            }
            self.undo_move(value);
        }
        mates
    }

    fn push_history(&mut self, transition: Transition) {
        self.history.push(HistoryEntry {
            keys: self.inner.keys(),
            side_to_move: self.inner.side_to_move(),
            in_check: self.inner.in_check(),
            transition,
        });
    }

    fn to_partial_position(&self) -> PartialPosition {
        let mut partial = PartialPosition::empty();
        partial.side_to_move_set(self.side_to_move());
        assert!(partial.ply_set(self.ply()), "Yasai ply is always nonzero");
        for square in Square::all() {
            partial.piece_set(square, self.piece_at(square));
        }
        for color in Color::all() {
            *partial.hand_of_a_player_mut(color) = self.hand(color);
        }
        partial
    }
}

fn validate_position(position: &PartialPosition) -> Result<(), PositionError> {
    let mut king_counts = [0_u8; 2];
    let mut piece_counts = [0_u16; 8];
    for square in Square::all() {
        let Some(piece) = position.piece_at(square) else {
            continue;
        };
        let kind = piece.piece_kind();
        if is_dead_piece(kind, square.relative_rank(piece.color())) {
            return Err(PositionError::DeadPiece { piece, square });
        }
        if kind == PieceKind::King {
            king_counts[piece.color().array_index()] += 1;
        }
        piece_counts[material_index(kind)] += 1;
    }
    for color in Color::all() {
        match king_counts[color.array_index()] {
            0 => return Err(PositionError::MissingKing { color }),
            1 => {}
            _ => return Err(PositionError::MultipleKings { color }),
        }
        for kind in HAND_PIECE_KINDS {
            let count = hand_piece_count(position.hand_of_a_player(color), kind);
            let maximum = maximum_hand_count(kind);
            if count > maximum {
                return Err(PositionError::InvalidHandCount {
                    color,
                    kind,
                    count,
                    maximum,
                });
            }
            piece_counts[material_index(kind)] += u16::from(count);
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
            return Err(PositionError::InvalidPieceCount {
                kind,
                count,
                maximum,
            });
        }
    }
    Ok(())
}

pub(crate) fn hand_piece_count(hand: Hand, kind: PieceKind) -> u8 {
    hand.count(kind)
        .expect("HAND_PIECE_KINDS contains only droppable pieces")
}

const fn is_dead_piece(kind: PieceKind, relative_rank: u8) -> bool {
    match kind {
        PieceKind::Pawn | PieceKind::Lance => relative_rank == 1,
        PieceKind::Knight => relative_rank <= 2,
        _ => false,
    }
}

const fn maximum_hand_count(kind: PieceKind) -> u8 {
    match kind {
        PieceKind::Pawn => 18,
        PieceKind::Lance | PieceKind::Knight | PieceKind::Silver | PieceKind::Gold => 4,
        PieceKind::Bishop | PieceKind::Rook => 2,
        _ => unreachable!(),
    }
}

const fn material_index(kind: PieceKind) -> usize {
    match kind {
        PieceKind::Pawn | PieceKind::ProPawn => 0,
        PieceKind::Lance | PieceKind::ProLance => 1,
        PieceKind::Knight | PieceKind::ProKnight => 2,
        PieceKind::Silver | PieceKind::ProSilver => 3,
        PieceKind::Gold => 4,
        PieceKind::Bishop | PieceKind::ProBishop => 5,
        PieceKind::Rook | PieceKind::ProRook => 6,
        PieceKind::King => 7,
    }
}

const fn maximum_material_count(kind: PieceKind) -> u16 {
    match kind {
        PieceKind::Pawn => 18,
        PieceKind::Lance | PieceKind::Knight | PieceKind::Silver | PieceKind::Gold => 4,
        PieceKind::Bishop | PieceKind::Rook | PieceKind::King => 2,
        PieceKind::ProPawn
        | PieceKind::ProLance
        | PieceKind::ProKnight
        | PieceKind::ProSilver
        | PieceKind::ProBishop
        | PieceKind::ProRook => unreachable!(),
    }
}

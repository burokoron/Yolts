use shogi_core::{Color, Hand, Move, PartialPosition, Piece, Square};
use shogi_usi_parser::FromUsi;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepetitionOutcome {
    None,
    Draw,
    Win,
    Loss,
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
    #[must_use]
    pub fn startpos() -> Self {
        Self::from_partial(PartialPosition::startpos())
    }

    /// SFENから局面を構築する。
    ///
    /// # Errors
    ///
    /// `SFEN`の構文が不正な場合に返す。局面の合法性は呼び出し側の責務とする。
    ///
    /// # Panics
    ///
    /// `shogi_usi_parser`が受理した局面でも、持ち駒数などがYasaiの前提を
    /// 満たさない場合は、Yasaiの局面構築中にpanicする可能性がある。
    pub fn from_sfen(sfen: &str) -> Result<Self, shogi_usi_parser::Error> {
        PartialPosition::from_usi(&format!("sfen {sfen}")).map(Self::from_partial)
    }

    fn from_partial(partial: PartialPosition) -> Self {
        let inner = yasai::Position::new(partial);
        let mut position = Self {
            inner,
            history: Vec::new(),
        };
        position.push_history(Transition::Initial);
        position
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

    /// 現在手番側が王手され、合法な応手がないかを返す。
    /// 王手されていないが合法手がないステイルメイトは含めない。
    #[must_use]
    pub fn is_mated(&self) -> bool {
        self.in_check() && self.inner.legal_moves().is_empty()
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

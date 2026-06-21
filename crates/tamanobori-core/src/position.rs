use shogi_core::{Color, Move, PartialPosition, Square};
use shogi_usi_parser::FromUsi;

pub use yasai::{DeclarationRule, NyugyokuStats, RepetitionOutcome};

/// 現在手番側から見たルール由来の終局結果。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuleTerminalOutcome {
    NoLegalMovesLoss,
    RepetitionDraw,
    PerpetualCheckWin,
    PerpetualCheckLoss,
    MaxPlyDraw,
}

#[derive(Clone, Debug)]
pub struct Position {
    inner: yasai::Position,
}

impl AsRef<yasai::Position> for Position {
    fn as_ref(&self) -> &yasai::Position {
        &self.inner
    }
}

/// 合法手の有無、千日手、最大手数から終局結果を合成する。
///
/// 合法手なし、千日手・連続王手、最大手数の順に判定し、先に成立した結果を優先する。
/// `Superior` / `Inferior` は探索省略用の関係であり、終局にはしない。
#[must_use]
pub(crate) fn rule_terminal_outcome_from_state(
    has_legal_moves: bool,
    repetition: RepetitionOutcome,
    max_ply: Option<u16>,
    current_ply: u16,
) -> Option<RuleTerminalOutcome> {
    if !has_legal_moves {
        return Some(RuleTerminalOutcome::NoLegalMovesLoss);
    }

    match repetition {
        RepetitionOutcome::None | RepetitionOutcome::Superior | RepetitionOutcome::Inferior => {}
        RepetitionOutcome::Draw => return Some(RuleTerminalOutcome::RepetitionDraw),
        RepetitionOutcome::Win => return Some(RuleTerminalOutcome::PerpetualCheckWin),
        RepetitionOutcome::Loss => return Some(RuleTerminalOutcome::PerpetualCheckLoss),
    }

    max_ply
        .is_some_and(|limit| current_ply >= limit)
        .then_some(RuleTerminalOutcome::MaxPlyDraw)
}

impl Position {
    /// 平手初期局面を構築する。
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
        Self {
            inner: yasai::Position::new(partial),
        }
    }

    #[must_use]
    pub fn to_sfen(&self) -> String {
        self.to_partial_position().to_sfen_owned()
    }

    /// 合法手を進める。
    ///
    /// # Panics
    ///
    /// `value`が現在局面の合法手でない場合にpanicする可能性がある。
    pub fn do_move(&mut self, value: Move) {
        self.inner.do_move(value);
    }

    /// 直前の着手を戻す。
    ///
    /// `value`は直前に進めた着手と一致していなければならない。
    /// この事前条件を満たすことは呼び出し側の責務である。
    pub fn undo_move(&mut self, value: Move) {
        self.inner.undo_move(value);
    }

    /// 王手されていない局面でnull moveを進める。
    ///
    /// # Panics
    ///
    /// 王手中に呼んだ場合にpanicする。Yasaiのnull moveは王手情報を引き継がないため、
    /// 呼び出し側は探索のnull move pruning条件で王手中を除外する必要がある。
    pub fn do_null_move(&mut self) {
        assert!(
            !self.inner.in_check(),
            "null move must not be played while in check"
        );
        self.inner.do_null_move();
    }

    /// 直前のnull moveを戻す。
    ///
    /// 直前の操作がnull moveであることは呼び出し側の責務である。
    pub fn undo_null_move(&mut self) {
        self.inner.undo_null_move();
    }

    #[must_use]
    pub fn repetition_outcome(&self) -> RepetitionOutcome {
        self.inner.repetition_outcome()
    }

    /// 合法手、千日手、最大手数から現在局面の終局結果を返す。
    ///
    /// 合法手なし、千日手、最大手数の順に判定し、先に成立した結果を優先する。
    /// そのため、最大手数に到達した局面でも合法手がなければ敗北を返す。
    /// 宣言勝ちは通常着手とは別のrule actionとして扱うため、この判定には含めない。
    #[must_use]
    pub fn rule_terminal_outcome(&self, max_ply: Option<u16>) -> Option<RuleTerminalOutcome> {
        rule_terminal_outcome_from_state(
            !self.inner.legal_moves().is_empty(),
            self.repetition_outcome(),
            max_ply,
            self.inner.ply(),
        )
    }

    #[must_use]
    pub fn can_declare_win(&self, rule: DeclarationRule) -> bool {
        self.inner.can_declare_win(rule)
    }

    #[must_use]
    pub fn nyugyoku_stats(&self, rule: DeclarationRule, side: Color) -> NyugyokuStats {
        self.inner.nyugyoku_stats(rule, side)
    }

    #[must_use]
    pub fn mate_in_one(&self) -> Option<Move> {
        self.inner.mate_in_one()
    }

    fn to_partial_position(&self) -> PartialPosition {
        let mut partial = PartialPosition::empty();
        partial.side_to_move_set(self.inner.side_to_move());
        assert!(
            partial.ply_set(self.inner.ply()),
            "Yasai ply is always nonzero"
        );
        for square in Square::all() {
            partial.piece_set(square, self.inner.piece_at(square));
        }
        for color in Color::all() {
            *partial.hand_of_a_player_mut(color) = self.inner.hand(color);
        }
        partial
    }
}

use crate::{Color, BOARD_SIZE, BOARD_SQUARES};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SquareError {
    FileOutOfRange { file: usize },
    RankOutOfRange { rank: usize },
    IndexOutOfRange { index: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Square {
    pub file: usize,
    pub rank: usize,
}

impl Square {
    /// 絶対座標の将棋盤 square を作成する。
    ///
    /// # Errors
    ///
    /// `file` または `rank` が `1..=9` の範囲外なら [`SquareError`] を返す。
    pub const fn new(file: usize, rank: usize) -> Result<Self, SquareError> {
        if file < 1 || file > BOARD_SIZE {
            return Err(SquareError::FileOutOfRange { file });
        }
        if rank < 1 || rank > BOARD_SIZE {
            return Err(SquareError::RankOutOfRange { rank });
        }

        Ok(Self { file, rank })
    }

    #[must_use]
    pub const fn index(self) -> usize {
        (self.rank - 1) * BOARD_SIZE + (self.file - 1)
    }

    /// 0 始まりの盤面 index を square に変換する。
    ///
    /// # Errors
    ///
    /// `index >= 81` の場合は [`SquareError::IndexOutOfRange`] を返す。
    pub const fn from_index(index: usize) -> Result<Self, SquareError> {
        if index >= BOARD_SQUARES {
            return Err(SquareError::IndexOutOfRange { index });
        }

        let file = index % BOARD_SIZE + 1;
        let rank = index / BOARD_SIZE + 1;
        Ok(Self { file, rank })
    }

    #[must_use]
    pub const fn to_current_view(self, side_to_move: Color) -> Self {
        match side_to_move {
            Color::Black => self,
            Color::White => Self {
                file: BOARD_SIZE + 1 - self.file,
                rank: BOARD_SIZE + 1 - self.rank,
            },
        }
    }

    #[must_use]
    pub const fn from_current_view(self, side_to_move: Color) -> Self {
        self.to_current_view(side_to_move)
    }
}

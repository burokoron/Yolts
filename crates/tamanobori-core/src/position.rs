use crate::{Color, HandPiece, Piece, Square, BOARD_SQUARES, HAND_PIECE_ORDER};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Board {
    squares: [Option<Piece>; BOARD_SQUARES],
}

impl Board {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            squares: [None; BOARD_SQUARES],
        }
    }

    #[must_use]
    pub fn get(&self, square: Square) -> Option<Piece> {
        self.squares[square.index()]
    }

    pub fn set(&mut self, square: Square, piece: Option<Piece>) {
        self.squares[square.index()] = piece;
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Hand {
    counts: [[u8; HAND_PIECE_ORDER.len()]; 2],
}

impl Hand {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            counts: [[0; HAND_PIECE_ORDER.len()]; 2],
        }
    }

    #[must_use]
    pub fn count(&self, color: Color, piece: HandPiece) -> u8 {
        self.counts[color.index()][piece.index()]
    }

    pub fn set_count(&mut self, color: Color, piece: HandPiece, count: u8) {
        self.counts[color.index()][piece.index()] = count;
    }
}

impl Default for Hand {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Move {
    Normal {
        from: Square,
        to: Square,
        promote: bool,
    },
    Drop {
        piece: HandPiece,
        to: Square,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Position {
    pub board: Board,
    pub hand: Hand,
    pub side_to_move: Color,
    pub ply: u16,
}

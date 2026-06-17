#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Color {
    Black,
    White,
}

impl Color {
    #[must_use]
    pub const fn opponent(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }

    #[must_use]
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Black => 0,
            Self::White => 1,
        }
    }

    #[must_use]
    pub const fn sfen_token(self) -> char {
        match self {
            Self::Black => 'b',
            Self::White => 'w',
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PieceKind {
    Pawn,
    Lance,
    Knight,
    Silver,
    Gold,
    Bishop,
    Rook,
    King,
    ProPawn,
    ProLance,
    ProKnight,
    ProSilver,
    Horse,
    Dragon,
}

impl PieceKind {
    #[must_use]
    pub const fn promoted(self) -> Option<Self> {
        match self {
            Self::Pawn => Some(Self::ProPawn),
            Self::Lance => Some(Self::ProLance),
            Self::Knight => Some(Self::ProKnight),
            Self::Silver => Some(Self::ProSilver),
            Self::Bishop => Some(Self::Horse),
            Self::Rook => Some(Self::Dragon),
            Self::Gold
            | Self::King
            | Self::ProPawn
            | Self::ProLance
            | Self::ProKnight
            | Self::ProSilver
            | Self::Horse
            | Self::Dragon => None,
        }
    }

    #[must_use]
    pub const fn unpromoted(self) -> Self {
        match self {
            Self::ProPawn => Self::Pawn,
            Self::ProLance => Self::Lance,
            Self::ProKnight => Self::Knight,
            Self::ProSilver => Self::Silver,
            Self::Horse => Self::Bishop,
            Self::Dragon => Self::Rook,
            kind => kind,
        }
    }

    #[must_use]
    pub const fn sfen_char(self) -> char {
        match self {
            Self::Pawn | Self::ProPawn => 'P',
            Self::Lance | Self::ProLance => 'L',
            Self::Knight | Self::ProKnight => 'N',
            Self::Silver | Self::ProSilver => 'S',
            Self::Gold => 'G',
            Self::Bishop | Self::Horse => 'B',
            Self::Rook | Self::Dragon => 'R',
            Self::King => 'K',
        }
    }

    #[must_use]
    pub const fn is_promoted(self) -> bool {
        matches!(
            self,
            Self::ProPawn
                | Self::ProLance
                | Self::ProKnight
                | Self::ProSilver
                | Self::Horse
                | Self::Dragon
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandPiece {
    Pawn,
    Lance,
    Knight,
    Silver,
    Gold,
    Bishop,
    Rook,
}

pub const HAND_PIECE_ORDER: [HandPiece; 7] = [
    HandPiece::Rook,
    HandPiece::Bishop,
    HandPiece::Gold,
    HandPiece::Silver,
    HandPiece::Knight,
    HandPiece::Lance,
    HandPiece::Pawn,
];

impl HandPiece {
    #[must_use]
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Pawn => 0,
            Self::Lance => 1,
            Self::Knight => 2,
            Self::Silver => 3,
            Self::Gold => 4,
            Self::Bishop => 5,
            Self::Rook => 6,
        }
    }

    #[must_use]
    pub const fn piece_kind(self) -> PieceKind {
        match self {
            Self::Pawn => PieceKind::Pawn,
            Self::Lance => PieceKind::Lance,
            Self::Knight => PieceKind::Knight,
            Self::Silver => PieceKind::Silver,
            Self::Gold => PieceKind::Gold,
            Self::Bishop => PieceKind::Bishop,
            Self::Rook => PieceKind::Rook,
        }
    }

    #[must_use]
    pub const fn sfen_char(self) -> char {
        self.piece_kind().sfen_char()
    }
}

impl TryFrom<PieceKind> for HandPiece {
    type Error = HandPieceError;

    fn try_from(kind: PieceKind) -> Result<Self, Self::Error> {
        match kind {
            PieceKind::Pawn => Ok(Self::Pawn),
            PieceKind::Lance => Ok(Self::Lance),
            PieceKind::Knight => Ok(Self::Knight),
            PieceKind::Silver => Ok(Self::Silver),
            PieceKind::Gold => Ok(Self::Gold),
            PieceKind::Bishop => Ok(Self::Bishop),
            PieceKind::Rook => Ok(Self::Rook),
            PieceKind::King
            | PieceKind::ProPawn
            | PieceKind::ProLance
            | PieceKind::ProKnight
            | PieceKind::ProSilver
            | PieceKind::Horse
            | PieceKind::Dragon => Err(HandPieceError::NotDroppable { kind }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandPieceError {
    NotDroppable { kind: PieceKind },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

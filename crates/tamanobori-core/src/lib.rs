pub const FORMAL_NAME_JA: &str = "爆裂玉登太郎";
pub const ROMANIZED_NAME: &str = "Bakuretsu Tamanobori Taro";
pub const ENGINE_ID: &str = "bakuretsu_tamanobori_taro";
pub const SHORT_ID: &str = "tamanobori";
pub const ENGINE_FAMILY_ID: &str = "bakuretsu";
pub const USI_ID_NAME: &str = "BakuretsuTamanoboriTaro";

pub const RULE_VERSION: &str = "rule_v0";
pub const FEATURE_VERSION: &str = "feature_v0";
pub const MOVE_LABEL_VERSION: &str = "move_label_v0";
pub const NETWORK_OUTPUT_VERSION: &str = "network_output_v0";
pub const TARGET_VERSION: &str = "target_v0";
pub const SELFPLAY_LOG_VERSION: &str = "selfplay_log_v0";
pub const DATASET_VERSION: &str = "dataset_v0";
pub const CHECKPOINT_VERSION: &str = "checkpoint_v0";

pub const BOARD_SIZE: usize = 9;
pub const BOARD_SQUARES: usize = BOARD_SIZE * BOARD_SIZE;
pub const MOVE_LABEL_PLANES: usize = 27;
pub const MOVE_LABEL_SIZE: usize = BOARD_SQUARES * MOVE_LABEL_PLANES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VersionSet {
    pub rule: &'static str,
    pub feature: &'static str,
    pub move_label: &'static str,
    pub network_output: &'static str,
    pub target: &'static str,
    pub selfplay_log: &'static str,
    pub dataset: &'static str,
    pub checkpoint: &'static str,
}

pub const VERSION_SET: VersionSet = VersionSet {
    rule: RULE_VERSION,
    feature: FEATURE_VERSION,
    move_label: MOVE_LABEL_VERSION,
    network_output: NETWORK_OUTPUT_VERSION,
    target: TARGET_VERSION,
    selfplay_log: SELFPLAY_LOG_VERSION,
    dataset: DATASET_VERSION,
    checkpoint: CHECKPOINT_VERSION,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutcomeCause {
    WinNormal,
    WinNyugyoku,
    Draw,
    LossNyugyoku,
    LossNormal,
}

pub const OUTCOME_CAUSE_ORDER: [OutcomeCause; 5] = [
    OutcomeCause::WinNormal,
    OutcomeCause::WinNyugyoku,
    OutcomeCause::Draw,
    OutcomeCause::LossNyugyoku,
    OutcomeCause::LossNormal,
];

#[must_use]
pub const fn move_label_size() -> usize {
    MOVE_LABEL_SIZE
}

pub const STARTPOS_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

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

impl Position {
    /// 標準の初期局面を作成する。
    ///
    /// # Panics
    ///
    /// 組み込みの [`STARTPOS_SFEN`] 定数が不正な場合にのみ panic する。
    #[must_use]
    pub fn startpos() -> Self {
        Self::from_sfen(STARTPOS_SFEN).expect("STARTPOS_SFEN must be valid")
    }

    /// 最小構成の SFEN 局面を parse する。
    ///
    /// # Errors
    ///
    /// SFEN の field 数、盤面、手番、持ち駒、ply field が不正な場合は
    /// [`SfenError`] を返す。
    pub fn from_sfen(sfen: &str) -> Result<Self, SfenError> {
        let mut fields = sfen.split_ascii_whitespace();
        let board_field = fields.next().ok_or(SfenError::MissingField("board"))?;
        let side_field = fields.next().ok_or(SfenError::MissingField("side"))?;
        let hand_field = fields.next().ok_or(SfenError::MissingField("hand"))?;
        let ply_field = fields.next().ok_or(SfenError::MissingField("ply"))?;
        if fields.next().is_some() {
            return Err(SfenError::TooManyFields);
        }

        Ok(Self {
            board: parse_board(board_field)?,
            hand: parse_hand(hand_field)?,
            side_to_move: parse_side_to_move(side_field)?,
            ply: parse_ply(ply_field)?,
        })
    }

    #[must_use]
    pub fn to_sfen(&self) -> String {
        format!(
            "{} {} {} {}",
            format_board(&self.board),
            self.side_to_move.sfen_token(),
            format_hand(&self.hand),
            self.ply
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SfenError {
    MissingField(&'static str),
    TooManyFields,
    InvalidBoardRankCount { ranks: usize },
    InvalidBoardFileCount { rank: usize, files: usize },
    InvalidBoardDigit { rank: usize, digit: char },
    DanglingPromotion { rank: usize },
    InvalidPiece { token: char },
    PromotedGoldOrKing { token: char },
    InvalidSideToMove { value: String },
    InvalidHandToken { token: char },
    InvalidHandCount { value: String },
    InvalidHandPiece { token: char },
    DuplicateHandPiece { color: Color, piece: HandPiece },
    InvalidPly { value: String },
}

fn parse_side_to_move(value: &str) -> Result<Color, SfenError> {
    match value {
        "b" => Ok(Color::Black),
        "w" => Ok(Color::White),
        _ => Err(SfenError::InvalidSideToMove {
            value: value.to_owned(),
        }),
    }
}

fn parse_ply(value: &str) -> Result<u16, SfenError> {
    value.parse().map_err(|_| SfenError::InvalidPly {
        value: value.to_owned(),
    })
}

fn parse_board(value: &str) -> Result<Board, SfenError> {
    let ranks: Vec<&str> = value.split('/').collect();
    if ranks.len() != BOARD_SIZE {
        return Err(SfenError::InvalidBoardRankCount { ranks: ranks.len() });
    }

    let mut board = Board::empty();
    for (rank_index, rank_value) in ranks.iter().enumerate() {
        let rank = rank_index + 1;
        let mut file = 1usize;
        let mut chars = rank_value.chars();
        while let Some(token) = chars.next() {
            if token.is_ascii_digit() {
                let empty = token
                    .to_digit(10)
                    .ok_or(SfenError::InvalidBoardDigit { rank, digit: token })?
                    as usize;
                if empty == 0 || empty > BOARD_SIZE {
                    return Err(SfenError::InvalidBoardDigit { rank, digit: token });
                }
                file += empty;
                continue;
            }

            let promoted = token == '+';
            let piece_token = if promoted {
                chars.next().ok_or(SfenError::DanglingPromotion { rank })?
            } else {
                token
            };
            let piece = parse_board_piece(piece_token, promoted)?;
            if file > BOARD_SIZE {
                return Err(SfenError::InvalidBoardFileCount { rank, files: file });
            }
            let file_abs = BOARD_SIZE + 1 - file;
            let square =
                Square::new(file_abs, rank).expect("validated SFEN board square must be in range");
            board.set(square, Some(piece));
            file += 1;
        }

        if file != BOARD_SIZE + 1 {
            return Err(SfenError::InvalidBoardFileCount {
                rank,
                files: file - 1,
            });
        }
    }

    Ok(board)
}

fn parse_board_piece(token: char, promoted: bool) -> Result<Piece, SfenError> {
    let color = if token.is_ascii_uppercase() {
        Color::Black
    } else if token.is_ascii_lowercase() {
        Color::White
    } else {
        return Err(SfenError::InvalidPiece { token });
    };
    let base = token.to_ascii_uppercase();
    let kind = match base {
        'P' => PieceKind::Pawn,
        'L' => PieceKind::Lance,
        'N' => PieceKind::Knight,
        'S' => PieceKind::Silver,
        'G' => PieceKind::Gold,
        'B' => PieceKind::Bishop,
        'R' => PieceKind::Rook,
        'K' => PieceKind::King,
        _ => return Err(SfenError::InvalidPiece { token }),
    };
    let kind = if promoted {
        kind.promoted()
            .ok_or(SfenError::PromotedGoldOrKing { token })?
    } else {
        kind
    };

    Ok(Piece { color, kind })
}

fn parse_hand(value: &str) -> Result<Hand, SfenError> {
    if value == "-" {
        return Ok(Hand::empty());
    }

    let mut hand = Hand::empty();
    let mut count_digits = String::new();
    for token in value.chars() {
        if token.is_ascii_digit() {
            count_digits.push(token);
            continue;
        }

        let count = if count_digits.is_empty() {
            1
        } else {
            count_digits
                .parse::<u8>()
                .map_err(|_| SfenError::InvalidHandCount {
                    value: count_digits.clone(),
                })?
        };
        if count == 0 {
            return Err(SfenError::InvalidHandCount {
                value: std::mem::take(&mut count_digits),
            });
        }
        count_digits.clear();

        let color = if token.is_ascii_uppercase() {
            Color::Black
        } else if token.is_ascii_lowercase() {
            Color::White
        } else {
            return Err(SfenError::InvalidHandToken { token });
        };
        let piece = parse_hand_piece(token)?;
        if hand.count(color, piece) > 0 {
            return Err(SfenError::DuplicateHandPiece { color, piece });
        }
        hand.set_count(color, piece, count);
    }

    if !count_digits.is_empty() {
        return Err(SfenError::InvalidHandCount {
            value: count_digits,
        });
    }

    Ok(hand)
}

fn parse_hand_piece(token: char) -> Result<HandPiece, SfenError> {
    match token.to_ascii_uppercase() {
        'P' => Ok(HandPiece::Pawn),
        'L' => Ok(HandPiece::Lance),
        'N' => Ok(HandPiece::Knight),
        'S' => Ok(HandPiece::Silver),
        'G' => Ok(HandPiece::Gold),
        'B' => Ok(HandPiece::Bishop),
        'R' => Ok(HandPiece::Rook),
        _ => Err(SfenError::InvalidHandPiece { token }),
    }
}

fn format_board(board: &Board) -> String {
    let mut output = String::new();
    for rank in 1..=BOARD_SIZE {
        if rank > 1 {
            output.push('/');
        }

        let mut empty = 0usize;
        for file in (1..=BOARD_SIZE).rev() {
            let square = Square::new(file, rank).expect("formatter square must be in range");
            if let Some(piece) = board.get(square) {
                if empty > 0 {
                    output.push_str(&empty.to_string());
                    empty = 0;
                }
                push_piece_sfen(&mut output, piece);
            } else {
                empty += 1;
            }
        }

        if empty > 0 {
            output.push_str(&empty.to_string());
        }
    }

    output
}

fn push_piece_sfen(output: &mut String, piece: Piece) {
    if piece.kind.is_promoted() {
        output.push('+');
    }

    let token = piece.kind.sfen_char();
    output.push(match piece.color {
        Color::Black => token,
        Color::White => token.to_ascii_lowercase(),
    });
}

fn format_hand(hand: &Hand) -> String {
    let mut output = String::new();
    for color in [Color::Black, Color::White] {
        for piece in HAND_PIECE_ORDER {
            let count = hand.count(color, piece);
            if count == 0 {
                continue;
            }
            if count > 1 {
                output.push_str(&count.to_string());
            }
            let token = piece.sfen_char();
            output.push(match color {
                Color::Black => token,
                Color::White => token.to_ascii_lowercase(),
            });
        }
    }

    if output.is_empty() {
        "-".to_owned()
    } else {
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTRACT: &str = include_str!("../../../contracts/tamanobori_constants.env");
    const COMPLEX_SFEN: &str =
        "4+P1K2/6P+L+P/9/r+L3G2+b/ls3P3/5g3/3kn2br/1+l1+np2p+p/1s6s w P2gs2n10p 258";

    fn contract_value(key: &str) -> &'static str {
        for line in CONTRACT.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let Some((candidate, value)) = line.split_once('=') else {
                continue;
            };
            if candidate == key {
                return value;
            }
        }

        panic!("missing contract key: {key}");
    }

    fn outcome_cause_key(outcome: OutcomeCause) -> &'static str {
        match outcome {
            OutcomeCause::WinNormal => "win_normal",
            OutcomeCause::WinNyugyoku => "win_nyugyoku",
            OutcomeCause::Draw => "draw",
            OutcomeCause::LossNyugyoku => "loss_nyugyoku",
            OutcomeCause::LossNormal => "loss_normal",
        }
    }

    #[test]
    fn square_index_and_current_view_are_fixed() {
        let square = Square::new(2, 8).expect("square is valid");

        assert_eq!(square.index(), 64);
        assert_eq!(Square::from_index(64), Ok(square));
        assert_eq!(square.to_current_view(Color::Black), square);
        assert_eq!(
            square.to_current_view(Color::White),
            Square::new(8, 2).expect("rotated square is valid")
        );
        assert_eq!(
            Square::new(8, 2)
                .expect("current-view square is valid")
                .from_current_view(Color::White),
            square
        );
    }

    #[test]
    fn board_get_and_set_use_absolute_squares() {
        let mut board = Board::empty();
        let square = Square::new(5, 5).expect("square is valid");
        let piece = Piece {
            color: Color::Black,
            kind: PieceKind::King,
        };

        assert_eq!(board.get(square), None);
        board.set(square, Some(piece));
        assert_eq!(board.get(square), Some(piece));
        board.set(square, None);
        assert_eq!(board.get(square), None);
    }

    #[test]
    fn startpos_sfen_round_trips_and_uses_sfen_file_order() {
        let position = Position::from_sfen(STARTPOS_SFEN).expect("startpos SFEN is valid");

        assert_eq!(position.to_sfen(), STARTPOS_SFEN);
        assert_eq!(Position::startpos(), position);
        assert_eq!(position.side_to_move, Color::Black);
        assert_eq!(position.ply, 1);
        assert_eq!(
            position
                .board
                .get(Square::new(8, 2).expect("white rook square is valid")),
            Some(Piece {
                color: Color::White,
                kind: PieceKind::Rook,
            })
        );
        assert_eq!(
            position
                .board
                .get(Square::new(2, 8).expect("black rook square is valid")),
            Some(Piece {
                color: Color::Black,
                kind: PieceKind::Rook,
            })
        );
    }

    #[test]
    fn complex_sfen_round_trips_with_promotions_hands_and_white_to_move() {
        let position = Position::from_sfen(COMPLEX_SFEN).expect("complex SFEN is valid");

        assert_eq!(position.to_sfen(), COMPLEX_SFEN);
        assert_eq!(position.side_to_move, Color::White);
        assert_eq!(position.ply, 258);
        assert_eq!(position.hand.count(Color::Black, HandPiece::Pawn), 1);
        assert_eq!(position.hand.count(Color::White, HandPiece::Gold), 2);
        assert_eq!(position.hand.count(Color::White, HandPiece::Pawn), 10);
        assert_eq!(
            position
                .board
                .get(Square::new(5, 1).expect("promoted pawn square is valid")),
            Some(Piece {
                color: Color::Black,
                kind: PieceKind::ProPawn,
            })
        );
        assert_eq!(
            position
                .board
                .get(Square::new(8, 4).expect("black promoted lance square is valid")),
            Some(Piece {
                color: Color::Black,
                kind: PieceKind::ProLance,
            })
        );
    }

    #[test]
    fn invalid_sfen_fields_are_rejected() {
        assert_eq!(
            Position::from_sfen("9/9/9/9/9/9/9/9 b - 1"),
            Err(SfenError::InvalidBoardRankCount { ranks: 8 })
        );
        assert_eq!(
            Position::from_sfen("8/9/9/9/9/9/9/9/9 b - 1"),
            Err(SfenError::InvalidBoardFileCount { rank: 1, files: 8 })
        );
        assert_eq!(
            Position::from_sfen("9/9/9/9/9/9/9/9/9 x - 1"),
            Err(SfenError::InvalidSideToMove {
                value: "x".to_owned(),
            })
        );
        assert_eq!(
            Position::from_sfen("9/9/9/9/9/9/9/9/9 b K 1"),
            Err(SfenError::InvalidHandPiece { token: 'K' })
        );
        assert_eq!(
            Position::from_sfen("+G8/9/9/9/9/9/9/9/9 b - 1"),
            Err(SfenError::PromotedGoldOrKing { token: 'G' })
        );
        assert_eq!(
            Position::from_sfen("8+/9/9/9/9/9/9/9/9 b - 1"),
            Err(SfenError::DanglingPromotion { rank: 1 })
        );
        assert_eq!(
            Position::from_sfen("9/9/9/9/9/9/9/9/9 b 0P 1"),
            Err(SfenError::InvalidHandCount {
                value: "0".to_owned(),
            })
        );
        assert_eq!(
            Position::from_sfen("9/9/9/9/9/9/9/9/9 b PP 1"),
            Err(SfenError::DuplicateHandPiece {
                color: Color::Black,
                piece: HandPiece::Pawn,
            })
        );
        assert_eq!(
            Position::from_sfen("9/9/9/9/9/9/9/9/9 b 2PP 1"),
            Err(SfenError::DuplicateHandPiece {
                color: Color::Black,
                piece: HandPiece::Pawn,
            })
        );
        assert_eq!(
            Position::from_sfen("9/9/9/9/9/9/9/9/9 b - not-a-ply"),
            Err(SfenError::InvalidPly {
                value: "not-a-ply".to_owned(),
            })
        );
    }

    #[test]
    fn move_label_size_matches_spec() {
        assert_eq!(BOARD_SIZE.to_string(), contract_value("BOARD_SIZE"));
        assert_eq!(BOARD_SQUARES.to_string(), contract_value("BOARD_SQUARES"));
        assert_eq!(
            MOVE_LABEL_PLANES.to_string(),
            contract_value("MOVE_LABEL_PLANES")
        );
        assert_eq!(
            MOVE_LABEL_SIZE.to_string(),
            contract_value("MOVE_LABEL_SIZE")
        );
    }

    #[test]
    fn outcome_order_is_fixed() {
        let order = OUTCOME_CAUSE_ORDER
            .map(outcome_cause_key)
            .as_slice()
            .join(",");

        assert_eq!(order, contract_value("OUTCOME_CAUSE_ORDER"));
        assert_eq!(
            OUTCOME_CAUSE_ORDER,
            [
                OutcomeCause::WinNormal,
                OutcomeCause::WinNyugyoku,
                OutcomeCause::Draw,
                OutcomeCause::LossNyugyoku,
                OutcomeCause::LossNormal,
            ]
        );
    }

    #[test]
    fn constants_match_shared_contract() {
        assert_eq!(FORMAL_NAME_JA, contract_value("FORMAL_NAME_JA"));
        assert_eq!(ROMANIZED_NAME, contract_value("ROMANIZED_NAME"));
        assert_eq!(ENGINE_ID, contract_value("ENGINE_ID"));
        assert_eq!(SHORT_ID, contract_value("SHORT_ID"));
        assert_eq!(ENGINE_FAMILY_ID, contract_value("ENGINE_FAMILY_ID"));
        assert_eq!(USI_ID_NAME, contract_value("USI_ID_NAME"));
        assert_eq!(VERSION_SET.rule, contract_value("RULE_VERSION"));
        assert_eq!(VERSION_SET.feature, contract_value("FEATURE_VERSION"));
        assert_eq!(VERSION_SET.move_label, contract_value("MOVE_LABEL_VERSION"));
        assert_eq!(
            VERSION_SET.network_output,
            contract_value("NETWORK_OUTPUT_VERSION")
        );
        assert_eq!(VERSION_SET.target, contract_value("TARGET_VERSION"));
        assert_eq!(
            VERSION_SET.selfplay_log,
            contract_value("SELFPLAY_LOG_VERSION")
        );
        assert_eq!(VERSION_SET.dataset, contract_value("DATASET_VERSION"));
        assert_eq!(VERSION_SET.checkpoint, contract_value("CHECKPOINT_VERSION"));
    }
}

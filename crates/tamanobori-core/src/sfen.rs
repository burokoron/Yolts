use crate::{
    Board, Color, Hand, HandPiece, Piece, PieceKind, Position, Square, BOARD_SIZE, HAND_PIECE_ORDER,
};

pub const STARTPOS_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

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

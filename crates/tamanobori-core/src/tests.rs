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

fn square(file: usize, rank: usize) -> Square {
    Square::new(file, rank).expect("test square is valid")
}

fn expected_label(plane: MoveLabelPlane, to_square_current_view: Square) -> usize {
    move_label_from_parts(plane.index(), to_square_current_view.index())
        .expect("test label parts are valid")
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
fn black_and_white_forward_moves_share_up_label() {
    let black_label = encode_move_label(
        Move::Normal {
            from: square(7, 7),
            to: square(7, 6),
            promote: false,
        },
        Color::Black,
    )
    .expect("black pawn-like move is representable");
    let white_label = encode_move_label(
        Move::Normal {
            from: square(3, 3),
            to: square(3, 4),
            promote: false,
        },
        Color::White,
    )
    .expect("white pawn-like move is representable");

    assert_eq!(
        black_label,
        expected_label(MoveLabelPlane::Up, square(7, 6))
    );
    assert_eq!(white_label, black_label);
}

#[test]
fn promotion_adds_the_fixed_promote_offset() {
    let plain = encode_move_label(
        Move::Normal {
            from: square(7, 3),
            to: square(7, 2),
            promote: false,
        },
        Color::Black,
    )
    .expect("plain move is representable");
    let promoted = encode_move_label(
        Move::Normal {
            from: square(7, 3),
            to: square(7, 2),
            promote: true,
        },
        Color::Black,
    )
    .expect("promoting move is representable");

    assert_eq!(
        promoted - plain,
        PROMOTE_OFFSET * BOARD_SQUARES,
        "promote planes are offset by 10 full 81-square planes"
    );
    assert_eq!(
        promoted,
        expected_label(MoveLabelPlane::PromoteUp, square(7, 2))
    );
}

#[test]
fn knight_jumps_use_dedicated_current_side_planes() {
    let up2_left = encode_move_label(
        Move::Normal {
            from: square(5, 5),
            to: square(6, 3),
            promote: false,
        },
        Color::Black,
    )
    .expect("knight left jump is representable");
    let up2_right = encode_move_label(
        Move::Normal {
            from: square(5, 5),
            to: square(4, 3),
            promote: false,
        },
        Color::Black,
    )
    .expect("knight right jump is representable");

    assert_eq!(
        up2_left,
        expected_label(MoveLabelPlane::Up2Left, square(6, 3))
    );
    assert_eq!(
        up2_right,
        expected_label(MoveLabelPlane::Up2Right, square(4, 3))
    );
}

#[test]
fn drops_use_fixed_piece_planes() {
    for (piece, plane) in [
        (HandPiece::Pawn, MoveLabelPlane::DropPawn),
        (HandPiece::Lance, MoveLabelPlane::DropLance),
        (HandPiece::Knight, MoveLabelPlane::DropKnight),
        (HandPiece::Silver, MoveLabelPlane::DropSilver),
        (HandPiece::Gold, MoveLabelPlane::DropGold),
        (HandPiece::Bishop, MoveLabelPlane::DropBishop),
        (HandPiece::Rook, MoveLabelPlane::DropRook),
    ] {
        let actual = encode_move_label(
            Move::Drop {
                piece,
                to: square(5, 5),
            },
            Color::Black,
        )
        .expect("drop move is representable");

        assert_eq!(actual, expected_label(plane, square(5, 5)));
    }
}

#[test]
fn white_drop_uses_current_side_destination() {
    let black_label = encode_move_label(
        Move::Drop {
            piece: HandPiece::Bishop,
            to: square(7, 7),
        },
        Color::Black,
    )
    .expect("black drop is representable");
    let white_label = encode_move_label(
        Move::Drop {
            piece: HandPiece::Bishop,
            to: square(3, 3),
        },
        Color::White,
    )
    .expect("white drop is representable");

    assert_eq!(
        black_label,
        expected_label(MoveLabelPlane::DropBishop, square(7, 7))
    );
    assert_eq!(white_label, black_label);
}

#[test]
fn long_range_moves_collapse_to_direction_and_destination() {
    let far_rook_like = encode_move_label(
        Move::Normal {
            from: square(5, 9),
            to: square(5, 1),
            promote: false,
        },
        Color::Black,
    )
    .expect("long file move is representable");
    let near_rook_like = encode_move_label(
        Move::Normal {
            from: square(5, 3),
            to: square(5, 1),
            promote: false,
        },
        Color::Black,
    )
    .expect("short file move is representable");
    let bishop_like = encode_move_label(
        Move::Normal {
            from: square(9, 9),
            to: square(1, 1),
            promote: false,
        },
        Color::Black,
    )
    .expect("long diagonal move is representable");

    assert_eq!(far_rook_like, near_rook_like);
    assert_eq!(
        far_rook_like,
        expected_label(MoveLabelPlane::Up, square(5, 1))
    );
    assert_eq!(
        bishop_like,
        expected_label(MoveLabelPlane::UpRight, square(1, 1))
    );
}

#[test]
fn invalid_move_label_inputs_are_rejected() {
    assert_eq!(
        encode_move_label(
            Move::Normal {
                from: square(5, 5),
                to: square(7, 4),
                promote: false,
            },
            Color::Black,
        ),
        Err(MoveLabelError::UnrepresentableMove {
            from: square(5, 5),
            to: square(7, 4),
        })
    );
    assert_eq!(
        encode_move_label(
            Move::Normal {
                from: square(5, 5),
                to: square(5, 5),
                promote: false,
            },
            Color::Black,
        ),
        Err(MoveLabelError::ZeroLengthMove {
            square: square(5, 5),
        })
    );
    assert_eq!(
        move_label_from_parts(MOVE_LABEL_PLANES, 0),
        Err(MoveLabelError::PlaneOutOfRange {
            plane: MOVE_LABEL_PLANES,
        })
    );
    assert_eq!(
        move_label_from_parts(0, BOARD_SQUARES),
        Err(MoveLabelError::SquareIndexOutOfRange {
            index: BOARD_SQUARES,
        })
    );
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

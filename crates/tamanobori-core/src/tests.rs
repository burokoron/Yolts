use std::collections::HashSet;

use super::*;

fn square(file: u8, rank: u8) -> Square {
    Square::new(file, rank).expect("test square is valid")
}

fn normal(from: (u8, u8), to: (u8, u8)) -> Move {
    Move::Normal {
        from: square(from.0, from.1),
        to: square(to.0, to.1),
        promote: false,
    }
}

fn play_cycle(position: &mut Position, cycle: &[Move], count: usize) {
    for _ in 0..count {
        for &value in cycle {
            assert!(position.legal_moves().any(|candidate| candidate == value));
            position.do_move(value);
        }
    }
}

#[test]
fn constants_match_shared_contract() {
    assert_eq!(MOVE_LABEL_SIZE, 2_187);
    assert_eq!(
        MOVE_LABEL_VERSION,
        "move_label_v1_dlshogi27_current_side_plane_major"
    );
    assert_eq!(RULE_VERSION, "rule_v0");
}

#[test]
fn move_label_parts_use_plane_major_order() {
    let label =
        move_label_from_parts(MoveLabelPlane::DropRook.index(), 80).expect("fixed parts are valid");
    let parts = move_label_parts(label).expect("fixed label is valid");
    assert_eq!(parts.plane, MoveLabelPlane::DropRook);
    assert_eq!(parts.to_square_current_view, square(9, 9));
    assert_eq!(label, MoveLabelPlane::DropRook.index() * 81 + 80);
}

#[test]
fn fixed_planes_cover_knight_promotion_and_drop_contracts() {
    for index in 0..MOVE_LABEL_PLANES {
        assert_eq!(
            MoveLabelPlane::from_index(index)
                .expect("fixed plane index is valid")
                .index(),
            index
        );
    }
    assert_eq!(
        MoveLabelPlane::from_index(MOVE_LABEL_PLANES),
        Err(MoveLabelError::PlaneOutOfRange {
            plane: MOVE_LABEL_PLANES
        })
    );

    let knight = Move::Normal {
        from: square(3, 3),
        to: square(4, 1),
        promote: true,
    };
    let knight_label = encode_move_label(knight, Color::Black).expect("knight move is encodable");
    assert_eq!(
        move_label_parts(knight_label)
            .expect("knight label is valid")
            .plane,
        MoveLabelPlane::PromoteUp2Left
    );

    let drop = Move::Drop {
        piece: Piece::new(PieceKind::Rook, Color::White),
        to: square(5, 5),
    };
    let drop_label = encode_move_label(drop, Color::White).expect("rook drop is encodable");
    assert_eq!(
        move_label_parts(drop_label)
            .expect("drop label is valid")
            .plane,
        MoveLabelPlane::DropRook
    );
}

#[test]
fn current_side_move_labels_are_symmetric() {
    let black = Move::Normal {
        from: square(7, 7),
        to: square(7, 6),
        promote: false,
    };
    let white = Move::Normal {
        from: square(3, 3),
        to: square(3, 4),
        promote: false,
    };
    assert_eq!(
        encode_move_label(black, Color::Black),
        encode_move_label(white, Color::White)
    );
}

#[test]
fn startpos_sfen_and_yasai_state_round_trip() {
    let mut position = Position::from_sfen(STARTPOS_SFEN).expect("startpos is valid");
    assert_eq!(position.to_sfen(), STARTPOS_SFEN);
    assert_eq!(position.legal_moves().len(), 30);

    let before = (position.to_sfen(), position.keys());
    let value = position.legal_moves().next().expect("startpos has moves");
    position.do_move(value);
    position.undo_move(value);
    assert_eq!((position.to_sfen(), position.keys()), before);

    position.do_null_move();
    position.undo_null_move();
    assert_eq!((position.to_sfen(), position.keys()), before);
}

#[test]
fn invalid_sfen_and_invalid_rule_position_are_rejected() {
    assert!(matches!(
        Position::from_sfen("9/9/9/9/9/9/9/9/9 b - 1"),
        Err(PositionError::MissingKing {
            color: Color::Black
        })
    ));
    assert!(matches!(
        Position::from_sfen("not sfen"),
        Err(PositionError::InvalidSfenFieldCount { fields: 2 })
    ));
}

#[test]
fn legal_moves_have_unique_round_trip_labels() {
    for position in [
        Position::startpos(),
        Position::from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2")
            .expect("white startpos is valid"),
    ] {
        let mut labels = HashSet::new();
        for value in position.legal_moves() {
            let label = encode_move_label(value, position.side_to_move())
                .expect("Yasai legal move is encodable");
            assert!(labels.insert(label));
            assert_eq!(decode_move_label(label, &position), Ok(value));
        }
    }
}

#[test]
fn quiet_fourfold_repetition_is_draw() {
    let mut position =
        Position::from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").expect("fixture is valid");
    let cycle = [
        normal((5, 9), (4, 9)),
        normal((5, 1), (4, 1)),
        normal((4, 9), (5, 9)),
        normal((4, 1), (5, 1)),
    ];
    play_cycle(&mut position, &cycle, 3);
    assert_eq!(position.repetition_outcome(), RepetitionOutcome::Draw);
}

#[test]
fn perpetual_checker_loses_from_current_side_perspective() {
    let mut checker_to_move =
        Position::from_sfen("4k4/5R3/9/9/9/9/9/9/K8 b - 1").expect("fixture is valid");
    let loss_cycle = [
        normal((4, 2), (5, 2)),
        normal((5, 1), (4, 1)),
        normal((5, 2), (4, 2)),
        normal((4, 1), (5, 1)),
    ];
    play_cycle(&mut checker_to_move, &loss_cycle, 3);
    assert_eq!(
        checker_to_move.repetition_outcome(),
        RepetitionOutcome::Loss
    );

    let mut checked_side_to_move =
        Position::from_sfen("4k4/4R4/9/9/9/9/9/9/K8 w - 1").expect("fixture is valid");
    let win_cycle = [
        normal((5, 1), (4, 1)),
        normal((5, 2), (4, 2)),
        normal((4, 1), (5, 1)),
        normal((4, 2), (5, 2)),
    ];
    play_cycle(&mut checked_side_to_move, &win_cycle, 3);
    assert_eq!(
        checked_side_to_move.repetition_outcome(),
        RepetitionOutcome::Win
    );
}

#[test]
fn null_move_is_a_repetition_barrier_and_undo_restores_history() {
    let mut position =
        Position::from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").expect("fixture is valid");
    let cycle = [
        normal((5, 9), (4, 9)),
        normal((5, 1), (4, 1)),
        normal((4, 9), (5, 9)),
        normal((4, 1), (5, 1)),
    ];
    play_cycle(&mut position, &cycle, 3);
    assert_eq!(position.repetition_outcome(), RepetitionOutcome::Draw);
    position.do_null_move();
    assert_eq!(position.repetition_outcome(), RepetitionOutcome::None);
    position.undo_null_move();
    assert_eq!(position.repetition_outcome(), RepetitionOutcome::Draw);
}

#[test]
fn csa27_declaration_win_uses_side_specific_thresholds() {
    let position = Position::from_sfen(
        "4+P1K2/6P+L+P/9/r+L3G2+b/ls3P3/5g3/3kn2br/1+l1+np2p+p/1s6s w P2gs2n10p 258",
    )
    .expect("historical declaration fixture is valid");
    let info = position
        .declaration_win(DeclarationRuleConfig::csa27())
        .expect("white has a CSA 27-point declaration win");
    assert_eq!(info.side, Color::White);
    assert!(info.points >= 27);
    assert!(info.pieces_in_enemy_camp >= 10);

    let mut stricter = DeclarationRuleConfig::csa27();
    stricter.point_threshold_white = info.points + 1;
    assert_eq!(position.declaration_win(stricter), None);
    assert_eq!(
        Position::startpos().declaration_win(DeclarationRuleConfig::csa27()),
        None
    );

    let black = Position::from_sfen("4K4/PPPPPPPPP/4G4/9/9/9/9/9/4k4 b 2R2B 1")
        .expect("black declaration fixture is valid");
    let black_info = black
        .declaration_win(DeclarationRuleConfig::csa27())
        .expect("black exceeds the 28-point threshold");
    assert_eq!(black_info.side, Color::Black);
    assert!(black_info.points >= 28);

    let checked = Position::from_sfen("3rK4/PPPPPPPPP/4G4/9/9/9/9/9/4k4 b R2B 1")
        .expect("checked declaration fixture is valid");
    assert_eq!(
        checked.declaration_win(DeclarationRuleConfig::csa27()),
        None
    );
}

#[test]
fn mated_and_mate_in_one_use_yasai_state_and_restore_it() {
    let terminal =
        Position::from_sfen("8k/6G2/7K1/9/9/9/9/9/9 w - 1").expect("checkmate fixture is valid");
    assert!(terminal.is_mated());

    let mut mate_in_one =
        Position::from_sfen("8k/9/6GK1/9/9/9/9/9/9 b R 1").expect("fixture is valid");
    let before = (mate_in_one.to_sfen(), mate_in_one.keys());
    let winning_moves = mate_in_one.mate_in_one_moves();
    assert!(!winning_moves.is_empty());
    assert_eq!((mate_in_one.to_sfen(), mate_in_one.keys()), before);
}

#[test]
fn outcome_order_is_fixed() {
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

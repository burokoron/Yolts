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
            assert!(position
                .as_ref()
                .legal_moves()
                .into_iter()
                .any(|candidate| candidate == value));
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
fn plane_square_index_is_rank_major() {
    let square = square(1, 2);

    assert_eq!(square.array_index(), 1);
    assert_eq!(crate::view::plane_square_index(square), 9);
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
fn piece_feature_contract_has_fixed_plane_order() {
    assert_eq!(PIECE_FEATURE_PLANES, 28);

    for (index, kind) in PieceKind::all().into_iter().enumerate() {
        assert_eq!(piece_feature_plane(PieceFeatureSide::Own, kind), index);
        assert_eq!(
            piece_feature_plane(PieceFeatureSide::Opponent, kind),
            PieceKind::NUM + index
        );
    }
}

#[test]
fn startpos_piece_features_are_one_hot_and_side_symmetric() {
    let black = Position::startpos();
    let white =
        Position::from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2")
            .expect("white startpos is valid");
    let black_features = encode_piece_placement_features(&black);
    let white_features = encode_piece_placement_features(&white);

    assert_eq!(black_features, white_features);
    assert_eq!(
        black_features
            .iter()
            .flatten()
            .filter(|&&occupied| occupied)
            .count(),
        40
    );
    for square_index in 0..BOARD_SQUARES {
        assert!(
            black_features
                .iter()
                .filter(|plane| plane[square_index])
                .count()
                <= 1
        );
    }

    let own_pawn = piece_feature_plane(PieceFeatureSide::Own, PieceKind::Pawn);
    let opponent_pawn = piece_feature_plane(PieceFeatureSide::Opponent, PieceKind::Pawn);
    assert!(black_features[own_pawn][6 * 9 + 6]);
    assert!(black_features[opponent_pawn][2 * 9 + 2]);
    assert!(!black_features[own_pawn][2 * 9 + 2]);
}

#[test]
fn promoted_pieces_use_dedicated_feature_planes() {
    let position = Position::from_sfen("4k4/9/9/5+r3/4+P4/9/9/9/4K4 b - 1")
        .expect("promoted-piece fixture is valid");
    let features = encode_piece_placement_features(&position);
    let center = 4 * 9 + 4;
    let opponent_rook_square = 3 * 9 + 3;

    assert!(features[piece_feature_plane(PieceFeatureSide::Own, PieceKind::ProPawn)][center]);
    assert!(!features[piece_feature_plane(PieceFeatureSide::Own, PieceKind::Pawn)][center]);
    assert!(
        features[piece_feature_plane(PieceFeatureSide::Opponent, PieceKind::ProRook)]
            [opponent_rook_square]
    );
    assert!(
        !features[piece_feature_plane(PieceFeatureSide::Opponent, PieceKind::Rook)]
            [opponent_rook_square]
    );
}

#[test]
fn startpos_sfen_and_yasai_state_round_trip() {
    let mut position = Position::startpos();
    assert_eq!(
        position.to_sfen(),
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    );

    let before = (position.to_sfen(), position.as_ref().keys());
    let value = position
        .as_ref()
        .legal_moves()
        .into_iter()
        .next()
        .expect("startpos has moves");
    position.do_move(value);
    position.undo_move(value);
    assert_eq!((position.to_sfen(), position.as_ref().keys()), before);

    position.do_null_move();
    position.undo_null_move();
    assert_eq!((position.to_sfen(), position.as_ref().keys()), before);
}

#[test]
fn legal_moves_have_unique_round_trip_labels() {
    for position in [
        Position::startpos(),
        Position::from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2")
            .expect("white startpos is valid"),
        Position::from_sfen(
            "4+P1K2/6P+L+P/9/r+L3G2+b/ls3P3/5g3/3kn2br/1+l1+np2p+p/1s6s w P2gs2n10p 258",
        )
        .expect("complex middle-game fixture is valid"),
    ] {
        let moves = position.as_ref().legal_moves();
        let encoded = encode_legal_moves(&position).expect("Yasai legal moves are encodable");
        assert_eq!(moves.len(), encoded.len());

        let mut labels = HashSet::new();
        for (value, label) in moves.into_iter().zip(encoded) {
            assert!(labels.insert(label));
            assert_eq!(decode_move_label(label, &position), Ok(value));
        }
    }
}

#[test]
fn legal_move_mask_matches_encoded_legal_moves() {
    for position in [
        Position::startpos(),
        Position::from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2")
            .expect("white startpos is valid"),
        Position::from_sfen(
            "ln1g1g1nl/1ks3r2/1pp1ppbpp/p2p2p2/9/P1PP2P2/1P2PP1PP/1B1K2R2/LNSG1GSNL b Pp 31",
        )
        .expect("complex middle-game fixture is valid"),
    ] {
        let labels = encode_legal_moves(&position).expect("Yasai legal moves are encodable");
        let mask = make_legal_move_mask(&position).expect("legal mask is constructible");

        assert_eq!(
            mask.iter().filter(|&&is_legal| is_legal).count(),
            labels.len()
        );
        for label in labels {
            assert!(mask[label]);
        }
        assert!(mask.iter().any(|&is_legal| !is_legal));
    }
}

#[test]
fn no_move_position_has_empty_labels_and_mask() {
    let position = Position::from_sfen("8k/6G2/7K1/9/9/9/9/9/9 w - 10").expect("fixture is valid");

    assert_eq!(encode_legal_moves(&position), Ok(Vec::new()));
    assert_eq!(
        make_legal_move_mask(&position),
        Ok([false; MOVE_LABEL_SIZE])
    );
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
    assert_eq!(
        position.rule_terminal_outcome(None),
        Some(RuleTerminalOutcome::RepetitionDraw)
    );
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
    assert_eq!(
        checker_to_move.rule_terminal_outcome(None),
        Some(RuleTerminalOutcome::PerpetualCheckLoss)
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
    assert_eq!(
        checked_side_to_move.rule_terminal_outcome(None),
        Some(RuleTerminalOutcome::PerpetualCheckWin)
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
#[should_panic(expected = "null move must not be played while in check")]
fn null_move_rejects_checked_position() {
    let mut position =
        Position::from_sfen("k3r4/9/9/9/9/9/9/9/4K4 b - 1").expect("checked fixture is valid");
    position.do_null_move();
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

    let exact = Position::from_sfen("4K4/1PPPPPPPP/3BG4/9/9/9/9/9/4k4 b 2R4P 1")
        .expect("exact threshold fixture is valid");
    assert_eq!(
        exact.declaration_win(DeclarationRuleConfig::csa27()),
        Some(DeclarationWinInfo {
            side: Color::Black,
            points: 28,
            pieces_in_enemy_camp: 10,
        })
    );

    let insufficient_points = Position::from_sfen("4K4/1PPPPPPPP/3BG4/9/9/9/9/9/4k4 b 2R2P 1")
        .expect("26-point fixture is valid");
    assert_eq!(
        insufficient_points.declaration_win(DeclarationRuleConfig::csa27()),
        None
    );

    let insufficient_camp_pieces = Position::from_sfen("4K4/1PPPPPPPP/3B5/9/9/9/9/9/4k4 b 2R5P 1")
        .expect("nine-piece fixture is valid");
    assert_eq!(
        insufficient_camp_pieces.declaration_win(DeclarationRuleConfig::csa27()),
        None
    );
}

#[test]
fn mated_and_mate_in_one_use_yasai_state_and_restore_it() {
    let stalemate =
        Position::from_sfen("8k/6G2/7K1/9/9/9/9/9/9 w - 1").expect("stalemate fixture is valid");
    assert!(!stalemate.as_ref().in_check());
    assert!(!stalemate.is_mated());

    let terminal =
        Position::from_sfen("8k/6G2/7GR/9/9/9/9/9/K8 w - 1").expect("checkmate fixture is valid");
    assert!(terminal.as_ref().in_check());
    assert!(terminal.is_mated());

    let mut mate_in_one =
        Position::from_sfen("8k/9/6GK1/9/9/9/9/9/9 b R 1").expect("fixture is valid");
    let before = (mate_in_one.to_sfen(), mate_in_one.as_ref().keys());
    let winning_moves = mate_in_one.mate_in_one_moves();
    assert!(!winning_moves.is_empty());
    assert_eq!((mate_in_one.to_sfen(), mate_in_one.as_ref().keys()), before);
}

#[test]
fn rule_terminal_outcome_treats_all_no_move_positions_as_losses() {
    let stalemate = Position::from_sfen("8k/6G2/7K1/9/9/9/9/9/9 w - 10").expect("fixture is valid");
    assert!(!stalemate.as_ref().in_check());
    assert!(!stalemate.is_mated());
    assert_eq!(
        stalemate.rule_terminal_outcome(Some(10)),
        Some(RuleTerminalOutcome::NoLegalMovesLoss)
    );

    let checkmate =
        Position::from_sfen("8k/6G2/7GR/9/9/9/9/9/K8 w - 10").expect("fixture is valid");
    assert!(checkmate.is_mated());
    assert_eq!(
        checkmate.rule_terminal_outcome(None),
        Some(RuleTerminalOutcome::NoLegalMovesLoss)
    );
}

#[test]
fn rule_terminal_outcome_handles_max_ply_and_non_terminal_positions() {
    let checked = Position::from_sfen("k3r4/9/9/9/9/9/9/9/4K4 b - 9").expect("fixture is valid");
    assert!(checked.as_ref().in_check());
    assert!(!checked.as_ref().legal_moves().is_empty());
    assert_eq!(checked.rule_terminal_outcome(None), None);
    assert_eq!(checked.rule_terminal_outcome(Some(10)), None);
    assert_eq!(
        checked.rule_terminal_outcome(Some(9)),
        Some(RuleTerminalOutcome::MaxPlyDraw)
    );

    let declaration = Position::from_sfen(
        "4+P1K2/6P+L+P/9/r+L3G2+b/ls3P3/5g3/3kn2br/1+l1+np2p+p/1s6s w P2gs2n10p 258",
    )
    .expect("declaration fixture is valid");
    assert!(declaration
        .declaration_win(DeclarationRuleConfig::csa27())
        .is_some());
    assert_eq!(declaration.rule_terminal_outcome(None), None);
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

#[test]
fn hand_feature_contract_and_broadcast() {
    assert_eq!(HAND_FEATURE_PLANES, 56);
    assert_eq!(HAND_FEATURE_PLANES_PER_SIDE, 28);

    // HAND_FEATURE_PLANES_PER_SIDE が HAND_PIECES_WITH_MAX_COUNT の合計値と一致することを検証
    let expected_planes_per_side = crate::feature::HAND_PIECES_WITH_MAX_COUNT
        .iter()
        .map(|&(_, max)| max)
        .sum::<usize>();
    assert_eq!(HAND_FEATURE_PLANES_PER_SIDE, expected_planes_per_side);

    // 1. 空の持ち駒（平手初期局面など）ではすべてのプレーンがfalseになること
    let startpos = Position::startpos();
    let start_features = encode_hand_features(&startpos);
    for plane in &start_features {
        assert!(plane.iter().all(|&val| !val));
    }

    // 各planeが盤面全体へ正しくbroadcastされていることを確認
    let sfen_black = "4k4/9/9/9/9/9/9/9/4K4 b 3P4N5G1B8p1n4s2b3r 1";
    let pos_black = Position::from_sfen(sfen_black).expect("valid sfen");
    let features_black = encode_hand_features(&pos_black);
    for plane in &features_black {
        let first = plane[0];
        assert!(plane.iter().all(|&val| val == first));
    }

    // 2. hand_feature_plane のアクセサ確認
    // 正常系
    assert_eq!(
        hand_feature_plane(PieceFeatureSide::Own, PieceKind::Pawn, 1),
        Some(0)
    );
    assert_eq!(
        hand_feature_plane(PieceFeatureSide::Own, PieceKind::Pawn, 8),
        Some(7)
    );
    assert_eq!(
        hand_feature_plane(PieceFeatureSide::Own, PieceKind::Rook, 2),
        Some(27)
    );
    assert_eq!(
        hand_feature_plane(PieceFeatureSide::Opponent, PieceKind::Pawn, 1),
        Some(28)
    );
    assert_eq!(
        hand_feature_plane(PieceFeatureSide::Opponent, PieceKind::Rook, 2),
        Some(55)
    );

    // 異常系・境界値
    assert_eq!(
        hand_feature_plane(PieceFeatureSide::Own, PieceKind::Pawn, 0),
        None
    );
    assert_eq!(
        hand_feature_plane(PieceFeatureSide::Own, PieceKind::Pawn, 9),
        None
    );
    assert_eq!(
        hand_feature_plane(PieceFeatureSide::Own, PieceKind::King, 1),
        None
    );
    assert_eq!(
        hand_feature_plane(PieceFeatureSide::Own, PieceKind::ProPawn, 1),
        None
    );
}

#[test]
fn hand_feature_threshold_and_clip() {
    // 閾値ちょうど、未満、上限を超える場合の動作検証
    //
    // NOTE: 以下のSFENは「金5枚」「飛3枚」など、ルール上は物理的に存在しない
    // 不正な持ち駒枚数を含んでいますが、本テストは持ち駒特徴エンコーダが仕様上限（クリップ値）
    // を超えた入力に対しても正しくクリップしてエンコードできるかを検証するための
    // 純粋な単体テスト（境界値テスト）として割り切って設定しています。
    let sfen_black = "4k4/9/9/9/9/9/9/9/4K4 b 3P4N5G1B8p1n4s2b3r 1";
    let pos_black = Position::from_sfen(sfen_black).expect("valid sfen");
    let features_black = encode_hand_features(&pos_black);

    // 手番側（先手）の検証
    // 歩 (8 planes, 0..8): 3枚なので、0, 1, 2番目がtrue、3..8がfalse
    for (idx, plane) in features_black[0..8].iter().enumerate() {
        if idx < 3 {
            assert!(plane[0], "Own Pawn >= {} should be true", idx + 1);
        } else {
            assert!(!plane[0], "Own Pawn >= {} should be false", idx + 1);
        }
    }
    // 香 (4 planes, 8..12): 0枚なので、すべてfalse
    for plane in &features_black[8..12] {
        assert!(!plane[0]);
    }
    // 桂 (4 planes, 12..16): 4枚なので、すべてtrue
    for plane in &features_black[12..16] {
        assert!(plane[0]);
    }
    // 銀 (4 planes, 16..20): 0枚なので、すべてfalse
    for plane in &features_black[16..20] {
        assert!(!plane[0]);
    }
    // 金 (4 planes, 20..24): 5枚なので、すべてtrue (上限4でクリップ)
    for plane in &features_black[20..24] {
        assert!(plane[0]);
    }
    // 角 (2 planes, 24..26): 1枚なので、24がtrue、25がfalse
    assert!(features_black[24][0]);
    assert!(!features_black[25][0]);
    // 飛 (2 planes, 26..28): 0枚なので、すべてfalse
    for plane in &features_black[26..28] {
        assert!(!plane[0]);
    }

    // 相手側（後手）の検証
    // 歩 (8 planes, 28..36): 8枚なので、すべてtrue
    for plane in &features_black[28..36] {
        assert!(plane[0]);
    }
    // 香 (4 planes, 36..40): 0枚なので、すべてfalse
    for plane in &features_black[36..40] {
        assert!(!plane[0]);
    }
    // 桂 (4 planes, 40..44): 1枚なので、40がtrue、41..44がfalse
    assert!(features_black[40][0]);
    for plane in &features_black[41..44] {
        assert!(!plane[0]);
    }
    // 銀 (4 planes, 44..48): 4枚なので、すべてtrue
    for plane in &features_black[44..48] {
        assert!(plane[0]);
    }
    // 金 (4 planes, 48..52): 0枚なので、すべてfalse
    for plane in &features_black[48..52] {
        assert!(!plane[0]);
    }
    // 角 (2 planes, 52..54): 2枚なので、すべてtrue
    for plane in &features_black[52..54] {
        assert!(plane[0]);
    }
    // 飛 (2 planes, 54..56): 3枚なので、すべてtrue (上限2でクリップ)
    for plane in &features_black[54..56] {
        assert!(plane[0]);
    }
}

#[test]
fn hand_feature_symmetry() {
    // 先後と手番を反転した同等局面で同じ特徴になるか確認（対称性テスト）
    // 手番側（後手）: 歩3, 香0, 桂4, 銀0, 金5, 角1, 飛0 (SFEN: 3p4n5g1b)
    // 相手側（先手）: 歩8, 香0, 桂1, 銀4, 金0, 角2, 飛3 (SFEN: 8P1N4S2B3R)
    let sfen_black = "4k4/9/9/9/9/9/9/9/4K4 b 3P4N5G1B8p1n4s2b3r 1";
    let pos_black = Position::from_sfen(sfen_black).expect("valid sfen");
    let features_black = encode_hand_features(&pos_black);

    let sfen_white = "4k4/9/9/9/9/9/9/9/4K4 w 8P1N4S2B3R3p4n5g1b 1";
    let pos_white = Position::from_sfen(sfen_white).expect("valid sfen");
    let features_white = encode_hand_features(&pos_white);

    assert_eq!(features_black, features_white);
}

#[test]
fn hand_feature_plane_matches_encoder_iteration_order() {
    let mut expected = 0usize;
    for &(kind, max) in crate::feature::HAND_PIECES_WITH_MAX_COUNT {
        for threshold in 1..=max {
            assert_eq!(
                hand_feature_plane(PieceFeatureSide::Own, kind, threshold),
                Some(expected),
            );
            assert_eq!(
                hand_feature_plane(PieceFeatureSide::Opponent, kind, threshold),
                Some(expected + HAND_FEATURE_PLANES_PER_SIDE),
            );
            expected += 1;
        }
    }
    assert_eq!(expected, HAND_FEATURE_PLANES_PER_SIDE);
}

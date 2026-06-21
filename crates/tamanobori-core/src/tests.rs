use std::collections::HashSet;

use crate::position::rule_terminal_outcome_from_state;

use super::*;

fn square(file: u8, rank: u8) -> Square {
    Square::new(file, rank).expect("test square is valid")
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
fn rule_terminal_outcome_from_state_prioritizes_no_legal_moves() {
    assert_eq!(
        rule_terminal_outcome_from_state(false, RepetitionOutcome::Draw, None, 1),
        Some(RuleTerminalOutcome::NoLegalMovesLoss)
    );
    assert_eq!(
        rule_terminal_outcome_from_state(false, RepetitionOutcome::Win, Some(1), 1),
        Some(RuleTerminalOutcome::NoLegalMovesLoss)
    );
}

#[test]
fn rule_terminal_outcome_from_state_maps_repetition_outcomes() {
    assert_eq!(
        rule_terminal_outcome_from_state(true, RepetitionOutcome::Draw, None, 1),
        Some(RuleTerminalOutcome::RepetitionDraw)
    );
    assert_eq!(
        rule_terminal_outcome_from_state(true, RepetitionOutcome::Win, None, 1),
        Some(RuleTerminalOutcome::PerpetualCheckWin)
    );
    assert_eq!(
        rule_terminal_outcome_from_state(true, RepetitionOutcome::Loss, None, 1),
        Some(RuleTerminalOutcome::PerpetualCheckLoss)
    );
}

#[test]
fn rule_terminal_outcome_from_state_ignores_superior_and_inferior() {
    assert_eq!(
        rule_terminal_outcome_from_state(true, RepetitionOutcome::Superior, None, 1),
        None
    );
    assert_eq!(
        rule_terminal_outcome_from_state(true, RepetitionOutcome::Inferior, None, 1),
        None
    );
    assert_eq!(
        rule_terminal_outcome_from_state(true, RepetitionOutcome::None, None, 1),
        None
    );
}

#[test]
fn rule_terminal_outcome_from_state_handles_max_ply() {
    assert_eq!(
        rule_terminal_outcome_from_state(true, RepetitionOutcome::None, Some(9), 9),
        Some(RuleTerminalOutcome::MaxPlyDraw)
    );
    assert_eq!(
        rule_terminal_outcome_from_state(true, RepetitionOutcome::None, Some(10), 9),
        None
    );
}

#[test]
#[should_panic(expected = "null move must not be played while in check")]
fn null_move_rejects_checked_position() {
    let mut position =
        Position::from_sfen("k3r4/9/9/9/9/9/9/9/4K4 b - 1").expect("checked fixture is valid");
    position.do_null_move();
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

#[test]
fn attack_feature_contract_has_fixed_plane_order() {
    assert_eq!(ATTACK_FEATURE_PLANES, 34);

    for (index, kind) in PieceKind::all().into_iter().enumerate() {
        assert_eq!(attack_feature_plane(PieceFeatureSide::Own, kind), index);
        assert_eq!(
            attack_feature_plane(PieceFeatureSide::Opponent, kind),
            PieceKind::NUM + index
        );
    }

    assert_eq!(
        attack_count_feature_plane(PieceFeatureSide::Own, 1),
        Some(PieceKind::NUM * 2)
    );
    assert_eq!(
        attack_count_feature_plane(PieceFeatureSide::Own, 2),
        Some(PieceKind::NUM * 2 + 1)
    );
    assert_eq!(
        attack_count_feature_plane(PieceFeatureSide::Own, 3),
        Some(PieceKind::NUM * 2 + 2)
    );
    assert_eq!(
        attack_count_feature_plane(PieceFeatureSide::Opponent, 1),
        Some(PieceKind::NUM * 2 + 3)
    );
    assert_eq!(
        attack_count_feature_plane(PieceFeatureSide::Opponent, 2),
        Some(PieceKind::NUM * 2 + 4)
    );
    assert_eq!(
        attack_count_feature_plane(PieceFeatureSide::Opponent, 3),
        Some(PieceKind::NUM * 2 + 5)
    );

    assert_eq!(attack_count_feature_plane(PieceFeatureSide::Own, 0), None);
    assert_eq!(attack_count_feature_plane(PieceFeatureSide::Own, 4), None);
    assert_eq!(
        attack_count_feature_plane(PieceFeatureSide::Opponent, 0),
        None
    );
    assert_eq!(
        attack_count_feature_plane(PieceFeatureSide::Opponent, 4),
        None
    );
}

#[test]
fn attack_feature_piece_kind_planes_cover_stepping_and_promoted_pieces() {
    let startpos = Position::startpos();
    let start_features = encode_attack_features(&startpos);
    let own_pawn = attack_feature_plane(PieceFeatureSide::Own, PieceKind::Pawn);
    let own_knight = attack_feature_plane(PieceFeatureSide::Own, PieceKind::Knight);
    let opponent_pawn = attack_feature_plane(PieceFeatureSide::Opponent, PieceKind::Pawn);

    assert!(start_features[own_pawn][5 * 9 + 6]);
    assert!(start_features[own_knight][6 * 9]);
    assert!(start_features[own_knight][6 * 9 + 2]);
    assert!(start_features[opponent_pawn][3 * 9 + 2]);

    let promoted = Position::from_sfen("4+B4/9/9/9/9/9/9/9/9 b - 1")
        .expect("promoted-bishop fixture is valid");
    let promoted_features = encode_attack_features(&promoted);
    let own_pro_bishop = attack_feature_plane(PieceFeatureSide::Own, PieceKind::ProBishop);

    assert!(promoted_features[own_pro_bishop][3]);
    assert!(promoted_features[own_pro_bishop][5]);
    assert!(promoted_features[own_pro_bishop][13]);
}

#[test]
fn attack_feature_sliding_includes_blocker_not_beyond() {
    // 先手飛 9八。3八まで利き、遮蔽なし。
    let open = Position::from_sfen("9/9/9/9/9/9/9/R8/9 b - 1").expect("open-rook fixture is valid");
    // 先手飛 1八、後手歩 2八で遮蔽。2八は利きに含み、3八は含まない。
    let blocked =
        Position::from_sfen("9/9/9/9/9/9/9/7pR/9 b - 1").expect("blocked-rook fixture is valid");
    let open_features = encode_attack_features(&open);
    let blocked_features = encode_attack_features(&blocked);
    let own_rook = attack_feature_plane(PieceFeatureSide::Own, PieceKind::Rook);
    let blocker_square_index = 7 * 9 + 1;
    let beyond_square_index = 7 * 9 + 2;

    assert!(open_features[own_rook][beyond_square_index]);
    assert!(blocked_features[own_rook][blocker_square_index]);
    assert!(!blocked_features[own_rook][beyond_square_index]);
}

#[test]
fn attack_feature_count_threshold_and_clip() {
    let center = 4 * 9 + 4;
    let own_ge1 = attack_count_feature_plane(PieceFeatureSide::Own, 1).expect("threshold 1");
    let own_ge2 = attack_count_feature_plane(PieceFeatureSide::Own, 2).expect("threshold 2");
    let own_ge3 = attack_count_feature_plane(PieceFeatureSide::Own, 3).expect("threshold 3");
    let opp_ge1 = attack_count_feature_plane(PieceFeatureSide::Opponent, 1).expect("threshold 1");
    let opp_ge2 = attack_count_feature_plane(PieceFeatureSide::Opponent, 2).expect("threshold 2");
    let opp_ge3 = attack_count_feature_plane(PieceFeatureSide::Opponent, 3).expect("threshold 3");

    // 先手飛2枚が 5五へ利く。>=1, >=2 のみ立ち、>=3 は立たない。
    let two_rooks =
        Position::from_sfen("9/9/9/9/R7R/9/9/9/9 b - 1").expect("two-rook fixture is valid");
    let two_features = encode_attack_features(&two_rooks);
    let own_rook = attack_feature_plane(PieceFeatureSide::Own, PieceKind::Rook);

    assert!(two_features[own_rook][center]);
    assert!(two_features[own_ge1][center]);
    assert!(two_features[own_ge2][center]);
    assert!(!two_features[own_ge3][center]);

    // 先手角4・飛2が 5五へ6手利き。>=3 へ clip され、それ以上でも >=3 面のみ立つ。
    let four_plus =
        Position::from_sfen("B3R3B/9/9/9/9/9/9/9/B3R3B b - 1").expect("four-plus fixture is valid");
    let four_plus_features = encode_attack_features(&four_plus);

    assert!(
        four_plus
            .as_ref()
            .attack_map()
            .attack_count(Color::Black, square(5, 5))
            >= 4
    );
    assert!(four_plus_features[own_ge1][center]);
    assert!(four_plus_features[own_ge2][center]);
    assert!(four_plus_features[own_ge3][center]);

    // 後手角4・飛2が 5五へ6手利き (先手番)。相手側利き数面も同様に clip される。
    let opponent_four_plus = Position::from_sfen("b3r3b/9/9/9/9/9/9/9/b3r3b b - 1")
        .expect("opponent-four-plus fixture is valid");
    let opponent_features = encode_attack_features(&opponent_four_plus);

    assert!(
        opponent_four_plus
            .as_ref()
            .attack_map()
            .attack_count(Color::White, square(5, 5))
            >= 4
    );
    assert!(opponent_features[opp_ge1][center]);
    assert!(opponent_features[opp_ge2][center]);
    assert!(opponent_features[opp_ge3][center]);
}

#[test]
fn attack_feature_symmetry() {
    // 非対称配置: 先手飛 9八のみ / 後手番では盤面反転した後手飛 1二のみ。
    let black =
        Position::from_sfen("9/9/9/9/9/9/9/R8/9 b - 1").expect("black-rook fixture is valid");
    let white =
        Position::from_sfen("9/8r/9/9/9/9/9/9/9 w - 1").expect("white-rook fixture is valid");

    assert_eq!(
        encode_attack_features(&black),
        encode_attack_features(&white)
    );
}

#[test]
fn check_feature_contract_has_single_broadcast_plane() {
    assert_eq!(CHECK_FEATURE_PLANES, 1);
}

#[test]
fn check_feature_startpos_is_all_false() {
    let startpos = Position::startpos();
    let features = encode_check_features(&startpos);

    assert!(!startpos.as_ref().in_check());
    assert!(features[0].iter().all(|&val| !val));
}

#[test]
fn check_feature_current_side_in_check_broadcasts_true() {
    // 先手番: 後手飛が5筋で先手玉へ縦利き。
    let black_in_check = Position::from_sfen("9/9/9/9/4K4/9/9/9/4r4 b - 1")
        .expect("black-in-check fixture is valid");
    let black_features = encode_check_features(&black_in_check);

    assert!(black_in_check.as_ref().in_check());
    assert!(black_features[0].iter().all(|&val| val));

    // 後手番: 先手飛が5筋で後手玉へ縦利き。
    let white_in_check = Position::from_sfen("4k4/9/9/9/9/9/9/9/4R4 w - 1")
        .expect("white-in-check fixture is valid");
    let white_features = encode_check_features(&white_in_check);

    assert!(white_in_check.as_ref().in_check());
    assert!(white_features[0].iter().all(|&val| val));
}

#[test]
fn check_feature_ignores_opponent_check() {
    // 先手番だが王手されているのは後手玉のみ。
    let opponent_only = Position::from_sfen("4k4/9/9/9/9/9/9/9/4R4 b - 1")
        .expect("opponent-only-check fixture is valid");
    let features = encode_check_features(&opponent_only);

    assert!(!opponent_only.as_ref().in_check());
    assert!(features[0].iter().all(|&val| !val));
}

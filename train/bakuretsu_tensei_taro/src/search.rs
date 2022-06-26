use shogi_core::{Color, Move, Piece, PieceKind, Square};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use yasai::Position;

use crate::Eval;

pub struct HashTableValue {
    depth: u32,
    upper: i32,
    lower: i32,
    best_move: Option<Move>,
}

pub struct HashTable {
    pub pos: HashMap<u64, HashTableValue>,
}

pub struct MoveOrdering {
    pub piece_to_history: Vec<Vec<Vec<i64>>>,
}

pub struct NegaAlpha {
    pub my_turn: Color,
    pub start_time: Instant,
    pub max_time: i32,
    pub num_searched: u64,
    pub max_depth: u32,
    pub max_board_number: u16,
    pub best_move_pv: Option<Move>,
    pub eval: Eval,
    pub hash_table: HashTable,
    pub move_ordering: MoveOrdering,
}

impl NegaAlpha {
    fn evaluate(v: &NegaAlpha, pos: &mut Position) -> i32 {
        // 入玉宣言の確認
        if is_nyugyoku_win(pos) {
            return 30000 - pos.ply() as i32;
        }

        // 通常の評価
        let mut value = 0;
        for sq in Square::all() {
            let pc = pos.piece_at(sq);
            if let Some(ref pc) = pc {
                value += v.eval.pieces_in_board[pc.color().array_index()][sq.array_index()]
                    [pc.piece_kind().array_index() + 1];
            } else {
                value += v.eval.pieces_in_board[0][sq.array_index()][0];
            }
        }
        for color in Color::all() {
            let hand = pos.hand(color);
            for piece_type in PieceKind::all() {
                if piece_type == PieceKind::King {
                    break;
                }
                value += v.eval.pieces_in_hand[color.array_index()][piece_type.array_index()]
                    [hand.Hand_count(piece_type) as usize];
            }
        }

        if pos.side_to_move() == Color::Black {
            value
        } else {
            -value
        }
    }

    pub fn search(
        v: &mut NegaAlpha,
        pos: &mut Position,
        position_history: &mut HashSet<u64>,
        depth: u32,
        mut alpha: i32,
        mut beta: i32,
    ) -> i32 {
        // 探索局面数
        v.num_searched += 1;

        // 最大手数の計算
        if v.max_board_number < pos.ply() {
            v.max_board_number = pos.ply();
        }

        // 時間制限なら
        let end = v.start_time.elapsed();
        let elapsed_time = end.as_secs() as i32 * 1000 + end.subsec_nanos() as i32 / 1_000_000;
        if elapsed_time >= v.max_time {
            return alpha;
        }

        // 同一局面の確認
        if position_history.contains(&pos.key()) {
            if pos.in_check() {
                return 30000 - pos.ply() as i32;
            } else {
                return 0;
            }
        }

        // 置換表の確認
        let mut best_move = None;
        let hash_table_value = v.hash_table.pos.get(&pos.key());
        if let Some(hash_table_value) = hash_table_value {
            if depth <= hash_table_value.depth {
                let lower = hash_table_value.lower;
                let upper = hash_table_value.upper;
                if lower >= beta {
                    return lower;
                }
                if upper <= alpha || upper == lower {
                    return upper;
                }
                alpha = alpha.max(lower);
                beta = beta.min(upper);
            }
            best_move = hash_table_value.best_move;
        }

        // 探索深さ制限なら
        if depth == 0 {
            return NegaAlpha::evaluate(v, pos);
        }

        // Mate Distance Pruning
        let mating_value = -30000 + pos.ply() as i32;
        if mating_value > alpha {
            alpha = mating_value;
            if beta <= mating_value {
                return mating_value;
            }
        }
        let mating_value = 30000 - pos.ply() as i32;
        if mating_value < beta {
            beta = mating_value;
            if alpha >= mating_value {
                return mating_value;
            }
        }

        // Futility Pruning
        let value = NegaAlpha::evaluate(v, pos);
        if value <= alpha - 400 * depth as i32 {
            return value;
        }

        // 全合法手検索
        let legal_moves = pos.legal_moves();
        // 合法手なしなら
        if legal_moves.is_empty() {
            if pos.side_to_move() == v.my_turn {
                return -30000 + pos.ply() as i32;
            } else {
                return 30000 - pos.ply() as i32;
            }
        }

        // ムーブオーダリング
        let mut best_value = alpha;
        let mut move_list: Vec<(Move, i64)> = Vec::new();
        // ムーブオーダリング用の重み計算
        for m in legal_moves {
            let mut value = 0;
            // 置換表にある手
            if let Some(best_move) = best_move {
                if best_move == m {
                    value += 100000;
                }
            }
            // Piece To History
            let turn = pos.side_to_move().array_index();
            let piece = match m {
                Move::Normal {
                    from,
                    to: _,
                    promote: _,
                } => pos.piece_at(from).unwrap(),
                Move::Drop { piece, to: _ } => piece,
            };
            let to = m.to().array_index();
            value += v.move_ordering.piece_to_history[turn][piece.piece_kind().array_index()][to];
            move_list.push((m, value));
        }
        move_list.sort_by(|&i, &j| (-i.1).cmp(&(-j.1)));

        // 全合法手展開
        position_history.insert(pos.key());
        for m in move_list {
            pos.do_move(m.0);
            let value = -NegaAlpha::search(v, pos, position_history, depth - 1, -beta, -best_value);
            if best_value < value {
                best_value = value;
                best_move = Some(m.0);
                if depth == v.max_depth {
                    v.best_move_pv = Some(m.0);
                }
            }
            pos.undo_move(m.0);
            if best_value >= beta {
                // Piece To History
                let turn = pos.side_to_move().array_index();
                let piece = match m.0 {
                    Move::Normal {
                        from,
                        to: _,
                        promote: _,
                    } => pos.piece_at(from).unwrap(),
                    Move::Drop { piece, to: _ } => piece,
                };
                let to = m.0.to().array_index();
                v.move_ordering.piece_to_history[turn][piece.piece_kind().array_index()][to] +=
                    depth as i64 * depth as i64;
                break;
            }
        }
        position_history.remove(&pos.key());

        // 置換表へ登録
        let hash_table_value = v.hash_table.pos.entry(pos.key()).or_insert(HashTableValue {
            depth: 0,
            upper: 30000,
            lower: -30000,
            best_move: None,
        });
        if depth > hash_table_value.depth {
            if best_value <= alpha {
                hash_table_value.upper = best_value;
            } else if best_value >= beta {
                hash_table_value.lower = best_value;
            } else {
                hash_table_value.upper = best_value;
                hash_table_value.lower = best_value;
            }
            hash_table_value.depth = depth;
            hash_table_value.best_move = best_move;
        }

        best_value
    }
}

pub fn is_nyugyoku_win(pos: &Position) -> bool {
    // 入玉宣言の確認
    if pos.side_to_move() == Color::Black && !pos.in_check() {
        let sq = pos.king_position(Color::Black);
        if let Some(ref sq) = sq {
            if sq.rank() <= 3 {
                let mut value = 0;
                let mut count = 0;
                for sq in Square::all() {
                    if sq.rank() <= 3 {
                        let pc = pos.piece_at(sq);
                        if let Some(pc) = pc {
                            match pc {
                                Piece::B_P
                                | Piece::B_L
                                | Piece::B_N
                                | Piece::B_S
                                | Piece::B_G
                                | Piece::B_PP
                                | Piece::B_PL
                                | Piece::B_PN
                                | Piece::B_PS => {
                                    value += 1;
                                    count += 1;
                                }
                                Piece::B_B | Piece::B_R | Piece::B_PB | Piece::B_PR => {
                                    value += 5;
                                    count += 1;
                                }
                                _ => (),
                            }
                        }
                    }
                }
                if count >= 10 {
                    let hand = pos.hand(Color::Black);
                    for piece_type in PieceKind::all() {
                        if piece_type == PieceKind::King {
                            break;
                        }
                        match piece_type {
                            PieceKind::Pawn
                            | PieceKind::Lance
                            | PieceKind::Knight
                            | PieceKind::Silver
                            | PieceKind::Gold => value += hand.Hand_count(piece_type),
                            PieceKind::Bishop | PieceKind::Rook => {
                                value += hand.Hand_count(piece_type) * 5
                            }
                            _ => (),
                        }
                    }
                }
                if value >= 28 {
                    return true;
                }
            }
        }
    } else if pos.side_to_move() == Color::White && !pos.in_check() {
        let sq = pos.king_position(Color::White);
        if let Some(ref sq) = sq {
            if sq.rank() >= 7 {
                let mut value = 0;
                let mut count = 0;
                for sq in Square::all() {
                    if sq.rank() >= 7 {
                        let pc = pos.piece_at(sq);
                        if let Some(pc) = pc {
                            match pc {
                                Piece::W_P
                                | Piece::W_L
                                | Piece::W_N
                                | Piece::W_S
                                | Piece::W_G
                                | Piece::W_PP
                                | Piece::W_PL
                                | Piece::W_PN
                                | Piece::W_PS => {
                                    value += 1;
                                    count += 1;
                                }
                                Piece::W_B | Piece::W_R | Piece::W_PB | Piece::W_PR => {
                                    value += 5;
                                    count += 1;
                                }
                                _ => (),
                            }
                        }
                    }
                }
                if count >= 10 {
                    let hand = pos.hand(Color::White);
                    for piece_type in PieceKind::all() {
                        if piece_type == PieceKind::King {
                            break;
                        }
                        match piece_type {
                            PieceKind::Pawn
                            | PieceKind::Lance
                            | PieceKind::Knight
                            | PieceKind::Silver
                            | PieceKind::Gold => value += hand.Hand_count(piece_type),
                            PieceKind::Bishop | PieceKind::Rook => {
                                value += hand.Hand_count(piece_type) * 5
                            }
                            _ => (),
                        }
                    }
                }
                if value >= 27 {
                    return true;
                }
            }
        }
    }

    false
}

pub fn move_to_sfen(m: Move) -> String {
    match m {
        Move::Normal { from, to, promote } => {
            let from = format!(
                "{}{}",
                (b'1' + from.array_index() as u8 / 9) as char,
                (b'a' + from.array_index() as u8 % 9) as char
            );
            let to = format!(
                "{}{}",
                (b'1' + to.array_index() as u8 / 9) as char,
                (b'a' + to.array_index() as u8 % 9) as char
            );
            let promote = {
                if promote {
                    "+"
                } else {
                    ""
                }
            };
            format!("{from}{to}{promote}")
        }
        Move::Drop { to, piece } => {
            let to = format!(
                "{}{}",
                (b'1' + to.array_index() as u8 / 9) as char,
                (b'a' + to.array_index() as u8 % 9) as char
            );
            let piece = {
                match piece {
                    Piece::B_P | Piece::W_P => "P*".to_string(),
                    Piece::B_L | Piece::W_L => "L*".to_string(),
                    Piece::B_N | Piece::W_N => "N*".to_string(),
                    Piece::B_S | Piece::W_S => "S*".to_string(),
                    Piece::B_G | Piece::W_G => "G*".to_string(),
                    Piece::B_B | Piece::W_B => "B*".to_string(),
                    Piece::B_R | Piece::W_R => "R*".to_string(),
                    _ => unreachable!(),
                }
            };
            format!("{piece}{to}")
        }
    }
}

pub fn pv_to_sfen(
    v: &mut NegaAlpha,
    pos: &mut Position,
    position_history: &mut HashSet<u64>,
) -> String {
    let mut pv = "".to_string();
    let mut moves = Vec::new();

    loop {
        if position_history.contains(&pos.key()) {
            break;
        }
        let hash_table = v.hash_table.pos.get(&pos.key());
        if let Some(hash_table_value) = hash_table {
            let best_move = hash_table_value.best_move;
            if let Some(best_move) = best_move {
                let legal_moves = pos.legal_moves();
                let mut is_legal = false;
                for m in legal_moves {
                    if m == best_move {
                        is_legal = true;
                        break;
                    }
                }
                if !is_legal {
                    break;
                }
                pv += &move_to_sfen(best_move);
                pv += " ";
                position_history.insert(pos.key());
                pos.do_move(best_move);
                moves.push(best_move);
            } else {
                break;
            }
        } else {
            break;
        }
    }

    loop {
        let m = moves.pop();
        if let Some(m) = m {
            pos.undo_move(m);
            position_history.remove(&pos.key());
        } else {
            break;
        }
    }

    pv
}

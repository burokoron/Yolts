use std::collections::{HashMap, HashSet};
use std::time::Instant;
use yasai::{Color, Move, MoveType, Piece, PieceType, Position, Square};

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
    pub max_board_number: u32,
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
        for sq in Square::ALL {
            let pc = pos.piece_on(sq);
            if let Some(ref pc) = pc {
                value += v.eval.pieces_in_board[pc.color().index()][sq.index()]
                    [pc.piece_type().index() + 1];
            } else {
                value += v.eval.pieces_in_board[0][sq.index()][0];
            }
        }
        for color in Color::ALL {
            let hand = pos.hand(color);
            for piece_type in PieceType::ALL_HAND {
                value += v.eval.pieces_in_hand[color.index()][piece_type.index()]
                    [hand.num(piece_type) as usize];
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
        // 置換表にある手は最初に調べる
        if let Some(best_move) = best_move {
            if pos.is_legal_move(best_move) {
                move_list.push((best_move, 100000));
            }
        }
        // ムーブオーダリング用の重み計算
        for m in legal_moves {
            // Piece To History
            let turn = pos.side_to_move().index();
            let piece = m.piece().piece_type().index();
            let to = m.to().index();
            let value = v.move_ordering.piece_to_history[turn][piece][to];
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
                let turn = pos.side_to_move().index();
                let piece = m.0.piece().piece_type().index();
                let to = m.0.to().index();
                v.move_ordering.piece_to_history[turn][piece][to] += depth as i64 * depth as i64;
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
        let sq = pos.king(Color::Black);
        if let Some(ref sq) = sq {
            if sq.rank().is_opponent_field(Color::Black) {
                let mut value = 0;
                let mut count = 0;
                for sq in Square::ALL {
                    if sq.rank().is_opponent_field(Color::Black) {
                        let pc = pos.piece_on(sq);
                        if let Some(pc) = pc {
                            match pc {
                                Piece::BFU
                                | Piece::BGI
                                | Piece::BKE
                                | Piece::BKI
                                | Piece::BKY
                                | Piece::BNG
                                | Piece::BNK
                                | Piece::BNY
                                | Piece::BTO => {
                                    value += 1;
                                    count += 1;
                                }
                                Piece::BHI | Piece::BKA | Piece::BRY | Piece::BUM => {
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
                    for piece_type in PieceType::ALL_HAND {
                        match piece_type {
                            PieceType::FU
                            | PieceType::GI
                            | PieceType::KE
                            | PieceType::KI
                            | PieceType::KY => value += hand.num(piece_type),
                            PieceType::HI | PieceType::KA => value += hand.num(piece_type) * 5,
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
        let sq = pos.king(Color::White);
        if let Some(ref sq) = sq {
            if sq.rank().is_opponent_field(Color::White) {
                let mut value = 0;
                let mut count = 0;
                for sq in Square::ALL {
                    if sq.rank().is_opponent_field(Color::White) {
                        let pc = pos.piece_on(sq);
                        if let Some(pc) = pc {
                            match pc {
                                Piece::WFU
                                | Piece::WGI
                                | Piece::WKE
                                | Piece::WKI
                                | Piece::WKY
                                | Piece::WNG
                                | Piece::WNK
                                | Piece::WNY
                                | Piece::WTO => {
                                    value += 1;
                                    count += 1;
                                }
                                Piece::WHI | Piece::WKA | Piece::WRY | Piece::WUM => {
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
                    for piece_type in PieceType::ALL_HAND {
                        match piece_type {
                            PieceType::FU
                            | PieceType::GI
                            | PieceType::KE
                            | PieceType::KI
                            | PieceType::KY => value += hand.num(piece_type),
                            PieceType::HI | PieceType::KA => value += hand.num(piece_type) * 5,
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
    match m.move_type() {
        MoveType::Normal {
            from,
            to,
            is_promotion,
            piece: _,
        } => {
            let from = format!(
                "{}{}",
                (b'1' + from.index() as u8 / 9) as char,
                (b'a' + from.index() as u8 % 9) as char
            );
            let to = format!(
                "{}{}",
                (b'1' + to.index() as u8 / 9) as char,
                (b'a' + to.index() as u8 % 9) as char
            );
            let is_promotion = {
                if is_promotion {
                    "+"
                } else {
                    ""
                }
            };
            format!("{from}{to}{is_promotion}")
        }
        MoveType::Drop { to, piece } => {
            let to = format!(
                "{}{}",
                (b'1' + to.index() as u8 / 9) as char,
                (b'a' + to.index() as u8 % 9) as char
            );
            let piece = {
                match piece {
                    Piece::BFU | Piece::WFU => "P*".to_string(),
                    Piece::BKY | Piece::WKY => "L*".to_string(),
                    Piece::BKE | Piece::WKE => "N*".to_string(),
                    Piece::BGI | Piece::WGI => "S*".to_string(),
                    Piece::BKI | Piece::WKI => "G*".to_string(),
                    Piece::BKA | Piece::WKA => "B*".to_string(),
                    Piece::BHI | Piece::WHI => "R*".to_string(),
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
                if !pos.is_legal_move(best_move) {
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

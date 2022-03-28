use std::collections::HashMap;
use std::time::Instant;
use shogi::{Color, Move, PieceType, Position, Square};
use shogi::piece::Piece;

use crate::Eval;


pub struct HashTableValue {
    depth: f32,
    upper: i32,
    lower: i32,
}

pub struct HashTable {
    pub pos: HashMap<String, HashTableValue>,
}

pub struct MoveOrdering {
    pub pos: HashMap<String, Vec<(Move, i32)>>,
}


pub struct NegaAlpha {
    pub start_time: Instant,
    pub max_time: i32,
    pub num_searched: u64,
    pub max_depth: f32,
    pub best_move_pv: String,
    pub eval: Eval,
    pub hash_table: HashTable,
    pub from_to_move_ordering: MoveOrdering,
}

impl NegaAlpha {
    pub fn sfen(pos: &mut Position) -> String {
        let mut sfen = String::new();
        for sq in Square::iter() {
            let pc = pos.piece_at(sq);
            if let Some(ref pc) = *pc {
                sfen += &pc.color.index().to_string();
                sfen += &pc.piece_type.index().to_string();
            } else {
                sfen += "0";
            }
        }
        for piece_type in [PieceType::Rook, PieceType::Bishop, PieceType::Gold, PieceType::Silver, PieceType::Knight, PieceType::Lance, PieceType::Pawn] {
            for color in Color::iter() {
                sfen += &color.index().to_string();
                sfen += &pos.hand(Piece { piece_type, color }).to_string();
            }
        }
        sfen += &pos.side_to_move().index().to_string();

        return sfen;
    }

    fn evaluate(v: &NegaAlpha, pos: &mut Position) -> i32 {
        let mut value = 0;
        for sq in Square::iter() {
            let pc = pos.piece_at(sq);
            if let Some(ref pc) = *pc {
                value += v.eval.pieces_in_board[pc.color.index()][sq.index()][pc.piece_type.index()+1];
            } else {
                value += v.eval.pieces_in_board[0][sq.index()][0];
            }
        }
        for piece_type in [PieceType::Rook, PieceType::Bishop, PieceType::Gold, PieceType::Silver, PieceType::Knight, PieceType::Lance, PieceType::Pawn] {
            for color in Color::iter() {
                value += v.eval.pieces_in_hand[color.index()][piece_type.index()][pos.hand(Piece { piece_type, color }) as usize];
            }
        }

        if pos.side_to_move() == Color::Black {
            return value;
        } else {
            return -value;
        }
    }

    pub fn search(v: &mut NegaAlpha, pos: &mut Position, mut depth: f32, mut alpha: i32, mut beta: i32) -> i32 {
        // 探索局面数
        v.num_searched += 1;

        // 時間制限なら
        let end = v.start_time.elapsed();
        let elapsed_time = end.as_secs() as i32 * 1000 + end.subsec_nanos() as i32 / 1_000_000;
        if elapsed_time >= v.max_time {
            return alpha;
        }

        // 置換表の確認
        let sfen = NegaAlpha::sfen(pos);
        let hash_table_value = v.hash_table.pos.get(&sfen);
        if let Some(hash_table_value) = hash_table_value {
            if depth <= hash_table_value.depth {
                let lower = hash_table_value.lower;
                let upper = hash_table_value.upper;
                if lower >= beta {
                    return lower;
                }
                if upper <= alpha || upper == lower {
                    return  upper;
                }
                alpha = alpha.max(lower);
                beta = beta.max(upper);
            }
        }

        // 探索深さ制限なら
        if depth <= 0. {
            return NegaAlpha::evaluate(v, pos);
        }

        let mut best_value = alpha;
        let mut moves: Vec<Move> = Vec::new();
        // 現局面における合法手の評価値が高かった順に調べる
        let mut move_list: Vec<Move> = Vec::new();
        let from_to_move_ordering = v.from_to_move_ordering.pos.get(&sfen);
        if let Some(from_to_move_ordering) = from_to_move_ordering {
            let mut idx = (0..from_to_move_ordering.len()).collect::<Vec<_>>();
            idx.sort_unstable_by(|&i, &j| (-from_to_move_ordering[i].1).cmp(&(-from_to_move_ordering[j].1)));
            for i in idx {
                move_list.push(from_to_move_ordering[i].0);
            }
        }
        let mut from_to_move_ordering = Vec::new();
        for m in move_list {
            if pos.make_move(m).is_ok() {
                let value = - NegaAlpha::search(v, pos, depth - 1., -beta, -best_value);
                if best_value < value {
                    // ムーブオーダリング登録
                    from_to_move_ordering.push((m, value));
                    best_value = value;
                    if depth == v.max_depth {
                        v.best_move_pv = m.to_string();
                    }
                }
                if pos.unmake_move().is_ok() {
                    moves.push(m);
                }
                if best_value >= beta {
                    break;
                }
            }
        }

        // 全合法手展開
        'cut: for sq in Square::iter() {
            let pc = pos.piece_at(sq);
            if let Some(ref pc) = *pc {
                if pos.side_to_move() == pc.color {
                    let mut bb = pos.move_candidates(sq, *pc);
                    while bb.is_any() {
                        let to = bb.pop();
                        let m = Move::Normal { from: sq, to: to, promote: false };
                        if pos.make_move(m).is_ok() {
                            let value = - NegaAlpha::search(v, pos, depth - 1., -beta, -best_value);
                            if best_value < value {
                                // ムーブオーダリング登録
                                from_to_move_ordering.push((m, value));
                                best_value = value;
                                if depth == v.max_depth {
                                    v.best_move_pv = m.to_string();
                                }
                            }
                            if pos.unmake_move().is_ok() {
                                moves.push(m);
                            }
                            if best_value >= beta {
                                break 'cut;
                            }
                        }
                        let m = Move::Normal { from: sq, to: to, promote: true };
                        if pos.make_move(m).is_ok() {
                            let value = - NegaAlpha::search(v, pos, depth - 1., -beta, -best_value);
                            if best_value < value {
                                // ムーブオーダリング登録
                                from_to_move_ordering.push((m, value));
                                best_value = value;
                                if depth == v.max_depth {
                                    v.best_move_pv = m.to_string();
                                }
                            }
                            if pos.unmake_move().is_ok() {
                                moves.push(m);
                            }
                            if best_value >= beta {
                                break 'cut;
                            }
                        }
                    }
                }
            } else {
                for piece_type in PieceType::iter() {
                    let color = pos.side_to_move();
                    if pos.hand(Piece { piece_type, color }) > 0 {
                        let m = Move::Drop { to: sq, piece_type: piece_type };
                        if pos.make_move(m).is_ok() {
                            let value = - NegaAlpha::search(v, pos, depth - 1., -beta, -best_value);
                            if best_value < value {
                                // ムーブオーダリング登録
                                from_to_move_ordering.push((m, value));
                                best_value = value;
                                if depth == v.max_depth {
                                    v.best_move_pv = m.to_string();
                                }
                            }
                            if pos.unmake_move().is_ok() {
                                moves.push(m);
                            }
                            if best_value >= beta {
                                break 'cut;
                            }
                        }
                    }
                }
            }
        }
        v.from_to_move_ordering.pos.insert(sfen, from_to_move_ordering);

        // 置換表へ登録
        let sfen = NegaAlpha::sfen(pos);
        let hash_table_value = v.hash_table.pos.entry(sfen).or_insert(HashTableValue { depth: -1., upper: 30000, lower: -30000 });
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
        }

        if moves.len() != 0 {
            return best_value;
        } else {
            return -30000 + pos.ply() as i32;
        }
    }
}

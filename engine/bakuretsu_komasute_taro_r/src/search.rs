use std::collections::{HashMap, HashSet};
use std::time::Instant;
use yasai::{Color, Move, PieceType, Position, Square};

use crate::Eval;


pub struct HashTableValue {
    depth: f32,
    upper: i32,
    lower: i32,
}

pub struct HashTable {
    pub pos: HashMap<u64, HashTableValue>,
}

pub struct MoveOrdering {
    pub pos: HashMap<u64, Vec<(Move, i32)>>,
}

pub struct BrotherMoveOrdering {
    pub pos: HashMap<u16, HashMap<String, i32>>,
}


pub struct NegaAlpha {
    pub start_time: Instant,
    pub max_time: i32,
    pub num_searched: u64,
    pub max_depth: f32,
    pub max_board_number: u32,
    pub best_move_pv: Option<Move>,
    pub eval: Eval,
    pub hash_table: HashTable,
    pub from_to_move_ordering: MoveOrdering,
    pub brother_from_to_move_ordering: BrotherMoveOrdering,
}

impl NegaAlpha {
    /*
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
    */

    fn evaluate(v: &NegaAlpha, pos: &mut Position) -> i32 {
        let mut value = 0;
        for sq in Square::ALL {
            let pc = pos.piece_on(sq);
            if let Some(ref pc) = pc {
                value += v.eval.pieces_in_board[pc.color().index()][sq.index()][pc.piece_type().index()+1];
            } else {
                value += v.eval.pieces_in_board[0][sq.index()][0];
            }
        }
        for color in Color::ALL {
            let hand = pos.hand(color);
            for piece_type in PieceType::ALL_HAND {
                value += v.eval.pieces_in_hand[color.index()][piece_type.index()][hand.num(piece_type) as usize];
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

        // 置換表の確認
        let hash_table_value = v.hash_table.pos.get(&pos.key());
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
                beta = beta.min(upper);
            }
        }

        // 探索深さ制限なら
        if depth <= 0. {
            return NegaAlpha::evaluate(v, pos);
        }

        // 王手ならちょっと延長
        /*
        if (pos.in_check(Color::Black) || pos.in_check(Color::White)) && v.max_depth - 1. > depth {
            depth += 1.
        }
        */

        // Mate Distance Pruning
        let mating_value = -30000 + pos.ply() as i32;
        if mating_value > alpha {
            alpha = mating_value;
            if beta <= mating_value {
                return mating_value
            }
        }
        let mating_value = 30000 - pos.ply() as i32;
        if mating_value < beta {
            beta = mating_value;
            if alpha >= mating_value {
                return mating_value
            }
        }

        // Futility Pruning
        let value = NegaAlpha::evaluate(v, pos);
        if value <= alpha - 400 * depth as i32 {
            return value
        }

        let mut best_value = alpha;
        let mut moves: Vec<Move> = Vec::new();
        let mut ignore_moves: HashSet<String> = HashSet::new();
        // 現局面における合法手の評価値が高かった順に調べる
        let mut move_list: Vec<Move> = Vec::new();
        let from_to_move_ordering = v.from_to_move_ordering.pos.get(&pos.key());
        if let Some(from_to_move_ordering) = from_to_move_ordering {
            let mut idx = (0..from_to_move_ordering.len()).collect::<Vec<_>>();
            idx.sort_unstable_by(|&i, &j| (-from_to_move_ordering[i].1).cmp(&(-from_to_move_ordering[j].1)));
            for i in idx {
                move_list.push(from_to_move_ordering[i].0);
                ignore_moves.insert(from_to_move_ordering[i].0.to_string());
            }
        }/* else {
            // 兄弟局面の着手の評価値が高かった順に調べる
            let brother_from_to_move_ordering = v.brother_from_to_move_ordering.pos.get(&pos.ply());
            if let Some(brother_from_to_move_ordering) = brother_from_to_move_ordering {
                let mut brother = Vec::new();
                for (k, v) in brother_from_to_move_ordering {
                    brother.push((k, *v));
                }
                let mut idx = (0..brother_from_to_move_ordering.len()).collect::<Vec<_>>();
                idx.sort_unstable_by(|&i, &j| (-brother[i].1).cmp(&(-brother[j].1)));
                for i in idx {
                    move_list.push(Move::from_sfen(&brother[i].0).unwrap());
                    ignore_moves.insert(brother[i].0.to_string());
                }
            }
        }*/
        let mut from_to_move_ordering = Vec::new();
        //let mut brother_from_to_move_ordering: HashMap<String, i32> = HashMap::new();

        for m in move_list {
            pos.do_move(m);
            let value = - NegaAlpha::search(v, pos, depth - 1., -beta, -best_value);
            // ムーブオーダリング登録
            /*
            let brother = brother_from_to_move_ordering.entry(m.to_string()).or_insert(value);
            *brother = (*brother as f32 * 0.9 + value as f32 * 0.1) as i32;
            */
            if best_value < value {
                // ムーブオーダリング登録
                from_to_move_ordering.push((m, value));
                best_value = value;
                if depth == v.max_depth {
                    v.best_move_pv = Some(m);
                }
            }
            pos.undo_move(m);
            if best_value >= beta {
                break;
            }
        }

        // 全合法手展開
        let move_list = pos.legal_moves();
        if move_list.len() == 0 {
            return -30000 + pos.ply() as i32;
        }
        for m in move_list {
            pos.do_move(m);
            let value = - NegaAlpha::search(v, pos, depth - 1., -beta, -best_value);
            // ムーブオーダリング登録
            /*
            let brother = brother_from_to_move_ordering.entry(m.to_string()).or_insert(value);
            *brother = (*brother as f32 * 0.9 + value as f32 * 0.1) as i32;
            */
            if best_value < value {
                // ムーブオーダリング登録
                from_to_move_ordering.push((m, value));
                best_value = value;
                if depth == v.max_depth {
                    v.best_move_pv = Some(m);
                }
            }
            pos.undo_move(m);
            if best_value >= beta {
                break;
            }
        }
        v.from_to_move_ordering.pos.insert(pos.key(), from_to_move_ordering);
        //v.brother_from_to_move_ordering.pos.insert(pos.ply(), brother_from_to_move_ordering);

        // 置換表へ登録
        let hash_table_value = v.hash_table.pos.entry(pos.key()).or_insert(HashTableValue { depth: -1., upper: 30000, lower: -30000 });
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
        
        return best_value;
    }
}

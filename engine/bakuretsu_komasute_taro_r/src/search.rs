use shogi::{Color, Move, PieceType, Position, Square};
use shogi::piece::Piece;

pub struct NegaAlpha {
    pub max_depth: f32,
    pub num_searched: u64,
    pub best_move_pv: String,
}

impl NegaAlpha {
    pub fn search(v: &mut NegaAlpha, pos: &mut Position, depth: f32) -> i32 {
        // 探索局面数
        v.num_searched += 1;

        // 探索深さ制限なら
        if depth >= v.max_depth {
            return 0;
        }

        let mut best_value = -30000;
        let mut moves: Vec<Move> = Vec::new();
        for sq in Square::iter() {
            let pc = pos.piece_at(sq);
            if let Some(ref pc) = *pc {
                if pos.side_to_move() == pc.color {
                    let mut bb = pos.move_candidates(sq, *pc);
                    while bb.is_any() {
                        let to = bb.pop();
                        let m = Move::Normal { from: sq, to: to, promote: false };
                        if pos.make_move(m).is_ok() {
                            let value = - NegaAlpha::search(v, pos, depth + 1.);
                            if best_value < value {
                                best_value = value;
                                if depth == 0. {
                                    v.best_move_pv = m.to_string();
                                }
                            }
                            moves.push(m);
                            pos.unmake_move().unwrap();
                        }
                        let m = Move::Normal { from: sq, to: to, promote: true };
                        if pos.make_move(m).is_ok() {
                            let value = - NegaAlpha::search(v, pos, depth + 1.);
                            if best_value < value {
                                best_value = value;
                                if depth == 0. {
                                    v.best_move_pv = m.to_string();
                                }
                            }
                            moves.push(m);
                            pos.unmake_move().unwrap();
                        }
                    }
                }
            } else {
                for piece_type in PieceType::iter() {
                    let color = pos.side_to_move();
                    if pos.hand(Piece { piece_type, color }) > 0 {
                        let m = Move::Drop { to: sq, piece_type: piece_type };
                        if pos.make_move(m).is_ok() {
                            let value = - NegaAlpha::search(v, pos, depth + 1.);
                            if best_value < value {
                                best_value = value;
                                if depth == 0. {
                                    v.best_move_pv = m.to_string();
                                }
                            }
                            moves.push(m);
                            pos.unmake_move().unwrap();
                        }
                    }
                }
            }
        }

        if moves.len() != 0 {
            return best_value;
        } else {
            if depth == 0. {
                v.best_move_pv = "resign".to_string();
            }
            if pos.side_to_move() == Color::Black {
                return -30000 + pos.ply() as i32;
            } else {
                return 30000 - pos.ply() as i32;
            }
        }
    }
}

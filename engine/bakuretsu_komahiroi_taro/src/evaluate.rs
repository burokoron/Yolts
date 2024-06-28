use serde::{Deserialize, Serialize};
use shogi_core::{Color, Move, Piece, PieceKind, Square};
use yasai::Position;

pub const VALUE_SCALE: f32 = 1.;

#[derive(Deserialize, Serialize)]
pub struct EvalJson {
    pub params: Vec<f32>,
}
pub struct Evaluate {
    model: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
}

impl Evaluate {
    pub fn new(path: &str) -> Self {
        //! 局面評価のためのインスタンスを作成する
        //! ファイルから評価関数パラメータを読み込む
        //!
        //! - Arguments
        //!   - path: &str
        //!     - 評価関数ファイルのパス
        //! - Returns
        //!   - Self: Evaluate
        //!     - 局面評価のためのインスタンス

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .expect("Not found evaluate file.");

        let reader = std::io::BufReader::new(file);
        let eval_json: EvalJson =
            serde_json::from_reader(reader).expect("Cannot read evaluate file.");

        let mut model = vec![vec![vec![vec![vec![0f32; 31]; 95]; 31]; 95]; 2];
        let mut idx = 0;
        for i in model.iter_mut() {
            for j in i.iter_mut() {
                for k in j.iter_mut() {
                    for m in k.iter_mut() {
                        for n in m.iter_mut() {
                            *n = eval_json.params[idx];
                            idx += 1;
                        }
                    }
                }
            }
        }

        Evaluate { model }
    }

    pub fn inference_diff(&self, pos: &Position, mv: Option<Move>, value: Option<f32>) -> f32 {
        //! 局面を差分評価する
        //!
        //! - Arguments
        //!   - pos: &Position
        //!     - 評価したい局面
        //!   - mv: Move
        //!     - posからの指し手
        //!   - mut value: f32
        //!     - posの評価値
        //! - Returns
        //!   - value: i32
        //!     - 評価値

        // 差分計算元がない場合
        if mv.is_none() || value.is_none() {
            let mut value = 0.;

            let mut sqs_pcs: Vec<(Square, Piece)> = Vec::new();
            for sq in Square::all() {
                let pc = pos.piece_at(sq);
                if let Some(pc) = pc {
                    sqs_pcs.push((sq, pc));
                }
            }
            for (sq1, pc1) in sqs_pcs.iter() {
                for (sq2, pc2) in sqs_pcs.iter() {
                    if sq1.array_index() > sq2.array_index() {
                        continue;
                    }
                    value += self.model[0][sq1.array_index()][pc1.as_u8() as usize]
                        [sq2.array_index()][pc2.as_u8() as usize];
                }
            }

            let mut idx = 81;
            for color in Color::all() {
                let hand = pos.hand(color);
                for piece_type in PieceKind::all() {
                    if piece_type == PieceKind::King {
                        break;
                    }
                    let count = hand.Hand_count(piece_type) as usize;
                    if count != 0 {
                        value += self.model[0][0][0][idx][count];
                    }
                    idx += 1;
                }
            }

            return value;
        }

        let mv = if let Some(mv) = mv {
            mv
        } else {
            panic!("mv == None")
        };
        let mut value = if let Some(value) = value {
            value
        } else {
            panic!("value == None")
        };
        // 盤面の駒を動かす場合
        if let Move::Normal { from, to, promote } = mv {
            let from_piece = if let Some(from_piece) = pos.piece_at(from) {
                from_piece
            } else {
                panic!("Can not move the piece.");
            };
            let from_to_piece = if promote {
                if let Some(from_to_piece) = from_piece.promote() {
                    from_to_piece
                } else {
                    panic!("Can not promote the piece.");
                }
            } else {
                from_piece
            };
            // 盤面の差分計算
            for sq in Square::all() {
                if let Some(pc) = pos.piece_at(sq) {
                    // 移動元
                    if sq.array_index() <= from.array_index() {
                        value -= self.model[0][sq.array_index()][pc.as_u8() as usize]
                            [from.array_index()][from_piece.as_u8() as usize];
                    } else {
                        value -= self.model[0][from.array_index()][from_piece.as_u8() as usize]
                            [sq.array_index()][pc.as_u8() as usize];
                    }
                    // 移動先
                    if sq.array_index() <= to.array_index() {
                        value += self.model[0][sq.array_index()][pc.as_u8() as usize]
                            [to.array_index()][from_to_piece.as_u8() as usize];
                    } else {
                        value += self.model[0][to.array_index()][from_to_piece.as_u8() as usize]
                            [sq.array_index()][pc.as_u8() as usize];
                    }
                }
            }
            // 移動元×移動先の2駒関係が過剰に足されているので引く
            if from.array_index() <= to.array_index() {
                value -= self.model[0][from.array_index()][from_piece.as_u8() as usize]
                    [to.array_index()][from_to_piece.as_u8() as usize];
            } else {
                value -= self.model[0][to.array_index()][from_to_piece.as_u8() as usize]
                    [from.array_index()][from_piece.as_u8() as usize];
            }
            // 移動先の1駒関係が足せていないのでここで足す
            value += self.model[0][to.array_index()][from_to_piece.as_u8() as usize]
                [to.array_index()][from_to_piece.as_u8() as usize];

            // 駒取りの場合は追加
            if let Some(to_piece) = pos.piece_at(to) {
                //盤面の駒取り
                for sq in Square::all() {
                    if let Some(pc) = pos.piece_at(sq) {
                        // 移動先
                        if sq.array_index() <= to.array_index() {
                            value -= self.model[0][sq.array_index()][pc.as_u8() as usize]
                                [to.array_index()][to_piece.as_u8() as usize];
                        } else {
                            value -= self.model[0][to.array_index()][to_piece.as_u8() as usize]
                                [sq.array_index()][pc.as_u8() as usize];
                        }
                    }
                }
                // 移動元×移動先の2駒関係が過剰に引かれているので足す
                if from.array_index() <= to.array_index() {
                    value += self.model[0][from.array_index()][from_piece.as_u8() as usize]
                        [to.array_index()][to_piece.as_u8() as usize];
                } else {
                    value += self.model[0][to.array_index()][to_piece.as_u8() as usize]
                        [from.array_index()][from_piece.as_u8() as usize];
                }
                // 持ち駒
                let turn = pos.side_to_move();
                let idx = match turn {
                    Color::Black => 81,
                    Color::White => 88,
                };
                let piece_kind = to_piece.piece_kind();
                let piece_kind = if let Some(piece_kind) = piece_kind.unpromote() {
                    piece_kind
                } else {
                    piece_kind
                };
                let count = pos.hand(turn).count(piece_kind);
                if let Some(count) = count {
                    value +=
                        self.model[0][0][0][idx + piece_kind.array_index()][count as usize + 1];
                    if count > 0 {
                        value -=
                            self.model[0][0][0][idx + piece_kind.array_index()][count as usize];
                    }
                } else {
                    panic!("Can not capture piece.");
                }
            }

            return value;
        }

        // 駒打ちの場合
        if let Move::Drop { piece, to } = mv {
            // 持ち駒の差分評価
            let turn = pos.side_to_move();
            let idx = match turn {
                Color::Black => 81,
                Color::White => 88,
            };
            let piece_kind = piece.piece_kind();
            let count = pos.hand(turn).count(piece_kind);
            if let Some(count) = count {
                value -= self.model[0][0][0][idx + piece_kind.array_index()][count as usize];
                if count > 1 {
                    value +=
                        self.model[0][0][0][idx + piece_kind.array_index()][count as usize - 1];
                }
            } else {
                panic!("A piece not in the hand was used.");
            }
            // 盤面の差分計算
            for sq in Square::all() {
                if let Some(pc) = pos.piece_at(sq) {
                    if sq.array_index() <= to.array_index() {
                        value += self.model[0][sq.array_index()][pc.as_u8() as usize]
                            [to.array_index()][piece.as_u8() as usize];
                    } else {
                        value += self.model[0][to.array_index()][piece.as_u8() as usize]
                            [sq.array_index()][pc.as_u8() as usize];
                    }
                }
            }
            // 移動先の1駒関係が足せていないのでここで足す
            value += self.model[0][to.array_index()][piece.as_u8() as usize][to.array_index()]
                [piece.as_u8() as usize];

            return value;
        }

        unreachable!("Can not perform differential inference.");
    }
}

#[cfg(test)]
mod tests {
    use crate::evaluate::{EvalJson, Evaluate};
    use std::io::Write;
    use yasai::Position;

    #[test]
    fn load_inference() {
        let path = "test/load_inference.json";

        let eval = EvalJson {
            params: vec![0.; 17346050],
        };
        let mut file = std::fs::File::create(path).unwrap();
        let value = serde_json::to_string(&eval).unwrap();
        file.write_all(value.as_bytes()).unwrap();

        let pos = Position::default();

        let eval = Evaluate::new(path);
        let value = eval.inference_diff(&pos, None, None);

        assert!(value == 0.0);
    }
}

use serde::{Deserialize, Serialize};
use shogi_core::{Color, Move, PieceKind, Square};
use yasai::Position;

pub const VALUE_SCALE: f32 = 2499.;

#[derive(Deserialize, Serialize)]
pub struct EvalJson {
    pub embedding: Vec<Vec<f32>>,
    pub conv: Vec<Vec<Vec<f32>>>,
    pub dense: Vec<Vec<f32>>,
}

pub struct Evaluate {
    pub token: Vec<Vec<Vec<Vec<Vec<usize>>>>>,
    pub embedding: Vec<Vec<f32>>,
    pub conv_idx: [[usize; 81]; 81],
    pub conv: Vec<Vec<Vec<f32>>>,
    pub dense: Vec<Vec<f32>>,
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

        let mut token = vec![vec![vec![vec![vec![0usize; 31]; 95]; 31]; 95]; 2];
        let mut idx = 0;
        for i in token.iter_mut() {
            for j in i.iter_mut() {
                for k in j.iter_mut() {
                    for m in k.iter_mut() {
                        for n in m.iter_mut() {
                            *n = idx;
                            idx += 1;
                        }
                    }
                }
            }
        }
        let mut conv_idx = [[0usize; 81]; 81];
        idx = 0;
        for sq1 in Square::all() {
            for sq2 in Square::all() {
                if sq1.array_index() > sq2.array_index() {
                    continue;
                }
                conv_idx[sq1.array_index()][sq2.array_index()] = idx;
                idx += 1;
            }
        }

        Evaluate {
            token,
            embedding: eval_json.embedding,
            conv_idx,
            conv: eval_json.conv,
            dense: eval_json.dense,
        }
    }

    pub fn inference_diff(
        &self,
        pos: &Position,
        mv: Option<Move>,
        out_conv: Option<(f32, [f32; 2])>,
    ) -> (f32, [f32; 2]) {
        //! 局面を差分評価する
        //!
        //! - Arguments
        //!   - pos: &Position
        //!     - 評価したい局面
        //!   - mv: Move
        //!     - posからの指し手
        //!   - mut out_conv: (f32, [f32; 2])
        //!     - posの評価値と中間出力
        //! - Returns
        //!   - value: (f32, [f32; 2])
        //!     - (評価値, 2層目出力)

        // 差分計算元がない場合
        if mv.is_none() || out_conv.is_none() {
            // 埋め込み層 PP
            let mut out_embedding = [[0f32; 2]; 3335];
            let hidden_size = 2;
            let mut idx = 0;
            // 盤面
            for sq1 in Square::all() {
                for sq2 in Square::all() {
                    if sq1.array_index() > sq2.array_index() {
                        continue;
                    }
                    let pc1 = pos.piece_at(sq1);
                    if let Some(pc1) = pc1 {
                        let pc2 = pos.piece_at(sq2);
                        if let Some(pc2) = pc2 {
                            let token = self.token[0][sq1.array_index()][pc1.as_u8() as usize]
                                [sq2.array_index()][pc2.as_u8() as usize];
                            out_embedding[idx][0] = self.embedding[token][0];
                            out_embedding[idx][1] = self.embedding[token][1];
                        }
                    }
                    idx += 1;
                }
            }
            // 持ち駒
            let mut hand_idx = 81;
            for color in Color::all() {
                let hand = pos.hand(color);
                for piece_type in PieceKind::all() {
                    if piece_type == PieceKind::King {
                        break;
                    }
                    let count = hand.Hand_count(piece_type) as usize;
                    if count != 0 {
                        let token = self.token[0][0][0][hand_idx][count];
                        out_embedding[idx][0] = self.embedding[token][0];
                        out_embedding[idx][1] = self.embedding[token][1];
                    }
                    idx += 1;
                    hand_idx += 1;
                }
            }

            // 畳み込み層
            let mut out_conv = [0f32; 2];
            for (i, out_embed) in out_embedding.iter().enumerate() {
                for j in 0..hidden_size {
                    out_conv[j] += out_embed[j] * self.conv[j][0][i];
                }
            }

            // 活性化関数_1
            let mut out_activation_1 = [0f32; 2];
            for i in 0..hidden_size {
                if out_conv[i] <= -3.0 {
                    out_activation_1[i] = 0.0;
                } else if out_conv[i] >= 3.0 {
                    out_activation_1[i] = out_conv[i];
                } else {
                    out_activation_1[i] = out_conv[i] * (out_conv[i] + 3.0) / 6.0;
                }
            }

            // 全結合層
            let mut out_dense = 0.0;
            for (i, out_activate) in out_activation_1.iter().enumerate() {
                out_dense += out_activate * self.dense[0][i];
            }

            return (out_dense, out_conv);
        }

        let mv = if let Some(mv) = mv {
            mv
        } else {
            panic!("mv == None")
        };
        let mut out_conv = if let Some(out_conv) = out_conv {
            out_conv.1
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
                        let token = self.token[0][sq.array_index()][pc.as_u8() as usize]
                            [from.array_index()][from_piece.as_u8() as usize];
                        let idx = self.conv_idx[sq.array_index()][from.array_index()];
                        out_conv[0] -= self.embedding[token][0] * self.conv[0][0][idx];
                        out_conv[1] -= self.embedding[token][1] * self.conv[1][0][idx];
                    } else {
                        let token = self.token[0][from.array_index()][from_piece.as_u8() as usize]
                            [sq.array_index()][pc.as_u8() as usize];
                        let idx = self.conv_idx[from.array_index()][sq.array_index()];
                        out_conv[0] -= self.embedding[token][0] * self.conv[0][0][idx];
                        out_conv[1] -= self.embedding[token][1] * self.conv[1][0][idx];
                    }
                    // 移動先
                    if sq.array_index() <= to.array_index() {
                        let token = self.token[0][sq.array_index()][pc.as_u8() as usize]
                            [to.array_index()][from_to_piece.as_u8() as usize];
                        let idx = self.conv_idx[sq.array_index()][to.array_index()];
                        out_conv[0] += self.embedding[token][0] * self.conv[0][0][idx];
                        out_conv[1] += self.embedding[token][1] * self.conv[1][0][idx];
                    } else {
                        let token = self.token[0][to.array_index()][from_to_piece.as_u8() as usize]
                            [sq.array_index()][pc.as_u8() as usize];
                        let idx = self.conv_idx[to.array_index()][sq.array_index()];
                        out_conv[0] += self.embedding[token][0] * self.conv[0][0][idx];
                        out_conv[1] += self.embedding[token][1] * self.conv[1][0][idx];
                    }
                }
            }
            // 移動元×移動先の2駒関係が過剰に足されているので引く
            if from.array_index() <= to.array_index() {
                let token = self.token[0][from.array_index()][from_piece.as_u8() as usize]
                    [to.array_index()][from_to_piece.as_u8() as usize];
                let idx = self.conv_idx[from.array_index()][to.array_index()];
                out_conv[0] -= self.embedding[token][0] * self.conv[0][0][idx];
                out_conv[1] -= self.embedding[token][1] * self.conv[1][0][idx];
            } else {
                let token = self.token[0][to.array_index()][from_to_piece.as_u8() as usize]
                    [from.array_index()][from_piece.as_u8() as usize];
                let idx = self.conv_idx[to.array_index()][from.array_index()];
                out_conv[0] -= self.embedding[token][0] * self.conv[0][0][idx];
                out_conv[1] -= self.embedding[token][1] * self.conv[1][0][idx];
            }
            // 移動先の1駒関係が足せていないのでここで足す
            let token = self.token[0][to.array_index()][from_to_piece.as_u8() as usize]
                [to.array_index()][from_to_piece.as_u8() as usize];
            let idx = self.conv_idx[to.array_index()][to.array_index()];
            out_conv[0] += self.embedding[token][0] * self.conv[0][0][idx];
            out_conv[1] += self.embedding[token][1] * self.conv[1][0][idx];

            // 駒取りの場合は追加
            if let Some(to_piece) = pos.piece_at(to) {
                //盤面の駒取り
                for sq in Square::all() {
                    if let Some(pc) = pos.piece_at(sq) {
                        // 移動先
                        if sq.array_index() <= to.array_index() {
                            let token = self.token[0][sq.array_index()][pc.as_u8() as usize]
                                [to.array_index()][to_piece.as_u8() as usize];
                            let idx = self.conv_idx[sq.array_index()][to.array_index()];
                            out_conv[0] -= self.embedding[token][0] * self.conv[0][0][idx];
                            out_conv[1] -= self.embedding[token][1] * self.conv[1][0][idx];
                        } else {
                            let token = self.token[0][to.array_index()][to_piece.as_u8() as usize]
                                [sq.array_index()][pc.as_u8() as usize];
                            let idx = self.conv_idx[to.array_index()][sq.array_index()];
                            out_conv[0] -= self.embedding[token][0] * self.conv[0][0][idx];
                            out_conv[1] -= self.embedding[token][1] * self.conv[1][0][idx];
                        }
                    }
                }
                // 移動元×移動先の2駒関係が過剰に引かれているので足す
                if from.array_index() <= to.array_index() {
                    let token = self.token[0][from.array_index()][from_piece.as_u8() as usize]
                        [to.array_index()][to_piece.as_u8() as usize];
                    let idx = self.conv_idx[from.array_index()][to.array_index()];
                    out_conv[0] += self.embedding[token][0] * self.conv[0][0][idx];
                    out_conv[1] += self.embedding[token][1] * self.conv[1][0][idx];
                } else {
                    let token = self.token[0][to.array_index()][to_piece.as_u8() as usize]
                        [from.array_index()][from_piece.as_u8() as usize];
                    let idx = self.conv_idx[to.array_index()][from.array_index()];
                    out_conv[0] += self.embedding[token][0] * self.conv[0][0][idx];
                    out_conv[1] += self.embedding[token][1] * self.conv[1][0][idx];
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
                    let token =
                        self.token[0][0][0][idx + piece_kind.array_index()][count as usize + 1];
                    out_conv[0] += self.embedding[token][0]
                        * self.conv[0][0][3321 - 81 + idx + piece_kind.array_index()];
                    out_conv[1] += self.embedding[token][1]
                        * self.conv[1][0][3321 - 81 + idx + piece_kind.array_index()];
                    if count > 0 {
                        let token =
                            self.token[0][0][0][idx + piece_kind.array_index()][count as usize];
                        out_conv[0] -= self.embedding[token][0]
                            * self.conv[0][0][3321 - 81 + idx + piece_kind.array_index()];
                        out_conv[1] -= self.embedding[token][1]
                            * self.conv[1][0][3321 - 81 + idx + piece_kind.array_index()];
                    }
                } else {
                    panic!("Can not capture piece.");
                }
            }

            // 活性化関数_1
            let mut out_activation_1 = [0f32; 2];
            for i in 0..2 {
                if out_conv[i] <= -3.0 {
                    out_activation_1[i] = 0.0;
                } else if out_conv[i] >= 3.0 {
                    out_activation_1[i] = out_conv[i];
                } else {
                    out_activation_1[i] = out_conv[i] * (out_conv[i] + 3.0) / 6.0;
                }
            }

            // 全結合層
            let mut out_dense = 0.0;
            for (i, out_actiate) in out_activation_1.iter().enumerate() {
                out_dense += out_actiate * self.dense[0][i];
            }

            return (out_dense, out_conv);
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
                let token = self.token[0][0][0][idx + piece_kind.array_index()][count as usize];
                out_conv[0] -= self.embedding[token][0]
                    * self.conv[0][0][3321 - 81 + idx + piece_kind.array_index()];
                out_conv[1] -= self.embedding[token][1]
                    * self.conv[1][0][3321 - 81 + idx + piece_kind.array_index()];
                if count > 1 {
                    let token =
                        self.token[0][0][0][idx + piece_kind.array_index()][count as usize - 1];
                    out_conv[0] += self.embedding[token][0]
                        * self.conv[0][0][3321 - 81 + idx + piece_kind.array_index()];
                    out_conv[1] += self.embedding[token][1]
                        * self.conv[1][0][3321 - 81 + idx + piece_kind.array_index()];
                }
            } else {
                panic!("A piece not in the hand was used.");
            }
            // 盤面の差分計算
            for sq in Square::all() {
                if let Some(pc) = pos.piece_at(sq) {
                    if sq.array_index() <= to.array_index() {
                        let token = self.token[0][sq.array_index()][pc.as_u8() as usize]
                            [to.array_index()][piece.as_u8() as usize];
                        let idx = self.conv_idx[sq.array_index()][to.array_index()];
                        out_conv[0] += self.embedding[token][0] * self.conv[0][0][idx];
                        out_conv[1] += self.embedding[token][1] * self.conv[1][0][idx];
                    } else {
                        let token = self.token[0][to.array_index()][piece.as_u8() as usize]
                            [sq.array_index()][pc.as_u8() as usize];
                        let idx = self.conv_idx[to.array_index()][sq.array_index()];
                        out_conv[0] += self.embedding[token][0] * self.conv[0][0][idx];
                        out_conv[1] += self.embedding[token][1] * self.conv[1][0][idx];
                    }
                }
            }
            // 移動先の1駒関係が足せていないのでここで足す
            let token = self.token[0][to.array_index()][piece.as_u8() as usize][to.array_index()]
                [piece.as_u8() as usize];
            let idx = self.conv_idx[to.array_index()][to.array_index()];
            out_conv[0] += self.embedding[token][0] * self.conv[0][0][idx];
            out_conv[1] += self.embedding[token][1] * self.conv[1][0][idx];

            // 活性化関数_1
            let mut out_activation_1 = [0f32; 2];
            for i in 0..2 {
                if out_conv[i] <= -3.0 {
                    out_activation_1[i] = 0.0;
                } else if out_conv[i] >= 3.0 {
                    out_activation_1[i] = out_conv[i];
                } else {
                    out_activation_1[i] = out_conv[i] * (out_conv[i] + 3.0) / 6.0;
                }
            }

            // 全結合層
            let mut out_dense = 0.0;
            for (i, out_activate) in out_activation_1.iter().enumerate() {
                out_dense += out_activate * self.dense[0][i];
            }

            return (out_dense, out_conv);
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
            embedding: vec![vec![0.; 2]; 17346050],
            conv: vec![vec![vec![0.; 3335]; 1]; 2],
            dense: vec![vec![0.; 2]; 1],
        };
        let mut file = std::fs::File::create(path).unwrap();
        let value = serde_json::to_string(&eval).unwrap();
        file.write_all(value.as_bytes()).unwrap();

        let pos = Position::default();

        let eval = Evaluate::new(path);
        let value = eval.inference_diff(&pos, None, None);

        assert!(value.0 == 0.0);
    }
}

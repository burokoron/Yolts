use serde::{Deserialize, Serialize};
use shogi_core::{Color, Piece, PieceKind, Square};
use yasai::Position;

const VALUE_SCALE: f32 = 512.;

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

    pub fn inference(&self, pos: &Position) -> i32 {
        //! 局面を評価する
        //!
        //! - Arguments
        //!   - pos: &Position
        //!     - 評価したい局面
        //!
        //! - Returns
        //!   - value: i32
        //!     - 評価値

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
                value += self.model[0][sq1.array_index()][pc1.as_u8() as usize][sq2.array_index()]
                    [pc2.as_u8() as usize];
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

        (value * VALUE_SCALE) as i32
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
        let value = eval.inference(&pos);

        assert!(value == 0);
    }
}

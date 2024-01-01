use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};
use shogi_core::{Color, PartialPosition};
use std::{collections::HashMap, io::Write};
use yasai::Position;

use crate::search;

#[derive(Deserialize, Serialize)]
struct GameResult {
    black: u32,
    white: u32,
    draw: u32,
}

pub struct ThompsonSamplingBook {
    book: HashMap<String, GameResult>,
    rng: rand::rngs::ThreadRng,
}

impl ThompsonSamplingBook {
    pub fn new() -> Self {
        //! Thompson Sampling を用いた定跡のインスタンスを作成

        ThompsonSamplingBook {
            book: HashMap::new(),
            rng: rand::thread_rng(),
        }
    }

    pub fn load(&mut self, path: String) {
        //! 定跡ファイルの読み込み
        //!
        //! - Arguments
        //!   - path: String
        //!     - 定跡ファイルパス

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .expect("Not Found eval file.");
        let reader = std::io::BufReader::new(file);
        let book: HashMap<String, GameResult> =
            serde_json::from_reader(reader).expect("Cannot Read json file.");
        self.book = book;
    }

    pub fn save(&self, path: String) {
        //! 定跡ファイルの保存
        //!
        //! - Arguments
        //!   - path: String
        //!     - 定跡ファイルパス

        let mut file = std::fs::File::create(path).unwrap();
        let value = serde_json::to_string(&self.book).unwrap();
        file.write_all(value.as_bytes()).unwrap();
    }

    pub fn make(&mut self, ppos: PartialPosition) -> Option<String> {
        //! Thompson Samplingを用いた指し手の生成
        //!
        //! - 現在の局面が定跡登録済であれば指し手を生成する
        //! - 現在の局面が定跡にないのであれば指し手を生成しない
        //!
        //! - Arguments
        //!   - ppos: PartialPosition
        //!     - 現在の局面
        //! - Returns
        //!   - best_move: Option<String>
        //!     - sfen形式の指し手

        if self.book.contains_key(&ppos.to_sfen_owned()) {
            let pos = Position::new(ppos.clone());
            let legal_moves = pos.legal_moves();
            let mut best_value = -10.0;
            let mut best_move = None;
            for m in legal_moves {
                let mut p = ppos.clone();
                p.make_move(m);
                if !p.ply_set(1) {
                    panic!();
                }
                let (alpha, beta) = if let Some(game_result) = self.book.get(&p.to_sfen_owned()) {
                    (
                        (game_result.black + game_result.draw) as f32,
                        (game_result.white + game_result.draw) as f32,
                    )
                } else {
                    (1.0, 1.0)
                };
                let rand_beta = Beta::new(alpha, beta)
                    .expect("The beta distribution random number generator cannot be initialized.");
                let value = if ppos.side_to_move() == Color::Black {
                    rand_beta.sample(&mut self.rng)
                } else {
                    -rand_beta.sample(&mut self.rng)
                };
                if best_value < value {
                    best_value = value;
                    best_move = Some(m);
                }
            }
            best_move.map(search::move_to_sfen)
        } else {
            None
        }
    }

    pub fn entry(&mut self, ppos: PartialPosition, winner: Option<Color>) {
        //! 定跡の登録
        //!
        //! - Arguments
        //!   - ppos: PartialPosition
        //!     - 現在の局面
        //!   - winner: Option<Color>
        //!     - 勝利したプレイヤー

        let pos = self.book.entry(ppos.to_sfen_owned()).or_insert(GameResult {
            black: 1,
            white: 1,
            draw: 0,
        });
        match winner {
            Some(Color::Black) => pos.black += 1,
            Some(Color::White) => pos.white += 1,
            None => pos.draw += 1,
        }
    }

    pub fn search(&mut self, ppos: PartialPosition, narrow: u32) -> Option<String> {
        //! 定跡を検索し、手を返す
        //!
        //! - Arguments
        //!   - ppos: PartialPosition
        //!     - 現在の局面
        //!   - narrow: u32
        //!     - 訪問数がnarrow以上の手のみ選択する
        //! - Return
        //!   - best_move: Option<String>
        //!     - 選択された指し手
        //!     - 指し手が存在しないならNone

        let pos = Position::new(ppos.clone());
        let legal_moves = pos.legal_moves();
        let mut best_value = -10.0;
        let mut best_move = None;
        struct MultiPV {
            value: f32,
            m: String,
            black_wins: u32,
            white_wins: u32,
            draw: u32,
        }
        let mut multipv: Vec<MultiPV> = Vec::new();
        for m in legal_moves {
            let mut p = ppos.clone();
            p.make_move(m);
            if !p.ply_set(1) {
                panic!();
            }
            let mut black_wins = 0;
            let mut white_wins = 0;
            let mut draw = 0;
            let (alpha, beta) = if let Some(game_result) = self.book.get(&p.to_sfen_owned()) {
                black_wins = game_result.black;
                white_wins = game_result.white;
                draw = game_result.draw;
                ((black_wins + draw) as f32, (white_wins + draw) as f32)
            } else {
                (1.0, 1.0)
            };
            if alpha + beta - 2.0 < narrow as f32 {
                continue;
            }
            let rand_beta = Beta::new(alpha, beta)
                .expect("The beta distribution random number generator cannot be initialized.");
            let value = if ppos.side_to_move() == Color::Black {
                rand_beta.sample(&mut self.rng)
            } else {
                1.0 - rand_beta.sample(&mut self.rng)
            };
            multipv.push(MultiPV {
                value,
                m: search::move_to_sfen(m),
                black_wins,
                white_wins,
                draw,
            });
            if best_value < value {
                best_value = value;
                best_move = Some(m);
            }
        }
        multipv.sort_by(|i, j| (-i.value).partial_cmp(&(-j.value)).unwrap());
        for (i, pv) in multipv.iter().enumerate() {
            println!(
                "info score cp {} multipv {} pv {}",
                (pv.value * 100.0) as u32,
                i + 1,
                pv.m
            );
            println!(
                "info string {} move_rate={:.2}% black_wins={} white_wins={} draw={}",
                pv.m,
                pv.value * 100.0,
                pv.black_wins,
                pv.white_wins,
                pv.draw
            );
        }
        best_move.map(search::move_to_sfen)
    }
}

#[cfg(test)]
mod test {
    use crate::extra;
    use shogi_core::{Color, PartialPosition};

    #[test]
    fn make_entry() {
        let mut tsbook = extra::ThompsonSamplingBook::new();
        let ppos = PartialPosition::default();

        // 指し手の生成(None)
        let bestmove = tsbook.make(ppos.clone());
        assert!(bestmove.is_none());

        // 定跡の登録
        let winner = Some(Color::Black);
        tsbook.entry(ppos.clone(), winner);

        // 指し手の生成(Some)
        let bestmove = tsbook.make(ppos);
        assert!(bestmove.is_some());
    }

    #[test]
    fn entry_save_load() {
        let mut tsbook = extra::ThompsonSamplingBook::new();
        let ppos = PartialPosition::default();

        // 定跡の登録
        let winner = Some(Color::Black);
        tsbook.entry(ppos.clone(), winner);

        // 定跡の保存
        tsbook.save("test/book.json".to_string());

        // 定跡の読み込み
        tsbook.load("test/book.json".to_string());

        let game_result = tsbook.book.get(&ppos.to_sfen_owned());
        if let Some(game_result) = game_result {
            assert_eq!(game_result.black, 2);
            assert_eq!(game_result.white, 1);
            assert_eq!(game_result.draw, 0);
        } else {
            panic!("The position is not registered.")
        }
    }

    #[test]
    fn search() {
        let mut tsbook = extra::ThompsonSamplingBook::new();
        let ppos = PartialPosition::default();

        // 定跡の登録
        let winner = Some(Color::Black);
        tsbook.entry(ppos.clone(), winner);

        // 定跡の検索
        let bestmove = tsbook.search(ppos.clone(), 1);
        assert!(bestmove.is_none());

        let bestmove = tsbook.search(ppos, 0);
        assert!(bestmove.is_some());
    }
}

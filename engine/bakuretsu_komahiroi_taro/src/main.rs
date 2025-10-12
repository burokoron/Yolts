use shogi_core::{Color, Move, PartialPosition, Piece, Square};
use shogi_usi_parser::FromUsi;
use std::collections::HashSet;
use yasai::Position;

mod book;
mod evaluate;
mod search;
use evaluate::Evaluate;
use search::MATING_VALUE;

struct BakuretsuKomahiroiTaro {
    engine_name: String,
    author: String,
    eval_file: String,
    depth_limit: u32,
    book_file_path: String,
    narrow_book: u32,
    use_book: bool,
    search_mode: String,
    searcher: Option<search::NegaAlpha>,
}

impl BakuretsuKomahiroiTaro {
    fn new() -> Self {
        //! エンジンのインスタンスを作成
        //!
        //! - Returns
        //!   - Self: BakuretsuKomahiroiTaro
        //!     - エンジンのインスタンス

        BakuretsuKomahiroiTaro {
            engine_name: "爆裂駒拾太郎".to_string(),
            author: "burokoron".to_string(),
            eval_file: "eval.json".to_string(),
            depth_limit: 9,
            book_file_path: "book.json".to_string(),
            narrow_book: 10,
            use_book: true,
            search_mode: search::SEARCH_MODE_STANDARD.to_string(),
            searcher: None,
        }
    }

    fn usi(&self) {
        //! エンジン名(バージョン番号付き)とオプションを返答

        println!(
            "id name {} v{}",
            self.engine_name,
            env!("CARGO_PKG_VERSION")
        );
        println!("id author {}", self.author);
        println!(
            "option name EvalFile type string default {}",
            self.eval_file
        );
        println!(
            "option name DepthLimit type spin default {} min 0 max 1000",
            self.depth_limit
        );
        println!(
            "option name NarrowBook type spin default {} min 1 max 10000000",
            self.narrow_book
        );
        println!("option name UseBook type check default {}", self.use_book);
        println!(
            "option name SearchMode type combo default {} var Standard var Priority-27-Point var Absolute-27-Point",
            self.search_mode
        );
        println!("usiok");
    }

    fn isready(&mut self, tsbook: &mut book::ThompsonSamplingBook) {
        //! 対局の準備をする
        //!
        //! - Arguments
        //!  - tsbook: &mut ThompsonSamplingBook

        // 探索準備
        if self.searcher.is_none() {
            self.searcher = Some(search::NegaAlpha {
                my_turn: Color::Black, // 開始局面の手番は先手(Black)。後で局面に応じて上書きされる
                start_time: std::time::Instant::now(),
                max_time: 0,
                num_searched: 0,
                max_depth: 1,
                max_board_number: 0,
                best_move_pv: None,
                eval: Evaluate::new(&self.eval_file),
                search_mode: search::SearchMode::from_str(&self.search_mode),
                hash_table: vec![
                    search::HashTableValue {
                        key: 0,
                        depth: 0,
                        upper: MATING_VALUE,
                        lower: -MATING_VALUE,
                        best_move: None,
                        generation: 0,
                    };
                    200000
                ],
                hash_table_generation: 0,
                move_ordering: search::MoveOrdering {
                    piece_to_history: vec![vec![vec![0; 81]; 14]; 2],
                    killer_heuristic: vec![vec![None; 2]; self.depth_limit as usize + 1],
                    counter_move: vec![vec![vec![None; 2]; 81]; 14],
                },
                position_history: HashSet::new(),
                position_value: Vec::new(),
            })
        }
        // 定跡の読み込み
        if self.use_book {
            tsbook.load(self.book_file_path.clone());
        }

        println!("readyok");
    }

    fn setoption(&mut self, name: String, value: String) {
        //! エンジンのパラメータを設定する
        //!
        //! - Arguments
        //!   - name: String
        //!     - パラメータ名
        //!   - value: String
        //!     - 設定する値

        match &name[..] {
            "EvalFile" => {
                self.eval_file = value;
                self.searcher = None;
            }
            "DepthLimit" => {
                self.depth_limit = value
                    .parse()
                    .expect("Cannot convert the set DepthLimit value.")
            }
            "NarrowBook" => {
                self.narrow_book = value
                    .parse()
                    .expect("Cannot convert the set NarrowBook value.")
            }
            "UseBook" => {
                self.use_book = value
                    .parse()
                    .expect("Cannot convert the set UseBook value.")
            }
            "SearchMode" => {
                self.search_mode = value;
                self.searcher = None;
            }
            _ => (),
        }
    }

    fn usinewgame(&mut self) {
        //! 新規対局の準備をする
        //! - 何もしない
    }

    fn position(
        &self,
        startpos: &str,
        moves: Vec<&str>,
    ) -> (PartialPosition, Position, HashSet<u64>) {
        //! 現局面の反映
        //!
        //! - Arguments
        //!   - startpos: &str
        //!     - 開始局面のsfen局面
        //!   - moves: Vec<&str>
        //!     - 開始局面から現在までの手順
        //! - Returns
        //!   - (ppos: PartialPosition, pos: Position, position_history: HashSet<u64>)
        //!   - ppos: PartialPosition
        //!     - 現在の局面(shogi_core)
        //!   - pos: Position
        //!     - 現在の局面(yasai)
        //!   - position_history: HashSet<u64>
        //!     - 局面の履歴

        // 開始局面の反映
        let mut ppos =
            PartialPosition::from_usi(startpos).expect("Cannot convert SFEN format positions.");
        let mut pos = Position::new(ppos.clone());
        let mut position_history = HashSet::new();
        position_history.insert(pos.key());

        // 指し手を進める
        for m in moves {
            let m: Vec<char> = m.chars().collect();
            let m = {
                if m[1] == '*' {
                    let to = Square::new(m[2] as u8 - b'1' + 1, m[3] as u8 - b'a' + 1)
                        .expect("Cannot convert SFEN format moves.");
                    let piece = {
                        if pos.side_to_move() == Color::Black {
                            match m[0] {
                                'P' => Piece::B_P,
                                'L' => Piece::B_L,
                                'N' => Piece::B_N,
                                'S' => Piece::B_S,
                                'G' => Piece::B_G,
                                'B' => Piece::B_B,
                                'R' => Piece::B_R,
                                _ => unreachable!(),
                            }
                        } else {
                            match m[0] {
                                'P' => Piece::W_P,
                                'L' => Piece::W_L,
                                'N' => Piece::W_N,
                                'S' => Piece::W_S,
                                'G' => Piece::W_G,
                                'B' => Piece::W_B,
                                'R' => Piece::W_R,
                                _ => unreachable!(),
                            }
                        }
                    };
                    Move::Drop { piece, to }
                } else {
                    let from = Square::new(m[0] as u8 - b'1' + 1, m[1] as u8 - b'a' + 1)
                        .expect("Cannot convert SFEN format moves.");
                    let to = Square::new(m[2] as u8 - b'1' + 1, m[3] as u8 - b'a' + 1)
                        .expect("Cannot convert SFEN format moves.");
                    let promote = m.len() == 5;
                    Move::Normal { from, to, promote }
                }
            };
            ppos.make_move(m);
            pos.do_move(m);
            position_history.insert(pos.key());
        }
        position_history.remove(&pos.key());

        (ppos, pos, position_history)
    }

    fn go(
        &mut self,
        ppos: &PartialPosition,
        tsbook: &mut book::ThompsonSamplingBook,
        pos: &mut Position,
        position_history: &mut HashSet<u64>,
        max_time: i32,
    ) -> String {
        //! 思考し、最善手を返す
        //!
        //! - Arguments
        //!   - ppos: &PartialPosition
        //!     - 現在の局面(shogi_core)
        //!   - tsbook: &mut ThompsonSamplingBook
        //!     - 定跡
        //!   - pos: &mut Position
        //!     - 現在の局面(yasai)
        //!   - position_history: &mut HashSet<u64>
        //!     - 局面の履歴
        //!   - max_time: i32
        //!     - 探索制限時間
        //! - Returns
        //!   - best_move: String
        //!     - 最善手

        // 定跡の検索
        if self.use_book {
            if let Some(bestmove) = tsbook.search((*ppos).clone(), self.narrow_book) {
                return bestmove;
            }
        }

        // 入玉宣言の確認
        if search::is_nyugyoku_win(pos) {
            return "win".to_string();
        }

        // 探索部の初期化
        let mut best_move = "resign".to_string();
        if let Some(ref mut searcher) = self.searcher {
            searcher.my_turn = pos.side_to_move();
            searcher.start_time = std::time::Instant::now();
            searcher.max_time = max_time;
            searcher.num_searched = 0;
            searcher.max_depth = 1;
            searcher.max_board_number = pos.ply();
            searcher.best_move_pv = None;
            searcher.hash_table_generation += 1;
            searcher.move_ordering = search::MoveOrdering {
                piece_to_history: vec![vec![vec![0; 81]; 14]; 2],
                killer_heuristic: vec![vec![None; 2]; self.depth_limit as usize + 1],
                counter_move: vec![vec![vec![None; 2]; 81]; 14],
            };
            searcher.position_history.clone_from(position_history);
            searcher.position_value = vec![searcher.eval.inference_diff(pos, None, None); 1];

            // 探索
            for depth in 1..=self.depth_limit {
                searcher.max_depth = depth;
                let value = searcher.search(pos, false, depth, -MATING_VALUE, MATING_VALUE, None);
                let end = searcher.start_time.elapsed();
                let elapsed_time =
                    end.as_secs() as i32 * 1000 + end.subsec_nanos() as i32 / 1_000_000;
                let nps = if elapsed_time != 0 {
                    searcher.num_searched * 1000 / elapsed_time as u64
                } else {
                    searcher.num_searched
                };

                if elapsed_time < searcher.max_time {
                    best_move = {
                        if let Some(ref m) = searcher.best_move_pv {
                            search::move_to_sfen(*m)
                        } else {
                            "resign".to_string()
                        }
                    };
                    let mut pv = searcher.pv_to_sfen(pos, position_history);
                    if pv.is_empty() {
                        pv = "resign ".to_string();
                    }
                    print!(
                        "info depth {} seldepth {} time {} nodes {} ",
                        depth,
                        searcher.max_board_number - pos.ply(),
                        elapsed_time,
                        searcher.num_searched
                    );
                    println!("score cp {} nps {} pv {}", value, nps, pv);
                } else {
                    break;
                }

                // mateなら探索終了
                if value.abs() > MATING_VALUE - 1000 {
                    break;
                }
            }
        } else {
            panic!("Searcher is not loaded.");
        }

        best_move
    }

    fn stop(&self) {
        //! 思考停止コマンドに対応する
        //! - 未対応
    }

    fn ponderhit(&self) {
        //! 先読みが当たった場合に対応する
        //! - 未対応
    }

    fn quit(&self) {
        //! 終了コマンドに対応する
        //! - すぐに反応はできないが、終了する
        //! - 現時点で終了時にやるべきことはない
    }

    fn gameover(&self) {
        //! 対局終了通知に対応する
        //! - 今のところ対応の必要なし
    }
}

fn main() {
    // 初期化
    let engine = &mut BakuretsuKomahiroiTaro::new();
    let mut ppos = PartialPosition::default();
    let mut pos = Position::default();
    let mut position_history = HashSet::new();
    let tsbook = &mut book::ThompsonSamplingBook::new();

    loop {
        // 入力の受け取り
        let inputs: Vec<String> = {
            let mut line: String = String::new();
            std::io::stdin()
                .read_line(&mut line)
                .expect("Cannot read USI command.");
            line.split_whitespace()
                .map(|x| x.parse().expect("Cannot parse USI command."))
                .collect()
        };

        match &inputs[0][..] {
            "usi" => {
                // エンジン名を返答
                engine.usi();
            }
            "isready" => {
                // 対局準備
                engine.isready(tsbook);
            }
            "setoption" => {
                // エンジンのパラメータ設定
                engine.setoption(inputs[2].clone(), inputs[4].clone());
            }
            "usinewgame" => {
                // 新規対局準備
                engine.usinewgame();
            }
            "position" => {
                // 現局面の反映
                let mut moves: Vec<&str> = Vec::new();
                let startpos = {
                    if inputs[1] == "startpos" {
                        if inputs.len() > 3 {
                            for m in &inputs[3..] {
                                moves.push(m);
                            }
                        }
                        "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
                            .to_string()
                    } else {
                        if inputs.len() > 7 {
                            for m in &inputs[7..] {
                                moves.push(m);
                            }
                        }
                        format!(
                            "sfen {} {} {} {}",
                            inputs[2], inputs[3], inputs[4], inputs[5]
                        )
                    }
                };
                (ppos, pos, position_history) = engine.position(&startpos, moves);
            }
            "go" => {
                // 思考して指し手を返答
                // 120手で終局想定で時間制御
                let margin_time = 1000;
                let mut min_time = 0;
                let mut max_time = 0;
                if &inputs[5][..] == "byoyomi" {
                    let byoyomi: i32 = inputs[6].parse().expect("Cannot parse USI command.");
                    max_time += byoyomi;
                    min_time += byoyomi;
                    if pos.side_to_move() == Color::Black {
                        let btime: i32 = inputs[2].parse().expect("Cannot parse USI command.");
                        max_time += btime;
                    } else {
                        let wtime: i32 = inputs[4].parse().expect("Cannot parse USI command.");
                        max_time += wtime;
                    }
                } else if pos.side_to_move() == Color::Black {
                    let btime: i32 = inputs[2].parse().expect("Cannot parse USI command.");
                    let binc: i32 = inputs[6].parse().expect("Cannot parse USI command.");
                    max_time += btime;
                    max_time += binc;
                    min_time += binc;
                } else {
                    let wtime: i32 = inputs[4].parse().expect("Cannot parse USI command.");
                    let winc: i32 = inputs[8].parse().expect("Cannot parse USI command.");
                    max_time += wtime;
                    max_time += winc;
                    min_time += winc;
                }
                max_time -= margin_time;
                min_time -= margin_time;
                let mut remain_move_number = (120 - pos.ply() as i32) / 4;
                if remain_move_number <= 1 {
                    remain_move_number = 1
                }
                max_time = std::cmp::max(max_time / remain_move_number, min_time);
                let m = engine.go(&ppos, tsbook, &mut pos, &mut position_history, max_time);
                println!("bestmove {}", m);
            }
            "stop" => {
                // 思考停止コマンド
                engine.stop();
            }
            "ponderhit" => {
                // 先読みが当たった場合
                engine.ponderhit();
            }
            "quit" => {
                // 強制終了
                engine.quit();
                break;
            }
            "gameover" => {
                // 対局終了
                engine.gameover();
            }
            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use crate::book;
    use crate::evaluate;
    use crate::BakuretsuKomahiroiTaro;

    #[test]
    fn go() {
        let path = "test/go.json";

        let eval = evaluate::EvalJson {
            embedding: vec![vec![0.; 4]; 8673025],
            conv: vec![vec![vec![0.; 3335]; 1]; 4],
            dense: vec![vec![0.; 4]; 1],
        };

        let mut file = std::fs::File::create(path).unwrap();
        let value = serde_json::to_string(&eval).unwrap();
        file.write_all(value.as_bytes()).unwrap();

        let engine = &mut BakuretsuKomahiroiTaro::new();
        engine.setoption("EvalFile".to_string(), path.to_string());
        engine.setoption("DepthLimit".to_string(), "4".to_string());
        engine.setoption("UseBook".to_string(), "false".to_string());
        let tsbook = &mut book::ThompsonSamplingBook::new();
        engine.isready(tsbook);
        let mut pos;
        let ppos;
        let mut position_history;
        (ppos, pos, position_history) = engine.position(
            "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            vec!["7g7f"],
        );
        engine.go(&ppos, tsbook, &mut pos, &mut position_history, 10000);
    }
}

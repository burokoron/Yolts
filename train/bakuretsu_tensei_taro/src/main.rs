use encoding::all::WINDOWS_31J;
use encoding::{EncoderTrap, Encoding};
use shogi_core::{Color, Hand, Move, PartialPosition, Piece, Square};
use shogi_usi_parser::FromUsi;
use std::collections::{HashMap, HashSet};
use std::io::{stdout, Write};
use yasai::Position;

mod search;
use crate::search::MATING_VALUE;
mod extra;

#[derive(Clone)]
pub struct Eval {
    pieces_in_board: Vec<Vec<i32>>,
    pieces_in_hand: Vec<Vec<Vec<i32>>>,
}

struct BakuretsuTenseiTaro {
    engine_name: String,
    author: String,
    eval_file_path: String,
    eval_nyugyoku_file_path: String,
    eval: Eval,
    eval_nyugyoku: Eval,
    depth_limit: u32,
    book_file_path: String,
    narrow_book: u32,
    use_book: bool,
}

impl BakuretsuTenseiTaro {
    fn new() -> Self {
        //! エンジンのインスタンスを作成

        BakuretsuTenseiTaro {
            engine_name: "爆裂訓練太郎".to_string(),
            author: "burokoron".to_string(),
            eval_file_path: "eval.json".to_string(),
            eval_nyugyoku_file_path: "eval_nyugyoku.json".to_string(),
            eval: Eval {
                pieces_in_board: vec![vec![0; 31]; 81],
                pieces_in_hand: vec![vec![vec![0; 19]; 8]; 2],
            },
            eval_nyugyoku: Eval {
                pieces_in_board: vec![vec![0; 31]; 81],
                pieces_in_hand: vec![vec![vec![0; 19]; 8]; 2],
            },
            depth_limit: 9,
            book_file_path: "book.json".to_string(),
            narrow_book: 10,
            use_book: true,
        }
    }

    fn usi(&self) {
        //! エンジン名(バージョン番号付き)とオプションを返答

        print!("id name ");
        let mut out = stdout();
        let bytes = WINDOWS_31J
            .encode(&self.engine_name, EncoderTrap::Ignore)
            .expect("Cannot encode the engine name.");
        out.write_all(&bytes[..])
            .expect("Cannot write the engine name.");
        println!(" version {}", env!("CARGO_PKG_VERSION"));
        println!("id author {}", self.author);
        println!(
            "option name EvalFile type string default {}",
            self.eval_file_path
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
        println!("usiok");
    }

    fn isready(&mut self) {
        //! 対局の準備をする
        //! - 評価関数の読み込み

        // 通常の評価関数の読み込み
        let eval_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.eval_file_path)
            .expect("Not Found eval file.");
        let reader = std::io::BufReader::new(eval_file);
        let eval: serde_json::Value =
            serde_json::from_reader(reader).expect("Cannot Read json file.");
        // 盤面
        for sq in Square::all() {
            for piece in Piece::all() {
                let value = eval["pieces_dict"][sq.array_index().to_string()]
                    [piece.as_u8().to_string()]
                .as_i64()
                .expect("Cannot Convert eval_file value.");
                self.eval.pieces_in_board[sq.array_index()][piece.as_u8() as usize] = value as i32;
            }
        }
        // 持ち駒
        for color in Color::all() {
            for piece in Hand::all_hand_pieces() {
                let piece_idx = {
                    if color == Color::Black {
                        piece.array_index()
                    } else {
                        piece.array_index() + 7
                    }
                };
                match piece.array_index() {
                    0 => {
                        for i in 0..19 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()]
                                [i.to_string()]
                            .as_i64()
                            .expect("Cannot Convert eval_file value.");
                            self.eval.pieces_in_hand[color.array_index()][piece.array_index()][i] =
                                value as i32;
                        }
                    }
                    1 | 2 | 3 | 4 => {
                        for i in 0..5 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()]
                                [i.to_string()]
                            .as_i64()
                            .expect("Cannot Convert eval_file value.");
                            self.eval.pieces_in_hand[color.array_index()][piece.array_index()][i] =
                                value as i32;
                        }
                    }
                    5 | 6 => {
                        for i in 0..3 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()]
                                [i.to_string()]
                            .as_i64()
                            .expect("Cannot Convert eval_file value.");
                            self.eval.pieces_in_hand[color.array_index()][piece.array_index()][i] =
                                value as i32;
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        // 入玉用の評価関数の読み込み
        let eval_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.eval_nyugyoku_file_path)
            .expect("Not Found eval file.");
        let reader = std::io::BufReader::new(eval_file);
        let eval: serde_json::Value =
            serde_json::from_reader(reader).expect("Cannot Read json file.");
        // 盤面
        for sq in Square::all() {
            for piece in Piece::all() {
                let value = eval["pieces_dict"][sq.array_index().to_string()]
                    [piece.as_u8().to_string()]
                .as_i64()
                .expect("Cannot Convert eval_file value.");
                self.eval_nyugyoku.pieces_in_board[sq.array_index()][piece.as_u8() as usize] =
                    value as i32;
            }
        }
        // 持ち駒
        for color in Color::all() {
            for piece in Hand::all_hand_pieces() {
                let piece_idx = {
                    if color == Color::Black {
                        piece.array_index()
                    } else {
                        piece.array_index() + 7
                    }
                };
                match piece.array_index() {
                    0 => {
                        for i in 0..19 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()]
                                [i.to_string()]
                            .as_i64()
                            .expect("Cannot Convert eval_file value.");
                            self.eval_nyugyoku.pieces_in_hand[color.array_index()]
                                [piece.array_index()][i] = value as i32;
                        }
                    }
                    1 | 2 | 3 | 4 => {
                        for i in 0..5 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()]
                                [i.to_string()]
                            .as_i64()
                            .expect("Cannot Convert eval_file value.");
                            self.eval_nyugyoku.pieces_in_hand[color.array_index()]
                                [piece.array_index()][i] = value as i32;
                        }
                    }
                    5 | 6 => {
                        for i in 0..3 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()]
                                [i.to_string()]
                            .as_i64()
                            .expect("Cannot Convert eval_file value.");
                            self.eval_nyugyoku.pieces_in_hand[color.array_index()]
                                [piece.array_index()][i] = value as i32;
                        }
                    }
                    _ => unreachable!(),
                }
            }
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
            "EvalFile" => self.eval_file_path = value,
            "DepthLimit" => self.depth_limit = value.parse().unwrap(),
            "NarrowBook" => self.narrow_book = value.parse().unwrap(),
            "UseBook" => self.use_book = value.parse().unwrap(),
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
        let mut ppos = PartialPosition::from_usi(startpos).unwrap();
        let mut pos = Position::new(ppos.clone());
        let mut position_history = HashSet::new();
        position_history.insert(pos.key());

        // 指し手を進める
        for m in moves {
            let m: Vec<char> = m.chars().collect();
            let m = {
                if m[1] == '*' {
                    let to = Square::new(m[2] as u8 - b'1' + 1, m[3] as u8 - b'a' + 1).unwrap();
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
                    let from = Square::new(m[0] as u8 - b'1' + 1, m[1] as u8 - b'a' + 1).unwrap();
                    let to = Square::new(m[2] as u8 - b'1' + 1, m[3] as u8 - b'a' + 1).unwrap();
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
        tsbook: &mut extra::ThompsonSamplingBook,
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

        // 現在の局面が通常の評価関数で評価値が一定以下(互角)ならそのまま、そうでないなら入玉用評価関数に変更
        let mut nega = search::NegaAlpha {
            my_turn: pos.side_to_move(),
            start_time: std::time::Instant::now(),
            max_time,
            num_searched: 0,
            max_depth: 0,
            max_board_number: pos.ply(),
            best_move_pv: None,
            eval: self.eval.clone(),
            hash_table: search::HashTable {
                pos: HashMap::new(),
            },
            move_ordering: search::MoveOrdering {
                piece_to_history: vec![vec![vec![0; 81]; 14]; 2],
            },
        };
        let value = nega.search(pos, position_history, 0, -MATING_VALUE, MATING_VALUE);
        let is_eval_nyugyoku;
        if value > 2000 {
            nega.eval = self.eval_nyugyoku.clone();
            is_eval_nyugyoku = true;
        } else {
            is_eval_nyugyoku = false;
        }

        // 入玉宣言の確認
        if search::is_nyugyoku_win(pos) {
            return "win".to_string();
        }

        // 通常の探索
        let mut best_move = "resign".to_string();
        for depth in 1..=self.depth_limit {
            nega.max_depth = depth;
            let mut value = nega.search(pos, position_history, depth, -MATING_VALUE, MATING_VALUE);
            if is_eval_nyugyoku && value.abs() <= MATING_VALUE - 1000 {
                value = (value as f32 * (6842.0 / 5676.0)) as i32;
            }
            let end = nega.start_time.elapsed();
            let elapsed_time = end.as_secs() as i32 * 1000 + end.subsec_nanos() as i32 / 1_000_000;
            let nps = if elapsed_time != 0 {
                nega.num_searched * 1000 / elapsed_time as u64
            } else {
                nega.num_searched
            };

            if elapsed_time < nega.max_time {
                best_move = {
                    if let Some(ref m) = nega.best_move_pv {
                        search::move_to_sfen(*m)
                    } else {
                        "resign".to_string()
                    }
                };
                let mut pv = nega.pv_to_sfen(pos, position_history);
                if pv.is_empty() {
                    pv = "resign ".to_string();
                }
                print!(
                    "info depth {} seldepth {} time {} nodes {} ",
                    depth,
                    nega.max_board_number - pos.ply(),
                    elapsed_time,
                    nega.num_searched
                );
                println!("score cp {} pv {}nps {}", value, pv, nps);
            } else {
                break;
            }

            // mateなら探索終了
            if value.abs() > MATING_VALUE - 1000 {
                break;
            }
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
    let engine = &mut BakuretsuTenseiTaro::new();
    let mut ppos = PartialPosition::default();
    let mut pos = Position::default();
    let mut position_history = HashSet::new();
    let tsbook = &mut extra::ThompsonSamplingBook::new();

    loop {
        // 入力の受け取り
        let inputs: Vec<String> = {
            let mut line: String = String::new();
            std::io::stdin().read_line(&mut line).unwrap();
            line.split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect()
        };

        match &inputs[0][..] {
            "usi" => {
                // エンジン名を返答
                engine.usi();
            }
            "isready" => {
                // 対局準備
                tsbook.load(engine.book_file_path.clone());
                engine.isready();
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
                    let byoyomi: i32 = inputs[6].parse().unwrap();
                    max_time += byoyomi;
                    min_time += byoyomi;
                    if pos.side_to_move() == Color::Black {
                        let btime: i32 = inputs[2].parse().unwrap();
                        max_time += btime;
                    } else {
                        let wtime: i32 = inputs[4].parse().unwrap();
                        max_time += wtime;
                    }
                } else if pos.side_to_move() == Color::Black {
                    let btime: i32 = inputs[2].parse().unwrap();
                    let binc: i32 = inputs[6].parse().unwrap();
                    max_time += btime;
                    max_time += binc;
                    min_time += binc;
                } else {
                    let wtime: i32 = inputs[4].parse().unwrap();
                    let winc: i32 = inputs[8].parse().unwrap();
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
            "extra" => match &inputs[1][..] {
                "load" => {
                    // 定跡の読み込み
                    tsbook.load(inputs[2].clone());
                }
                "save" => {
                    // 定跡の保存
                    tsbook.save(inputs[2].clone());
                }
                "make" => {
                    // Thompson Sampling に基づいて指し手を生成
                    let ppos = PartialPosition::from_usi(&format!(
                        "sfen {} {} {} 1",
                        inputs[2], inputs[3], inputs[4]
                    ))
                    .unwrap();
                    let bestmove = tsbook.make(ppos);
                    if let Some(bestmove) = bestmove {
                        println!("{}", bestmove);
                    } else {
                        println!("None");
                    }
                }
                "entry" => {
                    // 局面の勝敗を定跡に登録
                    let ppos = PartialPosition::from_usi(&format!(
                        "sfen {} {} {} 1",
                        inputs[2], inputs[3], inputs[4]
                    ))
                    .unwrap();
                    let winner = match &inputs[5][..] {
                        "b" => Some(Color::Black),
                        "w" => Some(Color::White),
                        "d" => None,
                        _ => unreachable!(),
                    };
                    tsbook.entry(ppos, winner);
                }
                _ => (),
            },
            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::extra;
    use crate::BakuretsuTenseiTaro;

    #[test]
    fn go() {
        let engine = &mut BakuretsuTenseiTaro::new();
        engine.setoption("EvalFile".to_string(), "test/eval.json".to_string());
        engine.setoption("DepthLimit".to_string(), "4".to_string());
        engine.isready();
        let mut pos;
        let ppos;
        let tsbook = &mut extra::ThompsonSamplingBook::new();
        let mut position_history;
        (ppos, pos, position_history) = engine.position(
            "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            vec!["7g7f"],
        );
        engine.go(&ppos, tsbook, &mut pos, &mut position_history, 10000);
    }
}

use encoding::all::WINDOWS_31J;
use encoding::{EncoderTrap, Encoding};
use shogi_core::{Color, Hand, Move, PartialPosition, Piece, Square};
use shogi_usi_parser::FromUsi;
use std::collections::HashMap;
use std::io::{stdout, Write};
use std::{collections::HashSet, time::Instant};
use yasai::Position;

mod search;

#[derive(Clone)]
pub struct Eval {
    pieces_in_board: Vec<Vec<Vec<i32>>>,
    pieces_in_hand: Vec<Vec<Vec<i32>>>,
}

struct BakuretsuTenseiTaro {
    engine_name: String,
    author: String,
    eval_file_path: String,
    eval: Eval,
    depth_limit: u32,
}

impl BakuretsuTenseiTaro {
    fn usi(v: &BakuretsuTenseiTaro) {
        /*
        エンジン名(バージョン番号付き)とオプションを返答

        Args:
            v: &BakuretsuTenseiTaro
                エンジン実行中に一時保存するデータ群
        */
        print!("id name ");
        let mut out = stdout();
        let bytes = WINDOWS_31J
            .encode(&v.engine_name, EncoderTrap::Ignore)
            .unwrap();
        out.write_all(&bytes[..]).unwrap();
        println!(" version {}", env!("CARGO_PKG_VERSION"));
        println!("id author {}", v.author);
        println!(
            "option name EvalFile type string default {}",
            v.eval_file_path
        );
        println!(
            "option name DepthLimit type spin default {} min 0 max 1000",
            v.depth_limit
        );
        println!("usiok");
    }

    fn isready(v: &mut BakuretsuTenseiTaro) {
        /*
        対局の準備をする
        */

        // 評価関数の読み込み
        let eval_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&v.eval_file_path)
            .expect("Not Found eval file.");
        let reader = std::io::BufReader::new(eval_file);
        let eval: serde_json::Value =
            serde_json::from_reader(reader).expect("Can not Read json file.");
        // 盤面
        for sq in Square::all() {
            let value = eval["pieces_dict"][sq.array_index().to_string()]["0"]
                .as_i64()
                .unwrap();
            v.eval.pieces_in_board[Color::Black.array_index()][sq.array_index()][0] = value as i32;
            v.eval.pieces_in_board[Color::White.array_index()][sq.array_index()][0] = value as i32;
            for piece in Piece::all() {
                let value = eval["pieces_dict"][sq.array_index().to_string()]
                    [piece.as_u8().to_string()]
                .as_i64()
                .unwrap();
                let color = if piece.as_u8() <= 16 {
                    Color::Black
                } else {
                    Color::White
                };
                v.eval.pieces_in_board[color.array_index()][sq.array_index()]
                    [piece.piece_kind().array_index() as usize + 1] = value as i32;
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
                            .unwrap();
                            v.eval.pieces_in_hand[color.array_index()][piece.array_index()][i] =
                                value as i32;
                        }
                    }
                    1 | 2 | 3 | 4 => {
                        for i in 0..5 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()]
                                [i.to_string()]
                            .as_i64()
                            .unwrap();
                            v.eval.pieces_in_hand[color.array_index()][piece.array_index()][i] =
                                value as i32;
                        }
                    }
                    5 | 6 => {
                        for i in 0..3 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()]
                                [i.to_string()]
                            .as_i64()
                            .unwrap();
                            v.eval.pieces_in_hand[color.array_index()][piece.array_index()][i] =
                                value as i32;
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        println!("readyok");
    }

    fn setoption(v: &mut BakuretsuTenseiTaro, name: String, value: String) {
        /*
        エンジンのパラメータを設定する

        Args:
            v: &BakuretsuKomasuteTaroR
                エンジン実行中に一時保存するデータ群
            name: String
                パラメータ名
            value: String
                設定する値
        */

        match &name[..] {
            "EvalFile" => v.eval_file_path = value,
            "DepthLimit" => v.depth_limit = value.parse().unwrap(),
            _ => (),
        }
    }

    fn usinewgame() {
        /*
        新規対局の準備をする
        */
    }

    fn position(startpos: &str, moves: Vec<&str>) -> (Position, HashSet<u64>) {
        /*
        現局面の反映

        Args:
            startpos: &str
                開始局面のsfen局面
            moves: Vec<&str>
                開始局面から現在までの手順
        */

        // 開始局面の反映
        let mut pos = Position::new(PartialPosition::from_usi(startpos).unwrap());
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
            pos.do_move(m);
            position_history.insert(pos.key());
        }
        position_history.remove(&pos.key());

        (pos, position_history)
    }

    fn go(
        v: &mut BakuretsuTenseiTaro,
        pos: &mut Position,
        position_history: &mut HashSet<u64>,
        max_time: i32,
    ) -> String {
        /*
        思考し、最善手を返す
        */

        let mut nega = search::NegaAlpha {
            my_turn: pos.side_to_move(),
            start_time: Instant::now(),
            max_time,
            num_searched: 0,
            max_depth: 1,
            max_board_number: pos.ply(),
            best_move_pv: None,
            eval: v.eval.clone(),
            hash_table: search::HashTable {
                pos: HashMap::new(),
            },
            move_ordering: search::MoveOrdering {
                piece_to_history: vec![vec![vec![0; 81]; 14]; 2],
            },
        };

        // 入玉宣言の確認
        if search::is_nyugyoku_win(pos) {
            return "win".to_string();
        }

        // 通常の探索
        let mut best_move = "resign".to_string();
        for depth in 1..=v.depth_limit {
            nega.max_depth = depth;
            let value =
                search::NegaAlpha::search(&mut nega, pos, position_history, depth, -30000, 30000);
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
                let mut pv = search::pv_to_sfen(&mut nega, pos, position_history);
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
            if value.abs() > 29000 {
                break;
            }
        }

        best_move
    }

    fn stop() {
        /*
        思考停止コマンドに対応する
        */

        // 未対応
    }

    fn ponderhit() {
        /*
        先読みが当たった場合に対応する
        */

        // 未対応
    }

    fn quit() {
        /*
        強制終了
        */

        // すぐに反応はできないが、終了する
        std::process::exit(1);
    }

    fn gameover() {
        /*
        対局終了通知に対応する
        */

        // 今のところ対応の必要なし
    }
}

fn main() {
    // 初期化
    let engine = &mut BakuretsuTenseiTaro {
        engine_name: "爆裂訓練太郎".to_string(),
        author: "burokoron".to_string(),
        eval_file_path: "eval.json".to_string(),
        eval: Eval {
            pieces_in_board: vec![vec![vec![0; 31]; 81]; 2],
            pieces_in_hand: vec![vec![vec![0; 19]; 8]; 2],
        },
        depth_limit: 9,
    };
    let mut pos = Position::default();
    let mut position_history = HashSet::new();

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
                BakuretsuTenseiTaro::usi(engine);
            }
            "isready" => {
                // 対局準備
                BakuretsuTenseiTaro::isready(engine);
            }
            "setoption" => {
                // エンジンのパラメータ設定
                BakuretsuTenseiTaro::setoption(engine, inputs[2].clone(), inputs[4].clone());
            }
            "usinewgame" => {
                // 新規対局準備
                BakuretsuTenseiTaro::usinewgame();
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
                (pos, position_history) = BakuretsuTenseiTaro::position(&startpos, moves);
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
                let m = BakuretsuTenseiTaro::go(engine, &mut pos, &mut position_history, max_time);
                println!("bestmove {}", m);
            }
            "stop" => {
                // 思考停止コマンド
                BakuretsuTenseiTaro::stop();
            }
            "ponderhit" => {
                // 先読みが当たった場合
                BakuretsuTenseiTaro::ponderhit();
            }
            "quit" => {
                // 強制終了
                BakuretsuTenseiTaro::quit();
            }
            "gameover" => {
                // 対局終了
                BakuretsuTenseiTaro::gameover();
            }
            _ => (),
        }
    }
}

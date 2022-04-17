use std::time::Instant;
use std::collections::HashMap;
use yasai::{ Color, Square, PieceType, Position };

//mod search;


#[derive(Clone)]
pub struct Eval {
    pieces_in_board: Vec<Vec<Vec<i32>>>,
    pieces_in_hand: Vec<Vec<Vec<i32>>>,
}

struct BakuretsuKomasuteTaroR {
    engine_name: String,
    author: String,
    eval_file_path: String,
    eval: Eval,
}

impl BakuretsuKomasuteTaroR {
    fn usi(v: &BakuretsuKomasuteTaroR) {
        /*
        エンジン名(バージョン番号付き)とオプションを返答

        Args:
            v: &BakuretsuKomasuteTaroR
                エンジン実行中に一時保存するデータ群
        */

        println!("id name {} version {}", v.engine_name, env!("CARGO_PKG_VERSION"));
        println!("id author {}", v.author);
        println!("option name EvalFile type string default {}", v.eval_file_path);
        println!("usiok");
    }

    fn isready(v: &mut BakuretsuKomasuteTaroR) {
        /*
        対局の準備をする
        */

        // 評価関数の読み込み
        let eval_file = std::fs::OpenOptions::new().read(true).write(true).open(&v.eval_file_path).expect("Not Found eval file.");
        let reader = std::io::BufReader::new(eval_file);
        let eval: serde_json::Value = serde_json::from_reader(reader).expect("Can not Read json file.");
        // 盤面
        for color in Color::ALL {
            for sq in Square::ALL {
                let value = eval["pieces_dict"][sq.index().to_string()]["0"].as_i64().unwrap();
                v.eval.pieces_in_board[color.index()][sq.index()][0] = value as i32;
                for piece in PieceType::ALL {
                    if color == Color::Black {
                        let value = eval["pieces_dict"][sq.index().to_string()][(piece.index()+1).to_string()].as_i64().unwrap();
                        v.eval.pieces_in_board[color.index()][sq.index()][piece.index()+1] = value as i32;
                    } else {
                        let value = eval["pieces_dict"][sq.index().to_string()][(piece.index()+17).to_string()].as_i64().unwrap();
                        v.eval.pieces_in_board[color.index()][sq.index()][piece.index()+17] = value as i32;
                    }
                }
            }
        }
        // 持ち駒
        for color in Color::ALL {
            for piece in PieceType::ALL_HAND {
                let piece_idx = {
                    if color == Color::Black { piece.index() } else { piece.index() + 7 }
                };
                match piece.index() {
                    0 => {
                        for i in 0..19 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()][i.to_string()].as_i64().unwrap();
                            v.eval.pieces_in_hand[color.index()][piece.index()][i] = value as i32;
                        }
                    },
                    1 => {
                        for i in 0..5 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()][i.to_string()].as_i64().unwrap();
                            v.eval.pieces_in_hand[color.index()][piece.index()][i] = value as i32;
                        }
                    },
                    2 => {
                        for i in 0..5 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()][i.to_string()].as_i64().unwrap();
                            v.eval.pieces_in_hand[color.index()][piece.index()][i] = value as i32;
                        }
                    },
                    3 => {
                        for i in 0..5 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()][i.to_string()].as_i64().unwrap();
                            v.eval.pieces_in_hand[color.index()][piece.index()][i] = value as i32;
                        }
                    },
                    4 => {
                        for i in 0..5 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()][i.to_string()].as_i64().unwrap();
                            v.eval.pieces_in_hand[color.index()][piece.index()][i] = value as i32;
                        }
                    },
                    5 => {
                        for i in 0..3 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()][i.to_string()].as_i64().unwrap();
                            v.eval.pieces_in_hand[color.index()][piece.index()][i] = value as i32;
                        }
                    },
                    6 => {
                        for i in 0..3 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()][i.to_string()].as_i64().unwrap();
                            v.eval.pieces_in_hand[color.index()][piece.index()][i] = value as i32;
                        }
                    },
                    _ => unreachable!(),
                }
            }
        }
        
        println!("readyok");
    }

    fn setoption(v: &mut BakuretsuKomasuteTaroR, name: String, value: String) {
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

        match &*name {
            "EvalFile" => {
                v.eval_file_path = value;
            },
            _ => (),
        }

    }

    fn usinewgame() {
        /*
        新規対局の準備をする
        */

    }

    fn position(startpos: &str, moves: Vec<&str>) -> Position {
        /*
        現局面の反映

        Args:
            startpos: &str
                開始局面のsfen局面
            moves: Vec<&str>
                開始局面から現在までの手順
        */

        /*
        let mut pos = Position::new();
        pos.set_sfen(startpos).unwrap();
        for m in moves {
            pos.make_move(Move::from_sfen(m).unwrap()).unwrap();
        }
        */

        let mut pos = Position::default();

        return pos;
    }

    fn go(eval: Eval, pos: &mut Position, max_time: i32) -> String {
        /*
        思考し、最善手を返す
        */

        /*
        let mut nega = search::NegaAlpha {
            start_time: Instant::now(),
            max_time: max_time,
            num_searched: 0,
            max_depth: 1.,
            max_board_number: pos.ply(),
            best_move_pv: "resign".to_string(),
            eval: eval,
            hash_table: search::HashTable {
                pos: HashMap::new(),
            },
            from_to_move_ordering: search::MoveOrdering {
                pos: HashMap::new(),
            },
            brother_from_to_move_ordering: search::BrotherMoveOrdering {
                pos: HashMap::new(),
            },
        };
        */

        let mut best_move = "resign".to_string();
        /*
        for depth in 1..10 {
            nega.max_depth = depth as f32;
            let value = search::NegaAlpha::search(&mut nega, pos, depth as f32, -30000, 30000);
            let end = nega.start_time.elapsed();
            let elapsed_time = end.as_secs() as i32 * 1000 + end.subsec_nanos() as i32 / 1_000_000;
            let nps = if elapsed_time != 0 {
                nega.num_searched * 1000 / elapsed_time as u64
            } else {
                nega.num_searched
            };

            if elapsed_time < nega.max_time {
                print!("info depth {} seldepth {} time {} nodes {} ", depth, nega.max_board_number - pos.ply(), elapsed_time, nega.num_searched);
                println!("score cp {} pv {} nps {}", value, nega.best_move_pv, nps);
                best_move = nega.best_move_pv.clone();
            } else {
                break;
            }
        }
        */

        return best_move;
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
    let engine = &mut BakuretsuKomasuteTaroR {
        engine_name: "爆裂駒捨太郎R".to_string(),
        author: "burokoron".to_string(),
        eval_file_path: "eval.json".to_string(),
        eval: Eval {
            pieces_in_board: vec![vec![vec![0; 31]; 81]; 2],
            pieces_in_hand: vec![vec![vec![0; 19]; 8]; 2],
        }
    };
    let mut pos = Position::default();

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
            "usi" => {  // エンジン名を返答
                BakuretsuKomasuteTaroR::usi(&engine);
            },
            "isready" => {  // 対局準備
                BakuretsuKomasuteTaroR::isready(engine);
            },
            "setoption" => {  // エンジンのパラメータ設定
                BakuretsuKomasuteTaroR::setoption(engine, inputs[2].clone(), inputs[4].clone());
            },
            "usinewgame" => {  // 新規対局準備
                BakuretsuKomasuteTaroR::usinewgame();
            },
            "position" => {  // 現局面の反映
                let mut moves: Vec<&str> = Vec::new();
                let startpos = {
                    if inputs[1] == "startpos" {
                        if inputs.len() > 3 {
                            for m in &inputs[3..] {
                                moves.push(m);
                            }
                        }
                        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".to_string()
                    } else {
                        if inputs.len() > 6 {
                            for m in &inputs[6..] {
                                moves.push(m);
                            }
                        }
                        format!("{} {} {} {}", inputs[2], inputs[3], inputs[4], inputs[5])
                    }
                };
                pos = BakuretsuKomasuteTaroR::position(&startpos, moves);
                println!("{pos}");
            },
            "go" => {  // 思考して指し手を返答
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
                } else {
                    if pos.side_to_move() == Color::Black {
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
                }
                max_time -= margin_time;
                min_time -= margin_time;
                let mut remain_move_number = (120 - pos.ply() as i32) / 4;
                if remain_move_number <= 1 {
                    remain_move_number = 1
                }
                max_time = std::cmp::max(max_time / remain_move_number, min_time);
                let m = BakuretsuKomasuteTaroR::go(engine.eval.clone(), &mut pos, max_time);
                println!("bestmove {}", m);
            }
            "stop" => {  // 思考停止コマンド
                BakuretsuKomasuteTaroR::stop();
            },
            "ponderhit" => {  // 先読みが当たった場合
                BakuretsuKomasuteTaroR::ponderhit();
            },
            "quit" => {  // 強制終了
                BakuretsuKomasuteTaroR::quit();
            },
            "gameover" => {  // 対局終了
                BakuretsuKomasuteTaroR::gameover();
            },
            _ => (),
        }
    }
}

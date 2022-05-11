use encoding::all::WINDOWS_31J;
use encoding::{EncoderTrap, Encoding};
use std::collections::HashMap;
use std::io::{stdout, Write};
use std::{collections::HashSet, time::Instant};
use yasai::{Color, File, Move, Piece, PieceType, Position, Rank, Square};

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
        for color in Color::ALL {
            for sq in Square::ALL {
                let value = eval["pieces_dict"][sq.index().to_string()]["0"]
                    .as_i64()
                    .unwrap();
                v.eval.pieces_in_board[color.index()][sq.index()][0] = value as i32;
                for piece in PieceType::ALL {
                    if color == Color::Black {
                        let value = eval["pieces_dict"][sq.index().to_string()]
                            [(piece.index() + 1).to_string()]
                        .as_i64()
                        .unwrap();
                        v.eval.pieces_in_board[color.index()][sq.index()][piece.index() + 1] =
                            value as i32;
                    } else {
                        let value = eval["pieces_dict"][sq.index().to_string()]
                            [(piece.index() + 17).to_string()]
                        .as_i64()
                        .unwrap();
                        v.eval.pieces_in_board[color.index()][sq.index()][piece.index() + 1] =
                            value as i32;
                    }
                }
            }
        }
        // 持ち駒
        for color in Color::ALL {
            for piece in PieceType::ALL_HAND {
                let piece_idx = {
                    if color == Color::Black {
                        piece.index()
                    } else {
                        piece.index() + 7
                    }
                };
                match piece.index() {
                    0 => {
                        for i in 0..19 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()]
                                [i.to_string()]
                            .as_i64()
                            .unwrap();
                            v.eval.pieces_in_hand[color.index()][piece.index()][i] = value as i32;
                        }
                    }
                    1 | 2 | 3 | 4 => {
                        for i in 0..5 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()]
                                [i.to_string()]
                            .as_i64()
                            .unwrap();
                            v.eval.pieces_in_hand[color.index()][piece.index()][i] = value as i32;
                        }
                    }
                    5 | 6 => {
                        for i in 0..3 {
                            let value = eval["pieces_in_hand_dict"][piece_idx.to_string()]
                                [i.to_string()]
                            .as_i64()
                            .unwrap();
                            v.eval.pieces_in_hand[color.index()][piece.index()][i] = value as i32;
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

        if name == "EvalFile" {
            v.eval_file_path = value;
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

        // sfenを分解
        let startpos: Vec<String> = {
            startpos
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect()
        };

        // sfenの盤面部分のエンコード
        let mut pieces: Vec<Option<Piece>> = Vec::new();
        let mut promote = false;
        for c in startpos[0].chars() {
            match c {
                'P' => {
                    if promote {
                        pieces.push(Some(Piece::BTO));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::BFU));
                    }
                }
                'L' => {
                    if promote {
                        pieces.push(Some(Piece::BNY));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::BKY));
                    }
                }
                'N' => {
                    if promote {
                        pieces.push(Some(Piece::BNK));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::BKE));
                    }
                }
                'S' => {
                    if promote {
                        pieces.push(Some(Piece::BNG));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::BGI));
                    }
                }
                'G' => {
                    if promote {
                        panic!("Gold is not promotable.");
                    } else {
                        pieces.push(Some(Piece::BKI));
                    }
                }
                'B' => {
                    if promote {
                        pieces.push(Some(Piece::BUM));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::BKA));
                    }
                }
                'R' => {
                    if promote {
                        pieces.push(Some(Piece::BRY));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::BHI));
                    }
                }
                'K' => {
                    if promote {
                        panic!("King is not promotable.");
                    } else {
                        pieces.push(Some(Piece::BOU));
                    }
                }
                'p' => {
                    if promote {
                        pieces.push(Some(Piece::WTO));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::WFU));
                    }
                }
                'l' => {
                    if promote {
                        pieces.push(Some(Piece::WNY));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::WKY));
                    }
                }
                'n' => {
                    if promote {
                        pieces.push(Some(Piece::WNK));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::WKE));
                    }
                }
                's' => {
                    if promote {
                        pieces.push(Some(Piece::WNG));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::WGI));
                    }
                }
                'g' => {
                    if promote {
                        panic!("Gold is not promotable.");
                    } else {
                        pieces.push(Some(Piece::WKI));
                    }
                }
                'b' => {
                    if promote {
                        pieces.push(Some(Piece::WUM));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::WKA));
                    }
                }
                'r' => {
                    if promote {
                        pieces.push(Some(Piece::WRY));
                        promote = false;
                    } else {
                        pieces.push(Some(Piece::WHI));
                    }
                }
                'k' => {
                    if promote {
                        panic!("King is not promotable.");
                    } else {
                        pieces.push(Some(Piece::WOU));
                    }
                }
                '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' => {
                    for _ in 0..(c as i32 - '0' as i32) {
                        pieces.push(None);
                    }
                }
                '+' => promote = true,
                '/' => (),
                _ => unreachable!(),
            }
        }
        assert_eq!(pieces.len(), 81, "Cannot encode sfen.");

        // エンコードした局面をbitboardの形式に変換
        let mut board: [Option<Piece>; 81] = [None; 81];
        for i in 0..pieces.len() {
            board[(8 - i % 9) * 9 + i / 9] = pieces[i];
        }

        // 手番をエンコード
        let side_to_move = {
            match &startpos[1][..] {
                "b" => Color::Black,
                "w" => Color::White,
                _ => unreachable!(),
            }
        };

        // 持ち駒をエンコード
        let mut hand_nums = [[0; 7]; 2];
        let mut side_to_move_idx = 0;
        let mut piece_type_idx = 0;
        let mut piece_nums = 1;
        for c in startpos[2].chars() {
            match c {
                '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '0' => {
                    if piece_nums == 1 {
                        piece_nums = c as u8 - b'0';
                    } else {
                        piece_nums *= 10;
                        piece_nums += c as u8 - b'0';
                    }
                }
                '-' => piece_nums = 0,
                _ => {
                    hand_nums[side_to_move_idx][piece_type_idx] = piece_nums;
                    piece_nums = 1;
                }
            }
            match c {
                'P' => {
                    side_to_move_idx = 0;
                    piece_type_idx = 0;
                }
                'L' => {
                    side_to_move_idx = 0;
                    piece_type_idx = 1;
                }
                'N' => {
                    side_to_move_idx = 0;
                    piece_type_idx = 2;
                }
                'S' => {
                    side_to_move_idx = 0;
                    piece_type_idx = 3;
                }
                'G' => {
                    side_to_move_idx = 0;
                    piece_type_idx = 4;
                }
                'B' => {
                    side_to_move_idx = 0;
                    piece_type_idx = 5;
                }
                'R' => {
                    side_to_move_idx = 0;
                    piece_type_idx = 6;
                }
                'p' => {
                    side_to_move_idx = 1;
                    piece_type_idx = 0;
                }
                'l' => {
                    side_to_move_idx = 1;
                    piece_type_idx = 1;
                }
                'n' => {
                    side_to_move_idx = 1;
                    piece_type_idx = 2;
                }
                's' => {
                    side_to_move_idx = 1;
                    piece_type_idx = 3;
                }
                'g' => {
                    side_to_move_idx = 1;
                    piece_type_idx = 4;
                }
                'b' => {
                    side_to_move_idx = 1;
                    piece_type_idx = 5;
                }
                'r' => {
                    side_to_move_idx = 1;
                    piece_type_idx = 6;
                }
                '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '0' | '-' => (),
                _ => unreachable!(),
            }
        }
        hand_nums[side_to_move_idx][piece_type_idx] = piece_nums;

        // 手数のエンコード
        let ply: u32 = startpos[3].parse().unwrap();

        // bitboardに局面を反映
        let mut position_history = HashSet::new();
        let mut pos = Position::new(board, hand_nums, side_to_move, ply);
        position_history.insert(pos.key());

        // 指し手を進める
        for m in moves {
            let m: Vec<char> = m.chars().collect();
            let m = {
                if m[1] == '*' {
                    let to = Square::new(
                        File::ALL[m[2] as usize - '1' as usize],
                        Rank::ALL[m[3] as usize - 'a' as usize],
                    );
                    let piece = {
                        if pos.side_to_move() == Color::Black {
                            match m[0] {
                                'P' => Piece::BFU,
                                'L' => Piece::BKY,
                                'N' => Piece::BKE,
                                'S' => Piece::BGI,
                                'G' => Piece::BKI,
                                'B' => Piece::BKA,
                                'R' => Piece::BHI,
                                _ => unreachable!(),
                            }
                        } else {
                            match m[0] {
                                'P' => Piece::WFU,
                                'L' => Piece::WKY,
                                'N' => Piece::WKE,
                                'S' => Piece::WGI,
                                'G' => Piece::WKI,
                                'B' => Piece::WKA,
                                'R' => Piece::WHI,
                                _ => unreachable!(),
                            }
                        }
                    };
                    Move::new_drop(to, piece)
                } else {
                    let from = Square::new(
                        File::ALL[m[0] as usize - '1' as usize],
                        Rank::ALL[m[1] as usize - 'a' as usize],
                    );
                    let to = Square::new(
                        File::ALL[m[2] as usize - '1' as usize],
                        Rank::ALL[m[3] as usize - 'a' as usize],
                    );
                    let is_promotion = m.len() == 5;
                    let piece = pos.piece_on(from).unwrap();
                    Move::new_normal(from, to, is_promotion, piece)
                }
            };
            pos.do_move(m);
            position_history.insert(pos.key());
        }
        position_history.remove(&pos.key());

        (pos, position_history)
    }

    fn go(
        eval: Eval,
        pos: &mut Position,
        position_history: &mut HashSet<u64>,
        max_time: i32,
    ) -> String {
        /*
        思考し、最善手を返す
        */

        let mut nega = search::NegaAlpha {
            start_time: Instant::now(),
            max_time,
            num_searched: 0,
            max_depth: 1,
            max_board_number: pos.ply(),
            best_move_pv: None,
            eval,
            hash_table: search::HashTable {
                pos: HashMap::new(),
            },
            brother_to_move_ordering: search::BrotherMoveOrdering {
                pos: HashMap::new(),
            },
        };

        let mut best_move = "resign".to_string();
        for depth in 1..10 {
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
        engine_name: "爆裂転生太郎".to_string(),
        author: "burokoron".to_string(),
        eval_file_path: "eval.json".to_string(),
        eval: Eval {
            pieces_in_board: vec![vec![vec![0; 31]; 81]; 2],
            pieces_in_hand: vec![vec![vec![0; 19]; 8]; 2],
        },
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
                        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
                            .to_string()
                    } else {
                        if inputs.len() > 7 {
                            for m in &inputs[7..] {
                                moves.push(m);
                            }
                        }
                        format!("{} {} {} {}", inputs[2], inputs[3], inputs[4], inputs[5])
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
                let m = BakuretsuTenseiTaro::go(
                    engine.eval.clone(),
                    &mut pos,
                    &mut position_history,
                    max_time,
                );
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

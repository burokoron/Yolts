use std::time::Instant;
use shogi::{Move, Position};
use shogi::bitboard::Factory as BBFactory;

mod search;

struct BakuretsuKomasuteTaroR {
    engine_name: String,
    author: String,
    eval_file_path: String,
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

    fn isready() {
        /*
        対局の準備をする
        */
        
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

        let mut pos = Position::new();
        pos.set_sfen(startpos).unwrap();
        for m in moves {
            pos.make_move(Move::from_sfen(m).unwrap()).unwrap();
        }

        return pos;
    }

    fn go(pos: &mut Position) -> String {
        /*
        思考し、最善手を返す
        */

        let mut nega = search::NegaAlpha {
            num_searched: 0,
            max_depth: 3.,
            best_move_pv: "resign".to_string(),
        };

        let start = Instant::now();
        let value = search::NegaAlpha::search(&mut nega, pos, 0., -30000, 30000);
        let end = start.elapsed();
        let elapsed_time = end.as_secs() as f64 + end.subsec_nanos() as f64 / 1_000_000_000.;
        let nps = nega.num_searched as f64 / elapsed_time;
        println!("info depth {} time {} nodes {} score cp {} nps {}", nega.max_depth, elapsed_time, nega.num_searched, value, nps as u64);

        return nega.best_move_pv;
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
        eval_file_path: "eval.pkl".to_string(),
    };
    BBFactory::init();
    let mut pos = Position::new();

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
                BakuretsuKomasuteTaroR::isready();
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
            },
            "go" => {  // 思考して指し手を返答
                let m = BakuretsuKomasuteTaroR::go(&mut pos);
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

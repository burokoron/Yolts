use shogi::{Move, PieceType, Position, Square};
use shogi::piece::Piece;
use shogi::bitboard::Factory as BBFactory;

struct BakuretsuKomasuteTaroR {
    engine_name: String,
    version: String,
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

        println!("id name {} {}", v.engine_name, v.version);
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

        let mut moves: Vec<Move> = Vec::new();
        for sq in Square::iter() {
            let pc = pos.piece_at(sq);
            if let Some(ref pc) = *pc {
                if pos.side_to_move() == pc.color {
                    let mut bb = pos.move_candidates(sq, *pc);
                    while bb.is_any() {
                        let to = bb.pop();
                        let m = Move::Normal { from: sq, to: to, promote: false };
                        if pos.make_move(m).is_ok() {
                            moves.push(m);
                            pos.unmake_move().unwrap();
                        }
                        let m = Move::Normal { from: sq, to: to, promote: true };
                        if pos.make_move(m).is_ok() {
                            moves.push(m);
                            pos.unmake_move().unwrap();
                        }
                    }
                }
            } else {
                for piece_type in PieceType::iter() {
                    let color = pos.side_to_move();
                    if pos.hand(Piece { piece_type, color }) > 0 {
                        let m = Move::Drop { to: sq, piece_type: piece_type };
                        if pos.make_move(m).is_ok() {
                            moves.push(m);
                            pos.unmake_move().unwrap();
                        }
                    }
                }
            }
        }

        if moves.len() != 0 {
            return moves[0].to_string();
        } else {
            return "resign".to_string();
        }
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
        version: "Version 0.0.1".to_string(),
        author: "burokoron".to_string(),
        eval_file_path: "BakuretsuKomasuteTaroR/eval.pkl".to_string(),
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

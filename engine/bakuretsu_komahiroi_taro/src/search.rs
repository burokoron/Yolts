use arrayvec::ArrayVec;
use shogi_core::{Color, Move, Piece, PieceKind, Square};
use std::collections::HashSet;
use yasai::Position;

use crate::evaluate::Evaluate;
use crate::evaluate::VALUE_SCALE;

pub const MATING_VALUE: i32 = 30000;
const FUTILITY_MARGIN: i32 = 109 * 2;
const FUTILITY_RETURN_DEPTH: u32 = 14;
const NULL_MOVE_MARGIN1: i32 = 21;
const NULL_MOVE_MARGIN2: i32 = 421;
const NULL_MOVE_DYNAMIC_GAMMA: i32 = 235;

const KING_POSITION_SCALE: i32 = 99;
const NON_KING_POSITION_SCALE: i32 = 1;

#[derive(Clone)]
pub struct HashTableValue {
    pub key: u64,
    pub depth: u32,
    pub upper: i32,
    pub lower: i32,
    pub best_move: Option<Move>,
    pub generation: u32,
}

pub struct MoveOrdering {
    pub piece_to_history: Vec<Vec<Vec<i64>>>,
    pub killer_heuristic: Vec<Vec<Option<Move>>>,
    pub counter_move: Vec<Vec<Vec<Option<Move>>>>,
}

pub struct NegaAlpha {
    pub my_turn: Color,
    pub start_time: std::time::Instant,
    pub max_time: i32,
    pub num_searched: u64,
    pub max_depth: u32,
    pub max_board_number: u16,
    pub best_move_pv: Option<Move>,
    pub eval: Evaluate,
    pub search_mode: String,
    pub hash_table: Vec<HashTableValue>,
    pub hash_table_generation: u32,
    pub move_ordering: MoveOrdering,
    pub position_history: HashSet<u64>,
    pub position_value: Vec<(f32, [f32; 4])>,
}

impl NegaAlpha {
    /// 現在の局面に対する評価値の差分を計算する。
    ///
    /// 指定された局面が「入玉宣言」による即勝利条件を満たす場合は、手番に依らず `MATING_VALUE - ply` を返す。
    /// また、探索モードが "Priority-27-Point" または "Absolute-27-Point" の場合は、玉とそれ以外の駒に対する縦方向の位置に基づく補正を適用する。最終的な評価値は先手側が正、後手側が負となるように符号を調整して返す。
    ///
    /// # Parameters
    ///
    /// - `pos` – 評価対象の局面（`Position` の可変参照）。関数は局面情報を参照して評価を算出する。
    ///
    /// # Returns
    ///
    /// `i32` 評価値。値が大きいほど先手側に有利であり、値が小さいほど後手側に有利である。入玉宣言成立時は `MATING_VALUE - ply` を返す。
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// // `na` は NegaAlpha のインスタンス、`pos` は評価する Position
    /// // let mut na = NegaAlpha::new(...);
    /// // let mut pos = Position::from_sfen("...").unwrap();
    /// // let val = na.evaluate_diff(&mut pos);
    /// ```
    fn evaluate_diff(&mut self, pos: &mut Position) -> i32 {
        //! 局面の差分評価
        //!
        //! - Arguments
        //!   - pos: &mut Position
        //!     - 評価する局面
        //! - Returns
        //!   - value: i32
        //!     - 評価値

        // 入玉宣言の確認
        if is_nyugyoku_win(pos) {
            return MATING_VALUE - pos.ply() as i32;
        }

        // 局面評価回数+1
        self.num_searched += 1;

        // 通常の評価
        let mut value = if let Some(value) = self.position_value.last() {
            (value.0 * VALUE_SCALE) as i32
        } else {
            panic!("Empty position_value.")
        };

        // 入玉宣言用の補正評価
        if self.search_mode == "Priority-27-Point" || self.search_mode == "Absolute-27-Point" {
            for sq in Square::all() {
                let pc = pos.piece_at(sq);
                if let Some(pc) = pc {
                    match pc {
                        Piece::B_P
                        | Piece::B_L
                        | Piece::B_N
                        | Piece::B_S
                        | Piece::B_G
                        | Piece::B_PP
                        | Piece::B_PL
                        | Piece::B_PN
                        | Piece::B_PS
                        | Piece::B_B
                        | Piece::B_R
                        | Piece::B_PB
                        | Piece::B_PR => {
                            value += (10 - sq.rank() as i32) * NON_KING_POSITION_SCALE;
                        }
                        Piece::B_K => {
                            value += (10 - sq.rank() as i32) * KING_POSITION_SCALE;
                        }
                        Piece::W_P
                        | Piece::W_L
                        | Piece::W_N
                        | Piece::W_S
                        | Piece::W_G
                        | Piece::W_PP
                        | Piece::W_PL
                        | Piece::W_PN
                        | Piece::W_PS
                        | Piece::W_B
                        | Piece::W_R
                        | Piece::W_PB
                        | Piece::W_PR => {
                            value -= sq.rank() as i32 * NON_KING_POSITION_SCALE;
                        }
                        Piece::W_K => {
                            value -= sq.rank() as i32 * KING_POSITION_SCALE;
                        }
                        _ => unreachable!("Unrecognizable piece types."),
                    }
                }
            }
        }

        // 後手番なら評価値反転
        if pos.side_to_move() == Color::Black {
            value
        } else {
            -value
        }
    }

    /// 駒の取り返しのみを探索して評価値を返す。
    ///
    /// 現在の局面から直前の着手先に関連する取り返し（捕獲）だけを展開し、アルファ・ベータ窓で枝刈りを行う。探索中に時間制限を超えた場合は現状のアルファ値を返す。合法手が存在しない場合は詰み扱いの評価を返し、検索モードが "Absolute-27-Point" のときは手番（my_turn）に応じて符号と手数を考慮した値を返す。ムーブオーダリングには MVV-LVA スタイルの重み付けを用い、王手でない場合は最優先候補のみを試すことで部分的に枝刈りする。
    ///
    /// # 詳細
    /// - 探索ノード数と最大深さ情報（max_board_number）を更新する。
    /// - 静的評価差分（evaluate_diff）を基準にアルファを更新し、ベータカットをチェックする。
    /// - 合法手のうち、着手が捕獲でありかつ王手か直前着手先への取り返しである手のみを候補にする。非王手時は候補のうち最も評価の高い一手のみを試行する。
    /// - 各手を適用する前に評価推論（inference_diff）を積み、着手後に再帰的に negamax 形式で探索してから巻き戻す。
    ///
    /// # Returns
    /// 現局面に対する評価値（i32）。時間切れ時は与えられたアルファ値、局面が終局なら詰み評価を返す。
    fn recapture_search(
        &mut self,
        pos: &mut Position,
        previous_move_to: Square,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        //! 駒の取り返しのみの探索
        //!
        //! - Arguments
        //!  - pos: &mut Position
        //!    - 探索する局面
        //! - previous_move_to: Square
        //!  - 直前の指し手の着手先
        //! - mut alpha: i32
        //!   - アルファ値
        //! - beta: i32
        //!  - ベータ値
        //! - Returns
        //!  - value: i32
        //!   - 評価値

        // 最大手数の計算
        if self.max_board_number < pos.ply() {
            self.max_board_number = pos.ply();
        }

        // 時間制限なら
        let end = self.start_time.elapsed();
        let elapsed_time = end.as_secs() as i32 * 1000 + end.subsec_nanos() as i32 / 1_000_000;
        if elapsed_time >= self.max_time {
            return alpha;
        }

        let value = self.evaluate_diff(pos);

        if alpha < value {
            alpha = value;
        }
        if beta <= alpha {
            return alpha;
        }

        // 全合法手検索
        let legal_moves = pos.legal_moves();
        // 合法手なしなら
        if legal_moves.is_empty() {
            if self.search_mode == "Absolute-27-Point" {
                if pos.side_to_move() == self.my_turn {
                    return -MATING_VALUE + pos.ply() as i32;
                } else {
                    return MATING_VALUE - pos.ply() as i32;
                }
            } else {
                return -MATING_VALUE + pos.ply() as i32;
            }
        }

        // 王手がかかっているかどうか
        let is_check = pos.in_check();

        // ムーブオーダリング
        let mut move_list = ArrayVec::<(Move, i64), 593>::new();
        // ムーブオーダリング用の重み計算
        for m in legal_moves {
            let mut value = 0;
            if let Move::Normal {
                from,
                to,
                promote: _,
            } = m
            {
                // MVV-LVA
                if let Some(p) = pos.piece_at(to) {
                    let mut idx = p.piece_kind().array_index();
                    if idx > 8 {
                        idx -= 8;
                    }
                    value += idx as i64 * 10;

                    if let Some(p) = pos.piece_at(from) {
                        let mut idx = p.piece_kind().array_index();
                        if idx > 8 {
                            idx -= 8;
                        }
                        value += 9 - idx as i64;
                    }
                }
            }
            if let Move::Normal {
                from: _,
                to,
                promote: _,
            } = m
            {
                if is_check || previous_move_to == to {
                    move_list.push((m, value));
                }
            }
        }
        move_list.sort_by(|&i, &j| (-i.1).cmp(&(-j.1)));
        if !is_check {
            if let Some(first) = move_list.first().cloned() {
                move_list.clear();
                move_list.push(first);
            }
        }

        for m in move_list {
            self.position_value.push(self.eval.inference_diff(
                pos,
                Some(m.0),
                self.position_value.last().copied(),
            ));
            pos.do_move(m.0);
            let value = -self.recapture_search(pos, m.0.to(), -beta, -alpha);
            pos.undo_move(m.0);
            self.position_value.pop();

            if alpha < value {
                alpha = value;
            }
            if beta <= alpha {
                return alpha;
            }
        }

        alpha
    }
    /// 静的評価と捕獲系手のみを探索してアルファ–ベータ境界を更新する静止探索を実行する。
    ///
    /// 指定深さまで捕獲や王手を中心に探索し、静的評価値に基づいてアルファ値を更新して打ち切りを行う。
    /// タイムアウト時は現在のアルファを返し、合法手がない場合は詰み評価（モードにより符号が異なる）を返す。
    ///
    /// # Returns
    ///
    /// `i32` 探索後の評価値（アルファ境界に基づく上限または更新された評価）。
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // 概念例（実際の初期化はエンジン固有）
    /// let mut engine: NegaAlpha = /* 初期化 */;
    /// let mut pos: Position = /* 局面初期化 */;
    /// let value = engine.quiescence_search(&mut pos, 2, -1000, 1000);
    /// println!("静止探索の評価値: {}", value);
    /// ```
    fn quiescence_search(
        &mut self,
        pos: &mut Position,
        depth: u32,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        //! 静止探索
        //!
        //! - Arguments
        //!   - pos: &mut Position
        //!     - 静止探索する局面
        //!   - depth: u32
        //!     - 残り探索深さ
        //!   - mut alpha: i32
        //!     - アルファ値
        //!   - beta: i32
        //!     - ベータ値
        //! - Returns
        //!   - value: i32
        //!     - 評価値

        // 最大手数の計算
        if self.max_board_number < pos.ply() {
            self.max_board_number = pos.ply();
        }

        // 時間制限なら
        let end = self.start_time.elapsed();
        let elapsed_time = end.as_secs() as i32 * 1000 + end.subsec_nanos() as i32 / 1_000_000;
        if elapsed_time >= self.max_time {
            return alpha;
        }

        let value = self.evaluate_diff(pos);

        if alpha < value {
            alpha = value;
        }
        if beta <= alpha {
            return alpha;
        }

        // 全合法手検索
        let legal_moves = pos.legal_moves();
        // 合法手なしなら
        if legal_moves.is_empty() {
            if self.search_mode == "Absolute-27-Point" {
                if pos.side_to_move() == self.my_turn {
                    return -MATING_VALUE + pos.ply() as i32;
                } else {
                    return MATING_VALUE - pos.ply() as i32;
                }
            } else {
                return -MATING_VALUE + pos.ply() as i32;
            }
        }

        // 王手がかかっているかどうか
        let is_check = pos.in_check();

        // ムーブオーダリング
        let mut move_list = ArrayVec::<(Move, i64), 593>::new();
        // ムーブオーダリング用の重み計算
        for m in legal_moves {
            let mut value = 0;
            if let Move::Normal {
                from,
                to,
                promote: _,
            } = m
            {
                // MVV-LVA
                if let Some(p) = pos.piece_at(to) {
                    let mut idx = p.piece_kind().array_index();
                    if idx > 8 {
                        idx -= 8;
                    }
                    value += idx as i64 * 10;

                    if let Some(p) = pos.piece_at(from) {
                        let mut idx = p.piece_kind().array_index();
                        if idx > 8 {
                            idx -= 8;
                        }
                        value += 9 - idx as i64;
                    }
                }
            }
            if is_check || pos.is_check_move(m) || value > 0 {
                move_list.push((m, value));
            }
        }
        move_list.sort_by(|&i, &j| (-i.1).cmp(&(-j.1)));

        for m in move_list {
            self.position_value.push(self.eval.inference_diff(
                pos,
                Some(m.0),
                self.position_value.last().copied(),
            ));
            pos.do_move(m.0);
            let value = if depth - 1 != 0 {
                -self.quiescence_search(pos, depth - 1, -beta, -alpha)
            } else {
                -self.recapture_search(pos, m.0.to(), -beta, -alpha)
            };
            pos.undo_move(m.0);
            self.position_value.pop();

            if alpha < value {
                alpha = value;
            }
            if beta <= alpha {
                return alpha;
            }
        }

        alpha
    }

    /// ネガアルファ探索によって局面の評価値を返す。
    ///
    /// 深さ制御、置換表、メイト距離剪定、レイザリング、徒労枝刈り、ヌルムーブ剪定、LMR、キラームーブや
    /// カウンタームーブ等のムーブオーダリングを含む主探索ルーチンで、探索結果に応じた評価値（大きいほど先手側に有利）を返します。
    ///
    /// # パラメータ
    /// - `in_null_move` — 直前にヌルムーブを行った状態かどうか（ヌルムーブ連続適用を防ぐために使用）。
    /// - `previous_move` — 直前に指された手の (駒, 到点) を与えることでカウンタームーブ等の履歴ヒューリスティクスに利用されます。未指定の場合は `None`。
    ///
    /// # 戻り値
    /// - 探索された局面の評価値（`i32`）。正の値はメソッド実装で定義された「先手側に有利」を示します。メイトや引き分け等の特殊値は定数（例: `MATING_VALUE`）に基づき表現されます。
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # // 実際のエンジン状態構築は省略
    /// # use engine::search::NegaAlpha;
    /// # use yasai::Position;
    /// let mut engine: NegaAlpha = /* 初期化 */;
    /// let mut pos: Position = /* 局面 */;
    /// let score = engine.search(&mut pos, false, 6, -30000, 30000, None);
    /// println!("score = {}", score);
    /// ```
    pub fn search(
        &mut self,
        pos: &mut Position,
        in_null_move: bool,
        depth: u32,
        mut alpha: i32,
        mut beta: i32,
        previous_move: Option<(Piece, Square)>,
    ) -> i32 {
        //! ネガアルファ探索
        //!
        //! - Arguments
        //!   - pos: &mut Position
        //!     - 探索する局面
        //!   - in_null_move: bool
        //!     - null moveした直後かどうか
        //!   - depth: u32
        //!     - 残り探索深さ
        //!   - mut alpha: i32
        //!     - アルファ値
        //!   - mut beta: i32
        //!     - ベータ値
        //!  - previous_move: Option<(Piece, Square)>
        //!     - 直前の指し手
        //! - Returns
        //!   - value: i32
        //!     - 評価値

        // 最大手数の計算
        if self.max_board_number < pos.ply() {
            self.max_board_number = pos.ply();
        }

        // 時間制限なら
        let end = self.start_time.elapsed();
        let elapsed_time = end.as_secs() as i32 * 1000 + end.subsec_nanos() as i32 / 1_000_000;
        if elapsed_time >= self.max_time {
            return alpha;
        }

        // 王手がかかっているかどうか
        let in_check = pos.in_check();

        // 同一局面の確認
        if self.position_history.contains(&pos.key()) {
            if in_check {
                return MATING_VALUE - pos.ply() as i32;
            } else {
                return 0;
            }
        }

        // 置換表の確認
        let hash_table_index = pos.key() as usize % self.hash_table.len();
        let mut best_move = if self.hash_table[hash_table_index].key == pos.key() {
            if self.hash_table_generation == self.hash_table[hash_table_index].generation
                && depth <= self.hash_table[hash_table_index].depth
            {
                let lower = self.hash_table[hash_table_index].lower;
                let upper = self.hash_table[hash_table_index].upper;
                if lower >= beta {
                    return lower;
                }
                if upper <= alpha || upper == lower {
                    return upper;
                }
                alpha = alpha.max(lower);
                beta = beta.min(upper);
            }
            self.hash_table[hash_table_index].best_move
        } else {
            None
        };

        // 探索深さ制限なら
        if depth == 0 {
            return self.quiescence_search(pos, 2, alpha, beta);
        }

        // Mate Distance Pruning
        let mating_value = -MATING_VALUE + pos.ply() as i32;
        if mating_value > alpha {
            alpha = mating_value;
            if beta <= mating_value {
                return mating_value;
            }
        }
        let mating_value = MATING_VALUE - pos.ply() as i32;
        if mating_value < beta {
            beta = mating_value;
            if alpha >= mating_value {
                return mating_value;
            }
        }

        let static_value = self.evaluate_diff(pos);

        // Razoring
        if !in_check && static_value < alpha - 469 - 307 * depth as i32 * depth as i32 {
            let value = self.quiescence_search(pos, 3, alpha - 1, alpha);
            if value < alpha && value.abs() < MATING_VALUE - 1000 {
                return value;
            }
        }

        // Futility Pruning
        if !in_check
            && best_move.is_none()
            && depth < FUTILITY_RETURN_DEPTH
            && static_value - FUTILITY_MARGIN * depth as i32 >= beta
            && beta > -MATING_VALUE + 1000
            && static_value < MATING_VALUE - 1000
        {
            return beta + (static_value - beta) / 3;
        }

        // Null Move Pruning
        if depth != self.max_depth
            && depth >= 2
            && static_value >= beta - NULL_MOVE_MARGIN1 * depth as i32 + NULL_MOVE_MARGIN2
            && beta.abs() < MATING_VALUE - 1000
            && !in_check
            && !in_null_move
        {
            let reduce =
                ((static_value - beta) / NULL_MOVE_DYNAMIC_GAMMA).min(7) as u32 + depth / 3 + 5;
            pos.do_null_move();
            let value = -self.search(pos, true, depth - reduce.min(depth), -beta, -beta + 1, None);
            pos.undo_null_move();
            if value >= beta {
                return beta;
            }
        }

        // 全合法手検索
        let legal_moves = pos.legal_moves();
        // 合法手なしなら
        if legal_moves.is_empty() {
            if self.search_mode == "Absolute-27-Point" {
                if pos.side_to_move() == self.my_turn {
                    return -MATING_VALUE + pos.ply() as i32;
                } else {
                    return MATING_VALUE - pos.ply() as i32;
                }
            } else {
                return -MATING_VALUE + pos.ply() as i32;
            }
        }

        // ムーブオーダリング
        let mut best_value = alpha;
        let mut move_list = ArrayVec::<(Move, i64), 593>::new();
        // ムーブオーダリング用の重み計算
        for m in legal_moves {
            let mut value = 0;
            // 置換表にある手
            if let Some(best_move) = best_move {
                if best_move == m {
                    value += 100000;
                }
            }
            // Killer Heuristic
            if let Some(killer_move) =
                self.move_ordering.killer_heuristic[(self.max_depth - depth) as usize][0]
            {
                if killer_move == m {
                    value += 10000;
                }
            }
            if let Some(killer_move) =
                self.move_ordering.killer_heuristic[(self.max_depth - depth) as usize][1]
            {
                if killer_move == m {
                    value += 9000;
                }
            }
            // Counter Move
            if let Some(previous_move) = previous_move {
                let piece = previous_move.0;
                let to = previous_move.1.array_index();
                if let Some(counter_move) =
                    self.move_ordering.counter_move[piece.piece_kind().array_index()][to][0]
                {
                    if counter_move == m {
                        value += 1000;
                    }
                }
                if let Some(counter_move) =
                    self.move_ordering.counter_move[piece.piece_kind().array_index()][to][1]
                {
                    if counter_move == m {
                        value += 900;
                    }
                }
            }
            if let Move::Normal {
                from,
                to,
                promote: _,
            } = m
            {
                // MVV-LVA
                if let Some(p) = pos.piece_at(to) {
                    let mut idx = p.piece_kind().array_index();
                    if idx > 8 {
                        idx -= 8;
                    }
                    value += idx as i64 * 10;

                    if let Some(p) = pos.piece_at(from) {
                        let mut idx = p.piece_kind().array_index();
                        if idx > 8 {
                            idx -= 8;
                        }
                        value += 9 - idx as i64;
                    }
                }
            }
            // Piece To History
            let turn = pos.side_to_move().array_index();
            let piece = match m {
                Move::Normal {
                    from,
                    to: _,
                    promote: _,
                } => pos.piece_at(from).unwrap(),
                Move::Drop { piece, to: _ } => piece,
            };
            let to = m.to().array_index();
            value +=
                self.move_ordering.piece_to_history[turn][piece.piece_kind().array_index()][to];
            move_list.push((m, value));
        }
        move_list.sort_by(|&i, &j| (-i.1).cmp(&(-j.1)));

        // 全合法手展開
        let mut is_lmr = depth >= 2 && depth != self.max_depth;
        self.position_history.insert(pos.key());
        for (move_count, m) in move_list.iter().enumerate() {
            self.position_value.push(self.eval.inference_diff(
                pos,
                Some(m.0),
                self.position_value.last().copied(),
            ));
            let piece = match m.0 {
                Move::Normal {
                    from,
                    to: _,
                    promote: _,
                } => pos.piece_at(from).unwrap(),
                Move::Drop { piece, to: _ } => piece,
            };
            pos.do_move(m.0);
            // Late Move Reductions
            if is_lmr && move_count >= 11 && !pos.in_check() {
                let value = -self.search(
                    pos,
                    false,
                    depth - 2,
                    -beta,
                    -best_value,
                    Some((piece, m.0.to())),
                );
                if value <= best_value {
                    pos.undo_move(m.0);
                    self.position_value.pop();
                    continue;
                }
            }
            // 通常の探索
            let value = -self.search(
                pos,
                false,
                depth - 1,
                -beta,
                -best_value,
                Some((piece, m.0.to())),
            );
            if best_value < value {
                best_value = value;
                best_move = Some(m.0);
                if is_lmr && move_count < 11 {
                    is_lmr = false;
                }
                if depth == self.max_depth {
                    self.best_move_pv = Some(m.0);
                }
            }
            pos.undo_move(m.0);
            self.position_value.pop();
            if best_value >= beta {
                // Killer Heuristic
                if self.move_ordering.killer_heuristic[(self.max_depth - depth) as usize][0]
                    .is_none()
                {
                    self.move_ordering.killer_heuristic[(self.max_depth - depth) as usize][0] =
                        Some(m.0);
                } else if let Some(first_killer) =
                    self.move_ordering.killer_heuristic[(self.max_depth - depth) as usize][0]
                {
                    if first_killer != m.0 {
                        self.move_ordering.killer_heuristic[(self.max_depth - depth) as usize][1] =
                            Some(m.0);
                        self.move_ordering.killer_heuristic[(self.max_depth - depth) as usize]
                            .swap(0, 1);
                    }
                }
                // Piece To History
                let turn = pos.side_to_move().array_index();
                let piece = match m.0 {
                    Move::Normal {
                        from,
                        to: _,
                        promote: _,
                    } => pos.piece_at(from).unwrap(),
                    Move::Drop { piece, to: _ } => piece,
                };
                let to = m.0.to().array_index();
                self.move_ordering.piece_to_history[turn][piece.piece_kind().array_index()][to] +=
                    depth as i64 * depth as i64;
                // Counter Move
                if let Some(previous_move) = previous_move {
                    let piece = previous_move.0;
                    let to = previous_move.1.array_index();
                    if self.move_ordering.counter_move[piece.piece_kind().array_index()][to][0]
                        .is_none()
                    {
                        self.move_ordering.counter_move[piece.piece_kind().array_index()][to][0] =
                            Some(m.0);
                    } else if let Some(first_counter) =
                        self.move_ordering.counter_move[piece.piece_kind().array_index()][to][0]
                    {
                        if first_counter != m.0 {
                            self.move_ordering.counter_move[piece.piece_kind().array_index()][to]
                                [1] = Some(m.0);
                            self.move_ordering.counter_move[piece.piece_kind().array_index()][to]
                                .swap(0, 1);
                        }
                    }
                }
                break;
            }
        }
        self.position_history.remove(&pos.key());

        // 置換表へ登録
        if self.hash_table_generation != self.hash_table[hash_table_index].generation
            || depth > self.hash_table[hash_table_index].depth
        {
            self.hash_table[hash_table_index].key = pos.key();
            self.hash_table[hash_table_index].depth = depth;
            self.hash_table[hash_table_index].generation = self.hash_table_generation;
            if best_value <= alpha {
                self.hash_table[hash_table_index].upper = best_value;
                self.hash_table[hash_table_index].lower = -MATING_VALUE;
            } else if best_value >= beta {
                self.hash_table[hash_table_index].upper = MATING_VALUE;
                self.hash_table[hash_table_index].lower = best_value;
            } else {
                self.hash_table[hash_table_index].upper = best_value;
                self.hash_table[hash_table_index].lower = best_value;
            }
            self.hash_table[hash_table_index].best_move = best_move;
        }

        best_value
    }

    pub fn pv_to_sfen(
        &mut self,
        pos: &mut Position,
        position_history: &mut HashSet<u64>,
    ) -> String {
        //! 最善手順を置換表から集める
        //!
        //! - Arguments
        //!   - pos: &mut Position
        //!     - 最善手順を調べる局面
        //!   - position_history: &mut HashSet<u64>
        //!     - 局面の履歴
        //! - Returns
        //!   - pv: String
        //!     - 最善手順

        let mut pv = "".to_string();
        let mut moves = Vec::new();

        loop {
            if position_history.contains(&pos.key()) {
                break;
            }
            let hash_table_index = pos.key() as usize % self.hash_table.len();
            let best_move = if self.hash_table[hash_table_index].key == pos.key()
                && self.hash_table_generation == self.hash_table[hash_table_index].generation
            {
                self.hash_table[hash_table_index].best_move
            } else {
                None
            };
            if let Some(best_move) = best_move {
                let legal_moves = pos.legal_moves();
                let mut is_legal = false;
                for m in legal_moves {
                    if m == best_move {
                        is_legal = true;
                        break;
                    }
                }
                if !is_legal {
                    break;
                }
                pv += &move_to_sfen(best_move);
                pv += " ";
                position_history.insert(pos.key());
                pos.do_move(best_move);
                moves.push(best_move);
            } else {
                break;
            }
        }

        loop {
            let m = moves.pop();
            if let Some(m) = m {
                pos.undo_move(m);
                position_history.remove(&pos.key());
            } else {
                break;
            }
        }

        pv
    }
}

pub fn is_nyugyoku_win(pos: &Position) -> bool {
    //! 入玉宣言(27点法)の確認
    //!
    //! - Arguments
    //!   - pos: &Position
    //!     - 判定する局面
    //! - Returns
    //!   - : bool
    //!     - 判定結果

    if pos.side_to_move() == Color::Black && !pos.in_check() {
        let sq = pos.king_position(Color::Black);
        if let Some(ref sq) = sq {
            if sq.rank() <= 3 {
                let mut value = 0;
                let mut count = 0;
                for sq in Square::all() {
                    if sq.rank() <= 3 {
                        let pc = pos.piece_at(sq);
                        if let Some(pc) = pc {
                            match pc {
                                Piece::B_P
                                | Piece::B_L
                                | Piece::B_N
                                | Piece::B_S
                                | Piece::B_G
                                | Piece::B_PP
                                | Piece::B_PL
                                | Piece::B_PN
                                | Piece::B_PS => {
                                    value += 1;
                                    count += 1;
                                }
                                Piece::B_B | Piece::B_R | Piece::B_PB | Piece::B_PR => {
                                    value += 5;
                                    count += 1;
                                }
                                _ => (),
                            }
                        }
                    }
                }
                if count >= 10 {
                    let hand = pos.hand(Color::Black);
                    for piece_type in PieceKind::all() {
                        if piece_type == PieceKind::King {
                            break;
                        }
                        match piece_type {
                            PieceKind::Pawn
                            | PieceKind::Lance
                            | PieceKind::Knight
                            | PieceKind::Silver
                            | PieceKind::Gold => value += hand.Hand_count(piece_type),
                            PieceKind::Bishop | PieceKind::Rook => {
                                value += hand.Hand_count(piece_type) * 5
                            }
                            _ => (),
                        }
                    }
                }
                if value >= 28 {
                    return true;
                }
            }
        }
    } else if pos.side_to_move() == Color::White && !pos.in_check() {
        let sq = pos.king_position(Color::White);
        if let Some(ref sq) = sq {
            if sq.rank() >= 7 {
                let mut value = 0;
                let mut count = 0;
                for sq in Square::all() {
                    if sq.rank() >= 7 {
                        let pc = pos.piece_at(sq);
                        if let Some(pc) = pc {
                            match pc {
                                Piece::W_P
                                | Piece::W_L
                                | Piece::W_N
                                | Piece::W_S
                                | Piece::W_G
                                | Piece::W_PP
                                | Piece::W_PL
                                | Piece::W_PN
                                | Piece::W_PS => {
                                    value += 1;
                                    count += 1;
                                }
                                Piece::W_B | Piece::W_R | Piece::W_PB | Piece::W_PR => {
                                    value += 5;
                                    count += 1;
                                }
                                _ => (),
                            }
                        }
                    }
                }
                if count >= 10 {
                    let hand = pos.hand(Color::White);
                    for piece_type in PieceKind::all() {
                        if piece_type == PieceKind::King {
                            break;
                        }
                        match piece_type {
                            PieceKind::Pawn
                            | PieceKind::Lance
                            | PieceKind::Knight
                            | PieceKind::Silver
                            | PieceKind::Gold => value += hand.Hand_count(piece_type),
                            PieceKind::Bishop | PieceKind::Rook => {
                                value += hand.Hand_count(piece_type) * 5
                            }
                            _ => (),
                        }
                    }
                }
                if value >= 27 {
                    return true;
                }
            }
        }
    }

    false
}

pub fn move_to_sfen(m: Move) -> String {
    //! shogi_core::Move の指し手を文字列に変換する
    //!
    //! - Arguments
    //!   - m: Move
    //!     - 変換する指し手
    //! - Returns
    //!   - : String
    //!     - 文字列に変換した指し手

    match m {
        Move::Normal { from, to, promote } => {
            let from = format!(
                "{}{}",
                (b'1' + from.array_index() as u8 / 9) as char,
                (b'a' + from.array_index() as u8 % 9) as char
            );
            let to = format!(
                "{}{}",
                (b'1' + to.array_index() as u8 / 9) as char,
                (b'a' + to.array_index() as u8 % 9) as char
            );
            let promote = {
                if promote {
                    "+"
                } else {
                    ""
                }
            };
            format!("{from}{to}{promote}")
        }
        Move::Drop { to, piece } => {
            let to = format!(
                "{}{}",
                (b'1' + to.array_index() as u8 / 9) as char,
                (b'a' + to.array_index() as u8 % 9) as char
            );
            let piece = {
                match piece {
                    Piece::B_P | Piece::W_P => "P*".to_string(),
                    Piece::B_L | Piece::W_L => "L*".to_string(),
                    Piece::B_N | Piece::W_N => "N*".to_string(),
                    Piece::B_S | Piece::W_S => "S*".to_string(),
                    Piece::B_G | Piece::W_G => "G*".to_string(),
                    Piece::B_B | Piece::W_B => "B*".to_string(),
                    Piece::B_R | Piece::W_R => "R*".to_string(),
                    _ => unreachable!(),
                }
            };
            format!("{piece}{to}")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::search;
    use shogi_core::PartialPosition;
    use shogi_usi_parser::FromUsi;
    use yasai::Position;

    #[test]
    fn is_nyugyoku_win() {
        // 初期局面、入玉宣言できない
        let pos = Position::new(
            PartialPosition::from_usi(
                "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            )
            .expect("Initialization error"),
        );
        assert!(!search::is_nyugyoku_win(&pos));

        // 入玉宣言できる局面
        let pos = Position::new(
            PartialPosition::from_usi(
                "sfen 4+P1K2/6P+L+P/9/r+L3G2+b/ls3P3/5g3/3kn2br/1+l1+np2p+p/1s6s w P2gs2n10p 258",
            )
            .expect("Initialization error"),
        );
        assert!(search::is_nyugyoku_win(&pos));
    }
}
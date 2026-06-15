# docs/07_tactical_solver.md - CPU tactical solver 仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎における **CPU tactical solver** の仕様を定義する。

CPU tactical solver は、KLENT-0 や Gumbel MCTS の代替となる主探索ではない。役割は、ニューラルネットワークと 0-playout 方策改善だけでは落としやすい短期戦術・ルール上の確定手を補強することである。

対象は以下である。

```text
terminal / ルール処理との境界
宣言勝ち可能判定
相手玉への詰み探索
王手対応
探索方策への mask / forced move 適用
自己対局ログ・学習 target への反映
実装上の不変条件
検証項目
実験項目
```

関連仕様は以下を正とする。

```text
探索・着手選択全体: docs/03_search.md
ネットワーク出力: docs/05_outputs.md
KLENT-first 学習方針: docs/13_klent_training.md
設計概要: 爆裂玉登太郎_設計概要.md
```

---

## 1. 基本方針

### 1.1 tactical solver の役割

爆裂玉登太郎の標準探索は以下である。

```text
KLENT-0 + CPU tactical solver
```

KLENT-0 は root 局面で network の policy と action-value から改善方策を作る。探索木を展開しないため、短期的な詰み・王手対応を network の汎化能力だけに任せると、明らかな勝ちを逃す可能性がある。

CPU tactical solver は、以下のような局所的に確定できる情報を探索方策に反映する。

```text
宣言勝ちできるなら宣言勝ちする
相手玉への詰みが証明できるなら詰ます
王手中に合法応手が1手だけならその手を指す
詰み手の検出結果をログで検証できるようにする
```

### 1.2 tactical solver が主探索にならないようにする

CPU tactical solver は、通常の強さを作るための深い αβ探索ではない。

標準で扱う範囲は以下である。

```text
宣言勝ち可能判定
相手玉への詰み探索
王手対応
terminal / rule result の補助
```

長期的な局面評価、入玉計画、通常中盤の手の優劣は、network、KLENT 方策改善、Gumbel MCTS reanalyse に任せる。

### 1.3 強制手は方策改善より優先する

以下のようなルール上・戦術上の確定手は、policy や Q より優先する。

```text
terminal / ルール上の確定結果
宣言勝ち可能手
相手玉への詰みが証明された手
王手対応の唯一合法応手
```

確定手が複数ある場合は、以下の優先順位を使う。

```text
1. 宣言勝ち可能手
2. 相手玉への詰みが証明された手
3. Q_mode または pi_search が最大の手
4. deterministic tie-break
```

---

## 2. 探索全体との関係

### 2.1 着手選択フロー上の位置

標準の着手選択フローは以下である。

```text
1. 合法手を生成する
2. terminal / draw / 宣言勝ち可能判定を行う
3. CPU tactical solver の pre-check を開始する
4. network inference を開始する
5. pre-check で forced move が確定したらそれを返す
6. network 出力から policy prior と Q_mode を作る
7. KLENT-0 または Gumbel MCTS で探索方策を作る
8. CPU tactical solver の post-check で探索方策を補正する
9. sampling または argmax で着手する
```

pre-check の詰み探索と GPU inference は、同時に走らせてよい。pre-check が先に詰みを証明した場合は forced move を返す。GPU inference が先に終わった場合でも、pre-check の結果を待つかどうかは `TacticalConfig` の budget に従う。

### 2.2 pre-check と post-check

pre-check は、方策改善より前に確定手を検出する。

```text
宣言勝ち可能
相手玉への詰み
王手対応で唯一合法応手
```

post-check は、network と方策改善が作った分布に、pre-check / tactical solver で得られた確定手集合を反映する。

```text
declaration_win_moves への mask
mate_win_moves への mask
補正後の再正規化
```

### 2.3 terminal 判定との境界

terminal 判定は tactical solver の一部として実装してもよいが、概念上はルール処理である。

terminal node では network inference を行わない。

terminal に含めるものは以下である。

```text
合法手なし
詰み
宣言勝ち成立
千日手
持将棋引き分け
最大手数到達
自己対局の adjudication
```

合法手生成・move適用の異常は、通常の terminal reason ではなく実装バグとして扱い、学習 target 生成へ流さない。

CPU tactical solver は、terminal ではない局面で、次の着手選択に関する局所戦術を判定する。

---

## 3. TacticalConfig

CPU tactical solver の設定は、以下の構造体相当で管理する。

```python
@dataclass
class TacticalConfig:
    enabled: bool = True

    # rule / declaration
    enable_declaration_win_check: bool = True
    declaration_rule_id: str = "csa27"

    # mate search
    enable_mate_search: bool = True
    mate_search_algorithm: str = "dfpn"  # "dfpn" | "mate1" | "library"
    mate_search_max_ply: int | None = None
    mate_search_node_limit: int = 50_000
    mate_search_time_ms: int | None = 5
    mate_search_candidate_limit: int | None = None

    # check response
    enable_check_response_filter: bool = True

    # integration
    force_declaration_win: bool = True
    force_mate_win: bool = True

    # parallel execution
    run_precheck_in_parallel_with_inference: bool = True
    wait_for_precheck_before_final_select: bool = True

    # logging
    save_pre_tactical_pi: bool = True
    save_post_tactical_pi: bool = True
    save_masked_moves: bool = True
    debug: bool = False
```

初期実装の標準値は以下である。

```text
enable_declaration_win_check = True
enable_mate_search = True
mate_search_algorithm = "dfpn"
mate_search_node_limit = 50_000
mate_search_time_ms = 5
run_precheck_in_parallel_with_inference = True
enable_check_response_filter = True
```

詰み探索には node limit または time limit を必ず設定する。

---

## 4. TacticalResult

CPU tactical solver は、以下の構造体相当を返す。

```python
@dataclass
class TacticalResult:
    forced_move: int | None
    forced_reason: str | None

    winning_moves: set[int]
    declaration_win_moves: set[int]
    mate_win_moves: set[int]
    mate_in_one_moves: set[int]
    checking_moves: set[int]

    in_check: bool
    has_declaration_win: bool
    has_mate_win: bool

    searched_nodes: int
    elapsed_ms: float
    timed_out: bool
    flags: dict
```

各 field の意味は以下である。

| field | 意味 |
| --- | --- |
| `forced_move` | 方策改善を使わず即選択する手 |
| `forced_reason` | `declaration_win`, `mate_win`, `only_legal_response` など |
| `winning_moves` | 宣言勝ち手または詰みが証明された手集合 |
| `declaration_win_moves` | 宣言勝ち可能手集合 |
| `mate_win_moves` | 有界詰み探索で詰みが証明された手集合 |
| `mate_in_one_moves` | 相手玉を1手で詰みにする手集合 |
| `checking_moves` | 相手玉へ王手する手集合 |
| `in_check` | 現局面で自玉が王手されているか |
| `timed_out` | 詰み探索が time limit で打ち切られたか |
| `flags` | 実装依存の補助情報 |

`winning_moves` は、概念的には以下である。

```text
winning_moves = declaration_win_moves ∪ mate_win_moves
```

---

## 5. 宣言勝ち可能判定

### 5.1 方針

宣言勝ち可能判定は network に任せない。

現局面から厳密に計算できるため、探索側のルール処理として実装する。

```text
自玉位置
敵陣内の自駒数
持ち駒点数
盤上駒点数
王手されていないこと
その他ルール条件
```

### 5.2 DeclarationRuleConfig

宣言勝ち条件は、ルール差や実験設定で変わる可能性があるため、設定として分離する。

```python
@dataclass
class DeclarationRuleConfig:
    rule_id: str = "csa27"
    enemy_camp_ranks: tuple[int, ...] = (1, 2, 3)
    require_side_to_move: bool = True
    require_king_in_enemy_camp: bool = True
    require_not_in_check: bool = True

    point_threshold_black: int = 28
    point_threshold_white: int = 27
    rook_bishop_point: int = 5
    other_piece_point: int = 1

    min_pieces_in_enemy_camp_excluding_king: int = 10
    include_hand_pieces: bool = True
    include_board_pieces_in_enemy_camp: bool = True
    include_king_in_piece_count: bool = False
```

ルール定数は、feature encoder の入玉関連特徴、自己対局の adjudication、評価ログと必ず一致させる。

### 5.3 宣言勝ち手の扱い

宣言勝ちが可能な局面では、原則として即選択する。

```text
has_declaration_win = True:
  forced_move = declaration_move
  forced_reason = "declaration_win"
```

宣言勝ちが「指し手」として表現される場合は、専用 move label を使う。

宣言勝ちが game manager の特殊操作として表現される場合は、`selected_move` とは別に `terminal_action = declaration_win` を返す。

どちらを採用する場合でも、自己対局ログには以下を残す。

```text
declaration_win_available
declaration_win_selected
declaration_rule_id
own_nyugyoku_score
opp_nyugyoku_score
own_pieces_in_enemy_camp
opp_pieces_in_own_camp
```

---

## 6. 相手玉への詰み探索

### 6.1 1手詰め

1手詰めは、詰み探索の高速特殊ケースとして扱う。

処理は以下である。

```text
for move in legal_moves:
  position.do(move)
  if opponent has no legal move and opponent king is in check:
      mate_in_one_moves.add(move)
  position.undo(move)
```

実装上は、将棋ライブラリが提供する詰み判定または合法手生成を利用してよい。

### 6.2 有界詰み探索

pre-check では、相手玉への詰みを有界探索で調べる。

標準アルゴリズムは df-pn 系を第一候補とする。

```text
for move in legal_moves:
  position.do(move)
  if prove_mate_for_side_to_move(position, budget):
      mate_win_moves.add(move)
  position.undo(move)
```

探索は必ず以下の budget に従う。

```text
node limit
time limit
max ply または無制限設定
candidate limit
```

`mate_win_moves` が非空の場合、原則として即選択する。

```text
mate_win_moves != empty:
  forced_move = best move in mate_win_moves
  forced_reason = "mate_win"
```

複数ある場合の tie-break は以下とする。

```text
1. Q_mode が利用可能なら Q_mode 最大
2. pi_search が利用可能なら pi_search 最大
3. policy prior が利用可能なら prior 最大
4. move label の deterministic order
```

pre-check 段階では Q_mode がまだない場合がある。その場合は deterministic order で選んでよい。

### 6.3 CPU / GPU 並列実行

詰み探索は CPU 側で行い、network inference は GPU 側で行う。

標準では、pre-check の詰み探索と GPU inference を同時に開始できるようにする。

```text
CPU:
  declaration check
  mate search
  check response forced move detection

GPU:
  policy / Q / outcome / moves_left inference
```

詰みが証明された場合は、network 出力や KLENT 方策より詰み手を優先する。

### 6.4 timeout の扱い

詰み探索が timeout した場合、詰み不成立とはみなさない。

```text
timeout:
  mate result = unknown
  timed_out = True
  searched_nodes と elapsed_ms を保存する
```

unknown の候補手を負け手として扱ってはならない。

---

## 7. 王手対応

### 7.1 方針

現局面で自玉が王手されている場合、合法手生成により王手を解除する手だけが返ることを前提にする。

### 7.2 唯一合法応手

王手中で合法手が1手だけなら、forced move としてよい。

```text
if in_check and len(legal_moves) == 1:
  forced_move = legal_moves[0]
  forced_reason = "only_legal_response"
```

合法応手が複数ある場合は、KLENT / Gumbel MCTS の分布から着手を選ぶ。

### 7.3 王手中の詰み検出

王手中でも、相手玉を詰みにできる応手がある場合は、詰み手を優先してよい。

```text
if mate_win_moves:
  forced_move = best move in mate_win_moves
  forced_reason = "mate_win"
```

これは、相手玉を詰ませる手であれば自玉の王手も解消されている合法手であるためである。

---

## 8. 王手候補ログ

### 8.1 方針

CPU tactical solver は、解析用に `checking_moves` を保存する。

```text
checking_moves:
  相手玉へ王手する合法手集合
```

### 8.2 ログ項目

解析用に以下を保存する。

```text
checking_moves
mate_in_one_moves
mate_win_moves
in_check
num_legal_moves
```

これにより、以下を評価する。

```text
詰み手検出率
王手中の応手選択
policy が tactical move をどれくらい上位に置いたか
Q_mode が tactical move をどれくらい評価したか
```

---

## 9. 方策分布への統合

### 9.1 mask の優先順位

post-check で方策分布を補正する優先順位は以下である。

```text
1. declaration_win_moves が非空なら、それだけに mask
2. mate_win_moves が非空なら、それだけに mask
3. 補正後に合法手上で再正規化
```

### 9.2 apply_tactical_mask

標準処理は以下である。

```python
def apply_tactical_mask(
    pi: np.ndarray,
    legal_moves: list[int],
    tactical: TacticalResult,
) -> np.ndarray:
    allowed = set(legal_moves)

    if tactical.declaration_win_moves:
        allowed &= tactical.declaration_win_moves
    elif tactical.mate_win_moves:
        allowed &= tactical.mate_win_moves

    if not allowed:
        raise TacticalSolverError("tactical mask removed all moves")

    return renormalize_on_allowed_moves(pi, allowed)
```

`winning_moves` は、`declaration_win_moves` と `mate_win_moves` をまとめた上位概念として扱ってよい。

### 9.3 Q_mode と pi_search の保存

`tactical mask` は、着手選択に使う方策分布へ反映する。

保存する値は以下である。

```text
q_mode_pre_tactical:
  network / search 由来の Q_mode

pi_search_post_tactical:
  tactical mask 後の着手選択分布
```

---

## 10. KLENT-0 との統合

### 10.1 KLENT-0 root での使い方

KLENT-0 では、root 局面だけで以下を行う。

```text
1. tactical pre-check を開始する
2. network inference を開始する
3. pre-check で forced_move が得られた場合は即選択する
4. p_prior と Q_mode から pi_klent を作る
5. tactical post-check で pi_klent を補正する
6. sampling / argmax で着手する
```

pre-check と network inference は並列実行してよい。

### 10.2 target 保存

自己対局では、以下を保存する。

```text
p_prior
pi_klent_pre_tactical
pi_search_post_tactical
selected_move
tactical_flags
```

学習 target には、標準では `pi_search_post_tactical` を使う。

解析用に `pi_klent_pre_tactical` も保存し、network / KLENT 単体の弱点を測る。

### 10.3 tactical solver に依存しすぎないための評価

評価では以下を分けて測る。

```text
KLENT-0 only
KLENT-0 + declaration check
KLENT-0 + mate search
KLENT-0 + full tactical solver
```

---

## 11. Gumbel MCTS との統合

### 11.1 root での扱い

Gumbel MCTS でも、root では tactical solver を使う。

```text
declaration win:
  search せず即選択

mate win:
  search せず即選択
```

root candidate sampling は、tactical mask 後の合法手集合に対して行う。

### 11.2 leaf での扱い

Gumbel MCTS の leaf では、まず terminal 判定と宣言勝ち可能判定を優先する。

詰み探索を leaf で使うかどうかは、Gumbel MCTS の search config で管理する。

### 11.3 backup との関係

Tactical solver が検出した terminal / forced win は、network value ではなく rule value として backup する。

MCTS 内でも以下は守る。

```text
Q_normal_result と Q_nyugyoku を分ける
Q_mode だけを backup しない
leaf value は root 視点へ変換する
```

宣言勝ちは `nyugyoku` 成分の勝ちとして扱う。

通常の詰み勝ちは `normal_result` 成分の勝ちとして扱う。

---

## 12. 学習データへの反映

### 12.1 policy target

自己対局で tactical solver が分布を補正した場合、標準では補正後分布を policy target にする。

```text
policy_normal_target = pi_search_post_tactical
```

style モードの自己対局では、探索設定に応じて `policy_style_target` にも使う。

```text
style / nyugyoku / anti_nyugyoku 系:
  policy_style_target = pi_search_post_tactical
```

### 12.2 q target

Q target は、終局結果、λ-return、Gumbel MCTS reanalyse を正本にする。

局所的に結果が確定した tactical 情報は、以下の場合だけ確定 target として扱う。

```text
宣言勝ち確定:
  q_nyugyoku_target = +1

相手玉への詰み確定:
  q_normal_result_target = +1
```

これらを全合法手 target として使う場合は、探索アルゴリズム、node limit、time limit、timeout 有無をログに残す。

### 12.3 outcome target

terminal reason は、`outcome_cause` の分類体系に従って保存する。

```text
詰み / 投了 / 通常 adjudication:
  normal

宣言勝ち / 入玉条件 adjudication:
  nyugyoku

千日手 / 持将棋引き分け / 最大手数到達で勝敗未確定:
  draw
```

CPU tactical solver が宣言勝ちを選んで終局した場合、その終局原因は `nyugyoku` として保存する。

---

## 13. ログ仕様

### 13.1 基本ログ

CPU tactical solver は、最低限以下を記録する。

```text
tactical/enabled
tactical/in_check
tactical/forced_move
tactical/forced_reason
tactical/num_legal_moves
tactical/num_winning_moves
tactical/num_checking_moves
tactical/num_mate_in_one_moves
tactical/num_mate_win_moves
tactical/searched_nodes
tactical/elapsed_ms
tactical/timed_out
```

### 13.2 宣言勝ちログ

宣言勝ち関連では以下を記録する。

```text
nyugyoku/declaration_win_available
nyugyoku/declaration_win_selected
nyugyoku/declaration_rule_id
nyugyoku/own_king_in_enemy_camp
nyugyoku/opp_king_in_own_camp
nyugyoku/own_pieces_in_enemy_camp
nyugyoku/opp_pieces_in_own_camp
nyugyoku/own_nyugyoku_score
nyugyoku/opp_nyugyoku_score
nyugyoku/own_score_remaining
nyugyoku/opp_score_remaining
```

### 13.3 分布補正ログ

分布補正を行った場合は以下を保存する。

```text
policy/entropy_pre_tactical
policy/entropy_post_tactical
policy/kl_post_to_pre_tactical
policy/top1_pre_tactical
policy/top1_post_tactical
policy/selected_move_pre_tactical_rank
policy/selected_move_post_tactical_rank
```

sparse ログでは以下を保存する。

```text
pre_tactical_pi
post_tactical_pi
masked_moves
forced_move_reason
```

### 13.4 失敗解析ログ

以下のフラグを通常ログに含める。

```text
tactical/missed_mate_win
tactical/mask_removed_all_moves
tactical/tactical_solver_error
tactical/tactical_timeout
```

---

## 14. 実装 API

### 14.1 TacticalSolver class

標準 API は以下とする。

```python
class TacticalSolver:
    def __init__(self, config: TacticalConfig, rule_config: DeclarationRuleConfig):
        ...

    def analyze(
        self,
        position: "Position",
        legal_moves: list[int],
        context: "TacticalContext",
    ) -> TacticalResult:
        ...
```

### 14.2 TacticalContext

```python
@dataclass
class TacticalContext:
    root_side: int
    game_ply: int | None
    max_game_ply: int | None
    repetition_state: "RepetitionState | None"
    q_mode: np.ndarray | None = None
    pi_search: np.ndarray | None = None
    rng: "RandomState | None" = None
```

pre-check 段階では `q_mode` や `pi_search` が存在しない場合がある。

post-check 段階では、tie-break に `q_mode` や `pi_search` を使ってよい。

### 14.3 SearchEngine との接続

SearchEngine は、tactical solver の結果を以下のように扱う。

```python
tactical = tactical_solver.analyze(position, legal_moves, context)

if tactical.forced_move is not None:
    return make_forced_move_result(tactical)

network_output = infer_network(position)
converted = convert_network_output_for_search(network_output, legal_moves, config)
pi_klent = compute_klent_policy(converted.policy_prior, converted.q_mode, legal_moves, config)
pi_search = apply_tactical_mask(pi_klent, legal_moves, tactical)
selected_move = sample_or_argmax(pi_search, config)
```

並列実行する場合は、`analyze` を future として起動し、network inference と合流する。

---

## 15. performance 方針

### 15.1 CPU bottleneck を避ける

KLENT-first の主目的は、自己対局局面数と更新回数を稼ぐことである。

そのため、CPU tactical solver が GPU batched inference の足を引っ張らないようにする。

標準では以下を守る。

```text
node limit / time limit を必ず持つ
探索に使ったノード数と時間をログする
pre-check と GPU inference を並列化できるようにする
```

### 15.2 候補手制限

詰み探索が重い場合、以下の順で候補手を制限してよい。

```text
1. 王手中は全合法手を検査する
2. 終盤・玉周辺では全合法手を検査する
3. それ以外は policy / Q 上位 K 手を優先検査する
4. 未検査手は unknown として扱う
```

unknown は、詰みなしとはみなさない。

---

## 16. 実装上の不変条件

以下は必ず守る。

```text
TacticalResult の move は合法手集合の要素である
terminal node では network inference を呼ばない
宣言勝ち可能判定は network output に依存しない
相手玉への詰み手は policy / Q より優先する
tactical mask 後の確率分布は合法手上で正規化する
違法手確率は必ず0にする
Q_mode は network / search 由来の値として保存する
pi_search_post_tactical は tactical mask 後の分布として保存する
宣言勝ちルール定数は feature encoder / adjudication / evaluation と一致させる
詰み探索には depth / node / time limit を持たせる
timeout を詰みなしとして扱わない
```

---

## 17. 検証項目

### 17.1 宣言勝ち判定 test

以下を確認する。

```text
自玉が敵陣にいない場合、宣言勝ち不可
王手されている場合、宣言勝ち不可
点数が閾値未満の場合、宣言勝ち不可
敵陣内自駒数が不足する場合、宣言勝ち不可
条件をすべて満たす場合、宣言勝ち可能
先手 / 後手の点数閾値が rule config 通りに使われる
feature encoder の入玉関連特徴と同じ点数計算になる
```

### 17.2 mate search test

以下を確認する。

```text
1手詰めがある局面で mate_in_one_moves が非空になる
有界詰み探索で詰みが証明できる局面では mate_win_moves が非空になる
詰みがない局面で false positive しない
timeout 時に timed_out が立つ
mate_win_moves は legal_moves の部分集合である
```

### 17.3 check response test

以下を確認する。

```text
in_check が正しく立つ
王手中に合法応手だけが legal_moves として返る
合法応手が1手だけなら forced_move になる
合法応手が複数なら分布から着手選択する
王手中でも詰み手があれば詰みを優先する
```

### 17.4 integration test

以下を確認する。

```text
宣言勝ち可能なら network inference 前に forced result を返せる
詰み手がある場合、その手を選ぶ
tactical mask 後の確率和が1になる
pre_tactical_pi と post_tactical_pi が保存される
CPU pre-check と GPU inference の並列実行で結果が変わらない
```

### 17.5 performance test

以下を確認する。

```text
平均 elapsed_ms
p95 elapsed_ms
searched_nodes
timeout 発生率
自己対局 positions/sec への影響
GPU batch inference を阻害していないか
```

---

## 18. 実験項目

### 18.1 tactical solver ablation

以下を比較する。

```text
A. KLENT-0 only
B. A + 宣言勝ち判定
C. B + 1手詰め検出
D. B + 有界詰み探索
E. D + CPU / GPU 並列 pre-check
```

観測する指標は以下である。

```text
通常勝率
入玉宣言勝ち率
相手入玉負け率
詰み逃し率
短手数負け率
positions/sec
CPU elapsed_ms
GPU batch size
```

### 18.2 詰み探索 budget

以下を比較する。

```text
mate_search_time_ms
mate_search_node_limit
mate_search_candidate_limit
mate_search_algorithm
```

評価では、詰み逃し率と自己対局 throughput を同時に見る。

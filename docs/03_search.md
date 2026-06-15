# docs/03_search.md - 探索・着手選択仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎における探索・着手選択の実行仕様を定義する。

対象は以下である。

```text
探索モード
合法手 mask
terminal / 宣言勝ち判定
CPU tactical solver との合成
policy_normal / policy_style の混合
Q_normal_result / Q_nyugyoku の合成
KLENT-0 方策改善
Gumbel MCTS
実戦時探索
探索ログ
実装上の不変条件
検証項目
```

爆裂玉登太郎の主学習ループは、全局面で重い MCTS を回す方式ではなく、network の policy と action-value から 0-playout の改善方策を作る **KLENT-0 + CPU tactical solver** を基本とする。

Gumbel MCTS は、再解析、難所局面の深掘り、実戦時探索に使う。通常の自己対局全局面で Gumbel MCTS を必ず回す設計にはしない。

このドキュメントでは探索側の実行仕様を扱う。network の head 仕様、loss、target は `docs/05_outputs.md`、network 本体構造は `docs/04_network.md`、探索価値と視点変換は `docs/06_search_value.md`、KLENT-first の学習全体方針は `docs/13_klent_training.md` を正とする。

---

## 1. 基本方針

### 1.1 探索の役割

探索側の役割は、現在局面から合法手を選ぶことである。

爆裂玉登太郎では、探索側が以下を統合する。

```text
ルール処理:
  合法手生成
  terminal 判定
  宣言勝ち可能判定
  千日手・最大手数処理
  合法手生成・move適用異常の検出

短期戦術:
  詰み
  王手対応

network 出力:
  policy_normal
  policy_style
  Q_normal_result
  Q_nyugyoku
  outcome_cause
  moves_left

方策改善:
  KLENT-0
  Gumbel MCTS
```

重要なのは、**長期方針は network と方策改善に任せ、短期戦術とルール処理は探索側で補強する** ことである。

### 1.2 標準探索

標準の着手選択は以下とする。

```text
KLENT-0 + CPU tactical solver
```

KLENT-0 は、探索木を展開せず、root 局面の合法手に対して以下を行う。

```text
1. policy prior を作る
2. action-value を作る
3. 探索モードごとの Q_mode を作る
4. KLENT 改善方策 pi_klent を作る
5. tactical solver の結果で補正する
6. sampling または argmax で着手する
```

### 1.3 Gumbel MCTS の位置づけ

Gumbel MCTS は以下に使う。

```text
再解析
難所局面解析
実戦時 test-time search
KLENT-0 と探索ありの差分測定
```

主学習ループの全局面に使うものではない。

理由は以下である。

```text
自己対局局面数を優先したい
入玉関連局面を大量に集めたい
KLENT-0 の方が実装とログが単純である
Gumbel MCTS は効果が高い局面へ集中した方が効率がよい
```

### 1.4 探索側で扱わないこと

探索側は、network の学習 target の定義や loss weight を決めない。

探索側は、以下の値を使うだけである。

```text
policy_normal_logits
policy_style_logits
q_normal_result_raw
q_nyugyoku_raw
outcome_cause_logits
moves_left_pred
```

また、network module 内で合法手 mask や `tanh` や KLENT 方策計算を行わせない。これらは探索側または変換層の責務とする。

---

## 2. 探索モード

### 2.1 SearchModeConfig

探索モードは、以下の構造体相当で管理する。

```python
@dataclass
class SearchModeConfig:
    # policy blend
    alpha_style_self: float
    alpha_style_opponent_model: float

    # value blend
    lambda_nyugyoku: float

    # KLENT-0
    klent_alpha_entropy: float
    klent_beta_prior: float
    klent_eps: float = 1e-8

    # final move selection
    temperature: float = 1.0
    deterministic: bool = False

    # self-play exploration noise
    root_dirichlet_alpha: float | None = None
    root_noise_frac: float = 0.0

    # tactical solver
    enable_tactical_solver: bool = True
    enable_declaration_win_check: bool = True

    # Gumbel MCTS
    search_type: str = "klent0"  # "klent0" | "gumbel_mcts"
    num_simulations: int = 0
    max_depth: int | None = None
    gumbel_top_k: int | None = None
    sequential_halving_rounds: int | None = None

    # logging / debug
    save_policy_distribution: bool = True
    save_q_distribution: bool = True
```

`search_type = "klent0"` の場合、`num_simulations` は使わない。

`search_type = "gumbel_mcts"` の場合、root prior と leaf value は同じ network 出力変換に従う。

### 2.2 代表的な探索モード

代表的な探索モードは以下とする。

```text
normal:
  alpha_style_self = 0.0 〜 0.1
  alpha_style_opponent_model = 0.0 〜 0.1
  lambda_nyugyoku = 1.0

nyugyoku:
  alpha_style_self = 0.3 〜 0.5
  alpha_style_opponent_model = 0.0 〜 0.2
  lambda_nyugyoku = 1.5 〜 3.0

anti_nyugyoku:
  alpha_style_self = 0.0 〜 0.2
  alpha_style_opponent_model = 0.3 〜 0.6
  lambda_nyugyoku = 1.5 〜 3.0

mutual_nyugyoku:
  alpha_style_self = 0.3 〜 0.5
  alpha_style_opponent_model = 0.3 〜 0.5
  lambda_nyugyoku = 1.5 〜 3.0

bakuretsu:
  alpha_style_self = 0.2 〜 0.5
  alpha_style_opponent_model = 0.1 〜 0.4
  lambda_nyugyoku = 1.2 〜 2.5
```

### 2.3 root側と相手側の style 混合率

`policy_style_head` は、常に「その局面で手番を持つ側にとっての style policy」である。

そのため、探索木の中では、node の手番が root 側か相手側かで `α_style` を変える。

```python
def select_alpha_style(node_side: int, root_side: int, config: SearchModeConfig) -> float:
    if node_side == root_side:
        return config.alpha_style_self
    return config.alpha_style_opponent_model
```

KLENT-0 は root だけを評価するため、通常は `alpha_style_self` を使う。

Gumbel MCTS では、展開する各 node で `select_alpha_style` を使う。

### 2.4 自己対局時の相手モデル

自己対局では、先手・後手それぞれに探索設定を持たせる。

```python
@dataclass
class PlayerSearchConfig:
    mode_id: str
    config: SearchModeConfig
```

自分の着手時には、自分の `SearchModeConfig` で root 着手を選ぶ。

探索木の相手側 node では、対戦相手に割り当てられた探索設定を近似する。

例えば、先手が `nyugyoku`、後手が `normal` の場合は以下になる。

```text
先手の探索:
  root側: nyugyoku
  相手モデル側: normal

後手の探索:
  root側: normal
  相手モデル側: nyugyoku
```

初期実装で相手モデル設定の受け渡しが複雑になる場合は、以下の近似から始めてよい。

```text
KLENT-0:
  root側設定だけを使う

Gumbel MCTS:
  alpha_style_opponent_model だけで相手側 policy を近似する
```

---

## 3. 探索 API

### 3.1 SearchRequest

探索関数は、以下の構造体相当を受け取る。

```python
@dataclass
class SearchRequest:
    position: "Position"
    root_side: int
    config: SearchModeConfig

    # optional
    game_ply: int | None = None
    max_game_ply: int | None = None
    repetition_state: "RepetitionState | None" = None
    time_control: "TimeControl | None" = None
    rng: "RandomState | None" = None
```

`position` は、盤面、持ち駒、手番、合法手生成に必要な状態を持つ。

`root_side` は、この探索で着手を選ぶ側である。

### 3.2 SearchResult

探索関数は、以下の構造体相当を返す。

```python
@dataclass
class SearchResult:
    selected_move: int
    legal_moves: list[int]

    policy_prior: np.ndarray
    pi_klent: np.ndarray | None
    pi_search: np.ndarray

    q_normal_result: np.ndarray | None
    q_nyugyoku: np.ndarray | None
    q_total_result: np.ndarray | None
    q_mode: np.ndarray | None

    outcome_probs: np.ndarray | None
    v_normal_result: float | None
    v_nyugyoku: float | None
    v_total_result: float | None
    v_mode: float | None

    moves_left: float | None

    search_type: str
    search_config_id: str
    tactical_flags: dict
    terminal_flags: dict

    # Gumbel MCTS only
    visit_counts: np.ndarray | None = None
    q_normal_result_search: np.ndarray | None = None
    q_nyugyoku_search: np.ndarray | None = None
```

`pi_search` は最終的に着手選択に使った分布である。

KLENT-0 では、`pi_klent_pre_tactical` を CPU tactical solver で補正した分布を `pi_search_post_tactical` として保存し、`pi_search` は通常この補正後分布を指す。

Gumbel MCTS では、`pi_search` は MCTS による改善方策または visit distribution である。

### 3.3 Dense と sparse

内部表現は、実装容易性を優先して dense `[MOVE_LABEL_SIZE]` を標準とする。

ただし、自己対局ログ保存ではファイルサイズが大きくなるため、合法手だけの sparse 表現に変換してよい。

```python
@dataclass
class SparseMoveDistribution:
    moves: list[int]
    probs: list[float]
```

dense と sparse の変換では、以下を保証する。

```text
合法手以外の確率は0
合法手上の確率和は1
moves と probs の順序が一致する
move label は現在手番視点の encoder / decoder と整合する
```

---

## 4. 着手選択フロー

標準の着手選択フローは以下とする。

```text
1. position から合法手を生成する
2. terminal / draw / 宣言勝ち可能判定を行う
3. CPU tactical solver の pre-check を行う
4. network inference を行う
5. policy prior を作る
6. Q_normal_result / Q_nyugyoku / Q_mode を作る
7. KLENT-0 または Gumbel MCTS で探索方策を作る
8. CPU tactical solver の post-check で補正する
9. 温度・sampling 設定に従って着手する
10. SearchResult を返す
```

疑似コードは以下である。

```python
def select_move(request: SearchRequest) -> SearchResult:
    legal_moves = generate_legal_moves(request.position)

    terminal = check_terminal_or_rule_result(
        request.position,
        legal_moves,
        request.repetition_state,
        request.game_ply,
        request.max_game_ply,
    )
    if terminal.is_finished:
        return make_terminal_result(terminal, legal_moves, request.config)

    if request.config.enable_declaration_win_check:
        declaration = check_declaration_win(request.position, legal_moves)
        if declaration.is_win:
            return make_forced_move_result(declaration.move, legal_moves, reason="declaration_win")

    tactical_pre = run_tactical_precheck(request.position, legal_moves, request.config)
    if tactical_pre.has_forced_move:
        return make_forced_move_result(tactical_pre.move, legal_moves, reason=tactical_pre.reason)

    network_output = infer_network(request.position)
    converted = convert_network_output_for_search(
        network_output=network_output,
        legal_moves=legal_moves,
        root_side=request.root_side,
        node_side=request.root_side,
        config=request.config,
    )

    if request.config.search_type == "klent0":
        pi_search = compute_klent_policy(converted.policy_prior, converted.q_mode, legal_moves, request.config)
    elif request.config.search_type == "gumbel_mcts":
        pi_search = run_gumbel_mcts(request, converted, tactical_pre)
    else:
        raise ValueError(request.config.search_type)

    pi_search = apply_tactical_postprocess(pi_search, converted.q_mode, tactical_pre, legal_moves)
    selected_move = sample_or_argmax(pi_search, request.config, request.rng)

    return make_search_result(selected_move, legal_moves, converted, pi_search, tactical_pre)
```

---

## 5. 合法手・terminal・ルール処理

### 5.1 合法手生成

合法手生成は探索側の責務である。

network は合法手を知らない。

```text
network:
  全 MOVE_LABEL_SIZE に logits / raw value を出す

search:
  合法手 mask を作る
  違法手確率を0にする
  合法手上だけで softmax / KLENT / sampling を行う
```

合法手が空の場合は terminal として扱う。

### 5.2 terminal 判定

terminal node では network を評価しない。

terminal として扱うものは以下である。

```text
詰み
宣言勝ち成立
千日手
持将棋引き分け
最大手数到達
自己対局の adjudication
```

合法手生成済みの自己対局で `illegal_move` 相当の状態が発生した場合は、通常の terminal outcome ではなく実装バグとして扱う。その対局は学習 target に使わない。

terminal の outcome への変換は、学習 target 生成側の仕様に従う。

探索側では、terminal node を検出し、ルール上の結果を返す。

### 5.3 宣言勝ち可能判定

宣言勝ち可能判定は network に任せない。

現局面から厳密に計算できるため、探索側のルール処理で判定する。

```text
自玉位置
敵陣内の自駒数
持ち駒点数
盤上駒点数
王手されていないこと
その他ルール条件
```

宣言勝ちが可能で、宣言勝ちを選べる局面では、原則として即選択する。

### 5.4 千日手・最大手数

千日手、最大手数到達、持将棋引き分けは探索側または対局管理側で処理する。

初期実装では、反復回数や最大手数までの残りを network input に含めない。

したがって、これらのルール状態は network cache key や feature tensor ではなく、探索 request / game manager 側で管理する。

---

## 6. network 出力の探索用変換

### 6.1 変換の責務

network の raw output から探索で使う値へ変換する処理は、以下を行う。

```text
合法手 mask
policy_normal / policy_style の softmax
policy blend
q raw の tanh 変換
Q_total_result の計算
Q_mode の計算
KLENT 方策の計算
outcome softmax
V_normal_result / V_nyugyoku / V_total_result の計算
moves_left の expm1 変換
```

### 6.2 policy prior

まず、合法手上で policy を作る。

```text
p_normal = softmax_legal(policy_normal_logits)
p_style  = softmax_legal(policy_style_logits)
```

次に、softmax 後の確率を混合する。

```text
p_prior = normalize((1 - α_style) * p_normal + α_style * p_style)
```

logits を直接混合しない。

理由は以下である。

```text
policy_normal と policy_style の logit scale が一致するとは限らない
学習初期に style head が不安定になりやすい
確率分布として混ぜた方が探索モードの意味が明確である
```

### 6.3 root noise

自己対局の局面多様性を増やす場合、root prior に Dirichlet noise を混ぜてよい。

```text
p_prior_noisy = (1 - η) * p_prior + η * Dirichlet(α)
```

noise は KLENT 方策計算の前に混ぜる。

```text
p_prior -> p_prior_noisy -> pi_klent
```

評価対局、実戦時探索、再解析では原則として root noise を無効にする。

### 6.4 action-value

raw value は `tanh` で `[-1, 1]` に変換する。

```text
Q_normal_result = tanh(q_normal_result_raw)
Q_nyugyoku      = tanh(q_nyugyoku_raw)
```

総合勝敗価値は以下である。

```text
Q_total_result = Q_normal_result + Q_nyugyoku
```

探索モードごとの価値は以下である。

```text
Q_mode = Q_normal_result + λ_nyugyoku * Q_nyugyoku
```

`λ_nyugyoku = 1.0` の場合、通常勝ちと入玉勝ちを同じ勝敗価値として扱う。

`λ_nyugyoku > 1.0` の場合、自分の入玉勝ちを強く好み、相手入玉による負けを強く嫌う。

### 6.5 outcome 分布

`outcome_cause_logits` は softmax して以下を得る。

```text
P_win_normal
P_win_nyugyoku
P_draw
P_loss_nyugyoku
P_loss_normal
```

局面単位の value は以下で計算する。

```text
V_normal_result = P_win_normal - P_loss_normal
V_nyugyoku      = P_win_nyugyoku - P_loss_nyugyoku
V_total_result  = V_normal_result + V_nyugyoku
V_mode          = V_normal_result + λ_nyugyoku * V_nyugyoku
```

KLENT-0 の root 着手選択では、基本的に合法手ごとの `Q_mode` を使う。

`V_mode` は以下に使う。

```text
ログ
calibration 確認
Gumbel MCTS の leaf value
λ-return 作成
局面バッファ分類
```

### 6.6 moves_left

`moves_left_pred` は `log1p(moves_left)` の予測である。

探索側では以下で手数に戻す。

```text
moves_left = expm1(moves_left_pred)
```

初期実装では、`moves_left` を主探索価値には入れない。

導入する場合も小さい補正に留める。

```text
勝ち寄り局面:
  早い勝ちをわずかに好む

負け寄り局面:
  遅い負けをわずかに好む

入玉価値が高い局面:
  長期化を過度に嫌わない
```

標準では `moves_left` 補正は無効とする。

---

## 7. KLENT-0 方策改善

### 7.1 定義

KLENT 改善方策は、合法手上で以下とする。

```text
pi_klent(a | s) = softmax_legal(
    (Q_mode(s, a) + β * log(max(p_prior(a | s), ε))) / (α + β)
)
```

`α` は entropy regularization、`β` は prior policy からの乖離抑制である。

`ε` は `log(0)` を避けるための小さい値である。

### 7.2 意味

KLENT-0 は、prior policy と action-value を使って、探索なしで改善方策を作る。

直感的には以下である。

```text
Q_mode が高い手:
  確率が上がる

p_prior が高い手:
  確率が維持されやすい

α が大きい:
  分布が広がりやすい

β が大きい:
  prior から離れにくい
```

### 7.3 数値安定化

実装では、合法手以外を `-inf` 相当に mask する。

```python
def compute_klent_policy(prior, q_mode, legal_mask, alpha_entropy, beta_prior, eps):
    prior = mask_and_renormalize(prior, legal_mask)
    log_prior = np.log(np.maximum(prior, eps))
    logits = (q_mode + beta_prior * log_prior) / (alpha_entropy + beta_prior)
    logits = mask_illegal_logits(logits, legal_mask)
    return softmax(logits)
```

以下を満たす必要がある。

```text
違法手確率は0
合法手上の確率和は1
prior が0に近い手でも NaN を出さない
alpha_entropy + beta_prior > 0
```

### 7.4 最終着手分布

自己対局では、tactical 補正後の `pi_search_post_tactical` に温度をかけて sampling してよい。

```text
pi_select(a) ∝ pi_search_post_tactical(a)^(1 / temperature)
```

評価対局では、原則として tactical 補正後の `argmax` を使う。

```text
self-play:
  sampling

evaluation:
  argmax

実戦時探索:
  原則 argmax
```

`temperature = 0` は `argmax` として扱う。

### 7.5 target として保存する分布

自己対局で実際に sampling に使った分布と、学習 target として保存する分布は分けてよい。

```text
pi_klent_pre_tactical:
  KLENT 改善方策そのもの。tactical solver 補正前の解析用分布。

pi_search_post_tactical:
  tactical solver 補正後の policy target 標準分布。

pi_select:
  温度・noise 適用後に実際に着手選択へ使った分布。
```

初期実装では、以下を保存する。

```text
p_prior
pi_klent_pre_tactical
pi_search_post_tactical
pi_select
selected_move
```

---

## 8. CPU tactical solver との合成

### 8.1 目的

KLENT-0 は探索木を展開しないため、短期戦術の穴がそのまま勝ち逃しにつながる可能性がある。

特に将棋では以下が重要である。

```text
相手玉への詰み探索
王手への応手
詰み探索周辺
宣言勝ち可能判定
```

これらは network だけに任せず、CPU tactical solver で補強する。

### 8.2 TacticalResult

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

まずは以下を優先する。

```text
宣言勝ち可能判定
相手玉への詰み探索
王手対応
```

### 8.3 優先順位

着手選択の優先順位は以下とする。

```text
1. terminal / ルール上の確定結果
2. 宣言勝ち可能なら宣言勝ち
3. 相手玉への詰みが証明された手
4. 王手対応で唯一合法応手
5. KLENT / Gumbel MCTS による方策改善
6. tactical postprocess 後に sampling / argmax
```

勝ち確定手がある場合は、policy や Q より優先する。

### 8.4 補正方法

補正は単純でよい。

```text
winning_moves が非空:
  winning_moves だけに mask する
```

確率分布を補正する場合は、補正後に合法手上で再正規化する。

```python
def apply_tactical_mask(pi, legal_moves, tactical):
    allowed = set(legal_moves)

    if tactical.declaration_win_moves:
        allowed &= tactical.declaration_win_moves
    elif tactical.mate_win_moves:
        allowed &= tactical.mate_win_moves

    if not allowed:
        raise TacticalSolverError("tactical mask removed all moves")

    return renormalize_on_allowed_moves(pi, allowed)
```

### 8.5 学習ログへの反映

Tactical solver の結果で着手選択を補正した場合、以下を保存する。

```text
tactical_flags
pre_tactical_pi
post_tactical_pi
forced_move_reason
mate_search_algorithm
mate_search_node_limit
mate_search_time_ms
tactical_timeout
```

policy target には、標準で `post_tactical_pi` を使う。

Q target への反映は、宣言勝ち確定または相手玉への詰み確定のように結果が局所的に証明された場合に限定する。

## 9. Gumbel MCTS

### 9.1 使用場面

Gumbel MCTS は以下に使う。

```text
Gumbel MCTS reanalyse
難所局面解析
実戦時 small / medium / large budget search
KLENT-0 の弱点確認
```

対象局面は以下を優先する。

```text
入玉関連局面
評価が不安定な局面
policy entropy が高い局面
Q 上位手が僅差の局面
通常勝ちと入玉勝ちの選択が割れる局面
自己対局で負けにつながった疑いのある局面
```

### 9.2 MCTS node

MCTS node は、最低限以下を持つ。

```python
@dataclass
class SearchNode:
    position: "Position"
    side_to_move: int
    legal_moves: list[int]

    prior: np.ndarray
    visit_count: np.ndarray

    q_normal_result_sum: np.ndarray
    q_nyugyoku_sum: np.ndarray

    expanded: bool
    terminal: bool
    terminal_result: "TerminalResult | None"
```

内部統計の正本は `q_normal_result_sum` と `q_nyugyoku_sum` である。

`q_total_result_sum` や `q_mode_sum` は持たず、必要なときに派生値として計算する。

```text
q_total_result_mean = q_normal_result_mean + q_nyugyoku_mean
q_mode_mean = q_normal_result_mean + λ_nyugyoku * q_nyugyoku_mean
```

理由は、`λ_nyugyoku` を変える実験や視点変換で意味を失わないようにするためである。

### 9.3 leaf evaluation

leaf node では network を評価し、以下を得る。

```text
policy prior
V_normal_result
V_nyugyoku
V_total_result
V_mode
```

terminal node では network を評価しない。

leaf value は、root 側視点へ変換して backup する。

### 9.4 視点管理

MCTS では、node の手番が交互に変わる。

したがって、leaf の outcome 分布や value は、root 側視点へ変換してから backup する。

標準は以下とする。

```text
各 leaf で current side 視点の outcome_probs を得る
必要なら root side 視点へ変換する
V_normal_result / V_nyugyoku を root side 視点で計算する
backup_normal_result と backup_nyugyoku を別々に更新する
```

`Q_mode` だけを符号反転して backup してはならない。

### 9.5 root action sampling

Gumbel MCTS では、root action に Gumbel noise を加えて候補手を選ぶ。

概念的には以下である。

```text
score(a) = log(p_prior(a)) + gumbel(a) + value_score(a)
```

その後、限られた simulation budget を Sequential Halving 的に配分する。

初期実装では、mctx の実装を参考にしつつ、以下を満たせばよい。

```text
合法手上だけで root candidate を作る
visit distribution または improved policy を返す
simulation budget が小さくても破綻しない
KLENT-0 と比較できるログを保存する
```

### 9.6 action selection

Gumbel MCTS の最終方策は、以下のどちらかとする。

```text
visit_counts を温度変換した分布
Gumbel MCTS の improved policy
```

自己対局 reanalyse の target には、分布として保存する。

評価対局や実戦時探索では、原則として最善手を選ぶ。

### 9.7 budget tier

実戦時探索の budget tier は以下とする。

```text
klent0:
  0 simulation

small:
  数十 simulation

medium:
  数百 simulation

large:
  それ以上
```

具体値は実験で決める。

初期実装では、まず `klent0` と `small` の差を測る。

### 9.8 reanalyse 出力

Gumbel MCTS reanalyse では、以下を保存する。

```text
pi_reanalyse
Q_normal_result_reanalyse
Q_nyugyoku_reanalyse
outcome_cause_reanalyse
visit_counts
search_config_id
```

`pi_reanalyse` は policy target として使う。

`Q_normal_result_reanalyse` と `Q_nyugyoku_reanalyse` は action-value target として使う。

---

## 10. Monte-Carlo Graph Search

### 10.1 位置づけ

将棋では transposition が発生するため、同一局面への合流が重要になる場合がある。

その場合、tree ではなく graph として扱う Monte-Carlo Graph Search を検討する。

ただし、初期実装の標準にはしない。

理由は以下である。

```text
実装が複雑になる
視点管理と backup が難しくなる
NN cache や repetition state との関係が複雑になる
まず KLENT-0 と Gumbel MCTS の効果を見たい
```

### 10.2 導入条件

導入を検討する条件は以下である。

```text
Gumbel MCTS で同一局面への再訪問が多い
NN cache hit rate が高い
同じ局面を別 move order で何度も読んでいる
探索時間の多くが重複評価に使われている
```

### 10.3 注意点

graph 化する場合でも、以下は守る。

```text
局面 key に手番を含める
feature_version と model_version を含める
repetition state をどう扱うか明示する
value 成分を分けて保存する
Q_mode だけを graph node に保存しない
```

---

## 11. batched inference と NN cache

### 11.1 batched inference

自己対局では、多数局面をまとめて GPU 推論する。

```text
self-play worker
  -> inference queue
  -> GPU batch inference
  -> NetworkOutput
  -> search worker
```

探索側は、1局面ずつ同期的に network を呼び出す実装に固定しない。

初期実装では、同期 API から作ってよいが、内部は batched inference に置き換えられる境界にする。

### 11.2 SearchNetworkOutput

探索側では、生の `NetworkOutput` を以下へ変換する。

```python
@dataclass
class SearchNetworkOutput:
    prior: np.ndarray
    q_normal_result: np.ndarray
    q_nyugyoku: np.ndarray
    q_total_result: np.ndarray
    q_mode: np.ndarray

    outcome_probs: np.ndarray
    v_normal_result: float
    v_nyugyoku: float
    v_total_result: float
    v_mode: float

    moves_left: float
```

`SearchNetworkOutput` は探索モードに依存する。

ただし、raw `NetworkOutput` は探索モードに依存しない。

### 11.3 NN cache

NN cache は初期実装では無効とする。

```python
enable_nn_cache = False
```

Gumbel MCTS や実戦時探索で効果測定してから導入する。

cache key は以下を含める。

```text
盤面
持ち駒
手番
model version
feature version
```

探索モードの `α_style` や `λ_nyugyoku` は cache key に含めない。

理由は、これらは raw network output の後処理であり、network output 自体は変わらないためである。

反復関連特徴を input に含めない初期実装では、repetition state は cache key に含めない。ただし、terminal / draw 判定には repetition state を使うため、game manager 側で別途管理する。

### 11.4 cache のログ

cache を有効にした場合は、以下を記録する。

```text
nn_cache/enabled
nn_cache/hit_rate
nn_cache/key_build_us
nn_cache/lookup_us
nn_cache/miss_inference_us
nn_cache/effective_positions_per_sec
```

---

## 12. 時間管理

### 12.1 自己対局

自己対局では、原則として固定設定を使う。

```text
KLENT-0:
  常に0 simulation

Gumbel MCTS reanalyse:
  固定 simulation budget
```

自己対局の目的は、局面生成と学習 target 収集である。

そのため、複雑な時間管理よりも、再現性とスループットを優先する。

### 12.2 評価対局

評価対局では、条件を固定する。

```text
同じ探索モード
同じ simulation budget
同じ tactical solver 設定
同じ持ち時間または固定ノード数
```

通常勝率、入玉宣言勝ち率、相手入玉負け率などを比較するため、探索条件を混ぜない。

### 12.3 実戦時探索

実戦時には、利用可能な計算資源に応じて以下を選ぶ。

```text
KLENT-0 + CPU tactical solver
Gumbel MCTS small budget
Gumbel MCTS medium budget
Gumbel MCTS large budget
```

持ち時間対応は初期実装では必須にしない。

まず固定 budget で強さと速度を測る。

---

## 13. 探索ログ仕様

### 13.1 KLENT-0 ログ

KLENT-0 では、最低限以下を記録する。

```text
search/type
search/config_id
search/selected_move
search/num_legal_moves

policy/entropy_prior
policy/entropy_klent
policy/kl_klent_to_prior
policy/top1_prior_prob
policy/top1_klent_prob

q/Q_normal_result_selected
q/Q_nyugyoku_selected
q/Q_total_result_selected
q/Q_mode_selected
q/Q_mode_top1
q/Q_mode_gap_top1_top2

outcome/P_win_normal
outcome/P_win_nyugyoku
outcome/P_draw
outcome/P_loss_nyugyoku
outcome/P_loss_normal

value/V_normal_result
value/V_nyugyoku
value/V_total_result
value/V_mode

moves_left/pred

tactical/forced_move
 tactical/masked_move_count
 tactical/reason
```

### 13.2 Gumbel MCTS ログ

Gumbel MCTS では、追加で以下を記録する。

```text
mcts/num_simulations
mcts/num_expanded_nodes
mcts/max_depth
mcts/root_visit_entropy
mcts/root_best_visit_count
mcts/root_best_q_normal_result
mcts/root_best_q_nyugyoku
mcts/root_best_q_mode
mcts/search_time_ms
mcts/nodes_per_sec
mcts/nn_evaluations
mcts/cache_hit_rate
```

### 13.3 入玉関連ログ

入玉関連の探索挙動を確認するため、以下を通常ログに含める。

```text
nyugyoku/lambda_nyugyoku
nyugyoku/alpha_style_self
nyugyoku/alpha_style_opponent_model
nyugyoku/Q_nyugyoku_selected
nyugyoku/P_win_nyugyoku
nyugyoku/P_loss_nyugyoku
nyugyoku/declaration_win_available
nyugyoku/own_king_in_enemy_camp
nyugyoku/opp_king_in_own_camp
```

`own_king_in_enemy_camp` など現局面から計算できる情報は、network output ではなく探索ログ・局面分類ログとして扱う。

---

## 14. 実装上の不変条件

以下は必ず守る。

```text
合法手は探索側で生成する
policy は合法手上でのみ正規化する
違法手確率は必ず0にする
network module 内で合法手 mask を行わない
network module 内で q raw に tanh をかけない
network module 内で KLENT 方策を計算しない
terminal node では network 評価を使わない
宣言勝ち可能判定は探索側のルール処理で行う
CPU tactical solver の強制手は KLENT 方策より優先する
Q_normal_result と Q_nyugyoku は現在手番視点で定義する
MCTS backup では通常勝敗価値と入玉関連価値を分けて扱う
Q_mode だけを保存して視点変換しない
policy_style の混合率は root側と相手側で分ける
root noise は自己対局用であり、評価では原則無効にする
探索モードの α_style や λ_nyugyoku は NN cache key に含めない
```

---

## 15. 検証項目

### 15.1 合法手 mask test

最低限、以下を確認する。

```text
合法手以外の policy 確率が0になる
合法手上の policy 確率和が1になる
合法手以外の KLENT 確率が0になる
合法手上の KLENT 確率和が1になる
合法手が1つだけの場合、その手の確率が1になる
```

### 15.2 policy blend test

以下を確認する。

```text
α_style = 0 のとき p_prior が p_normal と一致する
α_style = 1 のとき p_prior が p_style と一致する
0 < α_style < 1 のとき合法手上で確率和が1になる
logits 直接混合をしていない
```

### 15.3 KLENT test

以下を確認する。

```text
pi_klent の確率和が1になる
違法手確率が0になる
prior が0に近くても NaN が出ない
Q_mode が高い手の確率が上がる
β を大きくすると prior に近づく
α を大きくすると分布が広がる
```

### 15.4 tactical solver integration test

以下を確認する。

```text
詰み手がある場合、その手を選ぶ
宣言勝ち可能な場合、宣言勝ちを選ぶ
winning_moves がある場合、それ以外が mask される
補正後の分布の確率和が1になる
```

### 15.5 terminal test

以下を確認する。

```text
terminal node で network inference が呼ばれない
合法手なし局面が terminal として処理される
千日手が draw として処理される
最大手数到達が設定通りに処理される
宣言勝ち条件が正しく判定される
```

### 15.6 Gumbel MCTS test

以下を確認する。

```text
合法手上だけで root action sampling する
指定 simulation 数を超えない
terminal leaf で network inference が呼ばれない
backup_normal_result と backup_nyugyoku が別々に更新される
root visit distribution の確率和が1になる
KLENT-0 と同じ network 変換関数を使う
```

### 15.7 視点変換 test

以下を確認する。

```text
現在手番視点の outcome 分布を相手視点へ変換できる
視点変換を2回適用すると元に戻る
MCTS leaf value が root 視点で backup される
Q_mode だけを単純符号反転していない
```

---

## 16. 実験項目

### 16.1 KLENT parameter 実験

以下を比較する。

```text
klent_alpha_entropy
klent_beta_prior
temperature
root_noise_frac
```

観測する指標は以下である。

```text
通常勝率
policy entropy
KL(pi_klent || p_prior)
短手数負け率
自己対局局面多様性
入玉関連局面到達率
```

### 16.2 探索モード実験

以下を比較する。

```text
normal
nyugyoku
anti_nyugyoku
mutual_nyugyoku
bakuretsu
```

観測する指標は以下である。

```text
通常勝率
入玉宣言勝ち率
相手入玉負け率
相入玉局面での勝率
通常勝ちできる局面で無理に入玉しない率
```

### 16.3 tactical solver ablation

以下を比較する。

```text
KLENT-0 only
KLENT-0 + 1手詰め検出
KLENT-0 + 有界詰み探索
KLENT-0 + 王手対応
KLENT-0 + 宣言勝ち判定
KLENT-0 + full tactical solver
```

観測する指標は以下である。

```text
詰み逃し率
短手数負け率
通常勝率
探索速度
```

### 16.4 Gumbel MCTS budget 実験

以下を比較する。

```text
KLENT-0
Gumbel MCTS small
Gumbel MCTS medium
Gumbel MCTS large
```

観測する指標は以下である。

```text
通常勝率
入玉宣言勝ち率
相手入玉負け率
KLENT-0 からの改善幅
positions/sec
nodes/sec
NN evaluations per move
```

### 16.5 reanalyse 対象選別実験

以下の対象局面選別を比較する。

```text
policy entropy 高
Q 上位手僅差
入玉関連局面
終盤不安定局面
自己対局で負けにつながった疑いのある局面
```

観測する指標は以下である。

```text
再解析後 policy target の品質
q target の安定性
通常勝率への寄与
入玉指標への寄与
reanalyse 計算効率
```

---

## 17. 未決事項

### 17.1 KLENT parameter の初期値

以下は実験で確定する。

```text
klent_alpha_entropy
klent_beta_prior
temperature
root_dirichlet_alpha
root_noise_frac
```

初期値は、分布が極端に collapse しないことを優先して決める。

### 17.2 moves_left 補正

`moves_left` を探索価値に入れるかは未決である。

標準では無効とする。

入れる場合も、通常勝率と入玉長期計画への悪影響を確認してからにする。

### 17.3 Gumbel MCTS の具体的な schedule

以下は未決である。

```text
small / medium / large の simulation 数
Sequential Halving の round 数
root candidate 数
Gumbel noise の扱い
visit distribution か improved policy のどちらを target にするか
```

初期実装では、mctx の方針を参考にしつつ、KLENT-0 との差を測れる最小構成から始める。

### 17.4 Monte-Carlo Graph Search

graph search は初期実装には含めない。

Gumbel MCTS の重複評価が明確にボトルネックになった場合に検討する。

### 17.5 相手モデル設定

自己対局で、探索木内の相手側 node に相手プレイヤーの実際の探索設定をどこまで反映するかは未決である。

初期実装では、`alpha_style_opponent_model` による近似でよい。

---

## 18. 参考にする既存研究・実装

### 18.1 KLENT

ここでいう KLENT は、KL regularization と entropy regularization を使う 0-playout 方策改善の系統を指す。

参考:

```text
https://arxiv.org/abs/2602.10894
https://arxiv.org/html/2602.10894v1
```

参考にする点は以下である。

```text
KL regularization と entropy regularization を使った方策改善
有限行動空間での closed-form policy improvement
search-free / low-resource 学習方針
λ-return を使った action-value 学習
```

爆裂玉登太郎では、KLENT の考え方を将棋向けに変更し、KLENT-0 を主学習ループの着手選択として使う。

### 18.2 Gumbel AlphaZero / Gumbel MuZero / Gumbel MCTS

参考:

```text
https://openreview.net/forum?id=bERaNdoegnO
https://github.com/google-deepmind/mctx
https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py
https://github.com/google-deepmind/mctx/blob/main/examples/policy_improvement_demo.py
```

参考にする点は以下である。

```text
Gumbel を使った root action sampling
Sequential Halving による simulation budget 配分
少数 simulation での policy improvement
MCTS を policy improvement operator として使う考え方
```

爆裂玉登太郎では、Gumbel MCTS を全局面の主自己対局には使わず、reanalyse、難所局面解析、実戦時探索に使う。

### 18.3 dlshogi

参考:

```text
https://github.com/TadaoYamaoka/DeepLearningShogi
https://github.com/TadaoYamaoka/DeepLearningShogi/blob/ae0cc8fbf5d3f2f0d010685ebc5eaba13b961999/cppshogi/cppshogi.h
https://github.com/TadaoYamaoka/DeepLearningShogi/blob/master/dlshogi/network/policy_value_network_exp___i.py
```

参考にする点は以下である。

```text
将棋向け合法手ラベル
policy head の move label plane 形式
入玉関連特徴
GPU batch 推論を前提にした policy/value network 実装
```

### 18.4 YaneuraOu

参考:

```text
https://github.com/yaneurao/YaneuraOu
```

参考にする点は以下である。

```text
将棋エンジンとしての合法手生成
詰み探索
探索ログ
USI エンジン実装
```

爆裂玉登太郎では、αβ探索を主探索として採用することは必須ではないが、合法手生成、詰み周辺、エンジン実装の参考にする。

---

## 19. 暫定結論

爆裂玉登太郎の標準探索は以下とする。

```text
KLENT-0
+
CPU tactical solver
```

着手選択の中心は以下である。

```text
p_prior = blend(policy_normal, policy_style)
Q_mode  = Q_normal_result + λ_nyugyoku * Q_nyugyoku
pi_klent = KLENT_policy_improvement(p_prior, Q_mode)
```

探索モードは以下で制御する。

```text
alpha_style_self
alpha_style_opponent_model
lambda_nyugyoku
klent_alpha_entropy
klent_beta_prior
temperature
```

短期戦術とルール処理は network に任せず、探索側で扱う。

```text
terminal 判定
宣言勝ち可能判定
相手玉への詰み探索
王手対応
```

Gumbel MCTS は、主自己対局の全局面ではなく、以下に使う。

```text
reanalyse
難所局面解析
実戦時探索
KLENT-0 との差分評価
```

最重要の不変条件は以下である。

```text
合法手上だけで policy / KLENT を正規化する
CPU tactical solver の強制手は KLENT より優先する
Q_normal_result と Q_nyugyoku を分けて扱う
Q_mode だけを保存して視点変換しない
terminal node では network 評価を使わない
```

# docs/06_search_value.md - 探索価値・視点変換仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎における **探索価値の組み立て方** と **視点変換** を定義する。

対象は以下である。

```text
network 出力から探索用 value への変換
V_normal_result / V_nyugyoku / V_total_result / V_mode
Q_normal_result / Q_nyugyoku / Q_total_result / Q_mode
現在手番視点・親視点・root視点の扱い
KLENT-0 で使う action-value
Gumbel MCTS で使う leaf value と backup
reanalyse target と value ログ
moves_left の扱い
CPU tactical solver との境界
実装上の不変条件
検証項目
実験項目
```

関連仕様は以下を正とする。

```text
全体方針: 爆裂玉登太郎_設計概要.md
探索・着手選択: docs/03_search.md
ネットワーク出力: docs/05_outputs.md
CPU tactical solver: docs/07_tactical_solver.md
自己対局ループ: docs/08_selfplay.md
学習ターゲット: docs/10_training_targets.md
KLENT-first 学習方針: docs/13_klent_training.md
```

このドキュメントでは、network の head shape や loss は定義しない。それらは `docs/05_outputs.md` と `docs/10_training_targets.md` を正とする。

---

## 1. 基本方針

### 1.1 value は成分別に扱う

爆裂玉登太郎では、探索価値を単一の勝敗 scalar として直接扱わない。

value は以下の2成分を基本にする。

```text
normal_result:
  通常勝ち・通常負けに関する価値

nyugyoku:
  自分の入玉勝ちと、相手入玉による負けに関する価値
```

局面単位では、`outcome_cause_head` から以下を作る。

```text
V_normal_result
V_nyugyoku
V_total_result
V_mode
```

合法手単位では、action-value head から以下を作る。

```text
Q_normal_result(s, a)
Q_nyugyoku(s, a)
Q_total_result(s, a)
Q_mode(s, a)
```

`V_total_result` と `Q_total_result` は通常勝敗としての総合価値である。

`V_mode` と `Q_mode` は、探索モードごとに入玉成分を重みづけした価値である。

### 1.2 Q_mode は派生値である

`Q_mode` は探索モードごとの着手選択に使う派生値である。

```text
Q_mode = Q_normal_result + λ_nyugyoku * Q_nyugyoku
```

`Q_mode` は保存してよいが、正本にはしない。

保存・backup・reanalyse・target 生成では、必ず以下を分けて扱う。

```text
Q_normal_result
Q_nyugyoku
```

`Q_mode` だけを保存して、あとから符号反転したり、別の `λ_nyugyoku` で再利用したりしてはならない。

### 1.3 視点を明示する

すべての value は、どの視点かを明示して扱う。

標準で扱う視点は以下である。

```text
current-side view:
  その局面の手番側から見た値

parent-side view:
  親 node の手番側から見た子局面の値

root-side view:
  探索 root で着手を選ぶ側から見た値
```

network 出力は常に current-side view である。

KLENT-0 は root 局面だけを評価するため、通常は current-side view と root-side view が一致する。

Gumbel MCTS では node の手番が交互に変わるため、leaf value を root-side view に変換してから backup する。

### 1.4 単一 scalar の符号反転を避ける

`V_mode` や `Q_mode` だけを単純に符号反転する実装は避ける。

特に `λ_nyugyoku` は探索モードで変わるため、`Q_mode` を正本として持つと、以下ができなくなる。

```text
λ_nyugyoku を変えた再評価
normal_result と nyugyoku の別々の calibration
入玉勝ちと通常勝ちの寄与分解
Gumbel MCTS reanalyse target の成分別保存
```

視点変換は、原則として outcome 分布を変換してから value を再計算する。

---

## 2. class order と value 成分

### 2.1 outcome class order

`outcome_cause_head` の class order は固定で以下とする。

```python
OUTCOME_WIN_NORMAL = 0
OUTCOME_WIN_NYUGYOKU = 1
OUTCOME_DRAW = 2
OUTCOME_LOSS_NYUGYOKU = 3
OUTCOME_LOSS_NORMAL = 4
OUTCOME_NUM_CLASSES = 5
```

この順序は checkpoint、ログ、reanalyse target、unit test の互換性に関わるため、途中で変更してはならない。

### 2.2 outcome probabilities

softmax 後の確率を以下で表す。

```text
P_win_normal    = P(class = OUTCOME_WIN_NORMAL)
P_win_nyugyoku  = P(class = OUTCOME_WIN_NYUGYOKU)
P_draw          = P(class = OUTCOME_DRAW)
P_loss_nyugyoku = P(class = OUTCOME_LOSS_NYUGYOKU)
P_loss_normal   = P(class = OUTCOME_LOSS_NORMAL)
```

確率は必ず以下を満たす。

```text
P_win_normal
+ P_win_nyugyoku
+ P_draw
+ P_loss_nyugyoku
+ P_loss_normal
= 1
```

### 2.3 局面 value 成分

通常勝敗成分は以下である。

```text
V_normal_result = P_win_normal - P_loss_normal
```

入玉成分は以下である。

```text
V_nyugyoku = P_win_nyugyoku - P_loss_nyugyoku
```

総合勝敗価値は以下である。

```text
V_total_result = V_normal_result + V_nyugyoku
```

探索モード価値は以下である。

```text
V_mode = V_normal_result + λ_nyugyoku * V_nyugyoku
```

`λ_nyugyoku = 1.0` なら、通常勝ちと入玉勝ちを同じ勝敗価値として扱う。

`λ_nyugyoku > 1.0` なら、自分の入玉勝ちを強く好み、相手入玉による負けを強く嫌う。

### 2.4 action-value 成分

network の raw output は以下である。

```python
q_normal_result_raw: Tensor  # [B, MOVE_LABEL_SIZE]
q_nyugyoku_raw: Tensor       # [B, MOVE_LABEL_SIZE]
```

探索側では、合法手 mask 後に以下へ変換する。

```text
Q_normal_result = tanh(q_normal_result_raw)
Q_nyugyoku      = tanh(q_nyugyoku_raw)
```

総合勝敗 action-value は以下である。

```text
Q_total_result = Q_normal_result + Q_nyugyoku
```

探索モード action-value は以下である。

```text
Q_mode = Q_normal_result + λ_nyugyoku * Q_nyugyoku
```

`Q_normal_result` と `Q_nyugyoku` は、合法手ごと、かつ current-side view で定義する。

### 2.5 value range

`Q_normal_result` と `Q_nyugyoku` はそれぞれ `[-1, 1]` の範囲である。

ただし、和である `Q_total_result` と `Q_mode` は `[-1, 1]` に収まるとは限らない。

```text
Q_total_result ∈ [-2, 2]
Q_mode ∈ [-(1 + λ_nyugyoku), 1 + λ_nyugyoku]
```

KLENT や Gumbel MCTS の value scale は、この範囲を前提に調整する。

必要なら `Q_mode` を探索アルゴリズム側で temperature / scale するが、標準では raw な成分和を使う。

---

## 3. 探索用変換 API

### 3.1 SearchValueConfig

探索価値の変換設定は、探索モード設定から value 部分を取り出した構造として扱う。

```python
@dataclass
class SearchValueConfig:
    lambda_nyugyoku: float = 1.0
    value_clip: float | None = None
    use_moves_left_adjustment: bool = False
    moves_left_win_bonus_scale: float = 0.0
    moves_left_loss_delay_scale: float = 0.0
```

初期実装では以下を標準とする。

```text
lambda_nyugyoku:
  SearchModeConfig に従う

value_clip:
  None

use_moves_left_adjustment:
  False
```

### 3.2 OutcomeValue

局面単位の value は以下の構造体相当で扱う。

```python
@dataclass
class OutcomeValue:
    probs: np.ndarray  # [5]

    v_normal_result: float
    v_nyugyoku: float
    v_total_result: float
    v_mode: float

    perspective_side: int
```

`perspective_side` は、この value がどちらの側から見た値かを表す。

network 直後は `perspective_side = position.side_to_move` である。

### 3.3 ActionValue

合法手単位の action-value は以下の構造体相当で扱う。

```python
@dataclass
class ActionValue:
    legal_moves: list[int]

    q_normal_result: np.ndarray  # [MOVE_LABEL_SIZE] or sparse
    q_nyugyoku: np.ndarray
    q_total_result: np.ndarray
    q_mode: np.ndarray

    perspective_side: int
```

内部実装は dense でも sparse でもよい。

ただし、以下を必ず満たす。

```text
合法手以外を着手選択に使わない
sparse の moves は legal_moves の部分集合である
q_normal_result と q_nyugyoku の順序は legal_moves と一致する
perspective_side を欠落させない
```

### 3.4 ConvertedNetworkOutput

policy 変換も含めた探索側の変換結果は以下の構造体相当とする。

```python
@dataclass
class ConvertedNetworkOutput:
    legal_moves: list[int]

    p_normal: np.ndarray
    p_style: np.ndarray
    p_prior: np.ndarray

    action_value: ActionValue
    outcome_value: OutcomeValue

    moves_left: float | None
```

`p_prior` は `policy_normal_head` と `policy_style_head` を合法手上で softmax した後、探索モードに応じて混合した分布である。

`action_value.q_mode` は KLENT-0 root 着手選択の主 value として使う。

`outcome_value.v_mode` は Gumbel MCTS leaf evaluation、calibration、局面分類、λ-return 用に使う。

### 3.5 変換関数

疑似コードは以下である。

```python
def convert_network_output_for_search(
    network_output: NetworkOutput,
    legal_moves: list[int],
    node_side: int,
    root_side: int,
    search_config: SearchModeConfig,
) -> ConvertedNetworkOutput:
    alpha_style = select_alpha_style(
        node_side=node_side,
        root_side=root_side,
        config=search_config,
    )

    p_normal = softmax_legal(network_output.policy_normal_logits, legal_moves)
    p_style = softmax_legal(network_output.policy_style_logits, legal_moves)
    p_prior = normalize((1.0 - alpha_style) * p_normal + alpha_style * p_style)

    q_normal = tanh(network_output.q_normal_result_raw)
    q_nyugyoku = tanh(network_output.q_nyugyoku_raw)
    q_normal = mask_values_to_legal(q_normal, legal_moves)
    q_nyugyoku = mask_values_to_legal(q_nyugyoku, legal_moves)

    q_total = q_normal + q_nyugyoku
    q_mode = q_normal + search_config.lambda_nyugyoku * q_nyugyoku

    outcome_probs = softmax(network_output.outcome_cause_logits)
    outcome_value = make_outcome_value(
        probs=outcome_probs,
        lambda_nyugyoku=search_config.lambda_nyugyoku,
        perspective_side=node_side,
    )

    moves_left = expm1(network_output.moves_left_pred)

    return ConvertedNetworkOutput(
        legal_moves=legal_moves,
        p_normal=p_normal,
        p_style=p_style,
        p_prior=p_prior,
        action_value=ActionValue(
            legal_moves=legal_moves,
            q_normal_result=q_normal,
            q_nyugyoku=q_nyugyoku,
            q_total_result=q_total,
            q_mode=q_mode,
            perspective_side=node_side,
        ),
        outcome_value=outcome_value,
        moves_left=moves_left,
    )
```

---

## 4. 視点変換

### 4.1 視点変換の基本

network は常に current-side view で出力する。

局面 `s` の手番が `A` のとき、network の value は `A` から見た値である。

親局面の手番が `B` で、子局面の手番が `A` である場合、通常は `A != B` である。

そのため、子局面の outcome 分布を親視点または root 視点に変換してから value を計算する。

### 4.2 outcome 分布の反転

相手視点へ変換する場合、outcome 分布は以下で変換する。

```text
flipped_P_win_normal    = P_loss_normal
flipped_P_win_nyugyoku  = P_loss_nyugyoku
flipped_P_draw          = P_draw
flipped_P_loss_nyugyoku = P_win_nyugyoku
flipped_P_loss_normal   = P_win_normal
```

疑似コードは以下である。

```python
def flip_outcome_probs(probs: np.ndarray) -> np.ndarray:
    return np.array([
        probs[OUTCOME_LOSS_NORMAL],
        probs[OUTCOME_LOSS_NYUGYOKU],
        probs[OUTCOME_DRAW],
        probs[OUTCOME_WIN_NYUGYOKU],
        probs[OUTCOME_WIN_NORMAL],
    ], dtype=probs.dtype)
```

この変換を2回適用すると元の分布に戻る。

```text
flip_outcome_probs(flip_outcome_probs(probs)) == probs
```

### 4.3 side への変換

任意の視点へ変換する関数は以下とする。

```python
def transform_outcome_probs_to_side(
    probs: np.ndarray,
    from_side: int,
    to_side: int,
) -> np.ndarray:
    if from_side == to_side:
        return probs
    return flip_outcome_probs(probs)
```

将棋は2人零和なので、`from_side != to_side` なら反転でよい。

ただし、反復状態や最大手数などの rule state は network の outcome 分布には含まれないため、terminal / draw 判定済みの場合は network 出力ではなく rule result を正とする。

### 4.4 成分 value の反転

成分 value で反転する場合は以下になる。

```text
flipped_V_normal_result = -V_normal_result
flipped_V_nyugyoku      = -V_nyugyoku
flipped_V_total_result  = -V_total_result
```

`V_mode` は `λ_nyugyoku` が同じなら以下で計算してよい。

```text
flipped_V_mode = flipped_V_normal_result + λ_nyugyoku * flipped_V_nyugyoku
```

ただし、標準実装では outcome 分布を変換してから value を再計算する。

理由は以下である。

```text
class order のバグを検出しやすい
calibration を class 単位で確認できる
reanalyse target と outcome target を同じ経路で扱える
```

### 4.5 action-value の視点

`Q_normal_result(s, a)` と `Q_nyugyoku(s, a)` は、局面 `s` の current-side view である。

root 局面で KLENT-0 を使う場合は、root の current-side view がそのまま root-side view であるため、特別な変換は不要である。

MCTS 内で edge value を backup する場合は、leaf value を root-side view へ変換してから、root action に紐づく統計へ加算する。

`Q_mode` だけを edge に保存しない。

### 4.6 terminal value の視点

terminal node では network を評価しない。

terminal result から outcome 分布相当を作る場合は、必要な視点で直接作る。

例:

```python
def terminal_result_to_outcome_probs(result, perspective_side):
    if result.is_draw:
        return one_hot(OUTCOME_DRAW)

    is_win = (result.winner == perspective_side)

    if is_win and result.outcome_category == "normal":
        return one_hot(OUTCOME_WIN_NORMAL)
    if is_win and result.outcome_category == "nyugyoku":
        return one_hot(OUTCOME_WIN_NYUGYOKU)
    if (not is_win) and result.outcome_category == "nyugyoku":
        return one_hot(OUTCOME_LOSS_NYUGYOKU)
    if (not is_win) and result.outcome_category == "normal":
        return one_hot(OUTCOME_LOSS_NORMAL)

    raise ValueError(result)
```

terminal value も `V_mode` だけに潰さず、必要に応じて outcome 分布、`V_normal_result`、`V_nyugyoku` を作る。

---

## 5. KLENT-0 での value 利用

### 5.1 KLENT-0 の value 入力

KLENT-0 では、root 局面の合法手ごとの `Q_mode` を使う。

```text
Q_mode(s, a) = Q_normal_result(s, a) + λ_nyugyoku * Q_nyugyoku(s, a)
```

`V_mode` は KLENT-0 の root 着手選択には直接使わない。

`V_mode` は以下に使う。

```text
ログ
calibration 確認
局面分類
λ-return 作成
Gumbel MCTS の leaf value
```

### 5.2 KLENT 改善方策

KLENT 改善方策は合法手上で以下とする。

```text
pi_klent(a | s) = softmax_legal(
    (Q_mode(s, a) + β * log(max(p_prior(a | s), ε))) / (α + β)
)
```

ここで、`p_prior` は `policy_normal_head` と `policy_style_head` の混合後分布である。

`α` は entropy regularization、`β` は prior からの乖離抑制である。

### 5.3 KLENT 用 value の数値安定化

実装では以下を満たす。

```text
合法手以外の Q_mode は選択に使わない
合法手以外の prior は 0 にする
prior は合法手上で再正規化する
log_prior は log(max(prior, ε)) とする
α + β > 0 を assert する
pi_klent は合法手上で確率和 1 にする
```

疑似コードは以下である。

```python
def compute_klent_policy(
    p_prior: np.ndarray,
    q_mode: np.ndarray,
    legal_moves: list[int],
    alpha_entropy: float,
    beta_prior: float,
    eps: float,
) -> np.ndarray:
    assert alpha_entropy + beta_prior > 0.0

    p = mask_and_renormalize(p_prior, legal_moves)
    log_p = np.log(np.maximum(p, eps))

    logits = (q_mode + beta_prior * log_p) / (alpha_entropy + beta_prior)
    logits = mask_illegal_logits(logits, legal_moves)
    return softmax(logits)
```

### 5.4 tactical solver との境界

CPU tactical solver の forced move や mask は、`Q_mode` を書き換えず、方策分布に反映する。

保存する値は以下を分ける。

```text
q_normal_result_pre_tactical
q_nyugyoku_pre_tactical
q_total_result_pre_tactical
q_mode_pre_tactical
pi_klent_pre_tactical
pi_search_post_tactical
```

宣言勝ちや詰みが証明された場合でも、network / KLENT 由来の `Q_mode` は解析用に残す。

学習 target に tactical 情報を使う場合は、`docs/10_training_targets.md` の tactical forced target の規則に従う。

### 5.5 KLENT-0 の SearchResult

KLENT-0 の `SearchResult` には、最低限以下を含める。

```python
@dataclass
class SearchResultValueFields:
    q_normal_result: SparseMoveValues | None
    q_nyugyoku: SparseMoveValues | None
    q_total_result: SparseMoveValues | None
    q_mode: SparseMoveValues | None

    v_normal_result: float | None
    v_nyugyoku: float | None
    v_total_result: float | None
    v_mode: float | None

    q_normal_result_selected: float | None
    q_nyugyoku_selected: float | None
    q_total_result_selected: float | None
    q_mode_selected: float | None
```

`selected_move` に対応する value は、ログ集計と reanalyse candidate 抽出に使う。

---

## 6. Gumbel MCTS での value 利用

### 6.1 基本方針

Gumbel MCTS は、再解析、難所局面解析、実戦時探索に使う。

Gumbel MCTS でも、network 出力の変換は KLENT-0 と同じ関数を使う。

ただし、MCTS では leaf value を backup するため、視点変換が必須になる。

### 6.2 node 統計

MCTS node は、少なくとも以下の value 統計を分けて持つ。

```python
@dataclass
class SearchNodeStats:
    visit_count: np.ndarray

    q_normal_result_sum: np.ndarray
    q_nyugyoku_sum: np.ndarray

    q_normal_result_mean: np.ndarray
    q_nyugyoku_mean: np.ndarray
```

`q_total_result_mean` と `q_mode_mean` は派生値として計算する。

```text
q_total_result_mean = q_normal_result_mean + q_nyugyoku_mean
q_mode_mean = q_normal_result_mean + λ_nyugyoku * q_nyugyoku_mean
```

`λ_nyugyoku` を変更した実験を行えるように、`q_mode_sum` だけを保存しない。

### 6.3 leaf evaluation

leaf node では network を評価し、current-side view の outcome 分布を得る。

その後、root-side view へ変換する。

```python
def evaluate_leaf_for_backup(
    position: Position,
    network_output: NetworkOutput,
    root_side: int,
    config: SearchModeConfig,
) -> BackupValue:
    current_side = position.side_to_move
    probs_current = softmax(network_output.outcome_cause_logits)
    probs_root = transform_outcome_probs_to_side(
        probs=probs_current,
        from_side=current_side,
        to_side=root_side,
    )
    value_root = make_outcome_value(
        probs=probs_root,
        lambda_nyugyoku=config.lambda_nyugyoku,
        perspective_side=root_side,
    )
    return BackupValue(
        normal_result=value_root.v_normal_result,
        nyugyoku=value_root.v_nyugyoku,
    )
```

backup するのは以下である。

```text
backup_normal_result
backup_nyugyoku
```

`backup_mode` だけを backup しない。

### 6.4 terminal leaf

terminal leaf では network を評価しない。

terminal result から root-side view の outcome 分布を作り、`V_normal_result` と `V_nyugyoku` を計算して backup する。

```text
checkmate / resign / normal_adjudication:
  normal_result 成分で勝敗を表す

declaration_win / nyugyoku_adjudication:
  nyugyoku 成分で勝敗を表す

repetition_draw / jishogi_draw / max_ply_draw:
  両成分 0
```

### 6.5 root action の action-value target

Gumbel MCTS reanalyse では、root の合法手ごとに以下を保存する。

```text
Q_normal_result_reanalyse
Q_nyugyoku_reanalyse
```

これらは root-side view、つまり root の current-side view に揃える。

`Q_mode_reanalyse` を保存してもよいが、派生値である。

学習 target の正本にはしない。

### 6.6 selection score

Gumbel MCTS の action selection で使う value score は、探索モードに応じて以下から作る。

```text
q_mode_mean = q_normal_result_mean + λ_nyugyoku * q_nyugyoku_mean
```

root candidate selection や sequential halving の score では `q_mode_mean` を使ってよい。

ただし、node 内部の統計は成分別に保存する。

### 6.7 backup の疑似コード

```python
def backup_path(path: list[EdgeRef], backup: BackupValue):
    for edge in path:
        edge.visit_count += 1
        edge.q_normal_result_sum += backup.normal_result
        edge.q_nyugyoku_sum += backup.nyugyoku
        edge.q_normal_result_mean = edge.q_normal_result_sum / edge.visit_count
        edge.q_nyugyoku_mean = edge.q_nyugyoku_sum / edge.visit_count
```

標準は、leaf value を root-side view に変換してから path 上へ同じ値を backup する方式とする。

node-side view で各 depth ごとに符号を変えながら backup する方式も可能だが、初期実装では採用しない。

理由は以下である。

```text
root action の reanalyse target を作りやすい
通常成分と入玉成分を同じ視点でログできる
Q_mode の単純符号反転バグを避けやすい
```

---

## 7. reanalyse target との関係

### 7.1 reanalyse target の正本

Gumbel MCTS reanalyse で保存する value target は以下である。

```text
q_normal_result_reanalyse
q_nyugyoku_reanalyse
outcome_cause_reanalyse
```

`q_total_result_reanalyse` と `q_mode_reanalyse` は派生値である。

保存する場合も、学習 target の正本にはしない。

### 7.2 perspective metadata

reanalyse target には必ず視点 metadata を保存する。

```python
@dataclass
class ReanalyseValueTarget:
    moves: list[int]
    q_normal_result: list[float]
    q_nyugyoku: list[float]

    perspective_side: int
    root_side: int
    side_to_move: int

    lambda_nyugyoku_used_for_search: float
    search_config_id: str
    num_simulations: int
```

TrainingSample へ変換する時点で、`perspective_side` はサンプル局面の `side_to_move` に揃える。

### 7.3 outcome_cause_reanalyse

Gumbel MCTS の root value から `outcome_cause_reanalyse` を作る場合は、以下のどちらかにする。

```text
A. search による root outcome 分布推定がある場合、それを保存する
B. outcome 分布推定がない場合、outcome_cause_reanalyse は None にする
```

`V_mode` だけから5クラス分布を逆算してはならない。

### 7.4 KLENT-0 との差分

reanalyse では、KLENT-0 と Gumbel MCTS の差分を分析するため、以下を保存する。

```text
q_normal_result_klent_pred
q_nyugyoku_klent_pred
q_normal_result_reanalyse
q_nyugyoku_reanalyse
pi_klent_pre_tactical
pi_reanalyse
selected_move_klent
best_move_reanalyse
```

差分が大きい局面は、priority を上げて学習に回してよい。

---

## 8. moves_left の扱い

### 8.1 標準では主価値に入れない

`moves_left_head` は補助予測であり、初期実装では主探索価値に入れない。

```text
V_mode に moves_left 補正を足さない
Q_mode に moves_left 補正を足さない
KLENT 方策に moves_left を直接入れない
```

理由は、入玉は長期計画になりやすく、単純に短い勝ちや短い終局を好む補正を入れると、入玉ルートを不当に嫌う可能性があるためである。

### 8.2 使う場合の小補正

moves_left を使う場合も、小さい tie-break 補正に留める。

候補は以下である。

```text
勝ち寄り局面:
  早い勝ちをわずかに好む

負け寄り局面:
  遅い負けをわずかに好む

入玉価値が高い局面:
  長期化を過度に嫌わない
```

初期実装では無効である。

### 8.3 ログ用途

moves_left は以下に使う。

```text
終盤性の分析
局面バッファ分類
reanalyse candidate 抽出
value calibration の補助
長期入玉ルートの可視化
```

`moves_left_pred` は `log1p(moves_left)` の予測であるため、探索側で使う場合は以下で戻す。

```text
moves_left = expm1(moves_left_pred)
```

---

## 9. CPU tactical solver との関係

### 9.1 役割分担

探索価値は、長期的な手の優劣を表す。

CPU tactical solver は、短期戦術とルール上の確定手を扱う。

したがって、以下を分ける。

```text
Q_mode:
  network / search が推定した長期価値

pi_search_post_tactical:
  tactical solver 補正後に実際の着手選択へ使う分布
```

### 9.2 forced move

以下がある場合、`Q_mode` より優先して着手する。

```text
宣言勝ち可能手
相手玉への詰みが証明された手
王手中の唯一合法応手
```

forced move を返す場合でも、可能なら network inference の結果をログに残してよい。

ただし、forced move によって `Q_normal_result` や `Q_nyugyoku` の値を書き換えない。

### 9.3 tactical target

tactical solver 由来で q target にしてよいのは、局所的に結果が確定するものに限定する。

```text
declaration_win:
  q_nyugyoku_target = +1

mate_win:
  q_normal_result_target = +1
```

自玉即死手の除外や危険手の推定を標準の q target として扱わない。

### 9.4 mask 異常

post-check の tactical mask で合法手が空になった場合は実装エラーである。

```text
tactical mask removed all moves:
  raise TacticalSolverError
  対局を通常 target に使わない
```

この場合、`Q_mode` の問題として処理しない。

---

## 10. calibration と整合性

### 10.1 V と Q の関係

`outcome_cause_head` 由来の `V_*` と、action-value head 由来の `Q_*` は、同じ成分体系に従う。

ただし、役割は異なる。

```text
V_*:
  局面全体の価値
  Gumbel MCTS leaf value
  calibration
  λ-return bootstrap
  局面分類

Q_*:
  合法手ごとの価値
  KLENT-0 root 着手選択
  Gumbel MCTS root action target
  action-value 学習
```

### 10.2 consistency check

同一局面で、policy prior に基づく期待 Q と V は大きく乖離しすぎないことが望ましい。

解析用に以下を計算してよい。

```text
E_Q_normal_result = Σ_a p_prior(a) * Q_normal_result(a)
E_Q_nyugyoku      = Σ_a p_prior(a) * Q_nyugyoku(a)
E_Q_total_result  = E_Q_normal_result + E_Q_nyugyoku

V_normal_result
V_nyugyoku
V_total_result
```

乖離はただちに loss にしない。

まずは calibration 指標としてログに出す。

### 10.3 入玉成分の calibration

入玉関連局面では、以下を別々に確認する。

```text
P_win_nyugyoku の calibration
P_loss_nyugyoku の calibration
Q_nyugyoku の selected move target との誤差
Q_nyugyoku の reanalyse target との誤差
```

通常勝敗成分だけで評価すると、入玉勝ちと相手入玉負けの改善を見落とす。

---

## 11. ログ仕様

### 11.1 共通 value ログ

探索ログには、最低限以下を保存する。

```text
value/p_win_normal
value/p_win_nyugyoku
value/p_draw
value/p_loss_nyugyoku
value/p_loss_normal

value/V_normal_result
value/V_nyugyoku
value/V_total_result
value/V_mode
value/lambda_nyugyoku

q/Q_normal_result_selected
q/Q_nyugyoku_selected
q/Q_total_result_selected
q/Q_mode_selected

q/Q_normal_result_top1
q/Q_nyugyoku_top1
q/Q_total_result_top1
q/Q_mode_top1
q/Q_mode_gap_top1_top2
```

### 11.2 KLENT-0 ログ

KLENT-0 では以下を保存する。

```text
p_prior
pi_klent_pre_tactical
pi_search_post_tactical
pi_select
selected_move

q_normal_result_sparse
q_nyugyoku_sparse
q_total_result_sparse
q_mode_sparse
```

dense で保存するか sparse で保存するかはログ設定で切り替えてよい。

標準は sparse とする。

### 11.3 Gumbel MCTS ログ

Gumbel MCTS では追加で以下を保存する。

```text
visit_counts
pi_reanalyse or pi_search
q_normal_result_search
q_nyugyoku_search
q_total_result_search
q_mode_search
leaf_value_normal_result_mean
leaf_value_nyugyoku_mean
num_simulations
max_depth_reached
```

`q_mode_search` は派生値である。

`q_normal_result_search` と `q_nyugyoku_search` を必ず保存する。

### 11.4 perspective ログ

視点変換バグを検出するため、以下を保存する。

```text
root_side
node_side
value_perspective_side
action_value_perspective_side
outcome_probs_before_transform
outcome_probs_after_transform
```

実運用ログで full distribution を常に保存すると重い場合は、debug mode のみでよい。

---

## 12. 実装上の不変条件

以下は必ず満たす。

```text
network output は current-side view である
outcome class order を変更しない
Q_normal_result と Q_nyugyoku は別々に保持する
Q_mode は派生値である
Q_mode だけを保存して正本にしない
Q_mode だけを単純符号反転しない
Gumbel MCTS backup は normal_result と nyugyoku を別々に行う
reanalyse target は Q_normal_result と Q_nyugyoku を分ける
terminal node では network を評価しない
CPU tactical solver は Q を書き換えず、方策分布または forced move として反映する
moves_left は標準では主探索価値に入れない
合法手以外の policy / Q は着手選択に使わない
```

---

## 13. 検証項目

### 13.1 outcome value test

```text
win_normal one-hot から V_normal_result = +1, V_nyugyoku = 0 が得られる
win_nyugyoku one-hot から V_normal_result = 0, V_nyugyoku = +1 が得られる
loss_nyugyoku one-hot から V_normal_result = 0, V_nyugyoku = -1 が得られる
loss_normal one-hot から V_normal_result = -1, V_nyugyoku = 0 が得られる
draw one-hot から全 value が 0 になる
```

### 13.2 perspective transform test

```text
flip_outcome_probs を2回適用すると元に戻る
win_normal は相手視点で loss_normal になる
win_nyugyoku は相手視点で loss_nyugyoku になる
draw は相手視点でも draw である
変換前後で確率和が 1 のままである
```

### 13.3 action-value test

```text
q raw に tanh が適用される
合法手以外の Q は着手選択に使われない
Q_total_result = Q_normal_result + Q_nyugyoku が成立する
Q_mode = Q_normal_result + λ_nyugyoku * Q_nyugyoku が成立する
λ_nyugyoku を変えると Q_mode だけが変わり、成分 Q は変わらない
```

### 13.4 KLENT value test

```text
Q_mode が高い手の pi_klent が上がる
prior が高い手は beta_prior に応じて維持される
合法手以外の pi_klent は 0 になる
pi_klent の合法手上の確率和が 1 になる
alpha_entropy + beta_prior <= 0 でエラーになる
```

### 13.5 Gumbel MCTS backup test

```text
leaf outcome が root-side view に変換される
backup_normal_result と backup_nyugyoku が別々に更新される
Q_mode だけを backup していない
terminal leaf では network inference が呼ばれない
reanalyse target に Q_normal_result と Q_nyugyoku が含まれる
```

### 13.6 tactical integration test

```text
forced move がある場合、Q_mode より forced move が優先される
forced move でも Q_normal_result / Q_nyugyoku は書き換えられない
post-check mask 後の pi_search_post_tactical は合法手上で確率和 1 になる
mask が全手を消した場合はエラーになる
```

---

## 14. 実験項目

### 14.1 λ_nyugyoku sweep

代表値は以下を候補にする。

```text
normal:
  λ_nyugyoku = 1.0

nyugyoku / anti_nyugyoku / mutual_nyugyoku:
  λ_nyugyoku = 1.5, 2.0, 3.0

bakuretsu:
  λ_nyugyoku = 1.2, 1.5, 2.0, 2.5
```

評価する指標は以下である。

```text
通常勝率
入玉宣言勝ち率
相手入玉負け率
通常勝ちできる局面で無理に入玉しない率
KLENT-0 と Gumbel MCTS の差
```

### 14.2 value scale 実験

`Q_mode` の scale が KLENT 方策に与える影響を調べる。

候補は以下である。

```text
Q_mode をそのまま使う
Q_mode に temperature をかける
Q_mode を clip する
Q_normal_result と Q_nyugyoku に別 scale をかける
```

標準は「そのまま使う」である。

### 14.3 moves_left 補正実験

標準では無効だが、以下を実験してよい。

```text
勝ち寄り局面で早い勝ちを小さく優遇
負け寄り局面で遅い負けを小さく優遇
入玉成分が大きい局面では補正を弱める
```

実験時は、通常勝率と入玉ルートの長期成功率を分けて評価する。

### 14.4 V-Q consistency loss

`V_*` と policy expectation of `Q_*` の整合性を auxiliary loss にするか検討する。

初期実装では loss に入れず、ログだけにする。

導入する場合は、以下を確認する。

```text
policy が未成熟な初期段階で悪影響がない
入玉関連局面で Q_nyugyoku が潰れない
reanalyse target と矛盾しない
```

---

## 15. 未決事項

### 15.1 Gumbel MCTS の final policy

Gumbel MCTS の最終方策を以下のどちらにするかは実験で決める。

```text
visit_counts を温度変換した分布
Gumbel MCTS の improved policy
```

どちらを使う場合でも、value target は `Q_normal_result` と `Q_nyugyoku` を分けて保存する。

### 15.2 MCTS 内部視点

初期実装では、leaf value を root-side view へ変換してから backup する方式を標準にする。

node-side view で各 node に保存し、参照時に変換する方式は実験候補とする。

### 15.3 value clipping

`Q_total_result` と `Q_mode` は `[-1, 1]` に収まらない。

KLENT の安定性に問題が出た場合は、value clipping や scaling を検討する。

標準では clip しない。

---

## 16. 初期実装の標準設定

初期実装では以下を採用する。

```text
value components:
  normal_result と nyugyoku の2成分

λ_nyugyoku:
  SearchModeConfig に従う

KLENT-0:
  Q_mode を使って pi_klent を作る

Gumbel MCTS:
  leaf outcome を root-side view へ変換し、normal_result / nyugyoku を別々に backup する

reanalyse target:
  Q_normal_result_reanalyse と Q_nyugyoku_reanalyse を保存する

moves_left:
  主探索価値には入れない

CPU tactical solver:
  Q を書き換えず、forced move または pi_search_post_tactical に反映する
```

# docs/10_training_targets.md - 学習ターゲット仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎における **学習ターゲット生成** の仕様を定義する。

対象は以下である。

```text
TrainingSample の形式
自己対局ログからの target 生成
policy_normal_target / policy_style_target
q_normal_result_target / q_nyugyoku_target
outcome_cause_target
moves_left_target
Gumbel MCTS reanalyse target
λ-return target
tactical solver 由来 target
source / mode / phase ごとの sample weight
loss 計算時の mask
実装上の不変条件
検証項目
実験項目
```

爆裂玉登太郎では、ネットワークが以下を同時に学習する。

```text
通常棋力用 policy
入玉棋風用 policy
合法手ごとの通常勝敗 action-value
合法手ごとの入玉関連 action-value
局面単位の終局原因分布
終局までの距離
```

このドキュメントは、自己対局ログ、教師データ、reanalyse 結果から、それぞれの head に与える target をどのように作るかを定義する。

関連仕様は以下を正とする。

```text
全体方針: 爆裂玉登太郎_設計概要.md
探索・着手選択: docs/03_search.md
ネットワーク本体: docs/04_network.md
ネットワーク出力: docs/05_outputs.md
CPU tactical solver: docs/07_tactical_solver.md
自己対局ループ: docs/08_selfplay.md
開始局面多様化: docs/09_start_position_diversity.md
KLENT-first 学習方針: docs/13_klent_training.md
```

---

## 1. 基本方針

### 1.1 学習 target は head ごとに分離する

学習 target は以下に分ける。

```text
policy_normal_target
policy_style_target
q_normal_result_target
q_nyugyoku_target
outcome_cause_target
moves_left_target
```

それぞれの意味は以下である。

| target | 対応 head | 内容 |
| --- | --- | --- |
| `policy_normal_target` | `policy_normal_head` | 通常棋力用の方策分布 |
| `policy_style_target` | `policy_style_head` | 入玉棋風・相入玉・相手入玉警戒などの方策分布 |
| `q_normal_result_target` | `q_normal_result_head` | 通常勝敗成分の action-value |
| `q_nyugyoku_target` | `q_nyugyoku_head` | 入玉勝ち・相手入玉負け成分の action-value |
| `outcome_cause_target` | `outcome_cause_head` | 現在手番視点の5クラス終局原因 |
| `moves_left_target` | `moves_left_head` | 終局までの残り手数の `log1p` 値 |

`Q_total_result` や `Q_mode` は派生値であり、学習 target の正本にはしない。

### 1.2 現在手番視点を標準にする

すべての target は、サンプル局面の `side_to_move` から見た値として作る。

```text
policy target:
  現在手番が指す合法手上の分布

q_normal_result_target:
  現在手番側の通常勝ちを +1、通常負けを -1

q_nyugyoku_target:
  現在手番側の入玉勝ちを +1、相手入玉による現在手番側の負けを -1

outcome_cause_target:
  現在手番側から見た win_normal / win_nyugyoku / draw / loss_nyugyoku / loss_normal

moves_left_target:
  現在局面から、その対局の終局までの残り手数
```

先手視点、後手視点、root 視点、親 node 視点を混ぜてはならない。

### 1.3 違法手には target を持たせない

policy target と action-value target は、合法手だけに定義する。

保存形式は sparse を標準とする。

```python
@dataclass
class SparseMoveDistribution:
    moves: list[int]
    probs: list[float]

@dataclass
class SparseMoveValues:
    moves: list[int]
    values: list[float]
```

以下を必ず満たす。

```text
moves は合法手だけを含む
moves は現在手番視点の move label である
probs / values は moves と同じ順序である
policy probs の合計は 1 である
違法手確率は保存しない
loss 計算時も合法手 mask を適用する
```

### 1.4 target の欠損を明示する

すべてのサンプルがすべての head の target を持つとは限らない。

例:

```text
normal mode の自己対局 step:
  policy_normal_target はある
  policy_style_target は None

style mode の自己対局 step:
  policy_style_target はある
  policy_normal_target は None または低重み target

自己対局の通常ログ:
  選択手の q target はある
  非選択合法手の q target はない

Gumbel MCTS reanalyse:
  全合法手 q target を持つ場合がある
```

欠損 target をゼロ target として扱ってはならない。

`None` と `0` は意味が違う。

### 1.5 tactical solver の扱いを分ける

CPU tactical solver は、短期戦術とルール上の確定手を補う。

policy target には、標準で tactical solver 補正後の方策分布を使う。

```text
policy target 標準:
  pi_search_post_tactical
```

Q target は、終局結果、λ-return、Gumbel MCTS reanalyse を正本にする。

局所的に結果が確定した tactical 情報は、以下の場合だけ確定 target として扱う。

```text
宣言勝ち確定:
  q_nyugyoku_target = +1

相手玉への詰み確定:
  q_normal_result_target = +1
```

これらを使う場合は、根拠となった tactical depth、node limit、forced_reason を必ずログに残す。

---

## 2. TrainingSample

### 2.1 基本形式

自己対局ログ、教師データ、reanalyse 結果から、学習側では以下の構造体相当を作る。

```python
@dataclass
class TrainingSample:
    sample_id: str

    # position
    features_ref: str | None
    sfen: str
    side_to_move: int
    legal_moves: list[int]

    # policy targets
    policy_normal_target: SparseMoveDistribution | None
    policy_style_target: SparseMoveDistribution | None

    # action-value targets
    q_normal_result_target: SparseMoveValues | None
    q_nyugyoku_target: SparseMoveValues | None

    # state targets
    outcome_cause_target: int | None
    moves_left_target: float | None

    # masks / weights
    sample_weight: float
    loss_weights: "TargetLossWeights"

    # provenance
    source: str
    source_game_id: str | None
    source_step_index: int | None
    source_ply: int | None
    absolute_ply: int | None
    terminal_ply: int | None
    model_version: str | None
    feature_version: str
    move_label_version: str
    search_config_id: str | None
    mode_id: str | None
    target_policy_head: str | None
    start_position_source: str | None
    tags: list[str]
```

`features_ref` は、feature tensor を別保存する場合の参照である。

通常は `sfen`、`side_to_move`、`feature_version` から feature を再生成できるため、feature tensor 自体を必ず保存する必要はない。

### 2.2 TargetLossWeights

サンプルごとの loss weight は以下の構造体相当で持つ。

```python
@dataclass
class TargetLossWeights:
    policy_normal: float = 1.0
    policy_style: float = 1.0
    q_normal_result: float = 1.0
    q_nyugyoku: float = 1.0
    outcome: float = 1.0
    moves_left: float = 1.0
```

全体の head weight と、サンプルごとの target weight は分ける。

```text
head weight:
  training config 側で head 全体の重要度を決める

sample / target weight:
  そのサンプルまたはその target の信頼度を表す
```

実際の loss では、概念的に以下を使う。

```text
L_total = Σ_head global_head_weight[head] * sample_target_weight[head] * L_head
```

### 2.3 source

`source` は target の由来を表す。

標準値は以下である。

```text
selfplay
reanalyse
teacher_search
teacher_move
supervised_kifu
tactical_forced
debug_test
```

`source` は target の信頼度、mini-batch sampling、評価分析に使う。

### 2.4 tags

`tags` には、サンプルの性質を表す文字列を入れる。

例:

```text
normal_mode
nyugyoku_mode
anti_nyugyoku_mode
mutual_nyugyoku_mode
bakuretsu_mode
initial_start
random_opening_start
shuffled_initial_start
position_buffer_start
nyugyoku_promising
opponent_nyugyoku_risk
mutual_nyugyoku_like
policy_uncertain
q_close_top_moves
tactical_masked
declaration_win_available
mate_in_one_available
max_ply_draw
repetition_draw
```

`tags` は loss 計算の正本ではないが、oversampling、ログ集計、ablation に使う。

---

## 3. 自己対局ログからの sample 生成

### 3.1 入力ログ

`SampleBuilder` は、自己対局ログから `TrainingSample` を作る。

入力は以下である。

```python
@dataclass
class SelfPlayGame:
    game_id: str
    config_id: str
    model_version: str
    feature_version: str
    move_label_version: str
    start_position: StartPositionInfo
    pairing_id: str
    sente_player: PlayerSpecLog
    gote_player: PlayerSpecLog
    steps: list[SelfPlayStep]
    result: GameResult
```

各 step には、少なくとも以下が必要である。

```python
@dataclass
class SelfPlayStep:
    ply: int
    absolute_ply: int
    side_to_move: int
    sfen: str | None
    features_ref: str | None
    legal_moves: list[int]
    selected_move: int

    mode_id: str
    target_policy_head: str

    p_prior: SparseMoveDistribution
    pi_klent_pre_tactical: SparseMoveDistribution
    pi_search_post_tactical: SparseMoveDistribution
    pi_select: SparseMoveDistribution

    q_normal_result_pred: SparseMoveValues | None
    q_nyugyoku_pred: SparseMoveValues | None
    outcome_probs: list[float] | None
    moves_left_pred: float | None

    tactical_flags: dict
    terminal_flags_before_move: dict
    nyugyoku_features: dict
    uncertainty_features: dict
    reanalyse_tags: list[str]
```

### 3.2 1 step から 1 sample を作る

標準では、自己対局の各 step から1つの `TrainingSample` を作る。

疑似コードは以下である。

```python
def build_sample_from_selfplay_step(game: SelfPlayGame, step: SelfPlayStep) -> TrainingSample:
    outcome = make_outcome_cause_target(
        side_to_move=step.side_to_move,
        result=game.result,
    )

    q_normal, q_nyugyoku = make_selected_move_q_targets(
        selected_move=step.selected_move,
        outcome_cause=outcome,
    )

    policy_normal, policy_style, weights = assign_policy_targets(
        step=step,
        target_distribution=step.pi_search_post_tactical,
    )

    moves_left = make_moves_left_target(
        terminal_ply=game.result.terminal_ply,
        step_ply=step.ply,
    )

    return TrainingSample(
        sfen=step.sfen,
        side_to_move=step.side_to_move,
        legal_moves=step.legal_moves,
        policy_normal_target=policy_normal,
        policy_style_target=policy_style,
        q_normal_result_target=q_normal,
        q_nyugyoku_target=q_nyugyoku,
        outcome_cause_target=outcome,
        moves_left_target=moves_left,
        sample_weight=1.0,
        loss_weights=weights,
        source="selfplay",
        tags=make_tags(game, step),
    )
```

### 3.3 terminal step は作らない

terminal node では network inference を呼ばないため、標準では terminal 局面そのものを `TrainingSample` にしない。

学習サンプルにするのは、着手選択が行われた非 terminal step である。

ただし、デバッグ用に terminal 局面を教師ありデータへ入れる実験は許可する。その場合は `source = "debug_test"` または明示的な `source` を付け、通常 training set と混ぜない。

### 3.4 position buffer 由来対局の扱い

position buffer 由来の開始局面から再開した対局では、target は **再開後の対局結果** から作る。

混同してはならないものは以下である。

```text
source_game_result:
  その局面が最初に収集された元対局の結果

restart_game_result:
  その局面から再開した今回の対局結果
```

training target に使うのは `restart_game_result` である。

`source_game_result` は、局面分類や分析には使ってよいが、再開対局 sample の勝敗 target には使わない。

### 3.5 start_position metadata

開始局面情報は sample に引き継ぐ。

最低限、以下を保存する。

```text
start_position.source
start_position.category
start_position.ply_offset
start_position.tags
```

これは以下に使う。

```text
source 別 loss 分析
シャッフル局面由来 target の重み調整
position buffer 由来 target の評価
通常平手分布からの乖離確認
```

---

## 4. outcome_cause_target

### 4.1 class order

`outcome_cause_target` の class order は固定で以下とする。

```python
OUTCOME_WIN_NORMAL = 0
OUTCOME_WIN_NYUGYOKU = 1
OUTCOME_DRAW = 2
OUTCOME_LOSS_NYUGYOKU = 3
OUTCOME_LOSS_NORMAL = 4
OUTCOME_NUM_CLASSES = 5
```

この順序は checkpoint 互換性に関わるため、途中で変更してはならない。

### 4.2 GameResult から outcome category への変換

`GameResult` は少なくとも以下を持つ。

```python
@dataclass
class GameResult:
    winner: int | None
    terminal_reason: str
    terminal_ply: int
    absolute_terminal_ply: int
    is_draw: bool
    is_nyugyoku_win: bool
    outcome_category: str  # "normal" | "nyugyoku" | "draw"
```

`terminal_reason` から `outcome_category` への標準対応は以下である。

| terminal_reason | outcome_category |
| --- | --- |
| `checkmate` | `normal` |
| `resign` | `normal` |
| `normal_adjudication` | `normal` |
| `declaration_win` | `nyugyoku` |
| `nyugyoku_adjudication` | `nyugyoku` |
| `repetition_draw` | `draw` |
| `jishogi_draw` | `draw` |
| `max_ply_draw` | `draw` |

最大手数到達時に勝敗未確定であれば、標準では `max_ply_draw` とする。

この表にない `terminal_reason` は、学習 target 生成の対象外である。標準の自己対局・探索・棋譜生成では、表外の終局理由を発生させない。

### 4.3 現在手番視点への変換

各 step の `side_to_move` から見て、以下で target を作る。

```python
def make_outcome_cause_target(side_to_move: int, result: GameResult) -> int:
    if result.winner is None or result.outcome_category == "draw":
        return OUTCOME_DRAW

    is_win = (result.winner == side_to_move)

    if is_win and result.outcome_category == "normal":
        return OUTCOME_WIN_NORMAL
    if is_win and result.outcome_category == "nyugyoku":
        return OUTCOME_WIN_NYUGYOKU
    if (not is_win) and result.outcome_category == "nyugyoku":
        return OUTCOME_LOSS_NYUGYOKU
    if (not is_win) and result.outcome_category == "normal":
        return OUTCOME_LOSS_NORMAL

    raise ValueError(result)
```

### 4.4 terminal_reason と winner の整合性

以下はエラーとして扱う。

```text
outcome_category == draw なのに winner が None ではない
outcome_category != draw なのに winner が None
terminal_reason が outcome_category に変換できない
is_nyugyoku_win と outcome_category が矛盾する
```

`resign` は `normal` として扱う。

ただし初期実装では、resign は標準無効である。

### 4.5 outcome loss

`outcome_cause_head` は cross entropy で学習する。

```text
L_outcome = CE(outcome_cause_logits, outcome_cause_target)
```

`outcome_cause_target is None` のサンプルでは、outcome loss を計算しない。

入玉勝ち・相手入玉負けは頻度が低いため、class weight または oversampling を使ってよい。

ただし、最初から強すぎる class weight をかけると calibration が崩れる可能性があるため、標準では oversampling と source weight を優先する。

---

## 5. q target

### 5.1 基本定義

`q_normal_result_target` と `q_nyugyoku_target` は、`outcome_cause_target` から作る。

| outcome_cause | q_normal_result | q_nyugyoku |
| --- | ---: | ---: |
| `win_normal` | +1 | 0 |
| `win_nyugyoku` | 0 | +1 |
| `draw` | 0 | 0 |
| `loss_nyugyoku` | 0 | -1 |
| `loss_normal` | -1 | 0 |

疑似コードは以下である。

```python
def outcome_to_q_components(outcome: int) -> tuple[float, float]:
    if outcome == OUTCOME_WIN_NORMAL:
        return +1.0, 0.0
    if outcome == OUTCOME_WIN_NYUGYOKU:
        return 0.0, +1.0
    if outcome == OUTCOME_DRAW:
        return 0.0, 0.0
    if outcome == OUTCOME_LOSS_NYUGYOKU:
        return 0.0, -1.0
    if outcome == OUTCOME_LOSS_NORMAL:
        return -1.0, 0.0
    raise ValueError(outcome)
```

### 5.2 自己対局 MC target

通常の自己対局ログでは、実際に選んだ手だけに Monte Carlo target を付ける。

```python
def make_selected_move_q_targets(selected_move: int, outcome_cause: int):
    q_normal, q_nyugyoku = outcome_to_q_components(outcome_cause)
    return (
        SparseMoveValues(moves=[selected_move], values=[q_normal]),
        SparseMoveValues(moves=[selected_move], values=[q_nyugyoku]),
    )
```

非選択合法手には target がない。

非選択合法手の target を 0 として埋めてはならない。

### 5.3 reanalyse full-action target

Gumbel MCTS reanalyse により、合法手ごとの action-value が得られる場合は、全合法手 target として保存してよい。

```python
@dataclass
class ReanalyseTarget:
    pi_reanalyse: SparseMoveDistribution | None
    q_normal_result_reanalyse: SparseMoveValues | None
    q_nyugyoku_reanalyse: SparseMoveValues | None
    outcome_cause_reanalyse: list[float] | None
    reanalyse_config_id: str
    num_simulations: int
    search_depth_limit: int | None
    target_quality: float
```

reanalyse target は以下を満たす必要がある。

```text
Q_normal_result と Q_nyugyoku を分ける
Q_mode だけを保存しない
root 視点・現在手番視点のどちらかを metadata で明示する
TrainingSample に入れる時点では現在手番視点へ揃える
全合法手 target の moves は legal_moves と一致する
```

### 5.4 λ-return target

最終結果だけの Monte Carlo target は分散が大きい。

そのため、自己対局中または postprocess で λ-return を作ってよい。

通常勝敗成分と入玉成分は分ける。

```text
G_normal_result[t]
G_nyugyoku[t]
```

概念的には以下である。

```text
G_t = r_t + γ * ((1 - λ) * V(s_{t+1}) + λ * G_{t+1})
```

爆裂玉登太郎では、将棋の途中報酬は標準では 0 とし、終局時に成分別報酬を与える。

```text
normal_result terminal reward:
  win_normal  = +1
  loss_normal = -1
  otherwise   = 0

nyugyoku terminal reward:
  win_nyugyoku  = +1
  loss_nyugyoku = -1
  otherwise     = 0
```

λ-return を使う場合、bootstrap value も成分別にする。

```text
V_normal_result = P_win_normal - P_loss_normal
V_nyugyoku      = P_win_nyugyoku - P_loss_nyugyoku
```

`V_total_result` や `V_mode` だけで bootstrap してはならない。

### 5.5 λ-return の視点変換

`V(s_{t+1})` は次局面の手番視点で出力される。

親局面の手番視点へ戻すには、outcome 分布または成分 value を反転する。

成分 value で扱う場合の標準は以下である。

```text
parent_V_normal_result = - child_V_normal_result
parent_V_nyugyoku      = - child_V_nyugyoku
```

ただし、より安全なのは outcome 分布を変換してから成分 value を計算する方式である。

```text
parent_P_win_normal    = child_P_loss_normal
parent_P_win_nyugyoku  = child_P_loss_nyugyoku
parent_P_draw          = child_P_draw
parent_P_loss_nyugyoku = child_P_win_nyugyoku
parent_P_loss_normal   = child_P_win_normal
```

実装では、どちらの方式を使ったかを `target_generation_config` に保存する。

### 5.6 tactical forced target

局所的に結果が確定する tactical 情報は、限定的に q target として使ってよい。

標準で許可するものは以下である。

| forced_reason | q_normal_result | q_nyugyoku | 対象手 |
| --- | ---: | ---: | --- |
| `declaration_win` | 0 | +1 | 宣言勝ち手 |
| `mate_win` | +1 | 0 | 相手玉への詰みが証明された手 |

使う場合は、通常の自己対局 MC target よりも `target_quality` を明示する。

`tactical_forced` target は、短期戦術の補助であり、通常の長期 Q 学習の代替ではない。

### 5.7 q loss

q head は、raw output に `tanh` を適用した値で loss を計算する。

```text
Q_normal_result = tanh(q_normal_result_raw)
Q_nyugyoku      = tanh(q_nyugyoku_raw)
```

基本 loss は Smooth L1 とする。

```text
L_q_normal_result = SmoothL1(Q_normal_result[moves], q_normal_result_target.values)
L_q_nyugyoku      = SmoothL1(Q_nyugyoku[moves], q_nyugyoku_target.values)
```

`q_*_target is None` の場合、その head の q loss は計算しない。

全合法手 target がある場合は合法手上で平均する。

選択手 target だけの場合は選択手だけで平均する。

---

## 6. policy target

### 6.1 policy target の種類

policy target は以下から作る。

```text
pi_search_post_tactical
pi_klent_pre_tactical
pi_reanalyse
教師探索の訪問回数分布
教師手 one-hot
```

標準は以下である。

```text
自己対局:
  pi_search_post_tactical

Gumbel MCTS reanalyse:
  pi_reanalyse

教師手のみ:
  one-hot
```

自己対局時に sampling に使った `pi_select` は、標準では policy target にしない。

`pi_select` には温度や root noise が反映されるため、学習 target としては意図的に使う場合だけ採用する。

### 6.2 normal mode の policy target

`normal` mode の step は、標準で `policy_normal_head` の target にする。

```text
policy_normal_target = pi_search_post_tactical
policy_style_target  = None
```

`policy_normal_head` は通常棋力の基盤であるため、初期学習では最優先にする。

### 6.3 style 系 mode の policy target

以下の mode は、標準で `policy_style_head` の target にする。

```text
nyugyoku
anti_nyugyoku
mutual_nyugyoku
bakuretsu
```

標準規則は以下である。

```text
policy_normal_target = None
policy_style_target  = pi_search_post_tactical
```

`bakuretsu` は通常棋力と入玉棋風の中間である。

初期実装では `policy_style_target` として扱い、通常棋力への悪影響を見ながら、一部を低重みで `policy_normal_head` に混ぜる実験を行う。

### 6.4 both target

`target_policy_head = "both"` の場合は、同じ分布を両方の policy head に与えてよい。

ただし、loss weight は分ける。

```text
policy_normal_target = target_distribution
policy_style_target  = target_distribution
policy_normal target weight = small or normal
policy_style target weight  = normal or small
```

`both` は標準ではなく、以下の用途に限定する。

```text
教師探索が通常棋力にも style にも有用な場合
bakuretsu mode の一部を normal に弱く混ぜる実験
style head が未成熟な初期段階で normal target を弱くコピーする場合
```

### 6.5 policy target assignment

疑似コードは以下である。

```python
def assign_policy_targets(step, target_distribution):
    weights = TargetLossWeights()

    if step.target_policy_head == "normal":
        return target_distribution, None, weights

    if step.target_policy_head == "style":
        return None, target_distribution, weights

    if step.target_policy_head == "both":
        weights.policy_normal = step.policy_normal_target_weight
        weights.policy_style = step.policy_style_target_weight
        return target_distribution, target_distribution, weights

    raise ValueError(step.target_policy_head)
```

`mode_id` だけで分岐せず、ログに保存された `target_policy_head` を正とする。

### 6.6 tactical postprocess と policy target

自己対局ログでは、以下を区別して保存する。

```text
p_prior
pi_klent_pre_tactical
pi_search_post_tactical
pi_select
```

policy target の標準は `pi_search_post_tactical` である。

理由は、短期戦術の安全性を policy にも蒸留したいためである。

ただし、以下の比較実験を行う。

```text
A. pi_klent_pre_tactical を target にする
B. pi_search_post_tactical を target にする
C. forced move だけ one-hot にする
D. tactical mask が入った sample だけ weight を下げる
```

### 6.7 one-hot target

教師手が1手だけの場合は one-hot として扱う。

```python
def make_one_hot_policy_target(legal_moves: list[int], move: int) -> SparseMoveDistribution:
    assert move in legal_moves
    return SparseMoveDistribution(moves=[move], probs=[1.0])
```

one-hot target は過度に鋭いため、必要に応じて label smoothing を使ってよい。

ただし、label smoothing を使う場合でも、確率を違法手へ漏らしてはならない。

### 6.8 distribution target の正規化

policy target は必ず合法手上で正規化する。

```text
sum(probs) = 1
prob >= 0
moves ⊆ legal_moves
```

丸め誤差で合計がずれる場合は再正規化する。

確率和が 0 の場合は sample を破棄するか、target generation error として扱う。

### 6.9 policy loss

policy loss は cross entropy とする。

```text
L_policy = - Σ_a pi_target(a) * log softmax_legal(policy_logits)(a)
```

`policy_normal_target is None` の場合、`L_policy_normal` は計算しない。

`policy_style_target is None` の場合、`L_policy_style` は計算しない。

---

## 7. moves_left_target

### 7.1 定義

`moves_left_target` は、現在局面から終局までの残り手数の `log1p` である。

局面 step の ply を `t`、終局 ply を `T` とする。

```text
moves_left = T - t
moves_left_target = log1p(max(T - t, 0))
```

`moves_left_head` は `log1p(moves_left)` を直接予測する。

### 7.2 ply と absolute_ply

開始局面が平手初期局面ではない場合、以下を区別する。

```text
ply:
  再開対局ログ内の手数

absolute_ply:
  平手初期局面から見た通算手数

ply_offset:
  開始局面が持つオフセット
```

`moves_left_target` は、標準では **再開対局内の終局までの残り手数** とする。

```text
moves_left_target = log1p(game.result.terminal_ply - step.ply)
```

`absolute_ply` は、局面分析や手数帯別評価に使う。

### 7.3 最大手数到達

最大手数到達で `max_ply_draw` になった場合も、`moves_left_target` は実際に打ち切られた手数までの残りで作る。

ただし、これは「自然な終局までの距離」ではなく「自己対局設定上の終局までの距離」である。

そのため、`terminal_reason = max_ply_draw` の sample は、moves_left loss weight を下げる実験を行う。

### 7.4 入玉局面での注意

入玉は長期計画になりやすい。

したがって、`moves_left_target` は補助信号であり、探索価値の主成分にはしない。

学習上も、`moves_left` を強くしすぎると以下が起きる可能性がある。

```text
長期入玉ルートを過小評価する
早い終局だけを好む
相入玉の粘りを悪く学習する
```

標準では `w_moves_left` を小さくする。

### 7.5 moves_left loss

loss は Smooth L1 とする。

```text
L_moves_left = SmoothL1(moves_left_pred, moves_left_target)
```

`moves_left_target is None` の場合、loss を計算しない。

---

## 8. reanalyse target

### 8.1 reanalyse の目的

Gumbel MCTS reanalyse は、KLENT-0 の弱点を補うために使う。

主な対象は以下である。

```text
入玉関連局面
policy entropy が高い局面
Q 上位手が僅差の局面
通常勝ちと入玉勝ちの選択が割れる局面
自己対局で負けにつながった疑いのある局面
tactical mask が大きく効いた局面
```

### 8.2 ReanalyseRecord

reanalyse 結果は、自己対局 sample に上書きしてもよいが、標準では別 record として保存する。

```python
@dataclass
class ReanalyseRecord:
    reanalyse_id: str
    source_sample_id: str
    sfen: str
    side_to_move: int
    legal_moves: list[int]

    pi_reanalyse: SparseMoveDistribution | None
    q_normal_result_reanalyse: SparseMoveValues | None
    q_nyugyoku_reanalyse: SparseMoveValues | None
    outcome_cause_reanalyse: list[float] | None

    search_config_id: str
    reanalyse_config_id: str
    num_simulations: int
    gumbel_top_k: int | None
    sequential_halving_rounds: int | None
    target_quality: float
    created_at: str
```

### 8.3 reanalyse から TrainingSample を作る

reanalyse sample は、以下のように target を置き換える。

```text
policy target:
  pi_reanalyse があれば pi_reanalyse
  なければ元 sample の policy target

q target:
  q_normal_result_reanalyse / q_nyugyoku_reanalyse があればそれを使う
  なければ元 sample の q target

outcome target:
  outcome_cause_reanalyse が hard label ではなく分布の場合、soft CE / KL loss の実験対象にする
  初期実装では元 sample の hard outcome target を維持する
```

初期実装では、reanalyse の主用途は以下である。

```text
pi_reanalyse を policy target として使う
Q_normal_result_reanalyse / Q_nyugyoku_reanalyse を action-value target として使う
```

### 8.4 target mixing

同じ局面に selfplay target と reanalyse target がある場合、以下のどちらかを使う。

```text
replace:
  reanalyse target を優先して置き換える

mix:
  selfplay target と reanalyse target を重み付きで混ぜる
```

初期実装では `replace` を標準とする。

`mix` は、reanalyse budget が小さく不安定な場合の実験候補とする。

### 8.5 reanalyse target quality

reanalyse target には品質指標を持たせる。

例:

```text
num_simulations
search_depth_limit
root_visit_count
policy_entropy_after_search
top1_visit_prob
q_top1_gap
tactical_mask_applied
terminal_found_by_search
```

`target_quality` が低い reanalyse target は、loss weight を下げるか、policy target だけに使う。

---

## 9. 教師あり初期化 target

### 9.1 目的

Phase 0 では、KLENT 自己対局を始められる最低限の policy と Q を作る。

主な教師は以下である。

```text
既存Yoltsの探索手
簡易αβ探索の手
詰み探索で確定した手
過去の自己対局棋譜
外部エンジンの固定深さ探索
```

### 9.2 policy 教師

教師が1手だけを返す場合は one-hot target にする。

教師探索が候補手分布を返す場合は、合法手上の分布として保存する。

```python
@dataclass
class TeacherPolicyTarget:
    legal_moves: list[int]
    target_policy: SparseMoveDistribution
    teacher_id: str
    teacher_depth: int | None
    teacher_nodes: int | None
    teacher_time_ms: int | None
```

`teacher_id` と探索条件は必ず保存する。

### 9.3 q 教師

教師探索から評価値を得る場合、初期実装では無理に `q_nyugyoku_target` へ変換しない。

通常の評価値は、主に `q_normal_result_target` の補助として扱う。

ただし、教師が入玉宣言勝ち、相手入玉負け、持将棋などを明示的に分類できる場合のみ、`q_nyugyoku_target` へ変換してよい。

評価値を勝率相当へ変換する場合は、変換式、scale、clip 範囲を metadata に保存する。

### 9.4 outcome 教師

棋譜の最終結果がある場合は、`outcome_cause_target` を作る。

終局原因が不明で単なる勝敗だけがある場合は、標準では以下のように扱う。

```text
勝ち:
  win_normal

負け:
  loss_normal

引き分け:
  draw
```

ただし、この target は入玉原因を区別できないため、`target_quality` を下げる。

入玉学習の主 target には、終局原因が明示された自己対局ログを使う。

### 9.5 style head の初期化

style 用データが少ない段階では、`policy_style_head` に通常 target を小さい重みで学習させてよい。

```text
policy_style_target = policy_normal_target
policy_style target weight = 0.05 〜 0.2
```

これは style head を完全なランダム初期値のままにしないための warmup である。

ただし、通常 target を強く入れすぎると style head が normal head と同じになり、棋風分離の意味が薄れる。

### 9.6 q_nyugyoku の初期化

入玉関連データが少ない段階では、`q_nyugyoku_head` は過信しない。

初期値として以下を許可する。

```text
q_nyugyoku_target がないサンプルでは loss をかけない
明確な入玉関連 terminal のみ q_nyugyoku_target をかける
入玉有望局面を oversampling する
q_nyugyoku loss weight を小さく始める
```

通常勝敗 target を `q_nyugyoku_head` に入れてはならない。

---

## 10. sample weight と sampling

### 10.1 基本方針

学習データは、source、mode、開始局面、終局原因の分布が偏る。

そのため、以下を分けて制御する。

```text
データセットに入れる比率
mini-batch sampling 比率
sample_weight
head loss weight
class / target weight
```

最初から複雑にしすぎず、初期実装では以下を標準にする。

```text
通常 selfplay sample:
  sample_weight = 1.0

style mode sample:
  policy_style は 1.0
  policy_normal は None または低重み

reanalyse sample:
  target_quality に応じて 1.0 以上

shuffled_initial 由来 sample:
  通常棋力への影響を見ながら 0.5 〜 1.0

入玉勝ち / 相手入玉負け sample:
  oversampling または sample_weight 上げ
```

### 10.2 head global weight

総合 loss は以下である。

```text
L_total =
    w_policy_normal    * L_policy_normal
  + w_policy_style     * L_policy_style
  + w_q_normal_result  * L_q_normal_result
  + w_q_nyugyoku       * L_q_nyugyoku
  + w_outcome          * L_outcome
  + w_moves_left       * L_moves_left
```

初期値の候補は以下である。

```text
w_policy_normal   = 1.0
w_policy_style    = 0.2
w_q_normal_result = 1.0
w_q_nyugyoku      = 0.3
w_outcome         = 1.0
w_moves_left      = 0.1
```

入玉関連データが増えたら、`w_policy_style` と `w_q_nyugyoku` を段階的に上げる。

### 10.3 mode 別 weight

mode 別 weight は、単に style head へ流すかどうかではなく、どの outcome 成分を強く見るかまで分ける。
`nyugyoku`、`anti_nyugyoku`、`mutual_nyugyoku` は似た名前でも学習目的が異なるため、同じ weight にしてはならない。

初期候補は以下である。

| mode | policy_normal | policy_style | q_normal_result | q_nyugyoku | outcome | moves_left | 主に強める対象 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `normal` | 1.0 | 0.0 | 1.0 | 0.2 | 1.0 | 0.1 | 通常棋力、通常勝敗 calibration |
| `nyugyoku` | 0.0 | 1.0 | 0.6 | 1.4 | 1.1 | 0.05 | 自玉の `win_nyugyoku`、`q_nyugyoku = +1` 側 |
| `anti_nyugyoku` | 0.0 | 1.0 | 0.8 | 1.6 | 1.4 | 0.1 | 相手入玉による `loss_nyugyoku`、`q_nyugyoku = -1` 側 |
| `mutual_nyugyoku` | 0.0 | 1.0 | 0.5 | 1.8 | 1.5 | 0.2 | `win_nyugyoku` と `loss_nyugyoku` の両方、相入玉長期局面 |
| `bakuretsu` | 0.1 | 0.8 | 1.0 | 0.8 | 1.0 | 0.1 | 通常棋力を残した入玉寄り方策 |

mode weight に加えて、outcome class weight も mode ごとに変える。

| mode | win_normal | win_nyugyoku | draw | loss_nyugyoku | loss_normal | 意図 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `normal` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 終局原因分布の基準 calibration |
| `nyugyoku` | 0.7 | 1.8 | 0.8 | 1.0 | 0.7 | 自玉入玉成功例を厚く学習する |
| `anti_nyugyoku` | 0.8 | 0.9 | 0.8 | 2.0 | 0.8 | 相手入玉で負けるパターンを厚く学習する |
| `mutual_nyugyoku` | 0.6 | 1.6 | 1.2 | 1.6 | 0.6 | 相入玉・持将棋・長期化を含む分布を厚く学習する |
| `bakuretsu` | 1.0 | 1.3 | 0.9 | 1.3 | 1.0 | 通常勝敗を崩さず入玉成分を少し強める |

各 mode の意味は以下とする。

```text
nyugyoku:
  自玉の入玉成功を増やすための mode。
  `win_nyugyoku` と `q_nyugyoku = +1` の学習を重くする。
  相手入玉負けの検出は行うが、主目的にはしない。

anti_nyugyoku:
  相手の入玉を防ぐための mode。
  `loss_nyugyoku` と `q_nyugyoku = -1` の学習を重くする。
  自玉入玉成功の棋風ではなく、防御・阻止の棋風を style policy に入れる。

mutual_nyugyoku:
  双方が入玉を狙う長期局面の mode。
  `win_nyugyoku` と `loss_nyugyoku` の両方を重くし、draw / jishogi 系も薄くしすぎない。
  終局までの距離が長くなるため、`moves_left` を `nyugyoku` より高めに残す。
```

これは初期値であり、通常棋力と入玉指標を見て調整する。

### 10.4 start position source 別 weight

開始局面 source によって、target の意味が少し変わる。

候補は以下である。

| start_position.source | 初期 weight | 注意 |
| --- | ---: | --- |
| `initial` | 1.0 | 通常棋力の主分布 |
| `random_opening` | 0.8 〜 1.0 | 合法到達局面なので比較的自然 |
| `shuffled_initial` | 0.5 〜 0.8 | cold start 補助。通常棋力に入れすぎない |
| `position_buffer` | 1.0 | category に応じて oversampling |
| `manual` | 0.0 | 通常 training には混ぜない |
| `test` | 0.0 | test 専用 |

`shuffled_initial` 由来 sample は、局面多様性には有用だが通常序盤分布とは異なる。

そのため、通常棋力が落ちる場合は、まず `shuffled_initial` 由来の `policy_normal` weight を下げる。

### 10.5 outcome class balancing

入玉勝ち・相手入玉負けは出現頻度が低い可能性が高い。

対応は以下の順で行う。

```text
1. position buffer の入玉関連 category を増やす
2. 入玉関連局面からの再開を増やす
3. 入玉 outcome sample を oversampling する
4. q_nyugyoku / outcome の sample_weight を上げる
5. class weight を導入する
```

最初から class weight を強くすると、確率 calibration が崩れる可能性がある。

### 10.6 duplicate と correlated sample

同一対局の連続局面は相関が強い。

mini-batch sampling では、以下を考慮する。

```text
同一 game_id からの sample 数を制限する
同一 start_position からの sample 数を制限する
同一 position_key の重複を削る
position buffer 由来局面では source_game_id の偏りを見る
```

---

## 11. loss 計算仕様

### 11.1 batch 形式

mini-batch では、target が欠損している head がある。

そのため、head ごとに有効 sample mask を持つ。

```python
@dataclass
class TrainingBatch:
    features: torch.Tensor
    legal_move_mask: torch.Tensor  # [B, MOVE_LABEL_SIZE]

    policy_normal_target: SparseBatchDistribution | None
    policy_style_target: SparseBatchDistribution | None
    q_normal_result_target: SparseBatchValues | None
    q_nyugyoku_target: SparseBatchValues | None
    outcome_cause_target: torch.Tensor | None
    moves_left_target: torch.Tensor | None

    target_masks: TargetMasks
    sample_weights: torch.Tensor
    target_loss_weights: TargetLossWeightsBatch
```

### 11.2 policy loss

policy loss は合法手上 softmax を使う。

```python
def policy_ce_loss(logits, legal_mask, sparse_target, sample_weight):
    masked_logits = mask_illegal_logits(logits, legal_mask)
    log_probs = log_softmax(masked_logits, dim=-1)
    loss = -sum_target_probs_times_log_probs(log_probs, sparse_target)
    return weighted_mean(loss, sample_weight)
```

違法手に確率を漏らしてはならない。

### 11.3 q loss

q loss は target が定義された move のみで計算する。

```python
q = torch.tanh(q_raw)
loss = smooth_l1(q[batch_indices, target_moves], target_values)
```

全合法手 target がある場合も、違法手は loss に含めない。

### 11.4 outcome loss

`outcome_cause_target` が hard label の場合は cross entropy を使う。

```text
L_outcome = CE(outcome_cause_logits, outcome_cause_target)
```

reanalyse から soft outcome 分布を使う場合は、KL / soft cross entropy を実験候補にする。

初期実装では hard label を標準にする。

### 11.5 moves_left loss

`moves_left_pred` は `[B, 1]`、target は `[B]` または `[B, 1]` とする。

```text
L_moves_left = SmoothL1(moves_left_pred.squeeze(-1), moves_left_target)
```

### 11.6 NaN / inf 防止

以下を必ず検査する。

```text
policy target の確率和が 1
policy target に NaN / inf がない
q target が [-1, 1] に収まる
moves_left_target が 0 以上
outcome_cause_target が 0..4 の範囲
legal_moves が空でない
```

異常 sample は、黙って補正せず、ログへ出して破棄する。

---

## 12. target 生成 pipeline

### 12.1 コンポーネント

学習ターゲット生成は、以下で構成する。

```text
GameLogReader
  自己対局ログを読む

SampleBuilder
  SelfPlayGame / SelfPlayStep から TrainingSample を作る

ReanalyseTargetMerger
  reanalyse 結果を TrainingSample に反映する

TargetValidator
  target の整合性を検証する

DatasetWriter
  training dataset を保存する

DatasetSampler
  mini-batch sampling を行う
```

### 12.2 SampleBuilderConfig

```python
@dataclass
class SampleBuilderConfig:
    config_id: str

    # policy
    policy_target_source: str = "post_tactical"  # "post_tactical" | "pre_tactical" | "select"
    use_pi_select_as_target: bool = False
    use_root_noise_in_target: bool = False

    # q
    q_target_source: str = "mc"  # "mc" | "lambda_return" | "reanalyse"
    use_lambda_return: bool = False
    lambda_return_lambda: float = 0.8
    lambda_return_gamma: float = 1.0

    # reanalyse
    prefer_reanalyse_policy: bool = True
    prefer_reanalyse_q: bool = True
    min_reanalyse_quality: float = 0.0

    # tactical
    use_tactical_forced_q_targets: bool = True
    tactical_q_target_weight: float = 1.0

    # balancing
    oversample_nyugyoku_outcomes: bool = True
    shuffled_initial_policy_normal_weight: float = 0.5

    # validation
    strict_validation: bool = True
```

### 12.3 生成手順

```text
1. SelfPlayGame を読む
2. GameResult の terminal_reason を outcome_category へ変換する
3. 各 SelfPlayStep について side_to_move 視点の outcome_cause_target を作る
4. outcome_cause_target から q_normal_result / q_nyugyoku の MC target を作る
5. pi_search_post_tactical を policy target として割り当てる
6. moves_left_target を作る
7. start_position / mode / tactical / nyugyoku tags を付与する
8. reanalyse 結果があれば policy / q target を置き換える
9. sample weight と loss weight を設定する
10. TargetValidator を通す
11. DatasetWriter で保存する
```

### 12.4 dataset metadata

dataset には以下を保存する。

```python
@dataclass
class TrainingDatasetMetadata:
    dataset_id: str
    created_at: str
    source_game_log_ids: list[str]
    source_reanalyse_ids: list[str]
    sample_builder_config_id: str
    feature_version: str
    move_label_version: str
    outcome_class_order: list[str]
    declaration_rule_id: str
    num_samples: int
    source_counts: dict[str, int]
    mode_counts: dict[str, int]
    outcome_counts: dict[str, int]
    start_position_source_counts: dict[str, int]
    validation_summary: dict[str, float | int]
```

---

## 13. ログ項目

### 13.1 target distribution ログ

```text
sample/num_samples
sample/source/selfplay_count
sample/source/reanalyse_count
sample/source/teacher_count
sample/mode/normal_count
sample/mode/nyugyoku_count
sample/mode/anti_nyugyoku_count
sample/mode/mutual_nyugyoku_count
sample/mode/bakuretsu_count
```

### 13.2 outcome target ログ

```text
outcome_target/win_normal_count
outcome_target/win_nyugyoku_count
outcome_target/draw_count
outcome_target/loss_nyugyoku_count
outcome_target/loss_normal_count
outcome_target/win_nyugyoku_rate
outcome_target/loss_nyugyoku_rate
```

### 13.3 policy target ログ

```text
policy_target/normal_count
policy_target/style_count
policy_target/both_count
policy_target/none_count
policy_target/entropy_normal_avg
policy_target/entropy_style_avg
policy_target/post_tactical_rate
policy_target/pre_tactical_rate
policy_target/reanalyse_rate
policy_target/one_hot_rate
```

### 13.4 q target ログ

```text
q_target/selected_move_only_count
q_target/full_legal_count
q_target/reanalyse_count
q_target/lambda_return_count
q_target/tactical_forced_count
q_target/q_normal_mean
q_target/q_nyugyoku_mean
q_target/q_nyugyoku_nonzero_rate
```

### 13.5 moves_left target ログ

```text
moves_left_target/mean
moves_left_target/p50
moves_left_target/p90
moves_left_target/p99
moves_left_target/max
moves_left_target/max_ply_draw_count
```

### 13.6 validation ログ

```text
target_validation/error_count
target_validation/warn_count
target_validation/invalid_policy_sum_count
target_validation/invalid_terminal_reason_count
target_validation/empty_legal_moves_count
target_validation/q_out_of_range_count
target_validation/outcome_invalid_count
target_validation/moves_left_negative_count
target_validation/dropped_sample_count
```

---

## 14. 実装上の不変条件

以下は必ず守る。

```text
TrainingSample は feature_version / move_label_version を持つ
policy target は合法手上の分布である
policy target の確率和は 1 である
違法手を policy / q target に含めない
target がない head は None として扱う
None target を 0 target として扱わない
outcome_cause class order は固定する
outcome_cause_target は step.side_to_move 視点で作る
q_normal_result と q_nyugyoku は分けて保存する
Q_mode だけを target 正本にしない
λ-return も normal_result と nyugyoku を分けて作る
policy_style は現在手番側の style policy として扱う
policy target の標準は pi_search_post_tactical とする
pi_select は標準では policy target にしない
start_position.source を sample metadata に引き継ぐ
position buffer 由来では source game result と restart game result を混同しない
moves_left_target は log1p(max(T - t, 0)) とする
terminal node は標準では TrainingSample にしない
reanalyse target は視点と search_config_id を保存する
tactical forced q target は根拠をログに残す
```

---

## 15. 検証項目

### 15.1 unit test

最低限、以下を unit test にする。

```text
terminal_reason から outcome_category への変換
GameResult から side_to_move 視点 outcome_cause_target への変換
outcome_cause_target から q_normal_result / q_nyugyoku への変換
policy target の合法手検証
policy target の確率和検証
normal mode の policy_normal_target 割り当て
style mode の policy_style_target 割り当て
both target の loss weight 設定
moves_left_target の log1p 計算
position buffer 由来対局で restart game result を使うこと
reanalyse target の視点変換
tactical forced target の q 成分割り当て
None target が loss に入らないこと
```

### 15.2 integration test

自己対局ログ1局から training sample を作り、以下を確認する。

```text
step 数と sample 数が一致する
terminal step が sample 化されていない
各 sample の legal_moves が空でない
selected_move が legal_moves に含まれる
policy target の moves が legal_moves の部分集合である
q target の moves が legal_moves の部分集合である
outcome target が全 step で手番視点になっている
勝者側 step と敗者側 step で outcome が反転している
moves_left_target が終局に近いほど小さい
```

### 15.3 distribution check

データセット作成後、以下を確認する。

```text
outcome 5クラスの件数
mode 別 sample 件数
start_position.source 別 sample 件数
policy_normal_target と policy_style_target の件数
q_nyugyoku 非ゼロ target 件数
reanalyse target 件数
max_ply_draw 件数
shuffled_initial 由来 sample 件数
```

---

## 16. 実験項目

### 16.1 policy target source

以下を比較する。

```text
A. pi_klent_pre_tactical
B. pi_search_post_tactical
C. pi_select
D. pi_reanalyse 優先
```

評価指標は以下である。

```text
通常勝率
詰み逃し率
policy entropy
KLENT-0 と Gumbel MCTS の差
入玉宣言勝ち率
相手入玉負け率
```

### 16.2 style target mixing

以下を比較する。

```text
A. style 系 sample は policy_style のみに入れる
B. bakuretsu sample の一部を policy_normal に低重みで入れる
C. style head warmup として normal target を 0.05 〜 0.2 で入れる
D. policy_normal と policy_style を完全分離する
```

通常棋力が落ちる場合は、`policy_normal` への style 混入を下げる。

### 16.3 q target source

以下を比較する。

```text
A. MC selected move target
B. λ-return selected move target
C. Gumbel MCTS reanalyse full-action target
D. tactical forced target 追加
```

評価指標は以下である。

```text
q_normal_result calibration
q_nyugyoku calibration
Q上位手の勝率
入玉有望局面での選択精度
相手入玉リスク局面での回避率
```

### 16.4 moves_left weight

以下を比較する。

```text
w_moves_left = 0.0
w_moves_left = 0.05
w_moves_left = 0.1
w_moves_left = 0.2
```

入玉関連指標が悪化する場合は、`w_moves_left` を下げる。

### 16.5 source weight

開始局面 source 別に以下を比較する。

```text
shuffled_initial sample を通常 weight にする
shuffled_initial sample の policy_normal weight を下げる
position_buffer の入玉 category を oversampling する
max_ply_draw sample の moves_left weight を下げる
```

### 16.6 outcome balancing

以下を比較する。

```text
入玉 outcome oversampling なし
win_nyugyoku / loss_nyugyoku oversampling
q_nyugyoku 非ゼロ sample oversampling
outcome class weight
position buffer quota 調整
```

確率 calibration を壊さないことを重視する。

---

## 17. 未決事項

現時点での未決事項は以下である。

```text
λ-return を初期実装から有効にするか、MC target で開始するか
pi_select を policy target に混ぜる実験をいつ行うか
reanalyse outcome_cause_reanalyse を soft target として使うか
q_nyugyoku の loss weight をどの段階で上げるか
shuffled_initial 由来 sample を policy_normal にどの程度入れるか
max_ply_draw sample の outcome / moves_left weight を下げるか
教師探索の評価値を q_normal_result へ変換する scale
入玉原因不明の外部棋譜をどう扱うか
```

初期実装では、以下を標準とする。

```text
policy target:
  pi_search_post_tactical

q target:
  自己対局は選択手 MC target
  reanalyse がある場合は full-action target で置き換える

outcome target:
  GameResult 由来の hard 5-class target

moves_left target:
  log1p(T - t)

loss weight:
  policy_normal と q_normal_result を優先
  policy_style と q_nyugyoku は低めから開始
```

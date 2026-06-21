# docs/02_overall_architecture.md - 全体アーキテクチャ仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎の **全体アーキテクチャ** を定義する。

対象は以下である。

```text
主要コンポーネントの責務分担
実行時データフロー
学習時データフロー
コンポーネント間インターフェース
version 管理境界
エラー・異常系の扱い
実装レイヤ分割
可観測性
実装上の不変条件
検証項目
実験項目
```

このドキュメントは、個別仕様の正本ではない。

個別仕様は以下を正とする。

```text
探索・着手選択: docs/03_search.md
ネットワーク本体: docs/04_network.md
ネットワーク出力: docs/05_outputs.md
探索価値・視点変換: docs/06_search_value.md
CPU tactical solver: docs/07_tactical_solver.md
自己対局ループ: docs/08_selfplay.md
開始局面多様化: docs/09_start_position_diversity.md
学習ターゲット: docs/10_training_targets.md
指し手ラベル: docs/11_move_label.md
実装・実験ロードマップ: docs/12_roadmap.md
KLENT-first 学習・探索方針: docs/13_klent_training.md
評価仕様: docs/14_evaluation.md
学習パイプライン: docs/15_training_pipeline.md
全体方針: 爆裂玉登太郎_設計概要.md
```

このドキュメントでは、それらをどのように接続するかを扱う。

---

## 1. 基本方針

### 1.1 概要と詳細仕様の中間層にする

`爆裂玉登太郎_設計概要.md` は、長期的に変わりにくい設計判断だけを書く概要である。

一方、`docs/03_search.md` 以降は、探索、network、target、自己対局、評価などの個別仕様である。

`docs/02_overall_architecture.md` は、その中間層として以下を担う。

```text
全体の構成単位を定義する
コンポーネントの責務境界を明確にする
実行時・学習時・評価時のデータの流れを整理する
個別仕様間で共有する interface と version 境界を整理する
```

したがって、このドキュメントでは、head の詳細 shape、loss 式、探索式、評価局面セットの細部を重複定義しない。

### 1.2 通常棋力と入玉棋風を同時に扱う

爆裂玉登太郎の目的は、通常の将棋として十分に強く、なおかつ入玉宣言勝ちを自然に狙えることである。

そのため、全体アーキテクチャは以下を分けて扱う。

```text
通常棋力:
  普通に勝つ力

入玉棋風:
  自玉の入玉勝ちを狙う力

相手入玉耐性:
  相手の入玉勝ちを防ぐ力

短期戦術:
  詰み、王手対応、宣言勝ち可能判定などの確定手処理
```

これらを単一の scalar や単一の総合 score に潰さない。

### 1.3 主学習ループは KLENT-0 + CPU tactical solver にする

標準の自己対局生成は以下を主軸にする。

```text
KLENT-0
+
CPU tactical solver
+
batched GPU inference
```

全局面で重い MCTS を行うのではなく、network の policy と action-value から root 局面の改善方策を作る。

目的は以下である。

```text
自己対局局面数を増やす
更新回数を増やす
入玉関連局面を効率よく集める
ログと target 生成を単純に保つ
```

ただし、将棋では短期戦術を network の汎化だけに任せると危険である。そこで、宣言勝ち可能判定、相手玉への詰み探索、王手対応は CPU tactical solver が補強する。

### 1.4 Gumbel MCTS は補助的に使う

Gumbel MCTS は、初期の主自己対局ループでは全局面に使わない。

用途は以下に限定する。

```text
Gumbel MCTS reanalyse
難所局面解析
実戦時 test-time search
KLENT-0 との差分測定
```

Gumbel MCTS は、自己対局全体を重くするためのものではなく、KLENT-0 だけでは不安定な局面を補正する policy improvement operator として扱う。

### 1.5 通常棋力を壊さない順序で増強する

全体の学習順序は、通常棋力を先に安定させてから、入玉関連データを段階的に増やす。

標準順序は以下である。

```text
1. 教師あり初期化で最低限の policy / Q / outcome を作る
2. KLENT-0 normal self-play を安定させる
3. 開始局面多様化と position buffer を導入する
4. style 系 mode を少量から増やす
5. 入玉関連局面を厚くする
6. Gumbel MCTS reanalyse で難所を補正する
7. formal evaluation gate で checkpoint と探索設定を選ぶ
```

`policy_normal_head` に入玉優先 target を大量に混ぜてはならない。style 系 target は原則として `policy_style_head` に流す。

---

## 2. 全体構成

### 2.1 論理構成

爆裂玉登太郎の論理構成は以下である。

```text
Rule / Position Core
  -> FeatureEncoder
  -> Policy / Action-Value / Outcome Network
  -> SearchValueConverter
  -> SearchEngine
       -> KLENT-0
       -> Gumbel MCTS
       -> CPU TacticalSolver
  -> SelfPlayLoop
  -> GameLog / PositionBuffer / ReanalyseCandidateQueue
  -> SampleBuilder / DatasetBuilder
  -> Trainer
  -> CheckpointManager
  -> EvaluationGateRunner

TrainingPipelineManager
  -> self-play / reanalyse / dataset build / trainer / evaluation gate を統括する
```

より大きな循環は以下である。

```text
checkpoint
  -> self-play / reanalyse
  -> raw logs
  -> TrainingSample dataset
  -> trainer
  -> candidate checkpoint
  -> evaluation gate
  -> promoted checkpoint
  -> next self-play
```

### 2.2 実行時構成

対局時または自己対局時の標準構成は以下である。

```text
Position
  -> legal move generation
  -> terminal / rule check
  -> CPU tactical pre-check
  -> FeatureEncoder
  -> InferenceServer
  -> NetworkOutput
  -> search value conversion
  -> KLENT-0 or Gumbel MCTS
  -> CPU tactical post-check
  -> selected move
  -> SearchResult
```

terminal node では network inference を行わない。

宣言勝ち可能局面では、通常の network policy label ではなく、探索側の rule action として処理する。

### 2.3 学習時構成

学習時の標準構成は以下である。

```text
SelfPlayGame logs
  -> SampleBuilder
  -> TrainingSample
  -> DatasetBuilder
  -> DatasetSampler
  -> FeatureEncoder
  -> Network
  -> loss
  -> optimizer
  -> checkpoint
```

Gumbel MCTS reanalyse が入る場合は以下を追加する。

```text
SelfPlayGame logs
  -> ReanalyseCandidateQueue
  -> Gumbel MCTS reanalyse
  -> reanalyse dataset
  -> ReanalyseTargetMerger
  -> TrainingSample dataset
```

position buffer は、データ生成側へ戻る循環として扱う。

```text
SelfPlayGame logs
  -> PositionBufferWriter
  -> position buffer
  -> StartPositionSampler
  -> future SelfPlayWorker
```

---

## 3. コンポーネント責務

### 3.1 Rule / Position Core

Rule / Position Core は、カスタム Yasai を局面状態・合法手・move 適用・局面履歴・ルール判定の正本として利用する。
`shogi_core` の基本型をそのまま共有し、玉登太郎側で盤面表現、駒移動規則、千日手履歴、宣言勝ち計算、一手詰め探索を再実装しない。

責務は以下である。

```text
Yasai Position と shogi_core 基本型の公開
SFEN / USI 変換
Yasai への合法手生成、move 適用 / undo / null move の委譲
Yasai の千日手・連続王手・優等局面・劣等局面・宣言勝ち・一手詰め判定の公開
合法手なし、千日手・連続王手、最大手数を合成する terminal 判定
固定 move label との変換
```

千日手履歴、宣言勝ち計算、一手詰め探索の正本は Yasai にあり、Yolts は終局方針の合成のみを担う。
`Superior` / `Inferior` は探索省略用の関係であり、終局にはしない。

以下は通常の終局理由ではなく、実装バグとして扱う。

```text
合法手生成異常
move 適用異常
position desync
illegal move が合法手として選ばれる状態
```

これらを `loss_normal`、`draw`、`max_ply_draw` などに変換してはならない。

### 3.2 MoveLabelEncoder

MoveLabelEncoder は、合法手と network 出力空間を接続する。

責務は以下である。

```text
通常着手を固定長 label に encode する
label を通常着手へ decode する
現在手番視点の座標変換を適用する
legal move mask を作る
```

初期実装では以下を使う。

```text
MOVE_LABEL_PLANES = 27
MOVE_LABEL_SIZE = 81 * 27 = 2187
```

label が表すのは以下だけである。

```text
盤上移動
盤上移動 + 成り
持ち駒打ち
```

以下は move label に含めない。

```text
宣言勝ち action
投了
千日手成立
持将棋引き分け
最大手数到達
illegal move marker
```

### 3.3 FeatureEncoder

FeatureEncoder は、Position を network input へ変換する。

責務は以下である。

```text
現在手番視点への正規化
盤上駒特徴の生成
利き特徴の生成
利き数特徴の生成
持ち駒特徴の生成
王手状態特徴の生成
入玉関連特徴の生成
feature_version の管理
```

network input は以下とする。

```python
features: Tensor  # [B, INPUT_CHANNELS, 9, 9]
```

反復回数、手数、最大手数までの残り、直前手種別は、初期実装の標準入力には含めない。

### 3.4 Network

Network は、局面から policy、action-value、outcome、moves_left を予測する。

責務は以下である。

```text
VoVNetV2-style shared trunk による盤面特徴抽出
head adapter による head 別特徴変換
policy_normal_head 出力
policy_style_head 出力
q_normal_result_head 出力
q_nyugyoku_head 出力
outcome_cause_head 出力
moves_left_head 出力
```

forward の戻り値は、概念的に以下である。

```python
@dataclass
class NetworkOutput:
    policy_normal_logits: Tensor
    policy_style_logits: Tensor
    q_normal_result_raw: Tensor
    q_nyugyoku_raw: Tensor
    outcome_cause_logits: Tensor
    moves_left_pred: Tensor
```

network module 内では以下を行わない。

```text
合法手 mask
policy softmax
q raw への tanh
policy_normal / policy_style の混合
Q_mode の合成
KLENT 方策改善
```

これらは探索側、変換層、loss 側の責務である。

### 3.5 SearchValueConverter

SearchValueConverter は、NetworkOutput を探索で使える値へ変換する。

責務は以下である。

```text
合法手 mask の適用
policy_normal / policy_style の合法手 softmax
探索モードに応じた policy prior の混合
q raw への tanh 適用
Q_normal_result / Q_nyugyoku / Q_total_result / Q_mode の生成
outcome_cause_logits から V 成分を作る
moves_left_pred の逆変換
視点情報の付与
```

`Q_mode` と `V_mode` は探索モード依存の派生値である。

正本として保存・target 化する値は、原則として以下に分ける。

```text
Q_normal_result
Q_nyugyoku
V_normal_result
V_nyugyoku
outcome_cause
```

### 3.6 SearchEngine

SearchEngine は、現在局面から着手を選ぶ。

責務は以下である。

```text
SearchRequest を受け取る
合法手を取得する
terminal / rule result を確認する
network inference を呼び出す
探索値変換を呼び出す
KLENT-0 または Gumbel MCTS を実行する
CPU tactical solver の結果を統合する
SearchResult を返す
```

標準探索は以下である。

```text
search_type = klent0
num_simulations = 0
CPU tactical solver enabled
```

Gumbel MCTS は、reanalyse、難所解析、実戦時探索で使う。

### 3.7 CPU TacticalSolver

CPU TacticalSolver は、network と KLENT-0 だけでは落としやすい短期戦術・ルール上の確定手を補う。

責務は以下である。

```text
宣言勝ち可能判定
相手玉への詰み探索
王手対応
勝ち確定手の forced move 化
探索方策への mask / 補正
tactical 結果のログ化
```

CPU TacticalSolver は、通常の局面評価を作る主探索ではない。

標準の優先順位は以下である。

```text
1. terminal / rule result
2. 宣言勝ち可能手
3. 相手玉への詰みが証明された手
4. KLENT-0 または Gumbel MCTS の探索方策
```

### 3.8 SelfPlayLoop

SelfPlayLoop は、学習用対局を生成する。

責務は以下である。

```text
開始局面を取得する
先手・後手の探索モードを割り当てる
SearchEngine を使って対局を進める
終局結果を保存する
各 step の探索結果を保存する
position buffer 用局面を抽出する
reanalyse candidate を抽出する
SelfPlayGame log を出力する
```

SelfPlayLoop は学習そのものを行わない。

loss weight、mini-batch sampling、checkpoint 更新は training pipeline 側の責務である。

### 3.9 StartPositionSampler / PositionBuffer

StartPositionSampler は、自己対局の開始局面を供給する。

標準 source は以下である。

```text
initial
random_opening
shuffled_initial
position_buffer
```

`manual` と `test` は、通常の自己対局 sampling 対象にはしない。

position buffer は、過去の自己対局から抽出した有用局面を保存する。

対象例は以下である。

```text
通常中盤局面
入玉有望局面
相手入玉リスク局面
相入玉になりやすい局面
policy / Q が不確実な局面
tactical に難しい局面
KLENT-0 と Gumbel MCTS の差が大きい局面
```

### 3.10 SampleBuilder / DatasetBuilder

SampleBuilder は、raw log から TrainingSample を作る。

責務は以下である。

```text
SelfPlayGame log の読み込み
各 step から TrainingSample を生成
終局結果から outcome_cause_target を生成
選択手または reanalyse から Q target を生成
policy target を head 別に割り当てる
moves_left target を生成
target 欠損を None として保持する
```

DatasetBuilder は、TrainingSample を dataset shard と registry に保存する。

以下を守る。

```text
欠損 target と 0 target を混同しない
違法手に policy / Q target を持たせない
Q_mode を target の正本にしない
失敗対局を training dataset に混ぜない
```

### 3.11 Trainer

Trainer は network を更新する。

責務は以下である。

```text
dataset sampling
feature encode
network forward
head 別 loss 計算
合法手 mask 適用
target mask 適用
backward
optimizer step
checkpoint 保存
training metrics 保存
```

Trainer は target の意味を再定義しない。

### 3.12 CheckpointManager

CheckpointManager は、checkpoint と metadata を保存する。

責務は以下である。

```text
model state 保存
optimizer / scheduler state 保存
EMA state 保存
version 保存
training config 保存
dataset lineage 保存
evaluation lineage 保存
promotion status 保存
```

少なくとも以下を分けて管理する。

```text
latest_checkpoint
candidate_checkpoint
selfplay_checkpoint
normal_champion_checkpoint
nyugyoku_optimized_config
```

`nyugyoku_optimized_config` は、共通 checkpoint に適用する探索設定であり、別重みの checkpoint を意味しない。

### 3.13 EvaluationGateRunner

EvaluationGateRunner は、checkpoint と探索設定の採用判断を行う。

責務は以下である。

```text
correctness fixture
offline validation
smoke self-play
formal game evaluation
speed evaluation
checkpoint gate 判定
normal_champion_checkpoint 更新
nyugyoku_optimized_config 更新
```

loss が下がっただけでは checkpoint を昇格しない。

通常棋力 gate と入玉最適化 gate は分けて扱う。

### 3.14 TrainingPipelineManager

TrainingPipelineManager は、自己対局生成、reanalyse、dataset build、trainer、evaluation gate を疎結合に接続する実行管理コンポーネントである。

責務は以下である。

```text
phase の決定
self-play run の起動
reanalyse run の起動
dataset build の起動
trainer run の起動
evaluation run の起動
checkpoint gate の判定
artifact registry の更新
```

TrainingPipelineManager は、target の中身、探索計算、loss 計算、評価指標の定義を再実装しない。各専門コンポーネントへ設定を渡し、run id、config id、dataset lineage、checkpoint lineage、evaluation lineage が欠落しないことを検査する。

---

## 4. 主要インターフェース

### 4.1 Position

`Position` は、探索・feature encode・合法手生成の入力である。

最低限、以下を保持する。

```text
盤面
持ち駒
手番
手数または ply
千日手判定に必要な履歴
宣言勝ち判定に必要な状態
```

`Position` は rule layer の所有物であり、network や trainer は直接ルールを再実装しない。
内部の将棋状態と局面履歴は `yasai::Position` だけが保持し、rule layer は SFEN 変換、
特徴量・指し手ラベル、終局方針の合成だけを追加で担う。

優等局面・劣等局面は公式な終局判定ではなく探索省略用の関係であり、
`rule_terminal_outcome` には含めない。

### 4.2 NetworkOutput

`NetworkOutput` は、network forward の唯一の標準出力である。

```python
@dataclass
class NetworkOutput:
    policy_normal_logits: Tensor      # [B, MOVE_LABEL_SIZE]
    policy_style_logits: Tensor       # [B, MOVE_LABEL_SIZE]
    q_normal_result_raw: Tensor       # [B, MOVE_LABEL_SIZE]
    q_nyugyoku_raw: Tensor            # [B, MOVE_LABEL_SIZE]
    outcome_cause_logits: Tensor      # [B, 5]
    moves_left_pred: Tensor           # [B, 1]
```

shape、意味、version は `docs/05_outputs.md` を正とする。

### 4.3 ConvertedNetworkOutput

`ConvertedNetworkOutput` は、探索側が使う変換済み出力である。

```python
@dataclass
class ConvertedNetworkOutput:
    legal_moves: list[int]
    p_normal: np.ndarray
    p_style: np.ndarray
    p_prior: np.ndarray
    q_normal_result: np.ndarray
    q_nyugyoku: np.ndarray
    q_total_result: np.ndarray
    q_mode: np.ndarray
    outcome_probs: np.ndarray
    v_normal_result: float
    v_nyugyoku: float
    v_total_result: float
    v_mode: float
    moves_left: float | None
    perspective_side: int
```

実装上は、`ActionValue` と `OutcomeValue` に分けてよい。

### 4.4 SearchRequest / SearchResult

`SearchRequest` は、SearchEngine への入力である。

```python
@dataclass
class SearchRequest:
    position: Position
    root_side: int
    config: SearchModeConfig
    game_ply: int | None = None
    max_game_ply: int | None = None
    repetition_state: object | None = None
    time_control: object | None = None
    rng: object | None = None
```

`SearchResult` は、着手選択結果と学習ログに必要な情報を返す。

```python
@dataclass
class SearchResult:
    selected_move: int | None
    terminal_action: str | None
    legal_moves: list[int]
    policy_prior: object | None
    pi_klent: object | None
    pi_search: object | None
    q_normal_result: object | None
    q_nyugyoku: object | None
    q_total_result: object | None
    q_mode: object | None
    outcome_probs: object | None
    moves_left: float | None
    search_type: str
    search_config_id: str
    tactical_flags: dict
    terminal_flags: dict
```

`selected_move` は通常着手の move label である。

宣言勝ちなど通常着手でない action は `terminal_action` で表す。

### 4.5 SelfPlayGame / SelfPlayStep

`SelfPlayGame` は、自己対局1局の raw log である。

少なくとも以下を含む。

```text
game_id
config_id
model_version
feature_version
move_label_version
start_position
pairing_id
sente_player
gote_player
steps
result
```

`SelfPlayStep` は、1手ごとの探索結果である。

少なくとも以下を含む。

```text
ply
side_to_move
sfen または features_ref
legal_moves
selected_move または terminal_action
mode_id
target_policy_head
p_prior
pi_klent_pre_tactical
pi_search_post_tactical
pi_select
q_normal_result_pred
q_nyugyoku_pred
q_total_result_pred
q_mode_pred
outcome_probs
moves_left_pred
tactical_flags
terminal_flags_before_move
nyugyoku_features
uncertainty_features
reanalyse_tags
```

### 4.6 TrainingSample

`TrainingSample` は、trainer が読む学習サンプルである。

正本 target は以下である。

```text
policy_normal_target
policy_style_target
q_normal_result_target
q_nyugyoku_target
outcome_cause_target
moves_left_target
```

派生値である以下を target の正本にしない。

```text
Q_total_result
Q_mode
V_total_result
V_mode
```

### 4.7 EvaluationResult

`EvaluationResult` は、checkpoint / config の採用判断に使う評価結果である。

少なくとも以下を含む。

```text
eval_run_id
checkpoint_id
mode_id
eval_suite_id
baseline_id
valid_games
wins
losses
draws
win_normal_games
win_declaration_games
loss_nyugyoku_games
failed_games
invalid_games
speed metrics
resource metrics
```

rate は raw count から再計算できる派生値として扱う。

---

## 5. 実行時データフロー

### 5.1 KLENT-0 標準着手フロー

標準の着手フローは以下である。

```text
1. Position から合法手を生成する
2. terminal / draw / 宣言勝ち可能判定を行う
3. CPU tactical pre-check を行う
4. FeatureEncoder で入力特徴量を作る
5. InferenceServer で NetworkOutput を得る
6. SearchValueConverter で探索値へ変換する
7. KLENT-0 方策改善を行う
8. CPU tactical post-check で方策を補正する
9. sampling または argmax で着手を選ぶ
10. SearchResult を返す
```

自己対局では通常 sampling を使う。

評価対局では deterministic argmax を標準とする。

### 5.2 forced move フロー

以下の場合は、network 出力や KLENT-0 より優先して処理する。

```text
terminal node
宣言勝ち可能
相手玉への詰みが証明された
王手対応で唯一合法応手
```

ただし、王手中でも複数の合法応手がある場合は、合法応手集合上で探索方策を使う。

### 5.3 Gumbel MCTS フロー

Gumbel MCTS の基本入力は KLENT-0 と同じである。

```text
Position
legal moves
NetworkOutput
ConvertedNetworkOutput
SearchModeConfig
TacticalResult
```

違いは、root だけでなく leaf node でも network を評価し、outcome 分布を視点変換して backup する点である。

Gumbel MCTS の結果は以下に使う。

```text
pi_reanalyse
Q_normal_result_reanalyse
Q_nyugyoku_reanalyse
実戦時 SearchResult
KLENT-0 との差分分析
```

---

## 6. 学習データフロー

### 6.1 教師あり初期化

初期化では、既存エンジン、簡易探索、詰み探索、過去棋譜などから最低限の policy / Q / outcome を作る。

標準フローは以下である。

```text
teacher data
  -> supervised dataset
  -> trainer
  -> seed checkpoint
```

目的は、KLENT-0 自己対局を始められる最低限の policy と Q を作ることである。

### 6.2 自己対局ログからの dataset build

標準フローは以下である。

```text
SelfPlayGame logs
  -> GameLogReader
  -> SampleBuilder
  -> TargetValidator
  -> DatasetWriter
  -> DatasetRegistry
```

各 step から原則1つの TrainingSample を作る。

terminal 局面そのものは、標準では TrainingSample にしない。

### 6.3 reanalyse の混合

reanalyse は、全局面ではなく難所局面を対象にする。

```text
SelfPlayGame logs
  -> ReanalyseCandidateQueue
  -> Gumbel MCTS reanalyse
  -> reanalyse results
  -> ReanalyseTargetMerger
  -> dataset update
```

reanalyse target は、policy と action-value を成分別に保存する。

`Q_mode` だけに潰して保存してはならない。

### 6.4 dataset sampling

DatasetSampler は、dataset 内の raw count を変更しない。

mini-batch sampling で以下を制御する。

```text
source 比率
mode 比率
outcome 比率
start_position.source 比率
tag 比率
reanalyse sample 比率
同一 game / 同一局面の重複抑制
```

入玉関連 sample が不足する場合は、まず position buffer category quota、mode pairing、outcome oversampling、reanalyse candidate を調整する。

---

## 7. version と互換性境界

### 7.1 固定すべき version

大規模学習を始める前に、少なくとも以下を固定する。

```text
rule_version
feature_version
move_label_version
network_output_version
target_version
selfplay_log_version
dataset_version
checkpoint_version
search_config_version
eval_suite_version
```

これらは、checkpoint、dataset、reanalyse result、evaluation result の互換性に直結する。

### 7.2 checkpoint load 条件

checkpoint を load する際は、少なくとも以下の一致を確認する。

```text
feature_version
move_label_version
network_output_version
checkpoint_version
```

一致しない場合は、明示的な migration を実装しない限り load を拒否する。

### 7.3 dataset load 条件

dataset を trainer に渡す際は、少なくとも以下の一致を確認する。

```text
rule_version
feature_version
move_label_version
target_version
dataset_version
```

一致しない場合、既存 sample をそのまま再利用してはならない。

### 7.4 config id

全ての生成物には、生成に使った config id を保存する。

対象は以下である。

```text
search_config_id
selfplay_config_id
start_position_config_id
pairing_schedule_id
dataset_build_config_id
dataset_mix_config_id
trainer_config_id
eval_config_id
```

あとから、checkpoint、dataset、自己対局ログ、評価結果を相互に辿れるようにする。

---

## 8. 探索モードと設定境界

### 8.1 SearchModeConfig

探索モードは、主に以下で表す。

```text
alpha_style_self
alpha_style_opponent_model
lambda_nyugyoku
klent_alpha_entropy
klent_beta_prior
temperature
deterministic
search_type
num_simulations
tactical solver flags
```

代表モードは以下である。

```text
normal
nyugyoku
anti_nyugyoku
mutual_nyugyoku
bakuretsu
```

### 8.2 root側と相手モデル側を分ける

`policy_style_head` は、常に「その局面で手番を持つ側にとっての style policy」である。

したがって、探索設定では以下を分ける。

```text
alpha_style_self:
  root側の style 混合率

alpha_style_opponent_model:
  相手側 node の style 混合率
```

KLENT-0 では root 局面だけを評価するため、通常は `alpha_style_self` が効く。

Gumbel MCTS では、node の手番が root 側か相手側かで使い分ける。

### 8.3 checkpoint と探索設定を分ける

同じ model checkpoint に対して、探索設定を変えることで棋風を切り替える。

以下は別概念である。

```text
normal_champion_checkpoint:
  通常モード gate を満たす model weight

nyugyoku_optimized_config:
  共通 checkpoint に適用する入玉最適化用探索設定
```

入玉最適化は、別の重みを必ず作ることではなく、checkpoint と探索設定の組み合わせを評価して選ぶことである。

---

## 9. エラー・異常系の扱い

### 9.1 通常の終局理由

通常の終局理由は、学習 target と評価集計に使える。

例:

```text
checkmate
resign
normal_adjudication
declaration_win
nyugyoku_adjudication
repetition_draw
jishogi_draw
max_ply_draw
```

これらは `outcome_category` に変換できる。

### 9.2 実装・実行異常

以下は通常の終局理由ではない。

```text
illegal move selected
legal move generation error
move apply error
position desync
feature encode error
move label encode / decode error
network NaN / inf
SearchResult schema error
SelfPlayGame schema error
TrainingSample schema error
engine crash
external process timeout
```

これらは quarantine に送る。

学習 target に変換してはならない。

### 9.3 quarantine

quarantine には以下を保存する。

```text
error_type
error_message
game_id
step_index
sfen
side_to_move
config_id
checkpoint_id
stack trace または診断ログ
再現用 random seed
```

quarantine data は通常 training set に混ぜない。

---

## 10. ログと可観測性

### 10.1 self-play log

SelfPlayGame log では、少なくとも以下を保存する。

```text
config / version / checkpoint
開始局面 metadata
player pairing
各 step の SearchResult
終局結果
reanalyse tags
position buffer tags
```

### 10.2 search log

SearchResult では、少なくとも以下を保存する。

```text
legal_moves
selected_move または terminal_action
policy_prior
pi_klent_pre_tactical
pi_search_post_tactical
q_normal_result
q_nyugyoku
q_mode
outcome_probs
moves_left
tactical_flags
terminal_flags
search_config_id
```

保存容量の都合で dense ではなく sparse にしてよい。

`q_total_result` と `q_mode` は解析用に保存してよい派生値である。互換性と再解析の正本は `q_normal_result` と `q_nyugyoku` に分けて保持する。

ただし、合法手集合、確率、value の対応関係を失ってはならない。

### 10.3 dataset log

dataset registry では、少なくとも以下を保存する。

```text
dataset_id
source counts
mode counts
outcome counts
start_position.source counts
tag counts
version
parent raw logs
reanalyse dataset lineage
validation result
```

### 10.4 checkpoint log

checkpoint metadata では、少なくとも以下を保存する。

```text
checkpoint_id
parent_checkpoint_id
trainer_run_id
dataset_id
network version
optimizer state 有無
EMA state 有無
evaluation lineage
promotion status
```

### 10.5 evaluation log

評価では、rate だけでなく raw count を保存する。

```text
valid_games
wins
losses
draws
win_normal_games
win_declaration_games
win_nyugyoku_adjudication_games
loss_normal_games
loss_nyugyoku_games
repetition_draw_games
jishogi_draw_games
max_ply_draw_games
failed_games
invalid_games
```

---

## 11. 実装レイヤ分割

### 11.1 core layer

```text
Position
Move
MoveLabelEncoder
RuleTerminalOutcome
```

この layer は、network や training に依存しない。

### 11.2 feature / model layer

```text
FeatureEncoder
Network
NetworkOutput
Checkpoint load / save
```

この layer は、Position を入力として受け取れるが、探索方策や学習 target を定義しない。

### 11.3 search layer

```text
SearchModeConfig
SearchRequest
SearchValueConverter
KLENT0PolicyImprover
GumbelMCTS
TacticalSolver
SearchEngine
SearchResult
```

この layer は、network output を使って着手を選ぶ。

loss や optimizer には依存しない。

### 11.4 data generation layer

```text
SelfPlayManager
SelfPlayWorker
InferenceServer
GameLogWriter
PositionBufferWriter
ReanalyseCandidateQueue
StartPositionSampler
```

この layer は、対局を生成し、raw log を保存する。

### 11.5 dataset / training layer

```text
GameLogReader
SampleBuilder
LambdaReturnBuilder
ReanalyseTargetMerger
TargetValidator
DatasetWriter
DatasetSampler
Trainer
```

この layer は、raw log から TrainingSample を作り、network を更新する。

### 11.6 evaluation / registry layer

```text
TrainingPipelineManager
EvaluationRunner
EvaluationGateRunner
DatasetRegistry
CheckpointManager
ArtifactRegistry
ReportBuilder
```

この layer は、checkpoint と成果物の採用判断、再現性、比較可能性を管理する。

---

## 12. 実装上の不変条件

以下は、初期実装から必ず守る。

```text
現在手番視点を feature / move label / policy / Q / outcome target で揃える
MOVE_LABEL_SIZE は指し手ラベル仕様と一致させる
宣言勝ちは move label に含めない
network module 内で合法手 mask を行わない
network module 内で policy softmax を行わない
network module 内で q raw に tanh を適用しない
合法手 mask は探索側・loss 側で適用する
違法手には policy / Q target を持たせない
terminal node では network inference を行わない
Q_mode / V_mode は派生値であり target の正本にしない
normal_result 成分と nyugyoku 成分を分けて保存する
欠損 target と 0 target を区別する
自己対局 loop は学習そのものを行わない
失敗対局や異常 sample を training dataset に混ぜない
評価失敗を勝敗として集計しない
checkpoint / dataset / log には version と config id を保存する
```

---

## 13. 検証項目

### 13.1 interface round-trip

```text
Position -> SFEN -> Position が一致する
Move -> label -> Move が一致する
後手番で current-side 座標変換が正しい
合法手集合と legal label 集合が一致する
```

### 13.2 network interface

```text
NetworkOutput の shape が仕様通りである
MOVE_LABEL_SIZE と policy / Q head の次元が一致する
outcome_cause_logits の class 数が 5 である
network forward が合法手情報に依存しない
NaN / inf を検出できる
```

### 13.3 search integration

```text
terminal node で network inference が呼ばれない
宣言勝ち可能局面で terminal_action が返る
詰みが証明された局面で forced move が返る
合法手以外に確率が割り当てられない
pi_search の合法手上の確率和が 1 である
Q_mode が Q_normal_result と Q_nyugyoku から再計算できる
```

### 13.4 self-play log

```text
SelfPlayGame schema が通る
各 step に version と config id が残る
selected_move が legal_moves に含まれる
terminal_action と selected_move が混在しない
終局結果が outcome category に変換できる
失敗対局が quarantine に分離される
```

### 13.5 TrainingSample

```text
policy target が合法手だけを含む
policy target の確率和が 1 である
q target が合法手だけを含む
None target と 0 target が区別される
outcome_cause_target の class order が固定されている
moves_left_target が終局までの手数と一致する
```

### 13.6 training pipeline

```text
dataset version と checkpoint version の互換性を検査できる
DatasetSampler が target の値を変更しない
Trainer が合法手 mask を loss 側で適用する
checkpoint metadata から dataset lineage を辿れる
評価 gate なしで promoted checkpoint にしない
```

### 13.7 evaluation

```text
failed_games と valid_games が分離される
raw count から rate を再計算できる
normal gate と nyugyoku optimization gate が分離される
同一 eval suite / seed / mode で再現できる
```

---

## 14. 実験項目

全体アーキテクチャとしての実験項目は以下である。

```text
KLENT-0 の自己対局 throughput
CPU tactical solver の forced rate
batched GPU inference の batch size / latency
SelfPlayGame log の保存容量
sparse policy / Q 保存形式
feature tensor 保存 vs 再生成
position buffer のカテゴリ別効果
reanalyse candidate 抽出条件
Gumbel MCTS reanalyse 比率
DatasetMixConfig の source / mode / outcome 比率
policy_style warmup schedule
normal_champion_checkpoint 更新頻度
nyugyoku_optimized_config 探索範囲
formal evaluation 実行頻度
```

実験では、通常棋力 gate と入玉最適化 gate を混同しない。

---

## 15. 初期実装のスコープ

初期実装で必須とするものは以下である。

```text
Rule / Position Core
MoveLabelEncoder
FeatureEncoder
Network forward
SearchValueConverter
KLENT-0
CPU TacticalSolver の宣言勝ち・詰み・王手対応
SelfPlayGame log
SampleBuilder
Trainer
CheckpointManager
最低限の EvaluationGateRunner
最低限の TrainingPipelineManager
```

初期実装で optional とするものは以下である。

```text
Gumbel MCTS reanalyse の本格運用
NN cache
Monte-Carlo Graph Search
Transformer optional block
大規模分散 trainer
高度な自動 hyperparameter 探索
moves_left を使った探索価値補正
```

optional 項目は、縦の最小構成が安定してから導入する。

---

## 16. 未決事項

全体アーキテクチャ上の未決事項は以下である。

```text
InferenceServer を self-play 専用にするか、評価・reanalyse と共通化するか
feature tensor を log に保存するか、常に SFEN から再生成するか
SelfPlayGame log と TrainingSample dataset の保存形式
reanalyse result を既存 sample に merge するか、別 shard として保持するか
quarantine の保存期間と再処理方法
EMA checkpoint を self-play に使うか raw checkpoint を使うか
formal evaluation の実行頻度
大規模化時の artifact registry の形式
```

これらは、実装後の速度、保存容量、通常勝率、入玉指標、再現性を見て決める。

---

## 17. 暫定結論

`docs/02_overall_architecture.md` では、爆裂玉登太郎を以下の循環として定義する。

```text
Rule / Position Core
  -> FeatureEncoder
  -> Network
  -> SearchValueConverter
  -> KLENT-0 + CPU tactical solver
  -> SelfPlayGame log
  -> TrainingSample dataset
  -> Trainer
  -> Checkpoint
  -> EvaluationGate
  -> 次の自己対局
```

重要な原則は以下である。

```text
主学習ループは KLENT-0 + CPU tactical solver とする
Gumbel MCTS は reanalyse / 難所解析 / 実戦時探索に使う
通常棋力を先に安定させてから入玉関連データを増やす
現在手番視点を全 interface で揃える
policy / Q / outcome / target の正本を混同しない
Q_mode / V_mode は派生値として扱う
宣言勝ちは move label に含めず rule action として扱う
自己対局・target 生成・trainer・評価を疎結合に保つ
checkpoint と探索設定を分けて管理する
異常対局を通常の終局結果に変換しない
```

この構成により、通常棋力、入玉棋風、相手入玉耐性、短期戦術、学習安定性をそれぞれ分解して改善できるようにする。

# docs/12_roadmap.md - 実装・実験ロードマップ

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎を実装・学習・評価していくためのロードマップを定義する。

対象は以下である。

```text
実装順序
コンポーネント間の依存関係
phase 別の目的
各 milestone の成果物
各 milestone の通過条件
検証ゲート
評価導入タイミング
実験導入タイミング
失敗時の戻り先
初期実装の範囲
```

このドキュメントは、個別仕様の正本ではない。

個別の仕様は以下を正とする。

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
KLENT-first 学習・探索方針: docs/13_klent_training.md
評価仕様: docs/14_evaluation.md
学習パイプライン: docs/15_training_pipeline.md
全体方針: 爆裂玉登太郎_設計概要.md
```

このロードマップでは、これらの仕様をどの順で実装し、どの時点で次の段階へ進めるかを扱う。

---

## 1. 基本方針

### 1.1 仕様の土台を先に固定する

最初に固定するものは以下である。

```text
盤面表現
合法手生成
終局原因分類
指し手ラベル
現在手番視点の座標変換
NetworkOutput interface
TrainingSample interface
SearchResult interface
```

これらは、checkpoint、自己対局ログ、training dataset、reanalyse dataset の互換性に直結する。

したがって、学習ループを大規模に回し始める前に、少なくとも以下の version は固定する。

```text
rule_version
feature_version
move_label_version
network_output_version
target_version
selfplay_log_version
```

### 1.2 縦に動く最小構成を早く作る

初期実装では、全機能を完璧に作り込むより先に、以下の縦の流れを通す。

```text
局面
  -> feature encoder
  -> network forward
  -> 探索用変換
  -> KLENT-0 着手選択
  -> 1局生成
  -> SelfPlayGame log
  -> TrainingSample
  -> training step
  -> checkpoint
```

ただし、縦に通すために不変条件を緩めてはならない。

特に以下は最初から守る。

```text
合法手 mask は探索側・loss 側で適用する
違法手には policy / Q target を持たせない
network 出力は現在手番視点に揃える
Q_mode は派生値として扱い、target の正本にしない
terminal node では network inference を行わない
宣言勝ちは move label には含めない
合法手生成や move 適用の異常は通常の終局理由にしない
```

### 1.3 通常棋力を壊さない順序で入玉要素を増やす

入玉を強く狙うデータを早すぎる段階で大量に混ぜると、`policy_normal_head` が不自然な棋風を吸収する危険がある。

標準順序は以下である。

```text
1. 通常棋力を支える policy / Q / outcome を先に安定させる
2. KLENT-0 normal self-play を成立させる
3. 入玉関連局面を position buffer で増やす
4. style 系 mode を少量から増やす
5. Gumbel MCTS reanalyse で難所を補正する
6. 実戦時探索と評価で強さを詰める
```

### 1.4 Gumbel MCTS は後段で導入する

主学習ループは `KLENT-0 + CPU tactical solver` とする。

Gumbel MCTS は以下の目的で後段導入する。

```text
再解析
難所局面の深掘り
KLENT-0 との差分測定
実戦時探索
```

全局面の自己対局で最初から Gumbel MCTS を回す設計にはしない。

### 1.5 評価は早く作るが、指標の正本化は段階的に行う

初期段階から最低限の評価対局と単体評価は用意する。

ただし、通常棋力、入玉成功率、相手入玉耐性、探索速度、学習安定性は性質が異なるため、単一の総合点に潰さない。

評価仕様の詳細は `docs/14_evaluation.md` で定義する。

このロードマップでは、評価の導入タイミングと必須ゲートだけを定義する。


### 1.6 棋力目標と入玉最適化目標

爆裂玉登太郎の主要な対外評価 baseline は、以下に固定する。

```text
baseline_engine = 爆裂駒拾太郎V4.2.0
baseline_search = 4手読み
```

爆裂玉登太郎側の標準評価条件は以下である。

```text
search_type = klent0
playouts = 0
enable_tactical_solver = True
enable_declaration_win_check = True
Gumbel MCTS = disabled
move_selection = deterministic argmax
```

ここでの `playouts = 0` は、MCTS / Gumbel MCTS の探索 playout を行わないことを表す。CPU tactical solver による宣言勝ち判定、詰み探索、王手対応は、標準の 0-playout 探索に含める。

棋力目標は以下とする。

```text
通常モード目標:
  爆裂玉登太郎 playouts=0 / normal mode が、
  爆裂駒拾太郎V4.2.0 4手読みに対して勝率 90%以上
```

この目標は、通常棋力を壊していないことを確認するための主要 gate である。

入玉最適化目標は以下とする。

```text
入玉最適化目標:
  爆裂玉登太郎 playouts=0 の勝率 80%以上を維持する設定の中で、
  入玉宣言勝ち割合を最大化する
```

勝率が 80% 未満の設定は、入玉宣言勝ち割合が高くても候補外とする。

主要指標は以下で定義する。

```text
win_rate_vs_bakuretsu_komahiro_v420_depth4
  = wins / games

declaration_win_rate_vs_bakuretsu_komahiro_v420_depth4
  = declaration_win_games / games

declaration_win_share_among_wins
  = declaration_win_games / wins
```

最適化の主指標は `declaration_win_rate_vs_bakuretsu_komahiro_v420_depth4` とする。

`declaration_win_share_among_wins` は補助指標として保存する。これは、勝っている対局のうちどれくらいが入玉宣言勝ちだったかを見るためであり、勝率制約の代替にはしない。

通常モードの 90% 目標と、勝率 80%以上での入玉宣言勝ち最大化目標は分けて扱う。

```text
通常モード 90% 目標:
  通常棋力の hard gate

勝率 80%以上での入玉宣言勝ち最大化:
  入玉棋風・探索設定・学習設定の constrained optimization
```

入玉最適化で得られた設定が通常モードの 90% 目標を満たす必要はない。ただし、通常モードの 90% 目標を満たす checkpoint を常に別途維持する。

---

## 2. milestone 全体像

ロードマップは以下の milestone に分ける。

| milestone | 名前 | 主目的 | 次へ進む条件 |
| --- | --- | --- | --- |
| M0 | 仕様固定・テスト基盤 | version と不変条件を固定する | 主要データ構造と round-trip test が通る |
| M1 | ルール・指し手ラベル | 合法手、終局、move label を安定させる | legal label / decode / terminal test が通る |
| M2 | feature encoder・network | forward と shape を安定させる | 全 head の shape / mask 外部適用 test が通る |
| M3 | target・dataset・loss | 学習 sample を作れるようにする | toy log から loss が計算できる |
| M4 | 教師あり初期化 | 最低限の policy / Q を作る | fixed validation で policy が random を明確に上回る |
| M5 | KLENT-0 search | network 出力から着手できる | KLENT 分布と SearchResult が整合する |
| M6 | CPU tactical solver 統合 | 確定手を取りこぼさない | 宣言勝ち・詰み・王手対応 test が通る |
| M7 | 自己対局 v1 | normal self-play を回す | 失敗対局なしで一定局数を生成できる |
| M8 | 開始局面多様化 | cold start と局面多様性を改善する | source metadata と品質フィルタが通る |
| M9 | position buffer | 有用局面から再開する | buffer sampling と restart target が整合する |
| M10 | style / 入玉増強 | 入玉関連 target を増やす | 通常棋力を維持しつつ入玉指標が改善する |
| M11 | Gumbel MCTS reanalyse | 難所局面を補正する | reanalyse target が成分別に保存される |
| M12 | 実戦時探索・評価強化 | 強さと安定性を詰める | 対 baseline 評価を再現可能に実行できる |
| M13 | scaling・ablation | モデル規模と設定を最適化する | 通常 90% gate と入玉最適化の比較ログが揃う |

milestone 番号は実装上の管理用であり、学習 phase 番号とは別である。

---

## 3. 依存関係

### 3.1 先に完了していないと危険な依存

以下は強い依存である。

```text
指し手ラベル
  -> network policy / Q head
  -> 合法手 mask
  -> SearchResult
  -> TrainingSample

終局原因分類
  -> outcome_cause target
  -> q_normal_result / q_nyugyoku target
  -> self-play result log
  -> evaluation 指標

現在手番視点
  -> feature encoder
  -> move label
  -> policy target
  -> Q target
  -> Gumbel MCTS backup

SelfPlayStep log
  -> SampleBuilder
  -> position buffer
  -> reanalyse candidate
  -> evaluation analysis
```

### 3.2 後回しにしてよい依存

以下は初期 self-play が動いた後でよい。

```text
Gumbel MCTS reanalyse
NN cache
Monte-Carlo Graph Search
moves_left の探索価値補正
Transformer optional block
大規模 model scaling
高度な自動 hyperparameter 調整
```

これらは、基本ループが安定してから効果測定する。

---

## 4. M0: 仕様固定・テスト基盤

### 4.1 目的

最初に、全コンポーネントが共有する version と不変条件を固定する。

### 4.2 成果物

```text
version 定数
共通 dataclass / schema
共通 enum
共通 test utility
局面 fixture
詰み fixture
宣言勝ち fixture
入玉関連 fixture
round-trip test suite
```

最低限用意する version は以下である。

```text
rule_version
feature_version
move_label_version
network_output_version
target_version
selfplay_log_version
search_config_version
```

### 4.3 通過条件

```text
全 enum の order が固定されている
outcome class order が固定されている
MOVE_LABEL_SIZE が固定されている
ログ schema に version が入る
test fixture が deterministic に生成できる
CI またはローカル test command で主要単体テストを一括実行できる
```

---

## 5. M1: ルール・指し手ラベル

### 5.1 目的

探索・学習・network 出力の土台になる合法手と label 空間を安定させる。

### 5.2 実装対象

```text
Yasai を正本とする Position 表現
shogi_core 基本型の共有
SFEN / USI 変換
合法手生成
move 適用 / undo
terminal 判定
通常千日手 / 連続王手の千日手管理
最大手数管理
宣言勝ち可能判定
1手詰め判定
MoveLabelEncoder
legal move labels
softmax_legal 用 mask
```

### 5.3 重要な不変条件

```text
MOVE_LABEL_PLANES = 27
MOVE_LABEL_SIZE = 2187
move label は現在手番視点
宣言勝ちは move label に含めない
通常着手 label は盤上移動・成り移動・持ち駒打ちだけを表す
合法手生成と move label encode の結果は一意である
合法手生成・move 適用の異常はバグとして扱う
駒移動規則と合法手判定を玉登太郎側で再実装しない
優等局面・劣等局面は探索で必要性を確認するまで実装しない
```

### 5.4 通過条件

```text
全合法手を encode できる
encode -> decode -> encode が一致する
後手番局面で current-side 座標変換が正しい
二歩・打ち歩詰め・行き所のない駒が合法手に出ない
詰み局面で terminal 判定できる
宣言勝ち可能局面を検出できる
通常千日手と連続王手の千日手を区別できる
1手詰め判定後に局面と履歴が復元される
最大手数到達は max_ply_draw として扱える
```

---

## 6. M2: feature encoder・network

### 6.1 目的

現在手番視点の入力特徴量から、全 head の raw output を返す network を実装する。

### 6.2 実装対象

```text
FeatureEncoder
入力 plane 定数
入玉関連特徴
VoVNetV2-style trunk
head adapter
policy_normal_head
policy_style_head
q_normal_result_head
q_nyugyoku_head
outcome_cause_head
moves_left_head
NetworkOutput
checkpoint save / load
```

### 6.3 重要な不変条件

```text
入力は [B, C, 9, 9]
盤面は現在手番視点へ正規化する
network module 内では合法手 mask を行わない
network module 内では policy softmax を行わない
network module 内では q raw に tanh を適用しない
policy / Q は [B, MOVE_LABEL_SIZE]
outcome は [B, 5]
moves_left は [B, 1]
```

### 6.4 通過条件

```text
batch size を変えて forward shape が一致する
CPU / GPU で shape test が通る
mixed precision で NaN が出ない
checkpoint load 後に同一入力の出力が一致する
feature encoder と network の INPUT_CHANNELS が一致する
合法手 mask を外部適用できる
```

---

## 7. M3: target・dataset・loss

### 7.1 目的

自己対局ログ、教師探索、reanalyse 結果から `TrainingSample` を作り、各 head の loss を計算できるようにする。

### 7.2 実装対象

```text
TrainingSample
SparseMoveDistribution
SparseMoveValues
TargetLossWeights
SampleBuilder
outcome_cause target 生成
q_normal_result / q_nyugyoku target 生成
policy_normal / policy_style target 割り当て
moves_left target 生成
loss 計算
dataset metadata
validation split
```

### 7.3 重要な不変条件

```text
すべての target は現在手番視点
policy target は合法手上の分布
Q target は合法手だけに定義する
非選択合法手の Q target 欠損を 0 として扱わない
target None と target 0 を区別する
Q_mode は target の正本にしない
terminal step は標準では TrainingSample にしない
position buffer 再開対局では restart_game_result を target に使う
```

### 7.4 通過条件

```text
toy SelfPlayGame から TrainingSample を生成できる
policy target の確率和が 1 である
違法手 target が生成されない
outcome_cause から q 成分 target が正しく作れる
moves_left = log1p(terminal_ply - step_ply) が作れる
欠損 target の loss が計算されない
NaN / inf を検出できる
```

---

## 8. M4: 教師あり初期化

### 8.1 目的

KLENT-0 自己対局を開始できる最低限の policy、Q、outcome を作る。

### 8.2 教師データ候補

```text
既存Yoltsの探索手
簡易αβ探索の手
詰み探索で確定した手
過去の自己対局棋譜
外部エンジンの固定深さ探索
手動 fixture
```

### 8.3 学習対象

```text
policy_normal_head
policy_style_head
q_normal_result_head
q_nyugyoku_head
outcome_cause_head
moves_left_head
```

ただし、初期段階では `policy_normal_head` の安定を最優先する。

`policy_style_head` と `q_nyugyoku_head` は、通常 target または入玉関連 fixture から弱く初期化する。

### 8.4 通過条件

```text
policy_normal が random policy を明確に上回る
合法手 mask 後の policy entropy が極端に collapse していない
q raw の分布が NaN / inf を含まない
outcome_cause の class order が validation で確認できる
詰み fixture で tactical forced target を作れる
checkpoint から search に接続できる
```

---

## 9. M5: KLENT-0 search

### 9.1 目的

network 出力から合法手上の改善方策を作り、着手選択できるようにする。

### 9.2 実装対象

```text
ConvertedNetworkOutput
softmax_legal
policy_normal / policy_style blend
Q_normal_result / Q_nyugyoku tanh 変換
Q_total_result
Q_mode
outcome probabilities
V_normal_result / V_nyugyoku / V_total_result / V_mode
moves_left expm1 変換
KLENT policy improvement
sample_or_argmax
SearchRequest
SearchResult
```

### 9.3 重要な不変条件

```text
policy logits は softmax 後の確率で blend する
Q_mode = Q_normal_result + λ_nyugyoku * Q_nyugyoku
KLENT は合法手上でだけ計算する
pi_search の確率和は 1
評価対局では deterministic argmax を使える
自己対局では sampling を使える
```

### 9.4 通過条件

```text
合法手が1手ならその手を選ぶ
policy prior が一様で Q が同値なら pi_klent も一様に近い
Q が高い手の確率が上がる
prior が高い手の確率が KL 項で維持される
λ_nyugyoku を変えると Q_mode の順位が変わる
SearchResult に p_prior / pi_klent / pi_search / Q 成分が保存される
```

---

## 10. M6: CPU tactical solver 統合

### 10.1 目的

KLENT-0 が落としやすい短期戦術とルール上の確定手を補強する。

### 10.2 実装対象

```text
TacticalConfig
TacticalResult
宣言勝ち可能判定
1手詰め
有界詰み探索
王手対応
pre-check
post-check
apply_tactical_mask
forced move result
探索ログ
```

### 10.3 優先順位

確定手がある場合の優先順位は以下とする。

```text
1. 宣言勝ち可能手
2. 相手玉への詰みが証明された手
3. Q_mode または pi_search が最大の手
4. deterministic tie-break
```

### 10.4 通過条件

```text
宣言勝ち可能局面で宣言勝ちを選ぶ
詰みが証明された局面で詰み手を選ぶ
王手中に唯一合法応手ならその手を選ぶ
詰み探索 timeout を詰み不成立として扱わない
pre-check と network inference の並列実行設定をログできる
post-check 後の pi_search が合法手上で正規化される
```

---

## 11. M7: 自己対局 v1

### 11.1 目的

`normal_vs_normal` を中心に、KLENT-0 自己対局を安定して生成する。

### 11.2 実装対象

```text
SelfPlayManager
SelfPlayWorker
InferenceServer
GameLogWriter
SelfPlayGame
SelfPlayStep
GameResult
PlayerSpec
PlayerPairing
TemperatureSchedule
RootNoiseSchedule
repetition state
max_game_ply handling
```

### 11.3 初期設定

```text
pairing:
  normal_vs_normal: 100%

search:
  search_type = klent0
  enable_tactical_solver = True
  enable_declaration_win_check = True

start position:
  initial を主軸にする
  shuffled_initial / random_2ply は M8 で本格導入する

resign:
  標準では無効

max_game_ply:
  到達時は max_ply_draw
```

### 11.4 保存する主要ログ

```text
p_prior
pi_klent_pre_tactical
pi_search_post_tactical
pi_select
selected_move
Q_normal_result
Q_nyugyoku
Q_total_result
Q_mode
outcome_probs
moves_left_pred
tactical_flags
terminal_flags_before_move
nyugyoku_features
uncertainty_features
```

### 11.5 通過条件

```text
一定局数をクラッシュなしで生成できる
全 step の selected_move が合法手である
pi_search_post_tactical の確率和が 1
終局理由が outcome category に変換できる
illegal move 相当の状態が通常終局として保存されない
SelfPlayGame から TrainingSample を生成できる
training step が checkpoint まで到達する
```

---

## 12. M8: 開始局面多様化

### 12.1 目的

学習初期の序盤 collapse を避け、position buffer が未成熟な段階でも局面分布を広げる。

### 12.2 実装対象

```text
StartPositionConfig
RandomOpeningConfig
StartPositionShuffleConfig
StartPositionSampler
StartPositionSample
StartPositionInfo
source weight normalization
品質フィルタ
重複排除
source metadata logging
```

### 12.3 source

標準 source は以下である。

```text
initial
random_opening
shuffled_initial
position_buffer
```

`manual` と `test` は通常の自己対局 sampling 対象にしない。

### 12.4 phase 別初期比率

開始局面の phase schedule は、以下を初期候補とする。

```text
Phase 1:
  initial: 50%
  shuffled_initial: 35%
  random_2ply: 10%
  position_buffer: 5%

Phase 2:
  initial: 45%
  shuffled_initial: 20%
  random_2ply: 10%
  position_buffer: 25%

Phase 3:
  initial: 40%
  shuffled_initial: 5%
  random_2ply: 5%
  position_buffer: 50%

Phase 4:
  initial: 50%
  shuffled_initial: 0% 〜 5%
  random_2ply: 0% 〜 5%
  position_buffer: 45% 〜 50%
```

position buffer が存在しない場合、その weight は `initial` と `random_2ply` に再配分する。

### 12.5 通過条件

```text
各 source から StartPositionSample を生成できる
source weight が有効 source 間で正規化される
無効 source が sampling されない
開始局面が合法局面である
開始局面が terminal ではない
開始局面で宣言勝ち条件を満たさない
feature encoder / move generator / tactical solver で処理できる
start_position.source / category / sfen / tags がログに残る
```

---

## 13. M9: position buffer

### 13.1 目的

自己対局で得た有用局面から再開し、入玉関連局面・難所局面・不確実局面を厚く学習する。

### 13.2 実装対象

```text
PositionBufferRecord
PositionBufferWriter
PositionBufferSampler
category score
recency weighting
deduplication
restart game metadata
source_game_id / source_ply
buffer_sample_weight
```

### 13.3 標準 category

```text
normal_midgame
nyugyoku_promising
opponent_nyugyoku_risk
mutual_nyugyoku_like
policy_uncertain
q_close_top_moves
tactical_hard
klent_gumbel_gap
```

### 13.4 重要な不変条件

```text
position buffer 由来の開始局面から再開した対局では restart_game_result を target に使う
source_game_result を再開対局の勝敗 target に使わない
source metadata は分析と sampling に使う
非 terminal かつ feature encodable な局面だけを再開候補にする
```

### 13.5 通過条件

```text
自己対局中に buffer candidate を保存できる
category score をログできる
重複局面を抑制できる
buffer 由来局面から自己対局を再開できる
restart game の TrainingSample に start_position metadata が引き継がれる
入玉関連 category の出現率を集計できる
```

---

## 14. M10: style / 入玉増強

### 14.1 目的

通常棋力を維持しながら、入玉勝ち・相手入玉警戒・相入玉耐性のデータを増やす。

### 14.2 導入する pairing

段階的に以下を混ぜる。

```text
nyugyoku_vs_normal
normal_vs_nyugyoku
bakuretsu_vs_normal
normal_vs_bakuretsu
anti_nyugyoku_vs_nyugyoku
nyugyoku_vs_anti_nyugyoku
mutual_nyugyoku_vs_mutual_nyugyoku
```

### 14.3 phase 別混合比

初期候補は以下である。

```text
Phase 2: style warmup
  normal_vs_normal: 80% 〜 90%
  nyugyoku_vs_normal: 5% 〜 10%
  normal_vs_nyugyoku: 5% 〜 10%

Phase 3: nyugyoku augmentation
  normal_vs_normal: 50% 〜 70%
  nyugyoku_vs_normal: 10% 〜 20%
  normal_vs_nyugyoku: 10% 〜 20%
  bakuretsu_vs_normal: 5% 〜 10%
  normal_vs_bakuretsu: 5% 〜 10%

Phase 4: anti / mutual augmentation
  normal_vs_normal: 40% 〜 60%
  nyugyoku_vs_normal: 10% 〜 15%
  normal_vs_nyugyoku: 10% 〜 15%
  anti_nyugyoku_vs_nyugyoku: 5% 〜 10%
  nyugyoku_vs_anti_nyugyoku: 5% 〜 10%
  mutual_nyugyoku_vs_mutual_nyugyoku: 5% 〜 10%
```

通常棋力が劣化した場合は、`normal_vs_normal` と `initial` / `normal_midgame` を増やす。

入玉サンプルが不足する場合は、mode 比率を上げるより先に、position buffer の入玉関連 category quota を増やす。

### 14.4 target 方針

```text
normal mode:
  policy_normal_target を中心にする

nyugyoku / anti_nyugyoku / mutual_nyugyoku / bakuretsu:
  policy_style_target を中心にする

通常 target を style head に弱く混ぜる:
  style head の初期安定化目的に限定する

style target を policy_normal_head に混ぜる:
  通常棋力を壊さない範囲で実験扱いにする
```

### 14.5 通過条件

```text
mode_id と target_policy_head がログに残る
style 系 target が policy_normal_target と混同されない
policy_style_head の entropy が極端に collapse しない
入玉関連 outcome の sample 数を集計できる
通常評価で明確な退行が出ていない
相手入玉リスク局面で anti_nyugyoku 設定の挙動を確認できる
```

---

## 15. M11: Gumbel MCTS reanalyse

### 15.1 目的

KLENT-0 だけでは不安定な局面に探索ありの改善 target を与える。

### 15.2 対象局面

```text
policy entropy が高い局面
Q の上位手が僅差の局面
入玉関連局面
評価が不安定な終盤局面
通常勝ちと入玉勝ちの選択が割れる局面
自己対局で悪手が疑われる局面
KLENT-0 と Gumbel MCTS の差が大きい局面
```

### 15.3 実装対象

```text
ReanalyseCandidateQueue
GumbelMCTSConfig
Gumbel MCTS root search
leaf outcome value
root-side perspective transform
成分別 backup
ReanalyseRecord
pi_reanalyse
q_normal_result_reanalyse
q_nyugyoku_reanalyse
outcome_cause_reanalyse
reanalyse target quality
```

### 15.4 重要な不変条件

```text
reanalyse target は Q_normal_result と Q_nyugyoku を分けて保存する
Q_mode だけを保存しない
leaf value は root 視点へ変換してから backup する
terminal leaf では network を評価しない
reanalyse の全合法手 target は legal_moves と一致する
```

### 15.5 通過条件

```text
候補局面を抽出できる
reanalyse を small budget で完走できる
pi_reanalyse の確率和が 1
成分別 Q target が保存される
reanalyse dataset から TrainingSample を生成できる
KLENT-0 target と reanalyse target の差分をログできる
```

---

## 16. M12: 実戦時探索・評価強化

### 16.1 目的

学習用自己対局とは別に、実戦時の強さと安定性を詰める。

### 16.2 実戦時探索 tier

```text
KLENT-0 + CPU tactical solver
Gumbel MCTS small budget
Gumbel MCTS medium budget
Gumbel MCTS large budget
```

### 16.3 評価カテゴリ

評価カテゴリは以下に分ける。

```text
標準対局評価
入玉関連モード比較
相手入玉耐性比較
Gumbel MCTS tier 別評価
tactical fixture 評価
診断用固定局面評価
開始局面 source 別の診断評価
探索速度評価
学習安定性評価
```

主要 gate の標準対局評価は、`balanced_positions_2025_24ply_500x2_v1` に固定する。

```text
source = やねうら王互角局面集2025
start_ply = 24
sampled_positions = 500
side_swapped_games_per_position = 2
total_games = 1000
```

通常棋力評価、入玉関連モード比較、相手入玉耐性比較、探索設定 sweep は、同じ1000局評価セットで比較する。

平手初期局面、入玉有望局面、相入玉局面、シャッフル局面からの評価は診断用の別実験として扱い、主要 gate には混ぜない。

### 16.4 対 baseline 評価

主要 baseline は以下とする。

```text
baseline_engine = 爆裂駒拾太郎V4.2.0
baseline_search = 4手読み
```

爆裂玉登太郎側は、まず 0-playout 条件で評価する。

```text
search_type = klent0
playouts = 0
Gumbel MCTS = disabled
CPU tactical solver = enabled
mode = normal / nyugyoku / anti_nyugyoku / mutual_nyugyoku / bakuretsu
```

通常棋力の主要 gate は以下である。

```text
normal mode:
  win_rate_vs_bakuretsu_komahiro_v420_depth4 >= 0.90
```

入玉最適化では、以下の制約を満たす設定だけを候補にする。

```text
playouts = 0
win_rate_vs_bakuretsu_komahiro_v420_depth4 >= 0.80
```

この制約を満たす範囲で、以下を最大化する。

```text
declaration_win_rate_vs_bakuretsu_komahiro_v420_depth4
```

評価では、先後、乱数 seed、開始局面、持ち時間またはノード・時間設定を固定し、checkpoint 間で比較可能にする。

### 16.5 通過条件

```text
固定評価セットを再現可能に実行できる
爆裂駒拾太郎V4.2.0 4手読みとの対局を自動実行できる
playouts=0 / normal mode の勝率を集計できる
playouts=0 の入玉宣言勝ち割合を集計できる
checkpoint 間の通常棋力退行を検出できる
入玉関連局面で outcome / Q / policy の挙動を分析できる
tactical fixture の成功率を集計できる
探索 tier 別の勝率・速度・メモリを比較できる
```

---

## 17. M13: scaling・ablation

### 17.1 目的

安定した基盤の上で、モデル規模、探索設定、開始局面混合、loss weight を調整する。

### 17.2 実験対象

```text
model size scaling
VoVNetV2 block 数
head adapter 幅
input feature ablation
policy blend α_style sweep
λ_nyugyoku sweep
KLENT α / β sweep
tactical solver budget
start position source mix
position buffer category quota
reanalyse budget
moves_left loss weight
outcome class balancing
```

### 17.3 主要最適化軸

M13 では、以下の2つを分けて最適化する。

```text
通常棋力 checkpoint:
  playouts=0 / normal mode で、
  爆裂駒拾太郎V4.2.0 4手読みに対して勝率 90%以上を目指す

入玉最適化 checkpoint / config:
  playouts=0 で勝率 80%以上を維持しながら、
  入玉宣言勝ち割合を最大化する
```

入玉最適化で調整する主な軸は以下である。

```text
α_style_self
α_style_opponent_model
λ_nyugyoku
policy_style_target の混合比
q_nyugyoku loss weight
position buffer の nyugyoku_promising quota
position buffer の opponent_nyugyoku_risk quota
anti_nyugyoku / mutual_nyugyoku pairing 比率
宣言勝ち関連 fixture / sample の oversampling
```

比較では、勝率、入玉宣言勝ち割合、通常勝ち割合、通常負け割合、相手入玉負け割合、引き分け割合を分けて保存する。

### 17.4 通過条件

```text
実験 config と checkpoint が紐づいている
評価結果が config_id / model_version と紐づいている
通常棋力と入玉指標を分けて比較できる
速度・メモリ・局面生成数を同時に比較できる
主要 ablation の結論を docs に反映できる
```

---

## 18. KLENT 学習 phase との対応

`docs/13_klent_training.md` の学習 phase と、この roadmap milestone の対応は以下である。

| KLENT 学習 phase | 内容 | 対応 milestone |
| --- | --- | --- |
| Phase 0 | 教師あり初期化 | M0〜M4 |
| Phase 1 | KLENT-0 normal self-play | M5〜M8 |
| Phase 2 | 入玉関連局面の増強 | M9〜M10 |
| Phase 3 | Gumbel MCTS reanalyse | M11 |
| Phase 4 | 実戦時探索 | M12〜M13 |

学習 phase は一方向に固定ではない。

評価で退行が見つかった場合は、以下のように戻る。

```text
通常棋力が劣化:
  M10 の style 混合比、M9 の position buffer quota、M3 の sample weight を見直す

policy collapse:
  M8 の開始局面多様化、M5 の KLENT entropy、M4 の教師あり初期化を見直す

Q が過信する:
  M3 の q target、M11 の reanalyse target、M13 の value scale を見直す

相手入玉に鈍い:
  M9 の opponent_nyugyoku_risk quota、M10 の anti_nyugyoku pairing を見直す
```

---

## 19. 検証ゲート一覧

### 19.1 表現・ルール gate

```text
SFEN / USI round-trip
legal move generation
move apply / undo
terminal detection
repetition detection
declaration win detection
move label encode / decode
current-side symmetry
```

### 19.2 network gate

```text
feature shape
network output shape
checkpoint compatibility
mixed precision safety
mask outside network
raw output distribution
head gradient norm
```

### 19.3 target gate

```text
outcome class order
terminal_reason mapping
policy probability sum
legal-only target
missing target handling
q component target
moves_left target
restart game target
```

### 19.4 search gate

```text
softmax_legal
policy blend
Q_mode construction
KLENT distribution
sampling / argmax
tactical forced move
post tactical normalization
SearchResult logging
```

### 19.5 self-play gate

```text
1局生成
batch inference
start position metadata
terminal reason consistency
GameResult consistency
TrainingSample generation
position buffer candidate
reanalyse candidate
reproducibility
```

### 19.6 evaluation gate

```text
fixed normal evaluation
fixed nyugyoku evaluation
fixed anti-nyugyoku evaluation
baseline evaluation vs 爆裂駒拾太郎V4.2.0 4手読み
playouts=0 normal mode win_rate >= 90% gate
playouts=0 win_rate >= 80% constrained declaration_win_rate optimization
tactical fixture evaluation
speed benchmark
memory benchmark
checkpoint regression check
```

---

## 20. ログと可観測性

### 20.1 共通ログキー

すべての主要ログには以下を含める。

```text
run_id
config_id
model_version
rule_version
feature_version
move_label_version
network_output_version
target_version
selfplay_log_version
random_seed
```

### 20.2 学習で必須の集計

```text
loss/policy_normal
loss/policy_style
loss/q_normal_result
loss/q_nyugyoku
loss/outcome
loss/moves_left
policy/entropy_normal
policy/entropy_style
q/raw_mean
q/raw_std
outcome/class_distribution
sample/source_distribution
sample/mode_distribution
sample/start_position_source_distribution
```

### 20.3 自己対局で必須の集計

```text
selfplay/game_count
selfplay/step_count
selfplay/terminal_reason_distribution
selfplay/outcome_category_distribution
selfplay/nyugyoku_win_rate
selfplay/repetition_draw_rate
selfplay/max_ply_draw_rate
selfplay/short_game_rate
selfplay/legal_move_count_mean
selfplay/policy_entropy
selfplay/tactical_forced_rate
selfplay/declaration_win_available_rate
selfplay/mate_win_found_rate
```

### 20.4 開始局面で必須の集計

```text
start_position/source_distribution
start_position/category_distribution
start_position/duplicate_rate
start_position/unique_sfen_count
start_position/filter_reject_reason
start_position/fallback_rate
```

### 20.5 reanalyse で必須の集計

```text
reanalyse/candidate_count
reanalyse/processed_count
reanalyse/target_quality
reanalyse/klent_policy_kl
reanalyse/q_normal_delta
reanalyse/q_nyugyoku_delta
reanalyse/budget_elapsed_ms
```

### 20.6 評価で必須の集計

```text
eval/baseline_engine
eval/baseline_search
eval/playouts
eval/mode_id
eval/game_count
eval/win_rate
eval/loss_rate
eval/draw_rate
eval/declaration_win_rate
eval/declaration_win_share_among_wins
eval/normal_win_rate
eval/nyugyoku_loss_rate
eval/terminal_reason_distribution
eval/sente_win_rate
eval/gote_win_rate
eval/avg_game_ply
eval/nps_or_positions_per_second
```

---

## 21. 初期実装の範囲

初期実装で必ず入れるものは以下である。

```text
現在手番視点の feature encoder
27 plane move label
VoVNetV2-style trunk
head adapter
6 head 出力
TrainingSample 生成
KLENT-0 方策改善
CPU tactical solver
normal self-play
開始局面多様化
position buffer
固定評価セット
```

初期実装で無理に最適化しないものは以下である。

```text
大規模 Gumbel MCTS 全面導入
NN cache の本格最適化
Monte-Carlo Graph Search
Transformer block の本格採用
moves_left の探索価値補正
完全自動 hyperparameter tuning
```

これらは、基盤の正しさと通常 self-play の安定性を確認してから導入する。

---

## 22. 失敗時の戻り先

### 22.1 自己対局が短手数で壊れる

優先して確認するものは以下である。

```text
合法手生成
王手対応
詰み探索
KLENT temperature
policy entropy
開始局面品質フィルタ
教師あり初期化の policy 精度
```

### 22.2 通常棋力が劣化する

優先して確認するものは以下である。

```text
style 系 pairing 比率
policy_normal_target への style 混入
shuffled_initial 比率
position buffer category quota
q_nyugyoku の loss weight
λ_nyugyoku 設定
```

### 22.3 対 baseline 勝率が目標に届かない

`playouts=0 / normal mode` が爆裂駒拾太郎V4.2.0 4手読みに対して勝率 90% に届かない場合、優先して確認するものは以下である。

```text
policy_normal_head の教師あり初期化
normal_vs_normal の自己対局比率
q_normal_result の target 品質
CPU tactical solver の取りこぼし
開始局面 source mix が通常分布から離れすぎていないか
style 系 target が policy_normal_head に混入していないか
λ_nyugyoku が normal mode で過剰になっていないか
Gumbel MCTS reanalyse target の不足
```

### 22.4 入玉宣言勝ち割合が伸びない

勝率 80%以上を維持しているにもかかわらず入玉宣言勝ち割合が伸びない場合、優先して確認するものは以下である。

```text
position buffer の nyugyoku_promising quota
宣言勝ち直前局面の fixture / oversampling
policy_style_head の entropy
q_nyugyoku calibration
α_style_self
α_style_opponent_model
λ_nyugyoku
nyugyoku / bakuretsu / mutual_nyugyoku pairing 比率
宣言勝ち fixture / rule test の失敗率
```

### 22.5 入玉サンプルが増えない

優先して確認するものは以下である。

```text
position buffer の nyugyoku_promising quota
opponent_nyugyoku_risk quota
start position source mix
nyugyoku mode pairing
policy_style_head の entropy
q_nyugyoku calibration
```

### 22.6 Q が不安定になる

優先して確認するものは以下である。

```text
選択手のみの MC target への依存
λ-return の bootstrap 視点変換
reanalyse target の不足
q loss weight
value scale
終局原因 mapping
```

### 22.7 tactical solver 依存が強すぎる

優先して確認するものは以下である。

```text
tactical forced rate
tactical target の混合比
詰み探索 budget
policy が詰み手を自然に上げているか
詰み fixture validation
```

---

## 23. ドキュメント更新順序

詳細仕様の追加・修正は、以下の順で整合性確認する。

```text
1. 爆裂玉登太郎_設計概要.md
2. 変更対象の docs/*.md
3. 依存する周辺 docs/*.md
4. docs/12_roadmap.md
5. docs/14_evaluation.md
6. docs/15_training_pipeline.md
```

`docs/12_roadmap.md` は、細かな実装仕様の正本ではなく、順序と通過条件の正本である。

仕様変更により実装順序、依存関係、検証ゲート、評価導入タイミングが変わる場合だけ、このドキュメントを更新する。学習 run、dataset、checkpoint、phase 遷移の実行仕様が変わる場合は `docs/15_training_pipeline.md` も更新する。

---

## 24. 暫定結論

初期実装は、以下の順に進める。

```text
1. version / schema / fixture / test 基盤を固める
2. 盤面表現、合法手生成、終局原因分類、指し手ラベルを実装する
3. feature encoder と VoVNetV2-style network を実装する
4. TrainingSample、loss、dataset pipeline を実装する
5. 教師あり初期化で最低限の policy / Q / outcome を作る
6. KLENT-0 search を実装する
7. CPU tactical solver を統合する
8. normal_vs_normal の自己対局を安定させる
9. 開始局面多様化を導入する
10. position buffer 由来局面からの再開を導入する
11. style / 入玉関連 pairing を段階的に増やす
12. Gumbel MCTS reanalyse を導入する
13. 実戦時探索、評価、ablation、scaling を進める
```

最初の成功条件は、強いAIを一気に作ることではない。

まず、仕様上の不変条件を破らず、`KLENT-0 + CPU tactical solver` の自己対局が安定して回り、そこから学習 sample と checkpoint が再現可能に生成される状態を作る。

最終的な主要目標は、`playouts=0 / normal mode` で爆裂駒拾太郎V4.2.0 4手読みに対して勝率 90%以上を達成することである。

そのうえで、`playouts=0` の勝率 80%以上を維持できる設定群の中から、入玉宣言勝ち割合が最大になる設定を探索する。

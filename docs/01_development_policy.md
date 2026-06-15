# docs/01_development_policy.md - 開発方針

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎の **開発方針** を定義する。

対象は以下である。

```text
プロジェクトの優先順位
仕様書の責務分担
実装時に守る不変条件
検証・評価の基本姿勢
ログ・再現性・異常系の扱い
実験と標準仕様の分離
ドキュメント更新方針
```

このドキュメントは、個別仕様の正本ではない。

個別仕様は以下を正とする。

```text
全体方針: 爆裂玉登太郎_設計概要.md
名称・コンセプト: docs/00_name.md
全体アーキテクチャ: docs/02_overall_architecture.md
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
```

このドキュメントでは、それらの仕様を実装・実験・修正していくときの判断基準を扱う。

---

## 1. 開発目標

### 1.1 通常棋力と入玉棋風を両立する

爆裂玉登太郎の目的は、通常の将棋として十分に強く、なおかつ入玉宣言勝ちを自然に狙える将棋AIを作ることである。

したがって、開発では以下を同時に満たすことを目指す。

```text
通常棋力を大きく損なわない
自玉の入玉を長期計画として扱える
相手の入玉リスクを早めに警戒できる
探索設定により棋風を切り替えられる
```

入玉宣言勝ちを増やすために通常棋力を壊してはならない。

通常棋力だけを高めて、入玉宣言勝ちをほとんど狙えない構成にもしてはならない。

### 1.2 主要目標を分けて管理する

主要目標は以下の2つに分ける。

```text
通常モード目標:
  爆裂玉登太郎 playouts=0 / normal mode が、
  爆裂駒拾太郎V4.2.0 4手読みに対して勝率 90%以上

入玉最適化目標:
  爆裂玉登太郎 playouts=0 の勝率 80%以上を維持する設定の中で、
  入玉宣言勝ち割合を最大化する
```

通常モード目標は、基本棋力を壊していないことを確認するための gate である。

入玉最適化目標は、勝率制約を満たす候補の中で棋風を選ぶための規則である。

勝率が 80% 未満の設定は、入玉宣言勝ち割合が高くても候補外とする。

### 1.3 単一スコアに潰さない

以下の指標は性質が違うため、単一の総合点に潰さない。

```text
通常勝率
入玉宣言勝ち率
入玉勝ち率
相手入玉負け率
通常負け率
引き分け率
詰み逃し率
KLENT-0 と Gumbel MCTS の差
推論速度
探索速度
学習速度
メモリ使用量
```

評価レポート、checkpoint gate、ablation では、少なくとも通常棋力、入玉棋風、相手入玉耐性、短期戦術、速度を分けて見る。

---

## 2. 設計上の優先順位

### 2.1 正しさを速度より優先する

最初に優先するのは、将棋ルール、指し手ラベル、視点変換、target 生成、ログ schema の正しさである。

速度最適化は重要だが、以下の不変条件を壊してまで行わない。

```text
合法手だけを policy / Q target にする
network 出力と target は現在手番視点に揃える
terminal node では network inference を行わない
宣言勝ちは move label に含めない
Q_mode / V_mode を target の正本にしない
異常終了や実装バグを通常の終局理由にしない
```

速度改善は、正しさを保ったうえで、profile と評価で効果を確認して導入する。

### 2.2 仕様の互換性を早めに固定する

以下は checkpoint、自己対局ログ、training dataset、reanalyse dataset の互換性に直結する。

```text
rule_version
feature_version
move_label_version
network_output_version
target_version
search_config_version
selfplay_log_version
eval_suite_version
```

大規模な自己対局・学習を始める前に、少なくとも以下の interface は固定する。

```text
Position / rule result
NetworkOutput
SearchResult
SelfPlayGame
TrainingSample
EvaluationGameResult
```

互換性を壊す変更を行う場合は、既存の dataset や checkpoint を暗黙に再利用しない。

### 2.3 縦に動く最小構成を早く作る

初期実装では、全機能を完全に作るより先に、以下の流れを通す。

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
  -> evaluation smoke test
```

ただし、最小構成のために仕様上の不変条件を緩めない。

縦に通す目的は、早期に interface の破綻を発見し、以降の実験を安全に回せるようにすることである。

### 2.4 通常棋力を壊さない順序で入玉要素を増やす

学習と自己対局は、通常棋力を先に安定させてから入玉関連データを増やす。

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

`policy_normal_head` に style 系 target を大量に混ぜてはならない。

style 系 target は原則として `policy_style_head` に流し、通常棋力を壊さない範囲でのみ弱く混ぜる。

---

## 3. アーキテクチャ方針

### 3.1 役割を分離する

爆裂玉登太郎は、以下の役割を分離して実装する。

```text
Rule / Position Core:
  合法手生成、move 適用、終局判定、宣言勝ち判定、千日手、持将棋、最大手数処理

FeatureEncoder:
  局面を現在手番視点の network input へ変換する

Policy / Action-Value / Outcome Network:
  policy、action-value、outcome、moves_left を出力する

SearchValueConverter:
  network 出力を探索用の policy / Q / V へ変換する

SearchEngine:
  KLENT-0、Gumbel MCTS、CPU tactical solver を合成して着手を選ぶ

SelfPlayLoop:
  対局生成、探索ログ保存、終局結果保存、position buffer 連携を行う

SampleBuilder / DatasetBuilder:
  raw log から TrainingSample を作る

Trainer:
  target と loss weight に従って checkpoint を更新する

EvaluationGateRunner:
  checkpoint と探索設定を評価し、採用可否を判定する
```

ある層の都合で、別の層の正本を増やさない。

### 3.2 network にルールを背負わせない

network は、局面評価・方策・action-value の予測を担当する。

以下は network output ではなく、ルール処理・探索側・ログ集計で扱う。

```text
合法手 mask
宣言勝ち action
投了
千日手成立
持将棋引き分け
最大手数到達
illegal move marker
現局面から厳密に計算できる宣言点数
現局面から厳密に計算できる敵陣内自駒数
```

network module 内で、合法手 mask、softmax、`tanh`、KLENT 方策計算を行わせない。

これらは探索側、変換層、または loss 側の責務とする。

### 3.3 CPU tactical solver を主探索にしない

標準探索は以下とする。

```text
KLENT-0 + CPU tactical solver
```

CPU tactical solver は、以下の確定的・短期的な処理を補強する。

```text
宣言勝ち可能判定
相手玉への詰み探索
王手対応
terminal / rule result の補助
```

通常中盤の長期評価、入玉計画、形勢判断は、network、KLENT 方策改善、Gumbel MCTS reanalyse に任せる。

CPU tactical solver を深い αβ探索の代替として肥大化させない。

### 3.4 Gumbel MCTS は補助的に使う

Gumbel MCTS は、初期の主自己対局ループで全局面に使わない。

用途は以下に限定する。

```text
reanalyse
難所局面解析
入玉関連局面の深掘り
KLENT-0 との差分測定
実戦時 test-time search
```

自己対局局面数と更新回数を増やす主軸は KLENT-0 とする。

---

## 4. データと target の方針

### 4.1 target の正本を増やさない

学習 target の正本は以下に限定する。

```text
policy_normal_target
policy_style_target
q_normal_result_target
q_nyugyoku_target
outcome_cause_target
moves_left_target
```

以下は派生値であり、target の正本にしない。

```text
Q_total_result
Q_mode
V_total_result
V_mode
```

特に、`Q_mode` だけを保存して別の `λ_nyugyoku` で再利用してはならない。

### 4.2 現在手番視点を標準にする

入力特徴量、指し手ラベル、policy、Q、outcome、target は、原則として現在手番視点で扱う。

```text
policy:
  現在手番が指す合法手上の分布

Q_normal_result:
  現在手番側の通常勝ちを正、通常負けを負

Q_nyugyoku:
  現在手番側の入玉勝ちを正、相手入玉による負けを負

outcome_cause:
  現在手番側から見た win_normal / win_nyugyoku / draw / loss_nyugyoku / loss_normal
```

先手視点、後手視点、root 視点、親 node 視点を暗黙に混ぜてはならない。

Gumbel MCTS や reanalyse で root 視点が必要な場合は、変換箇所を明示する。

### 4.3 欠損 target をゼロ target と混同しない

すべてのサンプルがすべての head の target を持つとは限らない。

欠損 target は `None` として明示する。

`None` と `0` は意味が違う。

```text
None:
  この head の target は存在しない

0:
  target として値 0 が存在する
```

loss 計算では、欠損 target を loss から除外する。

### 4.4 違法手には target を持たせない

policy target と action-value target は、合法手だけに定義する。

以下を必ず守る。

```text
moves は合法手だけを含む
moves は現在手番視点の move label である
policy probs の合計は 1 である
違法手確率は保存しない
loss 計算時も合法手 mask を適用する
```

違法手を学習で抑制するために、違法手へ明示的な負 target を入れない。

違法手の除外は、合法手生成、合法手 mask、loss mask によって保証する。

---

## 5. ログ・再現性・異常系

### 5.1 raw log を重視する

評価値や rate だけを保存してはならない。

可能な限り raw count と raw distribution を保存し、後から再集計できるようにする。

評価では少なくとも以下を raw count として保存する。

```text
games
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

自己対局、reanalyse、training sample、evaluation でも、source、version、config、seed、model version を保存する。

### 5.2 seed と config を保存する

再現性のため、実験 run では少なくとも以下を保存する。

```text
run_id
parent_run_id
random_seed
model_version
checkpoint_id
config_id
rule_version
feature_version
move_label_version
network_output_version
target_version
search_config_id
selfplay_config_id
trainer_config_id
eval_config_id
code_version
```

設定値をログの文章だけに埋め込まない。

機械的に比較・再実行できる形式で保存する。

### 5.3 異常系を通常結果に変換しない

以下は通常の終局理由ではなく、実装または実行の異常である。

```text
合法手生成異常
move 適用異常
feature encode error
move label encode / decode error
network NaN / inf
SearchResult schema error
SelfPlayGame schema error
TrainingSample schema error
engine crash
USI protocol error
desync
search exception
external process timeout
```

これらを `loss_normal`、`draw`、`max_ply_draw` などへ変換してはならない。

該当 game / sample / evaluation は `failed`、`invalid`、または `quarantine` として隔離する。

原因を修正するまで training dataset には入れない。

### 5.4 失敗を隠さない

失敗対局、invalid sample、NaN、schema mismatch、version mismatch は、学習や評価から除外するだけでなく、件数と代表例をレポートする。

特に評価 run に失敗対局が含まれる場合は、勝率だけを表示して正常な run のように扱わない。

---

## 6. 評価方針

### 6.1 評価は早く作る

初期段階から以下を用意する。

```text
unit test
fixture test
schema validation
small self-play smoke test
small training smoke test
small evaluation smoke test
```

正式な棋力判断の前でも、実装破綻を早期に検出する評価を入れる。

### 6.2 formal evaluation は固定条件で比較する

主要な対局評価は、評価仕様で定義された固定開始局面セットを使う。

標準評価では、評価目的ごとに開始局面を変えない。

通常棋力評価、入玉関連評価、相手入玉耐性評価、探索設定 sweep は、同じ開始局面セットで比較する。

開始局面分布を変える実験は診断用の別実験として扱い、主要 gate とは分ける。

### 6.3 評価対局は deterministic を標準にする

評価では再現性を優先する。

標準は以下である。

```text
move_selection = deterministic argmax
root noise = disabled
temperature = 0 または deterministic 扱い
Gumbel sampling = disabled unless evaluating Gumbel MCTS itself
```

自己対局では局面多様性のために sampling を使ってよいが、評価では checkpoint 間の比較を安定させる。

### 6.4 checkpoint は loss だけで採用しない

checkpoint の採用は、少なくとも以下を分けて判断する。

```text
correctness gate:
  schema、label、terminal、tactical fixture が壊れていないか

stability gate:
  自己対局と評価対局が失敗なく回るか

normal strength gate:
  通常モードの主要勝率を満たすか

nyugyoku optimization gate:
  勝率制約下で入玉宣言勝ち率を改善するか

speed gate:
  推論・探索・学習速度が実用範囲か
```

loss が下がっても、評価 gate を壊す checkpoint は採用しない。

---

## 7. 実験方針

### 7.1 標準仕様と実験を分ける

標準仕様に入れるものと、実験候補は分けて管理する。

実験候補は、以下を明示する。

```text
目的
仮説
変更点
影響する component
比較対象
評価指標
採用条件
戻し方
```

効果未確認の実験を、標準仕様として既成事実化しない。

### 7.2 ablation できる形で実装する

重要な機能は、可能な限り ablation できるようにする。

例:

```text
policy_style_head の混合率
λ_nyugyoku
CPU tactical solver の各 check
Gumbel MCTS reanalyse の有無
開始局面 source の混合比
position buffer sampling rule
network optional block
loss weight
```

ただし、ルール正しさや schema correctness に関わる機能は ablation 対象にしない。

### 7.3 変更は評価可能な単位に分ける

複数の大きな変更を同時に入れると、効果の切り分けが難しくなる。

実験では、できるだけ以下を守る。

```text
1つの仮説につき1つの主要変更にする
baseline を固定する
評価開始局面と seed を固定する
raw count を保存する
速度・失敗率も見る
```

---

## 8. 実装方針

### 8.1 仕様書を実装指示として成立させる

各 `docs/*.md` は、単なる説明ではなく、実装時に参照できる仕様書として書く。

ただし、ドキュメント内に「コーディングエージェントへの実装指示」のような独立項目を置く必要はない。

仕様、interface、不変条件、検証項目が十分に明確であれば、それ自体が実装指示として機能する。

### 8.2 実装詳細を仕様に押し込みすぎない

仕様書では、互換性、責務、入出力、保存形式、不変条件、検証条件を定義する。

一方で、以下のような局所的な実装都合は、必要以上に固定しない。

```text
内部 helper 関数名
一時変数名
ログ出力ライブラリ
並列化の細かい実装
ファイル分割の微細な単位
```

ただし、checkpoint や dataset の互換性に関わる名前・順序・shape・version は明確に固定する。

### 8.3 単体テストしやすい境界にする

以下は個別にテストできるように分離する。

```text
SFEN / USI parse
合法手生成
move apply / undo
terminal 判定
宣言勝ち判定
現在手番視点の座標変換
move label encode / decode
feature encode
network output shape
legal mask
SearchValueConverter
KLENT-0 方策計算
tactical solver forced move
TrainingSample 生成
evaluation aggregation
```

統合テストだけで正しさを保証しない。

### 8.4 型と schema を明示する

主要な境界では、dict の暗黙受け渡しではなく、dataclass、型付き構造体、schema validation を使う。

対象は以下である。

```text
NetworkOutput
SearchModeConfig
SearchResult
TacticalResult
SelfPlayStep
SelfPlayGame
StartPositionRecord
TrainingSample
TargetLossWeights
EvaluationGameResult
CheckpointMetadata
RunConfig
```

schema 破損は通常の失敗対局ではなく、実装異常として扱う。

---

## 9. ドキュメント方針

### 9.1 概要と詳細を分ける

`爆裂玉登太郎_設計概要.md` には、長期的に変わりにくい設計判断だけを書く。

詳細な shape、loss、target、視点変換、ログ項目、unit test、実装順序は各 `docs/*.md` に分ける。

同じ内容を複数ドキュメントに重複定義しない。

重複が必要な場合は、片方を正本として明示する。

### 9.2 改訂履歴より現在の理想形を優先する

現段階ではドラフトとして、ドキュメントは理想の最新状態へ直接書き換える。

改訂メモ、過去案、削除した実験案、採用しない理由の長い説明は、標準仕様には原則として残さない。

採用しない内容を記載する場合は、将来誤実装されやすいものに限定し、簡潔に書く。

### 9.3 不整合を見つけたら正本を更新する

詳細資料の作り込みによって概要や他の仕様と矛盾が見つかった場合は、以下の順で直す。

```text
1. どのドキュメントが正本か確認する
2. 正本の内容を最新の合意に合わせる
3. 参照側の重複記述を削るか、正本参照へ置き換える
4. version / schema / interface への影響を確認する
5. 必要なら roadmap / evaluation / training pipeline も更新する
```

### 9.4 ファイル名と参照名を安定させる

仕様書間の参照は、原則として `docs/xx_name.md` のパスで行う。

ファイル名を変更する場合は、関連ドキュメントの参照も更新する。

同じ役割の資料を複数名で並存させない。

---

## 10. 守るべき不変条件

以下は、初期実装から守る。

```text
合法手 mask は探索側・loss 側で適用する
違法手には policy / Q target を持たせない
指し手ラベルは現在手番視点で符号化する
network 出力は現在手番視点に揃える
policy と action-value は同じ MOVE_LABEL_SIZE を使う
宣言勝ちは move label に含めない
terminal node では network inference を行わない
Q_mode / V_mode は派生値であり、target の正本にしない
normal_result 成分と nyugyoku 成分を分けて保存する
欠損 target をゼロ target として扱わない
合法手生成や move 適用の異常は通常の終局理由にしない
失敗対局や invalid sample を training dataset に混ぜない
評価失敗と対局結果を混ぜない
checkpoint は loss だけで採用しない
```

これらを破る変更は、単なる実装変更ではなく仕様変更として扱う。

---

## 11. 初期開発で優先するもの

初期開発では、以下を優先する。

```text
1. ルール・指し手ラベル・視点変換の正しさ
2. NetworkOutput / SearchResult / TrainingSample の schema 固定
3. KLENT-0 + CPU tactical solver の最小実行
4. 1局自己対局から training step までの縦通し
5. 小規模評価 smoke test
6. formal evaluation の実行基盤
7. 通常棋力を壊さない self-play / training loop
8. style 系 mode と入玉関連局面の段階導入
9. Gumbel MCTS reanalyse の導入
10. 速度最適化と大規模化
```

最初から全機能を高性能にするより、互換性のある正しいデータ循環を作ることを優先する。

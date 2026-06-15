# docs/14_evaluation.md - 評価仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎の **評価仕様** を定義する。

対象は以下である。

```text
評価の基本方針
主要 baseline
通常棋力評価
入玉棋風評価
相手入玉耐性評価
固定局面評価
tactical solver 評価
KLENT-0 と Gumbel MCTS の差分評価
offline validation
速度・資源評価
評価ログ形式
集計指標
checkpoint 昇格条件
入玉最適化の選択規則
実装上の不変条件
検証項目
実験項目
```

爆裂玉登太郎は、通常の将棋として十分に強く、なおかつ入玉宣言勝ちを自然に狙えることを目的とする。

そのため、評価では以下を分けて扱う。

```text
通常棋力:
  普通に勝てるか

入玉棋風:
  勝率を維持しながら入玉宣言勝ちを増やせるか

相手入玉耐性:
  相手の入玉による負けを抑えられるか

短期戦術:
  詰み、王手対応を取りこぼさないか

探索・学習安定性:
  KLENT-0、Gumbel MCTS、network output、tactical solver が整合しているか
```

関連仕様は以下を正とする。

```text
全体方針: 爆裂玉登太郎_設計概要.md
探索・着手選択: docs/03_search.md
ネットワーク出力: docs/05_outputs.md
探索価値・視点変換: docs/06_search_value.md
CPU tactical solver: docs/07_tactical_solver.md
自己対局ループ: docs/08_selfplay.md
開始局面多様化: docs/09_start_position_diversity.md
学習ターゲット: docs/10_training_targets.md
指し手ラベル: docs/11_move_label.md
実装・実験ロードマップ: docs/12_roadmap.md
KLENT-first 学習・探索方針: docs/13_klent_training.md
学習パイプライン: docs/15_training_pipeline.md
```

---

## 1. 基本方針

### 1.1 単一の総合点に潰さない

評価指標は、単一の総合点に潰さない。

理由は以下である。

```text
通常勝率が高くても、入玉宣言勝ちをまったく狙えない可能性がある
入玉宣言勝ち率が高くても、通常棋力を壊している可能性がある
相手入玉を防げないと、入玉棋風としては不安定である
短期戦術の取りこぼしは、network の長期方針とは別に検出すべきである
探索速度やメモリ使用量は、強さとは別軸で管理すべきである
```

評価レポートでは、少なくとも以下を別々に表示する。

```text
通常勝率
入玉宣言勝ち率
入玉勝ち率
相手入玉負け率
通常負け率
引き分け率
詰み逃し率
KLENT-0 と Gumbel MCTS の差
推論・探索速度
```

### 1.2 評価対局は deterministic を標準にする

評価対局では、原則として stochastic sampling を使わない。

標準は以下である。

```text
move_selection = deterministic argmax
root noise = disabled
temperature = 0 または deterministic 扱い
Gumbel sampling = disabled unless evaluating Gumbel MCTS itself
```

自己対局では局面多様性のために sampling を使うが、評価では checkpoint 間の比較再現性を優先する。

### 1.3 評価開始局面は BalancedPositions2025 24手目に固定する

対局評価の開始局面は、今までの爆裂駒拾太郎の評価と同じく、**やねうら王互角局面集2025 の24手目局面**を標準とする。

```text
source = やねうら王互角局面集2025
source_url = https://github.com/yaneurao/YaneuraOu/releases/tag/BalancedPositions2025
start_ply = 24
sampled_positions = 500
side_swapped_games_per_position = 2
total_games = 1000
```

標準評価では、評価 run ごとに事前固定された seed で500局面をランダムサンプリングする。各局面について candidate 先手・baseline 後手の1局と、先後を入れ替えた1局を行い、合計1000局で集計する。

通常棋力評価、入玉関連評価、相手入玉耐性評価、探索設定 sweep は、すべてこの同一開始局面セットを使う。評価目的によって開始局面を変えると、勝率差・宣言勝ち率差・相手入玉負け率差が、棋力や探索設定の差なのか開始局面分布の差なのか切り分けにくくなるためである。

開始局面多様化、position buffer、将棋81万風シャッフル局面は学習用自己対局の機構であり、標準の対局評価 suite には混ぜない。シャッフル開始局面や入玉有望局面からの再開評価を行う場合は、診断用の別実験として扱い、主要 gate とは分ける。

### 1.4 対局評価と局面評価を分ける

評価は以下の2系統で行う。

```text
game evaluation:
  実際に対局を最後まで行い、勝敗・終局理由・速度を測る

position evaluation:
  固定局面で policy / Q / outcome / tactical solver の挙動を測る
```

通常棋力の最終判断は game evaluation を正とする。

一方で、学習途中の回帰検出、詰み逃し、Q calibration は position evaluation で早く検出する。

### 1.5 raw count を必ず保存する

勝率や宣言勝ち率だけを保存してはならない。

必ず以下の raw count を保存する。

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

rate は raw count から再計算できる派生値である。

### 1.6 評価失敗と対局結果を混ぜない

以下は通常の終局理由ではなく、評価実行エラーである。

```text
illegal move
desync
engine crash
USI protocol error
feature encode error
move label decode error
search exception
timeout of external process
```

これらを勝ち・負け・引き分けとして集計してはならない。

評価 run に失敗対局が含まれる場合は、`failed_games` として別集計し、run 全体を `valid = False` にしてよい。

---

## 2. 主要目標

### 2.1 主要 baseline

主要な対外 baseline は以下に固定する。

```text
baseline_engine = 爆裂駒拾太郎V4.2.0
baseline_search = 4手読み
baseline_id = bakuretsu_komahiro_v420_depth4
```

baseline engine の binary、commit、設定、探索深さ、定跡有無、スレッド数、評価関数 version は、評価ログに必ず保存する。

### 2.2 爆裂玉登太郎側の標準評価条件

爆裂玉登太郎側の標準評価条件は以下である。

```text
engine = 爆裂玉登太郎
search_type = klent0
playouts = 0
mode_id = normal
enable_tactical_solver = True
enable_declaration_win_check = True
Gumbel MCTS = disabled
root_noise = disabled
move_selection = deterministic argmax
```

ここでの `playouts = 0` は、MCTS / Gumbel MCTS の playout を行わないことを表す。

CPU tactical solver による以下は、標準の 0-playout 探索に含める。

```text
宣言勝ち可能判定
相手玉への詰み探索
王手対応
```

### 2.3 通常モード目標

通常モードの主要目標は以下である。

```text
爆裂玉登太郎 playouts=0 / normal mode が、
爆裂駒拾太郎V4.2.0 4手読みに対して勝率 90%以上
```

この目標は、通常棋力を壊していないことを確認する hard gate とする。正式判定は `balanced_positions_2025_24ply_500x2_v1` の1000局評価で行う。

主要指標名は以下に固定する。

```text
win_rate_vs_bakuretsu_komahiro_v420_depth4
```

定義は以下である。

```text
win_rate_vs_bakuretsu_komahiro_v420_depth4 = wins / valid_games
```

`valid_games` は、評価失敗を除いた対局数である。

### 2.4 入玉最適化目標

入玉最適化目標は以下である。

```text
爆裂玉登太郎 playouts=0 の勝率 80%以上を維持する設定の中で、
入玉宣言勝ち割合を最大化する
```

勝率が 80% 未満の設定は、入玉宣言勝ち割合が高くても候補外とする。正式判定は `balanced_positions_2025_24ply_500x2_v1` の1000局評価で行う。

主要指標名は以下に固定する。

```text
declaration_win_rate_vs_bakuretsu_komahiro_v420_depth4
```

定義は以下である。

```text
declaration_win_rate_vs_bakuretsu_komahiro_v420_depth4
  = declaration_win_games / valid_games
```

補助指標として以下も保存する。

```text
declaration_win_share_among_wins
  = declaration_win_games / wins
```

`declaration_win_share_among_wins` は、勝っている対局のうちどれくらいが入玉宣言勝ちだったかを見るための補助指標である。

勝率制約の代替として使ってはならない。

### 2.5 通常棋力 checkpoint と入玉最適化 config を分ける

以下は別々に管理する。

```text
normal_champion_checkpoint:
  通常モード 90% gate を満たす checkpoint

nyugyoku_optimized_config:
  勝率 80%以上を維持しながら入玉宣言勝ち率を最大化する探索設定
```

入玉最適化で得られた設定が通常モード 90% gate を満たす必要はない。

ただし、通常モード 90% gate を満たす checkpoint は常に別途維持する。

---

## 3. 評価モード

### 3.1 EvalMode

評価モードは以下の構造体相当で管理する。

```python
@dataclass
class EvalMode:
    mode_id: str
    description: str

    search_type: str              # "klent0" | "gumbel_mcts"
    num_simulations: int

    alpha_style_self: float
    alpha_style_opponent_model: float
    lambda_nyugyoku: float

    enable_tactical_solver: bool
    enable_declaration_win_check: bool
    enable_mate_search: bool

    deterministic: bool = True
    temperature: float = 0.0
    root_noise_frac: float = 0.0
```

`mode_id` は探索設定と評価結果を対応づけるために必ず保存する。

### 3.2 標準評価モード

標準評価モードは以下である。

```text
normal_klent0:
  通常棋力評価用

nyugyoku_klent0:
  自玉の入玉宣言勝ちを狙う評価用

anti_nyugyoku_klent0:
  相手入玉耐性評価用

mutual_nyugyoku_klent0:
  相入玉想定評価用

bakuretsu_klent0:
  通常棋力を残しつつ入玉価値をやや強く見る評価用

gumbel_small:
  少数 simulation の実戦時探索評価用

gumbel_medium:
  中規模 simulation の実戦時探索評価用
```

`normal_klent0` は通常モード 90% gate の正本である。

`nyugyoku_klent0`、`anti_nyugyoku_klent0`、`mutual_nyugyoku_klent0`、`bakuretsu_klent0` は入玉関連評価に使う。

Gumbel MCTS 系は、KLENT-0 の代替ではなく、実戦時探索・難所解析・差分測定として扱う。

### 3.3 評価では target_policy_head を使わない

`target_policy_head` は自己対局ログから学習 target を作るための情報であり、評価では使わない。

評価では、実際にどの探索モードで指したかを `mode_id` と `EvalMode` に保存する。

---

## 4. 評価セット

### 4.1 EvalSuite

評価セットは以下の構造体相当で管理する。

```python
@dataclass
class EvalSuite:
    suite_id: str
    suite_type: str  # "game" | "position" | "fixture"
    version: str
    description: str
    positions: list["EvalPosition"]
    tags: list[str]
```

`version` は固定し、局面を追加・削除・変更する場合は version を上げる。

### 4.2 EvalPosition

固定局面は以下の構造体相当で管理する。

```python
@dataclass
class EvalPosition:
    position_id: str
    sfen: str
    side_to_move: int
    source: str
    category: str
    expected: dict
    tags: list[str]
```

`expected` は fixture 評価にだけ必須である。

通常の対局評価局面では、期待手や期待結果を持たなくてよい。

### 4.3 標準評価セット一覧

標準評価セットは以下とする。

| suite_id | 用途 | 主要 gate に含めるか |
| --- | --- | --- |
| `balanced_positions_2025_24ply_500x2_v1` | やねうら王互角局面集2025の24手目から500局面をサンプリングし、先後入れ替えで1000局評価する標準対局評価 | yes |
| `mate_tactical_v1` | 詰み・王手 fixture | no |
| `check_response_v1` | 王手対応 fixture | no |
| `policy_uncertain_v1` | policy entropy が高い難所局面の診断 | no |
| `q_close_top_moves_v1` | 上位 Q が僅差の難所局面の診断 | no |

通常棋力、入玉関連評価、探索設定 sweep の主評価は `balanced_positions_2025_24ply_500x2_v1` を正とする。

fixture / position suite は、分析・回帰検出・ablation に使う。主要 gate の対局結果には混ぜない。

### 4.4 標準対局評価セットの作成

`balanced_positions_2025_24ply_500x2_v1` は以下で作る。

```text
1. やねうら王互角局面集2025 の24手目局面を読み込む
2. 固定 seed で500局面をランダムサンプリングする
3. 各局面について candidate 先手の対局を作る
4. 同じ局面を先後入れ替えして candidate 後手の対局を作る
5. 合計1000局を1つの正式評価 run として扱う
```

保存する metadata は以下である。

```text
balanced_positions_version
source_url
sample_seed
sample_index
position_id
original_sfen
swapped_sfen
start_ply = 24
pair_id
candidate_side
```

### 4.5 固定局面診断 suite の作成元

診断用の固定局面セットは以下から作る。

```text
自己対局 position buffer
Gumbel MCTS reanalyse 候補
詰み fixture
王手対応 fixture
手動作成局面
過去 checkpoint の失敗局面
```

局面を採用する場合は、以下を保存する。

```text
source_game_id
source_ply
source_model_version
source_mode_id
source_start_position
category_scores
reason_for_inclusion
```

### 4.6 固定局面セットのリーク扱い

評価用固定局面を学習に混ぜてはならない。

ただし、同じカテゴリの別局面を学習に使うことは許可する。

評価局面を reanalyse して診断に使うことはよいが、その reanalyse target を同じ checkpoint 系列の学習に戻す場合は、評価 suite version を分ける。

## 5. 対局評価

### 5.1 EvalMatchConfig

対局評価は以下の構造体相当で設定する。

```python
@dataclass
class EvalMatchConfig:
    eval_id: str
    suite_id: str
    num_games: int
    random_seed: int

    candidate: "PlayerEvalSpec"
    opponent: "PlayerEvalSpec"

    side_assignment: str  # "alternate" | "paired" | "fixed"
    start_position_policy: str  # "initial" | "fixed_suite" | "paired_suite"

    max_game_ply: int = 512
    repetition_rule: str = "standard"
    declaration_rule_id: str = "csa27"

    save_kifu: bool = True
    save_step_summary: bool = True
    save_full_search_log: bool = False
```

`eval_id` は評価 run 全体の識別子である。

### 5.2 PlayerEvalSpec

評価対象プレイヤーは以下で表す。

```python
@dataclass
class PlayerEvalSpec:
    player_id: str
    engine_id: str
    engine_version: str
    checkpoint_id: str | None
    command: str | None
    engine_options: dict
    eval_mode: EvalMode | None
    threads: int
    device: str | None
```

外部 engine の場合、`eval_mode` は `None` でよい。

ただし、探索深さ、持ち時間、スレッド数、hash、定跡設定など、結果に影響する option は必ず `engine_options` に保存する。

### 5.3 先後割り当て

標準対局評価では、同一開始局面を candidate 先手・candidate 後手の両方で評価する。

```text
side_assignment = "paired"
start_position_policy = "paired_suite"
```

`balanced_positions_2025_24ply_500x2_v1` では、500局面それぞれに対して先後入れ替えの2局を作る。これにより、開始局面の偏りと先後差を抑えて比較する。

### 5.4 対局数 tier

評価対局数は目的別に tier を分ける。

```text
smoke:
  16〜32局
  実装破壊、USI接続、明らかな退行を検出する

quick:
  64〜128局
  学習中の頻繁な checkpoint 比較に使う

standard:
  256〜512局
  checkpoint 昇格候補の比較に使う

final:
  1000局
  主要目標達成の正式確認に使う
```

評価 run には `eval_tier` を保存する。

90% gate や 80% constrained optimization の最終判断は、`final` の1000局評価を正とする。`final` は `balanced_positions_2025_24ply_500x2_v1` を使い、500局面 × 先後入れ替えで固定する。

### 5.5 対局打ち切り

標準の最大手数は自己対局仕様と同じく以下とする。

```text
max_game_ply = 512
max_ply_result = max_ply_draw
```

評価でも、最大手数到達時に点数・入玉条件で勝敗を付けない。

勝敗を付けたい場合は、別の実験設定として `evaluation_adjudication` を明示し、標準評価とは混ぜない。

### 5.6 投了

初期実装の標準評価では、爆裂玉登太郎側の投了は無効でよい。

外部 baseline が投了を返す場合は、`terminal_reason = resign` とし、通常勝敗に分類する。

投了しきい値を導入する場合は、評価設定に必ず保存し、投了なし評価と混ぜない。

---

## 6. 終局理由と outcome 集計

### 6.1 標準 terminal_reason

評価で扱う標準の `terminal_reason` は以下である。

```text
checkmate
resign
declaration_win
nyugyoku_adjudication
normal_adjudication
repetition_draw
jishogi_draw
max_ply_draw
```

以下は標準の `terminal_reason` に含めない。

```text
illegal_move
timeout
engine_crash
protocol_error
```

これらは評価実行エラーとして扱う。

### 6.2 outcome category

`terminal_reason` は以下の `outcome_category` へ変換する。

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

この表にない終局理由を評価指標へ混ぜてはならない。

### 6.3 勝敗・原因別 count

candidate から見た count は以下を保存する。

```python
@dataclass
class EvalOutcomeCounts:
    valid_games: int
    failed_games: int

    wins: int
    losses: int
    draws: int

    win_normal_games: int
    win_declaration_games: int
    win_nyugyoku_adjudication_games: int

    loss_normal_games: int
    loss_declaration_games: int
    loss_nyugyoku_adjudication_games: int

    repetition_draw_games: int
    jishogi_draw_games: int
    max_ply_draw_games: int
```

`win_declaration_games` は、candidate が宣言勝ちした対局数である。

`loss_declaration_games` は、相手が宣言勝ちした対局数である。

### 6.4 派生 rate

主な派生 rate は以下である。

```text
win_rate = wins / valid_games
loss_rate = losses / valid_games
draw_rate = draws / valid_games

normal_win_rate = win_normal_games / valid_games
declaration_win_rate = win_declaration_games / valid_games
nyugyoku_adjudication_win_rate = win_nyugyoku_adjudication_games / valid_games

normal_loss_rate = loss_normal_games / valid_games
opponent_declaration_loss_rate = loss_declaration_games / valid_games
opponent_nyugyoku_adjudication_loss_rate = loss_nyugyoku_adjudication_games / valid_games
opponent_nyugyoku_loss_rate = (loss_declaration_games + loss_nyugyoku_adjudication_games) / valid_games

declaration_win_share_among_wins = win_declaration_games / wins
nyugyoku_win_share_among_wins = (win_declaration_games + win_nyugyoku_adjudication_games) / wins
```

`wins = 0` の場合、`declaration_win_share_among_wins` は `None` とする。

---

## 7. 入玉関連評価

### 7.1 開始局面は標準評価セットに統一する

入玉関連評価も、通常棋力評価と同じく `balanced_positions_2025_24ply_500x2_v1` を使う。

```text
source = やねうら王互角局面集2025
start_ply = 24
sampled_positions = 500
side_swapped_games_per_position = 2
total_games = 1000
```

入玉有望局面、相手入玉リスク局面、相入玉風局面などの専用開始局面 suite は、標準の入玉評価には使わない。入玉関連評価は、同一の互角局面集合に対して探索モードや設定を変えたときに、勝率と宣言勝ち率がどう変わるかを見る。

### 7.2 入玉関連の主指標

`balanced_positions_2025_24ply_500x2_v1` 上で、candidate から見た以下を集計する。

```text
win_rate_vs_bakuretsu_komahiro_v420_depth4
declaration_win_rate_vs_bakuretsu_komahiro_v420_depth4
nyugyoku_win_rate_vs_bakuretsu_komahiro_v420_depth4
opponent_declaration_loss_rate_vs_bakuretsu_komahiro_v420_depth4
opponent_nyugyoku_loss_rate_vs_bakuretsu_komahiro_v420_depth4
draw_rate_vs_bakuretsu_komahiro_v420_depth4
```

定義は以下である。

```text
nyugyoku_win_rate
  = (win_declaration_games + win_nyugyoku_adjudication_games) / valid_games

opponent_nyugyoku_loss_rate
  = (loss_declaration_games + loss_nyugyoku_adjudication_games) / valid_games
```

### 7.3 探索モード比較

入玉関連評価では、同一の1000局評価セットで以下の評価モードを比較する。

```text
normal_klent0
nyugyoku_klent0
anti_nyugyoku_klent0
mutual_nyugyoku_klent0
bakuretsu_klent0
```

`normal_klent0` は通常棋力 gate の正本である。

`nyugyoku_klent0`、`mutual_nyugyoku_klent0`、`bakuretsu_klent0` は、勝率を維持しながら宣言勝ち率を増やせるかを見る。

`anti_nyugyoku_klent0` は、相手の宣言勝ち・入玉勝ちによる負けが増えないかを見る。

### 7.4 無理入玉傾向

入玉寄りの探索設定が、勝率を落としてまで入玉を狙っていないかを確認する。

主な診断指標は以下である。

```text
win_rate_delta_from_normal
declaration_win_rate_delta_from_normal
opponent_nyugyoku_loss_rate_delta_from_normal
draw_rate_delta_from_normal
```

入玉最適化では、勝率 80%以上の制約を満たす設定だけを候補にする。したがって、宣言勝ち率が上がっても `win_rate_delta_from_normal` が大きく悪化する設定は採用しない。

## 8. tactical solver 評価

### 8.1 tactical fixture

`tactical fixture` は、局面単位で tactical solver の正しさを検証する。

標準カテゴリは以下である。

```text
mate_in_one
mate_win_by_dfpn
only_legal_response
in_check_multiple_responses
no_forced_move
```

fixture には期待値を持たせる。

```python
@dataclass
class TacticalExpected:
    has_mate_win: bool | None
    mate_win_moves: set[int] | None
    forced_move: int | None
    forced_reason: str | None
```

### 8.2 tactical 指標

主な指標は以下である。

```text
mate_in_one_detection_rate
mate_win_detection_rate
only_legal_response_detection_rate
forced_move_accuracy
tactical_false_positive_rate
tactical_timeout_rate
tactical_elapsed_ms_p50
tactical_elapsed_ms_p95
searched_nodes_mean
```

`timeout` は詰みなしとして扱わない。

詰み探索 timeout は `unknown` として集計する。

## 9. KLENT-0 / Gumbel MCTS 差分評価

### 9.1 目的

Gumbel MCTS は、全局面の主学習ループではなく、再解析、難所局面、実戦時探索に使う。

評価では、KLENT-0 と Gumbel MCTS の差を測ることで、以下を検出する。

```text
KLENT-0 が即座に間違えやすい局面
policy prior が狭すぎる局面
Q の上位手が不安定な局面
入玉勝ちと通常勝ちの選択が割れる局面
reanalyse 候補に送るべき局面
```

### 9.2 position 差分指標

固定局面で以下を測る。

```text
top1_agreement_klent0_gumbel
topk_overlap_klent0_gumbel
policy_kl_div_klent0_to_gumbel
selected_move_q_gap
estimated_regret_by_gumbel_q
normal_component_delta
nyugyoku_component_delta
```

`estimated_regret_by_gumbel_q` は、Gumbel MCTS 側の root action-value を基準に、KLENT-0 選択手がどれくらい劣るかを表す診断値である。

これは真の regret ではないため、対局結果の代替にはしない。

### 9.3 対局差分指標

同じ checkpoint で以下を比較する。

```text
normal_klent0 vs baseline
normal_gumbel_small vs baseline
normal_gumbel_medium vs baseline
nyugyoku_klent0 vs baseline
nyugyoku_gumbel_small vs baseline
```

主な指標は以下である。

```text
win_rate_gain_from_gumbel
declaration_win_rate_gain_from_gumbel
opponent_nyugyoku_loss_rate_delta_from_gumbel
move_time_ms_delta_from_gumbel
```

Gumbel MCTS でだけ大きく改善する局面は、reanalyse 候補として保存する。

---

## 10. offline validation

### 10.1 目的

offline validation は、対局を大量に回す前に、network output と target の整合性を確認するために行う。

主な対象は以下である。

```text
policy_normal_head
policy_style_head
q_normal_result_head
q_nyugyoku_head
outcome_cause_head
moves_left_head
```

### 10.2 policy validation

policy は、固定 validation set 上で以下を測る。

```text
policy_normal_cross_entropy
policy_normal_top1_accuracy
policy_normal_top3_accuracy
policy_normal_legal_mass
policy_style_cross_entropy
policy_style_top1_accuracy
policy_style_top3_accuracy
policy_style_legal_mass
```

`legal_mass` は、合法手 mask 前の softmax を見る場合に、合法手へどれだけ確率が乗っているかを表す診断値である。

ただし、標準の policy 使用時は合法手 mask 後に softmax するため、`legal_mass` は補助指標である。

### 10.3 Q validation

Q は以下を測る。

```text
q_normal_result_mae
q_normal_result_rmse
q_nyugyoku_mae
q_nyugyoku_rmse
q_selected_move_mae
q_full_action_mae
q_rank_correlation_with_reanalyse
q_top1_agreement_with_reanalyse
```

選択手だけに target がある MC sample と、全合法手 target を持つ reanalyse sample は分けて集計する。

非選択合法手の target 欠損を 0 として扱ってはならない。

### 10.4 outcome validation

outcome は以下を測る。

```text
outcome_cross_entropy
outcome_accuracy
outcome_brier_score
outcome_ece
win_normal_recall
win_nyugyoku_recall
draw_recall
loss_nyugyoku_recall
loss_normal_recall
```

入玉勝ち・相手入玉負けは頻度が低いため、全体 accuracy だけでは見ない。

class 別 recall と calibration を必ず見る。

### 10.5 moves_left validation

moves_left は以下を測る。

```text
moves_left_mae_log1p
moves_left_rmse_log1p
moves_left_mae_ply
moves_left_bias_by_phase
```

入玉は長期計画になりやすいため、moves_left を「短く勝つほどよい」という評価には使わない。

moves_left は補助 head の健全性確認と局面分類に使う。

---

## 11. 速度・資源評価

### 11.1 速度指標

速度評価では以下を測る。

```text
inference_batch_size
inference_ms_p50
inference_ms_p95
search_ms_per_move_p50
search_ms_per_move_p95
tactical_ms_per_move_p50
tactical_ms_per_move_p95
moves_per_second
games_per_hour
```

GPU batched inference を使う場合は、batch size 分布も保存する。

```text
batch_size_mean
batch_size_p50
batch_size_p95
inference_queue_wait_ms_p50
inference_queue_wait_ms_p95
```

### 11.2 資源指標

資源評価では以下を測る。

```text
gpu_memory_peak_mb
cpu_memory_peak_mb
gpu_utilization_mean
cpu_utilization_mean
num_threads
model_parameters
checkpoint_size_mb
```

Gumbel MCTS 評価では以下も保存する。

```text
num_simulations
expanded_nodes
nn_cache_hit_rate
mcts_memory_peak_mb
```

### 11.3 強さと速度を混ぜない

速度が速いことは重要だが、勝率・入玉宣言勝ち率とは別に表示する。

評価レポートでは、以下のように並べて比較する。

```text
win_rate
prediction / search latency
games_per_hour
memory
```

速度のために探索設定を変えた場合は、別の `eval_mode` として扱う。

---

## 12. 評価ログ形式

### 12.1 EvalRun

評価 run 全体は以下の構造体相当で保存する。

```python
@dataclass
class EvalRun:
    eval_id: str
    eval_version: str
    created_at: str

    git_commit: str | None
    code_version: str
    rule_version: str
    feature_version: str
    move_label_version: str
    network_output_version: str
    target_version: str

    hardware: dict
    match_config: EvalMatchConfig
    suites: list[EvalSuite]

    games: list["EvalGameRecord"]
    position_results: list["EvalPositionResult"]
    summary: "EvalSummary"

    valid: bool
    errors: list[dict]
```

`eval_version` はログ schema の version である。

### 12.2 EvalGameRecord

1局ごとのログは以下を保存する。

```python
@dataclass
class EvalGameRecord:
    game_id: str
    eval_id: str

    candidate_side: int
    start_position_id: str | None
    start_sfen: str
    start_position_source: str
    start_position_category: str

    candidate_player: PlayerEvalSpec
    opponent_player: PlayerEvalSpec

    terminal_reason: str
    outcome_category: str
    winner: int | None
    winner_is_candidate: bool | None
    terminal_ply: int

    kifu_ref: str | None
    step_summary_ref: str | None

    counts: dict
    timings: dict
    valid: bool
    error: dict | None
```

`winner_is_candidate` は引き分けでは `None` とする。

### 12.3 EvalStepSummary

対局中の各手について、標準では軽量 summary を保存する。

```python
@dataclass
class EvalStepSummary:
    ply: int
    side_to_move: int
    sfen_hash: str
    selected_move: int
    selected_move_usi: str

    mode_id: str
    search_type: str

    q_normal_selected: float | None
    q_nyugyoku_selected: float | None
    q_mode_selected: float | None

    p_prior_selected: float | None
    pi_search_selected: float | None

    forced_reason: str | None
    mate_win_available: bool
    in_check: bool

    elapsed_ms: float
```

full search log は重いため、標準では保存しない。

難所局面や失敗対局だけ full log を保存してよい。

### 12.4 EvalPositionResult

固定局面評価の結果は以下を保存する。

```python
@dataclass
class EvalPositionResult:
    eval_id: str
    suite_id: str
    position_id: str
    sfen: str

    mode_id: str
    selected_move: int | None
    selected_move_usi: str | None

    p_normal_top_moves: list[tuple[int, float]]
    p_style_top_moves: list[tuple[int, float]]
    p_prior_top_moves: list[tuple[int, float]]
    pi_search_top_moves: list[tuple[int, float]]

    q_normal_top_moves: list[tuple[int, float]]
    q_nyugyoku_top_moves: list[tuple[int, float]]
    q_mode_top_moves: list[tuple[int, float]]

    outcome_probs: list[float] | None
    moves_left: float | None

    tactical_result: dict
    expected: dict | None
    passed: bool | None
    error: dict | None
```

fixed suite の診断では、top moves を保存して後から比較できるようにする。

### 12.5 EvalSummary

集計結果は以下を保存する。

```python
@dataclass
class EvalSummary:
    outcome_counts: EvalOutcomeCounts
    rates: dict[str, float | None]
    confidence_intervals: dict[str, tuple[float, float]]

    tactical_metrics: dict
    position_metrics: dict
    speed_metrics: dict
    resource_metrics: dict

    gate_results: dict[str, bool]
    notes: list[str]
```

summary だけでなく、raw game log も保存する。

---

## 13. 統計処理

### 13.1 信頼区間

主要 rate には信頼区間を付ける。

標準は Wilson interval とする。

対象は以下である。

```text
win_rate
declaration_win_rate
opponent_nyugyoku_loss_rate
draw_rate
```

信頼区間は gate 判定の補助であり、raw count も必ず表示する。

### 13.2 paired comparison

同じ開始局面集合で checkpoint を比較する場合は、paired comparison を優先する。

```text
同じ start_position_id
同じ先後割り当てまたは反転ペア
同じ baseline 設定
同じ eval_mode
```

paired comparison では、単純勝率差に加えて以下を保存する。

```text
paired_win_delta
paired_declaration_win_delta
paired_opponent_nyugyoku_loss_delta
```

### 13.3 sequential test

大量評価の途中で明らかに悪い checkpoint を打ち切るため、sequential test を導入してよい。

ただし、主要 gate の最終報告は、事前に決めた `eval_tier` と `num_games` で集計した結果を正とする。

打ち切り評価と正式評価を混ぜてはならない。

---

## 14. checkpoint gate

### 14.1 gate 種別

checkpoint gate は以下に分ける。

```text
correctness gate:
  実装・ルール・ラベル・tactical fixture が壊れていないか

stability gate:
  自己対局・評価対局が失敗なく回るか

normal strength gate:
  通常棋力が基準を満たすか

nyugyoku optimization gate:
  勝率制約を満たしながら入玉宣言勝ち率を改善するか

speed gate:
  推論・探索速度が実用範囲か
```

### 14.2 correctness gate

以下を満たさない checkpoint / code は対局評価へ進めない。

```text
move label round-trip test が通る
合法手 mask test が通る
terminal_reason mapping test が通る
宣言勝ちルール unit test が通る
詰み fixture が通る
王手対応 fixture が通る
network output shape test が通る
SearchResult schema test が通る
TrainingSample schema test が通る
```

### 14.3 stability gate

以下を満たす必要がある。

```text
failed_games = 0
invalid_games = 0
illegal_move = 0
engine crash = 0
NaN / inf 出力 = 0
feature encode error = 0
move decode error = 0
```

smoke 評価で失敗がある場合は、勝率を見ずに修正する。

### 14.4 normal strength gate

通常モードの hard gate は以下である。

```text
win_rate_vs_bakuretsu_komahiro_v420_depth4 >= 0.90
```

評価条件は以下で固定する。

```text
candidate mode = normal_klent0
candidate playouts = 0
baseline = bakuretsu_komahiro_v420_depth4
suite = balanced_positions_2025_24ply_500x2_v1
eval_tier = final
num_games = 1000
```

この gate を満たした checkpoint を `normal_champion_checkpoint` の候補にする。

### 14.5 nyugyoku optimization gate

入玉最適化候補は、まず以下を満たす必要がある。

```text
win_rate_vs_bakuretsu_komahiro_v420_depth4 >= 0.80
```

その上で、以下を最大化する。

```text
declaration_win_rate_vs_bakuretsu_komahiro_v420_depth4
```

評価条件は以下で固定する。

```text
suite = balanced_positions_2025_24ply_500x2_v1
eval_tier = final
num_games = 1000
```

同率の場合は、以下の順で tie-break する。

```text
1. win_rate が高い
2. opponent_nyugyoku_loss_rate が低い
3. draw_rate が低い
4. declaration_win_share_among_wins が高い
5. search_ms_per_move_p95 が低い
```

### 14.6 checkpoint 昇格

checkpoint 昇格では、以下を保存する。

```text
checkpoint_id
parent_checkpoint_id
eval_id
passed_gates
failed_gates
summary metrics
reason_for_promotion
```

`normal_champion_checkpoint` と `nyugyoku_optimized_config` は別々に管理する。

---

## 15. ablation / sweep 評価

### 15.1 探索設定 sweep

入玉最適化では、以下を sweep する。

```text
alpha_style_self
alpha_style_opponent_model
lambda_nyugyoku
klent_alpha_entropy
klent_beta_prior
```

主目的は、勝率 80%以上を維持したまま宣言勝ち率を最大化する設定を探すことである。

### 15.2 学習設定 sweep

学習設定では以下を比較する。

```text
policy_style_target の混合比
q_nyugyoku loss weight
outcome class balancing
nyugyoku sample oversampling
opponent_nyugyoku_risk sample oversampling
position buffer category quota
reanalyse target 混合比
```

比較では、通常棋力と入玉指標を分けて保存する。

### 15.3 tactical budget sweep

tactical solver の budget は以下を比較する。

```text
mate_search_node_limit
mate_search_time_ms
mate_search_max_ply
candidate_limit
```

主指標は以下である。

```text
win_rate
mate_miss_rate
tactical_timeout_rate
search_ms_per_move_p95
```

### 15.4 network / feature ablation

network / feature ablation では以下を比較する。

```text
model size
VoVNetV2 block 数
head adapter 幅
入玉関連入力特徴の有無
利き特徴の有無
policy_style_head の有無
q_nyugyoku_head の有無
moves_left_head の有無
```

`q_nyugyoku_head` や `policy_style_head` を外す ablation は、仕様上の標準構成ではなく、効果測定用の実験として扱う。

---

## 16. レポート形式

### 16.1 標準レポート

標準評価レポートには以下を含める。

```text
評価対象 checkpoint / config
baseline engine / config
評価 suite
対局数
raw counts
主要 rates
信頼区間
終局理由別 breakdown
先後別 breakdown
入玉関連 breakdown
tactical fixture / timeout / speed metrics
速度・資源指標
gate 判定
失敗対局一覧
代表棋譜リンク
```

### 16.2 指標テーブル

最低限、以下の表を出す。

```text
overall results
terminal reason breakdown
nyugyoku metrics
tactical metrics
speed metrics
by-side results
by-start-position-category results
```

### 16.3 棋譜サンプル

レポートには、代表棋譜を保存する。

```text
通常勝ち代表
入玉宣言勝ち代表
相手入玉負け代表
最大手数引き分け代表
Gumbel MCTS と KLENT-0 の選択が割れた代表
詰み逃しがあればその代表
```

棋譜は後から再生できる形式にする。

```text
USI move list
開始 SFEN
終局理由
各手の mode / selected_move / forced_reason
```

---

## 17. 実装コンポーネント

### 17.1 EvaluationRunner

`EvaluationRunner` は評価全体を管理する。

責務は以下である。

```text
EvalMatchConfig を受け取る
engine process を起動する
開始局面を供給する
対局を進行する
EvalGameRecord を保存する
EvalSummary を作る
```

### 17.2 EngineAdapter

engine ごとに adapter を分ける。

```text
TamanoboriAdapter
BakuretsuKomahiroAdapter
USIEngineAdapter
```

adapter は、探索設定とエンジン option を正しく反映し、結果を共通形式へ変換する。

### 17.3 MetricsAggregator

`MetricsAggregator` は、game log と position result から指標を計算する。

責務は以下である。

```text
raw count 集計
rate 計算
confidence interval 計算
先後別集計
開始局面カテゴリ別集計
終局理由別集計
gate 判定
```

### 17.4 ReportWriter

`ReportWriter` は、評価結果を以下へ出力する。

```text
JSONL raw log
JSON summary
Markdown report
CSV metrics table
kifu directory
```

Markdown report は人間が読むための要約であり、正本は JSON / JSONL の raw log とする。

---

## 18. 実装上の不変条件

以下を必ず守る。

```text
評価対局では stochastic sampling を標準にしない
主要 gate の対局評価は BalancedPositions2025 24手目 500局面×先後入れ替えの1000局に固定する
通常棋力の主評価にはシャッフル開始局面を混ぜない
failed game を勝敗に混ぜない
illegal move を terminal_reason にしない
max_ply_draw を勝敗 adjudication に変換しない
宣言勝ちは nyugyoku outcome として集計する
相手の宣言勝ちは opponent_nyugyoku_loss として集計する
勝率と宣言勝ち率は raw count から再計算可能にする
normal_champion_checkpoint と nyugyoku_optimized_config を混同しない
評価 suite version を固定する
baseline engine の version と option を保存する
```

---

## 19. 検証項目

### 19.1 unit test

```text
terminal_reason -> outcome_category mapping が正しい
EvalOutcomeCounts が正しく集計される
win_rate / declaration_win_rate が raw count から正しく計算される
wins = 0 のとき declaration_win_share_among_wins が None になる
failed game が valid_games に含まれない
先後別集計が正しい
開始局面カテゴリ別集計が正しい
Wilson interval が計算できる
```

### 19.2 integration test

```text
TamanoboriAdapter で1局評価できる
BakuretsuKomahiroAdapter で1局評価できる
平手初期局面から smoke 評価を実行できる
固定局面 suite を読み込める
tactical fixture を評価できる
EvalRun JSON を保存・再読み込みできる
Markdown report を生成できる
```

### 19.3 regression test

```text
既知 checkpoint の評価 summary が再現する
既知 fixture の pass/fail が再現する
同じ seed / 同じ設定で deterministic 評価が一致する
baseline option を変えた場合に eval_id が変わる
suite version を変えた場合に比較対象が分離される
```

---

## 20. 実験項目

初期実装後の実験項目は以下である。

```text
BalancedPositions2025 24手目 1000局での通常モード勝率
勝率 80%以上制約下の declaration_win_rate 最大化
alpha_style_self sweep
alpha_style_opponent_model sweep
lambda_nyugyoku sweep
position buffer category quota の効果
policy_style_head target 混合比の効果
q_nyugyoku loss weight の効果
tactical solver budget の効果
Gumbel MCTS small / medium の効果
シャッフル開始局面を学習に使った場合の通常評価への影響
入玉関連入力特徴の ablation
model size scaling
```

---

## 21. 暫定結論

評価仕様の中心は以下である。

```text
通常棋力:
  BalancedPositions2025 24手目 500局面×先後入れ替えの1000局で、
  playouts=0 / normal_klent0 が、
  爆裂駒拾太郎V4.2.0 4手読みに対して勝率 90%以上を目指す

入玉最適化:
  同じ1000局評価で playouts=0 の勝率 80%以上を維持する設定の中で、
  入玉宣言勝ち率を最大化する
```

通常モード 90% gate と、入玉宣言勝ち最大化は分けて扱う。

評価では、勝率、宣言勝ち率、相手入玉負け率、引き分け率、tactical 指標、探索速度を別々に保存する。

`docs/14_evaluation.md` は、これらの評価指標、ログ schema、gate、ablation の正本とする。

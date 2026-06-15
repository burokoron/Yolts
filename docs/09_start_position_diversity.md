# docs/09_start_position_diversity.md - 開始局面多様化仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎における **開始局面多様化** の仕様を定義する。

対象は以下である。

```text
開始局面多様化の基本方針
開始局面 source の種類
将棋81万風シャッフル局面
2手ランダム局面
position buffer 由来局面
phase 別の混合比
局面品質フィルタ
重複排除
開始局面ログ形式
自己対局ループとの境界
学習サンプル生成との関係
実装上の不変条件
検証項目
実験項目
```

爆裂玉登太郎では、自己対局の主ループを **KLENT-0 + CPU tactical solver** とする。

KLENT-0 は root 局面の policy / action-value に強く依存する。学習初期に序盤が collapse すると、似た局面ばかりが生成され、position buffer も偏る。

そのため、開始局面多様化は、学習初期の探索幅を確保し、自己対局データの局面分布を広げるための補助機構として導入する。

関連仕様は以下を正とする。

```text
全体方針: 爆裂玉登太郎_設計概要.md
探索・着手選択: docs/03_search.md
自己対局ループ: docs/08_selfplay.md
CPU tactical solver: docs/07_tactical_solver.md
KLENT-first 学習方針: docs/13_klent_training.md
```

---

## 1. 基本方針

### 1.1 人間定跡を標準では使わない

開始局面の多様化には、人間定跡を標準では使わない。

理由は以下である。

```text
人間定跡の分布に依存しすぎると、自己対局で発見すべき棋風が狭くなる
入玉寄りの棋風は人間定跡分布と合わない可能性がある
定跡選定・バージョン管理・著作権・棋譜出所管理が複雑になる
通常棋力と入玉棋風のどちらに効いたかを切り分けにくい
```

標準では、人間棋譜に依存しない以下の source で開始局面を多様化する。

```text
平手初期局面
2手ランダム局面
将棋81万風シャッフル局面
自己対局 position buffer 局面
```

### 1.2 cold start 補助と成熟後の多様化を分ける

開始局面多様化には、2つの役割がある。

```text
cold start 補助:
  policy / Q / position buffer が未成熟な段階で、序盤 collapse を避ける

成熟後の多様化:
  自己対局で得た有用局面から再開し、入玉関連局面と難所局面を厚く学習する
```

`shuffled_initial` と `random_opening` は、主に cold start 補助である。

`position_buffer` は、自己対局が進んだ後の主な多様化手段である。

### 1.3 シャッフル局面は長期的な主軸にしない

将棋81万風シャッフル局面は、平手初期局面とは異なる非標準の初期配置である。

局面多様性を増やす効果は期待できるが、通常将棋の自然な序盤分布とは異なる。

そのため、以下を標準方針とする。

```text
学習初期は高めに混ぜる
自己対局が安定したら段階的に下げる
position buffer が成熟したら低比率または無効にする
通常棋力の評価対局には使わない
```

### 1.4 評価対局と学習用自己対局を分ける

開始局面多様化は、学習用自己対局のための機構である。

主要 gate の対局評価では、開始局面 source を目的別に変えない。標準対局評価は `docs/14_evaluation.md` の `balanced_positions_2025_24ply_500x2_v1` に統一する。

```text
source = やねうら王互角局面集2025
start_ply = 24
sampled_positions = 500
side_swapped_games_per_position = 2
total_games = 1000
```

通常棋力評価、入玉関連評価、相手入玉耐性評価、探索設定 sweep は、同じ開始局面セットで比較する。評価目的ごとに開始局面を変えると、勝率差・宣言勝ち率差・相手入玉負け率差が、棋力や探索設定の差なのか開始局面分布の差なのか切り分けにくくなるためである。

以下は診断用の別実験として扱い、主要 gate には混ぜない。

```text
平手初期局面からの sanity check
入玉有望局面からの再開評価
相手入玉リスク局面からの再開評価
相入玉風局面からの再開評価
シャッフル開始局面からの耐性評価
開始局面 source 別の ablation
```

シャッフル局面からの勝率を、通常棋力の主指標として扱ってはならない。

### 1.5 開始局面 source は必ずログに残す

開始局面がどこから来たかは、学習データの性質に強く影響する。

したがって、自己対局ログには必ず以下を保存する。

```text
start_position.source
start_position.category
start_position.sfen
start_position.ply_offset
start_position.tags
```

特にシャッフル局面では、以下を保存する。

```text
shuffle_rule_id
shuffle_seed
shuffle_rank_8_permutation
shuffle_rank_9_permutation
is_shuffled_initial
```

position buffer 由来局面では、以下を保存する。

```text
source_game_id
source_ply
source_model_version
source_category_scores
buffer_sample_weight
```

---

## 2. 開始局面 source

### 2.1 source 一覧

標準で扱う source は以下である。

| source | 目的 | 標準用途 |
| --- | --- | --- |
| `initial` | 平手初期局面 | 全 phase の基準 |
| `random_opening` | 2手程度のランダム合法手で序盤を散らす | cold start 補助 |
| `shuffled_initial` | 駒配置をシャッフルした開始局面 | cold start 補助 |
| `position_buffer` | 自己対局で得た局面から再開 | 成熟後の主軸 |
| `manual` | 手動指定局面 | デバッグ・評価局面作成 |
| `test` | unit test 用局面 | テスト専用 |

通常の自己対局では、`manual` と `test` は sampling 対象にしない。

### 2.2 initial

`initial` は通常の平手初期局面である。

```text
source = "initial"
category = "hirate"
ply_offset = 0
```

学習全体を通して必ず一定割合を残す。

理由は以下である。

```text
通常将棋の主分布から離れすぎないようにする
序盤の基本的な棋力を維持する
通常評価対局との乖離を抑える
```

### 2.3 random_opening

`random_opening` は、平手初期局面から少数手のランダム合法手を進めた局面である。

初期実装の標準は **2手ランダム** とする。

```text
source = "random_opening"
category = "random_2ply"
random_opening_max_ply = 2
```

2手に留める理由は以下である。

```text
実装が単純である
局面が通常将棋の到達可能局面である
序盤分布を少しだけ散らせる
ランダム手による極端な悪形を増やしすぎない
```

4手以上のランダム局面は実験候補とする。

ランダム手は、人間定跡や教師 policy ではなく、合法手集合から生成する。

ただし、明らかに不正な局面を避けるため、生成後に品質フィルタを通す。

### 2.4 shuffled_initial

`shuffled_initial` は、平手初期局面の駒配置を一定制約でシャッフルした局面である。

標準 rule は以下とする。

```text
rule_id = "shogi81man_v1"
```

これは、将棋81万系の考え方を参考にした、人間棋譜に依存しない開始局面多様化である。

標準では以下を行う。

```text
先手側の8段目を8段目内でシャッフルする
先手側の9段目を9段目内でシャッフルする
後手側の1段目と2段目は、先手側配置を180度回転した配置にする
歩の段は固定する
手番は先手番から始める
持ち駒は空にする
```

この source は、cold start 補助であり、長期的には比率を下げる。

### 2.5 position_buffer

`position_buffer` は、過去の自己対局から抽出した局面を開始局面として再利用する。

目的は以下である。

```text
入玉有望局面を厚く学習する
相手入玉リスク局面を増やす
相入玉になりやすい局面を増やす
policy / Q が不安定な局面を再学習する
KLENT-0 と Gumbel MCTS の差が大きい局面を深掘りする
```

自己対局が成熟した後は、`position_buffer` を開始局面多様化の主軸にする。

---

## 3. 設定構造

### 3.1 StartPositionConfig

開始局面 sampling の設定は、以下の構造体相当で管理する。

```python
@dataclass
class StartPositionConfig:
    config_id: str

    # source switches
    use_initial_position: bool = True
    use_random_opening_positions: bool = False
    use_shuffled_initial_positions: bool = False
    use_position_buffer: bool = False

    # source weights
    initial_position_weight: float = 1.0
    random_opening_weight: float = 0.0
    shuffled_initial_position_weight: float = 0.0
    position_buffer_weight: float = 0.0

    # random opening
    random_opening_config: "RandomOpeningConfig" = field(default_factory=RandomOpeningConfig)

    # shuffled initial
    shuffle_config: "StartPositionShuffleConfig" = field(default_factory=StartPositionShuffleConfig)

    # position buffer
    position_buffer_config: "PositionBufferSamplingConfig" = field(default_factory=PositionBufferSamplingConfig)

    # common filters
    require_legal_position: bool = True
    require_non_terminal: bool = True
    require_no_declaration_win_at_start: bool = True
    require_feature_encodable: bool = True
    require_has_legal_moves: bool = True

    max_retry_per_source: int = 100
    max_total_retry: int = 300
    fallback_to_initial: bool = True
```

weight は有効な source の間で正規化して使う。

無効 source の weight が正であっても sampling 対象にしてはならない。

### 3.2 RandomOpeningConfig

```python
@dataclass
class RandomOpeningConfig:
    rule_id: str = "random_legal_v1"
    min_ply: int = 2
    max_ply: int = 2

    move_sampling: str = "uniform_legal"  # "uniform_legal" | "policy_prior"
    avoid_terminal_during_generation: bool = True
    avoid_declaration_win_during_generation: bool = True

    max_retry: int = 100
```

初期実装では `move_sampling = "uniform_legal"` とする。

`policy_prior` でランダム序盤を作る方式は、policy が未成熟な段階では collapse を助長する可能性があるため標準にしない。

### 3.3 StartPositionShuffleConfig

```python
@dataclass
class StartPositionShuffleConfig:
    enabled: bool = False
    rule_id: str = "shogi81man_v1"

    shuffle_black_rank_8: bool = True
    shuffle_black_rank_9: bool = True
    mirror_white_by_180deg: bool = True
    keep_pawns_fixed: bool = True
    side_to_move: str = "black"

    allow_empty_square_shuffle_on_rank_8: bool = True
    allow_king_file_change: bool = True
    allow_bishop_rook_file_change: bool = True

    require_legal_position: bool = True
    require_non_terminal: bool = True
    require_no_declaration_win_at_start: bool = True
    require_no_check_at_start: bool = True
    require_no_immediate_mate_at_start: bool = True

    deduplicate_by_sfen: bool = True
    max_retry: int = 100
```

`allow_empty_square_shuffle_on_rank_8 = True` は、8段目の空白マスも含めてシャッフルすることを表す。

### 3.4 PositionBufferSamplingConfig

```python
@dataclass
class PositionBufferSamplingConfig:
    enabled: bool = False

    min_buffer_size: int = 100_000
    min_start_ply: int = 8
    max_start_ply: int | None = 256

    category_weights: dict[str, float] = field(default_factory=dict)
    source_model_weighting: str = "latest_and_recent"  # "latest_only" | "latest_and_recent" | "all"
    recency_half_life_games: int | None = None

    deduplicate_by_position_key: bool = True
    max_same_source_game_per_batch: int | None = None
    max_same_opening_prefix_per_batch: int | None = None

    require_legal_position: bool = True
    require_non_terminal: bool = True
    require_no_declaration_win_at_start: bool = True
    require_feature_encodable: bool = True
    allow_in_check: bool = True

    max_retry: int = 100
```

`allow_in_check = True` とする理由は、実戦到達局面では王手中の局面も有効な学習対象だからである。

`shuffled_initial` と `random_opening` では、生成品質を安定させるため `require_no_check_at_start = True` を使う。

### 3.5 PhaseStartPositionSchedule

phase 別の source weight は、明示的な schedule として管理する。

```python
@dataclass
class PhaseStartPositionSchedule:
    phase_id: str
    initial_weight: float
    shuffled_initial_weight: float
    random_opening_weight: float
    position_buffer_weight: float

    min_steps: int | None = None
    max_steps: int | None = None
    min_games: int | None = None
    max_games: int | None = None

    activation_conditions: dict[str, float | int | bool] = field(default_factory=dict)
```

phase は training pipeline から渡してもよいし、自己対局 manager が checkpoint / buffer 状態から決めてもよい。

ただし、ログには必ず `phase_id` と実際の source weight を保存する。

---

## 4. phase 別の混合比

### 4.1 Phase 1: 初期学習直後

目的は、自己対局の序盤 collapse を避けることである。

候補比率は以下である。

```text
initial: 40% 〜 60%
shuffled_initial: 30% 〜 40%
random_2ply: 0% 〜 20%
position_buffer: 0% 〜 10%
```

標準初期値は以下とする。

```text
initial: 50%
shuffled_initial: 35%
random_2ply: 10%
position_buffer: 5%
```

position buffer が存在しない場合は、その weight を `initial` と `random_2ply` に再配分する。

### 4.2 Phase 2: 自己対局安定化

目的は、シャッフル依存を下げつつ、自己対局で得た局面を混ぜることである。

候補比率は以下である。

```text
initial: 40% 〜 50%
shuffled_initial: 10% 〜 25%
random_2ply: 5% 〜 15%
position_buffer: 20% 〜 40%
```

標準初期値は以下とする。

```text
initial: 45%
shuffled_initial: 20%
random_2ply: 10%
position_buffer: 25%
```

### 4.3 Phase 3: position buffer 主体

目的は、有用局面からの再開を主軸にすることである。

候補比率は以下である。

```text
initial: 30% 〜 50%
shuffled_initial: 0% 〜 10%
random_2ply: 0% 〜 10%
position_buffer: 40% 〜 60%
```

標準初期値は以下とする。

```text
initial: 40%
shuffled_initial: 5%
random_2ply: 5%
position_buffer: 50%
```

### 4.4 Phase 4: 本番評価に近い学習

目的は、通常の平手分布と有用再開局面のバランスを取ることである。

標準初期値は以下とする。

```text
initial: 50%
shuffled_initial: 0% 〜 5%
random_2ply: 0% 〜 5%
position_buffer: 45% 〜 50%
```

通常棋力が劣化した場合は、まず `initial` と `normal_midgame` を増やす。

入玉サンプルが不足する場合は、シャッフル率を上げるのではなく、position buffer の入玉関連 category quota を増やす。

### 4.5 自動調整条件

source weight は固定でもよいが、以下の指標で自動調整してよい。

```text
start_position/duplicate_rate
start_position/unique_sfen_count
selfplay/policy_entropy_opening
selfplay/short_game_rate
selfplay/nyugyoku_position_rate
selfplay/opponent_nyugyoku_risk_rate
eval/normal_win_rate
```

調整方針は以下である。

```text
序盤重複率が高い:
  shuffled_initial, random_2ply, root noise, temperature を上げる

通常勝率が下がる:
  shuffled_initial を下げる
  initial と normal_midgame を上げる

入玉局面が不足する:
  position_buffer の nyugyoku 系 category weight を上げる

短手数負けが増える:
  random_opening と shuffled_initial の品質フィルタを強める
  詰み探索ログと短手数負け局面を確認する
```

---

## 5. 将棋81万風シャッフル局面

### 5.1 rule_id

初期実装の rule id は以下とする。

```text
shogi81man_v1
```

この rule は、将棋81万の思想を参考にしつつ、爆裂玉登太郎の cold start 用に単純化したものとして扱う。

厳密に既存の将棋81万データセットを再現することを目的にしない。

### 5.2 対象段

平手初期局面を基準に、先手側の以下をシャッフルする。

```text
8段目
9段目
```

後手側は、先手側の配置を180度回転して作る。

```text
先手9段目 -> 後手1段目
先手8段目 -> 後手2段目
```

歩の段は固定する。

```text
先手7段目の歩は固定
後手3段目の歩は固定
```

### 5.3 8段目の扱い

平手初期局面の先手8段目には、角、飛、空白マスが含まれる。

標準では、空白マスも含めて8段目全体をシャッフルする。

```text
rank_8_slots = [empty, bishop, empty, empty, empty, empty, empty, rook, empty]
rank_8_shuffled = permutation(rank_8_slots)
```

これにより、飛角の初期位置が多様化する。

空白を含めずに飛角だけを入れ替える方式は、局面数が少ないため標準にしない。

### 5.4 9段目の扱い

平手初期局面の先手9段目は、以下の駒を含む。

```text
香 桂 銀 金 玉 金 銀 桂 香
```

標準では、この9枚を9段目内でシャッフルする。

```text
rank_9_slots = [lance, knight, silver, gold, king, gold, silver, knight, lance]
rank_9_shuffled = permutation(rank_9_slots)
```

玉の file 変更を許可する。

玉の file を固定する制限は、局面数を減らし、初期局面多様化の効果を弱めるため標準にしない。

### 5.5 先後対称性

後手配置は、先手配置を180度回転して作る。

この制約により、以下を保つ。

```text
先後の駒種数が一致する
先後の初期配置が対称になる
片側だけが構造的に有利な局面を避ける
```

手番は平手と同じく先手番から始める。

先手番の有利不利は通常将棋と同様に残るが、初期配置の非対称性は入れない。

### 5.6 持ち駒

シャッフル局面の持ち駒は空にする。

```text
black_hand = empty
white_hand = empty
```

持ち駒をランダム化する方式は、初期実装では採用しない。

理由は以下である。

```text
合法性・到達可能性の判断が複雑になる
宣言点数や即詰みの偏りが強くなる
開始局面として不自然な戦術局面が増えやすい
```

### 5.7 生成手順

疑似コードは以下である。

```python
def generate_shogi81man_v1(rng, config):
    position = make_hirate_initial_position()

    black_rank_8 = get_rank(position, side="black", rank=8)
    black_rank_9 = get_rank(position, side="black", rank=9)

    if config.shuffle_black_rank_8:
        black_rank_8 = rng.permutation(black_rank_8)

    if config.shuffle_black_rank_9:
        black_rank_9 = rng.permutation(black_rank_9)

    set_rank(position, side="black", rank=8, pieces=black_rank_8)
    set_rank(position, side="black", rank=9, pieces=black_rank_9)

    white_rank_1, white_rank_2 = rotate_180_for_white(
        black_rank_9=black_rank_9,
        black_rank_8=black_rank_8,
    )

    set_rank(position, side="white", rank=1, pieces=white_rank_1)
    set_rank(position, side="white", rank=2, pieces=white_rank_2)

    keep_pawns_fixed(position)
    clear_hands(position)
    set_side_to_move(position, BLACK)

    return position
```

### 5.8 品質フィルタ

生成したシャッフル局面は、必ず以下を確認する。

```text
駒数が正しい
先手玉と後手玉が1枚ずつ存在する
両玉が隣接していない
手番側が合法である
合法手が1手以上ある
開始時点で terminal ではない
開始時点で宣言勝ち条件を満たさない
開始時点で王手がかかっていない
開始時点で片側だけが明らかな即詰み状態ではない
feature encoder が処理できる
move generator が処理できる
CPU tactical solver が処理できる
```

`require_no_immediate_mate_at_start` は初期実装では有効にする。

標準では、開始時点の1手詰め確認と合法性確認を行う。

### 5.9 重複排除

シャッフル局面は、SFEN または position key で重複排除する。

```text
position_key = board + hands + side_to_move
```

反復回数や最大手数までの残りは、開始局面 key に含めない。

理由は、初期実装の network input に反復関連特徴を含めないためである。

### 5.10 sampling と事前生成

シャッフル局面は、以下のどちらで生成してもよい。

```text
on-the-fly 生成
事前生成した shuffled position pool から sampling
```

初期実装では on-the-fly 生成でよい。

ただし、実験再現性を重視する場合は、事前生成 pool を使う。

事前生成 pool では以下を保存する。

```python
@dataclass
class ShuffledInitialPositionRecord:
    position_id: str
    sfen: str
    rule_id: str
    shuffle_seed: int
    rank_8_permutation: list[int]
    rank_9_permutation: list[int]
    validation_flags: dict[str, bool]
    created_at: str
```

---

## 6. 2手ランダム局面

### 6.1 目的

2手ランダム局面の目的は、平手初期局面から到達可能な範囲で、序盤を軽く散らすことである。

シャッフル局面と違い、通常将棋の合法到達局面である。

一方で、完全ランダム手は悪形を作る可能性があるため、手数は短く保つ。

### 6.2 標準生成

標準は以下である。

```text
平手初期局面から開始
合法手から一様 sampling
2 ply 進める
生成後に品質フィルタを通す
```

疑似コードは以下である。

```python
def generate_random_opening_position(rng, config):
    position = make_hirate_initial_position()
    ply = rng.randint(config.min_ply, config.max_ply + 1)

    moves_played = []
    for _ in range(ply):
        legal_moves = generate_legal_moves(position)
        if not legal_moves:
            return None

        move = rng.choice(legal_moves)
        position.do(move)
        moves_played.append(move)

        if config.avoid_terminal_during_generation:
            if is_terminal(position):
                return None

    return position, moves_played
```

### 6.3 random_opening のフィルタ

生成後、以下を確認する。

```text
合法局面である
terminal ではない
宣言勝ち条件を満たさない
合法手が1手以上ある
feature encoder で処理できる
```

王手中の局面を許すかは config で制御する。

初期実装では、2手ランダム局面では王手中を許可してよい。

理由は、通常到達局面であり、王手対応学習にもなるためである。

ただし、短手数負けが増える場合は、`require_no_check_at_start` を追加する。

### 6.4 手数を伸ばす場合

`random_opening_max_ply > 2` は実験候補とする。

採用条件は以下である。

```text
短手数負け率が増えない
通常勝率が劣化しない
局面多様性が有意に増える
position buffer が未成熟な phase に限る
```

標準で4手以上にしない。

---

## 7. position buffer 由来局面

### 7.1 position buffer の役割

position buffer は、自己対局から得た有用局面を将来の開始局面として再利用するためのバッファである。

目的は以下である。

```text
入玉関連局面を増やす
相入玉局面を増やす
相手入玉リスク局面を増やす
policy / Q が迷っている局面を増やす
短期戦術の mask が多い局面を増やす
通常中盤局面を維持する
```

### 7.2 PositionBufferRecord

position buffer の1レコードは、以下の構造体相当とする。

```python
@dataclass
class PositionBufferRecord:
    position_id: str
    sfen: str
    position_key: str

    source_game_id: str
    source_ply: int
    absolute_ply: int
    side_to_move: int

    model_version: str
    feature_version: str
    move_label_version: str
    selfplay_config_id: str
    search_config_id: str
    pairing_id: str

    category_scores: dict[str, float]
    nyugyoku_features: dict[str, float | int | bool]
    uncertainty_features: dict[str, float | int | bool]
    tactical_features: dict[str, float | int | bool]

    source_result: "GameResultSummary | None"
    created_at: str
```

`source_result` は解析用である。

その局面から再開した対局の training target には、再開後の対局結果を使う。

### 7.3 category

標準 category は以下である。

```text
normal_midgame
nyugyoku_promising
opponent_nyugyoku_risk
mutual_nyugyoku_like
tactical_uncertain
policy_uncertain
q_close_top_moves
q_disagreement
reanalyse_interesting
```

各 category は排他的ではない。

1つの局面が複数 category に属してよい。

### 7.4 category score の例

`nyugyoku_promising` は以下から作る。

```text
自玉が敵陣に近い
自玉が敵陣内にいる
自駒の敵陣内枚数が多い
宣言点数の残りが小さい
P_win_nyugyoku が高い
Q_nyugyoku_selected が高い
```

`opponent_nyugyoku_risk` は以下から作る。

```text
相手玉が自陣に近い
相手玉が自陣内にいる
相手駒の自陣内枚数が多い
相手の宣言点数残りが小さい
P_loss_nyugyoku が高い
Q_nyugyoku_selected が低い
```

`policy_uncertain` は以下から作る。

```text
policy entropy が高い
pi_klent と p_prior の KL が大きい
top1 確率が低い
上位候補が拮抗している
```

`q_close_top_moves` は以下から作る。

```text
Q_mode 上位2手の差が小さい
Q_normal_result と Q_nyugyoku の推奨手が異なる
通常勝ちと入玉勝ちの選択が割れている
```

`tactical_uncertain` は以下から作る。

```text
mate_in_one が絡む
王手中で合法応手が少ない
詰み探索が timeout した
```

### 7.5 buffer sampling

position buffer からの sampling は、category quota を使う。

```python
category_weights = {
    "normal_midgame": 0.30,
    "nyugyoku_promising": 0.25,
    "opponent_nyugyoku_risk": 0.15,
    "mutual_nyugyoku_like": 0.10,
    "policy_uncertain": 0.10,
    "q_close_top_moves": 0.05,
    "tactical_uncertain": 0.05,
}
```

この値は初期候補であり、通常棋力と入玉指標を見て調整する。

入玉サンプルが不足する場合は、まず以下を上げる。

```text
nyugyoku_promising
opponent_nyugyoku_risk
mutual_nyugyoku_like
```

通常棋力が劣化した場合は、以下を上げる。

```text
initial
normal_midgame
```

### 7.6 source model の扱い

古い model 由来の局面も、多様性として有用な場合がある。

ただし、古すぎる model の局面は、現在の探索・feature・ルール定義と合わない可能性がある。

標準は以下とする。

```text
latest_and_recent:
  最新 model と直近数世代の model 由来局面を使う
```

`latest_only` は安定重視、`all` は多様性重視の実験候補とする。

### 7.7 position buffer のフィルタ

開始局面として使う前に、以下を確認する。

```text
合法局面である
terminal ではない
宣言勝ちがすでに成立していない
合法手が1手以上ある
feature encoder で変換できる
move label encoder / decoder と整合する
現在の rule_config と矛盾しない
```

position buffer 由来では、王手中を許可する。

ただし、以下は除外する。

```text
すでに詰んでいる
合法手がない
全合法手が tactical solver で異常扱いになる
手番・持ち駒・駒数が壊れている
```

---

## 8. StartPositionSampler

### 8.1 標準 API

開始局面 sampler は、以下の API を持つ。

```python
class StartPositionSampler:
    def __init__(
        self,
        config: StartPositionConfig,
        position_buffer: "PositionBuffer | None",
        rule_engine: "RuleEngine",
        tactical_solver: "TacticalSolver | None",
    ):
        ...

    def sample(self, rng: "RandomState") -> "StartPositionSample":
        ...
```

### 8.2 StartPositionSample

```python
@dataclass
class StartPositionSample:
    position: "Position"
    info: "StartPositionInfo"
```

`position` は、自己対局 worker がそのまま対局開始に使える局面である。

`info` はログへ保存する。

### 8.3 StartPositionInfo

```python
@dataclass
class StartPositionInfo:
    source: str
    category: str | None
    sfen: str
    position_key: str

    ply_offset: int
    absolute_ply: int | None

    source_game_id: str | None = None
    source_ply: int | None = None
    source_position_id: str | None = None

    generation_rule_id: str | None = None
    generation_seed: int | None = None
    validation_flags: dict[str, bool] = field(default_factory=dict)
    tags: dict[str, float | int | bool | str] = field(default_factory=dict)

    fallback_used: bool = False
    fallback_reason: str | None = None
```

`ply_offset` は、その開始局面から対局ログ内の `ply=0` までに、生成上どれだけ手が進んでいるかを表す。

```text
initial: 0
shuffled_initial: 0
random_2ply: 2
position_buffer: source_ply または absolute_ply
```

ただし、学習 target の `moves_left` は再開後の対局の終局までの残り手数で作る。

source game の終局までの残り手数を使ってはならない。

### 8.4 sampling 手順

標準手順は以下である。

```text
1. 有効 source の weight を正規化する
2. source を sampling する
3. source ごとの生成器で候補局面を作る
4. 共通品質フィルタを通す
5. source 固有フィルタを通す
6. StartPositionInfo を作る
7. 失敗時は retry する
8. retry 上限を超えたら fallback する
```

疑似コードは以下である。

```python
def sample_start_position(config, rng):
    for retry in range(config.max_total_retry):
        source = sample_source_by_weight(config, rng)
        candidate = generate_candidate(source, config, rng)
        if candidate is None:
            continue

        ok, flags = validate_start_position(candidate.position, config)
        if not ok:
            continue

        return make_start_position_sample(candidate, flags)

    if config.fallback_to_initial:
        return make_initial_position_sample(fallback_reason="retry_exhausted")

    raise StartPositionSamplingError("failed to sample valid start position")
```

### 8.5 fallback

fallback は、自己対局生成を止めないための安全機構である。

標準 fallback は `initial` とする。

fallback が発生した場合は必ずログに残す。

```text
start_position/fallback_used = True
start_position/fallback_reason
start_position/original_source
```

fallback 率が高い場合は、局面生成器または品質フィルタに問題がある。

---

## 9. 品質フィルタ

### 9.1 共通フィルタ

すべての source に共通するフィルタは以下である。

```text
駒数が正しい
両玉が存在する
手番が正しい
合法手生成に成功する
合法手が1手以上ある
terminal ではない
宣言勝ち条件がすでに成立していない
feature encoder で処理できる
move label encoder / decoder と整合する
```

### 9.2 source 別フィルタ

`initial`:

```text
常に通る。ただし実装バグ検出のため共通フィルタは通す
```

`random_opening`:

```text
通常到達局面である
生成途中で terminal になっていない
生成後に合法手がある
必要なら王手中を許す
```

`shuffled_initial`:

```text
王手中ではない
```

`position_buffer`:

```text
現在の rule / feature version で再解釈できる
terminal ではない
宣言勝ち済みではない
古すぎる model / feature version を必要に応じて除外する
```

### 9.3 宣言勝ち条件フィルタ

開始時点ですでに宣言勝ち可能な局面は、標準では除外する。

理由は以下である。

```text
開始直後に終局して自己対局サンプルがほとんど得られない
宣言勝ち可能判定の単体テストには有用だが、通常の自己対局開始局面としては偏る
入玉勝ち target が forced action に偏りすぎる
```

ただし、宣言勝ちテスト用の `manual` / `test` source では許可してよい。

### 9.4 tactical フィルタ

シャッフル局面では、浅い tactical filter を使う。

標準は以下である。

```text
相手玉の1手詰めが開始時点で存在しない
開始直後に terminal にならない
合法手が1手以上ある
```

tactical filter は、開始局面生成を軽量に保つため 1手詰め確認を基本にする。

### 9.5 到達可能性

`random_opening` と `position_buffer` は通常将棋から到達可能な局面である。

`shuffled_initial` は標準平手からの到達可能性を保証しない。

これは設計上許容する。

ただし、以下を満たす必要がある。

```text
将棋ルール上の局面表現として合法である
駒数・持ち駒・手番が一貫している
探索・feature・tactical solver が処理できる
```

到達可能性を厳密に検査する処理は初期実装の標準にしない。

---

## 10. 重複排除と多様性管理

### 10.1 position key

重複排除には以下の key を使う。

```text
board
hands
side_to_move
```

model version、feature version、search config は position key には含めない。

これらはログ metadata として保持する。

### 10.2 重複排除の単位

重複排除は以下の単位で行う。

```text
shuffled_initial pool 内
position buffer 内
1 generation batch 内
training window 内の統計
```

完全に全期間で一度も同じ局面を使わない必要はない。

重要なのは、短期間に同一局面へ過度に偏らないことである。

### 10.3 多様性指標

開始局面 sampling では、以下を継続的にログする。

```text
start_position/source_count
start_position/source_ratio
start_position/unique_position_key_count
start_position/duplicate_rate
start_position/fallback_rate
start_position/filter_reject_rate_by_reason
start_position/legal_moves_mean
start_position/legal_moves_p10
start_position/legal_moves_p90
start_position/in_check_rate
start_position/declaration_available_reject_rate
```

シャッフル局面では追加で以下をログする。

```text
shuffled/unique_rank_8_permutation_count
shuffled/unique_rank_9_permutation_count
shuffled/king_file_histogram
shuffled/rook_file_histogram
shuffled/bishop_file_histogram
shuffled/filter_reject_rate
```

position buffer では追加で以下をログする。

```text
buffer/category_sample_count
buffer/category_sample_ratio
buffer/source_model_version_count
buffer/source_ply_histogram
buffer/nyugyoku_feature_histogram
buffer/uncertainty_feature_histogram
```

### 10.4 局面偏りへの対策

局面が偏る場合の対策は、原因別に分ける。

```text
平手序盤が同じ手順に偏る:
  temperature を上げる
  KLENT entropy を上げる
  root noise を増やす
  random_2ply を増やす

シャッフル局面に偏る:
  shuffled_initial weight を下げる
  position buffer weight を上げる
  shuffled pool の重複排除を強める

入玉局面が不足する:
  position buffer の nyugyoku 系 quota を上げる
  nyugyoku / anti_nyugyoku mode からの自己対局を増やす

通常棋力が落ちる:
  initial と normal_midgame を増やす
  shuffled_initial を下げる
  style 系対局比率を下げる
```

---

## 11. 自己対局ログとの接続

### 11.1 SelfPlayGame への保存

`SelfPlayGame` には、開始局面情報を必ず保存する。

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
    steps: list["SelfPlayStep"]
    result: "GameResult"
```

### 11.2 StartPositionInfo.tags

`tags` には source ごとの情報を入れる。

`initial`:

```text
is_initial = True
```

`random_opening`:

```text
is_random_opening = True
random_opening_ply
random_opening_moves_usi
random_opening_rule_id
```

`shuffled_initial`:

```text
is_shuffled_initial = True
shuffle_rule_id
shuffle_seed
rank_8_permutation
rank_9_permutation
king_file_black
king_file_white
rook_file_black
bishop_file_black
```

`position_buffer`:

```text
is_position_buffer = True
source_game_id
source_ply
source_position_id
source_model_version
source_category
source_category_scores
buffer_sample_weight
```

### 11.3 SelfPlayStep との関係

開始局面から1手目の `SelfPlayStep.ply` は 0 とする。

ただし、通常の平手から見た手数に相当する値は `absolute_ply` として分ける。

```text
initial:
  ply = 0
  absolute_ply = 0

random_2ply:
  ply = 0
  absolute_ply = 2

position_buffer:
  ply = 0
  absolute_ply = source absolute ply

shuffled_initial:
  ply = 0
  absolute_ply = 0
```

`shuffled_initial` の `absolute_ply = 0` は、標準平手から0手進んだという意味ではなく、生成局面としての開始 ply である。

### 11.4 moves_left target

`moves_left_target` は、再開後の対局ログ内で計算する。

局面 step の ply を `t`、再開対局の終局 ply を `T` とすると以下である。

```text
moves_left = T - t
moves_left_target = log1p(max(T - t, 0))
```

position buffer 由来局面であっても、source game の終局 ply は使わない。

---

## 12. 学習サンプル生成との関係

### 12.1 source metadata を training sample に残す

training sample には、開始局面 source を tag として残す。

```python
@dataclass
class TrainingSampleMeta:
    start_position_source: str
    start_position_category: str | None
    start_position_phase_id: str
    start_position_tags: dict[str, float | int | bool | str]
```

これにより、source 別に loss / accuracy / calibration を分析できる。

### 12.2 source 別 sample weight

初期実装では、source による sample weight は基本的に 1.0 とする。

ただし、以下は実験候補とする。

```text
shuffled_initial 由来 sample を軽く down-weight する
position_buffer の入玉関連 sample を over-sampling する
normal_midgame を一定 quota で維持する
```

標準で `shuffled_initial` を大きく down-weight しない理由は、自己対局で得た合法局面の policy / Q target としては有用だからである。

ただし、通常棋力が劣化した場合は、以下を検討する。

```text
shuffled_initial の比率を下げる
shuffled_initial 由来の policy_normal_target 重みを下げる
policy_style_target 側へ寄せる
normal_midgame と initial 由来 sample を増やす
```

### 12.3 policy target への影響

開始局面 source は、policy target の意味を変えない。

```text
normal mode の step:
  policy_normal_target

style / nyugyoku / anti_nyugyoku / mutual_nyugyoku / bakuretsu mode の step:
  policy_style_target
```

ただし、source 別に target 分布の品質は違う可能性があるため、ログで分けて評価する。

### 12.4 outcome target への影響

開始局面 source は、終局原因の分類を変えない。

```text
win_normal
win_nyugyoku
draw
loss_nyugyoku
loss_normal
```

すべて、再開後の対局結果から現在手番視点で作る。

---

## 13. 実装上の不変条件

以下は必ず守る。

```text
開始局面 source を必ずログに保存する
source weight は有効 source の間で正規化する
無効 source は sampling しない
生成局面は自己対局開始前に必ず validate する
terminal 局面から通常自己対局を開始しない
宣言勝ち可能局面を通常開始局面にしない
position buffer 由来局面の source game result を training target に使わない
moves_left target は再開後の対局から作る
shuffled_initial は通常棋力評価の開始局面に使わない
```

---

## 14. 検証項目

### 14.1 unit test

最低限、以下をテストする。

```text
initial が正しい SFEN を返す
random_2ply が合法到達局面を返す
shogi81man_v1 が駒数を保存する
shogi81man_v1 が先後180度対称を満たす
shogi81man_v1 が歩段を固定する
shogi81man_v1 が持ち駒空で生成する
shogi81man_v1 が side_to_move=black で生成する
共通品質フィルタが terminal 局面を除外する
共通品質フィルタが宣言勝ち可能局面を除外する
position buffer sampling が category weight を反映する
fallback が initial を返し、fallback_reason を保存する
```

### 14.2 property test

シャッフル局面について、乱数 seed を変えて以下を検証する。

```text
両玉が必ず存在する
先後の駒種数が一致する
持ち駒が空である
合法手生成で例外が出ない
feature encoder で例外が出ない
move label encoder / decoder で例外が出ない
```

### 14.3 統計検証

一定数の開始局面を生成し、以下を確認する。

```text
source 比率が設定 weight に近い
重複率が高すぎない
filter reject rate が高すぎない
legal move count が極端に小さくない
王手中局面の比率が想定内である
宣言勝ち可能局面が通常 source に混じっていない
```

---

## 15. ログ仕様

### 15.1 基本ログ

```text
start_position/source
start_position/category
start_position/phase_id
start_position/ply_offset
start_position/absolute_ply
start_position/fallback_used
start_position/fallback_reason
start_position/validation_flags
```

### 15.2 source 比率ログ

```text
start_position/source_count/initial
start_position/source_count/random_opening
start_position/source_count/shuffled_initial
start_position/source_count/position_buffer

start_position/source_ratio/initial
start_position/source_ratio/random_opening
start_position/source_ratio/shuffled_initial
start_position/source_ratio/position_buffer
```

### 15.3 品質ログ

```text
start_position/filter_reject_rate
start_position/filter_reject_rate/legal
start_position/filter_reject_rate/terminal
start_position/filter_reject_rate/declaration_win
start_position/filter_reject_rate/no_legal_moves
start_position/filter_reject_rate/feature_error
start_position/filter_reject_rate/tactical_error

start_position/legal_moves_mean
start_position/legal_moves_min
start_position/legal_moves_p10
start_position/legal_moves_p50
start_position/legal_moves_p90
start_position/in_check_rate
```

### 15.4 学習指標との接続

source 別に以下を出す。

```text
loss/policy_normal/by_start_source
loss/policy_style/by_start_source
loss/q_normal_result/by_start_source
loss/q_nyugyoku/by_start_source
loss/outcome/by_start_source
loss/moves_left/by_start_source

calibration/outcome/by_start_source
calibration/q_nyugyoku/by_start_source
selfplay/result/by_start_source
selfplay/nyugyoku_win_rate/by_start_source
selfplay/short_game_rate/by_start_source
```

---

## 16. 実験項目

### 16.1 シャッフル比率 ablation

以下を比較する。

```text
A. shuffled_initial なし
B. shuffled_initial 10%
C. shuffled_initial 30%
D. shuffled_initial 40%
```

観測する指標は以下である。

```text
通常勝率
序盤多様性
policy entropy
短手数負け率
入玉関連局面発生率
position buffer のカテゴリ多様性
```

### 16.2 シャッフル制約 ablation

以下を比較する。

```text
A. 9段目のみシャッフル
B. 8段目のみシャッフル
C. 8段目 + 9段目シャッフル
D. 8段目 + 9段目シャッフル、玉 file 固定
E. 8段目 + 9段目シャッフル、玉 file 可変
```

初期候補は C / E である。

### 16.3 random opening 手数 ablation

以下を比較する。

```text
A. random_opening なし
B. 2 ply
C. 4 ply
D. 8 ply
```

4 ply 以上は、短手数負けと悪形増加に注意する。

### 16.4 position buffer quota ablation

以下を比較する。

```text
A. normal_midgame 中心
B. nyugyoku_promising 多め
C. opponent_nyugyoku_risk 多め
D. policy_uncertain / q_close_top_moves 多め
E. balanced
```

観測する指標は以下である。

```text
通常勝率
入玉宣言勝ち率
相手入玉負け率
相入玉局面勝率
q_nyugyoku calibration
outcome_cause calibration
```

### 16.5 source 別 loss weighting

以下を比較する。

```text
A. 全 source weight 1.0
B. shuffled_initial 由来 sample を 0.5 にする
C. shuffled_initial 由来の policy_normal だけ 0.5 にする
D. position_buffer 入玉系 sample を oversampling する
```

通常棋力と入玉指標の trade-off を見る。

### 16.6 phase schedule ablation

以下を比較する。

```text
A. シャッフル率を固定
B. phase で段階的に下げる
C. 指標に応じて自動調整する
```

標準候補は B である。

---

## 17. 失敗パターンと対策

### 17.1 シャッフル局面への過適合

症状:

```text
通常平手からの評価勝率が落ちる
序盤で不自然な駒組みを好む
通常勝ちできる局面で遠回りする
```

対策:

```text
shuffled_initial weight を下げる
initial / normal_midgame を増やす
shuffled_initial 由来の policy_normal loss weight を下げる
通常評価局面 reanalyse を増やす
```

### 17.2 position buffer の偏り

症状:

```text
同じ局面カテゴリばかりから再開する
同じ source game 由来局面が多い
特定 model version の局面に偏る
```

対策:

```text
category quota を導入する
source_game ごとの上限を入れる
position key 重複排除を強める
recency weight を調整する
```

### 17.3 短手数負けの増加

症状:

```text
開始後すぐ詰む
詰み関連の tactical forced move が多すぎる
```

対策:

```text
shuffled_initial の tactical filter を強める
random_opening_max_ply を下げる
詰み探索ログを確認する
短手数負け局面を reanalyse 候補に入れる
```

### 17.4 入玉局面が増えない

症状:

```text
P_win_nyugyoku が高い局面が少ない
宣言勝ちがほとんど出ない
相入玉局面が position buffer に入らない
```

対策:

```text
position buffer の nyugyoku_promising quota を上げる
nyugyoku / mutual_nyugyoku mode の自己対局比率を上げる
λ_nyugyoku を上げる
入玉有望局面の保存条件を緩める
```

---

## 18. 未決事項

### 18.1 shogi81man_v1 の制約

以下は実験で決める。

```text
玉 file を固定するか
8段目の空白マスを含めてシャッフルするか
開始時点の王手を完全禁止するか
浅い tactical filter の範囲をどこまでにするか
```

初期実装では、以下を標準とする。

```text
玉 file 可変
8段目は空白マス込みでシャッフル
開始時点の王手は禁止
相手玉の1手詰め有無を確認
```

### 18.2 シャッフル局面の sample weight

シャッフル局面由来のサンプルを通常の自己対局サンプルと同じ重みで学習するかは未決である。

初期実装では同じ重みにし、通常棋力への影響が出たら source 別重みを導入する。

### 18.3 position buffer の保存容量

position buffer の容量、世代管理、圧縮形式は別途詰める。

最低限、以下は必要である。

```text
position_key による重複排除
category_scores による sampling
source metadata の保存
古い feature / rule version の扱い
```

---

## 19. 参考URL

```text
Shogi816K: A Diverse Collection of Starting Positions for Reinforcement Learning in Shogi
https://cir.nii.ac.jp/crid/1050298124266234240

Game Programming Workshop 2023 program
https://www.gi-ipsj.org/gpw/2023/program.html
```

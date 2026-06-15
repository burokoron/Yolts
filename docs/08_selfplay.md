# docs/08_selfplay.md - 自己対局ループ仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎における **自己対局ループ** の仕様を定義する。

対象は以下である。

```text
自己対局の基本方針
探索モードの割り当て
対局生成フロー
開始局面多様化との接続
終局処理と adjudication
自己対局ログ形式
学習サンプル生成との境界
batched GPU inference
Gumbel MCTS reanalyse 候補の抽出
実装上の不変条件
検証項目
実験項目
```

爆裂玉登太郎では、全局面で重い MCTS を回して自己対局するのではなく、**KLENT-0 + CPU tactical solver** を主学習ループにする。

自己対局ループの主目的は以下である。

```text
通常棋力を維持・改善するための局面を大量に生成する
入玉関連局面を効率よく収集する
policy / action-value / outcome / moves_left の学習 target を作る
探索モード差による棋風データを分離して保存する
Gumbel MCTS reanalyse すべき難所局面を抽出する
```

関連仕様は以下を正とする。

```text
全体方針: 爆裂玉登太郎_設計概要.md
探索・着手選択: docs/03_search.md
ネットワーク本体: docs/04_network.md
ネットワーク出力: docs/05_outputs.md
CPU tactical solver: docs/07_tactical_solver.md
KLENT-first 学習方針: docs/13_klent_training.md
```

---

## 1. 基本方針

### 1.1 KLENT-0 自己対局を標準にする

標準の自己対局は以下で行う。

```text
KLENT-0
+
CPU tactical solver
+
batched GPU inference
```

各手番では、探索側が以下を行う。

```text
1. 合法手を生成する
2. terminal / rule result / 宣言勝ち可能判定を行う
3. CPU tactical solver の pre-check を行う
4. network inference を行う
5. policy prior と Q_mode を作る
6. KLENT 改善方策 pi_klent を作る
7. CPU tactical solver の post-check で方策を補正する
8. sampling または argmax で着手する
9. 自己対局ログを保存する
```

自己対局時は、評価対局よりも局面多様性を優先する。

そのため、通常は `pi_search` から sampling する。

### 1.2 Gumbel MCTS は全局面では使わない

Gumbel MCTS は以下に限定する。

```text
自己対局後の reanalyse
難所局面解析
入玉関連局面の深掘り
KLENT-0 との差分測定
実戦時探索
```

自己対局本体で全局面に Gumbel MCTS を使わない理由は以下である。

```text
自己対局局面数を優先したい
入玉関連局面を大量に集めたい
KLENT-0 の方がログと target が単純である
Gumbel MCTS は効果が高い局面へ集中した方が効率がよい
```

### 1.3 通常棋力を壊さない順序で style データを増やす

自己対局は通常モード中心から始める。

入玉優先モードや相入玉想定モードを最初から大量に混ぜると、`policy_normal_head` が不自然な入玉狙いを吸収し、通常棋力が劣化する可能性がある。

標準の順序は以下である。

```text
1. normal vs normal を中心にする
2. 通常棋力が安定したら style 系を少量混ぜる
3. 入玉関連局面バッファからの再開を増やす
4. anti_nyugyoku / mutual_nyugyoku を加える
5. Gumbel MCTS reanalyse で難所を補正する
```

### 1.4 人間定跡ではなく、ルール生成局面と自己対局局面で多様化する

開始局面の多様化には、人間定跡を標準では使わない。

ただし、学習初期の network が弱い段階では、自己対局で到達した position buffer だけに頼ると、局面の質と多様性が不足しやすい。特に KLENT-0 は root 局面の policy / Q に強く依存するため、序盤が collapse すると、position buffer に入る局面も似たものに偏る。

そのため、初期段階では、以下のような **人間棋譜に依存しないルール生成開始局面** を使う。

```text
平手初期局面
2手ランダム局面
将棋81万風シャッフル局面
自己対局 position buffer 局面
```

ここでの方針は、人間定跡を使わないという原則を維持しつつ、自己対局が未成熟な段階だけルールベースの開始局面多様化で補助することである。

### 1.5 将棋81万風シャッフル開始局面

初期実装では、技巧が採用した「将棋81万」系の考え方を参考に、平手初期局面の駒配置を一定の制約でシャッフルした局面を開始局面として使えるようにする。

標準の候補は以下である。

```text
1. 平手初期局面を基準にする
2. 先手側の8段目の駒を、8段目内でシャッフルする
3. 先手側の9段目の駒を、9段目内でシャッフルする
4. 後手側の1段目と2段目は、先手側配置を180度回転した配置にする
5. 歩の段は初期実装では固定する
6. 手番は平手初期局面と同じく先手番から始める
```

この方式では、先後の対称性を保ったまま、通常の平手初期局面とは異なる序盤を大量に作れる。

ただし、シャッフル局面は通常将棋の自然な序盤分布からは外れる可能性がある。そのため、以下を必ず確認する。

```text
合法局面である
開始時点で terminal ではない
開始時点で宣言勝ち条件を満たさない
feature encoder / move generator / tactical solver で処理できる
```

シャッフル局面は、通常棋力を直接測る評価対局には使わない。用途は、学習初期の自己対局で局面多様性を増やすことに限定する。

詳細なシャッフル制約、生成数、重複排除、局面品質フィルタは `docs/09_start_position_diversity.md` に分ける。

### 1.6 シャッフル率は段階的に下げる

シャッフル開始局面は cold start 補助であり、長期的な主軸にはしない。

標準方針は以下である。

```text
Phase 1: 初期学習直後
  平手初期局面とシャッフル局面を中心にする
  position buffer は未成熟なので少量または無効

Phase 2: 自己対局が安定し始めた段階
  シャッフル局面を減らす
  2手ランダム局面と position buffer 局面を混ぜる

Phase 3: position buffer が十分に成熟した段階
  position buffer を主な多様化手段にする
  シャッフル局面は少量の regularization 用に残すか無効化する

Phase 4: 本番評価に近い学習
  平手初期局面と position buffer を中心にする
  シャッフル局面は ablation 用または低比率に限定する
```

初期値の候補は以下である。

```text
Phase 1:
  initial: 40% 〜 60%
  shuffled_initial: 30% 〜 40%
  random_2ply: 0% 〜 20%
  position_buffer: 0% 〜 10%

Phase 2:
  initial: 40% 〜 50%
  shuffled_initial: 10% 〜 25%
  random_2ply: 5% 〜 15%
  position_buffer: 20% 〜 40%

Phase 3以降:
  initial: 30% 〜 50%
  shuffled_initial: 0% 〜 10%
  random_2ply: 0% 〜 10%
  position_buffer: 40% 〜 60%
```

局面多様性が不足する場合でも、シャッフル率を上げるだけでなく、temperature、KLENT entropy、root noise、position buffer のカテゴリ quota を併せて調整する。

---

## 2. 自己対局全体構成

### 2.1 コンポーネント

自己対局生成は、以下のコンポーネントで構成する。

```text
SelfPlayManager
  対局生成全体を管理する

SelfPlayWorker
  複数対局を進行する

InferenceServer
  GPU batched inference を提供する

SearchEngine
  KLENT-0 / Gumbel MCTS による着手選択を行う

TacticalSolver
  宣言勝ち・相手玉への詰み探索・王手対応を補う

PositionBufferWriter
  保存すべき局面を position buffer へ送る

GameLogWriter
  自己対局ログを保存する

SampleBuilder
  自己対局ログから学習サンプルを作る
```

### 2.2 基本データフロー

```text
model checkpoint
  -> SelfPlayManager
  -> SelfPlayWorker 群
  -> SearchEngine
  -> InferenceServer
  -> GameLogWriter
  -> SampleBuilder
  -> training dataset

SelfPlayWorker
  -> PositionBufferWriter
  -> position buffer
  -> future SelfPlayWorker start positions

GameLogWriter
  -> ReanalyseCandidateQueue
  -> Gumbel MCTS reanalyse
  -> reanalyse dataset
```

### 2.3 学習との境界

自己対局ループは、学習そのものを行わない。

自己対局ループの責務は以下である。

```text
対局を生成する
探索結果を保存する
終局結果を保存する
局面分類情報を保存する
学習サンプル生成に必要な raw 情報を保存する
```

以下は training pipeline 側の責務である。

```text
loss weight の適用
mini-batch sampling
oversampling
λ-return の最終計算
reanalyse target の混合
checkpoint 更新
```

ただし、自己対局ログは training pipeline が必要とする情報を欠落なく含める必要がある。

---

## 3. 設定構造

### 3.1 SelfPlayConfig

自己対局全体の設定は、以下の構造体相当で管理する。

```python
@dataclass
class SelfPlayConfig:
    config_id: str

    # model / feature
    model_version: str
    feature_version: str
    move_label_version: str

    # generation
    num_games: int
    max_game_ply: int = 512
    max_ply_result: str = "draw"
    resign_enabled: bool = False
    adjudication_enabled: bool = True
    random_seed: int | None = None

    # rule
    declaration_rule_id: str = "csa27"

    # players
    player_pairing_schedule: "PlayerPairingSchedule"

    # start position
    start_position_config: "StartPositionConfig"

    # search / tactical
    default_search_type: str = "klent0"
    enable_tactical_solver: bool = True
    enable_declaration_win_check: bool = True

    # exploration
    temperature_schedule: "TemperatureSchedule"
    root_noise_schedule: "RootNoiseSchedule | None" = None

    # inference
    inference_batch_size: int
    inference_max_wait_ms: int
    mixed_precision: str = "fp32"

    # logging
    save_dense_policy: bool = False
    save_sparse_policy: bool = True
    save_q_distribution: bool = True
    save_outcome_probs: bool = True
    save_feature_tensor: bool = False
    save_sfen: bool = True
    save_kifu: bool = True

    # reanalyse candidate
    enable_reanalyse_candidate_extraction: bool = True
```

`config_id` は、自己対局データ、学習データ、評価結果をあとから対応づけるために必ず保存する。

### 3.2 PlayerSpec

先手・後手には、それぞれ探索モードを割り当てる。

```python
@dataclass
class PlayerSpec:
    player_id: str
    mode_id: str
    search_config: "SearchModeConfig"
    target_policy_head: str  # "normal" | "style" | "both"
```

`target_policy_head` は、その対局から得た方策分布をどの policy head の target として使うかを表す。

標準は以下である。

```text
normal mode:
  target_policy_head = "normal"

nyugyoku / anti_nyugyoku / mutual_nyugyoku / bakuretsu:
  target_policy_head = "style"
```

ただし、通常棋力を壊さない範囲で、style 系データの一部を `policy_normal_head` に弱く混ぜる実験は許可する。

### 3.3 PlayerPairing

1局の先手・後手設定は以下で表す。

```python
@dataclass
class PlayerPairing:
    sente: PlayerSpec
    gote: PlayerSpec
    pairing_id: str
    sampling_weight: float
```

`pairing_id` はログに保存する。

例:

```text
normal_vs_normal
nyugyoku_vs_normal
normal_vs_nyugyoku
nyugyoku_vs_nyugyoku
anti_nyugyoku_vs_nyugyoku
nyugyoku_vs_anti_nyugyoku
bakuretsu_vs_normal
normal_vs_bakuretsu
```

### 3.4 StartPositionConfig

開始局面 sampling の詳細仕様は `docs/09_start_position_diversity.md` を正とする。

自己対局側では、開始局面を直接生成するのではなく、`StartPositionSampler` に `StartPositionConfig` を渡し、`StartPositionSample` と `StartPositionInfo` を受け取る。

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

    # source-specific configs
    random_opening_config: "RandomOpeningConfig" = field(default_factory=RandomOpeningConfig)
    shuffle_config: "StartPositionShuffleConfig" = field(default_factory=StartPositionShuffleConfig)
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

有効 source は以下である。

```text
initial
random_opening
shuffled_initial
position_buffer
```

`manual` と `test` は、通常の自己対局 sampling 対象にはしない。

source weight は、有効 source の間で正規化する。無効 source は、weight が正であっても sampling してはならない。

開始局面の phase schedule、2手ランダム、将棋81万風シャッフル、position buffer sampling、品質フィルタ、重複排除、source 別 metadata は `docs/09_start_position_diversity.md` に従う。

自己対局ログでは、開始局面 source を必ず保存する。

```text
start_position.source
start_position.category
start_position.sfen
start_position.ply_offset
start_position.tags
```

シャッフル開始局面と2手ランダム局面は、position buffer が未成熟な初期段階の補助として使う。position buffer が成熟した後は、通常中盤、入玉有望局面、相手入玉リスク局面、相入玉局面、policy / Q の不確実局面、tactical に難しい局面からの再開を主軸にする。

---

## 4. 探索モード割り当て

### 4.1 代表モード

自己対局で使う代表モードは以下である。

```text
normal:
  通常棋力を優先する

nyugyoku:
  自玉の入玉勝ちを積極的に狙う

anti_nyugyoku:
  相手の入玉を強く警戒する

mutual_nyugyoku:
  相手も入玉を狙う前提で相入玉耐性を作る

bakuretsu:
  通常棋力を残しつつ、入玉価値をやや強めに見る
```

各モードの `alpha_style_self`、`alpha_style_opponent_model`、`lambda_nyugyoku` は `docs/03_search.md` と `docs/13_klent_training.md` の探索モード定義に従う。

### 4.2 相手モデル設定

自己対局では、先手・後手それぞれに実際の探索設定を持たせる。

自分の着手時には、自分の `SearchModeConfig` を root 側設定として使う。

探索木の相手側 node では、原則として対戦相手の探索設定を近似する。

例:

```text
先手: nyugyoku
後手: normal

先手の探索:
  root側: nyugyoku
  相手モデル側: normal

後手の探索:
  root側: normal
  相手モデル側: nyugyoku
```

KLENT-0 は root だけを評価するため、初期実装では root 側設定だけでよい。

Gumbel MCTS を使う場合は、相手側 node で `alpha_style_opponent_model` を使う。

### 4.3 phase 別の混合比

自己対局の phase 別混合比の初期候補は以下とする。

```text
Phase 1: normal self-play
  normal_vs_normal: 100%

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

通常棋力が劣化した場合は、`normal_vs_normal` の比率を上げる。

入玉サンプルが不足する場合は、mode 混合比を上げるより先に、position buffer からの入玉関連局面再開を増やす。

---

## 5. 対局生成フロー

### 5.1 1局の流れ

```text
1. pairing を sampling する
2. 開始局面を sampling する
3. 対局状態を初期化する
4. 終局まで各手番で select_move を呼ぶ
5. SearchResult と局面分類情報を SelfPlayStep として保存する
6. 着手を反映する
7. terminal / adjudication / repetition / max_game_ply を判定する
8. GameResult を確定する
9. 全 step に終局結果由来の target 情報を付与できる形で保存する
10. position buffer へ候補局面を送る
11. reanalyse candidate を抽出する
```

### 5.2 疑似コード

```python
def play_selfplay_game(config: SelfPlayConfig, model, rng) -> SelfPlayGame:
    pairing = sample_pairing(config.player_pairing_schedule, rng)
    position, start_info = sample_start_position(config.start_position_config, rng)

    game = SelfPlayGame.new(config=config, pairing=pairing, start_info=start_info)
    repetition_state = RepetitionState.from_position(position)

    while True:
        terminal = check_terminal_or_adjudication(
            position=position,
            repetition_state=repetition_state,
            game_ply=game.current_ply,
            max_game_ply=config.max_game_ply,
        )
        if terminal.is_finished:
            game.result = make_game_result(terminal)
            break

        player = pairing.sente if position.side_to_move == SENTE else pairing.gote
        request = make_search_request(
            position=position,
            player=player,
            game_ply=game.current_ply,
            repetition_state=repetition_state,
            rng=rng,
        )

        search_result = select_move(request)
        step = make_selfplay_step(position, player, search_result, game.current_ply)
        game.steps.append(step)

        position.do(search_result.selected_move)
        repetition_state.update(position)

        collect_position_buffer_candidate(game, position, step)
        collect_reanalyse_candidate(game, step)

    finalize_targets_metadata(game)
    return game
```

### 5.3 着手選択と target 分布

自己対局では、以下を区別して保存する。

```text
p_prior:
  policy_normal / policy_style を混合した prior

pi_klent_pre_tactical:
  KLENT 方策そのもの

pi_search_post_tactical:
  tactical solver 補正後に着手選択へ使った分布

pi_select:
  温度・noise 適用後に sampling へ使った分布
```

学習 target の標準は以下である。

```text
policy target:
  pi_search_post_tactical

解析用:
  p_prior
  pi_klent_pre_tactical
  pi_select
```

sampling 温度や root noise をかけた分布をそのまま policy target にするかは実験項目とする。

初期実装では、`pi_search_post_tactical` を target とし、`pi_select` は着手選択ログとして保存する。

---

## 6. 終局処理

### 6.1 terminal reason

自己対局では、単なる勝敗だけでなく、終局理由を必ず保存する。

標準の `terminal_reason` は以下である。

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

`terminal_reason` は `outcome_cause_target` へ変換できる必要がある。

`illegal_move` は標準の `terminal_reason` に含めない。合法手生成、move 適用、探索側の整合性違反であり、発生した場合はバグとして対局を失敗させる。デバッグログに保存してもよいが、通常の学習 target 生成対象にはしない。

最大手数到達時の標準理由は `max_ply_draw` とする。`max_ply_adjudication` は標準では使わない。

### 6.2 outcome category

終局原因は、学習 target 生成時に以下へ対応づける。

```text
normal:
  checkmate
  resign
  normal_adjudication

nyugyoku:
  declaration_win
  nyugyoku_adjudication

draw:
  repetition_draw
  jishogi_draw
  max_ply_draw
```

標準の自己対局では、最大手数到達時に点数・入玉条件で勝敗を付けない。

理由は、最大手数を短く設定すると、実際には400手目付近で入玉勝ちできる軌道の局面を、早い手数で draw と誤学習する危険があるためである。

### 6.3 入玉宣言勝ちルール

入玉宣言勝ちは、CSA系の27点法を標準とする。

`DeclarationRuleConfig` の標準値は以下である。

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

条件は以下である。

```text
宣言側の手番である
宣言側の玉が敵陣三段目以内に入っている
宣言側の玉に王手がかかっていない
大駒5点、小駒1点で点数を計算する
先手は28点以上、後手は27点以上である
点数対象は、宣言側の持ち駒と、敵陣三段目以内にある宣言側の駒である
点数対象の盤上駒から玉は除く
敵陣三段目以内の宣言側の駒は、玉を除いて10枚以上である
```

宣言勝ち可能局面では、CPU tactical solver の pre-check で `declaration_win` を forced action として返す。自己対局ログでは以下を必ず保存する。

```text
declaration_rule_id
nyugyoku/declaration_win_available
nyugyoku/declaration_win_selected
nyugyoku/own_nyugyoku_score
nyugyoku/opp_nyugyoku_score
nyugyoku/own_pieces_in_enemy_camp_excluding_king
nyugyoku/opp_pieces_in_own_camp_excluding_king
```

宣言勝ちで終局した場合、`outcome_category = "nyugyoku"` とする。

### 6.4 最大手数

自己対局の標準最大手数は **512手** とする。

```python
max_game_ply = 512
max_ply_result = "draw"
```

手数が512手に達し、詰み・宣言勝ち・千日手などで勝敗が確定していない場合、終局理由は以下とする。

```text
terminal_reason = max_ply_draw
winner = None
is_draw = True
outcome_category = draw
```

320手到達で打ち切る設定は標準では使わない。

理由は、320手ルールの大会が存在する一方で、512手ルールの大会も存在するためである。学習時に320手で draw としてしまうと、400手目前後で勝ちに到達する入玉ルートを draw と誤って学習する可能性がある。したがって、自己対局の target 生成では長い側に合わせる。

最大手数到達時に、点数や入玉条件で勝敗を付ける `max_ply_adjudication` は標準無効とする。

### 6.5 resign は標準では無効

初期実装では resign を標準無効にする。

理由は以下である。

```text
q_nyugyoku と outcome_cause の target を終局まで観測したい
入玉逆転や相入玉の長期展開を切り捨てたくない
学習初期の value は calibration が悪く、投了判定に使いにくい
```

導入する場合は、以下を満たす。

```text
resign threshold を config に保存する
resign による終局を normal として扱う
resign なし評価も別途行う
入玉関連局面では resign threshold を厳しめにする
```

### 6.6 adjudication

自己対局スループットのため、adjudication を使ってよい。

ただし、adjudication は target を歪めるため、初期実装では保守的にする。

標準で許可するのは以下である。

```text
宣言勝ち条件成立で nyugyoku win
千日手で draw
512手到達で draw
```

大差かつ入玉無関係な局面での `normal_adjudication` は実験候補に留める。

入玉関連局面では、通常評価値だけによる早期 adjudication を避ける。

## 7. 自己対局ログ形式

### 7.1 SelfPlayGame

1局のログは、以下の構造体相当で保存する。

```python
@dataclass
class SelfPlayGame:
    game_id: str
    config_id: str
    model_version: str
    feature_version: str
    move_label_version: str

    start_position: "StartPositionInfo"
    pairing_id: str
    sente_player: "PlayerSpecLog"
    gote_player: "PlayerSpecLog"

    steps: list["SelfPlayStep"]
    result: "GameResult"

    kifu: str | None
    sfen_initial: str
    sfen_final: str

    random_seed: int | None
    created_at: str
```

`game_id` は一意である。

同じ局面を再現できるように、`random_seed`、開始局面、探索設定、model version を保存する。

### 7.2 StartPositionInfo

```python
@dataclass
class StartPositionInfo:
    source: str  # "initial" | "random_opening" | "shuffled_initial" | "position_buffer" | "manual" | "test"
    category: str | None
    source_game_id: str | None
    source_ply: int | None
    sfen: str
    ply_offset: int
    tags: dict[str, float | int | bool | str]
```

シャッフル開始局面では、`tags` に以下を保存する。

```text
shuffle_rule_id
shuffle_seed
is_shuffled_initial
is_random_opening
random_opening_ply
```

`ply_offset` は、開始局面が平手初期局面から何手進んだ状態かを表す。

`moves_left_target` を作るとき、対局ログ内の ply と実際の通算 ply を混同しないようにする。

### 7.3 PlayerSpecLog

```python
@dataclass
class PlayerSpecLog:
    player_id: str
    mode_id: str
    search_config_id: str
    alpha_style_self: float
    alpha_style_opponent_model: float
    lambda_nyugyoku: float
    klent_alpha_entropy: float
    klent_beta_prior: float
    temperature: float
    target_policy_head: str
```

### 7.4 SelfPlayStep

各手番では、最低限以下を保存する。

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
    selected_move_usi: str | None

    search_type: str
    search_config_id: str
    player_id: str
    mode_id: str

    p_prior: "SparseMoveDistribution"
    pi_klent_pre_tactical: "SparseMoveDistribution"
    pi_search_post_tactical: "SparseMoveDistribution"
    pi_select: "SparseMoveDistribution"

    q_normal_result_pred: "SparseMoveValues | None"
    q_nyugyoku_pred: "SparseMoveValues | None"
    q_total_result_pred: "SparseMoveValues | None"
    q_mode_pred: "SparseMoveValues | None"

    q_normal_result_selected: float | None
    q_nyugyoku_selected: float | None
    q_total_result_selected: float | None
    q_mode_selected: float | None

    outcome_probs: list[float] | None
    v_normal_result: float | None
    v_nyugyoku: float | None
    v_total_result: float | None
    v_mode: float | None
    moves_left_pred: float | None

    tactical_flags: dict
    terminal_flags_before_move: dict
    nyugyoku_features: dict
    uncertainty_features: dict
    reanalyse_priority: float
    reanalyse_tags: list[str]
```

`features_ref` は feature tensor を別ファイルに保存する場合の参照である。

通常は `sfen` と `feature_version` から再生成できるため、feature tensor 自体は保存しなくてよい。

### 7.5 SparseMoveDistribution

方策分布は、ファイルサイズ削減のため sparse 形式を標準にする。

```python
@dataclass
class SparseMoveDistribution:
    moves: list[int]
    probs: list[float]
```

以下を保証する。

```text
moves は合法手だけを含む
probs は moves と同じ順序である
確率和は 1 である
違法手確率は保存しない
```

### 7.6 SparseMoveValues

合法手ごとの Q は、以下の sparse 形式で保存してよい。

```python
@dataclass
class SparseMoveValues:
    moves: list[int]
    values: list[float]
```

`q_normal_result_pred` と `q_nyugyoku_pred` は分けて保存する。

`q_mode_pred` だけを保存してはいけない。

`q_mode_pred` は解析やデバッグのために追加保存してよいが、視点変換や target 生成の正本にはしない。

### 7.7 GameResult

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

    declaration_win_side: int | None
    declaration_rule_id: str | None
    adjudication_side: int | None
    repetition_count: int | None
    max_game_ply: int

    final_sfen: str
    result_flags: dict
```

`winner = None` は draw を表す。

---

## 8. 学習サンプル生成との境界

### 8.1 TrainingSample

自己対局ログから、学習側で以下のサンプルを作る。

```python
@dataclass
class TrainingSample:
    features_ref: str
    sfen: str
    side_to_move: int
    legal_moves: list[int]

    policy_normal_target: "SparseMoveDistribution | None"
    policy_style_target: "SparseMoveDistribution | None"

    q_normal_result_target: "SparseMoveValues | None"
    q_nyugyoku_target: "SparseMoveValues | None"

    outcome_cause_target: int
    moves_left_target: float

    sample_weight: float
    tags: list[str]
```

### 8.2 policy target の割り当て

標準規則は以下である。

```text
normal mode の step:
  policy_normal_target = pi_search_post_tactical
  policy_style_target = None

nyugyoku / anti_nyugyoku / mutual_nyugyoku / bakuretsu mode の step:
  policy_normal_target = None または低重み
  policy_style_target = pi_search_post_tactical
```

`bakuretsu` は通常棋力と入玉棋風の中間である。

初期実装では `policy_style_target` として扱い、通常棋力への悪影響を見ながら `policy_normal_head` への混合を検討する。

### 8.3 q target

自己対局の選択手に対して、終局結果から以下を作る。

| outcome_cause | q_normal_result | q_nyugyoku |
| --- | ---: | ---: |
| `win_normal` | +1 | 0 |
| `win_nyugyoku` | 0 | +1 |
| `draw` | 0 | 0 |
| `loss_nyugyoku` | 0 | -1 |
| `loss_normal` | -1 | 0 |

選択手以外の合法手には、自己対局結果だけでは target がない。

Gumbel MCTS reanalyse により合法手ごとの target が得られた場合のみ、全合法手 target として使う。

### 8.4 moves_left target

局面 step の ply を `t`、終局 ply を `T` とする。

```text
moves_left = T - t
moves_left_target = log1p(max(T - t, 0))
```

開始局面が position buffer 由来の場合でも、ログ内 ply と絶対 ply を分けて保存する。

`moves_left_target` は、その再開対局の終局までの残り手数を表す。

### 8.5 outcome_cause target

`outcome_cause_target` は、各 step の手番側から見た終局結果で作る。

```text
step.side_to_move == winner and outcome_category == normal:
  win_normal

step.side_to_move == winner and outcome_category == nyugyoku:
  win_nyugyoku

winner is None:
  draw

step.side_to_move != winner and outcome_category == nyugyoku:
  loss_nyugyoku

step.side_to_move != winner and outcome_category == normal:
  loss_normal
```

class order は `docs/05_outputs.md` に従う。

---

## 9. position buffer との接続

### 9.1 自己対局中に保存する局面

自己対局中、各 step から position buffer 候補を抽出する。

保存候補は以下である。

```text
通常中盤局面
自玉が敵陣に近い局面
相手玉が自陣に近い局面
宣言点数が増え始めた局面
相入玉になりやすい局面
P_win_nyugyoku が高い局面
P_loss_nyugyoku が高い局面
Q_nyugyoku_selected の絶対値が大きい局面
policy entropy が高い局面
Q 上位手が僅差の局面
短期戦術で mask が発生した局面
```

### 9.2 保存するメタデータ

position buffer へ送る局面には、最低限以下を付与する。

```python
@dataclass
class PositionBufferCandidate:
    sfen: str
    source_game_id: str
    source_ply: int
    absolute_ply: int
    side_to_move: int

    category_scores: dict[str, float]
    nyugyoku_features: dict
    uncertainty_features: dict
    tactical_features: dict

    result_after_restart_source: "GameResult | None"
    model_version: str
    feature_version: str
```

`category_scores` は、1局面が複数カテゴリに属せるようにする。

例:

```text
nyugyoku_promising = 0.8
opponent_nyugyoku_risk = 0.2
policy_uncertain = 0.5
```

### 9.3 再開局面の扱い

position buffer 由来の局面から再開する場合でも、対局結果は再開後の局面から見た自己対局結果として保存する。

以下を混同しない。

```text
source game の最終結果
restart game の最終結果
```

training target に使うのは restart game の最終結果である。

source game の最終結果は、局面分類や分析には使ってよい。

### 9.4 無効局面の除外

再開局面として使う前に、以下を確認する。

```text
合法局面である
terminal ではない
宣言勝ちがすでに成立していない
手番と持ち駒が正しい
feature encoder で変換できる
合法手が1手以上ある
```

---

## 10. reanalyse candidate 抽出

### 10.1 基本方針

自己対局本体では Gumbel MCTS を全局面に使わない。

代わりに、自己対局ログから Gumbel MCTS reanalyse すべき局面を抽出する。

### 10.2 優先度

reanalyse 優先度は以下の要素から作る。

```text
入玉関連局面である
P_win_nyugyoku または P_loss_nyugyoku が高い
Q_nyugyoku の上位手差が大きい、または符号が割れている
policy entropy が高い
KL(pi_klent || p_prior) が大きい
Q_mode 上位手が僅差である
tactical solver が多数の手を mask した
終局直前で value が外れていた
敗着候補である
通常勝ちと入玉勝ちの選択が割れる
```

### 10.3 ReanalyseCandidate

```python
@dataclass
class ReanalyseCandidate:
    sfen: str
    game_id: str
    ply: int
    side_to_move: int

    priority: float
    tags: list[str]

    legal_moves: list[int]
    p_prior: "SparseMoveDistribution"
    pi_klent_pre_tactical: "SparseMoveDistribution"
    pi_search_post_tactical: "SparseMoveDistribution"

    q_normal_result_pred: "SparseMoveValues | None"
    q_nyugyoku_pred: "SparseMoveValues | None"
    outcome_probs: list[float] | None

    search_config_id: str
    model_version: str
    feature_version: str
```

reanalyse の実行結果は別データセットとして保存し、自己対局ログそのものを書き換えない。

---

## 11. batched GPU inference

### 11.1 方針

自己対局では、複数 worker からの局面をまとめて GPU 推論する。

```text
SelfPlayWorker
  -> InferenceRequestQueue
  -> InferenceServer
  -> batch model.forward
  -> InferenceResponseQueue
  -> SelfPlayWorker
```

1局面ずつ同期推論する実装に固定しない。

初期実装は同期 API から始めてもよいが、内部境界は batched inference に置き換えられる形にする。

### 11.2 InferenceRequest

```python
@dataclass
class InferenceRequest:
    request_id: str
    worker_id: int
    game_id: str
    ply: int
    sfen: str
    features: "Tensor | np.ndarray"
    legal_moves: list[int]
    model_version: str
    feature_version: str
```

### 11.3 InferenceResponse

```python
@dataclass
class InferenceResponse:
    request_id: str
    network_output: "NetworkOutput"
    elapsed_ms: float
    batch_size: int
```

探索モード依存の `p_prior`、`Q_mode`、`pi_klent` は worker 側または SearchEngine 側で計算する。

raw `NetworkOutput` は探索モードに依存しないため、NN cache を使う場合も raw output を cache する。

### 11.4 mixed precision

初期実装では fp32 を正とする。

mixed precision を使う場合は、以下を検証してから有効化する。

```text
policy softmax が NaN にならない
KLENT 方策の確率和が1になる
q raw が極端に発散しない
outcome logits が極端化しない
fp32 と比較して着手分布が大きく崩れない
```

---

## 12. 探索 noise と sampling

### 12.1 自己対局時の探索性

自己対局では、局面多様性のために sampling を使う。

```text
pi_select(a) ∝ pi_search(a)^(1 / temperature)
```

`temperature = 0` は argmax として扱う。

### 12.2 temperature schedule

初期候補は以下である。

```text
序盤:
  temperature = 1.0 〜 1.5

中盤:
  temperature = 0.8 〜 1.0

終盤:
  temperature = 0.3 〜 0.8

宣言勝ち可能・詰み確定・唯一合法応手:
  forced move
```

実装上は、以下のような schedule で持つ。

```python
@dataclass
class TemperatureSchedule:
    opening_ply_threshold: int
    endgame_ply_threshold: int
    opening_temperature: float
    middlegame_temperature: float
    endgame_temperature: float
```

入玉は長期化しやすいため、単純に ply が進んだら常に低温にするのは危険である。

入玉関連局面では、policy collapse が起きない範囲で温度を維持してよい。

### 12.3 root noise

自己対局では、root prior に Dirichlet noise を混ぜてよい。

ただし、KLENT-0 は MCTS よりも root prior の影響が直接的であるため、noise は小さく始める。

```text
root_noise_frac = 0.0 〜 0.15
```

初期実装では root noise 無効から始め、局面多様性が不足した場合に導入する。

評価対局、reanalyse、実戦時探索では root noise を無効にする。

---

## 13. データ分割と重複管理

### 13.1 train / validation split

自己対局データは、game 単位で train / validation に分ける。

同一 game の局面が train と validation に混ざってはいけない。

標準は以下である。

```text
train: 95%
validation: 5%
```

評価用の固定局面セットは、自己対局ログとは別に管理する。

### 13.2 model version 単位の管理

自己対局データには、生成に使った model version を必ず保存する。

学習時には、古い model のデータと新しい model のデータを混ぜてよいが、sampling weight を制御する。

```text
新しい model の自己対局データ:
  高めの重み

古い model の自己対局データ:
  多様性目的で低めの重み

reanalyse 済みデータ:
  target 品質が高い場合は高めの重み
```

### 13.3 重複局面

同一局面が何度も現れることは許容する。

ただし、局面多様性の低下を検出するため、以下をログする。

```text
unique_position_count
position_duplicate_rate
opening_sequence_duplicate_rate
position_buffer_source_duplicate_rate
```

重複が多い場合は以下で対処する。

```text
temperature を上げる
root noise を導入する
2手ランダム局面と将棋81万風シャッフル局面を cold start 補助として増やす
position buffer からの再開率と category quota を調整する
pairing の style 比率を調整する
KLENT entropy regularization を上げる
```

---

## 14. ログ仕様

### 14.1 対局単位ログ

対局単位で以下を記録する。

```text
selfplay/game_id
selfplay/config_id
selfplay/model_version
selfplay/feature_version
selfplay/pairing_id
selfplay/start_position_source
selfplay/start_position_category
selfplay/shuffle_rule_id
selfplay/is_shuffled_initial
selfplay/is_random_opening
selfplay/terminal_reason
selfplay/winner
selfplay/terminal_ply
selfplay/max_game_ply
selfplay/is_draw
selfplay/is_nyugyoku_win
selfplay/declaration_rule_id
selfplay/num_steps
selfplay/random_seed
```

### 14.2 生成速度ログ

```text
selfplay/games_per_sec
selfplay/positions_per_sec
selfplay/avg_game_ply
selfplay/p50_game_ply
selfplay/p95_game_ply
selfplay/worker_utilization
inference/batch_size_avg
inference/batch_size_p95
inference/queue_wait_ms_avg
inference/forward_ms_avg
inference/positions_per_sec
```

### 14.3 探索・方策ログ

既存の探索ログに加えて、自己対局では以下を集計する。

```text
policy/entropy_prior_avg
policy/entropy_klent_avg
policy/entropy_search_avg
policy/kl_klent_to_prior_avg
policy/selected_move_rank_prior_avg
policy/selected_move_rank_klent_avg

q/q_normal_result_selected_avg
q/q_nyugyoku_selected_avg
q/q_mode_selected_avg
q/q_mode_gap_top1_top2_avg

outcome/P_win_normal_avg
outcome/P_win_nyugyoku_avg
outcome/P_loss_nyugyoku_avg

moves_left/pred_avg
```

### 14.4 入玉関連ログ

```text
nyugyoku/declaration_rule_id
nyugyoku/declaration_win_count
nyugyoku/declaration_win_rate
nyugyoku/nyugyoku_adjudication_count
nyugyoku/max_ply_draw_count
nyugyoku/own_king_in_enemy_camp_count
nyugyoku/opp_king_in_own_camp_count
nyugyoku/nyugyoku_promising_position_count
nyugyoku/opponent_nyugyoku_risk_position_count
nyugyoku/mutual_nyugyoku_like_position_count
nyugyoku/avg_own_score_remaining
nyugyoku/avg_opp_score_remaining
```

### 14.5 tactical ログ

```text
tactical/forced_move_rate
tactical/declaration_win_forced_count
tactical/mate_in_one_forced_count
tactical/mask_removed_all_moves_count
tactical/error_count
tactical/elapsed_ms_avg
tactical/elapsed_ms_p95
```

### 14.6 reanalyse candidate ログ

```text
reanalyse/candidate_count
reanalyse/candidate_rate
reanalyse/nyugyoku_candidate_count
reanalyse/policy_uncertain_candidate_count
reanalyse/q_close_candidate_count
reanalyse/tactical_candidate_count
reanalyse/avg_priority
```

---

## 15. 実装上の不変条件

以下は必ず守る。

```text
自己対局の各局は model_version / feature_version / move_label_version を保存する
先手・後手それぞれの探索設定を保存する
policy_style は現在手番側の style policy として扱う
自己対局では root 側の探索設定を各プレイヤーごとに切り替える
合法手は探索側で生成する
違法手確率は保存しない、または必ず0にする
terminal node では network inference を呼ばない
宣言勝ち可能判定は network output に依存しない
CPU tactical solver の forced move は KLENT 方策より優先する
pi_klent_pre_tactical と pi_search_post_tactical を区別する
policy target の標準は pi_search_post_tactical とする
Q_normal_result と Q_nyugyoku は分けて保存する
Q_mode だけを保存して target 生成の正本にしない
outcome_cause class order は docs/05_outputs.md に従う
terminal_reason は outcome_cause_target へ変換可能である
position buffer 由来の対局では source game result と restart game result を混同しない
シャッフル開始局面は start_position.source と shuffle_rule_id を保存する
入玉宣言勝ちルールは csa27 を標準とする
最大手数は 512 手を標準とし、到達時は draw とする
train / validation split は game 単位で行う
reanalyse 結果で元の自己対局ログを破壊的に上書きしない
```

---

## 16. 検証項目

### 16.1 1局生成 test

以下を確認する。

```text
初期局面から1局を生成できる
各 step に合法手が存在する
selected_move が legal_moves に含まれる
終局時に GameResult が作られる
terminal_ply と steps 長が整合する
kifu / sfen が保存される
```

### 16.2 探索設定 test

以下を確認する。

```text
先手・後手に異なる SearchModeConfig を割り当てられる
手番ごとに正しい player の設定が使われる
pairing_id がログに保存される
normal / nyugyoku / anti_nyugyoku / bakuretsu の設定が区別される
```

### 16.3 方策分布 test

以下を確認する。

```text
p_prior の確率和が1である
pi_klent_pre_tactical の確率和が1である
pi_search_post_tactical の確率和が1である
pi_select の確率和が1である
全分布の moves が legal_moves の部分集合である
forced move 時は該当手の確率が1になる
```

### 16.4 target 変換 test

以下を確認する。

```text
terminal_reason から outcome_category を作れる
各 step の side_to_move 視点で outcome_cause_target を作れる
csa27 の入玉宣言勝ち条件が判定できる
512手到達時に max_ply_draw になる
q_normal_result_target が定義通りである
q_nyugyoku_target が定義通りである
moves_left_target が非負である
policy target の割り当てが mode_id と一致する
```

### 16.5 position buffer 接続 test

以下を確認する。

```text
保存候補局面が合法局面である
terminal 局面を再開候補にしない
シャッフル開始局面が合法局面として生成される
シャッフル開始局面が開始時点で宣言勝ち可能になっていない
カテゴリスコアが保存される
position buffer 由来局面から対局を再開できる
source game result と restart game result が分離される
```

### 16.6 reanalyse candidate test

以下を確認する。

```text
入玉関連局面に reanalyse tag が付く
policy entropy が高い局面に tag が付く
Q 上位手が僅差の局面に tag が付く
priority が非負である
candidate が元ログを参照できる
```

### 16.7 reproducibility test

以下を確認する。

```text
同じ model / config / seed で同じ対局を再現できる
random seed がログに保存される
並列 worker 数を変えた場合の再現性要件が明記されている
```

並列実行では完全再現が難しい場合がある。

その場合でも、worker 単位 seed、game_id、pairing、開始局面、sampling に使った乱数系列を保存し、1局単位では再現可能にする。

---

## 17. 実験項目

### 17.1 pairing ratio 実験

以下を比較する。

```text
normal_vs_normal 100%
normal中心 + nyugyoku少量
normal中心 + bakuretsu少量
style比率高め
anti_nyugyoku / mutual_nyugyoku 追加
```

観測する指標は以下である。

```text
通常勝率
入玉宣言勝ち率
相手入玉負け率
通常勝ちできる局面で無理に入玉しない率
policy_normal の劣化
policy_style の有効性
```

### 17.2 start position mix 実験

以下を比較する。

```text
初期局面のみ
初期局面 + 2手ランダム局面
初期局面 + 将棋81万風シャッフル局面
初期局面 + シャッフル局面 + 2手ランダム局面
初期局面 + 通常中盤バッファ
初期局面 + 入玉有望局面バッファ
初期局面 + 相手入玉リスク局面バッファ
シャッフル局面を段階的に減らして position buffer へ移行
カテゴリ混合
```

観測する指標は以下である。

```text
局面多様性
序盤重複率
シャッフル局面由来サンプルの通常勝率への影響
position buffer 成熟前の短手数負け率
入玉関連局面到達率
通常勝率
q_nyugyoku calibration
P_win_nyugyoku / P_loss_nyugyoku calibration
```

### 17.3 sampling temperature 実験

以下を比較する。

```text
temperature 固定
temperature ply schedule
temperature entropy adaptive
temperature 入玉局面別 schedule
```

観測する指標は以下である。

```text
policy collapse
短手数負け率
局面多様性
通常勝率
入玉関連局面到達率
```

### 17.4 root noise 実験

以下を比較する。

```text
root_noise_frac = 0.0
root_noise_frac = 0.05
root_noise_frac = 0.10
root_noise_frac = 0.15
```

観測する指標は以下である。

```text
局面多様性
序盤重複率
通常勝率
短期戦術ミス
KLENT 方策の安定性
```

### 17.5 tactical target 反映実験

以下を比較する。

```text
pre_tactical_pi を policy target にする
post_tactical_pi を policy target にする
pre と post を両方保存し一定割合で混ぜる
```

観測する指標は以下である。

```text
詰み逃し率
KLENT-0 only の戦術改善
full tactical solver 有効時の勝率
policy calibration
```

### 17.6 reanalyse candidate selection 実験

以下を比較する。

```text
入玉関連優先
policy entropy 優先
Q 上位手僅差優先
敗着候補優先
混合 priority
```

観測する指標は以下である。

```text
reanalyse 1局面あたりの改善量
policy target 品質
q target 品質
通常勝率への寄与
入玉指標への寄与
```

---

## 18. 失敗パターンと対策

### 18.1 通常棋力の劣化

症状:

```text
通常勝率が下がる
寄せを放棄して入玉を狙う
序中盤で不自然な玉上がりが増える
policy_normal が style 的になる
```

対策:

```text
normal_vs_normal 比率を上げる
style 系 step を policy_normal_target に入れない
lambda_nyugyoku を下げる
alpha_style_self を下げる
通常勝ち局面の reanalyse を増やす
```

### 18.2 入玉サンプル不足

症状:

```text
win_nyugyoku / loss_nyugyoku がほとんど出ない
q_nyugyoku が学習されない
P_win_nyugyoku / P_loss_nyugyoku の calibration が悪い
```

対策:

```text
入玉関連局面バッファからの再開率を上げる
nyugyoku / anti_nyugyoku pairing を増やす
入玉関連サンプルを oversampling する
Gumbel MCTS reanalyse を入玉局面に集中する
```

### 18.3 policy collapse

症状:

```text
同じ序盤ばかり指す
pi_klent が少数手に集中する
position duplicate rate が上がる
```

対策:

```text
temperature を上げる
klent_alpha_entropy を上げる
klent_beta_prior を上げる
root noise を導入する
2手ランダム局面と将棋81万風シャッフル局面を cold start 補助として増やす
position buffer からの再開率と category quota を調整する
```

### 18.4 tactical solver 依存

症状:

```text
full tactical solver では勝つが KLENT-0 only では詰みを逃す
policy が詰み手を上位に置かない
Q が詰み手を高く評価しない
```

対策:

```text
post_tactical_pi を policy target に使う
詰み関連局面を oversampling する
確定 tactical target を q target に限定的に入れる
KLENT-0 only 評価を継続する
```

### 18.5 シャッフル開始局面への過適応

症状:

```text
平手初期局面からの通常勝率が上がらない
シャッフル局面では多様性があるが、自然な序盤で弱い
通常の駒組み感覚が崩れる
policy_normal が不自然な初期配置に引っ張られる
```

対策:

```text
シャッフル局面の比率を下げる
シャッフル局面を policy_style_target または低重み target として扱う
平手初期局面と通常中盤バッファの比率を上げる
評価対局は必ず平手初期局面または標準評価局面で行う
```

### 18.6 position buffer の偏り

症状:

```text
似た局面からばかり再開する
特定の入玉パターンだけ増える
通常局面の分布が痩せる
```

対策:

```text
カテゴリ別 quota を入れる
同一局面・類似局面の重複率をログする
古い局面を decay する
normal_midgame を一定割合維持する
```

---

## 19. 未決事項

### 19.1 pairing ratio の初期値

normal をどれだけ維持しながら style 系を混ぜるかは、通常勝率の劣化を見て決める。

初期実装では `normal_vs_normal` を強く優先する。

### 19.2 position buffer のカテゴリ閾値

`nyugyoku_promising`、`opponent_nyugyoku_risk`、`mutual_nyugyoku_like`、`policy_uncertain`、`q_close_top_moves`、`tactical_uncertain` などの category は `docs/09_start_position_diversity.md` を正とする。

このドキュメントでは、自己対局ログ・position buffer・reanalyse candidate へ category score を受け渡しできることだけを固定する。

### 19.3 シャッフル開始局面の調整余地

将棋81万風シャッフルの標準制約は `docs/09_start_position_diversity.md` を正とする。

自己対局側では、以下を不変条件として扱う。

```text
人間定跡ではなくルール生成開始局面として扱う
先手側8段目・9段目の段内シャッフルを基本候補にする
後手側は先手側の180度回転対称にする
歩段は初期実装では固定する
合法性・非terminal・宣言勝ち不可を検証する
通常棋力評価には使わない
シャッフル率は phase schedule と position buffer の成熟に応じて下げる
```

### 19.4 resign / adjudication の採用範囲

resign は初期実装では無効とする。

adjudication は最大手数到達と宣言勝ち条件を中心にし、通常評価値による早期打ち切りは慎重に導入する。

### 19.5 pi_select を policy target に使うか

探索性を強めた `pi_select` をそのまま target にすると、policy がノイズを学習する可能性がある。

標準では `pi_search_post_tactical` を target にする。

ただし、過度に deterministic な target で policy collapse する場合は、温度付き target を比較する。

### 19.6 並列実行時の完全再現性

並列 worker と batched inference の順序により、完全な全体再現は難しい可能性がある。

最低限、1局単位で再現できる情報を保存する。

---

## 20. 暫定結論

爆裂玉登太郎の自己対局ループは、以下を標準とする。

```text
KLENT-0 + CPU tactical solver による高速自己対局
normal vs normal 中心の段階的開始
学習初期の将棋81万風シャッフル開始局面
シャッフル率を段階的に下げて position buffer へ移行
style 系 pairing の少量混合
自己対局局面バッファからの再開
入玉関連局面の重点収集
Gumbel MCTS reanalyse 候補の抽出
```

自己対局ログでは、以下を必ず分けて保存する。

```text
p_prior
pi_klent_pre_tactical
pi_search_post_tactical
pi_select
Q_normal_result
Q_nyugyoku
outcome_probs
moves_left_pred
terminal_reason
max_game_ply
search_config_id
pairing_id
start_position.source
shuffle_rule_id
```

特に重要な不変条件は以下である。

```text
通常棋力を壊さないため、normal データを主軸にする
入玉棋風データは policy_style_head を中心に吸収する
Q_mode だけを保存して視点情報を失わない
終局原因は normal / nyugyoku / draw に分類できる形で保存する
入玉宣言勝ちは csa27 を標準とする
最大手数は 512 手とし、到達時は draw とする
position buffer 由来の再開対局は、restart game として独立に target を作る
```

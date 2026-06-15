# docs/15_training_pipeline.md - 学習パイプライン仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎における **学習パイプライン全体** の実行仕様を定義する。

対象は以下である。

```text
学習 run の全体構成
自己対局生成との接続
教師あり初期化との接続
Gumbel MCTS reanalyse との接続
dataset / shard / registry の管理
mini-batch sampling
trainer / optimizer / scheduler
checkpoint 保存・昇格
evaluation gate との接続
phase 遷移
ログ・可観測性
再現性
実装上の不変条件
検証項目
実験項目
```

このドキュメントは、個々の target、loss、探索アルゴリズム、評価指標の正本ではない。

個別仕様は以下を正とする。

```text
全体方針: 爆裂玉登太郎_設計概要.md
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
```

このドキュメントでは、それらをどの順序で実行し、どの成果物を保存し、どの gate を通して checkpoint を採用するかを扱う。

---

## 1. 基本方針

### 1.1 学習パイプラインの役割

学習パイプラインは、以下を統合する実行単位である。

```text
データ生成
  教師探索
  KLENT-0 自己対局
  position buffer 由来再開対局
  Gumbel MCTS reanalyse

データ変換
  raw log 読み込み
  TrainingSample 生成
  reanalyse target merge
  target validation
  dataset shard 保存

学習
  mini-batch sampling
  feature encode
  network forward
  head 別 loss
  optimizer step
  checkpoint 保存

評価
  offline validation
  smoke evaluation
  formal evaluation
  checkpoint gate
  champion / optimized config 更新
```

自己対局 loop、target 生成、評価の各詳細は別ドキュメントに置く。training pipeline は、それらを壊さないように接続し、run の再現性と可観測性を保証する。

### 1.2 自己対局と学習を分離する

自己対局 loop は学習そのものを行わない。

自己対局の責務は以下である。

```text
対局を生成する
探索結果を保存する
終局結果を保存する
局面分類情報を保存する
reanalyse candidate を抽出する
position buffer に局面を送る
```

training pipeline の責務は以下である。

```text
自己対局ログを読む
TrainingSample を作る
λ-return を計算する
reanalyse target を混合する
mini-batch sampling を行う
loss weight を適用する
checkpoint を更新する
評価 gate を実行する
```

この分離により、自己対局生成・target 生成・trainer を個別に検証できるようにする。

### 1.3 target の正本を増やさない

training pipeline は、新しい target 定義を勝手に作らない。

学習 target の正本は以下である。

```text
policy_normal_target
policy_style_target
q_normal_result_target
q_nyugyoku_target
outcome_cause_target
moves_left_target
```

`Q_total_result`、`Q_mode`、`V_total_result`、`V_mode` は派生値であり、target の正本にはしない。

特に、`Q_mode` だけを保存して別の `λ_nyugyoku` で再利用してはならない。

### 1.4 通常棋力を壊さない順序を守る

学習パイプラインは、通常棋力を先に安定させてから入玉関連データを増やす。

標準順序は以下である。

```text
1. 教師あり初期化で最低限の policy / Q / outcome を作る
2. KLENT-0 normal self-play を安定させる
3. position buffer と開始局面多様化を導入する
4. style 系 mode を少量から増やす
5. 入玉関連局面を厚くする
6. Gumbel MCTS reanalyse で難所を補正する
7. 評価 gate と ablation で scaling する
```

`policy_normal_head` に style 系 target を大量に混ぜてはならない。style 系 target は原則として `policy_style_head` に流し、通常棋力を壊さない範囲でのみ弱く混ぜる。

### 1.5 evaluation gate を学習 loop に組み込む

checkpoint は、loss が下がっただけでは採用しない。

少なくとも以下を分けて判断する。

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

通常モード 90% 目標と、勝率 80%以上での入玉宣言勝ち最大化は別々に管理する。

### 1.6 失敗対局や異常 sample を学習に混ぜない

以下は通常の終局結果ではなく、実装または実行の異常である。

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
```

これらを `loss_normal`、`draw`、`max_ply_draw` などに変換してはならない。

該当 game / sample は quarantine へ送る。原因を修正するまで training dataset には入れない。

---

## 2. 全体データフロー

### 2.1 標準フロー

標準の学習フローは以下である。

```text
teacher data / seed checkpoint
  -> supervised dataset
  -> Phase 0 training
  -> seed checkpoint
  -> KLENT-0 self-play
  -> SelfPlayGame logs
  -> TrainingSample dataset
  -> trainer
  -> candidate checkpoint
  -> evaluation
  -> promoted checkpoint
  -> next self-play
```

入玉関連局面と難所局面が増えた後は、以下を追加する。

```text
SelfPlayGame logs
  -> ReanalyseCandidateQueue
  -> Gumbel MCTS reanalyse
  -> reanalyse dataset
  -> TrainingSample merge
  -> trainer
```

position buffer は、自己対局生成と開始局面 sampling に使う。

```text
SelfPlayGame logs
  -> PositionBufferWriter
  -> position buffer
  -> StartPositionSampler
  -> future self-play
```

### 2.2 run 単位

1つの training run は、以下の粒度で管理する。

```text
training_run_id:
  1つの設定で、複数 checkpoint update と評価を含む実行単位

data_generation_run_id:
  特定 checkpoint と selfplay config で生成した自己対局 batch

dataset_id:
  training sample shard 群と metadata の単位

trainer_run_id:
  特定 dataset mix と trainer config で optimizer step を進めた単位

eval_run_id:
  特定 checkpoint / eval mode / eval suite で評価した単位
```

run id は、ログ、checkpoint、dataset metadata、evaluation result から相互に辿れるようにする。

### 2.3 checkpoint を中心にした循環

学習は checkpoint を中心に循環する。

```text
current_generation_checkpoint
  -> self-play / reanalyse
  -> dataset update
  -> trainer update
  -> candidate_checkpoint
  -> evaluation
  -> promoted_checkpoint
  -> next current_generation_checkpoint
```

ただし、常に最新 checkpoint を自己対局に使うとは限らない。

標準では以下を分ける。

```text
latest_checkpoint:
  training step の最新出力

candidate_checkpoint:
  evaluation 対象

selfplay_checkpoint:
  次の自己対局生成に使う checkpoint

normal_champion_checkpoint:
  通常モード gate を満たした最良 checkpoint

nyugyoku_optimized_config:
  共通 checkpoint に適用する、勝率制約下で入玉宣言勝ち率が高い探索設定
```

---

## 3. コンポーネント

### 3.1 TrainingPipelineManager

`TrainingPipelineManager` は、学習 run 全体を管理する。

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

`TrainingPipelineManager` は、target の中身や探索計算そのものを直接実装しない。各専門コンポーネントへ設定を渡し、成果物の整合性を検査する。

### 3.2 DataGenerationScheduler

`DataGenerationScheduler` は、どの checkpoint でどのデータを生成するかを決める。

扱う source は以下である。

```text
teacher_search
teacher_move
selfplay
reanalyse
debug_test
```

通常 training では、`debug_test` は dataset に混ぜない。

### 3.3 DatasetBuilder

`DatasetBuilder` は、raw log と reanalyse 結果から dataset shard を作る。

内部コンポーネントは以下である。

```text
GameLogReader
SampleBuilder
LambdaReturnBuilder
ReanalyseTargetMerger
TargetValidator
DatasetWriter
DatasetRegistryUpdater
```

`SampleBuilder` と target の詳細は `docs/10_training_targets.md` に従う。

### 3.4 DatasetSampler

`DatasetSampler` は、dataset shard から mini-batch を作る。

責務は以下である。

```text
source 別 sampling
mode 別 sampling
outcome 別 oversampling
start_position.source 別 sampling
reanalyse sample の混合
同一 game / 同一局面の偏り抑制
legal_move_mask の作成
head 別 target mask の作成
```

`DatasetSampler` は target の値を変更しない。sample / target weight の適用は設定に基づいて行う。

### 3.5 Trainer

`Trainer` は network を更新する。

責務は以下である。

```text
feature encode
network forward
loss 計算
backward
optimizer step
learning rate schedule
gradient clipping
mixed precision
EMA 更新
checkpoint 保存
training metric 保存
```

`Trainer` は合法手 mask を loss 側で必ず適用する。network module 内に合法手 mask、policy softmax、Q の `tanh` 変換を押し込まない。

### 3.6 CheckpointManager

`CheckpointManager` は checkpoint と関連 metadata を保存する。

責務は以下である。

```text
model state 保存
optimizer state 保存
scheduler state 保存
EMA state 保存
training config 保存
version 保存
parent checkpoint 保存
dataset lineage 保存
evaluation lineage 保存
promotion status 保存
```

checkpoint 互換性に関わる version が一致しない場合は、load を拒否する。

### 3.7 EvaluationGateRunner

`EvaluationGateRunner` は、checkpoint gate を実行する。

評価の詳細は `docs/14_evaluation.md` を正とする。

training pipeline 側では、少なくとも以下を呼び出す。

```text
correctness fixture
offline validation
smoke self-play
tiered game evaluation
formal 1000-game evaluation
speed evaluation
```

正式 gate は、`balanced_positions_2025_24ply_500x2_v1` の1000局評価を正本とする。

---

## 4. 設定構造

### 4.1 TrainingPipelineConfig

training pipeline 全体の設定は、以下の構造体相当で管理する。

```python
@dataclass
class TrainingPipelineConfig:
    config_id: str
    run_name: str

    # phase
    phase_id: str
    phase_schedule_id: str

    # versions
    rule_version: str
    feature_version: str
    move_label_version: str
    network_output_version: str
    target_version: str
    selfplay_log_version: str
    dataset_version: str
    checkpoint_version: str

    # checkpoints
    initial_checkpoint_id: str | None
    teacher_checkpoint_id: str | None
    selfplay_checkpoint_policy: str  # "latest_promoted" | "latest_candidate" | "normal_champion"

    # data generation
    selfplay_config_id: str | None
    start_position_schedule_id: str | None
    pairing_schedule_id: str | None
    reanalyse_config_id: str | None

    # dataset
    dataset_build_config_id: str
    dataset_mix_config_id: str
    validation_split_config_id: str

    # training
    trainer_config_id: str
    optimizer_config_id: str
    loss_config_id: str

    # evaluation
    eval_schedule_id: str
    gate_config_id: str

    # runtime
    output_root: str
    random_seed: int
    deterministic: bool = False
    resume_from_run_id: str | None = None
```

`config_id` は、すべての生成物に保存する。

### 4.2 PipelinePhase

学習 phase は、以下の enum 相当で管理する。

```python
class PipelinePhase:
    SUPERVISED_INIT = "supervised_init"
    KLENT0_NORMAL = "klent0_normal"
    POSITION_BUFFER_GROWTH = "position_buffer_growth"
    NYUGYOKU_AUGMENTATION = "nyugyoku_augmentation"
    REANALYSE = "reanalyse"
    EVAL_AND_SCALING = "eval_and_scaling"
```

これは `docs/13_klent_training.md` の Phase 0〜4、および `docs/12_roadmap.md` の milestone と対応づけるための実行管理用 ID である。

### 4.3 DatasetMixConfig

training dataset の混合比は、以下の構造体相当で管理する。

```python
@dataclass
class DatasetMixConfig:
    config_id: str

    source_weights: dict[str, float]
    mode_weights: dict[str, float]
    outcome_weights: dict[str, float]
    start_position_source_weights: dict[str, float]
    tag_weights: dict[str, float]

    max_samples_per_game_per_batch: int | None = 2
    max_samples_per_start_position_per_batch: int | None = 2
    max_duplicate_position_per_batch: int | None = 1

    min_reanalyse_quality: float = 0.0
    prefer_recent_selfplay: bool = True
    recent_checkpoint_half_life: int | None = None

    oversample_nyugyoku_outcomes: bool = True
    oversample_opponent_nyugyoku_risk: bool = True
```

weight は、dataset に入れる比率ではなく mini-batch sampling の比率である。

保存済み dataset の raw count は変更しない。

### 4.4 TrainerConfig

trainer 設定は、以下の構造体相当で管理する。

```python
@dataclass
class TrainerConfig:
    config_id: str

    batch_size: int
    num_optimizer_steps: int
    gradient_accumulation_steps: int = 1

    mixed_precision: str = "bf16"  # "fp32" | "fp16" | "bf16"
    compile_model: bool = False

    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)

    lr_schedule: str = "cosine"  # "constant" | "cosine" | "linear_warmup_cosine"
    warmup_steps: int = 1000
    min_learning_rate_ratio: float = 0.1

    grad_clip_norm: float | None = 5.0
    use_ema: bool = True
    ema_decay: float = 0.999

    checkpoint_interval_steps: int = 10000
    validation_interval_steps: int = 1000
    log_interval_steps: int = 100

    num_dataloader_workers: int = 4
    prefetch_batches: int = 4
```

初期値は候補であり、モデル規模と GPU に応じて調整する。

### 4.5 LossConfig

loss weight は `docs/10_training_targets.md` の head 定義に従い、pipeline 側では実行設定として管理する。

```python
@dataclass
class LossConfig:
    config_id: str

    w_policy_normal: float = 1.0
    w_policy_style: float = 0.2
    w_q_normal_result: float = 1.0
    w_q_nyugyoku: float = 0.3
    w_outcome: float = 1.0
    w_moves_left: float = 0.1

    policy_label_smoothing: float = 0.0
    q_loss: str = "smooth_l1"
    outcome_loss: str = "cross_entropy"
    moves_left_loss: str = "smooth_l1"

    ignore_missing_targets: bool = True
    strict_legal_mask: bool = True
```

`ignore_missing_targets = True` は、欠損 target を loss から除外することを表す。欠損 target をゼロ target として扱う意味ではない。

---

## 5. Dataset と artifact 管理

### 5.1 artifact 一覧

training pipeline は、以下の artifact を生成する。

```text
SelfPlayGame log
ReanalyseResult log
TrainingSample shard
TrainingDatasetMetadata
TrainingRunLog
Checkpoint
EvalRunLog
GateReport
PromotionRecord
```

すべての artifact は、最低限以下を持つ。

```text
artifact_id
artifact_type
created_at
producer_config_id
producer_code_version
parent_artifact_ids
schema_version
```

### 5.2 dataset shard

`TrainingSample` は shard 単位で保存する。

標準の保存方針は以下である。

```text
sample metadata:
  columnar format または jsonl

sfen:
  保存する

feature tensor:
  標準では保存しない
  feature_version と sfen から再生成する

policy / q target:
  sparse 表現で保存する

legal_moves:
  sparse label list として保存する
```

feature tensor を保存する場合は、`feature_version` と `encoder_config_id` を必ず保存する。

### 5.3 dataset split

validation split は、sample 単位ではなく source 単位のリークを避ける。

標準方針は以下である。

```text
selfplay:
  source_game_id 単位で train / validation を分ける

position_buffer restart:
  restart_game_id 単位で分ける
  source_game_id の偏りもログで見る

reanalyse:
  reanalysed position_id 単位で分ける

teacher data:
  棋譜または探索 batch 単位で分ける
```

同一局面またはほぼ同一の連続局面が train と validation にまたがることを減らす。

### 5.4 DatasetRegistry

dataset registry は、以下を保存する。

```python
@dataclass
class DatasetRegistryEntry:
    dataset_id: str
    dataset_version: str
    shard_ids: list[str]

    source_counts: dict[str, int]
    mode_counts: dict[str, int]
    outcome_counts: dict[str, int]
    start_position_source_counts: dict[str, int]
    tag_counts: dict[str, int]

    feature_version: str
    move_label_version: str
    target_version: str
    outcome_class_order: list[str]

    source_checkpoint_ids: list[str]
    source_selfplay_run_ids: list[str]
    source_reanalyse_run_ids: list[str]

    validation_summary: dict[str, float | int]
```

dataset registry を見れば、その dataset がどの checkpoint とどの self-play / reanalyse から作られたか分かるようにする。

---

## 6. mini-batch sampling

### 6.1 基本方針

mini-batch sampling は、単純な一様 sampling にしない。

理由は以下である。

```text
self-play sample が通常局面に偏りやすい
入玉勝ち・相手入玉負けは希少になりやすい
同一対局の連続局面は相関が強い
shuffled_initial 由来 sample は通常分布と異なる
reanalyse sample は高品質だが高コストで数が少ない
```

ただし、最初から複雑にしすぎると原因分析が難しくなるため、phase ごとに段階的に制御を増やす。

### 6.2 標準 batch 構成

初期実装の batch 構成候補は以下である。

```text
Phase 0 supervised_init:
  teacher_search / teacher_move 中心

Phase 1 klent0_normal:
  selfplay normal: 80% 〜 100%
  teacher / supervised replay: 0% 〜 20%

Phase 2 position_buffer_growth:
  selfplay normal: 60% 〜 80%
  position_buffer restart: 20% 〜 40%

Phase 3 nyugyoku_augmentation:
  normal selfplay: 40% 〜 60%
  style selfplay: 20% 〜 40%
  nyugyoku-related position buffer: 10% 〜 30%
  teacher / supervised replay: 0% 〜 10%

Phase 4 reanalyse:
  selfplay: 50% 〜 80%
  reanalyse: 20% 〜 50%
```

実際の比率は `DatasetMixConfig` で管理する。

### 6.3 source 別扱い

source 別の標準扱いは以下である。

| source | 標準用途 | 注意 |
| --- | --- | --- |
| `selfplay` | 主データ | mode / outcome / start source を見て sampling する |
| `reanalyse` | 難所補正 | target_quality を保存し、低品質な結果は混ぜない |
| `teacher_search` | Phase 0 と replay | 通常棋力の anchor として使う |
| `teacher_move` | policy 初期化 | Q target がない場合は policy のみ |
| `supervised_kifu` | 実験候補 | 出所・分布差・権利管理を明確にする |
| `tactical_forced` | 局所確定 target | 宣言勝ち確定・詰み確定だけに限定する |
| `debug_test` | テスト専用 | 通常 training には混ぜない |

### 6.4 mode 別扱い

mode 別の標準方針は以下である。

```text
normal:
  policy_normal / q_normal_result / outcome calibration を重視する

nyugyoku:
  policy_style / q_nyugyoku の正側を厚くする

anti_nyugyoku:
  policy_style / q_nyugyoku の負側を厚くする

mutual_nyugyoku:
  win_nyugyoku / loss_nyugyoku / draw を薄くしすぎない

bakuretsu:
  通常棋力と入玉成分の中間として扱う
```

mode の意味は探索設定と一致させる。単に `policy_style` に流すかどうかだけで mode weight を定義してはならない。

### 6.5 outcome balancing

入玉 outcome の不足は、以下の順で対処する。

```text
1. position buffer の入玉関連 category quota を増やす
2. 入玉関連局面からの restart を増やす
3. 入玉 outcome sample を oversampling する
4. q_nyugyoku / outcome の sample weight を上げる
5. outcome class weight を導入する
```

最初から強い class weight を使うと、outcome probability の calibration が崩れる可能性がある。

### 6.6 correlated sample 抑制

mini-batch 内では、以下を制限する。

```text
同一 source_game_id の sample 数
同一 restart_game_id の sample 数
同一 start_position_id の sample 数
同一 position_key の sample 数
同一 opening prefix の sample 数
```

制限により batch が作れない場合は、警告を出して制限を緩める。ただし、同一局面の重複を黙って大量に入れてはならない。

---

## 7. feature encode と batch 構築

### 7.1 feature encode の位置

標準では、feature tensor は training 時に `sfen` と `feature_version` から生成する。

理由は以下である。

```text
feature tensor 保存容量を減らせる
feature encoder のバグ修正時に再生成できる
sfen を正本にできる
```

ただし、大規模学習で feature encode が bottleneck になる場合は、feature cache を使ってよい。

### 7.2 feature cache

feature cache を使う場合は、cache key を以下で作る。

```text
position_key
side_to_move
feature_version
rule_version
```

反復状態、最大手数までの残り、直前手種別などは標準入力特徴に含めないため、feature cache key に入れない。

ただし、探索 request や terminal 判定で必要な rule state は別管理する。

### 7.3 TrainingBatch

trainer に渡す batch は以下の構造体相当とする。

```python
@dataclass
class TrainingBatch:
    features: torch.Tensor                  # [B, C, 9, 9]
    legal_move_mask: torch.Tensor           # [B, MOVE_LABEL_SIZE]

    policy_normal_target: SparseBatchDistribution | None
    policy_style_target: SparseBatchDistribution | None
    q_normal_result_target: SparseBatchValues | None
    q_nyugyoku_target: SparseBatchValues | None
    outcome_cause_target: torch.Tensor | None
    moves_left_target: torch.Tensor | None

    target_masks: TargetMasks
    sample_weights: torch.Tensor
    target_loss_weights: TargetLossWeightsBatch

    sample_ids: list[str]
    source: list[str]
    mode_id: list[str | None]
    tags: list[list[str]]
```

`legal_move_mask` は loss 計算に必須である。

### 7.4 batch validation

trainer に渡す直前に、以下を検査する。

```text
features に NaN / inf がない
legal_move_mask が空でない
policy target の確率和が 1
policy target の move が legal_move_mask に含まれる
q target の move が legal_move_mask に含まれる
q target が [-1, 1] に収まる
outcome target が 0..4 に収まる
moves_left_target が 0 以上
TargetMasks と target None が一致する
```

異常 batch は学習に使わない。該当 sample_id を quarantine log に保存する。

---

## 8. loss と optimizer

### 8.1 総合 loss

総合 loss は head 別 loss の重みつき和である。

```text
L_total =
    w_policy_normal   * L_policy_normal
  + w_policy_style    * L_policy_style
  + w_q_normal_result * L_q_normal_result
  + w_q_nyugyoku      * L_q_nyugyoku
  + w_outcome         * L_outcome
  + w_moves_left      * L_moves_left
```

各 head の loss は、target が存在する sample だけで計算する。

`sample_weight`、`target_loss_weight`、mode weight、source weight は、head loss の平均時に反映する。

### 8.2 policy loss

policy loss は、合法手 mask 後の cross entropy とする。

```text
L_policy = -Σ_a pi_target(a|s) log softmax_legal(logits)(a)
```

違法手に target 確率を持たせてはならない。

### 8.3 q loss

q loss は、target が定義された move のみで計算する。

```text
q = tanh(q_raw)
L_q = SmoothL1(q[target_moves], q_target_values)
```

選択手だけ target がある sample では、非選択合法手の q loss は計算しない。

Gumbel MCTS reanalyse により全合法手 target がある場合は、全合法手 target を使ってよい。

### 8.4 outcome loss

`outcome_cause_head` は、初期実装では hard label の cross entropy を標準とする。

reanalyse 由来の soft outcome target は実験候補とする。

### 8.5 moves_left loss

`moves_left_pred` は `log1p` scale の target を学習する。

`moves_left` は補助 head であり、入玉局面を短い勝ちへ過度に誘導するための直接報酬としては使わない。

### 8.6 optimizer

初期実装の optimizer は `AdamW` を標準候補とする。

標準候補は以下である。

```text
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 1e-4
scheduler = linear warmup + cosine decay
warmup_steps = 1000
grad_clip_norm = 5.0
mixed_precision = bf16 または fp32
EMA = enabled
```

大規模化する場合は、learning rate、batch size、weight decay、EMA decay を sweep する。

### 8.7 gradient monitoring

head が多いため、以下を必ずログに残す。

```text
loss/head
loss/source
loss/mode
loss/outcome
head_gradient_norm
trunk_gradient_norm
adapter_gradient_norm
q_raw_mean / q_raw_std
policy_logit_mean / policy_logit_std
outcome_entropy
moves_left_pred_mean
```

`policy_style_head` や `q_nyugyoku_head` の gradient が長期間ほぼ 0 の場合は、style / nyugyoku sample が不足している可能性がある。

---

## 9. phase 別 pipeline

### 9.1 Phase 0: supervised_init

目的は、KLENT-0 自己対局を始められる最低限の network を作ることである。

入力データは以下である。

```text
teacher_search
teacher_move
詰み探索で確定した局所 target
既存自己対局棋譜
外部固定深さ探索の結果
```

学習対象は以下である。

```text
policy_normal_head
policy_style_head
q_normal_result_head
q_nyugyoku_head
outcome_cause_head
moves_left_head
```

ただし、初期段階では `policy_normal_head` と `q_normal_result_head` を優先し、`policy_style_head` と `q_nyugyoku_head` は低重みで開始する。

通過条件は以下である。

```text
network output shape test が通る
loss が NaN / inf にならない
policy が random より明確に良い
legal mask loss test が通る
fixed validation で target と output が整合する
```

### 9.2 Phase 1: klent0_normal

目的は、通常モードの KLENT-0 self-play を成立させることである。

標準生成設定は以下である。

```text
pairing:
  normal_vs_normal 中心

search:
  KLENT-0 + CPU tactical solver

start position:
  initial
  random_2ply
  shuffled_initial
  少量または無効の position_buffer
```

training dataset は、normal selfplay を中心にする。

通過条件は以下である。

```text
自己対局が failed_games = 0 で一定局数生成できる
TrainingSample が validation を通る
policy entropy が collapse していない
短手数決着率が異常でない
通常 offline validation が改善する
```

### 9.3 Phase 2: position_buffer_growth

目的は、自己対局で得た有用局面から再開し、局面分布を広げることである。

導入するものは以下である。

```text
PositionBufferWriter
PositionBufferSampler
category quota
restart game target
source metadata
```

重要な不変条件は以下である。

```text
restart 対局 sample の target は restart_game_result から作る
元対局の source_game_result を勝敗 target に使わない
start_position.source / category / tags を sample に引き継ぐ
```

通過条件は以下である。

```text
position buffer sampling が schema validation を通る
restart target が current-side view で正しい
source 別 loss と outcome 分布がログで見える
重複率が許容範囲に収まる
```

### 9.4 Phase 3: nyugyoku_augmentation

目的は、通常棋力を維持しながら入玉関連局面と style target を増やすことである。

増やすデータは以下である。

```text
nyugyoku vs normal
normal vs nyugyoku
anti_nyugyoku vs nyugyoku
nyugyoku vs anti_nyugyoku
mutual_nyugyoku vs mutual_nyugyoku
bakuretsu vs normal
normal vs bakuretsu
入玉有望 position buffer restart
相手入玉リスク position buffer restart
相入玉風 position buffer restart
```

`policy_normal_head` の通常棋力を守るため、style mode の方策 target は原則として `policy_style_head` に流す。

通過条件は以下である。

```text
normal offline validation が大きく劣化していない
q_nyugyoku calibration が改善している
win_nyugyoku / loss_nyugyoku sample が増えている
相手入玉負け率が悪化していない
style policy の entropy が collapse していない
```

### 9.5 Phase 4: reanalyse

目的は、KLENT-0 では不安定な難所局面を Gumbel MCTS reanalyse で補正することである。

reanalyse 対象は以下である。

```text
policy entropy が高い局面
Q の上位手が僅差の局面
入玉関連局面
評価が不安定な終盤局面
通常勝ちと入玉勝ちの選択が割れる局面
自己対局で悪手が疑われる局面
```

reanalyse dataset は、以下を成分別に保存する。

```text
pi_reanalyse
Q_normal_result_reanalyse
Q_nyugyoku_reanalyse
outcome_cause_reanalyse
```

通過条件は以下である。

```text
reanalyse target が legal moves 上に定義されている
q_normal_result と q_nyugyoku が成分別に保存されている
target_quality が記録されている
KLENT-0 と reanalyse の差分が分析できる
```

### 9.6 Phase 5: eval_and_scaling

目的は、モデル規模、データ比率、探索設定を評価 gate に基づいて最適化することである。

実施するものは以下である。

```text
normal_klent0 formal evaluation
nyugyoku_klent0 sweep
anti_nyugyoku_klent0 sweep
mutual_nyugyoku_klent0 sweep
bakuretsu_klent0 sweep
Gumbel MCTS small / medium evaluation
model size scaling
loss weight ablation
position buffer quota ablation
```

正式な通常棋力 gate と入玉最適化 gate は `docs/14_evaluation.md` に従う。

---

## 10. reanalyse pipeline

### 10.1 ReanalyseCandidateQueue

reanalyse 候補は、自己対局ログから抽出する。

保存する metadata は以下である。

```python
@dataclass
class ReanalyseCandidate:
    candidate_id: str
    source_game_id: str
    source_step_index: int
    sfen: str
    side_to_move: int
    legal_moves: list[int]

    model_version: str
    search_config_id: str
    mode_id: str
    start_position_source: str

    reason_tags: list[str]
    priority: float
    uncertainty_features: dict
    nyugyoku_features: dict
    tactical_flags: dict
```

### 10.2 priority

priority は以下を使って決める。

```text
policy entropy
Q top gap
outcome entropy
q_normal_result と q_nyugyoku の符号衝突
入玉関連特徴
相手入玉リスク特徴
tactical solver 補正の有無
過去 checkpoint での失敗履歴
```

priority は reanalyse 実行順に使うだけであり、target value そのものを変えてはならない。

### 10.3 ReanalyseResult

reanalyse 結果は、以下の構造体相当で保存する。

```python
@dataclass
class ReanalyseResult:
    reanalyse_id: str
    candidate_id: str
    source_checkpoint_id: str
    reanalyse_checkpoint_id: str
    reanalyse_config_id: str

    legal_moves: list[int]
    pi_reanalyse: SparseMoveDistribution
    q_normal_result_reanalyse: SparseMoveValues | None
    q_nyugyoku_reanalyse: SparseMoveValues | None
    outcome_cause_reanalyse: list[float] | None

    search_type: str
    num_simulations: int
    target_quality: float
    completed: bool
    failure_reason: str | None
```

`failure_reason` がある結果は training dataset に混ぜない。

### 10.4 merge policy

reanalyse target は、元の selfplay target を常に無条件で上書きするわけではない。

標準方針は以下である。

```text
target_quality が閾値以上:
  policy target は pi_reanalyse を優先する
  q target は reanalyse の成分別 target を優先する

q target が片方の成分だけ存在:
  存在する成分だけ置き換える
  もう片方を 0 で埋めてはならない

policy target だけ存在:
  policy だけ置き換える
  q target は元の target を使う
```

置き換えた場合は、`source = "reanalyse"` または `tags` に `reanalysed` を付ける。

---

## 11. checkpoint 管理

### 11.1 checkpoint 内容

checkpoint には以下を保存する。

```python
@dataclass
class CheckpointMetadata:
    checkpoint_id: str
    parent_checkpoint_id: str | None
    created_at: str

    model_config_id: str
    network_output_version: str
    feature_version: str
    move_label_version: str
    target_version: str

    training_run_id: str
    trainer_run_id: str
    dataset_ids: list[str]
    optimizer_step: int
    seen_samples: int

    loss_summary: dict[str, float]
    validation_summary: dict[str, float]

    is_candidate: bool
    is_promoted: bool
    promotion_tags: list[str]
```

### 11.2 EMA checkpoint

EMA を使う場合は、以下を分けて保存する。

```text
raw checkpoint:
  optimizer が直接更新した重み

ema checkpoint:
  評価用の移動平均重み
```

評価では EMA checkpoint を標準候補にしてよい。

ただし、self-play に raw と EMA のどちらを使ったかを必ずログに残す。

### 11.3 checkpoint 互換性

checkpoint load 時には、以下を検査する。

```text
feature_version
move_label_version
network_output_version
target_version
outcome class order
MOVE_LABEL_SIZE
input channel count
head shape
```

不一致がある場合は、明示的な migration がない限り load を拒否する。

### 11.4 promotion record

checkpoint または探索設定の採用時には、以下を保存する。

```python
@dataclass
class PromotionRecord:
    promotion_id: str
    checkpoint_id: str | None
    config_id: str | None
    base_checkpoint_id: str | None
    eval_run_ids: list[str]
    gate_config_id: str
    passed_gates: list[str]
    failed_gates: list[str]
    summary_metrics: dict[str, float | int]
    promotion_type: str  # "normal_champion" | "nyugyoku_optimized_config" | "selfplay_candidate"
    reason: str
```

`normal_champion_checkpoint` と `nyugyoku_optimized_config` は別々に管理する。

`nyugyoku_optimized_config` は専用の入玉 checkpoint ではなく、共通 checkpoint に適用する探索設定として保存する。標準では、同じモデル重みを通常モードと入玉モードの両方で使い、`alpha_style_self`、`alpha_style_opponent_model`、`lambda_nyugyoku`、tactical solver 設定などの差で棋風を切り替える。

---

## 12. evaluation gate との接続

### 12.1 gate 実行タイミング

training pipeline では、以下のタイミングで評価を行う。

```text
checkpoint_interval:
  offline validation

candidate_interval:
  correctness gate
  smoke evaluation
  speed smoke

promotion_interval:
  tiered game evaluation

final_interval:
  balanced_positions_2025_24ply_500x2_v1 の1000局評価
```

正式な gate 判定は final evaluation を正本とする。

### 12.2 offline validation

offline validation では、以下を見る。

```text
policy_normal CE
policy_style CE
q_normal_result error
q_nyugyoku error
outcome CE / accuracy
moves_left error
source 別 loss
mode 別 loss
start_position.source 別 loss
outcome 別 loss
```

offline validation は早期検出用であり、通常棋力の最終判断ではない。

### 12.3 smoke evaluation

smoke evaluation は、正式評価の前に実行する。

目的は以下である。

```text
SearchEngine が動くか
feature encode が壊れていないか
move label decode が壊れていないか
self-play / eval game が失敗しないか
速度が極端に悪化していないか
```

smoke evaluation で failed game が出る場合、正式評価へ進めない。

### 12.4 normal champion gate

通常棋力の hard gate は以下である。

```text
candidate mode = normal_klent0
candidate playouts = 0
baseline = bakuretsu_komahiro_v420_depth4
suite = balanced_positions_2025_24ply_500x2_v1
num_games = 1000
win_rate_vs_bakuretsu_komahiro_v420_depth4 >= 0.90
```

この gate を満たす checkpoint を `normal_champion_checkpoint` 候補にする。

### 12.5 nyugyoku optimization gate

入玉最適化では、checkpoint ではなく探索設定を採用対象にする。

標準では、`normal_champion_checkpoint` または評価対象の共通 checkpoint に対して複数の `EvalMode` / `SearchModeConfig` を評価し、勝率制約を満たす設定を `nyugyoku_optimized_config` として保存する。

入玉専用 checkpoint を別に持つ設計にはしない。モデル重みを共通にし、設定だけで通常モードと入玉モードを切り替えられるようにする。

入玉最適化では、以下を制約とする。

```text
win_rate_vs_bakuretsu_komahiro_v420_depth4 >= 0.80
```

この制約を満たす候補の中で、以下を最大化する。

```text
declaration_win_rate_vs_bakuretsu_komahiro_v420_depth4
```

同率の場合は、評価仕様の tie-break に従う。

### 12.6 gate failure の扱い

gate 失敗時の基本対応は以下である。

| 失敗 | 主な戻り先 |
| --- | --- |
| correctness failure | schema / rule / label / target 実装 |
| stability failure | self-play / search / feature encode / runtime |
| normal strength 低下 | dataset mix、style 混合比、policy_normal weight |
| 入玉宣言勝ち率不足 | position buffer quota、style pairing、q_nyugyoku weight |
| 相手入玉負け増加 | anti_nyugyoku pairing、loss_nyugyoku oversampling |
| 速度悪化 | model size、batching、tactical budget、mixed precision |

失敗 checkpoint を自己対局生成に使う場合は、明示的に `selfplay_candidate` として隔離し、champion を上書きしない。

---

## 13. logging と可観測性

### 13.1 TrainingRunLog

training run では、以下を保存する。

```python
@dataclass
class TrainingRunLog:
    training_run_id: str
    config_id: str
    phase_id: str
    start_time: str
    end_time: str | None

    input_checkpoint_id: str | None
    output_checkpoint_ids: list[str]
    dataset_ids: list[str]

    total_optimizer_steps: int
    total_seen_samples: int
    total_seen_positions: int

    metrics: dict[str, list[tuple[int, float]]]
    warnings: list[str]
    errors: list[str]
```

### 13.2 必須 metric

最低限、以下をログに残す。

```text
train/loss_total
train/loss_policy_normal
train/loss_policy_style
train/loss_q_normal_result
train/loss_q_nyugyoku
train/loss_outcome
train/loss_moves_left

valid/loss_total
valid/loss_by_source
valid/loss_by_mode
valid/loss_by_outcome

batch/source_ratio
batch/mode_ratio
batch/outcome_ratio
batch/start_position_source_ratio
batch/reanalyse_ratio

model/policy_normal_entropy
model/policy_style_entropy
model/q_normal_result_mean
model/q_nyugyoku_mean
model/outcome_entropy
model/moves_left_mean

runtime/samples_per_sec
runtime/feature_encode_ms
runtime/forward_ms
runtime/backward_ms
runtime/dataloader_wait_ms
runtime/gpu_memory
```

### 13.3 入玉関連 metric

入玉関連では、以下を保存する。

```text
data/win_nyugyoku_samples
data/loss_nyugyoku_samples
data/nyugyoku_promising_samples
data/opponent_nyugyoku_risk_samples
data/mutual_nyugyoku_like_samples

valid/q_nyugyoku_error_win_nyugyoku
valid/q_nyugyoku_error_loss_nyugyoku
valid/outcome_loss_win_nyugyoku
valid/outcome_loss_loss_nyugyoku
valid/policy_style_loss_nyugyoku_mode
valid/policy_style_loss_anti_nyugyoku_mode
```

### 13.4 anomaly log

異常は通常 log と分けて保存する。

```python
@dataclass
class AnomalyRecord:
    anomaly_id: str
    source_artifact_id: str
    sample_id: str | None
    game_id: str | None
    checkpoint_id: str | None
    anomaly_type: str
    message: str
    context: dict
    action: str  # "discard_sample" | "quarantine_game" | "abort_run"
```

異常 sample を黙って補正して学習に混ぜてはならない。

---

## 14. 再現性

### 14.1 seed

以下の seed を分けて管理する。

```text
training_seed
selfplay_seed
start_position_seed
batch_sampling_seed
reanalyse_seed
evaluation_seed
```

完全 deterministic でなくてもよいが、run の再現に必要な seed と config は必ず保存する。

### 14.2 code version

すべての artifact に以下を保存する。

```text
git_commit
working_tree_dirty
build_id
python_version
cuda_version
pytorch_version
engine_binary_hash
```

外部 engine や教師探索を使う場合は、その binary hash と設定も保存する。

### 14.3 config snapshot

run 開始時点の config は、参照 ID だけでなく snapshot も保存する。

理由は以下である。

```text
後から config file が変更されても run を再現できる
checkpoint と dataset の lineage を追える
評価結果の比較時に設定差分を見られる
```

---

## 15. 分散実行

### 15.1 標準構成

初期実装では、以下の疎結合な構成を想定する。

```text
self-play workers:
  checkpoint を読み、SelfPlayGame log を出す

reanalyse workers:
  candidate queue を読み、ReanalyseResult を出す

dataset builder:
  log を読み、TrainingSample shard を出す

trainer:
  dataset shard を読み、checkpoint を出す

evaluator:
  checkpoint を読み、EvalRunLog を出す
```

各 worker は artifact を介して接続し、メモリ上の暗黙状態に依存しない。

### 15.2 stale data

自己対局データは、生成元 checkpoint が古くなる。

標準では、古いデータを即捨てずに、sampling weight を下げる。

```text
latest checkpoint 由来:
  high weight

recent checkpoint 由来:
  medium weight

ancient checkpoint 由来:
  low weight または replay buffer から除外
```

古いデータを使う場合でも、feature_version、move_label_version、target_version が一致している必要がある。

### 15.3 trainer と data generation の同期

trainer が checkpoint を更新するたびに self-play を即切り替えると、データ分布が不安定になる可能性がある。

標準では、self-play checkpoint の更新を gate 付きにする。

```text
latest training checkpoint
  -> offline validation
  -> smoke evaluation
  -> selfplay_candidate promotion
  -> next self-play workers
```

正式 champion でなくても self-play candidate として使ってよいが、stability gate を満たしている必要がある。

---

## 16. 実装上の不変条件

### 16.1 schema / version

```text
すべての artifact に schema_version を保存する
feature_version を欠落させない
move_label_version を欠落させない
target_version を欠落させない
outcome class order を checkpoint / dataset に保存する
config snapshot を保存する
```

### 16.2 target / loss

```text
すべての target は現在手番視点である
policy target は合法手上の分布である
Q target は合法手だけに定義する
欠損 target を 0 target として扱わない
Q_mode を target の正本にしない
違法手を loss に含めない
terminal node を通常 training sample にしない
```

### 16.3 data integrity

```text
failed game を training dataset に入れない
異常 sample を黙って補正しない
restart 対局の target に元対局結果を使わない
manual / test source を通常 training に混ぜない
評価 suite の固定局面を training に混ぜない
```

### 16.4 checkpoint / evaluation

```text
checkpoint は version 互換性を検査して load する
`normal_champion_checkpoint` と `nyugyoku_optimized_config` を分ける
loss だけで checkpoint を昇格しない
formal evaluation の raw count を保存する
failed_games と通常の勝敗を混ぜない
```

---

## 17. 検証項目

### 17.1 dataset build test

```text
SelfPlayGame log から TrainingSample shard を作れる
reanalyse result を target に merge できる
TargetValidator が異常 sample を検出できる
source / mode / outcome / start_position counts が metadata と一致する
train / validation split で source_game_id が漏れない
```

### 17.2 sampler test

```text
DatasetMixConfig の比率に近い batch が出る
同一 game_id 制限が効く
入玉 outcome oversampling が効く
欠損 target の mask が正しい
legal_move_mask と target move が一致する
```

### 17.3 trainer test

```text
toy batch で全 head の loss が計算できる
欠損 target の head loss が無視される
違法手 target があると validation で落ちる
backward で NaN / inf が出ない
checkpoint save / load 後に同一入力の出力が一致する
EMA checkpoint を保存・読み込みできる
```

### 17.4 pipeline integration test

```text
小規模 self-play -> dataset -> trainer -> checkpoint -> evaluation が通る
run id から全 artifact lineage を辿れる
途中 resume で optimizer step と dataset cursor が復元される
failed game が quarantine される
stability gate 失敗時に promotion されない
```

### 17.5 evaluation gate test

```text
correctness gate が fixture failure を検出する
smoke evaluation の failed game が gate failure になる
normal champion gate と nyugyoku optimization gate が別々に判定される
raw count から rate が再計算できる
```

---

## 18. 実験項目

初期実装後の実験項目は以下である。

```text
DatasetMixConfig の source 比率
normal selfplay と teacher replay の比率
shuffled_initial 由来 sample の policy_normal weight
position buffer category quota
nyugyoku outcome oversampling
loss head weight
q_nyugyoku weight
policy_style warmup schedule
reanalyse sample 比率
reanalyse target_quality 閾値
EMA decay
batch size / learning rate scaling
AdamW weight decay
mixed precision fp32 / bf16
feature cache 有無
selfplay checkpoint 更新頻度
formal evaluation の頻度
```

実験では、通常棋力 gate と入玉最適化 gate を混同しない。

---

## 19. 未決事項

現時点の未決事項は以下である。

```text
feature tensor を保存するか、常に再生成するか
大規模化時の dataset shard format
EMA checkpoint を self-play に使うか raw checkpoint を使うか
teacher replay をどの phase まで残すか
reanalyse target の soft outcome をいつ導入するか
formal evaluation の実行頻度
古い self-play data の decay 関数
分散 trainer を初期実装に含めるか
```

これらは実装後の速度、保存容量、通常棋力、入玉指標を見て決める。

---

## 20. 暫定結論

`docs/15_training_pipeline.md` は、爆裂玉登太郎の学習実行を以下の流れとして定義する。

```text
教師あり初期化
  -> KLENT-0 normal self-play
  -> target / dataset build
  -> trainer update
  -> checkpoint evaluation
  -> position buffer / style augmentation
  -> Gumbel MCTS reanalyse
  -> formal evaluation gate
```

重要な原則は以下である。

```text
自己対局と学習を分離する
target の正本を増やさない
欠損 target と 0 target を区別する
通常棋力を先に安定させる
style / 入玉データは段階的に増やす
reanalyse は全局面ではなく難所に使う
loss ではなく評価 gate で checkpoint を昇格する
`normal_champion_checkpoint` と `nyugyoku_optimized_config` を分ける
```

この仕様により、自己対局、target 生成、trainer、評価を疎結合に保ちながら、通常棋力と入玉宣言勝ち率を別軸で改善できるようにする。

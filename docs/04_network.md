# docs/04_network.md - ネットワーク本体仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎のニューラルネットワーク本体の構造を定義する。

対象は以下である。

```text
入力特徴量
trunk architecture
head 接続方式
モデル規模
推論 API
実装上の不変条件
検証項目
実験項目
```

各 head の出力意味、loss、target、視点変換、KLENT 改善方策での使い方は `docs/05_outputs.md` を正とする。

このドキュメントでは、`policy_normal_head` や `q_nyugyoku_head` などを「どのようなネットワーク構造から出すか」を扱う。

---

## 1. 基本方針

### 1.1 目的

ネットワークは、KLENT-first 自己対局を成立させるために、1回の forward で以下を同時に出す。

```text
通常方策
入玉棋風方策
合法手ごとの通常勝敗 action-value
合法手ごとの入玉関連 action-value
局面単位の終局原因分布
終局までの距離
```

爆裂玉登太郎では、全局面で重い MCTS を行うのではなく、policy と action-value から 0-playout の改善方策を作る。そのため、ネットワークは単なる局面 value だけでなく、合法手ごとの action-value を直接出す。

### 1.2 trunk は shared、head は adapter で分離

盤面理解の大部分は共通であるため、入力から中間表現までは shared trunk とする。

ただし、出力の意味が大きく異なるため、shared feature から各 head へ直接つなぐだけにはしない。今回の出力は policy 2系統、action-value 2系統、局面 outcome、moves_left を同時に扱うため、head adapter を標準で入れる。

基本形は以下である。

```text
features
  -> input stem
  -> VoVNetV2-style shared trunk
  -> shared feature map
      -> policy_adapter
          -> policy_normal_head
          -> policy_style_head
      -> q_adapter
          -> q_normal_result_head
          -> q_nyugyoku_head
      -> outcome_adapter
          -> outcome_cause_head
      -> moves_left_adapter
          -> moves_left_head
```

`policy_normal_head` と `policy_style_head` の出力層は共有しない。通常棋力と入玉棋風を混ぜすぎると、通常棋力の劣化や不自然な入玉狙いが起きやすいためである。

`q_normal_result_head` と `q_nyugyoku_head` の出力層も共有しない。通常勝敗価値と入玉関連価値は、同じ勝敗信号から派生していても、探索時の使われ方が異なるためである。

### 1.3 出力仕様は `docs/05_outputs.md` に従う

forward の戻り値は以下の構造体相当とする。

```python
@dataclass
class NetworkOutput:
    policy_normal_logits: torch.Tensor      # [B, MOVE_LABEL_SIZE]
    policy_style_logits: torch.Tensor       # [B, MOVE_LABEL_SIZE]
    q_normal_result_raw: torch.Tensor       # [B, MOVE_LABEL_SIZE]
    q_nyugyoku_raw: torch.Tensor            # [B, MOVE_LABEL_SIZE]
    outcome_cause_logits: torch.Tensor      # [B, 5]
    moves_left_pred: torch.Tensor           # [B, 1]
```

ネットワーク本体は全指し手ラベルに対して値を出す。合法手 mask、softmax、`tanh`、KLENT 方策計算はネットワーク外の変換層・探索側・loss 側で行う。

---

## 2. 入力特徴量

### 2.1 入力の基本形

初期実装では、盤面を `[B, C, 9, 9]` の dense plane として入力する。

```python
features: Tensor  # [B, INPUT_CHANNELS, 9, 9]
```

`INPUT_CHANNELS` は固定値として定義する。実装では、feature encoder と network の両方が同じ定数を参照する。

入力特徴量は、dlshogi の feature 設計を参考にしつつ、爆裂玉登太郎の現在手番視点仕様に合わせて整理する。

基本方針は以下である。

```text
盤上特徴:
  駒配置
  駒の利き
  利き数

盤面全体へ broadcast する特徴:
  持ち駒
  王手状態
  入玉関連特徴
```

### 2.2 盤上駒特徴

盤上駒は、先手・後手の絶対視点ではなく、原則として現在手番視点へ正規化する。

```text
own piece planes
opponent piece planes
```

駒種は成駒を含めた固定 plane とする。

```text
歩
香
桂
銀
金
角
飛
玉
と
成香
成桂
成銀
馬
龍
```

これにより、ネットワーク出力の「現在手番視点」という仕様と揃える。

### 2.3 利き特徴

dlshogi では、駒配置だけでなく駒の利きと利き数も入力特徴に含める。

爆裂玉登太郎でも、初期実装から以下を標準入力に含める。

```text
own attack by piece type
opponent attack by piece type
own attack count clipped
opponent attack count clipped
```

利き数は、上限で clip した複数 plane として持つ。

```text
attack_count >= 1
attack_count >= 2
attack_count >= 3
```

これは、王手・詰み・玉の進出・敵陣突破・入玉ルートの安全性に関係するためである。

### 2.4 持ち駒特徴

持ち駒は、盤面全体に broadcast した plane として入れる。

例:

```text
own_hand_pawn_count_plane
own_hand_lance_count_plane
...
opp_hand_pawn_count_plane
opp_hand_lance_count_plane
...
```

標準は、dlshogi と同様に持ち駒枚数を上限まで段階表現する方式とする。

```text
hand_piece_count >= 1
hand_piece_count >= 2
...
hand_piece_count >= max_count
```

対象は以下である。

```text
歩: 8
香: 4
桂: 4
銀: 4
金: 4
角: 2
飛: 2
```

count を単一の正規化 scalar plane として入れる方式も実装は容易だが、初期実装では既存将棋AI実装との近さを優先する。

### 2.5 王手状態特徴

王手状態は、全盤面に broadcast した binary plane として入れる。

```text
in_check_plane
```

王手対応は CPU tactical solver でも扱うが、network 側にも局面状態として与える。

### 2.6 入玉関連特徴

現局面から厳密に計算できる情報は output head にしない。

ただし、入玉宣言勝ちや相手入玉リスクはこのプロジェクトの主目的であり、ルール上の閾値が強く効くため、入力特徴量としては初期実装から入れる。

標準は dlshogi の `NYUGYOKU_FEATURES` に近い設計とする。

```text
nyugyoku_flag
nyugyoku_opp_field_count_features
nyugyoku_score_features
```

現在手番視点に正規化して、少なくとも以下を持つ。

```text
own_king_in_enemy_camp_plane
opp_king_in_own_camp_plane
own_pieces_in_enemy_camp_count_planes
opp_pieces_in_own_camp_count_planes
own_nyugyoku_score_remaining_planes
opp_nyugyoku_score_remaining_planes
```

`nyugyoku_score_remaining_planes` は、宣言勝ちに必要な点数までの残りを段階表現する。

```text
remaining_score <= 0
remaining_score <= 1
...
remaining_score <= MAX_NYUGYOKU_SCORE
```

初期値は以下を候補とする。

```text
MAX_NYUGYOKU_OPP_FIELD = 10
MAX_NYUGYOKU_SCORE = 20
```

これは dlshogi の実装に近い上限であり、feature 数を過剰に増やさずに入玉判定の閾値情報を与えられる。

### 2.7 反復・手数関連特徴は標準入力にしない

以下は、初期実装の network input には入れない。

```text
ply_normalized
repetition_count
last_move_type
最大手数までの残り
```

理由は以下である。

```text
将棋AIで一般的に強い入力特徴として定着していない
feature version を複雑にする
局面の本質的な盤面評価より、対局生成ルールに依存しやすい
千日手・最大手数・反復は探索側または終局処理側で扱える
```

千日手や最大手数到達を outcome target に反映することは行うが、network input としては標準採用しない。

---

## 3. 座標系と視点

### 3.1 現在手番視点を標準にする

入力特徴量は現在手番視点に正規化する。

```text
手番側から見て前方が同じ向きになるように盤面を変換する
手番側の駒を own plane に入れる
相手側の駒を opponent plane に入れる
```

これにより、以下の仕様と揃う。

```text
policy は現在手番が指す手の分布
Q_normal_result は現在手番側の通常勝敗価値
Q_nyugyoku は現在手番側の入玉勝ちを正、相手入玉負けを負
outcome_cause は現在手番視点
```

### 3.2 move label は固定長

policy と action-value は `[B, MOVE_LABEL_SIZE]` を出す。

`MOVE_LABEL_SIZE` は、将棋の全指し手ラベルを固定長で表す。

```text
盤上移動
成り・不成
持ち駒打ち
```

現在手番視点に正規化する場合、move encoder / decoder も同じ視点変換を使う。

ネットワーク内部では、合法手かどうかを判断しない。合法手 mask は探索側または loss 側の責務とする。

---

## 4. trunk architecture

### 4.1 第一候補

ネットワーク本体は **VoVNetV2-style CNN trunk** を第一候補とする。

採用理由は以下である。

```text
9x9盤面では局所特徴と中距離特徴の両方が重要
OSA により多段の特徴を一括集約できる
VoVNetV2 の identity mapping により深くしても最適化しやすい
VoVNetV2 の effective SE により channel attention を軽く入れられる
spatial downsampling なしの 9x9 盤面維持と相性がよい
複数 head に共有特徴を渡しやすい
GPU batched inference と相性がよい
```

### 4.2 基本構造

初期実装の基本構造は以下とする。

```text
Input [B, C, 9, 9]
  -> Stem Conv 3x3 + BN + SiLU
  -> VoVNetV2 OSA Block 1
  -> VoVNetV2 OSA Block 2
  -> Inception / Transformer optional block
  -> VoVNetV2 OSA Block 3
  -> VoVNetV2 OSA Block 4
  -> Inception / Transformer optional block
  -> Shared Feature [B, D, 9, 9]
```

盤面が小さいため、原則として spatial downsampling は行わない。

```text
stride = 1
padding = same
feature map size = 9x9 を維持
```

将棋では駒の位置関係が重要であり、9x9を途中で潰すと、指し手ごとの局所的な情報が失われやすい。

### 4.3 VoVNetV2 OSA Block

OSA block は以下の構造とする。

```text
x0
  -> conv3x3 -> x1
  -> conv3x3 -> x2
  -> conv3x3 -> x3
  -> conv3x3 -> x4
concat(x1, x2, x3, x4)
  -> conv1x1
  -> BatchNorm2d
  -> effective SE
  -> identity add with x0
  -> activation
```

`x0` を concat に含めるかどうかは実装選択にしてよいが、identity mapping は必須とする。

初期実装では、各 block の conv 数は 3〜5 を候補にする。

### 4.4 effective SE

VoVNetV2 では、OSA の出力に effective SE を入れる。

爆裂玉登太郎でも、標準で effective SE を入れる。

```text
eSE(x):
  s = global_average_pool(x)
  g = sigmoid(linear(s))
  return x * g
```

通常の SE のように channel 次元を一度小さくしすぎると、OSA で集約した多様な特徴を落とす可能性がある。

そのため、標準は channel reduction なし、または reduction を入れても小さくしすぎない形とする。

### 4.5 normalization

標準は `BatchNorm2d` とする。

理由は以下である。

```text
既存の将棋CNN実装で広く使われている
GPU推論時に畳み込みへ融合しやすい
GroupNorm は将棋AIでは有効性が安定していない
```

自己対局データ分布が変化する点には注意する。

対策は normalization 種別を変えることではなく、まず以下で行う。

```text
十分な batch size を確保する
学習時の BatchNorm running stats をログする
評価時に train/eval の差を検証する
必要なら BN momentum を調整する
必要なら post-training で BN statistics を再推定する
```

### 4.6 activation

標準は `SiLU` とする。

候補は以下である。

```text
ReLU
SiLU
```

初期実装では、速度と安定性のバランスから `SiLU` を第一候補、`ReLU` を軽量版候補とする。

---

## 5. head architecture

### 5.1 adapter は必須にする

今回の出力は通常の policy-value network より複雑である。

```text
policy_normal
policy_style
q_normal_result
q_nyugyoku
outcome_cause
moves_left
```

これらを shared feature からすべて直接出すと、head 間の勾配干渉が強くなりやすい。

そのため、head adapter を標準構成に含める。

標準形は以下である。

```text
shared_feature [B, D, 9, 9]
  -> policy_adapter [B, Dp, 9, 9]
      -> policy_normal_head
      -> policy_style_head
  -> q_adapter [B, Dq, 9, 9]
      -> q_normal_result_head
      -> q_nyugyoku_head
  -> state_adapter [B, Ds, 9, 9]
      -> outcome_cause_head
      -> moves_left_head
```

policy と q はどちらも指し手ラベル単位の出力を持つため、adapter の構造を近くしてよい。

state 系は局面単位の出力なので、global pooling 前提の adapter にする。

### 5.2 adapter block

adapter block は軽量な 1〜2 層の CNN とする。

```text
Conv3x3
BatchNorm2d
SiLU
Conv1x1
BatchNorm2d
SiLU
```

軽量版では 1x1 conv だけでもよい。

複雑な head ごとに trunk 全体を分けるのではなく、shared trunk は維持し、adapter で出力ごとの差分を吸収する。

### 5.3 policy heads

policy head は、9x9 feature map から `MOVE_LABEL_SIZE` logits を出す。

標準は、dlshogi と同様に move label plane を出して flatten する形とする。

```text
policy_adapter
  -> Conv1x1(out_channels = MOVE_LABEL_PLANES)
  -> flatten [B, 9 * 9 * MOVE_LABEL_PLANES]
  -> optional bias
  -> [B, MOVE_LABEL_SIZE]
```

`MOVE_LABEL_SIZE` は以下を満たす。

```text
MOVE_LABEL_SIZE = 9 * 9 * MOVE_LABEL_PLANES
```

`policy_normal_head` と `policy_style_head` は同じ構造だが、重みは共有しない。

flatten + large linear は、パラメータ数とメモリ帯域が増えやすいため標準にはしない。

### 5.4 action-value heads

action-value head も、policy と同じ move label plane 形式で出す。

```text
q_adapter
  -> Conv1x1(out_channels = MOVE_LABEL_PLANES)
  -> flatten [B, MOVE_LABEL_SIZE]
  -> optional bias
  -> q_*_raw [B, MOVE_LABEL_SIZE]
```

出力は以下である。

```text
q_normal_result_raw: [B, MOVE_LABEL_SIZE]
q_nyugyoku_raw:      [B, MOVE_LABEL_SIZE]
```

`q_normal_result_raw` と `q_nyugyoku_raw` は `tanh` 前の値である。

`tanh` は network module 内ではなく、探索用変換関数または loss 計算側で適用する。

理由は以下である。

```text
raw 値の分布をログで確認しやすい
loss 側で clipping / scaling 実験をしやすい
NetworkOutput の仕様を logits/raw に統一できる
```

### 5.5 outcome_cause_head

`outcome_cause_head` は局面単位の head である。

```text
state_adapter
  -> global pooling
  -> MLP
  -> logits [B, 5]
```

global pooling は以下を候補にする。

```text
mean pooling
mean + max pooling concat
attention pooling
```

初期実装では mean + max pooling concat を標準とする。

入玉関連では盤面全体の状態が効くため、mean pooling だけで弱い場合は attention pooling を試す。

### 5.6 moves_left_head

`moves_left_head` も局面単位の head である。

```text
state_adapter
  -> global pooling
  -> MLP
  -> scalar [B, 1]
```

出力は `log1p(moves_left)` の回帰値である。

推論時に手数へ戻す処理はネットワーク外で行う。

### 5.7 channel split 方式

adapter の代替案として、trunk の channel を大きめにし、shared feature を channel 方向に split して各 head に渡す方式も候補にする。

```text
shared_feature [B, D, 9, 9]
  -> split(policy_channels, q_channels, state_channels)
  -> each head
```

ただし、初期実装では adapter 方式を標準とする。

理由は以下である。

```text
head ごとの表現容量を明示しやすい
後から adapter だけ ablation しやすい
channel split は固定割り当ての影響が強い
```

---

## 6. Inception / Attention block

### 6.1 導入方針

VoVNetV2 OSA block を基本にしつつ、InceptionBlock と lightweight Transformer block を追加候補にする。

将棋では以下の両方が重要である。

```text
局所形・利き・駒の接触
遠距離の利き・玉の進出・敵陣内の点数・全体配置
```

CNN は局所形に強い。一方で、入玉や相入玉では盤面全体の関係を早めに扱いたい。

そのため、以下の設計を候補にする。

```text
OSA block で局所・中距離特徴を作る
InceptionBlock で縦横の長い利きを拾う
Transformer block で全81マスの相互作用を拾う
```

### 6.2 InceptionBlock

InceptionBlock は、dlshogi の実装を参考に、縦・横・局所を足し合わせる形を候補にする。

```text
x
  -> Conv 9x1
  -> Conv 1x9
  -> Conv 1x1
  -> sum
  -> BN + activation
  -> Conv 9x1
  -> Conv 1x9
  -> Conv 1x1
  -> sum
  -> BN
  -> residual add + activation
```

9x1 と 1x9 は、香・飛・角・玉の進出路など、盤面上の長い関係を軽く拾うために有効な可能性がある。

初期実装では、InceptionBlock は Transformer より先に試す価値が高い。

理由は以下である。

```text
CNN系なので実装が単純
BatchNorm2d と相性がよい
推論速度の悪化が比較的小さい
将棋の縦横方向の利きと相性がよい
```

### 6.3 Transformer block

Transformer block は、9x9 = 81 tokens の self-attention として実装する。

```text
feature [B, D, 9, 9]
  -> tokens [B, 81, D]
  -> MultiHeadSelfAttention
  -> FFN
  -> feature [B, D, 9, 9]
```

標準候補は以下である。

```text
heads = 4
embedding dim = D
relative position bias = optional
pre-norm ではなく BN/Conv 系との接続を優先
```

Transformer block を多く入れすぎると、局所形を作る CNN 成分が弱くなる可能性がある。

そのため、連続した Transformer block を大量に置く構成は標準にしない。

### 6.4 block schedule

初期実装では、block schedule を config で指定できるようにする。

```python
block_schedule: list[str]
# "O" = VoVNetV2 OSA block
# "I" = InceptionBlock
# "T" = TransformerBlock
```

標準候補は以下である。

```text
network_small_v2:
  O O I O O T

network_base_v2:
  O O I O O T O O I O O T
```

ResTNet 的な比較候補として、以下も試す。

```text
O O T O O T
O I O T O I O T
O O I O O I
O O O O
```

`O O T` の反復は、ResTNet の `R R T` 反復を VoVNetV2 に置き換えたものとみなす。

### 6.5 採用順序

初期の優先順位は以下とする。

```text
1. OSA only
2. OSA + Inception
3. OSA + late Transformer
4. OSA + repeated Transformer
```

実装の複雑さ、推論速度、通常勝率、入玉指標を見て段階的に採用する。

---

## 7. モデル規模

### 7.1 初期モデル

初期実装は、自己対局と反復実験の速度を優先する。

候補は以下である。

```text
channels D = 128
block_schedule = O O I O O T
layers per OSA block = 4
adapter hidden dim = 128
state head hidden dim = 256
```

このモデルを `network_small_v2` とする。

### 7.2 拡張モデル

学習ループが安定した後、以下を試す。

```text
network_tiny_v2:
  channels D = 64
  block_schedule = O O I O
  adapter hidden dim = 64

network_small_v2:
  channels D = 128
  block_schedule = O O I O O T
  adapter hidden dim = 128

network_base_v2:
  channels D = 192
  block_schedule = O O I O O T O O I O O T
  adapter hidden dim = 192

network_large_v2:
  channels D = 256
  block_schedule = O O I O O T O O I O O T O O I O O T
  adapter hidden dim = 256
```

最初から large にしない。

理由は以下である。

```text
feature encoder / target / KLENT 実装のバグを早く見つけたい
自己対局局面数を優先したい
head 設計の ablation を回しやすくしたい
```

### 7.3 モデル選定基準

モデル規模は、単純な loss だけで決めない。

以下を同時に見る。

```text
1秒あたりの batched inference 数
自己対局生成速度
通常勝率
入玉宣言勝ち率
相手入玉負け率
q_nyugyoku calibration
policy entropy
KLENT 方策の安定性
```

---

## 8. 推論形態

### 8.1 batched inference

自己対局では、多数局面をまとめて GPU 推論する。

```text
self-play workers
  -> inference queue
  -> GPU batch inference
  -> NetworkOutput
  -> workers
```

network は、1局面ずつ呼び出すのではなく、batch 前提で実装する。

### 8.2 mixed precision

学習と推論では mixed precision を使ってよい。

候補は以下である。

```text
fp32
fp16 autocast
bf16 autocast
```

初期実装では fp32 を正とし、数値が合うことを確認してから mixed precision を有効化する。

特に以下を確認する。

```text
policy softmax の NaN
q raw の発散
outcome logits の極端化
KLENT pi の確率和
```

### 8.3 NN cache

NN cache は標準では有効にしない。

今回のネットワークは、少なくとも初期実装では比較的軽量にする。そのため、局面 key の生成、hash 計算、辞書 lookup、cache miss 処理の方が、GPU batched inference より重くなる可能性がある。

使用する場合は、以下の条件を満たすときに限る。

```text
同一局面への再訪問が多い
cache hit rate が十分に高い
batch inference を阻害しない
key 生成時間 + lookup 時間 < 推論節約時間
```

標準設定は以下とする。

```python
enable_nn_cache = False
```

Gumbel MCTS や実戦時探索など、同一探索木内で再評価が起きやすい場合だけ有効化を検討する。

cache key は以下を含める。

```text
盤面
持ち駒
手番
model version
feature version
```

反復関連特徴を input に含めないため、初期実装では repetition state を cache key に含めない。

cache value は raw `NetworkOutput` とする。

探索モードごとの `α_style` や `λ_nyugyoku` は cache key に含めない。これらはネットワーク出力後の変換で使う値だからである。

cache を有効化する場合は、最低限以下をログする。

```text
nn_cache/enabled
nn_cache/hit_rate
nn_cache/key_build_us
nn_cache/lookup_us
nn_cache/miss_inference_us
nn_cache/effective_positions_per_sec
```

---

## 9. 学習時の扱い

### 9.1 forward は raw output のみ

network module は raw output だけを返す。

```python
output = model(features)
```

loss 計算側で以下を行う。

```text
legal mask
policy cross entropy
q raw の tanh 変換
q loss
outcome cross entropy
moves_left loss
```

### 9.2 head ごとの loss weight

loss weight の標準は `docs/05_outputs.md` に従う。

ネットワーク本体として重要なのは、head が互いに干渉しすぎないことである。

特に初期学習では以下に注意する。

```text
policy_normal_head を最優先する
policy_style_head は弱く始める
q_nyugyoku_head は入玉関連データが増えるまで過信しない
moves_left_head は補助信号として軽く扱う
```

### 9.3 gradient の監視

multi-head network では、特定 head の loss が trunk を壊す可能性がある。

最低限、以下をログする。

```text
grad_norm/trunk
grad_norm/policy_adapter
grad_norm/q_adapter
grad_norm/state_adapter
grad_norm/policy_normal_head
grad_norm/policy_style_head
grad_norm/q_normal_result_head
grad_norm/q_nyugyoku_head
grad_norm/outcome_cause_head
grad_norm/moves_left_head
```

必要なら、head ごとに gradient clipping または loss weight の warmup を入れる。

---

## 10. 実装 API

### 10.1 model config

ネットワーク構成は config で管理する。

```python
@dataclass
class NetworkConfig:
    input_channels: int
    move_label_size: int
    move_label_planes: int

    channels: int = 128
    block_schedule: tuple[str, ...] = ("O", "O", "I", "O", "O", "T")
    layers_per_osa_block: int = 4

    adapter_hidden_dim: int = 128
    state_head_hidden_dim: int = 256

    norm_type: str = "batchnorm"
    activation: str = "silu"
    use_effective_se: bool = True
    use_head_adapters: bool = True

    transformer_heads: int = 4
    transformer_relative_position_bias: bool = True

    enable_nn_cache: bool = False
```

### 10.2 model class

```python
class BakuretsuTamanoboriNetwork(nn.Module):
    def __init__(self, config: NetworkConfig):
        ...

    def forward(self, features: torch.Tensor) -> NetworkOutput:
        ...
```

### 10.3 feature encoder との境界

network は `features: Tensor` の意味を直接解釈しない。

以下は feature encoder 側の責務である。

```text
盤面を現在手番視点へ変換する
駒を plane に配置する
利きを plane に変換する
持ち駒を plane に変換する
王手状態を plane に変換する
入玉関連 scalar / count を plane に変換する
dtype / device を揃える
```

network 側は、`[B, INPUT_CHANNELS, 9, 9]` を受け取って `NetworkOutput` を返すだけにする。

---

## 11. 実装上の不変条件

以下は必ず守る。

```text
入力 feature map は [B, C, 9, 9]
trunk は 9x9 spatial size を維持する
forward は raw logits / raw value だけを返す
合法手 mask は network 内で行わない
policy head と q head は MOVE_LABEL_SIZE 次元を返す
outcome_cause_head は 5 class logits を返す
moves_left_head は [B, 1] を返す
q head に network 内で tanh をかけない
policy_normal_head と policy_style_head の重みは共有しない
q_normal_result_head と q_nyugyoku_head の重みは共有しない
head adapter を標準で使う
BatchNorm2d を標準 normalization とする
terminal node を network に特別扱いさせない
feature_version と model_version を checkpoint に保存する
```

---

## 12. 検証項目

### 12.1 shape test

最低限、以下を確認する。

```text
入力 [B, C, 9, 9] を受け取れる
policy_normal_logits が [B, MOVE_LABEL_SIZE]
policy_style_logits が [B, MOVE_LABEL_SIZE]
q_normal_result_raw が [B, MOVE_LABEL_SIZE]
q_nyugyoku_raw が [B, MOVE_LABEL_SIZE]
outcome_cause_logits が [B, 5]
moves_left_pred が [B, 1]
```

### 12.2 feature encoder test

最低限、以下を確認する。

```text
現在手番視点への盤面変換が正しい
先手番・後手番で own/opponent plane が入れ替わる
駒種 plane が正しい
成駒 plane が正しい
利き plane が正しい
利き数 clip が正しい
持ち駒段階 plane が正しい
王手状態 plane が正しい
入玉関連 plane が正しい
feature_version が保存される
```

入玉関連特徴については、以下を個別に確認する。

```text
own_king_in_enemy_camp_plane
opp_king_in_own_camp_plane
own_pieces_in_enemy_camp_count_planes
opp_pieces_in_own_camp_count_planes
own_nyugyoku_score_remaining_planes
opp_nyugyoku_score_remaining_planes
```

### 12.3 determinism test

`eval()` mode では、同じ入力に対して同じ出力になることを確認する。

```text
model.eval()
同一 features を2回 forward
全 output が一致する
```

BatchNorm を使うため、train/eval の挙動差を明示的に確認する。

### 12.4 mask 外部適用 test

network は違法手を知らない。

そのため、以下を確認する。

```text
network 単体では全 MOVE_LABEL_SIZE に値を出す
合法手 mask 後の softmax は外部関数で行う
違法手確率が0になることは変換関数側で確認する
```

### 12.5 mixed precision test

mixed precision 有効時に以下を確認する。

```text
NaN が出ない
Inf が出ない
policy softmax が破綻しない
q raw が極端に発散しない
outcome logits が極端に発散しない
```

### 12.6 checkpoint compatibility test

以下を保存・復元できることを確認する。

```text
NetworkConfig
state_dict
feature_version
move_label_size
move_label_planes
outcome class order
```

`MOVE_LABEL_SIZE`、`MOVE_LABEL_PLANES`、`INPUT_CHANNELS`、outcome class order が異なる checkpoint は原則として読み込まない。

---

## 13. ログ仕様

ネットワーク本体に関して、最低限以下をログする。

```text
model/num_parameters
model/num_trainable_parameters
model/input_channels
model/move_label_size
model/move_label_planes
model/channels
model/block_schedule
model/layers_per_osa_block
model/use_effective_se
model/use_head_adapters
model/norm_type

activation/policy_normal_logits_mean
activation/policy_normal_logits_std
activation/policy_style_logits_mean
activation/policy_style_logits_std
activation/q_normal_result_raw_mean
activation/q_normal_result_raw_std
activation/q_nyugyoku_raw_mean
activation/q_nyugyoku_raw_std
activation/outcome_cause_logits_mean
activation/outcome_cause_logits_std
activation/moves_left_pred_mean
activation/moves_left_pred_std

grad_norm/trunk
grad_norm/policy_adapter
grad_norm/q_adapter
grad_norm/state_adapter
grad_norm/policy_normal_head
grad_norm/policy_style_head
grad_norm/q_normal_result_head
grad_norm/q_nyugyoku_head
grad_norm/outcome_cause_head
grad_norm/moves_left_head

batchnorm/running_mean_abs_mean
batchnorm/running_var_mean
batchnorm/train_eval_output_diff

inference/batch_size
inference/positions_per_sec
inference/gpu_memory_mb
```

入玉関連の head は学習初期に弱くなりやすいので、`q_nyugyoku_raw` と `policy_style_logits` の分布は通常ログに含める。

---

## 14. 実験項目

### 14.1 trunk ablation

以下を比較する。

```text
ResNet-style trunk
VoVNetV2 OSA only
VoVNetV2 + Inception
VoVNetV2 + late Transformer
VoVNetV2 + repeated OOT pattern
```

観測する指標は以下である。

```text
通常勝率
入玉宣言勝ち率
相手入玉負け率
policy loss
q loss
outcome calibration
positions/sec
```

### 14.2 model size scaling

以下を比較する。

```text
network_tiny_v2
network_small_v2
network_base_v2
network_large_v2
```

最初は `network_small_v2` を基準にする。

### 14.3 input feature ablation

以下を比較する。

```text
A. 駒配置 + 持ち駒 + 王手
B. A + 利き特徴
C. B + 入玉関連特徴
```

反復・手数関連特徴は標準 ablation に含めない。

確認することは以下である。

```text
通常棋力が落ちないか
P_win_nyugyoku の calibration が改善するか
P_loss_nyugyoku の calibration が改善するか
q_nyugyoku の selected move error が下がるか
入玉有望局面での方策が改善するか
```

### 14.4 head adapter ablation

以下を比較する。

```text
shared_feature から全 head 直接分岐
policy/q/state adapter を分ける
policy_normal/style adapter も分ける
q_normal/q_nyugyoku adapter も分ける
channel split 方式
```

標準は `policy/q/state adapter を分ける` である。

過度な分離はパラメータ増加と過学習につながるため、通常勝率と入玉指標の両方で判断する。

### 14.5 normalization ablation

標準は `BatchNorm2d` とする。

比較候補は以下に留める。

```text
BatchNorm2d
BatchNorm2d + BN statistics recalibration
Normalizationなし
```

GroupNorm は優先実験には含めない。

### 14.6 NN cache ablation

以下を比較する。

```text
NN cache 無効
Gumbel MCTS 内だけ NN cache 有効
実戦時探索だけ NN cache 有効
全推論で NN cache 有効
```

確認することは以下である。

```text
positions/sec が改善するか
GPU batch size が小さくなりすぎないか
key 生成と lookup がボトルネックになっていないか
hit rate が十分に高いか
```

---

## 15. 未決事項

### 15.1 入玉関連 feature の細部

標準では dlshogi の `NYUGYOKU_FEATURES` に近い設計を採用する。

ただし、以下の細部は実装時に確定する。

```text
点数を「現在点数」で表すか「残り点数」で表すか
段階 plane の向き
MAX_NYUGYOKU_OPP_FIELD
MAX_NYUGYOKU_SCORE
先手28点 / 後手27点などルール差の扱い
```

重要なのは、feature encoder の unit test で宣言勝ち条件との整合性を保証することである。

### 15.2 move label 表現

policy / q head は move label plane 形式を標準とする。

ただし、以下は未決である。

```text
MOVE_LABEL_PLANES の class order
成り・不成のラベル順
打ち駒ラベル順
現在手番視点変換後の move decoder
```

### 15.3 Inception / Transformer の初期採用範囲

標準候補は `O O I O O T` とする。

ただし、実装負荷を下げたい場合は、最初に `O O O O` を作ってから `I` と `T` を足す。

```text
最小実装:
  O O O O

推奨初期実装:
  O O I O O T
```

### 15.4 Transformer block の形式

以下は未決である。

```text
relative position bias を入れるか
attention 前後を BatchNorm 系にするか LayerNorm 系にするか
FFN を 1x1 Conv で実装するか Linear で実装するか
Transformer block を OSA の後に置くか Inception の後に置くか
```

標準は、CNN trunk との接続を優先して 1x1 Conv + BatchNorm2d 系で実装する。

---

## 16. 参考にする既存実装・研究

### 16.1 dlshogi 入力特徴量

参考:

```text
https://github.com/TadaoYamaoka/DeepLearningShogi/blob/ae0cc8fbf5d3f2f0d010685ebc5eaba13b961999/cppshogi/cppshogi.h
```

dlshogi では、駒配置・利き・利き数に加えて、持ち駒、王手、入玉関連特徴を入力として扱う。

特に `NYUGYOKU_FEATURES` は、爆裂玉登太郎の目的と近いため参考にする。

### 16.2 dlshogi policy_value_network_exp___i.py

参考:

```text
https://github.com/TadaoYamaoka/DeepLearningShogi/blob/master/dlshogi/network/policy_value_network_exp___i.py
```

参考にする点は以下である。

```text
9x1 / 1x9 / 1x1 の InceptionBlock
最後付近の AttentionBlock
policy を move label plane として出して flatten する構造
BatchNorm2d を使う構造
```

### 16.3 VoVNetV2

参考:

```text
https://github.com/youngwanLEE/vovnet-detectron2
```

参考にする点は以下である。

```text
OSA module
identity mapping
目立ちすぎない channel attention としての effective SE
効率的な shared trunk
```

### 16.4 ResTNet

参考:

```text
https://arxiv.org/abs/2410.05347
```

参考にする点は以下である。

```text
board game で CNN と Transformer を混ぜる発想
Residual / Transformer の RRT 反復
局所特徴を先に作ってから global attention を使う構成
Transformer を多くしすぎない設計
```

ただし、この研究は主に囲碁・Hex の結果であるため、将棋にそのまま最適とは限らない。爆裂玉登太郎では、VoVNetV2 OSA を主軸にし、Inception / Transformer は ablation で採否を決める。

---

## 17. 暫定結論

初期実装のネットワークは以下とする。

```text
入力:
  現在手番視点の [B, INPUT_CHANNELS, 9, 9] dense plane
  駒配置
  利き
  利き数
  持ち駒
  王手状態
  dlshogi風の入玉関連特徴

trunk:
  VoVNetV2-style CNN
  OSA + identity mapping + effective SE
  9x9 spatial size を維持
  BatchNorm2d を標準
  network_small_v2 を基準

optional blocks:
  InceptionBlock
  lightweight TransformerBlock

heads:
  head adapter を必須にする
  policy_normal_head
  policy_style_head
  q_normal_result_head
  q_nyugyoku_head
  outcome_cause_head
  moves_left_head

forward:
  raw logits / raw value を返す
  合法手 mask、softmax、tanh、KLENT 方策計算は network 外で行う

NN cache:
  初期実装では無効
  Gumbel MCTS / 実戦時探索で効果測定してから使う
```

最重要の設計判断は以下である。

```text
通常棋力と入玉棋風を policy head で分離する
KLENT-0 のために合法手ごとの action-value を直接出す
通常勝敗価値と入玉関連価値を q head で分離する
複雑な multi-head 出力に対して adapter を標準で入れる
入力特徴量には入玉関連特徴を初期実装から含める
反復・手数関連特徴は初期入力に含めない
terminal / 詰み / 宣言勝ち可能判定は network ではなく探索側で扱う
```

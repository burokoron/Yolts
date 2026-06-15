# docs/05_outputs.md - ネットワーク出力仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎のニューラルネットワークが出力する head の仕様を定義する。

対象は以下である。

```text
policy_normal_head
policy_style_head
q_normal_result_head
q_nyugyoku_head
outcome_cause_head
moves_left_head
```

爆裂玉登太郎では、通常棋力と入玉棋風を同時に扱う必要がある。また、自己対局では KLENT 系の 0-playout 方策改善を主学習ループにする。

そのため、ネットワークは、方策、合法手ごとの action-value、局面単位の終局原因分布、終局までの距離を出力する。

宣言点数、自玉段、敵陣内自駒数など、現局面から厳密に計算できる情報はネットワーク出力にはしない。必要であれば入力特徴量または探索側のルール特徴として扱う。

---

## 1. 基本方針

### 1.1 出力を増やしすぎない

出力 head は以下に絞る。

```text
policy_normal_head
policy_style_head
q_normal_result_head
q_nyugyoku_head
outcome_cause_head
moves_left_head
```

方針は以下である。

- 方策は通常用と棋風用の2 headに分ける
- KLENT 用に合法手ごとの action-value を出す
- action-value は通常勝敗価値と入玉関連価値に分ける
- 局面単位の value は終局原因分布として出す
- 終局までの距離を補助的に予測する
- 現局面から計算できる静的情報は出力させない

### 1.2 通常棋力と入玉棋風を分離する

単一 policy に入玉優先の自己対局データを混ぜると、通常棋力が劣化する可能性がある。

そのため、方策 head は以下の2つに分ける。

```text
policy_normal_head
policy_style_head
```

`policy_normal_head` は通常棋力の維持を主目的とする。

`policy_style_head` は、入玉を狙う棋風・相入玉に持ち込む棋風・相手の入玉を警戒する棋風など、通常 policy とは異なる探索条件から得られた方策を吸収するために使う。

### 1.3 value は outcome 分布と action-value に分ける

入玉将棋では、単なる勝敗だけでは情報が不足する。

同じ勝ちでも、通常勝ちと入玉勝ちは意味が異なる。同じ負けでも、通常負けと相手入玉による負けは意味が異なる。

局面単位では、現在手番視点の終局原因分布を出力する。

合法手ごとの action-value は以下の2成分に分ける。

```text
Q_normal_result(s, a)
Q_nyugyoku(s, a)
```

`Q_normal_result` は通常勝敗価値を表す。

`Q_nyugyoku` は、自分の入玉勝ちを正、相手入玉による負けを負として表す。

総合勝敗価値は以下で復元する。

```text
Q_total_result(s, a) = Q_normal_result(s, a) + Q_nyugyoku(s, a)
```

探索モードごとの価値は以下で表す。

```text
Q_mode(s, a) = Q_normal_result(s, a) + λ_nyugyoku * Q_nyugyoku(s, a)
```

`λ_nyugyoku = 1.0` の場合、通常勝ちと入玉勝ちを同じ勝敗価値として扱う総合勝敗価値になる。`λ_nyugyoku` を大きくすると、自分の入玉勝ちを強く好み、相手入玉による負けを強く嫌う。

---

## 2. 出力一覧

ネットワークの forward は、以下の構造体相当を返す。

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

各 head の意味は以下の通りである。

| head | shape | 内容 |
| --- | --- | --- |
| `policy_normal_head` | `[B, MOVE_LABEL_SIZE]` | 通常棋力用の方策 logits |
| `policy_style_head` | `[B, MOVE_LABEL_SIZE]` | 入玉棋風用の方策 logits |
| `q_normal_result_head` | `[B, MOVE_LABEL_SIZE]` | 合法手ごとの通常勝敗価値 |
| `q_nyugyoku_head` | `[B, MOVE_LABEL_SIZE]` | 合法手ごとの入玉関連価値 |
| `outcome_cause_head` | `[B, 5]` | 終局原因つき勝敗分布 logits |
| `moves_left_head` | `[B, 1]` | 終局までの残り手数予測 |

`MOVE_LABEL_SIZE` は `docs/11_move_label.md` の指し手ラベル仕様で定義する固定長の指し手ラベル数である。

ネットワークは全指し手に対して logits または raw value を出す。合法手 mask はネットワーク外、つまり探索側または学習 loss 側で適用する。

---

## 3. policy_normal_head

### 3.1 目的

`policy_normal_head` は、通常の将棋として最も強い手を選ぶための方策を出力する。

これは爆裂玉登太郎の基本棋力を支える head であり、通常モードではこの head を主に使う。

### 3.2 出力形式

```python
policy_normal_logits: Tensor  # [B, MOVE_LABEL_SIZE]
```

- 出力は softmax 前の logits とする
- 合法手以外の logits は出力されるが、使用時に mask する
- policy loss 計算時も合法手 mask を適用する
- 推論時も合法手 mask 後に softmax する

### 3.3 学習ターゲット

主なターゲットは以下とする。

```text
既存エンジンまたは教師探索から得た方策分布
KLENT 改善方策
Gumbel MCTS 再解析から得た訪問回数分布
自己対局で通常モードから得た方策分布
```

教師手が1手だけの場合は one-hot として扱う。分布ターゲットがある場合は、合法手上の確率分布として扱う。

### 3.4 loss

基本は cross entropy とする。

```text
L_policy_normal = CE(policy_normal_logits, pi_normal_target)
```

`pi_normal_target` は合法手上の確率分布である。

---

## 4. policy_style_head

### 4.1 目的

`policy_style_head` は、手番側から見て入玉を狙う棋風を強めるための方策を出力する。

重要なのは、`policy_style_head` は常に「その局面で手番を持つ側にとっての style policy」である、という点である。

したがって、探索時に root 側が入玉優先モードだからといって、相手側の node でも同じ強さで `policy_style_head` を混ぜるとは限らない。

### 4.2 出力形式

```python
policy_style_logits: Tensor  # [B, MOVE_LABEL_SIZE]
```

形式は `policy_normal_head` と同じである。

### 4.3 学習ターゲット

主なターゲットは以下とする。

```text
入玉優先モードの KLENT 改善方策
相手入玉警戒モードの KLENT 改善方策
入玉有望局面からの自己対局で得た方策分布
Gumbel MCTS 再解析から得た style 方策分布
```

style 用データが少ない段階では、通常 target を小さい重みで学習する。

### 4.4 loss

基本は cross entropy とする。

```text
L_policy_style = CE(policy_style_logits, pi_style_target)
```

`pi_style_target` が存在しないサンプルでは、normal target を小さい重みで学習する。

### 4.5 探索時の使い方

探索時には、通常 policy と style policy を確率分布としてブレンドする。

```text
p_normal = softmax_legal(policy_normal_logits)
p_style  = softmax_legal(policy_style_logits)

p_prior = normalize((1 - α_style) * p_normal + α_style * p_style)
```

logits を直接ブレンドせず、softmax 後の確率をブレンドする。2つの head の logit scale が学習段階によって異なる可能性があるためである。

### 4.6 root側と相手側の policy 設定

`α_style` は探索全体で1つではなく、少なくとも以下の2つを持つ。

```text
α_style_self
α_style_opponent_model
```

意味は以下である。

```text
α_style_self:
  root側、つまりこの探索で着手を選ぶ側の node で使う style 混合率

α_style_opponent_model:
  相手側 node で、相手の指し手を予測するために使う style 混合率
```

node の手番が root 側なら `α_style_self` を使う。node の手番が相手側なら `α_style_opponent_model` を使う。

```python
def select_alpha_style(node_side, root_side, config):
    if node_side == root_side:
        return config.alpha_style_self
    else:
        return config.alpha_style_opponent_model
```

通常の対外対局では、相手を普通の将棋AIと仮定し、以下を初期値にする。

```text
自分が入玉優先モード:
  α_style_self = 0.3 〜 0.5
  α_style_opponent_model = 0.0 〜 0.1
```

相手も入玉を狙うAIであることが分かっている場合、または相入玉耐性を評価したい場合は、以下のようにする。

```text
相入玉想定モード:
  α_style_self = 0.3 〜 0.5
  α_style_opponent_model = 0.3 〜 0.5
```

相手の入玉狙いを強く警戒したい場合は、相手モデル側の style 混合率と `λ_nyugyoku` を上げる。

```text
相手入玉警戒モード:
  α_style_self = 0.0 〜 0.2
  α_style_opponent_model = 0.3 〜 0.6
  λ_nyugyoku = 大
```

この設計により、以下を制御できる。

```text
自分がどれくらい入玉を狙うか
相手がどれくらい入玉を狙うと仮定するか
相手入玉をどれくらい嫌うか
```

### 4.7 自己対局時の使い方

自己対局では、先手・後手それぞれに探索設定を持たせる。

```python
@dataclass
class PlayerSearchConfig:
    alpha_style_self: float
    alpha_style_opponent_model: float
    lambda_nyugyoku: float
```

自己対局の各プレイヤーは、自分の turn では自分に割り当てられた `PlayerSearchConfig` で着手を選ぶ。

相手側 node の policy 設定には、原則として対戦相手の実際の policy 設定を模倣する。

例えば、先手が入玉優先、後手が通常モードなら以下のようになる。

```text
先手の探索:
  root側: 入玉優先
  相手モデル側: 後手の通常モードを模倣

後手の探索:
  root側: 通常モード
  相手モデル側: 先手の入玉優先モードを模倣
```

自己対局の組み合わせ候補は以下である。

```text
normal vs normal
style vs normal
normal vs style
style vs style
anti-nyugyoku vs style
style vs anti-nyugyoku
bakuretsu vs normal
normal vs bakuretsu
```

---

## 5. action-value heads

### 5.1 目的

KLENT 系の 0-playout 方策改善では、探索なしで改善方策を作るために、合法手ごとの action-value が必要である。

爆裂玉登太郎では、action-value を以下の2成分に分ける。

```text
Q_normal_result(s, a)
Q_nyugyoku(s, a)
```

`Q_normal_result` は通常勝敗価値を表す。

`Q_nyugyoku` は、自分の入玉勝ちを正、相手入玉による負けを負として表す。

### 5.2 出力形式

```python
q_normal_result_raw: Tensor  # [B, MOVE_LABEL_SIZE]
q_nyugyoku_raw: Tensor       # [B, MOVE_LABEL_SIZE]
```

raw 出力は `tanh` で `[-1, 1]` に変換する。

```text
Q_normal_result = tanh(q_normal_result_raw)
Q_nyugyoku      = tanh(q_nyugyoku_raw)
```

総合勝敗価値は以下である。

```text
Q_total_result = Q_normal_result + Q_nyugyoku
```

探索モードごとの価値は以下である。

```text
Q_mode = Q_normal_result + λ_nyugyoku * Q_nyugyoku
```

`λ_nyugyoku = 1.0` のとき、通常勝ち・入玉勝ち・通常負け・相手入玉負けをすべて勝敗価値として扱う。

### 5.3 KLENT 改善方策

KLENT 改善方策は合法手上で以下とする。

```text
pi_klent(a | s) = softmax_legal(
    (Q_mode(s, a) + β * log(max(p_prior(a | s), ε))) / (α + β)
)
```

`α` は entropy regularization、`β` は prior policy からの乖離抑制を表す。

`p_prior` は `policy_normal_head` と `policy_style_head` を混合して作る。

### 5.4 学習ターゲット

各サンプルの終局原因から、現在手番視点で以下の target を作る。

| outcome_cause | normal_result | nyugyoku |
| --- | ---: | ---: |
| `win_normal` | +1 | 0 |
| `win_nyugyoku` | 0 | +1 |
| `draw` | 0 | 0 |
| `loss_nyugyoku` | 0 | -1 |
| `loss_normal` | -1 | 0 |

`q_normal_result_head` は `normal_result` を学習する。

`q_nyugyoku_head` は `nyugyoku` を学習する。

1手ごとの Monte Carlo return だけでなく、λ-return や Gumbel MCTS 再解析結果を target にしてよい。

### 5.5 loss

基本は MSE または Smooth L1 loss とする。

```text
L_q_normal_result = SmoothL1(Q_normal_result, q_normal_result_target)
L_q_nyugyoku      = SmoothL1(Q_nyugyoku, q_nyugyoku_target)
```

局面全体の全合法手に target がある場合は、合法手上で平均する。

実際に選んだ手だけに target がある場合は、選択手に対して loss を計算する。

### 5.6 視点管理

`Q_normal_result` と `Q_nyugyoku` は現在手番視点で定義する。

自己対局ログ、再解析、MCTS backup では、手番が反転するたびに意味が入れ替わる。

特に `Q_nyugyoku` は、正側が「現在手番側の入玉勝ち」、負側が「相手入玉による現在手番側の負け」である。

したがって、`Q_mode` だけを保存して単純に符号反転する実装は避ける。保存する場合は、`Q_normal_result` と `Q_nyugyoku` を分けるか、視点が明示された target として保存する。

---

## 6. outcome_cause_head

### 6.1 目的

`outcome_cause_head` は、現在手番視点で、終局結果と終局原因を同時に予測する。

単なる勝敗 value ではなく、以下を区別する。

```text
通常勝ち
入玉勝ち
引き分け
相手入玉による負け
通常負け
```

これにより、局面単位で以下を計算できる。

```text
通常勝敗価値
入玉関連価値
総合勝敗価値
```

### 6.2 出力形式

```python
outcome_cause_logits: Tensor  # [B, 5]
```

softmax 後の class order は固定で以下とする。

```python
OUTCOME_WIN_NORMAL = 0
OUTCOME_WIN_NYUGYOKU = 1
OUTCOME_DRAW = 2
OUTCOME_LOSS_NYUGYOKU = 3
OUTCOME_LOSS_NORMAL = 4
OUTCOME_NUM_CLASSES = 5
```

この順序は checkpoint 互換性に関わるため、途中で変更してはならない。

### 6.3 確率の定義

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

### 6.4 派生 value

通常勝敗価値は以下で計算する。

```text
V_normal_result = P_win_normal - P_loss_normal
```

入玉関連価値は以下で計算する。

```text
V_nyugyoku = P_win_nyugyoku - P_loss_nyugyoku
```

総合勝敗価値は以下で計算する。

```text
V_total_result = V_normal_result + V_nyugyoku
```

探索モードごとの局面価値は以下である。

```text
V_mode = V_normal_result + λ_nyugyoku * V_nyugyoku
```

### 6.5 perspective transform

`outcome_cause_head` は常に現在手番視点で出力する。

MCTSや再解析では手番が交互に変わるため、親ノード視点で子ノード評価を使う場合は、outcome 分布を変換してから value を計算する。

子ノードが相手手番視点で出した確率を、親手番視点に変換すると以下になる。

```text
parent_P_win_normal    = child_P_loss_normal
parent_P_win_nyugyoku  = child_P_loss_nyugyoku
parent_P_draw          = child_P_draw
parent_P_loss_nyugyoku = child_P_win_nyugyoku
parent_P_loss_normal   = child_P_win_normal
```

MCTS 実装では以下のどちらかを採用する。

```text
A. outcome確率分布をroot手番視点へ変換してから value を計算する
B. 各 node の手番視点を明示し、必要な視点へ変換してから value を計算する
```

標準は A とする。

### 6.6 学習ターゲット

各学習サンプルは、局面時点の手番から見た最終結果を以下の5クラスに変換する。

```text
win_normal
win_nyugyoku
draw
loss_nyugyoku
loss_normal
```

変換規則は以下である。

```text
最終勝者 == 局面の手番側 かつ 終局原因 == normal:
  win_normal

最終勝者 == 局面の手番側 かつ 終局原因 == nyugyoku:
  win_nyugyoku

引き分け:
  draw

最終勝者 != 局面の手番側 かつ 終局原因 == nyugyoku:
  loss_nyugyoku

最終勝者 != 局面の手番側 かつ 終局原因 == normal:
  loss_normal
```

### 6.7 終局原因の分類

終局原因の正本は、自己対局・探索・学習 target 生成で共通の `terminal_reason` とする。

`outcome_cause` では、`terminal_reason` をまず以下の `outcome_category` に変換し、そのうえで現在手番視点の5クラスへ変換する。

| `terminal_reason` | `outcome_category` |
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

自己対局保存形式では、単なる勝敗だけでなく `terminal_reason` と `outcome_category` を必ず保存する。

### 6.8 loss

基本は cross entropy とする。

```text
L_outcome = CE(outcome_cause_logits, outcome_cause_target)
```

入玉勝ち・入玉負けは通常の自己対局では出現頻度が低い可能性があるため、入玉有望局面からの再開自己対局や oversampling を使う。

---

## 7. moves_left_head

### 7.1 目的

`moves_left_head` は、現在局面から終局までの残り手数を予測する。

主な用途は以下である。

```text
学習の補助信号
探索価値の安定化
勝ち局面では早い勝ちを少し好む
負け局面では粘れる手を少し好む
長期入玉ルートの評価補助
```

ただし、`moves_left` を探索価値の主成分にはしない。

### 7.2 出力形式

```python
moves_left_pred: Tensor  # [B, 1]
```

残り手数そのものではなく、以下を予測する。

```text
y_moves_left = log1p(moves_left)
```

推論時には以下で手数に戻す。

```text
moves_left = expm1(moves_left_pred)
```

### 7.3 ターゲット

局面が棋譜中の ply `t` にあり、終局 ply が `T` の場合、以下とする。

```text
moves_left = T - t
```

教師値は以下で作る。

```text
y_moves_left = log1p(max(T - t, 0))
```

### 7.4 loss

Smooth L1 loss を使う。

```text
L_moves_left = SmoothL1(moves_left_pred, log1p(moves_left_target))
```

### 7.5 探索での使い方

`moves_left` はログ・解析・補助評価に使い、KLENT の主 value には強く入れない。

導入する場合は、以下のような小さい補正に留める。

```text
勝ち寄り局面:
  早い勝ちをわずかに優先する

負け寄り局面:
  遅い負けをわずかに優先する

入玉価値が高い局面:
  長期化を過度に嫌わない
```

入玉は長期計画になりやすいため、単純な「早く終わるほどよい」補正は危険である。

---

## 8. 補助 head の扱い

以下は現局面から厳密に計算できるため、head として出力しない。

```text
現在の宣言点数
現在の自玉段
現在の相手玉段
現在の敵陣内自駒数
現在の敵陣内相手駒数
現在の持ち駒点数
```

これらは必要に応じて以下で扱う。

```text
入力特徴量
探索側のルール特徴
局面バッファの分類特徴
評価ログの集計項目
```

未来予測として追加する候補は以下である。

```text
own_nyugyoku_distance_head
opp_nyugyoku_distance_head
final_point_margin_head
```

追加条件は以下である。

```text
outcome_cause_head と q_nyugyoku_head だけでは入玉関連の判断が不足している
入玉勝ち・入玉負けの calibration が悪い
相入玉局面での指し手選択が不安定
十分な教師ラベルを安定して作れる
head 追加による通常棋力劣化が許容範囲である
```

---

## 9. 総合 loss

総合 loss は以下とする。

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

入玉優先モードの自己対局データが増えたら `w_policy_style` と `w_q_nyugyoku` を上げる。

---

## 10. 推論時 API

### 10.1 生出力 API

ネットワークは logits と raw value をそのまま返す。

```python
output = model(features)

output.policy_normal_logits
output.policy_style_logits
output.q_normal_result_raw
output.q_nyugyoku_raw
output.outcome_cause_logits
output.moves_left_pred
```

### 10.2 探索用変換 API

探索側では、生出力を以下に変換する。

```python
@dataclass
class SearchNetworkOutput:
    prior: torch.Tensor                  # [B, MOVE_LABEL_SIZE]
    q_normal_result: torch.Tensor        # [B, MOVE_LABEL_SIZE]
    q_nyugyoku: torch.Tensor             # [B, MOVE_LABEL_SIZE]
    q_total_result: torch.Tensor         # [B, MOVE_LABEL_SIZE]
    q_mode: torch.Tensor                 # [B, MOVE_LABEL_SIZE]
    pi_klent: torch.Tensor               # [B, MOVE_LABEL_SIZE]
    outcome_probs: torch.Tensor          # [B, 5]
    v_normal_result: torch.Tensor        # [B]
    v_nyugyoku: torch.Tensor             # [B]
    v_total_result: torch.Tensor         # [B]
    moves_left: torch.Tensor             # [B]
```

変換処理では以下を行う。

```text
合法手 mask
policy_normal / policy_style の softmax
policy blend
q_normal_result / q_nyugyoku の tanh 変換
Q_total_result の計算
Q_mode の計算
KLENT 改善方策 pi_klent の計算
outcome softmax
V_normal_result / V_nyugyoku / V_total_result の計算
moves_left の expm1 変換
```

### 10.3 合法手がない局面

合法手がない局面、詰み局面、宣言勝ち確定局面などの terminal node では、ネットワーク評価を使わず、探索側で terminal value を返す。

ネットワークに terminal node を無理に評価させる必要はない。

---

## 11. 実験項目

### 11.1 head ablation

以下を比較する。

```text
A. policy_normal + q_normal_result + outcome
B. A + moves_left
C. A + policy_style + q_nyugyoku
D. C + moves_left
```

評価指標:

```text
通常勝率
入玉宣言勝ち率
相手入玉負け率
入玉有望局面からの勝率
相入玉局面での勝率
通常勝ちできる局面で無理に入玉しない率
```

### 11.2 outcome calibration

`outcome_cause_head` について以下を確認する。

```text
5クラス分類 accuracy
入玉勝ち class の recall / precision
入玉負け class の recall / precision
Brier score
ECE
局面タイプ別 calibration
```

特に `P_win_nyugyoku` と `P_loss_nyugyoku` の calibration は、入玉制御に直結するため重要である。

### 11.3 action-value calibration

`q_normal_result_head` と `q_nyugyoku_head` について以下を確認する。

```text
選択手に対する target error
合法手上の value range
Q_total_result と最終勝敗の相関
Q_nyugyoku と入玉勝ち・相手入玉負けの相関
局面タイプ別 calibration
```

### 11.4 policy blend 実験

`α_style` を変えて以下を比較する。

```text
α_style = 0.0
α_style = 0.1
α_style = 0.2
α_style = 0.3
α_style = 0.5
α_style = 0.7
```

観測する指標:

```text
通常勝率の劣化
入玉宣言勝ち率の増加
相手入玉負け率の変化
不自然な入玉狙いの増加
探索の安定性
```

### 11.5 lambda_nyugyoku 実験

`λ_nyugyoku` を変えて以下を比較する。

```text
λ_nyugyoku = 0.5
λ_nyugyoku = 1.0
λ_nyugyoku = 1.5
λ_nyugyoku = 2.0
λ_nyugyoku = 3.0
```

観測する指標:

```text
通常勝率
入玉宣言勝ち率
相手入玉負け率
相入玉局面での勝率
通常勝ちできる局面での寄せ放棄率
```

### 11.6 moves_left loss weight 実験

`w_moves_left` を変えて以下を比較する。

```text
w_moves_left = 0.0
w_moves_left = 0.05
w_moves_left = 0.1
w_moves_left = 0.2
```

確認すること:

```text
通常勝率への影響
終盤評価の安定性
長期入玉ルートを過小評価していないか
勝ち局面での寄せ速度
負け局面での粘り
```

---

## 12. 未決事項

### 12.1 moves_left の形式

候補は以下である。

```text
log1p 回帰
手数 bin 分類
回帰 + bin 分類の併用
```

標準は log1p 回帰とする。

### 12.2 style target の作り方

候補は以下である。

```text
入玉優先 KLENT 方策
探索価値の λ_nyugyoku を上げた自己対局の方策分布
入玉有望局面だけを抽出した方策分布
通常policyからの蒸留 + style探索データ
Gumbel MCTS 再解析による訪問回数分布
```

### 12.3 draw の扱い

千日手、持将棋、最大手数到達をすべて `draw` とするかは検討が必要である。

標準では、勝敗が確定しないものは `draw` にまとめる。

必要になれば、以下へ拡張する。

```text
P_draw_repetition
P_draw_jishogi
P_draw_max_moves
```

### 12.4 outcome class の粒度

標準では5クラスとする。

拡張候補は以下である。

```text
win_checkmate
win_resign
win_nyugyoku_declaration
win_nyugyoku_adjudication
draw_repetition
draw_jishogi
loss_nyugyoku_declaration
loss_nyugyoku_adjudication
loss_resign
loss_checkmate
```

細分化は教師データ量と calibration を見て判断する。

---

## 13. 実装上の不変条件と検証仕様

この章では、ネットワーク出力仕様として固定する事項と、仕様を満たしていることを確認するための検証項目を定義する。

### 13.1 outcome class order

`outcome_cause_head` の class order は以下で固定する。

```python
OUTCOME_WIN_NORMAL = 0
OUTCOME_WIN_NYUGYOKU = 1
OUTCOME_DRAW = 2
OUTCOME_LOSS_NYUGYOKU = 3
OUTCOME_LOSS_NORMAL = 4
OUTCOME_NUM_CLASSES = 5
```

この順序は checkpoint、教師ラベル、ログ、評価コードの互換性に関わるため、途中で変更してはならない。

### 13.2 NetworkOutput interface

ネットワークの forward は、以下の構造体相当の出力を返す。

```python
@dataclass
class NetworkOutput:
    policy_normal_logits: torch.Tensor
    policy_style_logits: torch.Tensor
    q_normal_result_raw: torch.Tensor
    q_nyugyoku_raw: torch.Tensor
    outcome_cause_logits: torch.Tensor
    moves_left_pred: torch.Tensor
```

各 tensor の shape は以下を満たす。

```text
policy_normal_logits:   [B, MOVE_LABEL_SIZE]
policy_style_logits:    [B, MOVE_LABEL_SIZE]
q_normal_result_raw:    [B, MOVE_LABEL_SIZE]
q_nyugyoku_raw:         [B, MOVE_LABEL_SIZE]
outcome_cause_logits:   [B, 5]
moves_left_pred:        [B, 1]
```

### 13.3 変換関数の仕様

ネットワークの生出力から探索・評価に使う値へ変換する処理は、以下の責務を持つ。

```python
def outcome_logits_to_probs(logits):
    ...

def outcome_probs_to_values(probs):
    ...

def transform_outcome_probs_to_opponent_view(probs):
    ...

def blend_policy_logits(policy_normal_logits, policy_style_logits, legal_mask, alpha_style):
    ...

def compute_q_values(q_normal_result_raw, q_nyugyoku_raw, legal_mask):
    ...

def compute_klent_policy(prior, q_mode, legal_mask, alpha_entropy, beta_prior, eps):
    ...
```

`outcome_probs_to_values` は、以下を返す。

```text
V_normal_result
V_nyugyoku
V_total_result
```

定義は以下である。

```text
V_normal_result = P_win_normal - P_loss_normal
V_nyugyoku      = P_win_nyugyoku - P_loss_nyugyoku
V_total_result  = V_normal_result + V_nyugyoku
```

`transform_outcome_probs_to_opponent_view` は、現在手番視点の outcome 分布を相手手番視点へ変換する。

変換規則は以下である。

```text
opponent_P_win_normal    = current_P_loss_normal
opponent_P_win_nyugyoku  = current_P_loss_nyugyoku
opponent_P_draw          = current_P_draw
opponent_P_loss_nyugyoku = current_P_win_nyugyoku
opponent_P_loss_normal   = current_P_win_normal
```

`blend_policy_logits` は名前に logits を含んでいても、logits を直接加算してはならない。

処理順序は以下とする。

```text
1. legal_mask を適用する
2. policy_normal_logits を合法手上で softmax する
3. policy_style_logits を合法手上で softmax する
4. softmax 後の確率分布を α_style で混合する
5. 合法手上で再正規化する
```

`compute_q_values` は raw value に `tanh` を適用し、違法手は探索対象から外す。

```text
Q_normal_result = tanh(q_normal_result_raw)
Q_nyugyoku      = tanh(q_nyugyoku_raw)
Q_total_result  = Q_normal_result + Q_nyugyoku
Q_mode          = Q_normal_result + λ_nyugyoku * Q_nyugyoku
```

`compute_klent_policy` は合法手上だけで `pi_klent` を作り、違法手確率を0にする。

### 13.4 視点管理

`V_mode` や `Q_mode` は、通常勝敗価値と入玉関連価値を合成したスカラーである。

MCTS 内、reanalyse 内、λ-return 計算内では、以下のいずれかの方式で視点を管理する。

```text
A. outcome_probs と action-value 成分を root 手番視点へ変換してから value を計算する
B. 各 node の手番視点を明示し、必要な視点へ変換してから value を計算する
```

標準は A とする。

### 13.5 検証項目

最低限、以下の検証を行う。

```text
outcome softmax の確率和が1になる
V_normal_result の計算が定義通りである
V_nyugyoku の計算が定義通りである
V_total_result の計算が定義通りである
outcome 分布の視点変換が定義通りである
視点変換を2回適用すると元の分布に戻る
q_normal_result が [-1, 1] に入る
q_nyugyoku が [-1, 1] に入る
Q_total_result の計算が定義通りである
Q_mode の計算が定義通りである
合法手 mask 後に違法手確率が0になる
policy blend 後の確率和が1になる
KLENT 改善方策の確率和が1になる
α_style = 0 のとき normal policy と一致する
α_style = 1 のとき style policy と一致する
moves_left_pred の expm1 変換が非負になる
terminal_reason から outcome_cause_target への変換が定義通りである
outcome_cause から q target への変換が定義通りである
```

### 13.6 ログ仕様

学習・自己対局・評価では、最低限以下を記録する。

```text
loss/policy_normal
loss/policy_style
loss/q_normal_result
loss/q_nyugyoku
loss/outcome
loss/moves_left
loss/total

outcome/P_win_normal_mean
outcome/P_win_nyugyoku_mean
outcome/P_draw_mean
outcome/P_loss_nyugyoku_mean
outcome/P_loss_normal_mean

value/V_normal_result_mean
value/V_nyugyoku_mean
value/V_total_result_mean

q/Q_normal_result_selected
q/Q_nyugyoku_selected
q/Q_total_result_selected
q/Q_mode_selected

policy/entropy_prior
policy/entropy_klent
policy/kl_klent_to_prior

moves_left/pred_mean
moves_left/target_mean
```

入玉関連のバグは通常勝率だけでは見えにくい。

そのため、`P_win_nyugyoku`、`P_loss_nyugyoku`、`V_nyugyoku`、`Q_nyugyoku` は必ず通常の学習ログ・評価ログに含める。

---

## 14. 暫定結論

採用するネットワーク出力は以下とする。

```text
policy_normal_head:
  通常棋力用の方策 logits

policy_style_head:
  入玉棋風用の方策 logits

q_normal_result_head:
  合法手ごとの通常勝敗価値

q_nyugyoku_head:
  合法手ごとの入玉関連価値

outcome_cause_head:
  win_normal / win_nyugyoku / draw / loss_nyugyoku / loss_normal の5クラス logits

moves_left_head:
  log1p(終局までの残り手数) の回帰出力
```

action-value は以下で計算する。

```text
Q_normal_result = tanh(q_normal_result_raw)
Q_nyugyoku      = tanh(q_nyugyoku_raw)
Q_total_result  = Q_normal_result + Q_nyugyoku
Q_mode          = Q_normal_result + λ_nyugyoku * Q_nyugyoku
```

outcome 分布からは以下を計算する。

```text
V_normal_result = P_win_normal - P_loss_normal
V_nyugyoku      = P_win_nyugyoku - P_loss_nyugyoku
V_total_result  = V_normal_result + V_nyugyoku
V_mode          = V_normal_result + λ_nyugyoku * V_nyugyoku
```

policy は通常用と棋風用を分ける。

```text
p_prior = normalize((1 - α_style) * p_normal + α_style * p_style)
```

KLENT 改善方策は以下で作る。

```text
pi_klent(a | s) = softmax_legal(
    (Q_mode(s, a) + β * log(max(p_prior(a | s), ε))) / (α + β)
)
```

現局面から厳密に計算できる宣言点数・玉段・敵陣内自駒数などは、ネットワーク出力にはしない。

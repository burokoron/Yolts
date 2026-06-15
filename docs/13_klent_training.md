# docs/13_klent_training.md - KLENT-first 学習・探索方針

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎における KLENT-first の学習・探索方針を定義する。

ここでいう KLENT は、KL regularization と entropy regularization を使う 0-playout 方策改善の系統を指す。

KLENT-first とは、初期学習から全局面で MCTS を行うのではなく、policy と action-value から 0-playout の改善方策を作り、自己対局局面数と更新回数を優先する方針である。

基本構成は以下である。

```text
主学習ループ:
  KLENT-0 + CPU詰み探索

探索補助:
  Gumbel MCTS reanalyse
  難所局面解析
  実戦時 test-time search
```

---

## 1. 基本アルゴリズム

各局面で、まず policy prior を作る。

```text
p_normal = softmax_legal(policy_normal_logits)
p_style  = softmax_legal(policy_style_logits)

p_prior = normalize((1 - α_style) * p_normal + α_style * p_style)
```

次に action-value を変換する。

```text
Q_normal_result = tanh(q_normal_result_raw)
Q_nyugyoku      = tanh(q_nyugyoku_raw)
```

総合勝敗価値と探索モード価値を作る。

```text
Q_total_result(s, a) = Q_normal_result(s, a) + Q_nyugyoku(s, a)

Q_mode(s, a) =
    Q_normal_result(s, a)
  + λ_nyugyoku * Q_nyugyoku(s, a)
```

`λ_nyugyoku = 1.0` の場合、通常勝ちと入玉勝ちを同じ勝敗価値として扱う総合勝敗価値になる。`λ_nyugyoku` を大きくすると、自分の入玉勝ちを強く好み、相手入玉による負けを強く嫌う。

KLENT 改善方策は合法手上で以下とする。

```text
pi_klent(a | s) = softmax_legal(
    (Q_mode(s, a) + β * log(max(p_prior(a | s), ε))) / (α + β)
)
```

`α` は entropy regularization、`β` は prior policy からの乖離抑制を表す。

自己対局では、CPU tactical solver 補正後の `pi_search_post_tactical` に温度をかけた `pi_select` から sampling する。評価対局では、補正後分布の argmax を使ってよい。

---

## 2. CPU tactical solver との合成

KLENT-0 は探索を行わないため、短期戦術の穴がそのまま負けにつながる危険がある。

将棋では特に以下を CPU 側で補う。

```text
相手玉への詰み探索
王手対応
宣言勝ち可能判定
合法手生成・move適用の異常検出
```

合法手生成・move適用の異常は、通常の終局理由ではなくバグとして扱う。発生した対局は学習 target に使わず、エラーログへ送る。


着手選択の順序は以下とする。

```text
1. 合法手を生成する
2. terminal / 宣言勝ち可能なら即処理する
3. 宣言勝ち・詰み探索・王手対応で強制手を抽出する
4. policy / Q から pi_klent を作る
5. tactical solver の結果で pi_klent を方策分布として補正する
   Q_mode は network / search 由来の値として保存する
6. sampling または argmax で着手する
```

補正は単純でよい。

```text
勝ち確定手:
  その手を選ぶ
```

詰み探索の結果を学習に使う場合は、補正後の方策分布を `policy` target として保存する。

---

## 3. 学習 phase

### 3.1 Phase 0: 教師あり初期化

目的は、KLENT 自己対局を始められる最低限の policy と Q を作ることである。

主な教師は以下である。

```text
既存Yoltsの探索手
簡易αβ探索の手
詰み探索で確定した手
過去の自己対局棋譜
外部エンジンの固定深さ探索
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

`policy_normal_head` は通常棋力を最優先して学習する。

`policy_style_head` と `q_nyugyoku_head` は、通常 target または入玉関連局面から弱く学習する。

### 3.2 Phase 1: KLENT-0 normal self-play

通常モード中心で自己対局する。

```text
α_style_self = 0.0 〜 0.1
α_style_opponent_model = 0.0 〜 0.1
λ_nyugyoku = 1.0
```

この phase の目的は、通常棋力を安定させることである。

開始局面は `docs/09_start_position_diversity.md` の phase schedule に従う。初期は平手初期局面を中心にしつつ、2手ランダム局面と将棋81万風シャッフル局面を cold start 補助として混ぜる。position buffer が未成熟な段階では、buffer 由来局面は少量または無効にする。

保存する主なデータは以下である。

```text
局面特徴
合法手
p_prior
pi_klent_pre_tactical
pi_search_post_tactical
pi_select
選択手
Q_normal_result
Q_nyugyoku
Q_total_result
Q_mode
終局結果
終局原因
moves_left
```

### 3.3 Phase 2: 入玉関連局面の増強

通常モードの自己対局を維持しつつ、position buffer 由来の入玉関連局面からの再開を混ぜる。シャッフル局面の比率は段階的に下げ、入玉サンプル不足はシャッフル率ではなく position buffer の入玉系 category quota で補う。

対象局面は以下である。

```text
自玉が敵陣に近い局面
相手玉が自陣に近い局面
宣言点数が増え始めた局面
相入玉になりやすい局面
P_win_nyugyoku または Q_nyugyoku が高い局面
P_loss_nyugyoku または Q_nyugyoku が低い局面
```

探索設定の例は以下である。

```text
入玉優先:
  α_style_self = 0.3 〜 0.5
  α_style_opponent_model = 0.0 〜 0.2
  λ_nyugyoku = 1.5 〜 3.0

相入玉想定:
  α_style_self = 0.3 〜 0.5
  α_style_opponent_model = 0.3 〜 0.5
  λ_nyugyoku = 1.5 〜 3.0

相手入玉警戒:
  α_style_self = 0.0 〜 0.2
  α_style_opponent_model = 0.3 〜 0.6
  λ_nyugyoku = 1.5 〜 3.0
```

`λ_nyugyoku` は、自分の入玉勝ちを強めると同時に、相手入玉による負けを強く嫌う。

### 3.4 Phase 3: Gumbel MCTS reanalyse

Gumbel MCTS は、全局面ではなく再解析対象局面に使う。

対象局面は以下である。

```text
policy entropy が高い局面
Q の上位手が僅差の局面
入玉関連局面
評価が不安定な終盤局面
通常勝ちと入玉勝ちの選択が割れる局面
自己対局で悪手が疑われる局面
```

reanalyse では以下を作る。

```text
pi_reanalyse
Q_normal_result_reanalyse
Q_nyugyoku_reanalyse
outcome_cause_reanalyse
```

`pi_reanalyse` は policy target として使う。

`Q_normal_result_reanalyse` と `Q_nyugyoku_reanalyse` は action-value target として使う。

### 3.5 Phase 4: 実戦時探索

実戦時には、計算資源に応じて以下を選ぶ。

```text
KLENT-0 + CPU tactical solver
Gumbel MCTS small budget
Gumbel MCTS medium budget
Gumbel MCTS large budget
```

実戦時探索は、学習用自己対局よりも強さ重視でよい。

---

## 4. 学習ターゲット

### 4.1 outcome_cause target

終局結果は、局面時点の手番から見て以下に変換する。

```text
win_normal
win_nyugyoku
draw
loss_nyugyoku
loss_normal
```

変換規則は `docs/05_outputs.md` に従う。

### 4.2 q target

`q_normal_result` と `q_nyugyoku` は、終局原因から以下の target を作る。

| outcome_cause | q_normal_result | q_nyugyoku |
| --- | ---: | ---: |
| `win_normal` | +1 | 0 |
| `win_nyugyoku` | 0 | +1 |
| `draw` | 0 | 0 |
| `loss_nyugyoku` | 0 | -1 |
| `loss_normal` | -1 | 0 |

実際に選んだ手だけに target がある場合は、選択手に対して学習する。

再解析で全合法手の target がある場合は、合法手上で学習する。

### 4.3 λ-return

1局の最終結果だけを target にすると分散が大きい。

そのため、自己対局中の network 推定値を使って λ-return を作ってよい。

```text
G_t = r_t + γ * ((1 - λ) * V(s_{t+1}) + λ * G_{t+1})
```

入玉系は通常勝敗と分けて return を作る。

```text
G_normal_result
G_nyugyoku
```

### 4.4 policy target

policy target は以下を使う。

```text
pi_search_post_tactical
pi_reanalyse
教師探索の訪問回数分布
教師手 one-hot
```

自己対局の標準 target は、CPU tactical solver 補正後の `pi_search_post_tactical` とする。
`pi_klent_pre_tactical` は解析用および ablation 用に保存する。
自己対局で使った sampling 温度と、学習 target として保存する分布の温度は分けてよい。

---

## 5. 探索モード定義

探索設定は以下の構造体相当で持つ。

```python
@dataclass
class SearchModeConfig:
    alpha_style_self: float
    alpha_style_opponent_model: float
    lambda_nyugyoku: float
    klent_alpha_entropy: float
    klent_beta_prior: float
    temperature: float
```

代表的な設定は以下である。

```text
normal:
  alpha_style_self = 0.0 〜 0.1
  alpha_style_opponent_model = 0.0 〜 0.1
  lambda_nyugyoku = 1.0

nyugyoku:
  alpha_style_self = 0.3 〜 0.5
  alpha_style_opponent_model = 0.0 〜 0.2
  lambda_nyugyoku = 1.5 〜 3.0

anti_nyugyoku:
  alpha_style_self = 0.0 〜 0.2
  alpha_style_opponent_model = 0.3 〜 0.6
  lambda_nyugyoku = 1.5 〜 3.0

mutual_nyugyoku:
  alpha_style_self = 0.3 〜 0.5
  alpha_style_opponent_model = 0.3 〜 0.5
  lambda_nyugyoku = 1.5 〜 3.0

bakuretsu:
  alpha_style_self = 0.2 〜 0.5
  alpha_style_opponent_model = 0.1 〜 0.4
  lambda_nyugyoku = 1.2 〜 2.5
```

---

## 6. outcome 分布との整合性

`outcome_cause_head` は局面単位の calibration に使う。

action-value は合法手ごとの着手選択に使う。

両者は同じ終局原因体系に従う。

```text
V_normal_result = P_win_normal - P_loss_normal
V_nyugyoku      = P_win_nyugyoku - P_loss_nyugyoku
V_total_result  = V_normal_result + V_nyugyoku

Q_total_result  = Q_normal_result + Q_nyugyoku
```

自己対局ログでは、局面単位の `V_*` と選択手の `Q_*` の両方を記録する。

---

## 7. 自己対局ログ仕様

自己対局1局には、最低限以下を保存する。

```python
@dataclass
class SelfPlayStep:
    features_ref: str
    side_to_move: int
    legal_moves: list[int]
    selected_move: int

    p_prior: np.ndarray
    pi_klent_pre_tactical: np.ndarray
    pi_search_post_tactical: np.ndarray
    pi_select: np.ndarray

    q_normal_result_pred: np.ndarray
    q_nyugyoku_pred: np.ndarray
    q_total_result_pred: np.ndarray
    q_mode_pred: np.ndarray

    q_normal_result_selected: float
    q_nyugyoku_selected: float
    q_total_result_selected: float
    q_mode_selected: float

    outcome_probs: np.ndarray
    moves_left_pred: float

    search_config_id: str
    tactical_flags: dict
```

`q_total_result_pred` と `q_mode_pred` は解析用に保存してよい派生値である。target 生成と視点変換の正本は `q_normal_result_pred` と `q_nyugyoku_pred` に分けて保持する。

終局時には以下を保存する。

```python
@dataclass
class GameResult:
    winner: int | None
    terminal_reason: str
    terminal_ply: int
    is_nyugyoku_win: bool
    is_draw: bool
```

学習用サンプル生成時に、各 step に以下を付与する。

```text
outcome_cause_target
q_normal_result_target
q_nyugyoku_target
moves_left_target
policy_normal_target
policy_style_target
```

`p_prior`、`pi_klent_pre_tactical`、`pi_search_post_tactical`、`pi_select` は全指し手 dense で保存してもよい。ファイルサイズが問題になる場合は、合法手だけの sparse 表現にする。

policy target の標準は `pi_search_post_tactical` である。`pi_klent_pre_tactical` は tactical solver 補正前の解析用、`pi_select` は温度・noise 適用後の実際の着手選択ログとして扱う。

---

## 8. Gumbel MCTS reanalyse

reanalyse は、KLENT-0 の弱点を補うために使う。

対象局面は以下の優先度で選ぶ。

```text
1. 入玉関連局面
2. 評価が不安定な局面
3. policy entropy が高い局面
4. Q上位手が僅差の局面
5. 自己対局で負けにつながった疑いのある局面
```

reanalyse の出力は以下である。

```text
pi_reanalyse
Q_normal_result_reanalyse
Q_nyugyoku_reanalyse
outcome_cause_reanalyse
```

MCTS 内部でも、通常勝敗価値と入玉関連価値を分けて保持する。

```text
backup_normal_result
backup_nyugyoku
```

合成 value だけを保存せず、視点と成分が分かる形でログに残す。

---

## 9. evaluation

評価では、通常棋力と入玉棋風を分けて観測する。

主な指標は以下である。

```text
通常勝率
入玉宣言勝ち率
相手入玉負け率
入玉有望局面からの勝率
相入玉局面での勝率
通常勝ちできる局面で無理に入玉しない率
詰み逃し率
KLENT-0 と Gumbel MCTS の差
```

学習中は以下を継続的にログする。

```text
policy entropy
KL(pi_klent || p_prior)
Q_normal_result calibration
Q_nyugyoku calibration
outcome_cause calibration
moves_left error
```

---

## 10. 失敗パターンと対策

### 10.1 policy collapse

症状:

```text
pi_klent が少数の手に極端に集中する
自己対局の局面多様性が落ちる
同じ序盤ばかり指す
```

対策:

```text
temperature を上げる
klent_alpha_entropy を上げる
klent_beta_prior を上げる
探索温度・root noise を調整し、開始局面多様化を強める（2手ランダム / 将棋81万風シャッフル / position buffer 再開率と category quota を調整する）
```

### 10.2 Q の過信

症状:

```text
Q が極端な値を出す
明らかな悪手を高く評価する
自己対局で短手数負けが増える
```

対策:

```text
Q loss weight を調整する
value clipping を入れる
教師探索データを増やす
Gumbel MCTS reanalyse を増やす
CPU tactical solver の mask を強める
```

### 10.3 通常棋力の劣化

症状:

```text
入玉を狙いすぎて通常勝ちを逃す
寄せを放棄する
序中盤で駒損を許容しすぎる
```

対策:

```text
normal vs normal の比率を上げる
α_style_self を下げる
λ_nyugyoku を下げる
policy_normal loss weight を上げる
通常勝ち局面の再解析を増やす
```

### 10.4 相手入玉への鈍感さ

症状:

```text
相手玉の進出を止めない
相手の宣言点数増加を軽視する
P_loss_nyugyoku が低く出る
Q_nyugyoku が相手入玉負けを十分に負側へ出さない
```

対策:

```text
相手入玉局面からの再開を増やす
anti_nyugyoku 設定の自己対局を増やす
λ_nyugyoku を上げる
q_nyugyoku の loss weight を上げる
相手入玉負けサンプルを oversampling する
```

---

## 11. 実装上の不変条件

以下は実装上の不変条件である。

```text
policy は合法手上でのみ正規化する
違法手確率は必ず0にする
Q_normal_result と Q_nyugyoku は現在手番視点で定義する
Q_mode だけを保存して視点変換しない
outcome_cause class order は固定する
terminal node はネットワーク評価せずルールで処理する
CPU tactical solver の強制手は KLENT 方策より優先する
```

---

## 12. 暫定結論

爆裂玉登太郎の主学習ループは以下とする。

```text
KLENT-0 自己対局
+
CPU tactical solver
+
入玉関連局面バッファ
+
Gumbel MCTS reanalyse
```

着手選択の中心は以下である。

```text
p_prior = blend(policy_normal, policy_style)
Q_mode  = Q_normal_result + λ_nyugyoku * Q_nyugyoku
pi_klent = KLENT_policy_improvement(p_prior, Q_mode)
```

通常棋力と入玉棋風は、policy head、action-value 成分、自己対局設定、評価指標を分けて管理する。

# docs/11_move_label.md - 指し手ラベル仕様

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎で使う **指し手ラベル** の仕様を定義する。

対象は以下である。

```text
MOVE_LABEL_PLANES
MOVE_LABEL_SIZE
現在手番視点での座標変換
盤上移動の label 化
成り・不成の扱い
持ち駒打ちの label 化
合法手 mask
policy / action-value head との接続
自己対局ログ・学習 target との接続
実装上の不変条件
検証項目
```

爆裂玉登太郎の network は、policy と action-value を固定長の `[B, MOVE_LABEL_SIZE]` として出力する。

そのため、将棋の合法手を、局面に依存しない固定長 index へ写像する必要がある。

この仕様では、dlshogi 系で広く使われる **着手先 81 マス × 27 move label plane** の形式を採用する。

```text
MOVE_LABEL_PLANES = 27
MOVE_LABEL_SIZE = 81 * 27 = 2187
```

この fixed label は、通常の盤上移動、成り移動、持ち駒打ちを表す。

合法手かどうかは label 仕様では判定しない。合法手生成、二歩、打ち歩詰め、行き所のない駒、王手放置、千日手、宣言勝ちなどのルール処理は、move generator / rule engine / search 側で扱う。

---

## 1. 基本方針

### 1.1 現在手番視点で符号化する

指し手ラベルは、絶対先手視点ではなく **現在手番視点** で符号化する。

```text
手番側から見て前方が常に同じ向きになる
手番側の前進は常に UP 系の direction になる
```

これにより、以下の network 仕様と揃える。

```text
入力特徴量は現在手番視点
policy は現在手番が指す手の分布
Q_normal_result は現在手番側の通常勝敗価値
Q_nyugyoku は現在手番側の入玉勝ちを正、相手入玉負けを負
outcome_cause は現在手番視点
```

### 1.2 fixed size にする

policy と action-value は同じ指し手ラベル空間を使う。

```python
MOVE_LABEL_PLANES = 27
BOARD_SIZE = 9
BOARD_SQUARES = 81
MOVE_LABEL_SIZE = MOVE_LABEL_PLANES * BOARD_SQUARES
```

network は、全 label に対して logits または raw value を出す。

```python
policy_normal_logits: Tensor  # [B, MOVE_LABEL_SIZE]
policy_style_logits: Tensor   # [B, MOVE_LABEL_SIZE]
q_normal_result_raw: Tensor   # [B, MOVE_LABEL_SIZE]
q_nyugyoku_raw: Tensor        # [B, MOVE_LABEL_SIZE]
```

合法手以外の label は出力されるが、探索側・loss 側で mask する。

### 1.3 label は通常着手だけを表す

`MOVE_LABEL_SIZE` に含めるのは、盤上移動と持ち駒打ちだけである。

```text
含める:
  盤上移動
  盤上移動 + 成り
  持ち駒打ち

含めない:
  宣言勝ち action
  投了
  千日手成立
  持将棋引き分け
  最大手数到達
  illegal move marker
```

宣言勝ちは、現局面から厳密に判定できる rule action であり、network policy / action-value の通常着手 label には含めない。

宣言勝ち可能局面では、探索側が `terminal_action = "declaration_win"` として処理する。

### 1.4 label version を固定する

指し手ラベルは checkpoint と training dataset の互換性に直結する。

初期実装の version は以下とする。

```text
move_label_version = "move_label_v1_dlshogi27_current_side_plane_major"
```

以下を変更する場合は、必ず `move_label_version` を変える。

```text
MOVE_LABEL_PLANES
plane order
index order
現在手番視点の座標変換
宣言勝ち action を label に含めるかどうか
```

---

## 2. 座標系

### 2.1 絶対座標

エンジン内部の盤面座標は、将棋の通常表記に合わせて以下で表す。

```python
file_abs: int  # 1..9
rank_abs: int  # 1..9
```

`file_abs` と `rank_abs` は、SFEN / USI との相互変換に使う絶対座標である。

### 2.2 現在手番視点座標

指し手 label では、絶対座標を現在手番視点へ変換して使う。

```python
BLACK = 0
WHITE = 1

def square_to_current_view(file_abs: int, rank_abs: int, side_to_move: int) -> tuple[int, int]:
    if side_to_move == BLACK:
        return file_abs, rank_abs
    return 10 - file_abs, 10 - rank_abs
```

この変換により、後手番では盤面を180度回転した座標で扱う。

逆変換は以下である。

```python
def square_from_current_view(file_cs: int, rank_cs: int, side_to_move: int) -> tuple[int, int]:
    if side_to_move == BLACK:
        return file_cs, rank_cs
    return 10 - file_cs, 10 - rank_cs
```

`cs` は `current-side` を表す。

### 2.3 square index

現在手番視点の `file_cs, rank_cs` から、0始まりの square index を作る。

```python
def square_index_current_view(file_cs: int, rank_cs: int) -> int:
    assert 1 <= file_cs <= 9
    assert 1 <= rank_cs <= 9
    return (rank_cs - 1) * 9 + (file_cs - 1)
```

逆変換は以下である。

```python
def square_from_index_current_view(square_index: int) -> tuple[int, int]:
    assert 0 <= square_index < 81
    rank_cs = square_index // 9 + 1
    file_cs = square_index % 9 + 1
    return file_cs, rank_cs
```

### 2.4 前方・左右の定義

現在手番視点では、手番側の前方を rank が小さくなる方向とする。

```text
UP    : rank_cs が -1
DOWN  : rank_cs が +1
LEFT  : file_cs が +1
RIGHT : file_cs が -1
```

ここでの LEFT / RIGHT は、手番側から見た左右である。

---

## 3. move label plane

### 3.1 plane 一覧

`MOVE_LABEL_PLANES = 27` とする。

| plane | name | 内容 |
| ---: | --- | --- |
| 0 | `UP` | 前方向 |
| 1 | `UP_LEFT` | 前左 |
| 2 | `UP_RIGHT` | 前右 |
| 3 | `LEFT` | 左 |
| 4 | `RIGHT` | 右 |
| 5 | `DOWN` | 後ろ |
| 6 | `DOWN_LEFT` | 後ろ左 |
| 7 | `DOWN_RIGHT` | 後ろ右 |
| 8 | `UP2_LEFT` | 桂馬の前左跳び |
| 9 | `UP2_RIGHT` | 桂馬の前右跳び |
| 10 | `PROMOTE_UP` | 前方向 + 成り |
| 11 | `PROMOTE_UP_LEFT` | 前左 + 成り |
| 12 | `PROMOTE_UP_RIGHT` | 前右 + 成り |
| 13 | `PROMOTE_LEFT` | 左 + 成り |
| 14 | `PROMOTE_RIGHT` | 右 + 成り |
| 15 | `PROMOTE_DOWN` | 後ろ + 成り |
| 16 | `PROMOTE_DOWN_LEFT` | 後ろ左 + 成り |
| 17 | `PROMOTE_DOWN_RIGHT` | 後ろ右 + 成り |
| 18 | `PROMOTE_UP2_LEFT` | 桂馬の前左跳び + 成り |
| 19 | `PROMOTE_UP2_RIGHT` | 桂馬の前右跳び + 成り |
| 20 | `DROP_PAWN` | 歩打ち |
| 21 | `DROP_LANCE` | 香打ち |
| 22 | `DROP_KNIGHT` | 桂打ち |
| 23 | `DROP_SILVER` | 銀打ち |
| 24 | `DROP_GOLD` | 金打ち |
| 25 | `DROP_BISHOP` | 角打ち |
| 26 | `DROP_ROOK` | 飛打ち |

この順序は固定であり、途中で変更してはならない。

### 3.2 direction vector

各 direction の current-side view における `(dfile, drank)` は以下である。

| base plane | name | dfile | drank |
| ---: | --- | ---: | ---: |
| 0 | `UP` | 0 | -1 |
| 1 | `UP_LEFT` | +1 | -1 |
| 2 | `UP_RIGHT` | -1 | -1 |
| 3 | `LEFT` | +1 | 0 |
| 4 | `RIGHT` | -1 | 0 |
| 5 | `DOWN` | 0 | +1 |
| 6 | `DOWN_LEFT` | +1 | +1 |
| 7 | `DOWN_RIGHT` | -1 | +1 |
| 8 | `UP2_LEFT` | +1 | -2 |
| 9 | `UP2_RIGHT` | -1 | -2 |

通常移動では base plane 0..9 を使う。

成り移動では、対応する base plane に `PROMOTE_OFFSET = 10` を足す。

```python
PROMOTE_OFFSET = 10
promote_plane = base_plane + PROMOTE_OFFSET
```

持ち駒打ちは direction を持たず、drop plane 20..26 を使う。

### 3.3 piece type と drop plane

持ち駒打ちの piece type と plane の対応は以下である。

```python
DROP_PLANE_BY_PIECE = {
    PAWN:   20,
    LANCE:  21,
    KNIGHT: 22,
    SILVER: 23,
    GOLD:   24,
    BISHOP: 25,
    ROOK:   26,
}
```

玉、成駒、と金、馬、龍を打つことはできないため、drop plane には含めない。

---

## 4. label index

### 4.1 plane-major order

label index は plane-major order とする。

```python
label = plane * 81 + to_square_index
```

逆変換は以下である。

```python
plane = label // 81
to_square_index = label % 81
```

したがって、範囲は以下である。

```text
0 <= plane < 27
0 <= to_square_index < 81
0 <= label < 2187
```

### 4.2 network tensor との対応

policy / q head は、内部的には以下の plane 形式で出してよい。

```python
move_plane_logits: Tensor  # [B, MOVE_LABEL_PLANES, 9, 9]
```

flatten した `[B, MOVE_LABEL_SIZE]` との対応は以下である。

```python
logits_flat = move_plane_logits.reshape(B, MOVE_LABEL_SIZE)
label = plane * 81 + rank_index * 9 + file_index
```

`rank_index = rank_cs - 1`、`file_index = file_cs - 1` である。

PyTorch の標準的な contiguous `[B, C, H, W]` flatten と同じ順序にする。

### 4.3 square-major order は使わない

以下の index order は標準では使わない。

```python
label = to_square_index * MOVE_LABEL_PLANES + plane
```

これを使うと、Conv1x1 head の `[B, plane, rank, file]` flatten と対応しにくくなるためである。

---

## 5. 盤上移動の encode

### 5.1 入力

盤上移動の encode は、少なくとも以下を受け取る。

```python
@dataclass(frozen=True)
class Move:
    from_square: Square | None
    to_square: Square
    drop_piece: PieceType | None
    promote: bool
```

盤上移動では以下を満たす。

```text
from_square is not None
drop_piece is None
```

### 5.2 現在手番視点への変換

まず、from / to を現在手番視点へ変換する。

```python
from_file_cs, from_rank_cs = square_to_current_view(
    move.from_square.file,
    move.from_square.rank,
    side_to_move,
)

to_file_cs, to_rank_cs = square_to_current_view(
    move.to_square.file,
    move.to_square.rank,
    side_to_move,
)
```

差分を計算する。

```python
dfile = to_file_cs - from_file_cs
drank = to_rank_cs - from_rank_cs
```

### 5.3 base direction の決定

`dfile, drank` から base plane 0..9 を決める。

桂馬は特殊ケースとして先に扱う。

```python
if (dfile, drank) == (+1, -2):
    base_plane = UP2_LEFT
elif (dfile, drank) == (-1, -2):
    base_plane = UP2_RIGHT
else:
    base_plane = direction_from_line_delta(dfile, drank)
```

`direction_from_line_delta` は、縦・横・斜めの直線方向だけを受け付ける。

```python
def direction_from_line_delta(dfile: int, drank: int) -> int:
    if dfile == 0 and drank < 0:
        return UP
    if dfile == 0 and drank > 0:
        return DOWN
    if drank == 0 and dfile > 0:
        return LEFT
    if drank == 0 and dfile < 0:
        return RIGHT
    if abs(dfile) == abs(drank):
        if dfile > 0 and drank < 0:
            return UP_LEFT
        if dfile < 0 and drank < 0:
            return UP_RIGHT
        if dfile > 0 and drank > 0:
            return DOWN_LEFT
        if dfile < 0 and drank > 0:
            return DOWN_RIGHT
    raise ValueError("move is not representable by move_label_v1")
```

歩、金、銀、玉、と金、成香、成桂、成銀、馬、龍などの1マス移動も、飛、角、香の長距離移動も、同じ direction plane を使う。

長距離移動の距離は label には持たない。距離と source square は、局面と着手先・direction から復元する。

### 5.4 成り flag

`move.promote == True` のとき、base plane に `PROMOTE_OFFSET` を足す。

```python
if move.promote:
    plane = base_plane + PROMOTE_OFFSET
else:
    plane = base_plane
```

すでに成っている駒の通常移動は、成り flag を持たないため base plane を使う。

```text
龍が1マス斜めへ動く: base plane
馬が1マス縦横へ動く: base plane
成銀が金として動く: base plane
```

`PROMOTE_*` plane は、「移動後に成る」ことだけを表す。

### 5.5 label 計算

最後に、to square index と plane から label を作る。

```python
to_square_index = square_index_current_view(to_file_cs, to_rank_cs)
label = plane * 81 + to_square_index
```

---

## 6. 持ち駒打ちの encode

### 6.1 入力

持ち駒打ちでは以下を満たす。

```text
from_square is None
drop_piece is not None
promote == False
```

打った直後に成ることはないため、drop move に成り flag は存在しない。

### 6.2 drop plane

`drop_piece` から drop plane を得る。

```python
plane = DROP_PLANE_BY_PIECE[move.drop_piece]
```

### 6.3 label 計算

to square を現在手番視点へ変換し、label を作る。

```python
to_file_cs, to_rank_cs = square_to_current_view(
    move.to_square.file,
    move.to_square.rank,
    side_to_move,
)

to_square_index = square_index_current_view(to_file_cs, to_rank_cs)
label = plane * 81 + to_square_index
```

### 6.4 drop の合法性

以下は label 仕様では判定しない。

```text
持ち駒が存在するか
打ち先が空か
二歩か
打ち歩詰めか
歩・香・桂の行き所があるか
王手放置にならないか
```

これらは legal move generator の責務である。

---

## 7. decode

### 7.1 decode には局面が必要

move label は、着手先と direction / drop type だけを持つ。

盤上移動の source square は label 単体からは分からないため、decode には局面が必要である。

```python
def decode_move_label(label: int, position: Position) -> Move:
    ...
```

`position.side_to_move` を使って、現在手番視点から絶対座標へ戻す。

### 7.2 安全な decode 方針

標準実装では、decode は合法手生成を使って行う。

```python
def decode_move_label(label: int, position: Position) -> Move:
    candidates = []
    for move in generate_legal_moves(position):
        if encode_move_label(move, position.side_to_move) == label:
            candidates.append(move)

    if len(candidates) != 1:
        raise InvalidMoveLabel(label, len(candidates))

    return candidates[0]
```

この方式は遅いが、正しさを優先できる。

高速化が必要な場合だけ、direction から source を逆探索する実装を追加する。

### 7.3 高速 decode の考え方

高速 decode では、以下を行う。

```text
1. label から plane と to_square を得る
2. drop plane なら drop move を作り、合法性を確認する
3. direction plane なら to_square から逆方向へ scan する
4. 最初に見つかった駒が現在手番側の駒で、その駒がその direction で動けるなら source とする
5. promotion flag を反映した move を作る
6. legal move generator または do/undo で最終合法性を確認する
```

縦・横・斜めの長距離移動では、to square から逆方向に見て最初に現れる駒だけが source になりうる。

途中に駒があれば、それより奥の駒は移動できないためである。

### 7.4 ambiguous decode はエラー

1つの局面で、複数の合法手が同じ label に encode されてはならない。

```text
同一局面内では、legal_moves の label は一意である
```

この不変条件が破れた場合は、move label 仕様または move generator のバグとして扱う。

---

## 8. 合法手 mask

### 8.1 legal move labels

探索側は、局面ごとに合法手を生成し、それぞれを label に変換する。

```python
def encode_legal_moves(position: Position) -> list[int]:
    labels = []
    for move in generate_legal_moves(position):
        labels.append(encode_move_label(move, position.side_to_move))
    assert len(labels) == len(set(labels))
    return labels
```

### 8.2 dense mask

policy / q head の dense 出力に対して、合法手 mask を作る。

```python
def make_legal_move_mask(legal_move_labels: list[int]) -> np.ndarray:
    mask = np.zeros(MOVE_LABEL_SIZE, dtype=np.bool_)
    for label in legal_move_labels:
        mask[label] = True
    return mask
```

合法手以外は、policy softmax や KLENT 方策計算で使ってはならない。

### 8.3 softmax_legal

policy logits から合法手上の分布を作る。

```python
def softmax_legal(logits: np.ndarray, legal_labels: list[int]) -> np.ndarray:
    masked_logits = np.full_like(logits, -np.inf)
    masked_logits[legal_labels] = logits[legal_labels]
    return softmax(masked_logits)
```

出力は dense `[MOVE_LABEL_SIZE]` でもよいが、違法手確率は必ず0にする。

### 8.4 sparse distribution

ログ保存と学習 target では、合法手だけを sparse にしてよい。

```python
@dataclass
class SparseMoveDistribution:
    moves: list[int]
    probs: list[float]
```

以下を必ず満たす。

```text
moves は合法手 label だけを含む
probs は moves と同じ順序である
sum(probs) == 1
同じ move label が重複しない
move label は現在手番視点である
```

### 8.5 sparse values

action-value のログと target は以下で保存してよい。

```python
@dataclass
class SparseMoveValues:
    moves: list[int]
    values: list[float]
```

以下を必ず満たす。

```text
moves は合法手 label だけを含む
values は moves と同じ順序である
q_normal_result と q_nyugyoku は別々に保存する
Q_mode だけを正本として保存しない
```

---

## 9. 成りの扱い

### 9.1 promote flag は move に従う

label の成り plane は、move object の `promote` flag に対応する。

```text
promote == False: base plane
promote == True : base plane + 10
```

### 9.2 成り可能でも不成なら base plane

成り可能な移動であっても、不成を選んだ場合は base plane を使う。

```text
銀が成れる位置へ不成で移動する: base plane
銀が成れる位置へ成って移動する: promote plane
```

同じ from / to に対して、成りと不成の2つの合法手が存在する場合、2つの異なる label を持つ。

### 9.3 成りが必須かどうかは label では判定しない

以下は legal move generator が判断する。

```text
歩・香が最終段へ進むときは成り必須
桂が最終2段へ進むときは成り必須
成れない駒は promote flag を持てない
成れる範囲に入っていない駒は promote flag を持てない
```

label encoder は、合法手 generator が出した move を前提にする。

encoder に不正な `promote=True` move が渡された場合はエラーにしてよい。

### 9.4 既成駒の移動

すでに成っている駒の移動は、promotion plane を使わない。

```text
龍が移動する: base plane
馬が移動する: base plane
と金が移動する: base plane
```

promotion plane は、移動によって新たに成る場合だけ使う。

---

## 10. 長距離駒の扱い

### 10.1 距離は label に含めない

飛、角、香、龍、馬の長距離移動では、距離は label に含めない。

```text
同じ direction と to_square を持つ移動は同じ plane を使う
source は局面から復元する
```

例:

```text
飛が同じ file 上を何マス進んでも、to_square が同じなら plane は UP または DOWN
角が斜めに何マス進んでも、to_square が同じなら対応する diagonal plane
香が前方に何マス進んでも、plane は UP
```

### 10.2 source の一意性

将棋の合法局面では、同じ direction と to_square に対し、合法な source は高々1つである。

逆方向に scan したとき、最初に現れる駒より奥の駒は、その駒を飛び越えられない。

この性質により、距離を label に含めなくても decode できる。

### 10.3 legal move generator による検証を必須にする

高速 decode を実装する場合でも、最終的には legal move generator または legality check で検証する。

```text
source が見つかること
その駒がその direction に動けること
path が空いていること
promote flag が合法であること
自玉を王手にさらさないこと
```

---

## 11. 宣言勝ち action

### 11.1 move label には含めない

宣言勝ちは、通常の盤上移動でも持ち駒打ちでもない。

そのため、`MOVE_LABEL_SIZE` には含めない。

```text
MOVE_LABEL_SIZE = 2187
宣言勝ち action 用に 2188 番目の label を追加しない
```

宣言勝ち可能判定は探索側の rule engine / tactical solver で行う。

### 11.2 SearchResult での扱い

宣言勝ちを選んだ場合、探索結果は通常着手ではなく terminal action として表す。

```python
@dataclass
class SearchResultActionFields:
    selected_move: int | None
    terminal_action: str | None  # None | "declaration_win"
```

通常着手では以下である。

```text
selected_move = move_label
terminal_action = None
```

宣言勝ちでは以下である。

```text
selected_move = None
terminal_action = "declaration_win"
```

既存の実装都合で sentinel を使う場合でも、sentinel は `0 <= label < MOVE_LABEL_SIZE` の範囲に入れてはならない。

```python
DECLARATION_ACTION_SENTINEL = -1
```

### 11.3 学習 target との関係

宣言勝ちは network policy の通常着手 target にはしない。

宣言勝ち可能局面では、探索側が確定処理するため、policy / q head が宣言勝ち action を出せる必要はない。

宣言勝ちで終局した対局の過去局面については、通常どおり outcome / q target を作る。

```text
宣言勝ちで勝った側から見た過去局面:
  outcome_cause_target = win_nyugyoku
  q_nyugyoku_target = +1

宣言勝ちで負けた側から見た過去局面:
  outcome_cause_target = loss_nyugyoku
  q_nyugyoku_target = -1
```

---

## 12. API

### 12.1 MoveLabelConfig

```python
@dataclass(frozen=True)
class MoveLabelConfig:
    version: str = "move_label_v1_dlshogi27_current_side_plane_major"

    board_size: int = 9
    board_squares: int = 81
    move_label_planes: int = 27
    move_label_size: int = 2187

    coordinate_view: str = "current_side_rotate_180"
    index_order: str = "plane_major"
    declaration_win_as_move_label: bool = False
```

### 12.2 MoveLabelEncoder

```python
class MoveLabelEncoder:
    config: MoveLabelConfig

    def encode_move(self, move: Move, side_to_move: int) -> int:
        ...

    def decode_label(self, label: int, position: Position) -> Move:
        ...

    def encode_legal_moves(self, position: Position) -> list[int]:
        ...

    def make_legal_mask(self, position: Position) -> np.ndarray:
        ...
```

### 12.3 dense / sparse conversion

```python
def dense_to_sparse_distribution(
    dense_probs: np.ndarray,
    legal_labels: list[int],
) -> SparseMoveDistribution:
    probs = [float(dense_probs[label]) for label in legal_labels]
    return SparseMoveDistribution(moves=legal_labels, probs=probs)
```

```python
def sparse_to_dense_distribution(
    sparse: SparseMoveDistribution,
) -> np.ndarray:
    dense = np.zeros(MOVE_LABEL_SIZE, dtype=np.float32)
    for label, prob in zip(sparse.moves, sparse.probs):
        dense[label] = prob
    return dense
```

`SparseMoveDistribution` を dense に戻したとき、違法手は0である。

---

## 13. network との接続

### 13.1 policy head

policy head は以下を出す。

```python
policy_normal_logits: Tensor  # [B, MOVE_LABEL_SIZE]
policy_style_logits: Tensor   # [B, MOVE_LABEL_SIZE]
```

探索側では、合法手 label だけを使って softmax する。

```text
p_normal = softmax_legal(policy_normal_logits, legal_labels)
p_style  = softmax_legal(policy_style_logits, legal_labels)
```

### 13.2 action-value head

action-value head は同じ label 空間を使う。

```python
q_normal_result_raw: Tensor  # [B, MOVE_LABEL_SIZE]
q_nyugyoku_raw: Tensor       # [B, MOVE_LABEL_SIZE]
```

探索側・loss 側で `tanh` を適用する。

```text
Q_normal_result = tanh(q_normal_result_raw)
Q_nyugyoku      = tanh(q_nyugyoku_raw)
```

合法手以外の Q は着手選択に使わない。

### 13.3 head flatten

Conv1x1 で plane を出す場合は、以下を標準にする。

```text
policy_adapter
  -> Conv1x1(out_channels = MOVE_LABEL_PLANES)
  -> [B, 27, 9, 9]
  -> flatten plane-major
  -> [B, 2187]
```

q head も同じ flatten order を使う。

```text
q_adapter
  -> Conv1x1(out_channels = MOVE_LABEL_PLANES)
  -> [B, 27, 9, 9]
  -> flatten plane-major
  -> [B, 2187]
```

policy head と q head で label order がずれると、探索・学習が破綻する。

---

## 14. 探索との接続

### 14.1 SearchRequest

探索は、局面から合法手を生成し、label に変換する。

```python
legal_moves = generate_legal_moves(position)
legal_labels = [encode_move_label(m, position.side_to_move) for m in legal_moves]
```

内部では `legal_labels` を使って policy / q を参照する。

### 14.2 SearchResult

通常着手を選んだ場合、`selected_move` は move label である。

```python
@dataclass
class SearchResult:
    selected_move: int | None
    legal_moves: list[int]
    pi_search: SparseMoveDistribution
    q_normal_result: SparseMoveValues | None
    q_nyugyoku: SparseMoveValues | None
    terminal_action: str | None = None
```

ここで `legal_moves` は、絶対 Move object ではなく move label の list である。

実際に盤面へ着手する直前に、label を `position` で decode する。

```python
move = decode_move_label(search_result.selected_move, position)
position.do(move)
```

### 14.3 tactical solver

CPU tactical solver も、探索分布へ反映するときは move label を使う。

```python
mate_win_moves: set[int]
declaration_win_moves: set[int]  # 通常 move label ではなく terminal action なら空でよい
winning_moves: set[int]
```

詰み手は通常着手なので move label で表す。

宣言勝ちは通常着手ではないため、`terminal_action` として扱う。

---

## 15. 自己対局ログとの接続

### 15.1 SelfPlayStep

自己対局 step では、合法手と選択手を current-side view の label で保存する。

```python
@dataclass
class SelfPlayStep:
    side_to_move: int
    sfen: str
    legal_moves: list[int]
    selected_move: int | None
    terminal_action: str | None

    p_prior: SparseMoveDistribution
    pi_klent_pre_tactical: SparseMoveDistribution
    pi_search_post_tactical: SparseMoveDistribution
    pi_select: SparseMoveDistribution

    q_normal_result_pred: SparseMoveValues | None
    q_nyugyoku_pred: SparseMoveValues | None

    move_label_version: str
```

`selected_move is None` かつ `terminal_action is not None` の step は、通常着手の policy target にはしない。

### 15.2 SFEN / USI との関係

ログには、人間可読性と再現性のために SFEN / 棋譜も保存してよい。

ただし、学習 target の正本は move label である。

```text
sfen:
  局面再生成用

usi move:
  デバッグ・棋譜出力用

move label:
  policy / q target 用
```

`move_label_version` が一致しない dataset を同一 training run に混ぜてはならない。

### 15.3 current-side label と side_to_move

move label は current-side view であるため、サンプルには必ず `side_to_move` と `sfen` を保存する。

```text
label だけでは絶対 move へ戻せない
sfen と side_to_move があって初めて decode できる
```

---

## 16. 学習 target との接続

### 16.1 policy target

policy target は、current-side view の move label 上の分布である。

```python
policy_normal_target: SparseMoveDistribution | None
policy_style_target: SparseMoveDistribution | None
```

以下を満たす。

```text
target.moves は legal_moves の部分集合である
target.probs の和は 1 である
target.moves は current-side view の move label である
```

### 16.2 q target

q target も同じ label 空間を使う。

```python
q_normal_result_target: SparseMoveValues | None
q_nyugyoku_target: SparseMoveValues | None
```

自己対局 Monte Carlo target では、通常は選択手だけに target を持つ。

```python
SparseMoveValues(moves=[selected_move], values=[q_value])
```

Gumbel MCTS reanalyse では、合法手全体に target を持ってよい。

```python
SparseMoveValues(moves=legal_moves, values=q_values)
```

非選択合法手の target が存在しない場合、それを0で埋めてはならない。

### 16.3 terminal action の target

宣言勝ちなどの terminal action は、policy target の move label には含めない。

宣言勝ちで終局した対局の過去 step は、終局原因に従って outcome / q target を作る。

terminal action そのものを policy head に学習させる設計にはしない。

---

## 17. 例

### 17.1 歩の前進

手番側の歩が1マス前へ進む場合、current-side view では以下になる。

```text
dfile = 0
drank = -1
base_plane = UP
```

不成なら `plane = UP`、成りなら `plane = PROMOTE_UP` である。

### 17.2 桂馬の跳び

手番側の桂馬が前左へ跳ぶ場合は以下である。

```text
dfile = +1
drank = -2
plane = UP2_LEFT
```

前右へ跳ぶ場合は以下である。

```text
dfile = -1
drank = -2
plane = UP2_RIGHT
```

### 17.3 角打ち

手番側が角を打つ場合、to square だけを使う。

```text
plane = DROP_BISHOP
label = DROP_BISHOP * 81 + to_square_index
```

### 17.4 後手の着手

後手番では、絶対座標を180度回転してから label を作る。

```python
file_cs = 10 - file_abs
rank_cs = 10 - rank_abs
```

これにより、後手の歩の前進も `UP` plane になる。

---

## 18. 実装上の不変条件

### 18.1 定数

以下は固定である。

```text
MOVE_LABEL_PLANES == 27
MOVE_LABEL_SIZE == 2187
PROMOTE_OFFSET == 10
DROP_PAWN == 20
DROP_ROOK == 26
```

### 18.2 range

すべての move label は以下を満たす。

```text
0 <= label < MOVE_LABEL_SIZE
```

宣言勝ち、投了、illegal move marker などの sentinel は、この範囲に入れてはならない。

### 18.3 legal labels uniqueness

任意の合法局面で、合法手 label は重複してはならない。

```python
labels = encode_legal_moves(position)
assert len(labels) == len(set(labels))
```

### 18.4 round trip

任意の合法手について、以下が成り立つ。

```python
label = encode_move_label(move, position.side_to_move)
decoded = decode_move_label(label, position)
assert decoded == move
```

### 18.5 current-side symmetry

局面と指し手を180度回転し、手番を反転した場合、label は同一になる。

```text
encode(move, black_position) == encode(rotated_move, rotated_white_position)
```

このテストにより、現在手番視点の変換漏れを検出する。

### 18.6 network head order

policy head と q head は、同じ label order を使う。

```text
policy_normal_logits[label]
policy_style_logits[label]
q_normal_result_raw[label]
q_nyugyoku_raw[label]
```

同じ `label` は同じ通常着手を表さなければならない。

---

## 19. 検証項目

### 19.1 unit test

必須 unit test は以下である。

```text
初期局面の全合法手 encode が範囲内である
初期局面の全合法手 label が一意である
初期局面の全合法手で encode -> decode が一致する
ランダム合法局面で encode -> decode が一致する
後手番局面で current-side 変換が正しい
成り・不成が別 label になる
持ち駒打ちが正しい drop plane になる
二歩などの違法打ちは legal mask に出ない
宣言勝ち action が MOVE_LABEL_SIZE 内に入らない
```

### 19.2 promotion test

以下を個別に確認する。

```text
成り可能位置への不成
成り可能位置への成り
成り必須手
成れない駒の移動
既成駒の移動
桂馬の成り
```

### 19.3 drop test

以下を個別に確認する。

```text
歩打ち
香打ち
桂打ち
銀打ち
金打ち
角打ち
飛打ち
二歩で歩打ちが legal mask に出ないこと
打ち歩詰めが legal mask に出ないこと
行き所のない駒打ちが legal mask に出ないこと
```

### 19.4 long-range test

以下を個別に確認する。

```text
飛の縦横移動
角の斜め移動
香の前方移動
龍の縦横移動と斜め1マス移動
馬の斜め移動と縦横1マス移動
途中に駒がある場合、奥の駒が source にならないこと
```

### 19.5 tensor order test

Conv1x1 head の plane と flatten label の対応を固定する。

```python
x = torch.zeros(1, 27, 9, 9)
x[0, plane, rank_index, file_index] = 1
flat = x.reshape(1, 2187)
assert flat[0, plane * 81 + rank_index * 9 + file_index] == 1
```

この test は、model 実装、loss 実装、探索実装のすべてで共有する。

---

## 20. ログ項目

指し手 label まわりで保存する項目は以下である。

```text
move_label_version
legal_moves
selected_move
terminal_action
p_prior
pi_klent_pre_tactical
pi_search_post_tactical
pi_select
q_normal_result
q_nyugyoku
q_total_result
q_mode
```

`legal_moves`、`selected_move`、各 sparse distribution / values の `moves` は、すべて current-side view の move label である。

デバッグ用に以下を追加保存してよい。

```text
selected_move_usi
legal_moves_usi
selected_move_plane
selected_move_to_square_index
```

これらは可読性のための補助情報であり、学習 target の正本ではない。

---

## 21. 実験項目

### 21.1 label order の実験は原則しない

label order は checkpoint と dataset の互換性に関わるため、通常の実験対象にはしない。

### 21.2 alternative label は別 version にする

以下のような alternative label を試す場合は、別 version として扱う。

```text
source square を含む巨大 action space
距離つき direction label
宣言勝ち action を label に含む設計
絶対先手視点 label
square-major order
```

初期実装では採用しない。

### 21.3 policy head 構造の実験

move label 仕様を変えずに、head 構造だけを変える実験は許可する。

```text
Conv1x1 plane head
Conv3x3 + Conv1x1 head
policy / q adapter の channel 数
policy_normal / policy_style の adapter 共有有無
```

この場合も、最終出力の `[B, 2187]` label order は変えてはならない。

---

## 22. まとめ

初期実装の指し手ラベルは以下で固定する。

```text
move_label_version = "move_label_v1_dlshogi27_current_side_plane_major"
MOVE_LABEL_PLANES = 27
MOVE_LABEL_SIZE = 2187
index = plane * 81 + to_square_index
座標 = 現在手番視点
宣言勝ち = move label には含めない
```

この仕様により、network head、探索、自己対局ログ、学習 target が同じ action space を共有できる。

合法手生成と rule action は探索側で扱い、network は固定長の通常着手 label に対して policy / action-value を出す。

# docs/00_name.md - 名称・コンセプト

## 0. このドキュメントの目的

このドキュメントでは、将棋AI **爆裂玉登太郎** の名称、読み、表記、略称、ログ・実装上の識別子、コンセプトを定義する。

対象は以下である。

```text
正式名称
読み
ローマ字表記
略称
内部ID / ログID
USI エンジン名
プロジェクトコンセプト
名前に込める設計意図
名称に関する不変条件
検証項目
未決事項
```

このドキュメントは、探索・学習・network・評価の詳細仕様の正本ではない。

関連仕様は以下を正とする。

```text
全体方針: 爆裂玉登太郎_設計概要.md
開発方針: docs/01_development_policy.md
全体アーキテクチャ: docs/02_overall_architecture.md
評価仕様: docs/14_evaluation.md
学習パイプライン: docs/15_training_pipeline.md
```

このドキュメントでは、プロジェクト名とコンセプトが、実装・ログ・評価・ドキュメント上でぶれないようにする。

---

## 1. 正式名称

### 1.1 日本語正式名称

正式名称は以下とする。

```text
爆裂玉登太郎
```

この名称は、ユーザー向け表示、ドキュメント見出し、評価レポート、リリースノートで使う標準表記である。

### 1.2 読み

読みは以下を第一候補として固定する。

```text
ばくれつ・たまのぼり・たろう
```

`玉登` は **たまのぼり** と読む。

`たまと` とは読まない。

### 1.3 ローマ字表記

ローマ字表記は以下を標準とする。

```text
Bakuretsu Tamanobori Taro
```

空白を使えない識別子では以下を使う。

```text
BakuretsuTamanoboriTaro
```

snake_case の識別子では以下を使う。

```text
bakuretsu_tamanobori_taro
```

略した内部コード名が必要な場合は、以下を使ってよい。

```text
Tamanobori
```

ただし、`Tamanobori` は内部コード名・短縮名であり、正式名称ではない。

---

## 2. 名前の意味

### 2.1 爆裂

`爆裂` は、既存の爆裂シリーズとの連続性を表す。

爆裂駒拾太郎から独立した新しい設計であっても、同じ系列の将棋AIとして扱えるようにするため、名称には `爆裂` を残す。

`爆裂` は以下を表す。

```text
既存シリーズとの連続性
実験的で攻めた設計
通常棋力だけでなく特徴的な棋風を持つAI
```

### 2.2 玉登

`玉登` は、玉が敵陣へ登っていく入玉のイメージを表す。

爆裂玉登太郎の中心コンセプトは、通常の将棋として十分に強いだけでなく、入玉宣言勝ちを自然に狙えることである。

`玉登` は以下を表す。

```text
自玉の入玉を長期計画として扱う
敵陣への玉の進出を評価できる
入玉宣言勝ちを単なる例外処理ではなく、勝ち方の一種として扱う
相手の入玉も早めに警戒する
```

### 2.3 太郎

`太郎` は、爆裂シリーズらしい親しみやすさを表す。

技術的には Policy / Action-Value / Outcome Network、KLENT-0、CPU tactical solver、Gumbel MCTS reanalyse などを使うが、名前自体は過度に硬くしない。

`太郎` は以下を表す。

```text
爆裂シリーズとしてのキャラクター性
覚えやすさ
通常の研究コード名ではなく、個別エンジンとしての識別しやすさ
```

---

## 3. コンセプト

### 3.1 一文での説明

爆裂玉登太郎は、**通常の将棋として十分に強く、なおかつ入玉宣言勝ちを自然に狙える将棋AI** である。

### 3.2 目指す性質

目指す性質は以下である。

```text
通常棋力を大きく損なわない
自玉の入玉を長期計画として扱える
相手の入玉リスクを早めに警戒できる
探索設定により棋風を切り替えられる
入玉宣言勝ちをルール上の例外ではなく、勝ち方の一種として扱える
```

### 3.3 目指さない性質

爆裂玉登太郎は、入玉だけを狙うネタエンジンではない。

以下は目指さない。

```text
通常棋力を犠牲にしてでも入玉だけを狙う
入玉できない局面でも不自然に玉を上げる
相手の即詰みや通常勝ちを軽視する
通常勝ちより入玉勝ちだけを常に優先する
ルールで厳密に判定できる宣言勝ちを network の推測に任せる
```

入玉宣言勝ちを自然に増やすことは重要だが、通常の将棋としての勝率と整合していなければならない。

---

## 4. 棋風の位置づけ

### 4.1 通常棋力

通常棋力は、普通の将棋として勝つ力である。

爆裂玉登太郎では、通常棋力を `policy_normal_head`、`q_normal_result_head`、通常モード評価、通常勝率 gate で管理する。

名称上の `玉登` は入玉棋風を強調するが、通常棋力を軽視する意味ではない。

### 4.2 入玉棋風

入玉棋風は、自玉の入玉を長期計画として扱い、条件が整えば入玉宣言勝ちを狙う性質である。

これは以下で実現する。

```text
policy_style_head
q_nyugyoku_head
入玉関連 outcome
探索モードごとの λ_nyugyoku
style 混合率
入玉関連局面の自己対局・再解析
```

名称上の `玉登` は、この棋風を端的に表す。

### 4.3 相手入玉耐性

爆裂玉登太郎は、自玉の入玉だけでなく、相手の入玉リスクも扱う。

相手入玉耐性は以下を含む。

```text
相手玉が自陣へ近づく兆候を検出する
相手の宣言勝ち条件を満たしにくくする
相手入玉による負けを q_nyugyoku / outcome で負の信号として扱う
anti_nyugyoku モードや mutual_nyugyoku モードで評価する
```

名称は自玉の入玉を強調しているが、相手入玉の防止も同じコンセプトに含める。

---

## 5. 表記ルール

### 5.1 ドキュメント上の表記

ドキュメントでは、初出では正式名称を使う。

```text
爆裂玉登太郎
```

同じ文脈で繰り返す場合は、以下の短縮表現を使ってよい。

```text
本エンジン
本プロジェクト
玉登太郎
```

ただし、仕様上の正式名称を定義する箇所では、常に `爆裂玉登太郎` を使う。

### 5.2 略称

略称は、文脈上明らかな場合に限って以下を使ってよい。

```text
玉登太郎
爆裂玉登
Tamanobori
```

略称を checkpoint 名、dataset 名、評価 run 名などの永続的な識別子に使う場合は、対応する正式名称を metadata に必ず保存する。

### 5.3 使わない表記

以下は標準表記として使わない。

```text
爆裂玉登り太郎
爆裂たまのぼり太郎
爆裂玉登太朗
爆裂玉登太郎AI
Bakuretsu Tamato Taro
Tamato
```

`玉登` の読みを `たまと` とする表記は使わない。

---

## 6. 実装・ログ上の識別子

### 6.1 EngineIdentity

実装では、名称情報を以下の構造体相当で管理する。

```python
@dataclass(frozen=True)
class EngineIdentity:
    display_name_ja: str = "爆裂玉登太郎"
    reading_ja: str = "ばくれつ・たまのぼり・たろう"
    display_name_ascii: str = "Bakuretsu Tamanobori Taro"
    engine_id: str = "bakuretsu_tamanobori_taro"
    short_id: str = "tamanobori"
    family_id: str = "bakuretsu"
```

`engine_id` は、ログ、dataset metadata、evaluation metadata、checkpoint metadata で使う安定識別子である。

### 6.2 USI エンジン名

USI の `id name` では、ASCII 互換性を優先して以下を標準とする。

```text
BakuretsuTamanoboriTaro
```

日本語表示が安全に扱える環境では、補助的に以下を表示してよい。

```text
爆裂玉登太郎
```

USI プロトコル、対局サーバ、評価スクリプト、ログ収集環境で文字化けやパース問題が起きる場合は、ASCII 名を正本として使う。

### 6.3 checkpoint metadata

checkpoint には、少なくとも以下を保存する。

```yaml
engine_identity:
  display_name_ja: "爆裂玉登太郎"
  reading_ja: "ばくれつ・たまのぼり・たろう"
  display_name_ascii: "Bakuretsu Tamanobori Taro"
  engine_id: "bakuretsu_tamanobori_taro"
  short_id: "tamanobori"
  family_id: "bakuretsu"
```

checkpoint の比較、評価、昇格、再現実験では、`engine_id` と version metadata を組み合わせて扱う。

### 6.4 evaluation metadata

評価ログでは、candidate と baseline を明確に区別する。

例:

```yaml
candidate:
  engine_id: "bakuretsu_tamanobori_taro"
  display_name_ja: "爆裂玉登太郎"
  checkpoint_id: "..."
  eval_mode_id: "normal_klent0"

baseline:
  engine_id: "bakuretsu_komahiro_taro"
  display_name_ja: "爆裂駒拾太郎"
  version: "V4.2.0"
  search: "depth4"
```

`爆裂玉登太郎` と `爆裂駒拾太郎` は別エンジンとして扱い、評価ログ上で混同してはならない。

---

## 7. 関連するモード名

### 7.1 mode_id は機能名として扱う

探索モードや評価モードの `mode_id` は、エンジン名とは分ける。

代表的な `mode_id` は以下である。

```text
normal
nyugyoku
anti_nyugyoku
mutual_nyugyoku
bakuretsu
```

`mode_id` は棋風・探索設定を表すものであり、エンジン名ではない。

### 7.2 bakuretsu mode

`bakuretsu` は探索モード名として使ってよい。

ただし、`bakuretsu` はエンジンファミリ名でもあるため、文脈を混同しないようにする。

```text
engine family: bakuretsu
engine id: bakuretsu_tamanobori_taro
mode id: bakuretsu
```

`mode_id = bakuretsu` は、通常棋力を残しつつ入玉価値をやや強く見る評価・探索設定を指す。

---

## 8. 名称に関する不変条件

### 8.1 正式名称を途中で変えない

大規模な自己対局、checkpoint、dataset、evaluation log を生成した後に正式名称を変えると、成果物の対応づけが崩れる。

したがって、以下は原則として変更しない。

```text
display_name_ja = 爆裂玉登太郎
reading_ja = ばくれつ・たまのぼり・たろう
engine_id = bakuretsu_tamanobori_taro
short_id = tamanobori
family_id = bakuretsu
```

どうしても変更する場合は、互換性を壊す変更として扱い、metadata version を上げる。

### 8.2 名前と棋風を乖離させない

`玉登` を名乗る以上、入玉関連の設計・評価を後付けの飾りにしてはならない。

少なくとも以下を維持する。

```text
入玉勝ちと通常勝ちを outcome 上で区別する
相手入玉負けを outcome / q_nyugyoku 上で区別する
入玉関連局面を自己対局・position buffer・reanalyse で扱う
勝率制約下で入玉宣言勝ち率を評価する
```

### 8.3 通常棋力を名前でごまかさない

`玉登` という名前は、通常棋力を犠牲にしてよいという意味ではない。

通常モードでは、通常棋力 gate を満たすことを目標にする。

入玉最適化では、勝率制約を満たした候補の中で入玉宣言勝ち率を最大化する。

---

## 9. 検証項目

### 9.1 metadata 検証

以下を検証する。

```text
checkpoint metadata に engine_identity が保存される
self-play log に engine_id が保存される
evaluation log に candidate / baseline の engine_id が保存される
dataset metadata から生成元 engine_id を追跡できる
USI id name が ASCII 名として安定して出る
```

### 9.2 表記揺れ検出

ドキュメント・ログ・設定ファイルで以下の誤表記が出ないことを確認する。

```text
爆裂玉登り太郎
爆裂玉登太朗
Bakuretsu Tamato Taro
Tamato
```

必要であれば、CI で簡易 grep を行う。

### 9.3 baseline との混同検出

評価ログでは、以下を検証する。

```text
candidate が bakuretsu_tamanobori_taro である
baseline が bakuretsu_komahiro_taro 系である
candidate と baseline の engine_id が同一になっていない
baseline version が明示されている
candidate checkpoint_id が明示されている
```

---

## 10. 未決事項

### 10.1 リリース時の日本語名表示

USI エンジン名として日本語名を併記するかは、実際に使う GUI、対局サーバ、評価スクリプトの互換性を見て決める。

標準の正本は ASCII の `BakuretsuTamanoboriTaro` とする。

### 10.2 ロゴ・アイコン

ロゴやアイコンは、このドキュメントでは定義しない。

将来的に作る場合は、以下のコンセプトに合わせる。

```text
爆裂シリーズらしさ
玉が敵陣へ登っていくイメージ
通常棋力と入玉棋風の両立
```

### 10.3 外部公開時の短縮名

外部公開時に短縮名を前面に出すかは未決とする。

ただし、正式名称は常に `爆裂玉登太郎` とし、短縮名だけで別エンジンのように扱わない。

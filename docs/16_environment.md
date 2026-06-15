# docs/16_environment.md - 環境構築とワークスペース運用

## 0. このドキュメントの目的

このドキュメントでは、爆裂玉登太郎の初期開発環境、品質チェック、実験ログの置き場を定義する。

対象は以下である。

```text
Rust workspace
Python / uv 環境
品質チェックコマンド
workspace/ の責務
実験ログの階層
コミット対象と非コミット対象の分離
```

---

## 1. 実装分担

初期環境は Rust と Python を分ける。

```text
Rust:
  将棋ルール、合法手、終局判定、宣言勝ち判定、move label、探索、USI engine

Python:
  network、dataset、training、checkpoint、evaluation、実験管理
```

Rust と Python の境界では、version、schema、artifact を明示する。

---

## 2. Rust workspace

Rust は Cargo workspace として管理する。

```text
crates/tamanobori-core:
  盤面、合法手、終局、宣言勝ち、move label、version 定数

crates/tamanobori-search:
  KLENT-0、CPU tactical solver 接続、SearchResult

crates/tamanobori-engine:
  USI engine binary

crates/tamanobori-tools:
  fixture 生成、schema 検証、評価補助 CLI
```

標準チェックは以下である。

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

まとめて実行する場合は以下を使う。

```powershell
.\scripts\check_rust.cmd
```

GitHub Actions でも同じ Rust gate を実行する。

---

## 3. Python / uv 環境

Python は `uv` で管理する。

標準チェック用スクリプトは、uv の cache を repo 内の `.uv-cache/` に置く。`.uv-cache/` は Git 管理しない。

標準チェックは以下である。

```powershell
uv run ruff format --check python tests/python
uv run ruff check python tests/python
uv run ty check python tests/python
uv run pytest
```

まとめて実行する場合は以下を使う。

```powershell
.\scripts\check_python.cmd
```

GitHub Actions でも同じ Python gate を実行する。

PyTorch は初期 scaffold の依存には定義しない。network 実装を入れる段階で、dependency group、optional extra、または実行環境別 lock のどれで扱うかを決めてから追加する。

---

## 4. workspace/ の責務

`workspace/` はローカル実験作業領域である。

人間が読む実験ログは Git 管理し、実行 artifact は Git 管理しない。

Git 管理するもの:

```text
workspace/README.md
workspace/logs/**
```

Git 管理しないもの:

```text
workspace/runs/
workspace/datasets/
workspace/checkpoints/
workspace/eval_artifacts/
workspace/reanalyse/
workspace/scratch/
```

---

## 5. 実験ログの階層

実験ログは `workspace/logs/` に置く。

```text
workspace/logs/experiments/
  通常の時系列実験ログ

workspace/logs/decisions/
  採用判断、設計判断、長期運用方針

workspace/logs/ablation/
  ablation 実験の比較結果

workspace/logs/evaluation/
  評価 suite ごとの要約

workspace/logs/failures/
  失敗、原因、戻し方

workspace/logs/templates/
  実験ログと判断ログの雛形
```

各ログには、目的、仮説、条件、結果、採用判断、次のアクションを残す。

---

## 6. 全体チェック

Rust と Python の標準チェックをまとめて実行する。

```powershell
.\scripts\check_all.cmd
```

初期環境整備では、このコマンドが通ることを M0 の最低条件にする。

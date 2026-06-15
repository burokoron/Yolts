# workspace

`workspace/` は、爆裂玉登太郎のローカル実験作業領域です。

このディレクトリでは、人間が読む実験ノートだけを Git 管理し、実行 artifact は管理しません。

## Tracked

- `logs/`: 実験内容、結果、採否判断、失敗分析を書く Markdown ログ

## Ignored

- `runs/`: run log や config snapshot
- `datasets/`: generated dataset shard
- `checkpoints/`: model checkpoint
- `eval_artifacts/`: 評価実行の raw artifact
- `reanalyse/`: reanalyse queue と結果
- `scratch/`: 一時検証用ファイル

大容量 artifact は必要に応じて repo 外へ移せますが、対応する実験ログには保存先と再現手順を残します。


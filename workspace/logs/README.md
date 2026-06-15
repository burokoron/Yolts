# Experiment Logs

`workspace/logs/` は、コードの実行出力ではなく、実験の意図、条件、結果、採否判断を書く場所です。

## Layout

- `experiments/`: 時系列の通常実験ログ
- `decisions/`: 採用判断や設計判断
- `ablation/`: ablation 実験の比較結果
- `evaluation/`: 評価 suite ごとの要約
- `failures/`: 失敗、原因、戻し方
- `templates/`: ログを書くときの雛形

## Minimum Fields

各ログには少なくとも以下を残します。

- 目的
- 仮説
- 条件
- 結果
- 採用判断
- 次のアクション


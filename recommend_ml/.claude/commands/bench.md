---
description: 指定モデルを標準グリッドで探索し、結果の比較表を出す
argument-hint: <model> [min_rating]
allowed-tools: Bash(./scripts/tune.sh:*), Bash(./scripts/report.sh:*), Bash(./scripts/status.sh:*)
---

`./scripts/tune.sh $1 $2` を実行する。モデルは item_knn / user_knn / ease / als / bpr のいずれか。
min_rating は省略可（省略時は全評価を正例とする NCF 系の流儀、3 を指定すると rating>=3 のみ）。

探索は数分から数十分かかる。実行は必ずバックグラウンドで開始し、完了を待たずに
`./scripts/status.sh` の内容を報告する。完了後は `./scripts/report.sh $2` の比較表を示し、
valid で選ばれた設定と test の数値がどう変わったかを一言添える。

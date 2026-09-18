---
description: 実行中の探索ジョブと進捗、現時点の結果表を確認する
allowed-tools: Bash(./scripts/status.sh:*), Bash(./scripts/report.sh:*)
---

`./scripts/status.sh` を実行して、実行中のジョブと各ログの進捗を報告する。
続けて `./scripts/report.sh` の比較表を示し、前回からの変化があれば指摘する。
長い出力をそのまま貼らず、要点（進捗の割合、最良設定、順位の変化）に絞って書く。

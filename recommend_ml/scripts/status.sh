#!/usr/bin/env bash
# 実行中の探索ジョブと、各ログの進捗を表示する。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 実行中のプロセス ==="
pgrep -af "recommend_ml" | grep -v "/bin/bash" || echo "なし"

echo
echo "=== ログの進捗 ==="
for log in results/logs/*.log; do
  [ -e "$log" ] || continue
  DONE=$(grep -c "^\[" "$log" || true)
  LAST=$(grep "^\[" "$log" | tail -1 || true)
  BEST=$(grep "^best on valid" "$log" | tail -1 || true)
  printf "%-40s %s件\n" "$(basename "$log")" "$DONE"
  [ -n "$LAST" ] && echo "    最新: $LAST"
  [ -n "$BEST" ] && echo "    最良: $BEST"
done

echo
echo "=== results.csv の行数 ==="
[ -e results/results.csv ] && wc -l < results/results.csv || echo "まだ無し"

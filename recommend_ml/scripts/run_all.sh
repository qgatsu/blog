#!/usr/bin/env bash
# 残りのモデルを順に探索する。
#
# 実行中の探索があれば、それが終わってから始める。tune.py は最後に
# results/results.csv を読み書きするため、同時に走らせると結果が壊れる。
set -uo pipefail
cd "$(dirname "$0")/.."

while pgrep -f "recommend_ml\.tune" >/dev/null; do
  sleep 15
done

for MR in "" "3"; do
  for MODEL in ease bpr als; do
    echo "===== $MODEL min_rating=${MR:-none} ====="
    ./scripts/tune.sh "$MODEL" $MR || echo "FAILED: $MODEL $MR"
  done
done
echo "===== 全て完了 ====="

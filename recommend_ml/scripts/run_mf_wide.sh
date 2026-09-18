#!/usr/bin/env bash
# ALS / BPR の再探索。前回の最良がグリッドの端 (factors=128) に張り付いたため、
# 範囲を広げて「端で頭打ちになっていないか」を確認する。
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "recommend_ml\.tune" >/dev/null; do sleep 15; done
for MR in "" "3"; do
  for MODEL in als bpr; do
    echo "===== $MODEL min_rating=${MR:-none} ====="
    ./scripts/tune.sh "$MODEL" $MR || echo "FAILED: $MODEL $MR"
  done
done
echo "===== 再探索 完了 ====="

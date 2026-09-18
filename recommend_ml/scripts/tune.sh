#!/usr/bin/env bash
# 指定モデルを標準グリッドで探索する。
#   ./scripts/tune.sh <model> [min_rating]
# 携帯から実行することを想定し、グリッドはここに固定で持つ。
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: ./scripts/tune.sh <model> [min_rating]}"
MIN_RATING="${2:-}"
MR_ARG=""
[ -n "$MIN_RATING" ] && MR_ARG="--min-rating $MIN_RATING"

case "$MODEL" in
  item_knn|user_knn)
    GRID='{"top_k":[3,5,10,20,50,100,200],"shrink":[0,10,100,500],"alpha":[0.0,0.25,0.5,0.75,1.0]}' ;;
  ease)
    GRID='{"regularization":[1,10,50,100,250,500,1000,2000]}' ;;
  als)
    GRID='{"factors":[128,256],"regularization":[0.001,0.01,0.1,1.0],"alpha":[0.1,0.5,1.0]}' ;;
  bpr)
    GRID='{"factors":[128,256],"learning_rate":[0.05,0.1],"regularization":[0.0001,0.001],"epochs":[100,200]}' ;;
  *)
    echo "unknown model: $MODEL (item_knn|user_knn|ease|als|bpr)" >&2; exit 1 ;;
esac

mkdir -p results/logs
LOG="results/logs/tune_${MODEL}_min${MIN_RATING:-none}.log"
echo "model=$MODEL min_rating=${MIN_RATING:-none} -> $LOG"
PYTHONPATH=src .venv/bin/python -m recommend_ml.tune --model "$MODEL" $MR_ARG --grid "$GRID" 2>&1 | tee "$LOG"

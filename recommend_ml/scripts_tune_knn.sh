#!/bin/bash
# 近傍法のグリッドサーチ。min_rating=None (NCF 系の流儀: 全評価を正例) と
# min_rating=3 (既存 02 との比較用) の両方で回す。
set -e
cd "$(dirname "$0")"
GRID='{"top_k":[3,5,10,20,50,100,200],"shrink":[0,10,100,500],"alpha":[0.0,0.25,0.5,0.75,1.0]}'
for MR in "" "--min-rating 3"; do
  for MODEL in item_knn user_knn; do
    echo "=== $MODEL $MR ==="
    PYTHONPATH=src .venv/bin/python -m recommend_ml.tune --model "$MODEL" $MR --grid "$GRID"
  done
done

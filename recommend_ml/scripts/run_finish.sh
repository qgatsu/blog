#!/usr/bin/env bash
# 残りの探索をまとめて片付ける。
#
#  1. BPR (全評価) : 前回の最良が factors=256 / epochs=200 と再び端に張り付いたため、
#                    さらに広げて頭打ちになる点を確認する。
#  2. ALS (rating>=3): 広げたグリッドで未実行。factors=256 は 1 設定 500 秒かかるので
#                    128 までに抑える。
#  3. BPR (rating>=3): 広げたグリッドで未実行。
set -uo pipefail
cd "$(dirname "$0")/.."
RUN() { PYTHONPATH=src .venv/bin/python -m recommend_ml.tune "$@"; }

echo "===== 1. bpr 全評価 (さらに拡張) ====="
RUN --model bpr \
  --grid '{"factors":[256,512],"learning_rate":[0.05],"regularization":[0.001],"epochs":[200,400]}' \
  2>&1 | tee results/logs/tune_bpr_minnone_wide.log

echo "===== 2. als rating>=3 ====="
RUN --model als --min-rating 3 \
  --grid '{"factors":[64,128],"regularization":[0.01,0.1,1.0],"alpha":[0.1,0.5,1.0]}' \
  2>&1 | tee results/logs/tune_als_min3.log

echo "===== 3. bpr rating>=3 ====="
RUN --model bpr --min-rating 3 \
  --grid '{"factors":[128,256],"learning_rate":[0.05],"regularization":[0.001],"epochs":[100,200]}' \
  2>&1 | tee results/logs/tune_bpr_min3.log

echo "===== 残りの探索 完了 ====="

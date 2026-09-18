#!/usr/bin/env bash
# 計算コストのスケーリング測定。時間を測るので、他のジョブが終わってから実行する。
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "recommend_ml\.(tune|sensitivity)" >/dev/null; do sleep 20; done
sleep 5
PYTHONPATH=src .venv/bin/python -m recommend_ml.scaling 2>&1 | tee results/logs/scaling.log
echo "===== スケーリング測定 完了 ====="

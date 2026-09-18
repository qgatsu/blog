#!/usr/bin/env bash
# 結果の比較表を出す。
#   ./scripts/report.sh [min_rating] [split]
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src .venv/bin/python -m recommend_ml.report \
  --min-rating "${1:-all}" --split "${2:-test}"

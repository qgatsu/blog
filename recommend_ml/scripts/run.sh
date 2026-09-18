#!/usr/bin/env bash
# 既定パラメータで手法を実行する。
#   ./scripts/run.sh <models> [min_rating]
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="${1:?usage: ./scripts/run.sh <models> [min_rating]}"
MIN_RATING="${2:-}"
MR_ARG=""
[ -n "$MIN_RATING" ] && MR_ARG="--min-rating $MIN_RATING"
PYTHONPATH=src .venv/bin/python -m recommend_ml.run --models "$MODELS" $MR_ARG

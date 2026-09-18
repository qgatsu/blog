#!/usr/bin/env bash
# MovieLens 1M を GroupLens から取得する。
#
# データはリポジトリに含めない。ML-1M の利用条件は再配布を許可していないため
# (同梱 README の "may not redistribute the data without separate permission")、
# 各自がここから取得する形にしている。
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="data/ml-1m"
URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"

if [ -e "$DEST/ratings.dat" ]; then
  echo "既に $DEST/ratings.dat があります"
  exit 0
fi

mkdir -p data
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "取得中: $URL"
curl -fsSL -o "$TMP/ml-1m.zip" "$URL"
unzip -q "$TMP/ml-1m.zip" -d "$TMP"
rm -f "$DEST"            # 既存のシンボリックリンクがあれば外す
mv "$TMP/ml-1m" "$DEST"
echo "展開しました -> $DEST"
ls "$DEST"

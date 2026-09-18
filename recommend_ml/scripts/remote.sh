#!/usr/bin/env bash
# Remote Control セッションを tmux 上で起動する。
#
# tmux を挟む理由: Remote Control は「このマシンで動いているセッション」を
# 携帯から操作する仕組みなので、端末を閉じてもセッションが生き続ける必要がある。
# すでに同名の tmux セッションがあれば、新規作成せずそれに接続する。
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${1:-recommend-ml}"

find_claude() {
  if command -v claude >/dev/null 2>&1; then
    command -v claude
    return
  fi
  # PATH に無い場合は VSCode 拡張に同梱されたバイナリを使う（更新で消える点に注意）
  ls -d "$HOME"/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null \
    | sort -V | tail -1
}

CLAUDE_BIN="$(find_claude)"
if [ -z "$CLAUDE_BIN" ]; then
  echo "claude が見つかりません。'claude install' でインストールしてください。" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "既存の tmux セッション '$SESSION' に接続します"
else
  echo "tmux セッション '$SESSION' を作成します (claude: $CLAUDE_BIN)"
  tmux new-session -d -s "$SESSION" -c "$PROJECT" \
    "$CLAUDE_BIN --remote-control $SESSION"
fi

echo "接続: tmux attach -t $SESSION   / 切り離し: Ctrl-b d"
tmux attach -t "$SESSION"

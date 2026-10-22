#!/usr/bin/env bash
set -euo pipefail

# Fetch wuji-retargeting (including its wuji-description submodule) into
# <repo>/3rd. Idempotent: skips the clone when a valid checkout
# already exists and repairs missing submodules. Can run standalone (e.g. to
# prefetch before a slow network window) or via create_env.sh. The target path
# intentionally matches create_env.sh / the editable install path.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
WUJI_DIR="$REPO_ROOT/3rd/wuji-retargeting"

START_TS=$SECONDS
echo "== fetch wuji-retargeting =="
echo "  target : $WUJI_DIR"

if [ ! -d "$WUJI_DIR" ]; then
  echo "[fetch] cloning wuji-retargeting (recursive, includes wuji-description)"
  mkdir -p "$REPO_ROOT/3rd"
  git clone --recursive https://github.com/wuji-technology/wuji-retargeting.git "$WUJI_DIR"
else
  if [ ! -d "$WUJI_DIR/.git" ]; then
    echo "[fetch] ERROR: $WUJI_DIR exists but is not a git checkout (interrupted clone?)" >&2
    echo "[fetch] remove it and rerun: rm -rf '$WUJI_DIR'" >&2
    exit 1
  fi
  echo "[fetch] checkout already present, skipping clone"
fi

echo "[fetch] ensuring wuji-description submodule is initialized"
git -C "$WUJI_DIR" submodule update --init --recursive

echo "[fetch] HEAD       : $(git -C "$WUJI_DIR" rev-parse --short HEAD)"
echo "[fetch] submodules :"
git -C "$WUJI_DIR" submodule status --recursive | sed 's/^/        /'
echo "fetch finished in $((SECONDS - START_TS))s"

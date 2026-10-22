#!/usr/bin/env bash
set -euo pipefail

# Independent environment for the SONIC v1.1 ONNX simulator. Builds its own
# .venv, fetches wuji-retargeting into <repo>/3rd, and downloads the
# public SONIC v1.1 ONNX policy into ./models. It does not touch the other two
# projects and never reads anything from $HOME.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
WUJI_DIR="$REPO_ROOT/3rd/wuji-retargeting"
MODELS_DIR="$PROJECT_DIR/models"
HF_BASE="${HF_BASE:-https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/sonic_v1_1}"

if [ ! -d "$WUJI_DIR" ]; then
  echo "Cloning wuji-retargeting into $WUJI_DIR"
  mkdir -p "$REPO_ROOT/3rd"
  git clone --recursive https://github.com/wuji-technology/wuji-retargeting.git "$WUJI_DIR"
fi

python3 -m venv "$PROJECT_DIR/.venv"
PY="$PROJECT_DIR/.venv/bin/python"
"$PY" -m pip install --upgrade pip
# The mujoco floor is a correctness constraint, not a wish: the render paths hand
# Renderer.render()'s array to deferred consumers (the async preview, the
# streaming publisher) without copying it, and that only holds while render()
# allocates its output per call -- verified on 3.12.0, see SceneView.render.
"$PY" -m pip install "mujoco>=3.12" numpy onnxruntime pyarrow PySide6 scipy websockets pyyaml
# See the comment in g1_ik/create_env.sh: wuji_sdk is a private hardware-only
# wheel, so wuji-retargeting is installed with --no-deps after the two solver
# dependencies.
"$PY" -m pip install nlopt "pin>=3.8.0"
"$PY" -m pip install --no-deps -e "$WUJI_DIR"

# SONIC v1.1 ONNX policy (encoder + decoder) is public on Hugging Face and weighs
# ~191 MB. The chunked parallel downloader lives in download_models.sh, which can
# also be run standalone (re-fetch weights without rebuilding .venv); HF_BASE may
# be overridden externally, e.g. an hf-mirror.com endpoint.
bash "$PROJECT_DIR/download_models.sh"

echo "Ready: $PY $PROJECT_DIR/launch_g1_sonic.py"

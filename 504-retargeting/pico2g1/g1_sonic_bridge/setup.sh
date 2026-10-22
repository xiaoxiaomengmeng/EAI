#!/usr/bin/env bash
set -euo pipefail

# Independent environment for the headless SONIC data bridge. It only resamples
# poses and publishes them over ZMQ, so it needs neither the G1 MuJoCo model,
# nor wuji-retargeting, nor the SONIC ONNX policy. Its urdf and skeleton files
# already live next to this script. Builds its own .venv and reads nothing from
# $HOME or the other two projects.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 -m venv "$PROJECT_DIR/.venv"
PY="$PROJECT_DIR/.venv/bin/python"
"$PY" -m pip install --upgrade pip
# torch is required, even though this script does no inference. dex-retargeting
# declares it as an optional dependency, so pip will not pull it in, yet
# dex_retargeting/__init__.py raises ImportError on import when torch is missing,
# and build_hand_retargeter imports it on every run. The bridge itself only calls
# torch.load once (for the 4 KB human_joints_info.pkl), so it is tempting to drop
# torch and read that archive with zipfile instead; doing so breaks hand
# retargeting entirely. Installed from the cpu index -- which dex-retargeting's
# own error message recommends -- so Linux does not pull the multi-GB CUDA build
# for a policy-free data converter. The cpu index also serves macOS arm64 wheels.
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install numpy pyarrow scipy websockets pyzmq dex-retargeting

echo "Ready: $PY $PROJECT_DIR/launch_g1_sonic_bridge.py"

#!/usr/bin/env bash
# Setup PyTorch 2.11.0 + CUDA 12.8 for Python 3.10 on RTX 5090 (sm_120).
# Default behavior: install into the currently active Python environment.

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
WHEEL_DIR="$HOME/Downloads/py310-cu128-torch"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_VERIFY="0"
TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
FIND_LINKS="https://mirrors.aliyun.com/pytorch-wheels/"

TORCH_WHEEL="torch-2.11.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
TORCHVISION_WHEEL="torchvision-0.26.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
TORCHAUDIO_WHEEL="torchaudio-2.11.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
TRITON_WHEEL="triton-3.6.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

usage() {
    cat <<EOF
Usage:
  bash ${SCRIPT_NAME} [options]

Options:
  --python <path>     Python executable used for installation.
                      Default: ${PYTHON_BIN}
  --skip-verify       Skip torch CUDA verification after install.
  -h, --help          Show this help.

Examples:
  # Install into current conda/venv environment
  bash ${SCRIPT_NAME}
EOF
}

log() {
    echo "[${SCRIPT_NAME}] $*"
}

fail() {
    echo "[${SCRIPT_NAME}] ERROR: $*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --python)
            [ "$#" -ge 2 ] || fail "--python requires a Python executable"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --skip-verify)
            SKIP_VERIFY="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown option: $1"
            ;;
    esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: ${PYTHON_BIN}"

PY_VERSION="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [ "$PY_VERSION" != "3.10" ]; then
    fail "This script installs cp310 wheels and requires Python 3.10, current: ${PY_VERSION}"
fi

log "Using Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
log "Python version: ${PY_VERSION}"
log "Wheel directory: ${WHEEL_DIR}"

for wheel in "$TORCH_WHEEL" "$TORCHVISION_WHEEL" "$TORCHAUDIO_WHEEL" "$TRITON_WHEEL"; do
    [ -f "${WHEEL_DIR}/${wheel}" ] || fail "Missing wheel: ${WHEEL_DIR}/${wheel}"
done

log "Upgrading pip"
"$PYTHON_BIN" -m pip install --upgrade pip

log "Installing PyTorch 2.11.0 + cu128 wheels for RTX 5090 / sm_120"
"$PYTHON_BIN" -m pip install \
    "${WHEEL_DIR}/${TORCH_WHEEL}" \
    "${WHEEL_DIR}/${TORCHVISION_WHEEL}" \
    "${WHEEL_DIR}/${TORCHAUDIO_WHEEL}" \
    "${WHEEL_DIR}/${TRITON_WHEEL}" \
    -f "$FIND_LINKS" \
    -i "$INDEX_URL" \
    --trusted-host "$TRUSTED_HOST"

if [ "$SKIP_VERIFY" = "1" ]; then
    log "Skip verification requested."
    exit 0
fi

log "Verifying torch + CUDA + GPU"
"$PYTHON_BIN" - <<'PY'
import sys
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check NVIDIA driver, CUDA runtime, and cu128 torch wheel.")

gpu_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"GPU Name: {gpu_name}")
print(f"CUDA Capability: sm_{capability[0]}{capability[1]}")

if "5090" not in gpu_name:
    print("WARNING: GPU name does not contain 5090; verify this is the intended machine.", file=sys.stderr)

if capability != (12, 0):
    print("WARNING: Expected RTX 5090 compute capability sm_120.", file=sys.stderr)
PY

log "Done. PyTorch cu128 environment is ready."

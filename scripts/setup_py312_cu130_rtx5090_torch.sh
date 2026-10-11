#!/usr/bin/env bash
# Setup PyTorch 2.11.0 + CUDA 13.0 for Python 3.12 on RTX 5090 (sm_120).
# Installs the default PyPI build (cu130) from the Tsinghua mirror, which is
# much faster than downloading the cu128 wheels from the Aliyun pytorch-wheels
# mirror. Requires an NVIDIA driver that supports CUDA 13.0 (>= 580).
# Default behavior: install into the currently active Python environment.

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_VERIFY="0"
WITH_TORCHAUDIO="0"
TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

TORCH_SPEC="torch==2.11.0"
TORCHVISION_SPEC="torchvision==0.26.0"
TORCHAUDIO_SPEC="torchaudio==2.11.0"

usage() {
    cat <<EOF
Usage:
  bash ${SCRIPT_NAME} [options]

Options:
  --python <path>     Python executable used for installation.
                      Default: ${PYTHON_BIN}
  --with-torchaudio   Also install ${TORCHAUDIO_SPEC}.
  --skip-verify       Skip torch CUDA verification after install.
  -h, --help          Show this help.

Examples:
  # Install into current conda/venv environment
  bash ${SCRIPT_NAME}

  # Install into a specific venv
  bash ${SCRIPT_NAME} --python .venv/bin/python
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
        --with-torchaudio)
            WITH_TORCHAUDIO="1"
            shift
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

if [ "$PY_VERSION" != "3.12" ]; then
    fail "This script targets Python 3.12, current: ${PY_VERSION}"
fi

log "Using Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
log "Python version: ${PY_VERSION}"
log "Index: ${INDEX_URL}"

if command -v nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    log "WARNING: nvidia-smi not found; CUDA verification will likely fail."
fi

SPECS=("$TORCH_SPEC" "$TORCHVISION_SPEC")
if [ "$WITH_TORCHAUDIO" = "1" ]; then
    SPECS+=("$TORCHAUDIO_SPEC")
fi

log "Upgrading pip"
"$PYTHON_BIN" -m pip install --upgrade pip \
    -i "$INDEX_URL" \
    --trusted-host "$TRUSTED_HOST"

log "Installing ${SPECS[*]} (PyPI default build, cu130) for RTX 5090 / sm_120"
"$PYTHON_BIN" -m pip install "${SPECS[@]}" \
    -i "$INDEX_URL" \
    --trusted-host "$TRUSTED_HOST" \
    --default-timeout 120

if [ "$SKIP_VERIFY" = "1" ]; then
    log "Skip verification requested."
    exit 0
fi

log "Verifying torch + CUDA + GPU"
"$PYTHON_BIN" - <<'PY'
import sys
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Runtime: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check that the NVIDIA driver supports CUDA 13.0 (>= 580).")

if not (torch.version.cuda or "").startswith("13."):
    print("WARNING: Expected a CUDA 13.x torch build; another build was installed.", file=sys.stderr)

gpu_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"GPU Name: {gpu_name}")
print(f"CUDA Capability: sm_{capability[0]}{capability[1]}")

if "5090" not in gpu_name:
    print("WARNING: GPU name does not contain 5090; verify this is the intended machine.", file=sys.stderr)

if capability != (12, 0):
    print("WARNING: Expected RTX 5090 compute capability sm_120.", file=sys.stderr)

x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print(f"bf16 matmul OK: {(x @ x).float().abs().mean().item():.4f}")
PY

log "Done. PyTorch cu130 environment is ready."

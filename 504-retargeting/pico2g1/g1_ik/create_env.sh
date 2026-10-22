#!/usr/bin/env bash
set -euo pipefail

# Independent environment for the lightweight IK simulator. Builds its own .venv
# and fetches wuji-retargeting into <repo>/3rd. It does not touch the
# other two projects and never reads anything from $HOME.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
WUJI_DIR="$REPO_ROOT/3rd/wuji-retargeting"

START_TS=$SECONDS
echo "== g1_ik env setup =="
echo "  project : $PROJECT_DIR"
echo "  repo    : $REPO_ROOT"
echo "  wuji    : $WUJI_DIR"
echo "  host py : $(python3 --version 2>&1)"

echo "[1/5] Fetching wuji-retargeting (delegates to fetch_wuji.sh)"
bash "$PROJECT_DIR/fetch_wuji.sh"

echo "[2/5] Creating .venv at $PROJECT_DIR/.venv"
python3 -m venv "$PROJECT_DIR/.venv"
PY="$PROJECT_DIR/.venv/bin/python"
echo "      venv python: $($PY --version 2>&1)"

echo "[3/5] Upgrading pip"
"$PY" -m pip install --upgrade pip
echo "      pip: $($PY -m pip --version)"
# See the mujoco comment in g1_sonic/setup.sh: preview and stream images
# are consumed without a copy, which needs render() to allocate per call.
echo "[4/5] Installing runtime deps (mujoco numpy pyarrow PySide6 scipy websockets pyyaml)"
"$PY" -m pip install "mujoco>=3.12" numpy pyarrow PySide6 scipy websockets pyyaml
# wuji-retargeting declares wuji_sdk (a private wheel that only ships with real
# glove/hand hardware and is not on PyPI), so a plain editable install aborts
# before installing anything. This simulator only touches
# wuji_retargeting.Retargeter, which never imports wuji_sdk, hence --no-deps
# after installing the two real solver dependencies by hand.
echo "      Installing solver deps nlopt + 'pin>=3.8.0' (pin wheels are large, this can take a while)"
"$PY" -m pip install nlopt "pin>=3.8.0"
echo "[5/5] Installing wuji-retargeting (editable, --no-deps) from $WUJI_DIR"
"$PY" -m pip install --no-deps -e "$WUJI_DIR"

echo "== Smoke check: importing every key package (abort on first failure) =="
"$PY" - <<'EOF'
import importlib

for name in (
    "mujoco", "numpy", "pyarrow", "PySide6.QtCore", "scipy",
    "websockets", "yaml", "nlopt", "pinocchio", "wuji_retargeting",
):
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "?")
    print(f"  ok  {name:<18} {version}")
EOF

echo "Ready: $PY $PROJECT_DIR/launch_g1_ik.py"
echo "Setup finished in $((SECONDS - START_TS))s"

#!/usr/bin/env bash
set -euo pipefail

# Download only the SONIC v1.1 ONNX policy into ./models -- no .venv, no pip,
# no wuji (companion to setup.sh, which delegates this step here). Standalone
# usage: bash download_models.sh
# Idempotent: complete files are skipped; interrupted transfers resume from the
# per-file chunk staging dir models/.seg across runs.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$PROJECT_DIR/models"
# Overridable for mirrors, e.g. HF_BASE=https://hf-mirror.com/nvidia/GEAR-SONIC/resolve/main/sonic_v1_1
HF_BASE="${HF_BASE:-https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/sonic_v1_1}"

# The SONIC v1.1 ONNX policy (encoder + decoder) is public on Hugging Face and
# weighs ~191 MB, so it is downloaded here instead of being committed to the
# repo. No token is required (nvidia/GEAR-SONIC is a public repository).
#
# huggingface.co throttles this transfer to ~1.5 KB/s on a single connection, but
# the cap is per-connection: 8 parallel range requests measure ~74 KB/s and 24
# measure ~513 KB/s, which turns a 2-hour download into under 8 minutes. Each
# file is therefore split into 2 MB chunks fetched concurrently and concatenated.
# Chunks resume across runs, and a file is only moved into place once the
# assembled byte count matches Content-Length exactly -- an interrupted setup
# must never leave a truncated .onnx behind, because the old size-blind
# "[ ! -f ]" check would then skip the re-download and onnxruntime would fail at
# load time with an error that looks nothing like a network problem.
CHUNK=2000000
PAR=24
SEG_DIR="$MODELS_DIR/.seg"
export HF_BASE CHUNK SEG_DIR

fsize() { [ -f "$1" ] && wc -c <"$1" | tr -d ' ' || echo 0; }

# Content-Length after following the LFS/xet redirect. Matched case-insensitively
# via tolower() rather than awk's IGNORECASE, which BSD awk on macOS lacks.
remote_size() {
  # curl 失败（DNS/断网）时管道仍返回 0，awk 输出 0，由调用方给出可读错误而非 set -e 裸退
  curl -sIL --max-time 30 "$HF_BASE/$1" \
    | awk 'tolower($1) ~ /^content-length:/ {n=$2} END {gsub(/\r/, "", n); print n+0}' || true
}

chunk_want() {  # $1=total size, $2=chunk index -> bytes expected in that chunk
  local hi=$(( $2 * CHUNK + CHUNK - 1 ))
  if [ "$hi" -ge "$1" ]; then hi=$(( $1 - 1 )); fi
  echo $(( hi - $2 * CHUNK + 1 ))
}

get_chunk() {  # $1=name, $2=total size, $3=chunk index
  local lo=$(( $3 * CHUNK )) want have f
  want=$(chunk_want "$2" "$3")
  f="$SEG_DIR/$1.$3"
  for _ in 1 2 3 4 5 6; do
    have=$(fsize "$f")
    if [ "$have" -eq "$want" ]; then return 0; fi
    # More bytes than requested means a botched append; restart the chunk.
    if [ "$have" -gt "$want" ]; then rm -f "$f"; have=0; fi
    # --speed-limit/--speed-time drop a connection that stalls below 1 KB/s for
    # 30s instead of hanging on it; the next iteration resumes where it stopped.
    curl -sfL --http1.1 --connect-timeout 15 --max-time 240 \
      --speed-limit 1024 --speed-time 30 \
      -r $(( lo + have ))-$(( lo + want - 1 )) "$HF_BASE/$1" >>"$f" || true
  done
  [ "$(fsize "$f")" -eq "$want" ]
}
export -f fsize chunk_want get_chunk

mkdir -p "$MODELS_DIR"
for f in model_encoder.onnx model_decoder.onnx observation_config.yaml; do
  size=$(remote_size "$f")
  if [ "$size" -le 0 ]; then
    echo "Could not determine the remote size of $f" >&2
    exit 1
  fi
  if [ "$(fsize "$MODELS_DIR/$f")" -eq "$size" ]; then
    echo "$f already complete"
    continue
  fi

  chunks=$(( (size + CHUNK - 1) / CHUNK ))
  echo "Downloading $f ($size bytes, $chunks chunks, $PAR parallel)"
  mkdir -p "$SEG_DIR"
  missing=()
  for round in 1 2 3 4 5 6 7 8; do
    missing=()
    for ((i = 0; i < chunks; i++)); do
      if [ "$(fsize "$SEG_DIR/$f.$i")" -ne "$(chunk_want "$size" "$i")" ]; then
        missing+=("$i")
      fi
    done
    if [ "${#missing[@]}" -eq 0 ]; then break; fi
    echo "  round $round: ${#missing[@]}/$chunks chunks outstanding"
    printf '%s\n' "${missing[@]}" \
      | xargs -P "$PAR" -I{} bash -c "get_chunk '$f' $size {}" || true
  done
  if [ "${#missing[@]}" -ne 0 ]; then
    echo "$f: ${#missing[@]} chunks never completed" >&2
    exit 1
  fi

  : >"$MODELS_DIR/$f.part"
  for ((i = 0; i < chunks; i++)); do cat "$SEG_DIR/$f.$i" >>"$MODELS_DIR/$f.part"; done
  got=$(fsize "$MODELS_DIR/$f.part")
  if [ "$got" -ne "$size" ]; then
    echo "$f: assembled $got bytes, expected $size" >&2
    exit 1
  fi
  mv "$MODELS_DIR/$f.part" "$MODELS_DIR/$f"
  rm -f "$SEG_DIR/$f".*
  echo "  $f OK ($got bytes)"
done
rmdir "$SEG_DIR" 2>/dev/null || true

echo "Models ready in $MODELS_DIR:"
ls -lh "$MODELS_DIR" | awk 'NR>1 {print "  " $NF " (" $5 ")"}'

#!/usr/bin/env bash
set -euo pipefail
run() {
  printf >&2 '%s > ' "$PWD"
  printf >&2 '%q ' "$@"
  printf >&2 '\n'
  "$@" || { local rc=$?; printf >&2 'Command failed (exit %s): %s\n' "$rc" "$1"; return "$rc"; }
}
cd "$(dirname "$0")/.."
: "${QWEN_STRICT_IMAGE:?Set the image built with Dockerfile.qwen-strict-tools}"
: "${QWEN_TOKENIZER_PATH:?Set an existing tokenizer directory with resolved files}"
run python3 scripts/verify_qwen_strict_tools.py
run docker run --rm --pull never --runtime runc --user "$(id -u):$(id -g)" --network none --read-only --memory 8g --cpus 3 \
  --tmpfs /tmp:rw,nosuid,size=512m --cap-drop ALL --security-opt no-new-privileges \
  -e NVIDIA_VISIBLE_DEVICES=void -e SGLANG_DEVICE=cpu -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e QWEN_TOKENIZER_PATH=/tokenizer -e PYTHONPATH=/repo/tests:/sgl-workspace/sglang/python \
  -v "$PWD:/repo:ro" -v "$QWEN_TOKENIZER_PATH:/tokenizer:ro" \
  --entrypoint bash "$QWEN_STRICT_IMAGE" -c \
  'python3 -m pytest /repo/tests/runtime_qwen_strict_tools.py -q -p no:cacheprovider && python3 /repo/tests/runtime_qwen_effort_alias.py -q && python3 /repo/validation/responses/runtime_stream_stability.py -q'

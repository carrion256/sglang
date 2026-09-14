#!/usr/bin/env bash
# CPU-only local validation; no build, GPU devices, network, or publication.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
GRAY='\033[0;90m'
NC='\033[0m'

run() {
  printf >&2 "${GRAY}%s >${NC} ${YELLOW}" "$(pwd)"
  printf >&2 "%q " "$@"
  printf >&2 "${NC}\n"
  if "$@"; then
    printf >&2 "${GREEN}[OK]${NC}\n"
  else
    local exit_code=$?
    printf >&2 "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf >&2 "${RED}[ERROR]${NC} Command failed with exit code %d: ${YELLOW}%s${NC}\n" "$exit_code" "$1"
    printf >&2 "${RED}        Working dir:${NC} %s\n" "$(pwd)"
    printf >&2 "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    return "$exit_code"
  fi
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${QWEN_TOKENIZER_PATH:?Set the pinned tokenizer directory documented in docs/invalid-token-failure.md}"
QWEN_TOKENIZER_PATH="$(realpath "$QWEN_TOKENIZER_PATH")"
run test -f "$QWEN_TOKENIZER_PATH/tokenizer.json"

TOKENIZER_ROOT="$QWEN_TOKENIZER_PATH"
if [[ -L "$QWEN_TOKENIZER_PATH/tokenizer.json" ]]; then
  TOKENIZER_ROOT="$(realpath "$QWEN_TOKENIZER_PATH/../..")"
fi

IMAGE='kanadaj/sglang-qwen38fn-sm120-turbo@sha256:872a2bda228e39aa9c1af729b47cc28f7862e7859e448f1a8868b85a4051f404'
run python3 "$ROOT/scripts/verify_responses_compat.py" --tokenizer "$QWEN_TOKENIZER_PATH"
run python3 "$ROOT/scripts/verify_invalid_token_failure.py"
run docker image inspect --format '{{.Id}}' "$IMAGE"
run docker run --rm --pull never --network none --read-only --cap-drop all \
  --security-opt no-new-privileges --cpus 4 --memory 12g --pids-limit 512 \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,exec,size=2g,mode=1777,uid="$(id -u)",gid="$(id -g)" \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache -e PYTHONDONTWRITEBYTECODE=1 \
  -e QWEN_TOKENIZER_PATH="$QWEN_TOKENIZER_PATH" \
  -v "$TOKENIZER_ROOT:$TOKENIZER_ROOT:ro" -v "$ROOT:/repo:ro" \
  -v "$ROOT/runtime.invalid-token-failure/python/sglang/srt/entrypoints/openai/serving_completions.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_completions.py:ro" \
  -v "$ROOT/runtime.invalid-token-failure/python/sglang/srt/entrypoints/openai/serving_chat.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:ro" \
  -v "$ROOT/runtime/python/sglang/srt/entrypoints/openai/protocol.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/protocol.py:ro" \
  -v "$ROOT/runtime.invalid-token-failure/python/sglang/srt/entrypoints/openai/serving_responses.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:ro" \
  -v "$ROOT/runtime/python/sglang/srt/entrypoints/openai/responses_compat.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/responses_compat.py:ro" \
  -v "$ROOT/runtime/python/sglang/srt/function_call/qwen3_coder_detector.py:/sgl-workspace/sglang/python/sglang/srt/function_call/qwen3_coder_detector.py:ro" \
  -v "$ROOT/runtime.invalid-token-failure/python/sglang/srt/managers/schedule_batch.py:/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py:ro" \
  --entrypoint python3 "$IMAGE" /repo/tests/runtime_invalid_token_failure.py -v

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${QWEN_TOKENIZER_PATH:?Set the pinned tokenizer directory}"
python3 "$ROOT/scripts/verify_responses_compat.py" --tokenizer "$QWEN_TOKENIZER_PATH"
IMAGE='kanadaj/sglang-qwen38fn-sm120-turbo@sha256:872a2bda228e39aa9c1af729b47cc28f7862e7859e448f1a8868b85a4051f404'
docker run --rm --pull never --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --cpus 4 --memory 12g --pids-limit 512 \
  --tmpfs /tmp:rw,exec,size=2g --tmpfs /root/.cache:rw,exec,size=1g \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e XDG_CACHE_HOME=/tmp/cache -e PYTHONDONTWRITEBYTECODE=1 \
  -e QWEN_TOKENIZER_PATH=/tokenizer \
  -v "$QWEN_TOKENIZER_PATH:/tokenizer:ro" -v "$ROOT:/repo:ro" \
  -v "$ROOT/runtime/python/sglang/srt/entrypoints/openai/protocol.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/protocol.py:ro" \
  -v "$ROOT/runtime/python/sglang/srt/entrypoints/openai/serving_chat.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:ro" \
  -v "$ROOT/runtime/python/sglang/srt/entrypoints/openai/serving_responses.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:ro" \
  -v "$ROOT/runtime/python/sglang/srt/entrypoints/openai/responses_compat.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/responses_compat.py:ro" \
  -v "$ROOT/runtime/python/sglang/srt/function_call/qwen3_coder_detector.py:/sgl-workspace/sglang/python/sglang/srt/function_call/qwen3_coder_detector.py:ro" \
  --entrypoint sh "$IMAGE" -c \
  'python3 /repo/tests/runtime_responses_compat.py -v && python3 /repo/tests/runtime_qwen_effort_alias.py -v && bash /repo/scripts/test.sh'

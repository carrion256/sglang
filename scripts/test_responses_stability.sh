#!/usr/bin/env bash
set -euo pipefail
run() {
  printf >&2 '%s > ' "$PWD"
  printf >&2 '%q ' "$@"
  printf >&2 '\n'
  "$@" || { local rc=$?; printf >&2 'Command failed (exit %s): %s\n' "$rc" "$1"; return "$rc"; }
}
cd "$(dirname "$0")/.."
: "${RESPONSES_STABILITY_IMAGE:?Set the image built with Dockerfile.responses-stability}"
: "${QWEN_TOKENIZER_PATH:?Set the pinned tokenizer directory}"
run python3 scripts/verify_responses_stability.py
run python3 -m unittest discover -s tests -p test_responses_stability_packaging.py -v
common=(--rm --pull never --network none --read-only --user "$(id -u):$(id -g)"
  --memory 8g --cpus 3 --pids-limit 512 --cap-drop ALL
  --tmpfs /tmp:rw,exec,size=2g,mode=1777
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache -e PYTHONDONTWRITEBYTECODE=1
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e SGLANG_DEVICE=cpu
  -e PYTHONPATH=/repo/tests:/sgl-workspace/sglang/python
  -e QWEN_TOKENIZER_PATH=/tokenizer
  -v "$PWD:/repo:ro" -v "$QWEN_TOKENIZER_PATH:/tokenizer:ro")
run docker run "${common[@]}" --entrypoint python3 "$RESPONSES_STABILITY_IMAGE" -c \
 'from sglang.test.test_utils import maybe_stub_sgl_kernel; maybe_stub_sgl_kernel(); import pytest,sys; sys.exit(pytest.main(sys.argv[1:]))' \
 /repo/validation/responses/test_tool_wrapper.py -q -p no:cacheprovider
run docker run "${common[@]}" --entrypoint python3 "$RESPONSES_STABILITY_IMAGE" /repo/validation/responses/runtime_stream_stability.py -v
run docker run "${common[@]}" --entrypoint python3 "$RESPONSES_STABILITY_IMAGE" /repo/validation/responses/run_upstream_responses.py

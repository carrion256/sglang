#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GRAY='\033[0;90m'
NC='\033[0m'

run() {
  printf >&2 "${GRAY}$(pwd) >${NC} "
  printf >&2 "${YELLOW}"
  printf >&2 "%q " "$@"
  printf >&2 "${NC}\n"

  "$@" || {
    local exit_code=$?
    printf >&2 "${RED}Command failed with exit code %s: %s${NC}\n" "$exit_code" "$1"
    return "$exit_code"
  }
}

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

run "$PYTHON" scripts/verify_hicache_wip.py
run "$PYTHON" -m unittest tests/test_hicache_wip_packaging.py -v

if [[ -n "${HICACHE_WIP_IMAGE:-}" ]]; then
  run docker run --rm --pull never --network none --read-only \
    --user "$(id -u):$(id -g)" \
    --memory 4g --cpus 4 --pids-limit 512 --cap-drop ALL \
    --tmpfs /tmp:rw,noexec,nosuid,size=1g,mode=1777 \
    -e HOME=/tmp/hicache-home \
    -e SGLANG_CACHE_DIR=/tmp/hicache-cache \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e SGLANG_DEVICE=cpu \
    -e QWEN_HICACHE_TEST_DEVICE=cpu \
    -e PYTHONPATH=/hicache-tests:/sgl-workspace/sglang/python \
    -v "$(pwd)/validation/hicache:/hicache-tests:ro" \
    -v "$(pwd)/validation/prefill:/prefill-tests:ro" \
    --entrypoint python3 "$HICACHE_WIP_IMAGE" -c \
    'from sglang.test.test_utils import maybe_stub_sgl_kernel; maybe_stub_sgl_kernel(); import pytest,sys; sys.exit(pytest.main(sys.argv[1:]))' \
    /hicache-tests/test_hicache_ple_local.py \
    /hicache-tests/test_hicache_file_local.py \
    /hicache-tests/test_hicache_qsa_local.py \
    /hicache-tests/test_hicache_load_order.py \
    /hicache-tests/test_qsa_short_extend.py \
    /prefill-tests/test_paged_prefill_cpu.py \
    -q -p no:cacheprovider
fi

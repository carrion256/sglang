#!/usr/bin/env bash
set -euo pipefail
run() {
  printf >&2 '%s > ' "$PWD"
  printf >&2 '%q ' "$@"
  printf >&2 '\n'
  "$@" || { local rc=$?; printf >&2 'Command failed (exit %s): %s\n' "$rc" "$1"; return "$rc"; }
}
cd "$(dirname "$0")/.."
: "${HICACHE_DIAGNOSTICS_IMAGE:?Set the image built with Dockerfile.hicache-wip}"
export HICACHE_DIAGNOSTICS_IMAGE
run bash scripts/test_hicache_diagnostics.sh
run docker run --rm --pull never --network none --read-only --memory 8g --cpus 3 \
  --tmpfs /tmp:rw,exec,size=2g,mode=1777 \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  -e SGLANG_DEVICE=cpu -e QWEN_HICACHE_TEST_DEVICE=cpu \
  -e PYTHONPATH=/repo/validation/hicache:/sgl-workspace/sglang/python \
  -v "$PWD:/repo:ro" --entrypoint python3 "$HICACHE_DIAGNOSTICS_IMAGE" \
  -c 'from sglang.test.test_utils import maybe_stub_sgl_kernel; maybe_stub_sgl_kernel(); import pytest; raise SystemExit(pytest.main(["/repo/validation/hicache/test_restore_isolation_cpu.py", "-q", "-p", "no:cacheprovider"]))'

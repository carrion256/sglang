#!/usr/bin/env bash
set -euo pipefail
run() {
  printf >&2 '%s > ' "$PWD"
  printf >&2 '%q ' "$@"
  printf >&2 '\n'
  "$@" || { local rc=$?; printf >&2 'Command failed (exit %s): %s\n' "$rc" "$1"; return "$rc"; }
}
cd "$(dirname "$0")/.."
: "${HICACHE_BOUNDARY_IMAGE:?Set the image built with Dockerfile.hicache-wip}"
run python3 scripts/verify_hicache_wip.py
run python3 -m unittest discover -s tests -p test_hicache_wip_packaging.py -v
run docker run --rm --pull never --network none --read-only --user "$(id -u):$(id -g)" \
  --memory 8g --cpus 3 --pids-limit 512 --cap-drop ALL \
  --tmpfs /tmp:rw,exec,size=2g,mode=1777 \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e SGLANG_DEVICE=cpu -e QWEN_HICACHE_TEST_DEVICE=cpu \
  -e PYTHONPATH=/repo/validation/hicache:/sgl-workspace/sglang/python -v "$PWD:/repo:ro" \
  --entrypoint python3 "$HICACHE_BOUNDARY_IMAGE" -c \
 'from sglang.test.test_utils import maybe_stub_sgl_kernel; maybe_stub_sgl_kernel(); import pytest,sys; sys.exit(pytest.main(sys.argv[1:]))' \
 /repo/validation/hicache/test_common_boundary.py /repo/validation/hicache/test_hicache_file_local.py \
 /repo/validation/hicache/test_hicache_qsa_local.py /repo/validation/hicache/test_hicache_ple_local.py \
 /repo/validation/hicache/test_hicache_load_order.py /repo/validation/hicache/test_qsa_short_extend.py -q -p no:cacheprovider

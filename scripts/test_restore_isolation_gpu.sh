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
rank="${1:?Specify physical GPU 0 or 1}"
[[ "$rank" == 0 || "$rank" == 1 ]] || exit 2
if systemctl is-active --quiet vllm-qwen38fn.service; then
  echo 'Stop Qwen during an authorized maintenance window first.' >&2
  exit 3
fi
run docker run --rm --pull never --name "qwen-restore-isolation-gpu${rank}" \
  --gpus "device=${rank}" --network none --read-only --memory 12g --cpus 4 \
  --tmpfs /tmp:rw,exec,size=4g,mode=1777 --shm-size 1g \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  -e QWEN_HICACHE_TEST_DEVICE=cuda -e QWEN_HICACHE_TP_RANK="$rank" \
  -e PYTHONPATH=/repo/validation/hicache:/sgl-workspace/sglang/python \
  -v "$PWD:/repo:ro" --entrypoint python3 \
  "$HICACHE_DIAGNOSTICS_IMAGE" -m pytest -q -p no:cacheprovider \
  /repo/validation/hicache/test_restore_isolation_gpu.py \
  /repo/validation/hicache/test_hicache_file_gpu_local.py \
  /repo/validation/hicache/test_hicache_qsa_gpu_local.py \
  /repo/validation/hicache/test_hicache_ple_gpu_local.py \
  /repo/validation/hicache/test_hicache_load_order_gpu.py

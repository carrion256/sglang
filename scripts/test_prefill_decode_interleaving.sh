#!/usr/bin/env bash
set -euo pipefail
run() {
  printf >&2 '%s > ' "$PWD"
  printf >&2 '%q ' "$@"
  printf >&2 '\n'
  if "$@"; then return 0; else
    local rc=$?
    printf >&2 'Command failed (exit %s) in %s: ' "$rc" "$PWD"
    printf >&2 '%q ' "$@"
    printf >&2 '\n'
    return "$rc"
  fi
}
cd "$(dirname "$0")/.."
: "${INTERLEAVING_IMAGE:?Set an image built with the selected interleaving profile}"
profile=${INTERLEAVING_PROFILE:-standalone}
case "$profile" in
  standalone) verifier=verify_prefill_decode_interleaving ;;
  hicache) verifier=verify_hicache_wip ;;
  *) echo 'INTERLEAVING_PROFILE must be standalone or hicache' >&2; exit 2 ;;
esac
run python3 -B "scripts/$verifier.py"
run docker run --rm --pull never --runtime=runc --network none --read-only \
  --user "$(stat -c '%u:%g' .)" --memory 8g --cpus 2 --pids-limit 512 \
  --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,exec,size=2g,mode=1777 \
  -e NVIDIA_VISIBLE_DEVICES=void -e SGLANG_DEVICE=cpu -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache \
  -e INTERLEAVE_BASELINE=/repo/runtime/python/sglang/srt \
  -e INTERLEAVE_CANDIDATE=/sgl-workspace/sglang/python/sglang/srt \
  -e INTERLEAVING_VERIFIER="$verifier" \
  -e PYTHONPATH=/repo/tests:/sgl-workspace/sglang/python \
  -v "$PWD:/repo:ro" --entrypoint python3 "$INTERLEAVING_IMAGE" -B -c '
import importlib,os,pathlib,sys
assert not list(pathlib.Path("/dev").glob("nvidia*"))
assert not pathlib.Path("/dev/dri").exists()
sys.path.insert(0,"/repo/scripts")
verify = importlib.import_module(os.environ["INTERLEAVING_VERIFIER"]).verify
print("Verified source files:", verify(pathlib.Path("/sgl-workspace/sglang"),False))
from sglang.test.test_utils import maybe_stub_sgl_kernel
maybe_stub_sgl_kernel()
import torch
assert not torch.cuda.is_available()
import pytest
raise SystemExit(pytest.main([
 "/repo/tests/runtime_prefill_decode_interleaving.py",
 "/repo/tests/runtime_fractional_interleaving.py",
 "/repo/tests/runtime_interleaving_cli.py",
 "-q","-p","no:cacheprovider",
]))'

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

run "$PYTHON" scripts/verify_rvn_w4a16.py
run "$PYTHON" -m unittest tests/test_rvn_w4a16_packaging.py -v

# In-image behavioral battery. Default is the local-only base the profile is
# built on; point RVN_W4A16_IMAGE at a built rvn-w4a16 image to test it.
RVN_W4A16_IMAGE="${RVN_W4A16_IMAGE:-localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284}"
run docker run --rm --pull never --runtime=runc --network none --read-only \
  --user "$(id -u):$(id -g)" \
  --memory 8g --cpus 2 --pids-limit 512 --cap-drop ALL \
  --tmpfs /tmp:rw,noexec,nosuid,size=6g,mode=1777 \
  -e TORCH_EXTENSIONS_DIR=/tmp/rvn-torch-extensions \
  -e HOME=/tmp/rvn-home \
  -e SGLANG_CACHE_DIR=/tmp/rvn-cache \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e SGLANG_DEVICE=cpu \
  -e NVIDIA_VISIBLE_DEVICES=void -e GLOO_SOCKET_IFNAME=lo -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e RVN_PREIMAGE=/sgl-workspace/sglang/python/sglang \
  -v "$(pwd):/rvn-repo:ro" \
  --entrypoint bash "$RVN_W4A16_IMAGE" -c '
    set -eu
    mkdir -p /tmp/rvn-home /tmp/rvn-cache /tmp/rvn-torch-extensions
    if [ -f /sgl-workspace/sglang/python/sglang/srt/models/rvn_ple_storage.py ]; then
      echo "RVN_W4A16_IMAGE must be the unpatched base image: this battery" >&2
      echo "applies patches/series.rvn-w4a16 to the image python tree itself." >&2
      exit 2
    fi
    mkdir -p /tmp/rvn-tree
    cp -r /sgl-workspace/sglang/python /tmp/rvn-tree/
    cd /tmp/rvn-tree
    for patch in $(cat /rvn-repo/patches/series.rvn-w4a16); do
      git apply "/rvn-repo/patches/$patch"
    done
    export RVN_PLE_TREE=/tmp/rvn-tree
    cd /tmp
    # Each suite is dependency-light by design and must own its pytest
    # process: the loader suite puts the patched tree python/ on
    # sys.path, which corrupts a shared session for the later suites.
    status=0
    for suite in test_rvn_text_config test_rvn_w4a16_dispatch \
                 test_rvn_ple_loader test_rvn_ple_wiring \
                 test_rvn_ple_loader_gpu test_convert test_inventory test_verify; do
      echo "### $suite"
      python3 -m pytest -q -p no:cacheprovider \
        "/rvn-repo/tests/rvn_ple/$suite.py" || status=1
    done
    exit "$status"
  '

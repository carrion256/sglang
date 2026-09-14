#!/usr/bin/env bash
# CPU-only local validation; no build, GPU devices, network, or publication.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${QWEN_MODEL_PATH:?Set to the release config directory documented in docs/qwen-multimodal-alias.md}"
CONFIG_PATH="$(realpath "$QWEN_MODEL_PATH/config.json")"
test -f "$CONFIG_PATH"
IMAGE='kanadaj/sglang-qwen38fn-sm120-turbo@sha256:872a2bda228e39aa9c1af729b47cc28f7862e7859e448f1a8868b85a4051f404'
python3 "$ROOT/scripts/verify_qwen_multimodal_alias.py"
docker image inspect "$IMAGE" >/dev/null
docker run --rm --pull never --network none --read-only --cap-drop all \
  --pids-limit 512 --memory 16g --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,exec,size=2g,uid="$(id -u)",gid="$(id -g)" \
  --tmpfs /home/ubuntu/.cache:rw,exec,size=1g,uid="$(id -u)",gid="$(id -g)" \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e QWEN_CONFIG_PATH=/qwen-config.json \
  -v "$CONFIG_PATH:/qwen-config.json:ro" -v "$ROOT:/repo:ro" \
  -v "$ROOT/runtime/python/sglang/srt/multimodal/processors/qwen_vl.py:/sgl-workspace/sglang/python/sglang/srt/multimodal/processors/qwen_vl.py:ro" \
  --entrypoint python3 "$IMAGE" /repo/tests/runtime_qwen_multimodal_alias.py -v

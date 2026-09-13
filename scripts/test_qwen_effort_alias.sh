#!/usr/bin/env bash
# CPU-only local validation; no build, GPU devices, network, or publication.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${QWEN_TOKENIZER_PATH:?Set to the pinned tokenizer directory documented in docs/qwen-effort-alias.md}"
IMAGE='kanadaj/sglang-qwen38fn-sm120-turbo@sha256:872a2bda228e39aa9c1af729b47cc28f7862e7859e448f1a8868b85a4051f404'
python3 - "$ROOT" "$QWEN_TOKENIZER_PATH" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / 'provenance/qwen-effort-alias.json').read_text())
checks = {root / 'patches' / manifest['patch']: manifest['patch_sha256']}
checks.update({root / 'runtime' / name: hashes['after'] for name, hashes in manifest['files'].items()})
checks.update({pathlib.Path(sys.argv[2]) / name: expected for name, expected in manifest['tokenizer']['files'].items()})
for path, expected in checks.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('Provenance hash mismatch: ' + str(path))
PY
docker image inspect "$IMAGE" >/dev/null
docker run --rm --pull never --network none \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e QWEN_TOKENIZER_PATH=/tokenizer \
  -v "$QWEN_TOKENIZER_PATH:/tokenizer:ro" -v "$ROOT:/repo:ro" \
  -v "$ROOT/runtime/python/sglang/srt/entrypoints/openai/serving_chat.py:/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:ro" \
  --entrypoint sh "$IMAGE" -c \
  'python3 /repo/tests/runtime_qwen_effort_alias.py -v && bash /repo/scripts/test.sh'

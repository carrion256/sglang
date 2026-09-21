# Quick start — deploy the image as a plain Docker container

One command. No Git clone, no external JSON files, no build. The model-specific
config (48-layer hybrid `layer_types`, MTP pack, PLE/ngram fields, indexer
settings) is **embedded in the image** at
`/opt/qwen-runtime/model_overrides.json` and merged automatically — you do not
pass a 2.8 KB `--json-model-override-args` blob.

```bash
docker run --rm --gpus '"device=0,1"' --ipc=host -p 127.0.0.1:30000:30000 \
  -v /absolute/path/to/Qwen3.8-Flash-Next-NVFP4:/model:ro \
  -e SGLANG_SM120_ONLINE_MXFP8=false \
  -e SGLANG_PLE_PACKED_NVFP4=1 \
  -e SGLANG_PLE_PACKED_FP8_REFERENCE=1 \
  -e SGLANG_PRIVATE_DRAFT_NVFP4_A16=1 \
  docker.io/kanadaj/sglang-qwen38fn-sm120-turbo:hicache-pr19-embed-20260921-v1@sha256:6acf6306887726b31ad0003de9021fdf0147ffbb1a29a4e3147421bf5063d1e2 \
  --model-path /model \
  --tp-size=2 --quantization=modelopt_mixed \
  --kv-cache-dtype=fp8_e4m3 --context-length=262144 \
  --mem-fraction-static=0.93 --page-size=64 --chunked-prefill-size=4096 \
  --reasoning-parser=auto --tool-call-parser=auto \
  --linear-attn-prefill-backend=flashinfer --linear-attn-decode-backend=flashinfer \
  --mamba-radix-cache-strategy=extra_buffer --mamba-track-interval=128 \
  --mamba-ssm-dtype=bfloat16 --gdn-mtp-cache-mode=none \
  --max-mamba-cache-size=124 --max-running-requests=16 \
  --cuda-graph-max-bs-decode=16 --moe-runner-backend=flashinfer_cutlass \
  --disable-custom-all-reduce --disable-prefill-cuda-graph \
  --ple-offload-embedding \
  --speculative-algorithm=NEXTN --speculative-num-steps=3 \
  --speculative-eagle-topk=1 --speculative-num-draft-tokens=4 \
  --speculative-draft-model-quantization=modelopt_mixed \
  --speculative-moe-runner-backend=flashinfer_cutlass \
  --model-loader-extra-config '{"enable_multithread_load":false,"num_threads":2}' \
  --startup-weight-load-mode=serial --mm-enable-dp-encoder \
  --enable-metrics --enable-cache-report \
  --host 0.0.0.0 --port 30000
```

Notes:

- The image entrypoint is already `python3 -m sglang.launch_server` from the
  digest-profile lineages; if you use a tag whose entrypoint is something else,
  prepend `--entrypoint python3 IMAGE -m sglang.launch_server` as in
  `docs/standalone.md`.
- `--gpus '"device=0,1"'` needs the nested quotes exactly as written.
- Serve is at `http://127.0.0.1:30000/v1/chat/completions` (OpenAI-compatible).

## Extended context (YaRN) — opt in explicitly

The embedded config is **native** (262 k, no rope scaling): correct defaults for
anyone who just runs the container. To serve the 1 M window like our production
fleet, pass only the YaRN override — the small form, not the full blob:

```bash
  -e SGLANG_YARN_ROPE_SCALING_FACTOR=4.0 \
  --context-length=1048576
```

`SGLANG_YARN_ROPE_SCALING_FACTOR` (patch `0046`) rewrites the embedded rope
config post-merge and recomputes `max_position_embeddings`; the CLI
`--json-model-override-args` YaRN blob (see
`docs/production-command.sh`) is equivalent if you prefer explicit args over env.
YaRN factor 4.0 requires the full-size checkpoint; at 1 M expect reduced
concurrency headroom and keep `--max-running-requests` conservative.

## HiCache (optional tiered KV + hybrid checkpoint persistence)

Add the flags our fleet runs (requires host RAM and disk budget — see
`docs/hicache-production-recipe.md`):

```bash
  -e SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/var/lib/docker/volumes/sglang-hicache/_data \
  -e SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=24G \
  ... --enable-hierarchical-cache --hicache-write-policy=write_back \
  --hicache-size=16 --hicache-storage-backend=file \
  --hicache-storage-prefetch-policy=wait_complete
```

The `hicache-pr19-embed` image tag above already contains the HiCache
checkpoint-preservation patches (`0040`–`0045`) so hybrid (Mamba/GDN) state
survives eviction/restore; earlier tags do not.

## Verification

```bash
curl -s http://127.0.0.1:30000/get_server_info | \
  python3 -c "import json,sys; si=json.load(sys.stdin); \
  print('ctx:', si['context_length'], '| merged:', bool(si['json_model_override_args']))"
```

A log line `Merged CLI model overrides onto embedded model config from
/opt/qwen-runtime/model_overrides.json` confirms the embedded merge ran.

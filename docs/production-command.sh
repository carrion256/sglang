# Production launch — LIVE reference (verified against GPUStack model 48, 2026-09-22)
# Image: hicache-pr19-embed = cumulative 0015-0019 + 0020-0033 + 0034-embedded-model-overrides
#        + 0046-yarn-factor + PR#19 hicache checkpoint patches 0040-0045.
# Example only: do not run on occupied GPUs/ports. Requires a complete local checkpoint.
#
# KEY CHANGE vs older revisions of this file: the ~2.8 KB --json-model-override-args
# blob (vocab/layers/layer_types/MTP/PLE/ngram/indexer fields) is EMBEDDED in the
# image at /opt/qwen-runtime/model_overrides.json and merged automatically at
# startup (log: "Merged CLI model overrides onto embedded model config ...").
# Only the YaRN extension stays explicit — here as a small CLI override (equivalent
# alternative: -e SGLANG_YARN_ROPE_SCALING_FACTOR=4.0 with context-length, no CLI arg;
# see docs/embedded-model-overrides.md).
#
# GPUStack env for this deployment (beyond embedded overrides):
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SAFETENSORS_FAST_GPU=1
#   SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1
#   SGLANG_SM120_ONLINE_MXFP8=false SGLANG_PLE_PACKED_NVFP4=1
#   SGLANG_PLE_PACKED_FP8_REFERENCE=1 SGLANG_PRIVATE_DRAFT_NVFP4_A16=1
#   SGLANG_PLE_SHARED_DIR=/var/lib/gpustack/ple-shared
#   SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/var/lib/gpustack/hicache
#   SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=24G
#   SGLANG_EMBEDDED_MODEL_OVERRIDES=/opt/qwen-runtime/model_overrides.json
docker run --rm --gpus '"device=2,3"' --ipc=host -p 127.0.0.1:30000:30000 \
  -v /absolute/path/to/Qwen3.8-Flash-Next-NVFP4:/model:ro \
  -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1 \
  -e SGLANG_SM120_ONLINE_MXFP8=false \
  -e SGLANG_PLE_PACKED_NVFP4=1 \
  -e SGLANG_PLE_PACKED_FP8_REFERENCE=1 \
  -e SGLANG_PRIVATE_DRAFT_NVFP4_A16=1 \
  -e SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/var/lib/gpustack/hicache \
  -e SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=24G \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 -e OMP_NUM_THREADS=1 \
  docker.io/kanadaj/sglang-qwen38fn-sm120-turbo:hicache-pr19-embed-20260921-v1@sha256:6acf6306887726b31ad0003de9021fdf0147ffbb1a29a4e3147421bf5063d1e2 \
  --model-path /model \
  --enable-metrics \
  --uvicorn-access-log-exclude-prefixes /metrics \
  --reasoning-parser=auto \
  --tool-call-parser=auto \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --linear-attn-prefill-backend=flashinfer \
  --linear-attn-decode-backend=flashinfer \
  --max-mamba-cache-size=124 \
  --mamba-radix-cache-strategy=extra_buffer \
  --mamba-track-interval=128 \
  --mamba-ssm-dtype=bfloat16 \
  --gdn-mtp-cache-mode=none \
  --tp-size=2 \
  --quantization=modelopt_mixed \
  --kv-cache-dtype=fp8_e4m3 \
  --context-length=1048576 \
  --mem-fraction-static=0.93 \
  --page-size=64 \
  --chunked-prefill-size=4096 \
  --max-running-requests=24 \
  --mamba-max-states-per-path=4 \
  --prefill-batches-before-decode=0.5 \
  --enable-cache-report \
  --cuda-graph-max-bs-decode=16 \
  --moe-runner-backend=flashinfer_cutlass \
  --disable-custom-all-reduce \
  --disable-prefill-cuda-graph \
  --json-model-override-args '{"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144, "rope_theta": 10000000, "mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25}, "max_position_embeddings": 1048576}, "max_position_embeddings": 1048576}' \
  --ple-offload-embedding \
  --speculative-algorithm=NEXTN \
  --speculative-num-steps=3 \
  --speculative-eagle-topk=1 \
  --speculative-num-draft-tokens=4 \
  --speculative-draft-model-quantization=modelopt_mixed \
  --speculative-moe-runner-backend=flashinfer_cutlass \
  --model-loader-extra-config '{"enable_multithread_load":false,"num_threads":2}' \
  --startup-weight-load-mode=serial \
  --mm-enable-dp-encoder \
  --enable-hierarchical-cache \
  --hicache-write-policy=write_back \
  --hicache-size=16 \
  --hicache-storage-backend=file \
  --hicache-storage-prefetch-policy=wait_complete \
  --host 0.0.0.0 \
  --port 30000

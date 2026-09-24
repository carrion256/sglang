#!/usr/bin/env bash
# run_baseline.sh — experimental, opt-in TP1 launch profile for the UNCHANGED
# RVN checkpoint (qwen38-flash-next-uncensored) on the patched fork image.
#
#   Usage: run_baseline.sh <gpu-index> [--dry-run] [--ple-offload[=GIB]] [--image IMG]
#
#   <gpu-index>       host GPU the single TP1 worker may use (required)
#   --dry-run         print the exact `docker run` command and exit; NO side
#                     effects, so the two real-run gates (host-RAM floor, image
#                     profile) are only announced, never enforced
#   --ple-offload[=G] HOST-RAM BUDGET GUARD for the pinned BF16 PLE table
#                     (default floor 100 GiB). It is NOT a memory mitigation:
#                     see MEMORY BOUND below.
#   --image IMG       override the pinned profile image (default rvn-w4a16:sim,
#                     the patched Dockerfile.rvn-w4a16 build). A real run aborts
#                     unless that image passes the PROFILE GATE below.
#
# This is an EXPERIMENT. It never touches the production workers
# (lilith-vllm / lilith-vllm-b): distinct container name, model name, port and
# cache dirs. Rollback = remove the container only.
#
# TEXT-ONLY PROFILE. The RVN checkpoint config declares
# architectures=[Qwen4ExpForCausalLM], model_type=qwen4_exp_text, with no
# vision_config, so every inherited LIL multimodal flag is removed:
# --image-processor-backend, --mm-feature-transport, --limit-mm-data-per-request,
# --media-url-max-file-size-mb and SGLANG_IMAGE_MAX_PIXELS. Nothing in
# runtime/python/sglang/srt/server_args.py requires them for a text arch — they
# only configure the vision encoder path this checkpoint never builds.
#
# MODEL CONFIG SOURCE (no override file, no CLI override blob): the server reads
# the RVN checkpoint's own /models/qwen38-flash-next-uncensored/config.json —
# rope_type=default with max_position_embeddings=262144 (no YaRN),
# mtp_num_hidden_layers=0 (no MTP), quant_algo=W4A16_NVFP4 with `*ple*` in
# exclude_modules, so PLE stays BF16 (docs/rvn-ple-storage-schema.md §1:
# RVN `reconstruction: bf16_direct`, not the legacy LIL
# SGLANG_PLE_PACKED_FP8_REFERENCE=1 path). Production instead passes a baked
# LIL text_config / YaRN blob; per patches/0034-embedded-model-overrides.patch
# an empty SGLANG_EMBEDDED_MODEL_OVERRIDES is a no-op, so the env file clears
# the key explicitly and no --json-model-override-args is passed at all.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE="$SCRIPT_DIR/env.rvn-w4a16-baseline"

# Patched PROFILE image — what `docker build -f Dockerfile.rvn-w4a16
# -t rvn-w4a16:sim .` produces (docs/rvn-w4a16.md "Build and run"). This is
# deliberately NOT the base image the production workers run
# (localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284 = Dockerfile.rvn-w4a16:4),
# because that base lacks the three patches this profile's flags require:
# 0047 adds the Qwen4ExpForCausalLM entry the RVN config declares, 0048 routes
# a quantized_layers-less W4A16_NVFP4 checkpoint through uniform ModelOpt FP4
# (without it --quantization=modelopt_mixed below fails mixed-precision
# validation), and 0051 makes Qwen4ExpForCausalLM eligible for
# --ple-offload-embedding. Every flag below is otherwise unreachable on the base.
# Like deploy/run.py's --image default this is an explicit pinned reference; the
# pin is by tag only because the profile image is built locally and never pushed
# (docs/rvn-w4a16.md: "This base is local-only"), so no registry digest exists
# to pin. For a reproducible real run, pin by image Id
# (--image "$(docker image inspect -f '{{.Id}}' rvn-w4a16:sim)") or pass a
# freshly built tag. The mutable tag is made safe by the PROFILE GATE below,
# which hashes the image's source tree instead of trusting the tag.
IMAGE=${IMAGE:-rvn-w4a16:sim}
# Baseline-owned host cache root. NEVER reuse the production namespaces
# /var/cache/sglang/qwen38-flash-next-kanadaj-a6d5284-tp1-gpu0[-b] or
# the shared /data/hicache/qwen38-flash-next-kanadaj-a6d5284.
CACHE_ROOT=/var/cache/sglang
NAME_PREFIX=rvn-w4a16-baseline
# Chat template: byte-identical to the one the production workers mount
# (tokenizer + template bytes unchanged).
CHAT_TEMPLATE=/nix/store/0l335cvqdqqfyrnlbwddp4nzzl98pw2s-qwen38-unsloth-chat-template.jinja
# Tokenizer source is TOKENIZER_PATH in the env file. Default = the RVN tree
# itself (/models/qwen38-flash-next-uncensored). Documented contingency: while
# its `hf download` (259 files) was still running on 2026-09-24, that tree had
# tokenizer.json, tokenizer_config.json, vocab.json and merges.txt absent, and
# the then-only option was TOKENIZER_PATH=/models/qwen38-flash-next (LIL tree,
# same 248320-token vocabulary, so tokenizer bytes stay unchanged). The
# orchestrator confirms tokenizer presence once the download completes; only then
# is that fallback used or deleted. --chat-template stays explicit and
# load-bearing: with --tokenizer-path aimed at another tree, sglang would
# otherwise auto-discover that tree's own chat_template.jinja instead of the
# byte-identical nix-store file this profile must mount.

DRY_RUN=0
PLE_MIN_FREE_GIB=100
GPU=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --ple-offload) shift ;;
    --ple-offload=*) PLE_MIN_FREE_GIB=${1#*=}; shift ;;
    --image) IMAGE=$2; shift 2 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) [[ -n $GPU ]] && { echo "usage: $0 <gpu-index> [options]" >&2; exit 2; }
       GPU=$1; shift ;;
  esac
done

[[ -n $GPU && $GPU =~ ^[0-9]+$ ]] || {
  echo "usage: $0 <gpu-index> [--dry-run] [--ple-offload[=GIB]] [--image IMG]" >&2; exit 2; }

# The floor feeds an arithmetic comparison: a non-numeric value would evaluate as
# 0 and silently bypass the gate, and 0 disables it outright. Require positive.
[[ $PLE_MIN_FREE_GIB =~ ^[1-9][0-9]*$ ]] || {
  echo "bad --ple-offload floor: '$PLE_MIN_FREE_GIB' (need a positive integer GiB)" >&2
  exit 2; }

# Serving parameters come from the env file (single source of truth).
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a

# MEMORY BOUND — unconditional gate, applies to EVERY real run:
# ORIGINAL BF16 PLE host lookup (--ple-offload-embedding with no packed envs)
# builds the n-gram table with torch.empty(..., pin_memory=True) in
# runtime/python/sglang/srt/models/qwen4_exp.py: ~95 GiB page-locked host RSS
# per process. This image has NO file-backed/shared BF16 PLE path: the shared
# MAP_SHARED table (patches/0032-shared-ple-host-table.patch) is absent from
# patches/series.production and SGLANG_PLE_SHARED_DIR has zero references in the
# shipped runtime tree — it is reachable only through PackedPLEStorage, i.e. the
# packed-NVFP4 path (WP2) this profile deliberately never enables. So nothing
# here can shrink that allocation; --ple-offload therefore only sets the
# host-RAM floor this script enforces. The --ulimit memlock=-1 below (as in
# production) is what lets the pinning succeed at all.
if (( DRY_RUN )); then
  echo "NOTE: real run enforces >= ${PLE_MIN_FREE_GIB} GiB MemAvailable for the" \
       "~95 GiB pinned BF16 PLE table (no file-backed path in this image)." >&2
else
  avail_gib=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
  if (( avail_gib < PLE_MIN_FREE_GIB )); then
    echo "ERROR: ${avail_gib} GiB MemAvailable < required ${PLE_MIN_FREE_GIB} GiB" \
         "for the ~95 GiB pinned BF16 PLE table." >&2
    echo "ERROR: free host RAM or raise the floor deliberately with" \
         "--ple-offload=<gib>; nothing was started." >&2
    exit 1
  fi
fi

# PROFILE GATE — unconditional for every real run, and it runs BEFORE `docker
# run` / before any host directory is created: the launch flags below need
# patches 0047 + 0048 + 0051, and on an image without them the failure surfaces
# only deep inside the server — after this profile has already pinned ~95 GiB of
# host RAM — so a wrong image must be caught here, not at model load.
# The check is the verifier that Dockerfile.rvn-w4a16:8 bakes into every profile
# image, run WITHOUT --apply: it hashes the image's own python/sglang tree
# against the post-patch inventory in provenance/rvn-w4a16.json, i.e. it proves
# all 8 changed files byte-for-byte. Rejected alternatives: `docker image
# inspect` labels, because the base and the patched image carry byte-identical
# label sets (measured 2026-09-25: Dockerfile.rvn-w4a16 adds no LABEL), so a
# label probe would reject even a correct image; and a
# `python3 -c "import sglang.srt.models.qwen4_exp_text_adapter"` probe, because
# importing it drags in torch/CUDA (seconds slower) and proves only patch 0047
# while the dispatch and offload-eligibility patches stay unchecked. This probe
# costs ~1.4 s in a `--rm`, GPU-less, `--network none` container, and
# `--pull never` keeps it from fetching a same-named image from a registry
# instead of using the local one. It is fail-closed: an absent verifier (the
# base image) exits 2, and any tree drift exits non-zero.
if (( DRY_RUN )); then
  echo "NOTE: real run also aborts unless $IMAGE passes the in-image rvn-w4a16" \
       "profile gate (see PROFILE GATE below)." >&2
else
  if gate=$(docker run --rm --pull never --network none \
              --entrypoint python3 "$IMAGE" -B \
              /opt/rvn-w4a16/scripts/verify_rvn_w4a16.py \
              --tree /sgl-workspace/sglang 2>&1); then
    echo "profile gate OK: $gate" >&2
  else
    echo "ERROR: image $IMAGE is not a verified rvn-w4a16 profile image; nothing" \
         "was started (no container, no cache dir, no pinned host RAM)." >&2
    echo "ERROR: build it with" >&2
    echo "         docker build -f Dockerfile.rvn-w4a16 -t rvn-w4a16:sim ." >&2
    echo "       or select another image with --image IMG / IMAGE=<ref>." >&2
    sed 's/^/       /' <<<"$gate" >&2
    exit 1
  fi
fi

# Exact env: every KEY=VALUE line of the env file, plus GPU pinning from the
# positional argument (mirrors production: CDI --gpus all + CUDA_VISIBLE_DEVICES
# selects the worker's GPU).
env_args=()
while IFS= read -r line || [[ -n $line ]]; do
  [[ -z $line || $line == \#* ]] && continue
  env_args+=(-e "$line")
done < "$ENV_FILE"
env_args+=(-e "CUDA_VISIBLE_DEVICES=$GPU" -e "NVIDIA_VISIBLE_DEVICES=$GPU")

mounts=(
  -v /models:/models:ro                                        # checkpoints, read-only
  -v "$CHAT_TEMPLATE":/etc/qwen38/chat-template.jinja:ro       # bytes unchanged
  -v "$CACHE_ROOT/$NAME_PREFIX-$GPU":/root/.cache              # baseline-owned caches
)

# Launch command inside the container. Values are expanded from the env file so
# the printed dry-run IS the command that would run.
INNER="exec python3 -m sglang.launch_server"
INNER+=" --model-path $MODEL"
INNER+=" --tokenizer-path $TOKENIZER_PATH"
INNER+=" --chat-template /etc/qwen38/chat-template.jinja"
INNER+=" --served-model-name $SERVED_MODEL_NAME"
INNER+=" --host 0.0.0.0 --port $PORT"
INNER+=" --enable-metrics --uvicorn-access-log-exclude-prefixes /metrics"
# Auto-detection driven by the (unchanged) chat template, not a LIL value; kept
# because dropping them would change what /v1/chat/completions returns as
# content vs reasoning. Production's --default-chat-template-kwargs
# '{"reasoning_effort":"xhigh"}' is a LIL serving default and is NOT passed.
INNER+=" --reasoning-parser=auto --tool-call-parser=auto"
# CHECKPOINT-DECLARED VALUES DELIBERATELY OVERRIDDEN (deliberate v1 memory /
# precision choices matching production, NOT checkpoint values):
# --mamba-ssm-dtype=bfloat16 while RVN config.json:79 declares
# "mamba_ssm_dtype": "float32", and --kv-cache-dtype=fp8_e4m3 while
# config.json:140 declares "kv_cache_quant_algo": null. Drop both to serve the
# checkpoint's own numerics.
# Hybrid-architecture backends (required to serve this arch at all).
INNER+=" --linear-attn-prefill-backend=flashinfer --linear-attn-decode-backend=flashinfer"
INNER+=" --max-mamba-cache-size=64 --mamba-radix-cache-strategy=extra_buffer"
INNER+=" --mamba-ssm-dtype=bfloat16 --mamba-track-interval=128"
# MTP off, explicitly: gdn_mtp_cache_mode defaults to 'full' (server_args.py
# help text), so 'none' is what grounds "RVN has no MTP" at the flag level
# alongside config mtp_num_hidden_layers=0 and the omitted --speculative-*.
INNER+=" --gdn-mtp-cache-mode=none"
# TP1, Marlin MoE, BF16 dense compute (SGLANG_SM120_ONLINE_MXFP8=false in env).
INNER+=" --tp-size $TP_SIZE --quantization=modelopt_mixed --moe-runner-backend=marlin"
INNER+=" --kv-cache-dtype=fp8_e4m3 --context-length $MAX_MODEL_LEN"
INNER+=" --mem-fraction-static $GPU_MEMORY_UTILIZATION --page-size 64"
INNER+=" --chunked-prefill-size $MAX_NUM_BATCHED_TOKENS --max-running-requests $MAX_NUM_SEQS"
# v1 safety: eager execution with graph capture OFF. --disable-cuda-graph is a
# DeprecatedStoreTrueAction here (server_args.py:8860) pointing at these two.
INNER+=" --cuda-graph-backend-decode=disabled --cuda-graph-backend-prefill=disabled"
INNER+=" --disable-radix-cache"
# ORIGINAL BF16 PLE host lookup (no SGLANG_PLE_PACKED_* env passed).
INNER+=" --ple-offload-embedding"
INNER+=" --model-loader-extra-config '{\"enable_multithread_load\":false,\"num_threads\":2}'"
INNER+=" --startup-weight-load-mode=serial"
# Deliberately absent: every --speculative-* (RVN has no MTP draft head), the
# --json-model-override-args YaRN blob, --enable-hierarchical-cache and all
# --hicache-* flags, --enable-cache-report, --disable-custom-all-reduce (the
# live TP1 workers do not pass it either), and all vision/mm flags (text-only).

# --entrypoint /bin/bash is REQUIRED, not cosmetic: Dockerfile.rvn-w4a16:10 sets
# ENTRYPOINT to ["python3","-m","sglang.launch_server"], so without the override
# `docker run … $IMAGE /bin/bash -lc …` executes
# `python3 -m sglang.launch_server /bin/bash -lc …` and argparse dies on the
# positional "/bin/bash" before the server ever starts. deploy/run.py:28 passes
# `--entrypoint sglang` and docs/rvn-w4a16.md:69 passes `--entrypoint python3`
# for the same reason; here the wrapper shell is what the quoted
# --model-loader-extra-config in INNER needs, so the entrypoint becomes bash and
# the launch line stays an argument to it.
cmd=(
  docker run --rm
  --name "$NAME_PREFIX-$GPU"
  --gpus all
  --ipc host --network host
  --shm-size 34359738368
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --ulimit stack=67108864:67108864
  "${env_args[@]}"
  "${mounts[@]}"
  --entrypoint /bin/bash
  "$IMAGE"
  -lc "$INNER"
)

if (( DRY_RUN )); then
  # Print the exact command, one argument per line, shell-quoted. No side
  # effects: nothing is created, no container is touched, no check is run.
  quote() { local s=$1; printf "'%s'" "${s//\'/\'\\\'\'}"; }
  out=()
  for a in "${cmd[@]}"; do out+=("$(quote "$a")"); done
  printf '%s' "${out[0]}"
  for ((i = 1; i < ${#out[@]}; i++)); do printf ' \\\n  %s' "${out[$i]}"; done
  printf '\n'
  exit 0
fi

# Real run: create ONLY the baseline-owned host cache dir (never prod's).
mkdir -p "$CACHE_ROOT/$NAME_PREFIX-$GPU"
exec "${cmd[@]}"

#!/usr/bin/env bash
# run_baseline.sh — experimental, opt-in TP1 launch profile for the RVN
# packed-NVFP4 checkpoint carrying a STAMPED MTP graft
# (/models/qwen38-flash-next-uncensored) on the patched fork image. It
# reproduces the live-verified recipe of docs/rvn-w4a16.md ("Serving the RVN
# candidate (verified recipe)") flag for flag; refreshed 2026-09-25.
#
#   Usage: run_baseline.sh <gpu-index> [--dry-run] [--ple-offload[=GIB]] [--image IMG]
#
#   <gpu-index>       host GPU the single TP1 worker may use (required)
#   --dry-run         print the exact `docker run` command and exit; NO side
#                     effects, so the two real-run gates (host-RAM floor, image
#                     profile) are only announced, never enforced
#   --ple-offload[=G] HOST-RAM BUDGET GUARD for the pinned PLE host table
#                     (default floor 60 GiB, sized against the packed table's
#                     measured 26.8222 GiB pin). It is NOT a memory mitigation:
#                     see MEMORY BOUND below.
#   --image IMG       override the pinned profile image (default rvn-w4a16:sim,
#                     the patched Dockerfile.rvn-w4a16 build). A real run aborts
#                     unless that image passes the PROFILE GATE below.
#
# This is an EXPERIMENT. It never touches the production workers
# (lilith-vllm / lilith-vllm-b): distinct container name, served model name,
# port and cache dir, and the CDI device flag hands this container exactly one
# GPU. Nothing here stops, restarts or reconfigures anything but its own
# `--rm` container. Rollback = remove this container only.
#
# TEXT-ONLY PROFILE. The checkpoint config declares
# architectures=[Qwen4ExpForCausalLM], model_type=qwen4_exp_text, with no
# vision_config, so every inherited LIL multimodal flag is removed:
# --image-processor-backend, --mm-feature-transport, --limit-mm-data-per-request,
# --media-url-max-file-size-mb and SGLANG_IMAGE_MAX_PIXELS. Nothing in
# runtime/python/sglang/srt/server_args.py requires them for a text arch — they
# only configure the vision encoder path this checkpoint never builds.
#
# MODEL CONFIG SOURCE (no override file, no CLI override blob): the server reads
# the checkpoint's own /models/qwen38-flash-next-uncensored/config.json —
# architectures=[Qwen4ExpForCausalLM], model_type=qwen4_exp_text, no
# rope_scaling (max_position_embeddings=262144, so no YaRN),
# quant_algo=W4A16_NVFP4 with `*ple*` in exclude_modules and
# ple_embedding_dtype=nvfp4. That stamp is what selects the manifest-backed
# PACKED NVFP4 PLE host table (SGLANG_PLE_PACKED_NVFP4=1, loader in
# patches/0049-rvn-ple-packed-loader.patch), reconstructed bf16_direct at load
# (docs/rvn-ple-storage-schema.md §1) and pinned in host RAM by
# --ple-offload-embedding. Per patches/0034-embedded-model-overrides.patch an
# empty SGLANG_EMBEDDED_MODEL_OVERRIDES is a no-op, so the env file clears the
# key explicitly and no --json-model-override-args is passed at all.
#
# WHICH DIRECTORY THIS IS (2026-09-25 rename): /models/qwen38-flash-next-uncensored
# is the served directory formerly named /models/rvn-qwen38-ple-nvfp4-mtp — the
# grafted copy (config mtp_num_hidden_layers=1 plus the graft tool's stamp
# rvn_mtp_graft={source:/models/qwen38-flash-next,
# encoder_version:rvn-mtp-graft-r1, count:1}, with mtp_graft.json beside it;
# tools/rvn_ple/graft_mtp.py produced it, tools/rvn_ple/verify.py --graft sealed
# 7/7). The earlier no-MTP candidate /models/rvn-qwen38-ple-nvfp4 and the 168 GB
# source copy were retired, so re-deriving a graft needs convert.py against a
# fresh source copy first (docs/rvn-w4a16.md "Naming note").
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE="$SCRIPT_DIR/env.rvn-w4a16-baseline"

# Patched PROFILE image — what `docker build -f Dockerfile.rvn-w4a16
# -t rvn-w4a16:sim .` produces (docs/rvn-w4a16.md "Build and run"). This is
# deliberately NOT the base image the production workers run
# (localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284 = Dockerfile.rvn-w4a16:4),
# because that base lacks the patches this profile's flags require: 0047 adds
# the Qwen4ExpForCausalLM entry the checkpoint config declares, 0048 routes a
# quantized_layers-less W4A16_NVFP4 checkpoint through uniform ModelOpt FP4
# (without it --quantization modelopt_mixed below fails mixed-precision
# validation), 0051 makes Qwen4ExpForCausalLM eligible for
# --ple-offload-embedding, and 0057-0059 are what let the NEXTN flags below run
# against a stamped graft at all. Every flag below is otherwise unreachable on
# the base.
# Like deploy/run.py's --image default this is an explicit pinned reference; the
# pin is by tag only because the profile image is built locally and never pushed
# (docs/rvn-w4a16.md: "This base is local-only"), so no registry digest exists
# to pin. For a reproducible real run, pin by image Id
# (--image "$(docker image inspect -f '{{.Id}}' rvn-w4a16:sim)") or pass a
# freshly built tag. The mutable tag is made safe by the PROFILE GATE below,
# which hashes the image's source tree instead of trusting the tag.
IMAGE=${IMAGE:-rvn-w4a16:sim}
# Baseline-owned host cache root and container name. NEVER reuse the production
# namespaces /var/cache/sglang/qwen38-flash-next-kanadaj-a6d5284-tp1-gpu0[-b] or
# the shared /data/hicache/qwen38-flash-next-kanadaj-a6d5284. The name/dir pair
# below is the one the verified recipe runs under ("rvn-ple-nvfp4-0" is a worker
# slot, not a GPU index — the recipe pins it on GPU 1), so re-launching the
# verified server reproduces its caches; a second concurrent copy of this
# profile collides on the name and docker refuses it, which is the point.
CACHE_ROOT=/var/cache/sglang
CACHE_DIR=rvn-ple-nvfp4-0
NAME=rvn-ple-nvfp4
# Tokenizer and chat template both come from the checkpoint dir: the launch
# passes no --tokenizer-path (sglang defaults it to --model-path) and
# --chat-template names that dir's own chat_template.jinja, exactly as the
# verified recipe does. History: this profile used to mount the
# /nix/store/0l335cvq…-qwen38-unsloth-chat-template.jinja the production workers
# mount and point --tokenizer-path at the RVN tree to keep sglang off that
# tree's template. The two templates are NOT byte-identical (checkpoint copy
# 169 lines / 8952 bytes, sha256 c3cf9e34…; nix-store unsloth copy 183 lines /
# 9993 bytes, sha256 12827f24…, which adds system-message merging and a
# reasoning_effort high→xhigh mapping), so the "unchanged template bytes"
# premise behind that mount no longer holds and the recipe serves the
# checkpoint's own template instead.

DRY_RUN=0
PLE_MIN_FREE_GIB=60
GPU=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --ple-offload) shift ;;
    --ple-offload=*) PLE_MIN_FREE_GIB=${1#*=}; shift ;;
    --image) IMAGE=$2; shift 2 ;;
    -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
# The packed NVFP4 PLE host table (SGLANG_PLE_PACKED_NVFP4=1 + --ple-offload-
# embedding) is built with torch.empty(..., pin_memory=True) in
# runtime/python/sglang/srt/models/qwen4_exp.py and pins 26.8222 GiB of
# page-locked host RAM per process — the number the server itself prints for
# this checkpoint (lilith 2026-09-25: "[rvn-ple] PLE table
# model.layers.1.ple.ple_embedding loaded from ple_storage.json: 320001536 rows
# x 160 cols, 26.8222 GiB pinned host bytes, reconstruction=bf16_direct"), i.e.
# 9/16 of the 47.6839 GiB a byte-per-element table of that shape needs
# (patches/0057 header). This image has NO file-backed or shared PLE path:
# patches/0032-shared-ple-host-table.patch is absent from
# patches/series.production and SGLANG_PLE_SHARED_DIR has zero references in the
# shipped runtime tree, so the table is always private and pinned. Nothing here
# can shrink it — --ple-offload therefore only sets the host-RAM floor this
# script enforces, and the --ulimit memlock=-1 below (as in production) is what
# lets the pinning succeed at all.
# HISTORY (do not re-apply without dropping the packed env): before the packed
# loader was adopted this profile ran the ORIGINAL BF16 lookup — the same
# 320001536x160 table at 2 bytes/element with no SGLANG_PLE_PACKED_* env — which
# pinned ~95 GiB per process and is why this floor used to be 100 GiB and why
# the guard once read "not launchable while both production workers run". The
# packed path is what the verified recipe, and this script, use.
if (( DRY_RUN )); then
  echo "NOTE: real run enforces >= ${PLE_MIN_FREE_GIB} GiB MemAvailable for the" \
       "26.8222 GiB pinned packed-NVFP4 PLE table (no file-backed path in this" \
       "image)." >&2
else
  avail_gib=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
  if (( avail_gib < PLE_MIN_FREE_GIB )); then
    echo "ERROR: ${avail_gib} GiB MemAvailable < required ${PLE_MIN_FREE_GIB} GiB" \
         "for the 26.8222 GiB pinned packed-NVFP4 PLE table." >&2
    echo "ERROR: free host RAM or raise the floor deliberately with" \
         "--ple-offload=<gib>; nothing was started." >&2
    exit 1
  fi
fi

# PROFILE GATE — unconditional for every real run, and it runs BEFORE `docker
# run` / before any host directory is created: the launch flags below need the
# profile patch set — 0047 (Qwen4ExpForCausalLM entry), 0048 (uniform
# W4A16_NVFP4 dispatch), 0051 (--ple-offload-embedding eligibility for the text
# arch) and, for the NEXTN flags, 0057-0059 (graft-aware draft gate, graft
# loader, draft-architecture remap) — and on an image without them the failure
# surfaces only deep inside the server, after this profile has already pinned
# 26.8 GiB of host RAM, so a wrong image must be caught here, not at model load.
# The check is the verifier that Dockerfile.rvn-w4a16:8 bakes into every profile
# image, run WITHOUT --apply: it hashes the image's own python/sglang tree
# against the post-patch inventory baked next to it in provenance/rvn-w4a16.json,
# i.e. every file in that inventory is checked byte-for-byte (never hardcode the
# file count or the patch range here: the baked profile range grew 0047-0052 →
# 0047-0053 mid-review when the mutable tag was rebuilt, which is exactly why the
# tag is not trusted).
# Rejected alternatives: `docker image inspect` labels — measured 2026-09-25, the
# base and the patched image carry byte-identical label sets (no LABEL added), so a
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
       "profile gate (see the PROFILE GATE block in this script)." >&2
else
  if gate=$(docker run --rm --pull never --network none \
              --entrypoint python3 "$IMAGE" -B \
              /opt/rvn-w4a16/scripts/verify_rvn_w4a16.py \
              --tree /sgl-workspace/sglang 2>&1); then
    echo "profile gate OK: $gate" >&2
  else
    echo "ERROR: image $IMAGE is not a verified rvn-w4a16 profile image; nothing" \
         "was started (no container, no cache dir, no pinned host RAM)." >&2
    # Name the cause rather than leaving a raw tool error to be interpreted: these
    # ways of failing mean different operator actions. Only positive evidence claims
    # a patch verdict — a bare docker daemon error must not read as "your tree is
    # corrupt". Every branch still aborts; only the explanation differs.
    if [[ $gate == *"can't open file '/opt/rvn-w4a16/scripts/verify_rvn_w4a16.py'"* ]]; then
      echo "ERROR: cause = the image has no /opt/rvn-w4a16 profile layer at all, so" \
           "it is the un-patched base (or a build without Dockerfile.rvn-w4a16, which" \
           "this gate rejects by design) — not a stale build." >&2
    elif [[ $gate == *"No such image"* ]]; then
      echo "ERROR: cause = $IMAGE is not present locally, and --pull never forbids" \
           "fetching it: no registry was contacted." >&2
    elif [[ $gate == *"Traceback (most recent call last)"* ]]; then
      echo "ERROR: cause = the profile verifier ran inside the image and rejected its" \
           "python/sglang tree: stale, partial or drifted patch set (traceback below)." >&2
    else
      echo "ERROR: cause = the probe itself failed (docker/image access, not a patch" \
           "verdict) — the indented output below says why." >&2
    fi
    echo "ERROR: build it with" >&2
    echo "         docker build -f Dockerfile.rvn-w4a16 -t rvn-w4a16:sim ." >&2
    echo "       or select another image with --image IMG / IMAGE=<ref>." >&2
    sed 's/^/       /' <<<"$gate" >&2
    exit 1
  fi
fi

# [launcher] keys of the env file: run_baseline.sh reads them to build CLI flags,
# so they are deliberately NOT passed to the container. The verified recipe
# passes exactly five -e keys and nothing else; exporting MODEL/MAX_*-style
# tuning knobs as well is how a printed dry-run drifts from the recipe it
# claims to reproduce (and how prod proves the split: its env says
# MAX_NUM_SEQS=16 while its CMD passes --max-running-requests=8).
LAUNCHER_KEYS=(
  MODEL
  SERVED_MODEL_NAME
  PORT
  TP_SIZE
  MAX_MODEL_LEN
  MAX_NUM_SEQS
  MAX_NUM_BATCHED_TOKENS
  GPU_MEMORY_UTILIZATION
)
is_launcher_key() {
  local key=$1 skip
  for skip in "${LAUNCHER_KEYS[@]}"; do
    [[ $key == "$skip" ]] && return 0
  done
  return 1
}

# Exact env: every non-launcher KEY=VALUE line of the env file, in file order.
env_args=()
while IFS= read -r line || [[ -n $line ]]; do
  [[ -z $line || $line == \#* ]] && continue
  if is_launcher_key "${line%%=*}"; then continue; fi
  env_args+=(-e "$line")
done < "$ENV_FILE"

# No GPU selection happens in the env. GPU pinning is the CDI device flag in
# cmd= below: docs/rvn-w4a16.md records that `--gpus all` +
# CUDA_VISIBLE_DEVICES does NOT isolate on this host and
# `--gpus '"device=N"'` is broken outright, while the verified recipe uses
# --device nvidia.com/gpu=<gpu> (live container: HostConfig.DeviceRequests =
# {Driver:cdi, DeviceIDs:[nvidia.com/gpu=1]}). Inside the container the visible
# GPU is always cuda:0, so no CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES is
# injected here either.
mounts=(
  -v /models:/models:ro                                        # checkpoints, read-only
  -v "$CACHE_ROOT/$CACHE_DIR":/root/.cache                     # baseline-owned caches
)

# Launch command inside the container. Values are expanded from the env file so
# the printed dry-run IS the command that would run; flag spelling follows the
# verified recipe verbatim.
INNER="exec python3 -m sglang.launch_server"
INNER+=" --model-path $MODEL"
# Checkpoint's own template (169 lines / 8952 bytes, sha256 c3cf9e34…). Explicit
# rather than left to auto-discovery so the served template is named in the run
# log; with no --tokenizer-path the tokenizer tree IS this dir, so discovery
# would land on the same file.
INNER+=" --chat-template $MODEL/chat_template.jinja"
INNER+=" --served-model-name $SERVED_MODEL_NAME"
INNER+=" --host 0.0.0.0 --port $PORT"
# TP1, Marlin MoE, BF16 dense compute (SGLANG_SM120_ONLINE_MXFP8=false in env).
INNER+=" --tp-size $TP_SIZE --quantization modelopt_mixed --moe-runner-backend marlin"
INNER+=" --kv-cache-dtype fp8_e4m3 --context-length $MAX_MODEL_LEN"
INNER+=" --mem-fraction-static $GPU_MEMORY_UTILIZATION --page-size 64"
INNER+=" --chunked-prefill-size $MAX_NUM_BATCHED_TOKENS --max-running-requests $MAX_NUM_SEQS"
# Decode CUDA graphs ON up to bs 8, prefill graphs OFF: the ~9x decode win
# measured for this profile (11.7 tok/s graphs-off → 103 tok/s with this pair,
# capture bs=[1,2,4], 0.12 GB, 4.2 s — docs/rvn-w4a16.md landmines). This pair
# REPLACES the old --cuda-graph-backend-decode=disabled +
# --cuda-graph-backend-prefill=disabled duo, and radix cache stays ENABLED: the
# old --disable-radix-cache is gone, because reuse is compatible with the
# graphs and with this checkpoint.
INNER+=" --cuda-graph-max-bs-decode=8 --disable-prefill-cuda-graph"
# Auto-detection driven by the served chat template, not a LIL value; kept
# because dropping them would change what /v1/chat/completions returns as
# content vs reasoning. Production's --default-chat-template-kwargs
# '{"reasoning_effort":"xhigh"}' is a LIL serving default and is NOT passed.
INNER+=" --reasoning-parser auto --tool-call-parser auto"
# CHECKPOINT-DECLARED VALUE DELIBERATELY OVERRIDDEN: --mamba-ssm-dtype bfloat16
# while the checkpoint config declares "mamba_ssm_dtype": "float32" — a
# memory-motivated choice matching production; drop it to serve the checkpoint's
# own numerics (the kv-cache-dtype override above is the other one).
# Hybrid-architecture backends (required to serve this arch at all).
INNER+=" --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer"
INNER+=" --max-mamba-cache-size 64 --mamba-radix-cache-strategy extra_buffer"
INNER+=" --mamba-track-interval 128 --mamba-ssm-dtype bfloat16"
# GDN MTP verify h-state cache mode: default is 'full' (caches h at every
# draft-token position); 'none' skips the intermediate caching and reconstructs
# h_K after verify, trading post-verify recovery compute for the
# intermediate_ssm buffer (server_args.py help). This is a NEXTN-verify memory
# choice, not an MTP on/off switch — the draft head itself is the graft's.
INNER+=" --gdn-mtp-cache-mode none"
# Packed NVFP4 PLE host lookup: env selects the manifest-backed packed table,
# the flag pins it in host RAM instead of building it as a CUDA tensor (see
# MEMORY BOUND above).
INNER+=" --ple-offload-embedding"
# NEXTN against the STAMPED graft — this is what the "0057 refuses NEXTN / RVN
# has no MTP" note was superseded for, and only for a stamped graft:
# patches/0057-rvn-nextn-draft-gate.patch still refuses --speculative-algorithm
# NEXTN for an RVN text checkpoint that ships no draft layer (any dir whose
# config lacks the rvn_mtp_graft stamp), and that refusal is correct — without a
# graft the draft worker rebuilds the full 48-layer target and puts the PLE
# table on the device. patches/0058-rvn-mtp-graft-loader.patch accepts a graft
# stamped rvn_mtp_graft={encoder_version:rvn-mtp-graft-r1, count:1} in the text
# contract, and patches/0059-rvn-mtp-draft-remap.patch remaps the draft to
# Qwen4ExpForCausalLMMTP and defaults the draft path to the target dir, so no
# --speculative-draft-model-path is needed. $MODEL carries that stamp
# (mtp_graft.json; tools/rvn_ple/verify.py --graft → 7/7), which is the whole
# reason these flags are legal here.
INNER+=" --speculative-algorithm NEXTN --speculative-num-steps 3"
INNER+=" --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
INNER+=" --speculative-draft-model-quantization modelopt_mixed"
INNER+=" --speculative-moe-runner-backend marlin"
# Serial loading + 2 threads (plus PYTORCH_CUDA_ALLOC_CONF in the env) is what
# keeps the load peak inside one 96 GB card; 0052 releases the loader-format MoE
# storage during the Marlin repack.
INNER+=" --model-loader-extra-config '{\"enable_multithread_load\":false,\"num_threads\":2}'"
# Deliberately absent (none of these is in the verified recipe): the
# --json-model-override-args YaRN blob, --tokenizer-path (defaults to $MODEL),
# --enable-metrics / --uvicorn-access-log-exclude-prefixes,
# --startup-weight-load-mode=serial, --enable-hierarchical-cache and all
# --hicache-* flags, --enable-cache-report, --disable-custom-all-reduce (the
# live TP1 workers do not pass it either), and all vision/mm flags (text-only).

# --entrypoint /bin/bash is REQUIRED, not cosmetic: Dockerfile.rvn-w4a16:10 sets
# ENTRYPOINT to ["python3","-m","sglang.launch_server"], so without the override
# `docker run … $IMAGE /bin/bash -lc …` executes
# `python3 -m sglang.launch_server /bin/bash -lc …` and argparse dies on the
# positional "/bin/bash" before the server ever starts. deploy/run.py:28 passes
# `--entrypoint sglang` and docs/rvn-w4a16.md:76 passes `--entrypoint python3`
# for the same reason; here the wrapper shell is what the quoted
# --model-loader-extra-config in INNER needs, so the entrypoint becomes bash and
# the launch line stays an argument to it.
cmd=(
  docker run --rm
  --name "$NAME"
  --device "nvidia.com/gpu=$GPU"                                  # CDI pin, one GPU
  --ipc host --network host
  --shm-size 32g
  --ulimit memlock=-1 --ulimit stack=67108864
  "${env_args[@]}"
  "${mounts[@]}"
  -w /sgl-workspace/sglang
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
mkdir -p "$CACHE_ROOT/$CACHE_DIR"
exec "${cmd[@]}"

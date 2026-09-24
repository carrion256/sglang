# RVN W4A16 baseline profile (experimental, opt-in)

- Launches the **UNCHANGED** RVN checkpoint (`/models/qwen38-flash-next-uncensored`) on the patched fork image with **all inherited LIL overrides removed** and v1 safety settings.
- One TP1 worker, `--rm`, baseline-owned names/caches. Never touches `lilith-vllm` / `lilith-vllm-b`.
- Reference = `docker inspect lilith-vllm` / `lilith-vllm-b` (image `localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284`), **not** `deploy/production/args.json`, which is a stale TP2/524288 profile that diverges from the live TP1/262144 containers.

## Run

- Print the exact command, zero side effects:
  `bash deploy/rvn-w4a16-baseline/run_baseline.sh 0 --dry-run`
- Launch (host-RAM guard enforced): `bash deploy/rvn-w4a16-baseline/run_baseline.sh <free-gpu>`
- `--ple-offload[=GIB]` = host-RAM **budget guard** (floor, default 100 GiB `MemAvailable`, positive integer only — `abc`/`0`/negative exit 2 instead of silently disabling the gate); it is not a memory mitigation — see Memory bound.
- `--image IMG` overrides the pinned default image.
- Serving parameters: `env.rvn-w4a16-baseline`, each key tagged `[prod]` / `[plan]` / `[launcher]` (the last are inert inside the container, as production proves: env `MAX_NUM_SEQS=16` vs CMD `--max-running-requests=8`).

## Settings vs production

Production column = live worker `lilith-vllm` (`-b` differs only in port 8102, GPU 1, and cache dir `…-gpu0-b`).

| Setting | Production (live) | Baseline v1 | Why |
|---|---|---|---|
| `--model-path` | `/models/qwen38-flash-next` | `/models/qwen38-flash-next-uncensored` | experiment subject, bytes unchanged |
| `--tokenizer-path` | unset (defaults to model path) | `/models/qwen38-flash-next-uncensored` (parameterized `TOKENIZER_PATH`) | default is the RVN tree itself, so no cross-tree dependency |
| `--tokenizer-path` contingency | n/a | set to `/models/qwen38-flash-next` only if the finished RVN download still lacks tokenizer files | observed mid-`hf download` (2026-09-24) that the tree then had no `tokenizer.json`/`tokenizer_config.json`/`vocab.json`/`merges.txt`, so `--model-path` alone failed at tokenizer init; the LIL tree's tokenizer is the same 248320-token vocab so bytes stay unchanged. Orchestrator confirms presence at download completion, then this fallback is used or deleted |
| `--chat-template` | `/nix/store/0l335cvq…-qwen38-unsloth-chat-template.jinja` → `/etc/qwen38/chat-template.jinja` | same file, same bytes | contract: template bytes unchanged. **Load-bearing, not cosmetic**: sglang auto-discovers a tokenizer-path tree's own `chat_template.jinja` (which differs byte-wise from this one), so the explicit path is what guarantees the required template bytes |
| `--served-model-name` | `qwen3.8-flash-next` | `rvn-w4a16-baseline` | model-name isolation |
| `--port` | `8101` / `8102` | `8111` | port isolation on host network |
| `--tp-size` | `1` | `1` | TP1 single worker |
| `--max-running-requests` | `8` | `1` | concurrency 1 initially |
| CUDA graphs | decode graphs (`--cuda-graph-max-bs-decode=8`), prefill graphs off | `--cuda-graph-backend-decode=disabled --cuda-graph-backend-prefill=disabled` | eager execution, capture OFF; `--disable-cuda-graph` is a `DeprecatedStoreTrueAction` (server_args.py:8860) so it is not used |
| `--moe-runner-backend` | `flashinfer_cutlass` | `marlin` | v1 plan: Marlin MoE (valid choice in `MOE_RUNNER_BACKEND_CHOICES`); **SM120 risk below** |
| Dense compute | `SGLANG_SM120_ONLINE_MXFP8=false` | same | BF16 dense compute, no online FP8 |
| `--quantization` | `modelopt_mixed` | same | reads RVN `hf_quant_config`/`quantization_config` (`quant_algo=W4A16_NVFP4`) |
| `--kv-cache-dtype` | `fp8_e4m3` | same | overrides the checkpoint's declared `"kv_cache_quant_algo": null` (config.json:140) — a deliberate memory/precision choice kept for v1 parity, not a checkpoint value |
| Model config source | baked/embedded override + `--json-model-override-args` YaRN blob | **RVN's own `config.json`**, no override file, no CLI blob | checkpoint-native: `rope_type=default`, `max_position_embeddings=262144`, `mtp_num_hidden_layers=0`, `exclude_modules` contains `*ple*` → BF16 PLE (`docs/rvn-ple-storage-schema.md` §1 `bf16_direct`) |
| `SGLANG_EMBEDDED_MODEL_OVERRIDES` | may point at a baked LIL text_config | `-e SGLANG_EMBEDDED_MODEL_OVERRIDES=` (empty ⇒ no-op per patch 0034) | never inherit the LIL override file |
| PLE lookup | `SGLANG_PLE_PACKED_NVFP4=1` + `SGLANG_PLE_PACKED_FP8_REFERENCE=1` + `--ple-offload-embedding` + `/nix/store/…packed_ple.py` host overlay | **ORIGINAL BF16 PLE host lookup**: `--ple-offload-embedding` only, no packed envs, no code overlay | LIL packed paths are LIL overrides; RVN PLE is BF16 by construction |
| PLE host table | pinned private tables per process | pinned private tables per process, **~95 GiB unavoidable** | this image has no shared/file-backed path: patch 0032 is absent from `patches/series.production` and `SGLANG_PLE_SHARED_DIR` has zero references in the runtime tree |
| Private draft head | `SGLANG_PRIVATE_DRAFT_NVFP4_A16=1` | **not set** | no draft head at all |
| Speculation | `NEXTN` 3/1/4 + `--speculative-draft-model-quantization=modelopt_mixed` + `--speculative-moe-runner-backend` | **all `--speculative-*` omitted** | RVN has no MTP (`mtp_num_hidden_layers=0`) |
| `--gdn-mtp-cache-mode` | `none` | `none`, kept explicitly | default is `full`, so passing it is what grounds MTP-off at the flag level |
| YaRN | YaRN factor 4.0 → 1M window | **omitted**, native 262144 | no extrapolation |
| `--context-length` | `262144` | `32768` start → extend to `262144` after v1 validation | short window first; 262144 is the checkpoint-native maximum |
| Prefix reuse | radix cache + `--enable-cache-report` | `--disable-radix-cache`, no cache report | reuse OFF initially |
| HiCache | `--enable-hierarchical-cache` + all `--hicache-*` + **shared** `/data/hicache/qwen38-flash-next-kanadaj-a6d5284` mount + `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` | **all omitted, no mount** | reuse OFF; that hicache dir is shared by both replicas, so mounting it would have crossed the isolation line |
| Vision / multimodal | `--image-processor-backend=torchvision`, `--mm-feature-transport=cpu`, `--limit-mm-data-per-request`, `--media-url-max-file-size-mb=8`, `SGLANG_IMAGE_MAX_PIXELS` | **all removed** | inherited LIL defaults; RVN is `Qwen4ExpForCausalLM` / `qwen4_exp_text` with no `vision_config`, and no code path requires them for a text arch |
| `--default-chat-template-kwargs` | `{"reasoning_effort":"xhigh"}` | **not passed** | LIL serving default, not a template byte |
| `--reasoning-parser` / `--tool-call-parser` | `auto` / `auto` | kept at `auto` | template-driven auto-detection; dropping them changes what the chat API returns as content vs reasoning — a behavior regression unrelated to the RVN experiment |
| `--disable-custom-all-reduce` | **not in the live CMD** (only in the stale `deploy/production/args.json`) | not passed | matches live TP1 workers |
| Mamba / linear-attn | `flashinfer` prefill+decode, cache 64, `extra_buffer`, track 128, `--mamba-ssm-dtype=bfloat16` | same values | backends are architecture-required; `bfloat16` **overrides the checkpoint's declared `"mamba_ssm_dtype": "float32"`** (config.json:79) — a numerics-affecting, memory-motivated v1 choice matching production, drop it to serve checkpoint numerics |
| Model loading | `--model-loader-extra-config '{"enable_multithread_load":false,"num_threads":2}'`, `--startup-weight-load-mode=serial` | same | inherited, unchanged |
| Host caches | `/var/cache/sglang/qwen38-flash-next-kanadaj-a6d5284-tp1-gpu0[-b]` | `/var/cache/sglang/rvn-w4a16-baseline-<gpu>` | Triton/inductor/tilelang isolation |
| HF cache mount | `/home/common/.cache/huggingface` | not mounted | model is local |
| Container name | `lilith-vllm` / `lilith-vllm-b` | `rvn-w4a16-baseline-<gpu>` | name isolation |
| Docker mechanics | `--gpus all` via CDI (`nvidia.com/gpu=all`) + `CUDA_VISIBLE_DEVICES`, `--ipc=host`, `--net=host`, `--shm-size=34359738368`, ulimits `memlock=-1`/`nofile=1048576`/`stack=67108864`, `/models` ro | all carried identically | "equivalent launch" requires these; only the GPU index, name, port and cache dir differ |

## Memory bound (read before a real run)

- ORIGINAL BF16 PLE host lookup (`--ple-offload-embedding`, no packed envs) builds the n-gram table with `torch.empty(..., pin_memory=True)` in `runtime/python/sglang/srt/models/qwen4_exp.py`: **~95 GiB page-locked host RSS per process**.
- There is **no file-backed or shared BF16 PLE path in this image**: `patches/0032-shared-ple-host-table.patch` is not in `patches/series.production`, and `SGLANG_PLE_SHARED_DIR` has zero references in the runtime tree (it exists only inside `PackedPLEStorage`, the packed-NVFP4 WP2 path this profile never enables). Production's own `SGLANG_PLE_SHARED_DIR` value is vestigial for the same reason.
- So `--ple-offload` cannot shrink the table; it sets the `MemAvailable` floor (default 100 GiB) the launcher enforces before starting anything, and the run aborts below it. `--ulimit memlock=-1` (as in production) is what makes the pinning possible.
- **Measured on lilith 2026-09-24: `MemAvailable: 34 GB`** (`awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo` → `34`), i.e. far below the 100 GiB floor. As written this profile is **not launchable on this host while both production workers run** — the guard aborts before starting anything (verified: `--ple-offload=999999` → exit 1, nothing created). Corollary: running the baseline needs a host with ~100 GiB spare pinned-capable RAM, or one production worker stopped, or the packed NVFP4 (WP2) PLE path instead of BF16.

## Risks and limits of the dry-run gate

- `--dry-run` proves only the command text; it cannot prove the server starts. Two known live-start risks: `--moe-runner-backend=marlin` on SM120 is non-default (`--fp4-gemm-backend` help says auto picks `marlin` on SM80–SM90 and `flashinfer_cutlass` on SM120), and the ~95 GiB pinned allocation needs real host headroom.
- If the finished RVN download still lacks tokenizer files and the LIL fallback is used, that is the profile's **single deliberate LIL-tree dependency**, justified by "tokenizer bytes unchanged" (same 248320 vocab). Any vocab/token-id mismatch between RVN `config.json` and that tokenizer fails at load — treat it as a live-start validation item next to Marlin-on-SM120.
- `MAX_MODEL_LEN` is the single knob for the 32768 → 262144 extension; 262144 is the checkpoint-native maximum, so no rope override is involved at either value.

## Coexistence checklist

- [ ] GPU index **free**: GPU0 = `lilith-vllm`, GPU1 = `lilith-vllm-b` (`nvidia-smi`).
- [ ] `free -g` shows ~100 GiB pinned-capable spare — enforced by the launcher, not skippable.
- [ ] Port `8111` free (`ss -ltnp | grep 8111`); prod keeps 8101/8102.
- [ ] Caches baseline-only: `/var/cache/sglang/rvn-w4a16-baseline-<gpu>`; prod's `…-kanadaj-a6d5284-tp1-gpu0[-b]` untouched.
- [ ] Printed command contains no `/data/hicache` mount (shared by both replicas), no `SGLANG_PLE_*` env, no `packed_ple.py` overlay, no vision flags, no `--json-model-override-args`.
- [ ] `docker ps` afterwards shows the two prod workers plus only `rvn-w4a16-baseline-<gpu>`.
- [ ] Dry-run output reviewed before the real run.

## Rollback

- `docker rm -f rvn-w4a16-baseline-<gpu>` — the whole rollback; container is `--rm` with no persistent state.
- Optional: `rm -rf /var/cache/sglang/rvn-w4a16-baseline-<gpu>`.
- Production workers, their caches and the shared HiCache store are never modified.

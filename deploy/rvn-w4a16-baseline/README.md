# RVN W4A16 baseline profile (experimental, opt-in)

- Launches the **UNCHANGED** RVN checkpoint (`/models/qwen38-flash-next-uncensored`) on the patched fork image with **all inherited LIL overrides removed** and v1 safety settings.
- One TP1 worker, `--rm`, baseline-owned names/caches. Never touches `lilith-vllm` / `lilith-vllm-b`.
- Reference = `docker inspect lilith-vllm` / `lilith-vllm-b` (image `localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284`), **not** `deploy/production/args.json`, which is a stale TP2/524288 profile that diverges from the live TP1/262144 containers.

## Run

- Print the exact command, zero side effects:
  `bash deploy/rvn-w4a16-baseline/run_baseline.sh 0 --dry-run`
- Launch (both real-run gates enforced first — host RAM, then image profile): `bash deploy/rvn-w4a16-baseline/run_baseline.sh <free-gpu>`
- `--ple-offload[=GIB]` = host-RAM **budget guard** (floor, default 100 GiB `MemAvailable`, positive integer only — `abc`/`0`/negative exit 2 instead of silently disabling the gate); it is not a memory mitigation — see Memory bound.
- `--image IMG` (or `IMAGE=…`) overrides the pinned default, which is the **patched profile image** `rvn-w4a16:sim` — the output of `docker build -f Dockerfile.rvn-w4a16 -t rvn-w4a16:sim .`, **not** the `localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284` base the production workers run (see *Image and profile gate*).
- Serving parameters: `env.rvn-w4a16-baseline`, each key tagged `[prod]` / `[plan]` / `[launcher]` (the last are inert inside the container, as production proves: env `MAX_NUM_SEQS=16` vs CMD `--max-running-requests=8`).

## Image and profile gate

- **Default image = the patched profile image** `rvn-w4a16:sim`, i.e. `docker build -f Dockerfile.rvn-w4a16 -t rvn-w4a16:sim .` (docs/rvn-w4a16.md "Build and run"). The base image `localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284` — what `lilith-vllm` / `lilith-vllm-b` run, and what this launcher previously defaulted to — **cannot serve this profile**: no 0047 means no `Qwen4ExpForCausalLM` entry for the RVN config's `architectures`, without 0048 `--quantization=modelopt_mixed` fails mixed-precision validation on a `quantized_layers`-less `W4A16_NVFP4` checkpoint, and without 0051 `--ple-offload-embedding` raises for this arch. All three fail only at model load — after ~95 GiB of host RAM is already pinned — so they must be caught before `docker run`.
- **Gate:** before any container or host directory is created, the launcher runs the verifier that `Dockerfile.rvn-w4a16` bakes into every profile image, WITHOUT `--apply`, so it hashes the image's own `python/sglang` tree against the post-patch inventory in `provenance/rvn-w4a16.json`:
  ```
  docker run --rm --pull never --network none --entrypoint python3 "$IMAGE" \
    -B /opt/rvn-w4a16/scripts/verify_rvn_w4a16.py --tree /sgl-workspace/sglang
  ```
  Non-zero ⇒ `exit 1` printing the verifier output: no container, no cache dir, no pinned RAM. A pass prints `profile gate OK: {…full_tree_verified":true…}` to stderr, so the run log records which tree actually booted. Measured 2026-09-25: `rvn-w4a16:sim` → exit 0 in **1.4 s**; the base image → exit 2 (it has no verifier at all, which is itself the fail-closed answer).
- **Why that probe:** a `docker image inspect` label check is impossible here — the base and the patched image carry byte-identical label sets (`Dockerfile.rvn-w4a16` adds no `LABEL`), so a label gate would reject even a correct image; a `python3 -c "import sglang.srt.models.qwen4_exp_text_adapter"` probe is slower (it drags in torch/CUDA) and proves only 0047, leaving 0048 and 0051 unchecked. The verifier is the build's own gate, so launcher and image cannot disagree about what "has the profile" means, and it also catches a drifted/partially-patched tree. `--pull never` keeps a name collision from pulling a same-named image off a registry instead of using the local one.
- **Why `--entrypoint /bin/bash`:** `Dockerfile.rvn-w4a16:10` sets `ENTRYPOINT ["python3","-m","sglang.launch_server"]`, so the previous form `docker run … "$IMAGE" /bin/bash -lc …` executes `python3 -m sglang.launch_server /bin/bash -lc …` and argparse dies on the `/bin/bash` positional before the server starts. The launcher now passes `--entrypoint /bin/bash`, mirroring `deploy/run.py:28` (`--entrypoint sglang`) and the manual recipe at `docs/rvn-w4a16.md:69` (`--entrypoint python3`). Bash rather than `python3`, because `INNER` depends on shell quoting for `--model-loader-extra-config`.
- **Tag pinning:** `rvn-w4a16:sim` is a mutable local tag — the profile image is never pushed, so unlike `deploy/run.py`'s digest-pinned default there is no registry digest to pin. For a reproducible run use `--image "$(docker image inspect -f '{{.Id}}' rvn-w4a16:sim)"`; the tag is safe as a default because the gate verifies tree bytes, never the tag name.
- **What the gate does NOT prove:** it checks the image's tree against the provenance copy baked into that same image, so it catches the defects it exists for — wrong image selected, stale build, partially applied or drifted tree, missing verifier (the base) — but it is not a supply-chain attestation of the build itself; a rebuilt image with a self-consistent tree+provenance pair passes by construction. Build provenance remains the job of `scripts/verify_rvn_w4a16.py` run from this repo against the un-patched base tree (`docs/rvn-w4a16.md`), and `rvn-w4a16:sim`'s status string is printed verbatim by the gate so a profile drift (e.g. an image built from a different patch range than the tree you are reviewing) is visible in the run log.

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

- `--dry-run` proves only the command text; it cannot prove the server starts, and — by design, it promises zero side effects — it enforces neither gate, so a wrong image still prints a plausible command. Check the image separately with the one-liner in *Image and profile gate*, or read the `profile gate OK:` line the real run prints before starting. Two known live-start risks: `--moe-runner-backend=marlin` on SM120 is non-default (`--fp4-gemm-backend` help says auto picks `marlin` on SM80–SM90 and `flashinfer_cutlass` on SM120), and the ~95 GiB pinned allocation needs real host headroom.
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
- [ ] Image gate passed: stderr shows `profile gate OK: {…}` for the image you are about to run (default `rvn-w4a16:sim`), printed before any container exists.

## Rollback

- `docker rm -f rvn-w4a16-baseline-<gpu>` — the whole rollback; container is `--rm` with no persistent state.
- Optional: `rm -rf /var/cache/sglang/rvn-w4a16-baseline-<gpu>`.
- Production workers, their caches and the shared HiCache store are never modified.

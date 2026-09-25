# RVN W4A16 baseline profile (experimental, opt-in)

- Launches the RVN packed-NVFP4 checkpoint **with the stamped MTP graft** (`/models/qwen38-flash-next-uncensored`, the 2026-09-25 rename of `/models/rvn-qwen38-ple-nvfp4-mtp`) on the patched fork image, reproducing the live-verified recipe of [`docs/rvn-w4a16.md`](../../docs/rvn-w4a16.md) ("Serving the RVN candidate (verified recipe)" + its landmine bullets) flag for flag.
- One TP1 worker, `--rm`, baseline-owned names/caches, one CDI-pinned GPU. **Never touches `lilith-vllm` / `lilith-vllm-b`**: nothing here stops, restarts or reconfigures another container or GPU; rollback removes this container only.
- Refreshed 2026-09-25 to match the recipe the server now serving (`rvn-serve-nextn`, GPU 1, image `rvn-w4a16:sim`) actually runs: 175-197 tok/s with NEXTN. The pre-refresh profile was a *pre-graft* design (BF16 PLE, graphs off, radix off, no speculation) and is what the sections below now document as history.
- Reference for production = `docker inspect lilith-vllm` / `lilith-vllm-b` (image `localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284`), **not** `deploy/production/args.json`, which is a stale TP2/524288 profile that diverges from the live TP1/262144 containers.

## Run

- Print the exact command, zero side effects (first-class, and the thing to read before any real run):
  `bash deploy/rvn-w4a16-baseline/run_baseline.sh 1 --dry-run`
- Launch (both real-run gates enforced first — host RAM floor, then image profile): `bash deploy/rvn-w4a16-baseline/run_baseline.sh <free-gpu>`
- `--ple-offload[=GIB]` = host-RAM **budget guard** (floor, default **60 GiB** `MemAvailable`, positive integer only — `abc`/`0`/negative exit 2 instead of silently disabling the gate); it is not a memory mitigation — see *Memory bound*.
- `--image IMG` (or `IMAGE=…`) overrides the pinned default, which is the **patched profile image** `rvn-w4a16:sim` — the output of `docker build -f Dockerfile.rvn-w4a16 -t rvn-w4a16:sim .`, **not** the `localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284` base the production workers run (see *Image and profile gate*).
- Serving parameters: `env.rvn-w4a16-baseline`. The file is split in two: the **CONTAINER ENV** keys are passed as `-e` in file order — exactly the five the verified recipe passes, nothing added — and the **LAUNCHER PARAMETERS** (`MODEL`, `SERVED_MODEL_NAME`, `PORT`, `TP_SIZE`, `MAX_*`, `GPU_MEMORY_UTILIZATION`) are read by the launcher to build CLI flags and deliberately *not* exported (`LAUNCHER_KEYS` in `run_baseline.sh`; production proves the split — its env says `MAX_NUM_SEQS=16` while its CMD passes `--max-running-requests=8`).

## Image and profile gate

- **Default image = the patched profile image** `rvn-w4a16:sim`, i.e. `docker build -f Dockerfile.rvn-w4a16 -t rvn-w4a16:sim .` (docs/rvn-w4a16.md "Build and run"). The base image `localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284` — what `lilith-vllm` / `lilith-vllm-b` run — **cannot serve this profile**: no 0047 ⇒ no `Qwen4ExpForCausalLM` entry for the checkpoint's `architectures`; without 0048 `--quantization modelopt_mixed` fails mixed-precision validation on a `quantized_layers`-less `W4A16_NVFP4` checkpoint; without 0051 `--ple-offload-embedding` raises for this arch; and without 0057-0059 the NEXTN flags below are refused (or, pre-0057, silently build a second 48-layer target). All of that surfaces only at model load — after 26.8 GiB of host RAM is already pinned — so it must be caught before `docker run`.
- **Gate:** before any container or host directory is created, the launcher runs the verifier that `Dockerfile.rvn-w4a16` bakes into every profile image, WITHOUT `--apply`, so it hashes the image's own `python/sglang` tree against the post-patch inventory in `provenance/rvn-w4a16.json`:
  ```
  docker run --rm --pull never --network none --entrypoint python3 "$IMAGE" \
    -B /opt/rvn-w4a16/scripts/verify_rvn_w4a16.py --tree /sgl-workspace/sglang
  ```
  Non-zero ⇒ `exit 1` printing the verifier output: no container, no cache dir, no pinned RAM. A pass prints `profile gate OK: {…full_tree_verified":true…}` to stderr, so the run log records which tree actually booted. Measured 2026-09-25: `rvn-w4a16:sim` → exit 0 in **1.4 s**; the base image → exit 2 (it has no verifier at all, which is itself the fail-closed answer).
  - **Alternative builds are rejected by design:** the probe is the profile's own baked path, so an image built without `Dockerfile.rvn-w4a16`'s `COPY` lines fails even if its tree happens to be correct — the launcher accepts only profile builds. Failure text always names the cause from positive evidence: no `/opt/rvn-w4a16` verifier layer ⇒ un-patched base (or a non-profile build), `No such image` ⇒ the ref is absent locally and `--pull never` refused to fetch it, a Python traceback ⇒ the verifier ran and refused this image's tree, anything else ⇒ probe/docker failure rather than a patch verdict. Only non-zero matters: every case aborts.
- **Why that probe:** a `docker image inspect` label check is impossible here — the base and the patched image carry byte-identical label sets (`Dockerfile.rvn-w4a16` adds no `LABEL`), so a label gate would reject even a correct image; a `python3 -c "import sglang.srt.models.qwen4_exp_text_adapter"` probe is slower (it drags in torch/CUDA) and proves only 0047. The verifier is the build's own gate, so launcher and image cannot disagree about what "has the profile" means, and it also catches a drifted/partially-patched tree. `--pull never` keeps a name collision from pulling a same-named image off a registry instead of using the local one.
- **Why `--entrypoint /bin/bash`:** `Dockerfile.rvn-w4a16:10` sets `ENTRYPOINT ["python3","-m","sglang.launch_server"]`, so `docker run … "$IMAGE" /bin/bash -lc …` executes `python3 -m sglang.launch_server /bin/bash -lc …` and argparse dies on the `/bin/bash` positional before the server starts. The launcher therefore passes `--entrypoint /bin/bash`, mirroring `deploy/run.py:28` (`--entrypoint sglang`) and the manual recipe at `docs/rvn-w4a16.md:76` (`--entrypoint python3`). Bash rather than `python3`, because `INNER` depends on shell quoting for `--model-loader-extra-config` — and because `tests/test_rvn_w4a16_packaging.py` pins the `--entrypoint /bin/bash … -lc "$INNER"` form. This is the **only** textual difference from the manual recipe: after `bash -lc` execs `exec python3 -m sglang.launch_server …`, the argv is identical (verified token-for-token, see *Recipe equivalence*).
  - Verified against the real image rather than only the printed command (2026-09-25): `docker run --rm --pull never --network none --entrypoint /bin/bash rvn-w4a16:sim -lc 'echo NEW_FORM_OK'` prints `NEW_FORM_OK` and exits 0, while the previous form prints **no** marker and exits 2 with the `launch_server` usage dump.
- **Tag pinning:** `rvn-w4a16:sim` is a mutable local tag — the profile image is never pushed, so unlike `deploy/run.py`'s digest-pinned default there is no registry digest to pin. For a reproducible run use `--image "$(docker image inspect -f '{{.Id}}' rvn-w4a16:sim)"`; the tag is safe as a default because the gate verifies tree bytes, never the tag name.
- **What the gate does NOT prove:** it checks the image's tree against the provenance copy baked into that same image, so it catches the defects it exists for — wrong image selected, stale build, partially applied or drifted tree, missing verifier (the base) — but it is not a supply-chain attestation of the build itself; a rebuilt image with a self-consistent tree+provenance pair passes by construction. Build provenance remains the job of `scripts/verify_rvn_w4a16.py` run from this repo against the un-patched base tree, and the gate prints the image's status string verbatim, so a profile drift (an image built from a different patch range than the tree you are reviewing) is visible in the run log.

## Recipe equivalence

`run_baseline.sh 1 --dry-run` prints 101 argv tokens. Compared against the live-verified recipe (`docs/rvn-w4a16.md` recipe + its NEXTN landmine additions, `SGLANG_PRIVATE_DRAFT_NVFP4_A16=1`, served dir `/models/qwen38-flash-next-uncensored`), the printed command is **identical token-for-token** after normalizing the one wrapper difference the packaging test pins:

| | manual recipe | `run_baseline.sh` |
|---|---|---|
| entrypoint | `--entrypoint python3 rvn-w4a16:sim -m sglang.launch_server …` | `--entrypoint /bin/bash rvn-w4a16:sim -lc "exec python3 -m sglang.launch_server …"` |

Everything else — `--name rvn-ple-nvfp4`, `--device nvidia.com/gpu=$GPU`, `--ipc host`, `--network host`, `--shm-size 32g`, `--ulimit memlock=-1`, `--ulimit stack=67108864`, the five `-e` keys in order, both `-v` mounts, `-w /sgl-workspace/sglang` and all 33 launch flags in recipe order — matches, and the two host-side ulimit values match what the running container reports (`memlock` soft=hard=-1, `stack` soft=hard=67108864, `ShmSize=34359738368`, `DeviceRequests={Driver:cdi, DeviceIDs:[nvidia.com/gpu=1]}`).

## Settings vs production

Production column = live worker `lilith-vllm` (`-b` differs only in port 8102, GPU 1, and cache dir `…-gpu0-b`).

| Setting | Production (live) | Baseline = verified recipe | Why |
|---|---|---|---|
| `--model-path` | `/models/qwen38-flash-next` | `/models/qwen38-flash-next-uncensored` | experiment subject: packed NVFP4 + stamped MTP graft (2026-09-25 rename of `/models/rvn-qwen38-ple-nvfp4-mtp`; the earlier no-MTP candidate `/models/rvn-qwen38-ple-nvfp4` and the 168 GB source copy are retired) |
| `--tokenizer-path` | unset (defaults to model path) | **not passed** (defaults to the same dir) | tokenizer + template are the checkpoint's own: the dir carries `tokenizer.json` + `tokenizer_config.json` (252 files; `vocab.json`/`merges.txt` are absent and the fast-tokenizer path does not need them), and the serving container proves it loads with no `--tokenizer-path`. The old mid-download contingency that pointed this at `/models/qwen38-flash-next` is gone |
| `--chat-template` | `/nix/store/0l335cvq…-qwen38-unsloth-chat-template.jinja` → `/etc/qwen38/chat-template.jinja` | `$MODEL/chat_template.jinja`, no mount | the recipe serves the checkpoint's own template. It is **not** byte-identical to the nix-store copy the profile used to mount: 169 lines / 8952 B (`sha256 c3cf9e34…`) vs 183 lines / 9993 B (`sha256 12827f24…`), the unsloth copy adding system-message merging and a `reasoning_effort` high→xhigh mapping. Explicit rather than auto-discovered so the served template is named in the run log |
| `--served-model-name` | `qwen3.8-flash-next` | `rvn-ple-nvfp4` | model-name isolation (metrics, client routing, cache namespaces) |
| `--port` | `8101` / `8102` | `8111` | port isolation on host network |
| `--tp-size` | `1` | `1` | TP1 single worker |
| `--max-running-requests` | `8` | `4` | recipe value; fits the KV/mamba pools alongside the draft |
| CUDA graphs | decode graphs (`--cuda-graph-max-bs-decode=8`), prefill graphs off | `--cuda-graph-max-bs-decode=8 --disable-prefill-cuda-graph` | the ~9x decode win (11.7 → 103 tok/s). **Replaces** the old `--cuda-graph-backend-decode=disabled --cuda-graph-backend-prefill=disabled` pair (`--disable-cuda-graph` is a `DeprecatedStoreTrueAction`, so it is not used either) |
| Prefix reuse | radix cache + `--enable-cache-report` | radix cache **enabled**, no cache report | the old `--disable-radix-cache` is dropped: reuse is compatible with the decode graphs and with this checkpoint |
| `--moe-runner-backend` | `flashinfer_cutlass` | `marlin` | recipe value; **SM120 risk below** |
| Dense compute | `SGLANG_SM120_ONLINE_MXFP8=false` | same | BF16 dense compute, no online FP8 |
| `--quantization` | `modelopt_mixed` | same | reads `hf_quant_config`/`quantization_config` (`quant_algo=W4A16_NVFP4`) |
| `--kv-cache-dtype` | `fp8_e4m3` | same | overrides the checkpoint's declared `"kv_cache_quant_algo": null` — deliberate memory/precision choice, not a checkpoint value |
| Model config source | baked/embedded override + `--json-model-override-args` YaRN blob | **the checkpoint's own `config.json`**, no override file, no CLI blob | checkpoint-native: no `rope_scaling`, `max_position_embeddings=262144`, `mtp_num_hidden_layers=1` + the `rvn_mtp_graft` stamp, `exclude_modules` contains `*ple*`, `ple_embedding_dtype=nvfp4` |
| `SGLANG_EMBEDDED_MODEL_OVERRIDES` | may point at a baked LIL text_config | `-e SGLANG_EMBEDDED_MODEL_OVERRIDES=` (empty ⇒ no-op per patch 0034) | never inherit the LIL override file |
| PLE lookup | `SGLANG_PLE_PACKED_NVFP4=1` + `SGLANG_PLE_PACKED_FP8_REFERENCE=1` + `--ple-offload-embedding` + `/nix/store/…packed_ple.py` host overlay | `SGLANG_PLE_PACKED_NVFP4=1` + `--ple-offload-embedding`, **no** FP8-reference env, **no** code overlay | manifest-backed packed NVFP4 host table (patch 0049), reconstructed `bf16_direct` at load; the legacy LIL FP8-roundtrip reference stays unset |
| PLE host table | pinned private tables per process | pinned private table, **26.8222 GiB** | server-measured (see *Measured numbers*); no shared/file-backed path exists in this image (patch 0032 absent from `patches/series.production`, `SGLANG_PLE_SHARED_DIR` unreferenced) |
| Private draft head | `SGLANG_PRIVATE_DRAFT_NVFP4_A16=1` | same, set | the draft head is the grafted MTP layer |
| Speculation | `NEXTN` 3/1/4 + draft quantization + draft MoE backend | `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-draft-model-quantization modelopt_mixed --speculative-moe-runner-backend marlin` | legal only because the served dir is a **stamped graft**: 0057 still refuses NEXTN for an unstamped RVN text dir, 0058 admits the graft in the text contract, 0059 remaps the draft to `Qwen4ExpForCausalLMMTP` and defaults the draft path to the target dir (hence no `--speculative-draft-model-path`) |
| `--gdn-mtp-cache-mode` | `none` | `none` | default is `full`; `none` skips intermediate h-state caching during MTP verify and reconstructs `h_K`, trading post-verify compute for the `intermediate_ssm` buffer (server_args.py help). A NEXTN-verify memory choice, **not** an MTP on/off switch |
| YaRN | YaRN factor 4.0 → 1M window | **omitted**, native 262144 | no extrapolation |
| `--context-length` | `262144` | `32768` | recipe value; 262144 is the checkpoint-native maximum, so raising `MAX_MODEL_LEN` needs no rope override |
| HiCache | `--enable-hierarchical-cache` + all `--hicache-*` + **shared** `/data/hicache/qwen38-flash-next-kanadaj-a6d5284` mount + `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` | **all omitted, no mount** | that hicache dir is shared by both replicas, so mounting it would cross the isolation line |
| Vision / multimodal | `--image-processor-backend=torchvision`, `--mm-feature-transport=cpu`, `--limit-mm-data-per-request`, `--media-url-max-file-size-mb=8`, `SGLANG_IMAGE_MAX_PIXELS` | **all removed** | inherited LIL defaults; the checkpoint is `Qwen4ExpForCausalLM` / `qwen4_exp_text` with no `vision_config`, and no code path requires them for a text arch |
| `--default-chat-template-kwargs` | `{"reasoning_effort":"xhigh"}` | **not passed** | LIL serving default, not a template byte |
| `--reasoning-parser` / `--tool-call-parser` | `auto` / `auto` | kept at `auto` | template-driven auto-detection; dropping them changes what the chat API returns as content vs reasoning — a behavior regression unrelated to the RVN experiment |
| `--enable-metrics`, `--uvicorn-access-log-exclude-prefixes`, `--startup-weight-load-mode=serial` | not in the live CMD (metrics only in the stale `deploy/production/args.json`) | **not passed** | not in the verified recipe (the pre-refresh profile passed them) |
| `--disable-custom-all-reduce` | **not in the live CMD** (only in the stale `deploy/production/args.json`) | not passed | matches live TP1 workers |
| Mamba / linear-attn | `flashinfer` prefill+decode, cache 64, `extra_buffer`, track 128, `--mamba-ssm-dtype=bfloat16` | same values | backends are architecture-required; `bfloat16` **overrides the checkpoint's declared `"mamba_ssm_dtype": "float32"`** — numerics-affecting, memory-motivated, matches production |
| Model loading | `--model-loader-extra-config '{"enable_multithread_load":false,"num_threads":2}'` | same, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | recipe load-peak pairing; 0052 releases the loader-format MoE storage during the Marlin repack. No `--startup-weight-load-mode` |
| Host caches | `/var/cache/sglang/qwen38-flash-next-kanadaj-a6d5284-tp1-gpu0[-b]` | `/var/cache/sglang/rvn-ple-nvfp4-0` → `/root/.cache` | recipe value (`-0` is a worker slot, not the GPU index); no `TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR`/`TILELANG_CACHE_DIR` are passed — `$HOME=/root`, so the default locations already land on that bind mount |
| HF cache mount | `/home/common/.cache/huggingface` | not mounted | model is local |
| Container name | `lilith-vllm` / `lilith-vllm-b` | `rvn-ple-nvfp4` | recipe name; a second concurrent copy collides and docker refuses it, which is a guard, not a bug |
| Docker mechanics | `--gpus all` via CDI (`nvidia.com/gpu=all`) + `CUDA_VISIBLE_DEVICES`, `--ipc=host`, `--net=host`, `--shm-size=34359738368`, ulimits `memlock=-1`/`nofile=1048576`/`stack=67108864`, `/models` ro | `--device nvidia.com/gpu=$GPU` (CDI), `--ipc host`, `--network host`, `--shm-size 32g`, ulimits `memlock=-1`/`stack=67108864`, `/models` ro, `-w /sgl-workspace/sglang` | **the GPU pin deliberately diverges from production**: docs/rvn-w4a16.md records that `--gpus all` + `CUDA_VISIBLE_DEVICES` does **not** isolate on this host and `--gpus '"device=N"'` is broken outright, so the recipe pins one CDI device and injects no `CUDA_*VISIBLE_DEVICES` (inside the container the GPU is always `cuda:0`) |

## Memory bound (read before a real run)

- The packed NVFP4 PLE host table (`SGLANG_PLE_PACKED_NVFP4=1` + `--ple-offload-embedding`) is built with `torch.empty(..., pin_memory=True)` in `runtime/python/sglang/srt/models/qwen4_exp.py` and pins **26.8222 GiB of page-locked host RAM per process** — the figure the server itself prints for this checkpoint (see *Measured numbers*), i.e. 9/16 of the 47.6839 GiB a byte-per-element 320001536×160 table needs (`patches/0057` note).
- There is **no file-backed or shared PLE path in this image**: `patches/0032-shared-ple-host-table.patch` is not in `patches/series.production`, and `SGLANG_PLE_SHARED_DIR` has zero references in the runtime tree (it exists only inside `PackedPLEStorage`). Production's own `SGLANG_PLE_SHARED_DIR` value is vestigial for the same reason. So the table is always private and pinned, `--ple-offload` cannot shrink it, and `--ulimit memlock=-1` (as in production) is what makes the pinning possible.
- `--ple-offload` therefore sets only the `MemAvailable` **floor** (default **60 GiB** — the 26.8222 GiB pin plus headroom for the loader peak and page-lock slack) that the launcher enforces before starting anything; below it the run aborts with nothing created.
- **HISTORY — the ~95 GiB figure is superseded.** Until this refresh the profile ran the ORIGINAL BF16 lookup (same table, 2 bytes/element, no `SGLANG_PLE_PACKED_*` env), which pinned **~95 GiB** per process; that is where the old 100 GiB floor and the "not launchable while both production workers run" note came from. Keep the floor guard, do not restore the old number unless the packed env is dropped.
- Measured host headroom (lilith, 131733972 kB total): `MemAvailable` was **34 GiB on 2026-09-24** with both production workers up, and is **18 GiB on 2026-09-25** with both production workers *and* this packed server running (`awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo`). A 60 GiB floor therefore still aborts on this host while the prod pair runs — by design, the guard is a floor, not an estimate of what the server needs. Raising `--ple-offload=<gib>` deliberately is the operator's call; nothing starts without host RAM the pin can actually lock.

## Measured numbers (lilith, 2026-09-25, this recipe on one 96 GB card)

- **Decode speed:** graphs off **11.7 tok/s** → graphs on (`--cuda-graph-max-bs-decode=8 --disable-prefill-cuda-graph`) **103 tok/s** → **NEXTN 175-197 tok/s**, accept length **2.5-3.7** (rate 0.5-0.9). NEXTN verification is lossless, so draft quality moves speed, never output.
- **Load end (server log):** target `Qwen4ExpForCausalLM` **mem usage=73.36 GB** (235.92 s), draft `Qwen4ExpForCausalLMMTP` **mem usage=2.95 GB** (40.06 s); then memory pools and graph capture leave ~9 GB free (`max_total_num_tokens=369408`, `available_gpu_mem=9.06 GB`).
- **PLE:** `[rvn-ple] PLE table model.layers.1.ple.ple_embedding loaded from ple_storage.json: 320001536 rows x 160 cols, **26.8222 GiB pinned host bytes**, reconstruction=bf16_direct`.
- **Graph capture (NEXTN):** target verify 0.14 GB / 2.96 s, draft decode 0.40 GB / 146.10 s (FlashInfer autotune cold), draft extend 0.26 GB / 0.84 s.
- Graft provenance: `tools/rvn_ple/graft_mtp.py` produced the served dir, `tools/rvn_ple/verify.py --graft` → **7/7**, and the sealed 9/9 + 7/7 reports live in `provenance/reports/`.

## Risks and limits of the dry-run gate

- `--dry-run` proves only the command text; it cannot prove the server starts, and — by design, it promises zero side effects — it enforces neither gate, so a wrong image still prints a plausible command. Check the image separately with the one-liner in *Image and profile gate*, or read the `profile gate OK:` line the real run prints before starting.
- Two known live-start risks survive from the pre-refresh profile: `--moe-runner-backend marlin` on SM120 is non-default (`--fp4-gemm-backend` help says auto picks `marlin` on SM80-SM90 and `flashinfer_cutlass` on SM120) — the recipe uses it anyway and the serving log shows it working; and the pinned PLE table plus the CUDA-graph capture need real host and GPU headroom, which the floor guard and `--mem-fraction-static 0.90` are there to protect.
- NEXTN is gated on the graft stamp. If `--model-path` is ever pointed at a directory without `rvn_mtp_graft` (`encoder_version: rvn-mtp-graft-r1`, `count: 1`) — e.g. a fresh `convert.py` output — patch 0057 refuses the launch. Re-deriving a graft after the retirement of the old candidate/source dirs needs `convert.py` against a fresh source copy first, then `graft_mtp.py`, then `verify.py --graft`.
- The served template is the checkpoint's own, so chat-format behavior differs from the production workers' unsloth template (system-message merging, `reasoning_effort` mapping). That is intended here — it is the template the 175-197 tok/s numbers were measured with — but it makes the two servers non-interchangeable as chat endpoints.

## Coexistence checklist

- [ ] GPU index **free** (`nvidia-smi`): GPU0 = `lilith-vllm`, GPU1 = `lilith-vllm-b` **and** currently the verified `rvn-serve-nextn` server on GPU 1 — a second copy on GPU 1 shares that card, and one under the same name is refused by docker.
- [ ] `free -g` / `MemAvailable` shows ≥ 60 GiB spare — enforced by the launcher, not skippable.
- [ ] Port `8111` free (`ss -ltnp | grep 8111`); prod keeps 8101/8102.
- [ ] Caches baseline-only: `/var/cache/sglang/rvn-ple-nvfp4-0`; prod's `…-kanadaj-a6d5284-tp1-gpu0[-b]` and the shared `/data/hicache/...` untouched.
- [ ] Printed command contains no `/data/hicache` mount, no `SGLANG_PLE_PACKED_FP8_REFERENCE`, no `SGLANG_PLE_SHARED_DIR`, no `packed_ple.py` overlay, no vision flags, no `--json-model-override-args`, and exactly the five `-e` keys.
- [ ] `docker ps` afterwards shows the two prod workers plus only `rvn-ple-nvfp4`.
- [ ] Dry-run output reviewed before the real run (and diffed against `docs/rvn-w4a16.md` if you touched a flag).
- [ ] Image gate passed: stderr shows `profile gate OK: {…}` for the image you are about to run (default `rvn-w4a16:sim`), printed before any container exists.

## Rollback

- `docker rm -f rvn-ple-nvfp4` — the whole rollback; the container is `--rm` with no persistent state and never owned anything else.
- Optional: `rm -rf /var/cache/sglang/rvn-ple-nvfp4-0`.
- Production workers, their caches and the shared HiCache store are never modified, stopped or restarted by anything in this directory.

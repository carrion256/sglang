# Qwen TP2 packed-PLE vision on SM120

**Unpublished CPU candidate:** [Responses compatibility and Qwen phase/order](docs/responses-compat.md)
adds separately attested boundary and streaming/nonstream ordering patches after the
effort-alias profile, including its `minimal` → `low` alias. The ordered nonstream
path constructs typed output directly and retains usage details and requested
logprobs; structural splitting is limited to recognized Qwen markers and the loaded
`qwen3_8_flash_next` / `_text` model types. The cumulative
[Qwen Flash-Next multimodal alias profile](docs/qwen-multimodal-alias.md) additionally
restores four existing Qwen VL processor paths under the release model type. The
[invalid generated-token failure profile](docs/invalid-token-failure.md) completes
the cumulative `0015` → `0016` → `0017` → `0018` → `0019` stack: scheduler
faults retain `InvalidTokenError` through Chat and Completions SSE, Responses
failures remain retrievable but cannot be replayed as `previous_response_id`, and
graceful cancellations are unchanged.
Historical production profiles below are unchanged; no deployment is implied.

Publishable source and deployment package for the locally accepted Qwen3.8
Flash-Next LIL NVFP4 stack. **No model weights, container archives, credentials,
private prompts, routing snapshots or failed optimization experiments.**

## Current production (2026-09-13)

**[Chat reasoning-effort fix: contract, image, tests and rollout](docs/chat-reasoning-effort.md)**.
Explicit Chat effort now wins over the server default; omitted effort remains
`medium`. Both TP2 replicas run the digest-pinned
`production-chat-effort-20260913-v2` image. Actual returned prompt-token IDs prove
requested `xhigh` renders xhigh, not medium. Serving flags and route policies are
unchanged. **[Full external Docker command](docs/chat-effort-command.sh)**.

The new `production-chat-effort` source profile is the production profile below
plus patch0014 (two files). Historical `production` and `combined` profiles remain
unchanged; neither silently includes the new Chat fix.

## Previous production source package (2026-09-11)

**[Exact production source, build, launch, validation and rollback](docs/production-20260911.md)**:
private NVFP4 W4A16 draft head, NEXTN3/topk1/draft4, Mamba512/track128.
`Dockerfile.production` reconstructs all **4,391** live SGLang source files.
The default `Dockerfile` additionally preserves published HF-path fix0012,
which the actual production image does not yet contain. Both profiles include
all earlier accepted vision/PLE/runtime fixes applicable to their ancestry.
**[Published production-source runtime: digest-pinned pull/run and audit](docs/production-image-20260911.md)**.
The image is a clean rebuild, not a push of the private live snapshot. No production restart was performed.

## Historical published no-repo Docker quickstart

**[Copyable pull/run, health, text and image requests](docs/standalone.md)** —
ordinary patched SGLang, not a defaults wrapper. The corrected HF-path vision
runtime is invoked with `--entrypoint python3 IMAGE -m sglang.launch_server`;
**all serving flags, packed-PLE opt-ins and the complete YaRN2 JSON remain outside
the container, visible and editable in the Docker command**. No Git clone,
external script or JSON file is required. Actual-image GPU-free CLI/parser tests
are verified; a new standalone GPU boot/download was not performed.

## What is included

- An ordered **13-patch combined series covering 36 changed paths** (production:
  12 patches / 35 changed paths), not only the two
  latest vision files. Complete changed runtime sources and clean preimages
  are included for offline audit and reconstruction.
- SM120 FP8 KV / GDN / online-FP8 foundation; LIL config aliases, packed loader
  corrections, mRoPE dispatch and language-only config behavior; mixed ModelOpt
  MTP loading; packed host PLE with global shard validation and TP2 reduction;
  visual FC1 MXFP8 output padding and visual FC2 Marlin logical-order bias.
- Complete external YaRN2 override, optional host-side launch helper, TP2 arguments, dedicated
  GPUStack backend-version template, synthetic CPU tests and image_url example.
- Hash inventory of **all 4,389 non-bytecode files in the accepted SGLang package**,
  extended to **4,390** with the vision adapter and **4,391** with the private
  draft head. There are no unexplained missing
  package paths after clean reconstruction. See `provenance/` for scope limits.

This is a deployment-specific derivative, not an upstream SGLang release.
See [NOTICE](NOTICE) for authorship, Apache-2.0 and mratsim attribution.

## Pinned identities and coverage

Foundation: [mratsim/sglang-qwen38fn-sm120-turbo](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo)
commit `b3a0fbbb859408a82171aec4f61eb1e8c5786d93`.
Underlying SGLang Git commit recovered from the preserved image:
`d91c3682b0b429e4c70df63cd57f819588ce29b0`.
**That Git commit alone is not the day-0 model implementation**: the official
image also has a 71-file day-0 overlay. The pinned official image below preserves
that overlay and its native/dependency ABI; do not build from bare Git HEAD and
assume equivalence.

`provenance/source-identity.json` records image identities. `patches/series`
defines order; `provenance/patches.json` records patch hashes and every affected
path. `runtime/` is an auditable changed-file snapshot, not a full importable
SGLang checkout. `upstream-preimages/` contains only the files needed to test
patch application offline. The verifier hashes complete installed files, not
just selected function signatures.

Patches 0001–0008 were recovered from the image's actual build layer and matched
against the previous local repository. 0004 includes local edits and 0008 was
untracked in that previous repository. 0009 captures remaining mRoPE/config/MTP
copies. The pinned day-0 image already supplies the matching generated
`.clang-format`; its preimage is retained so clean reconstruction tests reject
any drift without trying to create it again. 0010 captures accepted packed PLE
directly from the attested installed tree. 0011 is the saved vision build's exact two-file
source overlay. 0012 propagates the loader-resolved checkpoint identity to PLE
pre-reads (including HF repo IDs); its new evidence is separate from the original
vision attestation. `runtime-files.vision-v2.json` preserves the original inventory.
No failed MoE, QSA scratch/selector/metadata, PCIe all-reduce or
MoE finalize workaround was imported. Existing upstream kernels are retained.

## Build or use the immutable image

Current corrected runtime ([HF-path fix evidence](docs/ple-hf-path-fix.md)):

```text
kanadaj/sglang-qwen38fn-sm120-turbo:r22-tp2-vision-ple-hfpath-20260908-v1@sha256:4c0d09bcf0cb5906e5abe74edf00643d612c7d8f7306ca6e76d38be2d486da58
```

Two build paths (run from this repository):

```bash
# Reconstruct all active source deltas on the pinned official day-0 ABI.
docker build -t qwen-tp2-vision:source .
# Preserve the immutable vision ABI; apply only the HF-path fix (no launcher).
docker build -f Dockerfile.digest -t qwen-tp2-vision:digest .
```

The source Dockerfile starts from
`lmsysorg/sglang:qwen38flashnext@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae`
and checks preimages, ordered patches and the full final SGLang package. It
reconstructs source atop the pinned native environment; it is **not** a new
from-scratch CUDA/native dependency build. Dependency name/version inventory is
provided but is not a complete reproducible native build lockfile.

**Original source-package verification (historical):** clean patch application, full reconstructed-source hashes,
CPU tests (37 runtime/packaging + 2 launcher, also repeated from a clean staged
export), syntax, static secret scan and read-only registry-layer verification.
All 21,969 saved source/dependency-export files (1,919,615,048 bytes) were hash-checked
with zero missing/mismatched files. **Not performed here:** a fresh Docker
image build/full pull, GPU boot, GPU tests, remote publication or production access.
Neither rebuild promises a byte-identical OCI digest; use the published digest
when you need the exact runtime. Read-only Docker Hub verification established
the exact 68-layer day-0 ancestor and all 85 accepted-parent layer digests in the
vision image. Its three suffix layers contain only the two recorded non-bytecode
source files plus compilation bytecode; downloaded source bytes match exactly.
Two layer media types changed from Docker-v2 to OCI without changing content
digests. Final-image attestation is thus transitive from a fully hash-checked
accepted export plus independently downloaded final overlay, not a fresh running
container export. See `provenance/registry-chain-verification.json`. The digest
Dockerfile fails closed if the image source differs from the recorded set.

## Run: two available SM120 GPUs

Validated configuration was a pair of RTX PRO 6000 Blackwell GPUs, TP2,
`modelopt_mixed`, FP8 E4M3 KV, packed CPU PLE with FP8-reference rounding,
NEXTN 2 steps / 3 draft tokens, C16, 524,288 per-request context, YaRN factor 2,
static memory fraction 0.93, page size 64, chunked prefill 4096 and decode graph
max batch size 16. Prefill graphs and custom all-reduce are disabled. The
vision encoder is replicated via `--mm-enable-dp-encoder`; do **not** add
`--language-model-only`. Exact settings are in `deploy/args.json` and
`deploy/environment.json`, derived from the saved working vision definition.

Use matching LIL checkpoint files, including vision processor/tokenizer files,
and enough host RAM for loading plus packed tables. Accepted packed PLE tables
occupied about 26.82 GiB across two ranks; this does not bound peak RSS.
The pending-pair cap does not bound whole-file loader dictionaries or mmap
paging. The local-checkpoint scripts below do not download weights; the
[no-repo quickstart](docs/standalone.md) instead downloads them on first start.

```bash
# Print JSON argv without starting anything. Choose an unoccupied GPU pair.
python3 deploy/run.py --model-dir /path/to/checkpoint --gpus 0,1
# Explicit opt-in to start; API binds localhost by default.
python3 deploy/run.py --model-dir /path/to/checkpoint --gpus 0,1 --execute
```

The wrapper passes all saved flags and the complete override directly, so it
works with the immutable image even if its entrypoint is bypassed by an
orchestrator. It is an equivalent standalone wrapper, not a freshly boot-tested
launch. It uses host IPC and a read-only checkpoint mount; review these choices
and avoid occupied production GPUs/ports. Persistent compilation cache mounts
are optional local policy, not a source dependency. YaRN2 is extrapolation,
not a model-author guarantee of 524K multimodal accuracy.

## OpenAI `image_url` example

```bash
# Creates a synthetic red-square PNG in memory; prints JSON only.
python3 examples/image_request.py
# Opt-in request; no user LAN addresses or deployment IDs.
OPENAI_BASE_URL=http://localhost:30000/v1 MODEL_NAME=qwen-vision \
  python3 examples/image_request.py --send
```

`OPENAI_API_KEY` is optional and read only from the environment. The request
uses `messages[].content` with `type: image_url` and a PNG data URL. A 200 response
or nonzero image tokens alone is not accuracy validation.

## GPUStack deployment

`deploy/gpustack-template.json` is a **documentation template**, not a POST/PUT
payload. Create a **dedicated SGLang backend version** whose image is the exact
vision digest, rather than changing a shared backend or relying on per-model
`image_name`. Supply checkpoint storage, unused GPU selectors and arguments
from the deploy files through your installation's supported UI/API; add the
complete JSON override as `--json-model-override-args`. Start with one replica,
two GPUs; verify actual runtime digest, flags, health and image-dependent answers.

**GPUStack 2.2.3 hazard:** PUT of a non-Custom model with existing `image_name`
cleared `image_name` and `run_command`, even when supplied unchanged. A blind
scale/selector update can launch the old text image. A dedicated version avoids
reliance on that disappearing override. Read back the exact backend/model and
actual container digest after every change. This repository performs no API
writes and contains no routes, IDs or credential-bearing snapshots. Scaling,
draining and cutover remain separate operator-owned work; no current rollout
state is claimed here.

## Evidence and known limits

Saved same-pair qualification initially concluded **INCONCLUSIVE / NOT QUALIFIED**
for performance equivalence. The user subsequently accepted the vision/OCR and
latency caveats; that acceptance does not change the measurements:

| Check | Original | Vision |
|---|---:|---:|
| Short text normalized accuracy | 30/32 | 30/32, no flips |
| Estonia-long (repeats of one task) | 16/16 | 16/16 |
| Held-out synthetic vision | n/a | 35/36 |
| Warmed C1 aggregate tok/s | 192.92 | 195.18 |
| Warmed C4 aggregate tok/s | 615.13 | 624.78 |
| Warmed C16 aggregate tok/s | 1432.19 | 1436.04 |
| Cold 128K TTFT | 10.475 s | 10.676 s |
| Long-profile warm TTFT | 0.468 s | 0.897 s |

Vision categories: OCR 12/12, shape/count/color 12/12, layout 6/6,
ordered multi-image 5/6. The miss inserted a digit in reversed two-image OCR
(`VEXA 493` → `VEXA 4193`). It is not a 36/36 result. Warmed throughput used three
60-second repeats per concurrency; C1/C4 uncertainty allows regression and
sequential blocks leave time/order confounding. Cold TTFT increased 1.92%
(95% CI +1.64 to +2.20%); warm long-profile TTFT increased 91.53%
(95% CI +34.32 to +148.77%). No equivalence or speed-gain claim.

Actual vision benchmark KV allocation was **6,157,248 tokens**; same-pair text
control was 6,258,688. Restored vision later allocated 6,152,768. Historical
text-only 6,657,792 on another deployment is not an isolated encoder-cost
comparison. Neither text-only capacity results nor two mixed C16 smoke waves
prove full 16×256K multimodal residency. No fresh KLD or full-vision-capacity
qualification was performed for this package.

## CPU verification

Python 3.11 was exercised with CPU Torch 2.14.0, Triton 3.7.0 and safetensors 0.8.0.
Use an isolated environment (Linux x86-64 for Triton):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python triton==3.7.0 safetensors==0.8.0
PYTHON=.venv/bin/python bash scripts/test.sh
# Stdlib-only offline clean apply/hash/syntax check:
python3 scripts/verify_source.py
# Full installed image audit (inside an image or exported source root):
python3 scripts/verify_source.py --tree /sgl-workspace/sglang --complete
# Stage first so the scanner includes all intended publication files and diff:
git add -A
python3 scripts/secret_scan.py
# Optional read-only public registry re-verification (small manifests/layers only):
python3 scripts/verify_registry.py > registry-audit.json
```

CPU tests execute actual packed gather kernels in Triton interpreter mode,
actual loader/reduction methods with CPU seams, and actual vision adapter
classes with explicit parent substitutes. Fixtures are synthetic. They are not
NCCL, CUDA stream, CUDA-graph, CUTLASS or Marlin numerical certification.

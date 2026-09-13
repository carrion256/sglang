# Production source and configuration — 2026-09-11

## Which build is actually running?

Two TP2 replicas use **NEXTN steps3 / topk1 / draft tokens4**, private NVFP4 W4A16
draft head enabled with `SGLANG_PRIVATE_DRAFT_NVFP4_A16=1`, packed CPU NVFP4 PLE,
Mamba cache512 / tracking128, context524288, static memory0.93 and request cap16.
The target/verifier head and shared embeddings remain BF16. The private head is
constructed before graph capture, with independent module registries and packed
parameters; tied embeddings preserve the upstream sharing path. The gate defaults
off and accepts only `0` or `1` on the untied-head path.

Actual live image identity:
`sha256:69f1f64c62ca2b5d919d69bcd60efb7fc89d7bf410ba406a5de586051f358465`.
Docker's containerd-backed image `Descriptor` identifies this as an **OCI image
index**, not a config digest. It is locally resolvable; the private snapshot
itself was **not** pushed. A [separate clean production-source runtime is now
published](production-image-20260911.md), with its own registry index, platform
manifest and config digests. The durable GPUStack backend is
`qwen-private-draft-head-20260911-v1-custom`.

**Important discovery:** production still uses the earlier local-checkpoint
loader. It does **not** contain published patch0012 (the HF-repo-ID PLE loader
fix). Do not overwrite that accepted published fix just to match production:

| Recipe | Ordered patches | Meaning |
|---|---|---|
| `Dockerfile.production` | 0001–0011, 0013 | Exact currently deployed SGLang source; use a mounted local checkpoint |
| `Dockerfile` | 0001–0013 | Preserves HF-repo-ID fix plus private-head implementation; combined source verified, **not deployed or GPU-qualified as this combination** |
| `Dockerfile.digest`, old `docs/standalone.md`, old `deploy/*.json` | Historical published runtime/profile | Preserved historical reproducibility, **not current production** |

No prior fix was deleted. Production reconstruction differs from the combined
source in exactly `srt/model_loader/loader.py` and `srt/models/qwen4_exp.py`.
Patch0013 and the complete new `private_draft_head.py` / modified
`qwen4_exp_mtp.py` are tracked, not references to a maintainer's local files.
NEXTN3 is an external configuration change; no speculative mixed-chunk changes
are included.

## Reproduce source and build (on a separate build machine)

```bash
python3 scripts/verify_source.py --profile production
python3 scripts/verify_source.py                       # combined variant
# Expensive: do NOT run this on a busy production worker.
docker build --pull --no-cache -f Dockerfile.production -t qwen-runtime:production-20260911 .
```

Both recipes pin official day-0 image
`lmsysorg/sglang:qwen38flashnext@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae`.
Its source is SGLang commit `d91c3682b0b429e4c70df63cd57f819588ce29b0`
**plus the official 71-file overlay**. A bare upstream checkout is not equivalent.
Original foundation commit, all prior patch hashes, attribution and dependency
inventory remain in `provenance/`, `NOTICE` and `licenses/`.

Build guards check base preimages, ordered patch hashes and the complete final
source inventory. The historical fresh-base `.clang-format` collision fix is
preserved. This is a source/ABI-pinned reproducible recipe, **not a promise of a
byte-identical OCI image** (build timestamps, compiled caches and historical
private snapshot layers differ). Never publish old private container snapshots
or export their Config/Env: this recipe rebuilds from the public day-0 image,
without private container history, credentials or serving defaults.

## External launch configuration

[`production-command.sh`](production-command.sh) is the full explicit copyable
Docker invocation, not a custom launcher. It includes the entire nested YaRN2
JSON, all argument tokens and the five allowlisted environment settings.
[`../deploy/production/args.json`](../deploy/production/args.json) and
[`environment.json`](../deploy/production/environment.json) are its machine-readable
configuration. Replace only the host checkpoint mount, disjoint GPU pair and
published host port as needed. It binds the external endpoint to localhost by
default; expose it only through an authenticated gateway. Do not run it against
occupied production GPUs/ports. The exact-production profile requires an
already downloaded local checkpoint, not a Hugging Face repository ID.

The standalone wrapper normalizes the model mount to `/model`, host to `0.0.0.0`
and engine port to30000. Other argument tokens, including duplicate metrics
flags, are preserved from both live replicas. Docker orchestration/network/IPC
wrappers are equivalent standalone choices, not a verbatim private GPUStack
container configuration. GPU placement was 0/1 and2/3 at the accepted rollout;
instance IDs, host addresses and ports are deliberately not published here.

## Source attestation and current verification

Read-only `docker exec` Python hashing inventoried **4,391 non-bytecode regular
files under `python/sglang` on each of the two running replicas**. Their complete
path/hash inventories matched. Four files differed from the previous published
package: the two private-head implementation files and the two files changed by
published HF-path patch0012 (absent in production). Only selected runtime source
files were exported, and every exported byte sequence was hash-checked.

The complete retained official day-0 source export was copied into two clean
local reconstruction directories. Its historical missing `.clang-format` was
supplied from the tracked, previously fresh-image-verified base preimage (not
removed to bypass a collision). All ordered production patches applied and
**all4,391 paths and hashes matched the live inventory**, with no missing/extra
files. A separate clean combined reconstruction also verified all4,391 files
against its manifest. The local source export had two non-executable mode
warnings for old upstream executable files; these did not affect content hashes.
This attestation covers contents/path sets, not a complete filesystem-mode audit.

`provenance/production/runtime-files.json` is the full live inventory;
`deployed.json` records scope and normalized configuration.
`image-identity.json` records only safe image descriptors and RootFS hashes:
all88 vision-parent layer diffIDs remain a prefix of the92-layer deployed image.
It does not expose private image Config or history commands.
Actual two-replica output is saved in `provenance/production/live-readback.json`.

Verification in the earlier source-packaging task (the subsequent image build,
publication and CPU checks are recorded in [the image release](production-image-20260911.md)):
- Existing baseline:39 CPU/source tests passed.
- Updated suite:46 CPU/source tests passed; both offline profiles clean-apply.
- New checks exercise actual helper ownership/default/invalid-gate dispatch,
  the actual AST-isolated MTP install method (including tied fallback), both
  profile manifests and exact shell command/env token parity.
- No GPU launch, deployment restart, driver/network/model changes or full Docker
  rebuild were performed. No new full-model quality or performance measurements.
- Source hashes exclude `.pyc`/`__pycache__`; native dependencies, weights,
  generated caches and model templates are outside this live source hash audit.
  The pinned base and historical registry/source attestations remain their
  provenance, not a newly verified complete environment export.

## Accepted validation history (not rerun for Git publication)

The private-head same-pair opening/candidate/closing trial passed48/48 bounded
text/colour-vision checks per arm. Candidate C1 median throughput improved about
5.1%, but raw C1 ranges overlapped through acceptance drift; C16 normalized
compute improvement was not robust to the closing control. Broader qualification
reported HumanEval+156→157/164, LiveCodeBench72→72/100 and actual Estonia-long32/32
both. There were19 capped LiveCodeBench outputs in each arm and exchanged task
wins/losses; the user accepted these disclosed tradeoffs. Do not call this
statistical equivalence, uncensored pass@1 or full-vocabulary KLD qualification.

Subsequent same-pair private-head NEXTN2→3 trial reported C4+12.59%, C1+0.93%,
C16+2.46% and bounded Estonia8/8 each. These are historical trial results, not
fleet benchmarks. The authorized rolling promotion then verified direct and
routed text/red-blue vision, private/target separation on both TP ranks, width4
verifier graph capture, draft decode/extend captures and graph replay, while
retaining a healthy peer. Both final native monitoring targets were up. One
watcher sample saw startup503 and skipped its routed probe: a measurement gap,
not proof of perfect request-level continuity.

Earlier vision caveats remain:35/36 including an ordered multi-image OCR miss,
text30/32 both, Estonia16/16 both, and a measurable latency increase. Source
reconstruction does not resolve those accuracy limits. Automatic KV capacity is
allocation evidence, not full-capacity multimodal residency qualification.

## Rollback (requires separate deployment authorization)

1. Preserve the locally resolvable current image and protected private deployment
   record before retiring any container. Verify the image is actually available
   in the correct worker daemon; do not rely on a digest written in a document.
2. To roll back only NEXTN3, retain this image and gate1; change external steps3→2
   and draft tokens4→3, topk remains1, with all other arguments/env unchanged.
   This is the previously deployed private-head NEXTN2 configuration.
3. To roll back the private head itself, prefer the retained original vision
   image `sha256:99fef9b4927e7e7c0dbd185a6bfe55995cea78e0d5a3c53e4409afb91109dc52`
   with gate absent/off and previous2/1/3 profile. Disabling the gate on the
   candidate is a source-defined fallback, not a newly qualified rollback image.
4. In GPUStack, preserve the full canonical backend/config and complete route
   target sets. Updating canonical parameters alone did not restart engines in
   the inspected version. Use authorized maintenance, drain and selected-instance
   replacement sequentially, never both peers at once. Do not copy a private API
   snapshot into this repository or silently clear a shared image override.
5. Require replacement health, text/known-answer vision, intended speculation,
   TP placement and graph/private-head logs before draining its peer. Reconcile
   separately labeled monitoring targets to new instances and verify both up.
   Preserve fallback routes; do not restore retired experiments automatically.

No rollback commands were executed as part of this source-publication task.

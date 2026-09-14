# Qwen HiCache state transfer — experimental draft

## Status

**Do not merge or deploy this profile.** It preserves three incomplete HiCache
patches and the tests used to investigate them. Narrow state-transfer tests and
one exact live restoration fixture passed, but a later end-to-end generation
gate passed only 32 of 48 cases with repeated punctuation.

`Dockerfile.hicache-wip` and `patches/series.hicache-wip` are isolated from every
default and production profile. No launcher enables HiCache, no deployment
configuration changes, and the ordinary `Dockerfile` does not install this work.

## Defects addressed by the patch series

The Qwen Flash-Next runtime has state outside the ordinary full-attention KV and
Mamba recurrent buffers. Restoring only the existing host-pool components can
therefore reuse a prefix with incomplete model state.

1. `0020-hicache-ple-state.patch` adds the PLE short-convolution and N-gram
   slot tensors to Mamba host checkpoints. It includes their bytes in host-pool
   sizing, carries them through RAM and flat page representations, and waits for
   the first relevant transfer event before an early PLE read.
2. `0021-hicache-file-integrity.patch` treats missing, truncated, or unreadable
   file pages as cache misses so a prefetch worker can continue. It permits the
   complete PLE checkpoint format only with the tested built-in file backend;
   other storage backends remain rejected.
3. `0022-hicache-qsa-sidecar.patch` adds a required page-aligned sidecar for
   compressed QSA index keys, including packed MTP draft layers. It budgets the
   index inside the KV share of the existing host limit, requires complete pages,
   and waits for the corresponding layer transfer before QSA reads the index.

The patch preimages match the 4,391-file production Chat-effort inventory and
the immutable base image:

```text
kanadaj/sglang-qwen38fn-sm120-turbo@sha256:872a2bda228e39aa9c1af729b47cc28f7862e7859e448f1a8868b85a4051f404
```

`provenance/hicache-wip.json` records every hash transition, the ordered patch
hashes, and the resulting 4,392-file inventory digest. The one new source file is
`python/sglang/srt/mem_cache/qsa_pool_host.py`.

## Retained evidence

These results were collected on the preserved candidate before this draft was
packaged:

| Gate | Result | What it establishes |
|---|---:|---|
| Packaged CPU PLE/file/QSA fixtures | 55 passed | Layout, sizing, lifecycle, file-error, sidecar, and wait behavior |
| PLE transfer cases | 24 per GPU | Real kernel copies for the companion slot state |
| Combined PLE/Mamba/target/draft checkpoint | 1 passed | Repeated asynchronous relocation and event-ring reuse |
| File reconstruction checkpoint | 1 passed | GPU→RAM→file→RAM→GPU with a reconstructed backend |
| QSA relocation | 4 per GPU | Target and draft indices, two layouts, RAM and reconstructed files |
| Exact live restore fixture | cold 4/4; storage 4/4; GPU replay 4/4 | Positive storage and H2D use after a flush |
| Ordinary 200-tool catalogue/replay | **32/48** | **Blocking end-to-end corruption remains** |

The PLE GPU result covers the kernel transfer backend. Six direct-copy cases
were deselected after the unmodified parent failed them with the same invalid
argument error; this profile makes no direct-backend qualification claim.

The live restore transferred 81,788,928 more bytes than the preceding build,
exactly 832 bytes per restored KV token. This matches the added QSA index state
and supports the omitted-state diagnosis for that fixture.

The later 32/48 failure recorded no new host or storage reads. That means the
failure cannot be attributed solely to corrupt data restored by these patches.
Instrumented diagnostics also observed NaNs in CUDA-graph draft extension, but
they did not establish whether that path caused the emitted punctuation. This
draft contains no draft-extension workaround.

## Verification and CPU fixtures

The standard package test remains unchanged. Verify the isolated patch hashes,
declared transitions, and result inventory:

```bash
python3 scripts/verify_hicache_wip.py
python3 -m unittest tests/test_hicache_wip_packaging.py -v
```

Build the experimental image only for investigation:

```bash
docker build --pull=false -f Dockerfile.hicache-wip \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t qwen-hicache:wip .
HICACHE_WIP_IMAGE=qwen-hicache:wip bash scripts/test_hicache_wip.sh
```

The runner uses no network or GPUs. It executes the 55 CPU cases from
`validation/hicache/` inside the candidate image. The three GPU files are retained
for review and require an explicitly isolated GPU environment; the runner does
not claim or acquire an available GPU.

To verify and apply the patch series to a complete export of the exact base
image:

```bash
python3 scripts/verify_hicache_wip.py --tree /path/to/sglang --apply
```

The verifier checks the complete base inventory before applying anything and
the complete result inventory afterward. The Docker build runs this mode, so a
clean application against the exact parent is required to produce an image.

## Remaining merge blockers

- Isolate the repeated-punctuation failure and determine whether HiCache,
  scheduler overlap, speculative draft graphs, or another inherited path is
  responsible.
- Pass repeated no-logprob catalogue/replay gates after positive RAM and file
  restoration, including eviction and slot reuse.
- Repeat real GPU transfer tests on both ranks, long-context retrieval, mixed
  media, concurrency, cancellation, restart persistence, and a soak on the final
  implementation.
- Review support for storage backends other than the built-in file backend. They
  are intentionally rejected when PLE companion state is present.

Until those blockers are closed, HiCache should remain disabled for this model.

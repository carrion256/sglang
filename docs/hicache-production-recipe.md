# Qwen Flash Next: TP2 write-back HiCache recipe

This is the deployed configuration on two RTX PRO 6000 Blackwell Workstation GPUs
(96 GiB each), connected through PCIe 4.0, with 256 GiB system RAM. It is a measured
configuration for this model and hardware, not a universal cache-sizing formula.
Use the repository's Qwen TP2/MTP3 model-loading recipe and the opt-in
`Dockerfile.hicache-wip` profile, now including patch0033 for separate
prefill/decode scheduling. Build from this revision before using the new flag;
earlier published images do not include it:

```sh
docker build -f Dockerfile.hicache-wip -t local/qwen-hicache-interleaving .
INTERLEAVING_IMAGE=local/qwen-hicache-interleaving INTERLEAVING_PROFILE=hicache \
bash scripts/test_prefill_decode_interleaving.sh
```

Select that built image in the launch command and use these serving/cache overrides:

```sh
--tp-size=2 \
--context-length=524288 \
--mem-fraction-static=0.92 \
--max-total-tokens=4342208 \
--chunked-prefill-size=6144 \
--prefill-batches-before-decode=0.5 \
--max-running-requests=64 \
--cuda-graph-max-bs-decode=64 \
--kv-cache-dtype=fp8_e4m3 \
--page-size=64 \
--enable-hierarchical-cache \
--hicache-size=45 \
--hicache-write-policy=write_back \
--hicache-io-backend=kernel \
--hicache-mem-layout=page_first \
--hicache-storage-backend=file \
--hicache-storage-prefetch-policy=timeout \
--hicache-storage-backend-extra-config='{"max_size":500000000000,"enable_metadata_cache":true}' \
--enable-cache-report \
--enable-metrics
```

Set `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` to a persistent, model-specific mounted
directory. Do not share a cache namespace between different checkpoints/quants.
These flags are overrides, not a complete model-loading command: retain the pinned
model configuration, YaRN factor 2 / original context 262144, vision setup, packed
PLE host offload, and MTP configuration from the Qwen recipe.

| Setting | Deployed value / interpretation |
|---|---|
| Checkpoint | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, QAD revision `629bc3218833a38b475b719f34aa571666f4a03e` |
| Quantization | `modelopt_mixed`; packed NVFP4 PLE in host RAM |
| Interleaving | `0.5`: one prefill chunk followed by two decode turns when both have work |
| Prefill chunk | 6,144 tokens; this is not a request-count batch size |
| Active concurrency ceiling | 64 requests; does not promise 64 full-length contexts |
| RAM HiCache | 45 decimal GB **per TP rank**, 90 GB aggregate (~83.82 GiB) |
| Disk payload cap | 500 decimal GB **per TP rank**, 1 TB aggregate (~931.32 GiB); filesystem overhead is additional |
| Observed host KV capacity | 3,812,160 logical tokens, shared logical capacity across ranks; do not multiply tokens by TP2 |
| GPU KV | Observed 4,342,208 logical tokens at this configuration; the argument is a cap, actual capacity depends on startup memory availability |
| Recurrent state | `max-mamba-cache-size=512`, `mamba-radix-cache-strategy=extra_buffer`, `mamba-track-interval=128`, `mamba-ssm-dtype=bfloat16` |
| Speculation | NEXTN, 3 steps, top-k 1, 4 draft tokens, `gdn-mtp-cache-mode=none`; retained |
| Graphs | Target/draft decode and draft-extension graphs retained; prefill CUDA graphs disabled as in the existing deployment |
| Storage wait | `timeout`; useful hits still depend on compatible recurrent/QSA checkpoints and available transfer slots |

Host memory also holds PLE weights, recurrent state, process allocations and filesystem
cache. 90 GB is not total service RAM. Observed available host memory around this
configuration was roughly 45–50 GiB, so it does **not** establish a guaranteed 50 GiB
reserve. There is no automatic low-RAM service shutdown in this recipe.

## Why write-back when RAM KV is smaller than GPU KV

A full cache should evict eligible older entries and keep accepting useful data.
The problem here is **eviction eligibility**, not fullness itself.

In this implementation, ordinary Full-KV RAM eviction visits `evictable_host_leaves`.
`_is_host_leaf()` excludes nodes whose GPU data is still present. Write-through
copies GPU-resident entries into RAM eagerly, while those copies remain protected
from that ordinary RAM eviction path. The special reclaim path for redundant RAM
copies is enabled only for write-back.

With 4.34M GPU tokens and 3.81M RAM tokens, write-through can therefore fill the
entire RAM KV pool with protected duplicates before the larger GPU pool needs to
evict them. New backups cannot obtain RAM slots, and disk restores also need RAM
slots as staging space. A 1 TB disk tier does not solve the unavailable staging
space. This is a normal filled-cache workload, not an exceptional condition, and
LRU ordering cannot help while its candidate set excludes the occupied entries.

One captured request illustrates the consequence: storage had a compatible
100,608-token prefix, but only 1,792 host tokens could be allocated after eviction.
The resulting partial restore lacked the required recurrent checkpoint, reused
zero tokens and admitted all 102,104 input tokens for prefill. The GPU's logged
usage was only 0.02. Disk lookup success alone did not provide a usable restore.

Write-back backs up entries when the GPU needs to evict them. RAM then retains
colder data beyond the GPU working set, and the implementation can reclaim settled,
unlocked duplicate Full-KV RAM copies after restores. It does not remove copies
in the middle of an asynchronous transfer. This is why this recipe uses write-back
instead of write-through.

There is a cost: GPU eviction can now require a GPU-to-host copy and completion
checks on the eviction path. Under inadequate host capacity or held transfer locks,
write-back can still fail to back up an entry and fall back to recomputation. It is
not a guarantee of perfect cache hits. No controlled throughput comparison is
claimed by this diagnostics/tests PR.

The issue is specific to this implementation's protection rules; it is not a claim
that every write-through cache must be larger than its upstream tier. RAM larger
than GPU helps this implementation only when sufficient space remains for cold
entries and concurrent restore staging.

Code references in the reconstructed runtime:

- `python/sglang/srt/mem_cache/unified_cache/components/full_component.py`:
  `drive_host_eviction()` selects host leaves.
- `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py`:
  `_is_host_leaf()`, `drive_host_eviction()`, `_can_reclaim_full_host_duplicate()`,
  `_reclaim_full_host_duplicates()` and `evict_device_leaf()` define eligibility,
  write-back-only duplicate reclamation and demotion.
- `python/sglang/srt/mem_cache/unified_radix_cache.py`:
  the `is_write_back` branch, `_execute_backup_kv()` and failed-backup handling.

## Validation and limits

See [restore-isolation-validation.md](restore-isolation-validation.md) for the
110 CPU / 20 GPU checks and their limits. The live shared-prefix/divergent-record
probe passed 48/48, but had no storage hits. A separate old-fixture replay answered
4/4 correctly with no observed disk hits and therefore **failed its required-storage
coverage gate**. These results must not be described as a fresh successful disk
restore test or proof of all full-cache scheduler interleavings.

## Interleaving update

N=0.5 gives decode opportunities between prefill chunks, including speculative
NEXTN decoding. It counts batches, not GPU time or generated tokens. The current
chunked request keeps prefill priority over newcomers; this does not make cache
admission asynchronous. See [scheduling behavior and restrictions](prefill-decode-interleaving.md).

The recipe retains the existing published HiCache/PLE patch stack. No unpublished
cache, abort, tokenizer or throughput diagnostic patches are added. The underlying
scheduling flag still defaults to zero; this recipe explicitly selects0.5.
No production restart or new public container publication is implied by this
recipe change. Combined-profile CPU verification is separate from the historical
GPU/cache evidence above; a fresh full GPU qualification of this image is not claimed.

Recipe update validation: combined build verified4,394source files,28scheduling/
CLI tests and26subtests passed, and5packaging checks passed. The CLI test parses
the flag block above and confirms N0.5,TP2,chunk6144 and write-back HiCache.


### Checkpoint recovery update (2026-09-19)

The opt-in HiCache build includes patches0034/0035 for checkpoint preservation,
coordinated allocation recovery and prefetch namespace propagation. Existing
recipe flags, N=0.5 interleaving and disk format are unchanged. Routine
`action=shared_evict` retries log at DEBUG; exhausted retries and skipped backups
remain ERROR. See [validation and limitations](hicache-wip.md#checkpoint-preservation-and-namespace-fixes-2026-09-19)
before qualifying a deployment. Rebuild the selected image explicitly; a source
update does not change an already running service.

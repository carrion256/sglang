> Patch 0027 has separate GPU and runtime validation, completed 2026-09-15. Earlier results below apply to patches through 0026. See [paged-prefill results and limitations](qsa-paged-prefill.md).

**Scheduling update (2026-09-18):** this profile now includes patch0033 for
separate prefill/decode interleaving. The flag defaults to zero; the
[production recipe](hicache-production-recipe.md) explicitly selects N=0.5.
The earlier GPU receipts below predate this combined build and are historical
qualification evidence. The updated combined build has CPU/source verification;
a fresh full GPU qualification of that image is not claimed.

# Qwen HiCache state transfer — opt-in profile

## Status

**The documented TP2/MTP3 RAM-and-file configuration passed runtime qualification.**
The profile remains opt-in; other configurations require their own qualification.
The series preserves the Qwen-specific HiCache state transfers and adds missing
upstream QSA, router, and restore-order corrections. Historical candidates failed
with repeated punctuation; current-candidate results are recorded separately below.

`Dockerfile.hicache-wip` and `patches/series.hicache-wip` are isolated from every
default and production profile. No repository launcher enables HiCache, and the ordinary `Dockerfile` does not
install this work. The documented TP2 trial uses an explicitly selected derived image.

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

4. `0032-shared-ple-host-table.patch` shares host PLE tables between
   processes holding the identical padded vocab range. Without it each
   process allocates a private pinned copy of its rows: TP1 replicas each
   duplicate the full table (26.8 GiB per replica at the Qwen3.8-Flash
   checkpoint), and TP2 ranks each duplicate their half. With
   `SGLANG_PLE_SHARED_DIR` set, each table is a `MAP_SHARED` tmpfs file
   keyed by shape, dtype, checkpoint path and the padded (tp_size, start,
   end) range, so TP1 replicas share one full table while TP2 ranks share
   only with the matching rank of other replicas and never across ranks.
   Every process pins the mapping with `cudaHostRegister`; the device/host
   pointer equality that `gather_packed_kernel` relies on is verified at
   startup. Sharing is skipped (private pinned table) when the range is
   undeclared or the checkpoint path cannot be resolved, so distinct
   checkpoints never share a file. Sharers write identical checkpoint bytes
   into the same range, so no readiness protocol is required. Disabled
   unless the environment variable is set.


The patch preimages match the 4,392-file cumulative compatibility inventory and
the immutable base image published from current `main`:

```text
kanadaj/sglang-qwen38fn-sm120-turbo@sha256:f2859d1ccf824a5295088cf578eba89b0f3eeefff6ae7679c3f5d64af0689458
```

This parent already contains patches 0015–0019 for Responses, effort aliases,
multimodal aliases, and invalid-token failures. `provenance/hicache-wip.json`
records every hash transition, the ordered patch hashes, and the resulting
4,396-file inventory digest. The new sources are `qsa_pool_host.py`,
`cache_diagnostics.py`, `checkpoint_coordination.py`, and `prefetch_retry.py` under
`python/sglang/srt/mem_cache/`.

## Retained evidence

These historical results were collected on the preserved pre-rebase candidate.
They establish the patch behavior because all eight HiCache preimages are
byte-identical in the cumulative parent, but they do not replace fresh tests of
the rebuilt cumulative candidate:

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

The runner uses no network or GPUs. It executes the CPU suites from
`validation/hicache/` and `validation/prefill/` inside the candidate image;
the current checkpoint/admission/retry profile passes 265 cases. The three GPU files are retained
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

## Qualification scope and limits

- Results apply to the documented TP2/MTP3 configuration and exact runtime
  inventory. They do not establish support for every HiCache backend or layout.
- The built-in file backend and kernel transfer path are covered. Other storage
  backends remain intentionally rejected when PLE companion state is present;
  the direct transfer path is not qualified.
- The prior punctuation failures remain in the historical record. Current
  component regressions and full-model results are recorded below; passing them
  does not prove that every historical failure had one identical cause.
- RAM-only pressure tests did not establish a host hit. The combined file path
  did exercise RAM-to-GPU restoration with correct answers.
- Best-effort prefetch does not guarantee a disk hit for every request, and one
  observed tier report attributed a prefetched prefix to device cache.
- Very short video ordering remains a known limitation. The multi-frame video
  and image tests below use eight frames per color at four frames per second.
- Full-model disk persistence passed after restarting the same image:4/4 correct
  answers, two explicit16384-token disk-hit reports and49152 KV tokens restored
  per rank. Best-effort misses and the tier-attribution limitation remain as above.

### Upstream QSA and router corrections (2026-09-14)

Patch0023 backports merged [upstream38851](https://github.com/sgl-project/sglang/pull/38851).
The two runtime files preserve the upstream change; only surrounding formatting
needed adaptation. The upstream GPU regression is retained in
`validation/hicache/test_qsa_strided_zero_fill.py`. This fixes page-strided scratch
initialization, integer address width and FP8 gather conversion. It is a general
QSA correction, not a HiCache state-transfer feature; final publication should
keep that distinction. Component negatives establish the specific defects; full-model outcomes follow below.

The main-based RAM-only candidate without0023 failed32/48 ordinary tool/replay
cases; disabling scheduler overlap still failed8/48. Single replay and a reduced
mixed replay then completed without the punctuation cascade. These findings do
not qualify the configuration or establish that streaming serialization is broken.

Patch0024 adapts merged [upstream38290](https://github.com/sgl-project/sglang/pull/38290)
by moving PDL waits before bias loads in the Triton and radix router kernels.
It retains the existing bias API; the separate zero-bias optimization is not
required. Compiled PTX on SM120 shows pre-fix first bias load before the wait
and post-fix loads after it, for softmax and sigmoid. Both versions pass settled
routing numerics; the regression distinguishes dependency ordering. Standalone
GPU validation: `python3 validation/hicache/validate_router_pdl_gpu.py` in the
candidate runtime. Current cold-start and full-model results follow below.

### Upstream restore ordering

Patch 0025 adapts merged upstream PR36738 (H2D fencing behind in-flight forwards)
and open PR36743 (final-layer restore completion before deferred Mamba whole-slot
copy). Both waits preserve asynchronous GPU execution; they order dependent work
without disabling scheduler overlap or CUDA graphs. The controller's stream is
wired by the scheduler after cache assembly. The recurrent wait is a no-op without
a registered transfer counter or active load. CPU tests cover fence-before-submit,
empty queues, both checkpoint copy paths and cache-off behavior. A delayed GPU
restore regression fails on candidate3 with all3072 copied elements stale.
Candidate5 full-model and restoration results follow below.

### Short cached extensions

Patch0026 reuses upstream PR39446's bounds clamp before compressed-key gather.
A cached prefix can leave1–3 tokens, while padded compression groups still contain
four gather indices and write only reserved slot0. The prior fallback raises
IndexError and the fused kernel reads outside the source rows. The regression uses
the actual write planner and indexer method:1/2/3 rows fail before the patch;
4/5/8-row complete-group controls pass. Valid complete-group averages must remain
unchanged. Runtime restoration remains a separate qualification gate.

## Current candidate results (2026-09-14)

Candidate5 contains patches0020–0026 on the pinned cumulative parent. Its full
4393-file inventory passed clean application verification;67 CPU cache cases,
5 packaging cases and95 repository tests passed. Real GPU delayed-copy tests
cover both whole-slot restoration and reuse after an in-flight forward. QSA
strided gather and router dependency-order checks also passed.

The TP2 runtime retains MTP3, scheduler overlap, target/draft CUDA graphs,64
request slots,524288 context and image/video inputs. GPU fraction is0.92 with
4342208 KV tokens. Host allocation is30GB per rank, including companion state;
file payload cap is256GB per rank. This produces2541440 host KV tokens and371
Mamba checkpoint slots per rank. The RAM37 trial was stopped by a52GiB available
RAM threshold; the smaller host allocation preserves additional headroom.

| Current-candidate check | Observed result |
|---|---|
| RAM-only cold tool catalogue/replay |48/48|
| RAM-only mixed efforts |96/96|
| RAM-only pressure/retrieval |Correct answers; no host restoration observed, so this does not qualify RAM hits|
| Combined-cache cold retrieval |4/4|
| After idle GPU/RAM flush |4/4 correct; two explicit16384-token disk hits;49152 KV tokens restored per rank|
| Post-restoration tool catalogue/replay |48/48|
| Post-restoration mixed efforts |96/96|
| Image-bearing tool results |16/16|
| Direct Responses client streaming |1033 text updates across9.925s, completed final answer|
| Disk persistence across full restart |4/4 correct; positive disk hits and H2D restoration|
| Final post-restart tool replay |48/48|
| Mixed-load qualification |603.414s passed;64 actual active;72 background requests without client errors; clean cancellation/drain; media32/32;523264-token retrieval; six tool rounds288/288|

The strict disk harness did not pass its100% hit-rate assertion. The configured
`best_effort` prefetch policy allows a request to proceed without waiting for disk:
one case recomputed, and another reported a device hit despite a logged disk
prefetch. Two distinct cases explicitly reported disk-only hits and returned
exact expected answers. Transfer counters independently confirm H2D activity.
Do not report four disk hits or use correct recomputation alone as restoration
evidence. An earlier harness run also waited for more backup tokens than its
per-case restore criterion required; that failed invocation is retained.

### Short-prompt decode performance

The pinned benchmark measured30 seconds at each concurrency with HiCache enabled,
MTP3, normal target/draft graphs and scheduler overlap. Each cell reached its
requested active count, with no reported errors, detected loops, underfilling or
capacity limitation. These are measurements of this configuration, not a matched
cache-on/cache-off comparison.

| Concurrent requests | Aggregate output tokens/s |
|---:|---:|
|1|249.1|
|2|443.9|
|4|711.0|
|8|1074.7|
|16|1551.2|
|32|2170.0|
|64|2843.0|

No additional performance feature was disabled for this candidate. Prefill CUDA
graphs remain disabled as in the baseline; draft-extension graphs are enabled.

The selected instance remained healthy and idle after the final48-case replay,
with no automatic restart and about68GiB RAM available. The minimum sampled
availability over the preceding combined-cache soak and benchmark was62.502GiB.
A configuration-specific systemd monitor checks available RAM every2s and stops
that instance below52GiB; this operational policy is external to the image.

The tested image ID is`313128307b89435233cfd49a5470f710d66502c65daaa550441f5a3aa755483d`.
Qualification was performed before publication metadata was updated; all runtime
source hashes and patches remain unchanged.

## Checkpoint preservation and namespace fixes (2026-09-19)

Patches0034 and0035 address two ways an existing conversation could lose usable
HiCache state despite matching KV pages:

- Preserve Mamba checkpoint endpoints before device eviction cascades through
  component state. Wait for backup ownership to become safe before reclamation.
  Mark endpoint requirements explicitly; a prefix created by a tree split does
  not inherit the child's recurrent checkpoint requirement.
- Reject disk publication of an endpoint missing its required recurrent state.
  Intermediate KV nodes without a checkpoint remain legitimate. Track transfer
  generations and exact host pins so delayed acknowledgements, failed enqueue,
  mutation, and deletion cannot publish incomplete state or leak ownership.
- Coordinate checkpoint frontiers, reservations and rollback across attention
  ranks. Optional allocation failures get one coordinated eviction/retry; failed
  optional cache work is logged and serving continues with recomputation. An
  unrecoverable rank-identity mismatch or failure after device submission is not
  silently swallowed.
- Preserve request `extra_key` and `cache_salt` when disk prefetch starts from the
  shared empty root, in both Python HiCache backends and their callers. Reject
  conflicting non-root anchors before reservation/publication. Disk hashes and
  file formats are unchanged; this is not disk-level tenant isolation.
- Routine reservation eviction/retry messages are DEBUG. Exhausted reservations
  and skipped/failed checkpoint work remain ERROR.

### Validation and limits

The exact reconstructed profile passes 189 CPU tests, including asymmetric
failures with real two-process Gloo collectives, allocator ownership, eviction,
pending transfers, endpoint publication, namespace matching, and existing
PLE/file/QSA/load-order/prefill regressions. Five packaging tests verify patch
order, path/hash transitions, drift rejection and default-profile isolation.
A clean replay of all 15 patches verifies all 4,395 resulting source files.

Equivalent checkpoint changes were exercised in a TP2 production trial: four
RAM and four disk replays each reused61,440 tokens of65,536-token synthetic
inputs, with output token IDs matching cold references. A disk-restored appended
turn reused65,536 tokens and answered correctly. These are bounded observations,
not universal numerical equivalence. The operational image also retains local
diagnostics that are not part of this PR. Following the logging-only restart,
Chat, Responses and streaming smoke checks passed without unexpected restarts.

Salted disk-restored continuation still needs full-model qualification; its
namespace repair has CPU tree/caller/rank coverage. CPU host fixtures synthesize
residency and transfer events, not CUDA copies. Do not force cold-reference tests
with full-input logprobs on long prompts: a separate test exhausted CUDA memory
in prompt-logit conversion. Normal output-only smoke tests do not exercise that
allocation path. No cache format migration, new serving flags, or automatic
service action is introduced.

### Write-back admission completion (2026-09-19)

Patch0036 permits storage lookup below non-root, device-only anchors when the
cache uses write-back. Previously the scheduler required a RAM-backed or root
anchor, so useful disk suffixes could be skipped with `backup_pending` before
restore compatibility was even checked. Root/backed anchors and other write
policies keep their prior behavior. Namespace, companion-state compatibility,
threshold, capacity and transfer ownership checks are unchanged: eligibility
for lookup is not permission to reuse incomplete state.

Validation: the added lookup regression fails against the preceding scheduler
and passes with0036. The expanded CPU suite passes223 cases, covering legacy and
write-through negatives, exact suffix/logprob-boundary and namespace forwarding,
real prefetch rejection/reservation, device-only host pins, checkpoint recovery
and existing cache regressions. Five packaging checks pass; clean replay of all
16 patches verifies4395 runtime files. CPU containers expose no GPU devices,
network or production caches. No new GPU qualification or service restart was
performed for this amendment; earlier runtime evidence keeps its stated limits.


## Bounded prefetch retry and repeat-publication diagnostics (2026-09-20)

Patch0038 gives unified write-back requests one additional disk lookup when their
usable GPU/RAM prefix advances between enqueue and first admission. The probe
avoids Mamba copy-on-write. All attention ranks must agree on eligibility and the
request/anchor identity before retrying. Requests with generated output, positive
storage-hit accounting, prior admission or a consumed retry budget are excluded.
The remaining suffix must still meet the prefetch threshold. The retry budget is
consumed even if enqueue refuses the lookup, so admission cannot loop indefinitely.
Namespace, checkpoint compatibility and allocation checks remain in force.

Patch0037 includes publication history in missing-state diagnostics. Only
`missing_mamba` with a recorded prior successful publication moves to DEBUG.
First-publication failures and other failure reasons remain ERROR. An in-memory
counter and optional Prometheus counter
`sglang:hicache_repeat_publication_rejections_total` with `tp_rank`/`dp_rank` labels
retain visibility into these repeat rejections. Prior success does not prove the
entry still exists on disk; the message explicitly says
`storage_residency=unverified`. This change neither repairs missing checkpoints
nor changes cache eviction or disk retention.

Validation: 265 CPU tests, including the added 42 retry/history/lifecycle cases
and real two-rank Gloo coordination; five packaging checks; clean replay of all
18 patches and full verification of 4,396 source files. The test runner gives
native CPU hash extensions a separate ephemeral executable tmpfs while keeping
ordinary temporary files non-executable. The first run exposed that harness
restriction; rerunning with the dedicated extension directory passed.

The equivalent deployed changes passed 116 focused CPU tests and four bounded
Chat/Responses normal/streaming smoke checks. Those operational tests are separate
from this upstream source reconstruction. Neither establishes that the original
historical idle-session miss is recovered. No new public image, GPU qualification,
cache reset or service restart is performed by this upstream amendment.

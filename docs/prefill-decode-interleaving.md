# Separate prefill/decode interleaving

Long chunked prefills can repeatedly win scheduling while existing requests wait
for decode. This opt-in profile gives decode turns between prefill chunks, while
keeping prefill and speculative decode in separate batches.

## Configuration

Set `--prefill-batches-before-decode N`:

| N | Repeating turns when both kinds of work are available |
| --- | --- |
| 0 (default) | Existing strict prefill priority |
| 1 | Prefill, decode |
| 2 | Prefill, prefill, decode |
| 0.5 | Prefill, decode, decode |
| 0.1 | Prefill, then ten decode turns |

Positive integers and reciprocal integers are accepted; other fractions,
negative and nonfinite values are rejected. Reciprocal recognition uses floating
point tolerance. These are batch turns, not tokens or shares of GPU time. A
speculative decode turn can emit multiple tokens; a prefill chunk usually takes
much longer than a decode turn. No work is held idle merely to satisfy a ratio.

The experimental scope is TP2/PP1/DP1/DCP1 generation with overlap scheduling
and chunked prefill. DP attention, mixed chunking, disaggregated serving,
diffusion, HiSparse, pdmux, two-batch overlap and embedding modes are rejected
when enabled. The default zero leaves existing configurations unchanged.

Existing chunk priority, allocation, cancellation and retraction policies remain.
A parked chunk is stashed once; finishing a prefill joins decode through the
existing path. If filtering or retraction removes every decoder after yielding,
prefill is retried in the same selection step. Grammar/cache housekeeping still
runs before the yield. Admission can still block scheduling; this is not async
admission or round-robin scheduling among multiple long prefills.

## Standalone build and CPU checks

```bash
docker build -f Dockerfile.prefill-decode-interleaving -t sglang-interleaving .
INTERLEAVING_IMAGE=sglang-interleaving bash scripts/test_prefill_decode_interleaving.sh
```

The Dockerfile pins the published cumulative runtime by digest. A separate
one-patch series changes only `server_args.py` and `managers/scheduler.py`.
The verifier checks the complete 4,392-file base/result source inventories,
patch order, patch digest and changed-file transitions. Default profiles and
series are unchanged. This profile requires no HiCache patch profile and adds
no cache, abort, tokenizer or throughput logging changes.

The runner uses a CPU-only, network-disabled container with no production mounts.
Tests execute actual scheduler selection methods with explicit GPU/allocation
and admission doubles. They cover default equivalence, exact integer/reciprocal
sequences, absent work, bounded credits, cancellation, final-chunk merging,
retraction, housekeeping and real CLI registration. They do not prove GPU
allocator or tensor-parallel correctness.

## Validation limits

Clean standalone build passed full 4,392-file pre/post verification on
2026-09-18. CPU suite: **27 passed, 26 subtests passed**, one pytest
import-rewrite warning. No GPU devices or network were available to the runner.
The same scheduling logic was also exercised in a combined local runtime on
two 96GB SM120 GPUs with NEXTN speculative decoding and 6,144-token chunks:
N=2 and N=0.5 allowed decode while a long prefill remained incomplete. At N=0.5,
a 320,055-token synthetic prompt and an exact 1–1024 decoder completed correctly;
scheduler logs corroborated decode between incomplete chunks. Chat/Responses
streaming, tool continuations and controlled cancellation passed. One tool call
took 120 seconds despite passing; no performance improvement claim is made.

That GPU evidence includes unrelated local patches and is not a GPU qualification
of this standalone image. Sampled logs do not establish every P/D/D turn; exact
turn accounting is covered by CPU tests. No throughput benchmark, universal
optimal ratio, asynchronous admission or broader-topology support is claimed.

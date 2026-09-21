# Select a common restorable HiCache boundary

With KV pages A–D, recurrent checkpoints at B and D, and QSA index pages A–C, independent pool lookup selected recurrent D and index C, then returned the minimum C. There is no recurrent checkpoint at C. Restore could consequently request a missing file and discard a usable prefix. The correct common endpoint is B.

Patch0029 intersects valid endpoints for every auxiliary pool after finding the contiguous KV prefix. ALL_PAGES pools require complete prefix coverage; TRAILING_PAGES pools require consecutive tail checkpoints at the actual chosen endpoint. The result is independent of transfer order and supports multiple trailing pools and empty prefixes. Completeness checks stay intact; concurrent eviction can still cause an ordinary miss between lookup and read.

This is one runtime-file change in the existing opt-in HiCache profile. No disk format, transfer ordering, GPU kernel, cache size, scheduling policy or graph setting changes. It is independent of the Responses follow-up and open paged-prefill PR #12. Patch numbers reserve0027for that existing PR and0028for Responses; the local series need not have consecutive numbers.

## Validation

```bash
docker build -f Dockerfile.hicache-wip \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t local/hicache-boundary .
HICACHE_BOUNDARY_IMAGE=local/hicache-boundary bash scripts/test_hicache_boundary.sh
```

The verifier checks the clean cumulative patch chain and all4393 runtime source hashes. CPU tests include3072oracle comparisons, transfer permutations, tail lengths1/2/5,KV gaps,multiple trailing pools,and actual file reads on both rank namespaces with metadata cache on/off. The bounded regression runner also covers existing file,KV/QSA/recurrent state and load-order behavior.

The combined deployed candidate used the identical changed storage file. GPU checkpoint/relocation tests passed5/5 on each physical GPU, including backend reconstruction. Existing disk data reopened across a service transition; two per-request receipts explicitly restored16384tokens each from disk with zero device/host hits and correct answers. Four fixture answers were correct overall; do not interpret the aggregate prefetch counter as proof that all four restored from disk. These are results for the combined deployed profile, not a GPU deployment of the isolated PR image.

The combined318second soak reached64active decoders and passed near512k retrieval,media32/32 and tool/replay96/96; no OOM,cache-read failure,prefetch discard or unexpected restart was observed. RAM remained above71GiB in minute samples with the existing reserve guard. This does not prove multi-day stability or guarantee future hits. A separate non-thinking arithmetic probe fails intermittently across API modes; cause unresolved and not addressed here. No throughput improvement is claimed.

Standalone validation: image b00f76f22127 built from a clean pinned base with4393source hashes verified.5packaging checks and73CPU regressions pass. Personal final review checked endpoint intersection, contiguous coverage, multiple tail windows, zero-length input, transfer-order independence, existing pool counter semantics and package transition hashes. No demonstrated blocker in this file-level fix; the documented runtime and model-quality limits remain.

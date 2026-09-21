# QSA paged prefill — implementation record

Status: GPU and selected-profile runtime validation completed 2026-09-15. Earlier stage notes below are historical.
Base: kanadaj main 9d24d98484eb1062dab75feda52f25a22a09eb10.

Approved design: long-term-best removal of full-context temporary K/V copies for ordinary prefix-bearing prefill. Adapt the existing sparse chunk-prefill kernel to load physical slots from the request table, convert FP8 tiles to query dtype, and retain online softmax, GQA mapping and sparse selection. Keep no-prefix and speculative/decode paths unchanged. Package as an additional patch in the opt-in profile; do not modify default images or production.

Use existing kernel with a compile-time paged-input specialization. The contiguous specialization remains the reference. The paged wrapper receives CPU-derived max query length to avoid an extra GPU scalar read. Use 64-bit offsets and explicit causal/slot masks; all-invalid rows return zeros. Scope excludes scheduler fairness and scratch dtype rollback.

Acceptance: CPU dispatch/packaging tests, staged numerical GPU comparisons and allocation checks, then separately authorized concurrent long-context serving and cache restoration tests. Stop and wait before any GPU access. No deployment or claim of runtime qualification before those gates.

## CPU results

- 11 CPU tests passed in a separate container without GPU devices: paired metadata validation, strided FP8/BF16 pool identity, CPU-derived launch lengths, contiguous reference dispatch, and ordinary/no-prefix/speculative backend routing.
- 5 patch packaging tests passed; default profile isolation retained.
- All eight patches apply cleanly to the pinned cumulative base image. The verifier validated the complete resulting 4393-file source inventory.
- Python syntax and git diff whitespace checks passed.
- No GPU compilation, kernel execution, service restart, image deployment, or production configuration change occurred in this implementation stage.

## GPU gate plan (subsequently executed)

`validation/prefill/test_paged_prefill_gpu.py` stages 16 numerical cases (FP8/BF16, one/two KV heads, short/empty extensions, invalid entries), a 512k-prefix temporary-memory check and an address-offset test above 2^31. The latter needs about 4.1 GiB for its test pools. These tests have NOT run, and no tolerances or memory thresholds have been validated on hardware. They compare against both the previous contiguous kernel and an independent per-query reference.

After isolated GPU gates, the final candidate still needs matched prefill performance and full-engine concurrent long-context/HiCache/MTP/media validation. Patch 0027 is included only in the opt-in build chain. The pinned base image remains unchanged because latest main adds the merged patch series and verifier repair rather than a replacement runtime base.

Implementation edits two runtime files: sparse_attn.py and qwen_sparse_attn_backend.py. Existing decode scratch dtype, load waits and scheduler selection are unchanged. The contiguous prefill specialization remains callable as a numerical reference. FP8 conversion retains the current unit-scale assumption.

The implementation stage paused before GPU access as requested. The subsequent GPU window was explicitly authorized; results follow.

## Authorized GPU window

User authorized GPU access and continuation. Existing service stopped through systemd; baseline image and launcher retained. 18 isolated GPU cases passed on GPU0. Candidate image dd2b6d5cfa314e8db8eb5242e639f43c33c213c4286eb65b96f48821b47eee68. Proceed with a distinct candidate launcher at 0.91, cap4342208, same RAM/disk cache limits and normal MTP/graphs. Restore baseline launcher if candidate cannot pass. No LiteLLM actions.

## Isolated GPU results

18/18 tests passed on GPU0 in8.55s, including numerical comparisons, 512k temporary-memory bound and >2^31 offsets. One unregistered pytest marker warning is test metadata only.

Matched same-kernel old-allocation vs paged-input benchmark, five measured iterations after warmup, FP8 cache, BF16 query, two KV heads, head_dim256, 24 query heads. Figures are per attention invocation, not whole-model throughput:

| Requests | Context each | Extension each | Old ms | Paged ms | Old peak extra MiB | Paged peak extra MiB |
|---|---:|---:|---:|---:|---:|---:|
|1|32768|4096|2.798|3.076|176|48|
|1|524288|8|3.229|0.186|2048.094|0.094|
|1|524288|4096|5.544|3.095|2096|48|
|4|524288|1024|14.429|3.299|8240|48|

Outputs matched at atol/rtol0.01. The 32k case is approximately10% slower; do not claim a universal speedup. Benchmark code: validation/prefill/benchmark_paged_prefill.py. These figures exclude the KV pool itself and other engine allocations.

## Full-model runtime results — 2026-09-15

Selected runtime: candidate1 image dd2b6d5cfa314e8db8eb5242e639f43c33c213c4286eb65b96f48821b47eee68; mem fraction0.91, cap4342208 and actual4342208 KV tokens; TP2/MTP3/C64/524288 context, RAM HiCache60GB aggregate and disk512GB. The profiled token capacity remained above the cap, so lowering the fraction did not reduce this allocation. No new graph or overlap disablement. Baseline prefill graphs remain disabled.

- Concurrent long histories: 24/24 exact-value checks with finite output log probabilities over1033.455s. Eight independent300000-token histories and four500000-token histories, each cold and replayed after pressure.
- Reasoning effort:96/96 across Chat, Responses and Messages, streaming/non-streaming.
- Automatic tool catalogue/replay:48/48; image-bearing tool results16/16.
- Mixed-load functional gates over336.337s: media32/32,523264-token exact retrieval in66.738s, two additional48-case tool rounds96/96,64 background requests and zero background client errors, peak64 actual running requests.
- The mixed-load harness overall assertion FAILED: one of the16 intentionally cancelled warm-up streams had emitted only one event. The other63 had multiple events; retained48 streams all progressed. This is retained as a failed harness gate, not relabelled PASS.
- Focused follow-up: explicit barrier requires every one of64 client streams to emit at least two content deltas before cancellation. PASS in6.390s, minimum49 deltas, peak64 actual running, zero errors. This closes the warm-up progress proof gap without repeating the already passing functional gates.
- External clients reconnected during the window. Whole-engine timing is not a controlled benchmark. Own test streams were closed; an immediate global remaining-running count is not a drain verdict because external traffic remained and cancellations are asynchronous.
- No allocation-failure warnings, fatal OOMs, scheduler exceptions or automatic restarts observed in this invocation. Minimum sampled host MemAvailable during the recorded test window was60.675GiB; GPU free memory was sampled every5seconds and reached2330MiB, not an instantaneous peak measurement.

All4393 runtime source hashes verified in the running container. The immutable image manifest retains its build-time pending-validation label; this dated record provides the subsequent validation result. The package remains opt-in. The result qualifies the tested deployment/workloads, not all conceivable OOM conditions or every attention backend. HiCache best-effort misses and short-video ordering limitations remain; the long replay wave often recomputed rather than proving restoration. Per-layer get_key_buffer/get_value_buffer restoration waits remain in the call path.

## Review

Personal final review found no blocking defect in the approved scope. Checked paged64-bit addressing, causal/padding masks, empty-tile softmax behavior, dtype conversion, unchanged speculative/no-prefix dispatch, existing per-layer cache-transfer waits, package hashes and default-profile isolation. Validation includes the independent attention reference and preserved contiguous kernel. The32k microbenchmark regression remains a disclosed trade-off; no universal speed claim is made. Runtime code did not change after GPU tests.

Final packaged CPU run:78 tests passed plus5 packaging tests; two dependency deprecation warnings. The temporary trial watcher ended normally; permanent RAM-reserve monitoring remains active.

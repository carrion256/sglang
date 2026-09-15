# Responses stream stability

Two runtime changes, independent of HiCache and the separate paged-prefill PR #12:

- Qwen inner function/parameter markup is recognized only inside a tool_call wrapper. For example, ordinary text `Example syntax: <function=function></function>` previously became an attempted call to `function` during streaming. The pinned template requires the outer wrapper; wrapped calls retain their identity across arbitrary chunk boundaries.
- Exceptions during generation or completion validation emit one diagnostic `response.failed`, preserving output already emitted. Unknown identities remain rejected. Stored failed responses can be retrieved, but cannot overwrite a completed/cancelled response with the same ID. Completion cardinality failures use the same terminal path.

## Package

Patch0028 changes only serving_responses.py and qwen3_coder_detector.py. The new Dockerfile.responses-stability applies it to the pinned cumulative API image and verifies all4392 runtime source hashes. Earlier image recipes, overlays and their test expectations remain unchanged. This profile does not include HiCache.

```bash
docker build -f Dockerfile.responses-stability \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t local/responses-stability .
RESPONSES_STABILITY_IMAGE=local/responses-stability \
  QWEN_TOKENIZER_PATH=/path/to/pinned/tokenizer \
  bash scripts/test_responses_stability.sh
```

Use the tokenizer revision documented in docs/responses-compat.md. Resolve snapshot symlinks or mount a directory containing their targets. Tests run in bounded CPU-only containers with no GPU devices.

The standalone image and an existing HiCache image are distinct profiles. To combine manually, apply this patch only after verifying its two source-file before hashes. The HiCache source changes do not overlap this patch. Do not substitute the standalone profile's full inventory for a combined image inventory.

## Evidence and limits

The combined deployed candidate used byte-identical versions of these two changed source files. Its live checks passed literal-markup stream/nonstream2/2,tool-result images16/16,tool/replay96/96,media32/32 and near512k retrieval during a318second mixed soak with64active decoders. No Responses exception occurred in that bounded run. Those live receipts describe the combined profile, not a GPU deployment of this new standalone image.

One bundled upstream test expects a newline-concatenated replay string. The unchanged parent also fails that assertion; the cumulative API preserves separate text parts. The upstream runner adapts only that representation in a temporary copy; no tests are excluded. Original suite:60pass/1pre-existing failure. Adapted suite has61tests. The new HTTP suite inherits the predecessor suite, overriding only changed failure-contract assertions and adding two scenarios; historical profile tests remain intact.

Historical incident raw model text was unavailable, so the exact cause of two invalid tool-name incidents remains unproven. The patch does not prevent a model from generating unknown names. A separate arithmetic probe produced wrong non-thinking answers across all three APIs, streaming and nonstreaming; its cause remains unresolved. This PR does not claim to fix it, prove general model accuracy or establish multi-day stability. No performance feature is disabled by these changes.

Standalone validation: image cfc2c11fd275 built from a clean pinned base with4392source hashes verified.3packaging,4parser,77compatibility and61adapted upstream tests pass. Personal final review checked profile isolation, before/after hashes, parser boundaries, terminal diagnostics and existing stored-response identity protection. No demonstrated blocker in the reviewed change; arithmetic causation remains outside the proven fix.

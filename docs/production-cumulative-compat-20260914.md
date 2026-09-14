# Published cumulative compatibility runtime — 2026-09-14

## Immutable image

```text
docker.io/kanadaj/sglang-qwen38fn-sm120-turbo:production-cumulative-compat-20260914-v3
docker.io/kanadaj/sglang-qwen38fn-sm120-turbo@sha256:f2859d1ccf824a5295088cf578eba89b0f3eeefff6ae7679c3f5d64af0689458
```

Source revision: `facd7be72dc5abcfc8d99c9e6fa750e73ad8e350`.
Registry config digest:
`sha256:751a89104df49a9777ad577dc48aa4c2ed833c4f86b373c633f96ac5fc1756bb`.

The runtime keeps serving arguments external. It does not embed a production
launcher, model path, checkpoint, route, or GPUStack configuration.

## Included compatibility stack

The image applies, in order:

1. `0015-qwen-flash-next-effort-alias.patch`
2. `0016-responses-namespace-custom-boundary.patch`
3. `0017-responses-phase-order.patch`
4. `0018-qwen-flash-next-multimodal-alias.patch`
5. `0019-invalid-generated-token-failure.patch`

This includes Qwen-only effort aliases, Responses namespace/custom-tool and
phase/order behavior, release-name multimodal processor paths, and consistent
invalid-token failure propagation through Chat, Completions, non-Harmony
Responses, and Harmony Responses.

## Verification

The image was built from a fresh clean clone using:

```bash
docker buildx build --load --provenance=false --network=none \
  --no-cache --pull \
  --build-arg SOURCE_REVISION=facd7be72dc5abcfc8d99c9e6fa750e73ad8e350 \
  -f Dockerfile.invalid-token-failure \
  -t docker.io/kanadaj/sglang-qwen38fn-sm120-turbo:production-cumulative-compat-20260914-v3 .
```

Verification completed both from the clean source revision and inside the built
image:

- invalid-token runtime: 16 tests;
- Responses compatibility: 75 tests;
- effort aliases: 14 tests;
- multimodal aliases: 4 tests;
- full package: 90 tests;
- dedicated packaging: 17 tests;
- two independent complete source reconstructions;
- 4,392 expected and present source files, with zero missing, extra, or
  mismatched files.

The source-revision label, entrypoint, command, manifest, registry config, and
canonical manifest bytes were checked. Anonymous manifest access and pulls by
both tag and digest succeeded.

## Evidence boundary

This is a source, CPU/package, image, and registry-publication result. It is not
a production rollout. No GPU model boot, invalid-token fault injection in a
live speculative engine, or semantic video-order qualification was performed
for this published image.

Local publication receipts were retained outside the repository at
`/home/kanadaj/sglang-publication-20260914`.

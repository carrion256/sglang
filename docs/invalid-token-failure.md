# Invalid generated-token failures — published cumulative runtime

This profile is merged into `main` at
`facd7be72dc5abcfc8d99c9e6fa750e73ad8e350`. It preserves original PR #8
head `4059ace2b2faa2d7972f7704c8fdca0985a35b1a`, the reviewed functional
corrections, and the final Harmony terminal-event fix. The cumulative image is
published at the immutable digest documented in
[`production-cumulative-compat-20260914.md`](production-cumulative-compat-20260914.md).
It has not been deployed to the running production service.

## Behavior

An out-of-vocabulary generated token is a fatal engine error, not an ordinary
stop. The scheduler replaces the token only to keep downstream decoding safe,
excludes it from emitted output, records `FINISH_ABORT` with HTTP 500 and
`err_type=InvalidTokenError`, and does not allow a speculative output-length cap
to overwrite that error.

- **Chat:** a serialized integer status emits one `InvalidTokenError` SSE error,
  then `[DONE]`; it does not continue through the ordinary choice/usage path.
- **Completions:** uses the same integer-status and error-type behavior, emits
  `[DONE]`, and returns before any ordinary abort choice or usage event.
- **Responses:** non-stream and stream terminals use `status=failed`, attach a
  `server_error`, retain partial output, and emit `response.failed`, including
  the Harmony streaming path. Incomplete, failed/cancelled, and completed
  Harmony terminals now select the same event classes as non-Harmony. Stored
  failed responses remain retrievable. A failed response used as
  `previous_response_id` is rejected with HTTP 400 and
  `param=previous_response_id` before registry replay, preprocessing, or
  generation.
- **Cancellations:** an abort without an error status remains the existing
  graceful Chat/Completions abort or Responses `cancelled` terminal.

## Composition and source identity

`patches/series.invalid-token-failure` is the exact cumulative order:

1. `0015-qwen-flash-next-effort-alias.patch`
2. `0016-responses-namespace-custom-boundary.patch`
3. `0017-responses-phase-order.patch`
4. `0018-qwen-flash-next-multimodal-alias.patch`
5. `0019-invalid-generated-token-failure.patch`

Patch 0019 is regenerated against the post-0018 tree. Its
`serving_responses.py` retains the complete PR #5 phase/order parser behavior and
adds only PR #8 failure behavior. Its `serving_chat.py` retains the cumulative
effort behavior. The PR #7 `qwen_vl.py` bytes remain unchanged at SHA-256
`b47003e1f0840a057519adff46fc72a9318a61e2eb3ef8cedfa9eab19e98b7f7`.

The four post-0019 runtime files, including `serving_completions.py`, are under
`runtime.invalid-token-failure/`. The full 4,392-file result inventory is
`provenance/invalid-token-failure-runtime-files.json`; its SHA-256 is
`f7293cc004161bcad39bc3772939fd868f8fc9d6f09d2cdb7b7ffd5a23333e1d`.
`provenance/invalid-token-failure.json` binds base/head identities, patch and
inventory hashes, every changed-file preimage/result, test counts, and evidence
log hashes. The verifier fails closed on chain, series, runtime, patch,
inventory, PR #7 byte, count, or evidence drift.

## Verification

Pinned tokenizer/config metadata came from the recorded local release snapshot.
CPU/GPU-disabled checks completed against the exact base image:

- 16 focused runtime tests for scheduler/API error behavior, including the
  Harmony terminal-event regression captured RED before the runtime fix;
- 4 invalid-token packaging contract tests;
- 75 cumulative Responses tests;
- 14 effort tests;
- 4 multimodal tests;
- 90 full-package tests;
- 17 dedicated packaging tests;
- two independent exact-image reconstructions, each checking all 4,392 source
  files and producing tree digest
  `72d547ccf24958a58b97a931b8535b97327f4b8bfc03ef368e478d5abd58a490`;
- local Docker build plus image readback: 4,392 expected, 4,392 present, zero
  missing, extra, or mismatched source files. The recorded candidate build digest
  is `sha256:6122dbac2c26cb7852b9e8e44787e95ade096de82929dc9d2d93d53a86aaa227`.

Run the focused gate:

```bash
QWEN_TOKENIZER_PATH=/absolute/pinned/release/config-directory \
  bash scripts/test_invalid_token_failure.sh
```

Reconstruct from an extracted exact base image source tree:

```bash
python3 scripts/verify_invalid_token_failure.py --tree TREE --from-image
python3 scripts/verify_invalid_token_failure.py --tree TREE
```

Build the local cumulative image with a source-revision label:

```bash
docker build --pull=false \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -f Dockerfile.invalid-token-failure \
  -t sglang-pr8-final:local .
```

## Limits

No GPU generation was forced to produce an invalid token. The change makes the
existing fatal condition visible and replay-safe; it does not attempt generation
recovery. The published image passed exact source, CPU, package, and anonymous
registry-transfer checks, but has not been booted with the production model or
deployed.

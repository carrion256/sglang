# Invalid generated-token failures — CPU candidate only

The scheduler currently replaces an out-of-vocabulary generated token with an
EOS token and reports an ordinary stop. A speculative step at the output limit
can further replace that result with a length finish. Clients therefore receive
a successful response even though the engine produced an invalid token ID.

This profile reports that condition as an HTTP 500 `InvalidTokenError`, excludes
the faulty token from output, and prevents the length cap from hiding the
failure. Chat and Anthropic-compatible streams emit the serialized error and
terminate. Responses requests finish with `status=failed`, a `server_error`
payload, and `response.failed`; graceful aborts without an error status remain
cancelled.

## Composition

Patch `0019-invalid-generated-token-failure.patch` applies after the existing
effort and Responses compatibility patches. The patch changes the scheduler and
the Chat and Responses adapters because all three layers are required to carry
the failure to clients. Their post-patch bytes live under
`runtime.invalid-token-failure/`, leaving the predecessor profile's `runtime/`
snapshot unchanged. `Dockerfile.invalid-token-failure` combines those files with
the three unchanged API compatibility files. Model paths, serving arguments, and
runtime settings remain external.

## CPU validation

The runner requires the exact base image to be present locally and a pinned
Qwen tokenizer directory:

```bash
QWEN_TOKENIZER_PATH=/absolute/release/config-directory \
  bash scripts/test_invalid_token_failure.sh
python3 scripts/verify_invalid_token_failure.py
```

The runtime test covers negative, vocabulary-boundary, and very large token IDs;
speculative overruns at several output caps; invalid first tokens; unchanged
ordinary stop and length finishes; tokenizer-state cleanup; serialized HTTP
status values; Chat, Anthropic Messages, and Responses streaming; and Responses
non-streaming terminals. It runs in a read-only, network-disabled, GPU-disabled
container.

For a complete source reconstruction, export `python/sglang` from the exact base
image into `TREE`, then run:

```bash
python3 scripts/verify_invalid_token_failure.py --tree TREE --from-image
python3 scripts/verify_invalid_token_failure.py --tree TREE
```

## Limits

The candidate has CPU regression and full-source reconstruction coverage. It
has not been built, published, deployed, or exercised by deliberately forcing
an invalid token on a live GPU engine. The change does not attempt to recover
generation after an invalid token; it makes the existing fatal condition
visible to clients.

# Chat reasoning-effort precedence fix (2026-09-13)

**Status: implementation and CPU validation complete; publication and production
rollout are separate gates and are not claimed by this commit.**

## Contract

For Chat Completions, a non-null explicit `chat_template_kwargs.reasoning_effort`
continues to win a conflict with top-level `reasoning_effort`. Otherwise an
explicit non-null top-level effort wins over the server default. Null means no
override; omitted effort still uses the server's `medium` default. Other template
kwargs retain their existing merging behavior. The shared renderer's direct
Responses/tokenize callers retain their existing policy.

The production Qwen template supports `xhigh`, `medium`, and `low`; **`high` is
rejected by that unchanged template**, rather than silently converted to medium.
No template changes, new effort budgets, kernel optimizations, sampling changes,
or serving flags are included. Invalid values still fail protocol/template
validation. Defensive shallow copies prevent normalization from mutating a
caller-owned kwargs dictionary.

## Source and reconstruction

Adapted from `jpezzulli/sglang-rtxpro6000` commit
`acd23bbe1f8110e36fee0199b8942deb1c5dc477`, plus the defensive copies.
The input is the actual deployed image descriptor
`sha256:69f1f64c62ca2b5d919d69bcd60efb7fc89d7bf410ba406a5de586051f358465`.
Its full source profile is the existing `production` profile, not the combined
HF-repository-ID profile. This fix adds exactly two changed source files.

- `patches/0014-chat-reasoning-effort-precedence.patch`: apply **after**
  `series.production`, or to a source tree already verified as production.
- `runtime/python/sglang/srt/entrypoints/openai/{serving_chat,protocol}.py`:
  resulting source files for the narrow derived build.
- `provenance/chat-effort.json`: exact preimage/result hashes.
- `scripts/verify_chat_effort.py --tree /path/to/sglang`: full 4,391-file
  production-plus-effort source profile verification.

Existing `production` and `combined` profiles remain historical and unchanged.
The private production base is locally retained, not a publicly pullable base.
Do not pretend the clean published source-equivalent rebuild is byte-identical.
On the actual worker, the exercised build procedure is:

```bash
test "$(docker image inspect draft-head-only-candidate:v1 --format '{{.Id}}')" = \
  sha256:69f1f64c62ca2b5d919d69bcd60efb7fc89d7bf410ba406a5de586051f358465
docker build --pull=false --network none -f Dockerfile.reasoning-effort \
  -t kanadaj/sglang-qwen38fn-sm120-turbo:production-chat-effort-20260913-v2 .
```

This is a Dockerfile-derived runtime, **not `docker commit`**. The explicit normal
SGLang entrypoint has no bundled serving profile. All launch flags stay external;
retain the full existing production command rather than changing its TP2,
524288/YaRN2 target+draft, packed PLE, private W4A16 NEXTN3/1/4, Mamba512/track128,
vision, sampling or routing settings. Versioned historical tags are not replaced.

## Exercised tests

- RED on actual imported Chat conversion + actual deployed tokenizer: explicit
  xhigh produced medium token IDs, in both stream/nonstream conversion fixtures.
- RED shared-dictionary mutation tests on normalization; then GREEN with copies.
- `scripts/test.sh`: **77/77 passed** in the derived runtime, GPUs inaccessible,
  `TRITON_INTERPRET=1`, CPU thread limits, network off.
- `python3 tests/runtime_chat_effort.py -v`: **4/4 methods passed**, with subcases
  for default, top-level and nested xhigh/high/medium, conflicts, null, invalid
  values, independent requests, and shared renderer compatibility. Set
  `QWEN_TOKENIZER_PATH` to a read-only directory containing the real deployed
  tokenizer. No model weights are loaded. Streaming here tests the actual
  pre-generation conversion path, not an end-to-end SSE server.
- Full profile verification: **4,391 source paths/hashes passed**; only the two
  documented files differ from production.
- Independent Codex review: passed with no security or logic findings.

These establish CPU/source correctness, not production deployment, a new
performance gain, or completion of the frozen LiveCodeBench campaign. No GPU
serving replica was stopped during this implementation/test stage.

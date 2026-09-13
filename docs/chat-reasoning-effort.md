# Chat reasoning-effort precedence fix (2026-09-13)

**Status: published, rolled out to both production TP2 replicas, and verified.**
The frozen100 LiveCodeBench follow-up is a separate campaign; its completion or
score is not implied by this rollout receipt.

## Pull and run

```bash
docker pull docker.io/kanadaj/sglang-qwen38fn-sm120-turbo:production-chat-effort-20260913-v2@sha256:872a2bda228e39aa9c1af729b47cc28f7862e7859e448f1a8868b85a4051f404
```

Use the [complete external launch command](chat-effort-command.sh). Only its
image reference changed from the previous production-source command; no serving
flag is hidden in a wrapper. Actual-image CLI parsing accepted all47 external
argument tokens. This does not claim a separate standalone full-model boot.

## Production and publication verification

- Canonical model31 desired/ready **2/2**. Sequential replacements preserved a
  healthy peer: old254 → new258 on GPUs0/1; old255 → new259 on GPUs2/3.
- Both run image/index digest `872a2bda228e39aa9c1af729b47cc28f7862e7859e448f1a8868b85a4051f404`
  using durable backend version `qwen-chat-effort-20260913-v2-custom`.
- All effective serving flags and model environment variables unchanged,
  including private W4A16 NEXTN3/1/4, TP2, packed PLE, vision, 524288/YaRN2,
  Mamba512/track128. Routes/target IDs and fallback18 preserved; GLM38 remains0.
- Both replicas passed actual Chat prompt-ID equality: omitted effort matches
  **medium15 tokens**, explicit top-level or nested xhigh matches **xhigh57**;
  conflicting nested medium wins. Known-answer text and red/blue vision passed
  directly and through all three protected aliases. Routed counter deltas prove
  both replicas received work. Unsupported high and boolean effort returned400.
- Native monitoring reports both replicas up1 with all four Mamba metric families.
  The bounded watcher observed53 samples, zero without a healthy replica, zero
  unknown samples. It began during first startup; earlier maintenance had
  separate peer-health checks, not continuous watcher coverage.
- Anonymous registry read verified the versioned tag and index/platform
  manifests. Empty-auth Docker pull succeeded (daemon layers reused); the exact
  pulled digest repeated77/77 CPU tests,4/4 Chat test methods and4,391-file source
  verification. No independent cold-cache transfer is claimed for this image.
- Publication audit scanned all95 retained filesystem layers, including hidden
  overwritten/deleted files. The first68 match the previously audited public
  base; findings are identical to that base's already-dispositioned crypto/test
  fixtures and shared upstream SSH host keys. No new-layer credential hits or
  model payloads. The published v2 has exactly the audited v1 filesystem layers;
  its config replaces the unused inherited wrapper with normal SGLang startup.
  Legacy inherited launch files are unused; serving arguments remain external.
  Never expose the base's shared SSH host keys as an SSH service.

See `provenance/chat-effort-rollout.json` and
`provenance/chat-effort-publication.json`. These are sanitized receipts, not raw
management snapshots. The clean preimage→patch reconstruction also independently
verified all4,391 source files before and after applying0014.

## Rollback

The previous local image and backend remain retained, not overwritten:
`draft-head-only-candidate@sha256:69f1f64c62ca2b5d919d69bcd60efb7fc89d7bf410ba406a5de586051f358465`,
backend `qwen-private-draft-head-20260911-v1-custom`. Restore only the canonical
backend selection from the saved full ModelUpdate, then drain and replace one
selected replica at a time, proving its replacement healthy before retiring the
peer. Keep replicas2, all model flags/environment and complete route targets
unchanged. Retarget instance-specific monitoring after each verified replacement.
The exact private rollback payload remains outside Git; no rollback was needed.

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

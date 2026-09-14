# Responses namespace/custom boundary — CPU candidate only

This profile follows alias commit `04e0816a68638e85ddd4ff8764b401d3ed27997e`.
It does not upgrade the engine, change kernels/schedulers/checkpoints, or change
any deployment. No image is built or published. Historical production, combined,
Chat-effort and alias manifests/series retain their original meanings.

The exact base image and upstream reference heads are recorded in
`provenance/responses-compat.json`. References #39174, #38359, #38690 and #35216
informed the design; this is not a wholesale serving-file transplant. The serving
file starts from the attested installed runtime; patch 0016 contains its narrow
delta. The only new installed module is `responses_compat.py`.

## Contract

- Namespace function/custom members reach the existing rendering paths as qualified
  function declarations. Custom functions have one string property, `input`.
  Namespace and member descriptions are both retained with separate sections,
  along with schemas and strict flags. Custom grammar syntax and definitions
  are model-facing descriptions, not grammar enforcement.
- Identities use an exact request-local map, not prefix matching or dot splitting.
  Dotted flat/namespace/member names are supported. Qualified collisions, unsupported
  namespace members, malformed declarations and invalid forced choices fail closed.
  If a forced choice supplies both outer and nested identity fields, their shapes
  must be valid and overlapping fields must agree before generation starts.
- After the pinned SDK serializes internal calls, the wire boundary restores
  `name`/`namespace`, custom `input`, declarations and literal reasoning effort.
  This avoids losing extension fields through the SDK's older output unions.
- POST responses, SSE added/delta/done/terminal frames, retrieval and cancellation
  use the same mapping. Internal stored calls remain qualified function calls.
  Stateless input dictionaries retain extensions; stateful replay includes prior
  tool calls as well as assistant text. Structured results pass through existing
  Chat content processing, including images, rather than becoming empty strings.
  Stored history carries exact identities from the whole conversation, including
  explicit replay items and each preceding generation across no-tool turns.
  Historical identities are collision checks only, never active declarations that
  authorize newly generated calls. Output-only continuations use the same checks.
- Validated stored responses, identity maps and input history are published under
  the same storage lock. Failed or disconnected streams do not replace an existing
  response's identity/history. Stream adapters close wrapped iterators and abort
  unfinished generation, including disconnects before and after generation starts.
  Background responses require `store=true`; disabled or null storage is rejected
  before generation or state creation.
- Qwen native parsing preserves boundary newlines and literal `null` in Responses
  custom `input` only. The opt-in is per parser instance and comes from private
  request state. Ordinary Chat/function parsing keeps its previous behavior.
- Native custom input is **not an arbitrary-string transport**. Unescaped
  `</parameter>`, `</function>`, `</tool_call>` and `<parameter=...>` are syntax,
  not payload: they can terminate/truncate input or cause rejection. No native
  escaping/unescaping scheme is implemented. Use the JSON required-output path
  with an ordinary JSON-encoded `input` string for delimiter-bearing payloads;
  selecting the native parser does not automatically switch to that path.
  Tests pin the current discrepancy: embedded `</tool_call>` survives streaming
  input but truncates nonstream input. This is a limitation, not a round-trip guarantee.
- Embedded Chat replay supports assistant `tool_calls` with `id`,
  `type="function"`, and `function={name, arguments, namespace?}`. Each call uses
  the same exact identity checks and history registry as typed replay. Flat and
  dotted-flat names remain flat; they are never inferred to be namespace members.
  Legacy `function_call`, embedded custom calls, misplaced namespaces and ambiguous
  fields are rejected; use typed `custom_tool_call` for custom replay. Historical
  identities persist through no-declaration turns but never authorize generation.
- Successful selected-tool completion requires exactly one matching call;
  `required` needs at least one call. Empty/reasoning-only stopped generations
  fail before successful terminal events or storage. Length caps return
  `incomplete` (`response.incomplete`); engine aborts return `cancelled`
  (`response.failed`, since the pinned SDK has no cancellation event). Neither
  emits `response.completed`.
- Terminal length/abort classification precedes successful tool-output validation.
  Truncated JSON strings, objects and call arrays are not parsed as successful calls;
  native streams do not flush or complete pending calls on length/abort. Pending
  calls may have in-progress added/argument-delta events, but receive no executable
  completed item or arguments/input-done event. Unfinished calls are omitted from
  terminal output, storage and subsequent replay. Nonstream interrupted tool-bearing
  output is conservatively omitted (reasoning is retained); a stream may retain
  earlier calls already completed before the interruption. Ordinary auto-mode text
  remains available. Usage, request metadata and max-output-token reason survive;
  `store=false` still prevents retrieval. Successful-stop malformed JSON and exact
  identity/cardinality validation remain unchanged.
- Unknown generated identities fail closed. Required JSON streaming is buffered
  until a complete validated call list is available because the pinned incremental
  parser silently drops unknown names. Custom argument JSON is decoded only when
  complete, then emitted as a raw-input delta and done event. There is no claim of
  token-by-token custom-input latency.

## CPU reproduction

Use the four unmodified tokenizer metadata files pinned in
`provenance/qwen-effort-alias.json`; no weights or GPU are needed:

```bash
QWEN_TOKENIZER_PATH=/absolute/pinned/tokenizer bash scripts/test_responses_compat.sh
QWEN_TOKENIZER_PATH=/absolute/pinned/tokenizer bash scripts/test_qwen_effort_alias.sh
```

The candidate runner verifies all five mounted runtime files against the cumulative
inventory, as well as patch/tokenizer hashes. It uses the exact
existing image with no pull/network/GPU, read-only root and mounts, scratch caches,
dropped capabilities and CPU/memory/PID limits. Tests import the actual serving
modules and exercise `http_server.app` endpoints through ASGI TestClient. Only
generation is an injected controlled CPU manager, explicitly labelled **MOCK**.
There is no alternate endpoint implementation or mocked serving method. Existing
alias/Chat/Responses/tokenize tests and `scripts/test.sh` also run.

The pinned image contains OpenAI Python SDK 2.6.1. Imported SDK output and event
unions, JSON serialization, and the actual SDK client against ASGI endpoints are
tested for function/custom calls, empty and whitespace custom input, call IDs,
namespace extensions, streaming terminal output, and replay without declarations.
The SDK has no declared `namespace` field on either call model: it retains this
field as an extra, not a validated identity. Its default permissive client parser
preserves terminal outputs, but strict `Response.model_validate` rejects namespace
tool declarations because its tool union predates that declaration type. These
tests do not claim strict SDK schema support for namespaces or grammar enforcement.
They use a fixed mock API-key string, never personal credentials or network access.

For complete reconstruction, export `python/sglang` from the exact image into
`TREE`, then run:

```bash
python3 scripts/verify_responses_compat.py --tree TREE --from-image
python3 scripts/verify_responses_compat.py --tree TREE
```

The verifier checks all 4,391 image files, applies 0015 then 0016, and checks all
4,392 resulting paths and hashes including the new module. Alternatively, `--apply`
accepts a completely verified alias predecessor tree. Neither modifies historical
profiles. Source equivalence is not byte-identical image reproduction. Do not use
historical Dockerfiles with the changed overlay to claim reproduction of an
unchanged published image.

## Limits

These are CPU boundary/package results, not full Codex/model/GPU conformance.
Tests exercise the pinned Qwen3 coder parser and required JSON fallback. Harmony
shares normalized declarations/replay but has no GPU validation here. Native
grammar enforcement, hosted agent execution, `agent_message`/encrypted
cross-provider reasoning, `/responses/compact`, image encoding/accuracy,
performance and rollout are not implemented or certified. Unknown replay item
types still error. Existing in-memory response storage lifetime policy remains;
identity metadata follows stored response IDs and does not survive restarts.

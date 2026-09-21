# Selective strict tool defaults for Qwen Flash Next

Qwen can generate an almost-correct tool name or arguments absent from the advertised
schema. When the caller omits `strict`, the existing default is false and auto tool
calling can run without a structural grammar. This opt-in profile makes omitted
function-tool strictness true for the loaded Qwen Flash Next checkpoint.

- Explicit `strict:false` remains false; explicit true remains true.
- `tool_choice:auto` still permits plain text. `none` still disables tool constraints.
- Applies to Chat top-level and system/developer message-local tools, Responses
  functions/namespaces, and Messages custom tools through their shared Chat adapter.
- Uses the loaded checkpoint type, not a caller-supplied model alias.
- Other models retain existing defaults. Protocol field types remain unchanged.
- Normalization copies changed containers and leaves caller requests unchanged.
- Responses normalizes before registry flattening, while field presence is available.
- No global `SGLANG_TOOL_STRICT_LEVEL=2` override: that would also override explicit
  false. No reasoning, model, MTP, graph or cache-setting changes.

Strictness constrains tool syntax/schema, not whether a valid query is useful or
correct. Unsupported schemas can still trigger the runtime's existing grammar
fallback; the current parser logs `Error getting structure constraint` in that case.
This patch does not replace that error policy. Grammar compilation can add first-use
latency and constrained decoding can affect speculation acceptance; no throughput
speedup is claimed. Message-local tool normalization does not add template support
for previously unsupported message roles.

## Build and test

```sh
docker build -f Dockerfile.qwen-strict-tools -t local/qwen-strict-tools .
QWEN_STRICT_IMAGE=local/qwen-strict-tools \
QWEN_TOKENIZER_PATH=/path/to/tokenizer-with-resolved-files \
  bash scripts/test_qwen_strict_tools.sh
```

The profile builds from the pinned public cumulative image, applies existing0028
Responses stability followed by0032, and verifies the full source inventory. It has
no HiCache dependency. A tokenizer directory must contain readable tokenizer files;
if using a Hugging Face snapshot, resolve its relative blob symlinks when preparing
that standalone directory.

CPU checks exercise actual grammar construction, omitted/true/false, non-Qwen
behavior, source-request preservation, namespaces, Responses entrypoint order,
Messages conversion and message-local tools. Existing reasoning/Responses suites
are included (101checks total including imported fixture tests).

The first live probe misclassified Chat `tool_calls:null` as a client iteration
error. Its partial results are not counted as a passing gate. After correcting the
probe, explicit-strict and omitted-strict Chat/Responses matrices each passed16/16.
The initial portable runner also selected the predecessor Responses suite; the
corrected runner uses the already-merged stream-stability adapter without dropping
tests. Its complete rerun passed101checks.

## Deployed integration result — September16

The two-file patch was deployed on the QAD TP2/MTP3 runtime with existing cache
and acceleration settings retained. All28omitted-strict checks passed:16Chat/Responses
stream/plain × none/medium × tool/plain-answer cases,8equivalent Messages cases,
and a200-function Responses namespace in both stream modes followed by two
successful tool-result replay turns. No grammar-construction error, OOM or stream
exception was observed in that validation window. This is bounded compatibility
validation, not proof that every client schema is supported or a throughput study.

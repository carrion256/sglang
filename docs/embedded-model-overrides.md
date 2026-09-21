# Embedded model-config overrides — 20260921-v1

Runtime image `kanadaj/sglang-qwen38fn-sm120-turbo:embedded-overrides-20260921-v1`
(manifest list `sha256:f84a9ffac80f339696c79efe4739cf693500cfeedbc3cbd898d454a4be6f108b`),
built from `Dockerfile.embedded-model-overrides` on top of
`interleave-shared-ple-20260918-35c83ff` (`@sha256:abe360049f6c…`).

## What it changes

Patch `patches/0034-embedded-model-overrides.patch` adds
`SGLANG_EMBEDDED_MODEL_OVERRIDES` (path to a JSON file). At ServerArgs
resolution time the file is deep-merged **on top of** whatever
`--json-model-override-args` carried (file keys win; CLI-only keys are kept).
Dicts merge recursively; lists/scalars replace. Semantics:

- Env unset/empty → no-op, behavior identical to before.
- Env set to a nonexistent path → warn and skip (lets a shared image serve
  checkpoints that don't need overrides).
- Malformed JSON → **fail loud** at launch (a broken image must never silently
  serve with checkpoint-native defaults).

The image bakes `deploy/embedded-model-overrides.json` at
`/opt/qwen-runtime/model_overrides.json` and sets the ENV to it, so GPUStack
serving parameters no longer need the 2.8 KB YaRN/hybrid-MTP blob. The baked
file is byte-identical in content to the live production
`--json-model-override-args` (canonical-sha `28c3cd8c…`/merged result verified
against model 31 GET): 48-entry `layer_types`, YaRN factor 4.0,
max_position_embeddings 1048576.

## Verification

- Records + full-tree chain: `python3 scripts/verify_embedded_model_overrides.py
  --tree <extracted /sgl-workspace/sglang> --apply` →
  `clean_patch_apply: true, patch_chain_verified: true, source_files: 4394,
  full_tree_verified: true`. Runs in-image at build time as the final gate.
- Behavioral smoke (in-image): helper deep-merge correctness; method-level merge
  preserves CLI-only keys; opt-out untouched; garbage file raises.
- Deployment proof: trial container log shows
  `Merged embedded model overrides from /opt/qwen-runtime/model_overrides.json`
  and `server_args=` reports the fully merged `json_model_override_args`.

## GPUStack

Backend (id 2) version `qwen-embed-overrides-20260921-v1-custom` (copy of
`qwen-interleave-mamba124-20260918-v1-custom` + the ENV, image pinned to the
manifest digest). Trial model 48 `qwen38-embed-overrides-trial-20260921`,
GPUs 0/1, same 41 serving parameters minus the override blob.
Registration readback: `provenance` payloads under
`~/qwen-embed-override-build/` (`backend-after.json`, `create-response.json`).

## Rollback

`--json-model-override-args` keeps working unchanged; setting
`SGLANG_EMBEDDED_MODEL_OVERRIDES=""` in model env restores old behavior even
with the baked file present.

## Related bug context

This image is also the first production-qualified build carrying PR#13's
`is_inside_tool_call` guards (from `35c83ff`), which the running
`hicache-20260915-v2` image lacks — the confirmed cause of prose containing
detector tag markers being swallowed into phantom tool calls (unit test:
`~/sglang-tuning/stopbug/detector_unit.py`, 58 chars lost + 4 phantom calls on
the old image, 0 on the new one).

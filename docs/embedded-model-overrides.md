# Embedded model-config overrides — 20260921-v3

Runtime image `kanadaj/sglang-qwen38fn-sm120-turbo:embedded-overrides-20260921-v3`,
built from `Dockerfile.embedded-model-overrides` on top of
`interleave-shared-ple-20260918-35c83ff` (`@sha256:abe360049f6c…`).

## Design

Patch `patches/0034-embedded-model-overrides.patch` adds
`SGLANG_EMBEDDED_MODEL_OVERRIDES` (path to a JSON file). At ServerArgs
resolution time the CLI `--json-model-override-args` is deep-merged **on top
of** the baked file — **CLI keys win**, so the image provides the bulky
checkpoint-static baseline while every per-deployment knob stays an ordinary
argument:

- The baked file is the checkpoint's own `text_config` **verbatim** (63 keys —
  `layer_types` pattern, indexer, mtp, ple, vocab) with the checkpoint-native
  rope (`rope_type: default`, `max_position_embeddings: 262144`). It carries no
  bespoke values.
- Why the whole file is needed: HF config updates replace `text_config`
  wholesale, so overriding one nested key (e.g. rope) forces re-passing every
  field or they fall back to class defaults. Baking the full static list is
  the right home for that; the operator's arg then only carries what varies.
- YaRN stays configurable: pass
  `--json-model-override-args '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144, ...}},"max_position_embeddings":1048576}'`
  (~250 bytes) instead of the old 2.8 KB blob. Our fleet sets factor 4.0
  (1M window); **consumers of the image get native 262144 by default**.

Semantics: env unset → no-op (CLI behaves exactly as stock). Env set to a
nonexistent path → warn and skip. Malformed baked file **or** malformed CLI
JSON → **fail loud** at launch.

## Verification

- Records + full-tree chain: `python3 scripts/verify_embedded_model_overrides.py
  --tree <extracted /sgl-workspace/sglang> --apply` →
  `clean_patch_apply: true, patch_chain_verified: true, source_files: 4394,
  full_tree_verified: true`; runs in-image at build time as the final gate.
- In-image behavioral smoke (5 cases): native default unchanged; small CLI
  YaRN block wins with all 63 baked keys retained (vocab, ple_layer_ids,
  layer_types, mtp verified present); partial rope dict merges recursively
  (other rope keys survive); no-env leaves CLI untouched; malformed CLI JSON
  raises.

## GPUStack deployment

Backend (id 2) version `qwen-embed-overrides-20260921-v3-custom` — copy of the
v2 record with the v3 image digest. Model 48 carries the 250-byte YaRN
`--json-model-override-args` in `backend_parameters` (factor 4.0); everything
else static comes from the baked file. v1/v2 remain registered and are the
rollback path (template PUT + instance DELETE, ~5 min per replica).

## Naming history

Earlier iterations baked the YaRN-1M config directly (v1/v2, file-wins
precedence). v3 bakes the **native** config and flips precedence to
CLI-wins, which is what makes the rope factor a per-deployment argument
again. The intermediate `0046-embedded-yarn-factor-override` env-knob patch
was withdrawn in favor of this design; 0040–0045 stay reserved for PR #19's
hicache series renumber.

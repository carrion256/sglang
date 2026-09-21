# Qwen Flash-Next multimodal aliases — CPU candidate only

The release checkpoint reports `hf_config.model_type=qwen3_8_flash_next`, while
the shared Qwen VL processor's equivalent behavior is registered under the
development name `qwen4_exp`. The missing release alias disables four existing
paths: concurrent preprocessing policy, preprocessed video metadata, timestamp
token construction, and the image-only mRoPE fast path.

This profile adds the release model type to those four existing allowlists. It
does not add a new processor, change sampling, reorder media, modify model
weights, or affect text-only `qwen3_8_flash_next_text` checkpoints. Other model
types keep their existing behavior. The functional patch remains the reviewed
four-line allowlist change.

## Composition

This cumulative profile is based on main
`460545bf81f1ed24205232d9371e8eb1a02f3e46` and applies, in order:

1. `0015-qwen-flash-next-effort-alias.patch` (`minimal` → `low`, and the existing
   high aliases),
2. `0016-responses-namespace-custom-boundary.patch`,
3. `0017-responses-phase-order.patch`, and
4. `0018-qwen-flash-next-multimodal-alias.patch`.

The first three patches are the current Responses phase/order predecessor
contract documented in [responses-compat.md](responses-compat.md).
`Dockerfile.qwen-multimodal-alias` overlays the resulting six runtime files:
`serving_chat.py`, `protocol.py`, `serving_responses.py`, `responses_compat.py`,
`qwen3_coder_detector.py`, and `qwen_vl.py`. Its build step checks the complete
4,392-file source inventory and hashes, then compiles all six overlay files, so
a successful build cannot silently contain the old Responses or effort source.
All serving arguments and model paths remain external.

The cumulative inventory is
`provenance/qwen-multimodal-alias-runtime-files.json`; its identity and the
predecessor inventory, patch, and changed-file hashes are pinned in
`provenance/qwen-multimodal-alias.json`.

## CPU validation

Use a local directory containing the release checkpoint's `config.json`. The
runner resolves and mounts that file directly, so Hugging Face snapshot symlinks
remain valid:

```bash
QWEN_MODEL_PATH=/absolute/release/config-directory \
  bash scripts/test_qwen_multimodal_alias.sh
python3 scripts/verify_qwen_multimodal_alias.py
```

The runtime test compares `qwen3_8_flash_next` with the already-registered
`qwen4_exp` behavior. Its four focused tests cover worker policy, video metadata
and frame-sampling flags, timestamp token positions and embedding slices, and
image-only mRoPE positions. The runner uses a read-only, network-disabled,
GPU-disabled container.

For a complete source reconstruction, export `python/sglang` from the exact base
image into `TREE`, then run:

```bash
python3 scripts/verify_qwen_multimodal_alias.py --tree TREE --from-image
python3 scripts/verify_qwen_multimodal_alias.py --tree TREE
```

`--from-image` verifies the exact image source, applies 0015, 0016, 0017, and
0018 in that order, and checks all resulting paths and hashes. `--apply` instead
accepts a fully verified Responses phase/order predecessor tree and applies only
0018.

## Limits

These are source reconstruction and CPU structural tests. They are not a GPU
qualification and do not establish image or video semantic accuracy. Historical
live testing accepted image inputs but still misordered events in short
four-frame chronology cases. That video-ordering issue remains open; this patch
only restores the processor behavior already used by the equivalent development
model type. No image is published or deployed by this profile update.

# Qwen Flash-Next multimodal aliases — CPU candidate only

The release checkpoint reports `hf_config.model_type=qwen3_8_flash_next`, while
the shared Qwen VL processor's equivalent behavior is registered under the
development name `qwen4_exp`. The missing release alias disables four existing
paths: concurrent preprocessing policy, preprocessed video metadata, timestamp
token construction, and the image-only mRoPE fast path.

This profile adds the release model type to those four existing allowlists. It
does not add a new processor, change sampling, reorder media, modify model
weights, or affect text-only `qwen3_8_flash_next_text` checkpoints. Other model
types keep their existing behavior.

## Composition

Patch `0018-qwen-flash-next-multimodal-alias.patch` applies after the existing
Responses compatibility profile. `Dockerfile.qwen-multimodal-alias` includes
the five existing API overlay files and the resulting Qwen VL processor file.
All serving arguments and model paths remain external.

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
`qwen4_exp` behavior. It covers worker policy, video metadata and frame-sampling
flags, timestamp token positions and embedding slices, and image-only mRoPE
positions. The runner uses a read-only, network-disabled, GPU-disabled container.

For a complete source reconstruction, export `python/sglang` from the exact base
image into `TREE`, then run:

```bash
python3 scripts/verify_qwen_multimodal_alias.py --tree TREE --from-image
python3 scripts/verify_qwen_multimodal_alias.py --tree TREE
```

## Limits

CPU equality with the registered processor path does not establish semantic
video accuracy. Historical live testing accepted image inputs but still
misordered events in short four-frame chronology cases. That video-ordering
issue remains open; this patch only restores the processor behavior already
used by the equivalent development model type.

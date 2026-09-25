# RVN W4A16 text profile — 0047-0051

Opt-in deployment profile for serving the RVN Qwen4-Exp **W4A16_NVFP4**
text checkpoint: uniform W4A16 dispatch plus the manifest-driven
packed-NVFP4 PLE host tables. It is not part of `patches/series` or
`patches/series.production`; nothing outside this profile's own files
references these patches.

The PLE storage format this profile consumes is frozen in
`docs/rvn-ple-storage-schema.md` (format_version 1).

## Apply order (`patches/series.rvn-w4a16`)

| # | Patch | Adds |
| - | ----- | ---- |
| 1 | `0047-rvn-text-config.patch` | RVN text-only detection/config normalization + weight-name map (`models/qwen4_exp_text_adapter.py`), `qwen4_exp` load mixin, `qwen4_exp_text` config registry entry |
| 2 | `0048-rvn-w4a16-dispatch.patch` | Routes checkpoints with no `quantized_layers` map through uniform ModelOpt FP4 (`W4A16_NVFP4`) instead of failing mixed-precision validation |
| 3 | `0049-rvn-ple-packed-loader.patch` | `models/rvn_ple_storage.py`: manifest-first packed-NVFP4 PLE host tables (`ple_storage.json`) + `weight_utils` loader hook |
| 4 | `0050-rvn-ple-hooksite.patch` | Wires the packed-PLE manifest call site into the RVN text load path (multimodal path stays byte-identical) |
| 5 | `0051-rvn-ple-offload-eligibility.patch` | Extends `--ple-offload-embedding` host/pinned eligibility to `Qwen4ExpForCausalLM` so the text arch can build with the PLE table in pinned host RAM |
| 6 | `0052-rvn-marlin-moe-release.patch` | Frees loader-format MoE storage during the marlin repack (dead swizzle placeholders, per-expert repack buffer, originals dropped as replacements bind) so load peak fits one 96 GB card |
| 7 | `0053-rvn-marlin-skip-blockscale-swizzle.patch` | Never allocates the dead `w*_blockscale_swizzled` placeholders when the backend resolves to Marlin — the 7 GiB is freed at construction, not per-layer |
| 8 | `0054-rvn-ple-recon-mode-gate.patch` | Makes the manifest's `encoding.reconstruction` binding: serving a `bf16_direct` manifest with the legacy FP8 round-trip kernel enabled raises instead of silently degrading PLE numerics |
| 9 | `0055-rvn-marlin-repack-cycle-collect.patch` | Forces a gc pass after each Marlin repack so the superseded loader-format Parameters (held alive by a reference cycle; gen-2 never fires during load) are reclaimed per-layer — load ends at 73.38 GB instead of OOMing at 93.65 |
| 10 | `0056-rvn-ple-encoder-version-gate.patch` | Loader refuses any `encoder_version` other than the frozen `rvn-ple-nvfp4-r1`, matching the offline verifier's gate |
| 11 | `0057-rvn-nextn-draft-gate.patch` | Refuses `--speculative-algorithm NEXTN` on the RVN text model at argument-resolution time: the text contract is MTP-free (`mtp_num_hidden_layers=0`, loader rejects `model.mtp.*`), so the default draft would rebuild the target a second time and OOM — a clear error instead |

## Base image

Build base is the **locally deployed** image (Docker Id
`sha256:91cee840799be19916e1ba17ed10a517923f4fc70d54f5abd0247f700d01d77a`):

```
localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284
```

**This base is local-only** — it was never pushed to a registry, so
`docker build -f Dockerfile.rvn-w4a16 .` works only on a host that
already has it (`docker save`/`load` to move it). `provenance/rvn-w4a16.json`
records both the tag (`base_image`) and the Id (`base_image_id`); the
runtime-file inventory `provenance/rvn-w4a16-runtime-files.json` pins the
base's `python/sglang` tree byte-for-byte.

## Build and run

```bash
docker build -f Dockerfile.rvn-w4a16 -t rvn-w4a16:<tag> .
```

The build's last step is the provenance gate — it verifies the base
inventory, applies the ten patches in series order with `git apply`, and
re-hashes every source file against the manifest:

```
python3 -B /opt/rvn-w4a16/scripts/verify_rvn_w4a16.py --tree /sgl-workspace/sglang --apply
```

## Serving the RVN candidate (verified recipe)

The candidate checkpoint is produced by `tools/rvn_ple/convert.py` (schema
v1, self-contained directory incl. tokenizer/chat-template and a
`config.json` stamped `ple_embedding_dtype: "nvfp4"`). Single-GPU launch,
validated against a 96 GB card while a second worker holds the other GPU:

```bash
docker run --rm --name rvn-ple-nvfp4 \
  --device nvidia.com/gpu=1 --ipc host --network host \
  --shm-size 32g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e SGLANG_EMBEDDED_MODEL_OVERRIDES= \
  -e SGLANG_SM120_ONLINE_MXFP8=false \
  -e SGLANG_PLE_PACKED_NVFP4=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /models:/models:ro -v /var/cache/sglang/rvn-ple-nvfp4-0:/root/.cache \
  -w /sgl-workspace/sglang --entrypoint python3 rvn-w4a16:sim \
  -m sglang.launch_server \
  --model-path /models/rvn-qwen38-ple-nvfp4 \
  --chat-template /models/rvn-qwen38-ple-nvfp4/chat_template.jinja \
  --served-model-name rvn-ple-nvfp4 \
  --host 0.0.0.0 --port 8111 --tp-size 1 \
  --quantization modelopt_mixed --moe-runner-backend marlin \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 \
  --mem-fraction-static 0.90 --page-size 64 --chunked-prefill-size 4096 \
  --max-running-requests 4 \
  --cuda-graph-max-bs-decode=8 --disable-prefill-cuda-graph \
  --reasoning-parser auto --tool-call-parser auto \
  --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer \
  --max-mamba-cache-size 64 --mamba-radix-cache-strategy extra_buffer \
  --mamba-track-interval 128 --mamba-ssm-dtype bfloat16 \
  --gdn-mtp-cache-mode none \
  --ple-offload-embedding \
  --model-loader-extra-config '{"enable_multithread_load":false,"num_threads":2}'
```

Landmines learned the hard way:

- **GPU pinning**: use the CDI device flag `--device nvidia.com/gpu=1`;
  `--gpus '"device=N"'` is broken in this environment and
  `--device nvidia.com/gpu=all` + `CUDA_VISIBLE_DEVICES` does **not**
  isolate. Inside the container the visible GPU is always `cuda:0`.
- **`--ple-offload-embedding` is mandatory**: without it the
  320001536×160 table tries to build as a 95 GiB CUDA
  `VocabParallelEmbedding` and OOMs at construction. 0051 teaches the
  gate about the text arch; `SGLANG_PLE_PACKED_NVFP4=1` selects the
  manifest-backed packed host table (26.8 GiB pinned host RAM).
- **Serial weight loading** (`--model-loader-extra-config
  '{"enable_multithread_load":false,"num_threads":2}'`) plus
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduce load-time peak;
  0052 releases the loader-format MoE storage during the Marlin repack
  (dead swizzle placeholders + per-expert repack buffer + originals as they
  are replaced) so the load peak fits one 96 GB card.
- The `memlock` ulimit is required for the pinned host PLE table.
- **Decode CUDA graphs are a ~9x win here**: graph-off measured 11.7
  tok/s, `--cuda-graph-max-bs-decode=8 --disable-prefill-cuda-graph`
  measured 103 tok/s (capture bs=[1,2,4], 0.12 GB, 4.2 s). Radix cache
  stays enabled and is compatible.

## Verification

- Records + full-tree chain (host): `python3 scripts/verify_rvn_w4a16.py
  --tree <extracted /sgl-workspace/sglang> --apply` →
  `clean_patch_apply: true, patch_chain_verified: true, full_tree_verified: true`.
- Packaging (host): `python3 -m unittest tests/test_rvn_w4a16_packaging.py -v`
- In-image battery: `scripts/test_rvn_w4a16.sh` (`RVN_W4A16_IMAGE` selects
  the image; defaults to the local-only base above). The battery rebuilds
  the patched tree itself — it needs the **unpatched** base tree, so
  pointing it at a built `rvn-w4a16` image fails fast by design.

## Rollback

Nothing here mutates the base tag or any shared series, so rollback is
just running the previous image again —
`localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284` (or the last deployed
`rvn-w4a16:<tag>`).

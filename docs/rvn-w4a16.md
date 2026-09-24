# RVN W4A16 text profile — 0047-0050

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
inventory, applies the four patches in series order with `git apply`, and
re-hashes every source file against the manifest:

```
python3 -B /opt/rvn-w4a16/scripts/verify_rvn_w4a16.py --tree /sgl-workspace/sglang --apply
```

Serving is unchanged from the base image (same `sglang.launch_server`
entrypoint); launch flags live in `docs/quickstart-docker.md` and
`docs/production-command.sh`.

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

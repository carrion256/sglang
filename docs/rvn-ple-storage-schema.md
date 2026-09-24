# RVN PLE packed-NVFP4 storage contract (`ple_storage.json`) v1

Frozen interface for WP2. Converter (`tools/rvn_ple/convert.py`), verifier
(`tools/rvn_ple/verify.py`) and the loader (`python/sglang/srt/models/
rvn_ple_storage.py` + reuse of `packed_ple.py::PackedPLEStorage` /
`gather_packed_kernel`) MUST all conform to this document. Any change here
bumps `format_version`.

## 1. Numerical encoding (fixed)

- Weight codes: E2M1, two per `uint8`, **low nibble first** (value 2i in
  `byte[i] & 0x0F`, value 2i+1 in `byte[i] >> 4`). Nibble bits: sign `0x8`,
  magnitude `0..7 -> [0, .5, 1, 1.5, 2, 3, 4, 6]`.
- Block scales: one FP8 E4M3 (`float8_e4m3fn`, finite, non-negative on read)
  per `group_size = 16` consecutive columns (last axis). Row-major,
  **not** kernel-swizzled.
- Global scale: single positive float32 multiplier `g`, one per logical table
  (shared across all partitions), frozen before encoding and recorded as raw
  IEEE-754 bits: `g = amax / (6 * 448)`; all-zero table -> defined neutral
  `g = 1.0` with `amax == 0` recorded.
- Reconstruction (RVN default, `reconstruction == "bf16_direct"`):
  `out = bfloat16( E2M1(code) * e4m3_to_f32(scale) * g )` in float32, rounded
  to BF16 once. `"fp8_roundtrip"` (legacy LIL `SGLANG_PLE_PACKED_FP8_REFERENCE=1`)
  is out of scope for RVN v1 but must remain selectable in the loader.

## 2. Manifest `ple_storage.json` (checkpoint root)

```json
{
  "format_version": 1,
  "encoder_version": "rvn-ple-nvfp4-r1",
  "required_loader_feature": "ple-packed-nvfp4-v1",
  "source": {
    "repo": "0bserverx/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-NVFP4",
    "revision": "<git revision>",
    "source_table_sha256": "<sha256 over canonical tensor-payload concat, §4>",
    "source_dtype": "bfloat16",
    "amax": 0.0
  },
  "table": {
    "logical_rows": 0,
    "cols": 0,
    "partitioning": [
      {"part": 0, "source_shard": "model-00005-of-00098.safetensors",
       "source_tensor": "<exact tensor name>",
       "row_offset": 0, "rows": 0}
    ]
  },
  "encoding": {
    "weight_dtype": "e2m1-packed-u8-low-first",
    "scale_dtype": "float8_e4m3fn",
    "group_size": 16,
    "scale_layout": "row-major",
    "global_scale_bits": 0,
    "reconstruction": "bf16_direct"
  },
  "parts": [
    {"file": "rvn_ple_parts/part-00000.safetensors",
     "weight_tensor": "rvn_ple.packed.w0",
     "scale_tensor": "rvn_ple.packed.s0",
     "row_offset": 0, "rows": 0,
     "sha256_weights": "<hex>", "sha256_scales": "<hex>",
     "first_source_tensor": "<name>", "last_source_tensor": "<name>"}
  ],
  "retained_rewrites": {
    "<output shard containing retained non-PLE tensors>": ["<tensor>", "..."]
  }
}
```

Hard rules:
- `partitioning` is ordered by **numeric source ordering**, never lexicographic;
  it is a complete, non-overlapping cover of `0..logical_rows-1`.
- `parts` cover exactly the same rows as `partitioning`; every `rows` > 0;
  `sum(rows) == logical_rows`; sha256 values are over the raw little-endian
  tensor payload bytes of each stored tensor (§4).
- `global_scale_bits` is `struct.pack("<f", g)` reinterpreted as uint32;
  loaders reject non-finite or non-positive `g`.
- Generic model index (`model.safetensors.index.json`) references ONLY
  tensors that exist in candidate shards; a loader must fail clearly on a
  partial/unfinished candidate (no `ple_storage.json` + no packed tensors;
  manifest present but a part missing = hard error, never BF16 fallback).

## 3. Resume state (`.rvn_convert_state.json`, tool-owned, not shipped)

```json
{
  "key": {"source_revision": "...", "encoder_version": "...",
          "global_scale_bits": 0, "group_size": 16, "reconstruction": "bf16_direct"},
  "phase": "scan|encode|write|assemble",
  "completed_parts": [0, 1],
  "scan": {"amax": 0.0, "nonfinite_count": 0, "per_part_max": {"0": 0.0}}
}
```
Resume is legal only when the whole `key` matches an existing state file;
otherwise conversion restarts from `scan`. Atomic publish per part:
write `*.tmp` -> fsync -> rename -> append state.

## 4. Digest definitions

- Tensor payload digest: `sha256` over the raw contiguous little-endian
  bytes of the tensor (shape/dtype fixed by manifest; padding excluded).
- `source_table_sha256`: `sha256` over
  `concat(payload(part_i))` for parts in ascending `part` order.
- Non-PLE identity (WP2 verify): per logical tensor name, digest of
  `(dtype, shape tuple, payload sha256)`; file boundaries may change,
  tensor identity may not.

## 5. Memory contract (converter)

- Source slice budget <= 256 MiB per chunk; hard transient ceilings:
  4 GiB VRAM (converter runs serial on one GPU or CPU; never without an
  explicit reservation), 4 GiB anonymous RSS outside file cache.
- No full-BF16 PLE materialization anywhere; no Python list holding all
  encoded parts; bounded output parts written incrementally.
- Never mutate the source checkpoint; output tree is separate; hardlinks to
  unchanged shards are read-only (never rewritten in place).

## 6. Loader feature string

`required_loader_feature == "ple-packed-nvfp4-v1"`; loaders lacking it
raise, they do not fall back. The RVN exclusion of `*ple*` from *linear
quantization* must not block this table loader (manifest recognized first).

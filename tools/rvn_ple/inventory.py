#!/usr/bin/env python3
"""RVN Qwen3.8-Flash-Next checkpoint inventory tool.

Header-only inspection of a safetensors checkpoint (``model.safetensors.index.json``
plus its shards): every tensor is read from its shard's safetensors header, never
from its payload, so no tensor data is ever loaded.

Classification rules (first match wins; precedence is documented so it is
auditable). Name patterns are grounded in the qwen4_exp loader
(``runtime/python/sglang/srt/models/qwen4_exp.py``, ``load_weights`` /
``load_qwen4_exp_ple_shard``), ``model_loader/weight_utils.py`` and the frozen
RVN storage contract ``docs/rvn-ple-storage-schema.md``:

1. ``ple_embedding_table`` — PLE n-gram embedding tables:
   ``<mod_prefix>.ngram_embedding.weight`` (unpacked), the packed source pairs
   ``\\.ngram_embedding\\.shard_(\\d+)\\.(weight|weight_scale)$`` (qwen4_exp
   loader regex, canonical decimal partition ids -- the loader rejects
   leading zeros, so ordering is numeric: ``shard_2`` < ``shard_10``), the
   per-table scalars ``\\.ngram_embedding\\.weight_scale`` /
   ``.weight_scale_2``, and the candidate tensors ``rvn_ple.packed.w<i>`` /
   ``rvn_ple.packed.s<i>`` (storage contract §2).
2. ``expert_scale`` / ``expert_weight`` — routed experts, matched on the
   ``experts`` *segment* (``weight_utils._ROUTED_EXPERT_KEY_RE`` discipline,
   which excludes ``shared_expert`` because it requires a digit) plus the fused
   4D checkpoint form qwen4_exp accepts without a digit segment
   (``...mlp.experts.gate_up_proj`` / ``.down_proj``). A scale key is anything
   containing ``scale``: ``weight_scale``, ``weight_scale_inv`` (mxfp8 fused
   shards store component block scales as rows of ``weight_scale_inv``) and
   ``weight_scale_2``.
3. ``shared_expert`` — the ``shared_expert``/``shared_experts`` module segment.
4. ``ple_other`` — every remaining tensor under a ``ple`` module *segment*
   (never a bare ``*ple*`` glob): PLE projections (``key_proj``/``value_proj``),
   the short convolution (``conv1d``), the grouped norms
   (``norm_key``/``norm_query``/``norm_conv``) and ``ple_embedding`` buffers.
5. ``attention`` — ``self_attn``/``linear_attn``/``attention`` segments.
6. ``routing_norm`` — the MoE router and the norms. qwen4_exp names the router
   ``name.endswith(".gate")`` and it is anchored here as
   ``mlp.gate.(weight|e_score_correction_bias|bias)`` plus ``router``/
   ``gate_bias`` segments, so ``*.gate_proj.weight`` in dense/MLP/expert
   tensors can never fall into it (rules 1-3 already claimed those names).
   Norm tensors are any path segment containing ``norm`` (``input_layernorm``,
   ``post_attention_layernorm``, ``hc_norm``, ...).
7. ``embedding_head`` — ``embed_tokens`` / ``lm_head`` / ``word_embeddings``.
8. ``unmatched`` — everything else, e.g. the HyperConnection mix/combine
   weights (``attn_hyper_connection.input_mix_weight_*``, ``block_inject_weight``).
   Hyper-connection tensors have no contract category, so the count is reported
   honestly instead of inventing a category or stretching ``attention``.

Resource budget follows storage contract §1/§4: for each PLE table tensor the
packed NVFP4 footprint is ``rows*cols/2`` weight bytes (E2M1, two per byte)
plus ``rows*(cols//16)`` FP8-E4M3 block-scale bytes (``group_size = 16``,
row-major, per ``packed_ple.py::PackedPLEStorage``, which requires the last
dimension divisible by 16 -- a table that is not is a hard error here).
Tensors that are already packed (uint8 nibble payload) or are scales
(``scale`` in the name) carry over their existing bytes. Value-level checks
that need the payload -- a finite positive global scale ``g``
(``global_scale_bits``, contract §1/§2) -- belong to the verifier, not to this
header-only tool; the tool itself never emits a non-finite number.

Index/shard consistency failures are hard errors, backed by contract §2's rule
that the generic index references ONLY tensors that exist in the shards: a
missing or unreadable shard, a duplicate tensor (same name in two shards), a
declared tensor absent from its declared shard, an undeclared tensor, and a
malformed header all abort the run.

Output JSON is deterministic (sorted keys, natural-numeric name/shard ordering)
and self-describing (counts, tool version, estimate formula).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

TOOL_NAME = "rvn_ple.inventory"
TOOL_VERSION = "1.0.0"

INDEX_NAME = "model.safetensors.index.json"
SCALE_GROUP_SIZE = 16  # docs/rvn-ple-storage-schema.md §1

CATEGORIES = (
    "expert_weight",
    "expert_scale",
    "ple_embedding_table",
    "ple_other",
    "attention",
    "shared_expert",
    "routing_norm",
    "embedding_head",
    "unmatched",
)

# 1. PLE embedding tables. Partition ids are plain decimal: the qwen4_exp
#    loader requires str(int(id)), so ordering must be numeric (2 < 10).
# Candidate tensors from the storage contract §2 (`rvn_ple.packed.w<i>` /
# `rvn_ple.packed.s<i>`) are already in packed form.
_PLE_CANDIDATE_RE = re.compile(r"^rvn_ple\.packed\.[ws]\d+$")
_PLE_TABLE_RE = re.compile(
    r"(?:^|\.)ngram_embedding\.(?:shard_(\d+)\.)?weight(?:_scale)?(?:_2)?$"
    r"|^rvn_ple\.packed\.[ws]\d+$"
)
# 4. PLE module segment (precise segment match, never a bare *ple* glob).
_PLE_SEG_RE = re.compile(r"(^|\.)ple(\.|$)")
# 2. Routed experts: the `experts` segment (weight_utils requires a digit
#    after it, which is what keeps `shared_expert` out; the fused 4D qwen4_exp
#    form has no digit and is matched by the same segment).
_EXPERT_SEG_RE = re.compile(r"(^|\.)experts(\.|$)")
# 3. Shared expert module segment.
_SHARED_EXPERT_RE = re.compile(r"(^|\.)shared_experts?(\.|$)")
# 5. Attention segments.
_ATTENTION_RE = re.compile(r"\.(self_attn|linear_attn|attention)(\.|$)")
# 6. MoE router (anchored: `.gate_proj` is a different segment and cannot
#    match) and norm-bearing segments.
_ROUTER_RE = re.compile(
    r"(^|\.)mlp\.gate\.(?:weight|bias|e_score_correction_bias)$"
    r"|(^|\.)router(\.|$)|\.gate_bias$"
)
_NORM_SEG_RE = re.compile(r"(^|\.)([^.]*norm[^.]*)(\.|$)")
# 7. Embedding / output head.
_EMBED_HEAD_RE = re.compile(r"(^|\.)(lm_head|embed_tokens|word_embeddings)(\.|$)")

# Small files always hashed, regardless of --hash-files: tokenizer, chat
# template, config, and the RVN storage manifest (contract §2, checkpoint root).
_SMALL_FILE_RE = re.compile(
    r"^(?:config\.json|generation_config\.json|special_tokens_map\.json"
    r"|merges\.txt|vocab\.json|tokenizer[\w.\-]*|ple_storage\.json"
    r"|[\w.\-]*chat[\w.\-]*template[\w.\-]*)$"
)


class InventoryError(Exception):
    """Missing/unreadable checkpoint component or inconsistent checkpoint."""


def _natural_key(name: str):
    """Numeric-aware sort key: never lexicographic for digit runs (2 < 10)."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r"(\d+)", name)
    )


def classify_tensor(name: str):
    """Classify one tensor name into exactly one category.

    Returns ``(category, ple_partition_or_None)``; see the module docstring for
    the documented precedence.
    """
    m = _PLE_TABLE_RE.search(name)
    if m:
        return "ple_embedding_table", int(m.group(1)) if m.group(1) else None
    if _EXPERT_SEG_RE.search(name):
        return ("expert_scale" if "scale" in name else "expert_weight"), None
    if _SHARED_EXPERT_RE.search(name):
        return "shared_expert", None
    if _PLE_SEG_RE.search(name):
        return "ple_other", None
    if _ATTENTION_RE.search(name):
        return "attention", None
    if _ROUTER_RE.search(name) or _NORM_SEG_RE.search(name):
        return "routing_norm", None
    if _EMBED_HEAD_RE.search(name):
        return "embedding_head", None
    return "unmatched", None


def _rows_cols(shape):
    """(rows, cols) of a logical table: rows = product of leading dims."""
    if len(shape) < 2:
        return None
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    return rows, shape[-1]


def _packed_nvfp4_estimate(name, dtype, shape, byte_size):
    """Packed NVFP4 footprint of one PLE table tensor (storage contract §1).

    Unpacked logical table -> ``rows*cols/2`` nibble bytes +
    ``rows*(cols//16)`` FP8-E4M3 block-scale bytes, and the last dimension
    must be divisible by 16 (``PackedPLEStorage`` requires it). Everything
    already in packed form carries its bytes over unchanged: the contract §2
    candidate pair ``rvn_ple.packed.w<i>``/``.s<i>`` (whose scale tensor is
    stored as ``rows x cols/16``, so its last dim is *not* group-divisible)
    and uint8 nibble payloads, plus scale-named tensors
    (``weight_scale``/``weight_scale_inv``/``weight_scale_2``).
    """
    if _PLE_CANDIDATE_RE.match(name):
        return byte_size  # already packed w/s pair (contract §2)
    if "scale" in name:
        return byte_size
    if dtype == "U8":
        return byte_size  # already E2M1 nibble-packed
    rc = _rows_cols(shape)
    if rc is None:
        raise InventoryError(
            f"PLE table tensor {name!r} has no rows/cols (shape {shape})"
        )
    rows, cols = rc
    if cols % SCALE_GROUP_SIZE:
        raise InventoryError(
            f"PLE table tensor {name!r}: last dim {cols} is not divisible by "
            f"group_size {SCALE_GROUP_SIZE}, cannot pack NVFP4"
        )
    return rows * cols // 2 + rows * (cols // SCALE_GROUP_SIZE)


def _finite_float(value):
    try:
        f = float(value)
    except (OverflowError, TypeError, ValueError):
        return "non-finite"
    return f if math.isfinite(f) else "non-finite"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reject_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise InventoryError(f"duplicate tensor in shard header: {key}")
        out[key] = value
    return out


def read_shard_header(path: Path):
    """Parse one safetensors header; return ``{name: entry}``, no tensor data."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) != 8:
                raise InventoryError(f"unreadable shard (truncated header): {path}")
            (header_len,) = struct.unpack("<Q", head)
            raw = f.read(header_len)
            if len(raw) != header_len:
                raise InventoryError(
                    f"unreadable shard (truncated header JSON): {path}"
                )
        meta = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except InventoryError:
        raise
    except (OSError, ValueError) as exc:
        raise InventoryError(f"unreadable shard: {path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise InventoryError(f"unreadable shard (bad header): {path}")

    tensors = {}
    for name, entry in meta.items():
        if name == "__metadata__":
            continue
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("dtype"), str)
            or not isinstance(entry.get("shape"), list)
            or not all(isinstance(dim, int) and dim >= 0 for dim in entry["shape"])
        ):
            raise InventoryError(
                f"unreadable shard (bad tensor entry {name!r}): {path}"
            )
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(off, int) and off >= 0 for off in offsets)
            or offsets[0] > offsets[1]
        ):
            raise InventoryError(
                f"unreadable shard (bad data_offsets for {name!r}): {path}"
            )
        tensors[name] = {
            "dtype": entry["dtype"],
            "shape": list(entry["shape"]),
            "data_offsets": [offsets[0], offsets[1]],
            "byte_size": offsets[1] - offsets[0],
        }
    return tensors


def build_inventory(
    checkpoint_dir, *, hash_files: bool = False, config_name: str = "config.json"
):
    """Build the deterministic inventory document for ``checkpoint_dir``."""
    d = Path(checkpoint_dir)
    index_path = d / INDEX_NAME
    if not index_path.is_file():
        raise InventoryError(f"missing index: {index_path}")
    try:
        index = json.loads(index_path.read_text())
    except (OSError, ValueError) as exc:
        raise InventoryError(f"unreadable index: {index_path}: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()
        )
    ):
        raise InventoryError(f"index has no usable weight_map: {index_path}")

    # Shard order: numeric, never lexicographic (contract §2 partitioning rule).
    shard_files = sorted(set(weight_map.values()), key=_natural_key)
    seen_in = {}  # tensor name -> shards whose header declares it
    shard_headers = {}
    for shard in shard_files:
        path = d / shard
        if not path.is_file():
            raise InventoryError(f"missing shard: {shard}")
        header = read_shard_header(path)
        shard_headers[shard] = header
        for name in header:
            seen_in.setdefault(name, []).append(shard)

    for name, shards in sorted(seen_in.items(), key=lambda kv: _natural_key(kv[0])):
        if len(shards) > 1:
            raise InventoryError(
                f"duplicate tensor: {name!r} present in shards "
                f"{sorted(shards, key=_natural_key)}"
            )
    for name, shard in weight_map.items():
        if name not in shard_headers[shard]:
            raise InventoryError(
                f"missing tensor: {name!r} not found in declared shard {shard}"
            )
    for shard, header in shard_headers.items():
        for name in header:
            if name not in weight_map:
                raise InventoryError(
                    f"undeclared tensor: {name!r} in shard {shard} is not in the index"
                )

    tensors = []
    cat_bytes = {cat: 0 for cat in CATEGORIES}
    cat_count = {cat: 0 for cat in CATEGORIES}
    total_bytes = 0
    ple_bytes = 0
    packed_estimate = 0
    for name in sorted(weight_map, key=_natural_key):
        shard = weight_map[name]
        entry = shard_headers[shard][name]
        category, partition = classify_tensor(name)
        estimate = None
        if category == "ple_embedding_table":
            estimate = _packed_nvfp4_estimate(
                name, entry["dtype"], entry["shape"], entry["byte_size"]
            )
            ple_bytes += entry["byte_size"]
            packed_estimate += estimate
        tensors.append(
            {
                "name": name,
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "byte_size": entry["byte_size"],
                "shard": shard,
                "data_offsets": entry["data_offsets"],
                "category": category,
                "ple_partition": partition,
                "estimated_packed_nvfp4_bytes": estimate,
            }
        )
        cat_count[category] += 1
        cat_bytes[category] += entry["byte_size"]
        total_bytes += entry["byte_size"]
    non_ple_bytes = total_bytes - ple_bytes

    config_path = Path(config_name)
    if not config_path.is_absolute():
        config_path = d / config_path
    if not config_path.is_file():
        raise InventoryError(f"missing config file: {config_path}")
    small_names = sorted(
        {
            p.name
            for p in d.iterdir()
            if p.is_file() and _SMALL_FILE_RE.match(p.name)
        }
        | {
            str(config_path.relative_to(d))
            if config_path.parent == d
            else str(config_path)
        }
    )
    small_files = {}
    for name in small_names:
        path = d / name
        small_files[name] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }

    shard_payloads = {}
    if hash_files:
        for shard in shard_files:
            shard_payloads[shard] = _sha256_file(d / shard)

    return {
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "checkpoint_dir": str(d),
        "index": {
            "name": INDEX_NAME,
            "sha256": _sha256_file(index_path),
            "bytes": index_path.stat().st_size,
        },
        "shard_order": shard_files,
        "counts": {
            "tensors": len(tensors),
            "shards": len(shard_files),
            "by_category": cat_count,
        },
        "categories": {
            cat: {
                "count": cat_count[cat],
                "bytes": cat_bytes[cat],
                "gb_decimal": _finite_float(cat_bytes[cat] / 1e9),
                "gib_binary": _finite_float(cat_bytes[cat] / 2**30),
            }
            for cat in CATEGORIES
        },
        "tensors": tensors,
        "resource_budget": {
            "source_total_bytes": total_bytes,
            "ple_embedding_table_bytes": ple_bytes,
            "estimated_packed_nvfp4_bytes": packed_estimate,
            "non_ple_unchanged_bytes": non_ple_bytes,
            "candidate_total_bytes": non_ple_bytes + packed_estimate,
            "estimate_formula": (
                "per ple_embedding_table tensor: rows*cols/2 nibble bytes + "
                "rows*(cols//16) fp8-e4m3 block-scale bytes (group_size=16, "
                "row-major); already-packed uint8 payloads and scale-named "
                "tensors carry their byte_size over; cols % 16 != 0 is an error"
            ),
        },
        "hashes": {
            "small_files": small_files,
            "shard_payloads_hashed": bool(hash_files),
            "shard_payloads": shard_payloads,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Header-only safetensors checkpoint inventory (no tensor loads).",
    )
    parser.add_argument("--dir", required=True, help="checkpoint directory")
    parser.add_argument(
        "--out", default="rvn-inventory.json", help="output JSON path"
    )
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="also sha256 shard payloads (off by default)",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="config file to hash (relative to --dir unless absolute)",
    )
    args = parser.parse_args(argv)

    try:
        data = build_inventory(
            args.dir, hash_files=args.hash_files, config_name=args.config
        )
    except InventoryError as exc:
        print(f"{TOOL_NAME}: error: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    budget = data["resource_budget"]
    print(
        f"{out}: {data['counts']['tensors']} tensors / {data['counts']['shards']} shards; "
        f"ple_table={budget['ple_embedding_table_bytes']}B "
        f"packed_est={budget['estimated_packed_nvfp4_bytes']}B "
        f"candidate_total={budget['candidate_total_bytes']}B"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
